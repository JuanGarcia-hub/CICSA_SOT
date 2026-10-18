"""Servicio de aplicación para revisión humana y cola de revisión.

Coordinador del caso de uso (HU-01 a HU-04).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional
from uuid import uuid4

from sot_matcher.application.ports import (
    FeedbackRepository,
    FeatureExtractor,
    CandidateRetriever,
    FeedbackFilters,
)
from sot_matcher.domain.feedback import (
    FeedbackEvent,
    FeedbackDecision,
    CandidatePair,
    MatchingPrediction,
)


@dataclass(frozen=True)
class SubmitDecisionCommand:
    sot_id: str
    log_id: str
    decision: FeedbackDecision
    run_id: str | None = None
    reason_code: str | None = None
    comment: str | None = None
    reviewer_id: str | None = None
    supersedes_feedback_id: str | None = None
    feature_schema_version: str | None = None


@dataclass(frozen=True)
class ReviewQueueItem:
    sot_id: str
    candidate_count: int
    has_exact_match: bool
    has_semantic_candidates: bool
    date_match: bool
    priority: int  # mayor = más urgente
    estado_actual: str  # "Exacto", "Semántico", "Pendiente"


@dataclass(frozen=True)
class CandidateReview:
    sot_id: str
    sot_text: str
    sot_date_hint: Optional[date]
    candidates: list[CandidatePair]
    existing_feedback: list[FeedbackEvent]


class FeedbackService:
    def __init__(
        self,
        repository: FeedbackRepository,
        feature_extractor: FeatureExtractor,
        candidate_retriever: CandidateRetriever,
    ):
        self._repo = repository
        self._extractor = feature_extractor
        self._retriever = candidate_retriever

    def submit_decision(self, command: SubmitDecisionCommand) -> FeedbackEvent:
        """Registra una decisión (HU-01, HU-02, HU-03)."""
        # Validar decisión
        if command.decision not in FeedbackDecision:
            raise ValueError(f"Decisión inválida: {command.decision}")

        # Si corrige decisión previa, crear evento que referencia al anterior
        supersedes = command.supersedes_feedback_id
        if supersedes is None:
            # Buscar decisión previa para mismo par
            existing = self._repo.get_feedback_for_pair(command.sot_id, command.log_id)
            if existing:
                latest = existing[-1]
                if latest.decision != command.decision:
                    supersedes = latest.feedback_id

        event = FeedbackEvent(
            feedback_id=str(uuid4()),
            run_id=command.run_id,
            sot_id=command.sot_id,
            log_id=command.log_id,
            decision=command.decision,
            reason_code=command.reason_code,
            comment=command.comment,
            reviewer_id=command.reviewer_id,
            supersedes_feedback_id=supersedes,
            feature_schema_version=command.feature_schema_version or "1.0",
        )
        return self._repo.save_feedback(event)

    def get_review_queue(self, results_df, prepared_sots, prepared_logs) -> list[ReviewQueueItem]:
        """Cola de SOTs pendientes de revisión (HU-04).

        Prioriza:
        1. SOTs marcados como Pendiente
        2. SOTs con match semántico (baja confianza)
        3. SOTs con match exacto pero múltiples candidatos (ambigüedad)
        """
        items = []

        # Convertir resultados a dict para lookup rápido
        results_by_sot = {}
        for _, row in results_df.iterrows():
            results_by_sot[str(row["sot"])] = row

        for _, sot_row in prepared_sots.iterrows():
            sot_id = str(sot_row["sot"])
            result = results_by_sot.get(sot_id)

            if result is None:
                continue

            estado = str(result.get("estado", "Pendiente"))

            # Obtener candidatos del mismo día
            sot_date = sot_row.get("date_hint")
            same_day_logs = prepared_logs
            if sot_date is not None:
                same_day_logs = prepared_logs[prepared_logs["date"] == sot_date]

            candidate_count = len(same_day_logs)
            has_exact = estado == "Exacto"
            has_semantic = estado == "Semántico"

            # Calcular prioridad
            priority = 0
            if estado == "Pendiente":
                priority = 100
            elif has_semantic:
                conf = float(result.get("confianza", 0) or 0)
                priority = 50 + int((1 - conf) * 50)  # menor confianza = mayor prioridad
            elif has_exact:
                # Verificar si hubo ambigüedad (múltiples exactos mismo día)
                priority = 10

            items.append(ReviewQueueItem(
                sot_id=sot_id,
                candidate_count=candidate_count,
                has_exact_match=has_exact,
                has_semantic_candidates=has_semantic,
                date_match=sot_date is not None,
                priority=priority,
                estado_actual=estado,
            ))

        # Ordenar por prioridad descendente
        items.sort(key=lambda x: x.priority, reverse=True)
        return items

    def get_candidate_details(
        self,
        sot_id: str,
        sot_record: dict[str, Any],
        log_records: list[dict[str, Any]],
    ) -> CandidateReview:
        """Detalle de candidatos para una SOT específica (HU-04)."""
        # Recuperar candidatos usando CandidateRetriever
        candidates = self._retriever.retrieve(sot_record, log_records, limit=10)

        # Enriquecer con características
        for cand in candidates:
            log_rec = next((lr for lr in log_records if str(lr.get("record_id", "")) == cand.log_id), None)
            if log_rec:
                features = self._extractor.extract(sot_record, log_rec)
                # CandidatePair es frozen, crear uno nuevo
                cand = CandidatePair(
                    sot_id=cand.sot_id,
                    log_id=cand.log_id,
                    run_id=cand.run_id,
                    features=features,
                    feature_schema_version=self._extractor.schema_version,
                    semantic_score=cand.semantic_score,
                    classifier_probability=cand.classifier_probability,
                )

        # Feedback existente para esta SOT
        existing = []
        for log_rec in log_records:
            log_id = str(log_rec.get("record_id", ""))
            fb = self._repo.get_feedback_for_pair(sot_id, log_id)
            existing.extend(fb)

        return CandidateReview(
            sot_id=sot_id,
            sot_text=str(sot_record.get("source_text", "")),
            sot_date_hint=sot_record.get("date_hint"),
            candidates=candidates,
            existing_feedback=existing,
        )