"""Active Learning Loop para SOT Matcher.

Exporta casos PENDIENTE con top-K candidatos para anotación humana,
importa anotaciones y soporta re-entrenamiento incremental.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from sot_matcher.application.ports import (
    FeedbackRepository,
    FeatureExtractor,
    CandidateRetriever,
    MatchingClassifier,
    ModelRegistry,
    FeedbackFilters,
)
from sot_matcher.domain.feedback import (
    FeedbackDecision,
    FeedbackEvent,
    CandidatePair,
    LabeledPair,
)
from sot_matcher.application.training_pipeline import (
    TrainingPipeline,
    ShadowEvaluationResult,
)


class ActiveLearningStrategy(str, Enum):
    """Estrategias de selección de muestras para anotación."""
    UNCERTAINTY = "uncertainty"           # Muestras con probabilidad ~0.5
    MARGIN = "margin"                     # Menor margen entre top-2
    ENTROPY = "entropy"                   # Mayor entropía
    DIVERSITY = "diversity"               # Máxima diversidad (k-means++)
    HYBRID = "hybrid"                     # Combinación uncertainty + diversity
    LEAST_CONFIDENT = "least_confident"   # Menor confianza máxima
    DISAGREEMENT = "disagreement"         # Desacuerdo ensemble/classifier vs engine


@dataclass(frozen=True)
class ActiveLearningConfig:
    """Configuración del loop de active learning."""
    strategy: ActiveLearningStrategy = ActiveLearningStrategy.HYBRID
    batch_size: int = 50                  # Casos a exportar por iteración
    top_k_candidates: int = 5             # Candidatos por SOT
    uncertainty_threshold: float = 0.3    # Umbral para considerar "incierto"
    diversity_weight: float = 0.3         # Peso diversidad en hybrid
    min_candidates_per_sot: int = 2       # Mínimo candidatos por SOT
    include_engine_prediction: bool = True
    include_features: bool = True


@dataclass(frozen=True)
class AnnotationTask:
    """Tarea de anotación para un analista humano."""
    task_id: str
    sot_id: str
    sot_text: str
    sot_date_hint: Optional[str]
    candidates: list[dict[str, Any]]      # top-K candidatos con evidencia
    engine_prediction: Optional[str]      # Estado del motor actual
    engine_confidence: Optional[float]
    classifier_probability: Optional[float]
    created_at: str
    status: str = "pending"               # pending, annotated, skipped
    annotated_by: Optional[str] = None
    annotated_at: Optional[str] = None
    decision: Optional[FeedbackDecision] = None
    reason_code: Optional[str] = None
    comment: Optional[str] = None


@dataclass(frozen=True)
class ActiveLearningReport:
    """Reporte de una iteración de active learning."""
    iteration: int
    exported_count: int
    strategy: ActiveLearningStrategy
    uncertainty_stats: dict[str, float]
    exported_tasks: list[AnnotationTask]
    dataset_fingerprint: str
    created_at: str


class ActiveLearningLoop:
    """Orquesta el ciclo de active learning: export → anotar → importar → re-entrenar."""

    def __init__(
        self,
        repository: FeedbackRepository,
        feature_extractor: FeatureExtractor,
        candidate_retriever: CandidateRetriever,
        classifier: MatchingClassifier,
        model_registry: ModelRegistry,
        training_pipeline: TrainingPipeline,
    ):
        self._repo = repository
        self._extractor = feature_extractor
        self._retriever = candidate_retriever
        self._classifier = classifier
        self._registry = model_registry
        self._pipeline = training_pipeline

    def export_pending_for_annotation(
        self,
        prepared_sots: "pd.DataFrame",
        prepared_logs: "pd.DataFrame",
        results_df: "pd.DataFrame",
        config: ActiveLearningConfig = None,
        output_path: str | Path = None,
    ) -> ActiveLearningReport:
        """Exporta casos PENDIENTE con top-K candidatos para anotación humana.

        Args:
            prepared_sots: SOTs preparados
            prepared_logs: Bitácoras preparadas
            results_df: Resultados del motor actual (CrossingResult.results)
            config: Configuración de active learning
            output_path: Ruta para guardar CSV/JSON (None = solo retorna)

        Returns:
            ActiveLearningReport con tareas exportadas
        """
        config = config or ActiveLearningConfig()

        # 1. Obtener SOTs con estado PENDIENTE
        pending_sots = results_df[results_df["estado"] == "Pendiente"]
        if pending_sots.empty:
            return ActiveLearningReport(
                iteration=0,
                exported_count=0,
                strategy=config.strategy,
                uncertainty_stats={},
                exported_tasks=[],
                dataset_fingerprint="",
                created_at=datetime.utcnow().isoformat(),
            )

        # 2. Para cada SOT pendiente, obtener candidatos enriquecidos
        tasks = []
        uncertainties = []

        for _, row in pending_sots.iterrows():
            sot_id = str(row["sot"])
            engine_estado = str(row["estado"])
            engine_conf = row.get("confianza")

            # Obtener registro SOT
            sot_row = prepared_sots[prepared_sots["sot"] == sot_id]
            if sot_row.empty:
                continue
            sot_record = sot_row.iloc[0].to_dict()

            # Obtener bitácoras candidatas (mismo día si hay fecha)
            sot_date = sot_record.get("date_hint")
            if sot_date is not None:
                log_records = prepared_logs[prepared_logs["date"] == sot_date].to_dict("records")
            else:
                log_records = prepared_logs.to_dict("records")

            if not log_records:
                continue

            # Recuperar candidatos con features
            candidates = self._retriever.retrieve(
                sot_record, log_records, limit=config.top_k_candidates * 2
            )

            # Filtrar y enriquecer
            enriched_candidates = []
            for cand in candidates[:config.top_k_candidates]:
                log_rec = next((lr for lr in log_records if str(lr.get("record_id", "")) == cand.log_id), None)
                if not log_rec:
                    continue

                features = self._extractor.extract(sot_record, log_rec)
                enriched = {
                    "log_id": cand.log_id,
                    "log_text": log_rec.get("log_text", "")[:500],
                    "observation": log_rec.get("observation", "")[:300],
                    "materials_data": log_rec.get("materials_data", "")[:300],
                    "semantic_score": cand.semantic_score,
                    "classifier_probability": cand.classifier_probability,
                    "features": features if config.include_features else {},
                    "evidence": self._summarize_evidence(features),
                }
                enriched_candidates.append(enriched)

            if len(enriched_candidates) < config.min_candidates_per_sot:
                continue

            # Calcular incertidumbre del clasificador
            if self._classifier and hasattr(self._classifier, "_feature_names") and self._classifier._feature_names:
                proba = self._classifier.predict_proba([c.get("classifier_probability", 0.5) for c in enriched_candidates]) \
                    if any(c.get("classifier_probability") is not None for c in enriched_candidates) \
                    else None

                # Usar probabilidad del mejor candidato
                best_proba = max([c.get("classifier_probability", 0.5) for c in enriched_candidates]) if enriched_candidates else 0.5
                uncertainty = self._compute_uncertainty(best_proba)
                uncertainties.append(uncertainty)
            else:
                best_proba = None
                uncertainty = 0.5

            # Crear tarea
            task = AnnotationTask(
                task_id=f"task_{sot_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                sot_id=sot_id,
                sot_text=sot_record.get("source_text", "")[:1000],
                sot_date_hint=str(sot_date) if sot_date else None,
                candidates=enriched_candidates,
                engine_prediction=engine_estado,
                engine_confidence=float(engine_conf) if engine_conf is not None else None,
                classifier_probability=best_proba,
                created_at=datetime.utcnow().isoformat(),
            )
            tasks.append(task)

        # 3. Seleccionar según estrategia
        selected_tasks = self._select_by_strategy(tasks, uncertainties, config)

        # 4. Guardar si se especificó output_path
        if output_path:
            self._save_annotation_tasks(selected_tasks, output_path)

        # 5. Estadísticas
        uncertainty_stats = {
            "mean": float(np.mean(uncertainties)) if uncertainties else 0,
            "std": float(np.std(uncertainties)) if uncertainties else 0,
            "min": float(np.min(uncertainties)) if uncertainties else 0,
            "max": float(np.max(uncertainties)) if uncertainties else 0,
        }

        return ActiveLearningReport(
            iteration=self._get_next_iteration(),
            exported_count=len(selected_tasks),
            strategy=config.strategy,
            uncertainty_stats=uncertainty_stats,
            exported_tasks=selected_tasks,
            dataset_fingerprint=self._compute_fingerprint(selected_tasks),
            created_at=datetime.utcnow().isoformat(),
        )

    def import_annotations(
        self,
        annotations_path: str | Path,
        reviewer_id: str = "active_learning",
    ) -> int:
        """Importa anotaciones desde CSV/JSON exportado.

        Formato esperado (CSV):
        task_id, sot_id, log_id, decision, reason_code, comment

        Returns:
            Número de anotaciones importadas
        """
        path = Path(annotations_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        count = 0
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if self._import_annotation_row(row, reviewer_id):
                        count += 1
        elif path.suffix.lower() == ".json":
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
                for row in data:
                    if self._import_annotation_row(row, reviewer_id):
                        count += 1
        else:
            raise ValueError(f"Formato no soportado: {path.suffix}")

        return count

    def _import_annotation_row(self, row: dict[str, Any], reviewer_id: str) -> bool:
        """Importa una fila de anotación."""
        try:
            sot_id = str(row.get("sot_id", "")).strip()
            log_id = str(row.get("log_id", "")).strip()
            decision_str = str(row.get("decision", "")).strip().upper()

            if not sot_id or not log_id or decision_str not in ("CORRECT", "INCORRECT", "INCONCLUSIVE"):
                return False

            decision = FeedbackDecision(decision_str)
            reason = str(row.get("reason_code", "")).strip() or None
            comment = str(row.get("comment", "")).strip() or None

            # Verificar si ya existe feedback para este par
            existing = self._repo.get_feedback_for_pair(sot_id, log_id)
            supersedes = existing[-1].feedback_id if existing else None

            # Solo importar si es decisión de entrenamiento válida o INCONCLUSIVE
            if decision in (FeedbackDecision.CORRECT, FeedbackDecision.INCORRECT, FeedbackDecision.INCONCLUSIVE):
                event = FeedbackEvent(
                    sot_id=sot_id,
                    log_id=log_id,
                    decision=decision,
                    reason_code=reason,
                    comment=comment,
                    reviewer_id=reviewer_id,
                    supersedes_feedback_id=supersedes,
                    feature_schema_version="1.0",
                )
                self._repo.save_feedback(event)
                return True
            return False
        except Exception:
            return False

    def retrain_incremental(
        self,
        new_feedback_only: bool = True,
        config: ActiveLearningConfig = None,
    ) -> TrainingPipeline:
        """Re-entrena el modelo incrementalmente con nuevos datos.

        Args:
            new_feedback_only: Si True, usa solo feedback desde último entrenamiento
            config: Configuración para el entrenamiento

        Returns:
            TrainingPipeline entrenado
        """
        # En implementación completa, aquí se filtraría feedback nuevo
        # y se haría warm-start del modelo anterior
        # Por ahora, re-entrenamiento completo

        from sot_matcher.infrastructure.calibration import CalibrationConfig

        train_config = ActiveLearningConfig() if config is None else config
        calibration_config = CalibrationConfig(method="isotonic")

        # Obtener pares etiquetados
        labeled_pairs = self._repo.get_labeled_pairs("1.0")

        if len(labeled_pairs) < 20:
            raise ValueError(f"Insuficientes pares etiquetados ({len(labeled_pairs)} < 20)")

        # Entrenar
        pipeline = TrainingPipeline(
            self._repo, self._extractor, self._classifier, self._registry
        )

        report = pipeline.train(labeled_pairs, calibration_config)

        return pipeline

    def _select_by_strategy(
        self,
        tasks: list[AnnotationTask],
        uncertainties: list[float],
        config: ActiveLearningConfig,
    ) -> list[AnnotationTask]:
        """Selecciona tareas según estrategia de active learning."""
        if len(tasks) <= config.batch_size:
            return tasks

        strategy = config.strategy

        if strategy == ActiveLearningStrategy.UNCERTAINTY:
            # Ordenar por incertidumbre descendente
            indexed = list(zip(tasks, uncertainties))
            indexed.sort(key=lambda x: x[1], reverse=True)
            return [t for t, _ in indexed[:config.batch_size]]

        elif strategy == ActiveLearningStrategy.LEAST_CONFIDENT:
            # Menor confianza máxima
            indexed = list(zip(tasks, uncertainties))
            indexed.sort(key=lambda x: x[1], reverse=True)
            return [t for t, _ in indexed[:config.batch_size]]

        elif strategy == ActiveLearningStrategy.MARGIN:
            # Margin sampling: diferencia entre top-2 probabilidades
            margins = []
            for task in tasks:
                probs = [c.get("classifier_probability", 0.5) for c in task.candidates if c.get("classifier_probability") is not None]
                if len(probs) >= 2:
                    probs.sort(reverse=True)
                    margins.append(probs[0] - probs[1])
                else:
                    margins.append(1.0)
            indexed = list(zip(tasks, margins))
            indexed.sort(key=lambda x: x[1])  # menor margen = más incierto
            return [t for t, _ in indexed[:config.batch_size]]

        elif strategy == ActiveLearningStrategy.ENTROPY:
            # Entropía de la distribución de probabilidades
            entropies = []
            for task in tasks:
                probs = [c.get("classifier_probability", 0.5) for c in task.candidates if c.get("classifier_probability") is not None]
                if probs:
                    probs = np.array(probs)
                    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    entropies.append(entropy)
                else:
                    entropies.append(0)
            indexed = list(zip(tasks, entropies))
            indexed.sort(key=lambda x: x[1], reverse=True)
            return [t for t, _ in indexed[:config.batch_size]]

        elif strategy == ActiveLearningStrategy.DIVERSITY:
            # k-means++ para diversidad en espacio de features
            return self._diversity_sampling(tasks, config.batch_size)

        elif strategy == ActiveLearningStrategy.HYBRID:
            # Combinación uncertainty + diversity
            return self._hybrid_sampling(tasks, uncertainties, config)

        elif strategy == ActiveLearningStrategy.DISAGREEMENT:
            # Desacuerdo entre engine y classifier
            return self._disagreement_sampling(tasks, config)

        return tasks[:config.batch_size]

    def _diversity_sampling(self, tasks: list[AnnotationTask], k: int) -> list[AnnotationTask]:
        """k-means++ sampling en espacio de features para diversidad."""
        # Extraer feature vectors promedio por tarea
        vectors = []
        valid_tasks = []

        for task in tasks:
            feat_vectors = []
            for c in task.candidates:
                if c.get("features"):
                    feat_vec = list(c["features"].values())
                    if feat_vec:
                        feat_vectors.append(feat_vec)

            if feat_vectors:
                avg_vec = np.mean(feat_vectors, axis=0)
                vectors.append(avg_vec)
                valid_tasks.append(task)

        if len(vectors) <= k:
            return valid_tasks

        vectors = np.array(vectors)
        # Normalizar
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-10)

        # k-means++ selection
        selected_idx = [np.random.randint(len(vectors))]
        for _ in range(k - 1):
            if len(selected_idx) >= len(vectors):
                break
            dists = np.min(np.sum((vectors[:, None] - vectors[selected_idx]) ** 2, axis=2), axis=1)
            dists[selected_idx] = -1
            next_idx = np.argmax(dists)
            selected_idx.append(next_idx)

        return [valid_tasks[i] for i in selected_idx]

    def _hybrid_sampling(
        self,
        tasks: list[AnnotationTask],
        uncertainties: list[float],
        config: ActiveLearningConfig,
    ) -> list[AnnotationTask]:
        """Combina uncertainty y diversity sampling."""
        k = config.batch_size
        diversity_k = max(1, int(k * config.diversity_weight))
        uncertainty_k = k - diversity_k

        # Top uncertainty
        indexed = list(zip(tasks, uncertainties))
        indexed.sort(key=lambda x: x[1], reverse=True)
        uncertainty_selected = [t for t, _ in indexed[:uncertainty_k]]

        # Diversity del resto
        remaining = [t for t in tasks if t not in uncertainty_selected]
        if remaining and diversity_k > 0:
            diversity_selected = self._diversity_sampling(remaining, diversity_k)
            return uncertainty_selected + diversity_selected

        return uncertainty_selected

    def _disagreement_sampling(
        self,
        tasks: list[AnnotationTask],
        config: ActiveLearningConfig,
    ) -> list[AnnotationTask]:
        """Muestrea casos donde engine y classifier discrepan."""
        disagreements = []

        for task in tasks:
            engine_pred = task.engine_prediction
            clf_prob = task.classifier_probability

            if engine_pred == "Pendiente" and clf_prob is not None and clf_prob > 0.7:
                # Engine dice pendiente, classifier dice positivo con alta confianza
                disagreements.append((task, clf_prob))
            elif engine_pred in ("Exacto", "Semántico") and clf_prob is not None and clf_prob < 0.3:
                # Engine dice match, classifier dice negativo
                disagreements.append((task, 1 - clf_prob))

        if not disagreements:
            return tasks[:config.batch_size]

        disagreements.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in disagreements[:config.batch_size]]

    def _compute_uncertainty(self, proba: float) -> float:
        """Calcula incertidumbre de una probabilidad binaria."""
        # Entropía binaria normalizada
        if proba <= 0 or proba >= 1:
            return 0.0
        h = -proba * np.log2(proba) - (1 - proba) * np.log2(1 - proba)
        return h  # 0 a 1

    def _summarize_evidence(self, features: dict[str, Any]) -> dict[str, Any]:
        """Resume evidencia clave para anotador."""
        return {
            "incidencia_exacta": features.get("incidencia_exacta", 0) > 0.5,
            "sot_en_bitacora": features.get("sot_en_bitacora", 0) > 0.5,
            "fecha_coincide": features.get("fecha_coincide", 0) > 0.5,
            "dias_diferencia": features.get("dias_diferencia", -1),
            "coincidencia_zona": features.get("coincidencia_zona", 0) > 0.5,
            "coincidencia_actividad": features.get("coincidencia_actividad", 0) > 0.5,
            "tiene_materiales": features.get("tiene_materiales", 0) > 0.5,
        }

    def _save_annotation_tasks(
        self,
        tasks: list[AnnotationTask],
        output_path: str | Path,
    ) -> None:
        """Guarda tareas de anotación en CSV y JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # CSV para anotación humana
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "task_id", "sot_id", "sot_text", "sot_date_hint",
                "candidate_log_id", "candidate_text", "semantic_score",
                "classifier_probability", "evidence_summary",
                "engine_prediction", "engine_confidence",
                "decision", "reason_code", "comment", "annotator"
            ])
            for task in tasks:
                for c in task.candidates:
                    writer.writerow([
                        task.task_id, task.sot_id, task.sot_text, task.sot_date_hint or "",
                        c["log_id"], c["log_text"], c.get("semantic_score", ""),
                        c.get("classifier_probability", ""),
                        json.dumps(task._summarize_evidence(c.get("features", {}))),
                        task.engine_prediction, task.engine_confidence or "",
                        "", "", "", ""
                    ])

        # JSON completo para re-importación
        json_path = path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump([
                {
                    "task_id": t.task_id,
                    "sot_id": t.sot_id,
                    "sot_text": t.sot_text,
                    "sot_date_hint": t.sot_date_hint,
                    "candidates": t.candidates,
                    "engine_prediction": t.engine_prediction,
                    "engine_confidence": t.engine_confidence,
                    "classifier_probability": t.classifier_probability,
                    "created_at": t.created_at,
                }
                for t in tasks
            ], f, indent=2, ensure_ascii=False)

    def _get_next_iteration(self) -> int:
        """Obtiene número de siguiente iteración."""
        # En producción, persistir contador en BD o archivo
        try:
            iteration_file = Path("models/active_learning_iteration.txt")
            if iteration_file.exists():
                return int(iteration_file.read_text().strip()) + 1
        except Exception:
            pass
        return 1

    def _compute_fingerprint(self, tasks: list[AnnotationTask]) -> str:
        """Fingerprint del batch exportado."""
        import hashlib
        hasher = hashlib.sha256()
        for t in sorted(tasks, key=lambda x: x.task_id):
            hasher.update(f"{t.sot_id}|{t.task_id}".encode())
        return hasher.hexdigest()[:16]