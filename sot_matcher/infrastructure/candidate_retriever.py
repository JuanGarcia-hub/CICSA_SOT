"""Recuperador de candidatos para una SOT.

Implementa CandidateRetriever protocol. Reutiliza evidencia del motor actual.
"""

from __future__ import annotations

from typing import Any

from sot_matcher.application.ports import CandidateRetriever
from sot_matcher.domain.feedback import CandidatePair
from sot_matcher.domain.text import (
    normalize_text,
    extract_incident_ids,
    has_token_overlap,
    token_set,
)


class SotCandidateRetriever:
    """Recupera candidatos para una SOT basándose en evidencia del motor actual."""

    def __init__(self, embedder: Any = None):
        self._embedder = embedder

    def retrieve(
        self,
        sot_record: dict[str, Any],
        log_records: list[dict[str, Any]],
        limit: int = 10,
        context: dict[str, Any] | None = None,
    ) -> list[CandidatePair]:
        sot_id = str(sot_record.get("sot", ""))
        sot_text = str(sot_record.get("source_text", ""))
        sot_date = sot_record.get("date_hint")

        norm_sot = normalize_text(sot_text)
        sot_incs = extract_incident_ids(sot_text)
        sot_tokens = token_set(norm_sot)

        candidates = []

        for log_rec in log_records:
            log_id = str(log_rec.get("record_id", ""))
            log_text = str(log_rec.get("log_text", ""))
            obs_text = str(log_rec.get("observation", ""))
            combined_log = f"{log_text} {obs_text}".strip()
            norm_log = normalize_text(combined_log)

            # Filtrar por fecha si hay date_hint
            if sot_date is not None:
                log_date = log_rec.get("date")
                if log_date != sot_date:
                    continue

            # Calcular evidencia
            log_incs = extract_incident_ids(combined_log)
            incidencia_exacta = bool(sot_incs & log_incs)
            sot_digits = "".join(c for c in norm_sot if c.isdigit())
            sot_en_bitacora = bool(sot_digits and sot_digits in norm_log)
            token_overlap = has_token_overlap(sot_tokens, token_set(norm_log))

            # Score de evidencia
            evidence_score = 0
            if incidencia_exacta:
                evidence_score += 100
            if sot_en_bitacora:
                evidence_score += 50
            if token_overlap:
                evidence_score += 10

            # Similitud semántica si hay embedder
            semantic_score = None
            if self._embedder and norm_sot and norm_log:
                try:
                    vecs = self._embedder.encode([norm_sot, norm_log], normalize_embeddings=True)
                    semantic_score = float(vecs[0] @ vecs[1])
                    evidence_score += int(semantic_score * 20)
                except Exception:
                    pass

            candidates.append(CandidatePair(
                sot_id=sot_id,
                log_id=log_id,
                features={},  # se llenará después
                semantic_score=semantic_score,
                classifier_probability=None,
            ))

        # Ordenar por evidence_score descendente
        candidates.sort(key=lambda c: c.semantic_score or 0, reverse=True)
        return candidates[:limit]