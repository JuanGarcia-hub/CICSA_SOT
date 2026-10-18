"""Tipos de dominio para retroalimentación supervisada.

Principio: Domain no depende de SQLite, Streamlit, CatBoost ni Sentence Transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class FeedbackDecision(str, Enum):
    """Decisión de revisión humana sobre un par SOT-bitácora."""
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class FeedbackEvent:
    """Evento inmutable de retroalimentación. Correcciones crean nuevo evento."""
    feedback_id: str = field(default_factory=lambda: str(uuid4()))
    run_id: Optional[str] = None
    sot_id: str = ""
    log_id: str = ""
    decision: FeedbackDecision = FeedbackDecision.INCONCLUSIVE
    reason_code: Optional[str] = None
    comment: Optional[str] = None
    reviewer_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    supersedes_feedback_id: Optional[str] = None
    feature_schema_version: Optional[str] = None

    def is_valid_for_training(self) -> bool:
        """Solo CORRECT/INCORRECT entrenan; INCONCLUSIVE se excluye."""
        return self.decision in (FeedbackDecision.CORRECT, FeedbackDecision.INCORRECT)

    def label(self) -> int:
        """Etiqueta binaria: 1 = correcto, 0 = incorrecto."""
        if self.decision == FeedbackDecision.CORRECT:
            return 1
        if self.decision == FeedbackDecision.INCORRECT:
            return 0
        raise ValueError("INCONCLUSIVE no tiene etiqueta binaria")


@dataclass(frozen=True)
class CandidatePair:
    """Par candidato SOT-bitácora con características."""
    sot_id: str
    log_id: str
    run_id: Optional[str] = None
    features: dict[str, Any] = field(default_factory=dict)
    feature_schema_version: str = "1.0"
    semantic_score: Optional[float] = None
    classifier_probability: Optional[float] = None


@dataclass(frozen=True)
class MatchingPrediction:
    """Predicción del sistema (motor actual o clasificador) para un par."""
    sot_id: str
    log_id: str
    semantic_score: Optional[float] = None
    classifier_probability: Optional[float] = None
    model_version: Optional[str] = None
    recommended_decision: FeedbackDecision = FeedbackDecision.INCONCLUSIVE
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LabeledPair:
    """Par etiquetado listo para entrenamiento."""
    sot_id: str
    log_id: str
    label: int  # 1 = correcto, 0 = incorrecto
    features: dict[str, Any]
    feature_schema_version: str
    feedback_ids: tuple[str, ...]  # eventos que respaldan esta etiqueta