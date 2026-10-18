"""Contratos (interfaces) para inyección de dependencias.

Application coordina casos de uso mediante contratos.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, Any, Optional

from sot_matcher.domain.feedback import (
    FeedbackEvent,
    FeedbackDecision,
    CandidatePair,
    LabeledPair,
    MatchingPrediction,
)


@dataclass(frozen=True)
class FeedbackFilters:
    decision: FeedbackDecision | None = None
    reviewer_id: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    sot_id: str | None = None
    run_id: str | None = None
    limit: int = 100
    offset: int = 0


class FeedbackRepository(Protocol):
    """Persistencia de eventos de retroalimentación."""

    def save_feedback(self, event: FeedbackEvent) -> FeedbackEvent: ...
    def get_feedback(self, feedback_id: str) -> FeedbackEvent | None: ...
    def list_feedback(self, filters: FeedbackFilters) -> list[FeedbackEvent]: ...
    def get_labeled_pairs(self, dataset_version: str) -> list[LabeledPair]: ...
    def get_feedback_for_pair(self, sot_id: str, log_id: str) -> list[FeedbackEvent]: ...


class FeatureExtractor(Protocol):
    """Extracción de características reproducibles para un par candidato."""

    def extract(
        self,
        sot_record: dict[str, Any],
        log_record: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    @property
    def schema_version(self) -> str: ...


class CandidateRetriever(Protocol):
    """Recuperación de candidatos sin descartar prematuramente."""

    def retrieve(
        self,
        sot_record: dict[str, Any],
        log_records: list[dict[str, Any]],
        limit: int = 10,
        context: dict[str, Any] | None = None,
    ) -> list[CandidatePair]: ...


class MatchingClassifier(Protocol):
    """Interfaz independiente del algoritmo de clasificación."""

    def fit(
        self,
        training_data: Iterable[LabeledPair],
        validation_data: Iterable[LabeledPair],
    ) -> "TrainingReport": ...

    def predict_proba(self, feature_vectors: list[dict[str, Any]]) -> list[float]: ...

    def save(self, artifact_path: str) -> "ModelMetadata": ...

    @classmethod
    def load(cls, artifact_path: str) -> "MatchingClassifier": ...

    def metadata(self) -> "ModelMetadata": ...


@dataclass(frozen=True)
class TrainingReport:
    model_version: str
    feature_schema_version: str
    train_size: int
    val_size: int
    test_size: int
    metrics: dict[str, float]
    config: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class EvaluationReport:
    model_version: str
    baseline_version: str
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    calibration_metrics: dict[str, float] | None
    recommendation: str  # "ACTIVATE" | "REJECT" | "NEEDS_MORE_DATA"


@dataclass(frozen=True)
class ModelMetadata:
    model_version: str
    artifact_path: str
    feature_schema_version: str
    created_at: str
    metrics: dict[str, float]
    training_dataset_fingerprint: str
    status: str  # CANDIDATE, APPROVED, ACTIVE, REJECTED, RETIRED
    parent_model_version: str | None


class ModelRegistry(Protocol):
    """Registro de modelos versionados."""

    def register(self, metadata: ModelMetadata) -> None: ...
    def get_active_model(self) -> ModelMetadata | None: ...
    def activate(self, model_version: str, approval: str) -> None: ...
    def rollback(self, previous_version: str) -> None: ...
    def list_models(self, status: str | None = None) -> list[ModelMetadata]: ...