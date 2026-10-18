"""Fachada (Facade) que orquesta lectura, preparación y cruce del pipeline."""

from .pipeline import CrossingPipeline, PreparedInput, CrossingResult, build_pipeline
from .massive_search import (
    parse_sot_ids,
    ids_from_file,
    filter_results,
    summarize,
)
from .ports import (
    FeedbackRepository,
    FeatureExtractor,
    CandidateRetriever,
    MatchingClassifier,
    ModelRegistry,
    FeedbackFilters,
    TrainingReport,
    EvaluationReport,
    ModelMetadata,
)
from .feedback_service import FeedbackService, SubmitDecisionCommand, ReviewQueueItem, CandidateReview
from .training_pipeline import (
    TrainingPipeline,
    TrainConfig,
    DatasetReport,
    ShadowEvaluationResult,
)

__all__ = [
    "CrossingPipeline",
    "PreparedInput",
    "CrossingResult",
    "build_pipeline",
    "parse_sot_ids",
    "ids_from_file",
    "filter_results",
    "summarize",
    "FeedbackRepository",
    "FeatureExtractor",
    "CandidateRetriever",
    "MatchingClassifier",
    "ModelRegistry",
    "FeedbackFilters",
    "TrainingReport",
    "EvaluationReport",
    "ModelMetadata",
    "FeedbackService",
    "SubmitDecisionCommand",
    "ReviewQueueItem",
    "CandidateReview",
    "TrainingPipeline",
    "TrainConfig",
    "DatasetReport",
    "ShadowEvaluationResult",
]