"""API pública del paquete `sot_matcher`.

Expone la fachada del cruce y los tipos esenciales para que la UI y las
pruebas no dependan de los detalles internos.
"""

from .application import (
    CrossingPipeline,
    CrossingResult,
    PreparedInput,
    FeedbackService,
    SubmitDecisionCommand,
    ReviewQueueItem,
    CandidateReview,
    TrainingPipeline,
    TrainConfig,
    DatasetReport,
    FeedbackRepository,
    FeatureExtractor,
    CandidateRetriever,
    MatchingClassifier,
    ModelRegistry,
    FeedbackFilters,
    TrainingReport,
    EvaluationReport,
    ModelMetadata,
    parse_sot_ids,
    ids_from_file,
    filter_results,
    summarize,
    build_pipeline,
)
from .domain.matching import MatchingEngine, MatchingSettings
from .domain.feedback import (
    FeedbackDecision,
    FeedbackEvent,
    CandidatePair,
    MatchingPrediction,
    LabeledPair,
)
from .infrastructure import (
    SqliteFeedbackRepository,
    init_db,
    SotLogFeatureExtractor,
    SotCandidateRetriever,
    FileModelRegistry,
    CatBoostMatchingClassifier,
    InputValidationError,
    SheetRole,
    to_excel,
)
from .infrastructure.excel_reader import InputValidationError as ExcelInputValidationError, SheetRole as ExcelSheetRole

__all__ = [
    # Application - Pipeline
    "CrossingPipeline",
    "CrossingResult",
    "PreparedInput",
    "build_pipeline",
    "parse_sot_ids",
    "ids_from_file",
    "filter_results",
    "summarize",
    # Application - Feedback
    "FeedbackService",
    "SubmitDecisionCommand",
    "ReviewQueueItem",
    "CandidateReview",
    # Application - Training
    "TrainingPipeline",
    "TrainConfig",
    "DatasetReport",
    # Application - Ports
    "FeedbackRepository",
    "FeatureExtractor",
    "CandidateRetriever",
    "MatchingClassifier",
    "ModelRegistry",
    "FeedbackFilters",
    "TrainingReport",
    "EvaluationReport",
    "ModelMetadata",
    # Domain - Matching
    "MatchingEngine",
    "MatchingSettings",
    # Domain - Feedback
    "FeedbackDecision",
    "FeedbackEvent",
    "CandidatePair",
    "MatchingPrediction",
    "LabeledPair",
    # Infrastructure
    "SqliteFeedbackRepository",
    "init_db",
    "SotLogFeatureExtractor",
    "SotCandidateRetriever",
    "FileModelRegistry",
    "CatBoostMatchingClassifier",
    "InputValidationError",
    "SheetRole",
    "to_excel",
]