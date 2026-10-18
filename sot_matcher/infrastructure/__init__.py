"""Adaptadores de I/O para libros Excel y modelo de embeddings."""

from .excel_reader import (
    read_excel,
    read_workbook_meta,
    detect_sheet,
    classify_uploads,
    prepare_sots,
    prepare_logs,
    FileProfile,
    SheetRole,
    InputValidationError,
    SOT_SHEET,
    LOG_SHEET,
)
from .excel_export import to_excel
from .embedding import get_embedder, build_embedder, SentenceTransformerEmbedder
from .feedback_repository import SqliteFeedbackRepository, init_db
from .feature_extractor import SotLogFeatureExtractor
from .candidate_retriever import SotCandidateRetriever
from .model_registry import FileModelRegistry
from .classifier import CatBoostMatchingClassifier
from .calibration import (
    BaseCalibrator,
    TemperatureScaler,
    IsotonicCalibrator,
    PlattCalibrator,
    HistogramBinningCalibrator,
    IdentityCalibrator,
    CalibrationConfig,
    CalibrationMetrics,
    create_calibrator,
    compute_calibration_metrics,
    plot_reliability_diagram,
    compare_calibration_methods,
)
from .active_learning import (
    ActiveLearningLoop,
    ActiveLearningConfig,
    ActiveLearningStrategy,
    AnnotationTask,
    ActiveLearningReport,
)

__all__ = [
    "read_excel",
    "read_workbook_meta",
    "detect_sheet",
    "classify_uploads",
    "prepare_sots",
    "prepare_logs",
    "FileProfile",
    "SheetRole",
    "InputValidationError",
    "SOT_SHEET",
    "LOG_SHEET",
    "to_excel",
    "get_embedder",
    "build_embedder",
    "SentenceTransformerEmbedder",
    "SqliteFeedbackRepository",
    "init_db",
    "SotLogFeatureExtractor",
    "SotCandidateRetriever",
    "FileModelRegistry",
    "CatBoostMatchingClassifier",
    "BaseCalibrator",
    "TemperatureScaler",
    "IsotonicCalibrator",
    "PlattCalibrator",
    "HistogramBinningCalibrator",
    "IdentityCalibrator",
    "CalibrationConfig",
    "CalibrationMetrics",
    "create_calibrator",
    "compute_calibration_metrics",
    "plot_reliability_diagram",
    "compare_calibration_methods",
    "ActiveLearningLoop",
    "ActiveLearningConfig",
    "ActiveLearningStrategy",
    "AnnotationTask",
    "ActiveLearningReport",
]