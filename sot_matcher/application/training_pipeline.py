"""Pipeline de entrenamiento offline (HU-05, HU-06, HU-07).

Coordina dataset → entrenamiento → evaluación → registro.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from sot_matcher.application.ports import (
    FeedbackRepository,
    FeatureExtractor,
    MatchingClassifier,
    ModelRegistry,
    LabeledPair,
    TrainingReport,
    EvaluationReport,
    ModelMetadata,
    FeedbackFilters,
)
from sot_matcher.domain.feedback import FeedbackDecision, FeedbackEvent
from sot_matcher.domain.matching import MatchingEngine, MatchingSettings
from sot_matcher.infrastructure.calibration import (
    CalibrationConfig,
    CalibrationMetrics,
    create_calibrator,
    compute_calibration_metrics,
    plot_reliability_diagram,
)


@dataclass(frozen=True)
class DatasetReport:
    version: str
    total_events: int
    positive_pairs: int
    negative_pairs: int
    excluded_inconclusive: int
    excluded_conflicts: int
    feature_schema_version: str
    class_balance: float  # positivos / total
    dataset_fingerprint: str


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    group_by: str = "sot_id"  # evitar fuga por SOT
    classifier_config: dict[str, Any] = None
    calibration: CalibrationConfig = field(default_factory=lambda: CalibrationConfig(method="isotonic"))


@dataclass(frozen=True)
class ShadowEvaluationResult:
    """Resultado de evaluación en sombra (paralelo al motor actual)."""
    sot_id: str
    log_id: str
    true_label: int
    engine_prediction: str  # "Exacto" | "Semántico" | "Pendiente"
    engine_confidence: float | None
    classifier_probability: float | None
    classifier_prediction: int  # 0 | 1


class TrainingPipeline:
    def __init__(
        self,
        repository: FeedbackRepository,
        feature_extractor: FeatureExtractor,
        classifier: MatchingClassifier,
        model_registry: ModelRegistry,
    ):
        self._repo = repository
        self._extractor = feature_extractor
        self._classifier = classifier
        self._registry = model_registry
        self._calibrator: IsotonicRegression | None = None

    def build_dataset(
        self,
        prepared_sots: "pd.DataFrame",
        prepared_logs: "pd.DataFrame",
        filters: FeedbackFilters | None = None,
        dataset_version: str = "1.0",
    ) -> DatasetReport:
        """Construye dataset de entrenamiento a partir de feedback + registros preparados.

        Args:
            prepared_sots: DataFrame con columnas [sot, source_text, date_hint]
            prepared_logs: DataFrame con columnas [record_id, date, log_text, observation, materials_data]
            filters: Filtros opcionales para feedback
            dataset_version: Versión del esquema de features
        """
        import pandas as pd

        # 1. Recuperar eventos
        events = self._repo.list_feedback(filters or FeedbackFilters(limit=10000))

        # 2. Excluir INCONCLUSIVE
        labeled_events = [e for e in events if e.is_valid_for_training()]
        inconclusive = len(events) - len(labeled_events)

        # 3. Resolver conflictos (último evento por par, considerando supersedes)
        # Primero construir set de IDs superseded
        superseded_ids: set[str] = set()
        events_by_key: dict[tuple[str, str], list[FeedbackEvent]] = {}

        for e in sorted(labeled_events, key=lambda x: x.created_at):
            key = (e.sot_id, e.log_id)
            events_by_key.setdefault(key, []).append(e)
            if e.supersedes_feedback_id:
                superseded_ids.add(e.supersedes_feedback_id)

        # Para cada par, tomar el último evento que no esté superseded
        latest_by_pair: dict[tuple[str, str], FeedbackEvent] = {}
        for key, ev_list in events_by_key.items():
            for event in reversed(ev_list):
                if event.feedback_id not in superseded_ids:
                    latest_by_pair[key] = event
                    break

        # 4. Detectar conflictos reales (pares con decisiones contradictorias en historial no superseded)
        conflicts = 0
        for key, ev_list in events_by_key.items():
            valid_events = [e for e in ev_list if e.feedback_id not in superseded_ids]
            if len(valid_events) > 1:
                decisions = {e.decision for e in valid_events}
                if len(decisions) > 1:
                    conflicts += 1

        # 5. Construir mapas de lookup rápido
        sots_by_id = prepared_sots.set_index("sot").to_dict("index")
        logs_by_id = prepared_logs.set_index("record_id").to_dict("index")

        # 6. Construir LabeledPairs con características extraídas
        pairs = []
        for (sot_id, log_id), event in latest_by_pair.items():
            sot_rec = sots_by_id.get(sot_id)
            log_rec = logs_by_id.get(log_id)

            if sot_rec is None or log_rec is None:
                # Skip si no encontramos los registros originales
                continue

            features = self._extractor.extract(sot_rec, log_rec)

            pairs.append(LabeledPair(
                sot_id=sot_id,
                log_id=log_id,
                label=event.label(),
                features=features,
                feature_schema_version=self._extractor.schema_version,
                feedback_ids=(event.feedback_id,),
            ))

        # 7. Calcular fingerprint del dataset
        fingerprint = self._compute_dataset_fingerprint(pairs)

        return DatasetReport(
            version=dataset_version,
            total_events=len(events),
            positive_pairs=sum(1 for p in pairs if p.label == 1),
            negative_pairs=sum(1 for p in pairs if p.label == 0),
            excluded_inconclusive=inconclusive,
            excluded_conflicts=conflicts,
            feature_schema_version=self._extractor.schema_version,
            class_balance=sum(1 for p in pairs if p.label == 1) / len(pairs) if pairs else 0.0,
            dataset_fingerprint=fingerprint,
        )

    def train(
        self,
        labeled_pairs: Iterable[LabeledPair],
        config: TrainConfig = None,
    ) -> TrainingReport:
        config = config or TrainConfig()
        pairs = list(labeled_pairs)

        if not pairs:
            raise ValueError("No hay pares etiquetados para entrenar")

        # Verificar ambas clases
        labels = [p.label for p in pairs]
        if len(set(labels)) < 2:
            raise ValueError("Se requieren ambas clases (positivos y negativos)")

        # Separar por grupos (evitar fuga por SOT)
        groups = [p.sot_id for p in pairs]
        gss = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_state)
        train_idx, test_idx = next(gss.split(pairs, groups=groups))

        train_pairs = [pairs[i] for i in train_idx]
        test_pairs = [pairs[i] for i in test_idx]

        # Guardar test set para evaluación posterior (persistir en archivo)
        test_fingerprint = self._compute_dataset_fingerprint(test_pairs)
        self._persist_test_set(test_pairs, test_fingerprint)

        # Validación desde entrenamiento
        if config.val_size > 0:
            gss2 = GroupShuffleSplit(
                n_splits=1,
                test_size=config.val_size / (1 - config.test_size),
                random_state=config.random_state
            )
            train_idx2, val_idx2 = next(gss2.split(train_pairs, groups=[p.sot_id for p in train_pairs]))
            train_final = [train_pairs[i] for i in train_idx2]
            val_pairs = [train_pairs[i] for i in val_idx2]
        else:
            train_final = train_pairs
            val_pairs = []

        # Entrenar clasificador base
        report = self._classifier.fit(train_final, val_pairs or train_final)

        # Calibrar probabilidades si se solicita
        if config.calibration.method != "none":
            self._calibrate(train_final, val_pairs, config.calibration)

        # Guardar fingerprint del test set en el reporte para poder cargarlo después
        report = TrainingReport(
            model_version=report.model_version,
            feature_schema_version=report.feature_schema_version,
            train_size=report.train_size,
            val_size=report.val_size,
            test_size=len(test_pairs),
            metrics=report.metrics,
            config={**report.config, "calibration_method": config.calibration.method, "test_fingerprint": test_fingerprint},
            created_at=report.created_at,
        )

        return report

    def _calibrate(
        self,
        train_pairs: list[LabeledPair],
        val_pairs: list[LabeledPair],
        calibration_config: CalibrationConfig,
    ) -> None:
        """Calibra probabilidades del clasificador usando validation set."""
        if not val_pairs:
            return

        y_val = np.array([p.label for p in val_pairs])
        proba = self._classifier.predict_proba([p.features for p in val_pairs])

        # Crear y entrenar calibrador usando el módulo de calibración
        self._calibrator = create_calibrator(calibration_config)
        self._calibrator.fit(proba, y_val)

        # Guardar configuración de calibración en el clasificador para persistencia
        self._calibration_config = calibration_config

    def predict_proba_calibrated(self, feature_vectors: list[dict[str, Any]]) -> list[float]:
        """Predice probabilidades calibradas usando el calibrador del módulo de calibración."""
        raw_proba = self._classifier.predict_proba(feature_vectors)
        if hasattr(self, "_calibrator") and self._calibrator is not None:
            return self._calibrator.transform(np.array(raw_proba)).tolist()
        return raw_proba

    def evaluate(
        self,
        model: MatchingClassifier,
        test_pairs: Iterable[LabeledPair],
        baseline_predictions: list[ShadowEvaluationResult] | None = None,
    ) -> EvaluationReport:
        test_list = list(test_pairs)
        if not test_list:
            raise ValueError("Test set vacío")

        X_test = [p.features for p in test_list]
        y_true = [p.label for p in test_list]

        # Usar probabilidades calibradas si están disponibles
        if hasattr(self, "_calibrator") and self._calibrator is not None:
            y_proba = self.predict_proba_calibrated(X_test)
        else:
            y_proba = model.predict_proba(X_test)

        y_pred = [1 if p >= 0.5 else 0 for p in y_proba]

        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            precision_recall_curve, auc, brier_score_loss, log_loss
        )

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "pr_auc": float(auc(*precision_recall_curve(y_true, y_proba)[:2])),
            "brier_score": float(brier_score_loss(y_true, y_proba)),
            "log_loss": float(log_loss(y_true, y_proba)),
        }

        # Calibración metrics usando el módulo de calibración
        cal_metrics = compute_calibration_metrics(np.array(y_true), np.array(y_proba))
        calibration_metrics = {
            "ece": cal_metrics.ece,
            "mce": cal_metrics.mce,
            "brier_score": cal_metrics.brier_score,
            "log_loss": cal_metrics.log_loss,
        }

        # Baseline comparison
        baseline_metrics = {}
        if baseline_predictions:
            baseline_metrics = self._compare_with_baseline(baseline_predictions, y_true, y_pred)

        # Recomendación
        if metrics["precision"] >= 0.85 and metrics["recall"] >= 0.70:
            recommendation = "ACTIVATE"
        elif metrics["precision"] < 0.70:
            recommendation = "REJECT"
        else:
            recommendation = "NEEDS_MORE_DATA"

        return EvaluationReport(
            model_version=model.metadata().model_version,
            baseline_version="current_engine",
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            calibration_metrics=calibration_metrics,
            recommendation=recommendation,
        )

    def run_shadow_evaluation(
        self,
        prepared_sots: "pd.DataFrame",
        prepared_logs: "pd.DataFrame",
        results_df: "pd.DataFrame",
        classifier_model: MatchingClassifier | None = None,
    ) -> list[ShadowEvaluationResult]:
        """Ejecuta evaluación en sombra: compara clasificador vs motor actual en SOTs con feedback.

        Args:
            prepared_sots: SOTs preparados
            prepared_logs: Bitácoras preparadas
            results_df: Resultados del motor actual (CrossingResult.results)
            classifier_model: Modelo a evaluar (usa self._classifier si None)
        """
        import pandas as pd

        model = classifier_model or self._classifier
        if model is None or not hasattr(model, "_feature_names") or model._feature_names is None:
            raise ValueError("No hay modelo entrenado/cargado para evaluación en sombra")

        # Obtener eventos con feedback válido
        events = self._repo.list_feedback(FeedbackFilters(limit=10000))
        labeled_events = [e for e in events if e.is_valid_for_training()]

        # Resolver supersedes
        superseded_ids: set[str] = set()
        events_by_key: dict[tuple[str, str], list[FeedbackEvent]] = {}
        for e in sorted(labeled_events, key=lambda x: x.created_at):
            key = (e.sot_id, e.log_id)
            events_by_key.setdefault(key, []).append(e)
            if e.supersedes_feedback_id:
                superseded_ids.add(e.supersedes_feedback_id)

        latest_by_pair: dict[tuple[str, str], FeedbackEvent] = {}
        for key, ev_list in events_by_key.items():
            for event in reversed(ev_list):
                if event.feedback_id not in superseded_ids:
                    latest_by_pair[key] = event
                    break

        # Lookup maps
        sots_by_id = prepared_sots.set_index("sot").to_dict("index")
        logs_by_id = prepared_logs.set_index("record_id").to_dict("index")
        results_by_sot = results_df.set_index("sot").to_dict("index")

        shadow_results = []

        for (sot_id, log_id), event in latest_by_pair.items():
            sot_rec = sots_by_id.get(sot_id)
            log_rec = logs_by_id.get(log_id)
            result_rec = results_by_sot.get(sot_id)

            if sot_rec is None or log_rec is None or result_rec is None:
                continue

            # Extraer features
            features = self._extractor.extract(sot_rec, log_rec)

            # Predicción clasificador
            proba = model.predict_proba([features])[0]
            clf_pred = 1 if proba >= 0.5 else 0

            # Predicción motor actual
            engine_estado = result_rec.get("estado", "Pendiente")
            engine_conf = result_rec.get("confianza")

            shadow_results.append(ShadowEvaluationResult(
                sot_id=sot_id,
                log_id=log_id,
                true_label=event.label(),
                engine_prediction=engine_estado,
                engine_confidence=float(engine_conf) if engine_conf is not None else None,
                classifier_probability=proba,
                classifier_prediction=clf_pred,
            ))

        return shadow_results

    def _compare_with_baseline(
        self,
        baseline_predictions: list[ShadowEvaluationResult],
        y_true: list[int],
        y_pred: list[int],
    ) -> dict[str, float]:
        """Compara métricas del clasificador vs motor actual."""
        # Solo considerar casos donde el motor hizo una predicción (no Pendiente)
        engine_mask = [bp.engine_prediction != "Pendiente" for bp in baseline_predictions]
        if not any(engine_mask):
            return {"engine_coverage": 0.0}

        y_true_engine = [y_true[i] for i, m in enumerate(engine_mask) if m]
        y_pred_engine = [1 if bp.engine_prediction in ("Exacto", "Semántico") else 0 for i, bp in enumerate(baseline_predictions) if engine_mask[i]]
        y_pred_clf = [y_pred[i] for i, m in enumerate(engine_mask) if m]

        from sklearn.metrics import precision_score, recall_score, f1_score

        return {
            "engine_coverage": float(sum(engine_mask) / len(engine_mask)),
            "engine_precision": float(precision_score(y_true_engine, y_pred_engine, zero_division=0)),
            "engine_recall": float(recall_score(y_true_engine, y_pred_engine, zero_division=0)),
            "engine_f1": float(f1_score(y_true_engine, y_pred_engine, zero_division=0)),
            "classifier_precision_on_engine": float(precision_score(y_true_engine, y_pred_clf, zero_division=0)),
            "classifier_recall_on_engine": float(recall_score(y_true_engine, y_pred_clf, zero_division=0)),
        }

    def _persist_test_set(self, test_pairs: list[LabeledPair], fingerprint: str) -> None:
        """Persiste test set para reproducibilidad."""
        test_dir = Path("models/test_sets")
        test_dir.mkdir(parents=True, exist_ok=True)
        test_path = test_dir / f"test_{fingerprint}.json"

        data = {
            "fingerprint": fingerprint,
            "size": len(test_pairs),
            "pairs": [
                {
                    "sot_id": p.sot_id,
                    "log_id": p.log_id,
                    "label": p.label,
                    "features": p.features,
                    "feature_schema_version": p.feature_schema_version,
                    "feedback_ids": list(p.feedback_ids),
                }
                for p in test_pairs
            ],
        }
        test_path.write_text(json.dumps(data, indent=2))

    def load_test_set(self, fingerprint: str) -> list[LabeledPair]:
        """Carga test set persistido."""
        test_path = Path("models/test_sets") / f"test_{fingerprint}.json"
        if not test_path.exists():
            raise FileNotFoundError(f"Test set no encontrado: {test_path}")

        data = json.loads(test_path.read_text())
        return [
            LabeledPair(
                sot_id=p["sot_id"],
                log_id=p["log_id"],
                label=p["label"],
                features=p["features"],
                feature_schema_version=p["feature_schema_version"],
                feedback_ids=tuple(p["feedback_ids"]),
            )
            for p in data["pairs"]
        ]

    def _compute_dataset_fingerprint(self, pairs: list[LabeledPair]) -> str:
        """Calcula fingerprint determinista del dataset."""
        hasher = hashlib.sha256()
        for p in sorted(pairs, key=lambda x: (x.sot_id, x.log_id)):
            # Incluir features en fingerprint para detectar cambios en extractor
            feat_str = "|".join(f"{k}={v}" for k, v in sorted(p.features.items()))
            hasher.update(f"{p.sot_id}|{p.log_id}|{p.label}|{feat_str}".encode())
        return hasher.hexdigest()[:16]

    def register_candidate(self, model: MatchingClassifier, report: TrainingReport) -> ModelMetadata:
        meta = model.save(f"models/{model.metadata().model_version}.cbm")

        # Guardar calibrador si existe
        if hasattr(self, "_calibrator") and self._calibrator is not None:
            calib_path = Path(meta.artifact_path).with_suffix(".calib.pkl")
            import pickle
            calib_path.write_bytes(pickle.dumps(self._calibrator))
            meta = ModelMetadata(
                model_version=meta.model_version,
                artifact_path=meta.artifact_path,
                feature_schema_version=meta.feature_schema_version,
                created_at=meta.created_at,
                metrics=meta.metrics,
                training_dataset_fingerprint=meta.training_dataset_fingerprint,
                status=meta.status,
                parent_model_version=meta.parent_model_version,
            )

        self._registry.register(meta)
        return meta