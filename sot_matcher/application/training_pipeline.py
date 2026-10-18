"""Pipeline de entrenamiento offline (HU-05, HU-06, HU-07).

Coordina dataset → entrenamiento → evaluación → registro.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sot_matcher.application.ports import (
    FeedbackRepository,
    FeatureExtractor,
    MatchingClassifier,
    ModelRegistry,
    LabeledPair,
    TrainingReport,
    EvaluationReport,
    ModelMetadata,
)
from sot_matcher.domain.feedback import FeedbackDecision
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


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


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    group_by: str = "sot_id"  # evitar fuga por SOT
    classifier_config: dict[str, Any] = None


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

    def build_dataset(
        self,
        filters: Any = None,
        dataset_version: str = "1.0",
    ) -> DatasetReport:
        # 1. Recuperar eventos
        events = self._repo.list_feedback(filters or FeedbackFilters(limit=10000))

        # 2. Excluir INCONCLUSIVE
        labeled_events = [e for e in events if e.is_valid_for_training()]
        inconclusive = len(events) - len(labeled_events)

        # 3. Resolver conflictos (último evento por par, considerando supersedes)
        latest_by_pair: dict[tuple[str, str], FeedbackEvent] = {}
        for e in sorted(labeled_events, key=lambda x: x.created_at):
            key = (e.sot_id, e.log_id)
            if e.supersedes_feedback_id:
                continue
            latest_by_pair[key] = e

        # 4. Detectar conflictos (pares con decisiones contradictorias en historial)
        conflicts = 0
        # (simplificado: en producción verificar historial completo)

        # 5. Construir LabeledPairs con características
        # Requiere acceso a registros originales para extraer features
        pairs = []
        for (sot_id, log_id), event in latest_by_pair.items():
            pairs.append(LabeledPair(
                sot_id=sot_id,
                log_id=log_id,
                label=event.label(),
                features={},  # se llenará en entrenamiento
                feature_schema_version=self._extractor.schema_version,
                feedback_ids=(event.feedback_id,),
            ))

        return DatasetReport(
            version=dataset_version,
            total_events=len(events),
            positive_pairs=sum(1 for p in pairs if p.label == 1),
            negative_pairs=sum(1 for p in pairs if p.label == 0),
            excluded_inconclusive=inconclusive,
            excluded_conflicts=conflicts,
            feature_schema_version=self._extractor.schema_version,
            class_balance=sum(1 for p in pairs if p.label == 1) / len(pairs) if pairs else 0.0,
        )

    def train(
        self,
        labeled_pairs: Iterable[LabeledPair],
        config: TrainConfig = None,
    ) -> TrainingReport:
        config = config or TrainConfig()
        pairs = list(labeled_pairs)

        # Llenar características (requiere acceso a registros originales)
        # for pair in pairs:
        #     pair.features = self._extractor.extract(sot_rec, log_rec)

        # Separar por grupos (evitar fuga)
        groups = [p.sot_id for p in pairs]
        gss = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_state)
        train_idx, test_idx = next(gss.split(pairs, groups=groups))

        train_pairs = [pairs[i] for i in train_idx]
        test_pairs = [pairs[i] for i in test_idx]

        # Validación desde entrenamiento
        if config.val_size > 0:
            gss2 = GroupShuffleSplit(n_splits=1, test_size=config.val_size / (1 - config.test_size), random_state=config.random_state)
            train_idx2, val_idx2 = next(gss2.split(train_pairs, groups=[p.sot_id for p in train_pairs]))
            train_final = [train_pairs[i] for i in train_idx2]
            val_pairs = [train_pairs[i] for i in val_idx2]
        else:
            train_final = train_pairs
            val_pairs = []

        # Entrenar
        report = self._classifier.fit(train_final, val_pairs or train_final)

        # Guardar test set para evaluación posterior
        # (en producción: persistir test_pairs con fingerprint)

        return report

    def evaluate(
        self,
        model: MatchingClassifier,
        test_pairs: Iterable[LabeledPair],
        baseline_predictions: list = None,
    ) -> EvaluationReport:
        test_list = list(test_pairs)
        X_test = [p.features for p in test_list]
        y_true = [p.label for p in test_list]

        y_proba = model.predict_proba(X_test)
        y_pred = [1 if p >= 0.5 else 0 for p in y_proba]

        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            precision_recall_curve, auc
        )

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "pr_auc": float(auc(*precision_recall_curve(y_true, y_proba)[:2])),
        }

        baseline_metrics = {}
        if baseline_predictions:
            # Comparar contra motor actual en mismo set
            pass

        # Calibración
        calibration = {}

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
            calibration_metrics=calibration or None,
            recommendation=recommendation,
        )

    def register_candidate(self, model: MatchingClassifier, report: TrainingReport) -> ModelMetadata:
        meta = model.save(f"models/{model.metadata().model_version}.cbm")
        self._registry.register(meta)
        return meta