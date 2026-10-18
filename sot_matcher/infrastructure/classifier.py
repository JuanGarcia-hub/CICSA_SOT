"""Clasificador CatBoost para matching supervisado.

Implementa MatchingClassifier protocol.
"""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from catboost import CatBoostClassifier, Pool

from sot_matcher.application.ports import (
    MatchingClassifier,
    TrainingReport,
    ModelMetadata,
)
from sot_matcher.domain.feedback import LabeledPair as DomainLabeledPair


class CatBoostMatchingClassifier:
    """Wrapper CatBoost con metadatos y persistencia."""

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 42,
        verbose: bool = False,
        **kwargs,
    ):
        self._config = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "random_seed": random_seed,
            "verbose": verbose,
            **kwargs,
        }
        self._model = CatBoostClassifier(**self._config)
        self._feature_names: list[str] | None = None
        self._model_version: str | None = None
        self._feature_schema_version: str | None = None
        self._training_fingerprint: str | None = None
        self._metrics: dict[str, float] = {}

    def fit(
        self,
        training_data: Iterable[DomainLabeledPair],
        validation_data: Iterable[DomainLabeledPair],
    ) -> TrainingReport:
        train_list = list(training_data)
        val_list = list(validation_data)

        if not train_list or not val_list:
            raise ValueError("Entrenamiento y validación requieren datos")

        # Verificar ambas clases
        train_labels = [p.label for p in train_list]
        if len(set(train_labels)) < 2:
            raise ValueError("Se requieren ambas clases (positivos y negativos)")

        # Preparar matrices
        self._feature_names = list(train_list[0].features.keys())
        X_train = np.array([[p.features.get(f, 0.0) for f in self._feature_names] for p in train_list])
        y_train = np.array(train_labels)
        X_val = np.array([[p.features.get(f, 0.0) for f in self._feature_names] for p in val_list])
        y_val = np.array([p.label for p in val_list])

        train_pool = Pool(X_train, label=y_train, feature_names=self._feature_names)
        val_pool = Pool(X_val, label=y_val, feature_names=self._feature_names)

        self._model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        # Métricas
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        )
        val_pred = self._model.predict(X_val)
        val_proba = self._model.predict_proba(X_val)[:, 1]

        self._metrics = {
            "accuracy": float(accuracy_score(y_val, val_pred)),
            "precision": float(precision_score(y_val, val_pred, zero_division=0)),
            "recall": float(recall_score(y_val, val_pred, zero_division=0)),
            "f1": float(f1_score(y_val, val_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, val_proba)),
        }

        # Versionado
        self._model_version = f"catboost_{int(time.time())}"
        self._feature_schema_version = train_list[0].feature_schema_version
        self._training_fingerprint = self._compute_fingerprint(train_list)

        return TrainingReport(
            model_version=self._model_version,
            feature_schema_version=self._feature_schema_version,
            train_size=len(train_list),
            val_size=len(val_list),
            test_size=0,
            metrics=self._metrics,
            config=self._config,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def predict_proba(self, feature_vectors: list[dict[str, Any]]) -> list[float]:
        if self._model is None or self._feature_names is None:
            raise RuntimeError("Modelo no entrenado ni cargado")
        X = np.array([[v.get(f, 0.0) for f in self._feature_names] for v in feature_vectors])
        return self._model.predict_proba(X)[:, 1].tolist()

    def save(self, artifact_path: str) -> ModelMetadata:
        Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(artifact_path)
        meta = ModelMetadata(
            model_version=self._model_version or "unknown",
            artifact_path=artifact_path,
            feature_schema_version=self._feature_schema_version or "unknown",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            metrics=self._metrics,
            training_dataset_fingerprint=self._training_fingerprint or "",
            status="CANDIDATE",
            parent_model_version=None,
        )
        # Guardar metadatos extendidos junto al modelo (incluye feature_names)
        meta_path = Path(artifact_path).with_suffix(".meta.json")
        extended_meta = meta.__dict__.copy()
        extended_meta["feature_names"] = self._feature_names
        extended_meta["config"] = self._config
        meta_path.write_text(json.dumps(extended_meta, indent=2))
        return meta

    @classmethod
    def load(cls, artifact_path: str) -> "CatBoostMatchingClassifier":
        model = CatBoostClassifier()
        model.load_model(artifact_path)
        meta_path = Path(artifact_path).with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        else:
            meta = {}
        instance = cls()
        instance._model = model
        instance._feature_names = meta.get("feature_names")
        instance._model_version = meta.get("model_version")
        instance._feature_schema_version = meta.get("feature_schema_version")
        instance._training_fingerprint = meta.get("training_dataset_fingerprint")
        instance._metrics = meta.get("metrics", {})
        instance._config = meta.get("config", {})
        return instance

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_version=self._model_version or "unknown",
            artifact_path="",
            feature_schema_version=self._feature_schema_version or "unknown",
            created_at="",
            metrics=self._metrics,
            training_dataset_fingerprint=self._training_fingerprint or "",
            status="CANDIDATE",
            parent_model_version=None,
        )

    def _compute_fingerprint(self, data: list[DomainLabeledPair]) -> str:
        hasher = hashlib.sha256()
        for p in sorted(data, key=lambda x: (x.sot_id, x.log_id)):
            hasher.update(f"{p.sot_id}|{p.log_id}|{p.label}".encode())
        return hasher.hexdigest()[:16]