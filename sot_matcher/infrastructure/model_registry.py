"""Registro de modelos basado en archivos + SQLite.

Implementa ModelRegistry protocol.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from sot_matcher.application.ports import ModelRegistry, ModelMetadata


class FileModelRegistry:
    def __init__(self, db_path: str = "sot_matcher_feedback.db", models_dir: str = "models"):
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self._init_table()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_table(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_version TEXT PRIMARY KEY,
                    artifact_path TEXT NOT NULL,
                    feature_schema_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    training_dataset_fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    parent_model_version TEXT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_models_status ON model_registry(status);
            """)

    def register(self, metadata: ModelMetadata) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_registry
                (model_version, artifact_path, feature_schema_version, created_at,
                 metrics_json, training_dataset_fingerprint, status, parent_model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.model_version,
                    metadata.artifact_path,
                    metadata.feature_schema_version,
                    metadata.created_at,
                    json.dumps(metadata.metrics),
                    metadata.training_dataset_fingerprint,
                    metadata.status,
                    metadata.parent_model_version,
                ),
            )

    def get_active_model(self) -> ModelMetadata | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM model_registry WHERE status = 'ACTIVE' LIMIT 1"
            ).fetchone()
        return self._row_to_meta(row) if row else None

    def activate(self, model_version: str, approval: str) -> None:
        if approval != "APPROVED":
            raise ValueError("Requiere approval='APPROVED'")
        with self._conn() as conn:
            # Desactivar actual
            conn.execute("UPDATE model_registry SET status = 'RETIRED' WHERE status = 'ACTIVE'")
            # Activar nuevo
            conn.execute(
                "UPDATE model_registry SET status = 'ACTIVE' WHERE model_version = ?",
                (model_version,),
            )

    def rollback(self, previous_version: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE model_registry SET status = 'RETIRED' WHERE status = 'ACTIVE'")
            conn.execute(
                "UPDATE model_registry SET status = 'ACTIVE' WHERE model_version = ?",
                (previous_version,),
            )

    def list_models(self, status: str | None = None) -> list[ModelMetadata]:
        sql = "SELECT * FROM model_registry"
        params = ()
        if status:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_meta(r) for r in rows]

    def _row_to_meta(self, row: sqlite3.Row) -> ModelMetadata:
        return ModelMetadata(
            model_version=row["model_version"],
            artifact_path=row["artifact_path"],
            feature_schema_version=row["feature_schema_version"],
            created_at=row["created_at"],
            metrics=json.loads(row["metrics_json"]),
            training_dataset_fingerprint=row["training_dataset_fingerprint"],
            status=row["status"],
            parent_model_version=row["parent_model_version"],
        )