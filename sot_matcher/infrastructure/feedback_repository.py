"""Persistencia SQLite para eventos de feedback.

Implementa FeedbackRepository protocol.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sot_matcher.application.ports import FeedbackRepository, FeedbackFilters
from sot_matcher.domain.feedback import (
    FeedbackEvent,
    FeedbackDecision,
    LabeledPair,
)

SCHEMA_VERSION = "1.0"
DEFAULT_DB_PATH = Path("sot_matcher_feedback.db")
_LOCK = threading.Lock()


def _get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    """Crea tablas e índices si no existen (idempotente)."""
    with _get_conn(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS feedback_events (
            feedback_id TEXT PRIMARY KEY,
            run_id TEXT NULL,
            sot_id TEXT NOT NULL,
            log_id TEXT NOT NULL,
            decision TEXT NOT NULL CHECK (decision IN ('CORRECT','INCORRECT','INCONCLUSIVE')),
            reason_code TEXT NULL,
            comment TEXT NULL,
            reviewer_id TEXT NULL,
            created_at TEXT NOT NULL,
            supersedes_feedback_id TEXT NULL REFERENCES feedback_events(feedback_id),
            feature_schema_version TEXT NULL
        );

        CREATE TABLE IF NOT EXISTS matching_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            engine_version TEXT NOT NULL,
            settings_json TEXT NOT NULL,
            input_fingerprint TEXT NULL,
            status TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS candidate_predictions (
            prediction_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES matching_runs(run_id),
            sot_id TEXT NOT NULL,
            log_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            semantic_score REAL NULL,
            classifier_probability REAL NULL,
            model_version TEXT NULL,
            decision TEXT NOT NULL,
            evidence_json TEXT NOT NULL
        );

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

        CREATE INDEX IF NOT EXISTS idx_feedback_sot_log ON feedback_events(sot_id, log_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_decision_date ON feedback_events(decision, created_at);
        CREATE INDEX IF NOT EXISTS idx_predictions_run_sot_rank ON candidate_predictions(run_id, sot_id, rank);
        CREATE INDEX IF NOT EXISTS idx_models_status ON model_registry(status);
        """)


@contextmanager
def _conn(db_path: Path):
    with _LOCK:
        conn = _get_conn(db_path)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class SqliteFeedbackRepository:
    """Implementación SQLite de FeedbackRepository."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        init_db(self.db_path)

    def save_feedback(self, event: FeedbackEvent) -> FeedbackEvent:
        with _conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO feedback_events (
                    feedback_id, run_id, sot_id, log_id, decision,
                    reason_code, comment, reviewer_id, created_at,
                    supersedes_feedback_id, feature_schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.feedback_id,
                    event.run_id,
                    event.sot_id,
                    event.log_id,
                    event.decision.value,
                    event.reason_code,
                    event.comment,
                    event.reviewer_id,
                    event.created_at.isoformat(),
                    event.supersedes_feedback_id,
                    event.feature_schema_version,
                ),
            )
        return event

    def get_feedback(self, feedback_id: str) -> FeedbackEvent | None:
        with _conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM feedback_events WHERE feedback_id = ?", (feedback_id,)
            ).fetchone()
        return self._row_to_event(row) if row else None

    def list_feedback(self, filters: FeedbackFilters) -> list[FeedbackEvent]:
        where = []
        params: list[Any] = []
        if filters.decision:
            where.append("decision = ?")
            params.append(filters.decision.value)
        if filters.reviewer_id:
            where.append("reviewer_id = ?")
            params.append(filters.reviewer_id)
        if filters.date_from:
            where.append("created_at >= ?")
            params.append(filters.date_from)
        if filters.date_to:
            where.append("created_at <= ?")
            params.append(filters.date_to)
        if filters.sot_id:
            where.append("sot_id = ?")
            params.append(filters.sot_id)
        if filters.run_id:
            where.append("run_id = ?")
            params.append(filters.run_id)

        sql = "SELECT * FROM feedback_events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([filters.limit, filters.offset])

        with _conn(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_labeled_pairs(self, dataset_version: str) -> list[LabeledPair]:
        """Construye pares etiquetados resolviendo conflictos y superseded.

        Regla: para cada par (sot_id, log_id), se toma el evento más reciente
        que NO haya sido superseded por otro evento.
        """
        with _conn(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT * FROM feedback_events
                WHERE decision IN ('CORRECT', 'INCORRECT')
                AND feature_schema_version = ?
                ORDER BY sot_id, log_id, created_at
                """,
                (dataset_version,),
            ).fetchall()

        # Primero, construir set de IDs que han sido superseded
        superseded_ids: set[str] = set()
        events_by_key: dict[tuple[str, str], list[FeedbackEvent]] = {}

        for row in rows:
            event = self._row_to_event(row)
            key = (event.sot_id, event.log_id)
            events_by_key.setdefault(key, []).append(event)
            if event.supersedes_feedback_id:
                superseded_ids.add(event.supersedes_feedback_id)

        # Para cada par, tomar el último evento que no esté en superseded_ids
        latest_by_pair: dict[tuple[str, str], FeedbackEvent] = {}
        for key, events in events_by_key.items():
            # Iterar en orden cronológico inverso (más reciente primero)
            for event in reversed(events):
                if event.feedback_id not in superseded_ids:
                    latest_by_pair[key] = event
                    break

        return [
            LabeledPair(
                sot_id=event.sot_id,
                log_id=event.log_id,
                label=event.label(),
                features={},  # se llenará en TrainingPipeline
                feature_schema_version=event.feature_schema_version or SCHEMA_VERSION,
                feedback_ids=(event.feedback_id,),
            )
            for event in latest_by_pair.values()
        ]

    def get_feedback_for_pair(self, sot_id: str, log_id: str) -> list[FeedbackEvent]:
        with _conn(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM feedback_events WHERE sot_id = ? AND log_id = ? ORDER BY created_at",
                (sot_id, log_id),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> FeedbackEvent:
        return FeedbackEvent(
            feedback_id=row["feedback_id"],
            run_id=row["run_id"],
            sot_id=row["sot_id"],
            log_id=row["log_id"],
            decision=FeedbackDecision(row["decision"]),
            reason_code=row["reason_code"],
            comment=row["comment"],
            reviewer_id=row["reviewer_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            supersedes_feedback_id=row["supersedes_feedback_id"],
            feature_schema_version=row["feature_schema_version"],
        )