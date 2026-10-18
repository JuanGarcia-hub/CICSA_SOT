"""Tests de integración para SqliteFeedbackRepository."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from sot_matcher.infrastructure.feedback_repository import SqliteFeedbackRepository, init_db
from sot_matcher.application.ports import FeedbackFilters
from sot_matcher.domain.feedback import FeedbackDecision, FeedbackEvent, LabeledPair


@pytest.fixture
def temp_db_path():
    """Proporciona una ruta de DB temporal sin usar TemporaryDirectory context manager."""
    import uuid
    tmp_dir = Path(tempfile.gettempdir())
    db_path = tmp_dir / f"test_feedback_{uuid.uuid4().hex}.db"
    yield str(db_path)
    # Cleanup manual
    try:
        db_path.unlink(missing_ok=True)
    except PermissionError:
        pass  # Windows puede bloquear el archivo temporalmente


@pytest.fixture
def repo(temp_db_path):
    return SqliteFeedbackRepository(temp_db_path)


def test_init_db_creates_tables(temp_db_path):
    init_db(Path(temp_db_path))
    # Verificar que las tablas existen
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    expected = {
        "feedback_events",
        "matching_runs",
        "candidate_predictions",
        "model_registry",
    }
    assert expected.issubset(tables)


def test_save_and_get_feedback(repo):
    event = FeedbackEvent(
        sot_id="SOT_001",
        log_id="LOG_001",
        decision=FeedbackDecision.CORRECT,
        reviewer_id="tester",
        reason_code="incidencia_coincide",
    )
    saved = repo.save_feedback(event)

    assert saved.feedback_id == event.feedback_id
    assert saved.sot_id == "SOT_001"
    assert saved.decision == FeedbackDecision.CORRECT

    retrieved = repo.get_feedback(saved.feedback_id)
    assert retrieved is not None
    assert retrieved.feedback_id == saved.feedback_id
    assert retrieved.sot_id == "SOT_001"
    assert retrieved.decision == FeedbackDecision.CORRECT


def test_get_nonexistent_feedback(repo):
    result = repo.get_feedback("nonexistent_id")
    assert result is None


def test_list_feedback_with_filters(repo):
    # Insertar varios eventos
    for i in range(5):
        repo.save_feedback(FeedbackEvent(
            sot_id=f"SOT_{i}",
            log_id=f"LOG_{i}",
            decision=FeedbackDecision.CORRECT if i % 2 == 0 else FeedbackDecision.INCORRECT,
            reviewer_id="tester" if i < 3 else "other",
        ))

    # Filtrar por decisión
    correct = repo.list_feedback(FeedbackFilters(decision=FeedbackDecision.CORRECT, limit=10))
    assert len(correct) == 3
    assert all(e.decision == FeedbackDecision.CORRECT for e in correct)

    # Filtrar por revisor
    tester = repo.list_feedback(FeedbackFilters(reviewer_id="tester", limit=10))
    assert len(tester) == 3
    assert all(e.reviewer_id == "tester" for e in tester)

    # Filtrar por SOT
    sot_0 = repo.list_feedback(FeedbackFilters(sot_id="SOT_0", limit=10))
    assert len(sot_0) == 1
    assert sot_0[0].sot_id == "SOT_0"

    # Paginación
    page1 = repo.list_feedback(FeedbackFilters(limit=2, offset=0))
    page2 = repo.list_feedback(FeedbackFilters(limit=2, offset=2))
    assert len(page1) == 2
    assert len(page2) == 2
    assert page1[0].feedback_id != page2[0].feedback_id


def test_get_feedback_for_pair(repo):
    sot_id = "SOT_PAIR"
    log_id = "LOG_PAIR"

    # Sin feedback previo
    assert repo.get_feedback_for_pair(sot_id, log_id) == []

    # Añadir varios eventos para el mismo par
    e1 = repo.save_feedback(FeedbackEvent(
        sot_id=sot_id, log_id=log_id,
        decision=FeedbackDecision.INCONCLUSIVE, reviewer_id="r1",
    ))
    e2 = repo.save_feedback(FeedbackEvent(
        sot_id=sot_id, log_id=log_id,
        decision=FeedbackDecision.CORRECT, reviewer_id="r2",
    ))

    history = repo.get_feedback_for_pair(sot_id, log_id)
    assert len(history) == 2
    assert history[0].decision == FeedbackDecision.INCONCLUSIVE
    assert history[1].decision == FeedbackDecision.CORRECT


def test_get_labeled_pairs_excludes_inconclusive(repo):
    # Insertar eventos de diferentes tipos
    repo.save_feedback(FeedbackEvent(sot_id="S1", log_id="L1", decision=FeedbackDecision.CORRECT, feature_schema_version="1.0"))
    repo.save_feedback(FeedbackEvent(sot_id="S2", log_id="L2", decision=FeedbackDecision.INCORRECT, feature_schema_version="1.0"))
    repo.save_feedback(FeedbackEvent(sot_id="S3", log_id="L3", decision=FeedbackDecision.INCONCLUSIVE, feature_schema_version="1.0"))
    repo.save_feedback(FeedbackEvent(sot_id="S4", log_id="L4", decision=FeedbackDecision.CORRECT, feature_schema_version="2.0"))  # distinta versión

    pairs = repo.get_labeled_pairs("1.0")
    assert len(pairs) == 2  # solo CORRECT e INCORRECT de v1.0
    labels = {p.label for p in pairs}
    assert labels == {0, 1}
    assert all(p.feature_schema_version == "1.0" for p in pairs)


def test_get_labeled_pairs_resolves_supersedes(repo):
    # Evento original
    orig = repo.save_feedback(FeedbackEvent(
        sot_id="S1", log_id="L1",
        decision=FeedbackDecision.CORRECT,
        feature_schema_version="1.0",
    ))
    # Corrección que referencia al original
    corr = repo.save_feedback(FeedbackEvent(
        sot_id="S1", log_id="L1",
        decision=FeedbackDecision.INCORRECT,
        supersedes_feedback_id=orig.feedback_id,
        feature_schema_version="1.0",
    ))

    pairs = repo.get_labeled_pairs("1.0")
    # Solo debe quedar el último (la corrección), no el original
    sot1_pairs = [p for p in pairs if p.sot_id == "S1" and p.log_id == "L1"]
    assert len(sot1_pairs) == 1
    assert sot1_pairs[0].label == 0  # INCORRECT (la corrección)


def test_feedback_event_foreign_key_constraint(repo):
    # Intentar guardar evento con supersedes_feedback_id inexistente
    # SQLite con FK activado debería rechazarlo
    event = FeedbackEvent(
        sot_id="S1", log_id="L1",
        decision=FeedbackDecision.CORRECT,
        supersedes_feedback_id="nonexistent_id",
    )
    with pytest.raises(sqlite3.IntegrityError):
        repo.save_feedback(event)


def test_decision_check_constraint(temp_db_path):
    # Intentar guardar decisión inválida directamente en SQL
    init_db(Path(temp_db_path))
    conn = sqlite3.connect(temp_db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute(
            "INSERT INTO feedback_events (feedback_id, sot_id, log_id, decision, created_at) VALUES (?, ?, ?, ?, ?)",
            ("bad_1", "S1", "L1", "INVALID", "2024-01-01T00:00:00"),
        )
        conn.commit()
        pytest.fail("Debería haber lanzado IntegrityError")
    except sqlite3.IntegrityError:
        pass  # esperado
    finally:
        conn.close()