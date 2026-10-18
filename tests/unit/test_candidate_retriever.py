"""Tests unitarios para CandidateRetriever."""

import pytest
from sot_matcher.infrastructure.candidate_retriever import SotCandidateRetriever


@pytest.fixture
def retriever():
    return SotCandidateRetriever(embedder=None)


@pytest.fixture
def sample_sot_record():
    return {
        "sot": "90232501",
        "source_text": "INC000123 mantenimiento zona norte 30/01/2026",
        "date_hint": "2026-01-30",
    }


@pytest.fixture
def sample_log_records():
    return [
        {
            "record_id": "LOG_001",
            "log_text": "Atención INC000123 mantenimiento zona norte",
            "observation": "Cable dañado",
            "materials_data": "cable FO",
            "date": "2026-01-30",
        },
        {
            "record_id": "LOG_002",
            "log_text": "Inspección zona sur",
            "observation": "",
            "materials_data": "",
            "date": "2026-01-30",
        },
        {
            "record_id": "LOG_003",
            "log_text": "Mantenimiento zona norte",
            "observation": "",
            "materials_data": "",
            "date": "2026-01-31",  # fecha distinta
        },
    ]


def test_retrieve_same_date(retriever, sample_sot_record, sample_log_records):
    candidates = retriever.retrieve(sample_sot_record, sample_log_records, limit=10)
    # Solo 2 bitácoras tienen fecha 2026-01-30
    assert len(candidates) == 2
    log_ids = {c.log_id for c in candidates}
    assert log_ids == {"LOG_001", "LOG_002"}


def test_retrieve_priority_exact_incidencia(retriever, sample_sot_record, sample_log_records):
    candidates = retriever.retrieve(sample_sot_record, sample_log_records, limit=10)
    # LOG_001 tiene INC000123 exacto -> debe ir primero
    assert candidates[0].log_id == "LOG_001"
    assert candidates[0].semantic_score is None  # sin embedder


def test_retrieve_limit(retriever, sample_sot_record, sample_log_records):
    candidates = retriever.retrieve(sample_sot_record, sample_log_records, limit=1)
    assert len(candidates) == 1


def test_retrieve_empty_logs(retriever, sample_sot_record):
    candidates = retriever.retrieve(sample_sot_record, [], limit=10)
    assert candidates == []


def test_retrieve_no_date_filter(retriever):
    """Si SOT no tiene date_hint, no filtra por fecha."""
    sot = {"sot": "90232501", "source_text": "trabajo genérico", "date_hint": None}
    logs = [
        {"record_id": "L1", "log_text": "trabajo", "observation": "", "materials_data": "", "date": "2026-01-01"},
        {"record_id": "L2", "log_text": "trabajo", "observation": "", "materials_data": "", "date": "2026-02-01"},
    ]
    candidates = retriever.retrieve(sot, logs, limit=10)
    assert len(candidates) == 2