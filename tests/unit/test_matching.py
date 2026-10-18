from datetime import date

import pandas as pd

from sot_matcher.domain.matching import MatchingEngine, MatchingSettings


def test_exact_id_match_takes_priority_and_assigns_materials(fake_embedder):
    sots = pd.DataFrame([{"sot": "83839980", "source_text": "INC000123 trabajo programado para el dia 30/01/2026", "date_hint": None}])
    logs = pd.DataFrame(
        [{"date": date(2026, 1, 30), "log_text": "Atención INC000123 con cable", "observation": "", "record_id": "", "materials_data": "cable FO"}]
    )
    result = MatchingEngine(embedder=fake_embedder).match(sots, logs)
    assert result.loc[0, "estado"] == "Exacto"
    assert result.loc[0, "materias"] == "si"


def test_semantic_match_requires_exact_same_date(fake_embedder):
    sots = pd.DataFrame([{"sot": "1", "source_text": "recorrido preventivo talara 30/01/2026", "date_hint": None}])
    logs = pd.DataFrame(
        [
            {"date": date(2026, 1, 30), "log_text": "recorrido talara", "observation": "", "record_id": "", "materials_data": ""},
            {"date": date(2026, 1, 31), "log_text": "recorrido talara", "observation": "", "record_id": "", "materials_data": ""},
        ]
    )
    result = MatchingEngine(MatchingSettings(threshold=0.78), fake_embedder).match(sots, logs)
    assert result.loc[0, "estado"] == "Semántico"
    assert result.loc[0, "bitacoras"] == "recorrido talara"


def test_ambiguous_semantic_matches_remain_pending(fake_embedder):
    sots = pd.DataFrame([{"sot": "1", "source_text": "recorrido talara 30/01/2026", "date_hint": None}])
    logs = pd.DataFrame(
        [
            {"date": date(2026, 1, 30), "log_text": "recorrido talara", "observation": "", "record_id": "", "materials_data": ""},
            {"date": date(2026, 1, 30), "log_text": "patrullaje talara", "observation": "", "record_id": "", "materials_data": ""},
        ]
    )
    result = MatchingEngine(embedder=fake_embedder).match(sots, logs)
    assert result.loc[0, "bitacoras"] == "PENDIENTE"


def test_column_date_hint_assigns_without_text_date(fake_embedder):
    sots = pd.DataFrame(
        [{"sot": "7", "source_text": "trabajo nocturno", "date_hint": date(2026, 2, 22)}]
    )
    logs = pd.DataFrame(
        [{"date": date(2026, 2, 22), "log_text": "cambio de equipo nocturno", "observation": "", "record_id": "", "materials_data": ""}]
    )
    result = MatchingEngine(embedder=fake_embedder).match(sots, logs)
    assert result.loc[0, "estado"] == "Semántico"
    assert result.loc[0, "sot"] == "7"


def test_no_date_anywhere_remains_pending(fake_embedder):
    sots = pd.DataFrame([{"sot": "9", "source_text": "sin fecha visible", "date_hint": None}])
    logs = pd.DataFrame(
        [{"date": date(2026, 1, 30), "log_text": "cualquier bitacora", "observation": "", "record_id": "", "materials_data": ""}]
    )
    result = MatchingEngine(embedder=fake_embedder).match(sots, logs)
    assert result.loc[0, "bitacoras"] == "PENDIENTE"


def test_exact_evidence_in_other_date_assigns_nearest(fake_embedder):
    sots = pd.DataFrame(
        [{"sot": "83839980", "source_text": "INC000123 instalacion 30/01/2026", "date_hint": None}]
    )
    logs = pd.DataFrame(
        [{"date": date(2026, 2, 3), "log_text": "perdida INC000123", "observation": "", "record_id": "", "materials_data": ""}]
    )
    result = MatchingEngine(embedder=fake_embedder).match(sots, logs)
    assert result.loc[0, "estado"] == "Exacto"
    assert result.loc[0, "bitacoras"] == "perdida INC000123"


def test_exact_evidence_in_different_dates_picks_closest(fake_embedder):
    sots = pd.DataFrame(
        [{"sot": "83839980", "source_text": "INC000123 instalacion 30/01/2026", "date_hint": None}]
    )
    logs = pd.DataFrame(
        [
            {"date": date(2026, 2, 15), "log_text": "perdida INC000123 v1", "observation": "", "record_id": "", "materials_data": ""},
            {"date": date(2026, 2, 1), "log_text": "perdida INC000123 v2", "observation": "", "record_id": "", "materials_data": ""},
        ]
    )
    result = MatchingEngine(embedder=fake_embedder).match(sots, logs)
    assert result.loc[0, "estado"] == "Exacto"
    assert result.loc[0, "bitacoras"] == "perdida INC000123 v2"


def test_exact_evidence_equidistant_remains_pending(fake_embedder):
    sots = pd.DataFrame(
        [{"sot": "83839980", "source_text": "INC000123 instalacion 30/01/2026", "date_hint": None}]
    )
    logs = pd.DataFrame(
        [
            {"date": date(2026, 2, 1), "log_text": "perdida INC000123 v1", "observation": "", "record_id": "", "materials_data": ""},
            {"date": date(2026, 2, 1), "log_text": "perdida INC000123 v2", "observation": "", "record_id": "", "materials_data": ""},
        ]
    )
    result = MatchingEngine(embedder=fake_embedder).match(sots, logs)
    assert result.loc[0, "bitacoras"] == "PENDIENTE"