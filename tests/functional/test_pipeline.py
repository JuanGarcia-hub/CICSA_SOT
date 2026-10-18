"""Pruebas funcionales end-to-end sobre libros sintéticos en memoria."""

import io

import pandas as pd

from sot_matcher import (
    CrossingPipeline,
    InputValidationError,
    MatchingSettings,
    to_excel,
)
from sot_matcher.infrastructure.excel_export import OUTPUT_COLUMNS

DEFAULT_PIPELINE = CrossingPipeline()


def _run_results(synthetic_sot_bytes, synthetic_log_bytes, fake_embedder) -> pd.DataFrame:
    pipeline = CrossingPipeline(embedder=fake_embedder)
    prepared = pipeline.prepare(synthetic_sot_bytes, synthetic_log_bytes)
    return pipeline.cross(prepared, MatchingSettings(threshold=0.78)).results


def test_end_to_end_synthetic_assignment(synthetic_sot_bytes, synthetic_log_bytes, fake_embedder):
    results = _run_results(synthetic_sot_bytes, synthetic_log_bytes, fake_embedder)
    assert len(results) == 4
    assert int((results["estado"] != "Pendiente").sum()) == 4
    assert int((results["estado"] == "Exacto").sum()) == 2
    assert results.loc[results["sot"] == "83839981", "estado"].iloc[0] == "Exacto"
    assert not (results["bitacoras"] == "PENDIENTE").any()


def test_end_to_end_detects_sheets_and_ignores_noise(synthetic_sot_bytes, synthetic_log_bytes):
    prepared = DEFAULT_PIPELINE.prepare(synthetic_sot_bytes, synthetic_log_bytes)
    assert prepared.sot_sheet == "SOT 23-09-2026"
    assert prepared.log_sheet == "REPORTE ACTIVIDADES 2026"
    assert len(prepared.sots) == 4
    assert len(prepared.logs) == 5


def test_export_has_contractual_columns(synthetic_sot_bytes, synthetic_log_bytes, fake_embedder):
    results = _run_results(synthetic_sot_bytes, synthetic_log_bytes, fake_embedder)
    exported = to_excel(results)
    assert exported.startswith(b"PK")  # libro xlsx válido
    restored = pd.read_excel(io.BytesIO(exported), engine="calamine", sheet_name="RESULTADO")
    assert list(restored.columns) == OUTPUT_COLUMNS


def test_run_facade_combines_prepare_and_cross(synthetic_sot_bytes, synthetic_log_bytes, fake_embedder):
    outcome = CrossingPipeline(embedder=fake_embedder).run(
        synthetic_sot_bytes, synthetic_log_bytes, settings=MatchingSettings()
    )
    assert len(outcome.results) == 4


def test_classify_roles_mapping(synthetic_sot_bytes, synthetic_log_bytes):
    profiles = DEFAULT_PIPELINE.classify([("x.xlsx", synthetic_sot_bytes), ("y.xlsx", synthetic_log_bytes)])
    mapping = {profile.name: profile.role for profile in profiles}
    assert mapping["x.xlsx"] == "sot"
    assert mapping["y.xlsx"] == "bitacora"


def test_invalid_workbook_raises_friendly_error(make_workbook):
    junk = make_workbook({"Hoja1": pd.DataFrame({"90073467": [1, 2, 3]})})
    try:
        DEFAULT_PIPELINE.prepare(junk, junk)
        raise AssertionError("debería lanzar InputValidationError")
    except InputValidationError as error:
        assert "Hoja1" in str(error)