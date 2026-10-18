"""Pruebas funcionales contra los archivos reales del usuario.

Sirven como red de regresión de los conteos alcanzados con el pipeline
actual (max_seq_length=64, batch=64, pre-filtro léxico, caché de embeddings).
Si alguno de los archivos reales no existe, se omiten automáticamente.
"""

from __future__ import annotations

import pytest

from sot_matcher import CrossingPipeline, MatchingSettings
from sot_matcher.infrastructure.embedding import get_embedder
from tests.conftest import REAL_LOG_PATH, REAL_SOT_PATH, FakeEmbedder

pytestmark = pytest.mark.skipif(
    not (REAL_SOT_PATH.exists() and REAL_LOG_PATH.exists()),
    reason="Se requieren los archivos reales de SOT y bitácoras",
)


@pytest.fixture(scope="module")
def pipeline() -> CrossingPipeline:
    return CrossingPipeline()


@pytest.fixture(scope="module")
def prepared(pipeline: CrossingPipeline):
    return pipeline.prepare(str(REAL_SOT_PATH), str(REAL_LOG_PATH))


def test_real_prepare_detects_sheets_and_volumes(prepared) -> None:
    assert prepared.sot_sheet == "SOT 23-09-2026"
    assert prepared.log_sheet == "BITACORAS 2026"
    assert len(prepared.sots) == 13155
    assert len(prepared.logs) == 5312


def test_real_every_sot_has_date_hint(prepared) -> None:
    assert prepared.sots.date_hint.notna().all()


@pytest.mark.smoke
def test_real_fake_cross_anchors_are_stable(prepared) -> None:
    """Cruce con embedder falso (rápido, determinista): 1535 exactos / 14 semánticos."""
    results = CrossingPipeline(embedder=FakeEmbedder()).cross(
        prepared, MatchingSettings(threshold=0.78)
    ).results
    assert (results["estado"] == "Exacto").sum() == 1535
    assert (results["estado"] == "Semántico").sum() == 14
    assert (results["estado"] == "Pendiente").sum() == 11606


@pytest.mark.smoke
def test_real_model_cross_reaches_anchors(prepared) -> None:
    """Cruce con el modelo real: 1535 exactos / 142 semánticos, con evidencia exacta
    aun cuando la bitácora quedó registrada en otra fecha (desempate por cercanía).

    La primera ejecución de este test dura ~2 min (carga del modelo); las
    siguientes del mismo proceso son casi instantáneas gracias a la caché.
    """
    results = CrossingPipeline(embedder=get_embedder()).cross(
        prepared, MatchingSettings(threshold=0.78)
    ).results
    assert (results["estado"] == "Exacto").sum() == 1535
    assert (results["estado"] == "Semántico").sum() == 142
    assert (results["estado"] == "Pendiente").sum() == 11478
    assert (results["materias"] == "si").sum() == 1095