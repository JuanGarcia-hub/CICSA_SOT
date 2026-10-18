"""Fixtures compartidos: construcción de libros en memoria y embedder de prueba."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL_SOT_PATH = Path(r"C:\Users\USER\Downloads\SOT 23-09-2026.xlsx")
REAL_LOG_PATH = Path(r"C:\Users\USER\Downloads\R 0001 REPORTE DE ACTIVIDADES DIARIAS 2026 (3).xlsx")


class FakeEmbedder:
    """Embedder determinista: texto con 'talara' => [1, 0]; resto => [0, 1]."""

    def encode(self, sentences: list[str], **_kwargs: object) -> np.ndarray:
        vectors = [[1.0, 0.0] if "talara" in sentence else [0.0, 1.0] for sentence in sentences]
        return np.asarray(vectors, dtype=np.float64)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def make_workbook():
    def _make(sheets: dict[str, pd.DataFrame]) -> bytes:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)
        buffer.seek(0)
        return buffer.getvalue()

    return _make


@pytest.fixture
def synthetic_sot_bytes(make_workbook) -> bytes:
    sots = pd.DataFrame(
        {
            "codsolot": ["83839980", "83839981", "83839982", "83839983"],
            "observacionsot": [
                "Atención INC000123 30/01/2026",
                "recorrido preventivo talara 30/01/2026",
                "intervencion sin incidencia 12/03/2026",
                "cambio de equipo programado",
            ],
            "descripcion": ["aviso clasico", "revision de ruta", "mantenimiento menor", "trabajo nocturno"],
            "fecha_instalacion": [None, None, None, None],
            "fechageneracionsot": ["30/01/2026 10:00:00", "30/01/2026 10:05:00", "12/03/2026 08:00:00", "22/02/2026 08:00:00"],
        }
    )
    return make_workbook({"SOT 23-09-2026": sots})


@pytest.fixture
def synthetic_log_bytes(make_workbook) -> bytes:
    logs = pd.DataFrame(
        {
            "Marca temporal": ["2026-01-30 08:00", "2026-01-30 09:00", "2026-01-31 08:00", "2026-02-22 10:00", "2026-03-12 07:30"],
            "SOT / INC ": ["INC00000036596", "83839981", "83999999", "INC000345", "83839982"],
            "Fecha y Hora Inicio": ["2026-01-30 08:00:00", "2026-01-30 09:00:00", "2026-01-31 08:00:00", "2026-02-22 10:00:00", "2026-03-12 07:30:00"],
            "Adjuntar bitácora completa": [
                "Atencion INC00000036596 con cable FO",
                "recorrido talara por la zona",
                "patrullaje talara",
                "cambio de equipo en el distrito",
                "trabajo normal con ferreteria",
            ],
            "Observacion": ["", "sin observacion", "", "", ""],
            "LISTA DE MATERIALES DE AVERÍA": ["cable", "", "", "", "ferreteria"],
        }
    )
    hoja_basura = pd.DataFrame({"90073467": [11223344, 55667788, 99001122]})
    return make_workbook({"REPORTE ACTIVIDADES 2026": logs, "Hoja1": hoja_basura})