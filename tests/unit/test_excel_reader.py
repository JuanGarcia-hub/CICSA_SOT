import pandas as pd
import pytest

from sot_matcher.infrastructure import excel_reader as reader
from sot_matcher.infrastructure.excel_reader import (
    DateContentStrategy,
    DigitIdStrategy,
    IncidentPatternStrategy,
    InputValidationError,
    LongTextStrategy,
    SheetRole,
)


def test_detect_sheet_prefers_canonical_name(make_workbook):
    data = {
        "BITACORAS 2026": pd.DataFrame({"Marca temporal": ["2026-01-30"], "Actividad": ["a"]}),
        "BITACORAS 2025": pd.DataFrame({"Marca temporal": ["2025-01-30"], "SOT": ["1"]}),
        "RESUMEN": pd.DataFrame({"Total": [3]}),
    }
    source = make_workbook(data)
    assert reader.detect_sheet(source, SheetRole.BITACORA, preferred="BITACORAS 2026") == "BITACORAS 2026"


def test_detect_sheet_by_content_ignores_noise(make_workbook):
    data = {
        "Hoja1": pd.DataFrame({"90073467": [11223344, 55667788]}),
        "DATOS SOT": pd.DataFrame({"codsolot": ["83839980"], "observacionsot": ["trabajo 30/01/2026"]}),
    }
    source = make_workbook(data)
    assert reader.detect_sheet(source, SheetRole.SOT) == "DATOS SOT"


def test_detect_sheet_raises_when_none_fits(make_workbook):
    source = make_workbook({"Hoja1": pd.DataFrame({"90073467": [11223344, 55667788]})})
    with pytest.raises(InputValidationError) as exc:
        reader.detect_sheet(source, SheetRole.SOT)
    assert "Hoja1" in str(exc.value)


def test_date_content_strategy_picks_date_column():
    frame = pd.DataFrame({"CLAVE": ["2026-01-30", "2026-02-01"], "DETALLE": ["largo contenido operativo", "otra fila"]})
    assert DateContentStrategy().resolve(frame, False) == "CLAVE"


def test_date_content_strategy_ignores_numeric_columns():
    frame = pd.DataFrame({"ITEM": [1, 2, 3, 4], "FECHA": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]})
    assert DateContentStrategy().resolve(frame, False) == "FECHA"


def test_date_content_strategy_prefers_ejecucion_header():
    frame = pd.DataFrame(
        {
            "ITEM": [1, 2, 3, 4],
            "Fecha y Hora Asignación Cliente (DIA/MES/AÑO)": ["30/01/2026", "30/01/2026", "30/01/2026", "30/01/2026"],
            "Fecha y Hora Ejecucion (no borrar)": ["30/01/2026 05:15:00"] * 4,
        }
    )
    assert DateContentStrategy().resolve(frame, False) == "Fecha y Hora Ejecucion (no borrar)"


def test_long_text_strategy_picks_longest_detail():
    frame = pd.DataFrame({"ID": ["1", "2"], "DETALLE": ["un texto largo y descriptivo del trabajo", "otro texto amplio"]})
    assert LongTextStrategy(min_mean=5).resolve(frame, False) == "DETALLE"


def test_digit_id_strategy_picks_numeric_ids():
    frame = pd.DataFrame({"CLAVE": ["83839980", "83839981"], "OTRO": ["x", "y"]})
    assert DigitIdStrategy(min_digits=6).resolve(frame, False) == "CLAVE"


def test_incident_pattern_strategy_picks_inc_column():
    frame = pd.DataFrame({"CLAVE": ["INC00000036596", "INC00000036597"], "OTRO": ["a", "b"]})
    assert IncidentPatternStrategy().resolve(frame, False) == "CLAVE"


def test_classify_uploads_is_order_independent(synthetic_sot_bytes, synthetic_log_bytes):
    ordered = reader.classify_uploads([("sot.xlsx", synthetic_sot_bytes), ("log.xlsx", synthetic_log_bytes)])
    swapped = reader.classify_uploads([("log.xlsx", synthetic_log_bytes), ("sot.xlsx", synthetic_sot_bytes)])
    roles_ordered = {profile.name: profile.role for profile in ordered}
    roles_swapped = {profile.name: profile.role for profile in swapped}
    assert roles_ordered == roles_swapped
    assert set(roles_ordered.values()) == {"sot", "bitacora"}


def test_prepare_sots_with_column_date_fallback(make_workbook):
    frame = pd.DataFrame(
        {
            "codsolot": ["83839983"],
            "observacionsot": ["pendiente confirmar turno"],
            "descripcion": [""],
            "fechageneracionsot": ["22/02/2026 08:00:00"],
        }
    )
    source = make_workbook({"DATOS": frame})
    prepared = reader.prepare_sots(reader.read_excel(source, "DATOS"))
    assert prepared.loc[0, "sot"] == "83839983"
    assert prepared.loc[0, "date_hint"] == pd.Timestamp(2026, 2, 22).date()


def test_prepare_logs_canonical_with_unknown_headers(make_workbook):
    frame = pd.DataFrame(
        {
            "TS": ["2026-01-30 08:00", "2026-01-30 09:00"],
            "INC": ["INC00000036596", "83839981"],
            "AGUJA": ["largo texto de una bitacora completa de averia", "otro registro detallado de trabajo"],
            "LISTA MATERIALES": ["cable", ""],
        }
    )
    source = make_workbook({"DATOS": frame})
    logs = reader.prepare_logs(reader.read_excel(source, "DATOS"))
    assert list(logs.columns) == ["date", "log_text", "observation", "record_id", "materials_data"]
    assert len(logs) == 2
    assert logs.loc[0, "record_id"] == "INC00000036596"