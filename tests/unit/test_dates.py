import math
from datetime import date

from sot_matcher.domain.dates import extract_date, resolve_date


def test_extracts_numeric_dates():
    assert extract_date("trabajo para el día 23/01/2026") == date(2026, 1, 23)
    assert extract_date("entrega 2026-05-12") == date(2026, 5, 12)


def test_extracts_written_spanish_months():
    assert extract_date("Fecha: 17 de enero de 2026") == date(2026, 1, 17)
    assert extract_date("revisión el 3 setiembre 2026") == date(2026, 9, 3)


def test_invalid_dates_are_ignored():
    assert extract_date("fecha 31/02/2026") is None
    assert extract_date("sin referencia temporal") is None


def test_resolve_prefers_text_over_fallback():
    assert resolve_date("entro el 10/02/2026", fallback=date(2026, 1, 1)) == date(2026, 2, 10)


def test_resolve_uses_fallback_when_text_has_no_date():
    assert resolve_date("pendiente confirmar turno", fallback=date(2026, 2, 22)) == date(2026, 2, 22)


def test_resolve_ignores_nan_fallback():
    assert resolve_date("sin fecha", math.nan) is None
    assert resolve_date("sin fecha") is None