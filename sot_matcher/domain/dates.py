"""Extracción de fechas desde texto operacional y columnas explícitas."""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import date

SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5,
    "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9,
    "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
}


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFD", str(value if value is not None else "").lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def _valid_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_date(value: object) -> date | None:
    """Extrae la primera fecha válida en formatos numéricos o meses en español."""
    text = _normalize(value)
    numeric = re.search(r"\b(\d{1,2})[ /-](\d{1,2})[ /-](\d{4})\b", text)
    if numeric:
        return _valid_date(*map(int, numeric.groups()[::-1]))

    iso = re.search(r"\b(\d{4})[ /-](\d{1,2})[ /-](\d{1,2})\b", text)
    if iso:
        return _valid_date(*map(int, iso.groups()))

    written = re.search(r"\b(\d{1,2})\s+(?:de\s+)?([a-z]+)\s+(?:de\s+)?(\d{4})\b", text)
    if written and written.group(2) in SPANISH_MONTHS:
        return _valid_date(int(written.group(3)), SPANISH_MONTHS[written.group(2)], int(written.group(1)))
    return None


def resolve_date(text: object, fallback: object = None) -> date | None:
    """Resuelve la fecha objetivo de una SOT.

    Strategy de resolución: primero fecha explícita en el texto descriptivo;
    si no existe, usa la ``fallback`` (hint calculado desde columnas de fecha
    como ``fecha_instalacion``) provista por la capa de infraestructura.
    """
    text_date = extract_date(text)
    if text_date is not None:
        return text_date
    if fallback is None:
        return None
    if isinstance(fallback, float) and math.isnan(fallback):  # NaN sin acoplar a pandas
        return None
    return fallback