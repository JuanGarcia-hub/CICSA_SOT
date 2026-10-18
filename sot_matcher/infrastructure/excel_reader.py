"""Ingesta dinámica de libros Excel: detección de hojas y columnas por contenido.

Estrategias (Chain of Responsibility): se intenta primero el nombre predefinido /
alias estándar y, si no aparece, se infiere por el contenido de la hoja
(fechas parseables, IDs numéricos largos, texto de bitácora) para que la carga
funcione sin depender de los nombres exactos.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from typing import BinaryIO

import pandas as pd
from pandas import Timestamp

from ..domain.text import normalize_text

try:  # Motor de lectura rápido (Rust). Opcional; si falta se usa openpyxl.
    import python_calamine  # noqa: F401

    _HAS_CALAMINE = True
except ImportError:  # pragma: no cover - no debería ocurrir con requirements actuales
    _HAS_CALAMINE = False

SOT_SHEET = "PENDIENTE REGISTRO"
LOG_SHEET = "BITACORAS 2026"

SOT_ID_ALIASES = ("codsolot", "sot")
SOT_TEXT_ALIASES = ("observacionsot", "descripcion")
SOT_NAME_KEYWORDS = ("sot", "pendiente", "registro", "agenda", "orden")

LOG_DATE_ALIASES = ("fecha y hora inicio", "marca temporal", "fecha")
LOG_TEXT_ALIASES = ("adjuntar bitacora completa", "bitacora", "comentario", "trabajo")
LOG_OBS_ALIASES = ("observacion",)
LOG_ID_ALIASES = ("sot inc", "sot / inc", "sot", "incidencia")
LOG_NAME_KEYWORDS = ("bitacora", "actividad", "reporte", "trabajo")
PENALIZED_NAME_KEYWORDS = ("resumen", "cuadre", "hoja1", "hoja2")

MATERIAL_COLUMN_HINTS = ("material", "cable", "ferreteria", "reutilizado", "lista de materiales")
DATE_FALLBACK_COLUMNS = ("fecha_instalacion", "fecha_compromiso", "fechageneracionsot", "fechasolucion", "fecregistro")

MIN_SHEET_SCORE = 1.0


class InputValidationError(ValueError):
    """Indica una hoja o columna indispensable ausente o no reconocible."""


class SheetRole(StrEnum):
    SOT = "sot"
    BITACORA = "bitacora"


@dataclass
class FileProfile:
    """Perfil detectado de un archivo subido (permite corrección manual en la UI)."""

    name: str
    source: object
    meta: dict[str, pd.DataFrame]
    sot_sheet: str | None
    sot_score: float
    log_sheet: str | None
    log_score: float
    role: SheetRole | None = None
    sheet: str | None = None


# --------------------------------------------------------------------------
# Lectura
# --------------------------------------------------------------------------

def read_excel(source: BinaryIO | bytes | str, sheet_name: str) -> pd.DataFrame:
    handle = _as_handle(source)
    if _HAS_CALAMINE:
        return pd.read_excel(handle, sheet_name=sheet_name, engine="calamine")
    return pd.read_excel(handle, sheet_name=sheet_name)


def read_workbook_meta(source: BinaryIO | bytes | str) -> dict[str, pd.DataFrame]:
    """Devuelve cada hoja como DataFrame con encabezado y hasta 5 filas de muestra."""
    handle = _as_handle(source)
    if _HAS_CALAMINE:
        return pd.read_excel(handle, sheet_name=None, nrows=5, engine="calamine")
    return pd.read_excel(handle, sheet_name=None, nrows=5)


def _as_handle(source: BinaryIO | bytes | str) -> BinaryIO:
    if isinstance(source, bytes):
        return BytesIO(source)
    if isinstance(source, str):
        return open(source, "rb")
    source.seek(0)
    return source


# --------------------------------------------------------------------------
# Detección de hojas
# --------------------------------------------------------------------------

def _headerset(frame: pd.DataFrame) -> set[str]:
    return {normalize_text(column) for column in frame.columns}


def score_sheet(frame: pd.DataFrame, sheet_name: str, role: SheetRole) -> float:
    """Puntúa cuán compatible es una hoja con un rol."""
    headers = _headerset(frame)
    normalized_name = normalize_text(sheet_name)
    score = 0.0

    if role is SheetRole.SOT:
        for alias in tuple(SOT_ID_ALIASES) + tuple(SOT_TEXT_ALIASES):
            if normalize_text(alias) in headers:
                score += 1.5
        if any(k in normalized_name for k in SOT_NAME_KEYWORDS):
            score += 1.0
        if _has_long_digit_column(frame, min_digits=8):
            score += 1.0
    else:
        for alias in tuple(LOG_DATE_ALIASES) + tuple(LOG_TEXT_ALIASES) + tuple(LOG_ID_ALIASES):
            if normalize_text(alias) in headers:
                score += 1.0
        if any(k in normalized_name for k in LOG_NAME_KEYWORDS):
            score += 1.0
        if re.search(r"20\d{2}", normalized_name):
            score += 0.5
        if _has_date_column(frame):
            score += 1.0

    if any(k in normalized_name for k in PENALIZED_NAME_KEYWORDS):
        score = max(0.0, score - 1.0)
    return score


def detect_sheet(source: BinaryIO | bytes | str, role: SheetRole, preferred: str | None = None) -> str:
    """Devuelve el nombre de hoja más apto para ``role``.

    ``preferred`` preserva la compatibilidad con los libros canónicos al
    otorgar una bonificación; el contenido de la hoja manda si no encaja.
    """
    meta = read_workbook_meta(source)
    scores = {name: score_sheet(frame, name, role) for name, frame in meta.items()}
    if preferred and preferred in scores:
        scores[preferred] += 2.0
    best = max(scores, key=lambda name: scores[name])
    if scores[best] < MIN_SHEET_SCORE:
        raise InputValidationError(
            f"No se encontró ninguna hoja compatible con el rol '{role.value}' "
            f"(hojas disponibles: {', '.join(meta) or 'ninguna'})."
        )
    return best


def classify_uploads(name_sources: Iterable[tuple[str, BinaryIO | bytes]]) -> list[FileProfile]:
    """Asigna rol SOT / BITÁCORA a cada archivo aunque se suban en cualquier orden."""
    profiles: list[FileProfile] = []
    for name, source in name_sources:
        meta = read_workbook_meta(source)
        sot_name, sot_score = max(
            ((sheet, score_sheet(frame, sheet, SheetRole.SOT)) for sheet, frame in meta.items()),
            key=lambda item: item[1],
        )
        log_name, log_score = max(
            ((sheet, score_sheet(frame, sheet, SheetRole.BITACORA)) for sheet, frame in meta.items()),
            key=lambda item: item[1],
        )
        profiles.append(FileProfile(name, source, meta, sot_name, sot_score, log_name, log_score))

    if len(profiles) == 2:
        first, second = profiles
        first_is_sot = first.sot_score + second.log_score >= first.log_score + second.sot_score
        first.role, first.sheet = (SheetRole.SOT, first.sot_sheet) if first_is_sot else (SheetRole.BITACORA, first.log_sheet)
        second.role, second.sheet = (SheetRole.BITACORA, second.log_sheet) if first_is_sot else (SheetRole.SOT, second.sot_sheet)
    return profiles


# --------------------------------------------------------------------------
# Detección de columnas (Chain of Responsibility)
# --------------------------------------------------------------------------

class ColumnResolutionStrategy:
    """Interfaz de las estrategias de resolución de columnas."""

    def resolve(self, frame: pd.DataFrame, required: bool) -> str | None:
        raise NotImplementedError


class AliasStrategy(ColumnResolutionStrategy):
    def __init__(self, aliases: Iterable[str]) -> None:
        self._aliases = tuple(aliases)

    def resolve(self, frame: pd.DataFrame, required: bool) -> str | None:
        normalized = {normalize_text(column): column for column in frame.columns}
        for alias in self._aliases:
            column = normalized.get(normalize_text(alias))
            if column is not None:
                return column
        return None


class DateContentStrategy(ColumnResolutionStrategy):
    """Columna con el mayor porcentaje de valores parseables como fecha.

    Entre columnas válidas se prefiere la cabecera de ejecución/trabajo
    (donde la brigada registró la labor), luego las de inicio/asignación.
    """

    _PRIORITY_KEYWORDS = (("ejecucion", 0.5), ("inicio", 0.2), ("asignacion", 0.2), ("trabajo", 0.1))

    def resolve(self, frame: pd.DataFrame, required: bool) -> str | None:
        best, best_score = None, 0.0
        for column in frame.columns:
            ratio = _date_ratio(frame[column])
            if ratio < 0.8:
                continue
            header = normalize_text(column)
            score = ratio + sum(bonus for keyword, bonus in self._PRIORITY_KEYWORDS if keyword in header)
            if score > best_score:
                best, best_score = column, score
        return best


class LongTextStrategy(ColumnResolutionStrategy):
    """Columna con el texto promedio más largo (detalle de la bitácora)."""

    def __init__(self, min_mean: int = 20) -> None:
        self._min_mean = min_mean

    def resolve(self, frame: pd.DataFrame, required: bool) -> str | None:
        best, best_mean = None, 0.0
        for column in frame.columns:
            mean = _mean_text_length(frame[column])
            if mean > best_mean:
                best, best_mean = column, mean
        return best if best_mean >= self._min_mean else None


class DigitIdStrategy(ColumnResolutionStrategy):
    """Columna donde casi todos los valores son un ID numérico de al menos N dígitos."""

    def __init__(self, min_digits: int = 8) -> None:
        self._min_digits = min_digits

    def resolve(self, frame: pd.DataFrame, required: bool) -> str | None:
        best, best_ratio = None, 0.0
        for column in frame.columns:
            ratio = _digit_ratio(frame[column], self._min_digits)
            if ratio > best_ratio:
                best, best_ratio = column, ratio
        return best if best_ratio >= 0.85 else None


class IncidentPatternStrategy(ColumnResolutionStrategy):
    """Columna cuyos valores refieren incidencias ``INC`` o SOTs."""

    _PATTERN = r"(?i)\b(?:nroincidencia|nro\.?\s*incidencia|inc\s*\d+|sot\s*:)\b"

    def resolve(self, frame: pd.DataFrame, required: bool) -> str | None:
        best, best_ratio = None, 0.0
        for column in frame.columns:
            values = frame[column].dropna()
            if len(values) < 2:
                continue
            ratio = float(values.astype(str).str.contains(self._PATTERN, regex=True).mean())
            if ratio > best_ratio:
                best, best_ratio = column, ratio
        return best if best_ratio >= 0.3 else None


def resolve_column(frame: pd.DataFrame, strategies: Iterable[ColumnResolutionStrategy], required: bool, label: str) -> str | None:
    for strategy in strategies:
        column = strategy.resolve(frame, required=False)
        if column is not None:
            return column
    if required:
        raise InputValidationError(f"No se encontró una columna equivalente a: {label}")
    return None


def alias_columns(frame: pd.DataFrame, aliases: Iterable[str]) -> list[str]:
    """Todas las columnas que coinciden con algún alias, en orden de aparición."""
    normalized = {normalize_text(column): column for column in frame.columns}
    return [normalized[normalize_text(alias)] for alias in aliases if normalize_text(alias) in normalized]


# --------------------------------------------------------------------------
# Contenido de soporte para la detección
# --------------------------------------------------------------------------

def _coerce_dates(series: pd.Series) -> pd.Series:
    # Se enmascaran celdas numéricas (enteros, flotantes o cadenas numéricas):
    # pandas las interpretaría como nanosegundos desde 1970, contaminando la
    # detección de fechas con columnas tipo ITEM, año, SLA o IDs.
    def _is_numeric_cell(value: object) -> bool:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if isinstance(value, str):
            return bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", value))
        return False

    masked = pd.Series([None if _is_numeric_cell(value) else value for value in series], index=series.index)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # dayfirst + ISO genera un warning interno de pandas
            return pd.to_datetime(masked, errors="coerce", dayfirst=True)
    except (TypeError, ValueError, OverflowError):  # formatos mezclados
        return pd.to_datetime(masked, errors="coerce", dayfirst=True, format="mixed")


def _sample(series: pd.Series, limit: int = 300) -> pd.Series:
    values = series.dropna()
    return values.iloc[:limit] if len(values) > limit else values


def _date_ratio(series: pd.Series) -> float:
    values = _sample(series)
    if len(values) < 2:
        return 0.0
    return float(_coerce_dates(values).notna().sum() / len(values))


def _digit_ratio(series: pd.Series, min_digits: int) -> float:
    values = _sample(series)
    if len(values) < 2:
        return 0.0
    return float(values.astype(str).str.fullmatch(rf"\d{{{min_digits},}}").mean())


def _mean_text_length(series: pd.Series) -> float:
    values = _sample(series, limit=500).astype(str)
    lengths = values.str.len()
    non_empty = lengths[lengths > 0]
    return float(non_empty.mean()) if len(non_empty) >= 2 else 0.0


def _has_date_column(frame: pd.DataFrame) -> bool:
    return any(_date_ratio(frame[column]) >= 0.8 for column in frame.columns if frame[column].notna().sum() >= 2)


def _has_long_digit_column(frame: pd.DataFrame, min_digits: int) -> bool:
    return any(_digit_ratio(frame[column], min_digits) >= 0.85 for column in frame.columns)


# --------------------------------------------------------------------------
# Preparación de marcos canónicos
# --------------------------------------------------------------------------

def prepare_sots(frame: pd.DataFrame) -> pd.DataFrame:
    """Extrae las SOTs con su texto descriptivo y una fecha objetivo (hint de columnas)."""
    id_column = resolve_column(
        frame, [AliasStrategy(SOT_ID_ALIASES), DigitIdStrategy(min_digits=8)], required=True, label="ID de SOT"
    )
    text_columns = alias_columns(frame, SOT_TEXT_ALIASES)
    if not text_columns:
        fallback = resolve_column(
            frame, [LongTextStrategy(min_mean=30)], required=False, label="texto de la SOT"
        )
        text_columns = [fallback] if fallback else []
    if not text_columns:
        raise InputValidationError("No se encontró observacion ni descripcion en la hoja de SOT.")

    hint = _first_fallback_date(frame)
    result = pd.DataFrame({"sot": frame[id_column].astype("string")})
    result["source_text"] = frame[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
    result["date_hint"] = hint
    return result.dropna(subset=["sot"]).reset_index(drop=True)


def prepare_logs(frame: pd.DataFrame) -> pd.DataFrame:
    """Extrae las bitácoras con fecha, texto, observación, ID y materiales."""
    date_column = resolve_column(
        frame,
        [AliasStrategy(LOG_DATE_ALIASES), DateContentStrategy()],
        required=True,
        label="fecha de la bitácora",
    )
    log_column = resolve_column(
        frame,
        [AliasStrategy(LOG_TEXT_ALIASES), LongTextStrategy()],
        required=True,
        label="texto de la bitácora",
    )
    observation_column = resolve_column(frame, [AliasStrategy(LOG_OBS_ALIASES)], required=False, label="observación")
    id_column = resolve_column(
        frame, [AliasStrategy(LOG_ID_ALIASES), IncidentPatternStrategy()], required=False, label="ID / INC"
    )
    material_columns = [
        column for column in frame.columns if any(hint in normalize_text(column) for hint in MATERIAL_COLUMN_HINTS)
    ]

    result = pd.DataFrame({"date": _coerce_dates(frame[date_column]).dt.date, "log_text": frame[log_column].fillna("").astype(str)})
    result["observation"] = frame[observation_column].fillna("").astype(str) if observation_column else ""
    result["record_id"] = frame[id_column].fillna("").astype(str) if id_column else ""
    result["materials_data"] = frame[material_columns].fillna("").astype(str).agg(" ".join, axis=1) if material_columns else ""
    return result.dropna(subset=["date"]).reset_index(drop=True)


def _first_fallback_date(frame: pd.DataFrame) -> list[object]:
    """Primera fecha válida, por prioridad, entre las columnas de fecha explícitas."""
    parsed: dict[str, pd.Series] = {}
    for column in DATE_FALLBACK_COLUMNS:
        if column in frame.columns:
            parsed[column] = _coerce_dates(frame[column])
    if not parsed:
        return [None] * len(frame)

    combined: pd.Series | None = None
    for column in DATE_FALLBACK_COLUMNS:
        if column in parsed:
            combined = parsed[column] if combined is None else combined.fillna(parsed[column])
    dates: list[object] = []
    for value in combined:  # type: ignore[union-attr]
        if isinstance(value, Timestamp) and not pd.isna(value):
            dates.append(value.date())
        else:
            dates.append(None)
    return dates