"""Búsqueda masiva: filtrar el cruce sobre una lista explícita de SOTs.

La lista puede venir de una caja de texto o de un archivo y se interpreta
siempre por contenido (separadores de línea, coma, espacio o punto decimal);
nunca asume columnas o nombres fijos.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from io import BytesIO

import pandas as pd

_SEPARATOR = re.compile(r"[\s,;]+")
_NON_DIGIT = re.compile(r"\D")


def parse_sot_ids(values: Iterable[object], min_digits: int = 6) -> list[str]:
    """Extrae identificadores de SOT de un texto o colección, sin duplicados."""
    seen: set[str] = set()
    ids: list[str] = []
    for value in values:
        for token in _SEPARATOR.split(str(value or "").strip()):
            digits = _NON_DIGIT.sub("", token)
            if len(digits) >= min_digits and digits not in seen:
                seen.add(digits)
                ids.append(digits)
    return ids


def _parse_cells(frame: pd.DataFrame) -> list[str]:
    flattened: list[str] = []
    for value in frame.to_numpy().ravel():
        if value is not None and str(value).strip():
            flattened.append(str(value))
    return flattened


def ids_from_file(content: bytes, filename: str) -> list[str]:
    """Lee los SOT de un archivo .txt/.csv/.xlsx/.xlsm por contenido.

    En texto plano se considera cada línea una posible lista; en hojas se
    recorre la primera hoja completa (todas sus columnas) buscando dígitos.
    """
    lower = filename.lower()
    if lower.endswith((".xlsx", ".xlsm")):
        frame = pd.read_excel(BytesIO(content), sheet_name=0, engine="calamine", dtype=object)
        return parse_sot_ids(_parse_cells(frame))
    if lower.endswith(".csv"):
        frame = pd.read_csv(BytesIO(content), header=None, dtype=object)
        return parse_sot_ids(_parse_cells(frame))
    text = content.decode("utf-8", errors="replace")
    return parse_sot_ids(text.splitlines())


def _match_key(value: object) -> str:
    return _NON_DIGIT.sub("", str(value or ""))


def filter_results(results: pd.DataFrame, ids: Iterable[str], only_found: bool = True) -> pd.DataFrame:
    """Recorta el cruce a los SOTs de ``ids`` (opcionalmente solo asignados)."""
    wanted = set(parse_sot_ids(ids))
    keys = results["sot"].map(_match_key)
    filtered = results.loc[keys.isin(wanted)].copy()
    if only_found:
        filtered = filtered.loc[filtered["estado"] != "Pendiente"].copy()
    return filtered.reset_index(drop=True)


def summarize(results: pd.DataFrame, ids: Iterable[str]) -> dict[str, int]:
    """Contadores para la ficha de la búsqueda masiva."""
    wanted = set(parse_sot_ids(ids))
    keys = results["sot"].map(_match_key)
    subset = results.loc[keys.isin(wanted)]
    assigned = subset.loc[subset["estado"] != "Pendiente"]
    return {
        "solicitadas": len(wanted),
        "en_cruce": len(subset),
        "encontradas": len(assigned),
        "pendientes": len(subset) - len(assigned),
        "materias": int((assigned["materias"] == "si").sum()),
    }