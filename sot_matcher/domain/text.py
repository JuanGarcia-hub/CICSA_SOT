"""Normalización y extracción heurística sobre texto operacional."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable

MATERIAL_TERMS = ("material", "cable", "fibra", "conector", "adaptador", "mufa", "fat", "ferreteria")

TOKEN_STOPWORDS = frozenset(
    {
        "de", "del", "la", "el", "lo", "las", "los", "un", "una", "unos", "unas", "y", "o",
        "que", "a", "al", "se", "su", "sus", "en", "para", "por", "con", "sin", "es", "no",
        "dia", "mes", "ano", "fecha", "hora", "nota", "reg", "nro", "cliente",
    }
)


def safe_text(value: object) -> str:
    """Convierte valores nulos y no textuales a una cadena segura."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):  # NaN sin acoplarse a pandas
        return ""
    return str(value).strip()


def normalize_text(value: object) -> str:
    """Normaliza mayúsculas, tildes, espacios y caracteres no alfanuméricos.

    Ejemplo: ``"Marca temporal"`` -> ``"marca temporal"``; ``"SOT / INC"`` -> ``"sot inc"``.
    """
    text = unicodedata.normalize("NFD", safe_text(value).lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("_x000d_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def looks_like_id(value: object, min_digits: int = 6) -> bool:
    """Indica si el valor es una cadena de dígitos suficientemente larga (ID)."""
    text = safe_text(value)
    return bool(re.fullmatch(rf"\d{{{min_digits},}}", text))


def extract_incident_ids(value: object) -> set[str]:
    """Devuelve incidencias ``INC``, conservando sus ceros a la izquierda."""
    return set(re.findall(r"\binc\s*0*(\d+)\b", normalize_text(value)))


def token_set(text: str, min_length: int = 3) -> set[str]:
    """Tokens significativos de un texto ya normalizado (filtro de similitud barato)."""
    return {token for token in text.split() if len(token) >= min_length and token not in TOKEN_STOPWORDS}


def has_token_overlap(source_tokens: set[str], candidate_tokens: set[str]) -> bool:
    """Verifica afinidad léxica mínima entre una SOT y una bitácora candidata."""
    return bool(source_tokens and candidate_tokens and source_tokens.intersection(candidate_tokens))


def has_materials(values: Iterable[object]) -> bool:
    """Detecta uso de materiales respetando declaraciones explícitas de ausencia."""
    text = normalize_text(" ".join(safe_text(value) for value in values))
    if not text or re.search(r"\b(sin|no|0)\s+material(?:es)?\b", text):
        return False
    return any(term in text for term in MATERIAL_TERMS)