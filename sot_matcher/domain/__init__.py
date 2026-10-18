"""Capa de dominio: lógica pura de negocio sin dependencias de I/O."""

from .matching import MatchingEngine, MatchingSettings, Embedder
from .text import (
    normalize_text,
    safe_text,
    looks_like_id,
    extract_incident_ids,
    token_set,
    has_token_overlap,
    has_materials,
)
from .dates import extract_date, resolve_date
from .feedback import (
    FeedbackDecision,
    FeedbackEvent,
    CandidatePair,
    MatchingPrediction,
    LabeledPair,
)

__all__ = [
    "MatchingEngine",
    "MatchingSettings",
    "Embedder",
    "normalize_text",
    "safe_text",
    "looks_like_id",
    "extract_incident_ids",
    "token_set",
    "has_token_overlap",
    "has_materials",
    "extract_date",
    "resolve_date",
    "FeedbackDecision",
    "FeedbackEvent",
    "CandidatePair",
    "MatchingPrediction",
    "LabeledPair",
]