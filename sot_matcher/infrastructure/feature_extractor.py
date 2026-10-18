"""Extractor de características para pares SOT-bitácora.

Implementa FeatureExtractor protocol. Características reproducibles y versionadas.
"""

from __future__ import annotations

from typing import Any, Optional

from sot_matcher.application.ports import FeatureExtractor
from sot_matcher.domain.text import (
    normalize_text,
    extract_incident_ids,
    token_set,
    has_token_overlap,
    has_materials,
)
from sot_matcher.domain.dates import extract_date, resolve_date


class SotLogFeatureExtractor:
    """Extrae características numéricas/categóricas para un par (SOT, bitácora)."""

    SCHEMA_VERSION = "1.0"
    FEATURE_NAMES = [
        "incidencia_exacta",
        "sot_en_bitacora",
        "fecha_coincide",
        "dias_diferencia",
        "similitud_semantica",
        "coincidencia_zona",
        "coincidencia_actividad",
        "similitud_descripcion",
        "datos_fecha_disponibles",
        "datos_incidencia_disponibles",
        "tiene_materiales",
    ]

    def __init__(self, embedder: Any = None):
        self._embedder = embedder

    @property
    def schema_version(self) -> str:
        return self.SCHEMA_VERSION

    def extract(
        self,
        sot_record: dict[str, Any],
        log_record: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sot_text = str(sot_record.get("source_text", ""))
        log_text = str(log_record.get("log_text", ""))
        obs_text = str(log_record.get("observation", ""))
        materials_text = str(log_record.get("materials_data", ""))
        combined_log = f"{log_text} {obs_text} {materials_text}".strip()

        norm_sot = normalize_text(sot_text)
        norm_log = normalize_text(combined_log)

        # 1. Incidencia exacta (INC)
        sot_incs = extract_incident_ids(sot_text)
        log_incs = extract_incident_ids(combined_log)
        incidencia_exacta = 1.0 if sot_incs & log_incs else 0.0

        # 2. SOT ID en bitácora
        sot_digits = "".join(c for c in norm_sot if c.isdigit())
        sot_en_bitacora = 1.0 if sot_digits and sot_digits in norm_log else 0.0

        # 3. Fecha
        sot_date = resolve_date(sot_text, sot_record.get("date_hint"))
        log_date = log_record.get("date")
        fecha_coincide = 0.0
        dias_diferencia = -1.0
        datos_fecha_disponibles = 0.0

        if sot_date and log_date:
            datos_fecha_disponibles = 1.0
            from datetime import date
            if isinstance(log_date, str):
                from datetime import datetime
                try:
                    log_date = datetime.fromisoformat(log_date).date()
                except ValueError:
                    log_date = None
            if isinstance(log_date, date):
                diff = abs((sot_date - log_date).days)
                dias_diferencia = float(diff)
                fecha_coincide = 1.0 if diff == 0 else 0.0

        # 4. Similitud semántica (si hay embedder)
        similitud_semantica = 0.0
        if self._embedder and norm_sot and norm_log:
            try:
                vecs = self._embedder.encode([norm_sot, norm_log], normalize_embeddings=True)
                similitud_semantica = float(vecs[0] @ vecs[1])
            except Exception:
                pass

        # 5. Coincidencia zona/actividad (heurística por tokens)
        sot_tokens = token_set(norm_sot)
        log_tokens = token_set(norm_log)
        zona_keywords = {"zona", "sector", "area", "region", "ciudad", "localidad"}
        act_keywords = {"instalacion", "mantenimiento", "reparacion", "revision", "inspeccion"}

        coincidencia_zona = 1.0 if sot_tokens & log_tokens & zona_keywords else 0.0
        coincidencia_actividad = 1.0 if sot_tokens & log_tokens & act_keywords else 0.0

        # 6. Similitud descripción (Jaccard tokens)
        union = sot_tokens | log_tokens
        inter = sot_tokens & log_tokens
        similitud_descripcion = len(inter) / len(union) if union else 0.0

        # 7. Datos incidencia disponibles
        datos_incidencia_disponibles = 1.0 if (sot_incs or log_incs) else 0.0

        # 8. Materiales
        tiene_materiales = 1.0 if has_materials([log_text, obs_text, materials_text]) else 0.0

        return {
            "incidencia_exacta": incidencia_exacta,
            "sot_en_bitacora": sot_en_bitacora,
            "fecha_coincide": fecha_coincide,
            "dias_diferencia": dias_diferencia,
            "similitud_semantica": similitud_semantica,
            "coincidencia_zona": coincidencia_zona,
            "coincidencia_actividad": coincidencia_actividad,
            "similitud_descripcion": similitud_descripcion,
            "datos_fecha_disponibles": datos_fecha_disponibles,
            "datos_incidencia_disponibles": datos_incidencia_disponibles,
            "tiene_materiales": tiene_materiales,
        }