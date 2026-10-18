"""Motor de matching híbrido: fecha estricta, ID exacto y embeddings.

Flujo por SOT:
1. Se filtra estrictamente por fecha (texto o columna de fecha de la SOT).
2. Se intenta una coincidencia determinista (ID de SOT o incidencia ``INC``);
   más de una coincidencia => ambiguo, permanece ``PENDIENTE``.
3. Sin ID exacto, se delega en el *Strategy* de similitud semántica; solo
   supera si alcanza el umbral y una diferencia mínima frente al segundo.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .dates import resolve_date
from .text import (
    extract_incident_ids,
    has_materials,
    has_token_overlap,
    normalize_text,
    token_set,
)

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

OUTPUT_COLUMNS = ["sot", "bitacoras", "materias", "estado", "confianza"]
PENDIENTE = "PENDIENTE"


class Embedder(Protocol):
    """Abstracción (dependencia invertida) del codificador de vectores."""

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray: ...


@dataclass(frozen=True)
class MatchingSettings:
    threshold: float = 0.78
    ambiguity_margin: float = 0.03


class MatchingEngine:
    """Orquestador del cruce. Recibe opcionalmente un ``Embedder`` (Strategy)."""

    def __init__(self, settings: MatchingSettings | None = None, embedder: Embedder | None = None) -> None:
        self.settings = settings or MatchingSettings()
        self._embedder = embedder

    def match(self, sots: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
        if sots is None or sots.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        if logs is None or logs.empty:
            return pd.DataFrame([self._pending(str(sot)) for sot in sots["sot"]], columns=OUTPUT_COLUMNS)

        prepared = self._prepare_logs(logs)
        logs_by_date = {log_date: group for log_date, group in prepared.groupby("date", sort=False)}
        if "date_hint" not in sots.columns:
            sots = sots.assign(date_hint=[None] * len(sots))

        sot_ids = sots["sot"].astype(str).tolist()
        sources = sots["source_text"].fillna("").tolist()
        hints = [None if pd.isna(row) else row for row in sots["date_hint"].tolist()]
        results: list[dict[str, object]] = [self._pending("*")] * len(sots)
        per_date: dict[object, list[int]] = defaultdict(list)

        id_positions, incident_positions = self._exact_evidence_maps(prepared)

        for position, sot_id in enumerate(sot_ids):
            target_date = resolve_date(sources[position], hints[position])
            candidates = logs_by_date.get(target_date)
            if target_date is None or candidates is None or candidates.empty:
                winner = self._exact_anywhere(
                    sot_id, normalize_text(sot_id), extract_incident_ids(sources[position]),
                    prepared, target_date, id_positions, incident_positions,
                )
                results[position] = (
                    self._assigned(sot_id, prepared.loc[winner], "Exacto", 1.0) if winner is not None else self._pending(sot_id)
                )
            else:
                per_date[target_date].append(position)

        tasks: list[tuple[int, str, str, list[int], object]] = []
        for target_date, positions in per_date.items():
            group = logs_by_date[target_date]
            texts = group["match_text"].tolist()
            incidents = group["incident_ids"].tolist()
            index = group.index.tolist()
            candidate_tokens = [token_set(text) for text in texts]
            for position in positions:
                normalized_sot = normalize_text(sot_ids[position])
                source_incidents = extract_incident_ids(sources[position])
                same_day = [
                    p for p, (text, candidate_incidents) in enumerate(zip(texts, incidents))
                    if (normalized_sot and normalized_sot in text) or source_incidents.intersection(candidate_incidents)
                ]
                if len(same_day) == 1:
                    results[position] = self._assigned(sot_ids[position], group.loc[index[same_day[0]]], "Exacto", 1.0)
                elif len(same_day) > 1:
                    results[position] = self._pending(sot_ids[position])
                else:
                    winner = self._exact_anywhere(sot_ids[position], normalized_sot, source_incidents, prepared, target_date, id_positions, incident_positions)
                    if winner is not None:
                        results[position] = self._assigned(sot_ids[position], prepared.loc[winner], "Exacto", 1.0)
                    else:
                        results[position] = self._pending(sot_ids[position])
                        source_tokens = token_set(normalize_text(sources[position]))
                        if any(
                            has_token_overlap(source_tokens, candidate_tokens[p]) for p in range(len(texts))
                        ):
                            tasks.append((position, sot_ids[position], sources[position], index, target_date))

        self._resolve_semantic_tasks(tasks, prepared, results)
        return pd.DataFrame(results, columns=OUTPUT_COLUMNS)

    @staticmethod
    def _exact_evidence_maps(prepared: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """Mapas ID de SOT e INC -> posiciones de bitácoras (evidencia sin restringir fecha)."""
        import re as _re

        id_positions: dict[str, list[int]] = defaultdict(list)
        incident_positions: dict[str, list[int]] = defaultdict(list)
        for index, row in prepared.iterrows():
            key = _re.sub(r"\D", "", str(row["record_id"])) if row["record_id"] else ""
            if key:
                id_positions[key].append(index)
            for incident in row["incident_ids"]:
                incident_positions[incident].append(index)
        return id_positions, incident_positions

    @staticmethod
    def _exact_anywhere(
        sot_id: str,
        normalized_sot: str,
        source_incidents: set[str],
        prepared: pd.DataFrame,
        target_date: object,
        id_positions: dict[str, list[int]],
        incident_positions: dict[str, list[int]],
    ) -> int | None:
        """Devuelve LA bitácora con evidencia exacta aun en otra fecha, o ``None``.

        Desempate: el registro cronológicamente más cercano a la fecha objetivo.
        Si hay empate de cercanía, se entrega la decisión (ambigüedad), como en
        el criterio de margen de la similitud semántica.
        """
        import re as _re

        hits = set()
        key = _re.sub(r"\D", "", normalized_sot)
        if key:
            hits.update(id_positions.get(key, ()))
        for incident in source_incidents:
            hits.update(incident_positions.get(incident, ()))
        if not hits:
            return None
        if len(hits) == 1:
            return hits.pop()
        if target_date is None:
            return None
        def _distance(index: int) -> int:
            try:
                return abs((pd.Timestamp(target_date) - pd.Timestamp(str(prepared.loc[index, "date"]))).days)
            except (TypeError, ValueError):
                return float("inf")
        ordered = sorted(hits, key=_distance)
        return ordered[0] if len(ordered) == 1 or _distance(ordered[0]) < _distance(ordered[1]) else None

    @staticmethod
    def _prepare_logs(logs: pd.DataFrame) -> pd.DataFrame:
        prepared = logs.copy()
        prepared["match_text"] = (
            prepared["record_id"].fillna("") + " " + prepared["log_text"].fillna("") + " " + prepared["observation"].fillna("")
        ).map(normalize_text)
        prepared["incident_ids"] = prepared["match_text"].map(extract_incident_ids)
        return prepared

    def _resolve_semantic_tasks(
        self, tasks: list[tuple[int, str, str, list[int], object]], logs: pd.DataFrame, results: list[dict[str, object]]
    ) -> None:
        """Codifica cada texto único una sola vez (bitácoras y SOTs) y reutiliza el vector."""
        if not tasks:
            return
        model = self._get_embedder()

        unique_log_indices = sorted({log_index for (_, _, _, indices, _) in tasks for log_index in indices})
        log_source_texts = [str(logs.loc[index, "match_text"]) for index in unique_log_indices]
        unique_texts, inverse = np.unique(log_source_texts, return_inverse=True)
        log_vectors = model.encode(unique_texts.tolist(), normalize_embeddings=True, batch_size=64)
        vectors_by_log = {index: log_vectors[pos] for index, pos in zip(unique_log_indices, inverse)}

        task_sources = [normalize_text(source) for (_, _, source, _, _) in tasks]
        unique_sources, source_inverse = np.unique(task_sources, return_inverse=True)
        source_vectors = model.encode(unique_sources.tolist(), normalize_embeddings=True, batch_size=64)

        for (position, _sot, _source, candidate_indices, _), source_vector in zip(tasks, source_vectors[source_inverse]):
            scores = np.asarray([vectors_by_log[log_index] @ source_vector for log_index in candidate_indices])
            winner = int(np.argmax(scores))
            best_score = float(scores[winner])
            if len(scores) > 1:
                second_score = float(np.partition(scores, -2)[-2])
            else:
                second_score = -1.0
            if best_score >= self.settings.threshold and best_score - second_score >= self.settings.ambiguity_margin:
                results[position] = self._assigned(
                    _sot, logs.loc[candidate_indices[winner]], "Semántico", best_score
                )

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            from ..infrastructure.embedding import get_embedder
            self._embedder = get_embedder(DEFAULT_MODEL)
        return self._embedder

    @staticmethod
    def _pending(sot: str) -> dict[str, object]:
        return {"sot": sot, "bitacoras": PENDIENTE, "materias": "no", "estado": "Pendiente", "confianza": None}

    @staticmethod
    def _assigned(sot: str, row: pd.Series, method: str, score: float) -> dict[str, object]:
        materials = has_materials([row["log_text"], row["observation"], row["materials_data"]])
        return {
            "sot": sot,
            "bitacoras": row["log_text"],
            "materias": "si" if materials else "no",
            "estado": method,
            "confianza": round(score, 4),
        }