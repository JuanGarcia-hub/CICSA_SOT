"""Codificador de embeddings con caché de proceso (singleton + caché por texto).

El modelo es costoso de cargar (varios segundos); se carga una única vez por
proceso mediante un *singleton*, se explotan los hilos de CPU disponibles y
los vectores ya calculados se reutilizan entre ejecuciones (p.ej. al variar el
umbral en la UI) para acelerar el resultado final.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

from ..domain.matching import DEFAULT_MODEL, Embedder

_MAX_CACHED_VECTORS = 50_000
_RAW_MODEL: Any | None = None
_CACHED_WRAPPER: SentenceTransformerEmbedder | None = None
_LOCK = threading.Lock()


def _tune_torch_threads() -> None:
    try:
        import torch

        torch.set_num_threads(os.cpu_count() or 8)
    except Exception:  # noqa: BLE001, S110 - torch ausente o sin soporte
        pass


class SentenceTransformerEmbedder:
    """Implementación del *Strategy* ``Embedder`` con carga diferida y caché LRU."""

    def __init__(self, model_name: str = DEFAULT_MODEL, delegate: Any | None = None, max_cached: int = _MAX_CACHED_VECTORS) -> None:
        self._model_name = model_name
        self._delegate = delegate
        self._max_cached = max_cached
        self._cache: OrderedDict[tuple[str, bool], np.ndarray] = OrderedDict()

    def _ensure(self) -> Any:
        if self._delegate is None:
            self._delegate = get_sentence_transformer(self._model_name)
        return self._delegate

    def encode(self, sentences: list[str], **kwargs: object) -> np.ndarray:
        normalize = bool(kwargs.get("normalize_embeddings", False))
        results: list[np.ndarray | None] = [None] * len(sentences)
        missing_keys: list[tuple[str, bool]] = []
        missing_positions: list[int] = []

        for position, sentence in enumerate(sentences):
            key = (sentence, normalize)
            if key in self._cache:
                self._cache.move_to_end(key)
                results[position] = self._cache[key]
            else:
                missing_keys.append(key)
                missing_positions.append(position)

        if missing_keys:
            unique_keys = list(dict.fromkeys(missing_keys))
            vectors = np.asarray(
                self._ensure().encode([key[0] for key in unique_keys], **kwargs)
            )
            vector_by_key = dict(zip(unique_keys, vectors))
            for key, vector in vector_by_key.items():
                self._cache[key] = vector
                if len(self._cache) > self._max_cached:
                    self._cache.popitem(last=False)
            for position, key in zip(missing_positions, missing_keys):
                results[position] = vector_by_key[key]

        if not results:
            return np.empty((0,), dtype=np.float32)
        return np.stack(results)  # type: ignore[arg-type]


def get_sentence_transformer(model_name: str = DEFAULT_MODEL) -> Any:
    """Modelo crudo cargado una sola vez por proceso (singleton thread-safe)."""
    global _RAW_MODEL
    if _RAW_MODEL is None:
        with _LOCK:
            if _RAW_MODEL is None:
                from sentence_transformers import SentenceTransformer

                _tune_torch_threads()
                model = SentenceTransformer(model_name)
                # 64 tokens bastan para ranking de similitud y reducen ~2.3x el
                # costo de codificar bitácoras largas sin degradar el cruce.
                model.max_seq_length = 64
                _RAW_MODEL = model
    return _RAW_MODEL


def get_embedder(model_name: str = DEFAULT_MODEL) -> Embedder:
    """Embedder con caché por texto compartido en todo el proceso."""
    global _CACHED_WRAPPER
    if _CACHED_WRAPPER is None:
        with _LOCK:
            if _CACHED_WRAPPER is None:
                _CACHED_WRAPPER = SentenceTransformerEmbedder(model_name)
    return _CACHED_WRAPPER


def build_embedder(model_name: str = DEFAULT_MODEL) -> Callable[[], Embedder]:
    """Factory diferida útil para inyección perezosa de dependencias."""
    return lambda: get_embedder(model_name)