"""Capa de aplicación: fachada del pipeline de cruce SOT ↔ bitácoras.

`CrossingPipeline` aísla a la UI de los detalles de infraestructura y dominio;
expone progreso mediante un callback para alimentar barras de estado y
permite validar/reusar los datos preparados sin repetir lectura de Excel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import BinaryIO

import pandas as pd

from ..domain.matching import DEFAULT_MODEL, Embedder, MatchingEngine, MatchingSettings
from ..infrastructure import excel_reader
from ..infrastructure.embedding import SentenceTransformerEmbedder
from ..infrastructure.excel_reader import FileProfile, SheetRole

ProgressCallback = Callable[[int, str], None]


@dataclass
class PreparedInput:
    sot_sheet: str
    log_sheet: str
    sot_source: object
    log_source: object
    sots: pd.DataFrame
    logs: pd.DataFrame


@dataclass
class CrossingResult(PreparedInput):
    results: pd.DataFrame


class CrossingPipeline:
    """Fachada de alto nivel del cruce SOT ↔ bitácoras."""

    def __init__(self, settings: MatchingSettings | None = None, embedder: Embedder | None = None) -> None:
        self.settings = settings or MatchingSettings()
        self._embedder = embedder

    def prepare(
        self,
        sot_source: BinaryIO | bytes | str,
        log_source: BinaryIO | bytes | str,
        use_column_dates: bool = True,
        on_progress: ProgressCallback | None = None,
        sot_sheet: str | None = None,
        log_sheet: str | None = None,
    ) -> PreparedInput:
        emit = on_progress or (lambda _percent, _message: None)
        emit(10, "Detectando hojas de los libros…")
        if sot_sheet is None:
            sot_sheet = excel_reader.detect_sheet(sot_source, SheetRole.SOT, preferred=excel_reader.SOT_SHEET)
        if log_sheet is None:
            log_sheet = excel_reader.detect_sheet(log_source, SheetRole.BITACORA, preferred=excel_reader.LOG_SHEET)

        emit(35, "Preparando registros SOT y bitácoras…")
        sots = excel_reader.prepare_sots(excel_reader.read_excel(sot_source, sot_sheet))
        logs = excel_reader.prepare_logs(excel_reader.read_excel(log_source, log_sheet))
        if not use_column_dates:
            sots = sots.assign(date_hint=[None] * len(sots))

        emit(45, "Estructura validada.")
        return PreparedInput(sot_sheet, log_sheet, sot_source, log_source, sots, logs)

    def cross(
        self,
        prepared: PreparedInput,
        settings: MatchingSettings | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> CrossingResult:
        emit = on_progress or (lambda _percent, _message: None)
        if settings is not None:
            self.settings = settings
        emit(55, "Calculando coincidencias exactas y semánticas…")
        engine = MatchingEngine(self.settings, embedder=self._embedder)
        results = engine.match(prepared.sots, prepared.logs)
        emit(100, "Cruce completado.")
        return CrossingResult(
            sot_sheet=prepared.sot_sheet,
            log_sheet=prepared.log_sheet,
            sot_source=prepared.sot_source,
            log_source=prepared.log_source,
            sots=prepared.sots,
            logs=prepared.logs,
            results=results,
        )

    def run(
        self,
        sot_source: BinaryIO | bytes | str,
        log_source: BinaryIO | bytes | str,
        settings: MatchingSettings | None = None,
        use_column_dates: bool = True,
        on_progress: ProgressCallback | None = None,
    ) -> CrossingResult:
        prepared = self.prepare(sot_source, log_source, use_column_dates, on_progress)
        return self.cross(prepared, settings, on_progress)

    def classify(self, name_sources: list[tuple[str, BinaryIO | bytes | str]]) -> list[FileProfile]:
        """Clasifica los archivos subidos en roles SOT / bitácora (independiente del orden)."""
        return excel_reader.classify_uploads(name_sources)


def build_pipeline(model_name: str = DEFAULT_MODEL) -> CrossingPipeline:
    """Factory del pipeline con embedder diferido (utiliza la caché de modelo)."""
    return CrossingPipeline(embedder=SentenceTransformerEmbedder(model_name))