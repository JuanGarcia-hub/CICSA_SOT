"""API pública del paquete `sot_matcher`.

Expone la fachada del cruce y los tipos esenciales para que la UI y las
pruebas no dependan de los detalles internos.
"""

from .application.pipeline import CrossingPipeline, CrossingResult
from .domain.matching import MatchingEngine, MatchingSettings
from .infrastructure.excel_export import to_excel
from .infrastructure.excel_reader import InputValidationError, SheetRole

__all__ = [
    "CrossingPipeline",
    "CrossingResult",
    "InputValidationError",
    "MatchingEngine",
    "MatchingSettings",
    "SheetRole",
    "to_excel",
]