"""Tests unitarios para FeatureExtractor."""

import pytest
from sot_matcher.infrastructure.feature_extractor import SotLogFeatureExtractor


@pytest.fixture
def extractor():
    return SotLogFeatureExtractor(embedder=None)


def test_feature_extractor_schema_version(extractor):
    assert extractor.schema_version == "1.0"


def test_extract_incidencia_exacta(extractor):
    sot = {"source_text": "INC000123 trabajo programado"}
    log = {"log_text": "Atención INC000123 con cable", "observation": "", "materials_data": ""}
    features = extractor.extract(sot, log)
    assert features["incidencia_exacta"] == 1.0


def test_extract_no_incidencia(extractor):
    sot = {"source_text": "trabajo sin incidencia"}
    log = {"log_text": "otra bitácora", "observation": "", "materials_data": ""}
    features = extractor.extract(sot, log)
    assert features["incidencia_exacta"] == 0.0


def test_extract_sot_en_bitacora(extractor):
    sot = {"source_text": "SOT 90232501 mantenimiento"}
    log = {"log_text": "Atención SOT 90232501 completada", "observation": "", "materials_data": ""}
    features = extractor.extract(sot, log)
    assert features["sot_en_bitacora"] == 1.0


def test_extract_fecha_coincide(extractor):
    from datetime import date
    sot = {"source_text": "trabajo 30/01/2026", "date_hint": date(2026, 1, 30)}
    log = {"log_text": "mantenimiento", "observation": "", "materials_data": "", "date": date(2026, 1, 30)}
    features = extractor.extract(sot, log)
    assert features["fecha_coincide"] == 1.0
    assert features["dias_diferencia"] == 0.0


def test_extract_fecha_distinta(extractor):
    from datetime import date
    sot = {"source_text": "trabajo 30/01/2026", "date_hint": date(2026, 1, 30)}
    log = {"log_text": "mantenimiento", "observation": "", "materials_data": "", "date": date(2026, 2, 5)}
    features = extractor.extract(sot, log)
    assert features["fecha_coincide"] == 0.0
    assert features["dias_diferencia"] == 6.0


def test_extract_zona_actividad(extractor):
    sot = {"source_text": "mantenimiento zona norte"}
    log = {"log_text": "mantenimiento preventivo zona norte", "observation": "", "materials_data": ""}
    features = extractor.extract(sot, log)
    assert features["coincidencia_zona"] == 1.0
    assert features["coincidencia_actividad"] == 1.0


def test_extract_similaridad_descripcion(extractor):
    sot = {"source_text": "mantenimiento preventivo cable fibra"}
    log = {"log_text": "mantenimiento correctivo cable fibra óptica", "observation": "", "materials_data": ""}
    features = extractor.extract(sot, log)
    # Jaccard: intersection = {mantenimiento, cable, fibra} = 3, union = {mantenimiento, preventivo, cable, fibra, correctivo, óptica} = 6
    # 3/6 = 0.5
    assert features["similitud_descripcion"] > 0.3


def test_extract_materiales(extractor):
    sot = {"source_text": "trabajo"}
    log = {"log_text": "instalación", "observation": "", "materials_data": "cable FO conector"}
    features = extractor.extract(sot, log)
    assert features["tiene_materiales"] == 1.0


def test_extract_sin_materiales(extractor):
    sot = {"source_text": "trabajo"}
    log = {"log_text": "inspección visual sin materiales", "observation": "", "materials_data": ""}
    features = extractor.extract(sot, log)
    # has_materials detecta "sin materiales" -> False
    assert features["tiene_materiales"] == 0.0