
from sot_matcher.domain.text import (
    extract_incident_ids,
    has_materials,
    normalize_text,
    safe_text,
)


def test_normalize_text_removes_accents_and_noise():
    assert normalize_text("  Supervisión__ RÍO _x000D_ ") == "supervision rio"


def test_safe_text_handles_null_and_numeric():
    assert safe_text(None) == ""
    assert safe_text(83) == "83"
    assert safe_text("<NA>") != ""  # pandas NA textual no se borra


def test_extracts_incident_identifier_without_leading_zeros():
    assert extract_incident_ids("NroIncidencia: INC000000036596") == {"36596"}


def test_material_detection_excludes_explicit_absence():
    assert has_materials(["SIN MATERIALES", "cable FO"]) is False
    assert has_materials(["Se utilizó cable de fibra óptica"]) is True
    assert has_materials(["0 material"]) is False