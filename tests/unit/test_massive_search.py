import pandas as pd

from sot_matcher.application.massive_search import filter_results, ids_from_file, parse_sot_ids, summarize


def test_parse_sot_ids_separators_and_dedup():
    ids = parse_sot_ids(["90681747\n 90073467, 9007.3467\t90073467", "9007 3467", "INC9068ma1747"])
    assert ids == ["90681747", "90073467"]


def test_parse_sot_ids_ignores_short_numbers():
    assert parse_sot_ids(["123", "0.0833", "2026"], min_digits=6) == []
    assert parse_sot_ids(["1234567"], min_digits=6) == ["1234567"]


def test_ids_from_txt():
    content = b"90681747\r\n90073467, 9007.3467\n"
    assert ids_from_file(content, "lista.txt") == ["90681747", "90073467"]


def test_ids_from_xlsx_by_content():
    import io

    buffer = io.BytesIO()
    pd.DataFrame({"col_a": [90681747.0, "90073467"]}).to_excel(buffer, index=False)
    assert ids_from_file(buffer.getvalue(), "lista.xlsx") == ["90681747", "90073467"]


def test_filter_results_only_found_matches_list():
    results = pd.DataFrame(
        {
            "sot": ["90681747", "90073467", "90073468", "90073469"],
            "estado": ["Exacto", "Semántico", "Pendiente", "Exacto"],
            "materias": ["si", "si", "no", "si"],
            "bitacoras": ["a", "b", "PENDIENTE", "d"],
        }
    )
    filtered = filter_results(results, ["90681747", "90073467", "90073469"])
    assert filtered["sot"].tolist() == ["90681747", "90073467", "90073469"]


def test_filter_results_only_found_excludes_pending():
    results = pd.DataFrame(
        {
            "sot": ["90681747", "90073467", "90073468", "90073469"],
            "estado": ["Exacto", "Semántico", "Pendiente", "Exacto"],
            "materias": ["si", "si", "no", "si"],
            "bitacoras": ["a", "b", "PENDIENTE", "d"],
        }
    )
    filtered = filter_results(results, ["90681747", "90073467", "90073468"], only_found=True)
    assert filtered["sot"].tolist() == ["90681747", "90073467"]


def test_summarize_counts_found_pending_and_materials():
    results = pd.DataFrame(
        {
            "sot": ["90681747", "90073467", "90073468", "90073469"],
            "estado": ["Exacto", "Semántico", "Pendiente", "Exacto"],
            "materias": ["si", "no", "si", "no"],
            "bitacoras": ["a", "b", "PENDIENTE", "d"],
        }
    )
    summary = summarize(results, ["90681747", "90073467", "90073468", "90073469", "99999999"])
    assert summary == {"solicitadas": 5, "en_cruce": 4, "encontradas": 3, "pendientes": 1, "materias": 1}