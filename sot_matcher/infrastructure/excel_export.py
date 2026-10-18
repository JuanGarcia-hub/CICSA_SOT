"""Exportación de la estructura contractual del resultado."""

from __future__ import annotations

from io import BytesIO

import pandas as pd

OUTPUT_COLUMNS = ["sot", "bitacoras", "materias"]


def to_excel(results: pd.DataFrame) -> bytes:
    """Serializa el resultado a un libro Excel con reglas de presentación."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results.loc[:, OUTPUT_COLUMNS].to_excel(writer, sheet_name="RESULTADO", index=False)
        worksheet = writer.book["RESULTADO"]
        worksheet.freeze_panes = "A2"
        for column, width in {"A": 18, "B": 95, "C": 12}.items():
            worksheet.column_dimensions[column].width = width
    return output.getvalue()