"""Aplicación Streamlit: cruce inteligente de SOTs con bitácoras.

Interfaz profesional de tres pasos con detección automática de archivos/hojas,
corrección manual, búsqueda masiva por lista y exportación Excel.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from sot_matcher import (
    CrossingPipeline,
    InputValidationError,
    MatchingSettings,
    to_excel,
)
from sot_matcher.application.massive_search import filter_results, ids_from_file, parse_sot_ids, summarize

st.set_page_config(page_title="Cruce SOT · Bitácoras", page_icon="🔎", layout="wide")

PIPELINE = CrossingPipeline()
STATUS_COLORS = {"Exacto": "#15803d", "Semántico": "#1d4ed8", "Pendiente": "#b45309"}
STATUS_BG = {"Exacto": "#f0fdf4", "Semántico": "#eff6ff", "Pendiente": "#fffbeb"}
STATUS_ICONS = {"Exacto": "✅ Exacto", "Semántico": "🧠 Semántico", "Pendiente": "⏸️ Pendiente"}

THEME_CSS = """
<style>
:root {
  --brand: #1d4ed8;
  --brand-dark: #1e40af;
  --ink: #1f2430;
  --muted: #64748b;
  --border: #e2e8f0;
  --bg-soft: #f6f8fb;
  --bg-card: #ffffff;
  --bg-sidebar: #fbfcfe;
  --chip-exacto-bg: #f0fdf4; --chip-exacto-tx: #15803d; --chip-exacto-bd: #bbf7d0;
  --chip-sem-bg: #eff6ff;   --chip-sem-tx: #1d4ed8;   --chip-sem-bd: #bfdbfe;
  --chip-pend-bg: #fffbeb;  --chip-pend-tx: #b45309;  --chip-pend-bd: #fde68a;
}
html[data-theme="dark"] {
  --ink: #e8ecf3;
  --muted: #9aa7b8;
  --border: #2b3442;
  --bg-soft: #111827;
  --bg-card: #16202e;
  --bg-sidebar: #0f1622;
  --chip-exacto-bg: rgba(34,197,94,.14); --chip-exacto-tx: #4ade80; --chip-exacto-bd: rgba(34,197,94,.35);
  --chip-sem-bg: rgba(59,130,246,.14);   --chip-sem-tx: #60a5fa;   --chip-sem-bd: rgba(59,130,246,.35);
  --chip-pend-bg: rgba(251,191,36,.12);  --chip-pend-tx: #fbbf24;  --chip-pend-bd: rgba(251,191,36,.35);
}
html, body, [class*="css"] { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; color: var(--ink); }

/* Banner principal: un solo tono de azul */
.app-banner {
  background: linear-gradient(135deg, var(--brand-dark) 0%, var(--brand) 70%);
  border-radius: 16px;
  padding: 22px 26px;
  color: #ffffff;
  margin-bottom: 18px;
  box-shadow: 0 8px 24px rgba(29,78,216,.18);
}
.app-banner h1 { margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing: .2px; }
.app-banner p { margin: 6px 0 0; opacity: .92; font-size: .95rem; }
.app-banner .pill {
  display: inline-block; margin-top: 10px; padding: 3px 12px;
  border-radius: 999px; background: rgba(255,255,255,.18);
  border: 1px solid rgba(255,255,255,.4); font-size: .78rem; letter-spacing: .4px;
}

/* Stepper */
.steps { display: flex; gap: 10px; margin: 4px 0 20px; }
.step {
  flex: 1; display: flex; align-items: center; gap: 10px;
  border: 1px solid var(--border); border-radius: 12px; padding: 10px 14px;
  background: var(--bg-card); color: var(--muted); font-weight: 600; font-size: .86rem;
}
.step .num {
  width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center;
  background: var(--bg-soft); color: var(--muted); font-size: .8rem; font-weight: 700;
}
.step.active { border-color: var(--brand); background: var(--bg-soft); color: var(--brand); }
.step.active .num { background: var(--brand); color: #fff; }
.step.done .num { background: #16a34a; color: #fff; }
.step.done { color: #16a34a; }

/* Tarjetas de métricas: neutrales, borde izquierdo discreto */
.metric-card {
  border: 1px solid var(--border); border-left: 4px solid var(--border);
  border-radius: 12px; padding: 14px 16px; background: var(--bg-card);
  box-shadow: 0 2px 6px rgba(15,23,42,.04);
}
.metric-card .label { font-size: .74rem; text-transform: uppercase; letter-spacing: .6px; color: var(--muted); }
.metric-card .value { font-size: 1.55rem; font-weight: 700; color: var(--ink); margin-top: 2px; }
.metric-card.green { border-left-color: #16a34a; }
.metric-card.amber { border-left-color: #d97706; }

/* Leyenda de estados */
.legend { display: flex; gap: 8px; flex-wrap: wrap; margin: 2px 0 12px; }
.chip {
  display: inline-flex; align-items: center; gap: 6px; padding: 3px 12px;
  border-radius: 999px; font-size: .8rem; font-weight: 600;
}
.chip.exacto { background: var(--chip-exacto-bg); color: var(--chip-exacto-tx); border: 1px solid var(--chip-exacto-bd); }
.chip.semantico { background: var(--chip-sem-bg); color: var(--chip-sem-tx); border: 1px solid var(--chip-sem-bd); }
.chip.pendiente { background: var(--chip-pend-bg); color: var(--chip-pend-tx); border: 1px solid var(--chip-pend-bd); }

/* Callouts */
.callout {
  border: 1px dashed var(--border); background: var(--bg-soft);
  border-radius: 12px; padding: 16px 18px; color: var(--muted); font-size: .9rem;
}
.callout strong { color: var(--ink); }

/* Sidebar: sigue el tema claro/oscuro */
section[data-testid="stSidebar"] { background: var(--bg-sidebar) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--ink); }
.sidebar-brand { padding: 6px 0 14px; border-bottom: 1px solid var(--border); margin-bottom: 14px; }
.sidebar-brand .t { font-weight: 800; color: var(--brand); font-size: 1.02rem; letter-spacing: .2px; }
.sidebar-brand .s { font-size: .74rem; color: var(--muted); margin-top: 2px; }

/* Botones */
.stButton > button {
  border-radius: 10px; font-weight: 600; border: 1px solid var(--border);
  color: var(--ink);
  transition: all .15s ease;
}
.stButton > button:hover { border-color: var(--brand); }
.stDownloadButton > button {
  border-radius: 10px; font-weight: 600;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: var(--brand); border-color: var(--brand); color: #fff;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
  background: var(--brand-dark); border-color: var(--brand-dark);
}

/* Tablas */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
[data-testid="stFileUploaderDropzone"] {
  border-radius: 12px; border-width: 1px; border-style: dashed;
  border-color: var(--muted); background: var(--bg-soft);
}
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input { color: var(--ink); }
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def clear_session() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def header() -> None:
    st.markdown(
        """
        <div class="app-banner">
            <h1>🔎 Cruce SOT ↔ Bitácoras</h1>
            <p>Fecha estricta · identificador exacto · similitud semántica.
            El agente reconoce los archivos por su contenido; los casos dudosos quedan en PENDIENTE.</p>
            <span class="pill">DETECCIÓN DINÁMICA DE ARCHIVOS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stepper(active: int) -> None:
    steps = ["Carga de archivos", "Ejecutar cruce", "Resultado"]
    classes = []
    for index in range(3):
        if index < active:
            classes.append("done")
        elif index == active:
            classes.append("active")
        else:
            classes.append("")
    html = '<div class="steps">' + "".join(
        f'<div class="step {cls}"><span class="num">{"✓" if cls == "done" else index + 1}</span>{label}</div>'
        for index, (label, cls) in enumerate(zip(steps, classes))
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def sidebar() -> MatchingSettings:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="t">⚙️ Panel de configuración</div>
                <div class="s">Ajustes del cruce inteligente</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        threshold = st.slider("Umbral semántico", min_value=0.70, max_value=0.95, value=0.78, step=0.01)
        st.caption("Exige 0.03 de diferencia frente al segundo candidato para evitar asignaciones ambiguas.")
        use_column_dates = st.checkbox(
            "Incluir fecha de columnas (fecha_instalacion, fechageneracionsot…)", value=True
        )
        st.divider()
        if st.button("🔄 Nueva carga", width="stretch"):
            clear_session()
            st.rerun()
        st.caption("Versión 2.1 · arquitectura en capas")
    st.session_state["use_column_dates"] = use_column_dates
    return MatchingSettings(threshold=threshold)


def metric_row(metrics: list[tuple[str, str, str]]) -> None:
    """``metrics``: lista de (label, value, variante css)."""
    columns = st.columns(len(metrics))
    for column, (label, value, variant) in zip(columns, metrics):
        with column:
            st.markdown(
                f'<div class="metric-card {variant}"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>',
                unsafe_allow_html=True,
            )


def step_load() -> list | None:
    st.subheader("Carga de archivos")
    uploaded = st.file_uploader(
        "Arrastra los **dos** archivos `.xlsx` (Registro SOT y Reporte de actividades). El orden no importa.",
        type=["xlsx", "xlsm"],
        accept_multiple_files=True,
        key="uploads",
    )
    if not uploaded:
        st.markdown(
            '<div class="callout"><strong>Sin archivos aún.</strong> '
            "Sube el Registro de SOTs y el Reporte/INDICADORES de actividades; "
            "el sistema los reconocerá automáticamente por su contenido.</div>",
            unsafe_allow_html=True,
        )
        return None
    if len(uploaded) != 2:
        st.warning(f"Se esperan exactamente 2 archivos. Recibidos: **{len(uploaded)}**.")
        return None

    profiles = PIPELINE.classify([(file.name, file.getvalue()) for file in uploaded])
    filenames = [file.name for file in uploaded]
    detected_sot = next(profile for profile in profiles if profile.role == "sot")
    detected_log = next(profile for profile in profiles if profile.role == "bitacora")

    stored_sot = st.session_state.get("sot_name")
    if stored_sot and (stored_sot not in filenames or st.session_state.get("log_name") not in filenames):
        for key in ("prepared", "results", "elapsed"):
            st.session_state.pop(key, None)

    left, right = st.columns(2)
    with left:
        st.markdown("**📘 Registro de SOT**")
        sot_name = st.selectbox("Archivo SOT", filenames, index=filenames.index(detected_sot.name), key="asign_sot")
        sot_profile = next(p for p in profiles if p.name == sot_name)
        sot_sheets = list(sot_profile.meta.keys())
        default_sot_sheet = sot_profile.sheet if sot_profile.sheet in sot_sheets else sot_sheets[0]
        sot_sheet = st.selectbox("Hoja SOT", sot_sheets, index=sot_sheets.index(default_sot_sheet), key="sheet_sot")
    with right:
        st.markdown("**📗 Reporte de bitácoras**")
        log_name = st.selectbox("Archivo bitácoras", filenames, index=filenames.index(detected_log.name), key="asign_log")
        log_profile = next(p for p in profiles if p.name == log_name)
        log_sheets = list(log_profile.meta.keys())
        default_log_sheet = log_profile.sheet if log_profile.sheet in log_sheets else log_sheets[0]
        log_sheet = st.selectbox("Hoja bitácoras", log_sheets, index=log_sheets.index(default_log_sheet), key="sheet_log")

    if sot_name == log_name:
        st.error("Elegiste el mismo archivo para ambos roles: ajústalo en los selectores.")
        return None

    files_by_name = {file.name: file for file in uploaded}
    if st.button("✓ Validar estructura", type="primary"):
        try:
            with st.status("Detectando hojas y preparando registros…", expanded=True) as status:
                prepared = PIPELINE.prepare(
                    files_by_name[sot_name].getvalue(),
                    files_by_name[log_name].getvalue(),
                    use_column_dates=st.session_state.get("use_column_dates", True),
                    sot_sheet=sot_sheet,
                    log_sheet=log_sheet,
                )
                status.update(label="Estructura validada.", state="complete")
            st.session_state["prepared"] = prepared
            st.session_state["sot_name"] = sot_name
            st.session_state["log_name"] = log_name
        except (InputValidationError, ValueError, KeyError) as error:
            st.error(f"No se pudieron validar los archivos: {error}")

    prepared = st.session_state.get("prepared")
    if prepared:
        st.success(
            f"Listos: **{len(prepared.sots):,} SOTs** (hoja `{prepared.sot_sheet}`) y "
            f"**{len(prepared.logs):,} bitácoras** (hoja `{prepared.log_sheet}`)."
        )
    return uploaded


def step_cross() -> None:
    st.subheader("Ejecutar cruce")
    prepared = st.session_state.get("prepared")
    ready = prepared is not None
    if not ready:
        st.markdown(
            '<div class="callout"><strong>Aún no hay estructura validada.</strong> '
            "Completa el paso 1 y valida los archivos para poder cruzar.</div>",
            unsafe_allow_html=True,
        )
        return
    st.write(
        f"Archivo **SOT** (`{prepared.sot_sheet}`) cruzado contra las "
        f"**{len(prepared.logs):,} bitácoras** de `{prepared.log_sheet}`."
    )
    if st.button("⚡ Iniciar cruce", type="primary", disabled=not ready):
        started = time.perf_counter()
        try:
            results_holder = st.empty()
            progress = st.progress(45, text="Calculando coincidencias…")
            results = PIPELINE.cross(
                prepared,
                settings=st.session_state.get("settings"),
                on_progress=lambda pct, msg: progress.progress(pct, text=msg),
            )
            elapsed = time.perf_counter() - started
            progress.progress(100, text="Terminado.")
            st.session_state["results"] = results.results
            st.session_state["elapsed"] = elapsed
            results_holder.success(
                f"Cruce listo en **{elapsed:.1f}s**: {int((results.results['estado'] != 'Pendiente').sum()):,} asignados."
            )
        except Exception as error:  # noqa: BLE001 - errores del modelo/cruce a la UI
            st.error(f"El procesamiento falló: {error}")


def render_table(data: pd.DataFrame) -> None:
    """Renderiza una tabla con chips de estado coloreados y formato profesional."""
    view = data.rename(columns={"sot": "SOT", "bitacoras": "Bitácora", "materias": "Materiales", "confianza": "Confianza", "estado": "Estado"})
    styled = (
        view[["SOT", "Bitácora", "Materiales", "Confianza", "Estado"]]
        .style.format({"Confianza": "{:.3f}"}, na_rep="—", thousands=",")
        .map(
            lambda value: f"background-color: {STATUS_BG.get(value, '#f1f3f4')}; "
            f"color: {STATUS_COLORS.get(value, '#475569')}; font-weight: 600; border-radius: 999px;"
            if value in STATUS_COLORS
            else "",
            subset=["Estado"],
        )
    )
    st.dataframe(styled, width="stretch", hide_index=True)


def step_result() -> None:
    st.subheader("Resultado del cruce")
    results = st.session_state.get("results")
    if results is None:
        st.markdown(
            '<div class="callout"><strong>Sin resultados aún.</strong> '
            "Ejecuta el cruce en el paso 2: aquí verás el detalle y las descargas.</div>",
            unsafe_allow_html=True,
        )
        return

    assigned = int((results["estado"] != "Pendiente").sum())
    pending = len(results) - assigned
    with_materials = int((results["materias"] == "si").sum())

    metric_row(
        [
            ("Total de SOTs", f"{len(results):,}", ""),
            ("Asignados", f"{assigned:,}", "green"),
            ("Pendientes", f"{pending:,}", "amber"),
            ("Con materiales", f"{with_materials:,}", ""),
        ]
    )

    st.markdown(
        '<div class="legend">'
        '<span class="chip exacto">✅ Exacto</span>'
        '<span class="chip semantico">🧠 Semántico</span>'
        '<span class="chip pendiente">⏸ Pendiente</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    mode = st.radio("Mostrar", ["Todos", "Asignados", "Pendientes"], horizontal=True)
    mask = (
        pd.Series(True, index=results.index)
        if mode == "Todos"
        else results["estado"] != "Pendiente"
        if mode == "Asignados"
        else results["estado"] == "Pendiente"
    )
    render_table(results.loc[mask])

    elapsed = st.session_state.get("elapsed")
    if elapsed is not None:
        st.caption(f"⏱️ Tiempo de cruce: {elapsed:.1f}s")
    st.download_button(
        "⬇️ Descargar Excel (sot | bitacoras | materias)",
        data=to_excel(results),
        file_name="cruce_sot_bitacoras.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


def step_massive_search(results: pd.DataFrame) -> None:
    """Filtra el resultado sobre una lista explícita de SOTs y permite descargarla."""
    st.divider()
    st.subheader("🔎 Búsqueda masiva por lista")
    st.caption(
        "Pega una lista de SOTs (se separan por salto de línea, coma o espacio) o súbela "
        "desde un `.txt`/`.xlsx`/`.csv`. Los ID se leen por contenido y se deduplican."
    )

    uploaded = st.file_uploader(
        "Lista en archivo (opcional)", type=["txt", "xlsx", "xlsm", "csv"], key="list_upload"
    )
    pasted = st.text_area(
        "O pega los SOTs aquí",
        height=140,
        key="pasted_ids",
        placeholder="90681747 90073467\n90073863 …",
    )
    if uploaded is not None:
        ids = ids_from_file(uploaded.getvalue(), uploaded.name)
    elif pasted.strip():
        ids = parse_sot_ids([pasted])
    else:
        st.info("Aún no hay lista: los botones de abajo usarán el cruce completo.")
        return

    summary = summarize(results, ids)
    metric_row(
        [
            ("Solicitadas", f"{summary['solicitadas']:,}", ""),
            ("En el cruce", f"{summary['en_cruce']:,}", ""),
            ("Encontradas", f"{summary['encontradas']:,}", "green"),
            ("Pendientes", f"{summary['pendientes']:,}", "amber"),
            ("Con materiales", f"{summary['materias']:,}", ""),
        ]
    )

    only_found = st.checkbox("Solo encontradas (excluir Pendiente)", value=True)
    subset = filter_results(results, ids, only_found=only_found)
    st.write(f"**{len(subset):,}** filas coinciden con la lista.")
    if not subset.empty:
        render_table(subset)
    st.download_button(
        "⬇️ Descargar Excel filtrado por la lista",
        data=to_excel(subset),
        file_name="cruce_lista_sot.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


def main() -> None:
    inject_theme()
    settings = sidebar()
    st.session_state["settings"] = settings
    header()

    results = st.session_state.get("results")
    prepared = st.session_state.get("prepared")
    active = 0 if prepared is None else 1 if results is None else 2
    stepper(active)

    step_load()
    st.divider()
    step_cross()
    st.divider()
    step_result()
    if results is not None:
        step_massive_search(results)


if __name__ == "__main__":
    main()