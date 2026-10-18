"""Componente Streamlit para cola de revisión (HU-04)."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from sot_matcher.application.feedback_service import FeedbackService, ReviewQueueItem
from sot_matcher.domain.feedback import FeedbackDecision


def render_review_queue(feedback_service: FeedbackService, results_df, prepared_sots, prepared_logs) -> None:
    st.subheader("📋 Cola de revisión")

    # Cargar cola
    items = feedback_service.get_review_queue(results_df, prepared_sots, prepared_logs)

    if not items:
        st.info("No hay elementos en la cola de revisión.")
        return

    # Métricas resumen
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total en cola", len(items))
    with col2:
        pendientes = sum(1 for i in items if i.estado_actual == "Pendiente")
        st.metric("Pendientes", pendientes)
    with col3:
        semantic = sum(1 for i in items if i.estado_actual == "Semántico")
        st.metric("Semánticos", semantic)
    with col4:
        exactos = sum(1 for i in items if i.estado_actual == "Exacto")
        st.metric("Exactos", exactos)

    # Filtros
    with st.expander("🔍 Filtros", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            estado_filter = st.multiselect(
                "Estado",
                ["Pendiente", "Semántico", "Exacto"],
                default=["Pendiente", "Semántico", "Exacto"],
            )
        with filter_col2:
            min_priority = st.slider("Prioridad mínima", 0, 100, 0)
        with filter_col3:
            show_only_with_feedback = st.checkbox("Solo con feedback previo", False)

    # Aplicar filtros
    filtered_items = [
        i for i in items
        if i.estado_actual in estado_filter and i.priority >= min_priority
    ]

    if not filtered_items:
        st.info("No hay elementos que coincidan con los filtros.")
        return

    # Tabla de cola
    df = pd.DataFrame([{
        "SOT": item.sot_id,
        "Estado": item.estado_actual,
        "Candidatos": item.candidate_count,
        "Match exacto": "✅" if item.has_exact_match else "❌",
        "Cand. semánticos": "✅" if item.has_semantic_candidates else "❌",
        "Fecha coincide": "✅" if item.date_match else "❌",
        "Prioridad": item.priority,
    } for item in filtered_items])

    # Configurar estilo de la tabla
    def highlight_priority(val):
        if val >= 80:
            return "background-color: #fee2e2; color: #991b1b; font-weight: bold"
        elif val >= 50:
            return "background-color: #fef3c7; color: #92400e; font-weight: bold"
        elif val > 0:
            return "background-color: #dcfce7; color: #166534"
        return ""

    styled = df.style.map(highlight_priority, subset=["Prioridad"])
    st.dataframe(styled, width="stretch", hide_index=True, height=400)

    # Seleccionar SOT para revisar
    st.divider()
    sot_options = [""] + [item.sot_id for item in filtered_items]
    selected_sot = st.selectbox(
        "Seleccionar SOT para revisar candidatos",
        sot_options,
        format_func=lambda x: f"📋 {x}" if x else "— Seleccionar —",
    )

    if selected_sot:
        st.session_state["selected_review_sot"] = selected_sot
        render_candidate_review(feedback_service, selected_sot, results_df, prepared_sots, prepared_logs)


def render_candidate_review(
    feedback_service: FeedbackService,
    sot_id: str,
    results_df,
    prepared_sots,
    prepared_logs,
) -> None:
    st.markdown(f"### 🔍 Revisión detallada: SOT `{sot_id}`")

    # Obtener registro SOT
    sot_row = prepared_sots[prepared_sots["sot"] == sot_id]
    if sot_row.empty:
        st.error("SOT no encontrada en registros preparados.")
        return
    sot_record = sot_row.iloc[0].to_dict()

    # Obtener bitácoras del mismo día
    sot_date = sot_record.get("date_hint")
    if sot_date is not None:
        log_records = prepared_logs[prepared_logs["date"] == sot_date].to_dict("records")
    else:
        log_records = prepared_logs.to_dict("records")

    # Obtener resultado actual
    result_row = results_df[results_df["sot"] == sot_id]
    current_estado = "Pendiente"
    current_bitacora = "—"
    current_conf = None
    if not result_row.empty:
        current_estado = str(result_row.iloc[0].get("estado", "Pendiente"))
        current_bitacora = str(result_row.iloc[0].get("bitacoras", "—"))
        current_conf = result_row.iloc[0].get("confianza")

    # Mostrar info SOT
    with st.container(border=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**SOT:** `{sot_id}`")
            st.markdown(f"**Fecha objetivo:** {sot_date or 'No detectada'}")
            st.markdown(f"**Texto SOT:** {sot_record.get('source_text', '')[:300]}{'...' if len(str(sot_record.get('source_text', ''))) > 300 else ''}")
        with col2:
            estado_color = {"Exacto": "🟢", "Semántico": "🔵", "Pendiente": "🟡"}.get(current_estado, "⚪")
            st.metric("Estado actual", f"{estado_color} {current_estado}")
            if current_conf is not None:
                st.metric("Confianza", f"{current_conf:.3f}")
            st.markdown(f"**Bitácora asignada:** `{current_bitacora[:80]}{'...' if len(current_bitacora) > 80 else ''}`")

    # Obtener candidatos enriquecidos
    review = feedback_service.get_candidate_details(sot_id, sot_record, log_records)

    if not review.candidates:
        st.warning("No hay candidatos para esta SOT en la fecha objetivo.")
        return

    st.markdown("#### Candidatos disponibles")

    for idx, cand in enumerate(review.candidates):
        # Buscar registro de bitácora completo
        log_rec = next((lr for lr in log_records if str(lr.get("record_id", "")) == cand.log_id), None)
        if not log_rec:
            continue

        with st.expander(f"#{idx+1} Bitácora `{cand.log_id}` — Similitud: {cand.semantic_score:.3f}" if cand.semantic_score else f"#{idx+1} Bitácora `{cand.log_id}`"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Texto bitácora:** {log_rec.get('log_text', '')[:500]}")
                if log_rec.get("observation"):
                    st.markdown(f"**Observación:** {log_rec.get('observation')[:300]}")
                if log_rec.get("materials_data"):
                    st.markdown(f"**Materiales:** {log_rec.get('materials_data')[:300]}")

                # Mostrar características si existen
                if cand.features:
                    st.markdown("**Evidencia:**")
                    feat_cols = st.columns(3)
                    feat_items = list(cand.features.items())
                    for i, (k, v) in enumerate(feat_items):
                        with feat_cols[i % 3]:
                            if isinstance(v, float):
                                st.metric(k.replace("_", " ").title(), f"{v:.3f}")
                            else:
                                st.metric(k.replace("_", " ").title(), v)

            with col2:
                # Feedback existente para este par
                existing_fb = [fb for fb in review.existing_feedback if fb.log_id == cand.log_id]
                if existing_fb:
                    latest = existing_fb[-1]
                    decision_icon = {"CORRECT": "✅", "INCORRECT": "❌", "INCONCLUSIVE": "⏸️"}[latest.decision.value]
                    st.info(f"Feedback previo: {decision_icon} {latest.decision.value}")
                    if latest.comment:
                        st.caption(f"💬 {latest.comment}")
                    if latest.reason_code:
                        st.caption(f"🏷️ {latest.reason_code}")

                # Formulario de decisión
                render_feedback_form(
                    sot_id=sot_id,
                    log_id=cand.log_id,
                    run_id=st.session_state.get("current_run_id"),
                    key_suffix=f"{sot_id}_{cand.log_id}",
                )


def render_feedback_form(
    sot_id: str,
    log_id: str,
    run_id: str | None,
    key_suffix: str,
) -> None:
    """Formulario para HU-01/02/03 - Registrar decisión."""
    with st.form(f"feedback_form_{key_suffix}"):
        decision = st.radio(
            "Tu decisión",
            [FeedbackDecision.CORRECT, FeedbackDecision.INCORRECT, FeedbackDecision.INCONCLUSIVE],
            format_func=lambda d: {
                FeedbackDecision.CORRECT: "✅ Correcto — La bitácora corresponde a esta SOT",
                FeedbackDecision.INCORRECT: "❌ Incorrecto — La bitácora NO corresponde",
                FeedbackDecision.INCONCLUSIVE: "⏸️ No concluyente — No hay evidencia suficiente",
            }[d],
            index=2,  # default INCONCLUSIVE
            horizontal=False,
            key=f"decision_{key_suffix}",
        )

        reason = st.selectbox(
            "Motivo (opcional)",
            ["", "incidencia_coincide", "fecha_coincide", "descripcion_coincide",
             "materiales_coinciden", "falso_positivo_semantico", "otro"],
            key=f"reason_{key_suffix}",
        )

        comment = st.text_area(
            "Comentario (opcional)",
            height=80,
            placeholder="Detalles adicionales...",
            key=f"comment_{key_suffix}",
        )

        reviewer = st.text_input(
            "Tu ID / nombre (opcional)",
            placeholder="ej. analista_01",
            key=f"reviewer_{key_suffix}",
        )

        submitted = st.form_submit_button(
            "💾 Guardar decisión",
            type="primary",
            use_container_width=True,
        )

        if submitted:
            from sot_matcher.application.feedback_service import SubmitDecisionCommand
            command = SubmitDecisionCommand(
                sot_id=sot_id,
                log_id=log_id,
                decision=decision,
                run_id=run_id,
                reason_code=reason or None,
                comment=comment or None,
                reviewer_id=reviewer or None,
                feature_schema_version="1.0",
            )
            # Guardar en session_state para procesar después del rerun
            st.session_state[f"pending_feedback_{key_suffix}"] = command
            st.success("Decisión guardada. Procesando...")
            st.rerun()