"""Página de revisión de asociaciones (HU-04)."""

from __future__ import annotations

import streamlit as st

from sot_matcher.presentation.components.review_queue import render_review_queue
from sot_matcher.application.feedback_service import FeedbackService


def render_review_page(
    feedback_service: FeedbackService,
    results_df,
    prepared_sots,
    prepared_logs,
) -> None:
    """Página completa de revisión."""
    st.title("👁️ Revisión de asociaciones SOT ↔ Bitácoras")
    st.caption("Valida, corrige o rechaza las asociaciones propuestas por el motor de matching.")

    # Configuración del revisor
    _render_reviewer_config()

    # Métricas de la cola
    _render_queue_metrics(feedback_service, results_df, prepared_sots, prepared_logs)

    st.divider()

    # Procesar feedback pendiente del session_state
    _process_pending_feedback(feedback_service)

    # Renderizar cola de revisión
    render_review_queue(feedback_service, results_df, prepared_sots, prepared_logs)


def _render_reviewer_config() -> None:
    """Configuración del revisor en sidebar."""
    with st.sidebar:
        st.divider()
        st.markdown("### 👤 Configuración de revisor")
        reviewer_id = st.text_input(
            "Tu ID / nombre",
            value=st.session_state.get("reviewer_id", ""),
            placeholder="ej. analista_01",
            key="reviewer_id_input",
        )
        if reviewer_id:
            st.session_state["reviewer_id"] = reviewer_id
            st.caption(f"Revisor: **{reviewer_id}**")
        else:
            st.caption("⚠️ Sin revisor: el feedback se guardará sin autor")


def _render_queue_metrics(
    feedback_service: FeedbackService,
    results_df,
    prepared_sots,
    prepared_logs,
) -> None:
    """Métricas resumen de la cola de revisión."""
    items = feedback_service.get_review_queue(results_df, prepared_sots, prepared_logs)

    if not items:
        st.info("✅ No hay elementos pendientes de revisión.")
        return

    total = len(items)
    pendientes = sum(1 for i in items if i.estado_actual == "Pendiente")
    semantic = sum(1 for i in items if i.estado_actual == "Semántico")
    exactos = sum(1 for i in items if i.estado_actual == "Exacto")
    high_priority = sum(1 for i in items if i.priority >= 80)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total en cola", total)
    with col2:
        st.metric("🟡 Pendientes", pendientes, delta=f"{pendientes/total*100:.0f}%")
    with col3:
        st.metric("🔵 Semánticos", semantic, delta=f"{semantic/total*100:.0f}%")
    with col4:
        st.metric("🟢 Exactos", exactos, delta=f"{exactos/total*100:.0f}%")
    with col5:
        st.metric("🔴 Alta prioridad", high_priority)


def _process_pending_feedback(feedback_service: FeedbackService) -> None:
    """Procesa comandos de feedback guardados en session_state."""
    keys_to_process = [k for k in st.session_state.keys() if k.startswith("pending_feedback_")]
    for key in keys_to_process:
        command = st.session_state.pop(key)
        # Añadir reviewer_id si está configurado
        if "reviewer_id" in st.session_state and st.session_state["reviewer_id"]:
            from sot_matcher.application.feedback_service import SubmitDecisionCommand
            command = SubmitDecisionCommand(
                sot_id=command.sot_id,
                log_id=command.log_id,
                decision=command.decision,
                run_id=command.run_id,
                reason_code=command.reason_code,
                comment=command.comment,
                reviewer_id=st.session_state["reviewer_id"],
                supersedes_feedback_id=command.supersedes_feedback_id,
                feature_schema_version=command.feature_schema_version,
            )
        try:
            feedback_service.submit_decision(command)
            st.toast(f"✅ Feedback guardado: {command.sot_id} ↔ {command.log_id} = {command.decision.value}")
        except Exception as e:
            st.toast(f"❌ Error guardando feedback: {e}")