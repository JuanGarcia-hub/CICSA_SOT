"""Formulario de feedback reutilizable (HU-01, HU-02, HU-03)."""

from __future__ import annotations

import streamlit as st

from sot_matcher.domain.feedback import FeedbackDecision
from sot_matcher.application.feedback_service import SubmitDecisionCommand


def render_feedback_form(
    sot_id: str,
    log_id: str,
    run_id: str | None,
    key_suffix: str,
    on_submit_callback=None,
) -> SubmitDecisionCommand | None:
    """Renderiza formulario de decisión y retorna comando si se envió."""
    with st.form(f"feedback_form_{key_suffix}"):
        decision = st.radio(
            "Decisión",
            [FeedbackDecision.CORRECT, FeedbackDecision.INCORRECT, FeedbackDecision.INCONCLUSIVE],
            format_func=lambda d: {
                FeedbackDecision.CORRECT: "✅ Correcto",
                FeedbackDecision.INCORRECT: "❌ Incorrecto",
                FeedbackDecision.INCONCLUSIVE: "⏸️ No concluyente",
            }[d],
            index=2,
            horizontal=True,
            key=f"decision_{key_suffix}",
        )

        reason = st.selectbox(
            "Motivo",
            ["", "incidencia_coincide", "fecha_coincide", "descripcion_coincide",
             "materiales_coinciden", "falso_positivo_semantico", "otro"],
            key=f"reason_{key_suffix}",
        )

        comment = st.text_area(
            "Comentario",
            height=80,
            key=f"comment_{key_suffix}",
        )

        reviewer = st.text_input(
            "Revisor",
            key=f"reviewer_{key_suffix}",
        )

        submitted = st.form_submit_button("Guardar", type="primary", use_container_width=True)

        if submitted:
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
            if on_submit_callback:
                on_submit_callback(command)
            return command
    return None


def render_feedback_history(feedback_events: list) -> None:
    """Muestra historial de decisiones para un par."""
    if not feedback_events:
        st.caption("Sin feedback previo")
        return

    for fb in feedback_events:
        icon = {"CORRECT": "✅", "INCORRECT": "❌", "INCONCLUSIVE": "⏸️"}[fb.decision.value]
        with st.container(border=True):
            st.markdown(f"{icon} **{fb.decision.value}** — *{fb.created_at.strftime('%Y-%m-%d %H:%M')}*")
            if fb.reviewer_id:
                st.caption(f"👤 {fb.reviewer_id}")
            if fb.reason_code:
                st.caption(f"🏷️ {fb.reason_code}")
            if fb.comment:
                st.caption(f"💬 {fb.comment}")
            if fb.supersedes_feedback_id:
                st.caption(f"🔄 Corrige: {fb.supersedes_feedback_id[:8]}...")