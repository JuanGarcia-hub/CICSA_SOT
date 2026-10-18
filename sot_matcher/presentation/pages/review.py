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

    # Procesar feedback pendiente del session_state
    _process_pending_feedback(feedback_service)

    # Renderizar cola de revisión
    render_review_queue(feedback_service, results_df, prepared_sots, prepared_logs)


def _process_pending_feedback(feedback_service: FeedbackService) -> None:
    """Procesa comandos de feedback guardados en session_state."""
    keys_to_process = [k for k in st.session_state.keys() if k.startswith("pending_feedback_")]
    for key in keys_to_process:
        command = st.session_state.pop(key)
        try:
            feedback_service.submit_decision(command)
            st.toast(f"✅ Feedback guardado para {command.sot_id} ↔ {command.log_id}")
        except Exception as e:
            st.toast(f"❌ Error guardando feedback: {e}")