"""Página de gestión de modelos supervisados (HU-07)."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from sot_matcher.application.ports import ModelRegistry
from sot_matcher.domain.feedback import FeedbackDecision


def render_models_page(model_registry: ModelRegistry) -> None:
    """Página para evaluar, activar y revertir modelos."""
    st.title("🤖 Modelos supervisados")
    st.caption("Gestiona versiones de modelos CatBoost entrenados con feedback humano.")

    # Modelo activo actual
    active = model_registry.get_active_model()
    if active:
        st.success(f"🟢 **Modelo activo:** `{active.model_version}`")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precision", f"{active.metrics.get('precision', 0):.3f}")
        with col2:
            st.metric("Recall", f"{active.metrics.get('recall', 0):.3f}")
        with col3:
            st.metric("F1", f"{active.metrics.get('f1', 0):.3f}")
        st.caption(f"Entrenado: {active.created_at} | Features: v{active.feature_schema_version} | Data: {active.training_dataset_fingerprint}")
    else:
        st.warning("⚠️ No hay modelo activo. El motor usa solo matching híbrido (exacto + semántico).")

    st.divider()

    # Lista de todos los modelos
    st.subheader("📦 Registro de modelos")
    models = model_registry.list_models()

    if not models:
        st.info("No hay modelos registrados. Entrena uno desde la línea de comandos:")
        st.code("python -m sot_matcher.cli train --version 1.0")
        return

    # Tabla de modelos
    df = pd.DataFrame([{
        "Versión": m.model_version,
        "Estado": m.status,
        "Precision": f"{m.metrics.get('precision', 0):.3f}",
        "Recall": f"{m.metrics.get('recall', 0):.3f}",
        "F1": f"{m.metrics.get('f1', 0):.3f}",
        "ROC-AUC": f"{m.metrics.get('roc_auc', 0):.3f}",
        "Creado": m.created_at[:16].replace("T", " "),
        "Features": f"v{m.feature_schema_version}",
        "Data": m.training_dataset_fingerprint,
    } for m in models])

    # Colorear por estado
    def style_status(val):
        colors = {
            "ACTIVE": "background-color: #dcfce7; color: #166534; font-weight: bold",
            "APPROVED": "background-color: #dbeafe; color: #1e40af",
            "CANDIDATE": "background-color: #fef3c7; color: #92400e",
            "REJECTED": "background-color: #fee2e2; color: #991b1b",
            "RETIRED": "background-color: #f3f4f6; color: #6b7280",
        }
        return colors.get(val, "")

    styled = df.style.map(style_status, subset=["Estado"])
    st.dataframe(styled, width="stretch", hide_index=True)

    st.divider()

    # Acciones por modelo
    st.subheader("⚙️ Acciones")
    model_versions = [m.model_version for m in models]
    selected_version = st.selectbox("Seleccionar modelo", model_versions)

    if selected_version:
        selected = next(m for m in models if m.model_version == selected_version)
        _render_model_actions(model_registry, selected)


def _render_model_actions(model_registry: ModelRegistry, model) -> None:
    """Renderiza acciones disponibles para un modelo."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if model.status == "CANDIDATE":
            if st.button("✅ Aprobar para evaluación", type="primary", use_container_width=True):
                # En implementación real: activar tras evaluación
                model_registry.activate(model.model_version, "APPROVED")
                st.success(f"Modelo {model.model_version} marcado como APPROVED")
                st.rerun()

    with col2:
        if model.status in ("APPROVED", "CANDIDATE"):
            if st.button("🚀 Activar en producción", type="primary", use_container_width=True):
                if st.checkbox("Confirmo activar este modelo", key=f"confirm_activate_{model.model_version}"):
                    model_registry.activate(model.model_version, "APPROVED")
                    st.success(f"Modelo {model.model_version} ACTIVADO")
                    st.rerun()

    with col3:
        if model.status == "ACTIVE":
            if st.button("🔄 Rollback a anterior", use_container_width=True):
                # Buscar modelo RETIRED previo
                all_models = model_registry.list_models()
                retired = [m for m in all_models if m.status == "RETIRED"]
                if retired:
                    prev = retired[0]
                    if st.checkbox(f"Confirmo rollback a {prev.model_version}", key=f"confirm_rollback_{model.model_version}"):
                        model_registry.rollback(prev.model_version)
                        st.success(f"Rollback a {prev.model_version} completado")
                        st.rerun()
                else:
                    st.warning("No hay modelo previo para rollback")

    # Detalles del modelo seleccionado
    with st.expander("📊 Detalles completos", expanded=False):
        st.json({
            "model_version": model.model_version,
            "artifact_path": model.artifact_path,
            "feature_schema_version": model.feature_schema_version,
            "created_at": model.created_at,
            "metrics": model.metrics,
            "training_dataset_fingerprint": model.training_dataset_fingerprint,
            "status": model.status,
            "parent_model_version": model.parent_model_version,
        })

    # Evaluación comparativa (placeholder)
    st.divider()
    st.subheader("📈 Evaluación comparativa")
    st.info("Ejecuta evaluación en sombra para comparar contra motor actual:")
    st.code(f"python -m sot_matcher.cli evaluate --model {model.model_version}")