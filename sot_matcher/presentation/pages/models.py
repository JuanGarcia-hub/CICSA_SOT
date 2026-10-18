"""Página de gestión de modelos supervisados (HU-07)."""

from __future__ import annotations

import json
import streamlit as st
import pandas as pd

from sot_matcher.application.ports import ModelRegistry
from sot_matcher.domain.feedback import FeedbackDecision


def render_models_page(model_registry: ModelRegistry) -> None:
    """Página para evaluar, activar y revertir modelos."""
    st.title("🤖 Modelos supervisados")
    st.caption("Gestiona versiones de modelos CatBoost entrenados con feedback humano.")

    # Inicializar session state para confirmaciones
    if "model_action_confirm" not in st.session_state:
        st.session_state.model_action_confirm = {}

    # Modelo activo actual
    active = model_registry.get_active_model()
    _render_active_model_banner(active)

    st.divider()

    # Lista de todos los modelos
    st.subheader("📦 Registro de modelos")
    models = model_registry.list_models()

    if not models:
        st.info("No hay modelos registrados. Entrena uno desde la línea de comandos:")
        st.code("python -m sot_matcher.cli train --version 1.0 --sot-file SOT.xlsx --log-file LOGS.xlsx")
        return

    # Tabla de modelos
    _render_models_table(models)

    st.divider()

    # Acciones por modelo
    st.subheader("⚙️ Acciones")
    _render_model_actions(model_registry, models)

    st.divider()

    # Evaluación comparativa con shadow evaluation
    _render_shadow_evaluation_section(model_registry, models)


def _render_active_model_banner(active) -> None:
    """Banner del modelo activo con métricas clave."""
    if active:
        with st.container(border=True):
            st.success(f"🟢 **Modelo activo en producción:** `{active.model_version}`")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precision", f"{active.metrics.get('precision', 0):.3f}")
            with col2:
                st.metric("Recall", f"{active.metrics.get('recall', 0):.3f}")
            with col3:
                st.metric("F1", f"{active.metrics.get('f1', 0):.3f}")
            with col4:
                st.metric("ROC-AUC", f"{active.metrics.get('roc_auc', 0):.3f}")

            st.caption(
                f"📅 Entrenado: {active.created_at[:16].replace('T', ' ')} | "
                f"🔧 Features: v{active.feature_schema_version} | "
                f"📊 Data fingerprint: {active.training_dataset_fingerprint} | "
                f"📈 Test size: {active.metrics.get('test_size', 'N/A')}"
            )

            # Botón rápido de rollback
            if st.button("🔄 Rollback a versión anterior", type="secondary", use_container_width=True):
                _confirm_rollback(model_registry, active)
    else:
        st.warning("⚠️ **No hay modelo activo.** El motor usa solo matching híbrido (exacto + semántico).")
        st.info("💡 Entrena y activa un modelo supervisado para mejorar la precisión del matching.")


def _render_models_table(models) -> None:
    """Tabla estilizada del registro de modelos."""
    df = pd.DataFrame([{
        "Versión": m.model_version,
        "Estado": m.status,
        "Precision": f"{m.metrics.get('precision', 0):.3f}",
        "Recall": f"{m.metrics.get('recall', 0):.3f}",
        "F1": f"{m.metrics.get('f1', 0):.3f}",
        "ROC-AUC": f"{m.metrics.get('roc_auc', 0):.3f}",
        "Brier": f"{m.metrics.get('brier_score', 0):.3f}" if 'brier_score' in m.metrics else "—",
        "ECE": f"{m.metrics.get('ece', 0):.4f}" if 'ece' in m.metrics else "—",
        "Creado": m.created_at[:16].replace("T", " "),
        "Features": f"v{m.feature_schema_version}",
        "Data": m.training_dataset_fingerprint,
    } for m in models])

    def style_status(val):
        colors = {
            "ACTIVE": "background-color: #dcfce7; color: #166534; font-weight: bold",
            "APPROVED": "background-color: #dbeafe; color: #1e40af",
            "CANDIDATE": "background-color: #fef3c7; color: #92400e",
            "REJECTED": "background-color: #fee2e2; color: #991b1b",
            "RETIRED": "background-color: #f3f4f6; color: #6b7280",
        }
        return colors.get(val, "")

    def style_metric(val):
        """Colorea métricas numéricas."""
        try:
            v = float(val)
            if v >= 0.85:
                return "color: #166534; font-weight: bold"
            elif v >= 0.70:
                return "color: #1d4ed8"
            elif v >= 0.50:
                return "color: #d97706"
            else:
                return "color: #991b1b"
        except (ValueError, TypeError):
            return ""

    styled = (
        df.style
        .map(style_status, subset=["Estado"])
        .map(style_metric, subset=["Precision", "Recall", "F1", "ROC-AUC", "Brier"])
    )
    st.dataframe(styled, width="stretch", hide_index=True, height=min(400, 35 * len(models) + 40))


def _render_model_actions(model_registry: ModelRegistry, models) -> None:
    """Acciones por modelo con confirmaciones robustas."""
    model_versions = [m.model_version for m in models]
    selected_version = st.selectbox(
        "Seleccionar modelo",
        model_versions,
        format_func=lambda v: f"📦 {v}",
    )

    if not selected_version:
        return

    selected = next(m for m in models if m.model_version == selected_version)

    # Mostrar detalles en expander
    with st.expander(f"📋 Detalles: {selected.model_version}", expanded=True):
        _render_model_details(selected)

    st.divider()

    # Acciones según estado
    if selected.status == "CANDIDATE":
        _render_candidate_actions(model_registry, selected)
    elif selected.status == "APPROVED":
        _render_approved_actions(model_registry, selected)
    elif selected.status == "ACTIVE":
        _render_active_actions(model_registry, selected)
    elif selected.status == "REJECTED":
        _render_rejected_actions(model_registry, selected)
    elif selected.status == "RETIRED":
        _render_retired_actions(model_registry, selected)


def _render_model_details(model) -> None:
    """Detalles completos del modelo."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Versión:** `{model.model_version}`")
        st.markdown(f"**Estado:** {_status_badge(model.status)}")
        st.markdown(f"**Creado:** {model.created_at[:19].replace('T', ' ')}")
        st.markdown(f"**Features schema:** v{model.feature_schema_version}")
        st.markdown(f"**Data fingerprint:** `{model.training_dataset_fingerprint}`")
    with col2:
        st.markdown(f"**Artifact:** `{model.artifact_path}`")
        st.markdown(f"**Parent:** {model.parent_model_version or '—'}")
        st.json(model.metrics)


def _status_badge(status: str) -> str:
    badges = {
        "ACTIVE": "🟢 **ACTIVE**",
        "APPROVED": "🔵 **APPROVED**",
        "CANDIDATE": "🟡 **CANDIDATE**",
        "REJECTED": "🔴 **REJECTED**",
        "RETIRED": "⚫ **RETIRED**",
    }
    return badges.get(status, status)


def _render_candidate_actions(model_registry: ModelRegistry, model) -> None:
    """Acciones para modelo CANDIDATE."""
    st.warning("⚠️ Modelo candidato: requiere evaluación antes de activar.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Evaluar (shadow-eval)", type="primary", use_container_width=True):
            st.info("Ejecuta en terminal:\n```bash\npython -m sot_matcher.cli shadow-eval " + model.model_version + " --sot-file SOT.xlsx --log-file LOGS.xlsx\n```")

    with col2:
        if st.button("❌ Rechazar", use_container_width=True):
            if _confirm_action("reject", model.model_version, "¿Rechazar este modelo? No se podrá activar."):
                _update_model_status(model_registry, model.model_version, "REJECTED")
                st.success("Modelo marcado como REJECTED")
                st.rerun()


def _render_approved_actions(model_registry: ModelRegistry, model) -> None:
    """Acciones para modelo APPROVED."""
    st.success("✅ Modelo aprobado: listo para activación en producción.")

    if st.button("🚀 Activar en producción", type="primary", use_container_width=True):
        if _confirm_action("activate", model.model_version,
                          "¿Activar este modelo en producción?\nEl modelo activo actual pasará a RETIRED."):
            # Guardar versión anterior para rollback
            active = model_registry.get_active_model()
            parent = active.model_version if active else None

            model_registry.activate(model.model_version, "APPROVED")

            # Actualizar parent_model_version
            if parent:
                with model_registry._conn() as conn:
                    conn.execute(
                        "UPDATE model_registry SET parent_model_version = ? WHERE model_version = ?",
                        (parent, model.model_version)
                    )

            st.success(f"🚀 Modelo `{model.model_version}` ACTIVADO en producción")
            st.balloons()
            st.rerun()


def _render_active_actions(model_registry: ModelRegistry, model) -> None:
    """Acciones para modelo ACTIVE."""
    st.error("🟢 **Este modelo está ACTIVO en producción**")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Ver shadow-eval reciente", use_container_width=True):
            st.session_state["show_shadow_eval"] = model.model_version
            st.rerun()

    with col2:
        if st.button("🔄 Rollback a versión anterior", type="secondary", use_container_width=True):
            _confirm_rollback(model_registry, model)


def _render_rejected_actions(model_registry: ModelRegistry, model) -> None:
    """Acciones para modelo REJECTED."""
    st.error("🔴 Modelo rechazado: no puede activarse.")

    if st.button("♻️ Restaurar a CANDIDATE", use_container_width=True):
        if _confirm_action("restore", model.model_version, "¿Restaurar a CANDIDATE para re-evaluación?"):
            _update_model_status(model_registry, model.model_version, "CANDIDATE")
            st.success("Modelo restaurado a CANDIDATE")
            st.rerun()


def _render_retired_actions(model_registry: ModelRegistry, model) -> None:
    """Acciones para modelo RETIRED (ex-activo)."""
    st.info("⚫ Modelo retirado: fue activo anteriormente.")

    if st.button("🔄 Rollback a esta versión", type="primary", use_container_width=True):
        if _confirm_action("rollback", model.model_version,
                          f"¿Hacer rollback a `{model.model_version}`?\nEl modelo activo actual pasará a RETIRED."):
            model_registry.rollback(model.model_version)
            st.success(f"✅ Rollback a `{model.model_version}` completado")
            st.rerun()


def _confirm_rollback(model_registry: ModelRegistry, active_model) -> None:
    """Modal de confirmación de rollback."""
    all_models = model_registry.list_models()
    retired = [m for m in all_models if m.status == "RETIRED"]

    if not retired:
        st.warning("⚠️ No hay modelo previo para rollback")
        return

    prev = retired[0]
    if _confirm_action("rollback", active_model.model_version,
                      f"¿Hacer rollback a `{prev.model_version}`?\n"
                      f"El modelo activo `{active_model.model_version}` pasará a RETIRED."):
        model_registry.rollback(prev.model_version)
        st.success(f"✅ Rollback a `{prev.model_version}` completado")
        st.rerun()


def _confirm_action(action: str, model_version: str, message: str) -> bool:
    """Sistema de confirmación en 2 pasos usando session_state."""
    key = f"confirm_{action}_{model_version}"

    if key not in st.session_state.model_action_confirm:
        st.session_state.model_action_confirm[key] = False

    if not st.session_state.model_action_confirm[key]:
        if st.button(f"⚠️ {message}", key=f"btn_{key}", use_container_width=True):
            st.session_state.model_action_confirm[key] = True
            st.rerun()
        return False

    # Paso 2: confirmación explícita
    st.warning(message)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirmo", type="primary", key=f"yes_{key}", use_container_width=True):
            st.session_state.model_action_confirm[key] = False
            return True
    with col2:
        if st.button("❌ Cancelar", key=f"no_{key}", use_container_width=True):
            st.session_state.model_action_confirm[key] = False
            st.rerun()
    return False


def _update_model_status(model_registry: ModelRegistry, model_version: str, new_status: str) -> None:
    """Actualiza estado directamente en BD."""
    with model_registry._conn() as conn:
        conn.execute(
            "UPDATE model_registry SET status = ? WHERE model_version = ?",
            (new_status, model_version)
        )


def _render_shadow_evaluation_section(model_registry: ModelRegistry, models) -> None:
    """Sección de evaluación en sombra con resultados."""
    st.subheader("🌑 Evaluación en sombra (Shadow Evaluation)")

    # Selector de modelo para shadow eval
    eval_versions = [m.model_version for m in models if m.status in ("CANDIDATE", "APPROVED", "ACTIVE")]
    if not eval_versions:
        st.info("No hay modelos evaluables (requiere status CANDIDATE, APPROVED o ACTIVE)")
        return

    # Ver si hay shadow eval guardado en session_state
    if "show_shadow_eval" in st.session_state:
        eval_model = st.session_state.pop("show_shadow_eval")
    else:
        eval_model = st.selectbox(
            "Modelo a evaluar en sombra",
            eval_versions,
            format_func=lambda v: f"📊 {v}",
            key="shadow_eval_select",
        )

    if st.button("▶️ Ejecutar Shadow Evaluation", type="primary", use_container_width=True):
        st.info("""
        La shadow evaluation requiere archivos de datos reales.
        Ejecuta en terminal:
        ```bash
        python -m sot_matcher.cli shadow-eval {model} \\
            --sot-file SOT.xlsx --log-file LOGS.xlsx \\
            --output shadow_report.csv
        ```
        """.format(model=eval_model))

    st.caption("💡 La shadow evaluation compara el clasificador vs el motor actual en casos con feedback humano.")


# Backward compatibility
def _render_model_actions(model_registry: ModelRegistry, model) -> None:
    """Wrapper para compatibilidad."""
    models = model_registry.list_models()
    _render_model_actions(model_registry, models)