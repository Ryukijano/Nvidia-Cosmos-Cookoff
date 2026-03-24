"""Cosmos Sentinel Traffic Safety — Streamlit Page Module.

Ported from hf_space_repo/app.py (Gradio) to Streamlit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cosmos_sentinel_backend import (
    PREDICT_MODEL_NAME,
    cache_uploaded_video,
    ensure_sample_video,
    make_artifact_figure,
    make_badas_figure,
    make_badas_heatmap,
    make_reason_coverage_heatmap,
    make_risk_gauge,
    preload_runtime,
    run_pipeline,
    run_predict_only,
)


def badge_palette(value, kind):
    normalized = (value or "unknown").strip().lower()
    if kind == "incident":
        palettes = {
            "no_incident": ("#0f172a", "#22c55e", "No incident"),
            "near_miss": ("#1f2937", "#f59e0b", "Near miss"),
            "collision": ("#2b1a1a", "#ef4444", "Collision"),
            "multi_vehicle_collision": ("#2a1025", "#dc2626", "Multi-vehicle collision"),
            "unclear": ("#172033", "#64748b", "Unclear"),
        }
    else:
        palettes = {
            "none": ("#0f172a", "#22c55e", "None"),
            "low": ("#10261b", "#22c55e", "Low"),
            "moderate": ("#2b2110", "#f59e0b", "Moderate"),
            "high": ("#30151a", "#ef4444", "High"),
            "critical": ("#2a1025", "#dc2626", "Critical"),
            "unknown": ("#172033", "#64748b", "Unknown"),
        }
    return palettes.get(normalized, ("#172033", "#64748b", value or "Unknown"))


def format_html_cards(badas_result, reason_result, predict_payload=None):
    if not badas_result and not reason_result:
        return "<div style='color: #94a3b8; font-style: italic;'>Run the cached sample or upload a video to execute the pipeline.</div>"

    collision_triggered = bool((badas_result or {}).get("collision_detected"))
    gate_color = "#22c55e" if collision_triggered else "#64748b"
    gate_text = "Triggered" if collision_triggered else "Watching"

    incident = ((reason_result or {}).get("incident_type") or "unclear").lower()
    incident_colors = {
        "collision": ("#7f1d1d", "#ef4444"),
        "near_miss": ("#7c2d12", "#f97316"),
        "hazard": ("#713f12", "#f59e0b"),
    }
    inc_bg, inc_accent = incident_colors.get(incident, ("#1e293b", "#94a3b8"))

    severity = str((reason_result or {}).get("severity_label") or "unknown").lower()
    severity_colors = {
        "1": ("#14532d", "#22c55e"),
        "2": ("#713f12", "#f59e0b"),
        "3": ("#7c2d12", "#f97316"),
        "4": ("#7f1d1d", "#ef4444"),
        "5": ("#4c0519", "#e11d48"),
    }
    sev_bg, sev_accent = severity_colors.get(severity, ("#1e293b", "#94a3b8"))

    risk_score = (reason_result or {}).get("risk_score", 0)
    predict_modes = ", ".join((predict_payload or {}).get("modes") or []) if predict_payload else "Not run"

    cards_html = f"""
    <div style="display: flex; gap: 1.25rem; flex-wrap: wrap; margin-bottom: 1.5rem; font-family: 'Inter', sans-serif;">
        <div style="flex: 1; min-width: 220px; background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.7)); border-left: 5px solid {gate_color}; padding: 1.5rem; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.2), inset 0 1px 0 0 rgba(255,255,255,0.05); backdrop-filter: blur(12px);">
            <div style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Collision Gate</div>
            <div style="color: {gate_color}; font-size: 1.75rem; font-weight: 800; margin-top: 0.5rem; letter-spacing: -0.02em;">{gate_text}</div>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem; font-weight: 500;">BADAS V-JEPA2</div>
        </div>
        <div style="flex: 1; min-width: 220px; background: linear-gradient(145deg, {inc_bg}, rgba(15, 23, 42, 0.8)); border-left: 5px solid {inc_accent}; padding: 1.5rem; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.2), inset 0 1px 0 0 rgba(255,255,255,0.05); backdrop-filter: blur(12px);">
            <div style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Incident</div>
            <div style="color: #f8fafc; font-size: 1.75rem; font-weight: 800; margin-top: 0.5rem; text-transform: capitalize; letter-spacing: -0.02em;">{incident.replace("_", " ")}</div>
            <div style="color: #cbd5e1; font-size: 0.8rem; margin-top: 0.5rem; font-weight: 500;">Cosmos Reason 2</div>
        </div>
        <div style="flex: 1; min-width: 220px; background: linear-gradient(145deg, {sev_bg}, rgba(15, 23, 42, 0.8)); border-left: 5px solid {sev_accent}; padding: 1.5rem; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.2), inset 0 1px 0 0 rgba(255,255,255,0.05); backdrop-filter: blur(12px);">
            <div style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Severity</div>
            <div style="color: #f8fafc; font-size: 1.75rem; font-weight: 800; margin-top: 0.5rem; letter-spacing: -0.02em;">{severity.upper()}</div>
            <div style="color: #cbd5e1; font-size: 0.8rem; margin-top: 0.5rem; font-weight: 500;">Scale 1-5</div>
        </div>
        <div style="flex: 1; min-width: 220px; background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.7)); border-left: 5px solid #3b82f6; padding: 1.5rem; border-radius: 16px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.2), inset 0 1px 0 0 rgba(255,255,255,0.05); backdrop-filter: blur(12px);">
            <div style="color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Risk Score</div>
            <div style="color: #3b82f6; font-size: 1.75rem; font-weight: 800; margin-top: 0.5rem; letter-spacing: -0.02em;">{risk_score}/5</div>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 0.5rem; font-weight: 500;">Overall hazard rating</div>
        </div>
    </div>
    <div style="margin-top: 0.75rem; color: #64748b; font-size: 0.9rem; font-family: 'Inter', sans-serif;">
        <strong>Predict modes:</strong> {predict_modes}
    </div>
    """
    return cards_html


def format_reason_html(reason_result):
    if not reason_result:
        return "Run BADAS + Reason to populate the narrative panel."
    validation = (reason_result or {}).get("validation") or {}
    validation_flags = validation.get("flags") or {}
    fallback_override = (reason_result or {}).get("fallback_override") or {}
    lines = [
        "### Cosmos Reason 2 narrative",
        f"**Scene summary:** {(reason_result or {}).get('scene_summary') or 'N/A'}",
        f"**At-risk agent:** {(reason_result or {}).get('at_risk_agent') or 'N/A'}",
        f"**Explanation:** {(reason_result or {}).get('explanation') or 'N/A'}",
        f"**Time to impact:** {(reason_result or {}).get('time_to_impact')}",
        f"**Critical risk time:** {(reason_result or {}).get('critical_risk_time')}",
        "",
        "```text",
        (reason_result or {}).get("text") or "No Reason output captured.",
        "```",
    ]
    if not validation.get("is_reliable", True):
        lines.append("- **Warning:** Reason output was flagged as unreliable against BADAS evidence.")
    if validation_flags:
        lines.append(f"- **Validation flags:** `{json.dumps(validation_flags)}`")
    if fallback_override.get("applied"):
        lines.append(f"- **Fallback override:** `{json.dumps(fallback_override)}`")
    return "\n".join(lines)


def _render_cosmos_sentinel_page():
    st.markdown("""
    <section class="workspace-card">
        <p class="hub-eyebrow">Traffic Safety AI</p>
        <h2>Cosmos Sentinel Traffic Safety</h2>
        <p class="workspace-copy">
            BADAS V-JEPA2 collision detection, Cosmos Reason 2 incident narration, 
            and optional Cosmos Predict 2.5 counterfactual video generation.
        </p>
    </section>
    """, unsafe_allow_html=True)

    # Session state init
    if "cosmos_pipeline_state" not in st.session_state:
        st.session_state.cosmos_pipeline_state = None
    if "cosmos_input_video" not in st.session_state:
        st.session_state.cosmos_input_video = None

    with st.sidebar:
        st.markdown("### Cosmos Sentinel Controls")
        st.caption("GPU memory management: only one heavy model loaded at a time.")

        if st.button("Warm up BADAS + Reason + Predict"):
            with st.spinner("Preloading models..."):
                try:
                    result = preload_runtime(preload_badas=True, preload_reason=True, preload_predict=True)
                    st.success("Models warmed up successfully!")
                    st.text(result)
                except Exception as e:
                    st.error(f"Warmup failed: {e}")

        st.divider()
        st.markdown("### Sample Video")
        if st.button("Use cached sample video"):
            sample_path = ensure_sample_video()
            st.session_state.cosmos_input_video = sample_path
            st.success(f"Sample loaded: {Path(sample_path).name}")

    # Main layout
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload MP4 footage", type=["mp4"], key="cosmos_video_upload")
        if uploaded_file:
            cached_path = cache_uploaded_video(uploaded_file)
            st.session_state.cosmos_input_video = cached_path
            st.success(f"Cached: {Path(cached_path).name}")

        input_video = st.session_state.cosmos_input_video

        if input_video:
            st.video(input_video)
            st.text(f"Active input: {input_video}")

    with col2:
        st.markdown("#### Pipeline Controls")

        predict_selection = st.selectbox(
            "Cosmos Predict rollout set",
            ["both", "prevented_continuation", "observed_continuation"],
            index=0,
        )

        run_badas_reason = st.button("Run BADAS + Reason", type="primary", disabled=not input_video)
        run_predict = st.button("Run Cosmos Predict", disabled=not st.session_state.cosmos_pipeline_state)

    # Pipeline execution
    if run_badas_reason and input_video:
        with st.spinner("Running BADAS + Reason pipeline..."):
            try:
                result = run_pipeline(str(input_video), include_predict=False)
                st.session_state.cosmos_pipeline_state = result["pipeline_payload"]
                st.session_state.cosmos_last_result = result
                st.success("Pipeline completed!")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

    if run_predict and st.session_state.cosmos_pipeline_state:
        with st.spinner("Running Cosmos Predict..."):
            try:
                predict_payload, merged = run_predict_only(
                    st.session_state.cosmos_pipeline_state,
                    selection=predict_selection,
                    predict_model_name=PREDICT_MODEL_NAME,
                )
                st.session_state.cosmos_pipeline_state = merged
                st.session_state.cosmos_last_predict = predict_payload
                st.success("Predict completed!")
            except Exception as e:
                st.error(f"Predict failed: {e}")

    # Results display
    pipeline_payload = st.session_state.get("cosmos_pipeline_state")
    predict_payload = st.session_state.get("cosmos_last_predict")

    if pipeline_payload:
        iteration = ((pipeline_payload.get("iterations") or [{}]))[-1]
        steps = iteration.get("steps") or {}
        badas_result = (steps.get("badas") or {}).get("result") or {}
        reason_result = (steps.get("reason") or {}).get("result") or {}

        st.divider()
        st.markdown("### Results Dashboard")

        # Status cards
        st.html(format_html_cards(badas_result, reason_result, predict_payload))

        # Narrative
        with st.expander("Cosmos Reason Narrative", expanded=True):
            st.markdown(format_reason_html(reason_result))

        # Visualizations
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            fig_badas = make_badas_figure(badas_result)
            st.plotly_chart(fig_badas, use_container_width=True, key="cosmos_badas_timeline")

            fig_badas_hm = make_badas_heatmap(badas_result)
            st.plotly_chart(fig_badas_hm, use_container_width=True, key="cosmos_badas_heatmap")

        with viz_col2:
            fig_risk = make_risk_gauge(reason_result)
            st.plotly_chart(fig_risk, use_container_width=True, key="cosmos_risk_gauge")

            fig_reason_hm = make_reason_coverage_heatmap(reason_result)
            st.plotly_chart(fig_reason_hm, use_container_width=True, key="cosmos_reason_heatmap")

        # Artifacts
        st.subheader("Generated Artifacts")
        artifacts = pipeline_payload.get("artifacts") or {}

        artifact_cols = st.columns(3)
        with artifact_cols[0]:
            if artifacts.get("extracted_clip"):
                st.video(artifacts["extracted_clip"])
                st.caption("BADAS-focused clip")
        with artifact_cols[1]:
            if artifacts.get("badas_gradient_saliency"):
                st.image(artifacts["badas_gradient_saliency"])
                st.caption("BADAS gradient saliency")
        with artifact_cols[2]:
            fig_artifact = make_artifact_figure(artifacts)
            st.plotly_chart(fig_artifact, use_container_width=True, key="cosmos_artifact_status")

        # Predict outputs
        if predict_payload:
            st.subheader("Cosmos Predict Outputs")
            results = predict_payload.get("results") or {}
            for mode, result in results.items():
                output_video = result.get("output_video")
                if output_video:
                    st.video(output_video)
                    st.caption(f"Predict: {mode}")

        # JSON payloads
        with st.expander("Structured Payloads"):
            json_col1, json_col2 = st.columns(2)
            with json_col1:
                st.json({"badas": badas_result})
            with json_col2:
                st.json({"reason": reason_result})
