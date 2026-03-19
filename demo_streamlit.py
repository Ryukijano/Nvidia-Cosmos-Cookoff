import base64
import json
import os
from pathlib import Path
import subprocess
import sys

import plotly.graph_objects as go
import pandas as pd
import streamlit as st

# Load .env file at startup
def load_env_file():
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

load_env_file()

st.set_page_config(page_title="Cosmos Sentinel", page_icon="", layout="wide")

PIPELINE_JSON_PREFIX = "PIPELINE_JSON:"
BADAS_JSON_PREFIX = "BADAS_JSON:"
REASON_JSON_PREFIX = "REASON_JSON:"


@st.cache_data(show_spinner=False)
def load_video_bytes(video_path, modified_time):
    with open(video_path, "rb") as file_handle:
        return file_handle.read()


def render_video_path(video_path, container=None):
    """Render video as base64 data URI in HTML to bypass Streamlit media manager issues."""
    if not video_path or not os.path.exists(video_path):
        return False
    target = container or st
    video_bytes = load_video_bytes(video_path, os.path.getmtime(video_path))
    b64 = base64.b64encode(video_bytes).decode()
    html = f'''
    <video width="100%" controls style="border-radius:8px;background:#000;">
        <source src="data:video/mp4;base64,{b64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    '''
    target.markdown(html, unsafe_allow_html=True)
    return True


STEP_PROGRESS = {
    "📍 Step 1": (0.20, "BADAS V-JEPA2 scanning for predictive collisions"),
    "📍 Step 2": (0.40, "Extracting the BADAS-focused evidence clip"),
    "📍 Step 3": (0.70, "Cosmos Reason 2 analyzing the full video with BADAS focus guidance"),
    "🎉 Pure Cosmos Pipeline completed!": (1.00, "Pipeline completed"),
}


def extract_pipeline_payload(output_text):
    for line in output_text.splitlines():
        if line.startswith(PIPELINE_JSON_PREFIX):
            try:
                return json.loads(line.split(PIPELINE_JSON_PREFIX, 1)[1].strip())
            except json.JSONDecodeError:
                return None
    return None


def extract_latest_structured_json(output_text, prefix):
    payload = None
    for line in output_text.splitlines():
        if line.startswith(prefix):
            try:
                payload = json.loads(line.split(prefix, 1)[1].strip())
            except json.JSONDecodeError:
                continue
    return payload


def render_card(title, value, subtitle, accent):
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(16,24,40,0.95), rgba(10,18,33,0.95)); border: 1px solid rgba(255,255,255,0.08); border-left: 4px solid {accent}; border-radius: 18px; padding: 1rem 1.1rem; min-height: 120px; box-shadow: 0 10px 30px rgba(0,0,0,0.18);">
            <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 0.55rem;">{title}</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: white; line-height: 1.1; margin-bottom: 0.5rem;">{value}</div>
            <div style="font-size: 0.88rem; color: #cbd5e1;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
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


def render_status_badges(incident_type, severity_label):
    incident_bg, incident_accent, incident_label = badge_palette(incident_type, "incident")
    severity_bg, severity_accent, severity_text = badge_palette(severity_label, "severity")
    st.markdown(
        f"""
        <div style="display:flex; gap:0.9rem; flex-wrap:wrap; margin:0.35rem 0 1rem 0;">
            <div style="background:{incident_bg}; border:1px solid {incident_accent}; border-radius:999px; padding:0.55rem 0.9rem; color:#f8fafc; font-weight:700; letter-spacing:0.01em;">
                Incident: {incident_label}
            </div>
            <div style="background:{severity_bg}; border:1px solid {severity_accent}; border-radius:999px; padding:0.55rem 0.9rem; color:#f8fafc; font-weight:700; letter-spacing:0.01em;">
                Severity: {severity_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_badas_figure(badas_result):
    series = (badas_result or {}).get("prediction_series") or []
    figure = go.Figure()
    if series:
        times = [item["time_sec"] for item in series]
        probs = [item["probability"] for item in series]
        figure.add_trace(
            go.Scatter(
                x=times,
                y=probs,
                mode="lines+markers",
                line=dict(color="#22c55e", width=3),
                marker=dict(size=6, color="#22c55e"),
                fill="tozeroy",
                fillcolor="rgba(34,197,94,0.18)",
                name="Collision probability",
                hovertemplate="t=%{x:.2f}s<br>p=%{y:.2%}<extra></extra>",
            )
        )
    threshold = (badas_result or {}).get("threshold")
    if threshold is not None:
        figure.add_hline(y=threshold, line_dash="dash", line_color="#f59e0b")
    alert_time = (badas_result or {}).get("alert_time")
    if alert_time is not None:
        figure.add_vline(x=alert_time, line_dash="dot", line_color="#ef4444")
    figure.update_layout(
        title="BADAS predictive collision timeline",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font=dict(color="#e2e8f0"),
        xaxis_title="Time (s)",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure


def make_badas_heatmap(badas_result):
    series = (badas_result or {}).get("prediction_series") or []
    if not series:
        return None
    df = pd.DataFrame(series)
    df["bucket"] = df["sampled_frame"].astype(int)
    z = [df["probability"].tolist()]
    x = [f"{time_sec:.1f}s" for time_sec in df["time_sec"].tolist()]
    figure = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=["BADAS risk"],
            colorscale=[
                [0.0, "#052e16"],
                [0.35, "#166534"],
                [0.6, "#f59e0b"],
                [1.0, "#dc2626"],
            ],
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Risk"),
            hovertemplate="time=%{x}<br>prob=%{z:.2%}<extra></extra>",
        )
    )
    figure.update_layout(
        title="BADAS live risk heatmap",
        height=190,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font=dict(color="#e2e8f0"),
    )
    return figure


def make_reason_coverage_heatmap(reason_result):
    frame_metadata = (reason_result or {}).get("frame_metadata") or {}
    timestamps = frame_metadata.get("sampled_timestamps_sec") or []
    if not timestamps:
        return None
    bbox_count = max(1, int((reason_result or {}).get("bbox_count") or 0))
    z = [[bbox_count for _ in timestamps]]
    figure = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[f"{timestamp:.1f}s" for timestamp in timestamps],
            y=["Reason coverage"],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="BBox count proxy"),
            hovertemplate="sampled time=%{x}<br>coverage proxy=%{z}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Reason sampled-frame coverage map",
        height=190,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font=dict(color="#e2e8f0"),
    )
    return figure


def make_risk_gauge(reason_result):
    risk_score = (reason_result or {}).get("risk_score") or 0
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "/5", "font": {"size": 40, "color": "#f8fafc"}},
            title={"text": "Cosmos Reason 2 risk score", "font": {"size": 20, "color": "#cbd5e1"}},
            gauge={
                "axis": {"range": [0, 5], "tickwidth": 1, "tickcolor": "#94a3b8"},
                "bar": {"color": "#ef4444" if risk_score >= 4 else "#f59e0b" if risk_score >= 3 else "#22c55e"},
                "bgcolor": "rgba(15,23,42,0.30)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 2], "color": "rgba(34,197,94,0.20)"},
                    {"range": [2, 4], "color": "rgba(245,158,11,0.20)"},
                    {"range": [4, 5], "color": "rgba(239,68,68,0.24)"},
                ],
            },
        )
    )
    figure.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"))
    return figure


def make_artifact_figure(artifacts):
    labels = ["Clip", "BBox", "Risk", "GIF"]
    values = [1 if artifacts.get(key) else 0 for key in ["extracted_clip", "bbox_image", "risk_image", "overlay_gif"]]
    colors = ["#22c55e" if value else "#475569" for value in values]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=["ready" if value else "missing" for value in values],
            textposition="outside",
            hovertemplate="%{x}: %{text}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Artifact readiness",
        height=250,
        yaxis=dict(range=[0, 1.2], showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        font=dict(color="#e2e8f0"),
    )
    return figure


def render_reason_panel(reason_result):
    explanation = (reason_result or {}).get("explanation") or "No explanation extracted."
    at_risk_agent = (reason_result or {}).get("at_risk_agent") or "Not identified"
    time_to_impact = (reason_result or {}).get("time_to_impact")
    critical_risk_time = (reason_result or {}).get("critical_risk_time")
    scene_summary = (reason_result or {}).get("scene_summary") or "No scene summary extracted."
    incident_type = (reason_result or {}).get("incident_type") or "unclear"
    severity_label = (reason_result or {}).get("severity_label") or "unknown"
    validation = (reason_result or {}).get("validation") or {}
    validation_flags = validation.get("flags") or {}
    fallback_override = (reason_result or {}).get("fallback_override") or {}
    st.markdown("### Cosmos Reason 2 narrative")
    render_status_badges(incident_type, severity_label)
    if not validation.get("is_reliable", True):
        st.warning("Reason output was flagged as unreliable against BADAS evidence. The displayed classification may include a safety override.")
    if fallback_override.get("applied"):
        st.error("BADAS consistency guard overrode an inconsistent Reason result to avoid a false no-incident report.")
    st.markdown(
        f"""
        <div style="background: rgba(15,23,42,0.65); border: 1px solid rgba(148,163,184,0.20); border-radius: 18px; padding: 1rem 1.1rem; min-height: 180px;">
            <div style="color: #f8fafc; font-size: 1rem; line-height: 1.65; white-space: pre-wrap;">{(reason_result or {}).get('text', 'No Reason 2 output captured.')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(scene_summary)
    detail_cols = st.columns(4)
    detail_cols[0].metric("Critical risk time", f"{critical_risk_time:.2f}s" if isinstance(critical_risk_time, (int, float)) else "N/A")
    detail_cols[1].metric("Time to impact", f"{time_to_impact:.2f}s" if isinstance(time_to_impact, (int, float)) else "N/A")
    detail_cols[2].metric("At-risk agent", at_risk_agent)
    detail_cols[3].metric("BBox count", str(len((reason_result or {}).get("bboxes") or {})))
    class_cols = st.columns(2)
    class_cols[0].metric("Incident type", incident_type)
    class_cols[1].metric("Severity label", severity_label)
    st.info(explanation)
    parsing_summary = (reason_result or {}).get("parsing_summary") or {}
    frame_metadata = (reason_result or {}).get("frame_metadata") or {}
    focus_frame_metadata = (reason_result or {}).get("focus_frame_metadata") or {}
    video_input_count = int((reason_result or {}).get("video_input_count") or 0)
    with st.expander("Reason diagnostics", expanded=False):
        diag_cols = st.columns(5)
        diag_cols[0].metric("Parsed fields", str(parsing_summary.get("parsed_field_count", 0)))
        diag_cols[1].metric("Expected fields", str(parsing_summary.get("total_expected_fields", 0)))
        diag_cols[2].metric("Frames processed", str(frame_metadata.get("processed_frame_count", 0)))
        diag_cols[3].metric("Reliable", "Yes" if validation.get("is_reliable", True) else "No")
        diag_cols[4].metric("Video inputs", str(video_input_count))
        missing_fields = parsing_summary.get("missing_fields") or []
        if missing_fields:
            st.warning("Missing parsed fields: " + ", ".join(missing_fields))
        else:
            st.success("All tracked Reason fields were parsed")
        if validation_flags:
            st.markdown("#### Reason validation flags")
            st.json(validation_flags)
        if validation.get("note"):
            st.info(validation.get("note"))
        if fallback_override.get("applied"):
            st.markdown("#### Fallback override")
            st.json(fallback_override)
        if frame_metadata:
            st.markdown("#### Full-video sampling metadata")
            st.json(frame_metadata)
        if focus_frame_metadata:
            st.markdown("#### BADAS-focused clip sampling metadata")
            st.json(focus_frame_metadata)


def render_badas_diagnostics(badas_result):
    if not badas_result:
        return
    st.markdown("### BADAS diagnostics")
    diag_cols = st.columns(4)
    diag_cols[0].metric("Valid predictions", str(badas_result.get("valid_prediction_count") or 0))
    diag_cols[1].metric("NaN warmup", str(badas_result.get("nan_warmup_count") or 0))
    diag_cols[2].metric("Peak probability", f"{(badas_result.get('valid_prediction_max') or 0):.1%}")
    diag_cols[3].metric("Threshold crossings", str(((badas_result.get("threshold_summary") or {}).get("threshold_crossing_count")) or 0))
    with st.expander("BADAS stats and metadata", expanded=False):
        stats_df = pd.DataFrame(
            [
                {"metric": "valid_prediction_min", "value": badas_result.get("valid_prediction_min")},
                {"metric": "valid_prediction_mean", "value": badas_result.get("valid_prediction_mean")},
                {"metric": "valid_prediction_median", "value": badas_result.get("valid_prediction_median")},
                {"metric": "valid_prediction_std", "value": badas_result.get("valid_prediction_std")},
                {"metric": "valid_prediction_p90", "value": badas_result.get("valid_prediction_p90")},
                {"metric": "valid_prediction_p95", "value": badas_result.get("valid_prediction_p95")},
                {"metric": "first_valid_time", "value": badas_result.get("first_valid_time")},
                {"metric": "alert_source", "value": badas_result.get("alert_source")},
            ]
        )
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        st.markdown("#### Threshold runs")
        st.json((badas_result.get("threshold_summary") or {}).get("contiguous_alert_runs") or [])
        st.markdown("#### Model / video metadata")
        meta_cols = st.columns(2)
        meta_cols[0].json(badas_result.get("model_info") or {})
        meta_cols[1].json(badas_result.get("video_metadata") or {})


def render_predict_panel(predict_payload):
    if not predict_payload:
        st.markdown("### Cosmos Predict continuation")
        st.info("Run Cosmos Predict manually after BADAS + Reason complete to generate a context-aware continuation clip.")
        return
    st.markdown("### Cosmos Predict continuation")
    if not predict_payload.get("success"):
        st.error(predict_payload.get("error") or "Cosmos Predict did not produce a video output.")
        return
    results = predict_payload.get("results") or {}
    conditioning_source = (predict_payload.get("conditioning_source") or "unknown").replace("_", " ")
    fallback_applied = bool(predict_payload.get("fallback_applied"))
    first_result = next(iter(results.values()), {}) if results else {}
    conditioning_window = first_result.get("conditioning_window") or {}
    predict_cols = st.columns(4)
    predict_cols[0].metric("Modes", ", ".join([mode.replace("_", " ") for mode in (predict_payload.get("modes") or [])]) or "N/A")
    predict_cols[1].metric("Conditioning", conditioning_source)
    predict_cols[2].metric("Fallback", "Yes" if fallback_applied else "No")
    predict_cols[3].metric("Focus time", f"{conditioning_window.get('focus_time_sec', 0):.2f}s" if conditioning_window.get("focus_time_sec") is not None else "N/A")
    rollout_cols = st.columns(max(1, len(results) or 1), gap="medium")
    for idx, (mode, result) in enumerate(results.items()):
        target_col = rollout_cols[idx] if idx < len(rollout_cols) else st
        with target_col:
            st.markdown(f"#### {mode.replace('_', ' ').title()}")
            if result.get("output_video") and render_video_path(result.get("output_video")):
                pass
            else:
                st.info("No rollout generated")
            st.caption(
                f"Source: {(result.get('conditioning_source') or 'unknown').replace('_', ' ')} | Cached: {'Yes' if result.get('cached') else 'No'}"
            )
            if result.get("fallback_applied") and result.get("fallback_reason"):
                st.caption(f"Fallback reason: {result.get('fallback_reason')}")
    with st.expander("Predict diagnostics", expanded=False):
        if first_result.get("conditioning_clip"):
            st.markdown("#### Conditioning clip")
            render_video_path(first_result.get("conditioning_clip"))
        st.markdown("#### Conditioning metadata")
        st.json(first_result.get("conditioning_metadata") or {})
        if predict_payload.get("fallback_reasons"):
            st.markdown("#### Fallback reasons")
            st.json(predict_payload.get("fallback_reasons"))
        for mode, result in results.items():
            st.markdown(f"#### {mode.replace('_', ' ').title()} prompt")
            st.text_area(
                f"Predict prompt {mode}",
                result.get("prompt", ""),
                height=220,
                key=f"predict_prompt_text_area_{mode}",
            )
        st.markdown("#### Predict payload")
        st.json(predict_payload)


def merge_predict_payload_into_pipeline(payload, predict_payload):
    if not payload or not predict_payload:
        return payload
    merged_payload = json.loads(json.dumps(payload))
    merged_payload["predict"] = predict_payload
    merged_artifacts = merged_payload.get("artifacts") or {}
    for artifact_key, artifact_value in (predict_payload.get("artifacts") or {}).items():
        merged_artifacts[artifact_key] = artifact_value
    merged_payload["artifacts"] = merged_artifacts
    return merged_payload


def summarize_activity_feed(output_text, badas_result, reason_result):
    feed = []
    if "Starting Pure Cosmos Pipeline" in output_text:
        feed.append("Pipeline started and video queued for analysis")
    if "BADAS V-JEPA2 Collision Detection" in output_text:
        feed.append("BADAS is scanning the CCTV clip for developing collision risk")
    if badas_result:
        if badas_result.get("collision_detected"):
            feed.append(
                f"BADAS triggered at {badas_result.get('alert_time', 0):.2f}s with {badas_result.get('confidence', 0):.1%} confidence"
            )
        else:
            feed.append("BADAS finished without a hard trigger and selected the first valid predictive frame")
    if "Extracting Pre-Alert Clip" in output_text:
        feed.append("The system is extracting a BADAS-focused evidence clip around the highest-risk moment")
    if "Cosmos Reason 2 Risk Analysis" in output_text:
        feed.append("Cosmos Reason is reviewing the full CCTV video and the BADAS-focused clip to write an incident summary")
    if reason_result:
        incident_type = reason_result.get("incident_type") or "unclear"
        severity_label = reason_result.get("severity_label") or "unknown"
        feed.append(f"Reason classified the scene as {incident_type.replace('_', ' ')} with {severity_label} severity")
        validation = reason_result.get("validation") or {}
        if not validation.get("is_reliable", True):
            feed.append("Reason output was flagged as unreliable against BADAS evidence")
        fallback_override = reason_result.get("fallback_override") or {}
        if fallback_override.get("applied"):
            feed.append("BADAS consistency guard applied a safety override to prevent a false no-incident result")
        if reason_result.get("scene_summary"):
            feed.append(reason_result.get("scene_summary"))
    if "Pure Cosmos Pipeline completed!" in output_text:
        feed.append("Pipeline completed and final artifacts are ready")
    return feed


def run_pipeline_live(video_path, log_placeholder, activity_placeholder, status_placeholder, progress_placeholder, live_badas_placeholder, live_reason_placeholder):
    # Set up environment with UTF-8 encoding for subprocess
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token
    else:
        env.pop("HF_TOKEN", None)
        env.pop("HUGGINGFACE_HUB_TOKEN", None)
    
    process = subprocess.Popen(
        [sys.executable, "main_pipeline.py", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=os.getcwd(),
        env=env,
    )
    output_lines = []
    update_counter = 0
    status_placeholder.markdown("**Status:** Initializing pipeline")
    progress_placeholder.progress(0.02)
    latest_badas = None
    latest_reason = None
    for line in iter(process.stdout.readline, ""):
        output_lines.append(line)
        update_counter += 1
        text = "".join(output_lines)
        latest_badas = extract_latest_structured_json(text, BADAS_JSON_PREFIX) or latest_badas
        latest_reason = extract_latest_structured_json(text, REASON_JSON_PREFIX) or latest_reason
        for marker, (progress, status_text) in STEP_PROGRESS.items():
            if marker in line:
                progress_placeholder.progress(progress)
                status_placeholder.markdown(f"**Status:** {status_text}")
                break
        log_placeholder.text_area("Live pipeline logs", text, height=320, key=f"pipeline_logs_{update_counter}")
        activity_feed = summarize_activity_feed(text, latest_badas, latest_reason)
        activity_placeholder.markdown("### Live activity")
        activity_placeholder.markdown("\n".join([f"- {item}" for item in activity_feed[-6:]]) or "- Waiting for pipeline updates")
        live_badas_chart = make_badas_heatmap(latest_badas)
        if live_badas_chart is not None:
            live_badas_placeholder.plotly_chart(live_badas_chart, use_container_width=True, key=f"live_badas_chart_{update_counter}")
        else:
            live_badas_placeholder.info("BADAS heatmap will appear as soon as structured detector output is available")
        live_reason_chart = make_reason_coverage_heatmap(latest_reason)
        if live_reason_chart is not None:
            live_reason_placeholder.plotly_chart(live_reason_chart, use_container_width=True, key=f"live_reason_chart_{update_counter}")
        else:
            live_reason_placeholder.info("Reason sampled-frame coverage map will appear once the narrator emits structured output")
    process.stdout.close()
    process.wait()
    output_text = "".join(output_lines)
    payload = extract_pipeline_payload(output_text)
    progress_placeholder.progress(1.0)
    status_placeholder.markdown("**Status:** Completed" if process.returncode == 0 else "**Status:** Finished with errors")
    return process.returncode, output_text, payload


def current_iteration(payload):
    iterations = (payload or {}).get("iterations") or []
    return iterations[-1] if iterations else {}


st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, rgba(30,41,59,0.85), rgba(2,6,23,1));
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "pipeline_output" not in st.session_state:
    st.session_state.pipeline_output = ""
if "pipeline_payload" not in st.session_state:
    st.session_state.pipeline_payload = None
if "input_video_path" not in st.session_state:
    st.session_state.input_video_path = None
if "predict_payload" not in st.session_state:
    st.session_state.predict_payload = None
if "predict_selection" not in st.session_state:
    st.session_state.predict_selection = "both"

st.markdown("## 🚦 Cosmos Sentinel")
st.markdown("### BADAS V-JEPA2 + Cosmos Reason 2")
st.markdown("A detailed incident-analysis cockpit where BADAS gates risk, Cosmos Reason explains the event, and Cosmos Predict can manually simulate a context-aware continuation.")

REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_VIDEO_PATH = str(REPO_ROOT / "1_first.mp4")

with st.sidebar:
    st.markdown("## Pipeline stack")
    st.markdown("- **System 1**: BADAS V-JEPA2 predictive collision gate")
    st.markdown("- **System 2**: Cosmos Reason 2 full-video causal narrative and structured parsing")
    st.markdown("- **System 3**: Cosmos Predict 2.5 manual continuation rollout")
    st.markdown("- **Artifacts**: BADAS-focused clip, saliency, risk plot, overlay visuals, Predict rollout")
    st.markdown("---")
    st.caption("The UI is driven by real `PIPELINE_JSON`, `BADAS_JSON`, and `REASON_JSON` payloads from the pipeline.")

existing_payload = st.session_state.get("pipeline_payload")
existing_iteration = current_iteration(existing_payload)
existing_steps = existing_iteration.get("steps", {}) if existing_iteration else {}
existing_reason_result = (existing_steps.get("reason", {}) or {}).get("result") or {}
analysis_ready = bool(existing_reason_result)

control_col, preview_col = st.columns([1, 1.4], gap="large")

with control_col:
    st.markdown("### Input")
    uploaded_file = st.file_uploader(
        "Upload MP4 footage",
        type=["mp4"],
        help="Upload dashcam or traffic footage to run the full Cosmos pipeline. If you need larger uploads, configure Streamlit's server.maxUploadSize setting.",
        key="video_uploader",
    )
    use_sample = st.button("Use sample: 1_first.mp4", use_container_width=True)
    input_path = None
    if uploaded_file is not None:
        input_path = "./temp_input.mp4"
        with open(input_path, "wb") as file_handle:
            file_handle.write(uploaded_file.getbuffer())
    elif use_sample:
        input_path = SAMPLE_VIDEO_PATH
    elif st.session_state.get("input_video_path"):
        input_path = st.session_state.get("input_video_path")
    if input_path:
        st.session_state.input_video_path = input_path
        st.success(f"Ready: `{input_path}`")
    run_clicked = st.button("Run BADAS + Reason", type="primary", use_container_width=True, disabled=not bool(input_path))
    predict_selection = st.selectbox(
        "Cosmos Predict rollout set",
        options=["both", "prevented_continuation", "observed_continuation"],
        index=["both", "prevented_continuation", "observed_continuation"].index(st.session_state.get("predict_selection", "both")),
        format_func=lambda value: "Observed + Prevented" if value == "both" else value.replace("_", " ").title(),
        disabled=not analysis_ready,
    )
    st.session_state.predict_selection = predict_selection
    predict_clicked = st.button(
        "Run Cosmos Predict",
        use_container_width=True,
        disabled=not analysis_ready,
        help="Runs a manual context-aware continuation using the existing BADAS and Cosmos Reason outputs, with a fallback to the BADAS-focused clip if needed.",
    )
    if not analysis_ready:
        st.caption("Run BADAS + Reason first. Cosmos Predict stays manual and reuses the existing analysis outputs.")

with preview_col:
    st.markdown("### Preview")
    preview_video_path = st.session_state.get("input_video_path")
    if preview_video_path and os.path.exists(preview_video_path):
        render_video_path(preview_video_path)
    else:
        st.info("Upload a video or choose the sample clip to preview it here.")

status_placeholder = st.empty()
progress_placeholder = st.empty()
log_placeholder = st.empty()
activity_placeholder = st.empty()
live_visual_cols = st.columns(2, gap="medium")
live_badas_placeholder = live_visual_cols[0].empty()
live_reason_placeholder = live_visual_cols[1].empty()

current_input_video_path = st.session_state.get("input_video_path")
if run_clicked and current_input_video_path:
    with st.spinner("Running pipeline"):
        return_code, output_text, payload = run_pipeline_live(
            current_input_video_path,
            log_placeholder,
            activity_placeholder,
            status_placeholder,
            progress_placeholder,
            live_badas_placeholder,
            live_reason_placeholder,
        )
    st.session_state.pipeline_output = output_text
    st.session_state.pipeline_payload = payload
    st.session_state.predict_payload = None
    if payload is None:
        st.error("The pipeline did not emit a parseable `PIPELINE_JSON` payload.")
    elif return_code != 0:
        st.warning("The process exited with a non-zero code, but partial results may still be available below.")

payload = st.session_state.get("pipeline_payload")
output_text = st.session_state.get("pipeline_output", "")
iteration = current_iteration(payload)
steps = iteration.get("steps", {}) if iteration else {}
badas_step = steps.get("badas", {})
reason_step = steps.get("reason", {})
artifacts = (payload or {}).get("artifacts") or {}
badas_result = badas_step.get("result") or {}
reason_result = reason_step.get("result") or {}
overview = (payload or {}).get("overview") or {}
predict_payload = st.session_state.get("predict_payload") or {}
if predict_payload and payload and predict_payload.get("source_video_path") != (payload or {}).get("input_video"):
    st.session_state.predict_payload = None
    predict_payload = {}
if predict_clicked and payload and reason_result:
    with st.spinner("Running Cosmos Predict"):
        try:
            from cosmos_predict_runner import run_predict_bundle
            current_predict_selection = st.session_state.get("predict_selection", "both")
            selected_modes = ["prevented_continuation", "observed_continuation"] if current_predict_selection == "both" else [current_predict_selection]
            predict_payload = run_predict_bundle(
                (payload or {}).get("input_video") or st.session_state.get("input_video_path"),
                badas_context=badas_result,
                reason_context=reason_result,
                modes=selected_modes,
                fallback_conditioning_path=artifacts.get("extracted_clip"),
            )
            st.session_state.predict_payload = predict_payload
            st.session_state.pipeline_payload = merge_predict_payload_into_pipeline(payload, predict_payload)
            payload = st.session_state.get("pipeline_payload")
            artifacts = (payload or {}).get("artifacts") or {}
        except Exception as exc:
            predict_payload = {"success": False, "error": str(exc), "modes": [st.session_state.get("predict_selection", "both")]}
            st.session_state.predict_payload = predict_payload
incident_type = reason_result.get("incident_type") or "unclear"
severity_label = reason_result.get("severity_label") or "unknown"
_, incident_accent, incident_display = badge_palette(incident_type, "incident")
_, severity_accent, severity_display = badge_palette(severity_label, "severity")

metric_cols = st.columns(4, gap="medium")
with metric_cols[0]:
    render_card(
        "Collision gate",
        "Triggered" if badas_result.get("collision_detected") else "Watching",
        "BADAS V-JEPA2 predictive alert state",
        "#22c55e" if badas_result.get("collision_detected") else "#64748b",
    )
with metric_cols[1]:
    render_card(
        "Alert time",
        f"{badas_result.get('alert_time', 0):.2f}s" if badas_result.get("alert_time") is not None else "N/A",
        "Earliest sampled threshold crossing",
        "#38bdf8",
    )
with metric_cols[2]:
    render_card(
        "BADAS confidence",
        f"{badas_result.get('confidence', 0):.1%}" if badas_result else "N/A",
        "Confidence at the selected alert frame",
        "#f59e0b",
    )
with metric_cols[3]:
    render_card(
        "Reason risk",
        f"{(reason_result.get('risk_score') or 0)}/5" if reason_result else "N/A",
        "Cosmos Reason 2 safety severity",
        "#ef4444",
    )

predict_status_cols = st.columns(2, gap="medium")
with predict_status_cols[0]:
    current_predict_selection = st.session_state.get("predict_selection", "both")
    render_card(
        "Predict rollouts",
        " + ".join([mode.replace("_", " ").title() for mode in (predict_payload.get("modes") or ([current_predict_selection] if current_predict_selection != "both" else ["prevented_continuation", "observed_continuation"]))]) if (predict_payload or current_predict_selection) else "N/A",
        "Manual Cosmos Predict continuation set",
        "#a855f7",
    )
with predict_status_cols[1]:
    render_card(
        "Predict status",
        "Ready" if (predict_payload.get("success") and (predict_payload.get("results") or {})) else "Waiting" if reason_result else "Locked",
        "Manual continuation rollout bundle generated from BADAS + Reason outputs",
        "#22c55e" if (predict_payload.get("success") and (predict_payload.get("results") or {})) else "#64748b",
    )

status_cols = st.columns(2, gap="medium")
with status_cols[0]:
    render_card(
        "Incident class",
        incident_display,
        "Cosmos Reason 2 CCTV incident classification",
        incident_accent,
    )
with status_cols[1]:
    render_card(
        "Severity class",
        severity_display,
        "Cosmos Reason 2 estimated accident severity",
        severity_accent,
    )

badas_timing = badas_step.get("timing") or {}
if badas_timing:
    perf_cols = st.columns(2, gap="medium")
    with perf_cols[0]:
        duration_val = badas_timing.get("duration_sec")
        render_card(
            "BADAS latency",
            f"{duration_val:.2f}s" if isinstance(duration_val, (int, float)) else "N/A",
            "Wall-clock time for BADAS subprocess",
            "#38bdf8",
        )
    with perf_cols[1]:
        throughput_val = badas_timing.get("predictions_per_sec")
        render_card(
            "BADAS throughput",
            f"{throughput_val:.1f} pred/s" if isinstance(throughput_val, (int, float)) else "N/A",
            "Predictions per second",
            "#a855f7",
        )

main_left, main_right = st.columns([1.2, 0.8], gap="large")

with main_left:
    st.plotly_chart(make_badas_figure(badas_result), use_container_width=True, key="badas_figure_main")
    render_badas_diagnostics(badas_result)
    render_reason_panel(reason_result)
    render_predict_panel(predict_payload)
    if reason_result.get("counterfactual_prompt"):
        st.markdown("### Counterfactual prompt")
        st.warning(reason_result.get("counterfactual_prompt"))

with main_right:
    st.plotly_chart(make_risk_gauge(reason_result), use_container_width=True, key="risk_gauge_main")
    st.plotly_chart(make_artifact_figure(artifacts), use_container_width=True, key="artifact_figure_main")
    top_predictions = badas_result.get("top_predictions") or []
    if top_predictions:
        st.markdown("### Peak BADAS frames")
        top_prediction_df = pd.DataFrame(top_predictions)
        top_prediction_df = top_prediction_df.rename(columns={"original_frame_approx": "original_frame"})
        st.dataframe(
            top_prediction_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "time_sec": st.column_config.NumberColumn(format="%.2f s"),
                "probability": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
            },
        )
    if overview:
        st.markdown("### Run overview")
        st.json(overview)
    st.markdown("### Pipeline step states")
    for step_name, step_data in steps.items():
        icon = "✅" if step_data.get("success") else "⚠️"
        st.write(f"{icon} **{step_name.replace('_', ' ').title()}**")

gallery_tab, logs_tab, raw_tab = st.tabs(["Visual gallery", "Logs", "Raw outputs"])

with gallery_tab:
    if artifacts.get("extracted_clip"):
        st.markdown("### BADAS-focused evidence clip")
        render_video_path(artifacts.get("extracted_clip"))
    if artifacts.get("predict_conditioning_clip"):
        st.markdown("### Predict conditioning clip")
        render_video_path(artifacts.get("predict_conditioning_clip"))
    predict_video_cols = st.columns(2)
    if artifacts.get("predict_observed_continuation_video"):
        predict_video_cols[0].markdown("### Predict observed rollout")
        render_video_path(artifacts.get("predict_observed_continuation_video"), container=predict_video_cols[0])
    elif (predict_payload.get("results") or {}).get("observed_continuation"):
        predict_video_cols[0].info("Observed rollout not available")
    if artifacts.get("predict_prevented_continuation_video"):
        predict_video_cols[1].markdown("### Predict prevented rollout")
        render_video_path(artifacts.get("predict_prevented_continuation_video"), container=predict_video_cols[1])
    elif (predict_payload.get("results") or {}).get("prevented_continuation"):
        predict_video_cols[1].info("Prevented rollout not available")
    if artifacts.get("badas_gradient_saliency"):
        st.markdown("### BADAS gradient saliency")
        st.image(artifacts.get("badas_gradient_saliency"), caption="BADAS gradient-based collision saliency")
    gallery_cols = st.columns(3)
    if artifacts.get("bbox_image"):
        gallery_cols[0].image(artifacts.get("bbox_image"), caption="Bounding box visualization")
    else:
        gallery_cols[0].info("Bounding box image not available")
    if artifacts.get("risk_image"):
        gallery_cols[1].image(artifacts.get("risk_image"), caption="Risk visualization")
    else:
        gallery_cols[1].info("Risk image not available")
    if artifacts.get("overlay_gif"):
        gallery_cols[2].image(artifacts.get("overlay_gif"), caption="Overlay GIF")
    else:
        gallery_cols[2].info("Overlay GIF not available")
    strip_cols = st.columns(2)
    if artifacts.get("badas_frame_strip"):
        strip_cols[0].image(artifacts.get("badas_frame_strip"), caption="BADAS frame-strip risk overlays")
    else:
        strip_cols[0].info("BADAS frame-strip overlays not available")
    if artifacts.get("reason_frame_strip"):
        strip_cols[1].image(artifacts.get("reason_frame_strip"), caption="Reason frame-strip spatial overlays")
    else:
        strip_cols[1].info("Reason frame-strip overlays not available")
    st.caption("Cosmos Reason uses the full video as primary input and may also inspect the BADAS-focused clip as a secondary input. Cosmos Predict runs manually from the existing BADAS + Reason outputs using a short context-aware conditioning segment. The BADAS saliency panel is gradient-based, and any Reason strip view is only shown when trustworthy localization data exists.")

with logs_tab:
    log_cols = st.columns(2, gap="large")
    with log_cols[0]:
        st.text_area("Technical pipeline logs", output_text or "No logs captured yet.", height=420, key="final_logs")
    with log_cols[1]:
        st.markdown("### Mainstream activity summary")
        activity_items = summarize_activity_feed(output_text or "", badas_result, reason_result)
        if predict_payload.get("success") and (predict_payload.get("results") or {}):
            for mode, result in (predict_payload.get("results") or {}).items():
                if result.get("output_video"):
                    activity_items.append(
                        f"Cosmos Predict generated a {mode.replace('_', ' ')} rollout from the {(result.get('conditioning_source') or 'conditioning clip').replace('_', ' ')}"
                    )
        elif predict_payload.get("error"):
            activity_items.append("Cosmos Predict failed to generate a continuation rollout")
        st.markdown("\n".join([f"- {item}" for item in activity_items]) or "- No activity updates yet")
        st.caption("Reason uses full-video sampling at 4 FPS in this pipeline. Cosmos Predict remains a separate manual stage that reuses the existing analysis outputs and a short context-aware conditioning segment.")

with raw_tab:
    raw_left, raw_right = st.columns(2)
    with raw_left:
        st.markdown("#### Pipeline summary JSON")
        st.json(payload or {})
        if reason_result.get("badas_context"):
            st.markdown("#### BADAS context passed into Reason")
            st.json(reason_result.get("badas_context") or {})
        if predict_payload:
            st.markdown("#### Cosmos Predict payload")
            st.json(predict_payload)
    with raw_right:
        st.markdown("#### Reason 2 raw text")
        st.text_area("", reason_result.get("text", ""), height=420, key="reason_text_area")
        st.markdown("#### Reason prompt used")
        st.text_area(" ", reason_result.get("user_prompt", ""), height=260, key="reason_prompt_text_area")
        for mode, result in (predict_payload.get("results") or {}).items():
            st.markdown(f"#### Predict prompt used ({mode.replace('_', ' ').title()})")
            st.text_area(
                f"predict_prompt_view_{mode}",
                result.get("prompt", ""),
                height=220,
                key=f"predict_prompt_view_{mode}",
            )

st.markdown("---")
st.caption("Built for hackathon-grade storytelling with detailed BADAS diagnostics, real Reason outputs, manual Predict rollouts, and artifact-first debugging.")
