"""Cosmos Sentinel — Enterprise Command Center Dashboard (Streamlit page).

Rendered by _render_enterprise_dashboard() which is imported and called from app.py.
"""

from __future__ import annotations

import time
import threading
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Lazy imports for heavy deps
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Camera feed definitions (demo)
# ---------------------------------------------------------------------------
CAMERA_FEEDS = [
    {"id": "cam_01", "name": "Factory Floor A",   "location": "Seoul Plant — Bay 1", "icon": "🏭"},
    {"id": "cam_02", "name": "Loading Dock B",     "location": "Seoul Plant — Dock 2", "icon": "🚚"},
    {"id": "cam_03", "name": "Intersection C",     "location": "Gangnam Crosswalk",   "icon": "🚦"},
    {"id": "cam_04", "name": "Warehouse D",        "location": "Incheon Logistics",   "icon": "📦"},
]

# ROI assumptions for KPI panel
ROI_PARAMS = {
    "avg_accident_cost_usd": 120_000,
    "near_miss_prevention_rate": 0.75,
    "annual_monitoring_cost_usd": 36_000,
}

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def _ss() -> dict:
    """Alias for st.session_state (typed as dict for brevity)."""
    return st.session_state


def _init_ss() -> None:
    defaults: dict[str, Any] = {
        "ent_active_cam": "cam_01",
        "ent_stream_running": False,
        "ent_mode": "Static Upload",
        "ent_risk_series": {c["id"]: [] for c in CAMERA_FEEDS},
        "ent_incident_log": [],
        "ent_api_thread_started": False,
        "ent_accidents_prevented": 0,
        "ent_stream_manager": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Mock frame generator for demo camera feeds
# ---------------------------------------------------------------------------
def _generate_mock_frame(cam_id: str, t: float, w: int = 320, h: int = 180) -> np.ndarray:
    """Produce a synthetic BGR frame that looks vaguely like a surveillance feed."""
    rng = np.random.default_rng(int((ord(cam_id[-1]) * 1000 + t * 10) % 2**32))
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # gradient sky / floor
    frame[:h//2, :] = [int(30 + rng.integers(0, 20)), int(60 + rng.integers(0, 20)), int(100 + rng.integers(0, 20))]
    frame[h//2:, :] = [int(40 + rng.integers(0, 15)), int(40 + rng.integers(0, 15)), int(40 + rng.integers(0, 15))]
    # moving "vehicle" blobs
    for _ in range(rng.integers(1, 4)):
        cx = int(rng.integers(20, w - 20))
        cy = int(rng.integers(h // 2, h - 20))
        cv2.rectangle(frame, (cx - 15, cy - 8), (cx + 15, cy + 8),
                      (int(rng.integers(100, 230)), int(rng.integers(100, 230)), int(rng.integers(100, 230))), -1)
    # timestamp overlay
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, cam_id.upper(), (5, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    return frame


def _frame_to_png_bytes(frame_bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", frame_bgr)
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Simulated risk tick
# ---------------------------------------------------------------------------
def _tick_risk(cam_id: str) -> dict:
    """Generate a synthetic risk score for the mock camera feed."""
    rng = np.random.default_rng(int(time.time() * 100) % (2**32 - 1))
    base = {"cam_01": 0.12, "cam_02": 0.20, "cam_03": 0.35, "cam_04": 0.08}.get(cam_id, 0.15)
    prob = float(np.clip(base + rng.normal(0, 0.12), 0.0, 1.0))
    collision = prob > 0.65
    return {
        "ts": time.time(),
        "cam_id": cam_id,
        "probability": round(prob, 4),
        "collision": collision,
        "severity": "HIGH" if prob > 0.8 else ("MEDIUM" if prob > 0.5 else "LOW"),
        "incident_type": "near_miss" if collision else None,
    }


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------
def _risk_sparkline(risk_series: list[dict], cam_id: str) -> "go.Figure":
    if not _PLOTLY_AVAILABLE:
        return None
    times = [r["ts"] for r in risk_series]
    probs = [r["probability"] for r in risk_series]
    colors = ["red" if r["collision"] else "orange" if r["probability"] > 0.5 else "green"
              for r in risk_series]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(times))), y=probs,
        mode="lines+markers",
        line=dict(color="rgba(99,110,250,0.8)", width=2),
        marker=dict(color=colors, size=6),
        name="Risk",
    ))
    fig.add_hline(y=0.65, line_dash="dot", line_color="red", annotation_text="Alert threshold")
    fig.update_layout(
        height=180,
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="",
        yaxis=dict(range=[0, 1], title="P(collision)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(size=10),
        title=dict(text=f"Risk trend — {cam_id}", font=dict(size=11)),
    )
    return fig


def _risk_gauge(prob: float) -> "go.Figure":
    if not _PLOTLY_AVAILABLE:
        return None
    color = "red" if prob > 0.65 else "orange" if prob > 0.4 else "green"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "rgba(0,200,0,0.15)"},
                {"range": [40, 65], "color": "rgba(255,165,0,0.15)"},
                {"range": [65, 100], "color": "rgba(255,0,0,0.15)"},
            ],
            "threshold": {"line": {"color": "red", "width": 2}, "thickness": 0.75, "value": 65},
        },
        title={"text": "Collision Probability"},
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _incident_heatmap(incident_log: list[dict]) -> "go.Figure":
    if not _PLOTLY_AVAILABLE or not incident_log:
        return None
    hours = [0] * 24
    for inc in incident_log:
        try:
            h = int(inc.get("time", "00:00:00").split(":")[0])
            hours[h] += 1
        except Exception:
            pass
    fig = go.Figure(go.Bar(
        x=list(range(24)),
        y=hours,
        marker_color=["red" if v > 2 else "orange" if v > 0 else "lightgrey" for v in hours],
        name="Incidents/hour",
    ))
    fig.update_layout(
        title="Incidents by hour of day",
        xaxis_title="Hour (UTC+9)",
        yaxis_title="Count",
        height=200,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10),
    )
    return fig


def _roi_waterfall(incidents_prevented: int) -> "go.Figure":
    if not _PLOTLY_AVAILABLE:
        return None
    saved = incidents_prevented * ROI_PARAMS["avg_accident_cost_usd"] * ROI_PARAMS["near_miss_prevention_rate"]
    cost = ROI_PARAMS["annual_monitoring_cost_usd"]
    net = saved - cost
    fig = go.Figure(go.Waterfall(
        name="ROI",
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Monitoring Cost", "Accidents Prevented", "Net ROI"],
        y=[-cost, saved, 0],
        totals={"marker": {"color": "green" if net >= 0 else "red"}},
        connector={"line": {"color": "rgb(63,63,63)"}},
        texttemplate="%{y:$,.0f}",
        textposition="outside",
    ))
    fig.update_layout(
        title="Estimated Annual ROI",
        height=220,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10),
    )
    return fig


# ---------------------------------------------------------------------------
# API service thread launcher
# ---------------------------------------------------------------------------
def _start_api_server_thread(port: int = 7861) -> None:
    """Start the FastAPI service in a daemon thread (best-effort)."""
    if st.session_state.get("ent_api_thread_started"):
        return
    try:
        import uvicorn
        from api_service import app as _api_app

        def _run():
            uvicorn.run(_api_app, host="0.0.0.0", port=port, log_level="warning")

        t = threading.Thread(target=_run, daemon=True, name="sentinel_api")
        t.start()
        st.session_state["ent_api_thread_started"] = True
    except Exception as exc:
        st.session_state["ent_api_thread_started"] = f"failed: {exc}"


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------
def _render_enterprise_dashboard() -> None:
    _init_ss()

    st.markdown(
        """
        <div style='background:linear-gradient(135deg,#0f0f1a 0%,#1a1a2e 50%,#16213e 100%);
                    padding:20px 28px;border-radius:12px;margin-bottom:18px;
                    border:1px solid rgba(99,110,250,0.3)'>
          <h1 style='color:#e0e0ff;margin:0;font-size:1.8rem'>
            🛡️ Cosmos Sentinel — Enterprise Command Center
          </h1>
          <p style='color:#8888bb;margin:4px 0 0;font-size:0.9rem'>
            Real-time AI collision detection &amp; risk analytics for industrial safety
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Top-level tabs ──────────────────────────────────────────────────────
    tab_live, tab_api, tab_analytics = st.tabs(
        ["📡 Live Command Center", "🔌 API &amp; Webhooks", "📊 Analytics &amp; ROI"]
    )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — Live Command Center
    # ═══════════════════════════════════════════════════════════════════════
    with tab_live:
        _render_live_command_center()

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — API & Webhooks
    # ═══════════════════════════════════════════════════════════════════════
    with tab_api:
        _render_api_panel()

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — Analytics & ROI
    # ═══════════════════════════════════════════════════════════════════════
    with tab_analytics:
        _render_analytics_panel()


# ---------------------------------------------------------------------------
# Live Command Center sub-panel
# ---------------------------------------------------------------------------
def _render_live_command_center() -> None:
    st.markdown("### 📡 Multi-Camera Live Feed")

    # Mode selector
    col_mode, col_src, col_ctrl = st.columns([2, 3, 1])
    with col_mode:
        mode = st.radio(
            "Input mode",
            ["Static Upload", "Live RTSP Stream", "Demo Simulation"],
            index=["Static Upload", "Live RTSP Stream", "Demo Simulation"].index(
                st.session_state.get("ent_mode", "Demo Simulation")
            ),
            horizontal=True,
            key="ent_mode_radio",
        )
        st.session_state["ent_mode"] = mode

    rtsp_url = ""
    if mode == "Live RTSP Stream":
        with col_src:
            rtsp_url = st.text_input(
                "RTSP / MJPEG URL",
                value="rtsp://admin:admin@192.168.1.10:554/stream0",
                key="ent_rtsp_url",
            )
    elif mode == "Static Upload":
        with col_src:
            uploaded = st.file_uploader("Upload MP4 for live simulation", type=["mp4", "avi", "mov"], key="ent_upload")
            if uploaded:
                tmp = tempfile.mkdtemp()
                p = Path(tmp) / uploaded.name
                p.write_bytes(uploaded.read())
                rtsp_url = str(p)

    with col_ctrl:
        st.markdown("<br>", unsafe_allow_html=True)
        if mode == "Demo Simulation":
            if st.button("🔄 Refresh", key="ent_refresh"):
                for cam in CAMERA_FEEDS:
                    risk = _tick_risk(cam["id"])
                    st.session_state["ent_risk_series"][cam["id"]].append(risk)
                    if len(st.session_state["ent_risk_series"][cam["id"]]) > RISK_HISTORY_LEN:
                        st.session_state["ent_risk_series"][cam["id"]].pop(0)
                    if risk["collision"]:
                        st.session_state["ent_incident_log"].append({
                            "time": time.strftime("%H:%M:%S"),
                            "cam": cam["name"],
                            "probability": f"{risk['probability']:.1%}",
                            "severity": risk["severity"],
                            "incident_type": risk["incident_type"],
                        })
                        st.session_state["ent_accidents_prevented"] = (
                            st.session_state.get("ent_accidents_prevented", 0) + 1
                        )
        else:
            if not st.session_state["ent_stream_running"]:
                if st.button("▶ Start", key="ent_start"):
                    if rtsp_url:
                        mgr = _get_or_create_manager()
                        mgr.start(rtsp_url)
                        st.session_state["ent_stream_running"] = True
                        st.rerun()
            else:
                if st.button("⏹ Stop", key="ent_stop"):
                    mgr = st.session_state.get("ent_stream_manager")
                    if mgr:
                        mgr.stop()
                    st.session_state["ent_stream_running"] = False
                    st.rerun()

    st.divider()

    # ── KPI row ─────────────────────────────────────────────────────────────
    _render_kpi_row()

    st.divider()

    # ── Split-screen camera grid ─────────────────────────────────────────────
    st.markdown("#### Camera Grid")
    grid_cols = st.columns(2)

    for i, cam in enumerate(CAMERA_FEEDS):
        with grid_cols[i % 2]:
            _render_camera_card(cam, mode)

    st.divider()

    # ── Incident log sidebar (inline) ────────────────────────────────────────
    _render_incident_log()


def _render_kpi_row() -> None:
    col1, col2, col3, col4 = st.columns(4)

    incidents = st.session_state.get("ent_incident_log", [])
    prevented = st.session_state.get("ent_accidents_prevented", 0)
    all_risks = []
    for series in st.session_state.get("ent_risk_series", {}).values():
        all_risks.extend([r["probability"] for r in series])

    avg_risk = float(np.mean(all_risks)) if all_risks else 0.0
    peak_risk = float(np.max(all_risks)) if all_risks else 0.0
    roi_usd = int(prevented * ROI_PARAMS["avg_accident_cost_usd"] * ROI_PARAMS["near_miss_prevention_rate"])

    with col1:
        st.metric("🚨 Incidents Logged", len(incidents),
                  delta=f"+{len(incidents)}" if incidents else None,
                  delta_color="inverse")
    with col2:
        st.metric("🛡️ Accidents Prevented", prevented,
                  delta=f"+{prevented}" if prevented else None)
    with col3:
        st.metric("📈 Peak Risk Score", f"{peak_risk:.1%}",
                  delta=f"{avg_risk:.1%} avg", delta_color="inverse")
    with col4:
        st.metric("💰 Est. ROI Savings", f"${roi_usd:,}",
                  delta="↑ YoY" if roi_usd > 0 else None)


def _render_camera_card(cam: dict, mode: str) -> None:
    series = st.session_state["ent_risk_series"].get(cam["id"], [])
    latest_risk = series[-1] if series else None
    prob = latest_risk["probability"] if latest_risk else 0.0
    severity = latest_risk["severity"] if latest_risk else "LOW"
    alert_color = "#ff4444" if severity == "HIGH" else "#ffaa00" if severity == "MEDIUM" else "#00cc88"

    st.markdown(
        f"""
        <div style='border:1px solid {alert_color};border-radius:8px;padding:8px 12px;
                    background:rgba(0,0,0,0.2);margin-bottom:4px'>
          <b style='color:{alert_color}'>{cam['icon']} {cam['name']}</b>
          <span style='color:#888;font-size:0.8rem;float:right'>{cam['location']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mode == "Demo Simulation" and _CV2_AVAILABLE:
        frame = _generate_mock_frame(cam["id"], time.time())
        if _PLOTLY_AVAILABLE:
            gauge = _risk_gauge(prob)
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(_frame_to_png_bytes(frame), caption=f"{cam['name']} — live", use_container_width=True)
            with c2:
                st.plotly_chart(gauge, use_container_width=True, key=f"gauge_{cam['id']}_{int(time.time())}")
        else:
            st.image(_frame_to_png_bytes(frame), caption=cam["name"], use_container_width=True)
            st.progress(prob, text=f"Risk: {prob:.1%}")
    elif mode in ("Live RTSP Stream", "Static Upload"):
        mgr = st.session_state.get("ent_stream_manager")
        if mgr and st.session_state.get("ent_stream_running"):
            snap = mgr.latest_snapshot()
            if snap.latest_frame is not None:
                st.image(_frame_to_png_bytes(snap.latest_frame), caption=cam["name"], use_container_width=True)
            ev = snap.latest_risk
            if ev:
                st.progress(ev.probability, text=f"Risk: {ev.probability:.1%}")
        else:
            st.info("Stream not running. Press ▶ Start above.")
    else:
        st.info("Enable Demo Simulation or start an RTSP stream.")

    if series and _PLOTLY_AVAILABLE:
        sparkline = _risk_sparkline(series[-60:], cam["id"])
        if sparkline:
            st.plotly_chart(sparkline, use_container_width=True, key=f"spark_{cam['id']}_{int(time.time())}")


def _render_incident_log() -> None:
    log = st.session_state.get("ent_incident_log", [])
    st.markdown("#### 🚨 Incident Log")
    if not log:
        st.info("No incidents detected yet. Refresh the simulation or start a stream.")
        return

    rows = list(reversed(log[-20:]))
    header = "| Time | Camera | Probability | Severity | Type |"
    sep    = "|------|--------|-------------|----------|------|"
    lines  = [header, sep]
    for r in rows:
        sev_badge = (
            f"🔴 {r.get('severity','?')}" if r.get("severity") == "HIGH"
            else f"🟡 {r.get('severity','?')}" if r.get("severity") == "MEDIUM"
            else f"🟢 {r.get('severity','?')}"
        )
        lines.append(
            f"| {r.get('time','–')} | {r.get('cam', r.get('cam_id','–'))} "
            f"| {r.get('probability','–')} | {sev_badge} | {r.get('incident_type','–')} |"
        )
    st.markdown("\n".join(lines))


# ---------------------------------------------------------------------------
# API & Webhooks sub-panel
# ---------------------------------------------------------------------------
def _render_api_panel() -> None:
    st.markdown("### 🔌 Enterprise API — Developer Portal")

    col_info, col_demo = st.columns([1, 1])

    with col_info:
        st.markdown(
            """
            **Cosmos Sentinel provides a production-grade REST API for integrating
            AI-powered collision detection into your infrastructure.**

            | Plan | RPM | Price |
            |------|-----|-------|
            | Free | 10 | $0 |
            | Startup | 60 | $299/mo |
            | Enterprise | 200 | Custom |

            #### Authentication
            Pass your key in the `X-API-Key` header:
            ```bash
            curl -H "X-API-Key: YOUR_KEY" \\
                 -F "file=@clip.mp4" \\
                 https://your-space.hf.space/api/analyze/video
            ```

            #### Webhook alerts
            Register a URL to receive real-time alerts when the collision
            probability exceeds your configured threshold:
            ```json
            {
              "event": "high_risk_detection",
              "peak_collision_probability": 0.87,
              "severity": "HIGH",
              "alert_time_sec": 1.4
            }
            ```
            """
        )

    with col_demo:
        st.markdown("#### 🧪 Live API Demo")

        api_key_input = st.selectbox(
            "Select demo API key",
            ["demo-key-free-tier", "sentinel-enterprise-key"],
            key="ent_api_key_demo",
        )

        endpoint = st.selectbox(
            "Endpoint",
            ["/analyze/video (async)", "/analyze/frame (sync)", "/jobs (list)", "/plans (public)"],
            key="ent_endpoint_demo",
        )

        if st.button("▶ Call API", key="ent_call_api"):
            _demo_api_call(api_key_input, endpoint)

        st.markdown("---")
        st.markdown("#### 🔔 Webhook Configuration")
        wh_url = st.text_input(
            "Webhook target URL (e.g. Slack incoming webhook)",
            value="https://hooks.slack.com/services/YOUR/WEBHOOK",
            key="ent_wh_url",
        )
        wh_threshold = st.slider("Alert threshold", 0.0, 1.0, 0.7, 0.05, key="ent_wh_thresh")
        if st.button("Register Webhook", key="ent_wh_register"):
            st.session_state["ent_registered_webhook"] = {
                "url": wh_url, "threshold": wh_threshold
            }
            st.success(f"✅ Webhook registered — alerts fire when P(collision) ≥ {wh_threshold:.0%}")

        if st.session_state.get("ent_registered_webhook"):
            cfg = st.session_state["ent_registered_webhook"]
            st.markdown(
                f"""
                <div style='background:rgba(0,200,100,0.1);border:1px solid rgba(0,200,100,0.4);
                            border-radius:6px;padding:8px 12px;font-size:0.85rem'>
                🔔 Active webhook → <code>{cfg['url'][:50]}…</code><br>
                Threshold: <b>{cfg['threshold']:.0%}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("#### 🚀 Start API Server (background)")
    port = st.number_input("Port", min_value=7860, max_value=8000, value=7861, step=1, key="ent_api_port")
    if st.button("Launch API server (uvicorn)", key="ent_launch_api"):
        _start_api_server_thread(port=int(port))
        st.info(
            f"API server starting on port {port}. "
            f"OpenAPI docs at `http://localhost:{port}/docs`"
        )


def _demo_api_call(api_key: str, endpoint_label: str) -> None:
    """Simulate an API call and display the mock response."""
    import json as _json

    mock_responses = {
        "/analyze/video (async)": {
            "job_id": uuid.uuid4().hex[:12],
            "status": "queued",
            "poll_url": "/jobs/abc123def456",
        },
        "/analyze/frame (sync)": {
            "collision_detected": True,
            "peak_probability": 0.82,
            "alert_time_sec": 1.3,
            "risk_score": 4.1,
            "incident_type": "near_miss_pedestrian",
            "severity": "HIGH",
        },
        "/jobs (list)": [
            {"job_id": "abc123def456", "status": "completed", "created_at": time.time() - 60},
        ],
        "/plans (public)": {
            "free": {"rpm": 10, "features": ["video analysis", "frame analysis"]},
            "startup": {"rpm": 60, "features": ["video analysis", "frame analysis", "webhooks"]},
            "enterprise": {"rpm": 200, "features": ["all features", "SLA", "dedicated support"]},
        },
    }
    resp = mock_responses.get(endpoint_label, {"error": "unknown endpoint"})
    st.code(_json.dumps(resp, indent=2), language="json")
    st.caption(f"🔑 Key: `{api_key}` | Endpoint: `{endpoint_label.split(' ')[0]}`")


# ---------------------------------------------------------------------------
# Analytics & ROI sub-panel
# ---------------------------------------------------------------------------
def _render_analytics_panel() -> None:
    st.markdown("### 📊 Risk Analytics & Enterprise ROI")

    # ── Historical risk summary ──────────────────────────────────────────────
    all_events: list[dict] = []
    for cam_id, series in st.session_state.get("ent_risk_series", {}).items():
        for r in series:
            all_events.append({**r, "cam_id": cam_id})

    col_charts, col_roi = st.columns([3, 2])

    with col_charts:
        if all_events and _PLOTLY_AVAILABLE:
            # Risk distribution by camera
            cam_names = [c["name"] for c in CAMERA_FEEDS]
            cam_ids = [c["id"] for c in CAMERA_FEEDS]
            avg_risks = []
            for cid in cam_ids:
                probs = [r["probability"] for r in all_events if r["cam_id"] == cid]
                avg_risks.append(float(np.mean(probs)) if probs else 0.0)

            fig_bar = go.Figure(go.Bar(
                x=cam_names, y=avg_risks,
                marker_color=["red" if p > 0.65 else "orange" if p > 0.4 else "green" for p in avg_risks],
                text=[f"{p:.1%}" for p in avg_risks],
                textposition="auto",
            ))
            fig_bar.update_layout(
                title="Average Collision Probability by Camera",
                yaxis=dict(range=[0, 1], title="Avg P(collision)"),
                height=250,
                margin=dict(l=0, r=0, t=35, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=10),
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="analytics_bar")

            # Incident heatmap
            inc_fig = _incident_heatmap(st.session_state.get("ent_incident_log", []))
            if inc_fig:
                st.plotly_chart(inc_fig, use_container_width=True, key="analytics_heatmap")
        else:
            st.info("Run the simulation (Demo tab → Refresh) to generate analytics data.")

    with col_roi:
        st.markdown("#### ROI Calculator")
        prevented = st.session_state.get("ent_accidents_prevented", 0)
        acc_cost = st.number_input(
            "Avg accident cost ($)", value=ROI_PARAMS["avg_accident_cost_usd"],
            step=10_000, key="roi_acc_cost"
        )
        prevention_rate = st.slider(
            "Prevention rate", 0.0, 1.0,
            ROI_PARAMS["near_miss_prevention_rate"], 0.05, key="roi_prev_rate"
        )
        monitor_cost = st.number_input(
            "Annual monitoring cost ($)", value=ROI_PARAMS["annual_monitoring_cost_usd"],
            step=1_000, key="roi_monitor_cost"
        )
        saved = int(prevented * acc_cost * prevention_rate)
        net = saved - monitor_cost

        st.metric("Accidents Prevented", prevented)
        st.metric("Gross Savings", f"${saved:,}")
        st.metric("Net ROI", f"${net:,}", delta="Profitable" if net > 0 else "Break-even")

        if _PLOTLY_AVAILABLE:
            roi_params_local = {
                "avg_accident_cost_usd": acc_cost,
                "near_miss_prevention_rate": prevention_rate,
                "annual_monitoring_cost_usd": monitor_cost,
            }
            wf = _roi_waterfall_custom(prevented, roi_params_local)
            if wf:
                st.plotly_chart(wf, use_container_width=True, key="analytics_roi_wf")

    st.divider()

    # ── Market positioning ───────────────────────────────────────────────────
    st.markdown("#### 🇰🇷 Korea Market Fit")
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(
            """
            **🏭 Smart Manufacturing**
            Korean heavy industry (Hyundai, POSCO, Samsung) faces
            strict safety mandates under the Serious Accidents Punishment Act.
            Cosmos Sentinel provides real-time PPE & collision monitoring.
            """
        )
    with col_m2:
        st.markdown(
            """
            **🚦 Smart City Infrastructure**
            Seoul's autonomous traffic management program targets
            zero pedestrian fatalities. Our BADAS detection integrates
            directly with intersection cameras via RTSP.
            """
        )
    with col_m3:
        st.markdown(
            """
            **🤖 Autonomous Vehicle R&D**
            Hyundai Robotics and Kia EV labs require collision
            prediction systems for AV test tracks. Our API provides
            frame-level risk scoring at 30 fps.
            """
        )


def _roi_waterfall_custom(incidents_prevented: int, params: dict) -> "go.Figure | None":
    if not _PLOTLY_AVAILABLE:
        return None
    saved = incidents_prevented * params["avg_accident_cost_usd"] * params["near_miss_prevention_rate"]
    cost = params["annual_monitoring_cost_usd"]
    net = saved - cost
    fig = go.Figure(go.Waterfall(
        name="ROI",
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Monitoring Cost", "Accidents Prevented", "Net ROI"],
        y=[-cost, saved, 0],
        totals={"marker": {"color": "green" if net >= 0 else "red"}},
        connector={"line": {"color": "rgb(63,63,63)"}},
        texttemplate="%{y:$,.0f}",
        textposition="outside",
    ))
    fig.update_layout(
        title="Annual ROI Waterfall",
        height=220,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Stream manager helper
# ---------------------------------------------------------------------------
RISK_HISTORY_LEN = 120


def _get_or_create_manager():
    from livestream_backend import StreamManager
    mgr = st.session_state.get("ent_stream_manager")
    if mgr is None:
        mgr = StreamManager(use_full_pipeline=False)
        st.session_state["ent_stream_manager"] = mgr
    return mgr
