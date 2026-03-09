from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from explainability import ExplainabilitySpec
from model_manager import SpaceModelManager
from model_registry import MODEL_SPECS, get_model_source_summary
from predictor import MODEL_LABELS, PHASE_LABELS, normalize_model_key
from video_utils import (
    STREAMLIT_SERVER_MAX_UPLOAD_MB,
    SUPPORTED_VIDEO_TYPES,
    format_bytes,
    get_upload_size_bytes,
    get_workspace_free_bytes,
    probe_video_info,
    recommended_frame_stride,
    should_show_inline_preview,
    spool_uploaded_video,
)

st.set_page_config(page_title="Ryukijano's Project Portfolio", layout="wide")

MODEL_OPTION_LABELS = {
    "aiendo": "AI-Endo",
    "dinov2": "DINO-Endo",
    "vjepa2": "V-JEPA2 (slower first load)",
}

MODEL_LOAD_NOTES = {
    "aiendo": "AI-Endo uses the ResNet + MS-TCN + Transformer stack.",
    "dinov2": "DINO-Endo remains the default public model in this demo.",
    "vjepa2": "V-JEPA2 can take longer on the first load because the encoder checkpoint is several gigabytes.",
}

FALLBACK_EXPLAINABILITY_SPECS = {
    "aiendo": ExplainabilitySpec(
        encoder_mode="proxy",
        encoder_label="ResNet layer4 activation energy (proxy)",
        decoder_mode="attention",
        decoder_label="Temporal Transformer attention",
    ),
    "dinov2": ExplainabilitySpec(
        encoder_mode="attention",
        encoder_label="DINOv2 encoder self-attention",
        decoder_mode="attention",
        decoder_label="Fusion Transformer temporal attention",
        encoder_layer_count=12,
        encoder_head_count=6,
    ),
    "vjepa2": ExplainabilitySpec(
        encoder_mode="attention",
        encoder_label="V-JEPA2 encoder self-attention",
        decoder_mode="proxy",
        decoder_label="MLP decoder feature energy (proxy)",
        encoder_layer_count=24,
        encoder_head_count=16,
    ),
}


SPACE_TITLE = "Ryukijano's Project Portfolio"
FEATURED_PROJECT_TITLE = "DINO-Endo Surgery Workspace"
MODEL_SLIDER_KEY = "workspace-model-slider"
SELECTED_MODEL_STATE_KEY = "selected_model_key"


@dataclass(frozen=True)
class HostedProject:
    key: str
    title: str
    status: str
    summary: str
    highlights: tuple[str, ...]
    tags: tuple[str, ...]


HOSTED_PROJECTS = (
    HostedProject(
        key="dino-endo-surgery",
        title=FEATURED_PROJECT_TITLE,
        status="Live now",
        summary=(
            "Upload single frames or full videos, swap between DINO-Endo, AI-Endo, and V-JEPA2, "
            "and inspect optional explainability overlays inside one surgical phase-recognition workspace."
        ),
        highlights=(
            "Large video uploads with on-disk staging",
            "One-click JSON and CSV export",
            "Live encoder and decoder explainability",
            "Manual load and unload for GPU-safe model switching",
        ),
        tags=("Computer vision", "Medical video", "Multi-model inference"),
    ),
)


def _phase_index(phase: str) -> int:
    try:
        return PHASE_LABELS.index(phase)
    except ValueError:
        return -1


def _image_to_rgb(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def _model_option_label(model_key: str) -> str:
    return MODEL_OPTION_LABELS.get(model_key, MODEL_LABELS.get(model_key, model_key))


def _enabled_model_keys() -> list[str]:
    configured = os.getenv("SPACE_ENABLED_MODELS", "").strip()
    if not configured:
        return list(MODEL_SPECS.keys())

    enabled_keys = []
    seen = set()
    for token in configured.split(","):
        raw = token.strip()
        if not raw:
            continue
        normalized = normalize_model_key(raw)
        if normalized not in MODEL_SPECS:
            raise RuntimeError(f"SPACE_ENABLED_MODELS contains unsupported model '{raw}'")
        if normalized not in seen:
            enabled_keys.append(normalized)
            seen.add(normalized)

    if not enabled_keys:
        raise RuntimeError("SPACE_ENABLED_MODELS did not resolve to any supported models")
    return enabled_keys


def _default_model_key(enabled_model_keys: list[str]) -> str:
    configured = os.getenv("SPACE_DEFAULT_MODEL", "").strip()
    if not configured:
        return "dinov2" if "dinov2" in enabled_model_keys else enabled_model_keys[0]

    normalized = normalize_model_key(configured)
    if normalized not in enabled_model_keys:
        raise RuntimeError(
            f"SPACE_DEFAULT_MODEL '{configured}' is not enabled by SPACE_ENABLED_MODELS"
        )
    return normalized


def _space_caption(enabled_model_keys: list[str]) -> str:
    if enabled_model_keys == ["dinov2"]:
        return "Streamlit Hugging Face Space demo for the DINO-Endo phase-recognition stack."
    return "Streamlit Hugging Face Space demo for DINO-Endo, AI-Endo, and V-JEPA2 with one active model loaded at a time."


def _inject_app_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 2rem;
        }

        .hub-hero,
        .hub-card,
        .workspace-card {
            border-radius: 22px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.86), rgba(15, 23, 42, 0.66));
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
        }

        .hub-hero {
            padding: 2rem 2.25rem;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.18), rgba(16, 185, 129, 0.18), rgba(15, 23, 42, 0.9));
        }

        .hub-eyebrow {
            margin: 0;
            color: #67e8f9;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }

        .hub-hero h1,
        .workspace-card h2,
        .hub-card h3 {
            margin: 0.4rem 0 0 0;
            color: #f8fafc;
        }

        .hub-subtitle,
        .workspace-copy,
        .hub-card p,
        .hub-card li {
            color: rgba(226, 232, 240, 0.92);
            line-height: 1.55;
        }

        .hub-subtitle {
            margin-top: 0.8rem;
            max-width: 62rem;
            font-size: 1.03rem;
        }

        .hub-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .hub-chip,
        .hub-status {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.32rem 0.78rem;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .hub-chip {
            background: rgba(15, 23, 42, 0.56);
            border: 1px solid rgba(103, 232, 249, 0.24);
            color: #e2e8f0;
        }

        .hub-status {
            background: rgba(34, 197, 94, 0.18);
            border: 1px solid rgba(34, 197, 94, 0.28);
            color: #bbf7d0;
            margin-bottom: 0.7rem;
        }

        .hub-card,
        .workspace-card {
            padding: 1.25rem 1.4rem;
            height: 100%;
        }

        .hub-card ul {
            margin: 0.8rem 0 0 1rem;
            padding: 0;
        }

        .workspace-card {
            margin: 0.3rem 0 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hub_chips(labels: list[str] | tuple[str, ...]) -> str:
    return "".join(f'<span class="hub-chip">{label}</span>' for label in labels)


def _render_project_hub(enabled_model_keys: list[str]) -> None:
    featured = HOSTED_PROJECTS[0]
    enabled_labels = [_model_option_label(key) for key in enabled_model_keys]
    st.markdown(
        f"""
        <section class="hub-hero">
            <p class="hub-eyebrow">Ryukijano portfolio</p>
            <h1>{SPACE_TITLE}</h1>
            <p class="hub-subtitle">
                A polished portfolio shell for applied vision demos. {FEATURED_PROJECT_TITLE} is the first live workspace,
                and the layout is ready to host more projects later without rebuilding the overall site shell.
            </p>
            <div class="hub-chip-row">
                {_render_hub_chips(tuple(enabled_labels) + ("Portfolio ready", "Streamlit + Docker Space"))}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    metrics = st.columns(4)
    metrics[0].metric("Hosted projects", len(HOSTED_PROJECTS))
    metrics[1].metric("Model families", len(enabled_model_keys))
    metrics[2].metric("Explainability", "Opt-in")
    metrics[3].metric("Exports", "JSON + CSV")

    left_col, right_col = st.columns([1.8, 1.2], gap="large")
    with left_col:
        highlights_html = "".join(f"<li>{item}</li>" for item in featured.highlights)
        st.markdown(
            f"""
            <section class="hub-card">
                <span class="hub-status">{featured.status}</span>
                <h3>{featured.title}</h3>
                <p>{featured.summary}</p>
                <div class="hub-chip-row">{_render_hub_chips(featured.tags)}</div>
                <ul>{highlights_html}</ul>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            """
            <section class="hub-card">
                <span class="hub-status">Portfolio shell</span>
                <h3>Ready for more demos</h3>
                <p>
                    The top section now works as a reusable portfolio shell instead of a one-off page. Add more project cards
                    and workspace blocks here later, while keeping one shared brand, layout, and deployment target.
                </p>
                <ul>
                    <li>Keep each project's controls inside its own workspace section.</li>
                    <li>Reuse the same landing-page hero, metrics, and project-card layout.</li>
                    <li>Preserve one-model-at-a-time loading so future demos stay GPU-friendly.</li>
                </ul>
            </section>
            """,
            unsafe_allow_html=True,
        )


def _render_workspace_header(enabled_model_keys: list[str], model_key: str) -> None:
    selected_label = _model_option_label(model_key)
    selection_note = (
        "Use the model slider to move between DINO-Endo, AI-Endo, and V-JEPA2. "
        "Only one model stays loaded at a time so the Space remains responsive on shared GPU hardware."
    )
    st.markdown(
        f"""
        <section class="workspace-card">
            <p class="hub-eyebrow">Featured project</p>
            <h2>{FEATURED_PROJECT_TITLE}</h2>
            <p class="workspace-copy">
                {selection_note}
            </p>
            <div class="hub-chip-row">
                {_render_hub_chips(tuple(_model_option_label(key) for key in enabled_model_keys))}
                <span class="hub-chip">Selected: {selected_label}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _resolve_model_selection(enabled_model_keys: list[str], default_model_key: str) -> tuple[str | None, str]:
    previous_selected_model_key = st.session_state.get(SELECTED_MODEL_STATE_KEY)
    current_slider_value = st.session_state.get(MODEL_SLIDER_KEY)
    if current_slider_value not in enabled_model_keys:
        st.session_state[MODEL_SLIDER_KEY] = default_model_key

    if len(enabled_model_keys) == 1:
        model_key = enabled_model_keys[0]
        st.session_state[MODEL_SLIDER_KEY] = model_key
        return previous_selected_model_key, model_key

    model_key = st.select_slider(
        "Project model slider",
        options=enabled_model_keys,
        key=MODEL_SLIDER_KEY,
        format_func=_model_option_label,
        help="Prominent model-family slider for the DINO-Endo project workspace.",
    )
    return previous_selected_model_key, model_key


def _get_model_manager() -> SpaceModelManager:
    manager = st.session_state.get("model_manager")
    if manager is None:
        manager = SpaceModelManager()
        st.session_state["model_manager"] = manager
    return manager


def _clear_video_stage() -> None:
    temp_path = st.session_state.pop("staged_video_path", None)
    if temp_path is not None:
        Path(temp_path).unlink(missing_ok=True)
    st.session_state.pop("staged_video_signature", None)
    st.session_state.pop("staged_video_meta", None)


def _prepare_staged_video(uploaded_file):
    upload_size_bytes = get_upload_size_bytes(uploaded_file)
    signature = (
        getattr(uploaded_file, "name", "upload"),
        upload_size_bytes,
        getattr(uploaded_file, "type", ""),
    )
    staged_path = st.session_state.get("staged_video_path")
    staged_signature = st.session_state.get("staged_video_signature")
    if staged_signature == signature and staged_path is not None and Path(staged_path).exists():
        return Path(staged_path), st.session_state["staged_video_meta"]

    _clear_video_stage()
    temp_path = spool_uploaded_video(uploaded_file)
    try:
        meta = probe_video_info(temp_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    st.session_state["staged_video_signature"] = signature
    st.session_state["staged_video_path"] = str(temp_path)
    st.session_state["staged_video_meta"] = meta
    return temp_path, meta


def _records_to_frame(records):
    if not records:
        return pd.DataFrame(columns=["frame_index", "timestamp_sec", "phase", "confidence"])
    return pd.DataFrame.from_records(records)


def _download_payloads(df: pd.DataFrame):
    json_payload = df.to_json(orient="records", indent=2).encode("utf-8")
    csv_payload = df.to_csv(index=False).encode("utf-8")
    return json_payload, csv_payload


def _get_explainability_spec(manager: SpaceModelManager, model_key: str) -> ExplainabilitySpec:
    predictor = manager.get_loaded_predictor(model_key)
    if predictor is not None and hasattr(predictor, "get_explainability_spec"):
        return predictor.get_explainability_spec()
    return FALLBACK_EXPLAINABILITY_SPECS[model_key]


def _build_explainability_config(manager: SpaceModelManager, model_key: str):
    spec = _get_explainability_spec(manager, model_key)
    st.sidebar.markdown("### Explainability")
    enabled = st.sidebar.toggle(
        "Enable live encoder/decoder maps",
        value=False,
        help="Shows encoder heatmaps and decoder temporal strips on every processed frame. Leave this off if you want the fastest video analysis path.",
    )
    config = {"enabled": enabled}
    if not enabled:
        return config, spec

    st.sidebar.caption(f"Encoder view: {spec.encoder_label}")
    st.sidebar.caption(f"Decoder view: {spec.decoder_label}")
    if spec.encoder_mode == "attention" and spec.encoder_layer_count > 0 and spec.encoder_head_count > 0:
        default_layer = spec.encoder_layer_count - 1
        config["encoder_layer"] = st.sidebar.slider(
            "Encoder layer",
            min_value=1,
            max_value=spec.encoder_layer_count,
            value=default_layer + 1,
            key=f"explainability-layer-{model_key}",
        ) - 1
        config["encoder_head"] = st.sidebar.slider(
            "Encoder head",
            min_value=1,
            max_value=spec.encoder_head_count,
            value=1,
            key=f"explainability-head-{model_key}",
        ) - 1
    else:
        st.sidebar.info("This model uses a proxy encoder overlay instead of true encoder attention.")

    st.sidebar.caption("Decoder strips are rendered as temporal heat strips rather than projected back onto the frame.")
    return config, spec


def _render_explainability_panel(target, payload: dict | None, *, enabled: bool, spec: ExplainabilitySpec, title: str) -> None:
    with target.container():
        st.markdown(f"### {title}")
        if not enabled:
            st.caption("Turn on the explainability toggle in the sidebar to inspect encoder heatmaps and decoder temporal strips.")
            return

        st.caption(f"Encoder default: {spec.encoder_label}")
        st.caption(f"Decoder default: {spec.decoder_label}")
        if payload is None:
            st.info("Run image or video inference to populate this live explainability panel.")
            return

        layer_index = payload.get("encoder_layer")
        head_index = payload.get("encoder_head")
        encoder_caption = f"{payload['encoder_label']} ({payload['encoder_kind']})"
        if layer_index is not None and head_index is not None:
            encoder_caption += f" · layer {int(layer_index) + 1}, head {int(head_index) + 1}"
        st.caption(encoder_caption)
        st.image(payload["encoder_visualization"], use_container_width=True)

        st.caption(f"{payload['decoder_label']} ({payload['decoder_kind']})")
        st.image(payload["decoder_visualization"], use_container_width=True)

        notes = payload.get("notes")
        if notes:
            st.caption(notes)


def _analyse_video(
    video_path: str | Path,
    predictor,
    frame_stride: int,
    max_frames: int,
    *,
    explainability_config: dict | None = None,
    explainability_callback=None,
):
    temp_path = Path(video_path)
    capture = cv2.VideoCapture(str(temp_path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open uploaded video")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    progress = st.progress(0)
    status = st.empty()

    predictor.reset_state()
    records = []
    processed = 0
    frame_index = 0
    explain_enabled = bool(explainability_config and explainability_config.get("enabled"))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            result = predictor.predict(rgb, explainability=explainability_config if explain_enabled else None)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            probs = result.get("probs", [0.0, 0.0, 0.0, 0.0])
            record = {
                "frame_index": frame_index,
                "timestamp_sec": round(frame_index / fps, 3) if fps > 0 else None,
                "phase": result.get("phase", "unknown"),
                "phase_id": _phase_index(result.get("phase", "unknown")),
                "confidence": float(result.get("confidence", 0.0)),
                "frames_used": int(result.get("frames_used", processed + 1)),
                "idle": float(probs[0]) if len(probs) > 0 else 0.0,
                "marking": float(probs[1]) if len(probs) > 1 else 0.0,
                "injection": float(probs[2]) if len(probs) > 2 else 0.0,
                "dissection": float(probs[3]) if len(probs) > 3 else 0.0,
                "inference_ms": round(elapsed_ms, 3),
            }
            records.append(record)
            processed += 1

            if explain_enabled and explainability_callback is not None:
                explainability_callback(result.get("explainability"), processed, frame_index)

            if total_frames > 0:
                progress.progress(min(frame_index + 1, total_frames) / total_frames)
            else:
                progress.progress(min(processed / max_frames, 1.0))
            status.caption(f"Processed {processed} sampled frames")

            frame_index += 1
            if processed >= max_frames:
                break
    finally:
        capture.release()
        predictor.reset_state()

    progress.empty()
    status.empty()
    return records, {"fps": fps, "total_frames": total_frames, "sampled_frames": processed}


def _render_single_result(result: dict):
    probs = result.get("probs", [0.0, 0.0, 0.0, 0.0])
    metrics = st.columns(3)
    metrics[0].metric("Predicted phase", result.get("phase", "unknown").upper())
    metrics[1].metric("Confidence", f"{float(result.get('confidence', 0.0)):.1%}")
    metrics[2].metric("Frames used", int(result.get("frames_used", 1)))

    prob_df = pd.DataFrame({"phase": list(PHASE_LABELS), "probability": probs})
    st.bar_chart(prob_df.set_index("phase"))
    st.download_button(
        label="Download JSON",
        data=json.dumps(result, indent=2, default=str).encode("utf-8"),
        file_name="phase_prediction.json",
        mime="application/json",
        key="download-single-json",
    )


def _render_video_results(records, meta):
    if not records:
        st.warning("No frames were processed from the uploaded video.")
        return

    df = _records_to_frame(records)
    counts = Counter(df["phase"].tolist())
    dominant_phase, _ = counts.most_common(1)[0]

    metrics = st.columns(4)
    metrics[0].metric("Sampled frames", int(meta["sampled_frames"]))
    metrics[1].metric("Dominant phase", dominant_phase.upper())
    metrics[2].metric("Mean confidence", f"{df['confidence'].mean():.1%}")
    metrics[3].metric("Average inference", f"{df['inference_ms'].mean():.1f} ms")

    detail_cols = st.columns(5)
    detail_cols[0].metric("File size", meta.get("file_size_label", "Unknown"))
    detail_cols[1].metric("Duration", meta.get("duration_label", "Unknown"))
    detail_cols[2].metric("FPS", f"{meta.get('fps', 0.0):.2f}" if meta.get("fps") else "Unknown")
    detail_cols[3].metric("Frames", int(meta.get("total_frames", meta.get("frame_count", 0))))
    detail_cols[4].metric("Resolution", meta.get("resolution_label", "Unknown"))

    chart_df = df.copy()
    if "timestamp_sec" in chart_df and chart_df["timestamp_sec"].notna().any():
        chart_df = chart_df.set_index("timestamp_sec")
    else:
        chart_df = chart_df.set_index("frame_index")

    st.subheader("Confidence timeline")
    st.line_chart(chart_df[["confidence"]])

    st.subheader("Phase timeline")
    st.line_chart(chart_df[["phase_id"]])

    st.subheader("Per-frame predictions")
    st.dataframe(df, use_container_width=True, hide_index=True)

    json_payload, csv_payload = _download_payloads(df)
    left, right = st.columns(2)
    left.download_button("Download JSON", json_payload, file_name="phase_timeline.json", mime="application/json")
    right.download_button("Download CSV", csv_payload, file_name="phase_timeline.csv", mime="text/csv")


def main():
    enabled_model_keys = _enabled_model_keys()
    default_model_key = _default_model_key(enabled_model_keys)
    manager = _get_model_manager()
    _inject_app_styles()
    _render_project_hub(enabled_model_keys)
    previous_selected_model_key, model_key = _resolve_model_selection(enabled_model_keys, default_model_key)

    _render_workspace_header(enabled_model_keys, model_key)
    st.caption(_space_caption(enabled_model_keys))

    st.session_state[SELECTED_MODEL_STATE_KEY] = model_key
    if previous_selected_model_key is not None and previous_selected_model_key != model_key:
        manager.unload_model()

    explainability_config, explainability_spec = _build_explainability_config(manager, model_key)

    source_summary = get_model_source_summary(model_key)
    st.sidebar.markdown("### Runtime")
    st.sidebar.write(f"Selected model: `{MODEL_LABELS[model_key]}`")
    st.sidebar.caption(MODEL_LOAD_NOTES[model_key])
    st.sidebar.write(f"CUDA available: `{torch.cuda.is_available()}`")
    if torch.cuda.is_available():
        st.sidebar.write(f"Device: `{torch.cuda.get_device_name(torch.cuda.current_device())}`")
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            st.sidebar.write(
                f"GPU memory free: `{format_bytes(free_bytes)}` / `{format_bytes(total_bytes)}`"
            )
        except RuntimeError:
            pass
    st.sidebar.write(f"Model dir: `{source_summary['model_dir']}`")
    st.sidebar.write(f"HF repo: `{source_summary['repo_id'] or 'local-only'}`")
    if source_summary["subfolder"]:
        st.sidebar.write(f"Repo subfolder: `{source_summary['subfolder']}`")
    with st.sidebar.expander("Checkpoint requirements", expanded=False):
        st.write(", ".join(source_summary["required_files"]))
        if source_summary["optional_files"]:
            st.caption("Optional: " + ", ".join(source_summary["optional_files"]))
    st.sidebar.write(f"Video upload cap: `{STREAMLIT_SERVER_MAX_UPLOAD_MB} MB`")
    st.sidebar.write(f"Working storage free: `{format_bytes(get_workspace_free_bytes())}`")

    prepare_col, unload_col = st.sidebar.columns(2)
    if prepare_col.button("Load model", use_container_width=True):
        try:
            with st.spinner(f"Preparing {MODEL_LABELS[model_key]}..."):
                manager.get_predictor(model_key)
        except Exception as exc:
            st.sidebar.error(str(exc))
        else:
            st.sidebar.success(f"{MODEL_LABELS[model_key]} is ready.")
    if unload_col.button("Unload", use_container_width=True):
        manager.unload_model()
        st.sidebar.success("Model unloaded")

    manager_status = manager.status()
    if manager_status.is_loaded and manager_status.active_model_label:
        st.sidebar.success(f"Loaded model: {manager_status.active_model_label}")
    else:
        st.sidebar.info("No model is currently loaded.")
    if manager_status.last_error:
        st.sidebar.error(manager_status.last_error)

    image_tab, video_tab = st.tabs(["Image", "Video"])

    with image_tab:
        image_main_col, image_explain_col = st.columns([3, 2], gap="large")
        image_explain_placeholder = image_explain_col.empty()
        image_result = None

        with image_main_col:
            uploaded_image = st.file_uploader("Upload an RGB frame", type=["png", "jpg", "jpeg"], key="image-uploader")
            if uploaded_image is not None:
                rgb = _image_to_rgb(uploaded_image)
                st.image(rgb, caption=uploaded_image.name, use_container_width=True)
                if st.button("Run image inference", key="run-image"):
                    try:
                        with st.spinner(f"Running {MODEL_LABELS[model_key]} on {uploaded_image.name}..."):
                            predictor = manager.get_predictor(model_key)
                        predictor.reset_state()
                        started = time.perf_counter()
                        image_result = predictor.predict(
                            rgb,
                            explainability=explainability_config if explainability_config.get("enabled") else None,
                        )
                        image_result["inference_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
                        predictor.reset_state()
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        _render_single_result(image_result)

        _render_explainability_panel(
            image_explain_placeholder,
            image_result.get("explainability") if image_result else None,
            enabled=bool(explainability_config.get("enabled")),
            spec=explainability_spec,
            title="Explainability",
        )

    with video_tab:
        video_main_col, video_explain_col = st.columns([3, 2], gap="large")
        video_explain_placeholder = video_explain_col.empty()
        _render_explainability_panel(
            video_explain_placeholder,
            None,
            enabled=bool(explainability_config.get("enabled")),
            spec=explainability_spec,
            title="Explainability",
        )

        with video_main_col:
            frame_stride = st.slider("Analyze every Nth frame", min_value=1, max_value=30, value=5, step=1)
            max_frames = st.slider("Maximum sampled frames", min_value=10, max_value=600, value=180, step=10)
            uploaded_video = st.file_uploader(
                "Upload a video (MP4 preferred)",
                type=SUPPORTED_VIDEO_TYPES,
                key="video-uploader",
                help=(
                    f"Single-file uploads are enabled up to {STREAMLIT_SERVER_MAX_UPLOAD_MB} MB. "
                    "MP4 is preferred; MOV/AVI/MKV/WEBM/M4V stay enabled as fallback containers."
                ),
                max_upload_size=STREAMLIT_SERVER_MAX_UPLOAD_MB,
            )
            if uploaded_video is not None:
                try:
                    temp_path, video_meta = _prepare_staged_video(uploaded_video)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    info_cols = st.columns(5)
                    info_cols[0].metric("File size", video_meta["file_size_label"])
                    info_cols[1].metric("Duration", video_meta["duration_label"])
                    info_cols[2].metric("FPS", f"{video_meta.get('fps', 0.0):.2f}" if video_meta.get("fps") else "Unknown")
                    info_cols[3].metric("Frames", int(video_meta.get("frame_count", 0)))
                    info_cols[4].metric("Resolution", video_meta["resolution_label"])
                    if video_meta.get("format_name"):
                        st.caption(f"Container detected by ffprobe: {video_meta['format_name']}")

                    recommended_stride = recommended_frame_stride(video_meta.get("duration_seconds"))
                    st.caption(
                        f"Recommended frame stride for this video: every {recommended_stride} frame(s). "
                        "Use higher values for very long videos to keep analysis times reasonable."
                    )

                    if should_show_inline_preview(video_meta["file_size_bytes"]):
                        st.video(uploaded_video)
                    else:
                        st.info(
                            "Inline preview is disabled for uploads larger than "
                            "256 MB to avoid pushing very large media back through the browser. "
                            "The staged video on disk is still used for analysis."
                        )

                    if st.button("Analyze video", key="run-video"):
                        latest_payload = {"value": None}

                        def _video_explainability_callback(payload, processed_count: int, current_frame_index: int):
                            latest_payload["value"] = payload
                            _render_explainability_panel(
                                video_explain_placeholder,
                                payload,
                                enabled=True,
                                spec=explainability_spec,
                                title=f"Live explainability · sampled frame {processed_count}",
                            )

                        try:
                            with st.spinner(f"Running {MODEL_LABELS[model_key]} on {uploaded_video.name}..."):
                                predictor = manager.get_predictor(model_key)
                            records, analysis_meta = _analyse_video(
                                temp_path,
                                predictor,
                                frame_stride=frame_stride,
                                max_frames=max_frames,
                                explainability_config=explainability_config if explainability_config.get("enabled") else None,
                                explainability_callback=(
                                    _video_explainability_callback
                                    if explainability_config.get("enabled")
                                    else None
                                ),
                            )
                            meta = {
                                **video_meta,
                                **analysis_meta,
                            }
                        except Exception as exc:
                            st.error(str(exc))
                        else:
                            _render_video_results(records, meta)
                            if explainability_config.get("enabled"):
                                _render_explainability_panel(
                                    video_explain_placeholder,
                                    latest_payload["value"],
                                    enabled=True,
                                    spec=explainability_spec,
                                    title="Explainability",
                                )
            else:
                _clear_video_stage()


if __name__ == "__main__":
    main()
