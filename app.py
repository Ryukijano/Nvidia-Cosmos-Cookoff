from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from model_registry import MODEL_SPECS, ensure_model_artifacts, get_model_source_summary
from predictor import MODEL_LABELS, PHASE_LABELS, create_predictor, normalize_model_key

st.set_page_config(page_title="DINO-Endo Phase Recognition", layout="wide")


def _phase_index(phase: str) -> int:
    try:
        return PHASE_LABELS.index(phase)
    except ValueError:
        return -1


def _image_to_rgb(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


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
    return "DINO-first Streamlit Hugging Face Space demo for DINO-Endo, AI-Endo, and V-JEPA2."


def _ensure_predictor(model_key: str):
    active_key = st.session_state.get("active_model_key")
    active_predictor = st.session_state.get("active_predictor")

    if active_predictor is not None and active_key != model_key:
        active_predictor.unload()
        st.session_state.pop("active_predictor", None)
        st.session_state.pop("active_model_key", None)

    if st.session_state.get("active_predictor") is None:
        with st.spinner(f"Preparing {MODEL_LABELS[model_key]}..."):
            model_dir = ensure_model_artifacts(model_key)
            predictor = create_predictor(model_key, model_dir=str(model_dir))
            predictor.warm_up()
            st.session_state["active_predictor"] = predictor
            st.session_state["active_model_key"] = model_key

    return st.session_state["active_predictor"]


def _analyse_video(uploaded_file, predictor, frame_stride: int, max_frames: int):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = Path(tmp.name)

    capture = cv2.VideoCapture(str(temp_path))
    if not capture.isOpened():
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("Unable to open uploaded video")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    progress = st.progress(0)
    status = st.empty()

    predictor.reset_state()
    records = []
    processed = 0
    frame_index = 0

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
            result = predictor.predict(rgb)
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
        temp_path.unlink(missing_ok=True)
        predictor.reset_state()

    progress.empty()
    status.empty()
    return records, {"fps": fps, "total_frames": total_frames, "sampled_frames": processed}


def _records_to_frame(records):
    if not records:
        return pd.DataFrame(columns=["frame_index", "timestamp_sec", "phase", "confidence"])
    return pd.DataFrame.from_records(records)


def _download_payloads(df: pd.DataFrame):
    json_payload = df.to_json(orient="records", indent=2).encode("utf-8")
    csv_payload = df.to_csv(index=False).encode("utf-8")
    return json_payload, csv_payload


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
        data=json.dumps(result, indent=2).encode("utf-8"),
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
    dominant_phase, dominant_count = counts.most_common(1)[0]

    metrics = st.columns(4)
    metrics[0].metric("Sampled frames", int(meta["sampled_frames"]))
    metrics[1].metric("Dominant phase", dominant_phase.upper())
    metrics[2].metric("Mean confidence", f"{df['confidence'].mean():.1%}")
    metrics[3].metric("Average inference", f"{df['inference_ms'].mean():.1f} ms")

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

    st.title("DINO-Endo Surgical Phase Recognition")
    st.caption(_space_caption(enabled_model_keys))

    st.sidebar.markdown("### Model")
    if len(enabled_model_keys) == 1:
        model_key = enabled_model_keys[0]
        st.sidebar.write(MODEL_LABELS[model_key])
    else:
        model_key = st.sidebar.selectbox(
            "Model",
            options=enabled_model_keys,
            index=enabled_model_keys.index(default_model_key),
            format_func=lambda key: MODEL_LABELS[key],
        )

    source_summary = get_model_source_summary(model_key)
    st.sidebar.markdown("### Runtime")
    st.sidebar.write(f"CUDA available: `{torch.cuda.is_available()}`")
    if torch.cuda.is_available():
        st.sidebar.write(f"Device: `{torch.cuda.get_device_name(torch.cuda.current_device())}`")
    st.sidebar.write(f"Model dir: `{source_summary['model_dir']}`")
    st.sidebar.write(f"HF repo: `{source_summary['repo_id'] or 'local-only'}`")
    if source_summary["subfolder"]:
        st.sidebar.write(f"Repo subfolder: `{source_summary['subfolder']}`")

    image_tab, video_tab = st.tabs(["Image", "Video"])

    with image_tab:
        uploaded_image = st.file_uploader("Upload an RGB frame", type=["png", "jpg", "jpeg"], key="image-uploader")
        if uploaded_image is not None:
            rgb = _image_to_rgb(uploaded_image)
            st.image(rgb, caption=uploaded_image.name, use_container_width=True)
            if st.button("Run image inference", key="run-image"):
                predictor = _ensure_predictor(model_key)
                predictor.reset_state()
                started = time.perf_counter()
                result = predictor.predict(rgb)
                result["inference_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
                predictor.reset_state()
                _render_single_result(result)

    with video_tab:
        frame_stride = st.slider("Analyze every Nth frame", min_value=1, max_value=30, value=5, step=1)
        max_frames = st.slider("Maximum sampled frames", min_value=10, max_value=600, value=180, step=10)
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=["mp4", "mov", "avi", "mkv", "webm", "m4v"],
            key="video-uploader",
        )
        if uploaded_video is not None:
            st.video(uploaded_video)
            if st.button("Analyze video", key="run-video"):
                predictor = _ensure_predictor(model_key)
                records, meta = _analyse_video(uploaded_video, predictor, frame_stride=frame_stride, max_frames=max_frames)
                _render_video_results(records, meta)

    if st.sidebar.button("Unload active model"):
        predictor = st.session_state.get("active_predictor")
        if predictor is not None:
            predictor.unload()
            st.session_state.pop("active_predictor", None)
            st.session_state.pop("active_model_key", None)
        st.sidebar.success("Model unloaded")


if __name__ == "__main__":
    main()
