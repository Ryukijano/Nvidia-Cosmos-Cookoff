from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

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


def _analyse_video(video_path: str | Path, predictor, frame_stride: int, max_frames: int):
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

    previous_selected_model_key = st.session_state.get("selected_model_key")
    st.session_state["selected_model_key"] = model_key
    if previous_selected_model_key is not None and previous_selected_model_key != model_key:
        manager.unload_model()

    source_summary = get_model_source_summary(model_key)
    manager_status = manager.status()
    st.sidebar.markdown("### Runtime")
    st.sidebar.write(f"Selected model: `{MODEL_LABELS[model_key]}`")
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
    st.sidebar.write(f"Video upload cap: `{STREAMLIT_SERVER_MAX_UPLOAD_MB} MB`")
    st.sidebar.write(f"Working storage free: `{format_bytes(get_workspace_free_bytes())}`")

    if manager_status.is_loaded and manager_status.active_model_label:
        st.sidebar.success(f"Loaded model: {manager_status.active_model_label}")
    else:
        st.sidebar.info("No model is currently loaded.")
    if manager_status.last_error:
        st.sidebar.error(manager_status.last_error)

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

    image_tab, video_tab = st.tabs(["Image", "Video"])

    with image_tab:
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
                    result = predictor.predict(rgb)
                    result["inference_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
                    predictor.reset_state()
                except Exception as exc:
                    st.error(str(exc))
                else:
                    _render_single_result(result)

    with video_tab:
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
                    try:
                        with st.spinner(f"Running {MODEL_LABELS[model_key]} on {uploaded_video.name}..."):
                            predictor = manager.get_predictor(model_key)
                        records, analysis_meta = _analyse_video(
                            temp_path,
                            predictor,
                            frame_stride=frame_stride,
                            max_frames=max_frames,
                        )
                        meta = {
                            **video_meta,
                            **analysis_meta,
                        }
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        _render_video_results(records, meta)
        else:
            _clear_video_stage()


if __name__ == "__main__":
    main()
