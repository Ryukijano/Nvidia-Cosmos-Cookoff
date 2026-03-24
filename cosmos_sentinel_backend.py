import json
import os
import shutil
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np

from badas_detector import get_badas_model, run_badas_detector
from cosmos_risk_narrator import DEFAULT_REASON_MODEL_NAME, get_reason_model_bundle, run_risk_narrator
from extract_clip import extract_pre_alert_clip
from predict_backend import get_predict_inference, run_predict_bundle

SPACE_ROOT = Path(__file__).resolve().parent
_DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "cosmos_sentinel"
_DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface"


def _ensure_writable_dir(preferred: Path, fallback: Path, label: str) -> Path:
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except PermissionError:
        print(f"⚠️ {label} not writable at {preferred}; falling back to {fallback}")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


CACHE_ROOT = _ensure_writable_dir(
    Path(os.environ.get("COSMOS_SPACE_CACHE_DIR") or _DEFAULT_CACHE_ROOT),
    _DEFAULT_CACHE_ROOT,
    "COSMOS_SPACE_CACHE_DIR",
)
HF_HOME_PATH = _ensure_writable_dir(
    Path(os.environ.get("HF_HOME") or _DEFAULT_HF_HOME),
    _DEFAULT_HF_HOME,
    "HF_HOME",
)
SAMPLE_VIDEO_URL = os.environ.get(
    "COSMOS_SAMPLE_VIDEO_URL",
    "https://raw.githubusercontent.com/Ryukijano/Nvidia-Cosmos-Cookoff/main/1_first.mp4",
)
PREDICT_OUTPUT_ROOT = CACHE_ROOT / "predict_outputs"
PREDICT_MODEL_NAME = os.environ.get("COSMOS_PREDICT_MODEL", "2B/post-trained")

os.environ["HF_HOME"] = str(HF_HOME_PATH)
PREDICT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def existing_file(path):
    if not path:
        return None
    resolved = Path(path).resolve()
    return str(resolved) if resolved.exists() else None


@contextmanager
def working_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous)


def make_run_dir(prefix="pipeline"):
    run_dir = CACHE_ROOT / "runs" / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 100000}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_sample_video():
    sample_dir = CACHE_ROOT / "sample_videos"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "1_first.mp4"
    if not sample_path.exists():
        urllib.request.urlretrieve(SAMPLE_VIDEO_URL, sample_path)
    return str(sample_path)


def cache_uploaded_video(source_path):
    source_candidate = getattr(source_path, "name", source_path)
    source = Path(str(source_candidate))
    if not source.exists():
        raise FileNotFoundError(f"Input video not found: {source}")
    upload_dir = CACHE_ROOT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / f"{int(time.time())}_{source.name}"
    shutil.copy2(source, target)
    return str(target)


def preload_runtime(preload_badas=True, preload_reason=True, preload_predict=False, reason_model_name=DEFAULT_REASON_MODEL_NAME, predict_model_name=PREDICT_MODEL_NAME):
    steps = []
    ensure_sample_video()
    steps.append("Sample video cached")
    if preload_badas:
        get_badas_model()
        steps.append("BADAS model ready")
    if preload_reason:
        get_reason_model_bundle(reason_model_name)
        steps.append(f"Reason model ready: {reason_model_name}")
    if preload_predict:
        try:
            get_predict_inference(predict_model_name, str(PREDICT_OUTPUT_ROOT), True)
            steps.append(f"Predict model ready: {predict_model_name}")
        except Exception as e:
            steps.append(f"Predict model skipped: {e}")
    return "\n".join(steps)


def select_reason_focus_time(badas_result):
    result = badas_result or {}
    prediction_window_summary = result.get("prediction_window_summary") or {}
    top_predictions = result.get("top_predictions") or []
    if prediction_window_summary.get("peak_window_end_time") is not None:
        return float(prediction_window_summary.get("peak_window_end_time"))
    if top_predictions:
        return float(top_predictions[0].get("time_sec", 0.0))
    return float(result.get("alert_time", 0.0) or 0.0)


def extract_frame_at_time(video_path, time_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        cap.release()
        return None
    frame_index = max(0, int(round(time_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def apply_full_frame_risk_overlay(frame, intensity, title):
    if frame is None:
        return None
    overlay = frame.copy()
    heat_color = np.zeros_like(frame)
    heat_color[:, :] = (0, 0, 255)
    alpha = max(0.15, min(0.75, float(intensity)))
    frame = cv2.addWeighted(overlay, 1.0 - alpha, heat_color, alpha, 0)
    cv2.putText(frame, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return frame


def build_bbox_heat_overlay(frame, bboxes, title):
    if frame is None:
        return None
    height, width = frame.shape[:2]
    heat = np.zeros((height, width), dtype=np.float32)
    rendered = frame.copy()
    for label, bbox in (bboxes or {}).items():
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        px1 = max(0, min(width - 1, int(round(x1 * width))))
        py1 = max(0, min(height - 1, int(round(y1 * height))))
        px2 = max(px1 + 1, min(width, int(round(x2 * width))))
        py2 = max(py1 + 1, min(height, int(round(y2 * height))))
        heat[py1:py2, px1:px2] += 1.0
        cv2.rectangle(rendered, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(rendered, str(label), (px1, max(20, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if heat.max() > 0:
        heat = heat / heat.max()
        heat_u8 = np.uint8(255 * heat)
        heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        rendered = cv2.addWeighted(rendered, 0.65, heat_color, 0.35, 0)
    cv2.putText(rendered, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return rendered


def save_frame_strip(frames, output_path, resize_height=220):
    valid_frames = [frame for frame in frames if frame is not None]
    if not valid_frames:
        return None
    resized = []
    for frame in valid_frames:
        height, width = frame.shape[:2]
        scale = resize_height / max(height, 1)
        resized.append(cv2.resize(frame, (max(1, int(round(width * scale))), resize_height)))
    strip = cv2.hconcat(resized)
    cv2.imwrite(str(output_path), strip)
    return existing_file(output_path)


def create_badas_frame_strip(video_path, badas_result, output_path):
    top_predictions = (badas_result or {}).get("top_predictions") or []
    if not top_predictions:
        return None
    frames = []
    for item in top_predictions[:4]:
        frame = extract_frame_at_time(video_path, float(item.get("time_sec", 0.0)))
        if frame is None:
            continue
        frames.append(
            apply_full_frame_risk_overlay(
                frame,
                float(item.get("probability", 0.0)),
                f"BADAS {item.get('time_sec', 0.0):.2f}s | {item.get('probability', 0.0):.1%}",
            )
        )
    return save_frame_strip(frames, output_path)


def create_reason_frame_strip(clip_path, reason_payload, output_path):
    frame_metadata = (reason_payload or {}).get("frame_metadata") or {}
    timestamps = frame_metadata.get("sampled_timestamps_sec") or []
    if not timestamps:
        return None
    bboxes = (reason_payload or {}).get("bboxes") or {}
    if not bboxes:
        return None
    frames = []
    for timestamp in timestamps[:4]:
        frame = extract_frame_at_time(clip_path, float(timestamp))
        if frame is None:
            continue
        frames.append(build_bbox_heat_overlay(frame, bboxes, f"Reason {float(timestamp):.2f}s | bbox focus"))
    return save_frame_strip(frames, output_path)


def create_visualizations(source_video_path, clip_path, badas_result, reason_payload, run_dir=None):
    bboxes = (reason_payload or {}).get("bboxes") or {}
    risk_score = (reason_payload or {}).get("risk_score") or 0
    out_dir = Path(run_dir) if run_dir else Path.cwd()
    bbox_image = out_dir / "bboxes_visualization.png"
    risk_image = out_dir / "risk_visualization.png"
    overlay_gif = out_dir / "video_with_bboxes.gif"
    badas_strip_image = out_dir / "badas_frame_strip.png"
    reason_strip_image = out_dir / "reason_frame_strip.png"
    if bboxes:
        fig, ax = plt.subplots(figsize=(6, 6))
        for label, bbox in bboxes.items():
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, label, fontsize=12, color="red")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Detected Agents Bounding Boxes")
        ax.invert_yaxis()
        plt.savefig(bbox_image)
        plt.close()
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.barh(["Risk Score"], [risk_score], color="orange")
    ax.set_xlim(0, 5)
    ax.set_title("Collision Risk Assessment")
    plt.savefig(risk_image)
    plt.close()
    if bboxes:
        cap = cv2.VideoCapture(str(clip_path))
        frames = []
        frame_count = 0
        max_frames = 20
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            for label, bbox in bboxes.items():
                x1, y1, x2, y2 = bbox
                height, width = frame.shape[:2]
                cv2.rectangle(frame, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1 * width), int(y1 * height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
        cap.release()
        if frames:
            imageio.mimsave(overlay_gif, frames, fps=5, loop=0)
    badas_strip = create_badas_frame_strip(source_video_path, badas_result, badas_strip_image)
    reason_strip = create_reason_frame_strip(clip_path, reason_payload, reason_strip_image)
    return {
        "bbox_image": existing_file(bbox_image) if bboxes else None,
        "risk_image": existing_file(risk_image),
        "overlay_gif": existing_file(overlay_gif) if bboxes else None,
        "badas_frame_strip": badas_strip,
        "reason_frame_strip": reason_strip,
    }


def build_pipeline_overview(badas_result, reason_payload):
    threshold_summary = (badas_result or {}).get("threshold_summary") or {}
    prediction_window_summary = (badas_result or {}).get("prediction_window_summary") or {}
    parsing_summary = (reason_payload or {}).get("parsing_summary") or {}
    frame_metadata = (reason_payload or {}).get("frame_metadata") or {}
    validation = (reason_payload or {}).get("validation") or {}
    validation_flags = validation.get("flags") or {}
    return {
        "collision_gate_triggered": bool((badas_result or {}).get("collision_detected")),
        "alert_time_sec": (badas_result or {}).get("alert_time"),
        "reason_focus_time_sec": select_reason_focus_time(badas_result),
        "alert_confidence": (badas_result or {}).get("confidence"),
        "threshold_crossing_count": threshold_summary.get("threshold_crossing_count", 0),
        "peak_probability": (badas_result or {}).get("valid_prediction_max"),
        "peak_window_average_probability": prediction_window_summary.get("max_average_probability"),
        "incident_type": (reason_payload or {}).get("incident_type"),
        "severity_label": (reason_payload or {}).get("severity_label"),
        "reason_risk_score": (reason_payload or {}).get("risk_score"),
        "reason_bbox_count": (reason_payload or {}).get("bbox_count", 0),
        "reason_prompt_conditioned_by_badas": bool((reason_payload or {}).get("badas_context")),
        "reason_missing_fields": parsing_summary.get("missing_fields", []),
        "reason_processed_frame_count": frame_metadata.get("processed_frame_count"),
        "reason_output_reliable": validation.get("is_reliable"),
        "reason_second_pass_used": validation_flags.get("second_pass_used", False),
        "reason_fallback_override_applied": validation_flags.get("fallback_override_applied", False),
    }


def build_reason_payload(video_path, focus_video_path, badas_context):
    result_text, metadata = run_risk_narrator(video_path, badas_context=badas_context, focus_video_path=focus_video_path)
    payload = metadata.get("parsed_payload") or {}
    payload["video_path"] = video_path
    payload["focus_video_path"] = focus_video_path
    payload["user_prompt"] = metadata["user_prompt"]
    payload["badas_context"] = metadata["badas_context"]
    payload["frame_metadata"] = metadata["frame_metadata"]
    payload["focus_frame_metadata"] = metadata["focus_frame_metadata"]
    payload["video_input_count"] = metadata["video_input_count"]
    payload["model_metadata"] = metadata["model"]
    payload["generation_config"] = metadata["generation_config"]
    payload["input_token_count"] = metadata["input_token_count"]
    payload["output_token_count"] = metadata["output_token_count"]
    payload["text"] = payload.get("text") or result_text
    return payload


def run_pipeline(video_path, include_predict=False, predict_modes=None, predict_model_name=PREDICT_MODEL_NAME):
    run_dir = make_run_dir("pipeline")
    log_lines = ["🚀 Starting Cosmos Sentinel Gradio pipeline", f"Input video: {video_path}"]
    with working_directory(run_dir):
        log_lines.append("📍 Step 1: BADAS V-JEPA2 Collision Detection")
        badas_result = run_badas_detector(video_path)
        log_lines.append("📍 Step 2: Extracting Pre-Alert Clip")
        reason_focus_time = select_reason_focus_time(badas_result)
        clip_path = str(run_dir / "extracted_clip.mp4")
        extracted_clip = extract_pre_alert_clip(video_path, reason_focus_time, clip_path)
        if not extracted_clip:
            raise RuntimeError("Failed to extract BADAS-focused clip")
        log_lines.append("📍 Step 3: Cosmos Reason 2 Risk Analysis")
        reason_payload = build_reason_payload(video_path, clip_path, badas_result)
        visualizations = create_visualizations(video_path, clip_path, badas_result, reason_payload, run_dir=run_dir)
        predict_payload = None
        if include_predict:
            log_lines.append("📍 Step 4: Cosmos Predict continuation")
            selected_modes = predict_modes or ["prevented_continuation", "observed_continuation"]
            predict_payload = run_predict_bundle(
                video_path,
                badas_context=badas_result,
                reason_context=reason_payload,
                modes=selected_modes,
                model_name=predict_model_name,
                output_root=PREDICT_OUTPUT_ROOT / run_dir.name,
                fallback_conditioning_path=clip_path,
            )
        pipeline_payload = {
            "input_video": video_path,
            "pipeline_mode": "badas_reason_predict" if include_predict else "badas_reason_only",
            "iterations": [
                {
                    "iteration": 1,
                    "input_video": video_path,
                    "steps": {
                        "badas": {
                            "success": True,
                            "alert_time": badas_result.get("alert_time"),
                            "reason_focus_time": reason_focus_time,
                            "result": badas_result,
                        },
                        "clip_extraction": {
                            "success": True,
                            "clip_path": existing_file(extracted_clip),
                            "alert_time": badas_result.get("alert_time"),
                            "reason_focus_time": reason_focus_time,
                        },
                        "reason": {
                            "success": True,
                            "full_video_input": video_path,
                            "focus_clip_input": existing_file(extracted_clip),
                            "result": reason_payload,
                            "text": reason_payload.get("text", ""),
                            "visualizations": visualizations,
                        },
                    },
                }
            ],
            "artifacts": {
                "extracted_clip": existing_file(extracted_clip),
                "badas_gradient_saliency": existing_file((badas_result or {}).get("gradient_saliency_image")),
                "bbox_image": visualizations.get("bbox_image"),
                "risk_image": visualizations.get("risk_image"),
                "overlay_gif": visualizations.get("overlay_gif"),
                "badas_frame_strip": visualizations.get("badas_frame_strip"),
                "reason_frame_strip": visualizations.get("reason_frame_strip"),
            },
            "status": "completed",
            "overview": build_pipeline_overview(badas_result, reason_payload),
            "run_directory": str(run_dir),
        }
        if predict_payload:
            pipeline_payload["predict"] = predict_payload
            for artifact_key, artifact_value in (predict_payload.get("artifacts") or {}).items():
                pipeline_payload["artifacts"][artifact_key] = artifact_value
        log_lines.append("🎉 Cosmos Sentinel pipeline completed")
    return {
        "success": True,
        "logs": "\n".join(log_lines),
        "pipeline_payload": pipeline_payload,
        "badas_result": badas_result,
        "reason_result": reason_payload,
        "predict_payload": predict_payload,
        "run_directory": str(run_dir),
    }


def run_predict_only(pipeline_payload, selection="both", predict_model_name=PREDICT_MODEL_NAME):
    if not pipeline_payload:
        raise ValueError("Run BADAS + Reason before Predict")
    iteration = ((pipeline_payload.get("iterations") or [{}])[-1])
    steps = iteration.get("steps") or {}
    badas_result = (steps.get("badas") or {}).get("result") or {}
    reason_result = (steps.get("reason") or {}).get("result") or {}
    artifacts = (pipeline_payload.get("artifacts") or {})
    source_video = pipeline_payload.get("input_video")
    modes = ["prevented_continuation", "observed_continuation"] if selection == "both" else [selection]
    run_dir = make_run_dir("predict")
    predict_payload = run_predict_bundle(
        source_video,
        badas_context=badas_result,
        reason_context=reason_result,
        modes=modes,
        model_name=predict_model_name,
        output_root=PREDICT_OUTPUT_ROOT / run_dir.name,
        fallback_conditioning_path=artifacts.get("extracted_clip"),
    )
    merged = json.loads(json.dumps(pipeline_payload))
    merged["predict"] = predict_payload
    merged_artifacts = merged.get("artifacts") or {}
    for artifact_key, artifact_value in (predict_payload.get("artifacts") or {}).items():
        merged_artifacts[artifact_key] = artifact_value
    merged["artifacts"] = merged_artifacts
    return predict_payload, merged
