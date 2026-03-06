import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2

VENDORED_PREDICT_PATH = Path(__file__).resolve().parent / "cosmos-predict2.5"
OUTPUT_ROOT = Path("./predict_outputs")


def existing_file(path):
    return str(path) if path and Path(path).exists() else None


def sanitize_text(value):
    return " ".join(str(value or "").strip().split())


def select_predict_focus_time(badas_context, reason_context):
    reason_critical = reason_context.get("critical_risk_time")
    if isinstance(reason_critical, (int, float)):
        return float(reason_critical)
    prediction_window = (badas_context or {}).get("prediction_window_summary") or {}
    peak_start = prediction_window.get("peak_window_start_time")
    peak_end = prediction_window.get("peak_window_end_time")
    if isinstance(peak_start, (int, float)) and isinstance(peak_end, (int, float)):
        return float((peak_start + peak_end) / 2.0)
    top_predictions = (badas_context or {}).get("top_predictions") or []
    if top_predictions:
        return float(top_predictions[0].get("time_sec") or 0.0)
    alert_time = (badas_context or {}).get("alert_time")
    if isinstance(alert_time, (int, float)):
        return float(alert_time)
    return 0.0


def build_conditioning_window(badas_context, reason_context):
    focus_time = select_predict_focus_time(badas_context or {}, reason_context or {})
    prediction_window = (badas_context or {}).get("prediction_window_summary") or {}
    peak_start = prediction_window.get("peak_window_start_time")
    if isinstance(peak_start, (int, float)):
        start_time = max(0.0, min(float(peak_start), focus_time - 0.50))
    else:
        start_time = max(0.0, focus_time - 1.0)
    frame_spacing_sec = 0.25
    frame_count = 5
    end_time = start_time + frame_spacing_sec * (frame_count - 1)
    return {
        "focus_time_sec": float(focus_time),
        "start_time_sec": float(start_time),
        "end_time_sec": float(end_time),
        "frame_spacing_sec": float(frame_spacing_sec),
        "frame_count": int(frame_count),
    }


def build_conditioning_clip(source_video_path, window, output_path):
    source_video_path = str(source_video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(source_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = float(total_frames / fps) if fps else 0.0
    timestamps = []
    frames = []
    for idx in range(int(window["frame_count"])):
        timestamp = min(duration_sec, float(window["start_time_sec"]) + idx * float(window["frame_spacing_sec"]))
        frame_index = max(0, int(round(timestamp * fps))) if fps else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        timestamps.append(float(timestamp))
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames available for Cosmos Predict conditioning clip")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return {
        "clip_path": str(output_path),
        "frame_timestamps_sec": timestamps,
        "frame_count": int(len(frames)),
        "width": int(width),
        "height": int(height),
    }


def build_fallback_conditioning_metadata(fallback_conditioning_path):
    clip_path = Path(fallback_conditioning_path)
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "clip_path": str(clip_path),
        "frame_timestamps_sec": [],
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
    }


def infer_preventive_action(reason_context):
    text_candidates = [
        sanitize_text(reason_context.get("explanation")),
        sanitize_text(reason_context.get("counterfactual_prompt")),
        sanitize_text(reason_context.get("scene_summary")),
    ]
    combined = " ".join(candidate.lower() for candidate in text_candidates if candidate)
    if any(token in combined for token in ["brake", "braking", "slow", "slowing", "stop", "stopped"]):
        return "the visible braking and speed reduction continue"
    if any(token in combined for token in ["yield", "yielding", "gave way"]):
        return "the yielding behavior continues and conflict space is cleared"
    if any(token in combined for token in ["steer", "steering", "swerve", "lane correction", "turn away"]):
        return "the evasive steering correction continues and the vehicles maintain separation"
    if (reason_context.get("incident_type") or "").strip().lower() == "near_miss":
        return "the evasive action visible in the near-miss continues and prevents contact"
    return "the most plausible evasive action visible in the scene continues and reduces the chance of collision"


def build_predict_prompt(badas_context, reason_context, mode, window):
    scene_summary = sanitize_text(reason_context.get("scene_summary")) or "A traffic interaction is developing at a monitored road junction."
    incident_type = sanitize_text(reason_context.get("incident_type")) or "unclear"
    severity_label = sanitize_text(reason_context.get("severity_label")) or "unknown"
    explanation = sanitize_text(reason_context.get("explanation"))
    at_risk_agent = sanitize_text(reason_context.get("at_risk_agent")) or "the interacting road users"
    alert_time = (badas_context or {}).get("alert_time")
    prediction_window = (badas_context or {}).get("prediction_window_summary") or {}
    peak_start = prediction_window.get("peak_window_start_time")
    peak_end = prediction_window.get("peak_window_end_time")
    risk_context_parts = []
    if isinstance(alert_time, (int, float)):
        risk_context_parts.append(f"BADAS detected a high-risk interaction near {float(alert_time):.2f}s")
    if isinstance(peak_start, (int, float)) and isinstance(peak_end, (int, float)):
        risk_context_parts.append(f"the strongest risk window runs from {float(peak_start):.2f}s to {float(peak_end):.2f}s")
    risk_context_parts.append(f"Reason classified the event as {incident_type} with {severity_label} severity")
    if explanation:
        risk_context_parts.append(explanation)
    risk_context = "; ".join(risk_context_parts)
    base_prompt = [
        f"Observed scene context: {scene_summary}",
        f"Risk context: {risk_context}.",
        f"Focus on the road users already visible in the conditioning video, especially {at_risk_agent}.",
        f"This conditioning clip is centered on the critical interaction around {float(window['focus_time_sec']):.2f}s.",
    ]
    if mode == "prevented_continuation":
        preventive_action = infer_preventive_action(reason_context or {})
        base_prompt.extend([
            f"Counterfactual assumption: {preventive_action}.",
            "Task: Generate the next few seconds of physically plausible traffic evolution in which the preventive action continues to hold and the collision is reduced or avoided.",
        ])
    else:
        base_prompt.append("Task: Generate the next few seconds of physically plausible traffic evolution, preserving the likely immediate continuation of the observed event.")
    base_prompt.append("Preserve the same camera viewpoint, traffic layout, and agent identities. Avoid impossible physics, abrupt scene changes, visual glitches, or dramatic cinematic effects.")
    prompt = " ".join(base_prompt)
    words = prompt.split()
    if len(words) > 290:
        prompt = " ".join(words[:290])
    return prompt


def build_cache_key(source_video_path, badas_context, reason_context, mode, model_name, conditioning_source):
    payload = {
        "source_video_path": str(source_video_path),
        "mode": mode,
        "model_name": model_name,
        "conditioning_source": conditioning_source,
        "badas": {
            "alert_time": (badas_context or {}).get("alert_time"),
            "confidence": (badas_context or {}).get("confidence"),
            "valid_prediction_max": (badas_context or {}).get("valid_prediction_max"),
            "prediction_window_summary": (badas_context or {}).get("prediction_window_summary"),
            "top_predictions": ((badas_context or {}).get("top_predictions") or [])[:3],
        },
        "reason": {
            "incident_type": (reason_context or {}).get("incident_type"),
            "severity_label": (reason_context or {}).get("severity_label"),
            "critical_risk_time": (reason_context or {}).get("critical_risk_time"),
            "scene_summary": (reason_context or {}).get("scene_summary"),
            "explanation": (reason_context or {}).get("explanation"),
            "counterfactual_prompt": (reason_context or {}).get("counterfactual_prompt"),
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


@lru_cache(maxsize=2)
def get_predict_inference(model_name, output_root_str, disable_guardrails=True):
    vendored_path_str = str(VENDORED_PREDICT_PATH)
    if vendored_path_str not in sys.path:
        sys.path.insert(0, vendored_path_str)
    from cosmos_predict2.config import SetupArguments
    from cosmos_predict2.inference import Inference
    setup_args = SetupArguments(
        output_dir=Path(output_root_str),
        model=model_name,
        keep_going=True,
        disable_guardrails=disable_guardrails,
    )
    return Inference(setup_args)


def prepare_conditioning_input(source_video_path, badas_context, reason_context, output_root, fallback_conditioning_path=None):
    output_root = Path(output_root)
    conditioning_window = build_conditioning_window(badas_context, reason_context)
    context_cache_key = hashlib.sha256(
        json.dumps(
            {
                "source_video_path": str(source_video_path),
                "conditioning_window": conditioning_window,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    conditioning_clip_path = output_root / "conditioning" / f"conditioning_{context_cache_key}.mp4"
    try:
        conditioning_metadata = build_conditioning_clip(source_video_path, conditioning_window, conditioning_clip_path)
        return {
            "conditioning_source": "context_aware_segment",
            "conditioning_window": conditioning_window,
            "conditioning_metadata": conditioning_metadata,
            "fallback_applied": False,
            "fallback_reason": None,
        }
    except Exception as exc:
        fallback_path = existing_file(fallback_conditioning_path)
        if not fallback_path:
            raise
        return {
            "conditioning_source": "badas_focus_clip",
            "conditioning_window": conditioning_window,
            "conditioning_metadata": build_fallback_conditioning_metadata(fallback_path),
            "fallback_applied": True,
            "fallback_reason": str(exc),
        }


def execute_predict_generation(output_root, model_name, sample_name, conditioning_path, prompt):
    vendored_path_str = str(VENDORED_PREDICT_PATH)
    if vendored_path_str not in sys.path:
        sys.path.insert(0, vendored_path_str)
    from cosmos_predict2.config import InferenceArguments
    inference = get_predict_inference(model_name, str(output_root), True)
    inference_args = InferenceArguments(
        inference_type="video2world",
        name=sample_name,
        input_path=Path(conditioning_path),
        prompt=prompt,
        guidance=6,
        num_output_frames=77,
        num_steps=20,
    )
    output_paths = inference.generate([inference_args], output_root)
    return output_paths[0] if output_paths else None


def run_predict_scenario(source_video_path, badas_context=None, reason_context=None, mode="prevented_continuation", model_name="2B/post-trained", output_root=OUTPUT_ROOT, force_regenerate=False, fallback_conditioning_path=None):
    source_video_path = str(source_video_path)
    badas_context = badas_context or {}
    reason_context = reason_context or {}
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    conditioning_info = prepare_conditioning_input(
        source_video_path,
        badas_context,
        reason_context,
        output_root,
        fallback_conditioning_path=fallback_conditioning_path,
    )
    conditioning_source = conditioning_info["conditioning_source"]
    conditioning_window = conditioning_info["conditioning_window"]
    conditioning_metadata = conditioning_info["conditioning_metadata"]
    cache_key = build_cache_key(source_video_path, badas_context, reason_context, mode, model_name, conditioning_source)
    prompt = build_predict_prompt(badas_context, reason_context, mode, conditioning_window)
    sample_name = f"predict_{mode}_{cache_key}"
    output_video_path = output_root / f"{sample_name}.mp4"
    output_args_path = output_root / f"{sample_name}.json"
    if output_video_path.exists() and not force_regenerate:
        return {
            "success": True,
            "cached": True,
            "mode": mode,
            "model_name": model_name,
            "cache_key": cache_key,
            "source_video_path": source_video_path,
            "conditioning_source": conditioning_source,
            "conditioning_clip": existing_file(conditioning_metadata.get("clip_path")),
            "conditioning_metadata": conditioning_metadata,
            "conditioning_window": conditioning_window,
            "fallback_applied": conditioning_info.get("fallback_applied", False),
            "fallback_reason": conditioning_info.get("fallback_reason"),
            "prompt": prompt,
            "output_video": existing_file(output_video_path),
            "output_args_json": existing_file(output_args_path),
        }
    try:
        output_video = execute_predict_generation(
            output_root,
            model_name,
            sample_name,
            conditioning_metadata["clip_path"],
            prompt,
        )
    except Exception as exc:
        fallback_path = existing_file(fallback_conditioning_path)
        if conditioning_source == "badas_focus_clip" or not fallback_path or fallback_path == conditioning_metadata.get("clip_path"):
            raise
        fallback_conditioning_metadata = build_fallback_conditioning_metadata(fallback_path)
        fallback_conditioning_source = "badas_focus_clip"
        fallback_cache_key = build_cache_key(source_video_path, badas_context, reason_context, mode, model_name, fallback_conditioning_source)
        sample_name = f"predict_{mode}_{fallback_cache_key}"
        output_video_path = output_root / f"{sample_name}.mp4"
        output_args_path = output_root / f"{sample_name}.json"
        if output_video_path.exists() and not force_regenerate:
            return {
                "success": True,
                "cached": True,
                "mode": mode,
                "model_name": model_name,
                "cache_key": fallback_cache_key,
                "source_video_path": source_video_path,
                "conditioning_source": fallback_conditioning_source,
                "conditioning_clip": existing_file(fallback_conditioning_metadata.get("clip_path")),
                "conditioning_metadata": fallback_conditioning_metadata,
                "conditioning_window": conditioning_window,
                "fallback_applied": True,
                "fallback_reason": str(exc),
                "prompt": prompt,
                "output_video": existing_file(output_video_path),
                "output_args_json": existing_file(output_args_path),
            }
        output_video = execute_predict_generation(
            output_root,
            model_name,
            sample_name,
            fallback_conditioning_metadata["clip_path"],
            prompt,
        )
        conditioning_source = fallback_conditioning_source
        conditioning_metadata = fallback_conditioning_metadata
        cache_key = fallback_cache_key
        conditioning_info["fallback_applied"] = True
        conditioning_info["fallback_reason"] = str(exc)
    return {
        "success": bool(output_video),
        "cached": False,
        "mode": mode,
        "model_name": model_name,
        "cache_key": cache_key,
        "source_video_path": source_video_path,
        "conditioning_source": conditioning_source,
        "conditioning_clip": existing_file(conditioning_metadata.get("clip_path")),
        "conditioning_metadata": conditioning_metadata,
        "conditioning_window": conditioning_window,
        "fallback_applied": conditioning_info.get("fallback_applied", False),
        "fallback_reason": conditioning_info.get("fallback_reason"),
        "prompt": prompt,
        "output_video": existing_file(output_video),
        "output_args_json": existing_file(output_args_path),
    }


def run_predict_bundle(source_video_path, badas_context=None, reason_context=None, modes=None, model_name="2B/post-trained", output_root=OUTPUT_ROOT, force_regenerate=False, fallback_conditioning_path=None):
    modes = modes or ["prevented_continuation", "observed_continuation"]
    results = {}
    artifacts = {}
    for mode in modes:
        result = run_predict_scenario(
            source_video_path,
            badas_context=badas_context,
            reason_context=reason_context,
            mode=mode,
            model_name=model_name,
            output_root=output_root,
            force_regenerate=force_regenerate,
            fallback_conditioning_path=fallback_conditioning_path,
        )
        results[mode] = result
        if result.get("conditioning_clip") and not artifacts.get("predict_conditioning_clip"):
            artifacts["predict_conditioning_clip"] = result.get("conditioning_clip")
        if result.get("output_video"):
            artifacts[f"predict_{mode}_video"] = result.get("output_video")
    first_result = next(iter(results.values()), {})
    return {
        "success": any(result.get("success") for result in results.values()),
        "source_video_path": str(source_video_path),
        "model_name": model_name,
        "modes": list(modes),
        "results": results,
        "artifacts": artifacts,
        "fallback_applied": any(result.get("fallback_applied") for result in results.values()),
        "fallback_reasons": {mode: result.get("fallback_reason") for mode, result in results.items() if result.get("fallback_reason")},
        "conditioning_source": first_result.get("conditioning_source"),
    }


def run_predict_from_environment(source_video_path, mode=None):
    badas_context_raw = os.environ.get("COSMOS_PREDICT_BADAS_CONTEXT", "")
    reason_context_raw = os.environ.get("COSMOS_PREDICT_REASON_CONTEXT", "")
    try:
        badas_context = json.loads(badas_context_raw) if badas_context_raw else {}
    except json.JSONDecodeError:
        badas_context = {}
    try:
        reason_context = json.loads(reason_context_raw) if reason_context_raw else {}
    except json.JSONDecodeError:
        reason_context = {}
    mode = mode or os.environ.get("COSMOS_PREDICT_MODE", "prevented_continuation")
    return run_predict_scenario(source_video_path, badas_context=badas_context, reason_context=reason_context, mode=mode)


if __name__ == "__main__":
    source_video_path = sys.argv[1] if len(sys.argv) > 1 else "./extracted_clip.mp4"
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        payload = run_predict_from_environment(source_video_path, mode=mode)
    except Exception as exc:
        payload = {
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
    print(f"PREDICT_JSON: {json.dumps(payload)}")
