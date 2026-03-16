import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import cv2
from PIL import Image
import ast
import json
import os
import re
import sys
from functools import lru_cache
import numpy as np

DEFAULT_REASON_MODEL_NAME = os.environ.get("COSMOS_REASON_MODEL", "nvidia/Cosmos-Reason2-8B")

# Check if INT8 quantization is enabled (for 24GB VRAM optimization)
USE_INT8 = os.environ.get("COSMOS_USE_INT8", "0") == "1"

# HF Token handling
HF_TOKEN = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "").strip() or None
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN


@lru_cache(maxsize=2)
def get_reason_model_bundle(model_name=DEFAULT_REASON_MODEL_NAME):
    print(f"Loading base Cosmos Reason model: {model_name}")
    
    # Configure quantization for 24GB VRAM (INT8 reduces from 20GB to ~10GB)
    quantization_config = None
    if USE_INT8:
        print("   Using INT8 quantization for memory efficiency")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if not USE_INT8 else torch.float16,
        quantization_config=quantization_config,
        token=HF_TOKEN,
    )
    print("Using pretrained checkpoint only; adapter loading is disabled")
    processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN)
    quant_str = "INT8 quantized" if USE_INT8 else "FP16"
    print(f"[OK] Fine-tuned model loaded ({quant_str}) on {model.device}")
    return model_name, model, processor

# System and user prompts for CCTV traffic reasoning
SYSTEM_PROMPT = "You are Cosmos Risk Narrator, an AI traffic-safety analyst reading fixed CCTV or roadside camera footage. Your task is to detect possible accidents, distinguish near-misses from confirmed impacts, and estimate crash severity in clear operational language."

USER_PROMPT_TEMPLATE = """Analyze the CCTV traffic video for collision and severity assessment. Focus on:
- Traffic environment analysis including road geometry, density, visibility, and possible occlusions from the CCTV view
- Distinguishing no incident, near-miss, minor collision, serious collision, or multi-agent severe crash
- Severity scoring (0-5 scale) where 0=no incident, 1=low concern, 2=near miss, 3=minor collision risk/contact, 4=serious collision, 5=catastrophic or multi-vehicle severe crash
- Short causal explanation describing what makes the event severe or not severe
- Treat visible physical vehicle-to-vehicle contact, deformation, abrupt rebound, spin, or forced redirection after contact as evidence of a real collision rather than a near miss
- Use near_miss only if the agents clearly avoid contact in the visible frames
- If two cars visibly crash into each other in the clip, do not output no_incident
- Focus especially on frames near the BADAS alert time and the seconds immediately after it

Use the BADAS detector summary below as a high-priority cue for where risk may be developing, but verify it against the actual video frames before concluding that a crash happened.

Video inputs:
- Video 1 is the full CCTV video and should be treated as the primary source of truth for temporal context
- Video 2, if present, is a BADAS-focused evidence clip centered on the highest-risk moment and should be used to inspect the likely impact sequence more closely

Important reliability rule: precise visual grounding bounding boxes are not reliable from this video prompt alone. Do not invent placeholder boxes. If you cannot localize agents reliably from the sampled video frames, return an empty bbox set.

BADAS detector context:
{badas_context}

Output format:
Scene summary: [1-2 sentence summary of what the CCTV camera sees]
Incident type: [no_incident | near_miss | collision | multi_vehicle_collision | unclear]
Severity label: [none | low | moderate | high | critical]
Bounding boxes: [brief text summary only, or none]

Critical risk detected at: X.X seconds
Risk score: X/5
At-risk agent: [description]
Explanation: [clear sentence about why this is dangerous]
Time-to-impact if no action: X.X seconds

Bounding boxes (normalized 0-1):
- Return no lines here unless localization is genuinely reliable

Counterfactual collision prompt: [1-sentence prompt describing the collision that would occur if the driver didn't brake]"""

USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE + "\n\nReturn exactly one report block in the format above. Do not repeat the report."

SECOND_PASS_PROMPT_TEMPLATE = """Re-evaluate this CCTV clip only for collision confirmation and severity.

Rules:
- Decide whether visible contact or immediate post-contact forced redirection occurs.
- If contact, rebound, spin, impact overlap, or abrupt trajectory change after contact is visible, output collision or multi_vehicle_collision.
- If the clip is ambiguous, output unclear.
- Do not output no_incident if BADAS flagged a strong collision cue and the clip shows a plausible impact sequence.
- Do not output any bounding boxes. Leave the normalized bounding box section empty.

Video inputs:
- Video 1 is the full CCTV video
- Video 2, if present, is the BADAS-focused evidence clip around the likely impact window

BADAS detector context:
{badas_context}

Output format:
Scene summary: [1-2 sentence summary of what the CCTV camera sees]
Incident type: [no_incident | near_miss | collision | multi_vehicle_collision | unclear]
Severity label: [none | low | moderate | high | critical]
Bounding boxes: none

Critical risk detected at: X.X seconds
Risk score: X/5
At-risk agent: [description]
Explanation: [clear sentence about why this is dangerous]
Time-to-impact if no action: X.X seconds or N/A

Bounding boxes (normalized 0-1):

Counterfactual collision prompt: [1 sentence]"""

SECOND_PASS_PROMPT_TEMPLATE = SECOND_PASS_PROMPT_TEMPLATE + "\n\nReturn exactly one report block in the format above. Do not repeat the report."

GENERATION_CONFIG = {
    "max_new_tokens": 1024,
    "do_sample": False,
}

VIDEO_BBOX_SUPPORT_NOTE = "Qwen3-VL video prompting does not reliably support native per-video bbox grounding in this pipeline, so bbox output is treated as untrusted unless sourced from a dedicated image-grounding path."


def extract_sections(text):
    sections = {
        "scene_summary_line": "",
        "incident_type_line": "",
        "severity_label_line": "",
        "bounding_boxes_header": "",
        "critical_risk_line": "",
        "risk_score_line": "",
        "at_risk_agent_line": "",
        "explanation_line": "",
        "time_to_impact_line": "",
        "counterfactual_line": "",
    }
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Scene summary:"):
            sections["scene_summary_line"] = stripped
        elif stripped.startswith("Incident type:"):
            sections["incident_type_line"] = stripped
        elif stripped.startswith("Severity label:"):
            sections["severity_label_line"] = stripped
        elif stripped.startswith("Bounding boxes:"):
            sections["bounding_boxes_header"] = stripped
        elif stripped.startswith("Critical risk detected at:"):
            sections["critical_risk_line"] = stripped
        elif stripped.startswith("Risk score:"):
            sections["risk_score_line"] = stripped
        elif stripped.startswith("At-risk agent:"):
            sections["at_risk_agent_line"] = stripped
        elif stripped.startswith("Explanation:"):
            sections["explanation_line"] = stripped
        elif stripped.startswith("Time-to-impact if no action:"):
            sections["time_to_impact_line"] = stripped
        elif stripped.startswith("Counterfactual collision prompt:"):
            sections["counterfactual_line"] = stripped
    return sections

def extract_reason_payload(text):
    payload = {
        "text": text,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": "",
        "badas_context": {},
        "scene_summary": "",
        "incident_type": "",
        "severity_label": "",
        "risk_score": None,
        "critical_risk_time": None,
        "at_risk_agent": "",
        "explanation": "",
        "time_to_impact": None,
        "counterfactual_prompt": "",
        "bboxes": {},
    }
    sections = extract_sections(text)
    bbox_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if bbox_section:
                bbox_section = False
            continue
        if stripped == "Bounding boxes (normalized 0-1):":
            bbox_section = True
            continue
        if bbox_section and ":" in stripped:
            label, coords_text = stripped.split(":", 1)
            try:
                coords = ast.literal_eval(coords_text.strip())
                if isinstance(coords, (list, tuple)) and len(coords) == 4:
                    payload["bboxes"][label.strip()] = [float(value) for value in coords]
            except (ValueError, SyntaxError):
                pass
        if stripped.startswith("Scene summary:"):
            payload["scene_summary"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Incident type:"):
            payload["incident_type"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Severity label:"):
            payload["severity_label"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Risk score:"):
            match = re.search(r"(\d+)", stripped)
            if match:
                payload["risk_score"] = int(match.group(1))
        elif stripped.startswith("Critical risk detected at:"):
            match = re.search(r"(\d+(?:\.\d+)?)", stripped)
            if match:
                payload["critical_risk_time"] = float(match.group(1))
        elif stripped.startswith("At-risk agent:"):
            payload["at_risk_agent"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Explanation:"):
            payload["explanation"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Time-to-impact if no action:"):
            match = re.search(r"(\d+(?:\.\d+)?)", stripped)
            if match:
                payload["time_to_impact"] = float(match.group(1))
        elif stripped.startswith("Counterfactual collision prompt:"):
            payload["counterfactual_prompt"] = stripped.split(":", 1)[1].strip()
    parsed_fields = {
        "scene_summary": bool(payload["scene_summary"]),
        "incident_type": bool(payload["incident_type"]),
        "severity_label": bool(payload["severity_label"]),
        "risk_score": payload["risk_score"] is not None,
        "critical_risk_time": payload["critical_risk_time"] is not None,
        "at_risk_agent": bool(payload["at_risk_agent"]),
        "explanation": bool(payload["explanation"]),
        "time_to_impact": payload["time_to_impact"] is not None,
        "counterfactual_prompt": bool(payload["counterfactual_prompt"]),
        "bboxes": bool(payload["bboxes"]),
    }
    payload["bbox_count"] = len(payload["bboxes"])
    payload["bbox_labels"] = list(payload["bboxes"].keys())
    payload["sections"] = sections
    payload["parsing_summary"] = {
        "parsed_field_flags": parsed_fields,
        "missing_fields": [field for field, parsed in parsed_fields.items() if not parsed],
        "parsed_field_count": int(sum(1 for parsed in parsed_fields.values() if parsed)),
        "total_expected_fields": len(parsed_fields),
    }
    return payload


def payload_requires_second_pass(payload, badas_context):
    if not payload:
        return True
    incident_type = (payload.get("incident_type") or "").strip().lower()
    risk_score = payload.get("risk_score")
    badas_confidence = float((badas_context or {}).get("confidence") or 0.0)
    peak_probability = float((badas_context or {}).get("valid_prediction_max") or 0.0)
    collision_detected = bool((badas_context or {}).get("collision_detected"))
    explanation = (payload.get("explanation") or "").strip().lower()
    missing_core = not payload.get("scene_summary") or not payload.get("incident_type") or not payload.get("severity_label")
    undercalls_badas = incident_type == "no_incident" and (collision_detected or badas_confidence >= 0.45 or peak_probability >= 0.6)
    noncommittal_zero_risk = incident_type in {"", "no_incident"} and (risk_score in {None, 0}) and "no immediate risk" in explanation
    return bool(missing_core or undercalls_badas or noncommittal_zero_risk)


def apply_badas_consistency_fallback(payload, badas_context):
    payload = dict(payload or {})
    incident_type = (payload.get("incident_type") or "").strip().lower()
    confidence = float((badas_context or {}).get("confidence") or 0.0)
    peak_probability = float((badas_context or {}).get("valid_prediction_max") or 0.0)
    peak_window_average = float(((badas_context or {}).get("prediction_window_summary") or {}).get("max_average_probability") or 0.0)
    threshold_crossings = int((((badas_context or {}).get("threshold_summary") or {}).get("threshold_crossing_count") or 0))
    strong_badas_evidence = bool((badas_context or {}).get("collision_detected")) and (
        confidence >= 0.5 or peak_probability >= 0.7 or peak_window_average >= 0.6 or threshold_crossings >= 2
    )
    if incident_type != "no_incident" or not strong_badas_evidence:
        payload.setdefault("fallback_override", None)
        return payload
    focus_time = float((badas_context or {}).get("alert_time") or 0.0)
    payload["incident_type"] = "collision"
    payload["severity_label"] = payload.get("severity_label") if payload.get("severity_label") in {"high", "critical"} else "moderate"
    payload["risk_score"] = max(int(payload.get("risk_score") or 0), 3)
    payload["at_risk_agent"] = payload.get("at_risk_agent") or "vehicles in the BADAS high-risk collision window"
    payload["critical_risk_time"] = payload.get("critical_risk_time") if payload.get("critical_risk_time") is not None else focus_time
    explanation = (payload.get("explanation") or "").strip()
    if not explanation or "no immediate risk" in explanation.lower():
        payload["explanation"] = (
            f"BADAS detected a high-confidence collision sequence near {focus_time:.2f}s, so the inconsistent no_incident response was escalated to collision for safety-focused review."
        )
    payload["fallback_override"] = {
        "applied": True,
        "source": "badas_consistency_guard",
        "reason": "Reason output conflicted with strong BADAS collision evidence",
    }
    return payload


def attach_validation(payload, badas_context, second_pass_used=False, initial_payload=None):
    payload = apply_badas_consistency_fallback(payload, badas_context)
    payload = dict(payload or {})
    validation_flags = {
        "video_bbox_grounding_supported": False,
        "reason_bboxes_rejected": bool(payload.get("bboxes")),
        "second_pass_used": bool(second_pass_used),
        "initial_payload_replaced": bool(second_pass_used),
        "incident_conflicts_with_badas": False,
        "fallback_override_applied": bool((payload.get("fallback_override") or {}).get("applied")),
    }
    incident_type = (payload.get("incident_type") or "").strip().lower()
    badas_confidence = float((badas_context or {}).get("confidence") or 0.0)
    peak_probability = float((badas_context or {}).get("valid_prediction_max") or 0.0)
    if incident_type == "no_incident" and (bool((badas_context or {}).get("collision_detected")) or badas_confidence >= 0.45 or peak_probability >= 0.6):
        validation_flags["incident_conflicts_with_badas"] = True
    payload["raw_bboxes"] = payload.get("bboxes") or {}
    payload["bboxes"] = {}
    payload["bbox_count"] = 0
    payload["bbox_labels"] = []
    payload["validation"] = {
        "is_reliable": not validation_flags["incident_conflicts_with_badas"],
        "flags": validation_flags,
        "note": VIDEO_BBOX_SUPPORT_NOTE,
    }
    return payload


def summarize_badas_context(badas_context):
    if not badas_context:
        return "No BADAS detector context was provided. Infer risk only from the visible clip."
    threshold_summary = badas_context.get("threshold_summary") or {}
    prediction_window_summary = badas_context.get("prediction_window_summary") or {}
    top_predictions = badas_context.get("top_predictions") or []
    top_prediction_lines = []
    for item in top_predictions[:3]:
        top_prediction_lines.append(
            f"- sampled_frame={item.get('sampled_frame')} time={item.get('time_sec', 0.0):.2f}s probability={item.get('probability', 0.0):.2%}"
        )
    if not top_prediction_lines:
        top_prediction_lines.append("- no top prediction frames available")
    return "\n".join([
        f"collision_detected={bool(badas_context.get('collision_detected'))}",
        f"alert_time={float(badas_context.get('alert_time', 0.0)):.2f}s",
        f"alert_confidence={float(badas_context.get('confidence', 0.0)):.2%}",
        f"threshold={float(badas_context.get('threshold', 0.0)):.2f}",
        f"threshold_crossing_count={int(threshold_summary.get('threshold_crossing_count', 0))}",
        f"peak_probability={float(badas_context.get('valid_prediction_max', 0.0) or 0.0):.2%}",
        f"peak_window_average_probability={float(prediction_window_summary.get('max_average_probability', 0.0) or 0.0):.2%}",
        "top_prediction_frames:",
        *top_prediction_lines,
    ])


def build_user_prompt(badas_context):
    return USER_PROMPT_TEMPLATE.format(badas_context=summarize_badas_context(badas_context))


def build_second_pass_prompt(badas_context):
    return SECOND_PASS_PROMPT_TEMPLATE.format(badas_context=summarize_badas_context(badas_context))

def process_video(video_path, max_frames=96):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 4.0
    frame_interval = max(1, int(round(fps / target_fps))) if fps else 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    count = 0
    sampled_indices_all = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            sampled_indices_all.append(int(count))
        count += 1
    cap.release()
    if len(frames) > max_frames:
        selected_positions = np.linspace(0, len(frames) - 1, max_frames, dtype=int).tolist()
        frames = [frames[position] for position in selected_positions]
        sampled_indices = [sampled_indices_all[position] for position in selected_positions]
    else:
        sampled_indices = sampled_indices_all
    
    print(f"Processed {len(frames)} frames")
    return frames, {
        "video_path": video_path,
        "original_fps": float(fps) if fps else 0.0,
        "target_fps": float(target_fps),
        "frame_interval": int(frame_interval),
        "total_frames": int(total_frames),
        "duration_sec": float(total_frames / fps) if fps else 0.0,
        "processed_frame_count": int(len(frames)),
        "sampled_frame_indices": sampled_indices,
        "sampled_timestamps_sec": [float(index / fps) for index in sampled_indices] if fps else [],
        "max_frames": int(max_frames),
    }


def generate_reason_response(video_inputs, user_prompt, model_name=None):
    resolved_model_name, model, processor = get_reason_model_bundle(model_name or DEFAULT_REASON_MODEL_NAME)
    content = []
    for video_input in video_inputs:
        # Use image-based approach for reliability (Qwen3-VL video has bugs)
        frames = video_input.get("frames", [])
        if frames:
            # Sample key frames as images
            sample_indices = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, len(frames)-1] if len(frames) > 5 else range(len(frames))
            for idx in sample_indices:
                if idx < len(frames):
                    content.append({
                        "type": "image",
                        "image": frames[idx],
                    })
    content.append({"type": "text", "text": user_prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content}
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    print(f"Generating risk assessment from {len(content)-1} frames...")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            **GENERATION_CONFIG,
        )
    output = processor.batch_decode(generated, skip_special_tokens=True)[0]
    assistant_part = output.split("assistant")[-1].strip()
    return assistant_part, inputs, generated, {
        "base_model_name": resolved_model_name,
        "device": str(model.device),
        "dtype": str(getattr(model, "dtype", torch.float16)),
    }

def run_risk_narrator(video_path, badas_context=None, focus_video_path=None, model_name=None):
    frames, frame_metadata = process_video(video_path)
    user_prompt = build_user_prompt(badas_context or {})
    video_inputs = [
        {
            "label": "full_video",
            "frames": frames,
            "fps": float(frame_metadata.get("target_fps") or 4.0),
            "metadata": frame_metadata,
        }
    ]
    focus_frame_metadata = None
    if focus_video_path:
        focus_frames, focus_frame_metadata = process_video(focus_video_path, max_frames=48)
        if focus_frames:
            video_inputs.append(
                {
                    "label": "focus_clip",
                    "frames": focus_frames,
                    "fps": float(focus_frame_metadata.get("target_fps") or 4.0),
                    "metadata": focus_frame_metadata,
                }
            )
    assistant_part, inputs, generated, model_metadata = generate_reason_response(video_inputs, user_prompt, model_name=model_name)
    initial_payload = extract_reason_payload(assistant_part)
    second_pass_used = False
    if payload_requires_second_pass(initial_payload, badas_context or {}):
        second_pass_used = True
        second_pass_prompt = build_second_pass_prompt(badas_context or {})
        assistant_part, inputs, generated, model_metadata = generate_reason_response(video_inputs, second_pass_prompt, model_name=model_name)
        user_prompt = second_pass_prompt
        final_payload = extract_reason_payload(assistant_part)
    else:
        final_payload = initial_payload
    final_payload = attach_validation(final_payload, badas_context or {}, second_pass_used=second_pass_used, initial_payload=initial_payload)
    return assistant_part, {
        "parsed_payload": final_payload,
        "frame_metadata": frame_metadata,
        "focus_frame_metadata": focus_frame_metadata,
        "video_input_count": len(video_inputs),
        "user_prompt": user_prompt,
        "badas_context": badas_context or {},
        "model": {
            "base_model_name": model_metadata["base_model_name"],
            "checkpoint_source": "pretrained_huggingface_checkpoint",
            "adapter_loaded": False,
            "device": model_metadata["device"],
            "dtype": model_metadata["dtype"],
        },
        "generation_config": GENERATION_CONFIG,
        "input_token_count": int(inputs["input_ids"].shape[-1]),
        "output_token_count": int(generated.shape[-1]),
    }

if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "./extracted_clip.mp4"
    focus_video_path = sys.argv[2] if len(sys.argv) > 2 else None
    badas_context_raw = os.environ.get("COSMOS_BADAS_CONTEXT", "")
    try:
        badas_context = json.loads(badas_context_raw) if badas_context_raw else {}
    except json.JSONDecodeError:
        badas_context = {}
    result, metadata = run_risk_narrator(video_path, badas_context=badas_context, focus_video_path=focus_video_path)
    payload = metadata.get("parsed_payload") or extract_reason_payload(result)
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
    print("Reason 2 Output:")
    print(result)
    if payload["counterfactual_prompt"]:
        counterfactual = payload["counterfactual_prompt"]
        print(f"Counterfactual Prompt: {counterfactual}")
    else:
        print("Counterfactual prompt not found")
    print(f"REASON_JSON: {json.dumps(payload)}")
