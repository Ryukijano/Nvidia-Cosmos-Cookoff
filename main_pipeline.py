#!/usr/bin/env python3
"""
Pure Cosmos Pipeline: Sequential execution of BADAS detection -> clip extraction -> Reason 2 analysis
"""

import os
import sys
import subprocess
import json
import re
import ast
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

# Optimization settings for fast execution
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def extract_structured_json(output, prefix):
    for line in output.splitlines():
        if line.startswith(prefix):
            try:
                return json.loads(line.split(prefix, 1)[1].strip())
            except json.JSONDecodeError:
                return None
    return None

def existing_file(path):
    return path if path and os.path.exists(path) else None


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
    h, w = frame.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    rendered = frame.copy()
    for label, bbox in (bboxes or {}).items():
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        px1 = max(0, min(w - 1, int(round(x1 * w))))
        py1 = max(0, min(h - 1, int(round(y1 * h))))
        px2 = max(px1 + 1, min(w, int(round(x2 * w))))
        py2 = max(py1 + 1, min(h, int(round(y2 * h))))
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
        h, w = frame.shape[:2]
        scale = resize_height / max(h, 1)
        resized.append(cv2.resize(frame, (max(1, int(round(w * scale))), resize_height)))
    strip = cv2.hconcat(resized)
    cv2.imwrite(output_path, strip)
    return output_path if os.path.exists(output_path) else None


def create_badas_frame_strip(video_path, badas_result):
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
    return save_frame_strip(frames, "badas_frame_strip.png")


def create_reason_frame_strip(clip_path, reason_payload):
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
    return save_frame_strip(frames, "reason_frame_strip.png")


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

def run_pipeline(video_path="./nexar_data/sample_videos/traffic_0.mp4"):
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    print("🚀 Starting Pure Cosmos Pipeline")
    print(f"Input video: {video_path}")

    pipeline_summary = {
        "input_video": video_path,
        "pipeline_mode": "badas_reason_only",
        "iterations": [],
        "artifacts": {},
        "status": "running",
        "overview": {},
    }

    current_video = video_path
    iteration_summary = {
        "iteration": 1,
        "input_video": current_video,
        "steps": {},
    }

    # Step 1: BADAS Collision Detection
    print("\n📍 Step 1: BADAS V-JEPA2 Collision Detection")
    badas_result = None
    try:
        result = subprocess.run([sys.executable, "badas_detector.py", current_video], capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(result.stdout.strip())
            alert_time = 5.0
            reason_focus_time = 5.0
            badas_result = extract_structured_json(result.stdout, "BADAS_JSON:")
            if badas_result is not None:
                alert_time = float(badas_result.get("alert_time", alert_time))
                reason_focus_time = select_reason_focus_time(badas_result)
                if badas_result.get("collision_detected"):
                    confidence = float(badas_result.get("confidence", 0.0))
                    sampled_frame = badas_result.get("alert_frame_sampled")
                    original_frame = badas_result.get("alert_frame_original_approx")
                    print(
                        f"✅ BADAS detected collision at {alert_time:.2f}s "
                        f"(sampled frame {sampled_frame}, original frame ≈ {original_frame}, confidence {confidence:.2%})"
                    )
                else:
                    print(
                        f"⚠️ BADAS completed without threshold crossing; "
                        f"using fallback alert time {alert_time:.2f}s from first valid predictive frame"
                    )
                print(f"🎯 Reason clip will be centered on BADAS peak evidence near {reason_focus_time:.2f}s")
            else:
                print("⚠️ BADAS completed but no structured payload was found, using default alert time")
                reason_focus_time = alert_time
            iteration_summary["steps"]["badas"] = {
                "success": True,
                "alert_time": alert_time,
                "reason_focus_time": reason_focus_time,
                "result": badas_result,
                "stdout": result.stdout,
            }
        else:
            print(f"❌ BADAS failed: {result.stderr}")
            alert_time = 5.0
            reason_focus_time = alert_time
            iteration_summary["steps"]["badas"] = {
                "success": False,
                "alert_time": alert_time,
                "reason_focus_time": reason_focus_time,
                "stderr": result.stderr,
            }
    except Exception as e:
        print(f"❌ BADAS error: {e}, using default alert time")
        alert_time = 5.0
        reason_focus_time = alert_time
        iteration_summary["steps"]["badas"] = {
            "success": False,
            "alert_time": alert_time,
            "reason_focus_time": reason_focus_time,
            "error": str(e),
        }

    # Step 2: Extract Pre-Alert Clip
    print("\n📍 Step 2: Extracting Pre-Alert Clip")
    extracted_clip = "./extracted_clip.mp4"
    try:
        from extract_clip import extract_pre_alert_clip
        extract_pre_alert_clip(current_video, reason_focus_time, extracted_clip)
        print("✅ Clip extracted successfully")
        iteration_summary["steps"]["clip_extraction"] = {
            "success": True,
            "clip_path": existing_file(extracted_clip),
            "alert_time": alert_time,
            "reason_focus_time": reason_focus_time,
        }
    except Exception as e:
        print(f"❌ Clip extraction failed: {e}")
        iteration_summary["steps"]["clip_extraction"] = {
            "success": False,
            "clip_path": None,
            "reason_focus_time": reason_focus_time,
            "error": str(e),
        }
        pipeline_summary["iterations"].append(iteration_summary)
        pipeline_summary["status"] = "failed"
        print(f"PIPELINE_JSON: {json.dumps(pipeline_summary)}")
        return pipeline_summary

    # Step 3: Reason 2 Analysis
    print("\n📍 Step 3: Cosmos Reason 2 Risk Analysis")
    reason_payload = None
    reason_text = ""
    try:
        reason_env = os.environ.copy()
        reason_env["COSMOS_BADAS_CONTEXT"] = json.dumps(badas_result or {})
        result = subprocess.run([sys.executable, "cosmos_risk_narrator.py", current_video, extracted_clip], capture_output=True, text=True, timeout=300, env=reason_env)
        if result.returncode == 0:
            output = result.stdout
            print(output.strip())
            print("✅ Reason 2 analysis completed")
            reason_payload = extract_structured_json(output, "REASON_JSON:")
            if reason_payload is None:
                bboxes, risk_score, _ = parse_reason_output(output)
                reason_text = output
                reason_payload = {
                    "text": output,
                    "risk_score": risk_score,
                    "counterfactual_prompt": "",
                    "bboxes": bboxes,
                    "bbox_count": len(bboxes),
                    "critical_risk_time": None,
                    "at_risk_agent": "",
                    "explanation": "",
                    "time_to_impact": None,
                    "parsing_summary": {
                        "parsed_field_flags": {},
                        "missing_fields": [],
                        "parsed_field_count": 0,
                        "total_expected_fields": 0,
                    },
                }
            else:
                reason_text = reason_payload.get("text", "")
                bboxes = reason_payload.get("bboxes", {})
                risk_score = reason_payload.get("risk_score") or 0
            visualizations = create_visualizations(current_video, extracted_clip, badas_result, reason_payload, bboxes, risk_score)
            print("📊 Visualizations saved: bboxes_visualization.png, risk_visualization.png")
            iteration_summary["steps"]["reason"] = {
                "success": True,
                "full_video_input": current_video,
                "focus_clip_input": existing_file(extracted_clip),
                "result": reason_payload,
                "text": reason_text,
                "stdout": output,
                "visualizations": visualizations,
            }
        else:
            print(f"❌ Reason 2 failed: {result.stderr}")
            iteration_summary["steps"]["reason"] = {
                "success": False,
                "full_video_input": current_video,
                "focus_clip_input": existing_file(extracted_clip),
                "stderr": result.stderr,
                "result": None,
                "text": "",
                "visualizations": {},
            }
    except Exception as e:
        print(f"❌ Reason 2 error: {e}")
        iteration_summary["steps"]["reason"] = {
            "success": False,
            "full_video_input": current_video,
            "focus_clip_input": existing_file(extracted_clip),
            "error": str(e),
            "result": None,
            "text": "",
            "visualizations": {},
        }

    pipeline_summary["iterations"].append(iteration_summary)

    print("\n� Pure Cosmos Pipeline completed!")
    print("� Summary: BADAS gating and Cosmos Reason analysis completed")
    last_iteration = pipeline_summary["iterations"][-1] if pipeline_summary["iterations"] else {}
    last_reason = last_iteration.get("steps", {}).get("reason", {})
    last_visualizations = last_reason.get("visualizations", {})
    pipeline_summary["overview"] = build_pipeline_overview(badas_result, reason_payload)
    pipeline_summary["artifacts"] = {
        "extracted_clip": existing_file("./extracted_clip.mp4"),
        "badas_gradient_saliency": existing_file((badas_result or {}).get("gradient_saliency_image")),
        "bbox_image": existing_file(last_visualizations.get("bbox_image")),
        "risk_image": existing_file(last_visualizations.get("risk_image")),
        "overlay_gif": existing_file(last_visualizations.get("overlay_gif")),
        "badas_frame_strip": existing_file(last_visualizations.get("badas_frame_strip")),
        "reason_frame_strip": existing_file(last_visualizations.get("reason_frame_strip")),
    }
    pipeline_summary["status"] = "completed"
    print(f"PIPELINE_JSON: {json.dumps(pipeline_summary)}")
    return pipeline_summary

def parse_reason_output(output):
    bboxes = {}
    risk_score = 0
    prompt = ""
    lines = output.split('\n')
    bbox_section = False
    for line in lines:
        if 'Bounding boxes' in line:
            bbox_section = True
            continue
        if bbox_section:
            if line.strip() and ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    label = parts[0].strip()
                    try:
                        coords = ast.literal_eval(parts[1].strip())
                        bboxes[label] = coords
                    except:
                        pass
            elif line.strip() == '':
                bbox_section = False
        if 'Risk score:' in line:
            try:
                risk_score = int(line.split('/')[0].split(':')[1].strip())
            except:
                risk_score = 0
        if 'Counterfactual collision prompt:' in line:
            prompt = line.split(':', 1)[1].strip()
    return bboxes, risk_score, prompt

def create_visualizations(source_video_path, clip_path, badas_result, reason_payload, bboxes, risk_score):
    bbox_image = 'bboxes_visualization.png'
    risk_image = 'risk_visualization.png'
    overlay_gif = 'video_with_bboxes.gif'
    badas_strip_image = 'badas_frame_strip.png'
    reason_strip_image = 'reason_frame_strip.png'
    if bboxes:
        fig, ax = plt.subplots(figsize=(6,6))
        for label, bbox in bboxes.items():
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1, label, fontsize=12, color='red')
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_title('Detected Agents Bounding Boxes')
        ax.invert_yaxis()
        plt.savefig(bbox_image)
        plt.close()

    fig, ax = plt.subplots(figsize=(4,2))
    ax.barh(['Risk Score'], [risk_score], color='orange')
    ax.set_xlim(0,5)
    ax.set_title('Collision Risk Assessment')
    plt.savefig(risk_image)
    plt.close()

    if bboxes:
        try:
            import cv2
            import imageio
            
            cap = cv2.VideoCapture(clip_path)
            frames = []
            frame_count = 0
            max_frames = 20
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                for label, bbox in bboxes.items():
                    x1, y1, x2, y2 = bbox
                    h, w = frame.shape[:2]
                    cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0,255,0), 2)
                    cv2.putText(frame, label, (int(x1*w), int(y1*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_count += 1
            
            cap.release()
            
            if frames:
                imageio.mimsave(overlay_gif, frames, fps=5, loop=0)
                print("📹 Annotated video GIF created: video_with_bboxes.gif")
        except ImportError:
            print("⚠️ OpenCV or imageio not available for video overlay")
        except Exception as e:
            print(f"⚠️ Failed to create video overlay: {e}")
    badas_strip = create_badas_frame_strip(source_video_path, badas_result)
    reason_strip = create_reason_frame_strip(clip_path, reason_payload)
    return {
        "bbox_image": bbox_image if bboxes and os.path.exists(bbox_image) else None,
        "risk_image": risk_image if os.path.exists(risk_image) else None,
        "overlay_gif": overlay_gif if bboxes and os.path.exists(overlay_gif) else None,
        "badas_frame_strip": badas_strip if badas_strip else None,
        "reason_frame_strip": reason_strip if reason_strip else None,
    }

if __name__ == "__main__":
    run_pipeline()
