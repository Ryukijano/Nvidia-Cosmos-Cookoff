import torch
import cv2
import os
import sys
import json
import numpy as np

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if hf_token and "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = hf_token

# Add BADAS model path
sys.path.append("./nexar_data/badas_model")

# Import BADAS loader
from badas_loader import load_badas_model


def extract_window_frames(video_path, end_time_sec, target_fps, frame_count):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if original_fps <= 0:
        cap.release()
        return []
    duration_sec = max(0.0, float(frame_count - 1) / float(target_fps)) if target_fps else 0.0
    start_time_sec = max(0.0, float(end_time_sec) - duration_sec)
    timestamps = [start_time_sec + (idx / float(target_fps)) for idx in range(frame_count)] if target_fps else [float(end_time_sec)]
    frames = []
    for timestamp in timestamps:
        frame_index = max(0, int(round(timestamp * original_fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def prepare_video_tensor(model, frames_bgr):
    if not frames_bgr:
        return None
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    if model.processor:
        try:
            inputs = model.processor(frames_rgb, return_tensors="pt")
            if "pixel_values_videos" in inputs:
                return inputs["pixel_values_videos"].squeeze(0)
            if "pixel_values" in inputs:
                return inputs["pixel_values"].squeeze(0)
            return list(inputs.values())[0].squeeze(0)
        except Exception:
            pass
    if model.transform:
        transformed_frames = [model.transform(image=frame)["image"] for frame in frames_rgb]
        return torch.stack(transformed_frames)
    frames_tensor = torch.from_numpy(np.stack(frames_rgb).transpose(0, 3, 1, 2)).float() / 255.0
    return frames_tensor


def build_gradient_saliency_strip(model, video_path, focus_time_sec, sampled_fps):
    frames_bgr = extract_window_frames(video_path, focus_time_sec, sampled_fps, model.frame_count)
    if not frames_bgr:
        return None
    video_tensor = prepare_video_tensor(model, frames_bgr)
    if video_tensor is None:
        return None
    input_tensor = video_tensor.unsqueeze(0).to(model.device)
    input_tensor.requires_grad_(True)
    model.model.zero_grad(set_to_none=True)
    logits = model.model(input_tensor)
    if logits.ndim != 2 or logits.shape[-1] < 2:
        return None
    positive_logit = logits[0, 1]
    positive_logit.backward()
    gradients = input_tensor.grad.detach().abs().mean(dim=2).squeeze(0).cpu().numpy()
    overlay_frames = []
    selected_indices = np.linspace(0, len(frames_bgr) - 1, min(4, len(frames_bgr)), dtype=int)
    for frame_index in selected_indices:
        frame = frames_bgr[int(frame_index)].copy()
        grad_map = gradients[int(frame_index)]
        if grad_map.max() > 0:
            grad_map = grad_map / grad_map.max()
        heatmap = np.uint8(255 * grad_map)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        rendered = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        cv2.putText(
            rendered,
            f"BADAS gradient saliency | t={focus_time_sec:.2f}s",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        overlay_frames.append(rendered)
    if not overlay_frames:
        return None
    resized_frames = []
    resize_height = 220
    for frame in overlay_frames:
        height, width = frame.shape[:2]
        scale = resize_height / max(height, 1)
        resized_frames.append(cv2.resize(frame, (max(1, int(round(width * scale))), resize_height)))
    strip = cv2.hconcat(resized_frames)
    output_path = "badas_gradient_saliency.png"
    cv2.imwrite(output_path, strip)
    return output_path if os.path.exists(output_path) else None


def summarize_prediction_window(prediction_series, sampled_fps):
    if not prediction_series:
        return {
            'window_count': 0,
            'max_average_probability': None,
            'peak_window_start_time': None,
            'peak_window_end_time': None,
        }
    window_size = max(1, int(round(sampled_fps)))
    best_average = None
    best_start_idx = 0
    probabilities = [item['probability'] for item in prediction_series]
    for start_idx in range(0, max(1, len(probabilities) - window_size + 1)):
        window = probabilities[start_idx:start_idx + window_size]
        if not window:
            continue
        average = float(np.mean(window))
        if best_average is None or average > best_average:
            best_average = average
            best_start_idx = start_idx
    end_idx = min(len(prediction_series) - 1, best_start_idx + window_size - 1)
    return {
        'window_count': max(0, len(probabilities) - window_size + 1),
        'max_average_probability': best_average,
        'peak_window_start_time': float(prediction_series[best_start_idx]['time_sec']),
        'peak_window_end_time': float(prediction_series[end_idx]['time_sec']),
    }


def summarize_threshold_runs(collision_frames, sampled_fps):
    if not collision_frames:
        return {
            'threshold_crossing_count': 0,
            'threshold_crossing_times': [],
            'contiguous_alert_runs': [],
            'longest_alert_run_frames': 0,
            'longest_alert_run_sec': 0.0,
        }
    threshold_crossing_times = [float(frame_idx / sampled_fps) for frame_idx, _ in collision_frames]
    runs = []
    run_start = collision_frames[0][0]
    run_probs = [float(collision_frames[0][1])]
    previous_frame = collision_frames[0][0]
    for frame_idx, prob in collision_frames[1:]:
        if frame_idx == previous_frame + 1:
            run_probs.append(float(prob))
        else:
            runs.append((run_start, previous_frame, run_probs))
            run_start = frame_idx
            run_probs = [float(prob)]
        previous_frame = frame_idx
    runs.append((run_start, previous_frame, run_probs))
    contiguous_alert_runs = [
        {
            'start_frame': int(start_frame),
            'end_frame': int(end_frame),
            'start_time': float(start_frame / sampled_fps),
            'end_time': float(end_frame / sampled_fps),
            'duration_frames': int(end_frame - start_frame + 1),
            'duration_sec': float((end_frame - start_frame + 1) / sampled_fps),
            'max_probability': float(max(probabilities)),
            'mean_probability': float(np.mean(probabilities)),
        }
        for start_frame, end_frame, probabilities in runs
    ]
    longest_run = max(contiguous_alert_runs, key=lambda item: item['duration_frames'])
    return {
        'threshold_crossing_count': int(len(collision_frames)),
        'threshold_crossing_times': threshold_crossing_times,
        'contiguous_alert_runs': contiguous_alert_runs,
        'longest_alert_run_frames': int(longest_run['duration_frames']),
        'longest_alert_run_sec': float(longest_run['duration_sec']),
    }

def run_badas_detector(video_path, confidence_threshold=0.5):
    """Run BADAS-Open collision detection on video"""
    print("Loading BADAS-Open model...")
    model = load_badas_model()
    model_info = model.get_model_info()
    
    # Run prediction on entire video
    print(f"Analyzing video: {video_path}")
    predictions = np.asarray(model.predict(video_path), dtype=np.float32)
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    sampled_fps = float(model_info.get('target_fps') or original_fps)
    video_duration_sec = float(total_frames / original_fps) if original_fps else 0.0
    valid_mask = np.isfinite(predictions)
    valid_indices = np.flatnonzero(valid_mask)
    valid_predictions = predictions[valid_mask]
    nan_count = int((~valid_mask).sum())
    prediction_series = [
        {
            'sampled_frame': int(idx),
            'original_frame_approx': int(round((idx / sampled_fps) * original_fps)),
            'time_sec': float(idx / sampled_fps),
            'probability': float(predictions[idx]),
        }
        for idx in valid_indices
    ]
    top_predictions = sorted(
        prediction_series,
        key=lambda item: item['probability'],
        reverse=True,
    )[:5]
    saliency_focus_time = float(top_predictions[0]['time_sec']) if top_predictions else None
    gradient_saliency_image = None
    if saliency_focus_time is not None:
        try:
            gradient_saliency_image = build_gradient_saliency_strip(model, video_path, saliency_focus_time, sampled_fps)
            if gradient_saliency_image:
                print(f"Gradient saliency artifact saved: {gradient_saliency_image}")
        except Exception as exc:
            print(f"Warning: failed to build BADAS gradient saliency artifact: {exc}")
    prediction_window_summary = summarize_prediction_window(prediction_series, sampled_fps)
    valid_prediction_summary = {
        'min': float(valid_predictions.min()) if len(valid_predictions) > 0 else None,
        'max': float(valid_predictions.max()) if len(valid_predictions) > 0 else None,
        'mean': float(valid_predictions.mean()) if len(valid_predictions) > 0 else None,
        'median': float(np.median(valid_predictions)) if len(valid_predictions) > 0 else None,
        'std': float(valid_predictions.std()) if len(valid_predictions) > 0 else None,
        'p90': float(np.percentile(valid_predictions, 90)) if len(valid_predictions) > 0 else None,
        'p95': float(np.percentile(valid_predictions, 95)) if len(valid_predictions) > 0 else None,
    }
    
    print(f"Model info: {json.dumps(model_info)}")
    print(f"Video info: {total_frames} frames at {original_fps:.2f} FPS")
    print(f"Video duration: {video_duration_sec:.2f}s")
    print(f"Model sampling FPS: {sampled_fps:.2f}")
    print(f"Prediction array length: {len(predictions)}")
    print(f"Valid predictions: {len(valid_predictions)} | NaN warmup frames: {nan_count}")
    if len(valid_predictions) > 0:
        print(f"Prediction range: {valid_prediction_summary['min']:.3f} - {valid_prediction_summary['max']:.3f}")
        print(f"Mean prediction: {valid_prediction_summary['mean']:.3f}")
        print(f"Median prediction: {valid_prediction_summary['median']:.3f}")
        print(f"Prediction std: {valid_prediction_summary['std']:.3f}")
        print(f"P90/P95: {valid_prediction_summary['p90']:.3f} / {valid_prediction_summary['p95']:.3f}")
        first_valid_idx = int(valid_indices[0])
        print(f"First valid predictive frame: sampled frame {first_valid_idx} ({first_valid_idx / sampled_fps:.2f}s)")
        print("Top prediction frames:")
        for item in top_predictions:
            frame_idx = item['sampled_frame']
            prob = item['probability']
            time_sec = item['time_sec']
            original_frame = item['original_frame_approx']
            print(
                f"  sampled_frame={frame_idx} original_frame≈{original_frame} time={time_sec:.2f}s prob={prob:.2%}"
            )
        if prediction_window_summary['max_average_probability'] is not None:
            print(
                f"Peak 1-second window: {prediction_window_summary['peak_window_start_time']:.2f}s"
                f" - {prediction_window_summary['peak_window_end_time']:.2f}s"
                f" avg={prediction_window_summary['max_average_probability']:.2%}"
            )
    else:
        print("Prediction range: no valid predictions produced")
    
    # Find frames with high collision probability (lower threshold for better detection)
    collision_frames = []
    for frame_idx, prob in enumerate(predictions):
        if not np.isfinite(prob):
            continue
        if prob > confidence_threshold:
            collision_frames.append((frame_idx, prob))
            time_sec = frame_idx / sampled_fps
            original_frame = int(round(time_sec * original_fps))
            print(
                f"⚠️ Collision risk at sampled frame {frame_idx} "
                f"(original frame ≈ {original_frame}, {time_sec:.2f}s): {prob:.2%}"
            )
    threshold_summary = summarize_threshold_runs(collision_frames, sampled_fps)
    if threshold_summary['contiguous_alert_runs']:
        print("Threshold-crossing runs:")
        for run in threshold_summary['contiguous_alert_runs']:
            print(
                f"  {run['start_time']:.2f}s - {run['end_time']:.2f}s"
                f" duration={run['duration_sec']:.2f}s max={run['max_probability']:.2%}"
            )
    
    if collision_frames:
        # Return earliest high-risk frame
        earliest_frame, highest_prob = min(collision_frames, key=lambda x: x[0])  # Earliest frame
        alert_time = earliest_frame / sampled_fps
        alert_original_frame = int(round(alert_time * original_fps))
        print(
            f"🚨 BADAS Alert: Collision detected at {alert_time:.2f}s "
            f"(sampled frame {earliest_frame}, original frame ≈ {alert_original_frame}) "
            f"with {highest_prob:.2%} confidence"
        )
        return {
            'collision_detected': True,
            'alert_frame_sampled': int(earliest_frame),
            'alert_frame_original_approx': alert_original_frame,
            'alert_time': alert_time,
            'confidence': float(highest_prob),
            'threshold': float(confidence_threshold),
            'original_fps': float(original_fps),
            'sampled_fps': sampled_fps,
            'model_info': model_info,
            'video_metadata': {
                'video_path': video_path,
                'total_frames': int(total_frames),
                'original_fps': float(original_fps),
                'sampled_fps': float(sampled_fps),
                'duration_sec': video_duration_sec,
            },
            'prediction_count': int(len(predictions)),
            'valid_prediction_count': int(len(valid_predictions)),
            'nan_warmup_count': nan_count,
            'valid_prediction_min': valid_prediction_summary['min'],
            'valid_prediction_max': valid_prediction_summary['max'],
            'valid_prediction_mean': valid_prediction_summary['mean'],
            'valid_prediction_median': valid_prediction_summary['median'],
            'valid_prediction_std': valid_prediction_summary['std'],
            'valid_prediction_p90': valid_prediction_summary['p90'],
            'valid_prediction_p95': valid_prediction_summary['p95'],
            'first_valid_time': float(valid_indices[0] / sampled_fps) if len(valid_indices) > 0 else None,
            'alert_source': 'threshold_crossing',
            'prediction_window_summary': prediction_window_summary,
            'threshold_summary': threshold_summary,
            'top_predictions': top_predictions,
            'prediction_series': prediction_series,
            'gradient_saliency_image': gradient_saliency_image,
            'gradient_saliency_focus_time': saliency_focus_time,
        }
    else:
        print("⚠️ BADAS completed but no collision detected, using default alert time")
        # Return default alert for pipeline continuation
        default_frame = next(iter(valid_indices), len(predictions) // 4)
        default_time = float(default_frame / sampled_fps) if sampled_fps else 0.0
        default_original_frame = int(round(default_time * original_fps))
        return {
            'collision_detected': False,
            'alert_frame_sampled': int(default_frame),
            'alert_frame_original_approx': default_original_frame,
            'alert_time': default_time,
            'confidence': 0.0,
            'threshold': float(confidence_threshold),
            'original_fps': float(original_fps),
            'sampled_fps': sampled_fps,
            'model_info': model_info,
            'video_metadata': {
                'video_path': video_path,
                'total_frames': int(total_frames),
                'original_fps': float(original_fps),
                'sampled_fps': float(sampled_fps),
                'duration_sec': video_duration_sec,
            },
            'prediction_count': int(len(predictions)),
            'valid_prediction_count': int(len(valid_predictions)),
            'nan_warmup_count': nan_count,
            'valid_prediction_min': valid_prediction_summary['min'],
            'valid_prediction_max': valid_prediction_summary['max'],
            'valid_prediction_mean': valid_prediction_summary['mean'],
            'valid_prediction_median': valid_prediction_summary['median'],
            'valid_prediction_std': valid_prediction_summary['std'],
            'valid_prediction_p90': valid_prediction_summary['p90'],
            'valid_prediction_p95': valid_prediction_summary['p95'],
            'first_valid_time': float(valid_indices[0] / sampled_fps) if len(valid_indices) > 0 else None,
            'alert_source': 'fallback_first_valid_frame',
            'prediction_window_summary': prediction_window_summary,
            'threshold_summary': threshold_summary,
            'top_predictions': top_predictions,
            'prediction_series': prediction_series,
            'gradient_saliency_image': gradient_saliency_image,
            'gradient_saliency_focus_time': saliency_focus_time,
        }

if __name__ == "__main__":
    # Test on sample video or provided path
    video_path = sys.argv[1] if len(sys.argv) > 1 else "./nexar_data/sample_videos/sample_dashcam_2.mp4"
    result = run_badas_detector(video_path)
    
    if result['collision_detected']:
        print(f"BADAS Alert: Collision detected at {result['alert_time']:.2f}s with {result['confidence']:.2%} confidence")
        print("This would be our System 1 trigger for the Pure Cosmos Pipeline")
    else:
        print("No collision detected by BADAS")
    print(f"BADAS_JSON: {json.dumps(result)}")
