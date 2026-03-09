from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

STREAMLIT_SERVER_MAX_UPLOAD_MB = 4096
INLINE_VIDEO_PREVIEW_MAX_MB = 256
MIN_FREE_WORKSPACE_BUFFER_MB = int(os.getenv("SPACE_MIN_FREE_SPACE_BUFFER_MB", "2048"))
COPY_CHUNK_BYTES = 8 * 1024 * 1024
SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]
PHASE_OVERLAY_BGR = {
    "idle": (139, 125, 96),
    "marking": (255, 140, 79),
    "injection": (255, 198, 104),
    "dissection": (107, 107, 255),
    "unknown": (248, 250, 252),
}
HUD_BACKGROUND_BGR = (20, 10, 6)
HUD_TEXT_BGR = (252, 250, 248)
HUD_MUTED_BGR = (215, 200, 191)


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "Unknown"

    whole_seconds = int(round(seconds))
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def recommended_frame_stride(duration_seconds: float | None) -> int:
    if duration_seconds is None or duration_seconds <= 0:
        return 5
    if duration_seconds < 600:
        return 1
    if duration_seconds < 1800:
        return 2
    if duration_seconds < 3600:
        return 3
    return 5


def get_upload_size_bytes(uploaded_file) -> int:
    size = getattr(uploaded_file, "size", None)
    if size is not None:
        return int(size)

    current_position = uploaded_file.tell()
    uploaded_file.seek(0, os.SEEK_END)
    size = uploaded_file.tell()
    uploaded_file.seek(current_position)
    return int(size)


def should_show_inline_preview(file_size_bytes: int) -> bool:
    return file_size_bytes <= INLINE_VIDEO_PREVIEW_MAX_MB * 1024 * 1024


def get_workspace_free_bytes(temp_dir: Path | None = None) -> int:
    target_dir = Path(temp_dir or tempfile.gettempdir())
    target_dir.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(target_dir).free


def create_temp_video_path(
    *, suffix: str = ".mp4", prefix: str = "portfolio-overlay-", temp_dir: Path | None = None
) -> Path:
    target_dir = Path(temp_dir or tempfile.gettempdir())
    target_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=target_dir)
    os.close(fd)
    return Path(temp_name)


def create_overlay_video_writer(
    *, frame_size: tuple[int, int], fps: float, temp_dir: Path | None = None
) -> tuple[cv2.VideoWriter, Path]:
    effective_fps = fps if fps > 0 else 24.0
    candidates = (
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("MJPG", ".avi"),
        ("XVID", ".avi"),
    )
    tried = []
    for codec, suffix in candidates:
        temp_path = create_temp_video_path(
            prefix="portfolio-overlay-raw-",
            suffix=suffix,
            temp_dir=temp_dir,
        )
        writer = cv2.VideoWriter(
            str(temp_path),
            cv2.VideoWriter_fourcc(*codec),
            effective_fps,
            frame_size,
        )
        if writer.isOpened():
            return writer, temp_path
        writer.release()
        temp_path.unlink(missing_ok=True)
        tried.append(f"{codec}{suffix}")

    raise RuntimeError(
        "Unable to create annotated playback video for this environment. "
        f"Tried {', '.join(tried)}."
    )


def spool_uploaded_video(uploaded_file, suffix: str | None = None, temp_dir: Path | None = None) -> Path:
    file_size_bytes = get_upload_size_bytes(uploaded_file)
    if file_size_bytes > STREAMLIT_SERVER_MAX_UPLOAD_MB * 1024 * 1024:
        raise RuntimeError(
            f"This upload exceeds the {STREAMLIT_SERVER_MAX_UPLOAD_MB} MB Space limit."
        )

    target_dir = Path(temp_dir or tempfile.gettempdir())
    target_dir.mkdir(parents=True, exist_ok=True)

    free_bytes = get_workspace_free_bytes(target_dir)
    required_bytes = file_size_bytes + (MIN_FREE_WORKSPACE_BUFFER_MB * 1024 * 1024)
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Not enough working storage to stage this upload. "
            f"Need at least {format_bytes(required_bytes)} free, only {format_bytes(free_bytes)} available."
        )

    suffix = suffix or Path(getattr(uploaded_file, "name", "")).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=target_dir) as tmp:
        uploaded_file.seek(0)
        shutil.copyfileobj(uploaded_file, tmp, length=COPY_CHUNK_BYTES)
        temp_path = Path(tmp.name)

    uploaded_file.seek(0)
    return temp_path


def draw_prediction_overlay(
    frame: np.ndarray,
    *,
    phase: str,
    confidence: float,
    model_label: str,
    frame_index: int,
    fps: float,
    total_frames: int | None = None,
    sampled_frame: bool = True,
) -> np.ndarray:
    annotated = frame.copy()
    overlay = annotated.copy()
    height, width = annotated.shape[:2]
    margin = max(16, width // 64)
    available_width = max(120, width - (margin * 2))
    available_height = max(72, height - (margin * 2))
    panel_width = min(max(int(width * 0.44), 240), available_width)
    panel_height = min(max(int(height * 0.16), 88), min(140, available_height))
    accent = _phase_overlay_color_bgr(phase)

    panel_x1 = margin
    panel_y1 = max(margin, height - panel_height - margin)
    panel_x2 = panel_x1 + panel_width
    panel_y2 = panel_y1 + panel_height
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), HUD_BACKGROUND_BGR, -1)
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), accent, 2)
    cv2.rectangle(overlay, (panel_x1 + 12, panel_y1 + 12), (panel_x1 + 18, panel_y2 - 12), accent, -1)
    cv2.addWeighted(overlay, 0.68, annotated, 0.32, 0, annotated)

    base_scale = max(0.52, min(1.0, width / 1200.0))
    phase_label = phase.title() if phase and phase.lower() != "unknown" else "--"
    status_label = "Sampled frame" if sampled_frame else "Carry-forward overlay"
    _draw_shadowed_text(
        annotated,
        f"Phase: {phase_label}",
        (panel_x1 + 34, panel_y1 + 32),
        accent,
        font_scale=base_scale * 0.92,
        thickness=2,
    )
    _draw_shadowed_text(
        annotated,
        f"Confidence {confidence:.1%} • {status_label}",
        (panel_x1 + 34, panel_y1 + 58),
        HUD_TEXT_BGR,
        font_scale=base_scale * 0.62,
        thickness=1,
    )

    current_time = format_duration(frame_index / fps if fps > 0 else None)
    total_time = format_duration(total_frames / fps if fps > 0 and total_frames else None)
    header_text = model_label
    if current_time != "Unknown" and total_time != "Unknown":
        header_text = f"{header_text} • {current_time} / {total_time}"
    elif current_time != "Unknown":
        header_text = f"{header_text} • {current_time}"

    header_scale = max(0.46, min(0.62, width / 1600.0))
    text_width, text_height = cv2.getTextSize(
        header_text, cv2.FONT_HERSHEY_SIMPLEX, header_scale, 1
    )[0]
    header_width = min(text_width + 28, available_width)
    header_height = text_height + 18
    header_x2 = width - margin
    header_x1 = max(margin, header_x2 - header_width)
    header_y1 = margin
    header_y2 = header_y1 + header_height
    header_overlay = annotated.copy()
    cv2.rectangle(header_overlay, (header_x1, header_y1), (header_x2, header_y2), HUD_BACKGROUND_BGR, -1)
    cv2.rectangle(header_overlay, (header_x1, header_y1), (header_x2, header_y2), accent, 1)
    cv2.addWeighted(header_overlay, 0.62, annotated, 0.38, 0, annotated)
    _draw_shadowed_text(
        annotated,
        header_text,
        (header_x1 + 14, header_y1 + text_height + 5),
        HUD_MUTED_BGR,
        font_scale=header_scale,
        thickness=1,
    )

    if total_frames and total_frames > 0:
        progress_ratio = min(max((frame_index + 1) / total_frames, 0.0), 1.0)
        bar_x1 = margin
        bar_y1 = height - 8
        bar_x2 = width - margin
        bar_y2 = height - 4
        cv2.rectangle(annotated, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 67, 82), -1)
        cv2.rectangle(
            annotated,
            (bar_x1, bar_y1),
            (bar_x1 + int((bar_x2 - bar_x1) * progress_ratio), bar_y2),
            accent,
            -1,
        )

    return annotated


def transcode_video_for_streamlit(
    video_path: str | Path, *, temp_dir: Path | None = None
) -> tuple[Path, str | None]:
    input_path = Path(video_path)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return input_path, "ffmpeg is unavailable, so the raw overlay clip is being used for playback."

    output_path = create_temp_video_path(prefix="portfolio-overlay-final-", suffix=".mp4", temp_dir=temp_dir)
    command = [
        ffmpeg_path,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    except subprocess.TimeoutExpired:
        output_path.unlink(missing_ok=True)
        return input_path, "ffmpeg timed out while transcoding the overlay clip, so the raw overlay video is being used."
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        error_line = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown ffmpeg error"
        return (
            input_path,
            f"ffmpeg could not transcode the overlay clip for browser-friendly playback ({error_line}). "
            "Using the directly encoded clip instead.",
        )

    input_path.unlink(missing_ok=True)
    return output_path, None


def probe_video_info(video_path: str | Path) -> dict:
    path = Path(video_path)
    if not path.exists():
        raise RuntimeError(f"Video file not found: {path}")

    file_size_bytes = path.stat().st_size
    if file_size_bytes <= 0:
        raise RuntimeError(f"Video file is empty: {path}")

    info = {
        "path": str(path),
        "name": path.name,
        "file_size_bytes": file_size_bytes,
        "file_size_label": format_bytes(file_size_bytes),
    }

    capture = cv2.VideoCapture(str(path))
    open_cv_info = {}
    if capture.isOpened():
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        if width > 0 and height > 0:
            duration_seconds = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
            open_cv_info = {
                "duration_seconds": duration_seconds,
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
            }
    else:
        capture.release()

    ffprobe_info = {}
    if (
        not open_cv_info
        or open_cv_info.get("duration_seconds", 0.0) <= 0
        or open_cv_info.get("fps", 0.0) <= 0
        or open_cv_info.get("frame_count", 0) <= 0
    ):
        ffprobe_info = _probe_video_info_ffprobe(path)

    merged = {**info, **open_cv_info, **ffprobe_info}
    if merged.get("width", 0) <= 0 or merged.get("height", 0) <= 0:
        raise RuntimeError(f"Could not determine video dimensions for {path.name}")

    merged["duration_label"] = format_duration(merged.get("duration_seconds"))
    merged["resolution_label"] = f"{merged['width']}x{merged['height']}"
    return merged


def _probe_video_info_ffprobe(video_path: Path) -> dict:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        return {}

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration,format_name",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode != 0:
        error_message = result.stderr.strip() or "Unknown ffprobe error"
        raise RuntimeError(f"ffprobe failed for {video_path.name}: {error_message}")

    try:
        probe_data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unable to parse ffprobe output for {video_path.name}") from exc

    streams = probe_data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path.name}")

    stream = streams[0]
    format_info = probe_data.get("format", {})
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    duration_seconds = float(format_info.get("duration") or 0.0)
    fps = _parse_fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    frame_count = int(stream.get("nb_frames") or 0)
    if frame_count <= 0 and duration_seconds > 0 and fps > 0:
        frame_count = int(round(duration_seconds * fps))

    return {
        "duration_seconds": duration_seconds,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "format_name": format_info.get("format_name"),
    }


def _parse_fraction(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denominator_value = float(denominator)
            if denominator_value == 0:
                return 0.0
            return float(numerator) / denominator_value
        except ValueError:
            return 0.0

    try:
        return float(value)
    except ValueError:
        return 0.0


def _phase_overlay_color_bgr(phase: str | None) -> tuple[int, int, int]:
    phase_key = (phase or "").strip().lower()
    return PHASE_OVERLAY_BGR.get(phase_key, PHASE_OVERLAY_BGR["unknown"])


def _draw_shadowed_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    font_scale: float,
    thickness: int,
) -> None:
    shadow_origin = (origin[0] + 1, origin[1] + 1)
    cv2.putText(
        frame,
        text,
        shadow_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
