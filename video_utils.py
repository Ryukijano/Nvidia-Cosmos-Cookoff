from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

STREAMLIT_SERVER_MAX_UPLOAD_MB = 4096
INLINE_VIDEO_PREVIEW_MAX_MB = 256
MIN_FREE_WORKSPACE_BUFFER_MB = int(os.getenv("SPACE_MIN_FREE_SPACE_BUFFER_MB", "2048"))
COPY_CHUNK_BYTES = 8 * 1024 * 1024
SUPPORTED_VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]


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
