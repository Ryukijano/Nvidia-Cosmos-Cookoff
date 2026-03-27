"""Cosmos Sentinel — Live Stream / RTSP Backend.

Provides a rolling-window frame buffer that continuously samples frames from:
  - A local file path (looped playback, for demo purposes)
  - An RTSP URL (real camera feed)
  - An HTTP MJPEG stream
  - A NumPy frame injected programmatically (for Streamlit webcam / frame upload mode)

The buffer runs in a background thread and exposes the latest N frames plus a
live risk score computed by the lightweight BADAS frame-rate sampler.

Usage::

    from livestream_backend import StreamManager
    mgr = StreamManager()
    mgr.start("rtsp://192.168.1.10:554/stream0")  # or a local .mp4 path
    snapshot = mgr.latest_snapshot()
    mgr.stop()
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------
BUFFER_FRAMES = 30          # max frames kept in the ring buffer
SAMPLE_EVERY_N = 5          # analyse 1 in every N captured frames
RISK_HISTORY_LEN = 120      # keep ~2 minutes of risk scores @ 1 fps
TARGET_CAPTURE_FPS = 10     # desired capture rate for demo streams


@dataclass
class FrameRecord:
    frame_id: str
    timestamp: float
    frame_bgr: np.ndarray
    width: int
    height: int


@dataclass
class RiskEvent:
    timestamp: float
    collision_detected: bool
    probability: float
    alert_time_sec: float | None
    incident_type: str | None
    severity: str | None
    source_frame_id: str


@dataclass
class StreamSnapshot:
    """Latest state exported to Streamlit / dashboard consumers."""
    running: bool
    source: str
    frame_count: int
    fps: float
    latest_frame: np.ndarray | None
    risk_history: list[RiskEvent]
    latest_risk: RiskEvent | None
    incident_log: list[dict]
    stats: dict


# ---------------------------------------------------------------------------
# Lightweight BADAS frame-level scorer
# ---------------------------------------------------------------------------
def _score_frame(frame_bgr: np.ndarray, tmp_dir: str) -> dict:
    """Write frame as short clip and run BADAS on it."""
    try:
        from cosmos_sentinel_backend import run_pipeline

        clip_path = str(Path(tmp_dir) / f"live_{uuid.uuid4().hex[:8]}.mp4")
        h, w = frame_bgr.shape[:2]
        writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for _ in range(30):
            writer.write(frame_bgr)
        writer.release()

        result = run_pipeline(clip_path, include_predict=False)
        badas = result.get("badas_result") or {}
        overview = (result.get("pipeline_payload") or {}).get("overview") or {}
        Path(clip_path).unlink(missing_ok=True)
        return {
            "collision_detected": bool(badas.get("collision_detected", False)),
            "probability": float(badas.get("valid_prediction_max") or 0.0),
            "alert_time_sec": badas.get("alert_time"),
            "incident_type": overview.get("incident_type"),
            "severity": overview.get("severity_label"),
        }
    except Exception as exc:
        return {
            "collision_detected": False,
            "probability": 0.0,
            "alert_time_sec": None,
            "incident_type": None,
            "severity": None,
            "_error": str(exc),
        }


# ---------------------------------------------------------------------------
# Simulated risk scorer (used when heavy ML deps unavailable)
# ---------------------------------------------------------------------------
def _simulate_risk(frame_bgr: np.ndarray) -> dict:
    """
    Return a probabilistic risk score derived purely from frame pixel stats.
    Used as a lightweight fallback when the full BADAS pipeline is unavailable
    (e.g., no GPU) so the dashboard remains interactive for demos.
    """
    rng = np.random.default_rng(int(np.mean(frame_bgr)) * 17)
    base_prob = float(np.clip(np.std(frame_bgr.astype(float)) / 255.0, 0.0, 1.0))
    jitter = float(rng.uniform(-0.05, 0.25))
    prob = float(np.clip(base_prob + jitter, 0.0, 1.0))
    collision = prob > 0.65
    return {
        "collision_detected": collision,
        "probability": round(prob, 4),
        "alert_time_sec": round(float(rng.uniform(0.5, 3.0)), 2) if collision else None,
        "incident_type": "simulated_near_miss" if collision else None,
        "severity": ("HIGH" if prob > 0.8 else "MEDIUM") if collision else None,
        "_simulated": True,
    }


# ---------------------------------------------------------------------------
# Stream Manager
# ---------------------------------------------------------------------------
class StreamManager:
    """Thread-safe live stream manager with rolling risk analysis."""

    def __init__(
        self,
        use_full_pipeline: bool = False,
        on_incident: Callable[[RiskEvent], None] | None = None,
    ):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._analysis_thread: threading.Thread | None = None

        self._frame_buffer: deque[FrameRecord] = deque(maxlen=BUFFER_FRAMES)
        self._analysis_queue: queue.Queue = queue.Queue(maxsize=4)
        self._risk_history: deque[RiskEvent] = deque(maxlen=RISK_HISTORY_LEN)
        self._incident_log: list[dict] = []

        self._source: str = ""
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._last_fps_time: float = time.time()
        self._fps_frame_count: int = 0

        self._use_full_pipeline = use_full_pipeline
        self._on_incident = on_incident

        import tempfile
        self._tmp_dir = tempfile.mkdtemp(prefix="sentinel_stream_")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self, source: str) -> None:
        """Start capturing from *source* (file path, RTSP URL, or MJPEG URL)."""
        self.stop()
        self._stop_event.clear()
        self._source = source
        self._frame_count = 0
        self._frame_buffer.clear()
        self._risk_history.clear()

        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="sentinel_capture"
        )
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop, daemon=True, name="sentinel_analysis"
        )
        self._capture_thread.start()
        self._analysis_thread.start()

    def stop(self) -> None:
        """Stop all background threads."""
        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5.0)
        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_queue.put(None)  # poison pill
            self._analysis_thread.join(timeout=5.0)

    def inject_frame(self, frame_bgr: np.ndarray) -> None:
        """Push a single frame directly (for webcam / uploaded frame mode)."""
        rec = FrameRecord(
            frame_id=uuid.uuid4().hex[:8],
            timestamp=time.time(),
            frame_bgr=frame_bgr.copy(),
            width=frame_bgr.shape[1],
            height=frame_bgr.shape[0],
        )
        with self._lock:
            self._frame_buffer.append(rec)
            self._frame_count += 1

        try:
            self._analysis_queue.put_nowait(rec)
        except queue.Full:
            pass

    def latest_snapshot(self) -> StreamSnapshot:
        with self._lock:
            history = list(self._risk_history)
            incidents = list(self._incident_log[-50:])
            latest_frame = (
                self._frame_buffer[-1].frame_bgr.copy()
                if self._frame_buffer
                else None
            )
            latest_risk = history[-1] if history else None
            return StreamSnapshot(
                running=not self._stop_event.is_set(),
                source=self._source,
                frame_count=self._frame_count,
                fps=round(self._fps, 1),
                latest_frame=latest_frame,
                risk_history=history,
                latest_risk=latest_risk,
                incident_log=incidents,
                stats=self._compute_stats(history),
            )

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            print(f"[stream] Cannot open source: {self._source}")
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(source_fps / TARGET_CAPTURE_FPS))
        frame_idx = 0
        sleep_dt = 1.0 / TARGET_CAPTURE_FPS

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # Loop file-based sources
                if Path(self._source).exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_idx += 1
            if frame_idx % step != 0:
                continue

            rec = FrameRecord(
                frame_id=uuid.uuid4().hex[:8],
                timestamp=time.time(),
                frame_bgr=frame,
                width=frame.shape[1],
                height=frame.shape[0],
            )
            with self._lock:
                self._frame_buffer.append(rec)
                self._frame_count += 1
                self._fps_frame_count += 1
                now = time.time()
                elapsed = now - self._last_fps_time
                if elapsed >= 1.0:
                    self._fps = self._fps_frame_count / elapsed
                    self._fps_frame_count = 0
                    self._last_fps_time = now

            if frame_idx % SAMPLE_EVERY_N == 0:
                try:
                    self._analysis_queue.put_nowait(rec)
                except queue.Full:
                    pass

            time.sleep(sleep_dt)

        cap.release()

    def _analysis_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                rec = self._analysis_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if rec is None:
                break

            if self._use_full_pipeline:
                scores = _score_frame(rec.frame_bgr, self._tmp_dir)
            else:
                scores = _simulate_risk(rec.frame_bgr)

            event = RiskEvent(
                timestamp=rec.timestamp,
                collision_detected=scores["collision_detected"],
                probability=scores["probability"],
                alert_time_sec=scores.get("alert_time_sec"),
                incident_type=scores.get("incident_type"),
                severity=scores.get("severity"),
                source_frame_id=rec.frame_id,
            )
            with self._lock:
                self._risk_history.append(event)
                if event.collision_detected:
                    self._incident_log.append({
                        "time": time.strftime("%H:%M:%S", time.localtime(event.timestamp)),
                        "probability": f"{event.probability:.1%}",
                        "incident_type": event.incident_type or "collision",
                        "severity": event.severity or "UNKNOWN",
                        "frame_id": event.source_frame_id,
                    })

            if event.collision_detected and self._on_incident:
                try:
                    self._on_incident(event)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Analytics helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_stats(history: list[RiskEvent]) -> dict:
        if not history:
            return {
                "total_frames_analysed": 0,
                "incidents_detected": 0,
                "avg_probability": 0.0,
                "peak_probability": 0.0,
                "uptime_sec": 0.0,
            }
        probs = [e.probability for e in history]
        incidents = sum(1 for e in history if e.collision_detected)
        uptime = history[-1].timestamp - history[0].timestamp if len(history) > 1 else 0.0
        return {
            "total_frames_analysed": len(history),
            "incidents_detected": incidents,
            "avg_probability": round(float(np.mean(probs)), 4),
            "peak_probability": round(float(np.max(probs)), 4),
            "uptime_sec": round(uptime, 1),
        }


# ---------------------------------------------------------------------------
# Module-level singleton for Streamlit session sharing
# ---------------------------------------------------------------------------
_global_manager: StreamManager | None = None


def get_stream_manager(use_full_pipeline: bool = False) -> StreamManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = StreamManager(use_full_pipeline=use_full_pipeline)
    return _global_manager
