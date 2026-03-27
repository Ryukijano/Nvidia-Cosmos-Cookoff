"""Cosmos Sentinel — Enterprise API Service.

FastAPI wrapper around the Cosmos Sentinel pipeline providing:
  - API-key authentication
  - Per-key rate limiting
  - Video-upload analysis endpoint
  - Single-frame analysis endpoint
  - Webhook dispatch on high-risk detections
  - Job-status polling

Run standalone:
    uvicorn api_service:app --host 0.0.0.0 --port 7861 --reload

Or embed in the Streamlit Space via a background thread (see enterprise_dashboard_page.py).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import tempfile
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import cv2
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cosmos Sentinel Enterprise API",
    description=(
        "Production-grade API for AI-powered collision detection and risk narration. "
        "Designed for smart factory, autonomous vehicle, and smart-city integrations."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory API key store  (replace with DB in production)
# ---------------------------------------------------------------------------
_API_KEYS: dict[str, dict[str, Any]] = {
    "demo-key-free-tier": {"tier": "free", "owner": "demo", "rpm": 10},
    "sentinel-enterprise-key": {"tier": "enterprise", "owner": "enterprise_demo", "rpm": 200},
}

# ---------------------------------------------------------------------------
# Simple sliding-window rate limiter (per API key, per minute)
# ---------------------------------------------------------------------------
_rate_windows: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(api_key: str, rpm: int) -> None:
    now = time.time()
    window = _rate_windows[api_key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= rpm:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({rpm} rpm). Upgrade your plan.",
        )
    window.append(now)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
def _get_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> dict[str, Any]:
    key_info = _API_KEYS.get(x_api_key)
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key. Include a valid X-API-Key header.",
        )
    _check_rate_limit(x_api_key, key_info["rpm"])
    return {**key_info, "api_key": x_api_key}


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
_jobs: dict[str, dict[str, Any]] = {}


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: float
    completed_at: float | None = None
    result: dict | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Webhook config models
# ---------------------------------------------------------------------------
class WebhookConfig(BaseModel):
    url: str
    risk_threshold: float = 0.7
    include_payload: bool = True


_webhook_registry: dict[str, WebhookConfig] = {}


# ---------------------------------------------------------------------------
# Webhook dispatcher
# ---------------------------------------------------------------------------
async def _dispatch_webhook(cfg: WebhookConfig, payload: dict) -> None:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(cfg.url, json=payload)
    except Exception as exc:
        print(f"[webhook] dispatch failed to {cfg.url}: {exc}")


# ---------------------------------------------------------------------------
# Pipeline runner (wraps cosmos_sentinel_backend — lazy import to keep API
# server importable even when heavy ML deps are loading)
# ---------------------------------------------------------------------------
def _run_sentinel_on_file(video_path: str, include_predict: bool = False) -> dict:
    from cosmos_sentinel_backend import run_pipeline
    return run_pipeline(video_path, include_predict=include_predict)


def _run_sentinel_on_frame(frame_bgr: np.ndarray, tmp_dir: str) -> dict:
    """Wrap a single BGR frame as a 1-second 30fps stub clip then run BADAS."""
    from cosmos_sentinel_backend import run_pipeline

    tmp_path = str(Path(tmp_dir) / f"frame_{uuid.uuid4().hex}.mp4")
    h, w = frame_bgr.shape[:2]
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for _ in range(30):
        writer.write(frame_bgr)
    writer.release()
    return run_pipeline(tmp_path, include_predict=False)


# ---------------------------------------------------------------------------
# Background job executor
# ---------------------------------------------------------------------------
def _execute_job(job_id: str, video_path: str, include_predict: bool, api_key: str) -> None:
    _jobs[job_id]["status"] = "running"
    try:
        result = _run_sentinel_on_file(video_path, include_predict=include_predict)
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["completed_at"] = time.time()
        _jobs[job_id]["result"] = {
            "pipeline_mode": result.get("pipeline_payload", {}).get("pipeline_mode"),
            "overview": result.get("pipeline_payload", {}).get("overview"),
            "badas_summary": {
                k: v
                for k, v in (result.get("badas_result") or {}).items()
                if k in ("collision_detected", "alert_time", "confidence", "valid_prediction_max")
            },
            "reason_text": (result.get("reason_result") or {}).get("text", ""),
            "artifacts": result.get("pipeline_payload", {}).get("artifacts", {}),
        }

        # Fire webhooks if risk threshold exceeded
        overview = _jobs[job_id]["result"].get("overview") or {}
        peak = overview.get("peak_probability") or 0.0
        for cfg in _webhook_registry.values():
            if float(peak) >= cfg.risk_threshold:
                webhook_payload = {
                    "event": "high_risk_detection",
                    "job_id": job_id,
                    "peak_collision_probability": peak,
                    "alert_time_sec": overview.get("alert_time_sec"),
                    "incident_type": overview.get("incident_type"),
                    "severity": overview.get("severity_label"),
                    "api_key_owner": _API_KEYS.get(api_key, {}).get("owner"),
                }
                if not cfg.include_payload:
                    webhook_payload.pop("job_id")
                asyncio.run(_dispatch_webhook(cfg, webhook_payload))

    except Exception as exc:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["completed_at"] = time.time()
        _jobs[job_id]["error"] = str(exc)
    finally:
        try:
            Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["info"])
def root():
    return {
        "service": "Cosmos Sentinel Enterprise API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/analyze/video", "/analyze/frame", "/jobs/{job_id}", "/webhooks"],
    }


@app.get("/health", tags=["info"])
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/analyze/video", tags=["analysis"], status_code=202)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    include_predict: bool = False,
    key_info: dict = Depends(_get_api_key),
):
    """Upload a video file for async collision-risk analysis.

    Returns a ``job_id`` — poll ``/jobs/{job_id}`` for results.
    """
    if not file.filename or not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "Unsupported file type. Upload an MP4, AVI, MOV, or MKV file.")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = str(Path(tmp_dir) / f"{uuid.uuid4().hex}_{file.filename}")
    contents = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(contents)

    job_id = hashlib.md5(f"{tmp_path}{time.time()}".encode()).hexdigest()[:12]
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "completed_at": None,
        "result": None,
        "error": None,
        "owner": key_info.get("owner"),
    }
    background_tasks.add_task(
        _execute_job, job_id, tmp_path, include_predict, key_info["api_key"]
    )
    return {"job_id": job_id, "status": "queued", "poll_url": f"/jobs/{job_id}"}


@app.post("/analyze/frame", tags=["analysis"])
async def analyze_frame(
    file: UploadFile = File(...),
    key_info: dict = Depends(_get_api_key),
):
    """Upload a single image frame for synchronous BADAS risk scoring.

    Returns risk score immediately (no job queuing).
    """
    try:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
    except Exception as exc:
        raise HTTPException(400, f"Could not decode image: {exc}")

    tmp_dir = tempfile.mkdtemp()
    try:
        result = _run_sentinel_on_frame(frame, tmp_dir)
        overview = (result.get("pipeline_payload") or {}).get("overview") or {}
        badas = result.get("badas_result") or {}
        return {
            "collision_detected": badas.get("collision_detected", False),
            "peak_probability": badas.get("valid_prediction_max"),
            "alert_time_sec": badas.get("alert_time"),
            "risk_score": overview.get("reason_risk_score"),
            "incident_type": overview.get("incident_type"),
            "severity": overview.get("severity_label"),
        }
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}")


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["jobs"])
def get_job(job_id: str, key_info: dict = Depends(_get_api_key)):
    """Poll the status and result of an async analysis job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return job


@app.get("/jobs", tags=["jobs"])
def list_jobs(key_info: dict = Depends(_get_api_key)):
    """List all jobs for the authenticated API key owner."""
    owner = key_info.get("owner")
    return [
        {"job_id": j["job_id"], "status": j["status"], "created_at": j["created_at"]}
        for j in _jobs.values()
        if j.get("owner") == owner
    ]


@app.post("/webhooks", tags=["webhooks"], status_code=201)
def register_webhook(cfg: WebhookConfig, key_info: dict = Depends(_get_api_key)):
    """Register a webhook URL to receive high-risk collision alerts."""
    wid = hashlib.md5(f"{cfg.url}{key_info['api_key']}".encode()).hexdigest()[:8]
    _webhook_registry[wid] = cfg
    return {"webhook_id": wid, "url": cfg.url, "risk_threshold": cfg.risk_threshold}


@app.get("/webhooks", tags=["webhooks"])
def list_webhooks(key_info: dict = Depends(_get_api_key)):
    """List all registered webhooks."""
    return [
        {"webhook_id": wid, "url": cfg.url, "risk_threshold": cfg.risk_threshold}
        for wid, cfg in _webhook_registry.items()
    ]


@app.delete("/webhooks/{webhook_id}", tags=["webhooks"])
def delete_webhook(webhook_id: str, key_info: dict = Depends(_get_api_key)):
    """Remove a registered webhook."""
    if webhook_id not in _webhook_registry:
        raise HTTPException(404, f"Webhook '{webhook_id}' not found.")
    del _webhook_registry[webhook_id]
    return {"deleted": webhook_id}


@app.get("/plans", tags=["billing"])
def list_plans():
    """Return available API plans (public endpoint)."""
    return {
        "free": {"rpm": 10, "features": ["video analysis", "frame analysis"]},
        "startup": {"rpm": 60, "features": ["video analysis", "frame analysis", "webhooks", "job history"]},
        "enterprise": {"rpm": 200, "features": ["all features", "SLA", "dedicated support"]},
    }
