#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def extract_structured_json(output, prefix):
    for line in output.splitlines():
        if line.startswith(prefix):
            try:
                return json.loads(line.split(prefix, 1)[1].strip())
            except json.JSONDecodeError:
                return None
    return None


def run_process(command, timeout):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def existing_file(path):
    return path if path and os.path.exists(path) else None


def test_badas(video_path):
    result = run_process([sys.executable, "badas_detector.py", video_path], timeout=300)
    payload = extract_structured_json(result["stdout"], "BADAS_JSON:")
    success = result["returncode"] == 0 and payload is not None
    return {
        "success": success,
        "command": [sys.executable, "badas_detector.py", video_path],
        "result": payload,
        "returncode": result["returncode"],
        "stdout_tail": result["stdout"][-4000:],
        "stderr_tail": result["stderr"][-4000:],
    }


def extract_clip(video_path, alert_time, output_path):
    from extract_clip import extract_pre_alert_clip

    extract_pre_alert_clip(video_path, alert_time, output_path)
    return existing_file(output_path)


def test_reason(clip_path):
    result = run_process([sys.executable, "cosmos_risk_narrator.py", clip_path], timeout=300)
    payload = extract_structured_json(result["stdout"], "REASON_JSON:")
    success = result["returncode"] == 0 and payload is not None
    return {
        "success": success,
        "command": [sys.executable, "cosmos_risk_narrator.py", clip_path],
        "result": payload,
        "returncode": result["returncode"],
        "stdout_tail": result["stdout"][-4000:],
        "stderr_tail": result["stderr"][-4000:],
    }


def test_full_pipeline(video_path, timeout):
    result = run_process([sys.executable, "main_pipeline.py", video_path], timeout=timeout)
    payload = extract_structured_json(result["stdout"], "PIPELINE_JSON:")
    success = result["returncode"] == 0 and payload is not None
    return {
        "success": success,
        "command": [sys.executable, "main_pipeline.py", video_path],
        "result": payload,
        "returncode": result["returncode"],
        "stdout_tail": result["stdout"][-8000:],
        "stderr_tail": result["stderr"][-4000:],
    }


def build_report(video_path, run_full_pipeline, pipeline_timeout):
    report = {
        "input_video": video_path,
        "mode": "badas_reason_only",
        "checks": {},
        "artifacts": {},
        "status": "running",
    }

    badas_check = test_badas(video_path)
    report["checks"]["badas"] = badas_check

    alert_time = 5.0
    if badas_check["result"] and badas_check["result"].get("alert_time") is not None:
        alert_time = float(badas_check["result"]["alert_time"])

    clip_path = extract_clip(video_path, alert_time, "./validation_extracted_clip.mp4")
    report["artifacts"]["validation_extracted_clip"] = clip_path

    reason_check = test_reason(clip_path) if clip_path else {
        "success": False,
        "result": None,
        "returncode": None,
        "stdout_tail": "",
        "stderr_tail": "clip extraction failed",
    }
    report["checks"]["reason"] = reason_check

    if run_full_pipeline:
        pipeline_check = test_full_pipeline(video_path, pipeline_timeout)
        report["checks"]["full_pipeline"] = pipeline_check
        payload = pipeline_check.get("result") or {}
        artifacts = payload.get("artifacts") or {}
        report["artifacts"].update(artifacts)
    else:
        report["checks"]["full_pipeline"] = {
            "success": False,
            "skipped": True,
        }

    checks = report["checks"]
    report["status"] = "passed" if checks.get("badas", {}).get("success") and checks.get("reason", {}).get("success") else "partial"
    if run_full_pipeline and not checks.get("full_pipeline", {}).get("success"):
        report["status"] = "partial"
    return report


def print_summary(report):
    print("=" * 72)
    print("COSMOS PIPELINE VALIDATION REPORT")
    print("=" * 72)
    print(f"Input video: {report['input_video']}")
    print(f"Status: {report['status']}")
    print()

    badas = report["checks"].get("badas", {})
    badas_result = badas.get("result") or {}
    print("[BADAS]")
    print(f"success: {badas.get('success')}")
    print(f"collision_detected: {badas_result.get('collision_detected')}")
    print(f"alert_time: {badas_result.get('alert_time')}")
    print(f"confidence: {badas_result.get('confidence')}")
    print(f"peak_probability: {badas_result.get('valid_prediction_max')}")
    print(f"threshold_crossings: {((badas_result.get('threshold_summary') or {}).get('threshold_crossing_count'))}")
    print()

    reason = report["checks"].get("reason", {})
    reason_result = reason.get("result") or {}
    print("[REASON]")
    print(f"success: {reason.get('success')}")
    print(f"incident_type: {reason_result.get('incident_type')}")
    print(f"severity_label: {reason_result.get('severity_label')}")
    print(f"risk_score: {reason_result.get('risk_score')}")
    print(f"bbox_count: {reason_result.get('bbox_count')}")
    print(f"missing_fields: {(reason_result.get('parsing_summary') or {}).get('missing_fields')}")
    print()

    full_pipeline = report["checks"].get("full_pipeline", {})
    print("[FULL PIPELINE]")
    print(f"success: {full_pipeline.get('success')}")
    if full_pipeline.get("skipped"):
        print("skipped: True")
    else:
        pipeline_result = full_pipeline.get("result") or {}
        print(f"pipeline_status: {pipeline_result.get('status')}")
        print(f"overview: {pipeline_result.get('overview')}")
    print()

    print("[ARTIFACTS]")
    for key, value in report["artifacts"].items():
        print(f"{key}: {value}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", nargs="?", default=str(Path(__file__).resolve().parent / "1_first.mp4"))
    parser.add_argument("--stages-only", action="store_true")
    parser.add_argument("--pipeline-timeout", type=int, default=1500)
    args = parser.parse_args()

    video_path = str(Path(args.video_path))
    report = build_report(video_path, run_full_pipeline=not args.stages_only, pipeline_timeout=args.pipeline_timeout)
    print_summary(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
