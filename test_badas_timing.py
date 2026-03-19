#!/usr/bin/env python3
"""Tests for BADAS stage timing instrumentation (Issue #2)."""

import json
import os
import sys
import time
import types
import unittest
from unittest.mock import patch, MagicMock

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

# Stub out heavy dependencies that aren't available in the test environment
for _mod_name in ["torch", "cv2", "matplotlib", "matplotlib.pyplot", "numpy", "imageio"]:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _mod_name == "torch":
            _stub.backends = types.ModuleType("torch.backends")
            _stub.backends.cudnn = MagicMock()
            _stub.set_float32_matmul_precision = MagicMock()
            sys.modules["torch.backends"] = _stub.backends
            sys.modules["torch.backends.cudnn"] = _stub.backends.cudnn
        if _mod_name == "numpy":
            _stub.zeros_like = MagicMock()
            _stub.zeros = MagicMock()
            _stub.uint8 = MagicMock()
            _stub.float32 = "float32"
        if _mod_name == "matplotlib.pyplot":
            _stub.subplots = MagicMock()
            _stub.Rectangle = MagicMock()
            _stub.savefig = MagicMock()
            _stub.close = MagicMock()
        if _mod_name == "cv2":
            _stub.VideoCapture = MagicMock()
            _stub.addWeighted = MagicMock()
            _stub.putText = MagicMock()
            _stub.rectangle = MagicMock()
            _stub.resize = MagicMock()
            _stub.hconcat = MagicMock()
            _stub.imwrite = MagicMock()
            _stub.cvtColor = MagicMock()
            _stub.applyColorMap = MagicMock()
            _stub.FONT_HERSHEY_SIMPLEX = 0
            _stub.COLORMAP_JET = 0
            _stub.COLOR_BGR2RGB = 0
            _stub.CAP_PROP_FPS = 5
            _stub.CAP_PROP_POS_FRAMES = 1
        sys.modules[_mod_name] = _stub


# A canned BADAS_JSON payload that badas_detector.py would emit on success
CANNED_BADAS_JSON = {
    "collision_detected": True,
    "alert_time": 3.5,
    "confidence": 0.82,
    "valid_prediction_count": 25,
    "valid_prediction_max": 0.91,
    "valid_prediction_min": 0.02,
    "valid_prediction_mean": 0.35,
    "valid_prediction_median": 0.30,
    "valid_prediction_std": 0.20,
    "valid_prediction_p90": 0.78,
    "valid_prediction_p95": 0.85,
    "first_valid_time": 1.0,
    "alert_source": "threshold",
    "nan_warmup_count": 3,
    "threshold": 0.5,
    "alert_frame_sampled": 7,
    "alert_frame_original_approx": 105,
    "top_predictions": [
        {"time_sec": 3.5, "probability": 0.91, "sampled_frame": 7, "original_frame_approx": 105},
    ],
    "prediction_series": [],
    "threshold_summary": {"threshold_crossing_count": 2, "contiguous_alert_runs": []},
    "prediction_window_summary": {"peak_window_end_time": 4.0, "max_average_probability": 0.7},
    "model_info": {},
    "video_metadata": {},
}

CANNED_BADAS_STDOUT = (
    "BADAS running...\n"
    f"BADAS_JSON: {json.dumps(CANNED_BADAS_JSON)}\n"
    "Done\n"
)


def _make_subprocess_result(returncode, stdout, stderr=""):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


def _make_extract_clip_noop(video_path, alert_time, output_path):
    """Fake extract_pre_alert_clip that just creates an empty file."""
    with open(output_path, "wb") as f:
        f.write(b"\x00")


class TestBadasTimingOnSuccess(unittest.TestCase):
    """When BADAS succeeds, steps.badas.timing and top-level timing must be present."""

    @patch("main_pipeline.create_visualizations", return_value={})
    @patch("extract_clip.extract_pre_alert_clip", side_effect=_make_extract_clip_noop)
    @patch("subprocess.run")
    @patch("sys.argv", ["test"])
    def test_timing_fields_present_on_success(self, mock_subprocess, mock_extract, mock_viz):
        # BADAS succeeds, Reason also succeeds with minimal output
        reason_json = {
            "text": "test",
            "risk_score": 3,
            "incident_type": "near_miss",
            "severity_label": "moderate",
            "bboxes": {},
            "bbox_count": 0,
            "counterfactual_prompt": "",
            "parsing_summary": {"parsed_field_flags": {}, "missing_fields": [], "parsed_field_count": 0, "total_expected_fields": 0},
            "frame_metadata": {},
            "validation": {"is_reliable": True, "flags": {}},
        }
        reason_stdout = f"REASON_JSON: {json.dumps(reason_json)}\n"

        def subprocess_side_effect(cmd, **kwargs):
            script = cmd[1] if len(cmd) > 1 else ""
            if "badas_detector.py" in script:
                return _make_subprocess_result(0, CANNED_BADAS_STDOUT)
            if "cosmos_risk_narrator.py" in script:
                return _make_subprocess_result(0, reason_stdout)
            return _make_subprocess_result(1, "", "unknown script")

        mock_subprocess.side_effect = subprocess_side_effect

        from main_pipeline import run_pipeline
        result = run_pipeline(video_path="./fake_video.mp4")

        # --- Assert steps.badas.timing ---
        iterations = result.get("iterations", [])
        self.assertTrue(len(iterations) > 0, "Pipeline should have at least one iteration")
        badas_step = iterations[-1].get("steps", {}).get("badas", {})
        self.assertIn("timing", badas_step, "steps.badas must contain 'timing'")

        timing = badas_step["timing"]
        self.assertIn("duration_sec", timing)
        self.assertIsInstance(timing["duration_sec"], float)
        self.assertGreater(timing["duration_sec"], 0.0)

        self.assertIn("predictions_per_sec", timing)
        # On success with valid_prediction_count > 0, predictions_per_sec should be a positive float
        self.assertIsInstance(timing["predictions_per_sec"], float)
        self.assertGreater(timing["predictions_per_sec"], 0.0)

        # --- Assert top-level timing ---
        self.assertIn("timing", result, "Pipeline result must contain top-level 'timing'")
        top_timing = result["timing"]
        self.assertIn("total_duration_sec", top_timing)
        self.assertIsInstance(top_timing["total_duration_sec"], float)
        self.assertGreater(top_timing["total_duration_sec"], 0.0)

        self.assertIn("stages", top_timing)
        self.assertIn("badas", top_timing["stages"])
        self.assertIsInstance(top_timing["stages"]["badas"], float)
        self.assertGreater(top_timing["stages"]["badas"], 0.0)


class TestBadasTimingOnFailure(unittest.TestCase):
    """When BADAS fails, steps.badas.timing must still be present with throughput null."""

    @patch("main_pipeline.create_visualizations", return_value={})
    @patch("extract_clip.extract_pre_alert_clip", side_effect=_make_extract_clip_noop)
    @patch("subprocess.run")
    @patch("sys.argv", ["test"])
    def test_timing_present_on_badas_failure(self, mock_subprocess, mock_extract, mock_viz):
        reason_json = {
            "text": "test",
            "risk_score": 1,
            "incident_type": "unclear",
            "severity_label": "unknown",
            "bboxes": {},
            "bbox_count": 0,
            "counterfactual_prompt": "",
            "parsing_summary": {"parsed_field_flags": {}, "missing_fields": [], "parsed_field_count": 0, "total_expected_fields": 0},
            "frame_metadata": {},
            "validation": {"is_reliable": True, "flags": {}},
        }
        reason_stdout = f"REASON_JSON: {json.dumps(reason_json)}\n"

        def subprocess_side_effect(cmd, **kwargs):
            script = cmd[1] if len(cmd) > 1 else ""
            if "badas_detector.py" in script:
                return _make_subprocess_result(1, "", "BADAS crashed")
            if "cosmos_risk_narrator.py" in script:
                return _make_subprocess_result(0, reason_stdout)
            return _make_subprocess_result(1, "", "unknown script")

        mock_subprocess.side_effect = subprocess_side_effect

        from main_pipeline import run_pipeline
        result = run_pipeline(video_path="./fake_video.mp4")

        iterations = result.get("iterations", [])
        self.assertTrue(len(iterations) > 0)
        badas_step = iterations[-1].get("steps", {}).get("badas", {})

        # Timing must be present even on failure
        self.assertIn("timing", badas_step, "steps.badas must contain 'timing' even on failure")
        timing = badas_step["timing"]
        self.assertIn("duration_sec", timing)
        self.assertIsInstance(timing["duration_sec"], float)
        self.assertGreater(timing["duration_sec"], 0.0)

        # Throughput must be None on failure
        self.assertIn("predictions_per_sec", timing)
        self.assertIsNone(timing["predictions_per_sec"], "predictions_per_sec must be None on BADAS failure")

        # Top-level timing must still be present
        self.assertIn("timing", result)
        top_timing = result["timing"]
        self.assertIn("total_duration_sec", top_timing)
        self.assertIsInstance(top_timing["total_duration_sec"], float)
        self.assertGreater(top_timing["total_duration_sec"], 0.0)
        self.assertIn("stages", top_timing)
        self.assertIn("badas", top_timing["stages"])
        self.assertIsInstance(top_timing["stages"]["badas"], float)
        self.assertGreater(top_timing["stages"]["badas"], 0.0)


class TestBadasThroughputArithmetic(unittest.TestCase):
    """predictions_per_sec must equal valid_prediction_count / duration_sec."""

    @patch("main_pipeline.create_visualizations", return_value={})
    @patch("extract_clip.extract_pre_alert_clip", side_effect=_make_extract_clip_noop)
    @patch("subprocess.run")
    @patch("sys.argv", ["test"])
    def test_predictions_per_sec_value(self, mock_subprocess, mock_extract, mock_viz):
        reason_json = {
            "text": "test", "risk_score": 1, "incident_type": "unclear",
            "severity_label": "unknown", "bboxes": {}, "bbox_count": 0,
            "counterfactual_prompt": "",
            "parsing_summary": {"parsed_field_flags": {}, "missing_fields": [], "parsed_field_count": 0, "total_expected_fields": 0},
            "frame_metadata": {}, "validation": {"is_reliable": True, "flags": {}},
        }
        reason_stdout = f"REASON_JSON: {json.dumps(reason_json)}\n"

        def subprocess_side_effect(cmd, **kwargs):
            script = cmd[1] if len(cmd) > 1 else ""
            if "badas_detector.py" in script:
                return _make_subprocess_result(0, CANNED_BADAS_STDOUT)
            if "cosmos_risk_narrator.py" in script:
                return _make_subprocess_result(0, reason_stdout)
            return _make_subprocess_result(1, "", "unknown")

        mock_subprocess.side_effect = subprocess_side_effect

        # Control time.time() to get a known BADAS duration of exactly 2.0s
        call_count = {"n": 0}
        fake_times = [100.0, 102.0]  # badas_start=100, badas_end=102 → duration=2.0

        original_time = time.time

        def fake_time():
            if call_count["n"] < len(fake_times):
                val = fake_times[call_count["n"]]
                call_count["n"] += 1
                return val
            return original_time()

        with patch("main_pipeline.time") as mock_time_mod:
            mock_time_mod.time = fake_time

            from main_pipeline import run_pipeline
            result = run_pipeline(video_path="./fake_video.mp4")

        badas_step = result["iterations"][-1]["steps"]["badas"]
        timing = badas_step["timing"]

        # valid_prediction_count = 25, duration = 2.0s → 12.5 pred/s
        self.assertAlmostEqual(timing["duration_sec"], 2.0, places=1)
        self.assertAlmostEqual(timing["predictions_per_sec"], 25.0 / 2.0, places=1)

    @patch("main_pipeline.create_visualizations", return_value={})
    @patch("extract_clip.extract_pre_alert_clip", side_effect=_make_extract_clip_noop)
    @patch("subprocess.run")
    @patch("sys.argv", ["test"])
    def test_predictions_per_sec_none_when_zero_predictions(self, mock_subprocess, mock_extract, mock_viz):
        """If valid_prediction_count is 0 (or missing), throughput must be None."""
        zero_pred_badas = dict(CANNED_BADAS_JSON)
        zero_pred_badas["valid_prediction_count"] = 0
        zero_pred_stdout = f"BADAS_JSON: {json.dumps(zero_pred_badas)}\n"
        reason_json = {
            "text": "test", "risk_score": 1, "incident_type": "unclear",
            "severity_label": "unknown", "bboxes": {}, "bbox_count": 0,
            "counterfactual_prompt": "",
            "parsing_summary": {"parsed_field_flags": {}, "missing_fields": [], "parsed_field_count": 0, "total_expected_fields": 0},
            "frame_metadata": {}, "validation": {"is_reliable": True, "flags": {}},
        }
        reason_stdout = f"REASON_JSON: {json.dumps(reason_json)}\n"

        def subprocess_side_effect(cmd, **kwargs):
            script = cmd[1] if len(cmd) > 1 else ""
            if "badas_detector.py" in script:
                return _make_subprocess_result(0, zero_pred_stdout)
            if "cosmos_risk_narrator.py" in script:
                return _make_subprocess_result(0, reason_stdout)
            return _make_subprocess_result(1, "", "unknown")

        mock_subprocess.side_effect = subprocess_side_effect

        from main_pipeline import run_pipeline
        result = run_pipeline(video_path="./fake_video.mp4")

        badas_step = result["iterations"][-1]["steps"]["badas"]
        timing = badas_step["timing"]
        self.assertIsNone(timing["predictions_per_sec"],
                          "predictions_per_sec must be None when valid_prediction_count is 0")


class TestBadasTimingOnException(unittest.TestCase):
    """When BADAS throws an exception, timing must still be captured."""

    @patch("main_pipeline.create_visualizations", return_value={})
    @patch("extract_clip.extract_pre_alert_clip", side_effect=_make_extract_clip_noop)
    @patch("subprocess.run", side_effect=Exception("subprocess exploded"))
    @patch("sys.argv", ["test"])
    def test_timing_present_on_badas_exception(self, mock_subprocess, mock_extract, mock_viz):
        from main_pipeline import run_pipeline
        result = run_pipeline(video_path="./fake_video.mp4")

        iterations = result.get("iterations", [])
        self.assertTrue(len(iterations) > 0)
        badas_step = iterations[-1].get("steps", {}).get("badas", {})

        self.assertIn("timing", badas_step)
        timing = badas_step["timing"]
        self.assertIsInstance(timing["duration_sec"], float)
        self.assertGreater(timing["duration_sec"], 0.0)
        self.assertIsNone(timing["predictions_per_sec"])

        # Top-level timing
        self.assertIn("timing", result)
        self.assertGreater(result["timing"]["total_duration_sec"], 0.0)
        self.assertGreater(result["timing"]["stages"]["badas"], 0.0)


if __name__ == "__main__":
    unittest.main()
