#!/usr/bin/env python3
"""Tests for clip extraction timing instrumentation (Issue #3)."""

import json
import os
import sys
import time
import types
import unittest
from unittest.mock import patch, MagicMock

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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

CANNED_REASON_JSON = {
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

CANNED_REASON_STDOUT = f"REASON_JSON: {json.dumps(CANNED_REASON_JSON)}\n"


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


def _subprocess_badas_and_reason(cmd, **kwargs):
    script = cmd[1] if len(cmd) > 1 else ""
    if "badas_detector.py" in script:
        return _make_subprocess_result(0, CANNED_BADAS_STDOUT)
    if "cosmos_risk_narrator.py" in script:
        return _make_subprocess_result(0, CANNED_REASON_STDOUT)
    return _make_subprocess_result(1, "", "unknown script")


class TestClipExtractionTimingOnSuccess(unittest.TestCase):
    """When clip extraction succeeds, steps.clip_extraction.timing.duration_sec must be a positive float."""

    @patch("main_pipeline.create_visualizations", return_value={})
    @patch("extract_clip.extract_pre_alert_clip", side_effect=_make_extract_clip_noop)
    @patch("subprocess.run", side_effect=_subprocess_badas_and_reason)
    @patch("sys.argv", ["test"])
    def test_clip_extraction_timing_duration_on_success(self, mock_subprocess, mock_extract, mock_viz):
        from main_pipeline import run_pipeline
        result = run_pipeline(video_path="./fake_video.mp4")

        iterations = result.get("iterations", [])
        self.assertTrue(len(iterations) > 0, "Pipeline should have at least one iteration")
        clip_step = iterations[-1].get("steps", {}).get("clip_extraction", {})
        self.assertIn("timing", clip_step, "steps.clip_extraction must contain 'timing'")

        timing = clip_step["timing"]
        self.assertIn("duration_sec", timing)
        self.assertIsInstance(timing["duration_sec"], float)
        self.assertGreater(timing["duration_sec"], 0.0)


class TestClipExtractionRealtimeFactor(unittest.TestCase):
    """When clip extraction succeeds, realtime_factor must be a positive float."""

    @patch("main_pipeline.create_visualizations", return_value={})
    @patch("extract_clip.extract_pre_alert_clip", side_effect=_make_extract_clip_noop)
    @patch("subprocess.run", side_effect=_subprocess_badas_and_reason)
    @patch("sys.argv", ["test"])
    def test_realtime_factor_is_positive_float_on_success(self, mock_subprocess, mock_extract, mock_viz):
        from main_pipeline import run_pipeline
        result = run_pipeline(video_path="./fake_video.mp4")

        clip_step = result["iterations"][-1]["steps"]["clip_extraction"]
        timing = clip_step["timing"]
        self.assertIn("realtime_factor", timing, "timing must contain 'realtime_factor'")
        self.assertIsInstance(timing["realtime_factor"], float)
        self.assertGreater(timing["realtime_factor"], 0.0)


class TestClipExtractionTopLevelTiming(unittest.TestCase):
    """Top-level timing must include clip_extraction stage and total_duration_sec must equal sum of stages."""

    @patch("main_pipeline.create_visualizations", return_value={})
    @patch("extract_clip.extract_pre_alert_clip", side_effect=_make_extract_clip_noop)
    @patch("subprocess.run", side_effect=_subprocess_badas_and_reason)
    @patch("sys.argv", ["test"])
    def test_top_level_timing_includes_clip_extraction(self, mock_subprocess, mock_extract, mock_viz):
        from main_pipeline import run_pipeline
        result = run_pipeline(video_path="./fake_video.mp4")

        top_timing = result.get("timing", {})
        self.assertIn("stages", top_timing)
        self.assertIn("clip_extraction", top_timing["stages"],
                       "top-level timing.stages must include 'clip_extraction'")
        self.assertIsInstance(top_timing["stages"]["clip_extraction"], float)
        self.assertGreater(top_timing["stages"]["clip_extraction"], 0.0)

        # total_duration_sec must equal sum of all stage durations
        stage_sum = sum(top_timing["stages"].values())
        self.assertAlmostEqual(
            top_timing["total_duration_sec"], stage_sum, places=4,
            msg="total_duration_sec must equal sum of all stage durations",
        )


def _extract_clip_explodes(video_path, alert_time, output_path):
    raise RuntimeError("ffmpeg not found")


class TestClipExtractionTimingOnFailure(unittest.TestCase):
    """When clip extraction fails, timing must still be present with realtime_factor null."""

    @patch("extract_clip.extract_pre_alert_clip", side_effect=_extract_clip_explodes)
    @patch("subprocess.run", side_effect=_subprocess_badas_and_reason)
    @patch("sys.argv", ["test"])
    def test_timing_present_on_clip_extraction_failure(self, mock_subprocess, mock_extract):
        from main_pipeline import run_pipeline
        result = run_pipeline(video_path="./fake_video.mp4")

        iterations = result.get("iterations", [])
        self.assertTrue(len(iterations) > 0)
        clip_step = iterations[-1].get("steps", {}).get("clip_extraction", {})

        # Timing must be present even on failure
        self.assertIn("timing", clip_step, "steps.clip_extraction must contain 'timing' even on failure")
        timing = clip_step["timing"]
        self.assertIn("duration_sec", timing)
        self.assertIsInstance(timing["duration_sec"], float)
        self.assertGreater(timing["duration_sec"], 0.0)

        # realtime_factor must be None on failure
        self.assertIn("realtime_factor", timing)
        self.assertIsNone(timing["realtime_factor"],
                          "realtime_factor must be None when clip extraction fails")

        # Top-level timing must include clip_extraction stage duration
        top_timing = result.get("timing", {})
        self.assertIn("stages", top_timing)
        self.assertIn("clip_extraction", top_timing["stages"])
        self.assertIsInstance(top_timing["stages"]["clip_extraction"], float)
        self.assertGreater(top_timing["stages"]["clip_extraction"], 0.0)

        # total_duration_sec must equal sum of stages
        stage_sum = sum(top_timing["stages"].values())
        self.assertAlmostEqual(top_timing["total_duration_sec"], stage_sum, places=4)


if __name__ == "__main__":
    unittest.main()
