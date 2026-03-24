"""RL control module for calibration and steering."""

from .environment import (
    ControlEnvironment,
    HardwareTraceAdapter,
    StimCalibrationConfig,
    StimCalibrationEnvironment,
)
from .masking import (
    apply_masked_detector_weights,
    build_detector_parameter_mask,
    mask_population_perturbations,
    parameter_neighborhoods,
)
from .optimizer import PEPGOptimizer
from .training import TrainingConfig, run_simulator_training

__all__ = [
    "ControlEnvironment",
    "HardwareTraceAdapter",
    "StimCalibrationConfig",
    "StimCalibrationEnvironment",
    "PEPGOptimizer",
    "TrainingConfig",
    "run_simulator_training",
    "build_detector_parameter_mask",
    "parameter_neighborhoods",
    "apply_masked_detector_weights",
    "mask_population_perturbations",
]
