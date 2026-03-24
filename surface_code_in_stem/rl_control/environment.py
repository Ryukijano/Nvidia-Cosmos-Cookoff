"""Environment abstractions for RL-based calibration and steering.

Observation vectors are detector statistics (e.g., detector click rates), while
actions are perturbations applied to a control-parameter vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from surface_code_in_stem.surface_code import surface_code_circuit_string


class ControlEnvironment(Protocol):
    """Protocol for calibration environments.

    Implementations should expose detector-statistic observations and accept
    action vectors representing control-parameter perturbations.
    """

    action_dim: int

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, dict[str, float]]:
        """Apply an action, return (observation, reward, diagnostics)."""


@dataclass(frozen=True)
class StimCalibrationConfig:
    """Configuration for a Stim-backed calibration environment."""

    distance: int = 3
    rounds: int = 3
    shots: int = 128
    base_error_rate: float = 0.001
    seed: int = 0


@dataclass(frozen=True)
class QECEnvConfig:
    """Backward-compatible config used by the Streamlit Syndrome-Net page."""

    distance: int = 3
    rounds: int = 3
    noise: float = 0.001
    shots: int = 128
    seed: int = 0
    parameter_dim: int = 4


def qec_environment(config: QECEnvConfig) -> "StimCalibrationEnvironment":
    """Create a Stim calibration environment from legacy QEC config."""

    stim_config = StimCalibrationConfig(
        distance=config.distance,
        rounds=config.rounds,
        shots=config.shots,
        base_error_rate=config.noise,
        seed=config.seed,
    )
    return StimCalibrationEnvironment(stim_config, parameter_dim=config.parameter_dim)


class StimCalibrationEnvironment:
    """Stim-backed environment for policy calibration.

    The environment maintains a parameter vector ``theta``. Each action is added
    to ``theta``. The effective circuit error rate is computed as
    ``clip(base_error_rate + theta.sum(), 1e-6, 0.2)``.
    """

    def __init__(self, config: StimCalibrationConfig, parameter_dim: int = 4):
        if parameter_dim <= 0:
            raise ValueError("parameter_dim must be positive.")
        self.config = config
        self.action_dim = parameter_dim
        self._rng = np.random.default_rng(config.seed)
        self.theta = np.zeros(parameter_dim, dtype=np.float64)

    def _build_sampler(self, physical_error_rate: float):
        try:
            import stim
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError("Stim is required for StimCalibrationEnvironment.") from exc

        circuit_text = surface_code_circuit_string(
            self.config.distance,
            self.config.rounds,
            float(physical_error_rate),
        )
        circuit = stim.Circuit(circuit_text)
        sampler = circuit.compile_detector_sampler(seed=self.config.seed)
        return sampler

    def _evaluate(self) -> tuple[np.ndarray, float, dict[str, float]]:
        physical_error_rate = float(
            np.clip(self.config.base_error_rate + np.sum(self.theta), 1e-6, 0.2)
        )
        sampler = self._build_sampler(physical_error_rate)
        det_samples, obs_samples = sampler.sample(
            self.config.shots,
            separate_observables=True,
        )
        detector_rates = np.mean(det_samples, axis=0, dtype=np.float64)
        logical_error_rate = float(np.mean(obs_samples[:, 0], dtype=np.float64))
        reward = -logical_error_rate
        diagnostics = {
            "logical_error_rate": logical_error_rate,
            "effective_p": physical_error_rate,
        }
        return detector_rates, reward, diagnostics

    def reset(self) -> np.ndarray:
        self.theta.fill(0.0)
        observation, _, _ = self._evaluate()
        return observation

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, dict[str, float]]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,):
            raise ValueError(f"action must have shape ({self.action_dim},).")
        self.theta = np.clip(self.theta + action, -0.05, 0.05)
        return self._evaluate()


class HardwareTraceAdapter:
    """Adapter hooks for future hardware trace integration.

    Users should subclass this and implement trace acquisition plus reward
    extraction from experiment metadata.
    """

    def observation_from_trace(self, trace: np.ndarray) -> np.ndarray:
        """Convert raw hardware traces to detector-statistic observations."""
        trace = np.asarray(trace, dtype=np.float64)
        if trace.ndim != 2:
            raise ValueError("trace must be a 2D array [shots, detectors].")
        return np.mean(trace, axis=0)

    def reward_from_trace(self, trace: np.ndarray) -> float:
        """Compute reward from hardware trace.

        Placeholder implementation minimizes total detector activity.
        """
        obs = self.observation_from_trace(trace)
        return -float(np.mean(obs))
