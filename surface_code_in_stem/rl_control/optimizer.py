"""PEPG-style optimizer for policy search over control parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np


@dataclass
class PEPGState:
    """Serializable optimizer state."""

    mean: list[float]
    sigma: list[float]
    learning_rate: float
    sigma_learning_rate: float
    iteration: int
    seed: int
    rng_state: dict


class PEPGOptimizer:
    """Parameter-Exploring Policy Gradients (PEPG) with diagonal Gaussian."""

    def __init__(
        self,
        parameter_dim: int,
        *,
        seed: int = 0,
        init_sigma: float = 0.1,
        learning_rate: float = 0.05,
        sigma_learning_rate: float = 0.02,
        sigma_min: float = 1e-4,
    ):
        if parameter_dim <= 0:
            raise ValueError("parameter_dim must be positive")
        self.parameter_dim = parameter_dim
        self.mean = np.zeros(parameter_dim, dtype=np.float64)
        self.sigma = np.full(parameter_dim, init_sigma, dtype=np.float64)
        self.learning_rate = float(learning_rate)
        self.sigma_learning_rate = float(sigma_learning_rate)
        self.sigma_min = float(sigma_min)
        self.iteration = 0
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def ask(self, population_size: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample antithetic population and return (candidates, perturbations)."""
        if population_size <= 0 or population_size % 2 != 0:
            raise ValueError("population_size must be a positive even integer")
        half = population_size // 2
        eps = self._rng.standard_normal((half, self.parameter_dim))
        perturb = np.vstack([eps, -eps]) * self.sigma
        candidates = self.mean + perturb
        return candidates, perturb

    def tell(self, perturbations: np.ndarray, rewards: np.ndarray) -> None:
        """Update mean and sigma from population rewards."""
        perturbations = np.asarray(perturbations, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)
        if perturbations.shape[0] != rewards.shape[0]:
            raise ValueError("perturbations and rewards length mismatch")

        centered = rewards - np.mean(rewards)
        scale = np.std(rewards)
        if scale > 0:
            centered = centered / (scale + 1e-8)

        grad_mean = perturbations.T @ centered / len(rewards)
        self.mean += self.learning_rate * grad_mean

        sigma_grad = (
            ((perturbations**2 - self.sigma**2) / np.maximum(self.sigma, 1e-8)).T
            @ centered
            / len(rewards)
        )
        self.sigma = np.maximum(self.sigma + self.sigma_learning_rate * sigma_grad, self.sigma_min)
        self.iteration += 1

    def state_dict(self) -> PEPGState:
        """Return serializable state."""
        return PEPGState(
            mean=self.mean.tolist(),
            sigma=self.sigma.tolist(),
            learning_rate=self.learning_rate,
            sigma_learning_rate=self.sigma_learning_rate,
            iteration=self.iteration,
            seed=self.seed,
            rng_state=self._rng.bit_generator.state,
        )

    def load_state_dict(self, state: PEPGState) -> None:
        """Load optimizer from state."""
        self.mean = np.asarray(state.mean, dtype=np.float64)
        self.sigma = np.asarray(state.sigma, dtype=np.float64)
        self.learning_rate = state.learning_rate
        self.sigma_learning_rate = state.sigma_learning_rate
        self.iteration = state.iteration
        self.seed = state.seed
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = state.rng_state

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist optimizer checkpoint as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self.state_dict()), f, indent=2)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "PEPGOptimizer":
        """Restore optimizer from checkpoint."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        state = PEPGState(**payload)
        opt = cls(parameter_dim=len(state.mean), seed=state.seed)
        opt.load_state_dict(state)
        return opt
