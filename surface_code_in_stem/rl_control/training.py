"""Training loops for simulator-backed and hardware-adapted control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .environment import ControlEnvironment
from .masking import mask_population_perturbations
from .optimizer import PEPGOptimizer


@dataclass
class TrainingConfig:
    generations: int = 20
    population_size: int = 20
    checkpoint_every: int = 5
    checkpoint_dir: str = "checkpoints/rl_control"


def run_simulator_training(
    env: ControlEnvironment,
    optimizer: PEPGOptimizer,
    *,
    config: TrainingConfig,
    mask: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Run PEPG loop against a calibration environment."""
    history: list[dict[str, float]] = []
    observation = env.reset()

    for generation in range(config.generations):
        candidates, perturbations = optimizer.ask(config.population_size)
        rewards = np.zeros(config.population_size, dtype=np.float64)

        for idx in range(config.population_size):
            action = candidates[idx] - optimizer.mean
            _, reward, _ = env.step(action)
            rewards[idx] = reward

        used_perturb = perturbations
        if mask is not None:
            used_perturb = mask_population_perturbations(perturbations, observation, mask)

        optimizer.tell(used_perturb, rewards)
        observation = env.reset()

        metrics = {
            "generation": float(generation),
            "reward_mean": float(np.mean(rewards)),
            "reward_max": float(np.max(rewards)),
            "sigma_mean": float(np.mean(optimizer.sigma)),
        }
        history.append(metrics)

        if (generation + 1) % config.checkpoint_every == 0:
            ckpt = Path(config.checkpoint_dir) / f"pepg_gen_{generation + 1}.json"
            optimizer.save_checkpoint(ckpt)

    return history
