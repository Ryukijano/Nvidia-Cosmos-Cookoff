"""Evaluation script: recovery from randomized initialization."""

from __future__ import annotations

import numpy as np

from surface_code_in_stem.rl_control.environment import StimCalibrationConfig, StimCalibrationEnvironment
from surface_code_in_stem.rl_control.optimizer import PEPGOptimizer
from surface_code_in_stem.rl_control.training import TrainingConfig, run_simulator_training


def run(trials: int = 3, generations: int = 10) -> list[list[dict[str, float]]]:
    all_histories: list[list[dict[str, float]]] = []
    for seed in range(trials):
        env = StimCalibrationEnvironment(StimCalibrationConfig(seed=seed), parameter_dim=4)
        optimizer = PEPGOptimizer(parameter_dim=4, seed=seed)
        optimizer.mean = np.random.default_rng(seed).uniform(-0.03, 0.03, size=4)
        cfg = TrainingConfig(generations=generations, population_size=20, checkpoint_every=999)
        all_histories.append(run_simulator_training(env, optimizer, config=cfg))
    return all_histories


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--generations", type=int, default=10)
    args = parser.parse_args()
    out = run(trials=args.trials, generations=args.generations)
    print({"trials": len(out), "last": out[-1][-1]})
