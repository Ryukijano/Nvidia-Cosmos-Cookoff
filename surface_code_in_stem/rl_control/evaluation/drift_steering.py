"""Evaluation script: drift steering under synthetic sinusoidal/step drifts."""

from __future__ import annotations

import numpy as np

from surface_code_in_stem.rl_control.environment import StimCalibrationConfig, StimCalibrationEnvironment
from surface_code_in_stem.rl_control.optimizer import PEPGOptimizer
from surface_code_in_stem.rl_control.training import TrainingConfig, run_simulator_training


def run(kind: str = "sin", generations: int = 10) -> list[dict[str, float]]:
    config = StimCalibrationConfig(seed=7)
    env = StimCalibrationEnvironment(config, parameter_dim=4)
    optimizer = PEPGOptimizer(parameter_dim=4, seed=7)
    history: list[dict[str, float]] = []

    for t in range(generations):
        if kind == "sin":
            env.config = StimCalibrationConfig(
                **{**env.config.__dict__, "base_error_rate": 0.001 + 0.0005 * np.sin(0.25 * t)}
            )
        elif kind == "step":
            bump = 0.001 if t >= generations // 2 else 0.0
            env.config = StimCalibrationConfig(
                **{**env.config.__dict__, "base_error_rate": 0.001 + bump}
            )
        else:
            raise ValueError("kind must be 'sin' or 'step'.")

        cfg = TrainingConfig(generations=1, population_size=20, checkpoint_every=999)
        history.extend(run_simulator_training(env, optimizer, config=cfg))
    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["sin", "step"], default="sin")
    parser.add_argument("--generations", type=int, default=10)
    args = parser.parse_args()
    metrics = run(kind=args.kind, generations=args.generations)
    print(metrics[-1])
