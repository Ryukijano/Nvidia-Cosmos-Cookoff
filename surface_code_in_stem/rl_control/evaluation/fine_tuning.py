"""Evaluation script: fine-tune from a calibrated policy checkpoint."""

from __future__ import annotations

from surface_code_in_stem.rl_control.environment import StimCalibrationConfig, StimCalibrationEnvironment
from surface_code_in_stem.rl_control.optimizer import PEPGOptimizer
from surface_code_in_stem.rl_control.training import TrainingConfig, run_simulator_training


def run(checkpoint_path: str, generations: int = 10) -> list[dict[str, float]]:
    env = StimCalibrationEnvironment(StimCalibrationConfig(seed=123), parameter_dim=4)
    optimizer = PEPGOptimizer.load_checkpoint(checkpoint_path)
    cfg = TrainingConfig(generations=generations, population_size=20, checkpoint_every=generations + 1)
    return run_simulator_training(env, optimizer, config=cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--generations", type=int, default=10)
    args = parser.parse_args()
    metrics = run(args.checkpoint, generations=args.generations)
    print(metrics[-1])
