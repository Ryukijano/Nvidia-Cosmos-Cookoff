"""Helpers for quickly comparing nested surface-code policies.

The helpers in this module run tiny Stim simulations for both the original
static surface code builder and one of the dynamic builders. They return simple
tables of logical error rates that can be consumed by notebooks or tests
without requiring long simulation times.
"""

from __future__ import annotations

from importlib.util import find_spec
import multiprocessing as mp
from typing import Any, Callable, Dict, Iterable, TYPE_CHECKING

import numpy as np

from surface_code_in_stem.dynamic import hexagonal_surface_code
from surface_code_in_stem.decoders import DecoderMetadata, DecoderProtocol, MWPMDecoder
from surface_code_in_stem.surface_code import surface_code_circuit_string
from surface_code_in_stem.rl_control.nested_agent import NestedLearningAgent

if TYPE_CHECKING:
    import stim


CircuitArtifact = Any
StimBuilder = Callable[[int, int, float], CircuitArtifact]


def _logical_error_rate(
    circuit_string: str,
    shots: int,
    seed: int | None,
    decoder: DecoderProtocol | None = None,
) -> float:
    """Estimate post-decoding logical-observable-0 error probability."""

    try:
        import stim
    except ModuleNotFoundError as exc:  # pragma: no cover - handled by tests
        raise ImportError("Stim is required to sample logical error rates.") from exc

    circuit = stim.Circuit(circuit_string)
    if circuit.num_observables == 0:
        raise ValueError("Circuit must define observable 0 to estimate logical error rate.")

    sampler = circuit.compile_detector_sampler(seed=seed)
    detector_samples, observable_samples = sampler.sample(shots, separate_observables=True)

    active_decoder = decoder or MWPMDecoder()
    dem = None
    if isinstance(active_decoder, MWPMDecoder) and find_spec("pymatching") is not None:
        dem = circuit.detector_error_model(decompose_errors=True)
    metadata = DecoderMetadata(
        num_observables=circuit.num_observables,
        detector_error_model=dem,
        circuit=circuit,
        seed=seed,
    )
    decoded = active_decoder.decode(detector_samples, metadata=metadata)

    # Post-decoding logical error is the residual mismatch between decoder
    # predictions and the sampled observable values.
    logical_predictions = np.asarray(decoded.logical_predictions)

    if logical_predictions.shape != observable_samples.shape:
        raise ValueError(
            f"Decoder returned logical_predictions with shape {logical_predictions.shape}, "
            f"but expected {observable_samples.shape} to match observable_samples."
        )

    if logical_predictions.dtype != observable_samples.dtype:
        logical_predictions = logical_predictions.astype(observable_samples.dtype, copy=False)

    logical_mismatch = np.logical_xor(logical_predictions, observable_samples)
    return float(np.mean(logical_mismatch[:, 0]))


def _coerce_circuit_string(circuit: CircuitArtifact) -> str:
    if isinstance(circuit, str):
        return circuit
    return str(circuit)


def _evaluate_policy_task(
    task: tuple[str, StimBuilder, int, int, float, int, int | None],
) -> tuple[str, Dict[str, float | int | str | None] | None, str | None]:
    name, builder, distance, rounds, p, shots, seed = task
    builder_name = getattr(builder, "__name__", builder.__class__.__name__)
    try:
        circuit_string = _coerce_circuit_string(builder(distance, rounds, p))
        result: Dict[str, float | int | str | None] = {
            "builder": builder_name,
            "distance": distance,
            "rounds": rounds,
            "p": p,
            "shots": shots,
            "seed": seed,
            "logical_error_rate": _logical_error_rate(circuit_string, shots, seed),
        }
        return name, result, None
    except ImportError as exc:
        return name, None, str(exc)


def _run_policy_tasks(
    tasks: list[tuple[str, StimBuilder, int, int, float, int, int | None]],
) -> list[tuple[str, Dict[str, float | int | str | None] | None, str | None]]:
    if len(tasks) <= 1:
        return [_evaluate_policy_task(task) for task in tasks]

    processes = min(len(tasks), max(1, mp.cpu_count() or 1))
    try:
        with mp.get_context("spawn").Pool(processes=processes) as pool:
            return pool.map(_evaluate_policy_task, tasks)
    except (AttributeError, OSError, RuntimeError, TypeError):
        # Custom builders defined in interactive sessions are often not
        # picklable. Preserve the public API by falling back to serial
        # execution instead of failing before any simulation runs.
        return [_evaluate_policy_task(task) for task in tasks]


def compare_nested_policies(
    *,
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    seed: int | None = None,
    static_builder: StimBuilder = surface_code_circuit_string,
    dynamic_builder: StimBuilder = hexagonal_surface_code,
) -> Dict[str, Dict[str, float | int | str | None]]:
    """Run small simulations for static and dynamic builders.

    Returns a dictionary keyed by policy name containing the logical error rate
    and the simulation metadata used to generate it. Independent policy
    builders are evaluated concurrently with `multiprocessing.Pool` when
    possible.
    """

    if not isinstance(distance, int):
        raise ValueError("distance must be an integer.")
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3.")

    if not isinstance(rounds, int) or rounds <= 0:
        raise ValueError("rounds must be a positive integer.")

    if not isinstance(shots, int) or shots <= 0:
        raise ValueError("shots must be a positive integer.")

    if not isinstance(p, float) or not 0.0 <= p <= 1.0:
        raise ValueError("p must be a float between 0 and 1 (inclusive).")

    if not callable(static_builder):
        raise ValueError("static_builder must be callable.")
    if not callable(dynamic_builder):
        raise ValueError("dynamic_builder must be callable.")
    if find_spec("stim") is None:
        raise ImportError("Stim is required to sample logical error rates.")

    policies: Dict[str, StimBuilder] = {
        "static": static_builder,
        "dynamic": dynamic_builder,
    }
    tasks = [(name, builder, distance, rounds, p, shots, seed) for name, builder in policies.items()]
    results: Dict[str, Dict[str, float | int | str | None]] = {}
    for name, metrics, error in _run_policy_tasks(tasks):
        if error is not None:
            raise ImportError(error)
        if metrics is None:
            raise RuntimeError(f"Policy '{name}' did not return metrics.")
        results[name] = metrics

    return results


def tabulate_comparison(comparison: Dict[str, Dict[str, float | int | str | None]]) -> Iterable[Dict[str, float | int | str | None]]:
    """Flatten a comparison dictionary into a list of rows."""

    for policy, metrics in comparison.items():
        yield {"policy": policy, **metrics}


def train_nested_agent(
    distance: int,
    rounds: int,
    p: float,
    shots: int,
    epochs: int = 10,
    seed: int | None = None
) -> NestedLearningAgent:
    """
    Train a NestedLearningAgent using actual Stim simulations for the inner loop.
    
    The inner loop trains the agent to predict logical errors (or corrections) from syndromes.
    The outer loop adapts the agent's memory/hyperparameters based on overall logical error rate.
    """
    import torch
    import stim
    
    # 1. Setup Environment (Circuit)
    # We use the dynamic builder (e.g., hexagonal) as the environment
    circuit_str = _coerce_circuit_string(hexagonal_surface_code(distance, rounds, p))
    circuit = stim.Circuit(circuit_str)
    
    # 2. Initialize Agent
    # State dim: Number of detectors (syndrome bits)
    # Action dim: 2 (Predict logical observable flip: 0 or 1)
    num_detectors = circuit.num_detectors
    state_dim = num_detectors
    action_dim = 2 
    
    agent = NestedLearningAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Sampler for generating experience
    sampler = circuit.compile_detector_sampler(seed=seed)
    
    for epoch in range(epochs):
        # --- Inner Loop: Train on Batches of Syndromes ---
        # Generate a batch of experience
        batch_size = shots
        detector_samples, observable_samples = sampler.sample(batch_size, separate_observables=True)
        
        # Convert to PyTorch tensors
        # detectors: (batch, num_detectors) -> Float for NN input
        states = torch.from_numpy(detector_samples).float()
        
        # observables: (batch, num_observables) -> We care about observable 0
        # actions: The "correct" action is the actual logical flip (0 or 1)
        # If the agent predicts this correctly, it "decodes" the logical error state.
        target_actions = torch.from_numpy(observable_samples[:, 0].astype(np.int64))
        
        # Rewards: +1 for correct prediction, -1 for incorrect
        # We calculate this inside inner_loop implicitly via CrossEntropy, 
        # but let's provide explicit rewards for the interface.
        # Here we just pass 1.0s because CrossEntropy handles the "supervised" signal.
        rewards = torch.ones(batch_size) 
        
        # Run inner loop update
        inner_loss = agent.inner_loop(states, target_actions, rewards)
        
        # --- Outer Loop: Evaluate & Adapt ---
        # Evaluate performance using the standard comparison tool
        # This runs a separate validation set (potentially parallelized)
        comparison = compare_nested_policies(
            distance=distance,
            rounds=rounds,
            p=p,
            shots=shots, # Validation shots
            seed=seed + epoch if seed else None # Vary seed for validation
        )
        
        # Get the logical error rate of the dynamic policy (baseline)
        # In a full RL setup, we'd use the agent's own performance, 
        # but here we use the environment's difficulty as a proxy for "outer" adaptation needs.
        dynamic_error_rate = comparison.get("dynamic", {}).get("logical_error_rate", 1.0)
        
        # Performance metric for outer loop: 
        # If error rate is high, we might want to update memory more aggressively.
        # Let's use (1 - error_rate) as "performance".
        performance_metric = 1.0 - float(dynamic_error_rate)
        
        outer_loss = agent.outer_loop(states, performance_metric)
        
        print(f"Epoch {epoch+1}/{epochs} | Inner Loss: {inner_loss:.4f} | Outer Loss: {outer_loss:.4f} | Val Error Rate: {dynamic_error_rate:.4f}")
        
    return agent
