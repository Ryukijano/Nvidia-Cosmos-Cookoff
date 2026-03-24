"""Noise model abstractions for Stim circuit builders.

The models emit Stim instruction strings for gate, measurement, and reset noise.
Builders can optionally request extra correlated-event instructions per layer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
from typing import Iterable, List, Optional, Sequence


class NoiseModel(ABC):
    """Interface for noise-channel emission during circuit construction."""

    @abstractmethod
    def gate_noise(
        self,
        *,
        gate: str,
        pair_targets: Sequence[int],
        idle_targets: Sequence[int],
        layer_id: str,
    ) -> List[str]:
        """Return Stim instructions for a two-qubit gate layer."""

    @abstractmethod
    def measurement_noise(self, *, qubits: Sequence[int], layer_id: str) -> List[str]:
        """Return Stim instructions before a measurement operation."""

    @abstractmethod
    def reset_noise(self, *, qubits: Sequence[int], layer_id: str) -> List[str]:
        """Return Stim instructions immediately after reset operations."""

    def correlated_event_instructions(self, *, qubits: Sequence[int], layer_id: str) -> List[str]:
        """Optional extra non-local/correlated noise instructions."""
        return []


@dataclass
class IIDDepolarizingNoiseModel(NoiseModel):
    """Current baseline: IID depolarizing and IID X-flip reset/measurement noise."""

    p: float

    def gate_noise(self, *, gate: str, pair_targets: Sequence[int], idle_targets: Sequence[int], layer_id: str) -> List[str]:
        lines: List[str] = []
        if pair_targets:
            lines.append(f"DEPOLARIZE2({self.p}) {' '.join(map(str, pair_targets))}")
        if idle_targets:
            lines.append(f"DEPOLARIZE1({self.p}) {' '.join(map(str, idle_targets))}")
        lines.extend(self.correlated_event_instructions(qubits=list(pair_targets) + list(idle_targets), layer_id=layer_id))
        return lines

    def measurement_noise(self, *, qubits: Sequence[int], layer_id: str) -> List[str]:
        if not qubits:
            return []
        return [f"X_ERROR({self.p}) {' '.join(map(str, qubits))}"]

    def reset_noise(self, *, qubits: Sequence[int], layer_id: str) -> List[str]:
        if not qubits:
            return []
        return [f"X_ERROR({self.p}) {' '.join(map(str, qubits))}"]


@dataclass
class BiasedNoiseModel(NoiseModel):
    """Single-qubit biased channel plus depolarizing entangling noise.

    Useful for cat-qubit-like asymmetry where bit- and phase-flip rates differ.
    """

    gate_p: float = 0.0
    bit_flip_p: float = 0.0
    phase_flip_p: float = 0.0
    
    # Optional parameters for the new interface
    p_x: float = 0.0
    p_y: float = 0.0
    p_z: float = 0.0
    biased_pauli: str = "Z"
    bias_ratio: float = 10.0

    def __post_init__(self):
        # Support both old and new initialization signatures
        if self.p_x > 0 or self.p_z > 0:
            self.bit_flip_p = self.p_x
            self.phase_flip_p = self.p_z
            # Use average for gate_p if not explicitly provided
            if self.gate_p == 0.0:
                self.gate_p = self.p_x + self.p_y + self.p_z

    def gate_noise(self, *, gate: str, pair_targets: Sequence[int], idle_targets: Sequence[int], layer_id: str) -> List[str]:
        lines: List[str] = []
        if pair_targets:
            lines.append(f"DEPOLARIZE2({self.gate_p}) {' '.join(map(str, pair_targets))}")
        if idle_targets:
            lines.append(self._biased_channel(idle_targets))
        lines.extend(self.correlated_event_instructions(qubits=list(pair_targets) + list(idle_targets), layer_id=layer_id))
        return lines

    def measurement_noise(self, *, qubits: Sequence[int], layer_id: str) -> List[str]:
        return [self._biased_channel(qubits)] if qubits else []

    def reset_noise(self, *, qubits: Sequence[int], layer_id: str) -> List[str]:
        return [self._biased_channel(qubits)] if qubits else []

    def _biased_channel(self, qubits: Sequence[int]) -> str:
        return f"PAULI_CHANNEL_1({self.bit_flip_p},0,{self.phase_flip_p}) {' '.join(map(str, qubits))}"


@dataclass
class ErasureAwareNoiseModel(IIDDepolarizingNoiseModel):
    """IID depolarizing noise augmented with heralded erasure side information."""

    erasure_p: float = 0.0

    def correlated_event_instructions(self, *, qubits: Sequence[int], layer_id: str) -> List[str]:
        if not qubits or self.erasure_p <= 0:
            return []
        return [f"HERALDED_ERASE({self.erasure_p}) {' '.join(map(str, qubits))}"]


@dataclass
class CorrelatedBurstNoiseModel(IIDDepolarizingNoiseModel):
    """Adds reproducible spacetime burst events via correlated errors."""

    burst_probability: float = 0.0
    max_cluster_size: int = 4
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def correlated_event_instructions(self, *, qubits: Sequence[int], layer_id: str) -> List[str]:
        if not qubits or self.burst_probability <= 0:
            return []
        if self._rng.random() >= self.burst_probability:
            return []
        cluster_size = min(len(qubits), max(2, self.max_cluster_size))
        cluster = self._rng.sample(list(qubits), k=cluster_size)
        terms = " ".join(f"X{q}" for q in cluster)
        return [f"CORRELATED_ERROR({self.p}) {terms}"]


def resolve_noise_model(p: float, noise_model: Optional[NoiseModel]) -> NoiseModel:
    """Map legacy scalar `p` usage to the IID model when needed."""

    if noise_model is not None:
        return noise_model
    return IIDDepolarizingNoiseModel(p=p)
