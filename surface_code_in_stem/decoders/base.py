"""Typed interfaces and containers for detector-event decoders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class DecoderMetadata:
    """Metadata needed by decoders to map detector events to logical predictions."""

    num_observables: int
    detector_error_model: Any | None = None
    circuit: Any | None = None
    seed: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecoderInput:
    """Container for batched detector events and accompanying metadata."""

    detector_events: BoolArray
    metadata: DecoderMetadata


@dataclass(frozen=True)
class DecoderOutput:
    """Container for batched predicted logical observables."""

    logical_predictions: BoolArray
    decoder_name: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DecoderProtocol(Protocol):
    """Common protocol implemented by all decoder adapters."""

    name: str

    def decode(self, detector_events: BoolArray, metadata: DecoderMetadata) -> DecoderOutput:
        """Decode detector events into predicted logical observables."""
        ...
