"""Minimum-weight perfect matching decoder adapter.

The adapter prefers PyMatching when available and otherwise falls back to a
simple deterministic all-zero predictor for environments without optional
matching dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import BoolArray, DecoderMetadata, DecoderOutput, DecoderProtocol


@dataclass
class MWPMDecoder(DecoderProtocol):
    """MWPM decoder with a PyMatching-compatible detector-error-model path."""

    name: str = "mwpm"

    def _decode_with_pymatching(self, detector_events: BoolArray, metadata: DecoderMetadata) -> BoolArray:
        """Decode using `pymatching` if the package is installed."""

        try:
            from pymatching import Matching
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pymatching is not installed") from exc

        if metadata.detector_error_model is None:
            raise ValueError("MWPMDecoder requires detector_error_model in metadata for pymatching path.")

        matching = Matching.from_detector_error_model(metadata.detector_error_model)
        predictions = matching.decode_batch(detector_events)
        return np.asarray(predictions, dtype=np.bool_)

    def _fallback_decode(self, detector_events: BoolArray, metadata: DecoderMetadata) -> BoolArray:
        """Deterministic fallback path when PyMatching is unavailable."""

        shots = int(detector_events.shape[0])
        return np.zeros((shots, metadata.num_observables), dtype=np.bool_)

    def decode(self, detector_events: BoolArray, metadata: DecoderMetadata) -> DecoderOutput:
        events = np.asarray(detector_events, dtype=np.bool_)
        if events.ndim != 2:
            raise ValueError("detector_events must be a 2D bool array of shape (shots, num_detectors).")
        if metadata.num_observables <= 0:
            raise ValueError("metadata.num_observables must be > 0")

        backend: str
        try:
            logicals = self._decode_with_pymatching(events, metadata)
            backend = "pymatching"
        except ImportError:
            # Fallback is only used when the optional pymatching dependency is unavailable.
            logicals = self._fallback_decode(events, metadata)
            backend = "fallback"

        if logicals.shape != (events.shape[0], metadata.num_observables):
            raise ValueError("Decoder returned predictions with an unexpected shape.")

        return DecoderOutput(logical_predictions=logicals, decoder_name=self.name, diagnostics={"backend": backend})
