"""Confidence-aware decoding utilities.

This module introduces a small data container for syndrome batches and a
weighted-MWPM decoder that can consume optional per-detector confidence values.
When confidence data is omitted, decoding behavior falls back to standard
hard-decision MWPM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class SyndromeBatch:
    """Container for batched detector outcomes.

    Attributes:
        hard_bits: Binary detector outcomes with shape ``(shots, num_detectors)``.
        confidence: Optional confidence values in ``[0, 1]`` with the same shape
            as ``hard_bits``. Larger values indicate higher trust in the
            corresponding hard detector outcome.
    """

    hard_bits: np.ndarray
    confidence: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        hard_bits = np.asarray(self.hard_bits, dtype=np.uint8)
        if hard_bits.ndim != 2:
            raise ValueError("hard_bits must be a 2D array with shape (shots, num_detectors).")
        if np.any((hard_bits != 0) & (hard_bits != 1)):
            raise ValueError("hard_bits must contain only 0/1 values.")
        object.__setattr__(self, "hard_bits", hard_bits)

        if self.confidence is None:
            return

        confidence = np.asarray(self.confidence, dtype=np.float64)
        if confidence.shape != hard_bits.shape:
            raise ValueError("confidence must have the same shape as hard_bits.")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("confidence entries must lie in [0, 1].")
        object.__setattr__(self, "confidence", confidence)


class SyndromeDecoder(Protocol):
    """Decoder interface accepting hard syndromes with optional soft features."""

    def decode_batch(self, syndromes: SyndromeBatch) -> np.ndarray:
        """Decode a batch and return logical predictions per shot."""


class WeightedMWPMDecoder:
    """Weighted MWPM decoder with per-shot confidence reweighting.

    The decoder starts from a base matching graph extracted from a
    detector-error model. If confidence is available, edge weights are adjusted
    by endpoint confidence before running MWPM:

    ``w' = w * (1 + confidence_scale * (1 - mean_endpoint_confidence))``

    so low-confidence measurements get larger effective costs.

    ``decode_batch`` returns an array of shape ``(shots, num_fault_ids)``
    containing the predicted observable flips for every shot.
    """

    def __init__(self, detector_error_model: object, *, confidence_scale: float = 1.0) -> None:
        if confidence_scale < 0:
            raise ValueError("confidence_scale must be non-negative.")

        try:
            import pymatching
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise ImportError("pymatching is required for WeightedMWPMDecoder.") from exc

        self._pymatching = pymatching
        self._confidence_scale = float(confidence_scale)
        self._base_matching = pymatching.Matching.from_detector_error_model(detector_error_model)

        # Number of detector nodes; the boundary placeholder index equals this value.
        self._num_detectors: int = self._base_matching.num_detectors

        # Determine the number of observable fault IDs from the base matching.
        _zero_syndrome = np.zeros(self._num_detectors, dtype=np.uint8)
        self._num_fault_ids: int = len(self._base_matching.decode(_zero_syndrome))

        # --- Precompute edge data for vectorized per-shot weight adjustment ---
        # The base graph is retained for per-shot copies; this preserves all
        # detector nodes (including any that have no edges in the MWPM graph).
        # Edge endpoints and base weights are extracted once so the hot-path
        # confidence computation can be fully vectorised with NumPy.
        self._base_graph = self._base_matching.to_networkx()
        boundary = self._num_detectors

        edges_u: list[int] = []
        edges_v: list[int] = []
        base_weights: list[float] = []

        for u, v, data in self._base_graph.edges(data=True):
            edges_u.append(int(u))
            edges_v.append(int(v))
            base_weights.append(float(data.get("weight", 1.0)))

        # np.intp matches NumPy's native index type, avoiding implicit casts
        # during fancy-indexing into shot_confidence arrays.
        self._edge_u = np.array(edges_u, dtype=np.intp)
        self._edge_v = np.array(edges_v, dtype=np.intp)
        self._base_weights = np.array(base_weights, dtype=np.float64)
        # Ordered (u, v) pairs matching the arrays above, for O(E) weight updates.
        self._edge_pairs: list[tuple[int, int]] = list(zip(edges_u, edges_v))

        # Boolean masks: True when the endpoint is a real detector node.
        self._u_is_det: np.ndarray = self._edge_u != boundary
        self._v_is_det: np.ndarray = self._edge_v != boundary

        # Safe index arrays: replace the out-of-range boundary placeholder with
        # index 0 so numpy fancy-indexing never goes out of bounds; the boolean
        # masks ensure those slots contribute nothing to the confidence mean.
        self._edge_u_safe = np.where(self._u_is_det, self._edge_u, 0)
        self._edge_v_safe = np.where(self._v_is_det, self._edge_v, 0)

    def _compute_adjusted_weights(self, shot_confidence: np.ndarray) -> np.ndarray:
        """Return edge weights adjusted by per-detector confidence (vectorized).

        All per-edge arithmetic is done with NumPy so no Python-level loop over
        edges is needed.
        """
        conf_u = np.where(self._u_is_det, shot_confidence[self._edge_u_safe], 1.0)
        conf_v = np.where(self._v_is_det, shot_confidence[self._edge_v_safe], 1.0)

        num_real_endpoints = self._u_is_det.astype(np.float64) + self._v_is_det.astype(np.float64)
        mean_conf = np.where(
            num_real_endpoints > 0,
            (conf_u * self._u_is_det + conf_v * self._v_is_det) / np.maximum(num_real_endpoints, 1.0),
            1.0,
        )
        scale = 1.0 + self._confidence_scale * (1.0 - mean_conf)
        return self._base_weights * scale

    def decode_batch(self, syndromes: SyndromeBatch) -> np.ndarray:
        """Decode each shot and return predictions shaped ``(shots, num_fault_ids)``.

        When ``syndromes.confidence`` is ``None`` the base matching (without
        any weight adjustments) is used for every shot, which is equivalent to
        passing an all-ones confidence array.
        """
        hard = syndromes.hard_bits
        confidence = syndromes.confidence
        shots = hard.shape[0]
        predictions = np.zeros((shots, self._num_fault_ids), dtype=np.uint8)

        if confidence is None:
            for shot_ix, shot in enumerate(hard):
                predictions[shot_ix] = self._base_matching.decode(shot)
            return predictions

        for shot_ix, (shot, shot_confidence) in enumerate(zip(hard, confidence, strict=True)):
            adjusted_weights = self._compute_adjusted_weights(shot_confidence)
            # Copy the base graph and apply precomputed weights.  Copying rather
            # than building from scratch preserves isolated detector nodes (those
            # with no edges in the MWPM graph) so that syndrome arrays of full
            # length num_detectors are accepted by pymatching.
            graph = self._base_graph.copy()
            for i, (u, v) in enumerate(self._edge_pairs):
                graph[u][v]["weight"] = float(adjusted_weights[i])
            matching = self._pymatching.Matching(graph)
            predictions[shot_ix] = matching.decode(shot)

        return predictions
