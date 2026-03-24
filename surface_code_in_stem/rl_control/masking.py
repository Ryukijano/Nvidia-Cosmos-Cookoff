"""Factor-graph masking utilities for local detector-parameter neighborhoods."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def build_detector_parameter_mask(
    num_detectors: int,
    num_parameters: int,
    edges: Iterable[tuple[int, int]],
) -> np.ndarray:
    """Build a binary detector-parameter adjacency mask.

    Args:
        num_detectors: Number of detector nodes.
        num_parameters: Number of control-parameter nodes.
        edges: Iterable of (detector_idx, parameter_idx).
    """
    mask = np.zeros((num_detectors, num_parameters), dtype=np.float64)
    for det_idx, param_idx in edges:
        if not (0 <= det_idx < num_detectors):
            raise ValueError("detector index out of range")
        if not (0 <= param_idx < num_parameters):
            raise ValueError("parameter index out of range")
        mask[det_idx, param_idx] = 1.0
    return mask


def parameter_neighborhoods(mask: np.ndarray) -> dict[int, np.ndarray]:
    """Return detector neighborhoods for each parameter."""
    mask = np.asarray(mask, dtype=np.float64)
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    neighborhoods = defaultdict(list)
    for det_idx, param_idx in np.argwhere(mask > 0):
        neighborhoods[int(param_idx)].append(int(det_idx))
    return {k: np.asarray(v, dtype=np.int64) for k, v in neighborhoods.items()}


def apply_masked_detector_weights(
    detector_signal: np.ndarray,
    mask: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Aggregate detector signals into parameter-wise masked signals."""
    detector_signal = np.asarray(detector_signal, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if detector_signal.ndim != 1:
        raise ValueError("detector_signal must be a vector")
    if mask.ndim != 2:
        raise ValueError("mask must be a matrix")
    if mask.shape[0] != detector_signal.shape[0]:
        raise ValueError("mask rows must match detector_signal length")

    weighted = mask.T @ detector_signal
    if not normalize:
        return weighted
    degree = np.maximum(np.sum(mask, axis=0), 1.0)
    return weighted / degree


def mask_population_perturbations(
    perturbations: np.ndarray,
    detector_signal: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Variance-reducing mask: gate perturbation columns by local detector activity."""
    perturbations = np.asarray(perturbations, dtype=np.float64)
    param_weights = apply_masked_detector_weights(detector_signal, mask, normalize=True)
    return perturbations * param_weights[None, :]
