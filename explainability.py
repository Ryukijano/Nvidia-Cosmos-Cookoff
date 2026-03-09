from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class ExplainabilitySpec:
    encoder_mode: str
    encoder_label: str
    decoder_mode: str
    decoder_label: str
    encoder_layer_count: int = 0
    encoder_head_count: int = 0


class ModuleOutputRecorder:
    def __init__(self) -> None:
        self.handle = None
        self.output = None

    def attach(self, module) -> None:
        self.remove()
        self.handle = module.register_forward_hook(self._hook)

    def clear(self) -> None:
        self.output = None

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.output = None

    def _hook(self, module, inputs, output) -> None:  # pragma: no cover - hook signature
        if torch.is_tensor(output):
            self.output = output.detach()
        else:
            self.output = output


def clamp_index(index: int | None, upper_bound: int) -> int:
    if upper_bound <= 0:
        return 0
    if index is None:
        return upper_bound - 1
    return max(0, min(int(index), upper_bound - 1))


def normalize_map(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}")

    array = array.copy()
    min_value = float(array.min(initial=0.0))
    array -= min_value
    max_value = float(array.max(initial=0.0))
    if max_value > 0:
        array /= max_value
    return array


def resize_rgb_image(rgb_image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_LINEAR)


def feature_energy_map(feature_tensor: torch.Tensor, output_shape: tuple[int, int]) -> np.ndarray:
    tensor = feature_tensor.detach().float()
    while tensor.dim() > 3:
        tensor = tensor[0]
    if tensor.dim() == 3:
        tensor = tensor.abs().mean(dim=0)
    elif tensor.dim() != 2:
        raise ValueError(f"Unexpected feature tensor shape: {tuple(feature_tensor.shape)}")

    heatmap = normalize_map(tensor.cpu().numpy())
    height, width = output_shape
    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)


def render_heatmap_overlay(rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if heatmap.shape != rgb_image.shape[:2]:
        heatmap = cv2.resize(heatmap, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap((normalize_map(heatmap) * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_image, 1.0 - alpha, colored, alpha, 0.0)


def render_temporal_strip(values, *, active_index: int | None = None, cell_width: int = 12, height: int = 72) -> np.ndarray:
    sequence = np.asarray(values, dtype=np.float32).reshape(1, -1)
    if sequence.size == 0:
        sequence = np.zeros((1, 1), dtype=np.float32)

    normalized = normalize_map(sequence)
    strip = (normalized * 255.0).astype(np.uint8)
    strip = np.repeat(strip, height, axis=0)
    strip = np.repeat(strip, cell_width, axis=1)
    colored = cv2.applyColorMap(strip, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    if active_index is not None and sequence.shape[1] > 0:
        clamped = clamp_index(active_index, sequence.shape[1])
        x0 = clamped * cell_width
        x1 = min(colored.shape[1] - 1, x0 + cell_width - 1)
        cv2.rectangle(colored, (x0, 0), (x1, colored.shape[0] - 1), (255, 255, 255), 2)

    return colored
