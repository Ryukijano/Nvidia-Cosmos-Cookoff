from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch

from model_registry import ensure_model_artifacts
from predictor import MODEL_LABELS, create_predictor, normalize_model_key


@dataclass
class ModelManagerStatus:
    active_model_key: str | None
    active_model_label: str | None
    is_loaded: bool
    last_error: str | None


class SpaceModelManager:
    def __init__(self) -> None:
        self.current_model_key: str | None = None
        self.current_predictor: Any | None = None
        self.last_error: str | None = None

    def unload_model(self) -> None:
        if self.current_predictor is not None:
            if hasattr(self.current_predictor, "unload"):
                self.current_predictor.unload()
            self.current_predictor = None

        self.current_model_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_predictor(self, model_key: str):
        normalized_key = normalize_model_key(model_key)
        if self.current_model_key == normalized_key and self.current_predictor is not None:
            return self.current_predictor

        self.unload_model()
        self.last_error = None

        try:
            model_dir = ensure_model_artifacts(normalized_key)
            predictor = create_predictor(normalized_key, model_dir=str(model_dir))
            predictor.warm_up()
        except Exception as exc:
            self.unload_model()
            self.last_error = str(exc)
            raise

        self.current_model_key = normalized_key
        self.current_predictor = predictor
        return predictor

    def reset_predictor_state(self) -> None:
        if self.current_predictor is not None and hasattr(self.current_predictor, "reset_state"):
            self.current_predictor.reset_state()

    def status(self) -> ModelManagerStatus:
        active_model_label = None
        if self.current_model_key is not None:
            active_model_label = MODEL_LABELS[self.current_model_key]

        return ModelManagerStatus(
            active_model_key=self.current_model_key,
            active_model_label=active_model_label,
            is_loaded=self.current_predictor is not None,
            last_error=self.last_error,
        )
