from __future__ import annotations

import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import autocast
    MIXED_PRECISION_AVAILABLE = True
except ImportError:  # pragma: no cover
    MIXED_PRECISION_AVAILABLE = False

from model_registry import default_model_root
from model.resnet import ResNet
from model.mstcn import MultiStageModel
from model.transformer import Transformer
from explainability import (
    ExplainabilitySpec,
    ModuleOutputRecorder,
    clamp_index,
    feature_energy_map,
    render_heatmap_overlay,
    render_temporal_strip,
    resize_rgb_image,
)

PHASE_LABELS = ("idle", "marking", "injection", "dissection")
MODEL_LABELS = {
    "aiendo": "AI-Endo",
    "dinov2": "DINO-Endo",
    "vjepa2": "V-JEPA2",
}


def _app_root() -> Path:
    return Path(__file__).resolve().parent


def default_model_dir() -> str:
    return str(default_model_root())


def normalize_model_key(name: str | None) -> str:
    token = (name or "aiendo").lower().replace("-", "").replace("_", "").strip()
    if token in ("aiendo", "resnet", "aiendoresnet", "aiendoresnetmstcn", "aiendoresnetmstcntransformer"):
        return "aiendo"
    if token in ("dinov2", "dinov2endo", "dinoendo", "dino"):
        return "dinov2"
    if token in ("vjepa2", "vjepa", "vjepa2endo"):
        return "vjepa2"
    raise KeyError(f"Unsupported model key: {name}")


def _load_trusted_checkpoint(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover
        return torch.load(path, map_location=map_location)


def _strip_state_dict_prefixes(state_dict, prefixes):
    cleaned_state = {}
    for key, value in state_dict.items():
        while any(key.startswith(prefix) for prefix in prefixes):
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
        cleaned_state[key] = value
    return cleaned_state


def _validate_load_result(
    load_result,
    model_name: str,
    *,
    allowed_missing=(),
    allowed_missing_prefixes=(),
    allowed_unexpected=(),
    allowed_unexpected_prefixes=(),
):
    missing = [
        key
        for key in load_result.missing_keys
        if key not in allowed_missing and not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    unexpected = [
        key
        for key in load_result.unexpected_keys
        if key not in allowed_unexpected and not any(key.startswith(prefix) for prefix in allowed_unexpected_prefixes)
    ]
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing={missing[:10]}")
        if unexpected:
            problems.append(f"unexpected={unexpected[:10]}")
        raise RuntimeError(f"{model_name} checkpoint mismatch ({'; '.join(problems)})")


def _resolve_vendor_repo(repo_name: str, extra_candidates=()):
    app_root = _app_root()
    candidates = [app_root / repo_name]
    if len(app_root.parents) >= 2:
        candidates.append(app_root.parents[1] / repo_name)
    candidates.extend(extra_candidates)

    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(f"Required vendor repo '{repo_name}' not found. Stage it into this folder or keep the repo-root copy available.")


def _build_explainability_payload(
    *,
    display_image: np.ndarray,
    encoder_heatmap: np.ndarray,
    encoder_kind: str,
    encoder_label: str,
    decoder_values,
    decoder_kind: str,
    decoder_label: str,
    active_decoder_index: int | None = None,
    encoder_layer: int | None = None,
    encoder_head: int | None = None,
    notes: str | None = None,
) -> dict:
    payload = {
        "encoder_kind": encoder_kind,
        "encoder_label": encoder_label,
        "encoder_visualization": render_heatmap_overlay(display_image, encoder_heatmap),
        "decoder_kind": decoder_kind,
        "decoder_label": decoder_label,
        "decoder_visualization": render_temporal_strip(decoder_values, active_index=active_decoder_index),
    }
    if encoder_layer is not None:
        payload["encoder_layer"] = int(encoder_layer)
    if encoder_head is not None:
        payload["encoder_head"] = int(encoder_head)
    if notes:
        payload["notes"] = notes
    return payload


class Predictor:
    def __init__(self, model_dir: str | None = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir or default_model_dir()
        self.seq_length = 1024
        self.trans_seq = 30
        self.aug = A.Compose([A.Resize(height=224, width=224), A.Normalize()])
        self.frame_feature_cache = None
        self.label_dict = dict(enumerate(PHASE_LABELS))
        self.available = False
        self._resnet_activation = None
        self._resnet_activation_hook = None
        self._explainability_spec = ExplainabilitySpec(
            encoder_mode="proxy",
            encoder_label="ResNet layer4 activation energy (proxy)",
            decoder_mode="attention",
            decoder_label="Temporal Transformer attention",
        )

        self._norm_mean = None
        self._norm_std = None
        if self.device.type == "cuda":
            self._norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self._norm_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self._load_models(self.model_dir)

    def _load_models(self, model_dir: str):
        self.resnet = ResNet(out_channels=4, has_fc=False)
        paras = torch.load(os.path.join(model_dir, "resnet50.pth"), map_location=self.device)["model"]
        paras = {k: v for k, v in paras.items() if "fc" not in k and "embed" not in k}
        paras = {k.replace("share.", "resnet."): v for k, v in paras.items()}
        self.resnet.load_state_dict(paras, strict=True)
        self.resnet.to(self.device).eval()
        self._resnet_activation_hook = self.resnet.resnet.layer4[-1].relu.register_forward_hook(
            self._capture_resnet_activation
        )

        self.fusion = MultiStageModel(
            mstcn_stages=2,
            mstcn_layers=8,
            mstcn_f_maps=32,
            mstcn_f_dim=2048,
            out_features=4,
            mstcn_causal_conv=True,
            is_train=False,
        )
        fusion_weights = torch.load(os.path.join(model_dir, "fusion.pth"), map_location=self.device)
        fusion_load = self.fusion.load_state_dict(fusion_weights, strict=False)
        _validate_load_result(
            fusion_load,
            "AI-Endo fusion",
            allowed_unexpected_prefixes=("stage1.conv_out_classes.",),
        )
        self.fusion.to(self.device).eval()

        self.transformer = Transformer(32, 2048, 4, 30, d_model=32)
        trans_weights = torch.load(os.path.join(model_dir, "transformer.pth"), map_location=self.device)
        self.transformer.load_state_dict(trans_weights)
        self.transformer.to(self.device).eval()
        self.available = True

    def _amp_context(self):
        return autocast("cuda") if MIXED_PRECISION_AVAILABLE and self.device.type == "cuda" else nullcontext()

    def _preprocess_gpu(self, rgb_image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32, non_blocking=True).div_(255.0)
        if tensor.shape[-2:] != (224, 224):
            tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
        return (tensor - self._norm_mean) / self._norm_std

    def warm_up(self):
        dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self.predict(dummy)
        self.reset_state()

    def _capture_resnet_activation(self, module, inputs, output):  # pragma: no cover - hook signature
        self._resnet_activation = output.detach()

    def reset_state(self):
        self.frame_feature_cache = None
        self._resnet_activation = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_explainability_spec(self) -> ExplainabilitySpec:
        return self._explainability_spec

    def unload(self):
        self.available = False
        self.resnet.to("cpu")
        self.fusion.to("cpu")
        self.transformer.to("cpu")
        self.resnet = None
        self.fusion = None
        self.transformer = None
        self.frame_feature_cache = None
        self._resnet_activation = None
        if self._resnet_activation_hook is not None:
            self._resnet_activation_hook.remove()
            self._resnet_activation_hook = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cache_features(self, feature: torch.Tensor):
        if self.frame_feature_cache is None:
            self.frame_feature_cache = feature
        elif self.frame_feature_cache.shape[0] > self.seq_length:
            self.frame_feature_cache = torch.cat([self.frame_feature_cache[1:], feature], dim=0)
        else:
            self.frame_feature_cache = torch.cat([self.frame_feature_cache, feature], dim=0)

    @torch.inference_mode()
    def predict(self, rgb_image: np.ndarray, explainability: dict | None = None):
        explain_enabled = bool(explainability and explainability.get("enabled"))
        attention_meta = None
        display_image = resize_rgb_image(rgb_image, (224, 224)) if explain_enabled else None
        if self._norm_mean is not None:
            tensor = self._preprocess_gpu(rgb_image)
        else:
            processed = self.aug(image=rgb_image)["image"]
            chw = np.transpose(processed, (2, 0, 1))
            tensor = torch.from_numpy(chw).unsqueeze(0).contiguous().to(self.device)

        with self._amp_context():
            feature = self.resnet(tensor).clone()
            self._cache_features(feature)

            if self.frame_feature_cache is None:
                single_frame_feature = feature.unsqueeze(1)
                temporal_input = single_frame_feature.transpose(1, 2)
                temporal_feature = self.fusion(temporal_input)
                transformer_outputs = self.transformer(
                    temporal_feature.detach(),
                    single_frame_feature,
                    return_attention=explain_enabled,
                )
                if explain_enabled:
                    outputs, attention_meta = transformer_outputs
                else:
                    outputs = transformer_outputs
                final_logits = outputs[-1, -1, :]
                probs = F.softmax(final_logits.float(), dim=-1)
                pred_np = probs.detach().cpu().numpy()
                confidence = float(np.max(pred_np))
                phase_idx = max(0, min(3, int(np.argmax(pred_np))))
                phase = self.label_dict.get(phase_idx, "idle")
                frames_used = 1
                result = {"phase": phase, "probs": pred_np.tolist(), "confidence": confidence, "frames_used": frames_used}
                if explain_enabled and attention_meta is not None and display_image is not None and self._resnet_activation is not None:
                    encoder_heatmap = feature_energy_map(self._resnet_activation, display_image.shape[:2])
                    result["explainability"] = _build_explainability_payload(
                        display_image=display_image,
                        encoder_heatmap=encoder_heatmap,
                        encoder_kind="proxy",
                        encoder_label=self._explainability_spec.encoder_label,
                        decoder_values=attention_meta["decoder_strip"].detach().cpu().numpy(),
                        decoder_kind="attention",
                        decoder_label=self._explainability_spec.decoder_label,
                        active_decoder_index=frames_used - 1,
                        notes="Encoder view is a proxy activation map because the ResNet backbone is not attention-based.",
                    )
                return result

            if self.frame_feature_cache.shape[0] < 30:
                available_frames = self.frame_feature_cache.shape[0] + 1
                cat_frame_feature = torch.cat([self.frame_feature_cache, feature], dim=0).unsqueeze(0)
                temporal_input = cat_frame_feature.transpose(1, 2)
                temporal_feature = self.fusion(temporal_input)
                transformer_outputs = self.transformer(
                    temporal_feature.detach(),
                    cat_frame_feature,
                    return_attention=explain_enabled,
                )
                if explain_enabled:
                    outputs, attention_meta = transformer_outputs
                else:
                    outputs = transformer_outputs
                final_logits = outputs[-1, -1, :]
                probs = F.softmax(final_logits.float(), dim=-1)
                pred_np = probs.detach().cpu().numpy()
                confidence = float(np.max(pred_np))
                phase_idx = max(0, min(3, int(np.argmax(pred_np))))
                phase = self.label_dict.get(phase_idx, "idle")
                result = {
                    "phase": phase,
                    "probs": pred_np.tolist(),
                    "confidence": confidence,
                    "frames_used": available_frames,
                }
                if explain_enabled and attention_meta is not None and display_image is not None and self._resnet_activation is not None:
                    encoder_heatmap = feature_energy_map(self._resnet_activation, display_image.shape[:2])
                    result["explainability"] = _build_explainability_payload(
                        display_image=display_image,
                        encoder_heatmap=encoder_heatmap,
                        encoder_kind="proxy",
                        encoder_label=self._explainability_spec.encoder_label,
                        decoder_values=attention_meta["decoder_strip"].detach().cpu().numpy(),
                        decoder_kind="attention",
                        decoder_label=self._explainability_spec.decoder_label,
                        active_decoder_index=available_frames - 1,
                        notes="Encoder view is a proxy activation map because the ResNet backbone is not attention-based.",
                    )
                return result

            cat_frame_feature = self.frame_feature_cache.unsqueeze(0)
            temporal_input = cat_frame_feature.transpose(1, 2)
            temporal_feature = self.fusion(temporal_input)
            transformer_outputs = self.transformer(
                temporal_feature.detach(),
                cat_frame_feature,
                return_attention=explain_enabled,
            )
            if explain_enabled:
                outputs, attention_meta = transformer_outputs
            else:
                outputs = transformer_outputs
            final_logits = outputs[-1, -1, :]
            probs = F.softmax(final_logits.float(), dim=-1)
            pred_np = probs.detach().cpu().numpy()

        confidence = float(np.max(pred_np))
        phase_idx = max(0, min(3, int(np.argmax(pred_np))))
        phase = self.label_dict.get(phase_idx, "idle")
        frames_used = min(self.trans_seq, self.frame_feature_cache.shape[0])
        result = {"phase": phase, "probs": pred_np.tolist(), "confidence": confidence, "frames_used": frames_used}
        if explain_enabled and attention_meta is not None and display_image is not None and self._resnet_activation is not None:
            encoder_heatmap = feature_energy_map(self._resnet_activation, display_image.shape[:2])
            result["explainability"] = _build_explainability_payload(
                display_image=display_image,
                encoder_heatmap=encoder_heatmap,
                encoder_kind="proxy",
                encoder_label=self._explainability_spec.encoder_label,
                decoder_values=attention_meta["decoder_strip"].detach().cpu().numpy(),
                decoder_kind="attention",
                decoder_label=self._explainability_spec.decoder_label,
                active_decoder_index=frames_used - 1,
                notes="Encoder view is a proxy activation map because the ResNet backbone is not attention-based.",
            )
        return result


class PredictorDinoV2:
    def __init__(self, model_dir: str | None = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir or default_model_dir()
        self.seq_length = 30
        self.available = False
        self.backbone = None
        self.decoder = None
        self.label_dict = dict(enumerate(PHASE_LABELS))
        self.aug = A.Compose([
            A.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_LINEAR),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ])
        self.display_aug = A.Compose([
            A.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_LINEAR),
            A.CenterCrop(height=224, width=224),
        ])
        self.frame_features = []
        self._attention_recorder = ModuleOutputRecorder()
        self._attention_layer_index = None
        self._explainability_spec = ExplainabilitySpec(
            encoder_mode="attention",
            encoder_label="DINOv2 encoder self-attention",
            decoder_mode="attention",
            decoder_label="Fusion Transformer temporal attention",
        )
        self._load_models(self.model_dir)

    def _amp_context(self):
        return autocast("cuda") if MIXED_PRECISION_AVAILABLE and self.device.type == "cuda" else nullcontext()

    def _resolve_local_dino_repo(self):
        candidates = [_app_root() / "dinov2"]
        app_root = _app_root()
        if len(app_root.parents) >= 2:
            candidates.append(app_root.parents[1] / "dinov2")
        candidates.append(Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main")
        for candidate in candidates:
            if (candidate / "hubconf.py").is_file():
                return str(candidate)
        raise FileNotFoundError("Local DINOv2 repo not found. Stage dinov2/ into this folder or keep the repo-root copy available.")

    def _load_models(self, model_dir: str):
        repo_path = self._resolve_local_dino_repo()
        self.backbone = torch.hub.load(repo_path, "dinov2_vits14", source="local", pretrained=False)

        encoder_path = os.path.join(model_dir, "dinov2_vit14s_latest_checkpoint.pth")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError("DINOv2 encoder checkpoint not found")
        encoder_checkpoint = _load_trusted_checkpoint(encoder_path, map_location="cpu")
        encoder_state = encoder_checkpoint.get("student", encoder_checkpoint)
        encoder_state = _strip_state_dict_prefixes(encoder_state, ("module.", "model."))
        encoder_load = self.backbone.load_state_dict(encoder_state, strict=False)
        _validate_load_result(encoder_load, "DINOv2 backbone")
        self.backbone.to(self.device).eval()
        self._explainability_spec = ExplainabilitySpec(
            encoder_mode="attention",
            encoder_label="DINOv2 encoder self-attention",
            decoder_mode="attention",
            decoder_label="Fusion Transformer temporal attention",
            encoder_layer_count=len(self.backbone.blocks),
            encoder_head_count=int(self.backbone.num_heads),
        )

        decoder_path = os.path.join(model_dir, "fusion_transformer_decoder_best_model.pth")
        if not os.path.exists(decoder_path):
            raise FileNotFoundError("DINOv2 decoder checkpoint not found")
        decoder_checkpoint = _load_trusted_checkpoint(decoder_path, map_location="cpu")
        decoder_state = decoder_checkpoint.get("state_dict", decoder_checkpoint)
        decoder_state = _strip_state_dict_prefixes(decoder_state, ("module.", "model."))

        class FusionTransformerDecoder(nn.Module):
            def __init__(self, feature_dim=384, num_classes=4, mstcn_stages=2, mstcn_layers=8, mstcn_f_maps=16, mstcn_f_dim=256, seq_length=30, d_model=256):
                super().__init__()
                self.reduce = nn.Linear(feature_dim, mstcn_f_dim)
                self.mstcn = MultiStageModel(
                    mstcn_stages=mstcn_stages,
                    mstcn_layers=mstcn_layers,
                    mstcn_f_maps=mstcn_f_maps,
                    mstcn_f_dim=mstcn_f_dim,
                    out_features=num_classes,
                    mstcn_causal_conv=True,
                    is_train=False,
                )
                self.transformer = Transformer(
                    mstcn_f_maps=mstcn_f_maps,
                    mstcn_f_dim=mstcn_f_dim,
                    out_features=num_classes,
                    len_q=seq_length,
                    d_model=d_model,
                )

            def forward(self, x, return_attention=False):
                x = x.permute(0, 2, 1)
                x_reduced = self.reduce(x)
                mstcn_input = x_reduced.permute(0, 2, 1)
                temporal_features = self.mstcn(mstcn_input)
                if isinstance(temporal_features, (list, tuple)):
                    temporal_features = temporal_features[-1]
                elif isinstance(temporal_features, torch.Tensor) and temporal_features.dim() == 4:
                    temporal_features = temporal_features[-1]

                if temporal_features.shape[1] == mstcn_input.shape[1]:
                    transformer_input = temporal_features.detach()
                else:
                    transformer_input = mstcn_input.detach()

                transformer_outputs = self.transformer(
                    transformer_input,
                    x_reduced,
                    return_attention=return_attention,
                )
                if return_attention:
                    transformer_out, attention_meta = transformer_outputs
                    return transformer_out.permute(0, 2, 1), attention_meta
                return transformer_outputs.permute(0, 2, 1)

        self.decoder = FusionTransformerDecoder()
        decoder_load = self.decoder.load_state_dict(decoder_state, strict=False)
        _validate_load_result(
            decoder_load,
            "DINOv2 decoder",
            allowed_unexpected_prefixes=(
                "mstcn.stage1.conv_out_classes.",
                "mstcn.stages.conv_out_classes.",
            ),
        )
        self.decoder.to(self.device).eval()
        self.available = True

    def reset_state(self):
        self.frame_features = []
        self._attention_recorder.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def warm_up(self):
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        self.predict(dummy_img)
        self.reset_state()

    def get_explainability_spec(self) -> ExplainabilitySpec:
        return self._explainability_spec

    def _ensure_attention_hook(self, layer_index: int) -> None:
        clamped_layer = clamp_index(layer_index, self._explainability_spec.encoder_layer_count)
        if self._attention_layer_index == clamped_layer and self._attention_recorder.handle is not None:
            return
        self._attention_recorder.attach(self.backbone.blocks[clamped_layer].norm1)
        self._attention_layer_index = clamped_layer

    def _compute_encoder_attention_map(self, head_index: int, output_shape: tuple[int, int]) -> np.ndarray:
        if self._attention_recorder.output is None or self._attention_layer_index is None:
            raise RuntimeError("DINO encoder attention recorder did not capture any tokens")

        tokens = self._attention_recorder.output.to(self.device)
        block = self.backbone.blocks[self._attention_layer_index]
        attn_module = block.attn
        qkv = attn_module.qkv(tokens).reshape(tokens.shape[0], tokens.shape[1], 3, attn_module.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )
        q = qkv[0] * attn_module.scale
        k = qkv[1]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)

        head = clamp_index(head_index, attn.shape[1])
        patch_start = 1 + int(getattr(self.backbone, "num_register_tokens", 0))
        cls_attention = attn[0, head, 0, patch_start:]
        patch_count = int(cls_attention.numel())
        grid_size = int(math.sqrt(patch_count))
        if grid_size * grid_size != patch_count:
            raise RuntimeError(f"Unexpected DINO patch attention size: {patch_count}")
        heatmap = cls_attention.view(grid_size, grid_size).detach().cpu().numpy()
        return cv2.resize(heatmap, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_CUBIC)

    def unload(self):
        if self.backbone is not None:
            self.backbone.to("cpu")
        if self.decoder is not None:
            self.decoder.to("cpu")
        self.backbone = None
        self.decoder = None
        self.frame_features = []
        self._attention_recorder.remove()
        self._attention_layer_index = None
        self.available = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def predict(self, rgb_image: np.ndarray, explainability: dict | None = None):
        if not self.available or self.backbone is None or self.decoder is None:
            raise RuntimeError("DINO-Endo predictor is not available")

        explain_enabled = bool(explainability and explainability.get("enabled"))
        encoder_layer = clamp_index(
            explainability.get("encoder_layer") if explainability else None,
            self._explainability_spec.encoder_layer_count,
        )
        encoder_head = clamp_index(
            explainability.get("encoder_head") if explainability else None,
            self._explainability_spec.encoder_head_count,
        )
        if explain_enabled:
            self._ensure_attention_hook(encoder_layer)
            self._attention_recorder.clear()
            display_image = self.display_aug(image=rgb_image)["image"]
        else:
            display_image = None

        processed = self.aug(image=rgb_image)["image"]
        chw = np.transpose(processed, (2, 0, 1))
        tensor = torch.tensor(chw, dtype=torch.float32).unsqueeze(0).to(self.device)

        with self._amp_context():
            feats = self.backbone.forward_features(tensor)
            if isinstance(feats, dict):
                feats = feats.get("x_norm_clstoken", next(iter(feats.values())))
            if feats.dim() == 3:
                feats = feats.mean(dim=1)

        self.frame_features.append(feats.squeeze(0).detach().cpu())
        if len(self.frame_features) > self.seq_length:
            self.frame_features = self.frame_features[-self.seq_length:]

        available_frames = len(self.frame_features)
        seq = torch.stack(self.frame_features[-available_frames:]).unsqueeze(0).to(self.device)
        if available_frames < self.seq_length:
            last_frame = seq[:, -1:, :]
            padding = last_frame.repeat(1, self.seq_length - available_frames, 1)
            seq = torch.cat([seq, padding], dim=1)

        decoder_input = seq.transpose(1, 2)
        with self._amp_context():
            decoder_outputs = self.decoder(decoder_input, return_attention=explain_enabled)
            if explain_enabled:
                logits, attention_meta = decoder_outputs
            else:
                logits = decoder_outputs

        if logits.dim() != 3:
            raise ValueError(f"Unexpected DINOv2 decoder output shape: {tuple(logits.shape)}")
        if logits.shape[1] == len(self.label_dict):
            last = logits[0, :, -1]
        elif logits.shape[2] == len(self.label_dict):
            last = logits[0, -1, :]
        else:
            raise ValueError(f"Unexpected DINOv2 class dimension in decoder output: {tuple(logits.shape)}")

        probs = torch.softmax(last, dim=0)
        pred_np = probs.detach().cpu().numpy()
        confidence = float(np.max(pred_np))
        phase_idx = int(np.argmax(pred_np))
        phase = self.label_dict.get(phase_idx, "idle")
        result = {"phase": phase, "probs": pred_np.tolist(), "confidence": confidence, "frames_used": available_frames}
        if explain_enabled and display_image is not None:
            encoder_heatmap = self._compute_encoder_attention_map(encoder_head, display_image.shape[:2])
            result["explainability"] = _build_explainability_payload(
                display_image=display_image,
                encoder_heatmap=encoder_heatmap,
                encoder_kind="attention",
                encoder_label=self._explainability_spec.encoder_label,
                decoder_values=attention_meta["decoder_strip"].detach().cpu().numpy(),
                decoder_kind="attention",
                decoder_label=self._explainability_spec.decoder_label,
                active_decoder_index=available_frames - 1,
                encoder_layer=encoder_layer,
                encoder_head=encoder_head,
            )
        return result


class PredictorVJEPA2:
    def __init__(self, model_dir: str | None = None, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir or default_model_dir()
        self.available = False
        self.encoder = None
        self.decoder = None
        self.label_dict = dict(enumerate(PHASE_LABELS))
        self._clip_frames = 16
        self._tubelet_size = 2
        self._crop_size = 256
        self._decoder_seq_length = 30
        self._frame_buffer = []
        self._feature_buffer = []
        self._vjepa_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
        self._vjepa_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)
        self._attention_recorder = ModuleOutputRecorder()
        self._attention_layer_index = None
        self._rotate_queries_or_keys = None
        self._explainability_spec = ExplainabilitySpec(
            encoder_mode="attention",
            encoder_label="V-JEPA2 encoder self-attention",
            decoder_mode="proxy",
            decoder_label="MLP decoder feature energy (proxy)",
        )
        self._load_models(self.model_dir)

    def _amp_context(self):
        return autocast("cuda") if MIXED_PRECISION_AVAILABLE and self.device.type == "cuda" else nullcontext()

    def _resolve_vjepa_repo(self):
        extras = []
        app_root = _app_root()
        if len(app_root.parents) >= 2:
            extras.append(app_root.parents[1] / "webapp" / "vjepa2")
        return _resolve_vendor_repo("vjepa2", extras)

    @staticmethod
    def _clean_checkpoint_keys(state_dict):
        cleaned_state = {}
        for key, value in state_dict.items():
            while key.startswith("module.") or key.startswith("backbone."):
                if key.startswith("module."):
                    key = key[len("module.") :]
                elif key.startswith("backbone."):
                    key = key[len("backbone.") :]
            cleaned_state[key] = value
        return cleaned_state

    @staticmethod
    def _validate_load_result(load_result, model_name: str):
        if load_result.unexpected_keys:
            sample = ", ".join(load_result.unexpected_keys[:5])
            raise RuntimeError(f"{model_name} load had unexpected keys: {sample}")
        if load_result.missing_keys:
            sample = ", ".join(load_result.missing_keys[:5])
            raise RuntimeError(f"{model_name} load missed required keys: {sample}")

    def _extract_temporal_features(self, features: torch.Tensor) -> torch.Tensor:
        if isinstance(features, dict):
            features = features.get("x_norm_patchtokens", features.get("x_norm_clstoken", next(iter(features.values()))))

        if features.dim() == 2:
            return features.unsqueeze(1).repeat(1, self._clip_frames, 1)
        if features.dim() != 3:
            raise ValueError(f"Unexpected V-JEPA2 encoder output shape: {tuple(features.shape)}")

        temporal_tokens = self._clip_frames // self._tubelet_size
        if temporal_tokens <= 0:
            raise ValueError("Invalid V-JEPA2 temporal configuration")
        if features.shape[1] % temporal_tokens != 0:
            raise ValueError(
                f"Cannot reshape V-JEPA2 features of shape {tuple(features.shape)} into {temporal_tokens} temporal groups"
            )

        spatial_tokens = features.shape[1] // temporal_tokens
        features = features.view(features.shape[0], temporal_tokens, spatial_tokens, features.shape[2]).mean(dim=2)
        return features.repeat_interleave(self._tubelet_size, dim=1)[:, : self._clip_frames, :]

    def _preprocess_clip(self, frames) -> torch.Tensor:
        resized_frames = [cv2.resize(frame, (self._crop_size, self._crop_size), interpolation=cv2.INTER_LINEAR) for frame in frames]
        clip = np.stack(resized_frames, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.transpose(clip, (3, 0, 1, 2)))
        return (tensor - self._vjepa_mean) / self._vjepa_std

    def _load_models(self, model_dir: str):
        vjepa2_path = self._resolve_vjepa_repo()
        if str(vjepa2_path) not in sys.path:
            sys.path.insert(0, str(vjepa2_path))

        from src.models import vision_transformer as vjepa_vit
        from src.models.utils.modules import rotate_queries_or_keys
        from src.utils.checkpoint_loader import robust_checkpoint_loader
        self._rotate_queries_or_keys = rotate_queries_or_keys

        encoder_path = os.path.join(model_dir, "vjepa_encoder_human.pt")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError("V-JEPA2 encoder not found")

        checkpoint = robust_checkpoint_loader(encoder_path, map_location=torch.device("cpu"))
        encoder_state = self._clean_checkpoint_keys(checkpoint.get("encoder", checkpoint))

        self.encoder = vjepa_vit.vit_large(
            patch_size=16,
            num_frames=self._clip_frames,
            tubelet_size=self._tubelet_size,
            img_size=self._crop_size,
            uniform_power=True,
            use_sdpa=True,
            use_rope=True,
        )
        encoder_load = self.encoder.load_state_dict(encoder_state, strict=False)
        self._validate_load_result(encoder_load, "V-JEPA2 encoder")
        self.encoder.to(self.device).eval()
        self._explainability_spec = ExplainabilitySpec(
            encoder_mode="attention",
            encoder_label="V-JEPA2 encoder self-attention",
            decoder_mode="proxy",
            decoder_label="MLP decoder feature energy (proxy)",
            encoder_layer_count=len(self.encoder.blocks),
            encoder_head_count=int(self.encoder.num_heads),
        )

        decoder_path = os.path.join(model_dir, "mlp_decoder_human.pth")
        if not os.path.exists(decoder_path):
            raise FileNotFoundError("V-JEPA2 MLP decoder not found")

        decoder_checkpoint = torch.load(decoder_path, map_location="cpu")
        decoder_state = decoder_checkpoint.get("model", decoder_checkpoint)
        decoder_in_dim = int(decoder_checkpoint.get("in_dim", 1024))
        decoder_num_classes = int(decoder_checkpoint.get("num_classes", len(self.label_dict)))
        self._decoder_seq_length = int(decoder_checkpoint.get("seq_length", self._decoder_seq_length))

        class MLPDecoder(nn.Module):
            def __init__(self, in_dim=1024, hidden_dim=256, num_classes=4):
                super().__init__()
                self.norm = nn.LayerNorm(in_dim)
                self.fc1 = nn.Linear(in_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, num_classes)
                self.relu = nn.ReLU()
                self.drop = nn.Dropout(0.5)

            def forward(self, x):
                x = x.mean(dim=1)
                x = self.norm(x)
                x = self.drop(self.relu(self.fc1(x)))
                x = self.drop(self.relu(self.fc2(x)))
                return self.fc3(x)

        self.decoder = MLPDecoder(in_dim=decoder_in_dim, num_classes=decoder_num_classes)
        self.decoder.load_state_dict(decoder_state, strict=True)
        self.decoder.to(self.device).eval()
        self.available = True

    def reset_state(self):
        self._frame_buffer = []
        self._feature_buffer = []
        self._attention_recorder.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def warm_up(self):
        dummy = np.random.randint(0, 255, (self._crop_size, self._crop_size, 3), dtype=np.uint8)
        self.predict(dummy)
        self.reset_state()

    def get_explainability_spec(self) -> ExplainabilitySpec:
        return self._explainability_spec

    def _ensure_attention_hook(self, layer_index: int) -> None:
        clamped_layer = clamp_index(layer_index, self._explainability_spec.encoder_layer_count)
        if self._attention_layer_index == clamped_layer and self._attention_recorder.handle is not None:
            return
        self._attention_recorder.attach(self.encoder.blocks[clamped_layer].norm1)
        self._attention_layer_index = clamped_layer

    def _compute_encoder_attention_map(
        self,
        *,
        head_index: int,
        temporal_group_index: int,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        if self._attention_recorder.output is None or self._attention_layer_index is None:
            raise RuntimeError("V-JEPA2 encoder attention recorder did not capture any tokens")
        if self._rotate_queries_or_keys is None:
            raise RuntimeError("V-JEPA2 rotation helper is unavailable")

        tokens = self._attention_recorder.output.to(self.device)
        block = self.encoder.blocks[self._attention_layer_index]
        attn_module = block.attn
        qkv = attn_module.qkv(tokens).unflatten(-1, (3, attn_module.num_heads, -1)).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]

        patch_grid = self._crop_size // 16
        temporal_groups = self._clip_frames // self._tubelet_size
        if hasattr(attn_module, "separate_positions"):
            mask = torch.arange(int(temporal_groups * patch_grid * patch_grid), device=tokens.device)
            d_mask, h_mask, w_mask = attn_module.separate_positions(mask, patch_grid, patch_grid)
            offset = 0
            qd = self._rotate_queries_or_keys(q[..., offset : offset + attn_module.d_dim], pos=d_mask)
            kd = self._rotate_queries_or_keys(k[..., offset : offset + attn_module.d_dim], pos=d_mask)
            offset += attn_module.d_dim
            qh = self._rotate_queries_or_keys(q[..., offset : offset + attn_module.h_dim], pos=h_mask)
            kh = self._rotate_queries_or_keys(k[..., offset : offset + attn_module.h_dim], pos=h_mask)
            offset += attn_module.h_dim
            qw = self._rotate_queries_or_keys(q[..., offset : offset + attn_module.w_dim], pos=w_mask)
            kw = self._rotate_queries_or_keys(k[..., offset : offset + attn_module.w_dim], pos=w_mask)
            offset += attn_module.w_dim
            q_parts = [qd, qh, qw]
            k_parts = [kd, kh, kw]
            if offset < attn_module.head_dim:
                q_parts.append(q[..., offset:])
                k_parts.append(k[..., offset:])
            q = torch.cat(q_parts, dim=-1)
            k = torch.cat(k_parts, dim=-1)

        attn = ((q @ k.transpose(-2, -1)) * attn_module.scale).softmax(dim=-1)
        head = clamp_index(head_index, attn.shape[1])
        group_size = patch_grid * patch_grid
        group_index = clamp_index(temporal_group_index, temporal_groups)
        start = group_index * group_size
        end = start + group_size
        group_attention = attn[0, head, start:end, start:end].mean(dim=0)
        heatmap = group_attention.view(patch_grid, patch_grid).detach().cpu().numpy()
        return cv2.resize(heatmap, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_CUBIC)

    def unload(self):
        if self.encoder is not None:
            self.encoder.to("cpu")
        if self.decoder is not None:
            self.decoder.to("cpu")
        self.encoder = None
        self.decoder = None
        self._frame_buffer = []
        self._feature_buffer = []
        self._attention_recorder.remove()
        self._attention_layer_index = None
        self.available = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def predict(self, rgb_image: np.ndarray, explainability: dict | None = None):
        if not self.available:
            raise RuntimeError("V-JEPA2 predictor is not available")

        explain_enabled = bool(explainability and explainability.get("enabled"))
        encoder_layer = clamp_index(
            explainability.get("encoder_layer") if explainability else None,
            self._explainability_spec.encoder_layer_count,
        )
        encoder_head = clamp_index(
            explainability.get("encoder_head") if explainability else None,
            self._explainability_spec.encoder_head_count,
        )
        if explain_enabled:
            self._ensure_attention_hook(encoder_layer)
            self._attention_recorder.clear()

        frame = np.ascontiguousarray(rgb_image, dtype=np.uint8)
        self._frame_buffer.append(frame)
        if len(self._frame_buffer) > self._clip_frames:
            self._frame_buffer = self._frame_buffer[-self._clip_frames:]

        clip_frames = list(self._frame_buffer)
        while len(clip_frames) < self._clip_frames:
            clip_frames.append(clip_frames[-1])

        tensor = self._preprocess_clip(clip_frames).unsqueeze(0).to(self.device)
        with self._amp_context():
            features = self._extract_temporal_features(self.encoder(tensor))

        latest_feature_idx = min(len(self._frame_buffer), self._clip_frames) - 1
        latest_feature = features[0, latest_feature_idx].float().detach().cpu()
        self._feature_buffer.append(latest_feature)
        if len(self._feature_buffer) > self._decoder_seq_length:
            self._feature_buffer = self._feature_buffer[-self._decoder_seq_length:]

        available_frames = len(self._feature_buffer)
        seq = torch.stack(self._feature_buffer, dim=0).unsqueeze(0).to(self.device)
        if available_frames < self._decoder_seq_length:
            padding = seq[:, -1:, :].repeat(1, self._decoder_seq_length - available_frames, 1)
            seq = torch.cat([seq, padding], dim=1)

        with self._amp_context():
            logits = self.decoder(seq)

        probs = torch.softmax(logits[0], dim=0)
        pred_np = probs.detach().cpu().numpy()
        confidence = float(np.max(pred_np))
        phase_idx = int(np.argmax(pred_np))
        phase = self.label_dict.get(phase_idx, "idle")
        result = {"phase": phase, "probs": pred_np.tolist(), "confidence": confidence, "frames_used": available_frames}
        if explain_enabled:
            latest_group_index = latest_feature_idx // self._tubelet_size
            display_image = resize_rgb_image(frame, (self._crop_size, self._crop_size))
            encoder_heatmap = self._compute_encoder_attention_map(
                head_index=encoder_head,
                temporal_group_index=latest_group_index,
                output_shape=display_image.shape[:2],
            )
            decoder_proxy_values = [feature.abs().mean().item() for feature in self._feature_buffer]
            result["explainability"] = _build_explainability_payload(
                display_image=display_image,
                encoder_heatmap=encoder_heatmap,
                encoder_kind="attention",
                encoder_label=self._explainability_spec.encoder_label,
                decoder_values=decoder_proxy_values,
                decoder_kind="proxy",
                decoder_label=self._explainability_spec.decoder_label,
                active_decoder_index=available_frames - 1,
                encoder_layer=encoder_layer,
                encoder_head=encoder_head,
                notes="Decoder view is a proxy feature-energy strip because the V-JEPA2 classifier head is an MLP.",
            )
        return result


def create_predictor(model_key: str, model_dir: str | None = None, device: str | None = None):
    resolved_key = normalize_model_key(model_key)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_dir = model_dir or default_model_dir()

    if resolved_key == "aiendo":
        return Predictor(model_dir=resolved_model_dir, device=resolved_device)
    if resolved_key == "dinov2":
        return PredictorDinoV2(model_dir=resolved_model_dir, device=resolved_device)
    if resolved_key == "vjepa2":
        return PredictorVJEPA2(model_dir=resolved_model_dir, device=resolved_device)
    raise KeyError(f"Unsupported model key: {model_key}")
