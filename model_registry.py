from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

APP_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("SPACE_MODEL_DIR", APP_ROOT / "model")).expanduser().resolve()


def _default_hf_home() -> Path:
    data_dir = Path("/data")
    if data_dir.is_dir():
        return data_dir / ".huggingface"
    return APP_ROOT / ".cache" / "huggingface"


HF_HOME = Path(os.environ.setdefault("HF_HOME", str(_default_hf_home()))).expanduser().resolve()
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    required_files: Tuple[str, ...]
    optional_files: Tuple[str, ...] = ()


MODEL_SPECS: Dict[str, ModelSpec] = {
    "aiendo": ModelSpec(
        key="aiendo",
        label="AI-Endo",
        required_files=("resnet50.pth", "fusion.pth", "transformer.pth"),
    ),
    "dinov2": ModelSpec(
        key="dinov2",
        label="DINO-Endo",
        required_files=("dinov2_vit14s_latest_checkpoint.pth", "fusion_transformer_decoder_best_model.pth"),
        optional_files=("dinov2_decoder.pth",),
    ),
    "vjepa2": ModelSpec(
        key="vjepa2",
        label="V-JEPA2",
        required_files=("vjepa_encoder_human.pt", "mlp_decoder_human.pth"),
    ),
}


def _repo_env_name(model_key: str) -> str:
    prefix = {"aiendo": "AIENDO", "dinov2": "DINO", "vjepa2": "VJEPA2"}[model_key]
    return f"{prefix}_MODEL_REPO_ID"


def _revision_env_name(model_key: str) -> str:
    prefix = {"aiendo": "AIENDO", "dinov2": "DINO", "vjepa2": "VJEPA2"}[model_key]
    return f"{prefix}_MODEL_REVISION"


def _subfolder_env_name(model_key: str) -> str:
    prefix = {"aiendo": "AIENDO", "dinov2": "DINO", "vjepa2": "VJEPA2"}[model_key]
    return f"{prefix}_MODEL_SUBFOLDER"


def get_model_repo_id(model_key: str) -> str | None:
    return os.getenv(_repo_env_name(model_key)) or os.getenv("PHASE_MODEL_REPO_ID")


def get_model_revision(model_key: str) -> str | None:
    return os.getenv(_revision_env_name(model_key)) or os.getenv("PHASE_MODEL_REVISION")


def get_model_subfolder(model_key: str) -> str:
    return (os.getenv(_subfolder_env_name(model_key)) or "").strip("/")


def get_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


def ensure_model_root() -> Path:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    return MODEL_ROOT


def _remote_filename(model_key: str, filename: str) -> str:
    subfolder = get_model_subfolder(model_key)
    return f"{subfolder}/{filename}" if subfolder else filename


def _download_to_model_root(model_key: str, filename: str, *, optional: bool = False) -> Path | None:
    target = ensure_model_root() / filename
    if target.exists():
        return target

    repo_id = get_model_repo_id(model_key)
    if not repo_id:
        if optional:
            return None
        raise FileNotFoundError(
            f"Missing {filename} in {MODEL_ROOT}. Set { _repo_env_name(model_key) } or PHASE_MODEL_REPO_ID, "
            f"or copy the checkpoint into the local model directory."
        )

    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=_remote_filename(model_key, filename),
            repo_type="model",
            revision=get_model_revision(model_key),
            token=get_hf_token(),
        )
    except EntryNotFoundError:
        if optional:
            return None
        raise

    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != target.resolve():
        shutil.copy2(downloaded_path, target)
    return target


def ensure_model_artifacts(model_key: str) -> Path:
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model key: {model_key}")

    spec = MODEL_SPECS[model_key]
    ensure_model_root()

    for filename in spec.required_files:
        _download_to_model_root(model_key, filename, optional=False)
    for filename in spec.optional_files:
        _download_to_model_root(model_key, filename, optional=True)

    return MODEL_ROOT


def get_model_source_summary(model_key: str) -> dict:
    spec = MODEL_SPECS[model_key]
    return {
        "label": spec.label,
        "model_dir": str(MODEL_ROOT),
        "repo_id": get_model_repo_id(model_key),
        "revision": get_model_revision(model_key),
        "subfolder": get_model_subfolder(model_key),
        "required_files": list(spec.required_files),
        "optional_files": list(spec.optional_files),
    }
