from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

import sys

SCRIPT_PATH = Path(__file__).resolve()
SPACE_ROOT = SCRIPT_PATH.parents[1]
if str(SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACE_ROOT))

from model_registry import MODEL_SPECS


ENV_VAR_BY_FAMILY = {
    "aiendo": "AIENDO_MODEL_REPO_ID",
    "dinov2": "DINO_MODEL_REPO_ID",
    "vjepa2": "VJEPA2_MODEL_REPO_ID",
}


def _render_model_card(*, family: str, repo_id: str, copied_files: list[str]) -> str:
    spec = MODEL_SPECS[family]
    file_list = "\n".join(f"- `{name}`" for name in copied_files)
    return f"""---
tags:
- medical-imaging
- endoscopy
- surgical-phase-recognition
- {family}
---

# {spec.label} checkpoints for the AI-Endo Hugging Face Space

This repository stores the published checkpoint set for the **{spec.label}** phase-recognition path used by `hf_spaces/DINO-ENDO/`.

## Files

{file_list}

## Consumed by the Space

Set the following Space environment variable so the Streamlit Space can download these files lazily at runtime:

```text
{ENV_VAR_BY_FAMILY[family]}={repo_id}
```
"""


def _stage_model_family(*, family: str, model_dir: Path, staging_dir: Path, repo_id: str) -> int:
    spec = MODEL_SPECS[family]
    copied_files: list[str] = []
    total_bytes = 0

    for filename in spec.required_files:
        src = model_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required checkpoint: {src}")
        dst = staging_dir / filename
        shutil.copy2(src, dst)
        copied_files.append(filename)
        total_bytes += src.stat().st_size

    for filename in spec.optional_files:
        src = model_dir / filename
        if not src.exists():
            continue
        dst = staging_dir / filename
        shutil.copy2(src, dst)
        copied_files.append(filename)
        total_bytes += src.stat().st_size

    (staging_dir / "README.md").write_text(
        _render_model_card(family=family, repo_id=repo_id, copied_files=copied_files),
        encoding="utf-8",
    )
    return total_bytes


def _should_use_large_upload(mode: str, total_bytes: int) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return total_bytes >= 2 * 1024 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a model-family checkpoint repo for the HF Space.")
    parser.add_argument("--family", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face model repo ID.")
    parser.add_argument(
        "--model-dir",
        default=str(SPACE_ROOT / "model"),
        help="Directory containing the local checkpoints to publish.",
    )
    parser.add_argument(
        "--upload-mode",
        choices=("auto", "never", "always"),
        default="auto",
        help="Choose whether to force upload_large_folder for this family.",
    )
    parser.add_argument("--revision", default=None, help="Optional target revision or branch.")
    parser.add_argument("--private", action="store_true", help="Create the model repo as private.")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable name containing the Hugging Face write token.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    token = os.getenv(args.token_env) or None
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"hf-space-{args.family}-") as temp_dir:
        staging_dir = Path(temp_dir)
        total_bytes = _stage_model_family(
            family=args.family,
            model_dir=model_dir,
            staging_dir=staging_dir,
            repo_id=args.repo_id,
        )

        upload_kwargs = {
            "repo_id": args.repo_id,
            "repo_type": "model",
            "folder_path": str(staging_dir),
        }
        if args.revision:
            upload_kwargs["revision"] = args.revision

        if _should_use_large_upload(args.upload_mode, total_bytes):
            api.upload_large_folder(**upload_kwargs)
            mode = "upload_large_folder"
        else:
            api.upload_folder(**upload_kwargs)
            mode = "upload_folder"

    print(f"Published {args.family} checkpoints to {args.repo_id} via {mode}")
    print(f"Suggested Space variable: {ENV_VAR_BY_FAMILY[args.family]}={args.repo_id}")


if __name__ == "__main__":
    main()
