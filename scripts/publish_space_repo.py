from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

import sys

SCRIPT_PATH = Path(__file__).resolve()
SPACE_ROOT = SCRIPT_PATH.parents[1]
if str(SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACE_ROOT))

from stage_space_bundle import stage_bundle


def _space_variables(args: argparse.Namespace) -> dict[str, str]:
    variables = {
        "SPACE_ENABLED_MODELS": args.enabled_models,
        "SPACE_DEFAULT_MODEL": args.default_model,
    }
    if args.aiendo_model_repo_id:
        variables["AIENDO_MODEL_REPO_ID"] = args.aiendo_model_repo_id
    if args.dino_model_repo_id:
        variables["DINO_MODEL_REPO_ID"] = args.dino_model_repo_id
    if args.vjepa2_model_repo_id:
        variables["VJEPA2_MODEL_REPO_ID"] = args.vjepa2_model_repo_id
    return variables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the staged Docker Space bundle and set its variables.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face Space repo ID.")
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help="Optional pre-staged bundle directory. If omitted, a temporary bundle is staged automatically.",
    )
    parser.add_argument("--enabled-models", default="dinov2,aiendo,vjepa2")
    parser.add_argument("--default-model", default="dinov2")
    parser.add_argument("--aiendo-model-repo-id", default=None)
    parser.add_argument("--dino-model-repo-id", default=None)
    parser.add_argument("--vjepa2-model-repo-id", default=None)
    parser.add_argument("--revision", default=None, help="Optional target revision or branch.")
    parser.add_argument("--private", action="store_true", help="Create the Space repo as private.")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable name containing the Hugging Face write token.",
    )
    return parser.parse_args()


def _publish_bundle(api: HfApi, *, repo_id: str, bundle_dir: Path, revision: str | None) -> None:
    upload_kwargs = {
        "repo_id": repo_id,
        "repo_type": "space",
        "folder_path": str(bundle_dir),
    }
    if revision:
        upload_kwargs["revision"] = revision
    api.upload_folder(**upload_kwargs)


def main() -> None:
    args = parse_args()
    token = os.getenv(args.token_env) or None
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)

    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir).expanduser().resolve()
        if not bundle_dir.exists():
            raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
        _publish_bundle(api, repo_id=args.repo_id, bundle_dir=bundle_dir, revision=args.revision)
    else:
        with tempfile.TemporaryDirectory(prefix="hf-space-bundle-") as temp_dir:
            bundle_dir = stage_bundle(SPACE_ROOT, Path(temp_dir), overwrite=True)
            _publish_bundle(api, repo_id=args.repo_id, bundle_dir=bundle_dir, revision=args.revision)

    for key, value in _space_variables(args).items():
        api.add_space_variable(
            repo_id=args.repo_id,
            key=key,
            value=value,
            description=f"Managed by publish_space_repo.py for {key}",
        )

    print(f"Published Space bundle to {args.repo_id}")
    for key, value in _space_variables(args).items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
