from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT_FILES = (
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
    "Dockerfile",
    "README.md",
    "app.py",
    "explainability.py",
    "model_manager.py",
    "model_registry.py",
    "predictor.py",
    "requirements.txt",
    "runtime-requirements.txt",
    "start_space.py",
    "video_utils.py",
)

ROOT_DIRS = (
    ".streamlit",
    "dinov2",
    "model",
    "scripts",
    "vjepa2",
)

IGNORE_PATTERNS = (
    ".git",
    ".cache",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "*.egg-info",
    "*.ipynb",
    "*.pt",
    "*.pth",
    "*.pyc",
    "*.pyo",
    "assets",
    "notebooks",
    "tests",
)


def _copy_item(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required Space item: {src}")

    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*IGNORE_PATTERNS))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def stage_bundle(space_root: Path, output_dir: Path, overwrite: bool) -> Path:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ROOT_FILES:
        _copy_item(space_root / name, output_dir / name)
    for name in ROOT_DIRS:
        _copy_item(space_root / name, output_dir / name)

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a code-only Hugging Face Space bundle from the local DINO-ENDO scaffold."
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/dino_space_minimal_upload",
        help="Destination directory for the staged bundle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    space_root = script_path.parents[1]
    output_dir = Path(args.output_dir).expanduser().resolve()
    staged_dir = stage_bundle(space_root, output_dir, overwrite=args.overwrite)
    print(f"Staged Space bundle at {staged_dir}")


if __name__ == "__main__":
    main()
