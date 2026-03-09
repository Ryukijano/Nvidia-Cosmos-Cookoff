from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")
    if dst.exists():
        if not overwrite:
            print(f"Skipping existing {dst}")
            return
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns('.git', '__pycache__', '.pytest_cache', '.mypy_cache', '*.pyc', '*.pyo'),
    )
    print(f"Copied {src} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Copy vendored dinov2/ and vjepa2/ source trees into the Space folder.')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing destination directories.')
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    space_root = script_path.parents[1]
    repo_root = script_path.parents[3]

    copy_tree(repo_root / 'dinov2', space_root / 'dinov2', overwrite=args.overwrite)
    copy_tree(repo_root / 'vjepa2', space_root / 'vjepa2', overwrite=args.overwrite)


if __name__ == '__main__':
    main()
