from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

import sys

SCRIPT_PATH = Path(__file__).resolve()
SPACE_ROOT = SCRIPT_PATH.parents[1]
if str(SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACE_ROOT))

from predictor import create_predictor


MODEL_REQUIREMENTS = {
    "aiendo": ("resnet50.pth", "fusion.pth", "transformer.pth"),
    "dinov2": ("dinov2_vit14s_latest_checkpoint.pth", "fusion_transformer_decoder_best_model.pth"),
    "vjepa2": ("vjepa_encoder_human.pt", "mlp_decoder_human.pth"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the isolated HF Space predictors.")
    parser.add_argument("--model", choices=sorted(MODEL_REQUIREMENTS), required=True)
    parser.add_argument("--model-dir", default=str(SPACE_ROOT / "model"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    missing = [name for name in MODEL_REQUIREMENTS[args.model] if not (model_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required checkpoints in {model_dir}: {', '.join(missing)}")

    os.environ["SPACE_MODEL_DIR"] = str(model_dir)
    dummy = np.random.randint(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8)

    predictor = create_predictor(args.model, model_dir=str(model_dir), device=args.device)
    predictor.reset_state()
    result = predictor.predict(dummy)
    predictor.unload()

    print(f"model={args.model}")
    print(f"phase={result.get('phase')}")
    print(f"confidence={result.get('confidence')}")
    print(f"frames_used={result.get('frames_used')}")


if __name__ == "__main__":
    main()
