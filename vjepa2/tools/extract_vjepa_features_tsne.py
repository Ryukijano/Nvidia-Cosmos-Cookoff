#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Fix import path for V-JEPA modules
sys.path.append('/scratch/cbjp404/AI-Endo/vjepa2')
sys.path.append('/scratch/cbjp404/AI-Endo/vjepa2/src')

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.models import vision_transformer as video_vit
from src.utils.checkpoint_loader import robust_checkpoint_loader
from app.vjepa.transforms import make_transforms


PHASE_TO_ID = {
    "Idle": 0,
    "Marking": 1,
    "Injection": 2,
    "Dissection": 3,
    # lower-case fallbacks
    "idle": 0,
    "marking": 1,
    "injection": 2,
    "dissection": 3,
}


def load_encoder(checkpoint_path: str,
                 model_name: str,
                 crop_size: int,
                 frames_per_clip: int,
                 tubelet_size: int,
                 uniform_power: bool = True,
                 use_sdpa: bool = True,
                 use_rope: bool = True,
                 device: str = "cuda") -> nn.Module:
    model = video_vit.__dict__[model_name](
        img_size=crop_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_rope=use_rope,
    )
    model.eval()
    checkpoint = robust_checkpoint_loader(checkpoint_path, map_location=torch.device("cpu"))
    state = checkpoint["encoder"]
    # Strip DDP and wrapper prefixes
    new_state = {}
    for k, v in state.items():
        k = k.replace("module.", "")
        k = k.replace("backbone.", "")
        new_state[k] = v
    msg = model.load_state_dict(new_state, strict=False)
    print(f"Loaded encoder state_dict with msg: {msg}")
    model.to(device)
    return model


def build_transform(crop_size: int):
    # Deterministic transform matching training normalization, no randomness
    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )


def read_annotations(case_dir: Path) -> List[int]:
    annot = case_dir / "annotations.txt"
    labels: List[int] = []
    if annot.exists():
        with open(annot, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                lab = parts[1] if len(parts) > 1 else "Idle"
                labels.append(PHASE_TO_ID.get(lab, 0))
    return labels


def list_frames(case_dir: Path) -> List[Path]:
    frames = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        frames.extend(case_dir.glob(ext))
    frames = sorted(frames)
    return frames


@torch.no_grad()
def extract_features_for_case(model: nn.Module,
                              frames: List[Path],
                              transform,
                              frames_per_clip: int,
                              tubelet_size: int,
                              stride: int,
                              device: str) -> Tuple[np.ndarray, List[int]]:
    if len(frames) == 0:
        return np.zeros((0, model.embed_dim)), []

    features: List[np.ndarray] = []
    frame_indices: List[int] = []

    # Slide windows
    for start in range(0, max(1, len(frames) - frames_per_clip + 1), stride):
        end = min(start + frames_per_clip, len(frames))
        clip_paths = frames[start:end]
        # pad last frame if needed
        while len(clip_paths) < frames_per_clip:
            clip_paths.append(clip_paths[-1])

        # load images to numpy [T,H,W,C]
        imgs = []
        for p in clip_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(np.array(img, dtype=np.uint8))
            except Exception:
                if len(imgs) == 0:
                    imgs.append(np.zeros((crop_size, crop_size, 3), dtype=np.uint8))
                else:
                    imgs.append(imgs[-1])
        buffer = np.stack(imgs, axis=0)  # [T,H,W,C]

        # apply video transform → tensor [C,T,H,W]
        tensor = transform(buffer)  # normalized to training stats
        tensor = tensor.unsqueeze(0).to(device)  # [1,C,T,H,W]

        # forward → tokens [B,N,D]
        out = model(tensor)
        # decode into per-time tokens
        B, N, D = out.shape
        T = frames_per_clip // tubelet_size
        S = N // T
        out = out.view(B, T, S, D).mean(dim=2)  # [1,T,D]
        out = out[0].float().cpu().numpy()  # [T,D]

        # expand to per-frame by repeating each tubelet token
        per_frame = np.repeat(out, tubelet_size, axis=0)  # [F,D]
        per_frame = per_frame[:frames_per_clip]

        # assign global frame indices
        idxs = list(range(start, start + frames_per_clip))
        idxs = [min(i, len(frames) - 1) for i in idxs]

        features.append(per_frame)
        frame_indices.extend(idxs)

    features = np.concatenate(features, axis=0) if features else np.zeros((0, model.embed_dim))
    return features, frame_indices


def aggregate_labels(frame_indices: List[int], labels_seq: List[int]) -> List[int]:
    if not labels_seq:
        return [0 for _ in frame_indices]
    # Align by index; if labels shorter, clip; if longer, use mapped index
    out = []
    n = len(labels_seq)
    for i in frame_indices:
        j = min(i, n - 1)
        out.append(labels_seq[j])
    return out


def collect_cases(root: Path) -> List[Path]:
    if not root.exists():
        return []
    # prefer leaf directories containing frames
    all_cases = [d for d in root.iterdir() if d.is_dir()]
    # if root itself contains frames, treat as single case
    if not all_cases:
        return [root]

    # Filter to only human cases (matching ground_truth_files)
    human_cases = [
        'M_01132025160217_0000000U11350017_1_001_001-1',
        'M_01132025160217_0000000U11350017_1_001_003-1',
        'M_01132025160217_0000000U11350017_1_001_004-1',
        'M_01132025160217_0000000U11350017_1_001_006-1',
        'M_08272024133303_0000000000000scc_1_001_003-1'
    ]
    cases = [d for d in all_cases if d.name in human_cases]
    return cases


def run_tsne(all_feats: np.ndarray, all_labels: np.ndarray, out_path: Path, title: str):
    # optional PCA to 50 dims
    X = all_feats.astype(np.float32)
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=42).fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1500, random_state=42, init="pca")
    X2 = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    cmap = {
        0: (0.6, 0.6, 0.6),   # Idle - gray
        1: (0.2, 0.4, 1.0),   # Marking - blue
        2: (0.2, 0.8, 0.2),   # Injection - green
        3: (1.0, 0.2, 0.2),   # Dissection - red
    }
    for lab in sorted(np.unique(all_labels)):
        mask = all_labels == lab
        plt.scatter(X2[mask, 0], X2[mask, 1], s=5, c=[cmap.get(int(lab), (0, 0, 0))], label=str(lab), alpha=0.7)
    plt.legend(title="Phase", markerscale=3)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Extract V-JEPA features and plot t-SNE vs labels.")
    parser.add_argument("--checkpoint", required=True, help="Path to V-JEPA checkpoint (latest.pt)")
    parser.add_argument("--exvivo_dir", required=True, help="Path to ExvivoAnimalTrial root")
    parser.add_argument("--invivo_dir", required=True, help="Path to InvivoAnimalTrial root")
    parser.add_argument("--output_dir", required=True, help="Directory to save features and plots")
    parser.add_argument("--frames_per_clip", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--model_name", type=str, default="vit_large")
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    # model
    encoder = load_encoder(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        crop_size=args.crop_size,
        frames_per_clip=args.frames_per_clip,
        tubelet_size=args.tubelet_size,
        device=device,
    )
    transform = build_transform(args.crop_size)

    all_feats: List[np.ndarray] = []
    all_labels: List[int] = []

    for root_dir in [args.exvivo_dir, args.invivo_dir]:
        root = Path(root_dir)
        if root.exists() and root.is_dir():
            cases = collect_cases(root)
        else:
            cases = []  # Skip non-existent or non-directory paths
        for case_dir in cases:
            frames = list_frames(case_dir)
            labels_seq = read_annotations(case_dir)
            feats, frame_idxs = extract_features_for_case(
                model=encoder,
                frames=frames,
                transform=transform,
                frames_per_clip=args.frames_per_clip,
                tubelet_size=args.tubelet_size,
                stride=args.stride,
                device=device,
            )
            if feats.shape[0] == 0:
                continue
            labels = aggregate_labels(frame_idxs, labels_seq)
            all_feats.append(feats)
            all_labels.extend(labels)

    if not all_feats:
        print("No features extracted. Check directories and frame files.")
        return

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.array(all_labels, dtype=np.int64)

    # save arrays
    np.savez(Path(args.output_dir) / "vjepa_features_labels.npz", features=all_feats, labels=all_labels)

    # tsne
    out_png = Path(args.output_dir) / "vjepa_tsne_exvivo_invivo.png"
    run_tsne(all_feats, all_labels, out_png, title="V-JEPA features t-SNE (Exvivo+Invivo)")
    print(f"Saved t-SNE plot to {out_png}")


if __name__ == "__main__":
    main()


