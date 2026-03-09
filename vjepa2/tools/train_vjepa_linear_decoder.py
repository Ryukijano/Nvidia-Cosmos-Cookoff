#!/usr/bin/env python3
import argparse
import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import datetime


class NPZFeatureDataset(Dataset):
    def __init__(self, npz_paths, seq_length=30, overlap=0.75):
        self.samples = []
        self.seq_length = seq_length
        self._load_npzs(npz_paths, overlap)

    def _load_npzs(self, npz_paths, overlap):
        for p in npz_paths:
            data = np.load(p)
            feats = data["features"]  # [N, D]
            labels = data["labels"]   # [N]
            stride = max(1, int(self.seq_length * (1 - overlap)))
            for i in range(0, max(1, len(feats) - self.seq_length + 1), stride):
                x = feats[i:i + self.seq_length]
                y = labels[i:i + self.seq_length]
                if len(x) < self.seq_length:
                    pad = np.tile(x[-1:], (self.seq_length - len(x), 1))
                    x = np.vstack([x, pad])
                    y = np.pad(y, (0, self.seq_length - len(y)), mode='edge')
                # majority label in window
                vals, counts = np.unique(y, return_counts=True)
                maj = int(vals[np.argmax(counts)])
                self.samples.append((x.astype(np.float32), maj))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class MLPDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_classes: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):  # x: [B, T, D]
        x = x.mean(dim=1)  # temporal average
        x = self.norm(x)
        x = self.drop(self.relu(self.fc1(x)))
        x = self.drop(self.relu(self.fc2(x)))
        return self.fc3(x)


def class_weights_from_hist(dataloader, num_classes):
    counts = torch.zeros(num_classes)
    for _, y in dataloader:
        for c in range(num_classes):
            counts[c] += (y == c).sum()
    counts = counts + 1e-6
    weights = counts.max() / counts
    return weights


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'  # Changed from 12355 to avoid conflicts
    # Try NCCL first, fallback to Gloo if it fails
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=30))
    except RuntimeError:
        print(f"Rank {rank}: NCCL failed, trying Gloo")
        dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=30))
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train(rank, args):
    # Create datasets first to get in_dim
    train_ds = NPZFeatureDataset(args.train_npz, seq_length=args.seq_length, overlap=0.75)
    val_ds = NPZFeatureDataset(args.val_npz, seq_length=args.seq_length, overlap=0.0)
    sample_x, _ = train_ds[0]
    in_dim = sample_x.shape[-1]

    if args.world_size > 1:
        setup_ddp(rank, args.world_size)
        device = torch.device(f"cuda:{rank}")

        train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size, rank=rank)
        val_sampler = DistributedSampler(val_ds, num_replicas=args.world_size, rank=rank)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=0, pin_memory=True)

        model = MLPDecoder(in_dim=in_dim, num_classes=args.num_classes).to(device)
        model = DDP(model, device_ids=[rank])

        # class weights on rank 0 and broadcast
        if rank == 0:
            cw = class_weights_from_hist(train_loader, args.num_classes)
        else:
            cw = torch.zeros(args.num_classes)
        dist.broadcast(cw, 0)
        cw = cw.to(device)
    else:
        # Single GPU mode
        device = torch.device("cuda:0")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        model = MLPDecoder(in_dim=in_dim, num_classes=args.num_classes).to(device)
        cw = class_weights_from_hist(train_loader, args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    ckpt_path = Path(args.out_dir) / "mlp_decoder_best.pth" if rank == 0 else None
    os.makedirs(args.out_dir, exist_ok=True) if rank == 0 else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.world_size > 1:
            train_sampler.set_epoch(epoch)
        total_loss = 0.0
        total_acc_n = 0
        total_acc_d = 0
        with tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} Train", leave=False) as pbar:
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                total_acc_n += (preds == y).sum().item()
                total_acc_d += y.size(0)

                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(preds == y).float().mean().item():.4f}"})

        if args.world_size > 1:
            # reduce metrics
            metrics = torch.tensor([total_loss, total_acc_n, total_acc_d], device=device)
            dist.reduce(metrics, dst=0)
            train_loss = metrics[0].item() / max(1, metrics[2].item()) if rank == 0 else 0.0
            train_acc = metrics[1].item() / max(1, metrics[2].item()) if rank == 0 else 0.0
        else:
            train_loss = total_loss / max(1, total_acc_d)
            train_acc = total_acc_n / max(1, total_acc_d)

        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} Val", leave=False) as pbar:
                for x, y in pbar:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    preds = logits.argmax(dim=1)
                    all_y.append(y.cpu())
                    all_p.append(preds.cpu())

        all_y = torch.cat(all_y)
        all_p = torch.cat(all_p)

        if args.world_size > 1:
            dist.barrier()  # sync

            # gather all on rank 0
            all_y_list = [torch.zeros_like(all_y) for _ in range(args.world_size)]
            all_p_list = [torch.zeros_like(all_p) for _ in range(args.world_size)]
            dist.gather(all_y, gather_list=all_y_list if rank == 0 else None, dst=0)
            dist.gather(all_p, gather_list=all_p_list if rank == 0 else None, dst=0)

            if rank == 0:
                all_y = torch.cat(all_y_list)
                all_p = torch.cat(all_p_list)
                val_acc = float(accuracy_score(all_y.numpy(), all_p.numpy()))
                val_f1 = float(f1_score(all_y.numpy(), all_p.numpy(), average='macro'))

                print(f"Epoch {epoch}/{args.epochs} - train_loss {train_loss:.4f} acc {train_acc:.4f} | val_acc {val_acc:.4f} f1 {val_f1:.4f}")

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    torch.save({
                        'model': model.module.state_dict(),
                        'in_dim': in_dim,
                        'num_classes': args.num_classes,
                        'seq_length': args.seq_length,
                        'val_f1': val_f1,
                    }, ckpt_path)
        else:
            val_acc = float(accuracy_score(all_y.numpy(), all_p.numpy()))
            val_f1 = float(f1_score(all_y.numpy(), all_p.numpy(), average='macro'))

            print(f"Epoch {epoch}/{args.epochs} - train_loss {train_loss:.4f} acc {train_acc:.4f} | val_acc {val_acc:.4f} f1 {val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'model': model.state_dict(),
                    'in_dim': in_dim,
                    'num_classes': args.num_classes,
                    'seq_length': args.seq_length,
                    'val_f1': val_f1,
                }, ckpt_path)

    if args.world_size > 1:
        cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a small MLP decoder on V-JEPA features")
    parser.add_argument('--train_npz', nargs='+', required=True, help='Path(s) to training npz files')
    parser.add_argument('--val_npz', nargs='+', required=True, help='Path(s) to validation npz files')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--seq_length', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--world_size', type=int, default=3)
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)


