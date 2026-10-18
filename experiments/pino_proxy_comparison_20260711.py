#!/usr/bin/env python3
"""Reproducible proxy-model benchmark for PINO claims.

All models use the same FEM-labelled samples and the same fixed split.  The
operator runs are launched through ``pretrain_pino_v3.py``; this helper creates
the immutable manifest and trains the scalar MLP/CNN reference models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structural.training.pretrain_pino_v3 import FEMDatasetV3, collate_v3


ROOT = Path("/data/xvxinyao/xvxinyao/total_project/fem_dataset_v3")
OUT = Path("results/pino_proxy_model_comparison_20260711")
SEED = 20260711


def make_split() -> None:
    files = sorted(ROOT.glob("sample_*.npz"))
    if len(files) < 100:
        raise RuntimeError(f"Expected FEM samples in {ROOT}, found {len(files)}")
    rng = random.Random(SEED)
    rng.shuffle(files)
    n = len(files)
    n_train, n_cal = int(.60 * n), int(.20 * n)
    groups = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_cal],
        "sealed_test": files[n_train + n_cal:],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    index = {k: [{"path": str(p)} for p in v] for k, v in groups.items()}
    index["protocol"] = {
        "seed": SEED,
        "data_root": str(ROOT),
        "split": "60/20/20 sample-level fixed split",
        "fem_labels": "density, force, fixed_mask, displacement, stress, von_mises",
        "primary_stress_metric": "relative error of FEM peak von-Mises stress",
        "sf_metric": "absolute error of log safety factor; common allowable stress",
        "decision_metric": "identical calibration-only zero-false-safe threshold per model",
    }
    index["sha256_paths"] = hashlib.sha256(
        "\n".join(str(p) for p in files).encode()).hexdigest()
    with (OUT / "split_manifest.json").open("w") as f:
        json.dump(index, f, indent=2)
    print(json.dumps({k: len(v) for k, v in groups.items()}, indent=2))


class GeometryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, 128), nn.GELU(), nn.Dropout(.1),
                                 nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, f):
        return self.net(f).squeeze(1)


class ScalarCNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(5, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)


def geometry_features(x: torch.Tensor) -> torch.Tensor:
    """Deterministic descriptors from the same density/force/SDF input."""
    rho, force, sdf = x[:, :1], x[:, 1:4], x[:, 4:5]
    solid = (rho > .3).float()
    # Per-axis minimum solid cross-section; useful but deliberately low-capacity.
    mins = []
    for dim in (2, 3, 4):
        section = solid.mean(dim=tuple(d for d in (2, 3, 4) if d != dim))
        mins.append(section.amin(dim=2))
    # The operator input already normalizes force.  Use spatial means here so
    # descriptor magnitudes do not grow with the 64^3 discretization volume.
    force_sum = force.mean(dim=(2, 3, 4))
    force_abs = force.abs().mean(dim=(2, 3, 4))
    return torch.cat([
        rho.mean(dim=(2, 3, 4)), rho.std(dim=(2, 3, 4)),
        sdf.mean(dim=(2, 3, 4)), *mins, force_sum, force_abs,
    ], dim=1)


def make_loader(split: str, batch: int, shuffle: bool) -> DataLoader:
    # FEMDatasetV3 accepts the train/val layout.  For the sealed test, write a
    # temporary two-key view without changing the immutable manifest.
    manifest = OUT / "split_manifest.json"
    idx = json.loads(manifest.read_text())
    if split == "sealed_test":
        test_idx = OUT / "sealed_test_index.json"
        test_idx.write_text(json.dumps({"train": [], "val": idx["sealed_test"]}))
        manifest = test_idx
        split = "val"
    ds = FEMDatasetV3(index_file=str(manifest), split=split, augment=False, target_res=64)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4,
                      pin_memory=True, persistent_workers=True, collate_fn=collate_v3)


def train_scalar(kind: str, epochs: int, batch: int, lr: float) -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = make_loader("train", batch, True)
    val = make_loader("val", batch, False)
    model = GeometryMLP() if kind == "mlp" else ScalarCNN3D()
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state = float("inf"), None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for b in train:
            x = b["x"].to(device, non_blocking=True)
            y = torch.as_tensor(b["log_vm_max"], device=device).float()
            p = model(geometry_features(x) if kind == "mlp" else x)
            loss = torch.nn.functional.smooth_l1_loss(p, y)
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        model.eval(); errs = []
        with torch.no_grad():
            for b in val:
                x = b["x"].to(device, non_blocking=True)
                y = torch.as_tensor(b["log_vm_max"], device=device).float()
                p = model(geometry_features(x) if kind == "mlp" else x)
                errs.append(torch.abs(p-y).cpu())
        mae = torch.cat(errs).mean().item()
        history.append({"epoch": epoch, "cal_log_vm_mae": mae})
        if mae < best:
            best, best_state = mae, {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"{kind} epoch={epoch:03d} calibration_log_vm_mae={mae:.5f}", flush=True)
    dest = OUT / kind
    dest.mkdir(exist_ok=True)
    torch.save({"model_state_dict": best_state, "kind": kind, "params": sum(p.numel() for p in model.parameters()),
                "cal_log_vm_mae": best, "split_manifest": str(OUT / "split_manifest.json")}, dest / "best_model.pt")
    (dest / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "train-scalar"])
    ap.add_argument("--kind", choices=["mlp", "cnn"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    if a.action == "prepare":
        make_split()
    else:
        train_scalar(a.kind, a.epochs, a.batch, a.lr)
