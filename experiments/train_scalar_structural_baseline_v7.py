#!/usr/bin/env python3
"""Train scalar structural baselines on the task-disjoint delivered v7 index.

The baselines predict the same FEM ``log_vm_max`` supervision used by the PINO
scale head.  They intentionally do not predict dense fields; the field quality
comparison belongs to the FNO/PINO runs launched by the companion shell file.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structural.training.pretrain_pino_v3 import FEMDatasetV3, collate_v3


class GeometryMLP(nn.Module):
    def __init__(self, features: int = 13) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, 128), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)


class ScalarCNN3D(nn.Module):
    def __init__(self, channels: int = 6) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(channels, 16, 5, stride=2, padding=2), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv3d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(1)


def geometry_features(x: torch.Tensor) -> torch.Tensor:
    """Low-capacity geometric descriptors from the same structural proxy."""
    rho, force, sdf = x[:, :1], x[:, 1:4], x[:, 4:5]
    fixed = x[:, 5:6] if x.shape[1] > 5 else torch.zeros_like(rho)
    solid = (rho > 0.3).float()
    minima = []
    for axis in (2, 3, 4):
        section = solid.mean(dim=tuple(d for d in (2, 3, 4) if d != axis))
        minima.append(section.amin(dim=2))
    force_mean = force.mean(dim=(2, 3, 4))
    force_abs = force.abs().mean(dim=(2, 3, 4))
    return torch.cat([
        rho.mean(dim=(2, 3, 4)), rho.std(dim=(2, 3, 4)),
        sdf.mean(dim=(2, 3, 4)), fixed.mean(dim=(2, 3, 4)),
        *minima, force_mean, force_abs,
    ], dim=1)


def sealed_manifest(index: Path, destination: Path) -> Path:
    data = json.loads(index.read_text())
    rows = [r for r in data["rows"] if r["split"] == "sealed_test"]
    destination.write_text(json.dumps({"train": [], "val": rows}, indent=2))
    return destination


def loader(index: Path, split: str, batch: int, target_res: int) -> DataLoader:
    source = index
    dataset_split = split
    if split == "sealed_test":
        source = sealed_manifest(index, index.parent / ".pino_v7_sealed_test_view.json")
        dataset_split = "val"
    dataset = FEMDatasetV3(index_file=str(source), split=dataset_split, augment=False,
                           target_res=target_res, include_fixed_mask=True)
    return DataLoader(dataset, batch_size=batch, shuffle=(split == "train"), num_workers=4,
                      pin_memory=True, persistent_workers=True, collate_fn=collate_v3)


def evaluate(model: nn.Module, data: DataLoader, kind: str, device: torch.device) -> dict:
    model.eval()
    predictions, targets = [], []
    elapsed = 0.0
    with torch.no_grad():
        for batch in data:
            x = batch["x"].to(device, non_blocking=True)
            y = torch.as_tensor(batch["log_vm_max"], device=device).float()
            start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if start is not None:
                start.record()
            pred = model(geometry_features(x) if kind == "mlp" else x)
            if end is not None:
                end.record(); torch.cuda.synchronize(); elapsed += start.elapsed_time(end) / 1000.0
            predictions.append(pred.cpu()); targets.append(y.cpu())
    p, y = torch.cat(predictions), torch.cat(targets)
    return {
        "n": int(y.numel()),
        "log_vm_mae": float(torch.abs(p - y).mean()),
        "log_vm_rmse": float(torch.sqrt(torch.mean((p - y) ** 2))),
        "inference_s_per_sample": elapsed / max(int(y.numel()), 1) if device.type == "cuda" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["mlp", "cnn"], required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-res", type=int, default=48, choices=[48])
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train = loader(args.index, "train", args.batch_size, args.target_res)
    calibration = loader(args.index, "calibration", args.batch_size, args.target_res)
    model = GeometryMLP() if args.kind == "mlp" else ScalarCNN3D()
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_mae, best_state, history = float("inf"), None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train:
            x = batch["x"].to(device, non_blocking=True)
            y = torch.as_tensor(batch["log_vm_max"], device=device).float()
            pred = model(geometry_features(x) if args.kind == "mlp" else x)
            loss = torch.nn.functional.smooth_l1_loss(pred, y)
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        result = evaluate(model, calibration, args.kind, device)
        history.append({"epoch": epoch, **result})
        print(f"{args.kind} epoch={epoch:03d} calibration_log_vm_mae={result['log_vm_mae']:.5f}", flush=True)
        if result["log_vm_mae"] < best_mae:
            best_mae = result["log_vm_mae"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": args.kind, "state_dict": best_state, "target_res": args.target_res,
                "index": str(args.index.resolve())}, args.output / "best_model.pt")
    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    (args.output / "protocol.json").write_text(json.dumps({
        "input": "same delivered C48 structural proxy as PINO", "target": "FEM log10 peak von-Mises stress",
        "split": "task-disjoint source index", "calibration": "calibration split only",
    }, indent=2))


if __name__ == "__main__":
    main()
