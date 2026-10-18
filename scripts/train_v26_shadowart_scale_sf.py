#!/usr/bin/env python3
"""Fine-tune PINO scale/SF head on grouped ShadowArt FEM hard-set."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path("/home/lin/xvxinyao/total_project")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural.models.load_model import load_surrogate
from structural.training.pretrain_pino_v3 import FEMDatasetV3

YIELD_STRESS = 250e6
_SCALE_TRAIN_RES = 64  # unify mixed-source resolutions (dense=48, thin=64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index_file", type=Path, default=ROOT / "results/shadowart_fem_hardset_v1/index.json")
    p.add_argument("--base_ckpt", default=str(ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"))
    p.add_argument("--scale_ckpt", default=str(ROOT / "checkpoints/scale_head_v2_weighted/best_scale_head_v2_weighted.pt"))
    p.add_argument("--output_dir", type=Path, default=ROOT / "checkpoints/v26_shadowart_scale_sf")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2602)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--huber_beta", type=float, default=0.176)
    p.add_argument("--near_failure_weight", type=float, default=2.5)
    p.add_argument("--base_load_weight", type=float, default=1.5)
    p.add_argument(
        "--under_vm_weight",
        type=float,
        default=4.0,
        help="Extra penalty when pred log_vm < target log_vm (dangerous SF overestimate).",
    )
    p.add_argument(
        "--false_pass_weight",
        type=float,
        default=6.0,
        help="Extra penalty for samples that would become false PASS at sf_target.",
    )
    p.add_argument(
        "--selection",
        choices=["within_first", "safety_first"],
        default="safety_first",
        help="Best checkpoint rule; safety_first minimizes false PASS before accuracy.",
    )
    p.add_argument("--unfreeze_scale_branch", action="store_true")
    p.add_argument("--sf_target", type=float, default=1.5)
    p.add_argument("--success_log_mae", type=float, default=0.176)
    p.add_argument("--success_within_rate", type=float, default=0.80)
    return p.parse_args()


def load_density(density: np.ndarray) -> np.ndarray:
    if density.ndim == 4 and density.shape[0] == 1:
        density = density[0]
    if density.ndim == 5:
        density = density[0, 0]
    return np.asarray(density, dtype=np.float32)


def build_x(density: np.ndarray, force: np.ndarray) -> torch.Tensor:
    density = load_density(density)
    force = np.asarray(force, dtype=np.float32)
    solid = density > FEMDatasetV3.SOLID_THRESHOLD
    if solid.any() and (~solid).any():
        sdf = (distance_transform_edt(solid) - distance_transform_edt(~solid)).astype(np.float32)
        sdf = sdf / (density.shape[0] / 2)
    else:
        sdf = np.zeros_like(density, dtype=np.float32)
    force_norm = force / FEMDatasetV3.FORCE_SCALE
    x_np = np.concatenate([density[None], force_norm, sdf[None]], axis=0)
    return torch.from_numpy(x_np).float()


class ShadowArtScaleDataset(Dataset):
    def __init__(
        self,
        index_file: Path,
        split: str,
        near_failure_weight: float = 2.5,
        base_load_weight: float = 1.5,
    ):
        self.root = index_file.resolve().parent
        idx = json.loads(index_file.read_text())
        self.records = list(idx[split])
        self.split = split
        self.near_failure_weight = near_failure_weight
        self.base_load_weight = base_load_weight
        if not self.records:
            raise ValueError(f"No records for split={split} in {index_file}")
        print(f"ShadowArtScaleDataset[{split}]: {len(self.records)} samples")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        with np.load(self.root / rec["path"], allow_pickle=False) as data:
            x = build_x(data["density"], data["force"])
            if x.shape[-1] != _SCALE_TRAIN_RES:
                x = F.interpolate(
                    x.unsqueeze(0), size=(_SCALE_TRAIN_RES,) * 3,
                    mode="trilinear", align_corners=False,
                ).squeeze(0)
            sf = float(data["sf"]) if "sf" in data else float(rec["sf"])
            log_vm = float(data["log_vm_max"]) if "log_vm_max" in data else float(rec["log_vm_max"])
            multiplier = float(data["load_multiplier"]) if "load_multiplier" in data else float(rec["load_multiplier"])
        weight = 1.0
        if sf <= 10.0:
            weight *= self.near_failure_weight
        if abs(multiplier - 1.0) < 1e-6:
            weight *= self.base_load_weight
        return {
            "x": x,
            "log_vm": torch.tensor(log_vm, dtype=torch.float32),
            "sf": torch.tensor(sf, dtype=torch.float32),
            "weight": torch.tensor(weight, dtype=torch.float32),
            "case_id": int(rec["case_id"]),
            "is_base_load": bool(rec.get("is_base_load", False)),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "log_vm": torch.stack([b["log_vm"] for b in batch]),
        "sf": torch.stack([b["sf"] for b in batch]),
        "weight": torch.stack([b["weight"] for b in batch]),
        "case_id": [b["case_id"] for b in batch],
        "is_base_load": [b["is_base_load"] for b in batch],
    }


def load_model(args: argparse.Namespace, device: torch.device):
    pino, _ = load_surrogate(args.base_ckpt, device=device)
    pino.enable_enhanced_scale_head()
    if args.scale_ckpt and Path(args.scale_ckpt).exists():
        ckpt = torch.load(args.scale_ckpt, map_location=device, weights_only=False)
        pino.scale_head_v2.load_state_dict(ckpt["scale_head_v2_state_dict"])
        if ckpt.get("scale_branch_state_dict"):
            for name, state in ckpt["scale_branch_state_dict"].items():
                getattr(pino, name).load_state_dict(state)
    for p in pino.parameters():
        p.requires_grad_(False)
    for p in pino.scale_head_v2.parameters():
        p.requires_grad_(True)
    if args.unfreeze_scale_branch:
        for name in ["scale_lifting", "scale_down1", "scale_enc2", "scale_down2", "scale_bottleneck"]:
            for p in getattr(pino, name).parameters():
                p.requires_grad_(True)
    return pino


def predict_log_vm(pino, x: torch.Tensor, train_branch: bool) -> torch.Tensor:
    b, _, nx, ny, nz = x.shape
    grid = pino._get_grid((nx, ny, nz), x.device).expand(b, -1, -1, -1, -1)
    xg = torch.cat([x, grid], dim=1)
    if train_branch:
        return pino._scale_forward(xg).squeeze(-1)
    with torch.no_grad():
        h = pino.scale_lifting(xg)
        h = pino.scale_down1(h)
        for block in pino.scale_enc2:
            h = block(h)
        h = pino.scale_down2(h)
        for block in pino.scale_bottleneck:
            h = block(h)
        struct_feat = pino._compute_structural_features(x[:, :1])
        pool1 = F.adaptive_avg_pool3d(h, 1).flatten(1)
        pool4 = pino.scale_head_v2_pool4(h).flatten(1)
    combined = torch.cat([pool1, pool4, struct_feat], dim=1)
    return pino.scale_head_v2(combined).squeeze(-1)


def metrics_from_arrays(pred_log: np.ndarray, true_log: np.ndarray, true_sf: np.ndarray, sf_target: float) -> dict:
    pred_sf = YIELD_STRESS / np.maximum(10.0 ** pred_log, 1e-12)
    ratio = pred_sf / np.maximum(true_sf, 1e-12)
    log_err = np.log10(np.maximum(pred_sf, 1e-12)) - np.log10(np.maximum(true_sf, 1e-12))
    pred_pass = pred_sf >= sf_target
    true_pass = true_sf >= sf_target
    return {
        "n": int(len(true_sf)),
        "sf_ratio_mean": float(ratio.mean()),
        "sf_ratio_median": float(np.median(ratio)),
        "within_0p5x_rate": float(((ratio >= 0.5) & (ratio <= 1.5)).mean()),
        "log10_sf_mae": float(np.abs(log_err).mean()),
        "log10_sf_bias": float(log_err.mean()),
        "false_pass": int((pred_pass & ~true_pass).sum()),
        "false_fail": int((~pred_pass & true_pass).sum()),
    }


def best_key(metrics: dict, mode: str) -> tuple:
    """Higher is better."""
    if mode == "safety_first":
        near = metrics.get("near_failure", {})
        return (
            -metrics["false_pass"],
            -near.get("false_pass", metrics["false_pass"]),
            metrics["within_0p5x_rate"],
            -metrics["log10_sf_mae"],
            -metrics["false_fail"],
        )
    return (
        metrics["within_0p5x_rate"],
        -metrics["log10_sf_mae"],
        -metrics["false_pass"],
    )


@torch.no_grad()
def evaluate(pino, loader, device: torch.device, train_branch: bool, sf_target: float) -> dict:
    pino.eval()
    pred_logs, true_logs, true_sfs, base_mask = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        pred = predict_log_vm(pino, x, train_branch=train_branch).detach().cpu().numpy()
        pred_logs.extend(pred.tolist())
        true_logs.extend(batch["log_vm"].numpy().tolist())
        true_sfs.extend(batch["sf"].numpy().tolist())
        base_mask.extend(batch["is_base_load"])
    pred_logs = np.asarray(pred_logs)
    true_logs = np.asarray(true_logs)
    true_sfs = np.asarray(true_sfs)
    base_mask = np.asarray(base_mask, dtype=bool)
    out = metrics_from_arrays(pred_logs, true_logs, true_sfs, sf_target)
    out["vm_log_mae"] = float(np.abs(pred_logs - true_logs).mean())
    if base_mask.any():
        out["base_load"] = metrics_from_arrays(pred_logs[base_mask], true_logs[base_mask], true_sfs[base_mask], sf_target)
    nf_mask = true_sfs <= 5.0
    if nf_mask.any():
        out["near_failure"] = metrics_from_arrays(pred_logs[nf_mask], true_logs[nf_mask], true_sfs[nf_mask], sf_target)
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_set = ShadowArtScaleDataset(
        args.index_file, "train",
        near_failure_weight=args.near_failure_weight,
        base_load_weight=args.base_load_weight,
    )
    val_set = ShadowArtScaleDataset(args.index_file, "val")
    test_set = ShadowArtScaleDataset(args.index_file, "test")
    sample_weights = torch.tensor([train_set[i]["weight"].item() for i in range(len(train_set))], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    pino = load_model(args, device)
    params = [p for p in pino.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    current_best_key = None
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        pino.eval()
        t0 = time.time()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["log_vm"].to(device)
            true_sf = batch["sf"].to(device)
            weight = batch["weight"].to(device)
            pred = predict_log_vm(pino, x, train_branch=args.unfreeze_scale_branch)
            raw = F.smooth_l1_loss(pred, target, beta=args.huber_beta, reduction="none")
            dangerous_under_vm = (pred < target).float()
            pred_sf = YIELD_STRESS / torch.pow(10.0, pred).clamp(min=1e-12)
            false_pass = ((pred_sf >= args.sf_target) & (true_sf < args.sf_target)).float()
            safety_weight = 1.0 + dangerous_under_vm * (args.under_vm_weight - 1.0)
            safety_weight = safety_weight + false_pass * args.false_pass_weight
            weight = weight * safety_weight
            loss = (raw * weight).sum() / weight.sum().clamp(min=1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_metrics = evaluate(pino, val_loader, device, args.unfreeze_scale_branch, args.sf_target)
        test_smoke = evaluate(pino, test_loader, device, args.unfreeze_scale_branch, args.sf_target)
        key = best_key(val_metrics, args.selection)
        improved = current_best_key is None or key > current_best_key
        if improved:
            current_best_key = key
            best_metrics = val_metrics
            ckpt = {
                "scale_head_v2_state_dict": pino.scale_head_v2.state_dict(),
                "scale_branch_state_dict": {
                    name: getattr(pino, name).state_dict()
                    for name in ["scale_lifting", "scale_down1", "scale_enc2", "scale_down2", "scale_bottleneck"]
                } if args.unfreeze_scale_branch else None,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "test_smoke_metrics": test_smoke,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            }
            torch.save(ckpt, args.output_dir / "best_scale_sf.pt")
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val": val_metrics,
            "test_smoke": test_smoke,
            "elapsed_s": time.time() - t0,
            "lr": scheduler.get_last_lr()[0],
            "saved_best": improved,
        }
        history.append(row)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(
            f"E{epoch:03d}/{args.epochs} loss={row['train_loss']:.4f} "
            f"val within={val_metrics['within_0p5x_rate']:.3f} "
            f"logMAE={val_metrics['log10_sf_mae']:.3f} falsePASS={val_metrics['false_pass']} "
            f"test within={test_smoke['within_0p5x_rate']:.3f} "
            f"{'BEST' if improved else ''}",
            flush=True,
        )

    final_report = {
        "best_val_metrics": best_metrics,
        "acceptance": {
            "target_within_0p5x_rate": args.success_within_rate,
            "target_log10_sf_mae": args.success_log_mae,
            "requires_test_eval": True,
        },
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
