#!/usr/bin/env python3
"""Boundary-focused raw v29-style PINO fine-tuning.

Keeps the old ShadowArt-Hard train/val/test protocol, avoids isotonic in the
objective, and emphasizes the SF boundary that controls ACS/FEM fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path("/data/xvxinyao/xvxinyao/total_project")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune_v29_pinball import pinball_loss
from scripts.train_v26_shadowart_scale_sf import (
    ShadowArtScaleDataset,
    YIELD_STRESS,
    best_key,
    collate,
    evaluate,
    load_model,
    predict_log_vm,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index_file", type=Path, default=ROOT / "results/shadowart_fem_hardset_v1/index.json")
    p.add_argument("--base_ckpt", default=str(ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"))
    p.add_argument("--scale_ckpt", default=str(ROOT / "checkpoints/v27_shadowart_scale_sf_safety/best_scale_sf.pt"))
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--seed", type=int, default=2801)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sf_target", type=float, default=1.5)
    p.add_argument("--pinball_tau", type=float, default=0.84)
    p.add_argument("--pinball_lambda", type=float, default=0.7)
    p.add_argument("--huber_beta", type=float, default=0.176)
    p.add_argument("--under_vm_weight", type=float, default=7.0)
    p.add_argument("--false_pass_weight", type=float, default=16.0)
    p.add_argument("--boundary_weight", type=float, default=5.0)
    p.add_argument("--unsafe_boundary_weight", type=float, default=7.0)
    p.add_argument("--selection", choices=["safety_first", "within_first"], default="safety_first")
    p.add_argument("--unfreeze_scale_branch", action="store_true", default=True)
    return p.parse_args()


def sampler_weights(dataset: ShadowArtScaleDataset, boundary_weight: float, unsafe_boundary_weight: float):
    weights = []
    for i in range(len(dataset)):
        sf = float(dataset[i]["sf"])
        w = float(dataset[i]["weight"])
        if 1.0 <= sf <= 2.0:
            w *= boundary_weight
        if 0.8 <= sf < 1.5:
            w *= unsafe_boundary_weight
        weights.append(w)
    return torch.tensor(weights, dtype=torch.double)


def save_ckpt(path: Path, pino, args, epoch, val_metrics, test_metrics):
    torch.save(
        {
            "scale_head_v2_state_dict": pino.scale_head_v2.state_dict(),
            "scale_branch_state_dict": {
                name: getattr(pino, name).state_dict()
                for name in ["scale_lifting", "scale_down1", "scale_enc2", "scale_down2", "scale_bottleneck"]
            },
            "epoch": epoch,
            "val_metrics": val_metrics,
            "test_smoke_metrics": test_metrics,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        path,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_set = ShadowArtScaleDataset(args.index_file, "train", near_failure_weight=2.5, base_load_weight=1.5)
    val_set = ShadowArtScaleDataset(args.index_file, "val")
    test_set = ShadowArtScaleDataset(args.index_file, "test")
    weights = sampler_weights(train_set, args.boundary_weight, args.unsafe_boundary_weight)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    pino = load_model(args, device)
    params = [p for p in pino.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    v0 = evaluate(pino, val_loader, device, args.unfreeze_scale_branch, args.sf_target)
    t0 = evaluate(pino, test_loader, device, args.unfreeze_scale_branch, args.sf_target)
    best = best_key(v0, args.selection)
    save_ckpt(args.output_dir / "best_scale_sf.pt", pino, args, 0, v0, t0)
    history = [{"epoch": 0, "val": v0, "test_smoke": t0, "saved_best": True}]
    print(f"[init] val fp={v0['false_pass']} ff={v0['false_fail']} mae={v0['log10_sf_mae']:.3f} within={v0['within_0p5x_rate']:.3f} | test fp={t0['false_pass']} ff={t0['false_fail']} mae={t0['log10_sf_mae']:.3f}", flush=True)

    for epoch in range(1, args.epochs + 1):
        pino.eval()
        t_start = time.time()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["log_vm"].to(device)
            sf = batch["sf"].to(device)
            base_w = batch["weight"].to(device)
            pred = predict_log_vm(pino, x, train_branch=args.unfreeze_scale_branch)
            huber = F.smooth_l1_loss(pred, y, beta=args.huber_beta, reduction="none")
            pred_sf = YIELD_STRESS / torch.pow(10.0, pred).clamp(min=1e-12)
            false_pass = ((pred_sf >= args.sf_target) & (sf < args.sf_target)).float()
            under_vm = (pred < y).float()
            boundary = ((sf >= 1.0) & (sf <= 2.0)).float()
            unsafe_boundary = ((sf >= 0.8) & (sf < args.sf_target)).float()
            w = base_w * (1.0 + boundary * (args.boundary_weight - 1.0) + unsafe_boundary * (args.unsafe_boundary_weight - 1.0))
            w = w * (1.0 + under_vm * (args.under_vm_weight - 1.0) + false_pass * args.false_pass_weight)
            pin = pinball_loss(pred, y, args.pinball_tau)
            loss = ((huber + args.pinball_lambda * pin) * w).sum() / w.sum().clamp(min=1e-6)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()

        val = evaluate(pino, val_loader, device, args.unfreeze_scale_branch, args.sf_target)
        test = evaluate(pino, test_loader, device, args.unfreeze_scale_branch, args.sf_target)
        key = best_key(val, args.selection)
        improved = key > best
        if improved:
            best = key
            save_ckpt(args.output_dir / "best_scale_sf.pt", pino, args, epoch, val, test)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val": val,
            "test_smoke": test,
            "saved_best": improved,
            "elapsed_s": time.time() - t_start,
        }
        history.append(row)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"E{epoch:03d}/{args.epochs} L={row['train_loss']:.4f} | val fp={val['false_pass']} ff={val['false_fail']} mae={val['log10_sf_mae']:.3f} within={val['within_0p5x_rate']:.3f} | test fp={test['false_pass']} ff={test['false_fail']} mae={test['log10_sf_mae']:.3f} {'BEST' if improved else ''}", flush=True)


if __name__ == "__main__":
    main()
