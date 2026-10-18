#!/usr/bin/env python3
"""Boundary/ranking fine-tuning for raw v29-style PINO.

Preserves stress regression with Huber loss, while adding boundary weighting,
false-pass penalty, asymmetric pinball, and pairwise ranking on log-SF.
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
    p.add_argument("--scale_ckpt", required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=2901)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sf_target", type=float, default=1.5)
    p.add_argument("--huber_beta", type=float, default=0.176)
    p.add_argument("--pinball_tau", type=float, default=0.80)
    p.add_argument("--pinball_lambda", type=float, default=0.25)
    p.add_argument("--ranking_lambda", type=float, default=0.20)
    p.add_argument("--false_pass_weight", type=float, default=10.0)
    p.add_argument("--boundary_weight", type=float, default=4.0)
    p.add_argument("--unsafe_boundary_weight", type=float, default=6.0)
    p.add_argument("--selection", choices=["safety_first", "within_first"], default="safety_first")
    p.add_argument("--unfreeze_scale_branch", action="store_true", default=True)
    return p.parse_args()


def sampler_weights(dataset: ShadowArtScaleDataset, boundary_weight: float, unsafe_boundary_weight: float):
    w = []
    for i in range(len(dataset)):
        item = dataset[i]
        sf = float(item["sf"])
        wi = float(item["weight"])
        if 1.0 <= sf <= 2.0:
            wi *= boundary_weight
        if 0.8 <= sf < 1.5:
            wi *= unsafe_boundary_weight
        w.append(wi)
    return torch.tensor(w, dtype=torch.double)


def ranking_loss(pred_log_vm: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    # Predicted log-SF is log10(yield) - predicted log-stress.
    pred_log_sf = torch.log10(torch.tensor(YIELD_STRESS, device=pred_log_vm.device)) - pred_log_vm
    true_log_sf = torch.log10(torch.clamp(sf, min=1e-6))
    diff_true = true_log_sf[:, None] - true_log_sf[None, :]
    mask = torch.abs(diff_true) > 0.05
    if not torch.any(mask):
        return pred_log_vm.new_tensor(0.0)
    sign = torch.sign(diff_true[mask])
    diff_pred = (pred_log_sf[:, None] - pred_log_sf[None, :])[mask]
    return F.softplus(-sign * diff_pred).mean()


def save(path: Path, pino, args, epoch, val, test):
    torch.save(
        {
            "scale_head_v2_state_dict": pino.scale_head_v2.state_dict(),
            "scale_branch_state_dict": {
                name: getattr(pino, name).state_dict()
                for name in ["scale_lifting", "scale_down1", "scale_enc2", "scale_down2", "scale_bottleneck"]
            },
            "epoch": epoch,
            "val_metrics": val,
            "test_smoke_metrics": test,
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
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True),
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate)

    pino = load_model(args, device)
    params = [p for p in pino.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    val = evaluate(pino, val_loader, device, args.unfreeze_scale_branch, args.sf_target)
    test = evaluate(pino, test_loader, device, args.unfreeze_scale_branch, args.sf_target)
    best = best_key(val, args.selection)
    save(args.output_dir / "best_scale_sf.pt", pino, args, 0, val, test)
    history = [{"epoch": 0, "val": val, "test_smoke": test, "saved_best": True}]
    print(f"[init] val fp={val['false_pass']} ff={val['false_fail']} mae={val['log10_sf_mae']:.3f} | test fp={test['false_pass']} ff={test['false_fail']} mae={test['log10_sf_mae']:.3f}", flush=True)

    for ep in range(1, args.epochs + 1):
        pino.eval()
        t0 = time.time()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["log_vm"].to(device)
            sf = batch["sf"].to(device)
            weight = batch["weight"].to(device)
            pred = predict_log_vm(pino, x, train_branch=args.unfreeze_scale_branch)
            pred_sf = YIELD_STRESS / torch.pow(10.0, pred).clamp(min=1e-12)
            fp = ((pred_sf >= args.sf_target) & (sf < args.sf_target)).float()
            boundary = ((sf >= 1.0) & (sf <= 2.0)).float()
            unsafe_boundary = ((sf >= 0.8) & (sf < args.sf_target)).float()
            w = weight * (1.0 + boundary * (args.boundary_weight - 1.0) + unsafe_boundary * (args.unsafe_boundary_weight - 1.0))
            w = w * (1.0 + fp * args.false_pass_weight)
            huber = F.smooth_l1_loss(pred, target, beta=args.huber_beta, reduction="none")
            pin = pinball_loss(pred, target, args.pinball_tau)
            rank = ranking_loss(pred, sf)
            loss = ((huber + args.pinball_lambda * pin) * w).sum() / w.sum().clamp(min=1e-6)
            loss = loss + args.ranking_lambda * rank
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
            save(args.output_dir / "best_scale_sf.pt", pino, args, ep, val, test)
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), "val": val, "test_smoke": test, "saved_best": improved, "elapsed_s": time.time() - t0}
        history.append(row)
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"E{ep:03d}/{args.epochs} L={row['train_loss']:.4f} | val fp={val['false_pass']} ff={val['false_fail']} mae={val['log10_sf_mae']:.3f} | test fp={test['false_pass']} ff={test['false_fail']} mae={test['log10_sf_mae']:.3f} {'BEST' if improved else ''}", flush=True)


if __name__ == "__main__":
    main()
