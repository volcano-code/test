#!/usr/bin/env python3
"""Boundary fine-tuning with an old-ensemble anchor.

The previous boundary-only run showed lower ACS autonomous rate after adding
2k boundary FEM samples. This variant keeps the hard FEM supervision while
distilling old-distribution samples toward the established raw v29 ensemble.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

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


OLD_TEACHERS = [
    ROOT / "checkpoints/v29_v27_pinball/best_scale_sf.pt",
    ROOT / "checkpoints/v29b_v27_pinball_s2605/best_scale_sf.pt",
    ROOT / "checkpoints/v29c_v27_pinball_s2606/best_scale_sf.pt",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--index_file", type=Path, default=Path("/home/lin/xvxinyao/ShadowForge/results/boundary_fem_yield250_20260624/index.json"))
    p.add_argument("--base_ckpt", default=str(ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"))
    p.add_argument("--scale_ckpt", default=str(ROOT / "checkpoints/v29_v27_pinball/best_scale_sf.pt"))
    p.add_argument("--teacher_ckpts", nargs="*", default=[str(p) for p in OLD_TEACHERS])
    p.add_argument("--teacher_cache", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=4e-6)
    p.add_argument("--seed", type=int, default=3301)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sf_target", type=float, default=1.5)
    p.add_argument("--pinball_tau", type=float, default=0.78)
    p.add_argument("--pinball_lambda", type=float, default=0.18)
    p.add_argument("--huber_beta", type=float, default=0.176)
    p.add_argument("--under_vm_weight", type=float, default=4.0)
    p.add_argument("--false_pass_weight", type=float, default=8.0)
    p.add_argument("--boundary_weight", type=float, default=2.0)
    p.add_argument("--unsafe_boundary_weight", type=float, default=3.0)
    p.add_argument("--old_distill_lambda", type=float, default=0.45)
    p.add_argument("--boundary_distill_lambda", type=float, default=0.05)
    p.add_argument("--selection", choices=["safety_first", "within_first"], default="safety_first")
    p.add_argument("--unfreeze_scale_branch", action="store_true", default=True)
    return p.parse_args()


class AnchoredDataset(ShadowArtScaleDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_log = None
        self.is_boundary = np.asarray(
            [
                ("boundary_fem_yield250_20260624" in str(r.get("path", "")))
                or ("aug_boundary" in str(r.get("path", "")))
                for r in self.records
            ],
            dtype=bool,
        )

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item["sample_idx"] = int(idx)
        item["is_boundary"] = bool(self.is_boundary[idx])
        if self.teacher_log is not None:
            item["teacher_log"] = torch.tensor(float(self.teacher_log[idx]), dtype=torch.float32)
        return item


def anchor_collate(batch: list[dict]) -> dict:
    out = collate(batch)
    out["sample_idx"] = torch.tensor([b["sample_idx"] for b in batch], dtype=torch.long)
    out["is_boundary"] = torch.tensor([b["is_boundary"] for b in batch], dtype=torch.bool)
    if "teacher_log" in batch[0]:
        out["teacher_log"] = torch.stack([b["teacher_log"] for b in batch])
    return out


def sampler_weights(dataset: AnchoredDataset, boundary_weight: float, unsafe_boundary_weight: float):
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


@torch.no_grad()
def precompute_teacher_logs(args, dataset: AnchoredDataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=anchor_collate)
    preds = []
    for ckpt in args.teacher_ckpts:
        teacher_args = SimpleNamespace(**vars(args))
        teacher_args.scale_ckpt = ckpt
        teacher = load_model(teacher_args, device)
        teacher.eval()
        one = []
        for batch in loader:
            pred = predict_log_vm(teacher, batch["x"].to(device), train_branch=args.unfreeze_scale_branch)
            one.append(pred.detach().cpu().numpy())
        preds.append(np.concatenate(one, axis=0))
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.stack(preds, axis=0).mean(axis=0).astype(np.float32)


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

    train_set = AnchoredDataset(args.index_file, "train", near_failure_weight=2.5, base_load_weight=1.5)
    val_set = AnchoredDataset(args.index_file, "val")
    test_set = AnchoredDataset(args.index_file, "test")
    teacher_cache = args.teacher_cache or (args.output_dir / "teacher_train_log.npy")
    teacher_cache.parent.mkdir(parents=True, exist_ok=True)
    if teacher_cache.exists():
        train_set.teacher_log = np.load(teacher_cache)
    else:
        train_set.teacher_log = precompute_teacher_logs(args, train_set, device)
        np.save(teacher_cache, train_set.teacher_log)

    weights = sampler_weights(train_set, args.boundary_weight, args.unsafe_boundary_weight)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=0, collate_fn=anchor_collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=anchor_collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=anchor_collate)

    pino = load_model(args, device)
    params = [p for p in pino.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    v0 = evaluate(pino, val_loader, device, args.unfreeze_scale_branch, args.sf_target)
    t0 = evaluate(pino, test_loader, device, args.unfreeze_scale_branch, args.sf_target)
    best = best_key(v0, args.selection)
    save_ckpt(args.output_dir / "best_scale_sf.pt", pino, args, 0, v0, t0)
    history = [{"epoch": 0, "val": v0, "test_smoke": t0, "saved_best": True}]
    print(
        f"[init] val fp={v0['false_pass']} ff={v0['false_fail']} mae={v0['log10_sf_mae']:.3f} "
        f"within={v0['within_0p5x_rate']:.3f} | test fp={t0['false_pass']} ff={t0['false_fail']} "
        f"mae={t0['log10_sf_mae']:.3f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        pino.eval()
        t_start = time.time()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["log_vm"].to(device)
            sf = batch["sf"].to(device)
            base_w = batch["weight"].to(device)
            teacher = batch["teacher_log"].to(device)
            is_boundary = batch["is_boundary"].to(device)

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
            fem_loss = huber + args.pinball_lambda * pin
            distill_lambda = torch.where(
                is_boundary,
                torch.full_like(pred, args.boundary_distill_lambda),
                torch.full_like(pred, args.old_distill_lambda),
            )
            distill = F.smooth_l1_loss(pred, teacher, beta=args.huber_beta, reduction="none")
            loss = ((fem_loss + distill_lambda * distill) * w).sum() / w.sum().clamp(min=1e-6)
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
        print(
            f"E{epoch:03d}/{args.epochs} L={row['train_loss']:.4f} | val fp={val['false_pass']} "
            f"ff={val['false_fail']} mae={val['log10_sf_mae']:.3f} within={val['within_0p5x_rate']:.3f} | "
            f"test fp={test['false_pass']} ff={test['false_fail']} mae={test['log10_sf_mae']:.3f} "
            f"{'BEST' if improved else ''}",
            flush=True,
        )


if __name__ == "__main__":
    main()
