#!/usr/bin/env python3
"""Lightweight PINO hard-negative fine-tuning on current ShadowArt FEM aug data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_v26_shadowart_scale_sf import build_x
from structural.analysis.load_generator import LoadGenerator
from structural.models.load_model import load_surrogate

YIELD_STRESS = 50e6
TAU = 1.3
SCALE_BRANCH = ["scale_lifting", "scale_down1", "scale_enc2", "scale_down2", "scale_bottleneck"]


def load_geoattn_model(device: torch.device, base_ckpt: Path, geo_ckpt: Path):
    spec = importlib.util.spec_from_file_location("geoattn_head", ROOT / "certification/02_geoattn_head.py")
    geo = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(geo)

    model, _ = load_surrogate(str(base_ckpt), device)
    model.enable_enhanced_scale_head()
    ckpt = torch.load(geo_ckpt, map_location=device, weights_only=False)
    for name, state in ckpt["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(state)
    head = geo.patch_pino_with_attn(model, attn_dim=ckpt.get("attn_dim", 64), n_heads=ckpt.get("n_heads", 4))
    head.load_state_dict(ckpt["attn_scale_head_state_dict"])
    return model, head, ckpt


def load_aug_rows(paths: list[Path], seed: int):
    rows = []
    lg = LoadGenerator()
    for path in paths:
        data = np.load(path)
        tag = path.stem
        for i in range(len(data["dens"])):
            density = data["dens"][i].astype(np.float32)
            force = lg.standing_load(density)["force"] * float(data["mult"][i])
            rows.append(
                {
                    "source": tag,
                    "index": i,
                    "density": density,
                    "force": force.astype(np.float32),
                    "vm": float(data["vm"][i]),
                    "sf": float(data["sf"][i]),
                }
            )
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    return rows


def xg(model, density: np.ndarray, force: np.ndarray, device: torch.device):
    x = build_x(density.astype(np.float32), force.astype(np.float32)).unsqueeze(0)
    b, _, nx, ny, nz = x.shape
    grid = model._get_grid((nx, ny, nz), "cpu").expand(b, -1, -1, -1, -1)
    return torch.cat([x, grid], 1).to(device)


def predict_log_vm(model, rows: list[dict], device: torch.device, batch_size: int):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            xb = torch.cat([xg(model, r["density"], r["force"], device) for r in batch], 0)
            pred = model._scale_forward(xb).squeeze(-1)
            out.extend(float(x) for x in pred.detach().cpu().reshape(-1))
    return np.asarray(out, dtype=float)


def metrics(pred_log_vm: np.ndarray, rows: list[dict]):
    true_vm = np.asarray([r["vm"] for r in rows], dtype=float)
    true_log_vm = np.log10(np.maximum(true_vm, 1e-12))
    true_sf = np.asarray([r["sf"] for r in rows], dtype=float)
    pred_sf = YIELD_STRESS / np.maximum(10 ** pred_log_vm, 1e-12)
    pred_safe = pred_sf >= TAU
    true_safe = true_sf >= TAU
    false_pass = int(np.sum(pred_safe & ~true_safe))
    true_pass = int(np.sum(pred_safe & true_safe))
    selected = int(np.sum(pred_safe))
    return {
        "n": int(len(rows)),
        "log_vm_mae": float(np.mean(np.abs(pred_log_vm - true_log_vm))),
        "log_vm_corr": float(np.corrcoef(pred_log_vm, true_log_vm)[0, 1]) if len(rows) > 1 else None,
        "raw_safe_selected": selected,
        "raw_safe_precision": float(true_pass / selected) if selected else None,
        "raw_safe_coverage": float(true_pass / max(int(np.sum(true_safe)), 1)),
        "false_pass_count": false_pass,
        "false_pass_rate_selected": float(false_pass / selected) if selected else None,
        "true_safe_count": int(np.sum(true_safe)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--geo_ckpt", type=Path, default=ROOT / "checkpoints/v40_geoattn/best.pt")
    parser.add_argument("--output_ckpt", type=Path, default=ROOT / "checkpoints/pino_aug_hardneg_20260624/scale_geoattn.pt")
    parser.add_argument("--output_json", type=Path, default=ROOT / "results/pino_aug_hardneg_training_20260624.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = load_aug_rows([ROOT / "data/aug.npz", ROOT / "data/aug_fixed.npz"], args.seed)
    n_train = int(len(rows) * 0.8)
    train_rows, test_rows = rows[:n_train], rows[n_train:]

    model, head, geo_ckpt = load_geoattn_model(device, args.base_ckpt, args.geo_ckpt)
    before = metrics(predict_log_vm(model, test_rows, device, args.batch_size), test_rows)

    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for name in SCALE_BRANCH:
        for p in getattr(model, name).parameters():
            p.requires_grad_(True)
            params.append(p)
    for p in head.parameters():
        p.requires_grad_(True)
        params.append(p)

    target = torch.tensor([np.log10(max(r["vm"], 1e-12)) for r in train_rows], dtype=torch.float32, device=device)
    unsafe = torch.tensor([r["sf"] < TAU for r in train_rows], dtype=torch.bool, device=device)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    history = []
    rng = np.random.default_rng(args.seed)

    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_rows))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            batch = [train_rows[i] for i in ids]
            xb = torch.cat([xg(model, r["density"], r["force"], device) for r in batch], 0)
            y = target[torch.as_tensor(ids, device=device)]
            is_unsafe = unsafe[torch.as_tensor(ids, device=device)]
            pred = model._scale_forward(xb).squeeze(-1).reshape(-1)
            base_loss = F.smooth_l1_loss(pred, y, reduction="none")
            under_vm = F.relu(y - pred)
            pred_sf = YIELD_STRESS / torch.clamp(10 ** pred, min=1e-12)
            false_pass_penalty = torch.where(is_unsafe & (pred_sf >= TAU), under_vm * 8.0, torch.zeros_like(under_vm))
            near_weight = torch.where(torch.abs(torch.log10(torch.tensor(TAU, device=device)) - (torch.log10(torch.tensor(YIELD_STRESS, device=device)) - y)) < 0.25, 2.5, 1.0)
            loss = torch.mean(base_loss * near_weight + 4.0 * under_vm + false_pass_penalty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(ids)
        if ep == 0 or ep == args.epochs - 1 or (ep + 1) % 4 == 0:
            pred_test = predict_log_vm(model, test_rows, device, args.batch_size)
            row = {"epoch": ep + 1, "train_loss": total / len(train_rows), "test": metrics(pred_test, test_rows)}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    after = metrics(predict_log_vm(model, test_rows, device, args.batch_size), test_rows)
    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scale_branch_state_dict": {name: getattr(model, name).state_dict() for name in SCALE_BRANCH},
            "attn_scale_head_state_dict": head.state_dict(),
            "attn_dim": geo_ckpt.get("attn_dim", 64),
            "n_heads": geo_ckpt.get("n_heads", 4),
            "base_ckpt": str(args.base_ckpt),
            "source": "aug+aug_fixed hard-negative fine-tuning",
        },
        args.output_ckpt,
    )
    result = {
        "schema_version": "pino_aug_hardneg_training_v1",
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "before": before,
        "after": after,
        "history": history,
        "checkpoint": str(args.output_ckpt),
    }
    args.output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
