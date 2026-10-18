#!/usr/bin/env python3
"""Safe-side PINO fine-tuning with candidate-domain and boundary FEM data.

This experiment fine-tunes the PINO scale branch on real voxel/force/FEM labels.
Boundary samples are converted from the stored 250MPa protocol to the current
50MPa protocol by scaling force and stress by 0.2, which preserves safety factor.
The authoritative pipeline is not modified.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_candidate_pino_grouped import acs_metrics, metrics, split_groups, subset
from train_candidate_pino_field import load_scaleup_rows
from train_pino_aug_hardneg import SCALE_BRANCH, load_geoattn_model, xg
from structural.analysis.load_generator import LoadGenerator

YIELD_STRESS = 50e6
TAU = 1.3
SEED = 20260703


def load_candidate_rows() -> list[dict]:
    rows = []
    generator = LoadGenerator()
    for branch, path in (("free", ROOT / "data/aug.npz"), ("fixed", ROOT / "data/aug_fixed.npz")):
        payload = np.load(path)
        for index in range(len(payload["dens"])):
            density = payload["dens"][index].astype(np.float32)
            force = generator.standing_load(density)["force"] * float(payload["mult"][index])
            vm = float(payload["vm"][index])
            rows.append({
                "source": branch,
                "group": index // 16,
                "density": density,
                "force": force.astype(np.float32),
                "log_vm": float(np.log10(max(vm, 1e-12))),
                "sf": float(payload["sf"][index]),
            })
    return rows


def load_boundary_rows(limit: int | None = None) -> list[dict]:
    rows = []
    sample_dir = ROOT / "results/boundary_fem_yield250_20260624/samples"
    for path in sorted(sample_dir.glob("*.npz")):
        payload = np.load(path)
        density = payload["density"].astype(np.float32)
        force = payload["force"].astype(np.float32) * 0.2
        vm = float(payload["vm_max"]) * 0.2
        match = re.search(r"aug_(\d+)_", path.name)
        source_index = int(match.group(1)) if match else len(rows)
        rows.append({
            "source": "boundary_yield250_to50",
            "group": 10000 + source_index,
            "density": density,
            "force": force,
            "log_vm": float(np.log10(max(vm, 1e-12))),
            "sf": float(payload["sf"]),
        })
        if limit is not None and len(rows) >= limit:
            break
    return rows


@torch.no_grad()
def predict(model, rows: list[dict], device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        inputs = torch.cat([xg(model, row["density"], row["force"], device) for row in batch], dim=0)
        pred = model._scale_forward(inputs).squeeze(-1).reshape(-1)
        out.extend(float(value) for value in pred.detach().cpu())
    return np.asarray(out, dtype=float)


def threshold_safe_metrics(pred_log_vm: np.ndarray, rows: list[dict], threshold: float | None = None) -> dict:
    pred_log_sf = math.log10(YIELD_STRESS) - pred_log_vm
    true_safe = np.asarray([row["sf"] >= TAU for row in rows], dtype=bool)
    if threshold is None:
        threshold = 1e9
        best = -1
        for value in sorted(set(float(v) for v in pred_log_sf), reverse=True):
            accept = pred_log_sf >= value
            false_pass = int((accept & ~true_safe).sum())
            true_accept = int((accept & true_safe).sum())
            if false_pass == 0 and true_accept > best:
                threshold = value
                best = true_accept
    accept = pred_log_sf >= threshold
    return {
        "threshold_log_sf": float(threshold),
        "n": len(rows),
        "safe": int(true_safe.sum()),
        "unsafe": int((~true_safe).sum()),
        "auto_safe_accept_count": int(accept.sum()),
        "auto_safe_true": int((accept & true_safe).sum()),
        "auto_false_pass": int((accept & ~true_safe).sum()),
        "auto_safe_rate": float((accept & true_safe).sum() / max(int(true_safe.sum()), 1)),
    }


def cpu_state(module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--boundary-limit", type=int, default=1200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--start-ckpt", type=Path, default=ROOT / "checkpoints/pino_aug_hardneg_20260624/scale_geoattn_e8.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/pino_safe_side_boundary_20260703")
    parser.add_argument("--result", type=Path, default=ROOT / "results/pino_safe_side_boundary_20260703.json")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    candidate_rows = load_candidate_rows()
    splits = split_groups()
    train_rows = subset(candidate_rows, splits["train"])
    val_rows = subset(candidate_rows, splits["selection_validation"])
    cal_rows = subset(candidate_rows, splits["conformal_calibration"])
    test_rows = subset(candidate_rows, splits["heldout_test"])
    boundary_rows = load_boundary_rows(args.boundary_limit)
    scaleup_rows = load_scaleup_rows()

    model, head, geo_payload = load_geoattn_model(device, args.base_ckpt, args.start_ckpt)
    baseline_test_pred = predict(model, test_rows, device, args.batch_size)
    baseline_scaleup_pred = predict(model, scaleup_rows, device, args.batch_size)
    baseline = {
        "raw_candidate_heldout": metrics(baseline_test_pred, test_rows),
        "raw_external_scaleup": metrics(baseline_scaleup_pred, scaleup_rows),
        "safe_candidate_heldout": threshold_safe_metrics(baseline_test_pred, test_rows, None),
        "safe_external_scaleup": None,
    }
    baseline["safe_external_scaleup"] = threshold_safe_metrics(
        baseline_scaleup_pred,
        scaleup_rows,
        baseline["safe_candidate_heldout"]["threshold_log_sf"],
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name in SCALE_BRANCH:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
        trainable.append(parameter)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    combined_train = train_rows + boundary_rows
    best_key = None
    best_state = None
    history = []
    log_tau_sf = math.log10(TAU)
    log_yield = math.log10(YIELD_STRESS)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(combined_train))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            batch = [combined_train[int(i)] for i in ids]
            inputs = torch.cat([xg(model, row["density"], row["force"], device) for row in batch], dim=0)
            target_log_vm = torch.as_tensor([row["log_vm"] for row in batch], dtype=torch.float32, device=device)
            true_sf = torch.as_tensor([row["sf"] for row in batch], dtype=torch.float32, device=device)
            true_safe = true_sf >= TAU
            pred_log_vm = model._scale_forward(inputs).squeeze(-1).reshape(-1)
            pred_log_sf = log_yield - pred_log_vm
            boundary = torch.abs(torch.log10(true_sf) - log_tau_sf) < 0.18
            regression = F.smooth_l1_loss(pred_log_vm, target_log_vm, reduction="none")
            under_vm = F.relu(target_log_vm - pred_log_vm)
            false_safe_hinge = F.relu(pred_log_sf - (log_tau_sf - 0.03)) * (~true_safe).float()
            safe_margin_hinge = F.relu((log_tau_sf + 0.06) - pred_log_sf) * true_safe.float()
            weights = torch.where(boundary, 4.0, 1.0)
            loss = torch.mean(
                weights * regression
                + 5.0 * under_vm
                + 10.0 * false_safe_hinge
                + 0.7 * safe_margin_hinge
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)

        val_pred = predict(model, val_rows, device, args.batch_size)
        val_safe = threshold_safe_metrics(val_pred, val_rows)
        val_raw = metrics(val_pred, val_rows)
        key = (val_safe["auto_false_pass"], -val_safe["auto_safe_true"], val_raw["false_pass_count"], val_raw["log_vm_mae"])
        row = {"epoch": epoch, "train_loss": total / len(combined_train), "validation_safe": val_safe, "validation_raw": val_raw}
        history.append(row)
        print(json.dumps(row), flush=True)
        if best_key is None or key < best_key:
            best_key = key
            best_state = {
                "scale_branch_state_dict": {name: cpu_state(getattr(model, name)) for name in SCALE_BRANCH},
                "attn_scale_head_state_dict": cpu_state(head),
                "epoch": epoch,
            }

    for name, state in best_state["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(state)
    head.load_state_dict(best_state["attn_scale_head_state_dict"])

    val_pred = predict(model, val_rows, device, args.batch_size)
    test_pred = predict(model, test_rows, device, args.batch_size)
    cal_pred = predict(model, cal_rows, device, args.batch_size)
    scaleup_pred = predict(model, scaleup_rows, device, args.batch_size)
    validation_safe = threshold_safe_metrics(val_pred, val_rows)
    threshold = validation_safe["threshold_log_sf"]
    candidate_safe = threshold_safe_metrics(test_pred, test_rows, threshold)
    external_safe = threshold_safe_metrics(scaleup_pred, scaleup_rows, threshold)
    candidate_raw = metrics(test_pred, test_rows)
    external_raw = metrics(scaleup_pred, scaleup_rows)
    candidate_acs = acs_metrics(cal_pred, cal_rows, test_pred, test_rows, alpha=0.05)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "safe_side_boundary_scale.pt"
    torch.save({
        **best_state,
        "attn_dim": geo_payload.get("attn_dim", 64),
        "n_heads": geo_payload.get("n_heads", 4),
        "base_ckpt": str(args.base_ckpt),
        "start_ckpt": str(args.start_ckpt),
        "boundary_protocol": "force and vm_max scaled by 0.2 from stored 250MPa labels",
        "split_seed": SEED,
    }, checkpoint)
    promotion = {
        "candidate_zero_false_pass": candidate_safe["auto_false_pass"] == 0,
        "external_zero_false_pass": external_safe["auto_false_pass"] == 0,
        "candidate_auto_safe_at_least_50pct": candidate_safe["auto_safe_rate"] >= 0.50,
        "external_auto_safe_at_least_50pct": external_safe["auto_safe_rate"] >= 0.50,
    }
    result = {
        "schema": "pino_safe_side_boundary_v1",
        "created": "2026-07-03",
        "training_rows": {"candidate": len(train_rows), "boundary": len(boundary_rows)},
        "splits": splits,
        "baseline": baseline,
        "validation_safe": validation_safe,
        "candidate_heldout_safe": candidate_safe,
        "external_scaleup_safe": external_safe,
        "candidate_heldout_raw": candidate_raw,
        "external_scaleup_raw": external_raw,
        "candidate_acs": candidate_acs,
        "promotion_gate": promotion,
        "promotion_eligible_for_fem_replacement_claim": all(promotion.values()),
        "history": history,
        "checkpoint": str(checkpoint),
        "limitations": [
            "Boundary data comes from the older ShadowArt-Hard augmentation domain, converted to the 50MPa stress scale.",
            "Heldout candidate groups and external scaleup are audit-only.",
            "Labels are voxel-FEM references, not physical tests.",
            "The authoritative pipeline is not modified.",
        ],
    }
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "result": str(args.result),
        "validation_safe": validation_safe,
        "candidate_heldout_safe": candidate_safe,
        "external_scaleup_safe": external_safe,
        "promotion_eligible_for_fem_replacement_claim": result["promotion_eligible_for_fem_replacement_claim"],
    }, indent=2))


if __name__ == "__main__":
    main()
