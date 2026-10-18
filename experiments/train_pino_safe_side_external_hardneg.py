#!/usr/bin/env python3
"""Safe-side PINO fine-tuning with external hard negatives.

External scaleup rows are split into calibration/train and heldout subsets.
Only the calibration subset is used for hard-negative training and threshold
selection; heldout remains audit-only. The authoritative pipeline is not modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_pino_safe_side_boundary import (
    SEED,
    TAU,
    load_boundary_rows,
    load_candidate_rows,
    predict,
    threshold_safe_metrics,
    cpu_state,
)
from train_candidate_pino_grouped import metrics, split_groups, subset
from train_candidate_pino_field import load_scaleup_rows
from train_pino_aug_hardneg import SCALE_BRANCH, load_geoattn_model, xg

YIELD_STRESS = 50e6
HARDSET_ROOT = Path("/home/lin/xvxinyao/total_project/results/shadowart_fem_hardset_v1")


def parse_sf_values(text: str) -> set[float]:
    return {round(float(value), 3) for value in text.split(",") if value.strip()}


def load_hardset_rows(split: str, sf_values: set[float]) -> list[dict]:
    index_path = ROOT / "data/shadowart_fem_hardset_v1/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = []
    for rec in index[split]:
        sf = round(float(rec["sf"]), 3)
        if sf not in sf_values:
            continue
        path = HARDSET_ROOT / rec["path"]
        if not path.exists():
            continue
        payload = np.load(path)
        rows.append({
            "source": "hardset_near_boundary",
            "case_id": int(rec["case_id"]),
            "case_name": rec["case_name"],
            "density": payload["density"].astype(np.float32),
            "force": payload["force"].astype(np.float32),
            "log_vm": float(payload["log_vm_max"]),
            "sf": float(payload["sf"]),
        })
    return rows


def split_external(rows: list[dict], n_calibration: int, seed: int = SEED):
    indices = np.arange(len(rows))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    calibration = sorted(int(value) for value in indices[:n_calibration])
    heldout = sorted(int(value) for value in indices[n_calibration:])
    return [rows[i] for i in calibration], [rows[i] for i in heldout], {
        "seed": seed,
        "calibration_indices": calibration,
        "heldout_indices": heldout,
    }


def choose_joint_threshold(preds_and_rows: list[tuple[np.ndarray, list[dict]]]) -> dict:
    scores = np.concatenate([math.log10(YIELD_STRESS) - pred for pred, _ in preds_and_rows])
    truth = np.asarray([row["sf"] >= TAU for _, rows in preds_and_rows for row in rows], dtype=bool)
    threshold = 1e9
    best = -1
    for value in sorted(set(float(v) for v in scores), reverse=True):
        accept = scores >= value
        false_pass = int((accept & ~truth).sum())
        true_accept = int((accept & truth).sum())
        if false_pass == 0 and true_accept > best:
            threshold = value
            best = true_accept
    return {
        "threshold_log_sf": float(threshold),
        "calibration_combined": threshold_safe_metrics(
            math.log10(YIELD_STRESS) - scores,
            [{"sf": float(2.0 if y else 1.0)} for y in truth],
            threshold,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--boundary-limit", type=int, default=600)
    parser.add_argument("--external-calibration", type=int, default=120)
    parser.add_argument("--external-hard-repeat", type=int, default=3)
    parser.add_argument("--hardset-sf-values", default="0.8,1.0,1.2,1.5,2.0,3.0")
    parser.add_argument("--hardset-safe-repeat", type=int, default=2)
    parser.add_argument("--hardset-unsafe-repeat", type=int, default=1)
    parser.add_argument("--hardset-calibration", type=int, default=1)
    parser.add_argument("--false-safe-weight", type=float, default=12.0)
    parser.add_argument("--external-false-safe-weight", type=float, default=12.0)
    parser.add_argument("--safe-margin-weight", type=float, default=0.4)
    parser.add_argument("--hardset-safe-margin-weight", type=float, default=1.2)
    parser.add_argument("--safe-margin-log", type=float, default=0.08)
    parser.add_argument("--unsafe-buffer-log", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--start-ckpt", type=Path, default=ROOT / "checkpoints/pino_safe_side_boundary_20260703/safe_side_boundary_scale.pt")
    parser.add_argument("--result", type=Path, default=ROOT / "results/pino_safe_side_external_hardneg_20260703.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/pino_safe_side_external_hardneg_20260703")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    candidate_rows = load_candidate_rows()
    splits = split_groups()
    train_rows = subset(candidate_rows, splits["train"])
    val_rows = subset(candidate_rows, splits["selection_validation"])
    test_rows = subset(candidate_rows, splits["heldout_test"])
    boundary_rows = load_boundary_rows(args.boundary_limit)
    external_rows = load_scaleup_rows()
    for row in external_rows:
        row["source"] = "scaleup"
    external_cal, external_heldout, external_split = split_external(external_rows, args.external_calibration)
    hardset_sf_values = parse_sf_values(args.hardset_sf_values)
    hard_train = load_hardset_rows("train", hardset_sf_values)
    hard_val = load_hardset_rows("val", hardset_sf_values)
    hard_test = load_hardset_rows("test", hardset_sf_values)

    model, head, geo_payload = load_geoattn_model(device, args.base_ckpt, args.start_ckpt)
    # If start_ckpt is the safe-side delta, load its scale branch into the model.
    start_payload = torch.load(args.start_ckpt, map_location=device, weights_only=False)
    if "scale_branch_state_dict" in start_payload:
        for name, state in start_payload["scale_branch_state_dict"].items():
            getattr(model, name).load_state_dict(state)
    if "attn_scale_head_state_dict" in start_payload:
        head.load_state_dict(start_payload["attn_scale_head_state_dict"])

    baseline_candidate_pred = predict(model, test_rows, device, args.batch_size)
    baseline_external_pred = predict(model, external_heldout, device, args.batch_size)

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

    external_hard = [row for row in external_cal if row["sf"] < TAU]
    hard_safe = [row for row in hard_train if row["sf"] >= TAU]
    hard_unsafe = [row for row in hard_train if row["sf"] < TAU]
    combined_train = (
        train_rows
        + boundary_rows
        + external_cal
        + external_hard * args.external_hard_repeat
        + hard_safe * args.hardset_safe_repeat
        + hard_unsafe * args.hardset_unsafe_repeat
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    log_yield = math.log10(YIELD_STRESS)
    log_tau = math.log10(TAU)
    best_key = None
    best_state = None
    history = []

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
            is_external = torch.as_tensor([row.get("source", "") == "scaleup" for row in batch], dtype=torch.bool, device=device)
            is_hardset = torch.as_tensor([row.get("source", "") == "hardset_near_boundary" for row in batch], dtype=torch.bool, device=device)
            pred_log_vm = model._scale_forward(inputs).squeeze(-1).reshape(-1)
            pred_log_sf = log_yield - pred_log_vm
            boundary = torch.abs(torch.log10(true_sf) - log_tau) < 0.18
            regression = F.smooth_l1_loss(pred_log_vm, target_log_vm, reduction="none")
            under_vm = F.relu(target_log_vm - pred_log_vm)
            false_safe_hinge = F.relu(pred_log_sf - (log_tau - args.unsafe_buffer_log)) * (~true_safe).float()
            external_false_safe = false_safe_hinge * is_external.float()
            safe_margin_hinge = F.relu((log_tau + args.safe_margin_log) - pred_log_sf) * true_safe.float()
            hardset_safe_margin = safe_margin_hinge * (is_hardset & true_safe).float()
            weights = torch.where(boundary | is_hardset, 4.0, 1.0)
            loss = torch.mean(
                weights * regression
                + 5.0 * under_vm
                + args.false_safe_weight * false_safe_hinge
                + args.external_false_safe_weight * external_false_safe
                + args.safe_margin_weight * safe_margin_hinge
                + args.hardset_safe_margin_weight * hardset_safe_margin
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)

        val_pred = predict(model, val_rows, device, args.batch_size)
        ext_cal_pred = predict(model, external_cal, device, args.batch_size)
        hard_val_pred = predict(model, hard_val, device, args.batch_size)
        calibration_inputs = [(val_pred, val_rows), (ext_cal_pred, external_cal)]
        if args.hardset_calibration:
            calibration_inputs.append((hard_val_pred, hard_val))
        threshold_record = choose_joint_threshold(calibration_inputs)
        threshold = threshold_record["threshold_log_sf"]
        candidate_val = threshold_safe_metrics(val_pred, val_rows, threshold)
        external_cal_metric = threshold_safe_metrics(ext_cal_pred, external_cal, threshold)
        hard_val_metric = threshold_safe_metrics(hard_val_pred, hard_val, threshold)
        key = (
            candidate_val["auto_false_pass"] + external_cal_metric["auto_false_pass"] + hard_val_metric["auto_false_pass"],
            -(candidate_val["auto_safe_true"] + external_cal_metric["auto_safe_true"] + hard_val_metric["auto_safe_true"]),
        )
        row = {
            "epoch": epoch,
            "train_loss": total / len(combined_train),
            "threshold": threshold,
            "candidate_validation": candidate_val,
            "external_calibration": external_cal_metric,
            "hardset_validation_near_boundary": hard_val_metric,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if best_key is None or key < best_key:
            best_key = key
            best_state = {
                "scale_branch_state_dict": {name: cpu_state(getattr(model, name)) for name in SCALE_BRANCH},
                "attn_scale_head_state_dict": cpu_state(head),
                "epoch": epoch,
                "threshold_log_sf": threshold,
            }

    for name, state in best_state["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(state)
    head.load_state_dict(best_state["attn_scale_head_state_dict"])
    val_pred = predict(model, val_rows, device, args.batch_size)
    ext_cal_pred = predict(model, external_cal, device, args.batch_size)
    hard_val_pred = predict(model, hard_val, device, args.batch_size)
    calibration_inputs = [(val_pred, val_rows), (ext_cal_pred, external_cal)]
    if args.hardset_calibration:
        calibration_inputs.append((hard_val_pred, hard_val))
    threshold = choose_joint_threshold(calibration_inputs)["threshold_log_sf"]
    candidate_pred = predict(model, test_rows, device, args.batch_size)
    external_pred = predict(model, external_heldout, device, args.batch_size)
    hard_test_pred = predict(model, hard_test, device, args.batch_size)
    candidate_safe = threshold_safe_metrics(candidate_pred, test_rows, threshold)
    external_safe = threshold_safe_metrics(external_pred, external_heldout, threshold)
    hardset_safe = threshold_safe_metrics(hard_test_pred, hard_test, threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "safe_side_external_hardneg_scale.pt"
    torch.save({
        **best_state,
        "base_ckpt": str(args.base_ckpt),
        "start_ckpt": str(args.start_ckpt),
        "attn_dim": geo_payload.get("attn_dim", 64),
        "n_heads": geo_payload.get("n_heads", 4),
        "external_split": external_split,
        "split_seed": SEED,
    }, checkpoint)
    promotion = {
        "candidate_zero_false_pass": candidate_safe["auto_false_pass"] == 0,
        "external_zero_false_pass": external_safe["auto_false_pass"] == 0,
        "hardset_zero_false_pass": hardset_safe["auto_false_pass"] == 0,
        "candidate_auto_safe_at_least_90pct": candidate_safe["auto_safe_rate"] >= 0.90,
        "external_auto_safe_at_least_90pct": external_safe["auto_safe_rate"] >= 0.90,
        "hardset_auto_safe_at_least_90pct": hardset_safe["auto_safe_rate"] >= 0.90,
    }
    result = {
        "schema": "pino_safe_side_external_hardneg_v1",
        "created": "2026-07-03",
        "training_rows": {
            "candidate": len(train_rows),
            "boundary": len(boundary_rows),
            "external_calibration": len(external_cal),
            "external_hard_negatives_repeated": len(external_hard) * args.external_hard_repeat,
            "external_hard_repeat": args.external_hard_repeat,
            "hardset_train_near_boundary": len(hard_train),
            "hardset_safe_repeated": len(hard_safe) * args.hardset_safe_repeat,
            "hardset_unsafe_repeated": len(hard_unsafe) * args.hardset_unsafe_repeat,
            "hardset_sf_values": sorted(hardset_sf_values),
            "false_safe_weight": args.false_safe_weight,
            "external_false_safe_weight": args.external_false_safe_weight,
            "safe_margin_weight": args.safe_margin_weight,
            "hardset_safe_margin_weight": args.hardset_safe_margin_weight,
            "safe_margin_log": args.safe_margin_log,
            "unsafe_buffer_log": args.unsafe_buffer_log,
            "hardset_calibration": bool(args.hardset_calibration),
        },
        "calibration_rows": {
            "candidate_validation": len(val_rows),
            "external_calibration": len(external_cal),
            "hardset_validation_near_boundary": len(hard_val),
        },
        "sealed_test_rows": {
            "candidate_heldout": len(test_rows),
            "external_heldout": len(external_heldout),
            "hardset_test_near_boundary": len(hard_test),
        },
        "external_split": external_split,
        "baseline": {
            "candidate_heldout_safe": threshold_safe_metrics(baseline_candidate_pred, test_rows),
            "external_heldout_safe_at_candidate_threshold": threshold_safe_metrics(
                baseline_external_pred,
                external_heldout,
                threshold_safe_metrics(baseline_candidate_pred, test_rows)["threshold_log_sf"],
            ),
        },
        "selected_threshold_log_sf": threshold,
        "candidate_heldout_safe": candidate_safe,
        "external_heldout_safe": external_safe,
        "hardset_test_near_boundary_safe": hardset_safe,
        "candidate_heldout_raw": metrics(candidate_pred, test_rows),
        "external_heldout_raw": metrics(external_pred, external_heldout),
        "promotion_gate": promotion,
        "promotion_eligible_for_fem_replacement_claim": all(promotion.values()),
        "history": history,
        "checkpoint": str(checkpoint),
        "limitations": [
            "External calibration and hardset train/validation rows are part of the training/calibration protocol; candidate, external, and hardset heldout rows are audit-only.",
            "Labels are voxel-FEM references, not physical tests.",
            "The authoritative pipeline is not modified.",
        ],
    }
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "result": str(args.result),
        "candidate_heldout_safe": candidate_safe,
        "external_heldout_safe": external_safe,
        "hardset_test_near_boundary_safe": hardset_safe,
        "promotion_eligible_for_fem_replacement_claim": result["promotion_eligible_for_fem_replacement_claim"],
    }, indent=2))


if __name__ == "__main__":
    main()
