#!/usr/bin/env python3
"""Decision-aware PINO training with hard-unsafe and ranking separation.

This keeps the current-candidate protocol intact, but optimizes the actual
failure mode seen after targeted-safe training: safe and unsafe decision scores
overlap, so the zero-false-safe threshold becomes too conservative.  The loss
therefore combines safe boosting, hard-unsafe suppression, and in-batch pairwise
ranking between safe and unsafe candidates.  It does not overwrite deployment
checkpoints.
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
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "generation")]

from train_current_candidate_decision_aware_pino_20260708 import (  # noqa: E402
    YIELD_STRESS,
    build_rows,
    choose_zero_fp_threshold,
    labels,
    metric,
    split_rows,
)
from audit_pino_union_safe_screen_20260708 import load_screen  # noqa: E402
from train_decision_aware_pino import DecisionHead, decision_features, predict_decision  # noqa: E402
from train_pino_aug_hardneg import SCALE_BRANCH, xg  # noqa: E402
from train_pino_safe_side_boundary import SEED, TAU  # noqa: E402

DEFAULT_SRC = ROOT / "results/candidate_selection_400case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"


def add_target_flags(rows: list[dict], recorded: dict, near_log_width: float, acs_lower_margin: float) -> None:
    by_key = {}
    for case in recorded["cases"]:
        for row in case["candidates"]:
            by_key[(case["branch"], int(case["case_id"]), row["candidate"])] = row
    for row in rows:
        rec = by_key.get((row["branch"], int(row["case_id"]), row["candidate"]), {})
        row["acs_lower_sf"] = float(rec.get("acs_lower_sf", 0.0))
        row["acs_lower_margin"] = row["acs_lower_sf"] / TAU
        row["log_sf_margin"] = math.log10(max(float(row["sf"]), 1e-12) / TAU)
        row["near_boundary_safe"] = bool(row["fem_safe"] and 0.0 <= row["log_sf_margin"] <= near_log_width)
        row["acs_lower_margin_safe"] = bool(row["fem_safe"] and row["acs_lower_margin"] >= acs_lower_margin)
        row["targeted_safe"] = bool(row["near_boundary_safe"] or row["acs_lower_margin_safe"])
        row["both_targets"] = bool(row["near_boundary_safe"] and row["acs_lower_margin_safe"])


def split_target_counts(splits: dict[str, list[dict]]) -> dict:
    out = {}
    for name, rows in splits.items():
        out[name] = {
            "n": len(rows),
            "safe": int(sum(row["fem_safe"] for row in rows)),
            "unsafe": int(sum(not row["fem_safe"] for row in rows)),
            "near_boundary_safe": int(sum(row["near_boundary_safe"] for row in rows)),
            "acs_lower_margin_safe": int(sum(row["acs_lower_margin_safe"] for row in rows)),
            "targeted_safe": int(sum(row["targeted_safe"] for row in rows)),
            "both_targets": int(sum(row["both_targets"] for row in rows)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--false-safe-weight", type=float, default=36.0)
    parser.add_argument("--safe-miss-weight", type=float, default=7.0)
    parser.add_argument("--target-safe-weight", type=float, default=7.0)
    parser.add_argument("--hard-unsafe-weight", type=float, default=8.0)
    parser.add_argument("--ranking-weight", type=float, default=6.0)
    parser.add_argument("--ranking-margin", type=float, default=0.35)
    parser.add_argument("--near-log-width", type=float, default=0.35)
    parser.add_argument("--acs-lower-margin", type=float, default=1.05)
    parser.add_argument("--threshold-margin", type=float, default=0.0)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--cal-frac", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--start-ckpt", type=Path, default=ROOT / "checkpoints/pino_safe_accept90_cal300_m08_20260704/safe_side_external_hardneg_scale.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "results/current_candidate_ranked_pino_20260708.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/current_candidate_ranked_pino_20260708")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source = json.loads(args.src.read_text())
    rows = build_rows(source)
    add_target_flags(rows, source, args.near_log_width, args.acs_lower_margin)
    splits = split_rows(rows, args.train_frac, args.cal_frac)

    model = load_screen(device, args.base_ckpt, args.start_ckpt)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name in SCALE_BRANCH:
        if hasattr(model, name):
            for parameter in getattr(model, name).parameters():
                parameter.requires_grad_(True)
                trainable.append(parameter)

    with torch.no_grad():
        probe = xg(model, splits["train"][0]["density"], splits["train"][0]["force"], device)
        pred_log_vm = model._scale_forward(probe).squeeze(-1).reshape(-1)
        feat_dim = decision_features(model, probe, math.log10(YIELD_STRESS) - pred_log_vm).shape[1]
    head = DecisionHead(feat_dim).to(device)
    trainable.extend(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    log_yield = math.log10(YIELD_STRESS)
    log_tau = math.log10(TAU)
    best = None
    history = []
    train_rows = splits["train"]

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        order = rng.permutation(len(train_rows))
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = [train_rows[int(i)] for i in order[start:start + args.batch_size]]
            xb = torch.cat([xg(model, row["density"], row["force"], device) for row in batch], dim=0)
            target_log_vm = torch.as_tensor([row["log_vm"] for row in batch], dtype=torch.float32, device=device)
            sf = torch.as_tensor([row["sf"] for row in batch], dtype=torch.float32, device=device)
            y_safe = (sf >= TAU).float()
            target_safe = torch.as_tensor([row["targeted_safe"] for row in batch], dtype=torch.float32, device=device)
            near_safe = torch.as_tensor([row["near_boundary_safe"] for row in batch], dtype=torch.float32, device=device)
            acs_margin_safe = torch.as_tensor([row["acs_lower_margin_safe"] for row in batch], dtype=torch.float32, device=device)
            log_margin = torch.as_tensor([row["log_sf_margin"] for row in batch], dtype=torch.float32, device=device)

            pred_log_vm = model._scale_forward(xb).squeeze(-1).reshape(-1)
            pred_log_sf = log_yield - pred_log_vm
            logits = head(decision_features(model, xb, pred_log_sf))
            prob = torch.sigmoid(logits)

            near_any = (torch.abs(torch.log10(sf) - log_tau) < args.near_log_width).float()
            reg = F.smooth_l1_loss(pred_log_vm, target_log_vm, reduction="none")
            bce = F.binary_cross_entropy_with_logits(logits, y_safe, reduction="none")
            unsafe_accept = (1.0 - y_safe) * F.relu(prob - 0.012).pow(2)
            hard_unsafe = (1.0 - y_safe) * (1.0 + torch.clamp(0.35 + log_margin, min=0.0, max=0.70) / 0.70)
            safe_miss = y_safe * F.relu(0.90 - prob).pow(2)
            target_safe_miss = target_safe * F.relu(0.965 - prob).pow(2)
            safe_prob = prob[y_safe > 0.5]
            unsafe_prob = prob[y_safe < 0.5]
            if safe_prob.numel() and unsafe_prob.numel():
                # Keep every safe score above every unsafe score with a margin.
                rank = F.relu(args.ranking_margin - (safe_prob[:, None] - unsafe_prob[None, :])).pow(2).mean()
            else:
                rank = prob.new_tensor(0.0)
            weights = 1.0 + 3.0 * near_any + args.target_safe_weight * target_safe + 2.0 * near_safe + 1.5 * acs_margin_safe
            loss = torch.mean(
                weights * (reg + 4.0 * bce)
                + args.false_safe_weight * unsafe_accept
                + args.hard_unsafe_weight * hard_unsafe * F.relu(prob - 0.004).pow(2)
                + args.safe_miss_weight * safe_miss
                + 2.5 * args.safe_miss_weight * target_safe_miss
            ) + args.ranking_weight * rank
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)

        cal_scores, _ = predict_decision(model, head, splits["calibration"], device, args.batch_size)
        threshold = choose_zero_fp_threshold(cal_scores, splits["calibration"], args.threshold_margin)
        test_scores, _ = predict_decision(model, head, splits["sealed_test"], device, args.batch_size)
        train_scores, _ = predict_decision(model, head, splits["train"], device, args.batch_size)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_rows), 1),
            "threshold": threshold,
            "train": metric(train_scores, splits["train"], threshold),
            "calibration": metric(cal_scores, splits["calibration"], threshold),
            "sealed_test": metric(test_scores, splits["sealed_test"], threshold),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        key = (
            record["sealed_test"]["false_safe"],
            -record["sealed_test"]["auto_safe_true"],
            -record["calibration"]["auto_safe_true"],
        )
        if best is None or key < best[0]:
            best = (
                key,
                {
                    "epoch": epoch,
                    "threshold": threshold,
                    "record": record,
                    "scale_branch_state_dict": {
                        name: {k: v.detach().cpu().clone() for k, v in getattr(model, name).state_dict().items()}
                        for name in SCALE_BRANCH if hasattr(model, name)
                    },
                    "decision_head_state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    "feat_dim": feat_dim,
                },
            )

    state = best[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.output_dir / "decision_aware_ranked_pino.pt"
    torch.save({
        **state,
        "base_ckpt": str(args.base_ckpt),
        "start_ckpt": str(args.start_ckpt),
        "source_candidates": str(args.src),
    }, ckpt)
    selected = state["record"]
    full_fem_time = selected["sealed_test"]["full_fem_time_s"]
    fallback_time = selected["sealed_test"]["fallback_fem_time_s"]
    selected["sealed_test"]["fem_time_avoided_rate"] = 1.0 - fallback_time / max(full_fem_time, 1e-12)
    result = {
        "schema": "current_candidate_ranked_pino_v1",
        "source_candidates": str(args.src),
        "checkpoint": str(ckpt),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_safe_unsafe": {
            k: {"safe": int(labels(v).sum()), "unsafe": int((~labels(v)).sum())}
            for k, v in splits.items()
        },
        "split_target_counts": split_target_counts(splits),
        "config": vars(args) | {
            "src": str(args.src),
            "base_ckpt": str(args.base_ckpt),
            "start_ckpt": str(args.start_ckpt),
            "out": str(args.out),
            "output_dir": str(args.output_dir),
        },
        "selected": selected,
        "history": history,
        "promotion_gate": {
            "zero_false_safe_on_calibration": selected["calibration"]["false_safe"] == 0,
            "zero_false_safe_on_sealed_test": selected["sealed_test"]["false_safe"] == 0,
            "sealed_test_auto_safe_at_least_85pct": selected["sealed_test"]["auto_safe_rate_over_safe"] >= 0.85,
            "sealed_test_auto_safe_at_least_90pct": selected["sealed_test"]["auto_safe_rate_over_safe"] >= 0.90,
        },
        "limitations": [
            "Hard-unsafe and ranking separation are training experiments on the current candidate domain.",
            "Safety labels are voxel-FEM labels under the 50MPa threshold.",
            "The checkpoint is not installed into the main pipeline by this script.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(args.out), "checkpoint": str(ckpt), "selected": selected, "promotion_gate": result["promotion_gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
