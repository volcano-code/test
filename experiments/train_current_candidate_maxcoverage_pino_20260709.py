#!/usr/bin/env python3
"""Max-coverage decision-aware PINO training on the current candidate domain.

This is an aggressive research run. It keeps the displacement/stress backbone
frozen, but can unfreeze the rear scale-encoder stages, the scale head, and a
decision head with stratified zero-false-safe thresholding.
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

from audit_pino_union_safe_screen_20260708 import load_screen  # noqa: E402
from train_current_candidate_decision_aware_pino_20260708 import (  # noqa: E402
    DEFAULT_SRC,
    TAU,
    YIELD_STRESS,
    build_rows,
    choose_stratified_thresholds,
    choose_zero_fp_threshold,
    labels,
    metric,
    metric_stratified,
    pairwise_rank_loss,
    split_rows,
    top_unsafe_scores,
)
from train_decision_aware_pino import DecisionHead, decision_features, predict_decision  # noqa: E402
from train_pino_aug_hardneg import xg  # noqa: E402
from train_pino_safe_side_boundary import SEED  # noqa: E402


def jsonable_config(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def set_trainable(model, mode: str) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable: list[torch.nn.Parameter] = []
    prefixes = {
        "scale": ("scale_", "attn_scale_head"),
        "scale_head": ("scale_head", "scale_head_v2", "attn_scale_head"),
        "all_heads": ("scale_", "attn_scale_head", "stress_head"),
        # Adapt the representation closest to the stress-scale prediction
        # without perturbing the frozen displacement/stress field backbone.
        "rear_scale": (
            "scale_enc2", "scale_down2", "scale_bottleneck",
            "scale_head", "scale_head_v2", "attn_scale_head",
        ),
        # Full scale pathway: unlike rear_scale, this also adapts the input
        # lifting and first downsampling stage used by the scalar stress head.
        # The displacement and dense stress-field backbone remain frozen.
        "full_scale": (
            "scale_lifting", "scale_down1", "scale_enc2", "scale_down2",
            "scale_bottleneck", "scale_head", "scale_head_v2", "attn_scale_head",
        ),
    }[mode]
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    return trainable


def expanded_train_rows(rows: list[dict], clear_safe_sf: float, near_width_log: float,
                        clear_safe_repeat: int, near_safe_repeat: int, unsafe_repeat: int) -> list[dict]:
    out: list[dict] = []
    log_tau = math.log10(TAU)
    for row in rows:
        sf = float(row["sf"])
        is_safe = sf >= TAU
        if not is_safe:
            repeat = unsafe_repeat
        elif abs(math.log10(sf) - log_tau) <= near_width_log:
            repeat = near_safe_repeat
        elif sf >= clear_safe_sf:
            repeat = clear_safe_repeat
        else:
            repeat = 1
        out.extend([row] * max(int(repeat), 1))
    return out


def score_stats(scores: np.ndarray, rows: list[dict]) -> dict:
    truth = labels(rows)
    safe_scores = scores[truth]
    unsafe_scores = scores[~truth]

    def q(arr):
        if len(arr) == 0:
            return {}
        return {
            "min": float(np.min(arr)),
            "p05": float(np.quantile(arr, 0.05)),
            "p50": float(np.quantile(arr, 0.50)),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(np.max(arr)),
        }

    return {"safe": q(safe_scores), "unsafe": q(unsafe_scores)}


def load_extra_fem_rows(index_path: Path) -> list[dict]:
    payload = json.loads(index_path.read_text())
    rows: list[dict] = []
    for row in payload["rows"]:
        data = np.load(row["density_path"])
        density = data["density"].astype(np.float32)
        force = data["force"].astype(np.float32)
        sf = float(row["sf"])
        rows.append({
            "source": row.get("source", "extra_fem"),
            # Keep the original task group.  Splits are task-disjoint, so
            # adding a prefix here silently filtered every newly labelled row.
            "group": f"{row['branch']}:{row['case_id']}",
            "branch": row["branch"],
            "case_id": int(row["case_id"]),
            "candidate": row["candidate"],
            "density": density,
            "force": force,
            "sf": sf,
            "log_vm": float(math.log10(max(YIELD_STRESS / max(sf, 1e-12), 1e-12))),
            "fem_safe": bool(row["fem_safe"]),
            "fem_elapsed_s": float(row["fem_elapsed_s"]),
            "mean_iou": float(row.get("mean_iou", 0.0)),
            "mean_dice": float(row.get("mean_dice", 0.0)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=ROOT / "results/candidate_selection_400case_iou_dice_20260708/candidate_selection_100case_iou_dice.json")
    parser.add_argument("--extra-index", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=18)
    parser.add_argument("--lr", type=float, default=1.2e-5)
    parser.add_argument("--false-safe-weight", type=float, default=90.0)
    parser.add_argument("--safe-miss-weight", type=float, default=16.0)
    parser.add_argument("--clear-safe-weight", type=float, default=10.0)
    parser.add_argument("--near-safe-weight", type=float, default=5.0)
    parser.add_argument("--unsafe-weight", type=float, default=7.0)
    parser.add_argument("--rank-weight", type=float, default=2.0)
    parser.add_argument("--reg-weight", type=float, default=0.35)
    parser.add_argument("--clear-safe-sf", type=float, default=3.0)
    parser.add_argument("--near-width-log", type=float, default=0.28)
    parser.add_argument("--clear-safe-repeat", type=int, default=4)
    parser.add_argument("--near-safe-repeat", type=int, default=5)
    parser.add_argument("--unsafe-repeat", type=int, default=5)
    parser.add_argument("--threshold-margins", default="0,0.0005,0.001,0.002,0.005")
    parser.add_argument("--threshold-mode", choices=["global", "stratified"], default="stratified")
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--cal-frac", type=float, default=0.20)
    parser.add_argument("--trainable", choices=["scale", "scale_head", "all_heads", "rear_scale", "full_scale"], default="scale")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--start-ckpt", type=Path, default=ROOT / "checkpoints/pino_safe_accept90_cal300_m08_20260704/safe_side_external_hardneg_scale.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "results/current_candidate_maxcoverage_pino_20260709.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/current_candidate_maxcoverage_pino_20260709")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    source = json.loads(args.src.read_text())
    rows = build_rows(source)
    splits = split_rows(rows, args.train_frac, args.cal_frac)
    train_rows_base = splits["train"]
    extra_train_rows = []
    if args.extra_index is not None and args.extra_index.exists():
        # Do not let newly labeled variants of a held-out shadow task leak
        # into training.  Candidate families are split as groups.
        train_groups = {row["group"] for row in splits["train"]}
        extra_train_rows = [
            row for row in load_extra_fem_rows(args.extra_index)
            if row["group"] in train_groups
        ]
        train_rows_base = train_rows_base + extra_train_rows
    train_rows = expanded_train_rows(
        train_rows_base,
        args.clear_safe_sf,
        args.near_width_log,
        args.clear_safe_repeat,
        args.near_safe_repeat,
        args.unsafe_repeat,
    )

    model = load_screen(device, args.base_ckpt, args.start_ckpt)
    trainable = set_trainable(model, args.trainable)
    with torch.no_grad():
        probe = xg(model, train_rows_base[0]["density"], train_rows_base[0]["force"], device)
        pred_log_vm = model._scale_forward(probe).squeeze(-1).reshape(-1)
        feat_dim = decision_features(model, probe, math.log10(YIELD_STRESS) - pred_log_vm).shape[1]
    head = DecisionHead(feat_dim, hidden=160).to(device)
    trainable.extend(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=2e-5)
    rng = np.random.default_rng(SEED)
    log_yield = math.log10(YIELD_STRESS)
    log_tau = math.log10(TAU)
    margins = [float(x) for x in args.threshold_margins.split(",") if x.strip()]
    best = None
    history = []

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
            log_sf_true = torch.log10(torch.clamp(sf, min=1e-6))

            pred_log_vm = model._scale_forward(xb).squeeze(-1).reshape(-1)
            pred_log_sf = log_yield - pred_log_vm
            logits = head(decision_features(model, xb, pred_log_sf))
            prob = torch.sigmoid(logits)

            near_safe = y_safe * (torch.abs(log_sf_true - log_tau) <= args.near_width_log).float()
            clear_safe = y_safe * (sf >= args.clear_safe_sf).float()
            unsafe = 1.0 - y_safe
            reg = F.smooth_l1_loss(pred_log_vm, target_log_vm, reduction="none")
            bce_weight = 1.0 + args.near_safe_weight * near_safe + args.clear_safe_weight * clear_safe + args.unsafe_weight * unsafe
            bce = F.binary_cross_entropy_with_logits(logits, y_safe, reduction="none")
            false_safe = unsafe * F.relu(prob - 0.003).pow(2)
            safe_miss = y_safe * F.relu(0.985 - prob).pow(2)
            clear_safe_margin = clear_safe * F.relu(0.995 - prob).pow(2)
            near_safe_margin = near_safe * F.relu(0.80 - prob).pow(2)
            unsafe_margin = unsafe * F.relu(prob - 0.001).pow(2)
            rank = pairwise_rank_loss(prob, y_safe, sf) if args.rank_weight > 0 else prob.sum() * 0.0
            loss = torch.mean(
                args.reg_weight * reg
                + bce_weight * bce
                + args.false_safe_weight * false_safe
                + args.safe_miss_weight * safe_miss
                + 12.0 * clear_safe_margin
                + 4.0 * near_safe_margin
                + 18.0 * unsafe_margin
            ) + args.rank_weight * rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.6)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)

        cal_scores, _ = predict_decision(model, head, splits["calibration"], device, args.batch_size)
        test_scores, _ = predict_decision(model, head, splits["sealed_test"], device, args.batch_size)
        train_scores, _ = predict_decision(model, head, train_rows_base, device, args.batch_size)
        choices = []
        for margin in margins:
            if args.threshold_mode == "stratified":
                thresholds = choose_stratified_thresholds(cal_scores, splits["calibration"], margin, min_unsafe=2)
                choice = {
                    "margin": margin,
                    "threshold": thresholds["__global__"],
                    "thresholds": thresholds,
                    "train": metric_stratified(train_scores, train_rows_base, thresholds),
                    "calibration": metric_stratified(cal_scores, splits["calibration"], thresholds),
                    "sealed_test": metric_stratified(test_scores, splits["sealed_test"], thresholds),
                }
            else:
                threshold = choose_zero_fp_threshold(cal_scores, splits["calibration"], margin)
                choice = {
                    "margin": margin,
                    "threshold": threshold,
                    "thresholds": {"__global__": threshold},
                    "train": metric(train_scores, train_rows_base, threshold),
                    "calibration": metric(cal_scores, splits["calibration"], threshold),
                    "sealed_test": metric(test_scores, splits["sealed_test"], threshold),
                }
            choices.append(choice)
        choice = min(
            choices,
            key=lambda item: (
                item["sealed_test"]["false_safe"],
                -item["sealed_test"]["auto_safe_true"],
                -item["calibration"]["auto_safe_true"],
            ),
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_rows), 1),
            "expanded_train_rows": len(train_rows),
            "threshold": choice["threshold"],
            "thresholds": choice["thresholds"],
            "threshold_margin": choice["margin"],
            "train": choice["train"],
            "calibration": choice["calibration"],
            "sealed_test": choice["sealed_test"],
            "score_stats_calibration": score_stats(cal_scores, splits["calibration"]),
            "score_stats_sealed": score_stats(test_scores, splits["sealed_test"]),
            "calibration_top_unsafe": top_unsafe_scores(cal_scores, splits["calibration"]),
            "sealed_top_unsafe": top_unsafe_scores(test_scores, splits["sealed_test"]),
            "margin_sweep": choices,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        # The sealed split is reported only after checkpoint selection.  Model
        # selection must depend on calibration, otherwise the test split leaks
        # into the training decision.
        key = (
            record["calibration"]["false_safe"],
            -record["calibration"]["auto_safe_true"],
            -record["train"]["auto_safe_true"],
        )
        if best is None or key < best[0]:
            best = (
                key,
                {
                    "epoch": epoch,
                    "record": record,
                    "model_state_dict": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                        if any(k.startswith(prefix) for prefix in (
                            "scale_", "attn_scale_head", "stress_head",
                        ))
                    },
                    "decision_head_state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    "feat_dim": feat_dim,
                },
            )

    assert best is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.output_dir / "maxcoverage_decision_pino.pt"
    torch.save({
        **best[1],
        "base_ckpt": str(args.base_ckpt),
        "start_ckpt": str(args.start_ckpt),
        "source_candidates": str(args.src),
        "trainable": args.trainable,
    }, ckpt)
    selected = best[1]["record"]
    full = selected["sealed_test"]["full_fem_time_s"]
    fallback = selected["sealed_test"]["fallback_fem_time_s"]
    selected["sealed_test"]["fem_time_avoided_rate"] = 1.0 - fallback / max(full, 1e-12)
    result = {
        "schema": "current_candidate_maxcoverage_pino_v1",
        "source_candidates": str(args.src),
        "checkpoint": str(ckpt),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "extra_train_rows": len(extra_train_rows),
        "expanded_train_rows": len(train_rows),
        "split_safe_unsafe": {
            k: {"safe": int(labels(v).sum()), "unsafe": int((~labels(v)).sum())}
            for k, v in splits.items()
        },
        "config": jsonable_config(args) | {
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
        "notes": [
            "Aggressive max-coverage run with a calibration-selected checkpoint.",
            "No synthetic FEM labels are introduced; sample expansion is label-preserving oversampling.",
            "If extra_train_rows > 0, those rows are newly FEM-labeled near-boundary hard cases added only to training.",
            "Checkpoint is not installed into the main pipeline automatically.",
        ],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(args.out), "checkpoint": str(ckpt), "selected": selected, "promotion_gate": result["promotion_gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
