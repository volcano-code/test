#!/usr/bin/env python3
"""Audit a fail-closed union of two zero-false-pass PINO safe screens.

The union accepts a candidate as automatically safe only when at least one
individually calibrated screen accepts it.  This script evaluates the union on
the same sealed candidate and external held-out splits used by the threshold
audit scripts; it does not modify checkpoints or pipeline defaults.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_candidate_pino_field import load_scaleup_rows
from train_candidate_pino_grouped import split_groups, subset
from train_pino_aug_hardneg import load_geoattn_model
from train_pino_safe_side_boundary import SEED, load_candidate_rows, predict
from train_pino_safe_side_external_hardneg import split_external

YIELD_STRESS = 50e6
TAU = 1.3


def load_screen(device: torch.device, base_ckpt: Path, checkpoint: Path):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model, head, _ = load_geoattn_model(device, base_ckpt, checkpoint)
    for name, state in payload["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(state)
    if "attn_scale_head_state_dict" in payload:
        head.load_state_dict(payload["attn_scale_head_state_dict"])
    return model


def metrics(accepted: np.ndarray, rows: list[dict]) -> dict:
    truth = np.asarray([float(row["sf"]) >= TAU for row in rows], dtype=bool)
    return {
        "n": int(len(rows)),
        "safe": int(truth.sum()),
        "unsafe": int((~truth).sum()),
        "auto_safe_accept_count": int(accepted.sum()),
        "auto_safe_true": int((accepted & truth).sum()),
        "auto_false_pass": int((accepted & ~truth).sum()),
        "auto_safe_rate": float((accepted & truth).sum() / max(int(truth.sum()), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument(
        "--screen-a",
        type=Path,
        default=ROOT / "checkpoints/pino_safe_accept90_cal300_m08_20260704/safe_side_external_hardneg_scale.pt",
    )
    parser.add_argument(
        "--screen-b",
        type=Path,
        default=ROOT / "checkpoints/pino_safe_accept90_cal300_m14_single_20260706/safe_side_external_hardneg_scale.pt",
    )
    parser.add_argument("--threshold-a", type=float, default=0.1718017304102375)
    parser.add_argument("--threshold-b", type=float, default=0.22335402609505195)
    parser.add_argument("--external-calibration", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=ROOT / "results/pino_union_safe_screen_20260708.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    candidate_rows = load_candidate_rows()
    candidate_heldout = subset(candidate_rows, split_groups()["heldout_test"])
    external_rows = load_scaleup_rows()
    for row in external_rows:
        row["source"] = "scaleup"
    _, external_heldout, external_split = split_external(external_rows, args.external_calibration, SEED)

    rows_by_split = {
        "candidate_heldout": candidate_heldout,
        "external_heldout": external_heldout,
    }

    screens = [
        ("m08_cal300", args.screen_a, float(args.threshold_a)),
        ("m14_single_cal300", args.screen_b, float(args.threshold_b)),
    ]
    log_yield = math.log10(YIELD_STRESS)
    split_accepts: dict[str, list[np.ndarray]] = {k: [] for k in rows_by_split}
    individual = {}
    for name, checkpoint, threshold in screens:
        model = load_screen(device, args.base_ckpt, checkpoint)
        individual[name] = {"checkpoint": str(checkpoint), "threshold_log_sf": threshold}
        for split, rows in rows_by_split.items():
            pred_log_vm = predict(model, rows, device, args.batch_size)
            scores = log_yield - pred_log_vm
            accepted = scores >= threshold
            split_accepts[split].append(accepted)
            individual[name][split] = metrics(accepted, rows)

    union = {}
    joint_false = 0
    joint_true = 0
    for split, rows in rows_by_split.items():
        accepted = np.logical_or.reduce(split_accepts[split])
        union[split] = metrics(accepted, rows)
        joint_false += union[split]["auto_false_pass"]
        joint_true += union[split]["auto_safe_true"]

    result = {
        "schema": "pino_union_safe_screen_v1",
        "screens": individual,
        "union_rule": "auto-safe if any individually calibrated screen accepts; otherwise defer to FEM",
        "external_split": {
            "seed": external_split["seed"],
            "heldout_count": len(external_split["heldout_indices"]),
        },
        "union": {
            **union,
            "joint_false_pass": int(joint_false),
            "joint_auto_safe_true": int(joint_true),
            "promotion_gate": {
                "zero_false_pass": joint_false == 0,
                "candidate_auto_safe_at_least_85pct": union["candidate_heldout"]["auto_safe_rate"] >= 0.85,
                "external_auto_safe_at_least_85pct": union["external_heldout"]["auto_safe_rate"] >= 0.85,
            },
        },
    }
    result["union"]["promotion_eligible_85pct"] = all(result["union"]["promotion_gate"].values())
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["union"], indent=2))


if __name__ == "__main__":
    main()
