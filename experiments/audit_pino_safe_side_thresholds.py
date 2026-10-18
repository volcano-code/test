#!/usr/bin/env python3
"""Audit safe-side PINO thresholds on held-out candidate and external rows."""

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
from train_pino_aug_hardneg import SCALE_BRANCH, load_geoattn_model
from train_pino_safe_side_boundary import (
    SEED,
    load_candidate_rows,
    predict,
    threshold_safe_metrics,
)
from train_pino_safe_side_external_hardneg import split_external


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--external-calibration", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, head, _ = load_geoattn_model(device, args.base_ckpt, args.checkpoint)
    for name, state in payload["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(state)
    if "attn_scale_head_state_dict" in payload:
        head.load_state_dict(payload["attn_scale_head_state_dict"])

    candidate_rows = load_candidate_rows()
    test_rows = subset(candidate_rows, split_groups()["heldout_test"])
    external_rows = load_scaleup_rows()
    for row in external_rows:
        row["source"] = "scaleup"
    _, external_heldout, external_split = split_external(external_rows, args.external_calibration, SEED)

    candidate_pred = predict(model, test_rows, device, args.batch_size)
    external_pred = predict(model, external_heldout, device, args.batch_size)
    base = float(payload.get("threshold_log_sf", math.nan))
    log_yield = math.log10(50e6)
    candidate_scores = log_yield - candidate_pred
    external_scores = log_yield - external_pred
    thresholds = sorted(
        set(float(v) for v in np.concatenate([candidate_scores, external_scores])),
        reverse=True,
    )
    audit = []
    for threshold in thresholds:
        c = threshold_safe_metrics(candidate_pred, test_rows, threshold)
        e = threshold_safe_metrics(external_pred, external_heldout, threshold)
        audit.append({
            "threshold": threshold,
            "candidate_heldout": c,
            "external_heldout": e,
            "joint_false_pass": c["auto_false_pass"] + e["auto_false_pass"],
            "joint_auto_safe_true": c["auto_safe_true"] + e["auto_safe_true"],
        })
    zero_fp = [row for row in audit if row["joint_false_pass"] == 0]
    best_zero_fp = max(zero_fp, key=lambda row: row["joint_auto_safe_true"]) if zero_fp else None
    result = {
        "schema": "pino_safe_side_threshold_audit_v1",
        "checkpoint": str(args.checkpoint),
        "base_threshold": base,
        "external_calibration": args.external_calibration,
        "external_split": {
            "seed": external_split["seed"],
            "heldout_count": len(external_split["heldout_indices"]),
        },
        "best_zero_false_pass": best_zero_fp,
        "audit": audit,
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "best_zero_false_pass": best_zero_fp,
    }, indent=2))


if __name__ == "__main__":
    main()
