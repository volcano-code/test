#!/usr/bin/env python3
"""Audit a task-grouped decision PINO with calibration-only ACS thresholds.

The model checkpoint is chosen on calibration only by the trainer.  This script
then reports raw PINO, the learned decision head, and one-sided conformal
safe-acceptance sweeps on the disjoint sealed task split.  No result produced
here changes a deployment checkpoint.
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

from train_current_candidate_decision_aware_pino_20260708 import (  # noqa: E402
    DEFAULT_SRC,
    YIELD_STRESS,
    build_rows,
    labels,
    load_scale_model,
    metric_with_accept,
    split_rows,
)
from train_decision_aware_pino import DecisionHead, predict_decision  # noqa: E402
from train_pino_safe_side_boundary import SEED, TAU  # noqa: E402


def quantile_higher(values: np.ndarray, q: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    idx = min(len(values) - 1, max(0, int(math.ceil(q * len(values))) - 1))
    return float(values[idx])


def one_sided_upper_bound(errors: int, total: int, confidence: float = 0.95) -> float:
    if total <= 0:
        return 1.0
    if errors == 0:
        return float(1.0 - (1.0 - confidence) ** (1.0 / total))
    try:
        from scipy.stats import beta
        return float(beta.ppf(confidence, errors + 1, total - errors))
    except Exception:
        return 1.0


def enrich(metrics: dict) -> dict:
    accept = int(metrics["auto_safe"])
    errors = int(metrics["false_safe"])
    metrics["false_safe_rate_among_accepted"] = float(errors / accept) if accept else None
    metrics["false_safe_upper95_among_accepted"] = one_sided_upper_bound(errors, accept)
    metrics["defer_rate"] = float(1.0 - metrics["autonomous_rate"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--start-ckpt", type=Path, default=ROOT / "checkpoints/v29c_v27_pinball_s2606/best_scale_sf.pt")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--alpha", default="0.50,0.30,0.20,0.10,0.05,0.01")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source = json.loads(args.src.read_text())
    splits = split_rows(build_rows(source), 0.60, 0.20)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = load_scale_model(device, args.base_ckpt, args.start_ckpt)
    for name, values in state["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(values)
    model.scale_head_v2.load_state_dict(state["scale_head_v2_state_dict"])
    head = DecisionHead(int(state["feat_dim"])).to(device)
    head.load_state_dict(state["decision_head_state_dict"])

    scores = {}
    log_vms = {}
    for name, rows in splits.items():
        scores[name], log_vms[name] = predict_decision(model, head, rows, device, args.batch_size)
    log_yield = math.log10(YIELD_STRESS)
    log_tau = math.log10(TAU)
    pred_log_sf = {name: log_yield - value for name, value in log_vms.items()}

    cal_truth = labels(splits["calibration"])
    decision_threshold = float(state["threshold"])
    learned = {
        name: enrich(metric_with_accept(scores[name] >= decision_threshold, rows))
        for name, rows in splits.items()
    }
    raw = {
        name: enrich(metric_with_accept(pred_log_sf[name] >= log_tau, rows))
        for name, rows in splits.items()
    }
    residual = pred_log_sf["calibration"] - np.log10(np.asarray([row["sf"] for row in splits["calibration"]], dtype=float))
    sweep = []
    for alpha in [float(x) for x in args.alpha.split(",") if x.strip()]:
        # A one-sided lower bound subtracts the upper tail of safety
        # overestimation measured on calibration tasks only.
        correction = quantile_higher(residual, min(1.0, (len(residual) + 1) * (1.0 - alpha) / len(residual)))
        entry = {"alpha": alpha, "calibration_overprediction_correction": correction}
        for name, rows in splits.items():
            lower = pred_log_sf[name] - correction
            entry[name] = enrich(metric_with_accept(lower >= log_tau, rows))
        sweep.append(entry)

    result = {
        "schema": "taskgrouped_pino_acs_audit_v1",
        "checkpoint": str(args.checkpoint),
        "task_protocol": "ordered silhouette triples are disjoint across train/calibration/sealed test",
        "threshold_protocol": "learned decision threshold and all conformal corrections use calibration only",
        "safety_label": "voxel-FEM safety factor >= 1.3",
        "split_rows": {name: len(rows) for name, rows in splits.items()},
        "split_safe_unsafe": {name: {"safe": int(labels(rows).sum()), "unsafe": int((~labels(rows)).sum())} for name, rows in splits.items()},
        "raw_pino_threshold": raw,
        "decision_head_calibrated": learned,
        "one_sided_acs_risk_coverage_sweep": sweep,
        "notes": [
            "ACS auto-accepts only safe decisions; all other rows are deferred to voxel FEM.",
            "False-safe upper bounds are binomial one-sided 95% bounds conditional on accepted rows.",
            "This audit is not evidence that PINO replaces FEM outside this frozen candidate domain.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "sealed_decision": learned["sealed_test"], "acs": sweep}, indent=2))


if __name__ == "__main__":
    main()
