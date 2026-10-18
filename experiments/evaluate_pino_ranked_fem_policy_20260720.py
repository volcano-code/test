#!/usr/bin/env python3
"""Freeze Top-k on calibration tasks and audit it once on sealed tasks."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.evaluate_raw_pino_fem_reduction_20260719 import task_topk


METHOD = "raw_pino_safe_first_then_material"


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [max(0.0, center - radius), min(1.0, center + radius)]


def split_rows(payload: dict, split: str) -> list[dict]:
    return [row for row in payload["rows"] if row["split"] == split]


def evaluate_split(
    rows: list[dict], pred_log: np.ndarray, threshold_sf: float, topks: list[int]
) -> tuple[list[dict], list[dict]]:
    if len(rows) != len(pred_log):
        raise RuntimeError(f"Prediction count {len(pred_log)} != row count {len(rows)}")
    summaries, details = task_topk(rows, np.power(10.0, pred_log), threshold_sf, topks)
    filtered = [row for row in summaries if row["method"] == METHOD]
    for row in filtered:
        n = int(row["tasks_with_fem_safe_candidate"])
        row["safe_selection_95ci"] = wilson(int(row["tasks_with_safe_selection"]), n)
        row["exact_agreement_95ci"] = wilson(int(row["exact_lightest_safe_matches"]), n)
    return filtered, [row for row in details if row["method"] == METHOD]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration-split", default="calibration")
    parser.add_argument("--sealed-split", default="sealed_test")
    parser.add_argument("--threshold-sf", type=float, default=1.3)
    parser.add_argument("--top-k", default="1,2,3,5,10,20")
    parser.add_argument("--min-calibration-safe-selection", type=float, default=0.95)
    parser.add_argument("--min-calibration-exact-agreement", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"PINO checkpoint not found: {args.checkpoint}")

    payload = json.loads(args.index.read_text())
    calibration = split_rows(payload, args.calibration_split)
    sealed = split_rows(payload, args.sealed_split)
    calibration_tasks = {str(row["task"]) for row in calibration}
    sealed_tasks = {str(row["task"]) for row in sealed}
    overlap = calibration_tasks & sealed_tasks
    if overlap:
        raise RuntimeError(f"Task-disjoint audit violated by {len(overlap)} overlapping tasks")

    cache = np.load(args.predictions)
    topks = sorted({int(value) for value in args.top_k.split(",")})
    calibration_summary, calibration_details = evaluate_split(
        calibration, np.asarray(cache[args.calibration_split], dtype=np.float64),
        args.threshold_sf, topks,
    )
    admissible = [
        row for row in calibration_summary
        if row["safe_selection_rate"] >= args.min_calibration_safe_selection
        and row["exact_selection_agreement"] >= args.min_calibration_exact_agreement
    ]
    if not admissible:
        raise RuntimeError(
            "No k satisfies both calibration constraints; inspect calibration_sweep.csv"
        )
    frozen = min(admissible, key=lambda row: int(row["top_k"]))
    frozen_k = int(frozen["top_k"])
    sealed_summary, sealed_details = evaluate_split(
        sealed, np.asarray(cache[args.sealed_split], dtype=np.float64),
        args.threshold_sf, [frozen_k],
    )
    sealed_result = sealed_summary[0]

    policy = {
        "schema": "pino_ranked_fem_policy_v1",
        "selection_protocol": "k selected on task-disjoint calibration tasks only",
        "index": str(args.index.resolve()),
        "predictions": str(args.predictions.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "threshold_sf": float(args.threshold_sf),
        "ranking": "predicted-safe first, then material ascending, predicted SF descending",
        "top_k": frozen_k,
        "fallback": "expand",
        "fallback_description": "expand only when the verified prefix contains no FEM-safe candidate",
        "calibration_constraints": {
            "minimum_safe_selection_rate": args.min_calibration_safe_selection,
            "minimum_exact_selection_agreement": args.min_calibration_exact_agreement,
        },
        "calibration_tasks": len(calibration_tasks),
        "sealed_tasks": len(sealed_tasks),
    }
    result = {
        "schema": "pino_ranked_fem_task_disjoint_audit_v1",
        "policy": policy,
        "calibration_selected": frozen,
        "sealed_test": sealed_result,
        "interpretation": (
            "PINO only orders candidates. Every selected candidate is FEM-verified; "
            "exact agreement compares against exhaustive-FEM lightest-safe selection."
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frozen_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "calibration_details.json").write_text(json.dumps(calibration_details, indent=2) + "\n")
    (args.output / "sealed_details.json").write_text(json.dumps(sealed_details, indent=2) + "\n")
    with (args.output / "calibration_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(calibration_summary[0]))
        writer.writeheader(); writer.writerows(calibration_summary)
    with (args.output / "sealed_result.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sealed_result))
        writer.writeheader(); writer.writerow(sealed_result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
