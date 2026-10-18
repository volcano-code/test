#!/usr/bin/env python3
"""Exploratory calibration-only conditional ACS audit for frozen v15 PINO.

The previously opened sealed split is used only as an audit.  Any policy adopted
for paper reporting must be frozen and evaluated on a fresh task-disjoint split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from evaluate_strict_pino_acs_v5_20260713 import finite_quantile, metrics
from evaluate_strict_pino_dualside_acs_20260717 import decisions


TAU = 1.3
SAFE_ALPHAS = (0.005, 0.01, 0.02, 0.05, 0.10)
UNSAFE_ALPHAS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)


def grouped_policy(
    name: str,
    calibration_groups: np.ndarray,
    sealed_groups: np.ndarray,
    pred: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
) -> dict:
    threshold = math.log10(TAU)
    residual = pred["calibration"] - truth["calibration"]
    cal_truth_safe = truth["calibration"] >= threshold
    candidates: list[dict] = []

    unique_groups = sorted(np.unique(calibration_groups).tolist())
    for alpha_safe in SAFE_ALPHAS:
        for alpha_unsafe in UNSAFE_ALPHAS:
            q_safe = {
                str(group): finite_quantile(
                    residual[calibration_groups == group], 1.0 - alpha_safe
                )
                for group in unique_groups
            }
            q_unsafe = {
                str(group): finite_quantile(
                    -residual[calibration_groups == group], 1.0 - alpha_unsafe
                )
                for group in unique_groups
            }
            cal_q_safe = np.asarray([q_safe[str(group)] for group in calibration_groups])
            cal_q_unsafe = np.asarray([q_unsafe[str(group)] for group in calibration_groups])
            auto_safe = pred["calibration"] - cal_q_safe >= threshold
            auto_unsafe = pred["calibration"] + cal_q_unsafe < threshold
            conflict = auto_safe & auto_unsafe
            auto_safe &= ~conflict
            auto_unsafe &= ~conflict
            cal_metrics = metrics(auto_safe, auto_unsafe, cal_truth_safe, False)
            candidates.append({
                "alpha_safe": alpha_safe,
                "alpha_unsafe": alpha_unsafe,
                "q_safe_log_sf_by_group": q_safe,
                "q_unsafe_log_sf_by_group": q_unsafe,
                "calibration_conflicts": int(conflict.sum()),
                "calibration_metrics": cal_metrics,
            })

    valid = [
        row for row in candidates
        if row["calibration_metrics"]["false_safe_count"] == 0
    ]
    if not valid:
        raise RuntimeError(f"No zero-false-safe policy for strategy {name}")
    selected = max(valid, key=lambda row: (
        row["calibration_metrics"]["auto_decision_coverage"],
        row["calibration_metrics"]["auto_unsafe_coverage"],
        row["calibration_metrics"]["auto_safe_coverage"],
        -row["alpha_safe"],
    ))

    fallback_safe = max(selected["q_safe_log_sf_by_group"].values())
    fallback_unsafe = max(selected["q_unsafe_log_sf_by_group"].values())
    sealed_q_safe = np.asarray([
        selected["q_safe_log_sf_by_group"].get(str(group), fallback_safe)
        for group in sealed_groups
    ])
    sealed_q_unsafe = np.asarray([
        selected["q_unsafe_log_sf_by_group"].get(str(group), fallback_unsafe)
        for group in sealed_groups
    ])
    sealed_safe = pred["sealed_test"] - sealed_q_safe >= threshold
    sealed_unsafe = pred["sealed_test"] + sealed_q_unsafe < threshold
    sealed_conflict = sealed_safe & sealed_unsafe
    sealed_safe &= ~sealed_conflict
    sealed_unsafe &= ~sealed_conflict
    sealed_truth_safe = truth["sealed_test"] >= threshold
    selected["sealed_conflicts"] = int(sealed_conflict.sum())
    selected["sealed_metrics"] = metrics(
        sealed_safe, sealed_unsafe, sealed_truth_safe, True
    )
    return {
        "strategy": name,
        "groups": unique_groups,
        "selected_policy": selected,
    }


def quantile_groups(
    calibration_values: np.ndarray,
    sealed_values: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    edges = np.concatenate((
        [-np.inf],
        np.quantile(calibration_values, np.arange(1, bins) / bins),
        [np.inf],
    ))
    return (
        np.digitize(calibration_values, edges[1:-1]),
        np.digitize(sealed_values, edges[1:-1]),
        [float(item) for item in edges],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--branch", choices=("all", "fixed", "free"), default="all")
    args = parser.parse_args()

    records = json.loads(args.index.read_text())["rows"]
    full_split_records = {
        name: [row for row in records if row["split"] == name]
        for name in ("calibration", "sealed_test")
    }
    cached = np.load(args.predictions)
    full_pred = {name: np.asarray(cached[name], dtype=float) for name in full_split_records}
    for name in full_pred:
        if len(full_pred[name]) != len(full_split_records[name]):
            raise RuntimeError(f"Prediction cache mismatch for {name}")
    if args.branch == "all":
        split_records = full_split_records
        pred = full_pred
    else:
        split_records = {}
        pred = {}
        for name, rows in full_split_records.items():
            mask = np.asarray([row["branch"] == args.branch for row in rows], dtype=bool)
            split_records[name] = [row for row, keep in zip(rows, mask) if keep]
            pred[name] = full_pred[name][mask]
    truth = {
        name: np.log10(np.asarray([row["sf"] for row in rows], dtype=float))
        for name, rows in split_records.items()
    }

    strategies: list[dict] = []
    strategies.append(grouped_policy(
        "global",
        np.zeros(len(split_records["calibration"]), dtype=int),
        np.zeros(len(split_records["sealed_test"]), dtype=int),
        pred,
        truth,
    ))

    branch_groups = {
        name: np.asarray([
            0 if row["branch"] == "fixed" else 1
            for row in rows
        ], dtype=int)
        for name, rows in split_records.items()
    }
    branch_result = grouped_policy(
        "mondrian_branch_fixed_free",
        branch_groups["calibration"],
        branch_groups["sealed_test"],
        pred,
        truth,
    )
    branch_result["group_labels"] = {"0": "fixed", "1": "free"}
    strategies.append(branch_result)

    for bins in (2, 3, 4):
        cal_group, sealed_group, edges = quantile_groups(
            pred["calibration"], pred["sealed_test"], bins
        )
        result = grouped_policy(
            f"predicted_log_sf_{bins}_bins",
            cal_group,
            sealed_group,
            pred,
            truth,
        )
        result["bin_edges"] = edges
        strategies.append(result)

    calibration_occupancy = np.log10(np.asarray([
        row["proxy_occupied"] for row in split_records["calibration"]
    ], dtype=float))
    sealed_occupancy = np.log10(np.asarray([
        row["proxy_occupied"] for row in split_records["sealed_test"]
    ], dtype=float))
    for bins in (2, 3, 4):
        cal_group, sealed_group, edges = quantile_groups(
            calibration_occupancy, sealed_occupancy, bins
        )
        result = grouped_policy(
            f"proxy_occupancy_{bins}_bins",
            cal_group,
            sealed_group,
            pred,
            truth,
        )
        result["bin_edges_log10"] = edges
        strategies.append(result)

    selected = max(strategies, key=lambda row: (
        row["selected_policy"]["calibration_metrics"]["auto_decision_coverage"],
        row["selected_policy"]["calibration_metrics"]["auto_safe_coverage"],
    ))
    output = {
        "schema": "conditional_acs_exploratory_v1",
        "index": str(args.index.resolve()),
        "predictions": str(args.predictions.resolve()),
        "threshold_sf": TAU,
        "branch_filter": args.branch,
        "selection_protocol": (
            "strategy and quantiles selected using calibration rows only; existing "
            "previously opened sealed split used as exploratory audit only"
        ),
        "deployment_features": ["PINO predicted log10 SF", "fixed/free branch", "C48 proxy occupancy"],
        "strategies": strategies,
        "calibration_selected_strategy": selected["strategy"],
        "calibration_selected_policy": selected["selected_policy"],
        "validity_note": (
            "Promising results require confirmation after freezing the policy on a "
            "fresh task-disjoint sealed split."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "selected_strategy": output["calibration_selected_strategy"],
        "selected_policy": output["calibration_selected_policy"],
    }, indent=2))


if __name__ == "__main__":
    main()
