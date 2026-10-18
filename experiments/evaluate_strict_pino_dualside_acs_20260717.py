#!/usr/bin/env python3
"""Calibration-only dual-side ACS audit for a frozen strict PINO.

The safe and unsafe conformal corrections are selected independently.  Policy
selection uses calibration tasks only; sealed rows are evaluated once after the
policy is frozen.  Prediction caching keeps repeated policy audits inexpensive.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_strict_pino_acs_v5_20260713 import (
    TAU,
    finite_quantile,
    load_model,
    make_dataset,
    metrics,
    predict_log_sf,
)


def parse_grid(value: str) -> list[float]:
    values = sorted({float(item) for item in value.split(",") if item.strip()})
    if not values or any(item <= 0.0 or item >= 1.0 for item in values):
        raise ValueError("Alpha grids must contain values strictly between zero and one.")
    return values


def decisions(pred: np.ndarray, threshold: float, q_safe: float,
              q_unsafe: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    auto_safe = pred - q_safe >= threshold
    auto_unsafe = pred + q_unsafe < threshold
    conflict = auto_safe & auto_unsafe
    # Fail closed if unusual corrections make both inequalities true.
    auto_safe = auto_safe & ~conflict
    auto_unsafe = auto_unsafe & ~conflict
    return auto_safe, auto_unsafe, conflict


def predict_or_load(args: argparse.Namespace, splits: dict[str, list[dict]],
                    device: torch.device) -> dict[str, np.ndarray]:
    if args.prediction_cache.is_file():
        cached = np.load(args.prediction_cache)
        if (len(cached["calibration"]) == len(splits["calibration"])
                and len(cached["sealed_test"]) == len(splits["sealed_test"])):
            print(f"Using prediction cache: {args.prediction_cache}", flush=True)
            return {name: np.asarray(cached[name], dtype=float)
                    for name in ("calibration", "sealed_test")}
        raise RuntimeError("Prediction cache does not match the current index splits.")

    model = load_model(args.checkpoint, device)
    pred = {
        name: predict_log_sf(
            model,
            make_dataset(args.index, rows, args.target_res),
            device,
            args.batch_size,
        )
        for name, rows in splits.items()
    }
    args.prediction_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.prediction_cache, **pred)
    return pred


def row_audit(records: list[dict], pred: np.ndarray, truth: np.ndarray,
              threshold: float, q_safe: float, q_unsafe: float) -> list[dict]:
    auto_safe, auto_unsafe, conflict = decisions(pred, threshold, q_safe, q_unsafe)
    truth_safe = truth >= threshold
    rows: list[dict] = []
    for index, record in enumerate(records):
        if auto_safe[index]:
            decision = "SAFE"
        elif auto_unsafe[index]:
            decision = "UNSAFE"
        else:
            decision = "DEFER"
        if auto_safe[index] and not truth_safe[index]:
            error = "false_safe"
        elif auto_unsafe[index] and truth_safe[index]:
            error = "false_unsafe"
        else:
            error = None
        rows.append({
            "row": index,
            "task": record.get("task"),
            "path": record.get("path"),
            "candidate": record.get("candidate"),
            "sf_fem": float(10.0 ** truth[index]),
            "sf_pino": float(10.0 ** pred[index]),
            "truth_safe": bool(truth_safe[index]),
            "decision": decision,
            "error": error,
            "conflict_deferred": bool(conflict[index]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--safe-alphas", default="0.005,0.01,0.02,0.05,0.10")
    parser.add_argument("--unsafe-alphas", default="0.02,0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--target-res", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.index.read_text())
    records = payload["rows"]
    splits = {
        name: [row for row in records if row["split"] == name]
        for name in ("calibration", "sealed_test")
    }
    if not all(splits.values()):
        raise RuntimeError("Both calibration and sealed_test splits are required.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pred = predict_or_load(args, splits, device)
    truth = {
        name: np.log10(np.asarray([float(row["sf"]) for row in rows], dtype=float))
        for name, rows in splits.items()
    }
    threshold = math.log10(TAU)
    residual = pred["calibration"] - truth["calibration"]
    cal_truth_safe = truth["calibration"] >= threshold
    safe_alphas = parse_grid(args.safe_alphas)
    unsafe_alphas = parse_grid(args.unsafe_alphas)

    candidates: list[dict] = []
    for alpha_safe in safe_alphas:
        q_safe = finite_quantile(residual, 1.0 - alpha_safe)
        for alpha_unsafe in unsafe_alphas:
            q_unsafe = finite_quantile(-residual, 1.0 - alpha_unsafe)
            cal_safe, cal_unsafe, conflicts = decisions(
                pred["calibration"], threshold, q_safe, q_unsafe
            )
            cal_metrics = metrics(cal_safe, cal_unsafe, cal_truth_safe, False)
            candidates.append({
                "alpha_safe": alpha_safe,
                "alpha_unsafe": alpha_unsafe,
                "q_safe_log_sf": q_safe,
                "q_unsafe_log_sf": q_unsafe,
                "calibration_conflicts": int(conflicts.sum()),
                "calibration_metrics": cal_metrics,
            })

    valid = [row for row in candidates
             if row["calibration_metrics"]["false_safe_count"] == 0]
    if not valid:
        raise RuntimeError("No zero-false-safe policy exists on the supplied calibration grid.")
    selected = max(valid, key=lambda row: (
        row["calibration_metrics"]["auto_decision_coverage"],
        row["calibration_metrics"]["auto_unsafe_coverage"],
        row["calibration_metrics"]["auto_safe_coverage"],
        -row["alpha_safe"],
    ))

    sealed_safe, sealed_unsafe, sealed_conflicts = decisions(
        pred["sealed_test"], threshold,
        selected["q_safe_log_sf"], selected["q_unsafe_log_sf"],
    )
    sealed_truth_safe = truth["sealed_test"] >= threshold
    selected["sealed_conflicts"] = int(sealed_conflicts.sum())
    selected["sealed_metrics"] = metrics(
        sealed_safe, sealed_unsafe, sealed_truth_safe, True
    )

    audit = row_audit(
        splits["sealed_test"], pred["sealed_test"], truth["sealed_test"],
        threshold, selected["q_safe_log_sf"], selected["q_unsafe_log_sf"],
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sealed_row_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    errors = [row for row in audit if row["error"]]
    (args.out_dir / "sealed_errors.json").write_text(
        json.dumps(errors, indent=2) + "\n"
    )
    result = {
        "schema": "strict_pino_dualside_acs_v1",
        "checkpoint": str(args.checkpoint),
        "index": str(args.index),
        "threshold_sf": TAU,
        "selection_protocol": (
            "calibration only; maximize automatic decision coverage subject to "
            "zero calibration false-safe decisions; safe and unsafe alpha grids "
            "are independent"
        ),
        "rows": {name: len(rows) for name, rows in splits.items()},
        "safe_alpha_grid": safe_alphas,
        "unsafe_alpha_grid": unsafe_alphas,
        "selected_policy": selected,
        "all_calibration_policies": candidates,
        "sealed_error_counts": {
            "false_safe": sum(row["error"] == "false_safe" for row in errors),
            "false_unsafe": sum(row["error"] == "false_unsafe" for row in errors),
        },
        "validity_note": (
            "Sealed metrics are an audit, not a selection input. A subsequent "
            "model or policy revision requires a fresh task-disjoint sealed split."
        ),
    }
    (args.out_dir / "dualside_acs_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({
        "out": str(args.out_dir),
        "selected": selected,
        "sealed_error_counts": result["sealed_error_counts"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
