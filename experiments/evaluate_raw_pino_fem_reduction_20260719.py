#!/usr/bin/env python3
"""Evaluate Raw-PINO pre-screening as a reduction of voxel-FEM calls.

This is a validation-set diagnostic, not a conformal policy.  It evaluates:
1) a symmetric uncertainty band around the physical SF threshold; and
2) task-level recovery of the lightest FEM-safe candidate after checking only
   the first k candidates ordered by Raw PINO and material use.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shadowforge_pipeline.pino_ranked_fem import rank_pino_candidates


def triage_sweep(pred_log: np.ndarray, truth_log: np.ndarray,
                 threshold_log: float, margins: list[float]) -> list[dict]:
    truth_safe = truth_log >= threshold_log
    rows = []
    for margin in margins:
        auto_safe = pred_log >= threshold_log + margin
        auto_unsafe = pred_log < threshold_log - margin
        defer = ~(auto_safe | auto_unsafe)
        false_safe = int((auto_safe & ~truth_safe).sum())
        false_unsafe = int((auto_unsafe & truth_safe).sum())
        rows.append({
            "margin_log10_sf": float(margin),
            "auto_decision_count": int((auto_safe | auto_unsafe).sum()),
            "auto_decision_rate": float((auto_safe | auto_unsafe).mean()),
            "auto_safe_count": int(auto_safe.sum()),
            "auto_unsafe_count": int(auto_unsafe.sum()),
            "defer_count": int(defer.sum()),
            "fem_call_rate": float(defer.mean()),
            "fem_calls_avoided_rate": float(1.0 - defer.mean()),
            "false_safe_count": false_safe,
            "false_safe_rate_over_unsafe": float(
                false_safe / max(int((~truth_safe).sum()), 1)
            ),
            "false_unsafe_count": false_unsafe,
            "false_unsafe_rate_over_safe": float(
                false_unsafe / max(int(truth_safe.sum()), 1)
            ),
            "final_false_safe_after_fem": false_safe,
        })
    return rows


def task_topk(rows: list[dict], pred_sf: np.ndarray, threshold_sf: float,
              topks: list[int]) -> tuple[list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for index, (row, score) in enumerate(zip(rows, pred_sf)):
        item = dict(row)
        item["row_index"] = index
        item["sf_pino"] = float(score)
        item["sf_fem"] = float(row["sf"])
        item["truth_safe"] = bool(float(row["sf"]) >= threshold_sf)
        material = row.get("delivery_occupied")
        if material is None:
            material = row.get("proxy_occupied", row.get("occupied_voxels"))
        item["material"] = int(material)
        grouped[str(row["task"])].append(item)

    task_refs = []
    for task, candidates in sorted(grouped.items()):
        fem_safe = [item for item in candidates if item["truth_safe"]]
        if not fem_safe:
            continue
        reference = min(fem_safe, key=lambda item: (item["material"], item["row_index"]))
        # Prefer predicted-safe candidates; within each class, check lighter
        # candidates first.  Predicted-unsafe rows remain available after them.
        ordered = sorted(
            candidates,
            key=lambda item: (
                item["sf_pino"] < threshold_sf,
                item["material"],
                -item["sf_pino"],
                item["row_index"],
            ),
        )
        task_refs.append({"task": task, "reference": reference, "ordered": ordered})

    summaries = []
    details = []
    total_candidates = sum(len(item["ordered"]) for item in task_refs)
    methods = {
        "raw_pino_safe_first_then_material": lambda rows: rank_pino_candidates(
            rows, threshold_sf
        ),
        "material_only": lambda rows: sorted(
            rows, key=lambda row: (row["material"], row["row_index"])
        ),
        "raw_pino_sf_descending": lambda rows: sorted(
            rows,
            key=lambda row: (-row["sf_pino"], row["material"], row["row_index"]),
        ),
    }
    for method, order_fn in methods.items():
        for k in topks:
            selected_tasks = 0
            exact_matches = 0
            fem_calls = 0
            material_ratios = []
            for item in task_refs:
                ordered = order_fn(item["ordered"])
                checked = ordered[: min(k, len(ordered))]
                fem_calls += len(checked)
                checked_safe = [row for row in checked if row["truth_safe"]]
                selected = min(
                    checked_safe,
                    key=lambda row: (row["material"], row["row_index"]),
                ) if checked_safe else None
                if selected is not None:
                    selected_tasks += 1
                    exact = selected["path"] == item["reference"]["path"]
                    exact_matches += int(exact)
                    material_ratios.append(
                        selected["material"] / max(item["reference"]["material"], 1)
                    )
                else:
                    exact = False
                details.append({
                    "method": method,
                    "top_k": k,
                    "task": item["task"],
                    "reference_path": item["reference"]["path"],
                    "selected_path": selected["path"] if selected else None,
                    "selected_safe": selected is not None,
                    "exact_lightest_safe_match": exact,
                    "fem_calls": len(checked),
                })
            n_tasks = len(task_refs)
            summaries.append({
                "method": method,
                "top_k": k,
                "tasks_with_fem_safe_candidate": n_tasks,
                "tasks_with_safe_selection": selected_tasks,
                "safe_selection_rate": selected_tasks / max(n_tasks, 1),
                "exact_lightest_safe_matches": exact_matches,
                "exact_selection_agreement": exact_matches / max(n_tasks, 1),
                "mean_material_ratio_vs_full_fem": (
                    float(np.mean(material_ratios)) if material_ratios else None
                ),
                "fem_calls": fem_calls,
                "full_fem_calls": total_candidates,
                "fem_call_reduction_rate": 1.0 - fem_calls / max(total_candidates, 1),
                "selected_false_safe_count": 0,
            })
    return summaries, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--threshold-sf", type=float, default=1.3)
    parser.add_argument("--margins", default="0,0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50")
    parser.add_argument("--no-fit-zero-fs-margin", action="store_true")
    parser.add_argument("--top-k", default="1,2,3,5,10,20")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.index.read_text())
    rows = [row for row in payload["rows"] if row["split"] == args.split]
    cache = np.load(args.predictions)
    pred_log = np.asarray(cache[args.split], dtype=np.float64)
    if len(rows) != len(pred_log):
        raise RuntimeError(f"Prediction count {len(pred_log)} != row count {len(rows)}")
    truth_log = np.log10(np.asarray([float(row["sf"]) for row in rows]))
    threshold_log = math.log10(args.threshold_sf)

    margins = [float(value) for value in args.margins.split(",")]
    truth_safe = truth_log >= threshold_log
    # Smallest validation-fitted margin that excludes every known unsafe row
    # from automatic SAFE.  This is diagnostic and must not be applied to the
    # sealed split without freezing it first.
    if not args.no_fit_zero_fs_margin:
        unsafe_offsets = pred_log[~truth_safe] - threshold_log
        fitted_zero_fs_margin = float(max(0.0, unsafe_offsets.max()) + 1e-9)
        margins = sorted(set(margins + [fitted_zero_fs_margin]))
    topks = [int(value) for value in args.top_k.split(",")]
    sweep = triage_sweep(pred_log, truth_log, threshold_log, margins)
    topk, details = task_topk(rows, np.power(10.0, pred_log), args.threshold_sf, topks)
    zero_fs = [row for row in sweep if row["false_safe_count"] == 0]
    best_zero_fs = max(zero_fs, key=lambda row: row["auto_decision_rate"]) if zero_fs else None

    result = {
        "schema": "raw_pino_fem_reduction_diagnostic_v1",
        "scope": (
            "validation diagnostic only; no ACS and no sealed-test policy claim"
            if args.split == "calibration"
            else "sealed audit of a frozen Raw-PINO screening rule; no ACS"
        ),
        "index": str(args.index),
        "predictions": str(args.predictions),
        "split": args.split,
        "threshold_sf": args.threshold_sf,
        "rows": len(rows),
        "tasks": len({row["task"] for row in rows}),
        "safe": int((truth_log >= threshold_log).sum()),
        "unsafe": int((truth_log < threshold_log).sum()),
        "best_zero_false_safe_margin": best_zero_fs,
        "margin_sweep": sweep,
        "task_topk": topk,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "task_topk_details.json").write_text(json.dumps(details, indent=2) + "\n")
    for name, records in (("margin_sweep.csv", sweep), ("task_topk.csv", topk)):
        with (args.output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
