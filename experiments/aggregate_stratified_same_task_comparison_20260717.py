#!/usr/bin/env python3
"""Aggregate the frozen 12-task Mitra/Sadekar/ShadowForge comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/lin/xvxinyao/ShadowForge")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "results/mitra_stratified_low6_high6_fixed3_20260716/benchmark_manifest.json",
    )
    parser.add_argument(
        "--sadekar",
        type=Path,
        default=ROOT / "results/mitra_stratified_low6_high6_unified_eval_20260717/sadekar/unified_metrics.json",
    )
    parser.add_argument(
        "--shadowforge",
        type=Path,
        default=ROOT / "results/mitra_stratified_low6_high6_shadowforge_20260717/SUMMARY.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/mitra_stratified_low6_high6_unified_eval_20260717",
    )
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text())
    sadekar = json.loads(args.sadekar.read_text())
    shadowforge = json.loads(args.shadowforge.read_text())
    expected = {case["case_id"] for case in benchmark["cases"]}
    sf_rows = [row for row in shadowforge if row.get("case_id") in expected and row.get("mean_iou") is not None]
    if {row["case_id"] for row in sf_rows} != expected:
        missing = sorted(expected - {row["case_id"] for row in sf_rows})
        raise RuntimeError(f"ShadowForge same-task results are incomplete: {missing}")

    aggregate = [{
        "method": "Mitra-style",
        "tasks": len(benchmark["cases"]),
        "mean_iou": float(np.mean([case["mitra_mean_iou"] for case in benchmark["cases"]])),
        "std_iou_across_tasks": float(np.std([case["mitra_mean_iou"] for case in benchmark["cases"]])),
        "mean_dice": None,
        "mean_saving": 0.0,
        "single_body_rate": None,
        "watertight_rate": None,
        "evidence": str(args.benchmark),
    }]
    for row in sadekar["aggregate"]:
        aggregate.append({
            "method": row["method"],
            "tasks": row["tasks"],
            "mean_iou": row["mean_iou"],
            "std_iou_across_tasks": row["std_iou_across_tasks"],
            "mean_dice": row["mean_dice"],
            "mean_saving": None,
            "single_body_rate": row["single_body_rate"],
            "watertight_rate": row["watertight_rate"],
            "evidence": str(args.sadekar),
        })
    aggregate.append({
        "method": "ShadowForge",
        "tasks": len(sf_rows),
        "mean_iou": float(np.mean([row["mean_iou"] for row in sf_rows])),
        "std_iou_across_tasks": float(np.std([row["mean_iou"] for row in sf_rows])),
        "mean_dice": float(np.mean([row["mean_dice"] for row in sf_rows if row.get("mean_dice") is not None])),
        "mean_saving": float(np.mean([row["saving"] for row in sf_rows if row.get("saving") is not None])),
        "single_body_rate": float(np.mean([row.get("components6") == 1 for row in sf_rows])),
        "watertight_rate": float(np.mean([bool(row.get("watertight")) for row in sf_rows])),
        "evidence": str(args.shadowforge),
    })

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "protocol": benchmark["protocol"],
        "task_composition": "6 Mitra-low and 6 Mitra-high tasks, frozen before evaluating the other methods",
        "aggregate": aggregate,
    }
    (args.output / "same_task_comparison.json").write_text(json.dumps(payload, indent=2))
    with (args.output / "same_task_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
