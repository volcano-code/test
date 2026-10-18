#!/usr/bin/env python3
"""Summarize seed robustness and threshold stability for Stage-A soft ablations."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--canonical-threshold", type=float, default=0.05)
    parser.add_argument("--thresholds", default="0.03,0.04,0.05,0.06,0.07")
    args = parser.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]
    if args.canonical_threshold not in thresholds:
        thresholds.append(args.canonical_threshold)
        thresholds.sort()

    rows = []
    for path in sorted(args.root.glob("*/seed_*/occupancy_probability_core.npy")):
        metrics_path = path.with_name("metrics.json")
        if not metrics_path.exists() or path.parts[-3].startswith("_"):
            continue
        probability = np.load(path, mmap_mode="r")
        canonical = np.asarray(probability >= args.canonical_threshold)
        fractions, similarities = {}, {}
        for threshold in thresholds:
            mask = np.asarray(probability >= threshold)
            key = f"{threshold:.3f}"
            fractions[key] = float(mask.mean())
            similarities[key] = jaccard(mask, canonical)
        noncanonical = [similarities[f"{t:.3f}"] for t in thresholds
                        if t != args.canonical_threshold]
        values = np.asarray(list(fractions.values()), dtype=float)
        metrics = json.loads(metrics_path.read_text())
        rows.append({
            "variant": path.parts[-3],
            "seed": int(path.parts[-2].split("_")[-1]),
            "raw_soft_iou": float(metrics["raw_soft_iou"]),
            "canonical_threshold": args.canonical_threshold,
            "thresholds": thresholds,
            "occupied_fraction_by_threshold": fractions,
            "jaccard_to_canonical_by_threshold": similarities,
            "threshold_jaccard_mean": float(np.mean(noncanonical)),
            "threshold_jaccard_min": float(np.min(noncanonical)),
            "occupied_fraction_cv": float(values.std() / max(values.mean(), 1e-12)),
            "occupied_fraction_relative_range": float(
                (values.max() - values.min()) / max(fractions[f"{args.canonical_threshold:.3f}"], 1e-12)
            ),
        })

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    aggregate = []
    for variant, group in sorted(grouped.items()):
        def mean_std(key: str) -> tuple[float, float]:
            values = np.asarray([r[key] for r in group], dtype=float)
            return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0
        iou_mean, iou_std = mean_std("raw_soft_iou")
        jac_mean, jac_std = mean_std("threshold_jaccard_mean")
        jac_min_mean, jac_min_std = mean_std("threshold_jaccard_min")
        cv_mean, cv_std = mean_std("occupied_fraction_cv")
        aggregate.append({
            "variant": variant, "n_seeds": len(group),
            "seeds": sorted(r["seed"] for r in group),
            "raw_soft_iou_mean": iou_mean, "raw_soft_iou_std": iou_std,
            "threshold_jaccard_mean": jac_mean, "threshold_jaccard_std": jac_std,
            "threshold_jaccard_min_mean": jac_min_mean,
            "threshold_jaccard_min_std": jac_min_std,
            "occupied_fraction_cv_mean": cv_mean, "occupied_fraction_cv_std": cv_std,
        })

    payload = {
        "schema": "soft_seed_threshold_stability_v1",
        "definition": {
            "threshold_jaccard": "3D Jaccard between occupancy thresholded at t and the canonical t=0.05 occupancy",
            "occupied_fraction_cv": "coefficient of variation of occupied fraction over the threshold grid",
            "interpretation": "higher threshold Jaccard and lower occupied-fraction CV indicate greater threshold stability",
        },
        "rows": rows,
        "aggregate_by_variant": aggregate,
    }
    (args.root / "threshold_stability.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "| Variant | Seeds | Raw IoU mean±std | Threshold Jaccard mean±std | Worst-threshold Jaccard | Occupancy CV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['variant']} | {row['n_seeds']} | "
            f"{row['raw_soft_iou_mean']:.4f}±{row['raw_soft_iou_std']:.4f} | "
            f"{row['threshold_jaccard_mean']:.4f}±{row['threshold_jaccard_std']:.4f} | "
            f"{row['threshold_jaccard_min_mean']:.4f}±{row['threshold_jaccard_min_std']:.4f} | "
            f"{row['occupied_fraction_cv_mean']:.4f}±{row['occupied_fraction_cv_std']:.4f} |"
        )
    (args.root / "table_seed_threshold_stability.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"root": str(args.root), "runs": len(rows), "variants": len(aggregate)}, indent=2))


if __name__ == "__main__":
    main()
