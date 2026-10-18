#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean_std(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    return {
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for metrics_path in sorted(args.root.glob("*/*/seed_*/metrics.json")):
        run_dir = metrics_path.parent
        timing_path = run_dir / "timing.json"
        if not timing_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        timing = json.loads(timing_path.read_text())
        view_ious = np.asarray([view["iou"] for view in metrics["views"]], dtype=float)
        rows.append(
            {
                "task": run_dir.parents[1].name,
                "variant": run_dir.parent.name,
                "seed": int(run_dir.name.removeprefix("seed_")),
                "mean_iou": float(view_ious.mean()),
                "worst_view_iou": float(view_ious.min()),
                "view_iou_std": float(view_ious.std(ddof=0)),
                "mean_dice": float(metrics["mean_dice"]),
                **timing,
            }
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    aggregate = {}
    for variant, variant_rows in sorted(grouped.items()):
        aggregate[variant] = {
            "n": len(variant_rows),
            **{
                metric: mean_std([float(row[metric]) for row in variant_rows])
                for metric in (
                    "mean_iou",
                    "worst_view_iou",
                    "view_iou_std",
                    "mean_dice",
                    "elapsed_s",
                )
            },
        }

    by_key = {(row["task"], row["seed"], row["variant"]): row for row in rows}
    paired = {}
    for baseline in (
        "sequential_equal_view_exposure",
        "sequential_equal_optimizer_steps",
    ):
        deltas = defaultdict(list)
        for task, seed, variant in by_key:
            if variant != "joint_300":
                continue
            joint = by_key[(task, seed, "joint_300")]
            other = by_key.get((task, seed, baseline))
            if other is None:
                continue
            for metric in ("mean_iou", "worst_view_iou", "view_iou_std", "elapsed_s"):
                deltas[metric].append(float(joint[metric]) - float(other[metric]))
        paired[baseline] = {
            metric: mean_std(values) for metric, values in sorted(deltas.items())
        }

    summary = {
        "schema": "joint_view_formal_v1",
        "resolution": 128,
        "tasks": sorted({row["task"] for row in rows}),
        "seeds": sorted({row["seed"] for row in rows}),
        "rows": rows,
        "aggregate": aggregate,
        "paired_joint_minus_sequential": paired,
        "decision_rule": (
            "Treat joint-view updating as validated only when it improves paired "
            "worst-view IoU or reduces view-IoU variance under a matched budget "
            "without relying on additional view exposures."
        ),
    }
    output = args.root / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")

    print("variant\tn\tmean_iou\tworst_iou\tview_std\telapsed_s")
    for variant, values in aggregate.items():
        print(
            f"{variant}\t{values['n']}\t"
            f"{values['mean_iou']['mean']:.4f}+/-{values['mean_iou']['std']:.4f}\t"
            f"{values['worst_view_iou']['mean']:.4f}+/-"
            f"{values['worst_view_iou']['std']:.4f}\t"
            f"{values['view_iou_std']['mean']:.4f}+/-"
            f"{values['view_iou_std']['std']:.4f}\t"
            f"{values['elapsed_s']['mean']:.1f}"
        )


if __name__ == "__main__":
    main()
