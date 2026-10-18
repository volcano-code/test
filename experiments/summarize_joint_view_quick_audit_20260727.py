#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    rows = []
    task_root = args.root / args.task
    for run_dir in sorted(task_root.glob(f"*/seed_{args.seed}")):
        metrics_path = run_dir / "metrics.json"
        timing_path = run_dir / "timing.json"
        if not metrics_path.exists() or not timing_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        timing = json.loads(timing_path.read_text())
        view_ious = np.asarray([view["iou"] for view in metrics["views"]], dtype=float)
        rows.append(
            {
                "variant": run_dir.parent.name,
                "mean_iou": float(view_ious.mean()),
                "worst_view_iou": float(view_ious.min()),
                "view_iou_std": float(view_ious.std(ddof=0)),
                "mean_dice": float(metrics["mean_dice"]),
                **timing,
            }
        )

    summary = {
        "task": args.task,
        "seed": args.seed,
        "resolution": 128,
        "interpretation": {
            "same_view_exposure": [
                "joint_300",
                "sequential_equal_view_exposure",
            ],
            "same_optimizer_steps": [
                "joint_300",
                "sequential_equal_optimizer_steps",
            ],
            "decision_rule": (
                "Retain a joint-view benefit claim only if joint improves worst-view "
                "IoU or view variance under at least one matched budget without a "
                "material wall-clock regression; otherwise report it as an "
                "implementation choice rather than a validated contribution."
            ),
        },
        "rows": rows,
    }
    output = args.root / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")

    print("variant\tmean_iou\tworst_iou\tview_std\tsteps\tview_exposures\telapsed_s")
    for row in rows:
        print(
            f"{row['variant']}\t{row['mean_iou']:.4f}\t"
            f"{row['worst_view_iou']:.4f}\t{row['view_iou_std']:.4f}\t"
            f"{row['optimizer_steps']}\t{row['training_view_exposures']}\t"
            f"{row['elapsed_s']}"
        )


if __name__ == "__main__":
    main()
