#!/usr/bin/env python3
"""Summarize the controlled two-stage Table 2 pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def projection_rows(soft_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(soft_root.glob("*/*/seed_*/metrics.json")):
        data = json.loads(path.read_text())
        task, variant, seed = path.parts[-4:-1]
        fp_values = []
        for pred_path in sorted(path.parent.glob("pred_view_*.png")):
            index = pred_path.stem.rsplit("_", 1)[1]
            gt_path = path.parent / f"gt_view_{index}.png"
            pred = np.asarray(Image.open(pred_path).convert("L")) >= 128
            target = np.asarray(Image.open(gt_path).convert("L")) >= 128
            fp_values.append(float(np.logical_and(pred, ~target).mean()))
        rows.append({
            "task": task,
            "variant": variant,
            "seed": seed,
            "mean_iou": float(data["mean_iou"]),
            "worst_view_iou": float(min(view["iou"] for view in data["views"])),
            "mean_dice": float(data["mean_dice"]),
            "fp": float(np.mean(fp_values)),
        })
    return rows


def geometry_rows(geometry_root: Path, tolerance: float = 0.01) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(geometry_root.glob("*/*/MANIFEST.json")):
        data = json.loads(path.read_text())
        task, variant = path.parts[-3:-1]
        chosen = data.get("selected")
        if chosen is None:
            candidates = data.get("candidates", [])
            if not candidates:
                continue
            best_iou = max(float(row["mean_iou"]) for row in candidates)
            band = [row for row in candidates if float(row["mean_iou"]) >= best_iou - tolerance]
            chosen = max(band, key=lambda row: (float(row.get("saving", 0.0)), float(row["mean_iou"])))
        rows.append({
            "task": task,
            "variant": variant,
            "mean_iou": float(chosen["mean_iou"]),
            "saving": float(chosen["saving"]),
            "single": float(int(chosen.get("components", 0) == 1)),
            "watertight": float(bool(chosen.get("watertight", False))),
            "candidate": chosen.get("candidate"),
        })
    return rows


def aggregate(rows: list[dict], metrics: tuple[str, ...]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["variant"]].append(row)
    summary = []
    for variant, items in sorted(groups.items()):
        record = {"variant": variant, "n": len(items)}
        for metric in metrics:
            values = np.asarray([item[metric] for item in items], dtype=np.float64)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std())
        summary.append(record)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--soft-only", action="store_true")
    parser.add_argument("--tasks", nargs="*")
    args = parser.parse_args()
    soft_root = args.root / "soft" if (args.root / "soft").is_dir() else args.root
    soft_rows = projection_rows(soft_root)
    geometry = [] if args.soft_only else geometry_rows(args.root / "geometry")
    metric_paths = sorted(soft_root.glob("*/*/seed_*/metrics.json"))
    first_metrics = json.loads(metric_paths[0].read_text()) if metric_paths else {}
    parameter_resolution = int(first_metrics.get("parameter_resolution", 256))
    payload = {
        "protocol": {
            "tasks": args.tasks or sorted({row["task"] for row in soft_rows}),
            "seeds": sorted({int(row["seed"].split("_")[-1]) for row in soft_rows}),
            "soft_resolution": parameter_resolution,
            "control_resolution": 0,
            "parameter_resolution": parameter_resolution,
            "iterations": 2000,
        },
        "table2a_rows": soft_rows,
        "table2a_summary": aggregate(
            soft_rows, ("mean_iou", "worst_view_iou", "mean_dice", "fp")
        ),
        "table2b_rows": geometry,
        "table2b_summary": aggregate(geometry, ("mean_iou", "saving", "single", "watertight")) if geometry else [],
    }
    output = args.root / "pilot_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
