#!/usr/bin/env python3
"""Create a reproducible aggregate manifest from the four Figure-5 arms."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fig5_geometry_with_cubify_20260719"
TASKS = ("sword_fox", "future_mech", "parrot_snail", "owl_turtle")
METHODS = ("direct_cubify", "direct_marching_cubes", "sdf_only", "shadowforge_conversion")


def main() -> None:
    records = []
    for task in TASKS:
        rows = [json.loads((OUT / task / method / "metrics.json").read_text()) for method in METHODS]
        records.append({"task": task, "rows": rows})
    aggregate = []
    for method in METHODS:
        rows = [next(row for row in record["rows"] if row["method"] == method) for record in records]
        aggregate.append({
            "method": method,
            "n_tasks": len(rows),
            "mean_iou": float(np.mean([row["mean_iou"] for row in rows])),
            "std_iou": float(np.std([row["mean_iou"] for row in rows])),
            "mean_dice": float(np.mean([row["mean_dice"] for row in rows])),
            "std_dice": float(np.std([row["mean_dice"] for row in rows])),
            "single_body_rate": float(np.mean([row["single_body"] for row in rows])),
            "watertight_rate": float(np.mean([row["watertight"] for row in rows])),
            "mean_mesh_parts": float(np.mean([row["mesh_parts"] for row in rows])),
        })
    payload = {
        "schema": "fig5_geometry_conversion_with_cubify_v1",
        "controls": {
            "same_probability_field": True,
            "same_threshold_grid": [0.15, 0.25, 0.35, 0.45, 0.55],
            "same_targets": True,
            "same_projection_evaluator": True,
            "same_mesh_qa": True,
            "sdf_only_sigma": 1.0,
            "cubify": "pytorch3d.ops.cubify(thresh=0.5), no cleanup or smoothing",
        },
        "records": records,
        "aggregate": aggregate,
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
