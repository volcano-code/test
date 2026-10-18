#!/usr/bin/env python3
"""Collect generation ablation and resolution-sensitivity results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    paths = {
        32: RESULTS / "generation_pipeline_ablation_r32.json",
        48: RESULTS / "generation_pipeline_ablation_r48.json",
        64: RESULTS / "generation_pipeline_ablation.json",
        96: RESULTS / "generation_pipeline_ablation_r96.json",
    }
    rows = []
    for resolution, path in paths.items():
        payload = json.loads(path.read_text())
        full = payload["aggregate"]["full_pipeline"]
        carve = payload["aggregate"]["carve_no_robust_bridge"]
        rows.append({
            "resolution": resolution,
            "n_cases": full["n"],
            "full_mean_iou": full["mean_iou"],
            "full_min_iou": full["min_iou"],
            "full_material_saved_vs_hull": full["mean_material_saved_vs_hull"],
            "full_single_body_6_rate": full["single_body_6_rate"],
            "full_watertight_rate": full["watertight_rate"],
            "without_bridge_single_body_6_rate": carve["single_body_6_rate"],
            "without_watertight_watertight_rate": payload["aggregate"]["carve_plus_bridge"]["watertight_rate"],
        })
    summary = {
        "protocol": "generation_resolution_sensitivity_v1",
        "rows": rows,
        "across_resolution": {
            "mean_full_iou": sum(row["full_mean_iou"] for row in rows) / len(rows),
            "min_full_iou": min(row["full_min_iou"] for row in rows),
            "mean_material_saved_vs_hull": sum(row["full_material_saved_vs_hull"] for row in rows) / len(rows),
            "single_body_6_rate_all": min(row["full_single_body_6_rate"] for row in rows),
            "watertight_rate_all": min(row["full_watertight_rate"] for row in rows),
        },
    }
    output = RESULTS / "generation_resolution_sensitivity.json"
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
