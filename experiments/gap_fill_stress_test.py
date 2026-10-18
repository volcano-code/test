#!/usr/bin/env python3
"""Stress-test projection-gap repair under one-view boundary perturbations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.generation_pipeline_ablation import DEFAULT_CASES, load_mask, projection_metrics
from generation.voxel_ops import _build_tfs_canonical_planes, build_tfs_visual_hull, repair_tfs_projection_gaps


def main() -> None:
    resolution = 64
    rows = []
    for case, files in DEFAULT_CASES.items():
        masks = [load_mask(ROOT / "generation" / name, resolution) for name in files]
        perturbed = list(masks)
        perturbed[0] = masks[0].copy()
        # An isolated foreground patch models a thresholding/segmentation conflict
        # that has no support in the other two canonical views.
        perturbed[0][2:7, 2:7] = True

        hull, calibration = build_tfs_visual_hull(perturbed, resolution, apply_cleanup=False)
        repaired, info = repair_tfs_projection_gaps(perturbed, resolution, calibration=calibration)
        planes = _build_tfs_canonical_planes(perturbed, resolution, calibration)
        targets = [planes["x"], planes["y"], planes["z"]]
        before = projection_metrics(hull, targets)
        after = projection_metrics(repaired, targets)
        target_pixels = sum(int(target.sum()) for target in targets)
        before_fn = sum(int(np.logical_and(target, ~projection).sum()) for target, projection in zip(
            targets, [hull.max(0), hull.max(1), hull.max(2)]
        ))
        after_fn = sum(int(np.logical_and(target, ~projection).sum()) for target, projection in zip(
            targets, [repaired.max(0), repaired.max(1), repaired.max(2)]
        ))
        rows.append({
            "case": case,
            "before_mean_iou": before["mean_iou"],
            "after_mean_iou": after["mean_iou"],
            "before_fn_pixels": before_fn,
            "after_fn_pixels": after_fn,
            "target_pixels": target_pixels,
            "repair_voxels_added": info["repair_voxels_added"],
        })
    payload = {
        "protocol": "one_view_boundary_dilation_gap_stress_v1",
        "resolution": resolution,
        "n_cases": len(rows),
        "rows": rows,
        "aggregate": {
            "before_mean_iou": float(np.mean([r["before_mean_iou"] for r in rows])),
            "after_mean_iou": float(np.mean([r["after_mean_iou"] for r in rows])),
            "before_fn_pixels": int(sum(r["before_fn_pixels"] for r in rows)),
            "after_fn_pixels": int(sum(r["after_fn_pixels"] for r in rows)),
            "fn_recovery_rate": float(1.0 - sum(r["after_fn_pixels"] for r in rows) / max(sum(r["before_fn_pixels"] for r in rows), 1)),
            "mean_repair_voxels_added": float(np.mean([r["repair_voxels_added"] for r in rows])),
        },
    }
    output = ROOT / "results" / "gap_fill_stress_test.json"
    output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["aggregate"], indent=2))


if __name__ == "__main__":
    main()
