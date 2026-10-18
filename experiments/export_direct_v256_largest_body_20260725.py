#!/usr/bin/env python3
"""Export the largest 6-connected body from a frozen V256 soft proposal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.ndimage import generate_binary_structure, label

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadowforge_pipeline.unified_run import (
    _canonical_xyz_from_trainer_zyx,
    _generic_render,
    _generic_targets,
    export_stl_at_print_scale,
    mesh_qa,
)


S6 = generate_binary_structure(3, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-mm", type=float, default=100.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((args.soft_dir / "metrics.json").read_text())
    proposal = _canonical_xyz_from_trainer_zyx(
        np.load(args.soft_dir / "occupancy_soft_proposal.npy").astype(bool)
    )

    labels, component_count = label(proposal, structure=S6)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_label = int(sizes.argmax())
    largest = labels == largest_label

    render_args = SimpleNamespace(
        device=args.device,
        silhouettes=metrics["args"]["shadow_files"],
        mr=int(metrics["args"].get("mirror_mode", 0)),
        tsf=int(metrics["args"].get("tsf_mode", 0)),
        camera_mode=metrics["args"].get("camera_mode", "ortho"),
    )
    soft_info = {"metrics": metrics, "artifact_dir": str(args.soft_dir)}
    targets = _generic_targets(args.soft_dir, 256)
    renders = _generic_render(largest, render_args, soft_info)

    per_view = []
    projection_dir = args.output / "projections"
    projection_dir.mkdir(exist_ok=True)
    for index, (target, rendered) in enumerate(zip(targets, renders)):
        intersection = int(np.logical_and(target, rendered).sum())
        union = int(np.logical_or(target, rendered).sum())
        denominator = int(target.sum() + rendered.sum())
        per_view.append(
            {
                "index": index,
                "iou": float(intersection / max(union, 1)),
                "dice": float(2.0 * intersection / max(denominator, 1)),
            }
        )
        Image.fromarray(target.astype(np.uint8) * 255, mode="L").save(
            projection_dir / f"target_view_{index}.png"
        )
        Image.fromarray(rendered.astype(np.uint8) * 255, mode="L").save(
            projection_dir / f"projection_view_{index}.png"
        )

    np.save(args.output / "volume.npy", largest.astype(np.uint8))
    stl_path = args.output / "direct_v256_largest_body.stl"
    mesh = export_stl_at_print_scale(largest, stl_path, args.print_mm)
    watertight, mesh_parts, mesh_outer_parts = mesh_qa(largest)
    report = {
        "source": str(args.soft_dir),
        "representation": "direct V256, largest 6-connected body, production SDF export",
        "components_before": int(component_count),
        "components_after": 1,
        "source_voxels": int(proposal.sum()),
        "kept_voxels": int(largest.sum()),
        "kept_fraction": float(largest.sum() / max(int(proposal.sum()), 1)),
        "removed_fraction": float(1.0 - largest.sum() / max(int(proposal.sum()), 1)),
        "mean_iou": float(np.mean([row["iou"] for row in per_view])),
        "min_iou": float(min(row["iou"] for row in per_view)),
        "mean_dice": float(np.mean([row["dice"] for row in per_view])),
        "per_view": per_view,
        "mesh": {
            "stl": str(stl_path),
            "watertight": bool(watertight),
            "parts": int(mesh_parts),
            "outer_parts": int(mesh_outer_parts),
            "internal_parts": int(max(mesh_parts - mesh_outer_parts, 0)),
            "faces": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
            "volume_mm3": float(abs(mesh.volume)) if mesh.is_volume else None,
        },
    }
    (args.output / "QA.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
