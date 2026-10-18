#!/usr/bin/env python3
"""Locally reinforce a delivered voxel model without changing its silhouettes.

The repair fills only one-voxel notches and gaps inside the exact two-view
visual hull.  It therefore improves local neck continuity without the global
dilation used by the older display-only variant.  A material budget preserves
the requested minimum saving relative to the same visual hull.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import (
    binary_closing,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
    label,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "generation") not in sys.path:
    sys.path.insert(0, str(ROOT / "generation"))
from voxel_ops import sdf_watertight_mesh


S6 = generate_binary_structure(3, 1)
S18 = generate_binary_structure(3, 2)


def component_count(volume: np.ndarray) -> int:
    return int(label(volume, structure=S6)[1])


def project(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return volume.any(axis=0), volume.any(axis=1)


def overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    denom = int(a.sum() + b.sum())
    return intersection / max(union, 1), 2.0 * intersection / max(denom, 1)


def erosion_survival(volume: np.ndarray) -> float:
    eroded = binary_erosion(volume, structure=S6, iterations=1)
    return float(eroded.sum() / max(int(volume.sum()), 1))


def support_metrics(print_volume: np.ndarray, print_mm: float) -> dict:
    v = np.asarray(print_volume, dtype=bool)
    z0, z1 = np.where(v)[2].min(), np.where(v)[2].max()
    s2 = np.ones((3, 3), dtype=bool)
    from scipy.ndimage import binary_dilation

    unsupported_total = 0
    islands = 0
    for z in range(int(z0) + 1, int(z1) + 1):
        unsupported = v[:, :, z] & ~binary_dilation(v[:, :, z - 1], structure=s2)
        unsupported_total += int(unsupported.sum())
        labs, count = label(unsupported, structure=s2)
        for index in range(1, count + 1):
            points = np.argwhere(labs == index)
            if points.size and (
                len(points) >= 16
                or (np.ptp(points, axis=0).max() + 1) * print_mm / v.shape[0] >= 4.0
            ):
                islands += 1
    return {
        "unsupported_ratio": float(unsupported_total / max(int(v.sum()), 1)),
        "significant_islands": int(islands),
        "bed_area_mm2": float(v[:, :, int(z0)].sum() * (print_mm / v.shape[0]) ** 2),
    }


def to_mesh(volume: np.ndarray, target_mm: float) -> trimesh.Trimesh:
    mesh = sdf_watertight_mesh(np.asarray(volume, dtype=bool), smooth_sigma=0.6)
    mesh.apply_scale(target_mm / max(float(mesh.extents.max()), 1e-8))
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation([-center[0], -center[1], -mesh.bounds[0, 2]])
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-mm", type=float, default=100.0)
    parser.add_argument("--min-saving", type=float, default=0.20)
    args = parser.parse_args()

    raw = np.load(args.volume).astype(bool)
    if raw.ndim != 3 or not raw.any():
        raise ValueError(f"Expected a non-empty 3-D volume, got {raw.shape}")
    projection_x, projection_y = project(raw)
    hull = projection_x[None, :, :] & projection_y[:, None, :]
    max_voxels = int(np.floor((1.0 - args.min_saving) * int(hull.sum())))

    # S18 closing fills diagonal one-voxel cracks as well as axis-aligned
    # notches.  Intersecting with the hull makes every addition projection-safe.
    repaired = binary_closing(raw, structure=S18, iterations=1) & hull
    repaired_labels, repaired_count = label(repaired, structure=S6)
    if repaired_count > 1:
        sizes = np.bincount(repaired_labels.ravel())
        sizes[0] = 0
        repaired = repaired_labels == int(np.argmax(sizes))
    additions = repaired & ~raw
    if int(repaired.sum()) > max_voxels:
        # Prefer additions closest to the original and to locally thin regions.
        distance_to_raw = distance_transform_edt(~raw)
        local_radius = distance_transform_edt(raw)
        radius_hint = np.zeros(raw.shape, dtype=np.float32)
        radius_hint[raw] = np.minimum(local_radius[raw], 3.0)
        from scipy.ndimage import maximum_filter

        radius_hint = maximum_filter(radius_hint, size=3)
        points = np.argwhere(additions)
        scores = distance_to_raw[additions] - 0.15 * radius_hint[additions]
        keep_count = max(0, max_voxels - int(raw.sum()))
        order = np.argsort(scores, kind="stable")[:keep_count]
        limited = np.zeros_like(raw)
        limited[tuple(points[order].T)] = True
        repaired = raw | limited

    out_x, out_y = project(repaired)
    iou_x, dice_x = overlap(out_x, projection_x)
    iou_y, dice_y = overlap(out_y, projection_y)
    mesh = to_mesh(repaired, args.target_mm)

    # display_x_up maps display [X,Y,Z] to print [Y,Z,X].
    print_volume = np.transpose(repaired, (1, 2, 0))
    raw_print = np.transpose(raw, (1, 2, 0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "locally_reinforced_volume256.npy", repaired.astype(np.uint8))
    mesh.export(args.output_dir / "pcraft_locally_reinforced_display_model.stl")
    print_mesh = to_mesh(print_volume, args.target_mm)
    print_mesh.export(args.output_dir / "pcraft_locally_reinforced_print_x_up.stl")

    voxel_pitch = args.target_mm / max(raw.shape)
    raw_radius = distance_transform_edt(raw)
    repaired_radius = distance_transform_edt(repaired)
    report = {
        "schema": "pcraft_projection_guarded_local_reinforcement_v1",
        "source_volume": str(args.volume.resolve()),
        "target_mm": float(args.target_mm),
        "voxel_pitch_mm": float(voxel_pitch),
        "operation": "one-pass S18 local closing clipped to the exact two-view visual hull",
        "raw_voxels": int(raw.sum()),
        "reinforced_voxels": int(repaired.sum()),
        "added_voxels": int((repaired & ~raw).sum()),
        "visual_hull_voxels": int(hull.sum()),
        "saving_vs_hull": float(1.0 - repaired.sum() / max(int(hull.sum()), 1)),
        "projection": {
            "view_ious": [float(iou_x), float(iou_y)],
            "view_dices": [float(dice_x), float(dice_y)],
        },
        "fabrication": {
            "components": component_count(repaired),
            "erosion_survival_before": erosion_survival(raw),
            "erosion_survival_after": erosion_survival(repaired),
            "one_voxel_radius_fraction_before": float((raw & (raw_radius <= 1.01)).sum() / raw.sum()),
            "one_voxel_radius_fraction_after": float(
                (repaired & (repaired_radius <= 1.01)).sum() / repaired.sum()
            ),
            "print_proxy_before": support_metrics(raw_print, args.target_mm),
            "print_proxy_after": support_metrics(print_volume, args.target_mm),
            "mesh_watertight": bool(mesh.is_watertight),
            "mesh_components": int(len(mesh.split(only_watertight=False))),
        },
        "output_stl": "pcraft_locally_reinforced_display_model.stl",
        "print_stl": "pcraft_locally_reinforced_print_x_up.stl",
        "notes": [
            "This is a display-print variant; the research delivery remains unchanged.",
            "All added voxels lie inside the exact projection-feasible hull.",
            "The minimum-saving budget is enforced before export.",
        ],
    }
    (args.output_dir / "REINFORCEMENT_QA.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
