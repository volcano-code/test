#!/usr/bin/env python3
"""Build a robust, display-only print variant from a delivered voxel model.

This utility deliberately leaves the research-delivery STL untouched.  It
applies one conservative voxel closing/dilation pass to protect fragile
features at a chosen physical size, then exports a separate display plinth.
The plinth is not fused to the sculpture and is excluded from projection QA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, generate_binary_structure, label
from skimage.measure import marching_cubes


def largest_component(volume: np.ndarray) -> np.ndarray:
    labels, count = label(volume, structure=generate_binary_structure(3, 1))
    if count == 0:
        raise RuntimeError("The input volume is empty")
    return labels == int(np.bincount(labels.ravel())[1:].argmax() + 1)


def to_mesh(volume: np.ndarray, target_mm: float) -> trimesh.Trimesh:
    vertices, faces, _, _ = marching_cubes(volume.astype(np.float32), level=0.5)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    extent = float(mesh.extents.max())
    mesh.apply_scale(target_mm / max(extent, 1e-8))
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def make_plinth(mesh: trimesh.Trimesh, margin_mm: float, thickness_mm: float) -> trimesh.Trimesh:
    width, depth = mesh.extents[:2]
    return trimesh.creation.box(
        extents=(float(width + 2.0 * margin_mm), float(depth + 2.0 * margin_mm), thickness_mm),
        transform=trimesh.transformations.translation_matrix((0.0, 0.0, -float(mesh.extents[2]) * 0.5 - thickness_mm * 0.5)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-mm", type=float, default=100.0)
    parser.add_argument("--thicken-voxels", type=int, default=1)
    parser.add_argument("--plinth-margin-mm", type=float, default=10.0)
    parser.add_argument("--plinth-thickness-mm", type=float, default=6.0)
    args = parser.parse_args()
    if args.target_mm <= 0 or args.thicken_voxels < 0:
        raise ValueError("target-mm must be positive and thicken-voxels must be non-negative")

    raw = np.load(args.volume)
    if raw.ndim != 3:
        raise ValueError(f"Expected a 3-D volume, got {raw.shape}")
    raw = raw.astype(bool)
    structure = generate_binary_structure(3, 1)
    protected = binary_closing(raw, structure=structure, iterations=1)
    if args.thicken_voxels:
        protected = binary_dilation(protected, structure=structure, iterations=args.thicken_voxels)
    # Internal micro-cavities turn into detached nested shells after marching
    # cubes.  A display print should be a solid outer body instead.
    protected = binary_fill_holes(protected)
    protected = largest_component(protected)

    mesh = to_mesh(protected, args.target_mm)
    plinth = make_plinth(mesh, args.plinth_margin_mm, args.plinth_thickness_mm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "protected_display_volume.npy", protected.astype(np.uint8))
    mesh.export(args.output_dir / "pcraft_support_safe_display_model.stl")
    plinth.export(args.output_dir / "pcraft_display_plinth.stl")

    voxel_mm = args.target_mm / float(raw.shape[0])
    report = {
        "purpose": "display-only print variant; source delivery geometry remains unchanged",
        "input_volume": str(args.volume.resolve()),
        "target_mm": args.target_mm,
        "voxel_pitch_mm": voxel_mm,
        "closing_iterations": 1,
        "dilation_iterations": args.thicken_voxels,
        "raw_voxels": int(raw.sum()),
        "protected_voxels": int(protected.sum()),
        "volume_increase_fraction": float(protected.sum() / max(int(raw.sum()), 1) - 1.0),
        "model": {
            "path": "pcraft_support_safe_display_model.stl",
            "watertight": bool(mesh.is_watertight),
            "components": int(len(mesh.split(only_watertight=False))),
            "extents_mm": mesh.extents.tolist(),
        },
        "plinth": {
            "path": "pcraft_display_plinth.stl",
            "separate": True,
            "extents_mm": plinth.extents.tolist(),
        },
        "slicer_guidance": {
            "nozzle_mm": 0.4,
            "layer_height_mm": 0.2,
            "wall_loops": 3,
            "infill_percent": 15,
            "brim_mm": 8,
            "support": "tree, build plate only, 55 degree threshold",
            "support_interface": "enabled, 0.20 mm top Z distance; use a dedicated interface material when available",
            "handling": "remove supports from the thickened display variant, then mount it on the separate plinth",
        },
    }
    (args.output_dir / "PRINT_GUIDE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
