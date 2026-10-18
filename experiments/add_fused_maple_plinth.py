#!/usr/bin/env python3
"""Attach a display-only plinth below the silhouette-space vertical axis.

The input must be the fixed-display STL, not a print-oriented STL.  The
pipeline exports fixed-display meshes with image up mapped to physical +Z, so
placing the plinth below Z-min locates it beneath the maple silhouette before
any print-orientation transform.  The fused asset is for physical display and
is excluded from projection and material metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-width-mm", type=float, default=34.0)
    parser.add_argument("--base-depth-mm", type=float, default=34.0)
    parser.add_argument("--base-thickness-mm", type=float, default=4.0)
    parser.add_argument("--overlap-mm", type=float, default=0.8)
    parser.add_argument("--engine", default="blender")
    args = parser.parse_args()

    if min(args.base_width_mm, args.base_depth_mm, args.base_thickness_mm, args.overlap_mm) <= 0:
        raise ValueError("All plinth dimensions and overlap must be positive")

    model = trimesh.load(args.model, force="mesh")
    if not isinstance(model, trimesh.Trimesh) or model.is_empty:
        raise ValueError(f"Could not load a non-empty mesh from {args.model}")
    model.remove_unreferenced_vertices()
    # SDF exports can be watertight with globally inward-facing triangles.
    # Boolean engines require a positively oriented volume.
    trimesh.repair.fix_winding(model)
    trimesh.repair.fix_normals(model)
    if model.is_watertight and model.volume < 0:
        model.invert()
    if not model.is_volume:
        raise ValueError("Input is watertight but could not be oriented as a solid volume")
    bounds = np.asarray(model.bounds, dtype=float)
    center_xy = bounds[:, :2].mean(axis=0)
    z_min = float(bounds[0, 2])
    top = z_min + float(args.overlap_mm)
    bottom = z_min - float(args.base_thickness_mm)
    base = trimesh.creation.box(
        extents=(args.base_width_mm, args.base_depth_mm, top - bottom),
        transform=trimesh.transformations.translation_matrix(
            (float(center_xy[0]), float(center_xy[1]), (top + bottom) / 2.0)
        ),
    )

    fused = trimesh.boolean.union([model, base], engine=args.engine)
    if isinstance(fused, list):
        fused = trimesh.util.concatenate(fused)
    if not isinstance(fused, trimesh.Trimesh) or fused.is_empty:
        raise RuntimeError("Boolean union did not produce a mesh")
    fused.remove_unreferenced_vertices()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fused.export(args.output)
    record = {
        "schema": "pcraft_fused_maple_plinth_v1",
        "purpose": "display-only fused base; excluded from projection, saving, and structural metrics",
        "source_model": str(args.model.resolve()),
        "output_model": str(args.output.resolve()),
        "placement": "below fixed-display Z-min (silhouette bottom, before print orientation)",
        "base_extents_mm": [args.base_width_mm, args.base_depth_mm, args.base_thickness_mm + args.overlap_mm],
        "overlap_mm": args.overlap_mm,
        "watertight": bool(fused.is_watertight),
        "components": int(len(fused.split(only_watertight=False))),
        "faces": int(len(fused.faces)),
        "bounds_mm": np.asarray(fused.bounds).tolist(),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
