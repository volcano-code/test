#!/usr/bin/env python3
"""Package a larger, support-friendly display kit from a selected print STL.

The sculpture stays a standalone watertight part.  The locating plinth is
exported separately so it cannot change the validated silhouette geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
import sys
sys.path.insert(0, str(ROOT))

from shadowforge_pipeline.display_delivery import create_display_delivery


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--target-mm", type=float, default=100.0)
    ap.add_argument("--base-thickness-mm", type=float, default=5.0)
    args = ap.parse_args()

    mesh = trimesh.load(args.model, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Could not load mesh: {args.model}")
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        raise ValueError("The source display model must be watertight")
    scale = float(args.target_mm / max(float(mesh.extents.max()), 1e-8))
    mesh.apply_scale(scale)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.output_dir / "pcraft_display_model_100mm.stl"
    mesh.export(model_out)
    fixture = create_display_delivery(
        model_out, args.output_dir / "separate_plinth",
        margin_mm=10.0, base_thickness_mm=args.base_thickness_mm,
        clearance_mm=0.9, retainer_height_mm=1.8,
    )
    record = {
        "schema": "pcraft_printsafe_display_kit_v1",
        "source": str(args.model.resolve()),
        "model": str(model_out.resolve()),
        "target_mm": args.target_mm,
        "scale_factor": scale,
        "watertight": bool(mesh.is_watertight),
        "components": int(len(mesh.split(only_watertight=False))),
        "extents_mm": [round(float(x), 2) for x in mesh.extents],
        "print_guidance": {
            "material": "PLA or PETG",
            "layer_height_mm": 0.20,
            "wall_loops": 3,
            "infill_percent": 15,
            "brim_mm": 8,
            "support": "tree supports, build-plate only, 55 degree threshold",
            "support_interface": "enabled; 0.20 mm top Z distance",
            "note": "The separate plinth is a display fixture, not part of the silhouette-validated sculpture.",
        },
        "fixture": fixture,
    }
    (args.output_dir / "PRINT_GUIDE.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
