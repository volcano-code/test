#!/usr/bin/env python3
"""Keep the largest closed mesh component and emit topology/material QA.

This is intentionally non-destructive: the input mesh is never overwritten.
For meshes documented as one exterior shell plus disconnected interior pieces,
the largest closed component is the exterior shell.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh) or loaded.is_empty:
        raise RuntimeError(f"Could not load a non-empty mesh from {path}")
    return loaded


def weld_stl_triangle_soup(mesh: trimesh.Trimesh, digits_vertex: int = 6) -> trimesh.Trimesh:
    """Recover shared topology from an STL's duplicated per-face vertices.

    Binary and ASCII STL frequently duplicate every triangle's three vertices.
    Component analysis on that raw representation treats every face as an
    isolated component, even when the geometry is a connected surface.
    """
    welded = mesh.copy()
    welded.merge_vertices(digits_vertex=digits_vertex)
    # Remove exact duplicate faces introduced by vertex welding.  Do not fill
    # holes or alter geometry: a non-watertight result must remain visible to QA.
    welded.update_faces(welded.unique_faces())
    welded.remove_unreferenced_vertices()
    return welded


def edge_audit(mesh: trimesh.Trimesh) -> dict:
    incidence = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    return {
        "boundary_edges": int((incidence == 1).sum()),
        "nonmanifold_edges": int((incidence > 2).sum()),
        "watertight": bool(mesh.is_watertight),
        "components": int(len(mesh.split(only_watertight=False))),
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "volume_mm3": float(abs(mesh.volume)) if mesh.is_watertight else None,
    }


def component_score(mesh: trimesh.Trimesh) -> tuple[float, float, int]:
    # The exterior shell is expected to dominate both enclosed volume and area.
    volume = abs(float(mesh.volume)) if mesh.is_watertight else 0.0
    return (volume, float(mesh.area), int(len(mesh.faces)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--qa", type=Path, required=True)
    ap.add_argument("--weld-digits", type=int, default=6,
                    help="Decimal precision used to merge duplicate STL vertices.")
    args = ap.parse_args()

    raw_source = load_mesh(args.input)
    source = weld_stl_triangle_soup(raw_source, digits_vertex=args.weld_digits)
    parts = list(source.split(only_watertight=False))
    if not parts:
        raise RuntimeError("No mesh components found.")
    chosen_index, chosen = max(enumerate(parts), key=lambda item: component_score(item[1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chosen.export(args.output)

    before = edge_audit(source)
    after = edge_audit(chosen)
    source_volume = before["volume_mm3"]
    output_volume = after["volume_mm3"]
    material_removed = None
    if source_volume and output_volume is not None:
        material_removed = float(1.0 - output_volume / source_volume)
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "weld_digits": int(args.weld_digits),
        "raw_stl": edge_audit(raw_source),
        "input_components": len(parts),
        "kept_component_index": chosen_index,
        "removed_components": len(parts) - 1,
        "before": before,
        "after": after,
        "mesh_volume_reduction": material_removed,
        "accept": bool(after["watertight"] and after["components"] == 1
                       and after["boundary_edges"] == 0 and after["nonmanifold_edges"] == 0),
    }
    args.qa.parent.mkdir(parents=True, exist_ok=True)
    args.qa.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
