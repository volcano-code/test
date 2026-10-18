#!/usr/bin/env python3
"""Smooth surface-mesh FEM pilot for final STL artifacts.

This is intentionally separate from the voxel-FEM pipeline. It takes a printable
STL, repairs/simplifies the surface, tetrahedralizes it, solves linear elastic
self-weight FEM, and writes convergence-facing JSON summaries.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import pymeshfix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural.mesh_fem.calibration import sf_metrics
from structural.mesh_fem.mesh_fem_solver import (
    save_mesh_fem_result,
    solve_tet_mesh_fem,
    tetrahedralize_surface_mesh,
)


DEFAULT_STLS = [
    ROOT / "PRINT_DELIVERY_COMPLETE/heroes_spiderman_superman_batman/heroes_spiderman_superman_batman_complete.stl",
    ROOT / "PRINT_DELIVERY_COMPLETE/g1_dragon_peacock_trex/g1_dragon_peacock_trex_complete.stl",
]


def prepare_surface(src: Path, out_path: Path, reduction: float, repair_order: str = "post") -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = pv.read(str(src)).triangulate().clean()
    before = {"points": int(mesh.n_points), "cells": int(mesh.n_cells)}

    if repair_order in {"pre", "both"}:
        fix = pymeshfix.MeshFix(mesh)
        fix.repair(verbose=False, joincomp=True, remove_smallest_components=False)
        mesh = fix.mesh.triangulate().clean()

    if reduction > 0:
        mesh = mesh.decimate_pro(
            float(reduction),
            preserve_topology=True,
            splitting=False,
            boundary_vertex_deletion=False,
        ).triangulate().clean()

    if repair_order in {"post", "both"}:
        fix = pymeshfix.MeshFix(mesh)
        fix.repair(verbose=False, joincomp=True, remove_smallest_components=False)
        repaired = fix.mesh.triangulate().clean()
    else:
        repaired = mesh.triangulate().clean()
    repaired.save(str(out_path))

    return {
        "source": str(src),
        "prepared_surface": str(out_path),
        "reduction": float(reduction),
        "repair_order": repair_order,
        "before": before,
        "after": {"points": int(repaired.n_points), "cells": int(repaired.n_cells)},
        "surface_volume": float(repaired.volume),
        "surface_area": float(repaired.area),
    }


def run_one_level(
    stl: Path,
    out_dir: Path,
    reduction: float,
    quality: bool,
    nobisect: bool,
    geometry_scale: float,
    support_frac: float | None,
    support_thickness_override: float | None,
    repair_order: str,
    yield_stress: float,
) -> dict:
    tag = f"red{reduction:.3f}_q{int(quality)}_nb{int(nobisect)}".replace(".", "p")
    level_dir = out_dir / tag
    surface_path = level_dir / "prepared_surface.stl"
    t0 = time.time()
    prep = prepare_surface(stl, surface_path, reduction, repair_order=repair_order)
    prep_sec = time.time() - t0

    t0 = time.time()
    tet = tetrahedralize_surface_mesh(
        surface_path,
        target_tets=30000,
        geometry_scale=geometry_scale,
        quality=quality,
        nobisect=nobisect,
        quiet=True,
    )
    tet_sec = time.time() - t0

    t0 = time.time()
    support_thickness = support_thickness_override
    if support_thickness is None and support_frac is not None:
        height = float(tet.nodes[:, 1].max() - tet.nodes[:, 1].min())
        support_thickness = max(float(support_frac) * height, 1e-12)
    result = solve_tet_mesh_fem(tet, yield_stress=yield_stress, support_thickness=support_thickness)
    solve_sec = time.time() - t0
    save_mesh_fem_result(level_dir, result, surface_path)

    metrics = sf_metrics(result["von_mises"], yield_stress=yield_stress)
    summary = {
        "tag": tag,
        "input_stl": str(stl),
        "preprocess": prep,
        "tet": {
            "n_nodes": int(result["n_nodes"]),
            "n_elems": int(result["n_elems"]),
            "n_valid_elems": int(result["n_valid_elems"]),
            "n_degenerate_elems": int(result["n_degenerate_elems"]),
            "quality": bool(quality),
            "nobisect": bool(nobisect),
            "geometry_scale": float(geometry_scale),
            "support_frac": None if support_frac is None else float(support_frac),
            "support_thickness": None if support_thickness is None else float(support_thickness),
        },
        "fem": {
            "sf_min": float(result["sf_min"]),
            "vm_max": float(result["vm_max"]),
            "vm_mean": float(result["vm_mean"]),
            "disp_max": float(result["disp_max"]),
            **metrics,
        },
        "timing_sec": {
            "preprocess": prep_sec,
            "tetrahedralize": tet_sec,
            "solve": solve_sec,
            "total": prep_sec + tet_sec + solve_sec,
        },
        "artifacts": {
            "level_dir": str(level_dir),
            "summary": str(level_dir / "mesh_fem_verification.json"),
            "prepared_surface": str(surface_path),
        },
    }
    (level_dir / "smooth_fem_level_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", action="append", help="STL path. Can be passed multiple times.")
    parser.add_argument("--out", default=str(ROOT / "results/smooth_fem_truth"))
    parser.add_argument("--reductions", default="0.90", help="Comma-separated PyVista decimate reductions.")
    parser.add_argument("--levels", default="coarse,medium", choices=["coarse", "medium", "coarse,medium"])
    parser.add_argument("--repair_order", choices=["post", "pre", "both"], default="post")
    parser.add_argument("--physical_size_m", type=float, default=None, help="Scale each STL max dimension to this size in meters before FEM.")
    parser.add_argument("--support_frac", type=float, default=None, help="Fixed bottom support thickness as a fraction of scaled mesh height.")
    parser.add_argument(
        "--support_reference",
        choices=["tet", "original"],
        default="tet",
        help="Use repaired tet mesh height or original scaled STL height for --support_frac.",
    )
    parser.add_argument("--yield_stress", type=float, default=50.0e6)
    args = parser.parse_args()

    stls = [Path(x).resolve() for x in args.stl] if args.stl else DEFAULT_STLS
    reductions = [float(x) for x in args.reductions.split(",") if x.strip()]
    level_modes = []
    if "coarse" in args.levels:
        level_modes.append({"quality": False, "nobisect": True})
    if "medium" in args.levels:
        level_modes.append({"quality": True, "nobisect": True})

    out_root = Path(args.out)
    rows = []
    for stl in stls:
        case_dir = out_root / stl.stem
        geometry_scale = 1.0
        support_thickness_override = None
        surface = None
        if args.physical_size_m is not None:
            surface = pv.read(str(stl))
            bounds = np.asarray(surface.bounds, dtype=float)
            max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            geometry_scale = float(args.physical_size_m) / max(float(max_dim), 1e-12)
        if args.support_reference == "original" and args.support_frac is not None:
            if surface is None:
                surface = pv.read(str(stl))
            bounds = np.asarray(surface.bounds, dtype=float)
            original_height = float(bounds[3] - bounds[2]) * geometry_scale
            support_thickness_override = max(float(args.support_frac) * original_height, 1e-12)
        for reduction in reductions:
            for mode in level_modes:
                print(f"[smooth-fem] {stl.name} reduction={reduction} mode={mode}", flush=True)
                try:
                    rows.append(
                        run_one_level(
                            stl=stl,
                            out_dir=case_dir,
                            reduction=reduction,
                            geometry_scale=geometry_scale,
                            support_frac=args.support_frac,
                            support_thickness_override=support_thickness_override,
                            repair_order=args.repair_order,
                            yield_stress=float(args.yield_stress),
                            **mode,
                        )
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "tag": f"red{reduction:.3f}_q{int(mode['quality'])}_nb{int(mode['nobisect'])}".replace(".", "p"),
                            "input_stl": str(stl),
                            "status": "error",
                            "error": repr(exc),
                            "geometry_scale": float(geometry_scale),
                            "support_frac": None if args.support_frac is None else float(args.support_frac),
                            "support_reference": args.support_reference,
                            "support_thickness": None
                            if support_thickness_override is None
                            else float(support_thickness_override),
                        }
                    )
                    print(f"[smooth-fem:error] {stl.name}: {exc!r}", flush=True)

    summary = {
        "schema": "smooth_fem_truth_v1",
        "note": "Surface-mesh FEM pilot; independent of voxel-FEM labels.",
        "support_reference": args.support_reference,
        "repair_order": args.repair_order,
        "n_runs": len(rows),
        "rows": rows,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
