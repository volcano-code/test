#!/usr/bin/env python3
"""One-shot end-to-end candidate demo for the current StructShadow story."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, generate_binary_structure, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generation"))

from datasets import load_data_from_list
from voxel_ops import (
    build_tfs_visual_hull,
    calibrate_tfs_projection,
    connectivity_preserving_carve,
    evaluate_tfs_volume,
    make_watertight_printable,
    minimize_material_printable,
    repair_tfs_projection_gaps,
    sdf_watertight_mesh,
)
from structural.analysis.fem_solver import voxel_fem_solve
from structural.analysis.load_generator import LoadGenerator
from run_all_baselines import evaluate_metrics, load_pino


YIELD_STRESS = 50e6
TAU = 1.3


def n6(volume: np.ndarray) -> int:
    _, count = label(volume.astype(bool), structure=generate_binary_structure(3, 1))
    return int(count)


def safe_fem(volume: np.ndarray, load_multiplier: float) -> dict:
    load = LoadGenerator().standing_load(volume.astype(np.float32))
    t0 = time.time()
    fem = voxel_fem_solve(
        volume.astype(np.float32),
        load["force"] * float(load_multiplier),
        load["fixed_mask"],
        verbose=False,
    )
    solid = volume.astype(bool)
    vm = fem["von_mises"][solid]
    vm_max = float(np.nanmax(vm)) if vm.size else float("nan")
    sf = float(YIELD_STRESS / max(vm_max, 1e-12)) if np.isfinite(vm_max) else None
    return {
        "sf_fem": sf,
        "safe": bool(sf is not None and sf >= TAU),
        "vm_max": vm_max,
        "elapsed_s": float(time.time() - t0),
        "n_elements": int(fem.get("n_elements", int(solid.sum()))),
    }


def add_candidate(candidates: dict, name: str, volume: np.ndarray) -> None:
    if volume is not None and volume.any():
        candidates[name] = volume.astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silhouette_dir", type=Path, default=ROOT / "data/silhouettes/bench3")
    parser.add_argument("--names", nargs=3, default=["birds_owl", "x_bat", "sea_snail"])
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--load_multiplier", type=float, default=1.0)
    parser.add_argument("--pino_checkpoint", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model.pt")
    parser.add_argument("--fix_axis", action="store_true")
    parser.add_argument("--save_candidates_dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "results/end_to_end_candidate_demo_20260624.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pino = load_pino(str(args.pino_checkpoint), device)

    paths = [args.silhouette_dir / f"{name}.png" for name in args.names]
    silhouettes, _ = load_data_from_list(
        [str(p) for p in paths],
        True,
        (args.resolution, args.resolution),
        torch.device("cpu"),
        3,
        False,
    )
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in silhouettes]
    calibration = calibrate_tfs_projection(masks, search_res=min(96, args.resolution), fix_axis=bool(args.fix_axis))
    hull, _ = build_tfs_visual_hull(masks, args.resolution, calibration=calibration)
    gap, gap_info = repair_tfs_projection_gaps(masks, args.resolution, calibration=calibration)
    carve = connectivity_preserving_carve(gap)
    min_print, min_info = minimize_material_printable(gap, bridge_radius=1)
    watertight, water_info = make_watertight_printable(min_print, gap)

    candidates: dict[str, np.ndarray] = {}
    add_candidate(candidates, "visual_hull", hull)
    add_candidate(candidates, "gap_filled_hull", gap)
    add_candidate(candidates, "carve_no_bridge", carve)
    add_candidate(candidates, "min_printable_bridge_r1", min_print)
    add_candidate(candidates, "watertight_export", watertight)

    for radius in (1, 2):
        thick = binary_dilation(min_print, structure=np.ones((3, 3, 3), dtype=bool), iterations=radius)
        thick = np.logical_and(thick, gap)
        add_candidate(candidates, f"robust_thickened_r{radius}", thick)

    rows = []
    hull_voxels = max(int(hull.sum()), 1)
    if args.save_candidates_dir is not None:
        args.save_candidates_dir.mkdir(parents=True, exist_ok=True)
    for name, volume in candidates.items():
        metrics = evaluate_tfs_volume(volume.astype(np.float32), masks, calibration, image_size=args.resolution)
        try:
            mesh = sdf_watertight_mesh(volume, smooth_sigma=0.6)
            watertight_mesh = bool(mesh.is_watertight)
            if args.save_candidates_dir is not None:
                mesh.export(args.save_candidates_dir / f"{name}.obj")
        except Exception:
            watertight_mesh = False
        if args.save_candidates_dir is not None:
            np.save(args.save_candidates_dir / f"{name}.npy", volume.astype(np.uint8))
        fem = safe_fem(volume, args.load_multiplier)
        load = LoadGenerator().standing_load(volume.astype(np.float32))
        pino_metrics = evaluate_metrics(
            volume.astype(np.float32),
            hull.astype(np.float32),
            load["force"] * float(args.load_multiplier),
            pino,
            device,
        )
        row = {
            "candidate": name,
            "occupied_voxels": int(volume.sum()),
            "material_saved_vs_hull": float(1.0 - int(volume.sum()) / hull_voxels),
            "mean_iou": metrics["mean_iou"],
            "min_iou": metrics["min_iou"],
            "component_count_6": n6(volume),
            "watertight": watertight_mesh,
            "pino": {
                "sf_pino": float(pino_metrics["sf_min_pino"]),
                "raw_pino_safe": bool(pino_metrics["sf_min_pino"] >= TAU),
            },
            "fem": fem,
            "eligible": bool(metrics["mean_iou"] >= 0.95 and n6(volume) == 1 and watertight_mesh and fem["safe"]),
        }
        rows.append(row)

    eligible = [r for r in rows if r["eligible"]]
    selected = min(eligible, key=lambda r: r["occupied_voxels"]) if eligible else None
    result = {
        "schema_version": "end_to_end_candidate_demo_v1",
        "silhouettes": [str(p) for p in paths],
        "resolution": int(args.resolution),
        "branch": "fixed" if args.fix_axis else "free",
        "saved_candidates_dir": str(args.save_candidates_dir) if args.save_candidates_dir is not None else None,
        "load_multiplier": float(args.load_multiplier),
        "safety_threshold": TAU,
        "num_candidates_generated": len(rows),
        "num_eligible_candidates": len(eligible),
        "selected_candidate": selected,
        "candidate_rows": sorted(rows, key=lambda r: r["occupied_voxels"]),
        "pipeline_notes": {
            "gap_fill_added_voxels": gap_info.get("repair_voxels_added"),
            "bridge_voxels": min_info.get("bridge_voxels"),
            "watertight_material_saved": water_info.get("material_saved_ratio"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
