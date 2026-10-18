#!/usr/bin/env python3
"""Ablate the deterministic StructShadow generation/export stages."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import generate_binary_structure, label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generation.voxel_ops import (
    _build_tfs_canonical_planes,
    analyze_connected_components,
    build_tfs_visual_hull,
    connect6_min_bridges,
    connectivity_preserving_carve,
    make_watertight_printable,
    repair_tfs_projection_gaps,
)


DEFAULT_CASES = {
    "characters": ["Spider-Man.png", "superman.png", "batman.png"],
    "symbols": ["robust_spade.png", "robust_star.png", "robust_flower.png"],
    "objects": ["teddy2.png", "duck.png", "hand.png"],
    "geometry": ["robust_ring.png", "robust_hexagon.png", "robust_diamond.png"],
    "anime": ["anime_v40_cape_wizard.png", "anime_v40_magical_girl.png", "anime_v40_super_robot.png"],
    "icons": ["heart.png", "like.png", "victory.png"],
    "animals": ["bunny_0.png", "armadillo_0.png", "puma.png"],
    "robust_shapes": ["robust_cross.png", "robust_club.png", "robust_v38_gear.png"],
}


def load_mask(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    # Project inputs use bright silhouettes on a dark background.
    return image > 128


def projection_metrics(volume: np.ndarray, targets: list[np.ndarray]) -> dict:
    projections = [volume.max(axis=0), volume.max(axis=1), volume.max(axis=2)]
    ious = []
    for projection, target in zip(projections, targets):
        intersection = np.logical_and(projection, target).sum()
        union = np.logical_or(projection, target).sum()
        ious.append(float(intersection / union) if union else 1.0)
    return {"view_ious": ious, "mean_iou": float(np.mean(ious)), "min_iou": float(np.min(ious))}


def mesh_metrics(volume: np.ndarray) -> dict:
    try:
        import trimesh
        from skimage.measure import marching_cubes

        vertices, faces, _, _ = marching_cubes(np.pad(volume.astype(np.float32), 1), level=0.5)
        mesh = trimesh.Trimesh(vertices, faces, process=True)
        return {
            "mesh_watertight": bool(mesh.is_watertight),
            "mesh_winding_consistent": bool(mesh.is_winding_consistent),
            "mesh_vertices": int(len(mesh.vertices)),
            "mesh_faces": int(len(mesh.faces)),
        }
    except Exception as exc:
        return {"mesh_watertight": False, "mesh_error": str(exc)}


def summarize_stage(name: str, volume: np.ndarray, hull_voxels: int, targets: list[np.ndarray], elapsed: float) -> dict:
    cc26 = analyze_connected_components(volume)
    _, cc6 = label(volume, structure=generate_binary_structure(3, 1))
    return {
        "stage": name,
        "occupied_voxels": int(volume.sum()),
        "material_saved_vs_hull": float(1.0 - volume.sum() / max(hull_voxels, 1)),
        "component_count_26": int(cc26["component_count"]),
        "component_count_6": int(cc6),
        "elapsed_s": float(elapsed),
        **projection_metrics(volume, targets),
        **mesh_metrics(volume),
    }


def run_case(case_name: str, files: list[str], resolution: int) -> dict:
    input_masks = [load_mask(ROOT / "generation" / name, resolution) for name in files]
    rows = []

    t0 = time.perf_counter()
    hull, calibration = build_tfs_visual_hull(input_masks, resolution, apply_cleanup=False)
    planes = _build_tfs_canonical_planes(input_masks, resolution, calibration)
    targets = [planes["x"], planes["y"], planes["z"]]
    rows.append(summarize_stage("visual_hull", hull, int(hull.sum()), targets, time.perf_counter() - t0))

    t0 = time.perf_counter()
    repaired, repair_info = repair_tfs_projection_gaps(input_masks, resolution, calibration=calibration)
    rows.append(summarize_stage("gap_fill", repaired, int(hull.sum()), targets, time.perf_counter() - t0))

    t0 = time.perf_counter()
    carved = connectivity_preserving_carve(repaired, max_passes=40)
    carve_info = {"occupied_voxels": int(carved.sum())}
    rows.append(summarize_stage("carve_no_robust_bridge", carved, int(hull.sum()), targets, time.perf_counter() - t0))

    t0 = time.perf_counter()
    printable, bridge_voxels = connect6_min_bridges(carved, repaired, bridge_radius=1)
    printable_info = {"occupied_voxels": int(printable.sum()), "bridge_voxels": int(bridge_voxels)}
    rows.append(summarize_stage("carve_plus_bridge", printable, int(hull.sum()), targets, time.perf_counter() - t0))

    t0 = time.perf_counter()
    watertight, watertight_info = make_watertight_printable(printable, repaired, max_thicken=3)
    rows.append(summarize_stage("full_pipeline", watertight, int(hull.sum()), targets, time.perf_counter() - t0))

    return {
        "case": case_name,
        "files": files,
        "resolution": resolution,
        "calibration": calibration,
        "repair_info": repair_info,
        "carve_info": carve_info,
        "printable_info": printable_info,
        "watertight_info": watertight_info,
        "stages": rows,
    }


def aggregate(cases: list[dict]) -> dict:
    stage_names = [row["stage"] for row in cases[0]["stages"]]
    summary = {}
    for stage in stage_names:
        rows = [next(row for row in case["stages"] if row["stage"] == stage) for case in cases]
        summary[stage] = {
            "n": len(rows),
            "mean_iou": float(np.mean([row["mean_iou"] for row in rows])),
            "min_iou": float(np.min([row["min_iou"] for row in rows])),
            "mean_material_saved_vs_hull": float(np.mean([row["material_saved_vs_hull"] for row in rows])),
            "single_body_6_rate": float(np.mean([row["component_count_6"] == 1 for row in rows])),
            "watertight_rate": float(np.mean([row["mesh_watertight"] for row in rows])),
            "mean_elapsed_s": float(np.mean([row["elapsed_s"] for row in rows])),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "generation_pipeline_ablation.json")
    args = parser.parse_args()
    cases = [run_case(name, files, args.resolution) for name, files in DEFAULT_CASES.items()]
    payload = {
        "protocol": "deterministic_tfs_generation_ablation_v1",
        "resolution": args.resolution,
        "cases": cases,
        "aggregate": aggregate(cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
