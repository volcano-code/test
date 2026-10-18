#!/usr/bin/env python3
"""Controlled geometry alternatives and component ablations for ShadowForge."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure, label


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/geometry_alternatives_ablation_20260710"
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from mitra_baseline import mitra_carve
from voxel_ops import (
    _build_tfs_canonical_planes,
    calibrate_tfs_projection,
    build_tfs_visual_hull,
    connect6_min_bridges,
    connectivity_preserving_carve,
    make_watertight_printable,
    repair_tfs_projection_gaps,
)


CASES = {
    "characters": ["Spider-Man.png", "superman.png", "batman.png"],
    "symbols": ["robust_spade.png", "robust_star.png", "robust_flower.png"],
    "objects": ["teddy2.png", "duck.png", "hand.png"],
    "geometry": ["robust_ring.png", "robust_hexagon.png", "robust_diamond.png"],
    "anime": ["anime_v40_cape_wizard.png", "anime_v40_magical_girl.png", "anime_v40_super_robot.png"],
    "icons": ["heart.png", "like.png", "victory.png"],
    "animals": ["bunny_0.png", "armadillo_0.png", "puma.png"],
    "robust_shapes": ["robust_cross.png", "robust_club.png", "robust_v38_gear.png"],
}
RESOLUTIONS = (32, 48, 64, 96)
S6 = generate_binary_structure(3, 1)


def load_mask(name: str, resolution: int) -> np.ndarray:
    path = ROOT / "generation" / name
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)
    corners = np.r_[image[:4, :4].ravel(), image[-4:, -4:].ravel()]
    return image > 127 if np.median(corners) < 127 else image < 127


def mesh_qa(volume: np.ndarray) -> dict:
    import trimesh
    from skimage.measure import marching_cubes

    try:
        vertices, faces, _, _ = marching_cubes(np.pad(volume.astype(np.float32), 1), 0.5)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        parts = mesh.split(only_watertight=False)
        return {
            "watertight": bool(mesh.is_watertight),
            "mesh_parts": int(len(parts)),
            "mesh_vertices": int(len(mesh.vertices)),
            "mesh_faces": int(len(mesh.faces)),
        }
    except Exception as exc:
        return {"watertight": False, "mesh_parts": 0, "mesh_error": str(exc)}


def metrics(name: str, volume: np.ndarray, hull: np.ndarray, planes: dict, elapsed: float) -> dict:
    volume = np.asarray(volume).astype(bool)
    preds = [volume.max(0), volume.max(1), volume.max(2)]
    targets = [planes["x"], planes["y"], planes["z"]]
    ious, dices = [], []
    for pred, target in zip(preds, targets):
        inter = int(np.logical_and(pred, target).sum())
        union = int(np.logical_or(pred, target).sum())
        denom = int(pred.sum() + target.sum())
        ious.append(inter / max(union, 1))
        dices.append(2 * inter / max(denom, 1))
    _, components = label(volume, structure=S6)
    eroded = binary_erosion(volume, structure=S6)
    survival = float(eroded.sum() / max(int(volume.sum()), 1))
    return {
        "variant": name,
        "occupied_voxels": int(volume.sum()),
        "mean_iou": float(np.mean(ious)),
        "worst_iou": float(np.min(ious)),
        "mean_dice": float(np.mean(dices)),
        "view_ious": ious,
        "saving": float(1.0 - volume.sum() / max(int(hull.sum()), 1)),
        "components6": int(components),
        "single_body": bool(components == 1),
        "erosion_survival": survival,
        "wall_pass": bool(survival >= 0.15),
        "elapsed_s": float(elapsed),
        **mesh_qa(volume),
    }


def run_case(case_id: str, names: list[str], resolution: int) -> dict:
    masks = [load_mask(name, resolution) for name in names]
    calibration = calibrate_tfs_projection(masks, search_res=min(96, resolution), fix_axis=True)
    hull, _ = build_tfs_visual_hull(masks, resolution, calibration=calibration, apply_cleanup=False)
    planes = _build_tfs_canonical_planes(masks, resolution, calibration)
    rows, volumes = [], {}

    volumes["Visual hull only"] = hull
    volumes["Uniform hull shell"] = hull & ~binary_erosion(hull, structure=S6, iterations=1)

    t = time.perf_counter()
    gap, _ = repair_tfs_projection_gaps(masks, resolution, calibration=calibration)
    gap_time = time.perf_counter() - t
    volumes["Gap filling only"] = gap

    t = time.perf_counter()
    volumes["Projection-only carving"] = mitra_carve(gap)
    projection_carve_time = time.perf_counter() - t

    t = time.perf_counter()
    no_gap_carved = connectivity_preserving_carve(hull, max_passes=40)
    no_gap_bridged, _ = connect6_min_bridges(no_gap_carved, hull, bridge_radius=1)
    no_gap_final, _ = make_watertight_printable(no_gap_bridged, hull, max_thicken=3)
    no_gap_time = time.perf_counter() - t
    volumes["Guarded carving w/o gap filling"] = no_gap_final

    t = time.perf_counter()
    carved = connectivity_preserving_carve(gap, max_passes=40)
    bridged, _ = connect6_min_bridges(carved, gap, bridge_radius=1)
    guarded_time = time.perf_counter() - t
    volumes["Guarded carving w/o mesh repair"] = bridged

    t = time.perf_counter()
    full, _ = make_watertight_printable(bridged, gap, max_thicken=3)
    full_time = time.perf_counter() - t
    volumes["Full geometry pipeline"] = full

    times = {
        "Visual hull only": 0.0,
        "Uniform hull shell": 0.0,
        "Gap filling only": gap_time,
        "Projection-only carving": projection_carve_time,
        "Guarded carving w/o gap filling": no_gap_time,
        "Guarded carving w/o mesh repair": guarded_time,
        "Full geometry pipeline": gap_time + guarded_time + full_time,
    }
    for name, volume in volumes.items():
        rows.append(metrics(name, volume, hull, planes, times[name]))
        if resolution == 48 and case_id in {"objects", "animals"}:
            artifact = OUT / "artifacts" / case_id
            artifact.mkdir(parents=True, exist_ok=True)
            np.save(artifact / f"{name.lower().replace(' ', '_').replace('/', '_')}.npy", volume.astype(np.uint8))
    return {"case_id": case_id, "inputs": names, "resolution": resolution, "rows": rows}


def aggregate(records: list[dict]) -> list[dict]:
    variants = [row["variant"] for row in records[0]["rows"]]
    out = []
    for variant in variants:
        rows = [next(x for x in record["rows"] if x["variant"] == variant) for record in records]
        out.append({
            "variant": variant,
            "n": len(rows),
            "mean_iou": float(np.mean([x["mean_iou"] for x in rows])),
            "worst_iou": float(np.min([x["worst_iou"] for x in rows])),
            "mean_dice": float(np.mean([x["mean_dice"] for x in rows])),
            "mean_saving": float(np.mean([x["saving"] for x in rows])),
            "single_body_rate": float(np.mean([x["single_body"] for x in rows])),
            "watertight_rate": float(np.mean([x["watertight"] for x in rows])),
            "wall_pass_rate": float(np.mean([x["wall_pass"] for x in rows])),
            "mean_runtime_s": float(np.mean([x["elapsed_s"] for x in rows])),
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for resolution in RESOLUTIONS:
        for case_id, names in CASES.items():
            print(f"{resolution}: {case_id}", flush=True)
            records.append(run_case(case_id, names, resolution))
    summary = aggregate(records)
    payload = {
        "schema": "controlled_geometry_alternatives_ablation_v1",
        "fixed_camera": True,
        "resolutions": list(RESOLUTIONS),
        "cases": list(CASES),
        "n_case_resolution_rows": len(records),
        "records": records,
        "aggregate": summary,
    }
    (OUT / "geometry_ablation_results.json").write_text(json.dumps(payload, indent=2))
    columns = list(summary[0])
    with (OUT / "geometry_ablation_results.tsv").open("w") as f:
        f.write("\t".join(columns) + "\n")
        for row in summary:
            f.write("\t".join(str(row[c]) for c in columns) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
