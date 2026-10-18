#!/usr/bin/env python3
"""Evaluate the two-wall physical benchmark with one hard 256 protocol."""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure, label


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/physical_print_controlled_baselines_v2_20260715"
MANIFEST = OUT / "benchmark_manifest.json"
SAR = Path("/data/xvxinyao/ShadowArt-Revisited")
TAG = "shadowforge_two_wall_physical_v2_20260715"
S6 = generate_binary_structure(3, 1)

BASE = ROOT / "experiments/build_anime_two_wall_upright_saving_print_10cm_256_20260710.py"
spec = importlib.util.spec_from_file_location("two_wall_protocol", BASE)
tw = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tw)


def load_target(path: str) -> np.ndarray:
    return tw.n2.load_mask(Path(path), 256)


def targets_for(case: dict) -> tuple[np.ndarray, np.ndarray]:
    images = [load_target(path) for path in case["input_paths"]]
    return tw.image_to_wall_axes(images[0]), tw.image_to_wall_axes(images[1])


def project(volume: np.ndarray) -> list[np.ndarray]:
    volume = np.asarray(volume, dtype=bool)
    return [tw.tw.project_left_wall_x(volume), tw.tw.project_right_wall_y(volume)]


def scores(volume: np.ndarray, targets: tuple[np.ndarray, np.ndarray]) -> dict:
    ious, dices = [], []
    for pred, target in zip(project(volume), targets):
        inter = int(np.logical_and(pred, target).sum())
        union = int(np.logical_or(pred, target).sum())
        denom = int(pred.sum() + target.sum())
        ious.append(inter / max(union, 1))
        dices.append(2 * inter / max(denom, 1))
    return {
        "mean_iou": float(np.mean(ious)), "min_iou": float(np.min(ious)),
        "mean_dice": float(np.mean(dices)), "view_ious": ious,
    }


def two_wall_mitra(targets: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    left, right = targets
    occupied = tw.tw.visual_hull_two_wall(left, right).astype(bool)
    left_cover = occupied.sum(axis=0).astype(np.int32)
    right_cover = occupied.sum(axis=1).astype(np.int32)
    changed = True
    while changed:
        changed = False
        for x, y, z in np.argwhere(occupied):
            if left_cover[y, z] > 1 and right_cover[x, z] > 1:
                occupied[x, y, z] = False
                left_cover[y, z] -= 1
                right_cover[x, z] -= 1
                changed = True
    return occupied


def mesh_to_volume(path: Path, resolution: int = 256) -> tuple[np.ndarray, dict]:
    # Binary STL stores each triangle with independent vertices.  Process it
    # before topology QA so a closed exported surface is not counted as one
    # disconnected component per face.
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    raw_parts = len(mesh.split(only_watertight=False))
    raw_watertight = bool(mesh.is_watertight)
    extents = np.maximum(mesh.extents, 1e-9)
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale((resolution - 6.0) / float(extents.max()))
    mesh.apply_translation(np.full(3, (resolution - 1) / 2.0))
    grid = mesh.voxelized(pitch=1.0)
    if raw_watertight:
        grid = grid.fill()
    points = np.rint(grid.points).astype(np.int32)
    points = points[np.all((points >= 0) & (points < resolution), axis=1)]
    volume = np.zeros((resolution, resolution, resolution), dtype=bool)
    volume[points[:, 0], points[:, 1], points[:, 2]] = True
    return volume, {"mesh_parts": int(raw_parts), "watertight": raw_watertight}


def orientations():
    return list(itertools.product(itertools.permutations(range(3)), itertools.product((0, 1), repeat=3)))


def transform(volume: np.ndarray, perm, flips) -> np.ndarray:
    output = np.transpose(volume, perm)
    for axis, flip in enumerate(flips):
        if flip:
            output = np.flip(output, axis)
    return output


def sadekar_path(kind: str, case_id: str, seed: int) -> Path:
    base = SAR / f"{kind}_results" / TAG / kind / case_id / f"seed_{seed}"
    return base / ("sample_0_final_voxels.npy" if kind == "voxel" else "sample_0_output.obj")


def load_sadekar(kind: str, case_id: str, seed: int) -> tuple[np.ndarray, dict]:
    path = sadekar_path(kind, case_id, seed)
    if kind == "voxel":
        volume = np.squeeze(np.load(path)) > 0.05
        mesh_path = path.parent / "sample_0_output.obj"
        if mesh_path.is_file():
            _, qa = mesh_to_volume(mesh_path)
        else:
            qa = {"mesh_parts": None, "watertight": False}
        return volume, qa
    return mesh_to_volume(path)


def choose_global_transform(payload: dict, kind: str, seeds: list[int]) -> dict:
    case = next(row for row in payload["cases"] if row["split"] == "coordinate_calibration")
    targets = targets_for(case)
    candidates = orientations()
    totals = np.zeros(len(candidates), dtype=np.float64)
    for seed in seeds:
        volume, _ = load_sadekar(kind, case["case_id"], seed)
        for index, (perm, flips) in enumerate(candidates):
            totals[index] += scores(transform(volume, perm, flips), targets)["mean_iou"]
    best = int(np.argmax(totals))
    perm, flips = candidates[best]
    return {
        "perm": list(perm), "flips": list(flips),
        "calibration_mean_iou": float(totals[best] / len(seeds)),
        "selected_on": case["case_id"],
    }


def mesh_qa_from_volume(volume: np.ndarray) -> dict:
    from skimage.measure import marching_cubes
    vertices, faces, _, _ = marching_cubes(np.pad(volume.astype(np.float32), 1), 0.5)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return {"mesh_parts": int(len(mesh.split(only_watertight=False))), "watertight": bool(mesh.is_watertight)}


def qa(volume: np.ndarray, hull: np.ndarray, mesh_qa: dict) -> dict:
    _, components = label(volume, structure=S6)
    survival = binary_erosion(volume, structure=S6).sum() / max(int(volume.sum()), 1)
    saving = 1.0 - int(volume.sum()) / max(int(hull.sum()), 1)
    return {
        "occupied_voxels": int(volume.sum()), "components6": int(components),
        "single_body": bool(components == 1), "watertight": bool(mesh_qa["watertight"]),
        "mesh_parts": mesh_qa["mesh_parts"], "erosion_survival": float(survival),
        "saving": float(saving),
    }


def save_projections(case_id: str, method: str, seed: int | None, volume: np.ndarray) -> None:
    folder = OUT / "evaluated_projections" / case_id / method.lower().replace(" ", "_").replace("-", "_")
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "deterministic" if seed is None else f"seed_{seed}"
    for index, wall_mask in enumerate(project(volume)):
        image = tw.wall_axes_to_image(wall_mask)
        tw.n2.save_mask(image, folder / f"{suffix}_view{index}.png")


def main() -> None:
    payload = json.loads(MANIFEST.read_text())
    seeds = payload["protocol"]["sadekar_seeds"]
    missing = [
        str(sadekar_path(kind, case["case_id"], seed))
        for case in payload["cases"] for seed in seeds for kind in ("voxel", "mesh")
        if not sadekar_path(kind, case["case_id"], seed).is_file()
    ]
    if missing:
        raise RuntimeError(f"baseline artifacts missing: {len(missing)}")
    transforms = {kind: choose_global_transform(payload, kind, seeds) for kind in ("voxel", "mesh")}
    (OUT / "coordinate_normalization.json").write_text(json.dumps(transforms, indent=2))

    rows = []
    for case in (row for row in payload["cases"] if row["report"]):
        targets = targets_for(case)
        hull = tw.tw.visual_hull_two_wall(*targets)
        methods = [
            ("Mitra-style carving", None, two_wall_mitra(targets), None),
            ("ShadowForge", None, np.load(case["shadowforge_voxels256"]).astype(bool), Path(case["shadowforge_stl"])),
        ]
        for method, seed, volume, mesh_path in methods:
            mesh_info = mesh_to_volume(mesh_path)[1] if mesh_path else mesh_qa_from_volume(volume)
            save_projections(case["case_id"], method, seed, volume)
            rows.append({"case_id": case["case_id"], "method": method, "seed": seed, **scores(volume, targets), **qa(volume, hull, mesh_info)})
        for kind, method in (("voxel", "Sadekar voxel-based"), ("mesh", "Sadekar mesh-based")):
            norm = transforms[kind]
            for seed in seeds:
                volume, mesh_info = load_sadekar(kind, case["case_id"], seed)
                volume = transform(volume, norm["perm"], norm["flips"])
                save_projections(case["case_id"], method, seed, volume)
                rows.append({"case_id": case["case_id"], "method": method, "seed": seed, **scores(volume, targets), **qa(volume, hull, mesh_info)})

    fields = ["case_id", "method", "seed", "mean_iou", "min_iou", "mean_dice", "saving", "single_body", "watertight", "components6", "mesh_parts", "erosion_survival", "occupied_voxels", "view_ious"]
    with (OUT / "per_run_results.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    aggregates = []
    for method in payload["methods"]:
        values = [row for row in rows if row["method"].startswith(method.split(" (")[0])]
        aggregates.append({
            "method": method,
            "runs": len(values),
            "mean_iou": float(np.mean([row["mean_iou"] for row in values])),
            "std_iou": float(np.std([row["mean_iou"] for row in values])),
            "mean_dice": float(np.mean([row["mean_dice"] for row in values])),
            "mean_saving": float(np.mean([row["saving"] for row in values])),
            "single_body_rate": float(np.mean([row["single_body"] for row in values])),
            "watertight_rate": float(np.mean([row["watertight"] for row in values])),
        })
    (OUT / "aggregate_results.json").write_text(json.dumps(aggregates, indent=2))
    with (OUT / "aggregate_results.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps({"output": str(OUT), "aggregate": aggregates}, indent=2))


if __name__ == "__main__":
    main()
