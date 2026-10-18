#!/usr/bin/env python3
"""Evaluate every controlled benchmark artifact with one hard 256^3 protocol."""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure, label


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/controlled_benchmark_geometry_only_20260710"
MANIFEST = OUT / "benchmark_manifest.json"
SAR = Path("/data/xvxinyao/ShadowArt-Revisited")
TAG = "shadowforge_controlled_20260710"
S6 = generate_binary_structure(3, 1)
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from datasets import load_data_from_list
from mitra_baseline import build_mitra
from voxel_ops import (
    build_tfs_visual_hull,
    calibrate_tfs_projection,
    project_tfs_volume,
    resize_binary_mask,
)


def load_targets(paths: list[str]) -> list[np.ndarray]:
    out = []
    for path in paths:
        arr = np.asarray(Image.open(path).convert("L"))
        corners = np.r_[arr[:5, :5].ravel(), arr[-5:, -5:].ravel()]
        mask = arr > 127 if np.median(corners) < 127 else arr < 127
        out.append(resize_binary_mask(mask, 256))
    return out


def load_binary_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    corners = np.r_[arr[:5, :5].ravel(), arr[-5:, -5:].ravel()]
    return resize_binary_mask(arr > 127 if np.median(corners) < 127 else arr < 127, 256)


def mask_scores(preds: list[np.ndarray], targets: list[np.ndarray]) -> dict:
    ious, dices = [], []
    for pred, target in zip(preds, targets):
        inter = int(np.logical_and(pred, target).sum())
        union = int(np.logical_or(pred, target).sum())
        denom = int(pred.sum() + target.sum())
        ious.append(inter / max(union, 1)); dices.append(2 * inter / max(denom, 1))
    return {"native_renderer_mean_iou": float(np.mean(ious)), "native_renderer_mean_dice": float(np.mean(dices))}


def transformed(volume: np.ndarray, perm: tuple[int, int, int], flips: tuple[int, int, int]) -> np.ndarray:
    out = np.transpose(volume, perm)
    for axis, flip in enumerate(flips):
        if flip:
            out = np.flip(out, axis)
    return out


def orientations():
    return list(itertools.product(itertools.permutations(range(3)), itertools.product((0, 1), repeat=3)))


def hard_scores(volume: np.ndarray, targets: list[np.ndarray], calibration: dict) -> dict:
    preds = project_tfs_volume(volume.astype(bool), calibration)[:3]
    preds = [resize_binary_mask(p, 256) for p in preds]
    ious, dices = [], []
    for pred, target in zip(preds, targets):
        inter = int(np.logical_and(pred, target).sum())
        union = int(np.logical_or(pred, target).sum())
        denom = int(pred.sum() + target.sum())
        ious.append(inter / max(union, 1))
        dices.append(2 * inter / max(denom, 1))
    return {
        "mean_iou": float(np.mean(ious)), "worst_iou": float(np.min(ious)),
        "mean_dice": float(np.mean(dices)), "view_ious": ious, "preds": preds,
    }


def mesh_to_volume(path: Path, resolution: int = 256) -> tuple[np.ndarray, dict]:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = mesh.copy()
    parts = mesh.split(only_watertight=False)
    extents = np.maximum(mesh.extents, 1e-9)
    center = mesh.bounds.mean(axis=0)
    scale = (resolution - 6.0) / float(extents.max())
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    mesh.apply_translation(np.full(3, (resolution - 1) / 2.0))
    grid = mesh.voxelized(pitch=1.0)
    if mesh.is_watertight:
        grid = grid.fill()
    points = np.rint(grid.points).astype(np.int32)
    valid = np.all((points >= 0) & (points < resolution), axis=1)
    points = points[valid]
    volume = np.zeros((resolution, resolution, resolution), dtype=bool)
    volume[points[:, 0], points[:, 1], points[:, 2]] = True
    qa = {
        "mesh_parts": int(len(parts)), "mesh_watertight": bool(mesh.is_watertight),
        "mesh_vertices": int(len(mesh.vertices)), "mesh_faces": int(len(mesh.faces)),
        "volume_defined": bool(mesh.is_watertight),
    }
    return volume, qa


def volume_qa(volume: np.ndarray, hull: np.ndarray, mesh_qa: dict | None = None) -> dict:
    volume = volume.astype(bool)
    _, cc = label(volume, structure=S6)
    survival = float(binary_erosion(volume, structure=S6).sum() / max(int(volume.sum()), 1))
    out = {
        "occupied_voxels": int(volume.sum()), "components6": int(cc),
        "single_body": bool(cc == 1), "erosion_survival": survival,
        "wall_pass": bool(survival >= 0.15),
        "saving": float(1.0 - volume.sum() / max(int(hull.sum()), 1)),
    }
    if mesh_qa:
        out.update(mesh_qa)
        if not mesh_qa.get("volume_defined", True):
            out["saving"] = None
    return out


def marching_qa(volume: np.ndarray) -> dict:
    from skimage.measure import marching_cubes
    vertices, faces, _, _ = marching_cubes(np.pad(volume.astype(np.float32), 1), 0.5)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return {
        "mesh_parts": int(len(mesh.split(only_watertight=False))),
        "mesh_watertight": bool(mesh.is_watertight), "volume_defined": True,
    }


def mesh_path_qa(path: Path) -> dict:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return {
        "mesh_parts": int(len(mesh.split(only_watertight=False))),
        "mesh_watertight": bool(mesh.is_watertight), "volume_defined": bool(mesh.is_watertight),
        "source_mesh": str(path),
    }


def sadekar_path(kind: str, case_id: str, seed: int) -> Path:
    base = SAR / f"{kind}_results" / TAG / kind / case_id / f"seed_{seed}"
    return base / ("sample_0_final_voxels.npy" if kind == "voxel" else "sample_0_output.obj")


def choose_global_orientation(manifest: dict, kind: str) -> dict:
    validation = [c for c in manifest["cases"] if c["split"] == "validation"]
    seeds = manifest["protocol"]["sadekar_seeds"]
    candidates = orientations()
    totals = np.zeros(len(candidates), dtype=np.float64)
    count = 0
    for case in validation:
        targets = load_targets(case["input_paths"])
        calibration = calibrate_tfs_projection(targets, search_res=96, fix_axis=True)
        for seed in seeds:
            path = sadekar_path(kind, case["case_id"], seed)
            if kind == "voxel":
                volume = np.squeeze(np.load(path)) > 0.05
            else:
                volume, _ = mesh_to_volume(path)
            for idx, (perm, flips) in enumerate(candidates):
                totals[idx] += hard_scores(transformed(volume, perm, flips), targets, calibration)["mean_iou"]
            count += 1
    best = int(np.argmax(totals))
    perm, flips = candidates[best]
    return {
        "perm": list(perm), "flips": list(flips),
        "validation_mean_iou": float(totals[best] / max(count, 1)),
        "selected_on": f"{len(validation)} validation cases x {len(seeds)} seeds",
    }


def save_preds(case_id: str, method: str, seed: int | None, preds: list[np.ndarray]) -> None:
    folder = OUT / "evaluated_shadows" / case_id / method.lower().replace(" ", "_").replace("-", "_")
    folder.mkdir(parents=True, exist_ok=True)
    suffix = "deterministic" if seed is None else f"seed_{seed}"
    for idx, pred in enumerate(preds):
        Image.fromarray((~pred.astype(bool) * 255).astype(np.uint8)).save(folder / f"{suffix}_view{idx}.png")


def evaluate() -> None:
    manifest = json.loads(MANIFEST.read_text())
    missing = [
        str(sadekar_path(kind, c["case_id"], seed))
        for c in manifest["cases"] for seed in manifest["protocol"]["sadekar_seeds"] for kind in ("voxel", "mesh")
        if not sadekar_path(kind, c["case_id"], seed).exists()
    ]
    if missing:
        raise RuntimeError(f"baseline run incomplete; missing {len(missing)} artifacts")
    transforms = {kind: choose_global_orientation(manifest, kind) for kind in ("voxel", "mesh")}
    (OUT / "coordinate_normalization.json").write_text(json.dumps(transforms, indent=2))

    rows = []
    seeds = manifest["protocol"]["sadekar_seeds"]
    for case in manifest["cases"]:
        targets = load_targets(case["input_paths"])
        calibration = calibrate_tfs_projection(targets, search_res=96, fix_axis=True)
        hull, _ = build_tfs_visual_hull(targets, 256, calibration=calibration, apply_cleanup=False)
        for idx, target in enumerate(targets):
            folder = OUT / "evaluated_shadows" / case["case_id"] / "targets"
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray((~target * 255).astype(np.uint8)).save(folder / f"view{idx}.png")

        mitra, _ = build_mitra(targets, res=256, fix_axis=True)
        sf = np.load(case["shadowforge_voxels256"]).astype(bool)
        for method, volume, qa in (
            ("Mitra-style carving", mitra, marching_qa(mitra)),
            ("ShadowForge", sf, mesh_path_qa(Path(case["shadowforge_stl100mm"]))),
        ):
            score = hard_scores(volume, targets, calibration)
            save_preds(case["case_id"], method, None, score.pop("preds"))
            rows.append({"case_id": case["case_id"], "split": case["split"], "method": method, "seed": None,
                         **score, **volume_qa(volume, hull, qa)})
            np.save(OUT / "evaluated_shadows" / case["case_id"] / f"{method.lower().replace(' ', '_')}_voxels256.npy", volume.astype(np.uint8))

        for kind, method in (("voxel", "Sadekar voxel-based"), ("mesh", "Sadekar mesh-based")):
            transform = transforms[kind]
            for seed in seeds:
                path = sadekar_path(kind, case["case_id"], seed)
                if kind == "voxel":
                    volume = np.squeeze(np.load(path)) > 0.05
                    qa = mesh_path_qa(path.parent / "sample_0_output.obj")
                    qa["volume_defined"] = True
                    native = [load_binary_image(path.parent / f"sample_0view_{idx}_silh.png") for idx in range(3)]
                else:
                    volume, qa = mesh_to_volume(path)
                    native = [load_binary_image(path.parent / f"sample_0_pred_view_{idx}.png") for idx in range(3)]
                volume = transformed(volume, tuple(transform["perm"]), tuple(transform["flips"]))
                score = hard_scores(volume, targets, calibration)
                save_preds(case["case_id"], method, seed, score.pop("preds"))
                rows.append({"case_id": case["case_id"], "split": case["split"], "method": method,
                             "seed": seed, "artifact": str(path), **score, **mask_scores(native, targets),
                             **volume_qa(volume, hull, qa)})

    (OUT / "per_case_metrics.json").write_text(json.dumps(rows, indent=2))
    flat_columns = ["case_id", "split", "method", "seed", "mean_iou", "worst_iou", "mean_dice", "saving",
                    "single_body", "mesh_watertight", "wall_pass", "components6", "erosion_survival",
                    "native_renderer_mean_iou", "native_renderer_mean_dice", "artifact"]
    with (OUT / "per_case_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    test = [r for r in rows if r["split"] == "test"]
    aggregate = []
    for method in ("Mitra-style carving", "Sadekar voxel-based", "Sadekar mesh-based", "ShadowForge"):
        vals = [r for r in test if r["method"] == method]
        saving_vals = [r["saving"] for r in vals if r.get("saving") is not None]
        method_seeds = sorted({r["seed"] for r in vals if r.get("seed") is not None})
        if method_seeds:
            seed_iou = [np.mean([r["mean_iou"] for r in vals if r["seed"] == seed]) for seed in method_seeds]
            seed_dice = [np.mean([r["mean_dice"] for r in vals if r["seed"] == seed]) for seed in method_seeds]
        else:
            seed_iou = [np.mean([r["mean_iou"] for r in vals])]
            seed_dice = [np.mean([r["mean_dice"] for r in vals])]
        aggregate.append({
            "method": method, "case_seed_rows": len(vals),
            "mean_iou": float(np.mean([r["mean_iou"] for r in vals])),
            "std_iou": float(np.std(seed_iou)),
            "worst_view_iou": float(np.min([r["worst_iou"] for r in vals])),
            "mean_dice": float(np.mean([r["mean_dice"] for r in vals])),
            "std_dice": float(np.std(seed_dice)),
            "mean_saving": float(np.mean(saving_vals)) if len(saving_vals) == len(vals) else None,
            "saving_defined_rate": float(len(saving_vals) / max(len(vals), 1)),
            "single_body_rate": float(np.mean([r["single_body"] for r in vals])),
            "watertight_rate": float(np.mean([r["mesh_watertight"] for r in vals])),
            "wall_pass_rate": float(np.mean([r["wall_pass"] for r in vals])),
            "native_renderer_mean_iou_diagnostic": (
                float(np.mean([r["native_renderer_mean_iou"] for r in vals if "native_renderer_mean_iou" in r]))
                if any("native_renderer_mean_iou" in r for r in vals) else None
            ),
        })
    (OUT / "prior_comparison_results.json").write_text(json.dumps({"aggregate": aggregate, "transforms": transforms}, indent=2))
    columns = list(aggregate[0])
    with (OUT / "prior_comparison_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    evaluate()
