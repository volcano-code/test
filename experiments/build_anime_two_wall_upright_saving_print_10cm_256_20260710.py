#!/usr/bin/env python3
"""Build print-ready, material-saving two-wall fixed anime candidates.

The candidate search is performed in the fixed wall-coordinate field
``[x, y, z]``.  Before mesh extraction it is transposed to ``[z, y, x]`` so
that the top of each input image is physical +Z in the exported STL.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
BASE_SCRIPT = ROOT / "experiments/build_anime_two_wall_fixed_no_floor_10cm_256_20260709.py"
OUT = ROOT / "results/anime_two_wall_fixed_no_floor_physical_axes_qualityfirst_10cm_256_20260710"

spec = importlib.util.spec_from_file_location("two_wall", str(BASE_SCRIPT))
tw = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tw)
n2 = tw.n2


def sparse_cover(left_yz: np.ndarray, right_xz: np.ndarray, offset: int) -> np.ndarray:
    """One connected candidate seed covering both wall silhouettes."""
    r = left_yz.shape[0]
    vol = np.zeros((r, r, r), dtype=bool)
    for z in range(r):
        ys = np.flatnonzero(left_yz[:, z])
        xs = np.flatnonzero(right_xz[:, z])
        if not (xs.size and ys.size):
            continue
        for k in range(max(xs.size, ys.size)):
            vol[int(xs[(k + offset) % xs.size]), int(ys[k % ys.size]), z] = True
    return vol


def repair_projection(vol: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """Restore only missing projection rays inside the visual hull."""
    vol = tw.keep_largest(vol & hull)
    for y, z in np.argwhere(tw.project_left_wall_x(hull) & ~tw.project_left_wall_x(vol)):
        xs = np.flatnonzero(hull[:, y, z])
        if xs.size:
            vol[int(xs[len(xs) // 2]), y, z] = True
    for x, z in np.argwhere(tw.project_right_wall_y(hull) & ~tw.project_right_wall_y(vol)):
        ys = np.flatnonzero(hull[x, :, z])
        if ys.size:
            vol[x, int(ys[len(ys) // 2]), z] = True
    return tw.keep_largest(vol & hull)


def build_candidates(left: np.ndarray, right: np.ndarray) -> list[tuple[str, np.ndarray]]:
    hull = tw.visual_hull_two_wall(left, right)
    candidates: list[tuple[str, np.ndarray]] = [("hull", tw.keep_largest(hull))]
    for offset in (0, 7, 17, 31, 47):
        seed = sparse_cover(left, right, offset)
        for dilation in (6, 8, 10, 12, 14, 16, 18, 20):
            vol = binary_dilation(seed, n2.S6, iterations=dilation) & hull
            candidates.append((f"sparse_o{offset}_d{dilation}", repair_projection(vol, hull)))
    return candidates


def summary(metrics: dict) -> dict:
    return {
        "mean_iou": round(float(metrics["mean_iou"]), 6),
        "min_iou": round(float(metrics["min_iou"]), 6),
        "mean_dice": round(float(metrics["mean_dice"]), 6),
        "view_ious": [round(float(v), 6) for v in metrics["view_ious"]],
        "view_dices": [round(float(v), 6) for v in metrics["view_dices"]],
        "saving_vs_two_wall_hull": round(float(metrics["saving"]), 6),
        "components_6conn": int(metrics["components"]),
        "erosion_survival": round(float(metrics["erosion_survival"]), 6),
        "max_face_occupancy": round(float(metrics["max_face_occupancy"]), 6),
        "bbox_fill": round(float(metrics["bbox_fill"]), 6),
    }


def image_to_wall_axes(image: np.ndarray) -> np.ndarray:
    """Map image [row(top->bottom), col] to wall [horizontal, z(bottom->top)]."""
    return np.flipud(image).T.copy()


def wall_axes_to_image(mask: np.ndarray) -> np.ndarray:
    """Map wall [horizontal, z(bottom->top)] back to image coordinates."""
    return np.flipud(mask.T).copy()


def run_case(name: str, left_file: str, right_file: str) -> dict:
    left_path, right_path = tw.SRC / left_file, tw.SRC / right_file
    left_image = n2.load_mask(left_path, n2.RES_EXPORT)
    right_image = n2.load_mask(right_path, n2.RES_EXPORT)
    # Candidate coordinates are [physical X, physical Y, physical Z].
    # The screen image's row points down, whereas physical Z points up.
    left = image_to_wall_axes(left_image)   # [Y, Z]
    right = image_to_wall_axes(right_image) # [X, Z]
    hull = tw.visual_hull_two_wall(left, right)
    hvox = int(hull.sum())
    eligible = []
    for label, candidate in build_candidates(left, right):
        metrics = tw.metrics_two_wall(candidate, left, right, hvox)
        if (metrics["mean_iou"] >= 0.94 and metrics["min_iou"] >= 0.90
                and metrics["saving"] >= 0.30 and metrics["components"] == 1
                and metrics["erosion_survival"] >= 0.06
                and metrics["max_face_occupancy"] <= 0.18):
            eligible.append((label, candidate, metrics))
    if not eligible:
        return {"case": name, "accepted": False,
                "reason": "no candidate satisfied all fixed-view, fabrication-proxy, and >=30% saving gates"}

    # For physical exemplars, visual legibility is prioritized after the
    # minimum 30% reduction gate.  Material is the secondary tie-breaker.
    label, candidate, metrics = max(
        eligible,
        key=lambda item: (item[2]["min_iou"], item[2]["mean_iou"], item[2]["saving"]),
    )
    # n2.export_stl maps input axes [2,1,0] to physical [X,Y,Z].
    # Candidate axes are [X,Y,Z], hence export a [Z,Y,X] field.
    upright = np.transpose(candidate, (2, 1, 0)).copy()
    out = OUT / name
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "projections").mkdir(parents=True, exist_ok=True)
    stl = out / f"{name}_upright_saving_10cm_256.stl"
    mesh = n2.export_stl(upright, stl)
    np.save(out / f"{name}_upright_saving_voxels256.npy", upright.astype(np.uint8))
    left_projection = wall_axes_to_image(tw.project_left_wall_x(candidate))
    right_projection = wall_axes_to_image(tw.project_right_wall_y(candidate))
    n2.save_mask(left_image, out / "inputs/left_wall_x_target.png")
    n2.save_mask(right_image, out / "inputs/right_wall_y_target.png")
    n2.save_mask(left_projection, out / "projections/left_wall_x_projection.png")
    n2.save_mask(right_projection, out / "projections/right_wall_y_projection.png")
    tw.contact_sheet(name, left_image, right_image, left_projection, right_projection, out / "input_vs_projection.png")
    record = {
        "case": name, "accepted": True,
        "setting": "two-wall fixed, no-floor, physical-axis-correct, quality-first material-saving",
        "projection_model": "physical corner: image top -> physical +Z; image columns -> wall horizontal axes",
        "candidate": label, **summary(metrics),
        "mesh_parts": int(len(mesh.split(only_watertight=False))),
        "mesh_watertight": bool(mesh.is_watertight),
        "input_paths": [str(left_path), str(right_path)],
        "stl": str(stl),
        "voxels": str(out / f"{name}_upright_saving_voxels256.npy"),
        "preview": str(out / "input_vs_projection.png"),
    }
    (out / "CASE_MANIFEST.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = [run_case(*pair) for pair in tw.PAIRS]
    manifest = {"schema": "two_wall_upright_saving_v1", "n_cases": len(records),
                "n_accepted": sum(bool(row.get("accepted")) for row in records), "cases": records}
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in records:
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
