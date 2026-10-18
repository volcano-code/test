#!/usr/bin/env python3
"""Select two-wall physical exemplars with explicit print-orientation QA.

The display mesh remains in the fixed wall coordinate system.  A separate
print-oriented mesh is exported for the slicer; its inverse rigid transform is
recorded so the printed object can be returned to the wall-corner display pose.
"""

from __future__ import annotations

import importlib.util
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, label


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
BASE = ROOT / "experiments/build_anime_two_wall_upright_saving_print_10cm_256_20260710.py"
OUT = ROOT / "results/anime_two_wall_fdm_aware_reselected_10cm_256_20260710"
R, PRINT_MM = 256, 100.0
MIN_BED_MM2 = 80.0
S2 = np.ones((3, 3), dtype=bool)
S6 = generate_binary_structure(3, 1)

spec = importlib.util.spec_from_file_location("two_wall", str(BASE))
tw = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tw)
n2 = tw.n2


def support_metrics(v: np.ndarray) -> dict:
    """Conservative 45-degree, 3x3 previous-layer support proxy."""
    v = v.astype(bool)
    coords = np.where(v)
    if not coords[0].size:
        return {"unsupported_ratio": 1.0, "unsupported_voxels": 0, "significant_islands": 0,
                "bed_area_mm2": 0.0}
    z0, z1 = int(coords[2].min()), int(coords[2].max())
    unsupported_total = islands = 0
    for z in range(z0 + 1, z1 + 1):
        unsupported = v[:, :, z] & ~binary_dilation(v[:, :, z - 1], structure=S2)
        unsupported_total += int(unsupported.sum())
        labs, count = label(unsupported, structure=S2)
        for idx in range(1, count + 1):
            pts = np.argwhere(labs == idx)
            if pts.size == 0:
                continue
            span = int(np.ptp(pts, axis=0).max()) + 1
            if len(pts) >= 16 or span * PRINT_MM / R >= 4.0:
                islands += 1
    area = float(v[:, :, z0].sum() * (PRINT_MM / R) ** 2)
    return {
        "unsupported_ratio": float(unsupported_total / max(int(v.sum()), 1)),
        "unsupported_voxels": int(unsupported_total),
        "significant_islands": int(islands),
        "bed_area_mm2": area,
    }


def print_orientations(v: np.ndarray):
    # Each transform defines which display axis becomes printer +Z.
    for name, oriented in [
        ("display_z_up", v),
        ("display_z_down", v[:, :, ::-1]),
        ("display_x_up", np.transpose(v, (1, 2, 0))),
        ("display_x_down", np.transpose(v, (1, 2, 0))[:, :, ::-1]),
        ("display_y_up", np.transpose(v, (0, 2, 1))),
        ("display_y_down", np.transpose(v, (0, 2, 1))[:, :, ::-1]),
    ]:
        yield name, oriented.copy()


def candidate_pool(left: np.ndarray, right: np.ndarray):
    hull = tw.tw.visual_hull_two_wall(left, right)
    yield "hull", tw.tw.keep_largest(hull), hull
    for offset in (0, 7, 17, 31, 47):
        seed = tw.sparse_cover(left, right, offset)
        for dilation in (10, 12, 14, 16, 18, 20, 22, 24, 28, 32):
            volume = binary_dilation(seed, n2.S6, iterations=dilation) & hull
            yield f"sparse_o{offset}_d{dilation}", tw.repair_projection(volume, hull), hull


def choose_case(name: str, left_file: str, right_file: str, min_saving: float) -> dict:
    left_img = n2.load_mask(tw.tw.SRC / left_file, R)
    right_img = n2.load_mask(tw.tw.SRC / right_file, R)
    left, right = tw.image_to_wall_axes(left_img), tw.image_to_wall_axes(right_img)
    accepted = []
    for label_name, volume, hull in candidate_pool(left, right):
        metrics = tw.tw.metrics_two_wall(volume, left, right, int(hull.sum()))
        if not (metrics["mean_iou"] >= .97 and metrics["min_iou"] >= .95 and
                metrics["saving"] >= min_saving and metrics["components"] == 1 and
                metrics["erosion_survival"] >= .08):
            continue
        for orientation, print_volume in print_orientations(volume):
            fdm = support_metrics(print_volume)
            # This does not claim support-free printing.  It ranks candidates by
            # the amount and fragmentation of temporary support that slicing must add.
            score = (fdm["unsupported_ratio"], fdm["significant_islands"],
                     max(0.0, MIN_BED_MM2 - fdm["bed_area_mm2"]), -metrics["saving"])
            accepted.append((score, label_name, volume, hull, metrics, orientation, print_volume, fdm))
    if not accepted:
        return {"case": name, "accepted": False, "reason": "no candidate passed the fixed-view and wall-thickness gates"}
    _, label_name, volume, hull, metrics, orientation, print_volume, fdm = min(accepted, key=lambda item: item[0])
    directory = OUT / name
    (directory / "inputs").mkdir(parents=True, exist_ok=True)
    (directory / "projections").mkdir(parents=True, exist_ok=True)
    display_stl = directory / f"{name}_display_fixed_10cm_256.stl"
    print_stl = directory / f"{name}_print_{orientation}_10cm_256.stl"
    # export_stl expects voxel order [Z, Y, X].
    display_mesh = n2.export_stl(np.transpose(volume, (2, 1, 0)), display_stl)
    print_mesh = n2.export_stl(np.transpose(print_volume, (2, 1, 0)), print_stl)
    np.save(directory / "display_fixed_voxels256.npy", volume.astype(np.uint8))
    np.save(directory / f"print_{orientation}_voxels256.npy", print_volume.astype(np.uint8))
    left_pred = tw.wall_axes_to_image(tw.tw.project_left_wall_x(volume))
    right_pred = tw.wall_axes_to_image(tw.tw.project_right_wall_y(volume))
    n2.save_mask(left_img, directory / "inputs/left_wall_x_target.png")
    n2.save_mask(right_img, directory / "inputs/right_wall_y_target.png")
    n2.save_mask(left_pred, directory / "projections/left_wall_x_projection.png")
    n2.save_mask(right_pred, directory / "projections/right_wall_y_projection.png")
    tw.tw.contact_sheet(name, left_img, right_img, left_pred, right_pred, directory / "input_vs_projection.png")
    record = {
        "case": name,
        "accepted": True,
        "display_coordinate": "fixed two-wall coordinate; left image -> X wall, right image -> Y wall",
        "display_stl": str(display_stl),
        "print_stl": str(print_stl),
        "print_orientation": orientation,
        "inverse_pose_required_for_display": True,
        "candidate": label_name,
        "mean_iou": float(metrics["mean_iou"]), "min_iou": float(metrics["min_iou"]),
        "mean_dice": float(metrics["mean_dice"]), "saving_vs_two_wall_hull": float(metrics["saving"]),
        "components_6conn": int(metrics["components"]), "erosion_survival": float(metrics["erosion_survival"]),
        "display_mesh_parts": int(len(display_mesh.split(only_watertight=False))),
        "display_mesh_watertight": bool(display_mesh.is_watertight),
        "print_mesh_parts": int(len(print_mesh.split(only_watertight=False))),
        "print_mesh_watertight": bool(print_mesh.is_watertight),
        "fdm_proxy": fdm,
        "slicer_policy": "temporary supports may be used; accept only after the slicer confirms all supports are externally reachable",
        "input_paths": [str(tw.tw.SRC / left_file), str(tw.tw.SRC / right_file)],
    }
    (directory / "FDM_SELECTION_QA.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--min-saving", type=float, default=.25)
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    records = [choose_case(*pair, args.min_saving) for pair in tw.tw.PAIRS]
    (OUT / "MANIFEST.json").write_text(json.dumps({
        "schema": "two_wall_fdm_aware_reselection_v1", "print_mm": PRINT_MM,
        "minimum_saving": args.min_saving,
        "fdm_proxy": "45-degree previous-layer support proxy; final printer acceptance requires slicer inspection and a 50 mm trial",
        "cases": records,
    }, ensure_ascii=False, indent=2) + "\n")
    for record in records:
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
