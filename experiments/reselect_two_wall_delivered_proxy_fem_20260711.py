#!/usr/bin/env python3
"""Re-select two-wall print candidates with FEM on conservative delivered proxies.

This is the two-wall counterpart of the 48-core/256-delivery structural rule:
each actual 256^3 candidate is conservatively pooled to 48^3, and voxel Hex8 FEM
is evaluated in the candidate's selected print orientation.  No PINO is used.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import generate_binary_structure

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]
BASE = ROOT / "experiments/reselect_two_wall_fdm_aware_10cm_256_20260710.py"
OUT = ROOT / "results/anime_two_wall_delivered_proxy_fem_10cm_256_20260711"
TAU, PROXY_RES = 1.3, 48

spec = importlib.util.spec_from_file_location("two_wall_fdm", str(BASE))
tw = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tw)
from structural.analysis.fem_solver import voxel_fem_solve  # noqa: E402
from structural.analysis.load_generator import LoadGenerator  # noqa: E402


def conservative_pool(volume: np.ndarray, target: int = PROXY_RES) -> np.ndarray:
    edges = np.rint(np.linspace(0, volume.shape[0], target + 1)).astype(np.int64)
    pooled = np.maximum.reduceat(volume.astype(np.uint8), edges[:-1], axis=0)
    pooled = np.maximum.reduceat(pooled, edges[:-1], axis=1)
    pooled = np.maximum.reduceat(pooled, edges[:-1], axis=2)
    return pooled.astype(bool)


def fem_proxy(volume256: np.ndarray) -> dict:
    proxy = conservative_pool(volume256)
    load = LoadGenerator().standing_load(proxy.astype(np.float32))
    fem = voxel_fem_solve(proxy.astype(np.float32), load["force"], load["fixed_mask"], verbose=False)
    vm = fem["von_mises"][proxy]
    vm_max = float(np.max(vm)) if vm.size else float("inf")
    sf = float(50e6 / max(vm_max, 1e-12))
    return {"proxy_resolution": PROXY_RES, "proxy_occupied": int(proxy.sum()),
            "sf_fem_delivered_proxy": sf, "vm_max": vm_max, "safe": bool(sf >= TAU),
            "fem_time_s": float(fem["total_time_s"])}


def choose_case(name: str, left_file: str, right_file: str, min_saving: float) -> dict:
    left_img = tw.n2.load_mask(tw.tw.tw.SRC / left_file, tw.R)
    right_img = tw.n2.load_mask(tw.tw.tw.SRC / right_file, tw.R)
    left, right = tw.tw.image_to_wall_axes(left_img), tw.tw.image_to_wall_axes(right_img)
    candidates = []
    for candidate_name, volume, hull in tw.candidate_pool(left, right):
        metrics = tw.tw.tw.metrics_two_wall(volume, left, right, int(hull.sum()))
        if not (metrics["mean_iou"] >= .97 and metrics["min_iou"] >= .95 and
                metrics["saving"] >= min_saving and metrics["components"] == 1 and
                metrics["erosion_survival"] >= .08):
            continue
        for orientation, print_volume in tw.print_orientations(volume):
            fdm = tw.support_metrics(print_volume)
            structural = fem_proxy(print_volume)
            if structural["safe"]:
                score = (fdm["unsupported_ratio"], fdm["significant_islands"],
                         max(0.0, tw.MIN_BED_MM2 - fdm["bed_area_mm2"]), -metrics["saving"])
                candidates.append((score, candidate_name, volume, metrics, orientation, print_volume, fdm, structural))
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    if not candidates:
        record = {"case": name, "accepted": False,
                  "reason": "no candidate satisfied visual, manufacturing, FDM, and delivered-proxy FEM gates"}
        (directory / "FEM_DELIVERED_PROXY_QA.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
    _, candidate_name, display, metrics, orientation, printed, fdm, structural = min(candidates, key=lambda item: item[0])
    display_stl = directory / f"{name}_display_fixed_delivered_proxy_fem_10cm_256.stl"
    print_stl = directory / f"{name}_print_{orientation}_delivered_proxy_fem_10cm_256.stl"
    display_mesh = tw.n2.export_stl(np.transpose(display, (2, 1, 0)), display_stl)
    print_mesh = tw.n2.export_stl(np.transpose(printed, (2, 1, 0)), print_stl)
    np.save(directory / "display_fixed_voxels256.npy", display.astype(np.uint8))
    np.save(directory / f"print_{orientation}_voxels256.npy", printed.astype(np.uint8))
    left_pred = tw.tw.wall_axes_to_image(tw.tw.tw.project_left_wall_x(display))
    right_pred = tw.tw.wall_axes_to_image(tw.tw.tw.project_right_wall_y(display))
    tw.n2.save_mask(left_img, directory / "left_wall_x_target.png")
    tw.n2.save_mask(right_img, directory / "right_wall_y_target.png")
    tw.n2.save_mask(left_pred, directory / "left_wall_x_projection.png")
    tw.n2.save_mask(right_pred, directory / "right_wall_y_projection.png")
    tw.tw.tw.contact_sheet(name, left_img, right_img, left_pred, right_pred, directory / "input_vs_projection.png")
    record = {"case": name, "accepted": True, "candidate": candidate_name,
              "display_coordinate": "fixed two-wall coordinate; left image -> X wall, right image -> Y wall",
              "print_orientation": orientation, "mean_iou": float(metrics["mean_iou"]),
              "min_iou": float(metrics["min_iou"]), "mean_dice": float(metrics["mean_dice"]),
              "saving_vs_two_wall_hull": float(metrics["saving"]), "components_6conn": int(metrics["components"]),
              "erosion_survival": float(metrics["erosion_survival"]), "fdm_proxy": fdm,
              "structural": structural, "display_stl": str(display_stl), "print_stl": str(print_stl),
              "display_mesh_watertight": bool(display_mesh.is_watertight),
              "print_mesh_watertight": bool(print_mesh.is_watertight),
              "screen": "direct voxel Hex8 FEM on conservative 48^3 proxy of delivered 256^3 print orientation"}
    (directory / "FEM_DELIVERED_PROXY_QA.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = [choose_case(*pair, .25) for pair in tw.tw.tw.PAIRS]
    (OUT / "MANIFEST.json").write_text(json.dumps({"schema": "two_wall_delivered_proxy_fem_v1",
        "structural_proxy": "conservative 256-to-48 max pooling", "sf_threshold": TAU, "cases": records}, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
