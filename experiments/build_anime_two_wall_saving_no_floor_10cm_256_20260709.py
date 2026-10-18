#!/usr/bin/env python3
"""Search material-saving two-wall fixed no-floor anime examples."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
BASE_SCRIPT = ROOT / "experiments/build_anime_two_wall_fixed_no_floor_10cm_256_20260709.py"
OUT = ROOT / "results/anime_two_wall_saving_no_floor_10cm_256_20260709"

spec = importlib.util.spec_from_file_location("twowall", str(BASE_SCRIPT))
tw = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tw)
n2 = tw.n2


def sparse_layer_cover(left_yz: np.ndarray, right_xz: np.ndarray, offset: int = 0) -> np.ndarray:
    """Build one sparse support set that covers both wall projections.

    For each z layer, required left-wall rows y and right-wall columns x are
    paired cyclically. This preserves both projections after enough dilation,
    while avoiding the full Cartesian product of the visual hull.
    """
    r = left_yz.shape[0]
    vol = np.zeros((r, r, r), dtype=bool)
    for z in range(r):
        ys = np.flatnonzero(left_yz[:, z])
        xs = np.flatnonzero(right_xz[:, z])
        if ys.size == 0 or xs.size == 0:
            continue
        m = max(len(xs), len(ys))
        for k in range(m):
            x = int(xs[(k + offset) % len(xs)])
            y = int(ys[k % len(ys)])
            vol[x, y, z] = True
    return vol


def repair_to_targets(vol: np.ndarray, hull: np.ndarray) -> np.ndarray:
    vol = tw.keep_largest(vol & hull)
    miss_l = np.argwhere(tw.project_left_wall_x(hull) & ~tw.project_left_wall_x(vol))
    for y, z in miss_l:
        xs = np.where(hull[:, y, z])[0]
        if xs.size:
            vol[int(xs[len(xs) // 2]), y, z] = True
    miss_r = np.argwhere(tw.project_right_wall_y(hull) & ~tw.project_right_wall_y(vol))
    for x, z in miss_r:
        ys = np.where(hull[x, :, z])[0]
        if ys.size:
            vol[x, int(ys[len(ys) // 2]), z] = True
    return tw.keep_largest(vol & hull)


def candidates(left: np.ndarray, right: np.ndarray) -> list[tuple[str, np.ndarray]]:
    hull = tw.visual_hull_two_wall(left, right)
    out: list[tuple[str, np.ndarray]] = [("hull", tw.keep_largest(hull))]
    for offset in (0, 7, 17, 31):
        base = sparse_layer_cover(left, right, offset=offset)
        for it in (2, 3, 4, 5, 6, 8, 10):
            vol = binary_dilation(base, tw.n2.S6, iterations=it) & hull
            vol = repair_to_targets(vol, hull)
            out.append((f"sparse_o{offset}_d{it}", vol))
    return out


def run_case(case: str, left_name: str, right_name: str) -> dict:
    left_path = tw.SRC / left_name
    right_path = tw.SRC / right_name
    left = n2.load_mask(left_path, n2.RES_EXPORT)
    right = n2.load_mask(right_path, n2.RES_EXPORT)
    hull = tw.visual_hull_two_wall(left, right)
    hull_vox = int(hull.sum())
    rows = []
    for cname, vol in candidates(left, right):
        m = tw.metrics_two_wall(vol, left, right, hull_vox)
        ok = (
            m["mean_iou"] >= 0.94
            and m["min_iou"] >= 0.88
            and m["components"] == 1
            and m["erosion_survival"] >= 0.06
            and m["max_face_occupancy"] <= 0.18
            and m["saving"] >= 0.15
        )
        rows.append((ok, cname, m, vol))
    feasible = [r for r in rows if r[0]]
    if not feasible:
        best = max(rows, key=lambda r: (r[2]["mean_iou"], r[2]["saving"], -r[2]["max_face_occupancy"]))
        return {
            "case": case,
            "accepted": False,
            "best_candidate": best[1],
            "best_metrics": summarize_metrics(best[2]),
            "reason": "no candidate met saving+quality gates",
        }
    chosen = max(feasible, key=lambda r: (r[2]["saving"], r[2]["mean_iou"], -r[2]["max_face_occupancy"]))
    _, cname, m, vol = chosen
    d = OUT / case
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "projections").mkdir(parents=True, exist_ok=True)
    stl = d / f"{case}_10cm_256.stl"
    npy = d / f"{case}_voxels256.npy"
    base = d / "display_base_110x76mm.stl"
    mesh = n2.export_stl(vol, stl)
    n2.export_base(base)
    np.save(npy, vol.astype(np.uint8))
    pl = tw.project_left_wall_x(vol)
    pr = tw.project_right_wall_y(vol)
    n2.save_mask(left, d / "inputs/left_wall_x_view1_target.png")
    n2.save_mask(right, d / "inputs/right_wall_y_view2_target.png")
    n2.save_mask(pl, d / "projections/left_wall_x_view1_projection.png")
    n2.save_mask(pr, d / "projections/right_wall_y_view2_projection.png")
    tw.contact_sheet(case, left, right, pl, pr, d / "input_vs_projection.png")
    rec = {
        "case": case,
        "accepted": True,
        "setting": "two-wall fixed no-floor material-saving",
        "projection_model": "original fixed TFS without floor: view1->x/left wall, view2->y/right wall",
        "candidate": cname,
        **summarize_metrics(m),
        "mesh_parts": int(len(mesh.split(only_watertight=False))),
        "mesh_watertight": bool(mesh.is_watertight),
        "input_paths": [str(left_path), str(right_path)],
        "stl": str(stl),
        "voxels": str(npy),
        "display_base_stl": str(base),
        "preview": str(d / "input_vs_projection.png"),
    }
    (d / "CASE_MANIFEST.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rec


def summarize_metrics(m: dict) -> dict:
    return {
        "mean_iou": round(float(m["mean_iou"]), 6),
        "min_iou": round(float(m["min_iou"]), 6),
        "mean_dice": round(float(m["mean_dice"]), 6),
        "view_ious": [round(float(x), 6) for x in m["view_ious"]],
        "view_dices": [round(float(x), 6) for x in m["view_dices"]],
        "saving_vs_two_wall_hull": round(float(m["saving"]), 6),
        "components_6conn": int(m["components"]),
        "erosion_survival": round(float(m["erosion_survival"]), 6),
        "max_face_occupancy": round(float(m["max_face_occupancy"]), 6),
        "bbox_fill": round(float(m["bbox_fill"]), 6),
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for case, left, right in tw.PAIRS:
        rec = run_case(case, left, right)
        records.append(rec)
        if rec["accepted"]:
            print("OK", case, rec["candidate"], rec["mean_iou"], rec["mean_dice"], rec["saving_vs_two_wall_hull"], rec["max_face_occupancy"], flush=True)
        else:
            print("REJECT", case, rec["best_candidate"], rec["best_metrics"], flush=True)
    manifest = {
        "schema": "anime_two_wall_saving_no_floor_10cm_256_20260709",
        "n_cases": len(records),
        "n_accepted": sum(bool(r.get("accepted")) for r in records),
        "cases": records,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DONE", OUT)


if __name__ == "__main__":
    main()
