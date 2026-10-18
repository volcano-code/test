#!/usr/bin/env python3
"""Build anime physical examples for the two-wall fixed no-floor setting.

This keeps the original three-view fixed wall-corner convention but removes the
floor view:
  view1 -> x-axis projection -> left wall,  mask layout [Y, Z]
  view2 -> y-axis projection -> right wall, mask layout [X, Z]
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
BASE_SCRIPT = ROOT / "experiments/build_anime_n2_fixed_display_10cm_256_20260709.py"
OUT = ROOT / "results/anime_two_wall_fixed_no_floor_10cm_256_20260709"
SRC = Path("/home/lin/下载/original_anime_silhouettes_30 (4)")

spec = importlib.util.spec_from_file_location("n2base", str(BASE_SCRIPT))
n2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(n2)


PAIRS = [
    ("umbrella_miko_left__archer_right", "012_优雅的伞下武士.png", "009_神圣的弓箭手剪影.png"),
    ("fox_miko_left__violinist_right", "028_狐耳巫女的舞动身影.png", "027_哥特风格的小提琴手.png"),
    ("dark_knight_left__magic_scholar_right", "025_黑暗骑士的孤独身影.png", "019_魔法学者的黑色剪影.png"),
    ("future_mecha_left__skater_right", "022_未来机械战士的身影.png", "021_滑板少女的动态剪影.png"),
    ("dark_mage_left__samurai_right", "007_黑暗魔法使者的身影.png", "003_剑士剪影与风之衣饰.png"),
]


def visual_hull_two_wall(left_yz: np.ndarray, right_xz: np.ndarray) -> np.ndarray:
    return left_yz[None, :, :] & right_xz[:, None, :]


def project_left_wall_x(vol: np.ndarray) -> np.ndarray:
    return vol.any(axis=0)


def project_right_wall_y(vol: np.ndarray) -> np.ndarray:
    return vol.any(axis=1)


def keep_largest(vol: np.ndarray) -> np.ndarray:
    return n2.keep_largest(vol)


def smooth_candidate_two_wall(hull: np.ndarray, mode: str) -> np.ndarray:
    if mode == "hull":
        return keep_largest(hull)
    if mode == "center_depth_band":
        coords = np.indices(hull.shape)
        z = coords[2].astype(np.float32)
        center = (hull.shape[2] - 1) * 0.5
        band = np.abs(z - center) <= hull.shape[2] * 0.40
        vol = hull & band
    elif mode == "core_shell":
        er = n2.binary_erosion(hull, n2.S6, iterations=2)
        coords = np.indices(hull.shape)
        z = coords[2].astype(np.float32)
        center = (hull.shape[2] - 1) * 0.5
        band = np.abs(z - center) <= hull.shape[2] * 0.30
        vol = er | (hull & band)
    else:
        raise ValueError(mode)

    vol = keep_largest(vol)
    miss_l = np.argwhere(project_left_wall_x(hull) & ~project_left_wall_x(vol))
    for y, z in miss_l:
        xs = np.where(hull[:, y, z])[0]
        if xs.size:
            vol[int(xs[len(xs) // 2]), y, z] = True
    miss_r = np.argwhere(project_right_wall_y(hull) & ~project_right_wall_y(vol))
    for x, z in miss_r:
        ys = np.where(hull[x, :, z])[0]
        if ys.size:
            vol[x, int(ys[len(ys) // 2]), z] = True
    return keep_largest(vol)


def metrics_two_wall(vol: np.ndarray, left: np.ndarray, right: np.ndarray, hull_vox: int) -> dict:
    pl = project_left_wall_x(vol)
    pr = project_right_wall_y(vol)
    m = {
        "view_ious": [n2.iou(pl, left), n2.iou(pr, right)],
        "view_dices": [n2.dice(pl, left), n2.dice(pr, right)],
        "saving": float(1.0 - int(vol.sum()) / max(int(hull_vox), 1)),
        "components": n2.component_count(vol),
        "erosion_survival": n2.erosion_survival(vol),
        **n2.face_occupancy(vol),
    }
    m["mean_iou"] = float(np.mean(m["view_ious"]))
    m["min_iou"] = float(np.min(m["view_ious"]))
    m["mean_dice"] = float(np.mean(m["view_dices"]))
    return m


def contact_sheet(name: str, left: np.ndarray, right: np.ndarray, pl: np.ndarray, pr: np.ndarray, out: Path) -> None:
    return n2.contact_sheet(name, left, right, pl, pr, out)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    rejected = []
    for name, left_name, right_name in PAIRS:
        left_path = SRC / left_name
        right_path = SRC / right_name
        left = n2.load_mask(left_path, n2.RES_EXPORT)
        right = n2.load_mask(right_path, n2.RES_EXPORT)
        hull = visual_hull_two_wall(left, right)
        best = None
        for mode in ("hull", "center_depth_band", "core_shell"):
            vol = smooth_candidate_two_wall(hull, mode)
            m = metrics_two_wall(vol, left, right, int(hull.sum()))
            ok = (
                m["mean_iou"] >= 0.95
                and m["min_iou"] >= 0.90
                and m["components"] == 1
                and m["erosion_survival"] >= 0.08
                and m["max_face_occupancy"] <= 0.18
            )
            score = (ok, m["mean_iou"], -m["max_face_occupancy"], m["erosion_survival"], -abs(m["saving"] - 0.10))
            if best is None or score > best[0]:
                best = (score, mode, m, vol)
        assert best is not None
        score, mode, m, vol = best
        if not score[0]:
            rejected.append({
                "case": name,
                "reason": "failed fixed two-wall no-floor gates",
                "candidate_mode": mode,
                "metrics": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items() if k != "view_dices"},
            })
            print("REJECT", name, mode, round(m["mean_iou"], 6), round(m["max_face_occupancy"], 6), flush=True)
            continue

        d = OUT / name
        (d / "inputs").mkdir(parents=True)
        (d / "projections").mkdir(parents=True)
        stl = d / f"{name}_10cm_256.stl"
        npy = d / f"{name}_voxels256.npy"
        base = d / "display_base_110x76mm.stl"
        mesh = n2.export_stl(vol, stl)
        n2.export_base(base)
        np.save(npy, vol.astype(np.uint8))
        pl = project_left_wall_x(vol)
        pr = project_right_wall_y(vol)
        n2.save_mask(left, d / "inputs/left_wall_x_view1_target.png")
        n2.save_mask(right, d / "inputs/right_wall_y_view2_target.png")
        n2.save_mask(pl, d / "projections/left_wall_x_view1_projection.png")
        n2.save_mask(pr, d / "projections/right_wall_y_view2_projection.png")
        contact_sheet(name, left, right, pl, pr, d / "input_vs_projection.png")
        rec = {
            "case": name,
            "setting": "two-wall fixed no-floor",
            "projection_model": "original fixed TFS without floor: view1->x/left wall, view2->y/right wall",
            "input_paths": [str(left_path), str(right_path)],
            "candidate_mode": mode,
            "mean_iou": round(m["mean_iou"], 6),
            "min_iou": round(m["min_iou"], 6),
            "mean_dice": round(m["mean_dice"], 6),
            "view_ious": [round(float(x), 6) for x in m["view_ious"]],
            "view_dices": [round(float(x), 6) for x in m["view_dices"]],
            "saving_vs_two_wall_hull": round(m["saving"], 6),
            "components_6conn": int(m["components"]),
            "mesh_parts": int(len(mesh.split(only_watertight=False))),
            "mesh_watertight": bool(mesh.is_watertight),
            "erosion_survival": round(m["erosion_survival"], 6),
            "max_face_occupancy": round(m["max_face_occupancy"], 6),
            "bbox_fill": round(m["bbox_fill"], 6),
            "stl": str(stl),
            "voxels": str(npy),
            "display_base_stl": str(base),
            "preview": str(d / "input_vs_projection.png"),
        }
        records.append(rec)
        (d / "CASE_MANIFEST.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("OK", name, rec["mean_iou"], rec["mean_dice"], rec["saving_vs_two_wall_hull"], rec["max_face_occupancy"], flush=True)

    manifest = {
        "schema": "anime_two_wall_fixed_no_floor_10cm_256_20260709",
        "source": str(SRC),
        "n_exported": len(records),
        "n_rejected": len(rejected),
        "cases": records,
        "rejected": rejected,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DONE", OUT, "n=", len(records), "rejected=", len(rejected))


if __name__ == "__main__":
    main()
