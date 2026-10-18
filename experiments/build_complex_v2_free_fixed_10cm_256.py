#!/usr/bin/env python3
"""Evaluate and export complex-v2 candidates from free ∪ fixed branches."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
for p in [ROOT, ROOT / "generation", ROOT / "experiments"]:
    sys.path.insert(0, str(p))

from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from candidate_selection_batch import make_candidates, n6

_spec = importlib.util.spec_from_file_location(
    "export_high_quality_mesh", str(ROOT / "generation/export_high_quality_mesh.py")
)
ehqm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ehqm)

R = 256
PRINT_MM = 100.0
BENCH = ROOT / "data/silhouettes/complex_v2_filled"
OUT = ROOT / "results/complex_v2_free_fixed_10cm_256"
S6 = generate_binary_structure(3, 1)

EXAMPLES = {
    "complex_K_robot_dragon_violin": ["complex_robot", "complex_dragon", "complex_violin"],
    "complex_C_cathedral_bicycle_owl": ["complex_cathedral", "complex_bicycle", "complex_owl_totem"],
    "complex_F_ship_phoenix_robot": ["complex_ship", "complex_phoenix", "complex_robot"],
    "complex_A_dragon_castle_ship": ["complex_dragon", "complex_castle", "complex_ship"],
}


def save_binary_image(mask: np.ndarray, path: Path) -> None:
    img = ((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
    Image.fromarray(img).resize((512, 512), Image.Resampling.NEAREST).save(path)


def mesh_clean_scale(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    parts = mesh.split(only_watertight=False)
    if parts:
        mesh = max(parts, key=lambda x: len(x.faces))
    trimesh.repair.fill_holes(mesh)
    if mesh.volume < 0:
        mesh.invert()
    trimesh.repair.fix_normals(mesh)
    mesh.apply_scale(PRINT_MM / max(float(np.asarray(mesh.extents).max()), 1e-9))
    mesh.export(str(path))
    return mesh


def erosion_survival(volume: np.ndarray) -> float:
    return float(binary_erosion(volume.astype(bool), S6).sum() / max(int(volume.sum()), 1))


def load_masks(names: list[str]) -> list[np.ndarray]:
    paths = [BENCH / f"{name}.png" for name in names]
    sils, _ = load_data_from_list([str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False)
    return [np.asarray(x).squeeze().astype(np.float32) for x in sils]


def eval_branch(masks: list[np.ndarray], branch: str, fix_axis: bool) -> tuple[list[dict], dict]:
    candidates, aux = make_candidates(
        masks,
        R,
        fix_axis,
        print_mm=PRINT_MM,
        nozzle_mm=0.4,
        wall_safety=2.0,
    )
    gt6 = masks + [np.fliplr(m) for m in masks]
    hv = max(int(aux["hull"].sum()), 1)
    rows = []
    for cname, vol in candidates.items():
        vol = vol.astype(bool)
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=512)
        try:
            wt = bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight)
        except Exception:
            wt = False
        rows.append({
            "branch": branch,
            "candidate": cname,
            "fix_axis": bool(fix_axis),
            "mean_iou": float(ev["mean_iou"]),
            "view_ious": [float(x) for x in ev["view_ious"][:3]],
            "saving": float(1.0 - int(vol.sum()) / hv),
            "single": bool(n6(vol) == 1),
            "watertight": wt,
            "erosion_survival": float(erosion_survival(vol)),
            "_vol": vol,
            "_cal": aux["calibration"],
        })
    return rows, aux


def choose(rows: list[dict]) -> dict:
    balanced = [
        r for r in rows
        if r["single"]
        and r["watertight"]
        and r["erosion_survival"] >= 0.15
        and r["mean_iou"] >= 0.98
        and r["saving"] >= 0.40
    ]
    if balanced:
        return max(balanced, key=lambda r: (r["saving"], r["mean_iou"]))
    printable = [r for r in rows if r["single"] and r["watertight"] and r["erosion_survival"] >= 0.15]
    if printable:
        return max(printable, key=lambda r: (r["mean_iou"], r["saving"]))
    return max(rows, key=lambda r: (r["mean_iou"], r["saving"]))


def run_one(example: str, names: list[str]) -> dict:
    t0 = time.time()
    masks = load_masks(names)
    all_rows = []
    aux_by_branch = {}
    for branch, fix in [("fixed", True), ("free", False)]:
        rows, aux = eval_branch(masks, branch, fix)
        all_rows.extend(rows)
        aux_by_branch[branch] = aux
    chosen = choose(all_rows)

    d = OUT / example
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "silhouettes").mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        img = np.array(Image.open(BENCH / f"{name}.png").convert("L"))
        save_binary_image(img > 127, d / "inputs" / f"input{i}_{name}.png")
    ev = evaluate_tfs_volume(
        chosen["_vol"].astype(np.float32),
        masks + [np.fliplr(m) for m in masks],
        chosen["_cal"],
        image_size=512,
    )
    for i, name in enumerate(names):
        save_binary_image(ev["pred_masks"][i], d / "silhouettes" / f"shadow{i}_{name}.png")

    stem = f"{example}_{chosen['branch']}_{chosen['candidate']}"
    npy = d / f"{stem}_voxels256.npy"
    stl = d / f"{stem}_10cm_256.stl"
    np.save(npy, chosen["_vol"].astype(np.uint8))
    ehqm.export_marching_cubes(chosen["_vol"].astype(bool), stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
    mesh = mesh_clean_scale(stl)

    public_rows = []
    for r in all_rows:
        public_rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("_")})
    rec = {
        "example": example,
        "silhouettes": names,
        "selected": {
            "branch": chosen["branch"],
            "candidate": chosen["candidate"],
            "mean_iou": round(float(chosen["mean_iou"]), 4),
            "view_ious": [round(float(x), 4) for x in chosen["view_ious"]],
            "saving": round(float(chosen["saving"]), 4),
            "single": bool(chosen["single"]),
            "watertight": bool(chosen["watertight"]),
            "erosion_survival": round(float(chosen["erosion_survival"]), 4),
            "mesh_extents_mm": [round(float(x), 1) for x in mesh.extents],
            "mesh_faces": int(len(mesh.faces)),
            "stl": str(stl),
            "voxels": str(npy),
        },
        "all_candidates": sorted(public_rows, key=lambda r: (-r["mean_iou"], -r["saving"])),
        "sec": round(time.time() - t0, 1),
    }
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for example, names in EXAMPLES.items():
        rec = run_one(example, names)
        rows.append(rec)
        s = rec["selected"]
        print(
            f"{example}: selected={s['branch']}/{s['candidate']} "
            f"IoU={s['mean_iou']} saving={s['saving']} single={s['single']} watertight={s['watertight']}",
            flush=True,
        )
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump({"schema": "complex_v2_free_fixed_10cm_256", "n_examples": len(rows), "examples": rows}, f, indent=2, ensure_ascii=False)
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
