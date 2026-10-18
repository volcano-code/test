#!/usr/bin/env python3
"""Export top complex-v2 silhouette candidates at 256^3 / 10 cm."""
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
OUT = ROOT / "results/complex_v2_top4_10cm_256"
CANDIDATE = "carve_pf5"
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


def run_one(example: str, names: list[str]) -> dict:
    t0 = time.time()
    paths = [BENCH / f"{name}.png" for name in names]
    sils, _ = load_data_from_list([str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils]
    candidates, aux = make_candidates(
        masks,
        R,
        True,
        print_mm=PRINT_MM,
        nozzle_mm=0.4,
        wall_safety=2.0,
    )
    hull = aux["hull"].astype(bool)
    vol = candidates[CANDIDATE].astype(bool)
    gt6 = masks + [np.fliplr(m) for m in masks]
    ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=512)
    d = OUT / example
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "silhouettes").mkdir(parents=True, exist_ok=True)
    for i, (src, name) in enumerate(zip(paths, names)):
        img = np.array(Image.open(src).convert("L"))
        save_binary_image(img > 127, d / "inputs" / f"input{i}_{name}.png")
        save_binary_image(ev["pred_masks"][i], d / "silhouettes" / f"shadow{i}_{name}.png")
    npy = d / f"{example}_voxels256_pf5.npy"
    stl = d / f"{example}_10cm_256_pf5.stl"
    np.save(npy, vol.astype(np.uint8))
    ehqm.export_marching_cubes(vol.astype(bool), stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
    mesh = mesh_clean_scale(stl)
    return {
        "example": example,
        "silhouettes": names,
        "resolution": R,
        "print_mm": PRINT_MM,
        "candidate": CANDIDATE,
        "iou": round(float(ev["mean_iou"]), 4),
        "view_ious": [round(float(x), 4) for x in ev["view_ious"][:3]],
        "saving": round(float(1.0 - int(vol.sum()) / max(int(hull.sum()), 1)), 4),
        "single": bool(n6(vol) == 1),
        "watertight": bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight),
        "wall_mm_median_proxy": round(erosion_survival(vol), 4),
        "mesh_extents_mm": [round(float(x), 1) for x in mesh.extents],
        "mesh_faces": int(len(mesh.faces)),
        "stl": str(stl),
        "voxels": str(npy),
        "sec": round(time.time() - t0, 1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for example, names in EXAMPLES.items():
        rec = run_one(example, names)
        rows.append(rec)
        print(f"{example}: IoU={rec['iou']} saving={rec['saving']} single={rec['single']} watertight={rec['watertight']}")
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump({"schema": "complex_v2_top4_10cm_256", "n_examples": len(rows), "examples": rows}, f, indent=2, ensure_ascii=False)
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
