#!/usr/bin/env python3
"""Build curated representative fixed-gated physical experiment models.

Set contains:
  - four AAAI Fig.1 n=2-inspired models (A+star, A+heart, A+gear, I+tree)
    rebuilt as current three-view fixed TFS inputs with a support silhouette;
  - Chinese character examples;
  - one complex symbolic example.

Gate: fixed branch, IoU >= 0.975, single body, watertight,
erosion survival >= 0.15, and material saving in [40%, 85%].
"""
from __future__ import annotations

import importlib.util
import json
import shutil
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
IOU_MIN = 0.975
SAVE_MIN = 0.40
SAVE_MAX = 0.85
EROSION_MIN = 0.15
SYM = ROOT / "data/silhouettes/curated_symbolic_v1"
CPLX = ROOT / "data/silhouettes/complex_v2_filled"
OUT = ROOT / "results/curated_symbolic_fixed_10cm_256"
S6 = generate_binary_structure(3, 1)

EXAMPLES = {
    "fig1_A_star": ["aaai_A", "aaai_star", "support_pentagon"],
    "fig1_A_heart": ["aaai_A", "aaai_heart", "support_round_plaque"],
    "fig1_A_gear": ["support_disc", "aaai_A", "aaai_gear"],
    "fig1_I_tree": ["aaai_I", "aaai_tree", "support_tall_plaque"],
    "han_long_dragon": ["support_tall_plaque", "han_long_dragon", "support_tall_plaque"],
    "han_ying_shadow_candidate": ["han_ying_shadow", "support_disc", "support_tall_plaque"],
    "complex_robot_dragon_violin": ["complex_robot", "complex_dragon", "complex_violin"],
}


def path_for(name: str) -> Path:
    p = SYM / f"{name}.png"
    if p.exists():
        return p
    return CPLX / f"{name}.png"


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


def load_masks(names: list[str]) -> tuple[list[np.ndarray], list[Path]]:
    paths = [path_for(n) for n in names]
    sils, _ = load_data_from_list([str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False)
    return [np.asarray(x).squeeze().astype(np.float32) for x in sils], paths


def run_one(example: str, names: list[str]) -> dict:
    t0 = time.time()
    masks, paths = load_masks(names)
    candidates, aux = make_candidates(
        masks,
        R,
        True,
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
            "candidate": cname,
            "iou": float(ev["mean_iou"]),
            "view_ious": [float(x) for x in ev["view_ious"][:3]],
            "saving": float(1.0 - int(vol.sum()) / hv),
            "single": bool(n6(vol) == 1),
            "watertight": wt,
            "erosion_survival": float(erosion_survival(vol)),
            "_vol": vol,
            "_ev": ev,
        })
    gate = [
        r for r in rows
        if r["iou"] >= IOU_MIN
        and r["single"]
        and r["watertight"]
        and r["erosion_survival"] >= EROSION_MIN
        and SAVE_MIN <= r["saving"] <= SAVE_MAX
    ]
    if gate:
        pool = gate
        chosen = max(pool, key=lambda r: (r["saving"], r["iou"]))
    else:
        chosen = max(rows, key=lambda r: (r["iou"], r["saving"]))

    accepted = bool(chosen in gate)
    d = OUT / ("accepted" if accepted else "rejected") / example
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "silhouettes").mkdir(parents=True, exist_ok=True)
    for i, (src, name) in enumerate(zip(paths, names)):
        img = np.array(Image.open(src).convert("L"))
        save_binary_image(img > 127, d / "inputs" / f"input{i}_{name}.png")
        save_binary_image(chosen["_ev"]["pred_masks"][i], d / "silhouettes" / f"shadow{i}_{name}.png")

    stl = None
    npy = None
    mesh_info = {}
    if accepted:
        stem = f"{example}_FIXED_{chosen['candidate']}"
        npy = d / f"{stem}_voxels256.npy"
        stl = d / f"{stem}_10cm_256.stl"
        np.save(npy, chosen["_vol"].astype(np.uint8))
        ehqm.export_marching_cubes(chosen["_vol"].astype(bool), stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
        mesh = mesh_clean_scale(stl)
        mesh_info = {
            "mesh_extents_mm": [round(float(x), 1) for x in mesh.extents],
            "mesh_faces": int(len(mesh.faces)),
            "stl": str(stl),
            "voxels": str(npy),
        }

    public_rows = []
    for r in rows:
        public_rows.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("_")})
    return {
        "example": example,
        "silhouettes": names,
        "selected_branch": "fixed",
        "accepted": accepted,
        "gate": {
            "iou_min": IOU_MIN,
            "single": True,
            "watertight": True,
            "erosion_survival_min": EROSION_MIN,
            "saving_min": SAVE_MIN,
            "saving_max": SAVE_MAX,
        },
        "selected": {
            "candidate": chosen["candidate"],
            "iou": round(float(chosen["iou"]), 4),
            "view_ious": [round(float(x), 4) for x in chosen["view_ious"]],
            "saving": round(float(chosen["saving"]), 4),
            "single": bool(chosen["single"]),
            "watertight": bool(chosen["watertight"]),
            "erosion_survival": round(float(chosen["erosion_survival"]), 4),
            **mesh_info,
        },
        "all_candidates": sorted(public_rows, key=lambda r: (-r["iou"], -r["saving"])),
        "sec": round(time.time() - t0, 1),
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for example, names in EXAMPLES.items():
        rec = run_one(example, names)
        rows.append(rec)
        s = rec["selected"]
        print(
            f"{example}: accepted={rec['accepted']} fixed/{s['candidate']} "
            f"IoU={s['iou']} saving={s['saving']} single={s['single']} watertight={s['watertight']}",
            flush=True,
        )
    accepted = [r for r in rows if r["accepted"]]
    rejected = [r for r in rows if not r["accepted"]]
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump({
            "schema": "curated_symbolic_fixed_10cm_256_v1",
            "n_examples": len(rows),
            "n_accepted": len(accepted),
            "n_rejected": len(rejected),
            "examples": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
