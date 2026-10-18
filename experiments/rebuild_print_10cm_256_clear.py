#!/usr/bin/env python3
"""Rebuild 10 cm / 256^3 print examples with more recognizable silhouettes.

Keeps the previous delivery constraints: 256^3 generation, 100 mm target size,
scale-aware candidate family, and the pf5 printable candidate.
"""
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
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0
CANDIDATE = "carve_pf5"
BENCH = ROOT / "data/silhouettes/bench3"
OUT = ROOT / "results/print_10cm_256"
S6 = generate_binary_structure(3, 1)

EXAMPLES = {
    "owl_bat_turtle": ["birds_owl", "x_bat", "reptiles_turtle"],
    "eagle_frog_trex": ["birds_eagle", "reptiles_frog", "x_trex"],
    "parrot_dragon_snail": ["birds_parrot", "x_dragon", "sea_snail"],
    "bat_trex_snail": ["x_bat", "x_trex", "sea_snail"],
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
    ext = np.asarray(mesh.extents, dtype=float)
    mesh.apply_scale(PRINT_MM / max(float(ext.max()), 1e-9))
    mesh.export(str(path))
    return mesh


def erosion_survival(volume: np.ndarray) -> float:
    return float(binary_erosion(volume.astype(bool), S6).sum() / max(int(volume.sum()), 1))


def main() -> None:
    if OUT.exists():
        backup = OUT.with_name(f"{OUT.name}_backup_{time.strftime('%Y%m%d_%H%M%S')}")
        shutil.move(str(OUT), str(backup))
        print(f"backup={backup}")
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for example, names in EXAMPLES.items():
        t0 = time.time()
        paths = [BENCH / f"{name}.png" for name in names]
        sils, _ = load_data_from_list(
            [str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False
        )
        masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils]
        candidates, aux = make_candidates(
            masks,
            R,
            True,
            print_mm=PRINT_MM,
            nozzle_mm=NOZZLE_MM,
            wall_safety=WALL_SAFETY,
        )
        if CANDIDATE not in candidates:
            raise RuntimeError(f"{example}: missing {CANDIDATE}; got {sorted(candidates)}")

        hull = aux["hull"].astype(bool)
        vol = candidates[CANDIDATE].astype(bool)
        gt6 = masks + [np.fliplr(m) for m in masks]
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=512)
        iou = float(ev["mean_iou"])
        saving = float(1.0 - int(vol.sum()) / max(int(hull.sum()), 1))
        single = bool(n6(vol) == 1)
        watertight = bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight)

        d = OUT / example
        (d / "inputs").mkdir(parents=True, exist_ok=True)
        (d / "silhouettes").mkdir(parents=True, exist_ok=True)
        for i, (src, name) in enumerate(zip(paths, names)):
            img = np.array(Image.open(src).convert("L"))
            save_binary_image(img > 127, d / "inputs" / f"input{i}_{name}.png")
            save_binary_image(ev["pred_masks"][i], d / "silhouettes" / f"shadow{i}_{name}.png")

        stl = d / f"{example}_10cm_256_pf5.stl"
        np.save(d / f"{example}_voxels256_pf5.npy", vol.astype(np.uint8))
        ehqm.export_marching_cubes(
            vol.astype(bool),
            stl,
            smooth_sigma=0.6,
            taubin_iterations=3,
            upsample_factor=1,
        )
        mesh = mesh_clean_scale(stl)
        rec = {
            "example": example,
            "silhouettes": names,
            "resolution": R,
            "print_mm": PRINT_MM,
            "candidate": CANDIDATE,
            "saving": round(saving, 4),
            "wall_mm_median_proxy": round(erosion_survival(vol), 4),
            "single": single,
            "watertight": watertight,
            "iou": round(iou, 4),
            "mesh_extents_mm": [round(float(x), 1) for x in mesh.extents],
            "mesh_faces": int(len(mesh.faces)),
            "sec": round(time.time() - t0, 1),
        }
        rows.append(rec)
        print(
            f"{example}: IoU={rec['iou']} saving={rec['saving']} "
            f"single={single} watertight={watertight} extents={rec['mesh_extents_mm']}"
        )

    with open(OUT / "BATCH_MANIFEST.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump(
            {
                "schema": "print_10cm_256_clear_v1",
                "n_examples": len(rows),
                "resolution": R,
                "print_mm": PRINT_MM,
                "candidate": CANDIDATE,
                "examples": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
