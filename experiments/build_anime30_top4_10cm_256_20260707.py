#!/usr/bin/env python3
"""Rebuild top screened anime30 silhouette triples with the formal 10cm/256 pipeline."""

from __future__ import annotations

import csv
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
SAVE_MIN = 0.40
SAVE_MAX = 0.70
IOU_MIN = 0.970
EROSION_MIN = 0.15
OUT = ROOT / "results/anime30_top4_10cm_256_20260707"
SCREEN = ROOT / "results/table1_high_saving_replacement_20260707/anime30_screened_candidates.tsv"
S6 = generate_binary_structure(3, 1)


def save_mask(mask: np.ndarray, path: Path) -> None:
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
    vol = np.asarray(volume).astype(bool)
    return float(binary_erosion(vol, S6).sum() / max(int(vol.sum()), 1))


def load_top4() -> list[dict]:
    rows = list(csv.DictReader(SCREEN.open(), delimiter="\t"))
    return rows[:4]


def run_one(idx: int, row: dict) -> dict:
    t0 = time.time()
    paths = row["paths"].split("|")
    names = row["silhouettes"].split(",")
    example = f"anime30_top{idx}_{'_'.join(names)}_{row['mode']}"
    d = OUT / example
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "projections").mkdir(parents=True, exist_ok=True)

    sils, _ = load_data_from_list(paths, True, (R, R), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils]
    candidates, aux = make_candidates(
        masks,
        R,
        True,
        print_mm=PRINT_MM,
        nozzle_mm=NOZZLE_MM,
        wall_safety=WALL_SAFETY,
    )
    gt6 = masks + [np.fliplr(m) for m in masks]
    hull_voxels = max(int(aux["hull"].sum()), 1)
    cand_rows = []
    for cname, vol in candidates.items():
        vol = vol.astype(bool)
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=512)
        try:
            watertight = bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight)
        except Exception:
            watertight = False
        cand_rows.append({
            "candidate": cname,
            "mean_iou": float(ev["mean_iou"]),
            "min_iou": float(min(ev["view_ious"][:3])),
            "view_ious": [float(x) for x in ev["view_ious"][:3]],
            "saving": float(1.0 - int(vol.sum()) / hull_voxels),
            "single": bool(n6(vol) == 1),
            "watertight": watertight,
            "erosion_survival": erosion_survival(vol),
            "_vol": vol,
            "_ev": ev,
        })

    gate = [
        r for r in cand_rows
        if r["mean_iou"] >= IOU_MIN
        and SAVE_MIN <= r["saving"] <= SAVE_MAX
        and r["single"]
        and r["watertight"]
        and r["erosion_survival"] >= EROSION_MIN
    ]
    if gate:
        chosen = max(gate, key=lambda r: (r["saving"], r["mean_iou"]))
    else:
        in_range = [r for r in cand_rows if SAVE_MIN <= r["saving"] <= SAVE_MAX and r["single"] and r["watertight"]]
        chosen = max(in_range or cand_rows, key=lambda r: (r["mean_iou"], r["saving"]))

    accepted = chosen in gate
    for i, (src, name) in enumerate(zip(paths, names)):
        save_mask(np.asarray(Image.open(src).convert("L")) > 127, d / "inputs" / f"input{i}_{name}.png")
        save_mask(chosen["_ev"]["pred_masks"][i], d / "projections" / f"projection{i}_{name}.png")

    stem = f"{example}_{chosen['candidate']}"
    npy = d / f"{stem}_voxels256.npy"
    stl = d / f"{stem}_10cm_256.stl"
    np.save(npy, chosen["_vol"].astype(np.uint8))
    ehqm.export_marching_cubes(chosen["_vol"].astype(bool), stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
    mesh = mesh_clean_scale(stl)
    mesh_parts = len(mesh.split(only_watertight=False))

    public_candidates = []
    for r in cand_rows:
        public_candidates.append({
            "candidate": r["candidate"],
            "mean_iou": round(r["mean_iou"], 6),
            "min_iou": round(r["min_iou"], 6),
            "saving": round(r["saving"], 6),
            "single": r["single"],
            "watertight": r["watertight"],
            "erosion_survival": round(r["erosion_survival"], 6),
        })

    return {
        "example": example,
        "mode": row["mode"],
        "silhouettes": names,
        "input_paths": paths,
        "source_screen": {
            "res": int(row["res"]),
            "mean_iou": float(row["mean_iou"]),
            "min_iou": float(row["min_iou"]),
            "saving": float(row["saving"]),
            "status": row["status"],
        },
        "resolution": R,
        "print_mm": PRINT_MM,
        "fixed": True,
        "accepted": accepted,
        "gate": {
            "iou_min": IOU_MIN,
            "saving_range": [SAVE_MIN, SAVE_MAX],
            "single": True,
            "watertight": True,
            "erosion_survival_min": EROSION_MIN,
        },
        "selected": {
            "candidate": chosen["candidate"],
            "mean_iou": round(chosen["mean_iou"], 6),
            "min_iou": round(chosen["min_iou"], 6),
            "view_ious": [round(float(x), 6) for x in chosen["view_ious"]],
            "saving": round(chosen["saving"], 6),
            "single": chosen["single"],
            "watertight": chosen["watertight"],
            "mesh_parts": mesh_parts,
            "mesh_watertight": bool(mesh.is_watertight),
            "erosion_survival": round(chosen["erosion_survival"], 6),
            "mesh_extents_mm": [round(float(x), 3) for x in mesh.extents],
            "mesh_faces": int(len(mesh.faces)),
            "stl": str(stl),
            "voxels": str(npy),
        },
        "all_candidates": sorted(public_candidates, key=lambda r: (-r["mean_iou"], -r["saving"])),
        "sec": round(time.time() - t0, 1),
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(load_top4(), start=1):
        rec = run_one(i, row)
        records.append(rec)
        sel = rec["selected"]
        print(
            f"{rec['example']}: accepted={rec['accepted']} "
            f"{sel['candidate']} IoU={sel['mean_iou']} min={sel['min_iou']} "
            f"saving={sel['saving']} single={sel['single']} watertight={sel['watertight']} "
            f"parts={sel['mesh_parts']}",
            flush=True,
        )
        (OUT / "MANIFEST.partial.json").write_text(json.dumps(records, indent=2))
    summary = {
        "schema": "anime30_top4_10cm_256_20260707",
        "source_screen": str(SCREEN),
        "target": {
            "resolution": R,
            "print_mm": PRINT_MM,
            "saving_range": [SAVE_MIN, SAVE_MAX],
            "iou_min": IOU_MIN,
        },
        "n_examples": len(records),
        "accepted": sum(1 for r in records if r["accepted"]),
        "examples": records,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, indent=2))
    print("DONE", OUT)


if __name__ == "__main__":
    main()
