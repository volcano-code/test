#!/usr/bin/env python3
"""Build diverse anime30 triples with strict fixed-camera 10cm/256 export."""

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
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, generate_binary_structure

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
for p in [ROOT, ROOT / "generation", ROOT / "experiments"]:
    sys.path.insert(0, str(p))

from candidate_selection_batch import make_candidates, n6
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh

_spec = importlib.util.spec_from_file_location(
    "export_high_quality_mesh", str(ROOT / "generation/export_high_quality_mesh.py")
)
ehqm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ehqm)

R = 256
PRINT_MM = 100.0
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0
IOU_MIN = 0.970
SAVE_MIN = 0.30
SAVE_MAX = 0.70
EROSION_MIN = 0.15
OUT = ROOT / "results/anime30_diverse_fixed_10cm_256_20260707"
MASK_ROOT = ROOT / "results/anime30_adjusted_full_screen_20260704/adjusted_masks"
S6 = generate_binary_structure(3, 1)

EXAMPLES = [
    ("diverse1_close1_05_08_13", "close1", ["anime30_05", "anime30_08", "anime30_13"]),
    ("diverse2_thick1_07_14_15", "thick1", ["anime30_07", "anime30_14", "anime30_15"]),
    ("diverse3_thick1_09_28_29", "thick1", ["anime30_09", "anime30_28", "anime30_29"]),
    ("diverse4_close2_08_14_30", "close2", ["anime30_08", "anime30_14", "anime30_30"]),
    ("diverse5_thick1_13_21_23", "thick1", ["anime30_13", "anime30_21", "anime30_23"]),
    ("diverse6_close1_05_07_20", "close1", ["anime30_05", "anime30_07", "anime30_20"]),
]


def path_for(mode: str, name: str) -> str:
    return str(MASK_ROOT / mode / f"{name}_{mode}.png")


def save_mask(mask: np.ndarray, path: Path) -> None:
    img = ((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
    Image.fromarray(img).resize((512, 512), Image.Resampling.NEAREST).save(path)


def overlay_error(target: np.ndarray, pred: np.ndarray, path: Path) -> None:
    target = np.asarray(target) > 0.5
    pred = np.asarray(pred) > 0.5
    h, w = target.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    rgb[target & pred] = 0
    rgb[target & ~pred] = (230, 80, 40)
    rgb[~target & pred] = (40, 90, 230)
    Image.fromarray(rgb).resize((512, 512), Image.Resampling.NEAREST).save(path)


def contact_sheet(record: dict, masks: list[np.ndarray], pred_masks: list[np.ndarray], out: Path) -> None:
    cell = 180
    gap = 24
    w = 3 * cell + 4 * gap
    h = 2 * cell + 3 * gap + 42
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 8), record["example"], fill=(0, 0, 0))
    for i in range(3):
        target = Image.fromarray((masks[i] > 0.5).astype(np.uint8) * 255).resize((cell, cell), Image.Resampling.NEAREST)
        pred = Image.fromarray((pred_masks[i] > 0.5).astype(np.uint8) * 255).resize((cell, cell), Image.Resampling.NEAREST)
        x = gap + i * (cell + gap)
        canvas.paste(target.convert("RGB"), (x, gap + 28))
        canvas.paste(pred.convert("RGB"), (x, 2 * gap + 28 + cell))
        draw.rectangle((x, gap + 28, x + cell, gap + 28 + cell), outline=(130, 130, 130))
        draw.rectangle((x, 2 * gap + 28 + cell, x + cell, 2 * gap + 28 + 2 * cell), outline=(130, 130, 130))
        draw.text((x, gap + 10), record["silhouettes"][i], fill=(0, 0, 0))
    canvas.save(out)


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


def run_one(example: str, mode: str, names: list[str]) -> dict:
    t0 = time.time()
    paths = [path_for(mode, name) for name in names]
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
    rows = []
    for cname, vol in candidates.items():
        vol = vol.astype(bool)
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=512)
        try:
            watertight = bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight)
        except Exception:
            watertight = False
        rows.append({
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
        r for r in rows
        if r["mean_iou"] >= IOU_MIN
        and SAVE_MIN <= r["saving"] <= SAVE_MAX
        and r["single"]
        and r["watertight"]
        and r["erosion_survival"] >= EROSION_MIN
    ]
    if gate:
        # Display-oriented fixed export: prioritize fidelity and wall survival,
        # then prefer moderate saving instead of maximum material removal.
        chosen = max(gate, key=lambda r: (r["mean_iou"], r["erosion_survival"], -abs(r["saving"] - 0.45)))
    else:
        chosen = max(rows, key=lambda r: (r["mean_iou"], r["single"], r["watertight"], r["erosion_survival"]))

    for i, (src, name) in enumerate(zip(paths, names)):
        target = np.asarray(Image.open(src).convert("L")) > 127
        save_mask(target, d / "inputs" / f"input{i}_{name}.png")
        save_mask(chosen["_ev"]["pred_masks"][i], d / "projections" / f"projection{i}_{name}.png")
        overlay_error(target, chosen["_ev"]["pred_masks"][i], d / "projections" / f"overlay{i}_{name}.png")

    stem = f"{example}_{chosen['candidate']}"
    npy = d / f"{stem}_voxels256.npy"
    stl = d / f"{stem}_10cm_256.stl"
    np.save(npy, chosen["_vol"].astype(np.uint8))
    ehqm.export_marching_cubes(chosen["_vol"].astype(bool), stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
    mesh = mesh_clean_scale(stl)
    parts = len(mesh.split(only_watertight=False))

    record = {
        "example": example,
        "mode": mode,
        "silhouettes": names,
        "input_paths": paths,
        "resolution": R,
        "print_mm": PRINT_MM,
        "fixed": True,
        "accepted": bool(chosen in gate),
        "selection_policy": "fixed_only_display_oriented_iou_wall_survival_then_moderate_saving",
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
            "mesh_parts": parts,
            "mesh_watertight": bool(mesh.is_watertight),
            "erosion_survival": round(chosen["erosion_survival"], 6),
            "mesh_extents_mm": [round(float(x), 3) for x in mesh.extents],
            "mesh_faces": int(len(mesh.faces)),
            "stl": str(stl),
            "voxels": str(npy),
        },
        "all_candidates": [
            {
                "candidate": r["candidate"],
                "mean_iou": round(r["mean_iou"], 6),
                "min_iou": round(r["min_iou"], 6),
                "saving": round(r["saving"], 6),
                "single": r["single"],
                "watertight": r["watertight"],
                "erosion_survival": round(r["erosion_survival"], 6),
            }
            for r in sorted(rows, key=lambda x: (-x["mean_iou"], -x["erosion_survival"], x["saving"]))
        ],
        "sec": round(time.time() - t0, 1),
    }
    contact_sheet(record, masks, chosen["_ev"]["pred_masks"], d / f"{example}_target_vs_projection.png")
    return record


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for example, mode, names in EXAMPLES:
        rec = run_one(example, mode, names)
        records.append(rec)
        sel = rec["selected"]
        print(
            f"{example}: accepted={rec['accepted']} {sel['candidate']} "
            f"IoU={sel['mean_iou']} min={sel['min_iou']} saving={sel['saving']} "
            f"single={sel['single']} watertight={sel['watertight']} erosion={sel['erosion_survival']}",
            flush=True,
        )
        (OUT / "MANIFEST.partial.json").write_text(json.dumps(records, indent=2))
    summary = {
        "schema": "anime30_diverse_fixed_10cm_256_v1",
        "target": "strict fixed-camera physical-print candidates with higher silhouette diversity",
        "n_examples": len(records),
        "accepted": sum(1 for r in records if r["accepted"]),
        "examples": records,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, indent=2))
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
