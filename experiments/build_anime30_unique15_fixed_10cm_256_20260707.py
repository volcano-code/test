#!/usr/bin/env python3
"""Build five strict-fixed anime30 examples with 15 unique input silhouettes.

Each example saves six input silhouettes and six rendered/generated projections
(three primary views plus horizontal mirror views).
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
assert _spec.loader is not None
_spec.loader.exec_module(ehqm)

R = 256
PRINT_MM = 100.0
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0
IOU_MIN = 0.970
SAVE_MIN = 0.35
SAVE_MAX = 0.70
EROSION_MIN = 0.15
OUT = ROOT / "results/anime30_unique15_fixed_10cm_256_20260707"
MASK_ROOT = ROOT / "results/anime30_adjusted_full_screen_20260704/adjusted_masks/thick1"
S6 = generate_binary_structure(3, 1)

EXAMPLES = [
    ("unique1_08_15_28", ["anime30_08", "anime30_15", "anime30_28"]),
    ("unique2_13_14_27", ["anime30_13", "anime30_14", "anime30_27"]),
    ("unique3_05_07_20", ["anime30_05", "anime30_07", "anime30_20"]),
    ("unique4_01_09_22", ["anime30_01", "anime30_09", "anime30_22"]),
    ("unique5_02_16_30", ["anime30_02", "anime30_16", "anime30_30"]),
]


def path_for(name: str) -> str:
    return str(MASK_ROOT / f"{name}_thick1.png")


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


def contact_sheet(example: str, inputs6: list[np.ndarray], preds6: list[np.ndarray], out: Path) -> None:
    cell = 130
    gap = 18
    w = 6 * cell + 7 * gap
    h = 2 * cell + 4 * gap + 24
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), example, fill=(0, 0, 0))
    for i in range(6):
        x = gap + i * (cell + gap)
        y1 = gap + 24
        y2 = y1 + cell + gap
        inp = Image.fromarray((np.asarray(inputs6[i]) > 0.5).astype(np.uint8) * 255).resize((cell, cell), Image.Resampling.NEAREST)
        prd = Image.fromarray((np.asarray(preds6[i]) > 0.5).astype(np.uint8) * 255).resize((cell, cell), Image.Resampling.NEAREST)
        canvas.paste(inp.convert("RGB"), (x, y1))
        canvas.paste(prd.convert("RGB"), (x, y2))
        draw.rectangle((x, y1, x + cell, y1 + cell), outline=(120, 120, 120))
        draw.rectangle((x, y2, x + cell, y2 + cell), outline=(120, 120, 120))
    canvas.save(out)


def run_one(example: str, names: list[str]) -> dict:
    t0 = time.time()
    paths = [path_for(n) for n in names]
    d = OUT / example
    (d / "inputs6").mkdir(parents=True, exist_ok=True)
    (d / "projections6").mkdir(parents=True, exist_ok=True)

    sils, _ = load_data_from_list(paths, True, (R, R), torch.device("cpu"), 3, False)
    inputs6 = [np.asarray(x).squeeze().astype(np.float32) for x in sils[:6]]
    masks = inputs6[:3]
    cands, aux = make_candidates(masks, R, True, print_mm=PRINT_MM, nozzle_mm=NOZZLE_MM, wall_safety=WALL_SAFETY)
    gt6 = masks + [np.fliplr(m) for m in masks]
    hull_vox = max(int(aux["hull"].sum()), 1)

    rows = []
    for cname, vol in cands.items():
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
            "view_ious6": [float(x) for x in ev["view_ious"][:6]],
            "saving": float(1.0 - int(vol.sum()) / hull_vox),
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
        chosen = max(gate, key=lambda r: (r["mean_iou"], r["erosion_survival"], -abs(r["saving"] - 0.52)))
    else:
        chosen = max(rows, key=lambda r: (r["mean_iou"], r["single"], r["watertight"], r["erosion_survival"]))

    pred3 = [np.asarray(x) for x in chosen["_ev"]["pred_masks"][:3]]
    preds6 = pred3 + [np.fliplr(x) for x in pred3]
    for i, arr in enumerate(inputs6):
        save_mask(arr, d / "inputs6" / f"input_view{i}.png")
    for i, arr in enumerate(preds6):
        save_mask(arr, d / "projections6" / f"projection_view{i}.png")
    contact_sheet(example, inputs6, preds6, d / f"{example}_six_input_vs_projection.png")

    stem = f"{example}_{chosen['candidate']}"
    npy = d / f"{stem}_voxels256.npy"
    stl = d / f"{stem}_10cm_256.stl"
    np.save(npy, chosen["_vol"].astype(np.uint8))
    ehqm.export_marching_cubes(chosen["_vol"].astype(bool), stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
    mesh = mesh_clean_scale(stl)

    return {
        "example": example,
        "silhouettes": names,
        "input_paths": paths,
        "resolution": R,
        "print_mm": PRINT_MM,
        "fixed": True,
        "accepted": bool(chosen in gate),
        "selected": {
            "candidate": chosen["candidate"],
            "mean_iou": round(chosen["mean_iou"], 6),
            "min_iou": round(chosen["min_iou"], 6),
            "view_ious6": [round(float(x), 6) for x in chosen["view_ious6"]],
            "saving": round(chosen["saving"], 6),
            "single": chosen["single"],
            "watertight": chosen["watertight"],
            "mesh_watertight": bool(mesh.is_watertight),
            "mesh_parts": int(len(mesh.split(only_watertight=False))),
            "erosion_survival": round(chosen["erosion_survival"], 6),
            "stl": str(stl),
            "voxels": str(npy),
            "six_view_preview": str(d / f"{example}_six_input_vs_projection.png"),
        },
        "all_candidates": [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("_")}
            for r in rows
        ],
        "sec": round(time.time() - t0, 1),
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for example, names in EXAMPLES:
        rec = run_one(example, names)
        records.append(rec)
        s = rec["selected"]
        print(f"{example}: accepted={rec['accepted']} {s['candidate']} IoU={s['mean_iou']} saving={s['saving']} single={s['single']} watertight={s['watertight']}", flush=True)
        (OUT / "MANIFEST.partial.json").write_text(json.dumps(records, indent=2))
    used = [n for _, names in EXAMPLES for n in names]
    summary = {
        "schema": "anime30_unique15_fixed_10cm_256_v1",
        "unique_input_count": len(set(used)),
        "all_inputs_unique": len(set(used)) == len(used),
        "n_examples": len(records),
        "accepted": sum(1 for r in records if r["accepted"]),
        "examples": records,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, indent=2))
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
