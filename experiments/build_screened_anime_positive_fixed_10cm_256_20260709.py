#!/usr/bin/env python3
"""Build fixed 10cm/256 print candidates from screened anime silhouettes."""
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
SCREEN = ROOT / "results/anime_structure_friendly_screen_20260709/selected"
OUT = ROOT / "results/screened_anime_positive_fixed_10cm_256_20260709"
S6 = generate_binary_structure(3, 1)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(((np.asarray(mask) > 0.5).astype(np.uint8) * 255)).resize(
        (512, 512), Image.Resampling.NEAREST
    ).save(path)


def face_occupancy(volume: np.ndarray) -> dict:
    v = np.asarray(volume, dtype=bool)
    coords = np.argwhere(v)
    if coords.size == 0:
        return {"bbox_fill": 0.0, "max_face_occupancy": 1.0, "mean_face_occupancy": 1.0}
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    sub = v[mins[0] : maxs[0] + 1, mins[1] : maxs[1] + 1, mins[2] : maxs[2] + 1]
    faces = [
        float(sub[0].mean()),
        float(sub[-1].mean()),
        float(sub[:, 0, :].mean()),
        float(sub[:, -1, :].mean()),
        float(sub[:, :, 0].mean()),
        float(sub[:, :, -1].mean()),
    ]
    return {
        "bbox_fill": float(sub.mean()),
        "max_face_occupancy": float(max(faces)),
        "mean_face_occupancy": float(np.mean(faces)),
    }


def erosion_survival(volume: np.ndarray) -> float:
    return float(binary_erosion(volume.astype(bool), S6).sum() / max(int(volume.sum()), 1))


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


def contact_sheet(title: str, inputs: list[np.ndarray], preds: list[np.ndarray], out: Path) -> None:
    cell = 118
    gap = 14
    canvas = Image.new("RGB", (6 * (cell + gap) + gap, 2 * cell + 3 * gap + 28), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), title, fill=(0, 0, 0))
    for i in range(6):
        x = gap + i * (cell + gap)
        y0 = gap + 28
        y1 = y0 + cell + gap
        for y, arr in [(y0, inputs[i]), (y1, preds[i])]:
            img = Image.fromarray(((np.asarray(arr) > 0.5).astype(np.uint8) * 255)).resize(
                (cell, cell), Image.Resampling.NEAREST
            )
            canvas.paste(img.convert("RGB"), (x, y))
            draw.rectangle((x, y, x + cell, y + cell), outline=(120, 120, 120))
    canvas.save(out)


def selected_paths() -> list[Path]:
    return sorted(SCREEN.glob("*.png"))


def run_one(idx: int, paths: list[Path]) -> dict:
    t0 = time.time()
    name = f"screened_anime_{idx:02d}"
    d = OUT / name
    (d / "inputs6").mkdir(parents=True, exist_ok=True)
    (d / "projections6").mkdir(parents=True, exist_ok=True)
    sils, _ = load_data_from_list([str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False)
    base_masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils[:3]]
    gt6 = base_masks + [np.fliplr(m) for m in base_masks]
    cands, aux = make_candidates(base_masks, R, True, print_mm=PRINT_MM, nozzle_mm=NOZZLE_MM, wall_safety=WALL_SAFETY)
    hull_vox = max(int(aux["hull"].sum()), 1)
    rows = []
    for cname, vol0 in cands.items():
        vol = np.asarray(vol0, dtype=bool)
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=512)
        try:
            watertight = bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight)
        except Exception:
            watertight = False
        face = face_occupancy(vol)
        rows.append({
            "candidate": cname,
            "mean_iou": float(ev["mean_iou"]),
            "min_iou": float(min(ev["view_ious"][:6])),
            "mean_dice": float(ev["mean_dice"]),
            "saving": float(1.0 - int(vol.sum()) / hull_vox),
            "single": bool(n6(vol) == 1),
            "watertight": watertight,
            "erosion_survival": erosion_survival(vol),
            **face,
            "_vol": vol,
            "_ev": ev,
        })
    eligible = [
        r for r in rows
        if r["mean_iou"] >= 0.975
        and r["min_iou"] >= 0.95
        and 0.30 <= r["saving"] <= 0.70
        and r["single"]
        and r["watertight"]
        and r["erosion_survival"] >= 0.12
        and r["max_face_occupancy"] <= 0.05
    ]
    display_ok = eligible
    chosen = max(
        display_ok if display_ok else rows,
        key=lambda r: (
            r in display_ok,
            r["mean_iou"],
            -abs(r["saving"] - 0.50),
            r["single"],
            r["watertight"],
            -r["max_face_occupancy"],
        ),
    )
    pred6 = [np.asarray(x) for x in chosen["_ev"]["pred_masks"][:6]]
    for i, arr in enumerate(gt6):
        save_mask(arr, d / "inputs6" / f"input_{i:02d}.png")
    for i, arr in enumerate(pred6):
        save_mask(arr, d / "projections6" / f"projection_{i:02d}.png")
    contact_sheet(name, gt6, pred6, d / f"{name}_input_vs_projection.png")
    stem = f"{name}_{chosen['candidate']}"
    npy = d / f"{stem}_voxels256.npy"
    stl = d / f"{stem}_10cm_256.stl"
    mesh = None
    if display_ok:
        np.save(npy, chosen["_vol"].astype(np.uint8))
        ehqm.export_marching_cubes(chosen["_vol"], stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
        mesh = mesh_clean_scale(stl)
    rec = {
        "case": name,
        "input_paths": [str(p) for p in paths],
        "accepted": bool(display_ok and mesh is not None and mesh.is_watertight and len(mesh.split(only_watertight=False)) == 1),
        "rejection_policy": {
            "enabled": True,
            "reason": None if display_ok else "no candidate satisfied the non-slab display gate",
            "hard_gates": {
                "mean_iou": ">=0.975",
                "min_iou": ">=0.95",
                "saving": "0.30..0.70",
                "single_body": True,
                "watertight": True,
                "erosion_survival": ">=0.12",
                "max_face_occupancy": "<=0.05",
            },
            "slab_like_candidates_exported": False,
        },
        "selected": {
            "candidate": chosen["candidate"],
            "mean_iou": round(chosen["mean_iou"], 6),
            "min_iou": round(chosen["min_iou"], 6),
            "mean_dice": round(chosen["mean_dice"], 6),
            "saving": round(chosen["saving"], 6),
            "single": chosen["single"],
            "watertight_voxel_mesh": chosen["watertight"],
            "mesh_watertight": bool(mesh.is_watertight) if mesh is not None else None,
            "mesh_parts": int(len(mesh.split(only_watertight=False))) if mesh is not None else None,
            "erosion_survival": round(chosen["erosion_survival"], 6),
            "max_face_occupancy": round(chosen["max_face_occupancy"], 6),
            "stl": str(stl) if mesh is not None else None,
            "voxels": str(npy) if mesh is not None else None,
            "preview": str(d / f"{name}_input_vs_projection.png"),
        },
        "all_candidates": [
            {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("_")}
            for r in rows
        ],
        "sec": round(time.time() - t0, 1),
    }
    (d / "CASE_MANIFEST.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rec


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    paths = selected_paths()
    # Spread the combinations across the ranked list to avoid near-identical
    # triples while keeping all members from the structure-friendly pool.
    combos = [
        [paths[0], paths[7], paths[14]],
        [paths[1], paths[8], paths[15]],
        [paths[2], paths[9], paths[16]],
        [paths[3], paths[10], paths[17]],
        [paths[4], paths[11], paths[18]],
        [paths[5], paths[12], paths[19]],
    ]
    records = []
    for i, combo in enumerate(combos, 1):
        rec = run_one(i, combo)
        records.append(rec)
        s = rec["selected"]
        print(
            f"{rec['case']}: accepted={rec['accepted']} IoU={s['mean_iou']} "
            f"Dice={s['mean_dice']} saving={s['saving']} maxface={s['max_face_occupancy']} "
            f"parts={s['mesh_parts']}",
            flush=True,
        )
        (OUT / "MANIFEST.partial.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "schema": "screened_anime_positive_fixed_10cm_256_20260709",
        "source_screen": str(SCREEN),
        "n_cases": len(records),
        "accepted": int(sum(r["accepted"] for r in records)),
        "cases": records,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DONE", OUT)


if __name__ == "__main__":
    main()
