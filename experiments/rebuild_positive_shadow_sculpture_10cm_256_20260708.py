#!/usr/bin/env python3
"""Regenerate positive, non-mold 10cm/256 ShadowForge print examples.

This keeps the proven print_10cm_256 protocol but writes to a new folder and
adds a planar-face occupancy check.  Low face occupancy rejects rectangular
"block with cutouts" artifacts.
"""
from __future__ import annotations

import importlib.util
import json
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
CANDIDATE = "carve_pf5"
BENCH = ROOT / "data/silhouettes/bench3"
OUT = ROOT / "results/positive_shadow_sculpture_10cm_256_20260708"
S6 = generate_binary_structure(3, 1)

EXAMPLES = {
    "owl_bat_turtle": ["birds_owl", "x_bat", "reptiles_turtle"],
    "eagle_frog_trex": ["birds_eagle", "reptiles_frog", "x_trex"],
    "parrot_dragon_snail": ["birds_parrot", "x_dragon", "sea_snail"],
    "bat_trex_snail": ["x_bat", "x_trex", "sea_snail"],
}


def save_binary_image(mask: np.ndarray, path: Path, size: int = 512) -> None:
    img = ((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
    Image.fromarray(img).resize((size, size), Image.Resampling.NEAREST).save(path)


def face_occupancy(volume: np.ndarray) -> dict:
    v = np.asarray(volume, dtype=bool)
    coords = np.argwhere(v)
    if coords.size == 0:
        return {"bbox_fill": 0.0, "max_face_occupancy": 1.0, "mean_face_occupancy": 1.0}
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    sub = v[mins[0]:maxs[0] + 1, mins[1]:maxs[1] + 1, mins[2]:maxs[2] + 1]
    faces = [
        float(sub[0, :, :].mean()),
        float(sub[-1, :, :].mean()),
        float(sub[:, 0, :].mean()),
        float(sub[:, -1, :].mean()),
        float(sub[:, :, 0].mean()),
        float(sub[:, :, -1].mean()),
    ]
    return {
        "bbox_fill": float(sub.mean()),
        "max_face_occupancy": float(max(faces)),
        "mean_face_occupancy": float(np.mean(faces)),
        "face_occupancies": faces,
    }


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


def contact_sheet(rows: list[dict]) -> None:
    cell = 120
    gap = 14
    label_w = 170
    cols = 6
    canvas = Image.new("RGB", (label_w + cols * (cell + gap) + gap, len(rows) * (cell + 46) + gap), "white")
    draw = ImageDraw.Draw(canvas)
    for r, rec in enumerate(rows):
        y = gap + r * (cell + 46)
        draw.text((10, y + 42), rec["example"], fill=(0, 0, 0))
        imgs = rec["preview_masks"]
        for i, mask in enumerate(imgs):
            x = label_w + gap + i * (cell + gap)
            img = Image.fromarray((np.asarray(mask) > 0.5).astype(np.uint8) * 255).resize(
                (cell, cell), Image.Resampling.NEAREST
            )
            canvas.paste(img.convert("RGB"), (x, y))
            draw.rectangle((x, y, x + cell, y + cell), outline=(120, 120, 120))
    canvas.save(OUT / "INPUT_AND_PROJECTION_CONTACT_SHEET.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    sheet_rows = []
    for example, names in EXAMPLES.items():
        t0 = time.time()
        paths = [BENCH / f"{name}.png" for name in names]
        sils, _ = load_data_from_list([str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False)
        masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils]
        candidates, aux = make_candidates(
            masks, R, True, print_mm=PRINT_MM, nozzle_mm=NOZZLE_MM, wall_safety=WALL_SAFETY
        )
        vol = np.asarray(candidates[CANDIDATE], dtype=bool)
        hull = np.asarray(aux["hull"], dtype=bool)
        gt6 = masks + [np.fliplr(m) for m in masks]
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=512)
        d = OUT / example
        (d / "inputs").mkdir(parents=True, exist_ok=True)
        (d / "projections").mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(names):
            save_binary_image(masks[i], d / "inputs" / f"input{i}_{name}.png")
            save_binary_image(ev["pred_masks"][i], d / "projections" / f"projection{i}_{name}.png")
        stl = d / f"{example}_10cm_256_{CANDIDATE}.stl"
        npy = d / f"{example}_voxels256_{CANDIDATE}.npy"
        np.save(npy, vol.astype(np.uint8))
        ehqm.export_marching_cubes(vol, stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
        mesh = mesh_clean_scale(stl)
        face = face_occupancy(vol)
        rec = {
            "example": example,
            "silhouettes": names,
            "resolution": R,
            "print_mm": PRINT_MM,
            "candidate": CANDIDATE,
            "mean_iou": round(float(ev["mean_iou"]), 6),
            "min_iou": round(float(min(ev["view_ious"][:3])), 6),
            "view_ious6": [round(float(x), 6) for x in ev["view_ious"][:6]],
            "saving": round(float(1.0 - int(vol.sum()) / max(int(hull.sum()), 1)), 6),
            "single": bool(n6(vol) == 1),
            "watertight": bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight),
            "mesh_watertight": bool(mesh.is_watertight),
            "mesh_parts": int(len(mesh.split(only_watertight=False))),
            "erosion_survival": round(erosion_survival(vol), 6),
            **{k: (round(v, 6) if isinstance(v, float) else [round(float(x), 6) for x in v]) for k, v in face.items()},
            "positive_shape_pass": bool(face["max_face_occupancy"] <= 0.05),
            "stl": str(stl),
            "voxels": str(npy),
            "sec": round(time.time() - t0, 1),
        }
        rec["accepted"] = bool(
            rec["mean_iou"] >= 0.97
            and 0.40 <= rec["saving"] <= 0.70
            and rec["single"]
            and rec["mesh_watertight"]
            and rec["mesh_parts"] == 1
            and rec["positive_shape_pass"]
        )
        rows.append(rec)
        sheet_rows.append({"example": example, "preview_masks": masks + [np.asarray(x) for x in ev["pred_masks"][:3]]})
        print(
            f"{example}: accepted={rec['accepted']} IoU={rec['mean_iou']} "
            f"saving={rec['saving']} maxface={rec['max_face_occupancy']} parts={rec['mesh_parts']}",
            flush=True,
        )
    contact_sheet(sheet_rows)
    manifest = {
        "schema": "positive_shadow_sculpture_10cm_256_20260708",
        "source_protocol": "ShadowForge print_10cm_256 clear protocol with additional positive-shape gate",
        "rejected_protocols": {
            "objectlike_anime_fixed_10cm_256_20260708": "rejected because large bbox-face occupancy produced block/mold-like artifacts",
            "balanced_anime_fixed_search_10cm_256_20260708": "rejected because generated low-quality stitched silhouettes and mold-like artifacts",
        },
        "positive_shape_gate": {"max_face_occupancy": "<=0.05"},
        "n_examples": len(rows),
        "accepted": sum(1 for r in rows if r["accepted"]),
        "examples": rows,
        "contact_sheet": str(OUT / "INPUT_AND_PROJECTION_CONTACT_SHEET.png"),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
