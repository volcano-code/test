#!/usr/bin/env python3
"""Search fixed 10cm anime print candidates from the balanced silhouette pool.

The search is intentionally two-stage: a fast low-resolution fixed-branch
screen finds silhouette triples whose shadow null-space has printable material
reduction, then the best triples are re-run at 256 for final STL export.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, generate_binary_structure, label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
for item in (ROOT, ROOT / "generation", ROOT / "experiments"):
    sys.path.insert(0, str(item))

from candidate_selection_batch import make_candidates, n6
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh

_spec = importlib.util.spec_from_file_location(
    "export_high_quality_mesh", str(ROOT / "generation/export_high_quality_mesh.py")
)
ehqm = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ehqm)

SRC = Path("/home/lin/xvxinyao/download_anime_character_silhouettes_balanced_friendly_20260707/images")
OUT = ROOT / "results/balanced_anime_fixed_search_10cm_256_20260708"
SCREEN_R = 96
FINAL_R = 256
PRINT_MM = 100.0
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0
RANDOM_SEED = 2708
SCREEN_LIMIT = 60
FINAL_LIMIT = 8
S6 = generate_binary_structure(3, 1)


def load_mask(path: Path, res: int) -> np.ndarray:
    tensors, _ = load_data_from_list([str(path)], True, (res, res), torch.device("cpu"), 1, False)
    return np.asarray(tensors[0]).squeeze().astype(np.float32)


def mask_features(path: Path) -> dict:
    arr = np.asarray(Image.open(path).convert("L").resize((128, 128), Image.Resampling.LANCZOS))
    mask = arr > 127
    labs, comp = label(mask)
    sizes = np.bincount(labs.ravel())
    largest = int(sizes[1:].max()) if comp else 0
    area = float(mask.mean())
    ys, xs = np.where(mask)
    if len(xs) == 0:
        bbox_area = 1.0
        aspect = 0.0
    else:
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        bbox_area = max(w * h, 1)
        aspect = float(w / max(h, 1))
    return {
        "path": str(path),
        "area": area,
        "components": int(comp),
        "largest_frac": float(largest / max(int(mask.sum()), 1)),
        "bbox_fill": float(mask.sum() / bbox_area),
        "aspect": aspect,
    }


def is_structure_friendly(feat: dict) -> bool:
    return bool(
        0.12 <= feat["area"] <= 0.48
        and feat["components"] <= 4
        and feat["largest_frac"] >= 0.82
        and 0.18 <= feat["bbox_fill"] <= 0.82
        and 0.35 <= feat["aspect"] <= 1.55
    )


def erosion_survival(volume: np.ndarray) -> float:
    vol = np.asarray(volume, dtype=bool)
    return float(binary_erosion(vol, S6).sum() / max(int(vol.sum()), 1))


def evaluate_triple(paths: list[Path], res: int, export: bool = False, out_dir: Path | None = None) -> dict:
    sils, _ = load_data_from_list([str(p) for p in paths], True, (res, res), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils[:3]]
    gt6 = masks + [np.fliplr(m) for m in masks]
    cands, aux = make_candidates(masks, res, True, print_mm=PRINT_MM, nozzle_mm=NOZZLE_MM, wall_safety=WALL_SAFETY)
    hull_vox = max(int(aux["hull"].sum()), 1)
    rows = []
    for name, vol in cands.items():
        vol = np.asarray(vol, dtype=bool)
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=512)
        try:
            mesh = sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6)
            watertight = bool(mesh.is_watertight and mesh.is_winding_consistent)
        except Exception:
            watertight = False
        rows.append(
            {
                "candidate": name,
                "mean_iou": float(ev["mean_iou"]),
                "min_iou": float(min(ev["view_ious"][:3])),
                "view_ious6": [float(x) for x in ev["view_ious"][:6]],
                "saving": float(1.0 - int(vol.sum()) / hull_vox),
                "single": bool(n6(vol) == 1),
                "watertight": watertight,
                "erosion_survival": erosion_survival(vol),
                "_vol": vol,
                "_ev": ev,
            }
        )
    gate = [
        r
        for r in rows
        if r["mean_iou"] >= 0.965
        and r["min_iou"] >= 0.940
        and 0.40 <= r["saving"] <= 0.60
        and r["single"]
        and r["watertight"]
        and r["erosion_survival"] >= 0.12
    ]
    if gate:
        chosen = max(gate, key=lambda r: (-abs(r["saving"] - 0.52), r["mean_iou"], r["min_iou"]))
    else:
        chosen = max(rows, key=lambda r: (r["mean_iou"], -abs(r["saving"] - 0.52), r["single"], r["watertight"]))
    rec = {
        "paths": [str(p) for p in paths],
        "resolution": res,
        "accepted": bool(gate),
        "selected": {k: v for k, v in chosen.items() if not k.startswith("_")},
        "all_candidates": [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows],
    }
    if export and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir.name
        npy = out_dir / f"{stem}_{chosen['candidate']}_voxels{res}.npy"
        stl = out_dir / f"{stem}_{chosen['candidate']}_10cm_{res}.stl"
        np.save(npy, chosen["_vol"].astype(np.uint8))
        ehqm.export_marching_cubes(chosen["_vol"].astype(bool), stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
        mesh = trimesh.load(str(stl), force="mesh")
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        trimesh.repair.fill_holes(mesh)
        if mesh.volume < 0:
            mesh.invert()
        trimesh.repair.fix_normals(mesh)
        mesh.apply_scale(PRINT_MM / max(float(np.asarray(mesh.extents).max()), 1e-9))
        mesh.export(str(stl))
        parts = mesh.split(only_watertight=False)
        pred3 = [np.asarray(x) for x in chosen["_ev"]["pred_masks"][:3]]
        preds6 = pred3 + [np.fliplr(x) for x in pred3]
        save_contact_sheet(masks + [np.fliplr(m) for m in masks], preds6, out_dir / f"{stem}_six_input_vs_projection.png")
        rec["selected"].update(
            {
                "stl": str(stl),
                "voxels": str(npy),
                "mesh_watertight": bool(mesh.is_watertight),
                "mesh_parts": int(len(parts)),
                "six_view_preview": str(out_dir / f"{stem}_six_input_vs_projection.png"),
            }
        )
    return rec


def save_contact_sheet(inputs6: list[np.ndarray], preds6: list[np.ndarray], path: Path) -> None:
    cell = 132
    gap = 16
    left = 118
    canvas = Image.new("RGB", (left + 6 * cell + 7 * gap, 2 * cell + 3 * gap), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, gap + 48), "Target", fill=(0, 0, 0))
    draw.text((12, 2 * gap + cell + 48), "Projection", fill=(0, 0, 0))
    for row, imgs in enumerate((inputs6, preds6)):
        for i, arr in enumerate(imgs):
            x = left + gap + i * (cell + gap)
            y = gap + row * (cell + gap)
            img = Image.fromarray((np.asarray(arr) > 0.5).astype(np.uint8) * 255).resize((cell, cell), Image.Resampling.NEAREST)
            canvas.paste(img.convert("RGB"), (x, y))
            draw.rectangle((x, y, x + cell, y + cell), outline=(130, 130, 130))
    canvas.save(path)


def main() -> None:
    t0 = time.time()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    paths = sorted(SRC.glob("*.png"))
    feats = [mask_features(p) for p in paths]
    friendly = [Path(f["path"]) for f in feats if is_structure_friendly(f)]
    if len(friendly) < 18:
        friendly = paths

    random.seed(RANDOM_SEED)
    combos = list(itertools.combinations(friendly, 3))
    random.shuffle(combos)
    combos = combos[:SCREEN_LIMIT]

    screen_records = []
    for idx, combo in enumerate(combos, 1):
        try:
            rec = evaluate_triple(list(combo), SCREEN_R, export=False)
        except Exception as exc:
            rec = {"paths": [str(p) for p in combo], "resolution": SCREEN_R, "error": repr(exc), "accepted": False}
        screen_records.append(rec)
        if idx % 25 == 0:
            accepted = sum(1 for r in screen_records if r.get("accepted"))
            print(f"screened {idx}/{len(combos)} accepted={accepted}", flush=True)
            (OUT / "SCREEN.partial.json").write_text(json.dumps(screen_records, ensure_ascii=False, indent=2))

    ranked = sorted(
        [r for r in screen_records if "selected" in r],
        key=lambda r: (
            r.get("accepted", False),
            -abs(float(r["selected"]["saving"]) - 0.52),
            float(r["selected"]["mean_iou"]),
            float(r["selected"]["min_iou"]),
        ),
        reverse=True,
    )
    final_records = []
    used_paths: set[str] = set()
    for rec in ranked:
        combo = rec["paths"]
        if any(p in used_paths for p in combo):
            continue
        slug = f"anime_fixed_{len(final_records)+1:02d}"
        final = evaluate_triple([Path(p) for p in combo], FINAL_R, export=True, out_dir=OUT / slug)
        final["screen_selected"] = rec["selected"]
        final_records.append(final)
        used_paths.update(combo)
        s = final["selected"]
        print(
            f"{slug}: accepted={final['accepted']} {s['candidate']} "
            f"IoU={s['mean_iou']:.4f} min={s['min_iou']:.4f} saving={s['saving']:.4f} "
            f"parts={s.get('mesh_parts')} watertight={s.get('mesh_watertight')}",
            flush=True,
        )
        if len(final_records) >= FINAL_LIMIT:
            break

    summary = {
        "schema": "balanced_anime_fixed_search_10cm_256_20260708",
        "source_dir": str(SRC),
        "screen_resolution": SCREEN_R,
        "final_resolution": FINAL_R,
        "friendly_images": len(friendly),
        "screened_combinations": len(screen_records),
        "screen_accepted": sum(1 for r in screen_records if r.get("accepted")),
        "final_exported": len(final_records),
        "final_accepted": sum(1 for r in final_records if r.get("accepted")),
        "screen_records": screen_records,
        "final_records": final_records,
        "sec": round(time.time() - t0, 1),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
