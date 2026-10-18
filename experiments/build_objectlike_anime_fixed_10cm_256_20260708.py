#!/usr/bin/env python3
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
for p in (ROOT, ROOT / "generation", ROOT / "experiments"):
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

SRC = Path("/home/lin/下载/original_anime_silhouettes_30(1) (2)/original_anime_silhouettes_30")
OUT = ROOT / "results/objectlike_anime_fixed_10cm_256_20260708"
R = 256
PRINT_MM = 100.0
IOU_MIN = 0.970
MIN_VIEW = 0.950
SAVE_MIN = 0.40
SAVE_MAX = 0.60
BBOX_FILL_MIN = 0.20
EROSION_MIN = 0.15
S6 = generate_binary_structure(3, 1)

EXAMPLES = [
    ("blade_heroes", "Blade Heroes", ["01", "11", "25"]),
    ("arcane_casters", "Arcane Casters", ["02", "14", "27"]),
    ("stage_dancers", "Stage Dancers", ["08", "16", "22"]),
    ("mechanic_companions", "Mechanic Companions", ["12", "13", "30"]),
    ("commanders", "Commanders", ["06", "18", "28"]),
    ("shrine_mages", "Shrine Mages", ["05", "14", "27"]),
    ("street_duelists", "Street Duelists", ["15", "18", "23"]),
    ("future_idols", "Future Idols", ["08", "12", "18"]),
    ("hero_party", "Hero Party", ["06", "13", "21"]),
]


def path_for(idx: str) -> str:
    hits = sorted(SRC.glob(f"{idx}_*.png"))
    if len(hits) != 1:
        raise FileNotFoundError((idx, hits))
    return str(hits[0])


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(((np.asarray(mask) > 0.5).astype(np.uint8) * 255)).resize(
        (512, 512), Image.Resampling.NEAREST
    ).save(path)


def bbox_fill(volume: np.ndarray) -> float:
    coords = np.argwhere(np.asarray(volume, dtype=bool))
    if coords.size == 0:
        return 0.0
    ext = coords.max(axis=0) - coords.min(axis=0) + 1
    return float(np.asarray(volume, dtype=bool).sum() / max(int(np.prod(ext)), 1))


def erosion_survival(volume: np.ndarray) -> float:
    vol = np.asarray(volume, dtype=bool)
    return float(binary_erosion(vol, S6).sum() / max(int(vol.sum()), 1))


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


def contact_sheet(title: str, inputs6: list[np.ndarray], preds6: list[np.ndarray], out: Path) -> None:
    cell, gap, left = 128, 14, 112
    canvas = Image.new("RGB", (left + 6 * (cell + gap) + gap, 2 * cell + 3 * gap + 24), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), title, fill=(0, 0, 0))
    draw.text((gap, gap + 52), "Target", fill=(0, 0, 0))
    draw.text((gap, 2 * gap + cell + 52), "Projection", fill=(0, 0, 0))
    for row, imgs in enumerate((inputs6, preds6)):
        for i, arr in enumerate(imgs):
            x = left + gap + i * (cell + gap)
            y = gap + 24 + row * (cell + gap)
            img = Image.fromarray((np.asarray(arr) > 0.5).astype(np.uint8) * 255).resize(
                (cell, cell), Image.Resampling.NEAREST
            )
            canvas.paste(img.convert("RGB"), (x, y))
            draw.rectangle((x, y, x + cell, y + cell), outline=(120, 120, 120))
    canvas.save(out)


def run_one(slug: str, title: str, ids: list[str]) -> dict:
    t0 = time.time()
    d = OUT / slug
    (d / "inputs6").mkdir(parents=True, exist_ok=True)
    (d / "projections6").mkdir(parents=True, exist_ok=True)
    paths = [path_for(i) for i in ids]
    sils, _ = load_data_from_list(paths, True, (R, R), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils[:3]]
    inputs6 = masks + [np.fliplr(m) for m in masks]
    cands, aux = make_candidates(masks, R, True, print_mm=PRINT_MM, nozzle_mm=0.4, wall_safety=2.0)
    hull_vox = max(int(aux["hull"].sum()), 1)
    rows = []
    for cname, vol in cands.items():
        vol = np.asarray(vol, dtype=bool)
        ev = evaluate_tfs_volume(vol.astype(np.float32), inputs6, aux["calibration"], image_size=512)
        try:
            water = bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight)
        except Exception:
            water = False
        rows.append({
            "candidate": cname,
            "mean_iou": float(ev["mean_iou"]),
            "min_iou": float(min(ev["view_ious"][:3])),
            "view_ious6": [float(x) for x in ev["view_ious"][:6]],
            "saving": float(1.0 - int(vol.sum()) / hull_vox),
            "single": bool(n6(vol) == 1),
            "watertight": water,
            "erosion_survival": erosion_survival(vol),
            "bbox_fill": bbox_fill(vol),
            "_vol": vol,
            "_ev": ev,
        })
    gate = [
        r for r in rows
        if r["mean_iou"] >= IOU_MIN
        and r["min_iou"] >= MIN_VIEW
        and SAVE_MIN <= r["saving"] <= SAVE_MAX
        and r["single"]
        and r["watertight"]
        and r["erosion_survival"] >= EROSION_MIN
        and r["bbox_fill"] >= BBOX_FILL_MIN
    ]
    chosen = max(gate, key=lambda r: (r["bbox_fill"], r["mean_iou"], -abs(r["saving"] - 0.52))) if gate else max(
        rows, key=lambda r: (r["bbox_fill"], r["mean_iou"], r["single"], r["watertight"])
    )
    pred3 = [np.asarray(x) for x in chosen["_ev"]["pred_masks"][:3]]
    preds6 = pred3 + [np.fliplr(x) for x in pred3]
    for i, arr in enumerate(inputs6):
        save_mask(arr, d / "inputs6" / f"input_view{i}.png")
    for i, arr in enumerate(preds6):
        save_mask(arr, d / "projections6" / f"projection_view{i}.png")
    contact_sheet(title, inputs6, preds6, d / f"{slug}_six_input_vs_projection.png")
    stem = f"{slug}_{chosen['candidate']}"
    npy = d / f"{stem}_voxels256.npy"
    stl = d / f"{stem}_10cm_256.stl"
    np.save(npy, chosen["_vol"].astype(np.uint8))
    ehqm.export_marching_cubes(chosen["_vol"].astype(bool), stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
    mesh = mesh_clean_scale(stl)
    parts = int(len(mesh.split(only_watertight=False)))
    accepted = bool(chosen in gate and mesh.is_watertight and parts == 1)
    return {
        "example": slug,
        "group_name": title,
        "silhouette_ids": ids,
        "input_paths": paths,
        "fixed": True,
        "resolution": R,
        "print_mm": PRINT_MM,
        "accepted": accepted,
        "selected": {
            "candidate": chosen["candidate"],
            "mean_iou": round(chosen["mean_iou"], 6),
            "min_iou": round(chosen["min_iou"], 6),
            "view_ious6": [round(float(x), 6) for x in chosen["view_ious6"]],
            "saving": round(chosen["saving"], 6),
            "single": chosen["single"],
            "watertight": chosen["watertight"],
            "mesh_watertight": bool(mesh.is_watertight),
            "mesh_parts": parts,
            "erosion_survival": round(chosen["erosion_survival"], 6),
            "bbox_fill": round(chosen["bbox_fill"], 6),
            "stl": str(stl),
            "voxels": str(npy),
            "six_view_preview": str(d / f"{slug}_six_input_vs_projection.png"),
        },
        "all_candidates": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items() if not k.startswith("_")} for row in rows],
        "sec": round(time.time() - t0, 1),
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for slug, title, ids in EXAMPLES:
        rec = run_one(slug, title, ids)
        records.append(rec)
        s = rec["selected"]
        print(f"{slug}: accepted={rec['accepted']} {s['candidate']} IoU={s['mean_iou']} min={s['min_iou']} saving={s['saving']} fill={s['bbox_fill']} parts={s['mesh_parts']} water={s['mesh_watertight']}", flush=True)
        (OUT / "MANIFEST.partial.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    accepted = [r for r in records if r["accepted"]]
    summary = {
        "schema": "objectlike_anime_fixed_10cm_256_20260708",
        "rejected_folder": str(ROOT / "results/balanced_anime_fixed_search_10cm_256_20260708"),
        "source_dir": str(SRC),
        "objectlike_gate": {"bbox_fill_min": BBOX_FILL_MIN, "mean_iou_min": IOU_MIN, "min_view_iou_min": MIN_VIEW, "saving_range": [SAVE_MIN, SAVE_MAX], "mesh_parts": 1, "mesh_watertight": True},
        "n_examples": len(records),
        "accepted": len(accepted),
        "accepted_examples": [r["example"] for r in accepted],
        "examples": records,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
