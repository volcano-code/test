#!/usr/bin/env python3
"""Build fixed 10cm/256 candidates from structurally simpler anime silhouettes.

Selection is based on simple binary-shape criteria before generation:
single dominant component, moderate fill area, low perimeter complexity, and
limited holes.  The generator reuses the proven anime30 fixed 10cm/256
pipeline, then adds a max-face-occupancy QA check to reject block/cutout
artifacts.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, generate_binary_structure, label

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
SAVE_MIN = 0.40
SAVE_MAX = 0.70
EROSION_MIN = 0.15
MAX_FACE_MAX = 0.05
MASK_ROOT = ROOT / "results/anime30_adjusted_full_screen_20260704/adjusted_masks/thick1"
OUT = ROOT / "results/simple_anime_fixed_10cm_256_20260708"
S6 = generate_binary_structure(3, 1)


def mask_path(name: str) -> Path:
    return MASK_ROOT / f"{name}_thick1.png"


def load_mask(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((R, R), Image.Resampling.NEAREST)
    return np.asarray(img) > 127


def contour_complexity(mask: np.ndarray) -> dict:
    m = mask.astype(np.uint8)
    area = float(m.mean())
    cc, _ = label(m)
    num_components = int(cc)
    contours, hierarchy = cv2.findContours(m * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(float(cv2.arcLength(c, True)) for c in contours)
    holes = 0
    if hierarchy is not None:
        holes = int(sum(1 for h in hierarchy[0] if h[3] >= 0))
    complexity = float(perimeter / max(np.sqrt(float(m.sum())), 1.0))
    return {
        "area": area,
        "components": num_components,
        "holes": holes,
        "perimeter_complexity": complexity,
    }


def select_simple_names() -> list[str]:
    rows = []
    for path in sorted(MASK_ROOT.glob("anime30_*_thick1.png")):
        name = path.name.replace("_thick1.png", "")
        mask = load_mask(path)
        s = contour_complexity(mask)
        if not (0.08 <= s["area"] <= 0.45):
            continue
        if s["components"] > 3:
            continue
        if s["holes"] > 8:
            continue
        score = s["perimeter_complexity"] + 1.5 * max(0, s["components"] - 1) + 0.35 * s["holes"]
        rows.append({"name": name, "score": score, **s})
    rows.sort(key=lambda r: (r["score"], abs(r["area"] - 0.20)))

    # Keep a mix of proven anime30 IDs and low-complexity alternatives.  This
    # avoids choosing fifteen near-identical silhouettes while still prioritizing
    # simple, printable shapes.
    proven = ["anime30_08", "anime30_15", "anime30_28", "anime30_07", "anime30_13", "anime30_22"]
    selected = []
    seen = set()
    for name in proven:
        path = mask_path(name)
        if path.exists() and name not in seen:
            selected.append(name)
            seen.add(name)
    for row in rows:
        if row["name"] in seen:
            continue
        selected.append(row["name"])
        seen.add(row["name"])
        if len(selected) >= 18:
            break
    (OUT / "simple_silhouette_scores.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return selected


def save_mask(mask: np.ndarray, path: Path) -> None:
    img = ((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
    Image.fromarray(img).resize((512, 512), Image.Resampling.NEAREST).save(path)


def face_occupancy(volume: np.ndarray) -> dict:
    v = np.asarray(volume, dtype=bool)
    coords = np.argwhere(v)
    if coords.size == 0:
        return {"bbox_fill": 0.0, "max_face_occupancy": 1.0, "mean_face_occupancy": 1.0, "face_occupancies": []}
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
    vol = np.asarray(volume).astype(bool)
    return float(binary_erosion(vol, S6).sum() / max(int(vol.sum()), 1))


def contact_sheet(example: str, inputs6: list[np.ndarray], preds6: list[np.ndarray], out: Path) -> None:
    cell = 126
    gap = 16
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
    paths = [str(mask_path(n)) for n in names]
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
        face = face_occupancy(vol)
        rows.append({
            "candidate": cname,
            "mean_iou": float(ev["mean_iou"]),
            "min_iou": float(min(ev["view_ious"][:6])),
            "view_ious6": [float(x) for x in ev["view_ious"][:6]],
            "saving": float(1.0 - int(vol.sum()) / hull_vox),
            "single": bool(n6(vol) == 1),
            "watertight": watertight,
            "erosion_survival": erosion_survival(vol),
            **face,
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
        and r["max_face_occupancy"] <= MAX_FACE_MAX
    ]
    if gate:
        chosen = max(gate, key=lambda r: (-abs(r["saving"] - 0.55), r["mean_iou"], r["erosion_survival"]))
    else:
        chosen = max(rows, key=lambda r: (r["mean_iou"], r["single"], r["watertight"], -r["max_face_occupancy"]))

    pred3 = [np.asarray(x) for x in chosen["_ev"]["pred_masks"][:3]]
    preds6 = pred3 + [np.fliplr(x) for x in pred3]
    for i, arr in enumerate(inputs6):
        save_mask(arr, d / "inputs6" / f"input_view{i}.png")
    for i, arr in enumerate(preds6):
        save_mask(arr, d / "projections6" / f"projection_view{i}.png")
    preview = d / f"{example}_six_input_vs_projection.png"
    contact_sheet(example, inputs6, preds6, preview)

    stem = f"{example}_{chosen['candidate']}"
    npy = d / f"{stem}_voxels256.npy"
    stl = d / f"{stem}_10cm_256.stl"
    np.save(npy, chosen["_vol"].astype(np.uint8))
    ehqm.export_marching_cubes(chosen["_vol"].astype(bool), stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
    mesh = mesh_clean_scale(stl)

    selected = {
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
        "max_face_occupancy": round(chosen["max_face_occupancy"], 6),
        "bbox_fill": round(chosen["bbox_fill"], 6),
        "stl": str(stl),
        "voxels": str(npy),
        "six_view_preview": str(preview),
    }
    accepted = bool(
        chosen in gate
        and selected["mesh_watertight"]
        and selected["mesh_parts"] == 1
    )
    return {
        "example": example,
        "silhouettes": names,
        "input_paths": paths,
        "resolution": R,
        "print_mm": PRINT_MM,
        "fixed": True,
        "accepted": accepted,
        "selected": selected,
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
    selected = select_simple_names()
    groups = [
        ("simple_anime_01", selected[0:3]),
        ("simple_anime_02", selected[3:6]),
        ("simple_anime_03", selected[6:9]),
        ("simple_anime_04", selected[9:12]),
        ("simple_anime_05", selected[12:15]),
    ]
    records = []
    for example, names in groups:
        rec = run_one(example, names)
        records.append(rec)
        s = rec["selected"]
        print(
            f"{example}: accepted={rec['accepted']} names={names} "
            f"{s['candidate']} IoU={s['mean_iou']} saving={s['saving']} maxface={s['max_face_occupancy']}",
            flush=True,
        )
        (OUT / "MANIFEST.partial.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    used = [n for _, names in groups for n in names]
    summary = {
        "schema": "simple_anime_fixed_10cm_256_20260708",
        "selection_rule": "low binary contour complexity, moderate area, few components/holes; fixed 10cm_256 generation",
        "gates": {
            "mean_iou": f">={IOU_MIN}",
            "saving": f"{SAVE_MIN}..{SAVE_MAX}",
            "erosion_survival": f">={EROSION_MIN}",
            "max_face_occupancy": f"<={MAX_FACE_MAX}",
            "mesh_parts": 1,
            "mesh_watertight": True,
        },
        "unique_input_count": len(set(used)),
        "all_inputs_unique": len(set(used)) == len(used),
        "n_examples": len(records),
        "accepted": sum(1 for r in records if r["accepted"]),
        "examples": records,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
