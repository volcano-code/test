#!/usr/bin/env python3
"""Island-aware support-free attempt for the 10 cm / 256^3 print set.

This is a deterministic diagnostic:
  1. choose the best physical assignment among the three silhouettes;
  2. add material only inside the calibrated 256^3 visual hull;
  3. add sparse closure only for large unsupported islands.

The hard projection check is still evaluate_tfs_volume.  The support check is a
voxel FDM proxy: a voxel is self-supported if the previous build layer has solid
material in its 3x3 neighborhood.
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
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, find_objects, generate_binary_structure, label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
for p in [ROOT, ROOT / "generation", ROOT / "experiments"]:
    sys.path.insert(0, str(p))

from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from candidate_selection_batch import make_candidates, n6, keep_largest

_spec = importlib.util.spec_from_file_location(
    "export_high_quality_mesh", str(ROOT / "generation/export_high_quality_mesh.py")
)
ehqm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ehqm)

R = 256
PRINT_MM = 100.0
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0
BASE_CANDIDATE = "carve_pf5"
BENCH = ROOT / "data/silhouettes/bench3"
OUT = ROOT / "results/support_free_10cm_256_v2"
S2 = np.ones((3, 3), dtype=bool)
S6 = generate_binary_structure(3, 1)

MIN_SIGNIFICANT_AREA_VOX = 16
MIN_SIGNIFICANT_SPAN_MM = 4.0
MAX_PASSES = 4
BED_TARGET_MM2 = 80.0

EXAMPLES = {
    "owl_bat_turtle": ["birds_owl", "x_bat", "reptiles_turtle"],
    "eagle_frog_trex": ["birds_eagle", "reptiles_frog", "x_trex"],
    "parrot_dragon_snail": ["birds_parrot", "x_dragon", "sea_snail"],
    "bat_trex_snail": ["x_bat", "x_trex", "sea_snail"],
}


def save_binary_image(mask: np.ndarray, path: Path) -> None:
    img = ((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
    Image.fromarray(img).resize((512, 512), Image.Resampling.NEAREST).save(path)


def load_masks(names: list[str]) -> list[np.ndarray]:
    paths = [BENCH / f"{name}.png" for name in names]
    sils, _ = load_data_from_list(
        [str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False
    )
    return [np.asarray(x).squeeze().astype(np.float32) for x in sils]


def support_metrics(volume: np.ndarray, print_mm: float = PRINT_MM) -> dict:
    v = volume.astype(bool)
    voxel_mm = float(print_mm) / float(v.shape[0])
    if not v.any():
        return {
            "solid_voxels": 0,
            "z_min": None,
            "z_max": None,
            "unsupported_voxels": 0,
            "unsupported_ratio": 0.0,
            "unsupported_island_count": 0,
            "significant_unsupported_voxels": 0,
            "significant_unsupported_ratio": 0.0,
            "significant_unsupported_island_count": 0,
            "max_layer_unsupported": 0,
            "bed_contact_voxels": 0,
            "bed_contact_area_mm2": 0.0,
        }
    z_nonzero = np.where(v)[2]
    z_min = int(z_nonzero.min())
    z_max = int(z_nonzero.max())
    unsupported_total = 0
    island_total = 0
    significant_total = 0
    significant_islands = 0
    max_layer_unsupported = 0

    for z in range(z_min + 1, z_max + 1):
        support = binary_dilation(v[:, :, z - 1], structure=S2)
        unsupported = v[:, :, z] & ~support
        cnt = int(unsupported.sum())
        unsupported_total += cnt
        max_layer_unsupported = max(max_layer_unsupported, cnt)
        if not cnt:
            continue
        lab, nc = label(unsupported, structure=S2)
        island_total += int(nc)
        slices = find_objects(lab)
        for idx, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            comp = lab[slc] == idx
            area = int(comp.sum())
            span_vox = max(slc[0].stop - slc[0].start, slc[1].stop - slc[1].start)
            span_mm = float(span_vox * voxel_mm)
            if area >= MIN_SIGNIFICANT_AREA_VOX or span_mm >= MIN_SIGNIFICANT_SPAN_MM:
                significant_total += area
                significant_islands += 1

    solid = int(v.sum())
    bed_vox = int(v[:, :, z_min].sum())
    return {
        "solid_voxels": solid,
        "z_min": z_min,
        "z_max": z_max,
        "unsupported_voxels": int(unsupported_total),
        "unsupported_ratio": float(unsupported_total / max(solid, 1)),
        "unsupported_island_count": int(island_total),
        "significant_unsupported_voxels": int(significant_total),
        "significant_unsupported_ratio": float(significant_total / max(solid, 1)),
        "significant_unsupported_island_count": int(significant_islands),
        "max_layer_unsupported": int(max_layer_unsupported),
        "bed_contact_voxels": bed_vox,
        "bed_contact_area_mm2": float(bed_vox * voxel_mm * voxel_mm),
    }


def choose_assignment(example: str, names: list[str]) -> tuple[list[str], dict, dict]:
    """Use the already computed v1 orientation scan to avoid 6x 256^3 carving."""
    scan_path = ROOT / "results/support_free_10cm_256_v1/orientation_scan.json"
    trials = []
    best_floor = names[0]
    if scan_path.exists():
        scan = json.loads(scan_path.read_text())
        row = next((r for r in scan if r.get("example") == example), None)
        if row is not None:
            best_floor = row["best"]["floor_silhouette"]
            trials = row.get("all", [])

    remaining = [n for n in names if n != best_floor]
    ordered = [best_floor] + remaining
    masks = load_masks(ordered)
    candidates, aux = make_candidates(
        masks,
        R,
        True,
        print_mm=PRINT_MM,
        nozzle_mm=NOZZLE_MM,
        wall_safety=WALL_SAFETY,
    )
    return ordered, candidates, {**aux, "assignment_trials": trials}


def add_bed_pad(volume: np.ndarray, hull: np.ndarray, target_mm2: float) -> tuple[np.ndarray, int]:
    v = volume.astype(bool).copy()
    if not v.any():
        return v, 0
    voxel_area = (PRINT_MM / R) ** 2
    target_vox = int(np.ceil(target_mm2 / voxel_area))
    z_min = int(np.where(v)[2].min())
    bed = v[:, :, z_min]
    added = 0
    for _ in range(24):
        if int(bed.sum()) >= target_vox:
            break
        grown = binary_dilation(bed, structure=S2) & hull[:, :, z_min]
        add = grown & ~bed
        if not add.any():
            break
        bed |= add
        added += int(add.sum())
    v[:, :, z_min] = bed
    return v, added


def add_large_island_closure(volume: np.ndarray, hull: np.ndarray) -> tuple[np.ndarray, dict]:
    v = volume.astype(bool).copy()
    h = hull.astype(bool)
    before = int(v.sum())
    records = []

    for pass_idx in range(MAX_PASSES):
        added_this_pass = 0
        z_nonzero = np.where(v)[2]
        z_min = int(z_nonzero.min())
        z_max = int(z_nonzero.max())
        for z in range(z_max, z_min, -1):
            support = binary_dilation(v[:, :, z - 1], structure=S2)
            unsupported = v[:, :, z] & ~support
            if not unsupported.any():
                continue
            lab, nc = label(unsupported, structure=S2)
            if nc == 0:
                continue
            large = np.zeros_like(unsupported, dtype=bool)
            slices = find_objects(lab)
            for idx, slc in enumerate(slices, start=1):
                if slc is None:
                    continue
                comp = lab[slc] == idx
                area = int(comp.sum())
                span_vox = max(slc[0].stop - slc[0].start, slc[1].stop - slc[1].start)
                span_mm = float(span_vox * PRINT_MM / R)
                if area >= MIN_SIGNIFICANT_AREA_VOX or span_mm >= MIN_SIGNIFICANT_SPAN_MM:
                    large[slc] |= comp
            if not large.any():
                continue
            desired = binary_dilation(large, structure=S2) & h[:, :, z - 1]
            add = desired & ~v[:, :, z - 1]
            if add.any():
                v[:, :, z - 1] |= add
                added_this_pass += int(add.sum())
        records.append({"pass": pass_idx + 1, "added_voxels": int(added_this_pass)})
        if added_this_pass == 0:
            break
    v &= h
    v = keep_largest(v)
    return v.astype(bool), {
        "added_voxels": int(max(int(v.sum()) - before, 0)),
        "volume_before": before,
        "volume_after": int(v.sum()),
        "passes": records,
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


def run_one(example: str, original_names: list[str]) -> dict:
    t0 = time.time()
    names, candidates, aux = choose_assignment(example, original_names)
    masks = load_masks(names)
    hull = aux["hull"].astype(bool)
    hv = max(int(hull.sum()), 1)
    base = candidates[BASE_CANDIDATE].astype(bool)
    padded, pad_added = add_bed_pad(base, hull, BED_TARGET_MM2)
    opt, closure = add_large_island_closure(padded, hull)

    gt6 = masks + [np.fliplr(m) for m in masks]
    base_eval = evaluate_tfs_volume(base.astype(np.float32), gt6, aux["calibration"], image_size=512)
    opt_eval = evaluate_tfs_volume(opt.astype(np.float32), gt6, aux["calibration"], image_size=512)
    base_support = support_metrics(base)
    opt_support = support_metrics(opt)

    d = OUT / example
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "silhouettes").mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        img = np.array(Image.open(BENCH / f"{name}.png").convert("L"))
        save_binary_image(img > 127, d / "inputs" / f"input{i}_{name}.png")
        save_binary_image(opt_eval["pred_masks"][i], d / "silhouettes" / f"shadow{i}_{name}.png")

    np.save(d / f"{example}_support_free_v2_voxels256.npy", opt.astype(np.uint8))
    stl = d / f"{example}_support_free_v2_10cm_256.stl"
    ehqm.export_marching_cubes(
        opt.astype(bool),
        stl,
        smooth_sigma=0.6,
        taubin_iterations=3,
        upsample_factor=1,
    )
    mesh = mesh_clean_scale(stl)
    try:
        wt = bool(sdf_watertight_mesh(opt.astype(np.uint8), smooth_sigma=0.6).is_watertight)
    except Exception:
        wt = False

    rec = {
        "example": example,
        "original_silhouettes": original_names,
        "chosen_physical_order": {
            "floor_z_view0": names[0],
            "left_wall_x_view1": names[1],
            "right_wall_y_view2": names[2],
        },
        "assignment_trials": aux["assignment_trials"],
        "resolution": R,
        "print_mm": PRINT_MM,
        "base_candidate": BASE_CANDIDATE,
        "base": {
            "iou": round(float(base_eval["mean_iou"]), 4),
            "saving": round(float(1.0 - int(base.sum()) / hv), 4),
            "single": bool(n6(base) == 1),
            "watertight": bool(sdf_watertight_mesh(base.astype(np.uint8), smooth_sigma=0.6).is_watertight),
            "support": base_support,
        },
        "support_free_attempt": {
            "iou": round(float(opt_eval["mean_iou"]), 4),
            "saving": round(float(1.0 - int(opt.sum()) / hv), 4),
            "single": bool(n6(opt) == 1),
            "watertight": wt,
            "erosion_survival": round(erosion_survival(opt), 4),
            "support": opt_support,
            "pad_added_voxels": int(pad_added),
            "closure": closure,
            "mesh_extents_mm": [round(float(x), 1) for x in mesh.extents],
            "mesh_faces": int(len(mesh.faces)),
        },
        "passes_proxy_gate": bool(
            n6(opt) == 1
            and wt
            and float(opt_eval["mean_iou"]) >= 0.98
            and float(1.0 - int(opt.sum()) / hv) >= 0.45
            and opt_support["unsupported_ratio"] <= 0.01
            and opt_support["significant_unsupported_island_count"] == 0
            and opt_support["bed_contact_area_mm2"] >= BED_TARGET_MM2
        ),
        "sec": round(time.time() - t0, 1),
    }
    with open(d / "support_free_v2_metrics.json", "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for example, names in EXAMPLES.items():
        rec = run_one(example, names)
        rows.append(rec)
        b = rec["base"]["support"]
        o = rec["support_free_attempt"]["support"]
        print(
            f"{example}: floor={rec['chosen_physical_order']['floor_z_view0']} "
            f"IoU {rec['base']['iou']}->{rec['support_free_attempt']['iou']} "
            f"saving {rec['base']['saving']}->{rec['support_free_attempt']['saving']} "
            f"unsupported {b['unsupported_ratio']:.4f}/{b['significant_unsupported_island_count']}sig -> "
            f"{o['unsupported_ratio']:.4f}/{o['significant_unsupported_island_count']}sig "
            f"bed {b['bed_contact_area_mm2']:.1f}->{o['bed_contact_area_mm2']:.1f} "
            f"gate={rec['passes_proxy_gate']}",
            flush=True,
        )
    summary = {
        "schema": "support_free_10cm_256_v2",
        "method": "best physical assignment + bed pad inside hull + large-island sparse closure inside hull",
        "support_rule": "voxel is supported if layer z-1 has material in its 3x3 xy neighborhood",
        "significant_island_filter": {
            "min_area_vox": MIN_SIGNIFICANT_AREA_VOX,
            "min_span_mm": MIN_SIGNIFICANT_SPAN_MM,
        },
        "proxy_gate": {
            "iou_min": 0.98,
            "saving_min": 0.45,
            "unsupported_ratio_max": 0.01,
            "significant_unsupported_island_count": 0,
            "bed_contact_area_mm2_min": BED_TARGET_MM2,
        },
        "n_examples": len(rows),
        "n_pass_proxy_gate": int(sum(r["passes_proxy_gate"] for r in rows)),
        "examples": rows,
    }
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
