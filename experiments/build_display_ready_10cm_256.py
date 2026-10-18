#!/usr/bin/env python3
"""Build display-ready 10 cm / 256^3 examples.

Two outputs are produced for each example:
  - shadow_neutral: additions are clipped to the calibrated visual hull, so they
    cannot create projection false positives under the same three-axis model.
  - display_stand: an external low plinth is added for paper photos / handling.
    This is intentionally a presentation artifact, not a shadow-IoU result.
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
CANDIDATE = "carve_pf5"
BENCH = ROOT / "data/silhouettes/bench3"
SRC = ROOT / "results/print_10cm_256"
OUT = ROOT / "results/display_ready_10cm_256_v1"
S2 = np.ones((3, 3), dtype=bool)
S6 = generate_binary_structure(3, 1)

BASE_LAYERS = 5
BASE_DILATE_ITERS = 18
MAX_INTERNAL_ADD_FRAC_OF_HULL = 0.035
MAX_ISLANDS_PER_ROUND = 24
MAX_RIB_ROUNDS = 3


def save_binary_image(mask: np.ndarray, path: Path) -> None:
    img = ((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
    Image.fromarray(img).resize((512, 512), Image.Resampling.NEAREST).save(path)


def support_metrics(volume: np.ndarray) -> dict:
    v = volume.astype(bool)
    voxel_mm = PRINT_MM / float(v.shape[0])
    if not v.any():
        return {
            "unsupported_ratio": 0.0,
            "unsupported_island_count": 0,
            "significant_unsupported_island_count": 0,
            "bed_contact_area_mm2": 0.0,
        }
    z_nonzero = np.where(v)[2]
    z_min = int(z_nonzero.min())
    z_max = int(z_nonzero.max())
    unsupported_total = 0
    islands_total = 0
    significant = 0
    for z in range(z_min + 1, z_max + 1):
        support = binary_dilation(v[:, :, z - 1], structure=S2)
        unsupported = v[:, :, z] & ~support
        cnt = int(unsupported.sum())
        unsupported_total += cnt
        if not cnt:
            continue
        lab, nc = label(unsupported, structure=S2)
        islands_total += int(nc)
        for idx, slc in enumerate(find_objects(lab), start=1):
            if slc is None:
                continue
            comp = lab[slc] == idx
            area = int(comp.sum())
            span_vox = max(slc[0].stop - slc[0].start, slc[1].stop - slc[1].start)
            if area >= 16 or span_vox * voxel_mm >= 4.0:
                significant += 1
    bed_vox = int(v[:, :, z_min].sum())
    return {
        "unsupported_voxels": int(unsupported_total),
        "unsupported_ratio": float(unsupported_total / max(int(v.sum()), 1)),
        "unsupported_island_count": int(islands_total),
        "significant_unsupported_island_count": int(significant),
        "bed_contact_voxels": bed_vox,
        "bed_contact_area_mm2": float(bed_vox * voxel_mm * voxel_mm),
        "z_min": z_min,
        "z_max": z_max,
    }


def add_hull_clipped_base(volume: np.ndarray, hull: np.ndarray) -> tuple[np.ndarray, dict]:
    v = volume.astype(bool).copy()
    h = hull.astype(bool)
    z_min = int(np.where(v)[2].min())
    footprint = v[:, :, z_min].copy()
    for _ in range(BASE_DILATE_ITERS):
        grown = binary_dilation(footprint, structure=S2) & np.any(h[:, :, z_min:z_min + BASE_LAYERS], axis=2)
        if np.array_equal(grown, footprint):
            break
        footprint = grown
    before = int(v.sum())
    added = 0
    for z in range(z_min, min(z_min + BASE_LAYERS, R)):
        add = footprint & h[:, :, z] & ~v[:, :, z]
        v[:, :, z] |= add
        added += int(add.sum())
    return keep_largest(v & h), {
        "base_layers": BASE_LAYERS,
        "base_dilate_iters": BASE_DILATE_ITERS,
        "added_voxels": int(max(int(v.sum()) - before, 0)),
        "raw_added_layer_voxels": int(added),
    }


def support_chain(x: int, y: int, z: int, hull: np.ndarray, volume: np.ndarray) -> list[tuple[int, int, int]] | None:
    path = [(int(x), int(y), int(z))]
    cx, cy, cz = int(x), int(y), int(z)
    z_min = int(np.where(volume)[2].min())
    while cz > z_min:
        x0, x1 = max(cx - 1, 0), min(cx + 2, R)
        y0, y1 = max(cy - 1, 0), min(cy + 2, R)
        if volume[x0:x1, y0:y1, cz - 1].any():
            return path
        candidates = np.argwhere(hull[x0:x1, y0:y1, cz - 1])
        if candidates.size == 0:
            return None
        pts = [(int(x0 + a), int(y0 + b)) for a, b in candidates]
        pts.sort(key=lambda p: abs(p[0] - cx) + abs(p[1] - cy))
        cx, cy = pts[0]
        cz -= 1
        path.append((cx, cy, cz))
    return path


def add_limited_internal_ribs(volume: np.ndarray, hull: np.ndarray, hull_voxels: int) -> tuple[np.ndarray, dict]:
    v = volume.astype(bool).copy()
    h = hull.astype(bool)
    budget = int(round(MAX_INTERNAL_ADD_FRAC_OF_HULL * hull_voxels))
    added = 0
    rounds = []
    for round_idx in range(MAX_RIB_ROUNDS):
        if added >= budget:
            break
        z_nonzero = np.where(v)[2]
        z_min = int(z_nonzero.min())
        z_max = int(z_nonzero.max())
        islands = []
        for z in range(z_min + 1, z_max + 1):
            support = binary_dilation(v[:, :, z - 1], structure=S2)
            unsupported = v[:, :, z] & ~support
            lab, nc = label(unsupported, structure=S2)
            for idx, slc in enumerate(find_objects(lab), start=1):
                if slc is None:
                    continue
                comp_local = lab[slc] == idx
                area = int(comp_local.sum())
                if area < 16:
                    continue
                coords = np.argwhere(comp_local)
                cx = int(round(float(coords[:, 0].mean() + slc[0].start)))
                cy = int(round(float(coords[:, 1].mean() + slc[1].start)))
                islands.append((area, z, cx, cy))
        islands.sort(reverse=True)
        added_round = 0
        tried = 0
        for _, z, cx, cy in islands[:MAX_ISLANDS_PER_ROUND]:
            if added >= budget:
                break
            chain = support_chain(cx, cy, z, h, v)
            tried += 1
            if chain is None:
                continue
            for px, py, pz in chain:
                if added >= budget:
                    break
                if h[px, py, pz] and not v[px, py, pz]:
                    v[px, py, pz] = True
                    added += 1
                    added_round += 1
        rounds.append({"round": round_idx + 1, "tried_islands": int(tried), "added_voxels": int(added_round)})
        if added_round == 0:
            break
    return keep_largest(v & h), {
        "budget_voxels": int(budget),
        "added_voxels": int(added),
        "max_add_frac_of_hull": MAX_INTERNAL_ADD_FRAC_OF_HULL,
        "rounds": rounds,
    }


def add_external_display_stand(volume: np.ndarray) -> np.ndarray:
    v = volume.astype(bool).copy()
    z_min = int(np.where(v)[2].min())
    xs, ys, _ = np.where(v)
    cx = int(round(float(xs.mean())))
    cy = int(round(float(ys.mean())))
    xx, yy = np.meshgrid(np.arange(R), np.arange(R), indexing="ij")
    radius = max(28, int(0.28 * max(xs.max() - xs.min(), ys.max() - ys.min())))
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    footprint = v[:, :, z_min].copy()
    for _ in range(24):
        footprint = binary_dilation(footprint, structure=S2)
    stand = disk | footprint
    z0 = max(z_min - 6, 0)
    z1 = min(z_min + 2, R)
    v[:, :, z0:z1] |= stand[:, :, None]
    return keep_largest(v)


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


def evaluate_variant(volume: np.ndarray, masks: list[np.ndarray], cal: dict, hull_voxels: int) -> dict:
    gt6 = masks + [np.fliplr(m) for m in masks]
    ev = evaluate_tfs_volume(volume.astype(np.float32), gt6, cal, image_size=512)
    try:
        wt = bool(sdf_watertight_mesh(volume.astype(np.uint8), smooth_sigma=0.6).is_watertight)
    except Exception:
        wt = False
    return {
        "iou": round(float(ev["mean_iou"]), 4),
        "saving": round(float(1.0 - int(volume.sum()) / max(hull_voxels, 1)), 4),
        "single": bool(n6(volume) == 1),
        "watertight": wt,
        "erosion_survival": round(erosion_survival(volume), 4),
        "support": support_metrics(volume),
    }


def export_variant(volume: np.ndarray, path: Path) -> dict:
    ehqm.export_marching_cubes(
        volume.astype(bool),
        path,
        smooth_sigma=0.6,
        taubin_iterations=3,
        upsample_factor=1,
    )
    mesh = mesh_clean_scale(path)
    return {
        "stl": str(path),
        "mesh_extents_mm": [round(float(x), 1) for x in mesh.extents],
        "mesh_faces": int(len(mesh.faces)),
    }


def run_one(example: str, names: list[str]) -> dict:
    t0 = time.time()
    paths = [BENCH / f"{name}.png" for name in names]
    sils, _ = load_data_from_list([str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils]
    candidates, aux = make_candidates(
        masks,
        R,
        True,
        print_mm=PRINT_MM,
        nozzle_mm=NOZZLE_MM,
        wall_safety=WALL_SAFETY,
    )
    hull = aux["hull"].astype(bool)
    source_path = SRC / example / f"{example}_voxels256_pf5.npy"
    source = np.load(source_path).astype(bool) if source_path.exists() else candidates[CANDIDATE].astype(bool)
    source &= hull

    with_base, base_info = add_hull_clipped_base(source, hull)
    shadow_neutral, rib_info = add_limited_internal_ribs(with_base, hull, int(hull.sum()))
    display_stand = add_external_display_stand(shadow_neutral)

    d = OUT / example
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "silhouettes").mkdir(parents=True, exist_ok=True)
    for i, (src, name) in enumerate(zip(paths, names)):
        img = np.array(Image.open(src).convert("L"))
        save_binary_image(img > 127, d / "inputs" / f"input{i}_{name}.png")

    variants = {
        "source_pf5": source,
        "shadow_neutral": shadow_neutral,
        "display_stand": display_stand,
    }
    rec_variants = {}
    for variant_name, vol in variants.items():
        metrics = evaluate_variant(vol, masks, aux["calibration"], int(hull.sum()))
        stl_info = export_variant(vol, d / f"{example}_{variant_name}_10cm_256.stl")
        np.save(d / f"{example}_{variant_name}_voxels256.npy", vol.astype(np.uint8))
        ev = evaluate_tfs_volume(vol.astype(np.float32), masks + [np.fliplr(m) for m in masks], aux["calibration"], image_size=512)
        if variant_name != "display_stand":
            for i, name in enumerate(names):
                save_binary_image(ev["pred_masks"][i], d / "silhouettes" / f"{variant_name}_shadow{i}_{name}.png")
        rec_variants[variant_name] = {**metrics, **stl_info}

    return {
        "example": example,
        "silhouettes": names,
        "resolution": R,
        "print_mm": PRINT_MM,
        "base_info": base_info,
        "rib_info": rib_info,
        "variants": rec_variants,
        "sec": round(time.time() - t0, 1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SRC / "MANIFEST.json").read_text())
    rows = []
    for row in manifest["examples"]:
        rec = run_one(row["example"], row["silhouettes"])
        rows.append(rec)
        src = rec["variants"]["source_pf5"]
        sn = rec["variants"]["shadow_neutral"]
        ds = rec["variants"]["display_stand"]
        print(
            f"{rec['example']}: source IoU={src['iou']} saving={src['saving']} bed={src['support']['bed_contact_area_mm2']:.1f}; "
            f"shadow_neutral IoU={sn['iou']} saving={sn['saving']} bed={sn['support']['bed_contact_area_mm2']:.1f} "
            f"uns={sn['support']['unsupported_ratio']:.4f}; display_stand faces={ds['mesh_faces']}",
            flush=True,
        )
    summary = {
        "schema": "display_ready_10cm_256_v1",
        "policy": {
            "shadow_neutral": "add base/ribs only inside the calibrated visual hull; projection IoU is evaluated",
            "display_stand": "adds an external low plinth for paper photos and handling; not used as shadow-IoU evidence",
        },
        "n_examples": len(rows),
        "examples": rows,
    }
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
