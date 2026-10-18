#!/usr/bin/env python3
"""First deterministic support-free attempt for the 10 cm / 256^3 print set.

The physical convention is fixed to the tri-plane display:
  view0 -> floor projection (z axis), view1 -> left wall (x axis),
  view2 -> right wall (y axis).  Build direction is +z.

The optimizer adds only voxels inside the 256^3 visual hull.  Therefore additions
cannot introduce projection overflow under the same calibrated TFS projection.
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
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, label

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
OUT = ROOT / "results/support_free_10cm_256_v1"
S2 = np.ones((3, 3), dtype=bool)
S6 = generate_binary_structure(3, 1)

EXAMPLES = {
    "owl_bat_turtle": ["birds_owl", "x_bat", "reptiles_turtle"],
    "eagle_frog_trex": ["birds_eagle", "reptiles_frog", "x_trex"],
    "parrot_dragon_snail": ["birds_parrot", "x_dragon", "sea_snail"],
    "bat_trex_snail": ["x_bat", "x_trex", "sea_snail"],
}


def save_binary_image(mask: np.ndarray, path: Path) -> None:
    img = ((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
    Image.fromarray(img).resize((512, 512), Image.Resampling.NEAREST).save(path)


def support_metrics(volume: np.ndarray, print_mm: float = PRINT_MM) -> dict:
    """Voxel FDM support proxy for build direction +z.

    A voxel on layer z is supported if any solid voxel in the 3x3 neighborhood on
    layer z-1 exists.  This corresponds to a <=45 degree step in a voxel model.
    """
    v = volume.astype(bool)
    if not v.any():
        return {
            "solid_voxels": 0,
            "z_min": None,
            "z_max": None,
            "unsupported_voxels": 0,
            "unsupported_ratio": 0.0,
            "unsupported_island_count": 0,
            "max_layer_unsupported": 0,
            "bed_contact_voxels": 0,
            "bed_contact_area_mm2": 0.0,
        }
    z_nonzero = np.where(v)[2]
    z_min = int(z_nonzero.min())
    z_max = int(z_nonzero.max())
    unsupported_total = 0
    island_total = 0
    max_layer_unsupported = 0
    for z in range(z_min + 1, z_max + 1):
        support = binary_dilation(v[:, :, z - 1], structure=S2)
        unsupported = v[:, :, z] & ~support
        cnt = int(unsupported.sum())
        unsupported_total += cnt
        max_layer_unsupported = max(max_layer_unsupported, cnt)
        if cnt:
            _, nc = label(unsupported, structure=S2)
            island_total += int(nc)
    solid = int(v.sum())
    bed_vox = int(v[:, :, z_min].sum())
    voxel_mm = float(print_mm) / float(v.shape[0])
    return {
        "solid_voxels": solid,
        "z_min": z_min,
        "z_max": z_max,
        "unsupported_voxels": int(unsupported_total),
        "unsupported_ratio": float(unsupported_total / max(solid, 1)),
        "unsupported_island_count": int(island_total),
        "max_layer_unsupported": int(max_layer_unsupported),
        "bed_contact_voxels": bed_vox,
        "bed_contact_area_mm2": float(bed_vox * voxel_mm * voxel_mm),
    }


def add_support_free_closure(volume: np.ndarray, hull: np.ndarray, max_passes: int = 2) -> tuple[np.ndarray, dict]:
    """Add integrated self-supports inside hull using a 45-degree voxel rule."""
    v = volume.astype(bool).copy()
    h = hull.astype(bool)
    before = int(v.sum())
    pass_records = []

    for pass_idx in range(max_passes):
        added_this_pass = 0
        if not v.any():
            break
        z_nonzero = np.where(v)[2]
        z_min = int(z_nonzero.min())
        z_max = int(z_nonzero.max())
        # High-to-low order propagates supports all the way to the bed in one pass
        # when the hull contains a feasible downward path.
        for z in range(z_max, z_min, -1):
            support = binary_dilation(v[:, :, z - 1], structure=S2)
            unsupported = v[:, :, z] & ~support
            if not unsupported.any():
                continue
            # Prefer a compact 45-degree support footprint in the layer below.
            desired = binary_dilation(unsupported, structure=S2) & h[:, :, z - 1]
            add = desired & ~v[:, :, z - 1]
            if add.any():
                v[:, :, z - 1] |= add
                added_this_pass += int(add.sum())
        pass_records.append({"pass": pass_idx + 1, "added_voxels": int(added_this_pass)})
        if added_this_pass == 0:
            break

    v &= h
    v = keep_largest(v)
    after = int(v.sum())
    return v.astype(bool), {
        "added_voxels": int(max(after - before, 0)),
        "volume_before": before,
        "volume_after": after,
        "passes": pass_records,
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


def run_one(example: str, names: list[str]) -> dict:
    t0 = time.time()
    paths = [BENCH / f"{name}.png" for name in names]
    sils, _ = load_data_from_list(
        [str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False
    )
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
    base = candidates[BASE_CANDIDATE].astype(bool)
    opt, opt_info = add_support_free_closure(base, hull, max_passes=2)

    gt6 = masks + [np.fliplr(m) for m in masks]
    base_eval = evaluate_tfs_volume(base.astype(np.float32), gt6, aux["calibration"], image_size=512)
    opt_eval = evaluate_tfs_volume(opt.astype(np.float32), gt6, aux["calibration"], image_size=512)
    base_support = support_metrics(base)
    opt_support = support_metrics(opt)
    hv = max(int(hull.sum()), 1)

    d = OUT / example
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "silhouettes").mkdir(parents=True, exist_ok=True)
    for i, (src, name) in enumerate(zip(paths, names)):
        img = np.array(Image.open(src).convert("L"))
        save_binary_image(img > 127, d / "inputs" / f"input{i}_{name}.png")
        save_binary_image(opt_eval["pred_masks"][i], d / "silhouettes" / f"shadow{i}_{name}.png")

    np.save(d / f"{example}_support_free_voxels256.npy", opt.astype(np.uint8))
    stl = d / f"{example}_support_free_10cm_256.stl"
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
        "silhouettes_physical_order": {
            "floor_z_view0": names[0],
            "left_wall_x_view1": names[1],
            "right_wall_y_view2": names[2],
        },
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
            "closure": opt_info,
            "mesh_extents_mm": [round(float(x), 1) for x in mesh.extents],
            "mesh_faces": int(len(mesh.faces)),
        },
        "passes_basic_gate": bool(
            n6(opt) == 1
            and wt
            and float(opt_eval["mean_iou"]) >= 0.98
            and opt_support["unsupported_island_count"] == 0
            and opt_support["unsupported_ratio"] <= 0.01
            and opt_support["bed_contact_area_mm2"] >= 100.0
        ),
        "sec": round(time.time() - t0, 1),
    }
    with open(d / "support_free_metrics.json", "w") as f:
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
            f"{example}: IoU {rec['base']['iou']}->{rec['support_free_attempt']['iou']} "
            f"saving {rec['base']['saving']}->{rec['support_free_attempt']['saving']} "
            f"unsupported {b['unsupported_ratio']:.3f}/{b['unsupported_island_count']} -> "
            f"{o['unsupported_ratio']:.3f}/{o['unsupported_island_count']} "
            f"gate={rec['passes_basic_gate']}",
            flush=True,
        )
    summary = {
        "schema": "support_free_10cm_256_v1",
        "physical_order": ["floor_z_view0", "left_wall_x_view1", "right_wall_y_view2"],
        "support_rule": "voxel is supported if layer z-1 has material in its 3x3 xy neighborhood",
        "gate": {
            "iou_min": 0.98,
            "unsupported_island_count": 0,
            "unsupported_ratio_max": 0.01,
            "bed_contact_area_mm2_min": 100.0,
        },
        "n_examples": len(rows),
        "n_pass_basic_gate": int(sum(r["passes_basic_gate"] for r in rows)),
        "examples": rows,
    }
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
