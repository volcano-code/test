#!/usr/bin/env python3
"""Low-cost D4 orientation scan for support-friendly tri-plane printing."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, find_objects, label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
for p in [ROOT, ROOT / "generation", ROOT / "experiments"]:
    sys.path.insert(0, str(p))

from datasets import load_data_from_list
from voxel_ops import apply_d4_transform, build_tfs_visual_hull, calibrate_tfs_projection, evaluate_tfs_volume
from candidate_selection_batch import make_candidates, n6

R = 96
PRINT_MM = 100.0
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0
BASE_CANDIDATE = "carve_pf5"
BENCH = ROOT / "data/silhouettes/bench3"
OUT = ROOT / "results/support_free_10cm_96_d4_scan"
S2 = np.ones((3, 3), dtype=bool)
MIN_SIGNIFICANT_AREA_VOX = 4
MIN_SIGNIFICANT_SPAN_MM = 4.0

EXAMPLES = {
    "owl_bat_turtle": ["birds_owl", "x_bat", "reptiles_turtle"],
    "eagle_frog_trex": ["birds_eagle", "reptiles_frog", "x_trex"],
    "parrot_dragon_snail": ["birds_parrot", "x_dragon", "sea_snail"],
    "bat_trex_snail": ["x_bat", "x_trex", "sea_snail"],
}


def load_masks(names: list[str]) -> list[np.ndarray]:
    paths = [BENCH / f"{name}.png" for name in names]
    sils, _ = load_data_from_list([str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False)
    return [np.asarray(x).squeeze().astype(np.float32) for x in sils]


def best_floor_order(example: str, names: list[str]) -> list[str]:
    scan_path = ROOT / "results/support_free_10cm_256_v1/orientation_scan.json"
    if not scan_path.exists():
        return names
    scan = json.loads(scan_path.read_text())
    row = next((r for r in scan if r.get("example") == example), None)
    if row is None:
        return names
    floor = row["best"]["floor_silhouette"]
    return [floor] + [n for n in names if n != floor]


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
    island_total = 0
    significant_islands = 0
    for z in range(z_min + 1, z_max + 1):
        support = binary_dilation(v[:, :, z - 1], structure=S2)
        unsupported = v[:, :, z] & ~support
        cnt = int(unsupported.sum())
        unsupported_total += cnt
        if not cnt:
            continue
        lab, nc = label(unsupported, structure=S2)
        island_total += int(nc)
        for idx, slc in enumerate(find_objects(lab), start=1):
            if slc is None:
                continue
            comp = lab[slc] == idx
            area = int(comp.sum())
            span_vox = max(slc[0].stop - slc[0].start, slc[1].stop - slc[1].start)
            if area >= MIN_SIGNIFICANT_AREA_VOX or span_vox * voxel_mm >= MIN_SIGNIFICANT_SPAN_MM:
                significant_islands += 1
    bed_vox = int(v[:, :, z_min].sum())
    return {
        "unsupported_ratio": float(unsupported_total / max(int(v.sum()), 1)),
        "unsupported_island_count": int(island_total),
        "significant_unsupported_island_count": int(significant_islands),
        "bed_contact_area_mm2": float(bed_vox * voxel_mm * voxel_mm),
        "z_min": z_min,
        "z_max": z_max,
    }


def run_one(example: str, names: list[str]) -> dict:
    ordered = best_floor_order(example, names)
    base_masks = load_masks(ordered)
    hull_trials = []
    for ops in itertools.product(range(8), repeat=3):
        masks = [apply_d4_transform(base_masks[i], ops[i]).astype(np.float32) for i in range(3)]
        cal = calibrate_tfs_projection(masks, search_res=R, fix_axis=True)
        hull, _ = build_tfs_visual_hull(masks, R, calibration=cal)
        met = support_metrics(hull)
        hull_trials.append({
            "ops": list(ops),
            **met,
        })
    hull_trials.sort(key=lambda r: (
        r["significant_unsupported_island_count"],
        r["unsupported_ratio"],
        -r["bed_contact_area_mm2"],
    ))

    carve_trials = []
    for trial in hull_trials[:8]:
        ops = trial["ops"]
        masks = [apply_d4_transform(base_masks[i], ops[i]).astype(np.float32) for i in range(3)]
        candidates, aux = make_candidates(
            masks,
            R,
            True,
            print_mm=PRINT_MM,
            nozzle_mm=NOZZLE_MM,
            wall_safety=WALL_SAFETY,
        )
        vol = candidates[BASE_CANDIDATE].astype(bool)
        gt6 = masks + [np.fliplr(m) for m in masks]
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=R)
        met = support_metrics(vol)
        hv = max(int(aux["hull"].sum()), 1)
        carve_trials.append({
            "ops": ops,
            "mean_iou": float(ev["mean_iou"]),
            "saving": float(1.0 - int(vol.sum()) / hv),
            "single": bool(n6(vol) == 1),
            **met,
        })
    carve_trials.sort(key=lambda r: (
        r["significant_unsupported_island_count"],
        r["unsupported_ratio"],
        -r["bed_contact_area_mm2"],
        -r["saving"],
    ))
    return {
        "example": example,
        "physical_order": {
            "floor_z_view0": ordered[0],
            "left_wall_x_view1": ordered[1],
            "right_wall_y_view2": ordered[2],
        },
        "best_hull_trials": hull_trials[:8],
        "best_carve_trials": carve_trials,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for example, names in EXAMPLES.items():
        rec = run_one(example, names)
        rows.append(rec)
        best = rec["best_carve_trials"][0]
        print(
            f"{example}: floor={rec['physical_order']['floor_z_view0']} ops={best['ops']} "
            f"IoU={best['mean_iou']:.4f} saving={best['saving']:.4f} "
            f"unsupported={best['unsupported_ratio']:.4f} sig={best['significant_unsupported_island_count']} "
            f"bed={best['bed_contact_area_mm2']:.1f} single={best['single']}",
            flush=True,
        )
    summary = {
        "schema": "support_free_10cm_96_d4_scan",
        "resolution": R,
        "candidate": BASE_CANDIDATE,
        "support_rule": "voxel is supported if layer z-1 has material in its 3x3 xy neighborhood",
        "significant_island_filter": {
            "min_area_vox": MIN_SIGNIFICANT_AREA_VOX,
            "min_span_mm": MIN_SIGNIFICANT_SPAN_MM,
        },
        "examples": rows,
    }
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
