#!/usr/bin/env python3
"""Support-aware carving prototype for tri-plane support-free printing.

Carving starts from the fixed-axis visual hull and removes voxels only when all
hard guards pass:
  - not the last coverer of any axis projection pixel;
  - local 6-connectivity remains simple;
  - removing the voxel does not make any voxel on the layer above unsupported.

This is a generation-stage constraint, not post-hoc support insertion.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, find_objects, generate_binary_structure, label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
for p in [ROOT, ROOT / "generation", ROOT / "experiments"]:
    sys.path.insert(0, str(p))

from datasets import load_data_from_list
from voxel_ops import (
    _build_tfs_canonical_planes,
    build_tfs_visual_hull,
    calibrate_tfs_projection,
    evaluate_tfs_volume,
    project_tfs_volume,
    sdf_watertight_mesh,
)
from candidate_selection_batch import make_candidates, n6, keep_largest

R = 64
PRINT_MM = 100.0
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0
BENCH = ROOT / "data/silhouettes/bench3"
OUT = ROOT / "results/support_aware_carve_10cm_64"
S2 = np.ones((3, 3), dtype=bool)
S6 = generate_binary_structure(3, 1)
FACE6 = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

EXAMPLES = {
    "owl_bat_turtle": ["x_bat", "birds_owl", "reptiles_turtle"],
    "eagle_frog_trex": ["x_trex", "birds_eagle", "reptiles_frog"],
    "parrot_dragon_snail": ["x_dragon", "birds_parrot", "sea_snail"],
    "bat_trex_snail": ["x_trex", "x_bat", "sea_snail"],
    "library_ant_frog_parrot": ["bugs_ant", "reptiles_frog", "birds_parrot"],
    "library_ant_peacock_frog": ["bugs_ant", "x_peacock", "reptiles_frog"],
    "library_beetle_owl_frog": ["bugs_beetle", "birds_owl", "reptiles_frog"],
}


def load_masks(names: list[str]) -> list[np.ndarray]:
    paths = [BENCH / f"{name}.png" for name in names]
    sils, _ = load_data_from_list(
        [str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False
    )
    return [np.asarray(x).squeeze().astype(np.float32) for x in sils]


def support_metrics(volume: np.ndarray) -> dict:
    v = volume.astype(bool)
    voxel_mm = PRINT_MM / float(v.shape[0])
    if not v.any():
        return {"unsupported_ratio": 1.0, "unsupported_island_count": 0, "significant_unsupported_island_count": 0, "bed_contact_area_mm2": 0.0}
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
            if area >= 3 or span_vox * voxel_mm >= 4.0:
                significant_islands += 1
    bed_vox = int(v[:, :, z_min].sum())
    return {
        "unsupported_voxels": int(unsupported_total),
        "unsupported_ratio": float(unsupported_total / max(int(v.sum()), 1)),
        "unsupported_island_count": int(island_total),
        "significant_unsupported_island_count": int(significant_islands),
        "bed_contact_area_mm2": float(bed_vox * voxel_mm * voxel_mm),
        "z_min": z_min,
        "z_max": z_max,
    }


def simple_point(occ: np.ndarray, x: int, y: int, z: int) -> bool:
    x0, y0, z0 = max(x - 1, 0), max(y - 1, 0), max(z - 1, 0)
    win = occ[x0:x + 2, y0:y + 2, z0:z + 2].copy()
    cx, cy, cz = x - x0, y - y0, z - z0
    if win.sum() <= 1:
        return False
    win[cx, cy, cz] = False
    lab, _ = label(win, structure=S6)
    face_labels = set()
    for dx, dy, dz in FACE6:
        fx, fy, fz = cx + dx, cy + dy, cz + dz
        if 0 <= fx < win.shape[0] and 0 <= fy < win.shape[1] and 0 <= fz < win.shape[2] and win[fx, fy, fz]:
            face_labels.add(int(lab[fx, fy, fz]))
    return len(face_labels) == 1


def above_support_safe(occ: np.ndarray, x: int, y: int, z: int) -> bool:
    if z + 1 >= occ.shape[2]:
        return True
    x0, x1 = max(x - 1, 0), min(x + 2, occ.shape[0])
    y0, y1 = max(y - 1, 0), min(y + 2, occ.shape[1])
    above = np.argwhere(occ[x0:x1, y0:y1, z + 1])
    if above.size == 0:
        return True
    for ax, ay in above:
        gx, gy = int(x0 + ax), int(y0 + ay)
        sx0, sx1 = max(gx - 1, 0), min(gx + 2, occ.shape[0])
        sy0, sy1 = max(gy - 1, 0), min(gy + 2, occ.shape[1])
        support_count = int(occ[sx0:sx1, sy0:sy1, z].sum())
        if support_count <= 1:
            return False
    return True


def support_score(metrics: dict) -> tuple[int, float]:
    return (
        int(metrics["significant_unsupported_island_count"]),
        float(metrics["unsupported_ratio"]),
    )


def support_aware_carve(hull: np.ndarray, max_passes: int = 20) -> tuple[np.ndarray, dict]:
    occ = hull.astype(bool).copy()
    cnt = {a: occ.sum(axis=a) for a in (0, 1, 2)}
    start = int(occ.sum())
    pass_rows = []
    best_valid = None
    best_any = (support_score(support_metrics(occ)), 0.0, occ.copy(), support_metrics(occ))

    def checkpoint(current: np.ndarray) -> None:
        nonlocal best_valid, best_any
        met = support_metrics(current)
        saving = float(1.0 - int(current.sum()) / max(start, 1))
        score = (support_score(met), -saving)
        if best_any is None or score < (best_any[0], -best_any[1]):
            best_any = (support_score(met), saving, current.copy(), met)
        if met["unsupported_ratio"] <= 0.01 and met["significant_unsupported_island_count"] <= 10:
            if best_valid is None or saving > best_valid[0]:
                best_valid = (saving, current.copy(), met)

    checkpoint(occ)
    for p in range(max_passes):
        removed = 0
        # Higher z first removes overhanging roof material before support columns.
        coords = np.argwhere(occ)
        order = np.lexsort((coords[:, 1], coords[:, 0], -coords[:, 2]))
        for x, y, z in coords[order]:
            x, y, z = int(x), int(y), int(z)
            if not occ[x, y, z]:
                continue
            if cnt[0][y, z] <= 1 or cnt[1][x, z] <= 1 or cnt[2][x, y] <= 1:
                continue
            if not above_support_safe(occ, x, y, z):
                continue
            if not simple_point(occ, x, y, z):
                continue
            occ[x, y, z] = False
            cnt[0][y, z] -= 1
            cnt[1][x, z] -= 1
            cnt[2][x, y] -= 1
            removed += 1
            if removed % 250 == 0:
                checkpoint(occ)
        checkpoint(occ)
        met = support_metrics(occ)
        pass_rows.append({
            "pass": p + 1,
            "removed_voxels": int(removed),
            "remaining_voxels": int(occ.sum()),
            "unsupported_ratio": float(met["unsupported_ratio"]),
            "significant_unsupported_island_count": int(met["significant_unsupported_island_count"]),
        })
        if removed == 0:
            break
    selected_reason = "best_valid_support_checkpoint"
    if best_valid is not None:
        occ = best_valid[1]
    else:
        selected_reason = "best_available_support_score"
        occ = best_any[2]
    occ = keep_largest(occ)
    return occ, {
        "start_voxels": start,
        "final_voxels": int(occ.sum()),
        "removed_voxels": int(start - int(occ.sum())),
        "selected_reason": selected_reason,
        "passes": pass_rows,
    }


def self_support_subset(hull: np.ndarray) -> np.ndarray:
    h = hull.astype(bool)
    out = np.zeros_like(h)
    if not h.any():
        return out
    z_nonzero = np.where(h)[2]
    z_min = int(z_nonzero.min())
    z_max = int(z_nonzero.max())
    out[:, :, z_min] = h[:, :, z_min]
    for z in range(z_min + 1, z_max + 1):
        out[:, :, z] = h[:, :, z] & binary_dilation(out[:, :, z - 1], structure=S2)
    return keep_largest(out)


def support_chain_to_bed(x: int, y: int, z: int, hull: np.ndarray, occ: np.ndarray) -> list[tuple[int, int, int]] | None:
    path = [(int(x), int(y), int(z))]
    cx, cy, cz = int(x), int(y), int(z)
    z_min = int(np.where(hull)[2].min())
    while cz > z_min:
        x0, x1 = max(cx - 1, 0), min(cx + 2, hull.shape[0])
        y0, y1 = max(cy - 1, 0), min(cy + 2, hull.shape[1])
        if occ[x0:x1, y0:y1, cz - 1].any():
            return path
        candidates = np.argwhere(hull[x0:x1, y0:y1, cz - 1])
        if candidates.size == 0:
            return None
        candidates = [(int(x0 + a), int(y0 + b)) for a, b in candidates]
        candidates.sort(key=lambda p: (abs(p[0] - cx) + abs(p[1] - cy), abs(p[0] - x) + abs(p[1] - y)))
        cx, cy = candidates[0]
        cz -= 1
        path.append((cx, cy, cz))
    return path


def repair_projection_self_supported(
    occ: np.ndarray,
    hull: np.ndarray,
    masks: list[np.ndarray],
    cal: dict,
    max_rounds: int = 4,
) -> tuple[np.ndarray, dict]:
    planes = _build_tfs_canonical_planes(masks, occ.shape[0], cal)
    v = occ.astype(bool).copy()
    added = 0
    round_rows = []

    def projection_for_axis(axis: str) -> np.ndarray:
        if axis == "x":
            return v.max(axis=0)
        if axis == "y":
            return v.max(axis=1)
        return v.max(axis=2)

    for round_idx in range(max_rounds):
        added_round = 0
        for axis in ("z", "x", "y"):
            current = projection_for_axis(axis)
            missing = np.argwhere(planes[axis] & ~current)
            for a, b in missing:
                choices = []
                if axis == "x":
                    y, z = int(a), int(b)
                    for x in np.where(hull[:, y, z] & ~v[:, y, z])[0]:
                        chain = support_chain_to_bed(int(x), y, z, hull, v)
                        if chain is not None:
                            choices.append(chain)
                elif axis == "y":
                    x, z = int(a), int(b)
                    for y in np.where(hull[x, :, z] & ~v[x, :, z])[0]:
                        chain = support_chain_to_bed(x, int(y), z, hull, v)
                        if chain is not None:
                            choices.append(chain)
                else:
                    x, y = int(a), int(b)
                    for z in np.where(hull[x, y, :] & ~v[x, y, :])[0]:
                        chain = support_chain_to_bed(x, y, int(z), hull, v)
                        if chain is not None:
                            choices.append(chain)
                if not choices:
                    continue
                choices.sort(key=len)
                for px, py, pz in choices[0]:
                    if not v[px, py, pz]:
                        v[px, py, pz] = True
                        added += 1
                        added_round += 1
        round_rows.append({"round": round_idx + 1, "added_voxels": int(added_round)})
        if added_round == 0:
            break
    v = keep_largest(v)
    return v, {"added_voxels": int(added), "rounds": round_rows}


def eval_volume(vol: np.ndarray, masks: list[np.ndarray], cal: dict, hull_voxels: int) -> dict:
    gt6 = masks + [np.fliplr(m) for m in masks]
    ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, cal, image_size=R)
    try:
        wt = bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight)
    except Exception:
        wt = False
    return {
        "iou": float(ev["mean_iou"]),
        "saving": float(1.0 - int(vol.sum()) / max(hull_voxels, 1)),
        "single": bool(n6(vol) == 1),
        "watertight": wt,
        "support": support_metrics(vol),
    }


def run_one(example: str, names: list[str]) -> dict:
    t0 = time.time()
    masks = load_masks(names)
    cal = calibrate_tfs_projection(masks, search_res=R, fix_axis=True)
    hull, _ = build_tfs_visual_hull(masks, R, calibration=cal)
    cands, aux = make_candidates(masks, R, True, print_mm=PRINT_MM, nozzle_mm=NOZZLE_MM, wall_safety=WALL_SAFETY)
    baseline = cands["carve_pf5"].astype(bool)
    support_aware, carve_info = support_aware_carve(hull)
    subset = self_support_subset(hull)
    repaired, repair_info = repair_projection_self_supported(subset, hull, masks, cal)
    hv = int(hull.sum())
    rec = {
        "example": example,
        "physical_order": {"floor_z_view0": names[0], "left_wall_x_view1": names[1], "right_wall_y_view2": names[2]},
        "resolution": R,
        "hull_voxels": hv,
        "visual_hull": eval_volume(hull, masks, cal, hv),
        "baseline_carve_pf5": eval_volume(baseline, masks, aux["calibration"], hv),
        "support_aware_carve": {
            **eval_volume(support_aware, masks, cal, hv),
            "carve_info": carve_info,
        },
        "self_support_subset": eval_volume(subset, masks, cal, hv),
        "self_support_repaired": {
            **eval_volume(repaired, masks, cal, hv),
            "repair_info": repair_info,
        },
        "sec": round(time.time() - t0, 1),
    }
    s = rec["self_support_repaired"]
    rec["passes_proxy_gate"] = bool(
        s["iou"] >= 0.98
        and s["saving"] >= 0.25
        and s["single"]
        and s["support"]["unsupported_ratio"] <= 0.01
        and s["support"]["significant_unsupported_island_count"] <= 10
    )
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for example, names in EXAMPLES.items():
        rec = run_one(example, names)
        rows.append(rec)
        b = rec["baseline_carve_pf5"]
        s = rec["self_support_repaired"]
        print(
            f"{example}: baseline save={b['saving']:.3f} uns={b['support']['unsupported_ratio']:.3f} "
            f"sig={b['support']['significant_unsupported_island_count']} | "
            f"self-support-repaired save={s['saving']:.3f} IoU={s['iou']:.3f} "
            f"uns={s['support']['unsupported_ratio']:.3f} sig={s['support']['significant_unsupported_island_count']} "
            f"gate={rec['passes_proxy_gate']}",
            flush=True,
        )
    summary = {
        "schema": "support_aware_carve_10cm_64",
        "method": "projection-preserving support-aware carve plus self-support subset and self-supported projection repair",
        "proxy_gate": {
            "iou_min": 0.98,
            "saving_min": 0.25,
            "unsupported_ratio_max": 0.01,
            "significant_unsupported_island_count_max": 10,
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
