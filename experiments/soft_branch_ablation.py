"""Soft-branch ablation for ShadowForge candidate generation.

Compares geometry-only candidates, soft-derived candidates, and their union on
the existing heroes free/fixed soft seeds. This does not train a new model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generation"))
sys.path.insert(0, str(ROOT / "experiments"))

from candidate_selection_batch import make_candidates  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from voxel_ops import (  # noqa: E402
    calibrate_tfs_projection,
    clean_binary_volume,
    evaluate_tfs_volume,
    keep_largest_component,
    make_watertight_printable,
    minimize_material_printable,
    sdf_watertight_mesh,
)


S6 = generate_binary_structure(3, 1)
HEROES = ["Spider-Man", "superman", "batman"]
SOFT = {
    "free": ROOT / "data/soft_seeds/heroes_free_final_voxels.npy",
    "fixed": ROOT / "data/soft_seeds/heroes_fixed_final_voxels.npy",
}


def is_single(vol: np.ndarray) -> bool:
    return label(np.squeeze(vol).astype(bool), S6)[1] == 1


def is_watertight(vol: np.ndarray) -> bool:
    try:
        return bool(sdf_watertight_mesh(np.squeeze(vol).astype(np.uint8), 0.6).is_watertight)
    except Exception:
        return False


def row(name: str, source: str, vol: np.ndarray, masks, cal, hull_occ: int) -> dict:
    v = keep_largest_component(np.squeeze(vol).astype(bool))
    ev = evaluate_tfs_volume(v.astype(np.float32), masks, cal, image_size=256)
    occ = int(v.sum())
    return {
        "candidate": name,
        "source": source,
        "occupied_voxels": occ,
        "saving_vs_hull": float(1.0 - occ / max(hull_occ, 1)),
        "mean_iou": float(ev["mean_iou"]),
        "min_iou": float(min(ev["view_ious"])),
        "single_body": bool(is_single(v)),
        "watertight": bool(is_watertight(v)),
    }


def select(rows: list[dict], iou_gate: float = 0.95) -> dict | None:
    feasible = [
        r for r in rows
        if r["mean_iou"] >= iou_gate and r["single_body"] and r["watertight"]
    ]
    if not feasible:
        return None
    return sorted(feasible, key=lambda r: (-r["saving_vs_hull"], -r["mean_iou"]))[0]


def soft_candidates(seed_path: Path) -> dict[str, np.ndarray]:
    seed = np.squeeze(np.load(seed_path)).astype(np.float32) > 0.5
    seed = clean_binary_volume(seed, fill_holes=True, closing_iters=1, opening_iters=0, keep_largest=True)
    out = {"soft_seed_clean": seed}
    try:
        soft_min, _ = minimize_material_printable(seed, bridge_radius=1)
        out["soft_min_printable"] = soft_min
    except Exception:
        pass
    try:
        soft_wt, _ = make_watertight_printable(seed, seed)
        out["soft_watertight"] = soft_wt
    except Exception:
        pass
    return {k: keep_largest_component(v) for k, v in out.items() if np.asarray(v).any()}


def main() -> None:
    files = [str(ROOT / "generation" / f"{name}.png") for name in HEROES]
    masks_t, _ = load_data_from_list(files, True, (128, 128), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in masks_t]

    out = {
        "schema_version": "soft_branch_ablation_v1",
        "description": "Existing heroes soft seeds; no retraining.",
        "cases": [],
    }

    for branch, fix_axis in [("free", False), ("fixed", True)]:
        geom, aux = make_candidates(masks, 128, fix_axis)
        cal = aux["calibration"]
        # Soft seeds may come from a free differentiable optimization branch, so
        # evaluate them under their own best free TFS calibration as well.
        soft_cal = calibrate_tfs_projection(masks, search_res=96, fix_axis=False)
        hull_occ = int(np.squeeze(aux["hull"]).sum())

        geom_rows = [row(k, "geometry", v, masks, cal, hull_occ) for k, v in geom.items()]
        soft_rows = [row(k, "soft", v, masks, soft_cal, hull_occ) for k, v in soft_candidates(SOFT[branch]).items()]
        hybrid_rows = geom_rows + soft_rows

        out["cases"].append({
            "case": f"heroes_{branch}",
            "branch": branch,
            "fix_axis": bool(fix_axis),
            "hull_occupied_voxels": hull_occ,
            "n_geometry_candidates": len(geom_rows),
            "n_soft_candidates": len(soft_rows),
            "geometry_selected": select(geom_rows),
            "soft_selected": select(soft_rows),
            "hybrid_selected": select(hybrid_rows),
            "all_candidates": sorted(hybrid_rows, key=lambda r: (r["source"], r["candidate"])),
        })

    geom_sel = [c["geometry_selected"] for c in out["cases"] if c["geometry_selected"]]
    soft_sel = [c["soft_selected"] for c in out["cases"] if c["soft_selected"]]
    hyb_sel = [c["hybrid_selected"] for c in out["cases"] if c["hybrid_selected"]]
    out["aggregate"] = {
        "n_cases": len(out["cases"]),
        "geometry_pass": len(geom_sel),
        "soft_pass": len(soft_sel),
        "hybrid_pass": len(hyb_sel),
        "hybrid_selected_soft_count": sum(1 for r in hyb_sel if r["source"] == "soft"),
        "mean_geometry_saving": float(np.mean([r["saving_vs_hull"] for r in geom_sel])) if geom_sel else None,
        "mean_soft_saving": float(np.mean([r["saving_vs_hull"] for r in soft_sel])) if soft_sel else None,
        "mean_hybrid_saving": float(np.mean([r["saving_vs_hull"] for r in hyb_sel])) if hyb_sel else None,
    }

    out_path = ROOT / "results/soft_branch_ablation_heroes.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps(out["aggregate"], indent=2))


if __name__ == "__main__":
    main()
