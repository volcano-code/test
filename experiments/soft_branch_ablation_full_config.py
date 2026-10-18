#!/usr/bin/env python3
"""Full-config soft-branch ablation.

This script trains the soft branch from scratch under the requested full
configuration, then compares geometry-only candidates, trained-soft candidates,
and their union. It does not read data/soft_seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
for p in [ROOT, ROOT / "generation", ROOT / "experiments"]:
    sys.path.insert(0, str(p))

from candidate_selection_batch import make_candidates  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from voxel_ops import (  # noqa: E402
    clean_binary_volume,
    evaluate_tfs_volume,
    keep_largest_component,
    make_watertight_printable,
    minimize_material_printable,
    sdf_watertight_mesh,
)

PY = "/home/lin/anaconda3/envs/shadowart-revisited/bin/python"
DEFAULT_OUT = ROOT / "results/soft_branch_ablation_full_heroes_15000_lr0007"
HEROES = ["Spider-Man", "superman", "batman"]
S6 = generate_binary_structure(3, 1)


def is_single(vol: np.ndarray) -> bool:
    return label(np.squeeze(vol).astype(bool), S6)[1] == 1


def is_watertight(vol: np.ndarray) -> bool:
    try:
        return bool(sdf_watertight_mesh(np.squeeze(vol).astype(np.uint8), 0.6).is_watertight)
    except Exception:
        return False


def row(name: str, source: str, vol: np.ndarray, masks, cal, hull_occ: int) -> dict:
    v = keep_largest_component(np.squeeze(vol).astype(bool))
    ev = evaluate_tfs_volume(v.astype(np.float32), masks, cal, image_size=512)
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


def select(rows: list[dict], iou_gate: float) -> dict | None:
    feasible = [
        r for r in rows
        if r["mean_iou"] >= iou_gate and r["single_body"] and r["watertight"]
    ]
    if not feasible:
        return None
    return sorted(feasible, key=lambda r: (-r["saving_vs_hull"], -r["mean_iou"]))[0]


def train_soft(
    files: list[str],
    outdir: Path,
    branch: str,
    niter: int,
    lr: float,
    res: int,
    force: bool,
    vol_min: float,
    conn_w: float,
    loss_export_consistency_scale: float,
    loss_support_scale: float,
    loss_main_structure_scale: float,
    loss_manufacturability_scale: float,
    loss_export_distill_scale: float,
    loss_tfs_projection_scale: float,
    loss_volume_scale: float,
    loss_topo_connectivity_scale: float,
) -> Path:
    outdir = outdir.resolve()
    if force and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    npy = outdir / "sample_0_final_voxels.npy"
    cmd = [
        PY,
        "train.py",
        "cuda:0",
        str(outdir),
        str(niter),
        str(lr),
        "0",
        "-sdlist",
        *files,
        "-mr",
        "1",
        "-tsf",
        "1",
        "-fixedcalib",
        "1" if branch == "fixed" else "0",
        "-res",
        str(res),
        "-volmin",
        str(vol_min),
        "-connw",
        str(conn_w),
        "--loss_export_consistency_scale",
        str(loss_export_consistency_scale),
        "--loss_support_scale",
        str(loss_support_scale),
        "--loss_main_structure_scale",
        str(loss_main_structure_scale),
        "--loss_manufacturability_scale",
        str(loss_manufacturability_scale),
        "--loss_export_distill_scale",
        str(loss_export_distill_scale),
        "--loss_tfs_projection_scale",
        str(loss_tfs_projection_scale),
        "--loss_volume_scale",
        str(loss_volume_scale),
        "--loss_topo_connectivity_scale",
        str(loss_topo_connectivity_scale),
        "--post_export_bundle",
        "0",
    ]
    meta = {
        "branch": branch,
        "command": cmd,
        "num_iters": niter,
        "base_lr": lr,
        "resolution": res,
        "mirror_mode": 1,
        "tsf_mode": 1,
        "fixedcalib": branch == "fixed",
        "vol_min": vol_min,
        "conn_w": conn_w,
        "soft_losses_enabled": {
            "volume_minimization": vol_min > 0.0,
            "topological_connectivity": vol_min > 0.0 and conn_w > 0.0 and loss_topo_connectivity_scale > 0.0,
        },
        "loss_scales": {
            "export_consistency": loss_export_consistency_scale,
            "support": loss_support_scale,
            "main_structure": loss_main_structure_scale,
            "manufacturability": loss_manufacturability_scale,
            "export_distill": loss_export_distill_scale,
            "tfs_projection": loss_tfs_projection_scale,
            "volume": loss_volume_scale,
            "topo_connectivity": loss_topo_connectivity_scale,
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (outdir / "full_config_command.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if npy.exists():
        meta["skipped_existing_trained_output"] = True
        (outdir / "full_config_command.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return npy
    with open(outdir / "train.log", "w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=ROOT / "generation",
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "6", "CUDA_VISIBLE_DEVICES": "0"},
        )
    meta["returncode"] = proc.returncode
    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["output_exists"] = npy.exists()
    (outdir / "full_config_command.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if proc.returncode != 0 or not npy.exists():
        raise RuntimeError(f"{branch} soft training failed; see {outdir / 'train.log'}")
    return npy


def soft_candidates(npy: Path) -> dict[str, np.ndarray]:
    trained = np.squeeze(np.load(npy)).astype(np.float32) > 0.5
    trained = clean_binary_volume(trained, fill_holes=True, closing_iters=1, opening_iters=0, keep_largest=True)
    out = {"soft_trained_clean": trained}
    try:
        soft_min, _ = minimize_material_printable(trained, bridge_radius=1)
        out["soft_min_printable"] = soft_min
    except Exception:
        pass
    try:
        soft_wt, _ = make_watertight_printable(trained, trained)
        out["soft_watertight"] = soft_wt
    except Exception:
        pass
    return {k: keep_largest_component(v) for k, v in out.items() if np.asarray(v).any()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--niter", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=0.007)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--iou_gate", type=float, default=0.95)
    ap.add_argument("--vol_min", type=float, default=0.35)
    ap.add_argument("--conn_w", type=float, default=1.0)
    ap.add_argument("--loss_export_consistency_scale", type=float, default=1.0)
    ap.add_argument("--loss_support_scale", type=float, default=1.0)
    ap.add_argument("--loss_main_structure_scale", type=float, default=1.0)
    ap.add_argument("--loss_manufacturability_scale", type=float, default=1.0)
    ap.add_argument("--loss_export_distill_scale", type=float, default=1.0)
    ap.add_argument("--loss_tfs_projection_scale", type=float, default=1.0)
    ap.add_argument("--loss_volume_scale", type=float, default=1.0)
    ap.add_argument("--loss_topo_connectivity_scale", type=float, default=1.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    files = [str(ROOT / "generation" / f"{name}.png") for name in HEROES]
    masks_t, _ = load_data_from_list(files, True, (args.res, args.res), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in masks_t]

    args.out.mkdir(parents=True, exist_ok=True)
    out = {
        "schema_version": "soft_branch_ablation_full_config_v1",
        "description": "Full training; no data/soft_seeds are read.",
        "config": {
            "num_iters": args.niter,
            "base_lr": args.lr,
            "resolution": args.res,
            "mirror_mode": 1,
            "tsf_mode": 1,
            "branches": ["free", "fixed"],
            "case": "heroes_spiderman_superman_batman",
            "iou_gate": args.iou_gate,
            "vol_min": args.vol_min,
            "conn_w": args.conn_w,
            "topological_connectivity_loss": args.vol_min > 0.0 and args.conn_w > 0.0 and args.loss_topo_connectivity_scale > 0.0,
            "loss_scales": {
                "export_consistency": args.loss_export_consistency_scale,
                "support": args.loss_support_scale,
                "main_structure": args.loss_main_structure_scale,
                "manufacturability": args.loss_manufacturability_scale,
                "export_distill": args.loss_export_distill_scale,
                "tfs_projection": args.loss_tfs_projection_scale,
                "volume": args.loss_volume_scale,
                "topo_connectivity": args.loss_topo_connectivity_scale,
            },
        },
        "cases": [],
    }

    for branch, fix_axis in [("free", False), ("fixed", True)]:
        branch_dir = args.out / branch
        npy = train_soft(
            files,
            branch_dir,
            branch,
            args.niter,
            args.lr,
            args.res,
            args.force,
            args.vol_min,
            args.conn_w,
            args.loss_export_consistency_scale,
            args.loss_support_scale,
            args.loss_main_structure_scale,
            args.loss_manufacturability_scale,
            args.loss_export_distill_scale,
            args.loss_tfs_projection_scale,
            args.loss_volume_scale,
            args.loss_topo_connectivity_scale,
        )
        geom, aux = make_candidates(
            masks,
            args.res,
            fix_axis,
            print_mm=100.0,
            nozzle_mm=0.4,
            wall_safety=2.0,
        )
        cal = aux["calibration"]
        hull_occ = int(np.squeeze(aux["hull"]).sum())
        geom_rows = [row(k, "geometry", v, masks, cal, hull_occ) for k, v in geom.items()]
        soft_rows = [row(k, "soft_full_train", v, masks, cal, hull_occ) for k, v in soft_candidates(npy).items()]
        hybrid_rows = geom_rows + soft_rows
        out["cases"].append({
            "case": f"heroes_{branch}",
            "branch": branch,
            "fix_axis": bool(fix_axis),
            "trained_soft_voxels": str(npy),
            "hull_occupied_voxels": hull_occ,
            "n_geometry_candidates": len(geom_rows),
            "n_soft_candidates": len(soft_rows),
            "geometry_selected": select(geom_rows, args.iou_gate),
            "soft_selected": select(soft_rows, args.iou_gate),
            "hybrid_selected": select(hybrid_rows, args.iou_gate),
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
        "hybrid_selected_soft_count": sum(1 for r in hyb_sel if r["source"] == "soft_full_train"),
        "mean_geometry_saving": float(np.mean([r["saving_vs_hull"] for r in geom_sel])) if geom_sel else None,
        "mean_soft_saving": float(np.mean([r["saving_vs_hull"] for r in soft_sel])) if soft_sel else None,
        "mean_hybrid_saving": float(np.mean([r["saving_vs_hull"] for r in hyb_sel])) if hyb_sel else None,
    }
    out_path = args.out / "soft_branch_ablation_full_config.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps(out["aggregate"], indent=2))


if __name__ == "__main__":
    main()
