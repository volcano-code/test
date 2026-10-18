#!/usr/bin/env python3
"""Run the original ShadowArt-Revisited voxel pipeline for Table 1 cases."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import generate_binary_structure, label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SAR = Path("/data/xvxinyao/ShadowArt-Revisited")
PY = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")
VOXEL = SAR / "voxel"
RESULT_ROOT = SAR / "voxel_results" / "table1_shadowforge_original_sadekar_20260705"
OUT = ROOT / "results" / "baseline_table1_sadekar_original_pipeline_20260705"
GALLERY = ROOT / "results" / "gallery_FINAL"

CASES = [
    "dragon_peacock_trex",
    "parrot_squid_scorpion",
    "owl_bat_snail",
    "heroes_spiderman_superman_batman",
]


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore") if path.exists() else ""

    def find(label: str):
        m = re.search(rf"{re.escape(label)}\s*:\s*([-+0-9.eE]+)", text)
        return float(m.group(1)) if m else None

    return {
        "mean_iou": find("IOU metric"),
        "mean_dice": find("Dice metric"),
        "ms_ssim_metric": find("MISSIM metric"),
    }


def binary(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def recompute_saved_metrics(run_dir: Path) -> dict:
    ious, dices = [], []
    for gt_path in sorted(run_dir.glob("sample_0view_*.png")):
        if gt_path.name.endswith("_silh.png"):
            continue
        pred_path = gt_path.with_name(gt_path.stem + "_silh.png")
        if not pred_path.exists():
            continue
        gt, pred = binary(gt_path), binary(pred_path)
        inter = np.logical_and(gt, pred).sum()
        union = np.logical_or(gt, pred).sum()
        denom = gt.sum() + pred.sum()
        ious.append(float(inter / union) if union else 1.0)
        dices.append(float(2 * inter / denom) if denom else 1.0)
    return {
        "mean_iou": float(np.mean(ious)) if ious else None,
        "mean_dice": float(np.mean(dices)) if dices else None,
        "min_iou": float(np.min(ious)) if ious else None,
        "view_ious": ious,
        "view_dices": dices,
        "metric_source": "saved_sample_view_pngs",
    }


def volume_stats(run_dir: Path) -> dict:
    p = run_dir / "sample_0_final_voxels.npy"
    arr = np.load(p)
    arr = np.squeeze(arr) > 0.05
    occ = int(arr.sum())
    _, ncomp = label(arr, structure=generate_binary_structure(3, 1))
    return {
        "occupied_voxels": occ,
        "component_count_6": int(ncomp),
        "single_body": bool(ncomp == 1),
    }


def run_case(case: str) -> dict:
    case_out_name = f"table1_shadowforge_original_sadekar_20260705/{case}"
    run_dir = SAR / "voxel_results" / case_out_name
    shadow_files = [GALLERY / case / "inputs" / f"input{i}.png" for i in range(3)]
    if not all(p.exists() for p in shadow_files):
        raise FileNotFoundError(case)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT / f"{case}.log"
    if not (run_dir / "sample_0_final_voxels.npy").exists():
        cmd = [
            str(PY),
            "val.py",
            "cuda:0",
            case_out_name,
            "600",
            "0.01",
            "-swt",
            "10.0",
            "-l1wt",
            "10.0",
            "-mwt",
            "0.0",
            "-th",
            "0.05",
            "-zd",
            "1.7",
            "-imsz",
            "256",
            "-mr",
            "false",
            "-tsf",
            "true",
            "-sdlist",
            *[str(p) for p in shadow_files],
        ]
        OUT.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as f:
            subprocess.run(cmd, cwd=str(VOXEL), check=True, stdout=f, stderr=subprocess.STDOUT)
    metrics = parse_log(run_dir / "log.txt")
    fixed = recompute_saved_metrics(run_dir)
    stats = volume_stats(run_dir)
    return {
        "case": case,
        "run_dir": str(run_dir),
        "log": str(log_path),
        "metrics_from_log": metrics,
        "metrics_recomputed": fixed,
        **stats,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = []
    for case in CASES:
        row = run_case(case)
        rows.append(row)
        print(case, row["metrics_recomputed"], "components", row["component_count_6"], flush=True)
        (OUT / "partial_results.json").write_text(json.dumps(rows, indent=2))
    vals = [r["metrics_recomputed"] for r in rows]
    aggregate = {
        "method": "Sadekar original voxel pipeline",
        "cases": len(rows),
        "mean_iou": float(np.mean([v["mean_iou"] for v in vals])),
        "mean_dice": float(np.mean([v["mean_dice"] for v in vals])),
        "worst_iou": float(np.min([v["min_iou"] for v in vals])),
        "single_body": f"{sum(r['single_body'] for r in rows)}/{len(rows)}",
        "components_mean": float(np.mean([r["component_count_6"] for r in rows])),
        "elapsed_s": round(time.time() - started, 2),
    }
    out = {
        "schema": "sadekar_original_table1_v1",
        "pipeline": str(VOXEL / "val.py"),
        "command_protocol": {
            "Niter": 600,
            "lr": 0.01,
            "tsf": True,
            "mirror": False,
            "img_size": 256,
            "threshold": 0.05,
        },
        "aggregate": aggregate,
        "rows": rows,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(aggregate, indent=2))
    print("saved", OUT)


if __name__ == "__main__":
    main()
