#!/usr/bin/env python3
"""Run Mitra and Sadekar mesh/voxel baselines on latest anime30 fixed examples."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.ndimage import generate_binary_structure, label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SAR = Path("/data/xvxinyao/ShadowArt-Revisited")
PY = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")
OUT = ROOT / "results/anime30_baselines_mitra_sadekar_20260707"
SHADOWFORGE = ROOT / "results/anime30_diverse_fixed_50_60_10cm_256_20260707/MANIFEST.json"

sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]
from candidate_selection_batch import make_candidates
from datasets import load_data_from_list
from mitra_baseline import build_mitra
from voxel_ops import apply_d4_transform, compute_dice_np, compute_iou_np, resize_binary_mask, sdf_watertight_mesh

RES = 256
IMG_SIZE = 256
DEVICE = "cuda:0"
S6 = generate_binary_structure(3, 1)
CASES = [
    "diverse1_close1_05_08_13",
    "diverse3_thick1_09_28_29",
    "diverse4_close2_08_14_30",
    "diverse5_thick1_13_21_23",
]


def load_manifest_cases() -> dict[str, dict]:
    payload = json.loads(SHADOWFORGE.read_text())
    return {row["example"]: row for row in payload["examples"]}


def load_masks(paths: list[str]) -> list[np.ndarray]:
    masks, _ = load_data_from_list(paths, True, (RES, RES), torch.device("cpu"), 3, False)
    # Keep grayscale masks until resize_binary_mask/evaluation, which applies
    # the project-standard >127 threshold. Casting here would wrongly include
    # anti-aliased boundary pixels as foreground.
    return [np.asarray(x).squeeze().astype(np.float32) for x in masks[:3]]


def project3(volume: np.ndarray, calibration: dict) -> list[np.ndarray]:
    axes = {"x": 0, "y": 1, "z": 2}
    preds = []
    for i in range(3):
        axis_name = calibration["axis_for_view"].get(str(i), calibration["axis_for_view"].get(i))
        op_value = calibration["ops"].get(str(i), calibration["ops"].get(i))
        proj = np.max(volume.astype(bool), axis=axes[axis_name])
        proj = apply_d4_transform(proj, int(op_value))
        preds.append(resize_binary_mask(proj, IMG_SIZE))
    return preds


def binary_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((np.asarray(mask).astype(bool).astype(np.uint8) * 255)).save(path)


def score_masks(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    gts_r = [resize_binary_mask(m, IMG_SIZE) for m in gts]
    preds_r = [resize_binary_mask(p, IMG_SIZE) for p in preds]
    ious = [compute_iou_np(p, g) for p, g in zip(preds_r, gts_r)]
    dices = [compute_dice_np(p, g) for p, g in zip(preds_r, gts_r)]
    return {
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "min_iou": float(np.min(ious)),
        "view_ious": [float(x) for x in ious],
        "view_dices": [float(x) for x in dices],
    }


def voxel_qa(volume: np.ndarray, hull_voxels: int | None = None) -> dict:
    vol = np.asarray(volume).astype(bool)
    lab, cc = label(vol, structure=S6)
    sizes = np.bincount(lab.ravel())[1:]
    largest = int(sizes.max()) if len(sizes) else 0
    try:
        watertight = bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight)
    except Exception:
        watertight = False
    out = {
        "occupied_voxels": int(vol.sum()),
        "component_count_6": int(cc),
        "largest_component_ratio": float(largest / max(int(vol.sum()), 1)),
        "single_body": bool(cc == 1),
        "watertight": watertight,
    }
    if hull_voxels is not None:
        out["saving_vs_hull"] = float(1.0 - int(vol.sum()) / max(int(hull_voxels), 1))
    return out


def mesh_qa(path: Path) -> dict:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.merge_vertices()
    mesh.process()
    parts = mesh.split(only_watertight=False)
    return {
        "mesh_path": str(path),
        "single_body": bool(len(parts) == 1),
        "watertight": bool(mesh.is_watertight),
        "mesh_components": int(len(parts)),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "mesh_volume": float(mesh.volume) if mesh.is_watertight else None,
    }


def official_preds(run_dir: Path, pattern: str) -> list[np.ndarray]:
    return [binary_image(run_dir / pattern.format(i=i)) for i in range(3)]


def parse_log_metrics(path: Path) -> dict:
    text = path.read_text(errors="ignore") if path.exists() else ""
    out = {}
    for key, pattern in {
        "log_iou": r"IOU(?: metric)?\s*:\s*([-+0-9.eE]+)",
        "log_dice": r"Dice(?: metric)?\s*:\s*([-+0-9.eE]+)",
        "normal_loss": r"Normal loss\s*:\s*(?:tensor\()?([-+0-9.eE]+)",
    }.items():
        m = re.search(pattern, text)
        out[key] = float(m.group(1)) if m else None
    return out


def run_sadekar_voxel(case: str, paths: list[str]) -> tuple[Path, Path]:
    sub = f"anime30_baselines_20260707/voxel/{case}"
    run_dir = SAR / "voxel_results" / sub
    log = OUT / case / "sadekar_voxel_run.log"
    if not (run_dir / "sample_0_final_voxels.npy").exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(PY), "val.py", DEVICE, sub, "600", "0.01",
            "-swt", "10.0", "-l1wt", "10.0", "-mwt", "0.0",
            "-th", "0.05", "-zd", "1.7", "-imsz", str(IMG_SIZE),
            "-mr", "true", "-tsf", "true", "-sdlist", *paths,
        ]
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            subprocess.run(cmd, cwd=str(SAR / "voxel"), check=True, stdout=f, stderr=subprocess.STDOUT)
    return run_dir, log


def run_sadekar_mesh(case: str, paths: list[str]) -> tuple[Path, Path]:
    sub = f"anime30_baselines_20260707/mesh/{case}"
    run_dir = SAR / "mesh_results" / sub
    log = OUT / case / "sadekar_mesh_run.log"
    if not (run_dir / "sample_0_output.obj").exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(PY), "val.py", DEVICE, sub, "600", "0.01", "0",
            "-swt", "1.6", "-l1wt", "1.6", "-mwt", "0.0",
            "-ewt", "1.2", "-nwt", "0.03", "-lwt", "1.2",
            "-imsz", str(IMG_SIZE), "-mr", "true", "-tsf", "true",
            "-sdlist", *paths,
        ]
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            subprocess.run(cmd, cwd=str(SAR / "mesh"), check=True, stdout=f, stderr=subprocess.STDOUT)
    return run_dir, log


def run_case(case: str, rec: dict) -> dict:
    t0 = time.time()
    case_out = OUT / case
    case_out.mkdir(parents=True, exist_ok=True)
    paths = rec["input_paths"]
    masks = load_masks(paths)
    _, aux = make_candidates(masks, RES, True, print_mm=100.0, nozzle_mm=0.4, wall_safety=2.0)
    cal = aux["calibration"]
    hull_voxels = int(aux["hull"].sum())

    result = {
        "case": case,
        "silhouettes": rec["silhouettes"],
        "protocol": {
            "resolution": RES,
            "tsf": True,
            "mr": True,
            "shadowforge_fixed": bool(rec["fixed"]),
            "baseline_device": DEVICE,
        },
        "methods": {},
    }

    mitra_vol, mitra_info = build_mitra(masks, res=RES, fix_axis=True)
    mitra_preds = project3(mitra_vol, cal)
    np.save(case_out / "mitra_voxels256.npy", mitra_vol.astype(np.uint8))
    for i, pred in enumerate(mitra_preds):
        save_mask(pred, case_out / f"mitra_view{i}.png")
    result["methods"]["Mitra-style carving"] = {
        **score_masks(mitra_preds, masks),
        **voxel_qa(mitra_vol, hull_voxels),
        "build_info": mitra_info,
    }

    vox_dir, vox_log = run_sadekar_voxel(case, paths)
    voxel_preds = official_preds(vox_dir, "sample_0view_{i}_silh.png")
    voxel_arr = np.squeeze(np.load(vox_dir / "sample_0_final_voxels.npy")) > 0.05
    for i, pred in enumerate(voxel_preds):
        save_mask(pred, case_out / f"sadekar_voxel_view{i}.png")
    result["methods"]["Sadekar voxel"] = {
        **score_masks(voxel_preds, masks),
        **voxel_qa(voxel_arr, hull_voxels),
        "run_dir": str(vox_dir),
        "log": str(vox_log),
        **parse_log_metrics(vox_dir / "log.txt"),
    }

    mesh_dir, mesh_log = run_sadekar_mesh(case, paths)
    mesh_preds = official_preds(mesh_dir, "sample_0_pred_view_{i}.png")
    for i, pred in enumerate(mesh_preds):
        save_mask(pred, case_out / f"sadekar_mesh_view{i}.png")
    result["methods"]["Sadekar mesh"] = {
        **score_masks(mesh_preds, masks),
        **mesh_qa(mesh_dir / "sample_0_output.obj"),
        "run_dir": str(mesh_dir),
        "log": str(mesh_log),
        **parse_log_metrics(mesh_dir / "log.txt"),
    }

    sf_vol = np.load(rec["selected"]["voxels"]).astype(bool)
    sf_preds = project3(sf_vol, cal)
    result["methods"]["ShadowForge"] = {
        **score_masks(sf_preds, masks),
        **voxel_qa(sf_vol, hull_voxels),
        "stl": rec["selected"]["stl"],
        "source_manifest": str(SHADOWFORGE),
    }

    for i, gt in enumerate(masks):
        save_mask(gt, case_out / f"target_view{i}.png")
    result["elapsed_s"] = round(time.time() - t0, 2)
    (case_out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = load_manifest_cases()
    rows = []
    t0 = time.time()
    for case in CASES:
        row = run_case(case, cases[case])
        rows.append(row)
        print(json.dumps({
            "case": case,
            "elapsed_s": row["elapsed_s"],
            "metrics": {
                k: {
                    "mean_iou": round(v["mean_iou"], 4),
                    "mean_dice": round(v["mean_dice"], 4),
                    "single_body": v.get("single_body"),
                    "watertight": v.get("watertight"),
                    "saving": round(v.get("saving_vs_hull", 0.0), 4) if "saving_vs_hull" in v else None,
                }
                for k, v in row["methods"].items()
            },
        }, ensure_ascii=False), flush=True)
        (OUT / "partial_results.json").write_text(json.dumps(rows, indent=2))

    methods = list(rows[0]["methods"].keys())
    aggregate = []
    for method in methods:
        vals = [row["methods"][method] for row in rows]
        aggregate.append({
            "method": method,
            "cases": len(vals),
            "mean_iou": float(np.mean([v["mean_iou"] for v in vals])),
            "mean_dice": float(np.mean([v["mean_dice"] for v in vals])),
            "worst_iou": float(np.min([v["min_iou"] for v in vals])),
            "single_body": f"{sum(bool(v.get('single_body')) for v in vals)}/{len(vals)}",
            "watertight": f"{sum(bool(v.get('watertight')) for v in vals)}/{len(vals)}",
            "mean_saving_vs_hull": (
                float(np.mean([v["saving_vs_hull"] for v in vals if "saving_vs_hull" in v]))
                if any("saving_vs_hull" in v for v in vals) else None
            ),
        })

    out = {
        "schema": "anime30_latest_mitra_sadekar_baselines_v1",
        "source_shadowforge_manifest": str(SHADOWFORGE),
        "cases": CASES,
        "protocol": {"resolution": RES, "tsf": True, "mr": True, "fixed_targets": True},
        "elapsed_total_s": round(time.time() - t0, 2),
        "aggregate": aggregate,
        "rows": rows,
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    with (OUT / "summary.tsv").open("w") as f:
        f.write("method\tcases\tmean_iou\tmean_dice\tworst_iou\tsingle_body\twatertight\tmean_saving_vs_hull\n")
        for r in aggregate:
            f.write("\t".join(str(r[k]) for k in ["method", "cases", "mean_iou", "mean_dice", "worst_iou", "single_body", "watertight", "mean_saving_vs_hull"]) + "\n")
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
