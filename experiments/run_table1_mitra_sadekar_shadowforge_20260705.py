#!/usr/bin/env python3
"""Run Mitra-style and Sadekar-style baselines for the compact Table 1 draft.

This script is intentionally paper-facing but non-invasive: it writes artifacts
under results/ and does not edit the manuscript.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.ndimage import generate_binary_structure, label

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
WIRE = Path("/data/xvxinyao/xvxinyao/total_project/shadow_wire_fusion")
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from candidate_selection_batch import make_candidates, n6
from datasets import load_data_from_list
from mitra_baseline import build_mitra
from voxel_ops import (
    apply_d4_transform,
    build_tfs_visual_hull,
    compute_dice_np,
    compute_iou_np,
    resize_binary_mask,
    sdf_watertight_mesh,
)


OUT = ROOT / "results" / "baseline_table1_sadekar_style_20260705"
CASES = [
    "dragon_peacock_trex",
    "parrot_squid_scorpion",
    "owl_bat_snail",
    "heroes_spiderman_superman_batman",
]
GALLERY = ROOT / "results" / "gallery_FINAL"
SAD_SCRIPT = WIRE / "sadekar_baseline.py"
RES = 48
IMG_SIZE = 256


def load_masks(paths: list[Path], res: int) -> list[np.ndarray]:
    masks, _ = load_data_from_list([str(p) for p in paths], True, (res, res), torch.device("cpu"), 3, False)
    return [np.asarray(x).squeeze().astype(bool) for x in masks]


def save_mask(mask: np.ndarray, path: Path, size: int = IMG_SIZE) -> None:
    arr = (mask.astype(np.uint8) * 255)
    Image.fromarray(arr).resize((size, size), Image.Resampling.NEAREST).save(path)


def project3(volume: np.ndarray, calibration: dict, image_size: int) -> list[np.ndarray]:
    axes = {"x": 0, "y": 1, "z": 2}
    out = []
    for i in range(3):
        axis_name = calibration["axis_for_view"].get(str(i), calibration["axis_for_view"].get(i))
        op_value = calibration["ops"].get(str(i), calibration["ops"].get(i))
        axis = axes[axis_name]
        op = int(op_value)
        proj = np.max(volume.astype(bool), axis=axis)
        proj = apply_d4_transform(proj, op)
        out.append(resize_binary_mask(proj, image_size))
    return out


def score_volume(volume: np.ndarray, gt_masks: list[np.ndarray], calibration: dict, hull_voxels: int) -> dict:
    preds = project3(volume, calibration, IMG_SIZE)
    gts = [resize_binary_mask(m, IMG_SIZE) for m in gt_masks]
    ious = [compute_iou_np(p, g) for p, g in zip(preds, gts)]
    dices = [compute_dice_np(p, g) for p, g in zip(preds, gts)]
    labeled, cc = label(volume.astype(bool), structure=generate_binary_structure(3, 1))
    sizes = np.bincount(labeled.ravel())[1:]
    largest = int(sizes.max()) if len(sizes) else 0
    occ = int(volume.sum())
    try:
        wt = bool(sdf_watertight_mesh(volume.astype(np.uint8), smooth_sigma=0.6).is_watertight)
    except Exception:
        wt = False
    return {
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "min_iou": float(np.min(ious)),
        "view_ious": [float(x) for x in ious],
        "view_dices": [float(x) for x in dices],
        "occupied_voxels": occ,
        "material_saving_vs_hull": float(1.0 - occ / max(hull_voxels, 1)),
        "component_count_6": int(cc),
        "largest_component_ratio": float(largest / max(occ, 1)),
        "single_body": bool(cc == 1),
        "watertight_proxy": wt,
        "printable_proxy": bool(cc == 1 and wt),
        "pred_masks": preds,
    }


def load_ours(case_dir: Path) -> np.ndarray:
    files = sorted(case_dir.glob("*_voxels48.npy"))
    if not files:
        raise FileNotFoundError(f"missing ShadowForge voxels in {case_dir}")
    arr = np.load(files[0])
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(bool)


def run_sadekar(case: str, case_out: Path, input_paths: list[Path], calib_path: Path, hull_path: Path) -> np.ndarray:
    out = case_out / "sadekar"
    out.mkdir(parents=True, exist_ok=True)
    done = out / "summary.json"
    if not done.exists():
        cmd = [
            sys.executable,
            str(SAD_SCRIPT),
            "--voxels",
            str(hull_path),
            "--views",
            *[str(p) for p in input_paths],
            "--calib",
            str(calib_path),
            "--out_dir",
            str(out),
            "--n",
            str(RES),
            "--iters",
            "400",
            "--lr",
            "0.1",
            "--device",
            "cpu",
        ]
        log = out / "run.log"
        with log.open("w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    arr = np.load(out / "recon_voxels.npy")
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(bool)


def make_contact_sheet(rows: list[dict]) -> None:
    methods = ["Input", "Mitra-style", "Sadekar-style", "ShadowForge"]
    n_cases = len(rows)
    fig, axes = plt.subplots(n_cases * 3, len(methods), figsize=(8.5, 2.35 * n_cases))
    if n_cases == 1:
        axes = np.expand_dims(axes, 0)
    for r_idx, row in enumerate(rows):
        for v in range(3):
            rr = r_idx * 3 + v
            imgs = [
                row["gt_masks"][v],
                row["methods"]["Mitra-style"]["pred_masks"][v],
                row["methods"]["Sadekar-style"]["pred_masks"][v],
                row["methods"]["ShadowForge"]["pred_masks"][v],
            ]
            for c, img in enumerate(imgs):
                ax = axes[rr, c]
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if rr == 0:
                    ax.set_title(methods[c], fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"{row['case']}\\nv{v}", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "table1_sadekar_style_contact_sheet.png", dpi=220)
    fig.savefig(OUT / "table1_sadekar_style_contact_sheet.pdf")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    for case in CASES:
        t0 = time.time()
        case_dir = GALLERY / case
        input_paths = [case_dir / "inputs" / f"input{i}.png" for i in range(3)]
        if not all(p.exists() for p in input_paths):
            raise FileNotFoundError(f"missing inputs for {case}: {input_paths}")

        case_out = OUT / case
        case_out.mkdir(parents=True, exist_ok=True)
        masks = load_masks(input_paths, RES)
        gt_masks_256 = [resize_binary_mask(m, IMG_SIZE) for m in masks]

        rows_branch = json.loads((case_dir / "candidate_scores.json").read_text())
        chosen_branch = None
        for rec in rows_branch:
            if rec.get("candidate") in (case_dir / "MANIFEST.json").name:
                pass
        manifest = json.loads((GALLERY / "MANIFEST.json").read_text())
        man = next(x for x in manifest if x.get("example") == case)
        fix_axis = bool(man.get("branch") == "fixed")

        _, aux = make_candidates(masks, RES, fix_axis)
        calibration = aux["calibration"]
        hull = aux["hull"].astype(bool)
        hull_voxels = int(hull.sum())
        calib_path = case_out / "sample_0_calibration.json"
        hull_path = case_out / "sample_0_hull.npy"
        calib_path.write_text(json.dumps(calibration, indent=2))
        np.save(hull_path, hull.astype(np.float32))

        mitra, _ = build_mitra(masks, res=RES, fix_axis=fix_axis)
        sadekar = run_sadekar(case, case_out, input_paths, calib_path, hull_path)
        ours = load_ours(case_dir)

        methods = {
            "Mitra-style": score_volume(mitra, masks, calibration, hull_voxels),
            "Sadekar-style": score_volume(sadekar, masks, calibration, hull_voxels),
            "ShadowForge": score_volume(ours, masks, calibration, hull_voxels),
        }
        for name, m in methods.items():
            for i, pred in enumerate(m["pred_masks"]):
                save_mask(pred, case_out / f"{name.lower().replace('-', '').replace(' ', '_')}_view{i}.png")
        for i, gt in enumerate(gt_masks_256):
            save_mask(gt, case_out / f"input_view{i}.png")

        row = {
            "case": case,
            "branch": man.get("branch"),
            "selected": man.get("selected"),
            "hull_voxels": hull_voxels,
            "gt_masks": gt_masks_256,
            "methods": methods,
            "elapsed_s": round(time.time() - t0, 2),
        }
        rows.append(row)
        slim = {
            "case": case,
            "branch": row["branch"],
            "selected": row["selected"],
            "hull_voxels": hull_voxels,
            "elapsed_s": row["elapsed_s"],
            "methods": {
                k: {kk: vv for kk, vv in v.items() if kk != "pred_masks"}
                for k, v in methods.items()
            },
        }
        (case_out / "metrics.json").write_text(json.dumps(slim, indent=2))
        print(f"{case}: done in {row['elapsed_s']}s")

    agg = []
    for method in ["Mitra-style", "Sadekar-style", "ShadowForge"]:
        vals = [r["methods"][method] for r in rows]
        agg.append(
            {
                "Method": method,
                "Cases": len(rows),
                "Mean IoU": float(np.mean([v["mean_iou"] for v in vals])),
                "Mean Dice": float(np.mean([v["mean_dice"] for v in vals])),
                "Worst IoU": float(np.min([v["min_iou"] for v in vals])),
                "Printable proxy": f"{sum(v['printable_proxy'] for v in vals)}/{len(vals)}",
                "Single body": f"{sum(v['single_body'] for v in vals)}/{len(vals)}",
                "Watertight proxy": f"{sum(v['watertight_proxy'] for v in vals)}/{len(vals)}",
                "Material saving vs hull": float(np.mean([v["material_saving_vs_hull"] for v in vals])),
                "Components mean": float(np.mean([v["component_count_6"] for v in vals])),
            }
        )

    with (OUT / "table1_main_comparison.tsv").open("w") as f:
        keys = list(agg[0].keys())
        f.write("\t".join(keys) + "\n")
        for r in agg:
            f.write("\t".join(str(r[k]) for k in keys) + "\n")

    serial_rows = []
    for row in rows:
        serial_rows.append(
            {
                "case": row["case"],
                "branch": row["branch"],
                "selected": row["selected"],
                "hull_voxels": row["hull_voxels"],
                "elapsed_s": row["elapsed_s"],
                "methods": {
                    k: {kk: vv for kk, vv in v.items() if kk != "pred_masks"}
                    for k, v in row["methods"].items()
                },
            }
        )
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "schema": "table1_mitra_sadekar_shadowforge_v1",
                "cases": CASES,
                "resolution": RES,
                "image_size": IMG_SIZE,
                "sadekar_script": str(SAD_SCRIPT),
                "mitra_script": str(ROOT / "experiments" / "mitra_baseline.py"),
                "shadowforge_source": str(GALLERY),
                "elapsed_total_s": round(time.time() - started, 2),
                "aggregate": agg,
                "rows": serial_rows,
                "notes": [
                    "Sadekar-style is the local vanilla differentiable voxel baseline, not the original authors' code.",
                    "Mitra-style is the local projection-preserving carve baseline; original 2009 code is not public in this workspace.",
                    "Printable proxy requires one 6-connected component and SDF watertight mesh in this evaluator.",
                ],
            },
            indent=2,
        )
    )
    make_contact_sheet(rows)
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
