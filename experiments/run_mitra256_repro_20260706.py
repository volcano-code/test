#!/usr/bin/env python3
"""Re-run the local Mitra-style baseline at 256^3 on the current Table-1 cases."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import generate_binary_structure, label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from candidate_selection_batch import make_candidates
from datasets import load_data_from_list
from mitra_baseline import build_mitra
from voxel_ops import apply_d4_transform, compute_dice_np, compute_iou_np, resize_binary_mask

OUT = ROOT / "results" / "mitra256_repro_table1_20260706"
GALLERY = ROOT / "results" / "gallery_FINAL"
CASES = [
    "dragon_peacock_trex",
    "parrot_squid_scorpion",
    "owl_bat_snail",
    "heroes_spiderman_superman_batman",
]
RES = 256
IMG_SIZE = 256
S6 = generate_binary_structure(3, 1)


def load_masks(paths: list[Path]) -> list[np.ndarray]:
    masks, _ = load_data_from_list([str(p) for p in paths], True, (RES, RES), torch.device("cpu"), 3, False)
    return [np.asarray(x).squeeze().astype(bool) for x in masks]


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


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def score(volume: np.ndarray, gt_masks: list[np.ndarray], calibration: dict, hull_voxels: int) -> dict:
    preds = project3(volume, calibration)
    gts = [resize_binary_mask(m, IMG_SIZE) for m in gt_masks]
    ious = [compute_iou_np(p, g) for p, g in zip(preds, gts)]
    dices = [compute_dice_np(p, g) for p, g in zip(preds, gts)]
    labeled, cc = label(volume.astype(bool), structure=S6)
    sizes = np.bincount(labeled.ravel())[1:]
    largest = int(sizes.max()) if len(sizes) else 0
    occ = int(volume.sum())
    return {
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "min_iou": float(np.min(ious)),
        "view_ious": [float(x) for x in ious],
        "view_dices": [float(x) for x in dices],
        "occupied_voxels": occ,
        "hull_voxels": int(hull_voxels),
        "material_saving_vs_hull": float(1.0 - occ / max(hull_voxels, 1)),
        "component_count_6": int(cc),
        "largest_component_ratio": float(largest / max(occ, 1)),
        "single_body": bool(cc == 1),
    }, preds, gts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((GALLERY / "MANIFEST.json").read_text())
    rows = []
    t_all = time.time()
    for case in CASES:
        t0 = time.time()
        case_dir = GALLERY / case
        input_paths = [case_dir / "inputs" / f"input{i}.png" for i in range(3)]
        man = next(x for x in manifest if x.get("example") == case)
        fix_axis = bool(man.get("branch") == "fixed")
        masks = load_masks(input_paths)
        _, aux = make_candidates(masks, RES, fix_axis)
        calibration = aux["calibration"]
        hull_voxels = int(aux["hull"].astype(bool).sum())
        vol, info = build_mitra(masks, res=RES, fix_axis=fix_axis)
        metrics, preds, gts = score(vol, masks, calibration, hull_voxels)
        case_out = OUT / case
        case_out.mkdir(parents=True, exist_ok=True)
        np.save(case_out / "mitra_style_voxels256.npy", vol.astype(np.uint8))
        for i, (pred, gt) in enumerate(zip(preds, gts)):
            save_mask(pred, case_out / f"mitra256_view{i}.png")
            save_mask(gt, case_out / f"input_view{i}.png")
        row = {
            "case": case,
            "branch": man.get("branch"),
            "fix_axis": fix_axis,
            "resolution": RES,
            "elapsed_s": round(time.time() - t0, 2),
            "build_info": info,
            "metrics": metrics,
        }
        (case_out / "metrics.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(json.dumps({"case": case, "elapsed_s": row["elapsed_s"], **metrics}, ensure_ascii=False), flush=True)

    vals = [r["metrics"] for r in rows]
    aggregate = {
        "cases": len(rows),
        "resolution": RES,
        "mean_iou": float(np.mean([v["mean_iou"] for v in vals])),
        "mean_dice": float(np.mean([v["mean_dice"] for v in vals])),
        "worst_iou": float(np.min([v["min_iou"] for v in vals])),
        "single_body": f"{sum(v['single_body'] for v in vals)}/{len(vals)}",
        "components_mean": float(np.mean([v["component_count_6"] for v in vals])),
        "material_saving_vs_hull": float(np.mean([v["material_saving_vs_hull"] for v in vals])),
        "elapsed_total_s": round(time.time() - t_all, 2),
    }
    (OUT / "results.json").write_text(json.dumps({"aggregate": aggregate, "rows": rows}, indent=2))
    with (OUT / "summary.tsv").open("w") as f:
        f.write("method\tcases\tresolution\tmean_iou\tmean_dice\tworst_iou\tsingle_body\tcomponents_mean\tmaterial_saving_vs_hull\n")
        f.write(
            "\t".join(
                [
                    "Mitra-style",
                    str(aggregate["cases"]),
                    str(aggregate["resolution"]),
                    str(aggregate["mean_iou"]),
                    str(aggregate["mean_dice"]),
                    str(aggregate["worst_iou"]),
                    aggregate["single_body"],
                    str(aggregate["components_mean"]),
                    str(aggregate["material_saving_vs_hull"]),
                ]
            )
            + "\n"
        )
    print("AGG", json.dumps(aggregate, ensure_ascii=False), flush=True)
    print(f"saved={OUT}", flush=True)


if __name__ == "__main__":
    main()
