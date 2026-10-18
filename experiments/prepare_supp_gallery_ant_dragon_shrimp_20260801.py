#!/usr/bin/env python3
"""Prepare protocol-exact masks for the fourth supplementary gallery case."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
STRUCT_ROOT = Path("/data/xvxinyao/xvxinyao/StructShadow")
GEN = STRUCT_ROOT / "generation"
CASE = (
    STRUCT_ROOT
    / "print_ready_stls_diversity/by_group/n3_g5_ant_dragon_shrimp"
)
OUT = (
    ROOT
    / "paper_aaai_submission/supplement_gallery_one_case_20260801"
    / "ant_dragon_shrimp/protocol_exact"
)

sys.path.insert(0, str(GEN.parent))
sys.path.insert(0, str(GEN))

from datasets import load_data_from_list  # noqa: E402
from voxel_ops import (  # noqa: E402
    build_tfs_visual_hull,
    calibrate_tfs_projection,
    evaluate_tfs_volume,
    make_watertight_printable,
    minimize_material_printable,
)


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def main() -> None:
    names = ("bugs_ant", "x_dragon", "sea_shrimp")
    paths = [str(CASE / "silhouettes" / f"{name}.png") for name in names]
    masks_torch, _ = load_data_from_list(
        paths,
        True,
        (128, 128),
        torch.device("cpu"),
        3,
        False,
    )
    masks = [np.asarray(mask).squeeze().astype(np.float32) for mask in masks_torch]
    # Reuse the original StructShadow protocol instead of forcing the later
    # fixed-axis substitute. This recovers the calibration used to create the
    # authoritative MINIMAL delivery artifact.
    calibration = calibrate_tfs_projection(masks, search_res=96, fix_axis=False)
    hull, _ = build_tfs_visual_hull(masks, 128, calibration=calibration)
    minimal, _ = minimize_material_printable(hull, bridge_radius=1)
    watertight, _ = make_watertight_printable(minimal, hull)
    evaluation = evaluate_tfs_volume(
        watertight.astype(np.float32),
        masks,
        calibration,
        image_size=256,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        save_mask(OUT / f"target_{index}.png", evaluation["gt_masks"][index])
        save_mask(OUT / f"projection_{index}.png", evaluation["pred_masks"][index])

    record = {
        "case": "ant_dragon_shrimp",
        "source_silhouettes": paths,
        "source_stl": str(
            CASE / "g5_bugs_ant_x_dragon_sea_shrimp_MINIMAL.stl"
        ),
        "view_order": list(names),
        "calibration": calibration,
        "mean_iou_exact": evaluation["mean_iou"],
        "mean_dice_exact": evaluation["mean_dice"],
        "view_ious_exact": evaluation["view_ious"][:3],
        "view_dices_exact": evaluation["view_dices"][:3],
        "occupied_voxels": evaluation["occupied_voxels"],
        "components": evaluation.get("components"),
    }
    (OUT / "exact_evaluation.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
