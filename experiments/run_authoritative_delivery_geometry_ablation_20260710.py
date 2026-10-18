#!/usr/bin/env python3
"""Evaluate discrete geometry alternatives under the authoritative 48-to-256 delivery protocol."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/authoritative_delivery_geometry_ablation_20260710"
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from voxel_ops import (  # noqa: E402
    build_tfs_visual_hull,
    calibrate_tfs_projection,
    connect6_min_bridges,
    connectivity_preserving_carve,
    make_watertight_printable,
    repair_tfs_projection_gaps,
)
from shadowforge_pipeline.run import decouple_volume, manufacturing_metrics  # noqa: E402

CASES = {
    "characters": ["Spider-Man.png", "superman.png", "batman.png"],
    "symbols": ["robust_spade.png", "robust_star.png", "robust_flower.png"],
    "objects": ["teddy2.png", "duck.png", "hand.png"],
    "geometry": ["robust_ring.png", "robust_hexagon.png", "robust_diamond.png"],
    "anime": ["anime_v40_cape_wizard.png", "anime_v40_magical_girl.png", "anime_v40_super_robot.png"],
    "icons": ["heart.png", "like.png", "victory.png"],
    "animals": ["bunny_0.png", "armadillo_0.png", "puma.png"],
    "robust_shapes": ["robust_cross.png", "robust_club.png", "robust_v38_gear.png"],
}
CORE_RES, DELIVERY_RES, SHELL_VOXELS = 48, 256, 2
S6 = generate_binary_structure(3, 1)


def masks(names: list[str], resolution: int) -> list[np.ndarray]:
    result = []
    for name in names:
        image = cv2.imread(str(ROOT / "generation" / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(name)
        image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)
        corners = np.r_[image[:4, :4].ravel(), image[-4:, -4:].ravel()]
        result.append(image > 127 if np.median(corners) < 127 else image < 127)
    return result


def core_variants(core_masks: list[np.ndarray], calibration: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    hull, _ = build_tfs_visual_hull(core_masks, CORE_RES, calibration=calibration, apply_cleanup=False)
    gap, _ = repair_tfs_projection_gaps(core_masks, CORE_RES, calibration=calibration)
    carved = connectivity_preserving_carve(gap, max_passes=40)
    bridged, _ = connect6_min_bridges(carved, gap, bridge_radius=1)
    full, _ = make_watertight_printable(bridged, gap, max_thicken=3)
    no_gap = connectivity_preserving_carve(hull, max_passes=40)
    no_gap, _ = connect6_min_bridges(no_gap, hull, bridge_radius=1)
    no_gap, _ = make_watertight_printable(no_gap, hull, max_thicken=3)
    return {
        "Visual hull": hull,
        "w/o gap filling": no_gap,
        "w/o connectivity guard": carved,
        "w/o mesh repair": bridged,
        "Full geometry pipeline": full,
    }, hull


def evaluate_case(case_id: str, names: list[str]) -> dict:
    started = time.perf_counter()
    core_masks = masks(names, CORE_RES)
    calibration = calibrate_tfs_projection(core_masks, search_res=CORE_RES, fix_axis=True)
    variants, _ = core_variants(core_masks, calibration)
    delivery_masks = masks(names, DELIVERY_RES)
    hull256, _ = build_tfs_visual_hull(delivery_masks, DELIVERY_RES, calibration=calibration)
    rows = []
    for name, core in variants.items():
        delivery = decouple_volume(core, hull256, DELIVERY_RES, SHELL_VOXELS)
        metric = manufacturing_metrics(delivery, delivery_masks, calibration, int(hull256.sum()), DELIVERY_RES)
        rows.append({"variant": name, **metric})
    return {"case_id": case_id, "inputs": names, "elapsed_s": time.perf_counter() - started, "rows": rows}


def aggregate(records: list[dict]) -> list[dict]:
    names = [row["variant"] for row in records[0]["rows"]]
    result = []
    for name in names:
        rows = [next(row for row in record["rows"] if row["variant"] == name) for record in records]
        result.append({
            "variant": name,
            "n": len(rows),
            "mean_iou_256": float(np.mean([row["mean_iou"] for row in rows])),
            "mean_dice_256": float(np.mean([row["mean_dice"] for row in rows])),
            "worst_view_iou_256": float(np.min([row["min_iou"] for row in rows])),
            "mean_saving": float(np.mean([row["material_saved_vs_hull"] for row in rows])),
            "single_body_rate": float(np.mean([row["component_count_6"] == 1 for row in rows])),
            "watertight_rate": float(np.mean([row["watertight"] for row in rows])),
            "wall_pass_rate": float(np.mean([row["erosion_survival"] >= 0.15 for row in rows])),
        })
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for case_id, names in CASES.items():
        print(f"RUN {case_id}", flush=True)
        records.append(evaluate_case(case_id, names))
        (OUT / "partial.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    payload = {
        "schema": "authoritative_delivery_geometry_ablation_v1",
        "core_resolution": CORE_RES,
        "delivery_resolution": DELIVERY_RES,
        "shell_voxels": SHELL_VOXELS,
        "screen": "not_applicable_geometry_ablation",
        "records": records,
        "aggregate": aggregate(records),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "summary.tsv").open("w", encoding="utf-8") as handle:
        fields = list(payload["aggregate"][0])
        handle.write("\t".join(fields) + "\n")
        for row in payload["aggregate"]:
            handle.write("\t".join(str(row[field]) for field in fields) + "\n")
    print(json.dumps(payload["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
