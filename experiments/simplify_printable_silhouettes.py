#!/usr/bin/env python3
"""Build conservative, auditable printable-target variants from binary silhouettes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.draw import line
from skimage.morphology import binary_closing, binary_opening, disk


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "generation")]

from voxel_ops import build_tfs_visual_hull, calibrate_tfs_projection, compute_iou_np


S8 = np.ones((3, 3), dtype=bool)
S6 = ndimage.generate_binary_structure(3, 1)


def load_mask(path: Path, resolution: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize(
        (resolution, resolution), Image.Resampling.LANCZOS
    )
    mask = np.asarray(image) >= 128
    if mask.mean() > 0.5:
        mask = ~mask
    return mask


def component_stats(mask: np.ndarray) -> dict:
    labels, count = ndimage.label(mask, structure=S8)
    sizes = np.bincount(labels.ravel())[1:]
    distance = ndimage.distance_transform_edt(mask)
    return {
        "components_8": int(count),
        "foreground_pixels": int(mask.sum()),
        "thin_fraction_r1": float(
            np.logical_and(mask, distance <= 1.0).sum() / max(int(mask.sum()), 1)
        ),
        "thin_fraction_r2": float(
            np.logical_and(mask, distance <= 2.0).sum() / max(int(mask.sum()), 1)
        ),
        "largest_component_fraction": float(
            sizes.max(initial=0) / max(int(mask.sum()), 1)
        ),
    }


def fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    labels, count = ndimage.label(holes, structure=S8)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    result = mask.copy()
    for index in range(1, count + 1):
        if sizes[index] <= max_area:
            result[labels == index] = True
    return result


def keep_major_components(mask: np.ndarray, fraction: float, min_area: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=S8)
    if count <= 1:
        return mask.copy()
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max(initial=0))
    keep = np.zeros_like(mask)
    for index in range(1, count + 1):
        if sizes[index] >= max(min_area, int(round(largest * fraction))):
            keep |= labels == index
    return keep


def bridge_components(mask: np.ndarray, radius: int) -> tuple[np.ndarray, int]:
    result = mask.copy()
    bridges = 0
    while True:
        labels, count = ndimage.label(result, structure=S8)
        if count <= 1:
            return result, bridges
        sizes = np.bincount(labels.ravel())
        main_index = int(np.argmax(sizes[1:]) + 1)
        main = labels == main_index
        distance, nearest = ndimage.distance_transform_edt(~main, return_indices=True)

        best = None
        for index in range(1, count + 1):
            if index == main_index:
                continue
            component = labels == index
            weighted = np.where(component, distance, np.inf)
            y, x = np.unravel_index(np.argmin(weighted), result.shape)
            candidate = (float(distance[y, x]), y, x, int(nearest[0, y, x]), int(nearest[1, y, x]))
            if best is None or candidate < best:
                best = candidate

        if best is None:
            return result, bridges
        _, y0, x0, y1, x1 = best
        rr, cc = line(y0, x0, y1, x1)
        bridge = np.zeros_like(result)
        bridge[rr, cc] = True
        if radius > 0:
            bridge = ndimage.binary_dilation(bridge, structure=disk(radius))
        result |= bridge
        bridges += 1


def make_variant(
    mask: np.ndarray,
    *,
    major_fraction: float,
    min_area: int,
    bridge_radius: int,
    opening_radius: int,
    closing_radius: int,
    max_hole_area: int,
) -> tuple[np.ndarray, dict]:
    result = keep_major_components(mask, major_fraction, min_area)
    if opening_radius:
        result = binary_opening(result, disk(opening_radius))
    if closing_radius:
        result = binary_closing(result, disk(closing_radius))
    result = fill_small_holes(result, max_hole_area)
    result, bridges = bridge_components(result, bridge_radius)
    return result.astype(bool), {"inserted_bridges": bridges}


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def make_contact_sheet(
    output: Path,
    originals: list[np.ndarray],
    variants: dict[str, list[np.ndarray]],
) -> None:
    tile = 256
    label_h = 32
    names = ["original", *variants.keys()]
    canvas = Image.new("RGB", (tile * 3, (tile + label_h) * len(names)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    rows = {"original": originals, **variants}
    for row_index, name in enumerate(names):
        top = row_index * (tile + label_h)
        draw.text((8, top + 7), name.replace("_", " "), fill="black", font=font)
        for view_index, mask in enumerate(rows[name]):
            image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")
            canvas.paste(image, (view_index * tile, top + label_h))
    canvas.save(output)


def hull_components(masks: list[np.ndarray], resolution: int) -> tuple[int, int]:
    calibration = calibrate_tfs_projection(masks, search_res=min(64, resolution), fix_axis=True)
    hull, _ = build_tfs_visual_hull(
        masks, resolution, calibration=calibration, apply_cleanup=False
    )
    return int(ndimage.label(hull, structure=S6)[1]), int(hull.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    originals = [load_mask(path, args.resolution) for path in args.inputs]
    definitions = {
        "clean_major": {
            "major_fraction": 0.015,
            "min_area": 24,
            "bridge_radius": 1,
            "opening_radius": 0,
            "closing_radius": 1,
            "max_hole_area": 8,
        },
        "printable_light": {
            "major_fraction": 0.015,
            "min_area": 24,
            "bridge_radius": 2,
            "opening_radius": 1,
            "closing_radius": 2,
            "max_hole_area": 24,
        },
        "printable_balanced": {
            "major_fraction": 0.02,
            "min_area": 32,
            "bridge_radius": 2,
            "opening_radius": 2,
            "closing_radius": 3,
            "max_hole_area": 48,
        },
    }

    all_variants: dict[str, list[np.ndarray]] = {}
    report = {
        "inputs": [str(path.resolve()) for path in args.inputs],
        "resolution": args.resolution,
        "original": [],
        "variants": {},
    }
    for view_index, mask in enumerate(originals):
        report["original"].append({"view": view_index, **component_stats(mask)})
        save_mask(args.output / f"original_view_{view_index}.png", mask)

    original_hull_components, original_hull_voxels = hull_components(originals, 48)
    report["original_hull48"] = {
        "components_6": original_hull_components,
        "occupied_voxels": original_hull_voxels,
    }

    for name, definition in definitions.items():
        masks = []
        rows = []
        for view_index, original in enumerate(originals):
            mask, extra = make_variant(original, **definition)
            masks.append(mask)
            rows.append({
                "view": view_index,
                "iou_to_original": float(compute_iou_np(mask, original)),
                "retained_foreground": float(
                    np.logical_and(mask, original).sum() / max(int(original.sum()), 1)
                ),
                "added_pixel_fraction": float(
                    np.logical_and(mask, ~original).sum() / max(int(original.sum()), 1)
                ),
                **component_stats(mask),
                **extra,
            })
            save_mask(args.output / f"{name}_view_{view_index}.png", mask)
        components, voxels = hull_components(masks, 48)
        all_variants[name] = masks
        report["variants"][name] = {
            "definition": definition,
            "mean_iou_to_original": float(np.mean([row["iou_to_original"] for row in rows])),
            "min_iou_to_original": float(np.min([row["iou_to_original"] for row in rows])),
            "hull48_components_6": components,
            "hull48_occupied_voxels": voxels,
            "views": rows,
        }

    eligible = [
        (name, row)
        for name, row in report["variants"].items()
        if row["min_iou_to_original"] >= 0.95
    ]
    eligible.sort(
        key=lambda item: (
            item[1]["hull48_components_6"],
            -item[1]["mean_iou_to_original"],
        )
    )
    report["recommended_variant"] = eligible[0][0] if eligible else "clean_major"
    make_contact_sheet(args.output / "before_after_contact_sheet.png", originals, all_variants)
    (args.output / "simplification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "recommended_variant": report["recommended_variant"],
        "original_hull48_components": original_hull_components,
        "variants": {
            name: {
                "mean_iou_to_original": row["mean_iou_to_original"],
                "min_iou_to_original": row["min_iou_to_original"],
                "hull48_components_6": row["hull48_components_6"],
            }
            for name, row in report["variants"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
