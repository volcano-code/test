#!/usr/bin/env python3
"""Select anime silhouette triples using input-only topology criteria.

This screening never reads reconstruction outputs. It favors triples whose
individual masks and conservative three-view intersection remain connected
after mild erosion and low-resolution resampling.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


S4 = ndimage.generate_binary_structure(2, 1)
S8 = ndimage.generate_binary_structure(2, 2)
S6 = ndimage.generate_binary_structure(3, 1)


def load_mask(path: Path, resolution: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize(
        (resolution, resolution), Image.Resampling.NEAREST
    )
    mask = np.asarray(image) >= 128
    if mask.mean() > 0.5:
        mask = ~mask
    return mask


def components(mask: np.ndarray, structure: np.ndarray) -> int:
    return int(ndimage.label(mask, structure=structure)[1])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / max(int(union), 1))


def resize_mask(mask: np.ndarray, resolution: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    return np.asarray(
        image.resize((resolution, resolution), Image.Resampling.NEAREST)
    ) >= 128


def audit_mask(path: Path) -> dict | None:
    mask = load_mask(path, 256)
    area = int(mask.sum())
    fraction = float(mask.mean())
    if area == 0 or not 0.055 <= fraction <= 0.34:
        return None
    if components(mask, S4) != 1 or components(np.fliplr(mask), S4) != 1:
        return None

    eroded = ndimage.binary_erosion(mask, structure=S8)
    if not eroded.any() or components(eroded, S8) != 1:
        return None
    erosion_survival = float(eroded.sum() / area)

    mask48 = resize_mask(mask, 48)
    if components(mask48, S4) != 1:
        return None
    roundtrip = resize_mask(mask48, 256)
    lowres_iou = iou(mask, roundtrip)

    distance = ndimage.distance_transform_edt(mask)
    thin_fraction = float(np.logical_and(mask, distance <= 2.0).sum() / area)
    if erosion_survival < 0.70 or lowres_iou < 0.84 or thin_fraction > 0.34:
        return None
    return {
        "source_path": str(path.resolve()),
        "foreground_fraction": fraction,
        "erosion_survival": erosion_survival,
        "c48_roundtrip_iou": lowres_iou,
        "thin_fraction_r2": thin_fraction,
    }


def hull(masks: list[np.ndarray]) -> np.ndarray:
    x, y, z = masks
    n = x.shape[0]
    return (
        np.broadcast_to(x[None, :, :], (n, n, n))
        & np.broadcast_to(y[:, None, :], (n, n, n))
        & np.broadcast_to(z[:, :, None], (n, n, n))
    )


def audit_triple(rows: tuple[dict, dict, dict], resolution: int) -> dict | None:
    masks = [load_mask(Path(row["source_path"]), resolution) for row in rows]
    volume = hull(masks)
    if not volume.any() or components(volume, S6) != 1:
        return None
    projections = [volume.any(axis=axis) for axis in range(3)]
    view_ious = [iou(p, m) for p, m in zip(projections, masks, strict=True)]

    eroded = ndimage.binary_erosion(volume, structure=S6)
    erosion_connected = bool(eroded.any() and components(eroded, S6) == 1)
    if min(view_ious) < 0.78 or not erosion_connected:
        return None
    return {
        "resolution": resolution,
        "components_6": 1,
        "erosion_connected_6": erosion_connected,
        "occupied_voxels": int(volume.sum()),
        "view_ious": view_ious,
        "mean_iou": float(np.mean(view_ious)),
        "min_iou": float(np.min(view_ious)),
    }


def contact_sheet(cases: list[dict], output: Path) -> None:
    tile, label_w, header = 180, 190, 34
    canvas = Image.new("RGB", (label_w + 3 * tile, header + len(cases) * tile), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for col, label in enumerate(("View X", "View Y", "View Z")):
        draw.text((label_w + col * tile + 68, 10), label, fill="black", font=font)
    for row_index, case in enumerate(cases):
        y = header + row_index * tile
        draw.text((8, y + 56), case["case_id"], fill="black", font=font)
        draw.text((8, y + 76), "robust 1-body hull", fill=(0, 110, 90), font=font)
        for col, path_text in enumerate(case["target_paths_xyz"]):
            image = Image.open(path_text).convert("RGB").resize(
                (tile - 12, tile - 12), Image.Resampling.NEAREST
            )
            panel = Image.new("RGB", (tile, tile), "black")
            panel.paste(image, (6, 6))
            canvas.paste(panel, (label_w + col * tile, y))
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path(
            "/home/lin/xvxinyao/shadow/"
            "anime_shadow_processed_black_bg_light_simplified"
        ),
    )
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [row for path in sorted(args.pool.glob("*.png")) if (row := audit_mask(path))]
    rng = random.Random(args.seed)
    candidates = []
    for _ in range(args.attempts):
        triple = tuple(rng.sample(rows, 3))
        screen = audit_triple(triple, 64)
        if screen is None:
            continue
        per_mask = float(
            np.mean(
                [
                    row["erosion_survival"] * row["c48_roundtrip_iou"]
                    for row in triple
                ]
            )
        )
        score = 0.55 * screen["min_iou"] + 0.45 * per_mask
        candidates.append((score, triple, screen))
    candidates.sort(key=lambda item: -item[0])

    selected = []
    used: set[str] = set()
    for score, triple, _ in candidates:
        paths = [row["source_path"] for row in triple]
        if any(path in used for path in paths):
            continue
        final = audit_triple(triple, 256)
        if final is None:
            continue
        selected.append((score, triple, final))
        used.update(paths)
        if len(selected) == args.count:
            break
    if not selected:
        raise RuntimeError("No topology-friendly anime triples were found")

    inputs = args.output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, (score, triple, final) in enumerate(selected, 1):
        case_id = f"topology_friendly_anime_{index:02d}"
        targets = []
        for axis, row in zip(("x", "y", "z"), triple, strict=True):
            destination = inputs / f"{case_id}_{axis}.png"
            Image.fromarray(load_mask(Path(row["source_path"]), 256).astype(np.uint8) * 255).save(
                destination
            )
            targets.append(str(destination.resolve()))
        cases.append(
            {
                "case_id": case_id,
                "selection_score": score,
                "target_paths_xyz": targets,
                "sadekar_tfs_targets": [targets[1], targets[0], targets[2]],
                "input_audit": list(triple),
                "three_dimensional_topology_audit": final,
            }
        )

    manifest = {
        "title": "Input-only topology-friendly anime fixed-three-view shortlist",
        "selection_policy": {
            "uses_method_outputs": False,
            "individual_mask": [
                "one 4-connected component",
                "one component after one-pixel erosion",
                "one component after 256-to-48 resampling",
            ],
            "three_dimensional_intersection": [
                "one 6-connected component",
                "one component after one-voxel erosion",
            ],
        },
        "eligible_individual_masks": len(rows),
        "cases": cases,
    }
    (args.output / "topology_shortlist.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    contact_sheet(cases, args.output / "topology_shortlist.png")
    print(json.dumps({"output": str(args.output), "eligible_masks": len(rows), "cases": len(cases)}))


if __name__ == "__main__":
    main()
