#!/usr/bin/env python3
"""Rank ordered anime silhouette triples by exact fixed-axis compatibility.

The screen is input-only. It evaluates every assignment of three distinct
silhouettes to the X/Y/Z projection planes and never reads reconstruction
outputs. Results are written to a new folder so existing experiments remain
untouched.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
MASK_ROOT = ROOT / "results/anime30_adjusted_full_screen_20260704/adjusted_masks"
MODES = ("close1", "close2", "thick1")
POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def load_mask(path: Path, resolution: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize(
        (resolution, resolution), Image.Resampling.NEAREST
    )
    mask = np.asarray(image) >= 128
    if mask.mean() > 0.5:
        mask = ~mask
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / max(int(union), 1))


def exact_hull_projections(masks: list[np.ndarray]) -> list[np.ndarray]:
    mx, my, mz = masks
    n = mx.shape[0]
    volume = (
        np.broadcast_to(mx[None, :, :], (n, n, n))
        & np.broadcast_to(my[:, None, :], (n, n, n))
        & np.broadcast_to(mz[:, :, None], (n, n, n))
    )
    return [volume.any(axis=axis) for axis in range(3)]


def pair_relations(masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute packed existential relations used by three hull projections."""
    n = len(masks)
    packed_len = (masks[0].size + 7) // 8
    rel_ta = np.empty((n, n, packed_len), dtype=np.uint8)
    rel_aa = np.empty_like(rel_ta)
    rel_at = np.empty_like(rel_ta)
    mats = [mask.astype(np.uint8) for mask in masks]
    for i, left in enumerate(mats):
        for j, right in enumerate(mats):
            rel_ta[i, j] = np.packbits(((left.T @ right) > 0).reshape(-1))
            rel_aa[i, j] = np.packbits(((left @ right) > 0).reshape(-1))
            rel_at[i, j] = np.packbits(((left @ right.T) > 0).reshape(-1))
    return rel_ta, rel_aa, rel_at


def intersection_count(a: np.ndarray, b: np.ndarray) -> int:
    return int(POPCOUNT[np.bitwise_and(a, b)].sum())


def rank_mode(mode: str, resolution: int) -> list[dict]:
    names = [f"anime30_{index:02d}" for index in range(1, 31)]
    paths = [MASK_ROOT / mode / f"{name}_{mode}.png" for name in names]
    masks = [load_mask(path, resolution) for path in paths]
    packed = [np.packbits(mask.reshape(-1)) for mask in masks]
    areas = [int(mask.sum()) for mask in masks]
    rel_ta, rel_aa, rel_at = pair_relations(masks)

    rows: list[dict] = []
    for ix, iy, iz in itertools.permutations(range(len(names)), 3):
        # V[x,y,z] = Mx[y,z] & My[x,z] & Mz[x,y].
        # Each hull projection is a subset of its corresponding target, so
        # target IoU is simply retained target area / target area.
        iou_x = intersection_count(packed[ix], rel_ta[iz, iy]) / max(areas[ix], 1)
        iou_y = intersection_count(packed[iy], rel_aa[iz, ix]) / max(areas[iy], 1)
        iou_z = intersection_count(packed[iz], rel_at[iy, ix]) / max(areas[iz], 1)
        view_ious = [float(iou_x), float(iou_y), float(iou_z)]
        rows.append(
            {
                "mode": mode,
                "names_xyz": [names[ix], names[iy], names[iz]],
                "paths_xyz": [str(paths[ix]), str(paths[iy]), str(paths[iz])],
                "view_ious": view_ious,
                "min_iou": float(min(view_ious)),
                "mean_iou": float(np.mean(view_ious)),
                "mean_foreground_fraction": float(
                    np.mean([masks[ix].mean(), masks[iy].mean(), masks[iz].mean()])
                ),
            }
        )
    rows.sort(key=lambda row: (-row["min_iou"], -row["mean_iou"]))

    # Verify the packed formula against an explicit 3-D hull for the best row.
    best = rows[0]
    selected_masks = [load_mask(Path(path), resolution) for path in best["paths_xyz"]]
    projections = exact_hull_projections(selected_masks)
    explicit = [iou(target, pred) for target, pred in zip(selected_masks, projections, strict=True)]
    if not np.allclose(explicit, best["view_ious"], atol=1e-12):
        raise RuntimeError(f"Packed compatibility formula mismatch for {mode}: {explicit} != {best['view_ious']}")
    return rows


def select_disjoint(rows: list[dict], count: int, min_iou: float) -> list[dict]:
    selected: list[dict] = []
    used: set[str] = set()
    for row in rows:
        names = set(row["names_xyz"])
        if row["min_iou"] < min_iou or names & used:
            continue
        selected.append(row)
        used.update(names)
        if len(selected) == count:
            break
    return selected


def contact_sheet(rows: list[dict], output: Path) -> None:
    tile, gap, label_w, header = 190, 12, 215, 42
    canvas = Image.new(
        "RGB", (label_w + 3 * tile + 4 * gap, header + len(rows) * (tile + gap)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for col, label in enumerate(("X wall", "Y wall", "floor / Z")):
        draw.text((label_w + gap + col * (tile + gap) + 65, 14), label, fill="black", font=font)
    for row_index, row in enumerate(rows):
        y = header + row_index * (tile + gap)
        draw.text((8, y + 44), f"case {row_index + 1}: {row['mode']}", fill="black", font=font)
        draw.text((8, y + 66), f"worst={row['min_iou']:.5f}", fill=(0, 105, 85), font=font)
        draw.text((8, y + 84), f"mean={row['mean_iou']:.5f}", fill=(0, 105, 85), font=font)
        for col, (name, path) in enumerate(zip(row["names_xyz"], row["paths_xyz"], strict=True)):
            mask = load_mask(Path(path), 256)
            panel = Image.new("RGB", (tile, tile), "black")
            silhouette = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").convert("RGB")
            silhouette.thumbnail((tile - 12, tile - 26), Image.Resampling.NEAREST)
            panel.paste(silhouette, ((tile - silhouette.width) // 2, 20))
            draw_x = label_w + gap + col * (tile + gap)
            canvas.paste(panel, (draw_x, y))
            draw.text((draw_x + 6, y + 5), name, fill=(255, 255, 255), font=font)
    canvas.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--min-iou", type=float, default=0.995)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict] = []
    for mode in MODES:
        mode_rows = rank_mode(mode, args.resolution)
        rows.extend(mode_rows)
        print(
            f"{mode}: best min={mode_rows[0]['min_iou']:.6f} "
            f"mean={mode_rows[0]['mean_iou']:.6f} names={mode_rows[0]['names_xyz']}",
            flush=True,
        )
    rows.sort(key=lambda row: (-row["min_iou"], -row["mean_iou"]))

    selected = select_disjoint(rows, args.count, args.min_iou)
    fallback_threshold = args.min_iou
    while len(selected) < args.count and fallback_threshold > 0.985:
        fallback_threshold -= 0.001
        selected = select_disjoint(rows, args.count, fallback_threshold)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "anime3_ordered_fixed_axis_detail_screen_v1",
        "input_only": True,
        "resolution": args.resolution,
        "requested_min_iou": args.min_iou,
        "effective_disjoint_min_iou": fallback_threshold,
        "selected": selected,
        "top_100": rows[:100],
    }
    (args.output / "topology_compatibility.json").write_text(json.dumps(manifest, indent=2))
    contact_sheet(selected, args.output / "topology_compatibility_selected.png")
    print(json.dumps({"output": str(args.output), "selected": selected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
