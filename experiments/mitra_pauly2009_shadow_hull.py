#!/usr/bin/env python3
"""Reproduce the non-interactive shadow-hull stage of Mitra and Pauly (2009).

The paper first constructs a conservative shadow hull by intersecting the
generalized cones induced by the prescribed binary shadows and their known
projections.  It then offers interactive ARAP-based image adjustment and voxel
editing tools.  Those interactive constraints and the original application are
not public, so this script deliberately implements only the deterministic,
paper-specified shadow-hull stage under a declared fixed orthographic protocol.

This is not a replacement for the legacy ``mitra_baseline.py`` greedy-carving
surrogate.  It does not minimize material and it does not claim to reproduce
the unpublished interactive editing session from the paper.

For a fixed-axis protocol the input order follows ``--axes``:
  x: image coordinates (y, z), projection along x
  y: image coordinates (x, z), projection along y
  z: image coordinates (x, y), projection along z
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_mask(path: Path, resolution: int) -> np.ndarray:
    """Read a binary silhouette, treating the minority tone as foreground."""
    image = Image.open(path).convert("L").resize((resolution, resolution), Image.Resampling.NEAREST)
    mask = np.asarray(image) >= 128
    if float(mask.mean()) > 0.5:
        mask = ~mask
    if not mask.any():
        raise ValueError(f"No foreground pixels found in {path}")
    return mask


def cone_from_axis_mask(mask: np.ndarray, axis: str) -> np.ndarray:
    """Lift one orthographic foreground mask to its generalized cone."""
    resolution = mask.shape[0]
    if mask.shape != (resolution, resolution):
        raise ValueError("Expected a square silhouette mask")
    if axis == "x":
        return np.broadcast_to(mask[None, :, :], (resolution, resolution, resolution)).copy()
    if axis == "y":
        return np.broadcast_to(mask[:, None, :], (resolution, resolution, resolution)).copy()
    if axis == "z":
        return np.broadcast_to(mask[:, :, None], (resolution, resolution, resolution)).copy()
    raise ValueError(f"Unsupported axis '{axis}'; use x, y, or z")


def project_axis(volume: np.ndarray, axis: str) -> np.ndarray:
    axes = {"x": 0, "y": 1, "z": 2}
    return np.any(volume, axis=axes[axis])


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    return float(intersection / union) if union else 1.0


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(prediction, target).sum()
    denom = prediction.sum() + target.sum()
    return float(2.0 * intersection / denom) if denom else 1.0


def save_binary(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def save_obj(volume: np.ndarray, path: Path) -> None:
    """Export a marching-cubes surface when scikit-image is available."""
    from skimage.measure import marching_cubes

    if not volume.any() or volume.all():
        return
    vertices, faces, _, _ = marching_cubes(np.pad(volume.astype(np.float32), 1), level=0.5)
    vertices -= 1.0
    with path.open("w", encoding="utf-8") as handle:
        for x, y, z in vertices:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")


def build_shadow_hull(masks: list[np.ndarray], axes: list[str]) -> np.ndarray:
    if len(masks) != len(axes):
        raise ValueError("The number of silhouettes must equal the number of axes")
    if len(set(axes)) != len(axes):
        raise ValueError("Each fixed orthographic axis may occur at most once")
    hull = np.ones((masks[0].shape[0],) * 3, dtype=bool)
    for mask, axis in zip(masks, axes):
        hull &= cone_from_axis_mask(mask, axis)
    return hull


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silhouettes", nargs="+", type=Path, required=True)
    parser.add_argument("--axes", nargs="+", choices=("x", "y", "z"), required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-obj", action="store_true")
    args = parser.parse_args()

    if len(args.silhouettes) != len(args.axes):
        parser.error("--silhouettes and --axes must contain the same number of entries")
    if args.resolution < 8:
        parser.error("--resolution must be at least 8")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    masks = [read_mask(path, args.resolution) for path in args.silhouettes]
    hull = build_shadow_hull(masks, args.axes)

    per_view = []
    for index, (target, axis) in enumerate(zip(masks, args.axes)):
        rendered = project_axis(hull, axis)
        save_binary(target, output / f"target_view{index}_{axis}.png")
        save_binary(rendered, output / f"rendered_view{index}_{axis}.png")
        per_view.append(
            {
                "index": index,
                "axis": axis,
                "iou": iou(rendered, target),
                "dice": dice(rendered, target),
                "target_pixels": int(target.sum()),
                "rendered_pixels": int(rendered.sum()),
            }
        )

    np.save(output / "mitra_pauly2009_shadow_hull.npy", hull.astype(np.uint8))
    if not args.no_obj:
        save_obj(hull, output / "mitra_pauly2009_shadow_hull.obj")

    result = {
        "method": "Mitra and Pauly (2009) shadow-hull core reproduction",
        "protocol": {
            "projection": "fixed orthographic axes",
            "axes": args.axes,
            "resolution": args.resolution,
            "input_paths": [str(path.resolve()) for path in args.silhouettes],
        },
        "scope": {
            "implemented": "intersection of generalized cones / conservative voxel shadow hull",
            "not_implemented": [
                "interactive ARAP image deformation",
                "interactive brush, ray, and erosion editing",
                "example-based space filling",
            ],
            "reason": "The original interactive application and user constraints are not publicly available.",
        },
        "occupied_voxels": int(hull.sum()),
        "mean_iou": float(np.mean([row["iou"] for row in per_view])),
        "mean_dice": float(np.mean([row["dice"] for row in per_view])),
        "per_view": per_view,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
