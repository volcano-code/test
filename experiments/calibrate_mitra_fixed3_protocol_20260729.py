#!/usr/bin/env python3
"""Calibrate a Mitra shadow hull to the canonical fixed-three-view frame.

The Mitra reproduction stores a voxel array in NumPy index order, whereas the
paper scene uses a physical right-handed XYZ frame with Z up.  This script
audits every signed axis permutation and selects the transform whose genuine
orthographic projections best match the declared left-wall, right-wall, and
floor targets.  It does not alter or register the target silhouettes.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from skimage.measure import marching_cubes


def read_mask(path: Path, resolution: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = image.resize((resolution, resolution), Image.Resampling.NEAREST)
    mask = np.asarray(image) >= 128
    if mask.mean() > 0.5:
        mask = ~mask
    if not mask.any():
        raise ValueError(f"No foreground pixels found in {path}")
    return mask


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def iou(prediction: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(prediction, target).sum()
    if not union:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = prediction.sum() + target.sum()
    if not denominator:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def receiver_projection(volume: np.ndarray, axis: int) -> np.ndarray:
    """Project from the scene interior onto a receiver in image coordinates.

    Left wall: view along -X, screen right is +Y, screen up is +Z.
    Right wall: view along +Y, screen right is +X, screen up is +Z.
    Floor: view along -Z, screen right is +X, screen up is +Y.
    """

    projection = np.any(volume, axis=axis)
    return projection.T[::-1, :]


def transform_volume(
    volume: np.ndarray,
    permutation: tuple[int, int, int],
    signs: tuple[int, int, int],
) -> np.ndarray:
    transformed = np.transpose(volume, permutation)
    for axis, sign in enumerate(signs):
        if sign < 0:
            transformed = np.flip(transformed, axis=axis)
    return transformed


def export_mesh(volume: np.ndarray, output: Path) -> dict[str, object]:
    padded = np.pad(volume.astype(np.float32), 1)
    vertices, faces, _, _ = marching_cubes(padded, level=0.5)
    vertices -= 1.0
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.export(output / "mitra_calibrated.obj")
    mesh.export(output / "mitra_calibrated.stl")
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "components": int(len(mesh.split(only_watertight=False))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=Path, required=True)
    parser.add_argument("--targets", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.volume.is_file():
        raise FileNotFoundError(args.volume)
    for target in args.targets:
        if not target.is_file():
            raise FileNotFoundError(target)

    args.output.mkdir(parents=True, exist_ok=True)
    source = np.load(args.volume).astype(bool)
    if source.ndim != 3 or len(set(source.shape)) != 1:
        raise ValueError(f"Expected a cubic 3D volume, received {source.shape}")
    resolution = source.shape[0]
    targets = [read_mask(path, resolution) for path in args.targets]

    audit_rows: list[dict[str, object]] = []
    best: tuple[float, float, tuple[int, ...], tuple[int, ...], np.ndarray] | None = None
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            candidate = transform_volume(source, permutation, signs)
            predictions = [
                receiver_projection(candidate, axis) for axis in range(3)
            ]
            view_ious = [
                iou(prediction, target)
                for prediction, target in zip(predictions, targets)
            ]
            view_dices = [
                dice(prediction, target)
                for prediction, target in zip(predictions, targets)
            ]
            mean_iou = float(np.mean(view_ious))
            mean_dice = float(np.mean(view_dices))
            row = {
                "permutation": list(permutation),
                "signs": list(signs),
                "mean_iou": mean_iou,
                "mean_dice": mean_dice,
                "view_ious": view_ious,
                "view_dices": view_dices,
            }
            audit_rows.append(row)
            key = (mean_iou, mean_dice)
            if best is None or key > best[:2]:
                best = (
                    mean_iou,
                    mean_dice,
                    permutation,
                    signs,
                    candidate.copy(),
                )

    if best is None:
        raise RuntimeError("No coordinate transform was evaluated")

    mean_iou, mean_dice, permutation, signs, calibrated = best
    projections = [
        receiver_projection(calibrated, axis) for axis in range(3)
    ]
    for index, (target, prediction) in enumerate(zip(targets, projections)):
        save_mask(target, args.output / f"target_view_{index}.png")
        save_mask(prediction, args.output / f"projection_view_{index}.png")

    np.save(args.output / "mitra_calibrated.npy", calibrated.astype(np.uint8))
    mesh_stats = export_mesh(calibrated, args.output)
    audit_rows.sort(
        key=lambda row: (row["mean_iou"], row["mean_dice"]),
        reverse=True,
    )
    result = {
        "purpose": "canonical fixed-three-view Mitra coordinate calibration",
        "source_volume": str(args.volume.resolve()),
        "target_paths": [str(path.resolve()) for path in args.targets],
        "receiver_order": [
            "left wall: slender detective",
            "right wall: Spider-Man",
            "floor: spider",
        ],
        "viewing_convention": {
            "left_wall": "view from scene interior along -X; +Y is image right",
            "right_wall": "view from scene interior along +Y; +X is image right",
            "floor": "view from above along -Z; +X is image right",
        },
        "selected_transform": {
            "physical_xyz_from_source_axes": list(permutation),
            "signs": list(signs),
        },
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "per_view": [
            {
                "index": index,
                "iou": iou(prediction, target),
                "dice": dice(prediction, target),
            }
            for index, (prediction, target) in enumerate(
                zip(projections, targets)
            )
        ],
        "mesh": mesh_stats,
        "top_coordinate_hypotheses": audit_rows[:10],
    }
    (args.output / "orientation_audit.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
