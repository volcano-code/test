#!/usr/bin/env python3
"""Build an isolated ant/dragon/shrimp solid for physical corner display.

The archived StructShadow volume stores fixed-surface masks in protocol array
order ([Y, Z] and [X, Z]).  A vertical physical receiver displays those arrays
with Z as image height, so directly importing the archived STL transposes both
wall silhouettes.  This script rebuilds the same three-view visual hull in
physical receiver coordinates, then applies the existing printable-solid and
SDF meshing utilities.  It never modifies the archived source result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    generate_binary_structure,
    label,
)


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
VOXEL_OPS = Path("/data/xvxinyao/xvxinyao/StructShadow/generation")
SOURCE = Path(
    "/data/xvxinyao/xvxinyao/StructShadow/print_ready_stls_diversity/by_group/"
    "n3_g5_ant_dragon_shrimp/silhouettes"
)
OUT = (
    ROOT
    / "paper_aaai_submission/supplement_gallery_one_case_20260801"
    / "ant_dragon_shrimp/physical_consistent"
)
RESOLUTION = 128


def load_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    image = image.resize((RESOLUTION, RESOLUTION), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def keep_largest(volume: np.ndarray) -> np.ndarray:
    labels, count = label(volume)
    if count <= 1:
        return volume.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def iou(reference: np.ndarray, prediction: np.ndarray) -> float:
    union = np.logical_or(reference, prediction).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(reference, prediction).sum() / union)


def dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    denominator = reference.sum() + prediction.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(reference, prediction).sum() / denominator)


def main() -> None:
    sys.path.insert(0, str(VOXEL_OPS))
    from voxel_ops import (  # pylint: disable=import-error,import-outside-toplevel
        make_watertight_printable,
        minimize_material_printable,
        sdf_watertight_mesh,
    )

    ant = load_mask(SOURCE / "bugs_ant.png")
    dragon = load_mask(SOURCE / "x_dragon.png")
    shrimp = load_mask(SOURCE / "sea_shrimp.png")

    # Receiver geometry in the approved Blender corner.  The archived dragon
    # and shrimp masks already face left; preserve those authoritative masks so
    # their three-view compatibility with the floor constraint is unchanged.
    left_wall = dragon
    right_wall = shrimp
    plane_x_yz = np.flipud(right_wall).T
    plane_y_xz = np.flipud(np.fliplr(left_wall)).T
    plane_z_xy = ant

    hull = (
        plane_x_yz[None, :, :]
        & plane_y_xz[:, None, :]
        & plane_z_xy[:, :, None]
    )
    if not hull.any():
        raise RuntimeError("The physical-coordinate visual hull is empty")

    minimal, minimize_info = minimize_material_printable(
        hull,
        max_passes=24,
        bridge_radius=1,
    )
    thickened, watertight_info = make_watertight_printable(
        minimal,
        hull,
        max_thicken=3,
    )
    thickened = keep_largest(thickened)

    # Use the measured fidelity/material knee for the gallery solid.  The
    # dilation is clipped to the visual hull, so it improves object readability
    # without introducing silhouette false positives.
    selected = keep_largest(
        binary_fill_holes(
            binary_dilation(
                minimal,
                structure=generate_binary_structure(3, 1),
                iterations=3,
            )
            & hull
        )
    )
    mesh = sdf_watertight_mesh(selected, smooth_sigma=0.45, level=0.0)

    # Marching-cubes coordinates are voxel indices.  Export a grounded 100 mm
    # object while preserving the reconstructed proportions.
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    mesh.apply_scale(100.0 / float(extents.max()))
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_translation((0.0, 0.0, -float(mesh.bounds[0, 2])))

    OUT.mkdir(parents=True, exist_ok=True)
    stl = OUT / "ant_dragon_shrimp_physical_consistent_100mm.stl"
    mesh.export(stl)

    np.savez_compressed(
        OUT / "physical_consistent_volume.npz",
        hull=hull,
        minimal=minimal,
        selected=selected,
        thickened=thickened,
    )

    # Convert the reconstructed receiver arrays back to the exact image
    # convention used by the gallery.  These are measured projections from the
    # exported solid, not redrawn target silhouettes.
    projection_x = selected.max(axis=0)
    projection_y = selected.max(axis=1)
    projection_z = selected.max(axis=2)
    projected_right = np.flipud(projection_x.T)
    projected_left = np.fliplr(np.flipud(projection_y.T))
    projected_floor = projection_z
    targets = [left_wall, right_wall, ant]
    projections = [projected_left, projected_right, projected_floor]
    names = ["left_dragon", "right_shrimp", "floor_ant"]
    view_ious = []
    view_dices = []
    for name, target, projection in zip(names, targets, projections):
        save_mask(OUT / f"target_{name}.png", target)
        save_mask(OUT / f"projection_{name}.png", projection)
        view_ious.append(iou(target, projection))
        view_dices.append(dice(target, projection))

    evaluation = {
        "display_order": names,
        "view_ious": view_ious,
        "view_dices": view_dices,
        "mean_iou": float(np.mean(view_ious)),
        "mean_dice": float(np.mean(view_dices)),
    }
    (OUT / "physical_evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    audit = {
        "source_masks": {
            "floor": str(SOURCE / "bugs_ant.png"),
            "right_wall": str(SOURCE / "sea_shrimp.png"),
            "left_wall": str(SOURCE / "x_dragon.png"),
        },
        "resolution": RESOLUTION,
        "receiver_mapping": {
            "x_wall_yz": "flipud(sea_shrimp).T",
            "y_wall_xz": "flipud(fliplr(x_dragon)).T",
            "floor_xy": "bugs_ant",
        },
        "hull_voxels": int(hull.sum()),
        "minimal": minimize_info,
        "optional_thickening_diagnostic": watertight_info,
        "selected_voxels": int(selected.sum()),
        "selected_material_saved_ratio": float(1.0 - selected.sum() / hull.sum()),
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_winding_consistent": bool(mesh.is_winding_consistent),
        "physical_evaluation": evaluation,
        "stl": str(stl),
    }
    (OUT / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
