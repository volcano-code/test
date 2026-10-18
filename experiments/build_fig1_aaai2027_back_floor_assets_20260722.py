#!/usr/bin/env python3
"""Build real back-wall/floor meshes for the AAAI 2027 Figure-1 scene."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import generate_binary_structure, label
from skimage.measure import marching_cubes


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = ROOT / "results/fig1_symbol_pairs_pcraft_proposal_20260722"
OUT = ROOT / "results/fig1_aaai2027_back_floor_delivery_20260722"
RES = 256
PRINT_MM = 70.0
S6 = generate_binary_structure(3, 1)


CASES = {
    "A_2": SOURCE / "A_2",
    "A_0": SOURCE / "A_0",
    "I_7": SOURCE / "I_7",
}


def read_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((RES, RES), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def export_stl(volume: np.ndarray, path: Path) -> trimesh.Trimesh:
    vertices, faces, _, _ = marching_cubes(np.pad(volume.astype(np.float32), 1), level=0.5)
    vertices = (vertices - 1.0) / (RES - 1.0) - 0.5
    # Array rows increase downward.  Flip both image-row axes so the rear-wall
    # glyph is upright and the floor glyph reads from the frontal camera.
    vertices[:, 0] *= -1.0  # z
    vertices[:, 1] *= -1.0  # y
    mesh = trimesh.Trimesh(vertices=vertices[:, [2, 1, 0]], faces=faces, process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    if mesh.volume < 0:
        mesh.invert()
    mesh.apply_scale(PRINT_MM / max(float(mesh.extents.max()), 1e-9))
    mesh.export(path)
    return mesh


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.logical_and(a, b).sum() / max(int(np.logical_or(a, b).sum()), 1))


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows = []

    for name, source in CASES.items():
        case_out = OUT / name
        case_out.mkdir()
        back = read_mask(source / "inputs/left_wall_x_target.png")
        floor = read_mask(source / "inputs/right_wall_y_target.png")

        # volume[z, y, x]: rear-wall projection integrates y; floor integrates z.
        volume = back[:, None, :] & floor[None, :, :]
        rear_projection = volume.any(axis=1)
        floor_projection = volume.any(axis=0)
        np.save(case_out / "back_floor_volume256.npy", volume.astype(np.uint8))
        Image.fromarray((rear_projection * 255).astype(np.uint8)).save(case_out / "rear_projection.png")
        Image.fromarray((floor_projection * 255).astype(np.uint8)).save(case_out / "floor_projection.png")
        stl = case_out / "back_floor_delivery_70mm.stl"
        mesh = export_stl(volume, stl)
        _, components = label(volume, structure=S6)
        rows.append(
            {
                "case": name,
                "source_run": str(source),
                "resolution": RES,
                "views": ["rear_wall_xz", "floor_xy"],
                "rear_iou": iou(rear_projection, back),
                "floor_iou": iou(floor_projection, floor),
                "components_6": int(components),
                "watertight": bool(mesh.is_watertight),
                "faces": int(len(mesh.faces)),
                "stl": str(stl),
            }
        )

    manifest = {
        "schema": "fig1_aaai2027_back_floor_delivery_v1",
        "purpose": "Physical rear-wall/floor coordinate delivery for Figure 1",
        "projection_mechanism": "binary visual-hull intersection exported as a real mesh",
        "no_shadow_textures": True,
        "cases": rows,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
