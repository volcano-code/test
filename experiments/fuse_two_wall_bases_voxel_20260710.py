#!/usr/bin/env python3
"""Fuse a low display plinth into two-wall print models by voxel union.

The previous display files concatenated independent meshes.  This exporter
constructs the plinth in the same voxel field as the sculpture, so the output
has one topological body before mesh extraction.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from skimage import measure

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = ROOT / "results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709"
OUT = ROOT / "results/anime_two_wall_fixed_no_floor_wall_aligned_v3_voxel_fused_base_10cm_256_20260710"
S6 = generate_binary_structure(3, 1)
PRINT_MM = 100.0


def components(volume: np.ndarray) -> int:
    return int(label(volume.astype(bool), structure=S6)[1])


def project_xz(volume: np.ndarray) -> np.ndarray:
    # Two-wall protocol: the left wall receives the x-axis projection.
    return volume.any(axis=0)


def project_yz(volume: np.ndarray) -> np.ndarray:
    # The right wall receives the y-axis projection.
    return volume.any(axis=1)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(int((a | b).sum()), 1))


def add_fused_plinth(volume: np.ndarray) -> np.ndarray:
    """Add a compact plinth overlapping the lowermost printed model voxels."""
    solid = volume.astype(bool).copy()
    xyz = np.argwhere(solid)
    # export_mesh maps volume axis 0 to physical Z, the print direction.
    z0 = int(xyz[:, 0].min())
    contact = solid[z0]
    coords = np.argwhere(contact)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    footprint = np.zeros_like(contact)
    footprint[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1] = True
    # Five voxels give a >1.5-mm mechanical footprint at 10 cm / 256 voxels.
    footprint = binary_dilation(footprint, structure=np.ones((5, 5), dtype=bool), iterations=1)
    # The upper three layers overlap the sculpture volumetrically.  This is a
    # union in the occupancy field, not a coincident mesh contact.
    solid[: min(z0 + 3, solid.shape[0])] |= footprint[None, :, :]
    return solid


def export_mesh(volume: np.ndarray, path: Path) -> trimesh.Trimesh:
    padded = np.pad(volume.astype(np.float32), 1)
    vertices, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    vertices = (vertices - 1) / (volume.shape[0] - 1)
    vertices -= 0.5
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


def main() -> None:
    manifest = json.loads((SRC / "MANIFEST.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for case in manifest["cases"]:
        name = case["case"]
        source_volume = np.load(case["voxels"]).astype(bool)
        fused = add_fused_plinth(source_volume)
        before = (project_xz(source_volume), project_yz(source_volume))
        after = (project_xz(fused), project_yz(fused))
        destination = OUT / name
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / f"{name}_voxel_fused_base_256.npy", fused.astype(np.uint8))
        mesh = export_mesh(fused, destination / f"{name}_voxel_fused_base_10cm_256.stl")
        mesh_parts = len(mesh.split(only_watertight=False))
        if components(fused) != 1 or mesh_parts != 1 or not mesh.is_watertight:
            raise RuntimeError(f"fused base QA failed for {name}")
        record = {
            "case": name,
            "source_model": case["model_stl"],
            "fused_model": str(destination / f"{name}_voxel_fused_base_10cm_256.stl"),
            "voxel_volume": str(destination / f"{name}_voxel_fused_base_256.npy"),
            "voxel_components_6": components(fused),
            "mesh_parts": mesh_parts,
            "mesh_watertight": bool(mesh.is_watertight),
            "projection_iou_vs_source": [iou(before[0], after[0]), iou(before[1], after[1])],
            "base_added_voxels": int(fused.sum() - source_volume.sum()),
        }
        (destination / "FUSED_BASE_QA.json").write_text(json.dumps(record, indent=2) + "\n")
        cases.append(record)
    payload = {"schema": "two_wall_voxel_fused_base_v1", "source": str(SRC), "cases": cases}
    (OUT / "MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
