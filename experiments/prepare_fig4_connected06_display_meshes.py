#!/usr/bin/env python3
"""Prepare visualization-only copies for the matched Figure 4 Blender render.

The evaluated source meshes and all reported metrics remain unchanged. This
script removes numerical display artifacts that otherwise prevent the stored
reconstruction from being visible under a shared camera.
"""

from pathlib import Path

import numpy as np
import trimesh


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = (
    ROOT
    / "paper_aaai_submission/fig4_table1_connected06_soft_preview_20260726"
    / "blender_renders/display_meshes"
)
SOURCES = {
    "sadekar_mesh": Path(
        "/data/xvxinyao/ShadowArt-Revisited/mesh_results/"
        "fixed3_mr1_connected10_sadekar_20260722/mesh/"
        "connected_06/seed_17/sample_0_output.obj"
    ),
    "sadekar_voxel": Path(
        "/data/xvxinyao/ShadowArt-Revisited/voxel_results/"
        "fixed3_mr1_connected10_sadekar_20260722/voxel/"
        "connected_06/seed_17/sample_0_output.obj"
    ),
}


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def prepare_sadekar_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    low = np.quantile(mesh.vertices, 0.01, axis=0)
    high = np.quantile(mesh.vertices, 0.99, axis=0)
    valid_vertices = np.all(
        (mesh.vertices >= low) & (mesh.vertices <= high), axis=1
    )
    mesh.update_faces(np.all(valid_vertices[mesh.faces], axis=1))
    mesh.remove_unreferenced_vertices()
    return max(mesh.split(only_watertight=False), key=lambda part: part.area)


def prepare_sadekar_voxel(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    return max(mesh.split(only_watertight=False), key=lambda part: part.area)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    processors = {
        "sadekar_mesh": prepare_sadekar_mesh,
        "sadekar_voxel": prepare_sadekar_voxel,
    }
    for name, source in SOURCES.items():
        mesh = processors[name](load_mesh(source))
        target = OUT / f"{name}_display_only.obj"
        mesh.export(target)
        print(
            f"{name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces -> {target}"
        )


if __name__ == "__main__":
    main()
