#!/usr/bin/env python3
"""Render all proper axis-aligned orientations for the fourth gallery case."""

from __future__ import annotations

import importlib.util
import itertools
import os
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = ROOT / "experiments/render_supp_gallery_simulations_four_cases_centered_20260801.py"
OUT = ROOT / "results/ant_dragon_shrimp_orientation_audit_20260801"


def load_renderer():
    spec = importlib.util.spec_from_file_location("gallery_renderer", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def axis_transforms(include_reflections: bool = False) -> list[Matrix]:
    transforms = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rows = [[0.0] * 3 for _ in range(3)]
            for row, column in enumerate(permutation):
                rows[row][column] = signs[row]
            matrix = Matrix(rows)
            determinant = round(matrix.determinant())
            if determinant == 1 or (include_reflections and determinant == -1):
                transforms.append(matrix)
    return transforms


def proper_axis_rotations() -> list[Matrix]:
    return axis_transforms(include_reflections=False)


def main() -> None:
    renderer = load_renderer()
    case = next(
        case for case in renderer.CASES if case["name"] == "ant_dragon_shrimp"
    )
    mesh_path = Path(os.environ.get("PCRAFT_ORIENTATION_MESH", str(case["mesh"])))
    OUT.mkdir(parents=True, exist_ok=True)
    selected = os.environ.get("PCRAFT_ORIENTATION_INDEX")
    ortho_scale = float(os.environ.get("PCRAFT_ORIENTATION_ORTHO_SCALE", "258"))
    target_z = float(os.environ.get("PCRAFT_ORIENTATION_TARGET_Z", "-18"))
    resolution_x = int(os.environ.get("PCRAFT_ORIENTATION_RESOLUTION_X", "640"))
    resolution_y = int(os.environ.get("PCRAFT_ORIENTATION_RESOLUTION_Y", "323"))
    suffix = os.environ.get("PCRAFT_ORIENTATION_SUFFIX", "")
    include_reflections = os.environ.get("PCRAFT_ORIENTATION_INCLUDE_REFLECTIONS") == "1"
    direct_matrix = os.environ.get("PCRAFT_ORIENTATION_DIRECT_MATRIX") == "1"
    post_flip = os.environ.get("PCRAFT_ORIENTATION_POST_FLIP", "")
    transforms = axis_transforms(include_reflections=include_reflections)
    for index, basis in enumerate(transforms):
        if selected is not None and index != int(selected):
            continue
        bpy.ops.wm.open_mainfile(filepath=str(renderer.BASE_BLEND))
        if direct_matrix:
            # Euler angles cannot represent a handedness-changing transform.
            # Apply the signed permutation directly to mesh coordinates, then
            # restore the renderer's centered, floor-seated placement.
            model = renderer.replace_model(mesh_path, Matrix.Identity(3))
            model.data.transform(basis.to_4x4())
            model.data.update()
            low, high = renderer.bounds(model)
            model.location.x -= (low.x + high.x) * 0.5
            model.location.y -= (low.y + high.y) * 0.5
            model.location.z -= low.z
            bpy.context.view_layer.update()
        else:
            model = renderer.replace_model(mesh_path, basis)
        if post_flip:
            signs = tuple(
                -1.0 if axis in post_flip.lower() else 1.0
                for axis in ("x", "y", "z")
            )
            model.data.transform(
                Matrix.Diagonal((*signs, 1.0))
            )
            model.data.update()
            low, high = renderer.bounds(model)
            model.location.x -= (low.x + high.x) * 0.5
            model.location.y -= (low.y + high.y) * 0.5
            model.location.z -= low.z
            bpy.context.view_layer.update()
        renderer.configure_projectors(model, True)
        camera = bpy.data.objects.get("SimulationCamera")
        if camera is None:
            raise RuntimeError("Missing SimulationCamera")
        camera.data.ortho_scale = ortho_scale
        renderer.look_at(camera, Vector((10.0, 10.0, target_z)))
        scene = bpy.context.scene
        scene.camera = camera
        scene.render.resolution_x = resolution_x
        scene.render.resolution_y = resolution_y
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.filepath = str(
            OUT / f"orientation_{index:02d}{suffix}.png"
        )
        bpy.ops.render.render(write_still=True)
        (OUT / f"orientation_{index:02d}{suffix}.txt").write_text(
            "\n".join(" ".join(f"{value:.0f}" for value in row) for row in basis)
            + "\n",
            encoding="utf-8",
        )
        print(index, tuple(tuple(value for value in row) for row in basis))


if __name__ == "__main__":
    main()
