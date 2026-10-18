#!/usr/bin/env python3
"""Render every proper axis-aligned orientation of the locked urban-allies STL."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = ROOT / "experiments/render_supp_gallery_simulations_four_cases_centered_20260801.py"
OUT = (
    ROOT
    / "paper_aaai_submission/supplement_gallery_one_case_20260801"
    / "urban_allies/orientation_candidates"
)


def load_renderer():
    spec = importlib.util.spec_from_file_location("gallery_renderer", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proper_axis_rotations() -> list[Matrix]:
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rows = [[0.0] * 3 for _ in range(3)]
            for row, column in enumerate(permutation):
                rows[row][column] = signs[row]
            matrix = Matrix(rows)
            if round(matrix.determinant()) == 1:
                rotations.append(matrix)
    return rotations


def matrix_values(matrix: Matrix) -> list[list[int]]:
    return [[int(round(value)) for value in row] for row in matrix]


def main() -> None:
    renderer = load_renderer()
    case = next(item for item in renderer.CASES if item["name"] == "urban_allies")
    OUT.mkdir(parents=True, exist_ok=True)
    records = []

    for index, basis in enumerate(proper_axis_rotations()):
        bpy.ops.wm.open_mainfile(filepath=str(renderer.BASE_BLEND))
        model = renderer.replace_model(case["mesh"], basis)
        centers = renderer.configure_projectors(model, True)

        camera = bpy.data.objects.get("SimulationCamera")
        if camera is None:
            raise RuntimeError("Missing SimulationCamera")
        camera.data.ortho_scale = 258.0
        renderer.look_at(camera, Vector((10.0, 10.0, -18.0)))

        scene = bpy.context.scene
        scene.camera = camera
        scene.render.resolution_x = 640
        scene.render.resolution_y = 323
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        output = OUT / f"orientation_{index:02d}.png"
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)

        matrix = matrix_values(basis)
        matrix_file = OUT / f"orientation_{index:02d}.txt"
        matrix_file.write_text(
            "\n".join(" ".join(str(value) for value in row) for row in matrix) + "\n",
            encoding="utf-8",
        )
        records.append(
            {
                "orientation": index,
                "proper_rotation": True,
                "determinant": 1,
                "basis": matrix,
                "render": str(output),
                "basis_file": str(matrix_file),
                "projector_centers": centers,
            }
        )
        print(f"orientation {index:02d}: {matrix}", flush=True)

    (OUT / "orientations.json").write_text(
        json.dumps(
            {
                "source_stl": str(case["mesh"]),
                "source_blend": str(renderer.BASE_BLEND),
                "count": len(records),
                "includes_reflections": False,
                "orientations": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
