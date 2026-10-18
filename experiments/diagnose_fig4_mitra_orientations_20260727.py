#!/usr/bin/env python3
"""Render all 48 signed-axis frames of the Figure 4 Mitra mesh."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import bpy
import numpy as np


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path.insert(0, str(ROOT / "experiments"))
import render_fig4_connected06_blender_true_shadows_20260727 as base  # noqa: E402


OUT = (
    ROOT
    / "paper_aaai_submission/fig4_table1_connected06_soft_preview_20260726"
    / "blender_true_shadow_renders/mitra_orientation_audit_all48"
)
SOURCE, COLOR = base.METHODS["mitra"]


def permutation_sign(permutation: tuple[int, int, int]) -> int:
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(3)
        for j in range(i + 1, 3)
    )
    return -1 if inversions % 2 else 1


def signed_axis_frames():
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            yield permutation, signs


def apply_orientation(obj, permutation, signs):
    for vertex in obj.data.vertices:
        source = tuple(vertex.co)
        vertex.co = tuple(
            signs[axis] * source[permutation[axis]]
            for axis in range(3)
        )
    obj.data.update()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for index, (permutation, signs) in enumerate(signed_axis_frames()):
        base.clear_scene()
        geometry = base.import_geometry(SOURCE)
        apply_orientation(geometry, permutation, signs)
        base.normalize_geometry(geometry, base.DISPLAY_SCALE["mitra"])
        geometry, camera = base.setup_scene("mitra", COLOR)
        scene = bpy.context.scene
        scene.render.resolution_x = 450
        scene.render.resolution_y = 338
        camera.location = base.VIEWS["view_a"]
        base.point_camera(camera, base.Vector((0.0, 0.0, 0.92)))
        label = (
            f"{index:02d}_p{''.join(map(str, permutation))}"
            f"_s{''.join('p' if sign > 0 else 'm' for sign in signs)}"
        )
        scene.render.filepath = str(OUT / f"{label}.png")
        bpy.ops.render.render(write_still=True)
        print(label)


if __name__ == "__main__":
    main()
