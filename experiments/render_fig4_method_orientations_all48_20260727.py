#!/usr/bin/env python3
"""Render all 48 signed-axis frames for selected Figure 4 methods."""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import bpy


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path.insert(0, str(ROOT / "experiments"))
import render_fig4_connected06_blender_true_shadows_20260727 as base  # noqa: E402


OUT_ROOT = (
    ROOT
    / "paper_aaai_submission/fig4_table1_connected06_soft_preview_20260726"
    / "blender_true_shadow_renders/orientation_audit_all48"
)


def signed_axis_frames():
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            yield permutation, signs


def apply_orientation(obj, permutation, signs) -> None:
    for vertex in obj.data.vertices:
        source = tuple(vertex.co)
        vertex.co = tuple(
            signs[axis] * source[permutation[axis]]
            for axis in range(3)
        )
    obj.data.update()


def render_method(slug: str) -> None:
    source, color = base.METHODS[slug]
    output = OUT_ROOT / slug
    output.mkdir(parents=True, exist_ok=True)

    for index, (permutation, signs) in enumerate(signed_axis_frames()):
        base.clear_scene()
        geometry = base.import_geometry(source)
        apply_orientation(geometry, permutation, signs)
        if slug == "sadekar_mesh":
            base.remove_degenerate_display_faces(geometry)
        base.normalize_geometry(geometry, base.DISPLAY_SCALE[slug])
        _, camera = base.setup_scene(slug, color)
        scene = bpy.context.scene
        scene.render.resolution_x = 450
        scene.render.resolution_y = 338
        scene.render.resolution_percentage = 100
        camera.location = base.VIEWS["view_a"]
        base.point_camera(camera, base.Vector((0.0, 0.0, 0.92)))
        label = (
            f"{index:02d}_p{''.join(map(str, permutation))}"
            f"_s{''.join('p' if sign > 0 else 'm' for sign in signs)}"
        )
        scene.render.filepath = str(output / f"{label}.png")
        bpy.ops.render.render(write_still=True)
        print(f"{slug}: {label}", flush=True)


def main() -> None:
    selected = os.environ.get(
        "FIG4_AUDIT_METHODS",
        "sadekar_mesh,sadekar_voxel,pcraft",
    )
    slugs = [item.strip() for item in selected.split(",") if item.strip()]
    unknown = [slug for slug in slugs if slug not in base.METHODS]
    if unknown:
        raise ValueError(f"Unknown Figure 4 methods: {unknown}")
    missing = [
        str(base.METHODS[slug][0])
        for slug in slugs
        if not base.METHODS[slug][0].is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing Figure 4 geometry:\n" + "\n".join(missing))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        render_method(slug)
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
