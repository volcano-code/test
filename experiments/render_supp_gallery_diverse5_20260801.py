#!/usr/bin/env python3
"""Render the selected three-anime P-CRAFT case for the supplement gallery."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from mathutils import Matrix


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
TEMPLATE = ROOT / "experiments/render_supp_gallery_arcane_casters_20260801.py"
CASE = (
    ROOT
    / "results/anime30_diverse_fixed_50_60_10cm_256_20260707"
    / "diverse5_thick1_13_21_23"
)
OUT = (
    ROOT
    / "paper_aaai_submission/supplement_gallery_one_case_20260801"
    / "diverse5_thick1_13_21_23"
    / "renders"
)


def main() -> None:
    spec = importlib.util.spec_from_file_location("gallery_render_template", TEMPLATE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Blender rendering template: {TEMPLATE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CASE = CASE
    module.MESH = CASE / "diverse5_thick1_13_21_23_carve_pf4_10cm_256.stl"
    module.OUT = OUT

    # Preserve the common camera and scene while preventing this reflective,
    # highly detailed mesh from clipping the projection illumination.
    original_add_spot = module.add_spot
    original_add_area_light = module.add_area_light

    def add_spot_scaled(name, location, target, energy, color):
        return original_add_spot(name, location, target, energy * 0.38, color)

    def add_area_light_scaled(location, energy, size, target):
        return original_add_area_light(location, energy * 0.55, size, target)

    module.add_spot = add_spot_scaled
    module.add_area_light = add_area_light_scaled
    # Fixed-TFS receiver layout for this three-view case:
    #   T1 / original Z -> right wall,
    #   T2 / original X -> floor,
    #   T3 / original Y -> left wall.
    # The mapping keeps both wall silhouettes upright while preserving all
    # three true orthographic projections of the evidence STL.
    module.SIMULATION_BASIS = Matrix(
        (
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
            (-1.0, 0.0, 0.0),
        )
    )
    # Render all three protocol-aligned orthographic views before the shared
    # physical simulation. Numbered outputs remain x, y, z respectively.
    module.render_axis_view(
        0,
        (4.0, 0.0, 0.78),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.20, 0.48, 0.72, 1.0),
    )
    module.render_axis_view(
        1,
        (0.0, -4.0, 0.78),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.13, 0.58, 0.52, 1.0),
    )
    module.render_axis_view(
        2,
        (0.0, 0.0, 4.0),
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.45, 0.33, 0.69, 1.0),
    )
    module.render_simulation()
    print(OUT)


if __name__ == "__main__":
    main()
