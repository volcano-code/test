#!/usr/bin/env python3
"""Render the final five supplementary-gallery simulations without overwrites."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import sys

from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_supp_gallery_simulations_four_cases_centered_20260801 as renderer


ANT_DRAGON_SHRIMP_BASIS = Matrix(
    (
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
    )
)
ORIENTATION_13 = Matrix(
    (
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
    )
)


# Shared physical display convention for every gallery row.  The distant
# projectors approximate the orthographic protocol while retaining visible
# circular light pools in the Blender scene.
UNIFIED_DISPLAY_EXTENT = 58.0
UNIFIED_RECEIVER_RADIUS = 56.0
UNIFIED_SOURCE_DISTANCE = 2000.0
UNIFIED_LIGHT_ENERGY = 165_000_000.0
UNIFIED_CAMERA_ORTHO_SCALE = 274.0
TWO_VIEW_CAMERA_TARGET = (10.0, 10.0, 10.0)
THREE_VIEW_CAMERA_TARGET = (10.0, 10.0, 5.0)


def named_case(name: str) -> dict:
    return deepcopy(next(case for case in renderer.CASES if case["name"] == name))


def close_projector_case(name: str, *, floor_shadow: bool) -> dict:
    case = named_case(name)
    case.update(
        {
            "floor_shadow": floor_shadow,
            "output_subdir": "renders_gallery_final_20260801",
            "output_name": "simulation_final.png",
            "blend_name": f"{name}_gallery_final.blend",
            "display_extent": UNIFIED_DISPLAY_EXTENT,
            "receiver_radius": UNIFIED_RECEIVER_RADIUS,
            "wall_source_distance": UNIFIED_SOURCE_DISTANCE,
            "floor_source_height": UNIFIED_SOURCE_DISTANCE,
            "light_energy": UNIFIED_LIGHT_ENERGY,
            "camera_ortho_scale": UNIFIED_CAMERA_ORTHO_SCALE,
            "camera_target": TWO_VIEW_CAMERA_TARGET
            if not floor_shadow
            else THREE_VIEW_CAMERA_TARGET,
        }
    )
    return case


def cases() -> list[dict]:
    rows = [
        close_projector_case("dark_mage_samurai", floor_shadow=False),
        close_projector_case("parrot_snail", floor_shadow=False),
        close_projector_case("diverse5_thick1_13_21_23", floor_shadow=True),
        close_projector_case("ant_dragon_shrimp", floor_shadow=True),
    ]

    # Keep full-size geometry as each case's shadow caster while reducing only
    # camera-visible center geometry where it would cover a wall silhouette.
    # The offsets are parallel to the corresponding orthographic view axes:
    # they preserve screen placement while keeping the display copies outside
    # their proxies' shadow volumes and retaining the approved gold shading.
    rows[0].update(
        {
            "camera_model_scale": 0.80,
            "camera_model_offset": (-30.0, -30.0, 16.0),
        }
    )
    rows[1].update(
        {
            "camera_model_scale": 0.70,
            "camera_model_offset": (-30.0, -30.0, 16.0),
        }
    )
    rows[2].update(
        {
            "camera_model_scale": 0.80,
            "camera_model_offset": (-30.0, -30.0, 17.0),
        }
    )

    # Register the raw delivery STL so shrimp, ant, and dragon reach the
    # visual left wall, right wall, and floor, respectively.  This complete
    # raw-STL basis is the one verified by the accepted high-resolution
    # room render; the older orientation index referred to a differently
    # preregistered STL and is therefore intentionally not reused here.
    rows[3].update(
        {
            "basis": ANT_DRAGON_SHRIMP_BASIS,
            "camera_ortho_scale": UNIFIED_CAMERA_ORTHO_SCALE,
            "camera_target": THREE_VIEW_CAMERA_TARGET,
        }
    )

    # Keep the audited row-5 orientation while using the same physical model,
    # projector, and light-pool scale as every other gallery case.  Its wider
    # camera framing is retained initially so all three receivers stay visible.
    row5 = named_case("urban_allies")
    row5.update(
        {
            "basis": ORIENTATION_13,
            "output_subdir": "renders_gallery_final_20260801",
            "output_name": "simulation_final.png",
            "blend_name": "urban_allies_gallery_final.blend",
            "display_extent": UNIFIED_DISPLAY_EXTENT,
            "receiver_radius": UNIFIED_RECEIVER_RADIUS,
            "wall_source_distance": UNIFIED_SOURCE_DISTANCE,
            "floor_source_height": UNIFIED_SOURCE_DISTANCE,
            "light_energy": UNIFIED_LIGHT_ENERGY,
            "camera_ortho_scale": UNIFIED_CAMERA_ORTHO_SCALE,
            "camera_target": THREE_VIEW_CAMERA_TARGET,
            # Preserve the full-size physical projections while reducing only
            # the camera-visible center model so it does not cover the shadows.
            "camera_model_scale": 0.55,
            "camera_model_offset": (-30.0, -30.0, 17.0),
        }
    )
    rows.append(row5)
    return rows


def main() -> None:
    selected = os.environ.get("PCRAFT_GALLERY_CASE")
    for case in cases():
        if selected and case["name"] != selected:
            continue
        if not case["mesh"].is_file():
            raise FileNotFoundError(case["mesh"])
        renderer.render_case(case)


if __name__ == "__main__":
    main()
