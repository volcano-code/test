#!/usr/bin/env python3
"""Render axis-registration candidates for the five-case gallery audit."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

from mathutils import Matrix


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path.insert(0, str(ROOT / "experiments"))

import render_supp_gallery_simulations_four_cases_centered_20260801 as renderer


IDENTITY = Matrix.Identity(3)
ANT_YZ_A = Matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
)
ANT_YZ_B = Matrix(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
    )
)


def named_case(name: str) -> dict:
    return deepcopy(next(case for case in renderer.CASES if case["name"] == name))


def main() -> None:
    candidates = [
        ("dark_identity", "dark_mage_samurai", IDENTITY, False),
        ("parrot_identity", "parrot_snail", IDENTITY, False),
        ("diverse_identity", "diverse5_thick1_13_21_23", IDENTITY, True),
        ("ant_yz_a", "ant_dragon_shrimp", ANT_YZ_A, True),
        ("ant_yz_b", "ant_dragon_shrimp", ANT_YZ_B, True),
        (
            "urban_registered",
            "urban_allies",
            Matrix(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
            True,
        ),
    ]
    for label, name, basis, floor_shadow in candidates:
        case = named_case(name)
        case.update(
            {
                "basis": basis,
                "floor_shadow": floor_shadow,
                "output_subdir": "renders_axis_candidates_20260801",
                "output_name": f"{label}.png",
                "blend_name": f"{label}.blend",
                "display_extent": 58.0,
                "receiver_radius": 56.0,
                "wall_source_distance": 2000.0,
                "floor_source_height": 2000.0,
                "light_energy": 165_000_000.0,
                "camera_ortho_scale": 274.0,
                "camera_target": (10.0, 10.0, 10.0)
                if not floor_shadow
                else (10.0, 10.0, 0.0),
            }
        )
        renderer.render_case(case)


if __name__ == "__main__":
    main()
