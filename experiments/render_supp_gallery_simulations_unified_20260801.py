#!/usr/bin/env python3
"""Render all gallery simulations with one evidence-preserving scene protocol."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from mathutils import Matrix


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
TEMPLATE = ROOT / "experiments/render_supp_gallery_arcane_casters_20260801.py"
GALLERY = ROOT / "paper_aaai_submission/supplement_gallery_one_case_20260801"


CASES = (
    {
        "name": "dark_mage_samurai",
        "mesh": ROOT
        / "results/anime_two_wall_fixed_no_floor_physical_axes_qualityfirst_10cm_256_20260710"
        / "dark_mage_left__samurai_right"
        / "dark_mage_left__samurai_right_upright_saving_10cm_256.stl",
        "out": GALLERY / "dark_mage_samurai/renders",
        # View 1 (native Y) -> left wall; View 2 (native X) -> right wall.
        "basis": Matrix(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        "floor": False,
    },
    {
        "name": "parrot_snail",
        "mesh": ROOT
        / "results/full_pipeline_batch_2v3v_20260730/2v_simple_parrot_snail"
        / "selected_display_fixed.stl",
        "out": GALLERY / "parrot_snail/renders",
        # View 1 (native Y) -> left wall; View 2 (native X) -> right wall.
        "basis": Matrix(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        "floor": False,
    },
    {
        "name": "diverse5_thick1_13_21_23",
        "mesh": ROOT
        / "results/anime30_diverse_fixed_50_60_10cm_256_20260707"
        / "diverse5_thick1_13_21_23"
        / "diverse5_thick1_13_21_23_carve_pf4_10cm_256.stl",
        "out": GALLERY / "diverse5_thick1_13_21_23/renders",
        # View 1 (native Z) -> left wall; View 2 (native X) -> right wall;
        # View 3 (native Y) -> floor. This is a proper cyclic rotation.
        "basis": Matrix(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        "floor": True,
    },
)


def load_template():
    spec = importlib.util.spec_from_file_location("gallery_simulation_template", TEMPLATE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Blender rendering template: {TEMPLATE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_template()
    for case in CASES:
        module.MESH = case["mesh"]
        module.OUT = case["out"]
        module.SIMULATION_BASIS = case["basis"]
        module.SIMULATION_ENABLE_FLOOR_PROJECTION = case["floor"]
        module.SIMULATION_BLEND_NAME = f"{case['name']}_simulation_unified.blend"
        module.OUT.mkdir(parents=True, exist_ok=True)
        module.render_simulation()
        print(module.OUT / "simulation.png")


if __name__ == "__main__":
    main()
