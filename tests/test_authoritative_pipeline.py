from argparse import Namespace

import numpy as np

from shadowforge_pipeline.run import (
    choose_candidate,
    decouple_volume,
    n6,
    passes_core_candidate_gate,
    passes_manufacturing,
)
from candidate_selection_batch import fit_acs


def _args():
    return Namespace(
        min_iou=0.95,
        min_erosion_survival=0.15,
        min_saving=0.40,
        max_saving=0.85,
    )


def test_manufacturing_gate_requires_every_constraint():
    row = {
        "mean_iou": 0.98,
        "component_count_6": 1,
        "watertight": True,
        "mesh_parts": 1,
        "erosion_survival": 0.20,
        "material_saved_vs_hull": 0.55,
    }
    assert passes_manufacturing(row, _args())
    assert not passes_manufacturing({**row, "watertight": False}, _args())
    assert not passes_manufacturing({**row, "mesh_parts": 2}, _args())
    assert not passes_manufacturing({**row, "material_saved_vs_hull": 0.90}, _args())


def test_core_gate_defers_visual_iou_to_delivery():
    row = {
        "mean_iou": 0.80,
        "component_count_6": 1,
        "watertight": True,
        "mesh_parts": 1,
        "erosion_survival": 0.20,
        "material_saved_vs_hull": 0.55,
    }
    assert not passes_manufacturing(row, _args())
    assert passes_core_candidate_gate(row, _args())


def test_combined_selector_allows_soft_and_preserves_fixed_tie_break():
    rows = [
        {
            "candidate_id": "free:soft:learned_printable",
            "branch": "free",
            "source": "soft",
            "manufacturing_pass": True,
            "safety_pass": True,
            "material_saved_vs_hull": 0.61,
            "mean_iou": 0.99,
            "safety_score": 2.0,
        },
        {
            "candidate_id": "fixed:geometry:watertight_export",
            "branch": "fixed",
            "source": "geometry",
            "manufacturing_pass": True,
            "safety_pass": True,
            "material_saved_vs_hull": 0.60,
            "mean_iou": 0.985,
            "safety_score": 1.8,
        },
    ]
    selected = choose_candidate(rows)
    assert selected["candidate_id"] == "fixed:geometry:watertight_export"

    rows[1]["material_saved_vs_hull"] = 0.50
    selected = choose_candidate(rows)
    assert selected["candidate_id"] == "free:soft:learned_printable"


def test_decoupled_delivery_is_clipped_and_single_component():
    core = np.zeros((4, 4, 4), dtype=bool)
    core[1:3, 1:3, 1:3] = True
    hull = np.zeros((8, 8, 8), dtype=bool)
    hull[1:7, 1:7, 1:7] = True
    delivery = decouple_volume(core, hull, target=8, shell_voxels=1)
    assert delivery.shape == (8, 8, 8)
    assert not np.any(delivery & ~hull)
    assert delivery.any()
    assert n6(delivery) == 1


def test_recalibrated_acs_artifact_loads():
    model, q_safe, q_unsafe = fit_acs(0.01)
    assert model.predict([0.0]).shape == (1,)
    assert q_safe > 0.0
    assert q_unsafe > 0.0
