from argparse import Namespace

# Importing the authoritative entrypoint establishes the repository's existing
# generation/experiments module paths before the isolated backend is loaded.
from shadowforge_pipeline import run as _authoritative_run  # noqa: F401
from shadowforge_pipeline.fixed_xyz_carve import select_fixed_xyz_candidate


def _args():
    return Namespace(
        min_iou=0.95,
        min_erosion_survival=0.10,
        min_saving=0.30,
        max_saving=0.60,
    )


def _row(name, saving, iou=0.98, watertight=True):
    return {
        "candidate": name,
        "mean_iou": iou,
        "component_count_6": 1,
        "watertight": watertight,
        "erosion_survival": 0.20,
        "material_saved_vs_hull": saving,
    }


def test_auto_selection_prefers_gate_passing_candidate_nearest_target():
    rows = [
        _row("visual_hull", 0.00),
        _row("carve_pf3", 0.42),
        _row("carve_pf5", 0.51),
        _row("carve_pf6", 0.57),
    ]

    selected, accepted = select_fixed_xyz_candidate(
        rows,
        requested="auto",
        saving_target=0.50,
        args=_args(),
    )

    assert accepted
    assert selected["candidate"] == "carve_pf5"


def test_explicit_candidate_is_preserved_and_reports_failed_gate():
    rows = [
        _row("visual_hull", 0.00),
        _row("carve_pf6", 0.57, watertight=False),
    ]

    selected, accepted = select_fixed_xyz_candidate(
        rows,
        requested="carve_pf6",
        saving_target=0.50,
        args=_args(),
    )

    assert selected["candidate"] == "carve_pf6"
    assert not accepted


def test_auto_fallback_prefers_projection_quality_when_gate_is_empty():
    rows = [
        _row("carve_pf4", 0.49, iou=0.91, watertight=True),
        _row("visual_hull", 0.02, iou=0.94, watertight=True),
    ]

    selected, accepted = select_fixed_xyz_candidate(
        rows,
        requested="auto",
        saving_target=0.50,
        args=_args(),
    )

    assert not accepted
    assert selected["candidate"] == "visual_hull"
