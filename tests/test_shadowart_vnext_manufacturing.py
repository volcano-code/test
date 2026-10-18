"""Physical feature regressions: surviving main bodies cannot hide lost fins."""
import importlib
from pathlib import Path

import numpy as np
import pytest


def module():
    source = Path(__file__).parents[1] / "experiments/shadowart_vnext/manufacturing.py"
    assert source.is_file(), "physical manufacturing implementation is missing"
    return importlib.import_module("experiments.shadowart_vnext.manufacturing")


def test_thick_block_is_certified_without_using_boundary_min_edt():
    v = np.zeros((24,) * 3, bool)
    v[4:20, 4:20, 4:20] = True
    report = module().thickness_audit(v, .25)
    assert report["passed"]
    assert report["thin_voxels"] == 0
    assert report["component_count_6"] == 1
    assert report["local_thickness_mm"]["minimum_lower_bound"] >= .9


def test_closed_cavity_is_preserved_and_thick_walls_pass():
    v = np.zeros((28,) * 3, bool)
    v[3:25, 3:25, 3:25] = True
    v[9:19, 9:19, 9:19] = False
    before = v.copy()
    report = module().thickness_audit(v, .25)
    assert report["passed"]
    assert np.array_equal(v, before)


def test_wholly_eroded_attached_sheet_cannot_hide_behind_one_main_lobe():
    v = np.zeros((36,) * 3, bool)
    v[3:17, 9:27, 9:27] = True
    v[17:32, 12:24, 17:18] = True
    report = module().thickness_audit(v, .25)
    assert not report["passed"]
    assert report["component_count_6"] == 1
    assert report["thin_voxels"] > 100
    assert report["vanished_appendage_count"] >= 1
    assert any(row["classification"] == "thin_sheet" for row in report["thin_regions"])


def test_thin_neck_between_large_lobes_fails_at_multiple_scales():
    v = np.zeros((40,) * 3, bool)
    v[3:15, 10:30, 10:30] = True
    v[25:37, 10:30, 10:30] = True
    v[15:25, 19:21, 19:21] = True
    report = module().thickness_audit(v, .25)
    assert not report["passed"]
    assert any(row["surviving_components"] > 1 for row in report["neck_scales"])
    assert "NECK_SEPARATION" in report["reasons"]


def test_wall_valid_long_rod_erased_at_neck_scale_cannot_hide_behind_main_body():
    v = np.zeros((36,) * 3, bool)
    v[3:17, 8:28, 8:28] = True
    v[17:31, 16:20, 16:20] = True
    report = module().thickness_audit(v, .25)
    assert report["thin_voxels"] == 0  # The rod satisfies .9mm wall, not1.35mm neck.
    assert not report["passed"]
    assert "VANISHED_NECK_APPENDAGE" in report["reasons"]
    assert report["neck_vanished_appendage_count"] >= 1
    scale = next(row for row in report["neck_scales"] if row["requested_width_mm"] == 1.35)
    assert scale["surviving_components"] == 1
    assert any(row["classification"] == "long_neck_scale_vanished_appendage"
               for row in scale["vanished_wall_valid_regions"])


def test_diagonal_thin_wall_is_not_mistaken_for_long_axis_thickness():
    x, y, z = np.indices((30,) * 3)
    v = (abs(x - y) <= 1) & (x >= 4) & (x < 26) & (y >= 4) & (y < 26) & (z >= 4) & (z < 26)
    report = module().thickness_audit(v, .25)
    assert not report["passed"]
    assert report["thin_voxels"] > 0


def test_disconnected_bodies_and_underresolved_features_do_not_pass():
    v = np.zeros((20,) * 3, bool)
    v[2:8, 2:8, 2:8] = True
    v[12:18, 12:18, 12:18] = True
    assert not module().thickness_audit(v, .25)["passed"]
    tiny = np.zeros((12,) * 3, bool)
    tiny[2:10, 2:10, 2:10] = True
    tiny[10, 5, 5] = True
    report = module().thickness_audit(tiny, 1.)
    assert not report["passed"]
    assert report["review_required"]


@pytest.mark.parametrize("pitch", [0., -1., float("nan")])
def test_invalid_physical_scale_is_rejected(pitch):
    with pytest.raises(ValueError):
        module().thickness_audit(np.ones((3,) * 3, bool), pitch)
