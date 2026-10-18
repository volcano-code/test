"""Exact finite-candidate union selection for regional ray footprints."""
import importlib
import itertools

import numpy as np
import pytest


def solver_module():
    spec = importlib.util.find_spec("experiments.shadowart_vnext.regional_solver")
    assert spec is not None, "regional union solver has not been implemented"
    return importlib.import_module("experiments.shadowart_vnext.regional_solver")


def _actual_iou(prediction, target):
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return 1.0 if union == 0 else intersection / union


def test_exhaustive_hand_checked_family_has_one_floor_satisfying_union():
    target = np.array([[[1, 1], [0, 0]]], dtype=bool)
    base = np.zeros_like(target)
    additions = np.array(
        [
            [[[1, 0], [0, 0]]],
            [[[1, 0], [1, 0]]],
            [[[0, 1], [0, 0]]],
        ],
        dtype=bool,
    )

    feasible = []
    for size in range(3):
        for subset in itertools.combinations(range(3), size):
            prediction = base.copy()
            for index in subset:
                prediction |= additions[index]
            if _actual_iou(prediction, target) >= 1.0:
                feasible.append(subset)
    assert feasible == [(0, 2)]

    selected, report = solver_module().solve_union(
        base, target, additions, np.array([1.0]), max_selected=2
    )

    assert selected == [0, 2]
    assert report["status"] == "optimal"
    assert report["selected"] == [0, 2]
    assert report["exact_ious"] == [1.0]
    assert report["backend"] == "scipy.optimize.milp"
    assert report["backend_version"]
    assert "finite candidate family" in report["caveat"].lower()
    assert "exact mean-iou" in report["objective_caveat"].lower()
    assert report["printability_claim"] is False


def test_two_additions_are_required_when_each_alone_misses_a_view_floor():
    targets = np.ones((2, 1, 1), dtype=bool)
    base = np.zeros_like(targets)
    additions = np.array(
        [
            [[[1]], [[0]]],
            [[[0]], [[1]]],
        ],
        dtype=bool,
    )

    selected, report = solver_module().solve_union(
        base, targets, additions, [1.0, 1.0], max_selected=2
    )

    assert selected == [0, 1]
    assert report["exact_ious"] == [1.0, 1.0]


def test_overlapping_additions_are_not_double_counted_as_independent_gains():
    target = np.array([[[1, 1], [0, 0]]], dtype=bool)
    base = np.zeros_like(target)
    same_single_pixel = np.array([[[[1, 0], [0, 0]]]] * 2, dtype=bool)

    selected, report = solver_module().solve_union(
        base, target, same_single_pixel, [1.0], max_selected=2
    )

    assert selected is None
    assert report["status"] == "infeasible_for_finite_candidate_family"
    assert report["selected"] is None


def test_conservative_millionth_floor_rejects_iou_just_below_ceiling():
    target = np.array([[[1, 1], [1, 0]]], dtype=bool)
    base = np.array([[[1, 0], [0, 0]]], dtype=bool)
    additions = np.zeros((0, 1, 2, 2), dtype=bool)

    selected, report = solver_module().solve_union(
        base, target, additions, [0.3333331], max_selected=0
    )

    assert selected is None
    assert report["effective_floors"] == [0.333334]
    assert report["exact_ious"] == [pytest.approx(1 / 3)]


def test_empty_target_semantics_and_k_zero_return_empty_selection():
    target = np.zeros((1, 2, 2), dtype=bool)
    base = np.zeros_like(target)
    additions = np.zeros((0, 1, 2, 2), dtype=bool)

    selected, report = solver_module().solve_union(
        base, target, additions, [1.0], max_selected=0
    )

    assert selected == []
    assert report["exact_ious"] == [1.0]
    assert report["status"] == "optimal"


def test_empty_target_with_an_existing_false_positive_is_impossible():
    target = np.zeros((1, 1, 1), dtype=bool)
    base = np.ones_like(target)
    additions = np.zeros((0, 1, 1, 1), dtype=bool)

    selected, report = solver_module().solve_union(
        base, target, additions, [0.01], max_selected=0
    )

    assert selected is None
    assert report["exact_ious"] == [0.0]
    assert report["status"] == "infeasible_for_finite_candidate_family"


def test_zero_selection_budget_cannot_use_an_otherwise_sufficient_addition():
    target = np.ones((1, 1, 1), dtype=bool)
    base = np.zeros_like(target)
    additions = np.ones((1, 1, 1, 1), dtype=bool)

    selected, report = solver_module().solve_union(
        base, target, additions, [1.0], max_selected=0
    )

    assert selected is None
    assert report["status"] == "infeasible_for_finite_candidate_family"


def test_objective_prefers_foreground_gain_without_unnecessary_background():
    target = np.array([[[1, 1], [0, 0]]], dtype=bool)
    base = np.array([[[1, 0], [0, 0]]], dtype=bool)
    additions = np.array(
        [
            [[[0, 1], [0, 0]]],
            [[[0, 1], [1, 0]]],
        ],
        dtype=bool,
    )

    selected, report = solver_module().solve_union(
        base, target, additions, [0.0], max_selected=1
    )

    assert selected == [0]
    assert report["exact_ious"] == [1.0]


def test_inputs_are_not_mutated_or_retyped_in_place():
    base = np.array([[[0, 0], [0, 0]]], dtype=np.uint8)
    target = np.array([[[1, 0], [0, 0]]], dtype=np.int16)
    additions = np.array([[[[1, 0], [0, 0]]]], dtype=np.float32)
    floors = np.array([1.0], dtype=np.float64)
    snapshots = [value.copy() for value in (base, target, additions, floors)]

    selected, _ = solver_module().solve_union(
        base, target, additions, floors, max_selected=1
    )

    assert selected == [0]
    for value, snapshot in zip((base, target, additions, floors), snapshots):
        assert value.dtype == snapshot.dtype
        assert np.array_equal(value, snapshot)


@pytest.mark.parametrize(
    "base,target,additions,floors,max_selected,time_limit",
    [
        (np.zeros((0, 2, 2)), np.zeros((0, 2, 2)), np.zeros((0, 0, 2, 2)), [], 0, 1.0),
        (np.zeros((1, 2, 3)), np.zeros((1, 2, 3)), np.zeros((0, 1, 2, 3)), [0.0], 0, 1.0),
        (np.full((1, 1, 1), 2), np.zeros((1, 1, 1)), np.zeros((0, 1, 1, 1)), [0.0], 0, 1.0),
        (np.zeros((1, 1, 1)), np.zeros((1, 2, 2)), np.zeros((0, 1, 1, 1)), [0.0], 0, 1.0),
        (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((1, 2, 1, 1)), [0.0], 0, 1.0),
        (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((257, 1, 1, 1)), [0.0], 1, 1.0),
        (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((0, 1, 1, 1)), [np.nan], 0, 1.0),
        (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((0, 1, 1, 1)), [1.01], 0, 1.0),
        (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((1, 1, 1, 1)), [0.0], -1, 1.0),
        (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((1, 1, 1, 1)), [0.0], 2, 1.0),
        (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((1, 1, 1, 1)), [0.0], 1.0, 1.0),
        (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((0, 1, 1, 1)), [0.0], 0, 0.0),
        (np.zeros((1, 1, 1)), np.zeros((1, 1, 1)), np.zeros((0, 1, 1, 1)), [0.0], 0, np.inf),
    ],
)
def test_malformed_inputs_are_rejected(
    base, target, additions, floors, max_selected, time_limit
):
    with pytest.raises(ValueError):
        solver_module().solve_union(
            base,
            target,
            additions,
            floors,
            max_selected=max_selected,
            time_limit=time_limit,
        )
