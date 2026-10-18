from __future__ import annotations

import math

import pytest

from shadowforge_pipeline.soft_voxel_v6.statistics import (
    evaluate_soft_outcome,
    exact_sign_flip_pvalue,
    holm_adjust_two,
    ordered_mean,
)


TASKS = tuple(f"connected_{index:02d}" for index in range(1, 11))
ARMS = ("base", "j_only", "h_only", "full")


def _rows(
    *,
    full_iou: float = 0.96,
    base_iou: float = 0.94,
    j_effect: float = 0.01,
    h_effect: float = 0.01,
    j_dice: float = 0.001,
    h_fpr: float = -0.001,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in TASKS:
        values = {
            "base": (base_iou, 0.95, 0.98, 0.010),
            "j_only": (full_iou - h_effect, 0.959, 0.98, 0.010 - h_fpr),
            "h_only": (full_iou - j_effect, 0.959 - j_dice, 0.98, 0.009),
            "full": (full_iou, 0.959, 0.98, 0.010),
        }
        for arm in ARMS:
            iou, dice, recall, fpr = values[arm]
            rows.append(
                {
                    "task": task,
                    "arm": arm,
                    "mean_iou": iou,
                    "mean_dice": dice,
                    "mean_recall": recall,
                    "mean_fpr": fpr,
                }
            )
    return rows


def _activity(*, zeros: int = 0) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    result: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for task in TASKS:
        result[task] = {
            "base": {},
            "j_only": {"j": {"eligible": 2100, "zero": zeros, "active": 2100 - zeros}},
            "h_only": {"h": {"eligible": 2100, "zero": zeros, "active": 2100 - zeros}},
            "full": {
                "j": {"eligible": 2100, "zero": zeros, "active": 2100 - zeros},
                "h": {"eligible": 2100, "zero": zeros, "active": 2100 - zeros},
            },
        }
    return result


def test_ordered_mean_is_exact_task_equal_and_rejects_aliases() -> None:
    values = [1.0] + [0.0] * 9
    assert ordered_mean(values) == 0.1
    for bad in (values[:9], values + [0.0], [True] + values[1:], [math.nan] + values[1:]):
        with pytest.raises(ValueError):
            ordered_mean(bad)


def test_exact_sign_flip_enumerates_all_1024_inclusive_without_pseudocount() -> None:
    assert exact_sign_flip_pvalue([1.0] * 10) == 1 / 1024
    assert exact_sign_flip_pvalue([0.0] * 10) == 1.0
    assert exact_sign_flip_pvalue([-0.0] * 10) == 1.0


def test_holm_uses_j_then_h_tie_order_and_exact_point05_fails() -> None:
    tied = holm_adjust_two(0.02, 0.02)
    assert tied == {"j": 0.04, "h": 0.04}
    assert holm_adjust_two(0.025, 0.04) == {"j": 0.05, "h": 0.05}


def test_all_registered_gates_allow_the_only_two_loss_claim() -> None:
    report = evaluate_soft_outcome(_rows(), _activity())
    assert report["SOFT_PASS"] is True
    assert report["BOTH_CONFIRMED"] is True
    assert report["claim_allowed"] is True
    assert report["failure_codes"] == []


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("full_iou", math.nextafter(0.95, 0.0), "QUALITY_FULL_MEAN_IOU"),
        ("full_iou", math.nextafter(0.94, 0.0), "QUALITY_FULL_WORST_IOU"),
        ("j_effect", -math.nextafter(0.005, math.inf), "J_IOU_REVERSE"),
        ("h_effect", -math.nextafter(0.005, math.inf), "H_IOU_REVERSE"),
    ],
)
def test_quality_and_reverse_next_float_boundaries(field: str, value: float, code: str) -> None:
    kwargs = {field: value}
    report = evaluate_soft_outcome(_rows(**kwargs), _activity())
    assert code in report["failure_codes"]


def test_exact_quality_and_recall_floors_pass() -> None:
    rows = _rows(full_iou=(9.5 - 0.94) / 9.0)
    for row in rows:
        if row["task"] == "connected_01" and row["arm"] == "full":
            row["mean_iou"] = 0.94
        if row["arm"] in {"j_only", "h_only", "full"}:
            row["mean_recall"] = 0.0
        elif row["arm"] == "base":
            row["mean_recall"] = 0.005
    report = evaluate_soft_outcome(rows, _activity())
    assert "QUALITY_FULL_MEAN_IOU" not in report["failure_codes"]
    assert "QUALITY_FULL_WORST_IOU" not in report["failure_codes"]
    assert "RECALL_SAFETY" not in report["failure_codes"]


def test_recall_next_float_below_floor_fails() -> None:
    rows = _rows()
    rows[1]["mean_recall"] = 0.98 + math.nextafter(-0.005, -math.inf)
    report = evaluate_soft_outcome(rows, _activity())
    assert report["gates"]["recall_safety"] is False


def test_210_zero_passes_and_211_fails_without_fraction_rounding() -> None:
    assert evaluate_soft_outcome(_rows(), _activity(zeros=210))["gates"]["engineering_activity"] is True
    failed = evaluate_soft_outcome(_rows(), _activity(zeros=211))
    assert failed["gates"]["engineering_activity"] is False
    assert "ENGINEERING_ACTIVITY" in failed["failure_codes"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(dict(rows[-1])),
        lambda rows: rows.__setitem__(0, {**rows[0], "task": "connected_02"}),
        lambda rows: rows.__setitem__(0, {**rows[0], "mean_iou": True}),
        lambda rows: rows.__setitem__(0, {**rows[0], "mean_dice": math.inf}),
    ],
)
def test_rows_fail_closed_on_missing_extra_reordered_alias_or_nonfinite(mutation) -> None:
    rows = _rows()
    mutation(rows)
    with pytest.raises(ValueError):
        evaluate_soft_outcome(rows, _activity())


def test_each_individual_role_gate_can_fail() -> None:
    j_dice = evaluate_soft_outcome(_rows(j_dice=-0.001), _activity())
    h_fpr = evaluate_soft_outcome(_rows(h_fpr=0.001), _activity())
    assert j_dice["gates"]["j_dice_role"] is False
    assert h_fpr["gates"]["h_fpr_role"] is False
    assert j_dice["claim_allowed"] is False
    assert h_fpr["claim_allowed"] is False


def test_positive_count_signflip_and_activity_are_independent_gates() -> None:
    rows = _rows()
    for task_index in (0, 1, 2):
        full = rows[task_index * 4 + 3]
        h_only = rows[task_index * 4 + 2]
        full["mean_iou"] = h_only["mean_iou"]
    positive = evaluate_soft_outcome(rows, _activity())
    assert positive["gates"]["j_iou_positive_count"] is False
    activity = _activity()
    activity["connected_01"]["full"]["j"] = {"eligible": 0, "zero": 0, "active": 0}
    assert evaluate_soft_outcome(_rows(), activity)["gates"]["engineering_activity"] is False
