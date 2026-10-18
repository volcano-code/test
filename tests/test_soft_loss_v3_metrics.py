"""Behavioral contracts for exact soft-loss v3 delivery metrics."""

from __future__ import annotations

import json
import math
from numbers import Real

import pytest
import torch

from shadowforge_pipeline.soft_voxel_v3.metrics import (
    DeliveryMetricsPreflightError,
    evaluate_delivery_metrics,
)


THRESHOLDS = (0.04, 0.05, 0.06)


def _repeat_mask(mask: torch.Tensor, *, device: torch.device | str = "cpu") -> torch.Tensor:
    return mask.to(device=device, dtype=torch.bool).unsqueeze(0).repeat(6, 1, 1)


def _masks(mask: torch.Tensor, *, device: torch.device | str = "cpu") -> dict[float, torch.Tensor]:
    repeated = _repeat_mask(mask, device=device)
    return {threshold: repeated.clone() for threshold in THRESHOLDS}


def _single_pixel_target(
    *, height: int = 3, width: int = 3, device: torch.device | str = "cpu"
) -> torch.Tensor:
    target = torch.zeros((6, height, width), dtype=torch.float32, device=device)
    target[:, height // 2, width // 2] = 1.0
    return target


@pytest.mark.parametrize(
    ("target", "masks", "message"),
    [
        (torch.zeros((6, 3, 3)), {0.04: torch.zeros((6, 3, 3), dtype=torch.bool)}, "threshold"),
        (
            torch.zeros((6, 3, 3)),
            {"0.04": torch.zeros((6, 3, 3), dtype=torch.bool),
             "0.05": torch.zeros((6, 3, 3), dtype=torch.bool),
             "0.06": torch.zeros((6, 3, 3), dtype=torch.bool)},
            "numeric",
        ),
        (
            torch.zeros((6, 3, 3)),
            {threshold: torch.zeros((6, 3, 3)) for threshold in THRESHOLDS},
            "boolean",
        ),
        (
            torch.zeros((6, 3, 3)),
            {threshold: torch.zeros((3, 3), dtype=torch.bool) for threshold in THRESHOLDS},
            "six non-empty views",
        ),
        (
            torch.zeros((5, 3, 3)),
            {threshold: torch.zeros((5, 3, 3), dtype=torch.bool) for threshold in THRESHOLDS},
            "six non-empty views",
        ),
        (
            torch.zeros((6, 0, 3)),
            {threshold: torch.zeros((6, 0, 3), dtype=torch.bool) for threshold in THRESHOLDS},
            "six non-empty views",
        ),
        (
            torch.zeros((6, 3, 3)),
            {threshold: torch.zeros((6, 4, 3), dtype=torch.bool) for threshold in THRESHOLDS},
            "same shape",
        ),
    ],
)
def test_preflight_rejects_non_masks_wrong_thresholds_and_wrong_shapes(
    target: torch.Tensor, masks: object, message: str
) -> None:
    """Break caught: projections or malformed view sets reach metric calculation."""

    with pytest.raises(DeliveryMetricsPreflightError, match=message):
        evaluate_delivery_metrics(target, masks)  # type: ignore[arg-type]


def test_preflight_rejects_non_tensor_and_nonfinite_target() -> None:
    """Break caught: target thresholding runs before type and finiteness checks."""

    valid_masks = _masks(torch.zeros((3, 3), dtype=torch.bool))
    with pytest.raises(DeliveryMetricsPreflightError, match="torch.Tensor"):
        evaluate_delivery_metrics([], valid_masks)  # type: ignore[arg-type]

    nonfinite = _single_pixel_target()
    nonfinite[0, 0, 0] = float("nan")
    with pytest.raises(DeliveryMetricsPreflightError, match="finite"):
        evaluate_delivery_metrics(nonfinite, valid_masks)


def test_fpr_is_background_normalized_without_epsilon() -> None:
    """Break caught: FPR uses all pixels, pooled counts, or epsilon smoothing."""

    target = _single_pixel_target()
    prediction = target[0].to(torch.bool)
    prediction[0, 0] = True

    report = evaluate_delivery_metrics(target, _masks(prediction))
    view = report["thresholds"]["0.05"]["per_view"][0]

    assert (view["tp"], view["fp"], view["fn"], view["tn"]) == (1, 1, 0, 7)
    assert view["fpr"] == 0.125
    assert view["invalid_background_denominator"] is False


def test_task_fpr_is_equal_view_mean_not_pooled_background_counts() -> None:
    """Break caught: views with more target background dominate task FPR."""

    target_binary = torch.zeros((6, 3, 4), dtype=torch.bool)
    prediction = torch.zeros_like(target_binary)
    false_positives = (0, 1, 0, 2, 0, 3)
    for view_index, (foreground, fp_count) in enumerate(
        zip(range(1, 7), false_positives)
    ):
        target_binary[view_index].view(-1)[:foreground] = True
        prediction[view_index] = target_binary[view_index]
        prediction[view_index].view(-1)[foreground : foreground + fp_count] = True

    masks = {threshold: prediction.clone() for threshold in THRESHOLDS}
    report = evaluate_delivery_metrics(target_binary.to(torch.float32), masks)
    per_view = report["thresholds"]["0.05"]["per_view"]
    task_fpr = report["thresholds"]["0.05"]["aggregate"]["mean_fpr"]

    assert [view["fpr"] for view in per_view] == pytest.approx(
        [0.0, 0.1, 0.0, 0.25, 0.0, 0.5]
    )
    assert task_fpr == pytest.approx(17.0 / 120.0)
    pooled_fpr = 6.0 / 51.0
    assert task_fpr != pytest.approx(pooled_fpr)


def test_task_recall_is_equal_view_mean_not_pooled_foreground_counts() -> None:
    """Break caught: views with more target foreground dominate task recall."""

    target_binary = torch.zeros((6, 3, 4), dtype=torch.bool)
    prediction = torch.zeros_like(target_binary)
    true_positives = (1, 1, 1, 2, 1, 3)
    for view_index, (foreground, tp_count) in enumerate(
        zip(range(1, 7), true_positives)
    ):
        target_binary[view_index].view(-1)[:foreground] = True
        prediction[view_index].view(-1)[:tp_count] = True

    masks = {threshold: prediction.clone() for threshold in THRESHOLDS}
    report = evaluate_delivery_metrics(target_binary.to(torch.float32), masks)
    per_view = report["thresholds"]["0.05"]["per_view"]
    task_recall = report["thresholds"]["0.05"]["aggregate"]["mean_recall"]

    assert [(view["tp"], view["fn"]) for view in per_view] == [
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 2),
        (1, 4),
        (3, 3),
    ]
    assert [view["recall"] for view in per_view] == pytest.approx(
        [1.0, 0.5, 1.0 / 3.0, 0.5, 0.2, 0.5]
    )
    assert task_recall == pytest.approx(91.0 / 180.0)
    pooled_recall = 9.0 / 21.0
    assert task_recall != pytest.approx(pooled_recall)


def test_zero_background_is_none_and_invalid_not_smoothed() -> None:
    """Break caught: a foreground-only target receives a fabricated finite FPR."""

    target = torch.ones((6, 3, 3), dtype=torch.float32)
    report = evaluate_delivery_metrics(target, _masks(torch.ones((3, 3), dtype=torch.bool)))
    view = report["thresholds"]["0.05"]["per_view"][0]
    aggregate = report["thresholds"]["0.05"]["aggregate"]

    assert view["fpr"] is None
    assert view["invalid_background_denominator"] is True
    assert aggregate["mean_fpr"] is None
    assert aggregate["invalid_background_denominator"] is True
    assert aggregate["invalid_task_metric"] is True
    assert report["invalid_task_metric"] is True


def test_target_point_four_nine_and_five_zero_control_all_binary_geometry() -> None:
    """Break caught: target metrics use rounding, positivity, or a second mask rule."""

    target = torch.tensor([[[0.49, 0.50]]], dtype=torch.float32).repeat(6, 1, 1)
    already_binary = torch.tensor([[False, True]])

    report = evaluate_delivery_metrics(target, _masks(already_binary))
    view = report["thresholds"]["0.05"]["per_view"][0]

    assert (view["tp"], view["fp"], view["fn"], view["tn"]) == (1, 0, 0, 1)
    assert view["iou"] == 1.0
    assert view["dice"] == 1.0
    assert view["boundary_f1_2px"] == 1.0
    assert view["mscd_px"] == 0.0
    assert view["p95_symmetric_distance_px"] == 0.0


def test_threshold_summary_means_views_equally_then_takes_minimum_and_range() -> None:
    """Break caught: threshold IoU is pooled by foreground-pixel count."""

    target_binary = torch.zeros((6, 3, 7), dtype=torch.bool)
    for view_index in range(6):
        target_binary[view_index, 1, : view_index + 1] = True
    target = target_binary.to(torch.float32)

    masks = {threshold: target_binary.clone() for threshold in THRESHOLDS}
    masks[0.06][3:] = False

    report = evaluate_delivery_metrics(target, masks)

    assert report["task_iou_by_threshold"] == {"0.04": 1.0, "0.05": 1.0, "0.06": 0.5}
    assert report["worst_threshold_iou"] == 0.5
    assert report["threshold_iou_range"] == 0.5


def test_top_level_contour_metrics_use_exactly_threshold_point_zero_five() -> None:
    """Break caught: primary contour endpoints are read from 0.04 or 0.06."""

    target_mask = torch.zeros((5, 9), dtype=torch.bool)
    target_mask[2, 3] = True
    shifted_one = torch.zeros_like(target_mask)
    shifted_one[2, 4] = True
    shifted_two = torch.zeros_like(target_mask)
    shifted_two[2, 5] = True
    masks = {
        0.04: _repeat_mask(shifted_one),
        0.05: _repeat_mask(target_mask),
        0.06: _repeat_mask(shifted_two),
    }

    report = evaluate_delivery_metrics(
        _repeat_mask(target_mask).to(torch.float32), masks
    )
    aggregate_004 = report["thresholds"]["0.04"]["aggregate"]
    aggregate_005 = report["thresholds"]["0.05"]["aggregate"]
    aggregate_006 = report["thresholds"]["0.06"]["aggregate"]

    assert aggregate_004["mean_mscd_px"] > 0.0
    assert aggregate_004["mean_p95_symmetric_distance_px"] == 1.0
    assert aggregate_005["mean_mscd_px"] == 0.0
    assert aggregate_005["mean_p95_symmetric_distance_px"] == 0.0
    assert aggregate_006["mean_mscd_px"] > aggregate_004["mean_mscd_px"]
    assert aggregate_006["mean_p95_symmetric_distance_px"] == 2.0
    assert report["task_mscd_px"] == 0.0
    assert report["task_p95_symmetric_distance_px"] == 0.0


def test_task_contour_metrics_equal_mean_heterogeneous_views_not_sample_weighted() -> None:
    """Break caught: longer view contours receive more task-level weight."""

    target_binary = torch.zeros((6, 7, 12), dtype=torch.bool)
    prediction = torch.zeros_like(target_binary)
    for view_index, length in enumerate(range(1, 7)):
        target_binary[view_index, 3, 2 : 2 + length] = True
        start = 2 if view_index < 3 else 3
        prediction[view_index, 3, start : start + length] = True

    masks = {
        0.04: target_binary.clone(),
        0.05: prediction,
        0.06: target_binary.clone(),
    }
    report = evaluate_delivery_metrics(target_binary.to(torch.float32), masks)
    per_view = report["thresholds"]["0.05"]["per_view"]
    aggregate = report["thresholds"]["0.05"]["aggregate"]

    one_plus_three_root_half = 1.0 + 3.0 * math.sqrt(0.5)
    expected_mscd = [
        0.0,
        0.0,
        0.0,
        one_plus_three_root_half / 10.0,
        one_plus_three_root_half / 12.0,
        one_plus_three_root_half / 14.0,
    ]
    expected_task_mscd = 107.0 * one_plus_three_root_half / 2520.0

    assert [view["mscd_px"] for view in per_view] == pytest.approx(expected_mscd)
    assert [view["p95_symmetric_distance_px"] for view in per_view] == [
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
    ]
    assert aggregate["mean_mscd_px"] == pytest.approx(expected_task_mscd)
    assert aggregate["mean_p95_symmetric_distance_px"] == 0.5
    assert report["task_mscd_px"] == pytest.approx(expected_task_mscd)
    assert report["task_p95_symmetric_distance_px"] == 0.5

    contour_count_weighted_mscd = one_plus_three_root_half / 18.0
    contour_count_weighted_p95 = 2.0 / 3.0
    assert report["task_mscd_px"] != pytest.approx(contour_count_weighted_mscd)
    assert report["task_p95_symmetric_distance_px"] != pytest.approx(
        contour_count_weighted_p95
    )


def test_boundary_f1_includes_points_exactly_at_two_pixel_tolerance() -> None:
    """Break caught: the exact tolerance boundary is excluded with a strict inequality."""

    target_mask = torch.zeros((5, 7), dtype=torch.bool)
    target_mask[2, 2] = True
    predicted_mask = torch.zeros_like(target_mask)
    predicted_mask[2, 4] = True

    report = evaluate_delivery_metrics(
        _repeat_mask(target_mask).to(torch.float32), _masks(predicted_mask)
    )
    view = report["thresholds"]["0.05"]["per_view"][0]

    assert view["boundary_f1_2px"] == 1.0
    assert view["mscd_px"] == pytest.approx((1.0 + math.sqrt(2.5) + math.sqrt(2.5) + 2.0) / 4.0)


def test_empty_target_boundary_fails_typed_preflight_before_a_report() -> None:
    """Break caught: both-empty IoU/Dice silently bless invalid experiment input."""

    target = torch.zeros((6, 3, 4), dtype=torch.float32)
    with pytest.raises(DeliveryMetricsPreflightError, match="target boundary.*view 0"):
        evaluate_delivery_metrics(target, _masks(torch.zeros((3, 4), dtype=torch.bool)))


def test_empty_prediction_gets_diagonal_contour_penalties_and_explicit_flag() -> None:
    """Break caught: an empty prediction is dropped, NaN, or under-penalized."""

    target = _single_pixel_target(height=3, width=4)
    report = evaluate_delivery_metrics(target, _masks(torch.zeros((3, 4), dtype=torch.bool)))
    view = report["thresholds"]["0.05"]["per_view"][0]

    assert view["iou"] == 0.0
    assert view["dice"] == 0.0
    assert view["boundary_f1_2px"] == 0.0
    assert view["mscd_px"] == 5.0
    assert view["p95_symmetric_distance_px"] == 5.0
    assert view["invalid_predicted_boundary"] is True
    assert report["task_mscd_px"] == 5.0
    assert report["thresholds"]["0.05"]["aggregate"]["invalid_task_metric"] is True
    assert report["invalid_task_metric"] is True
    json.dumps(report, allow_nan=False)


def test_report_contains_only_json_safe_builtin_scalars() -> None:
    """Break caught: tensors, NumPy scalars, or non-finite values leak into artifacts."""

    target = _single_pixel_target()
    report = evaluate_delivery_metrics(target, _masks(target[0].to(torch.bool)))

    serialized = json.dumps(report, allow_nan=False)
    assert json.loads(serialized)["task_mscd_px"] == 0.0

    def assert_builtin(value: object) -> None:
        if isinstance(value, dict):
            assert all(isinstance(key, str) for key in value)
            for nested in value.values():
                assert_builtin(nested)
        elif isinstance(value, list):
            for nested in value:
                assert_builtin(nested)
        else:
            assert value is None or isinstance(value, (str, bool, int, float))
            if isinstance(value, Real) and not isinstance(value, (bool, int)):
                assert math.isfinite(float(value))

    assert_builtin(report)


def test_metrics_run_on_cpu() -> None:
    """Break caught: metric evaluation assumes a CUDA tensor or device."""

    target = _single_pixel_target(device="cpu")
    report = evaluate_delivery_metrics(target, _masks(target[0].to(torch.bool), device="cpu"))
    assert report["task_mscd_px"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu_and_mixed_devices_fail_preflight() -> None:
    """Break caught: CPU conversion changes semantics or mixed-device masks are accepted."""

    cpu_target = _single_pixel_target()
    cpu_masks = _masks(cpu_target[0].to(torch.bool))
    cuda_target = cpu_target.cuda()
    cuda_masks = {threshold: mask.cuda() for threshold, mask in cpu_masks.items()}

    assert evaluate_delivery_metrics(cuda_target, cuda_masks) == evaluate_delivery_metrics(
        cpu_target, cpu_masks
    )
    cuda_masks[0.06] = cpu_masks[0.06]
    with pytest.raises(DeliveryMetricsPreflightError, match="same device"):
        evaluate_delivery_metrics(cuda_target, cuda_masks)
