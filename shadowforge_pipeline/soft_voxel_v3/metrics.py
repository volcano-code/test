"""Exact JSON-safe three-threshold delivery metrics for soft-loss v3."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import numpy as np
from scipy.spatial import cKDTree
import torch

from .contours import boundary_midpoints, symmetric_contour_metrics


_THRESHOLDS = (0.04, 0.05, 0.06)
_BOUNDARY_TOLERANCE_PX = 2.0


class DeliveryMetricsPreflightError(ValueError):
    """Raised before evaluation when a delivery-metric input is invalid."""


def _validate_target(target: torch.Tensor) -> None:
    if not isinstance(target, torch.Tensor):
        raise DeliveryMetricsPreflightError("target must be a torch.Tensor")
    if target.is_complex():
        raise DeliveryMetricsPreflightError("target must be real-valued")
    if target.ndim != 3 or target.shape[0] != 6 or any(size == 0 for size in target.shape):
        raise DeliveryMetricsPreflightError(
            "target must contain exactly six non-empty views with shape [6,H,W]"
        )
    if not torch.isfinite(target).all().item():
        raise DeliveryMetricsPreflightError("target must be finite")


def _validated_masks(
    target: torch.Tensor, masks_by_threshold: Mapping[float, torch.Tensor]
) -> dict[float, torch.Tensor]:
    if not isinstance(masks_by_threshold, Mapping):
        raise DeliveryMetricsPreflightError("masks_by_threshold must be a mapping")
    if any(isinstance(key, bool) or not isinstance(key, Real) for key in masks_by_threshold):
        raise DeliveryMetricsPreflightError("threshold keys must be numeric")
    normalized = {float(key): value for key, value in masks_by_threshold.items()}
    if len(normalized) != len(masks_by_threshold) or set(normalized) != set(_THRESHOLDS):
        raise DeliveryMetricsPreflightError(
            "threshold keys must be exactly {0.04, 0.05, 0.06}"
        )

    for threshold in _THRESHOLDS:
        mask = normalized[threshold]
        if not isinstance(mask, torch.Tensor):
            raise DeliveryMetricsPreflightError(
                f"mask at threshold {threshold:.2f} must be a torch.Tensor"
            )
        if mask.dtype != torch.bool:
            raise DeliveryMetricsPreflightError(
                f"mask at threshold {threshold:.2f} must have boolean dtype"
            )
        if mask.ndim != 3 or mask.shape[0] != 6 or any(size == 0 for size in mask.shape):
            raise DeliveryMetricsPreflightError(
                f"mask at threshold {threshold:.2f} must contain exactly six non-empty views "
                "with shape [6,H,W]"
            )
        if mask.shape != target.shape:
            raise DeliveryMetricsPreflightError("target and every mask must have the same shape")
        if mask.device != target.device:
            raise DeliveryMetricsPreflightError("target and every mask must share the same device")
    return normalized


def _validate_target_boundaries(target_binary: np.ndarray) -> None:
    for view_index, view in enumerate(target_binary):
        if boundary_midpoints(view).size == 0:
            raise DeliveryMetricsPreflightError(
                f"target boundary is empty in view {view_index}"
            )


def _boundary_f1(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted_points = boundary_midpoints(prediction)
    if predicted_points.size == 0:
        return 0.0
    target_points = boundary_midpoints(target)
    predicted_to_target = cKDTree(target_points).query(predicted_points, k=1)[0]
    target_to_predicted = cKDTree(predicted_points).query(target_points, k=1)[0]
    precision = float(np.mean(predicted_to_target <= _BOUNDARY_TOLERANCE_PX))
    recall = float(np.mean(target_to_predicted <= _BOUNDARY_TOLERANCE_PX))
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _one_view(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())

    iou_denominator = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    background_denominator = fp + tn
    foreground_denominator = tp + fn
    iou = 1.0 if iou_denominator == 0 else float(tp / iou_denominator)
    dice = 1.0 if dice_denominator == 0 else float((2 * tp) / dice_denominator)
    fpr = None if background_denominator == 0 else float(fp / background_denominator)
    recall = None if foreground_denominator == 0 else float(tp / foreground_denominator)

    contour = symmetric_contour_metrics(prediction, target)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": iou,
        "dice": dice,
        "fpr": fpr,
        "recall": recall,
        "boundary_f1_2px": _boundary_f1(prediction, target),
        "mscd_px": float(contour.mscd_px),
        "p95_symmetric_distance_px": float(contour.p95_symmetric_distance_px),
        "invalid_iou_denominator": False,
        "invalid_dice_denominator": False,
        "invalid_background_denominator": background_denominator == 0,
        "invalid_foreground_denominator": foreground_denominator == 0,
        "invalid_predicted_boundary": bool(contour.invalid_predicted_boundary),
        "invalid_target_boundary": bool(contour.invalid_target_boundary),
    }


def _mean(per_view: list[dict[str, object]], key: str) -> float:
    return float(sum(float(view[key]) for view in per_view) / len(per_view))


def _optional_mean(per_view: list[dict[str, object]], key: str) -> float | None:
    values = [view[key] for view in per_view]
    if any(value is None for value in values):
        return None
    return float(sum(float(value) for value in values) / len(values))


def _aggregate(per_view: list[dict[str, object]]) -> dict[str, object]:
    invalid_background = any(bool(view["invalid_background_denominator"]) for view in per_view)
    invalid_foreground = any(bool(view["invalid_foreground_denominator"]) for view in per_view)
    invalid_prediction = any(bool(view["invalid_predicted_boundary"]) for view in per_view)
    invalid_target = any(bool(view["invalid_target_boundary"]) for view in per_view)
    return {
        "tp": sum(int(view["tp"]) for view in per_view),
        "fp": sum(int(view["fp"]) for view in per_view),
        "fn": sum(int(view["fn"]) for view in per_view),
        "tn": sum(int(view["tn"]) for view in per_view),
        "mean_iou": _mean(per_view, "iou"),
        "mean_dice": _mean(per_view, "dice"),
        "mean_fpr": _optional_mean(per_view, "fpr"),
        "mean_recall": _optional_mean(per_view, "recall"),
        "mean_boundary_f1_2px": _mean(per_view, "boundary_f1_2px"),
        "mean_mscd_px": _mean(per_view, "mscd_px"),
        "mean_p95_symmetric_distance_px": _mean(
            per_view, "p95_symmetric_distance_px"
        ),
        "invalid_iou_denominator": any(
            bool(view["invalid_iou_denominator"]) for view in per_view
        ),
        "invalid_dice_denominator": any(
            bool(view["invalid_dice_denominator"]) for view in per_view
        ),
        "invalid_background_denominator": invalid_background,
        "invalid_foreground_denominator": invalid_foreground,
        "invalid_predicted_boundary": invalid_prediction,
        "invalid_target_boundary": invalid_target,
        "invalid_task_metric": (
            invalid_background or invalid_foreground or invalid_prediction or invalid_target
        ),
    }


def _threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}"


def evaluate_delivery_metrics(
    target: torch.Tensor,
    masks_by_threshold: Mapping[float, torch.Tensor],
) -> dict[str, object]:
    """Evaluate six already-binarized masks at each frozen delivery threshold."""

    _validate_target(target)
    masks = _validated_masks(target, masks_by_threshold)
    target_binary = (target >= 0.5).detach().to(device="cpu").numpy().astype(bool, copy=False)
    _validate_target_boundaries(target_binary)

    threshold_reports: dict[str, object] = {}
    for threshold in _THRESHOLDS:
        predictions = masks[threshold].detach().to(device="cpu").numpy()
        per_view = [
            _one_view(predictions[index], target_binary[index]) for index in range(6)
        ]
        threshold_reports[_threshold_key(threshold)] = {
            "per_view": per_view,
            "aggregate": _aggregate(per_view),
        }

    task_iou_by_threshold = {
        key: float(report["aggregate"]["mean_iou"])
        for key, report in threshold_reports.items()
    }
    iou_values = list(task_iou_by_threshold.values())
    primary = threshold_reports["0.05"]["aggregate"]
    invalid_task_metric = any(
        bool(report["aggregate"]["invalid_task_metric"])
        for report in threshold_reports.values()
    )
    result: dict[str, object] = {
        "thresholds": threshold_reports,
        "task_iou_by_threshold": task_iou_by_threshold,
        "worst_threshold_iou": float(min(iou_values)),
        "threshold_iou_range": float(max(iou_values) - min(iou_values)),
        "task_mscd_px": float(primary["mean_mscd_px"]),
        "task_p95_symmetric_distance_px": float(
            primary["mean_p95_symmetric_distance_px"]
        ),
        "invalid_task_metric": invalid_task_metric,
    }
    if not all(
        math.isfinite(value)
        for value in (
            result["worst_threshold_iou"],
            result["threshold_iou_range"],
            result["task_mscd_px"],
            result["task_p95_symmetric_distance_px"],
        )
    ):
        raise RuntimeError("delivery metric aggregation produced a non-finite value")
    return result
