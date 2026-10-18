"""JSON-safe per-view evaluation for P-CRAFT soft-only experiments."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from scipy import ndimage


_PER_VIEW_FLOAT_METRICS = (
    "continuous_iou",
    "binary_iou",
    "dice",
    "ssim",
    "fpr",
    "recall",
    "boundary_f1_2px",
    "soft_binary_iou_gap",
)


def _as_view_images(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"{name} must have shape [views, height, width] or [views, 1, H, W]")
    return tensor


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def _boundary_f1(prediction: np.ndarray, target: np.ndarray, tolerance_px: float) -> float:
    pred_boundary = _mask_boundary(prediction)
    target_boundary = _mask_boundary(target)
    pred_count = int(pred_boundary.sum())
    target_count = int(target_boundary.sum())
    if pred_count == 0 and target_count == 0:
        return 1.0
    if pred_count == 0 or target_count == 0:
        return 0.0

    distance_to_target = ndimage.distance_transform_edt(~target_boundary)
    distance_to_prediction = ndimage.distance_transform_edt(~pred_boundary)
    precision = float((distance_to_target[pred_boundary] <= tolerance_px).mean())
    recall = float((distance_to_prediction[target_boundary] <= tolerance_px).mean())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _continuous_iou(prediction: np.ndarray, target: np.ndarray, eps: float) -> float:
    soft = np.clip(prediction.astype(np.float64, copy=False), 0.0, 1.0)
    truth = (target >= 0.5).astype(np.float64)
    intersection = np.minimum(soft, truth).sum()
    union = np.maximum(soft, truth).sum()
    return float(intersection / (union + eps))


def _ssim(prediction: np.ndarray, target: np.ndarray, window_size: int = 11) -> float:
    """Single-scale SSIM metric matching the trainer constants."""

    prediction = np.clip(prediction.astype(np.float64, copy=False), 0.0, 1.0)
    target = np.clip(target.astype(np.float64, copy=False), 0.0, 1.0)
    mean_prediction = ndimage.uniform_filter(prediction, window_size, mode="constant")
    mean_target = ndimage.uniform_filter(target, window_size, mode="constant")
    variance_prediction = (
        ndimage.uniform_filter(prediction * prediction, window_size, mode="constant")
        - mean_prediction**2
    )
    variance_target = (
        ndimage.uniform_filter(target * target, window_size, mode="constant")
        - mean_target**2
    )
    covariance = (
        ndimage.uniform_filter(prediction * target, window_size, mode="constant")
        - mean_prediction * mean_target
    )
    numerator = (2 * mean_prediction * mean_target + 0.01**2) * (
        2 * covariance + 0.03**2
    )
    denominator = (
        (mean_prediction**2 + mean_target**2 + 0.01**2)
        * (variance_prediction + variance_target + 0.03**2)
        + 1e-8
    )
    return float(np.clip(numerator / denominator, -1.0, 1.0).mean())


def _evaluate_one_view(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    boundary_tolerance_px: float,
    eps: float,
) -> Dict[str, float | int]:
    soft = np.clip(prediction.astype(np.float64, copy=False), 0.0, 1.0)
    truth = target >= 0.5
    binary = soft >= threshold

    tp = int(np.logical_and(binary, truth).sum())
    fp = int(np.logical_and(binary, ~truth).sum())
    fn = int(np.logical_and(~binary, truth).sum())
    tn = int(np.logical_and(~binary, ~truth).sum())

    intersection_soft = np.minimum(soft, truth.astype(np.float64)).sum()
    union_soft = np.maximum(soft, truth.astype(np.float64)).sum()
    continuous_iou = float(intersection_soft / (union_soft + eps))
    binary_iou = float(tp / (tp + fp + fn + eps))
    dice = float((2 * tp) / (2 * tp + fp + fn + eps))
    fpr = float(fp / (fp + tn + eps))
    recall = float(tp / (tp + fn + eps))
    boundary_f1 = _boundary_f1(binary, truth, boundary_tolerance_px)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "continuous_iou": continuous_iou,
        "binary_iou": binary_iou,
        "dice": dice,
        "ssim": _ssim(soft, truth.astype(np.float64)),
        "fpr": fpr,
        "recall": recall,
        "boundary_f1_2px": boundary_f1,
        "soft_binary_iou_gap": continuous_iou - binary_iou,
    }


def _aggregate(per_view: Sequence[Mapping[str, float | int]]) -> Dict[str, float | int]:
    aggregate: Dict[str, float | int] = {
        "tp": sum(int(view["tp"]) for view in per_view),
        "fp": sum(int(view["fp"]) for view in per_view),
        "fn": sum(int(view["fn"]) for view in per_view),
        "tn": sum(int(view["tn"]) for view in per_view),
    }
    for metric in _PER_VIEW_FLOAT_METRICS:
        aggregate[f"mean_{metric}"] = float(
            np.mean([float(view[metric]) for view in per_view])
        )
    aggregate["worst_view_binary_iou"] = float(
        min(float(view["binary_iou"]) for view in per_view)
    )
    return aggregate


def _threshold_key(value: float) -> str:
    return f"{float(value):.2f}"


def evaluate_soft_views(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    thresholds: Sequence[float] = (0.04, 0.05, 0.06),
    primary_threshold: float = 0.05,
    boundary_tolerance_px: float = 2.0,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Evaluate continuous and thresholded silhouettes at all registered cutoffs.

    The returned nested dictionary contains only Python scalars/lists/dicts and
    can therefore be written directly with :mod:`json`.  FPR is normalized by
    target-background pixels, i.e. ``FP / (FP + TN)``.
    """

    prediction = _as_view_images(prediction, "prediction")
    target = _as_view_images(target, "target")
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    if not thresholds:
        raise ValueError("at least one threshold is required")
    if boundary_tolerance_px < 0:
        raise ValueError("boundary_tolerance_px must be non-negative")

    normalized_thresholds = tuple(float(value) for value in thresholds)
    if not any(abs(value - float(primary_threshold)) < 1e-12 for value in normalized_thresholds):
        raise ValueError("primary_threshold must be present in thresholds")

    prediction_np = prediction.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()
    threshold_reports: Dict[str, Dict[str, object]] = {}
    for threshold in normalized_thresholds:
        per_view = [
            _evaluate_one_view(
                prediction_np[index],
                target_np[index],
                threshold=threshold,
                boundary_tolerance_px=float(boundary_tolerance_px),
                eps=float(eps),
            )
            for index in range(prediction_np.shape[0])
        ]
        threshold_reports[_threshold_key(threshold)] = {
            "threshold": threshold,
            "per_view": per_view,
            "aggregate": _aggregate(per_view),
        }

    primary_key = _threshold_key(float(primary_threshold))
    primary = threshold_reports[primary_key]
    return {
        "primary_threshold": float(primary_threshold),
        "per_view": primary["per_view"],
        "aggregate": primary["aggregate"],
        "thresholds": threshold_reports,
    }


def evaluate_rendered_views(
    raw_prediction: torch.Tensor,
    target: torch.Tensor,
    binary_masks_by_density_threshold: Mapping[float, torch.Tensor],
    *,
    proxy_prediction: torch.Tensor | None = None,
    primary_threshold: float = 0.05,
    boundary_tolerance_px: float = 2.0,
    eps: float = 1e-8,
) -> Dict[str, object]:
    """Evaluate exact binary renders produced from density thresholds.

    The mapping keys are *3-D density thresholds*.  Its boolean values must be
    rendered silhouettes that already passed the fixed per-view 0.8-of-max
    mask rule; this function never thresholds projection intensity.
    """

    if not np.isclose(float(boundary_tolerance_px), 2.0, rtol=0.0, atol=1e-12):
        raise ValueError("formal soft-loss-v2 metrics require a fixed 2 px boundary tolerance")

    raw_prediction = _as_view_images(raw_prediction, "raw_prediction")
    target = _as_view_images(target, "target")
    if raw_prediction.shape != target.shape:
        raise ValueError("raw_prediction and target shapes differ")
    if not binary_masks_by_density_threshold:
        raise ValueError("at least one density threshold render is required")
    if proxy_prediction is not None:
        proxy_prediction = _as_view_images(proxy_prediction, "proxy_prediction")
        if proxy_prediction.shape != target.shape:
            raise ValueError("proxy_prediction and target shapes differ")

    raw_np = raw_prediction.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()
    proxy_np = (
        proxy_prediction.detach().float().cpu().numpy()
        if proxy_prediction is not None
        else None
    )
    normalized_masks: Dict[float, np.ndarray] = {}
    for threshold, masks in binary_masks_by_density_threshold.items():
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("density thresholds must be finite and lie in [0, 1]")
        masks = _as_view_images(masks, f"binary_masks[{threshold}]")
        if masks.shape != target.shape:
            raise ValueError(f"binary masks at density threshold {threshold} have wrong shape")
        normalized_masks[threshold] = masks.detach().bool().cpu().numpy()
    serialized_keys = [_threshold_key(value) for value in normalized_masks]
    if len(serialized_keys) != len(set(serialized_keys)):
        raise ValueError("density thresholds must remain unique in JSON metric keys")
    if not any(abs(value - float(primary_threshold)) < 1e-12 for value in normalized_masks):
        raise ValueError("primary_threshold must be present in density threshold renders")

    reports: Dict[str, Dict[str, object]] = {}
    for threshold in sorted(normalized_masks):
        masks_np = normalized_masks[threshold]
        per_view: List[Dict[str, float | int]] = []
        for index in range(raw_np.shape[0]):
            truth = target_np[index] >= 0.5
            binary = masks_np[index]
            tp = int(np.logical_and(binary, truth).sum())
            fp = int(np.logical_and(binary, ~truth).sum())
            fn = int(np.logical_and(~binary, truth).sum())
            tn = int(np.logical_and(~binary, ~truth).sum())
            raw_iou = _continuous_iou(raw_np[index], target_np[index], eps)
            proxy_iou = (
                _continuous_iou(proxy_np[index], target_np[index], eps)
                if proxy_np is not None
                else None
            )
            binary_iou = float(tp / (tp + fp + fn + eps))
            per_view.append(
                {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "continuous_iou": raw_iou,
                    "raw_continuous_iou": raw_iou,
                    "proxy_continuous_iou": proxy_iou,
                    "binary_iou": binary_iou,
                    "dice": float(2 * tp / (2 * tp + fp + fn + eps)),
                    "ssim": _ssim(raw_np[index], target_np[index]),
                    "fpr": float(fp / (fp + tn + eps)),
                    "recall": float(tp / (tp + fn + eps)),
                    "boundary_f1_2px": _boundary_f1(
                        binary, truth, float(boundary_tolerance_px)
                    ),
                    "soft_binary_iou_gap": raw_iou - binary_iou,
                }
            )
        aggregate = _aggregate(per_view)
        aggregate["mean_raw_continuous_iou"] = float(
            np.mean([float(view["raw_continuous_iou"]) for view in per_view])
        )
        aggregate["mean_proxy_continuous_iou"] = (
            float(np.mean([float(view["proxy_continuous_iou"]) for view in per_view]))
            if proxy_np is not None
            else None
        )
        reports[_threshold_key(threshold)] = {
            "density_threshold": threshold,
            "per_view": per_view,
            "aggregate": aggregate,
        }

    primary = reports[_threshold_key(float(primary_threshold))]
    return {
        "density_threshold_semantics": True,
        "primary_threshold": float(primary_threshold),
        "per_view": primary["per_view"],
        "aggregate": primary["aggregate"],
        "thresholds": reports,
    }
