"""Scalar-only component-gradient diagnostics for soft-loss v3."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch


@dataclass(frozen=True)
class ComponentGradientSample:
    """One completed-update-aligned, JSON-safe component-gradient sample."""

    completed_updates: int
    base_norm: float
    dic_norm: float | None
    dts_norms: dict[float, float]
    base_dic_cosine: float | None
    base_dic_cosine_invalid: bool
    base_dts_cosines: dict[float, float | None]
    base_dts_cosine_invalid: dict[float, bool]
    dic_dts_cosines: dict[float, float | None]
    dic_dts_cosine_invalid: dict[float, bool]
    dts_pair_cosines: dict[float, dict[float, float | None]]
    dts_pair_cosine_invalid: dict[float, dict[float, bool]]
    dic_dts_support_ious: dict[float, float | None]
    dic_dts_support_iou_invalid: dict[float, bool]
    weighted_dic_base_ratio: float | None
    weighted_dic_base_ratio_invalid: bool
    weighted_dts_base_ratios: dict[float, float | None]
    weighted_dts_base_ratio_invalid: dict[float, bool]
    selected_dts_tau: float | None
    combined_aux_base_ratio: float | None
    combined_aux_base_ratio_invalid: bool
    base_combined_aux_cosine: float | None
    base_combined_aux_cosine_invalid: bool


def _finite_real(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric scalar")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric scalar") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if positive and normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not positive and normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _normalize_threshold(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric and finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _validate_logits(logits: object) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise ValueError("logits must be a torch.Tensor")
    if not logits.is_floating_point():
        raise ValueError("logits must use a floating-point dtype")
    if logits.numel() == 0:
        raise ValueError("logits must be non-empty")
    if not logits.requires_grad:
        raise ValueError("logits must be gradient-enabled")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must be finite")
    return logits


def _normalize_builders(
    dts_builders: object,
) -> dict[float, Callable[[], torch.Tensor]]:
    if not isinstance(dts_builders, Mapping):
        raise ValueError("dts_builders must be a mapping")
    normalized: dict[float, Callable[[], torch.Tensor]] = {}
    for raw_tau, builder in dts_builders.items():
        tau = _normalize_threshold(raw_tau, name="DTS threshold key")
        if tau in normalized:
            raise ValueError("dts_builders contains a duplicate normalized threshold")
        if not callable(builder):
            raise ValueError(f"DTS builder for threshold {tau} must be callable")
        normalized[tau] = builder
    return {tau: normalized[tau] for tau in sorted(normalized)}


def _measure_gradient(
    logits: torch.Tensor,
    builder: Callable[[], torch.Tensor],
    *,
    name: str,
) -> torch.Tensor:
    loss = builder()
    if not isinstance(loss, torch.Tensor):
        raise ValueError(f"{name} builder must return a torch.Tensor")
    if loss.ndim != 0:
        raise ValueError(f"{name} builder must return a scalar tensor")
    if not loss.is_floating_point():
        raise ValueError(f"{name} builder must return a floating-point tensor")
    if not torch.isfinite(loss).item():
        raise ValueError(f"{name} builder must return a finite tensor")
    if not loss.requires_grad:
        raise ValueError(f"{name} builder output must be gradient-enabled")
    try:
        (gradient,) = torch.autograd.grad(loss, logits, retain_graph=False)
    except RuntimeError as error:
        raise ValueError(
            f"{name} builder output is disconnected from logits or has an invalid gradient"
        ) from error
    finally:
        del loss
    if not isinstance(gradient, torch.Tensor) or gradient.shape != logits.shape:
        raise ValueError(f"{name} gradient has the wrong shape")
    detached = gradient.detach().reshape(-1).to(device="cpu")
    del gradient
    if not detached.is_floating_point() or not torch.isfinite(detached).all().item():
        raise ValueError(f"{name} gradient must be finite and floating-point")
    return detached


def _norm(vector: torch.Tensor, *, name: str) -> float:
    value = float(torch.linalg.vector_norm(vector.to(dtype=torch.float64)).item())
    if not math.isfinite(value):
        raise ValueError(f"{name} gradient norm must be finite")
    return value


def _cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    left_norm: float,
    right_norm: float,
) -> tuple[float | None, bool]:
    if (
        left_norm == 0.0
        or right_norm == 0.0
        or not math.isfinite(left_norm)
        or not math.isfinite(right_norm)
    ):
        return None, True
    left_unit = left.to(dtype=torch.float64) / left_norm
    right_unit = right.to(dtype=torch.float64) / right_norm
    value = float(torch.dot(left_unit, right_unit).item())
    if not math.isfinite(value):
        return None, True
    return min(1.0, max(-1.0, value)), False


def _ratio(numerator: float, denominator: float) -> tuple[float | None, bool]:
    if denominator == 0.0 or not math.isfinite(denominator):
        return None, True
    value = numerator / denominator
    if not math.isfinite(value):
        return None, True
    return float(value), False


def _support_iou(
    left: torch.Tensor, right: torch.Tensor, epsilon: float
) -> tuple[float | None, bool]:
    left_active = left.abs() > epsilon
    right_active = right.abs() > epsilon
    union = torch.logical_or(left_active, right_active)
    union_count = int(union.sum().item())
    if union_count == 0:
        return None, True
    intersection_count = int(torch.logical_and(left_active, right_active).sum().item())
    return float(intersection_count / union_count), False


def measure_component_gradients(
    logits: torch.Tensor,
    *,
    completed_updates: int,
    base_builder: Callable[[], torch.Tensor],
    dic_builder: Callable[[], torch.Tensor] | None,
    dts_builders: Mapping[float, Callable[[], torch.Tensor]],
    dic_scale: float,
    dts_scale: float,
    selected_dts_tau: float | None,
    support_epsilon: float = 1e-12,
) -> ComponentGradientSample:
    """Measure independent component gradients without touching ``logits.grad``."""

    logits = _validate_logits(logits)
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, Integral)
        or completed_updates < 0
    ):
        raise ValueError("completed_updates must be a non-negative integer")
    if not callable(base_builder):
        raise ValueError("base_builder must be callable")
    if dic_builder is not None and not callable(dic_builder):
        raise ValueError("dic_builder must be callable or None")
    normalized_dts = _normalize_builders(dts_builders)
    dic_scale_value = _finite_real(dic_scale, name="dic_scale")
    dts_scale_value = _finite_real(dts_scale, name="dts_scale")
    epsilon = _finite_real(support_epsilon, name="support_epsilon", positive=True)
    if dic_builder is None and dic_scale_value > 0.0:
        raise ValueError("dic_scale must be zero when dic_builder is None")

    selected_tau = None
    if selected_dts_tau is not None:
        selected_tau = _normalize_threshold(
            selected_dts_tau, name="selected_dts_tau"
        )
        if not normalized_dts:
            raise ValueError("selected_dts_tau is invalid with no DTS builders")
        if selected_tau not in normalized_dts:
            raise ValueError("selected_dts_tau is missing from dts_builders")
    elif dts_scale_value > 0.0:
        raise ValueError("an active dts_scale requires a selected DTS endpoint")

    base_gradient = _measure_gradient(logits, base_builder, name="base")
    base_norm = _norm(base_gradient, name="base")
    dic_gradient = None
    dic_norm = None
    if dic_builder is not None:
        dic_gradient = _measure_gradient(logits, dic_builder, name="DIC")
        dic_norm = _norm(dic_gradient, name="DIC")
    dts_gradients = {
        tau: _measure_gradient(logits, builder, name=f"DTS[{tau}]")
        for tau, builder in normalized_dts.items()
    }
    dts_norms = {
        tau: _norm(gradient, name=f"DTS[{tau}]")
        for tau, gradient in dts_gradients.items()
    }

    if dic_gradient is None:
        base_dic_cosine, base_dic_cosine_invalid = None, True
        weighted_dic_base_ratio, weighted_dic_base_ratio_invalid = None, True
    else:
        base_dic_cosine, base_dic_cosine_invalid = _cosine(
            base_gradient, dic_gradient, base_norm, dic_norm
        )
        weighted_dic_base_ratio, weighted_dic_base_ratio_invalid = _ratio(
            dic_scale_value * dic_norm, base_norm
        )

    base_dts_cosines: dict[float, float | None] = {}
    base_dts_cosine_invalid: dict[float, bool] = {}
    weighted_dts_base_ratios: dict[float, float | None] = {}
    weighted_dts_base_ratio_invalid: dict[float, bool] = {}
    for tau, gradient in dts_gradients.items():
        cosine, invalid = _cosine(base_gradient, gradient, base_norm, dts_norms[tau])
        base_dts_cosines[tau] = cosine
        base_dts_cosine_invalid[tau] = invalid
        ratio, invalid = _ratio(dts_scale_value * dts_norms[tau], base_norm)
        weighted_dts_base_ratios[tau] = ratio
        weighted_dts_base_ratio_invalid[tau] = invalid

    dic_dts_cosines: dict[float, float | None] = {}
    dic_dts_cosine_invalid: dict[float, bool] = {}
    dic_dts_support_ious: dict[float, float | None] = {}
    dic_dts_support_iou_invalid: dict[float, bool] = {}
    if dic_gradient is not None:
        for tau, gradient in dts_gradients.items():
            cosine, invalid = _cosine(dic_gradient, gradient, dic_norm, dts_norms[tau])
            dic_dts_cosines[tau] = cosine
            dic_dts_cosine_invalid[tau] = invalid
            overlap, invalid = _support_iou(dic_gradient, gradient, epsilon)
            dic_dts_support_ious[tau] = overlap
            dic_dts_support_iou_invalid[tau] = invalid

    dts_pair_cosines: dict[float, dict[float, float | None]] = {}
    dts_pair_cosine_invalid: dict[float, dict[float, bool]] = {}
    thresholds = list(dts_gradients)
    for index, left_tau in enumerate(thresholds):
        pair_values: dict[float, float | None] = {}
        pair_invalid: dict[float, bool] = {}
        for right_tau in thresholds[index + 1 :]:
            cosine, invalid = _cosine(
                dts_gradients[left_tau],
                dts_gradients[right_tau],
                dts_norms[left_tau],
                dts_norms[right_tau],
            )
            pair_values[right_tau] = cosine
            pair_invalid[right_tau] = invalid
        if pair_values:
            dts_pair_cosines[left_tau] = pair_values
            dts_pair_cosine_invalid[left_tau] = pair_invalid

    combined_gradient = torch.zeros_like(base_gradient, dtype=torch.float64)
    if dic_gradient is not None:
        combined_gradient = combined_gradient + dic_scale_value * dic_gradient.to(
            dtype=torch.float64
        )
    if selected_tau is not None:
        combined_gradient = combined_gradient + dts_scale_value * dts_gradients[
            selected_tau
        ].to(dtype=torch.float64)
    combined_norm = _norm(combined_gradient, name="combined auxiliary")
    combined_aux_base_ratio, combined_aux_base_ratio_invalid = _ratio(
        combined_norm, base_norm
    )
    base_combined_aux_cosine, base_combined_aux_cosine_invalid = _cosine(
        base_gradient, combined_gradient, base_norm, combined_norm
    )

    return ComponentGradientSample(
        completed_updates=int(completed_updates),
        base_norm=base_norm,
        dic_norm=dic_norm,
        dts_norms=dts_norms,
        base_dic_cosine=base_dic_cosine,
        base_dic_cosine_invalid=base_dic_cosine_invalid,
        base_dts_cosines=base_dts_cosines,
        base_dts_cosine_invalid=base_dts_cosine_invalid,
        dic_dts_cosines=dic_dts_cosines,
        dic_dts_cosine_invalid=dic_dts_cosine_invalid,
        dts_pair_cosines=dts_pair_cosines,
        dts_pair_cosine_invalid=dts_pair_cosine_invalid,
        dic_dts_support_ious=dic_dts_support_ious,
        dic_dts_support_iou_invalid=dic_dts_support_iou_invalid,
        weighted_dic_base_ratio=weighted_dic_base_ratio,
        weighted_dic_base_ratio_invalid=weighted_dic_base_ratio_invalid,
        weighted_dts_base_ratios=weighted_dts_base_ratios,
        weighted_dts_base_ratio_invalid=weighted_dts_base_ratio_invalid,
        selected_dts_tau=selected_tau,
        combined_aux_base_ratio=combined_aux_base_ratio,
        combined_aux_base_ratio_invalid=combined_aux_base_ratio_invalid,
        base_combined_aux_cosine=base_combined_aux_cosine,
        base_combined_aux_cosine_invalid=base_combined_aux_cosine_invalid,
    )
