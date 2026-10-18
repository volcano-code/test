"""Core v4 loss and slot primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
import operator
from typing import Literal

import torch
import torch.nn.functional as F

from shadowforge_pipeline.soft_voxel_v3.contours import ContourBands, validate_contour_bands
from shadowforge_pipeline.soft_voxel_v3.losses import balanced_lp_loss as _v3_balanced_lp_loss


_PROFILE_ORDER = (
    "l12_bal",
    "dic_only",
    "dts_only",
    "v4_full",
    "v4_no_l1",
    "v4_no_l2",
)


@dataclass(frozen=True)
class V4Profile:
    name: str
    l1_weight: float
    l2_weight: float
    use_dic: bool
    use_dts: bool


@dataclass(frozen=True)
class DirectionalDICResult:
    per_view_loss: torch.Tensor
    false_negative_rate: torch.Tensor
    false_positive_rate: torch.Tensor
    weight_false_negative: torch.Tensor
    weight_false_positive: torch.Tensor


@dataclass(frozen=True)
class AuxiliarySlot:
    component: Literal["dic", "dts"]
    endpoint: float | None
    multiplier: float


_PROFILES = {
    "l12_bal": V4Profile("l12_bal", 0.35, 1.0, False, False),
    "dic_only": V4Profile("dic_only", 0.35, 1.0, True, False),
    "dts_only": V4Profile("dts_only", 0.35, 1.0, False, True),
    "v4_full": V4Profile("v4_full", 0.35, 1.0, True, True),
    "v4_no_l1": V4Profile("v4_no_l1", 0.0, 1.0, True, True),
    "v4_no_l2": V4Profile("v4_no_l2", 0.35, 0.0, True, True),
}


def _real_scalar(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number, not boolean")
    if not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _finite_positive_scalar(name: str, value: float) -> float:
    scalar = _real_scalar(name, value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return scalar


def _finite_scalar(name: str, value: float) -> float:
    return _real_scalar(name, value)


def _dtype_compatible_positive_scalar(name: str, value: float, reference: torch.Tensor) -> float:
    scalar = _finite_positive_scalar(name, value)
    cast_value = reference.new_tensor(scalar)
    cast_scalar = float(cast_value.item())
    if not math.isfinite(cast_scalar) or cast_scalar <= 0.0:
        raise ValueError(f"{name} must be representable as a positive finite {reference.dtype}")
    return cast_scalar


def _validate_score(name: str, score: torch.Tensor) -> None:
    if not isinstance(score, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not score.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if score.ndim != 3 or any(size == 0 for size in score.shape):
        raise ValueError(f"{name} must have non-empty shape [views,H,W]")
    if not torch.isfinite(score).all().item():
        raise ValueError(f"{name} must be finite")


def _validate_target_binary(target_binary: torch.Tensor) -> None:
    if not isinstance(target_binary, torch.Tensor):
        raise ValueError("target_binary must be a torch.Tensor")
    if target_binary.dtype != torch.bool:
        raise ValueError("target_binary must have boolean dtype")
    if target_binary.ndim != 3 or any(size == 0 for size in target_binary.shape):
        raise ValueError("target_binary must have non-empty shape [views,H,W]")


def _validate_profile(profile: V4Profile) -> None:
    if not isinstance(profile, V4Profile):
        raise ValueError("profile must be a V4Profile")
    registered = _PROFILES.get(profile.name)
    if registered is None or profile is not registered:
        raise ValueError("profile must be a registered V4Profile")


def _validate_global_step(global_step_before: int) -> int:
    if isinstance(global_step_before, bool):
        raise ValueError("global_step_before must be a nonnegative integer")
    try:
        step = operator.index(global_step_before)
    except TypeError as error:
        raise ValueError("global_step_before must be a nonnegative integer") from error
    if step < 0:
        raise ValueError("global_step_before must be a nonnegative integer")
    return step


def _validate_progress(progress: float) -> float:
    return _real_scalar("progress", progress)


def _validate_endpoint(endpoint: float) -> float:
    value = _finite_scalar("endpoint", endpoint)
    if value not in (0.04, 0.06):
        raise ValueError("endpoint must be exactly 0.04 or 0.06")
    return value


def balanced_lp_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    p: int,
) -> torch.Tensor:
    return _v3_balanced_lp_loss(prediction, target, p=p)


def resolve_profile(name: str) -> V4Profile:
    try:
        return _PROFILES[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown v4 profile: {name!r}") from error


def registered_profiles() -> tuple[str, str, str, str, str, str]:
    return _PROFILE_ORDER


def compute_baseline_branch(
    projections: torch.Tensor,
    targets: torch.Tensor,
    profile: V4Profile,
) -> torch.Tensor:
    _validate_profile(profile)
    per_view = profile.l2_weight * balanced_lp_loss(projections, targets, p=2)
    per_view = per_view + profile.l1_weight * balanced_lp_loss(projections, targets, p=1)
    return per_view.mean()


def _scaled_softplus_from_signed(
    signed_scores: torch.Tensor,
    branch_scores: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temp_tensor = branch_scores.new_tensor(temperature)
    scaled = temp_tensor * F.softplus(signed_scores)
    if torch.isinf(signed_scores).any().item():
        zeros = torch.zeros_like(branch_scores)
        scaled = torch.where(torch.isposinf(signed_scores), branch_scores, scaled)
        scaled = torch.where(torch.isneginf(signed_scores), zeros, scaled)
    return scaled


def directional_dic_terms(
    score_tau05: torch.Tensor,
    bands: ContourBands,
    temperature: float = 0.05,
    eps: float = 1e-8,
) -> DirectionalDICResult:
    _validate_score("score_tau05", score_tau05)
    validate_contour_bands(bands)
    if score_tau05.shape != bands.inner.shape:
        raise ValueError("score_tau05 and contour bands must have the same shape")
    if score_tau05.device != bands.inner.device:
        raise ValueError("score_tau05 and contour bands must share a device")
    temp = _dtype_compatible_positive_scalar("temperature", temperature, score_tau05)
    eps_value = _dtype_compatible_positive_scalar("eps", eps, score_tau05)

    dtype = score_tau05.dtype
    inner = bands.inner
    outer = bands.outer
    inner_float = inner.to(dtype=dtype)
    outer_float = outer.to(dtype=dtype)
    dims = (-2, -1)

    detached_score = score_tau05.detach()
    false_negative_rate = (inner & (detached_score < 0.0)).to(dtype=dtype).sum(dim=dims)
    false_negative_rate = false_negative_rate / inner_float.sum(dim=dims)
    false_positive_rate = (outer & (detached_score >= 0.0)).to(dtype=dtype).sum(dim=dims)
    false_positive_rate = false_positive_rate / outer_float.sum(dim=dims)

    eps_tensor = score_tau05.new_tensor(eps_value)
    weight_false_positive = (false_positive_rate + eps_tensor) / (
        false_positive_rate + false_negative_rate + (2.0 * eps_tensor)
    )
    weight_false_positive = weight_false_positive.clamp(0.25, 0.75).detach()
    weight_false_negative = (1.0 - weight_false_positive).detach()
    false_negative_rate = false_negative_rate.detach()
    false_positive_rate = false_positive_rate.detach()

    inner_values = _scaled_softplus_from_signed((-score_tau05) / temp, -score_tau05, temp)
    mean_inner = torch.where(inner, inner_values, torch.zeros_like(score_tau05)).sum(dim=dims)
    mean_inner = mean_inner / inner_float.sum(dim=dims)
    outer_values = _scaled_softplus_from_signed(score_tau05 / temp, score_tau05, temp)
    mean_outer = torch.where(outer, outer_values, torch.zeros_like(score_tau05)).sum(dim=dims)
    mean_outer = mean_outer / outer_float.sum(dim=dims)
    per_view_loss = weight_false_negative * mean_inner + weight_false_positive * mean_outer

    return DirectionalDICResult(
        per_view_loss=per_view_loss,
        false_negative_rate=false_negative_rate,
        false_positive_rate=false_positive_rate,
        weight_false_negative=weight_false_negative,
        weight_false_positive=weight_false_positive,
    )


def directional_dic_loss(
    score_tau05: torch.Tensor,
    bands: ContourBands,
    temperature: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    return directional_dic_terms(score_tau05, bands, temperature=temperature, eps=eps).per_view_loss


def asymmetric_dts_loss(
    score: torch.Tensor,
    target_binary: torch.Tensor,
    endpoint: float,
    temperature: float = 0.05,
) -> torch.Tensor:
    _validate_score("score", score)
    _validate_target_binary(target_binary)
    if score.shape != target_binary.shape:
        raise ValueError("score and target_binary must have the same shape")
    if score.device != target_binary.device:
        raise ValueError("score and target_binary must share a device")
    value = _validate_endpoint(endpoint)
    temp = _dtype_compatible_positive_scalar("temperature", temperature, score)

    selected = ~target_binary if value == 0.04 else target_binary
    if not selected.any(dim=(-2, -1)).all().item():
        raise ValueError("every view must have non-empty selected support")

    per_view = []
    for view_score, view_selected in zip(score, selected):
        selected_scores = view_score[view_selected]
        branch_scores = selected_scores if value == 0.04 else -selected_scores
        per_view.append(_scaled_softplus_from_signed(branch_scores / temp, branch_scores, temp).mean())
    return torch.stack(per_view)


def _slot_multiplier(progress: float) -> float:
    clamped = min(max(progress, 0.0), 0.75)
    if clamped <= 0.60:
        return 0.0
    if clamped < 0.75:
        return (clamped - 0.60) / 0.15
    return 1.0


def auxiliary_slot(
    global_step_before: int,
    progress: float,
    profile: V4Profile,
) -> AuxiliarySlot | None:
    step = _validate_global_step(global_step_before)
    progress_value = _validate_progress(progress)
    _validate_profile(profile)
    multiplier = _slot_multiplier(progress_value)
    if multiplier == 0.0:
        return None

    phase = step % 4
    if phase in (0, 2):
        if not profile.use_dic:
            return None
        return AuxiliarySlot(component="dic", endpoint=None, multiplier=multiplier)
    if not profile.use_dts:
        return None
    endpoint = 0.04 if phase == 1 else 0.06
    return AuxiliarySlot(component="dts", endpoint=endpoint, multiplier=multiplier)
