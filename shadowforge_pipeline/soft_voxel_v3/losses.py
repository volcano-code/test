"""Balanced baseline, DIC and DTS losses for soft-loss v3."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .contours import ContourBands, validate_contour_bands
from .lovasz import lovasz_hinge_per_view


_BALANCE_EPS = 1e-8


@dataclass(frozen=True)
class V3Profile:
    """Pre-registered v3 objective profile."""

    name: str
    l1_weight: float
    l2_weight: float
    use_dic: bool
    use_dts: bool
    dts_thresholds: tuple[float, float] | None


_PROFILES = {
    "l12_bal": V3Profile("l12_bal", 0.35, 1.0, False, False, None),
    "dic_only": V3Profile("dic_only", 0.35, 1.0, True, False, None),
    "dts_only": V3Profile("dts_only", 0.35, 1.0, False, True, (0.04, 0.06)),
    "v3_full": V3Profile("v3_full", 0.35, 1.0, True, True, (0.04, 0.06)),
    "v3_zero_margin": V3Profile(
        "v3_zero_margin", 0.35, 1.0, True, True, (0.05, 0.05)
    ),
    "v3_no_l1": V3Profile("v3_no_l1", 0.0, 1.0, True, True, (0.04, 0.06)),
    "v3_no_l2": V3Profile("v3_no_l2", 0.35, 0.0, True, True, (0.04, 0.06)),
}


def _validate_projection_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if not isinstance(prediction, torch.Tensor):
        raise ValueError("prediction must be a torch.Tensor")
    if not isinstance(target, torch.Tensor):
        raise ValueError("target must be a torch.Tensor")
    if not prediction.is_floating_point():
        raise ValueError("prediction must use a floating-point dtype")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if prediction.ndim < 2 or any(size == 0 for size in prediction.shape):
        raise ValueError("prediction and target must have non-empty [views,...spatial] shape")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share a device")
    if not torch.isfinite(prediction).all().item() or not torch.isfinite(target).all().item():
        raise ValueError("prediction and target must be finite")


def _positive_temperature(temperature: float) -> float:
    try:
        value = float(temperature)
    except (TypeError, ValueError) as error:
        raise ValueError("temperature must be a finite positive scalar") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("temperature must be positive and finite")
    return value


def _finite_progress(progress: float) -> float:
    try:
        value = float(progress)
    except (TypeError, ValueError) as error:
        raise ValueError("progress must be a finite scalar") from error
    if not math.isfinite(value):
        raise ValueError("progress must be finite")
    return value


def _validate_dic_inputs(score_tau05: torch.Tensor, bands: ContourBands) -> None:
    if not isinstance(score_tau05, torch.Tensor):
        raise ValueError("score_tau05 must be a torch.Tensor")
    if not score_tau05.is_floating_point():
        raise ValueError("score_tau05 must use a floating-point dtype")
    if score_tau05.ndim != 3 or any(size == 0 for size in score_tau05.shape):
        raise ValueError("score_tau05 must have non-empty shape [views,H,W]")
    if not torch.isfinite(score_tau05).all().item():
        raise ValueError("score_tau05 must be finite")
    validate_contour_bands(bands)
    if score_tau05.shape != bands.inner.shape:
        raise ValueError("score_tau05 and contour bands must have the same shape")
    if score_tau05.device != bands.inner.device:
        raise ValueError("score_tau05 and contour bands must share a device")


def balanced_lp_loss(
    prediction: torch.Tensor, target: torch.Tensor, *, p: int
) -> torch.Tensor:
    """Return v2-parity foreground/background balanced L1 or L2 per view."""

    _validate_projection_pair(prediction, target)
    if p not in (1, 2):
        raise ValueError("p must be 1 or 2")
    dims = tuple(range(1, prediction.ndim))
    target_float = target.to(dtype=prediction.dtype)
    error = (prediction - target_float).abs()
    if p == 2:
        error = error.square()
    foreground = target_float
    background = 1.0 - target_float
    foreground_loss = (foreground * error).sum(dim=dims) / (
        foreground.sum(dim=dims) + _BALANCE_EPS
    )
    background_loss = (background * error).sum(dim=dims) / (
        background.sum(dim=dims) + _BALANCE_EPS
    )
    return 0.5 * foreground_loss + 0.5 * background_loss


def _dic_side_penalty(
    score: torch.Tensor, *, push_up: bool, temperature: float
) -> torch.Tensor:
    """Return one unweighted signed-margin softplus primitive."""

    signed_score = -score if push_up else score
    return temperature * F.softplus(signed_score / temperature)


def dic_margin_loss(
    score_tau05: torch.Tensor,
    bands: ContourBands,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Return equal-direction DIC soft margin loss per view."""

    _validate_dic_inputs(score_tau05, bands)
    value = _positive_temperature(temperature)
    inner = _dic_side_penalty(score_tau05, push_up=True, temperature=value)
    outer = _dic_side_penalty(score_tau05, push_up=False, temperature=value)
    dims = (-2, -1)
    inner_view = (inner * bands.inner).sum(dim=dims) / bands.inner.sum(dim=dims)
    outer_view = (outer * bands.outer).sum(dim=dims) / bands.outer.sum(dim=dims)
    return 0.5 * inner_view + 0.5 * outer_view


def dts_endpoint_loss(
    score: torch.Tensor,
    target_binary: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Return stable Lovasz loss for one selected DTS endpoint, per view."""

    if not isinstance(score, torch.Tensor):
        raise ValueError("score must be a torch.Tensor")
    if not score.is_floating_point():
        raise ValueError("score must use a floating-point dtype")
    value = _positive_temperature(temperature)
    return lovasz_hinge_per_view(score / value, target_binary)


def dts_schedule(progress: float) -> float:
    """Ramp DTS from zero to one over 50--65 percent progress."""

    value = (_finite_progress(progress) - 0.50) / 0.15
    return min(1.0, max(0.0, value))


def dic_schedule(progress: float) -> float:
    """Ramp DIC from zero to one over 65--80 percent progress."""

    value = (_finite_progress(progress) - 0.65) / 0.15
    return min(1.0, max(0.0, value))


def resolve_profile(name: str) -> V3Profile:
    """Resolve one exact pre-registered v3 profile."""

    try:
        return _PROFILES[name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown v3 profile: {name!r}") from error


def registered_profiles() -> tuple[str, str, str, str, str, str, str]:
    """Return profiles in frozen experiment-table order."""

    return (
        "l12_bal",
        "dic_only",
        "dts_only",
        "v3_full",
        "v3_zero_margin",
        "v3_no_l1",
        "v3_no_l2",
    )


def compute_baseline_branch(
    projections: torch.Tensor, targets: torch.Tensor, profile: V3Profile
) -> torch.Tensor:
    """Apply fixed profile L1/L2 weights, then average views equally."""

    if not isinstance(profile, V3Profile):
        raise ValueError("profile must be a V3Profile")
    per_view = profile.l2_weight * balanced_lp_loss(projections, targets, p=2)
    per_view = per_view + profile.l1_weight * balanced_lp_loss(
        projections, targets, p=1
    )
    return per_view.mean()


def compute_dic_branch(score_tau05: torch.Tensor, bands: ContourBands) -> torch.Tensor:
    """Average the per-view DIC values only at the branch boundary."""

    return dic_margin_loss(score_tau05, bands).mean()


def compute_dts_branch(score_tau: torch.Tensor, target_binary: torch.Tensor) -> torch.Tensor:
    """Average selected-endpoint per-view Lovasz values at the branch boundary."""

    return dts_endpoint_loss(score_tau, target_binary).mean()
