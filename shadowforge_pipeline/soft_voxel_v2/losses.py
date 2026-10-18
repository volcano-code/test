"""Differentiable loss primitives for the P-CRAFT soft-only trainer.

The functions in this module deliberately do not import any repair, carving,
meshing, PINO, or FEM code.  They operate only on density/projection tensors.
Unless stated otherwise, projection losses return one scalar per view; the
caller is responsible for applying the requested multi-view aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage


DEFAULT_MIRROR_PAIRS: Tuple[Tuple[int, int], ...] = ((0, 3), (1, 4), (2, 5))


@dataclass(frozen=True)
class LossProfile:
    """Resolved, serializable switches for one registered loss profile."""

    name: str
    l2_weight: float = 1.0
    l1_weight: float = 0.35
    balanced: bool = False
    use_proxy: bool = False
    use_gate: bool = False
    use_boundary: bool = False
    view_aggregation: str = "mean"
    legacy_terms: bool = False
    use_ssim: bool = False


@dataclass
class SoftLossResult:
    """Structured loss output retained by the trainer for step-level logging."""

    total: torch.Tensor
    per_view_total: torch.Tensor
    per_view_l2: torch.Tensor
    per_view_l1: torch.Tensor
    per_view_gate: torch.Tensor
    per_view_boundary: torch.Tensor
    proxy_weight: float
    gate_schedule: float
    boundary_schedule: float
    robust_rho: float

    def detached_scalars(self) -> dict[str, float]:
        """Return JSON-safe summary scalars without retaining the graph."""

        return {
            "loss": float(self.total.detach().cpu()),
            "loss_l2": float(self.per_view_l2.detach().mean().cpu()),
            "loss_l1": float(self.per_view_l1.detach().mean().cpu()),
            "loss_gate": float(self.per_view_gate.detach().mean().cpu()),
            "loss_boundary": float(self.per_view_boundary.detach().mean().cpu()),
            "proxy_weight": self.proxy_weight,
            "gate_schedule": self.gate_schedule,
            "boundary_schedule": self.boundary_schedule,
            "robust_rho": self.robust_rho,
        }


_PROFILES = {
    "legacy_full": LossProfile(name="legacy_full", legacy_terms=True),
    "l12_plain": LossProfile(name="l12_plain"),
    "l12_bal": LossProfile(name="l12_bal", balanced=True),
    "new_full": LossProfile(
        name="new_full",
        balanced=True,
        use_proxy=True,
        use_gate=True,
        use_boundary=True,
        view_aggregation="grouped_robust",
    ),
}


def resolve_profile(
    name: str,
    *,
    disable_proxy: bool = False,
    disable_gate: bool = False,
    disable_boundary: bool = False,
    view_aggregation: str | None = None,
) -> LossProfile:
    """Resolve a registered profile and apply independent component ablations.

    ``disable_proxy`` controls the normalized proxy auxiliary in L1/L2; the
    raw-domain base loss is always retained. The trainer separately selects
    the auxiliary domain: the threshold-proxy ablation passes raw P to
    Gate/Boundary and never constructs Q.
    """

    try:
        profile = _PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILES))
        raise ValueError(f"unknown loss profile {name!r}; choose one of: {choices}") from exc

    aggregation = view_aggregation or profile.view_aggregation
    if aggregation not in {"mean", "grouped_robust"}:
        raise ValueError("view_aggregation must be 'mean' or 'grouped_robust'")

    return replace(
        profile,
        use_proxy=profile.use_proxy and not disable_proxy,
        use_gate=profile.use_gate and not disable_gate,
        use_boundary=profile.use_boundary and not disable_boundary,
        view_aggregation=aggregation,
    )


def registered_profiles() -> Tuple[str, ...]:
    """Return stable CLI choices for the registered profiles."""

    return tuple(_PROFILES)


def _linear_ramp(progress: float, start: float, end: float) -> float:
    value = float(progress)
    if value <= start:
        return 0.0
    if value >= end:
        return 1.0
    return (value - start) / (end - start)


def proxy_weight_schedule(progress: float) -> float:
    """Proxy blend: off through 25%, linearly full at 65%."""

    return _linear_ramp(progress, 0.25, 0.65)


def gate_weight_schedule(progress: float) -> float:
    """Warm up the proxy-dependent gate on the same 25--65% interval."""

    return proxy_weight_schedule(progress)


def boundary_weight_schedule(progress: float) -> float:
    """Warm up the proxy-dependent boundary term on the 25--65% interval."""

    return proxy_weight_schedule(progress)


def robust_rho_schedule(progress: float) -> float:
    """Robust view weight: 0 until 70%, then linearly reaches .25."""

    return 0.25 * _linear_ramp(progress, 0.70, 1.0)


def threshold_margin_schedule(
    progress: float,
    *,
    initial_margin: float = 0.02,
    final_margin: float = 0.005,
) -> float:
    """Linearly anneal the compact-threshold half-width over training."""

    clipped = min(max(float(progress), 0.0), 1.0)
    return initial_margin + clipped * (final_margin - initial_margin)


def compact_soft_threshold(
    density: torch.Tensor,
    *,
    tau: float = 0.05,
    margin: float = 0.02,
) -> torch.Tensor:
    """Compact 3-D threshold proxy ``clip(.5 + (S-tau)/(2m), 0, 1)``.

    ``torch.where`` gives zero gradients at both plateaus, including their
    exact endpoints, and the prescribed constant gradient inside the ramp.
    """

    if margin <= 0:
        raise ValueError("margin must be positive")
    linear = 0.5 + (density - float(tau)) / (2.0 * float(margin))
    # Decimal thresholds such as .05 +/- .01 are not exactly representable.
    # Snap values within a few machine epsilons to the mathematical plateau so
    # endpoint gradients do not depend on float32 rounding direction.
    edge_tolerance = 8.0 * torch.finfo(linear.dtype).eps
    return torch.where(
        linear <= edge_tolerance,
        torch.zeros_like(linear),
        torch.where(linear >= 1.0 - edge_tolerance, torch.ones_like(linear), linear),
    )


def _view_reduction_dims(tensor: torch.Tensor) -> Tuple[int, ...]:
    if tensor.ndim < 2:
        raise ValueError("projection tensors must have a leading view dimension")
    return tuple(range(1, tensor.ndim))


def _validate_projection_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {prediction.shape} vs {target.shape}"
        )
    if prediction.ndim < 2:
        raise ValueError("expected [views, ...spatial] projection tensors")


def balanced_lp_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    p: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Foreground/background balanced L1 or L2, returned per view."""

    _validate_projection_pair(prediction, target)
    if p not in (1, 2):
        raise ValueError("p must be 1 or 2")

    dims = _view_reduction_dims(prediction)
    target_float = target.to(dtype=prediction.dtype)
    error = (prediction - target_float).abs()
    if p == 2:
        error = error.square()

    foreground = target_float
    background = 1.0 - target_float
    foreground_loss = (foreground * error).sum(dim=dims) / (
        foreground.sum(dim=dims) + eps
    )
    background_loss = (background * error).sum(dim=dims) / (
        background.sum(dim=dims) + eps
    )
    return 0.5 * foreground_loss + 0.5 * background_loss


def plain_lp_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    p: int,
) -> torch.Tensor:
    """Unbalanced per-view L1/L2 used by ``l12_plain`` and legacy adapters."""

    _validate_projection_pair(prediction, target)
    if p not in (1, 2):
        raise ValueError("p must be 1 or 2")
    error = (prediction - target.to(dtype=prediction.dtype)).abs()
    if p == 2:
        error = error.square()
    return error.mean(dim=_view_reduction_dims(error))


def dual_domain_balanced_loss(
    raw_projection: torch.Tensor,
    proxy_projection: torch.Tensor,
    target: torch.Tensor,
    *,
    p: int,
    proxy_weight: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Add the compact-threshold domain while retaining the raw base loss.

    The normalized additive form ``(raw + alpha * proxy) / (1 + alpha)``
    prevents a compact-proxy plateau from removing the only useful gradient
    when the proxy schedule reaches one.
    """

    if not 0.0 <= float(proxy_weight) <= 1.0:
        raise ValueError("proxy_weight must be in [0, 1]")
    raw = balanced_lp_loss(raw_projection, target, p=p, eps=eps)
    proxy = balanced_lp_loss(proxy_projection, target, p=p, eps=eps)
    alpha = float(proxy_weight)
    return (raw + alpha * proxy) / (1.0 + alpha)


def gate_statistics(
    proxy_projection: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable per-view foreground recall and background FPR."""

    _validate_projection_pair(proxy_projection, target)
    dims = _view_reduction_dims(proxy_projection)
    target_float = target.to(dtype=proxy_projection.dtype)
    background = 1.0 - target_float
    recall = (proxy_projection * target_float).sum(dim=dims) / (
        target_float.sum(dim=dims) + eps
    )
    fpr = (proxy_projection * background).sum(dim=dims) / (
        background.sum(dim=dims) + eps
    )
    return recall, fpr


def gate_loss(
    proxy_projection: torch.Tensor,
    target: torch.Tensor,
    *,
    recall_target: float = 0.985,
    fpr_target: float = 0.01,
    fpr_activation_recall: float = 0.975,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Recall-first fabrication gate penalty, returned per view."""

    recall, fpr = gate_statistics(proxy_projection, target, eps=eps)
    recall_penalty = torch.relu(float(recall_target) - recall).square()
    fpr_penalty = torch.relu(fpr - float(fpr_target)).square()
    fpr_is_active = (recall >= float(fpr_activation_recall)).to(fpr.dtype)
    return recall_penalty + fpr_is_active * fpr_penalty


def _target_boundary(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def boundary_weights(target: torch.Tensor, *, sigma_px: float = 2.0) -> torch.Tensor:
    """Create fixed Gaussian weights from distance to each target boundary.

    Target processing is intentionally detached and performed on CPU; targets
    are fixed, so callers should compute these weights once and cache them.
    The returned tensor is restored to the target device and floating dtype.
    """

    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    if target.ndim < 3:
        raise ValueError("target must have shape [views, ..., height, width]")

    target_np = target.detach().cpu().numpy() >= 0.5
    flattened = target_np.reshape((-1,) + target_np.shape[-2:])
    result = np.empty_like(flattened, dtype=np.float32)
    for index, mask in enumerate(flattened):
        boundary = _target_boundary(mask)
        if boundary.any():
            distance = ndimage.distance_transform_edt(~boundary)
            result[index] = np.exp(-(distance**2) / (2.0 * sigma_px**2))
        else:
            # A view without a target boundary must not become a full-image
            # reconstruction term under the name "boundary".
            result[index] = 0.0
    result = result.reshape(target_np.shape)
    dtype = target.dtype if target.is_floating_point() else torch.float32
    return torch.as_tensor(result, dtype=dtype, device=target.device)


def boundary_charbonnier_loss(
    proxy_projection: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    sigma_px: float = 2.0,
    charbonnier_eps: float = 1e-6,
    denominator_eps: float = 1e-8,
) -> torch.Tensor:
    """Gaussian boundary-weighted Charbonnier penalty, returned per view."""

    _validate_projection_pair(proxy_projection, target)
    if charbonnier_eps <= 0:
        raise ValueError("charbonnier_eps must be positive")
    if weights is None:
        weights = boundary_weights(target, sigma_px=sigma_px)
    if weights.shape != target.shape:
        raise ValueError("boundary weights must match target shape")

    dims = _view_reduction_dims(proxy_projection)
    target_float = target.to(dtype=proxy_projection.dtype)
    weight_float = weights.to(
        device=proxy_projection.device, dtype=proxy_projection.dtype
    )
    error = torch.sqrt((proxy_projection - target_float).square() + charbonnier_eps)
    return (weight_float * error).sum(dim=dims) / (
        weight_float.sum(dim=dims) + denominator_eps
    )


def group_mirror_pairs(
    per_view_losses: torch.Tensor,
    *,
    pairs: Sequence[Tuple[int, int]] = DEFAULT_MIRROR_PAIRS,
) -> torch.Tensor:
    """Average the registered opposite-camera pairs into three group losses."""

    if per_view_losses.ndim != 1:
        raise ValueError("per_view_losses must be one-dimensional")
    if not pairs:
        raise ValueError("at least one mirror pair is required")
    largest_index = max(max(pair) for pair in pairs)
    if per_view_losses.numel() <= largest_index:
        raise ValueError(
            f"mirror pairs require at least {largest_index + 1} views; "
            f"received {per_view_losses.numel()}"
        )
    return torch.stack(
        [0.5 * (per_view_losses[first] + per_view_losses[second]) for first, second in pairs]
    )


def aggregate_view_losses(
    per_view_losses: torch.Tensor,
    *,
    mode: str,
    beta: float = 4.0,
    rho: float = 0.0,
    pairs: Sequence[Tuple[int, int]] = DEFAULT_MIRROR_PAIRS,
) -> torch.Tensor:
    """Apply mean or pre-registered grouped robust multi-view aggregation."""

    if per_view_losses.ndim != 1 or per_view_losses.numel() == 0:
        raise ValueError("per_view_losses must be a non-empty one-dimensional tensor")
    if mode == "mean":
        return per_view_losses.mean()
    if mode != "grouped_robust":
        raise ValueError("mode must be 'mean' or 'grouped_robust'")
    if beta <= 0:
        raise ValueError("beta must be positive")
    if not 0.0 <= float(rho) <= 1.0:
        raise ValueError("rho must be in [0, 1]")

    grouped = group_mirror_pairs(per_view_losses, pairs=pairs)
    log_mean_exp = (
        torch.logsumexp(float(beta) * grouped, dim=0) - np.log(grouped.numel())
    ) / float(beta)
    return (1.0 - float(rho)) * grouped.mean() + float(rho) * log_mean_exp


def compute_soft_loss(
    raw_projection: torch.Tensor,
    proxy_projection: torch.Tensor | None,
    target: torch.Tensor,
    *,
    profile: LossProfile | str,
    progress: float,
    lambda_gate: float = 1.0,
    lambda_boundary: float = 1.0,
    cached_boundary_weights: torch.Tensor | None = None,
    robust_beta: float = 4.0,
) -> SoftLossResult:
    """Compute a registered L1/L2/new-full objective and aggregate its views.

    ``legacy_full`` is intentionally not reimplemented here: the project-local
    trainer must route that profile through its frozen legacy adapter so that a
    new implementation cannot silently change the control condition.
    """

    if isinstance(profile, str):
        profile = resolve_profile(profile)
    if profile.legacy_terms:
        raise NotImplementedError(
            "legacy_full must be evaluated by the trainer's frozen legacy-loss adapter"
        )
    if lambda_gate < 0 or lambda_boundary < 0:
        raise ValueError("lambda_gate and lambda_boundary must be non-negative")
    needs_proxy = profile.use_proxy or profile.use_gate or profile.use_boundary
    if needs_proxy and proxy_projection is None:
        raise ValueError("this profile requires a proxy projection")

    alpha = proxy_weight_schedule(progress) if profile.use_proxy else 0.0
    if profile.balanced:
        if profile.use_proxy:
            assert proxy_projection is not None
            per_view_l2 = dual_domain_balanced_loss(
                raw_projection, proxy_projection, target, p=2, proxy_weight=alpha
            )
            per_view_l1 = dual_domain_balanced_loss(
                raw_projection, proxy_projection, target, p=1, proxy_weight=alpha
            )
        else:
            per_view_l2 = balanced_lp_loss(raw_projection, target, p=2)
            per_view_l1 = balanced_lp_loss(raw_projection, target, p=1)
    else:
        per_view_l2 = plain_lp_loss(raw_projection, target, p=2)
        per_view_l1 = plain_lp_loss(raw_projection, target, p=1)

    zeros = torch.zeros_like(per_view_l2)
    gate_schedule_value = gate_weight_schedule(progress) if profile.use_gate else 0.0
    boundary_schedule_value = (
        boundary_weight_schedule(progress) if profile.use_boundary else 0.0
    )
    if profile.use_gate:
        assert proxy_projection is not None
        per_view_gate = gate_loss(proxy_projection, target)
    else:
        per_view_gate = zeros
    if profile.use_boundary:
        assert proxy_projection is not None
        per_view_boundary = boundary_charbonnier_loss(
            proxy_projection,
            target,
            weights=cached_boundary_weights,
        )
    else:
        per_view_boundary = zeros

    per_view_total = (
        profile.l2_weight * per_view_l2
        + profile.l1_weight * per_view_l1
        + float(lambda_gate) * gate_schedule_value * per_view_gate
        + float(lambda_boundary) * boundary_schedule_value * per_view_boundary
    )
    rho = robust_rho_schedule(progress) if profile.view_aggregation == "grouped_robust" else 0.0
    total = aggregate_view_losses(
        per_view_total,
        mode=profile.view_aggregation,
        beta=robust_beta,
        rho=rho,
    )
    return SoftLossResult(
        total=total,
        per_view_total=per_view_total,
        per_view_l2=per_view_l2,
        per_view_l1=per_view_l1,
        per_view_gate=per_view_gate,
        per_view_boundary=per_view_boundary,
        proxy_weight=alpha,
        gate_schedule=gate_schedule_value,
        boundary_schedule=boundary_schedule_value,
        robust_rho=rho,
    )
