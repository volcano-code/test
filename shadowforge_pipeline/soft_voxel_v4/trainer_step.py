"""One-update kernel and audit records for soft_voxel_v4."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Literal

import torch

from shadowforge_pipeline.soft_voxel_v3.contours import ContourBands, validate_contour_bands
from shadowforge_pipeline.soft_voxel_v3.lovasz import lovasz_hinge_per_view
from shadowforge_pipeline.soft_voxel_v3.thresholding import _protocol_device, render_thresholded

from .gradients import GuardedAuxiliaryGradient, guard_auxiliary_gradient
from .losses import (
    V4Profile,
    asymmetric_dts_loss,
    auxiliary_slot,
    compute_baseline_branch,
    directional_dic_loss,
    resolve_profile,
)


_ALLOWED_RATIOS = (0.04, 0.08, 0.12)
_FULL_PROFILES = {"v4_full", "v4_no_l1", "v4_no_l2"}


@dataclass(frozen=True)
class V4StepConfig:
    dic_unit_coefficient: float
    dts_unit_coefficient: float
    dic_max_ratio: float
    dts_max_ratio: float

    def __post_init__(self) -> None:
        _strict_positive_float("dic_unit_coefficient", self.dic_unit_coefficient)
        _strict_positive_float("dts_unit_coefficient", self.dts_unit_coefficient)
        _strict_ratio("dic_max_ratio", self.dic_max_ratio)
        _strict_ratio("dts_max_ratio", self.dts_max_ratio)


@dataclass(frozen=True)
class AuxiliarySlotAudit:
    global_step_before: int
    profile: Literal["dic_only", "dts_only", "v4_full", "v4_no_l1", "v4_no_l2"]
    cross_diagnostics_expected: bool
    component: Literal["dic", "dts"]
    endpoint: float | None
    multiplier: float
    unit_coefficient: float
    max_ratio: float
    loss_auxiliary: float
    raw_component_gradient_norm: float
    weighted_auxiliary_gradient_norm: float
    projected_auxiliary_gradient_norm: float
    final_auxiliary_gradient_norm: float
    anchor_cosine_before: float
    anchor_cosine_after: float
    removed_conflict_fraction: float
    auxiliary_base_ratio: float
    cap_applied: bool
    fully_projected_zero: bool
    diagnostic_dic_raw_gradient_norm: float | None
    diagnostic_dts_004_raw_gradient_norm: float | None
    diagnostic_dts_006_raw_gradient_norm: float | None
    dic_dts_004_support_intersection: int | None
    dic_dts_004_support_union: int | None
    dic_dts_006_support_intersection: int | None
    dic_dts_006_support_union: int | None
    dic_dts_004_cosine: float | None
    dic_dts_004_cosine_invalid: bool
    dic_dts_006_cosine: float | None
    dic_dts_006_cosine_invalid: bool
    dic_dts_004_support_iou: float | None
    dic_dts_004_support_iou_invalid: bool
    dic_dts_006_support_iou: float | None
    dic_dts_006_support_iou_invalid: bool


@dataclass(frozen=True)
class V4StepResult:
    global_step_before: int
    completed_updates_after: int
    progress: float
    slot_component: str | None
    slot_endpoint: float | None
    slot_multiplier: float
    loss_base: float
    loss_anchor: float | None
    loss_auxiliary: float | None
    auxiliary_scale: float
    pre_clip_gradient_norm: float
    post_clip_gradient_norm: float
    slot_audit: AuxiliarySlotAudit | None


def _strict_positive_float(name: str, value: object) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be an exact finite positive Python float")
    return value


def _strict_ratio(name: str, value: object) -> float:
    scalar = _strict_positive_float(name, value)
    if scalar not in _ALLOWED_RATIOS:
        raise ValueError(f"{name} must be exactly one of {_ALLOWED_RATIOS}")
    return scalar


def _validate_logits(logits: object) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise ValueError("logits must be a torch.Tensor")
    if not logits.is_floating_point():
        raise ValueError("logits must use a floating-point dtype")
    if (
        logits.ndim != 5
        or logits.shape[:2] != (1, 1)
        or logits.shape[2] == 0
        or logits.shape[2:] != (logits.shape[2],) * 3
    ):
        raise ValueError("logits must have non-empty cubic shape [1,1,D,H,W]")
    if not logits.is_leaf:
        raise ValueError("logits must be a leaf tensor")
    if not logits.requires_grad:
        raise ValueError("logits must be gradient-enabled")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must be finite")
    return logits


def _validate_optimizer(optimizer: object, logits: torch.Tensor) -> None:
    if not callable(getattr(optimizer, "zero_grad", None)):
        raise ValueError("optimizer must expose callable zero_grad")
    if not callable(getattr(optimizer, "step", None)):
        raise ValueError("optimizer must expose callable step")
    parameters: list[object] = []
    groups = getattr(optimizer, "param_groups", None)
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("params"), list):
                raise ValueError("optimizer parameter groups are malformed")
            parameters.extend(group["params"])
    elif hasattr(optimizer, "parameter"):
        parameters.append(getattr(optimizer, "parameter"))
    else:
        raise ValueError("optimizer must own the exact live logits object")
    if len(parameters) != 1 or parameters[0] is not logits:
        raise ValueError("optimizer must own the exact live logits object")


def _validate_protocol(protocol: object, logits: torch.Tensor) -> None:
    if not callable(getattr(protocol, "render_all", None)):
        raise ValueError("protocol must expose callable render_all")
    has_state = (
        getattr(protocol, "cameras", None) is not None
        or getattr(protocol, "renderer", None) is not None
        or hasattr(protocol, "device")
    )
    if has_state:
        try:
            protocol_device = _protocol_device(protocol)
        except ValueError as error:
            raise ValueError("protocol must expose a valid device") from error
        if protocol_device != logits.device:
            raise ValueError("protocol and logits must be on the same device")


def _validate_targets(targets: object, logits: torch.Tensor) -> torch.Tensor:
    if not isinstance(targets, torch.Tensor):
        raise ValueError("targets must be a torch.Tensor")
    if not targets.is_floating_point():
        raise ValueError("targets must use a floating-point dtype")
    if targets.ndim != 3 or targets.shape[0] != 6 or any(size == 0 for size in targets.shape[1:]):
        raise ValueError("targets must have non-empty shape [6,H,W]")
    if targets.device != logits.device:
        raise ValueError("targets and logits must be on the same device")
    if not torch.isfinite(targets).all().item():
        raise ValueError("targets must be finite")
    return targets


def _validate_profile(profile: object) -> V4Profile:
    if not isinstance(profile, V4Profile):
        raise ValueError("profile must be a V4Profile")
    if (
        type(profile.name) is not str
        or type(profile.l1_weight) is not float
        or type(profile.l2_weight) is not float
        or type(profile.use_dic) is not bool
        or type(profile.use_dts) is not bool
    ):
        raise ValueError("profile fields must use the exact registered schema")
    try:
        registered = resolve_profile(profile.name)
    except ValueError as error:
        raise ValueError("profile must be a registered V4Profile") from error
    if profile is not registered:
        raise ValueError("profile must be a registered V4Profile")
    return profile


def _validate_counts(global_step_before: object, total_steps: object) -> tuple[int, int]:
    if (
        isinstance(global_step_before, bool)
        or not isinstance(global_step_before, Integral)
        or global_step_before < 0
    ):
        raise ValueError("global_step_before must be a non-negative integer")
    if isinstance(total_steps, bool) or not isinstance(total_steps, Integral) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    step = int(global_step_before)
    total = int(total_steps)
    if step >= total:
        raise ValueError("global_step_before must be less than total_steps")
    return step, total


def _validate_active_bands(bands: object, targets: torch.Tensor) -> ContourBands:
    if not isinstance(bands, ContourBands):
        raise ValueError("bands must be a ContourBands instance while DIC is active")
    validate_contour_bands(bands)
    if bands.inner.shape != targets.shape or bands.outer.shape != targets.shape:
        raise ValueError("bands and targets must have the same shape")
    if bands.inner.device != targets.device or bands.outer.device != targets.device:
        raise ValueError("bands and targets must share a device")
    return bands


def _finite_scalar_loss(loss: object, *, name: str) -> torch.Tensor:
    if (
        not isinstance(loss, torch.Tensor)
        or loss.ndim != 0
        or not loss.is_floating_point()
        or not torch.isfinite(loss).item()
    ):
        raise RuntimeError(f"{name} loss must be a finite scalar tensor")
    return loss


def _fresh_grad(logits: torch.Tensor, loss: torch.Tensor, *, name: str) -> torch.Tensor:
    checked = _finite_scalar_loss(loss, name=name)
    try:
        (gradient,) = torch.autograd.grad(
            checked,
            logits,
            retain_graph=False,
            create_graph=False,
        )
    except RuntimeError as error:
        raise RuntimeError(f"{name} gradient must be connected and finite") from error
    if (
        not isinstance(gradient, torch.Tensor)
        or gradient.shape != logits.shape
        or not gradient.is_floating_point()
        or not torch.isfinite(gradient).all().item()
    ):
        raise RuntimeError(f"{name} gradient must be finite")
    return gradient.detach().clone()


def _finite_norm(tensor: torch.Tensor, *, name: str) -> float:
    value = float(torch.linalg.vector_norm(tensor.to(dtype=torch.float64)).item())
    if not math.isfinite(value):
        raise RuntimeError(f"{name} norm must be finite")
    return value


def _relation(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[float | None, bool, int, int, float | None, bool]:
    left_norm = _finite_norm(left, name="left raw gradient")
    right_norm = _finite_norm(right, name="right raw gradient")
    if left_norm == 0.0 or right_norm == 0.0:
        cosine = None
        cosine_invalid = True
    else:
        left_unit = left.to(dtype=torch.float64) / left_norm
        right_unit = right.to(dtype=torch.float64) / right_norm
        cosine = float(torch.dot(left_unit.reshape(-1), right_unit.reshape(-1)).item())
        cosine = min(1.0, max(-1.0, cosine))
        cosine_invalid = False
    left_support = left.abs() > 1e-12
    right_support = right.abs() > 1e-12
    union = torch.logical_or(left_support, right_support)
    union_count = int(union.sum().item())
    intersection_count = int(torch.logical_and(left_support, right_support).sum().item())
    if union_count == 0:
        iou = None
        iou_invalid = True
    else:
        iou = float(intersection_count / union_count)
        iou_invalid = False
    return cosine, cosine_invalid, intersection_count, union_count, iou, iou_invalid


def _selected_scale(config: V4StepConfig, component: str, multiplier: float) -> tuple[float, float]:
    if component == "dic":
        return config.dic_unit_coefficient * config.dic_max_ratio * multiplier, config.dic_max_ratio
    return config.dts_unit_coefficient * config.dts_max_ratio * multiplier, config.dts_max_ratio


def _check_unchanged(reference: torch.Tensor, current: torch.Tensor, *, name: str) -> None:
    if not torch.equal(reference, current):
        raise RuntimeError(f"{name} must remain unchanged")


def _build_audit(
    *,
    global_step_before: int,
    profile: V4Profile,
    slot: object,
    config: V4StepConfig,
    selected_loss: float,
    selected_raw_gradient: torch.Tensor,
    weighted_auxiliary_gradient: torch.Tensor,
    guard: GuardedAuxiliaryGradient,
    diagnostic_dic_gradient: torch.Tensor | None,
    diagnostic_dts_004_gradient: torch.Tensor | None,
    diagnostic_dts_006_gradient: torch.Tensor | None,
) -> AuxiliarySlotAudit:
    scale, max_ratio = _selected_scale(config, slot.component, slot.multiplier)
    raw_norm = _finite_norm(selected_raw_gradient, name="selected raw auxiliary")
    weighted_norm = _finite_norm(weighted_auxiliary_gradient, name="weighted auxiliary")
    cross_expected = profile.name in _FULL_PROFILES

    if cross_expected:
        assert diagnostic_dic_gradient is not None
        assert diagnostic_dts_004_gradient is not None
        assert diagnostic_dts_006_gradient is not None
        dic_norm = _finite_norm(diagnostic_dic_gradient, name="diagnostic DIC")
        dts_004_norm = _finite_norm(diagnostic_dts_004_gradient, name="diagnostic DTS[0.04]")
        dts_006_norm = _finite_norm(diagnostic_dts_006_gradient, name="diagnostic DTS[0.06]")
        rel_004 = _relation(diagnostic_dic_gradient, diagnostic_dts_004_gradient)
        rel_006 = _relation(diagnostic_dic_gradient, diagnostic_dts_006_gradient)
    else:
        dic_norm = None
        dts_004_norm = None
        dts_006_norm = None
        rel_004 = (None, True, None, None, None, True)
        rel_006 = (None, True, None, None, None, True)

    return AuxiliarySlotAudit(
        global_step_before=global_step_before,
        profile=profile.name,
        cross_diagnostics_expected=cross_expected,
        component=slot.component,
        endpoint=slot.endpoint,
        multiplier=float(slot.multiplier),
        unit_coefficient=(config.dic_unit_coefficient if slot.component == "dic" else config.dts_unit_coefficient),
        max_ratio=max_ratio,
        loss_auxiliary=selected_loss,
        raw_component_gradient_norm=raw_norm,
        weighted_auxiliary_gradient_norm=weighted_norm,
        projected_auxiliary_gradient_norm=guard.projected_norm,
        final_auxiliary_gradient_norm=guard.final_norm,
        anchor_cosine_before=guard.anchor_cosine_before,
        anchor_cosine_after=guard.anchor_cosine_after,
        removed_conflict_fraction=guard.removed_conflict_fraction,
        auxiliary_base_ratio=guard.auxiliary_base_ratio,
        cap_applied=guard.cap_applied,
        fully_projected_zero=guard.fully_projected_zero,
        diagnostic_dic_raw_gradient_norm=dic_norm,
        diagnostic_dts_004_raw_gradient_norm=dts_004_norm,
        diagnostic_dts_006_raw_gradient_norm=dts_006_norm,
        dic_dts_004_support_intersection=rel_004[2],
        dic_dts_004_support_union=rel_004[3],
        dic_dts_006_support_intersection=rel_006[2],
        dic_dts_006_support_union=rel_006[3],
        dic_dts_004_cosine=rel_004[0],
        dic_dts_004_cosine_invalid=rel_004[1],
        dic_dts_006_cosine=rel_006[0],
        dic_dts_006_cosine_invalid=rel_006[1],
        dic_dts_004_support_iou=rel_004[4],
        dic_dts_004_support_iou_invalid=rel_004[5],
        dic_dts_006_support_iou=rel_006[4],
        dic_dts_006_support_iou_invalid=rel_006[5],
    )


def train_one_v4_update(
    *,
    logits: torch.Tensor,
    optimizer: object,
    protocol: object,
    targets: torch.Tensor,
    bands: ContourBands | None,
    profile: V4Profile,
    global_step_before: int,
    total_steps: int,
    config: V4StepConfig,
) -> V4StepResult:
    logits = _validate_logits(logits)
    _validate_optimizer(optimizer, logits)
    _validate_protocol(protocol, logits)
    targets = _validate_targets(targets, logits)
    profile = _validate_profile(profile)
    step, total = _validate_counts(global_step_before, total_steps)
    if not isinstance(config, V4StepConfig):
        raise ValueError("config must be a V4StepConfig")
    progress = (step + 1) / float(total)
    slot = auxiliary_slot(step, progress, profile)

    try:
        optimizer.zero_grad(set_to_none=True)
        density = torch.sigmoid(logits)
        base_loss = _finite_scalar_loss(
            compute_baseline_branch(protocol.render_all(density), targets, profile),
            name="baseline",
        )
        base_grad = _fresh_grad(logits, base_loss, name="baseline")
        loss_base = float(base_loss.detach().item())

        if slot is None:
            logits.grad = base_grad.detach().clone()
            pre_clip = torch.nn.utils.clip_grad_norm_([logits], 1.0)
            pre_clip_gradient_norm = float(pre_clip)
            if not math.isfinite(pre_clip_gradient_norm):
                raise RuntimeError("pre-clip gradient norm must be finite")
            if logits.grad is None or not torch.isfinite(logits.grad).all().item():
                raise RuntimeError("post-clip gradient must be finite")
            post_clip_gradient_norm = _finite_norm(logits.grad, name="post-clip gradient")
            optimizer.step()
            return V4StepResult(
                global_step_before=step,
                completed_updates_after=step + 1,
                progress=float(progress),
                slot_component=None,
                slot_endpoint=None,
                slot_multiplier=0.0,
                loss_base=loss_base,
                loss_anchor=None,
                loss_auxiliary=None,
                auxiliary_scale=0.0,
                pre_clip_gradient_norm=pre_clip_gradient_norm,
                post_clip_gradient_norm=post_clip_gradient_norm,
                slot_audit=None,
            )

        target_bool = targets >= 0.5

        anchor_density = torch.sigmoid(logits)
        anchor_readout = render_thresholded(protocol, anchor_density, 0.05)
        anchor_loss = _finite_scalar_loss(
            lovasz_hinge_per_view(anchor_readout.score / 0.05, target_bool).mean(),
            name="anchor",
        )
        anchor_grad = _fresh_grad(logits, anchor_loss, name="anchor")
        loss_anchor = float(anchor_loss.detach().item())

        cross_expected = profile.name in _FULL_PROFILES
        diagnostic_dic_gradient = None
        diagnostic_dts_004_gradient = None
        diagnostic_dts_006_gradient = None
        selected_loss = None
        selected_raw_gradient = None

        if slot.component == "dic" or cross_expected:
            active_bands = _validate_active_bands(bands, targets)
            dic_density = torch.sigmoid(logits)
            dic_readout = render_thresholded(protocol, dic_density, 0.05)
            dic_loss = _finite_scalar_loss(
                directional_dic_loss(dic_readout.score, active_bands).mean(),
                name="DIC",
            )
            dic_gradient = _fresh_grad(logits, dic_loss, name="DIC")
            diagnostic_dic_gradient = dic_gradient
            if slot.component == "dic":
                selected_loss = float(dic_loss.detach().item())
                selected_raw_gradient = dic_gradient

        if slot.component == "dts" or cross_expected:
            for endpoint in (0.04, 0.06) if cross_expected else (slot.endpoint,):
                assert endpoint in (0.04, 0.06)
                dts_density = torch.sigmoid(logits)
                dts_readout = render_thresholded(protocol, dts_density, endpoint)
                dts_loss = _finite_scalar_loss(
                    asymmetric_dts_loss(dts_readout.score, target_bool, endpoint).mean(),
                    name="DTS",
                )
                dts_gradient = _fresh_grad(logits, dts_loss, name="DTS")
                if endpoint == 0.04:
                    diagnostic_dts_004_gradient = dts_gradient
                else:
                    diagnostic_dts_006_gradient = dts_gradient
                if slot.component == "dts" and endpoint == slot.endpoint:
                    selected_loss = float(dts_loss.detach().item())
                    selected_raw_gradient = dts_gradient

        if selected_loss is None or selected_raw_gradient is None:
            raise RuntimeError("selected auxiliary branch was not constructed")

        selected_scale, selected_max_ratio = _selected_scale(config, slot.component, slot.multiplier)
        weighted_auxiliary_gradient = selected_raw_gradient.detach().clone() * selected_scale
        base_before_guard = base_grad.detach().clone()
        anchor_before_guard = anchor_grad.detach().clone()
        weighted_before_guard = weighted_auxiliary_gradient.detach().clone()
        guard = guard_auxiliary_gradient(
            base_grad.detach().clone(),
            anchor_grad.detach().clone(),
            weighted_auxiliary_gradient.detach().clone(),
            selected_max_ratio,
        )
        _check_unchanged(base_before_guard, base_grad, name="base gradient")
        _check_unchanged(anchor_before_guard, anchor_grad, name="anchor gradient")
        _check_unchanged(weighted_before_guard, weighted_auxiliary_gradient, name="weighted auxiliary gradient")
        final_gradient = base_grad + guard.gradient
        if not torch.isfinite(final_gradient).all().item():
            raise RuntimeError("final gradient must be finite")
        logits.grad = final_gradient.detach().clone()
        pre_clip = torch.nn.utils.clip_grad_norm_([logits], 1.0)
        pre_clip_gradient_norm = float(pre_clip)
        if not math.isfinite(pre_clip_gradient_norm):
            raise RuntimeError("pre-clip gradient norm must be finite")
        if logits.grad is None or not torch.isfinite(logits.grad).all().item():
            raise RuntimeError("post-clip gradient must be finite")
        post_clip_gradient_norm = _finite_norm(logits.grad, name="post-clip gradient")
        slot_audit = _build_audit(
            global_step_before=step,
            profile=profile,
            slot=slot,
            config=config,
            selected_loss=selected_loss,
            selected_raw_gradient=selected_raw_gradient,
            weighted_auxiliary_gradient=weighted_auxiliary_gradient,
            guard=guard,
            diagnostic_dic_gradient=diagnostic_dic_gradient,
            diagnostic_dts_004_gradient=diagnostic_dts_004_gradient,
            diagnostic_dts_006_gradient=diagnostic_dts_006_gradient,
        )
        optimizer.step()
        return V4StepResult(
            global_step_before=step,
            completed_updates_after=step + 1,
            progress=float(progress),
            slot_component=slot.component,
            slot_endpoint=slot.endpoint,
            slot_multiplier=float(slot.multiplier),
            loss_base=loss_base,
            loss_anchor=loss_anchor,
            loss_auxiliary=selected_loss,
            auxiliary_scale=selected_scale,
            pre_clip_gradient_norm=pre_clip_gradient_norm,
            post_clip_gradient_norm=post_clip_gradient_norm,
            slot_audit=slot_audit,
        )
    except Exception:
        logits.grad = None
        raise


__all__ = [
    "AuxiliarySlotAudit",
    "V4StepConfig",
    "V4StepResult",
    "train_one_v4_update",
]
