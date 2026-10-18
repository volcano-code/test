"""Guard one auxiliary gradient against anchor conflict and over-budget norm."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers

import torch


_EPS = 1e-12
_TERMINAL_CORRECTION_ROUNDS = 8
_TERMINAL_ULP_REPAIR_ROUNDS = 4


@dataclass(frozen=True)
class GuardedAuxiliaryGradient:
    gradient: torch.Tensor
    raw_norm: float
    projected_norm: float
    final_norm: float
    anchor_cosine_before: float
    anchor_cosine_after: float
    removed_conflict_fraction: float
    auxiliary_base_ratio: float
    cap_applied: bool
    fully_projected_zero: bool


def _validate_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must be finite")


def _validate_max_ratio(max_ratio: float) -> float:
    if isinstance(max_ratio, bool):
        raise ValueError("max_ratio must be a finite real number, not boolean")
    if not isinstance(max_ratio, numbers.Real):
        raise ValueError("max_ratio must be a real number")
    value = float(max_ratio)
    if not math.isfinite(value):
        raise ValueError("max_ratio must be finite")
    if value < 0.0:
        raise ValueError("max_ratio must be >= 0")
    return value


def _check_no_alias(left_name: str, left: torch.Tensor, right_name: str, right: torch.Tensor) -> None:
    if torch._C._overlaps(left, right):
        raise ValueError(f"{left_name} and {right_name} must not alias or share storage")


def _finite_norm(name: str, tensor: torch.Tensor) -> float:
    max_abs = float(tensor.abs().max().item())
    if max_abs == 0.0:
        return 0.0
    scaled_norm = float(torch.linalg.vector_norm(tensor / max_abs).item())
    norm = max_abs * scaled_norm
    if not math.isfinite(norm):
        raise ValueError(f"{name} norm must be representable")
    return norm


def _scaled_unit_dot(vector: torch.Tensor, unit_vector: torch.Tensor) -> float:
    max_abs = float(vector.abs().max().item())
    if max_abs == 0.0:
        return 0.0
    dot = max_abs * float(torch.dot((vector / max_abs).reshape(-1), unit_vector.reshape(-1)).item())
    if not math.isfinite(dot):
        raise ValueError("dot product must be representable")
    return dot


def _cosine(dot: float, left_norm: float) -> float:
    if left_norm == 0.0:
        return 0.0
    value = dot / left_norm
    return float(min(1.0, max(-1.0, value)))


def _actual_dot(left: torch.Tensor, right: torch.Tensor) -> float:
    value = float(torch.dot(left.reshape(-1), right.reshape(-1)).item())
    if math.isnan(value):
        raise ValueError("dot product must not be NaN")
    return value


def _scale_toward_zero(scale: float, reference: torch.Tensor) -> float:
    if scale <= 0.0:
        return 0.0
    if scale >= 1.0:
        return 1.0
    scale_tensor = torch.tensor(scale, dtype=reference.dtype, device=reference.device)
    return float(torch.nextafter(scale_tensor, torch.zeros_like(scale_tensor)).item())


def _terminally_revalidate_final(
    final: torch.Tensor,
    anchor_grad: torch.Tensor,
    anchor_unit: torch.Tensor | None,
    anchor_norm: float,
    allowed_norm: float,
) -> torch.Tensor:
    corrected = final.clone()
    for _ in range(_TERMINAL_CORRECTION_ROUNDS):
        if not torch.isfinite(corrected).all().item():
            raise ValueError("final auxiliary gradient must be finite")
        actual_norm = _finite_norm("final auxiliary gradient", corrected)
        actual_dot = 0.0 if anchor_unit is None else _actual_dot(corrected, anchor_grad)
        if actual_dot >= 0.0 and actual_norm <= allowed_norm:
            return corrected.clone()

        updated = False
        if anchor_unit is not None and actual_dot < 0.0:
            corrected = corrected - (actual_dot / anchor_norm) * anchor_unit
            updated = True

        actual_norm = _finite_norm("final auxiliary gradient", corrected)
        if actual_norm > allowed_norm and actual_norm > 0.0:
            corrected = corrected * _scale_toward_zero(allowed_norm / actual_norm, corrected)
            updated = True

        if not updated:
            break

    if not torch.isfinite(corrected).all().item():
        raise ValueError("final auxiliary gradient must be finite")
    actual_norm = _finite_norm("final auxiliary gradient", corrected)
    actual_dot = 0.0 if anchor_unit is None else _actual_dot(corrected, anchor_grad)
    if actual_dot >= 0.0 and actual_norm <= allowed_norm:
        return corrected.clone()

    # The analytic correction can round to the same source-dtype tensor while
    # leaving a tiny negative dot.  Move every negative-contribution coordinate
    # exactly one representable value toward zero.  This is vectorized (one scan
    # per bounded round), decreases coordinate magnitudes, and is accepted only
    # when the source-dtype dot strictly improves.  The scaled norm may round one
    # way on an intermediate candidate and back on the next even though every
    # coordinate magnitude decreases, so only the terminal return enforces the
    # budget rather than stopping this bounded repair early.
    if anchor_unit is not None and actual_dot < 0.0 and actual_norm <= allowed_norm:
        zero = torch.zeros_like(corrected)
        for _ in range(_TERMINAL_ULP_REPAIR_ROUNDS):
            negative_contribution = (
                (corrected != 0)
                & (anchor_grad != 0)
                & torch.logical_xor(torch.signbit(corrected), torch.signbit(anchor_grad))
            )
            if not negative_contribution.any().item():
                break

            one_ulp_toward_zero = torch.nextafter(corrected, zero)
            candidate = torch.where(negative_contribution, one_ulp_toward_zero, corrected)
            if torch.equal(candidate, corrected):
                break
            if not torch.isfinite(candidate).all().item():
                break

            candidate_norm = _finite_norm("final auxiliary gradient", candidate)
            candidate_dot = _actual_dot(candidate, anchor_grad)
            if candidate_dot <= actual_dot:
                break

            corrected = candidate
            actual_norm = candidate_norm
            actual_dot = candidate_dot
            if actual_dot >= 0.0 and actual_norm <= allowed_norm:
                return corrected.clone()

    raise ValueError("final auxiliary gradient violates terminal guard contract")


def guard_auxiliary_gradient(
    base_grad: torch.Tensor,
    anchor_grad: torch.Tensor,
    aux_grad: torch.Tensor,
    max_ratio: float,
) -> GuardedAuxiliaryGradient:
    _validate_tensor("base_grad", base_grad)
    _validate_tensor("anchor_grad", anchor_grad)
    _validate_tensor("aux_grad", aux_grad)
    if base_grad.shape != anchor_grad.shape or base_grad.shape != aux_grad.shape:
        raise ValueError("base_grad, anchor_grad, and aux_grad must have the same shape")
    if base_grad.device != anchor_grad.device or base_grad.device != aux_grad.device:
        raise ValueError("base_grad, anchor_grad, and aux_grad must share a device")
    if base_grad.dtype != anchor_grad.dtype or base_grad.dtype != aux_grad.dtype:
        raise ValueError("base_grad, anchor_grad, and aux_grad must share a dtype")
    _check_no_alias("base_grad", base_grad, "anchor_grad", anchor_grad)
    _check_no_alias("base_grad", base_grad, "aux_grad", aux_grad)
    _check_no_alias("anchor_grad", anchor_grad, "aux_grad", aux_grad)
    ratio = _validate_max_ratio(max_ratio)

    raw = aux_grad.clone()
    raw_norm = _finite_norm("aux_grad", raw)
    base_norm = _finite_norm("base_grad", base_grad)
    anchor_norm = _finite_norm("anchor_grad", anchor_grad)

    if anchor_norm == 0.0:
        projected = raw.clone()
        anchor_cosine_before = 0.0
    else:
        anchor_unit = anchor_grad / anchor_norm
        raw_dot_unit = _scaled_unit_dot(raw, anchor_unit)
        anchor_cosine_before = _cosine(raw_dot_unit, raw_norm)
        eps_scale = _EPS * (1.0 / anchor_norm) * (1.0 / anchor_norm)
        correction = min(raw_dot_unit, 0.0) / (1.0 + eps_scale)
        projected = raw - correction * anchor_unit
        residual_dot = _scaled_unit_dot(projected, anchor_unit)
        if residual_dot < 0.0:
            projected = projected - residual_dot * anchor_unit

    projected = projected.clone()
    projected_norm = _finite_norm("projected auxiliary gradient", projected)
    if anchor_norm == 0.0:
        anchor_cosine_after = 0.0
    else:
        projected_dot_unit = _scaled_unit_dot(projected, anchor_unit)
        anchor_cosine_after = _cosine(projected_dot_unit, projected_norm)

    allowed_norm = ratio * base_norm
    cap_applied = projected_norm > allowed_norm
    if cap_applied and projected_norm > 0.0:
        final = projected * (allowed_norm / projected_norm)
    else:
        final = projected.clone()

    final = _terminally_revalidate_final(
        final=final,
        anchor_grad=anchor_grad,
        anchor_unit=anchor_unit if anchor_norm > 0.0 else None,
        anchor_norm=anchor_norm,
        allowed_norm=allowed_norm,
    )
    final_norm = _finite_norm("final auxiliary gradient", final)
    if raw_norm > 0.0:
        removed_conflict_fraction = min(1.0, max(0.0, (raw_norm - projected_norm) / raw_norm))
    else:
        removed_conflict_fraction = 0.0
    auxiliary_base_ratio = (final_norm / base_norm) if base_norm > 0.0 else 0.0
    fully_projected_zero = raw_norm > 0.0 and projected_norm == 0.0

    return GuardedAuxiliaryGradient(
        gradient=final.clone(),
        raw_norm=raw_norm,
        projected_norm=projected_norm,
        final_norm=final_norm,
        anchor_cosine_before=anchor_cosine_before,
        anchor_cosine_after=anchor_cosine_after,
        removed_conflict_fraction=removed_conflict_fraction,
        auxiliary_base_ratio=auxiliary_base_ratio,
        cap_applied=cap_applied,
        fully_projected_zero=fully_projected_zero,
    )
