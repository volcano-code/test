"""Registered physical core math and auxiliary objectives for soft-loss v7."""

from __future__ import annotations

import math
import operator
from typing import Sequence

import torch
import torch.nn.functional as F


_D4_STACK = (3, 0, 7, 7, 4, 3)
_PROFILE_SLOTS = {
    "base": frozenset(),
    "hit_only": frozenset({"hit"}),
    "barrier_only": frozenset({"barrier"}),
    "full": frozenset({"hit", "barrier"}),
}
_VALID_RATIOS = frozenset({0.10, 0.15, 0.20})
_EPS = 1e-12


def _apply_d4(mask: torch.Tensor, op_id: int) -> torch.Tensor:
    rotated = torch.rot90(mask, op_id % 4, dims=(-2, -1))
    if op_id >= 4:
        rotated = torch.flip(rotated, dims=(-1,))
    return rotated.contiguous().clone()


def _parse_suite_views(
    suite_views: Sequence[tuple[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(suite_views, (str, bytes)):
        raise ValueError("suite_views must be exactly labelled x,y,z")
    try:
        rows = tuple(suite_views)
    except TypeError as error:
        raise ValueError("suite_views must be exactly labelled x,y,z") from error
    if len(rows) != 3:
        raise ValueError("suite_views must contain exactly three views")
    masks: list[torch.Tensor] = []
    for required, row in zip(("x", "y", "z"), rows, strict=True):
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("each suite view must be an (axis, mask) pair")
        axis, mask = row
        if type(axis) is not str or axis != required:
            raise ValueError("suite view axes must be unique and ordered x,y,z")
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError("suite masks must be boolean torch.Tensor values")
        if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[0] != mask.shape[1]:
            raise ValueError("suite masks must be non-empty square matrices")
        masks.append(mask)
    reference = masks[0]
    if any(mask.shape != reference.shape for mask in masks[1:]):
        raise ValueError("suite masks must have equal shapes")
    if any(mask.device != reference.device for mask in masks[1:]):
        raise ValueError("suite masks must share a device")
    return masks[0], masks[1], masks[2]


def build_registered_target_stack(
    suite_views: Sequence[tuple[str, torch.Tensor]],
) -> torch.Tensor:
    """Register authoritative boolean x/y/z masks into six camera views."""

    target_x, target_y, target_z = _parse_suite_views(suite_views)
    stack = (
        _apply_d4(target_z, 3),
        target_x.contiguous().clone(),
        _apply_d4(target_y, 7),
        _apply_d4(target_z, 7),
        _apply_d4(target_x, 4),
        _apply_d4(target_y, 3),
    )
    return torch.stack(stack).contiguous().clone()


def _validate_volume(name: str, volume: torch.Tensor) -> None:
    if not isinstance(volume, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if volume.ndim != 3 or any(size == 0 for size in volume.shape):
        raise ValueError(f"{name} must be a non-empty cubic rank-three tensor")
    if not (volume.shape[0] == volume.shape[1] == volume.shape[2]):
        raise ValueError(f"{name} must be cubic")
    if (volume.is_floating_point() or volume.is_complex()) and not torch.isfinite(volume).all().item():
        raise ValueError(f"{name} must be finite")


def xyz_to_dhw(volume_xyz: torch.Tensor) -> torch.Tensor:
    """Map physical ``V[X,Y,Z]`` to renderer ``density[D,H,W]``."""

    _validate_volume("volume_xyz", volume_xyz)
    return torch.flip(volume_xyz.permute(2, 1, 0), dims=(0, 1, 2)).contiguous().clone()


def dhw_to_xyz(volume_dhw: torch.Tensor) -> torch.Tensor:
    """Invert :func:`xyz_to_dhw` with independent contiguous storage."""

    _validate_volume("volume_dhw", volume_dhw)
    return torch.flip(volume_dhw, dims=(0, 1, 2)).permute(2, 1, 0).contiguous().clone()


def build_registered_visual_hull_logits(
    suite_views: Sequence[tuple[str, torch.Tensor]],
    *,
    _resolution: int = 256,
    _allow_test_size: bool = False,
    _dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build exact native ``+2/-4`` physical visual-hull logits."""

    if type(_allow_test_size) is not bool:
        raise ValueError("_allow_test_size must be boolean")
    if type(_resolution) is not int or _resolution <= 0:
        raise ValueError("_resolution must be a positive integer")
    if not _allow_test_size and _resolution != 256:
        raise ValueError("production resolution must be exactly 256")
    if _dtype not in (torch.float32, torch.float64):
        raise ValueError("test dtype must be float32 or float64")
    if not _allow_test_size and _dtype != torch.float32:
        raise ValueError("production dtype must be exactly float32")
    target_x, target_y, target_z = _parse_suite_views(suite_views)
    if target_x.shape != (_resolution, _resolution):
        raise ValueError("suite mask shape must equal the registered resolution")
    hull_xyz = target_x[None, :, :] & target_y[:, None, :] & target_z[:, :, None]
    hull_dhw = xyz_to_dhw(hull_xyz)
    logits = torch.where(
        hull_dhw,
        torch.tensor(2.0, dtype=_dtype, device=hull_dhw.device),
        torch.tensor(-4.0, dtype=_dtype, device=hull_dhw.device),
    )
    return logits.reshape(1, 1, _resolution, _resolution, _resolution).contiguous().clone()


def _validate_projection_target(
    projection: torch.Tensor, target_binary: torch.Tensor
) -> None:
    if not isinstance(projection, torch.Tensor) or projection.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise ValueError("projection must be a float32 or float64 torch.Tensor")
    if not isinstance(target_binary, torch.Tensor) or target_binary.dtype != torch.bool:
        raise ValueError("target_binary must be a boolean torch.Tensor")
    if (
        projection.ndim != 3
        or projection.shape[0] != 6
        or projection.shape[-2] == 0
        or projection.shape[-2] != projection.shape[-1]
        or projection.shape != target_binary.shape
    ):
        raise ValueError("projection and target must have shape [6,N,N]")
    if projection.device != target_binary.device:
        raise ValueError("projection and target must share a device")
    if not torch.isfinite(projection).all().item():
        raise ValueError("projection must be finite")
    if (projection < 0).any().item():
        raise ValueError("projection must be nonnegative")
    limit = math.sqrt(torch.finfo(projection.dtype).max)
    if float(projection.detach().amax().item()) > limit:
        raise ValueError("projection magnitude is not safely representable")


def _normalized_q(projection: torch.Tensor) -> torch.Tensor:
    q = _normalized_projection(projection) - projection.new_tensor(0.8)
    if not torch.isfinite(q).all().item():
        raise ValueError("normalized projection is non-finite")
    return q


def _normalized_projection(projection: torch.Tensor) -> torch.Tensor:
    maximum = projection.detach().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    normalized = projection / maximum
    if not torch.isfinite(normalized).all().item():
        raise ValueError("normalized projection is non-finite")
    return normalized


def coverage_hit_loss(
    projection: torch.Tensor,
    target_binary: torch.Tensor,
    *,
    return_details: bool = False,
    _allow_test_size: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, object]]]:
    """Coverage-Hit: stable worst-quarter foreground margin, view-equal."""

    _validate_projection_target(projection, target_binary)
    if type(return_details) is not bool or type(_allow_test_size) is not bool:
        raise ValueError("boolean control flags must be exact booleans")
    if not _allow_test_size and (
        projection.dtype != torch.float32 or projection.shape[-1] != 256
    ):
        raise ValueError("production Coverage-Hit inputs must be native 256 float32")
    q = _normalized_q(projection)
    flat_q = q.reshape(6, -1)
    flat_target = target_binary.reshape(6, -1)
    losses: list[torch.Tensor] = []
    details: list[dict[str, object]] = []
    for view in range(6):
        indices = torch.nonzero(flat_target[view], as_tuple=False).flatten()
        count = int(indices.numel())
        if count == 0:
            raise ValueError("every registered view must contain foreground")
        values = flat_q[view, indices]
        penalty = projection.new_tensor(0.02) * F.softplus(
            (projection.new_tensor(0.02) - values) / projection.new_tensor(0.02)
        )
        selected_count = math.ceil(0.25 * count)
        order = torch.argsort(penalty, descending=True, stable=True)
        chosen = order[:selected_count]
        losses.append(penalty[chosen].mean())
        details.append(
            {
                "view": view,
                "foreground_count": count,
                "selected_count": selected_count,
                "selected_indices": [int(item) for item in indices[chosen].detach().cpu().tolist()],
            }
        )
    result = torch.stack(losses).mean()
    if not torch.isfinite(result).item():
        raise ValueError("Coverage-Hit loss is non-finite")
    return (result, details) if return_details else result


def _validate_barrier_density(
    projection: torch.Tensor,
    density_dhw: torch.Tensor,
    *,
    allow_test_size: bool,
) -> int:
    if not isinstance(density_dhw, torch.Tensor) or density_dhw.dtype != projection.dtype:
        raise ValueError("density_dhw must match projection floating dtype")
    size = projection.shape[-1]
    if density_dhw.shape != (1, 1, size, size, size):
        raise ValueError("density_dhw must have shape [1,1,N,N,N]")
    if density_dhw.device != projection.device:
        raise ValueError("density_dhw and projection must share a device")
    if not torch.isfinite(density_dhw).all().item():
        raise ValueError("density_dhw must be finite")
    if not allow_test_size and size != 256:
        raise ValueError("production barrier resolution must be exactly 256")
    if not allow_test_size and projection.dtype != torch.float32:
        raise ValueError("production barrier dtype must be exactly float32")
    return size


def leakage_barrier_loss(
    projection: torch.Tensor,
    target_binary: torch.Tensor,
    density_dhw: torch.Tensor,
    *,
    return_details: bool = False,
    _allow_test_size: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    """Leakage-Barrier on the exact detached Hit-causal physical footprint."""

    _validate_projection_target(projection, target_binary)
    if type(return_details) is not bool or type(_allow_test_size) is not bool:
        raise ValueError("boolean control flags must be exact booleans")
    _validate_barrier_density(projection, density_dhw, allow_test_size=_allow_test_size)
    normalized = _normalized_projection(projection)
    q = normalized - projection.new_tensor(0.8)
    detached_q = q.detach()
    missing = target_binary & (detached_q < projection.new_tensor(0.02))
    missing_z = _apply_d4(missing[0], 1) | _apply_d4(missing[3], 7)
    missing_x = missing[1].contiguous().clone() | _apply_d4(missing[4], 4)
    missing_y = _apply_d4(missing[2], 7) | _apply_d4(missing[5], 1)
    occupied_xyz = dhw_to_xyz(density_dhw.detach()[0, 0]) >= density_dhw.new_tensor(0.05)
    ray_union = (
        missing_x[None, :, :]
        | missing_y[:, None, :]
        | missing_z[:, :, None]
    )
    candidate_xyz = (~occupied_xyz & ray_union).detach()
    candidate_x = candidate_xyz.any(dim=0)
    candidate_y = candidate_xyz.any(dim=1)
    candidate_z = candidate_xyz.any(dim=2)
    footprint = build_registered_target_stack(
        (("x", candidate_x), ("y", candidate_y), ("z", candidate_z))
    ).detach()
    support = (
        footprint
        & ~target_binary
        & (detached_q > projection.new_tensor(-0.15))
    ).detach()
    flat_q = q.reshape(6, -1)
    flat_support = support.reshape(6, -1)
    losses: list[torch.Tensor] = []
    support_counts: list[int] = []
    selected_indices: list[list[int]] = []
    for view in range(6):
        indices = torch.nonzero(flat_support[view], as_tuple=False).flatten()
        count = int(indices.numel())
        support_counts.append(count)
        if count == 0:
            losses.append(projection.sum() * projection.new_tensor(0.0))
            selected_indices.append([])
            continue
        live_q = flat_q[view, indices]
        penalty = projection.new_tensor(0.02) * F.softplus(
            (live_q + projection.new_tensor(0.02)) / projection.new_tensor(0.02)
        )
        selected_count = math.ceil(0.05 * count)
        order = torch.argsort(penalty, descending=True, stable=True)
        chosen = order[:selected_count]
        losses.append(penalty[chosen].mean())
        selected_indices.append([int(item) for item in indices[chosen].detach().cpu().tolist()])
    result = torch.stack(losses).mean()
    if not torch.isfinite(result).item():
        raise ValueError("Leakage-Barrier loss is non-finite")
    details = {
        "candidate_xyz": candidate_xyz.contiguous().clone(),
        "support": support.contiguous().clone(),
        "support_counts": support_counts,
        "selected_indices": selected_indices,
    }
    return (result, details) if return_details else result


def _completed_update(completed_updates: object) -> int:
    if isinstance(completed_updates, bool):
        raise ValueError("completed_updates must be an integer in [0,3000)")
    try:
        value = operator.index(completed_updates)
    except TypeError as error:
        raise ValueError("completed_updates must be an integer in [0,3000)") from error
    if not 0 <= value < 3000:
        raise ValueError("completed_updates must be an integer in [0,3000)")
    return value


def v7_auxiliary_multiplier(completed_updates: int) -> float:
    """Return the next update's registered auxiliary ramp multiplier."""

    completed = _completed_update(completed_updates)
    if completed < 100:
        return 0.0
    if completed >= 299:
        return 1.0
    return (completed - 99) / 200.0


def v7_auxiliary_slot(completed_updates: int, profile: str) -> str:
    """Return ``hit``, ``barrier``, or Base-only ``base`` for the next update."""

    completed = _completed_update(completed_updates)
    if type(profile) is not str or profile not in _PROFILE_SLOTS:
        raise ValueError(f"unknown v7 profile: {profile!r}")
    slot = "hit" if completed % 4 in (0, 2) else "barrier"
    return slot if slot in _PROFILE_SLOTS[profile] else "base"


def v7_learning_rate(completed_updates: int) -> float:
    """Return the frozen learning rate for the next update."""

    completed = _completed_update(completed_updates)
    next_update = completed + 1
    if next_update <= 1500:
        return 0.007
    if next_update <= 2700:
        position = (next_update - 1501) / 1199.0
        return 0.0007 + 0.5 * (0.007 - 0.0007) * (1.0 + math.cos(math.pi * position))
    return 0.0007


def _validate_gradient(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be a float32 or float64 tensor")
    if value.numel() == 0 or not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be non-empty and finite")


def _finite_norm(name: str, value: torch.Tensor) -> float:
    maximum = float(value.detach().abs().amax().item())
    if maximum == 0.0:
        return 0.0
    scaled = float(torch.linalg.vector_norm(value / maximum).item())
    result = maximum * scaled
    if not math.isfinite(result):
        raise ValueError(f"{name} norm is not representable")
    return result


def _dot_with_unit(value: torch.Tensor, unit: torch.Tensor) -> float:
    maximum = float(value.detach().abs().amax().item())
    if maximum == 0.0:
        return 0.0
    result = maximum * float(torch.dot((value / maximum).reshape(-1), unit.reshape(-1)).item())
    if not math.isfinite(result):
        raise ValueError("gradient dot product is not representable")
    return result


def _real_scalar(name: str, value: object, *, minimum: float, maximum: float) -> float:
    if type(value) is not float:
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} is outside its registered range")
    return result


def _power_of_two_scale(value: torch.Tensor) -> float:
    maximum = float(value.abs().amax().item())
    if maximum == 0.0:
        return 1.0
    exponent = math.frexp(maximum)[1]
    return math.ldexp(1.0, -exponent)


def _direct_or_power_scaled_norm(name: str, value: torch.Tensor) -> float:
    direct = float(torch.linalg.vector_norm(value).item())
    if math.isfinite(direct):
        return direct
    scale = _power_of_two_scale(value)
    scaled = float(torch.linalg.vector_norm(value * scale).item())
    result = scaled / scale
    if not math.isfinite(result):
        raise ValueError(f"{name} norm is not representable")
    return result


def _direct_or_power_scaled_dot(left: torch.Tensor, right: torch.Tensor) -> float:
    direct = float(torch.dot(left.reshape(-1), right.reshape(-1)).item())
    if math.isfinite(direct):
        return direct
    left_scale = _power_of_two_scale(left)
    right_scale = _power_of_two_scale(right)
    scaled = float(
        torch.dot(
            (left * left_scale).reshape(-1),
            (right * right_scale).reshape(-1),
        ).item()
    )
    if not math.isfinite(scaled):
        raise ValueError("represented gradient dot is not resolvable")
    return scaled


def _represented_terminal_metrics(
    effective: torch.Tensor, base_grad: torch.Tensor
) -> tuple[float, float, float]:
    """Return direct promoted metrics, scaling by powers of two only on overflow."""

    work_effective = effective.to(dtype=torch.float64)
    work_base = base_grad.to(dtype=torch.float64)
    effective_norm = _direct_or_power_scaled_norm("effective auxiliary", work_effective)
    base_norm = _direct_or_power_scaled_norm("base_grad", work_base)
    direct_dot = _direct_or_power_scaled_dot(work_effective, work_base)
    if not all(math.isfinite(value) for value in (direct_dot, effective_norm, base_norm)):
        raise ValueError("represented gradient metrics are not finite")
    return direct_dot, effective_norm, base_norm


def _terminal_composed_gradient(
    base_grad: torch.Tensor,
    auxiliary: torch.Tensor,
    *,
    ratio: float,
    multiplier: float,
) -> tuple[torch.Tensor, float, float, bool]:
    """Revalidate the contribution after source-dtype addition cancellation."""

    result = base_grad.clone() + auxiliary
    if not torch.isfinite(result).all().item():
        raise ValueError("composed gradient is non-finite")
    # Validate the contribution actually represented by source-dtype addition.
    # Each repair moves offending coordinates exactly one ULP toward Base, so
    # its magnitude cannot grow. The loop is deliberately small and bounded for
    # native tensors; an unresolved ordinary rounding case falls back safely.
    for _ in range(8):
        effective = result - base_grad
        direct_dot, effective_norm, promoted_base_norm = _represented_terminal_metrics(
            effective, base_grad
        )
        cap_fraction = ratio * multiplier
        allowed_norm = promoted_base_norm * cap_fraction
        represented_ratio = (
            effective_norm / promoted_base_norm if promoted_base_norm > 0.0 else 0.0
        )
        if (
            direct_dot >= 0.0
            and effective_norm <= allowed_norm
            and represented_ratio <= cap_fraction
        ):
            return result.clone(), effective_norm, promoted_base_norm, False
        move = torch.zeros_like(effective, dtype=torch.bool)
        if direct_dot < 0.0 and promoted_base_norm > 0.0:
            negative = torch.logical_xor(torch.signbit(effective), torch.signbit(base_grad))
            move |= (effective != 0) & (base_grad != 0) & negative
        if effective_norm > allowed_norm:
            move |= effective != 0
        candidate = torch.where(move, torch.nextafter(result, base_grad), result)
        if torch.equal(candidate, result):
            break
        result = candidate
    # A finite ordinary input must never escape with an unsafe represented
    # contribution or fail merely because source rounding defeated repair.
    fallback = base_grad.clone()
    _, _, promoted_base_norm = _represented_terminal_metrics(
        torch.zeros_like(base_grad), base_grad
    )
    return fallback, 0.0, promoted_base_norm, True


def compose_v7_gradient(
    base_grad: torch.Tensor,
    auxiliary_grad: torch.Tensor | None,
    ratio: float,
    multiplier: float,
    *,
    return_telemetry: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    """Project one auxiliary against Base, cap it, and preserve Base bytes."""

    _validate_gradient("base_grad", base_grad)
    if type(ratio) is not float or ratio not in _VALID_RATIOS:
        raise ValueError("ratio must be exactly 0.10, 0.15, or 0.20")
    ratio_value = float(ratio)
    multiplier_value = _real_scalar("multiplier", multiplier, minimum=0.0, maximum=1.0)
    if type(return_telemetry) is not bool:
        raise ValueError("return_telemetry must be boolean")
    if multiplier_value == 0.0:
        if auxiliary_grad is not None:
            raise ValueError("off-slot auxiliary gradient was constructed")
        result = base_grad.clone()
        telemetry = {
            "ratio": ratio_value,
            "multiplier": multiplier_value,
            "base_norm": _finite_norm("base_grad", base_grad),
            "raw_auxiliary_norm": 0.0,
            "projected_auxiliary_norm": 0.0,
            "final_auxiliary_norm": 0.0,
            "final_auxiliary_base_ratio": 0.0,
            "conflict_projected": False,
            "cap_applied": False,
            "safe_zero_fallback": False,
        }
        return (result, telemetry) if return_telemetry else result
    if auxiliary_grad is None:
        raise ValueError("enabled auxiliary gradient is missing")
    _validate_gradient("auxiliary_grad", auxiliary_grad)
    if (
        auxiliary_grad.shape != base_grad.shape
        or auxiliary_grad.dtype != base_grad.dtype
        or auxiliary_grad.device != base_grad.device
    ):
        raise ValueError("auxiliary_grad is incompatible with base_grad")
    if torch._C._overlaps(base_grad, auxiliary_grad):
        raise ValueError("base_grad and auxiliary_grad must not alias or overlap")
    base_norm = _finite_norm("base_grad", base_grad)
    raw_norm = _finite_norm("auxiliary_grad", auxiliary_grad)
    projected = auxiliary_grad.clone()
    conflict_projected = False
    if base_norm == 0.0:
        projected.zero_()
    elif raw_norm > 0.0:
        base_unit = base_grad / base_norm
        dot_unit = _dot_with_unit(projected, base_unit)
        if dot_unit < 0.0:
            conflict_projected = True
            eps_scale = _EPS * (1.0 / base_norm) * (1.0 / base_norm)
            projected = projected - (dot_unit / (1.0 + eps_scale)) * base_unit
            residual = _dot_with_unit(projected, base_unit)
            if residual < 0.0:
                projected = projected - residual * base_unit
    projected_norm = _finite_norm("projected auxiliary", projected)
    allowed_norm = ratio_value * multiplier_value * base_norm
    cap_applied = projected_norm > allowed_norm
    if cap_applied and projected_norm > 0.0:
        scale = allowed_norm / projected_norm
        if base_grad.dtype == torch.float32:
            scale_tensor = torch.tensor(scale, dtype=base_grad.dtype, device=base_grad.device)
            scale = float(torch.nextafter(scale_tensor, torch.zeros_like(scale_tensor)).item())
        projected = projected * scale
    final_norm = _finite_norm("final auxiliary", projected)
    if base_norm > 0.0:
        base_unit = base_grad / base_norm
        for _ in range(4):
            residual = _dot_with_unit(projected, base_unit)
            if residual >= 0.0:
                break
            projected = projected - residual * base_unit
        final_norm = _finite_norm("final auxiliary", projected)
    if final_norm > allowed_norm and final_norm > 0.0:
        projected = projected * (allowed_norm / final_norm)
        final_norm = _finite_norm("final auxiliary", projected)
    result, final_norm, terminal_base_norm, safe_zero_fallback = _terminal_composed_gradient(
        base_grad,
        projected,
        ratio=ratio_value,
        multiplier=multiplier_value,
    )
    final_ratio = final_norm / terminal_base_norm if terminal_base_norm > 0.0 else 0.0
    telemetry = {
        "ratio": ratio_value,
        "multiplier": multiplier_value,
        "base_norm": terminal_base_norm,
        "raw_auxiliary_norm": raw_norm,
        "projected_auxiliary_norm": projected_norm,
        "final_auxiliary_norm": final_norm,
        "final_auxiliary_base_ratio": final_ratio,
        "conflict_projected": conflict_projected,
        "cap_applied": cap_applied,
        "safe_zero_fallback": safe_zero_fallback,
    }
    if not all(math.isfinite(value) for value in telemetry.values() if type(value) in (int, float)):
        raise ValueError("gradient telemetry is non-finite")
    return (result, telemetry) if return_telemetry else result


__all__ = [
    "build_registered_target_stack",
    "build_registered_visual_hull_logits",
    "compose_v7_gradient",
    "coverage_hit_loss",
    "dhw_to_xyz",
    "leakage_barrier_loss",
    "v7_auxiliary_multiplier",
    "v7_auxiliary_slot",
    "v7_learning_rate",
    "xyz_to_dhw",
]
