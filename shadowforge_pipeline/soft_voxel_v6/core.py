"""Frozen core math, views, initialization, and schedules for soft-loss v6."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from shadowforge_pipeline.soft_voxel_v3.lovasz import lovasz_hinge_per_view


TASK_ORDER = tuple(f"connected_{index:02d}" for index in range(1, 11))
PROFILE_ORDER = ("base", "j_only", "h_only", "full")
PROFILE_FLAGS: Mapping[str, Mapping[str, bool]] = MappingProxyType(
    {
        "base": MappingProxyType({"j": False, "h": False}),
        "j_only": MappingProxyType({"j": True, "h": False}),
        "h_only": MappingProxyType({"j": False, "h": True}),
        "full": MappingProxyType({"j": True, "h": True}),
    }
)

FIXED_CAMERA_AXES = ("y", "x", "z")
CANONICAL_VIEW_DESCRIPTORS = (
    ("y", "target_y", 0, 0.0, 0.0),
    ("x", "target_x", 1, 90.0, 0.0),
    ("z", "target_z", 2, 0.0, 90.0),
    ("mirror_y", "target_mirror_y", 3, 180.0, 0.0),
    ("mirror_x", "target_mirror_x", 4, 270.0, 0.0),
    ("mirror_z", "target_mirror_z", 5, 0.0, 270.0),
)

_CAMERA_DISTANCE = 1.7
_CAMERA_AZIMUTH_DEG = tuple(row[3] for row in CANONICAL_VIEW_DESCRIPTORS)
_CAMERA_ELEVATION_DEG = tuple(row[4] for row in CANONICAL_VIEW_DESCRIPTORS)
_INVERSE_D4_OP = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
_NATIVE_HULL_IDENTITY: Mapping[str, tuple[object, ...] | str] = MappingProxyType(
    {
        "canonical_first_three_labels": ("y", "x", "z"),
        "axis_for_view": ("z", "x", "y"),
        "d4_forward_ops": (3, 0, 7),
        "d4_inverse_ops": (1, 0, 7),
        "mirror_policy": "raw-horizontal-flip-then-independent-normalize",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fixed_camera_payload() -> dict[str, object]:
    """Serialize the six fixed labels, camera parameters, and canonical R/T bytes."""

    from pytorch3d.renderer import look_at_view_transform

    rotation, translation = look_at_view_transform(
        dist=_CAMERA_DISTANCE,
        elev=torch.tensor(_CAMERA_ELEVATION_DEG, dtype=torch.float32),
        azim=torch.tensor(_CAMERA_AZIMUTH_DEG, dtype=torch.float32),
        device=torch.device("cpu"),
    )
    rotation_bytes = rotation.detach().cpu().contiguous().numpy().tobytes(order="C")
    translation_bytes = translation.detach().cpu().contiguous().numpy().tobytes(order="C")
    metadata = {
        "schema": "pcraft_soft_voxel_v6_fixed_cameras_v1",
        "descriptors": [list(row) for row in CANONICAL_VIEW_DESCRIPTORS],
        "distance": _CAMERA_DISTANCE,
        "rotation_dtype": "float32",
        "rotation_shape": [6, 3, 3],
        "translation_dtype": "float32",
        "translation_shape": [6, 3],
        "native_hull_identity": {
            "canonical_first_three_labels": list(
                _NATIVE_HULL_IDENTITY["canonical_first_three_labels"]
            ),
            "axis_for_view": list(_NATIVE_HULL_IDENTITY["axis_for_view"]),
            "d4_forward_ops": list(_NATIVE_HULL_IDENTITY["d4_forward_ops"]),
            "d4_inverse_ops": list(_NATIVE_HULL_IDENTITY["d4_inverse_ops"]),
            "mirror_policy": _NATIVE_HULL_IDENTITY["mirror_policy"],
        },
    }
    metadata_bytes = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    payload_bytes = metadata_bytes + b"\0" + rotation_bytes + b"\0" + translation_bytes
    return {
        "metadata": metadata,
        "rotation_bytes": rotation_bytes,
        "translation_bytes": translation_bytes,
        "rotation_sha256": _sha256_bytes(rotation_bytes),
        "translation_sha256": _sha256_bytes(translation_bytes),
        "payload_sha256": _sha256_bytes(payload_bytes),
    }


def _parse_suite_views(
    suite_views: Sequence[tuple[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(suite_views, (str, bytes)):
        raise ValueError("suite_views must contain labelled x,y,z tensors")
    try:
        rows = tuple(suite_views)
    except TypeError as error:
        raise ValueError("suite_views must contain labelled x,y,z tensors") from error
    if len(rows) != 3:
        raise ValueError("suite_views must contain exactly x,y,z")
    parsed: list[torch.Tensor] = []
    for required_axis, row in zip(("x", "y", "z"), rows):
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("each suite view must be an (axis, tensor) pair")
        axis, mask = row
        if axis != required_axis:
            raise ValueError("suite view axes must be unique and ordered x,y,z")
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError("suite masks must be boolean torch.Tensor values")
        if mask.ndim != 2 or any(size == 0 for size in mask.shape):
            raise ValueError("suite masks must have non-empty [H,W] shape")
        parsed.append(mask)
    reference = parsed[0]
    if any(mask.shape != reference.shape for mask in parsed[1:]):
        raise ValueError("suite masks must have equal shapes")
    if any(mask.device != reference.device for mask in parsed[1:]):
        raise ValueError("suite masks must share a device")
    return parsed[0], parsed[1], parsed[2]


def map_suite_views(
    suite_views: Sequence[tuple[str, torch.Tensor]],
    mirrored_suite_views: Sequence[tuple[str, torch.Tensor]],
) -> torch.Tensor:
    """Map independently normalized base and raw-flip ``x,y,z`` masks."""

    target_x, target_y, target_z = _parse_suite_views(suite_views)
    mirror_x, mirror_y, mirror_z = _parse_suite_views(mirrored_suite_views)
    if any(
        mask.shape != target_x.shape
        for mask in (mirror_x, mirror_y, mirror_z)
    ):
        raise ValueError("base and mirrored suite masks must have equal shapes")
    if any(
        mask.device != target_x.device
        for mask in (mirror_x, mirror_y, mirror_z)
    ):
        raise ValueError("base and mirrored suite masks must share a device")
    canonical = (target_y, target_x, target_z)
    mirrors = (mirror_y, mirror_x, mirror_z)
    return torch.stack(canonical + mirrors).clone()


def _apply_d4(mask: torch.Tensor, op_id: int) -> torch.Tensor:
    rotated = torch.rot90(mask, int(op_id) % 4, dims=(-2, -1))
    return torch.flip(rotated, dims=(-1,)) if int(op_id) >= 4 else rotated


def build_fixed_visual_hull_logits(
    suite_views: Sequence[tuple[str, torch.Tensor]],
    *,
    resolution: int = 256,
    allow_test_size: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build exact ``+2/-3`` logits from the frozen native fixed-camera hull."""

    if type(resolution) is not int or resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    if type(allow_test_size) is not bool:
        raise ValueError("allow_test_size must be boolean")
    if not allow_test_size and resolution != 256:
        raise ValueError("production visual-hull resolution must be exactly 256")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    if not allow_test_size and dtype != torch.float32:
        raise ValueError("production visual-hull dtype must be torch.float32")

    target_x, target_y, target_z = _parse_suite_views(suite_views)
    if target_x.shape != (resolution, resolution):
        raise ValueError("suite mask shape must equal the requested square resolution")
    try:
        output_device = target_x.device if device is None else torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device must identify a torch device") from error

    canonical = (target_y, target_x, target_z)
    planes: dict[str, torch.Tensor] = {}
    for view_index, (axis_name, forward_op) in enumerate(
        zip(
            _NATIVE_HULL_IDENTITY["axis_for_view"],
            _NATIVE_HULL_IDENTITY["d4_forward_ops"],
        )
    ):
        inverse_op = _INVERSE_D4_OP[int(forward_op)]
        planes[axis_name] = _apply_d4(canonical[view_index], inverse_op)
    hull = (
        planes["x"][None, :, :]
        & planes["y"][:, None, :]
        & planes["z"][:, :, None]
    ).to(device=output_device)
    logits = torch.where(
        hull,
        torch.full((), 2.0, device=output_device, dtype=dtype),
        torch.full((), -3.0, device=output_device, dtype=dtype),
    )
    return logits.reshape(1, 1, resolution, resolution, resolution).contiguous().clone()


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_score_target(score: torch.Tensor, target_binary: torch.Tensor) -> None:
    if not isinstance(score, torch.Tensor) or not score.is_floating_point():
        raise ValueError("score must be a floating-point torch.Tensor")
    if not isinstance(target_binary, torch.Tensor) or target_binary.dtype != torch.bool:
        raise ValueError("target_binary must be a boolean torch.Tensor")
    if score.ndim != 3 or score.shape != target_binary.shape or any(size == 0 for size in score.shape):
        raise ValueError("score and target_binary must have equal non-empty [views,H,W] shape")
    if score.device != target_binary.device:
        raise ValueError("score and target_binary must share a device")
    if not torch.isfinite(score).all().item():
        raise ValueError("score must be finite")


def delivery_lovasz_loss(score: torch.Tensor, target_binary: torch.Tensor) -> torch.Tensor:
    """Exact v5 delivery Lovasz/Jaccard loss."""

    _validate_score_target(score, target_binary)
    result = lovasz_hinge_per_view(score / score.new_tensor(0.05), target_binary).mean()
    if not torch.isfinite(result).item():
        raise ValueError("Jaccard loss is non-finite")
    return result


def background_hard_negative_tail_loss(
    score: torch.Tensor,
    target_binary: torch.Tensor,
    *,
    fraction: float = 0.05,
    margin: float = 0.01,
    temperature: float = 0.02,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, object]]]:
    """Exact v5 stable row-major background top-tail loss."""

    _validate_score_target(score, target_binary)
    fraction_value = _finite_number("fraction", fraction)
    margin_value = _finite_number("margin", margin)
    temperature_value = _finite_number("temperature", temperature)
    if not 0.0 < fraction_value <= 1.0 or temperature_value <= 0.0:
        raise ValueError("fraction must be in (0,1] and temperature must be positive")
    if type(return_details) is not bool:
        raise ValueError("return_details must be boolean")
    losses: list[torch.Tensor] = []
    details: list[dict[str, object]] = []
    flat_score = score.reshape(score.shape[0], -1)
    flat_target = target_binary.reshape(target_binary.shape[0], -1)
    for view_index in range(score.shape[0]):
        background_indices = torch.nonzero(~flat_target[view_index], as_tuple=False).flatten()
        count = int(background_indices.numel())
        if count == 0:
            raise ValueError("each view must contain at least one background pixel")
        background_scores = flat_score[view_index, background_indices]
        penalty = score.new_tensor(temperature_value) * F.softplus(
            (background_scores + score.new_tensor(margin_value))
            / score.new_tensor(temperature_value)
        )
        k = max(1, math.ceil(fraction_value * count))
        order = torch.argsort(penalty, descending=True, stable=True)
        chosen_order = order[:k]
        selected = penalty[chosen_order]
        selected_scores = background_scores[chosen_order]
        selected_indices = background_indices[chosen_order]
        losses.append(selected.mean())
        details.append(
            {
                "view": view_index,
                "background_count": count,
                "selected_count": k,
                "selected_indices": [
                    int(value) for value in selected_indices.detach().cpu().tolist()
                ],
                "score_min": float(selected_scores.detach().min().item()),
                "score_mean": float(selected_scores.detach().mean().item()),
                "score_max": float(selected_scores.detach().max().item()),
            }
        )
    result = torch.stack(losses).mean()
    if not torch.isfinite(result).item():
        raise ValueError("FP-tail loss is non-finite")
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


def v6_auxiliary_multiplier(completed_updates: int) -> float:
    """Return the next update's auxiliary multiplier from completed updates."""

    completed = _completed_update(completed_updates)
    if completed < 600:
        return 0.0
    if completed >= 899:
        return 1.0
    return (completed - 599) / 300.0


def v6_component_budgets(completed_updates: int, profile: str) -> Mapping[str, float]:
    """Return independent J/H base-gradient budgets without reallocation."""

    if profile not in PROFILE_FLAGS:
        raise ValueError(f"unknown v6 profile: {profile!r}")
    multiplier = v6_auxiliary_multiplier(completed_updates)
    flags = PROFILE_FLAGS[profile]
    return MappingProxyType(
        {
            "j": multiplier * 0.20 if flags["j"] else 0.0,
            "h": multiplier * 0.10 if flags["h"] else 0.0,
        }
    )


def v6_learning_rate(completed_updates: int) -> float:
    """Return the learning rate for the next update under the frozen schedule."""

    completed = _completed_update(completed_updates)
    next_update = completed + 1
    if next_update <= 1500:
        return 0.007
    if next_update <= 2700:
        position = (next_update - 1501) / 1199.0
        return 0.0007 + 0.5 * (0.007 - 0.0007) * (
            1.0 + math.cos(math.pi * position)
        )
    return 0.0007


def _validate_gradient(name: str, value: torch.Tensor, base: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} gradient must be a floating-point tensor")
    if value.shape != base.shape or value.device != base.device or value.dtype != base.dtype:
        raise ValueError(f"{name} gradient is incompatible with base gradient")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} gradient is non-finite")


def _normalized_component(
    gradient: torch.Tensor,
    raw_norm: torch.Tensor,
    base_norm: torch.Tensor,
    budget: float,
) -> torch.Tensor:
    budget_tensor = gradient.new_tensor(budget)
    if gradient.dtype == torch.float32 and budget > 0.0:
        budget_tensor = torch.nextafter(budget_tensor, gradient.new_tensor(0.0))
    contribution = gradient * (base_norm * budget_tensor / raw_norm)
    upper = torch.nextafter(gradient.new_tensor(budget), gradient.new_tensor(math.inf))
    for _ in range(4):
        ratio = torch.linalg.vector_norm(contribution) / base_norm
        if float(ratio.item()) <= float(upper.item()):
            return contribution
        correction = torch.nextafter(budget_tensor / ratio, gradient.new_tensor(0.0))
        contribution = contribution * correction
    raise ValueError("normalized auxiliary gradient exceeds its budget")


def compose_v6_gradient(
    base_grad: torch.Tensor,
    j_grad: torch.Tensor | None,
    h_grad: torch.Tensor | None,
    *,
    completed_updates: int,
    profile: str,
    return_telemetry: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    """Normalize enabled J/H independently and add one combined gradient."""

    if profile not in PROFILE_FLAGS:
        raise ValueError(f"unknown v6 profile: {profile!r}")
    if type(return_telemetry) is not bool:
        raise ValueError("return_telemetry must be boolean")
    if not isinstance(base_grad, torch.Tensor) or not base_grad.is_floating_point():
        raise ValueError("base gradient must be a floating-point tensor")
    if not torch.isfinite(base_grad).all().item():
        raise ValueError("base gradient is non-finite")
    base_norm = torch.linalg.vector_norm(base_grad)
    base_norm_value = float(base_norm.item())
    if not math.isfinite(base_norm_value) or base_norm_value <= 1e-12:
        raise ValueError("base gradient norm must be finite and exceed 1e-12")

    multiplier = v6_auxiliary_multiplier(completed_updates)
    flags = PROFILE_FLAGS[profile]
    if multiplier == 0.0 and (j_grad is not None or h_grad is not None):
        raise ValueError("off-schedule auxiliary gradient was constructed")
    for name, enabled, gradient in (
        ("j", flags["j"], j_grad),
        ("h", flags["h"], h_grad),
    ):
        if not enabled and gradient is not None:
            raise ValueError(f"disabled {name} gradient was constructed")

    telemetry: dict[str, object] = {
        "multiplier": multiplier,
        "base_norm": base_norm_value,
        "j_budget": multiplier * 0.20 if flags["j"] else 0.0,
        "h_budget": multiplier * 0.10 if flags["h"] else 0.0,
        "j_raw_norm": 0.0,
        "h_raw_norm": 0.0,
        "j_ratio": 0.0,
        "h_ratio": 0.0,
        "combined_ratio": 0.0,
        "j_zero": False,
        "h_zero": False,
    }
    result = base_grad.clone()
    contributions: list[torch.Tensor] = []
    if multiplier > 0.0:
        for name, enabled, gradient, maximum in (
            ("j", flags["j"], j_grad, 0.20),
            ("h", flags["h"], h_grad, 0.10),
        ):
            if not enabled:
                continue
            if gradient is None:
                raise ValueError(f"enabled {name} gradient is missing")
            _validate_gradient(name, gradient, base_grad)
            raw_norm = torch.linalg.vector_norm(gradient)
            raw_value = float(raw_norm.item())
            if not math.isfinite(raw_value):
                raise ValueError(f"{name} gradient norm is non-finite")
            telemetry[f"{name}_raw_norm"] = raw_value
            if raw_value <= 1e-12:
                telemetry[f"{name}_zero"] = True
                continue
            contribution = _normalized_component(
                gradient, raw_norm, base_norm, multiplier * maximum
            )
            contributions.append(contribution)
            result = result + contribution
            telemetry[f"{name}_ratio"] = float(
                (torch.linalg.vector_norm(contribution) / base_norm).item()
            )
    if contributions:
        combined = contributions[0].clone()
        for contribution in contributions[1:]:
            combined = combined + contribution
        combined_ratio = float((torch.linalg.vector_norm(combined) / base_norm).item())
        telemetry["combined_ratio"] = combined_ratio
        bound = base_grad.new_tensor(multiplier * 0.30)
        upper = float(torch.nextafter(bound, base_grad.new_tensor(math.inf)).item())
        if combined_ratio > upper:
            raise ValueError("combined auxiliary gradient exceeds the 0.30 budget")
    if not torch.isfinite(result).all().item():
        raise ValueError("composed gradient is non-finite")
    return (result, telemetry) if return_telemetry else result


__all__ = [
    "CANONICAL_VIEW_DESCRIPTORS",
    "FIXED_CAMERA_AXES",
    "PROFILE_FLAGS",
    "PROFILE_ORDER",
    "TASK_ORDER",
    "background_hard_negative_tail_loss",
    "build_fixed_visual_hull_logits",
    "compose_v6_gradient",
    "delivery_lovasz_loss",
    "fixed_camera_payload",
    "map_suite_views",
    "v6_auxiliary_multiplier",
    "v6_component_budgets",
    "v6_learning_rate",
]
