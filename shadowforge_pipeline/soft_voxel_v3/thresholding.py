"""Exact hard-delivery thresholding and differentiable readout for v3."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import torch

from .protocol import SadekarTFSProtocol


_OCCUPANCY = 0.999
_READOUT_FRACTION = 0.8
_MAX_EPS = 1e-8


@dataclass(frozen=True)
class DeliveryReadout:
    """The hard occupancy and frozen formal delivery values for one threshold."""

    occupancy: torch.Tensor
    projection: torch.Tensor
    stopped_max: torch.Tensor
    score: torch.Tensor
    mask: torch.Tensor


class _DeliveryThresholdSTE(torch.autograd.Function):
    """Hard ``>= tau`` occupancy with a sigmoid derivative in backpropagation."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        density: torch.Tensor,
        tau: float,
        temperature: float,
    ) -> torch.Tensor:
        q = torch.sigmoid((density - float(tau)) / float(temperature))
        ctx.save_for_backward(q)
        ctx.temperature = float(temperature)
        return (density >= float(tau)).to(density.dtype) * density.new_tensor(_OCCUPANCY)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        (q,) = ctx.saved_tensors
        grad_density = grad_output * q * (1.0 - q) / ctx.temperature
        return grad_density, None, None


def _finite_scalar(name: str, value: float) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validate_density(density: torch.Tensor, tau: float, temperature: float) -> tuple[float, float]:
    if not isinstance(density, torch.Tensor):
        raise ValueError("density must be a torch.Tensor")
    if not density.is_floating_point():
        raise ValueError("density must use a floating-point dtype")
    if not torch.isfinite(density).all().item():
        raise ValueError("density must be finite")
    threshold = _finite_scalar("tau", tau)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    surrogate_temperature = _finite_scalar("temperature", temperature)
    if surrogate_temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return threshold, surrogate_temperature


def _validate_render_density_shape(density: torch.Tensor) -> None:
    """Preflight the exact ``SadekarTFSProtocol.render`` volume contract."""

    if density.ndim != 5 or density.shape[:2] != (1, 1):
        raise ValueError("density must have shape [1,1,D,H,W]")
    resolution = int(density.shape[-1])
    if resolution == 0 or density.shape[-3:] != (resolution, resolution, resolution):
        raise ValueError("density must be a non-empty cubic volume")


def _validate_projection(projection: torch.Tensor) -> None:
    if not isinstance(projection, torch.Tensor):
        raise ValueError("projection must be a torch.Tensor")
    if not projection.is_floating_point():
        raise ValueError("projection must use a floating-point dtype")
    if projection.ndim not in (2, 3):
        raise ValueError("projection must have shape [H,W] or [views,H,W]")
    if any(size == 0 for size in projection.shape):
        raise ValueError("projection dimensions must be non-empty")
    if not torch.isfinite(projection).all().item():
        raise ValueError("projection must be finite")


def _stopped_maximum(projection: torch.Tensor, max_eps: float) -> torch.Tensor:
    if projection.ndim == 2:
        return projection.detach().amax().clamp_min(max_eps)
    return projection.detach().amax(dim=(-2, -1), keepdim=True).clamp_min(max_eps)


def straight_through_density(
    density: torch.Tensor, tau: float, temperature: float = 0.01
) -> torch.Tensor:
    """Return exact ``0.999 * (density >= tau)`` with sigmoid STE gradients."""

    threshold, surrogate_temperature = _validate_density(density, tau, temperature)
    return _DeliveryThresholdSTE.apply(density, threshold, surrogate_temperature)


def rendered_margin(
    projection: torch.Tensor, readout_fraction: float = _READOUT_FRACTION, max_eps: float = _MAX_EPS
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached clamped maxima and the signed formal delivery margin."""

    _validate_projection(projection)
    fraction = _finite_scalar("readout_fraction", readout_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("readout_fraction must be in [0, 1]")
    clamp = _finite_scalar("max_eps", max_eps)
    if clamp <= 0.0:
        raise ValueError("max_eps must be positive")
    stopped_max = _stopped_maximum(projection, clamp)
    return stopped_max, projection - stopped_max * fraction


def formal_readout_mask(projection: torch.Tensor) -> torch.Tensor:
    """Apply the frozen inclusive 0.8-times-stopped-maximum delivery rule."""

    _, score = rendered_margin(projection)
    return score >= 0.0


def _device_from_state(state: object) -> torch.device | None:
    """Extract a concrete device from renderer/camera tensor state when present."""

    if isinstance(state, torch.Tensor):
        return state.device
    candidate = getattr(state, "device", None)
    if candidate is not None:
        try:
            return torch.device(candidate)
        except (TypeError, RuntimeError):
            pass
    for attribute in ("R", "T"):
        tensor = getattr(state, attribute, None)
        if tensor is not None:
            device = _device_from_state(tensor)
            if device is not None:
                return device
    return None


def _canonical_device(device: torch.device) -> torch.device:
    """Resolve generic CUDA to the concrete current CUDA device."""

    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _protocol_device(protocol: SadekarTFSProtocol) -> torch.device:
    """Prefer the renderer/camera device that will actually service rendering."""

    cameras = getattr(protocol, "cameras", None)
    if cameras is not None:
        for attribute in ("R", "T"):
            device = _device_from_state(getattr(cameras, attribute, None))
            if device is not None:
                return _canonical_device(device)
        device = _device_from_state(cameras)
        if device is not None:
            return _canonical_device(device)
    renderer = getattr(protocol, "renderer", None)
    if renderer is not None:
        device = _device_from_state(renderer)
        if device is not None:
            return _canonical_device(device)
    try:
        return _canonical_device(torch.device(protocol.device))
    except (AttributeError, TypeError, RuntimeError) as error:
        raise ValueError("protocol must expose a valid device") from error


def _protocol_matches_density_device(protocol: SadekarTFSProtocol, density: torch.Tensor) -> bool:
    return _protocol_device(protocol) == density.device


def render_thresholded(
    protocol: SadekarTFSProtocol, density: torch.Tensor, tau: float
) -> DeliveryReadout:
    """Threshold once, render all registered views, then apply the formal readout."""

    _validate_density(density, tau, 0.01)
    _validate_render_density_shape(density)
    if not _protocol_matches_density_device(protocol, density):
        raise ValueError("protocol and density must be on the same device")
    occupancy = straight_through_density(density, tau)
    projection = protocol.render_all(occupancy)
    stopped_max, score = rendered_margin(projection)
    return DeliveryReadout(
        occupancy=occupancy,
        projection=projection,
        stopped_max=stopped_max,
        score=score,
        mask=score >= 0.0,
    )


def dts_endpoint(global_step: int) -> float:
    """Return the pre-registered threshold endpoint for an optimizer step."""

    if isinstance(global_step, bool):
        raise ValueError("global_step must be a nonnegative integer")
    try:
        step = operator.index(global_step)
    except TypeError as error:
        raise ValueError("global_step must be a nonnegative integer") from error
    if step < 0:
        raise ValueError("global_step must be nonnegative")
    return 0.04 if step % 2 == 0 else 0.06
