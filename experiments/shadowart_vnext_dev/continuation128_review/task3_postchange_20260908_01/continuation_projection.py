"""Length-weighted full-ray probability continuation.

Finite exponents use a generalized mean over every positive-length segment
returned by :func:`rendering.iter_ray_segments`.  This is an optimization
surrogate, not optical opacity, and finite exponents are not threshold
equivalent to binary occupancy.  The schedule's final phase is a separate
exact ray maximum rather than a serialized floating-point infinity.
"""
import copy
import math

import torch

from .rendering import _axis_map, _key, _sizes, _volume_resolution, iter_ray_segments


CONTINUATION_SCHEDULE = {
    "version": "length_weighted_probability_power_mean_to_exact_max_v1",
    "progress_rule": "(step-1)/max(steps-1,1)",
    "generalized_mean_progress_lt": 0.8,
    "exponent_rule": "2**(5*progress/0.8)",
    "exponent_bounds": [1.0, 32.0],
    "exact_max_progress_gte": 0.8,
    "single_step_phase": "exact_max",
}


def continuation_schedule():
    """Return an owned, JSON-serializable declaration of the fixed schedule."""
    return copy.deepcopy(CONTINUATION_SCHEDULE)


def continuation_state(step, steps):
    """Return the phase and finite exponent for a validated training step."""
    if (type(step) is not int or type(steps) is not int or steps < 1
            or not 1 <= step <= steps):
        raise ValueError("step and steps must be integers with 1 <= step <= steps")
    progress = (step - 1) / max(steps - 1, 1)
    if steps == 1 or progress >= 0.8:
        return {"progress": progress, "phase": "exact_max", "exponent": None}
    exponent = 2.0 ** (5.0 * progress / 0.8)
    return {"progress": progress, "phase": "generalized_mean", "exponent": exponent}


def _validated_exponent(exponent):
    if (isinstance(exponent, bool) or not isinstance(exponent, (int, float))
            or not math.isfinite(exponent) or not 1.0 <= exponent <= 32.0):
        raise ValueError("exponent must be a finite scalar in [1,32]")
    return float(exponent)


def _axis_mean(probability, axis, exponent):
    maximum = probability.amax(dim=axis).detach()
    if exponent == 1.0:
        # Correct only possible floating reduction overshoot; the custom
        # backward below remains the exact linear arithmetic-mean derivative.
        return torch.minimum(probability.mean(dim=axis), maximum)
    safe = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    scaled = probability / safe.unsqueeze(axis)
    powered = scaled.pow(exponent).mean(dim=axis)
    result = maximum * powered.pow(1.0 / exponent)
    return torch.where(maximum > 0, torch.minimum(result, maximum), torch.zeros_like(result))


class _AxisGeneralizedMean(torch.autograd.Function):
    """Equal-length axial reduction with an explicit zero-ray subgradient."""
    @staticmethod
    def forward(ctx, probability, axis, exponent):
        ctx.save_for_backward(probability)
        ctx.axis, ctx.exponent = axis, exponent
        return _axis_mean(probability, axis, exponent)

    @staticmethod
    def backward(ctx, grad_output):
        probability, = ctx.saved_tensors
        count = probability.shape[ctx.axis]
        if ctx.exponent == 1.0:
            derivative = probability.new_full(probability.shape, 1.0 / count)
        else:
            mean = _axis_mean(probability, ctx.axis, ctx.exponent)
            safe = torch.where(mean > 0, mean, torch.ones_like(mean))
            derivative = (probability / safe.unsqueeze(ctx.axis)).pow(ctx.exponent - 1.0) / count
            derivative = torch.where(mean.unsqueeze(ctx.axis) > 0,
                                     derivative, torch.zeros_like(derivative))
        return derivative * grad_output.unsqueeze(ctx.axis), None, None


def _chunk_mean(probability, chunk, exponent):
    count = chunk.stop - chunk.start
    indices = torch.tensor(chunk.voxel_indices, device=probability.device)
    rays = torch.tensor(chunk.ray_indices, device=probability.device, dtype=torch.long)
    lengths = torch.tensor(chunk.lengths, device=probability.device, dtype=probability.dtype)
    values = probability.reshape(-1)[indices]
    totals = probability.new_zeros(count).index_add_(0, rays, lengths)
    weights = lengths / totals[rays]
    if exponent == 1.0:
        # Accumulate numerator and denominator in the same order.  In
        # particular, a float32 constant-one ray is exactly totals/totals,
        # rather than a normalized-weight sum that can round above one.
        numerator = probability.new_zeros(count).index_add_(0, rays, lengths * values)
        mean = torch.where(totals > 0, numerator / torch.where(
            totals > 0, totals, torch.ones_like(totals)), torch.zeros_like(totals))
        maximum = probability.new_zeros(count).scatter_reduce_(
            0, rays, values, reduce="amax", include_self=True).detach()
        mean = torch.minimum(mean, maximum)
    else:
        maximum = probability.new_zeros(count).scatter_reduce_(
            0, rays, values, reduce="amax", include_self=True).detach()
        safe = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
        numerator = probability.new_zeros(count).index_add_(
            0, rays, lengths * (values / safe[rays]).pow(exponent))
        powers = torch.where(totals > 0, numerator / torch.where(
            totals > 0, totals, torch.ones_like(totals)), torch.zeros_like(totals))
        result = maximum * powers.pow(1.0 / exponent)
        mean = torch.where(maximum > 0, torch.minimum(result, maximum), torch.zeros_like(result))
    return indices, rays, weights, values, mean


class _ObliqueGeneralizedMean(torch.autograd.Function):
    """Recompute bounded ray chunks in backward; retain no segment tensors."""
    @staticmethod
    def forward(ctx, probability, camera, image_size, extent, exponent):
        ctx.save_for_backward(probability)
        ctx.camera, ctx.image_size = camera, image_size
        ctx.extent, ctx.exponent = extent, exponent
        parts = [_chunk_mean(probability, chunk, exponent)[-1]
                 for chunk in iter_ray_segments(camera, image_size, probability.shape[0], extent)]
        return torch.cat(parts)

    @staticmethod
    def backward(ctx, grad_output):
        probability, = ctx.saved_tensors
        gradient = probability.new_zeros(probability.numel())
        for chunk in iter_ray_segments(ctx.camera, ctx.image_size,
                                       probability.shape[0], ctx.extent):
            indices, rays, weights, values, mean = _chunk_mean(
                probability, chunk, ctx.exponent)
            if ctx.exponent == 1.0:
                derivative = weights
            else:
                safe = torch.where(mean > 0, mean, torch.ones_like(mean))
                derivative = weights * (values / safe[rays]).pow(ctx.exponent - 1.0)
                derivative = torch.where(mean[rays] > 0,
                                         derivative, torch.zeros_like(derivative))
            gradient.index_add_(0, indices,
                                derivative * grad_output[chunk.start:chunk.stop][rays])
        return gradient.reshape_as(probability), None, None, None, None


def render_probability_generalized_mean(probability_xyz, cameras, image_size,
                                        *, exponent, extent=1.7):
    """Render the complete positive-ray generalized mean for finite ``q``.

    Segment lengths are normalized independently on each ray.  ``q=1`` is the
    exact weighted arithmetic mean, including its gradient at zero.  For
    ``q>1``, all-zero rays use the zero subgradient.  The detached ray maximum
    rescales powers for numerical stability; probabilities are never clipped
    and no nonzero floor is introduced.
    """
    exponent = _validated_exponent(exponent)
    if not isinstance(probability_xyz, torch.Tensor) or not probability_xyz.is_floating_point():
        raise ValueError("probability must be a floating torch tensor cube")
    try:
        resolution = _volume_resolution(probability_xyz)
    except ValueError as error:
        raise ValueError("probability must be a floating torch tensor cube") from error
    _sizes(image_size, resolution, extent)
    views = []
    for camera in cameras:
        mapping = _axis_map(_key(camera), image_size, resolution, extent)
        if mapping is None:
            values = _ObliqueGeneralizedMean.apply(
                probability_xyz, camera, image_size, extent, exponent)
        else:
            axis, indices, valid = mapping
            lines = _AxisGeneralizedMean.apply(probability_xyz, axis, exponent)
            index = torch.tensor(indices, device=probability_xyz.device)
            visible = torch.tensor(valid, device=probability_xyz.device,
                                   dtype=probability_xyz.dtype)
            values = lines.reshape(-1)[index] * visible
        views.append(values.reshape(image_size, image_size))
    return torch.stack(views) if views else probability_xyz.new_empty((0, image_size, image_size))


def render_probability_continuation(probability_xyz, cameras, image_size, *,
                                    step, steps, extent=1.7):
    """Render the scheduled finite generalized mean or the separate exact max."""
    state = continuation_state(step, steps)
    if state["phase"] == "exact_max":
        from .entity_projection import render_probability_max
        return render_probability_max(probability_xyz, cameras, image_size, extent=extent)
    return render_probability_generalized_mean(
        probability_xyz, cameras, image_size, exponent=state["exponent"], extent=extent)
