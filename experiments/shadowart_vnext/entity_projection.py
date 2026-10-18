"""Threshold-consistent continuous entity surrogate on complete positive rays.

max(p on ray)>=.5 iff that ray intersects an occupied p>=.5 cell.
This is a piecewise differentiable occupancy surrogate, NOT Beer-Lambert
opacity, a binary derivative, a thickness guarantee or a mesh-quality claim.
"""
import torch

from .rendering import _axis_map, _key, _sizes, _volume_resolution, iter_ray_segments


def _chunk_values(probability, chunk):
    indices = torch.tensor(chunk.voxel_indices, device=probability.device)
    rays = torch.tensor(chunk.ray_indices, device=probability.device, dtype=torch.long)
    values = probability.reshape(-1)[indices]
    maxima = probability.new_zeros(chunk.stop - chunk.start).scatter_reduce(
        0, rays, values, reduce='amax', include_self=False)
    return indices, rays, values, maxima


class _ObliqueMax(torch.autograd.Function):
    """Recompute streamed ray maxima in backward; retain only the native field."""
    @staticmethod
    def forward(ctx, probability, camera, image_size, extent):
        ctx.save_for_backward(probability)
        ctx.camera, ctx.image_size, ctx.extent = camera, image_size, extent
        parts = []
        for chunk in iter_ray_segments(camera, image_size, probability.shape[0], extent):
            parts.append(_chunk_values(probability, chunk)[-1])
        return torch.cat(parts)

    @staticmethod
    def backward(ctx, grad_output):
        probability, = ctx.saved_tensors
        gradient = probability.new_zeros(probability.numel())
        for chunk in iter_ray_segments(ctx.camera, ctx.image_size, probability.shape[0], ctx.extent):
            indices, rays, values, maxima = _chunk_values(probability, chunk)
            winners = (values == maxima[rays]).to(values.dtype)
            ties = maxima.new_zeros(maxima.shape).index_add_(0, rays, winners)
            weights = winners / ties[rays].clamp_min(1.)
            gradient.index_add_(0, indices, weights * grad_output[chunk.start:chunk.stop][rays])
        return gradient.reshape_as(probability), None, None, None


def render_probability_max(probability_xyz, cameras, image_size, extent=1.7):
    """Complete-ray max; caller supplies finite floating probabilities in [0,1].

    Empty rays have probability zero. Positive-length traversal and half-open
    parallel-cell ownership are exactly those of the hard occupancy renderer.
    Tied maxima share subgradients; there is no optical straight-through term.
    """
    if not isinstance(probability_xyz, torch.Tensor) or not probability_xyz.is_floating_point():
        raise ValueError('probability must be a floating torch tensor')
    resolution = _volume_resolution(probability_xyz)
    _sizes(image_size, resolution, extent)
    views = []
    for camera in cameras:
        mapping = _axis_map(_key(camera), image_size, resolution, extent)
        if mapping is None:
            values = _ObliqueMax.apply(probability_xyz, camera, image_size, extent)
        else:
            axis, indices, valid = mapping
            index = torch.tensor(indices, device=probability_xyz.device)
            visible = torch.tensor(valid, device=probability_xyz.device, dtype=probability_xyz.dtype)
            values = probability_xyz.amax(dim=axis).reshape(-1)[index] * visible
        views.append(values.reshape(image_size, image_size))
    return torch.stack(views) if views else probability_xyz.new_empty((0, image_size, image_size))


def entity_projection_metrics(fields, contract, targets):
    """Replay the actual max-forward at its training size and audit hard parity."""
    import numpy as np
    from .rendering import render_binary
    from .metrics import soft_projection_metrics
    size = fields['metadata']['training_projection']['image_size']
    with torch.no_grad():
        p = torch.from_numpy(np.array(fields['guidance'], copy=True))
        predicted = render_probability_max(p, contract.cameras, size, contract.extent).numpy()
    hard = render_binary(fields['occupancy'], contract.cameras, size, contract.extent)
    report = soft_projection_metrics(predicted, targets.render(contract.cameras, size, contract.layout))
    report.update(stage='entity_projection', projection_rule='complete_ray_max_cell_probability',
                  is_optical_opacity=False,
                  threshold_rule='max_cell_probability_on_positive_ray>=0.5',
                  threshold_matches_hard_occupancy=bool(np.array_equal(predicted >= .5, hard)))
    if not report['threshold_matches_hard_occupancy']:
        raise ValueError('FAILED_ENTITY_HARD_PROJECTION_PARITY')
    return report
