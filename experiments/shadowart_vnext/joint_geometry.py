"""Opt-in connected replacement transaction driven by exact hard shadow gains.

The temporary root/growth trajectory is diagnostic, not a publishable object.
An unsuccessful transaction returns the unchanged input; final serialized mesh,
thickness, slicing and structural checks remain the pipeline's responsibility.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy import ndimage as ndi

from .geometry import _box_ray_pixels, _enclosed_voids, _physical_rank, _quality, _pitch_from_occupied_bounds
from .manufacturing import FACE6, _positive, _volume, thickness_audit
from .rendering import _axis_map, _key, render_binary


class BlockProjection:
    """Exact opaque union with a solid cell box; no centre-ray approximation."""

    def __init__(self, prediction, targets, cameras, resolution, extent):
        self.cameras = tuple(cameras)
        self.targets = np.asarray(targets, bool).copy()
        self.prediction = np.asarray(prediction, bool).copy()
        if (self.targets.ndim != 3 or self.targets.shape != self.prediction.shape
                or self.targets.shape[0] != len(self.cameras) or not len(self.cameras)
                or self.targets.shape[1] != self.targets.shape[2] or not self.targets.shape[1]):
            raise ValueError('matching nonempty square targets and predictions required')
        self.size = self.targets.shape[1]
        self.resolution, self.extent = resolution, extent
        self.maps = [_axis_map(_key(c), self.size, resolution, extent) for c in self.cameras]
        self.cardinal = all(m is not None for m in self.maps)

    @staticmethod
    def _rectangle(summed, lo, hi):
        return summed[hi[:, 0], hi[:, 1]] - summed[lo[:, 0], hi[:, 1]] - summed[hi[:, 0], lo[:, 1]] + summed[lo[:, 0], lo[:, 1]]

    def qualities(self, lower, width):
        lower = np.asarray(lower)
        if (lower.ndim != 2 or lower.shape[1] != 3 or lower.dtype.kind not in 'iu'
                or type(width) is not int or width < 1 or np.any(lower < 0)
                or np.any(lower + width > self.resolution)):
            raise ValueError('unclipped integer cell boxes required')
        tp = (self.prediction & self.targets).sum((1, 2))
        union = (self.prediction | self.targets).sum((1, 2))
        gained = np.zeros((len(lower), len(self.cameras)), np.int64)
        leaked = np.zeros_like(gained)
        for i, mapping in enumerate(self.maps):
            if mapping is None:
                for j, lo in enumerate(lower):
                    pixels = _box_ray_pixels(lo, lo + width, [self.cameras[i]], self.size,
                                             self.resolution, self.extent)[0]
                    unseen = ~self.prediction[i].ravel()[pixels]
                    truth = self.targets[i].ravel()[pixels]
                    gained[j, i] = np.count_nonzero(unseen & truth)
                    leaked[j, i] = np.count_nonzero(unseen & ~truth)
            else:
                axis, indices, valid = mapping
                axes = [a for a in range(3) if a != axis]
                for mask, destination in ((self.targets[i] & ~self.prediction[i], gained),
                                           (~self.targets[i] & ~self.prediction[i], leaked)):
                    select = valid & mask.ravel()
                    plane = np.bincount(indices[select], minlength=self.resolution ** 2).reshape((self.resolution,) * 2)
                    integral = np.pad(plane.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                    destination[:, i] = self._rectangle(integral, lower[:, axes], lower[:, axes] + width)
        denominator = union[None] + leaked
        return np.divide(tp[None] + gained, denominator, out=np.ones_like(denominator, float), where=denominator > 0)

    def add(self, lower, width):
        for i, pixels in enumerate(_box_ray_pixels(lower, lower + width, self.cameras,
                                                   self.size, self.resolution, self.extent)):
            self.prediction[i].ravel()[pixels] = True


def _eligible_boxes(volume, protected, width):
    """Overlap material (face-touch for unit cells), retaining 6-connectivity."""
    if width == 1:
        return np.argwhere(ndi.binary_dilation(volume, structure=FACE6) & ~volume & ~protected)
    origin = -1 if width % 2 == 0 else 0
    touches = ndi.maximum_filter(volume, size=width, origin=origin, mode='constant', cval=0)
    full = ndi.minimum_filter(volume, size=width, origin=origin, mode='constant', cval=0)
    forbidden = ndi.maximum_filter(protected, size=width, origin=origin, mode='constant', cval=0)
    centers = np.argwhere(touches & ~full & ~forbidden)
    lo = centers - (width - 1) // 2
    return lo[((lo >= 0) & (lo + width <= volume.shape[0])).all(axis=1)]


def repair_joint(occupancy_xyz, guidance_xyz, targets, cameras, *, extent=1.7,
                 longest_mm=100., max_steps=256, min_wall_mm=.9, min_neck_mm=1.35,
                 callback=None):
    """Attempt a complete island-removal/connected-block-restoration transaction.

    No intermediate low-IoU state replaces the original. All projections are
    exact full-ray opaque projections. This finite greedy search is not a proof
    of feasibility or infeasibility and does not guarantee printing thickness.
    """
    volume = _volume(occupancy_xyz).copy()
    extent, longest_mm = _positive(extent, 'extent'), _positive(longest_mm, 'longest_mm')
    wall, neck = _positive(min_wall_mm, 'min_wall_mm'), _positive(min_neck_mm, 'min_neck_mm')
    if type(max_steps) is not int or not 0 <= max_steps <= 1024:
        raise ValueError('max_steps must be an integer in [0,1024]')
    if callback is not None and not callable(callback):
        raise ValueError('callback must be callable')
    guidance = np.asarray(guidance_xyz)
    if guidance.shape != volume.shape or not np.isfinite(guidance).all() or np.any(guidance < 0) or np.any(guidance > 1):
        raise ValueError('guidance must match occupancy and lie in [0,1]')
    targets = np.asarray(targets)
    cameras = tuple(cameras)
    if (targets.ndim != 3 or not cameras or targets.shape[0] != len(cameras)
            or not targets.shape[1] or targets.shape[1] != targets.shape[2]):
        raise ValueError('nonempty square targets matching cameras required')
    if not np.isin(targets, [0, 1]).all():
        raise ValueError('binary targets required')
    initial_prediction = render_binary(volume, cameras, targets.shape[-1], extent)
    scorer_validation = BlockProjection(initial_prediction, targets, cameras, volume.shape[0], extent)
    targets = scorer_validation.targets
    if not volume.any():
        raise ValueError('nonempty initial occupancy required')
    initial_quality = _quality(initial_prediction, targets, .98, .97)
    labels, _ = ndi.label(volume, FACE6)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    candidate = labels == sizes.argmax()
    del labels
    protected = _enclosed_voids(volume)
    # Future additions may enlarge the bounding box: use the full cube's pitch,
    # not the current smaller body's optimistic pitch when choosing block width.
    width = max(1, int(math.ceil(max(wall, neck) / (longest_mm / volume.shape[0]))))
    scorer = BlockProjection(render_binary(candidate, cameras, targets.shape[-1], extent),
                             targets, cameras, volume.shape[0], extent)
    root_quality = _quality(scorer.prediction, targets, .98, .97)
    quality = root_quality
    history = []
    proposals = 0
    stop_reason = 'STEP_BUDGET'
    for step in range(max_steps):
        if width > volume.shape[0]:
            stop_reason = 'THICKNESS_EXCEEDS_DOMAIN'
            break
        lower = _eligible_boxes(candidate, protected, width)
        if not len(lower):
            stop_reason = 'NO_CONNECTED_BLOCK'
            break
        if not scorer.cardinal and len(lower) > 256:
            # Explicit generic-camera shortlist; exact scoring still follows.
            centers = lower + (width - 1) // 2
            prior = guidance[tuple(centers.T)]
            selected = np.argsort(-prior, kind='stable')[:256]
            lower = lower[selected]
        values = scorer.qualities(lower, width)
        means, worst = values.mean(1), values.min(1)
        objective = means + worst
        old_objective = quality['mean_iou'] + quality['worst_iou']
        proposals += len(lower)
        best = float(objective.max())
        if best <= old_objective + 1e-12:
            stop_reason = 'NO_POSITIVE_EXACT_SHADOW_GAIN'
            break
        ties = np.flatnonzero(objective >= best - 1e-12)
        centers = lower[ties] + (width - 1) // 2
        index = ties[np.argmax(guidance[tuple(centers.T)])]
        lo = lower[index]
        candidate[tuple(slice(int(a), int(a + width)) for a in lo)] = True
        scorer.add(lo, width)
        quality = _quality(scorer.prediction, targets, .98, .97)
        row = {'step': step + 1, 'lower_xyz': lo.tolist(), 'width_voxels': width,
               'mean_iou': quality['mean_iou'], 'worst_iou': quality['worst_iou'],
               'evaluated_boxes': len(lower), 'cumulative_evaluated_boxes': proposals}
        history.append(row)
        if callback:
            callback(copy.deepcopy(row))
    actual = render_binary(candidate, cameras, targets.shape[-1], extent)
    if not np.array_equal(actual, scorer.prediction):
        raise RuntimeError('JOINT_BOX_PROJECTION_MISMATCH')
    candidate_quality = _quality(actual, targets, .98, .97)
    initial_physical = thickness_audit(volume, _pitch_from_occupied_bounds(volume, longest_mm), wall, neck)
    candidate_physical = thickness_audit(candidate, _pitch_from_occupied_bounds(candidate, longest_mm), wall, neck)
    reasons = []
    if max_steps == 0:
        reasons.append('ZERO_BUDGET_NO_COMMIT')
    if (candidate_quality['mean_iou'] < initial_quality['mean_iou'] - 1e-12
            or candidate_quality['worst_iou'] < initial_quality['worst_iou'] - 1e-12):
        reasons.append('SHADOW_REGRESSION_TRANSACTION_ROLLED_BACK')
    if np.any(candidate & protected) or (protected.any() and np.any(protected & ~_enclosed_voids(candidate))):
        reasons.append('ORIGINAL_CAVITY_CHANGED')
    if candidate_physical['component_count_6'] != 1 or _physical_rank(candidate_physical) < _physical_rank(initial_physical):
        reasons.append('PHYSICAL_REGRESSION')
    changed = not np.array_equal(candidate, volume)
    if not changed:
        reasons.append('NO_CHANGE')
    accepted = not reasons
    out = candidate.copy() if accepted else volume.copy()
    trace = {'algorithm': 'joint_connected_block_replacement_v1', 'accepted': accepted,
             'status': 'ACCEPTED_CANDIDATE' if accepted else 'ROLLED_BACK',
             'publication_eligible': False, 'structural_status': 'NOT_REQUESTED',
             'source_occupancy_mutated': False, 'global_hole_fill': False,
             'max_steps': max_steps, 'width_voxels': width,
             'generic_shortlist_budget': 256, 'all_cameras_cardinal': scorer.cardinal,
             'evaluated_boxes': proposals, 'stop_reason': stop_reason, 'reasons': reasons,
             'initial_quality': initial_quality, 'root_quality': root_quality,
             'candidate_quality': candidate_quality,
             'final_quality': candidate_quality if accepted else initial_quality,
             'initial_manufacturing': initial_physical, 'candidate_manufacturing': candidate_physical,
             'history': history}
    return out, trace, candidate.copy()
