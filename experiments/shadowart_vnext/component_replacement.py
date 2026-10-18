"""Bounded whole-island replacement by a root-attached thick-block sequence.

Deletion and growth are one transaction. Intermediate volumes may lose shadow;
only the complete candidate can commit, and it is never a print certificate.
The greedy finite family is not a feasibility/infeasibility proof.
"""
import copy
import math

import numpy as np
from scipy import ndimage as ndi

from .geometry import _enclosed_voids, _pitch_from_occupied_bounds, _quality
from .joint_geometry import BlockProjection
from .manufacturing import FACE6, _block_cover, _positive, _volume, thickness_audit
from .regional_geometry import cavities_preserved, manufacturing_improves, thick_join_preserved
from .rendering import render_binary


def neck_frontier(connected, protected, width):
    """Solid boxes with a centre face-adjacent to an existing inscribed neck core."""
    centres, _ = _block_cover(connected, width)
    frontier = ndi.binary_dilation(centres, structure=FACE6) & ~centres
    forbidden = ndi.maximum_filter(protected, size=width, mode='constant', cval=0)
    lower = np.argwhere(frontier & ~forbidden) - width // 2
    return lower[((lower >= 0) & (lower + width <= connected.shape[0])).all(axis=1)]


def replace_component(occupancy_xyz, guidance_xyz, component_mask, targets, cameras, *,
                      extent=1.7, longest_mm=100., block_width=None, max_steps=128, callback=None,
                      whole_path=False, allow_route_dip=False):
    if type(whole_path) is not bool:
        raise ValueError('whole_path must be boolean')
    if type(allow_route_dip) is not bool or (allow_route_dip and not whole_path):
        raise ValueError('allow_route_dip must be boolean and requires whole_path')
    original = _volume(occupancy_xyz).copy()
    selected = _volume(component_mask).copy()
    extent, longest_mm = _positive(extent, 'extent'), _positive(longest_mm, 'longest_mm')
    if selected.shape != original.shape or not selected.any() or np.any(selected & ~original):
        raise ValueError('nonempty component selection inside occupancy required')
    if type(max_steps) is not int or not 0 <= max_steps <= 1024:
        raise ValueError('max_steps must be integer in [0,1024]')
    minimum = max(1, math.ceil(1.35 / (longest_mm / original.shape[0])))
    width = minimum if block_width is None else block_width
    if type(width) is not int or not minimum <= width <= original.shape[0]:
        raise ValueError('block_width must be an integer meeting neck thickness within the domain')
    if callback is not None and not callable(callback):
        raise ValueError('callback must be callable')
    guidance = np.asarray(guidance_xyz)
    if guidance.shape != original.shape or not np.isfinite(guidance).all() or np.any((guidance < 0) | (guidance > 1)):
        raise ValueError('guidance must match occupancy in [0,1]')
    targets = np.asarray(targets)
    if not np.isin(targets, [0, 1]).all():
        raise ValueError('binary targets required')
    cameras = tuple(cameras)
    labels, _ = ndi.label(original, FACE6)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    ids = np.unique(labels[selected])
    if len(ids) != 1 or ids[0] == sizes.argmax() or int(selected.sum()) != sizes[ids[0]]:
        raise ValueError('selection must be one whole non-main 6-connected component')
    connected = labels == sizes.argmax()
    base = original & ~selected
    candidate = base.copy()
    del labels
    protected = _enclosed_voids(original)
    # BlockProjection validates camera/target cardinality and raster shape.
    if targets.ndim != 3 or not targets.shape[-1]:
        raise ValueError('nonempty square target rasters required')
    size = targets.shape[-1]
    initial_prediction = render_binary(original, cameras, size, extent)
    scorer = BlockProjection(render_binary(base, cameras, size, extent), targets, cameras,
                             original.shape[0], extent)
    def quality(prediction):
        return _quality(prediction, scorer.targets, .98, .97)
    initial_quality = quality(initial_prediction)
    current_quality = base_quality = quality(scorer.prediction)
    history, evaluated = [], 0
    used, path_attempts = 0, []
    stop_reason = 'STEP_BUDGET'
    while used < max_steps:
        lower = neck_frontier(connected, protected, width)
        if not len(lower):
            stop_reason = 'NO_ROOT_CONNECTED_NECK_FRONTIER'
            break
        if not scorer.cardinal and len(lower) > 256:
            centre = lower + width // 2
            lower = lower[np.argsort(-guidance[tuple(centre.T)], kind='stable')[:256]]
        values = scorer.qualities(lower, width)
        objectives = values.mean(1) + values.min(1)
        best = float(objectives.max())
        evaluated += len(lower)
        if best <= current_quality['mean_iou'] + current_quality['worst_iou'] + 1e-12:
            if whole_path and len(path_attempts) < 4:
                from .replacement_paths import best_whole_path
                routed, detail = best_whole_path(candidate, connected, selected, guidance, protected,
                                                  scorer, width, max_steps - used, allow_dip=allow_route_dip)
                path_attempts.append(detail)
                if routed is not None:
                    addition, prediction, row = routed
                    candidate |= addition
                    connected |= addition
                    scorer.prediction = prediction.copy()
                    current_quality = row['quality']
                    used += row['block_steps']
                    row.update(step=used, width_voxels=width)
                    history.append(row)
                    if callback:
                        callback(copy.deepcopy(row))
                    continue
            stop_reason = 'NO_POSITIVE_GREEDY_RAY_GAIN'
            break
        tied = np.flatnonzero(objectives >= best - 1e-12)
        centres = lower[tied] + width // 2
        index = tied[np.argmax(guidance[tuple(centres.T)])]
        lo = lower[index]
        box = tuple(slice(int(a), int(a + width)) for a in lo)
        candidate[box] = True
        connected[box] = True
        scorer.add(lo, width)
        current_quality = quality(scorer.prediction)
        used += 1
        row = {'step':used, 'lower_xyz':lo.tolist(), 'width_voxels':width,
               'operation':'solid_block', 'block_steps':1,
               'quality':current_quality, 'evaluated_boxes':len(lower)}
        history.append(row)
        if callback:
            callback(copy.deepcopy(row))
    actual = render_binary(candidate, cameras, size, extent)
    if not np.array_equal(actual, scorer.prediction):
        raise RuntimeError('COMPONENT_REPLACEMENT_PROJECTION_MISMATCH')
    final_quality = quality(actual)
    before = thickness_audit(original, _pitch_from_occupied_bounds(original, longest_mm))
    after = thickness_audit(candidate, _pitch_from_occupied_bounds(candidate, longest_mm))
    neck_ok = thick_join_preserved(candidate, base, candidate & ~original, width)
    reasons = []
    if max_steps == 0:
        reasons.append('ZERO_BUDGET_NO_COMMIT')
    if (final_quality['mean_iou'] < initial_quality['mean_iou'] - 1e-12 or
            final_quality['worst_iou'] < initial_quality['worst_iou'] - 1e-12 or
            np.any(np.asarray(final_quality['view_ious']) < np.asarray(initial_quality['view_ious']) - .002 - 1e-12)):
        reasons.append('PROJECTION_REGRESSION')
    if not cavities_preserved(candidate, protected):
        reasons.append('ORIGINAL_CAVITY_CHANGED')
    if not neck_ok:
        reasons.append('NEW_CONNECTION_NECK_NOT_ROOT_CONNECTED')
    if not manufacturing_improves(before, after):
        reasons.append('MANUFACTURING_NOT_PARETO_IMPROVED')
    accepted = not reasons
    trace = {'algorithm':'component_neck_connected_replacement_v1', 'accepted':accepted,
             'publication_eligible':False, 'structural_status':'NOT_REQUESTED',
             'source_occupancy_mutated':False, 'global_hole_fill':False,
             'selected_voxels':int(selected.sum()), 'max_steps':max_steps, 'width_voxels':width,
             'all_cameras_cardinal':scorer.cardinal, 'generic_shortlist_budget':256,
             'evaluated_boxes':evaluated, 'stop_reason':stop_reason, 'reasons':reasons,
             'initial_quality':initial_quality, 'base_quality':base_quality,
             'candidate_quality':final_quality, 'neck_guard_passed':neck_ok,
             'initial_manufacturing':before, 'candidate_manufacturing':after,
             'acceptance_scope':'nonregressing_component_transaction_not_final_delivery',
             'mean_regression_budget':0., 'worst_regression_budget':0., 'view_regression_budget':.002,
             'whole_path':whole_path, 'path_attempts':path_attempts, 'max_path_attempts':4,
             'allow_route_dip':allow_route_dip,
             'temporary_route_objective_dip_budget':.002 if allow_route_dip else 0.,
             'consumed_block_steps':used,
             'history':history}
    return (candidate.copy() if accepted else original.copy()), trace, candidate.copy()
