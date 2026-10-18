"""Opt-in regional constrained transactions, separate fidelity/search archives."""
import copy
import hashlib
import math

import numpy as np
from scipy import ndimage as ndi

from .geometry import _enclosed_voids, _pitch_from_occupied_bounds, _quality
from .manufacturing import FACE6, _positive, _volume, _block_cover, thickness_audit
from .rendering import render_binary


def within_budget(values, initial, mean_budget, view_budget):
    return bool(np.mean(values) >= np.mean(initial) - mean_budget - 1e-12
                and np.all(np.asarray(values) >= np.asarray(initial) - view_budget - 1e-12))


def cavities_preserved(volume, protected):
    return bool(not np.any(volume & protected) and
                (not protected.any() or not np.any(protected & ~_enclosed_voids(volume))))


def physical_vector(report):
    return (report['component_count_6'], report['thin_voxels'],
            *(value for s in report['neck_scales'] for value in
              (s['significant_components'], s['surviving_components'], s['vanished_wall_valid_voxels'])))


def manufacturing_improves(before, after):
    old, new = physical_vector(before), physical_vector(after)
    return bool(new[0] < old[0] and all(a <= b for a,b in zip(new[1:],old[1:]))
                and not (set(after['reasons']) - set(before['reasons']))
                and not (set(after['review_reasons']) - set(before['review_reasons'])))


def thick_join_preserved(candidate, retained_base, added, width):
    """New neck cores must reach retained-root neck cores, not touch thin corners."""
    if not added.any():
        return True
    labels, _ = ndi.label(retained_base, FACE6)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    root = labels == sizes.argmax()
    root_centres, _ = _block_cover(root, width)
    if not root_centres.any():
        return False
    centres, cover = _block_cover(candidate, width)
    if np.any(added & ~cover):
        return False
    base_centres, _ = _block_cover(retained_base, width)
    core_labels, _ = ndi.label(centres, FACE6)
    anchors = np.unique(core_labels[root_centres]); anchors = anchors[anchors != 0]
    new_labels = np.unique(core_labels[centres & ~base_centres])
    return bool(np.isin(new_labels, anchors).all())


def repair_regional(occupancy_xyz, guidance_xyz, targets, cameras, *, extent=1.7,
                    longest_mm=100., max_regions=16, max_primitives=24,
                    mean_budget=.001, view_budget=.002, time_limit=5., callback=None,
                    region_order='nearest', body_reinforcement=False):
    if type(body_reinforcement) is not bool:
        raise ValueError('body_reinforcement must be boolean')
    original = _volume(occupancy_xyz).copy()
    extent, longest_mm = _positive(extent, 'extent'), _positive(longest_mm, 'longest_mm')
    time_limit = _positive(time_limit, 'time_limit')
    if region_order not in ('nearest','largest'):
        raise ValueError('region order must be nearest or largest')
    if type(max_regions) is not int or not 0 <= max_regions <= 128:
        raise ValueError('max_regions must be integer in [0,128]')
    if type(max_primitives) is not int or not 1 <= max_primitives <= 256:
        raise ValueError('max_primitives must be integer in [1,256]')
    if any(not np.isfinite(b) or not 0 <= b <= .002 for b in (mean_budget, view_budget)):
        raise ValueError('absolute IoU budgets must be finite in [0,.002]')
    if callback is not None and not callable(callback):
        raise ValueError('callback must be callable')
    guidance = np.asarray(guidance_xyz)
    if guidance.shape != original.shape or not np.isfinite(guidance).all() or np.any((guidance < 0) | (guidance > 1)):
        raise ValueError('guidance must match occupancy in [0,1]')
    targets = np.asarray(targets); cameras = tuple(cameras)
    if (targets.ndim != 3 or targets.shape[0] != len(cameras) or not cameras
            or not targets.shape[1] or targets.shape[1] != targets.shape[2]
            or not np.isin(targets, [0,1]).all()):
        raise ValueError('binary square targets matching cameras required')
    targets = targets.astype(bool)
    if not original.any():
        raise ValueError('nonempty occupancy required')
    size = targets.shape[-1]
    def quality(v):
        return _quality(render_binary(v, cameras, size, extent), targets, .98, .97)
    def physical(v):
        return thickness_audit(v, _pitch_from_occupied_bounds(v, longest_mm))
    initial = quality(original)
    baseline = np.array(initial['view_ious'])
    search, archive = original.copy(), original.copy()
    archive_quality = initial
    history = []
    initial_physical = current_physical = physical(original)
    protected = _enclosed_voids(original)
    width = max(1, math.ceil(1.35 / (longest_mm / original.shape[0])))
    from .regional_candidates import component_order, propose
    from .regional_solver import solve_union
    order = component_order(original, width, order=region_order)[:max_regions] if max_regions else []
    for region_index, points in enumerate(order):
        labels, _ = ndi.label(search, FACE6)
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        selected_labels = np.unique(labels[tuple(points.T)])
        selected_labels = selected_labels[selected_labels != 0]
        if not len(selected_labels) or sizes.argmax() in selected_labels:
            continue
        # A previously modified island may have merged with another: select its
        # whole present component, not a stale subset that could cut a new neck.
        points = np.argwhere(np.isin(labels, selected_labels))
        del labels
        base, pred, primitives, projections, descriptions = propose(
            search, guidance, points, targets, cameras, protected, width, extent, max_primitives,
            **({'body_reinforcement':True} if body_reinforcement else {}))
        chosen, solver = solve_union(pred, targets, projections, np.maximum(0., baseline - view_budget),
                                     max_selected=min(4, len(primitives)), time_limit=time_limit)
        choices = [()] + ([(tuple(chosen))] if chosen is not None else [])
        choices += [(i,) for i in range(len(primitives))]
        scored = []
        for selection in dict.fromkeys(choices):
            projection = pred.copy()
            for index in selection:
                projection |= projections[index]
            q = _quality(projection, targets, .98, .97)
            if within_budget(q['view_ious'], baseline, mean_budget, view_budget):
                scored.append((selection, q))
        scored.sort(key=lambda row:(-row[1]['worst_iou'], -row[1]['mean_iou'], len(row[0])))
        row = {'region':region_index + 1, 'removed_component_voxels':len(points),
               'primitive_count':len(primitives), 'solver':solver, 'accepted':False,
               'candidate_rejections':[], 'budget_reference':'immutable_initial_occupancy'}
        old_vector = physical_vector(current_physical)
        # Expensive global audits are bounded separately and their limit is logged.
        for selection, expected in scored[:4]:
            candidate = base.copy()
            for index in selection:
                candidate.ravel()[primitives[index]] = True
            actual = quality(candidate)
            if not np.allclose(actual['view_ious'], expected['view_ious'], atol=1e-12, rtol=0):
                raise RuntimeError('REGIONAL_PROJECTION_MISMATCH')
            reasons = []
            if not cavities_preserved(candidate, protected):
                reasons.append('ORIGINAL_CAVITY_CHANGED')
            if not thick_join_preserved(candidate, base, candidate & ~search, width):
                reasons.append('NEW_CONNECTION_NECK_NOT_ROOT_CONNECTED')
            report = physical(candidate)
            vector = physical_vector(report)
            if not manufacturing_improves(current_physical, report):
                reasons.append('MANUFACTURING_NOT_PARETO_IMPROVED')
            if reasons:
                row['candidate_rejections'].append({'selection':list(selection), 'reasons':reasons,
                                                     'physical_vector':list(vector)})
                continue
            modified = np.flatnonzero(candidate != search)
            row.update(accepted=True, selection=list(selection), quality=actual,
                       physical_before=list(old_vector), physical_after=list(vector),
                       selected_operations=[descriptions[i] for i in selection],
                       modified_voxels=len(modified),
                       added_indices=np.flatnonzero(candidate & ~search).tolist(),
                       removed_indices=np.flatnonzero(search & ~candidate).tolist(),
                       modified_indices_sha256=hashlib.sha256(modified.tobytes()).hexdigest())
            search, current_physical = candidate, report
            if (actual['mean_iou'] >= archive_quality['mean_iou'] - 1e-12
                    and actual['worst_iou'] >= archive_quality['worst_iou'] - 1e-12):
                archive, archive_quality = search.copy(), actual
            break
        history.append(row)
        if callback:
            callback(copy.deepcopy(row))
    search_quality = quality(search)
    if search_quality['passed'] and current_physical['passed']:
        archive, archive_quality = search.copy(), search_quality
    trace = {'algorithm':'regional_constrained_replacement_v1', 'publication_eligible':False,
             'structural_status':'NOT_REQUESTED', 'global_hole_fill':False,
             'source_occupancy_mutated':False, 'max_regions':max_regions,
             'region_order':region_order,
             'max_primitives':max_primitives, 'max_selected':4, 'physical_audits_per_region':4,
             'solver_time_limit':time_limit, 'mean_budget':mean_budget, 'view_budget':view_budget,
             'budget_reference':'immutable_initial_occupancy', 'width_voxels':width,
             'accepted_regions':sum(r['accepted'] for r in history), 'history':history,
             'initial_quality':initial, 'search_quality':search_quality, 'final_quality':archive_quality,
             'initial_manufacturing':initial_physical, 'search_manufacturing':current_physical,
             'output_kind':'high_fidelity_archive', 'search_kind':'unpublished_manufacturing_search',
             'supported_quality_claim':False}
    if body_reinforcement:
        trace.update(algorithm='regional_body_path_replacement_v1', body_reinforcement=True)
    return archive, trace, search.copy()
