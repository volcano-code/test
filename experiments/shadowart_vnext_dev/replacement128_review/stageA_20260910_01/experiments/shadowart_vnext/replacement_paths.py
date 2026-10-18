"""Finite whole-path proposals across shadow-neutral depth gaps."""
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import route_through_array

from .geometry import _thicken_mask, _quality
from .manufacturing import _block_cover
from .rendering import render_binary


def best_whole_path(candidate, connected, selected, guidance, protected, scorer, width, remaining,
                    *, allow_dip=False):
    """Evaluate up to eight goal paths, charging every path centre to the budget.

    Goal cubes overlap the removed component. Routing on cube centres means
    face-adjacent steps preserve neck-core connectivity, including neutral steps.
    Endpoint-only gains rank goals but do not exclude them: intermediate route
    cubes may recover foreground that the endpoint itself does not cover.
    This is a deterministic shortlist, not a complete path or layout search.
    """
    r, c = candidate.shape[0], width // 2
    forbidden = ndi.maximum_filter(protected, size=width, mode='constant', cval=0)
    legal = np.zeros_like(candidate)
    legal[(slice(c, r - width + c + 1),) * 3] = True
    legal &= ~forbidden
    anchors = _block_cover(connected, width)[0] & legal
    dip_budget = .002 if allow_dip else 0.
    detail = {'evaluated_routes':0, 'goal_budget':8, 'remaining_block_steps':remaining, 'routes':[],
              'goal_policy':'ranked_without_endpoint_gain_gate_v2',
              'temporary_objective_dip_budget':dip_budget}
    if not anchors.any():
        return None, detail
    overlap = ndi.maximum_filter(selected, size=width, mode='constant', cval=0)
    full = ndi.minimum_filter(candidate, size=width, mode='constant', cval=0)
    goals = np.argwhere(overlap & ~full & legal)
    if not len(goals):
        return None, detail
    values = scorer.qualities(goals - c, width)
    rank = values.mean(1) + values.min(1)
    old = _quality(scorer.prediction, scorer.targets, .98, .97)
    old_score = old['mean_iou'] + old['worst_iou']
    best_score = old_score - dip_budget
    order = np.lexsort((-guidance[tuple(goals.T)], -rank))
    shortlist = []
    for i in order:
        goal = goals[i]
        if all(np.max(np.abs(goal - g)) >= width for g in shortlist):
            shortlist.append(goal)
        if len(shortlist) == 8:
            break
    _, nearest = ndi.distance_transform_edt(~anchors, return_indices=True)
    best = None
    for goal in shortlist:
        start = nearest[(slice(None), *goal)]
        lo = np.maximum(np.minimum(start, goal) - 4, c)
        hi = np.minimum(np.maximum(start, goal) + 5, r - width + c + 1)
        roi = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        cost = 1. + 3. * (1. - guidance[roi])
        cost[~legal[roi]] = np.inf
        detail['evaluated_routes'] += 1
        trial = {'goal_center_xyz':goal.tolist()}
        detail['routes'].append(trial)
        try:
            path, _ = route_through_array(cost, tuple(start - lo), tuple(goal - lo),
                                          fully_connected=False, geometric=False)
        except ValueError:
            trial['status'] = 'UNREACHABLE'
            continue
        trial['block_steps'] = len(path)
        if len(path) > remaining:
            trial['status'] = 'BLOCK_BUDGET'
            continue
        centres = np.asarray(path, int) + lo
        line = np.zeros_like(candidate); line[tuple(centres.T)] = True
        addition = _thicken_mask(line, width)
        if np.any(addition & protected):
            trial['status'] = 'PROTECTED_VOID'
            continue
        prediction = render_binary(candidate | addition, scorer.cameras, scorer.size, scorer.extent)
        q = _quality(prediction, scorer.targets, .98, .97)
        score = q['mean_iou'] + q['worst_iou']
        trial.update(status='EVALUATED', quality=q,
                     objective_gain_vs_current=score - old['mean_iou'] - old['worst_iou'])
        if score > best_score + 1e-12:
            best_score = score
            best = (addition, prediction, {'operation':'whole_thick_path',
                    'path_centres_xyz':centres.tolist(), 'block_steps':len(path), 'quality':q})
    return best, detail
