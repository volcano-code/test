"""Finite, connected full-path and shadow-replacement proposals for one island."""
import numpy as np
from scipy import ndimage as ndi

from .geometry import _box_ray_pixels, _thicken_mask
from .joint_geometry import BlockProjection, _eligible_boxes
from .manufacturing import FACE6, _block_cover
from .rendering import render_binary


def component_order(volume, width, order='nearest'):
    if order not in ('nearest','largest'):
        raise ValueError('region order must be nearest or largest')
    labels, count = ndi.label(volume, FACE6)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    root = labels == sizes.argmax()
    centres, _ = _block_cover(root, width)
    distance = ndi.distance_transform_edt(~(centres if centres.any() else root))
    rows = []
    boxes = ndi.find_objects(labels)
    for i, box in enumerate(boxes, 1):
        if i == sizes.argmax():
            continue
        points = np.argwhere(labels[box] == i) + np.array([s.start for s in box])
        values = distance[tuple(points.T)]
        rows.append((float(values.min()), int(sizes[i]), points))
    key = (lambda row: (row[0], row[1], tuple(row[2][0]))) if order=='nearest' else (
        lambda row: (-row[1], row[0], tuple(row[2][0])))
    return [p for _, _, p in sorted(rows, key=key)]


def propose(volume, guidance, points, targets, cameras, protected, width, extent, budget,
            *, body_reinforcement=False):
    """Return fixed base plus additive sparse primitives. Each primitive meets root.

    No optimality/completeness claim: two thick route costs and bounded connected
    boxes. Geometry outside the selected island and proposed additions is fixed.
    """
    if type(body_reinforcement) is not bool:
        raise ValueError('body_reinforcement must be boolean')
    base = volume.copy(); base[tuple(points.T)] = False
    labels, _ = ndi.label(base, FACE6)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    root = labels == sizes.argmax()
    size, resolution = targets.shape[-1], volume.shape[0]
    base_pred = render_binary(base, cameras, size, extent)
    current_pred = render_binary(volume, cameras, size, extent)
    lost = current_pred & ~base_pred & targets
    if body_reinforcement:
        # A retained/reinforced body already restores its old rays. Compensation
        # must also be able to recover pre-existing false negatives, otherwise
        # adding patches cannot offset the body's new false-positive footprint.
        lost = targets & ~base_pred
    primitives, projections, descriptions = [], [], []
    seen_geometry, projection_counts = set(), {}

    def add(mask, projection, description):
        if (body_reinforcement and description['kind'] != 'connected_replacement_box'
                and len(primitives) >= max(1, budget // 2)):
            # Keep at least half the finite pool for independent ray patches.
            # Reinforcement cannot consume the projection-compensation budget.
            return
        extra = np.flatnonzero(mask & ~base)
        if not len(extra) or np.any(protected.ravel()[extra]):
            return
        signature = np.packbits(projection & ~base_pred).tobytes()
        geometry_key = extra.tobytes()
        if geometry_key in seen_geometry or projection_counts.get(signature, 0) >= 2:
            return
        seen_geometry.add(geometry_key)
        projection_counts[signature] = projection_counts.get(signature, 0) + 1
        primitives.append(extra)
        projections.append(projection)
        descriptions.append(description)

    # Whole path: no requirement that its intermediate cells immediately gain IoU.
    centres, _ = _block_cover(root, width)
    anchor = centres if centres.any() else root
    distance, nearest = ndi.distance_transform_edt(~anchor, return_indices=True)
    part = np.zeros_like(base); part[tuple(points.T)] = True
    part_centres, _ = _block_cover(part, width)
    eligible = np.argwhere(part_centres) if part_centres.any() else points
    start = eligible[np.argmin(distance[tuple(eligible.T)])]
    end = nearest[(slice(None), *start)]
    margin = max(4, 2 * width)
    lo = np.maximum(np.minimum(start, end) - margin, 0)
    hi = np.minimum(np.maximum(start, end) + margin + 1, resolution)
    roi = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    from skimage.graph import route_through_array
    for prior_weight in (0., 3.):
        cost = 1. + prior_weight * (1. - guidance[roi])
        forbidden = ndi.maximum_filter(protected[roi], width, mode='constant', cval=0)
        cost[forbidden] = 1e9
        path, _ = route_through_array(cost, tuple(start - lo), tuple(end - lo),
                                      fully_connected=False, geometric=False)
        xyz = np.asarray(path, int) + lo
        line = np.zeros_like(base); line[tuple(xyz.T)] = True
        if body_reinforcement:
            from itertools import product
            from .regional_geometry import thick_join_preserved
            # Union of width-cubes around every body/path cell, not a surface
            # dilation followed by hole filling. Opposite placements reduce the
            # even-width grid bias; all candidates obey the same physical guard.
            offsets = sorted({-(width // 2), (width - 1) // 2})
            path_mask = part | _thicken_mask(line, width)
            uncovered = part & ~_block_cover(base | path_mask, width)[1]
            # Target thin body cells first. Already thick lobe surfaces need no
            # blanket growth. Whole-body alternatives are considered next, but
            # finite-pool and per-projection dedup limits may exclude them.
            for scope, seed in (('uncovered_voxels', uncovered), ('whole_body', part | line)):
                for origin in product(offsets, repeat=3):
                    mask = ndi.maximum_filter(seed, size=width, origin=origin,
                                              mode='constant', cval=0) | path_mask
                    candidate = base | mask
                    if np.any(mask & protected) or np.any(part & ~_block_cover(candidate, width)[1]):
                        continue
                    if not thick_join_preserved(candidate, base, candidate & ~volume, width):
                        continue
                    add(mask, render_binary(mask, cameras, size, extent),
                        {'kind':'reinforced_body_and_path', 'body_scope':scope,
                         'path_cells':len(path), 'prior_weight':prior_weight, 'width_voxels':width,
                         'cube_origin_xyz':list(origin)})
        mask = _thicken_mask(line, width) | part
        if body_reinforcement and not thick_join_preserved(base | mask, base, mask & ~volume, width):
            # Do not spend one of four expensive physical-audit slots on a path
            # already known to leave an unanchored neck core. Default unchanged.
            continue
        projection = render_binary(mask, cameras, size, extent)
        add(mask, projection, {'kind':'retain_and_complete_thick_path',
                              'path_cells':len(path), 'prior_weight':prior_weight})

    # Alternative depths of the lost foreground rays, attached to the retained root.
    coverage_scorer = BlockProjection(base_pred | ~lost, lost, cameras, resolution, extent)
    quality_scorer = BlockProjection(base_pred, targets, cameras, resolution, extent)
    proposals = []
    for block_width in sorted({width, width + 1, 2 * width}):
        if block_width > resolution or not lost.any():
            continue
        lower = _eligible_boxes(root, protected, block_width)
        if not coverage_scorer.cardinal and len(lower) > 256:
            # Generic-camera pilot is explicitly bounded, not claimed exhaustive.
            centre = lower + (block_width - 1) // 2
            lower = lower[np.argsort(-guidance[tuple(centre.T)], kind='stable')[:256]]
        if not len(lower):
            continue
        gains = coverage_scorer.qualities(lower, block_width).sum(1) * size ** 2
        keep = gains > 0
        lower, gains = lower[keep], gains[keep]
        if not len(lower):
            continue
        quality = quality_scorer.qualities(lower, block_width)
        rank = quality.mean(1) + quality.min(1)
        centre = lower + (block_width - 1) // 2
        order = np.lexsort((-guidance[tuple(centre.T)], -gains, -rank))
        # Allow projection deduplication to skip repeated depths before pool cap.
        for index in order[:max(256, budget * 16)]:
            proposals.append((float(rank[index]), float(gains[index]),
                              float(guidance[tuple(centre[index])]), lower[index], block_width))
    for _, _, _, lower, block_width in sorted(proposals, key=lambda p:(-p[0],-p[1],-p[2])):
        if len(primitives) >= budget:
            break
        projection = np.zeros_like(base_pred)
        for j, pixels in enumerate(_box_ray_pixels(lower, lower + block_width, cameras, size, resolution, extent)):
            projection[j].ravel()[pixels] = True
        mask = np.zeros_like(base)
        mask[tuple(slice(int(a), int(a + block_width)) for a in lower)] = True
        add(mask, projection, {'kind':'connected_replacement_box',
                              'lower_xyz':lower.tolist(), 'width_voxels':block_width})
    return base, base_pred, primitives[:budget], np.asarray(projections[:budget], bool).reshape(
        (-1, *base_pred.shape)), descriptions[:budget]
