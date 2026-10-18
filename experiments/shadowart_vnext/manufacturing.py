"""Conservative physical feature checks for a union of occupied voxel cells.

These are digital geometric checks, not a strength or physical-print certificate.
An inscribed solid block gives a local thickness lower bound equal to its smallest
side. Covering voxels by such blocks avoids calling boundary EDT values wall
thickness, and does not erase a thin appendage just because the main body survives.
The axis-aligned block certificate is conservative for oblique/curved features:
uncovered small tips require review, rather than being assumed printable.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

FACE6 = ndi.generate_binary_structure(3, 1)


def _volume(value):
    raw = np.asarray(value)
    if raw.ndim != 3 or len(set(raw.shape)) != 1 or not raw.shape[0]:
        raise ValueError("occupancy must be a nonempty XYZ cube")
    if not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all():
        raise ValueError("occupancy must contain only binary values")
    return np.ascontiguousarray(raw, dtype=bool)


def _positive(value, name):
    if not np.isscalar(value) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _block_cover(volume, width):
    """Return inscribed-block centres and their exact occupied-cell coverage.

    Reflection of an even-sized footprint matters: using the same origin for
    erosion and dilation would translate the cover by one cell.
    """
    width = max(1, int(width))
    centres = ndi.minimum_filter(volume, size=width, mode="constant", cval=0)
    covered = ndi.maximum_filter(centres, size=width,
                                 origin=-1 if width % 2 == 0 else 0,
                                 mode="constant", cval=0)
    return centres.astype(bool), covered.astype(bool) & volume


def _thin_mask(volume, pitch_mm, min_wall_mm):
    width = max(1, int(math.ceil(min_wall_mm / pitch_mm - 1e-12)))
    centres, cover = _block_cover(volume, width)
    return volume & ~cover, centres, cover


def thickness_audit(volume_xyz, pitch_mm, min_wall_mm=.9, min_neck_mm=1.35):
    """Audit local block-certified thickness, lost appendages and neck scales.

    ``pitch_mm`` must be measured from the actual uniform fabrication transform,
    not guessed from a nominal 100 mm grid. Every field is JSON serializable.
    ``passed`` is false for failures AND unresolved terminal/sampling features.
    """
    v = _volume(volume_xyz)
    pitch = _positive(pitch_mm, "pitch_mm")
    wall = _positive(min_wall_mm, "min_wall_mm")
    neck = _positive(min_neck_mm, "min_neck_mm")
    if neck < wall:
        raise ValueError("minimum neck must be at least minimum wall")
    occupied = int(v.sum())
    _, components = ndi.label(v, FACE6)
    reasons, review = [], []
    if occupied == 0:
        reasons.append("EMPTY_VOLUME")
    elif components != 1:
        reasons.append("DISCONNECTED_MATERIAL")
    if pitch >= wall:
        review.append("SAMPLING_REVIEW")

    thin, wall_centres, cover = _thin_mask(v, pitch, wall)
    labels, count = ndi.label(thin, FACE6)
    boxes = ndi.find_objects(labels)
    adjacent_to_thick = ndi.binary_dilation(cover, FACE6)
    thin_regions = []
    for index in range(1, count + 1):
        box = boxes[index - 1]
        mask = labels[box] == index
        cells = int(mask.sum())
        extents = np.array([b.stop - b.start for b in box], dtype=float) * pitch
        attached = bool(np.any(mask & adjacent_to_thick[box]))
        long_axes = int(np.count_nonzero(extents >= 3 * wall))
        large = cells * pitch ** 3 >= wall ** 3
        if long_axes >= 2 and large:
            classification = "thin_sheet"
        elif float(extents.max()) >= 2 * neck and large:
            classification = "thin_appendage"
        else:
            classification = "terminal_tip_or_sampling_uncertain"
        vanished = bool(attached and not np.any(mask & wall_centres[box]))
        thin_regions.append({
            "id": index, "voxels": cells, "volume_mm3": cells * pitch ** 3,
            "extent_mm": extents.tolist(),
            "bounds_ijk": [[b.start for b in box], [b.stop for b in box]],
            "classification": classification, "attached_to_thick_material": attached,
            "fully_vanished_under_wall_erosion": vanished,
        })
    if any(r["classification"] in ("thin_sheet", "thin_appendage") for r in thin_regions):
        reasons.append("THIN_FEATURE")
    if any(r["classification"] == "terminal_tip_or_sampling_uncertain" for r in thin_regions):
        review.append("TERMINAL_TIP_OR_SAMPLING_REVIEW")

    widths = sorted({1, int(math.ceil(wall / pitch)), int(math.ceil(neck / pitch)),
                     *[w for w in (2, 3, 4, 6, 8, 12, 16) if w <= v.shape[0]]})
    local = np.zeros(v.shape, np.float32)
    for width in widths:
        _, certified = _block_cover(v, width)
        local[certified] = max(width * pitch, 0.)
    values = local[v]
    local_summary = {
        "method": "largest_tested_inscribed_solid_block_minimum_width_lower_bound",
        "tested_block_widths_mm": [w * pitch for w in widths],
        "minimum_lower_bound": float(values.min()) if occupied else 0.,
        "p05_lower_bound": float(np.quantile(values, .05)) if occupied else 0.,
        "median_lower_bound": float(np.median(values)) if occupied else 0.,
        "maximum_lower_bound": float(values.max()) if occupied else 0.,
        "is_minimum_edt": False,
        "uncertainty": "conservative for oblique or curved features; exported mesh needs independent checks",
    }

    neck_scales = []
    neck_vanished_count = 0
    significant_cells = max(1, int(math.ceil(wall ** 3 / pitch ** 3)))
    for factor in (.75, 1., 1.25):
        requested = neck * factor
        width = max(1, int(math.ceil(requested / pitch - 1e-12)))
        eroded, neck_cover = _block_cover(v, width)
        eroded_labels, eroded_count = ndi.label(eroded, FACE6)
        sizes = np.bincount(eroded_labels.ravel())[1:]
        significant = int(np.count_nonzero(sizes >= significant_cells))
        # Opening back to the original grid detects a whole wall-valid rod that
        # disappears, which a surviving-component count alone cannot detect.
        lost_labels, lost_count = ndi.label(cover & ~neck_cover, FACE6)
        lost_boxes = ndi.find_objects(lost_labels)
        adjacent_to_neck_core = ndi.binary_dilation(neck_cover, FACE6)
        lost_regions = []
        for region in range(1, lost_count + 1):
            box = lost_boxes[region - 1]
            mask = lost_labels[box] == region
            cells = int(mask.sum())
            extents = np.array([b.stop - b.start for b in box], float) * pitch
            attached = bool(np.any(mask & adjacent_to_neck_core[box]))
            long_feature = bool(extents.max() >= 2 * neck and cells * pitch ** 3 >= wall ** 3)
            classification = ("long_neck_scale_vanished_appendage" if long_feature
                              else "neck_scale_terminal_or_sampling_uncertain")
            lost_regions.append({"id": region, "voxels": cells,
                                 "volume_mm3": cells * pitch ** 3,
                                 "extent_mm": extents.tolist(),
                                 "bounds_ijk": [[b.start for b in box], [b.stop for b in box]],
                                 "attached_to_neck_certified_material": attached,
                                 "classification": classification,
                                 "fully_uncovered_after_neck_opening": True})
            if factor == 1.:
                neck_vanished_count += int(attached)
                if long_feature:
                    reasons.append("VANISHED_NECK_APPENDAGE")
                else:
                    review.append("NECK_SCALE_TERMINAL_OR_SAMPLING_REVIEW")
        neck_scales.append({
            "requested_width_mm": requested, "tested_block_width_mm": width * pitch,
            "surviving_components": int(eroded_count), "significant_components": significant,
            "surviving_voxels": int(eroded.sum()),
            "component_volumes_mm3": sorted((sizes * pitch ** 3).tolist(), reverse=True),
            "vanished_wall_valid_regions": lost_regions,
            "vanished_wall_valid_voxels": int(np.count_nonzero(lost_labels)),
        })
        if factor <= 1. and significant > 1 and "NECK_SEPARATION" not in reasons:
            reasons.append("NECK_SEPARATION")
        if factor == 1. and occupied and not eroded.any():
            reasons.append("NO_MINIMUM_NECK_CORE")
        if factor <= 1. and eroded_count > significant and eroded_count > 1:
            review.append("SMALL_ERODED_LOBE_REVIEW")
    reasons = list(dict.fromkeys(reasons))
    review = list(dict.fromkeys(review))
    return {
        "passed": not reasons and not review,
        "status": "FAIL" if reasons else "REVIEW_REQUIRED" if review else "PASS",
        "reasons": reasons, "review_required": bool(review), "review_reasons": review,
        "pitch_mm": pitch, "minimum_wall_mm": wall, "minimum_neck_mm": neck,
        "occupied_voxels": occupied, "component_count_6": int(components),
        "thin_voxels": int(thin.sum()), "thin_regions": thin_regions,
        "vanished_appendage_count": sum(r["fully_vanished_under_wall_erosion"] for r in thin_regions),
        "neck_vanished_appendage_count": neck_vanished_count,
        "local_thickness_mm": local_summary, "neck_scales": neck_scales,
        "physical_print_validated": False, "structural_validation": "NOT_REQUESTED",
        "scope": "digital_occupied_cell_geometry_only_not_FEM_or_slicer_certification",
    }
