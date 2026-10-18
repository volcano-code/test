"""Behavioral contracts for v3 DIC bands and symmetric contour metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from shadowforge_pipeline.soft_voxel_v3.contours import (
    ContourBands,
    boundary_midpoints,
    make_contour_bands,
    symmetric_contour_metrics,
    validate_contour_bands,
)


def test_single_pixel_boundary_uses_four_half_pixel_midpoints() -> None:
    """Break caught: border transitions are omitted or placed at pixel centers."""

    points = boundary_midpoints(np.ones((1, 1), dtype=bool))

    expected = {(-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5)}
    assert {tuple(row) for row in points} == expected


def test_identical_masks_have_zero_symmetric_distance() -> None:
    """Break caught: either contour direction contributes nonzero self-distance."""

    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True

    result = symmetric_contour_metrics(mask, mask)

    assert result.mscd_px == 0.0
    assert result.p95_symmetric_distance_px == 0.0
    assert result.invalid_predicted_boundary is False
    assert result.invalid_target_boundary is False


def test_empty_prediction_gets_image_diagonal_penalty() -> None:
    """Break caught: missing predicted contours are silently excluded or under-penalized."""

    target = np.zeros((3, 4), dtype=bool)
    target[1, 2] = True

    result = symmetric_contour_metrics(np.zeros_like(target), target)

    assert result.mscd_px == 5.0
    assert result.p95_symmetric_distance_px == 5.0
    assert result.invalid_predicted_boundary is True
    assert result.invalid_target_boundary is False


def test_empty_target_boundary_fails_preflight() -> None:
    """Break caught: an invalid experiment target is treated as a valid zero contour."""

    with pytest.raises(ValueError, match="target boundary"):
        symmetric_contour_metrics(np.zeros((3, 4), dtype=bool), np.zeros((3, 4), dtype=bool))


def test_p95_assigns_each_direction_equal_mass_despite_unequal_counts() -> None:
    """Break caught: pooled sample-count weighting replaces equal directional mass."""

    prediction = np.zeros((4, 4), dtype=bool)
    prediction[0, 0] = True
    target = np.zeros((4, 4), dtype=bool)
    target[0, 1] = True
    target[0, 3] = True
    target[1, 0] = True

    result = symmetric_contour_metrics(prediction, target)

    # The 4 prediction-to-target samples weigh 1/8 each, while the 12 reverse
    # samples weigh 1/24 each.  The hand-sorted weighted CDF first reaches 95%
    # at sqrt(6.5); pooling all 16 samples instead would incorrectly return 3.
    assert result.p95_symmetric_distance_px == pytest.approx(math.sqrt(6.5))


def test_contour_bands_include_axial_radius_and_exclude_diagonal_beyond_radius() -> None:
    """Break caught: DIC bands use a dilation/Manhattan distance rather than exact EDT."""

    target = torch.zeros((1, 7, 7), dtype=torch.bool)
    target[0, 3, 3] = True

    bands = make_contour_bands(target)

    assert bands.outer[0, 3, 5].item() is True
    assert bands.outer[0, 4, 5].item() is False
    assert bands.inner[0, 3, 3].item() is True


def test_contour_band_preflight_requires_nonempty_independent_bands_per_view() -> None:
    """Break caught: empty DIC support in one view reaches optimization."""

    target = torch.zeros((2, 7, 7), dtype=torch.bool)
    target[:, 2:5, 2:5] = True
    valid = make_contour_bands(target)
    validate_contour_bands(valid)

    invalid = ContourBands(inner=valid.inner, outer=valid.outer.clone())
    invalid.outer[1] = False
    with pytest.raises(ValueError, match="outer"):
        validate_contour_bands(invalid)

