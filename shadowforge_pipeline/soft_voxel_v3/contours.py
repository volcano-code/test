"""DIC contour bands and half-pixel symmetric contour-distance diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import torch


@dataclass(frozen=True)
class ContourBands:
    """Independent foreground and background DIC supports, indexed by view."""

    inner: torch.Tensor
    outer: torch.Tensor


@dataclass(frozen=True)
class ContourMetrics:
    """Equal-direction contour distances and invalid-boundary diagnostics."""

    mscd_px: float
    p95_symmetric_distance_px: float
    invalid_predicted_boundary: bool
    invalid_target_boundary: bool


def _finite_nonnegative_radius(radius_px: float) -> float:
    try:
        radius = float(radius_px)
    except (TypeError, ValueError) as error:
        raise ValueError("radius_px must be a finite nonnegative scalar") from error
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius_px must be a finite nonnegative scalar")
    return radius


def _validate_band_tensor(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if tensor.dtype != torch.bool:
        raise ValueError(f"{name} must have boolean dtype")
    if tensor.ndim != 3 or any(size == 0 for size in tensor.shape):
        raise ValueError(f"{name} must have non-empty shape [views,H,W]")


def make_contour_bands(target_binary: torch.Tensor, radius_px: float = 2.0) -> ContourBands:
    """Build independent two-sided Euclidean DIC bands for every target view."""

    _validate_band_tensor("target_binary", target_binary)
    radius = _finite_nonnegative_radius(radius_px)
    target_cpu = target_binary.detach().to(device="cpu").numpy()

    inner = np.empty_like(target_cpu, dtype=bool)
    outer = np.empty_like(target_cpu, dtype=bool)
    for view_index, view in enumerate(target_cpu):
        inner[view_index] = view & (distance_transform_edt(view) <= radius)
        background = ~view
        outer[view_index] = background & (distance_transform_edt(background) <= radius)

    return ContourBands(
        inner=torch.from_numpy(inner).to(device=target_binary.device),
        outer=torch.from_numpy(outer).to(device=target_binary.device),
    )


def validate_contour_bands(bands: ContourBands) -> None:
    """Fail preflight when a DIC direction has no support in any view."""

    if not isinstance(bands, ContourBands):
        raise ValueError("bands must be a ContourBands instance")
    _validate_band_tensor("inner", bands.inner)
    _validate_band_tensor("outer", bands.outer)
    if bands.inner.shape != bands.outer.shape:
        raise ValueError("inner and outer bands must have the same shape")
    if bands.inner.device != bands.outer.device:
        raise ValueError("inner and outer bands must share a device")
    if not bands.inner.any(dim=(-2, -1)).all().item():
        raise ValueError("every view must have a non-empty inner contour band")
    if not bands.outer.any(dim=(-2, -1)).all().item():
        raise ValueError("every view must have a non-empty outer contour band")


def _mask_array(mask: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must have non-empty shape [H,W]")
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must have boolean dtype")
    return array


def boundary_midpoints(mask: np.ndarray) -> np.ndarray:
    """Return foreground/background edge midpoints in ``(x, y)`` coordinates."""

    binary = _mask_array(mask, "mask")
    height, width = binary.shape
    points: list[np.ndarray] = []

    rows = np.flatnonzero(binary[:, 0])
    points.append(np.column_stack((np.full(rows.size, -0.5), rows)))
    rows = np.flatnonzero(binary[:, -1])
    points.append(np.column_stack((np.full(rows.size, width - 0.5), rows)))
    columns = np.flatnonzero(binary[0, :])
    points.append(np.column_stack((columns, np.full(columns.size, -0.5))))
    columns = np.flatnonzero(binary[-1, :])
    points.append(np.column_stack((columns, np.full(columns.size, height - 0.5))))

    vertical_edges = binary[:, :-1] != binary[:, 1:]
    rows, columns = np.nonzero(vertical_edges)
    points.append(np.column_stack((columns + 0.5, rows)))
    horizontal_edges = binary[:-1, :] != binary[1:, :]
    rows, columns = np.nonzero(horizontal_edges)
    points.append(np.column_stack((columns, rows + 0.5)))

    nonempty = [point_set.astype(float, copy=False) for point_set in points if point_set.size]
    if not nonempty:
        return np.empty((0, 2), dtype=float)
    return np.concatenate(nonempty, axis=0)


def _directional_distances(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    return np.asarray(cKDTree(destination).query(source, k=1)[0], dtype=float)


def _equal_direction_p95(predicted_to_target: np.ndarray, target_to_predicted: np.ndarray) -> float:
    distances = np.concatenate((predicted_to_target, target_to_predicted))
    weights = np.concatenate(
        (
            np.full(predicted_to_target.size, 0.5 / predicted_to_target.size),
            np.full(target_to_predicted.size, 0.5 / target_to_predicted.size),
        )
    )
    order = np.argsort(distances, kind="stable")
    cumulative_weights = np.cumsum(weights[order])
    return float(distances[order][np.flatnonzero(cumulative_weights >= 0.95)[0]])


def symmetric_contour_metrics(prediction: np.ndarray, target: np.ndarray) -> ContourMetrics:
    """Compute MSCD and equal-direction weighted p95 from half-pixel contours."""

    predicted_mask = _mask_array(prediction, "prediction")
    target_mask = _mask_array(target, "target")
    if predicted_mask.shape != target_mask.shape:
        raise ValueError("prediction and target must have the same shape")

    target_points = boundary_midpoints(target_mask)
    if target_points.size == 0:
        raise ValueError("target boundary is empty")
    predicted_points = boundary_midpoints(predicted_mask)
    if predicted_points.size == 0:
        diagonal = float(math.hypot(*predicted_mask.shape))
        return ContourMetrics(
            mscd_px=diagonal,
            p95_symmetric_distance_px=diagonal,
            invalid_predicted_boundary=True,
            invalid_target_boundary=False,
        )

    predicted_to_target = _directional_distances(predicted_points, target_points)
    target_to_predicted = _directional_distances(target_points, predicted_points)
    return ContourMetrics(
        mscd_px=float(0.5 * predicted_to_target.mean() + 0.5 * target_to_predicted.mean()),
        p95_symmetric_distance_px=_equal_direction_p95(predicted_to_target, target_to_predicted),
        invalid_predicted_boundary=False,
        invalid_target_boundary=False,
    )
