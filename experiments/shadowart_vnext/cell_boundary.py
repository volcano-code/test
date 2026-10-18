"""Exact triangle boundaries of occupied XYZ grid cells.

This module is intentionally a diagnostic conversion, not a mesh repair step.
It emits each exposed cell face at its literal grid corners and preserves
edge/corner-only contacts, cavities, and their orientations.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh


# Counter-clockwise corner orders when viewed from outside the occupied cell.
_FACE_CORNERS = {
    (0, -1): ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
    (0, +1): ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
    (1, -1): ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
    (1, +1): ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
    (2, -1): ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
    (2, +1): ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
}


def _validate(volume_xyz, extent):
    """Return the validated input and scalar extent without copying either."""
    if not isinstance(volume_xyz, np.ndarray):
        raise ValueError('volume must be a boolean XYZ numpy array')
    if volume_xyz.dtype != np.bool_:
        raise ValueError('volume must have boolean dtype')
    if (volume_xyz.ndim != 3 or volume_xyz.shape[0] == 0 or
            len(set(volume_xyz.shape)) != 1):
        raise ValueError('volume must be a nonempty cubic XYZ array')
    if not volume_xyz.any():
        raise ValueError('volume must contain at least one occupied cell')
    if isinstance(extent, (bool, np.bool_)):
        raise ValueError('extent must be finite and positive')
    try:
        extent = float(extent)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('extent must be finite and positive') from error
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError('extent must be finite and positive')
    return volume_xyz, extent


def cell_boundary_mesh(volume_xyz, extent=1.7):
    """Build the exact exposed-face surface of a boolean cubic XYZ cell volume.

    Cell ``(i, j, k)`` has corners at
    ``-extent/2 + [i, j, k] * extent/resolution``.  A face is emitted only
    when its adjacent cell is empty or lies beyond the input boundary.  The
    implementation scans two-dimensional slices, retaining face coordinates
    and shared grid vertices only as they are needed; it deliberately does not
    allocate a resolution-cubed vertex lattice or repair nonmanifold contacts.
    """
    volume, extent = _validate(volume_xyz, extent)
    resolution = volume.shape[0]
    vertices_by_corner = {}
    corners = []
    faces = []

    def vertex_index(corner):
        index = vertices_by_corner.get(corner)
        if index is None:
            index = len(corners)
            vertices_by_corner[corner] = index
            corners.append(corner)
        return index

    def emit(cell, axis, side):
        quad = [vertex_index(tuple(cell[d] + offset[d] for d in range(3)))
                for offset in _FACE_CORNERS[(axis, side)]]
        faces.append((quad[0], quad[1], quad[2]))
        faces.append((quad[0], quad[2], quad[3]))

    # At any point only a pair of 2-D volume slices plus coordinates for faces
    # already known to be exposed are materialized.  This also handles each
    # outer boundary without a padded resolution-cubed copy.
    for axis in range(3):
        other_axes = tuple(index for index in range(3) if index != axis)
        for index in range(resolution):
            current = np.take(volume, index, axis)
            if index == 0:
                low = current
            else:
                low = current & ~np.take(volume, index - 1, axis)
            if index == resolution - 1:
                high = current
            else:
                high = current & ~np.take(volume, index + 1, axis)
            for side, exposed in ((-1, low), (+1, high)):
                for a, b in np.argwhere(exposed):
                    cell = [0, 0, 0]
                    cell[axis] = index
                    cell[other_axes[0]] = int(a)
                    cell[other_axes[1]] = int(b)
                    emit(cell, axis, side)

    grid_corners = np.asarray(corners, dtype=np.float64)
    vertices = -extent / 2 + grid_corners * (extent / resolution)
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64),
                           process=False, validate=False)
