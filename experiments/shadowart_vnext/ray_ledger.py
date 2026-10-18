"""Exact occupied-cell ray counts for simultaneous local material transactions."""
from dataclasses import dataclass

import numpy as np

from .manufacturing import _volume
from .rendering import _axis_map, _key, _sizes, iter_ray_segments


@dataclass(frozen=True)
class _Camera:
    origin: tuple
    right: tuple
    up: tuple
    direction: tuple
    scale: float


class RayLedger:
    def __init__(self, volume, cameras, image_size, extent=1.7):
        self.volume = _volume(volume).copy()
        self.volume.flags.writeable = False
        self.resolution = self.volume.shape[0]
        _sizes(image_size, self.resolution, extent)
        self.size, self.extent = int(image_size), float(extent)
        self.cameras = tuple(_Camera(*_key(c)) for c in cameras)
        if not self.cameras:
            raise ValueError('at least one camera required')
        self.maps = [_axis_map(_key(c), self.size, self.resolution, self.extent) for c in self.cameras]
        self.counts = self._count(self.volume)
        self.counts.flags.writeable = False

    @property
    def prediction(self):
        return self.counts > 0

    def _mask(self, mask):
        a = np.asarray(mask)
        if a.dtype != np.bool_ or a.shape != self.volume.shape:
            raise ValueError('boolean masks matching XYZ occupancy required')
        return a

    def _count(self, volume):
        views = []
        for camera, mapping in zip(self.cameras, self.maps):
            counts = np.zeros(self.size**2, np.int32)
            if mapping is not None:
                axis, indices, valid = mapping
                counts[valid] = volume.sum(axis=axis, dtype=np.int32).ravel()[indices[valid]]
            elif volume.any():
                flat = volume.ravel()
                for chunk in iter_ray_segments(camera, self.size, self.resolution, self.extent):
                    counts[chunk.start:chunk.stop] = np.bincount(
                        chunk.ray_indices[flat[chunk.voxel_indices]], minlength=chunk.stop-chunk.start)
            views.append(counts.reshape(self.size, self.size))
        return np.stack(views)

    def evaluate(self, add, remove):
        add, remove = self._mask(add), self._mask(remove)
        if np.any(add & remove):
            raise ValueError('add and remove masks must be disjoint')
        new = add & ~self.volume
        deleted = remove & self.volume
        counts = self.counts + self._count(new) - self._count(deleted)
        if np.any(counts < 0):
            raise RuntimeError('NEGATIVE_OCCUPIED_RAY_COUNT')
        return (self.volume | new) & ~deleted, counts

    def backproject(self, weights):
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != self.counts.shape or not np.isfinite(w).all():
            raise ValueError('finite weights matching view rasters required')
        out = np.zeros(self.volume.shape, np.float64)
        for i, (camera, mapping) in enumerate(zip(self.cameras, self.maps)):
            if mapping is not None:
                axis, indices, valid = mapping
                plane = np.bincount(indices[valid], weights=w[i].ravel()[valid],
                                    minlength=self.resolution**2).reshape((self.resolution,)*2)
                out += np.expand_dims(plane, axis)
            else:
                for chunk in iter_ray_segments(camera, self.size, self.resolution, self.extent):
                    values = w[i].ravel()[chunk.start:chunk.stop][chunk.ray_indices]
                    np.add.at(out.ravel(), chunk.voxel_indices, values)
        return out
