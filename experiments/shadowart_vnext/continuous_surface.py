"""Fixed-level native probability surface; no threshold search or grid upsample."""
import numpy as np

from .geometry import _sdf_mesh
from .manufacturing import _positive


def probability_surface(probability_xyz, extent=1.7):
    """Interpolate p=.5 between native cell centers with zero-probability padding.

    The interpolated scalar is NOT a signed distance. Original probabilities
    and cavities are not rewritten. This candidate must independently pass
    topology, serialization, revoxelization, projection and manufacturing QA.
    """
    p = np.asarray(probability_xyz)
    extent = _positive(extent, 'extent')
    if (p.ndim != 3 or len(set(p.shape)) != 1 or not p.size or p.dtype.kind != 'f'
            or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)):
        raise ValueError('finite floating [0,1] cubic native probabilities required')
    if not np.any(p > .5):
        raise ValueError('no positive-volume interior above fixed probability .5')
    field = np.pad(p.astype(np.float64) - .5, 1, constant_values=-.5)
    return _sdf_mesh(field, extent, p.shape[0])
