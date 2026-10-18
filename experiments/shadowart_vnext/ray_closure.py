"""Diagnostic root-connected filling inside the source's identical shadows.

Graphics-only candidate generation from immutable soft occupancy. This is NOT
global hole filling: all source enclosed voids are forbidden. Nothing is removed
and disconnected copies of hidden material are not introduced. Final mesh and
manufacturing QA, not this shadow identity, decide whether it is usable.
"""
import hashlib
import numpy as np
from scipy import ndimage as ndi
from .rendering import _axis_map, _key, _sizes
from .manufacturing import _block_cover


def shadow_closure(volume_xyz,cameras,extent=1.7,*,minimum_block_width=1):
    v=np.asarray(volume_xyz)
    if v.dtype!=np.bool_ or v.ndim!=3 or len(set(v.shape))!=1 or not v.any():
        raise ValueError('nonempty cubic boolean occupancy required')
    r=v.shape[0]; _sizes(2,r,extent)
    if type(minimum_block_width)!=int or not 1<=minimum_block_width<=r:
        raise ValueError('minimum_block_width must be an integer within the grid')
    if not cameras:
        raise ValueError('cameras required')
    allowed=np.ones_like(v)
    axes=set()
    for camera in cameras:
        mapping=_axis_map(_key(camera),2,r,extent)
        if mapping is None:
            raise ValueError('cardinal cameras outside volume required for shadow closure')
        axis=mapping[0]; axes.add(axis)
        allowed &= np.expand_dims(v.any(axis=axis),axis)
    # Flood only exterior air; no all-cavity fill operation.
    border=np.zeros_like(v)
    for axis in range(3):
        s=[slice(None)]*3; s[axis]=0; border[tuple(s)]=True
        s[axis]=-1; border[tuple(s)]=True
    exterior=ndi.binary_propagation(border & ~v,mask=~v)
    protected=~v & ~exterior
    allowed &= ~protected
    if minimum_block_width>1:
        # All added cells must belong to full solid blocks, not arbitrary hidden
        # sheets. Original material remains; this is not morphological removal.
        _,cover=_block_cover(allowed,minimum_block_width)
        allowed=v | cover
    source_labels,source_count=ndi.label(v)
    sizes=np.bincount(source_labels.ravel()); sizes[0]=0
    root=source_labels==int(sizes.argmax())
    reachable=ndi.binary_propagation(root,mask=allowed)
    out=v | reachable
    for axis in axes:
        if not np.array_equal(out.any(axis=axis),v.any(axis=axis)):
            raise RuntimeError('shadow closure changed a source column')
    added=np.flatnonzero(out & ~v)
    trace={'algorithm':'source_shadow_equivalent_root_closure_v1',
           'added_voxels':len(added),'removed_voxels':0,
           'added_indices_sha256':hashlib.sha256(added.astype('<i8').tobytes()).hexdigest(),
           'initial_components_6':int(source_count),
           'minimum_block_width':minimum_block_width,
           'final_components_6':int(ndi.label(out)[1]),'source_void_voxels':int(protected.sum()),
           'source_occupancy_mutated':False,'projection_identity_axes':sorted(axes),
           'publication_eligible':False,'structural_status':'NOT_REQUESTED',
           'manufacturing_status':'NOT_VERIFIED','global_hole_fill':False}
    return out,trace
