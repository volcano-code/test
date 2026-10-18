"""Physical orthographic projections of XYZ cells and independent triangle meshes.

Cells use half-open spatial ownership on parallel grid-plane rays. Nonparallel
edge/corner contacts of zero length do not count. Pixel centers are point rays;
there is no antialiasing or image-dependent opacity normalization.
"""
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import torch


@dataclass(frozen=True)
class RaySegments:
    start: int
    stop: int
    ray_indices: np.ndarray
    voxel_indices: np.ndarray
    lengths: np.ndarray


# CPU traversal cache is bounded independently of resolution/view count.
_SEGMENT_CACHE = OrderedDict()
_CACHE_BYTES = 0
_CACHE_LIMIT = 128 * 1024**2


def _key(camera):
    vectors = [np.asarray(getattr(camera, name), dtype=np.float64)
               for name in ('origin', 'right', 'up', 'direction')]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
        raise ValueError('camera vectors must be finite XYZ triples')
    origin, right, up, direction = vectors
    frame = np.stack([right, up, direction])
    if not np.allclose(frame @ frame.T, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError('camera frame must be orthonormal')
    if not np.allclose(np.cross(right, up), -direction, atol=1e-8, rtol=0):
        raise ValueError('camera requires right cross up = -direction')
    scale = float(camera.scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('camera scale must be finite and positive')
    return tuple(tuple(v) for v in vectors) + (scale,)


def _sizes(image_size, resolution, extent):
    if int(image_size) != image_size or image_size <= 0:
        raise ValueError('image_size must be a positive integer')
    if int(resolution) != resolution or resolution <= 0:
        raise ValueError('resolution must be a positive integer')
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError('extent must be finite and positive')


def _volume_resolution(volume):
    if len(volume.shape) != 3 or len(set(volume.shape)) != 1 or volume.shape[0] == 0:
        raise ValueError('volume must be a nonempty cubic XYZ array')
    return volume.shape[0]


def _origins(key, size, start, stop):
    origin, right, up, direction, scale = key
    pixels = np.arange(start, stop)
    x = ((pixels % size + .5) / size - .5) * scale
    y = (.5 - (pixels // size + .5) / size) * scale
    return np.asarray(origin) + x[:, None]*right + y[:, None]*up


def _parallel_cell_indices(coordinates, resolution, extent):
    """Half-open cell ownership on numerically parallel grid-plane rays.

    Trigonometric cardinal frames contain components such as cos(pi/2), which
    move opposite source planes by opposite roundoff amounts. Snap only these
    parallel coordinates within float64 rounding scale to the exact grid plane;
    a plane belongs to its upper-index cell, independently of ray direction.
    Nonparallel interval midpoints are never snapped by this helper.
    """
    grid=(coordinates+extent/2)*(resolution/extent)
    nearest=np.rint(grid)
    tolerance=32*np.finfo(np.float64).eps*np.maximum(resolution,np.abs(grid))
    stable=np.where(np.abs(grid-nearest)<=tolerance,nearest,grid)
    return np.floor(stable).astype(np.int64)


@lru_cache(maxsize=32)
def _axis_map(key, size, resolution, extent):
    d = np.asarray(key[3]); axis = int(np.argmax(np.abs(d)))
    # Only numerically exact cardinal directions use the reduction path.
    if np.max(np.abs(np.delete(d, axis))) > 1e-14:
        return None
    source = key[0][axis]
    if (d[axis] > 0 and source > -extent/2) or (d[axis] < 0 and source < extent/2):
        return None  # Cameras inside/behind the cube need clipped forward rays.
    others = [a for a in range(3) if a != axis]
    origins = _origins(key, size, 0, size*size)
    ij = _parallel_cell_indices(origins[:,others],resolution,extent)
    valid = ((ij >= 0) & (ij < resolution)).all(axis=1)
    indices = np.clip(ij[:, 0], 0, resolution-1)*resolution + np.clip(ij[:, 1], 0, resolution-1)
    indices.flags.writeable = False; valid.flags.writeable = False
    return axis, indices, valid


def _segments(key, size, resolution, extent, start, stop):
    """Siddon traversal: sort every crossed grid plane, integrate intervals."""
    global _CACHE_BYTES
    cache_key = (key, size, resolution, extent, start, stop)
    if cache_key in _SEGMENT_CACHE:
        _SEGMENT_CACHE.move_to_end(cache_key)
        return _SEGMENT_CACHE[cache_key]
    origins = _origins(key, size, start, stop)
    direction = np.asarray(key[3]); count = stop-start
    near = np.zeros(count); far = np.full(count, np.inf)
    crossings = []
    parallel_columns = {}
    planes = np.linspace(-extent/2, extent/2, resolution+1)
    for axis in range(3):
        if abs(direction[axis]) <= 1e-14:
            column = _parallel_cell_indices(origins[:,axis],resolution,extent)
            parallel_columns[axis] = column
            inside = (column >= 0) & (column < resolution)
            far[~inside] = -1
        else:
            times = (planes[None,:]-origins[:,axis,None])/direction[axis]
            near = np.maximum(near, np.minimum(times[:,0], times[:,-1]))
            far = np.minimum(far, np.maximum(times[:,0], times[:,-1]))
            crossings.append(times[:,1:-1])
    active = far > near
    near[~active] = 0.; far[~active] = 0.
    times = np.concatenate([near[:,None], far[:,None]]+crossings, axis=1)
    times = np.minimum(np.maximum(times, near[:,None]), far[:,None])
    times.sort(axis=1)
    lengths = np.diff(times, axis=1)
    # Tiny roundoff slivers at coincident plane crossings are zero contacts.
    good = lengths > np.finfo(np.float64).eps * max(1., extent)*16
    rays, intervals = np.nonzero(good)
    mids = (times[rays,intervals]+times[rays,intervals+1])*.5
    positions = origins[rays]+mids[:,None]*direction
    ijk = np.floor((positions+extent/2)*(resolution/extent)).astype(np.int64)
    for axis,column in parallel_columns.items():
        # A coordinate declared parallel in the slab traversal must not drift
        # between cells due to tiny ignored direction components in midpoint math.
        ijk[:,axis] = column[rays]
    ijk = np.clip(ijk, 0, resolution-1)
    indices = (ijk[:,0]*resolution+ijk[:,1])*resolution+ijk[:,2]
    arrays = (rays.astype(np.int32), indices.astype(np.int64), lengths[good])
    for array in arrays:
        array.flags.writeable = False
    result = RaySegments(start, stop, *arrays)
    nbytes = sum(a.nbytes for a in arrays)
    while _SEGMENT_CACHE and _CACHE_BYTES+nbytes > _CACHE_LIMIT:
        _, old = _SEGMENT_CACHE.popitem(last=False)
        _CACHE_BYTES -= old.ray_indices.nbytes+old.voxel_indices.nbytes+old.lengths.nbytes
    if nbytes <= _CACHE_LIMIT:
        _SEGMENT_CACHE[cache_key] = result
        _CACHE_BYTES += nbytes
    return result


def iter_ray_segments(camera, image_size, resolution, extent=1.7, chunk_rays=2048):
    """Yield exact positive-length segments; indices are chunk-local ray/XYZ-flat.

    Callers must not mutate the read-only arrays. The 128 MiB LRU cache stores
    geometry only. Large maps are streamed; they are never materialized whole.
    """
    _sizes(image_size, resolution, extent)
    if int(chunk_rays) != chunk_rays or chunk_rays <= 0:
        raise ValueError('chunk_rays must be a positive integer')
    key = _key(camera)
    for start in range(0, image_size*image_size, chunk_rays):
        yield _segments(key, image_size, resolution, extent, start,
                        min(start+chunk_rays, image_size*image_size))


def render_binary(volume_xyz, cameras, image_size, extent=1.7):
    volume = np.asarray(volume_xyz)
    resolution = _volume_resolution(volume); _sizes(image_size, resolution, extent)
    if volume.dtype != np.bool_:
        raise ValueError('binary volume must have boolean dtype')
    views = []
    for camera in cameras:
        mapping = _axis_map(_key(camera), image_size, resolution, extent)
        if mapping is not None:
            axis, indices, valid = mapping
            mask = volume.any(axis=axis).reshape(-1)[indices] & valid
        else:
            mask = np.zeros(image_size*image_size, bool)
            flat = volume.reshape(-1)
            for chunk in iter_ray_segments(camera,image_size,resolution,extent):
                hit = flat[chunk.voxel_indices]
                mask[chunk.start + chunk.ray_indices[hit]] = True
        views.append(mask.reshape(image_size,image_size))
    return np.stack(views) if views else np.zeros((0,image_size,image_size), bool)


class _ObliqueDepth(torch.autograd.Function):
    """Linear ray integral with a streamed transpose in backward.

    Keeping every gathered index/length in PyTorch's graph can consume gigabytes
    at native resolution. Geometry is recomputed or read from the bounded CPU
    cache; only the output opacity needs to survive until backward.
    """
    @staticmethod
    def forward(ctx, density, camera, image_size, extent):
        ctx.camera = camera; ctx.image_size = image_size; ctx.extent = extent
        ctx.resolution = density.shape[0]
        parts=[]; flat=density.reshape(-1)
        for chunk in iter_ray_segments(camera,image_size,ctx.resolution,extent):
            indices=torch.tensor(chunk.voxel_indices,device=density.device)
            rays=torch.tensor(chunk.ray_indices,device=density.device,dtype=torch.long)
            lengths=torch.tensor(chunk.lengths,device=density.device,dtype=density.dtype)
            parts.append(density.new_zeros(chunk.stop-chunk.start).index_add_(0,rays,flat[indices]*lengths))
        return torch.cat(parts)

    @staticmethod
    def backward(ctx, grad_output):
        gradient=grad_output.new_zeros(ctx.resolution**3)
        for chunk in iter_ray_segments(ctx.camera,ctx.image_size,ctx.resolution,ctx.extent):
            indices=torch.tensor(chunk.voxel_indices,device=gradient.device)
            rays=torch.tensor(chunk.ray_indices,device=gradient.device,dtype=torch.long)
            lengths=torch.tensor(chunk.lengths,device=gradient.device,dtype=gradient.dtype)
            gradient.index_add_(0,indices,grad_output[chunk.start:chunk.stop][rays]*lengths)
        return gradient.reshape((ctx.resolution,)*3),None,None,None


def render_optical(density_xyz, cameras, image_size, extent=1.7):
    """Differentiable Beer–Lambert opacity, sigma in inverse canonical length.

    Density is required to be floating, finite and nonnegative by the caller.
    No per-step full-volume validation/synchronization is done in this hot path.
    """
    if not isinstance(density_xyz, torch.Tensor) or not density_xyz.is_floating_point():
        raise ValueError('density must be a floating torch tensor')
    density = density_xyz; resolution = _volume_resolution(density)
    _sizes(image_size, resolution, extent)
    views = []
    for camera in cameras:
        mapping = _axis_map(_key(camera), image_size, resolution, extent)
        if mapping is not None:
            axis, indices, valid = mapping
            indices_t = torch.tensor(indices, device=density.device)
            valid_t = torch.tensor(valid, device=density.device, dtype=density.dtype)
            tau = density.sum(dim=axis).reshape(-1)[indices_t]*(extent/resolution)*valid_t
        else:
            tau = _ObliqueDepth.apply(density,camera,image_size,extent)
        views.append((-torch.expm1(-tau)).reshape(image_size,image_size))
    return torch.stack(views) if views else density.new_empty((0,image_size,image_size))


def visual_hull(targets_bool, cameras, resolution, extent=1.7):
    """Maximal occupied-cell set with zero false positives on the given rays.

    Unobserved cells remain occupied. This does NOT certify global IoU
    feasibility: foreground misses may persist, and other discretizations differ.
    """
    targets = np.asarray(targets_bool); cameras = tuple(cameras)
    if targets.ndim != 3 or targets.shape[0] != len(cameras) or targets.shape[1] != targets.shape[2] or targets.dtype != np.bool_:
        raise ValueError('targets must be boolean [views,H,H] matching cameras')
    size = targets.shape[1]; _sizes(size,resolution,extent)
    hull = np.ones((resolution,)*3, bool)
    for mask,camera in zip(targets,cameras):
        mapping = _axis_map(_key(camera),size,resolution,extent)
        if mapping is not None:
            axis, indices, valid = mapping
            keep = np.ones(resolution*resolution,bool)
            keep[indices[valid & ~mask.reshape(-1)]] = False
            hull &= np.expand_dims(keep.reshape(resolution,resolution),axis=axis)
        else:
            flat = hull.reshape(-1); target = mask.reshape(-1)
            for chunk in iter_ray_segments(camera,size,resolution,extent):
                background = ~target[chunk.start+chunk.ray_indices]
                flat[chunk.voxel_indices[background]] = False
    return hull


def _mesh_arrays(mesh_canonical):
    vertices = np.asarray(mesh_canonical.vertices, dtype=np.float64)
    faces = np.asarray(mesh_canonical.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('mesh vertices must be finite XYZ coordinates')
    if faces.size == 0:
        faces = np.empty((0,3),dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or (faces.size and (faces.min()<0 or faces.max()>=len(vertices))):
        raise ValueError('mesh requires valid triangle vertex indices')
    return vertices, faces


def _triangle_pixel_hits(projected, faces, image_size):
    """Yield pixel id, interpolated third coordinate, and oriented crossing sign."""
    for face_start in range(0,len(faces),8192):
        triangles = projected[faces[face_start:face_start+8192]]
        low = np.maximum(np.ceil(triangles[:,:,:2].min(axis=1)),0).astype(np.int64)
        high = np.minimum(np.floor(triangles[:,:,:2].max(axis=1)),image_size-1).astype(np.int64)
        widths = np.maximum(high-low+1,0)
        counts = widths[:,0]*widths[:,1]
        visible = counts>0
        triangles,low,widths,counts = triangles[visible],low[visible],widths[visible],counts[visible]
        begin=0
        while begin < len(counts):
            cumulative = np.cumsum(counts[begin:])
            end = begin+max(1,int(np.searchsorted(cumulative,500_000,side='right')))
            batch = triangles[begin:end]; n=counts[begin:end]
            owners = np.repeat(np.arange(len(n)),n)
            offsets = np.arange(int(n.sum()))-np.repeat(np.cumsum(n)-n,n)
            xy = low[begin:end][owners].copy()
            xy[:,0] += offsets % widths[begin:end,0][owners]
            xy[:,1] += offsets // widths[begin:end,0][owners]
            a=batch[:,0,:2]; b=batch[:,1,:2]-a; c=batch[:,2,:2]-a
            det=b[:,0]*c[:,1]-b[:,1]*c[:,0]
            delta=xy-a[owners]; divisor=det[owners]
            nonzero=np.abs(divisor)>1e-20
            safe=np.where(nonzero,divisor,1.)
            u=(delta[:,0]*c[owners,1]-delta[:,1]*c[owners,0])/safe
            v=(b[owners,0]*delta[:,1]-b[owners,1]*delta[:,0])/safe
            z=batch[owners,0,2]+u*(batch[owners,1,2]-batch[owners,0,2])+v*(batch[owners,2,2]-batch[owners,0,2])
            hit=nonzero & (u>=-1e-12) & (v>=-1e-12) & (u+v<=1+1e-12)
            yield xy[hit,1]*image_size+xy[hit,0], z[hit], np.sign(divisor[hit]).astype(np.int8)
            begin=end


def render_mesh(mesh_canonical, cameras, image_size):
    """Independent two-sided orthographic triangle rasterization at pixel centers.

    Triangle bounds generate candidates only; barycentric inclusion and forward
    ray distance decide actual hits. Nested cavity surfaces need no special fill.
    A mesh must expose finite vertices[N,3] and triangular faces[M,3].
    """
    _sizes(image_size,1,1.)
    vertices, faces = _mesh_arrays(mesh_canonical)
    views = []
    for camera in cameras:
        origin,right,up,direction,scale = _key(camera)
        relative = vertices-np.asarray(origin)
        projected = np.column_stack([(relative@right/scale+.5)*image_size-.5,
                                     (.5-relative@up/scale)*image_size-.5,
                                     relative@direction])
        mask = np.zeros(image_size*image_size,bool)
        for pixels, depth, _ in _triangle_pixel_hits(projected,faces,image_size):
            mask[pixels[depth>=0]]=True
        views.append(mask.reshape(image_size,image_size))
    return np.stack(views) if views else np.zeros((0,image_size,image_size),bool)


def voxelize_mesh(mesh_canonical, resolution, extent=1.7):
    """Deterministic odd/even mesh membership at all canonical XYZ grid centers.

    Requires closed, consistently wound components, but their relative winding
    does not affect nested-cavity parity. Surface depths are deduplicated, never
    globally hole-filled. Mixed crossing signs at one depth are tangent contacts
    and do not toggle parity. Open surfaces/odd intersection counts fail closed.
    Exactly-on-surface z centers use lower-inclusive/upper-exclusive intervals.
    Coincident/intersecting shells are not a supported manufacturing input.
    """
    _sizes(resolution,resolution,extent)
    vertices, faces = _mesh_arrays(mesh_canonical)
    if not len(faces):
        return np.zeros((resolution,)*3,bool)
    if not getattr(mesh_canonical,'is_watertight',False):
        raise ValueError('mesh voxelization requires a closed watertight mesh')
    if not getattr(mesh_canonical,'is_winding_consistent',False):
        raise ValueError('mesh voxelization requires consistently wound components')
    projected=np.column_stack([(vertices[:,0]/extent+.5)*resolution-.5,
                               (vertices[:,1]/extent+.5)*resolution-.5,vertices[:,2]])
    batches=list(_triangle_pixel_hits(projected,faces,resolution))
    if not batches or not any(len(b[0]) for b in batches):
        return np.zeros((resolution,)*3,bool)
    pixels=np.concatenate([b[0] for b in batches]); depths=np.concatenate([b[1] for b in batches])
    signs=np.concatenate([b[2] for b in batches]); del batches
    order=np.lexsort((depths,pixels)); pixels=pixels[order];depths=depths[order];signs=signs[order]
    tolerance=max(1.,extent)*1e-10
    starts=np.r_[True,(pixels[1:]!=pixels[:-1]) | (np.diff(depths)>tolerance)]
    groups=np.flatnonzero(starts)
    sign_min=np.minimum.reduceat(signs,groups); sign_max=np.maximum.reduceat(signs,groups)
    crossing=sign_min==sign_max
    pixels=pixels[groups][crossing];depths=depths[groups][crossing]
    counts=np.bincount(pixels,minlength=resolution**2)
    if np.any(counts%2):
        raise ValueError('ambiguous odd mesh intersection count; closed parity cannot be established')
    # Toggle each column at the first center >= a crossing, then prefix XOR.
    # Row order here is Y,X; transpose once to the shared XYZ volume convention.
    first_center=-extent/2+extent/(2*resolution)
    indices=np.ceil((depths-first_center)/(extent/resolution)).astype(np.int64)
    indices=np.clip(indices,0,resolution)
    changes=np.zeros((resolution*resolution,resolution+1),bool)
    np.logical_xor.at(changes,(pixels,indices),True)
    filled=np.logical_xor.accumulate(changes,axis=1)[:,:resolution]
    return filled.reshape(resolution,resolution,resolution).transpose(1,0,2).copy()
