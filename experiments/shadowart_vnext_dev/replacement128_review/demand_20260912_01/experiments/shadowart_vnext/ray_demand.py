"""Finite thick rooted primitives from missing foreground rays, not island shape."""
import hashlib
import time

import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP

from .geometry import _enclosed_voids, _thicken_mask
from .manufacturing import FACE6, _block_cover, _volume
from .ray_ledger import RayLedger
from .rendering import render_binary


def demand_primitives(base, guidance, demand, targets, cameras, *, extent=1.7,
                      width=2, max_primitives=64, protected=None):
    started=time.monotonic(); base=_volume(base).copy(); r=len(base)
    if type(width) is not int or not 1<=width<=r:
        raise ValueError('width must be integer in [1,resolution]')
    if type(max_primitives) is not int or not 0<=max_primitives<=256:
        raise ValueError('max_primitives must be integer in [0,256]')
    guidance=np.asarray(guidance,float)
    if guidance.shape!=base.shape or not np.isfinite(guidance).all() or np.any((guidance<0)|(guidance>1)):
        raise ValueError('guidance must match base in [0,1]')
    demand=np.asarray(demand); targets=np.asarray(targets)
    if (targets.ndim!=3 or targets.shape!=demand.shape or targets.shape[1]!=targets.shape[2]
            or not targets.shape[1] or not np.isin(targets,[0,1]).all()
            or not np.isin(demand,[0,1]).all()):
        raise ValueError('binary square target and demand rasters required')
    demand=demand.astype(bool); targets=targets.astype(bool)
    if np.any(demand & ~targets): raise ValueError('demand must be foreground')
    cameras=tuple(cameras)
    if len(cameras)!=len(targets): raise ValueError('camera count mismatch')
    ledger=RayLedger(base,cameras,targets.shape[-1],extent)
    if np.any(demand & ledger.prediction): raise ValueError('demand must be unoccluded by base')
    protected=_enclosed_voids(base) if protected is None else _volume(protected).copy()
    if protected.shape!=base.shape or np.any(protected & base): raise ValueError('invalid protected void')
    info={'algorithm':'full_ray_rooted_demand_family_v1','primitive_count':0,
          'max_primitives':max_primitives,'width_voxels':width,'demand_pixels':int(demand.sum()),
          'scope':'finite_whole_domain_endpoints_not_global_feasibility',
          'endpoint_attempts':0,'demand_uncovered_by_all_primitives':int(demand.sum())}
    primitives=[]; metadata=[]
    labels,_=ndi.label(base,FACE6); counts=np.bincount(labels.ravel()); counts[0]=0
    root=(labels==counts.argmax()) if counts.max() else np.zeros_like(base)
    anchors=_block_cover(root,width)[0]
    if not max_primitives or not demand.any() or not anchors.any():
        info.update(stop_reason='ZERO_BUDGET_OR_NO_DEMAND_OR_NO_THICK_ROOT',elapsed_seconds=time.monotonic()-started)
        return primitives,metadata,info
    def box_mean(values): return ndi.uniform_filter(values.astype(float),size=width,mode='constant',cval=0.)
    pixel_scale=max((ledger.size/r)**2 * len(cameras),1.)
    leakage=box_mean(ledger.backproject((~targets).astype(float)))/pixel_scale
    costs=1.+4.*leakage+.5*(1.-box_mean(guidance))
    legal=np.zeros_like(base)
    legal[(slice(width//2,r-(width-1)//2),)*3]=True
    legal &= ~ndi.maximum_filter(protected,size=width,mode='constant',cval=0)
    costs[~legal]=np.inf
    pathfinder=MCP(costs,fully_connected=False)
    distance,_=pathfinder.find_costs(np.argwhere(anchors & legal))
    available=legal & np.isfinite(distance)
    remaining=demand.copy(); all_covered=np.zeros_like(demand); seen=set()
    for attempt in range(max_primitives*4):
        info['endpoint_attempts']=attempt+1
        benefit=box_mean(ledger.backproject(remaining.astype(float)))
        score=np.divide(benefit,1.+distance,out=np.zeros_like(benefit),where=available)
        score[~available]=0.
        if not np.any(score>0):
            if np.array_equal(remaining,demand): break
            remaining=demand.copy(); continue
        endpoint=np.array(np.unravel_index(int(score.argmax()),base.shape))
        # Exclude nearby equivalent endpoints, while retaining alternative depths.
        lo=np.maximum(endpoint-width//2,0); hi=np.minimum(endpoint+width//2+1,r)
        available[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]=False
        path=np.asarray(pathfinder.traceback(tuple(endpoint)),int)
        centres=np.zeros_like(base); centres[tuple(path.T)]=True
        addition=_thicken_mask(centres,width)
        if np.any(addition & protected): raise RuntimeError('PATH_ENTERED_PROTECTED_VOID')
        indices=np.flatnonzero(addition & ~base)
        digest=hashlib.sha256(indices.tobytes()).hexdigest()
        if not len(indices) or digest in seen: continue
        seen.add(digest)
        projection=render_binary(addition,cameras,ledger.size,extent)
        hit=demand & projection
        if not hit.any(): continue
        primitives.append(indices)
        metadata.append({'id':len(primitives)-1,'endpoint_xyz':endpoint.tolist(),
            'path_centres':int(len(path)),'new_voxels':int(len(indices)),
            'demand_hits':int(hit.sum()),'indices_sha256':digest})
        remaining &= ~projection; all_covered |= projection
        if not remaining.any(): remaining=demand.copy()
        if len(primitives)>=max_primitives: break
    info.update(primitive_count=len(primitives),elapsed_seconds=time.monotonic()-started,
        demand_uncovered_by_all_primitives=int((demand & ~all_covered).sum()),
        stop_reason='FINITE_FAMILY_COMPLETE')
    return primitives,metadata,info
