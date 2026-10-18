"""Finite ray-guided depth relocations and local edit proposals; no acceptance gates."""
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import route_through_array

from .component_replacement import neck_frontier
from .geometry import _thicken_mask
from .joint_geometry import BlockProjection
from .manufacturing import FACE6, _block_cover


def largest(volume):
    labels, _ = ndi.label(volume, FACE6)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    return labels == sizes.argmax() if sizes.max() else np.zeros_like(volume)


def translated_seeds(selected, base, guidance, ledger, targets, protected, width):
    """Move along camera directions; rounded shifts are always exactly reprojected.

    This finite family preserves shape before thickening, not every silhouette.
    It is not a general ray-consistent reconstruction solver.
    """
    r = base.shape[0]; points = np.argwhere(selected); root = largest(base)
    displacement = np.argwhere(root).mean(0) - points.mean(0)
    missing = ledger.backproject(targets & ~ledger.prediction)
    leakage = ledger.backproject((~targets).astype(float))
    normalizer = max(float(missing.max()), float(leakage.max()), 1.)
    value = (missing - leakage) / normalizer + .05 * guidance
    shifts = set()
    for camera in ledger.cameras:
        direction = np.asarray(camera.direction)
        nearest = float(np.dot(displacement, direction))
        for distance in [-32,-16,-8,-4,-2,2,4,8,16,32,nearest-width,nearest,nearest+width]:
            shift = tuple(np.rint(direction * distance).astype(int))
            if any(shift): shifts.add(shift)
    proposals = []
    for shift in sorted(shifts):
        moved = points + shift
        if np.any(moved < 0) or np.any(moved >= r): continue
        body = np.zeros_like(base); body[tuple(moved.T)] = True
        thin = body & ~_block_cover(body, width)[1]
        if thin.any(): body |= _thicken_mask(thin, width)
        # No clipping is allowed during thickness construction.
        if thin.any():
            p = np.argwhere(thin)
            if np.any(p-width//2 < 0) or np.any(p+(width-1)//2 >= r): continue
        if np.any(body & protected): continue
        gain = float(value[body & ~base].sum())
        proposals.append((gain, shift, body))
    proposals.sort(key=lambda x:(-x[0],x[1]))
    anchors = _block_cover(root, width)[0]
    if not anchors.any(): return
    distance, nearest = ndi.distance_transform_edt(~anchors, return_indices=True)
    forbidden = ndi.maximum_filter(protected, size=width, mode='constant')
    for _, shift, body in proposals[:16]:
        goals = np.argwhere(_block_cover(body, width)[0])
        if not len(goals): continue
        goal = goals[np.argmin(distance[tuple(goals.T)])]
        start = nearest[(slice(None), *goal)]
        c = width//2
        lo = np.maximum(np.minimum(start,goal)-4,c)
        hi = np.minimum(np.maximum(start,goal)+5,r-width+c+1)
        roi = tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
        cost = 1. + 3.*(1.-guidance[roi]) + 2.*leakage[roi]/normalizer
        cost[forbidden[roi]] = np.inf
        try:
            path,_ = route_through_array(cost,tuple(start-lo),tuple(goal-lo),
                                         fully_connected=False,geometric=False)
        except ValueError: continue
        if len(path)>128: continue
        centres=np.asarray(path,int)+lo
        line=np.zeros_like(base); line[tuple(centres.T)]=True
        addition=body|_thicken_mask(line,width)
        if np.any(addition & protected): continue
        # Fixed domain for this branch: bbox of original/replacement/path + width.
        p=np.argwhere(selected|addition)
        lower=np.maximum(p.min(0)-width,0); upper=np.minimum(p.max(0)+width+1,r)
        editable=np.zeros_like(base)
        editable[tuple(slice(int(a),int(b)) for a,b in zip(lower,upper))]=True
        yield addition, editable, {'operation':'depth_relocation', 'shift_xyz':[int(v) for v in shift],
                                  'path_centres':len(path), 'editable_bounds':[lower.tolist(),upper.tolist()]}


def best_patch(volume, guidance, protected, width, targets, cameras, extent, editable):
    lower=neck_frontier(largest(volume),protected,width)
    if not len(lower): return None
    centres=lower+width//2
    # All added cells, not merely centres, must lie in this branch's fixed ROI.
    allowed=ndi.minimum_filter(editable,size=width,mode='constant',cval=0)
    lower=lower[allowed[tuple(centres.T)]]
    if not len(lower): return None
    from .rendering import render_binary
    prediction=render_binary(volume,cameras,targets.shape[-1],extent)
    scorer=BlockProjection(prediction,targets,cameras,len(volume),extent)
    if not scorer.cardinal and len(lower)>256:
        lower=lower[np.argsort(-guidance[tuple((lower+width//2).T)],kind='stable')[:256]]
    q=scorer.qualities(lower,width)
    rank=q.mean(1)+q.min(1)
    index=int(np.argmax(rank)); lo=lower[index]
    add=np.zeros_like(volume); add[tuple(slice(int(a),int(a+width)) for a in lo)]=True
    return add, {'operation':'foreground_patch','lower_xyz':lo.tolist(),
                 'box_proxies_scored':len(lower)}


def deletion_bundles(ledger, targets, editable, width):
    """Delete every editable blocker on a small background-error ray bundle."""
    fp=ledger.prediction & ~targets
    radius=max(1,int(np.ceil(width*ledger.size/ledger.resolution/2)))
    offered=0
    for i in np.argsort(-fp.sum((1,2)),kind='stable'):
        pixels=np.argwhere(fp[i])
        if not len(pixels): continue
        # Two spatially separated bundles per view, four globally.
        for index in sorted(set([len(pixels)//3,2*len(pixels)//3])):
            y,x=pixels[index]; weights=np.zeros(fp.shape,float)
            a,b=max(0,y-radius),min(ledger.size,y+radius+1)
            c,d=max(0,x-radius),min(ledger.size,x+radius+1)
            weights[i,a:b,c:d]=fp[i,a:b,c:d]
            remove=(ledger.backproject(weights)>0)&ledger.volume&editable
            if remove.any():
                yield remove, {'operation':'background_bundle','view':int(i),
                               'pixel_bounds':[int(a),int(b),int(c),int(d)],
                               'removed_voxels':int(remove.sum())}
                offered+=1
                if offered==4: return
