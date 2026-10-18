"""Bounded layout conditioning; failed search is not an infeasibility proof."""
from dataclasses import replace
import itertools

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import qmc

from .contracts import identity_layout, validate_layout, mask_hash


def iou_report(predictions, targets):
    p,t = np.asarray(predictions),np.asarray(targets)
    if p.ndim != 3 or p.shape != t.shape or not len(p):
        raise ValueError('projection dimensions mismatch')
    if not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('nonfinite projections')
    p,t = p.astype(bool),t.astype(bool)
    inter = (p&t).sum(axis=(1,2))
    union = (p|t).sum(axis=(1,2))
    values = np.divide(inter,union,out=np.ones(len(p),float),where=union!=0)
    return {'mean_iou':float(values.mean()), 'worst_iou':float(values.min()),
            'per_view_iou':values.tolist(), 'image_size':int(p.shape[-1]),
            'false_positive_pixels':(p&~t).sum(axis=(1,2)).tolist(),
            'false_negative_pixels':(~p&t).sum(axis=(1,2)).tolist()}


def quality_pass(report):
    return bool(np.isfinite(report['mean_iou']) and np.isfinite(report['worst_iou'])
                and report['mean_iou'] > .98 and report['worst_iou'] >= .97)


def feasibility_status(report):
    return 'SHADOW_FEASIBLE' if quality_pass(report) else 'UNKNOWN'


def candidate_layouts(n, count=512, seed=0):
    if n < 1 or not 1 <= count <= 512:
        raise ValueError('positive N and 1..512 layout candidates required')
    yield identity_layout(n)
    if count == 1:
        return
    points = qmc.Sobol(3*n,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(count))))
    low,high = np.log(.9),np.log(1.1)
    for point in points[:count-1]:
        x = point.reshape(n,3)
        logs = low+(high-low)*x[:,0]
        logs -= logs.mean()
        factor = min(1.,high/max(logs.max(),1e-30),low/min(logs.min(),-1e-30))
        logs *= factor
        yield validate_layout(np.c_[np.exp(logs),20*x[:,1:]-10])


def inside_projected_volume(targets, cameras, extent=1.7):
    corners = np.array(list(itertools.product([-.5*extent,.5*extent],repeat=3)))
    size = targets.shape[-1]
    for mask,camera in zip(targets,cameras):
        offsets = corners-camera.origin
        xy = np.c_[offsets @ camera.right, offsets @ camera.up]
        eq = ConvexHull(xy).equations
        rows,cols = np.where(mask)
        points = np.c_[((cols+.5)/size-.5)*camera.scale,
                       (.5-(rows+.5)/size)*camera.scale]
        if len(points) and np.any(points @ eq[:,:2].T+eq[:,2] > 1e-9):
            return False
    return True


def conditioning_hull(target_set, contract, layout):
    """Condition using a denser ray grid to avoid the nominal-ray weaving loophole.

    This is still a finite-grid heuristic, not a global physical feasibility proof.
    Original source geometry supplies the denser masks; no repeated mask resizing.
    """
    from .rendering import visual_hull
    size=conditioning_image_size(contract)
    masks=target_set.render(contract.cameras,size,layout)
    return visual_hull(masks,contract.cameras,contract.volume_resolution,extent=contract.extent)


def conditioning_image_size(contract):
    # Projected inscribed-cell disk covers at least one lattice point above this density.
    # Off-screen material is still unobserved; this is not continuous-beam certification.
    bound=int(np.floor(np.sqrt(2)*max(c.scale for c in contract.cameras)*
                       contract.volume_resolution/contract.extent))+1
    return max(2*contract.image_size,2*contract.volume_resolution,bound)


def search_layout(target_set, contract, *, candidates=512, callback=None):
    from .rendering import render_binary, visual_hull
    rows,shortlist = [],[]
    for index,layout in enumerate(candidate_layouts(contract.base_count,candidates)):
        masks = target_set.render(contract.cameras,contract.image_size,layout)
        if not inside_projected_volume(masks,contract.cameras,contract.extent):
            row = {'index':index,'status':'REJECTED_OUTSIDE_PROJECTED_VOLUME','layout':layout}
        else:
            hull = conditioning_hull(target_set,contract,layout)
            score = iou_report(render_binary(hull,contract.cameras,contract.image_size,extent=contract.extent),masks)
            cost = float(np.square(np.log(np.asarray(layout)[:,0])).sum()+
                         np.square(np.asarray(layout)[:,1:]/256).sum())
            row = {'index':index,'status':'EVALUATED','layout':layout,**score,
                   'transformation_cost':cost,'voxels':int(hull.sum())}
            rank = (score['worst_iou'],score['mean_iou'],-cost,-index)
            shortlist.append((rank,index,layout))
            shortlist.sort(reverse=True)
            shortlist = shortlist[:3]
        rows.append(row)
        if callback:
            callback(row)
    if not shortlist:
        raise ValueError('NO_VALID_LAYOUT: every candidate clips projected build volume')
    checks=[]
    # Re-evaluate top layouts from source at512; selection rule itself remains the declared256 rule.
    for _,index,layout in shortlist:
        masks = target_set.render(contract.cameras,contract.image_size,layout)
        hull = conditioning_hull(target_set,contract,layout)
        size = max(512,contract.image_size*2)
        high = iou_report(render_binary(hull,contract.cameras,size,extent=contract.extent),
                          target_set.render(contract.cameras,size,layout))
        checks.append({'index':index,**high})
    _,selected,layout = shortlist[0]
    masks = target_set.render(contract.cameras,contract.image_size,layout)
    frozen = replace(contract,layout=layout,input_hashes=target_set.input_hashes,
                     target_hashes=tuple(mask_hash(m) for m in masks))
    hull = conditioning_hull(target_set,contract,layout)
    low = rows[selected]
    high = checks[0]
    stable = max(abs(np.array(low['per_view_iou'])-high['per_view_iou'])) <= .001
    status = 'SHADOW_FEASIBLE' if stable and quality_pass(low) and quality_pass(high) else 'UNKNOWN'
    return frozen,masks,hull,{'status':status,'selected_index':selected,'candidates':rows,
                              'high_resolution_checks':checks,'converged':bool(stable),
                              'scope':'dense_ray_zero_false_positive_hull_heuristic_not_global_bound',
                              'hull_ray_image_size':conditioning_image_size(contract),
                              'targets':target_set.diagnostics(contract.cameras,contract.image_size,layout)}
