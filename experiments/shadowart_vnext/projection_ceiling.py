"""Exact per-view binary column relaxation for cardinal cell projections.

This bounds a fixed lattice renderer, NOT arbitrary surfaces, layouts, oblique
cameras, or manufacturing feasibility. Independent view masks need not coexist
as a single 3D object. No target or camera parameters are optimized here.
"""
import numpy as np

from .rendering import _axis_map, _key, _sizes


def optimal_groups(foreground, total, target_positive):
    fg, count = np.asarray(foreground), np.asarray(total)
    if (fg.ndim != 1 or count.shape != fg.shape or
            not np.issubdtype(fg.dtype,np.integer) or not np.issubdtype(count.dtype,np.integer)
            or np.any(fg<0) or np.any(count<fg)
            or not isinstance(target_positive,(int,np.integer))
            or target_positive < int(fg.sum())):
        raise ValueError('nonnegative integral group counts and total foreground required')
    selected=np.zeros(len(fg),bool)
    if target_positive == 0:
        return selected, 1.
    # At optimal IoU J a group helps iff fg - J*bg > 0. Sorting
    # fg/(fg+bg) therefore contains an optimal prefix, including ties.
    fraction=np.divide(fg,count,out=np.zeros(len(fg),float),where=count>0)
    order=np.argsort(-fraction,kind='stable')
    tp=np.cumsum(fg[order],dtype=np.int64)
    fp=np.cumsum((count-fg)[order],dtype=np.int64)
    scores=tp/(int(target_positive)+fp)
    if not len(scores) or not scores.max():
        return selected, 0.
    best=int(scores.argmax())
    selected[order[:best+1]]=True
    return selected, float(scores[best])


def cardinal_ceiling(targets, cameras, resolution=256, extent=1.7):
    targets=np.asarray(targets)
    if (targets.dtype != np.bool_ or targets.ndim != 3 or targets.shape[0]!=len(cameras)
            or not len(cameras) or targets.shape[1]!=targets.shape[2]):
        raise ValueError('binary square target per camera required')
    size=targets.shape[-1]; _sizes(size,resolution,extent)
    rows=[]
    for target,camera in zip(targets,cameras):
        mapping=_axis_map(_key(camera),size,resolution,extent)
        if mapping is None:
            raise ValueError('cardinal cameras outside volume required for this relaxation')
        axis,indices,valid=mapping
        total=np.bincount(indices[valid],minlength=resolution**2)
        positive=np.bincount(indices[valid & target.ravel()],minlength=resolution**2)
        chosen,score=optimal_groups(positive,total,int(target.sum()))
        rows.append({'camera':camera.name,'axis':axis,'upper_bound_iou':score,
                     'target_positive_pixels':int(target.sum()),
                     'unreachable_positive_pixels':int((target.ravel() & ~valid).sum()),
                     'selected_columns':int(chosen.sum())})
    return {'scope':'fixed_grid_cardinal_binary_point_ray_relaxation',
            'image_size':size,'volume_resolution':int(resolution),
            'mean_upper_bound':float(np.mean([r['upper_bound_iou'] for r in rows])),
            'worst_upper_bound':float(min(r['upper_bound_iou'] for r in rows)),
            'views':rows,'feasible_object_certified':False,
            'cross_view_consistency_enforced':False,'manufacturing_enforced':False,
            'arbitrary_mesh_bound':False}
