"""Bounded attached-block ray restoration retaining all source material.

Unlike the historical joint pilot, never discard islands before restoring their
shadows. This is a finite additive proposal family, not a general reconstruction
solver. Optional independent mesh feedback checks every transaction, and final
serialized multi-resolution/manufacturing QA is still required.
"""
import numpy as np
from scipy import ndimage as ndi
from .geometry import _enclosed_voids, _volume, _positive, occupancy_to_mesh
from .joint_geometry import BlockProjection, _eligible_boxes
from .rendering import render_binary,render_mesh
from .layout import iou_report


def restore(volume_xyz,guidance,targets,cameras,*,extent=1.7,width=4,steps=64,
            mesh_feedback=False,callback=None):
    v=_volume(volume_xyz).copy(); extent=_positive(extent,'extent')
    if not v.any() or type(width)!=int or not 1<=width<=v.shape[0] or type(steps)!=int or not 0<=steps<=256:
        raise ValueError('nonempty volume, valid width and integral steps in [0,256] required')
    guidance=np.asarray(guidance)
    if guidance.shape!=v.shape or not np.isfinite(guidance).all() or np.any(guidance<0) or np.any(guidance>1):
        raise ValueError('guidance must match volume and lie in [0,1]')
    targets=np.asarray(targets); size=targets.shape[-1]
    scorer=BlockProjection(render_binary(v,cameras,size,extent),targets,cameras,v.shape[0],extent)
    if not scorer.cardinal:
        raise ValueError('cardinal cameras required for this finite native restoration pilot')
    initial=iou_report(scorer.prediction,targets); quality=initial
    protected=_enclosed_voids(v)
    trace={'algorithm':'retain_source_attached_ray_restoration_v1','history':[],
           'initial_quality':initial,'width_voxels':width,'step_budget':steps,
           'mesh_feedback':mesh_feedback,'publication_eligible':False,
           'manufacturing_status':'NOT_VERIFIED','structural_status':'NOT_REQUESTED',
           'source_occupancy_mutated':False,'stop_reason':'STEP_BUDGET'}
    mesh_quality=iou_report(render_mesh(occupancy_to_mesh(v,extent),cameras,size),targets) if mesh_feedback else None
    for step in range(steps):
        # Only the retained main body seeds candidate blocks; no new free islands.
        labels,_=ndi.label(v); counts=np.bincount(labels.ravel()); counts[0]=0
        root=labels==int(counts.argmax())
        lower=_eligible_boxes(root,protected,width)
        if not len(lower):
            trace['stop_reason']='NO_ATTACHED_CANDIDATE'; break
        values=scorer.qualities(lower,width)
        means,worst=values.mean(1),values.min(1)
        valid=(means>quality['mean_iou']+1e-12)&(worst>=quality['worst_iou']-1e-12)
        valid &= (values>=np.asarray(initial['per_view_iou'])-.002-1e-12).all(1)
        eligible=np.flatnonzero(valid)
        if not len(eligible):
            trace['stop_reason']='NO_NONREGRESSING_RAY_GAIN'; break
        center=lower[eligible]+(width-1)//2
        ordering=np.lexsort((-guidance[tuple(center.T)],-worst[eligible],-means[eligible]))
        # Mesh feedback is a separate physical triangle renderer, not scorer reuse.
        accepted=False
        for index in eligible[ordering[:3]]:
            lo=lower[index]; candidate=v.copy()
            candidate[tuple(slice(int(x),int(x+width)) for x in lo)]=True
            qmesh=iou_report(render_mesh(occupancy_to_mesh(candidate,extent),cameras,size),targets) if mesh_feedback else None
            if mesh_feedback and (qmesh['mean_iou']<mesh_quality['mean_iou']-1e-12 or
                                  qmesh['worst_iou']<mesh_quality['worst_iou']-1e-12):
                continue
            added=int(np.count_nonzero(candidate & ~v))
            v=candidate; scorer.add(lo,width); quality=iou_report(scorer.prediction,targets)
            if mesh_feedback: mesh_quality=qmesh
            row={'step':step+1,'lower_xyz':lo.tolist(),'added_voxels':added,
                 'mean_iou':quality['mean_iou'],'worst_iou':quality['worst_iou'],
                 'mesh_quality':qmesh}
            trace['history'].append(row)
            if callback: callback(dict(row))
            accepted=True; break
        if not accepted:
            trace['stop_reason']='TOP_THREE_MESH_CANDIDATES_REGRESS'; break
    if not np.array_equal(render_binary(v,cameras,size,extent),scorer.prediction):
        raise RuntimeError('RESTORATION_RAY_PROJECTION_MISMATCH')
    trace['final_quality']=quality
    trace['final_mesh_quality']=mesh_quality
    return v,trace
