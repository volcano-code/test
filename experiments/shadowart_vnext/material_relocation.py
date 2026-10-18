"""Bounded beam of ray-guided relocation and atomic material-edit candidates.

This experimental transaction is not a printable-model publisher. All original
inputs are copied; failed candidates are returned separately from rollback.
"""
import copy
import hashlib
import math
import time

import numpy as np
from scipy import ndimage as ndi

from .geometry import _enclosed_voids, _pitch_from_occupied_bounds, _quality
from .manufacturing import FACE6, _block_cover, _positive, _volume, thickness_audit
from .ray_ledger import RayLedger
from .regional_geometry import cavities_preserved, manufacturing_improves, thick_join_preserved
from .relocation_proposals import best_patch, deletion_bundles, translated_seeds
from .rendering import render_binary


def retain_diverse(rows, limit=4):
    """Round-robin fidelity/physical priorities without discarding all trade-offs."""
    selected=[]
    keys=[lambda s:(-s['quality']['worst_iou'],-s['quality']['mean_iou']),
          lambda s:(-s['quality']['mean_iou'],-s['quality']['worst_iou']),
          lambda s:(s['components'],s['thin'], -s['quality']['worst_iou']),
          lambda s:(s['thin'],s['components'], -s['quality']['mean_iou'])]
    ordered=[sorted(rows,key=key) for key in keys]
    while len(selected)<min(limit,len({s['id'] for s in rows})):
        for ordered_rows in ordered:
            for row in ordered_rows:
                if all(row['id']!=s['id'] for s in selected):
                    selected.append(row); break
            if len(selected)>=limit: break
    return selected


def relocate_component(volume, guidance, selected, targets, cameras, *, extent=1.7,
                       longest_mm=100., max_evaluations=128, joint_edits=True, callback=None):
    started=time.monotonic()
    original=_volume(volume).copy(); selected=_volume(selected).copy()
    extent=_positive(extent,'extent'); longest_mm=_positive(longest_mm,'longest_mm')
    if type(max_evaluations) is not int or not 0<=max_evaluations<=1024:
        raise ValueError('max_evaluations must be an integer in [0,1024]')
    if type(joint_edits) is not bool: raise ValueError('joint_edits must be boolean')
    if callback is not None and not callable(callback): raise ValueError('callback must be callable')
    if selected.shape!=original.shape or not selected.any() or np.any(selected & ~original):
        raise ValueError('nonempty selected component inside occupancy required')
    labels,_=ndi.label(original,FACE6); sizes=np.bincount(labels.ravel()); sizes[0]=0
    ids=np.unique(labels[selected])
    if len(ids)!=1 or ids[0]==sizes.argmax() or selected.sum()!=sizes[ids[0]]:
        raise ValueError('select one whole non-main component')
    guidance=np.asarray(guidance,float)
    if guidance.shape!=original.shape or not np.isfinite(guidance).all() or np.any((guidance<0)|(guidance>1)):
        raise ValueError('guidance must match occupancy and lie in [0,1]')
    targets=np.asarray(targets); cameras=tuple(cameras)
    if (targets.ndim!=3 or not cameras or targets.shape[0]!=len(cameras) or
            not targets.shape[1] or targets.shape[1]!=targets.shape[2] or not np.isin(targets,[0,1]).all()):
        raise ValueError('binary square targets matching cameras required')
    targets=targets.astype(bool)
    source=RayLedger(original,cameras,targets.shape[-1],extent)
    initial=_quality(source.prediction,targets,.98,.97)
    base=original & ~selected
    protected=_enclosed_voids(original)
    width=max(1,math.ceil(1.35/(longest_mm/len(original))))
    if width>len(original): raise ValueError('neck width exceeds domain')
    evaluations=[]; seen=set(); archive=[]; attempts=0

    def evaluate(candidate,editable,operation,parent=None):
        nonlocal attempts,archive
        attempts+=1
        if len(evaluations)>=max_evaluations: return None
        digest=hashlib.sha256(np.packbits(candidate).tobytes()+np.packbits(editable).tobytes()).hexdigest()
        if digest in seen: return None
        seen.add(digest)
        if np.any((candidate!=original)&~editable): raise RuntimeError('OUTSIDE_EDIT_REGION')
        candidate,counts=source.evaluate(candidate & ~original,original & ~candidate)
        q=_quality(counts>0,targets,.98,.97)
        components=int(ndi.label(candidate,FACE6)[1])
        thin=int((candidate & ~_block_cover(candidate,width)[1]).sum())
        row={'id':len(evaluations),'parent':parent,'operation':operation,'quality':q,
             'components':components,'thin':thin,'added_voxels':int((candidate & ~original).sum()),
             'removed_voxels':int((original & ~candidate).sum())}
        evaluations.append(row)
        state=dict(row,volume=candidate,editable=editable)
        archive=retain_diverse(archive+[state],16)
        if callback: callback(copy.deepcopy(row))
        return state

    seed_states=[]
    p=np.argwhere(selected); editable=np.zeros_like(base)
    lo=np.maximum(p.min(0)-width,0); hi=np.minimum(p.max(0)+width+1,len(base))
    editable[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]=True
    if max_evaluations:
        seed_states.append(evaluate(base,editable,{'operation':'remove_selected_component'}))
        ledger=RayLedger(base,cameras,targets.shape[-1],extent)
        for addition,region,description in translated_seeds(selected,base,guidance,ledger,targets,protected,width):
            row=evaluate(base|addition,region,description)
            if row is not None: seed_states.append(row)
            if len(evaluations)>=max_evaluations: break
    beam=retain_diverse([s for s in seed_states if s is not None])
    # Eight finite expansion rounds. Keep parents so a poor expansion cannot
    # overwrite a better archive; source-relative final gates remain separate.
    rounds=0
    for round_index in range(8):
        if not beam or len(evaluations)>=max_evaluations: break
        rounds=round_index+1; pool=list(beam)
        for state in beam:
            current=state['volume']; region=state['editable']
            patch=best_patch(current,guidance,protected,width,targets,cameras,extent,region)
            if patch is not None:
                add,info=patch
                row=evaluate(current|add,region,info,state['id'])
                if row is not None: pool.append(row)
            if joint_edits and len(evaluations)<max_evaluations:
                ledger=RayLedger(current,cameras,targets.shape[-1],extent)
                for remove,info in deletion_bundles(ledger,targets,region,width):
                    deleted=current & ~remove
                    row=evaluate(deleted,region,info,state['id'])
                    if row is not None: pool.append(row)
                    if len(evaluations)>=max_evaluations: break
                    patch=best_patch(deleted,guidance,protected,width,targets,cameras,extent,region)
                    if patch is not None:
                        add,patch_info=patch
                        row=evaluate(deleted|add,region,{'operation':'delete_and_compensate',
                                     'delete':info,'add':patch_info},state['id'])
                        if row is not None: pool.append(row)
                    if len(evaluations)>=max_evaluations: break
            if len(evaluations)>=max_evaluations: break
        next_beam=retain_diverse(pool)
        if [s['id'] for s in next_beam]==[s['id'] for s in beam]: break
        beam=next_beam

    def physical(v): return thickness_audit(v,_pitch_from_occupied_bounds(v,longest_mm))
    before=physical(original); audits=[]; accepted=None
    for state in archive:
        candidate=state['volume']; q=state['quality']; reasons=[]
        actual=render_binary(candidate,cameras,targets.shape[-1],extent)
        if not np.array_equal(actual,source.evaluate(candidate & ~original,original & ~candidate)[1]>0):
            raise RuntimeError('RELOCATION_RAY_COUNT_MISMATCH')
        if (q['mean_iou']<initial['mean_iou']-1e-12 or q['worst_iou']<initial['worst_iou']-1e-12 or
                np.any(np.array(q['view_ious'])<np.array(initial['view_ious'])-.002-1e-12)):
            reasons.append('PROJECTION_REGRESSION')
        if not cavities_preserved(candidate,protected): reasons.append('ORIGINAL_CAVITY_CHANGED')
        if not thick_join_preserved(candidate,original & candidate,candidate & ~original,width):
            reasons.append('NEW_CONNECTION_NECK_NOT_ROOT_CONNECTED')
        after=physical(candidate)
        if not manufacturing_improves(before,after): reasons.append('MANUFACTURING_NOT_PARETO_IMPROVED')
        audits.append({'id':state['id'],'reasons':reasons,'physical':after})
        if not reasons:
            accepted=state; break
    chosen=accepted or (archive[0] if archive else None)
    candidate=chosen['volume'].copy() if chosen else base.copy()
    q=chosen['quality'] if chosen else _quality(render_binary(candidate,cameras,targets.shape[-1],extent),targets,.98,.97)
    selected_audit=next((a for a in audits if chosen and a['id']==chosen['id']),None)
    reasons=selected_audit['reasons'] if selected_audit else ['ZERO_BUDGET_NO_COMMIT']
    trace={'algorithm':'ray_guided_material_relocation_v1','accepted':accepted is not None,
           'publication_eligible':False,'structural_status':'NOT_REQUESTED','joint_edits':joint_edits,
           'width_voxels':width,'beam_width':4,'rounds':rounds,'max_rounds':8,
           'max_evaluations':max_evaluations,'evaluated_candidates':len(evaluations),'proposal_attempts':attempts,
           'physical_audits':len(audits),'max_physical_audits':16,'initial_quality':initial,
           'candidate_quality':q,'initial_manufacturing':before,
           'candidate_manufacturing':selected_audit['physical'] if selected_audit else physical(candidate),
           'reasons':reasons,'chosen_id':chosen['id'] if chosen else None,
           'history':evaluations,'audits':audits,'elapsed_seconds':time.monotonic()-started,
           'support_scope':'finite_depth_translation_and_local_ray_bundle_edits_not_global_solver'}
    return (candidate.copy() if accepted else original.copy()),trace,candidate
