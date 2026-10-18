"""Atomic lost-ray replacement by jointly selected thick rooted primitives."""
import copy
import hashlib
import math
import time

import numpy as np
from scipy import ndimage as ndi

from .geometry import _enclosed_voids, _pitch_from_occupied_bounds, _quality
from .manufacturing import FACE6, _block_cover, _positive, _volume, thickness_audit
from .material_relocation import retain_diverse
from .ray_demand import demand_primitives
from .ray_ledger import RayLedger
from .regional_geometry import cavities_preserved, manufacturing_improves, thick_join_preserved
from .regional_solver import solve_union
from .rendering import render_binary


def replace_by_demand(volume,guidance,selected,targets,cameras,*,extent=1.7,
                      longest_mm=100.,max_primitives=64,single_only=False,
                      time_limit=10.,callback=None):
    started=time.monotonic(); original=_volume(volume).copy(); selected=_volume(selected).copy()
    extent=_positive(extent,'extent'); longest_mm=_positive(longest_mm,'longest_mm')
    time_limit=_positive(time_limit,'time_limit')
    if type(max_primitives) is not int or not 0<=max_primitives<=256:
        raise ValueError('max_primitives must be integer in [0,256]')
    if type(single_only) is not bool: raise ValueError('single_only must be boolean')
    if callback is not None and not callable(callback): raise ValueError('callback must be callable')
    if selected.shape!=original.shape or not selected.any() or np.any(selected & ~original):
        raise ValueError('selected must be nonempty material inside original')
    labels,_=ndi.label(original,FACE6); counts=np.bincount(labels.ravel()); counts[0]=0
    ids=np.unique(labels[selected])
    if len(ids)!=1 or ids[0]==counts.argmax() or selected.sum()!=counts[ids[0]]:
        raise ValueError('select one whole non-main component')
    guidance=np.asarray(guidance,float); targets=np.asarray(targets); cameras=tuple(cameras)
    if guidance.shape!=original.shape or not np.isfinite(guidance).all() or np.any((guidance<0)|(guidance>1)):
        raise ValueError('guidance must match original in [0,1]')
    if (targets.ndim!=3 or not targets.shape[1] or targets.shape[1]!=targets.shape[2]
            or len(targets)!=len(cameras) or not np.isin(targets,[0,1]).all()):
        raise ValueError('binary square targets matching cameras required')
    targets=targets.astype(bool); size=targets.shape[-1]
    original_ledger=RayLedger(original,cameras,size,extent)
    initial=_quality(original_ledger.prediction,targets,.98,.97)
    protected=_enclosed_voids(original); width=max(1,math.ceil(1.35/(longest_mm/len(original))))
    if width>len(original): raise ValueError('neck width exceeds domain')
    def physical(v): return thickness_audit(v,_pitch_from_occupied_bounds(v,longest_mm))
    before=physical(original); base=original & ~selected
    variants=[('remove_island',base)]
    # Only zero-foreground-contribution material near the island may be removed.
    # Preserve all retained neck-core cover, not merely a centreline.
    p=np.argwhere(selected); lo=np.maximum(p.min(0)-width,0); hi=np.minimum(p.max(0)+width+1,len(base))
    region=np.zeros_like(base); region[tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))]=True
    foreground=original_ledger.backproject(targets.astype(float))
    core_cover=_block_cover(base,width)[1]
    removable=region & base & ~core_cover & (foreground==0)
    if removable.any(): variants.append(('remove_island_and_local_background',base & ~removable))
    pool=[]; seen=set(); histories=[]; families=[]
    def record(v,variant,selection,description):
        digest=hashlib.sha256(np.packbits(v).tobytes()).hexdigest()
        if digest in seen: return
        seen.add(digest)
        q=_quality(render_binary(v,cameras,size,extent),targets,.98,.97)
        row={'id':len(pool),'variant':variant,'selection':list(selection),'source':description,
             'quality':q,'components':int(ndi.label(v,FACE6)[1]),
             'thin':int((v & ~_block_cover(v,width)[1]).sum()),'volume':v.copy(),
             'added_voxels':int((v & ~original).sum()),'removed_voxels':int((original & ~v).sum())}
        pool.append(row)
    for variant,retained in variants:
        pred=render_binary(retained,cameras,size,extent)
        demand=targets & original_ledger.prediction & ~pred
        primitives,metadata,info=demand_primitives(retained,guidance,demand,targets,cameras,
            extent=extent,width=width,max_primitives=max_primitives,protected=protected)
        families.append({'variant':variant,'metadata':metadata,'generation':info})
        if callback: callback({'stage':'family','variant':variant,**info})
        record(retained,variant,[],'base_without_replacement')
        if not primitives: continue
        footprint_started=time.monotonic(); footprints=[]
        for indices in primitives:
            addition=np.zeros_like(original); addition.ravel()[indices]=True
            footprints.append(render_binary(addition,cameras,size,extent))
        footprints=np.stack(footprints)
        info['footprint_render_seconds']=time.monotonic()-footprint_started
        caps=[1] if single_only else sorted(set(min(c,len(primitives)) for c in (8,24,64)))
        floors=np.maximum(0.,np.array(initial['view_ious'])-.002)
        attempts=[(cap,floors,'initial_view_floor') for cap in caps]
        attempts.append((caps[-1],np.zeros(len(cameras)),'unconstrained_diagnostic_search'))
        constrained_incumbent=False
        for cap,floor,policy in attempts:
            if policy=='unconstrained_diagnostic_search' and constrained_incumbent: continue
            solve_started=time.monotonic()
            chosen,solver=solve_union(pred,targets,footprints,floor,max_selected=cap,time_limit=time_limit)
            history={'variant':variant,'max_selected':cap,'floor_policy':policy,
                     'solver':solver,'total_solve_seconds':time.monotonic()-solve_started}
            histories.append(history)
            if callback: callback({'stage':'solve',**copy.deepcopy(history)})
            if chosen is None: continue
            if policy=='initial_view_floor': constrained_incumbent=True
            candidate=retained.copy(); expected=pred.copy()
            for index in chosen:
                candidate.ravel()[primitives[index]]=True; expected |= footprints[index]
            if not np.array_equal(expected,render_binary(candidate,cameras,size,extent)):
                raise RuntimeError('DEMAND_UNION_PROJECTION_MISMATCH')
            record(candidate,variant,chosen,policy)
        del footprints
    shortlist=retain_diverse(pool,4); accepted=None; audits=[]
    for state in shortlist:
        candidate=state['volume']; q=state['quality']; reasons=[]
        if max_primitives==0: reasons.append('ZERO_BUDGET_NO_COMMIT')
        if (q['mean_iou']<initial['mean_iou']-1e-12 or q['worst_iou']<initial['worst_iou']-1e-12
                or np.any(np.array(q['view_ious'])<np.array(initial['view_ious'])-.002-1e-12)):
            reasons.append('PROJECTION_REGRESSION')
        if not cavities_preserved(candidate,protected): reasons.append('ORIGINAL_CAVITY_CHANGED')
        if not thick_join_preserved(candidate,original & candidate,candidate & ~original,width):
            reasons.append('NEW_CONNECTION_NECK_NOT_ROOT_CONNECTED')
        after=physical(candidate)
        if not manufacturing_improves(before,after): reasons.append('MANUFACTURING_NOT_PARETO_IMPROVED')
        state.update(reasons=reasons,manufacturing=after,accepted=not reasons,
            changed_indices=np.flatnonzero(candidate!=original).tolist())
        audits.append({k:v for k,v in state.items() if k!='volume'})
        if not reasons and accepted is None: accepted=state
    chosen=accepted or shortlist[0]
    trace={'algorithm':'joint_rooted_ray_demand_replacement_v1','accepted':accepted is not None,
           'publication_eligible':False,'structural_status':'NOT_REQUESTED','single_only':single_only,
           'max_primitives':max_primitives,'width_voxels':width,'time_limit_per_solve':time_limit,
           'initial_quality':initial,'candidate_quality':chosen['quality'],
           'initial_manufacturing':before,'candidate_manufacturing':chosen['manufacturing'],
           'selected_primitives':chosen['selection'],'chosen_id':chosen['id'],
           'reasons':chosen['reasons'],'families':families,'solves':histories,'finalists':audits,
           'physical_audits':len(audits),'elapsed_seconds':time.monotonic()-started,
           'timing_scope':'generation_and_footprint_render_separate; solve_total_includes_grouping_and_milp; internal_grouping_not_separately_timed',
           'scope':'finite_two_island_mechanism_not_whole_model_delivery'}
    return (accepted['volume'].copy() if accepted else original.copy()),trace,shortlist
