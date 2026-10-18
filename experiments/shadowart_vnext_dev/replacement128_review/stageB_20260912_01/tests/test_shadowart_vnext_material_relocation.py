import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.shadowart_vnext.rendering import render_binary


def cameras():
    return [SimpleNamespace(origin=(0.,0.,2.),right=(1.,0.,0.),up=(0.,1.,0.),direction=(0.,0.,-1.),scale=1.7),
            SimpleNamespace(origin=(2.,0.,0.),right=(0.,0.,-1.),up=(0.,1.,0.),direction=(-1.,0.,0.),scale=1.7)]


def scene():
    root=np.zeros((24,)*3,bool); root[2:8,4:12,2:8]=True
    selected=np.zeros_like(root); selected[4:8,6:10,14:18]=True
    target_volume=root.copy(); target_volume[4:8,6:10,6:10]=True
    return root|selected, selected, target_volume


def relocate(*args, **kwargs):
    name='experiments.shadowart_vnext.material_relocation'
    assert importlib.util.find_spec(name), 'joint material relocation missing'
    return importlib.import_module(name).relocate_component(*args, **kwargs)


def test_depth_relocation_removes_background_that_addition_alone_cannot_clear():
    v, selected, desired=scene(); original=v.copy(); cams=cameras()
    targets=render_binary(desired,cams,48)
    # Any superset of original retains these false-positive rays.
    assert np.any(render_binary(v,cams,48) & ~targets)
    out,trace,candidate=relocate(v,desired.astype(float),selected,targets,cams,
                               longest_mm=24.,max_evaluations=48)
    assert trace['accepted'], trace['reasons']
    np.testing.assert_array_equal(render_binary(out,cams,48),targets)
    assert trace['candidate_manufacturing']['component_count_6']==1
    assert trace['candidate_manufacturing']['thin_voxels']==0
    assert not out[selected].any()
    assert np.any(out & ~v)
    assert trace['evaluated_candidates']<=48
    assert trace['publication_eligible'] is False
    np.testing.assert_array_equal(v,original)
    np.testing.assert_array_equal(out,candidate)


def test_zero_budget_rolls_back_without_silently_running_search():
    v,selected,desired=scene()
    out,trace,_=relocate(v,desired.astype(float),selected,render_binary(desired,cameras(),48),
                         cameras(),longest_mm=24.,max_evaluations=0)
    assert not trace['accepted'] and trace['evaluated_candidates']==0
    np.testing.assert_array_equal(out,v)


def test_perfect_disconnected_targets_cannot_bypass_final_projection_gate():
    v=np.zeros((24,)*3,bool); v[4:10,4:10,4:10]=True
    selected=np.zeros_like(v); selected[12:16,4:8,12:16]=True; v|=selected
    out,trace,_=relocate(v,v.astype(float),selected,render_binary(v,cameras(),48),cameras(),
                         longest_mm=24.,max_evaluations=24)
    assert not trace['accepted']
    assert trace['physical_audits']<=16 and trace['evaluated_candidates']<=24
    np.testing.assert_array_equal(out,v)


def test_background_bundle_deletes_all_editable_occluders_not_independent_gains():
    from experiments.shadowart_vnext.ray_ledger import RayLedger
    from experiments.shadowart_vnext.relocation_proposals import deletion_bundles
    v=np.zeros((8,)*3,bool); v[2,3,[1,6]]=True
    l=RayLedger(v,cameras()[:1],8); targets=np.zeros((1,8,8),bool)
    remove,_=next(deletion_bundles(l,targets,np.ones_like(v),2))
    assert remove[2,3,1] and remove[2,3,6]
    assert not l.evaluate(np.zeros_like(v),remove)[1].any()
    editable=np.ones_like(v); editable[2,3,6]=False
    remove,_=next(deletion_bundles(l,targets,editable,2))
    assert l.evaluate(np.zeros_like(v),remove)[1][0,4,2]==1


def test_no_joint_control_disables_bundle_and_compensated_edits():
    v,selected,desired=scene()
    _,trace,_=relocate(v,desired.astype(float),selected,render_binary(desired,cameras(),48),cameras(),
                       longest_mm=24.,max_evaluations=32,joint_edits=False)
    assert all(r['operation']['operation'] not in ('background_bundle','delete_and_compensate')
               for r in trace['history'])


def test_original_cavity_is_not_filled_or_opened_to_accept_replacement():
    v=np.zeros((24,)*3,bool); v[2:10,2:22,2:10]=True
    selected=np.zeros_like(v); selected[10:18,6:18,14:22]=True
    selected[11:17,7:17,15:21]=False; v|=selected
    out,trace,_=relocate(v,v.astype(float),selected,render_binary(v,cameras(),48),cameras(),
                         longest_mm=24.,max_evaluations=16)
    assert not trace['accepted']
    assert any('ORIGINAL_CAVITY_CHANGED' in a['reasons'] for a in trace['audits'])
    np.testing.assert_array_equal(out,v)


def test_archive_retains_physical_alternative_when_fidelity_list_is_full():
    from experiments.shadowart_vnext import material_relocation as module
    assert hasattr(module,'retain_diverse'), 'archive needs physical as well as fidelity priorities'
    rows=[{'id':i,'quality':{'mean_iou':.99-i*.001,'worst_iou':.98-i*.001},
           'components':20,'thin':200} for i in range(17)]
    rows[-1].update(components=1,thin=0)
    kept=module.retain_diverse(rows,16)
    assert len(kept)==16 and len({s['id'] for s in kept})==16
    assert any(s['id']==16 for s in kept)


def test_each_exact_search_evaluation_is_accounted_for(monkeypatch):
    from experiments.shadowart_vnext.ray_ledger import RayLedger
    actual=RayLedger.evaluate; calls=[]
    def observed(self,add,remove):
        calls.append(1)
        return actual(self,add,remove)
    monkeypatch.setattr(RayLedger,'evaluate',observed)
    v=np.zeros((24,)*3,bool); v[4:10,4:10,4:10]=True
    selected=np.zeros_like(v); selected[12:16,4:8,12:16]=True; v|=selected
    _,trace,_=relocate(v,v.astype(float),selected,render_binary(v,cameras(),48),cameras(),
                       longest_mm=24.,max_evaluations=48)
    assert any(r['operation']['operation']=='background_bundle' for r in trace['history'])
    assert len(calls)==trace['evaluated_candidates']+trace['physical_audits']


def test_remote_material_outside_relocation_corridor_is_unchanged():
    v,selected,desired=scene(); v[18:20,18:20,18:20]=True
    desired[18:20,18:20,18:20]=True
    out,trace,candidate=relocate(v,desired.astype(float),selected,
        render_binary(desired,cameras(),48),cameras(),longest_mm=24.,max_evaluations=48)
    assert trace['accepted']
    np.testing.assert_array_equal(candidate[:,16:,:],v[:,16:,:])


def test_filament_root_cannot_be_used_as_a_thick_anchor():
    v=np.zeros((24,)*3,bool); v[2:20,2,2]=True
    selected=np.zeros_like(v); selected[6:8,6:8,6:8]=True; v|=selected
    out,trace,_=relocate(v,v.astype(float),selected,render_binary(v,cameras(),48),cameras(),
                         longest_mm=24.,max_evaluations=24)
    assert not trace['accepted']
    assert not any(r['operation']['operation']=='depth_relocation' for r in trace['history'])
    np.testing.assert_array_equal(out,v)


def test_probe_requires_separate_fresh_output_and_bounded_evaluations(tmp_path):
    name='experiments.shadowart_vnext_dev.run_relocation128_probe_20260912'
    assert importlib.util.find_spec(name), 'frozen relocation probe missing'
    parse=importlib.import_module(name).parse_args
    source=tmp_path/'soft'
    common=['--soft-artifact-dir',str(source),'--silhouettes','x.png']
    for tail in [['--output',str(source/'overwrite')],['--output',str(tmp_path)],
                 ['--output',str(tmp_path/'new'),'--max-evaluations','1025']]:
        with pytest.raises(SystemExit): parse(common+tail)
    args=parse(common+['--output',str(tmp_path/'new'),'--no-joint-edits'])
    assert args.no_joint_edits and args.max_evaluations==128


def test_complete_trace_and_live_callbacks_are_json_serializable():
    v,selected,desired=scene(); progress=[]
    def save(row): progress.append(json.dumps(row,allow_nan=False))
    _,trace,_=relocate(v,desired.astype(float),selected,render_binary(desired,cameras(),48),cameras(),
                       longest_mm=24.,max_evaluations=32,callback=save)
    assert progress
    assert json.loads(json.dumps(trace,allow_nan=False))['evaluated_candidates']==len(progress)


def test_probe_rejects_missing_second_island():
    runner=importlib.import_module('experiments.shadowart_vnext_dev.run_relocation128_probe_20260912')
    for islands in ([],[np.array([[1,2,3]])]):
        with pytest.raises(ValueError,match='exactly two'):
            runner.require_two_islands(islands)
    islands=[np.array([[1,2,3]]),np.array([[4,5,6]])]
    assert runner.require_two_islands(islands) is islands


@pytest.mark.parametrize('kwargs',[{'max_evaluations':-1},{'max_evaluations':True},
                                    {'max_evaluations':1025},{'joint_edits':'yes'}])
def test_relocation_rejects_invalid_search_controls(kwargs):
    v,selected,desired=scene()
    with pytest.raises(ValueError):
        relocate(v,desired.astype(float),selected,render_binary(desired,cameras(),48),
                 cameras(),longest_mm=24.,**kwargs)
