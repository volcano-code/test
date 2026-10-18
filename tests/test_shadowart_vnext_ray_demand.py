import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.shadowart_vnext.rendering import render_binary
from experiments.shadowart_vnext.regional_geometry import thick_join_preserved


def cameras():
    return [SimpleNamespace(origin=(0.,0.,2.),right=(1.,0.,0.),up=(0.,1.,0.),
                            direction=(0.,0.,-1.),scale=1.7)]


def family(*args,**kwargs):
    name='experiments.shadowart_vnext.ray_demand'
    assert importlib.util.find_spec(name), 'full-ray connected demand candidate family missing'
    return importlib.import_module(name).demand_primitives(*args,**kwargs)


def fixture():
    base=np.zeros((16,)*3,bool); base[2:6,2:14,2:6]=True
    desired=base.copy(); desired[5:10,3:5,3:5]=True; desired[5:10,11:13,3:5]=True
    target=render_binary(desired,cameras(),32)
    demand=target & ~render_binary(base,cameras(),32)
    return base,desired,target,demand


def test_demand_paths_cover_separate_patches_and_are_root_thick_connected():
    base,desired,target,demand=fixture(); saved=base.copy()
    primitives,meta,info=family(base,desired.astype(float),demand,target,cameras(),width=2,max_primitives=24)
    assert len(primitives)>1 and len(primitives)<=24
    union=base.copy()
    for indices in primitives:
        added=np.zeros_like(base); added.ravel()[indices]=True
        assert thick_join_preserved(base|added,base,added & ~base,2)
        union |= added
    assert not (demand & ~render_binary(union,cameras(),32)).any()
    assert len(meta)==len(primitives)
    json.dumps(info,allow_nan=False); json.dumps(meta,allow_nan=False)
    np.testing.assert_array_equal(base,saved)


def test_zero_budget_and_empty_demand_do_not_construct_candidates():
    base,desired,target,demand=fixture()
    assert family(base,desired.astype(float),demand,target,cameras(),max_primitives=0)[0]==[]
    assert family(base,desired.astype(float),np.zeros_like(demand),target,cameras())[0]==[]


def test_filament_root_cannot_anchor_thick_replacement():
    base,desired,target,demand=fixture(); base[:]=False; base[2:12,2,2]=True
    assert family(base,desired.astype(float),demand,target,cameras(),width=2)[0]==[]


@pytest.mark.parametrize('bad',[{'width':0},{'max_primitives':True},{'max_primitives':257}])
def test_family_rejects_invalid_controls(bad):
    base,desired,target,demand=fixture()
    with pytest.raises(ValueError): family(base,desired.astype(float),demand,target,cameras(),**bad)


def test_family_rejects_background_demand():
    base,desired,target,demand=fixture(); demand[0,0,0]=True
    with pytest.raises(ValueError): family(base,desired.astype(float),demand,target,cameras())


def replacement_scene():
    base,desired,_,_=fixture()
    selected=np.zeros_like(base)
    selected[3:10,3:5,12:14]=True; selected[3:10,11:13,12:14]=True
    selected[3:5,3:13,12:14]=True
    original=base|selected
    return original,selected,desired,render_binary(original,cameras(),32)


def replace(*args,**kwargs):
    name='experiments.shadowart_vnext.demand_replacement'
    assert importlib.util.find_spec(name), 'joint demand replacement transaction missing'
    return importlib.import_module(name).replace_by_demand(*args,**kwargs)


def test_joint_replacement_recovers_two_required_patches_single_cannot():
    original,selected,desired,target=replacement_scene(); saved=original.copy()
    out,trace,finalists=replace(original,desired.astype(float),selected,target,cameras(),
                              longest_mm=16.,max_primitives=24)
    assert trace['accepted'],trace
    np.testing.assert_array_equal(render_binary(out,cameras(),32),target)
    # Reusing some old coordinates is legal; retaining the entire old island
    # unchanged is not the contribution-replacement behavior of this fixture.
    assert np.any(selected & ~out)
    assert trace['candidate_manufacturing']['component_count_6']==1
    assert len(finalists)<=4 and len(trace['selected_primitives'])>1
    assert all(s['floor_policy']=='initial_view_floor' for s in trace['solves'])
    assert trace['finalists'][0]['changed_indices']
    single,single_trace,_=replace(original,desired.astype(float),selected,target,cameras(),
                                  longest_mm=16.,max_primitives=24,single_only=True)
    assert not single_trace['accepted']
    np.testing.assert_array_equal(single,original)
    np.testing.assert_array_equal(original,saved)
    json.dumps(trace,allow_nan=False)


def test_zero_demand_replacement_budget_rolls_back():
    original,selected,desired,target=replacement_scene()
    out,trace,_=replace(original,desired.astype(float),selected,target,cameras(),
                         longest_mm=16.,max_primitives=0)
    assert not trace['accepted']
    np.testing.assert_array_equal(out,original)


def test_demand_primitives_respect_explicit_void_and_oblique_camera():
    base,desired,_,_=fixture()
    angle=.35; c=cameras()[0]
    c.origin=(2*np.sin(angle),0.,2*np.cos(angle))
    c.direction=(-np.sin(angle),0.,-np.cos(angle)); c.right=(np.cos(angle),0.,-np.sin(angle))
    target=render_binary(desired,[c],24); demand=target & ~render_binary(base,[c],24)
    protected=np.zeros_like(base); protected[6:8,6:9,6:9]=True
    primitives,_,_=family(base,desired.astype(float),demand,target,[c],max_primitives=8,protected=protected)
    assert primitives
    for indices in primitives:
        assert not protected.ravel()[indices].any()


def test_demand_probe_rejects_source_overlap_and_invalid_limits(tmp_path):
    name='experiments.shadowart_vnext_dev.run_demand128_probe_20260912'
    assert importlib.util.find_spec(name),'frozen multi-finalist probe missing'
    parse=importlib.import_module(name).parse_args
    common=['--soft-artifact-dir',str(tmp_path/'soft'),'--silhouettes','x.png']
    for tail in [['--output',str(tmp_path)],['--output',str(tmp_path/'soft/sub')],
                 ['--output',str(tmp_path/'new'),'--max-primitives','257'],
                 ['--output',str(tmp_path/'new'),'--max-primitives','65'],
                 ['--output',str(tmp_path/'new'),'--time-limit','11'],
                 ['--output',str(tmp_path/'new'),'--time-limit','nan']]:
        with pytest.raises(SystemExit): parse(common+tail)
    args=parse(common+['--output',str(tmp_path/'new'),'--single-only'])
    assert args.single_only and args.max_primitives==64


@pytest.mark.parametrize('bad',[{'time_limit':0},{'single_only':'yes'},
                               {'callback':42},{'max_primitives':True}])
def test_transaction_rejects_bad_controls(bad):
    original,selected,desired,target=replacement_scene()
    with pytest.raises(ValueError): replace(original,desired.astype(float),selected,target,cameras(),**bad)


def test_probe_wrong_resolution_records_error_and_does_not_overwrite(tmp_path):
    from experiments.shadowart_vnext.contracts import build_contract
    name='experiments.shadowart_vnext_dev.run_demand128_probe_20260912'
    assert importlib.util.find_spec(name),'frozen multi-finalist probe missing'
    main=importlib.import_module(name).main
    source=tmp_path/'soft'; source.mkdir()
    (source/'CAMERA_CONTRACT.json').write_text(json.dumps(build_contract(1,image_size=8,resolution=8).to_dict()))
    output=tmp_path/'out'
    args=['--soft-artifact-dir',str(source),'--silhouettes','unused.png','--output',str(output)]
    with pytest.raises(ValueError,match='128'): main(args)
    report=(output/'REPORT.json').read_bytes()
    assert json.loads(report)['status']=='ERROR'
    with pytest.raises(FileExistsError): main(args)
    assert (output/'REPORT.json').read_bytes()==report


def test_probe_rejects_other_native128_camera_before_loading_fields(tmp_path):
    from experiments.shadowart_vnext.contracts import build_contract
    runner=importlib.import_module('experiments.shadowart_vnext_dev.run_demand128_probe_20260912')
    source=tmp_path/'soft'; source.mkdir()
    (source/'CAMERA_CONTRACT.json').write_text(json.dumps(build_contract(3,tsf=1,resolution=128).to_dict()))
    with pytest.raises(ValueError,match='FAILED_FROZEN_CONTRACT'):
        runner.main(['--soft-artifact-dir',str(source),'--silhouettes','unused.png','--output',str(tmp_path/'out')])
