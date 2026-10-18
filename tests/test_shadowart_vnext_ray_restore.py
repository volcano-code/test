import importlib
from pathlib import Path
import numpy as np
import pytest


def module():
    assert (Path(__file__).parents[1]/'experiments/shadowart_vnext/ray_restore.py').is_file(), 'ray restoration missing'
    return importlib.import_module('experiments.shadowart_vnext.ray_restore')


def test_restores_attached_shadow_without_deleting_exclusive_island():
    from experiments.shadowart_vnext.contracts import angle_camera
    from experiments.shadowart_vnext.rendering import render_binary
    v=np.zeros((16,)*3,bool); v[4:10,4:10,4:10]=True; v[13,13,13]=True
    goal=v.copy(); goal[10:12,6:8,6:8]=True
    cams=[angle_camera(0,0,0)]
    targets=render_binary(goal,cams,32,extent=2.)
    result,trace=module().restore(v,v.astype(float),targets,cams,extent=2.,width=2,steps=4,mesh_feedback=True)
    assert result[v].all()
    assert np.array_equal(render_binary(result,cams,32,2.),targets)
    assert trace['history']
    assert trace['final_mesh_quality']['mean_iou'] >= .95
    assert trace['publication_eligible'] is False


def test_zero_budget_is_identity_and_invalid_parameters_fail():
    from experiments.shadowart_vnext.contracts import angle_camera
    from experiments.shadowart_vnext.rendering import render_binary
    v=np.zeros((8,)*3,bool); v[2:6,2:6,2:6]=True
    cams=[angle_camera(0,0,0)]; targets=render_binary(v,cams,16)
    out,trace=module().restore(v,v.astype(float),targets,cams,steps=0)
    assert np.array_equal(out,v)
    assert not trace['history']
    for kwargs in ({'width':0},{'steps':-1},{'steps':1.5}):
        with pytest.raises(ValueError):
            module().restore(v,v.astype(float),targets,cams,**kwargs)


def test_respects_existing_cavity_and_input_immutability():
    from experiments.shadowart_vnext.contracts import angle_camera
    from experiments.shadowart_vnext.rendering import render_binary
    v=np.zeros((12,)*3,bool); v[2:10,2:10,2:10]=True; v[4:8,4:8,4:8]=False
    before=v.copy(); cams=[angle_camera(0,0,0)]
    out,_=module().restore(v,v.astype(float),render_binary(v,cams,24),cams,steps=2)
    assert not out[4:8,4:8,4:8].any()
    assert np.array_equal(v,before)
