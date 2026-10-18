"""Shadow-equivalent filling must retain topology/cavities and source identity."""
import importlib
from pathlib import Path
import numpy as np
import pytest
from scipy import ndimage as ndi


def module():
    assert (Path(__file__).parents[1]/'experiments/shadowart_vnext/ray_closure.py').is_file(), 'ray closure missing'
    return importlib.import_module('experiments.shadowart_vnext.ray_closure')


def test_fills_exterior_redundant_notch_without_changing_three_shadows():
    from experiments.shadowart_vnext.contracts import build_contract
    from experiments.shadowart_vnext.rendering import render_binary
    v=np.zeros((12,)*3,bool); v[2:10,2:10,2:10]=True
    # At z=2 there is an interior notch: x/y ray supports remain behind it.
    v[5,5,2:6]=False
    original=v.copy(); cameras=build_contract(3,tsf=1,resolution=12).cameras
    out,trace=module().shadow_closure(v,cameras)
    assert out[5,5,2:6].all()
    assert np.array_equal(v,original)
    assert np.array_equal(render_binary(v,cameras,37),render_binary(out,cameras,37))
    assert trace['added_voxels']==4
    assert trace['publication_eligible'] is False


def test_never_fills_enclosed_cavity():
    from experiments.shadowart_vnext.contracts import build_contract
    v=np.zeros((14,)*3,bool); v[2:12,2:12,2:12]=True; v[5:9,5:9,5:9]=False
    out,_=module().shadow_closure(v,build_contract(3,tsf=1,resolution=14).cameras)
    assert not out[5:9,5:9,5:9].any()
    assert np.array_equal(out,v)


def test_retains_islands_without_creating_new_ones():
    from experiments.shadowart_vnext.contracts import build_contract
    v=np.zeros((12,)*3,bool); v[2:6,2:6,2:6]=True; v[9,9,9]=True
    out,trace=module().shadow_closure(v,build_contract(3,tsf=1,resolution=12).cameras)
    assert out[v].all()
    assert ndi.label(out)[1] <= ndi.label(v)[1]
    assert trace['removed_voxels']==0


def test_rejects_oblique_and_nonboolean_inputs():
    from experiments.shadowart_vnext.contracts import angle_camera
    with pytest.raises(ValueError,match='cardinal'):
        module().shadow_closure(np.ones((8,)*3,bool),[angle_camera(0,30,0)])
    with pytest.raises(ValueError):
        module().shadow_closure(np.ones((8,)*3),[angle_camera(0,0,0)])


def test_thick_closure_does_not_introduce_uncovered_thin_material():
    from experiments.shadowart_vnext.contracts import build_contract
    from experiments.shadowart_vnext.manufacturing import _block_cover
    v=np.zeros((12,)*3,bool); v[2:7,2:7,2:7]=True
    v[2:10,2:10,2:10] |= np.random.default_rng(42).random((8,)*3)<.15
    out,trace=module().shadow_closure(v,build_contract(3,tsf=1,resolution=12).cameras,
                                     minimum_block_width=3)
    assert not (out & ~v & ~_block_cover(out,3)[1]).any()
    assert out[v].all()
    assert trace['minimum_block_width']==3
