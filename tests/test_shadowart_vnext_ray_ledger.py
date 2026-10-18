import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.shadowart_vnext.rendering import render_binary


def camera(angle=0., origin_distance=2.):
    d = np.array([np.sin(angle), 0., -np.cos(angle)])
    return SimpleNamespace(origin=tuple(-origin_distance*d), right=(np.cos(angle), 0., np.sin(angle)),
                           up=(0., 1., 0.), direction=tuple(d), scale=1.7)


def ledger(v, cameras, size):
    module = 'experiments.shadowart_vnext.ray_ledger'
    assert importlib.util.find_spec(module), 'exact ray ledger missing'
    return importlib.import_module(module).RayLedger(v, cameras, size)


def test_combined_deletion_counts_all_occluders_and_replacement_once():
    v = np.zeros((8,)*3, bool); v[2, 3, [1, 6]] = True
    before = v.copy(); l = ledger(v, [camera()], 8)
    assert l.counts[0, 4, 2] == 2
    remove = np.zeros_like(v); remove[2, 3, 1] = True
    _, counts = l.evaluate(np.zeros_like(v), remove)
    assert counts[0, 4, 2] == 1
    remove[2, 3, 6] = True
    _, counts = l.evaluate(np.zeros_like(v), remove)
    assert counts[0, 4, 2] == 0
    add = np.zeros_like(v); add[2, 3, 4] = True
    candidate, counts = l.evaluate(add, remove)
    assert counts[0, 4, 2] == 1 and candidate.sum() == 1
    np.testing.assert_array_equal(v, before)
    np.testing.assert_array_equal(l.counts > 0, render_binary(before, [camera()], 8))


@pytest.mark.parametrize('angle,distance', [(0.,2.),(np.pi,2.),(.47,2.),(.47,0.)])
def test_counts_edits_and_backprojection_match_complete_traversal(angle, distance):
    rng=np.random.default_rng(42); v=rng.random((8,)*3)<.2
    cams=[camera(angle,distance)]; l=ledger(v,cams,16)
    np.testing.assert_array_equal(l.prediction,render_binary(v,cams,16))
    remove=v & (rng.random(v.shape)<.4); add=~v & (rng.random(v.shape)<.2)
    candidate, counts=l.evaluate(add,remove)
    np.testing.assert_array_equal(counts>0,render_binary(candidate,cams,16))
    weights=rng.normal(size=(1,16,16))
    assert np.sum(l.backproject(weights)*v)==pytest.approx(np.sum(weights*l.counts),abs=1e-9)


def test_edit_validation_and_source_ownership():
    v=np.zeros((8,)*3,bool); v[2,3,4]=True; l=ledger(v,[camera()],8)
    v[:]=False
    assert l.counts.sum()==1
    m=np.zeros_like(v); m[2,3,4]=True
    with pytest.raises(ValueError): l.evaluate(m,m)
    with pytest.raises(ValueError): l.evaluate(m.astype(float),~m)
    with pytest.raises(ValueError): l.backproject(np.full((1,8,8),np.nan))


def test_ledger_freezes_camera_values_before_caller_mutates_them():
    v=np.zeros((8,)*3,bool); v[2,3,4]=True
    c=camera(.47); l=ledger(v,[c],16); original_counts=l.counts.copy()
    c.origin=(100.,100.,100.)
    _,counts=l.evaluate(np.zeros_like(v),v)
    assert not counts.any()
    np.testing.assert_array_equal(l.counts,original_counts)
