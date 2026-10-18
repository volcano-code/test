import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from experiments.shadowart_vnext.rendering import render_binary


def api():
    name = 'experiments.shadowart_vnext.regional_geometry'
    assert importlib.util.find_spec(name) is not None, 'regional geometry missing'
    return importlib.import_module(name)


def front():
    return SimpleNamespace(origin=(0., 0., 2.), right=(1., 0., 0.), up=(0., 1., 0.),
                           direction=(0., 0., -1.), scale=1.7)


def fixture():
    v = np.zeros((16,) * 3, bool)
    v[3:8, 4:12, 3:7] = True
    v[8:10, 4:12, 12:14] = True
    return v, render_binary(v, [front()], 32)


def test_regional_keeps_main_body_and_recovers_exclusive_shadow():
    v, target = fixture()
    before = v.copy()
    out, trace, search = api().repair_regional(v, v.astype(float), target, [front()],
                                              longest_mm=16., max_regions=1, max_primitives=32)
    assert search[3:8, 4:12, 3:7].all()
    assert ndi.label(search)[1] == 1
    np.testing.assert_array_equal(render_binary(search, [front()], 32), target)
    np.testing.assert_array_equal(v, before)
    assert trace['accepted_regions'] == 1
    assert trace['publication_eligible'] is False
    replay = before.copy()
    for row in trace['history']:
        if row['accepted']:
            replay.ravel()[row['removed_indices']] = False
            replay.ravel()[row['added_indices']] = True
    np.testing.assert_array_equal(replay, search)


def test_zero_budget_returns_both_unchanged_archives():
    v, target = fixture()
    out, trace, search = api().repair_regional(v, v.astype(float), target, [front()], max_regions=0)
    np.testing.assert_array_equal(out, v)
    np.testing.assert_array_equal(search, v)
    assert trace['accepted_regions'] == 0


def test_large_region_order_does_not_only_select_nearest_single_voxel():
    from experiments.shadowart_vnext.regional_candidates import component_order
    v = np.zeros((20,) * 3, bool); v[2:10,2:10,2:10] = True
    v[10,10,10] = True
    v[15:18,15:18,15:18] = True
    assert len(component_order(v,2,order='nearest')[0]) == 1
    assert len(component_order(v,2,order='largest')[0]) == 27


def test_absolute_budget_cannot_accumulate_drift():
    initial = np.array([.96, .97])
    assert api().within_budget(np.array([.959, .969]), initial, .001, .002)
    assert not api().within_budget(np.array([.958, .968]), initial, .001, .002)
    assert not api().within_budget(np.array([.957, .973]), initial, .001, .002)


def test_cavity_check_rejects_opening_not_only_filling():
    v = np.zeros((12,) * 3, bool)
    v[2:10, 2:10, 2:10] = True
    v[4:8, 4:8, 4:8] = False
    from experiments.shadowart_vnext.geometry import _enclosed_voids
    protected = _enclosed_voids(v)
    opened = v.copy(); opened[2:5, 5, 5] = False
    assert not api().cavities_preserved(opened, protected)
    assert api().cavities_preserved(v, protected)


def test_new_corner_neck_cannot_hide_behind_global_component_improvement():
    from experiments.shadowart_vnext.manufacturing import thickness_audit
    before = np.zeros((32,) * 3, bool)
    before[2:30,2:14,2:14] = True
    before[22:30,22:30,22:30] = True
    after = before.copy(); after[22:30,22:30,22:30] = False
    after[13:17,13:17,13:17] = True
    assert not api().manufacturing_improves(thickness_audit(before,.4), thickness_audit(after,.4))
    retained = before.copy(); retained[22:30,22:30,22:30] = False
    assert not api().thick_join_preserved(after, retained, after & ~before, 4)


def test_embedded_thick_connection_passes_local_neck_guard():
    root = np.zeros((16,) * 3, bool); root[2:8,2:8,2:8] = True
    candidate = root.copy(); candidate[5:11,4:7,4:7] = True
    assert api().thick_join_preserved(candidate, root, candidate & ~root, 2)


@pytest.mark.parametrize('kwargs', [{'max_regions':-1}, {'max_regions':True},
    {'max_primitives':0}, {'mean_budget':.5}, {'view_budget':float('nan')}, {'time_limit':0}])
def test_invalid_regional_budgets_fail_early(kwargs):
    v, target = fixture()
    with pytest.raises(ValueError):
        api().repair_regional(v, v.astype(float), target, [front()], **kwargs)


def test_whole_neutral_bridge_is_proposed_without_stepwise_iou_gain():
    from experiments.shadowart_vnext.regional_candidates import propose
    v = np.zeros((16,) * 3, bool)
    v[4:10, 4:10, 2:6] = True
    v[4:10, 4:10, 12:14] = True
    part = np.argwhere(v & (np.indices(v.shape)[2] >= 12))
    target = render_binary(v, [front()], 32)
    base, _, primitives, _, kinds = propose(v, v.astype(float), part, target, [front()],
                                            np.zeros_like(v), 2, 1.7, 16)
    paths = [i for i,k in enumerate(kinds) if k['kind']=='retain_and_complete_thick_path']
    assert paths
    candidate = base.copy(); candidate.ravel()[primitives[paths[0]]] = True
    assert candidate[:, :, 8].any()
    assert ndi.label(candidate)[1] == 1
    np.testing.assert_array_equal(render_binary(candidate, [front()], 32), target)


def test_same_shadow_different_depths_are_not_all_deduplicated():
    from experiments.shadowart_vnext.regional_candidates import propose
    v, target = fixture()
    points = np.argwhere(v & (np.indices(v.shape)[2] >= 12))
    _, _, primitives, projections, _ = propose(v, v.astype(float), points, target, [front()],
                                              np.zeros_like(v), 2, 1.7, 32)
    signatures = [np.packbits(p).tobytes() for p in projections]
    pairs = [(i,j) for i in range(len(signatures)) for j in range(i)
             if signatures[i] == signatures[j] and not np.array_equal(primitives[i],primitives[j])]
    assert pairs, 'same projection must retain distinct geometric alternatives'


def test_regional_cli_rejects_ignored_budgets(tmp_path):
    from experiments.run_shadowart_vnext import parse_args
    common = ['--silhouettes','x.png','--soft-artifact-dir','soft','--output',str(tmp_path)]
    args = parse_args(common + ['--geometry-method','regional','--regional-regions','8'])
    assert args.geometry_method == 'regional' and args.regional_regions == 8
    args = parse_args(common + ['--geometry-method','regional','--regional-order','largest'])
    assert args.regional_order == 'largest'
    for tail in (['--regional-regions','8'], ['--geometry-method','regional','--joint-steps','8'],
                 ['--geometry-method','regional','--geometry-rounds','1'],
                 ['--geometry-method','regional','--disable-geometry','surface'],
                 ['--geometry-method','regional','--regional-regions','129']):
        with pytest.raises(SystemExit):
            parse_args(common + tail)


def test_regional_real_cli_snapshot_and_search_archive(tmp_path):
    import json
    from PIL import Image
    from experiments.run_shadowart_vnext import main
    mask = np.zeros((8,8), np.uint8); mask[2:6,2:6] = 255
    p = tmp_path/'target.png'; Image.fromarray(mask).save(p)
    common = ['--silhouettes',str(p),'--resolution','8','--image-size','8','--device','cpu']
    soft = tmp_path/'soft'
    assert main(common + ['--output',str(soft),'--stage','soft','--steps','1','--layout-candidates','1']) == 0
    output = tmp_path/'regional'
    code = main(common + ['--output',str(output),'--soft-artifact-dir',str(soft/'soft_training'),
                          '--geometry-method','regional','--regional-regions','0'])
    assert code == 2
    assert (output/'diagnostics/regional_search_xyz.npy').exists()
    assert (output/'source_snapshot/regional_solver.py').exists()
    trace = json.loads((output/'diagnostics/geometry_trace.json').read_text())
    assert trace['accepted_regions'] == 0
    assert not (output/'delivery').exists()
