"""A complete replacement must recover shadow after a temporarily bad deletion."""
import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from experiments.shadowart_vnext.rendering import render_binary


def api():
    name = 'experiments.shadowart_vnext.component_replacement'
    assert importlib.util.find_spec(name) is not None, 'component replacement module missing'
    return importlib.import_module(name)


def front():
    return SimpleNamespace(origin=(0., 0., 2.), right=(1., 0., 0.), up=(0., 1., 0.),
                           direction=(0., 0., -1.), scale=1.7)


def scene():
    root = np.zeros((24,) * 3, bool); root[2:10, 4:20, 2:8] = True
    island = np.zeros_like(root); island[8:16, 4:20, 14:16] = True
    return root | island, island


def test_complete_block_sequence_replaces_island_at_another_depth():
    v, part = scene(); before = v.copy()
    target = render_binary(v, [front()], 48)
    out, trace, candidate = api().replace_component(v, v.astype(float), part, target,
        [front()], longest_mm=24., max_steps=128)
    assert trace['accepted']
    assert trace['base_quality']['mean_iou'] < .7, 'the temporary deletion must be allowed'
    assert len(trace['history']) > 4, 'a multi-block reconstruction, not the old four-box limit'
    assert ndi.label(out)[1] == 1
    assert not out[part].any(), 'same shadow must be rebuilt at root-attached depth'
    assert out[2:10, 4:20, 2:8].all(), 'retained main body must not be carved'
    np.testing.assert_array_equal(render_binary(out, [front()], 48), target)
    np.testing.assert_array_equal(v, before)
    np.testing.assert_array_equal(out, candidate)
    assert trace['publication_eligible'] is False
    assert trace['neck_guard_passed'] is True


def test_incomplete_replacement_is_diagnostic_only_and_rolls_back():
    v, part = scene()
    out, trace, candidate = api().replace_component(v, v.astype(float), part,
        render_binary(v, [front()], 48), [front()], longest_mm=24., max_steps=1)
    assert not trace['accepted']
    assert 'PROJECTION_REGRESSION' in trace['reasons']
    np.testing.assert_array_equal(out, v)
    assert not np.array_equal(candidate, v)
    assert len(trace['history']) == 1


def test_neck_frontier_requires_core_face_adjacency_not_corner_touch():
    root = np.zeros((10,) * 3, bool); root[2:5, 2:5, 2:5] = True
    lower = api().neck_frontier(root, np.zeros_like(root), 2)
    rows = {tuple(p) for p in lower}
    assert (4, 2, 2) in rows
    assert (4, 4, 4) not in rows
    protected = np.zeros_like(root); protected[5, 2, 2] = True
    rows = {tuple(p) for p in api().neck_frontier(root, protected, 2)}
    assert (4, 2, 2) not in rows
    assert all(min(p) >= 0 and max(p) <= 8 for p in rows)


def test_no_neck_core_returns_unchanged_not_a_fake_connection():
    v = np.zeros((12,) * 3, bool); v[2:10, 2, 2] = True
    part = np.zeros_like(v); part[5, 5, 5] = True; v |= part
    out, trace, _ = api().replace_component(v, v.astype(float), part,
        render_binary(v, [front()], 24), [front()], longest_mm=12., max_steps=8)
    assert not trace['accepted']
    assert trace['stop_reason'] == 'NO_ROOT_CONNECTED_NECK_FRONTIER'
    np.testing.assert_array_equal(out, v)


def test_whole_path_escapes_a_frontier_with_no_positive_single_block():
    v = np.zeros((24,) * 3, bool); v[4:10, 4:10, 4:10] = True
    part = np.zeros_like(v); part[12:16, 4:8, 12:16] = True; v |= part
    right = SimpleNamespace(origin=(2., 0., 0.), right=(0., 0., -1.), up=(0., 1., 0.),
                            direction=(-1., 0., 0.), scale=1.7)
    cameras = [front(), right]; target = render_binary(v, cameras, 48)
    _, greedy, _ = api().replace_component(v, v.astype(float), part, target, cameras,
        longest_mm=24., max_steps=32)
    assert greedy['stop_reason'] == 'NO_POSITIVE_GREEDY_RAY_GAIN'
    assert not greedy['history']
    out, routed, candidate = api().replace_component(v, v.astype(float), part, target, cameras,
        longest_mm=24., max_steps=32, whole_path=True)
    assert any(r.get('operation') == 'whole_thick_path' for r in routed['history'])
    assert routed['candidate_quality']['mean_iou'] > greedy['candidate_quality']['mean_iou']
    assert routed['consumed_block_steps'] <= 32
    assert routed['neck_guard_passed']
    assert routed['path_attempts'][0]['routes']
    assert all('status' in r for a in routed['path_attempts'] for r in a['routes'])
    # The literal two-view target contains gaps; a better diagnostic is not a
    # valid replacement for the original perfect but disconnected shadows.
    assert not routed['accepted']
    np.testing.assert_array_equal(out, v)
    assert not np.array_equal(candidate, v)


def test_original_hollow_island_cavity_cannot_be_opened_by_replacement():
    root = np.zeros((24,) * 3, bool); root[2:10, 2:22, 2:10] = True
    part = np.zeros_like(root); part[10:18, 6:18, 14:22] = True
    part[11:17, 7:17, 15:21] = False
    v = root | part
    out, trace, _ = api().replace_component(v, v.astype(float), part,
        render_binary(v, [front()], 48), [front()], longest_mm=24., max_steps=1)
    assert 'ORIGINAL_CAVITY_CHANGED' in trace['reasons']
    np.testing.assert_array_equal(out, v)


def test_whole_path_can_improve_even_when_its_endpoint_alone_regresses():
    root = np.zeros((16,) * 3, bool); root[2:5, 4:10, 3:7] = True
    selected = np.zeros_like(root); selected[11:14, 4:10, 3:7] = True
    original = root | selected
    desired = root.copy(); desired[6:10, 4:10, 3:7] = True
    target = render_binary(desired, [front()], 32)
    endpoint = root.copy(); endpoint[10:12, 4:6, 3:5] = True
    projection = render_binary(endpoint, [front()], 32)
    assert (projection & target).sum() / (projection | target).sum() == pytest.approx(9 / 23)
    _, trace, candidate = api().replace_component(original, np.ones_like(root, float),
        selected, target, [front()], longest_mm=16., block_width=2,
        max_steps=32, whole_path=True)
    assert trace['base_quality']['mean_iou'] == pytest.approx(3 / 7)
    paths = [row for row in trace['history'] if row['operation'] == 'whole_thick_path']
    assert paths, 'a bad endpoint alone must not exclude a beneficial complete path'
    assert paths[0]['quality']['mean_iou'] > 3 / 7
    assert trace['candidate_quality']['mean_iou'] > 3 / 7
    assert trace['neck_guard_passed']
    assert all(a['evaluated_routes'] <= 8 for a in trace['path_attempts'])
    assert sum(row['block_steps'] for row in trace['history']) == trace['consumed_block_steps'] <= 32
    assert candidate[root].all()
    np.testing.assert_array_equal(original, root | selected)


def test_temporary_route_dip_never_relaxes_final_projection_gate():
    v = np.zeros((128,) * 3, bool); v[4:64, 4:64, 4:64] = True
    part = np.zeros_like(v); part[67:69, 20:22, 67:69] = True; v |= part
    right = SimpleNamespace(origin=(2., 0., 0.), right=(0., 0., -1.), up=(0., 1., 0.),
                            direction=(-1., 0., 0.), scale=1.7)
    cameras = [front(), right]; target = render_binary(v, cameras, 128)
    out, trace, _ = api().replace_component(v, v.astype(float), part, target, cameras,
        longest_mm=128., max_steps=32, whole_path=True, allow_route_dip=True)
    previous = trace['base_quality']['mean_iou'] + trace['base_quality']['worst_iou']
    dips = []
    for row in trace['history']:
        objective = row['quality']['mean_iou'] + row['quality']['worst_iou']
        if row['operation'] == 'whole_thick_path' and objective < previous:
            dips.append(previous - objective)
        previous = objective
    assert dips and max(dips) <= .002 + 1e-12
    assert not trace['accepted']
    assert 'PROJECTION_REGRESSION' in trace['reasons']
    np.testing.assert_array_equal(out, v)


@pytest.mark.parametrize('selection', ['empty', 'partial', 'main', 'outside'])
def test_replacement_rejects_invalid_component_selection(selection):
    v, part = scene()
    if selection == 'empty': part[:] = False
    if selection == 'partial': part[8, 4, 14] = False
    if selection == 'main': part = v & ~part
    if selection == 'outside': part[0, 0, 0] = True
    with pytest.raises(ValueError):
        api().replace_component(v, v.astype(float), part, render_binary(v, [front()], 48),
                                [front()], max_steps=0)


@pytest.mark.parametrize('kwargs', [{'max_steps':True}, {'max_steps':-1}, {'max_steps':1025},
    {'block_width':1}, {'block_width':25}, {'block_width':2.5}, {'longest_mm':0}, {'whole_path':'yes'}])
def test_bad_or_physically_underwidth_budgets_fail_before_search(kwargs):
    v, part = scene()
    with pytest.raises(ValueError):
        api().replace_component(v, v.astype(float), part, render_binary(v, [front()], 48),
                               [front()], **({'longest_mm':24.} | kwargs))


def test_probe_rejects_source_tree_outputs_and_unbounded_budgets(tmp_path):
    name = 'experiments.shadowart_vnext_dev.run_replacement128_probe_20260910'
    assert importlib.util.find_spec(name) is not None, 'replacement probe missing'
    parse = importlib.import_module(name).parse_args
    source = tmp_path / 'soft'
    common = ['--soft-artifact-dir', str(source), '--silhouettes', 'x.png']
    for tail in (['--output', str(source / 'rewrite')], ['--output', str(tmp_path)],
                 ['--output', str(tmp_path / 'new'), '--steps', '1025'],
                 ['--output', str(tmp_path / 'new'), '--block-width', '1']):
        with pytest.raises(SystemExit):
            parse(common + tail)
    args = parse(common + ['--output', str(tmp_path / 'new'), '--steps', '64'])
    assert args.steps == 64
    args = parse(common + ['--output', str(tmp_path / 'new'), '--whole-path'])
    assert args.whole_path is True
    with pytest.raises(SystemExit):
        parse(common + ['--output', str(tmp_path / 'new'), '--allow-route-dip'])
    args = parse(common + ['--output', str(tmp_path / 'new'), '--whole-path', '--allow-route-dip'])
    assert args.allow_route_dip is True
