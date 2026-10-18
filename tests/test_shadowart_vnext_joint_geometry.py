"""Compound connected repair, with real cameras and independent binary rays."""
import importlib
import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import ndimage as ndi

from experiments.shadowart_vnext.rendering import render_binary


def api():
    name = 'experiments.shadowart_vnext.joint_geometry'
    assert importlib.util.find_spec(name) is not None, 'joint geometry implementation missing'
    return importlib.import_module(name)


def front():
    return SimpleNamespace(origin=(0., 0., 2.), right=(1., 0., 0.),
                           up=(0., 1., 0.), direction=(0., 0., -1.), scale=1.7)


@pytest.mark.parametrize('oblique', [False, True])
def test_exact_block_quality_matches_full_rerender(oblique):
    from experiments.shadowart_vnext.contracts import build_contract
    cams = build_contract(3, tsf=0 if oblique else 1, image_size=19, resolution=12).cameras
    rng = np.random.default_rng(17)
    v = rng.random((12,) * 3) > .94
    target = rng.random((len(cams), 19, 19)) > .5
    boxes = np.array([[0, 0, 0], [3, 4, 5], [9, 9, 9]])
    scorer = api().BlockProjection(render_binary(v, cams, 19), target, cams, 12, 1.7)
    actual = scorer.qualities(boxes, 3)
    for i, lo in enumerate(boxes):
        candidate = v.copy()
        candidate[tuple(slice(a, a + 3) for a in lo)] = True
        p = render_binary(candidate, cams, 19)
        expected = (p & target).sum((1, 2)) / (p | target).sum((1, 2))
        np.testing.assert_allclose(actual[i], expected, atol=1e-12)


def test_shifted_out_of_view_boxes_and_incremental_union_match_complete_rays():
    cam = front()
    cam.origin = (1.4, .3, 2.)
    v = np.zeros((12,) * 3, bool)
    target = np.ones((1, 19, 19), bool)
    pred = render_binary(v, [cam], 19)
    scorer = api().BlockProjection(pred, target, [cam], 12, 1.7)
    for lo in (np.array([0, 0, 0]), np.array([9, 9, 9])):
        values = scorer.qualities(lo[None], 3)
        v[tuple(slice(a, a + 3) for a in lo)] = True
        rendered = render_binary(v, [cam], 19)
        np.testing.assert_allclose(values[0], rendered.mean((1, 2)), atol=1e-12)
        scorer.add(lo, 3)
        np.testing.assert_array_equal(scorer.prediction, rendered)
    assert not pred.any()
    assert target.all()


def island_fixture():
    v = np.zeros((16,) * 3, bool)
    v[3:8, 4:12, 3:7] = True
    v[8:11, 4:12, 12:14] = True
    return v, render_binary(v, [front()], 32)


def test_joint_transaction_replaces_exclusive_island_without_losing_shadow():
    v, target = island_fixture()
    original = v.copy()
    out, trace, diagnostic = api().repair_joint(v, v.astype(float), target, [front()],
                                               longest_mm=16., max_steps=64)
    assert ndi.label(v)[1] == 2
    assert ndi.label(out)[1] == 1
    np.testing.assert_array_equal(render_binary(out, [front()], 32), target)
    np.testing.assert_array_equal(v, original)
    assert trace['accepted']
    assert trace['root_quality']['mean_iou'] < 1.
    assert trace['final_quality']['mean_iou'] == 1.
    assert not out[8:11, 4:12, 12:14].any()


def test_insufficient_budget_rolls_back_but_retains_diagnostic_candidate():
    v, target = island_fixture()
    out, trace, diagnostic = api().repair_joint(v, v.astype(float), target, [front()], max_steps=0)
    np.testing.assert_array_equal(out, v)
    assert not trace['accepted']
    assert ndi.label(diagnostic)[1] == 1
    assert trace['publication_eligible'] is False


def test_zero_budget_cannot_commit_even_shadow_redundant_island_removal():
    v = np.zeros((16,) * 3, bool)
    v[3:10, 4:12, 3:7] = True
    v[4:8, 5:10, 12:14] = True
    out, trace, _ = api().repair_joint(v, v.astype(float), render_binary(v, [front()], 32),
                                      [front()], max_steps=0)
    np.testing.assert_array_equal(out, v)
    assert not trace['accepted']


def test_single_cell_growth_uses_face_adjacency_not_impossible_overlap():
    v = np.zeros((8,) * 3, bool)
    v[3, 3, 3] = True
    boxes = api()._eligible_boxes(v, np.zeros_like(v), 1)
    assert len(boxes) == 6
    assert np.all(np.abs(boxes - np.array([3, 3, 3])).sum(1) == 1)


def test_connected_cavity_is_never_filled_or_opened():
    v = np.zeros((16,) * 3, bool)
    v[2:14, 2:14, 2:14] = True
    v[6:10, 6:10, 6:10] = False
    out, trace, _ = api().repair_joint(v, v.astype(float), render_binary(v, [front()], 32),
                                      [front()], max_steps=4, longest_mm=16.)
    assert not out[6:10, 6:10, 6:10].any()
    np.testing.assert_array_equal(out, v)


@pytest.mark.parametrize('kwargs', [{'max_steps':-1}, {'max_steps':True}, {'extent':0},
                                  {'longest_mm':float('nan')}, {'min_neck_mm':0}])
def test_invalid_joint_settings_fail_early(kwargs):
    v, target = island_fixture()
    with pytest.raises(ValueError):
        api().repair_joint(v, v.astype(float), target, [front()], **kwargs)


@pytest.mark.parametrize('target', [np.array(1), np.ones((1, 0, 0)),
                                  np.ones((1, 4, 5)), np.ones((2, 4, 4))])
def test_malformed_targets_fail_before_rendering(target, monkeypatch):
    v, _ = island_fixture()
    def unexpected_render(*args, **kwargs):
        pytest.fail('invalid target reached expensive renderer')
    monkeypatch.setattr(api(), 'render_binary', unexpected_render)
    with pytest.raises(ValueError, match='targets'):
        api().repair_joint(v, v.astype(float), target, [front()])


def test_joint_cli_rejects_ignored_local_budgets_and_records_its_own(tmp_path):
    from experiments.run_shadowart_vnext import parse_args
    common = ['--silhouettes', 'x.png', '--soft-artifact-dir', 'soft', '--output', str(tmp_path / 'run')]
    args = parse_args(common + ['--geometry-method', 'joint', '--joint-steps', '64'])
    assert args.geometry_method == 'joint' and args.joint_steps == 64
    for tail in (['--geometry-method', 'joint', '--geometry-rounds', '32'],
                 ['--geometry-method', 'joint', '--disable-geometry', 'reinforcement'],
                 ['--joint-steps', '64'], ['--geometry-method', 'joint', '--joint-steps', '1025']):
        with pytest.raises(SystemExit):
            parse_args(common + tail)


def test_real_cli_joint_snapshot_diagnostics_and_no_false_publication(tmp_path):
    import json
    from PIL import Image
    from experiments.run_shadowart_vnext import main
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 2:6] = 255
    image = tmp_path / 'target.png'
    Image.fromarray(mask).save(image)
    soft = tmp_path / 'soft'
    common = ['--silhouettes', str(image), '--resolution', '8', '--image-size', '8', '--device', 'cpu']
    assert main(common + ['--output', str(soft), '--stage', 'soft', '--steps', '1', '--layout-candidates', '1']) == 0
    out = tmp_path / 'joint'
    result = main(common + ['--output', str(out), '--soft-artifact-dir', str(soft / 'soft_training'),
                            '--geometry-method', 'joint', '--joint-steps', '0'])
    assert result == 2
    assert (out / 'diagnostics/joint_connected_candidate_xyz.npy').exists()
    assert json.loads((out / 'GEOMETRY_CONFIG.json').read_text())['method'] == 'joint_connected_block_replacement_v1'
    assert not (out / 'delivery').exists()
