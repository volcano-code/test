"""Actual-ray tests for the opt-in binary forward / optical surrogate probe."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.shadowart_vnext import rendering, training


def cameras():
    return [SimpleNamespace(origin=(0., 0., 2.), right=(1., 0., 0.),
                            up=(0., 1., 0.), direction=(0., 0., -1.), scale=1.7)]


def hard_projection(logits, cams=None, size=4):
    helper = getattr(training, 'render_hard_training_projection', None)
    assert callable(helper), 'missing hard-forward projection implementation'
    return helper(logits, cameras() if cams is None else cams, size, extent=1.7)


def test_accumulated_subthreshold_opacity_is_not_a_hard_shadow():
    logits = torch.full((4, 4, 4), -2., requires_grad=True)
    prediction = hard_projection(logits)
    assert torch.count_nonzero(prediction) == 0
    prediction.sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('depth', [0, 2, 3])
def test_one_occupied_sheet_anywhere_casts_exact_binary_shadow(depth):
    logits = torch.full((4, 4, 4), -8.)
    logits[:, :, depth] = 0.
    assert torch.equal(hard_projection(logits), torch.ones((1, 4, 4)))


def test_binary_forward_preserves_explicit_optical_surrogate_gradient():
    logits = torch.full((4, 4, 4), -8., requires_grad=True)
    hard_projection(logits).sum().backward()
    actual = logits.grad.clone()
    logits.grad = None
    sigma, _ = training.logits_to_density(logits)
    rendering.render_optical(sigma, cameras(), 4).sum().backward()
    assert (actual > 0).all()
    torch.testing.assert_close(actual, logits.grad)


def test_oblique_and_opposite_hard_forward_match_complete_binary_rays():
    from experiments.shadowart_vnext.contracts import build_contract
    cams = build_contract(3, tsf=0, image_size=8, resolution=8).cameras
    logits = torch.randn((8, 8, 8), generator=torch.Generator().manual_seed(17)) - .8
    expected = rendering.render_binary(logits.numpy() >= 0., cams, 13)
    actual = hard_projection(logits, cams, 13).numpy()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('mode', ['dense_soft', 'dense_hard_st'])
def test_dense_training_requires_original_source_derived_targets(mode):
    assert 'projection_mode' in __import__('inspect').signature(training.train_soft).parameters
    with pytest.raises(ValueError, match='source-derived'):
        training.train_soft(np.zeros((1, 2, 2), bool), cameras(), image_size=2,
                            resolution=4, steps=4, projection_mode=mode)


def test_matched_dense_arms_share_initialization_warmup_and_binary_export():
    assert 'projection_mode' in __import__('inspect').signature(training.train_soft).parameters
    targets = np.zeros((1, 2, 2), bool)
    targets[:, 0, 0] = True
    dense = np.zeros((1, 4, 4), bool)
    dense[:, 0:2, 0:2] = True
    kwargs = dict(resolution=4, image_size=2, steps=4, seed=17, supersampled_targets=dense)
    a = training.train_soft(targets, cameras(), projection_mode='dense_soft', **kwargs)
    b = training.train_soft(targets, cameras(), projection_mode='dense_hard_st', **kwargs)
    assert a['metadata']['initial_logits_sha256'] == b['metadata']['initial_logits_sha256']
    assert a['history'][0]['gradient_norm'] == b['history'][0]['gradient_norm']
    assert [row['hard_forward_weight'] for row in b['history']] == [0., 1., 1., 1.]
    assert a['metadata']['training_projection']['image_size'] == 4
    assert b['metadata']['training_projection']['target_rule'] == 'source_derived_binary'
    np.testing.assert_array_equal(b['occupancy'], b['guidance'] >= .5)


def test_dense_training_does_not_substitute_nominal_targets_for_dense_truth():
    assert 'projection_mode' in __import__('inspect').signature(training.train_soft).parameters
    dense = np.zeros((1, 4, 4), bool)
    dense[:, 0, 0] = True
    kwargs = dict(resolution=4, image_size=2, steps=2, seed=17,
                  supersampled_targets=dense, projection_mode='dense_soft')
    a = training.train_soft(np.zeros((1, 2, 2), bool), cameras(), **kwargs)
    b = training.train_soft(np.ones((1, 2, 2), bool), cameras(), **kwargs)
    np.testing.assert_array_equal(a['density'], b['density'])


def test_cli_routes_dense_training_and_rejects_artifact_override(tmp_path):
    from experiments.run_shadowart_vnext import parse_args
    common = ['--silhouettes', 'x.png', '--output', str(tmp_path / 'run')]
    args = parse_args(common + ['--projection-mode', 'dense_hard_st'])
    assert args.projection_mode == 'dense_hard_st'
    with pytest.raises(SystemExit):
        parse_args(common + ['--soft-artifact-dir', 'soft', '--projection-mode', 'area_soft'])
    with pytest.raises(SystemExit):
        parse_args(common + ['--projection-mode', 'dense_soft', '--calibration-steps', '500'])


def test_dense_cli_snapshot_preserves_mode_and_artifact_semantics(tmp_path):
    import json
    from PIL import Image
    from experiments.run_shadowart_vnext import main
    from experiments.shadowart_vnext.artifacts import load_soft
    from experiments.shadowart_vnext.contracts import CameraContract
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 2:6] = 255
    source = tmp_path / 'square.png'
    Image.fromarray(mask).save(source)
    out = tmp_path / 'run'
    assert main(['--silhouettes', str(source), '--output', str(out), '--stage', 'soft',
                 '--resolution', '4', '--image-size', '4', '--steps', '4', '--device', 'cpu',
                 '--layout-candidates', '1', '--projection-mode', 'dense_hard_st']) == 0
    contract = CameraContract.from_dict(json.loads((out / 'CAMERA_CONTRACT.json').read_text()))
    fields = load_soft(out / 'soft_training', contract)
    assert fields['metadata']['training_projection']['mode'] == 'dense_hard_st'
    assert fields['history'][-1]['hard_forward_weight'] == 1.
    assert not (out / 'delivery').exists()


def test_dense_hard_joint_chunks_and_full_batch_have_same_update():
    targets = np.zeros((2, 2, 2), bool)
    targets[:, 0, 0] = True
    dense = np.repeat(np.repeat(targets, 2, axis=1), 2, axis=2)
    kwargs = dict(resolution=4, image_size=2, steps=4, seed=17,
                  supersampled_targets=dense, projection_mode='dense_hard_st')
    a = training.train_soft(targets, cameras() * 2, view_chunk_size=1, **kwargs)
    b = training.train_soft(targets, cameras() * 2, view_chunk_size=2, **kwargs)
    np.testing.assert_allclose(a['density'], b['density'], atol=1e-6, rtol=1e-6)


def test_abbreviated_training_mode_cannot_bypass_replay_guard(tmp_path):
    from experiments.run_shadowart_vnext import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--silhouettes', 'x.png', '--output', str(tmp_path / 'run'),
                    '--soft-artifact-dir', 'soft', '--projection-m', 'dense_hard_st'])


@pytest.fixture(scope='module')
def dense_fields():
    from dataclasses import replace
    from experiments.shadowart_vnext.contracts import build_contract, mask_hash, file_hash
    targets = np.zeros((1, 4, 4), bool)
    targets[:, 1:3, 1:3] = True
    contract = build_contract(1, image_size=4, resolution=4)
    contract = replace(contract, input_hashes=(file_hash(__file__),),
                       target_hashes=(mask_hash(targets[0]),))
    fields = training.train_soft(targets, contract.cameras, resolution=4, image_size=4, steps=4,
                                projection_mode='dense_hard_st',
                                supersampled_targets=np.repeat(np.repeat(targets, 2, 1), 2, 2))
    return contract, fields


@pytest.mark.parametrize('key,value', [('mode', 'invented'), ('image_size', 1),
    ('target_rule', 'nominal_binary'), ('target_hashes', []),
    ('backward_rule', 'exact_binary_derivative'), ('hard_start_fraction', .1),
    ('hard_full_fraction', .2), ('initial_logits_sha256', 'bad'),
    ('history_weight', 0.), ('missing_projection', None)])
def test_artifact_rejects_false_projection_provenance(tmp_path, dense_fields, key, value):
    import copy
    from experiments.shadowart_vnext.artifacts import save_soft
    contract, original = dense_fields
    fields = copy.deepcopy(original)
    if key == 'initial_logits_sha256':
        fields['metadata'][key] = value
    elif key == 'history_weight':
        fields['history'][-1]['hard_forward_weight'] = value
    elif key == 'missing_projection':
        fields['metadata'].pop('training_projection')
    else:
        fields['metadata']['training_projection'][key] = value
    with pytest.raises(ValueError, match='PROJECTION_PROVENANCE'):
        save_soft(tmp_path / 'bad', fields, contract)
