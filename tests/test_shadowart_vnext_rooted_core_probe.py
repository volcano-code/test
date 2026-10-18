"""Mechanism-pilot contracts; no mesh-delivery claims from soft metrics."""
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.shadowart_vnext.contracts import build_contract, TargetSet
from experiments.shadowart_vnext.entity_projection import render_probability_max
from experiments.shadowart_vnext.losses import compute_loss


@pytest.fixture
def probe():
    name = 'experiments.shadowart_vnext.rooted_core_probe'
    assert importlib.util.find_spec(name) is not None, 'rooted-core pilot is not implemented'
    return importlib.import_module(name)


@pytest.fixture(scope='module')
def backend(tmp_path_factory):
    from experiments.shadowart_vnext.rooted_core import build_backend
    return build_backend(tmp_path_factory.mktemp('rooted_probe') / 'native')


def scene():
    p = np.full((8, 8, 8), .12, np.float32)
    p[2:6, 2:6, 2:6] = .87
    contract = build_contract(1, resolution=8, image_size=16)
    targets = np.zeros((1, 16, 16), np.float32)
    targets[:, 5:11, 5:11] = 1
    return p, contract, targets


def test_zero_coefficient_matches_existing_loss_optimizer_exactly(probe):
    p, contract, targets = scene()
    original = p.copy()
    z = torch.logit(torch.from_numpy(p.copy())).requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=.007)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        pred = render_probability_max(z.sigmoid(), contract.cameras, 16, contract.extent)
        loss, _ = compute_loss(pred, torch.from_numpy(targets))
        loss.backward()
        optimizer.step()
    result = probe.run_arm(p, targets, contract.cameras, width=2, root=(2, 2, 2),
                           backend=None, coefficient=0., steps=3, lr=.007)
    np.testing.assert_array_equal(result['probability'], z.detach().sigmoid().numpy())
    np.testing.assert_array_equal(p, original)
    assert len(result['history']) == 3
    assert [row['step'] for row in result['history']] == [1, 2, 3]
    assert result['checkpoint_selection'] == 'final_step'


def test_unmeasured_baseline_auxiliary_is_not_reported_as_zero_error(probe):
    p, contract, targets = scene()
    result = probe.run_arm(p, targets, contract.cameras, width=2, root=(2, 2, 2),
                           backend=None, coefficient=0., steps=1)
    assert result['history'][0]['aux_loss'] is None
    assert result['history'][0]['weighted_aux_loss'] == 0.


@pytest.mark.parametrize('options', [dict(steps=0), dict(steps=True), dict(lr=0),
                                    dict(coefficient=-.1), dict(coefficient=float('nan')),
                                    dict(width=0)])
def test_invalid_optimization_arguments_rejected(probe, options):
    p, contract, targets = scene()
    args = dict(width=2, root=(2, 2, 2), backend=None, coefficient=0., steps=1, lr=.007)
    args.update(options)
    with pytest.raises(ValueError):
        probe.run_arm(p, targets, contract.cameras, **args)


def test_endpoint_probabilities_are_not_silently_clipped(probe):
    p, contract, targets = scene()
    p[0, 0, 0] = 0
    with pytest.raises(ValueError, match='interior'):
        probe.run_arm(p, targets, contract.cameras, width=2, root=(2, 2, 2),
                      backend=None, coefficient=0., steps=1, lr=.007)


def test_calibrated_ratio_and_real_auxiliary_update(probe, backend):
    from experiments.shadowart_vnext.rooted_core import choose_root
    p, contract, targets = scene()
    root = choose_root(p, 2)
    calibration = probe.calibrate(p, targets, contract.cameras, width=2, root=root,
                                  backend=backend, gradient_ratio=.1)
    assert calibration['coefficient'] > 0
    assert calibration['weighted_gradient_ratio'] == pytest.approx(.1)
    # Independent formula: check actual gradients, not just self-reported ratio.
    from experiments.shadowart_vnext.rooted_core import rooted_core
    logits = torch.logit(torch.from_numpy(p.copy())).requires_grad_(True)
    truth = torch.from_numpy(targets)
    prediction = render_probability_max(logits.sigmoid(), contract.cameras, 16, contract.extent)
    base = ((prediction - truth).square() + .35 * (prediction - truth).abs()).mean()
    base_gradient, = torch.autograd.grad(base, logits)
    kp = render_probability_max(rooted_core(logits.sigmoid(), 2, root, backend),
                                 contract.cameras, 16, contract.extent)
    pixel = torch.nn.functional.binary_cross_entropy(kp, truth, reduction='none')
    auxiliary_loss = .5 * ((pixel * truth).sum() / truth.sum() +
                           (pixel * (1 - truth)).sum() / (1 - truth).sum())
    auxiliary_gradient, = torch.autograd.grad(auxiliary_loss, logits)
    actual_ratio = float((calibration['coefficient'] * auxiliary_gradient).norm() /
                         base_gradient.norm())
    assert actual_ratio == pytest.approx(.1, rel=1e-5)
    assert -1.00001 <= calibration['gradient_cosine'] <= 1.00001
    common = dict(width=2, root=root, backend=backend, steps=3, lr=.007)
    baseline = probe.run_arm(p, targets, contract.cameras, coefficient=0., **common)
    auxiliary = probe.run_arm(p, targets, contract.cameras,
                              coefficient=calibration['coefficient'], **common)
    assert baseline['initial_logits_sha256'] == auxiliary['initial_logits_sha256']
    assert not np.array_equal(baseline['probability'], auxiliary['probability'])
    assert all(np.isfinite(row['gradient_norm']) for row in auxiliary['history'])
    np.testing.assert_array_equal(p, scene()[0])


def test_scalar_diagnostics_do_not_convert_live_autograd_tensors(probe, backend):
    import warnings
    from experiments.shadowart_vnext.rooted_core import choose_root
    p, contract, targets = scene()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        probe.calibrate(p, targets, contract.cameras, width=2, root=choose_root(p, 2),
                        backend=backend)


def test_measurement_keeps_core_and_original_separate_and_checks_hard_parity(probe, backend):
    from experiments.shadowart_vnext.rooted_core import choose_root
    p, contract, _ = scene()
    target_set = TargetSet((np.ones((3, 3), bool),), ('synthetic',), ('synthetic',))
    report, core = probe.measure(p, contract, target_set, width=2,
                                root=choose_root(p, 2), backend=backend, sizes=(16, 32))
    assert report['mesh_qa_status'] == 'NOT_RUN'
    assert report['digital_manufacturing_passed'] is False
    assert len(report['evaluations']) == 2
    assert all(row['original_threshold_hard_parity'] and row['core_threshold_hard_parity']
               for row in report['evaluations'])
    assert core.shape == p.shape
    assert np.all(core <= p)


def test_runner_rejects_source_overlap_before_writing(tmp_path):
    name = 'experiments.shadowart_vnext_dev.run_rooted_core_probe_20260912'
    assert importlib.util.find_spec(name) is not None, 'rooted-core runner is not implemented'
    runner = importlib.import_module(name)
    source = tmp_path / 'source'
    source.mkdir()
    with pytest.raises(SystemExit):
        runner.parse_args(['--soft-artifact-dir', str(source), '--silhouettes', 'a.png',
                           '--output', str(source / 'out')])
    assert list(source.iterdir()) == []


def test_followup_budget_is_bounded_and_20_step_default_retained(tmp_path):
    runner = importlib.import_module('experiments.shadowart_vnext_dev.run_rooted_core_probe_20260912')
    args = ['--soft-artifact-dir', str(tmp_path / 'source'), '--silhouettes', 'a.png',
            '--output', str(tmp_path / 'output')]
    assert runner.parse_args(args).steps == 20
    assert runner.parse_args(args + ['--steps', '200']).steps == 200
    with pytest.raises(SystemExit):
        runner.parse_args(args + ['--steps', '201'])


def test_wrong_camera_cannot_be_relabelled_as_frozen_three_heroes(probe, tmp_path):
    contract = build_contract(3, tsf=1, resolution=128)
    (tmp_path / 'CAMERA_CONTRACT.json').write_text(json.dumps(contract.to_dict()))
    with pytest.raises(ValueError, match='FROZEN_CAMERA'):
        probe.load_frozen(tmp_path, [])


@pytest.fixture
def real_source():
    root = Path(__file__).resolve().parents[1]
    source = root / 'results/shadowart_vnext_heroes_native128_cont_fullminus4_s42_20260908/soft_training'
    if not (source / 'CAMERA_CONTRACT.json').is_file():
        pytest.skip('local frozen three-hero artifact is not distributed with unit tests')
    return root, source


def test_wrong_frozen_field_rejected_before_target_loading(probe, tmp_path, real_source):
    _, source = real_source
    (tmp_path / 'CAMERA_CONTRACT.json').write_bytes((source / 'CAMERA_CONTRACT.json').read_bytes())
    np.save(tmp_path / 'guidance.npy', np.zeros((2, 2, 2), np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match='FAILED_FROZEN_FIELD'):
        probe.load_frozen(tmp_path, [])


def test_reordered_hero_inputs_rejected(probe, real_source):
    root, source = real_source
    silhouettes = [root / 'generation' / name for name in
                   ('superman.png', 'Spider-Man.png', 'batman.png')]
    with pytest.raises(ValueError, match='FAILED_TARGET_HASH'):
        probe.load_frozen(source, silhouettes)
