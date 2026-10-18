"""Early-loss pairing must preserve the original from-scratch control trajectory."""
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.shadowart_vnext.contracts import build_contract
from experiments.shadowart_vnext.training import train_soft


def api():
    name = 'experiments.shadowart_vnext.rooted_core_early'
    assert importlib.util.find_spec(name) is not None, 'early-training experiment missing'
    return importlib.import_module(name)


def scene(n=1):
    contract = build_contract(n, tsf=1 if n == 3 else None,
                              resolution=6, image_size=8)
    masks = np.zeros((len(contract.cameras), 8, 8), bool)
    masks[:, 2:6, 2:6] = True
    dense = np.repeat(np.repeat(masks, 2, 1), 2, 2)
    return contract, masks, dense


@pytest.fixture(scope='module')
def backend(tmp_path_factory):
    from experiments.shadowart_vnext.rooted_core import build_backend
    return build_backend(tmp_path_factory.mktemp('early') / 'native')


@pytest.mark.parametrize('n', [1, 3])
def test_zero_auxiliary_matches_original_trainer_with_uninterrupted_adam(n):
    contract, masks, dense = scene(n)
    result = api().train_pair(dense, contract.cameras, resolution=6, steps=12,
                              warmup=3, ramp=3, gradient_ratio=0., backend=None)
    reference = train_soft(masks, contract.cameras, resolution=6, image_size=8,
                           steps=12, lr=.007, seed=42, device='cpu',
                           pixel_supersampling=2, supersampled_targets=dense,
                           projection_mode='dense_entity_continuation',
                           initialization_mode='full_minus4')
    for arm in result['arms'].values():
        np.testing.assert_array_equal(arm['probability'], reference['guidance'])
        assert arm['optimizer_step'] == 12
    assert result['common']['initial_logit_value'] == -4.
    assert result['common']['checkpoint_step'] == 3
    assert result['arms']['baseline']['start_state_sha256'] == result['arms']['rooted_core']['start_state_sha256']


def test_auxiliary_begins_after_shared_warmup_and_ramps_once(backend):
    contract, _, dense = scene()
    before = dense.copy()
    result = api().train_pair(dense, contract.cameras, resolution=6, steps=15,
                              warmup=3, ramp=3, gradient_ratio=.1, backend=backend)
    np.testing.assert_array_equal(dense, before)
    assert result['calibration']['next_training_step'] == 4
    assert result['calibration']['weighted_gradient_ratio'] == pytest.approx(.1)
    baseline = result['arms']['baseline']; other = result['arms']['rooted_core']
    assert baseline['history'][:3] == other['history'][:3]
    assert [row['auxiliary_ramp'] for row in other['history'][:6]] == [0., 0., 0., 1/3, 2/3, 1.]
    assert all(row['aux_loss'] is None for row in baseline['history'])
    assert not np.array_equal(baseline['probability'], other['probability'])
    assert baseline['start_state_sha256'] == other['start_state_sha256']
    assert baseline['optimizer_step'] == other['optimizer_step'] == 15
    assert all(np.isfinite(row['gradient_norm']) for row in other['history'])


@pytest.mark.parametrize('settings', [dict(steps=0), dict(warmup=0), dict(warmup=12),
                                      dict(ramp=0), dict(ramp=20), dict(resolution=True),
                                      dict(gradient_ratio=-.1), dict(gradient_ratio=float('nan')),
                                      dict(lr=0), dict(width=0)])
def test_invalid_schedule_or_parameters_fail(settings):
    contract, _, dense = scene()
    args = dict(resolution=6, steps=12, warmup=3, ramp=3, gradient_ratio=0., backend=None)
    args.update(settings)
    with pytest.raises((ValueError, TypeError)):
        api().train_pair(dense, contract.cameras, **args)


def test_no_target_hull_core_does_not_fallback_to_old_trained_shape():
    contract, _, dense = scene()
    with pytest.raises(ValueError, match='anchor'):
        api().train_pair(np.zeros_like(dense), contract.cameras, resolution=6,
                         steps=12, warmup=3, ramp=3, gradient_ratio=0., backend=None)


def test_protocol_loader_never_requires_density_or_guidance(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = repo / 'results/shadowart_vnext_heroes_native128_cont_fullminus4_s42_20260908/soft_training/CAMERA_CONTRACT.json'
    if not source.exists():
        pytest.skip('local frozen protocol fixture unavailable')
    copied = tmp_path / 'CAMERA_CONTRACT.json'
    copied.write_bytes(source.read_bytes())
    images = [repo / 'generation' / n for n in ('Spider-Man.png', 'superman.png', 'batman.png')]
    contract, targets = api().load_protocol(copied, images)
    assert contract.volume_resolution == 128
    assert len(targets.masks) == 3
    assert list(tmp_path.iterdir()) == [copied]
    with pytest.raises(ValueError, match='TARGET'):
        api().load_protocol(copied, images[::-1])


def test_wrong_camera_protocol_rejected(tmp_path):
    contract, _, _ = scene(3)
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps(contract.to_dict()))
    with pytest.raises(ValueError, match='CAMERA'):
        api().load_protocol(path, [])


def test_runner_will_not_write_inside_protocol_source_tree(tmp_path):
    name = 'experiments.shadowart_vnext_dev.run_rooted_core_early_20260912'
    assert importlib.util.find_spec(name) is not None, 'early runner missing'
    runner = importlib.import_module(name)
    source = tmp_path / 'protocol'
    source.mkdir()
    args = ['--camera-contract', str(source / 'CAMERA_CONTRACT.json'),
            '--silhouettes', 'a.png', '--output', str(source / 'new')]
    with pytest.raises(SystemExit):
        runner.parse_args(args)
    assert not (source / 'new').exists()


def test_diagnostic_failure_happens_only_after_both_training_results_are_saved(tmp_path):
    runner = importlib.import_module('experiments.shadowart_vnext_dev.run_rooted_core_early_20260912')
    assert hasattr(runner, 'train_then_diagnose'), 'diagnostics still interrupt training'
    contract, _, dense = scene()
    def save(name, probability, metadata):
        np.save(tmp_path / (name + '.npy'), probability, allow_pickle=False)
        (tmp_path / (name + '.json')).write_text(json.dumps(metadata))
    def train():
        return api().train_pair(dense, contract.cameras, resolution=6, steps=12,
                                warmup=3, ramp=3, gradient_ratio=0., backend=None,
                                arm_callback=save)
    def diagnose(name, probability):
        assert name == 'baseline'
        raise RuntimeError('deliberate diagnostic failure')
    with pytest.raises(RuntimeError, match='deliberate diagnostic'):
        runner.train_then_diagnose(train, diagnose)
    for arm in ('baseline', 'rooted_core'):
        assert np.load(tmp_path / (arm + '.npy')).shape == (6, 6, 6)
        assert len(json.loads((tmp_path / (arm + '.json')).read_text())['history']) == 12
