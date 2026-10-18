"""Full-ray probability continuation stays below max and ends at exact max."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.shadowart_vnext import rendering
from experiments.shadowart_vnext.contracts import build_contract


def axial(distance=2.0, scale=1.7):
    return [SimpleNamespace(origin=(0.0, 0.0, distance), right=(1.0, 0.0, 0.0),
                            up=(0.0, 1.0, 0.0), direction=(0.0, 0.0, -1.0), scale=scale)]


def continuation_module():
    return __import__('experiments.shadowart_vnext.continuation_projection', fromlist=['*'])


def project(probability, cameras, size=1, exponent=2.0):
    return continuation_module().render_probability_generalized_mean(
        probability, cameras, size, exponent=exponent, extent=1.7)


def test_two_cell_equal_length_power_means_are_hand_computed():
    p = torch.empty((2, 2, 2), dtype=torch.float64)
    p[:, :, 0] = 0.25
    p[:, :, 1] = 0.75
    torch.testing.assert_close(project(p, axial(), exponent=1.0), torch.tensor([[[0.5]]], dtype=p.dtype))
    torch.testing.assert_close(project(p, axial(), exponent=2.0),
                               torch.tensor([[[np.sqrt(0.3125)]]], dtype=p.dtype))


def test_q1_has_exact_linear_gradient_including_zero_cells():
    p = torch.zeros((2, 2, 2), dtype=torch.float64, requires_grad=True)
    prediction = project(p, axial(), exponent=1.0)
    prediction.sum().backward()
    expected = torch.zeros_like(p)
    expected[1, 1, :] = 0.5
    torch.testing.assert_close(p.grad, expected, rtol=0, atol=0)


@pytest.mark.parametrize('exponent', [1.0, 3.5, 32.0])
def test_constant_probabilities_remain_constant_on_nonempty_rays(exponent):
    p = torch.full((4, 4, 4), 0.37, dtype=torch.float64)
    torch.testing.assert_close(project(p, axial(), size=4, exponent=exponent),
                               torch.full((1, 4, 4), 0.37, dtype=p.dtype))


def test_all_zero_q_greater_than_one_has_finite_zero_value_and_subgradient():
    p = torch.zeros((4, 4, 4), dtype=torch.float64, requires_grad=True)
    result = project(p, axial(), size=4, exponent=7.0)
    result.sum().backward()
    assert torch.equal(result, torch.zeros_like(result))
    assert torch.equal(p.grad, torch.zeros_like(p))
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize('exponent', [2.0, 5.5])
def test_axis_gradient_matches_finite_differences(exponent):
    p = (0.1 + 0.8 * torch.rand((3, 3, 3), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(7))).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: project(x, axial(), size=3, exponent=exponent),
                                    (p,), eps=1e-6, atol=2e-5, rtol=2e-4)


def test_oblique_weighted_value_and_gradient_use_complete_segment_lengths():
    camera = build_contract(3, tsf=0, resolution=3, image_size=3).cameras[1]
    p = (0.1 + 0.8 * torch.rand((3, 3, 3), dtype=torch.float64,
                               generator=torch.Generator().manual_seed(11))).requires_grad_()
    chunk = next(rendering.iter_ray_segments(camera, 1, 3, 1.7))
    lengths = chunk.lengths
    values = p.detach().numpy().reshape(-1)[chunk.voxel_indices]
    expected = (np.sum(lengths * values**2.5) / np.sum(lengths)) ** (1.0 / 2.5)
    torch.testing.assert_close(project(p, [camera], exponent=2.5),
                               torch.tensor([[[expected]]], dtype=p.dtype))
    assert torch.autograd.gradcheck(lambda x: project(x, [camera], exponent=2.5),
                                    (p,), eps=1e-6, atol=2e-5, rtol=2e-4)


def test_oblique_q1_zero_field_keeps_exact_segment_length_gradient():
    camera = build_contract(3, tsf=0, resolution=3, image_size=3).cameras[1]
    p = torch.zeros((3, 3, 3), dtype=torch.float64, requires_grad=True)
    chunk = next(rendering.iter_ray_segments(camera, 1, 3, 1.7))
    expected = torch.zeros_like(p).reshape(-1)
    expected.index_add_(0, torch.from_numpy(chunk.voxel_indices.copy()),
                        torch.from_numpy((chunk.lengths / chunk.lengths.sum()).copy()))
    project(p, [camera], exponent=1.0).sum().backward()
    torch.testing.assert_close(p.grad, expected.reshape_as(p), rtol=0, atol=1e-15)


def test_generalized_mean_is_bounded_by_complete_ray_max_without_ghost_floor():
    from experiments.shadowart_vnext.entity_projection import render_probability_max
    cameras = build_contract(4, tsf=1, resolution=4, image_size=4).cameras
    p = 0.49 * torch.rand((4, 4, 4), dtype=torch.float64,
                          generator=torch.Generator().manual_seed(19))
    mean = project(p, cameras, size=7, exponent=6.0)
    maximum = render_probability_max(p, cameras, 7, extent=1.7)
    assert torch.all(mean <= maximum + 1e-14)
    assert not (mean >= 0.5).any()


@pytest.mark.parametrize('exponent', [1.0, 1.0001, 31.999, 32.0])
def test_float32_oblique_constant_one_never_rounds_above_probability_range(exponent):
    cameras = build_contract(6, tsf=0, resolution=3, image_size=3).cameras
    prediction = project(torch.ones((3, 3, 3), dtype=torch.float32), cameras,
                         size=17, exponent=exponent)
    assert torch.all(prediction <= 1.0)
    assert torch.all(prediction >= 0.0)
    nonempty = rendering.render_binary(np.ones((3, 3, 3), bool), cameras, 17)
    torch.testing.assert_close(prediction[torch.from_numpy(nonempty)],
                               torch.ones(int(nonempty.sum()), dtype=torch.float32), rtol=0, atol=0)


@pytest.mark.parametrize('exponent', [1.0, 1.0001, 32.0])
def test_float32_oblique_all_zero_has_no_nonzero_stabilization_floor(exponent):
    cameras = build_contract(3, tsf=0, resolution=3, image_size=3).cameras[1:2]
    prediction = project(torch.zeros((3, 3, 3), dtype=torch.float32), cameras,
                         size=17, exponent=exponent)
    assert torch.equal(prediction, torch.zeros_like(prediction))


@pytest.mark.parametrize('exponent', [1.0, 1.0001, 32.0])
def test_float32_roundoff_never_exceeds_each_complete_ray_max(exponent):
    from experiments.shadowart_vnext.entity_projection import render_probability_max
    cameras = build_contract(6, tsf=0, resolution=3, image_size=3).cameras
    p = torch.rand((3, 3, 3), dtype=torch.float32,
                   generator=torch.Generator().manual_seed(0))
    prediction = project(p, cameras, size=31, exponent=exponent)
    maximum = render_probability_max(p, cameras, 31, extent=1.7)
    assert torch.all(prediction <= maximum)


def test_empty_rays_are_zero_and_each_depth_contributes_without_truncation():
    p = torch.ones((3, 3, 3), dtype=torch.float64)
    expected_nonempty = torch.from_numpy(rendering.render_binary(
        np.ones((3, 3, 3), bool), axial(scale=5.1), 5, extent=1.7)).to(p.dtype)
    torch.testing.assert_close(project(p, axial(scale=5.1), size=5, exponent=4.0), expected_nonempty)
    values = []
    for depth in (0, 1, 2):
        sheet = torch.zeros_like(p)
        sheet[:, :, depth] = 0.9
        values.append(project(sheet, axial(), exponent=2.0))
    torch.testing.assert_close(values[0], values[1])
    torch.testing.assert_close(values[1], values[2])
    torch.testing.assert_close(values[0], torch.tensor([[[0.9 / np.sqrt(3.0)]]], dtype=p.dtype))


def test_inside_camera_uses_only_forward_positive_length_segments():
    p = torch.full((4, 4, 4), 0.1, dtype=torch.float64)
    p[:, :, 2:] = 0.9  # Behind origin z=0 while the camera looks toward -Z.
    torch.testing.assert_close(project(p, axial(distance=0.0), size=4, exponent=3.0),
                               torch.full((1, 4, 4), 0.1, dtype=p.dtype))


def test_oblique_autograd_retains_only_bounded_native_state():
    camera = build_contract(3, tsf=0, resolution=4, image_size=4).cameras[1]
    p = torch.rand((4, 4, 4), dtype=torch.float64, requires_grad=True)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: (saved.append(x.numel()) or x), lambda x: x):
        project(p, [camera], size=32, exponent=4.0).sum().backward()
    assert sum(saved) <= 2 * p.numel(), 'streamed oblique graph retained per-ray segments'


@pytest.mark.parametrize('value', [0.999, 32.001, float('inf'), float('nan'), torch.tensor(2.0)])
def test_exponent_validation_rejects_non_scalar_finite_range_values(value):
    with pytest.raises(ValueError, match='exponent'):
        project(torch.ones((2, 2, 2)), axial(), exponent=value)


@pytest.mark.parametrize('value', [np.ones((2, 2, 2)), torch.ones((2, 2)), torch.ones((2, 2, 3)),
                                   torch.ones((2, 2, 2), dtype=torch.int64)])
def test_probability_validation_requires_a_floating_cube(value):
    with pytest.raises(ValueError, match='probability'):
        project(value, axial())


def test_schedule_starts_linear_and_ends_in_separate_exact_max_phase():
    state = continuation_module().continuation_state
    assert state(1, 11) == {'progress': 0.0, 'phase': 'generalized_mean', 'exponent': 1.0}
    assert state(5, 11)['exponent'] == pytest.approx(2.0**2.5)
    assert state(9, 11) == {'progress': 0.8, 'phase': 'exact_max', 'exponent': None}
    assert state(11, 11) == {'progress': 1.0, 'phase': 'exact_max', 'exponent': None}
    assert state(1, 1) == {'progress': 0.0, 'phase': 'exact_max', 'exponent': None}


def test_single_step_forward_is_exact_max_not_a_finite_exponent_alias():
    from experiments.shadowart_vnext.entity_projection import render_probability_max
    cameras = build_contract(3, tsf=1, resolution=3, image_size=3).cameras
    p = torch.rand((3, 3, 3), dtype=torch.float64,
                   generator=torch.Generator().manual_seed(29))
    actual = continuation_module().render_probability_continuation(
        p, cameras, 7, step=1, steps=1, extent=1.7)
    torch.testing.assert_close(actual, render_probability_max(p, cameras, 7, extent=1.7))


@pytest.mark.parametrize('step,steps', [(0, 2), (3, 2), (1, 0), (1.0, 2), (1, True)])
def test_schedule_validates_integer_step_range(step, steps):
    with pytest.raises(ValueError, match='step'):
        continuation_module().continuation_state(step, steps)


def tiny_training_inputs(view_count=1):
    targets = np.zeros((view_count, 2, 2), bool)
    targets[:, 0, 0] = True
    dense = np.repeat(np.repeat(targets, 2, axis=1), 2, axis=2)
    return targets, dense


def test_default_old_mode_trajectory_is_unchanged_by_explicit_default():
    from experiments.shadowart_vnext import training
    targets, _ = tiny_training_inputs()
    kwargs = dict(resolution=4, image_size=2, steps=2, seed=23)
    implicit = training.train_soft(targets, axial(), **kwargs)
    explicit = training.train_soft(targets, axial(), projection_mode='area_soft', **kwargs)
    for key in ('occupancy', 'density', 'probabilities', 'guidance', 'soft_projections',
                'hard_projections', 'hard_projections_512'):
        np.testing.assert_array_equal(implicit[key], explicit[key])
    assert implicit['history'] == explicit['history']
    assert implicit['metadata'] == explicit['metadata']


def test_continuation_training_preserves_initialization_and_joint_chunk_result():
    from experiments.shadowart_vnext import training
    targets, dense = tiny_training_inputs(2)
    kwargs = dict(resolution=4, image_size=2, steps=4, seed=17,
                  supersampled_targets=dense, initialization_mode='full_minus4')
    control = training.train_soft(targets, axial() * 2, projection_mode='dense_soft', **kwargs)
    chunked = training.train_soft(targets, axial() * 2, projection_mode='dense_entity_continuation',
                                  view_chunk_size=1, **kwargs)
    joined = training.train_soft(targets, axial() * 2, projection_mode='dense_entity_continuation',
                                 view_chunk_size=2, **kwargs)
    assert chunked['metadata']['initial_logits_sha256'] == control['metadata']['initial_logits_sha256']
    np.testing.assert_allclose(chunked['guidance'], joined['guidance'], atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(chunked['occupancy'], chunked['guidance'] >= 0.5)


@pytest.fixture(scope='module')
def continuation_fields():
    from dataclasses import replace
    from experiments.shadowart_vnext import training
    from experiments.shadowart_vnext.contracts import file_hash, mask_hash
    targets, dense = tiny_training_inputs()
    contract = build_contract(1, image_size=2, resolution=4)
    contract = replace(contract, input_hashes=(file_hash(__file__),),
                       target_hashes=(mask_hash(targets[0]),))
    fields = training.train_soft(targets, contract.cameras, resolution=4, image_size=2,
                                 steps=51, seed=31, supersampled_targets=dense,
                                 projection_mode='dense_entity_continuation',
                                 initialization_mode='full_minus4')
    return contract, targets, fields


def test_gap_checkpoints_are_post_update_at_exact_cadence_and_finish_with_parity(continuation_fields):
    contract, _, fields = continuation_fields
    metadata = fields['metadata']
    assert metadata['training_projection']['continuation_schedule'] == continuation_module().continuation_schedule()
    assert metadata['training_projection']['gap_checkpoint_cadence'] == {
        'rule': 'step_1_every_50_and_final', 'steps': [1, 50, 51],
        'phase': 'post_optimizer_update', 'image_size': 4,
    }
    checkpoints = [(row['step'], row.get('continuation_gap_checkpoint')) for row in fields['history']]
    assert [step for step, checkpoint in checkpoints if checkpoint is not None] == [1, 50, 51]
    for row in fields['history']:
        state = continuation_module().continuation_state(row['step'], 51)
        assert row['loss_metrics_phase'] == 'pre_optimizer_update'
        assert row['projection_phase'] == state['phase']
        assert row['projection_exponent'] == state['exponent']
    first = fields['history'][0]['continuation_gap_checkpoint']
    assert first['phase'] == 'post_optimizer_update'
    assert first['scope'] == 'training_projection_gap_only_not_mesh_or_manufacturing'
    assert first['image_size'] == 4
    assert first['projection_exponent'] == 1.0
    final = fields['history'][-1]['continuation_gap_checkpoint']
    assert final['projection_phase'] == 'exact_max'
    assert final['projection_exponent'] is None
    assert final['per_view_disagreement_pixels'] == [0]
    assert final['threshold_surrogate_iou'] == final['hard_iou']
    assert metadata['checkpoint_selection'] == 'final_step'
    assert metadata['export_step'] == 51


def test_endpoint_projection_threshold_exactly_matches_fixed_binary_export(continuation_fields):
    contract, _, fields = continuation_fields
    probability = torch.from_numpy(fields['guidance'])
    prediction = continuation_module().render_probability_continuation(
        probability, contract.cameras, 7, step=51, steps=51, extent=contract.extent)
    expected = rendering.render_binary(fields['occupancy'], contract.cameras, 7, extent=contract.extent)
    np.testing.assert_array_equal(prediction.numpy() >= 0.5, expected)


@pytest.mark.parametrize('mutation', [
    'schedule', 'cadence', 'history_exponent', 'loss_phase', 'missing_checkpoint',
    'checkpoint_phase', 'checkpoint_scope', 'checkpoint_nonfinite',
    'endpoint_disagreement', 'continuation_source', 'entity_source',
])
def test_artifact_rejects_forged_continuation_schedule_history_and_sources(
        tmp_path, continuation_fields, mutation):
    import copy
    from experiments.shadowart_vnext.artifacts import save_soft
    contract, _, original = continuation_fields
    fields = copy.deepcopy(original)
    if mutation == 'schedule':
        fields['metadata']['training_projection']['continuation_schedule']['exponent_bounds'][1] = 31.0
    elif mutation == 'cadence':
        fields['metadata']['training_projection']['gap_checkpoint_cadence']['steps'] = [1, 51]
    elif mutation == 'history_exponent':
        fields['history'][0]['projection_exponent'] = 2.0
    elif mutation == 'loss_phase':
        fields['history'][0]['loss_metrics_phase'] = 'post_optimizer_update'
    elif mutation == 'missing_checkpoint':
        fields['history'][49].pop('continuation_gap_checkpoint')
    elif mutation == 'checkpoint_phase':
        fields['history'][0]['continuation_gap_checkpoint']['phase'] = 'pre_optimizer_update'
    elif mutation == 'checkpoint_scope':
        fields['history'][0]['continuation_gap_checkpoint']['scope'] = 'mesh_quality'
    elif mutation == 'checkpoint_nonfinite':
        fields['history'][0]['continuation_gap_checkpoint']['hard_iou']['mean_iou'] = float('nan')
    elif mutation == 'endpoint_disagreement':
        fields['history'][-1]['continuation_gap_checkpoint']['per_view_disagreement_pixels'] = [1]
    elif mutation == 'continuation_source':
        fields['metadata']['source_hashes'].pop('continuation_projection.py')
    else:
        fields['metadata']['source_hashes'].pop('entity_projection.py')
    with pytest.raises(ValueError, match='PROJECTION_PROVENANCE|CONTINUATION'):
        save_soft(tmp_path / mutation, fields, contract)


def test_tiny_cli_train_save_load_records_actual_final_max_stage(tmp_path):
    import json
    from PIL import Image
    from experiments.run_shadowart_vnext import main
    from experiments.shadowart_vnext.artifacts import load_soft
    from experiments.shadowart_vnext.contracts import CameraContract
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 2:6] = 255
    source = tmp_path / 'target.png'
    Image.fromarray(mask).save(source)
    out = tmp_path / 'run'
    assert main(['--silhouettes', str(source), '--output', str(out), '--stage', 'soft',
                 '--resolution', '4', '--image-size', '4', '--steps', '4', '--device', 'cpu',
                 '--layout-candidates', '1', '--initialization-mode', 'full_minus4',
                 '--projection-mode', 'dense_entity_continuation']) == 0
    contract = CameraContract.from_dict(json.loads((out / 'CAMERA_CONTRACT.json').read_text()))
    fields = load_soft(out / 'soft_training', contract)
    projection = fields['metadata']['training_projection']
    assert projection['mode'] == 'dense_entity_continuation'
    assert projection['backward_rule'] == 'length_weighted_probability_generalized_mean_then_exact_ray_max'
    assert set(fields['metadata']['source_hashes']) >= {
        'training.py', 'losses.py', 'rendering.py', 'entity_projection.py', 'continuation_projection.py'}
    assert (out / 'source_snapshot' / 'continuation_projection.py').exists()
    stage = json.loads((out / 'STAGE_METRICS.json').read_text())['entity_projection']
    assert stage['stage'] == 'entity_projection_actual_final_max'
    assert stage['actual_final_training_forward'] is True
    assert stage['threshold_matches_hard_occupancy'] is True
    assert not (out / 'delivery').exists()
