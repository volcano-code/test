"""Entity projection must not turn accumulated subthreshold material solid."""
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.shadowart_vnext import rendering, training
from experiments.shadowart_vnext.contracts import build_contract


def project(probability, cameras, size=8):
    module = importlib.import_module('experiments.shadowart_vnext.entity_projection')
    return module.render_probability_max(probability, cameras, size, extent=1.7)


def axial(distance=2.):
    return [SimpleNamespace(origin=(0., 0., distance), right=(1., 0., 0.),
                            up=(0., 1., 0.), direction=(0., 0., -1.), scale=1.7)]


def test_subthreshold_material_cannot_accumulate_into_entity_shadow():
    assert importlib.util.find_spec('experiments.shadowart_vnext.entity_projection'), 'missing entity projector'
    p = torch.full((8, 8, 8), .2, dtype=torch.float64, requires_grad=True)
    prediction = project(p, axial())
    torch.testing.assert_close(prediction, torch.full((1, 8, 8), .2, dtype=torch.float64))
    assert not (prediction >= .5).any()
    prediction.sum().backward()
    torch.testing.assert_close(p.grad, torch.full_like(p, .125))


@pytest.mark.parametrize('depth', [0, 3, 7])
@pytest.mark.parametrize('distance', [1.7, 2.5])
def test_single_threshold_sheet_casts_shadow_at_all_depths(depth, distance):
    p = torch.full((8, 8, 8), .1)
    p[:, :, depth] = .5
    torch.testing.assert_close(project(p, axial(distance)), torch.full((1, 8, 8), .5))


@pytest.mark.parametrize('tsf', [0, 1])
def test_threshold_parity_for_oblique_signed_and_cardinal_rays(tsf):
    cams = build_contract(4, tsf=tsf, resolution=8, image_size=8).cameras
    p = torch.rand((8, 8, 8), generator=torch.Generator().manual_seed(17))
    p[p < .8] *= .5
    prediction = project(p, cams, 13).numpy() >= .5
    np.testing.assert_array_equal(prediction, rendering.render_binary(p.numpy() >= .5, cams, 13))


def test_inside_camera_only_integrates_forward_segments():
    p = torch.full((8, 8, 8), .1)
    p[:, :, 7] = .9  # Behind the camera looking in -Z.
    torch.testing.assert_close(project(p, axial(0.)), torch.full((1, 8, 8), .1))


def test_oblique_max_backward_matches_finite_differences_and_bounded_saved_state():
    cams = build_contract(3, tsf=0, resolution=4, image_size=4).cameras[1:2]
    p = (.1 + .8 * torch.rand((4, 4, 4), dtype=torch.float64,
                             generator=torch.Generator().manual_seed(17))).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: project(x, cams, 3), (p,), eps=1e-6)
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda x: (saved.append(x.numel()) or x), lambda x: x):
        project(p, cams, 32).sum().backward()
    assert sum(saved) <= 2 * p.numel(), 'oblique ray graph retains per-segment tensors'


def test_entity_cli_snapshot_records_real_forward_and_rejects_false_provenance(tmp_path):
    from PIL import Image
    from experiments.run_shadowart_vnext import main
    from experiments.shadowart_vnext.artifacts import load_soft, save_soft
    from experiments.shadowart_vnext.contracts import CameraContract
    mask = np.zeros((8, 8), np.uint8); mask[2:6, 2:6] = 255
    source = tmp_path/'target.png'; Image.fromarray(mask).save(source)
    out = tmp_path/'run'
    assert main(['--silhouettes', str(source), '--output', str(out), '--stage', 'soft',
                 '--resolution', '4', '--image-size', '4', '--steps', '4', '--device', 'cpu',
                 '--layout-candidates', '1', '--projection-mode', 'dense_entity_max']) == 0
    c = CameraContract.from_dict(json.loads((out/'CAMERA_CONTRACT.json').read_text()))
    f = load_soft(out/'soft_training', c)
    assert f['metadata']['training_projection']['backward_rule'] == 'ray_max_subgradient'
    assert 'entity_projection.py' in f['metadata']['source_hashes']
    assert (out/'source_snapshot/entity_projection.py').exists()
    report = json.loads((out/'STAGE_METRICS.json').read_text())['entity_projection']
    assert report['threshold_matches_hard_occupancy'] is True
    assert report['threshold_rule'] == 'max_cell_probability_on_positive_ray>=0.5'
    assert report['image_size'] == 8
    assert not (out/'delivery').exists()
    f['metadata']['training_projection']['backward_rule'] = 'optical_autograd'
    with pytest.raises(ValueError, match='PROJECTION_PROVENANCE'):
        save_soft(tmp_path/'tampered', f, c)


def test_max_training_preserves_joint_update_and_initialization():
    targets = np.zeros((2, 4, 4), bool); targets[:, 1:3, 1:3] = True
    dense = np.repeat(np.repeat(targets, 2, 1), 2, 2)
    kwargs = dict(resolution=4, image_size=4, steps=4, seed=42, supersampled_targets=dense)
    control = training.train_soft(targets, axial()*2, projection_mode='dense_soft', **kwargs)
    a = training.train_soft(targets, axial()*2, projection_mode='dense_entity_max', view_chunk_size=1, **kwargs)
    b = training.train_soft(targets, axial()*2, projection_mode='dense_entity_max', view_chunk_size=2, **kwargs)
    assert a['metadata']['initial_logits_sha256'] == control['metadata']['initial_logits_sha256']
    np.testing.assert_allclose(a['guidance'], b['guidance'], atol=1e-6)
    np.testing.assert_array_equal(a['occupancy'], a['guidance'] >= .5)
