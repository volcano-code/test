"""Real thin-neck regressions for opt-in joint island/body/path thickening."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from experiments.shadowart_vnext.regional_candidates import propose
from experiments.shadowart_vnext.regional_geometry import repair_regional, thick_join_preserved
from experiments.shadowart_vnext.manufacturing import _block_cover
from experiments.shadowart_vnext.rendering import render_binary
from test_shadowart_vnext_regional_geometry import front


def two_lobes():
    root = np.zeros((32,) * 3, bool)
    root[4:20, 4:24, 2:8] = True
    part = np.zeros_like(root)
    part[8:12, 8:12, 18:22] = True
    part[8:12, 18:22, 18:22] = True
    part[9, 12:18, 19] = True
    return root | part, part


def proposals(v, part, **kwargs):
    return propose(v, v.astype(float), np.argwhere(part), render_binary(v, [front()], 64),
                   [front()], np.zeros_like(v), 2, 1.7, 24, **kwargs)


def test_path_only_reproduces_detached_neck_core_with_actual_added_mask():
    v, part = two_lobes()
    base, _, primitives, _, descriptions = proposals(v, part)
    paths = [i for i, d in enumerate(descriptions) if d['kind'] == 'retain_and_complete_thick_path']
    assert paths
    for i in paths:
        candidate = base.copy(); candidate.ravel()[primitives[i]] = True
        assert ndi.label(candidate)[1] == 1
        assert not thick_join_preserved(candidate, base, candidate & ~v, 2)


def test_joint_body_path_repairs_both_lobes_without_changing_shadow_or_source():
    v, part = two_lobes(); before = v.copy()
    base, _, primitives, projections, descriptions = proposals(v, part, body_reinforcement=True)
    bodies = [i for i, d in enumerate(descriptions) if d['kind'] == 'reinforced_body_and_path']
    assert bodies
    for i in bodies:
        candidate = base.copy(); candidate.ravel()[primitives[i]] = True
        assert candidate[v].all(), 'reinforcement must retain the entire original body'
        assert ndi.label(candidate)[1] == 1
        assert thick_join_preserved(candidate, base, candidate & ~v, 2)
        assert not np.any(part & ~_block_cover(candidate, 2)[1])
        np.testing.assert_array_equal(render_binary(candidate, [front()], 64),
                                      render_binary(v, [front()], 64))
    np.testing.assert_array_equal(v, before)
    assert len(primitives) <= 24


def test_body_reinforcement_default_is_byte_identical_to_explicit_false():
    v, part = two_lobes()
    default, explicit = proposals(v, part), proposals(v, part, body_reinforcement=False)
    for a, b in zip(default[:2], explicit[:2]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(default[2], explicit[2]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(default[3], explicit[3])
    assert default[4] == explicit[4]


def test_targeted_body_candidate_does_not_expand_already_thick_lobe_surfaces():
    v, part = two_lobes()
    base, _, primitives, _, descriptions = proposals(v, part, body_reinforcement=True)
    selective = [(p, d) for p, d in zip(primitives, descriptions)
                 if d.get('body_scope') == 'uncovered_voxels']
    assert selective, 'provide partial-body repairs before uniformly expanded alternatives'
    for indices, description in selective:
        candidate = base.copy(); candidate.ravel()[indices] = True
        assert thick_join_preserved(candidate, base, candidate & ~v, 2)
        assert not candidate[12, 18:22, 18:22].any(), 'outer distal lobe face needs no material'
        assert not candidate[7, 18:22, 18:22].any()
        assert not np.any(part & ~_block_cover(candidate, 2)[1])


def test_body_pool_reserves_capacity_for_projection_compensation():
    from test_shadowart_vnext_regional_geometry import fixture
    v, target = fixture()
    part = np.argwhere(v & (np.indices(v.shape)[2] >= 12))
    _, _, primitives, _, descriptions = propose(v, v.astype(float), part, target, [front()],
        np.zeros_like(v), 2, 1.7, 4, body_reinforcement=True)
    assert len(primitives) <= 4
    assert sum(d['kind'] == 'reinforced_body_and_path' for d in descriptions) <= 2
    assert any(d['kind'] == 'connected_replacement_box' for d in descriptions)


def test_body_compensation_can_recover_preexisting_missing_foreground():
    v, part = two_lobes()
    desired = v.copy(); desired[20:22, 8:12, 2:6] = True
    target = render_binary(desired, [front()], 64)
    missing = target & ~render_binary(v, [front()], 64)
    assert missing.any()
    _, _, _, projections, descriptions = propose(v, v.astype(float), np.argwhere(part),
        target, [front()], np.zeros_like(v), 2, 1.7, 24, body_reinforcement=True)
    assert any(np.any(projection & missing) for projection, description in zip(projections, descriptions)
               if description['kind'] == 'connected_replacement_box')


def test_reinforcement_rejects_protected_voxels_instead_of_clipping_body():
    v, part = two_lobes()
    protected = ~v
    _, _, _, _, descriptions = propose(v, v.astype(float), np.argwhere(part),
        render_binary(v, [front()], 64), [front()], protected, 2, 1.7, 24,
        body_reinforcement=True)
    assert not any(d['kind'] == 'reinforced_body_and_path' for d in descriptions)


@pytest.mark.parametrize('value', [1, 'yes', None])
def test_body_option_requires_boolean(value):
    v, part = two_lobes()
    with pytest.raises(ValueError, match='body_reinforcement'):
        proposals(v, part, body_reinforcement=value)
    with pytest.raises(ValueError, match='body_reinforcement'):
        repair_regional(v, v.astype(float), render_binary(v, [front()], 64), [front()],
                        max_regions=0, body_reinforcement=value)


def test_body_cli_opt_in_and_config_binding(tmp_path):
    import json
    from PIL import Image
    from experiments.run_shadowart_vnext import main, parse_args
    common = ['--silhouettes', 'x.png', '--output', str(tmp_path)]
    assert not parse_args(common).regional_body_reinforcement
    with pytest.raises(SystemExit):
        parse_args(common + ['--regional-body-reinforcement'])
    mask = np.zeros((8, 8), np.uint8); mask[2:6, 2:6] = 255
    p = tmp_path / 'target.png'; Image.fromarray(mask).save(p)
    common = ['--silhouettes', str(p), '--resolution', '8', '--image-size', '8', '--device', 'cpu']
    soft, output = tmp_path / 'soft', tmp_path / 'body'
    assert main(common + ['--output', str(soft), '--stage', 'soft', '--steps', '1',
                          '--layout-candidates', '1']) == 0
    assert main(common + ['--output', str(output), '--soft-artifact-dir', str(soft / 'soft_training'),
        '--geometry-method', 'regional', '--regional-regions', '0', '--regional-body-reinforcement']) == 2
    settings = json.loads((output / 'GEOMETRY_CONFIG.json').read_text())
    trace = json.loads((output / 'diagnostics/geometry_trace.json').read_text())
    assert settings['body_reinforcement'] is True
    assert trace['body_reinforcement'] is True
    assert settings['method'] == trace['algorithm'] == 'regional_body_path_replacement_v1'
    assert (output / 'source_snapshot/regional_candidates.py').exists()
    assert not (output / 'delivery').exists()
