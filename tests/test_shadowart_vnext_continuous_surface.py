"""Native probability interpolation must not secretly harden or rescale data."""
import importlib

import numpy as np
import pytest


def surface(p, extent=1.7):
    module = importlib.import_module('experiments.shadowart_vnext.continuous_surface')
    return module.probability_surface(p, extent)


def test_interpolated_boundary_uses_fixed_half_probability_not_binary_sdf():
    assert importlib.util.find_spec('experiments.shadowart_vnext.continuous_surface'), 'missing continuous surface'
    p = np.full((4, 4, 4), .8, np.float32); before = p.copy()
    mesh = surface(p)
    # First center=-.85+.2125; surface .375 pitch outward towards padded zero.
    np.testing.assert_allclose(mesh.bounds, [[-.796875]*3, [.796875]*3], atol=2e-7)
    np.testing.assert_array_equal(p, before)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0


def test_closed_cavity_is_retained_and_native_center_membership_reloads():
    from experiments.shadowart_vnext.rendering import voxelize_mesh
    p = np.ones((8, 8, 8), np.float32); p[2:6, 2:6, 2:6] = 0
    mesh = surface(p)
    assert len(mesh.split()) == 2
    assert sorted(np.sign(part.volume) for part in mesh.split()) == [-1, 1]
    np.testing.assert_array_equal(voxelize_mesh(mesh, 8), p >= .5)


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan')])
def test_bad_probability_is_rejected(value):
    with pytest.raises(ValueError):
        surface(np.full((4, 4, 4), value, np.float32))


def test_empty_and_non_cubic_fields_are_rejected():
    with pytest.raises(ValueError):
        surface(np.zeros((4, 4, 4), np.float32))
    with pytest.raises(ValueError):
        surface(np.ones((3, 4, 4), np.float32))

