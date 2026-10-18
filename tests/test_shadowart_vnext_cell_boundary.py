"""Exact exposed-cell boundary regressions for the diagnostic mesh builder."""
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh


def boundary():
    from experiments.shadowart_vnext.cell_boundary import cell_boundary_mesh
    return cell_boundary_mesh


def rendering():
    from experiments.shadowart_vnext import rendering as module
    return module


def camera(direction, *, scale=1.93):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    right = np.cross(direction, [0., 1., 0.])
    if np.linalg.norm(right) < .1:
        right = np.cross(direction, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, direction)
    return SimpleNamespace(origin=tuple(-direction * 3), direction=tuple(direction),
                           right=tuple(right), up=tuple(up), scale=scale)


def test_single_cell_has_literal_bounds_volume_faces_and_outward_winding():
    """Catches missing exterior faces, wrong grid origin, and reversed quads."""
    volume = np.ones((1, 1, 1), dtype=bool)
    mesh = boundary()(volume, extent=2.)

    assert len(mesh.faces) == 12
    assert len(mesh.vertices) == 8
    np.testing.assert_allclose(mesh.bounds, [[-1., -1., -1.], [1., 1., 1.]])
    assert mesh.volume == pytest.approx(8.)
    centers = mesh.triangles_center
    assert np.all(np.einsum('ij,ij->i', mesh.face_normals, centers) > 0)
    assert np.array_equal(volume, np.ones((1, 1, 1), dtype=bool))


def test_asymmetric_rectangular_solid_has_exact_outer_bounds_and_surface_area():
    """Catches XYZ axis transposition and accidental interior-face emission."""
    volume = np.zeros((5, 5, 5), dtype=bool)
    volume[1:3, 2:5, 0:2] = True  # 2 x 3 x 2 cells
    mesh = boundary()(volume, extent=5.)

    np.testing.assert_allclose(mesh.bounds, [[-1.5, -0.5, -2.5], [.5, 2.5, -.5]])
    assert len(mesh.faces) == 64  # 32 exposed square faces, two triangles each
    assert mesh.area == pytest.approx(32.)
    assert mesh.volume == pytest.approx(12.)


def test_hollow_cuboid_keeps_void_faces_and_orients_them_into_the_cavity():
    """Catches cavity filling and orientations that erase the negative volume."""
    volume = np.ones((3, 3, 3), dtype=bool)
    volume[1, 1, 1] = False
    mesh = boundary()(volume, extent=3.)

    assert len(mesh.faces) == 120  # 54 outer + 6 cavity square faces
    assert mesh.volume == pytest.approx(26.)
    cavity_center = np.zeros(3)
    inward = ((np.abs(mesh.triangles_center[:, 0] - .5) < 1e-12) &
              (np.abs(mesh.triangles_center[:, 1]) < .5) &
              (np.abs(mesh.triangles_center[:, 2]) < .5))
    assert inward.any()
    assert np.all(np.einsum('ij,ij->i', mesh.face_normals[inward],
                            cavity_center - mesh.triangles_center[inward]) > 0)


def test_diagonal_edge_contact_preserves_the_exact_nonmanifold_boundary():
    """Catches implicit welding, repair, or connection across a zero-area contact."""
    volume = np.zeros((2, 2, 2), dtype=bool)
    volume[0, 0, 0] = volume[1, 1, 0] = True
    mesh = boundary()(volume, extent=2.)

    assert len(mesh.faces) == 24
    assert len(mesh.vertices) == 14  # two cubes share precisely one grid edge
    assert not mesh.is_watertight
    assert mesh.volume == pytest.approx(2.)


@pytest.mark.parametrize('volume', [
    np.zeros((2, 2, 2), dtype=bool),
    np.ones((2, 3, 2), dtype=bool),
    np.ones((2, 2, 2), dtype=np.uint8),
    [[[True]]],
])
def test_rejects_empty_non_cubic_or_non_boolean_volumes(volume):
    """Catches lossy coercion and malformed diagnostic inputs."""
    with pytest.raises(ValueError):
        boundary()(volume)


@pytest.mark.parametrize('extent', [0., -1., np.inf, -np.inf, np.nan, True])
def test_rejects_nonpositive_or_nonfinite_extent(extent):
    """Catches invalid world-coordinate scales before geometry is emitted."""
    with pytest.raises(ValueError):
        boundary()(np.ones((1, 1, 1), dtype=bool), extent=extent)


@pytest.mark.parametrize('direction', [(0, 0, -1), (1, .37, -.61)])
def test_mesh_and_binary_silhouettes_match_at_generic_pixel_centers(direction):
    """Catches a mesh surface shifted from the physical XYZ cell boundary."""
    volume = np.zeros((5, 5, 5), dtype=bool)
    volume[0:2, 1:4, 2:5] = True
    volume[3:5, 0:2, 0:2] = True
    mesh = boundary()(volume, extent=2.)
    cam = camera(direction)
    expected = rendering().render_binary(volume, [cam], 17, extent=2.)
    actual = rendering().render_mesh(mesh, [cam], 17)
    assert np.array_equal(actual, expected)


def test_obj_and_stl_roundtrips_keep_the_same_projection(tmp_path):
    """Catches frame drift or dependence on transient trimesh processing state."""
    volume = np.zeros((4, 4, 4), dtype=bool)
    volume[0:3, 1:3, 0:2] = True
    volume[3, 3, 3] = True
    mesh = boundary()(volume, extent=2.)
    cam = camera((.31, -.8, -.49))
    expected = rendering().render_mesh(mesh, [cam], 19)
    for suffix in ('obj', 'stl'):
        path = tmp_path / f'boundary.{suffix}'
        mesh.export(path)
        reloaded = trimesh.load(path, force='mesh')
        assert np.array_equal(rendering().render_mesh(reloaded, [cam], 19), expected)
