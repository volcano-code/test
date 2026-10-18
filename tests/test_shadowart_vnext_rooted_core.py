"""Hand-derived topology, widest-path, alignment, and derivative contracts."""
import ctypes
import importlib
import importlib.util
import subprocess

import numpy as np
import pytest
import torch


def api():
    name = "experiments.shadowart_vnext.rooted_core"
    assert importlib.util.find_spec(name) is not None, "rooted-core operator missing"
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    return api().build_backend(tmp_path_factory.mktemp("rooted") / "build")


def test_operator_is_available():
    assert callable(api().rooted_core)


def test_backend_rejects_existing_directory_and_captured_source_works(tmp_path, backend):
    with pytest.raises(FileExistsError):
        api().build_backend(tmp_path)
    other = api().build_backend(tmp_path / "captured", source_path=backend.source_path)
    assert other.source_sha256 == backend.source_sha256
    p = torch.ones((2, 2, 2), dtype=torch.float64)
    assert torch.equal(api().rooted_core(p, 2, (0, 0, 0), other), p)
    with pytest.raises(FileNotFoundError):
        api().build_backend(tmp_path / "missing", source_path=tmp_path / "absent.cpp")
    with pytest.raises(subprocess.CalledProcessError):
        api().build_backend(tmp_path / "invalid", source_path=__file__)


def test_native_invalid_arguments_return_error_status_without_exception(backend):
    field = np.ones(1, dtype=np.float64)
    provenance = np.zeros(1, dtype=np.int64)
    pointer = field.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    index_pointer = provenance.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    for n, root in [(0, 0), (1, 1), (2**62, 0)]:
        assert backend.library.rooted_reconstruct_f64(pointer, n, root, pointer, index_pointer) != 0


@pytest.mark.parametrize("neck_width", [0, 1, 2])
def test_thick_blocks_need_thick_neck(neck_width, backend):
    p = torch.zeros((12,) * 3, dtype=torch.float64)
    p[1:4, 2:5, 2:5] = 1
    p[8:11, 2:5, 2:5] = 1
    if neck_width:
        p[4:8, 2:2 + neck_width, 2:2 + neck_width] = 1
    want = p.clone() if neck_width == 2 else torch.zeros_like(p)
    if neck_width != 2:
        want[1:4, 2:5, 2:5] = 1
    assert torch.equal(api().rooted_core(p, 2, (1, 2, 2), backend), want)


def test_six_connectivity_accepts_faces_rejects_diagonals(backend):
    p = torch.zeros((4,) * 3, dtype=torch.float64)
    p[0, 0, 0] = p[1, 0, 0] = p[2, 1, 0] = 1
    want = torch.zeros_like(p)
    want[0, 0, 0] = want[1, 0, 0] = 1
    assert torch.equal(api().rooted_reconstruction(p, (0, 0, 0), backend), want)


def test_hollow_shell_preserves_cavity(backend):
    p = torch.zeros((12,) * 3, dtype=torch.float64)
    p[1:11, 1:11, 1:11] = 1
    p[3:9, 3:9, 3:9] = 0
    assert torch.equal(api().rooted_core(p, 2, (1, 1, 1), backend), p)


@pytest.mark.parametrize("width", [1, 2, 3, 5])
def test_far_boundary_block_does_not_translate_or_shrink(width, backend):
    p = torch.zeros((5,) * 3, dtype=torch.float64)
    p[5 - width:, 5 - width:, 5 - width:] = 1
    assert torch.equal(api().rooted_core(p, width, (5 - width,) * 3, backend), p)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_root_cannot_seed_material(dtype, backend):
    p = torch.zeros((4,) * 3, dtype=dtype)
    assert torch.equal(api().rooted_core(p, 2, (0, 0, 0), backend), p)
    p[2:, 2:, 2:] = 1
    out = api().rooted_core(p, 2, (0, 0, 0), backend)
    assert torch.isfinite(out).all() and not out.any()


def test_path_value_and_gradient_reach_weak_bottleneck(backend):
    p = torch.full((3,) * 3, .01, dtype=torch.float64)
    p[:, 1, 1] = torch.tensor([.9, .3, .8], dtype=p.dtype)
    p.requires_grad_()
    out = api().rooted_reconstruction(p, (0, 1, 1), backend)
    assert out[:, 1, 1].tolist() == pytest.approx([.9, .3, .3])
    out[2, 1, 1].backward()
    want = torch.zeros_like(p)
    want[1, 1, 1] = 1
    assert torch.equal(p.grad, want)


def test_longer_wider_route_beats_short_low_capacity_route(backend):
    p = torch.full((3,) * 3, .01, dtype=torch.float64)
    p[:, 0, 0] = torch.tensor([.9, .2, .8], dtype=p.dtype)
    p[:, 1, 0] = torch.tensor([.85, .7, .75], dtype=p.dtype)
    p.requires_grad_()
    out = api().rooted_reconstruction(p, (0, 0, 0), backend)
    assert out[2, 0, 0].item() == pytest.approx(.7)
    out[2, 0, 0].backward()
    want = torch.zeros_like(p)
    want[1, 1, 0] = 1
    assert torch.equal(p.grad, want)


def iterative_reference(capacity, root):
    """Synchronous six-neighbour max-min fixed point, independent of heap code."""
    out = np.zeros_like(capacity)
    out[root] = capacity[root]
    for _ in range(capacity.size):
        following = out.copy()
        for axis in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(None, -1)
            hi[axis] = slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            following[lo] = np.maximum(following[lo], np.minimum(capacity[lo], out[hi]))
            following[hi] = np.maximum(following[hi], np.minimum(capacity[hi], out[lo]))
        if np.array_equal(out, following):
            return out
        out = following
    raise AssertionError("reference did not converge")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("root", [(0, 0, 0), (3, 3, 3), (1, 2, 1)])
def test_random_forward_matches_independent_fixed_point(dtype, root, backend):
    for seed in range(4):
        p = torch.tensor(np.random.default_rng(seed).uniform(.01, .99, (4,) * 3), dtype=dtype)
        want = iterative_reference(p.numpy(), root)
        np.testing.assert_array_equal(api().rooted_reconstruction(p, root, backend).numpy(), want)


@pytest.mark.parametrize("width", [1, 2, 3])
def test_unique_capacities_gradcheck_and_nonmutation(width, backend):
    rng = np.random.default_rng(71 + width)
    # Distinct originals with gaps > 0.001; pooling duplicates a selected original,
    # but finite differences do not change the active original bottleneck.
    values = rng.permutation(np.linspace(.1, .9, 64)).reshape((4,) * 3)
    p = torch.tensor(values, dtype=torch.float64).requires_grad_()
    before = p.detach().clone()
    function = lambda field: api().rooted_core(field, width, (0, 0, 0), backend)
    assert torch.autograd.gradcheck(function, (p,), eps=1e-6, atol=1e-5, rtol=1e-4)
    out = function(p)
    assert out.dtype == p.dtype and out.shape == p.shape
    assert torch.all(out <= p)
    assert torch.equal(out, function(p)) and torch.equal(p, before)
    out.sum().backward()
    assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    if width == 1:
        assert torch.autograd.gradcheck(
            lambda field: api().rooted_reconstruction(field, (2, 1, 0), backend), (p,))


def test_noncontiguous_input_and_deterministic_tie_gradients(backend):
    p = torch.full((3,) * 3, .4, dtype=torch.float64).transpose(0, 2).requires_grad_()
    outputs, grads = [], []
    for _ in range(2):
        out = api().rooted_reconstruction(p, (1, 1, 1), backend)
        outputs.append(out)
        grads.append(torch.autograd.grad(out.sum(), p)[0])
    assert torch.equal(outputs[0], p)
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(grads[0], grads[1])
    assert grads[0].sum() == 27


@pytest.mark.parametrize("bad", [
    torch.zeros((2, 2)), torch.zeros((2, 2, 3)), torch.zeros((0, 0, 0)),
    torch.zeros((2,) * 3, dtype=torch.int64), torch.zeros((2,) * 3, dtype=torch.float16),
    torch.full((2,) * 3, float("nan")), torch.full((2,) * 3, float("inf")),
    torch.full((2,) * 3, -.01), torch.full((2,) * 3, 1.01), np.zeros((2,) * 3),
])
def test_invalid_probability_rejected(bad, backend):
    for fn in [lambda: api().rooted_core(bad, 1, (0, 0, 0), backend),
               lambda: api().rooted_reconstruction(bad, (0, 0, 0), backend)]:
        with pytest.raises((ValueError, TypeError)):
            fn()


@pytest.mark.parametrize("root", [(), (0, 0), (0, 0, 0, 0), (-1, 0, 0),
                                       (2, 0, 0), (0., 0, 0), (True, 0, 0), None])
def test_invalid_root_rejected(root, backend):
    p = torch.ones((3,) * 3, dtype=torch.float64)
    with pytest.raises((ValueError, TypeError)):
        api().rooted_core(p, 2, root, backend)


@pytest.mark.parametrize("width", [0, -1, 4, 1.5, True, None])
def test_invalid_width_rejected(width, backend):
    p = torch.ones((3,) * 3, dtype=torch.float64)
    with pytest.raises((ValueError, TypeError)):
        api().rooted_core(p, width, (0, 0, 0), backend)
    with pytest.raises((ValueError, TypeError)):
        api().choose_root(p, width)


def test_root_selection_largest_component_then_padded_distance_then_flat_tie():
    p = np.zeros((14,) * 3, dtype=np.float64)
    p[1:4, 1:4, 1:4] = 1
    p[6:12, 6:12, 6:12] = 1
    before = p.copy()
    assert api().choose_root(p, 2) == (8, 8, 8)
    np.testing.assert_array_equal(p, before)
    # Whole-domain anchors need outside padding for a correct interior root.
    assert api().choose_root(torch.ones((6,) * 3), 2) == (2, 2, 2)
    assert api().choose_root(torch.ones((5,) * 3), 2) == (1, 1, 1)
    # Equal components resolve deterministically to the first flat component.
    p[6:12, 6:12, 6:12] = 0
    p[8:11, 8:11, 8:11] = 1
    assert api().choose_root(p, 2) == (1, 1, 1)
    with pytest.raises(ValueError):
        api().choose_root(np.zeros((3,) * 3), 2)


def test_root_selection_threshold_is_inclusive_and_rejects_invalid_values():
    assert api().choose_root(np.full((2,) * 3, .5), 2) == (0, 0, 0)
    for bad in [np.full((2,) * 3, .499), np.full((2,) * 3, np.nan),
                np.full((2,) * 3, 1.1), np.zeros((2, 2))]:
        with pytest.raises((ValueError, TypeError)):
            api().choose_root(bad, 1)


def test_largest_component_wins_over_smaller_deeper_component():
    p = np.zeros((16,) * 3, dtype=np.float64)
    p[1:3, 1:14, 1:14] = 1  # 338 voxels, depth 1.
    p[7:11, 7:11, 7:11] = 1  # 64 voxels, depth 2.
    assert api().choose_root(p, 1) == (1, 1, 1)


def test_core_gradient_flows_to_only_weak_original_neck_voxel(backend):
    p = torch.full((6,) * 3, .01, dtype=torch.float64)
    p[:, 2:4, 2:4] = .9
    p[2, 2, 2] = .3
    p.requires_grad_()
    out = api().rooted_core(p, 2, (0, 2, 2), backend)
    assert out[5, 2, 2].item() == pytest.approx(.3)
    out[5, 2, 2].backward()
    want = torch.zeros_like(p)
    want[2, 2, 2] = 1
    assert torch.equal(p.grad, want)
