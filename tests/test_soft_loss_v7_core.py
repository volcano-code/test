from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from shadowforge_pipeline.soft_voxel_v3.data import normalize_target
from shadowforge_pipeline.soft_voxel_v7 import (
    RegisteredEARProtocol,
    build_registered_target_stack,
    build_registered_visual_hull_logits,
    compose_v7_gradient,
    coverage_hit_loss,
    dhw_to_xyz,
    leakage_barrier_loss,
    registered_camera_payload,
    v7_auxiliary_multiplier,
    v7_auxiliary_slot,
    v7_learning_rate,
    xyz_to_dhw,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_INVENTORY = (
    PROJECT_ROOT
    / ".superpowers/sdd/2026-08-29-soft-loss-v7-hit-barrier"
    / "v3-v6-baseline.sha256"
)
CONNECTED_01 = (
    PROJECT_ROOT / "results/fixed3_mr1_connected10_pending_20260722/inputs"
)


def _labelled(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    return (("x", x), ("y", y), ("z", z))


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _projection_from_q(q: torch.Tensor) -> torch.Tensor:
    return q + q.new_tensor(0.8)


def test_registered_target_stack_is_exact_d4_camera_order_and_independent() -> None:
    # Counts and asymmetric locations make every one of the six transformed
    # masks unique, so a swapped D4 operation cannot hide behind fixture symmetry.
    x = torch.tensor(
        [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.bool,
    )
    y = torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.bool,
    )
    z = torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
        dtype=torch.bool,
    )
    expected = torch.tensor(
        [
            [[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        ],
        dtype=torch.bool,
    )

    first = build_registered_target_stack(_labelled(x, y, z))
    second = build_registered_target_stack(_labelled(x, y, z))

    assert torch.equal(first, expected)
    assert len({_tensor_sha256(view) for view in first}) == 6
    assert first.is_contiguous() and first.dtype == torch.bool
    assert first.data_ptr() != second.data_ptr()
    assert all(first.data_ptr() != item.data_ptr() for item in (x, y, z))
    first[0, 0, 0] = ~first[0, 0, 0]
    assert torch.equal(second, expected)


@pytest.mark.parametrize(
    "bad",
    [
        lambda x, y, z: (("y", y), ("x", x), ("z", z)),
        lambda x, y, z: (("x", x), ("x", y), ("z", z)),
        lambda x, y, z: (("x", x), ("y", y)),
        lambda x, y, z: (("x", x.float()), ("y", y), ("z", z)),
        lambda x, y, z: (("x", x), ("y", y[:-1]), ("z", z)),
        lambda x, y, z: (("x", x), ("y", y), ("z", [[False]])),
    ],
)
def test_registered_target_stack_rejects_reorder_duplicates_and_invalid_masks(bad) -> None:
    x = torch.zeros((3, 3), dtype=torch.bool)
    y = torch.zeros((3, 3), dtype=torch.bool)
    z = torch.zeros((3, 3), dtype=torch.bool)
    with pytest.raises(ValueError):
        build_registered_target_stack(bad(x, y, z))


def test_xyz_dhw_adapter_has_exact_literal_and_round_trips_without_aliasing() -> None:
    xyz = torch.arange(8, dtype=torch.int64).reshape(2, 2, 2)
    expected_dhw = torch.tensor([[[7, 3], [5, 1]], [[6, 2], [4, 0]]])

    dhw = xyz_to_dhw(xyz)
    returned = dhw_to_xyz(dhw)

    assert torch.equal(dhw, expected_dhw)
    assert torch.equal(returned, xyz)
    assert dhw.dtype == xyz.dtype and dhw.device == xyz.device
    assert dhw.is_contiguous() and returned.is_contiguous()
    assert dhw.data_ptr() != xyz.data_ptr() and returned.data_ptr() != dhw.data_ptr()
    with pytest.raises(ValueError):
        xyz_to_dhw(torch.zeros((2, 2)))
    with pytest.raises(ValueError):
        dhw_to_xyz(torch.zeros((2, 2, 3)))


def test_registered_hull_is_physical_xyz_then_dhw_with_exact_logits() -> None:
    x = torch.zeros((3, 3), dtype=torch.bool)  # [Y,Z]
    y = torch.zeros((3, 3), dtype=torch.bool)  # [X,Z]
    z = torch.zeros((3, 3), dtype=torch.bool)  # [X,Y]
    x[1, 2] = True
    y[0, 2] = True
    z[0, 1] = True
    expected = torch.zeros((1, 1, 3, 3, 3), dtype=torch.bool)
    expected[0, 0, 0, 1, 2] = True  # xyz(0,1,2) -> dhw(0,1,2)

    first = build_registered_visual_hull_logits(
        _labelled(x, y, z), _resolution=3, _allow_test_size=True
    )
    second = build_registered_visual_hull_logits(
        _labelled(x, y, z), _resolution=3, _allow_test_size=True
    )

    assert first.shape == (1, 1, 3, 3, 3)
    assert first.dtype == torch.float32 and first.is_contiguous()
    assert set(first.unique().tolist()) == {-4.0, 2.0}
    assert torch.equal(first == 2.0, expected)
    assert first.data_ptr() != second.data_ptr()
    with pytest.raises(ValueError):
        build_registered_visual_hull_logits(_labelled(x, y, z))
    with pytest.raises(ValueError):
        build_registered_visual_hull_logits(
            _labelled(x, y, z), _resolution=3, _allow_test_size=True, _dtype=torch.float16
        )
    with pytest.raises(ValueError):
        build_registered_visual_hull_logits(
            _labelled(x, y, z), _resolution=3, _dtype=torch.float32
        )


def test_native_hull_is_exact_256_float32_and_independent_across_calls() -> None:
    x = torch.zeros((256, 256), dtype=torch.bool)
    y = torch.zeros((256, 256), dtype=torch.bool)
    z = torch.zeros((256, 256), dtype=torch.bool)
    x[23, 199] = True
    y[17, 199] = True
    z[17, 23] = True
    first = build_registered_visual_hull_logits(_labelled(x, y, z))
    second = build_registered_visual_hull_logits(_labelled(x, y, z))
    assert first.shape == (1, 1, 256, 256, 256)
    assert first.dtype == torch.float32 and first.is_contiguous()
    assert int((first == 2.0).sum().item()) == 1
    assert first[0, 0, 56, 232, 238].item() == 2.0
    assert _tensor_sha256(first) == _tensor_sha256(second)
    assert first.data_ptr() != second.data_ptr()


def test_registered_camera_payload_binds_scale_projection_and_physical_contract() -> None:
    payload = registered_camera_payload()
    metadata = payload["metadata"]
    assert metadata == {
        "schema": "pcraft_soft_voxel_v7_registered_ear_v1",
        "descriptors": [
            ["z", "D4_3(Bz)", 0, 0.0, 0.0],
            ["x", "Bx", 1, 90.0, 0.0],
            ["y", "D4_7(By)", 2, 0.0, 90.0],
            ["mirror_z", "D4_7(Bz)", 3, 180.0, 0.0],
            ["mirror_x", "D4_4(Bx)", 4, 270.0, 0.0],
            ["mirror_y", "D4_3(By)", 5, 0.0, 270.0],
        ],
        "d4_stack": [3, 0, 7, 7, 4, 3],
        "xyz_to_dhw": "density[d,h,w]=Vxyz[N-1-w,N-1-h,N-1-d]",
        "mirror_policy": "registered-base-D4-only; no raw mirror inputs",
        "distance": 1.7,
        "volume_extent_world": 1.7,
        "samples_per_ray": 150,
        "ray_min_depth": 0.1,
        "ray_max_depth": 2.55,
        "readout": "red channel; P >= 0.8 * stopgrad(max(P)) per view",
        "rotation_dtype": "float32",
        "rotation_shape": [6, 3, 3],
        "translation_dtype": "float32",
        "translation_shape": [6, 3],
        "scale_dtype": "float32",
        "scale_shape": [6, 3],
        "projection_dtype": "float32",
        "projection_shape": [6, 4, 4],
    }
    assert payload["rotation_sha256"] == "2c204607238adff3fa073b41080e258b38dfdbca9e66df78fb29a5b64f3a154f"
    assert payload["translation_sha256"] == "6368ebea8ccac1e1108496cf44acee1b98eb2f5582b40fe15474d18900d337d6"
    assert payload["scale_sha256"] == "7ebced18a08ef839c3435ee07a2b99e283799fd0125ce2c391662989dec2d975"
    assert payload["projection_sha256"] == "d945766503795dc14a6f810c3b53eec8ac203e417be798dcc15bcc0611075732"
    assert len(payload["rotation_bytes"]) == 6 * 3 * 3 * 4
    assert len(payload["translation_bytes"]) == 6 * 3 * 4
    assert len(payload["scale_bytes"]) == 6 * 3 * 4
    assert len(payload["projection_bytes"]) == 6 * 4 * 4 * 4
    assert registered_camera_payload() == payload


@pytest.mark.filterwarnings(
    "ignore:NDCGridRaysampler is deprecated,:PendingDeprecationWarning:"
    "pytorch3d\\.renderer\\.implicit\\.raysampling"
)
def test_renderer_asymmetric_xyz_impulse_has_frozen_screen_signs_and_keeps_scale() -> None:
    density = torch.zeros((1, 1, 17, 17, 17), dtype=torch.float32)
    density[0, 0, 4, 10, 12] = 0.999  # physical XYZ=(4,6,12)
    protocol = RegisteredEARProtocol(
        torch.device("cpu"),
        _image_size=17,
        _n_pts_per_ray=150,
        _allow_test_size=True,
    )
    real_renderer = protocol.renderer
    passed_scales: list[torch.Tensor] = []

    class CaptureRenderer:
        def __call__(self, *, cameras, volumes):
            passed_scales.append(cameras.scale_xyz.detach().clone())
            return real_renderer(cameras=cameras, volumes=volumes)

    protocol.renderer = CaptureRenderer()
    projections = protocol.render_all(density)
    maxima = [torch.nonzero(view == view.max(), as_tuple=False)[0].tolist() for view in projections]

    assert maxima == [[6, 12], [6, 12], [12, 4], [6, 4], [6, 4], [12, 12]]
    assert len(passed_scales) == 6
    expected_scale = torch.tensor([[20.0 / 17.0] * 3], dtype=torch.float32)
    assert all(torch.equal(scale.cpu(), expected_scale) for scale in passed_scales)
    assert torch.equal(protocol.binary_mask(projections), projections >= 0.8 * projections.detach().amax(dim=(-2, -1), keepdim=True))


@pytest.mark.filterwarnings(
    "ignore:NDCGridRaysampler is deprecated,:PendingDeprecationWarning:"
    "pytorch3d\\.renderer\\.implicit\\.raysampling"
)
def test_protocol_rejects_nonproduction_overrides_without_private_test_gate() -> None:
    with pytest.raises(ValueError):
        RegisteredEARProtocol(torch.device("cpu"), _image_size=17)
    with pytest.raises(ValueError):
        RegisteredEARProtocol(torch.device("cpu"), _n_pts_per_ray=16)
    protocol = RegisteredEARProtocol(
        torch.device("cpu"), _image_size=5, _n_pts_per_ray=8, _allow_test_size=True
    )
    with pytest.raises(ValueError):
        protocol.render(torch.zeros((1, 1, 5, 5, 4)), 0)
    with pytest.raises(ValueError):
        protocol.render(torch.zeros((1, 1, 5, 5, 5), dtype=torch.float64), 0)
    with pytest.raises(ValueError):
        protocol.render(torch.zeros((1, 1, 5, 5, 5)), True)
    with pytest.raises(ValueError):
        RegisteredEARProtocol(torch.device("cpu"), _allow_test_size=1)


def test_coverage_hit_uses_per_view_q_stable_top_quarter_and_selected_live_gradient() -> None:
    q = torch.full((6, 3, 3), -0.4, dtype=torch.float64)
    target = torch.zeros((6, 3, 3), dtype=torch.bool)
    target[:, 0, 0] = True
    target[:, 0, 1] = True
    target[:, 0, 2] = True
    target[:, 1, 0] = True
    target[:, 1, 1] = True
    q[:, 2, 2] = 0.2  # per-view normalization anchor
    q[:, 0, 0] = 0.0
    q[:, 0, 1] = 0.0
    q[:, 0, 2] = 0.01
    q[:, 1, 0] = 0.03
    q[:, 1, 1] = 0.04
    projection = _projection_from_q(q).requires_grad_()

    loss, details = coverage_hit_loss(
        projection, target, return_details=True, _allow_test_size=True
    )
    loss.backward()

    expected = 0.02 * math.log1p(math.exp(1.0))
    assert loss.item() == pytest.approx(expected, abs=1e-15)
    assert [row["foreground_count"] for row in details] == [5] * 6
    assert [row["selected_indices"] for row in details] == [[0, 1]] * 6
    assert projection.grad is not None
    assert torch.equal(torch.nonzero(projection.grad != 0)[:, 1:], torch.tensor([[0, 0], [0, 1]]).repeat(6, 1))
    assert projection.grad[:, 2, 2].abs().sum().item() == 0.0  # detached per-view max


def test_coverage_hit_requires_six_nonempty_boolean_registered_views() -> None:
    projection = torch.ones((6, 2, 2), dtype=torch.float32)
    target = torch.ones((6, 2, 2), dtype=torch.bool)
    with pytest.raises(ValueError):
        coverage_hit_loss(projection, target)
    target[3] = False
    with pytest.raises(ValueError):
        coverage_hit_loss(projection, target, _allow_test_size=True)
    for bad_projection, bad_target in (
        (projection[:5], torch.ones((5, 2, 2), dtype=torch.bool)),
        (projection.to(torch.int64), torch.ones_like(target)),
        (projection, target.float()),
        (projection[:, :, :1], target),
    ):
        with pytest.raises(ValueError):
            coverage_hit_loss(bad_projection, bad_target, _allow_test_size=True)
    bad = projection.clone()
    bad[0, 0, 0] = math.nan
    with pytest.raises(ValueError):
        coverage_hit_loss(bad, torch.ones_like(target), _allow_test_size=True)
    extreme = torch.full_like(projection, torch.finfo(torch.float32).max)
    with pytest.raises(ValueError):
        coverage_hit_loss(extreme, torch.ones_like(target), _allow_test_size=True)
    with pytest.raises(ValueError):
        coverage_hit_loss(projection, torch.ones_like(target), _allow_test_size=1)


def _barrier_toy() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_x = torch.zeros((3, 3), dtype=torch.bool)
    target_y = torch.zeros((3, 3), dtype=torch.bool)
    target_z = torch.zeros((3, 3), dtype=torch.bool)
    target_x[1, 2] = True
    target_y[1, 2] = True
    target_z[1, 1] = True
    target = build_registered_target_stack(_labelled(target_x, target_y, target_z))
    q = torch.full((6, 3, 3), -0.4, dtype=torch.float64)
    q[:, 0, 0] = 0.2
    q[target] = 0.2
    q[1, 1, 2] = 0.02
    q[0, 1, 0] = 0.03  # flat 3
    q[0, 1, 2] = 0.00
    q[2, 2, 0] = 0.04  # flat 6
    q[2, 2, 2] = 0.00
    q[3, 1, 0] = 0.10  # flat 3
    q[3, 1, 2] = 0.00
    q[5, 2, 0] = -0.15  # strict inactive boundary
    q[5, 2, 2] = 0.20  # flat 8
    density_xyz = torch.zeros((3, 3, 3), dtype=torch.float64)
    density_xyz[1, 1, 2] = 0.05
    density_dhw = xyz_to_dhw(density_xyz).reshape(1, 1, 3, 3, 3)
    projection = _projection_from_q(q)
    projection[1, 1, 2] = math.nextafter(0.82, -math.inf)  # sole strict missing ray
    projection[5, 2, 0] = 0.65  # exact strict-inactive normalized boundary
    return projection, target, density_dhw


def test_barrier_binding_n3_fixture_has_exact_candidates_support_selection_and_loss() -> None:
    projection, target, density = _barrier_toy()
    projection.requires_grad_()

    loss, details = leakage_barrier_loss(
        projection, target, density, return_details=True, _allow_test_size=True
    )
    loss.backward()

    expected_candidates = torch.zeros((3, 3, 3), dtype=torch.bool)
    expected_candidates[0, 1, 2] = True
    expected_candidates[2, 1, 2] = True
    assert torch.equal(details["candidate_xyz"], expected_candidates)
    assert details["support_counts"] == [2, 0, 2, 2, 0, 1]
    assert details["selected_indices"] == [[3], [], [6], [3], [], [8]]
    assert loss.item() == pytest.approx(0.07543323157521781, abs=1e-15)
    assert projection.grad is not None
    live = torch.nonzero(projection.grad != 0, as_tuple=False).tolist()
    assert live == [[0, 1, 0], [2, 2, 0], [3, 1, 0], [5, 2, 2]]
    assert density.grad is None


def test_barrier_strict_next_float_boundaries_and_hard_density_threshold() -> None:
    projection, target, density = _barrier_toy()
    below_hit = math.nextafter(0.82, -math.inf)
    above_hit = math.nextafter(0.82, math.inf)
    below_live = math.nextafter(0.65, -math.inf)
    above_live = math.nextafter(0.65, math.inf)

    base_loss, base = leakage_barrier_loss(
        projection, target, density, return_details=True, _allow_test_size=True
    )
    assert math.isfinite(base_loss.item())
    assert base["support_counts"][-1] == 1

    for normalized_p, candidate_count in ((below_hit, 2), (0.82, 2), (above_hit, 0)):
        altered = projection.clone()
        altered[1, 1, 2] = normalized_p
        _, details = leakage_barrier_loss(
            altered, target, density, return_details=True, _allow_test_size=True
        )
        assert int(details["candidate_xyz"].sum().item()) == candidate_count

    for normalized_p, support_count in ((below_live, 0), (0.65, 0), (above_live, 1)):
        altered = projection.clone()
        altered[5, 2, 0] = normalized_p
        _, details = leakage_barrier_loss(
            altered, target, density, return_details=True, _allow_test_size=True
        )
        assert details["support_counts"][-1] == 1 + support_count

    low_density = density.clone()
    low_density[low_density == 0.05] = math.nextafter(0.05, -math.inf)
    _, low_details = leakage_barrier_loss(
        projection, target, low_density, return_details=True, _allow_test_size=True
    )
    assert int(low_details["candidate_xyz"].sum().item()) == 3


def test_barrier_float32_decisions_use_detached_q_at_next_float_boundaries() -> None:
    projection64, target, density64 = _barrier_toy()
    projection = projection64.float()
    density = density64.float()
    hit_boundary = torch.tensor(0.82, dtype=torch.float32)
    live_boundary = torch.tensor(0.65, dtype=torch.float32)
    assert (hit_boundary - torch.tensor(0.8, dtype=torch.float32)).item() == 0.019999980926513672
    assert (live_boundary - torch.tensor(0.8, dtype=torch.float32)).item() == -0.15000003576278687

    for normalized_p, candidate_count in (
        (torch.nextafter(hit_boundary, torch.tensor(-math.inf)), 2),
        (hit_boundary, 2),
        (torch.nextafter(hit_boundary, torch.tensor(math.inf)), 0),
    ):
        altered = projection.clone()
        altered[1, 1, 2] = normalized_p
        _, details = leakage_barrier_loss(
            altered, target, density, return_details=True, _allow_test_size=True
        )
        assert int(details["candidate_xyz"].sum().item()) == candidate_count

    for normalized_p, last_support_count in (
        (torch.nextafter(live_boundary, torch.tensor(-math.inf)), 1),
        (live_boundary, 1),
        (torch.nextafter(live_boundary, torch.tensor(math.inf)), 2),
    ):
        altered = projection.clone()
        altered[5, 2, 0] = normalized_p
        _, details = leakage_barrier_loss(
            altered, target, density, return_details=True, _allow_test_size=True
        )
        assert details["support_counts"][-1] == last_support_count


def test_barrier_deduplicates_cross_view_candidates_and_uses_ceil_stable_top_five_percent() -> None:
    projection, target, density = _barrier_toy()
    duplicate_projection = projection.clone()
    duplicate_projection[4, 1, 0] = math.nextafter(0.82, -math.inf)
    _, deduplicated = leakage_barrier_loss(
        duplicate_projection, target, density, return_details=True, _allow_test_size=True
    )
    assert int(deduplicated["candidate_xyz"].sum().item()) == 2

    x21 = torch.zeros((21, 21), dtype=torch.bool)
    x21[10, 10] = True
    empty21 = torch.zeros_like(x21)
    target21 = build_registered_target_stack(_labelled(x21, empty21, empty21))
    q21 = torch.zeros((6, 21, 21), dtype=torch.float64)
    q21[target21] = 0.2
    q21[1, 10, 10] = 0.02
    q21[:, 0, 0] = 0.2
    density21 = torch.zeros((1, 1, 21, 21, 21), dtype=torch.float64)
    projection21 = _projection_from_q(q21)
    projection21[1, 10, 10] = math.nextafter(0.82, -math.inf)
    _, details = leakage_barrier_loss(
        projection21,
        target21,
        density21,
        return_details=True,
        _allow_test_size=True,
    )
    assert details["support_counts"] == [21, 0, 21, 21, 0, 21]
    assert details["selected_indices"] == [[210, 211], [], [210, 211], [210, 211], [], [210, 211]]


def test_barrier_fully_occupied_or_empty_support_is_connected_exact_zero() -> None:
    projection, target, _ = _barrier_toy()
    projection.requires_grad_()
    density = torch.ones((1, 1, 3, 3, 3), dtype=torch.float64, requires_grad=True)
    loss, details = leakage_barrier_loss(
        projection, target, density, return_details=True, _allow_test_size=True
    )
    assert loss.item() == 0.0 and details["support_counts"] == [0] * 6
    loss.backward()
    assert projection.grad is not None and torch.count_nonzero(projection.grad).item() == 0
    assert density.grad is None


def test_barrier_rejects_invalid_shapes_dtypes_aliases_nonfinite_and_extreme_overflow() -> None:
    projection, target, density = _barrier_toy()
    with pytest.raises(ValueError):
        leakage_barrier_loss(projection[:5], target[:5], density, _allow_test_size=True)
    with pytest.raises(ValueError):
        leakage_barrier_loss(projection.float(), target, density, _allow_test_size=True)
    with pytest.raises(ValueError):
        leakage_barrier_loss(projection, target.float(), density, _allow_test_size=True)
    with pytest.raises(ValueError):
        leakage_barrier_loss(projection, target, density.reshape(3, 3, 3), _allow_test_size=True)
    with pytest.raises(ValueError):
        leakage_barrier_loss(projection, target, density.float(), _allow_test_size=True)
    bad_projection = projection.clone()
    bad_projection[0, 0, 0] = math.inf
    with pytest.raises(ValueError):
        leakage_barrier_loss(bad_projection, target, density, _allow_test_size=True)
    bad_density = density.clone()
    bad_density[0, 0, 0, 0, 0] = math.nan
    with pytest.raises(ValueError):
        leakage_barrier_loss(projection, target, bad_density, _allow_test_size=True)
    extreme = torch.full_like(projection, torch.finfo(torch.float64).max)
    with pytest.raises(ValueError):
        leakage_barrier_loss(extreme, target, density, _allow_test_size=True)
    with pytest.raises(ValueError):
        leakage_barrier_loss(projection, target, density, _allow_test_size=1)
    with pytest.raises(ValueError):
        leakage_barrier_loss(projection, target, density)


def test_v7_multiplier_slots_and_learning_rate_use_completed_before_boundaries() -> None:
    assert {value: v7_auxiliary_multiplier(value) for value in (0, 99, 100, 101, 298, 299, 300, 2999)} == {
        0: 0.0,
        99: 0.0,
        100: 1.0 / 200.0,
        101: 2.0 / 200.0,
        298: 199.0 / 200.0,
        299: 1.0,
        300: 1.0,
        2999: 1.0,
    }
    assert [v7_auxiliary_slot(i, "full") for i in range(4)] == ["hit", "barrier", "hit", "barrier"]
    assert [v7_auxiliary_slot(i, "base") for i in range(4)] == ["base"] * 4
    assert [v7_auxiliary_slot(i, "hit_only") for i in range(4)] == ["hit", "base", "hit", "base"]
    assert [v7_auxiliary_slot(i, "barrier_only") for i in range(4)] == ["base", "barrier", "base", "barrier"]
    assert v7_learning_rate(0) == 0.007
    assert v7_learning_rate(1499) == 0.007
    assert v7_learning_rate(1500) == 0.007
    assert v7_learning_rate(1501) == 0.0007 + 0.5 * (0.007 - 0.0007) * (1.0 + math.cos(math.pi / 1199.0))
    assert v7_learning_rate(2699) == 0.0007
    assert v7_learning_rate(2700) == 0.0007
    assert v7_learning_rate(2999) == 0.0007
    for invalid in (True, 1.0, math.nextafter(100.0, math.inf), -1, 3000, "100"):
        with pytest.raises(ValueError):
            v7_auxiliary_multiplier(invalid)
        with pytest.raises(ValueError):
            v7_learning_rate(invalid)
    with pytest.raises(ValueError):
        v7_auxiliary_slot(0, "unknown")


@pytest.mark.parametrize("ratio", [0.10, 0.15, 0.20])
def test_gradient_projects_conflict_caps_norm_and_keeps_inputs_bitwise(ratio: float) -> None:
    base = torch.tensor([3.0, 4.0], dtype=torch.float64)
    auxiliary = torch.tensor([-8.0, 2.0], dtype=torch.float64)
    base_sha = _tensor_sha256(base)
    auxiliary_sha = _tensor_sha256(auxiliary)

    result, telemetry = compose_v7_gradient(
        base, auxiliary, ratio, 0.5, return_telemetry=True
    )
    contribution = result - base

    assert torch.dot(contribution, base).item() >= 0.0
    assert torch.linalg.vector_norm(contribution).item() <= math.nextafter(
        ratio * 0.5 * 5.0, math.inf
    )
    assert telemetry["ratio"] == ratio
    assert telemetry["multiplier"] == 0.5
    assert telemetry["conflict_projected"] is True
    assert all(math.isfinite(v) for v in telemetry.values() if type(v) in (int, float))
    assert _tensor_sha256(base) == base_sha and _tensor_sha256(auxiliary) == auxiliary_sha
    assert result.data_ptr() != base.data_ptr() and result.data_ptr() != auxiliary.data_ptr()


def test_gradient_zero_base_or_multiplier_returns_base_and_rejects_constructed_off_slot() -> None:
    zero = torch.zeros(3, dtype=torch.float32)
    auxiliary = torch.ones(3, dtype=torch.float32)
    result, telemetry = compose_v7_gradient(zero, auxiliary, 0.10, 1.0, return_telemetry=True)
    assert torch.equal(result, zero) and result.data_ptr() != zero.data_ptr()
    assert telemetry["base_norm"] == 0.0 and telemetry["final_auxiliary_norm"] == 0.0
    base = torch.tensor([3.0, 4.0])
    assert torch.equal(compose_v7_gradient(base, None, 0.10, 0.0), base)
    with pytest.raises(ValueError, match="off-slot"):
        compose_v7_gradient(base, auxiliary[:2], 0.10, 0.0)
    with pytest.raises(ValueError, match="missing"):
        compose_v7_gradient(base, None, 0.10, 1.0)


def test_gradient_fail_closed_validation_aliasing_and_finite_extremes() -> None:
    base = torch.tensor([3.0, 4.0], dtype=torch.float32)
    auxiliary = torch.tensor([1.0, 2.0], dtype=torch.float32)
    for ratio in (True, 0.1 + 1e-12, 0, "0.10"):
        with pytest.raises(ValueError):
            compose_v7_gradient(base, auxiliary, ratio, 1.0)
    for multiplier in (True, 1, -math.nextafter(0.0, math.inf), math.nextafter(1.0, math.inf), "1"):
        with pytest.raises(ValueError):
            compose_v7_gradient(base, auxiliary, 0.10, multiplier)
    with pytest.raises(ValueError):
        compose_v7_gradient(base, base, 0.10, 1.0)
    with pytest.raises(ValueError):
        compose_v7_gradient(base, base[:], 0.10, 1.0)
    with pytest.raises(ValueError):
        compose_v7_gradient(base, auxiliary.double(), 0.10, 1.0)
    with pytest.raises(ValueError):
        compose_v7_gradient(base, torch.ones(3), 0.10, 1.0)
    with pytest.raises(ValueError):
        compose_v7_gradient(base, torch.tensor([math.inf, 0.0]), 0.10, 1.0)
    with pytest.raises(ValueError):
        compose_v7_gradient(
            torch.tensor([torch.finfo(torch.float32).max]),
            torch.tensor([torch.finfo(torch.float32).max]),
            0.10,
            1.0,
        )
    with pytest.raises(ValueError):
        compose_v7_gradient(base, auxiliary, 0.10, 1.0, return_telemetry=1)


@pytest.mark.parametrize(
    ("base_values", "auxiliary_values", "dtype"),
    [
        (
            [-1.2888376712799072, -1.4524918794631958],
            [-0.612324595451355, 1.184743046760559],
            torch.float32,
        ),
        (
            [-0.43122029304504395, -0.9840558171272278],
            [-1.1085643768310547, 0.8045943975448608],
            torch.float32,
        ),
        (
            [1.4011954, 0.2133064],
            [-0.1435223, -0.6835628],
            torch.float32,
        ),
        (
            [0.4996406137943268, -1.1001375913619995],
            [1.318636178970337, -0.7803286910057068],
            torch.float32,
        ),
        (
            [-0.4098202414239914, 1.2507489198843105],
            [0.9814279144137116, -0.8540244604186222],
            torch.float64,
        ),
    ],
    ids=(
        "fix1-negative-promoted-dot",
        "fix1-represented-cap-excess",
        "fix1-terminal-error",
        "fix2-promoted-cap-excess",
        "fix2-direct-dot-negative",
    ),
)
def test_gradient_review_witnesses_always_return_safe_represented_contribution(
    base_values: list[float], auxiliary_values: list[float], dtype: torch.dtype
) -> None:
    base = torch.tensor(base_values, dtype=dtype)
    auxiliary = torch.tensor(auxiliary_values, dtype=dtype)
    base_sha = _tensor_sha256(base)
    auxiliary_sha = _tensor_sha256(auxiliary)

    result, telemetry = compose_v7_gradient(
        base, auxiliary, 0.10, 1.0, return_telemetry=True
    )
    represented = (result - base).to(torch.float64)
    promoted_base = base.to(torch.float64)
    promoted_dot = torch.dot(represented, promoted_base).item()
    exact_ratio = (
        torch.linalg.vector_norm(represented) / torch.linalg.vector_norm(promoted_base)
    ).item()

    assert torch.isfinite(result).all().item()
    assert promoted_dot >= 0.0
    assert exact_ratio <= 0.10
    assert telemetry["base_norm"] == torch.linalg.vector_norm(promoted_base).item()
    assert telemetry["final_auxiliary_norm"] == torch.linalg.vector_norm(represented).item()
    assert telemetry["final_auxiliary_base_ratio"] == exact_ratio
    assert _tensor_sha256(base) == base_sha
    assert _tensor_sha256(auxiliary) == auxiliary_sha


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradient_seeded_adversarial_outputs_and_telemetry_use_same_terminal_metrics(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(20260829)
    ratios = (0.10, 0.15, 0.20)
    multipliers = (0.005, 0.5, 1.0)
    for index in range(24):
        base = torch.randn(17, generator=generator, dtype=dtype)
        auxiliary = torch.randn(17, generator=generator, dtype=dtype)
        ratio = ratios[index % len(ratios)]
        multiplier = multipliers[(index // len(ratios)) % len(multipliers)]

        result, telemetry = compose_v7_gradient(
            base, auxiliary, ratio, multiplier, return_telemetry=True
        )
        represented = (result - base).to(torch.float64)
        promoted_base = base.to(torch.float64)
        represented_norm = torch.linalg.vector_norm(represented).item()
        promoted_base_norm = torch.linalg.vector_norm(promoted_base).item()
        exact_ratio = represented_norm / promoted_base_norm

        assert torch.dot(represented, promoted_base).item() >= 0.0
        assert exact_ratio <= ratio * multiplier
        assert telemetry["base_norm"] == promoted_base_norm
        assert telemetry["final_auxiliary_norm"] == represented_norm
        assert telemetry["final_auxiliary_base_ratio"] == exact_ratio


def _load_connected_01_registration():
    paths = [CONNECTED_01 / f"connected_01_view_{axis}.png" for axis in "xyz"]
    masks = []
    for path in paths:
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert raw is not None
        masks.append(torch.from_numpy(normalize_target(raw, 256, False) > 0))
    views = _labelled(*masks)
    targets = build_registered_target_stack(views)
    hull_dhw = build_registered_visual_hull_logits(views)[0, 0] == 2.0
    return paths, views, targets, hull_dhw


def test_connected_01_cpu_registration_and_hull_hash_sentinel() -> None:
    paths, views, targets, hull_dhw = _load_connected_01_registration()
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == [
        "075ea08eb090ccd5fc2350b4e7393524cb7c154da5fd17e4656b1fb8d367146c",
        "ed684af3ebc6c054bfe272d9b9be2e6df0617231b9691873572517b7f2afa547",
        "5de721e47808bfd7785eb94e40f804f9659718282a135b2b27cf6d99d6ab6f1b",
    ]
    hull_xyz = dhw_to_xyz(hull_dhw)
    assert _tensor_sha256(targets) == "e7ff318efce18002d57a86d01db879ce147bea7a70ad6fd759ddc27cedf92ef6"
    assert _tensor_sha256(hull_xyz) == "6af5b6c72693ddcf4433671d154bcbc39f2633898f5bfc556cade76f2a8b259d"
    assert _tensor_sha256(hull_dhw) == "cfa5e7c2ef33f5c3e731d8f8496d4611c71531d861a8acb0bfb311984b486fb1"
    assert int(hull_dhw.sum().item()) == 798753
    assert targets.sum(dim=(-2, -1)).tolist() == [13416, 15917, 12343, 13416, 15917, 12343]
    with pytest.raises(ValueError):
        build_registered_target_stack((views[1], views[0], views[2]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="frozen renderer sentinel requires CUDA")
@pytest.mark.filterwarnings(
    "ignore:NDCGridRaysampler is deprecated,:PendingDeprecationWarning:"
    "pytorch3d\\.renderer\\.implicit\\.raysampling"
)
def test_connected_01_cuda_renderer_sentinel() -> None:
    _, _, targets, hull_dhw = _load_connected_01_registration()
    device = torch.device("cuda:0")
    protocol = RegisteredEARProtocol(device)
    density = hull_dhw.to(device=device, dtype=torch.float32).reshape(1, 1, 256, 256, 256) * 0.999
    predicted = protocol.binary_mask(protocol.render_all(density)).cpu()
    intersection = (predicted & targets).sum(dim=(-2, -1), dtype=torch.float64)
    union = (predicted | targets).sum(dim=(-2, -1), dtype=torch.float64)
    false_positive = (predicted & ~targets).sum(dim=(-2, -1))
    mean_iou = (intersection / union).mean().item()
    assert false_positive.tolist() == [0, 0, 0, 0, 0, 0]
    assert mean_iou == pytest.approx(0.9932741630045697, abs=1e-7)


def test_public_exports_and_signatures_are_exact() -> None:
    import shadowforge_pipeline.soft_voxel_v7 as public

    expected = {
        "RegisteredEARProtocol",
        "build_registered_target_stack",
        "build_registered_visual_hull_logits",
        "compose_v7_gradient",
        "coverage_hit_loss",
        "dhw_to_xyz",
        "leakage_barrier_loss",
        "registered_camera_payload",
        "v7_auxiliary_multiplier",
        "v7_auxiliary_slot",
        "v7_learning_rate",
        "xyz_to_dhw",
    }
    assert set(public.__all__) == expected
    assert all(hasattr(public, name) for name in expected)
    assert tuple(inspect.signature(build_registered_target_stack).parameters) == ("suite_views",)
    assert tuple(inspect.signature(xyz_to_dhw).parameters) == ("volume_xyz",)
    assert tuple(inspect.signature(dhw_to_xyz).parameters) == ("volume_dhw",)
    assert tuple(inspect.signature(v7_auxiliary_multiplier).parameters) == ("completed_updates",)
    assert tuple(inspect.signature(v7_auxiliary_slot).parameters) == ("completed_updates", "profile")
    assert tuple(inspect.signature(v7_learning_rate).parameters) == ("completed_updates",)


def test_protected_157_entry_inventory_remains_exact() -> None:
    rows = [line.split(maxsplit=1) for line in BASELINE_INVENTORY.read_text().splitlines() if line]
    assert len(rows) == 157
    for expected_sha256, path_text in rows:
        path = Path(path_text)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256, path_text
