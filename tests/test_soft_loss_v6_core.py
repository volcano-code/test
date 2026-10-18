from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from shadowforge_pipeline.soft_voxel_v3.data import normalize_target
from shadowforge_pipeline.soft_voxel_v5.quick_ablation import (
    background_hard_negative_tail_loss as v5_h_loss,
)
from shadowforge_pipeline.soft_voxel_v5.quick_ablation import (
    delivery_lovasz_loss as v5_j_loss,
)
from shadowforge_pipeline.soft_voxel_v6 import (
    CANONICAL_VIEW_DESCRIPTORS,
    FIXED_CAMERA_AXES,
    PROFILE_FLAGS,
    PROFILE_ORDER,
    TASK_ORDER,
    background_hard_negative_tail_loss,
    build_fixed_visual_hull_logits,
    compose_v6_gradient,
    delivery_lovasz_loss,
    fixed_camera_payload,
    map_suite_views,
    v6_auxiliary_multiplier,
    v6_component_budgets,
    v6_learning_rate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_INVENTORY = (
    PROJECT_ROOT
    / ".superpowers/sdd/2026-08-28-soft-loss-v6-256-ablation"
    / "v3-v4-v5-baseline.sha256"
)


def _labelled(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    return (("x", x), ("y", y), ("z", z))


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def test_canonical_constants_and_profile_flags_are_exact() -> None:
    assert TASK_ORDER == tuple(f"connected_{index:02d}" for index in range(1, 11))
    assert PROFILE_ORDER == ("base", "j_only", "h_only", "full")
    assert {name: dict(PROFILE_FLAGS[name]) for name in PROFILE_ORDER} == {
        "base": {"j": False, "h": False},
        "j_only": {"j": True, "h": False},
        "h_only": {"j": False, "h": True},
        "full": {"j": True, "h": True},
    }
    assert FIXED_CAMERA_AXES == ("y", "x", "z")
    with pytest.raises(TypeError):
        PROFILE_FLAGS["base"]["j"] = True


def test_fixed_camera_payload_serializes_labels_parameters_and_canonical_rt_bytes() -> None:
    from pytorch3d.renderer import look_at_view_transform

    payload = fixed_camera_payload()
    expected_descriptors = (
        ("y", "target_y", 0, 0.0, 0.0),
        ("x", "target_x", 1, 90.0, 0.0),
        ("z", "target_z", 2, 0.0, 90.0),
        ("mirror_y", "target_mirror_y", 3, 180.0, 0.0),
        ("mirror_x", "target_mirror_x", 4, 270.0, 0.0),
        ("mirror_z", "target_mirror_z", 5, 0.0, 270.0),
    )
    assert CANONICAL_VIEW_DESCRIPTORS == expected_descriptors
    reference_r, reference_t = look_at_view_transform(
        dist=1.7,
        elev=torch.tensor((0.0, 0.0, 90.0, 0.0, 0.0, 270.0)),
        azim=torch.tensor((0.0, 90.0, 0.0, 180.0, 270.0, 0.0)),
        device=torch.device("cpu"),
    )
    metadata = {
        "schema": "pcraft_soft_voxel_v6_fixed_cameras_v1",
        "descriptors": [list(row) for row in expected_descriptors],
        "distance": 1.7,
        "rotation_dtype": "float32",
        "rotation_shape": [6, 3, 3],
        "translation_dtype": "float32",
        "translation_shape": [6, 3],
        "native_hull_identity": {
            "canonical_first_three_labels": ["y", "x", "z"],
            "axis_for_view": ["z", "x", "y"],
            "d4_forward_ops": [3, 0, 7],
            "d4_inverse_ops": [1, 0, 7],
            "mirror_policy": "raw-horizontal-flip-then-independent-normalize",
        },
    }
    expected_hash = hashlib.sha256(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\0"
        + reference_r.contiguous().numpy().tobytes(order="C")
        + b"\0"
        + reference_t.contiguous().numpy().tobytes(order="C")
    ).hexdigest()

    assert payload["metadata"] == metadata
    assert payload["rotation_bytes"] == reference_r.contiguous().numpy().tobytes(order="C")
    assert payload["translation_bytes"] == reference_t.contiguous().numpy().tobytes(order="C")
    assert payload["rotation_sha256"] == "2c204607238adff3fa073b41080e258b38dfdbca9e66df78fb29a5b64f3a154f"
    assert payload["translation_sha256"] == "6368ebea8ccac1e1108496cf44acee1b98eb2f5582b40fe15474d18900d337d6"
    assert payload["payload_sha256"] == expected_hash
    assert expected_hash == "6e96f730d99bdf9cdd6a3c7453c629c6eac91371208ffabe7b5af0fccb912e49"
    mutated = json.loads(json.dumps(metadata))
    mutated["native_hull_identity"]["axis_for_view"][0] = "x"
    mutated_hash = hashlib.sha256(
        json.dumps(
            mutated, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\0"
        + payload["rotation_bytes"]
        + b"\0"
        + payload["translation_bytes"]
    ).hexdigest()
    assert mutated_hash != payload["payload_sha256"]
    assert fixed_camera_payload() == payload


def test_mapper_requires_independently_normalized_raw_flip_masks_and_validates_both_sequences() -> None:
    assert tuple(inspect.signature(map_suite_views).parameters) == (
        "suite_views",
        "mirrored_suite_views",
    )
    raw_y = np.array(
        [
            [0, 0, 255, 0, 0, 255, 0, 0, 255],
            [0, 0, 0, 255, 0, 0, 0, 0, 0],
            [0, 0, 255, 0, 255, 255, 0, 0, 255],
            [0, 0, 255, 0, 0, 0, 0, 0, 0],
            [0, 0, 255, 255, 0, 255, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 255],
            [255, 0, 0, 0, 0, 0, 255, 0, 0],
        ],
        dtype=np.uint8,
    )
    raw_x = np.rot90(raw_y).copy()
    raw_z = np.roll(raw_y, 1, axis=0)
    raw_xyz = (raw_x, raw_y, raw_z)
    base = tuple(torch.from_numpy(normalize_target(raw, 16) > 0) for raw in raw_xyz)
    mirrored = tuple(
        torch.from_numpy(normalize_target(cv2.flip(raw, 1), 16) > 0)
        for raw in raw_xyz
    )
    assert not torch.equal(mirrored[1], torch.flip(base[1], dims=(-1,)))

    mapped = map_suite_views(_labelled(*base), _labelled(*mirrored))

    assert mapped.shape == (6, 16, 16)
    assert torch.equal(mapped[0], base[1])
    assert torch.equal(mapped[1], base[0])
    assert torch.equal(mapped[2], base[2])
    assert torch.equal(mapped[3], mirrored[1])
    assert torch.equal(mapped[4], mirrored[0])
    assert torch.equal(mapped[5], mirrored[2])
    assert mapped.data_ptr() not in {
        tensor.data_ptr() for tensor in base + mirrored
    }

    duplicate_axis = (("x", base[0]), ("x", base[1]), ("z", base[2]))
    wrong_order = (("y", mirrored[1]), ("x", mirrored[0]), ("z", mirrored[2]))
    with pytest.raises(ValueError):
        map_suite_views(duplicate_axis, _labelled(*mirrored))
    with pytest.raises(ValueError):
        map_suite_views(_labelled(*base), wrong_order)
    with pytest.raises(ValueError):
        map_suite_views(_labelled(*base), _labelled(mirrored[0][:-1], mirrored[1], mirrored[2]))
    with pytest.raises(ValueError):
        map_suite_views(_labelled(*base), _labelled(mirrored[0].float(), mirrored[1], mirrored[2]))
    with pytest.raises(ValueError):
        map_suite_views(_labelled(*base), (("x", mirrored[0]), ("y", mirrored[1]), ("z", [[False]])))
    if torch.cuda.is_available():
        with pytest.raises(ValueError):
            map_suite_views(
                _labelled(*base),
                _labelled(mirrored[0].cuda(), mirrored[1], mirrored[2]),
            )


def test_asymmetric_toy_hull_uses_native_xyz_extrusion_and_six_neighbor_orientation() -> None:
    # Fixed native mapping is view(y,x,z)->axis(z,x,y), D4 ops (3,0,7).
    # Construct input masks by applying those forward ops to desired canonical
    # planes.  Their intersection contains (X,Y,Z)=(0,1,2),(1,1,2), two
    # 6-neighbors along native X only.
    plane_x = torch.zeros((4, 4), dtype=torch.bool)  # [Y,Z]
    plane_y = torch.zeros((4, 4), dtype=torch.bool)  # [X,Z]
    plane_z = torch.zeros((4, 4), dtype=torch.bool)  # [X,Y]
    plane_x[1, 2] = True
    plane_y[0, 2] = True
    plane_y[1, 2] = True
    plane_z[0, 1] = True
    plane_z[1, 1] = True
    mask_x = plane_x.clone()  # canonical view 1, op 0
    mask_y = torch.rot90(plane_z, 3, dims=(-2, -1))  # canonical view 0, op 3
    mask_z = torch.flip(torch.rot90(plane_y, 3, dims=(-2, -1)), dims=(-1,))  # op 7

    logits = build_fixed_visual_hull_logits(
        _labelled(mask_x, mask_y, mask_z),
        resolution=4,
        allow_test_size=True,
        dtype=torch.float64,
    )
    hull = logits[0, 0] == 2.0

    expected = torch.zeros((4, 4, 4), dtype=torch.bool)
    expected[0, 1, 2] = True
    expected[1, 1, 2] = True
    assert torch.equal(hull, expected)
    assert int(torch.count_nonzero(hull[1:] & hull[:-1]).item()) == 1
    assert int(torch.count_nonzero(hull[:, 1:] & hull[:, :-1]).item()) == 0
    assert int(torch.count_nonzero(hull[:, :, 1:] & hull[:, :, :-1]).item()) == 0


def test_hull_logits_are_exact_finite_nonaliased_and_threshold_to_the_hull() -> None:
    plane_x = torch.zeros((5, 5), dtype=torch.bool)
    plane_y = torch.zeros((5, 5), dtype=torch.bool)
    plane_z = torch.zeros((5, 5), dtype=torch.bool)
    plane_x[2, 4] = True
    plane_y[1, 4] = True
    plane_z[1, 2] = True
    mask_x = plane_x.clone()
    mask_y = torch.rot90(plane_z, 3, dims=(-2, -1))
    mask_z = torch.flip(torch.rot90(plane_y, 3, dims=(-2, -1)), dims=(-1,))
    labelled = _labelled(mask_x, mask_y, mask_z)

    first = build_fixed_visual_hull_logits(
        labelled, resolution=5, allow_test_size=True, dtype=torch.float32
    )
    second = build_fixed_visual_hull_logits(
        labelled, resolution=5, allow_test_size=True, dtype=torch.float32
    )
    expected_hull = torch.zeros((1, 1, 5, 5, 5), dtype=torch.bool)
    expected_hull[0, 0, 1, 2, 4] = True
    density = torch.sigmoid(first)

    assert first.shape == (1, 1, 5, 5, 5)
    assert first.dtype == torch.float32 and first.device.type == "cpu"
    assert torch.isfinite(first).all().item()
    assert set(first.unique().tolist()) == {-3.0, 2.0}
    assert torch.equal(first == 2.0, expected_hull)
    assert torch.equal(density >= 0.05, expected_hull)
    low = torch.nextafter(first.new_tensor(0.05), first.new_tensor(-math.inf))
    high = torch.nextafter(first.new_tensor(0.05), first.new_tensor(math.inf))
    assert torch.equal(density >= low, expected_hull)
    assert torch.equal(density >= high, expected_hull)
    assert _tensor_sha256(first) == _tensor_sha256(second)
    assert first.data_ptr() != second.data_ptr()


def test_native_256_initialization_is_identical_across_all_four_arms() -> None:
    plane_x = torch.zeros((256, 256), dtype=torch.bool)
    plane_y = torch.zeros((256, 256), dtype=torch.bool)
    plane_z = torch.zeros((256, 256), dtype=torch.bool)
    plane_x[23, 199] = True
    plane_y[17, 199] = True
    plane_z[17, 23] = True
    x = plane_x.clone()
    y = torch.rot90(plane_z, 3, dims=(-2, -1))
    z = torch.flip(torch.rot90(plane_y, 3, dims=(-2, -1)), dims=(-1,))
    views = _labelled(x, y, z)

    snapshots = [build_fixed_visual_hull_logits(views) for _ in PROFILE_ORDER]
    hashes = [_tensor_sha256(snapshot) for snapshot in snapshots]

    assert all(snapshot.shape == (1, 1, 256, 256, 256) for snapshot in snapshots)
    assert len(set(hashes)) == 1
    assert len({snapshot.data_ptr() for snapshot in snapshots}) == 4
    for snapshot in snapshots:
        assert snapshot[0, 0, 17, 23, 199].item() == 2.0
        assert int(torch.count_nonzero(snapshot == 2.0).item()) == 1
        assert torch.equal(torch.sigmoid(snapshot) >= 0.05, snapshot == 2.0)


def test_hull_builder_rejects_nonproduction_size_without_explicit_test_permission() -> None:
    mask = torch.zeros((4, 4), dtype=torch.bool)
    with pytest.raises(ValueError):
        build_fixed_visual_hull_logits(_labelled(mask, mask, mask), resolution=4)
    with pytest.raises(ValueError):
        build_fixed_visual_hull_logits(
            _labelled(mask, mask, mask), resolution=True, allow_test_size=True
        )
    with pytest.raises(ValueError):
        build_fixed_visual_hull_logits(
            _labelled(mask, mask, mask), resolution=4, allow_test_size=True, dtype=torch.int64
        )
    with pytest.raises(ValueError):
        build_fixed_visual_hull_logits(
            _labelled(mask, mask, mask), resolution=5, allow_test_size=True
        )


def test_j_and_h_losses_are_exactly_equivalent_to_v5_on_ordinary_fixtures() -> None:
    score = torch.tensor(
        [
            [[-0.04, 0.01, -0.03], [0.02, -0.02, 0.00]],
            [[0.04, -0.05, -0.01], [-0.02, 0.03, -0.06]],
        ],
        dtype=torch.float64,
    )
    target = torch.tensor(
        [
            [[False, True, False], [True, False, False]],
            [[True, False, False], [False, True, False]],
        ]
    )

    assert torch.equal(delivery_lovasz_loss(score, target), v5_j_loss(score, target))
    actual_h, actual_details = background_hard_negative_tail_loss(
        score, target, return_details=True
    )
    expected_h, expected_details = v5_h_loss(score, target, return_details=True)
    assert torch.equal(actual_h, expected_h)
    assert actual_details == expected_details


def test_h_is_background_only_and_stable_ties_use_row_major_order() -> None:
    score = torch.zeros((1, 4, 5), dtype=torch.float64, requires_grad=True)
    target = torch.zeros_like(score, dtype=torch.bool)
    target[0, 0, 0] = True
    target[0, 3, 4] = True

    loss, details = background_hard_negative_tail_loss(
        score, target, fraction=0.10, return_details=True
    )
    loss.backward()

    # 18 background pixels -> ceil(.10 * 18) = 2. Stable ties choose flat 1,2.
    assert details[0]["selected_indices"] == [1, 2]
    assert score.grad is not None
    assert score.grad[0, 0, 0].item() == 0.0
    assert score.grad[0, 3, 4].item() == 0.0
    assert torch.count_nonzero(score.grad).item() == 2


def test_h_and_j_fail_closed_on_invalid_or_nonfinite_inputs_and_h_handles_extremes() -> None:
    extreme = torch.tensor(
        [[[-1000.0, 1000.0], [0.0, -1.0]]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([[[False, False], [True, False]]])
    h_value = background_hard_negative_tail_loss(extreme, target)
    j_value = delivery_lovasz_loss(extreme, target)
    assert torch.isfinite(h_value).item() and torch.isfinite(j_value).item()
    (h_value + j_value).backward()
    assert extreme.grad is not None and torch.isfinite(extreme.grad).all().item()
    assert extreme.grad[0, 1, 0].item() != 0.0  # J may use foreground.

    bad = extreme.detach().clone()
    bad[0, 0, 0] = math.nan
    with pytest.raises(ValueError):
        delivery_lovasz_loss(bad, target)
    with pytest.raises(ValueError):
        background_hard_negative_tail_loss(bad, target)
    with pytest.raises(ValueError):
        background_hard_negative_tail_loss(extreme.detach(), torch.ones_like(target))
    for kwargs in (
        {"fraction": 0.0},
        {"fraction": math.nextafter(1.0, math.inf)},
        {"temperature": 0.0},
        {"margin": math.inf},
    ):
        with pytest.raises(ValueError):
            background_hard_negative_tail_loss(extreme.detach(), target, **kwargs)


def test_auxiliary_multiplier_uses_completed_update_boundaries() -> None:
    expected = {
        0: 0.0,
        599: 0.0,
        600: 1.0 / 300.0,
        601: 2.0 / 300.0,
        749: 0.5,
        899: 1.0,
        900: 1.0,
        2999: 1.0,
    }
    for completed, value in expected.items():
        assert v6_auxiliary_multiplier(completed) == value
    for invalid in (True, 1.0, math.nextafter(600.0, math.inf), -1, 3000):
        with pytest.raises(ValueError):
            v6_auxiliary_multiplier(invalid)
    assert dict(v6_component_budgets(599, "full")) == {"j": 0.0, "h": 0.0}
    assert dict(v6_component_budgets(600, "full")) == {
        "j": 0.20 / 300.0,
        "h": 0.10 / 300.0,
    }
    assert dict(v6_component_budgets(900, "base")) == {"j": 0.0, "h": 0.0}
    assert dict(v6_component_budgets(900, "j_only")) == {"j": 0.20, "h": 0.0}
    assert dict(v6_component_budgets(900, "h_only")) == {"j": 0.0, "h": 0.10}
    assert dict(v6_component_budgets(900, "full")) == {"j": 0.20, "h": 0.10}
    with pytest.raises(ValueError):
        v6_component_budgets(900, "unknown")


def test_learning_rate_is_for_the_next_update_with_exact_frozen_endpoints() -> None:
    assert v6_learning_rate(0) == 0.007
    assert v6_learning_rate(1499) == 0.007
    assert v6_learning_rate(1500) == 0.007
    expected_second_anneal = 0.0007 + 0.5 * (0.007 - 0.0007) * (
        1.0 + math.cos(math.pi / 1199.0)
    )
    assert v6_learning_rate(1501) == expected_second_anneal
    assert math.nextafter(v6_learning_rate(1501), math.inf) <= 0.007
    assert v6_learning_rate(2699) == 0.0007
    assert v6_learning_rate(2700) == 0.0007
    assert v6_learning_rate(2999) == 0.0007
    for invalid in (True, 1.0, math.nextafter(1500.0, math.inf), -1, 3000):
        with pytest.raises(ValueError):
            v6_learning_rate(invalid)


def test_gradient_components_are_identical_in_single_and_full_profiles() -> None:
    base = torch.tensor([3.0, 4.0], dtype=torch.float64)
    j = torch.tensor([7.0, 0.0], dtype=torch.float64)
    h = torch.tensor([0.0, 11.0], dtype=torch.float64)
    base_bytes = _tensor_sha256(base)
    j_bytes = _tensor_sha256(j)
    h_bytes = _tensor_sha256(h)

    j_only = compose_v6_gradient(base, j, None, completed_updates=900, profile="j_only")
    h_only = compose_v6_gradient(base, None, h, completed_updates=900, profile="h_only")
    full, telemetry = compose_v6_gradient(
        base, j, h, completed_updates=900, profile="full", return_telemetry=True
    )

    assert torch.equal(j_only - base, full - h_only)
    assert torch.equal(h_only - base, full - j_only)
    assert j_only.tolist() == [4.0, 4.0]
    assert h_only.tolist() == [3.0, 4.5]
    assert full.tolist() == [4.0, 4.5]
    assert telemetry["j_ratio"] == 0.20
    assert telemetry["h_ratio"] == 0.10
    assert telemetry["combined_ratio"] <= math.nextafter(0.30, math.inf)
    assert _tensor_sha256(base) == base_bytes
    assert _tensor_sha256(j) == j_bytes
    assert _tensor_sha256(h) == h_bytes
    assert full.data_ptr() != base.data_ptr()


def test_gradient_composition_constructs_no_disabled_or_off_schedule_branch() -> None:
    base = torch.tensor([3.0, 4.0])
    aux = torch.tensor([1.0, 0.0])
    result = compose_v6_gradient(base, None, None, completed_updates=599, profile="full")
    assert torch.equal(result, base)
    assert result.data_ptr() != base.data_ptr()
    with pytest.raises(ValueError, match="off-schedule"):
        compose_v6_gradient(base, aux, None, completed_updates=599, profile="full")
    with pytest.raises(ValueError, match="disabled h"):
        compose_v6_gradient(base, aux, aux, completed_updates=900, profile="j_only")
    with pytest.raises(ValueError, match="disabled j"):
        compose_v6_gradient(base, aux, None, completed_updates=900, profile="base")


def test_gradient_composition_zero_norm_and_finite_extremes_fail_closed() -> None:
    base = torch.tensor([3.0, 4.0], dtype=torch.float32)
    zero = torch.zeros_like(base)
    result, telemetry = compose_v6_gradient(
        base, zero, None, completed_updates=900, profile="j_only", return_telemetry=True
    )
    assert torch.equal(result, base)
    assert telemetry["j_zero"] is True
    assert telemetry["j_ratio"] == 0.0
    assert all(
        math.isfinite(value)
        for value in telemetry.values()
        if type(value) in (int, float)
    )

    with pytest.raises(ValueError):
        compose_v6_gradient(torch.zeros(2), zero, None, completed_updates=900, profile="j_only")
    with pytest.raises(ValueError):
        compose_v6_gradient(
            torch.tensor([torch.finfo(torch.float32).max]),
            torch.ones(1),
            None,
            completed_updates=900,
            profile="j_only",
        )
    with pytest.raises(ValueError):
        compose_v6_gradient(base, torch.tensor([math.inf, 0.0]), None, completed_updates=900, profile="j_only")
    with pytest.raises(ValueError):
        compose_v6_gradient(base, torch.ones(3), None, completed_updates=900, profile="j_only")
    with pytest.raises(ValueError):
        compose_v6_gradient(base, torch.ones(2, dtype=torch.float64), None, completed_updates=900, profile="j_only")


def test_float32_random_components_respect_explicit_nextafter_ratio_bounds() -> None:
    generator = torch.Generator().manual_seed(17)
    base = torch.randn(4096, generator=generator, dtype=torch.float32)
    j = torch.randn(4096, generator=generator, dtype=torch.float32)
    h = torch.randn(4096, generator=generator, dtype=torch.float32)

    _, telemetry = compose_v6_gradient(
        base, j, h, completed_updates=900, profile="full", return_telemetry=True
    )
    upper_j = float(torch.nextafter(torch.tensor(0.20), torch.tensor(math.inf)).item())
    upper_h = float(torch.nextafter(torch.tensor(0.10), torch.tensor(math.inf)).item())
    upper_combined = float(torch.nextafter(torch.tensor(0.30), torch.tensor(math.inf)).item())
    assert telemetry["j_ratio"] <= upper_j
    assert telemetry["h_ratio"] <= upper_h
    assert telemetry["combined_ratio"] <= upper_combined


def test_public_exports_and_signatures_are_frozen() -> None:
    assert tuple(inspect.signature(build_fixed_visual_hull_logits).parameters) == (
        "suite_views",
        "resolution",
        "allow_test_size",
        "device",
        "dtype",
    )
    assert tuple(inspect.signature(v6_learning_rate).parameters) == ("completed_updates",)
    assert tuple(inspect.signature(compose_v6_gradient).parameters) == (
        "base_grad",
        "j_grad",
        "h_grad",
        "completed_updates",
        "profile",
        "return_telemetry",
    )
    import shadowforge_pipeline.soft_voxel_v6 as public

    expected = {
        "CANONICAL_VIEW_DESCRIPTORS",
        "FIXED_CAMERA_AXES",
        "PROFILE_FLAGS",
        "PROFILE_ORDER",
        "TASK_ORDER",
        "background_hard_negative_tail_loss",
        "build_fixed_visual_hull_logits",
        "compose_v6_gradient",
        "delivery_lovasz_loss",
        "fixed_camera_payload",
        "map_suite_views",
        "v6_auxiliary_multiplier",
        "v6_component_budgets",
        "v6_learning_rate",
    }
    assert set(public.__all__) == expected
    assert all(hasattr(public, name) for name in expected)


def test_protected_v3_v4_v5_baseline_inventory_remains_exact() -> None:
    rows = [line.split(maxsplit=1) for line in BASELINE_INVENTORY.read_text().splitlines() if line]
    assert len(rows) == 59
    for expected_sha256, relative_path in rows:
        actual = hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected_sha256, relative_path
