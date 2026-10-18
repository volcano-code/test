import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

import shadowforge_pipeline.soft_voxel_v4.gradients as v4_gradients
from shadowforge_pipeline.soft_voxel_v4 import (
    AuxiliarySlot,
    GuardedAuxiliaryGradient,
    V4Profile,
    auxiliary_slot,
    guard_auxiliary_gradient,
    resolve_profile,
)


def _gradients_source() -> str:
    return Path(v4_gradients.__file__).read_text(encoding="utf-8")


def _shared_storage_alias_call():
    shared = torch.arange(4.0)
    return guard_auxiliary_gradient(shared[:2], shared[1:3], torch.ones(2), 1.0)


def _assert_terminal_guard_contract(
    base: torch.Tensor,
    anchor: torch.Tensor,
    aux: torch.Tensor,
    ratio: float,
) -> None:
    try:
        guarded = guard_auxiliary_gradient(base, anchor, aux, max_ratio=ratio)
    except ValueError:
        return

    final = guarded.gradient
    final_dot = torch.dot(final.reshape(-1), anchor.reshape(-1)).item()
    final_norm = torch.linalg.vector_norm(final).item()
    budget = ratio * torch.linalg.vector_norm(base).item()

    assert torch.isfinite(final).all().item()
    assert math.isfinite(guarded.raw_norm)
    assert math.isfinite(guarded.projected_norm)
    assert math.isfinite(guarded.final_norm)
    assert math.isfinite(guarded.anchor_cosine_before)
    assert math.isfinite(guarded.anchor_cosine_after)
    assert math.isfinite(guarded.removed_conflict_fraction)
    assert math.isfinite(guarded.auxiliary_base_ratio)
    assert final_dot >= 0.0
    assert final_norm <= budget


def _review_round_5_seeded_witness(
    target_dtype: torch.dtype,
    target_dim: int,
    target_scale: float,
    target_ratio: float,
):
    """Replay the independent review sweep exactly up to one failing case."""
    generator = torch.Generator().manual_seed(20260827)
    for dtype in (torch.float32, torch.float64):
        scales = (1e-2, 1.0, 1e3, 1e10) if dtype == torch.float32 else (1e-2, 1.0, 1e3, 1e100)
        for dim in (2, 7, 23, 64):
            for scale in scales:
                for ratio in (0.0, 1e-6, 0.1, 1.0):
                    vectors = []
                    for _ in range(3):
                        vector = torch.randn(dim, generator=generator, dtype=dtype)
                        vectors.append(vector / torch.linalg.vector_norm(vector) * scale)
                    if (dtype, dim, scale, ratio) == (
                        target_dtype,
                        target_dim,
                        target_scale,
                        target_ratio,
                    ):
                        return (*vectors, ratio)
    raise AssertionError("requested seeded witness is not part of the review sweep")


def _review_round_5_intermediate_budget_witness(
    dtype: torch.dtype,
    dim: int,
    scale: float,
    seed: int,
):
    generator = torch.Generator().manual_seed(900000 + seed + dim)
    vectors = []
    for _ in range(3):
        vector = torch.randn(dim, generator=generator, dtype=dtype)
        vectors.append(vector / torch.linalg.vector_norm(vector) * scale)
    return vectors


@pytest.mark.parametrize(
    ("profile_name", "expected"),
    [
        ("l12_bal", [None, None, None, None, None, None, None, None]),
        (
            "dic_only",
            [
                ("dic", None),
                None,
                ("dic", None),
                None,
                ("dic", None),
                None,
                ("dic", None),
                None,
            ],
        ),
        (
            "dts_only",
            [
                None,
                ("dts", 0.04),
                None,
                ("dts", 0.06),
                None,
                ("dts", 0.04),
                None,
                ("dts", 0.06),
            ],
        ),
        (
            "v4_full",
            [
                ("dic", None),
                ("dts", 0.04),
                ("dic", None),
                ("dts", 0.06),
                ("dic", None),
                ("dts", 0.04),
                ("dic", None),
                ("dts", 0.06),
            ],
        ),
    ],
)
def test_auxiliary_slot_uses_fixed_two_cycle_table_without_reallocation(profile_name, expected):
    profile = resolve_profile(profile_name)

    observed = []
    for step in range(8):
        slot = auxiliary_slot(step, 0.90, profile)
        observed.append(None if slot is None else (slot.component, slot.endpoint))

    assert observed == expected


def test_auxiliary_slot_uses_clamped_multiplier_only_and_returns_none_when_zero():
    profile = resolve_profile("v4_full")
    below = math.nextafter(0.60, 0.0)
    above_start = math.nextafter(0.60, 1.0)
    below_full = math.nextafter(0.75, 0.0)
    above_full = math.nextafter(0.75, 1.0)

    assert auxiliary_slot(0, -10.0, profile) is None
    assert auxiliary_slot(0, 0.60, profile) is None

    rising = auxiliary_slot(1, above_start, profile)
    assert rising is not None
    assert rising.multiplier == pytest.approx((above_start - 0.60) / 0.15, rel=0.0, abs=0.0)

    almost_full = auxiliary_slot(2, below_full, profile)
    assert almost_full is not None
    assert almost_full.multiplier == pytest.approx((below_full - 0.60) / 0.15, rel=0.0, abs=0.0)

    full = auxiliary_slot(3, 0.75, profile)
    assert full is not None
    assert full.multiplier == 1.0
    above = auxiliary_slot(0, above_full, profile)
    assert above is not None
    assert above.multiplier == 1.0


def test_auxiliary_slot_records_are_immutable_and_exact_type_checked():
    profile = resolve_profile("v4_full")
    slot = auxiliary_slot(0, 1.0, profile)

    assert isinstance(slot, AuxiliarySlot)
    with pytest.raises(FrozenInstanceError):
        slot.multiplier = 0.0
    with pytest.raises(ValueError, match="nonnegative integer"):
        auxiliary_slot(True, 1.0, profile)
    with pytest.raises(ValueError, match="nonnegative integer"):
        auxiliary_slot(-1, 1.0, profile)
    with pytest.raises(ValueError, match="nonnegative integer"):
        auxiliary_slot(1.5, 1.0, profile)
    with pytest.raises(ValueError, match="finite"):
        auxiliary_slot(0, float("nan"), profile)
    with pytest.raises(ValueError, match="boolean"):
        auxiliary_slot(0, False, profile)
    with pytest.raises(ValueError, match="V4Profile"):
        auxiliary_slot(0, 1.0, object())
    with pytest.raises(ValueError, match="real number"):
        auxiliary_slot(0, "0.9", profile)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_guard_auxiliary_gradient_handles_conflicting_vectors_exactly(dtype, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    base = torch.tensor([3.0, 4.0], dtype=dtype, device=device)
    anchor = torch.tensor([1.0, 0.0], dtype=dtype, device=device)
    aux = torch.tensor([-2.0, 1.0], dtype=dtype, device=device)
    base_before = base.clone()
    anchor_before = anchor.clone()
    aux_before = aux.clone()

    guarded = guard_auxiliary_gradient(base, anchor, aux, max_ratio=10.0)

    expected = torch.tensor([0.0, 1.0], dtype=dtype, device=device)
    assert isinstance(guarded, GuardedAuxiliaryGradient)
    assert guarded.gradient.data_ptr() != aux.data_ptr()
    assert torch.allclose(guarded.gradient, expected, rtol=0.0, atol=0.0)
    assert guarded.raw_norm == pytest.approx(math.sqrt(5.0), rel=0.0, abs=1e-6 if dtype == torch.float32 else 1e-12)
    assert guarded.projected_norm == pytest.approx(1.0, rel=0.0, abs=1e-6 if dtype == torch.float32 else 1e-12)
    assert guarded.final_norm == pytest.approx(1.0, rel=0.0, abs=1e-6 if dtype == torch.float32 else 1e-12)
    assert guarded.anchor_cosine_before == pytest.approx(-2.0 / math.sqrt(5.0), abs=1e-6 if dtype == torch.float32 else 1e-12)
    assert guarded.anchor_cosine_after == pytest.approx(0.0, abs=1e-7 if dtype == torch.float32 else 1e-14)
    assert guarded.removed_conflict_fraction == pytest.approx((math.sqrt(5.0) - 1.0) / math.sqrt(5.0), abs=1e-6 if dtype == torch.float32 else 1e-12)
    assert guarded.auxiliary_base_ratio == pytest.approx(0.2, abs=1e-6 if dtype == torch.float32 else 1e-12)
    assert guarded.cap_applied is False
    assert guarded.fully_projected_zero is False
    assert torch.dot(guarded.gradient, anchor).item() >= 0.0
    assert torch.equal(base, base_before)
    assert torch.equal(anchor, anchor_before)
    assert torch.equal(aux, aux_before)


def test_guard_auxiliary_gradient_before_cosine_normalizes_nonunit_anchor_once():
    base = torch.tensor([3.0, 4.0], dtype=torch.float64)
    anchor = torch.tensor([2.0, 0.0], dtype=torch.float64)
    aux = torch.tensor([-2.0, 1.0], dtype=torch.float64)

    guarded = guard_auxiliary_gradient(base, anchor, aux, max_ratio=10.0)

    assert guarded.anchor_cosine_before == pytest.approx(
        -2.0 / math.sqrt(5.0), rel=0.0, abs=1e-12
    )


def test_guard_auxiliary_gradient_after_cosine_uses_projected_vector_before_cap():
    base = torch.zeros(2, dtype=torch.float64)
    anchor = torch.tensor([2.0, 0.0], dtype=torch.float64)
    aux = torch.tensor([3.0, 0.0], dtype=torch.float64)

    guarded = guard_auxiliary_gradient(base, anchor, aux, max_ratio=0.0)

    assert guarded.anchor_cosine_after == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert guarded.projected_norm == pytest.approx(3.0, rel=0.0, abs=1e-12)
    assert guarded.final_norm == 0.0
    assert guarded.cap_applied is True
    assert guarded.fully_projected_zero is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_guard_auxiliary_gradient_leaves_orthogonal_and_aligned_vectors_unchanged(dtype):
    base = torch.tensor([3.0, 4.0], dtype=dtype)
    anchor = torch.tensor([1.0, 0.0], dtype=dtype)
    orthogonal = torch.tensor([0.0, 2.0], dtype=dtype)
    aligned = torch.tensor([2.0, 0.0], dtype=dtype)

    orthogonal_result = guard_auxiliary_gradient(base, anchor, orthogonal, max_ratio=10.0)
    aligned_result = guard_auxiliary_gradient(base, anchor, aligned, max_ratio=10.0)

    assert torch.equal(orthogonal_result.gradient, orthogonal)
    assert orthogonal_result.anchor_cosine_before == pytest.approx(0.0, abs=1e-7 if dtype == torch.float32 else 1e-14)
    assert orthogonal_result.anchor_cosine_after == pytest.approx(0.0, abs=1e-7 if dtype == torch.float32 else 1e-14)
    assert orthogonal_result.removed_conflict_fraction == 0.0
    assert torch.equal(aligned_result.gradient, aligned)
    assert aligned_result.anchor_cosine_before == pytest.approx(1.0, abs=1e-7 if dtype == torch.float32 else 1e-14)
    assert aligned_result.anchor_cosine_after == pytest.approx(1.0, abs=1e-7 if dtype == torch.float32 else 1e-14)
    assert aligned_result.removed_conflict_fraction == 0.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_guard_auxiliary_gradient_handles_zero_anchor_and_zero_base_explicitly(dtype):
    anchor_zero = torch.zeros(2, dtype=dtype)
    base = torch.tensor([3.0, 4.0], dtype=dtype)
    aux = torch.tensor([-2.0, 1.0], dtype=dtype)

    no_anchor = guard_auxiliary_gradient(base, anchor_zero, aux, max_ratio=10.0)
    assert torch.equal(no_anchor.gradient, aux)
    assert no_anchor.anchor_cosine_before == 0.0
    assert no_anchor.anchor_cosine_after == 0.0
    assert no_anchor.cap_applied is False

    base_zero = torch.zeros(2, dtype=dtype)
    projected_zero = guard_auxiliary_gradient(base_zero, torch.tensor([1.0, 0.0], dtype=dtype), aux, max_ratio=10.0)
    assert torch.equal(projected_zero.gradient, torch.zeros_like(aux))
    assert projected_zero.projected_norm > 0.0
    assert projected_zero.final_norm == 0.0
    assert projected_zero.auxiliary_base_ratio == 0.0
    assert projected_zero.cap_applied is True
    assert projected_zero.fully_projected_zero is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_guard_auxiliary_gradient_fully_projects_collinear_conflict_to_zero(dtype):
    base = torch.tensor([3.0, 4.0], dtype=dtype)
    anchor = torch.tensor([1.0, 0.0], dtype=dtype)
    aux = torch.tensor([-2.0, 0.0], dtype=dtype)

    guarded = guard_auxiliary_gradient(base, anchor, aux, max_ratio=10.0)

    assert torch.equal(guarded.gradient, torch.zeros_like(aux))
    assert guarded.projected_norm == 0.0
    assert guarded.final_norm == 0.0
    assert guarded.anchor_cosine_after == 0.0
    assert guarded.removed_conflict_fraction == 1.0
    assert guarded.cap_applied is False
    assert guarded.fully_projected_zero is True


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_guard_auxiliary_gradient_caps_norm_without_exceeding_base_ratio(dtype):
    base = torch.tensor([3.0, 4.0], dtype=dtype)
    anchor = torch.tensor([1.0, 0.0], dtype=dtype)
    aux = torch.tensor([0.0, 10.0], dtype=dtype)

    guarded = guard_auxiliary_gradient(base, anchor, aux, max_ratio=0.5)

    expected = torch.tensor([0.0, 2.5], dtype=dtype)
    assert torch.allclose(guarded.gradient, expected, rtol=0.0, atol=1e-6 if dtype == torch.float32 else 1e-12)
    assert guarded.projected_norm == pytest.approx(10.0, abs=1e-6 if dtype == torch.float32 else 1e-12)
    assert guarded.final_norm == pytest.approx(2.5, abs=1e-6 if dtype == torch.float32 else 1e-12)
    assert guarded.auxiliary_base_ratio == pytest.approx(0.5, abs=1e-6 if dtype == torch.float32 else 1e-12)
    assert guarded.cap_applied is True
    assert guarded.final_norm <= 0.5 * torch.linalg.vector_norm(base).item() + (1e-6 if dtype == torch.float32 else 1e-12)


def test_guard_auxiliary_gradient_terminal_revalidates_float32_post_cap_roundoff_case():
    base = torch.tensor(
        [
            20529.0859375,
            -6321.244140625,
            -19186.408203125,
            897.289306640625,
            -7565.7373046875,
            22585.873046875,
            -15653.1923828125,
            -1862.5118408203125,
            6635.89013671875,
            32479.322265625,
            372.99951171875,
            -17362.251953125,
            20357.962890625,
            9420.2978515625,
            -19855.744140625,
            32609.3125,
            38422.078125,
            24688.37890625,
            32006.041015625,
            -15109.2470703125,
            18290.966796875,
            31741.2109375,
            14642.20703125,
        ],
        dtype=torch.float32,
    )
    anchor = torch.tensor(
        [
            -2216.998779296875,
            -31777.861328125,
            -5238.87939453125,
            8824.8466796875,
            32869.140625,
            37134.4765625,
            -1829.65771484375,
            29072.884765625,
            -19061.01953125,
            24275.384765625,
            3875.693359375,
            -38674.5859375,
            17485.013671875,
            -8052.0439453125,
            25852.515625,
            13382.33203125,
            -39701.1328125,
            -511.1001281738281,
            29255.359375,
            -20380.638671875,
            -13910.96484375,
            29483.642578125,
            -24586.16796875,
        ],
        dtype=torch.float32,
    )
    aux = torch.tensor(
        [
            5372.78564453125,
            -20802.033203125,
            37208.80078125,
            24128.28515625,
            -4140.79833984375,
            -33389.8671875,
            -14320.8037109375,
            631.9494018554688,
            34446.71875,
            -31112.8046875,
            4080.06103515625,
            16439.017578125,
            3775.545166015625,
            25026.58203125,
            3205.937255859375,
            36914.203125,
            8211.9423828125,
            6972.9306640625,
            -4378.00244140625,
            7662.9091796875,
            -9160.0458984375,
            6020.62255859375,
            -16686.451171875,
        ],
        dtype=torch.float32,
    )

    _assert_terminal_guard_contract(base, anchor, aux, ratio=0.0008850272498687788)


def test_guard_auxiliary_gradient_terminal_revalidates_simple_large_cap_case():
    base = torch.tensor([-5.123350620269775, 0.7072904109954834], dtype=torch.float32)
    anchor = torch.tensor([23.37151336669922, 1.7574986219406128], dtype=torch.float32)
    aux = torch.tensor([-8.959799766540527, -0.8650903701782227], dtype=torch.float32)

    _assert_terminal_guard_contract(base, anchor, aux, ratio=1.4056423062300447e-06)


@pytest.mark.parametrize(
    ("dtype", "dim", "scale"),
    [
        pytest.param(torch.float32, 23, 1e-2, id="float32-dim23-scale1e-2"),
        pytest.param(torch.float64, 64, 1e3, id="float64-dim64-scale1e3"),
    ],
)
def test_guard_auxiliary_gradient_repairs_seeded_ulp_residual_without_false_rejection(
    dtype,
    dim,
    scale,
):
    base, anchor, aux, ratio = _review_round_5_seeded_witness(dtype, dim, scale, 1.0)
    base_before = base.clone()
    anchor_before = anchor.clone()
    aux_before = aux.clone()

    guarded = guard_auxiliary_gradient(base, anchor, aux, max_ratio=ratio)

    final = guarded.gradient
    budget = ratio * torch.linalg.vector_norm(base).item()
    assert torch.isfinite(final).all().item()
    assert torch.dot(final.reshape(-1), anchor.reshape(-1)).item() >= 0.0
    assert torch.linalg.vector_norm(final).item() <= budget
    assert math.isfinite(guarded.raw_norm)
    assert math.isfinite(guarded.projected_norm)
    assert math.isfinite(guarded.final_norm)
    assert torch.equal(base, base_before)
    assert torch.equal(anchor, anchor_before)
    assert torch.equal(aux, aux_before)


@pytest.mark.parametrize(
    ("dtype", "dim", "scale", "seed", "ratio"),
    [
        pytest.param(torch.float32, 23, 1e-3, 18, 0.1, id="float32-dim23-seed18"),
        pytest.param(torch.float32, 64, 1.0, 32, 1e-6, id="float32-dim64-seed32"),
        pytest.param(torch.float64, 2, 1e-3, 37, 1e-6, id="float64-dim2-seed37"),
    ],
)
def test_guard_auxiliary_gradient_continues_ulp_repair_past_intermediate_norm_rounding(
    dtype,
    dim,
    scale,
    seed,
    ratio,
):
    base, anchor, aux = _review_round_5_intermediate_budget_witness(dtype, dim, scale, seed)
    base_before = base.clone()
    anchor_before = anchor.clone()
    aux_before = aux.clone()

    guarded = guard_auxiliary_gradient(base, anchor, aux, max_ratio=ratio)

    final = guarded.gradient
    budget = ratio * torch.linalg.vector_norm(base).item()
    assert torch.isfinite(final).all().item()
    assert torch.dot(final.reshape(-1), anchor.reshape(-1)).item() >= 0.0
    assert torch.linalg.vector_norm(final).item() <= budget
    assert math.isfinite(guarded.raw_norm)
    assert math.isfinite(guarded.projected_norm)
    assert math.isfinite(guarded.final_norm)
    assert torch.equal(base, base_before)
    assert torch.equal(anchor, anchor_before)
    assert torch.equal(aux, aux_before)


@pytest.mark.parametrize(
    ("base", "expected_norm", "rtol"),
    [
        pytest.param(
            torch.tensor([1e38, 1e38], dtype=torch.float32),
            1e38 * float(torch.linalg.vector_norm(torch.ones(2, dtype=torch.float32)).item()),
            1e-6,
            id="float32-2d",
        ),
        pytest.param(
            torch.tensor([1e308, 1e308], dtype=torch.float64),
            1e308 * float(torch.linalg.vector_norm(torch.ones(2, dtype=torch.float64)).item()),
            1e-12,
            id="float64-2d",
        ),
        pytest.param(
            torch.tensor([2e37] * 23, dtype=torch.float32),
            2e37 * float(torch.linalg.vector_norm(torch.ones(23, dtype=torch.float32)).item()),
            1e-6,
            id="float32-23d",
        ),
    ],
)
def test_guard_auxiliary_gradient_terminal_revalidation_accepts_large_representable_uncapped_vectors(
    base,
    expected_norm,
    rtol,
):
    anchor = torch.zeros_like(base)
    aux = base.clone()

    guarded = guard_auxiliary_gradient(base, anchor, aux, max_ratio=1.0)

    assert torch.equal(guarded.gradient, aux)
    assert guarded.raw_norm == pytest.approx(expected_norm, rel=rtol, abs=0.0)
    assert guarded.projected_norm == pytest.approx(expected_norm, rel=rtol, abs=0.0)
    assert guarded.final_norm == pytest.approx(expected_norm, rel=rtol, abs=0.0)
    assert guarded.anchor_cosine_before == 0.0
    assert guarded.anchor_cosine_after == 0.0
    assert guarded.removed_conflict_fraction == 0.0
    assert guarded.auxiliary_base_ratio == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert guarded.cap_applied is False
    assert guarded.fully_projected_zero is False
    assert torch.isfinite(guarded.gradient).all().item()
    assert math.isfinite(guarded.raw_norm)
    assert math.isfinite(guarded.projected_norm)
    assert math.isfinite(guarded.final_norm)
    assert torch.dot(guarded.gradient.reshape(-1), anchor.reshape(-1)).item() == 0.0


@pytest.mark.parametrize(("dtype", "value"), [(torch.float32, 1e38), (torch.float64, 1e308)])
def test_guard_auxiliary_gradient_extreme_finite_inputs_never_report_nonfinite_outputs(dtype, value):
    base = torch.tensor([value, 0.0], dtype=dtype)
    anchor = torch.tensor([value, 0.0], dtype=dtype)
    aux = torch.tensor([-value, 0.0], dtype=dtype)

    guarded = guard_auxiliary_gradient(base, anchor, aux, max_ratio=1.0)

    assert torch.equal(guarded.gradient, torch.zeros_like(aux))
    assert math.isfinite(guarded.raw_norm)
    assert math.isfinite(guarded.projected_norm)
    assert math.isfinite(guarded.final_norm)
    assert math.isfinite(guarded.anchor_cosine_before)
    assert math.isfinite(guarded.anchor_cosine_after)
    assert math.isfinite(guarded.removed_conflict_fraction)
    assert math.isfinite(guarded.auxiliary_base_ratio)
    assert guarded.fully_projected_zero is True


def test_guard_auxiliary_gradient_public_exports_and_scope_exclude_forbidden_v3_imports():
    import shadowforge_pipeline.soft_voxel_v4 as v4

    assert v4.GuardedAuxiliaryGradient is GuardedAuxiliaryGradient
    assert v4.guard_auxiliary_gradient is guard_auxiliary_gradient
    source = _gradients_source()
    assert "soft_voxel_v3.trainer" not in source
    assert "soft_voxel_v3.calibration" not in source
    assert "soft_voxel_v3.registry" not in source


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: guard_auxiliary_gradient([0.0, 0.0], torch.zeros(2), torch.zeros(2), 1.0), "torch.Tensor"),
        (lambda: guard_auxiliary_gradient(torch.zeros(0), torch.zeros(0), torch.zeros(0), 1.0), "non-empty"),
        (lambda: guard_auxiliary_gradient(torch.zeros(2, dtype=torch.int64), torch.zeros(2), torch.zeros(2), 1.0), "floating"),
        (lambda: guard_auxiliary_gradient(torch.tensor([float("nan"), 0.0]), torch.zeros(2), torch.zeros(2), 1.0), "finite"),
        (lambda: guard_auxiliary_gradient(torch.zeros(2), torch.zeros(3), torch.zeros(2), 1.0), "shape"),
        (lambda: guard_auxiliary_gradient(torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float32), 1.0), "dtype"),
        (lambda: guard_auxiliary_gradient(torch.zeros(2), torch.zeros(2), torch.zeros(2), True), "boolean"),
        (lambda: guard_auxiliary_gradient(torch.zeros(2), torch.zeros(2), torch.zeros(2), "1.0"), "real number"),
        (lambda: guard_auxiliary_gradient(torch.zeros(2), torch.zeros(2), torch.zeros(2), float("nan")), "finite"),
        (lambda: guard_auxiliary_gradient(torch.zeros(2), torch.zeros(2), torch.zeros(2), -0.1), ">= 0"),
        (lambda: guard_auxiliary_gradient(*(lambda base: (base, base, torch.ones(2), 1.0))(torch.zeros(2))), "alias"),
        (_shared_storage_alias_call, "alias"),
    ],
)
def test_guard_auxiliary_gradient_rejects_invalid_inputs(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_guard_auxiliary_gradient_rejects_cross_device_inputs():
    with pytest.raises(ValueError, match="device"):
        guard_auxiliary_gradient(
            torch.zeros(2, device="cuda"),
            torch.zeros(2, device="cpu"),
            torch.zeros(2, device="cuda"),
            1.0,
        )
