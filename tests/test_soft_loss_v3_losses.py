import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from shadowforge_pipeline.soft_voxel_v2 import balanced_lp_loss as v2_balanced_lp_loss
import shadowforge_pipeline.soft_voxel_v3.losses as v3_losses
from shadowforge_pipeline.soft_voxel_v3.contours import ContourBands
from shadowforge_pipeline.soft_voxel_v3.losses import (
    V3Profile,
    balanced_lp_loss,
    compute_baseline_branch,
    compute_dic_branch,
    compute_dts_branch,
    dic_margin_loss,
    dic_schedule,
    dts_endpoint_loss,
    dts_schedule,
    registered_profiles,
    resolve_profile,
)
from shadowforge_pipeline.soft_voxel_v3.lovasz import lovasz_hinge_per_view


def _bands(inner: torch.Tensor, outer: torch.Tensor) -> ContourBands:
    return ContourBands(inner=inner, outer=outer)


@pytest.mark.parametrize("p", [1, 2])
def test_balanced_lp_matches_v2_value_and_prediction_gradient_exactly(p):
    target = torch.tensor(
        [
            [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    prediction_v2 = torch.tensor(
        [
            [[0.2, 0.9, 0.7], [0.1, 0.4, 0.6]],
            [[0.8, 0.3, 0.2], [0.9, 0.5, 0.6]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    prediction_v3 = prediction_v2.detach().clone().requires_grad_(True)

    v2_loss = v2_balanced_lp_loss(prediction_v2, target, p=p)
    v3_loss = balanced_lp_loss(prediction_v3, target, p=p)
    v2_loss.sum().backward()
    v3_loss.sum().backward()

    assert torch.equal(v3_loss, v2_loss)
    assert torch.equal(prediction_v3.grad, prediction_v2.grad)


def test_balanced_lp_uses_equal_region_weights_without_proxy_normalization():
    prediction = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])

    # Foreground error is 1 and background mean error is 1/3.
    assert balanced_lp_loss(prediction, target, p=1).item() == pytest.approx(2.0 / 3.0)
    assert balanced_lp_loss(prediction, target, p=2).item() == pytest.approx(2.0 / 3.0)


def test_dic_pushes_inner_up_and_outer_down():
    inner = torch.zeros(1, 3, 3, dtype=torch.bool)
    outer = torch.zeros_like(inner)
    inner[0, 1, 1] = True
    outer[0, 0, 0] = True
    score = torch.zeros(1, 3, 3, dtype=torch.float64, requires_grad=True)

    dic_margin_loss(score, _bands(inner, outer)).mean().backward()

    assert (score.grad[inner] < 0).all()
    assert (score.grad[outer] > 0).all()


def test_dic_uses_independent_inner_and_outer_means_before_equal_mix():
    inner = torch.tensor([[[True, False], [False, False]]])
    outer = torch.tensor([[[False, True], [True, True]]])
    score = torch.tensor(
        [[[0.0, 0.05], [0.05, 0.05]]], dtype=torch.float64, requires_grad=True
    )

    loss = dic_margin_loss(score, _bands(inner, outer))

    inner_side = 0.05 * math.log(2.0)
    outer_side = 0.05 * math.log1p(math.e)
    expected = 0.5 * inner_side + 0.5 * outer_side
    pooled_wrong = (inner_side + 3.0 * outer_side) / 4.0
    assert loss.item() == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert loss.item() != pytest.approx(pooled_wrong, rel=0.0, abs=1e-6)


@pytest.mark.parametrize(
    ("push_up", "expected_derivative"), [(True, -0.5), (False, 0.5)]
)
def test_dic_side_zero_margin_value_and_raw_derivative_are_exact(
    push_up, expected_derivative
):
    score = torch.zeros(1, dtype=torch.float64, requires_grad=True)

    side_loss = v3_losses._dic_side_penalty(
        score, push_up=push_up, temperature=0.05
    )
    side_loss.sum().backward()

    assert side_loss.item() == pytest.approx(
        0.05 * math.log(2.0), rel=0.0, abs=1e-15
    )
    assert score.grad.item() == pytest.approx(expected_derivative, abs=1e-15)


@pytest.mark.parametrize("push_up", [True, False])
def test_dic_side_raw_derivative_magnitude_is_bounded_by_one(push_up):
    score = torch.tensor([-100.0, 0.0, 100.0], requires_grad=True)

    v3_losses._dic_side_penalty(
        score, push_up=push_up, temperature=0.05
    ).sum().backward()

    assert torch.isfinite(score.grad).all()
    assert score.grad.abs().max().item() <= 1.0


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("case", ["empty", "full", "tied"])
def test_lovasz_empty_full_and_tied_masks_have_finite_gradients(device_name, case):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    labels_by_case = {
        "empty": [[False, False], [False, False]],
        "full": [[True, True], [True, True]],
        "tied": [[True, False], [True, False]],
    }
    logits = torch.zeros((1, 2, 2), device=device, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([labels_by_case[case]], device=device, dtype=torch.bool)

    loss = lovasz_hinge_per_view(logits, labels)
    loss.sum().backward()

    assert loss.shape == (1,)
    assert loss.dtype == torch.float64
    assert loss.device == logits.device
    assert torch.isfinite(loss).all()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_lovasz_ties_preserve_row_major_order(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    logits = torch.zeros((1, 2, 2), device=device, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[[True, False], [True, False]]], device=device)

    loss = lovasz_hinge_per_view(logits, labels)
    loss.sum().backward()

    # Stable row-major labels [1,0,1,0] give Lovasz weights [1/2,1/6,1/3,0].
    expected_gradient = torch.tensor(
        [[[-0.5, 1.0 / 6.0], [-1.0 / 3.0, 0.0]]],
        device=device,
        dtype=torch.float64,
    )
    assert loss.item() == pytest.approx(1.0, abs=1e-15)
    assert torch.allclose(logits.grad, expected_gradient, rtol=0.0, atol=1e-15)


def test_dts_uses_score_over_temperature_and_no_endpoint_half_factor():
    score = torch.tensor(
        [[[0.0, 0.025], [0.0, -0.025]]], dtype=torch.float64, requires_grad=True
    )
    labels = torch.tensor([[[True, False], [True, False]]])

    loss = dts_endpoint_loss(score, labels, temperature=0.05)
    loss.sum().backward()

    # logits=[0,.5,0,-.5], errors=[1,1.5,1,.5], sorted labels=[0,1,1,0].
    assert loss.item() == pytest.approx(7.0 / 6.0, rel=0.0, abs=1e-15)
    expected_gradient = torch.tensor(
        [[[-20.0 / 3.0, 20.0 / 3.0], [-20.0 / 3.0, 0.0]]],
        dtype=torch.float64,
    )
    assert torch.allclose(score.grad, expected_gradient, rtol=0.0, atol=1e-14)


def test_dts_rejects_non_tensor_score_before_temperature_scaling():
    labels = torch.tensor([[[True, False]]])

    with pytest.raises(ValueError, match="score must be a torch.Tensor"):
        dts_endpoint_loss([[[0.0, 0.0]]], labels)


def test_dts_rejects_integral_score_before_temperature_scaling():
    score = torch.tensor([[[0, 0]]], dtype=torch.int64)
    labels = torch.tensor([[[True, False]]])

    with pytest.raises(ValueError, match="score must use a floating-point dtype"):
        dts_endpoint_loss(score, labels)


def test_profiles_are_exact_factorial_rows_and_frozen():
    assert registered_profiles() == (
        "l12_bal",
        "dic_only",
        "dts_only",
        "v3_full",
        "v3_zero_margin",
        "v3_no_l1",
        "v3_no_l2",
    )
    expected = {
        "l12_bal": (0.35, 1.0, False, False, None),
        "dic_only": (0.35, 1.0, True, False, None),
        "dts_only": (0.35, 1.0, False, True, (0.04, 0.06)),
        "v3_full": (0.35, 1.0, True, True, (0.04, 0.06)),
        "v3_zero_margin": (0.35, 1.0, True, True, (0.05, 0.05)),
        "v3_no_l1": (0.0, 1.0, True, True, (0.04, 0.06)),
        "v3_no_l2": (0.35, 0.0, True, True, (0.04, 0.06)),
    }
    for name, fields in expected.items():
        profile = resolve_profile(name)
        assert (
            profile.l1_weight,
            profile.l2_weight,
            profile.use_dic,
            profile.use_dts,
            profile.dts_thresholds,
        ) == fields
        assert profile.name == name
    with pytest.raises(FrozenInstanceError):
        resolve_profile("v3_full").l1_weight = 99.0


def test_primary_four_profiles_form_full_dic_dts_factorial():
    observed = {
        (resolve_profile(name).use_dic, resolve_profile(name).use_dts)
        for name in ("l12_bal", "dic_only", "dts_only", "v3_full")
    }
    assert observed == {(False, False), (True, False), (False, True), (True, True)}


def test_schedules_use_only_registered_breakpoints():
    assert dts_schedule(-1.0) == 0.0
    assert dts_schedule(0.50) == 0.0
    assert dts_schedule(0.575) == pytest.approx(0.5, abs=1e-12)
    assert dts_schedule(0.65) == 1.0
    assert dts_schedule(2.0) == 1.0
    assert dic_schedule(-1.0) == 0.0
    assert dic_schedule(0.65) == 0.0
    assert dic_schedule(0.725) == pytest.approx(0.5, abs=1e-12)
    assert dic_schedule(0.80) == 1.0
    assert dic_schedule(2.0) == 1.0


def test_branch_boundaries_apply_fixed_baseline_weights_and_equal_view_means():
    projections = torch.tensor(
        [[[0.0, 1.0]], [[0.25, 0.75]]], dtype=torch.float64
    )
    targets = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=torch.float64)
    profile = resolve_profile("v3_full")

    baseline = compute_baseline_branch(projections, targets, profile)

    # Per-view numerators (L2 + .35*L1) are [1.35, .825]. Each one-pixel
    # region uses the frozen v2 denominator (1 + 1e-8).
    assert baseline.ndim == 0
    assert baseline.item() == pytest.approx(1.0875 / (1.0 + 1e-8), abs=1e-15)


def test_dic_and_dts_reduce_views_only_at_branch_boundary():
    score = torch.zeros((2, 1, 2), dtype=torch.float64)
    labels = torch.tensor([[[True, False]], [[True, False]]])
    inner = labels.clone()
    outer = ~labels

    assert compute_dic_branch(score, _bands(inner, outer)).ndim == 0
    assert compute_dic_branch(score, _bands(inner, outer)).item() == pytest.approx(
        0.05 * math.log(2.0), abs=1e-15
    )
    assert compute_dts_branch(score, labels).ndim == 0
    assert compute_dts_branch(score, labels).item() == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: balanced_lp_loss(torch.zeros(1, 2), torch.zeros(1, 3), p=1), "shape"),
        (lambda: balanced_lp_loss(torch.zeros(1, 2), torch.zeros(1, 2), p=3), "p"),
        (lambda: balanced_lp_loss(torch.tensor([[float("nan"), 0.0]]), torch.zeros(1, 2), p=1), "finite"),
        (lambda: balanced_lp_loss(torch.zeros(1, 2), torch.tensor([[0.0, float("inf")]]), p=1), "finite"),
        (lambda: dic_margin_loss(torch.zeros(1, 1, 2), _bands(torch.ones(1, 1, 2), torch.ones(1, 1, 2, dtype=torch.bool))), "boolean"),
        (lambda: dic_margin_loss(torch.zeros(1, 1, 2), _bands(torch.tensor([[[True, False]]]), torch.tensor([[[False, True]]])), 0.0), "positive"),
        (lambda: dic_margin_loss(torch.tensor([[[float("nan"), 0.0]]]), _bands(torch.tensor([[[True, False]]]), torch.tensor([[[False, True]]]))), "finite"),
        (lambda: dic_margin_loss(torch.zeros(1, 1, 3), _bands(torch.tensor([[[True, False]]]), torch.tensor([[[False, True]]]))), "shape"),
        (lambda: lovasz_hinge_per_view(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)), "boolean"),
        (lambda: lovasz_hinge_per_view(torch.zeros(1, 1, 2), torch.zeros(2, 1, 2, dtype=torch.bool)), "shape"),
        (lambda: lovasz_hinge_per_view(torch.tensor([[[float("inf"), 0.0]]]), torch.zeros(1, 1, 2, dtype=torch.bool)), "finite"),
        (lambda: dts_endpoint_loss(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2, dtype=torch.bool), temperature=-0.1), "positive"),
        (lambda: dts_endpoint_loss(torch.tensor([[[float("nan"), 0.0]]]), torch.zeros(1, 1, 2, dtype=torch.bool)), "finite"),
        (lambda: dts_endpoint_loss(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)), "boolean"),
        (lambda: resolve_profile("unknown"), "unknown"),
        (lambda: dts_schedule(float("nan")), "finite"),
        (lambda: dic_schedule(float("inf")), "finite"),
    ],
)
def test_loss_preflight_rejects_invalid_inputs(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kind", ["balanced", "dic", "lovasz", "dts"])
def test_loss_preflight_rejects_cross_device_inputs(kind):
    score = torch.zeros((1, 1, 2), device="cuda")
    labels = torch.tensor([[[True, False]]], device="cpu")
    bands = _bands(labels, ~labels)

    calls = {
        "balanced": lambda: balanced_lp_loss(score, labels.float(), p=1),
        "dic": lambda: dic_margin_loss(score, bands),
        "lovasz": lambda: lovasz_hinge_per_view(score, labels),
        "dts": lambda: dts_endpoint_loss(score, labels),
    }
    with pytest.raises(ValueError, match="device"):
        calls[kind]()


def test_public_loss_api_is_exported_from_v3_package():
    import shadowforge_pipeline.soft_voxel_v3 as v3

    assert v3.V3Profile is V3Profile
    assert v3.lovasz_hinge_per_view is lovasz_hinge_per_view
    assert v3.compute_dts_branch is compute_dts_branch
