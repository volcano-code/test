import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from shadowforge_pipeline.soft_voxel_v3.contours import ContourBands
from shadowforge_pipeline.soft_voxel_v3.losses import balanced_lp_loss as v3_balanced_lp_loss
import shadowforge_pipeline.soft_voxel_v4.losses as v4_losses
from shadowforge_pipeline.soft_voxel_v4 import (
    AuxiliarySlot,
    DirectionalDICResult,
    V4Profile,
    asymmetric_dts_loss,
    auxiliary_slot,
    compute_baseline_branch,
    directional_dic_loss,
    directional_dic_terms,
    registered_profiles,
    resolve_profile,
)


def _bands(inner: torch.Tensor, outer: torch.Tensor) -> ContourBands:
    return ContourBands(inner=inner, outer=outer)


def _v4_loss_source() -> str:
    return Path(v4_losses.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize("p", [1, 2])
def test_v4_baseline_balanced_lp_matches_v3_value_and_prediction_gradient_exactly(p):
    target = torch.tensor(
        [
            [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    prediction_v3 = torch.tensor(
        [
            [[0.2, 0.9, 0.7], [0.1, 0.4, 0.6]],
            [[0.8, 0.3, 0.2], [0.9, 0.5, 0.6]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    prediction_v4 = prediction_v3.detach().clone().requires_grad_(True)

    v3_loss = v3_balanced_lp_loss(prediction_v3, target, p=p)
    v4_loss = v4_losses.balanced_lp_loss(prediction_v4, target, p=p)
    v3_loss.sum().backward()
    v4_loss.sum().backward()

    assert torch.equal(v4_loss, v3_loss)
    assert torch.equal(prediction_v4.grad, prediction_v3.grad)


def test_v4_profiles_are_exact_factorial_rows_and_frozen():
    assert registered_profiles() == (
        "l12_bal",
        "dic_only",
        "dts_only",
        "v4_full",
        "v4_no_l1",
        "v4_no_l2",
    )
    expected = {
        "l12_bal": (0.35, 1.0, False, False),
        "dic_only": (0.35, 1.0, True, False),
        "dts_only": (0.35, 1.0, False, True),
        "v4_full": (0.35, 1.0, True, True),
        "v4_no_l1": (0.0, 1.0, True, True),
        "v4_no_l2": (0.35, 0.0, True, True),
    }
    for name, fields in expected.items():
        profile = resolve_profile(name)
        assert isinstance(profile, V4Profile)
        assert (
            profile.l1_weight,
            profile.l2_weight,
            profile.use_dic,
            profile.use_dts,
        ) == fields
        assert profile.name == name
    with pytest.raises(FrozenInstanceError):
        resolve_profile("v4_full").l1_weight = 0.0
    with pytest.raises(ValueError, match="unknown"):
        resolve_profile("v4")


def test_compute_baseline_branch_applies_exact_fixed_weights_and_branch_mean():
    projections = torch.tensor(
        [[[0.0, 1.0]], [[0.25, 0.75]]], dtype=torch.float64
    )
    targets = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=torch.float64)

    baseline = compute_baseline_branch(projections, targets, resolve_profile("v4_full"))

    assert baseline.ndim == 0
    assert baseline.item() == pytest.approx(1.0875 / (1.0 + 1e-8), abs=1e-15)


def test_directional_dic_pushes_inner_up_and_outer_down():
    inner = torch.zeros(1, 3, 3, dtype=torch.bool)
    outer = torch.zeros_like(inner)
    inner[0, 1, 1] = True
    outer[0, 0, 0] = True
    score = torch.zeros(1, 3, 3, dtype=torch.float64, requires_grad=True)

    directional_dic_loss(score, _bands(inner, outer)).mean().backward()

    assert (score.grad[inner] < 0).all()
    assert (score.grad[outer] > 0).all()


def test_directional_dic_uses_independent_normalization_before_equal_zero_error_mix():
    inner = torch.tensor([[[True, False], [False, False]]])
    outer = torch.tensor([[[False, True], [True, True]]])
    score = torch.tensor(
        [[[0.05, -0.05], [-0.10, -0.05]]], dtype=torch.float64, requires_grad=True
    )

    terms = directional_dic_terms(score, _bands(inner, outer))

    inner_side = 0.05 * math.log1p(math.exp(-1.0))
    outer_side = (
        0.05 * math.log1p(math.exp(-1.0))
        + 0.05 * math.log1p(math.exp(-2.0))
        + 0.05 * math.log1p(math.exp(-1.0))
    ) / 3.0
    expected = 0.5 * inner_side + 0.5 * outer_side
    pooled_wrong = (
        inner_side
        + 0.05 * math.log1p(math.exp(-1.0))
        + 0.05 * math.log1p(math.exp(-2.0))
        + 0.05 * math.log1p(math.exp(-1.0))
    ) / 4.0
    assert terms.per_view_loss.item() == pytest.approx(expected, rel=0.0, abs=1e-15)
    assert terms.per_view_loss.item() != pytest.approx(pooled_wrong, rel=0.0, abs=1e-6)
    assert torch.equal(terms.weight_false_negative, torch.full((1,), 0.5, dtype=torch.float64))
    assert torch.equal(terms.weight_false_positive, torch.full((1,), 0.5, dtype=torch.float64))


def test_directional_dic_returns_detached_zero_error_half_weights_exactly():
    inner = torch.tensor([[[True, False], [False, False]]])
    outer = torch.tensor([[[False, True], [True, True]]])
    score = torch.tensor([[[0.2, -0.2], [-0.4, -0.1]]], dtype=torch.float64, requires_grad=True)

    terms = directional_dic_terms(score, _bands(inner, outer))
    terms.per_view_loss.sum().backward()

    assert isinstance(terms, DirectionalDICResult)
    assert terms.false_negative_rate.requires_grad is False
    assert terms.false_positive_rate.requires_grad is False
    assert terms.weight_false_negative.requires_grad is False
    assert terms.weight_false_positive.requires_grad is False
    assert torch.equal(terms.false_negative_rate, torch.zeros(1, dtype=torch.float64))
    assert torch.equal(terms.false_positive_rate, torch.zeros(1, dtype=torch.float64))
    assert torch.equal(terms.weight_false_negative, torch.full((1,), 0.5, dtype=torch.float64))
    assert torch.equal(terms.weight_false_positive, torch.full((1,), 0.5, dtype=torch.float64))
    assert score.grad is not None


def test_directional_dic_clips_weights_at_both_extrema_and_zero_score_counts_as_false_positive():
    inner = torch.tensor(
        [
            [[True, False], [False, False]],
            [[True, True], [False, False]],
        ]
    )
    outer = torch.tensor(
        [
            [[False, True], [True, True]],
            [[False, False], [True, False]],
        ]
    )
    score = torch.tensor(
        [
            [[0.2, 0.0], [0.4, 0.1]],
            [[-0.2, -0.3], [-0.4, -0.1]],
        ],
        dtype=torch.float64,
    )

    terms = directional_dic_terms(score, _bands(inner, outer))

    assert terms.false_negative_rate.tolist() == pytest.approx([0.0, 1.0], abs=0.0)
    assert terms.false_positive_rate.tolist() == pytest.approx([1.0, 0.0], abs=0.0)
    assert terms.weight_false_positive.tolist() == pytest.approx([0.75, 0.25], abs=0.0)
    assert terms.weight_false_negative.tolist() == pytest.approx([0.25, 0.75], abs=0.0)


@pytest.mark.parametrize(("dtype", "value"), [(torch.float32, 1e38), (torch.float64, 1e308)])
def test_directional_dic_extreme_finite_scores_stay_finite_when_exact_loss_is_zero(dtype, value):
    inner = torch.tensor([[[True, False]]], dtype=torch.bool)
    outer = torch.tensor([[[False, True]]], dtype=torch.bool)
    score = torch.tensor([[[value, -value]]], dtype=dtype)

    terms = directional_dic_terms(score, _bands(inner, outer))

    assert torch.equal(terms.per_view_loss, torch.zeros(1, dtype=dtype))
    assert torch.isfinite(terms.per_view_loss).all().item()
    assert torch.isfinite(terms.false_negative_rate).all().item()
    assert torch.isfinite(terms.false_positive_rate).all().item()
    assert torch.isfinite(terms.weight_false_negative).all().item()
    assert torch.isfinite(terms.weight_false_positive).all().item()


def test_asymmetric_dts_background_endpoint_has_exact_gradient_and_no_half_factor():
    score = torch.tensor(
        [[[0.0, 0.05], [0.1, -0.05]]], dtype=torch.float64, requires_grad=True
    )
    labels = torch.tensor([[[True, False], [False, True]]], dtype=torch.bool)

    loss = asymmetric_dts_loss(score, labels, endpoint=0.04)
    loss.sum().backward()

    expected_value = 0.5 * (
        0.05 * math.log1p(math.exp(1.0)) + 0.05 * math.log1p(math.exp(2.0))
    )
    expected_gradient = torch.tensor(
        [[[0.0, 0.365529289315], [0.440398538989, 0.0]]], dtype=torch.float64
    )
    assert loss.item() == pytest.approx(expected_value, rel=0.0, abs=1e-15)
    assert torch.allclose(score.grad, expected_gradient, rtol=0.0, atol=1e-12)


def test_asymmetric_dts_foreground_endpoint_has_exact_gradient_and_no_half_factor():
    score = torch.tensor(
        [[[0.0, 0.05], [0.1, -0.05]]], dtype=torch.float64, requires_grad=True
    )
    labels = torch.tensor([[[True, False], [False, True]]], dtype=torch.bool)

    loss = asymmetric_dts_loss(score, labels, endpoint=0.06)
    loss.sum().backward()

    expected_value = 0.5 * (
        0.05 * math.log(2.0) + 0.05 * math.log1p(math.exp(1.0))
    )
    expected_gradient = torch.tensor(
        [[[-0.25, 0.0], [0.0, -0.365529289315]]], dtype=torch.float64
    )
    assert loss.item() == pytest.approx(expected_value, rel=0.0, abs=1e-15)
    assert torch.allclose(score.grad, expected_gradient, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("endpoint", [0.04, 0.06])
def test_asymmetric_dts_constructs_only_the_selected_branch(monkeypatch, endpoint):
    score = torch.tensor([[[0.1, -0.2], [0.3, -0.4]]], dtype=torch.float64)
    labels = torch.tensor([[[True, False], [False, True]]], dtype=torch.bool)
    recorded_inputs: list[torch.Tensor] = []
    original_softplus = v4_losses.F.softplus

    def softplus_spy(argument: torch.Tensor) -> torch.Tensor:
        recorded_inputs.append(argument.detach().clone())
        return original_softplus(argument)

    monkeypatch.setattr(v4_losses.F, "softplus", softplus_spy)

    loss = asymmetric_dts_loss(score, labels, endpoint=endpoint)

    assert loss.shape == (1,)
    assert len(recorded_inputs) == 1
    expected = (
        score[~labels] / 0.05
        if endpoint == 0.04
        else -score[labels] / 0.05
    )
    assert torch.equal(recorded_inputs[0], expected)


@pytest.mark.parametrize(
    ("score", "target", "endpoint", "expected"),
    [
        (
            torch.tensor([[[1e38]]], dtype=torch.float32),
            torch.tensor([[[False]]], dtype=torch.bool),
            0.04,
            torch.tensor([1e38], dtype=torch.float32),
        ),
        (
            torch.tensor([[[1e308]]], dtype=torch.float64),
            torch.tensor([[[False]]], dtype=torch.bool),
            0.04,
            torch.tensor([1e308], dtype=torch.float64),
        ),
        (
            torch.tensor([[[-1e38]]], dtype=torch.float32),
            torch.tensor([[[True]]], dtype=torch.bool),
            0.06,
            torch.tensor([1e38], dtype=torch.float32),
        ),
        (
            torch.tensor([[[-1e308]]], dtype=torch.float64),
            torch.tensor([[[True]]], dtype=torch.bool),
            0.06,
            torch.tensor([1e308], dtype=torch.float64),
        ),
    ],
)
def test_asymmetric_dts_extreme_finite_scores_stay_finite_when_exact_loss_is_finite(
    score, target, endpoint, expected
):
    loss = asymmetric_dts_loss(score, target, endpoint=endpoint)

    assert torch.equal(loss, expected)
    assert torch.isfinite(loss).all().item()


def test_v4_public_exports_and_loss_scope_exclude_forbidden_v3_imports():
    import shadowforge_pipeline.soft_voxel_v4 as v4

    assert v4.V4Profile is V4Profile
    assert v4.AuxiliarySlot is AuxiliarySlot
    assert v4.compute_baseline_branch is compute_baseline_branch
    assert v4.directional_dic_loss is directional_dic_loss
    assert v4.asymmetric_dts_loss is asymmetric_dts_loss
    source = _v4_loss_source()
    assert "soft_voxel_v3.trainer" not in source
    assert "soft_voxel_v3.calibration" not in source
    assert "soft_voxel_v3.registry" not in source
    assert "soft_voxel_v3.lovasz" not in source


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: compute_baseline_branch(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), object()), "V4Profile"),
        (lambda: directional_dic_terms([[[0.0]]], _bands(torch.ones(1, 1, 1, dtype=torch.bool), torch.ones(1, 1, 1, dtype=torch.bool))), "torch.Tensor"),
        (lambda: directional_dic_terms(torch.zeros(0, 1, 1), _bands(torch.ones(1, 1, 1, dtype=torch.bool), torch.ones(1, 1, 1, dtype=torch.bool))), "non-empty"),
        (lambda: directional_dic_terms(torch.zeros(1, 1, 1, dtype=torch.bool), _bands(torch.ones(1, 1, 1, dtype=torch.bool), torch.ones(1, 1, 1, dtype=torch.bool))), "floating"),
        (lambda: directional_dic_terms(torch.tensor([[[float("nan")]]]), _bands(torch.ones(1, 1, 1, dtype=torch.bool), torch.ones(1, 1, 1, dtype=torch.bool))), "finite"),
        (lambda: directional_dic_terms(torch.zeros(1, 1, 2), _bands(torch.ones(1, 1, 2), torch.ones(1, 1, 2, dtype=torch.bool))), "boolean"),
        (lambda: directional_dic_terms(torch.zeros(1, 1, 3), _bands(torch.tensor([[[True, False]]]), torch.tensor([[[False, True]]]))), "shape"),
        (lambda: directional_dic_terms(torch.zeros(1, 1, 2), _bands(torch.tensor([[[False, False]]]), torch.tensor([[[False, True]]]))), "non-empty inner"),
        (lambda: directional_dic_terms(torch.zeros(1, 1, 2), _bands(torch.tensor([[[True, False]]]), torch.tensor([[[False, False]]]))), "non-empty outer"),
        (lambda: directional_dic_terms(torch.zeros(1, 1, 2), _bands(torch.tensor([[[True, False]]]), torch.tensor([[[False, True]]])), temperature=0.0), "positive"),
        (lambda: directional_dic_terms(torch.zeros(1, 1, 2), _bands(torch.tensor([[[True, False]]]), torch.tensor([[[False, True]]])), temperature=True), "boolean"),
        (lambda: directional_dic_terms(torch.zeros(1, 1, 2), _bands(torch.tensor([[[True, False]]]), torch.tensor([[[False, True]]])), eps=False), "boolean"),
        (lambda: directional_dic_terms(torch.zeros(1, 1, 2), _bands(torch.tensor([[[True, False]]]), torch.tensor([[[False, True]]])), temperature="0.05"), "real number"),
        (lambda: directional_dic_terms(torch.zeros(1, 1, 2, dtype=torch.float32), _bands(torch.tensor([[[True, False]]]), torch.tensor([[[False, True]]])), temperature=1e-50), "representable"),
        (lambda: directional_dic_terms(torch.zeros(1, 1, 2), _bands(torch.tensor([[[True, False]]]), torch.tensor([[[False, True]]])), eps="1e-8"), "real number"),
        (lambda: directional_dic_terms(torch.zeros(1, 1, 2, dtype=torch.float32), _bands(torch.tensor([[[True, False]]]), torch.tensor([[[False, True]]])), eps=1e-50), "representable"),
        (lambda: asymmetric_dts_loss([[[0.0, 0.0]]], torch.zeros(1, 1, 2, dtype=torch.bool), endpoint=0.04), "torch.Tensor"),
        (lambda: asymmetric_dts_loss(torch.zeros(0, 1, 1), torch.zeros(0, 1, 1, dtype=torch.bool), endpoint=0.04), "non-empty"),
        (lambda: asymmetric_dts_loss(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2, dtype=torch.bool), endpoint=0.05), "endpoint"),
        (lambda: asymmetric_dts_loss(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2, dtype=torch.bool), endpoint=True), "endpoint"),
        (lambda: asymmetric_dts_loss(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2, dtype=torch.bool), endpoint="0.04"), "real number"),
        (lambda: asymmetric_dts_loss(torch.zeros(1, 1, 2, dtype=torch.bool), torch.zeros(1, 1, 2, dtype=torch.bool), endpoint=0.04), "floating"),
        (lambda: asymmetric_dts_loss(torch.tensor([[[float("inf"), 0.0]]]), torch.zeros(1, 1, 2, dtype=torch.bool), endpoint=0.04), "finite"),
        (lambda: asymmetric_dts_loss(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), endpoint=0.04), "boolean"),
        (lambda: asymmetric_dts_loss(torch.zeros(1, 1, 3), torch.tensor([[[True, False]]]), endpoint=0.04), "shape"),
        (lambda: asymmetric_dts_loss(torch.zeros(1, 1, 2), [[[False, False]]], endpoint=0.04), "torch.Tensor"),
        (lambda: asymmetric_dts_loss(torch.zeros(1, 1, 2), torch.tensor([[[True, True]]]), endpoint=0.04), "selected support"),
        (lambda: asymmetric_dts_loss(torch.zeros(1, 1, 2), torch.tensor([[[False, False]]]), endpoint=0.06), "selected support"),
        (lambda: asymmetric_dts_loss(torch.zeros(1, 1, 2, dtype=torch.float32), torch.tensor([[[False, True]]]), endpoint=0.04, temperature="0.05"), "real number"),
        (lambda: asymmetric_dts_loss(torch.zeros(1, 1, 2, dtype=torch.float32), torch.tensor([[[False, True]]]), endpoint=0.04, temperature=1e-50), "representable"),
    ],
)
def test_v4_loss_preflight_rejects_invalid_inputs(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kind", ["dic", "dts"])
def test_v4_loss_preflight_rejects_cross_device_inputs(kind):
    score = torch.zeros((1, 1, 2), device="cuda")
    labels = torch.tensor([[[True, False]]], device="cpu")
    bands = _bands(labels, ~labels)

    calls = {
        "dic": lambda: directional_dic_terms(score, bands),
        "dts": lambda: asymmetric_dts_loss(score, labels, endpoint=0.04),
    }
    with pytest.raises(ValueError, match="device"):
        calls[kind]()


@pytest.mark.parametrize("consumer", [compute_baseline_branch, auxiliary_slot])
def test_v4_public_profile_consumers_reject_forged_and_mutated_profiles(consumer):
    forged = V4Profile("v4_full", 0.35, 1.0, True, True)
    mutated = V4Profile("v4_full", 0.35, 1.0, True, True)
    object.__setattr__(mutated, "use_dts", False)

    args = (torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)) if consumer is compute_baseline_branch else (0, 1.0)

    with pytest.raises(ValueError, match="registered V4Profile"):
        consumer(*args, forged)
    with pytest.raises(ValueError, match="registered V4Profile"):
        consumer(*args, mutated)
