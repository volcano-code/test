import math

import pytest
import torch

from shadowforge_pipeline.soft_voxel_v2 import (
    aggregate_view_losses,
    balanced_lp_loss,
    boundary_charbonnier_loss,
    boundary_weight_schedule,
    boundary_weights,
    compact_soft_threshold,
    compute_soft_loss,
    dual_domain_balanced_loss,
    evaluate_soft_views,
    gate_loss,
    gate_weight_schedule,
    group_mirror_pairs,
    proxy_weight_schedule,
    resolve_profile,
    robust_rho_schedule,
    threshold_margin_schedule,
)


def test_compact_threshold_has_exact_plateaus_and_linear_gradient():
    density = torch.tensor([0.02, 0.04, 0.05, 0.06, 0.08], requires_grad=True)
    proxy = compact_soft_threshold(density, tau=0.05, margin=0.01)

    assert torch.allclose(proxy, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    proxy.sum().backward()
    assert torch.allclose(density.grad, torch.tensor([0.0, 0.0, 50.0, 0.0, 0.0]))


@pytest.mark.parametrize("p", [1, 2])
def test_balanced_lp_weights_foreground_and_background_equally(p):
    target = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    pred = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])

    # Foreground error is 1, background mean error is 1/3; each region gets 0.5.
    expected = 0.5 * 1.0 + 0.5 * (1.0 / 3.0)
    loss = balanced_lp_loss(pred, target, p=p)

    assert loss.shape == (1,)
    assert loss.item() == pytest.approx(expected)


def test_dual_domain_loss_adds_normalized_proxy_auxiliary_per_view():
    target = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    raw = target.clone()
    proxy = 1.0 - target

    raw_only = dual_domain_balanced_loss(raw, proxy, target, p=2, proxy_weight=0.0)
    full_proxy_auxiliary = dual_domain_balanced_loss(
        raw, proxy, target, p=2, proxy_weight=1.0
    )
    blended = dual_domain_balanced_loss(raw, proxy, target, p=2, proxy_weight=0.25)

    assert torch.allclose(raw_only, torch.zeros(2))
    assert torch.allclose(full_proxy_auxiliary, torch.full((2,), 0.5))
    assert torch.allclose(blended, torch.full((2,), 0.2))


def test_dual_domain_keeps_nonzero_raw_gradient_at_full_proxy_weight():
    target = torch.tensor([[[1.0, 0.0]]])
    raw = torch.tensor([[[0.4, 0.4]]], requires_grad=True)
    # A detached plateau emulates a compact proxy with zero local derivative.
    proxy_plateau = torch.zeros_like(raw).detach()

    loss = dual_domain_balanced_loss(
        raw,
        proxy_plateau,
        target,
        p=2,
        proxy_weight=1.0,
    ).sum()
    loss.backward()

    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum().item() > 0.0


def test_gate_loss_uses_recall_guard_before_fpr_penalty():
    target = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])

    low_recall = torch.tensor([[[0.90, 0.90, 1.00, 1.00]]])
    low_recall_loss = gate_loss(low_recall, target)
    assert low_recall_loss.item() == pytest.approx((0.985 - 0.90) ** 2)

    high_recall = torch.tensor([[[0.98, 0.98, 0.03, 0.03]]])
    high_recall_loss = gate_loss(high_recall, target)
    expected = (0.985 - 0.98) ** 2 + (0.03 - 0.01) ** 2
    assert high_recall_loss.item() == pytest.approx(expected)


def test_boundary_weights_peak_at_target_boundary_and_loss_is_per_view():
    target = torch.zeros((1, 9, 9))
    target[:, 3:6, 3:6] = 1.0
    weights = boundary_weights(target, sigma_px=2.0)

    assert weights.shape == target.shape
    assert weights[0, 3, 4].item() == pytest.approx(1.0)
    assert weights[0, 0, 0] < weights[0, 2, 4]

    perfect = boundary_charbonnier_loss(target, target, weights=weights)
    perturbed = target.clone()
    perturbed[0, 3, 4] = 0.0
    wrong = boundary_charbonnier_loss(perturbed, target, weights=weights)
    assert perfect.shape == (1,)
    assert wrong.item() > perfect.item()


def test_mirror_pair_grouping_and_grouped_robust_formula():
    losses = torch.tensor([1.0, 2.0, 3.0, 5.0, 6.0, 7.0])
    grouped = group_mirror_pairs(losses)
    assert torch.allclose(grouped, torch.tensor([3.0, 4.0, 5.0]))

    mean = aggregate_view_losses(losses, mode="mean")
    robust = aggregate_view_losses(losses, mode="grouped_robust", beta=4.0, rho=0.25)
    lse = torch.logsumexp(4.0 * grouped, dim=0) / 4.0 - math.log(3.0) / 4.0
    expected = 0.75 * grouped.mean() + 0.25 * lse
    assert mean.item() == pytest.approx(4.0)
    assert robust.item() == pytest.approx(expected.item())


def test_schedules_follow_preregistered_breakpoints():
    assert proxy_weight_schedule(0.24) == 0.0
    assert proxy_weight_schedule(0.25) == 0.0
    assert proxy_weight_schedule(0.45) == pytest.approx(0.5)
    assert proxy_weight_schedule(0.65) == 1.0
    assert proxy_weight_schedule(0.90) == 1.0

    assert gate_weight_schedule(0.25) == 0.0
    assert gate_weight_schedule(0.65) == 1.0
    assert boundary_weight_schedule(0.25) == 0.0
    assert boundary_weight_schedule(0.65) == 1.0

    assert robust_rho_schedule(0.69) == 0.0
    assert robust_rho_schedule(0.70) == 0.0
    assert robust_rho_schedule(0.85) == pytest.approx(0.125)
    assert robust_rho_schedule(1.0) == 0.25

    assert threshold_margin_schedule(0.0) == pytest.approx(0.02)
    assert threshold_margin_schedule(1.0) == pytest.approx(0.005)


def test_profiles_and_component_disables_are_explicit_and_independent():
    legacy = resolve_profile("legacy_full")
    plain = resolve_profile("l12_plain")
    balanced = resolve_profile("l12_bal")
    full = resolve_profile("new_full")

    assert legacy.legacy_terms and not legacy.balanced
    assert not plain.legacy_terms and not plain.balanced
    assert balanced.balanced and not balanced.use_proxy
    assert full.balanced and full.use_proxy and full.use_gate and full.use_boundary
    assert full.l1_weight == pytest.approx(0.35)
    assert full.l2_weight == pytest.approx(1.0)

    ablated = resolve_profile(
        "new_full", disable_proxy=True, disable_gate=True, disable_boundary=False,
        view_aggregation="mean",
    )
    assert not ablated.use_proxy
    assert not ablated.use_gate
    assert ablated.use_boundary
    assert ablated.view_aggregation == "mean"

    with pytest.raises(ValueError):
        resolve_profile("not_a_profile")


def test_composite_new_loss_matches_preregistered_weighted_formula():
    target = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])
    raw = torch.tensor([[[0.8, 0.1]], [[0.7, 0.2]]], requires_grad=True)
    proxy = torch.tensor([[[0.9, 0.03]], [[0.8, 0.04]]], requires_grad=True)
    weights = torch.ones_like(target)
    profile = resolve_profile("new_full", view_aggregation="mean")

    result = compute_soft_loss(
        raw,
        proxy,
        target,
        profile=profile,
        progress=0.65,
        lambda_gate=0.4,
        lambda_boundary=0.2,
        cached_boundary_weights=weights,
    )

    expected_per_view = (
        dual_domain_balanced_loss(raw, proxy, target, p=2, proxy_weight=1.0)
        + 0.35 * dual_domain_balanced_loss(raw, proxy, target, p=1, proxy_weight=1.0)
        + 0.4 * gate_loss(proxy, target)
        + 0.2 * boundary_charbonnier_loss(proxy, target, weights=weights)
    )
    assert torch.allclose(result.per_view_total, expected_per_view)
    assert torch.allclose(result.total, expected_per_view.mean())
    assert result.proxy_weight == 1.0
    result.total.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert proxy.grad is not None and torch.isfinite(proxy.grad).all()


def test_metrics_report_per_view_confusion_fpr_and_threshold_stability():
    target = torch.tensor(
        [
            [[1.0, 1.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )
    prediction = torch.tensor(
        [
            [[0.90, 0.80], [0.20, 0.01]],
            [[0.90, 0.20], [0.01, 0.01]],
        ]
    )

    report = evaluate_soft_views(
        prediction,
        target,
        thresholds=(0.04, 0.05, 0.06),
        primary_threshold=0.05,
        boundary_tolerance_px=2.0,
    )

    assert report["primary_threshold"] == 0.05
    assert set(report["thresholds"]) == {"0.04", "0.05", "0.06"}
    assert len(report["per_view"]) == 2

    first = report["per_view"][0]
    assert (first["tp"], first["fp"], first["fn"], first["tn"]) == (2, 1, 0, 1)
    assert first["binary_iou"] == pytest.approx(2.0 / 3.0)
    assert first["dice"] == pytest.approx(4.0 / 5.0)
    # Crucially, FPR is normalized by target-background pixels: FP / (FP + TN).
    assert first["fpr"] == pytest.approx(1.0 / 2.0)
    assert first["recall"] == pytest.approx(1.0)
    assert 0.0 <= first["boundary_f1_2px"] <= 1.0
    assert first["soft_binary_iou_gap"] == pytest.approx(
        first["continuous_iou"] - first["binary_iou"]
    )

    aggregate = report["aggregate"]
    assert aggregate["worst_view_binary_iou"] == pytest.approx(0.5)
    assert aggregate["mean_binary_iou"] == pytest.approx((2.0 / 3.0 + 0.5) / 2.0)


def test_metrics_are_json_serializable_scalars():
    import json

    prediction = torch.tensor([[[0.9, 0.0], [0.0, 0.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    report = evaluate_soft_views(prediction, target)
    json.dumps(report)
