"""Small analytic loss tests; changes to reductions or ramps must be observable."""
import importlib

import pytest
import torch


def api():
    return importlib.import_module("experiments.shadowart_vnext.losses")


def test_core_uses_view_equal_unbalanced_l1_l2_and_frozen_deletion_weight():
    m = api()
    p = torch.tensor([[[.2, .8]], [[.4, .6]]], requires_grad=True)
    y = torch.tensor([[[0., 1.]], [[1., 0.]]])
    loss, diag = m.compute_loss(p, y, m.LossConfig(), .5)
    assert loss.item() == pytest.approx(.34)
    assert diag["l1"] == pytest.approx(.4)
    assert diag["l2"] == pytest.approx(.2)
    one, _ = m.compute_loss(p, y, m.LossConfig(core="l1", l1_weight=.35), 1.)
    assert one.item() == pytest.approx(.14)
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_aux_ramp_and_balancing_do_not_change_when_one_class_absent():
    m = api()
    p = torch.full((2, 2, 2), .5, requires_grad=True)
    y = torch.stack([torch.zeros(2, 2), torch.ones(2, 2)])
    c = m.LossConfig(core="l2", auxiliary="ray_bce", aux_weight=2.)
    a, _ = m.compute_loss(p, y, c, .2)
    b, d = m.compute_loss(p, y, c, .25)
    assert a.item() == pytest.approx(.25)
    assert b.item() == pytest.approx(.25 + .69314718)
    assert d["aux_ramp"] == pytest.approx(.5)
    b.backward()
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("aux", ["none", "lovasz", "ray_bce"])
def test_exact_empty_and_full_predictions_have_finite_gradients(aux):
    m = api()
    y = torch.stack([torch.zeros(3, 3), torch.ones(3, 3)])
    p = y.clone().requires_grad_()
    loss, d = m.compute_loss(p, y, m.LossConfig(auxiliary=aux, aux_weight=0. if aux == "none" else 1.), 1.)
    assert loss.item() < 1e-5
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_legacy_control_reports_the_truthful_ssim_and_aux_cannot_be_added():
    m = api()
    p = torch.zeros(1, 12, 12, requires_grad=True)
    loss, d = m.compute_loss(p, torch.ones_like(p), m.LossConfig(control="current_proposal"), 1.)
    assert d["ssim_method"] == "uniform11_zero_padding_legacy"
    assert loss.item() == pytest.approx(1. + .35 + .8 * 144 / 145 + .18 * d["ssim"])
    with pytest.raises(ValueError):
        m.LossConfig(control="current_proposal", auxiliary="lovasz", aux_weight=.1)


@pytest.mark.parametrize("kwargs", [{"core":"other"}, {"aux_weight":float("nan")}, {"auxiliary":"lovasz+ray_bce"}, {"aux_weight":1.}])
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        api().LossConfig(**kwargs)


def test_invalid_predictions_and_progress_fail_before_optimization():
    m = api()
    for p, y, progress in [(torch.ones(1,2,2)*1.1, torch.zeros(1,2,2), .5),
                           (torch.ones(0,2,2), torch.zeros(0,2,2), .5),
                           (torch.zeros(1,2,2), torch.zeros(1,2,2), float("nan"))]:
        with pytest.raises(ValueError):
            m.compute_loss(p, y, m.LossConfig(), progress)
