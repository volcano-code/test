import json
import math
from dataclasses import FrozenInstanceError, asdict, fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shadowforge_pipeline.soft_voxel_v3.contours import ContourBands
from shadowforge_pipeline.soft_voxel_v4 import (
    GuardedAuxiliaryGradient,
    auxiliary_slot,
    guard_auxiliary_gradient,
    resolve_profile,
)
import shadowforge_pipeline.soft_voxel_v4.trainer_step as trainer_step_module
from shadowforge_pipeline.soft_voxel_v4.trainer_step import (
    AuxiliarySlotAudit,
    V4StepConfig,
    V4StepResult,
    train_one_v4_update,
)


class _ToyProtocol:
    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.inputs: list[torch.Tensor] = []

    def render_all(self, density: torch.Tensor) -> torch.Tensor:
        self.inputs.append(density)
        flat = density[0, 0].reshape(2, 4)
        multipliers = density.new_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).view(6, 1, 1)
        return flat.unsqueeze(0) * multipliers


class _ToyOptimizer:
    def __init__(self, parameter: torch.Tensor, *, lr: float = 0.25) -> None:
        self.parameter = parameter
        self.lr = lr
        self.param_groups = [{"params": [parameter]}]
        self.zero_grad_calls: list[bool] = []
        self.step_calls = 0
        self.step_grad: torch.Tensor | None = None
        self.state = {"steps": 0}

    def zero_grad(self, *, set_to_none: bool = False) -> None:
        self.zero_grad_calls.append(set_to_none)
        if set_to_none:
            self.parameter.grad = None
        elif self.parameter.grad is not None:
            self.parameter.grad.zero_()

    def step(self) -> None:
        self.step_calls += 1
        self.state["steps"] += 1
        assert self.parameter.grad is not None
        self.step_grad = self.parameter.grad.detach().clone()
        with torch.no_grad():
            self.parameter.add_(self.parameter.grad, alpha=-self.lr)


def _logits(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(
        [-1.6, -1.1, -0.7, -0.2, 0.1, 0.4, 0.9, 1.5],
        dtype=torch.float64,
        device=device,
    ).reshape(1, 1, 2, 2, 2).requires_grad_()


def _targets(device: torch.device | str = "cpu") -> torch.Tensor:
    target = torch.tensor(
        [
            [[0.5, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.5, 0.0]],
            [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
            [[1.0, 1.0, 0.0, 0.0], [0.0, 0.5, 1.0, 0.0]],
            [[0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.5, 0.0]],
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.5]],
        ],
        dtype=torch.float64,
        device=device,
    )
    return target


def _bands(device: torch.device | str = "cpu") -> ContourBands:
    target_bool = _targets(device) >= 0.5
    return ContourBands(inner=target_bool, outer=~target_bool)


def _config(
    *,
    dic_unit_coefficient: float = 2.0,
    dts_unit_coefficient: float = 3.0,
    dic_max_ratio: float = 0.04,
    dts_max_ratio: float = 0.08,
) -> V4StepConfig:
    return V4StepConfig(
        dic_unit_coefficient=dic_unit_coefficient,
        dts_unit_coefficient=dts_unit_coefficient,
        dic_max_ratio=dic_max_ratio,
        dts_max_ratio=dts_max_ratio,
    )


def _kwargs(
    profile_name: str,
    *,
    device: torch.device | str = "cpu",
    global_step_before: int = 13,
    total_steps: int = 20,
    bands: object | None = None,
) -> tuple[dict[str, object], torch.Tensor, _ToyOptimizer, _ToyProtocol]:
    resolved_device = torch.device(device)
    logits = _logits(resolved_device)
    optimizer = _ToyOptimizer(logits)
    protocol = _ToyProtocol(resolved_device)
    supplied_bands = _bands(resolved_device) if bands is None else bands
    return (
        {
            "logits": logits,
            "optimizer": optimizer,
            "protocol": protocol,
            "targets": _targets(resolved_device),
            "bands": supplied_bands,
            "profile": resolve_profile(profile_name),
            "global_step_before": global_step_before,
            "total_steps": total_steps,
            "config": _config(),
        },
        logits,
        optimizer,
        protocol,
    )


def _clip_to_unit(gradient: torch.Tensor) -> torch.Tensor:
    norm = float(torch.linalg.vector_norm(gradient).item())
    if norm <= 1.0 or norm == 0.0:
        return gradient.clone()
    return gradient * (1.0 / norm)


def _make_branch_patch(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    records: dict[str, object] = {
        "thresholded": [],
        "target_bools": [],
        "dts_endpoints": [],
        "dic_calls": 0,
        "dts_calls": 0,
        "anchor_calls": 0,
    }
    anchor_weights = torch.tensor([0.5, -1.5, 2.5, -3.5, 4.5, -5.5], dtype=torch.float64)
    dic_weights = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=torch.float64)
    dts_004_weights = torch.tensor([2.0, -1.0, 4.0, -3.0, 6.0, -5.0], dtype=torch.float64)
    dts_006_weights = torch.tensor([-3.0, 2.0, -5.0, 4.0, -7.0, 6.0], dtype=torch.float64)

    def _view_loss(score: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        local = weights.to(device=score.device, dtype=score.dtype).view(6, 1)
        return score.reshape(6, -1).mean(dim=1) * local.reshape(-1)

    def fake_render_thresholded(protocol: object, density: torch.Tensor, tau: float) -> object:
        records["thresholded"].append((id(density), float(tau), density.device.type))
        return SimpleNamespace(score=protocol.render_all(density) + density.new_tensor(float(tau)))

    def fake_base(projections: torch.Tensor, targets: torch.Tensor, profile: object) -> torch.Tensor:
        return ((projections - targets) ** 2).mean() + 0.35 * (projections - targets).abs().mean()

    def fake_anchor(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        records["anchor_calls"] = int(records["anchor_calls"]) + 1
        records["target_bools"].append(labels.detach().clone())
        return _view_loss(logits, anchor_weights)

    def fake_dic(score: torch.Tensor, bands: ContourBands) -> torch.Tensor:
        records["dic_calls"] = int(records["dic_calls"]) + 1
        return _view_loss(score, dic_weights)

    def fake_dts(score: torch.Tensor, labels: torch.Tensor, endpoint: float) -> torch.Tensor:
        records["dts_calls"] = int(records["dts_calls"]) + 1
        records["target_bools"].append(labels.detach().clone())
        records["dts_endpoints"].append(float(endpoint))
        if endpoint == 0.04:
            return _view_loss(score, dts_004_weights)
        if endpoint == 0.06:
            return _view_loss(score, dts_006_weights)
        raise AssertionError(endpoint)

    monkeypatch.setattr(trainer_step_module, "render_thresholded", fake_render_thresholded)
    monkeypatch.setattr(trainer_step_module, "compute_baseline_branch", fake_base)
    monkeypatch.setattr(trainer_step_module, "lovasz_hinge_per_view", fake_anchor)
    monkeypatch.setattr(trainer_step_module, "directional_dic_loss", fake_dic)
    monkeypatch.setattr(trainer_step_module, "asymmetric_dts_loss", fake_dts)
    return records


def _expected_update(
    *,
    logits: torch.Tensor,
    protocol: _ToyProtocol,
    targets: torch.Tensor,
    bands: ContourBands,
    profile_name: str,
    step: int,
    total_steps: int,
    config: V4StepConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    expected_logits = logits.detach().clone().requires_grad_(True)
    expected_protocol = _ToyProtocol(protocol.device)
    profile = resolve_profile(profile_name)
    progress = (step + 1) / float(total_steps)
    slot = auxiliary_slot(step, progress, profile)
    assert slot is not None

    base_density = torch.sigmoid(expected_logits)
    projections = expected_protocol.render_all(base_density)
    base_loss = ((projections - targets) ** 2).mean()
    base_loss = base_loss + 0.35 * (projections - targets).abs().mean()
    (base_grad,) = torch.autograd.grad(base_loss, expected_logits)

    anchor_density = torch.sigmoid(expected_logits)
    anchor_score = expected_protocol.render_all(anchor_density) + anchor_density.new_tensor(0.05)
    anchor_loss = (anchor_score.reshape(6, -1).mean(dim=1) * anchor_density.new_tensor([0.5, -1.5, 2.5, -3.5, 4.5, -5.5])).mean()
    (anchor_grad,) = torch.autograd.grad(anchor_loss, expected_logits)

    if slot.component == "dic":
        aux_density = torch.sigmoid(expected_logits)
        aux_score = expected_protocol.render_all(aux_density) + aux_density.new_tensor(0.05)
        aux_loss = (aux_score.reshape(6, -1).mean(dim=1) * aux_density.new_tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])).mean()
        (raw_aux_grad,) = torch.autograd.grad(aux_loss, expected_logits)
        scale = config.dic_unit_coefficient * config.dic_max_ratio * slot.multiplier
        max_ratio = config.dic_max_ratio
    else:
        assert slot.endpoint in (0.04, 0.06)
        aux_density = torch.sigmoid(expected_logits)
        aux_score = expected_protocol.render_all(aux_density) + aux_density.new_tensor(slot.endpoint)
        weights = (
            [2.0, -1.0, 4.0, -3.0, 6.0, -5.0]
            if slot.endpoint == 0.04
            else [-3.0, 2.0, -5.0, 4.0, -7.0, 6.0]
        )
        aux_loss = (aux_score.reshape(6, -1).mean(dim=1) * aux_density.new_tensor(weights)).mean()
        (raw_aux_grad,) = torch.autograd.grad(aux_loss, expected_logits)
        scale = config.dts_unit_coefficient * config.dts_max_ratio * slot.multiplier
        max_ratio = config.dts_max_ratio

    weighted_aux = raw_aux_grad * scale
    guarded = guard_auxiliary_gradient(
        base_grad.detach().clone(),
        anchor_grad.detach().clone(),
        weighted_aux.detach().clone(),
        max_ratio,
    )
    final_grad = base_grad + guarded.gradient
    clipped = _clip_to_unit(final_grad)
    return clipped, base_grad, anchor_grad, guarded.gradient, slot.multiplier


@pytest.mark.parametrize(
    ("profile_name", "step", "expected_component", "expected_endpoint", "expected_multiplier"),
    [
        ("l12_bal", 11, None, None, 0.0),
        ("dic_only", 11, None, None, 0.0),
        ("dts_only", 11, None, None, 0.0),
        ("v4_full", 11, None, None, 0.0),
        ("v4_no_l1", 11, None, None, 0.0),
        ("v4_no_l2", 11, None, None, 0.0),
        ("dic_only", 12, "dic", None, (0.65 - 0.60) / 0.15),
        ("dts_only", 13, "dts", 0.04, (0.70 - 0.60) / 0.15),
        ("v4_full", 14, "dic", None, 1.0),
        ("v4_full", 15, "dts", 0.06, 1.0),
        ("v4_no_l1", 12, "dic", None, (0.65 - 0.60) / 0.15),
        ("v4_no_l2", 15, "dts", 0.06, 1.0),
    ],
)
def test_train_one_v4_update_preserves_exact_slot_phase_boundaries_and_profiles(
    profile_name: str,
    step: int,
    expected_component: str | None,
    expected_endpoint: float | None,
    expected_multiplier: float,
) -> None:
    kwargs, _, _, _ = _kwargs(profile_name, global_step_before=step, total_steps=20)

    result = train_one_v4_update(**kwargs)

    assert result.global_step_before == step
    assert result.completed_updates_after == step + 1
    assert result.progress == (step + 1) / 20.0
    assert result.slot_component == expected_component
    assert result.slot_endpoint == expected_endpoint
    assert result.slot_multiplier == pytest.approx(expected_multiplier, rel=0.0, abs=1e-12)
    assert (result.slot_audit is None) is (expected_component is None)


def test_base_only_step_uses_one_fresh_grad_one_clip_and_one_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _, optimizer, protocol = _kwargs("v4_full", global_step_before=11, total_steps=20)
    original_sigmoid = torch.sigmoid
    original_grad = torch.autograd.grad
    original_clip = torch.nn.utils.clip_grad_norm_
    sigmoid_outputs: list[torch.Tensor] = []
    grad_calls: list[dict[str, object]] = []
    clip_calls: list[tuple[list[torch.Tensor], float]] = []

    def record_sigmoid(value: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        output = original_sigmoid(value, *args, **kwargs)
        if value is kwargs_["logits"]:
            sigmoid_outputs.append(output)
        return output

    def record_grad(*args: object, **kwargs: object):
        grad_calls.append(dict(kwargs))
        return original_grad(*args, **kwargs)

    def record_clip(parameters: object, max_norm: float, *args: object, **kwargs: object):
        parameter_list = list(parameters)  # type: ignore[arg-type]
        clip_calls.append((parameter_list, max_norm))
        return original_clip(parameter_list, max_norm, *args, **kwargs)

    kwargs_ = kwargs
    monkeypatch.setattr(torch, "sigmoid", record_sigmoid)
    monkeypatch.setattr(torch.autograd, "grad", record_grad)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)

    result = train_one_v4_update(**kwargs)

    assert len(sigmoid_outputs) == 1
    assert protocol.inputs == [sigmoid_outputs[0]]
    assert len(grad_calls) == 1
    assert grad_calls[0].get("retain_graph", False) is False
    assert grad_calls[0].get("create_graph", False) is False
    assert optimizer.zero_grad_calls == [True]
    assert len(clip_calls) == 1
    assert clip_calls[0][0] == [kwargs["logits"]]
    assert clip_calls[0][1] == 1.0
    assert optimizer.step_calls == 1
    assert result.slot_audit is None


@pytest.mark.parametrize(
    ("profile_name", "step", "poison_bands", "forbidden"),
    [
        ("dts_only", 13, object(), "directional_dic_loss"),
        ("dic_only", 12, None, "asymmetric_dts_loss"),
    ],
)
def test_single_component_profiles_construct_only_the_selected_branch(
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
    step: int,
    poison_bands: object | None,
    forbidden: str,
) -> None:
    kwargs, _, optimizer, _ = _kwargs(
        profile_name,
        global_step_before=step,
        total_steps=20,
        bands=_bands() if poison_bands is None else poison_bands,
    )
    records = _make_branch_patch(monkeypatch)
    original_grad = torch.autograd.grad
    grad_calls: list[dict[str, object]] = []

    def record_grad(*args: object, **kwargs: object):
        grad_calls.append(dict(kwargs))
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", record_grad)
    if forbidden == "asymmetric_dts_loss":
        monkeypatch.setattr(
            trainer_step_module,
            "asymmetric_dts_loss",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("disabled DTS used")),
        )

    result = train_one_v4_update(**kwargs)

    assert optimizer.step_calls == 1
    assert len(grad_calls) == 3
    assert all(call.get("retain_graph", False) is False for call in grad_calls)
    assert result.slot_audit is not None
    assert result.slot_audit.cross_diagnostics_expected is False
    assert result.slot_audit.diagnostic_dic_raw_gradient_norm is None
    assert result.slot_audit.diagnostic_dts_004_raw_gradient_norm is None
    assert result.slot_audit.diagnostic_dts_006_raw_gradient_norm is None
    assert result.slot_audit.dic_dts_004_support_intersection is None
    assert result.slot_audit.dic_dts_004_support_union is None
    assert result.slot_audit.dic_dts_006_support_intersection is None
    assert result.slot_audit.dic_dts_006_support_union is None
    assert result.slot_audit.dic_dts_004_cosine is None
    assert result.slot_audit.dic_dts_004_cosine_invalid is True
    assert result.slot_audit.dic_dts_006_cosine is None
    assert result.slot_audit.dic_dts_006_cosine_invalid is True
    assert result.slot_audit.dic_dts_004_support_iou is None
    assert result.slot_audit.dic_dts_004_support_iou_invalid is True
    assert result.slot_audit.dic_dts_006_support_iou is None
    assert result.slot_audit.dic_dts_006_support_iou_invalid is True
    expected_taus = [0.05, 0.04] if profile_name == "dts_only" else [0.05, 0.05]
    assert [tau for _, tau, _ in records["thresholded"]] == expected_taus


def test_full_profile_constructs_five_fresh_branches_but_only_selected_aux_is_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, logits, optimizer, protocol = _kwargs("v4_full", global_step_before=13, total_steps=20)
    records = _make_branch_patch(monkeypatch)
    original_sigmoid = torch.sigmoid
    original_grad = torch.autograd.grad
    original_guard = trainer_step_module.guard_auxiliary_gradient
    sigmoid_outputs: list[torch.Tensor] = []
    guard_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []
    grad_calls: list[dict[str, object]] = []

    def record_sigmoid(value: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        output = original_sigmoid(value, *args, **kwargs)
        if value is logits:
            sigmoid_outputs.append(output)
        return output

    def record_grad(*args: object, **kwargs: object):
        grad_calls.append(dict(kwargs))
        return original_grad(*args, **kwargs)

    def record_guard(
        base_grad: torch.Tensor,
        anchor_grad: torch.Tensor,
        aux_grad: torch.Tensor,
        max_ratio: float,
    ) -> GuardedAuxiliaryGradient:
        guard_inputs.append(
            (base_grad.detach().clone(), anchor_grad.detach().clone(), aux_grad.detach().clone(), max_ratio)
        )
        return original_guard(base_grad, anchor_grad, aux_grad, max_ratio)

    monkeypatch.setattr(torch, "sigmoid", record_sigmoid)
    monkeypatch.setattr(torch.autograd, "grad", record_grad)
    monkeypatch.setattr(trainer_step_module, "guard_auxiliary_gradient", record_guard)

    before = logits.detach().clone()
    result = train_one_v4_update(**kwargs)
    observed_grad_calls = len(grad_calls)
    expected_grad, base_grad, anchor_grad, guarded_grad, multiplier = _expected_update(
        logits=before,
        protocol=protocol,
        targets=kwargs["targets"],
        bands=kwargs["bands"],
        profile_name="v4_full",
        step=13,
        total_steps=20,
        config=kwargs["config"],
    )

    assert len(sigmoid_outputs) == 5
    assert len({id(value) for value in sigmoid_outputs}) == 5
    assert observed_grad_calls == 5
    assert [tau for _, tau, _ in records["thresholded"]] == [0.05, 0.05, 0.04, 0.06]
    assert optimizer.step_calls == 1
    assert len(guard_inputs) == 1
    assert guard_inputs[0][3] == kwargs["config"].dts_max_ratio
    assert torch.allclose(logits, before - optimizer.lr * expected_grad, rtol=0.0, atol=2e-7)
    wrong_with_anchor = before - optimizer.lr * _clip_to_unit(base_grad + guarded_grad + anchor_grad)
    assert not torch.allclose(logits, wrong_with_anchor, rtol=0.0, atol=1e-9)
    assert result.slot_audit is not None
    assert result.slot_component == "dts"
    assert result.slot_endpoint == 0.04
    assert result.slot_multiplier == pytest.approx(multiplier, rel=0.0, abs=1e-12)
    assert result.slot_audit.cross_diagnostics_expected is True
    assert result.slot_audit.diagnostic_dic_raw_gradient_norm is not None
    assert result.slot_audit.diagnostic_dts_004_raw_gradient_norm == pytest.approx(
        result.slot_audit.raw_component_gradient_norm, rel=0.0, abs=1e-12
    )
    assert result.slot_audit.diagnostic_dts_006_raw_gradient_norm is not None
    assert result.slot_audit.weighted_auxiliary_gradient_norm > 0.0
    assert result.slot_audit.final_auxiliary_gradient_norm == pytest.approx(
        float(torch.linalg.vector_norm(guarded_grad).item()), rel=0.0, abs=1e-12
    )


def test_selected_auxiliary_uses_view_mean_and_excludes_anchor_from_parameter_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, logits, optimizer, protocol = _kwargs("dic_only", global_step_before=12, total_steps=20)
    _make_branch_patch(monkeypatch)
    before = logits.detach().clone()

    result = train_one_v4_update(**kwargs)
    expected_grad, base_grad, anchor_grad, guarded_grad, _ = _expected_update(
        logits=before,
        protocol=protocol,
        targets=kwargs["targets"],
        bands=kwargs["bands"],
        profile_name="dic_only",
        step=12,
        total_steps=20,
        config=kwargs["config"],
    )
    wrong_sum = _clip_to_unit(base_grad + guarded_grad * 6.0)
    wrong_with_anchor = _clip_to_unit(base_grad + guarded_grad + anchor_grad)

    assert optimizer.step_calls == 1
    assert torch.allclose(logits, before - optimizer.lr * expected_grad, rtol=0.0, atol=2e-7)
    assert not torch.allclose(logits, before - optimizer.lr * wrong_sum, rtol=0.0, atol=1e-9)
    assert not torch.allclose(logits, before - optimizer.lr * wrong_with_anchor, rtol=0.0, atol=1e-9)
    assert result.loss_anchor is not None
    assert result.loss_auxiliary is not None


def test_step_result_and_slot_audit_are_exact_frozen_json_safe_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _, _, _ = _kwargs("v4_full", global_step_before=14, total_steps=20)
    _make_branch_patch(monkeypatch)

    result = train_one_v4_update(**kwargs)
    payload = asdict(result)

    assert [field.name for field in fields(V4StepConfig)] == [
        "dic_unit_coefficient",
        "dts_unit_coefficient",
        "dic_max_ratio",
        "dts_max_ratio",
    ]
    assert [field.name for field in fields(AuxiliarySlotAudit)] == [
        "global_step_before",
        "profile",
        "cross_diagnostics_expected",
        "component",
        "endpoint",
        "multiplier",
        "unit_coefficient",
        "max_ratio",
        "loss_auxiliary",
        "raw_component_gradient_norm",
        "weighted_auxiliary_gradient_norm",
        "projected_auxiliary_gradient_norm",
        "final_auxiliary_gradient_norm",
        "anchor_cosine_before",
        "anchor_cosine_after",
        "removed_conflict_fraction",
        "auxiliary_base_ratio",
        "cap_applied",
        "fully_projected_zero",
        "diagnostic_dic_raw_gradient_norm",
        "diagnostic_dts_004_raw_gradient_norm",
        "diagnostic_dts_006_raw_gradient_norm",
        "dic_dts_004_support_intersection",
        "dic_dts_004_support_union",
        "dic_dts_006_support_intersection",
        "dic_dts_006_support_union",
        "dic_dts_004_cosine",
        "dic_dts_004_cosine_invalid",
        "dic_dts_006_cosine",
        "dic_dts_006_cosine_invalid",
        "dic_dts_004_support_iou",
        "dic_dts_004_support_iou_invalid",
        "dic_dts_006_support_iou",
        "dic_dts_006_support_iou_invalid",
    ]
    assert [field.name for field in fields(V4StepResult)] == [
        "global_step_before",
        "completed_updates_after",
        "progress",
        "slot_component",
        "slot_endpoint",
        "slot_multiplier",
        "loss_base",
        "loss_anchor",
        "loss_auxiliary",
        "auxiliary_scale",
        "pre_clip_gradient_norm",
        "post_clip_gradient_norm",
        "slot_audit",
    ]
    assert type(result.global_step_before) is int
    assert type(result.completed_updates_after) is int
    assert type(result.slot_multiplier) is float
    assert result.slot_audit is not None
    assert type(result.slot_audit.cross_diagnostics_expected) is bool
    assert type(result.slot_audit.component) is str
    assert type(result.slot_audit.cap_applied) is bool

    def assert_python(value: object) -> None:
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for nested in value.values():
                assert_python(nested)
            return
        if isinstance(value, list):
            for nested in value:
                assert_python(nested)
            return
        assert value is None or type(value) in (str, int, float, bool)
        if type(value) is float:
            assert math.isfinite(value)

    assert_python(payload)
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    with pytest.raises(FrozenInstanceError):
        result.loss_base = 0.0
    with pytest.raises(FrozenInstanceError):
        result.slot_audit.component = "dic"


def _invalid_case(name: str, kwargs: dict[str, object]) -> None:
    if name == "logits-nonleaf":
        kwargs["logits"] = _logits() * 1.0
    elif name == "optimizer-wrong-owner":
        wrong = _logits()
        kwargs["optimizer"] = _ToyOptimizer(wrong)
    elif name == "protocol-mismatch":
        kwargs["protocol"] = _ToyProtocol("meta")
    elif name == "targets-wrong-shape":
        kwargs["targets"] = torch.ones((5, 2, 4), dtype=torch.float64)
    elif name == "profile-malformed-equality":
        kwargs["profile"] = SimpleNamespace(
            name="v4_full",
            l1_weight=np.array(0.35),
            l2_weight=np.array(1.0),
            use_dic=np.bool_(True),
            use_dts=np.bool_(True),
        )
    elif name == "step-bool":
        kwargs["global_step_before"] = True
    elif name == "step-range":
        kwargs["global_step_before"] = 20
    elif name == "config-bool":
        kwargs["config"] = SimpleNamespace(
            dic_unit_coefficient=True,
            dts_unit_coefficient=3.0,
            dic_max_ratio=0.04,
            dts_max_ratio=0.08,
        )
    elif name == "config-ratio":
        kwargs["config"] = SimpleNamespace(
            dic_unit_coefficient=2.0,
            dts_unit_coefficient=3.0,
            dic_max_ratio=0.05,
            dts_max_ratio=0.08,
        )
    else:
        raise AssertionError(name)


@pytest.mark.parametrize(
    "name",
    [
        "logits-nonleaf",
        "optimizer-wrong-owner",
        "protocol-mismatch",
        "targets-wrong-shape",
        "profile-malformed-equality",
        "step-bool",
        "step-range",
        "config-bool",
        "config-ratio",
    ],
)
def test_static_preflight_failures_happen_before_zero_grad_or_render(name: str) -> None:
    kwargs, _, optimizer, protocol = _kwargs("v4_full", global_step_before=13, total_steps=20)
    _invalid_case(name, kwargs)

    with pytest.raises(ValueError):
        train_one_v4_update(**kwargs)

    assert optimizer.zero_grad_calls == []
    assert optimizer.step_calls == 0
    assert protocol.inputs == []


@pytest.mark.parametrize("failure", ["branch", "guard", "clip-return", "clip-gradient"])
def test_branch_guard_and_clipping_failures_roll_back_without_step(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    kwargs, logits, optimizer, _ = _kwargs("v4_full", global_step_before=13, total_steps=20)
    before = logits.detach().clone()
    state_before = dict(optimizer.state)
    _make_branch_patch(monkeypatch)

    if failure == "branch":
        monkeypatch.setattr(
            trainer_step_module,
            "directional_dic_loss",
            lambda score, bands: score.sum() * 0.0 + float("inf"),
        )
    elif failure == "guard":
        monkeypatch.setattr(
            trainer_step_module,
            "guard_auxiliary_gradient",
            lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("injected guard failure")),
        )
    else:
        def bad_clip(parameters: object, max_norm: float) -> torch.Tensor:
            parameter = list(parameters)[0]  # type: ignore[arg-type]
            if failure == "clip-gradient":
                parameter.grad.fill_(float("nan"))
                return parameter.new_tensor(1.0)
            return parameter.new_tensor(float("inf"))

        monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", bad_clip)

    with pytest.raises((ValueError, RuntimeError), match="finite|guard"):
        train_one_v4_update(**kwargs)

    assert optimizer.step_calls == 0
    assert optimizer.state == state_before
    assert torch.equal(logits.detach(), before)
    assert logits.grad is None


def test_zero_grad_failure_rolls_back_dirty_grad_without_render_or_step() -> None:
    kwargs, logits, _, protocol = _kwargs("v4_full", global_step_before=13, total_steps=20)
    before = logits.detach().clone()

    class _ExplodingZeroGradOptimizer(_ToyOptimizer):
        def zero_grad(self, *, set_to_none: bool = False) -> None:
            super().zero_grad(set_to_none=set_to_none)
            self.parameter.grad = torch.full_like(self.parameter, 7.0)
            raise RuntimeError("injected zero_grad failure")

    optimizer = _ExplodingZeroGradOptimizer(logits)
    kwargs["optimizer"] = optimizer
    state_before = dict(optimizer.state)

    with pytest.raises(RuntimeError, match="zero_grad"):
        train_one_v4_update(**kwargs)

    assert optimizer.zero_grad_calls == [True]
    assert optimizer.step_calls == 0
    assert optimizer.state == state_before
    assert protocol.inputs == []
    assert torch.equal(logits.detach(), before)
    assert logits.grad is None


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else []),
    ids=lambda device: device.type,
)
def test_full_active_step_keeps_gradients_on_one_device_and_no_guard_input_aliasing(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
) -> None:
    kwargs, _, optimizer, _ = _kwargs("v4_full", device=device, global_step_before=15, total_steps=20)
    _make_branch_patch(monkeypatch)
    original_guard = trainer_step_module.guard_auxiliary_gradient
    storages: list[tuple[int, int, int]] = []

    def record_guard(
        base_grad: torch.Tensor,
        anchor_grad: torch.Tensor,
        aux_grad: torch.Tensor,
        max_ratio: float,
    ) -> GuardedAuxiliaryGradient:
        storages.append((base_grad.data_ptr(), anchor_grad.data_ptr(), aux_grad.data_ptr()))
        assert base_grad.device.type == device.type
        assert anchor_grad.device.type == device.type
        assert aux_grad.device.type == device.type
        return original_guard(base_grad, anchor_grad, aux_grad, max_ratio)

    monkeypatch.setattr(trainer_step_module, "guard_auxiliary_gradient", record_guard)

    result = train_one_v4_update(**kwargs)

    assert optimizer.step_calls == 1
    assert optimizer.step_grad is not None
    assert optimizer.step_grad.device.type == device.type
    assert storages and len(set(storages[0])) == 3
    assert json.loads(json.dumps(asdict(result), allow_nan=False)) == asdict(result)
