"""Behavioral contracts for exactly one soft-loss-v3 optimizer update."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, asdict, fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import shadowforge_pipeline.soft_voxel_v3.trainer as trainer_module
from shadowforge_pipeline.soft_voxel_v3.contours import ContourBands
from shadowforge_pipeline.soft_voxel_v3.losses import (
    V3Profile,
    compute_baseline_branch,
    compute_dts_branch,
    dts_schedule,
    resolve_profile,
)
from shadowforge_pipeline.soft_voxel_v3.thresholding import render_thresholded
from shadowforge_pipeline.soft_voxel_v3.trainer import (
    SoftTrainingStepError,
    StepRecord,
    train_one_update,
)


class _CountingProtocol:
    """Differentiable renderer boundary double with exact input capture."""

    def __init__(self, device: torch.device, *, scale: float = 1.0) -> None:
        self.device = device
        self.scale = scale
        self.inputs: list[torch.Tensor] = []

    def render_all(self, density: torch.Tensor) -> torch.Tensor:
        self.inputs.append(density)
        projection = density[0, 0].mean(dim=0) * self.scale
        return projection.unsqueeze(0).expand(6, -1, -1)


class _CountingOptimizer:
    """Small optimizer double that performs a real SGD parameter update."""

    def __init__(self, parameter: torch.Tensor, *, lr: float = 0.1) -> None:
        self.parameter = parameter
        self.lr = lr
        self.zero_grad_calls: list[bool] = []
        self.step_calls = 0
        self.step_grad: torch.Tensor | None = None

    def zero_grad(self, *, set_to_none: bool = False) -> None:
        self.zero_grad_calls.append(set_to_none)
        if set_to_none:
            self.parameter.grad = None
        elif self.parameter.grad is not None:
            self.parameter.grad.zero_()

    def step(self) -> None:
        self.step_calls += 1
        assert self.parameter.grad is not None
        self.step_grad = self.parameter.grad.detach().clone()
        with torch.no_grad():
            self.parameter.add_(self.parameter.grad, alpha=-self.lr)


def _logits(device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(
        [-1.4, -0.8, -0.2, 0.1, 0.4, 0.7, 1.0, 1.3],
        device=device,
        dtype=torch.float64,
    ).reshape(1, 1, 2, 2, 2).requires_grad_()


def _targets(device: torch.device | str = "cpu") -> torch.Tensor:
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device, dtype=torch.float64)
    return target.unsqueeze(0).repeat(6, 1, 1)


def _bands(device: torch.device | str = "cpu") -> ContourBands:
    target_binary = _targets(device) >= 0.5
    return ContourBands(inner=target_binary, outer=~target_binary)


def _kwargs(
    profile_name: str,
    *,
    device: torch.device | str = "cpu",
    completed_updates: int = 8,
    total_steps: int = 10,
    bands: object | None = None,
    scale: float = 1.0,
) -> tuple[dict[str, object], torch.Tensor, _CountingOptimizer, _CountingProtocol]:
    resolved_device = torch.device(device)
    logits = _logits(resolved_device)
    optimizer = _CountingOptimizer(logits)
    protocol = _CountingProtocol(resolved_device, scale=scale)
    supplied_bands = _bands(resolved_device) if bands is None else bands
    return (
        {
            "logits": logits,
            "optimizer": optimizer,
            "protocol": protocol,
            "targets": _targets(resolved_device),
            "bands": supplied_bands,
            "profile": resolve_profile(profile_name),
            "completed_updates": completed_updates,
            "total_steps": total_steps,
            "lambda_dic": 0.2,
            "lambda_dts": 0.3,
        },
        logits,
        optimizer,
        protocol,
    )


@pytest.mark.parametrize(
    ("profile_name", "expected_renders", "has_dic", "has_dts"),
    [
        ("l12_bal", 1, False, False),
        ("dic_only", 2, True, False),
        ("dts_only", 2, False, True),
        ("v3_full", 3, True, True),
    ],
)
def test_fully_active_profiles_render_each_selected_branch_once(
    profile_name: str, expected_renders: int, has_dic: bool, has_dts: bool
) -> None:
    """Break caught: a branch is omitted, duplicated, or rendered while disabled."""

    kwargs, logits, optimizer, protocol = _kwargs(profile_name)
    before = logits.detach().clone()

    record = train_one_update(**kwargs)

    assert len(protocol.inputs) == expected_renders
    assert optimizer.zero_grad_calls == [True]
    assert optimizer.step_calls == 1
    assert not torch.equal(logits, before)
    assert (record.loss_dic is not None) is has_dic
    assert (record.loss_dts is not None) is has_dts
    assert (record.dts_endpoint is not None) is has_dts


def test_full_update_uses_one_fresh_sigmoid_tensor_and_graph_per_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: any rendered branch receives a reused or wrong sigmoid graph."""

    kwargs, logits, _, protocol = _kwargs("v3_full")
    original_sigmoid = torch.sigmoid
    original_render_thresholded = trainer_module.render_thresholded
    sigmoid_outputs: list[torch.Tensor] = []
    threshold_calls: list[tuple[torch.Tensor, float]] = []

    def recording_sigmoid(value: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        output = original_sigmoid(value, *args, **kwargs)
        if value is logits:
            sigmoid_outputs.append(output)
        return output

    def recording_render_thresholded(
        selected_protocol: object, density: torch.Tensor, tau: float
    ):
        threshold_calls.append((density, tau))
        return original_render_thresholded(selected_protocol, density, tau)

    monkeypatch.setattr(torch, "sigmoid", recording_sigmoid)
    monkeypatch.setattr(
        trainer_module, "render_thresholded", recording_render_thresholded
    )

    train_one_update(**kwargs)

    assert len(sigmoid_outputs) == 3
    assert len({id(value) for value in sigmoid_outputs}) == 3
    assert all(value.grad_fn is not None for value in sigmoid_outputs)
    assert protocol.inputs[0] is sigmoid_outputs[0]
    assert len(threshold_calls) == 2
    assert threshold_calls[0][0] is sigmoid_outputs[1]
    assert threshold_calls[0][1] == 0.05
    assert threshold_calls[1][0] is sigmoid_outputs[2]
    assert threshold_calls[1][1] == 0.04


def test_one_zero_one_clip_at_limit_one_and_one_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: gradients are cleared, clipped, or applied component-wise."""

    kwargs, _, optimizer, _ = _kwargs("v3_full")
    original_clip = torch.nn.utils.clip_grad_norm_
    clip_calls: list[tuple[list[torch.Tensor], float]] = []

    def recording_clip(parameters: object, max_norm: float, *args: object, **kwargs: object):
        parameter_list = list(parameters)  # type: ignore[arg-type]
        clip_calls.append((parameter_list, max_norm))
        return original_clip(parameter_list, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)

    train_one_update(**kwargs)

    assert optimizer.zero_grad_calls == [True]
    assert optimizer.step_calls == 1
    assert len(clip_calls) == 1
    assert clip_calls[0][0] == [kwargs["logits"]]
    assert clip_calls[0][1] == 1.0


def test_every_backward_uses_default_nonretained_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: any branch retains its renderer graph after backward."""

    kwargs, _, _, _ = _kwargs("v3_full")
    original_backward = torch.Tensor.backward
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def recording_backward(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> None:
        calls.append((args, kwargs))
        original_backward(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", recording_backward)

    train_one_update(**kwargs)

    assert calls == [((), {}), ((), {}), ((), {})]


@pytest.mark.parametrize(
    ("profile_name", "completed_updates", "total_steps", "endpoint"),
    [
        ("dts_only", 8, 10, 0.04),
        ("dts_only", 9, 11, 0.06),
        ("v3_zero_margin", 8, 10, 0.05),
        ("v3_zero_margin", 9, 11, 0.05),
    ],
)
def test_progress_step_identities_and_zero_based_endpoint_selection(
    profile_name: str, completed_updates: int, total_steps: int, endpoint: float
) -> None:
    """Break caught: progress or DTS parity is based on one-based/next state."""

    kwargs, _, _, _ = _kwargs(
        profile_name,
        completed_updates=completed_updates,
        total_steps=total_steps,
    )

    record = train_one_update(**kwargs)

    assert record.global_step_before == completed_updates
    assert record.completed_updates_after == completed_updates + 1
    assert record.progress == (completed_updates + 1) / float(total_steps)
    assert record.dts_endpoint == endpoint


def test_selected_dts_gradient_is_not_halved() -> None:
    """Break caught: the per-step selected DTS endpoint is multiplied by 0.5."""

    kwargs, logits, _, protocol = _kwargs("dts_only")
    kwargs["lambda_dts"] = 0.02
    probabilities = torch.tensor(
        [0.02, 0.03, 0.045, 0.055, 0.065, 0.08, 0.025, 0.07],
        dtype=logits.dtype,
    ).reshape_as(logits)
    with torch.no_grad():
        logits.copy_(torch.logit(probabilities))
    before = logits.detach().clone()

    expected_logits = before.clone().requires_grad_()
    expected_protocol = _CountingProtocol(torch.device("cpu"))
    profile = resolve_profile("dts_only")
    base = compute_baseline_branch(
        expected_protocol.render_all(torch.sigmoid(expected_logits)),
        _targets(),
        profile,
    )
    (base_gradient,) = torch.autograd.grad(base, expected_logits)
    readout = render_thresholded(expected_protocol, torch.sigmoid(expected_logits), 0.04)
    dts = compute_dts_branch(readout.score, _targets() >= 0.5)
    (dts_gradient,) = torch.autograd.grad(dts, expected_logits)
    full_gradient = base_gradient + 0.02 * dts_schedule(0.9) * dts_gradient
    half_gradient = base_gradient + 0.5 * 0.02 * dts_schedule(0.9) * dts_gradient

    def clipped(gradient: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.vector_norm(gradient)
        return gradient * min(1.0, 1.0 / float(norm.item()))

    expected_full = before - 0.1 * clipped(full_gradient)
    expected_half = before - 0.1 * clipped(half_gradient)

    train_one_update(**kwargs)

    assert len(protocol.inputs) == 2
    assert torch.allclose(logits, expected_full, rtol=0.0, atol=1e-12)
    assert not torch.allclose(logits, expected_half, rtol=0.0, atol=1e-9)


def test_actual_parameter_change_and_pre_post_clip_norms_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: norm telemetry is post-clip twice or the clipped grad is not stepped."""

    kwargs, logits, optimizer, _ = _kwargs("l12_bal", scale=20.0)
    before = logits.detach().clone()
    original_clip = torch.nn.utils.clip_grad_norm_
    captured_pre_clip_norms: list[float] = []

    def capturing_clip(parameters: object, max_norm: float, *args: object, **kwargs: object):
        parameter_list = list(parameters)  # type: ignore[arg-type]
        gradient = parameter_list[0].grad
        assert gradient is not None
        captured_pre_clip_norms.append(float(torch.linalg.vector_norm(gradient).item()))
        return original_clip(parameter_list, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capturing_clip)

    record = train_one_update(**kwargs)

    assert optimizer.step_grad is not None
    observed_post = float(torch.linalg.vector_norm(optimizer.step_grad).item())
    assert len(captured_pre_clip_norms) == 1
    assert record.pre_clip_grad_norm == pytest.approx(
        captured_pre_clip_norms[0], rel=0.0, abs=1e-12
    )
    assert captured_pre_clip_norms[0] > 1.0
    assert record.post_clip_grad_norm == pytest.approx(observed_post, abs=1e-12)
    assert record.post_clip_grad_norm == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(logits, before - optimizer.lr * optimizer.step_grad)


def test_fully_active_record_reports_exact_observed_losses_endpoint_and_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: active telemetry swaps, weights, fabricates, or retains branch loss."""

    kwargs, _, _, _ = _kwargs("v3_full", completed_updates=8, total_steps=10)
    original_base = trainer_module.compute_baseline_branch
    original_dic = trainer_module.compute_dic_branch
    original_dts = trainer_module.compute_dts_branch
    observed: dict[str, float] = {}

    def recording_base(*args: object, **kwargs: object) -> torch.Tensor:
        loss = original_base(*args, **kwargs)
        observed["base"] = float(loss.detach().item())
        return loss

    def recording_dic(*args: object, **kwargs: object) -> torch.Tensor:
        loss = original_dic(*args, **kwargs)
        observed["dic"] = float(loss.detach().item())
        return loss

    def recording_dts(*args: object, **kwargs: object) -> torch.Tensor:
        loss = original_dts(*args, **kwargs)
        observed["dts"] = float(loss.detach().item())
        return loss

    monkeypatch.setattr(trainer_module, "compute_baseline_branch", recording_base)
    monkeypatch.setattr(trainer_module, "compute_dic_branch", recording_dic)
    monkeypatch.setattr(trainer_module, "compute_dts_branch", recording_dts)

    record = train_one_update(**kwargs)
    payload = asdict(record)

    assert observed.keys() == {"base", "dic", "dts"}
    assert record.loss_base == observed["base"]
    assert record.loss_dic == observed["dic"]
    assert record.loss_dts == observed["dts"]
    assert record.dts_endpoint == 0.04
    assert record.dic_scale == 0.2
    assert record.dts_scale == 0.3
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_step_record_is_exact_frozen_json_safe_graph_free_schema() -> None:
    """Break caught: telemetry retains tensors/graphs, fake zeros, or mutable state."""

    kwargs, _, _, _ = _kwargs("l12_bal", bands=object())
    record = train_one_update(**kwargs)
    payload = asdict(record)

    assert [field.name for field in fields(StepRecord)] == [
        "global_step_before",
        "completed_updates_after",
        "progress",
        "dts_endpoint",
        "loss_base",
        "loss_dic",
        "loss_dts",
        "dic_scale",
        "dts_scale",
        "pre_clip_grad_norm",
        "post_clip_grad_norm",
    ]
    assert type(record.global_step_before) is int
    assert type(record.completed_updates_after) is int
    for name, value in payload.items():
        assert not isinstance(value, torch.Tensor), name
        assert value is None or type(value) in (int, float), name
        if type(value) is float:
            assert math.isfinite(value)
    assert record.loss_dic is None
    assert record.loss_dts is None
    assert record.dts_endpoint is None
    assert record.dic_scale == 0.0
    assert record.dts_scale == 0.0
    json.dumps(payload, allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        record.loss_base = 99.0


@pytest.mark.parametrize("profile_name", ["l12_bal", "dts_only"])
def test_dic_inactive_profiles_never_touch_supplied_poison_bands(profile_name: str) -> None:
    """Break caught: DIC artifacts are validated or accessed for DIC-inactive profiles."""

    kwargs, _, _, protocol = _kwargs(profile_name, bands=object())

    record = train_one_update(**kwargs)

    assert record.loss_dic is None
    assert record.dic_scale == 0.0
    assert len(protocol.inputs) == (1 if profile_name == "l12_bal" else 2)


def test_dic_schedule_inactive_allows_no_bands_but_active_requires_them_before_dic_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: bands are eagerly required or missing bands reach DIC rendering."""

    inactive_kwargs, _, inactive_optimizer, inactive_protocol = _kwargs(
        "dic_only", completed_updates=0, bands=object()
    )
    inactive_kwargs["bands"] = None
    inactive = train_one_update(**inactive_kwargs)
    assert inactive.loss_dic is None
    assert inactive.dic_scale == 0.0
    assert inactive_optimizer.step_calls == 1
    assert len(inactive_protocol.inputs) == 1

    active_kwargs, _, active_optimizer, active_protocol = _kwargs(
        "dic_only", completed_updates=8, bands=object()
    )
    active_kwargs["bands"] = None
    clip_calls = 0

    def unexpected_clip(*args: object, **kwargs: object) -> None:
        nonlocal clip_calls
        clip_calls += 1

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", unexpected_clip)
    with pytest.raises(ValueError, match="bands"):
        train_one_update(**active_kwargs)

    assert active_optimizer.zero_grad_calls == [True]
    assert active_optimizer.step_calls == 0
    assert len(active_protocol.inputs) == 1
    assert clip_calls == 0


@pytest.mark.parametrize(
    "invalid_bands",
    [
        object(),
        ContourBands(
            inner=torch.zeros((6, 2, 2), dtype=torch.bool),
            outer=torch.ones((6, 2, 2), dtype=torch.bool),
        ),
        ContourBands(
            inner=torch.ones((6, 1, 1), dtype=torch.bool),
            outer=torch.ones((6, 1, 1), dtype=torch.bool),
        ),
    ],
)
def test_active_malformed_bands_fail_before_dic_render(invalid_bands: object) -> None:
    """Break caught: malformed DIC support is discovered only after a DIC render."""

    kwargs, _, optimizer, protocol = _kwargs("dic_only", bands=invalid_bands)

    with pytest.raises(ValueError):
        train_one_update(**kwargs)

    assert optimizer.step_calls == 0
    assert len(protocol.inputs) == 1


class _FiniteLossNonfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        return value.sum() * 0.0

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> torch.Tensor:
        return torch.full_like(grad_output.expand(6, 2, 2), float("nan"))


@pytest.mark.parametrize("failure", ["base", "dic", "dts", "gradient"])
def test_nonfinite_branch_or_accumulated_gradient_fails_without_clip_or_step(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: nonfinite training state mutates parameters or reaches clipping."""

    kwargs, logits, optimizer, _ = _kwargs("v3_full")
    before_bytes = logits.detach().cpu().numpy().tobytes()

    if failure == "base":
        monkeypatch.setattr(
            trainer_module,
            "compute_baseline_branch",
            lambda projection, targets, profile: projection.sum() * 0.0 + float("inf"),
        )
    elif failure == "dic":
        monkeypatch.setattr(
            trainer_module,
            "compute_dic_branch",
            lambda score, bands: score.sum() * 0.0 + float("inf"),
        )
    elif failure == "dts":
        monkeypatch.setattr(
            trainer_module,
            "compute_dts_branch",
            lambda score, target: score.sum() * 0.0 + float("inf"),
        )
    else:
        monkeypatch.setattr(
            trainer_module,
            "compute_baseline_branch",
            lambda projection, targets, profile: _FiniteLossNonfiniteGradient.apply(
                projection
            ),
        )

    clip_calls = 0

    def unexpected_clip(*args: object, **kwargs: object) -> None:
        nonlocal clip_calls
        clip_calls += 1

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", unexpected_clip)

    with pytest.raises(SoftTrainingStepError, match="finite"):
        train_one_update(**kwargs)

    assert optimizer.step_calls == 0
    assert clip_calls == 0
    assert logits.detach().cpu().numpy().tobytes() == before_bytes


@pytest.mark.parametrize("failure", ["returned_norm", "clipped_gradient"])
def test_nonfinite_clipping_result_fails_without_optimizer_step(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a nonfinite clip result is reported or applied as a valid update."""

    kwargs, logits, optimizer, _ = _kwargs("l12_bal")
    before_bytes = logits.detach().cpu().numpy().tobytes()

    def bad_clip(parameters: object, max_norm: float) -> torch.Tensor:
        parameter = list(parameters)[0]  # type: ignore[arg-type]
        if failure == "clipped_gradient":
            parameter.grad.fill_(float("nan"))
            return parameter.new_tensor(1.0)
        return parameter.new_tensor(float("inf"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", bad_clip)

    with pytest.raises(SoftTrainingStepError, match="finite"):
        train_one_update(**kwargs)

    assert optimizer.step_calls == 0
    assert logits.detach().cpu().numpy().tobytes() == before_bytes


def test_protocol_preflight_prefers_matching_camera_state_over_stale_device_field() -> None:
    """Break caught: a stale generic protocol.device overrides the real camera device."""

    kwargs, _, optimizer, protocol = _kwargs("v3_full")
    protocol.device = torch.device("meta")
    protocol.cameras = SimpleNamespace(R=torch.empty(1, device="cpu"))

    train_one_update(**kwargs)

    assert optimizer.zero_grad_calls == [True]
    assert len(protocol.inputs) == 3
    assert optimizer.step_calls == 1


def test_protocol_preflight_rejects_foreign_camera_before_any_side_effect() -> None:
    """Break caught: matching generic device hides foreign renderer/camera state."""

    kwargs, _, optimizer, protocol = _kwargs("l12_bal")
    protocol.cameras = SimpleNamespace(R=torch.empty(1, device="meta"))

    with pytest.raises(ValueError, match="device"):
        train_one_update(**kwargs)

    assert optimizer.zero_grad_calls == []
    assert protocol.inputs == []
    assert optimizer.step_calls == 0


def test_equality_equivalent_malformed_profile_fails_before_any_side_effect() -> None:
    """Break caught: NumPy arrays/non-bool flags pass registered-row equality."""

    kwargs, _, optimizer, protocol = _kwargs("l12_bal")
    malformed = V3Profile(
        "l12_bal",
        np.array(0.35),
        np.array(1.0),
        np.bool_(False),
        np.bool_(False),
        None,
    )
    assert malformed == resolve_profile("l12_bal")
    kwargs["profile"] = malformed

    with pytest.raises(ValueError, match="profile"):
        train_one_update(**kwargs)

    assert optimizer.zero_grad_calls == []
    assert protocol.inputs == []
    assert optimizer.step_calls == 0


def _invalid_case(name: str, kwargs: dict[str, object]) -> None:
    if name == "logits-not-tensor":
        kwargs["logits"] = []
    elif name == "logits-empty":
        kwargs["logits"] = torch.empty((1, 1, 0, 0, 0), requires_grad=True)
    elif name == "logits-nonfinite":
        value = _logits()
        with torch.no_grad():
            value[0, 0, 0, 0, 0] = float("nan")
        kwargs["logits"] = value
    elif name == "logits-integral":
        kwargs["logits"] = torch.ones((1, 1, 2, 2, 2), dtype=torch.int64)
    elif name == "logits-no-grad":
        kwargs["logits"] = torch.ones((1, 1, 2, 2, 2))
    elif name == "logits-nonleaf":
        kwargs["logits"] = _logits() * 1.0
    elif name == "logits-wrong-shape":
        kwargs["logits"] = torch.ones((1, 1, 2, 2, 3), requires_grad=True)
    elif name == "optimizer-no-zero":
        kwargs["optimizer"] = SimpleNamespace(zero_grad=None, step=lambda: None)
    elif name == "optimizer-no-step":
        kwargs["optimizer"] = SimpleNamespace(zero_grad=lambda **unused: None, step=None)
    elif name == "protocol-no-render":
        kwargs["protocol"] = SimpleNamespace(device=torch.device("cpu"), render_all=None)
    elif name == "protocol-device-mismatch":
        kwargs["protocol"] = SimpleNamespace(
            device=torch.device("meta"), render_all=lambda density: None
        )
    elif name == "targets-not-tensor":
        kwargs["targets"] = []
    elif name == "targets-integral":
        kwargs["targets"] = torch.ones((6, 2, 2), dtype=torch.int64)
    elif name == "targets-nonfinite":
        target = _targets()
        target[0, 0, 0] = float("inf")
        kwargs["targets"] = target
    elif name == "targets-wrong-shape":
        kwargs["targets"] = torch.ones((5, 2, 2))
    elif name == "targets-device-mismatch":
        kwargs["targets"] = torch.empty((6, 2, 2), device="meta")
    elif name == "profile-wrong-type":
        kwargs["profile"] = object()
    elif name == "profile-bad-thresholds":
        kwargs["profile"] = V3Profile("v3_full", 0.35, 1.0, True, True, (0.04,))
    elif name == "profile-inconsistent-flags":
        kwargs["profile"] = V3Profile("l12_bal", 0.35, 1.0, False, True, None)
    elif name == "profile-unregistered-state":
        kwargs["profile"] = V3Profile("l12_bal", 0.5, 1.0, False, False, None)
    elif name == "completed-bool":
        kwargs["completed_updates"] = True
    elif name == "completed-negative":
        kwargs["completed_updates"] = -1
    elif name == "completed-fractional":
        kwargs["completed_updates"] = 1.5
    elif name == "completed-out-of-range":
        kwargs["completed_updates"] = 10
    elif name == "total-bool":
        kwargs["total_steps"] = True
    elif name == "total-nonpositive":
        kwargs["total_steps"] = 0
    elif name == "total-fractional":
        kwargs["total_steps"] = 10.5
    elif name == "lambda-dic-negative":
        kwargs["lambda_dic"] = -0.1
    elif name == "lambda-dic-nonfinite":
        kwargs["lambda_dic"] = float("nan")
    elif name == "lambda-dts-bool":
        kwargs["lambda_dts"] = True
    elif name == "lambda-dts-nonfinite":
        kwargs["lambda_dts"] = float("inf")
    else:
        raise AssertionError(name)


@pytest.mark.parametrize(
    "name",
    [
        "logits-not-tensor",
        "logits-empty",
        "logits-nonfinite",
        "logits-integral",
        "logits-no-grad",
        "logits-nonleaf",
        "logits-wrong-shape",
        "optimizer-no-zero",
        "optimizer-no-step",
        "protocol-no-render",
        "protocol-device-mismatch",
        "targets-not-tensor",
        "targets-integral",
        "targets-nonfinite",
        "targets-wrong-shape",
        "targets-device-mismatch",
        "profile-wrong-type",
        "profile-bad-thresholds",
        "profile-inconsistent-flags",
        "profile-unregistered-state",
        "completed-bool",
        "completed-negative",
        "completed-fractional",
        "completed-out-of-range",
        "total-bool",
        "total-nonpositive",
        "total-fractional",
        "lambda-dic-negative",
        "lambda-dic-nonfinite",
        "lambda-dts-bool",
        "lambda-dts-nonfinite",
    ],
)
def test_static_preflight_fails_before_zero_grad_or_render(name: str) -> None:
    """Break caught: malformed static inputs produce any optimizer/renderer side effect."""

    kwargs, _, optimizer, protocol = _kwargs("v3_full")
    _invalid_case(name, kwargs)

    with pytest.raises(ValueError):
        train_one_update(**kwargs)

    assert optimizer.zero_grad_calls == []
    assert optimizer.step_calls == 0
    assert protocol.inputs == []


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")]
    + ([torch.device("cuda")] if torch.cuda.is_available() else []),
    ids=lambda device: device.type,
)
def test_finite_update_backward_smoke_on_available_devices(device: torch.device) -> None:
    """Break caught: the kernel assumes CPU or fails to backpropagate on CUDA."""

    kwargs, logits, optimizer, _ = _kwargs("v3_full", device=device)

    record = train_one_update(**kwargs)

    assert optimizer.step_calls == 1
    assert optimizer.step_grad is not None
    assert torch.isfinite(logits).all().item()
    assert torch.isfinite(optimizer.step_grad).all().item()
    assert math.isfinite(record.pre_clip_grad_norm)
    assert math.isfinite(record.post_clip_grad_norm)


def test_public_trainer_api_is_exported_from_v3_package() -> None:
    """Break caught: Task 8 cannot import the frozen Task 7 kernel API."""

    import shadowforge_pipeline.soft_voxel_v3 as v3

    assert v3.StepRecord is StepRecord
    assert v3.SoftTrainingStepError is SoftTrainingStepError
    assert v3.train_one_update is train_one_update
