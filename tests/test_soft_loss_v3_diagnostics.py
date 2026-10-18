import json
import math
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, asdict
from fractions import Fraction

import pytest
import torch

from shadowforge_pipeline.soft_voxel_v3.diagnostics import (
    ComponentGradientSample,
    measure_component_gradients,
)


def _linear_builder(logits, gradient, calls=None, name=None):
    gradient = torch.as_tensor(gradient, dtype=logits.dtype, device=logits.device)

    def build():
        if calls is not None:
            calls[name] = calls.get(name, 0) + 1
        return (logits * gradient).sum()

    return build


def _valid_kwargs(logits):
    return {
        "completed_updates": 8,
        "base_builder": _linear_builder(logits, [1.0, 0.0]),
        "dic_builder": None,
        "dts_builders": {},
        "dic_scale": 0.0,
        "dts_scale": 0.0,
        "selected_dts_tau": None,
    }


def _assert_python_storage(value):
    assert not isinstance(value, torch.Tensor)
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, (str, int, float, bool))
            _assert_python_storage(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_python_storage(item)
    else:
        assert value is None or isinstance(value, (str, int, float, bool))


def test_exact_norms_all_pairwise_cosines_support_and_weighted_ratios():
    """Break caught: a raw vector, pair, support, or scale is misreported."""
    logits = torch.zeros(4, dtype=torch.float64, requires_grad=True)

    sample = measure_component_gradients(
        logits,
        completed_updates=8,
        base_builder=_linear_builder(logits, [3.0, 0.0, 0.0, 0.0]),
        dic_builder=_linear_builder(logits, [4.0, 0.0, 0.0, 0.0]),
        dts_builders={
            0.04: _linear_builder(logits, [-2.0, 0.0, 0.0, 0.0]),
            0.06: _linear_builder(logits, [0.0, 5.0, 0.0, 0.0]),
        },
        dic_scale=0.5,
        dts_scale=0.25,
        selected_dts_tau=0.04,
    )

    assert sample.completed_updates == 8
    assert sample.base_norm == 3.0
    assert sample.dic_norm == 4.0
    assert sample.dts_norms == {0.04: 2.0, 0.06: 5.0}
    assert sample.base_dic_cosine == 1.0
    assert sample.base_dic_cosine_invalid is False
    assert sample.base_dts_cosines == {0.04: -1.0, 0.06: 0.0}
    assert sample.base_dts_cosine_invalid == {0.04: False, 0.06: False}
    assert sample.dic_dts_cosines == {0.04: -1.0, 0.06: 0.0}
    assert sample.dic_dts_cosine_invalid == {0.04: False, 0.06: False}
    assert sample.dts_pair_cosines == {0.04: {0.06: 0.0}}
    assert sample.dts_pair_cosine_invalid == {0.04: {0.06: False}}
    assert sample.dic_dts_support_ious == {0.04: 1.0, 0.06: 0.0}
    assert sample.dic_dts_support_iou_invalid == {0.04: False, 0.06: False}
    assert sample.weighted_dic_base_ratio == pytest.approx(2.0 / 3.0)
    assert sample.weighted_dic_base_ratio_invalid is False
    assert sample.weighted_dts_base_ratios == pytest.approx(
        {0.04: 1.0 / 6.0, 0.06: 5.0 / 12.0}
    )
    assert sample.weighted_dts_base_ratio_invalid == {0.04: False, 0.06: False}
    assert sample.selected_dts_tau == 0.04
    assert sample.combined_aux_base_ratio == 0.5
    assert sample.combined_aux_base_ratio_invalid is False
    assert sample.base_combined_aux_cosine == 1.0
    assert sample.base_combined_aux_cosine_invalid is False


def test_empty_support_union_and_zero_norm_cosines_are_explicitly_invalid():
    """Break caught: empty support or zero vectors are silently reported as zero."""
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    sample = measure_component_gradients(
        logits,
        completed_updates=8,
        base_builder=_linear_builder(logits, [1.0, 0.0]),
        dic_builder=_linear_builder(logits, [0.0, 0.0]),
        dts_builders={0.04: _linear_builder(logits, [0.0, 0.0])},
        dic_scale=1.0,
        dts_scale=1.0,
        selected_dts_tau=0.04,
    )

    assert sample.dic_dts_support_ious == {0.04: None}
    assert sample.dic_dts_support_iou_invalid == {0.04: True}
    assert sample.base_dic_cosine is None
    assert sample.base_dic_cosine_invalid is True
    assert sample.base_dts_cosines == {0.04: None}
    assert sample.base_dts_cosine_invalid == {0.04: True}
    assert sample.dic_dts_cosines == {0.04: None}
    assert sample.dic_dts_cosine_invalid == {0.04: True}
    assert sample.combined_aux_base_ratio == 0.0
    assert sample.combined_aux_base_ratio_invalid is False
    assert sample.base_combined_aux_cosine is None
    assert sample.base_combined_aux_cosine_invalid is True


def test_support_uses_strict_configured_epsilon_boundary():
    """Break caught: support uses >= or ignores the supplied epsilon."""
    logits = torch.zeros(3, dtype=torch.float64, requires_grad=True)

    sample = measure_component_gradients(
        logits,
        completed_updates=8,
        base_builder=_linear_builder(logits, [1.0, 0.0, 0.0]),
        dic_builder=_linear_builder(logits, [1e-3, 2e-3, 0.0]),
        dts_builders={0.04: _linear_builder(logits, [1e-3, 0.0, 3e-3])},
        dic_scale=1.0,
        dts_scale=1.0,
        selected_dts_tau=0.04,
        support_epsilon=1e-3,
    )

    # The exactly-epsilon first entry is inactive: supports are {1} and {2}.
    assert sample.dic_dts_support_ious == {0.04: 0.0}
    assert sample.dic_dts_support_iou_invalid == {0.04: False}


def test_even_and_odd_selected_endpoints_use_vector_sum_and_exclude_nonselected():
    """Break caught: norms are added, or both DTS endpoints enter one update."""
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    def measure(completed_updates, selected):
        return measure_component_gradients(
            logits,
            completed_updates=completed_updates,
            base_builder=_linear_builder(logits, [1.0, 0.0]),
            dic_builder=_linear_builder(logits, [1.0, 0.0]),
            dts_builders={
                0.04: _linear_builder(logits, [-1.0, 0.0]),
                0.06: _linear_builder(logits, [0.0, 4.0]),
            },
            dic_scale=1.0,
            dts_scale=1.0,
            selected_dts_tau=selected,
        )

    even = measure(8, 0.04)
    odd = measure(9, 0.06)

    assert even.selected_dts_tau == 0.04
    assert even.combined_aux_base_ratio == 0.0
    assert even.base_combined_aux_cosine is None
    assert even.base_combined_aux_cosine_invalid is True
    assert odd.selected_dts_tau == 0.06
    assert odd.combined_aux_base_ratio == pytest.approx(math.sqrt(17.0))
    assert odd.base_combined_aux_cosine == pytest.approx(1.0 / math.sqrt(17.0))
    assert odd.base_combined_aux_cosine_invalid is False


@pytest.mark.parametrize("completed_updates", [8, 9])
def test_zero_margin_selects_only_tau_005_on_both_parities(completed_updates):
    """Break caught: diagnostics infer primary parity instead of using selection."""
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    sample = measure_component_gradients(
        logits,
        completed_updates=completed_updates,
        base_builder=_linear_builder(logits, [2.0, 0.0]),
        dic_builder=None,
        dts_builders={0.05: _linear_builder(logits, [0.0, 3.0])},
        dic_scale=0.0,
        dts_scale=0.5,
        selected_dts_tau=0.05,
    )

    assert sample.selected_dts_tau == 0.05
    assert sample.combined_aux_base_ratio == 0.75
    assert sample.base_combined_aux_cosine == 0.0
    assert sample.base_combined_aux_cosine_invalid is False


def test_dic_dts_and_both_inactive_have_no_fake_component_records():
    """Break caught: an inactive branch is represented by a fake zero gradient."""
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    dic_inactive = measure_component_gradients(
        logits,
        completed_updates=8,
        base_builder=_linear_builder(logits, [2.0, 0.0]),
        dic_builder=None,
        dts_builders={0.04: _linear_builder(logits, [0.0, 2.0])},
        dic_scale=0.0,
        dts_scale=0.5,
        selected_dts_tau=0.04,
    )
    dts_inactive = measure_component_gradients(
        logits,
        completed_updates=8,
        base_builder=_linear_builder(logits, [2.0, 0.0]),
        dic_builder=_linear_builder(logits, [1.0, 0.0]),
        dts_builders={},
        dic_scale=0.5,
        dts_scale=0.0,
        selected_dts_tau=None,
    )
    both_inactive = measure_component_gradients(
        logits,
        **_valid_kwargs(logits),
    )

    assert dic_inactive.dic_norm is None
    assert dic_inactive.dic_dts_cosines == {}
    assert dic_inactive.dic_dts_support_ious == {}
    assert dic_inactive.combined_aux_base_ratio == 0.5
    assert dts_inactive.dts_norms == {}
    assert dts_inactive.weighted_dts_base_ratios == {}
    assert dts_inactive.combined_aux_base_ratio == 0.25
    assert both_inactive.dic_norm is None
    assert both_inactive.dts_norms == {}
    assert both_inactive.combined_aux_base_ratio == 0.0
    assert both_inactive.combined_aux_base_ratio_invalid is False
    assert both_inactive.base_combined_aux_cosine is None
    assert both_inactive.base_combined_aux_cosine_invalid is True


def test_selected_endpoint_with_zero_dts_scale_remains_a_calibration_record():
    """Break caught: a selected raw calibration endpoint is rejected at zero scale."""
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    sample = measure_component_gradients(
        logits,
        completed_updates=8,
        base_builder=_linear_builder(logits, [1.0, 0.0]),
        dic_builder=None,
        dts_builders={0.04: _linear_builder(logits, [0.0, 2.0])},
        dic_scale=0.0,
        dts_scale=0.0,
        selected_dts_tau=0.04,
    )

    assert sample.dts_norms == {0.04: 2.0}
    assert sample.weighted_dts_base_ratios == {0.04: 0.0}
    assert sample.weighted_dts_base_ratio_invalid == {0.04: False}
    assert sample.combined_aux_base_ratio == 0.0


def test_active_dic_scale_without_builder_fails_before_any_builder_runs():
    """Break caught: an active DIC scale silently omits its missing gradient."""
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    calls = {"base": 0, "dts": 0}

    base_builder = _linear_builder(logits, [1.0, 0.0], calls, "base")
    dts_builder = _linear_builder(logits, [0.0, 1.0], calls, "dts")
    error = None
    try:
        measure_component_gradients(
            logits,
            completed_updates=8,
            base_builder=base_builder,
            dic_builder=None,
            dts_builders={0.04: dts_builder},
            dic_scale=0.25,
            dts_scale=0.0,
            selected_dts_tau=0.04,
        )
    except ValueError as caught:
        error = caught

    assert calls == {"base": 0, "dts": 0}
    assert isinstance(error, ValueError)
    assert "dic_scale" in str(error)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_each_component_uses_one_fresh_graph_without_touching_existing_grad(device_name):
    """Break caught: builders share a graph, run twice, or mutate logits.grad."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    logits = torch.tensor([1.0, -2.0], device=device, dtype=torch.float64, requires_grad=True)
    existing_grad = torch.tensor([7.0, -9.0], device=device, dtype=torch.float64)
    logits.grad = existing_grad
    calls = {}
    graph_nodes = []

    def fresh_builder(name, multiplier):
        def build():
            calls[name] = calls.get(name, 0) + 1
            loss = multiplier * logits.square().sum()
            graph_nodes.append(loss.grad_fn)
            return loss

        return build

    sample = measure_component_gradients(
        logits,
        completed_updates=9,
        base_builder=fresh_builder("base", 1.0),
        dic_builder=fresh_builder("dic", 2.0),
        dts_builders={
            0.04: fresh_builder("dts04", 3.0),
            0.06: fresh_builder("dts06", 4.0),
        },
        dic_scale=0.25,
        dts_scale=0.5,
        selected_dts_tau=0.06,
    )

    assert calls == {"base": 1, "dic": 1, "dts04": 1, "dts06": 1}
    assert len({id(node) for node in graph_nodes}) == 4
    assert logits.grad is existing_grad
    assert torch.equal(logits.grad, torch.tensor([7.0, -9.0], device=device, dtype=torch.float64))
    _assert_python_storage(asdict(sample))
    json.dumps(asdict(sample), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        sample.base_norm = 99.0


def test_every_component_autograd_call_explicitly_disables_graph_retention(monkeypatch):
    """Break caught: any component autograd call retains its fresh graph."""
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    original_autograd_grad = torch.autograd.grad
    retain_graph_values = []

    def recording_autograd_grad(*args, **kwargs):
        retain_graph_values.append(kwargs.get("retain_graph", "not supplied"))
        return original_autograd_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", recording_autograd_grad)
    sample = measure_component_gradients(
        logits,
        completed_updates=8,
        base_builder=_linear_builder(logits, [3.0, 0.0]),
        dic_builder=_linear_builder(logits, [0.0, 4.0]),
        dts_builders={
            0.04: _linear_builder(logits, [-5.0, 0.0]),
            0.06: _linear_builder(logits, [0.0, -6.0]),
        },
        dic_scale=0.5,
        dts_scale=0.25,
        selected_dts_tau=0.04,
    )

    assert retain_graph_values == [False, False, False, False]
    assert sample.base_norm == 3.0
    assert sample.dic_norm == 4.0
    assert sample.dts_norms == {0.04: 5.0, 0.06: 6.0}


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_zero_base_norm_marks_every_ratio_and_base_cosine_invalid(device_name):
    """Break caught: a zero base denominator produces inf, nan, or a fake zero."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    logits = torch.zeros(2, device=device, dtype=torch.float64, requires_grad=True)

    sample = measure_component_gradients(
        logits,
        completed_updates=8,
        base_builder=_linear_builder(logits, [0.0, 0.0]),
        dic_builder=_linear_builder(logits, [1.0, 0.0]),
        dts_builders={0.04: _linear_builder(logits, [0.0, 1.0])},
        dic_scale=1.0,
        dts_scale=1.0,
        selected_dts_tau=0.04,
    )

    assert sample.weighted_dic_base_ratio is None
    assert sample.weighted_dic_base_ratio_invalid is True
    assert sample.weighted_dts_base_ratios == {0.04: None}
    assert sample.weighted_dts_base_ratio_invalid == {0.04: True}
    assert sample.combined_aux_base_ratio is None
    assert sample.combined_aux_base_ratio_invalid is True
    assert sample.base_dic_cosine is None
    assert sample.base_dic_cosine_invalid is True
    assert sample.base_dts_cosines == {0.04: None}
    assert sample.base_combined_aux_cosine is None
    assert sample.base_combined_aux_cosine_invalid is True


class _DuplicateNormalizedMapping(Mapping):
    def __init__(self, builder):
        self.builder = builder

    def __getitem__(self, key):
        return self.builder

    def __iter__(self):
        return iter((0.04, Fraction(1, 25)))

    def __len__(self):
        return 2

    def items(self):
        return ((0.04, self.builder), (Fraction(1, 25), self.builder))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("completed_updates", True, "completed_updates"),
        ("completed_updates", -1, "completed_updates"),
        ("completed_updates", 1.5, "completed_updates"),
        ("base_builder", 1, "base_builder"),
        ("dic_builder", 1, "dic_builder"),
        ("dts_builders", [], "mapping"),
        ("dic_scale", -1.0, "dic_scale"),
        ("dic_scale", float("inf"), "dic_scale"),
        ("dic_scale", True, "dic_scale"),
        ("dts_scale", -1.0, "dts_scale"),
        ("dts_scale", float("nan"), "dts_scale"),
        ("support_epsilon", 0.0, "support_epsilon"),
        ("support_epsilon", -1.0, "support_epsilon"),
        ("support_epsilon", float("inf"), "support_epsilon"),
        ("support_epsilon", True, "support_epsilon"),
    ],
)
def test_scalar_and_callable_preflight_rejects_invalid_inputs(field, value, match):
    """Break caught: malformed scalar/callable inputs reach autograd."""
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    kwargs = _valid_kwargs(logits)
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        measure_component_gradients(logits, **kwargs)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_logits_preflight_rejects_invalid_inputs_on_available_devices(device_name):
    """Break caught: invalid logits enter builders or autograd."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    invalid_logits = [
        [0.0, 0.0],
        torch.tensor([0, 0], device=device),
        torch.tensor([float("nan"), 0.0], device=device, requires_grad=True),
        torch.zeros(2, device=device),
        torch.empty(0, device=device, requires_grad=True),
    ]

    for logits in invalid_logits:
        with pytest.raises(ValueError, match="logits"):
            measure_component_gradients(
                logits,
                completed_updates=0,
                base_builder=lambda: None,
                dic_builder=None,
                dts_builders={},
                dic_scale=0.0,
                dts_scale=0.0,
                selected_dts_tau=None,
            )


def test_dts_mapping_and_selection_preflight_is_fail_closed_before_builders():
    """Break caught: malformed endpoints partially invoke diagnostic builders."""
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    base_calls = {"count": 0}

    def base_builder():
        base_calls["count"] += 1
        return logits.sum()

    invalid_cases = [
        ({"0.04": _linear_builder(logits, [1.0, 0.0])}, 0.0, None, "numeric"),
        ({float("nan"): _linear_builder(logits, [1.0, 0.0])}, 0.0, None, "finite"),
        ({0.04: 1}, 0.0, None, "callable"),
        ({0.04: _linear_builder(logits, [1.0, 0.0])}, 1.0, None, "selected"),
        ({0.04: _linear_builder(logits, [1.0, 0.0])}, 1.0, 0.06, "missing"),
        ({}, 0.0, 0.04, "no DTS builders"),
        (
            _DuplicateNormalizedMapping(_linear_builder(logits, [1.0, 0.0])),
            0.0,
            None,
            "duplicate",
        ),
    ]
    for builders, scale, selected, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            measure_component_gradients(
                logits,
                completed_updates=0,
                base_builder=base_builder,
                dic_builder=None,
                dts_builders=builders,
                dic_scale=0.0,
                dts_scale=scale,
                selected_dts_tau=selected,
            )
    assert base_calls == {"count": 0}


class _WrongShapeGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.sum()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.ones(3, device=grad_output.device, dtype=grad_output.dtype)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("non_tensor", "Tensor"),
        ("non_scalar", "scalar"),
        ("integral", "floating"),
        ("nonfinite_loss", "finite"),
        ("no_grad", "gradient"),
        ("disconnected", "disconnected"),
        ("nonfinite_gradient", "gradient.*finite"),
        ("wrong_shape_gradient", "gradient"),
    ],
)
def test_builder_output_and_gradient_preflight_fails_closed(device_name, case, match):
    """Break caught: an invalid builder result leaks into a partial sample."""
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    device = torch.device(device_name)
    logits = torch.zeros(2, device=device, dtype=torch.float64, requires_grad=True)

    def builder():
        if case == "non_tensor":
            return 1.0
        if case == "non_scalar":
            return logits * 1.0
        if case == "integral":
            return torch.tensor(1, device=device)
        if case == "nonfinite_loss":
            return logits.sum() * 0.0 + float("inf")
        if case == "no_grad":
            return logits.detach().sum()
        if case == "disconnected":
            return torch.ones((), device=device, dtype=torch.float64, requires_grad=True)
        if case == "nonfinite_gradient":
            return torch.sqrt(logits).sum()
        if case == "wrong_shape_gradient":
            return _WrongShapeGradient.apply(logits)
        raise AssertionError(case)

    kwargs = _valid_kwargs(logits)
    kwargs["base_builder"] = builder
    with pytest.raises(ValueError, match=match):
        measure_component_gradients(logits, **kwargs)


def test_builder_internal_exception_type_and_message_are_not_relabelled():
    """Break caught: ordinary builder failures are obscured as contract errors."""
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    def builder():
        raise KeyError("builder sentinel")

    kwargs = _valid_kwargs(logits)
    kwargs["base_builder"] = builder
    with pytest.raises(KeyError, match="builder sentinel"):
        measure_component_gradients(logits, **kwargs)


def test_public_diagnostics_api_is_exported_from_v3_package():
    """Break caught: downstream trainer imports cannot reach the frozen API."""
    import shadowforge_pipeline.soft_voxel_v3 as v3

    assert v3.ComponentGradientSample is ComponentGradientSample
    assert v3.measure_component_gradients is measure_component_gradients
