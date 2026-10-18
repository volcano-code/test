"""Behavioral contracts for the frozen soft-loss-v3 delivery readout."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from shadowforge_pipeline.soft_voxel_v2.protocol import SadekarTFSProtocol as V2Protocol
from shadowforge_pipeline.soft_voxel_v3 import thresholding as thresholding_module
from shadowforge_pipeline.soft_voxel_v3.thresholding import (
    dts_endpoint,
    formal_readout_mask,
    rendered_margin,
    render_thresholded,
    straight_through_density,
)


class _CapturingProtocol:
    """Small real boundary double that records the exact renderer input."""

    def __init__(self, device: torch.device, projection: torch.Tensor) -> None:
        self.device = device
        self._projection = projection
        self.calls = 0
        self.received: torch.Tensor | None = None

    def render_all(self, density: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.received = density
        return self._projection


def test_ste_forward_is_bit_exact_and_backward_is_sigmoid() -> None:
    """Break caught: hard occupancy or its sigmoid-surrogate derivative drifts."""

    density = torch.tensor([0.03, 0.05, 0.08], dtype=torch.float64, requires_grad=True)
    output = straight_through_density(density, 0.05)
    expected = (density.detach() >= 0.05).to(density.dtype) * density.new_tensor(0.999)

    assert torch.equal(output, expected)
    output.sum().backward()
    q = torch.sigmoid((density.detach() - 0.05) / 0.01)
    assert torch.allclose(density.grad, q * (1 - q) / 0.01, atol=1e-12, rtol=1e-12)


def test_zero_projection_is_all_background_and_finite() -> None:
    """Break caught: an empty render gains foreground pixels or nonfinite readout."""

    projection = torch.zeros(6, 8, 8, requires_grad=True)
    maximum, score = rendered_margin(projection)

    assert torch.equal(maximum, torch.full((6, 1, 1), 1e-8))
    assert not formal_readout_mask(projection).any()
    score.sum().backward()
    assert torch.isfinite(projection.grad).all()


def test_endpoint_is_global_step_pure_function() -> None:
    """Break caught: DTS endpoints stop alternating from the global update step."""

    assert [dts_endpoint(i) for i in range(6)] == [0.04, 0.06, 0.04, 0.06, 0.04, 0.06]


@pytest.mark.parametrize("tau", [0.04, 0.05, 0.06])
@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else []),
    ids=lambda device: device.type,
)
def test_renderer_receives_exact_hard_occupancy_at_every_registered_threshold(
    tau: float, device: torch.device
) -> None:
    """Break caught: rendering receives soft or incorrectly scaled occupancy."""

    density = torch.tensor(
        [0.039, 0.04, 0.05, 0.06, 0.061, 0.039, 0.05, 0.061], device=device
    ).reshape(1, 1, 2, 2, 2)
    projection = torch.tensor([[[0.2]]], device=device)
    protocol = _CapturingProtocol(device, projection)

    readout = render_thresholded(protocol, density, tau)

    expected = (density >= tau).to(density.dtype) * density.new_tensor(0.999)
    assert protocol.calls == 1
    assert protocol.received is not None
    assert torch.equal(protocol.received, expected)
    assert torch.equal(readout.occupancy, expected)


def test_readout_mask_matches_v2_and_is_exactly_score_nonnegative() -> None:
    """Break caught: v3's delivery predicate drifts from its formal v2 rule."""

    projection = torch.tensor(
        [[[0.0, 0.4, 0.5], [0.2, 0.1, 0.0]], [[0.6, 0.48, 0.0], [0.3, 0.1, 0.0]]]
    )
    protocol = _CapturingProtocol(torch.device("cpu"), projection)
    density = torch.full((1, 1, 1, 1, 1), 0.05)

    readout = render_thresholded(protocol, density, 0.05)

    assert torch.equal(readout.mask, V2Protocol.binary_mask(readout.projection))
    assert torch.equal(readout.score >= 0, formal_readout_mask(readout.projection))


def test_readout_tie_is_zero_score_and_foreground() -> None:
    """Break caught: the frozen inclusive 0.8-max boundary becomes exclusive."""

    projection = torch.tensor([[[1.0, 0.8]]])
    maximum, score = rendered_margin(projection)

    assert torch.equal(maximum, torch.tensor([[[1.0]]]))
    assert score[0, 0, 1].item() == 0.0
    assert formal_readout_mask(projection)[0, 0, 1].item() is True


@pytest.mark.parametrize(
    "projection, selected",
    [
        (torch.zeros(1, 2, 3), (0, 1, 2)),
        (torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1e-12, 0.0]]]), (0, 1, 1)),
    ],
    ids=["empty", "one-pixel-near-empty"],
)
def test_stopped_max_has_exactly_bounded_one_hot_margin_gradient(
    projection: torch.Tensor, selected: tuple[int, int, int]
) -> None:
    """Break caught: the stopped maximum leaks gradient into any other pixel."""

    projection = projection.requires_grad_()
    _, score = rendered_margin(projection)
    selector = torch.zeros_like(score)
    selector[selected] = 1.0

    (score * selector).sum().backward()

    expected = torch.zeros_like(projection)
    expected[selected] = 1.0
    assert torch.isfinite(projection.grad).all()
    assert torch.equal(projection.grad.abs(), expected)


def test_nonempty_stopped_max_contributes_no_gradient() -> None:
    """Break caught: the non-clamped maximum leaks a negative score gradient."""

    projection = torch.tensor([[[1.0, 0.8]]], requires_grad=True)
    _, score = rendered_margin(projection)

    score[0, 0, 1].backward()

    assert torch.equal(projection.grad, torch.tensor([[[0.0, 1.0]]]))


@pytest.mark.parametrize(
    "density,tau,temperature",
    [
        (torch.tensor([0.05]), -0.01, 0.01),
        (torch.tensor([0.05]), 1.01, 0.01),
        (torch.tensor([0.05]), 0.05, 0.0),
        (torch.tensor([0.05]), 0.05, -0.01),
        (torch.tensor([0], dtype=torch.int64), 0.05, 0.01),
        (torch.tensor([float("nan")]), 0.05, 0.01),
        (torch.tensor([float("inf")]), 0.05, 0.01),
    ],
    ids=["tau-below-range", "tau-above-range", "zero-temperature", "negative-temperature", "integer-density", "nan-density", "infinite-density"],
)
def test_ste_rejects_invalid_threshold_temperature_or_density(
    density: torch.Tensor, tau: float, temperature: float
) -> None:
    """Break caught: invalid threshold inputs silently create undefined occupancy."""

    with pytest.raises(ValueError):
        straight_through_density(density, tau, temperature)


@pytest.mark.parametrize(
    "projection",
    [torch.zeros(1), torch.zeros(1, 2, 3, 4), torch.tensor([[[float("nan")]]]), torch.tensor([[[float("inf")]]])],
    ids=["rank-one", "rank-four", "nan", "infinite"],
)
def test_margin_rejects_invalid_projection(projection: torch.Tensor) -> None:
    """Break caught: malformed renderer output enters the formal delivery readout."""

    with pytest.raises(ValueError):
        rendered_margin(projection)


def test_render_rejects_device_mismatch_before_calling_protocol() -> None:
    """Break caught: a foreign-device density reaches the renderer boundary."""

    density = torch.full((1, 1, 1, 1, 1), 0.05)
    protocol = _CapturingProtocol(torch.device("meta"), torch.ones(1, 1, 1))

    with pytest.raises(ValueError):
        render_thresholded(protocol, density, 0.05)

    assert protocol.calls == 0


def test_render_rejects_invalid_density_before_calling_protocol() -> None:
    """Break caught: invalid density reaches the renderer before preflight fails."""

    protocol = _CapturingProtocol(torch.device("cpu"), torch.ones(1, 1, 1))

    with pytest.raises(ValueError):
        render_thresholded(protocol, torch.tensor([[[[[float("nan")]]]]]), 0.05)

    assert protocol.calls == 0


@pytest.mark.parametrize(
    "density",
    [
        torch.ones(1, 1, 2, 2),
        torch.ones(2, 1, 2, 2, 2),
        torch.ones(1, 2, 2, 2, 2),
        torch.ones(1, 1, 2, 2, 3),
        torch.ones(1, 1, 0, 0, 0),
    ],
    ids=["wrong-rank", "wrong-batch-leading-dimension", "wrong-channel-leading-dimension", "non-cubic", "empty"],
)
def test_render_rejects_invalid_density_shape_before_calling_protocol(
    density: torch.Tensor,
) -> None:
    """Break caught: malformed volume shapes enter Sadekar's renderer boundary."""

    protocol = _CapturingProtocol(torch.device("cpu"), torch.ones(1, 1, 1))

    with pytest.raises(ValueError):
        render_thresholded(protocol, density, 0.05)

    assert protocol.calls == 0


def test_render_prefers_camera_state_device_over_generic_protocol_device() -> None:
    """Break caught: stale protocol.device rejects a renderer whose camera is local."""

    density = torch.full((1, 1, 1, 1, 1), 0.05)
    protocol = _CapturingProtocol(torch.device("meta"), torch.ones(1, 1, 1))
    protocol.cameras = SimpleNamespace(R=torch.empty(1, device="cpu"))

    render_thresholded(protocol, density, 0.05)

    assert protocol.calls == 1


def test_unindexed_cuda_protocol_device_canonicalizes_to_current_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: generic CUDA matches every GPU instead of the current one."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    protocol = _CapturingProtocol(torch.device("cuda"), torch.ones(1, 1, 1))
    current_density = SimpleNamespace(device=torch.device("cuda", 1))
    other_density = SimpleNamespace(device=torch.device("cuda", 0))

    assert thresholding_module._protocol_matches_density_device(protocol, current_density)
    assert not thresholding_module._protocol_matches_density_device(protocol, other_density)


def test_indexed_camera_state_device_rejects_other_cuda_index() -> None:
    """Break caught: a CUDA camera on another index is accepted by device type alone."""

    protocol = _CapturingProtocol(torch.device("cuda"), torch.ones(1, 1, 1))
    protocol.cameras = SimpleNamespace(R=SimpleNamespace(device=torch.device("cuda", 1)))
    density = SimpleNamespace(device=torch.device("cuda", 0))

    assert not thresholding_module._protocol_matches_density_device(protocol, density)
