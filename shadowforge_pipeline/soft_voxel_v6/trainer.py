"""Authenticated native-256 soft-loss-v6 trainer and durable lifecycle."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Mapping, Sequence
import uuid

import cv2
import numpy as np
import torch

from shadowforge_pipeline.soft_voxel_v3.data import normalize_target
from shadowforge_pipeline.soft_voxel_v3.losses import balanced_lp_loss
from shadowforge_pipeline.soft_voxel_v3.protocol import SadekarTFSProtocol
from shadowforge_pipeline.soft_voxel_v3.thresholding import render_thresholded
from shadowforge_pipeline.soft_voxel_v6.core import (
    PROFILE_FLAGS,
    PROFILE_ORDER,
    background_hard_negative_tail_loss,
    build_fixed_visual_hull_logits,
    compose_v6_gradient,
    delivery_lovasz_loss,
    map_suite_views,
    v6_auxiliary_multiplier,
    v6_learning_rate,
)
from shadowforge_pipeline.soft_voxel_v6.provenance import (
    atomic_bytes,
    atomic_json,
    canonical_hash,
    canonical_json_bytes,
    decode_canonical_json,
    environment_identity,
    reject_symlink_components,
    relative_regular_file,
    revalidate_snapshot,
    sha256_file,
    snapshot_regular_file,
)


class TrainingStateError(RuntimeError):
    """A scientific or durable state failed a finite/authentication check."""


@dataclass(frozen=True)
class StepTelemetry:
    completed_updates: int
    learning_rate: float
    auxiliary_multiplier: float
    base_loss: float
    j_loss: float | None
    h_loss: float | None
    base_norm: float
    j_raw_norm: float
    h_raw_norm: float
    j_ratio: float
    h_ratio: float
    combined_ratio: float
    j_zero: bool
    h_zero: bool
    pre_clip_norm: float
    post_clip_norm: float
    parameter_delta_norm: float
    clip_calls: int
    optimizer_steps: int
    elapsed_seconds: float
    peak_cuda_allocated_bytes: int
    tail_details: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ResumableRun:
    run_dir: str
    config_sha256: str
    completed_updates: int
    checkpoint_sha256: str


@dataclass(frozen=True)
class CompletedRun:
    run_dir: str
    config_sha256: str
    completed_updates: int
    artifacts_sha256: str
    metrics: dict[str, object]


@dataclass
class _Runtime:
    logits: torch.Tensor
    optimizer: torch.optim.Optimizer
    protocol: object
    targets: torch.Tensor


class _TestProtocol:
    """Private differentiable CPU renderer used only by the explicit small seam."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def render_all(self, density: torch.Tensor) -> torch.Tensor:
        view = density[0, 0].mean(dim=0)
        return view.unsqueeze(0).repeat(6, 1, 1)


class _TestStepTimer:
    def __init__(self) -> None:
        self._calls = 0

    def __call__(self) -> float:
        self._calls += 1
        return 0.0 if self._calls == 1 else 0.001


_STATUS_SCHEMA = "pcraft_soft_loss_v6_status_v1"
_CHECKPOINT_SCHEMA = "pcraft_soft_loss_v6_checkpoint_v1"
_CONFIG_SCHEMA = "pcraft_soft_loss_v6_resolved_config_v1"
_TOP_AUDIT_FILES = ("COMMAND.json", "RUN.json", "environment.json", "sources.json", "inputs.json")


def _finite(name: str, value: object) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise TrainingStateError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingStateError(f"{name} must be finite")
    return result


def _tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def required_checkpoint_steps(total_updates: int = 3000) -> tuple[int, ...]:
    if type(total_updates) is not int or total_updates <= 0:
        raise ValueError("total_updates must be a positive integer")
    values = {0, min(2, total_updates), total_updates}
    values.update(range(200, total_updates + 1, 200))
    return tuple(sorted(values))


def required_diagnostic_steps(total_updates: int = 3000) -> tuple[int, ...]:
    if total_updates == 3000:
        return (2400, 2700, 3000)
    return ()


def _validate_arm(arm: object) -> str:
    if type(arm) is not str or arm not in PROFILE_ORDER:
        raise ValueError("arm must be one of the four frozen v6 profiles")
    return arm


def _train_one_update(
    *,
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    protocol: object,
    targets: torch.Tensor,
    arm: str,
    completed_updates: int,
    timer=time.perf_counter,
) -> StepTelemetry:
    """Execute exactly one update with independent base/J/H graphs."""

    _validate_arm(arm)
    if (
        not isinstance(logits, torch.Tensor)
        or not logits.is_floating_point()
        or not logits.requires_grad
        or not logits.is_leaf
        or logits.ndim != 5
        or logits.shape[:2] != (1, 1)
        or not torch.isfinite(logits).all().item()
    ):
        raise ValueError("logits must be finite leaf [1,1,D,D,D] floating state")
    if (
        not isinstance(targets, torch.Tensor)
        or not targets.is_floating_point()
        or targets.ndim != 3
        or targets.shape[0] != 6
        or targets.device != logits.device
        or not torch.isfinite(targets).all().item()
    ):
        raise ValueError("targets must be six finite floating views on the logits device")
    if not callable(getattr(protocol, "render_all", None)):
        raise ValueError("protocol must provide render_all")
    if not callable(getattr(optimizer, "zero_grad", None)) or not callable(
        getattr(optimizer, "step", None)
    ):
        raise ValueError("optimizer must provide zero_grad and step")
    multiplier = v6_auxiliary_multiplier(completed_updates)
    learning_rate = v6_learning_rate(completed_updates)
    target_binary = targets >= 0.5
    started = float(timer())

    base_projection = protocol.render_all(torch.sigmoid(logits))
    base_loss = (
        balanced_lp_loss(base_projection, targets, p=2)
        + 0.35 * balanced_lp_loss(base_projection, targets, p=1)
    ).mean()
    if not torch.isfinite(base_loss).item():
        raise TrainingStateError("base loss must be finite")
    (base_grad,) = torch.autograd.grad(base_loss, logits, retain_graph=False)

    j_grad: torch.Tensor | None = None
    h_grad: torch.Tensor | None = None
    j_loss: torch.Tensor | None = None
    h_loss: torch.Tensor | None = None
    tail_details: list[dict[str, object]] = []
    flags = PROFILE_FLAGS[arm]
    if multiplier > 0.0 and flags["j"]:
        j_readout = render_thresholded(protocol, torch.sigmoid(logits), 0.05)
        j_loss = delivery_lovasz_loss(j_readout.score, target_binary)
        (j_grad,) = torch.autograd.grad(j_loss, logits, retain_graph=False)
    if multiplier > 0.0 and flags["h"]:
        h_readout = render_thresholded(protocol, torch.sigmoid(logits), 0.05)
        h_result = background_hard_negative_tail_loss(
            h_readout.score,
            target_binary,
            fraction=0.05,
            margin=0.01,
            temperature=0.02,
            return_details=True,
        )
        h_loss, tail_details = h_result
        (h_grad,) = torch.autograd.grad(h_loss, logits, retain_graph=False)

    composed = compose_v6_gradient(
        base_grad,
        j_grad,
        h_grad,
        completed_updates=completed_updates,
        profile=arm,
        return_telemetry=True,
    )
    if isinstance(composed, tuple):
        final_grad, diagnostics = composed
    else:  # a deliberately narrow failure-injection seam
        final_grad = composed
        diagnostics = {
            "base_norm": float(torch.linalg.vector_norm(base_grad).item()),
            "j_raw_norm": 0.0,
            "h_raw_norm": 0.0,
            "j_ratio": 0.0,
            "h_ratio": 0.0,
            "combined_ratio": 0.0,
            "j_zero": False,
            "h_zero": False,
        }
    if not torch.isfinite(final_grad).all().item():
        raise TrainingStateError("composed gradient must be finite")
    before = logits.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    logits.grad = final_grad.detach().clone()
    pre_clip = float(torch.linalg.vector_norm(logits.grad).item())
    if not math.isfinite(pre_clip):
        raise TrainingStateError("pre-clip gradient must be finite")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    returned_norm = torch.nn.utils.clip_grad_norm_([logits], 1.0)
    if not math.isfinite(float(returned_norm)) or logits.grad is None or not torch.isfinite(logits.grad).all().item():
        raise TrainingStateError("clipped gradient must be finite")
    post_clip = float(torch.linalg.vector_norm(logits.grad).item())
    optimizer.step()
    if not torch.isfinite(logits).all().item():
        raise TrainingStateError("parameters must remain finite after optimizer step")
    delta = float(torch.linalg.vector_norm(logits.detach() - before).item())
    ended = float(timer())
    elapsed = ended - started
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise TrainingStateError("elapsed time must be finite and non-negative")
    peak = 0
    if logits.device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(logits.device))
    values = {
        "base_loss": float(base_loss.detach().item()),
        "j_loss": None if j_loss is None else float(j_loss.detach().item()),
        "h_loss": None if h_loss is None else float(h_loss.detach().item()),
    }
    for name, value in values.items():
        if value is not None:
            _finite(name, value)
    return StepTelemetry(
        completed_updates=completed_updates + 1,
        learning_rate=learning_rate,
        auxiliary_multiplier=multiplier,
        base_loss=values["base_loss"],
        j_loss=values["j_loss"],
        h_loss=values["h_loss"],
        base_norm=_finite("base norm", diagnostics["base_norm"]),
        j_raw_norm=_finite("J raw norm", diagnostics["j_raw_norm"]),
        h_raw_norm=_finite("H raw norm", diagnostics["h_raw_norm"]),
        j_ratio=_finite("J ratio", diagnostics["j_ratio"]),
        h_ratio=_finite("H ratio", diagnostics["h_ratio"]),
        combined_ratio=_finite("combined ratio", diagnostics["combined_ratio"]),
        j_zero=bool(diagnostics["j_zero"]),
        h_zero=bool(diagnostics["h_zero"]),
        pre_clip_norm=pre_clip,
        post_clip_norm=post_clip,
        parameter_delta_norm=delta,
        clip_calls=1,
        optimizer_steps=1,
        elapsed_seconds=elapsed,
        peak_cuda_allocated_bytes=peak,
        tail_details=tuple(copy.deepcopy(tail_details)),
    )


def evaluate_six_view_metrics(
    predicted: torch.Tensor, target: torch.Tensor, *, completed_updates: int
) -> dict[str, object]:
    if type(completed_updates) is not int or completed_updates <= 0:
        raise ValueError("completed_updates must be a positive integer")
    if (
        not isinstance(predicted, torch.Tensor)
        or predicted.dtype != torch.bool
        or not isinstance(target, torch.Tensor)
        or target.dtype != torch.bool
        or predicted.shape != target.shape
        or predicted.ndim != 3
        or predicted.shape[0] != 6
        or any(size == 0 for size in predicted.shape)
    ):
        raise ValueError("metrics require equal boolean tensors with six non-empty views")
    rows: list[dict[str, object]] = []
    for index in range(6):
        p = predicted[index]
        t = target[index]
        tp = int(torch.count_nonzero(p & t).item())
        fp = int(torch.count_nonzero(p & ~t).item())
        fn = int(torch.count_nonzero(~p & t).item())
        tn = int(torch.count_nonzero(~p & ~t).item())
        union = tp + fp + fn
        dice_denominator = 2 * tp + fp + fn
        foreground = tp + fn
        background = fp + tn
        if min(union, dice_denominator, foreground, background) <= 0:
            raise ValueError(f"invalid metric denominator in view {index}")
        row = {
            "view": index,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "iou": tp / union,
            "dice": (2 * tp) / dice_denominator,
            "recall": tp / foreground,
            "fpr": fp / background,
        }
        if not all(math.isfinite(value) for value in row.values() if type(value) is float):
            raise ValueError("metric must be finite")
        rows.append(row)
    aggregate = {
        f"mean_{name}": sum(float(row[name]) for row in rows) / 6.0
        for name in ("iou", "dice", "recall", "fpr")
    }
    return {
        "schema": "pcraft_soft_loss_v6_metrics_v1",
        "completed_updates": completed_updates,
        "threshold": 0.05,
        "per_view": rows,
        "aggregate": aggregate,
    }


def _test_targets(resolution: int) -> torch.Tensor:
    target = torch.zeros((6, resolution, resolution), dtype=torch.float32)
    lo = max(1, resolution // 4)
    hi = min(resolution - 1, resolution - lo)
    target[:, lo:hi, lo:hi] = 1.0
    return target


def _make_test_resolved_config(
    *, run_dir: Path, arm: str = "full", total_updates: int = 4, resolution: int = 4
) -> dict[str, object]:
    arm = _validate_arm(arm)
    if type(total_updates) is not int or total_updates < 2:
        raise ValueError("test total_updates must be at least two")
    if type(resolution) is not int or resolution < 3:
        raise ValueError("test resolution must be at least three")
    run_path = Path(run_dir).resolve()
    target = _test_targets(resolution)
    hull = torch.ones((1, 1, resolution, resolution, resolution), dtype=torch.bool)
    logits = torch.where(hull, torch.tensor(2.0), torch.tensor(-3.0)).float()
    hard = torch.sigmoid(logits) >= 0.05
    config: dict[str, object] = {
        "schema": _CONFIG_SCHEMA,
        "run": {"root": str(run_path.parent), "run_id": f"test/{arm}/seed_17", "path": str(run_path)},
        "task": "connected_01",
        "arm": arm,
        "seed": 17,
        "device": "cpu",
        "resolution": resolution,
        "export_resolution": resolution,
        "updates": total_updates,
        "inputs": {"suite": {"path": "test-only", "sha256": "0" * 64}, "raw_xyz": []},
        "targets": {"mapped_six_sha256": _tensor_hash(target >= 0.5), "shape": [6, resolution, resolution]},
        "cameras": {"schema": "test-only"},
        "hull": {"sha256": _tensor_hash(hull), "shape": [resolution] * 3},
        "initialization": {
            "values": {"inside": 2.0, "outside": -3.0},
            "logits_sha256": _tensor_hash(logits),
            "step_zero_hard_sha256": _tensor_hash(hard),
        },
        "objective": {"base": "L2_balanced+0.35*L1_balanced", "j": "delivery_lovasz", "h": "background_fp_tail", "no_reallocation": True},
        "schedule": {"learning_rate": "v6_learning_rate", "auxiliary": "v6_auxiliary_multiplier", "clip_norm": 1.0},
        "protocol": {"density_threshold": 0.05, "ste_temperature": 0.01, "readout": "score>=0 inclusive"},
        "checkpoint_policy": {"forced": [0, 2], "interval": 200, "final_only": True, "best": False},
        "specification": {"path": "test-only", "sha256": "1" * 64},
        "implementation": {"bundle_sha256": "2" * 64, "files_sha256": {}},
        "environment": {"schema": "test-only", "requested_device": "cpu"},
        "test_only": True,
    }
    return json.loads(canonical_json_bytes(config))


def _validate_test_resolved_config(config: object) -> dict[str, object]:
    if type(config) is not dict or config.get("test_only") is not True:
        raise ValueError("test config must carry the private test-only marker")
    try:
        reject_symlink_components(Path(config["run"]["path"]), must_exist=False)
        expected = _make_test_resolved_config(
            run_dir=Path(config["run"]["path"]),
            arm=config["arm"],
            total_updates=config["updates"],
            resolution=config["resolution"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("test resolved config has the wrong schema") from error
    if canonical_json_bytes(config) != canonical_json_bytes(expected):
        raise ValueError("test resolved config differs from independent reconstruction")
    reject_symlink_components(Path(expected["run"]["path"]), must_exist=False)
    return expected


def _validate_production_resolved_config(config: object) -> dict[str, object]:
    if type(config) is not dict or config.get("test_only") is not None:
        raise ValueError("production resolved config has the wrong schema")
    from experiments.run_soft_loss_v6_ablation10_256 import rebuild_resolved_config

    expected = rebuild_resolved_config(config)
    if canonical_json_bytes(config) != canonical_json_bytes(expected):
        raise ValueError("resolved config differs from independent reconstruction")
    if expected.get("resolution") != 256 or expected.get("export_resolution") != 256 or expected.get("updates") != 3000:
        raise ValueError("production config must be fresh native 256 for 3000 updates")
    return expected


def _seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _load_production_targets(config: Mapping[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    resolution = int(config["resolution"])
    rows = config["inputs"]["raw_xyz"]
    base: list[torch.Tensor] = []
    mirrors: list[torch.Tensor] = []
    for item in rows:
        path = Path(item["path"])
        snapshot = snapshot_regular_file(path)
        payload = path.read_bytes()
        if (
            hashlib.sha256(payload).hexdigest() != snapshot.sha256
            or snapshot.sha256 != item["sha256"]
        ):
            raise ValueError("authenticated input bytes changed during runtime loading")
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("authenticated input is not a grayscale image")
        base.append(torch.from_numpy(normalize_target(image, resolution, False) > 0))
        mirrors.append(torch.from_numpy(normalize_target(cv2.flip(image, 1), resolution, False) > 0))
        revalidate_snapshot(snapshot)
    labelled_base = tuple(zip(("x", "y", "z"), base))
    labelled_mirrors = tuple(zip(("x", "y", "z"), mirrors))
    mapped = map_suite_views(labelled_base, labelled_mirrors)
    logits = build_fixed_visual_hull_logits(labelled_base, resolution=resolution)
    hull = logits == 2.0
    hard = torch.sigmoid(logits) >= 0.05
    targets_identity = config["targets"]
    if (
        _tensor_hash(torch.stack(base)) != targets_identity["base_sha256"]
        or _tensor_hash(torch.stack(mirrors)) != targets_identity["mirror_sha256"]
        or _tensor_hash(mapped) != targets_identity["mapped_six_sha256"]
        or list(mapped.shape) != targets_identity["shape"]
        or _tensor_hash(hull[0, 0]) != config["hull"]["sha256"]
        or _tensor_hash(logits) != config["initialization"]["logits_sha256"]
        or _tensor_hash(hard) != config["initialization"]["step_zero_hard_sha256"]
        or not torch.equal(hull, hard)
    ):
        raise ValueError("runtime targets/hull/init differ from authenticated identities")
    return mapped.float(), logits


def _build_runtime(config: Mapping[str, object]) -> _Runtime:
    device = torch.device(config["device"])
    if config.get("test_only") is True:
        resolution = int(config["resolution"])
        targets = _test_targets(resolution).to(device)
        logits = torch.full((1, 1, resolution, resolution, resolution), 2.0, device=device)
        protocol: object = _TestProtocol()
    else:
        targets_cpu, logits_cpu = _load_production_targets(config)
        targets = targets_cpu.to(device)
        logits = logits_cpu.to(device)
        protocol = SadekarTFSProtocol(
            device=device,
            image_size=256,
            zdist=1.7,
            volume_extent_world=1.7,
            n_pts_per_ray=150,
        )
    logits = logits.to(dtype=torch.float32).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=0.007)
    return _Runtime(logits, optimizer, protocol, targets)


def _capture_rng(device: torch.device) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().cpu(),
        "cuda": [item.cpu() for item in torch.cuda.get_rng_state_all()] if device.type == "cuda" else [],
    }


def _restore_rng(payload: Mapping[str, object], device: torch.device) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.random.set_rng_state(payload["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state_all([item.cpu() for item in payload["cuda"]])


def _torch_bytes(payload: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _checkpoint_name(completed: int) -> str:
    return f"step_{completed:04d}.pt"


def _write_checkpoint(
    run_dir: Path,
    *,
    config_hash: str,
    runtime: _Runtime,
    completed: int,
    predecessor_sha256: str | None,
    audit: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
    step_times: Sequence[float],
) -> str:
    payload = {
        "schema": _CHECKPOINT_SCHEMA,
        "config_sha256": config_hash,
        "completed_updates": completed,
        "predecessor_sha256": predecessor_sha256,
        "logits": runtime.logits.detach().cpu().to(torch.float32).contiguous(),
        "optimizer": runtime.optimizer.state_dict(),
        "rng": _capture_rng(runtime.logits.device),
        "audit": copy.deepcopy(dict(audit)),
        "history": copy.deepcopy(list(history)),
        "step_times": list(step_times),
    }
    path = run_dir / "checkpoints" / _checkpoint_name(completed)
    atomic_bytes(path, _torch_bytes(payload), create_once=True)
    digest = sha256_file(path)
    atomic_json(
        run_dir / "checkpoints/LATEST.json",
        {
            "schema": "pcraft_soft_loss_v6_checkpoint_pointer_v1",
            "completed_updates": completed,
            "checkpoint": f"checkpoints/{path.name}",
            "sha256": digest,
        },
    )
    return digest


def _read_status(run_dir: Path) -> dict[str, object]:
    status = decode_canonical_json(run_dir / "STATUS.json", label="v6 STATUS")
    state = status.get("state")
    common = {"schema", "state", "config_sha256", "completed_updates"}
    expected = {
        "running": common,
        "failed": common | {"error"},
        "completed": common | {"artifacts_sha256"},
    }
    if state not in expected or set(status) != expected[state] or status.get("schema") != _STATUS_SCHEMA:
        raise ValueError("STATUS has the wrong exact state schema")
    if type(status["completed_updates"]) is not int or status["completed_updates"] < 0:
        raise ValueError("STATUS completed update is invalid")
    return status


def _read_checkpoint_pointer(run_dir: Path) -> dict[str, object]:
    pointer = decode_canonical_json(
        run_dir / "checkpoints/LATEST.json", label="checkpoint pointer"
    )
    if (
        set(pointer) != {"schema", "completed_updates", "checkpoint", "sha256"}
        or pointer["schema"] != "pcraft_soft_loss_v6_checkpoint_pointer_v1"
        or type(pointer["completed_updates"]) is not int
        or pointer["completed_updates"] < 0
        or type(pointer["sha256"]) is not str
        or len(pointer["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in pointer["sha256"])
    ):
        raise ValueError("checkpoint pointer schema mismatch")
    return pointer


def _expected_audits(config: Mapping[str, object], config_hash: str) -> dict[str, object]:
    return {
        "COMMAND.json": {"schema": "pcraft_soft_loss_v6_command_v1", "resolved_config": config, "config_sha256": config_hash},
        "RUN.json": {"schema": "pcraft_soft_loss_v6_run_v1", "config_sha256": config_hash, "run": config["run"], "task": config["task"], "arm": config["arm"], "seed": 17},
        "environment.json": config["environment"],
        "sources.json": {"schema": "pcraft_soft_loss_v6_sources_v1", "specification": config["specification"], "implementation": config["implementation"]},
        "inputs.json": {"schema": "pcraft_soft_loss_v6_inputs_v1", "inputs": config["inputs"], "targets": config["targets"], "cameras": config["cameras"], "hull": config["hull"], "initialization": config["initialization"]},
    }


def _validate_audits(run_dir: Path, config: Mapping[str, object], config_hash: str) -> None:
    for name, expected in _expected_audits(config, config_hash).items():
        observed = decode_canonical_json(run_dir / name, label=name)
        if canonical_json_bytes(observed) != canonical_json_bytes(expected):
            raise ValueError(f"{name} does not match the authenticated resolved config")


def _safe_checkpoint_files(run_dir: Path) -> list[Path]:
    directory = reject_symlink_components(run_dir / "checkpoints")
    results: list[Path] = []
    for path in directory.iterdir():
        if path.name == "LATEST.json":
            continue
        reject_symlink_components(path)
        if not path.is_file() or not path.name.startswith("step_") or not path.name.endswith(".pt"):
            raise ValueError("checkpoint directory contains an unexpected entry")
        results.append(path)
    return sorted(results)


def _state_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.device == right.device
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray):
        return (
            isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and np.array_equal(left, right)
        )
    if type(left) is dict:
        return set(left) == set(right) and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if type(left) in (tuple, list):
        return len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _stable_file_bytes(path: Path, *, label: str) -> tuple[bytes, str]:
    snapshot = snapshot_regular_file(path)
    payload = Path(snapshot.path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != snapshot.sha256:
        raise ValueError(f"{label} bytes changed during authenticated loading")
    revalidate_snapshot(snapshot)
    return payload, digest


def _validate_rng_state(
    payload: object, config: Mapping[str, object], genesis: object | None
) -> None:
    if type(payload) is not dict or set(payload) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("checkpoint RNG has the wrong exact schema")
    seed = int(config["seed"])
    expected_python = random.Random(seed).getstate()
    expected_numpy = np.random.RandomState(seed).get_state()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    expected_torch = generator.get_state()
    if (
        not _state_equal(payload["python"], expected_python)
        or not _state_equal(payload["numpy"], expected_numpy)
        or not _state_equal(payload["torch"], expected_torch)
    ):
        raise ValueError("checkpoint RNG differs from the exact seeded state")
    cuda = payload["cuda"]
    if type(cuda) is not list:
        raise ValueError("checkpoint CUDA RNG has the wrong exact type")
    if str(config["device"]).startswith("cuda"):
        if not cuda or len(cuda) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state is missing")
        for device_index, state in enumerate(cuda):
            if (
                not isinstance(state, torch.Tensor)
                or state.dtype != torch.uint8
                or state.device.type != "cpu"
                or state.ndim != 1
                or state.numel() == 0
            ):
                raise ValueError("checkpoint CUDA RNG state is malformed")
            expected_generator = torch.Generator(device=f"cuda:{device_index}")
            expected_generator.manual_seed(seed)
            if not torch.equal(state, expected_generator.get_state().cpu()):
                raise ValueError("checkpoint CUDA RNG differs from the exact seeded state")
    elif cuda:
        raise ValueError("CPU checkpoint must not carry CUDA RNG state")
    if genesis is not None and not _state_equal(payload, genesis):
        raise ValueError("checkpoint RNG state changed across a deterministic chain")


def _validate_adam_state(
    payload: object, logits: torch.Tensor, completed: int
) -> None:
    if type(payload) is not dict or set(payload) != {"state", "param_groups"}:
        raise ValueError("checkpoint Adam state has the wrong exact schema")
    groups = payload["param_groups"]
    if type(groups) is not list or len(groups) != 1 or type(groups[0]) is not dict:
        raise ValueError("checkpoint Adam parameter groups are malformed")
    scalar = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    expected_group = torch.optim.Adam([scalar], lr=0.007).state_dict()["param_groups"][0]
    expected_group["lr"] = 0.007 if completed == 0 else v6_learning_rate(completed - 1)
    if not _state_equal(groups[0], expected_group):
        raise ValueError("checkpoint Adam parameter group differs from the frozen optimizer")
    state = payload["state"]
    if completed == 0:
        if type(state) is not dict or state:
            raise ValueError("genesis Adam state must be empty")
        return
    if type(state) is not dict or set(state) != {0} or type(state[0]) is not dict:
        raise ValueError("checkpoint Adam state does not own exactly one parameter")
    moment = state[0]
    if set(moment) != {"step", "exp_avg", "exp_avg_sq"}:
        raise ValueError("checkpoint Adam moment schema is invalid")
    step = moment["step"]
    if (
        not isinstance(step, torch.Tensor)
        or step.dtype != torch.float32
        or step.device.type != "cpu"
        or step.shape != torch.Size([])
        or float(step.item()) != float(completed)
    ):
        raise ValueError("checkpoint Adam step does not match completed updates")
    for name in ("exp_avg", "exp_avg_sq"):
        value = moment[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.shape != logits.shape
            or not value.is_contiguous()
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError(f"checkpoint Adam {name} is malformed")
    if torch.any(moment["exp_avg_sq"] < 0).item():
        raise ValueError("checkpoint Adam second moment must be nonnegative")


def _validate_tail_details(details: object) -> None:
    if type(details) is not tuple or len(details) != 6:
        raise ValueError("checkpoint H-tail telemetry must contain six views")
    exact = {
        "view", "background_count", "selected_count", "selected_indices",
        "score_min", "score_mean", "score_max",
    }
    for index, detail in enumerate(details):
        if type(detail) is not dict or set(detail) != exact or detail["view"] != index:
            raise ValueError("checkpoint H-tail telemetry has the wrong exact schema")
        background = detail["background_count"]
        selected = detail["selected_count"]
        indices = detail["selected_indices"]
        if (
            type(background) is not int
            or background <= 0
            or type(selected) is not int
            or selected != max(1, math.ceil(0.05 * background))
            or type(indices) is not list
            or len(indices) != selected
            or any(type(value) is not int or value < 0 for value in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("checkpoint H-tail selection telemetry is invalid")
        scores = tuple(detail[name] for name in ("score_min", "score_mean", "score_max"))
        if (
            any(type(value) is not float or not math.isfinite(value) for value in scores)
            or not scores[0] <= scores[1] <= scores[2]
        ):
            raise ValueError("checkpoint H-tail score telemetry is invalid")


def _validate_telemetry_row(
    payload: object, *, index: int, config: Mapping[str, object]
) -> StepTelemetry:
    exact = {field.name for field in fields(StepTelemetry)}
    if type(payload) is not dict or set(payload) != exact:
        raise ValueError("checkpoint history row has the wrong exact telemetry schema")
    integer_fields = {"completed_updates", "clip_calls", "optimizer_steps", "peak_cuda_allocated_bytes"}
    boolean_fields = {"j_zero", "h_zero"}
    optional_float_fields = {"j_loss", "h_loss"}
    float_fields = exact - integer_fields - boolean_fields - optional_float_fields - {"tail_details"}
    if any(type(payload[name]) is not int for name in integer_fields):
        raise ValueError("checkpoint telemetry integer field has the wrong exact type")
    if any(type(payload[name]) is not bool for name in boolean_fields):
        raise ValueError("checkpoint telemetry boolean field has the wrong exact type")
    if any(
        type(payload[name]) is not float or not math.isfinite(payload[name])
        for name in float_fields
    ):
        raise ValueError("checkpoint telemetry float field is nonfinite or mistyped")
    for name in optional_float_fields:
        if payload[name] is not None and (
            type(payload[name]) is not float or not math.isfinite(payload[name])
        ):
            raise ValueError("checkpoint telemetry loss is nonfinite or mistyped")
    if type(payload["tail_details"]) is not tuple:
        raise ValueError("checkpoint tail telemetry has the wrong exact type")
    row = StepTelemetry(**payload)
    completed_before = index - 1
    multiplier = v6_auxiliary_multiplier(completed_before)
    learning_rate = v6_learning_rate(completed_before)
    flags = PROFILE_FLAGS[str(config["arm"])]
    if (
        row.completed_updates != index
        or row.learning_rate != learning_rate
        or row.auxiliary_multiplier != multiplier
        or row.clip_calls != 1
        or row.optimizer_steps != 1
        or row.base_loss < 0.0
        or row.base_norm <= 1e-12
        or row.pre_clip_norm <= 0.0
        or row.post_clip_norm <= 0.0
        or row.parameter_delta_norm < 0.0
        or row.elapsed_seconds <= 0.0
        or row.peak_cuda_allocated_bytes < 0
    ):
        raise ValueError("checkpoint telemetry violates the frozen update contract")
    if row.post_clip_norm > min(row.pre_clip_norm, 1.0) + 1e-6:
        raise ValueError("checkpoint telemetry violates the one-clip contract")
    if str(config["device"]).startswith("cpu") and row.peak_cuda_allocated_bytes != 0:
        raise ValueError("CPU checkpoint reports impossible CUDA allocation")
    active = {
        "j": multiplier > 0.0 and bool(flags["j"]),
        "h": multiplier > 0.0 and bool(flags["h"]),
    }
    for name, maximum in (("j", 0.20), ("h", 0.10)):
        loss = getattr(row, f"{name}_loss")
        raw_norm = getattr(row, f"{name}_raw_norm")
        ratio = getattr(row, f"{name}_ratio")
        zero = getattr(row, f"{name}_zero")
        if not active[name]:
            if loss is not None or raw_norm != 0.0 or ratio != 0.0 or zero:
                raise ValueError("checkpoint telemetry constructs a disabled auxiliary")
            continue
        if loss is None or loss < 0.0 or raw_norm < 0.0:
            raise ValueError("checkpoint active auxiliary telemetry is invalid")
        expected_zero = raw_norm <= 1e-12
        upper = float(
            np.nextafter(
                np.float32(multiplier * maximum), np.float32(math.inf)
            )
        )
        if zero is not expected_zero or ratio < 0.0 or ratio > upper:
            raise ValueError("checkpoint auxiliary budget/zero telemetry is invalid")
        if expected_zero and ratio != 0.0:
            raise ValueError("checkpoint zero auxiliary has a nonzero ratio")
    combined_upper = float(
        np.nextafter(np.float32(multiplier * 0.30), np.float32(math.inf))
    )
    if row.combined_ratio < 0.0 or row.combined_ratio > combined_upper:
        raise ValueError("checkpoint combined auxiliary budget is invalid")
    if active["h"]:
        _validate_tail_details(row.tail_details)
    elif row.tail_details:
        raise ValueError("checkpoint carries H-tail details while H is inactive")
    return row


def _validate_history_and_audit(
    payload: Mapping[str, object], *, completed: int, config: Mapping[str, object]
) -> tuple[list[dict[str, object]], list[float]]:
    history = payload["history"]
    times = payload["step_times"]
    if type(history) is not list or len(history) != completed:
        raise ValueError("checkpoint history length differs from completed updates")
    if type(times) is not list or len(times) != completed:
        raise ValueError("checkpoint timing prefix length differs from completed updates")
    audit = _initial_audit(str(config["arm"]))
    validated: list[dict[str, object]] = []
    for index, raw_row in enumerate(history, start=1):
        row = _validate_telemetry_row(raw_row, index=index, config=config)
        if type(times[index - 1]) is not float or times[index - 1] != row.elapsed_seconds:
            raise ValueError("checkpoint timing prefix differs from telemetry history")
        _update_audit(audit, row, str(config["arm"]))
        validated.append(raw_row)
    if not _state_equal(payload["audit"], audit):
        raise ValueError("checkpoint audit is not the exact telemetry-derived prefix")
    return validated, times


def _load_chain(
    run_dir: Path,
    *,
    config: Mapping[str, object],
    config_hash: str,
    expected_completed: int | None = None,
    allow_test_steps: bool = False,
    allow_unpublished_tail: bool = False,
) -> tuple[dict[str, object], str, dict[int, str]]:
    pointer = _read_checkpoint_pointer(run_dir)
    latest = pointer["completed_updates"]
    if expected_completed is not None and latest != expected_completed:
        raise ValueError("checkpoint pointer completed update mismatch")
    files = _safe_checkpoint_files(run_dir)
    try:
        observed_steps = [int(path.stem.split("_")[1]) for path in files]
    except (IndexError, ValueError) as error:
        raise ValueError("checkpoint name is malformed") from error
    if len(observed_steps) != len(set(observed_steps)):
        raise ValueError("checkpoint generations are not uniquely named")
    unpublished = sorted(step for step in observed_steps if step > latest)
    if unpublished and not allow_unpublished_tail:
        raise ValueError("checkpoint generations are not strictly ordered")
    if len(unpublished) > 1:
        raise ValueError("checkpoint chain has multiple unpublished generations")
    effective_latest = unpublished[0] if unpublished else latest
    if unpublished:
        total = int(config["updates"])
        if allow_test_steps:
            if effective_latest > total:
                raise ValueError("unpublished test checkpoint exceeds configured total")
        else:
            future = tuple(
                step for step in required_checkpoint_steps(total) if step > latest
            )
            if not future or effective_latest != future[0]:
                raise ValueError("unpublished checkpoint is not the next registered generation")
    expected_names = {
        _checkpoint_name(step)
        for step in required_checkpoint_steps(int(config["updates"]))
        if step <= latest
    }
    expected_names.add(_checkpoint_name(latest))
    if unpublished:
        expected_names.add(_checkpoint_name(effective_latest))
    observed_names = {path.name for path in files}
    if not allow_test_steps and observed_names != expected_names:
        raise ValueError("checkpoint chain has extra or missing generations")
    if allow_test_steps and not {
        _checkpoint_name(0),
        _checkpoint_name(min(2, latest)),
        _checkpoint_name(latest),
    } <= observed_names:
        raise ValueError("test checkpoint chain lacks a mandatory generation")
    predecessor: str | None = None
    latest_payload: dict[str, object] | None = None
    latest_digest = ""
    digests: dict[int, str] = {}
    previous_history: list[dict[str, object]] = []
    previous_times: list[float] = []
    genesis_rng: object | None = None
    previous_completed = -1
    exact = {
        "schema", "config_sha256", "completed_updates", "predecessor_sha256",
        "logits", "optimizer", "rng", "audit", "history", "step_times",
    }
    resolution = int(config["resolution"])
    for path in files:
        completed = int(path.stem.split("_")[1])
        if completed <= previous_completed or completed > effective_latest:
            raise ValueError("checkpoint generations are not strictly ordered")
        raw, digest = _stable_file_bytes(path, label="checkpoint")
        try:
            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        except Exception as error:
            raise ValueError("checkpoint payload is malformed") from error
        if completed > latest:
            end_record = raw.rfind(b"PK\x05\x06")
            if (
                end_record < 0
                or end_record + 22 > len(raw)
                or end_record
                + 22
                + int.from_bytes(raw[end_record + 20 : end_record + 22], "little")
                != len(raw)
            ):
                raise ValueError("unpublished checkpoint payload has trailing bytes")
        revalidate_snapshot(snapshot_regular_file(path))
        if type(payload) is not dict or set(payload) != exact:
            raise ValueError("checkpoint payload has the wrong exact schema")
        logits = payload["logits"]
        if (
            payload["schema"] != _CHECKPOINT_SCHEMA
            or payload["config_sha256"] != config_hash
            or type(payload["completed_updates"]) is not int
            or payload["completed_updates"] != completed
            or payload["predecessor_sha256"] != predecessor
            or not isinstance(logits, torch.Tensor)
            or logits.dtype != torch.float32
            or logits.device.type != "cpu"
            or logits.shape != (1, 1, resolution, resolution, resolution)
            or not logits.is_contiguous()
            or not torch.isfinite(logits).all().item()
        ):
            raise ValueError("checkpoint chain identity/state mismatch")
        if completed == 0 and _tensor_hash(logits) != config["initialization"]["logits_sha256"]:
            raise ValueError("genesis checkpoint differs from authenticated initialization")
        _validate_adam_state(payload["optimizer"], logits, completed)
        _validate_rng_state(payload["rng"], config, genesis_rng)
        if genesis_rng is None:
            genesis_rng = payload["rng"]
        history, times = _validate_history_and_audit(
            payload, completed=completed, config=config
        )
        if (
            not _state_equal(history[: len(previous_history)], previous_history)
            or not _state_equal(times[: len(previous_times)], previous_times)
        ):
            raise ValueError("checkpoint history/telemetry prefix changed across generations")
        predecessor = digest
        latest_payload = payload
        latest_digest = digest
        digests[completed] = digest
        previous_history = history
        previous_times = times
        previous_completed = completed
    if latest_payload is None:
        raise ValueError("checkpoint chain is empty")
    checkpoint_relative = f"checkpoints/{_checkpoint_name(latest)}"
    if (
        pointer["checkpoint"] != checkpoint_relative
        or pointer["sha256"] != digests.get(latest)
    ):
        raise ValueError("checkpoint pointer hash/path mismatch")
    return latest_payload, latest_digest, digests


def _revalidate_static(config: Mapping[str, object]) -> None:
    if config.get("test_only") is True:
        return
    from experiments.run_soft_loss_v6_ablation10_256 import revalidate_static_identities

    revalidate_static_identities(config)


def _running_status_payload(
    config_hash: str, completed_updates: int
) -> dict[str, object]:
    return {
        "schema": _STATUS_SCHEMA,
        "state": "running",
        "config_sha256": config_hash,
        "completed_updates": completed_updates,
    }


def _recover_checkpoint_publication(
    run_dir: Path, config: Mapping[str, object], config_hash: str
) -> None:
    status = _read_status(run_dir)
    if status["config_sha256"] != config_hash:
        raise ValueError("STATUS does not match the authenticated resolved config")
    if status["state"] != "running":
        return
    status_completed = int(status["completed_updates"])
    pointer = _read_checkpoint_pointer(run_dir)
    pointer_completed = int(pointer["completed_updates"])
    if pointer_completed < status_completed:
        raise ValueError("checkpoint pointer is behind durable STATUS")
    if pointer_completed == status_completed:
        checkpoint, digest, _ = _load_chain(
            run_dir,
            config=config,
            config_hash=config_hash,
            expected_completed=pointer_completed,
            allow_test_steps=config.get("test_only") is True,
            allow_unpublished_tail=True,
        )
        recovered_completed = int(checkpoint["completed_updates"])
        if recovered_completed == pointer_completed:
            return
        _, stable_digest = _stable_file_bytes(
            run_dir / "checkpoints" / _checkpoint_name(recovered_completed),
            label="unpublished checkpoint",
        )
        if stable_digest != digest:
            raise ValueError("unpublished checkpoint changed after authentication")
        _revalidate_static(config)
        atomic_json(
            run_dir / "checkpoints/LATEST.json",
            {
                "schema": "pcraft_soft_loss_v6_checkpoint_pointer_v1",
                "completed_updates": recovered_completed,
                "checkpoint": f"checkpoints/{_checkpoint_name(recovered_completed)}",
                "sha256": digest,
            },
        )
        atomic_json(
            run_dir / "STATUS.json",
            _running_status_payload(config_hash, recovered_completed),
        )
        return
    _, _, digests = _load_chain(
        run_dir,
        config=config,
        config_hash=config_hash,
        expected_completed=pointer_completed,
        allow_test_steps=config.get("test_only") is True,
    )
    if sorted(step for step in digests if step > status_completed) != [
        pointer_completed
    ]:
        raise ValueError("checkpoint pointer skipped a durable STATUS publication")
    _revalidate_static(config)
    atomic_json(
        run_dir / "STATUS.json",
        _running_status_payload(config_hash, pointer_completed),
    )


def recover_interrupted_checkpoint_publication(
    resolved_config: Mapping[str, object],
) -> None:
    """Publish only a fully authenticated checkpoint left by an interrupted write."""

    config = (
        _validate_test_resolved_config(resolved_config)
        if resolved_config.get("test_only") is True
        else _validate_production_resolved_config(resolved_config)
    )
    _revalidate_static(config)
    run_dir = reject_symlink_components(Path(config["run"]["path"]))
    config_hash = canonical_hash(config)
    _validate_audits(run_dir, config, config_hash)
    _recover_checkpoint_publication(run_dir, config, config_hash)
    _revalidate_static(config)


def validate_resumable_run(
    resolved_config: Mapping[str, object], *, expected_completed_updates: int | None = None
) -> ResumableRun:
    config = (
        _validate_test_resolved_config(resolved_config)
        if resolved_config.get("test_only") is True
        else _validate_production_resolved_config(resolved_config)
    )
    _revalidate_static(config)
    run_dir = reject_symlink_components(Path(config["run"]["path"]))
    config_hash = canonical_hash(config)
    _validate_audits(run_dir, config, config_hash)
    status = _read_status(run_dir)
    if status["state"] != "running" or status["config_sha256"] != config_hash:
        raise ValueError("resumable run requires compatible running STATUS")
    completed = status["completed_updates"]
    if expected_completed_updates is not None and completed != expected_completed_updates:
        raise ValueError("resumable run is at the wrong completed update")
    payload, digest, _ = _load_chain(
        run_dir,
        config=config,
        config_hash=config_hash,
        expected_completed=completed,
        allow_test_steps=config.get("test_only") is True,
    )
    if payload["completed_updates"] != completed:
        raise ValueError("checkpoint and STATUS completed updates differ")
    _revalidate_static(config)
    return ResumableRun(str(run_dir), config_hash, completed, digest)


def _artifact_names(total: int) -> set[str]:
    names = set(_TOP_AUDIT_FILES) | {"checkpoints/LATEST.json"}
    names.update(f"checkpoints/{_checkpoint_name(step)}" for step in required_checkpoint_steps(total))
    names.update(f"diagnostics/step_{step:04d}.json" for step in required_diagnostic_steps(total))
    names.update(
        {
            "artifacts/final_logits.npy", "artifacts/final_density.npy",
            "artifacts/final_hard_005.npy", "artifacts/metrics.json",
            "artifacts/audit.json", "artifacts/resource.json",
        }
    )
    return names


def _validate_completed_namespace(
    run_dir: Path,
    config: Mapping[str, object],
    chain_digests: Mapping[int, str],
) -> None:
    expected_files = _artifact_names(int(config["updates"])) | {
        "STATUS.json",
        "artifacts/artifacts.json",
    }
    if config.get("test_only") is True:
        expected_files.update(
            f"checkpoints/{_checkpoint_name(completed)}"
            for completed in chain_digests
        )
    expected_directories: set[str] = set()
    for relative in expected_files:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for entry in run_dir.rglob("*"):
        reject_symlink_components(entry)
        relative = entry.relative_to(run_dir).as_posix()
        if entry.is_file():
            observed_files.add(relative)
        elif entry.is_dir():
            observed_directories.add(relative)
        else:
            raise ValueError("completed run namespace contains a non-file entry")
    if observed_files != expected_files or observed_directories != expected_directories:
        raise ValueError("completed run namespace has extra or missing entries")


def _decode_canonical_bytes(raw: bytes, *, label: str) -> dict[str, object]:
    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}") from error
    if type(payload) is not dict or canonical_json_bytes(payload) != raw:
        raise ValueError(f"{label} must be an exact canonical JSON object")
    return payload


def _load_exact_npy(
    run_dir: Path,
    relative: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    expected_sha256: str,
) -> np.ndarray:
    raw, digest = _stable_file_bytes(
        relative_regular_file(run_dir, relative), label=relative
    )
    if digest != expected_sha256:
        raise ValueError(f"completed artifact hash mismatch: {relative}")
    buffer = io.BytesIO(raw)
    try:
        array = np.load(buffer, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"terminal artifact is not an exact NPY array: {relative}") from error
    if buffer.tell() != len(raw):
        raise ValueError(f"terminal NPY has trailing bytes: {relative}")
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != dtype
        or array.shape != shape
        or not array.flags.c_contiguous
    ):
        raise ValueError(f"terminal NPY dtype/shape/layout mismatch: {relative}")
    return array


def _expected_terminal_audit(
    checkpoint: Mapping[str, object], config: Mapping[str, object]
) -> dict[str, object]:
    audit = copy.deepcopy(checkpoint["audit"])
    flags = PROFILE_FLAGS[str(config["arm"])]
    for name in ("j", "h"):
        eligible = int(audit[f"{name}_full_ramp_eligible"])
        audit[f"{name}_full_ramp_raw_zero_fraction"] = (
            int(audit[f"{name}_full_ramp_zero"]) / eligible
            if flags[name] and eligible > 0
            else None
        )
    return audit


def _expected_resource(
    checkpoint: Mapping[str, object], config: Mapping[str, object]
) -> dict[str, object]:
    history = checkpoint["history"]
    times = checkpoint["step_times"]
    device = torch.device(str(config["device"]))
    return {
        "schema": "pcraft_soft_loss_v6_resource_v1",
        "step_times": list(times),
        "all_finite_positive": all(
            type(value) is float and math.isfinite(value) and value > 0.0
            for value in times
        ),
        "peak_cuda_allocated_bytes": max(
            (int(row["peak_cuda_allocated_bytes"]) for row in history), default=0
        ),
        "device_total_memory_bytes": (
            int(torch.cuda.get_device_properties(device).total_memory)
            if device.type == "cuda"
            else None
        ),
        "cuda_executed": device.type == "cuda",
    }


def _recompute_terminal_state(
    config: Mapping[str, object], checkpoint_logits: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    runtime = _build_runtime(config)
    with torch.no_grad():
        runtime.logits.copy_(checkpoint_logits.to(runtime.logits.device))
        density = torch.sigmoid(runtime.logits)
        hard = density >= 0.05
        readout = render_thresholded(runtime.protocol, density, 0.05)
        metrics = evaluate_six_view_metrics(
            readout.mask,
            runtime.targets >= 0.5,
            completed_updates=int(config["updates"]),
        )
    return (
        density.detach().cpu().contiguous().numpy().astype(np.float32, copy=False),
        hard.detach().cpu().contiguous().numpy().astype(np.uint8, copy=False),
        metrics,
    )


def _validate_terminal_evidence(
    run_dir: Path,
    config: Mapping[str, object],
    config_hash: str,
    checkpoint: Mapping[str, object],
    chain_digests: Mapping[int, str],
    *,
    expected_manifest_sha256: str,
) -> dict[str, object]:
    _validate_completed_namespace(run_dir, config, chain_digests)
    manifest_path = relative_regular_file(run_dir, "artifacts/artifacts.json")
    raw_manifest, manifest_digest = _stable_file_bytes(
        manifest_path, label="artifact manifest"
    )
    if manifest_digest != expected_manifest_sha256:
        raise ValueError("completed artifact manifest hash mismatch")
    manifest = _decode_canonical_bytes(raw_manifest, label="artifact manifest")
    if (
        set(manifest) != {"schema", "config_sha256", "files_sha256"}
        or manifest["schema"] != "pcraft_soft_loss_v6_artifacts_v1"
        or manifest["config_sha256"] != config_hash
    ):
        raise ValueError("artifact manifest schema/config mismatch")
    total = int(config["updates"])
    expected = _artifact_names(total)
    files = manifest["files_sha256"]
    if type(files) is not dict or set(files) != expected:
        raise ValueError("artifact manifest has extra or missing files")
    for relative, expected_digest in files.items():
        if (
            type(expected_digest) is not str
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise ValueError(f"completed artifact digest is invalid: {relative}")
        snapshot = snapshot_regular_file(relative_regular_file(run_dir, relative))
        if snapshot.sha256 != expected_digest:
            raise ValueError(f"completed artifact hash mismatch: {relative}")
        revalidate_snapshot(snapshot)
    for completed, digest in chain_digests.items():
        relative = f"checkpoints/{_checkpoint_name(completed)}"
        if relative in files and files[relative] != digest:
            raise ValueError("artifact manifest differs from authenticated checkpoint chain")
    resolution = int(config["resolution"])
    shape = (1, 1, resolution, resolution, resolution)
    logits = _load_exact_npy(
        run_dir,
        "artifacts/final_logits.npy",
        dtype=np.dtype(np.float32),
        shape=shape,
        expected_sha256=files["artifacts/final_logits.npy"],
    )
    density = _load_exact_npy(
        run_dir,
        "artifacts/final_density.npy",
        dtype=np.dtype(np.float32),
        shape=shape,
        expected_sha256=files["artifacts/final_density.npy"],
    )
    hard = _load_exact_npy(
        run_dir,
        "artifacts/final_hard_005.npy",
        dtype=np.dtype(np.uint8),
        shape=shape,
        expected_sha256=files["artifacts/final_hard_005.npy"],
    )
    checkpoint_array = checkpoint["logits"].contiguous().numpy()
    if not np.isfinite(logits).all() or not np.array_equal(logits, checkpoint_array):
        raise ValueError("terminal logits differ from the authenticated final checkpoint")
    expected_density, expected_hard, recomputed_metrics = _recompute_terminal_state(
        config, checkpoint["logits"]
    )
    if (
        not np.isfinite(density).all()
        or np.any(density < 0.0)
        or np.any(density > 1.0)
        or not np.array_equal(density, expected_density)
    ):
        raise ValueError("terminal density is not registered-device sigmoid(final logits)")
    if not np.isin(hard, (0, 1)).all() or not np.array_equal(hard, expected_hard):
        raise ValueError("terminal hard occupancy differs from density >= .05")
    semantic_json = {
        "metrics": recomputed_metrics,
        "audit": _expected_terminal_audit(checkpoint, config),
        "resource": _expected_resource(checkpoint, config),
    }
    observed: dict[str, dict[str, object]] = {}
    for name, expected_payload in semantic_json.items():
        relative = f"artifacts/{name}.json"
        raw, digest = _stable_file_bytes(
            relative_regular_file(run_dir, relative), label=name
        )
        if digest != files[relative]:
            raise ValueError(f"completed artifact hash mismatch: {relative}")
        payload = _decode_canonical_bytes(raw, label=name)
        if canonical_json_bytes(payload) != canonical_json_bytes(expected_payload):
            raise ValueError(f"terminal {name} differs from independent reconstruction")
        observed[name] = payload
    for step in required_diagnostic_steps(total):
        relative = f"diagnostics/step_{step:04d}.json"
        raw, digest = _stable_file_bytes(
            relative_regular_file(run_dir, relative), label="diagnostic"
        )
        if digest != files[relative]:
            raise ValueError(f"completed artifact hash mismatch: {relative}")
        diagnostic = _decode_canonical_bytes(raw, label="diagnostic")
        expected_diagnostic = {
            "schema": "pcraft_soft_loss_v6_diagnostic_v1",
            "completed_updates": step,
            "telemetry": checkpoint["history"][step - 1],
        }
        if canonical_json_bytes(diagnostic) != canonical_json_bytes(expected_diagnostic):
            raise ValueError("diagnostic is not the exact telemetry snapshot")
    _validate_completed_namespace(run_dir, config, chain_digests)
    return observed["metrics"]


def validate_completed_run(resolved_config: Mapping[str, object]) -> CompletedRun:
    config = (
        _validate_test_resolved_config(resolved_config)
        if resolved_config.get("test_only") is True
        else _validate_production_resolved_config(resolved_config)
    )
    _revalidate_static(config)
    run_dir = reject_symlink_components(Path(config["run"]["path"]))
    config_hash = canonical_hash(config)
    _validate_audits(run_dir, config, config_hash)
    total = int(config["updates"])
    status = _read_status(run_dir)
    if status["state"] != "completed" or status["config_sha256"] != config_hash or status["completed_updates"] != total:
        raise ValueError("completed STATUS does not match this exact final run")
    checkpoint, _, chain_digests = _load_chain(
        run_dir,
        config=config,
        config_hash=config_hash,
        expected_completed=total,
        allow_test_steps=config.get("test_only") is True,
    )
    metrics = _validate_terminal_evidence(
        run_dir,
        config,
        config_hash,
        checkpoint,
        chain_digests,
        expected_manifest_sha256=str(status["artifacts_sha256"]),
    )
    _revalidate_static(config)
    return CompletedRun(str(run_dir), config_hash, total, str(status["artifacts_sha256"]), metrics)


def _initial_audit(arm: str) -> dict[str, object]:
    return {
        "schema": "pcraft_soft_loss_v6_audit_v1",
        "arm": arm,
        "eligible_updates": 0,
        "j_zero_updates": 0,
        "h_zero_updates": 0,
        "j_active_updates": 0,
        "h_active_updates": 0,
        "parameters_changed_updates": 0,
        "clip_calls": 0,
        "optimizer_steps": 0,
        "max_j_ratio": 0.0,
        "max_h_ratio": 0.0,
        "max_combined_ratio": 0.0,
        "j_full_ramp_eligible": 0,
        "h_full_ramp_eligible": 0,
        "j_full_ramp_zero": 0,
        "h_full_ramp_zero": 0,
    }


def _update_audit(audit: dict[str, object], row: StepTelemetry, arm: str) -> None:
    audit["clip_calls"] += row.clip_calls
    audit["optimizer_steps"] += row.optimizer_steps
    if row.parameter_delta_norm > 0.0:
        audit["parameters_changed_updates"] += 1
    audit["max_j_ratio"] = max(audit["max_j_ratio"], row.j_ratio)
    audit["max_h_ratio"] = max(audit["max_h_ratio"], row.h_ratio)
    audit["max_combined_ratio"] = max(audit["max_combined_ratio"], row.combined_ratio)
    if row.auxiliary_multiplier > 0.0:
        audit["eligible_updates"] += 1
        for name, enabled in (("j", PROFILE_FLAGS[arm]["j"]), ("h", PROFILE_FLAGS[arm]["h"])):
            if not enabled:
                continue
            if getattr(row, f"{name}_zero"):
                audit[f"{name}_zero_updates"] += 1
            else:
                audit[f"{name}_active_updates"] += 1
            if row.completed_updates >= 901:
                audit[f"{name}_full_ramp_eligible"] += 1
                if getattr(row, f"{name}_zero"):
                    audit[f"{name}_full_ramp_zero"] += 1


def _finalize(
    run_dir: Path,
    config: Mapping[str, object],
    config_hash: str,
    runtime: _Runtime,
    audit: dict[str, object],
    history: Sequence[Mapping[str, object]],
    step_times: Sequence[float],
) -> CompletedRun:
    total = int(config["updates"])
    with torch.no_grad():
        density = torch.sigmoid(runtime.logits)
        readout = render_thresholded(runtime.protocol, density, 0.05)
        metrics = evaluate_six_view_metrics(readout.mask, runtime.targets >= 0.5, completed_updates=total)
    for name, array in (
        ("final_logits.npy", runtime.logits.detach().cpu().numpy().astype(np.float32)),
        ("final_density.npy", density.detach().cpu().numpy().astype(np.float32)),
        ("final_hard_005.npy", (density >= 0.05).detach().cpu().numpy().astype(np.uint8)),
    ):
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        atomic_bytes(run_dir / "artifacts" / name, buffer.getvalue(), create_once=True)
    flags = PROFILE_FLAGS[str(config["arm"])]
    for name in ("j", "h"):
        eligible = int(audit[f"{name}_full_ramp_eligible"])
        audit[f"{name}_full_ramp_raw_zero_fraction"] = (
            int(audit[f"{name}_full_ramp_zero"]) / eligible
            if flags[name] and eligible > 0
            else None
        )
    resources = {
        "schema": "pcraft_soft_loss_v6_resource_v1",
        "step_times": list(step_times),
        "all_finite_positive": all(math.isfinite(value) and value > 0.0 for value in step_times),
        "peak_cuda_allocated_bytes": max(
            (int(row["peak_cuda_allocated_bytes"]) for row in history), default=0
        ),
        "device_total_memory_bytes": (
            int(torch.cuda.get_device_properties(runtime.logits.device).total_memory)
            if runtime.logits.device.type == "cuda"
            else None
        ),
        "cuda_executed": runtime.logits.device.type == "cuda",
    }
    atomic_json(run_dir / "artifacts/metrics.json", metrics, create_once=True)
    atomic_json(run_dir / "artifacts/audit.json", audit, create_once=True)
    atomic_json(run_dir / "artifacts/resource.json", resources, create_once=True)
    files = {
        relative: sha256_file(relative_regular_file(run_dir, relative))
        for relative in sorted(_artifact_names(total))
    }
    artifact = {
        "schema": "pcraft_soft_loss_v6_artifacts_v1",
        "config_sha256": config_hash,
        "files_sha256": files,
    }
    artifact_path = run_dir / "artifacts/artifacts.json"
    atomic_json(artifact_path, artifact, create_once=True)
    artifact_hash = sha256_file(artifact_path)
    for relative, digest in files.items():
        if sha256_file(relative_regular_file(run_dir, relative)) != digest:
            raise TrainingStateError("artifact changed before terminal publication")
    checkpoint, _, chain_digests = _load_chain(
        run_dir,
        config=config,
        config_hash=config_hash,
        expected_completed=total,
        allow_test_steps=config.get("test_only") is True,
    )
    validated_metrics = _validate_terminal_evidence(
        run_dir,
        config,
        config_hash,
        checkpoint,
        chain_digests,
        expected_manifest_sha256=artifact_hash,
    )
    atomic_json(
        run_dir / "STATUS.json",
        {
            "schema": _STATUS_SCHEMA,
            "state": "completed",
            "config_sha256": config_hash,
            "completed_updates": total,
            "artifacts_sha256": artifact_hash,
        },
    )
    status = _read_status(run_dir)
    if (
        status["state"] != "completed"
        or status["config_sha256"] != config_hash
        or status["completed_updates"] != total
        or status["artifacts_sha256"] != artifact_hash
    ):
        raise TrainingStateError("terminal STATUS publication failed authentication")
    _revalidate_static(config)
    return CompletedRun(
        str(run_dir), config_hash, total, artifact_hash, validated_metrics
    )


def _train_impl(
    config: dict[str, object], *, resume: bool, absolute_cap: int
) -> ResumableRun | CompletedRun:
    config_hash = canonical_hash(config)
    run_dir = Path(config["run"]["path"])
    total = int(config["updates"])
    device = torch.device(config["device"])
    if absolute_cap > total or absolute_cap <= 0:
        raise ValueError("max_updates absolute cap exceeds the configured total")
    _revalidate_static(config)
    if resume:
        recover_interrupted_checkpoint_publication(config)
        state = validate_resumable_run(config)
        runtime = _build_runtime(config)
        checkpoint, predecessor, _ = _load_chain(
            run_dir,
            config=config,
            config_hash=config_hash,
            expected_completed=state.completed_updates,
            allow_test_steps=config.get("test_only") is True,
        )
        with torch.no_grad():
            runtime.logits.copy_(checkpoint["logits"].to(device))
        runtime.optimizer.load_state_dict(checkpoint["optimizer"])
        _restore_rng(checkpoint["rng"], device)
        completed = int(checkpoint["completed_updates"])
        audit = copy.deepcopy(checkpoint["audit"])
        history = copy.deepcopy(checkpoint["history"])
        step_times = list(checkpoint["step_times"])
    else:
        if run_dir.exists():
            raise FileExistsError("fresh run path must be absent")
        run_dir.mkdir(parents=True)
        runtime = _build_runtime(config)
        _seed(17, device)
        completed = 0
        audit = _initial_audit(str(config["arm"]))
        history: list[dict[str, object]] = []
        step_times: list[float] = []
        expected_audits = _expected_audits(config, config_hash)
        for name, payload in expected_audits.items():
            atomic_json(run_dir / name, payload, create_once=True)
        atomic_json(
            run_dir / "STATUS.json",
            {"schema": _STATUS_SCHEMA, "state": "running", "config_sha256": config_hash, "completed_updates": 0},
            create_once=True,
        )
        predecessor = _write_checkpoint(
            run_dir,
            config_hash=config_hash,
            runtime=runtime,
            completed=0,
            predecessor_sha256=None,
            audit=audit,
            history=history,
            step_times=step_times,
        )
    if absolute_cap < completed:
        raise ValueError("max_updates absolute cap is behind durable progress")
    durable_completed = completed
    try:
        while completed < absolute_cap:
            try:
                row = _train_one_update(
                    logits=runtime.logits,
                    optimizer=runtime.optimizer,
                    protocol=runtime.protocol,
                    targets=runtime.targets,
                    arm=str(config["arm"]),
                    completed_updates=completed,
                    timer=(
                        _TestStepTimer()
                        if config.get("test_only") is True
                        else time.perf_counter
                    ),
                )
            except ValueError as error:
                raise TrainingStateError(str(error)) from error
            completed = row.completed_updates
            payload = asdict(row)
            history.append(payload)
            step_times.append(row.elapsed_seconds)
            _update_audit(audit, row, str(config["arm"]))
            if completed in required_diagnostic_steps(total):
                atomic_json(
                    run_dir / "diagnostics" / f"step_{completed:04d}.json",
                    {"schema": "pcraft_soft_loss_v6_diagnostic_v1", "completed_updates": completed, "telemetry": payload},
                    create_once=True,
                )
            if completed in required_checkpoint_steps(total) or completed == absolute_cap:
                predecessor = _write_checkpoint(
                    run_dir,
                    config_hash=config_hash,
                    runtime=runtime,
                    completed=completed,
                    predecessor_sha256=predecessor,
                    audit=audit,
                    history=history,
                    step_times=step_times,
                )
                durable_completed = completed
                atomic_json(
                    run_dir / "STATUS.json",
                    {"schema": _STATUS_SCHEMA, "state": "running", "config_sha256": config_hash, "completed_updates": completed},
                )
    except TrainingStateError as error:
        atomic_json(
            run_dir / "STATUS.json",
            {
                "schema": _STATUS_SCHEMA,
                "state": "failed",
                "config_sha256": config_hash,
                "completed_updates": durable_completed,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    _revalidate_static(config)
    if completed < total:
        return validate_resumable_run(config, expected_completed_updates=completed)
    return _finalize(run_dir, config, config_hash, runtime, audit, history, step_times)


def train_cell(
    resolved_config: Mapping[str, object], *, resume: bool = False, max_updates: int | None = None
) -> ResumableRun | CompletedRun:
    """Train one production cell to an absolute registered orchestration cap."""

    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    if max_updates is not None and (type(max_updates) is not int or max_updates not in {2, 200, 3000}):
        raise ValueError("max_updates must be one of 2, 200, 3000")
    config = _validate_production_resolved_config(resolved_config)
    cap = 3000 if max_updates is None else max_updates
    return _train_impl(config, resume=resume, absolute_cap=cap)


def _train_cell_for_test(
    resolved_config: Mapping[str, object], *, resume: bool = False, max_updates: int | None = None
) -> ResumableRun | CompletedRun:
    config = _validate_test_resolved_config(resolved_config)
    cap = int(config["updates"]) if max_updates is None else max_updates
    if type(cap) is not int or cap <= 0:
        raise ValueError("test max_updates must be a positive integer absolute cap")
    return _train_impl(config, resume=resume, absolute_cap=cap)
