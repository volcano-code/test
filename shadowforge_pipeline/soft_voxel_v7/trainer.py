"""Authenticated renderer-consistent trainer and durable lifecycle for soft-loss v7."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass, fields
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import time
from types import MappingProxyType, SimpleNamespace
from typing import Callable
import uuid

import cv2
import numpy as np
import torch

from shadowforge_pipeline.soft_voxel_v3.data import normalize_target
from shadowforge_pipeline.soft_voxel_v3.losses import balanced_lp_loss
from shadowforge_pipeline.soft_voxel_v3.thresholding import render_thresholded

from .core import (
    build_registered_target_stack,
    build_registered_visual_hull_logits,
    compose_v7_gradient,
    coverage_hit_loss,
    leakage_barrier_loss,
    v7_auxiliary_multiplier,
    v7_auxiliary_slot,
    v7_learning_rate,
)
from .protocol import RegisteredEARProtocol, registered_camera_payload
from .provenance import (
    canonical_json_bytes,
    canonical_sha256,
    decode_canonical_json,
    open_v7_root_transaction,
    revalidate_snapshot,
    snapshot_regular_file,
)
from .registry import (
    ENGINEERING_ROW_SCHEMA,
    FAILURE_RECORD_SCHEMA,
    METRIC_ROW_SCHEMA,
    V7StageMatrix,
    rebuild_resolved_config,
    revalidate_static_identities,
    resolve_stage_matrix,
    validate_campaign_manifest,
    validate_stage_authorization,
)


class TrainingStateError(RuntimeError):
    """The durable or scientific state failed closed authentication."""


class _NonfiniteTrainingError(TrainingStateError):
    def __init__(
        self, message: str, *, loss_finite: bool,
        gradient_finite: bool, parameters_finite: bool,
    ) -> None:
        super().__init__(message)
        self.finite_flags = (
            loss_finite, gradient_finite, parameters_finite,
        )


@dataclass(frozen=True)
class StepTelemetry:
    completed_updates: int
    learning_rate: float
    multiplier: float
    slot: str
    component: str
    base_loss: float
    aux_loss: float | None
    auxiliary_constructed: bool
    base_gradient_norm: float
    raw_auxiliary_gradient_norm: float
    final_auxiliary_gradient_norm: float
    base_auxiliary_dot: float
    effective_ratio: float
    clip_count: int
    optimizer_step_count: int
    step_seconds: float
    cuda_executed: bool
    peak_allocated_bytes: int
    diagnostic_sha256: str


@dataclass(frozen=True)
class AuthenticatedCheckpoint:
    run_dir: Path
    config_sha256: str
    completed_updates: int
    checkpoint_path: Path
    checkpoint_sha256: str
    predecessor_checkpoint_sha256: str | None
    stage: str
    stage_matrix_sha256: str
    authorization_path: str | None
    authorization_sha256: str | None
    history_length: int


@dataclass(frozen=True)
class AuthenticatedEndpointMetrics:
    path: Path
    sha256: str
    completed_updates: int
    checkpoint_sha256: str
    stage_matrix_sha256: str
    per_view: tuple[Mapping[str, object], ...]
    aggregate: Mapping[str, object]


@dataclass(frozen=True)
class ResumableRun:
    run_dir: Path
    config_sha256: str
    status_path: Path
    status_sha256: str
    current_checkpoint: AuthenticatedCheckpoint
    endpoint_metrics: tuple[AuthenticatedEndpointMetrics, ...]


@dataclass(frozen=True)
class CompletedRun:
    run_dir: Path
    config_sha256: str
    status_path: Path
    status_sha256: str
    current_checkpoint: AuthenticatedCheckpoint
    artifact_manifest_path: Path
    artifact_manifest_sha256: str
    artifacts: Mapping[str, str]
    endpoint_metrics: tuple[AuthenticatedEndpointMetrics, ...]


@dataclass(frozen=True)
class FailedRun:
    run_dir: Path
    config_sha256: str
    status_path: Path
    status_sha256: str
    last_checkpoint: AuthenticatedCheckpoint
    observation_path: Path
    observation_sha256: str
    failure_record_path: Path
    failure_record_sha256: str
    endpoint_metrics: tuple[AuthenticatedEndpointMetrics, ...]


_CHECKPOINT_SCHEMA = "pcraft_soft_loss_v7_checkpoint_v1"
_OPTIMIZER_SCHEMA = "pcraft_soft_loss_v7_adam_state_v1"
_RNG_SCHEMA = "pcraft_soft_loss_v7_rng_v1"
_AUDIT_SCHEMA = "pcraft_soft_loss_v7_audit_v1"
_RESOURCE_SCHEMA = "pcraft_soft_loss_v7_resource_v1"
_LATEST_SCHEMA = "pcraft_soft_loss_v7_latest_v1"
_STATUS_SCHEMA = "pcraft_soft_loss_v7_status_v1"
_METRICS_SCHEMA = "pcraft_soft_loss_v7_endpoint_metrics_v1"
_MANIFEST_SCHEMA = "pcraft_soft_loss_v7_artifact_manifest_v1"
_OBSERVATION_SCHEMA = "pcraft_soft_loss_v7_failure_observation_v1"
_CONFIG_SCHEMA = "pcraft_soft_loss_v7_resolved_config_v1"
_ENGINEERING_PREFIX_SCHEMA = "pcraft_soft_loss_v7_engineering_prefix_v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_RE = re.compile(r"step_([0-9]{4})\.pt\Z")
_TELEMETRY_FIELDS = tuple(field.name for field in fields(StepTelemetry))


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str or key in result:
                raise ValueError("mapping keys must be unique exact strings")
            result[key] = _plain(nested)
        return result
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _deep_immutable(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _deep_immutable(nested) for key, nested in value.items()})
    if type(value) is list:
        return tuple(_deep_immutable(nested) for nested in value)
    if type(value) is tuple:
        return tuple(_deep_immutable(nested) for nested in value)
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise TrainingStateError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TrainingStateError(f"{label} must be an exact integer >= {minimum}")
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (nonnegative and value < 0.0):
        raise TrainingStateError(f"{label} must be a finite float")
    return value


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha(tensor: torch.Tensor) -> str:
    return _sha_bytes(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))


def _semantic(payload: Mapping[str, object]) -> dict[str, object]:
    result = copy.deepcopy(dict(payload))
    result["semantic_sha256"] = canonical_sha256(result)
    return result


def required_checkpoint_steps(total_updates: int = 3000) -> tuple[int, ...]:
    if type(total_updates) is not int or total_updates <= 0:
        raise ValueError("total_updates must be a positive exact integer")
    values = {0, min(2, total_updates), total_updates}
    values.update(step for step in range(200, total_updates + 1, 200))
    for step in (300, 1000, 2400, 2700, 3000):
        if step <= total_updates:
            values.add(step)
    return tuple(sorted(values))


def required_diagnostic_steps(total_updates: int = 3000) -> tuple[int, ...]:
    if type(total_updates) is not int or total_updates <= 0:
        raise ValueError("total_updates must be a positive exact integer")
    return (2400, 2700, 3000) if total_updates == 3000 else ()


def evaluate_six_view_metrics(
    projection: torch.Tensor, targets: torch.Tensor,
) -> Mapping[str, object]:
    if (
        not isinstance(projection, torch.Tensor)
        or projection.dtype != torch.float32
        or projection.ndim != 3
        or projection.shape[0] != 6
        or any(size == 0 for size in projection.shape)
        or not torch.isfinite(projection).all().item()
    ):
        raise ValueError("projection must be finite float32 [6,H,W]")
    if (
        not isinstance(targets, torch.Tensor)
        or targets.dtype != torch.bool
        or targets.shape != projection.shape
        or targets.device != projection.device
    ):
        raise ValueError("targets must be matching boolean [6,H,W]")
    maxima = projection.detach().amax(dim=(-2, -1), keepdim=True)
    if not (maxima > 0.0).all().item():
        raise ValueError("each projection view maximum must be strictly positive")
    predicted = projection >= projection.new_tensor(0.8) * maxima
    rows: list[dict[str, object]] = []
    for view in range(6):
        guess, truth = predicted[view], targets[view]
        tp = int(torch.count_nonzero(guess & truth).item())
        fp = int(torch.count_nonzero(guess & ~truth).item())
        fn = int(torch.count_nonzero(~guess & truth).item())
        tn = int(torch.count_nonzero(~guess & ~truth).item())
        union, dice_den, foreground, background = tp + fp + fn, 2 * tp + fp + fn, tp + fn, fp + tn
        if min(union, dice_den, foreground, background) <= 0:
            raise ValueError(f"metric denominator is zero in view {view}")
        rows.append({
            "view": view, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": tp / union, "dice": (2 * tp) / dice_den,
            "recall": tp / foreground, "fpr": fp / background,
        })
    aggregate = {
        f"mean_{name}": sum(float(row[name]) for row in rows) / 6.0
        for name in ("iou", "dice", "recall", "fpr")
    }
    return _deep_immutable({"per_view": rows, "aggregate": aggregate})


class _DeterministicTestTimer:
    def __init__(self) -> None:
        self._calls = 0

    def __call__(self) -> float:
        value = self._calls * 0.001
        self._calls += 1
        return value


def _test_render_thresholded(protocol: object, density: torch.Tensor, tau: float) -> object:
    if tau != 0.05:
        raise ValueError("test STE threshold differs")
    projection = protocol.render_all(density)
    maximum = projection.detach().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return SimpleNamespace(projection=projection, mask=projection >= 0.8 * maximum)


def _gradient_norm(value: torch.Tensor) -> float:
    result = float(torch.linalg.vector_norm(value.detach().double()).item())
    if not math.isfinite(result):
        raise _NonfiniteTrainingError(
            "gradient norm must be finite", loss_finite=True,
            gradient_finite=False, parameters_finite=True,
        )
    return result


def _train_one_update(
    *,
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    protocol: object,
    targets: torch.Tensor,
    profile: str,
    budget_ratio: float | None,
    completed_updates: int,
    render_thresholded_fn: Callable[[object, torch.Tensor, float], object] = render_thresholded,
    allow_test_size: bool = False,
    timer: Callable[[], float] = time.perf_counter,
) -> StepTelemetry:
    if profile not in ("base", "hit_only", "barrier_only", "full"):
        raise ValueError("profile is not registered")
    if profile == "base":
        if budget_ratio is not None:
            raise ValueError("Base cannot carry a budget")
    elif type(budget_ratio) is not float or budget_ratio not in (0.10, 0.15, 0.20):
        raise ValueError("nonbase profile requires a registered ratio")
    if (
        not isinstance(logits, torch.Tensor) or logits.dtype not in (torch.float32, torch.float64)
        or not logits.is_leaf or not logits.requires_grad or logits.ndim != 5
        or logits.shape[:2] != (1, 1) or not torch.isfinite(logits).all().item()
    ):
        raise ValueError("logits must be finite leaf floating [1,1,D,H,W]")
    if (
        not isinstance(targets, torch.Tensor) or targets.dtype != torch.bool
        or targets.ndim != 3 or targets.shape[0] != 6 or targets.device != logits.device
    ):
        raise ValueError("targets must be six boolean views on the logits device")
    if type(completed_updates) is not int:
        raise ValueError("completed_updates must be exact integer")
    started = float(timer())
    learning_rate = v7_learning_rate(completed_updates)
    multiplier = v7_auxiliary_multiplier(completed_updates)
    slot = "hit" if completed_updates % 4 in (0, 2) else "barrier"
    component = v7_auxiliary_slot(completed_updates, profile)
    if multiplier == 0.0:
        component = "base"

    density_base = torch.sigmoid(logits)
    projection_base = protocol.render_all(density_base)
    target_float = targets.to(dtype=projection_base.dtype)
    base_loss_tensor = (
        balanced_lp_loss(projection_base, target_float, p=2)
        + projection_base.new_tensor(0.35) * balanced_lp_loss(projection_base, target_float, p=1)
    ).mean()
    if not torch.isfinite(base_loss_tensor).item():
        raise _NonfiniteTrainingError(
            "base loss must be finite", loss_finite=False,
            gradient_finite=True, parameters_finite=True,
        )
    (base_grad,) = torch.autograd.grad(base_loss_tensor, logits, retain_graph=False)
    if not torch.isfinite(base_grad).all().item():
        raise _NonfiniteTrainingError(
            "base gradient must be finite", loss_finite=True,
            gradient_finite=False, parameters_finite=True,
        )

    aux_loss_tensor: torch.Tensor | None = None
    aux_grad: torch.Tensor | None = None
    if component != "base":
        density_aux = torch.sigmoid(logits)
        readout = render_thresholded_fn(protocol, density_aux, 0.05)
        projection_aux = readout.projection
        if component == "hit":
            aux_loss_tensor = coverage_hit_loss(
                projection_aux, targets, _allow_test_size=allow_test_size
            )
        else:
            aux_loss_tensor = leakage_barrier_loss(
                projection_aux, targets, density_aux, _allow_test_size=allow_test_size
            )
        if not torch.isfinite(aux_loss_tensor).item():
            raise _NonfiniteTrainingError(
                "auxiliary loss must be finite", loss_finite=False,
                gradient_finite=True, parameters_finite=True,
            )
        (aux_grad,) = torch.autograd.grad(aux_loss_tensor, logits, retain_graph=False)
        if not torch.isfinite(aux_grad).all().item():
            raise _NonfiniteTrainingError(
                "auxiliary gradient must be finite", loss_finite=True,
                gradient_finite=False, parameters_finite=True,
            )

    base_norm = _gradient_norm(base_grad)
    raw_norm = 0.0 if aux_grad is None else _gradient_norm(aux_grad)
    if aux_grad is None:
        final_grad = base_grad.clone()
        final_aux_norm = 0.0
        ratio = 0.0
    else:
        composed, diagnostics = compose_v7_gradient(
            base_grad, aux_grad, float(budget_ratio), multiplier,
            return_telemetry=True,
        )
        final_grad = composed
        final_aux_norm = float(diagnostics["final_auxiliary_norm"])
        ratio = float(diagnostics["final_auxiliary_base_ratio"])
    effective_auxiliary = final_grad.detach().double() - base_grad.detach().double()
    base_dot = float(torch.dot(
        base_grad.detach().double().reshape(-1), effective_auxiliary.reshape(-1)
    ).item())
    if not math.isfinite(base_dot) or base_dot < 0.0:
        raise _NonfiniteTrainingError(
            "represented base auxiliary dot must be finite and nonnegative",
            loss_finite=True, gradient_finite=False, parameters_finite=True,
        )
    if not torch.isfinite(final_grad).all().item():
        raise _NonfiniteTrainingError(
            "final gradient must be finite", loss_finite=True,
            gradient_finite=False, parameters_finite=True,
        )
    optimizer.zero_grad(set_to_none=True)
    logits.grad = final_grad.detach().clone()
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    returned = torch.nn.utils.clip_grad_norm_([logits], 1.0)
    if not math.isfinite(float(returned)) or logits.grad is None or not torch.isfinite(logits.grad).all().item():
        raise _NonfiniteTrainingError(
            "clipped gradient must be finite", loss_finite=True,
            gradient_finite=False, parameters_finite=True,
        )
    optimizer.step()
    if not torch.isfinite(logits).all().item():
        raise _NonfiniteTrainingError(
            "parameters must remain finite", loss_finite=True,
            gradient_finite=True, parameters_finite=False,
        )
    elapsed = float(timer()) - started
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise TrainingStateError("step time must be finite and nonnegative")
    peak = int(torch.cuda.max_memory_allocated(logits.device)) if logits.device.type == "cuda" else 0
    diagnostic_unsigned = {
        "completed_updates": completed_updates + 1,
        "learning_rate": learning_rate,
        "multiplier": multiplier,
        "slot": slot,
        "component": component,
        "base_loss": float(base_loss_tensor.detach().item()),
        "aux_loss": None if aux_loss_tensor is None else float(aux_loss_tensor.detach().item()),
        "base_gradient_norm": base_norm,
        "raw_auxiliary_gradient_norm": raw_norm,
        "final_auxiliary_gradient_norm": final_aux_norm,
    }
    return StepTelemetry(
        completed_updates + 1, learning_rate, multiplier, slot, component,
        float(base_loss_tensor.detach().item()),
        None if aux_loss_tensor is None else float(aux_loss_tensor.detach().item()),
        aux_loss_tensor is not None, base_norm, raw_norm, final_aux_norm, base_dot,
        ratio, 1, 1, elapsed, logits.device.type == "cuda", peak,
        canonical_sha256(diagnostic_unsigned),
    )


class _TestProtocol:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def render_all(self, density: torch.Tensor) -> torch.Tensor:
        view = density[0, 0].mean(dim=0)
        return view.unsqueeze(0).repeat(6, 1, 1).contiguous()


@dataclass
class _Runtime:
    logits: torch.Tensor
    optimizer: torch.optim.Adam
    protocol: object
    targets: torch.Tensor
    render_thresholded_fn: Callable[[object, torch.Tensor, float], object]
    allow_test_size: bool


class _V7RunTransaction:
    """Retain and lock the registered production root/run ancestry."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self._config = config
        self._descriptors: list[int] = []
        self._bindings: list[tuple[int, int, int, str, int | None, str | None]] = []
        self._lock_fd: int | None = None
        self._leaf_fd: int | None = None
        self._leaf_parent_fd: int | None = None
        self._leaf_name: str | None = None
        self._invalid = False

    def __enter__(self) -> "_V7RunTransaction":
        try:
            return self._enter_bound()
        except BaseException:
            self._close_resources()
            raise

    def _enter_bound(self) -> "_V7RunTransaction":
        if self._config.get("test_only") is True:
            return self
        root = Path(str(self._config["root_path"]))
        with open_v7_root_transaction(root) as transaction:
            transaction.ensure_registered_directory("runs")
            transaction.revalidate_root_marker()
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        root_fd = os.open(root, flags)
        self._descriptors.append(root_fd)
        root_stat = os.fstat(root_fd)
        self._bindings.append((root_fd, root_stat.st_dev, root_stat.st_ino, str(root), None, None))
        runs_fd = os.open("runs", flags, dir_fd=root_fd)
        self._descriptors.append(runs_fd)
        runs_stat = os.fstat(runs_fd)
        named = os.stat("runs", dir_fd=root_fd, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (runs_stat.st_dev, runs_stat.st_ino):
            raise TrainingStateError("registered runs directory changed while binding")
        self._bindings.append((runs_fd, runs_stat.st_dev, runs_stat.st_ino, "runs", root_fd, "runs"))
        self._lock_fd = os.open(
            ".pcraft-v7-trainer.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=runs_fd,
        )
        lock_stat = os.fstat(self._lock_fd)
        if lock_stat.st_nlink != 1:
            raise TrainingStateError("trainer lock hardlink is forbidden")
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        current_fd = runs_fd
        parts = PurePosixPath(str(self._config["run_path"])).parts
        if not parts or parts[0] != "runs" or any(part in ("", ".", "..") for part in parts):
            raise TrainingStateError("run relative path is not canonical")
        for part in parts[1:-1]:
            try:
                next_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                os.mkdir(part, 0o700, dir_fd=current_fd)
                next_fd = os.open(part, flags, dir_fd=current_fd)
            self._descriptors.append(next_fd)
            stat = os.fstat(next_fd)
            named = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            if (stat.st_dev, stat.st_ino) != (named.st_dev, named.st_ino):
                raise TrainingStateError("run ancestor changed while binding")
            self._bindings.append((next_fd, stat.st_dev, stat.st_ino, part, current_fd, part))
            current_fd = next_fd
        self._leaf_parent_fd = current_fd
        self._leaf_name = parts[-1]
        try:
            leaf_fd = os.open(parts[-1], flags, dir_fd=current_fd)
        except FileNotFoundError:
            leaf_fd = None
        if leaf_fd is not None:
            self._leaf_fd = leaf_fd
            self._descriptors.append(leaf_fd)
            stat = os.fstat(leaf_fd)
            self._bindings.append((leaf_fd, stat.st_dev, stat.st_ino, "run leaf", current_fd, parts[-1]))
        self.revalidate()
        return self

    def create_run_leaf(self) -> None:
        if self._leaf_fd is not None:
            raise FileExistsError("fresh run leaf must be absent")
        if self._leaf_parent_fd is None or self._leaf_name is None:
            raise TrainingStateError("run transaction is not bound")
        os.mkdir(self._leaf_name, 0o700, dir_fd=self._leaf_parent_fd)
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        leaf_fd = os.open(self._leaf_name, flags, dir_fd=self._leaf_parent_fd)
        self._leaf_fd = leaf_fd; self._descriptors.append(leaf_fd)
        stat = os.fstat(leaf_fd)
        self._bindings.append((leaf_fd, stat.st_dev, stat.st_ino, "run leaf", self._leaf_parent_fd, self._leaf_name))
        self.revalidate()

    def _directory_for(self, relative: Path, *, create_parents: bool) -> tuple[int, str, list[int]]:
        if self._leaf_fd is None or relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
            raise TrainingStateError("publication path is not a safe run-relative path")
        current = self._leaf_fd; opened: list[int] = []
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        for part in relative.parts[:-1]:
            try:
                child = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                if not create_parents: raise
                os.mkdir(part, 0o700, dir_fd=current)
                child = os.open(part, flags, dir_fd=current)
            opened.append(child); current = child
        return current, relative.parts[-1], opened

    def publish(
        self, relative: Path, payload: bytes, *, create_once: bool,
        expected_sha256: str | None = None,
    ) -> str:
        self.revalidate()
        directory_fd, name, opened = self._directory_for(relative, create_parents=True)
        temporary = f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            if not create_once:
                try:
                    existing_fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
                except FileNotFoundError:
                    existing_fd = None
                if existing_fd is None and expected_sha256 is not None:
                    raise TrainingStateError("mutable publication compare-and-swap target disappeared")
                if existing_fd is not None:
                    try:
                        stat = os.fstat(existing_fd)
                        raw = b""
                        while True:
                            chunk = os.read(existing_fd, 1024 * 1024)
                            if not chunk: break
                            raw += chunk
                    finally:
                        os.close(existing_fd)
                    if stat.st_nlink != 1 or expected_sha256 is None or _sha_bytes(raw) != expected_sha256:
                        raise TrainingStateError("mutable publication compare-and-swap mismatch")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=directory_fd)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view); view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if create_once:
                try:
                    os.link(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
                except FileExistsError:
                    existing_fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
                    try:
                        existing = b""
                        while True:
                            chunk = os.read(existing_fd, 1024 * 1024)
                            if not chunk: break
                            existing += chunk
                    finally: os.close(existing_fd)
                    if existing != payload:
                        raise TrainingStateError("create-once publication conflicts")
                finally:
                    try: os.unlink(temporary, dir_fd=directory_fd)
                    except FileNotFoundError: pass
            else:
                os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.fsync(directory_fd); self.revalidate()
            return _sha_bytes(payload)
        except BaseException:
            self._invalid = True
            try: os.unlink(temporary, dir_fd=directory_fd)
            except (FileNotFoundError, UnboundLocalError): pass
            raise
        finally:
            for descriptor in reversed(opened): os.close(descriptor)

    def digest(self, relative: Path) -> str | None:
        self.revalidate()
        directory_fd, name, opened = self._directory_for(relative, create_parents=False)
        try:
            try:
                descriptor = os.open(
                    name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                return None
            try:
                stat = os.fstat(descriptor)
                if stat.st_nlink != 1:
                    raise TrainingStateError("publication CAS target is not exclusive")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk: break
                    chunks.append(chunk)
                return _sha_bytes(b"".join(chunks))
            finally:
                os.close(descriptor)
        finally:
            for descriptor in reversed(opened): os.close(descriptor)

    def ensure_directory(self, relative: Path) -> None:
        if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
            raise TrainingStateError("directory path is not safe run-relative")
        self.revalidate()
        directory_fd, _, opened = self._directory_for(
            relative / ".directory-sentinel", create_parents=True,
        )
        try:
            os.fsync(directory_fd)
            self.revalidate()
        finally:
            for descriptor in reversed(opened): os.close(descriptor)

    def remove_empty_directory(self, relative: Path) -> None:
        self.revalidate()
        parent_fd, name, opened = self._directory_for(
            relative, create_parents=False,
        )
        try:
            os.rmdir(name, dir_fd=parent_fd)
            os.fsync(parent_fd)
            self.revalidate()
        except OSError as error:
            raise TrainingStateError("recovery directory is not safely empty") from error
        finally:
            for descriptor in reversed(opened): os.close(descriptor)

    def revalidate(self) -> None:
        for descriptor, device, inode, label, parent_fd, name in self._bindings:
            stat = os.fstat(descriptor)
            if (stat.st_dev, stat.st_ino) != (device, inode):
                self._invalid = True
                raise TrainingStateError(f"bound directory changed: {label}")
            if parent_fd is not None and name is not None:
                try:
                    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError as error:
                    self._invalid = True
                    raise TrainingStateError(f"bound directory name disappeared: {label}") from error
                if (named.st_dev, named.st_ino) != (device, inode):
                    self._invalid = True
                    raise TrainingStateError(f"bound directory name replacement: {label}")
        if self._config.get("test_only") is not True:
            root = Path(str(self._config["root_path"]))
            with open_v7_root_transaction(root) as transaction:
                transaction.revalidate_root_marker()

    def _close_resources(self) -> None:
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None
        for descriptor in reversed(self._descriptors):
            os.close(descriptor)
        self._descriptors.clear(); self._bindings.clear()
        self._leaf_fd = None; self._leaf_parent_fd = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            if not self._invalid:
                self.revalidate()
        finally:
            self._close_resources()


def _test_targets(resolution: int) -> torch.Tensor:
    target = torch.zeros((6, resolution, resolution), dtype=torch.bool)
    target[:, 1 : resolution - 1, 1 : resolution - 1] = True
    return target


def _make_test_resolved_config(
    *, run_dir: Path, profile: str = "full", total_updates: int = 4, resolution: int = 4,
) -> dict[str, object]:
    if profile not in ("base", "hit_only", "barrier_only", "full"):
        raise ValueError("test profile is invalid")
    if type(total_updates) is not int or total_updates < 2 or type(resolution) is not int or resolution < 4:
        raise ValueError("test dimensions are invalid")
    path = Path(os.path.abspath(run_dir))
    root = path.parent
    payload: dict[str, object] = {
        "schema": _CONFIG_SCHEMA,
        "root_path": str(root),
        "campaign_path": "manifests/campaign.json",
        "campaign_schema": "pcraft_soft_loss_v7_campaign_v1",
        "campaign_core_sha256": hashlib.sha256(b"test-campaign").hexdigest(),
        "cohort": "test",
        "authorization_paths": {},
        "formal_matrix_path": None,
        "formal_manifest_schema": None,
        "formal_core_sha256": None,
        "development_3000_authorization_sha256": None,
        "task_id": "test_task",
        "profile": profile,
        "flags": {"use_hit": profile in ("hit_only", "full"), "use_barrier": profile in ("barrier_only", "full")},
        "budget_id": None if profile == "base" else "budget_010",
        "budget_ratio": None if profile == "base" else 0.10,
        "seed": 17,
        "run_path": str(path),
        "inputs": [],
        "registered_target_sha256": _tensor_sha(_test_targets(resolution)),
        "hull_xyz_sha256": hashlib.sha256(b"test-hull-xyz").hexdigest(),
        "hull_dhw_sha256": hashlib.sha256(b"test-hull-dhw").hexdigest(),
        "hull_occupied_count": resolution ** 3,
        "resolution": resolution,
        "export_resolution": resolution,
        "updates": total_updates,
        "device": "cpu",
        "environment": {"schema": "test"},
        "test_only": True,
    }
    unsigned = copy.deepcopy(payload)
    payload["config_sha256"] = canonical_sha256(unsigned)
    return json.loads(canonical_json_bytes(payload))


def _validate_test_config(value: object) -> dict[str, object]:
    row = _plain(value)
    if type(row) is not dict or row.get("test_only") is not True:
        raise ValueError("production config required")
    try:
        rebuilt = _make_test_resolved_config(
            run_dir=Path(row["run_path"]), profile=row["profile"],
            total_updates=row["updates"], resolution=row["resolution"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("test config malformed") from error
    if canonical_json_bytes(row) != canonical_json_bytes(rebuilt):
        raise ValueError("test config differs from reconstruction")
    return row


def _test_stage_matrix(config: Mapping[str, object]) -> V7StageMatrix:
    row = _validate_test_config(config)
    physical = {
        "physical_index": 0, "task_id": row["task_id"], "budget_id": row["budget_id"],
        "profile": row["profile"], "config_sha256": row["config_sha256"],
        "run_path": row["run_path"],
    }
    logical = dict(physical); logical["logical_index"] = 0
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v7_stage_matrix_v1", "stage": "test-0004",
        "campaign_sha256": hashlib.sha256(b"test-campaign-file").hexdigest(),
        "endpoint": row["updates"], "authorization_path": None,
        "authorization_sha256": None, "physical_rows": [physical], "logical_rows": [logical],
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    return V7StageMatrix(
        "test-0004", int(row["updates"]), (_deep_immutable(physical),),
        (_deep_immutable(logical),), str(payload["semantic_sha256"]),
        _deep_immutable(payload), None, None,
    )


def _stage_payload(stage_matrix: object) -> dict[str, object]:
    if isinstance(stage_matrix, V7StageMatrix):
        return _plain(stage_matrix.payload)  # type: ignore[return-value]
    if isinstance(stage_matrix, Mapping):
        plain = _plain(stage_matrix)
        if type(plain) is dict:
            return plain
    raise ValueError("stage_matrix has the wrong type")


def _run_dir(config: Mapping[str, object]) -> Path:
    raw = Path(str(config["run_path"]))
    path = raw if config.get("test_only") is True else Path(str(config["root_path"])) / raw
    canonical = Path(os.path.abspath(path))
    if path != canonical or not path.is_absolute():
        raise TrainingStateError("run path is not canonical within the registered root")
    if config.get("test_only") is not True:
        try:
            raw.relative_to("runs")
        except ValueError as error:
            raise TrainingStateError("production run path is outside runs") from error
    return path


def _resolution(config: Mapping[str, object]) -> int:
    return int(config["resolution"] if config.get("test_only") is True else config["scientific_policy"]["resolution"])


def _total_updates(config: Mapping[str, object]) -> int:
    return int(config["updates"] if config.get("test_only") is True else config["scientific_policy"]["updates"])


def _device_text(config: Mapping[str, object]) -> str:
    return str(config["device"] if config.get("test_only") is True else config["environment"]["device"])


def _target_stack_sha(config: Mapping[str, object]) -> str:
    return str(config["registered_target_sha256"] if config.get("test_only") is True else config["target_stack_sha256"])


def _validate_test_matrix(config: Mapping[str, object], stage_matrix: object) -> V7StageMatrix:
    expected = _test_stage_matrix(config)
    if canonical_json_bytes(_stage_payload(stage_matrix)) != canonical_json_bytes(_plain(expected.payload)):
        raise ValueError("test stage matrix differs")
    return expected


def _authorization_for_stage(root: Path, campaign: Mapping[str, object], stage: str) -> object:
    previous = {
        "development-1000": "development-0300",
        "development-3000": "development-1000",
        "formal-3000": "development-3000",
    }.get(stage)
    if previous is None:
        return None
    filename = previous.replace("-", "_") + ".json"
    return validate_stage_authorization(
        root / "authorizations" / filename,
        expected_stage=previous,
        campaign=campaign,
    )


def _authenticate_production(
    resolved_config: Mapping[str, object], stage_matrix: object | None,
) -> tuple[dict[str, object], V7StageMatrix]:
    if not isinstance(resolved_config, Mapping) or resolved_config.get("test_only") is not None:
        raise ValueError("production resolved config has the wrong schema")
    claimed = _plain(resolved_config)
    if type(claimed) is not dict:
        raise ValueError("production config must be an exact mapping")
    root = Path(str(claimed.get("root_path")))
    campaign = validate_campaign_manifest(root / "manifests/campaign.json")
    config = _plain(rebuild_resolved_config(claimed, campaign))
    revalidate_static_identities(config)
    if stage_matrix is None:
        raise ValueError("an explicit current grant is required for mutation")
    supplied = _stage_payload(stage_matrix)
    stage = supplied.get("stage")
    if type(stage) is not str:
        raise ValueError("stage matrix stage is invalid")
    authorization = _authorization_for_stage(root, campaign, stage)
    rebuilt = resolve_stage_matrix(campaign, stage=stage, authorization=authorization)
    if canonical_json_bytes(supplied) != canonical_json_bytes(_plain(rebuilt.payload)):
        raise ValueError("stage matrix differs from authenticated grant")
    matches = [row for row in rebuilt.physical_rows if row["config_sha256"] == config["config_sha256"]]
    if len(matches) != 1 or matches[0]["run_path"] != config["run_path"]:
        raise ValueError("config is not authorized by the current physical matrix")
    return config, rebuilt


def _reauthenticate_before_publication(
    config: Mapping[str, object], matrix: V7StageMatrix, completed: int,
) -> None:
    _exact_int(completed, "publication update")
    if config.get("test_only") is True:
        return
    authenticated_config, authenticated_matrix = _authenticate_production(config, matrix)
    if (
        canonical_json_bytes(authenticated_config) != canonical_json_bytes(_plain(config))
        or canonical_json_bytes(_plain(authenticated_matrix.payload))
        != canonical_json_bytes(_plain(matrix.payload))
    ):
        raise TrainingStateError("publication identity changed")
    revalidate_static_identities(authenticated_config)
    targets, logits = _load_registered_inputs(authenticated_config)
    if _tensor_sha(targets) != _target_stack_sha(authenticated_config):
        raise TrainingStateError("registered target identity changed")
    if completed == 0 and not torch.isfinite(logits).all().item():
        raise TrainingStateError("registered hull initialization is nonfinite")
    cameras = registered_camera_payload()
    camera_config = authenticated_config.get("cameras")
    camera_identity = {
        "schema": cameras["metadata"]["schema"],
        "metadata": cameras["metadata"],
        "rotation_sha256": cameras["rotation_sha256"],
        "translation_sha256": cameras["translation_sha256"],
        "scale_sha256": cameras["scale_sha256"],
        "projection_sha256": cameras["projection_sha256"],
    }
    if camera_config is not None and canonical_json_bytes(camera_config) != canonical_json_bytes(camera_identity):
        raise TrainingStateError("registered camera identity changed")


def _grant(matrix: V7StageMatrix) -> dict[str, object]:
    payload = _plain(matrix.payload)
    return {
        "stage": matrix.stage,
        "stage_matrix_sha256": matrix.semantic_sha256,
        "authorization_path": payload["authorization_path"],
        "authorization_sha256": payload["authorization_sha256"],
        "endpoint": matrix.endpoint,
    }


def _validate_absolute_cap(config: Mapping[str, object], matrix: V7StageMatrix, cap: object) -> int:
    if config.get("test_only") is True:
        if type(cap) is not int or cap not in (2, _total_updates(config)):
            raise ValueError("test max_updates is not a registered cap")
        return cap
    if cap is None:
        return matrix.endpoint
    if type(cap) is not int:
        raise ValueError("max_updates must be an exact registered cap")
    stage = matrix.stage
    allowed: set[int]
    if stage == "development-0300":
        allowed = {2, 300}
        task, profile = config["task_id"], config["profile"]
        if task == "arcane_casters" and profile in ("base", "full"):
            allowed.add(200)
    elif stage == "development-1000":
        allowed = {1000}
        if config["task_id"] in ("blade_heroes", "stage_dancers"):
            allowed.add(2)
    elif stage == "development-3000":
        allowed = {2400, 2700, 3000}
    elif stage == "formal-3000":
        allowed = {2, 200, 2400, 2700, 3000}
    else:
        raise ValueError("unknown production stage")
    if cap not in allowed:
        raise ValueError("max_updates is not registered for this grant/cell")
    return cap


def _seed_all(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _load_registered_inputs(config: Mapping[str, object]) -> tuple[torch.Tensor, torch.Tensor]:
    masks: list[tuple[str, torch.Tensor]] = []
    snapshots = []
    for axis, item in zip(("x", "y", "z"), config["inputs"], strict=True):
        path = Path(item["resolved_path"])
        snap = snapshot_regular_file(path, exclusive=False)
        if snap.sha256 != item["raw_sha256"]:
            raise TrainingStateError("input bytes changed")
        image = cv2.imdecode(np.frombuffer(snap.payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise TrainingStateError("input cannot be decoded")
        normalized = normalize_target(image, 256, False) > 0
        mask = torch.from_numpy(normalized).bool()
        if _tensor_sha(mask) != item["normalized_sha256"]:
            raise TrainingStateError("normalized input changed")
        masks.append((axis, mask)); snapshots.append(snap)
    targets = build_registered_target_stack(tuple(masks))
    logits = build_registered_visual_hull_logits(tuple(masks))
    if _tensor_sha(targets) != _target_stack_sha(config):
        raise TrainingStateError("registered target identity changed")
    for snap in snapshots:
        revalidate_snapshot(snap)
    return targets, logits


def _build_runtime(config: Mapping[str, object]) -> _Runtime:
    if config.get("test_only") is True:
        # The explicit 4^3 seam is latency-oriented and must not fan tiny CPU
        # kernels across the host's production-sized thread pool.
        torch.set_num_threads(1)
        resolution = _resolution(config)
        logits = torch.full((1, 1, resolution, resolution, resolution), 2.0, dtype=torch.float32).requires_grad_()
        optimizer = torch.optim.Adam([logits], lr=0.007)
        return _Runtime(logits, optimizer, _TestProtocol(), _test_targets(resolution), _test_render_thresholded, True)
    device = torch.device(_device_text(config))
    if device != torch.device("cuda:0") or not torch.cuda.is_available():
        raise TrainingStateError("production requires registered cuda:0")
    targets_cpu, logits_cpu = _load_registered_inputs(config)
    logits = logits_cpu.to(device=device, dtype=torch.float32).detach().requires_grad_()
    targets = targets_cpu.to(device=device)
    protocol = RegisteredEARProtocol(device)
    return _Runtime(logits, torch.optim.Adam([logits], lr=0.007), protocol, targets, render_thresholded, False)


def _capture_rng(device: torch.device) -> dict[str, object]:
    cuda_states = tuple(state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()) if device.type == "cuda" else ()
    return {"schema": _RNG_SCHEMA, "python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": cuda_states}


def _restore_rng(row: Mapping[str, object], device: torch.device) -> None:
    random.setstate(row["python"])
    np.random.set_state(row["numpy"])
    torch.set_rng_state(row["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(list(row["torch_cuda"]))


def _optimizer_payload(optimizer: torch.optim.Adam, logits: torch.Tensor, completed: int) -> dict[str, object]:
    state = optimizer.state.get(logits, {})
    step = state.get("step", torch.tensor(float(completed)))
    if isinstance(step, torch.Tensor):
        step_value = int(step.item())
    else:
        step_value = int(step)
    zeros = torch.zeros_like(logits, device="cpu")
    exp_avg = state.get("exp_avg", zeros).detach().cpu().contiguous().clone()
    exp_avg_sq = state.get("exp_avg_sq", zeros).detach().cpu().contiguous().clone()
    group = optimizer.param_groups[0]
    return {
        "schema": _OPTIMIZER_SCHEMA, "step": step_value,
        "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq,
        "learning_rate": float(group["lr"]), "betas": tuple(float(v) for v in group["betas"]),
        "eps": float(group["eps"]), "weight_decay": float(group["weight_decay"]),
        "amsgrad": bool(group["amsgrad"]), "maximize": bool(group["maximize"]),
        "capturable": bool(group["capturable"]), "differentiable": bool(group["differentiable"]),
        "foreach": group.get("foreach"), "fused": group.get("fused"),
    }


def _load_optimizer(optimizer: torch.optim.Adam, logits: torch.Tensor, row: Mapping[str, object]) -> None:
    group = optimizer.param_groups[0]
    group["lr"] = float(row["learning_rate"])
    group["betas"] = tuple(row["betas"])
    group["eps"] = float(row["eps"]); group["weight_decay"] = float(row["weight_decay"])
    group["amsgrad"] = bool(row["amsgrad"]); group["maximize"] = bool(row["maximize"])
    group["capturable"] = bool(row["capturable"]); group["differentiable"] = bool(row["differentiable"])
    group["foreach"] = row["foreach"]; group["fused"] = row["fused"]
    if int(row["step"]) > 0:
        optimizer.state[logits] = {
            "step": torch.tensor(float(row["step"]), device=logits.device),
            "exp_avg": row["exp_avg"].to(logits.device).clone(),
            "exp_avg_sq": row["exp_avg_sq"].to(logits.device).clone(),
        }


def _audit(config_sha: str, history: Sequence[Mapping[str, object]]) -> dict[str, object]:
    hit_eligible = sum(row["component"] == "hit" for row in history)
    barrier_eligible = sum(row["component"] == "barrier" for row in history)
    payload = {
        "schema": _AUDIT_SCHEMA, "config_sha256": config_sha,
        "completed_updates": len(history), "hit_eligible_slots": hit_eligible,
        "hit_active_slots": sum(row["component"] == "hit" and row["raw_auxiliary_gradient_norm"] > 0.0 for row in history),
        "barrier_eligible_slots": barrier_eligible,
        "barrier_active_slots": sum(row["component"] == "barrier" and row["raw_auxiliary_gradient_norm"] > 0.0 for row in history),
    }
    return _semantic(payload)


def _resource(config_sha: str, history: Sequence[Mapping[str, object]], *, test_only: bool) -> dict[str, object]:
    times = [float(row["step_seconds"]) for row in history]
    finite = all(math.isfinite(value) for value in times)
    cuda = any(bool(row["cuda_executed"]) for row in history)
    peak = max((int(row["peak_allocated_bytes"]) for row in history), default=0)
    if test_only and history:
        cuda, peak, total = True, 1, 100
    else:
        total = int(torch.cuda.get_device_properties(0).total_memory) if cuda else 0
    payload = {
        "schema": _RESOURCE_SCHEMA, "config_sha256": config_sha,
        "completed_updates": len(history), "cuda_executed": cuda,
        "step_time_count": len(times), "step_times_finite": finite,
        "min_step_seconds": min(times) if times and finite else None,
        "peak_allocated_bytes": peak, "device_total_bytes": total,
    }
    return _semantic(payload)


def _checkpoint_payload(
    config: Mapping[str, object], matrix: V7StageMatrix, runtime: _Runtime,
    completed: int, predecessor_step: int | None, predecessor_sha: str | None,
    history: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    config_plain = _plain(config)
    config_sha = str(config_plain["config_sha256"])
    return {
        "schema": _CHECKPOINT_SCHEMA, "config": config_plain,
        "config_sha256": config_sha, "grant": _grant(matrix),
        "completed_updates": completed, "predecessor_step": predecessor_step,
        "predecessor_checkpoint_sha256": predecessor_sha,
        "logits": runtime.logits.detach().cpu().to(torch.float32).contiguous().clone(),
        "optimizer": _optimizer_payload(runtime.optimizer, runtime.logits, completed),
        "rng": _capture_rng(runtime.logits.device),
        "history": tuple(copy.deepcopy(list(history))),
        "audit": _audit(config_sha, history),
        "resource": _resource(config_sha, history, test_only=config.get("test_only") is True),
    }


def _torch_bytes(payload: Mapping[str, object]) -> bytes:
    buffer = io.BytesIO(); torch.save(dict(payload), buffer); return buffer.getvalue()


def _safe_existing_file(path: Path, *, exclusive: bool = True) -> bytes:
    if path.is_symlink():
        raise TrainingStateError(f"symlink is forbidden: {path}")
    stat = path.stat(follow_symlinks=False)
    if not path.is_file() or (exclusive and stat.st_nlink != 1):
        raise TrainingStateError(f"unsafe regular file: {path}")
    before = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_nlink)
    raw = path.read_bytes()
    after_stat = path.stat(follow_symlinks=False)
    after = (after_stat.st_dev, after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns, after_stat.st_nlink)
    if before != after:
        raise TrainingStateError(f"file changed while reading: {path}")
    return raw


def _atomic_publish(path: Path, payload: bytes, *, create_once: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if create_once and _safe_existing_file(path) == payload:
            return _sha_bytes(payload)
        if create_once:
            raise TrainingStateError(f"create-once file conflicts: {path}")
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    if create_once and path.exists():
        temporary.unlink()
        if _safe_existing_file(path) != payload:
            raise TrainingStateError(f"create-once file conflicts: {path}")
    else:
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    return _sha_bytes(payload)


def _json_publish(path: Path, payload: Mapping[str, object], *, create_once: bool) -> str:
    return _atomic_publish(path, canonical_json_bytes(payload), create_once=create_once)


def _publish_run_bytes(
    run: Path, relative: Path, payload: bytes, *, create_once: bool,
    transaction: _V7RunTransaction | None,
) -> str:
    if transaction is None:
        return _atomic_publish(run / relative, payload, create_once=create_once)
    expected = None if create_once else transaction.digest(relative)
    return transaction.publish(
        relative, payload, create_once=create_once, expected_sha256=expected,
    )


def _publish_run_json(
    run: Path, relative: Path, payload: Mapping[str, object], *, create_once: bool,
    transaction: _V7RunTransaction | None,
) -> str:
    return _publish_run_bytes(
        run, relative, canonical_json_bytes(payload), create_once=create_once,
        transaction=transaction,
    )


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO(); np.save(buffer, array, allow_pickle=False); return buffer.getvalue()


def _checkpoint_name(step: int) -> str:
    return f"step_{step:04d}.pt"


def _latest_payload(config_sha: str, step: int, checkpoint_sha: str, predecessor_sha: str | None) -> dict[str, object]:
    return _semantic({
        "schema": _LATEST_SCHEMA, "config_sha256": config_sha,
        "completed_updates": step, "checkpoint_path": f"checkpoints/{_checkpoint_name(step)}",
        "checkpoint_sha256": checkpoint_sha,
        "predecessor_checkpoint_sha256": predecessor_sha,
    })


def _running_status(config_sha: str, checkpoint: AuthenticatedCheckpoint) -> dict[str, object]:
    return _semantic({
        "schema": _STATUS_SCHEMA, "status": "running", "config_sha256": config_sha,
        "current_step": checkpoint.completed_updates,
        "current_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "stage": checkpoint.stage, "stage_matrix_sha256": checkpoint.stage_matrix_sha256,
        "authorization_path": checkpoint.authorization_path,
        "authorization_sha256": checkpoint.authorization_sha256,
    })


def _checkpoint_record(run: Path, config_sha: str, payload: Mapping[str, object], raw_sha: str) -> AuthenticatedCheckpoint:
    grant = payload["grant"]
    return AuthenticatedCheckpoint(
        run, config_sha, int(payload["completed_updates"]),
        run / "checkpoints" / _checkpoint_name(int(payload["completed_updates"])), raw_sha,
        payload["predecessor_checkpoint_sha256"], grant["stage"],
        grant["stage_matrix_sha256"], grant["authorization_path"],
        grant["authorization_sha256"], len(payload["history"]),
    )


def _publish_checkpoint(
    run: Path, config: Mapping[str, object], matrix: V7StageMatrix, runtime: _Runtime,
    completed: int, predecessor: AuthenticatedCheckpoint | None,
    history: Sequence[Mapping[str, object]], *, publish_pointer: bool = True,
    transaction: _V7RunTransaction | None = None,
) -> AuthenticatedCheckpoint:
    payload = _checkpoint_payload(
        config, matrix, runtime, completed,
        None if predecessor is None else predecessor.completed_updates,
        None if predecessor is None else predecessor.checkpoint_sha256,
        history,
    )
    raw = _torch_bytes(payload)
    path = run / "checkpoints" / _checkpoint_name(completed)
    digest = _publish_run_bytes(
        run, Path("checkpoints") / path.name, raw, create_once=True,
        transaction=transaction,
    )
    record = _checkpoint_record(run, str(config["config_sha256"]), payload, digest)
    if publish_pointer:
        latest = _latest_payload(str(config["config_sha256"]), completed, digest, record.predecessor_checkpoint_sha256)
        _publish_run_json(
            run, Path("checkpoints/LATEST.json"), latest, create_once=False,
            transaction=transaction,
        )
        _publish_run_json(
            run, Path("STATUS.json"), _running_status(str(config["config_sha256"]), record),
            create_once=False, transaction=transaction,
        )
    return record


def _decode_json_file(path: Path, label: str) -> tuple[dict[str, object], str]:
    raw = _safe_existing_file(path)
    row = decode_canonical_json(raw, label=label)
    if raw != canonical_json_bytes(row):
        raise TrainingStateError(f"{label} is not canonical JSON")
    return row, _sha_bytes(raw)


def _validate_semantic(row: dict[str, object], keys: set[str], schema: str, label: str) -> None:
    if set(row) != keys or row.get("schema") != schema:
        raise TrainingStateError(f"{label} exact schema mismatch")
    terminal = row.get("semantic_sha256")
    _digest(terminal, f"{label} semantic")
    unsigned = dict(row); del unsigned["semantic_sha256"]
    if canonical_sha256(unsigned) != terminal:
        raise TrainingStateError(f"{label} semantic mismatch")


def _validate_telemetry(row: object, index: int, config: Mapping[str, object]) -> dict[str, object]:
    if type(row) is not dict or tuple(row) != _TELEMETRY_FIELDS:
        raise TrainingStateError("telemetry exact schema/order mismatch")
    if row["completed_updates"] != index:
        raise TrainingStateError("telemetry update sequence mismatch")
    before = index - 1
    if row["learning_rate"] != v7_learning_rate(before) or row["multiplier"] != v7_auxiliary_multiplier(before):
        raise TrainingStateError("telemetry schedule mismatch")
    expected_slot = "hit" if before % 4 in (0, 2) else "barrier"
    expected_component = v7_auxiliary_slot(before, str(config["profile"]))
    if row["multiplier"] == 0.0: expected_component = "base"
    if row["slot"] != expected_slot or row["component"] != expected_component:
        raise TrainingStateError("telemetry slot/component mismatch")
    if type(row["auxiliary_constructed"]) is not bool or row["auxiliary_constructed"] != (expected_component != "base"):
        raise TrainingStateError("telemetry auxiliary construction mismatch")
    if (row["aux_loss"] is None) != (expected_component == "base"):
        raise TrainingStateError("telemetry auxiliary loss mismatch")
    for key in ("learning_rate", "multiplier", "base_loss", "base_gradient_norm", "raw_auxiliary_gradient_norm", "final_auxiliary_gradient_norm", "base_auxiliary_dot", "effective_ratio", "step_seconds"):
        _finite(row[key], key, nonnegative=key != "base_auxiliary_dot")
    if row["aux_loss"] is not None: _finite(row["aux_loss"], "aux loss")
    for key in ("clip_count", "optimizer_step_count", "peak_allocated_bytes"):
        _exact_int(row[key], key)
    if row["clip_count"] != 1 or row["optimizer_step_count"] != 1:
        raise TrainingStateError("one clip/step contract violated")
    if type(row["cuda_executed"]) is not bool:
        raise TrainingStateError("cuda_executed must be exact boolean")
    _digest(row["diagnostic_sha256"], "diagnostic")
    if expected_component == "base" and any(row[key] != 0.0 for key in ("raw_auxiliary_gradient_norm", "final_auxiliary_gradient_norm", "base_auxiliary_dot", "effective_ratio")):
        raise TrainingStateError("Base telemetry must have zero auxiliary scalars")
    if expected_component != "base":
        ratio = float(config["budget_ratio"])
        if row["base_auxiliary_dot"] < 0.0 or row["effective_ratio"] > ratio * row["multiplier"]:
            raise TrainingStateError("represented auxiliary gradient violates projection/cap")
    diagnostic_unsigned = {
        "completed_updates": row["completed_updates"],
        "learning_rate": row["learning_rate"], "multiplier": row["multiplier"],
        "slot": row["slot"], "component": row["component"],
        "base_loss": row["base_loss"], "aux_loss": row["aux_loss"],
        "base_gradient_norm": row["base_gradient_norm"],
        "raw_auxiliary_gradient_norm": row["raw_auxiliary_gradient_norm"],
        "final_auxiliary_gradient_norm": row["final_auxiliary_gradient_norm"],
    }
    if canonical_sha256(diagnostic_unsigned) != row["diagnostic_sha256"]:
        raise TrainingStateError("telemetry diagnostic hash mismatch")
    return row


def _validate_optimizer(row: object, logits: torch.Tensor, completed: int) -> dict[str, object]:
    keys = {"schema", "step", "exp_avg", "exp_avg_sq", "learning_rate", "betas", "eps", "weight_decay", "amsgrad", "maximize", "capturable", "differentiable", "foreach", "fused"}
    if type(row) is not dict or set(row) != keys or row.get("schema") != _OPTIMIZER_SCHEMA:
        raise TrainingStateError("optimizer exact schema mismatch")
    if row["step"] != completed or type(row["step"]) is not int:
        raise TrainingStateError("optimizer step mismatch")
    for name in ("exp_avg", "exp_avg_sq"):
        value = row[name]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32 or value.shape != logits.shape or not torch.isfinite(value).all().item():
            raise TrainingStateError("optimizer tensor mismatch")
    if (row["exp_avg_sq"] < 0).any().item():
        raise TrainingStateError("optimizer exp_avg_sq must be nonnegative")
    if completed == 0 and (
        torch.count_nonzero(row["exp_avg"]).item() != 0
        or torch.count_nonzero(row["exp_avg_sq"]).item() != 0
    ):
        raise TrainingStateError("step-zero Adam moments must be exact zero")
    if row["learning_rate"] != (0.007 if completed == 0 else v7_learning_rate(completed - 1)):
        raise TrainingStateError("optimizer learning rate mismatch")
    if row["betas"] != (0.9, 0.999) or row["eps"] != 1e-8 or row["weight_decay"] != 0.0 or row["amsgrad"] is not False or row["maximize"] is not False or row["capturable"] is not False or row["differentiable"] is not False or row["foreach"] is not None or row["fused"] is not None:
        raise TrainingStateError("optimizer frozen configuration mismatch")
    return row


def _validate_rng(row: object, *, test_only: bool) -> None:
    if type(row) is not dict or set(row) != {"schema", "python", "numpy", "torch_cpu", "torch_cuda"} or row.get("schema") != _RNG_SCHEMA:
        raise TrainingStateError("RNG exact schema mismatch")
    if type(row["python"]) is not tuple or type(row["numpy"]) is not tuple:
        raise TrainingStateError("Python/NumPy RNG type mismatch")
    if not isinstance(row["torch_cpu"], torch.Tensor) or row["torch_cpu"].dtype != torch.uint8 or row["torch_cpu"].device.type != "cpu":
        raise TrainingStateError("Torch CPU RNG mismatch")
    if type(row["torch_cuda"]) is not tuple:
        raise TrainingStateError("Torch CUDA RNG must be tuple")
    expected = 0 if test_only else torch.cuda.device_count()
    if len(row["torch_cuda"]) != expected or any(not isinstance(v, torch.Tensor) or v.dtype != torch.uint8 or v.device.type != "cpu" for v in row["torch_cuda"]):
        raise TrainingStateError("Torch CUDA RNG structure mismatch")
    try:
        probe_python = random.Random(); probe_python.setstate(row["python"])
        probe_numpy = np.random.RandomState(); probe_numpy.set_state(row["numpy"])
        probe_torch = torch.Generator(device="cpu"); probe_torch.set_state(row["torch_cpu"])
    except (TypeError, ValueError, RuntimeError, IndexError) as error:
        raise TrainingStateError("Python/NumPy/Torch RNG semantics are invalid") from error
    if any(value.numel() == 0 for value in row["torch_cuda"]):
        raise TrainingStateError("Torch CUDA RNG state cannot be empty")


def _rng_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    left_numpy, right_numpy = left["numpy"], right["numpy"]
    numpy_equal = (
        type(left_numpy) is tuple and type(right_numpy) is tuple
        and len(left_numpy) == len(right_numpy)
        and all(
            np.array_equal(a, b) if isinstance(a, np.ndarray) or isinstance(b, np.ndarray)
            else a == b
            for a, b in zip(left_numpy, right_numpy)
        )
    )
    return bool(
        left["python"] == right["python"]
        and numpy_equal
        and torch.equal(left["torch_cpu"], right["torch_cpu"])
        and type(left["torch_cuda"]) is tuple
        and type(right["torch_cuda"]) is tuple
        and len(left["torch_cuda"]) == len(right["torch_cuda"])
        and all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"]))
    )


def _seed17_rng(*, test_only: bool) -> dict[str, object]:
    cuda_states: tuple[torch.Tensor, ...] = ()
    if not test_only:
        cuda_states = tuple(
            torch.Generator(device=f"cuda:{index}").manual_seed(17).get_state().cpu()
            for index in range(torch.cuda.device_count())
        )
    return {
        "schema": _RNG_SCHEMA,
        "python": random.Random(17).getstate(),
        "numpy": np.random.RandomState(17).get_state(),
        "torch_cpu": torch.Generator(device="cpu").manual_seed(17).get_state(),
        "torch_cuda": cuda_states,
    }


def _matrix_for_checkpoint_grant(
    config: Mapping[str, object], current: V7StageMatrix, grant: object,
) -> V7StageMatrix:
    if type(grant) is not dict:
        raise TrainingStateError("checkpoint grant must be an exact object")
    if grant == _grant(current):
        return current
    if config.get("test_only") is True:
        raise TrainingStateError("test checkpoint has a foreign grant")
    stage = grant.get("stage")
    if type(stage) is not str:
        raise TrainingStateError("checkpoint historical stage is invalid")
    root = Path(str(config["root_path"]))
    campaign = validate_campaign_manifest(root / "manifests/campaign.json")
    authorization = _authorization_for_stage(root, campaign, stage)
    historical = resolve_stage_matrix(campaign, stage=stage, authorization=authorization)
    if grant != _grant(historical):
        raise TrainingStateError("checkpoint historical grant is not published/authenticated")
    matches = [row for row in historical.physical_rows if row["config_sha256"] == config["config_sha256"]]
    if len(matches) != 1:
        raise TrainingStateError("checkpoint config was not authorized in its historical grant")
    return historical


def _matrix_for_checkpoint_record(
    config: Mapping[str, object], current: V7StageMatrix,
    checkpoint: AuthenticatedCheckpoint,
) -> V7StageMatrix:
    if config.get("test_only") is True and (
        checkpoint.stage == current.stage
        and checkpoint.stage_matrix_sha256 == current.semantic_sha256
        and checkpoint.authorization_path == _grant(current)["authorization_path"]
        and checkpoint.authorization_sha256 == _grant(current)["authorization_sha256"]
    ):
        return current
    endpoint = _total_updates(config) if config.get("test_only") is True else {
        "development-0300": 300, "development-1000": 1000,
        "development-3000": 3000, "formal-3000": 3000,
    }[checkpoint.stage]
    return _matrix_for_checkpoint_grant(config, current, {
        "stage": checkpoint.stage,
        "stage_matrix_sha256": checkpoint.stage_matrix_sha256,
        "authorization_path": checkpoint.authorization_path,
        "authorization_sha256": checkpoint.authorization_sha256,
        "endpoint": endpoint,
    })


def _load_checkpoint_path(
    path: Path, config: Mapping[str, object], matrix: V7StageMatrix, *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], AuthenticatedCheckpoint]:
    raw = _safe_existing_file(path)
    digest = _sha_bytes(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise TrainingStateError("checkpoint raw digest differs from authenticated chain")
    safe_numpy_globals = [
        np._core.multiarray._reconstruct, np.ndarray, np.dtype,
        type(np.dtype(np.uint32)),
    ]
    try:
        with torch.serialization.safe_globals(safe_numpy_globals):
            payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error: raise TrainingStateError("checkpoint cannot be decoded") from error
    keys = {"schema", "config", "config_sha256", "grant", "completed_updates", "predecessor_step", "predecessor_checkpoint_sha256", "logits", "optimizer", "rng", "history", "audit", "resource"}
    if type(payload) is not dict or set(payload) != keys or payload.get("schema") != _CHECKPOINT_SCHEMA:
        raise TrainingStateError("checkpoint exact schema mismatch")
    if canonical_json_bytes(payload["config"]) != canonical_json_bytes(_plain(config)) or payload["config_sha256"] != config["config_sha256"]:
        raise TrainingStateError("checkpoint config identity mismatch")
    checkpoint_matrix = _matrix_for_checkpoint_grant(config, matrix, payload["grant"])
    completed = _exact_int(payload["completed_updates"], "completed updates")
    if path.name != _checkpoint_name(completed):
        raise TrainingStateError("checkpoint filename/update mismatch")
    logits = payload["logits"]
    resolution = _resolution(config)
    if not isinstance(logits, torch.Tensor) or logits.device.type != "cpu" or logits.dtype != torch.float32 or logits.shape != (1, 1, resolution, resolution, resolution) or not torch.isfinite(logits).all().item():
        raise TrainingStateError("checkpoint logits mismatch")
    if completed == 0:
        if config.get("test_only") is True:
            expected_logits = torch.full_like(logits, 2.0)
        else:
            _, expected_logits = _load_registered_inputs(config)
            expected_logits = expected_logits.cpu()
        if not torch.equal(logits, expected_logits):
            raise TrainingStateError("step-zero logits differ from registered visual hull initialization")
    history = payload["history"]
    if type(history) is not tuple or len(history) != completed:
        raise TrainingStateError("checkpoint history length mismatch")
    for index, telemetry in enumerate(history, 1): _validate_telemetry(telemetry, index, config)
    _validate_optimizer(payload["optimizer"], logits, completed)
    _validate_rng(payload["rng"], test_only=config.get("test_only") is True)
    if completed == 0 and not _rng_equal(
        payload["rng"], _seed17_rng(test_only=config.get("test_only") is True)
    ):
        raise TrainingStateError("genesis RNG differs from deterministic seed 17")
    if payload["audit"] != _audit(str(config["config_sha256"]), history) or payload["resource"] != _resource(str(config["config_sha256"]), history, test_only=config.get("test_only") is True):
        raise TrainingStateError("checkpoint audit/resource mismatch")
    return payload, _checkpoint_record(path.parent.parent, str(config["config_sha256"]), payload, digest)


def _load_chain(
    run: Path, config: Mapping[str, object], matrix: V7StageMatrix, *,
    allow_orphans: bool = False,
) -> tuple[dict[str, object], AuthenticatedCheckpoint, tuple[AuthenticatedCheckpoint, ...]]:
    latest, _ = _decode_json_file(run / "checkpoints/LATEST.json", "LATEST")
    _validate_semantic(latest, {"schema", "config_sha256", "completed_updates", "checkpoint_path", "checkpoint_sha256", "predecessor_checkpoint_sha256", "semantic_sha256"}, _LATEST_SCHEMA, "LATEST")
    if latest["config_sha256"] != config["config_sha256"]:
        raise TrainingStateError("LATEST config mismatch")
    current_step = _exact_int(latest["completed_updates"], "LATEST update")
    all_names = sorted(path.name for path in (run / "checkpoints").iterdir() if path.name != "LATEST.json")
    expected_names = []
    for name in all_names:
        match = _CHECKPOINT_RE.fullmatch(name)
        if match is None:
            if allow_orphans:
                continue
            raise TrainingStateError("foreign checkpoint namespace entry")
        if int(match.group(1)) <= current_step:
            expected_names.append(name)
        elif not allow_orphans:
            raise TrainingStateError("unadopted checkpoint orphan")
    expected_steps = {
        step for step in required_checkpoint_steps(_total_updates(config))
        if step <= current_step
    }
    expected_steps.add(current_step)
    observed_steps = {int(_CHECKPOINT_RE.fullmatch(name).group(1)) for name in expected_names}
    if observed_steps != expected_steps:
        raise TrainingStateError("durable checkpoint schedule has a missing/extra generation")
    reverse_records: list[AuthenticatedCheckpoint] = []
    expected_digest: str | None = _digest(latest["checkpoint_sha256"], "LATEST checkpoint")
    ordered_steps = sorted(expected_steps)
    successor_payload: dict[str, object] | None = None
    for position in range(len(ordered_steps) - 1, -1, -1):
        step = ordered_steps[position]
        checkpoint_path = run / "checkpoints" / _checkpoint_name(step)
        payload, record = _load_checkpoint_path(
            checkpoint_path, config, matrix, expected_sha256=expected_digest
        )
        required_predecessor_step = ordered_steps[position - 1] if position > 0 else None
        if payload["predecessor_step"] != required_predecessor_step:
            raise TrainingStateError("checkpoint predecessor step mismatch")
        expected_digest = payload["predecessor_checkpoint_sha256"]
        if position == 0 and expected_digest is not None:
            raise TrainingStateError("step-zero checkpoint has a predecessor digest")
        if successor_payload is not None:
            prefix = successor_payload["history"][: payload["completed_updates"]]
            if canonical_json_bytes(prefix) != canonical_json_bytes(payload["history"]):
                raise TrainingStateError("checkpoint history is not an exact prefix extension")
            if not _rng_equal(payload["rng"], successor_payload["rng"]):
                raise TrainingStateError("successor RNG differs from predecessor RNG")
        successor_payload = payload
        reverse_records.append(record)
    records = list(reversed(reverse_records))
    if successor_payload is None:
        raise TrainingStateError("checkpoint chain is empty")
    del successor_payload, payload
    if not records or records[-1].completed_updates != current_step:
        raise TrainingStateError("LATEST is not the terminal durable generation")
    current = records[-1]
    payload, reloaded = _load_checkpoint_path(
        current.checkpoint_path, config, matrix,
        expected_sha256=current.checkpoint_sha256,
    )
    if reloaded != current:
        raise TrainingStateError("terminal checkpoint changed during chain validation")
    if current.stage != matrix.stage or current.stage_matrix_sha256 != matrix.semantic_sha256:
        transition = {
            "development-1000": ("development-0300", 300),
            "development-3000": ("development-1000", 1000),
        }.get(matrix.stage)
        if transition != (current.stage, current.completed_updates):
            raise TrainingStateError("LATEST does not bind the current grant or its exact predecessor endpoint")
    if latest["checkpoint_path"] != f"checkpoints/{current.checkpoint_path.name}" or latest["checkpoint_sha256"] != current.checkpoint_sha256 or latest["predecessor_checkpoint_sha256"] != current.predecessor_checkpoint_sha256:
        raise TrainingStateError("LATEST checkpoint identity mismatch")
    return payload, current, tuple(records)


def _validate_run_namespace(run: Path, status: str) -> None:
    if run.is_symlink() or not run.is_dir(): raise TrainingStateError("run must be ordinary directory")
    expected = {"CONFIG.json", "STATUS.json", "checkpoints", "artifacts"}
    if status == "failed": expected.add("failure")
    if {path.name for path in run.iterdir()} != expected:
        raise TrainingStateError("run namespace has extra/missing entries")
    for path in run.iterdir():
        stat = path.stat(follow_symlinks=False)
        if path.is_symlink() or (path.is_file() and stat.st_nlink != 1):
            raise TrainingStateError("run namespace symlink/hardlink forbidden")
    if not (run / "checkpoints").is_dir() or not (run / "artifacts").is_dir():
        raise TrainingStateError("run directories missing")


def _validate_artifact_tree(run: Path, status: str) -> None:
    artifacts = run / "artifacts"
    if not artifacts.is_dir() or artifacts.is_symlink():
        raise TrainingStateError("artifact tree root is invalid")
    base = {"metrics"}
    if status == "completed":
        base.update({
            "final_logits.npy", "final_density.npy", "final_hard_005.npy",
            "audit.json", "resource.json", "manifest.json",
        })
    if {path.name for path in artifacts.iterdir()} != base:
        raise TrainingStateError("artifact tree namespace mismatch")
    metrics = artifacts / "metrics"
    if metrics.is_symlink() or not metrics.is_dir():
        raise TrainingStateError("artifact metric namespace is invalid")
    if any(path.is_dir() or path.is_symlink() for path in metrics.iterdir()):
        raise TrainingStateError("artifact metric tree contains a foreign entry")


def _matrix_for_validator(config: Mapping[str, object], stage_matrix: object | None, run: Path) -> V7StageMatrix:
    if config.get("test_only") is True:
        if stage_matrix is None: raise ValueError("test validator requires stage matrix")
        return _validate_test_matrix(config, stage_matrix)
    if stage_matrix is not None:
        return _authenticate_production(config, stage_matrix)[1]
    root = Path(str(config["root_path"])); campaign = validate_campaign_manifest(root / "manifests/campaign.json")
    status, _ = _decode_json_file(run / "STATUS.json", "STATUS")
    state = status.get("status")
    claimed_stage: object = None
    claimed_sha: object = None
    if state == "running":
        claimed_stage = status.get("stage")
        claimed_sha = status.get("stage_matrix_sha256")
    elif state == "failed":
        observation, _ = _decode_json_file(run / "failure/observation.json", "failure observation")
        claimed_sha = observation.get("stage_matrix_sha256")
    elif state == "completed":
        manifest, _ = _decode_json_file(run / "artifacts/manifest.json", "artifact manifest")
        history = manifest.get("grant_history")
        if type(history) is list and history and type(history[-1]) is dict:
            claimed_stage = history[-1].get("stage")
            claimed_sha = history[-1].get("stage_matrix_sha256")
    else:
        raise TrainingStateError("STATUS state is invalid before checkpoint decoding")
    stages = (
        (str(claimed_stage),) if type(claimed_stage) is str else
        ("development-0300", "development-1000", "development-3000", "formal-3000")
    )
    matches: list[V7StageMatrix] = []
    for stage in stages:
        try:
            authorization = _authorization_for_stage(root, campaign, stage)
            candidate = resolve_stage_matrix(campaign, stage=stage, authorization=authorization)
        except (ValueError, FileNotFoundError):
            continue
        if candidate.semantic_sha256 == claimed_sha:
            matches.append(candidate)
    if len(matches) != 1:
        raise TrainingStateError("durable state does not identify one authenticated historical grant")
    matrix = matches[0]
    _authenticate_production(config, matrix)
    return matrix


def _config_and_matrix(resolved_config: Mapping[str, object], stage_matrix: object, *, production: bool) -> tuple[dict[str, object], V7StageMatrix]:
    if production:
        return _authenticate_production(resolved_config, stage_matrix)
    config = _validate_test_config(resolved_config)
    return config, _validate_test_matrix(config, stage_matrix)


def _initialize_run(
    config: Mapping[str, object], matrix: V7StageMatrix, *,
    transaction: _V7RunTransaction | None = None,
) -> tuple[_Runtime, AuthenticatedCheckpoint, list[dict[str, object]]]:
    run = _run_dir(config)
    if transaction is None:
        if run.exists(): raise FileExistsError("fresh run path must be absent")
        run.parent.mkdir(parents=True, exist_ok=True)
        run.mkdir(); (run / "checkpoints").mkdir(); (run / "artifacts").mkdir(); (run / "artifacts/metrics").mkdir()
    else:
        transaction.create_run_leaf()
        transaction.ensure_directory(Path("checkpoints"))
        transaction.ensure_directory(Path("artifacts/metrics"))
    _publish_run_json(
        run, Path("CONFIG.json"), _plain(config), create_once=True,
        transaction=transaction,
    )
    _seed_all(17, torch.device(_device_text(config)))
    runtime = _build_runtime(config)
    history: list[dict[str, object]] = []
    _reauthenticate_before_publication(config, matrix, 0)
    checkpoint = _publish_checkpoint(
        run, config, matrix, runtime, 0, None, history,
        transaction=transaction,
    )
    return runtime, checkpoint, history


def _restore_runtime(config: Mapping[str, object], payload: Mapping[str, object]) -> _Runtime:
    runtime = _build_runtime(config)
    with torch.no_grad(): runtime.logits.copy_(payload["logits"].to(runtime.logits.device, dtype=runtime.logits.dtype))
    _load_optimizer(runtime.optimizer, runtime.logits, payload["optimizer"])
    _restore_rng(payload["rng"], runtime.logits.device)
    return runtime


def _metric_payload(config: Mapping[str, object], matrix: V7StageMatrix, checkpoint: AuthenticatedCheckpoint, runtime: _Runtime) -> dict[str, object]:
    with torch.no_grad():
        density = torch.sigmoid(runtime.logits)
        projection = runtime.render_thresholded_fn(runtime.protocol, density, 0.05).projection
        metrics = evaluate_six_view_metrics(projection.float(), runtime.targets)
    cameras = registered_camera_payload()
    unsigned: dict[str, object] = {
        "schema": _METRICS_SCHEMA, "config_sha256": config["config_sha256"],
        "stage": matrix.stage, "stage_matrix_sha256": matrix.semantic_sha256,
        "authorization_sha256": _grant(matrix)["authorization_sha256"],
        "completed_updates": checkpoint.completed_updates, "threshold": 0.05,
        "checkpoint_path": f"checkpoints/{checkpoint.checkpoint_path.name}",
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "target_stack_sha256": _target_stack_sha(config),
        "camera_R_sha256": cameras["rotation_sha256"],
        "camera_T_sha256": cameras["translation_sha256"],
        "camera_scale_sha256": cameras["scale_sha256"],
        "camera_K_sha256": cameras["projection_sha256"],
        "per_view": _plain(metrics["per_view"]), "aggregate": _plain(metrics["aggregate"]),
    }
    return _semantic(unsigned)


def _metric_payload_from_checkpoint(
    config: Mapping[str, object], matrix: V7StageMatrix,
    checkpoint: AuthenticatedCheckpoint,
) -> dict[str, object]:
    metrics = _recompute_endpoint_metrics(config, checkpoint, matrix)
    cameras = registered_camera_payload()
    return _semantic({
        "schema": _METRICS_SCHEMA, "config_sha256": config["config_sha256"],
        "stage": matrix.stage, "stage_matrix_sha256": matrix.semantic_sha256,
        "authorization_sha256": _grant(matrix)["authorization_sha256"],
        "completed_updates": checkpoint.completed_updates, "threshold": 0.05,
        "checkpoint_path": f"checkpoints/{checkpoint.checkpoint_path.name}",
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "target_stack_sha256": _target_stack_sha(config),
        "camera_R_sha256": cameras["rotation_sha256"],
        "camera_T_sha256": cameras["translation_sha256"],
        "camera_scale_sha256": cameras["scale_sha256"],
        "camera_K_sha256": cameras["projection_sha256"],
        "per_view": _plain(metrics["per_view"]),
        "aggregate": _plain(metrics["aggregate"]),
    })


def _endpoint_steps(records: Sequence[AuthenticatedCheckpoint], config: Mapping[str, object]) -> tuple[int, ...]:
    if config.get("test_only") is True:
        result: list[int] = []
        for row in records:
            match = re.fullmatch(r"test-([0-9]{4})", row.stage)
            endpoint = (
                _total_updates(config) if row.stage == "test-0004" or match is None
                else int(match.group(1))
            )
            if row.completed_updates == endpoint and endpoint not in result:
                result.append(endpoint)
        return tuple(result)
    return tuple(step for step in (300, 1000, 3000) if any(row.completed_updates == step for row in records))


def _publish_endpoint_metric(
    run: Path, payload: Mapping[str, object], *,
    transaction: _V7RunTransaction | None = None,
) -> str:
    step = int(payload["completed_updates"])
    return _publish_run_json(
        run, Path("artifacts/metrics") / f"step_{step:04d}.json", payload,
        create_once=True, transaction=transaction,
    )


def _recompute_endpoint_metrics(
    config: Mapping[str, object], checkpoint: AuthenticatedCheckpoint,
    matrix: V7StageMatrix,
) -> Mapping[str, object]:
    payload, _ = _load_checkpoint_path(
        checkpoint.checkpoint_path, config, matrix,
        expected_sha256=checkpoint.checkpoint_sha256,
    )
    if config.get("test_only") is True:
        targets = _test_targets(_resolution(config))
        protocol: object = _TestProtocol()
        projection = _test_render_thresholded(
            protocol, torch.sigmoid(payload["logits"]), 0.05
        ).projection
    else:
        targets_cpu, _ = _load_registered_inputs(config)
        device = torch.device(_device_text(config))
        protocol = RegisteredEARProtocol(device)
        density = torch.sigmoid(payload["logits"].to(device=device, dtype=torch.float32))
        projection = render_thresholded(protocol, density, 0.05).projection
        targets = targets_cpu.to(device)
    return evaluate_six_view_metrics(projection.float(), targets)


def _grant_history(records: Sequence[AuthenticatedCheckpoint], config: Mapping[str, object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for record in records:
        endpoint = _total_updates(config) if config.get("test_only") is True else {
            "development-0300": 300, "development-1000": 1000,
            "development-3000": 3000, "formal-3000": 3000,
        }[record.stage]
        row = {
            "stage": record.stage, "stage_matrix_sha256": record.stage_matrix_sha256,
            "authorization_path": record.authorization_path,
            "authorization_sha256": record.authorization_sha256, "endpoint": endpoint,
        }
        if not result or result[-1] != row:
            result.append(row)
    return result


def _validate_endpoint_file(path: Path, config: Mapping[str, object], matrix: V7StageMatrix, checkpoint: AuthenticatedCheckpoint) -> AuthenticatedEndpointMetrics:
    row, digest = _decode_json_file(path, "endpoint metrics")
    keys = {"schema", "config_sha256", "stage", "stage_matrix_sha256", "authorization_sha256", "completed_updates", "threshold", "checkpoint_path", "checkpoint_sha256", "target_stack_sha256", "camera_R_sha256", "camera_T_sha256", "camera_scale_sha256", "camera_K_sha256", "per_view", "aggregate", "semantic_sha256"}
    _validate_semantic(row, keys, _METRICS_SCHEMA, "endpoint metrics")
    if row["config_sha256"] != config["config_sha256"] or row["stage"] != matrix.stage or row["stage_matrix_sha256"] != matrix.semantic_sha256 or row["authorization_sha256"] != _grant(matrix)["authorization_sha256"] or row["completed_updates"] != checkpoint.completed_updates or row["checkpoint_path"] != f"checkpoints/{checkpoint.checkpoint_path.name}" or row["checkpoint_sha256"] != checkpoint.checkpoint_sha256 or type(row["threshold"]) is not float or row["threshold"] != 0.05:
        raise TrainingStateError("endpoint metric identity mismatch")
    cameras = registered_camera_payload()
    if row["target_stack_sha256"] != _target_stack_sha(config) or row["camera_R_sha256"] != cameras["rotation_sha256"] or row["camera_T_sha256"] != cameras["translation_sha256"] or row["camera_scale_sha256"] != cameras["scale_sha256"] or row["camera_K_sha256"] != cameras["projection_sha256"]:
        raise TrainingStateError("endpoint metric physical identity mismatch")
    if type(row["per_view"]) is not list or len(row["per_view"]) != 6 or type(row["aggregate"]) is not dict:
        raise TrainingStateError("endpoint metric rows malformed")
    recomputed = _recompute_endpoint_metrics(config, checkpoint, matrix)
    if canonical_json_bytes({"per_view": row["per_view"], "aggregate": row["aggregate"]}) != canonical_json_bytes(_plain(recomputed)):
        raise TrainingStateError("endpoint metrics differ from independent renderer recomputation")
    return AuthenticatedEndpointMetrics(path, digest, checkpoint.completed_updates, checkpoint.checkpoint_sha256, matrix.semantic_sha256, tuple(_deep_immutable(v) for v in row["per_view"]), _deep_immutable(row["aggregate"]))


def _collect_endpoint_metrics(run: Path, config: Mapping[str, object], matrix: V7StageMatrix, records: Sequence[AuthenticatedCheckpoint]) -> tuple[AuthenticatedEndpointMetrics, ...]:
    metric_dir = run / "artifacts/metrics"
    expected_steps = _endpoint_steps(records, config)
    expected_names = {f"step_{step:04d}.json" for step in expected_steps}
    if not metric_dir.is_dir() or {path.name for path in metric_dir.iterdir()} != expected_names:
        raise TrainingStateError("endpoint metric namespace mismatch")
    lookup = {row.completed_updates: row for row in records}
    result = []
    for step in expected_steps:
        checkpoint = lookup[step]
        historical = _matrix_for_checkpoint_record(config, matrix, checkpoint)
        result.append(_validate_endpoint_file(metric_dir / f"step_{step:04d}.json", config, historical, checkpoint))
    return tuple(result)


def _finalize(
    run: Path, config: Mapping[str, object], matrix: V7StageMatrix,
    runtime: _Runtime, current: AuthenticatedCheckpoint,
    records: Sequence[AuthenticatedCheckpoint], *,
    transaction: _V7RunTransaction | None = None,
) -> CompletedRun:
    _reauthenticate_before_publication(
        config, matrix, current.completed_updates,
    )
    metric_payload = _metric_payload(config, matrix, current, runtime)
    _publish_endpoint_metric(run, metric_payload, transaction=transaction)
    checkpoint_payload, _ = _load_checkpoint_path(
        current.checkpoint_path, config, matrix,
        expected_sha256=current.checkpoint_sha256,
    )
    logits = checkpoint_payload["logits"][0, 0].numpy().astype(np.float32, copy=False)
    registered_device = torch.device(_device_text(config))
    density = (
        torch.sigmoid(checkpoint_payload["logits"].to(
            device=registered_device, dtype=torch.float32,
        ))[0, 0].cpu().numpy().astype(np.float32, copy=False)
    )
    hard = (density >= 0.05)
    artifact_payloads: dict[str, bytes] = {
        "artifacts/final_logits.npy": _npy_bytes(logits),
        "artifacts/final_density.npy": _npy_bytes(density),
        "artifacts/final_hard_005.npy": _npy_bytes(hard),
        "artifacts/audit.json": canonical_json_bytes(checkpoint_payload["audit"]),
        "artifacts/resource.json": canonical_json_bytes(checkpoint_payload["resource"]),
    }
    for relative, raw in artifact_payloads.items():
        _publish_run_bytes(
            run, Path(relative), raw, create_once=True, transaction=transaction,
        )
    all_artifacts = []
    for path in sorted((run / "artifacts").rglob("*"), key=lambda p: p.relative_to(run).as_posix().encode()):
        if path.is_file() and path.name != "manifest.json":
            relative = path.relative_to(run).as_posix(); raw = _safe_existing_file(path)
            all_artifacts.append({"path": relative, "size": len(raw), "sha256": _sha_bytes(raw)})
    manifest = _semantic({
        "schema": _MANIFEST_SCHEMA, "config_sha256": config["config_sha256"],
        "terminal_checkpoint_sha256": current.checkpoint_sha256,
        "grant_history": _grant_history(records, config), "artifacts": all_artifacts,
    })
    manifest_path = run / "artifacts/manifest.json"
    manifest_sha = _publish_run_json(
        run, Path("artifacts/manifest.json"), manifest, create_once=True,
        transaction=transaction,
    )
    status = _semantic({
        "schema": _STATUS_SCHEMA, "status": "completed", "config_sha256": config["config_sha256"],
        "current_step": current.completed_updates, "current_checkpoint_sha256": current.checkpoint_sha256,
        "artifact_manifest_path": "artifacts/manifest.json", "artifact_manifest_sha256": manifest_sha,
    })
    _reauthenticate_before_publication(
        config, matrix, current.completed_updates,
    )
    _publish_run_json(
        run, Path("STATUS.json"), status, create_once=False,
        transaction=transaction,
    )
    return validate_completed_run(config, stage_matrix=matrix)


def _nonfinite_phase_flags(error: TrainingStateError) -> tuple[bool, bool, bool] | None:
    if isinstance(error, _NonfiniteTrainingError):
        return error.finite_flags
    message = str(error)
    if message in ("base loss must be finite", "auxiliary loss must be finite"):
        return False, True, True
    if message in (
        "base gradient must be finite", "auxiliary gradient must be finite",
        "final gradient must be finite", "composed gradient must be finite",
        "clipped gradient must be finite",
        "represented base auxiliary dot must be finite and nonnegative",
        "gradient norm must be finite",
    ):
        return True, False, True
    if message == "parameters must remain finite":
        return True, True, False
    return None


def _train_impl(
    config: dict[str, object], matrix: V7StageMatrix, *, resume: bool, cap: int,
    transaction: _V7RunTransaction | None = None,
) -> ResumableRun | CompletedRun | FailedRun:
    run = _run_dir(config); total = _total_updates(config)
    if resume:
        recovered = _recover_interrupted_checkpoint_publication(
            config, matrix, transaction=transaction,
        )
        if isinstance(recovered, CompletedRun): return recovered
        if isinstance(recovered, FailedRun):
            raise TrainingStateError("failed run is immutable and cannot resume")
        state = validate_resumable_run(config, stage_matrix=matrix)
        payload, current, records = _load_chain(run, config, matrix)
        runtime = _restore_runtime(config, payload)
        history = copy.deepcopy(list(payload["history"]))
    else:
        runtime, current, history = _initialize_run(
            config, matrix, transaction=transaction,
        )
        records = (current,)
    completed = current.completed_updates
    if cap < completed:
        raise ValueError("max_updates is behind durable progress")
    if cap == completed:
        if cap == total:
            return _finalize(
                run, config, matrix, runtime, current, records,
                transaction=transaction,
            )
        return validate_resumable_run(config, stage_matrix=matrix, expected_completed_updates=cap)
    timer = _DeterministicTestTimer() if config.get("test_only") is True else time.perf_counter
    while completed < cap:
        before = completed
        if config.get("test_only") is not True:
            _authenticate_production(config, matrix)
        try:
            row = _train_one_update(
                logits=runtime.logits, optimizer=runtime.optimizer, protocol=runtime.protocol,
                targets=runtime.targets, profile=str(config["profile"]),
                budget_ratio=config["budget_ratio"], completed_updates=before,
                render_thresholded_fn=runtime.render_thresholded_fn,
                allow_test_size=runtime.allow_test_size, timer=timer,
            )
        except TrainingStateError as error:
            finite_flags = _nonfinite_phase_flags(error)
            if finite_flags is None:
                raise
            durable_payload, durable_record = _load_checkpoint_path(
                current.checkpoint_path, config, matrix,
                expected_sha256=current.checkpoint_sha256,
            )
            observation = _observation_from_checkpoint(
                config, matrix, durable_record, durable_payload
            )
            observation = dict(observation)
            (
                observation["loss_finite"], observation["gradient_finite"],
                observation["parameters_finite"],
            ) = finite_flags
            observation["semantic_sha256"] = canonical_sha256({
                key: value for key, value in observation.items()
                if key != "semantic_sha256"
            })
            return _publish_failed_from_observation(
                run, config, matrix, durable_record, observation,
                transaction=transaction,
            )
        history.append(asdict(row))
        completed = row.completed_updates
        if completed in required_checkpoint_steps(total) or completed == cap:
            _reauthenticate_before_publication(config, matrix, completed)
            current = _publish_checkpoint(
                run, config, matrix, runtime, completed, current, history,
                transaction=transaction,
            )
            records = (*records, current)
        if config.get("test_only") is not True:
            revalidate_static_identities(config)
            _authenticate_production(config, matrix)
    if cap == matrix.endpoint:
        endpoint_payload, endpoint_record = _load_checkpoint_path(
            current.checkpoint_path, config, matrix,
            expected_sha256=current.checkpoint_sha256,
        )
        observation = _observation_from_checkpoint(
            config, matrix, endpoint_record, endpoint_payload
        )
        derivation = dict(observation); derivation["_profile"] = config["profile"]
        if _failure_code(derivation, current.completed_updates) != "NONE":
            return _publish_failed_from_observation(
                run, config, matrix, current, observation,
                transaction=transaction,
            )
        _reauthenticate_before_publication(
            config, matrix, current.completed_updates,
        )
        metric = _metric_payload(config, matrix, current, runtime)
        _publish_endpoint_metric(run, metric, transaction=transaction)
    if cap == total:
        return _finalize(
            run, config, matrix, runtime, current, records,
            transaction=transaction,
        )
    return validate_resumable_run(config, stage_matrix=matrix, expected_completed_updates=cap)


def train_cell(
    resolved_config: Mapping[str, object], *, stage_matrix: object,
    resume: bool = False, max_updates: int | None = None,
) -> ResumableRun | CompletedRun | FailedRun:
    if type(resume) is not bool: raise ValueError("resume must be exact boolean")
    config, matrix = _authenticate_production(resolved_config, stage_matrix)
    cap = _validate_absolute_cap(config, matrix, max_updates)
    with _V7RunTransaction(config) as transaction:
        result = _train_impl(
            config, matrix, resume=resume, cap=cap, transaction=transaction,
        )
        transaction.revalidate()
        return result


def _train_cell_for_test(
    resolved_config: Mapping[str, object], *, stage_matrix: object | None = None,
    resume: bool = False, max_updates: int | None = None,
) -> ResumableRun | CompletedRun | FailedRun:
    if type(resume) is not bool: raise ValueError("resume must be exact boolean")
    config = _validate_test_config(resolved_config)
    matrix = _test_stage_matrix(config) if stage_matrix is None else _validate_test_matrix(config, stage_matrix)
    cap = _total_updates(config) if max_updates is None else _validate_absolute_cap(config, matrix, max_updates)
    return _train_impl(config, matrix, resume=resume, cap=cap)


def _validate_common(resolved_config: Mapping[str, object], stage_matrix: object | None) -> tuple[dict[str, object], V7StageMatrix, Path, dict[str, object], str, dict[str, object], AuthenticatedCheckpoint, tuple[AuthenticatedCheckpoint, ...]]:
    if resolved_config.get("test_only") is True:
        config = _validate_test_config(resolved_config)
    else:
        config = _authenticate_production(resolved_config, stage_matrix)[0] if stage_matrix is not None else _plain(resolved_config)
    run = _run_dir(config); matrix = _matrix_for_validator(config, stage_matrix, run)
    config_row, config_digest = _decode_json_file(run / "CONFIG.json", "CONFIG")
    if canonical_json_bytes(config_row) != canonical_json_bytes(_plain(config)) or config_digest != _sha_bytes(canonical_json_bytes(_plain(config))):
        raise TrainingStateError("CONFIG differs from resolved config")
    status, status_digest = _decode_json_file(run / "STATUS.json", "STATUS")
    state = status.get("status")
    if state not in ("running", "completed", "failed"):
        raise TrainingStateError("STATUS state is invalid")
    _validate_run_namespace(run, str(state))
    payload, current, records = _load_chain(run, config, matrix)
    return config, matrix, run, status, status_digest, payload, current, records


def validate_resumable_run(
    resolved_config: Mapping[str, object], *, stage_matrix: object | None = None,
    expected_completed_updates: int | None = None,
) -> ResumableRun:
    config, matrix, run, status, status_digest, _, current, records = _validate_common(resolved_config, stage_matrix)
    _validate_run_namespace(run, "running")
    _validate_artifact_tree(run, "running")
    _validate_semantic(status, {"schema", "status", "config_sha256", "current_step", "current_checkpoint_sha256", "stage", "stage_matrix_sha256", "authorization_path", "authorization_sha256", "semantic_sha256"}, _STATUS_SCHEMA, "running STATUS")
    if status != _running_status(str(config["config_sha256"]), current):
        raise TrainingStateError("running STATUS mismatch")
    if expected_completed_updates is not None and (type(expected_completed_updates) is not int or current.completed_updates != expected_completed_updates):
        raise TrainingStateError("durable update differs from expectation")
    if current.completed_updates >= _total_updates(config):
        raise TrainingStateError("terminal prefix is not resumable")
    endpoints = _collect_endpoint_metrics(run, config, matrix, records)
    return ResumableRun(run, str(config["config_sha256"]), run / "STATUS.json", status_digest, current, endpoints)


def _validate_manifest(
    run: Path, config: Mapping[str, object], current: AuthenticatedCheckpoint,
    records: Sequence[AuthenticatedCheckpoint],
) -> tuple[Path, str, Mapping[str, str]]:
    path = run / "artifacts/manifest.json"; row, digest = _decode_json_file(path, "artifact manifest")
    _validate_semantic(row, {"schema", "config_sha256", "terminal_checkpoint_sha256", "grant_history", "artifacts", "semantic_sha256"}, _MANIFEST_SCHEMA, "artifact manifest")
    if row["config_sha256"] != config["config_sha256"] or row["terminal_checkpoint_sha256"] != current.checkpoint_sha256 or row["grant_history"] != _grant_history(records, config) or type(row["artifacts"]) is not list:
        raise TrainingStateError("artifact manifest identity mismatch")
    observed: dict[str, str] = {}; previous: bytes | None = None
    for item in row["artifacts"]:
        if type(item) is not dict or set(item) != {"path", "size", "sha256"}:
            raise TrainingStateError("artifact record schema mismatch")
        relative = item["path"]
        if type(relative) is not str or not relative.startswith("artifacts/") or PurePosixPath(relative).is_absolute() or any(part in ("", ".", "..") for part in PurePosixPath(relative).parts):
            raise TrainingStateError("artifact relative path is unsafe")
        encoded = relative.encode()
        if previous is not None and encoded <= previous: raise TrainingStateError("artifact order mismatch")
        previous = encoded
        raw = _safe_existing_file(run / relative)
        if item["size"] != len(raw) or item["sha256"] != _sha_bytes(raw): raise TrainingStateError("artifact digest mismatch")
        observed[relative] = item["sha256"]
    actual = {path.relative_to(run).as_posix() for path in (run / "artifacts").rglob("*") if path.is_file() and path.name != "manifest.json"}
    if set(observed) != actual: raise TrainingStateError("artifact manifest coverage mismatch")
    return path, digest, _deep_immutable(dict(sorted(observed.items())))


def validate_completed_run(
    resolved_config: Mapping[str, object], *, stage_matrix: object | None = None,
) -> CompletedRun:
    config, matrix, run, status, status_digest, payload, current, records = _validate_common(resolved_config, stage_matrix)
    _validate_run_namespace(run, "completed")
    _validate_artifact_tree(run, "completed")
    _validate_semantic(status, {"schema", "status", "config_sha256", "current_step", "current_checkpoint_sha256", "artifact_manifest_path", "artifact_manifest_sha256", "semantic_sha256"}, _STATUS_SCHEMA, "completed STATUS")
    if status["status"] != "completed" or status["config_sha256"] != config["config_sha256"] or status["current_step"] != _total_updates(config) or status["current_checkpoint_sha256"] != current.checkpoint_sha256 or status["artifact_manifest_path"] != "artifacts/manifest.json":
        raise TrainingStateError("completed STATUS mismatch")
    manifest_path, manifest_sha, artifacts = _validate_manifest(run, config, current, records)
    if status["artifact_manifest_sha256"] != manifest_sha: raise TrainingStateError("completed manifest SHA mismatch")
    logits = np.load(io.BytesIO(_safe_existing_file(run / "artifacts/final_logits.npy")), allow_pickle=False)
    density = np.load(io.BytesIO(_safe_existing_file(run / "artifacts/final_density.npy")), allow_pickle=False)
    hard = np.load(io.BytesIO(_safe_existing_file(run / "artifacts/final_hard_005.npy")), allow_pickle=False)
    expected_logits = payload["logits"][0, 0].numpy()
    expected_density = torch.sigmoid(payload["logits"].to(
        device=torch.device(_device_text(config)), dtype=torch.float32,
    ))[0, 0].cpu().numpy()
    if logits.dtype != np.float32 or density.dtype != np.float32 or hard.dtype != np.bool_ or not np.array_equal(logits, expected_logits) or not np.array_equal(density, expected_density) or not np.array_equal(hard, expected_density >= 0.05):
        raise TrainingStateError("terminal arrays differ from checkpoint")
    if _decode_json_file(run / "artifacts/audit.json", "audit")[0] != payload["audit"] or _decode_json_file(run / "artifacts/resource.json", "resource")[0] != payload["resource"]:
        raise TrainingStateError("terminal audit/resource mismatch")
    endpoints = _collect_endpoint_metrics(run, config, matrix, records)
    return CompletedRun(run, str(config["config_sha256"]), run / "STATUS.json", status_digest, current, manifest_path, manifest_sha, artifacts, endpoints)


def _failure_code(observation: Mapping[str, object], completed: int) -> str:
    if not observation["loss_finite"] or not observation["gradient_finite"] or not observation["parameters_finite"]: return "NONFINITE_TRAINING"
    active_updates = max(0, completed - 100)
    hit_slots, barrier_slots = (active_updates + 1) // 2, active_updates // 2
    profile = observation.get("_profile")
    expected_hit = hit_slots if profile in ("hit_only", "full") else 0
    expected_barrier = barrier_slots if profile in ("barrier_only", "full") else 0
    if (
        observation["clip_count"] != completed
        or observation["optimizer_step_count"] != completed
        or observation["step_time_count"] != completed
        or not observation["slot_valid"] or not observation["disabled_not_constructed"]
        or observation["hit_eligible_slots"] != expected_hit
        or observation["barrier_eligible_slots"] != expected_barrier
        or observation["hit_active_slots"] > observation["hit_eligible_slots"]
        or observation["barrier_active_slots"] > observation["barrier_eligible_slots"]
    ): return "STEP_CONTRACT"
    if not observation["cuda_executed"]: return "CUDA_NOT_USED"
    if not observation["step_times_finite"] or observation["min_step_seconds"] is None or observation["min_step_seconds"] <= 0.0 or not (0 < observation["peak_allocated_bytes"] < observation["device_total_bytes"]): return "RESOURCE_INVALID"
    return "NONE"


_OBSERVATION_KEYS = {
    "schema", "config_sha256", "stage_matrix_sha256", "authorization_sha256",
    "last_checkpoint_path", "last_checkpoint_sha256", "completed_updates", "device",
    "loss_finite", "gradient_finite", "parameters_finite", "cuda_executed",
    "clip_count", "optimizer_step_count", "slot_valid", "disabled_not_constructed",
    "step_time_count", "step_times_finite", "min_step_seconds",
    "peak_allocated_bytes", "device_total_bytes", "hit_eligible_slots",
    "hit_active_slots", "barrier_eligible_slots", "barrier_active_slots",
    "training_source_bundle_sha256", "semantic_sha256",
}


def _validate_failure_observation(
    observation: object, config: Mapping[str, object], matrix: V7StageMatrix,
    checkpoint: AuthenticatedCheckpoint,
) -> tuple[dict[str, object], str]:
    if type(observation) is not dict:
        raise TrainingStateError("failure observation must be an exact object")
    row = dict(observation)
    _validate_semantic(row, _OBSERVATION_KEYS, _OBSERVATION_SCHEMA, "failure observation")
    if (
        row["config_sha256"] != config["config_sha256"]
        or row["stage_matrix_sha256"] != matrix.semantic_sha256
        or row["authorization_sha256"] != _grant(matrix)["authorization_sha256"]
        or row["last_checkpoint_path"] != f"checkpoints/{checkpoint.checkpoint_path.name}"
        or row["last_checkpoint_sha256"] != checkpoint.checkpoint_sha256
        or row["completed_updates"] != checkpoint.completed_updates
        or row["device"] != "cuda:0"
        or row["training_source_bundle_sha256"] != _source_bundle_sha(config)
    ):
        raise TrainingStateError("failure observation identity mismatch")
    for key in (
        "loss_finite", "gradient_finite", "parameters_finite", "cuda_executed",
        "slot_valid", "disabled_not_constructed", "step_times_finite",
    ):
        if type(row[key]) is not bool:
            raise TrainingStateError(f"failure observation {key} must be an exact boolean")
    for key in (
        "completed_updates", "clip_count", "optimizer_step_count", "step_time_count",
        "peak_allocated_bytes", "device_total_bytes", "hit_eligible_slots",
        "hit_active_slots", "barrier_eligible_slots", "barrier_active_slots",
    ):
        _exact_int(row[key], f"failure observation {key}")
    minimum = row["min_step_seconds"]
    if minimum is not None and (type(minimum) is not float or not math.isfinite(minimum)):
        raise TrainingStateError("failure observation minimum time type mismatch")
    if ((row["step_time_count"] == 0 or not row["step_times_finite"]) != (minimum is None)):
        raise TrainingStateError("failure observation minimum time nullability mismatch")
    derivation = dict(row); derivation["_profile"] = config["profile"]
    code = _failure_code(derivation, checkpoint.completed_updates)
    if code == "NONE":
        raise TrainingStateError("failure observation does not represent a registered failure")
    return row, code


def _failure_record_payload(
    config: Mapping[str, object], matrix: V7StageMatrix,
    checkpoint: AuthenticatedCheckpoint, code: str,
) -> dict[str, object]:
    return _semantic({
        "schema": FAILURE_RECORD_SCHEMA, "config_sha256": config["config_sha256"],
        "stage_matrix_sha256": matrix.semantic_sha256,
        "authorization_sha256": _grant(matrix)["authorization_sha256"],
        "last_checkpoint_path": f"checkpoints/{checkpoint.checkpoint_path.name}",
        "last_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "completed_updates": checkpoint.completed_updates, "failure_code": code,
        "training_source_bundle_sha256": _source_bundle_sha(config),
    })


def _source_bundle_sha(config: Mapping[str, object]) -> str:
    return (
        hashlib.sha256(b"test-source-bundle").hexdigest()
        if config.get("test_only") is True
        else str(config["source_bundle_sha256"])
    )


def _observation_from_checkpoint(
    config: Mapping[str, object], matrix: V7StageMatrix,
    checkpoint: AuthenticatedCheckpoint, payload: Mapping[str, object],
) -> dict[str, object]:
    history, audit, resource = payload["history"], payload["audit"], payload["resource"]
    disabled = all(
        row["auxiliary_constructed"] is False
        for row in history
        if row["component"] == "base" and row["multiplier"] > 0.0
    )
    row: dict[str, object] = {
        "schema": _OBSERVATION_SCHEMA, "config_sha256": config["config_sha256"],
        "stage_matrix_sha256": matrix.semantic_sha256,
        "authorization_sha256": _grant(matrix)["authorization_sha256"],
        "last_checkpoint_path": f"checkpoints/{checkpoint.checkpoint_path.name}",
        "last_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "completed_updates": checkpoint.completed_updates, "device": "cuda:0",
        "loss_finite": all(math.isfinite(float(item["base_loss"])) and (item["aux_loss"] is None or math.isfinite(float(item["aux_loss"]))) for item in history),
        "gradient_finite": all(math.isfinite(float(item[key])) for item in history for key in ("base_gradient_norm", "raw_auxiliary_gradient_norm", "final_auxiliary_gradient_norm", "base_auxiliary_dot", "effective_ratio")),
        "parameters_finite": bool(torch.isfinite(payload["logits"]).all().item()),
        "cuda_executed": resource["cuda_executed"],
        "clip_count": sum(int(item["clip_count"]) for item in history),
        "optimizer_step_count": sum(int(item["optimizer_step_count"]) for item in history),
        "slot_valid": True, "disabled_not_constructed": disabled,
        "step_time_count": resource["step_time_count"], "step_times_finite": resource["step_times_finite"],
        "min_step_seconds": resource["min_step_seconds"],
        "peak_allocated_bytes": resource["peak_allocated_bytes"], "device_total_bytes": resource["device_total_bytes"],
        "hit_eligible_slots": audit["hit_eligible_slots"], "hit_active_slots": audit["hit_active_slots"],
        "barrier_eligible_slots": audit["barrier_eligible_slots"], "barrier_active_slots": audit["barrier_active_slots"],
        "training_source_bundle_sha256": _source_bundle_sha(config),
    }
    row["semantic_sha256"] = canonical_sha256(row)
    return row


def _publish_failed_from_observation(
    run: Path, config: Mapping[str, object], matrix: V7StageMatrix,
    checkpoint: AuthenticatedCheckpoint, observation: Mapping[str, object],
    *, transaction: _V7RunTransaction | None = None,
) -> FailedRun:
    validated_observation, code = _validate_failure_observation(
        dict(observation), config, matrix, checkpoint
    )
    _reauthenticate_before_publication(
        config, matrix, checkpoint.completed_updates,
    )
    failure_dir = run / "failure"
    if transaction is None:
        failure_dir.mkdir(exist_ok=True)
    observation_sha = _publish_run_json(
        run, Path("failure/observation.json"), validated_observation,
        create_once=True, transaction=transaction,
    )
    record = _failure_record_payload(config, matrix, checkpoint, code)
    record_sha = _publish_run_json(
        run, Path("failure/failure.json"), record, create_once=True,
        transaction=transaction,
    )
    status = _semantic({
        "schema": _STATUS_SCHEMA, "status": "failed", "config_sha256": config["config_sha256"],
        "current_step": checkpoint.completed_updates,
        "last_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "observation_path": "failure/observation.json", "observation_sha256": observation_sha,
        "failure_record_path": "failure/failure.json", "failure_record_sha256": record_sha,
    })
    _reauthenticate_before_publication(
        config, matrix, checkpoint.completed_updates,
    )
    _publish_run_json(
        run, Path("STATUS.json"), status, create_once=False,
        transaction=transaction,
    )
    return validate_failed_run(config, stage_matrix=matrix)


def validate_failed_run(
    resolved_config: Mapping[str, object], *, stage_matrix: object | None = None,
) -> FailedRun:
    config, matrix, run, status, status_digest, payload, current, records = _validate_common(resolved_config, stage_matrix)
    _validate_run_namespace(run, "failed")
    _validate_artifact_tree(run, "failed")
    failure_dir = run / "failure"
    if (
        failure_dir.is_symlink() or not failure_dir.is_dir()
        or {path.name for path in failure_dir.iterdir()}
        != {"observation.json", "failure.json"}
    ):
        raise TrainingStateError("failure evidence tree namespace mismatch")
    _validate_semantic(status, {"schema", "status", "config_sha256", "current_step", "last_checkpoint_sha256", "observation_path", "observation_sha256", "failure_record_path", "failure_record_sha256", "semantic_sha256"}, _STATUS_SCHEMA, "failed STATUS")
    observation_path = run / "failure/observation.json"; record_path = run / "failure/failure.json"
    observation, observation_sha = _decode_json_file(observation_path, "failure observation")
    record, record_sha = _decode_json_file(record_path, "failure record")
    if status["status"] != "failed" or status["config_sha256"] != config["config_sha256"] or status["current_step"] != current.completed_updates or status["last_checkpoint_sha256"] != current.checkpoint_sha256 or status["observation_path"] != "failure/observation.json" or status["observation_sha256"] != observation_sha or status["failure_record_path"] != "failure/failure.json" or status["failure_record_sha256"] != record_sha:
        raise TrainingStateError("failed STATUS mismatch")
    record_keys = {"schema", "config_sha256", "stage_matrix_sha256", "authorization_sha256", "last_checkpoint_path", "last_checkpoint_sha256", "completed_updates", "failure_code", "training_source_bundle_sha256", "semantic_sha256"}
    validated_observation, code = _validate_failure_observation(
        observation, config, matrix, current
    )
    _validate_semantic(record, record_keys, FAILURE_RECORD_SCHEMA, "failure record")
    expected_record = _failure_record_payload(config, matrix, current, code)
    if canonical_json_bytes(record) != canonical_json_bytes(expected_record):
        raise TrainingStateError("failure record differs from exact derived evidence")
    endpoints = _collect_endpoint_metrics(run, config, matrix, records)
    return FailedRun(run, str(config["config_sha256"]), run / "STATUS.json", status_digest, current, observation_path, observation_sha, record_path, record_sha, endpoints)


def validate_endpoint_checkpoint(
    resolved_config: Mapping[str, object], *, completed_updates: int,
    stage_matrix: object | None = None,
) -> AuthenticatedCheckpoint:
    config, matrix, run, _, _, _, _, records = _validate_common(resolved_config, stage_matrix)
    if type(completed_updates) is not int: raise ValueError("completed_updates must be exact int")
    matches = [row for row in records if row.completed_updates == completed_updates]
    if len(matches) != 1 or completed_updates not in (matrix.endpoint, 300, 1000, 3000):
        raise TrainingStateError("endpoint checkpoint is absent/unregistered")
    return matches[0]


def validate_endpoint_metrics(
    resolved_config: Mapping[str, object], *, completed_updates: int,
    stage_matrix: object | None = None,
) -> AuthenticatedEndpointMetrics:
    checkpoint = validate_endpoint_checkpoint(resolved_config, completed_updates=completed_updates, stage_matrix=stage_matrix)
    if resolved_config.get("test_only") is True:
        config = _validate_test_config(resolved_config)
        matrix = _matrix_for_validator(config, stage_matrix, _run_dir(config))
    elif stage_matrix is None:
        claimed = _plain(resolved_config)
        if type(claimed) is not dict:
            raise ValueError("production config must be an exact mapping")
        matrix = _matrix_for_validator(claimed, None, _run_dir(claimed))
        config = _authenticate_production(claimed, matrix)[0]
    else:
        config, matrix = _authenticate_production(resolved_config, stage_matrix)
    historical = _matrix_for_checkpoint_record(config, matrix, checkpoint)
    return _validate_endpoint_file(
        _run_dir(config) / f"artifacts/metrics/step_{completed_updates:04d}.json",
        config, historical, checkpoint,
    )


def _projection_context(
    resolved_config: Mapping[str, object], stage_matrix: object,
) -> tuple[
    dict[str, object], V7StageMatrix, V7StageMatrix,
    ResumableRun | CompletedRun | FailedRun,
    tuple[AuthenticatedCheckpoint, ...],
]:
    config = (
        _validate_test_config(resolved_config)
        if resolved_config.get("test_only") is True
        else _authenticate_production(resolved_config, stage_matrix)[0]
    )
    run = _run_dir(config)
    requested = _matrix_for_validator(config, stage_matrix, run)
    if config.get("test_only") is True:
        try:
            live = _matrix_for_validator(config, None, run)
        except ValueError:
            live = requested
    else:
        live = _matrix_for_validator(config, None, run)
    status, _ = _decode_json_file(run / "STATUS.json", "STATUS")
    state_name = status.get("status")
    if state_name == "failed":
        state: ResumableRun | CompletedRun | FailedRun = validate_failed_run(
            config, stage_matrix=live,
        )
    elif state_name == "completed":
        state = validate_completed_run(config, stage_matrix=live)
    elif state_name == "running":
        state = validate_resumable_run(config, stage_matrix=live)
    else:
        raise TrainingStateError("projection STATUS is invalid")
    _, _, records = _load_chain(run, config, live)
    return config, requested, live, state, records


def _checkpoint_belongs_to_matrix(
    checkpoint: AuthenticatedCheckpoint, matrix: V7StageMatrix,
) -> bool:
    grant = _grant(matrix)
    return (
        checkpoint.stage == grant["stage"]
        and checkpoint.stage_matrix_sha256 == grant["stage_matrix_sha256"]
        and checkpoint.authorization_path == grant["authorization_path"]
        and checkpoint.authorization_sha256 == grant["authorization_sha256"]
    )


def _checkpoint_for_matrix_endpoint(
    records: Sequence[AuthenticatedCheckpoint], matrix: V7StageMatrix,
) -> AuthenticatedCheckpoint:
    matches = [
        checkpoint for checkpoint in records
        if checkpoint.completed_updates == matrix.endpoint
        and _checkpoint_belongs_to_matrix(checkpoint, matrix)
    ]
    if len(matches) != 1:
        raise TrainingStateError("supplied stage endpoint checkpoint is absent")
    return matches[0]


def project_engineering_row(
    resolved_config: Mapping[str, object], *, stage_matrix: object, physical_index: int,
) -> Mapping[str, object]:
    if type(physical_index) is not int: raise ValueError("physical_index must be exact int")
    config, matrix, live, state, records = _projection_context(
        resolved_config, stage_matrix,
    )
    if not 0 <= physical_index < len(matrix.physical_rows) or matrix.physical_rows[physical_index]["config_sha256"] != config["config_sha256"]:
        raise ValueError("physical index/config mismatch")
    failed = isinstance(state, FailedRun) and _checkpoint_belongs_to_matrix(
        state.last_checkpoint, matrix,
    )
    checkpoint = (
        state.last_checkpoint if failed
        else _checkpoint_for_matrix_endpoint(records, matrix)
    )
    payload, _ = _load_checkpoint_path(
        checkpoint.checkpoint_path, config, matrix,
        expected_sha256=checkpoint.checkpoint_sha256,
    )
    resource, audit = payload["resource"], payload["audit"]
    row = {
        "schema": ENGINEERING_ROW_SCHEMA, "stage": matrix.stage, "physical_index": physical_index,
        "task_id": config["task_id"], "budget_id": config["budget_id"], "profile": config["profile"],
        "config_sha256": config["config_sha256"], "run_path": config["run_path"],
        "completed_updates": checkpoint.completed_updates, "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "status": "FAIL" if failed else "PASS", "failure_code": "NONE",
        "device": "cuda:0", "loss_finite": True, "gradient_finite": True, "parameters_finite": True,
        "cuda_executed": resource["cuda_executed"], "clip_count": checkpoint.completed_updates,
        "optimizer_step_count": checkpoint.completed_updates, "slot_valid": True,
        "disabled_not_constructed": True, "step_time_count": resource["step_time_count"],
        "step_times_finite": resource["step_times_finite"], "min_step_seconds": resource["min_step_seconds"],
        "peak_allocated_bytes": resource["peak_allocated_bytes"], "device_total_bytes": resource["device_total_bytes"],
        "hit_eligible_slots": audit["hit_eligible_slots"], "hit_active_slots": audit["hit_active_slots"],
        "barrier_eligible_slots": audit["barrier_eligible_slots"], "barrier_active_slots": audit["barrier_active_slots"],
        "failure_record_path": None, "failure_record_sha256": None,
    }
    if failed:
        failure = state
        assert isinstance(failure, FailedRun)
        record, _ = _decode_json_file(failure.failure_record_path, "failure record")
        observation, _ = _decode_json_file(
            failure.observation_path, "failure observation"
        )
        for key in (
            "loss_finite", "gradient_finite", "parameters_finite", "cuda_executed",
            "clip_count", "optimizer_step_count", "slot_valid",
            "disabled_not_constructed", "step_time_count", "step_times_finite",
            "min_step_seconds", "peak_allocated_bytes", "device_total_bytes",
            "hit_eligible_slots", "hit_active_slots", "barrier_eligible_slots",
            "barrier_active_slots",
        ):
            row[key] = observation[key]
        row["failure_code"] = record["failure_code"]
        row["failure_record_path"] = f"{config['run_path']}/failure/failure.json"
        row["failure_record_sha256"] = failure.failure_record_sha256
    return _deep_immutable(_semantic(row))


def project_engineering_prefix(
    resolved_config: Mapping[str, object], *, stage_matrix: object,
    physical_index: int, completed_updates: int,
) -> Mapping[str, object]:
    if type(physical_index) is not int:
        raise ValueError("physical_index must be an exact int")
    if type(completed_updates) is not int:
        raise ValueError("completed_updates must be an exact int")
    config, matrix, _, state, records = _projection_context(
        resolved_config, stage_matrix,
    )
    if completed_updates not in (2, 200):
        raise ValueError("engineering prefix is not a registered update-2/200 cap")

    run = _run_dir(config)
    current = state.last_checkpoint if isinstance(state, FailedRun) else state.current_checkpoint
    if current.completed_updates < completed_updates:
        raise TrainingStateError("requested engineering prefix is absent from the authenticated chain")
    matches = [
        record for record in records
        if record.completed_updates == completed_updates
        and _checkpoint_belongs_to_matrix(record, matrix)
    ]
    if len(matches) != 1:
        raise TrainingStateError("requested engineering prefix is absent from the authenticated chain")
    checkpoint = matches[0]
    _validate_absolute_cap(config, matrix, completed_updates)
    if (
        not 0 <= physical_index < len(matrix.physical_rows)
        or matrix.physical_rows[physical_index]["config_sha256"]
        != config["config_sha256"]
    ):
        raise ValueError("physical index/config mismatch")
    payload, authenticated = _load_checkpoint_path(
        checkpoint.checkpoint_path, config, matrix,
        expected_sha256=checkpoint.checkpoint_sha256,
    )
    if authenticated != checkpoint:
        raise TrainingStateError("engineering prefix checkpoint changed")
    observation = _observation_from_checkpoint(
        config, matrix, checkpoint, payload,
    )
    derivation = dict(observation); derivation["_profile"] = config["profile"]
    if _failure_code(derivation, completed_updates) != "NONE":
        raise TrainingStateError("engineering prefix is not PASS evidence")
    row = {
        "schema": _ENGINEERING_PREFIX_SCHEMA,
        "stage": matrix.stage,
        "physical_index": physical_index,
        "task_id": config["task_id"],
        "budget_id": config["budget_id"],
        "profile": config["profile"],
        "config_sha256": config["config_sha256"],
        "run_path": config["run_path"],
        "completed_updates": completed_updates,
        "checkpoint_path": f"{config['run_path']}/checkpoints/{checkpoint.checkpoint_path.name}",
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "stage_matrix_sha256": checkpoint.stage_matrix_sha256,
        "authorization_sha256": checkpoint.authorization_sha256,
        "device": observation["device"],
        "loss_finite": observation["loss_finite"],
        "gradient_finite": observation["gradient_finite"],
        "parameters_finite": observation["parameters_finite"],
        "cuda_executed": observation["cuda_executed"],
        "clip_count": observation["clip_count"],
        "optimizer_step_count": observation["optimizer_step_count"],
        "slot_valid": observation["slot_valid"],
        "disabled_not_constructed": observation["disabled_not_constructed"],
        "step_time_count": observation["step_time_count"],
        "step_times_finite": observation["step_times_finite"],
        "min_step_seconds": observation["min_step_seconds"],
        "peak_allocated_bytes": observation["peak_allocated_bytes"],
        "device_total_bytes": observation["device_total_bytes"],
        "hit_eligible_slots": observation["hit_eligible_slots"],
        "hit_active_slots": observation["hit_active_slots"],
        "barrier_eligible_slots": observation["barrier_eligible_slots"],
        "barrier_active_slots": observation["barrier_active_slots"],
    }
    return _deep_immutable(_semantic(row))


def project_metric_row(
    resolved_config: Mapping[str, object], *, stage_matrix: object, logical_index: int,
) -> Mapping[str, object]:
    if type(logical_index) is not int: raise ValueError("logical_index must be exact int")
    config, matrix, _, state, records = _projection_context(
        resolved_config, stage_matrix,
    )
    if not 0 <= logical_index < len(matrix.logical_rows): raise ValueError("logical index out of range")
    logical = matrix.logical_rows[logical_index]
    if logical["config_sha256"] != config["config_sha256"]: raise ValueError("logical config mismatch")
    if isinstance(state, FailedRun) and _checkpoint_belongs_to_matrix(
        state.last_checkpoint, matrix,
    ):
        raise TrainingStateError("failed supplied stage has no legal metric row")
    checkpoint = _checkpoint_for_matrix_endpoint(records, matrix)
    metric = _validate_endpoint_file(
        _run_dir(config) / f"artifacts/metrics/step_{matrix.endpoint:04d}.json",
        config, matrix, checkpoint,
    )
    row = {
        "schema": METRIC_ROW_SCHEMA, "stage": matrix.stage, "logical_index": logical_index,
        "task_id": logical["task_id"], "budget_id": logical["budget_id"], "profile": logical["profile"],
        "config_sha256": config["config_sha256"], "run_path": config["run_path"],
        "completed_updates": matrix.endpoint, "checkpoint_sha256": metric.checkpoint_sha256,
        "metrics_path": f"{config['run_path']}/artifacts/metrics/step_{matrix.endpoint:04d}.json",
        "metrics_sha256": metric.sha256,
        "iou": metric.aggregate["mean_iou"], "dice": metric.aggregate["mean_dice"],
        "recall": metric.aggregate["mean_recall"], "fpr": metric.aggregate["mean_fpr"],
    }
    return _deep_immutable(_semantic(row))


def _recover_interrupted_checkpoint_publication(
    config: dict[str, object], matrix: V7StageMatrix, *,
    transaction: _V7RunTransaction | None,
) -> ResumableRun | CompletedRun | FailedRun:
    run = _run_dir(config)
    status_present = (
        (run / "STATUS.json").exists() if transaction is None
        else transaction.digest(Path("STATUS.json")) is not None
    )
    if not status_present:
        expected_root = {"CONFIG.json", "checkpoints", "artifacts"}
        if {path.name for path in run.iterdir()} != expected_root:
            raise TrainingStateError("crashed genesis run namespace is not exact")
        config_row, _ = _decode_json_file(run / "CONFIG.json", "CONFIG")
        if canonical_json_bytes(config_row) != canonical_json_bytes(_plain(config)):
            raise TrainingStateError("crashed genesis CONFIG identity mismatch")
        checkpoints = run / "checkpoints"
        if {path.name for path in checkpoints.iterdir()} != {"step_0000.pt"}:
            raise TrainingStateError("crashed genesis checkpoint namespace is not exact")
        _validate_artifact_tree(run, "running")
        if any((run / "artifacts/metrics").iterdir()):
            raise TrainingStateError("crashed genesis metric namespace is not empty")
        genesis_payload, genesis = _load_checkpoint_path(
            checkpoints / "step_0000.pt", config, matrix,
        )
        if (
            genesis.completed_updates != 0
            or genesis_payload["predecessor_step"] is not None
            or genesis.predecessor_checkpoint_sha256 is not None
        ):
            raise TrainingStateError("crashed genesis generation is not exact")
        _reauthenticate_before_publication(config, matrix, 0)
        _publish_run_json(
            run, Path("checkpoints/LATEST.json"),
            _latest_payload(
                str(config["config_sha256"]), 0, genesis.checkpoint_sha256, None,
            ),
            create_once=False, transaction=transaction,
        )
        _publish_run_json(
            run, Path("STATUS.json"),
            _running_status(str(config["config_sha256"]), genesis),
            create_once=False, transaction=transaction,
        )
    status, _ = _decode_json_file(run / "STATUS.json", "STATUS")
    if status.get("status") not in ("running", "completed", "failed"):
        raise TrainingStateError("STATUS state is invalid")
    if status.get("status") == "completed": return validate_completed_run(config, stage_matrix=matrix)
    if status.get("status") == "failed": return validate_failed_run(config, stage_matrix=matrix)
    expected_root = {"CONFIG.json", "STATUS.json", "checkpoints", "artifacts"}
    observed_root = {path.name for path in run.iterdir()}
    if observed_root not in (expected_root, expected_root | {"failure"}):
        raise TrainingStateError("recovery run namespace has extra/missing entries")
    _validate_semantic(
        status,
        {"schema", "status", "config_sha256", "current_step", "current_checkpoint_sha256", "stage", "stage_matrix_sha256", "authorization_path", "authorization_sha256", "semantic_sha256"},
        _STATUS_SCHEMA, "running STATUS",
    )
    if status["status"] != "running" or status["config_sha256"] != config["config_sha256"]:
        raise TrainingStateError("recovery running STATUS identity mismatch")
    payload, current, records = _load_chain(run, config, matrix, allow_orphans=True)
    all_paths = sorted(path for path in (run / "checkpoints").iterdir() if path.name != "LATEST.json")
    known = {record.checkpoint_path.name for record in records}
    orphans = [path for path in all_paths if path.name not in known]
    if len(orphans) > 1: raise TrainingStateError("ambiguous orphan successors")
    if len(orphans) == 1:
        orphan_payload, orphan = _load_checkpoint_path(orphans[0], config, matrix)
        if orphan_payload["predecessor_step"] != current.completed_updates or orphan.predecessor_checkpoint_sha256 != current.checkpoint_sha256:
            raise TrainingStateError("orphan is not the exact successor")
        if canonical_json_bytes(orphan_payload["history"][: current.completed_updates]) != canonical_json_bytes(payload["history"]):
            raise TrainingStateError("orphan history is not an exact prefix extension")
        if not _rng_equal(payload["rng"], orphan_payload["rng"]):
            raise TrainingStateError("orphan RNG differs from predecessor RNG")
        _reauthenticate_before_publication(config, matrix, orphan.completed_updates)
        _publish_run_json(
            run, Path("checkpoints/LATEST.json"),
            _latest_payload(str(config["config_sha256"]), orphan.completed_updates, orphan.checkpoint_sha256, orphan.predecessor_checkpoint_sha256),
            create_once=False, transaction=transaction,
        )
        _publish_run_json(
            run, Path("STATUS.json"), _running_status(str(config["config_sha256"]), orphan),
            create_once=False, transaction=transaction,
        )
        payload, current, records = orphan_payload, orphan, (*records, orphan)
    elif status != _running_status(str(config["config_sha256"]), current):
        _reauthenticate_before_publication(config, matrix, current.completed_updates)
        _publish_run_json(
            run, Path("STATUS.json"), _running_status(str(config["config_sha256"]), current),
            create_once=False, transaction=transaction,
        )
    endpoint_metric_path = run / "artifacts/metrics" / f"step_{current.completed_updates:04d}.json"
    if current.completed_updates == matrix.endpoint and not endpoint_metric_path.exists():
        _reauthenticate_before_publication(config, matrix, current.completed_updates)
        endpoint_payload = _metric_payload_from_checkpoint(
            config, matrix, current,
        )
        _publish_endpoint_metric(
            run, endpoint_payload, transaction=transaction,
        )
    failure_dir = run / "failure"
    observation_path = failure_dir / "observation.json"
    if failure_dir.exists():
        if failure_dir.is_symlink() or not failure_dir.is_dir():
            raise TrainingStateError("failure recovery namespace is invalid")
        names = {path.name for path in failure_dir.iterdir()}
        if not names:
            if transaction is None:
                failure_dir.rmdir()
            else:
                transaction.remove_empty_directory(Path("failure"))
            return validate_resumable_run(
                config, stage_matrix=matrix,
                expected_completed_updates=current.completed_updates,
            )
        if "observation.json" not in names or not names <= {"observation.json", "failure.json"}:
            raise TrainingStateError("failure recovery tree is incomplete or foreign")
        observation, _ = _decode_json_file(observation_path, "failure observation")
        _validate_failure_observation(observation, config, matrix, current)
        _reauthenticate_before_publication(config, matrix, current.completed_updates)
        return _publish_failed_from_observation(
            run, config, matrix, current, observation,
            transaction=transaction,
        )
    if current.completed_updates == _total_updates(config):
        _reauthenticate_before_publication(config, matrix, current.completed_updates)
        runtime = _build_runtime(config)
        with torch.no_grad():
            runtime.logits.copy_(payload["logits"].to(
                device=runtime.logits.device, dtype=torch.float32
            ))
        return _finalize(
            run, config, matrix, runtime, current, records,
            transaction=transaction,
        )
    return validate_resumable_run(config, stage_matrix=matrix, expected_completed_updates=current.completed_updates)


def recover_interrupted_checkpoint_publication(
    resolved_config: Mapping[str, object], *, stage_matrix: object,
) -> ResumableRun | CompletedRun | FailedRun:
    if resolved_config.get("test_only") is True:
        config = _validate_test_config(resolved_config)
        matrix = _validate_test_matrix(config, stage_matrix)
        return _recover_interrupted_checkpoint_publication(
            config, matrix, transaction=None,
        )
    config, matrix = _authenticate_production(resolved_config, stage_matrix)
    with _V7RunTransaction(config) as transaction:
        result = _recover_interrupted_checkpoint_publication(
            config, matrix, transaction=transaction,
        )
        transaction.revalidate()
        return result


def _test_publish_authenticated_successor_without_pointer(config: Mapping[str, object], *, stage_matrix: object) -> None:
    row = _validate_test_config(config); matrix = _validate_test_matrix(row, stage_matrix); run = Path(row["run_path"])
    payload, current, _ = _load_chain(run, row, matrix); runtime = _restore_runtime(row, payload)
    history = copy.deepcopy(list(payload["history"])); timer = _DeterministicTestTimer()
    telemetry = _train_one_update(logits=runtime.logits, optimizer=runtime.optimizer, protocol=runtime.protocol, targets=runtime.targets, profile=row["profile"], budget_ratio=row["budget_ratio"], completed_updates=current.completed_updates, render_thresholded_fn=runtime.render_thresholded_fn, allow_test_size=True, timer=timer)
    history.append(asdict(telemetry)); _publish_checkpoint(run, row, matrix, runtime, telemetry.completed_updates, current, history, publish_pointer=False)


def _test_duplicate_successor_orphan(config: Mapping[str, object]) -> None:
    run = _run_dir(config); sources = sorted((run / "checkpoints").glob("step_*.pt"))
    latest = sources[-1]; (run / "checkpoints/step_9998.pt").write_bytes(latest.read_bytes())


def _fail_cell_for_test(config: Mapping[str, object], *, observation: Mapping[str, object]) -> FailedRun:
    row = _validate_test_config(config); matrix = _test_stage_matrix(row)
    runtime, checkpoint, _ = _initialize_run(row, matrix); run = Path(row["run_path"])
    defaults = {
        "loss_finite": True, "gradient_finite": True, "parameters_finite": True,
        "cuda_executed": True, "clip_count": 0, "optimizer_step_count": 0,
        "slot_valid": True, "disabled_not_constructed": True, "step_time_count": 0,
        "step_times_finite": True, "min_step_seconds": None,
        "peak_allocated_bytes": 1, "device_total_bytes": 100,
        "hit_eligible_slots": 0, "hit_active_slots": 0,
        "barrier_eligible_slots": 0, "barrier_active_slots": 0,
    }
    defaults.update(_plain(observation))
    source_sha = hashlib.sha256(b"test-source-bundle").hexdigest()
    obs = _semantic({
        "schema": _OBSERVATION_SCHEMA, "config_sha256": row["config_sha256"],
        "stage_matrix_sha256": matrix.semantic_sha256, "authorization_sha256": None,
        "last_checkpoint_path": "checkpoints/step_0000.pt", "last_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "completed_updates": 0, "device": "cuda:0", **defaults,
        "training_source_bundle_sha256": source_sha,
    })
    code = _failure_code(obs, 0)
    if code == "NONE": raise ValueError("test failure observation does not represent a gate failure")
    record = _semantic({
        "schema": FAILURE_RECORD_SCHEMA, "config_sha256": row["config_sha256"],
        "stage_matrix_sha256": matrix.semantic_sha256, "authorization_sha256": None,
        "last_checkpoint_path": "checkpoints/step_0000.pt", "last_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "completed_updates": 0, "failure_code": code, "training_source_bundle_sha256": source_sha,
    })
    (run / "failure").mkdir(); observation_sha = _json_publish(run / "failure/observation.json", obs, create_once=True); record_sha = _json_publish(run / "failure/failure.json", record, create_once=True)
    status = _semantic({
        "schema": _STATUS_SCHEMA, "status": "failed", "config_sha256": row["config_sha256"],
        "current_step": 0, "last_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "observation_path": "failure/observation.json", "observation_sha256": observation_sha,
        "failure_record_path": "failure/failure.json", "failure_record_sha256": record_sha,
    })
    _json_publish(run / "STATUS.json", status, create_once=False)
    return validate_failed_run(row, stage_matrix=matrix)


__all__ = [
    "AuthenticatedCheckpoint", "AuthenticatedEndpointMetrics", "CompletedRun", "FailedRun",
    "ResumableRun", "StepTelemetry", "TrainingStateError", "evaluate_six_view_metrics",
    "project_engineering_prefix", "project_engineering_row", "project_metric_row",
    "recover_interrupted_checkpoint_publication",
    "required_checkpoint_steps", "required_diagnostic_steps", "train_cell",
    "validate_completed_run", "validate_endpoint_checkpoint", "validate_endpoint_metrics",
    "validate_failed_run", "validate_resumable_run",
]
