"""Crash-safe lifecycle and artifact publication for soft-loss v4."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import stat
import time
from typing import Mapping, Sequence
import uuid

import cv2
import numpy as np
import torch

from shadowforge_pipeline.soft_voxel_v3.contours import make_contour_bands
from shadowforge_pipeline.soft_voxel_v3.data import normalize_target
from shadowforge_pipeline.soft_voxel_v3.metrics import evaluate_delivery_metrics
from shadowforge_pipeline.soft_voxel_v3.protocol import SadekarTFSProtocol

from .config import (
    BoundFileSnapshot,
    ResolvedV4Run,
    build_parser,
    canonical_trainer_argv,
    resolve_run_configuration,
    revalidate_resolved_dependencies,
    revalidate_resolved_run,
    step_config_from_resolved,
    validate_resolved_config,
)
from .diagnostics import (
    validate_gradient_engineering_gate,
    validate_redundancy_gate,
)
from .losses import auxiliary_slot, resolve_profile
from .provenance import (
    atomic_checkpoint,
    atomic_json,
    atomic_text,
    canonical_hash,
    canonical_json_bytes,
    environment_manifest,
    publish_json_once,
    sha256_file,
    snapshot_regular_file,
)
from .trainer_step import (
    AuxiliarySlotAudit,
    V4StepConfig,
    V4StepResult,
    train_one_v4_update,
)


_CHECKPOINT_SCHEMA = "pcraft_soft_loss_v4_checkpoint_v1"
_TRANSACTION_SCHEMA = "pcraft_soft_loss_v4_transaction_v1"
_POINTER_SCHEMA = "pcraft_soft_loss_v4_checkpoint_pointer_v1"
_HISTORY_SCHEMA = "pcraft_soft_loss_v4_history_row_v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TRANSACTION_PATH = re.compile(r"transactions/step_([0-9]{8})\Z")
_TRANSACTION_FILES = (
    "checkpoint.pt",
    "history_segment.jsonl",
    "transaction_manifest.json",
)
_FULL_PROFILES = {"v4_full", "v4_no_l1", "v4_no_l2"}
_AUXILIARY_PROFILES = {"dic_only", "dts_only", *_FULL_PROFILES}
_THRESHOLDS = (0.04, 0.05, 0.06)


@dataclass(frozen=True)
class ResumeState:
    completed_updates: int
    last_result: V4StepResult | None
    rng_state: dict[str, object]
    step_times: tuple[float, ...]
    peak_cuda_bytes: int
    slot_audits: tuple[AuxiliarySlotAudit, ...]
    committed_transaction: str | None = None
    committed_transaction_sha256: str | None = None
    committed_history: tuple[dict[str, object], ...] = ()


class V4TrainingStepError(RuntimeError):
    """A scientific update failed before a valid optimizer transition."""


class ResumePreflightError(ValueError):
    """Persisted lifecycle evidence failed validation before live mutation."""


@dataclass(frozen=True)
class AuthenticatedTransactionCheckpoint:
    completed_updates: int
    transaction: str
    transaction_manifest_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class AuthenticatedCheckpointSnapshot:
    identity: AuthenticatedTransactionCheckpoint
    checkpoint: BoundFileSnapshot


@dataclass(frozen=True)
class AuthenticatedCompletedRun:
    run_dir: str
    output_root_identity: dict[str, object]
    config: dict[str, object]
    config_hash: str
    status_sha256: str
    artifact_manifest_sha256: str
    artifacts_sha256: dict[str, str]
    transaction_checkpoints: tuple[AuthenticatedTransactionCheckpoint, ...]


@dataclass(frozen=True)
class AuthenticatedResumableRun:
    run_dir: str
    output_root_identity: dict[str, object]
    config: dict[str, object]
    config_hash: str
    status_sha256: str
    completed_updates: int
    latest_transaction: AuthenticatedTransactionCheckpoint
    pending_terminalization: str | None


@dataclass(frozen=True)
class _Runtime:
    logits: torch.Tensor
    optimizer: torch.optim.Optimizer
    protocol: object
    targets: torch.Tensor
    bands: object | None


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_mapping(value: object, keys: set[str], *, name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} has the wrong exact schema")
    return value


def _finite_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_live_state(
    logits: object, optimizer: object
) -> tuple[torch.Tensor, torch.optim.Adam]:
    if (
        not isinstance(logits, torch.Tensor)
        or not logits.is_leaf
        or not logits.requires_grad
        or not logits.is_floating_point()
        or not torch.isfinite(logits).all().item()
    ):
        raise ValueError("logits must be one finite floating leaf tensor")
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError("optimizer must be the fixed Adam optimizer")
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(owned) != 1 or owned[0] is not logits:
        raise ValueError("optimizer must own exactly the live logits tensor")
    return logits, optimizer


def _capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _validated_rng_state(value: object) -> dict[str, object]:
    payload = _exact_mapping(
        value, {"python", "numpy", "torch", "cuda"}, name="rng_state"
    )
    try:
        probe = random.Random()
        probe.setstate(payload["python"])
    except Exception as error:
        raise ValueError("Python RNG state is malformed") from error
    try:
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(payload["numpy"])
    except Exception as error:
        raise ValueError("NumPy RNG state is malformed") from error
    torch_state = payload["torch"]
    if (
        not isinstance(torch_state, torch.Tensor)
        or torch_state.device.type != "cpu"
        or torch_state.dtype != torch.uint8
        or torch_state.ndim != 1
    ):
        raise ValueError("Torch RNG state is malformed")
    cuda_state = payload["cuda"]
    if type(cuda_state) not in (list, tuple) or any(
        not isinstance(item, torch.Tensor)
        or item.device.type != "cpu"
        or item.dtype != torch.uint8
        or item.ndim != 1
        for item in cuda_state
    ):
        raise ValueError("CUDA RNG state is malformed")
    return {
        "python": copy.deepcopy(payload["python"]),
        "numpy": copy.deepcopy(payload["numpy"]),
        "torch": torch_state.clone(),
        "cuda": [item.clone() for item in cuda_state],
    }


def _restore_rng_state(value: Mapping[str, object]) -> None:
    state = _validated_rng_state(dict(value))
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology differs from the checkpoint")
        torch.cuda.set_rng_state_all(state["cuda"])


def _audit_from_mapping(value: object) -> AuxiliarySlotAudit:
    if isinstance(value, AuxiliarySlotAudit):
        value = asdict(value)
    expected = {field.name for field in fields(AuxiliarySlotAudit)}
    payload = _exact_mapping(value, expected, name="slot audit")
    try:
        return AuxiliarySlotAudit(**payload)
    except (TypeError, ValueError) as error:
        raise ValueError("slot audit fields are malformed") from error


def _result_from_mapping(value: object) -> V4StepResult:
    if isinstance(value, V4StepResult):
        value = asdict(value)
    expected = {field.name for field in fields(V4StepResult)}
    payload = _exact_mapping(value, expected, name="V4StepResult")
    if type(payload["global_step_before"]) is not int or payload["global_step_before"] < 0:
        raise ValueError("result global step is malformed")
    if (
        type(payload["completed_updates_after"]) is not int
        or payload["completed_updates_after"] != payload["global_step_before"] + 1
    ):
        raise ValueError("result completed update identity is malformed")
    for name in (
        "progress",
        "slot_multiplier",
        "loss_base",
        "auxiliary_scale",
        "pre_clip_gradient_norm",
        "post_clip_gradient_norm",
    ):
        _finite_float(payload[name], name=f"result {name}", nonnegative=True)
    for name in ("slot_endpoint", "loss_anchor", "loss_auxiliary"):
        if payload[name] is not None:
            _finite_float(payload[name], name=f"result {name}")
    if payload["slot_component"] is not None and payload["slot_component"] not in (
        "dic",
        "dts",
    ):
        raise ValueError("result slot component is malformed")
    audit = None
    if payload["slot_audit"] is not None:
        audit = _audit_from_mapping(payload["slot_audit"])
        if (
            audit.global_step_before != payload["global_step_before"]
            or audit.component != payload["slot_component"]
            or audit.endpoint != payload["slot_endpoint"]
            or audit.multiplier != payload["slot_multiplier"]
        ):
            raise ValueError("result slot audit identity mismatch")
    elif payload["slot_component"] is not None:
        raise ValueError("active result slot requires an audit")
    normalized = dict(payload)
    normalized["slot_audit"] = audit
    return V4StepResult(**normalized)


def _validated_times(value: object, *, completed_updates: int) -> tuple[float, ...]:
    if type(value) not in (list, tuple):
        raise ValueError("step_times must be a sequence")
    times = tuple(
        _finite_float(item, name="step time", nonnegative=True) for item in value
    )
    if len(times) != completed_updates:
        raise ValueError("step-time count must equal completed updates")
    return times


def _validate_slot_audit_prefix(
    audits: Sequence[AuxiliarySlotAudit | Mapping[str, object]],
    *,
    profile: object,
    total_steps: int,
    completed_updates: int,
) -> tuple[AuxiliarySlotAudit, ...]:
    registered = resolve_profile(profile.name if hasattr(profile, "name") else profile)
    if type(total_steps) is not int or total_steps <= 0:
        raise ValueError("total_steps must be a positive exact integer")
    if (
        type(completed_updates) is not int
        or completed_updates < 0
        or completed_updates > total_steps
    ):
        raise ValueError("completed_updates is outside the run")
    if type(audits) not in (list, tuple):
        raise ValueError("slot audits must be a sequence")
    expected: list[tuple[int, object]] = []
    for step in range(completed_updates):
        slot = auxiliary_slot(step, (step + 1) / float(total_steps), registered)
        if slot is not None:
            expected.append((step, slot))
    if len(audits) != len(expected):
        raise ValueError("slot audit prefix is incomplete or contains unexpected slots")
    from .diagnostics import _validate_audit

    result: list[AuxiliarySlotAudit] = []
    for raw, (step, slot) in zip(audits, expected):
        audit = _audit_from_mapping(raw)
        result.append(_validate_audit(audit, step, slot, registered))
    return tuple(result)


def _validate_resolved_identity(value: object) -> tuple[dict[str, object], str]:
    config = validate_resolved_config(value)
    return config, canonical_hash(config)


def _validated_checkpoint_payload(
    payload: object,
    *,
    expected_config: Mapping[str, object] | None = None,
    logits: torch.Tensor | None = None,
) -> tuple[ResumeState, torch.Tensor, dict[str, object], dict[str, object]]:
    required = {
        "schema",
        "resolved_config",
        "config_hash",
        "completed_updates",
        "logits",
        "optimizer",
        "last_result",
        "rng_state",
        "step_times",
        "peak_cuda_bytes",
        "slot_audits",
    }
    try:
        record = _exact_mapping(payload, required, name="checkpoint")
        if record["schema"] != _CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema mismatch")
        config, digest = _validate_resolved_identity(record["resolved_config"])
        if record["config_hash"] != digest:
            raise ValueError("checkpoint configuration hash mismatch")
        if expected_config is not None:
            expected, expected_digest = _validate_resolved_identity(dict(expected_config))
            if canonical_json_bytes(config) != canonical_json_bytes(expected) or digest != expected_digest:
                raise ValueError("checkpoint resolved configuration differs")
        completed = record["completed_updates"]
        if type(completed) is not int or completed < 0 or completed > config["steps"]:
            raise ValueError("checkpoint completed_updates is malformed")
        stored_logits = record["logits"]
        if (
            not isinstance(stored_logits, torch.Tensor)
            or not stored_logits.is_floating_point()
            or not torch.isfinite(stored_logits).all().item()
        ):
            raise ValueError("checkpoint logits are malformed")
        if logits is not None and (
            stored_logits.shape != logits.shape or stored_logits.dtype != logits.dtype
        ):
            raise ValueError("checkpoint logits do not match the live tensor")
        last_result = (
            None
            if record["last_result"] is None
            else _result_from_mapping(record["last_result"])
        )
        if completed == 0:
            if last_result is not None:
                raise ValueError("step-zero checkpoint must have no last result")
        elif last_result is None or last_result.completed_updates_after != completed:
            raise ValueError("positive checkpoint last result does not match its step")
        rng = _validated_rng_state(record["rng_state"])
        times = _validated_times(record["step_times"], completed_updates=completed)
        peak = record["peak_cuda_bytes"]
        if type(peak) is not int or peak < 0:
            raise ValueError("checkpoint peak CUDA bytes is malformed")
        profile = resolve_profile(config["profile"]["name"])
        audits = _validate_slot_audit_prefix(
            record["slot_audits"],
            profile=profile,
            total_steps=config["steps"],
            completed_updates=completed,
        )
        optimizer_state = record["optimizer"]
        if type(optimizer_state) is not dict or set(optimizer_state) != {
            "state",
            "param_groups",
        }:
            raise ValueError("checkpoint optimizer state is malformed")
        probe_logits = stored_logits.detach().clone().requires_grad_()
        probe_optimizer = torch.optim.Adam([probe_logits], lr=config["learning_rate"])
        probe_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        owned = [p for group in probe_optimizer.param_groups for p in group["params"]]
        if len(owned) != 1 or owned[0] is not probe_logits:
            raise ValueError("checkpoint optimizer ownership is malformed")
    except ResumePreflightError:
        raise
    except Exception as error:
        raise ResumePreflightError(str(error)) from error
    return (
        ResumeState(
            completed_updates=completed,
            last_result=last_result,
            rng_state=rng,
            step_times=times,
            peak_cuda_bytes=peak,
            slot_audits=audits,
        ),
        stored_logits.detach().clone(),
        copy.deepcopy(optimizer_state),
        config,
    )


def build_checkpoint_payload(
    *,
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    completed_updates: int,
    last_result: V4StepResult | None,
    rng_state: Mapping[str, object],
    step_times: Sequence[float],
    peak_cuda_bytes: int,
    slot_audits: Sequence[AuxiliarySlotAudit],
    resolved_config: Mapping[str, object],
) -> dict[str, object]:
    live_logits, adam = _validate_live_state(logits, optimizer)
    config, digest = _validate_resolved_identity(dict(resolved_config))
    if type(completed_updates) is not int:
        raise ValueError("completed_updates must be an exact integer")
    result = None if last_result is None else _result_from_mapping(last_result)
    rng = _validated_rng_state(dict(rng_state))
    times = _validated_times(step_times, completed_updates=completed_updates)
    if type(peak_cuda_bytes) is not int or peak_cuda_bytes < 0:
        raise ValueError("peak_cuda_bytes must be a non-negative exact integer")
    profile = resolve_profile(config["profile"]["name"])
    audits = _validate_slot_audit_prefix(
        slot_audits,
        profile=profile,
        total_steps=config["steps"],
        completed_updates=completed_updates,
    )
    if completed_updates == 0:
        if result is not None:
            raise ValueError("step-zero checkpoint must have no last result")
    elif result is None or result.completed_updates_after != completed_updates:
        raise ValueError("last result does not match the completed update")
    return {
        "schema": _CHECKPOINT_SCHEMA,
        "resolved_config": copy.deepcopy(config),
        "config_hash": digest,
        "completed_updates": completed_updates,
        "logits": live_logits.detach().to(device="cpu").clone(),
        "optimizer": copy.deepcopy(adam.state_dict()),
        "last_result": None if result is None else asdict(result),
        "rng_state": rng,
        "step_times": list(times),
        "peak_cuda_bytes": peak_cuda_bytes,
        "slot_audits": [asdict(audit) for audit in audits],
    }


def _optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        if type(state) is not dict:
            continue
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def restore_checkpoint_payload(
    path: Path,
    *,
    expected_config: Mapping[str, object],
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> ResumeState:
    live_logits, adam = _validate_live_state(logits, optimizer)
    try:
        snapshot = snapshot_regular_file(Path(path))
        payload = torch.load(io.BytesIO(snapshot.payload), map_location="cpu", weights_only=False)
    except Exception as error:
        raise ResumePreflightError(f"could not load checkpoint: {path}") from error
    state, stored_logits, optimizer_state, _ = _validated_checkpoint_payload(
        payload, expected_config=expected_config, logits=live_logits
    )
    old_logits = live_logits.detach().clone()
    old_optimizer = copy.deepcopy(adam.state_dict())
    old_rng = _capture_rng_state()
    try:
        adam.load_state_dict(optimizer_state)
        _optimizer_state_to_device(adam, live_logits.device)
        with torch.no_grad():
            live_logits.copy_(stored_logits.to(device=live_logits.device))
        _restore_rng_state(state.rng_state)
    except Exception as error:
        adam.load_state_dict(old_optimizer)
        _optimizer_state_to_device(adam, live_logits.device)
        with torch.no_grad():
            live_logits.copy_(old_logits)
        _restore_rng_state(old_rng)
        raise ResumePreflightError("checkpoint restoration failed atomically") from error
    return state


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _run_lock(run_dir: Path):
    candidate = Path(os.path.abspath(run_dir))
    if not candidate.is_dir() or candidate.is_symlink() or candidate.resolve(strict=True) != candidate:
        raise ValueError("run lock target must be a canonical non-symlink directory")
    descriptor = os.open(
        candidate,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _canonical_transaction(value: object) -> tuple[str, int]:
    if type(value) is not str:
        raise ResumePreflightError("transaction path must be a canonical string")
    match = _TRANSACTION_PATH.fullmatch(value)
    if match is None:
        raise ResumePreflightError(
            "transaction path must be canonical transactions/step_[0-9]{8}"
        )
    return value, int(match.group(1))


def _safe_transactions_directory(run_dir: Path, *, create: bool = False) -> Path:
    root = Path(os.path.abspath(run_dir))
    transactions = root / "transactions"
    if os.path.lexists(transactions):
        metadata = transactions.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ResumePreflightError("transactions directory must not be a symlink")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ResumePreflightError("transactions path must be a directory")
    elif create:
        transactions.mkdir()
        _fsync_directory(root)
    else:
        raise ResumePreflightError("transactions directory is missing")
    if transactions.resolve(strict=True).parent != root:
        raise ResumePreflightError("transactions directory resolves outside the run")
    return transactions


def _safe_generation(run_dir: Path, transaction: object) -> tuple[str, Path, int]:
    relative, completed = _canonical_transaction(transaction)
    transactions = _safe_transactions_directory(run_dir)
    generation = Path(os.path.abspath(run_dir)) / relative
    try:
        metadata = generation.lstat()
    except OSError as error:
        raise ResumePreflightError("referenced transaction generation is missing") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ResumePreflightError("transaction generation must not be a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ResumePreflightError("transaction generation must be a directory")
    if generation.resolve(strict=True).parent != transactions.resolve(strict=True):
        raise ResumePreflightError("transaction generation resolves outside the run")
    names = {path.name for path in generation.iterdir()}
    if names != set(_TRANSACTION_FILES):
        raise ResumePreflightError("transaction generation has the wrong exact payload set")
    return relative, generation, completed


def _safe_transaction_file(generation: Path, name: str) -> Path:
    if name not in _TRANSACTION_FILES:
        raise ValueError("unknown transaction payload")
    path = generation / name
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ResumePreflightError(f"transaction {name} is missing") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ResumePreflightError(f"transaction {name} must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode) or path.resolve(strict=True).parent != generation:
        raise ResumePreflightError(f"transaction {name} is not a canonical regular file")
    return path


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return canonical_json_bytes(payload) + b"\n"


def _json_snapshot(path: Path, *, root: Path, name: str) -> tuple[dict[str, object], str]:
    snapshot = snapshot_regular_file(path, root=root)
    try:
        payload = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_pairs(pairs, name=name),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResumePreflightError(f"{name} is malformed") from error
    if type(payload) is not dict or snapshot.payload != _json_bytes(payload):
        raise ResumePreflightError(f"{name} is not exact canonical JSON")
    return payload, snapshot.sha256


def _unique_pairs(
    pairs: list[tuple[str, object]], *, name: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError(f"{name} contains a duplicate or foreign key")
        result[key] = value
    return result


def _read_pointer(run_dir: Path) -> dict[str, object]:
    payload, _ = _json_snapshot(
        Path(run_dir) / "checkpoint_latest.json", root=Path(run_dir), name="checkpoint pointer"
    )
    pointer = _exact_mapping(
        payload, {"schema", "transaction", "manifest_sha256"}, name="checkpoint pointer"
    )
    if pointer["schema"] != _POINTER_SCHEMA:
        raise ResumePreflightError("checkpoint pointer schema mismatch")
    _canonical_transaction(pointer["transaction"])
    _digest(pointer["manifest_sha256"], name="pointer manifest_sha256")
    return pointer


def _history_row(value: object) -> dict[str, object]:
    expected = {"schema", *(field.name for field in fields(V4StepResult))}
    row = _exact_mapping(value, expected, name="history row")
    if row["schema"] != _HISTORY_SCHEMA:
        raise ValueError("history row schema mismatch")
    result = _result_from_mapping(
        {field.name: row[field.name] for field in fields(V4StepResult)}
    )
    return {"schema": _HISTORY_SCHEMA, **asdict(result)}


def _history_bytes(
    rows: Sequence[Mapping[str, object]],
) -> tuple[bytes, list[dict[str, object]]]:
    if type(rows) not in (list, tuple):
        raise ValueError("history segment must be a sequence")
    checked = [_history_row(dict(row)) for row in rows]
    return b"".join(_json_bytes(row) for row in checked), checked


def _read_history(path: Path, *, generation: Path) -> list[dict[str, object]]:
    snapshot = snapshot_regular_file(path, root=generation)
    if not snapshot.payload:
        return []
    rows: list[dict[str, object]] = []
    for raw in snapshot.payload.splitlines(keepends=True):
        if not raw.endswith(b"\n"):
            raise ResumePreflightError("transaction history lacks a final newline")
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=lambda pairs: _unique_pairs(pairs, name="history row"),
            )
            row = _history_row(payload)
        except Exception as error:
            raise ResumePreflightError("transaction history row is malformed") from error
        if raw != _json_bytes(row):
            raise ResumePreflightError("transaction history row is not canonical")
        rows.append(row)
    return rows


@dataclass(frozen=True)
class _VerifiedChain:
    completed_updates: int
    final_transaction: str
    final_manifest_sha256: str
    history: tuple[dict[str, object], ...]
    checkpoints: tuple[AuthenticatedTransactionCheckpoint, ...]


def _manifest(path: Path, *, generation: Path, expected_digest: str) -> dict[str, object]:
    payload, observed = _json_snapshot(path, root=generation, name="transaction manifest")
    if observed != expected_digest:
        raise ResumePreflightError("transaction manifest hash mismatch")
    required = {
        "schema",
        "config_hash",
        "completed_updates",
        "checkpoint_sha256",
        "history_segment_sha256",
        "history_interval",
        "previous_transaction",
        "previous_transaction_sha256",
    }
    manifest = _exact_mapping(payload, required, name="transaction manifest")
    if manifest["schema"] != _TRANSACTION_SCHEMA:
        raise ResumePreflightError("transaction manifest schema mismatch")
    if (
        type(manifest["completed_updates"]) is not int
        or manifest["completed_updates"] < 0
    ):
        raise ResumePreflightError(
            "transaction completed_updates must be a non-negative exact integer"
        )
    return manifest


def _validate_generation(
    run_dir: Path,
    transaction: str,
    manifest_digest: str,
    *,
    expected_config: Mapping[str, object],
    logits: torch.Tensor | None = None,
) -> tuple[dict[str, object], ResumeState, list[dict[str, object]], AuthenticatedTransactionCheckpoint]:
    relative, generation, path_completed = _safe_generation(run_dir, transaction)
    manifest_path = _safe_transaction_file(generation, "transaction_manifest.json")
    checkpoint_path = _safe_transaction_file(generation, "checkpoint.pt")
    history_path = _safe_transaction_file(generation, "history_segment.jsonl")
    manifest = _manifest(manifest_path, generation=generation, expected_digest=manifest_digest)
    expected, expected_hash = _validate_resolved_identity(dict(expected_config))
    if manifest["config_hash"] != expected_hash:
        raise ResumePreflightError("transaction configuration hash mismatch")
    checkpoint_snapshot = snapshot_regular_file(checkpoint_path, root=generation)
    history_snapshot = snapshot_regular_file(history_path, root=generation)
    if checkpoint_snapshot.sha256 != _digest(
        manifest["checkpoint_sha256"], name="checkpoint_sha256"
    ):
        raise ResumePreflightError("transaction checkpoint hash mismatch")
    if history_snapshot.sha256 != _digest(
        manifest["history_segment_sha256"], name="history_segment_sha256"
    ):
        raise ResumePreflightError("transaction history hash mismatch")
    try:
        checkpoint_payload = torch.load(
            io.BytesIO(checkpoint_snapshot.payload), map_location="cpu", weights_only=False
        )
    except Exception as error:
        raise ResumePreflightError("transaction checkpoint is unreadable") from error
    state, _, _, _ = _validated_checkpoint_payload(
        checkpoint_payload, expected_config=expected, logits=logits
    )
    if (
        state.completed_updates != path_completed
        or manifest["completed_updates"] != path_completed
    ):
        raise ResumePreflightError("transaction path/checkpoint step mismatch")
    rows = _read_history(history_path, generation=generation)
    interval = _exact_mapping(
        manifest["history_interval"],
        {"first_completed_update", "last_completed_update", "row_count"},
        name="history interval",
    )
    if type(interval["row_count"]) is not int or interval["row_count"] != len(rows):
        raise ResumePreflightError("transaction history row count mismatch")
    if path_completed == 0:
        if rows or interval != {
            "first_completed_update": None,
            "last_completed_update": None,
            "row_count": 0,
        }:
            raise ResumePreflightError("step-zero transaction interval must be empty")
    else:
        if any(
            type(interval[key]) is not int or interval[key] <= 0
            for key in ("first_completed_update", "last_completed_update")
        ):
            raise ResumePreflightError(
                "transaction history interval endpoints must be positive exact integers"
            )
        if not rows:
            raise ResumePreflightError("positive transaction history is empty")
        observed = [row["completed_updates_after"] for row in rows]
        if observed != list(range(observed[0], path_completed + 1)):
            raise ResumePreflightError("transaction history interval is not contiguous")
        if interval != {
            "first_completed_update": observed[0],
            "last_completed_update": observed[-1],
            "row_count": len(rows),
        }:
            raise ResumePreflightError("transaction history interval metadata mismatch")
        if state.last_result is None or asdict(state.last_result) != {
            field.name: rows[-1][field.name] for field in fields(V4StepResult)
        }:
            raise ResumePreflightError("checkpoint result and history tail mismatch")
    identity = AuthenticatedTransactionCheckpoint(
        completed_updates=path_completed,
        transaction=relative,
        transaction_manifest_sha256=manifest_digest,
        checkpoint_path=f"{relative}/checkpoint.pt",
        checkpoint_sha256=checkpoint_snapshot.sha256,
    )
    return manifest, state, rows, identity


def _verify_transaction_chain(
    run_dir: Path,
    *,
    expected_config: Mapping[str, object],
    expected_final_step: int | None = None,
    logits: torch.Tensor | None = None,
) -> _VerifiedChain:
    root = Path(os.path.abspath(run_dir))
    if not root.is_dir() or root.is_symlink() or root.resolve(strict=True) != root:
        raise ResumePreflightError("run directory is not canonical")
    expected, _ = _validate_resolved_identity(dict(expected_config))
    pointer = _read_pointer(root)
    transaction = pointer["transaction"]
    manifest_digest = pointer["manifest_sha256"]
    reverse: list[
        tuple[str, str, dict[str, object], ResumeState, list[dict[str, object]], AuthenticatedTransactionCheckpoint]
    ] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    while True:
        relative, _ = _canonical_transaction(transaction)
        digest = _digest(manifest_digest, name="transaction manifest digest")
        if relative in seen_paths or digest in seen_hashes:
            raise ResumePreflightError("transaction predecessor chain contains a cycle")
        seen_paths.add(relative)
        seen_hashes.add(digest)
        manifest, state, rows, identity = _validate_generation(
            root,
            relative,
            digest,
            expected_config=expected,
            logits=logits,
        )
        reverse.append((relative, digest, manifest, state, rows, identity))
        previous = manifest["previous_transaction"]
        previous_digest = manifest["previous_transaction_sha256"]
        if previous is None or previous_digest is None:
            if previous is not None or previous_digest is not None:
                raise ResumePreflightError("transaction predecessor identity is incomplete")
            break
        transaction = previous
        manifest_digest = previous_digest
    chain = list(reversed(reverse))
    if not chain or chain[0][3].completed_updates != 0 or chain[0][0] != "transactions/step_00000000":
        raise ResumePreflightError("transaction chain does not begin at step zero")
    history: list[dict[str, object]] = []
    previous_completed = 0
    for index, (relative, digest, manifest, state, rows, _) in enumerate(chain):
        if index == 0:
            if manifest["previous_transaction"] is not None or manifest["previous_transaction_sha256"] is not None:
                raise ResumePreflightError("step-zero transaction has a predecessor")
        else:
            prior_relative, prior_digest, _, _, _, _ = chain[index - 1]
            if (
                manifest["previous_transaction"] != prior_relative
                or manifest["previous_transaction_sha256"] != prior_digest
                or not rows
                or rows[0]["completed_updates_after"] != previous_completed + 1
            ):
                raise ResumePreflightError("transaction predecessor/history identity mismatch")
        history.extend(rows)
        previous_completed = state.completed_updates
    final = chain[-1]
    completed = final[3].completed_updates
    if completed != len(history):
        raise ResumePreflightError("transaction history length mismatches final checkpoint")
    if expected_final_step is not None and completed != expected_final_step:
        raise ResumePreflightError("transaction chain is not at the required final step")
    interval = expected["checkpoint_policy"]["commit_interval"]
    if (
        completed != 0
        and completed % interval != 0
        and completed != expected["steps"]
    ):
        raise ResumePreflightError(
            "transaction generation violates the frozen checkpoint cadence"
        )
    expected_generations = list(range(0, completed + 1, interval))
    if expected_generations[-1] != completed:
        expected_generations.append(completed)
    observed_generations = [item[3].completed_updates for item in chain]
    if observed_generations != expected_generations:
        raise ResumePreflightError(
            "transaction generation schedule is incomplete"
        )
    return _VerifiedChain(
        completed_updates=completed,
        final_transaction=final[0],
        final_manifest_sha256=final[1],
        history=tuple(history),
        checkpoints=tuple(item[5] for item in chain),
    )


def _before_pointer_publication_hook(run_dir: Path, completed_updates: int) -> None:
    del run_dir, completed_updates


def commit_training_transaction(
    run_dir: Path,
    *,
    checkpoint_payload: Mapping[str, object],
    history_segment: Sequence[Mapping[str, object]],
    previous_transaction_sha256: str | None,
) -> Path:
    root = Path(os.path.abspath(run_dir))
    if not root.is_dir() or root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("run_dir must be a canonical existing directory")
    state, _, _, config = _validated_checkpoint_payload(checkpoint_payload)
    rows_bytes, rows = _history_bytes(history_segment)
    completed = state.completed_updates
    interval = config["checkpoint_policy"]["commit_interval"]
    if (
        completed != 0
        and completed % interval != 0
        and completed != config["steps"]
    ):
        raise ValueError("transaction generation violates the frozen checkpoint cadence")
    config_digest = canonical_hash(config)
    with _run_lock(root):
        if completed == 0:
            if rows or previous_transaction_sha256 is not None or os.path.lexists(root / "checkpoint_latest.json"):
                raise ValueError("step-zero transaction requires no history/predecessor/pointer")
            previous_transaction = None
        else:
            if not rows:
                raise ValueError("positive transaction requires history")
            predecessor_digest = _digest(
                previous_transaction_sha256, name="previous_transaction_sha256"
            )
            pointer = _read_pointer(root)
            if pointer["manifest_sha256"] != predecessor_digest:
                raise ValueError("transaction predecessor is not the current pointer")
            previous_transaction = pointer["transaction"]
            previous_relative, previous_generation, previous_completed = _safe_generation(
                root, previous_transaction
            )
            expected_completed = min(previous_completed + interval, config["steps"])
            if completed != expected_completed:
                raise ValueError(
                    "transaction generation skips the frozen checkpoint interval"
                )
            previous_manifest = _manifest(
                _safe_transaction_file(previous_generation, "transaction_manifest.json"),
                generation=previous_generation,
                expected_digest=predecessor_digest,
            )
            if previous_manifest["config_hash"] != config_digest:
                raise ValueError("transaction predecessor configuration mismatch")
            observed = [row["completed_updates_after"] for row in rows]
            if observed != list(range(previous_completed + 1, completed + 1)):
                raise ValueError("transaction history interval is not contiguous")
            if state.last_result is None or asdict(state.last_result) != {
                field.name: rows[-1][field.name] for field in fields(V4StepResult)
            }:
                raise ValueError("checkpoint result does not match history tail")
            previous_transaction = previous_relative
        transactions = _safe_transactions_directory(root, create=True)
        name = f"step_{completed:08d}"
        generation = transactions / name
        if os.path.lexists(generation):
            raise FileExistsError(f"transaction generation already exists: {generation}")
        temporary = transactions / f".{name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        temporary.mkdir()
        renamed = False
        try:
            atomic_checkpoint(temporary / "checkpoint.pt", checkpoint_payload)
            atomic_text(temporary / "history_segment.jsonl", rows_bytes.decode("utf-8"))
            interval = {
                "first_completed_update": None if not rows else rows[0]["completed_updates_after"],
                "last_completed_update": None if not rows else rows[-1]["completed_updates_after"],
                "row_count": len(rows),
            }
            manifest = {
                "schema": _TRANSACTION_SCHEMA,
                "config_hash": config_digest,
                "completed_updates": completed,
                "checkpoint_sha256": sha256_file(temporary / "checkpoint.pt"),
                "history_segment_sha256": sha256_file(temporary / "history_segment.jsonl"),
                "history_interval": interval,
                "previous_transaction": previous_transaction,
                "previous_transaction_sha256": previous_transaction_sha256,
            }
            atomic_json(temporary / "transaction_manifest.json", manifest)
            manifest_digest = sha256_file(temporary / "transaction_manifest.json")
            _fsync_directory(temporary)
            os.rename(temporary, generation)
            renamed = True
            _fsync_directory(transactions)
            _before_pointer_publication_hook(root, completed)
            revalidate_resolved_dependencies(config)
            if completed > 0:
                _verify_transaction_chain(
                    root,
                    expected_config=config,
                    expected_final_step=previous_completed,
                )
            _validate_generation(
                root,
                f"transactions/{name}",
                manifest_digest,
                expected_config=config,
            )
            atomic_json(
                root / "checkpoint_latest.json",
                {
                    "schema": _POINTER_SCHEMA,
                    "transaction": f"transactions/{name}",
                    "manifest_sha256": manifest_digest,
                },
            )
        finally:
            if not renamed and temporary.exists():
                shutil.rmtree(temporary)
        return generation


def restore_latest_transaction(
    run_dir: Path,
    *,
    expected_config: Mapping[str, object],
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> ResumeState:
    live_logits, adam = _validate_live_state(logits, optimizer)
    config, _ = _validate_resolved_identity(dict(expected_config))
    chain = _verify_transaction_chain(
        Path(run_dir), expected_config=config, logits=live_logits
    )
    generation = Path(run_dir) / chain.final_transaction
    restored = restore_checkpoint_payload(
        _safe_transaction_file(generation, "checkpoint.pt"),
        expected_config=config,
        logits=live_logits,
        optimizer=adam,
    )
    return ResumeState(
        completed_updates=restored.completed_updates,
        last_result=restored.last_result,
        rng_state=restored.rng_state,
        step_times=restored.step_times,
        peak_cuda_bytes=restored.peak_cuda_bytes,
        slot_audits=restored.slot_audits,
        committed_transaction=chain.final_transaction,
        committed_transaction_sha256=chain.final_manifest_sha256,
        committed_history=chain.history,
    )


def _timer() -> float:
    return float(time.perf_counter())


def _post_update_hook(completed_updates: int) -> None:
    del completed_updates


def _before_artifact_publication_hook(run_dir: Path) -> None:
    del run_dir


def _after_artifact_publication_hook(run_dir: Path) -> None:
    del run_dir


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _decode_shadow(snapshot: BoundFileSnapshot, size: int) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.frombuffer(snapshot.payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"shadow snapshot is not a decodable grayscale image: {snapshot.path}")
    return (
        normalize_target(image, size, False),
        normalize_target(cv2.flip(image, 1), size, False),
    )


def _build_runtime(resolved: ResolvedV4Run, device: torch.device, profile: object) -> _Runtime:
    if not isinstance(resolved, ResolvedV4Run):
        raise ValueError("runtime requires a resolved v4 run")
    registered = resolve_profile(profile.name if hasattr(profile, "name") else profile)
    images = [_decode_shadow(snapshot, 256) for snapshot in resolved.shadow_snapshots]
    ordered = [pair[0] for pair in images] + [pair[1] for pair in images]
    targets = torch.from_numpy(np.stack(ordered).astype(np.float32) / 255.0).to(device)
    protocol = SadekarTFSProtocol(
        device=device,
        image_size=resolved.config["protocol"]["image_size"],
        zdist=resolved.config["protocol"]["zdist"],
        volume_extent_world=resolved.config["protocol"]["volume_extent"],
        n_pts_per_ray=resolved.config["protocol"]["samples_per_ray"],
    )
    resolution = resolved.config["resolution"]
    logits = torch.full(
        (1, 1, resolution, resolution, resolution),
        -4.0,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([logits], lr=resolved.config["learning_rate"])
    bands = make_contour_bands(targets >= 0.5, radius_px=2.0) if registered.use_dic else None
    return _Runtime(logits, optimizer, protocol, targets, bands)


def _publish_bytes_once(path: Path, payload: bytes) -> None:
    destination = Path(path)
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise ValueError("artifact parent must be a canonical directory")
    if destination.parent.resolve(strict=True) != destination.parent:
        raise ValueError("artifact parent contains a symlink")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError:
        snapshot = snapshot_regular_file(destination, root=destination.parent)
        if snapshot.payload != payload:
            raise FileExistsError(f"immutable artifact differs: {destination}")
        return
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(destination.parent)


def _status_payload(
    state: str,
    config_hash: str,
    completed_updates: int,
    *,
    artifact_manifest_sha256: str | None = None,
) -> dict[str, object]:
    common: dict[str, object] = {
        "schema": "pcraft_soft_loss_v4_status_v1",
        "state": state,
        "config_hash": config_hash,
        "completed_updates": completed_updates,
    }
    if state == "completed":
        common["artifact_manifest_sha256"] = _digest(
            artifact_manifest_sha256, name="artifact manifest SHA"
        )
    elif state == "failed":
        common["diagnostic"] = "failure.json"
    elif state != "running":
        raise ValueError("unknown lifecycle state")
    return common


def _read_status(run_dir: Path) -> tuple[dict[str, object], str]:
    payload, digest = _json_snapshot(
        Path(run_dir) / "STATUS.json", root=Path(run_dir), name="STATUS"
    )
    common = {"schema", "state", "config_hash", "completed_updates"}
    state = payload.get("state")
    expected = (
        common
        if state == "running"
        else common | {"artifact_manifest_sha256"}
        if state == "completed"
        else common | {"diagnostic"}
        if state == "failed"
        else set()
    )
    _exact_mapping(payload, expected, name="STATUS")
    if payload["schema"] != "pcraft_soft_loss_v4_status_v1":
        raise ValueError("STATUS schema mismatch")
    _digest(payload["config_hash"], name="STATUS config hash")
    if type(payload["completed_updates"]) is not int or payload["completed_updates"] < 0:
        raise ValueError("STATUS completed update is malformed")
    if state == "running" and payload["completed_updates"] != 0:
        raise ValueError("running STATUS must remain at update zero")
    if state == "failed" and payload["diagnostic"] != "failure.json":
        raise ValueError("failed STATUS diagnostic drifted")
    if state == "completed":
        _digest(payload["artifact_manifest_sha256"], name="STATUS artifact manifest SHA")
    return payload, digest


def _command_payload(args: argparse.Namespace, config: Mapping[str, object]) -> dict[str, object]:
    checked = validate_resolved_config(dict(config))
    return {
        "schema": "pcraft_soft_loss_v4_command_v1",
        "fresh_argv": list(canonical_trainer_argv(args, resume=False)),
        "resume_argv": list(canonical_trainer_argv(args, resume=True)),
        "resolved_config": checked,
        "config_hash": canonical_hash(checked),
    }


def _run_manifest_payload(config: Mapping[str, object]) -> dict[str, object]:
    checked = validate_resolved_config(dict(config))
    return {
        "schema": "pcraft_soft_loss_v4_run_v1",
        "config_hash": canonical_hash(checked),
        "config": checked,
        "run_identity": checked["run_identity"],
        "profile": checked["profile"],
        "protocol": checked["protocol"],
        "scope": {
            "soft_only": True,
            "final_step_export_only": True,
            "downstream_geometry": False,
        },
        "checkpoint_policy": checked["checkpoint_policy"],
    }


def _environment_payload(config: Mapping[str, object]) -> dict[str, object]:
    device = torch.device(config["device"])
    payload = environment_manifest(device)
    if payload["interpreter"] != config["interpreter"]:
        raise ValueError("interpreter identity changed after configuration")
    if payload["environment"] != config["determinism"]["environment"]:
        raise ValueError("allowlisted environment changed after configuration")
    return payload


def _authenticate_static_run_evidence(
    root: Path, config: Mapping[str, object]
) -> tuple[str, str]:
    """Authenticate immutable run/environment manifests and return their digests."""

    run_manifest, run_manifest_digest = _json_snapshot(
        root / "run_manifest.json", root=root, name="run manifest"
    )
    if canonical_json_bytes(run_manifest) != canonical_json_bytes(
        _run_manifest_payload(config)
    ):
        raise ValueError("run manifest identity mismatch")
    environment, environment_digest = _json_snapshot(
        root / "environment.json", root=root, name="environment"
    )
    if canonical_json_bytes(environment) != canonical_json_bytes(
        _environment_payload(config)
    ):
        raise ValueError("environment manifest identity mismatch")
    return run_manifest_digest, environment_digest


_COMMON_ARTIFACTS = (
    "COMMAND.json",
    "run_manifest.json",
    "environment.json",
    "history.jsonl",
    "occupancy_probability.npy",
    "occupancy_binary_tau_0p04.npy",
    "occupancy_binary_tau_0p05.npy",
    "occupancy_binary_tau_0p06.npy",
    *(f"binary_tau_{label}_view_{view}.png" for label in ("0p04", "0p05", "0p06") for view in range(6)),
    "metrics.json",
    "gradient_audit.json",
    "resource_metrics.json",
    "checkpoint_latest.json",
    "checkpoint_final.pt",
)


def _retained_names(config: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        f"checkpoint_step_{update:08d}.pt"
        for update in config["checkpoint_policy"]["retained_updates"]
    )


def _expected_artifact_names(
    config: Mapping[str, object], chain: _VerifiedChain
) -> tuple[str, ...]:
    transaction_names = tuple(
        f"{item.transaction}/{name}"
        for item in chain.checkpoints
        for name in _TRANSACTION_FILES
    )
    return tuple(sorted((*_COMMON_ARTIFACTS, *_retained_names(config), *transaction_names)))


def _preflight_reserved(run_dir: Path) -> None:
    reserved = {
        *_COMMON_ARTIFACTS,
        "artifact_manifest.json",
        "STATUS.json",
        "failure.json",
        "checkpoint_step_00001600.pt",
        "checkpoint_step_00001800.pt",
        "transactions",
    }
    for name in reserved:
        path = Path(run_dir) / name
        if not os.path.lexists(path):
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"reserved path must not be a symlink: {name}")


def _load_command(run_dir: Path) -> dict[str, object]:
    payload, _ = _json_snapshot(
        Path(run_dir) / "COMMAND.json", root=Path(run_dir), name="COMMAND"
    )
    command = _exact_mapping(
        payload,
        {"schema", "fresh_argv", "resume_argv", "resolved_config", "config_hash"},
        name="COMMAND",
    )
    if command["schema"] != "pcraft_soft_loss_v4_command_v1":
        raise ValueError("COMMAND schema mismatch")
    config = validate_resolved_config(command["resolved_config"])
    if command["config_hash"] != canonical_hash(config):
        raise ValueError("COMMAND configuration hash mismatch")
    for key in ("fresh_argv", "resume_argv"):
        if type(command[key]) is not list or any(type(item) is not str for item in command[key]):
            raise ValueError("COMMAND argv is malformed")
    base = {
        "run_kind": config["run_kind"],
        "device": config["device"],
        "output_root": config["output_root_identity"]["path"],
        "run_id": config["run_identity"]["run_id"],
        "stage": config["run_identity"]["stage"],
        "task_id": config["run_identity"]["task_id"],
        "replicate_role": config["run_identity"]["replicate_role"],
        "replicate_index": config["run_identity"]["replicate_index"],
        "steps": config["steps"],
        "learning_rate": config["learning_rate"],
        "shadow_files": [item["path"] for item in config["shadow_inputs_xyz"]],
        "resolution": config["resolution"],
        "export_resolution": config["export_resolution"],
        "profile": config["profile"]["name"],
        "suite_manifest": config["suite_manifest"]["path"],
        "specification": config["specification"]["path"],
        "unit_coefficients": None
        if config["unit_coefficient_identity"] is None
        else config["unit_coefficient_identity"]["path"],
        "candidate_registry": None
        if config["candidate_registry_identity"] is None
        else config["candidate_registry_identity"]["path"],
        "candidate_id": None
        if config["budget_identity"] is None
        else config["budget_identity"]["candidate_id"],
        "export_final_step": True,
        "soft_only_eval": True,
    }
    for key, resume in (("fresh_argv", False), ("resume_argv", True)):
        args = argparse.Namespace(**base, resume=resume)
        if command[key] != list(canonical_trainer_argv(args, resume=resume)):
            raise ValueError(f"COMMAND {key} differs from canonical resolved identity")
    return command


def _summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "median_seconds": None, "p95_seconds": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.percentile(array, 95)),
    }


def _gradient_audit_payload(
    config: Mapping[str, object], audits: Sequence[AuxiliarySlotAudit]
) -> dict[str, object]:
    profile = resolve_profile(config["profile"]["name"])
    samples = [asdict(audit) for audit in audits]
    if profile.name == "l12_bal":
        if samples:
            raise ValueError("base/calibration runs cannot contain auxiliary audits")
        engineering = None
        redundancy = None
    else:
        checked = _validate_slot_audit_prefix(
            audits,
            profile=profile,
            total_steps=config["steps"],
            completed_updates=config["steps"],
        )
        engineering = asdict(
            validate_gradient_engineering_gate(
                checked, profile=profile, total_steps=config["steps"]
            )
        )
        redundancy = (
            asdict(
                validate_redundancy_gate(
                    checked, profile=profile, total_steps=config["steps"]
                )
            )
            if profile.name in _FULL_PROFILES
            else None
        )
    return {
        "schema": "pcraft_soft_loss_v4_gradient_audit_v1",
        "config_hash": canonical_hash(config),
        "profile": profile.name,
        "total_steps": config["steps"],
        "step_config": config["step_config"],
        "eligible_slots": len(samples),
        "samples": samples,
        "engineering_gate": engineering,
        "redundancy_gate": redundancy,
    }


def _resource_payload(
    config: Mapping[str, object], step_times: Sequence[float], peak_cuda_bytes: int
) -> dict[str, object]:
    times = _validated_times(step_times, completed_updates=config["steps"])
    physical = None
    device = torch.device(config["device"])
    if device.type == "cuda":
        physical = int(torch.cuda.get_device_properties(device).total_memory)
    return {
        "schema": "pcraft_soft_loss_v4_resources_v1",
        "config_hash": canonical_hash(config),
        "step_times_seconds": list(times),
        "peak_cuda_allocated_bytes": peak_cuda_bytes,
        "physical_cuda_total_bytes": physical,
        "all_updates": _summary(times),
        "updates_21_to_end": None if len(times) < 21 else _summary(times[20:]),
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _evaluate_and_publish(
    run_dir: Path,
    *,
    config: Mapping[str, object],
    runtime: _Runtime,
) -> dict[str, object]:
    with torch.no_grad():
        probability = torch.sigmoid(runtime.logits).detach()
        masks: dict[float, torch.Tensor] = {}
        for threshold in _THRESHOLDS:
            hard = probability >= threshold
            projection = runtime.protocol.render_all(hard.to(probability.dtype) * 0.999)
            masks[threshold] = runtime.protocol.binary_mask(projection).to(dtype=torch.bool)
    _publish_bytes_once(
        run_dir / "occupancy_probability.npy",
        _npy_bytes(probability.to(device="cpu").numpy()),
    )
    for threshold, label in zip(_THRESHOLDS, ("0p04", "0p05", "0p06")):
        hard_array = (probability >= threshold).to(device="cpu").numpy()
        _publish_bytes_once(
            run_dir / f"occupancy_binary_tau_{label}.npy", _npy_bytes(hard_array)
        )
        for view, mask in enumerate(masks[threshold]):
            pixels = mask.to(device="cpu").numpy().astype(np.uint8) * 255
            success, encoded = cv2.imencode(".png", pixels)
            if not success:
                raise RuntimeError("PNG encoding failed")
            _publish_bytes_once(
                run_dir / f"binary_tau_{label}_view_{view}.png", encoded.tobytes()
            )
    report = evaluate_delivery_metrics(runtime.targets, masks)
    metrics: dict[str, object] = {
        "schema": "pcraft_soft_loss_v4_metrics_v1",
        "config_hash": canonical_hash(config),
        **report,
    }
    publish_json_once(run_dir / "metrics.json", metrics)
    return metrics


def _publish_failure(
    run_dir: Path,
    *,
    config_hash: str,
    completed_updates: int,
    error: Exception,
) -> None:
    diagnostic = {
        "schema": "pcraft_soft_loss_v4_failure_v1",
        "config_hash": config_hash,
        "completed_updates": completed_updates,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    with _run_lock(run_dir):
        status, _ = _read_status(run_dir)
        if status["state"] != "running":
            raise ValueError("only a running run may publish failure evidence")
        publish_json_once(run_dir / "failure.json", diagnostic)
        atomic_json(
            run_dir / "STATUS.json",
            _status_payload("failed", config_hash, completed_updates),
        )


def _materialize_final(
    run_dir: Path,
    *,
    config: Mapping[str, object],
    runtime: _Runtime,
    step_times: Sequence[float],
    peak_cuda_bytes: int,
    audits: Sequence[AuxiliarySlotAudit],
) -> dict[str, object]:
    chain = _verify_transaction_chain(
        run_dir, expected_config=config, expected_final_step=config["steps"]
    )
    history_bytes = b"".join(_json_bytes(row) for row in chain.history)
    _publish_bytes_once(run_dir / "history.jsonl", history_bytes)
    final_generation = run_dir / chain.final_transaction
    final_checkpoint = snapshot_regular_file(
        final_generation / "checkpoint.pt", root=final_generation
    )
    _publish_bytes_once(run_dir / "checkpoint_final.pt", final_checkpoint.payload)
    by_step = {item.completed_updates: item for item in chain.checkpoints}
    for update in config["checkpoint_policy"]["retained_updates"]:
        identity = by_step.get(update)
        if identity is None:
            raise ValueError("retained checkpoint is not reachable")
        source = snapshot_regular_file(run_dir / identity.checkpoint_path, root=run_dir)
        _publish_bytes_once(run_dir / f"checkpoint_step_{update:08d}.pt", source.payload)
    metrics = _evaluate_and_publish(run_dir, config=config, runtime=runtime)
    publish_json_once(run_dir / "gradient_audit.json", _gradient_audit_payload(config, audits))
    publish_json_once(
        run_dir / "resource_metrics.json",
        _resource_payload(config, step_times, peak_cuda_bytes),
    )
    _before_artifact_publication_hook(run_dir)
    revalidate_resolved_dependencies(config)
    chain = _verify_transaction_chain(
        run_dir, expected_config=config, expected_final_step=config["steps"]
    )
    artifacts: dict[str, str] = {}
    for relative in _expected_artifact_names(config, chain):
        artifacts[relative] = snapshot_regular_file(
            run_dir / relative, root=run_dir
        ).sha256
    manifest = {
        "schema": "pcraft_soft_loss_v4_artifacts_v1",
        "config_hash": canonical_hash(config),
        "artifacts_sha256": artifacts,
    }
    publish_json_once(run_dir / "artifact_manifest.json", manifest)
    revalidate_resolved_dependencies(config)
    _verify_transaction_chain(
        run_dir, expected_config=config, expected_final_step=config["steps"]
    )
    _after_artifact_publication_hook(run_dir)
    revalidate_resolved_dependencies(config)
    _verify_transaction_chain(
        run_dir, expected_config=config, expected_final_step=config["steps"]
    )
    manifest_snapshot = snapshot_regular_file(run_dir / "artifact_manifest.json", root=run_dir)
    for relative, digest in artifacts.items():
        if snapshot_regular_file(run_dir / relative, root=run_dir).sha256 != digest:
            raise ValueError("artifact changed before terminal STATUS")
    with _run_lock(run_dir):
        status, _ = _read_status(run_dir)
        if status["state"] != "running":
            raise ValueError("terminal STATUS compare-and-swap failed")
        atomic_json(
            run_dir / "STATUS.json",
            _status_payload(
                "completed",
                canonical_hash(config),
                config["steps"],
                artifact_manifest_sha256=manifest_snapshot.sha256,
            ),
        )
    return metrics


def _artifact_manifest(run_dir: Path) -> tuple[dict[str, object], str]:
    payload, digest = _json_snapshot(
        run_dir / "artifact_manifest.json", root=run_dir, name="artifact manifest"
    )
    manifest = _exact_mapping(
        payload, {"schema", "config_hash", "artifacts_sha256"}, name="artifact manifest"
    )
    if manifest["schema"] != "pcraft_soft_loss_v4_artifacts_v1":
        raise ValueError("artifact manifest schema mismatch")
    if type(manifest["artifacts_sha256"]) is not dict or any(
        type(key) is not str or _DIGEST.fullmatch(value) is None
        for key, value in manifest["artifacts_sha256"].items()
    ):
        raise ValueError("artifact hash mapping is malformed")
    return manifest, digest


def _validate_exact_namespace(
    run_dir: Path, chain: _VerifiedChain, artifacts: Mapping[str, object]
) -> None:
    expected_files = set(artifacts) | {"artifact_manifest.json", "STATUS.json"}
    expected_top_files = {
        name for name in expected_files if "/" not in name
    }
    entries = {path.name: path for path in run_dir.iterdir()}
    if set(entries) != expected_top_files | {"transactions"}:
        raise ValueError("completed artifact namespace has unexpected or missing paths")
    for name in expected_top_files:
        path = entries[name]
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or path.resolve(strict=True).parent != run_dir
        ):
            raise ValueError(
                f"completed artifact is not a canonical regular file: {name}"
            )
    transactions = entries["transactions"]
    metadata = transactions.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or transactions.resolve(strict=True).parent != run_dir
    ):
        raise ValueError("completed transactions path is not a canonical directory")
    transaction_names = {Path(item.transaction).name for item in chain.checkpoints}
    transaction_entries = {path.name: path for path in transactions.iterdir()}
    if set(transaction_entries) != transaction_names:
        raise ValueError("completed transaction namespace contains an orphan")
    for name in transaction_names:
        path = transaction_entries[name]
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("completed transaction generation is not a directory")


def _authenticate_terminal_artifacts(
    root: Path, config: Mapping[str, object]
) -> tuple[_VerifiedChain, dict[str, object], str, dict[str, object]]:
    """Authenticate final evidence without relying on terminal STATUS state."""

    config_hash = canonical_hash(config)
    revalidate_resolved_dependencies(config)
    _authenticate_static_run_evidence(root, config)
    chain = _verify_transaction_chain(
        root, expected_config=config, expected_final_step=config["steps"]
    )
    manifest, manifest_digest = _artifact_manifest(root)
    if (
        manifest["config_hash"] != config_hash
        or set(manifest["artifacts_sha256"])
        != set(_expected_artifact_names(config, chain))
    ):
        raise ValueError("artifact manifest identity/set mismatch")
    _validate_exact_namespace(root, chain, manifest["artifacts_sha256"])
    for relative, digest in manifest["artifacts_sha256"].items():
        if snapshot_regular_file(root / relative, root=root).sha256 != digest:
            raise ValueError(f"artifact hash mismatch: {relative}")
    expected_history = b"".join(_json_bytes(row) for row in chain.history)
    if snapshot_regular_file(root / "history.jsonl", root=root).payload != expected_history:
        raise ValueError("materialized history differs from the transaction chain")
    final_identity = chain.checkpoints[-1]
    if (
        snapshot_regular_file(root / "checkpoint_final.pt", root=root).sha256
        != final_identity.checkpoint_sha256
    ):
        raise ValueError("final checkpoint differs from the final transaction")
    by_step = {item.completed_updates: item for item in chain.checkpoints}
    for update in config["checkpoint_policy"]["retained_updates"]:
        retained = snapshot_regular_file(
            root / f"checkpoint_step_{update:08d}.pt", root=root
        )
        if update not in by_step or retained.sha256 != by_step[update].checkpoint_sha256:
            raise ValueError("retained checkpoint differs from its transaction")
    _, final_state, _, _ = _validate_generation(
        root,
        final_identity.transaction,
        final_identity.transaction_manifest_sha256,
        expected_config=config,
    )
    gradient_audit, _ = _json_snapshot(
        root / "gradient_audit.json", root=root, name="gradient audit"
    )
    if canonical_json_bytes(gradient_audit) != canonical_json_bytes(
        _gradient_audit_payload(config, final_state.slot_audits)
    ):
        raise ValueError("gradient audit differs from checkpoint evidence")
    resources, _ = _json_snapshot(
        root / "resource_metrics.json", root=root, name="resource metrics"
    )
    if canonical_json_bytes(resources) != canonical_json_bytes(
        _resource_payload(config, final_state.step_times, final_state.peak_cuda_bytes)
    ):
        raise ValueError("resource metrics differ from checkpoint evidence")
    try:
        probability = np.load(
            io.BytesIO(
                snapshot_regular_file(
                    root / "occupancy_probability.npy", root=root
                ).payload
            ),
            allow_pickle=False,
        )
    except Exception as error:
        raise ValueError("occupancy probability artifact is malformed") from error
    if (
        not isinstance(probability, np.ndarray)
        or probability.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or probability.ndim != 5
        or probability.shape[:2] != (1, 1)
        or probability.shape[2:] != (probability.shape[2],) * 3
        or not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
    ):
        raise ValueError("occupancy probability artifact has invalid semantics")
    for threshold, label in zip(_THRESHOLDS, ("0p04", "0p05", "0p06")):
        try:
            hard = np.load(
                io.BytesIO(
                    snapshot_regular_file(
                        root / f"occupancy_binary_tau_{label}.npy", root=root
                    ).payload
                ),
                allow_pickle=False,
            )
        except Exception as error:
            raise ValueError("hard occupancy artifact is malformed") from error
        if (
            hard.dtype != np.bool_
            or hard.shape != probability.shape
            or not np.array_equal(hard, probability >= threshold)
        ):
            raise ValueError("hard occupancy artifact disagrees with probability")
        for view in range(6):
            png = snapshot_regular_file(
                root / f"binary_tau_{label}_view_{view}.png", root=root
            )
            decoded = cv2.imdecode(
                np.frombuffer(png.payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
            )
            if (
                decoded is None
                or decoded.ndim != 2
                or decoded.size == 0
                or not set(np.unique(decoded)).issubset({0, 255})
            ):
                raise ValueError("binary PNG artifact is malformed")
    metrics, _ = _json_snapshot(root / "metrics.json", root=root, name="metrics")
    if (
        set(metrics)
        != {
            "schema",
            "config_hash",
            "thresholds",
            "task_iou_by_threshold",
            "worst_threshold_iou",
            "threshold_iou_range",
            "task_mscd_px",
            "task_p95_symmetric_distance_px",
            "invalid_task_metric",
        }
        or metrics["schema"] != "pcraft_soft_loss_v4_metrics_v1"
        or metrics["config_hash"] != config_hash
    ):
        raise ValueError("metrics schema/config identity mismatch")
    revalidate_resolved_dependencies(config)
    _verify_transaction_chain(
        root, expected_config=config, expected_final_step=config["steps"]
    )
    return chain, manifest, manifest_digest, metrics


def _pending_terminalization_kind(root: Path) -> str | None:
    has_completed = os.path.lexists(root / "artifact_manifest.json")
    has_failed = os.path.lexists(root / "failure.json")
    if has_completed and has_failed:
        raise ValueError("running run contains both completed and failed pending evidence")
    if has_completed:
        return "completed"
    if has_failed:
        return "failed"
    return None


def _validate_resumable_namespace(
    root: Path, chain: _VerifiedChain, *, pending: str | None
) -> None:
    expected_files = {
        "COMMAND.json",
        "run_manifest.json",
        "environment.json",
        "STATUS.json",
        "checkpoint_latest.json",
    }
    if pending == "failed":
        expected_files.add("failure.json")
    elif pending is not None:
        raise ValueError("completed pending evidence uses the terminal artifact namespace")
    entries = {path.name: path for path in root.iterdir()}
    if set(entries) != expected_files | {"transactions"}:
        raise ValueError("resumable run namespace has unexpected or missing paths")
    for name in expected_files:
        path = entries[name]
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or path.resolve(strict=True).parent != root
        ):
            raise ValueError(f"resumable evidence is not a canonical regular file: {name}")
    transactions = _safe_transactions_directory(root)
    expected_generations = {Path(item.transaction).name for item in chain.checkpoints}
    transaction_entries = {path.name: path for path in transactions.iterdir()}
    if set(transaction_entries) != expected_generations:
        raise ValueError("resumable transaction namespace contains an orphan")
    for name in expected_generations:
        metadata = transaction_entries[name].lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("resumable transaction generation is not a directory")


def _authenticate_running_chain(
    root: Path, config: Mapping[str, object]
) -> _VerifiedChain:
    revalidate_resolved_dependencies(config)
    static_identity = _authenticate_static_run_evidence(root, config)
    chain = _verify_transaction_chain(root, expected_config=config)
    _validate_resumable_namespace(root, chain, pending=None)
    revalidate_resolved_dependencies(config)
    confirmed_static_identity = _authenticate_static_run_evidence(root, config)
    confirmed_chain = _verify_transaction_chain(root, expected_config=config)
    _validate_resumable_namespace(root, confirmed_chain, pending=None)
    if static_identity != confirmed_static_identity or chain != confirmed_chain:
        raise ValueError("resumable evidence changed during authentication")
    return confirmed_chain


def _authenticate_pending_failure(
    root: Path, config: Mapping[str, object]
) -> tuple[_VerifiedChain, dict[str, object], str]:
    revalidate_resolved_dependencies(config)
    static_identity = _authenticate_static_run_evidence(root, config)
    chain = _verify_transaction_chain(root, expected_config=config)
    failure, failure_digest = _json_snapshot(
        root / "failure.json", root=root, name="failure"
    )
    _exact_mapping(
        failure,
        {"schema", "config_hash", "completed_updates", "error_type", "error"},
        name="failure",
    )
    if (
        failure["schema"] != "pcraft_soft_loss_v4_failure_v1"
        or failure["config_hash"] != canonical_hash(config)
        or type(failure["completed_updates"]) is not int
        or failure["completed_updates"] < 0
        or failure["completed_updates"] > config["steps"]
        or type(failure["error_type"]) is not str
        or not failure["error_type"]
        or type(failure["error"]) is not str
    ):
        raise ValueError("pending failure evidence is invalid")
    if failure["completed_updates"] != chain.completed_updates:
        raise ValueError("pending failure is not checkpoint-aligned")
    _validate_resumable_namespace(root, chain, pending="failed")
    revalidate_resolved_dependencies(config)
    confirmed_static_identity = _authenticate_static_run_evidence(root, config)
    confirmed_chain = _verify_transaction_chain(root, expected_config=config)
    confirmed_failure, confirmed_failure_digest = _json_snapshot(
        root / "failure.json", root=root, name="failure"
    )
    _validate_resumable_namespace(root, confirmed_chain, pending="failed")
    if (
        static_identity != confirmed_static_identity
        or chain != confirmed_chain
        or failure_digest != confirmed_failure_digest
        or failure != confirmed_failure
    ):
        raise ValueError("pending failure evidence changed during authentication")
    return confirmed_chain, confirmed_failure, confirmed_failure_digest


def _authenticate_resumable_evidence(
    root: Path, config: Mapping[str, object], pending: str | None
) -> _VerifiedChain:
    if pending == "completed":
        chain, manifest, manifest_digest, _ = _authenticate_terminal_artifacts(
            root, config
        )
        confirmed_chain = _verify_transaction_chain(
            root, expected_config=config, expected_final_step=config["steps"]
        )
        confirmed_manifest, confirmed_manifest_digest = _artifact_manifest(root)
        _validate_exact_namespace(
            root, confirmed_chain, confirmed_manifest["artifacts_sha256"]
        )
        revalidate_resolved_dependencies(config)
        if (
            chain != confirmed_chain
            or manifest != confirmed_manifest
            or manifest_digest != confirmed_manifest_digest
        ):
            raise ValueError("pending completed evidence changed during authentication")
        return confirmed_chain
    if pending == "failed":
        chain, _, _ = _authenticate_pending_failure(root, config)
        return chain
    return _authenticate_running_chain(root, config)


def _validate_resumable_run_locked(
    root: Path,
    *,
    expected_config: Mapping[str, object],
) -> AuthenticatedResumableRun:
    expected = validate_resolved_config(dict(expected_config))
    command = _load_command(root)
    config = validate_resolved_config(command["resolved_config"])
    if canonical_json_bytes(config) != canonical_json_bytes(expected):
        raise ValueError("resumable run configuration differs from expected")
    if config["run_identity"]["run_dir"] != str(root):
        raise ValueError("resumable run path differs from its configuration")
    config_hash = canonical_hash(config)
    status, status_digest = _read_status(root)
    if status["state"] != "running":
        raise ValueError("only a running run is resumable")
    if status["config_hash"] != config_hash:
        raise ValueError("running STATUS configuration mismatch")
    pending = _pending_terminalization_kind(root)
    chain = _authenticate_resumable_evidence(root, config, pending)
    if chain.completed_updates > config["steps"]:
        raise ValueError("resumable checkpoint exceeds the frozen total updates")
    confirmed_command = _load_command(root)
    confirmed_status, confirmed_status_digest = _read_status(root)
    confirmed_pending = _pending_terminalization_kind(root)
    if (
        command != confirmed_command
        or status != confirmed_status
        or status_digest != confirmed_status_digest
        or confirmed_status["state"] != "running"
        or pending != confirmed_pending
    ):
        raise ValueError("resumable identity changed during authentication")
    final_chain = _authenticate_resumable_evidence(root, config, confirmed_pending)
    if chain != final_chain:
        raise ValueError("resumable chain changed during final authentication")
    return AuthenticatedResumableRun(
        run_dir=str(root),
        output_root_identity=copy.deepcopy(config["output_root_identity"]),
        config=copy.deepcopy(config),
        config_hash=config_hash,
        status_sha256=confirmed_status_digest,
        completed_updates=final_chain.completed_updates,
        latest_transaction=final_chain.checkpoints[-1],
        pending_terminalization=pending,
    )


def validate_resumable_run(
    run_dir: Path,
    *,
    expected_config: Mapping[str, object],
) -> AuthenticatedResumableRun:
    """Authenticate a running run for Task 5 without writes or live-state restore."""

    root = Path(os.path.abspath(run_dir))
    if not root.is_dir() or root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("resumable run directory is not canonical")
    _preflight_reserved(root)
    with _run_lock(root):
        _preflight_reserved(root)
        return _validate_resumable_run_locked(root, expected_config=expected_config)


def validate_completed_run(
    run_dir: Path, *, expected_config: Mapping[str, object] | None = None
) -> AuthenticatedCompletedRun:
    root = Path(os.path.abspath(run_dir))
    if not root.is_dir() or root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("completed run directory is not canonical")
    _preflight_reserved(root)
    command = _load_command(root)
    config = validate_resolved_config(command["resolved_config"])
    config_hash = canonical_hash(config)
    if expected_config is not None:
        expected = validate_resolved_config(dict(expected_config))
        if canonical_json_bytes(expected) != canonical_json_bytes(config):
            raise ValueError("completed run configuration differs from expected")
    if config["run_identity"]["run_dir"] != str(root):
        raise ValueError("completed run path differs from its configuration")
    revalidate_resolved_dependencies(config)
    status, status_digest = _read_status(root)
    if status["state"] != "completed":
        raise ValueError("run is not completed")
    if status["config_hash"] != config_hash or status["completed_updates"] != config["steps"]:
        raise ValueError("completed STATUS identity mismatch")
    chain, manifest, manifest_digest, _ = _authenticate_terminal_artifacts(root, config)
    if manifest_digest != status["artifact_manifest_sha256"]:
        raise ValueError("STATUS does not bind the authenticated artifact manifest")
    return AuthenticatedCompletedRun(
        run_dir=str(root),
        output_root_identity=copy.deepcopy(config["output_root_identity"]),
        config=copy.deepcopy(config),
        config_hash=config_hash,
        status_sha256=status_digest,
        artifact_manifest_sha256=manifest_digest,
        artifacts_sha256=copy.deepcopy(manifest["artifacts_sha256"]),
        transaction_checkpoints=chain.checkpoints,
    )


def authenticate_transaction_checkpoint(
    run_dir: Path,
    *,
    completed_updates: int,
    expected_config: Mapping[str, object] | None = None,
) -> AuthenticatedCheckpointSnapshot:
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError("completed_updates must be a non-negative exact integer")
    completed = validate_completed_run(run_dir, expected_config=expected_config)
    matches = [
        item
        for item in completed.transaction_checkpoints
        if item.completed_updates == completed_updates
    ]
    if len(matches) != 1:
        raise ValueError("requested checkpoint is not reachable from the completed run")
    identity = matches[0]
    snapshot = snapshot_regular_file(Path(completed.run_dir) / identity.checkpoint_path, root=Path(completed.run_dir))
    if snapshot.sha256 != identity.checkpoint_sha256:
        raise ValueError("requested checkpoint changed after completed-run authentication")
    return AuthenticatedCheckpointSnapshot(identity=identity, checkpoint=snapshot)


def _initialize_fresh(
    args: argparse.Namespace, resolved: ResolvedV4Run, command_argv: Sequence[str] | None
) -> Path:
    config = resolved.config
    root = Path(config["output_root_identity"]["path"])
    run_dir = Path(config["run_identity"]["run_dir"])
    selected = list(canonical_trainer_argv(args, resume=False))
    if command_argv is not None and list(command_argv) != selected:
        raise ValueError("caller command argv differs from the canonical fresh command")
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if os.path.lexists(run_dir):
            raise FileExistsError(f"run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
        if run_dir.resolve(strict=True) != run_dir or not run_dir.is_relative_to(root):
            raise ValueError("run directory escaped its output root")
        revalidate_resolved_run(resolved)
        publish_json_once(run_dir / "COMMAND.json", _command_payload(args, config))
        publish_json_once(run_dir / "run_manifest.json", _run_manifest_payload(config))
        publish_json_once(run_dir / "environment.json", _environment_payload(config))
        publish_json_once(
            run_dir / "STATUS.json", _status_payload("running", resolved.config_hash, 0)
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    return run_dir


def _resume_preflight(
    args: argparse.Namespace,
    resolved: ResolvedV4Run,
    command_argv: Sequence[str] | None,
) -> tuple[Path, dict[str, object]]:
    run_dir = Path(resolved.config["run_identity"]["run_dir"])
    if not run_dir.is_dir() or run_dir.is_symlink() or run_dir.resolve(strict=True) != run_dir:
        raise ValueError("resume run directory is missing or noncanonical")
    _preflight_reserved(run_dir)
    command = _load_command(run_dir)
    selected = list(canonical_trainer_argv(args, resume=True))
    if command_argv is not None and list(command_argv) != selected:
        raise ValueError("caller command argv differs from the canonical resume command")
    if command["resume_argv"] != selected:
        raise ValueError("resume command identity differs from immutable COMMAND")
    if canonical_json_bytes(command["resolved_config"]) != canonical_json_bytes(resolved.config):
        raise ValueError("resume configuration differs from immutable COMMAND")
    revalidate_resolved_run(resolved)
    status, _ = _read_status(run_dir)
    if status["config_hash"] != resolved.config_hash:
        raise ValueError("resume STATUS configuration mismatch")
    return run_dir, status


def run(
    args: argparse.Namespace, *, command_argv: Sequence[str] | None = None
) -> dict[str, object]:
    resolved = resolve_run_configuration(args)
    config = resolved.config
    profile = resolve_profile(config["profile"]["name"])
    device = torch.device(config["device"])
    if args.resume:
        run_dir, status = _resume_preflight(args, resolved, command_argv)
        if status["state"] == "completed":
            validate_completed_run(run_dir, expected_config=config)
            metrics, _ = _json_snapshot(run_dir / "metrics.json", root=run_dir, name="metrics")
            return metrics
        if status["state"] == "failed":
            raise ValueError("failed run cannot resume")
        authenticated_resume = validate_resumable_run(
            run_dir, expected_config=config
        )
        if authenticated_resume.pending_terminalization == "completed":
            with _run_lock(run_dir):
                locked_status, _ = _read_status(run_dir)
                if (
                    locked_status["state"] != "running"
                    or locked_status["config_hash"] != resolved.config_hash
                ):
                    raise ValueError("pending terminalization lost its running STATUS")
                _, _, artifact_digest, metrics = _authenticate_terminal_artifacts(
                    run_dir, config
                )
                atomic_json(
                    run_dir / "STATUS.json",
                    _status_payload(
                        "completed",
                        resolved.config_hash,
                        config["steps"],
                        artifact_manifest_sha256=artifact_digest,
                    ),
                )
            validate_completed_run(run_dir, expected_config=config)
            return metrics
        if authenticated_resume.pending_terminalization == "failed":
            with _run_lock(run_dir):
                locked_status, _ = _read_status(run_dir)
                if (
                    locked_status["state"] != "running"
                    or locked_status["config_hash"] != resolved.config_hash
                ):
                    raise ValueError("pending failure lost its running STATUS")
                _, failure, _ = _authenticate_pending_failure(run_dir, config)
                atomic_json(
                    run_dir / "STATUS.json",
                    _status_payload(
                        "failed", resolved.config_hash, failure["completed_updates"]
                    ),
                )
            raise ValueError("failed run cannot resume")
    else:
        run_dir = _initialize_fresh(args, resolved, command_argv)
        status = {"state": "running"}
    _seed_everything(config["seed"])
    runtime = _build_runtime(resolved, device, profile)
    _validate_live_state(runtime.logits, runtime.optimizer)
    if args.resume:
        state = restore_latest_transaction(
            run_dir,
            expected_config=config,
            logits=runtime.logits,
            optimizer=runtime.optimizer,
        )
    else:
        zero_payload = build_checkpoint_payload(
            logits=runtime.logits,
            optimizer=runtime.optimizer,
            completed_updates=0,
            last_result=None,
            rng_state=_capture_rng_state(),
            step_times=(),
            peak_cuda_bytes=0,
            slot_audits=(),
            resolved_config=config,
        )
        generation = commit_training_transaction(
            run_dir,
            checkpoint_payload=zero_payload,
            history_segment=(),
            previous_transaction_sha256=None,
        )
        pointer = _read_pointer(run_dir)
        state = ResumeState(
            0,
            None,
            _capture_rng_state(),
            (),
            0,
            (),
            committed_transaction=generation.relative_to(run_dir).as_posix(),
            committed_transaction_sha256=pointer["manifest_sha256"],
            committed_history=(),
        )
    completed = state.completed_updates
    last_result = state.last_result
    step_times = list(state.step_times)
    peak_cuda_bytes = state.peak_cuda_bytes
    audits = list(state.slot_audits)
    pending_history: list[dict[str, object]] = []
    predecessor_digest = state.committed_transaction_sha256
    while completed < config["steps"]:
        started = _timer()
        try:
            raw_result = train_one_v4_update(
                logits=runtime.logits,
                optimizer=runtime.optimizer,
                protocol=runtime.protocol,
                targets=runtime.targets,
                bands=runtime.bands,
                profile=profile,
                global_step_before=completed,
                total_steps=config["steps"],
                config=resolved.step_config,
            )
            result = _result_from_mapping(raw_result)
            if result.global_step_before != completed:
                raise V4TrainingStepError("training step returned the wrong update identity")
            elapsed = _finite_float(
                float(_timer() - started), name="elapsed step time", nonnegative=True
            )
        except Exception as error:
            scientific = error if isinstance(error, V4TrainingStepError) else V4TrainingStepError(str(error))
            _publish_failure(
                run_dir,
                config_hash=resolved.config_hash,
                completed_updates=state.completed_updates,
                error=scientific,
            )
            raise scientific from (None if scientific is error else error)
        completed = result.completed_updates_after
        last_result = result
        step_times.append(elapsed)
        if result.slot_audit is not None:
            audits.append(result.slot_audit)
        pending_history.append({"schema": _HISTORY_SCHEMA, **asdict(result)})
        if device.type == "cuda":
            peak_cuda_bytes = max(peak_cuda_bytes, int(torch.cuda.max_memory_allocated(device)))
        if completed % config["checkpoint_policy"]["commit_interval"] == 0 or completed == config["steps"]:
            checkpoint = build_checkpoint_payload(
                logits=runtime.logits,
                optimizer=runtime.optimizer,
                completed_updates=completed,
                last_result=last_result,
                rng_state=_capture_rng_state(),
                step_times=step_times,
                peak_cuda_bytes=peak_cuda_bytes,
                slot_audits=audits,
                resolved_config=config,
            )
            commit_training_transaction(
                run_dir,
                checkpoint_payload=checkpoint,
                history_segment=pending_history,
                previous_transaction_sha256=predecessor_digest,
            )
            pointer = _read_pointer(run_dir)
            predecessor_digest = pointer["manifest_sha256"]
            state = ResumeState(
                completed,
                last_result,
                _capture_rng_state(),
                tuple(step_times),
                peak_cuda_bytes,
                tuple(audits),
                committed_transaction=pointer["transaction"],
                committed_transaction_sha256=predecessor_digest,
                committed_history=(),
            )
            pending_history = []
        _post_update_hook(completed)
    return _materialize_final(
        run_dir,
        config=config,
        runtime=runtime,
        step_times=step_times,
        peak_cuda_bytes=peak_cuda_bytes,
        audits=audits,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuthenticatedCheckpointSnapshot",
    "AuthenticatedCompletedRun",
    "AuthenticatedResumableRun",
    "AuthenticatedTransactionCheckpoint",
    "ResumePreflightError",
    "ResumeState",
    "V4TrainingStepError",
    "authenticate_transaction_checkpoint",
    "build_checkpoint_payload",
    "commit_training_transaction",
    "main",
    "restore_checkpoint_payload",
    "restore_latest_transaction",
    "run",
    "validate_completed_run",
    "validate_resumable_run",
]
