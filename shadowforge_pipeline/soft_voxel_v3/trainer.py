"""Exactly one optimizer update for the isolated soft-loss-v3 objective."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import copy
from dataclasses import asdict, dataclass, fields
import fcntl
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import stat
import statistics
import sys
import time
import uuid

import cv2
import numpy as np
import torch

from .contours import ContourBands, make_contour_bands, validate_contour_bands
from .data import normalize_target
from .diagnostics import ComponentGradientSample, measure_component_gradients
from .losses import (
    V3Profile,
    compute_baseline_branch,
    compute_dic_branch,
    compute_dts_branch,
    dic_schedule,
    dts_schedule,
    registered_profiles,
    resolve_profile,
)
from .metrics import evaluate_delivery_metrics
from .protocol import (
    CAMERA_AZIMUTH_DEG,
    CAMERA_ELEVATION_DEG,
    SadekarTFSProtocol,
)
from .provenance import (
    atomic_checkpoint,
    atomic_json,
    atomic_text,
    canonical_hash,
    environment_manifest,
    implementation_source_manifest,
    interpreter_manifest,
    normalize_builtin,
    sha256_file,
)
from .thresholding import _protocol_device, render_thresholded


@dataclass(frozen=True)
class StepRecord:
    """Detached, JSON-safe telemetry for one completed optimizer update."""

    global_step_before: int
    completed_updates_after: int
    progress: float
    dts_endpoint: float | None
    loss_base: float
    loss_dic: float | None
    loss_dts: float | None
    dic_scale: float
    dts_scale: float
    pre_clip_grad_norm: float
    post_clip_grad_norm: float


class SoftTrainingStepError(RuntimeError):
    """A non-finite state prevented the optimizer update."""


@dataclass(frozen=True)
class ResumeState:
    """Exact checkpoint state plus its verified committed history prefix."""

    completed_updates: int
    last_record: StepRecord | None
    rng_state: dict[str, object]
    gradient_samples: tuple[ComponentGradientSample, ...]
    step_times: tuple[float, ...]
    peak_cuda_bytes: int
    calibration_records: tuple[dict[str, object], ...]
    committed_transaction: str | None = None
    committed_transaction_sha256: str | None = None
    committed_history: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class _Runtime:
    logits: torch.Tensor
    optimizer: torch.optim.Optimizer
    protocol: SadekarTFSProtocol
    targets: torch.Tensor
    bands: ContourBands | None


@dataclass(frozen=True)
class _ShadowInputSnapshot:
    path: str
    sha256: str
    payload: bytes


class ResumePreflightError(ValueError):
    """Existing audit bytes failed validation and were left untouched."""


@dataclass(frozen=True)
class _VerifiedTransactionChain:
    final_transaction: str
    final_manifest_sha256: str
    completed_updates: int
    history: tuple[dict[str, object], ...]
    files: tuple[str, ...]


def _validate_logits(logits: object) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise ValueError("logits must be a torch.Tensor")
    if not logits.is_floating_point():
        raise ValueError("logits must use a real floating-point dtype")
    if (
        logits.ndim != 5
        or logits.shape[:2] != (1, 1)
        or logits.shape[2] == 0
        or logits.shape[2:] != (logits.shape[2],) * 3
    ):
        raise ValueError("logits must have non-empty cubic shape [1,1,D,H,W]")
    if not logits.requires_grad:
        raise ValueError("logits must be gradient-enabled")
    if not logits.is_leaf:
        raise ValueError("logits must be a leaf tensor")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must be finite")
    return logits


def _validate_optimizer(optimizer: object) -> None:
    if not callable(getattr(optimizer, "zero_grad", None)):
        raise ValueError("optimizer must expose callable zero_grad")
    if not callable(getattr(optimizer, "step", None)):
        raise ValueError("optimizer must expose callable step")


def _validate_optimizer_owns_logits(
    optimizer: object, logits: torch.Tensor
) -> None:
    """Require the optimizer's sole live parameter to be this exact tensor."""

    parameters: list[object] = []
    groups = getattr(optimizer, "param_groups", None)
    if isinstance(groups, list):
        for group in groups:
            if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
                raise ValueError("optimizer parameter groups are malformed")
            parameters.extend(group["params"])
    elif hasattr(optimizer, "parameter"):
        parameters.append(getattr(optimizer, "parameter"))
    else:
        raise ValueError("optimizer must own the exact live logits object")
    if len(parameters) != 1 or parameters[0] is not logits:
        raise ValueError("optimizer must own the exact live logits object")


def _validate_protocol(protocol: object, logits: torch.Tensor) -> None:
    if not callable(getattr(protocol, "render_all", None)):
        raise ValueError("protocol must expose callable render_all")
    has_explicit_device_state = (
        getattr(protocol, "cameras", None) is not None
        or getattr(protocol, "renderer", None) is not None
        or hasattr(protocol, "device")
    )
    if has_explicit_device_state:
        try:
            protocol_device = _protocol_device(protocol)
        except ValueError as error:
            raise ValueError("protocol must expose a valid device") from error
        if protocol_device != logits.device:
            raise ValueError("protocol and logits must be on the same device")


def _validate_targets(targets: object, logits: torch.Tensor) -> torch.Tensor:
    if not isinstance(targets, torch.Tensor):
        raise ValueError("targets must be a torch.Tensor")
    if not targets.is_floating_point():
        raise ValueError("targets must use a real floating-point dtype")
    if (
        targets.ndim != 3
        or targets.shape[0] != 6
        or targets.shape[1] == 0
        or targets.shape[2] == 0
    ):
        raise ValueError("targets must have non-empty shape [6,H,W]")
    if targets.device != logits.device:
        raise ValueError("targets and logits must be on the same device")
    if not torch.isfinite(targets).all().item():
        raise ValueError("targets must be finite")
    return targets


def _validate_profile(profile: object) -> V3Profile:
    if not isinstance(profile, V3Profile):
        raise ValueError("profile must be a V3Profile")
    if (
        type(profile.name) is not str
        or type(profile.l1_weight) is not float
        or type(profile.l2_weight) is not float
        or type(profile.use_dic) is not bool
        or type(profile.use_dts) is not bool
        or (
            profile.dts_thresholds is not None
            and (
                type(profile.dts_thresholds) is not tuple
                or len(profile.dts_thresholds) != 2
                or any(type(value) is not float for value in profile.dts_thresholds)
            )
        )
    ):
        raise ValueError("profile fields must use the exact registered schema")
    try:
        registered = resolve_profile(profile.name)
    except ValueError as error:
        raise ValueError("profile must be a registered V3Profile") from error
    if profile != registered:
        raise ValueError("profile state must match its registered V3Profile")
    return profile


def _validate_update_counts(
    completed_updates: object, total_steps: object
) -> tuple[int, int]:
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, Integral)
        or completed_updates < 0
    ):
        raise ValueError("completed_updates must be a non-negative integer")
    if (
        isinstance(total_steps, bool)
        or not isinstance(total_steps, Integral)
        or total_steps <= 0
    ):
        raise ValueError("total_steps must be a positive integer")
    completed = int(completed_updates)
    total = int(total_steps)
    if completed >= total:
        raise ValueError("completed_updates must be less than total_steps")
    return completed, total


def _validate_lambda(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative scalar")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite non-negative scalar") from error
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return normalized


def _validate_active_bands(
    bands: object, targets: torch.Tensor
) -> ContourBands:
    if not isinstance(bands, ContourBands):
        raise ValueError("bands must be a ContourBands instance while DIC is active")
    validate_contour_bands(bands)
    if bands.inner.shape != targets.shape:
        raise ValueError("bands and targets must have the same shape")
    if bands.inner.device != targets.device:
        raise ValueError("bands and targets must share a device")
    return bands


def _finite_loss(loss: object, *, name: str) -> torch.Tensor:
    if (
        not isinstance(loss, torch.Tensor)
        or loss.ndim != 0
        or not loss.is_floating_point()
        or not torch.isfinite(loss).item()
    ):
        raise SoftTrainingStepError(f"{name} loss must be a finite scalar tensor")
    return loss


def _finite_gradient(logits: torch.Tensor, *, stage: str) -> torch.Tensor:
    gradient = logits.grad
    if (
        not isinstance(gradient, torch.Tensor)
        or gradient.shape != logits.shape
        or not gradient.is_floating_point()
        or not torch.isfinite(gradient).all().item()
    ):
        raise SoftTrainingStepError(f"logits gradient must be finite {stage}")
    return gradient


def train_one_update(
    *,
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    protocol: SadekarTFSProtocol,
    targets: torch.Tensor,
    bands: ContourBands | None,
    profile: V3Profile,
    completed_updates: int,
    total_steps: int,
    lambda_dic: float,
    lambda_dts: float,
) -> StepRecord:
    """Perform exactly one sequential, accumulated-gradient optimizer update."""

    logits = _validate_logits(logits)
    _validate_optimizer(optimizer)
    _validate_optimizer_owns_logits(optimizer, logits)
    _validate_protocol(protocol, logits)
    targets = _validate_targets(targets, logits)
    profile = _validate_profile(profile)
    completed, total = _validate_update_counts(completed_updates, total_steps)
    lambda_dic_value = _validate_lambda(lambda_dic, name="lambda_dic")
    lambda_dts_value = _validate_lambda(lambda_dts, name="lambda_dts")

    progress = (completed + 1) / float(total)
    dic_weight = dic_schedule(progress) if profile.use_dic else 0.0
    dts_weight = dts_schedule(progress) if profile.use_dts else 0.0
    dic_scale = lambda_dic_value * dic_weight if dic_weight > 0.0 else 0.0
    dts_scale = lambda_dts_value * dts_weight if dts_weight > 0.0 else 0.0

    optimizer.zero_grad(set_to_none=True)

    density = torch.sigmoid(logits)
    projections = protocol.render_all(density)
    base = _finite_loss(
        compute_baseline_branch(projections, targets, profile), name="baseline"
    )
    loss_base = float(base.detach().item())
    base.backward()
    del base, projections, density

    loss_dic = None
    if dic_weight > 0.0:
        active_bands = _validate_active_bands(bands, targets)
        density = torch.sigmoid(logits)
        readout = render_thresholded(protocol, density, 0.05)
        dic = _finite_loss(compute_dic_branch(readout.score, active_bands), name="DIC")
        loss_dic = float(dic.detach().item())
        (dic_scale * dic).backward()
        del dic, readout, density

    loss_dts = None
    dts_endpoint = None
    if dts_weight > 0.0:
        assert profile.dts_thresholds is not None
        dts_endpoint = float(profile.dts_thresholds[completed % 2])
        density = torch.sigmoid(logits)
        readout = render_thresholded(protocol, density, dts_endpoint)
        dts = _finite_loss(
            compute_dts_branch(readout.score, targets >= 0.5), name="DTS"
        )
        loss_dts = float(dts.detach().item())
        (dts_scale * dts).backward()
        del dts, readout, density

    _finite_gradient(logits, stage="before clipping")
    pre_clip = torch.nn.utils.clip_grad_norm_([logits], 1.0)
    try:
        pre_clip_grad_norm = float(pre_clip)
    except (TypeError, ValueError, RuntimeError) as error:
        raise SoftTrainingStepError("pre-clip gradient norm must be finite") from error
    if not math.isfinite(pre_clip_grad_norm):
        raise SoftTrainingStepError("pre-clip gradient norm must be finite")
    clipped_gradient = _finite_gradient(logits, stage="after clipping")
    post_clip_grad_norm = float(torch.linalg.vector_norm(clipped_gradient).item())
    if not math.isfinite(post_clip_grad_norm):
        raise SoftTrainingStepError("post-clip gradient norm must be finite")

    optimizer.step()

    return StepRecord(
        global_step_before=completed,
        completed_updates_after=completed + 1,
        progress=float(progress),
        dts_endpoint=dts_endpoint,
        loss_base=loss_base,
        loss_dic=loss_dic,
        loss_dts=loss_dts,
        dic_scale=float(dic_scale),
        dts_scale=float(dts_scale),
        pre_clip_grad_norm=pre_clip_grad_norm,
        post_clip_grad_norm=post_clip_grad_norm,
    )


_CHECKPOINT_SCHEMA = "pcraft_soft_loss_v3_checkpoint_v1"
_TRANSACTION_SCHEMA = "pcraft_soft_loss_v3_transaction_v1"
_POINTER_SCHEMA = "pcraft_soft_loss_v3_checkpoint_pointer_v1"
_STATUS_SCHEMA = "pcraft_soft_loss_v3_status_v1"
_THRESHOLDS = (0.04, 0.05, 0.06)
_COMMIT_INTERVAL = 100
_CALIBRATION_UPDATES = (1000, 1300, 1600)
_TRANSACTION_PATH = re.compile(r"transactions/step_[0-9]{8}\Z")
_TEMPORARY_GENERATION_NAME = re.compile(
    r"\.step_[0-9]{8}\.tmp\.[A-Za-z0-9_.-]+\Z"
)
_TRANSACTION_PAYLOAD_NAMES = (
    "checkpoint.pt",
    "history_segment.jsonl",
    "transaction_manifest.json",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the sole project-local v3 trainer CLI."""

    parser = argparse.ArgumentParser(
        description="P-CRAFT soft-loss-v3 trainer (soft-only, final-step export)"
    )
    parser.add_argument("device")
    parser.add_argument("output_dir")
    parser.add_argument("steps", type=int)
    parser.add_argument("lr", type=float)
    parser.add_argument("-sdlist", "--shadow-files", nargs=3, required=True)
    parser.add_argument("-mr", "--mirror-mode", type=int, default=1)
    parser.add_argument("-tsf", "--tsf-mode", type=int, default=1)
    parser.add_argument("-tsf4", "--tsf4-mode", type=int, default=0)
    parser.add_argument("-cam", "--camera-mode", default="ortho")
    parser.add_argument("-imsz", "--image-size", type=int, default=256)
    parser.add_argument("--res", type=int, choices=(128, 256), required=True)
    parser.add_argument("-eres", "--export-res", dest="export_res", type=int, required=True)
    parser.add_argument("--control-res", type=int, default=0)
    parser.add_argument("--sadekar-zdist", type=float, default=1.7)
    parser.add_argument("--sadekar-volume-extent", type=float, default=1.7)
    parser.add_argument("--sadekar-npts-per-ray", type=int, default=150)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--profile", choices=registered_profiles(), required=True)
    parser.add_argument("--lambda-dic", type=float, required=True)
    parser.add_argument("--lambda-dts", type=float, required=True)
    parser.add_argument("--suite-manifest", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--weight-config", "--weights", dest="weight_config")
    parser.add_argument("--calibration-probe", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__OUTPUT_LATEST__",
        default=None,
    )
    parser.add_argument("--export-final-step", action="store_true")
    parser.add_argument("--soft-only-eval", action="store_true", required=True)
    return parser


def _exact_float(value: object, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    )


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative scalar")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return normalized


def _hashed_path(path_value: object, *, name: str) -> dict[str, str]:
    if not isinstance(path_value, (str, os.PathLike)):
        raise ValueError(f"{name} must be a filesystem path")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _snapshotted_path(
    path_value: object, *, name: str
) -> tuple[dict[str, str], _ShadowInputSnapshot]:
    if not isinstance(path_value, (str, os.PathLike)):
        raise ValueError(f"{name} must be a filesystem path")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    return (
        {"path": str(path), "sha256": digest},
        _ShadowInputSnapshot(path=str(path), sha256=digest, payload=payload),
    )


def _load_frozen_weights(path: Path) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid frozen weight config: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("frozen weight config must be a mapping")
    if payload.get("schema") != "pcraft_soft_loss_v3_weights_v1":
        raise ValueError("frozen weight config has the wrong schema")
    if payload.get("status") != "APPROVED_FROZEN":
        raise ValueError("weight config must have APPROVED_FROZEN status")
    lambdas = payload.get("lambdas")
    if not isinstance(lambdas, Mapping) or set(lambdas) != {
        "lambda_dic",
        "lambda_dts",
    }:
        raise ValueError("frozen weight config must contain exact lambdas")
    _finite_nonnegative(lambdas["lambda_dic"], name="frozen lambda_dic")
    _finite_nonnegative(lambdas["lambda_dts"], name="frozen lambda_dts")
    return payload


def resolved_config(args: argparse.Namespace) -> dict[str, object]:
    """Validate and bind every scientific byte before any run-directory write."""

    if not bool(getattr(args, "soft_only_eval", False)):
        raise ValueError("--soft-only-eval is mandatory")
    try:
        profile = resolve_profile(args.profile)
    except (AttributeError, ValueError) as error:
        raise ValueError("unknown v3 profile") from error
    if (
        args.mirror_mode != 1
        or args.tsf_mode != 1
        or args.tsf4_mode != 0
    ):
        raise ValueError("fixed protocol requires mirror=1, TFS=1, TFS4=0")
    if args.camera_mode != "ortho" or args.image_size != 256:
        raise ValueError("fixed protocol requires orthographic camera and image size 256")
    if args.res not in (128, 256) or args.export_res != args.res:
        raise ValueError("resolution must be 128 or 256 and export resolution must match")
    if args.control_res != 0:
        raise ValueError("fixed protocol requires control resolution 0")
    if not _exact_float(args.sadekar_zdist, 1.7):
        raise ValueError("fixed protocol z distance must be 1.7")
    if not _exact_float(args.sadekar_volume_extent, 1.7):
        raise ValueError("fixed protocol volume extent must be 1.7")
    if args.sadekar_npts_per_ray != 150:
        raise ValueError("fixed protocol requires 150 samples per ray")
    if args.seed != 17:
        raise ValueError("fixed seed must be 17")
    if isinstance(args.steps, bool) or not isinstance(args.steps, Integral) or args.steps <= 0:
        raise ValueError("steps must be a positive integer")
    learning_rate = _finite_nonnegative(args.lr, name="learning rate")
    if learning_rate != 0.007:
        raise ValueError("learning rate is frozen to exactly 0.007")
    lambda_dic = _finite_nonnegative(args.lambda_dic, name="lambda_dic")
    lambda_dts = _finite_nonnegative(args.lambda_dts, name="lambda_dts")
    if not isinstance(args.shadow_files, Sequence) or isinstance(args.shadow_files, str):
        raise ValueError("exactly three ordered x/y/z shadow paths are required")
    if len(args.shadow_files) != 3:
        raise ValueError("exactly three ordered x/y/z shadow paths are required")
    shadow_pairs = [
        _snapshotted_path(path, name=f"{axis}-shadow input")
        for axis, path in zip("xyz", args.shadow_files)
    ]
    shadow_inputs = [pair[0] for pair in shadow_pairs]
    setattr(
        args,
        "_soft_loss_v3_shadow_snapshots",
        tuple(pair[1] for pair in shadow_pairs),
    )
    suite = _hashed_path(args.suite_manifest, name="suite manifest")
    spec = _hashed_path(args.spec, name="revised specification")
    calibration = bool(args.calibration_probe)
    weight: dict[str, str] | None
    if calibration:
        if profile.name != "l12_bal":
            raise ValueError("calibration probe mode is legal only with l12_bal")
        if args.steps != 2000 or args.res != 128:
            raise ValueError("calibration probe mode requires the frozen 2000-step 128 run")
        if args.weight_config is not None:
            raise ValueError("calibration probe mode must not receive a frozen weight file")
        if lambda_dic != 0.0 or lambda_dts != 0.0:
            raise ValueError("calibration probe mode requires zero applied lambdas")
        weight = None
    else:
        if args.weight_config is None:
            raise ValueError("normal runs require a frozen weight config")
        weight = _hashed_path(args.weight_config, name="frozen weight config")
        frozen = _load_frozen_weights(Path(weight["path"]))
        frozen_lambdas = frozen["lambdas"]
        assert isinstance(frozen_lambdas, Mapping)
        if not _exact_float(frozen_lambdas["lambda_dic"], lambda_dic) or not _exact_float(
            frozen_lambdas["lambda_dts"], lambda_dts
        ):
            raise ValueError("CLI lambdas do not match the APPROVED_FROZEN weight config")
    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as error:
        raise ValueError("device must be explicit and valid") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device is unavailable: {device}")
    environment = environment_manifest(device)
    return {
        "schema": "pcraft_soft_loss_v3_config_v1",
        "device": str(device),
        "steps": int(args.steps),
        "learning_rate": learning_rate,
        "resolution": int(args.res),
        "export_resolution": int(args.export_res),
        "profile": profile.name,
        "lambda_dic": lambda_dic,
        "lambda_dts": lambda_dts,
        "seed": 17,
        "shadow_inputs_xyz": shadow_inputs,
        "suite_manifest": suite,
        "specification": spec,
        "weight_config": weight,
        "calibration_probe": calibration,
        "protocol": {
            "mirror_mode": 1,
            "tsf_mode": 1,
            "tsf4_mode": 0,
            "camera_mode": "ortho",
            "image_size": 256,
            "control_res": 0,
            "zdist": 1.7,
            "volume_extent": 1.7,
            "samples_per_ray": 150,
        },
        "density_thresholds": list(_THRESHOLDS),
        "final_step_export_only": True,
        "soft_only": True,
        "implementation": implementation_source_manifest(),
        "interpreter": interpreter_manifest(),
        "determinism": {
            "fixed_seed": True,
            "mode": "best_effort",
            "bitwise_reproducible": False,
            "reason": environment["nondeterminism_reason"],
            "environment": environment["environment"],
        },
    }


def _validate_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _validate_rng_state(state: object) -> dict[str, object]:
    if not isinstance(state, Mapping) or set(state) != {
        "python",
        "numpy",
        "torch",
        "cuda",
    }:
        raise ValueError("rng_state must contain exact Python/NumPy/Torch/CUDA state")
    torch_state = state["torch"]
    cuda_state = state["cuda"]
    if (
        not isinstance(torch_state, torch.Tensor)
        or torch_state.dtype != torch.uint8
        or torch_state.ndim != 1
    ):
        raise ValueError("rng_state torch state is malformed")
    if not isinstance(cuda_state, (list, tuple)) or any(
        not isinstance(item, torch.Tensor)
        or item.dtype != torch.uint8
        or item.ndim != 1
        for item in cuda_state
    ):
        raise ValueError("rng_state CUDA state is malformed")
    saved_python = random.getstate()
    saved_numpy = np.random.get_state()
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
    except Exception as error:
        raise ValueError("rng_state Python or NumPy state is malformed") from error
    finally:
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)
    return copy.deepcopy(dict(state))


def _restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    cuda_state = state["cuda"]
    if cuda_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _step_record_from_mapping(payload: object) -> StepRecord:
    if not isinstance(payload, Mapping) or set(payload) != {
        field.name for field in fields(StepRecord)
    }:
        raise ValueError("last_record must have the exact StepRecord schema")
    integers = ("global_step_before", "completed_updates_after")
    required_floats = (
        "progress",
        "loss_base",
        "dic_scale",
        "dts_scale",
        "pre_clip_grad_norm",
        "post_clip_grad_norm",
    )
    optional_floats = ("dts_endpoint", "loss_dic", "loss_dts")
    values = dict(payload)
    for name in integers:
        if isinstance(values[name], bool) or not isinstance(values[name], Integral):
            raise ValueError(f"last_record {name} must be an integer")
        values[name] = int(values[name])
    for name in required_floats:
        if isinstance(values[name], bool) or not isinstance(values[name], Real):
            raise ValueError(f"last_record {name} must be a finite float")
        values[name] = float(values[name])
        if not math.isfinite(values[name]):
            raise ValueError(f"last_record {name} must be finite")
    for name in optional_floats:
        if values[name] is not None:
            if isinstance(values[name], bool) or not isinstance(values[name], Real):
                raise ValueError(f"last_record {name} must be a finite float or null")
            values[name] = float(values[name])
            if not math.isfinite(values[name]):
                raise ValueError(f"last_record {name} must be finite")
    if values["completed_updates_after"] != values["global_step_before"] + 1:
        raise ValueError("last_record completed-update identity is inconsistent")
    if values["loss_dic"] is None and values["dic_scale"] != 0.0:
        raise ValueError("last_record inactive DIC loss must have zero scale")
    if values["loss_dts"] is None:
        if values["dts_scale"] != 0.0 or values["dts_endpoint"] is not None:
            raise ValueError("last_record inactive DTS loss must have zero scale and null endpoint")
    else:
        endpoint = values["dts_endpoint"]
        if endpoint not in (0.04, 0.05, 0.06):
            raise ValueError("last_record DTS endpoint is not registered")
        if endpoint in (0.04, 0.06):
            expected_endpoint = (
                0.04 if values["global_step_before"] % 2 == 0 else 0.06
            )
            if endpoint != expected_endpoint:
                raise ValueError("last_record DTS endpoint parity is inconsistent")
    return StepRecord(**values)


def _float_key_mapping(
    value: object,
    *,
    name: str,
    optional_values: bool = False,
    bool_values: bool = False,
) -> dict[float, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: dict[float, object] = {}
    for raw_key, raw_value in value.items():
        if isinstance(raw_key, bool):
            raise ValueError(f"{name} threshold keys must be finite numbers")
        try:
            key = float(raw_key)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} threshold keys must be finite numbers") from error
        if not math.isfinite(key) or key in result:
            raise ValueError(f"{name} threshold keys must be unique finite numbers")
        if bool_values:
            if type(raw_value) is not bool:
                raise ValueError(f"{name} values must be boolean")
            normalized: object = raw_value
        elif raw_value is None and optional_values:
            normalized = None
        else:
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise ValueError(f"{name} values must be finite floats")
            normalized = float(raw_value)
            if not math.isfinite(normalized):
                raise ValueError(f"{name} values must be finite")
        result[key] = normalized
    return {key: result[key] for key in sorted(result)}


def _nested_float_key_mapping(
    value: object, *, name: str, optional_values: bool, bool_values: bool
) -> dict[float, dict[float, object]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    outer: dict[float, dict[float, object]] = {}
    for raw_key, nested in value.items():
        try:
            key = float(raw_key)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} threshold keys must be finite numbers") from error
        if not math.isfinite(key) or key in outer:
            raise ValueError(f"{name} threshold keys must be unique finite numbers")
        outer[key] = _float_key_mapping(
            nested,
            name=f"{name}[{key}]",
            optional_values=optional_values,
            bool_values=bool_values,
        )
    return {key: outer[key] for key in sorted(outer)}


def _component_sample_from_mapping(payload: object) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != {
        field.name for field in fields(ComponentGradientSample)
    }:
        raise ValueError("gradient sample must have the exact ComponentGradientSample schema")
    values = dict(payload)
    if isinstance(values["completed_updates"], bool) or not isinstance(
        values["completed_updates"], Integral
    ):
        raise ValueError("gradient sample completed_updates must be an integer")
    values["completed_updates"] = int(values["completed_updates"])
    if values["completed_updates"] < 0:
        raise ValueError("gradient sample completed_updates must be non-negative")
    for name in ("base_norm",):
        values[name] = _finite_nonnegative(values[name], name=name)
    for name in (
        "dic_norm",
        "base_dic_cosine",
        "weighted_dic_base_ratio",
        "selected_dts_tau",
        "combined_aux_base_ratio",
        "base_combined_aux_cosine",
    ):
        raw = values[name]
        if raw is not None:
            if isinstance(raw, bool) or not isinstance(raw, Real):
                raise ValueError(f"gradient sample {name} must be finite or null")
            normalized = float(raw)
            if not math.isfinite(normalized):
                raise ValueError(f"gradient sample {name} must be finite")
            values[name] = normalized
    for name in (
        "base_dic_cosine_invalid",
        "weighted_dic_base_ratio_invalid",
        "combined_aux_base_ratio_invalid",
        "base_combined_aux_cosine_invalid",
    ):
        if type(values[name]) is not bool:
            raise ValueError(f"gradient sample {name} must be boolean")
    values["dts_norms"] = _float_key_mapping(values["dts_norms"], name="dts_norms")
    for name in (
        "base_dts_cosines",
        "dic_dts_cosines",
        "dic_dts_support_ious",
        "weighted_dts_base_ratios",
    ):
        values[name] = _float_key_mapping(
            values[name], name=name, optional_values=True
        )
    for name in (
        "base_dts_cosine_invalid",
        "dic_dts_cosine_invalid",
        "dic_dts_support_iou_invalid",
        "weighted_dts_base_ratio_invalid",
    ):
        values[name] = _float_key_mapping(
            values[name], name=name, bool_values=True
        )
    values["dts_pair_cosines"] = _nested_float_key_mapping(
        values["dts_pair_cosines"],
        name="dts_pair_cosines",
        optional_values=True,
        bool_values=False,
    )
    values["dts_pair_cosine_invalid"] = _nested_float_key_mapping(
        values["dts_pair_cosine_invalid"],
        name="dts_pair_cosine_invalid",
        optional_values=False,
        bool_values=True,
    )
    return values


def _validated_samples(value: object) -> tuple[ComponentGradientSample, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("gradient_samples must be a sequence")
    samples: list[ComponentGradientSample] = []
    for item in value:
        mapping = asdict(item) if isinstance(item, ComponentGradientSample) else item
        samples.append(ComponentGradientSample(**_component_sample_from_mapping(mapping)))
    completed = [sample.completed_updates for sample in samples]
    if completed != sorted(set(completed)):
        raise ValueError("gradient_samples must have unique increasing update identities")
    return tuple(samples)


def _validated_times(value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("step_times must be a sequence")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError("step_times must contain finite non-negative floats")
        normalized = float(item)
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError("step_times must contain finite non-negative floats")
        result.append(normalized)
    return tuple(result)


def _validated_calibration_records(
    value: object,
    *,
    samples: Sequence[ComponentGradientSample],
    completed_updates: int,
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("calibration_records must be a sequence")
    expected_keys = {"completed_updates", "g0", "g_dic", "g_dts_004", "g_dts_006"}
    samples_by_update = {sample.completed_updates: sample for sample in samples}
    records: list[dict[str, object]] = []
    seen: set[int] = set()
    for item in value:
        if not isinstance(item, Mapping) or set(item) != expected_keys:
            raise ValueError("calibration records must have the exact five-field schema")
        raw_update = item["completed_updates"]
        if isinstance(raw_update, bool) or not isinstance(raw_update, Integral):
            raise ValueError("calibration completed_updates must be an integer")
        update = int(raw_update)
        if (
            update not in _CALIBRATION_UPDATES
            or update > completed_updates
            or update in seen
        ):
            raise ValueError(
                "calibration updates must be unique registered updates at or before the checkpoint"
            )
        seen.add(update)
        normalized: dict[str, object] = {"completed_updates": update}
        for name in ("g0", "g_dic", "g_dts_004", "g_dts_006"):
            raw_value = item[name]
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise ValueError("calibration gradients must be finite non-negative numbers")
            number = float(raw_value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError("calibration gradients must be finite non-negative numbers")
            normalized[name] = number
        sample = samples_by_update.get(update)
        if (
            sample is None
            or sample.dic_norm is None
            or set(sample.dts_norms) != {0.04, 0.06}
        ):
            raise ValueError("calibration record has no matching complete gradient sample")
        expected_values = {
            "g0": sample.base_norm,
            "g_dic": sample.dic_norm,
            "g_dts_004": sample.dts_norms[0.04],
            "g_dts_006": sample.dts_norms[0.06],
        }
        if any(normalized[name] != float(expected_values[name]) for name in expected_values):
            raise ValueError("calibration record disagrees with its gradient sample")
        records.append(normalized)
    if [record["completed_updates"] for record in records] != sorted(seen):
        raise ValueError("calibration records must have increasing update identities")
    return tuple(records)


def _validate_step_last_record(
    completed_updates: object, last_record: object
) -> tuple[int, StepRecord | None]:
    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, Integral)
        or completed_updates < 0
    ):
        raise ValueError("completed_updates must be a non-negative integer")
    completed = int(completed_updates)
    record = None
    if last_record is not None:
        record = _step_record_from_mapping(
            asdict(last_record) if isinstance(last_record, StepRecord) else last_record
        )
    if completed == 0 and record is not None:
        raise ValueError("last_record must be null at committed step zero")
    if completed > 0 and (
        record is None or record.completed_updates_after != completed
    ):
        raise ValueError("last_record must match the positive completed update")
    return completed, record


def build_checkpoint_payload(
    *,
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    completed_updates: int,
    last_record: StepRecord | None,
    rng_state: Mapping[str, object],
    gradient_samples: Sequence[ComponentGradientSample],
    step_times: Sequence[float],
    peak_cuda_bytes: int,
    calibration_records: Sequence[Mapping[str, object]],
    config_hash: str,
) -> dict[str, object]:
    """Build the exact, self-contained v3 continuation checkpoint payload."""

    live_logits = _validate_logits(logits)
    _validate_optimizer(optimizer)
    _validate_optimizer_owns_logits(optimizer, live_logits)
    completed, record = _validate_step_last_record(completed_updates, last_record)
    digest = _validate_digest(config_hash, name="config_hash")
    rng = _validate_rng_state(rng_state)
    samples = _validated_samples(gradient_samples)
    times = _validated_times(step_times)
    if len(times) != completed:
        raise ValueError("step_times must contain one sample per completed update")
    if (
        isinstance(peak_cuda_bytes, bool)
        or not isinstance(peak_cuda_bytes, Integral)
        or peak_cuda_bytes < 0
    ):
        raise ValueError("peak_cuda_bytes must be a non-negative integer")
    calibration = _validated_calibration_records(
        calibration_records,
        samples=samples,
        completed_updates=completed,
    )
    return {
        "schema": _CHECKPOINT_SCHEMA,
        "config_hash": digest,
        "completed_updates": completed,
        "logits": live_logits.detach().to(device="cpu").clone(),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "last_record": None if record is None else asdict(record),
        "rng_state": rng,
        "gradient_samples": [asdict(sample) for sample in samples],
        "step_times": list(times),
        "peak_cuda_bytes": int(peak_cuda_bytes),
        "calibration_records": [dict(item) for item in calibration],
    }


def _decode_checkpoint_payload(
    payload: object,
    *,
    expected_config_hash: str,
    logits: torch.Tensor,
) -> tuple[ResumeState, torch.Tensor, Mapping[str, object]]:
    required = {
        "schema",
        "config_hash",
        "completed_updates",
        "logits",
        "optimizer",
        "last_record",
        "rng_state",
        "gradient_samples",
        "step_times",
        "peak_cuda_bytes",
        "calibration_records",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ResumePreflightError("checkpoint must have the exact v3 schema")
    if payload["schema"] != _CHECKPOINT_SCHEMA:
        raise ResumePreflightError("checkpoint schema mismatch")
    expected = _validate_digest(expected_config_hash, name="expected_config_hash")
    if payload["config_hash"] != expected:
        raise ResumePreflightError("checkpoint configuration hash mismatch")
    completed, record = _validate_step_last_record(
        payload["completed_updates"], payload["last_record"]
    )
    stored_logits = payload["logits"]
    if (
        not isinstance(stored_logits, torch.Tensor)
        or stored_logits.shape != logits.shape
        or stored_logits.dtype != logits.dtype
        or not stored_logits.is_floating_point()
        or not torch.isfinite(stored_logits).all().item()
    ):
        raise ResumePreflightError("checkpoint logits do not match the live tensor")
    optimizer_state = payload["optimizer"]
    if not isinstance(optimizer_state, Mapping):
        raise ResumePreflightError("checkpoint optimizer state is malformed")
    rng = _validate_rng_state(payload["rng_state"])
    samples = _validated_samples(payload["gradient_samples"])
    if any(sample.completed_updates > completed for sample in samples):
        raise ResumePreflightError("gradient sample is ahead of the checkpoint")
    times = _validated_times(payload["step_times"])
    if len(times) != completed:
        raise ResumePreflightError("checkpoint step-time count does not match update count")
    peak = payload["peak_cuda_bytes"]
    if isinstance(peak, bool) or not isinstance(peak, Integral) or peak < 0:
        raise ResumePreflightError("checkpoint peak CUDA bytes are malformed")
    calibration = _validated_calibration_records(
        payload["calibration_records"],
        samples=samples,
        completed_updates=completed,
    )
    state = ResumeState(
        completed_updates=completed,
        last_record=record,
        rng_state=rng,
        gradient_samples=samples,
        step_times=times,
        peak_cuda_bytes=int(peak),
        calibration_records=calibration,
    )
    return state, stored_logits.detach().clone(), optimizer_state


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def restore_checkpoint_payload(
    path: Path,
    *,
    expected_config_hash: str,
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> ResumeState:
    """Validate a checkpoint completely, then restore it transactionally."""

    live_logits = _validate_logits(logits)
    _validate_optimizer(optimizer)
    _validate_optimizer_owns_logits(optimizer, live_logits)
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except Exception as error:
        raise ResumePreflightError(f"could not load checkpoint: {path}") from error
    state, stored_logits, optimizer_state = _decode_checkpoint_payload(
        payload,
        expected_config_hash=expected_config_hash,
        logits=live_logits,
    )
    old_logits = live_logits.detach().clone()
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    old_rng = _capture_rng_state()
    try:
        optimizer.load_state_dict(optimizer_state)
        _validate_optimizer_owns_logits(optimizer, live_logits)
        _optimizer_state_to_device(optimizer, live_logits.device)
        with torch.no_grad():
            live_logits.copy_(stored_logits.to(device=live_logits.device))
        _restore_rng_state(state.rng_state)
    except Exception as error:
        optimizer.load_state_dict(old_optimizer)
        _optimizer_state_to_device(optimizer, live_logits.device)
        with torch.no_grad():
            live_logits.copy_(old_logits)
        _restore_rng_state(old_rng)
        raise ResumePreflightError("checkpoint restoration failed atomically") from error
    return state


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _run_exclusive_lock(run_dir: Path):
    """Serialize one run's compare/validate/publish critical sections."""

    directory = Path(run_dir).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("run lock target must be a directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(directory, flags)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _canonical_transaction_path(value: object) -> str:
    if not isinstance(value, str) or _TRANSACTION_PATH.fullmatch(value) is None:
        raise ResumePreflightError(
            "transaction path must be canonical transactions/step_[0-9]{8}"
        )
    return value


def _safe_transactions_directory(run_dir: Path, *, create: bool = False) -> Path:
    root = Path(run_dir).resolve(strict=True)
    transactions = root / "transactions"
    if transactions.is_symlink():
        raise ResumePreflightError("transactions directory must not be a symlink")
    if create:
        transactions.mkdir(exist_ok=True)
    if not transactions.is_dir():
        raise ResumePreflightError("transactions directory is missing")
    if transactions.resolve(strict=True).parent != root:
        raise ResumePreflightError("transactions directory resolves outside run_dir")
    return transactions


def _safe_transaction_generation(run_dir: Path, transaction: object) -> tuple[str, Path]:
    relative = _canonical_transaction_path(transaction)
    transactions = _safe_transactions_directory(run_dir)
    generation = Path(run_dir).resolve(strict=True) / relative
    if generation.is_symlink():
        raise ResumePreflightError("transaction generation must not be a symlink")
    if not generation.is_dir():
        raise ResumePreflightError("referenced transaction generation is missing")
    if generation.resolve(strict=True).parent != transactions.resolve(strict=True):
        raise ResumePreflightError("transaction generation resolves outside run_dir")
    return relative, generation


def _safe_transaction_file(generation: Path, name: str) -> Path:
    if name not in _TRANSACTION_PAYLOAD_NAMES:
        raise ValueError(f"unknown transaction payload name: {name}")
    path = generation / name
    if path.is_symlink():
        raise ResumePreflightError(f"transaction {name} must not be a symlink")
    if not path.is_file():
        raise ResumePreflightError(f"transaction {name} is missing")
    if path.resolve(strict=True).parent != generation.resolve(strict=True):
        raise ResumePreflightError(f"transaction {name} resolves outside its generation")
    return path


def _history_record(payload: object) -> dict[str, object]:
    expected = {field.name for field in fields(StepRecord)} | {"effective_loss"}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("history row must have the exact lifecycle schema")
    record_payload = {field.name: payload[field.name] for field in fields(StepRecord)}
    record = _step_record_from_mapping(record_payload)
    if record.completed_updates_after != record.global_step_before + 1:
        raise ValueError("history row update identities are inconsistent")
    if record.loss_dic is None:
        if record.dic_scale != 0.0:
            raise ValueError("inactive DIC loss is legal only with zero scale")
    elif record.dic_scale < 0.0:
        raise ValueError("DIC scale must be non-negative")
    if record.loss_dts is None:
        if record.dts_scale != 0.0 or record.dts_endpoint is not None:
            raise ValueError("inactive DTS loss is legal only with zero scale and null endpoint")
    elif record.dts_endpoint is None or record.dts_scale < 0.0:
        raise ValueError("active DTS loss requires an endpoint and non-negative scale")
    effective = payload["effective_loss"]
    if isinstance(effective, bool) or not isinstance(effective, Real):
        raise ValueError("history effective_loss must be a finite float")
    effective_value = float(effective)
    expected_effective = record.loss_base
    if record.loss_dic is not None:
        expected_effective += record.dic_scale * record.loss_dic
    if record.loss_dts is not None:
        expected_effective += record.dts_scale * record.loss_dts
    if not math.isfinite(effective_value) or not math.isclose(
        effective_value, expected_effective, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("history effective_loss does not match unweighted losses and scales")
    result = asdict(record)
    result["effective_loss"] = effective_value
    return result


def _history_segment_bytes(history_segment: Sequence[Mapping[str, object]]) -> tuple[bytes, list[dict[str, object]]]:
    if isinstance(history_segment, (str, bytes)) or not isinstance(
        history_segment, Sequence
    ):
        raise ValueError("history_segment must be a sequence of mappings")
    rows = [_history_record(row) for row in history_segment]
    encoded = b"".join(
        json.dumps(
            normalize_builtin(row),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    return encoded, rows


def _validate_checkpoint_for_commit(
    payload: object,
) -> tuple[int, StepRecord | None, str]:
    required = {
        "schema",
        "config_hash",
        "completed_updates",
        "logits",
        "optimizer",
        "last_record",
        "rng_state",
        "gradient_samples",
        "step_times",
        "peak_cuda_bytes",
        "calibration_records",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("checkpoint payload must have the exact v3 checkpoint schema")
    if payload["schema"] != _CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint payload schema mismatch")
    completed, record = _validate_step_last_record(
        payload["completed_updates"], payload["last_record"]
    )
    digest = _validate_digest(payload["config_hash"], name="checkpoint config_hash")
    stored_logits = payload["logits"]
    if (
        not isinstance(stored_logits, torch.Tensor)
        or not stored_logits.is_floating_point()
        or stored_logits.ndim != 5
        or stored_logits.shape[:2] != (1, 1)
        or stored_logits.shape[2] == 0
        or stored_logits.shape[2:] != (stored_logits.shape[2],) * 3
        or not torch.isfinite(stored_logits).all().item()
    ):
        raise ValueError("checkpoint logits are malformed")
    if not isinstance(payload["optimizer"], Mapping):
        raise ValueError("checkpoint optimizer state must be a mapping")
    _validate_rng_state(payload["rng_state"])
    samples = _validated_samples(payload["gradient_samples"])
    if any(sample.completed_updates > completed for sample in samples):
        raise ValueError("checkpoint gradient sample is ahead of the committed update")
    times = _validated_times(payload["step_times"])
    if len(times) != completed:
        raise ValueError("checkpoint step-time count must match completed updates")
    peak = payload["peak_cuda_bytes"]
    if isinstance(peak, bool) or not isinstance(peak, Integral) or peak < 0:
        raise ValueError("checkpoint peak CUDA bytes must be a non-negative integer")
    _validated_calibration_records(
        payload["calibration_records"],
        samples=samples,
        completed_updates=completed,
    )
    return completed, record, digest


def _read_pointer(run_dir: Path) -> dict[str, str]:
    path = run_dir / "checkpoint_latest.json"
    if path.is_symlink():
        raise ResumePreflightError("checkpoint pointer must not be a symlink")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResumePreflightError("checkpoint_latest.json is missing or malformed") from error
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "transaction",
        "manifest_sha256",
    }:
        raise ResumePreflightError("checkpoint pointer has the wrong schema")
    if payload["schema"] != _POINTER_SCHEMA:
        raise ResumePreflightError("checkpoint pointer schema mismatch")
    transaction = _canonical_transaction_path(payload["transaction"])
    digest = _validate_digest(payload["manifest_sha256"], name="manifest_sha256")
    return {
        "schema": _POINTER_SCHEMA,
        "transaction": transaction,
        "manifest_sha256": digest,
    }


def _manifest_metadata(path: Path, expected_sha256: str) -> dict[str, object]:
    if sha256_file(path) != expected_sha256:
        raise ResumePreflightError("transaction manifest hash mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ResumePreflightError("transaction manifest is malformed") from error
    required = {
        "schema",
        "transaction",
        "completed_updates",
        "config_hash",
        "checkpoint_sha256",
        "history_segment_sha256",
        "previous_transaction",
        "previous_transaction_sha256",
        "history_interval",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ResumePreflightError("transaction manifest has the wrong schema")
    if payload["schema"] != _TRANSACTION_SCHEMA:
        raise ResumePreflightError("transaction manifest schema mismatch")
    return dict(payload)


def commit_training_transaction(
    run_dir: Path,
    *,
    checkpoint_payload: Mapping[str, object],
    history_segment: Sequence[Mapping[str, object]],
    previous_transaction_sha256: str | None,
) -> Path:
    """Commit one fsynced append-only checkpoint/history transaction generation."""

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError("run_dir must already exist")
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("checkpoint_payload must be a mapping")
    completed, last, config_digest = _validate_checkpoint_for_commit(
        checkpoint_payload
    )
    rows_bytes, rows = _history_segment_bytes(history_segment)
    with _run_exclusive_lock(run_dir):
        return _commit_training_transaction_locked(
            run_dir,
            checkpoint_payload=checkpoint_payload,
            completed=completed,
            last=last,
            config_digest=config_digest,
            rows_bytes=rows_bytes,
            rows=rows,
            previous_transaction_sha256=previous_transaction_sha256,
        )


def _commit_training_transaction_locked(
    run_dir: Path,
    *,
    checkpoint_payload: Mapping[str, object],
    completed: int,
    last: StepRecord | None,
    config_digest: str,
    rows_bytes: bytes,
    rows: list[dict[str, object]],
    previous_transaction_sha256: str | None,
) -> Path:
    """Validate the live predecessor and publish while the run lock is held."""

    if completed == 0:
        if rows or previous_transaction_sha256 is not None:
            raise ValueError("step-zero transaction must have empty history and no predecessor")
        if (run_dir / "checkpoint_latest.json").exists() or (
            run_dir / "checkpoint_latest.json"
        ).is_symlink():
            raise ValueError("step-zero transaction requires an absent pointer")
        previous_transaction = None
    else:
        if not rows:
            raise ValueError("positive transaction must contain its contiguous history segment")
        previous_digest = _validate_digest(
            previous_transaction_sha256, name="previous_transaction_sha256"
        )
        pointer = _read_pointer(run_dir)
        if pointer["manifest_sha256"] != previous_digest:
            raise ValueError("transaction predecessor is not the currently committed pointer")
        previous_transaction, previous_generation = _safe_transaction_generation(
            run_dir, pointer["transaction"]
        )
        previous_manifest = _manifest_metadata(
            _safe_transaction_file(previous_generation, "transaction_manifest.json"),
            previous_digest,
        )
        if previous_manifest["config_hash"] != config_digest:
            raise ValueError("transaction predecessor configuration mismatch")
        previous_completed = previous_manifest["completed_updates"]
        if isinstance(previous_completed, bool) or not isinstance(previous_completed, int):
            raise ValueError("previous transaction completed_updates is malformed")
        expected_steps = list(range(previous_completed + 1, completed + 1))
        observed_steps = [int(row["completed_updates_after"]) for row in rows]
        if observed_steps != expected_steps:
            raise ValueError("transaction history interval is not contiguous")
        assert last is not None
        last_row_record = {
            field.name: rows[-1][field.name] for field in fields(StepRecord)
        }
        if asdict(last) != last_row_record:
            raise ValueError("checkpoint last_record does not match the final history row")
    first = None if not rows else int(rows[0]["completed_updates_after"])
    final = None if not rows else int(rows[-1]["completed_updates_after"])
    transactions = _safe_transactions_directory(run_dir, create=True)
    generation_name = f"step_{completed:08d}"
    generation = transactions / generation_name
    if generation.exists() or generation.is_symlink():
        raise FileExistsError(f"transaction generation already exists: {generation}")
    temporary = transactions / f".{generation_name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        atomic_checkpoint(temporary / "checkpoint.pt", checkpoint_payload)
        atomic_text(temporary / "history_segment.jsonl", rows_bytes.decode("utf-8"))
        manifest = {
            "schema": _TRANSACTION_SCHEMA,
            "transaction": f"transactions/{generation_name}",
            "completed_updates": completed,
            "config_hash": config_digest,
            "checkpoint_sha256": sha256_file(temporary / "checkpoint.pt"),
            "history_segment_sha256": sha256_file(temporary / "history_segment.jsonl"),
            "previous_transaction": previous_transaction,
            "previous_transaction_sha256": previous_transaction_sha256,
            "history_interval": {
                "first_completed_update": first,
                "last_completed_update": final,
                "row_count": len(rows),
            },
        }
        atomic_json(temporary / "transaction_manifest.json", manifest)
        manifest_digest = sha256_file(temporary / "transaction_manifest.json")
        _fsync_directory(temporary)
        os.rename(temporary, generation)
        _fsync_directory(transactions)
        atomic_json(
            run_dir / "checkpoint_latest.json",
            {
                "schema": _POINTER_SCHEMA,
                "transaction": f"transactions/{generation_name}",
                "manifest_sha256": manifest_digest,
            },
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return generation


def _read_history_segment(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ResumePreflightError("transaction history segment is unreadable") from error
    for line in lines:
        if not line:
            raise ResumePreflightError("transaction history contains a blank row")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ResumePreflightError("transaction history contains malformed JSON") from error
        rows.append(_history_record(payload))
    return rows


def _validate_generation(
    run_dir: Path,
    transaction: str,
    manifest_sha256: str,
    *,
    expected_config_hash: str,
    logits: torch.Tensor | None = None,
) -> tuple[dict[str, object], int, StepRecord | None, list[dict[str, object]]]:
    transaction, generation = _safe_transaction_generation(run_dir, transaction)
    manifest_path = _safe_transaction_file(generation, "transaction_manifest.json")
    checkpoint_path = _safe_transaction_file(generation, "checkpoint.pt")
    history_path = _safe_transaction_file(generation, "history_segment.jsonl")
    manifest = _manifest_metadata(manifest_path, manifest_sha256)
    if manifest["transaction"] != transaction:
        raise ResumePreflightError("transaction manifest path identity mismatch")
    if manifest["config_hash"] != expected_config_hash:
        raise ResumePreflightError("transaction configuration hash mismatch")
    checkpoint_digest = _validate_digest(
        manifest["checkpoint_sha256"], name="transaction checkpoint hash"
    )
    history_digest = _validate_digest(
        manifest["history_segment_sha256"], name="transaction history hash"
    )
    if sha256_file(checkpoint_path) != checkpoint_digest:
        raise ResumePreflightError("transaction checkpoint hash mismatch")
    if sha256_file(history_path) != history_digest:
        raise ResumePreflightError("transaction history hash mismatch")
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as error:
        raise ResumePreflightError("transaction checkpoint is unreadable") from error
    try:
        completed, last_record, checkpoint_config = _validate_checkpoint_for_commit(
            checkpoint
        )
    except (TypeError, ValueError) as error:
        raise ResumePreflightError("transaction checkpoint payload is malformed") from error
    if checkpoint_config != expected_config_hash:
        raise ResumePreflightError("transaction checkpoint configuration mismatch")
    if logits is not None:
        decoded, _, _ = _decode_checkpoint_payload(
            checkpoint, expected_config_hash=expected_config_hash, logits=logits
        )
        if decoded.completed_updates != completed or decoded.last_record != last_record:
            raise ResumePreflightError("transaction checkpoint identity is inconsistent")
    if manifest["completed_updates"] != completed:
        raise ResumePreflightError("transaction checkpoint step mismatch")
    if transaction != f"transactions/step_{completed:08d}":
        raise ResumePreflightError("transaction path does not match its completed step")
    rows = _read_history_segment(history_path)
    interval = manifest["history_interval"]
    if not isinstance(interval, Mapping) or set(interval) != {
        "first_completed_update",
        "last_completed_update",
        "row_count",
    }:
        raise ResumePreflightError("transaction history interval is malformed")
    if interval["row_count"] != len(rows):
        raise ResumePreflightError("transaction history row count mismatch")
    if completed == 0:
        if rows or interval["first_completed_update"] is not None or interval[
            "last_completed_update"
        ] is not None:
            raise ResumePreflightError("step-zero transaction interval must be empty")
    else:
        if not rows:
            raise ResumePreflightError("positive transaction has no history rows")
        observed = [int(row["completed_updates_after"]) for row in rows]
        if observed != list(range(observed[0], completed + 1)):
            raise ResumePreflightError("transaction history interval is not contiguous")
        if interval["first_completed_update"] != observed[0] or interval[
            "last_completed_update"
        ] != observed[-1]:
            raise ResumePreflightError("transaction history interval metadata mismatch")
        assert last_record is not None
        if asdict(last_record) != {
            field.name: rows[-1][field.name] for field in fields(StepRecord)
        }:
            raise ResumePreflightError("checkpoint last_record and history tail differ")
    return manifest, completed, last_record, rows


def _verify_transaction_chain(
    run_dir: Path,
    *,
    expected_config_hash: str | None = None,
    expected_final_step: int | None = None,
    logits: torch.Tensor | None = None,
) -> _VerifiedTransactionChain:
    """Nonmutatingly validate every linked generation and its semantic payloads."""

    run_dir = Path(run_dir).resolve(strict=True)
    pointer = _read_pointer(run_dir)
    transaction = pointer["transaction"]
    manifest_digest = pointer["manifest_sha256"]
    if expected_config_hash is None:
        _, head_generation = _safe_transaction_generation(run_dir, transaction)
        head_manifest = _manifest_metadata(
            _safe_transaction_file(head_generation, "transaction_manifest.json"),
            manifest_digest,
        )
        expected = _validate_digest(
            head_manifest.get("config_hash"), name="transaction config_hash"
        )
    else:
        expected = _validate_digest(
            expected_config_hash, name="expected_config_hash"
        )
    reverse_chain: list[
        tuple[str, str, dict[str, object], int, list[dict[str, object]]]
    ] = []
    seen_digests: set[str] = set()
    seen_transactions: set[str] = set()
    while True:
        transaction = _canonical_transaction_path(transaction)
        if manifest_digest in seen_digests or transaction in seen_transactions:
            raise ResumePreflightError("transaction predecessor chain contains a cycle")
        seen_digests.add(manifest_digest)
        seen_transactions.add(transaction)
        manifest, completed, _, rows = _validate_generation(
            run_dir,
            transaction,
            manifest_digest,
            expected_config_hash=expected,
            logits=logits,
        )
        reverse_chain.append((transaction, manifest_digest, manifest, completed, rows))
        previous_transaction = manifest["previous_transaction"]
        previous_digest = manifest["previous_transaction_sha256"]
        if previous_transaction is None or previous_digest is None:
            if previous_transaction is not None or previous_digest is not None:
                raise ResumePreflightError("transaction predecessor identity is incomplete")
            break
        transaction = _canonical_transaction_path(previous_transaction)
        manifest_digest = _validate_digest(
            previous_digest, name="previous_transaction_sha256"
        )
    chain = list(reversed(reverse_chain))
    if chain[0][3] != 0 or chain[0][0] != "transactions/step_00000000":
        raise ResumePreflightError("transaction chain does not begin at committed step zero")
    history: list[dict[str, object]] = []
    previous_completed = 0
    for index, (current_path, _, manifest, completed, rows) in enumerate(chain):
        if index == 0:
            if (
                manifest["previous_transaction"] is not None
                or manifest["previous_transaction_sha256"] is not None
            ):
                raise ResumePreflightError("step-zero transaction has a predecessor")
        else:
            prior_path, prior_digest, _, _, _ = chain[index - 1]
            if (
                manifest["previous_transaction"] != prior_path
                or manifest["previous_transaction_sha256"] != prior_digest
            ):
                raise ResumePreflightError("transaction predecessor identity mismatch")
            if not rows or rows[0]["completed_updates_after"] != previous_completed + 1:
                raise ResumePreflightError("transaction chain has a history gap or overlap")
        if current_path != f"transactions/step_{completed:08d}":
            raise ResumePreflightError("transaction path identity mismatch")
        history.extend(rows)
        previous_completed = completed
    final_transaction, final_digest, _, final_completed, _ = chain[-1]
    if final_completed != len(history):
        raise ResumePreflightError("verified history prefix length mismatches checkpoint step")
    if expected_final_step is not None and final_completed != expected_final_step:
        raise ResumePreflightError("transaction final step does not match the requested run")
    files = tuple(
        sorted(
            f"{path}/{name}"
            for path, _, _, _, _ in chain
            for name in _TRANSACTION_PAYLOAD_NAMES
        )
    )
    return _VerifiedTransactionChain(
        final_transaction=final_transaction,
        final_manifest_sha256=final_digest,
        completed_updates=final_completed,
        history=tuple(history),
        files=files,
    )


def restore_latest_transaction(
    run_dir: Path,
    *,
    expected_config_hash: str,
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> ResumeState:
    """Walk and rehash the complete pointer-linked chain before restoring state."""

    run_dir = Path(run_dir).resolve()
    expected = _validate_digest(expected_config_hash, name="expected_config_hash")
    live_logits = _validate_logits(logits)
    _validate_optimizer(optimizer)
    _validate_optimizer_owns_logits(optimizer, live_logits)
    verified = _verify_transaction_chain(
        run_dir,
        expected_config_hash=expected,
        logits=live_logits,
    )
    _, final_generation = _safe_transaction_generation(
        run_dir, verified.final_transaction
    )
    restored = restore_checkpoint_payload(
        _safe_transaction_file(final_generation, "checkpoint.pt"),
        expected_config_hash=expected,
        logits=live_logits,
        optimizer=optimizer,
    )
    return ResumeState(
        completed_updates=restored.completed_updates,
        last_record=restored.last_record,
        rng_state=restored.rng_state,
        gradient_samples=restored.gradient_samples,
        step_times=restored.step_times,
        peak_cuda_bytes=restored.peak_cuda_bytes,
        calibration_records=restored.calibration_records,
        committed_transaction=verified.final_transaction,
        committed_transaction_sha256=verified.final_manifest_sha256,
        committed_history=verified.history,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _load_bound_targets(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    """Decode the exact x/y/z bytes captured and hashed by resolved_config."""

    snapshots = getattr(args, "_soft_loss_v3_shadow_snapshots", None)
    if not isinstance(snapshots, tuple) or len(snapshots) != 3 or not all(
        isinstance(snapshot, _ShadowInputSnapshot) for snapshot in snapshots
    ):
        raise ValueError("resolved shadow byte snapshots are missing")
    expected_paths = tuple(str(Path(path).resolve()) for path in args.shadow_files)
    if tuple(snapshot.path for snapshot in snapshots) != expected_paths:
        raise ValueError("resolved shadow snapshot paths no longer match x/y/z inputs")
    decoded: list[np.ndarray] = []
    for snapshot in snapshots:
        if hashlib.sha256(snapshot.payload).hexdigest() != snapshot.sha256:
            raise ValueError("resolved shadow byte snapshot hash mismatch")
        image = cv2.imdecode(
            np.frombuffer(snapshot.payload, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if image is None:
            raise ValueError(f"shadow snapshot is not a decodable grayscale image: {snapshot.path}")
        decoded.append(image)
    images = [
        normalize_target(image, args.image_size, False) for image in decoded
    ]
    images.extend(
        normalize_target(cv2.flip(image, 1), args.image_size, False)
        for image in decoded
    )
    array = np.stack(images).astype(np.float32) / 255.0
    return torch.from_numpy(array).to(device)


def _build_runtime(
    args: argparse.Namespace, device: torch.device, profile: V3Profile
) -> _Runtime:
    targets = _load_bound_targets(args, device)
    if targets.shape != (6, args.image_size, args.image_size):
        raise ValueError("fixed protocol must produce exactly six target views")
    protocol = SadekarTFSProtocol(
        device=device,
        image_size=args.image_size,
        zdist=args.sadekar_zdist,
        volume_extent_world=args.sadekar_volume_extent,
        n_pts_per_ray=args.sadekar_npts_per_ray,
    )
    logits = torch.full(
        (1, 1, args.res, args.res, args.res),
        -4.0,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([logits], lr=args.lr)
    bands = None
    if profile.use_dic or bool(args.calibration_probe):
        bands = make_contour_bands(targets >= 0.5)
    return _Runtime(
        logits=logits,
        optimizer=optimizer,
        protocol=protocol,
        targets=targets,
        bands=bands,
    )


def _validate_runtime(runtime: object) -> _Runtime:
    if not isinstance(runtime, _Runtime):
        raise ValueError("runtime factory must return _Runtime")
    logits = _validate_logits(runtime.logits)
    _validate_optimizer(runtime.optimizer)
    _validate_optimizer_owns_logits(runtime.optimizer, logits)
    _validate_protocol(runtime.protocol, logits)
    _validate_targets(runtime.targets, logits)
    return runtime


def _history_from_step(record: object, completed_updates: int) -> dict[str, object]:
    if not isinstance(record, StepRecord):
        raise SoftTrainingStepError("train_one_update must return a StepRecord")
    if (
        record.global_step_before != completed_updates
        or record.completed_updates_after != completed_updates + 1
    ):
        raise SoftTrainingStepError("returned StepRecord update identity is invalid")
    payload = asdict(record)
    effective = record.loss_base
    if record.loss_dic is None:
        if record.dic_scale != 0.0:
            raise SoftTrainingStepError("inactive DIC loss has a nonzero scale")
    else:
        effective += record.dic_scale * record.loss_dic
    if record.loss_dts is None:
        if record.dts_scale != 0.0 or record.dts_endpoint is not None:
            raise SoftTrainingStepError("inactive DTS loss has active telemetry")
    else:
        if record.dts_endpoint is None:
            raise SoftTrainingStepError("active DTS loss has no endpoint")
        effective += record.dts_scale * record.loss_dts
    if not math.isfinite(effective):
        raise SoftTrainingStepError("effective loss must be finite")
    payload["effective_loss"] = float(effective)
    return _history_record(payload)


def _probe_identity(
    completed_updates: int, total_steps: int, profile: V3Profile
) -> tuple[float, float | None]:
    if completed_updates <= 0 or completed_updates > total_steps:
        raise ValueError("probe completed update must be in [1,total_steps]")
    if completed_updates < total_steps:
        progress = (completed_updates + 1) / float(total_steps)
        parity_index = completed_updates % 2
    else:
        progress = 1.0
        parity_index = (completed_updates - 1) % 2
    endpoint = None
    if profile.use_dts:
        assert profile.dts_thresholds is not None
        endpoint = float(profile.dts_thresholds[parity_index])
    return float(progress), endpoint


def _active_diagnostic_invalid_flags(
    sample: object, profile: V3Profile
) -> list[str]:
    """Label only invalid diagnostics belonging to profile-active components."""

    flags: list[str] = []
    if profile.use_dic:
        for name in ("base_dic_cosine_invalid", "weighted_dic_base_ratio_invalid"):
            if bool(getattr(sample, name)):
                flags.append(name)
    if profile.use_dts:
        selected = getattr(sample, "selected_dts_tau", None)
        for name in ("base_dts_cosine_invalid", "weighted_dts_base_ratio_invalid"):
            mapping = getattr(sample, name)
            if selected in mapping and bool(mapping[selected]):
                flags.append(f"{name}[{selected}]")
    if profile.use_dic and profile.use_dts:
        selected = getattr(sample, "selected_dts_tau", None)
        for name in ("dic_dts_cosine_invalid", "dic_dts_support_iou_invalid"):
            mapping = getattr(sample, name)
            if selected in mapping and bool(mapping[selected]):
                flags.append(f"{name}[{selected}]")
    if profile.use_dic or profile.use_dts:
        for name in (
            "combined_aux_base_ratio_invalid",
            "base_combined_aux_cosine_invalid",
        ):
            if bool(getattr(sample, name)):
                flags.append(name)
    return flags


def _gradient_probe(
    runtime: _Runtime,
    *,
    completed_updates: int,
    total_steps: int,
    profile: V3Profile,
    lambda_dic: float,
    lambda_dts: float,
    calibration: bool,
) -> ComponentGradientSample:
    progress, endpoint = _probe_identity(completed_updates, total_steps, profile)
    include_dic = calibration or profile.use_dic
    include_dts = calibration or profile.use_dts
    if include_dic and runtime.bands is None:
        raise ValueError("DIC gradient probe requires contour bands")

    def base_builder() -> torch.Tensor:
        density = torch.sigmoid(runtime.logits)
        projections = runtime.protocol.render_all(density)
        return compute_baseline_branch(projections, runtime.targets, profile)

    dic_builder = None
    if include_dic:
        def build_dic() -> torch.Tensor:
            density = torch.sigmoid(runtime.logits)
            readout = render_thresholded(runtime.protocol, density, 0.05)
            assert runtime.bands is not None
            return compute_dic_branch(readout.score, runtime.bands)

        dic_builder = build_dic
    dts_builders: dict[float, object] = {}
    dts_thresholds: tuple[float, ...] = ()
    if calibration:
        dts_thresholds = (0.04, 0.06)
        endpoint = 0.04
    elif include_dts:
        assert profile.dts_thresholds is not None
        dts_thresholds = tuple(dict.fromkeys(profile.dts_thresholds))
    for threshold in dts_thresholds:
        def build_dts(tau: float = float(threshold)) -> torch.Tensor:
            density = torch.sigmoid(runtime.logits)
            readout = render_thresholded(runtime.protocol, density, tau)
            return compute_dts_branch(readout.score, runtime.targets >= 0.5)

        dts_builders[float(threshold)] = build_dts
    dic_scale = (
        0.0
        if calibration or not profile.use_dic
        else lambda_dic * dic_schedule(progress)
    )
    dts_scale = (
        0.0
        if calibration or not profile.use_dts
        else lambda_dts * dts_schedule(progress)
    )
    return measure_component_gradients(
        runtime.logits,
        completed_updates=completed_updates,
        base_builder=base_builder,
        dic_builder=dic_builder,
        dts_builders=dts_builders,
        dic_scale=dic_scale,
        dts_scale=dts_scale,
        selected_dts_tau=endpoint if include_dts else None,
    )


def _should_probe(
    completed_updates: int,
    total_steps: int,
    profile: V3Profile,
    *,
    calibration: bool,
) -> bool:
    if calibration:
        return completed_updates in _CALIBRATION_UPDATES
    if completed_updates % 50 != 0 or not (profile.use_dic or profile.use_dts):
        return False
    progress, _ = _probe_identity(completed_updates, total_steps, profile)
    return (profile.use_dic and dic_schedule(progress) > 0.0) or (
        profile.use_dts and dts_schedule(progress) > 0.0
    )


def _validate_resume_compatibility(
    args: argparse.Namespace, profile: V3Profile, state: ResumeState
) -> None:
    """Reject ahead or shifted scientific state before any resume-side write."""

    if state.completed_updates > args.steps:
        raise ResumePreflightError("checkpoint is ahead of requested final step")
    expected_probes = [
        update
        for update in range(1, state.completed_updates + 1)
        if _should_probe(
            update,
            args.steps,
            profile,
            calibration=bool(args.calibration_probe),
        )
    ]
    observed_probes = [sample.completed_updates for sample in state.gradient_samples]
    if observed_probes != expected_probes:
        raise ResumePreflightError("checkpoint gradient probe identities are incompatible")
    calibration_updates = [
        record.get("completed_updates") for record in state.calibration_records
    ]
    if bool(args.calibration_probe):
        if calibration_updates != expected_probes:
            raise ResumePreflightError(
                "checkpoint calibration probe identities are incompatible"
            )
    elif calibration_updates:
        raise ResumePreflightError(
            "normal checkpoint must not contain calibration probe records"
        )


def _calibration_record(sample: ComponentGradientSample) -> dict[str, object]:
    if sample.dic_norm is None or set(sample.dts_norms) != {0.04, 0.06}:
        raise ValueError("calibration sample must contain DIC and both DTS endpoints")
    return {
        "completed_updates": sample.completed_updates,
        "g0": sample.base_norm,
        "g_dic": sample.dic_norm,
        "g_dts_004": sample.dts_norms[0.04],
        "g_dts_006": sample.dts_norms[0.06],
    }


_timer = time.perf_counter


def _post_update_hook(completed_updates: int) -> None:
    """Private deterministic process-interruption seam; production is a no-op."""

    del completed_updates


def _failure_diagnostic_path(run_dir: Path) -> Path:
    root = Path(run_dir).resolve(strict=True)
    return root.parent / f".{root.name}.failure.json"


def _failure_diagnostic_relative(run_dir: Path) -> str:
    return os.path.relpath(_failure_diagnostic_path(run_dir), Path(run_dir).resolve())


def _safe_failure_diagnostic_path(run_dir: Path) -> Path:
    root = Path(run_dir).resolve(strict=True)
    path = _failure_diagnostic_path(root)
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ResumePreflightError(
            "scientific failure diagnostic is missing"
        ) from error
    except OSError as error:
        raise ResumePreflightError(
            "scientific failure diagnostic is inaccessible"
        ) from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ResumePreflightError(
            "scientific failure diagnostic must not be a symlink"
        )
    if not stat.S_ISREG(metadata.st_mode):
        raise ResumePreflightError(
            "scientific failure diagnostic must be a regular file"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ResumePreflightError(
            "scientific failure diagnostic is inaccessible"
        ) from error
    if resolved.parent != root.parent or resolved.name != f".{root.name}.failure.json":
        raise ResumePreflightError(
            "scientific failure diagnostic is not the canonical safe sibling"
        )
    return path


def _validated_failure_diagnostic(
    run_dir: Path,
    *,
    expected_config_hash: str,
    requested_steps: int,
    expected_completed_updates: int | None = None,
) -> tuple[Path, dict[str, object]]:
    path = _safe_failure_diagnostic_path(run_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResumePreflightError(
            "scientific failure diagnostic is malformed"
        ) from error
    required = {
        "schema",
        "config_hash",
        "completed_updates",
        "error_type",
        "error",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ResumePreflightError(
            "scientific failure diagnostic has the wrong schema"
        )
    if payload["schema"] != "pcraft_soft_loss_v3_failure_v1":
        raise ResumePreflightError(
            "scientific failure diagnostic schema mismatch"
        )
    if payload["config_hash"] != expected_config_hash:
        raise ResumePreflightError(
            "scientific failure diagnostic configuration mismatch"
        )
    completed = payload["completed_updates"]
    if type(completed) is not int or completed < 0 or completed > requested_steps:
        raise ResumePreflightError(
            "scientific failure diagnostic completed update is outside requested bounds"
        )
    if expected_completed_updates is not None and completed != expected_completed_updates:
        raise ResumePreflightError(
            "scientific failure diagnostic and failed STATUS update identities differ"
        )
    for field in ("error_type", "error"):
        if type(payload[field]) is not str or not payload[field]:
            raise ResumePreflightError(
                f"scientific failure diagnostic {field} must be a non-empty string"
            )
    return path, dict(payload)


def _status(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResumePreflightError("STATUS.json is missing or malformed") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != _STATUS_SCHEMA:
        raise ResumePreflightError("STATUS.json schema mismatch")
    state = payload.get("state")
    if state not in ("running", "completed", "failed"):
        raise ResumePreflightError("STATUS.json state is invalid")
    common = {"schema", "state", "config_hash", "completed_updates"}
    state_keys = {
        "running": common,
        "completed": common | {"artifact_manifest_sha256"},
        "failed": common | {"diagnostic"},
    }
    if set(payload) != state_keys[state]:
        raise ResumePreflightError(f"STATUS.json {state} schema mismatch")
    try:
        _validate_digest(payload["config_hash"], name="STATUS config_hash")
    except ValueError as error:
        raise ResumePreflightError("STATUS.json config_hash is invalid") from error
    completed = payload["completed_updates"]
    if type(completed) is not int or completed < 0:
        raise ResumePreflightError(
            "STATUS.json completed_updates must be a non-negative integer"
        )
    if state == "running" and completed != 0:
        raise ResumePreflightError("running STATUS.json must identify step zero")
    if state == "completed":
        try:
            _validate_digest(
                payload["artifact_manifest_sha256"],
                name="STATUS artifact_manifest_sha256",
            )
        except ValueError as error:
            raise ResumePreflightError(
                "STATUS.json artifact manifest hash is invalid"
            ) from error
    if state == "failed" and (
        type(payload["diagnostic"]) is not str or not payload["diagnostic"]
    ):
        raise ResumePreflightError("STATUS.json diagnostic path is invalid")
    if state == "failed" and payload["diagnostic"] != _failure_diagnostic_relative(
        path.parent
    ):
        raise ResumePreflightError(
            "failed STATUS.json diagnostic path is not the canonical safe sibling"
        )
    return dict(payload)


def _cas_terminal_status(
    run_dir: Path,
    *,
    config_hash: str,
    state: str,
    completed_updates: int,
    artifact_manifest_sha256: str | None = None,
    diagnostic: str | None = None,
) -> None:
    with _run_exclusive_lock(run_dir):
        _cas_terminal_status_locked(
            run_dir,
            config_hash=config_hash,
            state=state,
            completed_updates=completed_updates,
            artifact_manifest_sha256=artifact_manifest_sha256,
            diagnostic=diagnostic,
        )


def _cas_terminal_status_locked(
    run_dir: Path,
    *,
    config_hash: str,
    state: str,
    completed_updates: int,
    artifact_manifest_sha256: str | None = None,
    diagnostic: str | None = None,
) -> None:
    """Perform terminal CAS while the caller holds the per-run lock."""

    current = _status(run_dir / "STATUS.json")
    if current.get("state") != "running" or current.get("config_hash") != config_hash:
        raise RuntimeError("STATUS compare-and-swap requires the compatible running state")
    if state == "completed":
        if artifact_manifest_sha256 is None:
            raise ValueError("completed status requires an artifact manifest hash")
        if os.path.lexists(_failure_diagnostic_path(run_dir)):
            _validated_failure_diagnostic(
                run_dir,
                expected_config_hash=config_hash,
                requested_steps=completed_updates,
            )
            raise RuntimeError(
                "completed STATUS cannot supersede a recorded scientific failure"
            )
        payload: dict[str, object] = {
            "schema": _STATUS_SCHEMA,
            "state": "completed",
            "config_hash": config_hash,
            "completed_updates": completed_updates,
            "artifact_manifest_sha256": artifact_manifest_sha256,
        }
    elif state == "failed":
        expected_diagnostic = _failure_diagnostic_relative(run_dir)
        if diagnostic != expected_diagnostic:
            raise ValueError(
                "failed status requires the canonical sibling diagnostic path"
            )
        _validated_failure_diagnostic(
            run_dir,
            expected_config_hash=config_hash,
            requested_steps=completed_updates,
            expected_completed_updates=completed_updates,
        )
        payload = {
            "schema": _STATUS_SCHEMA,
            "state": "failed",
            "config_hash": config_hash,
            "completed_updates": completed_updates,
            "diagnostic": diagnostic,
        }
    else:
        raise ValueError("terminal status must be completed or failed")
    atomic_json(run_dir / "STATUS.json", payload)


def _run_manifest(
    args: argparse.Namespace, config: Mapping[str, object], config_hash: str
) -> dict[str, object]:
    return {
        "schema": "pcraft_soft_loss_v3_run_v1",
        "config_hash": config_hash,
        "config": config,
        "profile": args.profile,
        "protocol": {
            "name": "sadekar_tfs_fixed_six_view",
            "camera_azimuth_deg": list(CAMERA_AZIMUTH_DEG),
            "camera_elevation_deg": list(CAMERA_ELEVATION_DEG),
            "projection_binary_rule": "protocol.binary_mask",
            "density_thresholds": list(_THRESHOLDS),
        },
        "scope": {
            "soft_only": True,
            "final_step_export_only": True,
            "downstream_geometry": False,
        },
    }


def _validate_existing_audits(
    run_dir: Path,
    args: argparse.Namespace,
    config: Mapping[str, object],
    config_hash: str,
) -> dict[str, object]:
    status = _status(run_dir / "STATUS.json")
    if status.get("config_hash") != config_hash:
        raise ResumePreflightError("existing STATUS configuration mismatch")
    expected_environment = environment_manifest(torch.device(args.device))
    for name in ("COMMAND.json", "run_manifest.json", "environment.json"):
        path = run_dir / name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResumePreflightError(f"existing {name} is missing or malformed") from error
        if not isinstance(payload, Mapping):
            raise ResumePreflightError(f"existing {name} must be a mapping")
        if name == "COMMAND.json":
            if (
                set(payload) != {"schema", "argv", "resolved_config", "config_hash"}
                or payload.get("schema") != "pcraft_soft_loss_v3_command_v1"
                or not isinstance(payload.get("argv"), list)
                or not all(isinstance(item, str) for item in payload["argv"])
                or payload.get("config_hash") != config_hash
                or payload.get("resolved_config") != config
            ):
                raise ResumePreflightError("existing COMMAND configuration mismatch")
        elif name == "run_manifest.json":
            if payload != _run_manifest(args, config, config_hash):
                raise ResumePreflightError(
                    "existing run manifest configuration mismatch"
                )
        elif payload != expected_environment:
            raise ResumePreflightError("existing environment manifest mismatch")
    return status


def _resume_status_after_failure_preflight(
    run_dir: Path,
    *,
    expected_config_hash: str,
    requested_steps: int,
) -> dict[str, object]:
    """Recover a durable pending failure before runtime construction or writes."""

    with _run_exclusive_lock(run_dir):
        current = _status(run_dir / "STATUS.json")
        if current["config_hash"] != expected_config_hash:
            raise ResumePreflightError(
                "existing STATUS configuration mismatch during failure preflight"
            )
        diagnostic_present = os.path.lexists(_failure_diagnostic_path(run_dir))
        if current["state"] != "failed" and not diagnostic_present:
            return current
        expected_completed = (
            int(current["completed_updates"])
            if current["state"] == "failed"
            else None
        )
        _, diagnostic = _validated_failure_diagnostic(
            run_dir,
            expected_config_hash=expected_config_hash,
            requested_steps=requested_steps,
            expected_completed_updates=expected_completed,
        )
        if current["state"] == "completed":
            raise ResumePreflightError(
                "completed STATUS conflicts with a recorded scientific failure"
            )
        if current["state"] == "failed":
            raise ResumePreflightError(
                "failed scientific failure is terminal and cannot resume"
            )
        completed = int(diagnostic["completed_updates"])
        _cas_terminal_status_locked(
            run_dir,
            config_hash=expected_config_hash,
            state="failed",
            completed_updates=completed,
            diagnostic=_failure_diagnostic_relative(run_dir),
        )
        raise ResumePreflightError(
            "pending scientific failure was recovered as terminal failed"
        )


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex}{path.suffix}"
    )
    try:
        with temporary.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_binary_png(path: Path, mask: torch.Tensor) -> None:
    array = mask.detach().to(device="cpu").numpy().astype(bool, copy=False)
    output = array.astype(np.uint8) * 255
    temporary = path.with_name(
        f".{path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex}{path.suffix}"
    )
    try:
        if not cv2.imwrite(str(temporary), output):
            raise OSError(f"failed to write image {path}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _link_or_copy_immutable(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"immutable checkpoint differs: {destination}")
        return
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _format_threshold(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def _primary_safety_aggregate(metrics: Mapping[str, object]) -> Mapping[str, object]:
    try:
        thresholds = metrics["thresholds"]
        assert isinstance(thresholds, Mapping)
        threshold = thresholds["0.05"]
        assert isinstance(threshold, Mapping)
        aggregate = threshold["aggregate"]
        assert isinstance(aggregate, Mapping)
        return aggregate
    except (AssertionError, KeyError, TypeError) as error:
        raise ValueError(
            'metrics must expose primary safety at thresholds["0.05"]["aggregate"]'
        ) from error


def _evaluate_and_write(
    run_dir: Path,
    runtime: _Runtime,
    *,
    args: argparse.Namespace,
    config_hash: str,
    last_record: StepRecord,
) -> dict[str, object]:
    with torch.no_grad():
        probability = torch.sigmoid(runtime.logits)
        binary_volumes: dict[float, torch.Tensor] = {}
        masks: dict[float, torch.Tensor] = {}
        for threshold in _THRESHOLDS:
            volume = probability >= threshold
            binary_volumes[threshold] = volume
            projection = runtime.protocol.render_all(
                volume.to(dtype=probability.dtype) * 0.999
            )
            masks[threshold] = runtime.protocol.binary_mask(projection)
    _atomic_npy(
        run_dir / "occupancy_probability.npy",
        probability[0, 0].detach().to(device="cpu").numpy().astype(np.float32),
    )
    for threshold, volume in binary_volumes.items():
        label = _format_threshold(threshold)
        _atomic_npy(
            run_dir / f"occupancy_binary_tau_{label}.npy",
            volume[0, 0].detach().to(device="cpu").numpy().astype(np.uint8),
        )
        for view_index in range(6):
            _atomic_binary_png(
                run_dir / f"binary_tau_{label}_view_{view_index}.png",
                masks[threshold][view_index],
            )
    report = evaluate_delivery_metrics(runtime.targets, masks)
    _primary_safety_aggregate(report)
    report.update(
        {
            "schema": "pcraft_soft_loss_v3_metrics_v1",
            "config_hash": config_hash,
            "profile": args.profile,
            "steps": args.steps,
            "final_step_exported": True,
            "final_loss": float(
                _history_from_step(last_record, last_record.global_step_before)[
                    "effective_loss"
                ]
            ),
        }
    )
    atomic_json(run_dir / "metrics.json", report)
    return report


def _summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "median_seconds": None, "p95_seconds": None}
    return {
        "count": len(values),
        "median_seconds": float(statistics.median(values)),
        "p95_seconds": float(np.percentile(np.asarray(values, dtype=float), 95)),
    }


def _resource_payload(
    step_times: Sequence[float], peak_cuda_bytes: int, device: torch.device
) -> dict[str, object]:
    physical = None
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        physical = int(torch.cuda.get_device_properties(index).total_memory)
    return {
        "schema": "pcraft_soft_loss_v3_resources_v1",
        "step_times": list(step_times),
        "peak_cuda_allocated_bytes": int(peak_cuda_bytes),
        "physical_device_memory_bytes": physical,
        "all_updates": _summary(step_times),
        "updates_21_end": _summary(step_times[20:]),
    }


def _reachable_transaction_files(
    run_dir: Path,
    *,
    expected_config_hash: str | None = None,
    expected_final_step: int | None = None,
) -> list[str]:
    verified = _verify_transaction_chain(
        run_dir,
        expected_config_hash=expected_config_hash,
        expected_final_step=expected_final_step,
    )
    return list(verified.files)


def _quarantine_unreferenced_transactions(run_dir: Path) -> None:
    """Move only unreachable generations aside after the live chain verifies."""

    run_dir = Path(run_dir).resolve(strict=True)
    with _run_exclusive_lock(run_dir):
        transactions = _safe_transactions_directory(run_dir)
        reachable = {
            Path(relative).parts[1]
            for relative in _reachable_transaction_files(run_dir)
        }
        unreferenced = [
            path
            for path in transactions.iterdir()
            if path.name not in reachable
        ]
        if not unreferenced:
            return
        quarantine = run_dir / "quarantine"
        if quarantine.is_symlink():
            raise ResumePreflightError("quarantine directory must not be a symlink")
        if quarantine.exists() and not quarantine.is_dir():
            raise ResumePreflightError("quarantine path must be a directory")
        for source in unreferenced:
            if source.is_symlink():
                raise ResumePreflightError(
                    "unreferenced transaction generation must not be a symlink"
                )
            if not source.is_dir():
                raise ResumePreflightError(
                    "unreferenced transaction generation must be a directory"
                )
            is_generation = re.fullmatch(r"step_[0-9]{8}", source.name) is not None
            is_temporary = _TEMPORARY_GENERATION_NAME.fullmatch(source.name) is not None
            if not (is_generation or is_temporary):
                raise ResumePreflightError(
                    "unreferenced transaction generation name is unsafe"
                )
            if source.resolve(strict=True).parent != transactions.resolve(strict=True):
                raise ResumePreflightError(
                    "unreferenced transaction generation resolves outside run_dir"
                )
            for descendant in source.rglob("*"):
                if descendant.is_symlink():
                    raise ResumePreflightError(
                        "unreferenced transaction payload must not be a symlink"
                    )
                if not descendant.resolve(strict=True).is_relative_to(
                    source.resolve(strict=True)
                ):
                    raise ResumePreflightError(
                        "unreferenced transaction payload resolves outside its generation"
                    )
        quarantine.mkdir(exist_ok=True)
        for source in unreferenced:
            destination = quarantine / (
                f"{source.name}.orphan.{os.getpid()}.{uuid.uuid4().hex}"
            )
            os.rename(source, destination)
        _fsync_directory(transactions)
        _fsync_directory(quarantine)


def _required_top_level_artifacts(*, calibration: bool) -> list[str]:
    names = [
        "COMMAND.json",
        "run_manifest.json",
        "environment.json",
        "history.jsonl",
        "occupancy_probability.npy",
        "metrics.json",
        "gradient_audit.json",
        "resource_metrics.json",
        "checkpoint_latest.json",
        "checkpoint_final.pt",
    ]
    names.extend(
        f"occupancy_binary_tau_{_format_threshold(threshold)}.npy"
        for threshold in _THRESHOLDS
    )
    names.extend(
        f"binary_tau_{_format_threshold(threshold)}_view_{view}.png"
        for threshold in _THRESHOLDS
        for view in range(6)
    )
    if calibration:
        names.extend(
            _calibration_checkpoint_name(update) for update in _CALIBRATION_UPDATES
        )
    return sorted(names)


def _preflight_reserved_top_level_paths(run_dir: Path) -> None:
    """Reject unsafe reserved entries before a resumed run can write anything."""

    root = Path(run_dir).resolve(strict=True)
    reserved = set(_required_top_level_artifacts(calibration=False))
    reserved.update(
        _calibration_checkpoint_name(update) for update in _CALIBRATION_UPDATES
    )
    reserved.update({"artifact_manifest.json", "STATUS.json"})
    for name in sorted(reserved):
        candidate = root / name
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ResumePreflightError(
                f"reserved artifact path is inaccessible: {name}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ResumePreflightError(
                f"reserved artifact path must not be a symlink: {name}"
            )
        if not stat.S_ISREG(metadata.st_mode):
            raise ResumePreflightError(
                f"reserved artifact path must be a regular file: {name}"
            )
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ResumePreflightError(
                f"reserved artifact path is inaccessible: {name}"
            ) from error
        if resolved.parent != root:
            raise ResumePreflightError(
                f"reserved artifact path resolves outside run_dir: {name}"
            )


def _safe_artifact_file(run_dir: Path, relative: object) -> Path:
    if type(relative) is not str or not relative:
        raise ResumePreflightError("artifact path must be a non-empty string")
    posix = PurePosixPath(relative)
    if (
        posix.is_absolute()
        or not posix.parts
        or any(part in ("", ".", "..") for part in posix.parts)
        or posix.as_posix() != relative
    ):
        raise ResumePreflightError("artifact path must be canonical and relative")
    root = Path(run_dir).resolve(strict=True)
    candidate = root
    for index, component in enumerate(posix.parts):
        candidate = candidate / component
        if candidate.is_symlink():
            raise ResumePreflightError(f"artifact path must not contain a symlink: {relative}")
        if index < len(posix.parts) - 1 and not candidate.is_dir():
            raise ResumePreflightError(f"artifact directory is missing: {relative}")
    if not candidate.is_file():
        raise ResumePreflightError(f"artifact file is missing: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ResumePreflightError(f"artifact file is inaccessible: {relative}") from error
    if not resolved.is_relative_to(root):
        raise ResumePreflightError(f"artifact resolves outside run_dir: {relative}")
    return candidate


def _validate_artifact_hashes(
    run_dir: Path, artifacts: Mapping[str, object]
) -> None:
    for relative, expected in artifacts.items():
        try:
            digest = _validate_digest(expected, name=f"artifact hash {relative}")
        except ValueError as error:
            raise ResumePreflightError(
                f"artifact hash is malformed: {relative}"
            ) from error
        path = _safe_artifact_file(run_dir, relative)
        if sha256_file(path) != digest:
            raise ResumePreflightError(f"completed artifact hash mismatch: {relative}")


def _artifact_manifest(
    run_dir: Path,
    config_hash: str,
    *,
    expected_final_step: int | None = None,
    calibration: bool = False,
) -> dict[str, object]:
    verified = _verify_transaction_chain(
        run_dir,
        expected_config_hash=config_hash,
        expected_final_step=expected_final_step,
    )
    relatives = sorted(
        set(_required_top_level_artifacts(calibration=calibration)) | set(verified.files)
    )
    hashes = {
        relative: sha256_file(_safe_artifact_file(run_dir, relative))
        for relative in relatives
    }
    return {
        "schema": "pcraft_soft_loss_v3_artifacts_v1",
        "config_hash": config_hash,
        "artifacts_sha256": hashes,
    }


def _write_or_validate_artifact_manifest(
    run_dir: Path, payload: Mapping[str, object]
) -> str:
    path = run_dir / "artifact_manifest.json"
    if path.is_symlink():
        raise ResumePreflightError("artifact manifest must not be a symlink")
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError("existing artifact manifest is malformed") from error
        if existing != normalize_builtin(payload):
            raise RuntimeError("immutable artifact manifest differs from recomputation")
    else:
        atomic_json(path, payload)
    digest = sha256_file(_safe_artifact_file(run_dir, "artifact_manifest.json"))
    artifacts = payload["artifacts_sha256"]
    assert isinstance(artifacts, Mapping)
    _validate_artifact_hashes(run_dir, artifacts)
    return digest


def _revalidate_terminal_publication(
    run_dir: Path,
    *,
    config_hash: str,
    final_step: int,
    artifacts: Mapping[str, object],
    artifact_manifest_sha256: str,
) -> None:
    """Recheck every immutable dependency immediately before terminal CAS."""

    _verify_transaction_chain(
        run_dir,
        expected_config_hash=config_hash,
        expected_final_step=final_step,
    )
    _validate_artifact_hashes(run_dir, artifacts)
    if sha256_file(
        _safe_artifact_file(run_dir, "artifact_manifest.json")
    ) != artifact_manifest_sha256:
        raise ResumePreflightError(
            "artifact manifest changed before terminal publication"
        )


def _validate_completed_run(
    run_dir: Path,
    status: Mapping[str, object],
    config_hash: str,
    total_steps: int,
    *,
    calibration: bool = False,
) -> dict[str, object]:
    verified = _verify_transaction_chain(
        run_dir,
        expected_config_hash=config_hash,
        expected_final_step=total_steps,
    )
    required_status = {
        "schema",
        "state",
        "config_hash",
        "completed_updates",
        "artifact_manifest_sha256",
    }
    if set(status) != required_status or status.get("state") != "completed":
        raise ResumePreflightError("completed STATUS schema mismatch")
    if status.get("config_hash") != config_hash or status.get(
        "completed_updates"
    ) != total_steps:
        raise ResumePreflightError("completed STATUS does not match this final-step run")
    manifest_path = _safe_artifact_file(run_dir, "artifact_manifest.json")
    if sha256_file(manifest_path) != status.get("artifact_manifest_sha256"):
        raise ResumePreflightError("completed artifact manifest hash mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ResumePreflightError("completed artifact manifest is malformed") from error
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {"schema", "config_hash", "artifacts_sha256"}
        or manifest.get("schema") != "pcraft_soft_loss_v3_artifacts_v1"
        or manifest.get("config_hash") != config_hash
        or not isinstance(manifest.get("artifacts_sha256"), Mapping)
    ):
        raise ResumePreflightError("completed artifact manifest schema mismatch")
    artifacts = manifest["artifacts_sha256"]
    required = set(_required_top_level_artifacts(calibration=calibration)) | set(
        verified.files
    )
    if set(artifacts) != required:
        raise ResumePreflightError(
            "completed artifact manifest must contain the exact canonical artifact set"
        )
    _validate_artifact_hashes(run_dir, artifacts)
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ResumePreflightError("completed metrics are malformed") from error
    if metrics.get("steps") != total_steps or metrics.get("final_step_exported") is not True:
        raise ResumePreflightError("completed metrics are not a final-step export")
    _primary_safety_aggregate(metrics)
    return dict(metrics)


def _materialize_history(run_dir: Path, history: Sequence[Mapping[str, object]]) -> None:
    rows_bytes, rows = _history_segment_bytes(history)
    if len(rows) != len(history):
        raise RuntimeError("history normalization changed the row count")
    atomic_text(run_dir / "history.jsonl", rows_bytes.decode("utf-8"))


def _calibration_checkpoint_name(completed_updates: int) -> str:
    return f"calibration_checkpoint_step_{completed_updates:08d}.pt"


def _retain_calibration_checkpoint(run_dir: Path, completed_updates: int) -> None:
    source = (
        run_dir
        / "transactions"
        / f"step_{completed_updates:08d}"
        / "checkpoint.pt"
    )
    if not source.is_file():
        raise RuntimeError("calibration checkpoint transaction is not committed")
    _link_or_copy_immutable(
        source, run_dir / _calibration_checkpoint_name(completed_updates)
    )


def _failure_diagnostic(
    run_dir: Path,
    *,
    config_hash: str,
    completed_updates: int,
    error: SoftTrainingStepError,
) -> Path:
    diagnostic = _failure_diagnostic_path(run_dir)
    payload = {
        "schema": "pcraft_soft_loss_v3_failure_v1",
        "config_hash": config_hash,
        "completed_updates": completed_updates,
        "error_type": type(error).__name__,
        "error": str(error),
    }
    if os.path.lexists(diagnostic):
        safe_diagnostic = _safe_failure_diagnostic_path(run_dir)
        try:
            existing = json.loads(safe_diagnostic.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ResumePreflightError(
                "existing scientific failure diagnostic is malformed"
            ) from error
        if existing != payload:
            raise RuntimeError("immutable sibling failure diagnostic differs")
    else:
        atomic_json(diagnostic, payload)
    return diagnostic


def _finalize_completed_run(
    run_dir: Path,
    *,
    runtime: _Runtime,
    args: argparse.Namespace,
    profile: V3Profile,
    device: torch.device,
    config_digest: str,
    completed_updates: int,
    last_record: StepRecord,
    committed_history: Sequence[Mapping[str, object]],
    gradient_samples: Sequence[ComponentGradientSample],
    calibration_records: Sequence[Mapping[str, object]],
    step_times: Sequence[float],
    peak_cuda_bytes: int,
) -> dict[str, object]:
    """Publish every final artifact and terminal STATUS under one run lock."""

    with _run_exclusive_lock(run_dir):
        current_status = _status(run_dir / "STATUS.json")
        if (
            current_status["state"] != "running"
            or current_status["config_hash"] != config_digest
        ):
            raise RuntimeError(
                "finalization requires the compatible running STATUS state"
            )
        if args.calibration_probe:
            for update in _CALIBRATION_UPDATES:
                _retain_calibration_checkpoint(run_dir, update)
        _materialize_history(run_dir, committed_history)
        pointer = _read_pointer(run_dir)
        _, final_generation = _safe_transaction_generation(
            run_dir, pointer["transaction"]
        )
        _link_or_copy_immutable(
            _safe_transaction_file(final_generation, "checkpoint.pt"),
            run_dir / "checkpoint_final.pt",
        )
        metrics = _evaluate_and_write(
            run_dir,
            runtime,
            args=args,
            config_hash=config_digest,
            last_record=last_record,
        )
        audit_samples = [asdict(sample) for sample in gradient_samples]
        calibration_audit_records: list[dict[str, object]] = []
        for raw_record in calibration_records:
            audit_record = dict(raw_record)
            update = audit_record.get("completed_updates")
            if isinstance(update, bool) or not isinstance(update, Integral):
                raise RuntimeError("calibration record update identity is malformed")
            checkpoint_name = _calibration_checkpoint_name(int(update))
            checkpoint_path = run_dir / checkpoint_name
            if not checkpoint_path.is_file():
                raise RuntimeError("calibration record checkpoint is missing")
            audit_record["checkpoint"] = checkpoint_name
            audit_record["checkpoint_sha256"] = sha256_file(checkpoint_path)
            calibration_audit_records.append(audit_record)
        atomic_json(
            run_dir / "gradient_audit.json",
            {
                "schema": "pcraft_soft_loss_v3_gradient_audit_v1",
                "config_hash": config_digest,
                "samples": audit_samples,
                "calibration_records": calibration_audit_records,
                "active_invalid_flags": [
                    {
                        "completed_updates": sample.completed_updates,
                        "flags": _active_diagnostic_invalid_flags(sample, profile),
                    }
                    for sample in gradient_samples
                ],
            },
        )
        atomic_json(
            run_dir / "resource_metrics.json",
            _resource_payload(step_times, peak_cuda_bytes, device),
        )
        artifact_payload = _artifact_manifest(
            run_dir,
            config_digest,
            expected_final_step=completed_updates,
            calibration=bool(args.calibration_probe),
        )
        artifact_digest = _write_or_validate_artifact_manifest(
            run_dir, artifact_payload
        )
        artifacts = artifact_payload["artifacts_sha256"]
        assert isinstance(artifacts, Mapping)
        _revalidate_terminal_publication(
            run_dir,
            config_hash=config_digest,
            final_step=completed_updates,
            artifacts=artifacts,
            artifact_manifest_sha256=artifact_digest,
        )
        _cas_terminal_status_locked(
            run_dir,
            config_hash=config_digest,
            state="completed",
            completed_updates=completed_updates,
            artifact_manifest_sha256=artifact_digest,
        )
        return metrics


def run(
    args: argparse.Namespace,
    *,
    command_argv: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run or transactionally resume one exact, isolated soft-loss-v3 trajectory."""

    config = resolved_config(args)
    config_digest = canonical_hash(config)
    run_dir = Path(args.output_dir).resolve()
    resume_requested = args.resume is not None
    if resume_requested and not run_dir.is_dir():
        raise ResumePreflightError("resume run directory does not exist")
    if not resume_requested and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    profile = resolve_profile(args.profile)
    status: dict[str, object] | None = None
    if resume_requested:
        _preflight_reserved_top_level_paths(run_dir)
        status = _validate_existing_audits(run_dir, args, config, config_digest)
        status = _resume_status_after_failure_preflight(
            run_dir,
            expected_config_hash=config_digest,
            requested_steps=args.steps,
        )
        if status["state"] == "failed":
            raise ResumePreflightError("failed runs are terminal and cannot resume")
        if status["state"] == "completed":
            return _validate_completed_run(
                run_dir,
                status,
                config_digest,
                args.steps,
                calibration=bool(args.calibration_probe),
            )
        if os.path.lexists(run_dir / "artifact_manifest.json"):
            with _run_exclusive_lock(run_dir):
                current_status = _status(run_dir / "STATUS.json")
                if (
                    current_status["state"] != "running"
                    or current_status["config_hash"] != config_digest
                ):
                    raise RuntimeError(
                        "pending terminalization requires compatible running STATUS"
                    )
                manifest_path = _safe_artifact_file(
                    run_dir, "artifact_manifest.json"
                )
                artifact_digest = sha256_file(manifest_path)
                pending_completed = {
                    "schema": _STATUS_SCHEMA,
                    "state": "completed",
                    "config_hash": config_digest,
                    "completed_updates": args.steps,
                    "artifact_manifest_sha256": artifact_digest,
                }
                metrics = _validate_completed_run(
                    run_dir,
                    pending_completed,
                    config_digest,
                    args.steps,
                    calibration=bool(args.calibration_probe),
                )
                try:
                    artifact_payload = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError as error:
                    raise ResumePreflightError(
                        "pending artifact manifest is malformed"
                    ) from error
                artifacts = artifact_payload["artifacts_sha256"]
                assert isinstance(artifacts, Mapping)
                _revalidate_terminal_publication(
                    run_dir,
                    config_hash=config_digest,
                    final_step=args.steps,
                    artifacts=artifacts,
                    artifact_manifest_sha256=artifact_digest,
                )
                _cas_terminal_status_locked(
                    run_dir,
                    config_hash=config_digest,
                    state="completed",
                    completed_updates=args.steps,
                    artifact_manifest_sha256=artifact_digest,
                )
                return metrics
        resume_value = args.resume
        if resume_value != "__OUTPUT_LATEST__":
            resume_path = Path(resume_value).resolve()
            if resume_path != (run_dir / "checkpoint_latest.json").resolve():
                raise ResumePreflightError(
                    "v3 resume must use this run's authoritative checkpoint pointer"
                )

    _seed_everything(args.seed)
    device = torch.device(args.device)
    runtime = _validate_runtime(_build_runtime(args, device, profile))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if resume_requested:
        resume_state = restore_latest_transaction(
            run_dir,
            expected_config_hash=config_digest,
            logits=runtime.logits,
            optimizer=runtime.optimizer,
        )
        _validate_resume_compatibility(args, profile, resume_state)
        _quarantine_unreferenced_transactions(run_dir)
        completed_updates = resume_state.completed_updates
        last_record = resume_state.last_record
        gradient_samples = list(resume_state.gradient_samples)
        step_times = list(resume_state.step_times)
        peak_cuda_bytes = resume_state.peak_cuda_bytes
        calibration_records = list(resume_state.calibration_records)
        committed_history = list(resume_state.committed_history)
        previous_transaction_sha256 = resume_state.committed_transaction_sha256
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        completed_updates = 0
        last_record = None
        gradient_samples: list[ComponentGradientSample] = []
        step_times: list[float] = []
        peak_cuda_bytes = 0
        calibration_records: list[dict[str, object]] = []
        committed_history: list[dict[str, object]] = []
        step_zero = build_checkpoint_payload(
            logits=runtime.logits,
            optimizer=runtime.optimizer,
            completed_updates=0,
            last_record=None,
            rng_state=_capture_rng_state(),
            gradient_samples=(),
            step_times=(),
            peak_cuda_bytes=0,
            calibration_records=(),
            config_hash=config_digest,
        )
        with _run_exclusive_lock(run_dir):
            if any(run_dir.iterdir()):
                raise FileExistsError(
                    f"refusing to overwrite non-empty run directory: {run_dir}"
                )
            atomic_json(
                run_dir / "COMMAND.json",
                {
                    "schema": "pcraft_soft_loss_v3_command_v1",
                    "argv": list(
                        command_argv if command_argv is not None else sys.argv
                    ),
                    "resolved_config": config,
                    "config_hash": config_digest,
                },
            )
            atomic_json(
                run_dir / "run_manifest.json",
                _run_manifest(args, config, config_digest),
            )
            atomic_json(run_dir / "environment.json", environment_manifest(device))
            atomic_json(
                run_dir / "STATUS.json",
                {
                    "schema": _STATUS_SCHEMA,
                    "state": "running",
                    "config_hash": config_digest,
                    "completed_updates": 0,
                },
            )
            zero_completed, zero_last, zero_config = _validate_checkpoint_for_commit(
                step_zero
            )
            zero_rows_bytes, zero_rows = _history_segment_bytes(())
            _commit_training_transaction_locked(
                run_dir,
                checkpoint_payload=step_zero,
                completed=zero_completed,
                last=zero_last,
                config_digest=zero_config,
                rows_bytes=zero_rows_bytes,
                rows=zero_rows,
                previous_transaction_sha256=None,
            )
            previous_transaction_sha256 = _read_pointer(run_dir)[
                "manifest_sha256"
            ]

    history_buffer: list[dict[str, object]] = []
    try:
        while completed_updates < args.steps:
            start_time = _timer()
            record = train_one_update(
                logits=runtime.logits,
                optimizer=runtime.optimizer,
                protocol=runtime.protocol,
                targets=runtime.targets,
                bands=runtime.bands,
                profile=profile,
                completed_updates=completed_updates,
                total_steps=args.steps,
                lambda_dic=args.lambda_dic,
                lambda_dts=args.lambda_dts,
            )
            row = _history_from_step(record, completed_updates)
            elapsed = float(_timer() - start_time)
            if not math.isfinite(elapsed) or elapsed < 0.0:
                raise SoftTrainingStepError("step time must be finite and non-negative")
            completed_updates = record.completed_updates_after
            last_record = record
            history_buffer.append(row)
            step_times.append(elapsed)
            if device.type == "cuda":
                peak_cuda_bytes = max(
                    peak_cuda_bytes,
                    int(torch.cuda.max_memory_allocated(device)),
                )
            if _should_probe(
                completed_updates,
                args.steps,
                profile,
                calibration=bool(args.calibration_probe),
            ):
                try:
                    sample = _gradient_probe(
                        runtime,
                        completed_updates=completed_updates,
                        total_steps=args.steps,
                        profile=profile,
                        lambda_dic=args.lambda_dic,
                        lambda_dts=args.lambda_dts,
                        calibration=bool(args.calibration_probe),
                    )
                except (ValueError, RuntimeError) as error:
                    raise SoftTrainingStepError(
                        f"gradient probe failed at update {completed_updates}: {error}"
                    ) from error
                gradient_samples.append(sample)
                if args.calibration_probe:
                    calibration_records.append(_calibration_record(sample))
            commit_now = (
                completed_updates % _COMMIT_INTERVAL == 0
                or completed_updates == args.steps
            )
            if commit_now:
                checkpoint = build_checkpoint_payload(
                    logits=runtime.logits,
                    optimizer=runtime.optimizer,
                    completed_updates=completed_updates,
                    last_record=last_record,
                    rng_state=_capture_rng_state(),
                    gradient_samples=gradient_samples,
                    step_times=step_times,
                    peak_cuda_bytes=peak_cuda_bytes,
                    calibration_records=calibration_records,
                    config_hash=config_digest,
                )
                commit_training_transaction(
                    run_dir,
                    checkpoint_payload=checkpoint,
                    history_segment=history_buffer,
                    previous_transaction_sha256=previous_transaction_sha256,
                )
                committed_history.extend(history_buffer)
                history_buffer.clear()
                previous_transaction_sha256 = _read_pointer(run_dir)[
                    "manifest_sha256"
                ]
                if args.calibration_probe and completed_updates in _CALIBRATION_UPDATES:
                    _retain_calibration_checkpoint(run_dir, completed_updates)
            _post_update_hook(completed_updates)
    except SoftTrainingStepError as error:
        with _run_exclusive_lock(run_dir):
            current_status = _status(run_dir / "STATUS.json")
            if (
                current_status["state"] != "running"
                or current_status["config_hash"] != config_digest
            ):
                raise RuntimeError(
                    "failure terminalization requires compatible running STATUS"
                ) from error
            diagnostic = _failure_diagnostic(
                run_dir,
                config_hash=config_digest,
                completed_updates=completed_updates,
                error=error,
            )
            _cas_terminal_status_locked(
                run_dir,
                config_hash=config_digest,
                state="failed",
                completed_updates=completed_updates,
                diagnostic=os.path.relpath(diagnostic, run_dir),
            )
        raise

    if history_buffer:
        raise RuntimeError("final completed update was not transactionally committed")
    if completed_updates != args.steps or last_record is None:
        raise RuntimeError("final committed state does not match the requested update count")
    if len(committed_history) != args.steps:
        raise RuntimeError("verified committed history is incomplete")
    return _finalize_completed_run(
        run_dir,
        runtime=runtime,
        args=args,
        profile=profile,
        device=device,
        config_digest=config_digest,
        completed_updates=completed_updates,
        last_record=last_record,
        committed_history=committed_history,
        gradient_samples=gradient_samples,
        calibration_records=calibration_records,
        step_times=step_times,
        peak_cuda_bytes=peak_cuda_bytes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    run(parsed, command_argv=list(argv) if argv is not None else sys.argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
