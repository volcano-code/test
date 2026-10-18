"""Minimal CUDA harness for the frozen soft-loss v5 exploratory ablation."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from shadowforge_pipeline.soft_voxel_v3.losses import balanced_lp_loss
from shadowforge_pipeline.soft_voxel_v3.lovasz import lovasz_hinge_per_view


PROFILE_ORDER = ("l12_bal", "jaccard_only", "fp_tail_only", "v5_full")
TASK_ORDER = tuple(f"connected_{index:02d}" for index in range(1, 11))
INFERENCE_SCOPE = "exploratory_connected_01_10_reuse_after_v4_diagnosis"
_COMPONENTS = {
    "l12_bal": (False, False),
    "jaccard_only": (True, False),
    "fp_tail_only": (False, True),
    "v5_full": (True, True),
}


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_score_target(score: torch.Tensor, target_binary: torch.Tensor) -> None:
    if not isinstance(score, torch.Tensor) or not score.is_floating_point():
        raise ValueError("score must be a floating-point torch.Tensor")
    if not isinstance(target_binary, torch.Tensor) or target_binary.dtype != torch.bool:
        raise ValueError("target_binary must be a boolean torch.Tensor")
    if score.ndim != 3 or score.shape != target_binary.shape or any(size == 0 for size in score.shape):
        raise ValueError("score and target_binary must have equal non-empty [views,H,W] shape")
    if score.device != target_binary.device:
        raise ValueError("score and target_binary must share a device")
    if not torch.isfinite(score).all().item():
        raise ValueError("score must be finite")


def delivery_lovasz_loss(score: torch.Tensor, target_binary: torch.Tensor) -> torch.Tensor:
    """Mean frozen v3 Lovasz hinge on dimensionless ``score / .05``."""

    _validate_score_target(score, target_binary)
    result = lovasz_hinge_per_view(score / score.new_tensor(0.05), target_binary).mean()
    if not torch.isfinite(result).item():
        raise ValueError("Jaccard loss is non-finite")
    return result


def background_hard_negative_tail_loss(
    score: torch.Tensor,
    target_binary: torch.Tensor,
    *,
    fraction: float = 0.05,
    margin: float = 0.01,
    temperature: float = 0.02,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, object]]]:
    """Mean stable row-major top-tail penalty over target-background pixels."""

    _validate_score_target(score, target_binary)
    fraction_value = _finite_number("fraction", fraction)
    margin_value = _finite_number("margin", margin)
    temperature_value = _finite_number("temperature", temperature)
    if not 0.0 < fraction_value <= 1.0 or temperature_value <= 0.0:
        raise ValueError("fraction must be in (0,1] and temperature must be positive")
    losses: list[torch.Tensor] = []
    details: list[dict[str, object]] = []
    flat_score = score.reshape(score.shape[0], -1)
    flat_target = target_binary.reshape(target_binary.shape[0], -1)
    for view_index in range(score.shape[0]):
        background_indices = torch.nonzero(~flat_target[view_index], as_tuple=False).flatten()
        count = int(background_indices.numel())
        if count == 0:
            raise ValueError("each view must contain at least one background pixel")
        background_scores = flat_score[view_index, background_indices]
        penalty = score.new_tensor(temperature_value) * F.softplus(
            (background_scores + score.new_tensor(margin_value))
            / score.new_tensor(temperature_value)
        )
        k = max(1, math.ceil(fraction_value * count))
        order = torch.argsort(penalty, descending=True, stable=True)
        chosen_order = order[:k]
        selected = penalty[chosen_order]
        selected_scores = background_scores[chosen_order]
        selected_indices = background_indices[chosen_order]
        losses.append(selected.mean())
        details.append(
            {
                "view": view_index,
                "background_count": count,
                "selected_count": k,
                "selected_indices": [int(value) for value in selected_indices.detach().cpu().tolist()],
                "score_min": float(selected_scores.detach().min().item()),
                "score_mean": float(selected_scores.detach().mean().item()),
                "score_max": float(selected_scores.detach().max().item()),
            }
        )
    result = torch.stack(losses).mean()
    if not torch.isfinite(result).item():
        raise ValueError("FP-tail loss is non-finite")
    return (result, details) if return_details else result


def v5_ramp(progress: float) -> float:
    value = _finite_number("progress", progress)
    if value <= 0.20:
        return 0.0
    if value >= 0.30:
        return 1.0
    return (value - 0.20) / 0.10


def compose_v5_gradient(
    base_grad: torch.Tensor,
    jaccard_grad: torch.Tensor | None,
    tail_grad: torch.Tensor | None,
    *,
    progress: float,
    profile: str,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    """Independently normalize enabled components and add them to base once."""

    if profile not in _COMPONENTS:
        raise ValueError(f"unknown v5 profile: {profile!r}")
    if not isinstance(base_grad, torch.Tensor) or not base_grad.is_floating_point():
        raise ValueError("base_grad must be a floating-point tensor")
    if not torch.isfinite(base_grad).all().item():
        raise ValueError("base gradient is non-finite")
    base_norm = torch.linalg.vector_norm(base_grad)
    base_norm_value = float(base_norm.item())
    if not math.isfinite(base_norm_value) or base_norm_value <= 1e-12:
        raise ValueError("base gradient norm must exceed 1e-12")
    use_jaccard, use_tail = _COMPONENTS[profile]
    multiplier = v5_ramp(progress)
    result = base_grad.clone()
    diagnostics: dict[str, object] = {
        "base_norm": base_norm_value,
        "ramp": multiplier,
        "jaccard_raw_norm": 0.0,
        "tail_raw_norm": 0.0,
        "jaccard_ratio": 0.0,
        "tail_ratio": 0.0,
        "combined_ratio": 0.0,
        "jaccard_zero": False,
        "tail_zero": False,
    }
    contributions: list[torch.Tensor] = []
    for name, enabled, gradient, budget in (
        ("jaccard", use_jaccard, jaccard_grad, 0.20),
        ("tail", use_tail, tail_grad, 0.10),
    ):
        if not enabled or multiplier == 0.0:
            if gradient is not None and not enabled:
                raise ValueError(f"disabled {name} gradient was constructed")
            continue
        if gradient is None or gradient.shape != base_grad.shape or gradient.device != base_grad.device:
            raise ValueError(f"enabled {name} gradient is missing or incompatible")
        if not torch.isfinite(gradient).all().item():
            raise ValueError(f"{name} gradient is non-finite")
        raw_norm = torch.linalg.vector_norm(gradient)
        raw_value = float(raw_norm.item())
        diagnostics[f"{name}_raw_norm"] = raw_value
        if raw_value <= 1e-12:
            diagnostics[f"{name}_zero"] = True
            continue
        contribution = gradient * (base_norm * (multiplier * budget) / raw_norm)
        contributions.append(contribution)
        result = result + contribution
        diagnostics[f"{name}_ratio"] = float(torch.linalg.vector_norm(contribution).item() / base_norm_value)
    if contributions:
        combined = torch.stack(contributions).sum(dim=0)
        diagnostics["combined_ratio"] = float(torch.linalg.vector_norm(combined).item() / base_norm_value)
    if float(diagnostics["combined_ratio"]) > multiplier * 0.30 + 1e-6:
        raise ValueError("combined auxiliary gradient exceeds the 0.30 bound")
    return (result, diagnostics) if return_diagnostics else result


def build_matrix(tasks: Sequence[str], profiles: Sequence[str]) -> tuple[tuple[str, str], ...]:
    task_tuple = tuple(tasks)
    profile_tuple = tuple(profiles)
    if len(set(task_tuple)) != len(task_tuple) or any(task not in TASK_ORDER for task in task_tuple):
        raise ValueError("tasks must be unique registered connected tasks")
    if len(set(profile_tuple)) != len(profile_tuple) or any(profile not in PROFILE_ORDER for profile in profile_tuple):
        raise ValueError("profiles must be unique registered v5 profiles")
    return tuple((task, profile) for task in task_tuple for profile in profile_tuple)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    try:
        torch.save(dict(payload), temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _canonical_hash(payload: Mapping[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _manifest_entry(manifest_path: Path, task: str) -> dict[str, object]:
    manifest = _read_json(manifest_path, label="v4 ablation manifest")
    matrix = manifest.get("matrix")
    if type(matrix) is not list:
        raise ValueError("v4 ablation manifest matrix is missing")
    matches = [
        row for row in matrix
        if type(row) is dict and row.get("task_id") == task and row.get("profile") == "l12_bal"
    ]
    if len(matches) != 1:
        raise ValueError(f"v4 manifest must contain one l12_bal row for {task}")
    return matches[0]


def _v4_config_on_disk(run_dir: Path, entry: Mapping[str, object]) -> dict[str, object]:
    command_path = run_dir / "COMMAND.json"
    if command_path.is_file():
        command = _read_json(command_path, label="v4 COMMAND")
        config = command.get("resolved_config")
    else:
        config_path = run_dir / "resolved_config.json"
        config = _read_json(config_path, label="v4 resolved config")
    if type(config) is not dict:
        raise ValueError("v4 resolved config is malformed")
    return config


def _validate_v4_completed(run_dir: Path, *, expected_config: Mapping[str, object]) -> object:
    """Route reuse through v4's complete provenance/transaction authenticator."""

    from shadowforge_pipeline.soft_voxel_v4.trainer import validate_completed_run

    return validate_completed_run(run_dir, expected_config=expected_config)


def validate_v4_base_cell(
    source_v4_root: Path,
    manifest_path: Path,
    task: str,
) -> dict[str, object]:
    """Authenticate the exact reusable v4 base cell without copying its files."""

    root = Path(source_v4_root).resolve(strict=True)
    entry = _manifest_entry(Path(manifest_path), task)
    run_dir_value = entry.get("run_dir")
    if type(run_dir_value) is not str:
        run_dir = root / "runs" / "ablation10" / task / "l12_bal"
    else:
        run_dir = Path(run_dir_value)
    expected = root / "runs" / "ablation10" / task / "l12_bal"
    if run_dir.resolve(strict=True) != expected.resolve(strict=True):
        raise ValueError("v4 base run directory is not canonical")
    config = _v4_config_on_disk(run_dir, entry)
    checks = {
        "task": (config.get("run_identity", {}).get("task_id") if type(config.get("run_identity")) is dict else None, task),
        "profile": (config.get("profile", {}).get("name") if type(config.get("profile")) is dict else None, "l12_bal"),
        "resolution": (config.get("resolution"), 128),
        "steps": (config.get("steps"), 2000),
        "seed": (config.get("seed"), 17),
        "learning_rate": (config.get("learning_rate"), 0.007),
    }
    for name, (actual, required) in checks.items():
        if actual != required:
            raise ValueError(f"v4 base {name} does not match the frozen value")
    declared = entry.get("resolved_config")
    if type(declared) is not dict or config != declared:
        raise ValueError("v4 on-disk and manifest resolved configurations differ")
    authenticated = _validate_v4_completed(run_dir, expected_config=declared)
    inputs = config.get("shadow_inputs_xyz")
    if type(inputs) is not list or len(inputs) != 3:
        raise ValueError("v4 base input paths are malformed")
    paths: list[str] = []
    for axis, row in zip("xyz", inputs):
        if type(row) is not dict or row.get("axis") != axis or type(row.get("path")) is not str:
            raise ValueError("v4 base input paths/order are malformed")
        input_path = Path(row["path"])
        if not input_path.is_file():
            raise ValueError("v4 base input path is unavailable")
        if type(row.get("sha256")) is str and _sha256(input_path) != row["sha256"]:
            raise ValueError("v4 base input SHA differs")
        paths.append(str(input_path))
    status = _read_json(run_dir / "STATUS.json", label="v4 STATUS")
    if status.get("state") != "completed" or status.get("completed_updates") != 2000:
        raise ValueError("v4 base status is not completed at 2000 updates")
    metric_path = run_dir / "metrics.json"
    metric_sha256 = _sha256(metric_path)
    status_sha256 = _sha256(run_dir / "STATUS.json")
    if (
        getattr(authenticated, "status_sha256", None) != status_sha256
        or getattr(authenticated, "artifacts_sha256", {}).get("metrics.json") != metric_sha256
    ):
        raise ValueError("v4 authenticated STATUS/artifact/metrics binding differs")
    metrics = _parse_source_v4_metrics(_read_json(metric_path, label="v4 metrics"))
    return {
        "task": task,
        "profile": "l12_bal",
        "input_paths": paths,
        "source_metric_path": str(metric_path),
        "source_metric_sha256": metric_sha256,
        "source_status_path": str(run_dir / "STATUS.json"),
        "source_status_sha256": status_sha256,
        "source_artifact_manifest_sha256": getattr(
            authenticated, "artifact_manifest_sha256", None
        ),
        **metrics,
    }


def _safe_divide(numerator: int, denominator: int, *, label: str) -> float:
    if denominator <= 0:
        raise ValueError(f"{label} denominator must be positive")
    return numerator / denominator


def evaluate_binary_masks(mask: torch.Tensor, target_binary: torch.Tensor) -> dict[str, object]:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean")
    if not isinstance(target_binary, torch.Tensor) or target_binary.dtype != torch.bool:
        raise ValueError("target_binary must be boolean")
    if mask.ndim != 3 or mask.shape != target_binary.shape or mask.shape[0] != 6:
        raise ValueError("mask and target must have equal [6,H,W] shape")
    per_view: list[dict[str, object]] = []
    for prediction, target in zip(mask, target_binary):
        tp = int(torch.count_nonzero(prediction & target).item())
        fp = int(torch.count_nonzero(prediction & ~target).item())
        fn = int(torch.count_nonzero(~prediction & target).item())
        tn = int(torch.count_nonzero(~prediction & ~target).item())
        per_view.append(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "iou": _safe_divide(tp, tp + fp + fn, label="IoU"),
                "dice": _safe_divide(2 * tp, 2 * tp + fp + fn, label="Dice"),
                "recall": _safe_divide(tp, tp + fn, label="Recall"),
                "fpr": _safe_divide(fp, fp + tn, label="FPR"),
            }
        )
    result: dict[str, object] = {"per_view": per_view}
    for metric in ("iou", "dice", "recall", "fpr"):
        result[f"mean_{metric}"] = sum(float(view[metric]) for view in per_view) / 6.0
    result.update({name: sum(int(view[name]) for view in per_view) for name in ("tp", "fp", "fn", "tn")})
    return result


def _profile_flags(profile: str) -> tuple[bool, bool]:
    try:
        return _COMPONENTS[profile]
    except KeyError as error:
        raise ValueError(f"unknown v5 profile: {profile!r}") from error


def _torch_load(path: Path, device: torch.device) -> dict[str, object]:
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except (OSError, RuntimeError, EOFError) as error:
        raise ValueError("checkpoint is corrupt") from error
    if type(value) is not dict:
        raise ValueError("checkpoint is malformed")
    return value


def _initial_audit(config_hash: str, profile: str, steps: int) -> dict[str, object]:
    use_jaccard, use_tail = _profile_flags(profile)
    return {
        "schema": "pcraft_soft_loss_v5_audit_v1",
        "config_hash": config_hash,
        "profile": profile,
        "steps": steps,
        "finite": True,
        "eligible_updates": 0,
        "jaccard_zero_updates": 0,
        "tail_zero_updates": 0,
        "jaccard_active_updates": 0,
        "tail_active_updates": 0,
        "jaccard_full_ramp_active_updates": 0,
        "tail_full_ramp_active_updates": 0,
        "max_jaccard_ratio": 0.0,
        "max_tail_ratio": 0.0,
        "max_combined_ratio": 0.0,
        "selected_count_min": None,
        "selected_count_max": None,
        "selected_score_min": None,
        "selected_score_mean_sum": 0.0,
        "selected_score_max": None,
        "selected_observations": 0,
        "parameters_changed_updates": 0,
        "enabled": {"jaccard": use_jaccard, "tail": use_tail},
        "active_after_ramp": profile == "l12_bal",
    }


def _active_at_full_ramp(
    audit: Mapping[str, object], *, use_jaccard: bool, use_tail: bool
) -> bool:
    """Require every enabled component to have positive norm at ramp multiplier one."""

    return bool(
        (not use_jaccard or int(audit.get("jaccard_full_ramp_active_updates", 0)) > 0)
        and (not use_tail or int(audit.get("tail_full_ramp_active_updates", 0)) > 0)
    )


def _update_tail_audit(audit: dict[str, object], details: list[dict[str, object]]) -> None:
    for row in details:
        count = int(row["selected_count"])
        score_min = float(row["score_min"])
        score_mean = float(row["score_mean"])
        score_max = float(row["score_max"])
        prior_min = audit["selected_count_min"]
        prior_max = audit["selected_count_max"]
        audit["selected_count_min"] = count if prior_min is None else min(int(prior_min), count)
        audit["selected_count_max"] = count if prior_max is None else max(int(prior_max), count)
        old_score_min = audit["selected_score_min"]
        old_score_max = audit["selected_score_max"]
        audit["selected_score_min"] = score_min if old_score_min is None else min(float(old_score_min), score_min)
        audit["selected_score_max"] = score_max if old_score_max is None else max(float(old_score_max), score_max)
        audit["selected_score_mean_sum"] = float(audit["selected_score_mean_sum"]) + score_mean
        audit["selected_observations"] = int(audit["selected_observations"]) + 1


def _checkpoint_payload(
    config: Mapping[str, object], config_hash: str, logits: torch.Tensor,
    optimizer: torch.optim.Optimizer, completed_updates: int, audit: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "pcraft_soft_loss_v5_checkpoint_v1",
        "config": dict(config),
        "config_hash": config_hash,
        "completed_updates": completed_updates,
        "logits": logits.detach(),
        "optimizer": optimizer.state_dict(),
        "audit": dict(audit),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if logits.device.type == "cuda" else None,
    }


def _write_progress(
    run_dir: Path, config: Mapping[str, object], config_hash: str, logits: torch.Tensor,
    optimizer: torch.optim.Optimizer, completed_updates: int, audit: Mapping[str, object],
) -> None:
    _atomic_torch_save(
        run_dir / "checkpoint_latest.pt",
        _checkpoint_payload(config, config_hash, logits, optimizer, completed_updates, audit),
    )
    _atomic_json(
        run_dir / "STATUS.json",
        {
            "schema": "pcraft_soft_loss_v5_status_v1",
            "state": "running",
            "config_hash": config_hash,
            "completed_updates": completed_updates,
        },
    )


def _validate_config(config: Mapping[str, object]) -> tuple[str, int, int, float, torch.device]:
    if type(config) is not dict:
        raise ValueError("config must be an exact mapping")
    profile = config.get("profile")
    _profile_flags(profile)  # type: ignore[arg-type]
    resolution = config.get("resolution")
    steps = config.get("steps")
    learning_rate = config.get("learning_rate")
    if type(resolution) is not int or resolution <= 0 or type(steps) is not int or steps <= 0:
        raise ValueError("resolution and steps must be positive integers")
    learning_rate_value = _finite_number("learning_rate", learning_rate)
    if learning_rate_value <= 0.0 or config.get("seed") != 17:
        raise ValueError("learning rate/seed are invalid")
    device = torch.device(str(config.get("device")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    return str(profile), resolution, steps, learning_rate_value, device


def train_cell(
    run_dir: Path,
    config: Mapping[str, object],
    *,
    protocol: object,
    targets: torch.Tensor,
    resume: bool = False,
    max_updates: int | None = None,
) -> dict[str, object]:
    """Train or resume one cell, atomically committing every 100 updates."""

    profile, resolution, total_steps, learning_rate, device = _validate_config(config)
    if targets.shape[0] != 6 or targets.device != device or not torch.isfinite(targets).all().item():
        raise ValueError("targets must be six finite views on the configured device")
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    allowed = {"checkpoint_latest.pt", "metrics.json", "audit.json", "STATUS.json"}
    extras = {item.name for item in root.iterdir()} - allowed
    if extras:
        raise ValueError("run directory contains unexpected files")
    canonical_config = json.loads(json.dumps(dict(config), sort_keys=True, allow_nan=False))
    config_hash = _canonical_hash(canonical_config)
    torch.manual_seed(17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(17)
    logits = torch.full((1, 1, resolution, resolution, resolution), -4.0, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    audit = _initial_audit(config_hash, profile, total_steps)
    start = 0
    checkpoint_path = root / "checkpoint_latest.pt"
    status_path = root / "STATUS.json"
    if resume:
        if not checkpoint_path.is_file() or not status_path.is_file():
            raise ValueError("resume requires checkpoint and status")
        status = _read_json(status_path, label="v5 STATUS")
        checkpoint = _torch_load(checkpoint_path, device)
        if (
            status.get("state") != "running"
            or status.get("config_hash") != config_hash
            or checkpoint.get("config_hash") != config_hash
            or checkpoint.get("config") != canonical_config
            or status.get("completed_updates") != checkpoint.get("completed_updates")
        ):
            raise ValueError("resume checkpoint/status/config mismatch")
        start = int(checkpoint["completed_updates"])
        if not 0 <= start <= total_steps:
            raise ValueError("resume completed update is invalid")
        loaded_logits = checkpoint.get("logits")
        if not isinstance(loaded_logits, torch.Tensor) or loaded_logits.shape != logits.shape or not torch.isfinite(loaded_logits).all().item():
            raise ValueError("resume logits are invalid")
        with torch.no_grad():
            logits.copy_(loaded_logits.to(device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        loaded_audit = checkpoint.get("audit")
        if type(loaded_audit) is not dict:
            raise ValueError("resume audit is invalid")
        audit = loaded_audit
        audit.setdefault("jaccard_full_ramp_active_updates", 0)
        audit.setdefault("tail_full_ramp_active_updates", 0)
        rng = checkpoint.get("torch_rng_state")
        if isinstance(rng, torch.Tensor):
            torch.set_rng_state(rng.cpu())
        if device.type == "cuda" and type(checkpoint.get("cuda_rng_state")) is list:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
    elif any(path.exists() for path in (checkpoint_path, status_path, root / "metrics.json", root / "audit.json")):
        raise ValueError("fresh run directory is not empty")
    stop = total_steps if max_updates is None else min(total_steps, start + max_updates)
    if max_updates is not None and (type(max_updates) is not int or max_updates <= 0):
        raise ValueError("max_updates must be a positive integer")
    target_binary = targets >= 0.5
    use_jaccard, use_tail = _profile_flags(profile)
    for step in range(start, stop):
        progress = (step + 1) / total_steps
        density = torch.sigmoid(logits)
        projection = protocol.render_all(density)
        base_loss = (
            balanced_lp_loss(projection, targets, p=2)
            + 0.35 * balanced_lp_loss(projection, targets, p=1)
        ).mean()
        if not torch.isfinite(base_loss).item():
            raise ValueError("base loss is non-finite")
        base_grad = torch.autograd.grad(base_loss, logits)[0]
        jaccard_grad = tail_grad = None
        tail_details: list[dict[str, object]] = []
        if v5_ramp(progress) > 0.0:
            audit["eligible_updates"] = int(audit["eligible_updates"]) + 1
            if use_jaccard:
                from shadowforge_pipeline.soft_voxel_v3.thresholding import render_thresholded

                readout = render_thresholded(protocol, torch.sigmoid(logits), 0.05)
                jaccard_loss = delivery_lovasz_loss(readout.score, target_binary)
                jaccard_grad = torch.autograd.grad(jaccard_loss, logits)[0]
            if use_tail:
                from shadowforge_pipeline.soft_voxel_v3.thresholding import render_thresholded

                readout = render_thresholded(protocol, torch.sigmoid(logits), 0.05)
                tail_result = background_hard_negative_tail_loss(
                    readout.score,
                    target_binary,
                    fraction=0.05,
                    margin=0.01,
                    temperature=0.02,
                    return_details=True,
                )
                tail_loss, tail_details = tail_result
                tail_grad = torch.autograd.grad(tail_loss, logits)[0]
        final_grad, diagnostics = compose_v5_gradient(
            base_grad,
            jaccard_grad,
            tail_grad,
            progress=progress,
            profile=profile,
            return_diagnostics=True,
        )
        if not torch.isfinite(final_grad).all().item():
            raise ValueError("composed gradient is non-finite")
        if v5_ramp(progress) > 0.0:
            for name, enabled in (("jaccard", use_jaccard), ("tail", use_tail)):
                if enabled:
                    if diagnostics[f"{name}_zero"]:
                        audit[f"{name}_zero_updates"] = int(audit[f"{name}_zero_updates"]) + 1
                    else:
                        audit[f"{name}_active_updates"] = int(audit[f"{name}_active_updates"]) + 1
                        if float(diagnostics["ramp"]) == 1.0:
                            key = f"{name}_full_ramp_active_updates"
                            audit[key] = int(audit[key]) + 1
                    audit[f"max_{name}_ratio"] = max(
                        float(audit[f"max_{name}_ratio"]), float(diagnostics[f"{name}_ratio"])
                    )
            audit["max_combined_ratio"] = max(
                float(audit["max_combined_ratio"]), float(diagnostics["combined_ratio"])
            )
            if tail_details:
                _update_tail_audit(audit, tail_details)
        before = logits.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        logits.grad = final_grad
        torch.nn.utils.clip_grad_norm_([logits], 1.0)
        optimizer.step()
        if not torch.isfinite(logits).all().item():
            raise ValueError("parameters became non-finite")
        if not torch.equal(before, logits.detach()):
            audit["parameters_changed_updates"] = int(audit["parameters_changed_updates"]) + 1
        completed = step + 1
        if completed % 100 == 0 or completed == stop:
            _write_progress(root, canonical_config, config_hash, logits, optimizer, completed, audit)
    if stop < total_steps:
        return _read_json(status_path, label="v5 STATUS")
    eligible = int(audit["eligible_updates"])
    for name, enabled in (("jaccard", use_jaccard), ("tail", use_tail)):
        if enabled and eligible > 0:
            zero_fraction = int(audit[f"{name}_zero_updates"]) / eligible
            audit[f"{name}_zero_fraction"] = zero_fraction
    audit["active_after_ramp"] = _active_at_full_ramp(
        audit, use_jaccard=use_jaccard, use_tail=use_tail
    )
    with torch.no_grad():
        from shadowforge_pipeline.soft_voxel_v3.thresholding import render_thresholded

        final_readout = render_thresholded(protocol, torch.sigmoid(logits), 0.05)
        metrics = evaluate_binary_masks(final_readout.mask, target_binary)
        if use_tail:
            _, final_tail = background_hard_negative_tail_loss(
                final_readout.score, target_binary, return_details=True
            )
            metrics["tail_selection_per_view"] = final_tail
    metrics.update(
        {
            "schema": "pcraft_soft_loss_v5_metrics_v1",
            "config_hash": config_hash,
            "task": config.get("task"),
            "profile": profile,
            "steps": total_steps,
            "threshold": 0.05,
        }
    )
    observations = int(audit["selected_observations"])
    if observations:
        audit["selected_score_mean"] = float(audit["selected_score_mean_sum"]) / observations
    _atomic_json(root / "metrics.json", metrics)
    _atomic_json(root / "audit.json", audit)
    _atomic_torch_save(
        checkpoint_path,
        _checkpoint_payload(canonical_config, config_hash, logits, optimizer, total_steps, audit),
    )
    completed_status = {
        "schema": "pcraft_soft_loss_v5_status_v1",
        "state": "completed",
        "config_hash": config_hash,
        "completed_updates": total_steps,
        "metrics_sha256": _sha256(root / "metrics.json"),
        "audit_sha256": _sha256(root / "audit.json"),
        "checkpoint_sha256": _sha256(checkpoint_path),
    }
    _atomic_json(status_path, completed_status)
    return completed_status


def _parse_source_v4_metrics(payload: Mapping[str, object]) -> dict[str, object]:
    thresholds = payload.get("thresholds")
    primary = thresholds.get("0.05") if type(thresholds) is dict else None
    per_view = primary.get("per_view") if type(primary) is dict else None
    if type(per_view) is not list or len(per_view) != 6:
        raise ValueError("v4 .05 metrics must have six views")
    parsed_views: list[dict[str, object]] = []
    for row in per_view:
        if type(row) is not dict:
            raise ValueError("v4 view metric is malformed")
        parsed: dict[str, object] = {}
        for count in ("tp", "fp", "fn", "tn"):
            if type(row.get(count)) is not int or int(row[count]) < 0:
                raise ValueError("v4 count is malformed")
            parsed[count] = int(row[count])
        for metric in ("iou", "dice", "recall", "fpr"):
            parsed[metric] = _finite_number(f"v4 {metric}", row.get(metric))
        parsed_views.append(parsed)
    result: dict[str, object] = {"per_view": parsed_views}
    for metric in ("iou", "dice", "recall", "fpr"):
        result[f"mean_{metric}"] = sum(float(view[metric]) for view in parsed_views) / 6.0
    result.update({name: sum(int(view[name]) for view in parsed_views) for name in ("tp", "fp", "fn", "tn")})
    return result


def _build_v5_config(
    *, task: str, profile: str, steps: int, device: str, input_paths: Sequence[str]
) -> dict[str, object]:
    return {
        "schema": "pcraft_soft_loss_v5_config_v1",
        "task": task,
        "profile": profile,
        "resolution": 128,
        "views": 6,
        "image_size": 256,
        "steps": steps,
        "seed": 17,
        "optimizer": "Adam",
        "learning_rate": 0.007,
        "initial_logit": -4.0,
        "gradient_clip": 1.0,
        "threshold": 0.05,
        "input_paths": list(input_paths),
        "device": device,
        "inference_scope": INFERENCE_SCOPE,
    }


def _validate_expected_v5_config(config: Mapping[str, object]) -> dict[str, object]:
    try:
        canonical = json.loads(json.dumps(dict(config), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("expected config is not canonical JSON") from error
    required = {
        "schema", "task", "profile", "resolution", "views", "image_size", "steps",
        "seed", "optimizer", "learning_rate", "initial_logit", "gradient_clip",
        "threshold", "input_paths", "device", "inference_scope",
    }
    if set(canonical) != required:
        raise ValueError("expected config has the wrong exact fields")
    if (
        canonical["schema"] != "pcraft_soft_loss_v5_config_v1"
        or canonical["task"] not in TASK_ORDER
        or canonical["profile"] not in PROFILE_ORDER
        or canonical["resolution"] != 128
        or canonical["views"] != 6
        or canonical["image_size"] != 256
        or type(canonical["steps"]) is not int
        or canonical["steps"] not in (200, 2000)
        or canonical["seed"] != 17
        or canonical["optimizer"] != "Adam"
        or canonical["learning_rate"] != 0.007
        or canonical["initial_logit"] != -4.0
        or canonical["gradient_clip"] != 1.0
        or canonical["threshold"] != 0.05
        or type(canonical["input_paths"]) is not list
        or len(canonical["input_paths"]) != 3
        or any(type(path) is not str for path in canonical["input_paths"])
        or type(canonical["device"]) is not str
        or not canonical["device"].startswith("cuda:")
        or canonical["inference_scope"] != INFERENCE_SCOPE
    ):
        raise ValueError("expected config differs from the frozen v5 configuration")
    return canonical


def _load_completed_v5(
    run_dir: Path, expected_config: Mapping[str, object]
) -> dict[str, object]:
    expected = _validate_expected_v5_config(expected_config)
    expected_hash = _canonical_hash(expected)
    task = str(expected["task"])
    profile = str(expected["profile"])
    steps = int(expected["steps"])
    status = _read_json(run_dir / "STATUS.json", label="v5 STATUS")
    if status.get("state") != "completed":
        raise ValueError("v5 cell is not completed")
    if status.get("config_hash") != expected_hash:
        raise ValueError("v5 STATUS does not share the expected config_hash")
    if status.get("completed_updates") != steps:
        raise ValueError("v5 STATUS does not share the expected config steps")
    metrics_path = run_dir / "metrics.json"
    audit_path = run_dir / "audit.json"
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    if not all(path.is_file() for path in (metrics_path, audit_path, checkpoint_path)):
        raise ValueError("v5 cell artifacts are incomplete")
    if (
        status.get("metrics_sha256") != _sha256(metrics_path)
        or status.get("audit_sha256") != _sha256(audit_path)
        or status.get("checkpoint_sha256") != _sha256(checkpoint_path)
    ):
        raise ValueError("v5 cell artifact SHA mismatch")
    metrics = _read_json(metrics_path, label="v5 metrics")
    if metrics.get("task") != task or metrics.get("profile") != profile or metrics.get("steps") != steps:
        raise ValueError("v5 metrics identity mismatch")
    if metrics.get("config_hash") != expected_hash:
        raise ValueError("v5 metrics do not share the expected config_hash")
    for key in ("mean_iou", "mean_dice", "mean_recall", "mean_fpr"):
        _finite_number(key, metrics.get(key))
    audit = _read_json(audit_path, label="v5 audit")
    if audit.get("finite") is not True:
        raise ValueError("v5 audit is non-finite")
    if audit.get("config_hash") != expected_hash:
        raise ValueError("v5 audit does not share the expected config_hash")
    checkpoint = _torch_load(checkpoint_path, torch.device("cpu"))
    if (
        checkpoint.get("config_hash") != expected_hash
        or checkpoint.get("config") != expected
        or checkpoint.get("completed_updates") != steps
    ):
        raise ValueError("v5 checkpoint does not share the exact expected config_hash/config")
    return {**metrics, "audit": audit}


def run_ablation(
    output_root: Path,
    source_v4_root: Path,
    *,
    device: str,
    steps: int,
    profiles: Sequence[str],
    tasks: Sequence[str],
    smoke: bool,
    resume: bool,
) -> list[dict[str, object]]:
    """Execute the selected cells sequentially and return their metric rows."""

    task_tuple = (tuple(tasks)[0],) if smoke else tuple(tasks)
    profile_tuple = tuple(profiles)
    effective_steps = 200 if smoke else steps
    matrix = build_matrix(task_tuple, profile_tuple)
    if smoke and len(task_tuple) != 1:
        raise AssertionError("smoke task reduction failed")
    source_root = Path(source_v4_root).resolve(strict=True)
    manifest_path = source_root / "manifests" / "ablation10.json"
    destination_root = Path(output_root) / "smoke" if smoke else Path(output_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    allow_reuse = (
        not smoke
        and effective_steps == 2000
        and task_tuple == TASK_ORDER
        and profile_tuple == PROFILE_ORDER
    )
    rows: list[dict[str, object]] = []
    total = len(matrix)
    sources = {
        task: validate_v4_base_cell(source_root, manifest_path, task)
        for task in task_tuple
    }
    for index, (task, profile) in enumerate(matrix, 1):
        source = sources[task]
        if profile == "l12_bal" and allow_reuse:
            row = source
        else:
            run_dir = destination_root / "runs" / task / profile
            config = _build_v5_config(
                task=task,
                profile=profile,
                steps=effective_steps,
                device=device,
                input_paths=source["input_paths"],
            )
            if run_dir.exists() and (run_dir / "STATUS.json").is_file():
                try:
                    row = _load_completed_v5(run_dir, config)
                except ValueError:
                    if not resume:
                        raise
                    row = None
                else:
                    rows.append(row)
                    print(f"{index}/{total} {task} {profile} {float(row['mean_iou']):.6f} {float(row['mean_dice']):.6f}", flush=True)
                    continue
            else:
                row = None
            from shadowforge_pipeline.soft_voxel_v3.data import load_targets
            from shadowforge_pipeline.soft_voxel_v3.protocol import SadekarTFSProtocol

            torch_device = torch.device(device)
            protocol = SadekarTFSProtocol(torch_device, image_size=256)
            targets = load_targets(source["input_paths"], mirror=True, size=256, device=torch_device)
            train_cell(run_dir, config, protocol=protocol, targets=targets, resume=bool(resume and row is None and (run_dir / "checkpoint_latest.pt").is_file()))
            row = _load_completed_v5(run_dir, config)
        rows.append(row)
        print(f"{index}/{total} {task} {profile} {float(row['mean_iou']):.6f} {float(row['mean_dice']):.6f}", flush=True)
    return rows


def one_sided_sign_flip_pvalue(deltas: Sequence[float]) -> float:
    values = tuple(_finite_number("delta", item) for item in deltas)
    if not values:
        raise ValueError("sign-flip test requires deltas")
    observed = sum(values) / len(values)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = sum(sign * value for sign, value in zip(signs, values)) / len(values)
        if statistic >= observed:
            extreme += 1
    return extreme / (2 ** len(values))


def classify_summary(summary: Mapping[str, object]) -> str:
    if summary.get("engineering_valid") is not True:
        return "INCONCLUSIVE"
    mean_iou = _finite_number("mean_delta_iou", summary.get("mean_delta_iou"))
    mean_dice = _finite_number("mean_delta_dice", summary.get("mean_delta_dice"))
    if mean_iou < 0.0 or mean_dice < 0.0:
        return "HARMFUL"
    if summary.get("all_success_gates") is True:
        return "V5_SUCCESS"
    return "POSITIVE_BUT_SUBSTANTIVE_FAIL"


def _task_mean(rows: Sequence[Mapping[str, object]], metric: str) -> float:
    return sum(_finite_number(metric, row.get(metric)) for row in rows) / len(rows)


def analyze_ablation(
    output_root: Path,
    source_v4_root: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    """Require the complete 10x4 matrix, then atomically publish CSV and report."""

    output = Path(output_root)
    source = Path(source_v4_root)
    manifest = Path(manifest_path) if manifest_path is not None else source / "manifests" / "ablation10.json"
    manifest_payload = _read_json(manifest, label="v4 ablation manifest")
    manifest_matrix = manifest_payload.get("matrix")
    reusable = {
        (row.get("task_id"), row.get("profile"))
        for row in manifest_matrix
        if type(row) is dict
    } if type(manifest_matrix) is list else set()
    available = sum((task, "l12_bal") in reusable for task in TASK_ORDER)
    available += sum(
        (output / "runs" / task / profile / "STATUS.json").is_file()
        for task in TASK_ORDER
        for profile in PROFILE_ORDER[1:]
    )
    if available != 40:
        raise ValueError(f"analysis requires exactly 40 result rows; found {available}")
    sources = {
        task: validate_v4_base_cell(source, manifest, task) for task in TASK_ORDER
    }
    rows: list[dict[str, object]] = []
    engineering_valid = True
    for task, profile in build_matrix(TASK_ORDER, PROFILE_ORDER):
        if profile == "l12_bal":
            row = sources[task]
            row["reused_v4"] = True
        else:
            expected_config = _build_v5_config(
                task=task,
                profile=profile,
                steps=2000,
                device="cuda:0",
                input_paths=sources[task]["input_paths"],
            )
            row = _load_completed_v5(output / "runs" / task / profile, expected_config)
            audit = row.pop("audit")
            engineering_valid = engineering_valid and bool(audit.get("active_after_ramp"))
            row["reused_v4"] = False
        rows.append(row)
    if len(rows) != 40:
        raise ValueError("analysis requires exactly 40 result rows")
    if any(not math.isfinite(float(row[key])) for row in rows for key in ("mean_iou", "mean_dice", "mean_recall", "mean_fpr")):
        raise ValueError("analysis evidence is non-finite")
    grouped = {(str(row["task"]), str(row["profile"])): row for row in rows}
    arms = {
        profile: {
            metric: _task_mean([grouped[(task, profile)] for task in TASK_ORDER], metric)
            for metric in ("mean_iou", "mean_dice", "mean_recall", "mean_fpr")
        }
        for profile in PROFILE_ORDER
    }

    def contrast(treatment: str, control: str) -> dict[str, object]:
        values: dict[str, list[float]] = {}
        for metric in ("mean_iou", "mean_dice", "mean_recall", "mean_fpr"):
            values[metric] = [
                float(grouped[(task, treatment)][metric]) - float(grouped[(task, control)][metric])
                for task in TASK_ORDER
            ]
        return {
            "treatment": treatment,
            "control": control,
            "task_deltas": {metric: value for metric, value in values.items()},
            "means": {metric: sum(value) / 10.0 for metric, value in values.items()},
            "positive_iou_tasks": sum(delta > 0.0 for delta in values["mean_iou"]),
            "minimum_iou_delta": min(values["mean_iou"]),
            "iou_sign_flip_p": one_sided_sign_flip_pvalue(values["mean_iou"]),
        }

    full_base = contrast("v5_full", "l12_bal")
    jaccard = contrast("v5_full", "fp_tail_only")
    tail = contrast("v5_full", "jaccard_only")
    full_iou = [float(grouped[(task, "v5_full")]["mean_iou"]) for task in TASK_ORDER]
    recall_deltas = full_base["task_deltas"]["mean_recall"]
    safety_pass = min(recall_deltas) >= -0.005 and full_base["means"]["mean_fpr"] <= 0.001
    success = bool(
        engineering_valid
        and full_base["means"]["mean_iou"] >= 0.010
        and full_base["means"]["mean_dice"] >= 0.005
        and full_base["positive_iou_tasks"] >= 8
        and full_base["minimum_iou_delta"] >= -0.005
        and full_base["iou_sign_flip_p"] < 0.05
        and sum(full_iou) / 10.0 >= 0.95
        and min(full_iou) >= 0.94
        and safety_pass
        and jaccard["means"]["mean_iou"] >= 0.0
        and tail["means"]["mean_iou"] >= 0.0
        and tail["means"]["mean_fpr"] <= 0.0
    )
    summary = {
        "engineering_valid": engineering_valid,
        "safety_pass": safety_pass,
        "mean_delta_iou": full_base["means"]["mean_iou"],
        "mean_delta_dice": full_base["means"]["mean_dice"],
        "all_success_gates": success,
    }
    report: dict[str, object] = {
        "schema": "pcraft_soft_loss_v5_report_v1",
        "inference_scope": INFERENCE_SCOPE,
        "row_count": 40,
        "arms": arms,
        "contrasts": {"full_vs_base": full_base, "jaccard_contribution": jaccard, "tail_contribution": tail},
        "full_mean_iou": sum(full_iou) / 10.0,
        "full_worst_iou": min(full_iou),
        "label": classify_summary(summary),
    }
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    rows_path = analysis / "rows.csv"
    handle, temporary = tempfile.mkstemp(prefix=".rows.csv.", dir=analysis)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            fields = [
                "task", "profile", "reused_v4", "mean_iou", "mean_dice",
                "mean_recall", "mean_fpr", "tp", "fp", "fn", "tn",
                "source_metric_path", "source_metric_sha256", "source_status_path",
                "source_status_sha256", "source_artifact_manifest_sha256",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, rows_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    _atomic_json(analysis / "report.json", report)
    return report


__all__ = [
    "INFERENCE_SCOPE", "PROFILE_ORDER", "TASK_ORDER", "analyze_ablation",
    "background_hard_negative_tail_loss", "build_matrix", "classify_summary",
    "compose_v5_gradient", "delivery_lovasz_loss", "evaluate_binary_masks",
    "one_sided_sign_flip_pvalue", "run_ablation", "train_cell", "v5_ramp",
    "validate_v4_base_cell",
]
