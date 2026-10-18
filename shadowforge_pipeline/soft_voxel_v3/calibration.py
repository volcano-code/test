"""Gradient-only soft-loss-v3 calibration and immutable weight publication.

The calibration consumer deliberately knows only immutable manifests, retained
checkpoint bytes, and component-gradient norms.  It neither imports execution
code nor opens task-outcome artifacts.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from statistics import median
from types import MappingProxyType
import uuid

import shadowforge_pipeline.soft_voxel_v3.provenance as provenance_module
from shadowforge_pipeline.soft_voxel_v3.provenance import (
    canonical_hash,
    canonical_json_bytes,
)


_UPDATES = (1000, 1300, 1600)
_AXES = ("x", "y", "z")
_DEVELOPMENT_IDS = ("arcane_casters", "blade_heroes", "stage_dancers")
_FORMAL_IDS = tuple(f"connected_{index:02d}" for index in range(1, 11))
_PRIMARY_VARIANTS = ("l12_bal", "dic_only", "dts_only", "v3_full")
_APPENDIX_VARIANTS = ("v3_zero_margin", "v3_no_l1", "v3_no_l2")
_WEIGHT_SCHEMA = "pcraft_soft_loss_v3_weights_v1"
_STAGE_SCHEMA = "pcraft_soft_loss_v3_calibration_stage_v1"
_SUITE_SCHEMA = "pcraft_soft_loss_v3_suite_v1"
_MASTER_SCHEMA = "pcraft_soft_loss_v3_experiment_manifest_v1"
_RUN_SCHEMA = "pcraft_soft_loss_v3_run_v1"
_AUDIT_SCHEMA = "pcraft_soft_loss_v3_gradient_audit_v1"
_ARTIFACT_SCHEMA = "pcraft_soft_loss_v3_artifacts_v1"
_STATUS_SCHEMA = "pcraft_soft_loss_v3_status_v1"
_CONFIG_SCHEMA = "pcraft_soft_loss_v3_config_v1"
_FROZEN_STATUS = "APPROVED_FROZEN"
_FACTOR = 0.15
_EPSILON = 1e-12
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_NAME = re.compile(r"calibration_checkpoint_step_([0-9]{8})\.pt\Z")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OBSOLETE_PROJECT_PREFIX = "/home/lin/xvxinyao/ShadowForge/"

# This is an explicit policy, not a substring sweep: legitimate calibration
# fields such as ``checkpoint`` and documented median formulae remain legal.
_FORBIDDEN_QUALITY_KEYS = frozenset(
    {
        "iou",
        "dice",
        "fpr",
        "recall",
        "precision",
        "ssim",
        "boundary_f1",
        "boundary-f1",
        "mscd",
        "best",
        "best_checkpoint",
        "checkpoint_selection",
        "selection_criterion",
        "score",
        "rank",
        "quality",
        "quality_metric",
        "quality_metrics",
        "threshold_quality_summary",
    }
)
_FORBIDDEN_QUALITY_PREFIXES = (
    "iou_",
    "dice_",
    "fpr_",
    "recall_",
    "precision_",
    "ssim_",
    "boundary_f1_",
    "mscd_",
    "best_",
    "score_",
    "rank_",
    "quality_",
    "threshold_quality_",
)
_FORBIDDEN_QUALITY_ARTIFACT = "metrics.json"
_TASK8_SAMPLE_QUALITY_EXCEPTIONS = frozenset(
    {
        ("samples", "dic_dts_support_ious"),
        ("samples", "dic_dts_support_iou_invalid"),
    }
)
_ARTIFACT_INTEGRITY_ONLY_NAMES = frozenset(
    {"metrics.json", "resource_metrics.json"}
)
_RUN_IMPLEMENTATION_DIGEST_PATH = ("config", "implementation", "files_sha256")
_PROVENANCE_IMPLEMENTATION_DIGEST_PATH = ("implementation", "files_sha256")
_WEIGHT_IMPLEMENTATION_DIGEST_PATH = (
    "provenance",
    "implementation",
    "files_sha256",
)
_COMPOUND_QUALITY_ALIASES = frozenset(
    {
        "iou",
        "ious",
        "meaniou",
        "supportiou",
        "supportious",
        "boundaryf1",
        "qualitymetric",
        "qualitymetrics",
        "checkpointselection",
    }
)
_FORBIDDEN_QUALITY_TOKENS = frozenset(
    {
        "iou",
        "ious",
        "dice",
        "dices",
        "fpr",
        "fprs",
        "recall",
        "recalls",
        "precision",
        "precisions",
        "ssim",
        "ssims",
        "mscd",
        "mscds",
        "score",
        "scores",
        "rank",
        "ranks",
        "ranking",
        "rankings",
        "quality",
        "qualities",
        "metric",
        "metrics",
        "best",
        "selection",
        "selections",
    }
)

_CALIBRATION_EXECUTED_SOURCE_NAMES = frozenset(
    {
        "experiments/soft_loss_v3_registry.py",
        "experiments/bind_soft_loss_v3_suites.py",
        "experiments/run_soft_loss_v3.py",
        "experiments/calibrate_soft_loss_v3.py",
        "shadowforge_pipeline/soft_voxel_v3/calibration.py",
        "shadowforge_pipeline/soft_voxel_v3/provenance.py",
        "shadowforge_pipeline/soft_voxel_v3/trainer.py",
    }
)
_STAGE_SCIENTIFIC_FIELDS = (
    "schema",
    "status",
    "calibration_run",
    "checkpoints",
    "master_manifest",
    "development_suite",
    "specification",
    "inputs_xyz",
    "executed_source_bundle",
    "publication",
)
_WEIGHT_SCIENTIFIC_FIELDS = (
    "schema",
    "status",
    "lambdas",
    "calibration",
    "measurements",
    "provenance",
)

_CALIBRATION_DESCRIPTION = {
    "factor": _FACTOR,
    "epsilon": _EPSILON,
    "formula_dic": "0.15 * median(g0) / (median(g_dic) + 1e-12)",
    "formula_dts": (
        "0.15 * median(g0) / "
        "(median(g_dts_004,g_dts_006 by ordered update) + 1e-12)"
    ),
}


@dataclass(frozen=True)
class CalibrationMeasurement:
    """One registered component-gradient measurement and retained checkpoint."""

    completed_updates: int
    checkpoint: Path
    checkpoint_sha256: str
    g0: float
    g_dic: float
    g_dts_004: float
    g_dts_006: float


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    data: bytes
    sha256: str
    identity: tuple[int, int, int, int, int, int, int]
    root: Path | None
    label: str


@dataclass(frozen=True)
class _DependencySnapshot:
    files: tuple[_FileSnapshot, ...]


@dataclass(frozen=True)
class _CalibrationContext:
    measurements: tuple[
        CalibrationMeasurement,
        CalibrationMeasurement,
        CalibrationMeasurement,
    ]
    provenance: dict[str, object]
    dependencies: _DependencySnapshot


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON constant must be finite, found {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _decode_json(data: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _absolute(path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _reject_traversal(path: Path, *, label: str) -> None:
    if ".." in path.parts:
        raise ValueError(f"{label} contains path traversal: {path}")


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink component: {current}")


def _safe_directory(path: Path, *, label: str) -> Path:
    absolute = _absolute(path)
    _reject_traversal(absolute, label=label)
    _reject_symlink_components(absolute, label=label)
    try:
        resolved = absolute.resolve(strict=True)
        metadata = os.lstat(resolved)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {label}: {absolute}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def _snapshot_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _snapshot_file(
    path: Path,
    *,
    label: str,
    root: Path | None = None,
) -> _FileSnapshot:
    """Read and hash one canonical regular file through one no-follow fd."""

    absolute = _absolute(path)
    _reject_traversal(absolute, label=label)
    _reject_symlink_components(absolute, label=label)
    try:
        resolved = absolute.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {label}: {absolute}") from error
    safe_root = None if root is None else _safe_directory(root, label=f"{label} root")
    if safe_root is not None and not resolved.is_relative_to(safe_root):
        raise ValueError(f"{label} escapes its canonical containment root")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _snapshot_identity(before) != _snapshot_identity(after):
            raise ValueError(f"{label} changed during stable snapshot")
        current = os.lstat(resolved)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != after.st_dev
            or current.st_ino != after.st_ino
        ):
            raise ValueError(f"{label} path changed during stable snapshot")
        _reject_symlink_components(resolved, label=label)
        if resolved.resolve(strict=True) != resolved:
            raise ValueError(f"{label} is not a canonical path")
        data = b"".join(chunks)
        if len(data) != after.st_size:
            raise ValueError(f"{label} size changed during stable snapshot")
    finally:
        os.close(descriptor)
    return _FileSnapshot(
        path=resolved,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        identity=_snapshot_identity(after),
        root=safe_root,
        label=label,
    )


def _verify_snapshot_unchanged(snapshot: object) -> None:
    """Revalidate every byte and inode in a file or dependency snapshot."""

    if isinstance(snapshot, _FileSnapshot):
        files = (snapshot,)
    elif isinstance(snapshot, _DependencySnapshot):
        files = snapshot.files
    else:
        raise TypeError("snapshot must be a calibration file/dependency snapshot")
    for original in files:
        current = _snapshot_file(
            original.path,
            label=original.label,
            root=original.root,
        )
        if (
            current.identity != original.identity
            or current.sha256 != original.sha256
            or current.data != original.data
        ):
            raise ValueError(f"{original.label} changed after dependency snapshot")


def _dependency_bundle(snapshots: Sequence[_FileSnapshot]) -> _DependencySnapshot:
    unique: dict[Path, _FileSnapshot] = {}
    for snapshot in snapshots:
        existing = unique.get(snapshot.path)
        if existing is not None and (
            existing.sha256 != snapshot.sha256 or existing.data != snapshot.data
        ):
            raise ValueError(f"dependency snapshot disagrees for {snapshot.path}")
        unique[snapshot.path] = snapshot
    return _DependencySnapshot(tuple(unique[path] for path in sorted(unique, key=str)))


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} schema has missing={sorted(expected - observed)} "
            f"extra={sorted(observed - expected)}"
        )


def _exact_value(value: object, expected: object, *, label: str) -> None:
    if type(value) is not type(expected):
        raise ValueError(f"{label} has wrong exact type")
    if isinstance(expected, Mapping):
        observed = value
        _exact_keys(observed, set(expected), label=label)
        for key, item in expected.items():
            _exact_value(observed[key], item, label=f"{label}.{key}")
        return
    if isinstance(expected, list):
        observed = value
        if len(observed) != len(expected):
            raise ValueError(f"{label} has wrong exact list length")
        for index, (actual, wanted) in enumerate(zip(observed, expected)):
            _exact_value(actual, wanted, label=f"{label}[{index}]")
        return
    if value != expected:
        raise ValueError(f"{label} does not match the frozen value")


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonempty_string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _absolute_path_text(value: object, *, label: str) -> str:
    text = _nonempty_string(value, label=label)
    path = Path(text)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a canonical absolute path")
    return text


def _reject_quality_fields(
    value: object,
    *,
    label: str,
    allowed_key_paths: frozenset[tuple[str, ...]] = frozenset(),
    ignored_subtree_paths: frozenset[tuple[str, ...]] = frozenset(),
    _path: tuple[str, ...] = (),
) -> None:
    # Source filenames are immutable-integrity identities, not calibration
    # inputs.  Callers may exempt only an exact ``files_sha256`` subtree that
    # is independently schema-checked and rebound to current source bytes by
    # ``_validate_implementation`` / ``_load_context``.
    if _path in ignored_subtree_paths:
        return
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if type(raw_key) is not str:
                raise ValueError(f"{label} has a non-string JSON key")
            field_path = (*_path, raw_key)
            key_path = raw_key.replace("\\", "/").casefold()
            camel_split = re.sub(
                r"(?<=[a-z0-9])(?=[A-Z])",
                "_",
                re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", raw_key),
            )
            ordered_tokens = tuple(
                token.casefold()
                for token in re.findall(r"[A-Za-z0-9]+", camel_split)
                if token
            )
            tokens = set(ordered_tokens)
            normalized_key = "_".join(ordered_tokens)
            compound_aliases = {
                "".join(ordered_tokens[start:end])
                for start in range(len(ordered_tokens))
                for end in range(start + 1, min(len(ordered_tokens), start + 3) + 1)
            }
            forbidden = (
                any(
                    part == _FORBIDDEN_QUALITY_ARTIFACT
                    for part in key_path.split("/")
                )
                or normalized_key in _FORBIDDEN_QUALITY_KEYS
                or any(
                    normalized_key.startswith(prefix)
                    for prefix in _FORBIDDEN_QUALITY_PREFIXES
                )
                or bool(
                    tokens.intersection(_FORBIDDEN_QUALITY_TOKENS)
                )
                or ({"boundary", "f1"} <= tokens)
                or bool(
                    compound_aliases.intersection(
                        _COMPOUND_QUALITY_ALIASES | _FORBIDDEN_QUALITY_TOKENS
                    )
                )
            )
            if field_path not in allowed_key_paths and forbidden:
                raise ValueError(
                    f"forbidden task-quality metric or artifact key in {label}: {raw_key}"
                )
            _reject_quality_fields(
                item,
                label=f"{label}.{raw_key}",
                allowed_key_paths=allowed_key_paths,
                ignored_subtree_paths=ignored_subtree_paths,
                _path=field_path,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_quality_fields(
                item,
                label=f"{label}[{index}]",
                allowed_key_paths=allowed_key_paths,
                ignored_subtree_paths=ignored_subtree_paths,
                _path=_path,
            )
        return
    if isinstance(value, str):
        normalized = value.replace("\\", "/").casefold()
        if any(part == _FORBIDDEN_QUALITY_ARTIFACT for part in normalized.split("/")):
            raise ValueError(f"forbidden task-quality artifact reference in {label}")


def _validate_float(value: object, *, label: str) -> float:
    if type(value) is not float:
        raise ValueError(f"{label} must have exact float type")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if value < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _validate_measurement_identity(
    measurements: Sequence[CalibrationMeasurement],
    *,
    snapshot_checkpoints: bool,
) -> tuple[
    tuple[CalibrationMeasurement, CalibrationMeasurement, CalibrationMeasurement],
    tuple[_FileSnapshot, ...],
]:
    if isinstance(measurements, (str, bytes)) or not isinstance(measurements, Sequence):
        raise ValueError("measurements must be a sequence of exactly three records")
    if len(measurements) != 3:
        raise ValueError("calibration requires exactly three measurements")
    normalized: list[CalibrationMeasurement] = []
    snapshots: list[_FileSnapshot] = []
    seen_paths: set[Path] = set()
    seen_inodes: set[tuple[int, int]] = set()
    for expected_update, item in zip(_UPDATES, measurements):
        if not isinstance(item, CalibrationMeasurement):
            raise ValueError("measurements must contain CalibrationMeasurement records")
        if type(item.completed_updates) is not int:
            raise ValueError("calibration update identity must have exact integer type")
        if item.completed_updates != expected_update:
            raise ValueError("calibration measurements must be ordered at 1000, 1300, 1600")
        if not isinstance(item.checkpoint, Path):
            raise ValueError("checkpoint must have Path type")
        if not item.checkpoint.is_absolute() or ".." in item.checkpoint.parts:
            raise ValueError("checkpoint must be a resolved canonical absolute path")
        if item.checkpoint in seen_paths:
            raise ValueError("duplicate calibration checkpoint")
        expected_name = f"calibration_checkpoint_step_{expected_update:08d}.pt"
        if item.checkpoint.name != expected_name:
            raise ValueError("checkpoint name does not match its registered update")
        expected_digest = _digest(item.checkpoint_sha256, label="checkpoint sha256")
        gradients = {
            "g0": _validate_float(item.g0, label="g0"),
            "g_dic": _validate_float(item.g_dic, label="g_dic"),
            "g_dts_004": _validate_float(item.g_dts_004, label="g_dts_004"),
            "g_dts_006": _validate_float(item.g_dts_006, label="g_dts_006"),
        }
        if snapshot_checkpoints:
            snapshot = _snapshot_file(item.checkpoint, label=f"checkpoint {expected_update}")
            if snapshot.path != item.checkpoint:
                raise ValueError("checkpoint path is not canonical and resolved")
            if snapshot.sha256 != expected_digest:
                raise ValueError("checkpoint sha256 changed from the registered hash")
            inode = (snapshot.identity[0], snapshot.identity[1])
            if snapshot.path in seen_paths or inode in seen_inodes:
                raise ValueError("duplicate calibration checkpoint")
            seen_paths.add(snapshot.path)
            seen_inodes.add(inode)
            snapshots.append(snapshot)
        elif item.checkpoint in seen_paths:
            raise ValueError("duplicate calibration checkpoint")
        else:
            seen_paths.add(item.checkpoint)
        normalized.append(
            CalibrationMeasurement(
                completed_updates=expected_update,
                checkpoint=item.checkpoint,
                checkpoint_sha256=expected_digest,
                **gradients,
            )
        )
    return (normalized[0], normalized[1], normalized[2]), tuple(snapshots)


def derive_calibrated_weights(
    measurements: Sequence[CalibrationMeasurement],
) -> dict[str, float]:
    """Derive the two frozen coefficients from exact registered gradients."""

    checked, _ = _validate_measurement_identity(measurements, snapshot_checkpoints=True)
    baseline = float(median(item.g0 for item in checked))
    dic = float(median(item.g_dic for item in checked))
    dts = float(
        median(
            value
            for item in checked
            for value in (item.g_dts_004, item.g_dts_006)
        )
    )
    for label, value in (("baseline", baseline), ("DIC", dic), ("DTS", dts)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} gradient median must be finite and positive, not zero")
    lambda_dic = _FACTOR * baseline / (dic + _EPSILON)
    lambda_dts = _FACTOR * baseline / (dts + _EPSILON)
    if not math.isfinite(lambda_dic) or lambda_dic <= 0.0:
        raise ValueError("derived lambda_dic must be finite and positive")
    if not math.isfinite(lambda_dts) or lambda_dts <= 0.0:
        raise ValueError("derived lambda_dts must be finite and positive")
    return {"lambda_dic": lambda_dic, "lambda_dts": lambda_dts}


def _checkpoint_from_record(
    record: Mapping[str, object],
    *,
    expected_update: int,
    run_root: Path,
) -> tuple[CalibrationMeasurement, _FileSnapshot]:
    _exact_keys(
        record,
        {
            "completed_updates",
            "g0",
            "g_dic",
            "g_dts_004",
            "g_dts_006",
            "checkpoint",
            "checkpoint_sha256",
        },
        label="calibration audit record",
    )
    if type(record["completed_updates"]) is not int:
        raise ValueError("calibration record update must have exact integer type")
    if record["completed_updates"] != expected_update:
        raise ValueError("calibration records are not in exact 1000, 1300, 1600 order")
    name = _nonempty_string(record["checkpoint"], label="calibration checkpoint name")
    expected_name = f"calibration_checkpoint_step_{expected_update:08d}.pt"
    if name != expected_name or PurePosixPath(name).as_posix() != name:
        raise ValueError("calibration checkpoint name is not canonical for its update")
    checkpoint = run_root / name
    snapshot = _snapshot_file(
        checkpoint,
        label=f"calibration checkpoint {expected_update}",
        root=run_root,
    )
    digest = _digest(record["checkpoint_sha256"], label="checkpoint_sha256")
    if snapshot.sha256 != digest:
        raise ValueError("calibration checkpoint hash changed from gradient audit")
    measurement = CalibrationMeasurement(
        completed_updates=expected_update,
        checkpoint=snapshot.path,
        checkpoint_sha256=digest,
        g0=_validate_float(record["g0"], label="g0"),
        g_dic=_validate_float(record["g_dic"], label="g_dic"),
        g_dts_004=_validate_float(record["g_dts_004"], label="g_dts_004"),
        g_dts_006=_validate_float(record["g_dts_006"], label="g_dts_006"),
    )
    return measurement, snapshot


_CALIBRATION_SAMPLE_KEYS = {
    "completed_updates",
    "base_norm",
    "dic_norm",
    "dts_norms",
    "base_dic_cosine",
    "base_dic_cosine_invalid",
    "base_dts_cosines",
    "base_dts_cosine_invalid",
    "dic_dts_cosines",
    "dic_dts_cosine_invalid",
    "dts_pair_cosines",
    "dts_pair_cosine_invalid",
    "dic_dts_support_ious",
    "dic_dts_support_iou_invalid",
    "weighted_dic_base_ratio",
    "weighted_dic_base_ratio_invalid",
    "weighted_dts_base_ratios",
    "weighted_dts_base_ratio_invalid",
    "selected_dts_tau",
    "combined_aux_base_ratio",
    "combined_aux_base_ratio_invalid",
    "base_combined_aux_cosine",
    "base_combined_aux_cosine_invalid",
}
_ENDPOINT_KEYS = {"0.04", "0.06"}


def _sample_optional_float(
    value: object,
    invalid: object,
    *,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> float | None:
    if type(invalid) is not bool:
        raise ValueError(f"{label} invalid flag must have exact boolean type")
    if value is None:
        if invalid is not True:
            raise ValueError(f"{label} null value requires an invalid flag")
        return None
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must have exact finite float type")
    number = value
    if invalid is not False:
        raise ValueError(f"{label} finite value must not be marked invalid")
    if lower is not None and number < lower:
        raise ValueError(f"{label} is below its valid range")
    if upper is not None and number > upper:
        raise ValueError(f"{label} is above its valid range")
    return number


def _sample_endpoint_mapping(
    values: object,
    flags: object,
    *,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> dict[str, float | None]:
    value_map = _mapping(values, label=label)
    flag_map = _mapping(flags, label=f"{label} invalid flags")
    _exact_keys(value_map, _ENDPOINT_KEYS, label=label)
    _exact_keys(flag_map, _ENDPOINT_KEYS, label=f"{label} invalid flags")
    return {
        endpoint: _sample_optional_float(
            value_map[endpoint],
            flag_map[endpoint],
            label=f"{label}[{endpoint}]",
            lower=lower,
            upper=upper,
        )
        for endpoint in ("0.04", "0.06")
    }


def _validate_calibration_sample(
    value: object,
    *,
    expected_update: int,
) -> dict[str, object]:
    sample = _mapping(value, label=f"gradient sample {expected_update}")
    _exact_keys(sample, _CALIBRATION_SAMPLE_KEYS, label="gradient sample")
    if type(sample["completed_updates"]) is not int:
        raise ValueError("gradient sample update must have exact integer type")
    if sample["completed_updates"] != expected_update:
        raise ValueError("gradient samples are not in exact 1000, 1300, 1600 order")
    base_norm = _validate_float(sample["base_norm"], label="sample base_norm")
    dic_norm = _validate_float(sample["dic_norm"], label="sample dic_norm")
    dts_norms = _mapping(sample["dts_norms"], label="sample dts_norms")
    _exact_keys(dts_norms, _ENDPOINT_KEYS, label="sample dts_norms")
    normalized_dts = {
        endpoint: _validate_float(dts_norms[endpoint], label=f"sample dts_norms[{endpoint}]")
        for endpoint in ("0.04", "0.06")
    }
    _sample_optional_float(
        sample["base_dic_cosine"],
        sample["base_dic_cosine_invalid"],
        label="sample base_dic_cosine",
        lower=-1.0,
        upper=1.0,
    )
    for value_name, flag_name in (
        ("base_dts_cosines", "base_dts_cosine_invalid"),
        ("dic_dts_cosines", "dic_dts_cosine_invalid"),
    ):
        _sample_endpoint_mapping(
            sample[value_name],
            sample[flag_name],
            label=f"sample {value_name}",
            lower=-1.0,
            upper=1.0,
        )
    _sample_endpoint_mapping(
        sample["dic_dts_support_ious"],
        sample["dic_dts_support_iou_invalid"],
        label="sample dic_dts_support_ious",
        lower=0.0,
        upper=1.0,
    )
    _sample_optional_float(
        sample["weighted_dic_base_ratio"],
        sample["weighted_dic_base_ratio_invalid"],
        label="sample weighted_dic_base_ratio",
        lower=0.0,
    )
    _sample_endpoint_mapping(
        sample["weighted_dts_base_ratios"],
        sample["weighted_dts_base_ratio_invalid"],
        label="sample weighted_dts_base_ratios",
        lower=0.0,
    )
    if type(sample["selected_dts_tau"]) is not float or sample["selected_dts_tau"] != 0.04:
        raise ValueError("calibration sample selected_dts_tau must be exact 0.04")
    _sample_optional_float(
        sample["combined_aux_base_ratio"],
        sample["combined_aux_base_ratio_invalid"],
        label="sample combined_aux_base_ratio",
        lower=0.0,
    )
    _sample_optional_float(
        sample["base_combined_aux_cosine"],
        sample["base_combined_aux_cosine_invalid"],
        label="sample base_combined_aux_cosine",
        lower=-1.0,
        upper=1.0,
    )
    pair_values = _mapping(sample["dts_pair_cosines"], label="sample DTS pair cosines")
    pair_flags = _mapping(
        sample["dts_pair_cosine_invalid"], label="sample DTS pair cosine flags"
    )
    _exact_keys(pair_values, {"0.04"}, label="sample DTS pair cosines")
    _exact_keys(pair_flags, {"0.04"}, label="sample DTS pair cosine flags")
    pair_value = _mapping(pair_values["0.04"], label="sample DTS pair 0.04")
    pair_flag = _mapping(pair_flags["0.04"], label="sample DTS pair 0.04 flags")
    _exact_keys(pair_value, {"0.06"}, label="sample DTS pair 0.04")
    _exact_keys(pair_flag, {"0.06"}, label="sample DTS pair 0.04 flags")
    _sample_optional_float(
        pair_value["0.06"],
        pair_flag["0.06"],
        label="sample DTS pair cosine 0.04/0.06",
        lower=-1.0,
        upper=1.0,
    )
    normalized = dict(sample)
    normalized["base_norm"] = base_norm
    normalized["dic_norm"] = dic_norm
    normalized["dts_norms"] = normalized_dts
    return normalized


def _validate_calibration_samples(value: object) -> tuple[dict[str, object], ...]:
    samples = _list(value, label="gradient audit samples")
    if len(samples) != 3:
        raise ValueError("gradient audit must contain exactly three Task 8 samples")
    return tuple(
        _validate_calibration_sample(sample, expected_update=update)
        for update, sample in zip(_UPDATES, samples)
    )


def _validate_calibration_invalid_flags(value: object) -> None:
    records = _list(value, label="gradient audit active_invalid_flags")
    if len(records) != 3:
        raise ValueError("gradient audit must contain three active-invalid records")
    for update, raw_record in zip(_UPDATES, records):
        record = _mapping(raw_record, label="active-invalid record")
        _exact_keys(record, {"completed_updates", "flags"}, label="active-invalid record")
        if type(record["completed_updates"]) is not int or record["completed_updates"] != update:
            raise ValueError("active-invalid records have wrong update/order identity")
        flags = _list(record["flags"], label="active-invalid flags")
        if flags:
            raise ValueError("l12_bal calibration must have no active auxiliary invalid flags")


def _measurements_from_audit_snapshot(
    snapshot: _FileSnapshot,
    *,
    expected_config_hash: str | None = None,
) -> tuple[
    tuple[CalibrationMeasurement, CalibrationMeasurement, CalibrationMeasurement],
    tuple[_FileSnapshot, ...],
    dict[str, object],
]:
    payload = _decode_json(snapshot.data, label="gradient audit")
    _reject_quality_fields(
        payload,
        label="gradient audit",
        allowed_key_paths=_TASK8_SAMPLE_QUALITY_EXCEPTIONS,
    )
    _exact_keys(
        payload,
        {"schema", "config_hash", "samples", "calibration_records", "active_invalid_flags"},
        label="gradient audit",
    )
    if payload["schema"] != _AUDIT_SCHEMA:
        raise ValueError("gradient audit schema mismatch")
    config_hash = _digest(payload["config_hash"], label="gradient audit config_hash")
    if expected_config_hash is not None and config_hash != expected_config_hash:
        raise ValueError("gradient audit config identity mismatch")
    samples = _validate_calibration_samples(payload["samples"])
    _validate_calibration_invalid_flags(payload["active_invalid_flags"])
    records = _list(payload["calibration_records"], label="calibration records")
    if len(records) != 3:
        raise ValueError("gradient audit must contain exactly three calibration records")
    run_root = snapshot.path.parent
    measurements: list[CalibrationMeasurement] = []
    checkpoints: list[_FileSnapshot] = []
    for update, raw_record in zip(_UPDATES, records):
        measurement, checkpoint = _checkpoint_from_record(
            _mapping(raw_record, label="calibration record"),
            expected_update=update,
            run_root=run_root,
        )
        measurements.append(measurement)
        checkpoints.append(checkpoint)
    for measurement, sample in zip(measurements, samples):
        expected = {
            "g0": sample["base_norm"],
            "g_dic": sample["dic_norm"],
            "g_dts_004": sample["dts_norms"]["0.04"],
            "g_dts_006": sample["dts_norms"]["0.06"],
        }
        if any(getattr(measurement, name) != value for name, value in expected.items()):
            raise ValueError(
                "calibration record gradient values disagree with the Task 8 sample"
            )
    checked, _ = _validate_measurement_identity(measurements, snapshot_checkpoints=False)
    if len({(item.identity[0], item.identity[1]) for item in checkpoints}) != 3:
        raise ValueError("duplicate calibration checkpoints")
    return checked, tuple(checkpoints), payload


def load_calibration_measurements(
    path: Path,
) -> tuple[CalibrationMeasurement, CalibrationMeasurement, CalibrationMeasurement]:
    """Load the exact Task-8 audit and bind its three retained checkpoints."""

    audit = _snapshot_file(path, label="gradient audit")
    measurements, _, _ = _measurements_from_audit_snapshot(audit)
    return measurements


def _protocol() -> dict[str, object]:
    return {
        "views": 3,
        "axes": ["x", "y", "z"],
        "seed": 17,
        "camera": "TFS",
        "mirror_enabled": True,
        "image_size": 256,
        "zdist": 1.7,
        "extent": 1.7,
        "run_policy": "fixed_seed_single_run",
        "determinism": "best_effort_cuda",
        "bitwise_reproducible": False,
        "disclosure": (
            "CUDA grid_sampler_3d_backward may not provide bitwise "
            "deterministic backward execution."
        ),
    }


def _appendix_controls() -> dict[str, object]:
    return {
        "variants": list(_APPENDIX_VARIANTS),
        "included_in_primary_factorial": False,
        "timing": "after_locked_formal_factorial_matrix",
    }


def _formal_run_roles() -> dict[str, object]:
    return {
        "primary": "confirmatory_first_completed_run",
        "repeat": "technical_sensitivity_audit_only",
        "repeat_is_independent_sample": False,
    }


def _primary_matrix() -> list[dict[str, object]]:
    return [
        {"variant": "l12_bal", "balanced_l1_l2": True, "dic": False, "dts": False},
        {"variant": "dic_only", "balanced_l1_l2": True, "dic": True, "dts": False},
        {"variant": "dts_only", "balanced_l1_l2": True, "dic": False, "dts": True},
        {"variant": "v3_full", "balanced_l1_l2": True, "dic": True, "dts": True},
    ]


def _registered_source_path(
    source: str,
    resolved: str,
    *,
    root: Path,
    label: str,
) -> _FileSnapshot:
    source_path = Path(source)
    resolved_path = Path(resolved)
    if not source_path.is_absolute() or not resolved_path.is_absolute():
        raise ValueError(f"{label} paths must be absolute")
    if source != resolved:
        if not source.startswith(_OBSOLETE_PROJECT_PREFIX):
            raise ValueError(f"{label} has an unregistered source-path rewrite")
        suffix = source[len(_OBSOLETE_PROJECT_PREFIX) :]
        if not suffix or ".." in Path(suffix).parts or Path(suffix).is_absolute():
            raise ValueError(f"{label} obsolete-prefix rewrite is unsafe")
        if resolved_path != _PROJECT_ROOT / suffix:
            raise ValueError(f"{label} source/resolved identity mismatch")
    return _snapshot_file(resolved_path, label=label, root=root)


def _validate_source_record(
    value: object,
    *,
    role: str,
) -> tuple[dict[str, object], dict[str, object], _FileSnapshot, Path]:
    record = _mapping(value, label=f"{role} source_manifest")
    _exact_keys(
        record,
        {"source_path", "resolved_path", "sha256", "semantic_sha256"},
        label=f"{role} source_manifest",
    )
    source = _absolute_path_text(record["source_path"], label="source manifest source_path")
    resolved = _absolute_path_text(
        record["resolved_path"], label="source manifest resolved_path"
    )
    if source != resolved:
        raise ValueError("bound source manifest source_path/resolved_path must match")
    root = Path(resolved).parent
    snapshot = _snapshot_file(Path(resolved), label=f"{role} source manifest", root=root)
    if snapshot.sha256 != _digest(record["sha256"], label="source manifest sha256"):
        raise ValueError("bound source manifest bytes changed")
    payload = _decode_json(snapshot.data, label=f"{role} source manifest")
    if role == "development":
        semantics = _development_source_semantics(payload)
    else:
        semantics = _formal_source_semantics(payload)
    if canonical_hash(semantics) != _digest(
        record["semantic_sha256"], label="source manifest semantic_sha256"
    ):
        raise ValueError("source manifest semantic identity changed")
    return dict(record), semantics, snapshot, root


def _development_source_semantics(payload: Mapping[str, object]) -> dict[str, object]:
    _exact_keys(payload, {"protocol", "foreground", "tasks"}, label="development source")
    _exact_value(payload["protocol"], "fixed-3-view TFS, mr=1, 256^3", label="development protocol")
    _exact_value(payload["foreground"], "white-on-black", label="development foreground")
    tasks = _list(payload["tasks"], label="development source tasks")
    if len(tasks) != 3:
        raise ValueError("development source must contain exactly three ordered tasks")
    normalized: list[dict[str, object]] = []
    for index, (case_id, raw_task) in enumerate(zip(_DEVELOPMENT_IDS, tasks)):
        task = _mapping(raw_task, label=f"development source task {index}")
        _exact_keys(task, {"task", "views"}, label=f"development source task {index}")
        _exact_value(task["task"], case_id, label="development task identity")
        views = _list(task["views"], label=f"{case_id} views")
        if len(views) != 3:
            raise ValueError(f"{case_id} must have exactly three x/y/z views")
        checked: list[str] = []
        for view_index, raw_path in enumerate(views):
            text = _absolute_path_text(raw_path, label=f"{case_id} source view")
            if Path(text).name != f"view_{view_index}.png":
                raise ValueError(f"{case_id} source views are not in x/y/z order")
            checked.append(text)
        normalized.append({"task": case_id, "views": checked})
    return {
        "protocol": payload["protocol"],
        "foreground": payload["foreground"],
        "tasks": normalized,
    }


def _formal_source_semantics(payload: Mapping[str, object]) -> dict[str, object]:
    required = {"schema", "status", "protocol", "cases"}
    if not required.issubset(payload):
        raise ValueError("formal source manifest is missing required schema fields")
    _exact_value(payload["schema"], "fixed3_mr1_connected10_v1", label="formal source schema")
    _exact_value(payload["status"], _FROZEN_STATUS, label="formal source status")
    expected_protocol = {
        "views": 3,
        "declared_rendered_views": 6,
        "axes": ["x", "y", "z"],
        "camera": "fixed orthographic TFS",
        "tsf": 1,
        "mr": 1,
        "zdist": 1.7,
        "resolution": 256,
        "seeds": [17, 29, 43],
        "evaluation": "shared canonical masks and prescribed mirrors; no registration",
    }
    _exact_value(payload["protocol"], expected_protocol, label="formal source protocol")
    cases = _list(payload["cases"], label="formal source cases")
    if len(cases) != 10:
        raise ValueError("formal source must contain exactly ten ordered cases")
    normalized: list[dict[str, object]] = []
    for index, (case_id, raw_case) in enumerate(zip(_FORMAL_IDS, cases)):
        case = _mapping(raw_case, label=f"formal source case {index}")
        if "case_id" not in case or "target_paths_xyz" not in case:
            raise ValueError("formal source case is missing identity/targets")
        _exact_value(case["case_id"], case_id, label="formal case identity")
        targets = _list(case["target_paths_xyz"], label=f"{case_id} targets")
        if len(targets) != 3:
            raise ValueError(f"{case_id} must have exactly three targets")
        checked: list[str] = []
        for axis, raw_path in zip(_AXES, targets):
            text = _absolute_path_text(raw_path, label=f"{case_id} {axis} target")
            if not Path(text).name.endswith(f"_view_{axis}.png"):
                raise ValueError(f"{case_id} targets are not in x/y/z order")
            checked.append(text)
        normalized.append({"case_id": case_id, "target_paths_xyz": checked})
    return {
        "schema": payload["schema"],
        "status": payload["status"],
        "protocol": expected_protocol,
        "cases": normalized,
    }


def _suite_semantics(payload: Mapping[str, object]) -> dict[str, object]:
    fields = [
        "schema",
        "status",
        "role",
        "source_manifest",
        "tasks",
        "task_count",
        "input_count",
        "input_order",
        "protocol",
        "primary_variants",
        "appendix_controls",
    ]
    if payload.get("role") == "formal":
        fields.append("formal_run_roles")
    return {field: payload[field] for field in fields}


def _validate_bound_suite(
    snapshot: _FileSnapshot,
    *,
    role: str,
    validate_inputs: bool,
) -> tuple[dict[str, object], list[_FileSnapshot]]:
    payload = _decode_json(snapshot.data, label=f"{role} suite")
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError(f"{role} suite is not finite canonical JSON")
    expected_keys = {
        "schema",
        "status",
        "role",
        "source_manifest",
        "tasks",
        "task_count",
        "input_count",
        "input_order",
        "protocol",
        "primary_variants",
        "appendix_controls",
        "semantic_sha256",
    }
    if role == "formal":
        expected_keys.add("formal_run_roles")
    _exact_keys(payload, expected_keys, label=f"{role} suite")
    _exact_value(payload["schema"], _SUITE_SCHEMA, label="suite schema")
    _exact_value(payload["status"], _FROZEN_STATUS, label="suite status")
    _exact_value(payload["role"], role, label="suite role")
    _exact_value(payload["input_order"], ["x", "y", "z"], label="suite input order")
    _exact_value(payload["protocol"], _protocol(), label="suite protocol")
    _exact_value(payload["primary_variants"], list(_PRIMARY_VARIANTS), label="suite variants")
    _exact_value(payload["appendix_controls"], _appendix_controls(), label="suite appendix")
    if role == "formal":
        _exact_value(payload["formal_run_roles"], _formal_run_roles(), label="formal run roles")
    if canonical_hash(_suite_semantics(payload)) != _digest(
        payload["semantic_sha256"], label="suite semantic_sha256"
    ):
        raise ValueError("suite semantic identity mismatch")
    source_record, source_semantics, source_snapshot, source_root = _validate_source_record(
        payload["source_manifest"], role=role
    )
    del source_record
    task_ids = _DEVELOPMENT_IDS if role == "development" else _FORMAL_IDS
    tasks = _list(payload["tasks"], label=f"{role} tasks")
    if len(tasks) != len(task_ids):
        raise ValueError(f"{role} suite has wrong ordered task count")
    if type(payload["task_count"]) is not int or payload["task_count"] != len(task_ids):
        raise ValueError("suite task_count must have exact integer value")
    if type(payload["input_count"]) is not int or payload["input_count"] != 3 * len(task_ids):
        raise ValueError("suite input_count must have exact integer value")
    dependencies = [source_snapshot]
    seen_paths: set[Path] = set()
    seen_hashes: set[str] = set()
    for task_index, (case_id, raw_task) in enumerate(zip(task_ids, tasks)):
        task = _mapping(raw_task, label=f"{role} task {task_index}")
        _exact_keys(task, {"case_id", "inputs"}, label=f"{role} task {task_index}")
        _exact_value(task["case_id"], case_id, label="suite case identity")
        inputs = _list(task["inputs"], label=f"{case_id} inputs")
        if len(inputs) != 3:
            raise ValueError(f"{case_id} must contain three ordered inputs")
        source_task = source_semantics["tasks" if role == "development" else "cases"][task_index]
        source_paths = source_task["views" if role == "development" else "target_paths_xyz"]
        for input_index, (axis, raw_input, source_path) in enumerate(
            zip(_AXES, inputs, source_paths)
        ):
            item = _mapping(raw_input, label=f"{case_id} input {input_index}")
            _exact_keys(
                item,
                {"axis", "source_path", "resolved_path", "sha256"},
                label=f"{case_id} input {input_index}",
            )
            _exact_value(item["axis"], axis, label=f"{case_id} input axis")
            _exact_value(item["source_path"], source_path, label=f"{case_id} input source")
            resolved = _absolute_path_text(item["resolved_path"], label=f"{case_id} resolved input")
            digest = _digest(item["sha256"], label=f"{case_id} input sha256")
            if validate_inputs:
                input_snapshot = _registered_source_path(
                    source_path,
                    resolved,
                    root=source_root,
                    label=f"{case_id} {axis} input",
                )
                if input_snapshot.sha256 != digest:
                    raise ValueError(f"{case_id} {axis} input bytes changed")
                if input_snapshot.path in seen_paths or digest in seen_hashes:
                    raise ValueError("duplicate suite input path or bytes")
                seen_paths.add(input_snapshot.path)
                seen_hashes.add(digest)
                dependencies.append(input_snapshot)
    return payload, dependencies


def _master_semantics(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        field: payload[field]
        for field in (
            "schema",
            "status",
            "suites",
            "design_spec",
            "primary_matrix",
            "appendix_controls",
            "protocol",
            "determinism",
            "formal_run_roles",
        )
    }


def _validate_master(
    snapshot: _FileSnapshot,
    *,
    development_snapshot: _FileSnapshot,
) -> tuple[dict[str, object], dict[str, object], list[_FileSnapshot]]:
    payload = _decode_json(snapshot.data, label="master manifest")
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError("master manifest is not finite canonical JSON")
    _exact_keys(
        payload,
        {
            "schema",
            "status",
            "suites",
            "design_spec",
            "primary_matrix",
            "appendix_controls",
            "protocol",
            "determinism",
            "formal_run_roles",
            "semantic_sha256",
        },
        label="master manifest",
    )
    _exact_value(payload["schema"], _MASTER_SCHEMA, label="master schema")
    _exact_value(payload["status"], _FROZEN_STATUS, label="master status")
    _exact_value(payload["primary_matrix"], _primary_matrix(), label="master primary matrix")
    _exact_value(payload["appendix_controls"], _appendix_controls(), label="master appendix")
    _exact_value(payload["protocol"], _protocol(), label="master protocol")
    _exact_value(payload["formal_run_roles"], _formal_run_roles(), label="master run roles")
    determinism = _mapping(payload["determinism"], label="master determinism")
    _exact_keys(
        determinism,
        {"policy", "bitwise_reproducible", "reason"},
        label="master determinism",
    )
    _exact_value(
        determinism["policy"],
        "fixed_seed_single_run_best_effort_cuda",
        label="master determinism policy",
    )
    _exact_value(determinism["bitwise_reproducible"], False, label="master determinism bit")
    _exact_value(determinism["reason"], _protocol()["disclosure"], label="master reason")
    if canonical_hash(_master_semantics(payload)) != _digest(
        payload["semantic_sha256"], label="master semantic_sha256"
    ):
        raise ValueError("master semantic identity mismatch")
    suites = _mapping(payload["suites"], label="master suites")
    _exact_keys(suites, {"development", "formal"}, label="master suites")
    suite_records: dict[str, Mapping[str, object]] = {}
    dependencies: list[_FileSnapshot] = []
    for role, expected_path, task_count in (
        ("development", "manifests/development_suite.json", 3),
        ("formal", "manifests/formal_suite.json", 10),
    ):
        record = _mapping(suites[role], label=f"master {role} suite")
        _exact_keys(
            record,
            {"path", "sha256", "semantic_sha256", "task_count", "input_count"},
            label=f"master {role} suite",
        )
        _exact_value(record["path"], expected_path, label=f"master {role} path")
        if type(record["task_count"]) is not int or record["task_count"] != task_count:
            raise ValueError(f"master {role} task_count mismatch")
        if type(record["input_count"]) is not int or record["input_count"] != 3 * task_count:
            raise ValueError(f"master {role} input_count mismatch")
        _digest(record["sha256"], label=f"master {role} sha256")
        _digest(record["semantic_sha256"], label=f"master {role} semantic_sha256")
        suite_records[role] = record
    expected_development = snapshot.path.parent / str(suite_records["development"]["path"])
    if development_snapshot.path != expected_development:
        raise ValueError("development suite argument does not match master identity")
    if development_snapshot.sha256 != suite_records["development"]["sha256"]:
        raise ValueError("development suite current hash does not match master")
    formal_snapshot = _snapshot_file(
        snapshot.path.parent / str(suite_records["formal"]["path"]),
        label="formal suite",
        root=snapshot.path.parent,
    )
    if formal_snapshot.sha256 != suite_records["formal"]["sha256"]:
        raise ValueError("formal suite current hash does not match master")
    formal_payload, formal_dependencies = _validate_bound_suite(
        formal_snapshot, role="formal", validate_inputs=False
    )
    if formal_payload["semantic_sha256"] != suite_records["formal"]["semantic_sha256"]:
        raise ValueError("formal suite semantic identity does not match master")
    design = _mapping(payload["design_spec"], label="master design_spec")
    _exact_keys(design, {"path", "sha256"}, label="master design_spec")
    _absolute_path_text(design["path"], label="master spec path")
    _digest(design["sha256"], label="master spec sha256")
    dependencies.extend([formal_snapshot, *formal_dependencies])
    return payload, dict(design), dependencies


def _validate_implementation(
    value: object,
) -> tuple[dict[str, object], list[_FileSnapshot]]:
    implementation = _mapping(value, label="implementation identity")
    _exact_keys(
        implementation,
        {"algorithm", "files_sha256", "bundle_sha256"},
        label="implementation identity",
    )
    _exact_value(implementation["algorithm"], "sha256", label="implementation algorithm")
    files = _mapping(implementation["files_sha256"], label="implementation files")
    if not files or not all(type(name) is str and name for name in files):
        raise ValueError("implementation files must be a non-empty string-keyed mapping")
    checked_files = {
        name: _digest(digest, label=f"implementation {name}")
        for name, digest in files.items()
    }
    if canonical_hash(checked_files) != _digest(
        implementation["bundle_sha256"], label="implementation bundle_sha256"
    ):
        raise ValueError("implementation bundle identity mismatch")
    normalized = {
        "algorithm": "sha256",
        "files_sha256": checked_files,
        "bundle_sha256": implementation["bundle_sha256"],
    }
    current = provenance_module.implementation_source_manifest()
    _exact_value(normalized, current, label="current authoritative v3 source bundle")
    root = _safe_directory(
        Path(provenance_module.AUTHORITATIVE_SOURCE_ROOT),
        label="authoritative v3 source root",
    )
    dependencies: list[_FileSnapshot] = []
    for relative, digest in sorted(checked_files.items()):
        logical = PurePosixPath(relative)
        if (
            logical.is_absolute()
            or not logical.parts
            or any(part in ("", ".", "..") for part in logical.parts)
            or logical.as_posix() != relative
            or logical.suffix != ".py"
        ):
            raise ValueError("implementation source name is not canonical Python")
        snapshot = _snapshot_file(
            root / Path(*logical.parts),
            label=f"authoritative v3 source {relative}",
            root=root,
        )
        if snapshot.sha256 != digest:
            raise ValueError(f"current authoritative v3 source changed: {relative}")
        dependencies.append(snapshot)
    return normalized, dependencies


def _validate_executed_source_bundle(
    value: object,
) -> tuple[dict[str, object], list[_FileSnapshot]]:
    bundle = _mapping(value, label="calibration executed-source bundle")
    _exact_keys(
        bundle,
        {"algorithm", "files", "bundle_sha256"},
        label="calibration executed-source bundle",
    )
    _exact_value(bundle["algorithm"], "sha256", label="executed-source algorithm")
    files = _mapping(bundle["files"], label="executed-source files")
    _exact_keys(files, set(_CALIBRATION_EXECUTED_SOURCE_NAMES), label="executed-source files")
    normalized_files: dict[str, object] = {}
    dependencies: list[_FileSnapshot] = []
    seen_paths: set[Path] = set()
    seen_inodes: set[tuple[int, int]] = set()
    for logical_name in sorted(_CALIBRATION_EXECUTED_SOURCE_NAMES):
        record = _mapping(files[logical_name], label=f"executed source {logical_name}")
        _exact_keys(record, {"path", "sha256"}, label=f"executed source {logical_name}")
        path_text = _absolute_path_text(record["path"], label=f"executed source {logical_name} path")
        if not Path(path_text).as_posix().endswith("/" + logical_name):
            raise ValueError(f"executed source path does not match logical name: {logical_name}")
        snapshot = _snapshot_file(Path(path_text), label=f"executed source {logical_name}")
        digest = _digest(record["sha256"], label=f"executed source {logical_name} sha256")
        if snapshot.sha256 != digest:
            raise ValueError(f"executed source bytes changed: {logical_name}")
        inode = (snapshot.identity[0], snapshot.identity[1])
        if snapshot.path in seen_paths or inode in seen_inodes:
            raise ValueError("executed-source bundle contains a duplicate source file")
        seen_paths.add(snapshot.path)
        seen_inodes.add(inode)
        normalized_files[logical_name] = {"path": str(snapshot.path), "sha256": digest}
        dependencies.append(snapshot)
    expected_bundle = _digest(bundle["bundle_sha256"], label="executed-source bundle_sha256")
    if canonical_hash(normalized_files) != expected_bundle:
        raise ValueError("executed-source bundle identity mismatch")
    return {
        "algorithm": "sha256",
        "files": normalized_files,
        "bundle_sha256": expected_bundle,
    }, dependencies


def _validate_overlapping_source_identity(
    implementation: object,
    executed_bundle: object,
) -> None:
    """Require stage/run claims for every overlapping v3 source to be identical."""

    authoritative = _mapping(implementation, label="run implementation")
    authoritative_files = _mapping(
        authoritative["files_sha256"], label="run implementation files"
    )
    executed = _mapping(executed_bundle, label="stage executed-source bundle")
    executed_files = _mapping(executed["files"], label="stage executed-source files")
    required = {"calibration.py", "provenance.py", "trainer.py"}
    if not required.issubset(authoritative_files):
        raise ValueError("run implementation is missing required v3 calibration sources")
    root = _safe_directory(
        Path(provenance_module.AUTHORITATIVE_SOURCE_ROOT),
        label="authoritative v3 source root",
    )
    prefix = "shadowforge_pipeline/soft_voxel_v3/"
    overlap: set[str] = set()
    for relative, raw_digest in authoritative_files.items():
        logical_name = prefix + relative
        if logical_name not in executed_files:
            continue
        overlap.add(relative)
        record = _mapping(
            executed_files[logical_name], label=f"overlapping source {logical_name}"
        )
        authoritative_path = (root / Path(*PurePosixPath(relative).parts)).resolve(
            strict=True
        )
        if (
            record["path"] != str(authoritative_path)
            or record["sha256"] != raw_digest
        ):
            raise ValueError(
                f"stage/run source overlap disagrees for {logical_name}"
            )
    if not required.issubset(overlap):
        raise ValueError(
            "stage executed-source bundle is missing required run implementation overlap"
        )


def _validate_interpreter(value: object) -> dict[str, object]:
    interpreter = _mapping(value, label="interpreter identity")
    _exact_keys(interpreter, {"path", "sha256", "python"}, label="interpreter identity")
    path = _absolute_path_text(interpreter["path"], label="interpreter path")
    digest = _digest(interpreter["sha256"], label="interpreter sha256")
    python = _nonempty_string(interpreter["python"], label="interpreter python")
    return {"path": path, "sha256": digest, "python": python}


def _validate_config(
    value: object,
    *,
    development_payload: Mapping[str, object],
    development_snapshot: _FileSnapshot,
    spec_record: Mapping[str, object],
) -> tuple[dict[str, object], list[_FileSnapshot]]:
    config = _mapping(value, label="run config")
    _exact_keys(
        config,
        {
            "schema",
            "device",
            "steps",
            "learning_rate",
            "resolution",
            "export_resolution",
            "profile",
            "lambda_dic",
            "lambda_dts",
            "seed",
            "shadow_inputs_xyz",
            "suite_manifest",
            "specification",
            "weight_config",
            "calibration_probe",
            "protocol",
            "density_thresholds",
            "final_step_export_only",
            "soft_only",
            "implementation",
            "interpreter",
            "determinism",
        },
        label="run config",
    )
    frozen = {
        "schema": _CONFIG_SCHEMA,
        "steps": 2000,
        "learning_rate": 0.007,
        "resolution": 128,
        "export_resolution": 128,
        "profile": "l12_bal",
        "lambda_dic": 0.0,
        "lambda_dts": 0.0,
        "seed": 17,
        "weight_config": None,
        "calibration_probe": True,
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
        "density_thresholds": [0.04, 0.05, 0.06],
        "final_step_export_only": True,
        "soft_only": True,
    }
    for key, expected in frozen.items():
        _exact_value(config[key], expected, label=f"run config {key}")
    _nonempty_string(config["device"], label="run device")
    suite = _mapping(config["suite_manifest"], label="run suite_manifest")
    _exact_keys(suite, {"path", "sha256"}, label="run suite_manifest")
    _exact_value(suite["path"], str(development_snapshot.path), label="run suite path")
    _exact_value(suite["sha256"], development_snapshot.sha256, label="run suite hash")
    specification = _mapping(config["specification"], label="run specification")
    _exact_keys(specification, {"path", "sha256"}, label="run specification")
    _exact_value(specification, spec_record, label="run specification identity")
    inputs = _list(config["shadow_inputs_xyz"], label="run shadow_inputs_xyz")
    development_tasks = _list(development_payload["tasks"], label="development tasks")
    arcane = _mapping(development_tasks[0], label="arcane_casters suite task")
    registered_inputs = _list(arcane["inputs"], label="arcane_casters inputs")
    if len(inputs) != 3:
        raise ValueError("run must bind exactly three x/y/z inputs")
    for axis, raw_input, registered in zip(_AXES, inputs, registered_inputs):
        item = _mapping(raw_input, label=f"run {axis} input")
        _exact_keys(item, {"path", "sha256"}, label=f"run {axis} input")
        registered_item = _mapping(registered, label=f"registered {axis} input")
        _exact_value(item["path"], registered_item["resolved_path"], label=f"run {axis} path")
        _exact_value(item["sha256"], registered_item["sha256"], label=f"run {axis} hash")
    implementation, source_dependencies = _validate_implementation(
        config["implementation"]
    )
    interpreter = _validate_interpreter(config["interpreter"])
    interpreter_snapshot = _snapshot_file(Path(interpreter["path"]), label="locked interpreter")
    if interpreter_snapshot.sha256 != interpreter["sha256"]:
        raise ValueError("locked interpreter bytes changed")
    determinism = _mapping(config["determinism"], label="run determinism")
    _exact_keys(
        determinism,
        {"fixed_seed", "mode", "bitwise_reproducible", "reason", "environment"},
        label="run determinism",
    )
    _exact_value(determinism["fixed_seed"], True, label="run determinism seed")
    _exact_value(determinism["mode"], "best_effort", label="run determinism mode")
    _exact_value(determinism["bitwise_reproducible"], False, label="run determinism bit")
    _nonempty_string(determinism["reason"], label="run determinism reason")
    environment = _mapping(determinism["environment"], label="run determinism environment")
    if not all(type(key) is str and type(item) is str for key, item in environment.items()):
        raise ValueError("run determinism environment must be an exact string mapping")
    normalized = dict(config)
    normalized["implementation"] = implementation
    normalized["interpreter"] = interpreter
    return normalized, [interpreter_snapshot, *source_dependencies]


def _validate_run_manifest(
    snapshot: _FileSnapshot,
    *,
    development_payload: Mapping[str, object],
    development_snapshot: _FileSnapshot,
    spec_record: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], list[_FileSnapshot]]:
    payload = _decode_json(snapshot.data, label="run manifest")
    _reject_quality_fields(
        payload,
        label="run manifest and config",
        ignored_subtree_paths=frozenset({_RUN_IMPLEMENTATION_DIGEST_PATH}),
    )
    _exact_keys(payload, {"schema", "config_hash", "config", "profile", "protocol", "scope"}, label="run manifest")
    _exact_value(payload["schema"], _RUN_SCHEMA, label="run schema")
    config_hash = _digest(payload["config_hash"], label="run config_hash")
    config, dependencies = _validate_config(
        payload["config"],
        development_payload=development_payload,
        development_snapshot=development_snapshot,
        spec_record=spec_record,
    )
    if canonical_hash(config) != config_hash:
        raise ValueError("run config hash does not match exact config")
    _exact_value(payload["profile"], "l12_bal", label="run profile")
    _exact_value(
        payload["protocol"],
        {
            "name": "sadekar_tfs_fixed_six_view",
            "camera_azimuth_deg": [0.0, 90.0, 0.0, 180.0, 270.0, 0.0],
            "camera_elevation_deg": [0.0, 0.0, 90.0, 0.0, 0.0, 270.0],
            "projection_binary_rule": "protocol.binary_mask",
            "density_thresholds": [0.04, 0.05, 0.06],
        },
        label="run protocol",
    )
    _exact_value(
        payload["scope"],
        {"soft_only": True, "final_step_export_only": True, "downstream_geometry": False},
        label="run scope",
    )
    return payload, config, dependencies


def _validate_artifact_integrity_manifest(
    snapshot: _FileSnapshot,
    *,
    config_hash: str,
    run_manifest_snapshot: _FileSnapshot,
    audit_snapshot: _FileSnapshot,
    measurements: Sequence[CalibrationMeasurement],
) -> dict[str, object]:
    """Validate Task 8 integrity names without resolving outcome artifacts."""

    payload = _decode_json(snapshot.data, label="artifact manifest")
    _exact_keys(
        payload,
        {"schema", "config_hash", "artifacts_sha256"},
        label="artifact manifest",
    )
    _exact_value(payload["schema"], _ARTIFACT_SCHEMA, label="artifact schema")
    _exact_value(payload["config_hash"], config_hash, label="artifact config identity")
    artifacts = _mapping(payload["artifacts_sha256"], label="artifact hashes")
    missing = _ARTIFACT_INTEGRITY_ONLY_NAMES - set(artifacts)
    if missing:
        raise ValueError(
            "artifact manifest must contain required metrics.json and "
            "resource_metrics.json integrity entries"
        )
    checked: dict[str, str] = {}
    for raw_name, raw_digest in artifacts.items():
        name = _nonempty_string(raw_name, label="artifact relative name")
        logical = PurePosixPath(name)
        if (
            "\\" in name
            or logical.is_absolute()
            or logical.as_posix() != name
            or any(part in ("", ".", "..") for part in logical.parts)
        ):
            raise ValueError(f"artifact name is not a canonical relative path: {name}")
        digest = _digest(raw_digest, label=f"artifact hash {name}")
        if name not in _ARTIFACT_INTEGRITY_ONLY_NAMES:
            _reject_quality_fields({name: digest}, label="artifact integrity map")
        checked[name] = digest
    for measurement in measurements:
        if checked.get(measurement.checkpoint.name) != measurement.checkpoint_sha256:
            raise ValueError(
                "calibration checkpoint is not reachable from artifact manifest"
            )
    for name, expected in (
        ("run_manifest.json", run_manifest_snapshot.sha256),
        ("gradient_audit.json", audit_snapshot.sha256),
    ):
        if checked.get(name) != expected:
            raise ValueError(f"artifact manifest does not bind current {name}")
    return payload


def _stage_semantics(payload: Mapping[str, object]) -> dict[str, object]:
    return {field: payload[field] for field in _STAGE_SCIENTIFIC_FIELDS}


def _validate_stage(
    snapshot: _FileSnapshot,
) -> tuple[dict[str, object], list[_FileSnapshot]]:
    payload = _decode_json(snapshot.data, label="calibration stage manifest")
    _reject_quality_fields(payload, label="calibration stage manifest")
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError("calibration stage manifest is not finite canonical JSON")
    _exact_keys(
        payload,
        {
            "schema",
            "status",
            "calibration_run",
            "checkpoints",
            "master_manifest",
            "development_suite",
            "specification",
            "inputs_xyz",
            "executed_source_bundle",
            "publication",
            "semantic_sha256",
        },
        label="calibration stage manifest",
    )
    _exact_value(payload["schema"], _STAGE_SCHEMA, label="calibration stage schema")
    _exact_value(payload["status"], _FROZEN_STATUS, label="calibration stage status")
    if canonical_hash(_stage_semantics(payload)) != _digest(
        payload["semantic_sha256"], label="calibration stage semantic_sha256"
    ):
        raise ValueError("calibration stage semantic identity mismatch")
    run = _mapping(payload["calibration_run"], label="calibration stage run")
    _exact_keys(
        run,
        {
            "path",
            "task",
            "profile",
            "resolution",
            "export_resolution",
            "steps",
            "seed",
            "soft_only",
            "calibration_probe",
            "config_hash",
            "run_manifest_sha256",
            "gradient_audit_sha256",
            "artifact_manifest_sha256",
            "status_sha256",
        },
        label="calibration stage run",
    )
    expected_run = {
        "task": "arcane_casters",
        "profile": "l12_bal",
        "resolution": 128,
        "export_resolution": 128,
        "steps": 2000,
        "seed": 17,
        "soft_only": True,
        "calibration_probe": True,
    }
    _absolute_path_text(run["path"], label="calibration run path")
    for key, expected in expected_run.items():
        _exact_value(run[key], expected, label=f"calibration stage run {key}")
    for key in (
        "config_hash",
        "run_manifest_sha256",
        "gradient_audit_sha256",
        "artifact_manifest_sha256",
        "status_sha256",
    ):
        _digest(run[key], label=f"calibration stage run {key}")
    checkpoints = _list(payload["checkpoints"], label="calibration stage checkpoints")
    if len(checkpoints) != 3:
        raise ValueError("calibration stage must register exactly three checkpoints")
    for update, raw_checkpoint in zip(_UPDATES, checkpoints):
        checkpoint = _mapping(raw_checkpoint, label="calibration stage checkpoint")
        _exact_keys(checkpoint, {"completed_updates", "path", "sha256"}, label="calibration stage checkpoint")
        if type(checkpoint["completed_updates"]) is not int or checkpoint["completed_updates"] != update:
            raise ValueError("calibration stage checkpoints have wrong order/update identity")
        expected_name = f"calibration_checkpoint_step_{update:08d}.pt"
        _exact_value(checkpoint["path"], expected_name, label="calibration stage checkpoint path")
        _digest(checkpoint["sha256"], label="calibration stage checkpoint sha256")
    for name in ("master_manifest", "development_suite"):
        record = _mapping(payload[name], label=f"stage {name}")
        _exact_keys(record, {"path", "sha256", "semantic_sha256"}, label=f"stage {name}")
        _absolute_path_text(record["path"], label=f"stage {name} path")
        _digest(record["sha256"], label=f"stage {name} sha256")
        _digest(record["semantic_sha256"], label=f"stage {name} semantic_sha256")
    specification = _mapping(payload["specification"], label="stage specification")
    _exact_keys(specification, {"path", "sha256"}, label="stage specification")
    _absolute_path_text(specification["path"], label="stage specification path")
    _digest(specification["sha256"], label="stage specification sha256")
    inputs = _list(payload["inputs_xyz"], label="stage inputs_xyz")
    if len(inputs) != 3:
        raise ValueError("calibration stage must bind exactly three x/y/z inputs")
    for axis, raw_input in zip(_AXES, inputs):
        item = _mapping(raw_input, label=f"stage {axis} input")
        _exact_keys(item, {"axis", "source_path", "resolved_path", "sha256"}, label=f"stage {axis} input")
        _exact_value(item["axis"], axis, label=f"stage {axis} axis")
        _absolute_path_text(item["source_path"], label=f"stage {axis} source_path")
        _absolute_path_text(item["resolved_path"], label=f"stage {axis} resolved_path")
        _digest(item["sha256"], label=f"stage {axis} sha256")
    executed_sources, source_dependencies = _validate_executed_source_bundle(
        payload["executed_source_bundle"]
    )
    _exact_value(
        payload["executed_source_bundle"],
        executed_sources,
        label="stage executed-source identity",
    )
    publication = _mapping(payload["publication"], label="stage publication")
    _exact_keys(
        publication,
        {"policy", "completed_status_sha256"},
        label="stage publication",
    )
    _exact_value(
        publication["policy"],
        "create_once_after_completed_calibrate128",
        label="stage publication policy",
    )
    _exact_value(
        publication["completed_status_sha256"],
        run["status_sha256"],
        label="stage post-completion STATUS identity",
    )
    return payload, source_dependencies


def _load_context(
    *,
    calibration_run: Path,
    stage_manifest: Path,
    master_manifest: Path,
    development_suite: Path,
) -> _CalibrationContext:
    run_root = _safe_directory(calibration_run, label="calibration run")
    stage_snapshot = _snapshot_file(stage_manifest, label="calibration stage manifest")
    stage, stage_source_dependencies = _validate_stage(stage_snapshot)
    stage_run = _mapping(stage["calibration_run"], label="stage calibration_run")
    if str(run_root) != stage_run["path"]:
        raise ValueError("calibration run argument does not match stage identity")

    master_snapshot = _snapshot_file(master_manifest, label="master manifest")
    development_snapshot = _snapshot_file(development_suite, label="development suite")
    stage_master = _mapping(stage["master_manifest"], label="stage master_manifest")
    stage_development = _mapping(stage["development_suite"], label="stage development_suite")
    if str(master_snapshot.path) != stage_master["path"] or master_snapshot.sha256 != stage_master["sha256"]:
        raise ValueError("master manifest argument/hash does not match stage identity")
    if str(development_snapshot.path) != stage_development["path"] or development_snapshot.sha256 != stage_development["sha256"]:
        raise ValueError("development suite argument/hash does not match stage identity")

    development_payload, development_dependencies = _validate_bound_suite(
        development_snapshot, role="development", validate_inputs=True
    )
    if development_payload["semantic_sha256"] != stage_development["semantic_sha256"]:
        raise ValueError("development suite semantic identity mismatch")
    master_payload, master_spec, master_dependencies = _validate_master(
        master_snapshot, development_snapshot=development_snapshot
    )
    if master_payload["semantic_sha256"] != stage_master["semantic_sha256"]:
        raise ValueError("master semantic identity mismatch")
    master_development = _mapping(
        _mapping(master_payload["suites"], label="master suites")["development"],
        label="master development suite",
    )
    if (
        master_development["sha256"] != development_snapshot.sha256
        or master_development["semantic_sha256"] != development_payload["semantic_sha256"]
    ):
        raise ValueError("master/development suite cross identity mismatch")

    stage_spec = _mapping(stage["specification"], label="stage specification")
    if dict(stage_spec) != master_spec:
        raise ValueError("stage/master specification identity mismatch")
    spec_snapshot = _snapshot_file(Path(str(stage_spec["path"])), label="revised specification")
    if spec_snapshot.sha256 != stage_spec["sha256"]:
        raise ValueError("revised specification bytes changed")

    development_tasks = _list(development_payload["tasks"], label="development tasks")
    arcane = _mapping(development_tasks[0], label="arcane_casters task")
    registered_inputs = _list(arcane["inputs"], label="arcane_casters inputs")
    _exact_value(stage["inputs_xyz"], registered_inputs, label="stage arcane_casters inputs")

    run_manifest_snapshot = _snapshot_file(
        run_root / "run_manifest.json", label="calibration run manifest", root=run_root
    )
    audit_snapshot = _snapshot_file(
        run_root / "gradient_audit.json", label="gradient audit", root=run_root
    )
    artifact_snapshot = _snapshot_file(
        run_root / "artifact_manifest.json", label="artifact manifest", root=run_root
    )
    status_snapshot = _snapshot_file(
        run_root / "STATUS.json", label="calibration run STATUS", root=run_root
    )
    for key, snapshot in (
        ("run_manifest_sha256", run_manifest_snapshot),
        ("gradient_audit_sha256", audit_snapshot),
        ("artifact_manifest_sha256", artifact_snapshot),
        ("status_sha256", status_snapshot),
    ):
        if stage_run[key] != snapshot.sha256:
            raise ValueError(f"calibration run {key} does not match stage hash")

    run_manifest, config, config_dependencies = _validate_run_manifest(
        run_manifest_snapshot,
        development_payload=development_payload,
        development_snapshot=development_snapshot,
        spec_record=stage_spec,
    )
    del run_manifest
    config_hash = canonical_hash(config)
    if config_hash != stage_run["config_hash"]:
        raise ValueError("calibration run config identity does not match stage")
    _validate_overlapping_source_identity(
        config["implementation"],
        stage["executed_source_bundle"],
    )

    measurements, checkpoint_snapshots, _ = _measurements_from_audit_snapshot(
        audit_snapshot, expected_config_hash=config_hash
    )
    stage_checkpoints = _list(stage["checkpoints"], label="stage checkpoints")
    for measurement, raw_stage_checkpoint in zip(measurements, stage_checkpoints):
        stage_checkpoint = _mapping(raw_stage_checkpoint, label="stage checkpoint")
        if (
            stage_checkpoint["completed_updates"] != measurement.completed_updates
            or stage_checkpoint["path"] != measurement.checkpoint.name
            or stage_checkpoint["sha256"] != measurement.checkpoint_sha256
        ):
            raise ValueError("stage and gradient audit checkpoint identities differ")

    _validate_artifact_integrity_manifest(
        artifact_snapshot,
        config_hash=config_hash,
        run_manifest_snapshot=run_manifest_snapshot,
        audit_snapshot=audit_snapshot,
        measurements=measurements,
    )

    status = _decode_json(status_snapshot.data, label="calibration run STATUS")
    _exact_keys(
        status,
        {"schema", "state", "config_hash", "completed_updates", "artifact_manifest_sha256"},
        label="calibration run STATUS",
    )
    _exact_value(status["schema"], _STATUS_SCHEMA, label="STATUS schema")
    _exact_value(status["state"], "completed", label="STATUS state")
    _exact_value(status["config_hash"], config_hash, label="STATUS config identity")
    _exact_value(status["completed_updates"], 2000, label="STATUS completed updates")
    _exact_value(
        status["artifact_manifest_sha256"],
        artifact_snapshot.sha256,
        label="STATUS artifact manifest identity",
    )

    implementation = _mapping(config["implementation"], label="config implementation")
    interpreter = _mapping(config["interpreter"], label="config interpreter")
    provenance: dict[str, object] = {
        "stage_manifest": {
            "path": str(stage_snapshot.path),
            "sha256": stage_snapshot.sha256,
            "semantic_sha256": stage["semantic_sha256"],
        },
        "calibration_run": dict(stage_run),
        "master_manifest": dict(stage_master),
        "development_suite": dict(stage_development),
        "specification": dict(stage_spec),
        "inputs_xyz": [dict(_mapping(item, label="stage input")) for item in stage["inputs_xyz"]],
        "implementation": dict(implementation),
        "interpreter": dict(interpreter),
        "executed_source_bundle": dict(
            _mapping(stage["executed_source_bundle"], label="stage executed sources")
        ),
    }
    _reject_quality_fields(
        provenance,
        label="calibration provenance",
        ignored_subtree_paths=frozenset({_PROVENANCE_IMPLEMENTATION_DIGEST_PATH}),
    )
    dependencies = _dependency_bundle(
        [
            stage_snapshot,
            master_snapshot,
            development_snapshot,
            spec_snapshot,
            run_manifest_snapshot,
            audit_snapshot,
            artifact_snapshot,
            status_snapshot,
            *development_dependencies,
            *master_dependencies,
            *config_dependencies,
            *checkpoint_snapshots,
            *stage_source_dependencies,
        ]
    )
    return _CalibrationContext(
        measurements=measurements,
        provenance=provenance,
        dependencies=dependencies,
    )


def _measurement_payload(measurement: CalibrationMeasurement) -> dict[str, object]:
    return {
        "completed_updates": measurement.completed_updates,
        "checkpoint": str(measurement.checkpoint),
        "checkpoint_sha256": measurement.checkpoint_sha256,
        "g0": measurement.g0,
        "g_dic": measurement.g_dic,
        "g_dts_004": measurement.g_dts_004,
        "g_dts_006": measurement.g_dts_006,
    }


def _weight_payload(
    measurements: Sequence[CalibrationMeasurement],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    checked, _ = _validate_measurement_identity(measurements, snapshot_checkpoints=True)
    weights = derive_calibrated_weights(checked)
    _reject_quality_fields(
        provenance,
        label="calibration provenance",
        ignored_subtree_paths=frozenset({_PROVENANCE_IMPLEMENTATION_DIGEST_PATH}),
    )
    payload: dict[str, object] = {
        "schema": _WEIGHT_SCHEMA,
        "status": _FROZEN_STATUS,
        "lambdas": weights,
        "calibration": dict(_CALIBRATION_DESCRIPTION),
        "measurements": [_measurement_payload(item) for item in checked],
        "provenance": json.loads(canonical_json_bytes(provenance)),
    }
    _reject_quality_fields(
        payload,
        label="proposed frozen weight payload",
        ignored_subtree_paths=frozenset({_WEIGHT_IMPLEMENTATION_DIGEST_PATH}),
    )
    payload["semantic_sha256"] = canonical_hash(
        {field: payload[field] for field in _WEIGHT_SCIENTIFIC_FIELDS}
    )
    return payload


def _safe_output_target(path: Path) -> Path:
    target = _absolute(path)
    _reject_traversal(target, label="weight output")
    _reject_symlink_components(target, label="weight output")
    if os.path.lexists(target):
        metadata = os.lstat(target)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("weight output target is a symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("weight output target is not a regular file")
    return target


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_weight_bytes(
    path: Path,
    payload: bytes,
    *,
    dependencies: _DependencySnapshot,
) -> Path:
    target = _safe_output_target(path)
    _verify_snapshot_unchanged(dependencies)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target.parent, label="weight output parent")
    created = False
    created_identity: tuple[int, int] | None = None
    accepted_identity: tuple[int, int] | None = None
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        if os.path.lexists(target):
            existing = _snapshot_file(target, label="existing immutable weight config")
            if existing.data != payload:
                raise FileExistsError("immutable weight config differs from existing bytes")
            accepted_identity = (existing.identity[0], existing.identity[1])
        else:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(temporary, flags, 0o644)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            temporary_metadata = os.lstat(temporary)
            if not stat.S_ISREG(temporary_metadata.st_mode):
                raise ValueError("temporary immutable weight config is not a regular file")
            temporary_identity = (
                temporary_metadata.st_dev,
                temporary_metadata.st_ino,
            )
            try:
                os.link(temporary, target, follow_symlinks=False)
                created = True
                created_identity = temporary_identity
                accepted_identity = temporary_identity
                _fsync_directory(target.parent)
            except FileExistsError:
                existing = _snapshot_file(target, label="concurrent immutable weight config")
                if existing.data != payload:
                    raise FileExistsError("immutable weight config differs from concurrent bytes")
                accepted_identity = (existing.identity[0], existing.identity[1])
        _verify_snapshot_unchanged(dependencies)
        published = _snapshot_file(target, label="published immutable weight config")
        if published.path != target:
            raise ValueError("published immutable weight target path changed")
        published_identity = (published.identity[0], published.identity[1])
        if accepted_identity is None or published_identity != accepted_identity:
            raise ValueError("published immutable weight target identity changed")
        if published.data != payload:
            raise ValueError("published immutable weight target bytes changed")
    except Exception:
        if created and created_identity is not None:
            try:
                current = os.lstat(target)
                if (
                    stat.S_ISREG(current.st_mode)
                    and (current.st_dev, current.st_ino) == created_identity
                ):
                    target.unlink()
                    _fsync_directory(target.parent)
            except FileNotFoundError:
                pass
        raise
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def _write_with_context(
    path: Path,
    *,
    measurements: Sequence[CalibrationMeasurement],
    provenance: Mapping[str, object],
    dependencies: _DependencySnapshot,
) -> Path:
    payload = _weight_payload(measurements, provenance)
    return _publish_weight_bytes(
        path,
        canonical_json_bytes(payload),
        dependencies=dependencies,
    )


def _context_from_provenance(provenance: Mapping[str, object]) -> _CalibrationContext:
    _exact_keys(
        provenance,
        {
            "stage_manifest",
            "calibration_run",
            "master_manifest",
            "development_suite",
            "specification",
            "inputs_xyz",
            "implementation",
            "interpreter",
            "executed_source_bundle",
        },
        label="calibration provenance",
    )
    stage = _mapping(provenance["stage_manifest"], label="provenance stage_manifest")
    run = _mapping(provenance["calibration_run"], label="provenance calibration_run")
    master = _mapping(provenance["master_manifest"], label="provenance master_manifest")
    development = _mapping(provenance["development_suite"], label="provenance development_suite")
    context = _load_context(
        calibration_run=Path(_absolute_path_text(run.get("path"), label="provenance run path")),
        stage_manifest=Path(_absolute_path_text(stage.get("path"), label="provenance stage path")),
        master_manifest=Path(_absolute_path_text(master.get("path"), label="provenance master path")),
        development_suite=Path(
            _absolute_path_text(development.get("path"), label="provenance development path")
        ),
    )
    _exact_value(dict(provenance), context.provenance, label="calibration provenance identity")
    return context


def write_frozen_weight_config(
    path: Path,
    *,
    measurements: Sequence[CalibrationMeasurement],
    provenance: Mapping[str, object],
) -> Path:
    """Create one immutable canonical weight file, or accept identical bytes."""

    if not isinstance(provenance, Mapping):
        raise ValueError("calibration provenance must be a mapping")
    _reject_quality_fields(
        provenance,
        label="calibration provenance",
        ignored_subtree_paths=frozenset({_PROVENANCE_IMPLEMENTATION_DIGEST_PATH}),
    )
    context = _context_from_provenance(provenance)
    checked, _ = _validate_measurement_identity(measurements, snapshot_checkpoints=True)
    if checked != context.measurements:
        raise ValueError("measurements do not match immutable calibration provenance")
    return _write_with_context(
        path,
        measurements=checked,
        provenance=context.provenance,
        dependencies=context.dependencies,
    )


def _measurements_from_weight(value: object) -> tuple[
    CalibrationMeasurement,
    CalibrationMeasurement,
    CalibrationMeasurement,
]:
    records = _list(value, label="frozen weight measurements")
    if len(records) != 3:
        raise ValueError("frozen weights must contain exactly three measurements")
    measurements: list[CalibrationMeasurement] = []
    for update, raw_record in zip(_UPDATES, records):
        record = _mapping(raw_record, label="frozen weight measurement")
        _exact_keys(
            record,
            {
                "completed_updates",
                "checkpoint",
                "checkpoint_sha256",
                "g0",
                "g_dic",
                "g_dts_004",
                "g_dts_006",
            },
            label="frozen weight measurement",
        )
        if type(record["completed_updates"]) is not int or record["completed_updates"] != update:
            raise ValueError("frozen weight measurement update/order mismatch")
        checkpoint = Path(_absolute_path_text(record["checkpoint"], label="weight checkpoint"))
        measurements.append(
            CalibrationMeasurement(
                completed_updates=update,
                checkpoint=checkpoint,
                checkpoint_sha256=_digest(record["checkpoint_sha256"], label="weight checkpoint sha256"),
                g0=_validate_float(record["g0"], label="weight g0"),
                g_dic=_validate_float(record["g_dic"], label="weight g_dic"),
                g_dts_004=_validate_float(record["g_dts_004"], label="weight g_dts_004"),
                g_dts_006=_validate_float(record["g_dts_006"], label="weight g_dts_006"),
            )
        )
    checked, _ = _validate_measurement_identity(measurements, snapshot_checkpoints=True)
    return checked


def _deep_read_only(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_read_only(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_read_only(item) for item in value)
    return value


def validate_frozen_weight_config(path: Path) -> Mapping[str, object]:
    """Strictly rederive a frozen configuration and rehash every dependency."""

    snapshot = _snapshot_file(path, label="frozen weight config")
    payload = _decode_json(snapshot.data, label="frozen weight config")
    _reject_quality_fields(
        payload,
        label="frozen weight config",
        ignored_subtree_paths=frozenset({_WEIGHT_IMPLEMENTATION_DIGEST_PATH}),
    )
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError("frozen weight config is not finite canonical JSON")
    _exact_keys(
        payload,
        {
            "schema",
            "status",
            "lambdas",
            "calibration",
            "measurements",
            "provenance",
            "semantic_sha256",
        },
        label="frozen weight config",
    )
    _exact_value(payload["schema"], _WEIGHT_SCHEMA, label="weight schema")
    _exact_value(payload["status"], _FROZEN_STATUS, label="weight status")
    _exact_value(payload["calibration"], _CALIBRATION_DESCRIPTION, label="weight calibration formula")
    semantic = _digest(payload["semantic_sha256"], label="weight semantic_sha256")
    scientific = {field: payload[field] for field in _WEIGHT_SCIENTIFIC_FIELDS}
    if canonical_hash(scientific) != semantic:
        raise ValueError("frozen weight semantic hash mismatch")
    measurements = _measurements_from_weight(payload["measurements"])
    context = _context_from_provenance(
        _mapping(payload["provenance"], label="weight provenance")
    )
    if measurements != context.measurements:
        raise ValueError("frozen measurements disagree with calibration provenance")
    expected = derive_calibrated_weights(measurements)
    lambdas = _mapping(payload["lambdas"], label="frozen lambdas")
    _exact_keys(lambdas, {"lambda_dic", "lambda_dts"}, label="frozen lambdas")
    for name in ("lambda_dic", "lambda_dts"):
        value = _validate_float(lambdas[name], label=f"frozen {name}")
        if value <= 0.0 or value != expected[name]:
            raise ValueError(f"stored {name} does not match recomputed calibration")
    _verify_snapshot_unchanged(context.dependencies)
    _verify_snapshot_unchanged(snapshot)
    return _deep_read_only(payload)


def build_calibration_parser() -> argparse.ArgumentParser:
    """Build the five-path, non-abbreviating calibration CLI."""

    parser = argparse.ArgumentParser(
        description="Freeze soft-loss-v3 coefficients from gradient-only calibration",
        allow_abbrev=False,
    )
    parser.add_argument("--calibration-run", type=Path, required=True)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--development-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def calibration_main(argv: Sequence[str] | None = None) -> int:
    """Validate one registered calibration run and publish frozen weights."""

    args = build_calibration_parser().parse_args(argv)
    context = _load_context(
        calibration_run=args.calibration_run,
        stage_manifest=args.stage_manifest,
        master_manifest=args.master_manifest,
        development_suite=args.development_suite,
    )
    _write_with_context(
        args.output,
        measurements=context.measurements,
        provenance=context.provenance,
        dependencies=context.dependencies,
    )
    return 0
