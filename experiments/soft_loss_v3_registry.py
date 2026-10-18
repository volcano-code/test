"""Immutable development/formal suite binding for the soft-loss-v3 study.

The source-manifest semantic hashes deliberately whitelist only fields that
select or define scientific inputs.  Quality/audit fields in the formal
benchmark are byte-bound but cannot select, filter, or reorder cases.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import uuid

from shadowforge_pipeline.soft_voxel_v3.provenance import (
    canonical_hash,
    canonical_json_bytes,
)


PROJECT_ROOT = Path("/home/lin/xvxinyao/P-CRAFT")
OBSOLETE_PROJECT_PREFIX = "/home/lin/xvxinyao/ShadowForge/"
SUITE_SCHEMA = "pcraft_soft_loss_v3_suite_v1"
MASTER_SCHEMA = "pcraft_soft_loss_v3_experiment_manifest_v1"
STATUS = "APPROVED_FROZEN"
AXES = ("x", "y", "z")
DEVELOPMENT_IDS = ("arcane_casters", "blade_heroes", "stage_dancers")
FORMAL_IDS = tuple(f"connected_{index:02d}" for index in range(1, 11))
PRIMARY_VARIANTS = ("l12_bal", "dic_only", "dts_only", "v3_full")
APPENDIX_VARIANTS = ("v3_zero_margin", "v3_no_l1", "v3_no_l2")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_STAGE_NAME = re.compile(r"\.pcraft-soft-loss-v3-stage-([0-9a-f]{32})\Z")
_CLEANUP_NAME = re.compile(r"\.pcraft-soft-loss-v3-cleanup-([0-9a-f]{32})\Z")
_RETIRED_NAME = re.compile(r"\.pcraft-soft-loss-v3-retired-([0-9a-f]{32})\Z")
_REMOVE_NAME = re.compile(r"\.pcraft-soft-loss-v3-remove-([0-9a-f]{32})\Z")
_STAGE_FILE = re.compile(r"payload_([0-9]{3})\.json\Z")
_WRITE_NAME = re.compile(
    r"\.pcraft-soft-loss-v3-write-"
    r"(stage_manifest\.json|payload_[0-9]{3}\.json)-([0-9a-f]{32})\Z"
)
_RETIREMENT_DIRECTORY = ".pcraft-soft-loss-v3-retirement"
_STAGE_SCHEMA = "pcraft_soft_loss_v3_publication_stage_v1"
_RENAME_NOREPLACE = 1

# Explicit scientific whitelists.  ``semantic_sha256`` is intentionally not
# selected because it is the digest of these fields.
_SUITE_SCIENTIFIC_FIELDS = (
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
)
_MASTER_SCIENTIFIC_FIELDS = (
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
_FORMAL_PROTOCOL_SOURCE_FIELDS = (
    "views",
    "declared_rendered_views",
    "axes",
    "camera",
    "tsf",
    "mr",
    "zdist",
    "resolution",
    "seeds",
    "evaluation",
)
_DEVELOPMENT_SOURCE_PROTOCOL = "fixed-3-view TFS, mr=1, 256^3"
_DEVELOPMENT_SOURCE_FOREGROUND = "white-on-black"
_FORMAL_SOURCE_PROTOCOL: dict[str, object] = {
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


@dataclass(frozen=True)
class TaskInputs:
    case_id: str
    targets: tuple[str, str, str]
    role: str


@dataclass(frozen=True)
class SuiteInputIdentity:
    """One ordered, byte-bound input from an approved suite."""

    axis: str
    source_path: str
    resolved_path: Path
    sha256: str


@dataclass(frozen=True)
class SuiteTaskIdentity:
    """One suite task with its exact x/y/z input identities."""

    case_id: str
    inputs: tuple[SuiteInputIdentity, SuiteInputIdentity, SuiteInputIdentity]


@dataclass(frozen=True)
class SuiteIdentity:
    """Stable suite bytes, semantics and ordered task/input identities."""

    path: Path
    sha256: str
    semantic_sha256: str
    role: str
    tasks: tuple[SuiteTaskIdentity, ...]


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    data: bytes
    sha256: str


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
        "variants": list(APPENDIX_VARIANTS),
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


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON constant must be finite, found {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _decode_json(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {label}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _absolute_argument(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _reject_traversal(path: Path, *, label: str) -> None:
    if ".." in path.parts:
        raise ValueError(f"{label} contains dot-dot path traversal: {path}")


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = _absolute_argument(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")


def _safe_directory(path: Path, *, label: str) -> Path:
    absolute = _absolute_argument(path)
    _reject_traversal(absolute, label=label)
    _reject_symlink_components(absolute, label=label)
    try:
        resolved = absolute.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing {label}: {absolute}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def _safe_regular_file(path: Path, *, label: str, root: Path | None = None) -> Path:
    absolute = _absolute_argument(path)
    if not absolute.is_absolute():  # pragma: no cover - guarded by helper
        raise ValueError(f"{label} must be absolute")
    _reject_traversal(absolute, label=label)
    _reject_symlink_components(absolute, label=label)
    try:
        resolved = absolute.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing {label}: {absolute}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    if root is not None:
        safe_root = _safe_directory(root, label=f"{label} root")
        if not resolved.is_relative_to(safe_root):
            raise ValueError(f"{label} escapes its canonical root: {resolved}")
    return resolved


def _snapshot_stat(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _snapshot_file(path: Path, *, label: str, root: Path | None = None) -> _FileSnapshot:
    """Read and hash one canonical regular file through one stable no-follow fd."""

    resolved = _safe_regular_file(path, label=label, root=root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} fd is not a regular file: {resolved}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _snapshot_stat(before) != _snapshot_stat(after):
            raise ValueError(f"{label} changed during stable snapshot: {resolved}")
        current = os.lstat(resolved)
        if (
            not stat.S_ISREG(current.st_mode)
            or current.st_dev != after.st_dev
            or current.st_ino != after.st_ino
        ):
            raise ValueError(f"{label} path changed during stable snapshot: {resolved}")
        _reject_symlink_components(resolved, label=label)
        if resolved.resolve(strict=True) != resolved:
            raise ValueError(f"{label} lost canonical containment during snapshot")
        data = b"".join(chunks)
        if len(data) != after.st_size:
            raise ValueError(f"{label} size changed during stable snapshot: {resolved}")
    finally:
        os.close(descriptor)
    return _FileSnapshot(
        path=resolved,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _verify_snapshot_unchanged(
    snapshot: _FileSnapshot,
    *,
    label: str,
    root: Path | None = None,
) -> None:
    current = _snapshot_file(snapshot.path, label=label, root=root)
    if current.data != snapshot.data or current.sha256 != snapshot.sha256:
        raise ValueError(f"{label} changed after its first stable snapshot")


def _source_path_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    source = Path(value)
    if not source.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    _reject_traversal(source, label=label)
    return value


def _resolve_input_path(value: object, *, root: Path, label: str) -> tuple[str, Path]:
    source_text = _source_path_text(value, label=label)
    if source_text.startswith(OBSOLETE_PROJECT_PREFIX):
        suffix = source_text[len(OBSOLETE_PROJECT_PREFIX) :]
        if not suffix or ".." in Path(suffix).parts or Path(suffix).is_absolute():
            raise ValueError(f"{label} has an ambiguous obsolete-prefix rewrite")
        candidate = Path(PROJECT_ROOT) / suffix
    else:
        candidate = Path(source_text)
    try:
        resolved = _safe_regular_file(candidate, label=label, root=root)
    except FileNotFoundError as exc:
        if not source_text.startswith(OBSOLETE_PROJECT_PREFIX):
            raise FileNotFoundError(
                f"missing {label}; no unregistered path rewrite is permitted: {source_text}"
            ) from exc
        raise
    return source_text, resolved


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _exact_keys(mapping: Mapping[str, object], expected: set[str], *, label: str) -> None:
    observed = set(mapping)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{label} schema has missing={missing} extra={extra}")


def _require_exact_value(value: object, expected: object, *, label: str) -> None:
    """Validate a frozen JSON value recursively without numeric coercion."""

    if type(value) is not type(expected):
        raise ValueError(
            f"{label} has wrong exact type: {type(value).__name__}, "
            f"expected {type(expected).__name__}"
        )
    if isinstance(expected, Mapping):
        observed = value
        _exact_keys(observed, set(expected), label=label)
        for key, expected_item in expected.items():
            _require_exact_value(observed[key], expected_item, label=f"{label}.{key}")
        return
    if isinstance(expected, list):
        observed = value
        if len(observed) != len(expected):
            raise ValueError(f"{label} has the wrong exact list length")
        for index, (observed_item, expected_item) in enumerate(zip(observed, expected)):
            _require_exact_value(
                observed_item,
                expected_item,
                label=f"{label}[{index}]",
            )
        return
    if value != expected:
        raise ValueError(f"{label} does not match the frozen exact value")


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _development_source_semantics(payload: Mapping[str, object]) -> dict[str, object]:
    _exact_keys(payload, {"protocol", "foreground", "tasks"}, label="development manifest")
    _require_exact_value(
        payload["protocol"],
        _DEVELOPMENT_SOURCE_PROTOCOL,
        label="development protocol",
    )
    _require_exact_value(
        payload["foreground"],
        _DEVELOPMENT_SOURCE_FOREGROUND,
        label="development foreground",
    )
    raw_tasks = _require_list(payload["tasks"], label="development tasks")
    if len(raw_tasks) != len(DEVELOPMENT_IDS):
        raise ValueError("development manifest must contain exactly three ordered tasks")
    tasks: list[dict[str, object]] = []
    for index, raw_task in enumerate(raw_tasks):
        task = _require_mapping(raw_task, label=f"development task {index}")
        _exact_keys(task, {"task", "views"}, label=f"development task {index}")
        expected_id = DEVELOPMENT_IDS[index]
        if task["task"] != expected_id:
            raise ValueError(f"expected ordered development task {expected_id}")
        views = _require_list(task["views"], label=f"{expected_id} views")
        if len(views) != 3:
            raise ValueError(f"{expected_id} must have exactly three x/y/z views")
        checked: list[str] = []
        for view_index, value in enumerate(views):
            source = _source_path_text(value, label=f"{expected_id} view {view_index}")
            if Path(source).name != f"view_{view_index}.png":
                raise ValueError(f"{expected_id} view order must be view_0, view_1, view_2")
            checked.append(source)
        tasks.append({"task": expected_id, "views": checked})
    return {
        "protocol": payload["protocol"],
        "foreground": payload["foreground"],
        "tasks": tasks,
    }


def _formal_source_semantics(payload: Mapping[str, object]) -> dict[str, object]:
    required = {"schema", "status", "protocol", "cases"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"formal manifest is missing fields: {sorted(missing)}")
    if payload["schema"] != "fixed3_mr1_connected10_v1":
        raise ValueError("unexpected formal benchmark schema")
    if payload["status"] != STATUS:
        raise ValueError(f"formal benchmark must have status {STATUS}")
    protocol = _require_mapping(payload["protocol"], label="formal protocol")
    _require_exact_value(protocol, _FORMAL_SOURCE_PROTOCOL, label="formal protocol")
    raw_cases = _require_list(payload["cases"], label="formal cases")
    if len(raw_cases) != len(FORMAL_IDS):
        raise ValueError("formal benchmark must contain exactly ten ordered cases")
    cases: list[dict[str, object]] = []
    for index, raw_case in enumerate(raw_cases):
        case = _require_mapping(raw_case, label=f"formal case {index}")
        if "case_id" not in case or "target_paths_xyz" not in case:
            raise ValueError(f"formal case {index} is missing case_id or target_paths_xyz")
        expected_id = FORMAL_IDS[index]
        if case["case_id"] != expected_id:
            raise ValueError(f"expected ordered formal case {expected_id}")
        targets = _require_list(case["target_paths_xyz"], label=f"{expected_id} target_paths_xyz")
        if len(targets) != 3:
            raise ValueError(f"{expected_id} target_paths_xyz must contain exactly three x/y/z paths")
        checked: list[str] = []
        for axis, value in zip(AXES, targets):
            source = _source_path_text(value, label=f"{expected_id} {axis} target")
            if not Path(source).name.endswith(f"_view_{axis}.png"):
                raise ValueError(f"{expected_id} target_paths_xyz is not in x/y/z order")
            checked.append(source)
        cases.append({"case_id": expected_id, "target_paths_xyz": checked})
    return {
        "schema": payload["schema"],
        "status": payload["status"],
        "protocol": {field: protocol[field] for field in _FORMAL_PROTOCOL_SOURCE_FIELDS},
        "cases": cases,
    }


def _read_source_manifest(
    path: Path,
    *,
    root: Path,
    label: str,
) -> tuple[_FileSnapshot, dict[str, object]]:
    snapshot = _snapshot_file(path, label=label, root=root)
    return snapshot, _decode_json(snapshot.data, label=label)


def _source_record(
    snapshot: _FileSnapshot,
    semantic_payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "source_path": str(snapshot.path),
        "resolved_path": str(snapshot.path),
        "sha256": snapshot.sha256,
        "semantic_sha256": canonical_hash(semantic_payload),
    }


def _scientific_suite(payload: Mapping[str, object]) -> dict[str, object]:
    fields = list(_SUITE_SCIENTIFIC_FIELDS)
    if payload.get("role") == "formal":
        fields.append("formal_run_roles")
    return {field: payload[field] for field in fields}


def _finish_suite(payload: dict[str, object]) -> dict[str, object]:
    payload["semantic_sha256"] = canonical_hash(_scientific_suite(payload))
    return payload


def _bind_task_inputs(
    cases: Sequence[tuple[str, Sequence[str]]],
    *,
    root: Path,
) -> tuple[list[dict[str, object]], list[_FileSnapshot]]:
    tasks: list[dict[str, object]] = []
    snapshots: list[_FileSnapshot] = []
    resolved_paths: set[str] = set()
    digests: set[str] = set()
    for case_id, raw_targets in cases:
        inputs: list[dict[str, object]] = []
        for axis, raw_target in zip(AXES, raw_targets):
            source, resolved = _resolve_input_path(
                raw_target,
                root=root,
                label=f"{case_id} {axis} input",
            )
            resolved_text = str(resolved)
            snapshot = _snapshot_file(
                resolved,
                label=f"{case_id} {axis} input",
                root=root,
            )
            digest = snapshot.sha256
            if resolved_text in resolved_paths:
                raise ValueError(f"duplicate resolved input path: {resolved_text}")
            if digest in digests:
                raise ValueError(f"duplicate input sha256: {digest}")
            resolved_paths.add(resolved_text)
            digests.add(digest)
            snapshots.append(snapshot)
            inputs.append(
                {
                    "axis": axis,
                    "source_path": source,
                    "resolved_path": resolved_text,
                    "sha256": digest,
                }
            )
        tasks.append({"case_id": case_id, "inputs": inputs})
    return tasks, snapshots


def _base_suite(
    *,
    role: str,
    source_manifest: Mapping[str, object],
    tasks: list[dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SUITE_SCHEMA,
        "status": STATUS,
        "role": role,
        "source_manifest": dict(source_manifest),
        "tasks": tasks,
        "task_count": len(tasks),
        "input_count": sum(len(task["inputs"]) for task in tasks),
        "input_order": list(AXES),
        "protocol": _protocol(),
        "primary_variants": list(PRIMARY_VARIANTS),
        "appendix_controls": _appendix_controls(),
    }
    if role == "formal":
        payload["formal_run_roles"] = _formal_run_roles()
    return _finish_suite(payload)


def build_binding_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind immutable soft-loss-v3 suites",
        allow_abbrev=False,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--formal-benchmark", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    return parser


def bind_development_suite(path: Path) -> Mapping[str, object]:
    root = _safe_directory(path, label="development root")
    manifest = root / "MANIFEST.json"
    alternatives = [
        candidate
        for candidate in root.iterdir()
        if candidate != manifest
        and candidate.suffix.lower() == ".json"
        and "manifest" in candidate.stem.lower()
    ]
    if alternatives:
        raise ValueError(f"ambiguous alternative development manifests: {alternatives}")
    source_snapshot, source = _read_source_manifest(
        manifest,
        root=root,
        label="development manifest",
    )
    semantics = _development_source_semantics(source)
    cases = [
        (task["task"], task["views"])
        for task in semantics["tasks"]
    ]
    tasks, input_snapshots = _bind_task_inputs(cases, root=root)
    payload = _base_suite(
        role="development",
        source_manifest=_source_record(source_snapshot, semantics),
        tasks=tasks,
    )
    for snapshot in [source_snapshot, *input_snapshots]:
        _verify_snapshot_unchanged(snapshot, label="development binding dependency", root=root)
    return payload


def bind_formal_suite(path: Path, benchmark: Path) -> Mapping[str, object]:
    root = _safe_directory(path, label="formal root")
    source_snapshot, source = _read_source_manifest(
        benchmark,
        root=root,
        label="formal benchmark manifest",
    )
    semantics = _formal_source_semantics(source)
    cases = [
        (case["case_id"], case["target_paths_xyz"])
        for case in semantics["cases"]
    ]
    tasks, input_snapshots = _bind_task_inputs(cases, root=root)
    payload = _base_suite(
        role="formal",
        source_manifest=_source_record(source_snapshot, semantics),
        tasks=tasks,
    )
    for snapshot in [source_snapshot, *input_snapshots]:
        _verify_snapshot_unchanged(snapshot, label="formal binding dependency", root=root)
    return payload


def _validate_source_record(
    value: object,
    *,
    role: str,
) -> tuple[Path, Path, _FileSnapshot]:
    record = _require_mapping(value, label="source_manifest")
    _exact_keys(
        record,
        {"source_path", "resolved_path", "sha256", "semantic_sha256"},
        label="source_manifest",
    )
    source_text = _source_path_text(record["source_path"], label="source manifest source_path")
    resolved_text = _source_path_text(
        record["resolved_path"], label="source manifest resolved_path"
    )
    if source_text != resolved_text:
        raise ValueError("source manifest source_path and resolved_path must match")
    manifest_path = Path(resolved_text)
    root = manifest_path.parent
    snapshot = _snapshot_file(
        manifest_path,
        label="bound source manifest",
        root=root,
    )
    resolved = snapshot.path
    expected_bytes = _digest(record["sha256"], label="source manifest sha256")
    if snapshot.sha256 != expected_bytes:
        raise ValueError("source manifest sha256 changed after binding")
    raw = _decode_json(snapshot.data, label="bound source manifest")
    semantics = (
        _development_source_semantics(raw)
        if role == "development"
        else _formal_source_semantics(raw)
    )
    expected_semantic = _digest(
        record["semantic_sha256"], label="source manifest semantic_sha256"
    )
    if canonical_hash(semantics) != expected_semantic:
        raise ValueError("source manifest semantic_sha256 changed after binding")
    return resolved, root, snapshot


def _validate_suite_payload(
    payload: Mapping[str, object],
    *,
    expected_role: str,
) -> list[TaskInputs]:
    if expected_role not in {"development", "formal"}:
        raise ValueError("expected_role must be development or formal")
    expected_keys = set(_SUITE_SCIENTIFIC_FIELDS) | {"semantic_sha256"}
    if expected_role == "formal":
        expected_keys.add("formal_run_roles")
    _exact_keys(payload, expected_keys, label=f"{expected_role} suite")
    if payload["schema"] != SUITE_SCHEMA or payload["status"] != STATUS:
        raise ValueError("suite schema/status does not match the frozen registry")
    if payload["role"] != expected_role:
        raise ValueError(f"suite role does not match expected role {expected_role}")
    expected_semantic = _digest(payload["semantic_sha256"], label="suite semantic_sha256")
    if canonical_hash(_scientific_suite(payload)) != expected_semantic:
        raise ValueError("suite semantic_sha256 does not match scientific fields")
    _, source_root, source_snapshot = _validate_source_record(
        payload["source_manifest"], role=expected_role
    )
    _require_exact_value(payload["input_order"], ["x", "y", "z"], label="suite input_order")
    _require_exact_value(payload["protocol"], _protocol(), label="suite protocol")
    _require_exact_value(
        payload["primary_variants"],
        list(PRIMARY_VARIANTS),
        label="suite primary variants",
    )
    _require_exact_value(
        payload["appendix_controls"],
        _appendix_controls(),
        label="suite appendix controls",
    )
    if expected_role == "formal":
        _require_exact_value(
            payload["formal_run_roles"],
            _formal_run_roles(),
            label="formal run roles",
        )

    raw_tasks = _require_list(payload["tasks"], label="suite tasks")
    expected_ids = DEVELOPMENT_IDS if expected_role == "development" else FORMAL_IDS
    if len(raw_tasks) != len(expected_ids):
        raise ValueError("suite has the wrong ordered task count")
    if type(payload["task_count"]) is not int or payload["task_count"] != len(expected_ids):
        raise ValueError("suite task_count is not exact")
    if type(payload["input_count"]) is not int or payload["input_count"] != 3 * len(expected_ids):
        raise ValueError("suite input_count is not exact")

    tasks: list[TaskInputs] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_digests: set[str] = set()
    input_snapshots: list[_FileSnapshot] = []
    for task_index, raw_task in enumerate(raw_tasks):
        task = _require_mapping(raw_task, label=f"suite task {task_index}")
        _exact_keys(task, {"case_id", "inputs"}, label=f"suite task {task_index}")
        expected_id = expected_ids[task_index]
        if task["case_id"] != expected_id or not isinstance(task["case_id"], str):
            raise ValueError(f"ordered suite task expected {expected_id}")
        if expected_id in seen_ids:
            raise ValueError(f"duplicate suite case ID: {expected_id}")
        seen_ids.add(expected_id)
        inputs = _require_list(task["inputs"], label=f"{expected_id} inputs")
        if len(inputs) != 3:
            raise ValueError(f"{expected_id} must have exactly three inputs")
        targets: list[str] = []
        for input_index, raw_input in enumerate(inputs):
            item = _require_mapping(raw_input, label=f"{expected_id} input {input_index}")
            _exact_keys(
                item,
                {"axis", "source_path", "resolved_path", "sha256"},
                label=f"{expected_id} input {input_index}",
            )
            axis = AXES[input_index]
            if item["axis"] != axis:
                raise ValueError(f"{expected_id} inputs are not in exact x/y/z order")
            source, resolved = _resolve_input_path(
                item["source_path"],
                root=source_root,
                label=f"{expected_id} {axis} input",
            )
            if source != item["source_path"] or str(resolved) != item["resolved_path"]:
                raise ValueError(f"{expected_id} {axis} resolved path does not match source")
            digest = _digest(item["sha256"], label=f"{expected_id} {axis} sha256")
            snapshot = _snapshot_file(
                resolved,
                label=f"{expected_id} {axis} bound input",
                root=source_root,
            )
            if snapshot.sha256 != digest:
                raise ValueError(f"{expected_id} {axis} input sha256 changed after binding")
            resolved_text = str(resolved)
            if resolved_text in seen_paths:
                raise ValueError(f"duplicate resolved input path: {resolved_text}")
            if digest in seen_digests:
                raise ValueError(f"duplicate input sha256: {digest}")
            seen_paths.add(resolved_text)
            seen_digests.add(digest)
            input_snapshots.append(snapshot)
            targets.append(resolved_text)
        tasks.append(
            TaskInputs(
                case_id=expected_id,
                targets=(targets[0], targets[1], targets[2]),
                role=expected_role,
            )
        )
    _verify_snapshot_unchanged(
        source_snapshot,
        label="bound source manifest",
        root=source_root,
    )
    for snapshot in input_snapshots:
        _verify_snapshot_unchanged(
            snapshot,
            label="bound input dependency",
            root=source_root,
        )
    return tasks


def _read_bound_suite(
    path: Path,
    expected_role: str,
) -> tuple[dict[str, object], list[TaskInputs], str]:
    absolute = _absolute_argument(path)
    snapshot = _snapshot_file(
        absolute,
        label=f"{expected_role} bound suite",
        root=absolute.parent,
    )
    payload = _decode_json(snapshot.data, label=f"{expected_role} bound suite")
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError(f"{expected_role} bound suite is not finite canonical JSON")
    tasks = _validate_suite_payload(payload, expected_role=expected_role)
    _verify_snapshot_unchanged(
        snapshot,
        label=f"{expected_role} bound suite",
        root=absolute.parent,
    )
    return payload, tasks, snapshot.sha256


def load_bound_suite(path: Path, expected_role: str) -> list[TaskInputs]:
    return _read_bound_suite(path, expected_role)[1]


def load_bound_suite_identity(path: Path, expected_role: str) -> SuiteIdentity:
    """Return suite and ordered input identities from one strict snapshot chain."""

    payload, validated_tasks, suite_sha256 = _read_bound_suite(path, expected_role)
    raw_tasks = payload["tasks"]
    assert isinstance(raw_tasks, list)
    identities: list[SuiteTaskIdentity] = []
    for raw_task, validated in zip(raw_tasks, validated_tasks):
        assert isinstance(raw_task, Mapping)
        raw_inputs = raw_task["inputs"]
        assert isinstance(raw_inputs, list)
        inputs: list[SuiteInputIdentity] = []
        for raw_input, resolved_text in zip(raw_inputs, validated.targets):
            assert isinstance(raw_input, Mapping)
            inputs.append(
                SuiteInputIdentity(
                    axis=str(raw_input["axis"]),
                    source_path=str(raw_input["source_path"]),
                    resolved_path=Path(resolved_text),
                    sha256=str(raw_input["sha256"]),
                )
            )
        identities.append(
            SuiteTaskIdentity(
                case_id=validated.case_id,
                inputs=(inputs[0], inputs[1], inputs[2]),
            )
        )
    absolute = _absolute_argument(path)
    return SuiteIdentity(
        path=absolute.resolve(strict=True),
        sha256=suite_sha256,
        semantic_sha256=str(payload["semantic_sha256"]),
        role=expected_role,
        tasks=tuple(identities),
    )


def _task_identity(task: TaskInputs, *, expected_role: str) -> tuple[str, list[str], list[str]]:
    if not isinstance(task, TaskInputs):
        raise ValueError("suite disjointness requires TaskInputs values")
    if task.role != expected_role:
        raise ValueError(f"task {task.case_id!r} has role {task.role!r}, expected {expected_role}")
    if not isinstance(task.case_id, str) or not task.case_id:
        raise ValueError("task case_id must be a non-empty string")
    if not isinstance(task.targets, tuple) or len(task.targets) != 3:
        raise ValueError(f"task {task.case_id} must have exactly three ordered targets")
    paths: list[str] = []
    digests: list[str] = []
    snapshots: list[_FileSnapshot] = []
    for axis, target in zip(AXES, task.targets):
        if not isinstance(target, str):
            raise ValueError(f"task {task.case_id} {axis} target must be a string")
        snapshot = _snapshot_file(
            Path(target),
            label=f"task {task.case_id} {axis} target",
        )
        paths.append(str(snapshot.path))
        digests.append(snapshot.sha256)
        snapshots.append(snapshot)
    if len(set(paths)) != len(paths):
        raise ValueError(f"duplicate path within task {task.case_id}")
    if len(set(digests)) != len(digests):
        raise ValueError(f"duplicate sha256 within task {task.case_id}")
    for snapshot in snapshots:
        _verify_snapshot_unchanged(snapshot, label=f"task {task.case_id} target")
    return task.case_id, paths, digests


def assert_suite_disjoint(
    development: Sequence[TaskInputs],
    formal: Sequence[TaskInputs],
) -> None:
    development_ids: set[str] = set()
    development_paths: set[str] = set()
    development_digests: set[str] = set()
    for task in development:
        case_id, paths, digests = _task_identity(task, expected_role="development")
        if case_id in development_ids:
            raise ValueError(f"duplicate development case ID: {case_id}")
        if development_paths.intersection(paths):
            raise ValueError(f"duplicate development input path in {case_id}")
        if development_digests.intersection(digests):
            raise ValueError(f"duplicate development input sha256 in {case_id}")
        development_ids.add(case_id)
        development_paths.update(paths)
        development_digests.update(digests)

    formal_ids: set[str] = set()
    formal_paths: set[str] = set()
    formal_digests: set[str] = set()
    for task in formal:
        case_id, paths, digests = _task_identity(task, expected_role="formal")
        if case_id in formal_ids:
            raise ValueError(f"duplicate formal case ID: {case_id}")
        if formal_paths.intersection(paths):
            raise ValueError(f"duplicate formal input path in {case_id}")
        if formal_digests.intersection(digests):
            raise ValueError(f"duplicate formal input sha256 in {case_id}")
        if case_id in development_ids:
            raise ValueError(f"case ID overlap between suites: {case_id}")
        if development_paths.intersection(paths):
            raise ValueError(f"input path overlap between suites at {case_id}")
        if development_digests.intersection(digests):
            raise ValueError(f"input sha256/bytes overlap between suites at {case_id}")
        formal_ids.add(case_id)
        formal_paths.update(paths)
        formal_digests.update(digests)


def _safe_output_target(path: Path) -> Path:
    absolute = _absolute_argument(path)
    _reject_traversal(absolute, label="output target")
    _reject_symlink_components(absolute, label="output target")
    if os.path.lexists(absolute):
        if absolute.is_symlink():
            raise ValueError(f"output target is a symlink: {absolute}")
        if not absolute.is_file():
            raise ValueError(f"output target is not a regular file: {absolute}")
    return absolute


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_bound_output_subdirectory(
    output_descriptor: int,
    relative: Path,
) -> int:
    """Open/create a no-follow directory chain relative to the locked root fd."""

    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("publication parent must be safely relative to output root")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    current = os.dup(output_descriptor)
    try:
        for part in relative.parts:
            if part in {"", "."}:
                continue
            try:
                following = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                os.mkdir(part, mode=0o755, dir_fd=current)
                os.fsync(current)
                following = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def _verify_bound_directory(
    descriptor: int,
    path: Path,
    *,
    label: str,
) -> None:
    """Require a bound directory fd to remain reachable at its exact lexical path."""

    opened = os.fstat(descriptor)
    try:
        current = os.lstat(path)
    except FileNotFoundError as error:
        raise ValueError(f"{label} was replaced during publication") from error
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or opened.st_dev != current.st_dev
        or opened.st_ino != current.st_ino
    ):
        raise ValueError(f"{label} was replaced during publication")


def _publication_records(
    files: Mapping[Path, bytes],
    *,
    output_root: Path,
) -> tuple[list[tuple[Path, bytes]], list[dict[str, object]]]:
    targets: list[tuple[Path, bytes]] = []
    records: list[dict[str, object]] = []
    seen_relative: set[str] = set()
    for index, (raw_target, payload) in enumerate(files.items()):
        if not isinstance(payload, bytes):
            raise TypeError("publication payloads must be exact bytes")
        target = _safe_output_target(raw_target)
        if not target.is_relative_to(output_root):
            raise ValueError(f"publication target escapes output root: {target}")
        relative = target.relative_to(output_root).as_posix()
        if not relative or relative in seen_relative or ".." in Path(relative).parts:
            raise ValueError(f"duplicate or unsafe publication target: {relative!r}")
        seen_relative.add(relative)
        stage_name = f"payload_{index:03d}.json"
        targets.append((target, payload))
        records.append(
            {
                "relative_path": relative,
                "stage_name": stage_name,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return targets, records


def _stage_manifest(token: str, records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema": _STAGE_SCHEMA,
        "token": token,
        "files": records,
    }


def _binding_from_stat(value: os.stat_result, data: bytes) -> dict[str, object]:
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "data": data,
    }


def _same_bound_inode(value: os.stat_result, binding: Mapping[str, object]) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and value.st_dev == binding.get("device")
        and value.st_ino == binding.get("inode")
    )


def _same_bound_directory(
    value: os.stat_result,
    binding: Mapping[str, object],
) -> bool:
    return (
        stat.S_ISDIR(value.st_mode)
        and value.st_dev == binding.get("device")
        and value.st_ino == binding.get("inode")
    )


def _directory_binding(value: os.stat_result) -> dict[str, object]:
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError("publication staging entry is not a directory")
    return {"device": value.st_dev, "inode": value.st_ino}


def _snapshot_regular_at(
    descriptor: int,
    name: str,
    *,
    label: str,
) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        opened = os.open(name, flags, dir_fd=descriptor)
    except OSError as error:
        raise ValueError(f"{label} is missing, replaced, or not a regular file") from error
    try:
        before = os.fstat(opened)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(opened, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(opened)
        try:
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError as error:
            raise ValueError(f"{label} changed during snapshot") from error
        if (
            not _same_bound_inode(after, _binding_from_stat(before, data))
            or not _same_bound_inode(current, _binding_from_stat(after, data))
            or before.st_size != after.st_size
            or after.st_size != len(data)
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise ValueError(f"{label} changed during snapshot")
        return _binding_from_stat(after, data)
    finally:
        os.close(opened)


def _rename_noreplace_at(
    source_descriptor: int,
    source_name: str,
    destination_descriptor: int,
    destination_name: str,
) -> None:
    """Atomically move one name without ever replacing the destination entry."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("renameat2(RENAME_NOREPLACE) is required for publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_descriptor,
        os.fsencode(source_name),
        destination_descriptor,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            os.strerror(error_number),
            destination_name,
        )


def _write_regular_at(
    descriptor: int,
    name: str,
    data: bytes,
    *,
    label: str,
) -> dict[str, object]:
    if name != "stage_manifest.json" and _STAGE_FILE.fullmatch(name) is None:
        raise ValueError(f"{label} has a noncanonical durable name")
    temporary_name = (
        f".pcraft-soft-loss-v3-write-{name}-{uuid.uuid4().hex}"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    opened = os.open(temporary_name, flags, 0o600, dir_fd=descriptor)
    try:
        view = memoryview(data)
        while view:
            written = os.write(opened, view)
            if written <= 0:
                raise OSError(f"short write while creating {label}")
            view = view[written:]
        os.fsync(opened)
        created = os.fstat(opened)
        if not stat.S_ISREG(created.st_mode) or created.st_size != len(data):
            raise ValueError(f"{label} was not created as exact regular bytes")
        binding = _binding_from_stat(created, data)
        _rename_noreplace_at(descriptor, temporary_name, descriptor, name)
        promoted = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not _same_bound_inode(promoted, binding):
            raise ValueError(f"{label} changed inode during atomic promotion")
        os.fsync(descriptor)
        return binding
    finally:
        os.close(opened)


def _conditional_unlink_bound_entry(
    descriptor: int,
    name: str,
    binding: Mapping[str, object],
    *,
    label: str,
) -> None:
    """Remove only the recorded inode, without unlinking the validated pathname."""

    try:
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError(f"{label} disappeared before conditional cleanup") from error
    if not _same_bound_inode(current, binding):
        raise ValueError(f"{label} inode changed before conditional cleanup")
    quarantine = f".pcraft-soft-loss-v3-remove-{uuid.uuid4().hex}"
    os.rename(
        name,
        quarantine,
        src_dir_fd=descriptor,
        dst_dir_fd=descriptor,
    )
    moved = os.stat(quarantine, dir_fd=descriptor, follow_symlinks=False)
    if not _same_bound_inode(moved, binding):
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            os.rename(
                quarantine,
                name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
        raise ValueError(f"{label} was replaced in the cleanup race window")
    current_moved = os.stat(quarantine, dir_fd=descriptor, follow_symlinks=False)
    if not _same_bound_inode(current_moved, binding):
        raise ValueError(f"{label} quarantine inode changed before cleanup")
    os.unlink(quarantine, dir_fd=descriptor)
    os.fsync(descriptor)


def _unlink_bound_entry_under_cooperative_lock(
    descriptor: int,
    name: str,
    binding: Mapping[str, object],
    *,
    label: str,
) -> None:
    """Rollback one exact final without creating crash-strandable parent debris."""

    observed = _snapshot_regular_at(descriptor, name, label=label)
    if (
        observed.get("device") != binding.get("device")
        or observed.get("inode") != binding.get("inode")
        or observed.get("data") != binding.get("data")
    ):
        raise ValueError(f"{label} changed before cooperative rollback")
    # The output-root flock excludes every in-contract namespace writer across
    # this final check and unlink.  A process that ignores it is out of scope.
    os.unlink(name, dir_fd=descriptor)
    os.fsync(descriptor)


def _verify_named_directory_binding(
    parent_descriptor: int,
    name: str,
    descriptor: int,
    binding: Mapping[str, object],
    *,
    label: str,
) -> None:
    opened = os.fstat(descriptor)
    try:
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError(f"{label} disappeared or was moved") from error
    if (
        not _same_bound_directory(opened, binding)
        or not _same_bound_directory(current, binding)
    ):
        raise ValueError(f"{label} inode changed or was replaced")


def _quarantine_bound_directory(
    *,
    parent_descriptor: int,
    source_name: str,
    cleanup_name: str,
    descriptor: int,
    binding: Mapping[str, object],
    label: str,
) -> None:
    """Atomically move only the recorded directory inode off its active name."""

    _verify_named_directory_binding(
        parent_descriptor,
        source_name,
        descriptor,
        binding,
        label=label,
    )
    try:
        os.stat(cleanup_name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise ValueError(f"{label} cleanup quarantine already exists")
    _rename_noreplace_at(
        parent_descriptor,
        source_name,
        parent_descriptor,
        cleanup_name,
    )
    try:
        moved = os.stat(
            cleanup_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as error:
        raise ValueError(f"{label} disappeared during quarantine") from error
    if not _same_bound_directory(moved, binding):
        try:
            os.stat(source_name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            try:
                os.rename(
                    cleanup_name,
                    source_name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
            except OSError:
                pass
        raise ValueError(f"{label} was replaced in the quarantine race window")
    try:
        os.stat(source_name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise ValueError(f"{label} active name was replaced during quarantine")
    _verify_named_directory_binding(
        parent_descriptor,
        cleanup_name,
        descriptor,
        binding,
        label=f"{label} quarantine",
    )
    os.fsync(parent_descriptor)


def _open_retirement_directory(
    output_descriptor: int,
    *,
    create: bool,
) -> int | None:
    """Bind the mode-0700 private namespace used only after active quarantine."""

    if create:
        try:
            os.mkdir(_RETIREMENT_DIRECTORY, mode=0o700, dir_fd=output_descriptor)
        except FileExistsError:
            pass
        else:
            os.fsync(output_descriptor)
    try:
        current = os.stat(
            _RETIREMENT_DIRECTORY,
            dir_fd=output_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        if not create:
            return None
        raise
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            _RETIREMENT_DIRECTORY,
            flags,
            dir_fd=output_descriptor,
        )
    except OSError as error:
        raise ValueError("publication retirement namespace is unsafe") from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(current.st_mode)
        or not _same_bound_directory(opened, _directory_binding(current))
        or stat.S_IMODE(opened.st_mode) != 0o700
        or opened.st_uid != os.geteuid()
    ):
        os.close(descriptor)
        raise ValueError("publication retirement namespace identity is invalid")
    return descriptor


def _retire_bound_directory(
    *,
    source_parent_descriptor: int,
    source_name: str,
    retirement_descriptor: int,
    retired_name: str,
    descriptor: int,
    binding: Mapping[str, object],
    label: str,
) -> None:
    """Move a bound active quarantine into the private deletion namespace."""

    _verify_named_directory_binding(
        source_parent_descriptor,
        source_name,
        descriptor,
        binding,
        label=label,
    )
    try:
        os.stat(retired_name, dir_fd=retirement_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise ValueError(f"{label} private retirement name already exists")
    _rename_noreplace_at(
        source_parent_descriptor,
        source_name,
        retirement_descriptor,
        retired_name,
    )
    try:
        moved = os.stat(
            retired_name,
            dir_fd=retirement_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as error:
        raise ValueError(f"{label} disappeared during private retirement") from error
    if not _same_bound_directory(moved, binding):
        # The moved entry is foreign.  It is preserved and never selected for deletion.
        raise ValueError(f"{label} was replaced in the retirement race window")
    try:
        os.stat(source_name, dir_fd=source_parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise ValueError(f"{label} active name was replaced during retirement")
    _verify_named_directory_binding(
        retirement_descriptor,
        retired_name,
        descriptor,
        binding,
        label=f"{label} private retirement",
    )
    os.fsync(source_parent_descriptor)
    os.fsync(retirement_descriptor)


def _remove_retired_directory(
    *,
    retirement_descriptor: int,
    retired_name: str,
    descriptor: int,
    directory_binding: Mapping[str, object],
    file_bindings: Mapping[str, object],
    label: str,
) -> None:
    """Delete a verified transaction only inside the bound private namespace."""

    if _RETIRED_NAME.fullmatch(retired_name) is None:
        raise ValueError(f"{label} has a noncanonical private retirement name")
    _verify_named_directory_binding(
        retirement_descriptor,
        retired_name,
        descriptor,
        directory_binding,
        label=label,
    )
    if set(os.listdir(descriptor)) != set(file_bindings):
        raise ValueError(f"{label} contents changed before cleanup")
    payload_names = sorted(
        name for name in file_bindings if _STAGE_FILE.fullmatch(name) is not None
    )
    write_names = sorted(
        name for name in file_bindings if _WRITE_NAME.fullmatch(name) is not None
    )
    remove_names = sorted(
        name for name in file_bindings if _REMOVE_NAME.fullmatch(name) is not None
    )
    other_names = (
        set(file_bindings)
        - set(payload_names)
        - set(write_names)
        - set(remove_names)
    )
    if not other_names.issubset({"stage_manifest.json"}):
        raise ValueError(f"{label} contains a non-owned cleanup entry")
    ordered_names = [*payload_names, *write_names, *remove_names]
    if "stage_manifest.json" in file_bindings:
        ordered_names.append("stage_manifest.json")
    for name in ordered_names:
        binding = _require_mapping(
            file_bindings[name], label=f"{label} file binding {name}"
        )
        _conditional_unlink_bound_entry(
            descriptor,
            name,
            binding,
            label=f"{label} file {name}",
        )
    _verify_named_directory_binding(
        retirement_descriptor,
        retired_name,
        descriptor,
        directory_binding,
        label=label,
    )
    os.rmdir(retired_name, dir_fd=retirement_descriptor)
    removed = os.fstat(descriptor)
    if not _same_bound_directory(removed, directory_binding) or removed.st_nlink != 0:
        raise ValueError(f"{label} removal did not unlink the recorded directory inode")
    try:
        os.stat(retired_name, dir_fd=retirement_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise ValueError(f"{label} retirement name was replaced during removal")
    os.fsync(retirement_descriptor)


def _remove_quarantined_directory(
    *,
    parent_descriptor: int,
    cleanup_name: str,
    descriptor: int,
    directory_binding: Mapping[str, object],
    file_bindings: Mapping[str, object],
    label: str,
) -> None:
    """Retire an active quarantine before any destructive cleanup operation."""

    cleanup_match = _CLEANUP_NAME.fullmatch(cleanup_name)
    if cleanup_match is None:
        raise ValueError(f"{label} has a noncanonical active cleanup name")
    retirement_descriptor = _open_retirement_directory(
        parent_descriptor,
        create=True,
    )
    assert retirement_descriptor is not None
    retired_name = f".pcraft-soft-loss-v3-retired-{cleanup_match.group(1)}"
    try:
        _retire_bound_directory(
            source_parent_descriptor=parent_descriptor,
            source_name=cleanup_name,
            retirement_descriptor=retirement_descriptor,
            retired_name=retired_name,
            descriptor=descriptor,
            binding=directory_binding,
            label=label,
        )
        _remove_retired_directory(
            retirement_descriptor=retirement_descriptor,
            retired_name=retired_name,
            descriptor=descriptor,
            directory_binding=directory_binding,
            file_bindings=file_bindings,
            label=f"{label} private retirement",
        )
    finally:
        os.close(retirement_descriptor)


def _write_stage(
    output_root: Path,
    targets: list[tuple[Path, bytes]],
    records: list[dict[str, object]],
    *,
    output_descriptor: int,
) -> tuple[Path, dict[str, object]]:
    token = uuid.uuid4().hex
    stage_name = f".pcraft-soft-loss-v3-stage-{token}"
    stage = output_root / stage_name
    os.mkdir(stage_name, mode=0o700, dir_fd=output_descriptor)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(stage_name, flags, dir_fd=output_descriptor)
    directory_binding = _directory_binding(os.fstat(descriptor))
    manifest = _stage_manifest(token, records)
    bindings: dict[str, dict[str, object]] = {}
    try:
        bindings["stage_manifest.json"] = _write_regular_at(
            descriptor,
            "stage_manifest.json",
            canonical_json_bytes(manifest),
            label="publication stage manifest",
        )
        for (_, payload), record in zip(targets, records):
            name = str(record["stage_name"])
            bindings[name] = _write_regular_at(
                descriptor, name, payload, label="publication staged payload"
            )
        os.fsync(descriptor)
        os.fsync(output_descriptor)
    except BaseException:
        try:
            cleanup_name = f".pcraft-soft-loss-v3-cleanup-{token}"
            _quarantine_bound_directory(
                parent_descriptor=output_descriptor,
                source_name=stage_name,
                cleanup_name=cleanup_name,
                descriptor=descriptor,
                binding=directory_binding,
                label="partial publication stage",
            )
            _remove_quarantined_directory(
                parent_descriptor=output_descriptor,
                cleanup_name=cleanup_name,
                descriptor=descriptor,
                directory_binding=directory_binding,
                file_bindings=bindings,
                label="partial publication stage quarantine",
            )
        except (OSError, ValueError):
            pass
        raise
    finally:
        os.close(descriptor)
    return stage, manifest


def _load_owned_stage(
    stage: Path,
    *,
    output_root: Path,
    output_descriptor: int,
) -> dict[str, object]:
    name_match = _STAGE_NAME.fullmatch(stage.name)
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(stage.name, flags, dir_fd=output_descriptor)
    except OSError as error:
        raise ValueError(f"missing publication staging entry: {stage}") from error
    try:
        initial = os.fstat(descriptor)
        if name_match is None or not stat.S_ISDIR(initial.st_mode):
            raise ValueError(f"ambiguous publication staging entry: {stage}")
        marker = _snapshot_regular_at(
            descriptor, "stage_manifest.json", label="publication stage manifest"
        )
        marker_data = marker["data"]
        assert isinstance(marker_data, bytes)
        payload = _decode_json(marker_data, label="publication stage manifest")
        if marker_data != canonical_json_bytes(payload):
            raise ValueError("publication stage manifest is not canonical JSON")
        _exact_keys(payload, {"schema", "token", "files"}, label="publication stage manifest")
        _require_exact_value(payload["schema"], _STAGE_SCHEMA, label="publication stage schema")
        _require_exact_value(payload["token"], name_match.group(1), label="publication stage token")
        raw_records = _require_list(payload["files"], label="publication stage files")
        if not raw_records:
            raise ValueError("publication stage has no files")
        records: list[dict[str, object]] = []
        bindings: dict[str, dict[str, object]] = {"stage_manifest.json": marker}
        expected_entries = {"stage_manifest.json"}
        seen_relative: set[str] = set()
        for index, raw_record in enumerate(raw_records):
            record = _require_mapping(raw_record, label=f"publication stage file {index}")
            _exact_keys(
                record,
                {"relative_path", "stage_name", "size", "sha256"},
                label=f"publication stage file {index}",
            )
            relative = record["relative_path"]
            stage_name = record["stage_name"]
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                raise ValueError("publication stage relative path is unsafe")
            if ".." in Path(relative).parts or relative in seen_relative:
                raise ValueError("publication stage relative path is duplicate or unsafe")
            if (
                not isinstance(stage_name, str)
                or _STAGE_FILE.fullmatch(stage_name) is None
                or stage_name != f"payload_{index:03d}.json"
            ):
                raise ValueError("publication stage payload name is noncanonical")
            if type(record["size"]) is not int or record["size"] < 0:
                raise ValueError("publication stage payload size has wrong exact type")
            digest = _digest(record["sha256"], label="publication stage payload sha256")
            staged = _snapshot_regular_at(
                descriptor, stage_name, label="publication staged payload"
            )
            staged_data = staged["data"]
            assert isinstance(staged_data, bytes)
            if len(staged_data) != record["size"] or hashlib.sha256(staged_data).hexdigest() != digest:
                raise ValueError("publication staged payload differs from its manifest")
            target = output_root / relative
            if not target.is_relative_to(output_root):
                raise ValueError("publication staged target escapes output root")
            seen_relative.add(relative)
            expected_entries.add(stage_name)
            bindings[stage_name] = staged
            records.append(dict(record))
        if set(os.listdir(descriptor)) != expected_entries:
            raise ValueError("publication stage contains ambiguous extra or missing entries")
        current = os.stat(stage.name, dir_fd=output_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != initial.st_dev
            or current.st_ino != initial.st_ino
        ):
            raise ValueError("publication staging directory changed during validation")
        return {
            "path": stage,
            "manifest": dict(payload),
            "records": records,
            "bindings": bindings,
            "device": current.st_dev,
            "inode": current.st_ino,
        }
    finally:
        os.close(descriptor)


def _load_recoverable_stage_fragment(
    name: str,
    *,
    output_root: Path,
    output_descriptor: int,
) -> dict[str, object]:
    """Bind a crash fragment without accepting changed complete stage bytes."""

    name_match = (
        _STAGE_NAME.fullmatch(name)
        or _CLEANUP_NAME.fullmatch(name)
        or _RETIRED_NAME.fullmatch(name)
    )
    if name_match is None:
        raise ValueError("publication recovery entry has a noncanonical name")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=output_descriptor)
    except OSError as error:
        raise ValueError("publication recovery entry is missing or replaced") from error
    try:
        initial = os.fstat(descriptor)
        directory_binding = _directory_binding(initial)
        entries = set(os.listdir(descriptor))
        if not entries:
            records: list[dict[str, object]] = []
            bindings: dict[str, dict[str, object]] = {}
        else:
            write_entries = sorted(
                entry for entry in entries if _WRITE_NAME.fullmatch(entry) is not None
            )
            if len(write_entries) > 1:
                raise ValueError("publication recovery has ambiguous partial writes")
            remove_entries = sorted(
                entry for entry in entries if _REMOVE_NAME.fullmatch(entry) is not None
            )
            if len(remove_entries) > 1:
                raise ValueError("publication recovery has ambiguous inner quarantines")
            if remove_entries and (
                _CLEANUP_NAME.fullmatch(name) is None
                and _RETIRED_NAME.fullmatch(name) is None
            ):
                raise ValueError("active publication stage contains a cleanup quarantine")
            if "stage_manifest.json" in entries:
                marker_name = "stage_manifest.json"
            elif len(remove_entries) == 1:
                marker_name = remove_entries[0]
            elif len(write_entries) == 1 and _WRITE_NAME.fullmatch(
                write_entries[0]
            ).group(1) == "stage_manifest.json":
                partial = _snapshot_regular_at(
                    descriptor,
                    write_entries[0],
                    label="publication recovery partial stage manifest write",
                )
                records = []
                bindings = {write_entries[0]: partial}
                marker_name = ""
            else:
                raise ValueError(
                    "non-empty publication recovery fragment has no owned manifest"
                )
            if marker_name:
                marker = _snapshot_regular_at(
                    descriptor,
                    marker_name,
                    label="publication recovery stage manifest",
                )
                marker_data = marker["data"]
                assert isinstance(marker_data, bytes)
                payload = _decode_json(
                    marker_data, label="publication recovery stage manifest"
                )
                if marker_data != canonical_json_bytes(payload):
                    raise ValueError("publication recovery stage manifest is not canonical JSON")
                _exact_keys(
                    payload,
                    {"schema", "token", "files"},
                    label="publication recovery stage manifest",
                )
                _require_exact_value(
                    payload["schema"],
                    _STAGE_SCHEMA,
                    label="publication recovery stage schema",
                )
                _require_exact_value(
                    payload["token"],
                    name_match.group(1),
                    label="publication recovery stage token",
                )
                raw_records = _require_list(
                    payload["files"], label="publication recovery stage files"
                )
                if not raw_records:
                    raise ValueError("publication recovery stage has no file records")
                records = []
                bindings = {marker_name: marker}
                expected_entries = {"stage_manifest.json"}
                record_by_name: dict[str, dict[str, object]] = {}
                seen_relative: set[str] = set()
                for index, raw_record in enumerate(raw_records):
                    record = _require_mapping(
                        raw_record, label=f"publication recovery stage file {index}"
                    )
                    _exact_keys(
                        record,
                        {"relative_path", "stage_name", "size", "sha256"},
                        label=f"publication recovery stage file {index}",
                    )
                    relative = record["relative_path"]
                    stage_name = record["stage_name"]
                    if (
                        not isinstance(relative, str)
                        or not relative
                        or Path(relative).is_absolute()
                        or ".." in Path(relative).parts
                        or relative in seen_relative
                    ):
                        raise ValueError("publication recovery relative path is unsafe")
                    if (
                        not isinstance(stage_name, str)
                        or stage_name != f"payload_{index:03d}.json"
                        or _STAGE_FILE.fullmatch(stage_name) is None
                    ):
                        raise ValueError("publication recovery payload name is noncanonical")
                    if type(record["size"]) is not int or record["size"] < 0:
                        raise ValueError(
                            "publication recovery payload size has wrong exact type"
                        )
                    digest = _digest(
                        record["sha256"], label="publication recovery payload sha256"
                    )
                    if not (output_root / relative).is_relative_to(output_root):
                        raise ValueError("publication recovery target escapes output root")
                    seen_relative.add(relative)
                    expected_entries.add(stage_name)
                    copied_record = dict(record)
                    records.append(copied_record)
                    record_by_name[stage_name] = copied_record
                    if stage_name in entries:
                        observed = _snapshot_regular_at(
                            descriptor,
                            stage_name,
                            label="publication recovery staged payload",
                        )
                        observed_data = observed["data"]
                        assert isinstance(observed_data, bytes)
                        if (
                            len(observed_data) != record["size"]
                            or hashlib.sha256(observed_data).hexdigest() != digest
                        ):
                            raise ValueError(
                                "publication recovery payload differs from its manifest"
                            )
                        bindings[stage_name] = observed
                if write_entries:
                    write_name = write_entries[0]
                    write_match = _WRITE_NAME.fullmatch(write_name)
                    assert write_match is not None
                    intended_name = write_match.group(1)
                    if (
                        intended_name == "stage_manifest.json"
                        or intended_name in entries
                        or intended_name not in record_by_name
                    ):
                        raise ValueError(
                            "publication recovery partial write has no unique missing target"
                        )
                    bindings[write_name] = _snapshot_regular_at(
                        descriptor,
                        write_name,
                        label="publication recovery partial payload write",
                    )
                inner_payloads = [
                    entry for entry in remove_entries if entry != marker_name
                ]
                if marker_name != "stage_manifest.json" and any(
                    _STAGE_FILE.fullmatch(entry) is not None for entry in entries
                ):
                    raise ValueError(
                        "quarantined recovery manifest coexists with staged payloads"
                    )
                for inner_name in inner_payloads:
                    observed = _snapshot_regular_at(
                        descriptor,
                        inner_name,
                        label="publication recovery inner quarantine",
                    )
                    observed_data = observed["data"]
                    assert isinstance(observed_data, bytes)
                    candidates = [
                        record
                        for stage_name, record in record_by_name.items()
                        if stage_name not in entries
                        and len(observed_data) == record["size"]
                        and hashlib.sha256(observed_data).hexdigest()
                        == record["sha256"]
                    ]
                    if len(candidates) != 1:
                        raise ValueError(
                            "publication recovery inner quarantine is not one exact missing payload"
                        )
                    bindings[inner_name] = observed
                if entries != set(bindings):
                    raise ValueError("publication recovery stage has foreign entries")
                if (
                    _STAGE_NAME.fullmatch(name) is not None
                    and not remove_entries
                    and not write_entries
                    and entries == expected_entries
                ):
                    raise ValueError(
                        "complete publication stage failed strict validation and is not recoverable debris"
                    )
            elif entries != set(bindings):
                raise ValueError("partial manifest recovery has foreign entries")
        current = os.stat(name, dir_fd=output_descriptor, follow_symlinks=False)
        if not _same_bound_directory(current, directory_binding):
            raise ValueError("publication recovery directory changed during validation")
        return {
            "path": output_root / name,
            "records": records,
            "bindings": bindings,
            "device": current.st_dev,
            "inode": current.st_ino,
        }
    finally:
        os.close(descriptor)


def _remove_stage_fragment(
    fragment: Mapping[str, object],
    *,
    output_root: Path,
    output_descriptor: int,
) -> None:
    path = Path(fragment["path"])
    source_name = path.name
    source_match = _STAGE_NAME.fullmatch(source_name)
    cleanup_match = _CLEANUP_NAME.fullmatch(source_name)
    retired_match = _RETIRED_NAME.fullmatch(source_name)
    if path.parent != output_root or (
        source_match is None and cleanup_match is None and retired_match is None
    ):
        raise ValueError("publication recovery fragment escaped its bound root")
    token = (source_match or cleanup_match or retired_match).group(1)  # type: ignore[union-attr]
    cleanup_name = f".pcraft-soft-loss-v3-cleanup-{token}"
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source_name, flags, dir_fd=output_descriptor)
    directory_binding = {
        "device": fragment["device"],
        "inode": fragment["inode"],
    }
    try:
        _verify_named_directory_binding(
            output_descriptor,
            source_name,
            descriptor,
            directory_binding,
            label="publication recovery fragment",
        )
        if source_match is not None:
            _quarantine_bound_directory(
                parent_descriptor=output_descriptor,
                source_name=source_name,
                cleanup_name=cleanup_name,
                descriptor=descriptor,
                binding=directory_binding,
                label="publication recovery stage",
            )
        bindings = _require_mapping(
            fragment["bindings"], label="publication recovery bindings"
        )
        if retired_match is not None:
            _remove_retired_directory(
                retirement_descriptor=output_descriptor,
                retired_name=source_name,
                descriptor=descriptor,
                directory_binding=directory_binding,
                file_bindings=bindings,
                label="publication recovery private retirement",
            )
        else:
            _remove_quarantined_directory(
                parent_descriptor=output_descriptor,
                cleanup_name=cleanup_name,
                descriptor=descriptor,
                directory_binding=directory_binding,
                file_bindings=bindings,
                label="publication recovery quarantine",
            )
    finally:
        os.close(descriptor)


def _load_owned_stages_and_recover_debris(
    output_root: Path,
    *,
    output_descriptor: int,
) -> list[dict[str, object]]:
    retirement_descriptor = _open_retirement_directory(
        output_descriptor,
        create=False,
    )
    if retirement_descriptor is not None:
        try:
            retirement_entries = sorted(os.listdir(retirement_descriptor))
            if any(
                _RETIRED_NAME.fullmatch(name) is None
                for name in retirement_entries
            ):
                raise ValueError(
                    "publication retirement namespace contains a foreign entry"
                )
            retirement_root = output_root / _RETIREMENT_DIRECTORY
            for name in retirement_entries:
                fragment = _load_recoverable_stage_fragment(
                    name,
                    output_root=retirement_root,
                    output_descriptor=retirement_descriptor,
                )
                _remove_stage_fragment(
                    fragment,
                    output_root=retirement_root,
                    output_descriptor=retirement_descriptor,
                )
        finally:
            os.close(retirement_descriptor)
    for name in sorted(os.listdir(output_descriptor)):
        if _CLEANUP_NAME.fullmatch(name) is None:
            continue
        fragment = _load_recoverable_stage_fragment(
            name,
            output_root=output_root,
            output_descriptor=output_descriptor,
        )
        _remove_stage_fragment(
            fragment,
            output_root=output_root,
            output_descriptor=output_descriptor,
        )
    owned: list[dict[str, object]] = []
    for name in sorted(os.listdir(output_descriptor)):
        if _STAGE_NAME.fullmatch(name) is None:
            continue
        try:
            stage = _load_owned_stage(
                output_root / name,
                output_root=output_root,
                output_descriptor=output_descriptor,
            )
        except ValueError:
            fragment = _load_recoverable_stage_fragment(
                name,
                output_root=output_root,
                output_descriptor=output_descriptor,
            )
            _remove_stage_fragment(
                fragment,
                output_root=output_root,
                output_descriptor=output_descriptor,
            )
        else:
            owned.append(stage)
    return owned


def _stage_matches(stage: Mapping[str, object], records: list[dict[str, object]]) -> bool:
    return stage["records"] == records


def _open_owned_stage_descriptor(
    stage: Mapping[str, object],
    *,
    output_root: Path,
    output_descriptor: int,
) -> int:
    path = Path(stage["path"])
    if path.parent != output_root or _STAGE_NAME.fullmatch(path.name) is None:
        raise ValueError("owned publication stage is outside its bound output root")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.name, flags, dir_fd=output_descriptor)
    except OSError as error:
        raise ValueError("owned publication stage path changed before cleanup") from error
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_dev != stage.get("device")
        or opened.st_ino != stage.get("inode")
    ):
        os.close(descriptor)
        raise ValueError("owned publication stage inode changed before cleanup")
    return descriptor


def _verify_owned_stage_binding(
    stage: Mapping[str, object],
    *,
    output_root: Path,
    output_descriptor: int,
    stage_descriptor: int,
) -> None:
    path = Path(stage["path"])
    if path.parent != output_root:
        raise ValueError("owned publication stage lost output-root containment")
    opened = os.fstat(stage_descriptor)
    try:
        current = os.stat(
            path.name,
            dir_fd=output_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as error:
        raise ValueError("owned publication stage was moved before cleanup") from error
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or opened.st_dev != stage.get("device")
        or opened.st_ino != stage.get("inode")
        or current.st_dev != opened.st_dev
        or current.st_ino != opened.st_ino
    ):
        raise ValueError("owned publication stage was replaced before cleanup")


def _verify_owned_stage_files(
    stage: Mapping[str, object],
    *,
    stage_descriptor: int,
) -> None:
    bindings = _require_mapping(stage.get("bindings"), label="owned stage bindings")
    expected_entries = set(bindings)
    if set(os.listdir(stage_descriptor)) != expected_entries:
        raise ValueError("owned publication stage contents changed")
    ordered_names = [
        *(str(record["stage_name"]) for record in stage["records"]),
        "stage_manifest.json",
    ]
    if set(ordered_names) != expected_entries:
        raise ValueError("owned publication stage binding set is malformed")
    for name in ordered_names:
        expected = _require_mapping(bindings[name], label=f"owned stage binding {name}")
        observed = _snapshot_regular_at(
            stage_descriptor,
            name,
            label=f"owned publication staged file {name}",
        )
        if (
            observed.get("device") != expected.get("device")
            or observed.get("inode") != expected.get("inode")
            or observed.get("data") != expected.get("data")
        ):
            raise ValueError(f"owned publication staged file {name} changed inode or bytes")


def _remove_owned_stage(
    stage: Mapping[str, object],
    *,
    output_root: Path,
    output_descriptor: int,
    stage_descriptor: int | None = None,
) -> None:
    path = Path(stage["path"])
    name_match = _STAGE_NAME.fullmatch(path.name)
    if name_match is None:
        raise ValueError("owned publication stage name is noncanonical")
    owned_descriptor = stage_descriptor is None
    descriptor = (
        _open_owned_stage_descriptor(
            stage,
            output_root=output_root,
            output_descriptor=output_descriptor,
        )
        if stage_descriptor is None
        else stage_descriptor
    )
    try:
        _verify_owned_stage_binding(
            stage,
            output_root=output_root,
            output_descriptor=output_descriptor,
            stage_descriptor=descriptor,
        )
        _verify_owned_stage_files(stage, stage_descriptor=descriptor)
        _verify_owned_stage_binding(
            stage,
            output_root=output_root,
            output_descriptor=output_descriptor,
            stage_descriptor=descriptor,
        )
        directory_binding = {
            "device": stage["device"],
            "inode": stage["inode"],
        }
        cleanup_name = f".pcraft-soft-loss-v3-cleanup-{name_match.group(1)}"
        _quarantine_bound_directory(
            parent_descriptor=output_descriptor,
            source_name=path.name,
            cleanup_name=cleanup_name,
            descriptor=descriptor,
            binding=directory_binding,
            label="owned publication stage",
        )
        _remove_quarantined_directory(
            parent_descriptor=output_descriptor,
            cleanup_name=cleanup_name,
            descriptor=descriptor,
            directory_binding=directory_binding,
            file_bindings=_require_mapping(
                stage["bindings"], label="owned stage bindings"
            ),
            label="owned publication stage quarantine",
        )
    finally:
        if owned_descriptor:
            os.close(descriptor)


def _locked_output_root(output_root: Path) -> tuple[Path, int]:
    output = _absolute_argument(output_root)
    _reject_traversal(output, label="output root")
    _reject_symlink_components(output, label="output root")
    output.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output, label="output root")
    if not output.is_dir():
        raise ValueError(f"output root is not a directory: {output}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(output, flags)
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode):
        os.close(descriptor)
        raise ValueError("output root fd is not a directory")
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    return output, descriptor


def _bind_existing_final(
    descriptor: int,
    name: str,
    expected: bytes,
    *,
    label: str,
) -> dict[str, object] | None:
    try:
        current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(current.st_mode):
        raise FileExistsError(f"{label} is not a regular immutable publication")
    snapshot = _snapshot_regular_at(descriptor, name, label=label)
    if snapshot["data"] != expected:
        raise FileExistsError(f"{label} differs; refusing immutable overwrite")
    return snapshot


def _verify_bound_final(
    descriptor: int,
    name: str,
    binding: Mapping[str, object],
    expected: bytes,
    *,
    label: str,
) -> None:
    observed = _snapshot_regular_at(descriptor, name, label=label)
    if (
        observed.get("device") != binding.get("device")
        or observed.get("inode") != binding.get("inode")
        or observed.get("data") != expected
    ):
        raise ValueError(f"{label} changed inode or bytes during publication")


def _verify_publication_boundaries(
    output: Path,
    output_descriptor: int,
    parents: Mapping[Path, int],
) -> None:
    _verify_bound_directory(output_descriptor, output, label="publication output root")
    for parent, descriptor in parents.items():
        _verify_bound_directory(descriptor, parent, label="publication target parent")


def _reject_reserved_target_parent_debris(
    parent_descriptors: Mapping[Path, int],
) -> None:
    """Fail closed on transaction temp names in any current target parent."""

    for parent, descriptor in parent_descriptors.items():
        reserved = sorted(
            name
            for name in os.listdir(descriptor)
            if _REMOVE_NAME.fullmatch(name) is not None
            or _WRITE_NAME.fullmatch(name) is not None
        )
        if reserved:
            raise ValueError(
                f"publication target parent {parent} contains reserved "
                f"transaction temporary entries: {reserved}"
            )


def _rollback_created_finals(
    created: Sequence[tuple[Path, Mapping[str, object]]],
    parent_descriptors: Mapping[Path, int],
) -> None:
    for target, binding in reversed(created):
        descriptor = parent_descriptors[target.parent]
        try:
            _unlink_bound_entry_under_cooperative_lock(
                descriptor,
                target.name,
                binding,
                label=f"transaction-created final {target.name}",
            )
        except (OSError, ValueError):
            # Missing or replacement content is not transaction-owned and must survive.
            continue


def _publish_bundle(
    files: Mapping[Path, bytes],
    *,
    output_root: Path,
    precommit: Callable[[], None] | None = None,
) -> None:
    if not files:
        raise ValueError("publication bundle cannot be empty")
    output, lock_descriptor = _locked_output_root(output_root)
    parent_descriptors: dict[Path, int] = {}
    stage_descriptor: int | None = None
    try:
        targets, records = _publication_records(files, output_root=output)
        for target, _ in targets:
            parent = target.parent
            if parent not in parent_descriptors:
                parent_descriptors[parent] = _open_bound_output_subdirectory(
                    lock_descriptor, parent.relative_to(output)
                )
        _verify_publication_boundaries(output, lock_descriptor, parent_descriptors)
        _reject_reserved_target_parent_debris(parent_descriptors)
        if precommit is not None:
            precommit()
        _verify_publication_boundaries(output, lock_descriptor, parent_descriptors)
        owned_stages = _load_owned_stages_and_recover_debris(
            output,
            output_descriptor=lock_descriptor,
        )
        matching = [stage for stage in owned_stages if _stage_matches(stage, records)]
        nonmatching = [stage for stage in owned_stages if not _stage_matches(stage, records)]
        final_bindings: dict[Path, dict[str, object]] = {}
        for target, expected in targets:
            binding = _bind_existing_final(
                parent_descriptors[target.parent],
                target.name,
                expected,
                label=f"immutable publication {target}",
            )
            if binding is not None:
                final_bindings[target] = binding
        exists = [path in final_bindings for path, _ in targets]

        if all(exists):
            if nonmatching:
                raise FileExistsError("nonidentical owned staging conflicts with accepted bundle")
            if precommit is not None:
                precommit()
            _verify_publication_boundaries(output, lock_descriptor, parent_descriptors)
            for target, expected in targets:
                _verify_bound_final(
                    parent_descriptors[target.parent],
                    target.name,
                    final_bindings[target],
                    expected,
                    label=f"immutable publication {target}",
                )
            for stage in matching:
                _remove_owned_stage(
                    stage,
                    output_root=output,
                    output_descriptor=lock_descriptor,
                )
            if precommit is not None:
                precommit()
            for target, expected in targets:
                _verify_bound_final(
                    parent_descriptors[target.parent],
                    target.name,
                    final_bindings[target],
                    expected,
                    label=f"terminal immutable publication {target}",
                )
            _verify_publication_boundaries(output, lock_descriptor, parent_descriptors)
            return

        if any(exists):
            if nonmatching or not matching:
                raise FileExistsError(
                    "ambiguous or nonidentical partial immutable publication set"
                )
            active_stage = matching[0]
            for duplicate in matching[1:]:
                _remove_owned_stage(
                    duplicate,
                    output_root=output,
                    output_descriptor=lock_descriptor,
                )
            preserve_stage_on_failure = True
        else:
            for stage in nonmatching:
                _remove_owned_stage(
                    stage,
                    output_root=output,
                    output_descriptor=lock_descriptor,
                )
            if matching:
                active_stage = matching[0]
                for duplicate in matching[1:]:
                    _remove_owned_stage(
                        duplicate,
                        output_root=output,
                        output_descriptor=lock_descriptor,
                    )
                preserve_stage_on_failure = True
            else:
                stage_path, _ = _write_stage(
                    output,
                    targets,
                    records,
                    output_descriptor=lock_descriptor,
                )
                active_stage = _load_owned_stage(
                    stage_path,
                    output_root=output,
                    output_descriptor=lock_descriptor,
                )
                preserve_stage_on_failure = False

        created: list[tuple[Path, Mapping[str, object]]] = []
        stage_descriptor = _open_owned_stage_descriptor(
            active_stage,
            output_root=output,
            output_descriptor=lock_descriptor,
        )
        _verify_owned_stage_binding(
            active_stage,
            output_root=output,
            output_descriptor=lock_descriptor,
            stage_descriptor=stage_descriptor,
        )
        try:
            if precommit is not None:
                precommit()
            _verify_publication_boundaries(output, lock_descriptor, parent_descriptors)
            _verify_owned_stage_binding(
                active_stage,
                output_root=output,
                output_descriptor=lock_descriptor,
                stage_descriptor=stage_descriptor,
            )
            _verify_owned_stage_files(active_stage, stage_descriptor=stage_descriptor)
            for target, expected in targets:
                if target in final_bindings:
                    _verify_bound_final(
                        parent_descriptors[target.parent],
                        target.name,
                        final_bindings[target],
                        expected,
                        label=f"preexisting partial publication {target}",
                    )
        except BaseException:
            if not preserve_stage_on_failure:
                try:
                    _remove_owned_stage(
                        active_stage,
                        output_root=output,
                        output_descriptor=lock_descriptor,
                        stage_descriptor=stage_descriptor,
                    )
                except (OSError, ValueError):
                    pass
            raise
        try:
            _verify_owned_stage_binding(
                active_stage,
                output_root=output,
                output_descriptor=lock_descriptor,
                stage_descriptor=stage_descriptor,
            )
            stage_bindings = _require_mapping(
                active_stage["bindings"], label="active stage bindings"
            )
            remaining_missing = sum(
                target not in final_bindings for target, _ in targets
            )
            for (target, expected), record in zip(targets, records):
                if target in final_bindings:
                    continue
                parent_descriptor = parent_descriptors[target.parent]
                _verify_bound_directory(
                    parent_descriptor, target.parent, label="publication target parent"
                )
                os.link(
                    str(record["stage_name"]),
                    target.name,
                    src_dir_fd=stage_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                stage_binding = _require_mapping(
                    stage_bindings[str(record["stage_name"])],
                    label="active staged payload binding",
                )
                created.append((target, stage_binding))
                os.fsync(parent_descriptor)
                linked = _snapshot_regular_at(
                    parent_descriptor,
                    target.name,
                    label=f"transaction-created final {target}",
                )
                if (
                    linked.get("device") != stage_binding.get("device")
                    or linked.get("inode") != stage_binding.get("inode")
                    or linked.get("data") != expected
                ):
                    raise ValueError("staged payload changed before immutable final link")
                final_bindings[target] = linked
                remaining_missing -= 1
                _verify_bound_directory(
                    parent_descriptor, target.parent, label="publication target parent"
                )
                if remaining_missing:
                    if precommit is not None:
                        precommit()
                    _verify_publication_boundaries(
                        output, lock_descriptor, parent_descriptors
                    )
                    _verify_owned_stage_binding(
                        active_stage,
                        output_root=output,
                        output_descriptor=lock_descriptor,
                        stage_descriptor=stage_descriptor,
                    )
                    _verify_owned_stage_files(
                        active_stage, stage_descriptor=stage_descriptor
                    )
                    for bound_target, bound_expected in targets:
                        if bound_target not in final_bindings:
                            continue
                        _verify_bound_final(
                            parent_descriptors[bound_target.parent],
                            bound_target.name,
                            final_bindings[bound_target],
                            bound_expected,
                            label=f"between-link immutable publication {bound_target}",
                        )
        except BaseException:
            _rollback_created_finals(created, parent_descriptors)
            if not preserve_stage_on_failure:
                try:
                    _remove_owned_stage(
                        active_stage,
                        output_root=output,
                        output_descriptor=lock_descriptor,
                        stage_descriptor=stage_descriptor,
                    )
                except (OSError, ValueError):
                    pass
            raise
        try:
            if precommit is not None:
                precommit()
            _verify_publication_boundaries(output, lock_descriptor, parent_descriptors)
            _verify_owned_stage_binding(
                active_stage,
                output_root=output,
                output_descriptor=lock_descriptor,
                stage_descriptor=stage_descriptor,
            )
            _verify_owned_stage_files(active_stage, stage_descriptor=stage_descriptor)
            for target, expected in targets:
                _verify_bound_final(
                    parent_descriptors[target.parent],
                    target.name,
                    final_bindings[target],
                    expected,
                    label=f"terminal immutable publication {target}",
                )
        except BaseException:
            _rollback_created_finals(created, parent_descriptors)
            if not preserve_stage_on_failure:
                try:
                    _remove_owned_stage(
                        active_stage,
                        output_root=output,
                        output_descriptor=lock_descriptor,
                        stage_descriptor=stage_descriptor,
                    )
                except (OSError, ValueError):
                    pass
            raise
        try:
            _remove_owned_stage(
                active_stage,
                output_root=output,
                output_descriptor=lock_descriptor,
                stage_descriptor=stage_descriptor,
            )
            if precommit is not None:
                precommit()
            for target, expected in targets:
                _verify_bound_final(
                    parent_descriptors[target.parent],
                    target.name,
                    final_bindings[target],
                    expected,
                    label=f"post-cleanup immutable publication {target}",
                )
            _verify_publication_boundaries(output, lock_descriptor, parent_descriptors)
        except BaseException:
            _rollback_created_finals(created, parent_descriptors)
            raise
    finally:
        if stage_descriptor is not None:
            os.close(stage_descriptor)
        for descriptor in parent_descriptors.values():
            os.close(descriptor)
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


def _master_payload(
    output_root: Path,
    development_path: Path,
    formal_path: Path,
    spec_path: Path,
) -> dict[str, object]:
    output = _absolute_argument(output_root)
    expected_development = output / "manifests" / "development_suite.json"
    expected_formal = output / "manifests" / "formal_suite.json"
    if _absolute_argument(development_path) != expected_development:
        raise ValueError("development suite path is not the registered relative path")
    if _absolute_argument(formal_path) != expected_formal:
        raise ValueError("formal suite path is not the registered relative path")
    development_payload, development, development_sha256 = _read_bound_suite(
        expected_development, "development"
    )
    formal_payload, formal, formal_sha256 = _read_bound_suite(expected_formal, "formal")
    assert_suite_disjoint(development, formal)
    spec_absolute = _absolute_argument(spec_path)
    spec_snapshot = _snapshot_file(
        spec_absolute,
        label="revised-design spec",
        root=spec_absolute.parent,
    )
    spec = spec_snapshot.path
    determinism = {
        "policy": "fixed_seed_single_run_best_effort_cuda",
        "bitwise_reproducible": False,
        "reason": _protocol()["disclosure"],
    }
    payload: dict[str, object] = {
        "schema": MASTER_SCHEMA,
        "status": STATUS,
        "suites": {
            "development": {
                "path": "manifests/development_suite.json",
                "sha256": development_sha256,
                "semantic_sha256": development_payload["semantic_sha256"],
                "task_count": len(development),
                "input_count": 3 * len(development),
            },
            "formal": {
                "path": "manifests/formal_suite.json",
                "sha256": formal_sha256,
                "semantic_sha256": formal_payload["semantic_sha256"],
                "task_count": len(formal),
                "input_count": 3 * len(formal),
            },
        },
        "design_spec": {"path": str(spec), "sha256": spec_snapshot.sha256},
        "primary_matrix": _primary_matrix(),
        "appendix_controls": _appendix_controls(),
        "protocol": _protocol(),
        "determinism": determinism,
        "formal_run_roles": _formal_run_roles(),
    }
    payload["semantic_sha256"] = canonical_hash(
        {field: payload[field] for field in _MASTER_SCIENTIFIC_FIELDS}
    )
    _verify_snapshot_unchanged(
        spec_snapshot,
        label="revised-design spec",
        root=spec_absolute.parent,
    )
    return payload


def write_master_manifest(
    output_root: Path,
    development_path: Path,
    formal_path: Path,
    spec_path: Path,
) -> Path:
    output = _absolute_argument(output_root)
    destination = output / "experiment_manifest.json"
    payload = _master_payload(output, development_path, formal_path, spec_path)
    expected = canonical_json_bytes(payload)

    def verify_dependencies() -> None:
        current = canonical_json_bytes(
            _master_payload(output, development_path, formal_path, spec_path)
        )
        if current != expected:
            raise ValueError("master dependencies changed before publication")

    _publish_bundle(
        {destination: expected},
        output_root=output,
        precommit=verify_dependencies,
    )
    return destination


def binding_main(argv: Sequence[str] | None = None) -> int:
    args = build_binding_parser().parse_args(argv)
    development = bind_development_suite(args.development_root)
    formal_benchmark = _absolute_argument(args.formal_benchmark)
    formal = bind_formal_suite(formal_benchmark.parent, formal_benchmark)

    with tempfile.TemporaryDirectory(prefix="pcraft-soft-loss-v3-binding-") as raw_stage:
        stage = Path(raw_stage)
        development_path = stage / "manifests" / "development_suite.json"
        formal_path = stage / "manifests" / "formal_suite.json"
        development_path.parent.mkdir(parents=True)
        development_path.write_bytes(canonical_json_bytes(development))
        formal_path.write_bytes(canonical_json_bytes(formal))
        master = _master_payload(stage, development_path, formal_path, args.spec)
        master_bytes = canonical_json_bytes(master)

        def verify_dependencies() -> None:
            current = canonical_json_bytes(
                _master_payload(stage, development_path, formal_path, args.spec)
            )
            if current != master_bytes:
                raise ValueError("binding dependencies changed before publication")

        output = _absolute_argument(args.output_root)
        _publish_bundle(
            {
                output / "manifests" / "development_suite.json": development_path.read_bytes(),
                output / "manifests" / "formal_suite.json": formal_path.read_bytes(),
                output / "experiment_manifest.json": master_bytes,
            },
            output_root=output,
            precommit=verify_dependencies,
        )
    return 0
