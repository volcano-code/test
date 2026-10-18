"""Immutable v4 suite projection, candidate registry, and bundle publication."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
import uuid

from experiments.soft_loss_v3_registry import load_bound_suite_identity
from shadowforge_pipeline.soft_voxel_v4.config import bootstrap_v4_root
from shadowforge_pipeline.soft_voxel_v4.losses import registered_profiles
from shadowforge_pipeline.soft_voxel_v4.provenance import (
    canonical_hash,
    canonical_json_bytes,
    revalidate_snapshot,
    snapshot_regular_file,
)


SUITE_SCHEMA = "pcraft_soft_loss_v4_suite_v1"
REGISTRY_SCHEMA = "pcraft_soft_loss_v4_candidate_registry_v1"
DEVELOPMENT_IDS = ("arcane_casters", "blade_heroes", "stage_dancers")
FORMAL_IDS = tuple(f"connected_{index:02d}" for index in range(1, 11))
AXES = ("x", "y", "z")
CANDIDATES = (
    ("r04_r04", 0, 0.04, 0.04),
    ("r08_r08", 1, 0.08, 0.08),
    ("r12_r12", 2, 0.12, 0.12),
    ("r12_r08", 3, 0.12, 0.08),
    ("r08_r12", 4, 0.08, 0.12),
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_STAGE_RE = re.compile(r"\.pcraft-soft-loss-v4-stage-([0-9a-f]{32})\Z")
_STAGE_SCHEMA = "pcraft_soft_loss_v4_publication_stage_v1"
_RENAME_NOREPLACE = 1
_STAGE_NAME = re.compile(r"\.pcraft-soft-loss-v4-stage-([0-9a-f]{32})\Z")
_CLEANUP_NAME = re.compile(r"\.pcraft-soft-loss-v4-cleanup-([0-9a-f]{32})\Z")
_RETIRED_NAME = re.compile(r"\.pcraft-soft-loss-v4-retired-([0-9a-f]{32})\Z")
_REMOVE_NAME = re.compile(
    r"\.pcraft-soft-loss-v4-remove-([0-9a-f]{2,256})-([0-9a-f]{32})\Z"
)
_STAGE_FILE = re.compile(r"payload_([0-9]{3})\.json\Z")
_WRITE_NAME = re.compile(
    r"\.pcraft-soft-loss-v4-write-"
    r"(stage_manifest\.json|payload_[0-9]{3}\.json)-([0-9a-f]{32})\Z"
)
_RETIREMENT_DIRECTORY = ".pcraft-soft-loss-v4-retirement"
_PROTOCOL = {
    "views": 3,
    "rendered_views": 6,
    "axes": ["x", "y", "z"],
    "mirror_enabled": True,
    "camera": "fixed orthographic TFS",
    "image_size": 256,
    "zdist": 1.7,
    "volume_extent": 1.7,
    "samples_per_ray": 150,
    "seed": 17,
    "density_thresholds": [0.04, 0.05, 0.06],
}


@dataclass(frozen=True)
class V4BoundInput:
    axis: str
    source_path: str
    resolved_path: str
    sha256: str


@dataclass(frozen=True)
class V4TaskInputs:
    case_id: str
    inputs: tuple[V4BoundInput, V4BoundInput, V4BoundInput]
    role: str


@dataclass(frozen=True)
class V4SuiteIdentity:
    path: str
    sha256: str
    semantic_sha256: str
    role: str
    tasks: tuple[V4TaskInputs, ...]


def _decode(raw: bytes, *, label: str) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant in {label}: {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}") from error
    if type(payload) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return payload


# Compatibility helpers used by the v4-namespaced copy of the final approved
# v3 publication state machine below.  Scientific suite parsing does not use
# these looser Mapping helpers.
def _decode_json(raw: bytes, *, label: str) -> dict[str, object]:
    return _decode(raw, label=label)


def _absolute_argument(path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _reject_traversal(path: Path, *, label: str) -> None:
    if ".." in path.parts:
        raise ValueError(f"{label} contains dot-dot path traversal: {path}")


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = _absolute_argument(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _exact_keys(
    mapping: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if set(mapping) != expected:
        raise ValueError(f"{label} has the wrong exact schema")


def _require_exact_value(value: object, expected: object, *, label: str) -> None:
    if type(value) is not type(expected):
        raise ValueError(f"{label} has the wrong exact type")
    if isinstance(expected, Mapping):
        observed = value
        _exact_keys(observed, set(expected), label=label)
        for key, expected_item in expected.items():
            _require_exact_value(observed[key], expected_item, label=f"{label}.{key}")
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise ValueError(f"{label} has the wrong exact list length")
        for index, (observed_item, expected_item) in enumerate(zip(value, expected)):
            _require_exact_value(
                observed_item, expected_item, label=f"{label}[{index}]"
            )
    elif value != expected:
        raise ValueError(f"{label} does not match the frozen exact value")


def _remove_intended_name(entry: str) -> str:
    """Decode the durable original name carried by a cleanup quarantine."""

    match = _REMOVE_NAME.fullmatch(entry)
    if match is None or len(match.group(1)) % 2:
        raise ValueError("publication cleanup quarantine name is malformed")
    try:
        intended = bytes.fromhex(match.group(1)).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("publication cleanup quarantine name is malformed") from error
    if (
        intended != "stage_manifest.json"
        and _STAGE_FILE.fullmatch(intended) is None
        and _WRITE_NAME.fullmatch(intended) is None
    ):
        raise ValueError("publication cleanup quarantine has an unknown original name")
    return intended


def _exact_mapping(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} has the wrong exact schema")
    return value


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_absolute(path: Path, *, label: str, must_exist: bool = True) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate != Path(os.path.abspath(candidate)):
        raise ValueError(f"{label} must be an absolute canonical path")
    if ".." in candidate.parts:
        raise ValueError(f"{label} contains traversal")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if must_exist:
                raise ValueError(f"missing {label}: {candidate}")
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink component")
    if must_exist and candidate.resolve(strict=True) != candidate:
        raise ValueError(f"{label} is not canonical")
    return candidate


def _semantic(payload: Mapping[str, object], *, label: str) -> str:
    if "semantic_sha256" not in payload:
        raise ValueError(f"{label} lacks semantic_sha256")
    recorded = _digest(payload["semantic_sha256"], label=f"{label} semantic_sha256")
    scientific = dict(payload)
    del scientific["semantic_sha256"]
    if canonical_hash(scientific) != recorded:
        raise ValueError(f"{label} semantic hash mismatch")
    return recorded


def _deep_read_only(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_read_only(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_read_only(item) for item in value)
    return value


def _suite_payload(source_path: Path, expected_role: str) -> tuple[dict[str, object], object]:
    if expected_role not in ("development", "formal"):
        raise ValueError("expected_role must be development or formal")
    source = load_bound_suite_identity(Path(source_path), expected_role)
    source_snapshot = snapshot_regular_file(Path(source.path))
    if source_snapshot.sha256 != source.sha256:
        raise ValueError("v3 source suite bytes changed during projection")
    expected_ids = DEVELOPMENT_IDS if expected_role == "development" else FORMAL_IDS
    if tuple(task.case_id for task in source.tasks) != expected_ids:
        raise ValueError("v3 source suite task order/count drifted")
    tasks: list[dict[str, object]] = []
    input_snapshots = []
    for task in source.tasks:
        if len(task.inputs) != 3 or tuple(item.axis for item in task.inputs) != AXES:
            raise ValueError("v3 source suite x/y/z order drifted")
        inputs = []
        for item in task.inputs:
            resolved = Path(item.resolved_path)
            snapshot = snapshot_regular_file(resolved)
            if snapshot.sha256 != item.sha256 or snapshot.path != str(resolved):
                raise ValueError("v3 source suite input identity drifted")
            input_snapshots.append(snapshot)
            inputs.append(
                {
                    "axis": item.axis,
                    "source_path": str(resolved),
                    "resolved_path": str(resolved),
                    "sha256": item.sha256,
                }
            )
        tasks.append({"case_id": task.case_id, "inputs": inputs})
    payload: dict[str, object] = {
        "schema": SUITE_SCHEMA,
        "status": "REGISTERED",
        "role": expected_role,
        "source_suite": {
            "path": source.path.as_posix() if isinstance(source.path, Path) else str(source.path),
            "sha256": source.sha256,
            "semantic_sha256": source.semantic_sha256,
        },
        "tasks": tasks,
        "task_count": len(tasks),
        "input_count": 3 * len(tasks),
        "input_order": list(AXES),
        "protocol": dict(_PROTOCOL),
        "profiles": list(registered_profiles()),
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    for snapshot in input_snapshots:
        revalidate_snapshot(snapshot)
    revalidate_snapshot(source_snapshot)
    return payload, source_snapshot


def _expected_suite_path(output_path: Path, role: str) -> tuple[Path, Path]:
    target = _canonical_absolute(Path(output_path), label="suite output", must_exist=False)
    root = target.parent.parent
    expected = root / "manifests" / f"{role}_suite.json"
    if target != expected:
        raise ValueError("suite output is not the canonical v4 role child")
    return root, target


def bind_v4_suite(source_path: Path, output_path: Path, expected_role: str) -> Path:
    """Project one strict v3 suite into the exact v4 schema, create-once."""

    payload, source_snapshot = _suite_payload(source_path, expected_role)
    root, target = _expected_suite_path(output_path, expected_role)
    marker = bootstrap_v4_root(root)

    def precommit() -> None:
        revalidate_snapshot(source_snapshot)
        current, _ = _suite_payload(source_path, expected_role)
        if canonical_json_bytes(current) != canonical_json_bytes(payload):
            raise ValueError("v3 suite dependencies changed before v4 publication")
        if not marker.is_file():
            raise ValueError("v4 root marker disappeared")

    _publish_bundle(
        {target: canonical_json_bytes(payload)}, output_root=root, precommit=precommit
    )
    return target


def _validate_suite_payload(
    payload: dict[str, object], *, snapshot: object, expected_role: str
) -> V4SuiteIdentity:
    _exact_mapping(
        payload,
        {
            "schema", "status", "role", "source_suite", "tasks", "task_count",
            "input_count", "input_order", "protocol", "profiles", "semantic_sha256",
        },
        label="v4 suite",
    )
    if payload["schema"] != SUITE_SCHEMA or payload["status"] != "REGISTERED":
        raise ValueError("v4 suite schema/status mismatch")
    if payload["role"] != expected_role:
        raise ValueError("v4 suite role mismatch")
    semantic = _semantic(payload, label="v4 suite")
    _require_exact_value(
        payload["input_order"], list(AXES), label="v4 suite input order"
    )
    _require_exact_value(payload["protocol"], _PROTOCOL, label="v4 suite protocol")
    _require_exact_value(
        payload["profiles"],
        list(registered_profiles()),
        label="v4 suite profile registry",
    )
    source_record = _exact_mapping(
        payload["source_suite"], {"path", "sha256", "semantic_sha256"}, label="source suite"
    )
    source_path = _canonical_absolute(Path(source_record["path"]), label="source suite")
    source_identity = load_bound_suite_identity(source_path, expected_role)
    if (
        str(source_identity.path) != str(source_path)
        or source_identity.sha256 != _digest(source_record["sha256"], label="source suite sha256")
        or source_identity.semantic_sha256
        != _digest(source_record["semantic_sha256"], label="source suite semantic_sha256")
    ):
        raise ValueError("v4 suite source identity mismatch")
    expected_ids = DEVELOPMENT_IDS if expected_role == "development" else FORMAL_IDS
    raw_tasks = payload["tasks"]
    if type(raw_tasks) is not list or len(raw_tasks) != len(expected_ids):
        raise ValueError("v4 suite task count/order mismatch")
    if type(payload["task_count"]) is not int or payload["task_count"] != len(expected_ids):
        raise ValueError("v4 suite task_count mismatch")
    if type(payload["input_count"]) is not int or payload["input_count"] != 3 * len(expected_ids):
        raise ValueError("v4 suite input_count mismatch")
    tasks: list[V4TaskInputs] = []
    input_snapshots = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for raw, source_task, expected_id in zip(raw_tasks, source_identity.tasks, expected_ids):
        task = _exact_mapping(raw, {"case_id", "inputs"}, label="v4 suite task")
        if type(task["case_id"]) is not str or task["case_id"] != expected_id:
            raise ValueError("v4 suite case order drifted")
        raw_inputs = task["inputs"]
        if type(raw_inputs) is not list or len(raw_inputs) != 3:
            raise ValueError("v4 suite task must contain x/y/z")
        inputs: list[V4BoundInput] = []
        for raw_input, source_input, axis in zip(raw_inputs, source_task.inputs, AXES):
            item = _exact_mapping(
                raw_input, {"axis", "source_path", "resolved_path", "sha256"}, label="v4 suite input"
            )
            if item["axis"] != axis or type(item["axis"]) is not str:
                raise ValueError("v4 suite input axis drifted")
            path = _canonical_absolute(Path(item["resolved_path"]), label="v4 suite input")
            if item["source_path"] != str(path) or str(source_input.resolved_path) != str(path):
                raise ValueError("v4 suite input canonical path drifted")
            digest = _digest(item["sha256"], label="v4 suite input sha256")
            current = snapshot_regular_file(path)
            if current.sha256 != digest or digest != source_input.sha256:
                raise ValueError("v4 suite input hash changed")
            input_snapshots.append(current)
            if str(path) in seen_paths or digest in seen_hashes:
                raise ValueError("duplicate v4 suite input path/hash")
            seen_paths.add(str(path))
            seen_hashes.add(digest)
            inputs.append(V4BoundInput(axis, str(path), str(path), digest))
        tasks.append(V4TaskInputs(expected_id, (inputs[0], inputs[1], inputs[2]), expected_role))
    for current in input_snapshots:
        revalidate_snapshot(current)
    revalidate_snapshot(snapshot)
    return V4SuiteIdentity(
        path=snapshot.path,
        sha256=snapshot.sha256,
        semantic_sha256=semantic,
        role=expected_role,
        tasks=tuple(tasks),
    )


def load_v4_suite_identity(path: Path, expected_role: str) -> V4SuiteIdentity:
    if expected_role not in ("development", "formal"):
        raise ValueError("expected_role must be development or formal")
    absolute = _canonical_absolute(Path(path), label="v4 suite")
    root = absolute.parent.parent
    expected = root / "manifests" / f"{expected_role}_suite.json"
    if absolute != expected:
        raise ValueError("v4 suite is not at its canonical root path")
    snapshot = snapshot_regular_file(absolute, root=root)
    payload = _decode(snapshot.payload, label="v4 suite")
    if snapshot.payload != canonical_json_bytes(payload):
        raise ValueError("v4 suite is not canonical JSON")
    return _validate_suite_payload(payload, snapshot=snapshot, expected_role=expected_role)


def _validated_identity(suite: object, role: str) -> V4SuiteIdentity:
    if not isinstance(suite, V4SuiteIdentity) or suite.role != role:
        raise ValueError(f"{role} suite identity has wrong type/role")
    expected_ids = DEVELOPMENT_IDS if role == "development" else FORMAL_IDS
    if tuple(task.case_id for task in suite.tasks) != expected_ids:
        raise ValueError(
            f"{role} task order/count drifted before overlap/duplicate validation"
        )
    return suite


def assert_v4_suites_disjoint(
    development: V4SuiteIdentity, formal: V4SuiteIdentity
) -> None:
    dev = _validated_identity(development, "development")
    frm = _validated_identity(formal, "formal")
    ids: set[str] = set()
    paths: set[str] = set()
    hashes: set[str] = set()
    for suite in (dev, frm):
        local_ids: set[str] = set()
        local_paths: set[str] = set()
        local_hashes: set[str] = set()
        for task in suite.tasks:
            if not isinstance(task, V4TaskInputs) or task.role != suite.role:
                raise ValueError("suite task role/type drifted")
            if task.case_id in local_ids or task.case_id in ids:
                raise ValueError("case ID overlap or duplicate")
            local_ids.add(task.case_id)
            if len(task.inputs) != 3:
                raise ValueError("suite task input count drifted")
            for item, axis in zip(task.inputs, AXES):
                if not isinstance(item, V4BoundInput) or item.axis != axis:
                    raise ValueError("suite input identity/order drifted")
                _digest(item.sha256, label="suite input sha256")
                if item.resolved_path in local_paths or item.resolved_path in paths:
                    raise ValueError("input path overlap or duplicate")
                if item.sha256 in local_hashes or item.sha256 in hashes:
                    raise ValueError("input hash overlap or duplicate")
                local_paths.add(item.resolved_path)
                local_hashes.add(item.sha256)
        ids.update(local_ids)
        paths.update(local_paths)
        hashes.update(local_hashes)


def _candidate_rows() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": candidate_id,
            "order_index": order,
            "dic_max_ratio": dic,
            "dts_max_ratio": dts,
        }
        for candidate_id, order, dic, dts in CANDIDATES
    ]


def _candidate_registry_payload(
    unit_path: Path,
    unit_payload: Mapping[str, object],
    *,
    unit_bytes: bytes | None = None,
) -> dict[str, object]:
    if unit_bytes is None:
        unit_snapshot = snapshot_regular_file(unit_path, root=unit_path.parent.parent)
        unit_path_text = unit_snapshot.path
        unit_sha256 = unit_snapshot.sha256
    else:
        if type(unit_bytes) is not bytes or unit_bytes != canonical_json_bytes(dict(unit_payload)):
            raise ValueError("prospective unit bytes differ from their canonical payload")
        unit_path_text = str(_canonical_absolute(unit_path, label="unit coefficients", must_exist=False))
        unit_sha256 = hashlib.sha256(unit_bytes).hexdigest()
    semantic = _digest(unit_payload.get("semantic_sha256"), label="unit semantic_sha256")
    payload: dict[str, object] = {
        "schema": REGISTRY_SCHEMA,
        "status": "REGISTERED",
        "unit_coefficients": {
            "path": unit_path_text,
            "sha256": unit_sha256,
            "semantic_sha256": semantic,
        },
        "candidates": _candidate_rows(),
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    return payload


def validate_candidate_registry(path: Path) -> Mapping[str, object]:
    absolute = _canonical_absolute(Path(path), label="candidate registry")
    root = absolute.parent.parent
    if absolute != root / "manifests" / "candidate_registry.json":
        raise ValueError("candidate registry is not at its canonical v4 path")
    snapshot = snapshot_regular_file(absolute, root=root)
    payload = _decode(snapshot.payload, label="candidate registry")
    if snapshot.payload != canonical_json_bytes(payload):
        raise ValueError("candidate registry is not canonical JSON")
    _exact_mapping(
        payload,
        {"schema", "status", "unit_coefficients", "candidates", "semantic_sha256"},
        label="candidate registry",
    )
    if payload["schema"] != REGISTRY_SCHEMA or payload["status"] != "REGISTERED":
        raise ValueError("candidate registry schema/status mismatch")
    _semantic(payload, label="candidate registry")
    unit = _exact_mapping(
        payload["unit_coefficients"], {"path", "sha256", "semantic_sha256"}, label="unit identity"
    )
    unit_path = _canonical_absolute(Path(unit["path"]), label="unit coefficients")
    if unit_path != root / "analysis" / "unit_coefficients.json":
        raise ValueError("candidate registry crosses v4 roots")
    unit_snapshot = snapshot_regular_file(unit_path, root=root)
    if unit_snapshot.sha256 != _digest(unit["sha256"], label="unit sha256"):
        raise ValueError("candidate registry unit bytes changed")
    from shadowforge_pipeline.soft_voxel_v4.calibration import validate_unit_coefficients

    unit_payload = validate_unit_coefficients(unit_path)
    if unit_payload["semantic_sha256"] != _digest(
        unit["semantic_sha256"], label="unit semantic sha256"
    ):
        raise ValueError("candidate registry unit semantic identity mismatch")
    if payload["candidates"] != _candidate_rows():
        raise ValueError("candidate registry rows/order/ratios drifted")
    for row, expected in zip(payload["candidates"], _candidate_rows()):
        if type(row) is not dict or any(type(row[key]) is not type(expected[key]) for key in expected):
            raise ValueError("candidate registry row scalar types drifted")
    revalidate_snapshot(snapshot, root=root)
    revalidate_snapshot(unit_snapshot, root=root)
    return json.loads(canonical_json_bytes(payload))


def build_binding_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind immutable v3 suites into the soft-loss-v4 root",
        allow_abbrev=False,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--formal-source", type=Path, required=True)
    return parser


def binding_main(argv: Sequence[str] | None = None) -> int:
    args = build_binding_parser().parse_args(argv)
    # Validate every source and disjointness before bootstrap writes even a marker.
    development_payload, development_source = _suite_payload(
        args.development_source, "development"
    )
    formal_payload, formal_source = _suite_payload(args.formal_source, "formal")
    development_probe = _identity_from_payload(development_payload, Path(args.development_source))
    formal_probe = _identity_from_payload(formal_payload, Path(args.formal_source))
    assert_v4_suites_disjoint(development_probe, formal_probe)
    root = _canonical_absolute(Path(args.output_root), label="output root", must_exist=False)
    marker = bootstrap_v4_root(root)
    development_target = root / "manifests" / "development_suite.json"
    formal_target = root / "manifests" / "formal_suite.json"
    expected_dev = canonical_json_bytes(development_payload)
    expected_formal = canonical_json_bytes(formal_payload)

    def precommit() -> None:
        revalidate_snapshot(development_source)
        revalidate_snapshot(formal_source)
        current_dev, _ = _suite_payload(args.development_source, "development")
        current_formal, _ = _suite_payload(args.formal_source, "formal")
        if canonical_json_bytes(current_dev) != expected_dev or canonical_json_bytes(current_formal) != expected_formal:
            raise ValueError("suite source drift before publication")
        assert_v4_suites_disjoint(
            _identity_from_payload(current_dev, Path(args.development_source)),
            _identity_from_payload(current_formal, Path(args.formal_source)),
        )
        if not marker.is_file():
            raise ValueError("v4 root marker disappeared")

    _publish_bundle(
        {development_target: expected_dev, formal_target: expected_formal},
        output_root=root,
        precommit=precommit,
    )
    load_v4_suite_identity(development_target, "development")
    load_v4_suite_identity(formal_target, "formal")
    return 0


def _identity_from_payload(payload: Mapping[str, object], source_path: Path) -> V4SuiteIdentity:
    role = str(payload["role"])
    tasks = []
    for task in payload["tasks"]:
        inputs = tuple(V4BoundInput(**item) for item in task["inputs"])
        tasks.append(V4TaskInputs(task["case_id"], inputs, role))
    return V4SuiteIdentity(
        str(source_path.resolve(strict=True)),
        hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        str(payload["semantic_sha256"]),
        role,
        tuple(tasks),
    )


# The publication state machine below is a v4-namespaced copy of the final
# approved v3 create-once/recovery implementation. Keeping the copy here
# avoids importing or changing v3 while preserving its crash semantics.
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
        f".pcraft-soft-loss-v4-write-{name}-{uuid.uuid4().hex}"
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
    encoded_name = name.encode("utf-8").hex()
    quarantine = f".pcraft-soft-loss-v4-remove-{encoded_name}-{uuid.uuid4().hex}"
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


def _remove_empty_retirement_namespace(
    parent_descriptor: int, descriptor: int
) -> None:
    """Remove the verified private namespace once every retired entry is gone."""

    if os.listdir(descriptor):
        raise ValueError("publication retirement namespace is not empty")
    opened = os.fstat(descriptor)
    current = os.stat(
        _RETIREMENT_DIRECTORY,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_dev != current.st_dev
        or opened.st_ino != current.st_ino
    ):
        raise ValueError("publication retirement namespace changed before removal")
    os.rmdir(_RETIREMENT_DIRECTORY, dir_fd=parent_descriptor)
    removed = os.fstat(descriptor)
    if removed.st_dev != opened.st_dev or removed.st_ino != opened.st_ino or removed.st_nlink != 0:
        raise ValueError("publication retirement namespace removal changed identity")
    os.fsync(parent_descriptor)


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
        if _REMOVE_NAME.fullmatch(name) is not None:
            _unlink_bound_entry_under_cooperative_lock(
                descriptor,
                name,
                binding,
                label=f"{label} file {name}",
            )
        else:
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
    retired_name = f".pcraft-soft-loss-v4-retired-{cleanup_match.group(1)}"
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
        _remove_empty_retirement_namespace(
            parent_descriptor, retirement_descriptor
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
    stage_name = f".pcraft-soft-loss-v4-stage-{token}"
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
            cleanup_name = f".pcraft-soft-loss-v4-cleanup-{token}"
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
            elif (
                len(remove_entries) == 1
                and _remove_intended_name(remove_entries[0]) == "stage_manifest.json"
            ):
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
                    intended_name = _remove_intended_name(inner_name)
                    record = record_by_name.get(intended_name)
                    if (
                        record is None
                        or intended_name in entries
                        or len(observed_data) != record["size"]
                        or hashlib.sha256(observed_data).hexdigest()
                        != record["sha256"]
                    ):
                        raise ValueError(
                            "publication recovery inner quarantine differs from its named payload"
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
    cleanup_name = f".pcraft-soft-loss-v4-cleanup-{token}"
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
            _remove_empty_retirement_namespace(
                output_descriptor, retirement_descriptor
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
        cleanup_name = f".pcraft-soft-loss-v4-cleanup-{name_match.group(1)}"
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
                stage_binding = _require_mapping(
                    stage_bindings[str(record["stage_name"])],
                    label="active staged payload binding",
                )
                try:
                    os.link(
                        str(record["stage_name"]),
                        target.name,
                        src_dir_fd=stage_descriptor,
                        dst_dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                except BaseException:
                    # A test hook or asynchronous Python exception can arrive
                    # after link(2) succeeded. Bind that exact stage inode so
                    # ordinary rollback cannot strand an unowned partial final.
                    try:
                        linked_after_error = _snapshot_regular_at(
                            parent_descriptor,
                            target.name,
                            label=f"interrupted transaction-created final {target}",
                        )
                    except (OSError, ValueError):
                        pass
                    else:
                        if (
                            linked_after_error.get("device") == stage_binding.get("device")
                            and linked_after_error.get("inode") == stage_binding.get("inode")
                            and linked_after_error.get("data") == expected
                        ):
                            created.append((target, stage_binding))
                            final_bindings[target] = linked_after_error
                    raise
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




__all__ = [
    "V4BoundInput",
    "V4SuiteIdentity",
    "V4TaskInputs",
    "assert_v4_suites_disjoint",
    "bind_v4_suite",
    "binding_main",
    "build_binding_parser",
    "load_v4_suite_identity",
    "validate_candidate_registry",
]
