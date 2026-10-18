"""Fail-closed CLI and resolved scientific identity for soft-loss v4."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Mapping

import torch

from .losses import registered_profiles, resolve_profile
from .trainer_step import V4StepConfig


LOCKED_INTERPRETER = "/home/lin/anaconda3/envs/shadowart-revisited/bin/python"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPECIFICATION_SHA256 = "c5d055ab05f8c8202807c718f338a8d8c904038d53d114e7dc567e980ec358c8"
V3_BASELINE_MANIFEST_SHA256 = "770e276eb8b07f040107552b2bfb5fc8a72f8d5289533ef50c41e55c2966bd02"
PARSER_NAMESPACE_KEYS = frozenset(
    {
        "run_kind",
        "device",
        "output_root",
        "run_id",
        "stage",
        "task_id",
        "replicate_role",
        "replicate_index",
        "steps",
        "learning_rate",
        "shadow_files",
        "resolution",
        "export_resolution",
        "profile",
        "suite_manifest",
        "specification",
        "unit_coefficients",
        "candidate_registry",
        "candidate_id",
        "resume",
        "export_final_step",
        "soft_only_eval",
    }
)
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CANDIDATES = (
    ("r04_r04", 0, 0.04, 0.04),
    ("r08_r08", 1, 0.08, 0.08),
    ("r12_r12", 2, 0.12, 0.12),
    ("r12_r08", 3, 0.12, 0.08),
    ("r08_r12", 4, 0.08, 0.12),
)
_CONFIG_KEYS = {
    "schema",
    "stack_version",
    "run_kind",
    "run_identity",
    "output_root_identity",
    "device",
    "steps",
    "learning_rate",
    "resolution",
    "export_resolution",
    "optimizer",
    "seed",
    "shadow_inputs_xyz",
    "suite_manifest",
    "specification",
    "profile",
    "objective",
    "schedule",
    "gradient_guard",
    "step_config",
    "budget_identity",
    "unit_coefficient_identity",
    "candidate_registry_identity",
    "checkpoint_policy",
    "protocol",
    "evaluation_thresholds",
    "soft_only",
    "final_step_export_only",
    "implementation",
    "interpreter",
    "determinism",
}


@dataclass(frozen=True)
class BoundFileSnapshot:
    path: str
    sha256: str
    payload: bytes


@dataclass(frozen=True)
class V4BudgetCandidate:
    candidate_id: str
    order_index: int
    dic_max_ratio: float
    dts_max_ratio: float


@dataclass(frozen=True)
class ResolvedV4Run:
    config: dict[str, object]
    config_hash: str
    step_config: V4StepConfig
    shadow_snapshots: tuple[BoundFileSnapshot, BoundFileSnapshot, BoundFileSnapshot]
    root_snapshot: BoundFileSnapshot
    suite_snapshot: BoundFileSnapshot
    specification_snapshot: BoundFileSnapshot
    unit_coefficient_snapshot: BoundFileSnapshot | None
    candidate_registry_snapshot: BoundFileSnapshot | None
    source_snapshots: tuple[BoundFileSnapshot, ...]
    interpreter_snapshot: BoundFileSnapshot


def _strict_str(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty exact str")
    return value


def _strict_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact bool")
    return value


def _strict_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an exact built-in integer")
    return value


def _strict_float(
    name: str,
    value: object,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite Python float")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _digest(name: str, value: object) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identifier(name: str, value: object) -> str:
    text = _strict_str(name, value)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ValueError(f"{name} must be a canonical lowercase identifier")
    return text


def _run_id(value: object) -> str:
    text = _strict_str("run_id", value)
    if "\\" in text:
        raise ValueError("run_id must use canonical POSIX separators")
    pure = PurePosixPath(text)
    if pure.is_absolute() or pure.as_posix() != text or not pure.parts:
        raise ValueError("run_id must be a canonical relative POSIX path")
    if any(part in ("", ".", "..") or _IDENTIFIER.fullmatch(part) is None for part in pure.parts):
        raise ValueError("run_id contains an invalid path component")
    return text


def _canonical_directory(path_value: object) -> Path:
    text = _strict_str("output_root", path_value)
    path = Path(text)
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise ValueError("output_root must be absolute and canonical")
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError("output_root must already exist") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("output_root must not be a symlink")
    if not stat.S_ISDIR(metadata.st_mode) or path.resolve(strict=True) != path:
        raise ValueError("output_root must be a canonical directory without symlinks")
    return path


def _canonical_path(name: str, value: object) -> Path:
    text = _strict_str(name, value)
    path = Path(text)
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise ValueError(f"{name} must be an absolute canonical path")
    return path


def _json_from_snapshot(snapshot: BoundFileSnapshot, *, name: str) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key in {name}: {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(snapshot.payload.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be valid UTF-8 JSON") from error
    if type(payload) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    from .provenance import canonical_json_bytes

    canonical_json_bytes(payload)
    return payload


def _semantic_payload(payload: dict[str, object], *, name: str) -> str:
    from .provenance import canonical_hash

    if "semantic_sha256" not in payload:
        raise ValueError(f"{name} must contain semantic_sha256")
    recorded = _digest(f"{name} semantic_sha256", payload["semantic_sha256"])
    content = dict(payload)
    del content["semantic_sha256"]
    if canonical_hash(content) != recorded:
        raise ValueError(f"{name} semantic digest mismatch")
    return recorded


def v4_root_manifest_payload() -> dict[str, object]:
    return {
        "schema": "pcraft_soft_loss_v4_root_v1",
        "namespace": "soft_loss_v4",
        "stack_version": 4,
        "specification_sha256": SPECIFICATION_SHA256,
        "v3_baseline_manifest_sha256": V3_BASELINE_MANIFEST_SHA256,
    }


def bootstrap_v4_root(output_root: Path) -> Path:
    """Create or authenticate one v4 campaign root and immutable marker."""

    from .provenance import publish_json_once

    root = Path(output_root)
    if not root.is_absolute() or root != Path(os.path.abspath(root)):
        raise ValueError("v4 root must be absolute and canonical")
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        missing: list[Path] = []
        ancestor = root
        while not os.path.lexists(ancestor):
            missing.append(ancestor)
            ancestor = ancestor.parent
        if ancestor.resolve(strict=True) != ancestor or ancestor.is_symlink():
            raise ValueError("v4 root parent must contain no symlink")
        try:
            root.mkdir(parents=True)
        except FileExistsError:
            pass
        for created in reversed(missing):
            if created.is_symlink() or created.resolve(strict=True) != created:
                raise ValueError("v4 root parent must contain no symlink")
    else:
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("v4 root must not be a symlink")
        if not stat.S_ISDIR(metadata.st_mode) or root.resolve(strict=True) != root:
            raise ValueError("v4 root must be a canonical directory")
    marker = root / "V4_ROOT.json"
    descriptor = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        entries = tuple(root.iterdir())
        if not os.path.lexists(marker) and entries:
            raise ValueError("refusing to adopt a non-empty foreign v4 root")
        publish_json_once(marker, v4_root_manifest_payload())
        if json.loads(marker.read_text(encoding="utf-8")) != v4_root_manifest_payload():
            raise ValueError("v4 root marker differs from the canonical payload")
    finally:
        os.close(descriptor)
    return marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P-CRAFT soft-loss-v4 trainer (soft-only, final-step export)",
        allow_abbrev=False,
    )
    parser.add_argument("--run-kind", choices=("calibration_source", "experiment"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--replicate-role",
        choices=("calibration", "primary", "technical_repeat", "appendix"),
        required=True,
    )
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--steps", type=int, choices=(200, 1000, 2000), required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--shadow-files", nargs=3, required=True)
    parser.add_argument("--resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--export-resolution", type=int, choices=(128, 256), required=True)
    parser.add_argument("--profile", choices=registered_profiles(), required=True)
    parser.add_argument("--suite-manifest", required=True)
    parser.add_argument("--specification", required=True)
    parser.add_argument("--unit-coefficients")
    parser.add_argument("--candidate-registry")
    parser.add_argument("--candidate-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-final-step", action="store_true", required=True)
    parser.add_argument("--soft-only-eval", action="store_true", required=True)
    return parser


def _validate_namespace(args: argparse.Namespace) -> None:
    if type(args) is not argparse.Namespace or set(vars(args)) != PARSER_NAMESPACE_KEYS:
        raise ValueError("trainer Namespace must contain the exact parser key set")
    for name in (
        "run_kind",
        "device",
        "output_root",
        "run_id",
        "stage",
        "task_id",
        "replicate_role",
        "profile",
        "suite_manifest",
        "specification",
    ):
        _strict_str(name, getattr(args, name))
    for name in ("unit_coefficients", "candidate_registry", "candidate_id"):
        value = getattr(args, name)
        if value is not None and type(value) is not str:
            raise ValueError(f"{name} must be an exact str or null")
    _strict_int("replicate_index", args.replicate_index, minimum=0)
    _strict_int("steps", args.steps, minimum=1)
    _strict_int("resolution", args.resolution, minimum=1)
    _strict_int("export_resolution", args.export_resolution, minimum=1)
    _strict_float("learning_rate", args.learning_rate, positive=True)
    _strict_bool("resume", args.resume)
    _strict_bool("export_final_step", args.export_final_step)
    _strict_bool("soft_only_eval", args.soft_only_eval)
    if type(args.shadow_files) is not list or len(args.shadow_files) != 3 or any(
        type(path) is not str for path in args.shadow_files
    ):
        raise ValueError("shadow_files must be an exact ordered list of three paths")


def _validate_run_matrix(args: argparse.Namespace, *, suite_role: str) -> None:
    if args.steps not in (200, 1000, 2000):
        raise ValueError("steps must be exactly 200, 1000, or 2000")
    if args.learning_rate != 0.007:
        raise ValueError("learning rate is frozen to exactly 0.007")
    if args.resolution not in (128, 256) or args.export_resolution != args.resolution:
        raise ValueError("resolution/export resolution must match at 128 or 256")
    if args.steps in (200, 1000) and args.resolution != 128:
        raise ValueError("200/1000-step runs require resolution 128")
    if args.run_kind == "calibration_source":
        if (
            args.profile != "l12_bal"
            or args.steps != 2000
            or args.resolution != 128
            or args.replicate_role != "calibration"
            or args.replicate_index != 0
            or suite_role != "development"
        ):
            raise ValueError("calibration_source requires l12_bal 2000/128 development calibration role")
        if any(value is not None for value in (args.unit_coefficients, args.candidate_registry, args.candidate_id)):
            raise ValueError("calibration_source accepts no unit or candidate identity")
    else:
        if args.profile in ("v4_no_l1", "v4_no_l2"):
            if args.replicate_role != "appendix" or args.replicate_index != 0 or args.steps != 2000:
                raise ValueError("appendix profiles require appendix role index 0 and 2000 steps")
        elif args.replicate_role == "primary":
            if args.replicate_index != 0:
                raise ValueError("primary runs require replicate index 0")
        elif args.replicate_role == "technical_repeat":
            if args.replicate_index != 1:
                raise ValueError("technical repeats require replicate index 1")
        else:
            raise ValueError("non-appendix experiments require primary or technical_repeat role")
        optional = (args.unit_coefficients, args.candidate_registry, args.candidate_id)
        if args.profile == "l12_bal":
            if any(value is not None for value in optional):
                raise ValueError("shared baseline accepts no unit or candidate inputs")
        elif any(value is None for value in optional):
            raise ValueError("auxiliary profiles require unit, registry, and candidate ID")


def _identity_record(payload: object, *, keys: set[str], name: str) -> dict[str, object]:
    if type(payload) is not dict or set(payload) != keys:
        raise ValueError(f"{name} must have the exact key set")
    result = dict(payload)
    if "path" in keys:
        _canonical_path(f"{name}.path", result["path"])
    for key in keys & {"sha256", "semantic_sha256", "status_sha256", "artifact_manifest_sha256", "config_hash"}:
        _digest(f"{name}.{key}", result[key])
    return result


def _validated_suite(snapshot: BoundFileSnapshot, *, root: Path) -> tuple[dict[str, object], str]:
    payload = _json_from_snapshot(snapshot, name="suite manifest")
    expected = {
        "schema",
        "status",
        "role",
        "source_suite",
        "tasks",
        "task_count",
        "input_count",
        "input_order",
        "protocol",
        "profiles",
        "semantic_sha256",
    }
    if set(payload) != expected or payload["schema"] != "pcraft_soft_loss_v4_suite_v1" or payload["status"] != "REGISTERED":
        raise ValueError("suite manifest has the wrong exact schema/status")
    role = payload["role"]
    if role not in ("development", "formal"):
        raise ValueError("suite role must be development or formal")
    expected_path = root / "manifests" / f"{role}_suite.json"
    if Path(snapshot.path) != expected_path:
        raise ValueError("suite path is not the role-appropriate canonical root child")
    source = _identity_record(
        payload["source_suite"],
        keys={"path", "sha256", "semantic_sha256"},
        name="source_suite",
    )
    del source
    if payload["input_order"] != ["x", "y", "z"] or payload["profiles"] != list(registered_profiles()):
        raise ValueError("suite input order or profile registry drifted")
    protocol = payload["protocol"]
    expected_protocol = {
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
    if protocol != expected_protocol:
        raise ValueError("suite protocol drifted")
    tasks = payload["tasks"]
    if type(tasks) is not list or not tasks:
        raise ValueError("suite tasks must be a non-empty list")
    case_ids: list[str] = []
    paths: list[str] = []
    digests: list[str] = []
    for task in tasks:
        if type(task) is not dict or set(task) != {"case_id", "inputs"}:
            raise ValueError("suite task has the wrong schema")
        case_ids.append(_identifier("suite case_id", task["case_id"]))
        inputs = task["inputs"]
        if type(inputs) is not list or len(inputs) != 3:
            raise ValueError("suite task must contain ordered x/y/z inputs")
        for expected_axis, item in zip("xyz", inputs):
            if type(item) is not dict or set(item) != {"axis", "source_path", "resolved_path", "sha256"}:
                raise ValueError("suite input has the wrong schema")
            if item["axis"] != expected_axis:
                raise ValueError("suite x/y/z input order drifted")
            source_path = _canonical_path("suite source_path", item["source_path"])
            resolved_path = _canonical_path("suite resolved_path", item["resolved_path"])
            if source_path != resolved_path:
                raise ValueError("suite source_path and resolved_path identity differ")
            paths.append(str(resolved_path))
            digests.append(_digest("suite input sha256", item["sha256"]))
    if len(case_ids) != len(set(case_ids)) or len(paths) != len(set(paths)) or len(digests) != len(set(digests)):
        raise ValueError("suite task/input identities must be unique")
    if payload["task_count"] != len(tasks) or payload["input_count"] != len(paths):
        raise ValueError("suite task/input counts mismatch")
    semantic = _semantic_payload(payload, name="suite manifest")
    return payload, semantic


def _select_suite_task(
    suite: dict[str, object],
    *,
    task_id: str,
    snapshots: tuple[BoundFileSnapshot, BoundFileSnapshot, BoundFileSnapshot],
) -> None:
    matches = [task for task in suite["tasks"] if task["case_id"] == task_id]
    if len(matches) != 1:
        raise ValueError("task_id is not unique suite membership")
    for axis, item, snapshot in zip("xyz", matches[0]["inputs"], snapshots):
        if item["axis"] != axis or item["resolved_path"] != snapshot.path or item["source_path"] != snapshot.path or item["sha256"] != snapshot.sha256:
            raise ValueError("ordered x/y/z path+byte suite membership mismatch")


def _validated_stage_identity(identity: object, *, root: Path) -> None:
    from .provenance import snapshot_regular_file

    checked = _identity_record(
        identity,
        keys={"path", "sha256", "semantic_sha256"},
        name="calibration_stage",
    )
    path = Path(checked["path"])
    expected_path = root / "analysis" / "calibration_stage.json"
    if path != expected_path:
        raise ValueError(
            "calibration stage must be the canonical post-run "
            "analysis/calibration_stage.json artifact"
        )
    snapshot = snapshot_regular_file(path, root=root)
    if snapshot.sha256 != checked["sha256"]:
        raise ValueError("calibration stage byte identity mismatch")
    payload = _json_from_snapshot(snapshot, name="calibration stage")
    if _semantic_payload(payload, name="calibration stage") != checked["semantic_sha256"]:
        raise ValueError("calibration stage semantic identity mismatch")


def _validated_unit_and_registry(
    *,
    root: Path,
    unit_snapshot: BoundFileSnapshot,
    registry_snapshot: BoundFileSnapshot,
    candidate_id: str,
    root_identity: dict[str, object],
) -> tuple[float, float, V4BudgetCandidate, str, str]:
    if Path(unit_snapshot.path) != root / "analysis" / "unit_coefficients.json":
        raise ValueError("unit coefficient artifact is not at its canonical root path")
    if Path(registry_snapshot.path) != root / "manifests" / "candidate_registry.json":
        raise ValueError("candidate registry is not at its canonical root path")
    unit = _json_from_snapshot(unit_snapshot, name="unit coefficients")
    unit_keys = {
        "schema",
        "status",
        "calibration_run",
        "calibration_stage",
        "calibration_task_identity",
        "development_suite_identity",
        "estimator",
        "coefficients",
        "semantic_sha256",
    }
    if set(unit) != unit_keys or unit["schema"] != "pcraft_soft_loss_v4_unit_coefficients_v1" or unit["status"] != "APPROVED_FROZEN":
        raise ValueError("unit coefficient artifact has the wrong exact schema/status")
    unit_semantic = _semantic_payload(unit, name="unit coefficients")
    calibration_run = _identity_record(
        unit["calibration_run"],
        keys={"path", "run_id", "config_hash", "status_sha256", "artifact_manifest_sha256"},
        name="calibration_run",
    )
    _run_id(calibration_run["run_id"])
    calibration_path = Path(calibration_run["path"])
    if not calibration_path.is_relative_to(root):
        raise ValueError("calibration run crosses v4 output roots")
    _validated_stage_identity(unit["calibration_stage"], root=root)
    task_identity = unit["calibration_task_identity"]
    if type(task_identity) is not dict or set(task_identity) != {"task_id"}:
        raise ValueError("calibration_task_identity has wrong schema")
    task_id = _identifier("calibration task", task_identity["task_id"])
    suite_identity = _identity_record(
        unit["development_suite_identity"],
        keys={"path", "sha256", "semantic_sha256", "role"},
        name="development_suite_identity",
    )
    if suite_identity["role"] != "development":
        raise ValueError("unit coefficients must bind a development suite")
    estimator = unit["estimator"]
    if type(estimator) is not dict or set(estimator) != {"name", "completed_updates", "dic_endpoint", "dts_endpoints", "candidate_independent"}:
        raise ValueError("unit estimator has the wrong exact schema")
    _strict_str("estimator name", estimator["name"])
    updates = estimator["completed_updates"]
    if type(updates) is not list or not updates or any(type(item) is not int or item <= 0 for item in updates) or updates != sorted(set(updates)):
        raise ValueError("estimator completed_updates must be increasing positive exact integers")
    if estimator["dic_endpoint"] != 0.05 or estimator["dts_endpoints"] != [0.04, 0.06] or estimator["candidate_independent"] is not True:
        raise ValueError("unit estimator endpoints/candidate independence drifted")
    coefficients = unit["coefficients"]
    if type(coefficients) is not dict or set(coefficients) != {"dic", "dts"}:
        raise ValueError("unit coefficients must contain exactly DIC and DTS")
    dic_coefficient = _strict_float("DIC unit coefficient", coefficients["dic"], positive=True)
    dts_coefficient = _strict_float("DTS unit coefficient", coefficients["dts"], positive=True)

    from .trainer import validate_completed_run

    authenticated = validate_completed_run(calibration_path)
    if (
        authenticated.config_hash != calibration_run["config_hash"]
        or authenticated.status_sha256 != calibration_run["status_sha256"]
        or authenticated.artifact_manifest_sha256 != calibration_run["artifact_manifest_sha256"]
        or authenticated.output_root_identity != root_identity
        or authenticated.config.get("run_kind") != "calibration_source"
    ):
        raise ValueError("authenticated calibration run identity mismatch")
    auth_run_identity = authenticated.config.get("run_identity")
    auth_suite = authenticated.config.get("suite_manifest")
    if not isinstance(auth_run_identity, Mapping) or auth_run_identity.get("task_id") != task_id:
        raise ValueError("calibration task substitution detected")
    if not isinstance(auth_suite, Mapping) or dict(auth_suite) != suite_identity:
        raise ValueError("calibration development suite substitution detected")
    if auth_run_identity.get("run_id") is not None and auth_run_identity.get("run_id") != calibration_run["run_id"]:
        raise ValueError("calibration run_id mismatch")

    registry = _json_from_snapshot(registry_snapshot, name="candidate registry")
    registry_keys = {"schema", "status", "unit_coefficients", "candidates", "semantic_sha256"}
    if set(registry) != registry_keys or registry["schema"] != "pcraft_soft_loss_v4_candidate_registry_v1" or registry["status"] != "REGISTERED":
        raise ValueError("candidate registry has the wrong exact schema/status")
    registry_semantic = _semantic_payload(registry, name="candidate registry")
    unit_identity = _identity_record(
        registry["unit_coefficients"],
        keys={"path", "sha256", "semantic_sha256"},
        name="registry unit_coefficients",
    )
    if unit_identity != {"path": unit_snapshot.path, "sha256": unit_snapshot.sha256, "semantic_sha256": unit_semantic}:
        raise ValueError("candidate registry does not bind the supplied unit artifact")
    expected_rows = [
        {
            "candidate_id": candidate,
            "order_index": order,
            "dic_max_ratio": dic,
            "dts_max_ratio": dts,
        }
        for candidate, order, dic, dts in _CANDIDATES
    ]
    if registry["candidates"] != expected_rows:
        raise ValueError("candidate rows/order/ratios have drifted")
    rows = [row for row in expected_rows if row["candidate_id"] == candidate_id]
    if len(rows) != 1:
        raise ValueError("unknown candidate ID")
    row = rows[0]
    candidate = V4BudgetCandidate(
        candidate_id=row["candidate_id"],
        order_index=row["order_index"],
        dic_max_ratio=row["dic_max_ratio"],
        dts_max_ratio=row["dts_max_ratio"],
    )
    return dic_coefficient, dts_coefficient, candidate, unit_semantic, registry_semantic


def _checkpoint_policy(run_kind: str, steps: int) -> dict[str, object]:
    if run_kind == "calibration_source":
        return {"name": "unit_calibration", "commit_interval": 100, "retained_updates": []}
    if steps == 2000:
        return {"name": "convergence", "commit_interval": 100, "retained_updates": [1600, 1800]}
    return {"name": "final_only", "commit_interval": 100, "retained_updates": []}


def resolve_run_configuration(args: argparse.Namespace) -> ResolvedV4Run:
    """Resolve, validate, snapshot, and hash every v4 scientific dependency."""

    _validate_namespace(args)
    if not args.export_final_step or not args.soft_only_eval:
        raise ValueError("final-step export and soft-only evaluation acknowledgements are mandatory")
    run_id = _run_id(args.run_id)
    stage = _identifier("stage", args.stage)
    task_id = _identifier("task_id", args.task_id)
    root = _canonical_directory(args.output_root)
    from .provenance import (
        canonical_hash,
        environment_manifest,
        revalidate_snapshot,
        snapshot_implementation_sources,
        snapshot_regular_file,
    )

    marker_path = root / "V4_ROOT.json"
    root_snapshot = snapshot_regular_file(marker_path, root=root)
    if _json_from_snapshot(root_snapshot, name="v4 root marker") != v4_root_manifest_payload():
        raise ValueError("output_root is not the canonical v4 namespace")
    suite_snapshot = snapshot_regular_file(_canonical_path("suite_manifest", args.suite_manifest), root=root)
    suite, suite_semantic = _validated_suite(suite_snapshot, root=root)
    _validate_run_matrix(args, suite_role=suite["role"])
    shadow_snapshots = tuple(
        snapshot_regular_file(_canonical_path(f"{axis}-shadow", path))
        for axis, path in zip("xyz", args.shadow_files)
    )
    if len(shadow_snapshots) != 3:
        raise AssertionError("three snapshots required")
    _select_suite_task(suite, task_id=task_id, snapshots=shadow_snapshots)
    specification_snapshot = snapshot_regular_file(
        _canonical_path("specification", args.specification)
    )
    if specification_snapshot.sha256 != SPECIFICATION_SHA256:
        raise ValueError("specification bytes do not match the frozen v4 specification")
    if Path(os.path.abspath(sys.executable)) != Path(LOCKED_INTERPRETER):
        raise ValueError("v4 trainer requires the locked interpreter")
    interpreter_snapshot = snapshot_regular_file(Path(sys.executable).resolve(strict=True))
    source_manifest, source_snapshots = snapshot_implementation_sources()
    environment = environment_manifest(torch.device(args.device))
    if args.device == "cpu":
        device = "cpu"
    elif re.fullmatch(r"cuda:[0-9]+", args.device) is not None:
        device = args.device
        index = int(args.device.split(":", 1)[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"requested CUDA device is unavailable: {args.device}")
    else:
        raise ValueError("device must be exactly cpu or explicit cuda:N")
    profile = resolve_profile(args.profile)
    run_dir = root.joinpath(*PurePosixPath(run_id).parts)
    if not run_dir.is_relative_to(root):
        raise ValueError("run_id resolves outside output_root")
    root_identity: dict[str, object] = {
        "path": str(root),
        "marker_path": root_snapshot.path,
        "marker_sha256": root_snapshot.sha256,
    }
    unit_snapshot = None
    registry_snapshot = None
    budget_identity = None
    unit_identity = None
    registry_identity = None
    serialized_step_config = None
    if args.profile == "l12_bal":
        private_step_config = V4StepConfig(1.0, 1.0, 0.04, 0.04)
    else:
        unit_snapshot = snapshot_regular_file(_canonical_path("unit_coefficients", args.unit_coefficients), root=root)
        registry_snapshot = snapshot_regular_file(_canonical_path("candidate_registry", args.candidate_registry), root=root)
        dic_coefficient, dts_coefficient, candidate, unit_semantic, registry_semantic = _validated_unit_and_registry(
            root=root,
            unit_snapshot=unit_snapshot,
            registry_snapshot=registry_snapshot,
            candidate_id=_strict_str("candidate_id", args.candidate_id),
            root_identity=root_identity,
        )
        private_step_config = V4StepConfig(
            dic_coefficient,
            dts_coefficient,
            candidate.dic_max_ratio,
            candidate.dts_max_ratio,
        )
        serialized_step_config = {
            "dic_unit_coefficient": dic_coefficient,
            "dts_unit_coefficient": dts_coefficient,
            "dic_max_ratio": candidate.dic_max_ratio,
            "dts_max_ratio": candidate.dts_max_ratio,
        }
        budget_identity = {
            "candidate_id": candidate.candidate_id,
            "order_index": candidate.order_index,
            "dic_max_ratio": candidate.dic_max_ratio,
            "dts_max_ratio": candidate.dts_max_ratio,
        }
        unit_identity = {"path": unit_snapshot.path, "sha256": unit_snapshot.sha256, "semantic_sha256": unit_semantic}
        registry_identity = {"path": registry_snapshot.path, "sha256": registry_snapshot.sha256, "semantic_sha256": registry_semantic}
    config: dict[str, object] = {
        "schema": "pcraft_soft_loss_v4_config_v1",
        "stack_version": 4,
        "run_kind": args.run_kind,
        "run_identity": {
            "stage": stage,
            "task_id": task_id,
            "replicate_role": args.replicate_role,
            "replicate_index": args.replicate_index,
            "run_id": run_id,
            "run_dir": str(run_dir),
        },
        "output_root_identity": root_identity,
        "device": device,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "resolution": args.resolution,
        "export_resolution": args.export_resolution,
        "optimizer": {"name": "Adam", "initialization_logit": -4.0, "gradient_clip_norm": 1.0},
        "seed": 17,
        "shadow_inputs_xyz": [
            {"axis": axis, "path": snapshot.path, "sha256": snapshot.sha256}
            for axis, snapshot in zip("xyz", shadow_snapshots)
        ],
        "suite_manifest": {"path": suite_snapshot.path, "sha256": suite_snapshot.sha256, "semantic_sha256": suite_semantic, "role": suite["role"]},
        "specification": {"path": specification_snapshot.path, "sha256": specification_snapshot.sha256},
        "profile": {"name": profile.name, "l1_weight": profile.l1_weight, "l2_weight": profile.l2_weight, "use_dic": profile.use_dic, "use_dts": profile.use_dts},
        "objective": {"density_parameterization": "sigmoid(logits)", "view_reduction": "mean_over_six_views", "formula": "l2_weight*L2 + l1_weight*L1", "l1_weight": profile.l1_weight, "l2_weight": profile.l2_weight},
        "schedule": {
            "progress_formula": "(global_step_before+1)/total_steps",
            "ramp_start": 0.60,
            "ramp_full": 0.75,
            "slot_cycle": [
                {"slot": 0, "component": "dic", "endpoint": None},
                {"slot": 1, "component": "dts", "endpoint": 0.04},
                {"slot": 2, "component": "dic", "endpoint": None},
                {"slot": 3, "component": "dts", "endpoint": 0.06},
            ],
        },
        "gradient_guard": {"anchor_threshold": 0.05, "anchor_in_scalar_objective": False, "projection_epsilon": 1e-12, "support_threshold": 1e-12, "cap_relative_to_base": True, "one_auxiliary_per_update": True, "clip_after_composition": 1.0},
        "step_config": serialized_step_config,
        "budget_identity": budget_identity,
        "unit_coefficient_identity": unit_identity,
        "candidate_registry_identity": registry_identity,
        "checkpoint_policy": _checkpoint_policy(args.run_kind, args.steps),
        "protocol": {"name": "sadekar_tfs_fixed_six_view", "mirror_mode": 1, "tsf_mode": 1, "tsf4_mode": 0, "camera_mode": "ortho", "image_size": 256, "control_resolution": 0, "zdist": 1.7, "volume_extent": 1.7, "samples_per_ray": 150, "camera_azimuth_deg": [0.0, 90.0, 0.0, 180.0, 270.0, 0.0], "camera_elevation_deg": [0.0, 0.0, 90.0, 0.0, 0.0, 270.0], "delivery_readout": "P - 0.8*stopgrad(max(P))"},
        "evaluation_thresholds": [0.04, 0.05, 0.06],
        "soft_only": True,
        "final_step_export_only": True,
        "implementation": source_manifest,
        "interpreter": {"path": interpreter_snapshot.path, "sha256": interpreter_snapshot.sha256, "python": environment["interpreter"]["python"]},
        "determinism": {"fixed_seed": True, "mode": "best_effort", "bitwise_reproducible": False, "reason": environment["nondeterminism_reason"], "environment": environment["environment"]},
    }
    validate_resolved_config(config)
    for snapshot, declared_root in (
        (root_snapshot, root),
        (suite_snapshot, root),
        (specification_snapshot, None),
        *((snapshot, None) for snapshot in shadow_snapshots),
        *((snapshot, PROJECT_ROOT) for snapshot in source_snapshots),
        (interpreter_snapshot, None),
    ):
        revalidate_snapshot(snapshot, root=declared_root)
    return ResolvedV4Run(
        config=config,
        config_hash=canonical_hash(config),
        step_config=private_step_config,
        shadow_snapshots=shadow_snapshots,
        root_snapshot=root_snapshot,
        suite_snapshot=suite_snapshot,
        specification_snapshot=specification_snapshot,
        unit_coefficient_snapshot=unit_snapshot,
        candidate_registry_snapshot=registry_snapshot,
        source_snapshots=source_snapshots,
        interpreter_snapshot=interpreter_snapshot,
    )


def resolved_config(args: argparse.Namespace) -> dict[str, object]:
    from .provenance import canonical_json_bytes

    payload = resolve_run_configuration(args).config
    return json.loads(canonical_json_bytes(payload))


def _exact_mapping(value: object, keys: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} has wrong exact keys")
    return value


def validate_resolved_config(payload: object) -> dict[str, object]:
    """Validate the complete built-in-only resolved schema without touching files."""

    from .provenance import canonical_json_bytes

    if type(payload) is not dict or set(payload) != _CONFIG_KEYS:
        raise ValueError("resolved config has the wrong exact top-level keys")
    canonical_json_bytes(payload)
    if payload["schema"] != "pcraft_soft_loss_v4_config_v1" or payload["stack_version"] != 4:
        raise ValueError("resolved config schema/version mismatch")
    run_kind = payload["run_kind"]
    if run_kind not in ("calibration_source", "experiment"):
        raise ValueError("resolved run kind is invalid")
    run_identity = _exact_mapping(payload["run_identity"], {"stage", "task_id", "replicate_role", "replicate_index", "run_id", "run_dir"}, "run_identity")
    _identifier("run_identity.stage", run_identity["stage"])
    _identifier("run_identity.task_id", run_identity["task_id"])
    run_id = _run_id(run_identity["run_id"])
    if run_identity["replicate_role"] not in ("calibration", "primary", "technical_repeat", "appendix"):
        raise ValueError("resolved replicate role is invalid")
    _strict_int("run_identity.replicate_index", run_identity["replicate_index"], minimum=0)
    run_dir = _canonical_path("run_identity.run_dir", run_identity["run_dir"])
    root_identity = _exact_mapping(payload["output_root_identity"], {"path", "marker_path", "marker_sha256"}, "output_root_identity")
    root = _canonical_path("output_root_identity.path", root_identity["path"])
    if _canonical_path("output_root_identity.marker_path", root_identity["marker_path"]) != root / "V4_ROOT.json":
        raise ValueError("resolved root marker path drifted")
    _digest("output_root_identity.marker_sha256", root_identity["marker_sha256"])
    if run_dir != root.joinpath(*PurePosixPath(run_id).parts) or not run_dir.is_relative_to(root):
        raise ValueError("resolved run directory does not match run_id/root")
    if payload["device"] != "cpu" and (
        type(payload["device"]) is not str
        or re.fullmatch(r"cuda:[0-9]+", payload["device"]) is None
    ):
        raise ValueError("resolved device must be cpu or explicit cuda:N")
    steps = _strict_int("steps", payload["steps"], minimum=1)
    if steps not in (200, 1000, 2000):
        raise ValueError("resolved steps are not registered")
    if _strict_float("learning_rate", payload["learning_rate"], positive=True) != 0.007:
        raise ValueError("resolved learning rate drifted")
    resolution = _strict_int("resolution", payload["resolution"], minimum=1)
    export_resolution = _strict_int("export_resolution", payload["export_resolution"], minimum=1)
    if resolution not in (128, 256) or export_resolution != resolution or (steps < 2000 and resolution != 128):
        raise ValueError("resolved resolution/step matrix drifted")
    optimizer = _exact_mapping(payload["optimizer"], {"name", "initialization_logit", "gradient_clip_norm"}, "optimizer")
    if canonical_json_bytes(optimizer) != canonical_json_bytes({"name": "Adam", "initialization_logit": -4.0, "gradient_clip_norm": 1.0}):
        raise ValueError("resolved optimizer identity drifted")
    if payload["seed"] != 17 or type(payload["seed"]) is not int:
        raise ValueError("resolved seed drifted")
    if type(payload["shadow_inputs_xyz"]) is not list or len(payload["shadow_inputs_xyz"]) != 3:
        raise ValueError("shadow_inputs_xyz must contain exactly three rows")
    shadow_paths: list[str] = []
    shadow_digests: list[str] = []
    for axis, item in zip("xyz", payload["shadow_inputs_xyz"]):
        _exact_mapping(item, {"axis", "path", "sha256"}, "shadow input")
        if item["axis"] != axis:
            raise ValueError("resolved shadow axes drifted")
        shadow_paths.append(str(_canonical_path("shadow input path", item["path"])))
        shadow_digests.append(_digest("shadow input sha256", item["sha256"]))
    if len(set(shadow_paths)) != 3 or len(set(shadow_digests)) != 3:
        raise ValueError("resolved shadow identities must be unique")
    suite = _exact_mapping(payload["suite_manifest"], {"path", "sha256", "semantic_sha256", "role"}, "suite_manifest")
    if suite["role"] not in ("development", "formal"):
        raise ValueError("resolved suite role is invalid")
    if _canonical_path("suite_manifest.path", suite["path"]) != root / "manifests" / f"{suite['role']}_suite.json":
        raise ValueError("resolved suite canonical path drifted")
    _digest("suite_manifest.sha256", suite["sha256"])
    _digest("suite_manifest.semantic_sha256", suite["semantic_sha256"])
    specification = _exact_mapping(payload["specification"], {"path", "sha256"}, "specification")
    _canonical_path("specification.path", specification["path"])
    if specification["sha256"] != SPECIFICATION_SHA256:
        raise ValueError("resolved specification digest drifted")
    profile = _exact_mapping(payload["profile"], {"name", "l1_weight", "l2_weight", "use_dic", "use_dts"}, "profile")
    objective = _exact_mapping(payload["objective"], {"density_parameterization", "view_reduction", "formula", "l1_weight", "l2_weight"}, "objective")
    schedule = _exact_mapping(payload["schedule"], {"progress_formula", "ramp_start", "ramp_full", "slot_cycle"}, "schedule")
    expected_cycle = [
        {"slot": 0, "component": "dic", "endpoint": None},
        {"slot": 1, "component": "dts", "endpoint": 0.04},
        {"slot": 2, "component": "dic", "endpoint": None},
        {"slot": 3, "component": "dts", "endpoint": 0.06},
    ]
    if canonical_json_bytes(schedule) != canonical_json_bytes({
        "progress_formula": "(global_step_before+1)/total_steps",
        "ramp_start": 0.60,
        "ramp_full": 0.75,
        "slot_cycle": expected_cycle,
    }):
        raise ValueError("resolved auxiliary schedule drifted")
    guard = _exact_mapping(payload["gradient_guard"], {"anchor_threshold", "anchor_in_scalar_objective", "projection_epsilon", "support_threshold", "cap_relative_to_base", "one_auxiliary_per_update", "clip_after_composition"}, "gradient_guard")
    if canonical_json_bytes(guard) != canonical_json_bytes({
        "anchor_threshold": 0.05,
        "anchor_in_scalar_objective": False,
        "projection_epsilon": 1e-12,
        "support_threshold": 1e-12,
        "cap_relative_to_base": True,
        "one_auxiliary_per_update": True,
        "clip_after_composition": 1.0,
    }):
        raise ValueError("resolved gradient guard drifted")
    checkpoint_policy = _exact_mapping(payload["checkpoint_policy"], {"name", "commit_interval", "retained_updates"}, "checkpoint_policy")
    if canonical_json_bytes(checkpoint_policy) != canonical_json_bytes(_checkpoint_policy(run_kind, steps)):
        raise ValueError("resolved checkpoint policy drifted")
    protocol = _exact_mapping(payload["protocol"], {"name", "mirror_mode", "tsf_mode", "tsf4_mode", "camera_mode", "image_size", "control_resolution", "zdist", "volume_extent", "samples_per_ray", "camera_azimuth_deg", "camera_elevation_deg", "delivery_readout"}, "protocol")
    expected_protocol = {
        "name": "sadekar_tfs_fixed_six_view",
        "mirror_mode": 1,
        "tsf_mode": 1,
        "tsf4_mode": 0,
        "camera_mode": "ortho",
        "image_size": 256,
        "control_resolution": 0,
        "zdist": 1.7,
        "volume_extent": 1.7,
        "samples_per_ray": 150,
        "camera_azimuth_deg": [0.0, 90.0, 0.0, 180.0, 270.0, 0.0],
        "camera_elevation_deg": [0.0, 0.0, 90.0, 0.0, 0.0, 270.0],
        "delivery_readout": "P - 0.8*stopgrad(max(P))",
    }
    if canonical_json_bytes(protocol) != canonical_json_bytes(expected_protocol):
        raise ValueError("resolved protocol drifted")
    implementation = _exact_mapping(payload["implementation"], {"schema", "algorithm", "files_sha256", "bundle_sha256"}, "implementation")
    if implementation["schema"] != "pcraft_soft_loss_v4_source_manifest_v1" or implementation["algorithm"] != "sha256":
        raise ValueError("resolved implementation schema/algorithm drifted")
    files = implementation["files_sha256"]
    from .provenance import IMPLEMENTATION_SOURCE_FILES, canonical_hash

    if type(files) is not dict or set(files) != set(IMPLEMENTATION_SOURCE_FILES):
        raise ValueError("resolved implementation source whitelist drifted")
    for relative, digest in files.items():
        if type(relative) is not str:
            raise ValueError("resolved implementation path is malformed")
        _digest("implementation source digest", digest)
    if implementation["bundle_sha256"] != canonical_hash(files):
        raise ValueError("resolved implementation bundle digest drifted")
    interpreter = _exact_mapping(payload["interpreter"], {"path", "sha256", "python"}, "interpreter")
    if interpreter["path"] != str(Path(LOCKED_INTERPRETER).resolve(strict=True)):
        raise ValueError("resolved interpreter path drifted")
    _digest("interpreter.sha256", interpreter["sha256"])
    _strict_str("interpreter.python", interpreter["python"])
    determinism = _exact_mapping(payload["determinism"], {"fixed_seed", "mode", "bitwise_reproducible", "reason", "environment"}, "determinism")
    if type(determinism["environment"]) is not dict or any(type(key) is not str or type(value) is not str for key, value in determinism["environment"].items()):
        raise ValueError("determinism environment must be an exact string mapping")
    if (
        determinism["fixed_seed"] is not True
        or determinism["mode"] != "best_effort"
        or determinism["bitwise_reproducible"] is not False
        or type(determinism["reason"]) is not str
        or "grid_sampler_3d_backward" not in determinism["reason"]
    ):
        raise ValueError("resolved determinism declaration drifted")
    registered = resolve_profile(profile["name"])
    if canonical_json_bytes(profile) != canonical_json_bytes({"name": registered.name, "l1_weight": registered.l1_weight, "l2_weight": registered.l2_weight, "use_dic": registered.use_dic, "use_dts": registered.use_dts}):
        raise ValueError("resolved profile does not match its registry")
    if canonical_json_bytes(objective) != canonical_json_bytes({
        "density_parameterization": "sigmoid(logits)",
        "view_reduction": "mean_over_six_views",
        "formula": "l2_weight*L2 + l1_weight*L1",
        "l1_weight": registered.l1_weight,
        "l2_weight": registered.l2_weight,
    }):
        raise ValueError("resolved objective identity drifted")
    auxiliary = profile["name"] != "l12_bal"
    identity_fields = (payload["step_config"], payload["budget_identity"], payload["unit_coefficient_identity"], payload["candidate_registry_identity"])
    if auxiliary and any(value is None for value in identity_fields):
        raise ValueError("auxiliary config requires all coefficient/candidate identities")
    if not auxiliary and any(value is not None for value in identity_fields):
        raise ValueError("base/calibration config must have null auxiliary identities")
    if auxiliary:
        _exact_mapping(payload["step_config"], {"dic_unit_coefficient", "dts_unit_coefficient", "dic_max_ratio", "dts_max_ratio"}, "step_config")
        budget = _exact_mapping(payload["budget_identity"], {"candidate_id", "order_index", "dic_max_ratio", "dts_max_ratio"}, "budget_identity")
        unit = _identity_record(payload["unit_coefficient_identity"], keys={"path", "sha256", "semantic_sha256"}, name="unit_coefficient_identity")
        registry = _identity_record(payload["candidate_registry_identity"], keys={"path", "sha256", "semantic_sha256"}, name="candidate_registry_identity")
        if Path(unit["path"]) != root / "analysis" / "unit_coefficients.json" or Path(registry["path"]) != root / "manifests" / "candidate_registry.json":
            raise ValueError("resolved unit/registry canonical paths drifted")
        step_config = V4StepConfig(**payload["step_config"])
        expected_rows = [
            {
                "candidate_id": candidate,
                "order_index": order,
                "dic_max_ratio": dic,
                "dts_max_ratio": dts,
            }
            for candidate, order, dic, dts in _CANDIDATES
        ]
        if not any(canonical_json_bytes(budget) == canonical_json_bytes(row) for row in expected_rows):
            raise ValueError("resolved budget identity is not registered")
        if step_config.dic_max_ratio != budget["dic_max_ratio"] or step_config.dts_max_ratio != budget["dts_max_ratio"]:
            raise ValueError("step config and budget identity ratios differ")
    role = run_identity["replicate_role"]
    index = run_identity["replicate_index"]
    if run_kind == "calibration_source":
        if (
            registered.name != "l12_bal"
            or steps != 2000
            or resolution != 128
            or role != "calibration"
            or index != 0
            or suite["role"] != "development"
        ):
            raise ValueError("resolved calibration-source matrix drifted")
    elif registered.name in ("v4_no_l1", "v4_no_l2"):
        if role != "appendix" or index != 0 or steps != 2000:
            raise ValueError("resolved appendix matrix drifted")
    elif (role, index) not in (("primary", 0), ("technical_repeat", 1)):
        raise ValueError("resolved experiment replicate matrix drifted")
    if payload["evaluation_thresholds"] != [0.04, 0.05, 0.06] or payload["soft_only"] is not True or payload["final_step_export_only"] is not True:
        raise ValueError("evaluation/scope constants drifted")
    return json.loads(canonical_json_bytes(payload))


def step_config_from_resolved(payload: Mapping[str, object]) -> V4StepConfig | None:
    checked = validate_resolved_config(dict(payload))
    if checked["step_config"] is None:
        return None
    return V4StepConfig(**checked["step_config"])


def canonical_trainer_argv(args: argparse.Namespace, *, resume: bool) -> tuple[str, ...]:
    _validate_namespace(args)
    _strict_bool("resume selector", resume)
    values = [
        LOCKED_INTERPRETER,
        "-m",
        "shadowforge_pipeline.soft_voxel_v4.trainer",
        "--run-kind",
        args.run_kind,
        "--device",
        args.device,
        "--output-root",
        args.output_root,
        "--run-id",
        args.run_id,
        "--stage",
        args.stage,
        "--task-id",
        args.task_id,
        "--replicate-role",
        args.replicate_role,
        "--replicate-index",
        str(args.replicate_index),
        "--steps",
        str(args.steps),
        "--learning-rate",
        "0.007",
        "--shadow-files",
        *args.shadow_files,
        "--resolution",
        str(args.resolution),
        "--export-resolution",
        str(args.export_resolution),
        "--profile",
        args.profile,
        "--suite-manifest",
        args.suite_manifest,
        "--specification",
        args.specification,
    ]
    if args.unit_coefficients is not None:
        values.extend(("--unit-coefficients", args.unit_coefficients))
    if args.candidate_registry is not None:
        values.extend(("--candidate-registry", args.candidate_registry))
    if args.candidate_id is not None:
        values.extend(("--candidate-id", args.candidate_id))
    if resume:
        values.append("--resume")
    values.extend(("--export-final-step", "--soft-only-eval"))
    return tuple(values)


def revalidate_resolved_run(resolved: ResolvedV4Run) -> None:
    if not isinstance(resolved, ResolvedV4Run):
        raise ValueError("resolved run must be a ResolvedV4Run")
    from .provenance import revalidate_snapshot

    root = Path(resolved.config["output_root_identity"]["path"])
    revalidate_snapshot(resolved.root_snapshot, root=root)
    revalidate_snapshot(resolved.suite_snapshot, root=root)
    revalidate_snapshot(resolved.specification_snapshot)
    for snapshot in resolved.shadow_snapshots:
        revalidate_snapshot(snapshot)
    if resolved.unit_coefficient_snapshot is not None:
        revalidate_snapshot(resolved.unit_coefficient_snapshot, root=root)
    if resolved.candidate_registry_snapshot is not None:
        revalidate_snapshot(resolved.candidate_registry_snapshot, root=root)
    for snapshot in resolved.source_snapshots:
        revalidate_snapshot(snapshot, root=PROJECT_ROOT)
    revalidate_snapshot(resolved.interpreter_snapshot)
    revalidate_resolved_dependencies(resolved.config)


def revalidate_resolved_dependencies(payload: Mapping[str, object]) -> None:
    """Rehash all public dependencies represented by a resolved configuration."""

    checked = validate_resolved_config(dict(payload))
    from .provenance import snapshot_regular_file

    root = Path(checked["output_root_identity"]["path"])
    root_snapshot = snapshot_regular_file(
        Path(checked["output_root_identity"]["marker_path"]), root=root
    )
    if (
        root_snapshot.sha256
        != checked["output_root_identity"]["marker_sha256"]
        or _json_from_snapshot(root_snapshot, name="v4 root marker")
        != v4_root_manifest_payload()
    ):
        raise ValueError("v4 root marker payload/identity changed")
    identities: list[tuple[dict[str, object], Path | None]] = [
        (checked["suite_manifest"], root),
        (checked["specification"], None),
        *(({"path": item["path"], "sha256": item["sha256"]}, None) for item in checked["shadow_inputs_xyz"]),
        *(({"path": str(PROJECT_ROOT / relative), "sha256": digest}, PROJECT_ROOT) for relative, digest in checked["implementation"]["files_sha256"].items()),
        (checked["interpreter"], None),
    ]
    if checked["unit_coefficient_identity"] is not None:
        unit_identity = checked["unit_coefficient_identity"]
        registry_identity = checked["candidate_registry_identity"]
        unit_snapshot = snapshot_regular_file(Path(unit_identity["path"]), root=root)
        registry_snapshot = snapshot_regular_file(Path(registry_identity["path"]), root=root)
        if unit_snapshot.sha256 != unit_identity["sha256"]:
            raise ValueError("unit coefficient dependency hash changed")
        if registry_snapshot.sha256 != registry_identity["sha256"]:
            raise ValueError("candidate registry dependency hash changed")
        dic, dts, candidate, unit_semantic, registry_semantic = _validated_unit_and_registry(
            root=root,
            unit_snapshot=unit_snapshot,
            registry_snapshot=registry_snapshot,
            candidate_id=checked["budget_identity"]["candidate_id"],
            root_identity=checked["output_root_identity"],
        )
        if (
            unit_semantic != unit_identity["semantic_sha256"]
            or registry_semantic != registry_identity["semantic_sha256"]
            or checked["step_config"]
            != {
                "dic_unit_coefficient": dic,
                "dts_unit_coefficient": dts,
                "dic_max_ratio": candidate.dic_max_ratio,
                "dts_max_ratio": candidate.dts_max_ratio,
            }
            or checked["budget_identity"] != asdict(candidate)
        ):
            raise ValueError("auxiliary transitive dependency identity changed")
    for identity, containment in identities:
        snapshot = snapshot_regular_file(Path(identity["path"]), root=containment)
        if snapshot.sha256 != identity["sha256"]:
            raise ValueError(f"dependency hash changed: {identity['path']}")
    if checked["implementation"]["bundle_sha256"] != __import__(
        "shadowforge_pipeline.soft_voxel_v4.provenance", fromlist=["canonical_hash"]
    ).canonical_hash(checked["implementation"]["files_sha256"]):
        raise ValueError("implementation bundle digest mismatch")


__all__ = [
    "BoundFileSnapshot",
    "ResolvedV4Run",
    "V4BudgetCandidate",
    "bootstrap_v4_root",
    "build_parser",
    "canonical_trainer_argv",
    "resolve_run_configuration",
    "resolved_config",
    "step_config_from_resolved",
    "v4_root_manifest_payload",
    "validate_resolved_config",
]
