"""Authenticated development funnel and formal runner for soft-loss v7."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
from typing import Mapping, Sequence

from experiments.soft_loss_v3_registry import load_bound_suite_identity
from shadowforge_pipeline.soft_voxel_v7.provenance import (
    canonical_json_bytes,
    canonical_sha256,
    decode_canonical_json,
    open_v7_root_transaction,
    revalidate_snapshot,
    snapshot_regular_file,
)
from shadowforge_pipeline.soft_voxel_v7.registry import (
    V7StageAuthorization,
    V7StageMatrix,
    build_campaign_manifest,
    build_formal_matrix_manifest,
    build_stage_authorization,
    build_training_source_bundle,
    publish_campaign_manifest,
    publish_formal_matrix_manifest,
    publish_stage_authorization,
    resolve_stage_matrix,
    validate_campaign_manifest,
    validate_formal_matrix_manifest,
    validate_stage_authorization,
    validate_training_source_bundle,
)
from shadowforge_pipeline.soft_voxel_v7.suite import (
    select_v7_formal_tasks,
    validate_v7_suite,
    write_v7_suite_bundle,
)
from shadowforge_pipeline.soft_voxel_v7.trainer import (
    CompletedRun,
    FailedRun,
    ResumableRun,
    TrainingStateError,
    project_engineering_prefix,
    project_engineering_row,
    project_metric_row,
    train_cell,
    validate_completed_run,
    validate_failed_run,
    validate_resumable_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_OUTPUT_ROOT = Path("/data/xvxinyao/pcraft_soft_loss_v7_hit_barrier_256_20260829")
APPROVED_DEVELOPMENT_SOURCE = Path(
    "/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel/"
    "runs/soft_loss_v3_20260826/manifests/development_suite.json"
)
APPROVED_DEVELOPMENT_SHA256 = "15da2ecc37424bb1513f922f5aac6b4217b547744e53099ecaaafc8243c08fe0"
APPROVED_DEVELOPMENT_SEMANTIC_SHA256 = "43c44793492084b0eb4b61d4e582bff63819af6a6d32ffc3bef072cfb987abc4"
LOCKED_INTERPRETER = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python3.10")
LOCKED_INTERPRETER_SHA256 = "7365743bea8011d4ada1d0379699045fe411de80f620c39e4c4625855f0b1772"
RUNNER_LOCK = Path("/tmp/pcraft_soft_loss_v7_hit_barrier_256_20260829.runner.lock")
REQUIRED_ENVIRONMENT = {
    "PYTHONHASHSEED": "17",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0",
    "NVIDIA_VISIBLE_DEVICES": "0",
}
DEVELOPMENT_TASKS = ("arcane_casters", "blade_heroes", "stage_dancers")
FORMAL_TASKS = tuple(f"v7_formal_{index:02d}" for index in range(1, 11))
PROFILES = ("base", "hit_only", "barrier_only", "full")
NONBASE_PROFILES = PROFILES[1:]
BUDGET_IDS = ("budget_010", "budget_015", "budget_020")
SEED = 17
SELECTOR_SEED = 20260829
GATE_SCHEMA = "pcraft_soft_loss_v7_runner_engineering_gate_v1"
EVENT_SCHEMA = "pcraft_soft_loss_v7_runner_event_v1"
EVIDENCE_SCHEMA = "pcraft_soft_loss_v7_stage_evidence_v1"
_GATE_PATHS = {
    ("development-0300", 2): Path("runs/RUNNER_GATE_development-0300_step-0002.json"),
    ("development-0300", 200): Path("runs/RUNNER_GATE_development-0300_step-0200.json"),
    ("development-1000", 2): Path("runs/RUNNER_GATE_development-1000_step-0002.json"),
    ("formal-3000", 2): Path("runs/RUNNER_GATE_formal-3000_step-0002.json"),
    ("formal-3000", 200): Path("runs/RUNNER_GATE_formal-3000_step-0200.json"),
}
_AUTH_PATHS = {
    "development-0300": Path("authorizations/development_0300.json"),
    "development-1000": Path("authorizations/development_1000.json"),
    "development-3000": Path("authorizations/development_3000.json"),
}


@dataclass(frozen=True)
class _RunState:
    kind: str
    evidence: object | None


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(nested) for nested in value]
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the registered P-CRAFT v7 Coverage-Hit/Leakage-Barrier ablation.",
        allow_abbrev=False,
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser


def _canonical_output_root(path: Path) -> Path:
    if type(path) is not type(Path()) or not path.is_absolute():
        raise ValueError("output root must be an absolute canonical path")
    canonical = Path(os.path.abspath(path))
    if path != canonical or path.is_symlink():
        raise ValueError("output root must be an absolute canonical path")
    return canonical


def _validate_runtime_authority() -> object:
    if {key: os.environ.get(key) for key in REQUIRED_ENVIRONMENT} != REQUIRED_ENVIRONMENT:
        raise RuntimeError("required runner environment identity is not exact")
    executable = Path(sys.executable).resolve()
    if executable != LOCKED_INTERPRETER:
        raise RuntimeError("runner interpreter path is not the locked executable")
    snapshot = snapshot_regular_file(LOCKED_INTERPRETER, exclusive=False)
    if snapshot.sha256 != LOCKED_INTERPRETER_SHA256 or Path(snapshot.path) != LOCKED_INTERPRETER:
        raise RuntimeError("runner interpreter bytes differ from locked authority")
    return snapshot


def _revalidate_runtime_authority(snapshot: object) -> None:
    if {key: os.environ.get(key) for key in REQUIRED_ENVIRONMENT} != REQUIRED_ENVIRONMENT:
        raise RuntimeError("runner environment changed during execution")
    if Path(sys.executable).resolve() != LOCKED_INTERPRETER:
        raise RuntimeError("runner interpreter changed during execution")
    revalidate_snapshot(snapshot)


def _validate_development_source() -> None:
    identity = load_bound_suite_identity(APPROVED_DEVELOPMENT_SOURCE, "development")
    if (
        identity.sha256 != APPROVED_DEVELOPMENT_SHA256
        or identity.semantic_sha256 != APPROVED_DEVELOPMENT_SEMANTIC_SHA256
        or tuple(task.case_id for task in identity.tasks) != DEVELOPMENT_TASKS
    ):
        raise RuntimeError("approved development source identity changed")


def _json_line(value: Mapping[str, object]) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value) + b"\n")
    sys.stdout.buffer.flush()


def _run_path(root: Path, cohort: str, task: str, budget: str | None, profile: str) -> str:
    if cohort == "formal":
        relative = Path("runs/formal") / task / profile / "seed_17"
    elif profile == "base":
        relative = Path("runs/development") / task / "base" / "seed_17"
    else:
        if budget not in BUDGET_IDS:
            raise ValueError("development nonbase path requires a registered budget")
        relative = Path("runs/development") / task / str(budget) / profile / "seed_17"
    return relative.as_posix()


def _dry_run(root: Path, device: str) -> int:
    if os.path.lexists(root):
        raise RuntimeError("dry-run output root must remain absent")
    _validate_development_source()
    formal = select_v7_formal_tasks(PROJECT_ROOT)
    if tuple(task.case_id for task in formal) != FORMAL_TASKS:
        raise RuntimeError("input-only formal selector order changed")
    _json_line({
        "schema": "pcraft_soft_loss_v7_dry_summary_v1",
        "output_root": str(root), "device": device, "seed": SEED,
        "selector_seed": SELECTOR_SEED, "development_universe_count": 30,
        "formal_count": 40, "optimizer_updates": 0, "root_was_absent": True,
    })
    for task in DEVELOPMENT_TASKS:
        rows = [(None, "base"), *[(budget, profile) for budget in BUDGET_IDS for profile in NONBASE_PROFILES]]
        for budget, profile in rows:
            _json_line({
                "schema": "pcraft_soft_loss_v7_dry_cell_v1", "cohort": "development",
                "task_id": task, "budget_id": budget, "profile": profile,
                "run_path": _run_path(root, "development", task, budget, profile),
            })
    for task in FORMAL_TASKS:
        for profile in PROFILES:
            _json_line({
                "schema": "pcraft_soft_loss_v7_dry_cell_v1", "cohort": "formal",
                "task_id": task, "budget_id": None, "profile": profile,
                "run_path": _run_path(root, "formal", task, None, profile),
            })
    stage_rows = (
        ("development-0300", 300, 10, 10, "10 physical/10 logical; arcane only; three candidate budgets"),
        ("development-1000", 1000, 21, 24, "21 physical/24 logical; conditional top-two budgets; shared Base"),
        ("development-3000", 3000, 12, 12, "12 physical/12 logical; conditional selected budget"),
        ("formal-3000", 3000, 40, 40, "40 physical/40 logical; blocked until development PASS"),
    )
    for stage, endpoint, physical, logical, note in stage_rows:
        _json_line({
            "schema": "pcraft_soft_loss_v7_dry_stage_v1", "stage": stage,
            "endpoint": endpoint, "physical_count": physical,
            "logical_count": logical, "execution_note": note,
        })
    if os.path.lexists(root):
        raise RuntimeError("dry-run unexpectedly created output root")
    return 0


@contextmanager
def _runner_lock():
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(RUNNER_LOCK, flags, 0o600)
    locked = False
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
            raise RuntimeError("runner lock file identity is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another v7 runner owns the process lock") from error
        locked = True
        named = os.stat(RUNNER_LOCK, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (info.st_dev, info.st_ino):
            raise RuntimeError("runner lock path changed after acquisition")
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _bootstrap(root: Path, *, resume: bool) -> Mapping[str, object]:
    if resume and not _exists(root):
        raise RuntimeError("--resume requires an existing v7 root")
    if not resume and _exists(root):
        raise RuntimeError("fresh execution requires the v7 root to be absent")
    if resume:
        manifests = root / "manifests"
        development = manifests / "development_suite.json"
        formal = manifests / "formal_suite.json"
        campaign_path = manifests / "campaign.json"
        if _exists(formal) and not _exists(development):
            raise RuntimeError("illegal bootstrap prefix: formal-without-development")
        if _exists(campaign_path) and (not _exists(development) or not _exists(formal)):
            raise RuntimeError("illegal bootstrap prefix: campaign-without-both-suites")
        if not _exists(campaign_path):
            write_v7_suite_bundle(
                root, development_source=APPROVED_DEVELOPMENT_SOURCE,
                project_root=PROJECT_ROOT,
            )
    else:
        write_v7_suite_bundle(
            root, development_source=APPROVED_DEVELOPMENT_SOURCE,
            project_root=PROJECT_ROOT,
        )
    development_suite = validate_v7_suite(root / "manifests/development_suite.json", expected_kind="development")
    formal_suite = validate_v7_suite(root / "manifests/formal_suite.json", expected_kind="formal")
    campaign_path = root / "manifests/campaign.json"
    if _exists(campaign_path):
        campaign = validate_campaign_manifest(campaign_path)
    else:
        source_bundle = build_training_source_bundle(PROJECT_ROOT)
        campaign = build_campaign_manifest(
            root, development_suite, formal_suite, source_bundle, device="cuda:0"
        )
        publish_campaign_manifest(root, campaign)
        campaign = validate_campaign_manifest(campaign_path)
    validate_training_source_bundle(campaign["core"]["source_bundle"], PROJECT_ROOT)
    _validate_committed_prefix(root, campaign)
    return campaign


def _validate_committed_prefix(
    root: Path, campaign: Mapping[str, object],
) -> tuple[V7StageAuthorization, ...]:
    stages = ("development-0300", "development-1000", "development-3000")
    present = [_exists(root / _AUTH_PATHS[stage]) for stage in stages]
    leading = 0
    while leading < len(present) and present[leading]:
        leading += 1
    if any(present[leading:]):
        raise RuntimeError("published development authorizations are not one contiguous prefix")
    authorizations: list[V7StageAuthorization] = []
    for index, stage in enumerate(stages[:leading]):
        authorization = validate_stage_authorization(
            root / _AUTH_PATHS[stage], expected_stage=stage, campaign=campaign
        )
        authorizations.append(authorization)
        if authorization.status == "MECHANISM_NOT_CONFIRMED" and index + 1 != leading:
            raise RuntimeError("a negative development authorization forbids a later prefix")
        if index < 2 and authorization.status not in ("ADVANCE", "MECHANISM_NOT_CONFIRMED"):
            raise RuntimeError("nonterminal development authorization status is invalid")
    formal_path = root / "manifests/formal_matrix.json"
    if _exists(formal_path):
        if len(authorizations) != 3 or authorizations[-1].status != "PASS":
            raise RuntimeError("formal matrix exists without exact development PASS")
        validate_formal_matrix_manifest(formal_path, campaign)
    return tuple(authorizations)


def _event(
    *, event: str, stage: str | None = None, task_id: str | None = None,
    profile: str | None = None, budget_id: str | None = None,
    completed_updates: int | None = None, status: str,
) -> None:
    _json_line({
        "schema": EVENT_SCHEMA, "event": event, "stage": stage,
        "task_id": task_id, "profile": profile, "budget_id": budget_id,
        "completed_updates": completed_updates, "status": status,
    })


def _discover(config: Mapping[str, object], matrix: object) -> _RunState:
    run = Path(str(config["root_path"])) / str(config["run_path"])
    if not _exists(run):
        return _RunState("absent", None)
    errors: list[BaseException] = []
    for kind, validator in (
        ("completed", validate_completed_run),
        ("failed", validate_failed_run),
        ("running", validate_resumable_run),
    ):
        try:
            return _RunState(kind, validator(config, stage_matrix=matrix))
        except (TrainingStateError, ValueError, OSError) as error:
            errors.append(error)
    raise RuntimeError(f"existing run is not an authenticated state: {run}") from errors[-1]


def _completed_updates(state: _RunState) -> int:
    if state.kind == "failed":
        return int(state.evidence.last_checkpoint.completed_updates)
    if state.kind in ("running", "completed"):
        return int(state.evidence.current_checkpoint.completed_updates)
    return 0


def _state_from_result(result: object, target: int) -> _RunState:
    if isinstance(result, FailedRun):
        return _RunState("failed", result)
    if isinstance(result, CompletedRun):
        return _RunState("completed", result)
    if isinstance(result, ResumableRun):
        return _RunState("running", result)
    # Only injected focused-test fakes can reach this seam.
    if hasattr(result, "last_checkpoint"):
        return _RunState("failed", result)
    return _RunState("completed" if target == 3000 else "running", result)


def _advance(
    config: Mapping[str, object], matrix: object, physical: Mapping[str, object], *, target: int,
) -> _RunState:
    state = _discover(config, matrix)
    if state.kind == "failed":
        _event(event="cell", stage=matrix.stage, task_id=physical["task_id"], profile=physical["profile"], budget_id=physical["budget_id"], completed_updates=_completed_updates(state), status="SKIPPED")
        return state
    current = _completed_updates(state)
    if state.kind == "completed" or current == target:
        if state.kind == "completed" and target != 3000:
            raise RuntimeError("completed cell is beyond a missing earlier orchestration gate")
        _event(event="cell", stage=matrix.stage, task_id=physical["task_id"], profile=physical["profile"], budget_id=physical["budget_id"], completed_updates=current, status="SKIPPED")
        return state
    if current > target:
        raise RuntimeError("run progressed beyond a missing gate; backfill is forbidden")
    resume = state.kind == "running"
    result = train_cell(config, stage_matrix=matrix, resume=resume, max_updates=target)
    returned = _state_from_result(result, target)
    if isinstance(result, FailedRun):
        returned = _RunState("failed", validate_failed_run(config, stage_matrix=matrix))
    elif isinstance(result, CompletedRun):
        returned = _RunState("completed", validate_completed_run(config, stage_matrix=matrix))
    elif isinstance(result, ResumableRun):
        returned = _RunState("running", validate_resumable_run(config, stage_matrix=matrix, expected_completed_updates=target))
    completed = (
        _completed_updates(returned)
        if returned.evidence is not None
        and (hasattr(returned.evidence, "current_checkpoint") or returned.kind == "failed")
        else target
    )
    status = (
        "FAILED" if returned.kind == "failed"
        else "COMPLETED" if returned.kind == "completed"
        else "RESUMED" if resume
        else "STARTED"
    )
    _event(
        event="cell", stage=matrix.stage, task_id=physical["task_id"],
        profile=physical["profile"], budget_id=physical["budget_id"],
        completed_updates=completed, status=status,
    )
    return returned


def _config_index(campaign: Mapping[str, object], *, formal: object | None = None) -> dict[str, Mapping[str, object]]:
    rows = formal.payload["formal_configs"] if formal is not None else campaign["development_configs"]
    return {str(row["config_sha256"]): row for row in rows}


def _stage_gate_rows(matrix: object, *, cap: int) -> tuple[Mapping[str, object], ...]:
    rows = tuple(matrix.physical_rows)
    if matrix.stage == "development-0300" and cap == 2:
        return rows
    if matrix.stage == "development-0300" and cap == 200:
        return tuple(row for row in rows if row["profile"] == "base" or row["profile"] == "full")
    if matrix.stage == "development-1000" and cap == 2:
        return tuple(row for row in rows if row["task_id"] != "arcane_casters")
    if matrix.stage == "formal-3000" and cap in (2, 200):
        return rows
    raise ValueError("unregistered runner engineering gate")


def _gate_payload(
    matrix: object, *, cap: int, covered_rows: Sequence[Mapping[str, object]],
    engineering_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if type(cap) is not int or len(covered_rows) != len(engineering_rows):
        raise RuntimeError("engineering gate shape is invalid")
    plain_rows = []
    for expected, raw in zip(covered_rows, engineering_rows, strict=True):
        row = dict(raw)
        if (
            row.get("schema") != "pcraft_soft_loss_v7_engineering_prefix_v1"
            or row.get("stage") != matrix.stage
            or row.get("completed_updates") != cap
            or row.get("config_sha256") != expected["config_sha256"]
            or row.get("physical_index") != expected["physical_index"]
            or row.get("checkpoint_path") != f"{expected['run_path']}/checkpoints/step_{cap:04d}.pt"
            or row.get("stage_matrix_sha256") != matrix.semantic_sha256
            or row.get("authorization_sha256") != matrix.payload["authorization_sha256"]
            or not row.get("loss_finite") or not row.get("gradient_finite")
            or not row.get("parameters_finite") or not row.get("cuda_executed")
            or row.get("clip_count") != cap or row.get("optimizer_step_count") != cap
            or not row.get("slot_valid") or not row.get("disabled_not_constructed")
            or row.get("step_time_count") != cap or not row.get("step_times_finite")
            or type(row.get("min_step_seconds")) is not float
            or not math.isfinite(row["min_step_seconds"]) or row["min_step_seconds"] <= 0.0
            or type(row.get("peak_allocated_bytes")) is not int
            or type(row.get("device_total_bytes")) is not int
            or not 0 < row["peak_allocated_bytes"] < row["device_total_bytes"]
        ):
            raise RuntimeError("engineering prefix does not satisfy exact PASS gate")
        unsigned = dict(row); semantic = unsigned.pop("semantic_sha256", None)
        if semantic != canonical_sha256(unsigned):
            raise RuntimeError("engineering prefix semantic hash mismatch")
        plain_rows.append(row)
    payload: dict[str, object] = {
        "schema": GATE_SCHEMA, "stage": matrix.stage, "cap": cap,
        "campaign_sha256": matrix.payload["campaign_sha256"],
        "stage_matrix_sha256": matrix.semantic_sha256,
        "authorization_sha256": matrix.payload["authorization_sha256"],
        "ordered_config_sha256": [row["config_sha256"] for row in covered_rows],
        "engineering_rows": plain_rows,
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload


def _gate_exists(root: Path, matrix: object, cap: int) -> bool:
    return _exists(root / _GATE_PATHS[(matrix.stage, cap)])


def _validate_dependencies(root: Path, campaign: Mapping[str, object], matrix: object) -> None:
    current = validate_campaign_manifest(root / "manifests/campaign.json")
    if canonical_json_bytes(_plain_json(current)) != canonical_json_bytes(_plain_json(campaign)):
        raise RuntimeError("campaign changed during runner orchestration")
    validate_training_source_bundle(campaign["core"]["source_bundle"], PROJECT_ROOT)
    predecessor = {
        "development-1000": "development-0300",
        "development-3000": "development-1000",
        "formal-3000": "development-3000",
    }.get(matrix.stage)
    authorization = None
    if predecessor is not None:
        authorization = validate_stage_authorization(
            root / _AUTH_PATHS[predecessor], expected_stage=predecessor, campaign=campaign
        )
        if authorization.raw_sha256 != matrix.payload["authorization_sha256"]:
            raise RuntimeError("stage predecessor authorization changed")
    rebuilt = resolve_stage_matrix(
        campaign, stage=matrix.stage, authorization=authorization
    )
    if canonical_json_bytes(_plain_json(rebuilt.payload)) != canonical_json_bytes(_plain_json(matrix.payload)):
        raise RuntimeError("stage matrix changed during runner orchestration")


def _ensure_gate(
    root: Path, campaign: Mapping[str, object], matrix: object,
    config_by_sha: Mapping[str, Mapping[str, object]], *, cap: int,
) -> bool:
    covered = _stage_gate_rows(matrix, cap=cap)
    relative = _GATE_PATHS[(matrix.stage, cap)]
    present = _gate_exists(root, matrix, cap)
    if not present:
        preflight: list[tuple[Mapping[str, object], Mapping[str, object], _RunState]] = []
        for physical in covered:
            config = config_by_sha[str(physical["config_sha256"])]
            state = _discover(config, matrix)
            preflight.append((physical, config, state))
        if any(state.kind == "failed" for _, _, state in preflight):
            return False
        for _, _, state in preflight:
            if state.kind != "absent" and _completed_updates(state) > cap:
                raise RuntimeError("missing engineering gate cannot be backfilled after later progress")
            if state.kind not in ("absent", "running"):
                raise RuntimeError("missing engineering gate has a non-resumable covered state")
        for physical, config, state in preflight:
            if state.kind != "running" or _completed_updates(state) != cap:
                state = _advance(config, matrix, physical, target=cap)
                if state.kind == "failed":
                    return False
                if state.kind != "running" or _completed_updates(state) != cap:
                    raise RuntimeError("engineering cell did not stop exactly at registered cap")
    _validate_dependencies(root, campaign, matrix)
    projections = [
        project_engineering_prefix(
            config_by_sha[str(row["config_sha256"])], stage_matrix=matrix,
            physical_index=int(row["physical_index"]), completed_updates=cap,
        )
        for row in covered
    ]
    expected = _gate_payload(matrix, cap=cap, covered_rows=covered, engineering_rows=projections)
    checkpoint_relatives = [Path(str(row["checkpoint_path"])) for row in projections]
    with open_v7_root_transaction(root) as transaction:
        transaction.ensure_registered_directory("runs")
        checkpoint_snapshots = [transaction.snapshot(path) for path in checkpoint_relatives]
        for snapshot, row in zip(checkpoint_snapshots, projections, strict=True):
            if snapshot.sha256 != row["checkpoint_sha256"]:
                raise RuntimeError("engineering checkpoint bytes differ from prefix projection")
        if present:
            snapshot = transaction.snapshot(relative)
            observed = decode_canonical_json(snapshot.payload, label="v7 runner engineering gate")
            if snapshot.payload != canonical_json_bytes(observed) or canonical_json_bytes(observed) != canonical_json_bytes(expected):
                raise RuntimeError("existing engineering gate bytes differ from reconstruction")
        else:
            transaction.publish(relative, canonical_json_bytes(expected))
            transaction.fsync(Path("runs"))
        for initial, path in zip(checkpoint_snapshots, checkpoint_relatives, strict=True):
            if transaction.snapshot(path) != initial:
                raise RuntimeError("engineering checkpoint changed across gate publication")
    _validate_dependencies(root, campaign, matrix)
    _event(event="gate", stage=matrix.stage, completed_updates=cap, status="PASS")
    return True


def _stage_evidence(
    matrix: object, engineering_rows: Sequence[Mapping[str, object]],
    metric_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    engineering = [dict(row) for row in engineering_rows]
    metrics = [dict(row) for row in metric_rows]
    payload: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA, "stage": matrix.stage,
        "campaign_sha256": matrix.payload["campaign_sha256"],
        "endpoint": matrix.endpoint,
        "physical_matrix_sha256": canonical_sha256([dict(row) for row in matrix.physical_rows]),
        "logical_matrix_sha256": canonical_sha256([dict(row) for row in matrix.logical_rows]),
        "engineering_rows": engineering, "metric_rows": metrics,
        "evidence_sha256": canonical_sha256({"engineering_rows": engineering, "metric_rows": metrics}),
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload


def _endpoint_evidence(
    matrix: object, config_by_sha: Mapping[str, Mapping[str, object]],
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    engineering = [
        project_engineering_row(
            config_by_sha[str(row["config_sha256"])], stage_matrix=matrix,
            physical_index=int(row["physical_index"]),
        )
        for row in matrix.physical_rows
    ]
    failed_budgets = {
        row["budget_id"] for row in engineering
        if row["status"] == "FAIL" and row["budget_id"] is not None
    }
    base_failed = any(row["profile"] == "base" and row["status"] == "FAIL" for row in engineering)
    any_failed = any(row["status"] == "FAIL" for row in engineering)
    metrics: list[Mapping[str, object]] = []
    if not base_failed and not (matrix.stage == "development-3000" and any_failed):
        for logical in matrix.logical_rows:
            if logical["budget_id"] in failed_budgets:
                continue
            metrics.append(
                project_metric_row(
                    config_by_sha[str(logical["config_sha256"])], stage_matrix=matrix,
                    logical_index=int(logical["logical_index"]),
                )
            )
    return engineering, metrics


def _authorization_if_present(
    root: Path, campaign: Mapping[str, object], stage: str,
) -> V7StageAuthorization | None:
    path = root / _AUTH_PATHS[stage]
    if not _exists(path):
        return None
    return validate_stage_authorization(path, expected_stage=stage, campaign=campaign)


def _run_development_stage(
    root: Path, campaign: Mapping[str, object], *, stage: str,
    predecessor: V7StageAuthorization | None,
) -> V7StageAuthorization | int:
    existing = _authorization_if_present(root, campaign, stage)
    if existing is not None:
        matrix = resolve_stage_matrix(
            campaign, stage=stage, authorization=predecessor
        )
        configs = _config_index(campaign)
        required_caps = (
            (2, 200) if stage == "development-0300"
            else (2,) if stage == "development-1000"
            else ()
        )
        for cap in required_caps:
            if not _ensure_gate(root, campaign, matrix, configs, cap=cap):
                raise RuntimeError("published authorization depends on a non-PASS engineering gate")
        engineering, metrics = _endpoint_evidence(matrix, configs)
        evidence = _stage_evidence(matrix, engineering, metrics)
        if existing.evidence is None or canonical_json_bytes(
            _plain_json(existing.evidence)
        ) != canonical_json_bytes(_plain_json(evidence)):
            raise RuntimeError("published authorization endpoint evidence differs from reconstruction")
        rebuilt = build_stage_authorization(
            campaign, stage=stage, evidence=evidence
        )
        if (
            canonical_json_bytes(_plain_json(rebuilt.evidence))
            != canonical_json_bytes(_plain_json(existing.evidence))
            or canonical_json_bytes(_plain_json(rebuilt.payload))
            != canonical_json_bytes(_plain_json(existing.payload))
            or rebuilt.raw_sha256 != existing.raw_sha256
        ):
            raise RuntimeError("published authorization bytes differ from endpoint reconstruction")
        _event(event="authorization", stage=stage, status=existing.status)
        return existing
    matrix = resolve_stage_matrix(campaign, stage=stage, authorization=predecessor)
    configs = _config_index(campaign)
    if stage == "development-0300":
        if not _ensure_gate(root, campaign, matrix, configs, cap=2):
            _event(event="stop", stage=stage, status="FAILED")
            return 4
        started = time.perf_counter()
        if not _ensure_gate(root, campaign, matrix, configs, cap=200):
            _event(event="stop", stage=stage, status="FAILED")
            return 4
        elapsed = time.perf_counter() - started
        print(f"development step-200 observed_wall_seconds={elapsed:.3f}; ETA is informational only", file=sys.stderr)
    elif stage == "development-1000":
        if not _ensure_gate(root, campaign, matrix, configs, cap=2):
            _event(event="stop", stage=stage, status="FAILED")
            return 4
    for physical in matrix.physical_rows:
        config = configs[str(physical["config_sha256"])]
        _advance(config, matrix, physical, target=matrix.endpoint)
    engineering, metrics = _endpoint_evidence(matrix, configs)
    authorization = build_stage_authorization(
        campaign, stage=stage, evidence=_stage_evidence(matrix, engineering, metrics)
    )
    publish_stage_authorization(root, authorization)
    authorization = validate_stage_authorization(
        root / _AUTH_PATHS[stage], expected_stage=stage, campaign=campaign
    )
    _event(event="authorization", stage=stage, status=authorization.status)
    return authorization


def _run_formal(
    root: Path, campaign: Mapping[str, object], authorization: V7StageAuthorization,
) -> int:
    formal_path = root / "manifests/formal_matrix.json"
    if _exists(formal_path):
        formal = validate_formal_matrix_manifest(formal_path, campaign)
        formal_status = "SKIPPED"
    else:
        formal = build_formal_matrix_manifest(campaign, authorization)
        publish_formal_matrix_manifest(root, formal)
        formal = validate_formal_matrix_manifest(formal_path, campaign)
        formal_status = "PASS"
    _event(event="formal-matrix", stage="formal-3000", status=formal_status)
    matrix = resolve_stage_matrix(campaign, stage="formal-3000", authorization=authorization)
    configs = _config_index(campaign, formal=formal)
    if not _ensure_gate(root, campaign, matrix, configs, cap=2):
        _event(event="stop", stage="formal-3000", status="FAILED")
        return 4
    started = time.perf_counter()
    if not _ensure_gate(root, campaign, matrix, configs, cap=200):
        _event(event="stop", stage="formal-3000", status="FAILED")
        return 4
    elapsed = time.perf_counter() - started
    print(f"formal step-200 observed_wall_seconds={elapsed:.3f}; ETA is informational only", file=sys.stderr)
    for physical in matrix.physical_rows:
        config = configs[str(physical["config_sha256"])]
        state = _advance(config, matrix, physical, target=3000)
        if state.kind == "failed":
            _event(event="stop", stage="formal-3000", status="FAILED")
            return 4
    for physical in matrix.physical_rows:
        config = configs[str(physical["config_sha256"])]
        completed = validate_completed_run(config, stage_matrix=matrix)
        if completed.current_checkpoint.completed_updates != 3000:
            raise RuntimeError("formal cell is not completed at update 3000")
    _event(event="complete", stage="formal-3000", completed_updates=3000, status="COMPLETED")
    return 0


def _execute(root: Path, *, resume: bool) -> int:
    campaign = _bootstrap(root, resume=resume)
    _event(event="bootstrap", status="PASS")
    predecessor: V7StageAuthorization | None = None
    for stage in ("development-0300", "development-1000", "development-3000"):
        result = _run_development_stage(
            root, campaign, stage=stage, predecessor=predecessor
        )
        if type(result) is int:
            return result
        predecessor = result
        if result.status == "MECHANISM_NOT_CONFIRMED":
            _event(event="stop", stage=stage, status="MECHANISM_NOT_CONFIRMED")
            return 3
    if predecessor is None or predecessor.status != "PASS":
        raise RuntimeError("formal execution requires exact development PASS")
    return _run_formal(root, campaign, predecessor)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        root = _canonical_output_root(args.output_root)
        authority = _validate_runtime_authority()
        if args.dry_run:
            result = _dry_run(root, args.device)
        else:
            if root != REAL_OUTPUT_ROOT:
                raise RuntimeError(f"production output root must be exactly {REAL_OUTPUT_ROOT}")
            with _runner_lock():
                result = _execute(root, resume=args.resume)
        _revalidate_runtime_authority(authority)
        return result
    except (ValueError, RuntimeError, OSError) as error:
        if argv is None:
            print(f"error: {error}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
