"""Immutable campaign registry and pure funnel decisions for soft-loss v7."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
from types import MappingProxyType

import torch
import cv2
import numpy as np

from .core import v7_learning_rate
from .protocol import registered_camera_payload
from .provenance import (
    EXPECTED_SPECIFICATION_SHA256,
    PROJECT_ROOT,
    SPECIFICATION_PATH,
    canonical_json_bytes,
    canonical_sha256,
    decode_canonical_json,
    open_v7_root_transaction,
    revalidate_snapshot,
    snapshot_regular_file,
)
from .suite import AuthenticatedV7Suite, DEVELOPMENT_IDS, FORMAL_IDS, validate_v7_suite


CAMPAIGN_SCHEMA = "pcraft_soft_loss_v7_campaign_v1"
CAMPAIGN_CORE_SCHEMA = "pcraft_soft_loss_v7_campaign_core_v1"
CONFIG_SCHEMA = "pcraft_soft_loss_v7_resolved_config_v1"
FORMAL_CORE_SCHEMA = "pcraft_soft_loss_v7_formal_core_v1"
FORMAL_MATRIX_SCHEMA = "pcraft_soft_loss_v7_formal_matrix_v1"
STAGE_MATRIX_SCHEMA = "pcraft_soft_loss_v7_stage_matrix_v1"
METRIC_ROW_SCHEMA = "pcraft_soft_loss_v7_metric_row_v1"
ENGINEERING_ROW_SCHEMA = "pcraft_soft_loss_v7_engineering_row_v1"
EVIDENCE_SCHEMA = "pcraft_soft_loss_v7_stage_evidence_v1"
AUTHORIZATION_SCHEMA = "pcraft_soft_loss_v7_stage_authorization_v1"
FAILURE_RECORD_SCHEMA = "pcraft_soft_loss_v7_failure_record_v1"
SOURCE_BUNDLE_SCHEMA = "pcraft_soft_loss_v7_training_source_bundle_v1"

DEVELOPMENT_TASKS = tuple(DEVELOPMENT_IDS)
FORMAL_TASKS = tuple(FORMAL_IDS)
PROFILES = ("base", "hit_only", "barrier_only", "full")
NONBASE_PROFILES = PROFILES[1:]
BUDGET_IDS = ("budget_010", "budget_015", "budget_020")
BUDGET_RATIOS = MappingProxyType({"budget_010": 0.10, "budget_015": 0.15, "budget_020": 0.20})
STAGES = ("development-0300", "development-1000", "development-3000", "formal-3000")
STAGE_ENDPOINTS = MappingProxyType({
    "development-0300": 300,
    "development-1000": 1000,
    "development-3000": 3000,
    "formal-3000": 3000,
})
_PREVIOUS_STAGE = {
    "development-1000": "development-0300",
    "development-3000": "development-1000",
}
_NEXT_STAGE = {
    "development-0300": "development-1000",
    "development-1000": "development-3000",
    "development-3000": None,
}
_AUTH_FILENAMES = {
    "development-0300": "development_0300.json",
    "development-1000": "development_1000.json",
    "development-3000": "development_3000.json",
}
_EVIDENCE_FILENAMES = {
    stage: name.replace(".json", ".evidence.json")
    for stage, name in _AUTH_FILENAMES.items()
}


def _authorization_chain_stages(stage: str, *, include_self: bool) -> tuple[str, ...]:
    order = ("development-0300", "development-1000", "development-3000")
    index = order.index(stage)
    return order[: index + (1 if include_self else 0)]


def _snapshot_authorization_chain(transaction: object, stages: Sequence[str]) -> dict[Path, object]:
    result = {}
    for stage in stages:
        for filename in (_AUTH_FILENAMES[stage], _EVIDENCE_FILENAMES[stage]):
            relative = Path("authorizations") / filename
            result[relative] = transaction.snapshot(relative)
    return result


def _require_unchanged_transaction_snapshots(
    transaction: object, snapshots: Mapping[Path, object], *, label: str,
) -> None:
    for relative, snapshot in snapshots.items():
        if transaction.snapshot(relative) != snapshot:
            raise ValueError(f"{label} changed during retained transaction")


def _snapshot_campaign_dependencies(
    transaction: object, campaign: Mapping[str, object], *, include_campaign: bool,
) -> dict[Path, object]:
    relatives = [
        Path("manifests/development_suite.json"),
        Path("manifests/formal_suite.json"),
    ]
    if include_campaign:
        relatives.insert(0, Path("manifests/campaign.json"))
    snapshots = {relative: transaction.snapshot(relative) for relative in relatives}
    if include_campaign and snapshots[Path("manifests/campaign.json")].sha256 != _campaign_digest(campaign):
        raise ValueError("published campaign bytes differ from registry identity")
    for kind in ("development", "formal"):
        relative = Path(f"manifests/{kind}_suite.json")
        if snapshots[relative].sha256 != campaign["core"][f"{kind}_suite"]["sha256"]:
            raise ValueError(f"published {kind} suite bytes differ from campaign")
    return snapshots
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_INVENTORY_PATH = PROJECT_ROOT / ".superpowers/sdd/2026-08-29-soft-loss-v7-hit-barrier/v3-v6-baseline.sha256"
_REQUIRED_ENVIRONMENT = {
    "PYTHONHASHSEED": "17",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0",
    "NVIDIA_VISIBLE_DEVICES": "0",
}

_SOURCE_RELATIVES = (
    "shadowforge_pipeline/__init__.py",
    "shadowforge_pipeline/soft_voxel_v7/__init__.py",
    "shadowforge_pipeline/soft_voxel_v7/core.py",
    "shadowforge_pipeline/soft_voxel_v7/protocol.py",
    "shadowforge_pipeline/soft_voxel_v7/provenance.py",
    "shadowforge_pipeline/soft_voxel_v7/suite.py",
    "shadowforge_pipeline/soft_voxel_v7/registry.py",
    "shadowforge_pipeline/soft_voxel_v7/trainer.py",
    "experiments/run_soft_loss_v7_hit_barrier_256.py",
    "experiments/soft_loss_v3_registry.py",
    "shadowforge_pipeline/soft_voxel_v3/__init__.py",
    "shadowforge_pipeline/soft_voxel_v3/contours.py",
    "shadowforge_pipeline/soft_voxel_v3/data.py",
    "shadowforge_pipeline/soft_voxel_v3/diagnostics.py",
    "shadowforge_pipeline/soft_voxel_v3/losses.py",
    "shadowforge_pipeline/soft_voxel_v3/lovasz.py",
    "shadowforge_pipeline/soft_voxel_v3/protocol.py",
    "shadowforge_pipeline/soft_voxel_v3/provenance.py",
    "shadowforge_pipeline/soft_voxel_v3/thresholding.py",
)


@dataclass(frozen=True)
class V7StageMatrix:
    stage: str
    endpoint: int
    physical_rows: tuple[Mapping[str, object], ...]
    logical_rows: tuple[Mapping[str, object], ...]
    semantic_sha256: str
    payload: Mapping[str, object]
    campaign: Mapping[str, object] | None = None
    predecessor: V7StageAuthorization | None = None


@dataclass(frozen=True)
class V7DevelopmentDecision:
    stage: str
    status: str
    ranked_budget_ids: tuple[str, ...]
    surviving_budget_ids: tuple[str, ...]
    authorized_budget_ids: tuple[str, ...]
    selected_budget_id: str | None
    next_stage: str | None


@dataclass(frozen=True)
class V7StageAuthorization:
    stage: str
    status: str
    ranked_budget_ids: tuple[str, ...]
    surviving_budget_ids: tuple[str, ...]
    selected_budget_id: str | None
    next_stage: str | None
    semantic_sha256: str
    raw_sha256: str
    payload: Mapping[str, object]
    evidence: Mapping[str, object] | None = None
    matrix: V7StageMatrix | None = None
    predecessor: V7StageAuthorization | None = None


@dataclass(frozen=True)
class V7FormalMatrixManifest:
    semantic_sha256: str
    payload: Mapping[str, object]


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise TypeError("registry JSON keys must be exact built-in strings")
            if key in result:
                raise ValueError("registry JSON contains a duplicate key")
            result[key] = _plain(nested)
        return result
    if isinstance(value, tuple):
        return [_plain(nested) for nested in value]
    if isinstance(value, list):
        return [_plain(nested) for nested in value]
    return value


def _immutable(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _immutable(nested) for key, nested in value.items()})
    if type(value) is list:
        return tuple(_immutable(nested) for nested in value)
    return value


def _exact_dict(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} has the wrong exact schema")
    return value


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an exact integer >= {minimum}")
    return value


def _finite_float(value: object, *, label: str, low: float = 0.0, high: float = 1.0) -> float:
    if type(value) is not float or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label} must be an exact finite float in [{low},{high}]")
    return value


def _decode_registry_json(raw: bytes, *, label: str) -> dict[str, object]:
    return decode_canonical_json(raw, label=label)


def expected_training_source_paths(project_root: Path) -> tuple[Path, ...]:
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be a pathlib.Path")
    return tuple(Path(relative) for relative in _SOURCE_RELATIVES)


def _canonical_project_root(project_root: Path) -> Path:
    if type(project_root) is not type(Path()):
        raise TypeError("project_root must be an exact built-in Path")
    if not project_root.is_absolute():
        raise ValueError("project_root must be absolute")
    canonical = Path(os.path.abspath(project_root))
    if canonical != project_root or project_root.is_symlink() or not project_root.is_dir():
        raise ValueError("project_root must be an existing canonical ordinary directory")
    return canonical


def build_training_source_bundle(project_root: Path) -> Mapping[str, object]:
    root = _canonical_project_root(project_root)
    files: dict[str, str] = {}
    snapshots = []
    for relative in _SOURCE_RELATIVES:
        path = root / relative
        try:
            snapshot = snapshot_regular_file(path, root=root, exclusive=True)
        except (OSError, ValueError) as error:
            raise ValueError(f"training source is missing or unsafe: {relative}") from error
        files[relative] = snapshot.sha256
        snapshots.append(snapshot)
    payload: dict[str, object] = {
        "schema": SOURCE_BUNDLE_SCHEMA,
        "project_root": str(root),
        "files_sha256": files,
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    for snapshot in snapshots:
        revalidate_snapshot(snapshot, root=root)
    return _immutable(payload)


def _monitored_module_names() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _SOURCE_RELATIVES:
        if not relative.endswith(".py"):
            continue
        name = relative[:-3].replace("/", ".")
        if name.endswith(".__init__"):
            name = name[:-9]
        result[name] = relative
    return result


def _validate_loaded_training_sources(bundle: Mapping[str, object], root: Path) -> None:
    expected = _monitored_module_names()
    files = bundle["files_sha256"]
    prefixes = (
        "shadowforge_pipeline.soft_voxel_v7",
        "shadowforge_pipeline.soft_voxel_v3",
        "experiments.run_soft_loss_v7_hit_barrier_256",
        "experiments.soft_loss_v3_registry",
    )
    snapshots = []
    for name, module in tuple(sys.modules.items()):
        if name not in expected and not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        raw = getattr(module, "__file__", None)
        if raw is None:
            raise ValueError(f"loaded monitored training module has no source file: {name}")
        try:
            path = Path(str(raw)).resolve(strict=True)
        except (OSError, ValueError) as error:
            raise ValueError(f"loaded training module source is unavailable: {name}") from error
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"loaded training module is foreign: {name}") from error
        try:
            snapshot = snapshot_regular_file(path, root=root, exclusive=True)
        except (OSError, ValueError) as error:
            raise ValueError(f"loaded training module source is unsafe: {name}") from error
        if name not in expected or relative != expected[name] or files.get(relative) != snapshot.sha256:
            raise ValueError(f"loaded project training module is outside the closure: {name}")
        snapshots.append(snapshot)
    for snapshot in snapshots:
        revalidate_snapshot(snapshot, root=root)


def validate_training_source_bundle(payload: object, project_root: Path) -> Mapping[str, object]:
    plain = _plain(payload)
    row = _exact_dict(plain, {"schema", "project_root", "files_sha256", "semantic_sha256"}, label="source bundle")
    root = _canonical_project_root(project_root)
    if row["schema"] != SOURCE_BUNDLE_SCHEMA or row["project_root"] != str(root):
        raise ValueError("source bundle schema/root mismatch")
    files = row["files_sha256"]
    if type(files) is not dict or len(files) != len(_SOURCE_RELATIVES) or set(files) != set(_SOURCE_RELATIVES):
        raise ValueError("source bundle has the wrong exact closure")
    for relative, digest in files.items():
        _digest(digest, label=f"source {relative}")
    semantic = _digest(row["semantic_sha256"], label="source semantic")
    unsigned = dict(row)
    del unsigned["semantic_sha256"]
    if canonical_sha256(unsigned) != semantic:
        raise ValueError("source bundle semantic hash mismatch")
    rebuilt = build_training_source_bundle(root)
    if canonical_json_bytes(row) != canonical_json_bytes(_plain(rebuilt)):
        raise ValueError("source bundle differs from independent reconstruction")
    if root == PROJECT_ROOT:
        _validate_loaded_training_sources(row, root)
    return _immutable(row)


def _camera_identity() -> dict[str, object]:
    payload = registered_camera_payload()
    return {
        "schema": payload["metadata"]["schema"],
        "metadata": payload["metadata"],
        "rotation_sha256": payload["rotation_sha256"],
        "translation_sha256": payload["translation_sha256"],
        "scale_sha256": payload["scale_sha256"],
        "projection_sha256": payload["projection_sha256"],
    }


def _environment(device: str, *, _test: bool = False) -> dict[str, object]:
    if type(_test) is not bool:
        raise TypeError("environment test seam must be an exact boolean")
    if device != "cuda:0":
        raise ValueError("environment device must be exactly cuda:0")
    if not _test:
        observed = {key: os.environ.get(key) for key in _REQUIRED_ENVIRONMENT}
        if observed != _REQUIRED_ENVIRONMENT:
            raise ValueError("required v7 ambient environment identity is not exact")
    else:
        observed = dict(_REQUIRED_ENVIRONMENT)
    try:
        import pytorch3d
        pytorch3d_version = str(getattr(pytorch3d, "__version__", "unknown"))
    except ImportError:
        pytorch3d_version = "unavailable"
    interpreter = snapshot_regular_file(Path(sys.executable).resolve(), exclusive=False)
    return {
        "schema": "pcraft_soft_loss_v7_environment_v1",
        "python": sys.version.split()[0],
        "interpreter": {"path": interpreter.path, "sha256": interpreter.sha256},
        "cuda_build": None if torch.version.cuda is None else str(torch.version.cuda),
        "device": device,
        "seed": 17,
        "environment": observed,
        "packages": {
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "opencv": str(cv2.__version__),
            "pytorch3d": pytorch3d_version,
        },
    }


def _scientific_policy() -> dict[str, object]:
    return {
        "resolution": 256,
        "export_resolution": 256,
        "seed": 17,
        "updates": 3000,
        "optimizer": {"name": "Adam", "initial_lr": 0.007},
        "learning_rate": {
            "updates_1_1500": 0.007,
            "updates_1501_2700": "cosine_0.007_to_0.0007_inclusive",
            "updates_2701_3000": 0.0007,
            "identity": "v7_learning_rate(completed_updates)",
        },
        "base_objective": "balanced_L2_plus_0.35_balanced_L1",
        "auxiliary_forward": "fresh_sigmoid_then_v3_render_thresholded_tau_0.05_projection",
        "coverage_hit": {"margin": 0.02, "temperature": 0.02, "top_fraction": 0.25},
        "leakage_barrier": {"margin": -0.02, "temperature": 0.02, "top_fraction": 0.05, "causal": True},
        "auxiliary_schedule": {"updates_1_100": 0.0, "updates_101_300": "linear_1_over_200_to_1", "updates_301_3000": 1.0},
        "slots": ["hit", "barrier", "hit", "barrier"],
        "gradient_guard": "project_nonnegative_base_dot_then_cap_ratio",
        "gradient_epsilon": 1e-12,
        "clip_norm": 1.0,
        "one_clip_one_step": True,
        "checkpoint_policy": {
            "forced_steps": [0, 2, 300, 1000, 3000],
            "interval": 200,
            "diagnostic_steps": [2400, 2700, 3000],
            "early_stop": False,
            "best_checkpoint": False,
        },
        "evaluator": {"views": 6, "threshold": 0.05, "task_equal": True},
    }


def _decision_policy() -> dict[str, object]:
    return {
        "budgets": [{"budget_id": key, "ratio": BUDGET_RATIOS[key]} for key in BUDGET_IDS],
        "development_tasks": list(DEVELOPMENT_TASKS),
        "formal_tasks": list(FORMAL_TASKS),
        "ranking": ["descending_min_deletion_iou", "descending_full_iou", "ascending_budget"],
        "development_1000": {"full_base_iou_min": -0.002, "single_base_iou_min": -0.003, "deletion_iou_strict_positive": True, "hit_recall_strict_positive": True, "barrier_fpr_strict_positive": True},
        "development_3000": {"deletion_iou_min": 0.010, "deletion_dice_min": 0.005, "positive_tasks": 3, "full_mean_iou_min": 0.95, "full_worst_iou_min": 0.94, "single_base_iou_dice_min": -0.001},
    }


def _suite_identity(suite: AuthenticatedV7Suite) -> dict[str, object]:
    return {"schema": "pcraft_soft_loss_v7_suite_v1", "kind": suite.kind, "path": suite.path, "sha256": suite.sha256, "semantic_sha256": suite.semantic_sha256}


def _task_record(task: object) -> dict[str, object]:
    inputs = []
    for item in task.inputs:
        inputs.append({"axis": item.axis, "source_path": item.source_path, "resolved_path": item.resolved_path, "raw_sha256": item.raw_sha256, "normalized_sha256": item.normalized_sha256})
    return {
        "task_id": task.case_id,
        "inputs": inputs,
        "registered_target_sha256": task.registered_target_sha256,
        "hull_xyz_sha256": task.hull_xyz_sha256,
        "hull_dhw_sha256": task.hull_dhw_sha256,
        "hull_occupied_count": task.hull_occupied_count,
    }


def _private_development_task_records() -> list[dict[str, object]]:
    records = []
    for index, task in enumerate(DEVELOPMENT_TASKS):
        records.append({
            "task_id": task,
            "inputs": [
                {
                    "axis": axis,
                    "source_path": f"input/{task}_{axis}.png",
                    "resolved_path": f"/test/{task}_{axis}.png",
                    "raw_sha256": hashlib.sha256(f"raw-{task}-{axis}".encode()).hexdigest(),
                    "normalized_sha256": hashlib.sha256(f"norm-{task}-{axis}".encode()).hexdigest(),
                }
                for axis in ("x", "y", "z")
            ],
            "registered_target_sha256": hashlib.sha256(f"target-{task}".encode()).hexdigest(),
            "hull_xyz_sha256": hashlib.sha256(f"xyz-{task}".encode()).hexdigest(),
            "hull_dhw_sha256": hashlib.sha256(f"dhw-{task}".encode()).hexdigest(),
            "hull_occupied_count": 1000 + index,
        })
    return records


def _private_formal_task_records() -> list[dict[str, object]]:
    records = []
    for index, task in enumerate(FORMAL_TASKS):
        records.append({
            "task_id": task,
            "inputs": [
                {
                    "axis": axis,
                    "source_path": f"formal/{task}_{axis}.png",
                    "resolved_path": f"/test/formal/{task}_{axis}.png",
                    "raw_sha256": hashlib.sha256(f"formal-raw-{task}-{axis}".encode()).hexdigest(),
                    "normalized_sha256": hashlib.sha256(f"formal-norm-{task}-{axis}".encode()).hexdigest(),
                }
                for axis in ("x", "y", "z")
            ],
            "registered_target_sha256": hashlib.sha256(f"formal-target-{task}".encode()).hexdigest(),
            "hull_xyz_sha256": hashlib.sha256(f"formal-xyz-{task}".encode()).hexdigest(),
            "hull_dhw_sha256": hashlib.sha256(f"formal-dhw-{task}".encode()).hexdigest(),
            "hull_occupied_count": 2000 + index,
        })
    return records


def _authority_snapshots() -> tuple[object, object, tuple[object, ...]]:
    specification = snapshot_regular_file(SPECIFICATION_PATH, root=PROJECT_ROOT, exclusive=True)
    if specification.sha256 != EXPECTED_SPECIFICATION_SHA256:
        raise ValueError("campaign specification differs from the approved bytes")
    inventory = snapshot_regular_file(_INVENTORY_PATH, root=PROJECT_ROOT, exclusive=True)
    try:
        lines = inventory.payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("protected inventory is not UTF-8") from error
    if len(lines) != 157:
        raise ValueError("protected inventory must contain exactly 157 entries")
    protected = []
    seen: set[str] = set()
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("protected inventory row is malformed")
        expected = _digest(line[:64], label="protected inventory digest")
        raw_path = line[66:]
        if not raw_path or raw_path in seen:
            raise ValueError("protected inventory path is empty or duplicated")
        seen.add(raw_path)
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        snapshot = snapshot_regular_file(
            path,
            root=PROJECT_ROOT if path.is_relative_to(PROJECT_ROOT) else None,
            exclusive=False,
        )
        if snapshot.sha256 != expected:
            raise ValueError(f"protected inventory entry changed: {raw_path}")
        protected.append(snapshot)
    return specification, inventory, tuple(protected)


def _revalidate_authorities(
    specification: object, inventory: object, protected: Sequence[object],
) -> None:
    revalidate_snapshot(specification, root=PROJECT_ROOT)
    revalidate_snapshot(inventory, root=PROJECT_ROOT)
    for snapshot in protected:
        path = Path(snapshot.path)
        revalidate_snapshot(
            snapshot, root=PROJECT_ROOT if path.is_relative_to(PROJECT_ROOT) else None
        )


def _profile_flags(profile: str) -> dict[str, bool]:
    return {"use_hit": profile in ("hit_only", "full"), "use_barrier": profile in ("barrier_only", "full")}


def _config(
    core: Mapping[str, object], task: Mapping[str, object], *, cohort: str,
    profile: str, budget_id: str | None, campaign_core_sha256: str,
    formal_core_sha256: str | None = None, development_authorization_sha256: str | None = None,
) -> dict[str, object]:
    if profile == "base":
        if budget_id is not None:
            raise ValueError("Base config cannot carry a budget")
        ratio = None
        run_path = f"runs/{cohort}/{task['task_id']}/base/seed_17" if cohort == "development" else f"runs/formal/{task['task_id']}/base/seed_17"
    else:
        if cohort == "development":
            if budget_id not in BUDGET_IDS:
                raise ValueError("nonbase development config requires a registered budget")
            ratio = BUDGET_RATIOS[budget_id]
            run_path = f"runs/development/{task['task_id']}/{budget_id}/{profile}/seed_17"
        else:
            if budget_id not in BUDGET_IDS or BUDGET_RATIOS[budget_id] != core["formal_budget_ratio"]:
                raise ValueError("formal nonbase config must bind the selected budget")
            ratio = BUDGET_RATIOS[budget_id]
            run_path = f"runs/formal/{task['task_id']}/{profile}/seed_17"
    payload: dict[str, object] = {
        "schema": CONFIG_SCHEMA,
        "root_path": core["root_path"],
        "campaign_path": core["campaign_path"],
        "campaign_schema": CAMPAIGN_SCHEMA,
        "campaign_core_sha256": campaign_core_sha256,
        "cohort": cohort,
        "authorization_paths": {stage: f"authorizations/{name}" for stage, name in _AUTH_FILENAMES.items()},
        "formal_matrix_path": "manifests/formal_matrix.json" if cohort == "formal" else None,
        "formal_manifest_schema": FORMAL_MATRIX_SCHEMA if cohort == "formal" else None,
        "formal_core_sha256": formal_core_sha256,
        "development_3000_authorization_sha256": development_authorization_sha256,
        "task_id": task["task_id"],
        "profile": profile,
        "flags": _profile_flags(profile),
        "budget_id": budget_id,
        "budget_ratio": ratio,
        "seed": 17,
        "run_path": run_path,
        "inputs": copy.deepcopy(task["inputs"]),
        "target_stack_sha256": task["registered_target_sha256"],
        "hull_xyz_sha256": task["hull_xyz_sha256"],
        "hull_dhw_sha256": task["hull_dhw_sha256"],
        "hull_occupied_count": task["hull_occupied_count"],
        "initialization": {"inside_logit": 2.0, "outside_logit": -4.0, "dtype": "float32", "shape": [1, 1, 256, 256, 256]},
        "cameras": copy.deepcopy(core["cameras"]),
        "scientific_policy": copy.deepcopy(core["scientific_policy"]),
        "source_bundle_sha256": core["source_bundle"]["semantic_sha256"],
        "environment": copy.deepcopy(core["environment"]),
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def _build_campaign_from_records(
    root: Path, development: dict[str, object], formal: dict[str, object],
    source_bundle: Mapping[str, object], task_records: Sequence[Mapping[str, object]], *, device: str,
    _test_mode: bool = False,
) -> Mapping[str, object]:
    if device != "cuda:0":
        raise ValueError("production device must be exactly cuda:0")
    root_text = str(Path(os.path.abspath(root)))
    spec, inventory, protected = _authority_snapshots()
    core: dict[str, object] = {
        "schema": CAMPAIGN_CORE_SCHEMA,
        "root_path": root_text,
        "campaign_path": "manifests/campaign.json",
        "device": device,
        "development_suite": development,
        "formal_suite": formal,
        "specification": {"path": spec.path, "sha256": spec.sha256},
        "protected_inventory": {"path": inventory.path, "sha256": inventory.sha256, "entry_count": 157},
        "environment": _environment(device, _test=_test_mode),
        "source_bundle": _plain(source_bundle),
        "tasks": [copy.deepcopy(dict(row)) for row in task_records],
        "profiles": list(PROFILES),
        "scientific_policy": _scientific_policy(),
        "decision_policy": _decision_policy(),
        "cameras": _camera_identity(),
    }
    core_sha = canonical_sha256(core)
    configs = []
    for task in task_records:
        configs.append(_config(core, task, cohort="development", profile="base", budget_id=None, campaign_core_sha256=core_sha))
        for budget in BUDGET_IDS:
            for profile in NONBASE_PROFILES:
                configs.append(_config(core, task, cohort="development", profile=profile, budget_id=budget, campaign_core_sha256=core_sha))
    payload: dict[str, object] = {
        "schema": CAMPAIGN_SCHEMA,
        "core": core,
        "campaign_core_sha256": core_sha,
        "development_configs": configs,
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    _revalidate_authorities(spec, inventory, protected)
    return _immutable(payload)


def build_campaign_manifest(
    root: Path, development_suite: AuthenticatedV7Suite, formal_suite: AuthenticatedV7Suite,
    source_bundle: Mapping[str, object], *, device: str,
) -> Mapping[str, object]:
    if type(root) is not type(Path()) or not root.is_absolute() or root != Path(os.path.abspath(root)):
        raise ValueError("campaign root must be an absolute canonical Path")
    if device != "cuda:0":
        raise ValueError("production device must be exactly cuda:0")
    if not isinstance(development_suite, AuthenticatedV7Suite) or not isinstance(formal_suite, AuthenticatedV7Suite):
        raise TypeError("campaign suites must be authenticated Task 2 suites")
    current_development = validate_v7_suite(Path(development_suite.path), expected_kind="development")
    current_formal = validate_v7_suite(Path(formal_suite.path), expected_kind="formal")
    if current_development != development_suite or current_formal != formal_suite:
        raise ValueError("campaign suite identities changed")
    if Path(development_suite.path).parents[1] != root or Path(formal_suite.path).parents[1] != root:
        raise ValueError("campaign suites must belong to the same registered root")
    validated_bundle = validate_training_source_bundle(source_bundle, PROJECT_ROOT)
    records = [_task_record(task) for task in development_suite.tasks]
    if tuple(row["task_id"] for row in records) != DEVELOPMENT_TASKS:
        raise ValueError("development task order changed")
    result = _build_campaign_from_records(root, _suite_identity(development_suite), _suite_identity(formal_suite), validated_bundle, records, device=device)
    validate_v7_suite(Path(development_suite.path), expected_kind="development")
    validate_v7_suite(Path(formal_suite.path), expected_kind="formal")
    validate_training_source_bundle(validated_bundle, PROJECT_ROOT)
    return result


def _build_campaign_manifest_for_test(root: Path, *, source_bundle: Mapping[str, object], device: str) -> Mapping[str, object]:
    if device != "cuda:0":
        raise ValueError("production device must be exactly cuda:0")
    if not source_bundle:
        raise ValueError("source bundle is required")
    records = _private_development_task_records()
    suite = lambda kind: {"schema": "pcraft_soft_loss_v7_suite_v1", "kind": kind, "path": f"{root}/manifests/{kind}_suite.json", "sha256": hashlib.sha256(kind.encode()).hexdigest(), "semantic_sha256": hashlib.sha256((kind + "-semantic").encode()).hexdigest()}
    return _build_campaign_from_records(Path(root), suite("development"), suite("formal"), source_bundle, records, device=device, _test_mode=True)


def _validate_config(config: object, campaign: Mapping[str, object]) -> dict[str, object]:
    claimed = _plain(config)
    if type(claimed) is not dict:
        raise ValueError("resolved config must be an exact object")
    core = campaign["core"]
    task_matches = [row for row in core["tasks"] if row["task_id"] == claimed.get("task_id")]
    if len(task_matches) != 1:
        raise ValueError("resolved config task is not registered")
    try:
        rebuilt = _config(
            core,
            task_matches[0],
            cohort="development",
            profile=claimed["profile"],
            budget_id=claimed["budget_id"],
            campaign_core_sha256=campaign["campaign_core_sha256"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("resolved config has the wrong exact schema") from error
    if canonical_json_bytes(claimed) != canonical_json_bytes(rebuilt):
        raise ValueError("resolved config differs from the immutable campaign registry")
    unsigned = dict(claimed)
    digest = unsigned.pop("config_sha256", None)
    if _digest(digest, label="config sha") != canonical_sha256(unsigned):
        raise ValueError("resolved config SHA mismatch")
    return claimed


def rebuild_resolved_config(claimed: Mapping[str, object], campaign: Mapping[str, object]) -> Mapping[str, object]:
    validated_campaign = _validate_campaign_payload(campaign)
    plain = _plain(claimed)
    if type(plain) is not dict:
        raise ValueError("resolved config must be an exact object")
    if plain.get("cohort") == "development":
        validated = _validate_config(plain, validated_campaign)
    elif plain.get("cohort") == "formal":
        root = Path(validated_campaign["core"]["root_path"])
        formal = validate_formal_matrix_manifest(
            root / "manifests/formal_matrix.json", validated_campaign
        )
        matches = [
            _plain(row) for row in formal.payload["formal_configs"]
            if row["config_sha256"] == plain.get("config_sha256")
        ]
        if len(matches) != 1 or canonical_json_bytes(matches[0]) != canonical_json_bytes(plain):
            raise ValueError("formal config is not an exact member of authenticated manifest")
        unsigned = dict(plain)
        digest = unsigned.pop("config_sha256", None)
        if _digest(digest, label="formal config sha") != canonical_sha256(unsigned):
            raise ValueError("formal resolved config SHA mismatch")
        validated = plain
    else:
        raise ValueError("resolved config cohort is invalid")
    return _immutable(validated)


def revalidate_static_identities(config: Mapping[str, object]) -> None:
    claimed = _plain(config)
    if type(claimed) is not dict:
        raise ValueError("resolved config must be an exact object")
    root = Path(claimed["root_path"])
    campaign_path = root / claimed["campaign_path"]
    campaign = validate_campaign_manifest(campaign_path)
    rebuild_resolved_config(claimed, campaign)
    validate_training_source_bundle(
        campaign["core"]["source_bundle"],
        Path(campaign["core"]["source_bundle"]["project_root"]),
    )
    for item in claimed["inputs"]:
        snapshot = snapshot_regular_file(Path(item["resolved_path"]), exclusive=False)
        if snapshot.sha256 != item["raw_sha256"]:
            raise ValueError("registered input bytes changed")


def _validate_campaign_payload(payload: object) -> dict[str, object]:
    row = _exact_dict(_plain(payload), {"schema", "core", "campaign_core_sha256", "development_configs", "semantic_sha256"}, label="campaign")
    if row["schema"] != CAMPAIGN_SCHEMA:
        raise ValueError("campaign schema mismatch")
    core = _exact_dict(
        row["core"],
        {
            "schema", "root_path", "campaign_path", "device",
            "development_suite", "formal_suite", "specification",
            "protected_inventory", "environment", "source_bundle", "tasks",
            "profiles", "scientific_policy", "decision_policy", "cameras",
        },
        label="campaign core",
    )
    if core.get("schema") != CAMPAIGN_CORE_SCHEMA or core.get("device") != "cuda:0":
        raise ValueError("campaign core schema/device mismatch")
    source_bundle = core["source_bundle"]
    if type(source_bundle) is not dict or type(source_bundle.get("project_root")) is not str:
        raise ValueError("campaign source bundle identity is invalid")
    source_root = Path(source_bundle["project_root"])
    private_seam = source_root != PROJECT_ROOT
    if (
        type(core["root_path"]) is not str
        or not Path(core["root_path"]).is_absolute()
        or core["root_path"] != os.path.abspath(core["root_path"])
        or core["campaign_path"] != "manifests/campaign.json"
        or core["profiles"] != list(PROFILES)
        or core["scientific_policy"] != _scientific_policy()
        or core["decision_policy"] != _decision_policy()
        or core["cameras"] != _camera_identity()
        or core["environment"] != _environment("cuda:0", _test=private_seam)
    ):
        raise ValueError("campaign core differs from frozen policy identities")
    for kind in ("development", "formal"):
        suite = _exact_dict(
            core[f"{kind}_suite"],
            {"schema", "kind", "path", "sha256", "semantic_sha256"},
            label=f"campaign {kind} suite",
        )
        if suite["schema"] != "pcraft_soft_loss_v7_suite_v1" or suite["kind"] != kind:
            raise ValueError("campaign suite identity mismatch")
        if suite["path"] != str(
            Path(core["root_path"]) / "manifests" / f"{kind}_suite.json"
        ):
            raise ValueError("campaign suite path is not canonical for its root")
        _digest(suite["sha256"], label="suite sha")
        _digest(suite["semantic_sha256"], label="suite semantic")
    specification, inventory, protected = _authority_snapshots()
    if core["specification"] != {"path": specification.path, "sha256": specification.sha256}:
        raise ValueError("campaign specification identity is not authoritative")
    if core["protected_inventory"] != {
        "path": inventory.path, "sha256": inventory.sha256, "entry_count": 157
    }:
        raise ValueError("campaign protected inventory identity is not authoritative")
    tasks = core["tasks"]
    if type(tasks) is not list or len(tasks) != 3 or tuple(task.get("task_id") for task in tasks if type(task) is dict) != DEVELOPMENT_TASKS:
        raise ValueError("campaign task registry/order mismatch")
    for task in tasks:
        task_row = _exact_dict(
            task,
            {"task_id", "inputs", "registered_target_sha256", "hull_xyz_sha256", "hull_dhw_sha256", "hull_occupied_count"},
            label="campaign task",
        )
        if type(task_row["inputs"]) is not list or len(task_row["inputs"]) != 3:
            raise ValueError("campaign task must contain three inputs")
        for axis, item in zip(("x", "y", "z"), task_row["inputs"], strict=True):
            input_row = _exact_dict(item, {"axis", "source_path", "resolved_path", "raw_sha256", "normalized_sha256"}, label="campaign input")
            if input_row["axis"] != axis or any(type(input_row[key]) is not str for key in ("source_path", "resolved_path")):
                raise ValueError("campaign input identity mismatch")
            _digest(input_row["raw_sha256"], label="raw input sha")
            _digest(input_row["normalized_sha256"], label="normalized input sha")
        for key in ("registered_target_sha256", "hull_xyz_sha256", "hull_dhw_sha256"):
            _digest(task_row[key], label=key)
        _exact_int(task_row["hull_occupied_count"], label="hull occupied", minimum=1)
    validate_training_source_bundle(core["source_bundle"], source_root)
    if private_seam:
        expected_tasks = _private_development_task_records()
    else:
        suite = validate_v7_suite(Path(core["development_suite"]["path"]), expected_kind="development")
        if _suite_identity(suite) != core["development_suite"]:
            raise ValueError("campaign development suite identity changed")
        expected_tasks = [_task_record(task) for task in suite.tasks]
    if canonical_json_bytes(tasks) != canonical_json_bytes(expected_tasks):
        raise ValueError("campaign task records differ from authenticated suite")
    if canonical_sha256(core) != _digest(row["campaign_core_sha256"], label="campaign core sha"):
        raise ValueError("campaign core SHA mismatch")
    unsigned = dict(row)
    terminal = unsigned.pop("semantic_sha256", None)
    if canonical_sha256(unsigned) != _digest(terminal, label="campaign semantic"):
        raise ValueError("campaign semantic SHA mismatch")
    configs = row["development_configs"]
    if type(configs) is not list or len(configs) != 30:
        raise ValueError("campaign must contain exactly 30 development configs")
    expected = [(task, None, "base") for task in DEVELOPMENT_TASKS]
    expected = []
    for task in DEVELOPMENT_TASKS:
        expected.append((task, None, "base"))
        expected.extend((task, budget, profile) for budget in BUDGET_IDS for profile in NONBASE_PROFILES)
    if [(item.get("task_id"), item.get("budget_id"), item.get("profile")) if type(item) is dict else None for item in configs] != expected:
        raise ValueError("development config universe/order mismatch")
    for config in configs:
        _validate_config(config, row)
    _revalidate_authorities(specification, inventory, protected)
    return row


def publish_campaign_manifest(root: Path, payload: Mapping[str, object]) -> None:
    campaign = _validate_campaign_payload(payload)
    if root != Path(campaign["core"]["root_path"]):
        raise ValueError("campaign root argument differs from campaign identity")
    expected_development = campaign["core"]["development_suite"]
    expected_formal = campaign["core"]["formal_suite"]
    development = validate_v7_suite(Path(expected_development["path"]), expected_kind="development")
    formal = validate_v7_suite(Path(expected_formal["path"]), expected_kind="formal")
    if _suite_identity(development) != expected_development or _suite_identity(formal) != expected_formal:
        raise ValueError("campaign suite dependency changed before publication")
    source_root = Path(campaign["core"]["source_bundle"]["project_root"])
    specification, inventory, protected = _authority_snapshots()
    with open_v7_root_transaction(root) as transaction:
        transaction.ensure_registered_directory("manifests")
        transaction.revalidate_root_marker()
        dependency_snapshots = _snapshot_campaign_dependencies(
            transaction, campaign, include_campaign=False
        )
        validate_training_source_bundle(campaign["core"]["source_bundle"], source_root)
        transaction.publish(Path("manifests/campaign.json"), canonical_json_bytes(campaign))
        transaction.fsync(Path("manifests"))
        _require_unchanged_transaction_snapshots(
            transaction, dependency_snapshots, label="campaign suite dependencies"
        )
        validate_training_source_bundle(campaign["core"]["source_bundle"], source_root)
        _revalidate_authorities(specification, inventory, protected)
        transaction.revalidate_root_marker()
    if _suite_identity(validate_v7_suite(Path(expected_development["path"]), expected_kind="development")) != expected_development or _suite_identity(validate_v7_suite(Path(expected_formal["path"]), expected_kind="formal")) != expected_formal:
        raise ValueError("campaign suite dependency changed after publication")
    _revalidate_authorities(specification, inventory, protected)


def validate_campaign_manifest(path: Path) -> Mapping[str, object]:
    if type(path) is not type(Path()) or path.name != "campaign.json" or path.parent.name != "manifests":
        raise ValueError("campaign path must be canonical manifests/campaign.json")
    root = path.parent.parent
    initial = snapshot_regular_file(path, root=root, exclusive=True)
    payload = _decode_registry_json(initial.payload, label="v7 campaign")
    if initial.payload != canonical_json_bytes(payload):
        raise ValueError("campaign is not canonical JSON")
    validated = _validate_campaign_payload(payload)
    if validated["core"]["root_path"] != str(root):
        raise ValueError("campaign root identity mismatch")
    with open_v7_root_transaction(root) as transaction:
        snapshot = transaction.snapshot(Path("manifests/campaign.json"))
        if snapshot != initial:
            raise ValueError("campaign identity changed before retained validation")
        for kind in ("development", "formal"):
            expected = validated["core"][f"{kind}_suite"]
            if transaction.snapshot(Path(f"manifests/{kind}_suite.json")).sha256 != expected["sha256"]:
                raise ValueError("campaign suite file identity changed")
        validate_training_source_bundle(
            validated["core"]["source_bundle"],
            Path(validated["core"]["source_bundle"]["project_root"]),
        )
        transaction.revalidate_root_marker()
    for kind in ("development", "formal"):
        expected = validated["core"][f"{kind}_suite"]
        observed = validate_v7_suite(Path(expected["path"]), expected_kind=kind)
        if _suite_identity(observed) != expected:
            raise ValueError("campaign suite semantic dependency changed")
    return _immutable(validated)


def _campaign_digest(campaign: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(_plain(campaign))).hexdigest()


def _config_index(campaign: Mapping[str, object]) -> dict[tuple[str, str | None, str], Mapping[str, object]]:
    return {(row["task_id"], row["budget_id"], row["profile"]): row for row in campaign["development_configs"]}


def _physical_row(index: int, config: Mapping[str, object]) -> dict[str, object]:
    return {"physical_index": index, "task_id": config["task_id"], "budget_id": config["budget_id"], "profile": config["profile"], "config_sha256": config["config_sha256"], "run_path": config["run_path"]}


def _logical_row(index: int, physical: Mapping[str, object], *, budget_id: str | None = None) -> dict[str, object]:
    return {"logical_index": index, "physical_index": physical["physical_index"], "task_id": physical["task_id"], "budget_id": physical["budget_id"] if budget_id is None else budget_id, "profile": physical["profile"], "config_sha256": physical["config_sha256"], "run_path": physical["run_path"]}


def _authorization_payload(value: object) -> dict[str, object]:
    if isinstance(value, V7StageAuthorization):
        return _plain(value.payload)
    if isinstance(value, Mapping):
        return _plain(value)
    raise ValueError("authorization has the wrong type")


def resolve_stage_matrix(campaign: Mapping[str, object], *, stage: str, authorization: object = None) -> V7StageMatrix:
    campaign_row = _validate_campaign_payload(campaign)
    return _resolve_stage_matrix_validated(
        campaign_row, stage=stage, authorization=authorization
    )


def _resolve_stage_matrix_validated(
    campaign_row: Mapping[str, object], *, stage: str, authorization: object = None,
) -> V7StageMatrix:
    if type(stage) is not str or stage not in STAGES:
        raise ValueError("unknown stage")
    if stage == "formal-3000":
        auth = _validate_authorization(
            authorization, expected_stage="development-3000", campaign=campaign_row
        )
        if auth["status"] != "PASS":
            raise ValueError("formal work requires development-3000 PASS")
        root = Path(campaign_row["core"]["root_path"])
        formal = validate_formal_matrix_manifest(
            root / "manifests/formal_matrix.json", campaign_row
        )
        return _formal_stage_matrix(campaign_row, auth, formal)
    index = _config_index(campaign_row)
    physical: list[dict[str, object]] = []
    logical: list[dict[str, object]] = []
    auth_path = None
    auth_sha = None
    if stage == "development-0300":
        if authorization is not None:
            raise ValueError("stage 1 must not carry authorization")
        task = DEVELOPMENT_TASKS[0]
        physical.append(_physical_row(0, index[(task, None, "base")]))
        for budget in BUDGET_IDS:
            for profile in NONBASE_PROFILES:
                physical.append(_physical_row(len(physical), index[(task, budget, profile)]))
        logical = [_logical_row(i, row) for i, row in enumerate(physical)]
    else:
        auth = _validate_authorization(
            authorization, expected_stage=_PREVIOUS_STAGE[stage], campaign=campaign_row
        )
        expected_previous = _PREVIOUS_STAGE[stage]
        if auth["status"] != "ADVANCE" or auth["next_stage"] != stage:
            raise ValueError("authorization does not grant this stage")
        auth_path = f"authorizations/{_AUTH_FILENAMES[expected_previous]}"
        auth_sha = hashlib.sha256(canonical_json_bytes(auth)).hexdigest()
        budgets = tuple(auth["ranked_budget_ids"][:2]) if stage == "development-1000" else (auth["selected_budget_id"],)
        if stage == "development-1000":
            for task in DEVELOPMENT_TASKS:
                base = _physical_row(len(physical), index[(task, None, "base")])
                physical.append(base)
                for budget in budgets:
                    for profile in NONBASE_PROFILES:
                        physical.append(_physical_row(len(physical), index[(task, budget, profile)]))
            physical_lookup = {(r["task_id"], r["budget_id"], r["profile"]): r for r in physical}
            base_lookup = {(r["task_id"], r["profile"]): r for r in physical if r["profile"] == "base"}
            for budget in budgets:
                for task in DEVELOPMENT_TASKS:
                    for profile in PROFILES:
                        row = base_lookup[(task, "base")] if profile == "base" else physical_lookup[(task, budget, profile)]
                        logical.append(_logical_row(len(logical), row, budget_id=budget))
        else:
            budget = budgets[0]
            if budget not in BUDGET_IDS:
                raise ValueError("selected budget is invalid")
            for task in DEVELOPMENT_TASKS:
                physical.append(_physical_row(len(physical), index[(task, None, "base")]))
                for profile in NONBASE_PROFILES:
                    physical.append(_physical_row(len(physical), index[(task, budget, profile)]))
            logical = [_logical_row(i, row, budget_id=budget if row["profile"] == "base" else None) for i, row in enumerate(physical)]
    payload: dict[str, object] = {
        "schema": STAGE_MATRIX_SCHEMA,
        "stage": stage,
        "campaign_sha256": _campaign_digest(campaign_row),
        "endpoint": STAGE_ENDPOINTS[stage],
        "authorization_path": auth_path,
        "authorization_sha256": auth_sha,
        "physical_rows": physical,
        "logical_rows": logical,
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    predecessor = authorization if isinstance(authorization, V7StageAuthorization) else None
    return V7StageMatrix(
        stage, STAGE_ENDPOINTS[stage], tuple(_immutable(row) for row in physical),
        tuple(_immutable(row) for row in logical), payload["semantic_sha256"],
        _immutable(payload), _immutable(campaign_row), predecessor,
    )


_METRIC_KEYS = {"schema", "stage", "logical_index", "task_id", "budget_id", "profile", "config_sha256", "run_path", "completed_updates", "checkpoint_sha256", "metrics_path", "metrics_sha256", "iou", "dice", "recall", "fpr", "semantic_sha256"}
_ENGINEERING_KEYS = {"schema", "stage", "physical_index", "task_id", "budget_id", "profile", "config_sha256", "run_path", "completed_updates", "checkpoint_sha256", "status", "failure_code", "device", "loss_finite", "gradient_finite", "parameters_finite", "cuda_executed", "clip_count", "optimizer_step_count", "slot_valid", "disabled_not_constructed", "step_time_count", "step_times_finite", "min_step_seconds", "peak_allocated_bytes", "device_total_bytes", "hit_eligible_slots", "hit_active_slots", "barrier_eligible_slots", "barrier_active_slots", "failure_record_path", "failure_record_sha256", "semantic_sha256"}
_FAILURE_KEYS = {
    "schema", "config_sha256", "stage_matrix_sha256", "authorization_sha256",
    "last_checkpoint_path", "last_checkpoint_sha256", "completed_updates",
    "failure_code", "training_source_bundle_sha256", "semantic_sha256",
}


def _validate_metric_row(value: object, expected: Mapping[str, object], matrix: V7StageMatrix) -> dict[str, object]:
    row = _exact_dict(_plain(value), _METRIC_KEYS, label="metric row")
    if row["schema"] != METRIC_ROW_SCHEMA or row["stage"] != matrix.stage:
        raise ValueError("metric row schema/stage mismatch")
    for key in ("logical_index", "completed_updates"):
        _exact_int(row[key], label=key)
    for key in ("checkpoint_sha256", "metrics_sha256", "semantic_sha256"):
        _digest(row[key], label=key)
    for key in ("iou", "dice", "recall", "fpr"):
        _finite_float(row[key], label=key)
    for key in ("logical_index", "task_id", "budget_id", "profile", "config_sha256", "run_path"):
        if row[key] != expected[key]:
            raise ValueError("metric row differs from the authorized logical matrix")
    if row["completed_updates"] != matrix.endpoint or row["metrics_path"] != f"{row['run_path']}/artifacts/metrics/step_{matrix.endpoint:04d}.json":
        raise ValueError("metric row endpoint/path mismatch")
    unsigned = dict(row); terminal = unsigned.pop("semantic_sha256")
    if canonical_sha256(unsigned) != terminal:
        raise ValueError("metric row semantic SHA mismatch")
    return row


def _eligible_slots(completed_updates: int) -> tuple[int, int]:
    active_updates = max(0, completed_updates - 100)
    return (active_updates + 1) // 2, active_updates // 2


def _derived_failure_code(row: Mapping[str, object], expected: Mapping[str, object], completed_updates: int) -> str:
    if not row["loss_finite"] or not row["gradient_finite"] or not row["parameters_finite"]:
        return "NONFINITE_TRAINING"
    profile = expected["profile"]
    enabled_hit = profile in ("hit_only", "full")
    enabled_barrier = profile in ("barrier_only", "full")
    hit_slots, barrier_slots = _eligible_slots(completed_updates)
    expected_hit = hit_slots if enabled_hit else 0
    expected_barrier = barrier_slots if enabled_barrier else 0
    step_contract = (
        row["clip_count"] != completed_updates
        or row["optimizer_step_count"] != completed_updates
        or row["step_time_count"] != completed_updates or not row["slot_valid"]
        or not row["disabled_not_constructed"]
        or row["hit_eligible_slots"] != expected_hit
        or row["barrier_eligible_slots"] != expected_barrier
        or row["hit_active_slots"] > row["hit_eligible_slots"]
        or row["barrier_active_slots"] > row["barrier_eligible_slots"]
    )
    if step_contract:
        return "STEP_CONTRACT"
    if not row["cuda_executed"]:
        return "CUDA_NOT_USED"
    if not row["step_times_finite"] or row["min_step_seconds"] is None or row["min_step_seconds"] <= 0.0 or not (0 < row["peak_allocated_bytes"] < row["device_total_bytes"]):
        return "RESOURCE_INVALID"
    return "NONE"


def _validate_engineering_row(value: object, expected: Mapping[str, object], matrix: V7StageMatrix) -> dict[str, object]:
    row = _exact_dict(_plain(value), _ENGINEERING_KEYS, label="engineering row")
    if row["schema"] != ENGINEERING_ROW_SCHEMA or row["stage"] != matrix.stage or row["device"] != "cuda:0":
        raise ValueError("engineering row schema/stage/device mismatch")
    for key in ("physical_index", "completed_updates", "clip_count", "optimizer_step_count", "step_time_count", "peak_allocated_bytes", "device_total_bytes", "hit_eligible_slots", "hit_active_slots", "barrier_eligible_slots", "barrier_active_slots"):
        _exact_int(row[key], label=key)
    for key in ("loss_finite", "gradient_finite", "parameters_finite", "cuda_executed", "slot_valid", "disabled_not_constructed", "step_times_finite"):
        if type(row[key]) is not bool:
            raise ValueError(f"{key} must be an exact boolean")
    if row["min_step_seconds"] is not None and (type(row["min_step_seconds"]) is not float or not math.isfinite(row["min_step_seconds"])):
        raise ValueError("min_step_seconds must be finite float or null")
    if ((row["step_time_count"] == 0 or not row["step_times_finite"]) != (row["min_step_seconds"] is None)):
        raise ValueError("minimum step time nullability differs from observed timing evidence")
    for key in ("checkpoint_sha256", "semantic_sha256"):
        _digest(row[key], label=key)
    for key in ("physical_index", "task_id", "budget_id", "profile", "config_sha256", "run_path"):
        if row[key] != expected[key]:
            raise ValueError("engineering row differs from physical matrix")
    if row["completed_updates"] > matrix.endpoint:
        raise ValueError("engineering completed prefix exceeds authorized endpoint")
    derived = _derived_failure_code(row, expected, row["completed_updates"])
    expected_status = "PASS" if derived == "NONE" else "FAIL"
    if row["status"] != expected_status or row["failure_code"] != derived:
        raise ValueError("engineering status/failure code is not derived from evidence")
    if expected_status == "PASS":
        if row["completed_updates"] != matrix.endpoint:
            raise ValueError("PASS engineering evidence requires the authorized endpoint")
        if row["failure_record_path"] is not None or row["failure_record_sha256"] is not None:
            raise ValueError("PASS engineering row cannot have a failure record")
    else:
        if row["failure_record_path"] != f"{row['run_path']}/failure/failure.json":
            raise ValueError("FAIL engineering row has wrong failure record path")
        _digest(row["failure_record_sha256"], label="failure record sha")
    unsigned = dict(row); terminal = unsigned.pop("semantic_sha256")
    if canonical_sha256(unsigned) != terminal:
        raise ValueError("engineering row semantic SHA mismatch")
    return row


def _validate_failure_record(row: Mapping[str, object], matrix: V7StageMatrix) -> None:
    if matrix.campaign is None:
        return
    campaign = _plain(matrix.campaign)
    root = Path(campaign["core"]["root_path"])
    failure_relative = Path(str(row["failure_record_path"]))
    checkpoint_relative = Path(
        f"{row['run_path']}/checkpoints/step_{row['completed_updates']:04d}.pt"
    )
    failure_snapshot = snapshot_regular_file(
        root / failure_relative, root=root, exclusive=True
    )
    if failure_snapshot.sha256 != row["failure_record_sha256"]:
        raise ValueError("failure record raw digest differs from engineering row")
    failure = _exact_dict(
        _decode_registry_json(failure_snapshot.payload, label="v7 failure record"),
        _FAILURE_KEYS, label="failure record",
    )
    if failure_snapshot.payload != canonical_json_bytes(failure):
        raise ValueError("failure record is not canonical JSON")
    expected = {
        "schema": FAILURE_RECORD_SCHEMA,
        "config_sha256": row["config_sha256"],
        "stage_matrix_sha256": matrix.semantic_sha256,
        "authorization_sha256": matrix.payload["authorization_sha256"],
        "last_checkpoint_path": checkpoint_relative.as_posix(),
        "last_checkpoint_sha256": row["checkpoint_sha256"],
        "completed_updates": row["completed_updates"],
        "failure_code": row["failure_code"],
        "training_source_bundle_sha256": campaign["core"]["source_bundle"]["semantic_sha256"],
    }
    expected["semantic_sha256"] = canonical_sha256(expected)
    if canonical_json_bytes(failure) != canonical_json_bytes(expected):
        raise ValueError("failure record disagrees with engineering evidence")
    checkpoint = snapshot_regular_file(
        root / checkpoint_relative, root=root, exclusive=True
    )
    if checkpoint.sha256 != row["checkpoint_sha256"]:
        raise ValueError("failure last checkpoint digest differs")
    revalidate_snapshot(failure_snapshot, root=root)
    revalidate_snapshot(checkpoint, root=root)


def _validate_evidence_rows(matrix: V7StageMatrix, metric_rows: Sequence[object], engineering_rows: Sequence[object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if type(metric_rows) not in (list, tuple) or type(engineering_rows) not in (list, tuple):
        raise ValueError("evidence rows must be exact lists/tuples")
    if len(engineering_rows) != len(matrix.physical_rows):
        raise ValueError("engineering evidence must cover every physical row")
    engineering = [_validate_engineering_row(row, matrix.physical_rows[i], matrix) for i, row in enumerate(engineering_rows)]
    for row in engineering:
        if row["status"] == "FAIL":
            _validate_failure_record(row, matrix)
    failed_budget = {row["budget_id"] for row in engineering if row["status"] == "FAIL" and row["budget_id"] is not None}
    base_failed = any(row["profile"] == "base" and row["status"] == "FAIL" for row in engineering)
    if matrix.stage == "development-3000" and any(row["status"] == "FAIL" for row in engineering):
        expected_logical = []
    elif base_failed:
        expected_logical = []
    else:
        expected_logical = [row for row in matrix.logical_rows if row["budget_id"] not in failed_budget]
    if len(metric_rows) != len(expected_logical):
        raise ValueError("metric evidence shape does not match engineering-surviving units")
    metrics = [_validate_metric_row(row, expected_logical[i], matrix) for i, row in enumerate(metric_rows)]
    engineering_by_physical = {row["physical_index"]: row for row in engineering}
    logical_by_index = {row["logical_index"]: row for row in matrix.logical_rows}
    for row in metrics:
        physical_index = logical_by_index[row["logical_index"]]["physical_index"]
        if row["checkpoint_sha256"] != engineering_by_physical[physical_index]["checkpoint_sha256"]:
            raise ValueError("metric and engineering rows disagree on checkpoint SHA")
    if matrix.stage == "development-1000":
        grouped: dict[tuple[str, int], dict[str, object]] = {}
        for row in metrics:
            if row["profile"] != "base":
                continue
            key = (row["task_id"], row["physical_index"] if "physical_index" in row else next(x["physical_index"] for x in matrix.logical_rows if x["logical_index"] == row["logical_index"]))
            other = grouped.get(key)
            stripped = {k: v for k, v in row.items() if k not in {"logical_index", "budget_id", "semantic_sha256"}}
            if other is not None and stripped != other:
                raise ValueError("stage-2 duplicate Base metric identities differ")
            grouped[key] = stripped
    return metrics, engineering


def _by_budget_profile(metrics: Sequence[Mapping[str, object]]) -> dict[str, dict[str, list[Mapping[str, object]]]]:
    result = {budget: {profile: [] for profile in PROFILES} for budget in BUDGET_IDS}
    for row in metrics:
        budget = row["budget_id"]
        if budget in result:
            result[budget][row["profile"]].append(row)
    return result


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("ordered mean requires nonempty values")
    return sum(values) / len(values)


def _budget_effects(rows: Mapping[str, Sequence[Mapping[str, object]]]) -> tuple[float, float, float]:
    full = rows["full"]
    hit = rows["hit_only"]
    barrier = rows["barrier_only"]
    if not full or len(full) != len(hit) or len(full) != len(barrier):
        raise ValueError("budget metrics are incomplete")
    hit_effect = _mean([float(f["iou"]) - float(b["iou"]) for f, b in zip(full, barrier, strict=True)])
    barrier_effect = _mean([float(f["iou"]) - float(h["iou"]) for f, h in zip(full, hit, strict=True)])
    return hit_effect, barrier_effect, _mean([float(row["iou"]) for row in full])


def _rank(survivors: Sequence[str], grouped: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]]) -> tuple[str, ...]:
    order = {budget: index for index, budget in enumerate(BUDGET_IDS)}
    return tuple(sorted(survivors, key=lambda budget: (-min(_budget_effects(grouped[budget])[:2]), -_budget_effects(grouped[budget])[2], order[budget])))


def evaluate_development_decision(stage: str, logical_metric_rows: Sequence[object], engineering_rows: Sequence[object]) -> V7DevelopmentDecision:
    if stage not in _NEXT_STAGE:
        raise ValueError("decision stage must be a development stage")
    # Reconstruct the matrix identity directly from row indices; the evidence builder
    # performs full matrix authentication. This pure entry remains usable by analyzers.
    if not engineering_rows:
        raise ValueError("engineering rows are required")
    first_eng = _plain(engineering_rows[0])
    if type(first_eng) is not dict:
        raise ValueError("engineering row has wrong type")
    physical_rows = []
    for expected_index, raw in enumerate(engineering_rows):
        row = _plain(raw)
        if type(row) is not dict:
            raise ValueError("engineering row has wrong type")
        if row.get("physical_index") != expected_index:
            raise ValueError("engineering rows are not in exact physical order")
        physical_rows.append({key: row[key] for key in ("physical_index", "task_id", "budget_id", "profile", "config_sha256", "run_path")})
    logical_rows = []
    observed_logical_indices: list[int] = []
    for raw in logical_metric_rows:
        row = _plain(raw)
        if type(row) is not dict:
            raise ValueError("metric row has wrong type")
        logical_index = row.get("logical_index")
        if type(logical_index) is not int:
            raise ValueError("metric logical index has wrong exact type")
        observed_logical_indices.append(logical_index)
        # Find by logical identity, including repeated Base physical ownership.
        candidates = [p for p in physical_rows if p["task_id"] == row.get("task_id") and p["profile"] == row.get("profile") and (p["budget_id"] == row.get("budget_id") or p["profile"] == "base") and p["config_sha256"] == row.get("config_sha256") and p["run_path"] == row.get("run_path")]
        if len(candidates) != 1:
            raise ValueError("metric row has no unique physical identity")
        logical_rows.append({"logical_index": row.get("logical_index"), **candidates[0], "budget_id": row.get("budget_id")})
    if observed_logical_indices != sorted(set(observed_logical_indices)):
        raise ValueError("metric rows are not in exact logical order")
    pseudo_payload = {"schema": STAGE_MATRIX_SCHEMA, "stage": stage, "campaign_sha256": "0" * 64, "endpoint": STAGE_ENDPOINTS[stage], "authorization_path": None, "authorization_sha256": None, "physical_rows": physical_rows, "logical_rows": logical_rows}
    pseudo_payload["semantic_sha256"] = canonical_sha256(pseudo_payload)
    matrix = V7StageMatrix(stage, STAGE_ENDPOINTS[stage], tuple(_immutable(r) for r in physical_rows), tuple(_immutable(r) for r in logical_rows), pseudo_payload["semantic_sha256"], _immutable(pseudo_payload))
    # Validate engineering against its own exact matrix; validate metrics using the
    # surviving logical projection computed from engineering.
    metrics, engineering = _validate_evidence_rows(matrix, logical_metric_rows, engineering_rows)
    failed_base = any(row["profile"] == "base" and row["status"] == "FAIL" for row in engineering)
    failed_budgets = {row["budget_id"] for row in engineering if row["status"] == "FAIL" and row["budget_id"] is not None}
    if failed_base or (stage == "development-3000" and any(row["status"] == "FAIL" for row in engineering)):
        return V7DevelopmentDecision(stage, "MECHANISM_NOT_CONFIRMED", (), (), (), None, None)
    grouped = _by_budget_profile(metrics)
    candidates = [budget for budget in BUDGET_IDS if budget not in failed_budgets and grouped[budget]["full"]]
    if stage == "development-0300":
        survivors = candidates
        ranked = _rank(survivors, grouped) if survivors else ()
        if len(ranked) < 2:
            return V7DevelopmentDecision(stage, "MECHANISM_NOT_CONFIRMED", ranked, tuple(survivors), (), None, None)
        return V7DevelopmentDecision(stage, "ADVANCE", ranked, tuple(survivors), ranked[:2], None, _NEXT_STAGE[stage])
    if stage == "development-1000":
        passing = []
        for budget in candidates:
            rows = grouped[budget]
            if any(len(rows[p]) != 3 for p in PROFILES):
                raise ValueError("stage-1000 budget rows are incomplete")
            full, base, hit, barrier = rows["full"], rows["base"], rows["hit_only"], rows["barrier_only"]
            gates = (
                _mean([f["iou"] - b["iou"] for f, b in zip(full, base, strict=True)]) >= -0.002,
                _mean([h["iou"] - b["iou"] for h, b in zip(hit, base, strict=True)]) >= -0.003,
                _mean([h["iou"] - b["iou"] for h, b in zip(barrier, base, strict=True)]) >= -0.003,
                _budget_effects(rows)[0] > 0.0,
                _budget_effects(rows)[1] > 0.0,
                _mean([f["recall"] - b["recall"] for f, b in zip(full, barrier, strict=True)]) > 0.0,
                _mean([h["fpr"] - f["fpr"] for h, f in zip(hit, full, strict=True)]) > 0.0,
            )
            if all(gates):
                passing.append(budget)
        ranked = _rank(passing, grouped) if passing else ()
        if not ranked:
            return V7DevelopmentDecision(stage, "MECHANISM_NOT_CONFIRMED", (), (), (), None, None)
        return V7DevelopmentDecision(stage, "ADVANCE", ranked, tuple(passing), (ranked[0],), ranked[0], _NEXT_STAGE[stage])
    if len(candidates) != 1:
        raise ValueError("stage-3000 must contain exactly one selected budget")
    budget = candidates[0]
    rows = grouped[budget]
    if any(len(rows[p]) != 3 for p in PROFILES):
        raise ValueError("stage-3000 rows are incomplete")
    full, base, hit, barrier = rows["full"], rows["base"], rows["hit_only"], rows["barrier_only"]
    hit_iou = [f["iou"] - b["iou"] for f, b in zip(full, barrier, strict=True)]
    barrier_iou = [f["iou"] - h["iou"] for f, h in zip(full, hit, strict=True)]
    hit_dice = [f["dice"] - b["dice"] for f, b in zip(full, barrier, strict=True)]
    barrier_dice = [f["dice"] - h["dice"] for f, h in zip(full, hit, strict=True)]
    gates = (
        _mean(hit_iou) >= 0.010, _mean(barrier_iou) >= 0.010,
        _mean(hit_dice) >= 0.005, _mean(barrier_dice) >= 0.005,
        all(value > 0.0 for value in hit_iou), all(value > 0.0 for value in barrier_iou),
        _mean([r["iou"] for r in full]) >= 0.95, min(r["iou"] for r in full) >= 0.94,
        _mean([h["iou"] - b["iou"] for h, b in zip(hit, base, strict=True)]) >= -0.001,
        _mean([h["dice"] - b["dice"] for h, b in zip(hit, base, strict=True)]) >= -0.001,
        _mean([h["iou"] - b["iou"] for h, b in zip(barrier, base, strict=True)]) >= -0.001,
        _mean([h["dice"] - b["dice"] for h, b in zip(barrier, base, strict=True)]) >= -0.001,
    )
    status = "PASS" if all(gates) else "MECHANISM_NOT_CONFIRMED"
    return V7DevelopmentDecision(stage, status, (budget,), (budget,), (), budget, None)


def _matrix_hash(rows: Sequence[Mapping[str, object]]) -> str:
    return canonical_sha256([_plain(row) for row in rows])


def _build_stage_evidence_for_test(matrix: V7StageMatrix, metric_rows: Sequence[object], engineering_rows: Sequence[object]) -> Mapping[str, object]:
    metrics, engineering = _validate_evidence_rows(matrix, metric_rows, engineering_rows)
    payload: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "stage": matrix.stage,
        "campaign_sha256": matrix.payload["campaign_sha256"],
        "endpoint": matrix.endpoint,
        "physical_matrix_sha256": _matrix_hash(matrix.physical_rows),
        "logical_matrix_sha256": _matrix_hash(matrix.logical_rows),
        "engineering_rows": engineering,
        "metric_rows": metrics,
        "evidence_sha256": canonical_sha256({"engineering_rows": engineering, "metric_rows": metrics}),
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    return _immutable(payload)


def _validate_stage_evidence(value: object, matrix: V7StageMatrix) -> dict[str, object]:
    row = _exact_dict(_plain(value), {"schema", "stage", "campaign_sha256", "endpoint", "physical_matrix_sha256", "logical_matrix_sha256", "engineering_rows", "metric_rows", "evidence_sha256", "semantic_sha256"}, label="stage evidence")
    if row["schema"] != EVIDENCE_SCHEMA or row["stage"] != matrix.stage or row["campaign_sha256"] != matrix.payload["campaign_sha256"] or row["endpoint"] != matrix.endpoint:
        raise ValueError("stage evidence identity mismatch")
    if row["physical_matrix_sha256"] != _matrix_hash(matrix.physical_rows) or row["logical_matrix_sha256"] != _matrix_hash(matrix.logical_rows):
        raise ValueError("stage evidence matrix hash mismatch")
    metrics, engineering = _validate_evidence_rows(matrix, row["metric_rows"], row["engineering_rows"])
    if row["evidence_sha256"] != canonical_sha256({"engineering_rows": engineering, "metric_rows": metrics}):
        raise ValueError("stage evidence row hash mismatch")
    unsigned = dict(row); terminal = unsigned.pop("semantic_sha256")
    if canonical_sha256(unsigned) != terminal:
        raise ValueError("stage evidence semantic hash mismatch")
    return row


def _authorization_from_payload(
    payload: dict[str, object], *, evidence: Mapping[str, object] | None = None,
    matrix: V7StageMatrix | None = None,
    predecessor: V7StageAuthorization | None = None,
) -> V7StageAuthorization:
    raw = canonical_json_bytes(payload)
    return V7StageAuthorization(
        payload["stage"], payload["status"], tuple(payload["ranked_budget_ids"]),
        tuple(payload["surviving_budget_ids"]), payload["selected_budget_id"],
        payload["next_stage"], payload["semantic_sha256"],
        hashlib.sha256(raw).hexdigest(), _immutable(payload),
        None if evidence is None else _immutable(_plain(evidence)), matrix, predecessor,
    )


def _authorization_payload_from_decision(campaign: Mapping[str, object], matrix: V7StageMatrix, evidence: Mapping[str, object], decision: V7DevelopmentDecision) -> dict[str, object]:
    authorized_logical = []
    authorized_config = []
    if decision.status == "ADVANCE":
        # The next matrix is deterministic from this authorization. Store its logical
        # grant projection in budget/task/profile order without circular hashes.
        budgets = decision.authorized_budget_ids
        for budget in budgets:
            for task in DEVELOPMENT_TASKS:
                for profile in PROFILES:
                    authorized_logical.append({"task_id": task, "budget_id": budget, "profile": profile})
        index = _config_index(campaign)
        for task in DEVELOPMENT_TASKS:
            authorized_config.append(index[(task, None, "base")]["config_sha256"])
            for budget in budgets:
                for profile in NONBASE_PROFILES:
                    authorized_config.append(
                        index[(task, budget, profile)]["config_sha256"]
                    )
    payload: dict[str, object] = {
        "schema": AUTHORIZATION_SCHEMA,
        "stage": matrix.stage,
        "campaign_sha256": _campaign_digest(campaign),
        "stage_matrix_sha256": matrix.semantic_sha256,
        "evidence_sha256": evidence["evidence_sha256"],
        "status": decision.status,
        "ranked_budget_ids": list(decision.ranked_budget_ids),
        "surviving_budget_ids": list(decision.surviving_budget_ids),
        "selected_budget_id": decision.selected_budget_id,
        "next_stage": decision.next_stage,
        "authorized_physical_config_sha256": authorized_config,
        "authorized_logical_rows": authorized_logical,
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload


def _build_stage_authorization_for_test(campaign: Mapping[str, object], *, stage: str, matrix: V7StageMatrix, metric_rows: Sequence[object], engineering_rows: Sequence[object]) -> V7StageAuthorization:
    evidence = _build_stage_evidence_for_test(matrix, metric_rows, engineering_rows)
    decision = evaluate_development_decision(stage, metric_rows, engineering_rows)
    return _authorization_from_payload(
        _authorization_payload_from_decision(campaign, matrix, evidence, decision),
        evidence=evidence, matrix=matrix, predecessor=matrix.predecessor,
    )


def build_stage_authorization(campaign: Mapping[str, object], *, stage: str, evidence: Mapping[str, object]) -> V7StageAuthorization:
    campaign_row = _validate_campaign_payload(campaign)
    if stage == "development-0300":
        matrix = resolve_stage_matrix(campaign_row, stage=stage)
    else:
        root = Path(campaign_row["core"]["root_path"])
        previous = _PREVIOUS_STAGE.get(stage)
        if previous is None:
            raise ValueError("unknown development authorization stage")
        authorization = validate_stage_authorization(root / "authorizations" / _AUTH_FILENAMES[previous], expected_stage=previous, campaign=campaign_row)
        matrix = resolve_stage_matrix(campaign_row, stage=stage, authorization=authorization)
    evidence_row = _validate_stage_evidence(evidence, matrix)
    decision = evaluate_development_decision(stage, evidence_row["metric_rows"], evidence_row["engineering_rows"])
    return _authorization_from_payload(
        _authorization_payload_from_decision(campaign_row, matrix, evidence_row, decision),
        evidence=evidence_row, matrix=matrix, predecessor=matrix.predecessor,
    )


def _validate_authorization_shape(value: object, *, expected_stage: str, campaign: Mapping[str, object]) -> dict[str, object]:
    row = _exact_dict(_plain(value), {"schema", "stage", "campaign_sha256", "stage_matrix_sha256", "evidence_sha256", "status", "ranked_budget_ids", "surviving_budget_ids", "selected_budget_id", "next_stage", "authorized_physical_config_sha256", "authorized_logical_rows", "semantic_sha256"}, label="stage authorization")
    if row["schema"] != AUTHORIZATION_SCHEMA or row["stage"] != expected_stage or row["campaign_sha256"] != _campaign_digest(campaign):
        raise ValueError("stage authorization identity mismatch")
    for key in ("stage_matrix_sha256", "evidence_sha256", "semantic_sha256"):
        _digest(row[key], label=key)
    allowed_status = (
        {"ADVANCE", "MECHANISM_NOT_CONFIRMED"}
        if expected_stage in ("development-0300", "development-1000")
        else {"PASS", "MECHANISM_NOT_CONFIRMED"}
    )
    if row["status"] not in allowed_status:
        raise ValueError("stage authorization status is invalid")
    for key in ("ranked_budget_ids", "surviving_budget_ids", "authorized_physical_config_sha256", "authorized_logical_rows"):
        if type(row[key]) is not list:
            raise ValueError("stage authorization list has wrong type")
    if any(type(x) is not str or x not in BUDGET_IDS for x in row["ranked_budget_ids"] + row["surviving_budget_ids"]):
        raise ValueError("authorization contains an unknown budget")
    if len(set(row["ranked_budget_ids"])) != len(row["ranked_budget_ids"]) or len(set(row["surviving_budget_ids"])) != len(row["surviving_budget_ids"]):
        raise ValueError("authorization budget lists contain duplicates")
    if row["selected_budget_id"] is not None and row["selected_budget_id"] not in BUDGET_IDS:
        raise ValueError("authorization selected budget is invalid")
    for digest in row["authorized_physical_config_sha256"]:
        _digest(digest, label="authorized config sha")
    if len(set(row["authorized_physical_config_sha256"])) != len(row["authorized_physical_config_sha256"]):
        raise ValueError("authorized physical config list contains duplicates")
    for logical in row["authorized_logical_rows"]:
        logical_row = _exact_dict(
            logical, {"task_id", "budget_id", "profile"},
            label="authorized logical row",
        )
        if (
            logical_row["task_id"] not in DEVELOPMENT_TASKS
            or logical_row["budget_id"] not in BUDGET_IDS
            or logical_row["profile"] not in PROFILES
        ):
            raise ValueError("authorized logical row has an unknown identity")
    if row["status"] == "MECHANISM_NOT_CONFIRMED":
        if row["next_stage"] is not None or row["authorized_physical_config_sha256"] or row["authorized_logical_rows"]:
            raise ValueError("negative authorization cannot grant later execution")
    elif expected_stage == "development-0300":
        if row["next_stage"] != "development-1000" or row["selected_budget_id"] is not None or len(row["ranked_budget_ids"]) < 2:
            raise ValueError("stage-300 ADVANCE authorization is inconsistent")
    elif expected_stage == "development-1000":
        if row["next_stage"] != "development-3000" or row["selected_budget_id"] is None or row["ranked_budget_ids"][:1] != [row["selected_budget_id"]]:
            raise ValueError("stage-1000 ADVANCE authorization is inconsistent")
    else:
        if row["status"] != "PASS" or row["next_stage"] is not None or row["selected_budget_id"] is None or row["authorized_physical_config_sha256"] or row["authorized_logical_rows"]:
            raise ValueError("stage-3000 terminal authorization is inconsistent")
    unsigned = dict(row); terminal = unsigned.pop("semantic_sha256")
    if canonical_sha256(unsigned) != terminal:
        raise ValueError("authorization semantic hash mismatch")
    return row


def _validate_authorization(
    value: object, *, expected_stage: str, campaign: Mapping[str, object]
) -> dict[str, object]:
    if not isinstance(value, V7StageAuthorization):
        raise ValueError("stage authorization requires authenticated evidence proof")
    row = _validate_authorization_shape(
        value.payload, expected_stage=expected_stage, campaign=campaign
    )
    if value.evidence is None or value.matrix is None:
        raise ValueError("stage authorization evidence/matrix proof is unavailable")
    if expected_stage == "development-0300":
        if value.predecessor is not None:
            raise ValueError("stage-300 authorization cannot have a predecessor")
        expected_matrix = _resolve_stage_matrix_validated(campaign, stage=expected_stage)
    else:
        previous_stage = _PREVIOUS_STAGE[expected_stage]
        if value.predecessor is None:
            raise ValueError("stage authorization predecessor proof is unavailable")
        _validate_authorization(
            value.predecessor, expected_stage=previous_stage, campaign=campaign
        )
        expected_matrix = _resolve_stage_matrix_validated(
            campaign, stage=expected_stage, authorization=value.predecessor
        )
    if canonical_json_bytes(_plain(value.matrix.payload)) != canonical_json_bytes(
        _plain(expected_matrix.payload)
    ):
        raise ValueError("stage authorization matrix proof differs from reconstruction")
    evidence = _validate_stage_evidence(value.evidence, expected_matrix)
    decision = evaluate_development_decision(
        expected_stage, evidence["metric_rows"], evidence["engineering_rows"]
    )
    expected = _authorization_payload_from_decision(
        campaign, expected_matrix, evidence, decision
    )
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise ValueError("stage authorization differs from recomputed evidence decision")
    if value.raw_sha256 != hashlib.sha256(canonical_json_bytes(row)).hexdigest():
        raise ValueError("stage authorization raw digest differs")
    return row


def _validate_authorization_payload(
    value: object, *, expected_stage: str, campaign: Mapping[str, object]
) -> dict[str, object]:
    """Authenticate an authorization object, never a merely self-hashed mapping."""
    return _validate_authorization(
        value, expected_stage=expected_stage, campaign=_validate_campaign_payload(campaign)
    )


def publish_stage_authorization(root: Path, payload: object) -> None:
    if not isinstance(payload, V7StageAuthorization):
        raise ValueError("authorization publication requires an authenticated object")
    row = _plain(payload.payload)
    if row.get("stage") not in _AUTH_FILENAMES:
        raise ValueError("authorization payload/stage is invalid")
    campaign = validate_campaign_manifest(root / "manifests/campaign.json")
    if root != Path(campaign["core"]["root_path"]):
        raise ValueError("authorization root argument differs from campaign")
    validated = _validate_authorization(
        payload, expected_stage=row["stage"], campaign=campaign
    )
    evidence = _validate_stage_evidence(payload.evidence, payload.matrix)
    specification, inventory, protected = _authority_snapshots()
    source_root = Path(campaign["core"]["source_bundle"]["project_root"])
    with open_v7_root_transaction(root) as transaction:
        transaction.ensure_registered_directory("authorizations")
        campaign_snapshots = _snapshot_campaign_dependencies(
            transaction, campaign, include_campaign=True
        )
        predecessor_snapshots = _snapshot_authorization_chain(
            transaction,
            _authorization_chain_stages(row["stage"], include_self=False),
        )
        if payload.predecessor is not None:
            previous = payload.predecessor.stage
            previous_snapshot = predecessor_snapshots[
                Path("authorizations") / _AUTH_FILENAMES[previous]
            ]
            if previous_snapshot.sha256 != payload.predecessor.raw_sha256:
                raise ValueError("authorization predecessor bytes changed")
        transaction.publish(
            Path("authorizations") / _EVIDENCE_FILENAMES[row["stage"]],
            canonical_json_bytes(evidence),
        )
        transaction.publish(Path("authorizations") / _AUTH_FILENAMES[row["stage"]], canonical_json_bytes(validated))
        transaction.fsync(Path("authorizations"))
        _require_unchanged_transaction_snapshots(
            transaction, campaign_snapshots, label="authorization campaign/suite dependencies"
        )
        _require_unchanged_transaction_snapshots(
            transaction, predecessor_snapshots, label="authorization predecessor chain"
        )
        _revalidate_authorities(specification, inventory, protected)
        validate_training_source_bundle(campaign["core"]["source_bundle"], source_root)


def _validate_stage_authorization_transaction(
    transaction: object, *, expected_stage: str, campaign: Mapping[str, object]
) -> V7StageAuthorization:
    predecessor = None
    if expected_stage != "development-0300":
        predecessor = _validate_stage_authorization_transaction(
            transaction,
            expected_stage=_PREVIOUS_STAGE[expected_stage], campaign=campaign,
        )
    auth_relative = Path("authorizations") / _AUTH_FILENAMES[expected_stage]
    evidence_relative = Path("authorizations") / _EVIDENCE_FILENAMES[expected_stage]
    auth_snapshot = transaction.snapshot(auth_relative)
    evidence_snapshot = transaction.snapshot(evidence_relative)
    row = _decode_registry_json(auth_snapshot.payload, label="v7 stage authorization")
    evidence = _decode_registry_json(evidence_snapshot.payload, label="v7 stage evidence")
    if auth_snapshot.payload != canonical_json_bytes(row) or evidence_snapshot.payload != canonical_json_bytes(evidence):
        raise ValueError("authorization or evidence is not canonical JSON")
    matrix = _resolve_stage_matrix_validated(
        campaign, stage=expected_stage, authorization=predecessor
    )
    candidate = _authorization_from_payload(
        row, evidence=evidence, matrix=matrix, predecessor=predecessor
    )
    if candidate.raw_sha256 != auth_snapshot.sha256:
        raise ValueError("authorization file SHA differs")
    validated = _validate_authorization(
        candidate, expected_stage=expected_stage, campaign=campaign
    )
    if validated["evidence_sha256"] != evidence["evidence_sha256"]:
        raise ValueError("authorization evidence digest differs")
    if transaction.snapshot(auth_relative) != auth_snapshot or transaction.snapshot(evidence_relative) != evidence_snapshot:
        raise ValueError("authorization evidence chain changed during validation")
    return candidate


def validate_stage_authorization(path: Path, *, expected_stage: str, campaign: Mapping[str, object]) -> V7StageAuthorization:
    if expected_stage not in _AUTH_FILENAMES:
        raise ValueError("expected authorization stage is invalid")
    root = Path(_validate_campaign_payload(campaign)["core"]["root_path"])
    canonical = root / "authorizations" / _AUTH_FILENAMES[expected_stage]
    if path != canonical:
        raise ValueError("authorization path is not canonical")
    with open_v7_root_transaction(root) as transaction:
        campaign_snapshot = transaction.snapshot(Path("manifests/campaign.json"))
        if campaign_snapshot.sha256 != _campaign_digest(campaign):
            raise ValueError("authorization campaign file identity changed")
        result = _validate_stage_authorization_transaction(
            transaction, expected_stage=expected_stage, campaign=campaign
        )
        if transaction.snapshot(Path("manifests/campaign.json")) != campaign_snapshot:
            raise ValueError("authorization campaign changed during validation")
        return result


def _build_formal_from_tasks(
    campaign_row: Mapping[str, object], auth: Mapping[str, object],
    formal_suite_identity: Mapping[str, object], tasks: Sequence[Mapping[str, object]],
) -> V7FormalMatrixManifest:
    root = Path(campaign_row["core"]["root_path"])
    auth_raw_sha = hashlib.sha256(canonical_json_bytes(auth)).hexdigest()
    core: dict[str, object] = {
        "schema": FORMAL_CORE_SCHEMA,
        "root_path": str(root),
        "formal_manifest_path": "manifests/formal_matrix.json",
        "campaign_semantic_sha256": campaign_row["semantic_sha256"],
        "formal_suite": copy.deepcopy(dict(formal_suite_identity)),
        "development_3000_authorization_path": f"authorizations/{_AUTH_FILENAMES['development-3000']}",
        "development_3000_authorization_sha256": auth_raw_sha,
        "selected_budget_id": auth["selected_budget_id"],
        "formal_budget_ratio": BUDGET_RATIOS[auth["selected_budget_id"]],
        "formal_tasks": tasks,
        "formal_policy": {"profiles": list(PROFILES), "endpoint": 3000, "formal_updates_blocked_before_pass": True},
        "campaign_path": "manifests/campaign.json",
        "cameras": campaign_row["core"]["cameras"],
        "scientific_policy": campaign_row["core"]["scientific_policy"],
        "source_bundle": campaign_row["core"]["source_bundle"],
        "environment": campaign_row["core"]["environment"],
    }
    core_sha = canonical_sha256(core)
    configs = []
    for task in tasks:
        for profile in PROFILES:
            configs.append(_config(core, task, cohort="formal", profile=profile, budget_id=None if profile == "base" else auth["selected_budget_id"], campaign_core_sha256=campaign_row["campaign_core_sha256"], formal_core_sha256=core_sha, development_authorization_sha256=auth_raw_sha))
    payload: dict[str, object] = {"schema": FORMAL_MATRIX_SCHEMA, "core": core, "formal_core_sha256": core_sha, "formal_configs": configs}
    payload["semantic_sha256"] = canonical_sha256(payload)
    return V7FormalMatrixManifest(payload["semantic_sha256"], _immutable(payload))


def build_formal_matrix_manifest(campaign: Mapping[str, object], development_3000_authorization: object) -> V7FormalMatrixManifest:
    campaign_row = _validate_campaign_payload(campaign)
    auth = _validate_authorization(
        development_3000_authorization,
        expected_stage="development-3000", campaign=campaign_row,
    )
    if auth["status"] != "PASS" or auth["selected_budget_id"] not in BUDGET_IDS:
        raise ValueError("formal matrix requires authenticated development-3000 PASS")
    root = Path(campaign_row["core"]["root_path"])
    published_auth = validate_stage_authorization(
        root / "authorizations" / _AUTH_FILENAMES["development-3000"],
        expected_stage="development-3000", campaign=campaign_row,
    )
    if canonical_json_bytes(_plain(published_auth.payload)) != canonical_json_bytes(auth):
        raise ValueError("formal matrix requires the exact published development PASS")
    formal_suite = validate_v7_suite(root / "manifests/formal_suite.json", expected_kind="formal")
    tasks = [_task_record(task) for task in formal_suite.tasks]
    return _build_formal_from_tasks(
        campaign_row, auth, _suite_identity(formal_suite), tasks
    )


def _build_formal_matrix_manifest_for_test(
    campaign: Mapping[str, object], development_3000_authorization: object,
) -> V7FormalMatrixManifest:
    campaign_row = _validate_campaign_payload(campaign)
    auth = _validate_authorization(
        development_3000_authorization,
        expected_stage="development-3000", campaign=campaign_row,
    )
    if auth["status"] != "PASS" or auth["selected_budget_id"] not in BUDGET_IDS:
        raise ValueError("formal matrix requires development-3000 PASS")
    tasks = _private_formal_task_records()
    return _build_formal_from_tasks(
        campaign_row, auth, campaign_row["core"]["formal_suite"], tasks
    )


def _formal_stage_matrix(
    campaign: Mapping[str, object], auth: Mapping[str, object],
    formal: V7FormalMatrixManifest,
) -> V7StageMatrix:
    payload_row = _validate_formal_payload(formal.payload, campaign)
    expected_authorization_sha = hashlib.sha256(canonical_json_bytes(auth)).hexdigest()
    if (
        auth["status"] != "PASS"
        or payload_row["core"]["development_3000_authorization_sha256"]
        != expected_authorization_sha
        or payload_row["core"]["selected_budget_id"] != auth["selected_budget_id"]
        or payload_row["core"]["formal_budget_ratio"]
        != BUDGET_RATIOS[auth["selected_budget_id"]]
    ):
        raise ValueError("formal matrix does not bind the exact supplied development PASS")
    physical = [_physical_row(index, config) for index, config in enumerate(payload_row["formal_configs"])]
    logical = [_logical_row(index, row) for index, row in enumerate(physical)]
    payload: dict[str, object] = {
        "schema": STAGE_MATRIX_SCHEMA,
        "stage": "formal-3000",
        "campaign_sha256": _campaign_digest(campaign),
        "endpoint": 3000,
        "authorization_path": f"authorizations/{_AUTH_FILENAMES['development-3000']}",
        "authorization_sha256": expected_authorization_sha,
        "physical_rows": physical,
        "logical_rows": logical,
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    return V7StageMatrix(
        "formal-3000", 3000,
        tuple(_immutable(row) for row in physical),
        tuple(_immutable(row) for row in logical),
        payload["semantic_sha256"], _immutable(payload), _immutable(_plain(campaign)), None,
    )


def _resolve_formal_stage_matrix_for_test(
    campaign: Mapping[str, object], development_3000_authorization: object,
    formal: V7FormalMatrixManifest,
) -> V7StageMatrix:
    campaign_row = _validate_campaign_payload(campaign)
    auth = _validate_authorization(
        development_3000_authorization,
        expected_stage="development-3000", campaign=campaign_row,
    )
    if auth["status"] != "PASS":
        raise ValueError("formal stage requires development PASS")
    return _formal_stage_matrix(campaign_row, auth, formal)


def publish_formal_matrix_manifest(root: Path, payload: object) -> None:
    row = _plain(payload.payload if isinstance(payload, V7FormalMatrixManifest) else payload)
    campaign = validate_campaign_manifest(root / "manifests/campaign.json")
    if root != Path(campaign["core"]["root_path"]):
        raise ValueError("formal root argument differs from campaign")
    try:
        published = validate_stage_authorization(
            root / "authorizations" / _AUTH_FILENAMES["development-3000"],
            expected_stage="development-3000", campaign=campaign,
        )
    except (OSError, ValueError, AttributeError, KeyError) as error:
        raise ValueError("formal publication requires published development-3000 PASS") from error
    if published.status != "PASS":
        raise ValueError("formal publication requires published development-3000 PASS")
    validated = _validate_formal_payload(row, campaign)
    if (
        validated["core"]["development_3000_authorization_sha256"] != published.raw_sha256
        or validated["core"]["selected_budget_id"] != published.selected_budget_id
    ):
        raise ValueError("formal publication requires exact published development-3000 PASS")
    specification, inventory, protected = _authority_snapshots()
    source_root = Path(campaign["core"]["source_bundle"]["project_root"])
    with open_v7_root_transaction(root) as transaction:
        campaign_snapshots = _snapshot_campaign_dependencies(
            transaction, campaign, include_campaign=True
        )
        authorization_snapshots = _snapshot_authorization_chain(
            transaction,
            _authorization_chain_stages("development-3000", include_self=True),
        )
        auth_snapshot = authorization_snapshots[
            Path("authorizations") / _AUTH_FILENAMES["development-3000"]
        ]
        if auth_snapshot.sha256 != published.raw_sha256:
            raise ValueError("formal dependencies changed before publication")
        transaction.publish(Path("manifests/formal_matrix.json"), canonical_json_bytes(validated))
        transaction.fsync(Path("manifests"))
        _require_unchanged_transaction_snapshots(
            transaction, campaign_snapshots, label="formal campaign/suite dependencies"
        )
        _require_unchanged_transaction_snapshots(
            transaction, authorization_snapshots, label="formal authorization chain"
        )
        _revalidate_authorities(specification, inventory, protected)
        validate_training_source_bundle(campaign["core"]["source_bundle"], source_root)
    observed_suite = validate_v7_suite(
        Path(campaign["core"]["formal_suite"]["path"]), expected_kind="formal"
    )
    if _suite_identity(observed_suite) != campaign["core"]["formal_suite"]:
        raise ValueError("formal suite changed after publication")
    final_authorization = validate_stage_authorization(
        root / "authorizations" / _AUTH_FILENAMES["development-3000"],
        expected_stage="development-3000", campaign=campaign,
    )
    if (
        final_authorization.status != "PASS"
        or final_authorization.raw_sha256
        != validated["core"]["development_3000_authorization_sha256"]
        or final_authorization.selected_budget_id
        != validated["core"]["selected_budget_id"]
    ):
        raise ValueError("formal publication terminal PASS binding changed")


def _validate_formal_payload(
    value: object, campaign: Mapping[str, object], *,
    _authenticated_suite: AuthenticatedV7Suite | None = None,
) -> dict[str, object]:
    row = _exact_dict(_plain(value), {"schema", "core", "formal_core_sha256", "formal_configs", "semantic_sha256"}, label="formal matrix")
    if row["schema"] != FORMAL_MATRIX_SCHEMA or type(row["core"]) is not dict or row["core"].get("schema") != FORMAL_CORE_SCHEMA:
        raise ValueError("formal matrix schema mismatch")
    if canonical_sha256(row["core"]) != row["formal_core_sha256"]:
        raise ValueError("formal core SHA mismatch")
    core = _exact_dict(
        row["core"],
        {
            "schema", "root_path", "formal_manifest_path",
            "campaign_semantic_sha256", "formal_suite",
            "development_3000_authorization_path",
            "development_3000_authorization_sha256", "selected_budget_id",
            "formal_budget_ratio", "formal_tasks", "formal_policy",
            "campaign_path", "cameras", "scientific_policy", "source_bundle",
            "environment",
        },
        label="formal core",
    )
    if core["campaign_semantic_sha256"] != campaign["semantic_sha256"]:
        raise ValueError("formal matrix belongs to a foreign campaign")
    if (
        core["root_path"] != campaign["core"]["root_path"]
        or core["formal_manifest_path"] != "manifests/formal_matrix.json"
        or core["campaign_path"] != "manifests/campaign.json"
        or core["formal_suite"] != campaign["core"]["formal_suite"]
        or core["development_3000_authorization_path"] != f"authorizations/{_AUTH_FILENAMES['development-3000']}"
        or core["selected_budget_id"] not in BUDGET_IDS
        or core["formal_budget_ratio"] != BUDGET_RATIOS[core["selected_budget_id"]]
        or core["formal_policy"] != {"profiles": list(PROFILES), "endpoint": 3000, "formal_updates_blocked_before_pass": True}
        or core["cameras"] != campaign["core"]["cameras"]
        or core["scientific_policy"] != campaign["core"]["scientific_policy"]
        or core["source_bundle"] != campaign["core"]["source_bundle"]
        or core["environment"] != campaign["core"]["environment"]
    ):
        raise ValueError("formal core differs from frozen campaign policy")
    tasks = core["formal_tasks"]
    if type(tasks) is not list or len(tasks) != 10 or tuple(task.get("task_id") for task in tasks if type(task) is dict) != FORMAL_TASKS:
        raise ValueError("formal task registry/order mismatch")
    source_root = Path(campaign["core"]["source_bundle"]["project_root"])
    if source_root != PROJECT_ROOT:
        expected_tasks = _private_formal_task_records()
    else:
        suite = _authenticated_suite
        if suite is None:
            suite = validate_v7_suite(Path(core["formal_suite"]["path"]), expected_kind="formal")
        if _suite_identity(suite) != core["formal_suite"]:
            raise ValueError("formal suite identity changed")
        expected_tasks = [_task_record(task) for task in suite.tasks]
    if canonical_json_bytes(tasks) != canonical_json_bytes(expected_tasks):
        raise ValueError("formal tasks differ from authenticated suite")
    configs = row["formal_configs"]
    if type(configs) is not list or len(configs) != 40 or [(r.get("task_id"), r.get("profile")) for r in configs] != [(task, profile) for task in FORMAL_TASKS for profile in PROFILES]:
        raise ValueError("formal matrix must contain exact ordered 10x4 configs")
    expected_configs = []
    for task in tasks:
        for profile in PROFILES:
            expected_configs.append(
                _config(
                    core, task, cohort="formal", profile=profile,
                    budget_id=None if profile == "base" else core["selected_budget_id"],
                    campaign_core_sha256=campaign["campaign_core_sha256"],
                    formal_core_sha256=row["formal_core_sha256"],
                    development_authorization_sha256=core["development_3000_authorization_sha256"],
                )
            )
    if canonical_json_bytes(configs) != canonical_json_bytes(expected_configs):
        raise ValueError("formal configs differ from independent reconstruction")
    unsigned = dict(row); terminal = unsigned.pop("semantic_sha256")
    if canonical_sha256(unsigned) != terminal:
        raise ValueError("formal matrix semantic SHA mismatch")
    return row


def validate_formal_matrix_manifest(path: Path, campaign: Mapping[str, object]) -> V7FormalMatrixManifest:
    campaign_row = _validate_campaign_payload(campaign)
    root = Path(campaign_row["core"]["root_path"])
    if path != root / "manifests/formal_matrix.json":
        raise ValueError("formal matrix path is not canonical")
    authenticated_suite = validate_v7_suite(
        root / "manifests/formal_suite.json", expected_kind="formal"
    )
    with open_v7_root_transaction(root) as transaction:
        snapshot = transaction.snapshot(Path("manifests/formal_matrix.json"))
        row = _decode_registry_json(snapshot.payload, label="v7 formal matrix")
        if snapshot.payload != canonical_json_bytes(row):
            raise ValueError("formal matrix is not canonical JSON")
        validated = _validate_formal_payload(
            row, campaign_row, _authenticated_suite=authenticated_suite
        )
        auth_snapshot = transaction.snapshot(
            Path("authorizations") / _AUTH_FILENAMES["development-3000"]
        )
        if auth_snapshot.sha256 != validated["core"]["development_3000_authorization_sha256"]:
            raise ValueError("formal matrix development authorization bytes changed")
        suite_snapshot = transaction.snapshot(Path("manifests/formal_suite.json"))
        if suite_snapshot.sha256 != validated["core"]["formal_suite"]["sha256"]:
            raise ValueError("formal matrix suite bytes changed")
        result = V7FormalMatrixManifest(validated["semantic_sha256"], _immutable(validated))
        authorization_snapshots = _snapshot_authorization_chain(
            transaction,
            _authorization_chain_stages("development-3000", include_self=True),
        )
        _require_unchanged_transaction_snapshots(
            transaction, authorization_snapshots, label="formal validation authorization chain"
        )
    authorization = validate_stage_authorization(
        root / "authorizations" / _AUTH_FILENAMES["development-3000"],
        expected_stage="development-3000", campaign=campaign_row,
    )
    if (
        authorization.status != "PASS"
        or authorization.raw_sha256
        != validated["core"]["development_3000_authorization_sha256"]
        or authorization.selected_budget_id != validated["core"]["selected_budget_id"]
    ):
        raise ValueError("formal matrix does not bind exact published development PASS")
    formal_suite = validate_v7_suite(
        root / "manifests/formal_suite.json", expected_kind="formal"
    )
    if _suite_identity(formal_suite) != validated["core"]["formal_suite"]:
        raise ValueError("formal suite semantic identity changed")
    return result


__all__ = [
    "V7DevelopmentDecision", "V7FormalMatrixManifest", "V7StageAuthorization", "V7StageMatrix",
    "build_campaign_manifest", "publish_campaign_manifest", "validate_campaign_manifest",
    "rebuild_resolved_config", "revalidate_static_identities", "resolve_stage_matrix",
    "evaluate_development_decision", "build_stage_authorization", "publish_stage_authorization",
    "validate_stage_authorization", "build_formal_matrix_manifest", "publish_formal_matrix_manifest",
    "validate_formal_matrix_manifest", "expected_training_source_paths",
    "build_training_source_bundle", "validate_training_source_bundle",
]
