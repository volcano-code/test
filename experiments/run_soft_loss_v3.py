"""Immutable stage orchestration for the registered soft-loss-v3 study."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
import fcntl
import importlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import MappingProxyType, ModuleType

import experiments.soft_loss_v3_registry as registry_module
from experiments.soft_loss_v3_registry import SuiteIdentity, load_bound_suite_identity
from shadowforge_pipeline.soft_voxel_v3.calibration import validate_frozen_weight_config
from shadowforge_pipeline.soft_voxel_v3.provenance import (
    canonical_hash,
    canonical_json_bytes,
    implementation_source_manifest,
    normalize_builtin,
)


_PRIMARY = ("l12_bal", "dic_only", "dts_only", "v3_full")
_APPENDIX = ("v3_zero_margin", "v3_no_l1", "v3_no_l2")
_DEVELOPMENT = ("arcane_casters", "blade_heroes", "stage_dancers")
_FORMAL = tuple(f"connected_{index:02d}" for index in range(1, 11))

_STAGE_MATRIX = {
    "bind_manifests": ((), (), "development", 0, 0, 0),
    "calibrate128": (("arcane_casters",), ("l12_bal",), "calibration", 0, 128, 2000),
    "smoke128": (("arcane_casters",), _PRIMARY, "development", 0, 128, 200),
    "screen128": (_DEVELOPMENT, _PRIMARY, "development", 0, 128, 1000),
    "confirm128": (_DEVELOPMENT, _PRIMARY, "development", 0, 128, 2000),
    "cuda2step256": (("arcane_casters",), _PRIMARY, "integration", 0, 256, 2),
    "resource256": (("arcane_casters",), ("l12_bal", "v3_full"), "resource", 0, 256, 200),
    "transfer256": (_DEVELOPMENT, _PRIMARY, "development", 0, 256, 2000),
    "sentinel_repeat256": (("arcane_casters",), _PRIMARY, "technical_repeat", 1, 256, 2000),
    "formal256_primary": (_FORMAL, _PRIMARY, "primary", 0, 256, 2000),
    "formal256_repeat": (_FORMAL, _PRIMARY, "technical_repeat", 1, 256, 2000),
    "appendix256": (_FORMAL, _APPENDIX, "exploratory", 0, 256, 2000),
}

_UPSTREAM = {
    "bind_manifests": None,
    "calibrate128": None,
    "smoke128": None,
    "screen128": ("smoke128", "ADVANCE"),
    "confirm128": ("screen128", "ADVANCE"),
    "cuda2step256": ("confirm128", "ADVANCE"),
    "resource256": ("cuda2step256", "ADVANCE"),
    "transfer256": ("resource256", "ADVANCE"),
    "sentinel_repeat256": ("transfer256", "ADVANCE"),
    "formal256_primary": ("sentinel_repeat256", "ADVANCE"),
    "formal256_repeat": ("formal256_primary", "COMPLETE_ONLY"),
    "appendix256": ("formal_conclusion", "LOCKED"),
}

_PROFILES = {
    "l12_bal": (0.35, 1.0, False, False, None),
    "dic_only": (0.35, 1.0, True, False, None),
    "dts_only": (0.35, 1.0, False, True, (0.04, 0.06)),
    "v3_full": (0.35, 1.0, True, True, (0.04, 0.06)),
    "v3_zero_margin": (0.35, 1.0, True, True, (0.05, 0.05)),
    "v3_no_l1": (0.0, 1.0, True, True, (0.04, 0.06)),
    "v3_no_l2": (0.35, 0.0, True, True, (0.04, 0.06)),
}

_CALIBRATION_EXECUTED_SOURCES = (
    "experiments/soft_loss_v3_registry.py",
    "experiments/bind_soft_loss_v3_suites.py",
    "experiments/run_soft_loss_v3.py",
    "experiments/calibrate_soft_loss_v3.py",
    "shadowforge_pipeline/soft_voxel_v3/calibration.py",
    "shadowforge_pipeline/soft_voxel_v3/provenance.py",
    "shadowforge_pipeline/soft_voxel_v3/trainer.py",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCKED_INTERPRETER = Path(
    "/home/lin/anaconda3/envs/shadowart-revisited/bin/python"
)
_SUMMARIZER_SOURCE = _PROJECT_ROOT / "experiments" / "summarize_soft_loss_v3.py"
_RUNTIME_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "PATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "TORCH_DETERMINISTIC",
        "TORCH_USE_CUDA_DSA",
        "NVIDIA_TF32_OVERRIDE",
        "TMPDIR",
    }
)
_ORDINARY_DRIVER_SOURCES = (
    "experiments/run_soft_loss_v3.py",
    "experiments/soft_loss_v3_registry.py",
)


def _task8_atomic_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Return the exact JSON encoding emitted by Task 8's ``atomic_json``."""

    normalized = normalize_builtin(payload)
    return (
        json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _require_locked_parent_interpreter() -> None:
    try:
        current = Path(sys.executable).resolve(strict=True)
        locked = _LOCKED_INTERPRETER.resolve(strict=True)
    except OSError as error:
        raise ValueError("runner requires the locked parent interpreter") from error
    if current != locked:
        raise ValueError(
            "runner parent sys.executable does not resolve to the locked interpreter"
        )


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class RunSpec:
    """One completely identified scientific cell in a frozen stage matrix."""

    stage: str
    task: str
    variant: str
    profile: str
    replicate_role: str
    replicate_index: int
    inputs_xyz: tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]
    resolution: int
    export_resolution: int
    steps: int
    output_path: Path
    run_id: str
    device: str
    optimizer: str
    learning_rate: float
    initialization_logit: float
    gradient_clip: float
    seed: int
    l1_weight: float
    l2_weight: float
    use_dic: bool
    use_dts: bool
    dic_threshold: float
    dts_thresholds: tuple[float, float] | None
    dic_schedule: tuple[float, float]
    dts_schedule: tuple[float, float]
    evaluation_thresholds: tuple[float, float, float]
    mirror_mode: int
    tsf_mode: int
    tsf4_mode: int
    camera_mode: str
    image_size: int
    control_resolution: int
    zdist: float
    volume_extent: float
    samples_per_ray: int
    final_step_export: bool
    soft_only: bool
    frozen_lambdas: Mapping[str, float] | None
    master_identity: Mapping[str, object]
    suite_identity: Mapping[str, object]
    specification_identity: Mapping[str, object]
    source_identity: Mapping[str, object]
    weight_identity: Mapping[str, object] | None
    inference_eligible: bool

    def __post_init__(self) -> None:
        for name in (
            "inputs_xyz",
            "frozen_lambdas",
            "master_identity",
            "suite_identity",
            "specification_identity",
            "source_identity",
            "weight_identity",
        ):
            object.__setattr__(self, name, _deep_freeze(getattr(self, name)))


@dataclass(frozen=True)
class StageContext:
    """Orchestration-only paths used to validate and construct one stage."""

    output_root: Path
    master_manifest: Path
    development_suite: Path
    formal_suite: Path
    frozen_weights: Path | None
    authorization: Path | None
    formal_campaign: Path | None


def required_upstream(stage: str) -> tuple[str, str] | None:
    """Return the exact direct evidence edge for one registered stage."""

    if type(stage) is not str or stage not in _UPSTREAM:
        raise ValueError(f"unknown registered stage: {stage!r}")
    return _UPSTREAM[stage]


def _ordinary_source_identity() -> dict[str, object]:
    """Bind the recursive Task 8 package and the two Task 11 orchestration drivers."""

    task8 = implementation_source_manifest()
    drivers: dict[str, object] = {}
    snapshots: list[object] = []
    for logical in _ORDINARY_DRIVER_SOURCES:
        snapshot = registry_module._snapshot_file(
            _PROJECT_ROOT / logical,
            label=f"runner source {logical}",
            root=_PROJECT_ROOT,
        )
        drivers[logical] = {
            "path": str(snapshot.path),
            "sha256": snapshot.sha256,
        }
        snapshots.append(snapshot)
    if implementation_source_manifest() != task8:
        raise ValueError("Task 8 implementation source changed while constructing identity")
    for logical, snapshot in zip(_ORDINARY_DRIVER_SOURCES, snapshots):
        registry_module._verify_snapshot_unchanged(
            snapshot,
            label=f"runner source {logical}",
            root=_PROJECT_ROOT,
        )
    bound = {"task8": task8, "drivers": drivers}
    return {
        "schema": "pcraft_soft_loss_v3_runner_source_v1",
        **bound,
        "bundle_sha256": canonical_hash(bound),
    }


def _validate_weight_causal_lineage(
    weight: Mapping[str, object],
    *,
    output_root: Path,
    master_path: Path,
    master_sha256: str,
    master_semantic: str,
    development: SuiteIdentity,
    specification_path: Path,
    specification_sha256: str,
) -> None:
    provenance = registry_module._require_mapping(
        weight.get("provenance"), label="frozen-weight provenance"
    )
    registry_module._exact_keys(
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
        label="frozen-weight provenance",
    )
    expected_stage = output_root / "manifests" / "stages" / "calibrate128.json"
    stage_record = registry_module._require_mapping(
        provenance["stage_manifest"], label="weight calibration stage"
    )
    registry_module._exact_keys(
        stage_record,
        {"path", "sha256", "semantic_sha256"},
        label="weight calibration stage",
    )
    if stage_record["path"] != str(expected_stage):
        raise ValueError("frozen weight does not bind the current calibration stage path")
    stage_snapshot = registry_module._snapshot_file(
        expected_stage,
        label="current calibration stage",
        root=output_root,
    )
    stage_payload = registry_module._decode_json(
        stage_snapshot.data, label="current calibration stage"
    )
    if stage_snapshot.data != canonical_json_bytes(stage_payload):
        raise ValueError("current calibration stage is not canonical JSON")
    if (
        stage_record["sha256"] != stage_snapshot.sha256
        or stage_record["semantic_sha256"] != stage_payload.get("semantic_sha256")
    ):
        raise ValueError("frozen weight calibration-stage identity is stale")
    calibration_run = registry_module._require_mapping(
        provenance["calibration_run"], label="weight calibration run"
    )
    expected_run = (
        output_root
        / "calibrate128"
        / "steps_2000"
        / "arcane_casters"
        / "l12_bal"
        / "replicate_0"
        / "seed_17"
    )
    if calibration_run.get("path") != str(expected_run):
        raise ValueError("frozen weight calibration run belongs to a different root/lineage")
    if stage_payload.get("calibration_run") != dict(calibration_run):
        raise ValueError("frozen weight calibration run differs from the current stage")

    expected_records = (
        (
            "master_manifest",
            {
                "path": str(master_path),
                "sha256": master_sha256,
                "semantic_sha256": master_semantic,
            },
        ),
        (
            "development_suite",
            {
                "path": str(development.path),
                "sha256": development.sha256,
                "semantic_sha256": development.semantic_sha256,
            },
        ),
        (
            "specification",
            {"path": str(specification_path), "sha256": specification_sha256},
        ),
    )
    for label, expected in expected_records:
        record = registry_module._require_mapping(
            provenance[label], label=f"weight {label}"
        )
        registry_module._exact_keys(record, set(expected), label=f"weight {label}")
        if dict(record) != expected:
            raise ValueError(f"frozen weight {label} belongs to a different campaign")
    task = next((item for item in development.tasks if item.case_id == "arcane_casters"), None)
    if task is None:
        raise ValueError("development suite lacks the calibration task")
    expected_inputs = [
        {
            "axis": item.axis,
            "source_path": item.source_path,
            "resolved_path": str(item.resolved_path),
            "sha256": item.sha256,
        }
        for item in task.inputs
    ]
    if canonical_hash({"inputs_xyz": provenance["inputs_xyz"]}) != canonical_hash(
        {"inputs_xyz": expected_inputs}
    ):
        raise ValueError("frozen weight inputs differ from current calibration inputs")
    if provenance["implementation"] != implementation_source_manifest():
        raise ValueError("frozen weight Task 8 implementation source is stale")
    current_executed_sources, _ = _calibration_executed_source_bundle()
    if provenance["executed_source_bundle"] != current_executed_sources:
        raise ValueError("frozen weight executed-source bundle is stale")
    registry_module._verify_snapshot_unchanged(
        stage_snapshot,
        label="current calibration stage",
        root=output_root,
    )


def _validated_stage_context(
    stage: str,
    context: StageContext,
) -> Mapping[str, object]:
    """Validate Task 9/10 identities before constructing a matrix."""

    required_upstream(stage)
    if not isinstance(context, StageContext):
        raise ValueError("context must be a StageContext")
    output_root = _canonical_output_root(context.output_root)
    expected_master_path = output_root / "experiment_manifest.json"
    expected_development_path = output_root / "manifests" / "development_suite.json"
    expected_formal_path = output_root / "manifests" / "formal_suite.json"
    supplied = (
        (Path(context.master_manifest), expected_master_path, "master manifest"),
        (Path(context.development_suite), expected_development_path, "development suite"),
        (Path(context.formal_suite), expected_formal_path, "formal suite"),
    )
    for candidate, expected, label in supplied:
        if (
            not candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.resolve(strict=False) != expected
        ):
            raise ValueError(f"{label} is not at its exact registered output-root path")
    master_path = _canonical_file(context.master_manifest, label="master manifest")
    development = load_bound_suite_identity(context.development_suite, "development")
    formal = load_bound_suite_identity(context.formal_suite, "formal")
    master_snapshot = registry_module._snapshot_file(
        master_path,
        label="master manifest",
        root=master_path.parent,
    )
    master = registry_module._decode_json(master_snapshot.data, label="master manifest")
    if master_snapshot.data != canonical_json_bytes(master):
        raise ValueError("master manifest is not canonical JSON")
    registry_module._exact_keys(
        master,
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
    if master["schema"] != registry_module.MASTER_SCHEMA or master["status"] != registry_module.STATUS:
        raise ValueError("master manifest schema/status is not approved")
    master_semantic = registry_module._digest(
        master["semantic_sha256"], label="master semantic_sha256"
    )
    master_scientific = {
        field: master[field]
        for field in registry_module._MASTER_SCIENTIFIC_FIELDS
    }
    if canonical_hash(master_scientific) != master_semantic:
        raise ValueError("master manifest semantic identity mismatch")
    suites = registry_module._require_mapping(master["suites"], label="master suites")
    registry_module._exact_keys(suites, {"development", "formal"}, label="master suites")
    for role, identity in (("development", development), ("formal", formal)):
        record = registry_module._require_mapping(suites[role], label=f"master {role}")
        registry_module._exact_keys(
            record,
            {"path", "sha256", "semantic_sha256", "task_count", "input_count"},
            label=f"master {role}",
        )
        expected_path = (master_path.parent / str(record["path"])).resolve(strict=True)
        if expected_path != identity.path:
            raise ValueError(f"master {role} suite path identity mismatch")
        if record["sha256"] != identity.sha256 or record["semantic_sha256"] != identity.semantic_sha256:
            raise ValueError(f"master {role} suite hash identity mismatch")
        expected_count = 3 if role == "development" else 10
        if type(record["task_count"]) is not int or record["task_count"] != expected_count:
            raise ValueError(f"master {role} task count mismatch")
        if type(record["input_count"]) is not int or record["input_count"] != 3 * expected_count:
            raise ValueError(f"master {role} input count mismatch")
    spec_record = registry_module._require_mapping(master["design_spec"], label="design spec")
    registry_module._exact_keys(spec_record, {"path", "sha256"}, label="design spec")
    spec_path = _canonical_file(Path(str(spec_record["path"])), label="design specification")
    spec_snapshot = registry_module._snapshot_file(
        spec_path,
        label="design specification",
        root=spec_path.parent,
    )
    if spec_snapshot.sha256 != spec_record["sha256"]:
        raise ValueError("design specification hash changed")
    expected_master = registry_module._master_payload(
        output_root,
        development.path,
        formal.path,
        spec_path,
    )
    if master != expected_master:
        raise ValueError("master manifest differs from the exact approved Task 9 payload")

    expects_weight = stage not in {"bind_manifests", "calibrate128"}
    if not expects_weight:
        if context.frozen_weights is not None:
            raise ValueError(f"{stage} forbids an irrelevant frozen weight")
        weight_identity = None
    else:
        if context.frozen_weights is None:
            raise ValueError(f"{stage} requires frozen weights")
        expected_weight_path = output_root / "analysis" / "weights_frozen.json"
        supplied_weight = Path(context.frozen_weights)
        if (
            not supplied_weight.is_absolute()
            or ".." in supplied_weight.parts
            or supplied_weight.resolve(strict=False) != expected_weight_path
        ):
            raise ValueError(
                "frozen weight is not at the exact output-root analysis path"
            )
        weight_path = _canonical_file(context.frozen_weights, label="frozen weights")
        weight_snapshot = registry_module._snapshot_file(
            weight_path,
            label="frozen weights",
            root=weight_path.parent,
        )
        weight = validate_frozen_weight_config(weight_path)
        if weight.get("schema") != "pcraft_soft_loss_v3_weights_v1" or weight.get("status") != "APPROVED_FROZEN":
            raise ValueError("weight config is not APPROVED_FROZEN")
        lambdas = weight.get("lambdas")
        if not isinstance(lambdas, Mapping) or set(lambdas) != {"lambda_dic", "lambda_dts"}:
            raise ValueError("weight config has malformed lambdas")
        normalized_lambdas: dict[str, float] = {}
        for name in ("lambda_dic", "lambda_dts"):
            value = lambdas[name]
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError("frozen lambdas must be exact positive floats")
            normalized_lambdas[name] = value
        semantic = weight.get("semantic_sha256")
        if type(semantic) is not str or len(semantic) != 64:
            raise ValueError("weight config semantic identity is malformed")
        _validate_weight_causal_lineage(
            weight,
            output_root=output_root,
            master_path=master_path,
            master_sha256=master_snapshot.sha256,
            master_semantic=master_semantic,
            development=development,
            specification_path=spec_path,
            specification_sha256=spec_snapshot.sha256,
        )
        weight_identity = {
            "path": str(weight_path),
            "sha256": weight_snapshot.sha256,
            "semantic_sha256": semantic,
            "status": "APPROVED_FROZEN",
            "lambdas": normalized_lambdas,
        }

    campaign_identity: dict[str, object] | None = None
    campaign_payload: dict[str, object] | None = None
    campaign_snapshot: object | None = None
    if stage not in {"formal256_primary", "formal256_repeat"}:
        if context.formal_campaign is not None:
            raise ValueError(f"{stage} forbids an irrelevant formal campaign")
    else:
        if context.formal_campaign is None:
            raise ValueError(f"{stage} requires the registered formal campaign path")
        expected_campaign = output_root / "manifests" / "formal_campaign.json"
        supplied_campaign = Path(context.formal_campaign)
        if (
            not supplied_campaign.is_absolute()
            or ".." in supplied_campaign.parts
            or supplied_campaign.resolve(strict=False) != expected_campaign
        ):
            raise ValueError("formal campaign path is not the registered output-root path")
        if os.path.lexists(supplied_campaign):
            campaign_identity, campaign_payload, campaign_snapshot = (
                _load_formal_campaign_identity(supplied_campaign, output_root)
            )
        elif stage == "formal256_repeat":
            raise FileNotFoundError("formal repeat requires the frozen formal campaign")

    master_identity = _identity(
        path=master_path,
        sha256=master_snapshot.sha256,
        semantic_sha256=master_semantic,
    )
    development_identity = _identity(
        path=development.path,
        sha256=development.sha256,
        semantic_sha256=development.semantic_sha256,
    )
    formal_identity = _identity(
        path=formal.path,
        sha256=formal.sha256,
        semantic_sha256=formal.semantic_sha256,
    )
    specification_identity = _identity(
        path=spec_path,
        sha256=spec_snapshot.sha256,
    )
    authorization_weight_identity = None
    if weight_identity is not None:
        authorization_weight_identity = {
            key: weight_identity[key]
            for key in ("path", "sha256", "semantic_sha256")
        }
    authorization_identity = _validate_authorization(
        stage,
        context,
        output_root=output_root,
        master_identity=master_identity,
        development_identity=development_identity,
        formal_identity=formal_identity,
        specification_identity=specification_identity,
        weight_identity=authorization_weight_identity,
        campaign_identity=campaign_identity if stage == "formal256_repeat" else None,
        campaign_payload=campaign_payload if stage == "formal256_repeat" else None,
    )

    current_development = load_bound_suite_identity(development.path, "development")
    current_formal = load_bound_suite_identity(formal.path, "formal")
    if current_development != development or current_formal != formal:
        raise ValueError("suite/input dependency changed during context validation")
    registry_module._verify_snapshot_unchanged(
        master_snapshot,
        label="master manifest",
        root=master_path.parent,
    )
    registry_module._verify_snapshot_unchanged(
        spec_snapshot,
        label="design specification",
        root=spec_path.parent,
    )
    if campaign_snapshot is not None:
        registry_module._verify_snapshot_unchanged(
            campaign_snapshot,
            label="formal campaign",
            root=output_root,
        )

    def suite_payload(identity: SuiteIdentity) -> dict[str, object]:
        return {
            "path": str(identity.path),
            "sha256": identity.sha256,
            "semantic_sha256": identity.semantic_sha256,
        }

    tasks: dict[str, object] = {}
    for suite in (development, formal):
        for task in suite.tasks:
            tasks[task.case_id] = {
                "case_id": task.case_id,
                "inputs": [
                    {
                        "axis": item.axis,
                        "source_path": item.source_path,
                        "resolved_path": str(item.resolved_path),
                        "sha256": item.sha256,
                    }
                    for item in task.inputs
                ],
            }
    return {
        "output_root": str(output_root),
        "tasks": tasks,
        "master": master_identity,
        "development": suite_payload(development),
        "formal": suite_payload(formal),
        "specification": specification_identity,
        "weights": weight_identity,
        "authorization": authorization_identity,
        "campaign_payload": campaign_payload,
        "implementation": _ordinary_source_identity(),
    }


def _reject_symlink_components(path: Path, *, label: str) -> None:
    absolute = path if path.is_absolute() else Path.cwd() / path
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")


def _canonical_file(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must be an absolute non-traversing path")
    _reject_symlink_components(candidate, label=label)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {label}: {candidate}") from error
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _canonical_output_root(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("output root must be an absolute non-traversing path")
    existing = candidate
    while not existing.exists():
        if existing.parent == existing:
            raise ValueError("output root has no existing canonical parent")
        existing = existing.parent
    _reject_symlink_components(existing, label="output root")
    if not existing.is_dir():
        raise ValueError("output root parent is not a directory")
    resolved = candidate.resolve(strict=False)
    if os.path.lexists(candidate) and not candidate.is_dir():
        raise ValueError("output root must be a directory")
    return resolved


def _load_formal_campaign_identity(
    path: Path,
    output_root: Path,
) -> tuple[dict[str, object], dict[str, object], object]:
    expected_path = output_root / "manifests" / "formal_campaign.json"
    campaign_path = _canonical_file(Path(path), label="formal campaign")
    if campaign_path != expected_path:
        raise ValueError("formal campaign is not at its registered output-root path")
    snapshot = registry_module._snapshot_file(
        campaign_path,
        label="formal campaign",
        root=output_root,
    )
    payload = registry_module._decode_json(snapshot.data, label="formal campaign")
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError("formal campaign is not canonical JSON")
    registry_module._exact_keys(
        payload,
        {"schema", "status", "primary", "repeats", "pairs", "semantic_sha256"},
        label="formal campaign",
    )
    if (
        payload["schema"] != "pcraft_soft_loss_v3_formal_campaign_v1"
        or payload["status"] != "APPROVED_FROZEN"
    ):
        raise ValueError("formal campaign schema/status is not approved")
    semantic = registry_module._digest(
        payload["semantic_sha256"],
        label="formal campaign semantic_sha256",
    )
    if canonical_hash(
        {key: value for key, value in payload.items() if key != "semantic_sha256"}
    ) != semantic:
        raise ValueError("formal campaign semantic identity mismatch")
    primary = payload["primary"]
    repeats = payload["repeats"]
    pairs = payload["pairs"]
    if not all(isinstance(value, list) and len(value) == 40 for value in (primary, repeats, pairs)):
        raise ValueError("formal campaign requires exact 40 primary/repeat/pair cells")
    seen: set[tuple[str, str]] = set()
    pair_fields = {
        "task_variant",
        "primary_run_id",
        "repeat_run_id",
        "primary_path",
        "repeat_path",
        "scientific_config_hash",
    }
    for index, (first_value, second_value, pair_value) in enumerate(
        zip(primary, repeats, pairs)
    ):
        first = registry_module._require_mapping(first_value, label=f"campaign primary[{index}]")
        second = registry_module._require_mapping(second_value, label=f"campaign repeat[{index}]")
        pair = registry_module._require_mapping(pair_value, label=f"campaign pair[{index}]")
        registry_module._exact_keys(pair, pair_fields, label=f"campaign pair[{index}]")
        cell = (first.get("task"), first.get("variant"))
        if (
            any(type(item) is not str for item in cell)
            or cell in seen
            or second.get("task") != cell[0]
            or second.get("variant") != cell[1]
            or pair["task_variant"] != [cell[0], cell[1]]
            or pair["primary_run_id"] != first.get("run_id")
            or pair["repeat_run_id"] != second.get("run_id")
            or pair["primary_path"] != first.get("path")
            or pair["repeat_path"] != second.get("path")
            or pair["scientific_config_hash"] != first.get("scientific_config_hash")
            or second.get("scientific_config_hash") != first.get("scientific_config_hash")
        ):
            raise ValueError("formal campaign pairing is malformed, duplicate, or non-scientific")
        seen.add((str(cell[0]), str(cell[1])))
    expected_cells = {(task, variant) for task in _FORMAL for variant in _PRIMARY}
    if seen != expected_cells:
        raise ValueError("formal campaign cell matrix is incomplete")
    registry_module._verify_snapshot_unchanged(
        snapshot,
        label="formal campaign",
        root=output_root,
    )
    return (
        {
            "path": str(campaign_path),
            "sha256": snapshot.sha256,
            "semantic_sha256": semantic,
        },
        dict(payload),
        snapshot,
    )


def _identity(
    *,
    path: Path,
    sha256: str,
    semantic_sha256: object | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path), "sha256": sha256}
    if semantic_sha256 is not None:
        if type(semantic_sha256) is not str or len(semantic_sha256) != 64:
            raise ValueError("semantic identity must be a SHA-256 digest")
        result["semantic_sha256"] = semantic_sha256
    return result


def _validate_dependency_record(
    value: object,
    expected: Mapping[str, object] | None,
    *,
    label: str,
) -> None:
    if expected is None:
        if value is not None:
            raise ValueError(f"authorization {label} must be null")
        return
    record = registry_module._require_mapping(value, label=f"authorization {label}")
    registry_module._exact_keys(record, set(expected), label=f"authorization {label}")
    if dict(record) != dict(expected):
        raise ValueError(f"authorization {label} identity is stale or mismatched")


def _validate_source_stage_identity(
    value: object,
    *,
    output_root: Path,
    source_stage: str,
    master_identity: Mapping[str, object] | None = None,
    development_identity: Mapping[str, object] | None = None,
    formal_identity: Mapping[str, object] | None = None,
    specification_identity: Mapping[str, object] | None = None,
    weight_identity: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    record = registry_module._require_mapping(value, label="source stage manifest")
    registry_module._exact_keys(
        record,
        {"path", "sha256", "semantic_sha256"},
        label="source stage manifest",
    )
    expected_path = output_root / "manifests" / "stages" / f"{source_stage}.json"
    path = _canonical_file(Path(str(record["path"])), label="source stage manifest")
    if path != expected_path:
        raise ValueError("authorization source stage path is not the registered path")
    snapshot = registry_module._snapshot_file(
        path,
        label="source stage manifest",
        root=path.parent,
    )
    if snapshot.sha256 != record["sha256"]:
        raise ValueError("authorization source stage hash changed")
    payload = registry_module._decode_json(snapshot.data, label="source stage manifest")
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError("source stage manifest is not canonical JSON")
    registry_module._exact_keys(
        payload,
        {
            "schema",
            "status",
            "stage",
            "authorization",
            "formal_campaign",
            "run_count",
            "runs",
            "semantic_sha256",
        },
        label="source stage manifest",
    )
    if (
        payload.get("schema") != "pcraft_soft_loss_v3_stage_v1"
        or payload.get("status") != "APPROVED_FROZEN"
        or payload.get("stage") != source_stage
    ):
        raise ValueError("authorization source stage schema/status/identity mismatch")
    semantic = payload.get("semantic_sha256")
    if type(semantic) is not str or canonical_hash(
        {key: item for key, item in payload.items() if key != "semantic_sha256"}
    ) != semantic:
        raise ValueError("authorization source stage semantic identity mismatch")
    if semantic != record["semantic_sha256"]:
        raise ValueError("authorization source stage semantic hash is stale")
    registered = payload.get("runs")
    expected_count = len(_expected_cells(source_stage))
    if (
        type(payload.get("run_count")) is not int
        or payload["run_count"] != expected_count
        or not isinstance(registered, list)
        or len(registered) != expected_count
    ):
        raise ValueError("source stage manifest has a non-exact registration count")
    run_fields = {
        "run_id",
        "path",
        "run_spec",
        "scientific_config_hash",
        "expected_config_hash",
        "expected_artifacts",
    }
    specs: list[RunSpec] = []
    for index, raw_run in enumerate(registered):
        run = registry_module._require_mapping(
            raw_run, label=f"source stage run[{index}]"
        )
        registry_module._exact_keys(
            run, run_fields, label=f"source stage run[{index}]"
        )
        raw_spec = registry_module._require_mapping(
            run["run_spec"], label=f"source stage RunSpec[{index}]"
        )
        registry_module._exact_keys(
            raw_spec,
            set(RunSpec.__dataclass_fields__),
            label=f"source stage RunSpec[{index}]",
        )
        values = dict(raw_spec)
        values["output_path"] = Path(str(values["output_path"]))
        for name in (
            "inputs_xyz",
            "dic_schedule",
            "dts_schedule",
            "evaluation_thresholds",
        ):
            if not isinstance(values[name], list):
                raise ValueError(f"source stage RunSpec {name} has wrong exact shape")
            values[name] = tuple(values[name])
        if values["dts_thresholds"] is not None:
            if not isinstance(values["dts_thresholds"], list):
                raise ValueError("source stage RunSpec DTS thresholds have wrong shape")
            values["dts_thresholds"] = tuple(values["dts_thresholds"])
        try:
            spec = RunSpec(**values)
        except (TypeError, ValueError) as error:
            raise ValueError("source stage contains a malformed RunSpec") from error
        if (
            run["run_id"] != spec.run_id
            or run["path"] != str(spec.output_path)
            or run["scientific_config_hash"] != scientific_config_hash(spec)
            or run["expected_artifacts"] != _expected_artifact_namespace(spec)
        ):
            raise ValueError("source stage run differs from its exact RunSpec registration")
        registered_config = registry_module._digest(
            run["expected_config_hash"],
            label=f"source stage run[{index}] expected config hash",
        )
        if expected_config_hash(spec) != registered_config:
            raise ValueError(
                "source stage expected config differs from current Task 8 resolved config"
            )
        if master_identity is not None and dict(spec.master_identity) != dict(master_identity):
            raise ValueError("source stage master identity is stale")
        expected_suite = formal_identity if spec.task in _FORMAL else development_identity
        if expected_suite is not None and dict(spec.suite_identity) != dict(expected_suite):
            raise ValueError("source stage suite identity is stale")
        if specification_identity is not None and dict(spec.specification_identity) != dict(
            specification_identity
        ):
            raise ValueError("source stage specification identity is stale")
        if weight_identity is None:
            if spec.weight_identity is not None:
                raise ValueError("source stage has an unexpected frozen-weight identity")
        else:
            if spec.weight_identity is None or {
                key: spec.weight_identity[key]
                for key in ("path", "sha256", "semantic_sha256")
            } != dict(weight_identity):
                raise ValueError("source stage frozen-weight identity is stale")
        specs.append(spec)
    _validate_stage_spec_sequence(source_stage, specs, output_root=output_root)
    _verify_spec_dependencies(specs)
    if all(
        identity is not None
        for identity in (
            master_identity,
            development_identity,
            formal_identity,
            specification_identity,
        )
    ):
        edge = required_upstream(source_stage)
        current_campaign_identity: dict[str, object] | None = None
        current_campaign_payload: dict[str, object] | None = None
        campaign_path: Path | None = None
        if source_stage in {"formal256_primary", "formal256_repeat"}:
            campaign_path = output_root / "manifests" / "formal_campaign.json"
            current_campaign_identity, current_campaign_payload, _ = (
                _load_formal_campaign_identity(campaign_path, output_root)
            )
            if payload["formal_campaign"] != current_campaign_identity:
                raise ValueError("source stage formal campaign identity is stale")
        elif payload["formal_campaign"] is not None:
            raise ValueError("non-formal source stage has an unexpected campaign identity")
        if edge is None:
            if payload["authorization"] is not None:
                raise ValueError("ungated source stage has an unexpected authorization")
        else:
            prior_path = (
                output_root / "analysis" / "formal_conclusion.json"
                if source_stage == "appendix256"
                else output_root
                / "authorizations"
                / f"{edge[0]}__to__{source_stage}.json"
            )
            prior_context = StageContext(
                output_root=output_root,
                master_manifest=Path(str(master_identity["path"])),
                development_suite=Path(str(development_identity["path"])),
                formal_suite=Path(str(formal_identity["path"])),
                frozen_weights=(
                    None
                    if weight_identity is None
                    else Path(str(weight_identity["path"]))
                ),
                authorization=prior_path,
                formal_campaign=campaign_path,
            )
            current_authorization = _validate_authorization(
                source_stage,
                prior_context,
                output_root=output_root,
                master_identity=master_identity,
                development_identity=development_identity,
                formal_identity=formal_identity,
                specification_identity=specification_identity,
                weight_identity=weight_identity,
                campaign_identity=(
                    current_campaign_identity
                    if source_stage == "formal256_repeat"
                    else None
                ),
                campaign_payload=(
                    current_campaign_payload
                    if source_stage == "formal256_repeat"
                    else None
                ),
            )
            if payload["authorization"] != current_authorization:
                raise ValueError("source stage transitive authorization identity is stale")
    registry_module._verify_snapshot_unchanged(
        snapshot,
        label="source stage manifest",
        root=path.parent,
    )
    return dict(record), dict(payload)


def _validate_completion_only_run(
    path: Path,
    config_hash: str,
    total_steps: int,
    status_sha256: str,
    artifact_manifest_sha256: str,
    *,
    output_root: Path,
) -> None:
    """Validate completion integrity without parsing scientific metric values."""

    if type(total_steps) is not int or total_steps != 2000:
        raise ValueError("formal completion requires the exact final step")
    expected_config = registry_module._digest(config_hash, label="completion config hash")
    expected_status = registry_module._digest(status_sha256, label="completion STATUS hash")
    expected_artifact = registry_module._digest(
        artifact_manifest_sha256,
        label="completion artifact-manifest hash",
    )
    run_dir = Path(path)
    if not run_dir.is_absolute() or ".." in run_dir.parts:
        raise ValueError("completion run path must be absolute and non-traversing")
    if not run_dir.is_relative_to(output_root):
        raise ValueError("completion run path escapes the registered output root")
    _reject_symlink_components(run_dir, label="completion run")
    resolved = run_dir.resolve(strict=True)
    if resolved != run_dir or not resolved.is_dir():
        raise ValueError("completion run path must be a canonical directory")

    from shadowforge_pipeline.soft_voxel_v3 import trainer as task8_trainer

    task8_trainer._preflight_reserved_top_level_paths(resolved)
    status_path = resolved / "STATUS.json"
    status_snapshot = registry_module._snapshot_file(
        status_path,
        label="completed STATUS",
        root=resolved,
    )
    if status_snapshot.sha256 != expected_status:
        raise ValueError("completion STATUS hash is stale")
    status = task8_trainer._status(status_path)
    if (
        status.get("state") != "completed"
        or status.get("config_hash") != expected_config
        or status.get("completed_updates") != total_steps
    ):
        raise ValueError("completion STATUS is not the registered final-step run")
    if status.get("artifact_manifest_sha256") != expected_artifact:
        raise ValueError("completion STATUS artifact-manifest identity mismatch")

    verified = task8_trainer._verify_transaction_chain(
        resolved,
        expected_config_hash=expected_config,
        expected_final_step=total_steps,
    )
    manifest_path = resolved / "artifact_manifest.json"
    manifest_snapshot = registry_module._snapshot_file(
        manifest_path,
        label="completed artifact manifest",
        root=resolved,
    )
    if manifest_snapshot.sha256 != expected_artifact:
        raise ValueError("completion artifact-manifest hash is stale")
    manifest = registry_module._decode_json(
        manifest_snapshot.data,
        label="completed artifact manifest",
    )
    if manifest_snapshot.data != _task8_atomic_json_bytes(manifest):
        raise ValueError("completed artifact manifest does not use Task 8 atomic JSON")
    registry_module._exact_keys(
        manifest,
        {"schema", "config_hash", "artifacts_sha256"},
        label="completed artifact manifest",
    )
    artifacts = registry_module._require_mapping(
        manifest["artifacts_sha256"],
        label="completed artifacts",
    )
    if (
        manifest["schema"] != "pcraft_soft_loss_v3_artifacts_v1"
        or manifest["config_hash"] != expected_config
    ):
        raise ValueError("completed artifact manifest identity mismatch")
    required = set(task8_trainer._required_top_level_artifacts(calibration=False)) | set(
        verified.files
    )
    if set(artifacts) != required:
        raise ValueError("completion artifact manifest has a noncanonical artifact set")
    # This hashes metrics.json as an opaque registered artifact.  It deliberately
    # does not call Task 8's completed-run wrapper, which parses metric values.
    task8_trainer._validate_artifact_hashes(resolved, artifacts)
    _validate_command_output_path(resolved, run_dir)
    registry_module._verify_snapshot_unchanged(
        manifest_snapshot,
        label="completed artifact manifest",
        root=resolved,
    )
    registry_module._verify_snapshot_unchanged(
        status_snapshot,
        label="completed STATUS",
        root=resolved,
    )


def _validate_completion_only(
    completion: object,
    source_stage: Mapping[str, object],
    campaign: Mapping[str, object],
    *,
    output_root: Path,
) -> None:
    """Validate the exact ordered formal-primary completion handoff."""

    if (
        source_stage.get("schema") != "pcraft_soft_loss_v3_stage_v1"
        or source_stage.get("status") != "APPROVED_FROZEN"
        or source_stage.get("stage") != "formal256_primary"
        or type(source_stage.get("run_count")) is not int
        or source_stage.get("run_count") != 40
    ):
        raise ValueError("completion source stage must register exact formal primary")
    registry_module._exact_keys(
        source_stage,
        {
            "schema",
            "status",
            "stage",
            "authorization",
            "formal_campaign",
            "run_count",
            "runs",
            "semantic_sha256",
        },
        label="completion source stage",
    )
    if (
        campaign.get("schema") != "pcraft_soft_loss_v3_formal_campaign_v1"
        or campaign.get("status") != "APPROVED_FROZEN"
    ):
        raise ValueError("completion requires the registered formal campaign")
    registered = source_stage.get("runs")
    campaign_primary = campaign.get("primary")
    if not isinstance(registered, list) or len(registered) != 40:
        raise ValueError("completion source stage must contain exact ordered 40 runs")
    if not isinstance(campaign_primary, list) or len(campaign_primary) != 40:
        raise ValueError("completion campaign must contain exact ordered 40 primary rows")
    if not isinstance(completion, list) or len(completion) != 40:
        raise ValueError("completion must contain the exact ordered 40-cell list")
    fields = {
        "run_id",
        "path",
        "expected_config_hash",
        "status_sha256",
        "artifact_manifest_sha256",
    }
    observed: set[tuple[str, str]] = set()
    registered_fields = {
        "run_id",
        "path",
        "run_spec",
        "scientific_config_hash",
        "expected_config_hash",
        "expected_artifacts",
    }
    campaign_fields = {
        "run_id",
        "path",
        "task",
        "variant",
        "replicate_role",
        "replicate_index",
        "inference_eligible",
        "scientific_config_hash",
        "run_spec",
    }
    expected_cells = _expected_cells("formal256_primary")
    for index, (value, registered_value, campaign_value) in enumerate(
        zip(completion, registered, campaign_primary)
    ):
        record = registry_module._require_mapping(value, label=f"completion[{index}]")
        registry_module._exact_keys(record, fields, label=f"completion[{index}]")
        run = registry_module._require_mapping(
            registered_value,
            label=f"registered formal run[{index}]",
        )
        registry_module._exact_keys(
            run, registered_fields, label=f"registered formal run[{index}]"
        )
        cell = registry_module._require_mapping(
            campaign_value,
            label=f"campaign formal primary[{index}]",
        )
        registry_module._exact_keys(
            cell, campaign_fields, label=f"campaign formal primary[{index}]"
        )
        run_spec = registry_module._require_mapping(
            run["run_spec"], label=f"registered formal RunSpec[{index}]"
        )
        campaign_spec = registry_module._require_mapping(
            cell["run_spec"], label=f"campaign formal RunSpec[{index}]"
        )
        task, variant, role, replicate = expected_cells[index]
        if (
            dict(run_spec) != dict(campaign_spec)
            or run_spec.get("stage") != "formal256_primary"
            or run_spec.get("task") != task
            or run_spec.get("variant") != variant
            or run_spec.get("replicate_role") != role
            or run_spec.get("replicate_index") != replicate
            or run_spec.get("inference_eligible") is not True
            or cell.get("task") != task
            or cell.get("variant") != variant
            or cell.get("replicate_role") != role
            or cell.get("replicate_index") != replicate
            or cell.get("inference_eligible") is not True
            or cell.get("run_id") != run.get("run_id")
            or cell.get("path") != run.get("path")
            or cell.get("scientific_config_hash") != run.get("scientific_config_hash")
        ):
            raise ValueError("completion source stage differs from formal campaign primary")
        for name in ("run_id", "path", "expected_config_hash"):
            if type(record[name]) is not str or record[name] != run.get(name):
                raise ValueError("completion is not the exact ordered registered cell list")
        run_path = Path(str(record["path"]))
        expected_path = output_root / str(record["run_id"])
        if run_path != expected_path or not run_path.is_relative_to(output_root):
            raise ValueError("completion run path escapes its registered output hierarchy")
        identity = (str(record["run_id"]), str(record["path"]))
        if identity in observed:
            raise ValueError("completion contains a duplicate registered cell")
        observed.add(identity)


def _validate_report_semantic(payload: Mapping[str, object], *, label: str) -> None:
    semantic = registry_module._digest(
        payload.get("semantic_sha256"), label=f"{label} semantic_sha256"
    )
    if canonical_hash(
        {key: value for key, value in payload.items() if key != "semantic_sha256"}
    ) != semantic:
        raise ValueError(f"{label} semantic hash mismatch")


def _validate_report_identity(
    value: object,
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    record = registry_module._require_mapping(value, label=label)
    registry_module._exact_keys(record, set(expected), label=label)
    if dict(record) != dict(expected):
        raise ValueError(f"{label} is stale or mismatched")


def _validate_current_summarizer_identity(value: object) -> None:
    record = registry_module._require_mapping(value, label="report summarizer source")
    registry_module._exact_keys(
        record, {"path", "sha256"}, label="report summarizer source"
    )
    path = _canonical_file(_SUMMARIZER_SOURCE, label="Task 12 summarizer source")
    snapshot = registry_module._snapshot_file(
        path, label="Task 12 summarizer source", root=path.parent
    )
    expected = {"path": str(path), "sha256": snapshot.sha256}
    if dict(record) != expected:
        raise ValueError("report summarizer source identity is stale or foreign")
    registry_module._verify_snapshot_unchanged(
        snapshot, label="Task 12 summarizer source", root=path.parent
    )


def _validate_failure_codes(
    value: object,
    *,
    label: str,
    must_be_empty: bool,
) -> list[str]:
    if not isinstance(value, list) or any(type(item) is not str or not item for item in value):
        raise ValueError(f"{label} must be a list of nonempty strings")
    result = list(value)
    if result != sorted(set(result)):
        raise ValueError(f"{label} must be sorted and unique")
    if must_be_empty and result:
        raise ValueError(f"{label} must be empty on success")
    if not must_be_empty and not result:
        raise ValueError(f"{label} must be nonempty on STOP")
    return result


def _exact_int(value: object, expected: int, *, label: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{label} must be exact int {expected}")


def _finite_float(
    value: object,
    *,
    label: str,
    expected: float | None = None,
    nonnegative: bool = False,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite exact float")
    if expected is not None and value != expected:
        raise ValueError(f"{label} has the wrong fixed value")
    if nonnegative and value < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _exact_bool(value: object, expected: bool, *, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{label} must be exact bool {expected}")


def _validate_cuda_branch_checks(value: object, *, label: str) -> None:
    checks = registry_module._require_mapping(value, label=label)
    expected_active = {
        "l12_bal": (False, False),
        "dic_only": (True, False),
        "dts_only": (False, True),
        "v3_full": (True, True),
    }
    registry_module._exact_keys(checks, set(expected_active), label=label)
    fields = {
        "completed_updates",
        "final_progress",
        "expected_dic_active",
        "expected_dts_active",
        "observed_dic_active",
        "observed_dts_active",
        "finite_gradients",
        "soft_only",
    }
    for variant, expected_pair in expected_active.items():
        record = registry_module._require_mapping(
            checks[variant], label=f"{label}.{variant}"
        )
        registry_module._exact_keys(record, fields, label=f"{label}.{variant}")
        _exact_int(
            record["completed_updates"], 2, label=f"{label}.{variant}.completed_updates"
        )
        _finite_float(
            record["final_progress"],
            expected=1.0,
            label=f"{label}.{variant}.final_progress",
        )
        for name, expected in (
            ("expected_dic_active", expected_pair[0]),
            ("expected_dts_active", expected_pair[1]),
            ("observed_dic_active", expected_pair[0]),
            ("observed_dts_active", expected_pair[1]),
            ("finite_gradients", True),
            ("soft_only", True),
        ):
            _exact_bool(record[name], expected, label=f"{label}.{variant}.{name}")


def _validate_gate_result(
    value: object,
    *,
    label: str,
    require_pass: bool,
    expected_gate: str,
    source_stage: str,
) -> None:
    result = registry_module._require_mapping(value, label=label)
    registry_module._exact_keys(
        result,
        {"schema", "gate", "status", "criteria", "observed", "failure_codes"},
        label=label,
    )
    if result["schema"] != "pcraft_soft_loss_v3_gate_result_v1":
        raise ValueError(f"{label} schema mismatch")
    if result["gate"] != expected_gate:
        raise ValueError(f"{label} gate identity differs from the exact expected gate")
    expected_status = "PASS" if require_pass else result["status"]
    if expected_status not in {"PASS", "STOP"} or result["status"] != expected_status:
        raise ValueError(f"{label} status must PASS for ADVANCE")
    _validate_failure_codes(
        result["failure_codes"],
        label=f"{label} failure codes",
        must_be_empty=result["status"] == "PASS",
    )
    if result["status"] != "PASS":
        raise ValueError(f"{label} STOP child cannot authorize ADVANCE")
    criteria = registry_module._require_mapping(
        result["criteria"], label=f"{label} criteria"
    )
    observed = registry_module._require_mapping(
        result["observed"], label=f"{label} observed"
    )

    if expected_gate == "gradient_budget":
        if source_stage not in {"smoke128", "transfer256"}:
            raise ValueError(f"{label} gradient gate is bound to the wrong source stage")
        resolution = 128 if source_stage == "smoke128" else 256
        probe_count = 8 if resolution == 128 else 57
        joint_counts = {"0.04": 1 if resolution == 128 else 14, "0.06": 1}
        criteria_fields = {
            "resolution",
            "ratio_min",
            "ratio_max",
            "combined_median_max",
            "combined_p95_max",
            "adverse_cosine_threshold",
            "adverse_cosine_fraction_max",
            "component_active_l2_epsilon",
            "support_abs_epsilon",
            "eligible_inactive_policy",
        }
        registry_module._exact_keys(criteria, criteria_fields, label=f"{label} criteria")
        _exact_int(criteria["resolution"], resolution, label=f"{label} criteria.resolution")
        for name, expected in (
            ("ratio_min", 0.10),
            ("ratio_max", 0.20),
            ("combined_median_max", 0.35),
            ("combined_p95_max", 1.0),
            ("adverse_cosine_threshold", -0.5),
            ("adverse_cosine_fraction_max", 0.10),
            ("component_active_l2_epsilon", 1e-12),
            ("support_abs_epsilon", 1e-12),
        ):
            _finite_float(
                criteria[name], expected=expected, label=f"{label} criteria.{name}"
            )
        if criteria["eligible_inactive_policy"] != "stop_any_eligible_inactive_sample":
            raise ValueError(f"{label} criteria eligible-inactive policy mismatch")
        observed_fields = {
            "expected_probe_count",
            "observed_probe_count",
            "dic_ratio_median",
            "dts_ratio_median",
            "combined_ratio_median",
            "combined_ratio_p95",
            "adverse_cosine_fraction",
            "endpoint_diagnostics",
        }
        registry_module._exact_keys(observed, observed_fields, label=f"{label} observed")
        _exact_int(
            observed["expected_probe_count"],
            probe_count,
            label=f"{label} observed.expected_probe_count",
        )
        _exact_int(
            observed["observed_probe_count"],
            probe_count,
            label=f"{label} observed.observed_probe_count",
        )
        for name in (
            "dic_ratio_median",
            "dts_ratio_median",
            "combined_ratio_median",
            "combined_ratio_p95",
            "adverse_cosine_fraction",
        ):
            _finite_float(observed[name], label=f"{label} observed.{name}")
        endpoints = registry_module._require_mapping(
            observed["endpoint_diagnostics"], label=f"{label} endpoint diagnostics"
        )
        registry_module._exact_keys(
            endpoints, {"0.04", "0.06"}, label=f"{label} endpoint diagnostics"
        )
        endpoint_fields = {
            "joint_sample_count",
            "dic_dts_cosine_median",
            "dic_dts_support_iou_median",
        }
        for endpoint, expected_count in joint_counts.items():
            record = registry_module._require_mapping(
                endpoints[endpoint], label=f"{label} endpoint {endpoint}"
            )
            registry_module._exact_keys(
                record, endpoint_fields, label=f"{label} endpoint {endpoint}"
            )
            _exact_int(
                record["joint_sample_count"],
                expected_count,
                label=f"{label} endpoint {endpoint}.joint_sample_count",
            )
            for name in ("dic_dts_cosine_median", "dic_dts_support_iou_median"):
                _finite_float(record[name], label=f"{label} endpoint {endpoint}.{name}")
        return

    if expected_gate == "development_factorial":
        if source_stage not in {"screen128", "confirm128", "transfer256"}:
            raise ValueError(f"{label} development gate is bound to the wrong stage")
        screen = source_stage == "screen128"
        criteria_fields = {
            "task_count",
            "dic_relative_threshold",
            "dts_absolute_threshold",
            "iou_noninferiority_delta",
            "recall_floor_delta",
            "fpr_ceiling_delta",
            "require_nonincreasing_range",
        }
        registry_module._exact_keys(criteria, criteria_fields, label=f"{label} criteria")
        _exact_int(criteria["task_count"], 3, label=f"{label} criteria.task_count")
        for name, expected in (
            ("dic_relative_threshold", 0.025 if screen else 0.05),
            ("dts_absolute_threshold", 0.0005 if screen else 0.001),
            ("iou_noninferiority_delta", -0.001),
            ("recall_floor_delta", -0.005),
            ("fpr_ceiling_delta", 0.001),
        ):
            _finite_float(
                criteria[name], expected=expected, label=f"{label} criteria.{name}"
            )
        _exact_bool(
            criteria["require_nonincreasing_range"],
            not screen,
            label=f"{label} criteria.require_nonincreasing_range",
        )
        observed_fields = {
            "dic_task_equal_effect",
            "dic_conditional_task_equal_effects",
            "dts_task_equal_effect",
            "dts_conditional_task_equal_effects",
            "mean_iou_delta_tau_0.05",
            "min_task_recall_delta",
            "mean_fpr_delta_tau_0.05",
            "mean_threshold_range_delta",
            "per_task",
        }
        registry_module._exact_keys(observed, observed_fields, label=f"{label} observed")
        for name in (
            "dic_task_equal_effect",
            "dts_task_equal_effect",
            "mean_iou_delta_tau_0.05",
            "min_task_recall_delta",
            "mean_fpr_delta_tau_0.05",
            "mean_threshold_range_delta",
        ):
            _finite_float(observed[name], label=f"{label} observed.{name}")
        for name, fields in (
            ("dic_conditional_task_equal_effects", {"base_to_dic", "dts_to_full"}),
            ("dts_conditional_task_equal_effects", {"base_to_dts", "dic_to_full"}),
        ):
            values = registry_module._require_mapping(
                observed[name], label=f"{label} observed.{name}"
            )
            registry_module._exact_keys(values, fields, label=f"{label} observed.{name}")
            for field in fields:
                _finite_float(values[field], label=f"{label} observed.{name}.{field}")
        per_task = registry_module._require_mapping(
            observed["per_task"], label=f"{label} observed.per_task"
        )
        registry_module._exact_keys(
            per_task, set(_DEVELOPMENT), label=f"{label} observed.per_task"
        )
        task_fields = {
            "dic_base_to_dic_relative",
            "dic_dts_to_full_relative",
            "dts_base_to_dts_absolute",
            "dts_dic_to_full_absolute",
            "iou_delta_tau_0.05",
            "recall_delta_tau_0.05",
            "fpr_delta_tau_0.05",
            "threshold_range_delta",
        }
        for task in _DEVELOPMENT:
            task_record = registry_module._require_mapping(
                per_task[task], label=f"{label} observed.per_task.{task}"
            )
            registry_module._exact_keys(
                task_record, task_fields, label=f"{label} observed.per_task.{task}"
            )
            for name in task_fields:
                _finite_float(
                    task_record[name], label=f"{label} observed.per_task.{task}.{name}"
                )
        return

    if expected_gate == "cuda_integration":
        if source_stage not in {"cuda2step256", "resource256"}:
            raise ValueError(f"{label} CUDA gate is bound to the wrong source stage")
        registry_module._exact_keys(
            criteria,
            {"run_count", "updates_per_run", "final_progress"},
            label=f"{label} criteria",
        )
        _exact_int(criteria["run_count"], 4, label=f"{label} criteria.run_count")
        _exact_int(
            criteria["updates_per_run"], 2, label=f"{label} criteria.updates_per_run"
        )
        _finite_float(
            criteria["final_progress"],
            expected=1.0,
            label=f"{label} criteria.final_progress",
        )
        registry_module._exact_keys(
            observed, {"branch_checks"}, label=f"{label} observed"
        )
        _validate_cuda_branch_checks(
            observed["branch_checks"], label=f"{label} observed.branch_checks"
        )
        return

    if expected_gate == "resource_feasibility":
        if source_stage != "resource256":
            raise ValueError(f"{label} resource gate is bound to the wrong source stage")
        criteria_fields = {
            "integration_run_count",
            "benchmark_run_count",
            "warmup_updates",
            "timed_updates",
            "peak_memory_fraction_max",
            "full_to_baseline_p95_max",
            "baseline_p95_must_be_positive",
        }
        registry_module._exact_keys(criteria, criteria_fields, label=f"{label} criteria")
        for name, expected in (
            ("integration_run_count", 4),
            ("benchmark_run_count", 2),
            ("warmup_updates", 20),
            ("timed_updates", 180),
        ):
            _exact_int(criteria[name], expected, label=f"{label} criteria.{name}")
        _finite_float(
            criteria["peak_memory_fraction_max"],
            expected=0.90,
            label=f"{label} criteria.peak_memory_fraction_max",
        )
        _finite_float(
            criteria["full_to_baseline_p95_max"],
            expected=3.5,
            label=f"{label} criteria.full_to_baseline_p95_max",
        )
        _exact_bool(
            criteria["baseline_p95_must_be_positive"],
            True,
            label=f"{label} criteria.baseline_p95_must_be_positive",
        )
        observed_fields = {
            "physical_cuda_bytes",
            "peak_cuda_bytes_by_variant",
            "peak_memory_fraction_by_variant",
            "median_seconds_by_variant",
            "p95_seconds_by_variant",
            "full_to_baseline_p95_ratio",
            "integration_branch_checks",
        }
        registry_module._exact_keys(observed, observed_fields, label=f"{label} observed")
        physical = observed["physical_cuda_bytes"]
        if type(physical) is not int or physical <= 0:
            raise ValueError(f"{label} observed physical CUDA bytes must be a positive int")
        variants = {"l12_bal", "v3_full"}
        peak = registry_module._require_mapping(
            observed["peak_cuda_bytes_by_variant"], label=f"{label} observed peak bytes"
        )
        registry_module._exact_keys(peak, variants, label=f"{label} observed peak bytes")
        for variant in variants:
            value = peak[variant]
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} observed peak bytes has wrong exact type")
        for name in (
            "peak_memory_fraction_by_variant",
            "median_seconds_by_variant",
            "p95_seconds_by_variant",
        ):
            values = registry_module._require_mapping(
                observed[name], label=f"{label} observed.{name}"
            )
            registry_module._exact_keys(values, variants, label=f"{label} observed.{name}")
            for variant in variants:
                _finite_float(
                    values[variant],
                    nonnegative=True,
                    label=f"{label} observed.{name}.{variant}",
                )
        _finite_float(
            observed["full_to_baseline_p95_ratio"],
            nonnegative=True,
            label=f"{label} observed.full_to_baseline_p95_ratio",
        )
        _validate_cuda_branch_checks(
            observed["integration_branch_checks"],
            label=f"{label} observed.integration_branch_checks",
        )
        return

    if expected_gate == "repeat_tolerance":
        if source_stage != "sentinel_repeat256":
            raise ValueError(f"{label} repeat gate is bound to the wrong source stage")
        registry_module._exact_keys(
            criteria, {"pair_count", "tolerance", "metrics"}, label=f"{label} criteria"
        )
        _exact_int(criteria["pair_count"], 4, label=f"{label} criteria.pair_count")
        _finite_float(
            criteria["tolerance"], expected=1e-4, label=f"{label} criteria.tolerance"
        )
        if criteria["metrics"] != ["task_mscd_px", "worst_threshold_iou"]:
            raise ValueError(f"{label} criteria metrics mismatch")
        registry_module._exact_keys(
            observed,
            {
                "paired_differences",
                "max_abs_task_mscd_px",
                "max_abs_worst_threshold_iou",
            },
            label=f"{label} observed",
        )
        pairs = registry_module._require_mapping(
            observed["paired_differences"], label=f"{label} observed.paired_differences"
        )
        registry_module._exact_keys(
            pairs, set(_PRIMARY), label=f"{label} observed.paired_differences"
        )
        for variant in _PRIMARY:
            record = registry_module._require_mapping(
                pairs[variant], label=f"{label} observed.paired_differences.{variant}"
            )
            registry_module._exact_keys(
                record,
                {"task_mscd_px", "worst_threshold_iou"},
                label=f"{label} observed.paired_differences.{variant}",
            )
            for name in ("task_mscd_px", "worst_threshold_iou"):
                _finite_float(
                    record[name],
                    label=f"{label} observed.paired_differences.{variant}.{name}",
                )
        for name in ("max_abs_task_mscd_px", "max_abs_worst_threshold_iou"):
            _finite_float(observed[name], nonnegative=True, label=f"{label} observed.{name}")
        return

    raise ValueError(f"{label} uses an unknown or invented Task 12 gate")


def _validate_task12_gate_policy(value: object, *, source_stage: str) -> None:
    policy = registry_module._require_mapping(value, label="report policy")
    registry_module._exact_keys(
        policy,
        {
            "paper_eligible",
            "task_equal_aggregation",
            "fixed_seed_single_run_best_effort",
            "bitwise_reproducible",
            "metric_invalid_policy",
        },
        label="report policy",
    )
    task_equal = source_stage in {"screen128", "confirm128", "transfer256"}
    if (
        policy["paper_eligible"] is not False
        or policy["task_equal_aggregation"] is not task_equal
        or type(policy["task_equal_aggregation"]) is not bool
        or policy["fixed_seed_single_run_best_effort"] is not True
        or policy["bitwise_reproducible"] is not False
        or policy["metric_invalid_policy"] != "fail_closed"
    ):
        raise ValueError("Task 12 gate report policy/task-equal stage binding mismatch")


def _task12_collect_stage_rows(
    root: Path,
    stage: str,
    *,
    source_identity: object,
) -> list[dict[str, object]]:
    """Execute the exact report-bound Task 12 source under a stable snapshot."""

    identity = registry_module._require_mapping(
        source_identity, label="Task 12 collector source identity"
    )
    registry_module._exact_keys(
        identity, {"path", "sha256"}, label="Task 12 collector source identity"
    )
    expected_path = _canonical_file(
        _SUMMARIZER_SOURCE, label="Task 12 summarizer source"
    )
    if identity["path"] != str(expected_path):
        raise ValueError("Task 12 collector path differs from report-bound source")
    expected_sha256 = registry_module._digest(
        identity["sha256"], label="Task 12 collector source sha256"
    )
    snapshot = registry_module._snapshot_file(
        expected_path,
        label="Task 12 collector source",
        root=expected_path.parent,
    )
    if snapshot.sha256 != expected_sha256:
        raise ValueError("Task 12 collector source differs from report-bound bytes")

    module_name = "experiments.summarize_soft_loss_v3"
    module = ModuleType(module_name)
    module.__file__ = str(expected_path)
    module.__package__ = "experiments"
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        code = compile(snapshot.data, str(expected_path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        collector = module.__dict__.get("collect_stage_rows")
        if (
            not callable(collector)
            or getattr(collector, "__globals__", None) is not module.__dict__
            or getattr(getattr(collector, "__code__", None), "co_filename", None)
            != str(expected_path)
            or sys.modules.get(module_name) is not module
        ):
            raise ValueError("Task 12 collector object is not derived from bound source bytes")
        registry_module._verify_snapshot_unchanged(
            snapshot,
            label="Task 12 collector source",
            root=expected_path.parent,
        )
        try:
            rows = collector(root, stage)
        finally:
            registry_module._verify_snapshot_unchanged(
                snapshot,
                label="Task 12 collector source",
                root=expected_path.parent,
            )
        if sys.modules.get(module_name) is not module:
            raise ValueError("Task 12 collector module provenance changed during collection")
    finally:
        if previous is None:
            if sys.modules.get(module_name) is module:
                sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    if not isinstance(rows, list):
        raise ValueError("Task 12 collector returned a non-list result")
    return rows


def _validate_smoke_collector_evidence(
    rows: object,
    evidence: Sequence[Mapping[str, object]],
    collection: Mapping[str, object],
) -> None:
    if not isinstance(rows, list) or len(rows) != 4 or len(evidence) != 4:
        raise ValueError("smoke collector/evidence must contain the exact ordered four rows")
    changed_count = 0
    nested_hashes = {
        "status_sha256": "status",
        "artifact_manifest_sha256": "artifact_manifest",
        "metrics_sha256": "metrics",
        "gradient_audit_sha256": "gradient_audit",
        "resource_metrics_sha256": "resource_metrics",
        "history_sha256": "history",
    }
    for index, (raw_row, item) in enumerate(zip(rows, evidence)):
        row = registry_module._require_mapping(
            raw_row, label=f"smoke Task 12 collected row[{index}]"
        )
        for name in ("run_id", "path", "expected_config_hash"):
            if row.get(name) != item[name] or type(row.get(name)) is not str:
                raise ValueError("smoke collector order/identity differs from report evidence")
        for evidence_name, row_name in nested_hashes.items():
            nested = registry_module._require_mapping(
                row.get(row_name), label=f"smoke collector row[{index}].{row_name}"
            )
            if nested.get("sha256") != item[evidence_name]:
                raise ValueError("smoke collector artifact hash differs from report evidence")
        artifact = registry_module._require_mapping(
            row.get("artifact_manifest"), label="smoke collector artifact manifest"
        )
        artifact_payload = registry_module._require_mapping(
            artifact.get("payload"), label="smoke collector artifact payload"
        )
        artifact_hashes = registry_module._require_mapping(
            artifact_payload.get("artifacts_sha256"),
            label="smoke collector artifact hashes",
        )
        if artifact_hashes.get("occupancy_probability.npy") != item[
            "occupancy_probability_sha256"
        ]:
            raise ValueError("smoke occupancy evidence hash differs from collector")
        parameter_change = registry_module._require_mapping(
            row.get("parameter_change"), label="smoke collector parameter change"
        )
        changed = parameter_change.get("changed")
        if type(changed) is not bool:
            raise ValueError("smoke collector parameter_change.changed must be exact bool")
        changed_count += int(changed)
    reported = collection.get("parameter_changed_count")
    if type(reported) is not int or reported != 4 or reported != changed_count:
        raise ValueError("smoke parameter_changed_count must equal four collector rows")


def _current_stage_identity(output_root: Path, stage: str) -> dict[str, object]:
    path = output_root / "manifests" / "stages" / f"{stage}.json"
    canonical = _canonical_file(path, label=f"{stage} prerequisite stage manifest")
    snapshot = registry_module._snapshot_file(
        canonical, label=f"{stage} prerequisite stage manifest", root=path.parent
    )
    payload = registry_module._decode_json(
        snapshot.data, label=f"{stage} prerequisite stage manifest"
    )
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError(f"{stage} prerequisite stage manifest is not canonical JSON")
    if (
        payload.get("schema") != "pcraft_soft_loss_v3_stage_v1"
        or payload.get("status") != "APPROVED_FROZEN"
        or payload.get("stage") != stage
    ):
        raise ValueError(f"{stage} prerequisite stage identity is malformed")
    semantic = registry_module._digest(
        payload.get("semantic_sha256"),
        label=f"{stage} prerequisite semantic_sha256",
    )
    if canonical_hash(
        {key: value for key, value in payload.items() if key != "semantic_sha256"}
    ) != semantic:
        raise ValueError(f"{stage} prerequisite stage semantic hash mismatch")
    registry_module._verify_snapshot_unchanged(
        snapshot, label=f"{stage} prerequisite stage manifest", root=path.parent
    )
    return {"path": str(canonical), "sha256": snapshot.sha256, "semantic_sha256": semantic}


def _validate_task12_gate_report(
    payload: object,
    *,
    source_stage: str,
    next_stage: str,
    source_identity: Mapping[str, object],
    source_payload: Mapping[str, object],
    output_root: Path,
) -> dict[str, object]:
    """Validate the exact ordinary Task 12 report consumed by an ADVANCE edge."""

    report = registry_module._require_mapping(payload, label="Task 12 gate report")
    registry_module._exact_keys(
        report,
        {
            "schema",
            "status",
            "source_stage",
            "next_stage",
            "gate",
            "source_stage_manifest",
            "prerequisite_stage_manifests",
            "formal_campaign",
            "registered_run_count",
            "completed_run_count",
            "run_evidence",
            "checks",
            "failure_codes",
            "policy",
            "summarizer_source",
            "semantic_sha256",
        },
        label="Task 12 gate report",
    )
    gate_binding = {
        "smoke128": ("gradient_budget", None),
        "screen128": ("development_factorial", None),
        "confirm128": ("development_factorial", None),
        "cuda2step256": ("cuda_integration", None),
        "resource256": ("resource_feasibility", "cuda_integration"),
        "transfer256": ("development_factorial", "gradient_budget"),
        "sentinel_repeat256": ("repeat_tolerance", None),
    }.get(source_stage)
    if gate_binding is None:
        raise ValueError("Task 12 gate report source stage is not registered")
    expected_gate, expected_additional_gate = gate_binding
    if (
        report["schema"] != "pcraft_soft_loss_v3_gate_report_v1"
        or report["status"] != "ADVANCE"
        or report["source_stage"] != source_stage
        or report["next_stage"] != next_stage
        or report["gate"] != expected_gate
        or report["formal_campaign"] is not None
    ):
        raise ValueError("Task 12 gate report schema/status/edge/gate mismatch")
    _validate_report_semantic(report, label="Task 12 gate report")
    _validate_report_identity(
        report["source_stage_manifest"], source_identity, label="report source stage"
    )

    prerequisites = report["prerequisite_stage_manifests"]
    if not isinstance(prerequisites, list):
        raise ValueError("report prerequisite stage manifests must be a list")
    prerequisite_stage = {
        "resource256": "cuda2step256",
        "sentinel_repeat256": "transfer256",
    }.get(source_stage)
    expected_prerequisites = (
        []
        if prerequisite_stage is None
        else [_current_stage_identity(output_root, prerequisite_stage)]
    )
    if prerequisites != expected_prerequisites:
        raise ValueError("report prerequisite stage identities are stale or mismatched")

    registered = source_payload.get("runs")
    if not isinstance(registered, list):
        raise ValueError("report source registration is malformed")
    count = len(registered)
    if (
        type(report["registered_run_count"]) is not int
        or report["registered_run_count"] != count
        or type(report["completed_run_count"]) is not int
        or report["completed_run_count"] != count
    ):
        raise ValueError("Task 12 gate report registered/completed count mismatch")
    evidence = report["run_evidence"]
    if not isinstance(evidence, list) or len(evidence) != count:
        raise ValueError("Task 12 gate report evidence count/order mismatch")
    evidence_fields = {
        "run_id",
        "path",
        "expected_config_hash",
        "completed",
        "status_sha256",
        "artifact_manifest_sha256",
        "metrics_sha256",
        "gradient_audit_sha256",
        "resource_metrics_sha256",
        "history_sha256",
        "occupancy_probability_sha256",
    }
    for index, (raw_evidence, raw_run) in enumerate(zip(evidence, registered)):
        item = registry_module._require_mapping(
            raw_evidence, label=f"Task 12 run evidence[{index}]"
        )
        run = registry_module._require_mapping(
            raw_run, label=f"Task 12 registered run[{index}]"
        )
        registry_module._exact_keys(
            item, evidence_fields, label=f"Task 12 run evidence[{index}]"
        )
        for name in ("run_id", "path", "expected_config_hash"):
            if type(item[name]) is not str or item[name] != run.get(name):
                raise ValueError("Task 12 run evidence differs from source registration")
        if item["completed"] is not True:
            raise ValueError("ADVANCE report requires completed=true for every run")
        for name in evidence_fields - {"run_id", "path", "expected_config_hash", "completed"}:
            registry_module._digest(
                item[name], label=f"Task 12 run evidence[{index}].{name}"
            )

    checks = registry_module._require_mapping(report["checks"], label="report checks")
    registry_module._exact_keys(
        checks, {"collection", "primary_gate", "additional_gate"}, label="report checks"
    )
    collection = registry_module._require_mapping(
        checks["collection"], label="report collection check"
    )
    registry_module._exact_keys(
        collection,
        {
            "status",
            "expected_count",
            "completed_count",
            "parameter_changed_count",
            "invalid_run_ids",
            "failure_codes",
        },
        label="report collection check",
    )
    if (
        collection["status"] != "PASS"
        or type(collection["expected_count"]) is not int
        or collection["expected_count"] != count
        or type(collection["completed_count"]) is not int
        or collection["completed_count"] != count
        or type(collection["parameter_changed_count"]) is not int
        or collection["parameter_changed_count"] < 0
        or not isinstance(collection["invalid_run_ids"], list)
        or collection["invalid_run_ids"]
    ):
        raise ValueError("ADVANCE report collection check must PASS exactly")
    _validate_failure_codes(
        collection["failure_codes"],
        label="report collection failure codes",
        must_be_empty=True,
    )
    if checks["primary_gate"] is None:
        raise ValueError("ADVANCE report requires a primary PASS gate")
    _validate_gate_result(
        checks["primary_gate"],
        label="report primary gate",
        require_pass=True,
        expected_gate=expected_gate,
        source_stage=source_stage,
    )
    if expected_additional_gate is not None:
        if checks["additional_gate"] is None:
            raise ValueError("ADVANCE report requires its additional PASS gate")
        _validate_gate_result(
            checks["additional_gate"],
            label="report additional gate",
            require_pass=True,
            expected_gate=expected_additional_gate,
            source_stage=source_stage,
        )
    elif checks["additional_gate"] is not None:
        raise ValueError("report has an unexpected additional gate")

    _validate_failure_codes(
        report["failure_codes"], label="report failure codes", must_be_empty=True
    )
    _validate_task12_gate_policy(report["policy"], source_stage=source_stage)
    _validate_current_summarizer_identity(report["summarizer_source"])
    if source_stage == "smoke128":
        collector_rows = _task12_collect_stage_rows(
            output_root,
            "smoke128",
            source_identity=report["summarizer_source"],
        )
        _validate_smoke_collector_evidence(collector_rows, evidence, collection)
    return dict(report)


def _validate_task12_completion_report(
    payload: object,
    *,
    source_identity: Mapping[str, object],
    campaign_identity: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact Task 12 formal-primary completion/STOP report."""

    report = registry_module._require_mapping(payload, label="Task 12 completion report")
    registry_module._exact_keys(
        report,
        {
            "schema",
            "status",
            "source_stage",
            "next_stage",
            "source_stage_manifest",
            "formal_campaign",
            "registered_run_count",
            "completed_run_count",
            "completion",
            "metric_values_parsed",
            "failure_codes",
            "policy",
            "summarizer_source",
            "semantic_sha256",
        },
        label="Task 12 completion report",
    )
    status = report["status"]
    if (
        report["schema"] != "pcraft_soft_loss_v3_completion_report_v1"
        or status not in {"COMPLETE_ONLY", "STOP"}
        or report["source_stage"] != "formal256_primary"
        or report["next_stage"] != "formal256_repeat"
    ):
        raise ValueError("Task 12 completion report schema/status/edge mismatch")
    _validate_report_semantic(report, label="Task 12 completion report")
    _validate_report_identity(
        report["source_stage_manifest"], source_identity, label="report source stage"
    )
    _validate_report_identity(
        report["formal_campaign"], campaign_identity, label="report formal campaign"
    )
    if type(report["registered_run_count"]) is not int or report["registered_run_count"] != 40:
        raise ValueError("completion report registered_run_count must be exactly 40")
    completed = report["completed_run_count"]
    if status == "COMPLETE_ONLY":
        if type(completed) is not int or completed != 40:
            raise ValueError("COMPLETE_ONLY completed_run_count must be exactly 40")
        completion = report["completion"]
        if not isinstance(completion, list) or len(completion) != 40:
            raise ValueError("COMPLETE_ONLY report requires exact ordered 40 completion rows")
        fields = {
            "run_id",
            "path",
            "expected_config_hash",
            "status_sha256",
            "artifact_manifest_sha256",
        }
        for index, raw_item in enumerate(completion):
            item = registry_module._require_mapping(
                raw_item, label=f"completion report row[{index}]"
            )
            registry_module._exact_keys(item, fields, label=f"completion report row[{index}]")
            for name in ("run_id", "path"):
                if type(item[name]) is not str or not item[name]:
                    raise ValueError(f"completion report row[{index}] {name} is malformed")
            for name in ("expected_config_hash", "status_sha256", "artifact_manifest_sha256"):
                registry_module._digest(item[name], label=f"completion report row[{index}].{name}")
        _validate_failure_codes(
            report["failure_codes"], label="completion report failure codes", must_be_empty=True
        )
    else:
        if type(completed) is not int or not 0 <= completed <= 39:
            raise ValueError("STOP completed_run_count must be an exact partial integer in [0,39]")
        if report["completion"] is not None:
            raise ValueError("STOP completion report must have null completion")
        stop_codes = _validate_failure_codes(
            report["failure_codes"], label="completion report failure codes", must_be_empty=False
        )
        if not set(stop_codes).issubset(
            {"primary_run_incomplete", "primary_artifact_invalid"}
        ):
            raise ValueError("completion STOP failure code is outside the frozen vocabulary")
    if report["metric_values_parsed"] is not False:
        raise ValueError("completion report metric_values_parsed must be false")
    policy = registry_module._require_mapping(report["policy"], label="completion report policy")
    registry_module._require_exact_value(
        policy,
        {
            "completion_only": True,
            "primary_outcomes_opaque": True,
            "repeats_inference_eligible": False,
        },
        label="completion report policy",
    )
    _validate_current_summarizer_identity(report["summarizer_source"])
    return dict(report)


def _validate_authorization(
    stage: str,
    context: StageContext,
    *,
    output_root: Path,
    master_identity: Mapping[str, object],
    development_identity: Mapping[str, object],
    formal_identity: Mapping[str, object],
    specification_identity: Mapping[str, object],
    weight_identity: Mapping[str, object] | None,
    campaign_identity: Mapping[str, object] | None,
    campaign_payload: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    edge = required_upstream(stage)
    if edge is None:
        if context.authorization is not None:
            raise ValueError(f"{stage} forbids an irrelevant authorization")
        return None
    source_stage, required_status = edge
    if stage == "appendix256":
        if context.authorization is None:
            raise ValueError("appendix256 requires the LOCKED formal conclusion")
        conclusion_path = _canonical_file(
            context.authorization,
            label="formal conclusion",
        )
        expected_conclusion = output_root / "analysis" / "formal_conclusion.json"
        if conclusion_path != expected_conclusion:
            raise ValueError("formal conclusion is not at its registered output-root path")
        conclusion_snapshot = registry_module._snapshot_file(
            conclusion_path,
            label="formal conclusion",
            root=output_root,
        )
        analysis_snapshot = registry_module._snapshot_file(
            output_root / "analysis" / "analysis_manifest.json",
            label="formal analysis manifest",
            root=output_root,
        )
        campaign_identity, _, campaign_snapshot = _load_formal_campaign_identity(
            output_root / "manifests" / "formal_campaign.json",
            output_root,
        )
        try:
            statistics_module = importlib.import_module(
                "experiments.soft_loss_v3_statistics"
            )
            validator = statistics_module.validate_formal_conclusion
        except (ImportError, AttributeError) as error:
            raise ValueError(
                "appendix256 requires Task 13's strict formal-conclusion validator"
            ) from error
        conclusion = validator(
            conclusion_path,
            expected_analysis_sha256=analysis_snapshot.sha256,
            expected_campaign_sha256=campaign_identity["sha256"],
        )
        if not isinstance(conclusion, Mapping) or (
            conclusion.get("schema") != "pcraft_soft_loss_v3_conclusion_v1"
            or conclusion.get("status") != "LOCKED"
        ):
            raise ValueError("appendix256 requires a schema-valid LOCKED conclusion")
        semantic = conclusion.get("semantic_sha256")
        if type(semantic) is not str or len(semantic) != 64:
            raise ValueError("formal conclusion semantic identity is malformed")
        registry_module._verify_snapshot_unchanged(
            analysis_snapshot,
            label="formal analysis manifest",
            root=output_root,
        )
        registry_module._verify_snapshot_unchanged(
            campaign_snapshot,
            label="formal campaign",
            root=output_root,
        )
        registry_module._verify_snapshot_unchanged(
            conclusion_snapshot,
            label="formal conclusion",
            root=output_root,
        )
        return {
            "path": str(conclusion_path),
            "sha256": conclusion_snapshot.sha256,
            "semantic_sha256": semantic,
            "status": required_status,
            "source_stage": source_stage,
            "next_stage": stage,
        }
    if context.authorization is None:
        raise ValueError(f"{stage} requires authorization")
    path = _canonical_file(context.authorization, label="authorization")
    expected_path = output_root / "authorizations" / f"{source_stage}__to__{stage}.json"
    if path != expected_path:
        raise ValueError("authorization path is not the registered DAG edge path")
    snapshot = registry_module._snapshot_file(
        path,
        label="authorization",
        root=path.parent,
    )
    payload = registry_module._decode_json(snapshot.data, label="authorization")
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError("authorization is not canonical JSON")
    registry_module._exact_keys(
        payload,
        {
            "schema",
            "status",
            "source_stage",
            "next_stage",
            "report",
            "dependencies",
            "completion",
            "semantic_sha256",
        },
        label="authorization",
    )
    if payload["schema"] != "pcraft_soft_loss_v3_authorization_v1":
        raise ValueError("authorization schema mismatch")
    if payload["status"] != required_status:
        raise ValueError(
            f"authorization status must be {required_status}; STOP/wrong status cannot advance"
        )
    if payload["source_stage"] != source_stage or payload["next_stage"] != stage:
        raise ValueError("authorization source/next stage edge mismatch")
    if canonical_hash(
        {key: item for key, item in payload.items() if key != "semantic_sha256"}
    ) != payload["semantic_sha256"]:
        raise ValueError("authorization semantic hash mismatch")
    if required_status == "ADVANCE" and payload["completion"] is not None:
        raise ValueError("ADVANCE authorization completion must be null")

    report = registry_module._require_mapping(payload["report"], label="authorization report")
    registry_module._exact_keys(report, {"path", "sha256", "schema"}, label="authorization report")
    report_path = _canonical_file(Path(str(report["path"])), label="authorization report")
    if required_status == "ADVANCE":
        expected_report_path = output_root / "analysis" / f"{source_stage}_gate.json"
        expected_report_schema = "pcraft_soft_loss_v3_gate_report_v1"
    else:
        expected_report_path = output_root / "analysis" / "formal_primary_completion.json"
        expected_report_schema = "pcraft_soft_loss_v3_completion_report_v1"
    if report_path != expected_report_path:
        raise ValueError("authorization report is not at its exact analysis path")
    if report.get("schema") != expected_report_schema:
        raise ValueError("authorization report has the wrong registered schema")
    report_snapshot = registry_module._snapshot_file(
        report_path,
        label="authorization report",
        root=output_root,
    )
    if report_snapshot.sha256 != report["sha256"]:
        raise ValueError("authorization report hash changed")
    report_payload = registry_module._decode_json(
        report_snapshot.data,
        label="authorization report",
    )
    if report_snapshot.data != canonical_json_bytes(report_payload):
        raise ValueError("authorization report is not canonical JSON")
    if (
        report_payload.get("schema") != expected_report_schema
        or report_payload.get("status") != required_status
        or report_payload.get("source_stage") != source_stage
        or report_payload.get("next_stage") != stage
    ):
        raise ValueError("authorization report schema identity mismatch")

    dependencies = registry_module._require_mapping(
        payload["dependencies"], label="authorization dependencies"
    )
    registry_module._exact_keys(
        dependencies,
        {
            "master_manifest",
            "development_suite",
            "formal_suite",
            "specification",
            "frozen_weights",
            "source_stage_manifest",
            "formal_campaign",
        },
        label="authorization dependencies",
    )
    _validate_dependency_record(
        dependencies["master_manifest"], master_identity, label="master manifest"
    )
    _validate_dependency_record(
        dependencies["development_suite"], development_identity, label="development suite"
    )
    _validate_dependency_record(
        dependencies["formal_suite"], formal_identity, label="formal suite"
    )
    _validate_dependency_record(
        dependencies["specification"], specification_identity, label="specification"
    )
    _validate_dependency_record(
        dependencies["frozen_weights"], weight_identity, label="frozen weights"
    )
    current_source_identity, source_stage_payload = _validate_source_stage_identity(
        dependencies["source_stage_manifest"],
        output_root=output_root,
        source_stage=source_stage,
        master_identity=master_identity,
        development_identity=development_identity,
        formal_identity=formal_identity,
        specification_identity=specification_identity,
        weight_identity=weight_identity,
    )
    _validate_dependency_record(
        dependencies["formal_campaign"], campaign_identity, label="formal campaign"
    )
    if required_status == "ADVANCE":
        _validate_task12_gate_report(
            report_payload,
            source_stage=source_stage,
            next_stage=stage,
            source_identity=current_source_identity,
            source_payload=source_stage_payload,
            output_root=output_root,
        )
    else:
        if campaign_identity is None:
            raise ValueError("COMPLETE_ONLY requires the registered formal campaign")
        validated_report = _validate_task12_completion_report(
            report_payload,
            source_identity=current_source_identity,
            campaign_identity=campaign_identity,
        )
        if canonical_json_bytes(
            {"completion": validated_report["completion"]}
        ) != canonical_json_bytes({"completion": payload["completion"]}):
            raise ValueError(
                "completion report and authorization completion are not data-equivalent"
            )
        current_campaign = campaign_payload
        if current_campaign is None:
            _, loaded_campaign, _ = _load_formal_campaign_identity(
                Path(str(campaign_identity["path"])), output_root
            )
            current_campaign = loaded_campaign
        completion_rows = validated_report["completion"]
        assert isinstance(completion_rows, list)
        for raw_record in completion_rows:
            record = registry_module._require_mapping(
                raw_record, label="validated completion-only row"
            )
            _validate_completion_only_run(
                Path(str(record["path"])),
                str(record["expected_config_hash"]),
                2000,
                str(record["status_sha256"]),
                str(record["artifact_manifest_sha256"]),
                output_root=output_root,
            )
        _validate_completion_only(
            payload["completion"],
            source_stage_payload,
            current_campaign,
            output_root=output_root,
        )
    registry_module._verify_snapshot_unchanged(
        report_snapshot,
        label="authorization report",
        root=output_root,
    )
    registry_module._verify_snapshot_unchanged(
        snapshot,
        label="authorization",
        root=path.parent,
    )
    return {
        "path": str(path),
        "sha256": snapshot.sha256,
        "semantic_sha256": payload["semantic_sha256"],
        "status": payload["status"],
        "source_stage": source_stage,
        "next_stage": stage,
    }


def _task_inputs(
    identity: Mapping[str, object], task: str
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    tasks = identity.get("tasks")
    if not isinstance(tasks, Mapping) or task not in tasks:
        raise ValueError(f"registered task is missing from suite identity: {task}")
    task_record = tasks[task]
    if not isinstance(task_record, Mapping):
        raise ValueError(f"registered task identity is malformed: {task}")
    raw_inputs = task_record.get("inputs")
    if not isinstance(raw_inputs, list) or len(raw_inputs) != 3:
        raise ValueError(f"registered task must contain exact x/y/z inputs: {task}")
    inputs = tuple(raw_inputs)
    if tuple(item.get("axis") for item in inputs if isinstance(item, Mapping)) != (
        "x",
        "y",
        "z",
    ):
        raise ValueError(f"registered task inputs are not in x/y/z order: {task}")
    return inputs  # type: ignore[return-value]


def build_stage_specs(stage: str, context: StageContext) -> list[RunSpec]:
    """Construct the exact Cartesian cells for one independently invoked stage."""

    required_upstream(stage)
    if not isinstance(context, StageContext):
        raise ValueError("context must be a StageContext")
    identity = _validated_stage_context(stage, context)
    tasks, variants, role, replicate_index, resolution, steps = _STAGE_MATRIX[stage]
    if stage == "bind_manifests":
        return []
    weights = identity.get("weights")
    if stage == "calibrate128":
        if context.frozen_weights is not None or weights is not None:
            raise ValueError("calibrate128 forbids a frozen weight identity")
        lambdas = None
        weight_identity = None
    else:
        if not isinstance(weights, Mapping):
            raise ValueError("post-calibration stages require frozen weights")
        raw_lambdas = weights.get("lambdas")
        if not isinstance(raw_lambdas, Mapping) or set(raw_lambdas) != {
            "lambda_dic",
            "lambda_dts",
        }:
            raise ValueError("frozen weight identity has malformed lambdas")
        lambdas = {
            "lambda_dic": float(raw_lambdas["lambda_dic"]),
            "lambda_dts": float(raw_lambdas["lambda_dts"]),
        }
        weight_identity = dict(weights)

    specs: list[RunSpec] = []
    for task in tasks:
        inputs = _task_inputs(identity, task)
        suite_key = "formal" if task in _FORMAL else "development"
        suite_identity = identity.get(suite_key)
        if not isinstance(suite_identity, Mapping):
            raise ValueError(f"missing {suite_key} suite identity")
        for variant in variants:
            l1, l2, use_dic, use_dts, dts_thresholds = _PROFILES[variant]
            output_path = (
                Path(context.output_root)
                / stage
                / f"steps_{steps}"
                / task
                / variant
                / f"replicate_{replicate_index}"
                / "seed_17"
            )
            run_id = output_path.relative_to(context.output_root).as_posix()
            specs.append(
                RunSpec(
                    stage=stage,
                    task=task,
                    variant=variant,
                    profile=variant,
                    replicate_role=role,
                    replicate_index=replicate_index,
                    inputs_xyz=inputs,
                    resolution=resolution,
                    export_resolution=resolution,
                    steps=steps,
                    output_path=output_path,
                    run_id=run_id,
                    device="cuda:0",
                    optimizer="Adam",
                    learning_rate=0.007,
                    initialization_logit=-4.0,
                    gradient_clip=1.0,
                    seed=17,
                    l1_weight=l1,
                    l2_weight=l2,
                    use_dic=use_dic,
                    use_dts=use_dts,
                    dic_threshold=0.05,
                    dts_thresholds=dts_thresholds,
                    dic_schedule=(0.65, 0.80),
                    dts_schedule=(0.50, 0.65),
                    evaluation_thresholds=(0.04, 0.05, 0.06),
                    mirror_mode=1,
                    tsf_mode=1,
                    tsf4_mode=0,
                    camera_mode="ortho",
                    image_size=256,
                    control_resolution=0,
                    zdist=1.7,
                    volume_extent=1.7,
                    samples_per_ray=150,
                    final_step_export=True,
                    soft_only=True,
                    frozen_lambdas=lambdas,
                    master_identity=dict(identity["master"]),
                    suite_identity=dict(suite_identity),
                    specification_identity=dict(identity["specification"]),
                    source_identity=dict(identity["implementation"]),
                    weight_identity=weight_identity,
                    inference_eligible=stage == "formal256_primary",
                )
            )
    campaign_payload = identity.get("campaign_payload")
    if campaign_payload is not None:
        campaign = registry_module._require_mapping(
            campaign_payload,
            label="formal campaign projection",
        )
        key = "primary" if stage == "formal256_primary" else "repeats"
        if campaign.get(key) != [_campaign_cell(spec) for spec in specs]:
            raise ValueError(f"{stage} RunSpecs differ from the frozen formal campaign")
    elif stage == "formal256_repeat" and identity.get("campaign_payload", object()) is None:
        raise ValueError("formal repeat requires a validated frozen campaign")
    return specs


def _positive_max_runs(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("max-runs must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("max-runs must be a positive integer")
    return parsed


def build_runner_parser() -> argparse.ArgumentParser:
    """Build the one-stage, non-abbreviating runner CLI."""

    parser = argparse.ArgumentParser(
        description="Run one immutable soft-loss-v3 stage",
        allow_abbrev=False,
    )
    parser.add_argument("--stage", choices=tuple(_STAGE_MATRIX), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--development-suite", type=Path, required=True)
    parser.add_argument("--formal-suite", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--formal-campaign", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=_positive_max_runs)
    return parser


def _scientific_projection(spec: RunSpec) -> dict[str, object]:
    """Positive whitelist of fields capable of changing one scientific run."""

    return {
        "task": spec.task,
        "variant": spec.variant,
        "profile": spec.profile,
        "inputs_xyz": [dict(item) for item in spec.inputs_xyz],
        "resolution": spec.resolution,
        "export_resolution": spec.export_resolution,
        "steps": spec.steps,
        "device": spec.device,
        "optimizer": spec.optimizer,
        "learning_rate": spec.learning_rate,
        "initialization_logit": spec.initialization_logit,
        "gradient_clip": spec.gradient_clip,
        "seed": spec.seed,
        "l1_weight": spec.l1_weight,
        "l2_weight": spec.l2_weight,
        "use_dic": spec.use_dic,
        "use_dts": spec.use_dts,
        "dic_threshold": spec.dic_threshold,
        "dts_thresholds": spec.dts_thresholds,
        "dic_schedule": spec.dic_schedule,
        "dts_schedule": spec.dts_schedule,
        "evaluation_thresholds": spec.evaluation_thresholds,
        "mirror_mode": spec.mirror_mode,
        "tsf_mode": spec.tsf_mode,
        "tsf4_mode": spec.tsf4_mode,
        "camera_mode": spec.camera_mode,
        "image_size": spec.image_size,
        "control_resolution": spec.control_resolution,
        "zdist": spec.zdist,
        "volume_extent": spec.volume_extent,
        "samples_per_ray": spec.samples_per_ray,
        "final_step_export": spec.final_step_export,
        "soft_only": spec.soft_only,
        "frozen_lambdas": spec.frozen_lambdas,
        "master_identity": spec.master_identity,
        "suite_identity": spec.suite_identity,
        "specification_identity": spec.specification_identity,
        "source_identity": spec.source_identity,
        "weight_identity": spec.weight_identity,
    }


def scientific_config_hash(spec: RunSpec) -> str:
    """Hash exactly the registered scientific whitelist for a run cell."""

    if not isinstance(spec, RunSpec):
        raise ValueError("scientific_config_hash requires a RunSpec")
    return canonical_hash(_scientific_projection(spec))


def _identity_path(identity: Mapping[str, object], *, label: str) -> str:
    value = identity.get("path")
    if type(value) is not str or not value:
        raise ValueError(f"{label} identity has no canonical path")
    return value


def trainer_argv(spec: RunSpec, *, resume: bool = False) -> list[str]:
    """Return the exact argument vector for the sole local Task 8 trainer."""

    if not isinstance(spec, RunSpec):
        raise ValueError("trainer_argv requires a RunSpec")
    if type(resume) is not bool:
        raise ValueError("resume must be an exact bool")
    lambda_dic = 0.0 if spec.frozen_lambdas is None else spec.frozen_lambdas["lambda_dic"]
    lambda_dts = 0.0 if spec.frozen_lambdas is None else spec.frozen_lambdas["lambda_dts"]
    argv = [
        str(_LOCKED_INTERPRETER),
        "-m",
        "shadowforge_pipeline.soft_voxel_v3.trainer",
        spec.device,
        str(spec.output_path),
        str(spec.steps),
        repr(spec.learning_rate),
        "--shadow-files",
        *(str(item["resolved_path"]) for item in spec.inputs_xyz),
        "--mirror-mode",
        str(spec.mirror_mode),
        "--tsf-mode",
        str(spec.tsf_mode),
        "--tsf4-mode",
        str(spec.tsf4_mode),
        "--camera-mode",
        spec.camera_mode,
        "--image-size",
        str(spec.image_size),
        "--res",
        str(spec.resolution),
        "--export-res",
        str(spec.export_resolution),
        "--control-res",
        str(spec.control_resolution),
        "--sadekar-zdist",
        repr(spec.zdist),
        "--sadekar-volume-extent",
        repr(spec.volume_extent),
        "--sadekar-npts-per-ray",
        str(spec.samples_per_ray),
        "--seed",
        str(spec.seed),
        "--profile",
        spec.profile,
        "--lambda-dic",
        repr(float(lambda_dic)),
        "--lambda-dts",
        repr(float(lambda_dts)),
        "--suite-manifest",
        _identity_path(spec.suite_identity, label="suite"),
        "--spec",
        _identity_path(spec.specification_identity, label="specification"),
    ]
    if spec.weight_identity is None:
        argv.append("--calibration-probe")
    else:
        argv.extend(
            (
                "--weight-config",
                _identity_path(spec.weight_identity, label="weight"),
            )
        )
    argv.extend(("--export-final-step", "--soft-only-eval"))
    if resume:
        argv.append("--resume")
    return argv


def expected_config_hash(spec: RunSpec) -> str:
    """Derive the hash through the real Task 8 parser/resolver implementation."""

    _require_locked_parent_interpreter()
    from shadowforge_pipeline.soft_voxel_v3 import trainer as task8_trainer

    with _controlled_parent_environment():
        parsed = task8_trainer.build_parser().parse_args(trainer_argv(spec)[3:])
        return canonical_hash(task8_trainer.resolved_config(parsed))


def _run_spec_payload(spec: RunSpec) -> dict[str, object]:
    return {
        "stage": spec.stage,
        "task": spec.task,
        "variant": spec.variant,
        "profile": spec.profile,
        "replicate_role": spec.replicate_role,
        "replicate_index": spec.replicate_index,
        "inputs_xyz": [dict(item) for item in spec.inputs_xyz],
        "resolution": spec.resolution,
        "export_resolution": spec.export_resolution,
        "steps": spec.steps,
        "output_path": str(spec.output_path),
        "run_id": spec.run_id,
        "device": spec.device,
        "optimizer": spec.optimizer,
        "learning_rate": spec.learning_rate,
        "initialization_logit": spec.initialization_logit,
        "gradient_clip": spec.gradient_clip,
        "seed": spec.seed,
        "l1_weight": spec.l1_weight,
        "l2_weight": spec.l2_weight,
        "use_dic": spec.use_dic,
        "use_dts": spec.use_dts,
        "dic_threshold": spec.dic_threshold,
        "dts_thresholds": spec.dts_thresholds,
        "dic_schedule": spec.dic_schedule,
        "dts_schedule": spec.dts_schedule,
        "evaluation_thresholds": spec.evaluation_thresholds,
        "mirror_mode": spec.mirror_mode,
        "tsf_mode": spec.tsf_mode,
        "tsf4_mode": spec.tsf4_mode,
        "camera_mode": spec.camera_mode,
        "image_size": spec.image_size,
        "control_resolution": spec.control_resolution,
        "zdist": spec.zdist,
        "volume_extent": spec.volume_extent,
        "samples_per_ray": spec.samples_per_ray,
        "final_step_export": spec.final_step_export,
        "soft_only": spec.soft_only,
        "frozen_lambdas": spec.frozen_lambdas,
        "master_identity": spec.master_identity,
        "suite_identity": spec.suite_identity,
        "specification_identity": spec.specification_identity,
        "source_identity": spec.source_identity,
        "weight_identity": spec.weight_identity,
        "inference_eligible": spec.inference_eligible,
    }


def _expected_artifact_namespace(spec: RunSpec) -> dict[str, object]:
    thresholds = ("0p04", "0p05", "0p06")
    top_level = [
        "COMMAND.json",
        "run_manifest.json",
        "environment.json",
        "history.jsonl",
        "metrics.json",
        "gradient_audit.json",
        "resource_metrics.json",
        "occupancy_probability.npy",
        "checkpoint_latest.json",
        "checkpoint_final.pt",
    ]
    top_level.extend(f"occupancy_binary_tau_{threshold}.npy" for threshold in thresholds)
    top_level.extend(
        f"binary_tau_{threshold}_view_{view}.png"
        for threshold in thresholds
        for view in range(6)
    )
    if spec.stage == "calibrate128":
        top_level.extend(
            f"calibration_checkpoint_step_{update:08d}.pt"
            for update in (1000, 1300, 1600)
        )
    return {
        "status": "STATUS.json",
        "artifact_manifest": "artifact_manifest.json",
        "top_level": sorted(top_level),
        "transaction_namespace": "transactions/step_<8 digits>/",
        "transaction_files": [
            "checkpoint.pt",
            "history_segment.jsonl",
            "transaction_manifest.json",
        ],
    }


def _expected_cells(stage: str) -> list[tuple[str, str, str, int]]:
    tasks, variants, role, replicate, _, _ = _STAGE_MATRIX[stage]
    return [
        (task, variant, role, replicate)
        for task in tasks
        for variant in variants
    ]


def _validate_stage_spec_sequence(
    stage: str,
    specs: Sequence[RunSpec],
    *,
    output_root: Path | None = None,
) -> list[RunSpec]:
    required_upstream(stage)
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
        raise ValueError("stage specs must be a sequence")
    checked = list(specs)
    if any(not isinstance(spec, RunSpec) for spec in checked):
        raise ValueError("stage specs must contain only RunSpec values")
    observed = [
        (spec.task, spec.variant, spec.replicate_role, spec.replicate_index)
        for spec in checked
    ]
    expected = _expected_cells(stage)
    if observed != expected or len(set(observed)) != len(observed):
        raise ValueError("stage specs do not match the exact ordered matrix; duplicate or hidden cell")
    _, _, role, replicate, resolution, steps = _STAGE_MATRIX[stage]
    roots: set[Path] = set()
    for spec in checked:
        if (
            spec.stage != stage
            or spec.resolution != resolution
            or spec.export_resolution != resolution
            or spec.steps != steps
        ):
            raise ValueError("stage RunSpec identity differs from its frozen matrix")
        if type(spec.replicate_index) is not int or type(spec.inference_eligible) is not bool:
            raise ValueError("stage replicate/inference identity has wrong exact type")
        if spec.inference_eligible is not (stage == "formal256_primary"):
            raise ValueError("stage inference eligibility differs from its frozen role")
        l1, l2, use_dic, use_dts, dts_thresholds = _PROFILES[spec.variant]
        frozen_values: dict[str, object] = {
            "profile": spec.variant,
            "replicate_role": role,
            "replicate_index": replicate,
            "device": "cuda:0",
            "optimizer": "Adam",
            "learning_rate": 0.007,
            "initialization_logit": -4.0,
            "gradient_clip": 1.0,
            "seed": 17,
            "l1_weight": l1,
            "l2_weight": l2,
            "use_dic": use_dic,
            "use_dts": use_dts,
            "dic_threshold": 0.05,
            "dts_thresholds": dts_thresholds,
            "dic_schedule": (0.65, 0.80),
            "dts_schedule": (0.50, 0.65),
            "evaluation_thresholds": (0.04, 0.05, 0.06),
            "mirror_mode": 1,
            "tsf_mode": 1,
            "tsf4_mode": 0,
            "camera_mode": "ortho",
            "image_size": 256,
            "control_resolution": 0,
            "zdist": 1.7,
            "volume_extent": 1.7,
            "samples_per_ray": 150,
            "final_step_export": True,
            "soft_only": True,
        }
        for field, expected_value in frozen_values.items():
            actual = getattr(spec, field)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(f"stage RunSpec {field} differs from its frozen constant")
        if not isinstance(spec.output_path, Path) or not spec.output_path.is_absolute():
            raise ValueError("stage RunSpec output path must be absolute")
        root = output_root if output_root is not None else spec.output_path.parents[5]
        roots.add(root)
        expected_run_id = (
            f"{stage}/steps_{steps}/{spec.task}/{spec.variant}/"
            f"replicate_{replicate}/seed_17"
        )
        if type(spec.run_id) is not str or spec.run_id != expected_run_id:
            raise ValueError("stage RunSpec run ID differs from its frozen hierarchy")
        if spec.output_path != root / expected_run_id:
            raise ValueError("stage RunSpec output path differs from its frozen hierarchy")
        if len(spec.inputs_xyz) != 3 or tuple(
            item.get("axis") if isinstance(item, Mapping) else None
            for item in spec.inputs_xyz
        ) != ("x", "y", "z"):
            raise ValueError("stage RunSpec inputs differ from registered x/y/z order")
        for item in spec.inputs_xyz:
            if set(item) != {"axis", "source_path", "resolved_path", "sha256"}:
                raise ValueError("stage RunSpec input identity is malformed")
            if any(type(item[key]) is not str for key in item):
                raise ValueError("stage RunSpec input identity has wrong exact type")
        if stage == "calibrate128":
            if spec.frozen_lambdas is not None or spec.weight_identity is not None:
                raise ValueError("calibrate128 RunSpec must not bind frozen lambdas or weights")
        else:
            if not isinstance(spec.frozen_lambdas, Mapping) or not isinstance(
                spec.weight_identity, Mapping
            ):
                raise ValueError("post-calibration RunSpec requires frozen lambdas and weights")
            weight_lambdas = spec.weight_identity.get("lambdas")
            if (
                not isinstance(weight_lambdas, Mapping)
                or dict(spec.frozen_lambdas) != dict(weight_lambdas)
                or set(spec.frozen_lambdas) != {"lambda_dic", "lambda_dts"}
                or any(type(value) is not float for value in spec.frozen_lambdas.values())
            ):
                raise ValueError("RunSpec raw frozen lambdas differ from its weight identity")
        for identity in (
            spec.master_identity,
            spec.suite_identity,
            spec.specification_identity,
            spec.source_identity,
        ):
            if not isinstance(identity, Mapping):
                raise ValueError("RunSpec scientific identity is malformed")
    if len(roots) > 1:
        raise ValueError("stage RunSpec paths do not share one output root")
    if len({spec.output_path for spec in checked}) != len(checked) or len(
        {spec.run_id for spec in checked}
    ) != len(checked):
        raise ValueError("stage RunSpec paths and run IDs must be unique")
    return checked


def _verify_identity_record(value: Mapping[str, object], *, label: str) -> None:
    path_value = value.get("path")
    digest = value.get("sha256")
    if type(path_value) is not str or type(digest) is not str:
        raise ValueError(f"{label} identity is malformed")
    path = _canonical_file(Path(path_value), label=label)
    snapshot = registry_module._snapshot_file(path, label=label, root=path.parent)
    if snapshot.sha256 != digest:
        raise ValueError(f"{label} bytes changed")


def _verify_registered_inputs(spec: RunSpec) -> None:
    role = "formal" if spec.task in _FORMAL else "development"
    suite = load_bound_suite_identity(
        Path(str(spec.suite_identity["path"])), role
    )
    task = next((item for item in suite.tasks if item.case_id == spec.task), None)
    if task is None:
        raise ValueError("RunSpec task is not registered in its suite")
    expected = [
        {
            "axis": item.axis,
            "source_path": item.source_path,
            "resolved_path": str(item.resolved_path),
            "sha256": item.sha256,
        }
        for item in task.inputs
    ]
    if canonical_hash({"inputs_xyz": expected}) != canonical_hash(
        {"inputs_xyz": [dict(item) for item in spec.inputs_xyz]}
    ):
        raise ValueError("RunSpec inputs do not match the exact registered suite task row")


def _verify_spec_dependencies(specs: Sequence[RunSpec]) -> None:
    for spec in specs:
        for label, identity in (
            ("master manifest", spec.master_identity),
            ("suite manifest", spec.suite_identity),
            ("specification", spec.specification_identity),
        ):
            _verify_identity_record(identity, label=label)
        if spec.weight_identity is not None:
            _verify_identity_record(spec.weight_identity, label="frozen weights")
        for item in spec.inputs_xyz:
            _verify_identity_record(
                {"path": item["resolved_path"], "sha256": item["sha256"]},
                label=f"{spec.task} {item['axis']} input",
            )
        _verify_registered_inputs(spec)
        if _ordinary_source_identity() != spec.source_identity:
            raise ValueError("ordinary runner implementation/source driver bundle changed")


def _immutable_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    dependencies: Sequence[RunSpec],
    wrapper_snapshots: Sequence[tuple[object, str, Path]] = (),
    validators: Sequence[Callable[[], None]] = (),
) -> Path:
    destination = Path(path)
    if not destination.is_absolute() or ".." in destination.parts:
        raise ValueError("manifest output must be an absolute non-traversing path")
    expected = canonical_json_bytes(payload)

    def precommit() -> None:
        _verify_spec_dependencies(dependencies)
        for validator in validators:
            validator()
        for snapshot, label, root in wrapper_snapshots:
            registry_module._verify_snapshot_unchanged(
                snapshot,
                label=label,
                root=root,
            )

    publication_root = destination.parent.parent if destination.parent.name in {"stages", "manifests"} else destination.parent
    registry_module._publish_bundle(
        {destination: expected},
        output_root=publication_root,
        precommit=precommit,
    )
    return destination


def _stage_wrapper_identities(
    stage: str,
    root: Path,
    specs: Sequence[RunSpec],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if not specs:
        raise ValueError("stage wrapper validation requires registered RunSpecs")
    first = specs[0]
    development = load_bound_suite_identity(
        root / "manifests" / "development_suite.json", "development"
    )
    formal = load_bound_suite_identity(root / "manifests" / "formal_suite.json", "formal")

    def suite_identity(value: SuiteIdentity) -> dict[str, object]:
        return {
            "path": str(value.path),
            "sha256": value.sha256,
            "semantic_sha256": value.semantic_sha256,
        }

    development_identity = suite_identity(development)
    formal_identity = suite_identity(formal)
    for spec in specs:
        expected_suite = formal_identity if spec.task in _FORMAL else development_identity
        if (
            dict(spec.master_identity) != dict(first.master_identity)
            or dict(spec.specification_identity) != dict(first.specification_identity)
            or dict(spec.source_identity) != dict(first.source_identity)
            or dict(spec.suite_identity) != expected_suite
            or (
                None if spec.weight_identity is None else dict(spec.weight_identity)
            )
            != (None if first.weight_identity is None else dict(first.weight_identity))
        ):
            raise ValueError("stage RunSpecs do not share the exact registered identities")

    campaign_identity: dict[str, object] | None = None
    campaign_payload: dict[str, object] | None = None
    campaign_path: Path | None = None
    if stage in {"formal256_primary", "formal256_repeat"}:
        campaign_path = root / "manifests" / "formal_campaign.json"
        campaign_identity, campaign_payload, _ = _load_formal_campaign_identity(
            campaign_path, root
        )

    edge = required_upstream(stage)
    authorization_path: Path | None = None
    if edge is not None:
        authorization_path = (
            root / "analysis" / "formal_conclusion.json"
            if stage == "appendix256"
            else root / "authorizations" / f"{edge[0]}__to__{stage}.json"
        )
    weight_path = (
        None
        if first.weight_identity is None
        else Path(str(first.weight_identity["path"]))
    )
    context = StageContext(
        output_root=root,
        master_manifest=Path(str(first.master_identity["path"])),
        development_suite=development.path,
        formal_suite=formal.path,
        frozen_weights=weight_path,
        authorization=authorization_path,
        formal_campaign=campaign_path,
    )
    authoritative = _validated_stage_context(stage, context)
    expected_weight = authoritative["weights"]
    for spec in specs:
        expected_suite = (
            authoritative["formal"] if spec.task in _FORMAL else authoritative["development"]
        )
        if (
            dict(spec.master_identity) != dict(authoritative["master"])
            or dict(spec.suite_identity) != dict(expected_suite)
            or dict(spec.specification_identity) != dict(authoritative["specification"])
            or dict(spec.source_identity) != dict(authoritative["implementation"])
            or (
                None if spec.weight_identity is None else dict(spec.weight_identity)
            )
            != (None if expected_weight is None else dict(expected_weight))
        ):
            raise ValueError(
                "stage RunSpec source/scientific identity is foreign or non-authoritative"
            )
    validated_campaign = campaign_identity
    if authoritative["campaign_payload"] is not None:
        validated_campaign, _, _ = _load_formal_campaign_identity(
            root / "manifests" / "formal_campaign.json", root
        )
    return authoritative["authorization"], validated_campaign


def _calibration_executed_source_bundle() -> tuple[dict[str, object], tuple[object, ...]]:
    project_root = Path(__file__).resolve().parents[1]
    files: dict[str, object] = {}
    snapshots: list[object] = []
    seen: set[tuple[int, int]] = set()
    for logical in _CALIBRATION_EXECUTED_SOURCES:
        snapshot = registry_module._snapshot_file(
            project_root / logical,
            label=f"calibration executed source {logical}",
            root=project_root,
        )
        current = snapshot.path.stat()
        inode = (current.st_dev, current.st_ino)
        if inode in seen:
            raise ValueError("calibration executed sources must have unique inodes")
        seen.add(inode)
        files[logical] = {"path": str(snapshot.path), "sha256": snapshot.sha256}
        snapshots.append(snapshot)
    return {
        "algorithm": "sha256",
        "files": files,
        "bundle_sha256": canonical_hash(files),
    }, tuple(snapshots)


def _write_calibration_stage(spec: RunSpec, root: Path) -> Path:
    if not is_completed_run(spec.output_path, spec):
        raise ValueError("calibrate128 stage requires its strict completed run")
    run_snapshots = {
        name: registry_module._snapshot_file(
            spec.output_path / name,
            label=f"calibration {name}",
            root=spec.output_path,
        )
        for name in (
            "run_manifest.json",
            "gradient_audit.json",
            "artifact_manifest.json",
            "STATUS.json",
        )
    }
    checkpoints: list[dict[str, object]] = []
    checkpoint_snapshots: list[object] = []
    for update in (1000, 1300, 1600):
        name = f"calibration_checkpoint_step_{update:08d}.pt"
        snapshot = registry_module._snapshot_file(
            spec.output_path / name,
            label=f"calibration checkpoint {update}",
            root=spec.output_path,
        )
        checkpoints.append(
            {"completed_updates": update, "path": name, "sha256": snapshot.sha256}
        )
        checkpoint_snapshots.append(snapshot)
    executed_bundle, source_snapshots = _calibration_executed_source_bundle()
    status_digest = run_snapshots["STATUS.json"].sha256
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_calibration_stage_v1",
        "status": "APPROVED_FROZEN",
        "calibration_run": {
            "path": str(spec.output_path),
            "task": spec.task,
            "profile": spec.profile,
            "resolution": spec.resolution,
            "export_resolution": spec.export_resolution,
            "steps": spec.steps,
            "seed": spec.seed,
            "soft_only": spec.soft_only,
            "calibration_probe": True,
            "config_hash": expected_config_hash(spec),
            "run_manifest_sha256": run_snapshots["run_manifest.json"].sha256,
            "gradient_audit_sha256": run_snapshots["gradient_audit.json"].sha256,
            "artifact_manifest_sha256": run_snapshots["artifact_manifest.json"].sha256,
            "status_sha256": status_digest,
        },
        "checkpoints": checkpoints,
        "master_manifest": dict(spec.master_identity),
        "development_suite": dict(spec.suite_identity),
        "specification": dict(spec.specification_identity),
        "inputs_xyz": [dict(item) for item in spec.inputs_xyz],
        "executed_source_bundle": executed_bundle,
        "publication": {
            "policy": "create_once_after_completed_calibrate128",
            "completed_status_sha256": status_digest,
        },
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    destination = root / "manifests" / "stages" / "calibrate128.json"

    def precommit() -> None:
        _verify_spec_dependencies([spec])
        if not is_completed_run(spec.output_path, spec):
            raise ValueError("calibrate128 run changed from completed state")
        for name, snapshot in run_snapshots.items():
            registry_module._verify_snapshot_unchanged(
                snapshot,
                label=f"calibration {name}",
                root=spec.output_path,
            )
        for snapshot in checkpoint_snapshots:
            registry_module._verify_snapshot_unchanged(
                snapshot,
                label="calibration checkpoint",
                root=spec.output_path,
            )
        project_root = Path(__file__).resolve().parents[1]
        for snapshot in source_snapshots:
            registry_module._verify_snapshot_unchanged(
                snapshot,
                label="calibration executed source",
                root=project_root,
            )

    registry_module._publish_bundle(
        {destination: canonical_json_bytes(payload)},
        output_root=root,
        precommit=precommit,
    )
    return destination


def write_stage_manifest(
    stage: str,
    specs: Sequence[RunSpec],
    output_root: Path,
) -> Path:
    """Create the immutable, outcome-independent registration for one stage."""

    _require_locked_parent_interpreter()
    root = _canonical_output_root(output_root)
    checked = _validate_stage_spec_sequence(stage, specs, output_root=root)
    if stage == "calibrate128":
        if len(checked) != 1:
            raise ValueError("calibrate128 requires its unique registered run")
        return _write_calibration_stage(checked[0], root)
    for spec in checked:
        if not spec.output_path.is_relative_to(root):
            raise ValueError("registered run path escapes output root")
    authorization_identity, campaign_identity = _stage_wrapper_identities(
        stage, root, checked
    )
    runs = [
        {
            "run_id": spec.run_id,
            "path": str(spec.output_path),
            "run_spec": _run_spec_payload(spec),
            "scientific_config_hash": scientific_config_hash(spec),
            "expected_config_hash": expected_config_hash(spec),
            "expected_artifacts": _expected_artifact_namespace(spec),
        }
        for spec in checked
    ]
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_stage_v1",
        "status": "APPROVED_FROZEN",
        "stage": stage,
        "authorization": authorization_identity,
        "formal_campaign": campaign_identity,
        "run_count": len(runs),
        "runs": runs,
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    destination = root / "manifests" / "stages" / f"{stage}.json"
    if (
        checked[0].replicate_role == "technical_repeat"
        and not os.path.lexists(destination)
        and any(os.path.lexists(spec.output_path) for spec in checked)
    ):
        raise ValueError(
            "first technical-repeat stage publication requires all run paths absent"
        )
    expected_wrappers = (authorization_identity, campaign_identity)

    def revalidate_wrappers() -> None:
        if _stage_wrapper_identities(stage, root, checked) != expected_wrappers:
            raise ValueError("stage authorization/campaign wrapper changed")

    return _immutable_json(
        destination,
        payload,
        dependencies=checked,
        validators=(revalidate_wrappers,),
    )


def _campaign_cell(spec: RunSpec) -> dict[str, object]:
    return {
        "run_id": spec.run_id,
        "path": str(spec.output_path),
        "task": spec.task,
        "variant": spec.variant,
        "replicate_role": spec.replicate_role,
        "replicate_index": spec.replicate_index,
        "inference_eligible": spec.inference_eligible,
        "scientific_config_hash": scientific_config_hash(spec),
        "run_spec": _run_spec_payload(spec),
    }


def write_formal_campaign_manifest(
    primary: Sequence[RunSpec],
    repeats: Sequence[RunSpec],
    path: Path,
) -> Path:
    """Freeze the paired 80-cell formal campaign before its first process."""

    _require_locked_parent_interpreter()
    destination = Path(path)
    if not destination.is_absolute() or ".." in destination.parts:
        raise ValueError("formal campaign path must be absolute and non-traversing")
    root = destination.parents[1]
    expected_destination = root / "manifests" / "formal_campaign.json"
    if destination != expected_destination:
        raise ValueError("formal campaign must use its exact registered output-root path")
    checked_primary = _validate_stage_spec_sequence(
        "formal256_primary", primary, output_root=root
    )
    checked_repeats = _validate_stage_spec_sequence(
        "formal256_repeat", repeats, output_root=root
    )
    if len(checked_primary) != 40 or len(checked_repeats) != 40:
        raise ValueError("formal campaign requires exact ordered 40+40 matrices")
    if not os.path.lexists(destination) and any(
        os.path.lexists(spec.output_path)
        for spec in (*checked_primary, *checked_repeats)
    ):
        raise ValueError(
            "first formal campaign publication requires all 80 run paths absent"
        )
    authoritative = _validated_stage_context(
        "formal256_primary",
        StageContext(
            output_root=root,
            master_manifest=root / "experiment_manifest.json",
            development_suite=root / "manifests" / "development_suite.json",
            formal_suite=root / "manifests" / "formal_suite.json",
            frozen_weights=root / "analysis" / "weights_frozen.json",
            authorization=(
                root
                / "authorizations"
                / "sentinel_repeat256__to__formal256_primary.json"
            ),
            formal_campaign=destination,
        ),
    )
    expected_weight = authoritative["weights"]
    for spec in (*checked_primary, *checked_repeats):
        expected_suite = (
            authoritative["formal"] if spec.task in _FORMAL else authoritative["development"]
        )
        if (
            dict(spec.master_identity) != dict(authoritative["master"])
            or dict(spec.suite_identity) != dict(expected_suite)
            or dict(spec.specification_identity) != dict(authoritative["specification"])
            or dict(spec.source_identity) != dict(authoritative["implementation"])
            or (
                None if spec.weight_identity is None else dict(spec.weight_identity)
            )
            != (None if expected_weight is None else dict(expected_weight))
        ):
            raise ValueError(
                "formal campaign contains a foreign/non-authoritative scientific identity"
            )
    pairs: list[dict[str, object]] = []
    for first, second in zip(checked_primary, checked_repeats):
        if (first.task, first.variant) != (second.task, second.variant):
            raise ValueError("formal campaign primary/repeat pairing mismatch")
        first_hash = scientific_config_hash(first)
        if scientific_config_hash(second) != first_hash:
            raise ValueError("formal campaign pair has different scientific identity")
        pairs.append(
            {
                "task_variant": [first.task, first.variant],
                "primary_run_id": first.run_id,
                "repeat_run_id": second.run_id,
                "primary_path": str(first.output_path),
                "repeat_path": str(second.output_path),
                "scientific_config_hash": first_hash,
            }
        )
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_formal_campaign_v1",
        "status": "APPROVED_FROZEN",
        "primary": [_campaign_cell(spec) for spec in checked_primary],
        "repeats": [_campaign_cell(spec) for spec in checked_repeats],
        "pairs": pairs,
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    return _immutable_json(
        destination,
        payload,
        dependencies=[*checked_primary, *checked_repeats],
    )


def _validated_registered_run_path(
    run_dir: Path,
    spec: RunSpec,
    *,
    require_exists: bool,
) -> Path | None:
    candidate = Path(run_dir)
    if candidate != spec.output_path:
        raise ValueError("run directory does not match RunSpec output identity")
    root = _runner_output_root(spec)
    if (
        not candidate.is_absolute()
        or ".." in candidate.parts
        or not candidate.is_relative_to(root)
    ):
        raise ValueError("run path escapes the exact registered output root")
    _reject_symlink_components(candidate, label="registered run path")
    if not os.path.lexists(candidate):
        if require_exists:
            raise FileNotFoundError(f"registered run path is missing: {candidate}")
        return None
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("run directory must be a non-symlink directory")
    resolved = candidate.resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if resolved != candidate or not resolved.is_relative_to(resolved_root):
        raise ValueError("run path resolves outside/differs from its registered path")
    return resolved


def _validate_command_output_path(run_dir: Path, expected_output: Path) -> None:
    from shadowforge_pipeline.soft_voxel_v3 import trainer as task8_trainer

    snapshot = registry_module._snapshot_file(
        run_dir / "COMMAND.json",
        label="completed COMMAND",
        root=run_dir,
    )
    payload = registry_module._decode_json(snapshot.data, label="completed COMMAND")
    if snapshot.data != _task8_atomic_json_bytes(payload):
        raise ValueError("completed COMMAND does not use Task 8 atomic JSON")
    registry_module._exact_keys(
        payload,
        {"schema", "argv", "resolved_config", "config_hash"},
        label="completed COMMAND",
    )
    argv = payload["argv"]
    if (
        payload["schema"] != "pcraft_soft_loss_v3_command_v1"
        or not isinstance(argv, list)
        or not argv
        or any(type(item) is not str for item in argv)
    ):
        raise ValueError("completed COMMAND schema/argv is malformed")
    if argv[:3] == [
        str(_LOCKED_INTERPRETER),
        "-m",
        "shadowforge_pipeline.soft_voxel_v3.trainer",
    ]:
        trainer_args = argv[3:]
    else:
        trainer_args = argv[1:]
    try:
        parsed = task8_trainer.build_parser().parse_args(trainer_args)
    except SystemExit as error:
        raise ValueError("completed COMMAND trainer argv is malformed") from error
    if type(parsed.output_dir) is not str or Path(parsed.output_dir) != expected_output:
        raise ValueError("completed COMMAND output path is not the registered run path")
    registry_module._verify_snapshot_unchanged(
        snapshot,
        label="completed COMMAND",
        root=run_dir,
    )


def is_completed_run(run_dir: Path, spec: RunSpec) -> bool:
    """Return true only after Task 8 strictly validates the complete run."""

    if not isinstance(spec, RunSpec):
        raise ValueError("is_completed_run requires a RunSpec")
    resolved = _validated_registered_run_path(
        Path(run_dir), spec, require_exists=False
    )
    if resolved is None:
        return False
    from shadowforge_pipeline.soft_voxel_v3 import trainer as task8_trainer

    task8_trainer._preflight_reserved_top_level_paths(resolved)
    status = task8_trainer._status(resolved / "STATUS.json")
    expected = expected_config_hash(spec)
    if status.get("config_hash") != expected:
        raise ValueError("run STATUS config identity is stale or mismatched")
    state = status.get("state")
    if state == "running":
        return False
    if state == "failed":
        raise ValueError("failed run is terminal and cannot be reused")
    if state != "completed":
        raise ValueError("run STATUS state is invalid")
    if status.get("completed_updates") != spec.steps:
        raise ValueError("completed run is not at the exact final step")
    task8_trainer._validate_completed_run(
        resolved,
        status,
        expected,
        spec.steps,
        calibration=spec.stage == "calibrate128",
    )
    _validate_command_output_path(resolved, spec.output_path)
    return True


def _validate_compatible_incomplete(run_dir: Path, spec: RunSpec) -> None:
    """Reject stale/corrupt resume state before the runner starts a process."""

    from shadowforge_pipeline.soft_voxel_v3 import trainer as task8_trainer

    resolved = _validated_registered_run_path(
        Path(run_dir), spec, require_exists=True
    )
    assert resolved is not None
    task8_trainer._preflight_reserved_top_level_paths(resolved)
    argv = trainer_argv(spec, resume=True)
    args = task8_trainer.build_parser().parse_args(argv[3:])
    config = task8_trainer.resolved_config(args)
    digest = canonical_hash(config)
    status = task8_trainer._validate_existing_audits(resolved, args, config, digest)
    if status.get("state") != "running" or status.get("config_hash") != digest:
        raise ValueError("resume requires a compatible incomplete running state")
    completed_updates = status.get("completed_updates")
    if (
        type(completed_updates) is not int
        or completed_updates != 0
    ):
        raise ValueError("resume running STATUS must retain the Task 8 step-zero marker")
    verified = task8_trainer._verify_transaction_chain(
        resolved,
        expected_config_hash=digest,
    )
    chain_updates = getattr(verified, "completed_updates", None)
    if (
        type(chain_updates) is not int
        or chain_updates < 0
        or chain_updates >= spec.steps
    ):
        raise ValueError("resume transaction-chain progress is malformed or not incomplete")


def _runner_output_root(spec: RunSpec) -> Path:
    try:
        root = spec.output_path.parents[5]
    except IndexError as error:
        raise ValueError("RunSpec output path does not have the frozen hierarchy") from error
    expected = (
        root
        / spec.stage
        / f"steps_{spec.steps}"
        / spec.task
        / spec.variant
        / f"replicate_{spec.replicate_index}"
        / f"seed_{spec.seed}"
    )
    if spec.output_path != expected:
        raise ValueError("RunSpec output path does not match the frozen hierarchy")
    return root


def _log_paths(spec: RunSpec, output_root: Path) -> tuple[Path, Path]:
    base = (
        output_root
        / "logs"
        / spec.stage
        / f"steps_{spec.steps}"
        / spec.task
        / spec.variant
        / f"replicate_{spec.replicate_index}"
        / f"seed_{spec.seed}"
    )
    return base / "stdout.log", base / "stderr.log"


def _log_root(spec: RunSpec, output_root: Path) -> Path:
    return _log_paths(spec, output_root)[0].parent


@contextmanager
def _single_run_orchestration_lock(spec: RunSpec, output_root: Path):
    base = _log_root(spec, output_root)
    _reject_symlink_components(base, label="run orchestration lock hierarchy")
    base.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(base, label="run orchestration lock hierarchy")
    if not base.resolve(strict=True).is_relative_to(output_root):
        raise ValueError("run orchestration lock hierarchy escapes output root")
    lock_path = base / ".orchestration.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("run orchestration lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("concurrent run orchestration lock is already held") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _attempt_log_paths(spec: RunSpec, output_root: Path) -> tuple[Path, Path]:
    base = _log_root(spec, output_root)
    _reject_symlink_components(base, label="attempt log hierarchy")
    base.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(base, label="attempt log hierarchy")
    attempts: list[int] = []
    for entry in base.iterdir():
        if entry.name in {".orchestration.lock", "stdout.log", "stderr.log"}:
            if entry.is_symlink():
                raise ValueError("historical log/lock path must not be a symlink")
            continue
        if (
            not entry.name.startswith("attempt_")
            or len(entry.name) != len("attempt_000001")
            or not entry.name.removeprefix("attempt_").isdigit()
            or entry.is_symlink()
            or not entry.is_dir()
        ):
            raise ValueError("attempt log hierarchy contains an unregistered entry")
        attempts.append(int(entry.name.removeprefix("attempt_")))
    number = max(attempts, default=0) + 1
    attempt = base / f"attempt_{number:06d}"
    attempt.mkdir(mode=0o700)
    _reject_symlink_components(attempt, label="attempt log directory")
    return attempt / "stdout.log", attempt / "stderr.log"


def _controlled_launch_environment() -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in sorted(_RUNTIME_ENVIRONMENT_ALLOWLIST)
        if name in os.environ
    }
    environment["PYTHONPATH"] = str(_PROJECT_ROOT)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


@contextmanager
def _controlled_parent_environment():
    """Resolve Task 8 config under exactly the child process environment."""

    previous = dict(os.environ)
    controlled = _controlled_launch_environment()
    os.environ.clear()
    os.environ.update(controlled)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _validate_prelaunch_paths(
    spec: RunSpec,
    output_root: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    for path, label in (
        (spec.output_path, "run path"),
        (stdout_path, "stdout log path"),
        (stderr_path, "stderr log path"),
    ):
        if not path.is_absolute() or ".." in path.parts or not path.is_relative_to(output_root):
            raise ValueError(f"{label} escapes the registered output root")
        _reject_symlink_components(path, label=label)
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(output_root):
            raise ValueError(f"{label} resolves outside the registered output root")
    if stdout_path.is_relative_to(spec.output_path) or stderr_path.is_relative_to(
        spec.output_path
    ):
        raise ValueError("runner logs must remain outside run directories")


def _revalidate_runtime_registration(
    stage: str,
    specs: Sequence[RunSpec],
    output_root: Path,
    config_hashes: Mapping[str, str],
    registration_path: Path | None,
) -> None:
    checked = _validate_stage_spec_sequence(stage, specs, output_root=output_root)
    _verify_spec_dependencies(checked)
    for spec in checked:
        expected = registry_module._digest(
            config_hashes.get(spec.run_id),
            label=f"registered config hash for {spec.run_id}",
        )
        if expected_config_hash(spec) != expected:
            raise ValueError("current Task 8 config changed from the registered config")
    if stage == "calibrate128":
        if registration_path is not None:
            raise ValueError("calibration registration must remain post-completion")
        return
    expected_path = output_root / "manifests" / "stages" / f"{stage}.json"
    if registration_path != expected_path:
        raise ValueError("stage registration path differs from the exact registered path")
    snapshot = registry_module._snapshot_file(
        expected_path,
        label="stage registration",
        root=output_root,
    )
    payload = registry_module._decode_json(snapshot.data, label="stage registration")
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError("stage registration is not canonical JSON")
    registry_module._exact_keys(
        payload,
        {
            "schema",
            "status",
            "stage",
            "authorization",
            "formal_campaign",
            "run_count",
            "runs",
            "semantic_sha256",
        },
        label="stage registration",
    )
    if (
        payload["schema"] != "pcraft_soft_loss_v3_stage_v1"
        or payload["status"] != "APPROVED_FROZEN"
        or payload["stage"] != stage
        or payload["run_count"] != len(checked)
        or canonical_hash(
            {key: value for key, value in payload.items() if key != "semantic_sha256"}
        )
        != payload["semantic_sha256"]
    ):
        raise ValueError("stage registration schema/status/semantic identity changed")
    runs = payload["runs"]
    if not isinstance(runs, list) or len(runs) != len(checked):
        raise ValueError("stage registration has a non-exact run list")
    for index, (raw, spec) in enumerate(zip(runs, checked)):
        record = registry_module._require_mapping(
            raw, label=f"stage registration run[{index}]"
        )
        registry_module._exact_keys(
            record,
            {
                "run_id",
                "path",
                "run_spec",
                "scientific_config_hash",
                "expected_config_hash",
                "expected_artifacts",
            },
            label=f"stage registration run[{index}]",
        )
        if (
            record["run_id"] != spec.run_id
            or record["path"] != str(spec.output_path)
            or canonical_hash(
                registry_module._require_mapping(
                    record["run_spec"], label=f"stage registration RunSpec[{index}]"
                )
            )
            != canonical_hash(_run_spec_payload(spec))
            or record["scientific_config_hash"] != scientific_config_hash(spec)
            or record["expected_config_hash"] != config_hashes[spec.run_id]
            or record["expected_artifacts"] != _expected_artifact_namespace(spec)
        ):
            raise ValueError("stage registration differs from the complete RunSpec/config")
    authorization, campaign = _stage_wrapper_identities(stage, output_root, checked)
    if payload["authorization"] != authorization or payload["formal_campaign"] != campaign:
        raise ValueError("stage registration authorization/campaign identity changed")
    registry_module._verify_snapshot_unchanged(
        snapshot,
        label="stage registration",
        root=output_root,
    )


def run_specs(
    specs: Sequence[RunSpec],
    *,
    dry_run: bool,
    resume: bool,
    max_runs: int | None,
) -> dict[str, int]:
    """Register and execute cells in supplied order without shell interpolation."""

    _require_locked_parent_interpreter()
    if type(dry_run) is not bool or type(resume) is not bool:
        raise ValueError("dry_run and resume must be exact bool values")
    if max_runs is not None and (
        type(max_runs) is not int or max_runs <= 0
    ):
        raise ValueError("max_runs must be a positive integer or None")
    if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence) or not specs:
        raise ValueError("run_specs requires a non-empty registered stage sequence")
    checked = list(specs)
    if any(not isinstance(spec, RunSpec) for spec in checked):
        raise ValueError("run_specs accepts only RunSpec values")
    stage = checked[0].stage
    checked = _validate_stage_spec_sequence(stage, checked)
    output_root = _runner_output_root(checked[0])
    if any(_runner_output_root(spec) != output_root for spec in checked):
        raise ValueError("all RunSpecs must share one canonical output root")
    _verify_spec_dependencies(checked)
    config_hashes = {
        spec.run_id: expected_config_hash(spec)
        for spec in checked
    }
    registration_path: Path | None = None
    if stage != "calibrate128":
        registration_path = write_stage_manifest(stage, checked, output_root)

    counts = {
        "registered": len(checked),
        "completed": 0,
        "launched": 0,
        "resumed": 0,
        "dry_run": 0,
        "limited": 0,
    }
    for spec in checked:
        _revalidate_runtime_registration(
            stage, checked, output_root, config_hashes, registration_path
        )
        lock = (
            nullcontext()
            if dry_run
            else _single_run_orchestration_lock(spec, output_root)
        )
        with lock:
            _revalidate_runtime_registration(
                stage, checked, output_root, config_hashes, registration_path
            )
            complete = is_completed_run(spec.output_path, spec)
            if complete:
                counts["completed"] += 1
                _revalidate_runtime_registration(
                    stage, checked, output_root, config_hashes, registration_path
                )
                continue
            if max_runs is not None and counts["launched"] >= max_runs:
                counts["limited"] += 1
                _revalidate_runtime_registration(
                    stage, checked, output_root, config_hashes, registration_path
                )
                continue
            if dry_run:
                counts["dry_run"] += 1
                _revalidate_runtime_registration(
                    stage, checked, output_root, config_hashes, registration_path
                )
                continue
            exists = os.path.lexists(spec.output_path)
            use_resume = False
            if exists:
                if not resume:
                    raise FileExistsError(
                        f"incomplete run exists and --resume was not supplied: {spec.output_path}"
                    )
                _validate_compatible_incomplete(spec.output_path, spec)
                use_resume = True
            stdout_path, stderr_path = _attempt_log_paths(spec, output_root)
            _validate_prelaunch_paths(spec, output_root, stdout_path, stderr_path)
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                _revalidate_runtime_registration(
                    stage, checked, output_root, config_hashes, registration_path
                )
                _validate_prelaunch_paths(spec, output_root, stdout_path, stderr_path)
                completed = subprocess.run(
                    trainer_argv(spec, resume=use_resume),
                    check=False,
                    stdout=stdout,
                    stderr=stderr,
                    cwd=_PROJECT_ROOT,
                    env=_controlled_launch_environment(),
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"trainer failed for {spec.run_id} with exit code {completed.returncode}"
                )
            counts["launched"] += 1
            if use_resume:
                counts["resumed"] += 1
            _revalidate_runtime_registration(
                stage, checked, output_root, config_hashes, registration_path
            )
            completed_after_launch = is_completed_run(spec.output_path, spec)
            _revalidate_runtime_registration(
                stage, checked, output_root, config_hashes, registration_path
            )
            if not completed_after_launch:
                raise RuntimeError(
                    f"trainer returned without a strict completed run: {spec.run_id}"
                )
    _revalidate_runtime_registration(
        stage, checked, output_root, config_hashes, registration_path
    )
    if stage == "calibrate128" and counts["completed"] + counts["launched"] == 1:
        write_stage_manifest(stage, checked, output_root)
    return counts


def _formal_repeat_specs(primary: Sequence[RunSpec], output_root: Path) -> list[RunSpec]:
    checked = _validate_stage_spec_sequence("formal256_primary", primary)
    repeats: list[RunSpec] = []
    for spec in checked:
        path = (
            output_root
            / "formal256_repeat"
            / "steps_2000"
            / spec.task
            / spec.variant
            / "replicate_1"
            / "seed_17"
        )
        repeats.append(
            replace(
                spec,
                stage="formal256_repeat",
                replicate_role="technical_repeat",
                replicate_index=1,
                output_path=path,
                run_id=path.relative_to(output_root).as_posix(),
                inference_eligible=False,
            )
        )
    return _validate_stage_spec_sequence("formal256_repeat", repeats)


def _validate_runner_options(args: argparse.Namespace) -> None:
    stage = args.stage
    needs_weight = stage not in {"bind_manifests", "calibrate128"}
    if needs_weight is (args.weights is None):
        raise ValueError(
            f"{stage} {'requires' if needs_weight else 'forbids an irrelevant'} frozen weight"
        )
    needs_authorization = required_upstream(stage) is not None
    if needs_authorization is (args.authorization is None):
        raise ValueError(
            f"{stage} {'requires' if needs_authorization else 'forbids an irrelevant'} authorization"
        )
    formal = stage in {"formal256_primary", "formal256_repeat"}
    if formal is (args.formal_campaign is None):
        raise ValueError(
            f"{stage} {'requires' if formal else 'forbids an irrelevant'} formal campaign"
        )
    if args.dry_run and args.resume:
        raise ValueError("--resume is irrelevant to --dry-run")


def runner_main(argv: Sequence[str] | None = None) -> int:
    """Validate and run exactly one independently selected registered stage."""

    _require_locked_parent_interpreter()
    if argv is not None and (
        isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence)
    ):
        raise ValueError("argv must be a sequence of argument strings or None")
    args = build_runner_parser().parse_args(argv)
    _validate_runner_options(args)
    output_root = Path(args.output_root)
    if args.formal_campaign is not None:
        expected_campaign = output_root / "manifests" / "formal_campaign.json"
        supplied_campaign = Path(args.formal_campaign)
        if (
            not supplied_campaign.is_absolute()
            or ".." in supplied_campaign.parts
            or supplied_campaign.resolve(strict=False) != expected_campaign.resolve(strict=False)
        ):
            raise ValueError("formal campaign must use the registered output-root path")
    context = StageContext(
        output_root=output_root,
        master_manifest=Path(args.master_manifest),
        development_suite=Path(args.development_suite),
        formal_suite=Path(args.formal_suite),
        frozen_weights=None if args.weights is None else Path(args.weights),
        authorization=None if args.authorization is None else Path(args.authorization),
        formal_campaign=(
            None if args.formal_campaign is None else Path(args.formal_campaign)
        ),
    )
    specs = build_stage_specs(args.stage, context)
    if args.stage == "bind_manifests":
        sys.stdout.write('{"registered":0}\n')
        return 0
    if args.stage == "formal256_primary":
        repeats = _formal_repeat_specs(specs, output_root)
        write_formal_campaign_manifest(specs, repeats, Path(args.formal_campaign))
    counts = run_specs(
        specs,
        dry_run=args.dry_run,
        resume=args.resume,
        max_runs=args.max_runs,
    )
    sys.stdout.write(canonical_json_bytes(counts).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(runner_main())
