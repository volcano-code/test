"""Registered formal inference for the P-CRAFT soft-loss-v3 factorial study.

The module deliberately separates pure, authenticated 40-cell matrix
arithmetic from filesystem publication.  Primary and technical-repeat rows
are never pooled: the former are the ten confirmatory units and the latter are
an audit-only reproduction matrix.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from io import StringIO
import itertools
import json
import math
from numbers import Real
import os
from pathlib import Path
from types import MappingProxyType
from typing import Callable

import numpy as np

import experiments.soft_loss_v3_registry as registry
from shadowforge_pipeline.soft_voxel_v3.provenance import (
    canonical_hash,
    canonical_json_bytes,
    normalize_builtin,
)


_TASKS = tuple(f"connected_{index:02d}" for index in range(1, 11))
_VARIANTS = ("l12_bal", "dic_only", "dts_only", "v3_full")
_ROLES = {
    ("primary", 0): ("formal256_primary", True),
    ("technical_repeat", 1): ("formal256_repeat", False),
}
_ROW_KEYS = {
    "schema",
    "stage",
    "run_id",
    "path",
    "task",
    "variant",
    "replicate_role",
    "replicate_index",
    "inference_eligible",
    "resolution",
    "steps",
    "scientific_config_hash",
    "expected_config_hash",
    "run_spec",
    "stage_manifest",
    "formal_campaign",
    "status",
    "artifact_manifest",
    "metrics",
    "gradient_audit",
    "resource_metrics",
    "history",
    "parameter_change",
}
_CAMPAIGN_CELL_KEYS = {
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
_PAIR_KEYS = {
    "task_variant",
    "primary_run_id",
    "repeat_run_id",
    "primary_path",
    "repeat_path",
    "scientific_config_hash",
}
_IDENTITY_KEYS = {"path", "sha256", "semantic_sha256"}
_DIGEST_LENGTH = 64
_EXPORT_NAMES = (
    "task_rows.tsv",
    "factorial_task_contrasts.tsv",
    "confirmatory_tests.json",
    "conditional_effects.json",
    "interactions.json",
    "safety_checks.json",
    "retention_decision.json",
    "repeat_audit.json",
    "aggregate_metrics.json",
)
_LOCKED_DRAWS = 100_000
_LOCKED_SEED = 20260826
_MATRIX_TOKEN = object()
_HYPOTHESIS_KEYS = {
    "component",
    "endpoint",
    "task_effects",
    "mean_effect",
    "exact_sign_flip_p_greater",
    "holm_adjusted_p",
    "ci_low",
    "ci_high",
    "statistic_valid",
    "significant",
    "ci_above_zero",
    "passed",
    "failure_codes",
}
_SAFETY_CRITERIA = {
    "mean_primary_effect_strict_lower": 0.0,
    "mean_iou_0_05_delta_min": -0.001,
    "per_task_recall_0_05_delta_min": -0.005,
    "mean_fpr_0_05_delta_max": 0.001,
}
_SAFETY_COMPARISON_KEYS = {
    "name",
    "control_variant",
    "candidate_variant",
    "task_primary_effects",
    "mean_primary_effect",
    "task_iou_0_05_effects",
    "mean_iou_0_05_effect",
    "task_recall_0_05_effects",
    "min_task_recall_0_05_effect",
    "task_fpr_0_05_effects",
    "mean_fpr_0_05_effect",
    "checks",
    "invalid_evidence",
}
_SAFETY_CHECK_KEYS = {
    "primary_beneficial",
    "iou_noninferior",
    "recall_noninferior",
    "fpr_guard",
}


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: object, *, label: str) -> list[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a non-string sequence")
    return list(value)


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected or any(type(key) is not str for key in value):
        raise ValueError(f"{label} has extra or missing keys")


def _string(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty exact string")
    return value


def _digest(value: object, *, label: str) -> str:
    text = _string(value, label=label)
    if len(text) != _DIGEST_LENGTH or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _exact_int(value: object, *, label: str, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{label} must be an exact {qualifier}integer")
    return value


def _finite(value: object, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric and not bool")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{label} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _canonical_path(value: object, *, label: str, require_exists: bool = False) -> Path:
    if not isinstance(value, Path):
        raise ValueError(f"{label} must be a Path")
    if not value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{label} must be absolute and non-traversing")
    current = Path(value.anchor)
    for part in value.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")
    if require_exists:
        try:
            resolved = value.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"{label} does not exist") from error
        if resolved != value or not value.is_file():
            raise ValueError(f"{label} must be a canonical regular file")
    elif value.resolve(strict=False) != value:
        raise ValueError(f"{label} is not canonical")
    return value


def _identity(value: object, *, label: str) -> dict[str, object]:
    record = _mapping(value, label=label)
    _exact_keys(record, _IDENTITY_KEYS, label=label)
    path = Path(_string(record["path"], label=f"{label}.path"))
    _canonical_path(path, label=f"{label}.path")
    return {
        "path": str(path),
        "sha256": _digest(record["sha256"], label=f"{label}.sha256"),
        "semantic_sha256": _digest(record["semantic_sha256"], label=f"{label}.semantic_sha256"),
    }


def _plain_identity(value: object, *, label: str) -> dict[str, object]:
    record = _mapping(value, label=label)
    _exact_keys(record, {"path", "sha256"}, label=label)
    path = Path(_string(record["path"], label=f"{label}.path"))
    _canonical_path(path, label=f"{label}.path")
    return {"path": str(path), "sha256": _digest(record["sha256"], label=f"{label}.sha256")}


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _semantic_payload(payload: Mapping[str, object], *, label: str) -> dict[str, object]:
    semantic = _digest(payload.get("semantic_sha256"), label=f"{label}.semantic_sha256")
    body = {key: value for key, value in payload.items() if key != "semantic_sha256"}
    if canonical_hash(body) != semantic:
        raise ValueError(f"{label} semantic SHA-256 is stale")
    return dict(payload)


@dataclass(frozen=True, slots=True)
class FactorialTaskContrast:
    task: str
    dic_marginal_mscd_improvement_px: float
    dts_marginal_worst_threshold_iou_improvement: float
    m_d_interaction_px: float
    m_t_interaction: float
    dic_base_to_dic_mscd_improvement_px: float
    dic_dts_to_full_mscd_improvement_px: float
    dts_base_to_dts_worst_threshold_iou_improvement: float
    dts_dic_to_full_worst_threshold_iou_improvement: float

    def __post_init__(self) -> None:
        _string(self.task, label="contrast task")
        for name in (
            "dic_marginal_mscd_improvement_px",
            "dts_marginal_worst_threshold_iou_improvement",
            "m_d_interaction_px",
            "m_t_interaction",
            "dic_base_to_dic_mscd_improvement_px",
            "dic_dts_to_full_mscd_improvement_px",
            "dts_base_to_dts_worst_threshold_iou_improvement",
            "dts_dic_to_full_worst_threshold_iou_improvement",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), label=name))


@dataclass(frozen=True, slots=True, init=False)
class ValidatedFormalMatrix:
    replicate_role: str
    replicate_index: int
    inference_eligible: bool
    tasks: tuple[str, ...]
    variants: tuple[str, ...]
    row_count: int
    matrix_sha256: str
    formal_suite_identity: Mapping[str, object]
    formal_campaign_identity: Mapping[str, object]
    stage_manifest_sha256: str
    _rows: tuple[Mapping[str, object], ...]
    _campaign_pairs: tuple[Mapping[str, object], ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ValueError("ValidatedFormalMatrix is constructible only by validate_formal_matrix")

    @classmethod
    def _validated(
        cls,
        token: object,
        *,
        replicate_role: str,
        replicate_index: int,
        inference_eligible: bool,
        rows: Sequence[Mapping[str, object]],
        pairs: Sequence[Mapping[str, object]],
        matrix_sha256: str,
        suite_identity: Mapping[str, object],
        campaign_identity: Mapping[str, object],
        stage_manifest_sha256: str,
    ) -> "ValidatedFormalMatrix":
        if token is not _MATRIX_TOKEN:
            raise ValueError("invalid matrix construction token")
        instance = object.__new__(cls)
        for name, value in (
            ("replicate_role", replicate_role),
            ("replicate_index", replicate_index),
            ("inference_eligible", inference_eligible),
            ("tasks", _TASKS),
            ("variants", _VARIANTS),
            ("row_count", 40),
            ("matrix_sha256", matrix_sha256),
            ("formal_suite_identity", _deep_freeze(suite_identity)),
            ("formal_campaign_identity", _deep_freeze(campaign_identity)),
            ("stage_manifest_sha256", stage_manifest_sha256),
            ("_rows", _deep_freeze(rows)),
            ("_campaign_pairs", _deep_freeze(pairs)),
        ):
            object.__setattr__(instance, name, value)
        return instance


def _validate_formal_suite(payload: Mapping[str, object]) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    expected = {
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
        "formal_run_roles",
        "semantic_sha256",
    }
    _exact_keys(payload, expected, label="formal suite")
    if payload["schema"] != "pcraft_soft_loss_v3_suite_v1" or payload["status"] != "APPROVED_FROZEN" or payload["role"] != "formal":
        raise ValueError("formal suite schema/status/role is invalid")
    if payload["task_count"] != 10 or type(payload["task_count"]) is not int:
        raise ValueError("formal suite task count must be exactly ten")
    if payload["input_count"] != 30 or type(payload["input_count"]) is not int:
        raise ValueError("formal suite input count must be exactly thirty")
    if payload["input_order"] != ["x", "y", "z"] or payload["primary_variants"] != list(_VARIANTS):
        raise ValueError("formal suite order or primary variants changed")
    _semantic_payload(payload, label="formal suite")
    raw_tasks = _sequence(payload["tasks"], label="formal suite tasks")
    if len(raw_tasks) != 10:
        raise ValueError("formal suite must contain ten tasks")
    by_task: dict[str, list[dict[str, object]]] = {}
    for index, raw_task in enumerate(raw_tasks):
        task = _mapping(raw_task, label=f"formal suite task[{index}]")
        _exact_keys(task, {"case_id", "inputs"}, label=f"formal suite task[{index}]")
        expected_task = _TASKS[index]
        if task["case_id"] != expected_task:
            raise ValueError("formal suite task order changed")
        inputs = _sequence(task["inputs"], label=f"{expected_task} inputs")
        if len(inputs) != 3:
            raise ValueError("formal suite task must contain x/y/z inputs")
        normalized: list[dict[str, object]] = []
        for axis_index, raw_input in enumerate(inputs):
            item = _mapping(raw_input, label=f"{expected_task} input[{axis_index}]")
            _exact_keys(item, {"axis", "source_path", "resolved_path", "sha256"}, label=f"{expected_task} input[{axis_index}]")
            axis = "xyz"[axis_index]
            if item["axis"] != axis:
                raise ValueError("formal input order is not x/y/z")
            source = _string(item["source_path"], label="formal input source_path")
            resolved = _string(item["resolved_path"], label="formal input resolved_path")
            _canonical_path(Path(resolved), label="formal input resolved_path")
            normalized.append({"axis": axis, "source_path": source, "resolved_path": resolved, "sha256": _digest(item["sha256"], label="formal input sha256")})
        by_task[expected_task] = normalized
    return dict(payload), by_task


def _validate_campaign(payload: Mapping[str, object]) -> tuple[dict[str, object], list[Mapping[str, object]], list[Mapping[str, object]], list[Mapping[str, object]]]:
    _exact_keys(payload, {"schema", "status", "primary", "repeats", "pairs", "semantic_sha256"}, label="formal campaign")
    if payload["schema"] != "pcraft_soft_loss_v3_formal_campaign_v1" or payload["status"] != "APPROVED_FROZEN":
        raise ValueError("formal campaign schema/status is invalid")
    _semantic_payload(payload, label="formal campaign")
    primary = _sequence(payload["primary"], label="campaign primary")
    repeats = _sequence(payload["repeats"], label="campaign repeats")
    pairs = _sequence(payload["pairs"], label="campaign pairs")
    if len(primary) != 40 or len(repeats) != 40 or len(pairs) != 40:
        raise ValueError("formal campaign must contain exact 40+40+40 records")
    expected_cells = [(task, variant) for task in _TASKS for variant in _VARIANTS]
    for position, (raw_primary, raw_repeat, raw_pair, expected_cell) in enumerate(zip(primary, repeats, pairs, expected_cells)):
        first = _mapping(raw_primary, label=f"campaign primary[{position}]")
        second = _mapping(raw_repeat, label=f"campaign repeat[{position}]")
        pair = _mapping(raw_pair, label=f"campaign pair[{position}]")
        _exact_keys(first, _CAMPAIGN_CELL_KEYS, label=f"campaign primary[{position}]")
        _exact_keys(second, _CAMPAIGN_CELL_KEYS, label=f"campaign repeat[{position}]")
        _exact_keys(pair, _PAIR_KEYS, label=f"campaign pair[{position}]")
        task, variant = expected_cell
        if (first["task"], first["variant"]) != expected_cell or (second["task"], second["variant"]) != expected_cell:
            raise ValueError("campaign task/variant order changed")
        if (first["replicate_role"], first["replicate_index"], first["inference_eligible"]) != ("primary", 0, True):
            raise ValueError("campaign primary role/index/eligibility changed")
        if (second["replicate_role"], second["replicate_index"], second["inference_eligible"]) != ("technical_repeat", 1, False):
            raise ValueError("campaign repeat role/index/eligibility changed")
        scientific = _digest(first["scientific_config_hash"], label="campaign scientific config")
        if second["scientific_config_hash"] != scientific:
            raise ValueError("campaign primary/repeat scientific pairing changed")
        if pair != {
            "task_variant": [task, variant],
            "primary_run_id": first["run_id"],
            "repeat_run_id": second["run_id"],
            "primary_path": first["path"],
            "repeat_path": second["path"],
            "scientific_config_hash": scientific,
        }:
            raise ValueError("campaign pair differs from its two cells")
    return dict(payload), primary, repeats, pairs


def _validate_metrics(value: object, *, label: str, expected_config: str, expected_variant: str) -> dict[str, object]:
    payload = _mapping(value, label=label)
    _exact_keys(
        payload,
        {
            "schema",
            "config_hash",
            "profile",
            "steps",
            "final_step_exported",
            "final_loss",
            "thresholds",
            "task_iou_by_threshold",
            "worst_threshold_iou",
            "threshold_iou_range",
            "task_mscd_px",
            "task_p95_symmetric_distance_px",
            "invalid_task_metric",
        },
        label=label,
    )
    if payload["schema"] != "pcraft_soft_loss_v3_metrics_v1" or payload["config_hash"] != expected_config or payload["profile"] != expected_variant:
        raise ValueError(f"{label} identity differs from its registered row")
    if type(payload["steps"]) is not int or payload["steps"] != 2000 or payload["final_step_exported"] is not True:
        raise ValueError(f"{label} is not a final 2000-step export")
    _finite(payload["final_loss"], label=f"{label}.final_loss")
    thresholds = _mapping(payload["thresholds"], label=f"{label}.thresholds")
    if list(thresholds) != ["0.04", "0.05", "0.06"]:
        raise ValueError(f"{label} threshold order changed")
    aggregate_keys = {
        "tp", "fp", "fn", "tn", "mean_iou", "mean_dice", "mean_fpr", "mean_recall",
        "mean_boundary_f1_2px", "mean_mscd_px", "mean_p95_symmetric_distance_px",
        "invalid_iou_denominator", "invalid_dice_denominator", "invalid_background_denominator",
        "invalid_foreground_denominator", "invalid_predicted_boundary", "invalid_target_boundary",
        "invalid_task_metric",
    }
    for threshold, raw_record in thresholds.items():
        record = _mapping(raw_record, label=f"{label}.thresholds[{threshold}]")
        _exact_keys(record, {"per_view", "aggregate"}, label=f"{label}.thresholds[{threshold}]")
        _sequence(record["per_view"], label=f"{label}.thresholds[{threshold}].per_view")
        aggregate = _mapping(record["aggregate"], label=f"{label}.thresholds[{threshold}].aggregate")
        _exact_keys(aggregate, aggregate_keys, label=f"{label}.thresholds[{threshold}].aggregate")
        for name in ("tp", "fp", "fn", "tn"):
            if type(aggregate[name]) is not int or aggregate[name] < 0:
                raise ValueError(f"{label} aggregate count is invalid")
        for name in ("mean_iou", "mean_dice", "mean_boundary_f1_2px", "mean_mscd_px", "mean_p95_symmetric_distance_px"):
            _finite(aggregate[name], label=f"{label}.{name}")
        for name in ("invalid_iou_denominator", "invalid_dice_denominator", "invalid_background_denominator", "invalid_foreground_denominator", "invalid_predicted_boundary", "invalid_target_boundary", "invalid_task_metric"):
            if type(aggregate[name]) is not bool:
                raise ValueError(f"{label}.{name} must be bool")
        for name, invalid_name in (("mean_fpr", "invalid_background_denominator"), ("mean_recall", "invalid_foreground_denominator")):
            if aggregate[name] is None:
                if aggregate[invalid_name] is not True:
                    raise ValueError(f"{label}.{name} null lacks its invalid flag")
            else:
                _finite(aggregate[name], label=f"{label}.{name}")
    task_iou = _mapping(payload["task_iou_by_threshold"], label=f"{label}.task_iou_by_threshold")
    if list(task_iou) != ["0.04", "0.05", "0.06"]:
        raise ValueError(f"{label} task IoU thresholds changed")
    for key, value_ in task_iou.items():
        _finite(value_, label=f"{label}.task_iou[{key}]")
    _finite(payload["worst_threshold_iou"], label=f"{label}.worst_threshold_iou")
    _finite(payload["threshold_iou_range"], label=f"{label}.threshold_iou_range", nonnegative=True)
    _finite(payload["task_mscd_px"], label=f"{label}.task_mscd_px", nonnegative=True)
    _finite(payload["task_p95_symmetric_distance_px"], label=f"{label}.task_p95", nonnegative=True)
    if type(payload["invalid_task_metric"]) is not bool:
        raise ValueError(f"{label}.invalid_task_metric must be bool")
    return normalize_builtin(payload)


def _validate_collected_auxiliaries(row: Mapping[str, object], *, path: Path) -> None:
    for field, filename in (
        ("gradient_audit", "gradient_audit.json"),
        ("resource_metrics", "resource_metrics.json"),
    ):
        wrapper = _mapping(row[field], label=f"formal row {field}")
        _exact_keys(wrapper, {"path", "sha256", "payload"}, label=f"formal row {field}")
        if wrapper["path"] != str(path / filename):
            raise ValueError(f"formal row {field} path is noncanonical")
        _digest(wrapper["sha256"], label=f"formal row {field} sha256")
        _mapping(wrapper["payload"], label=f"formal row {field} payload")

    history = _mapping(row["history"], label="formal row history")
    _exact_keys(history, {"path", "sha256", "rows"}, label="formal row history")
    if history["path"] != str(path / "history.jsonl"):
        raise ValueError("formal row history path is noncanonical")
    _digest(history["sha256"], label="formal row history sha256")
    _sequence(history["rows"], label="formal row history rows")

    parameter = _mapping(row["parameter_change"], label="formal row parameter change")
    _exact_keys(
        parameter,
        {"reference_probability", "max_abs_delta", "changed"},
        label="formal row parameter change",
    )
    reference = _finite(
        parameter["reference_probability"], label="formal row reference probability"
    )
    if not 0.0 <= reference <= 1.0:
        raise ValueError("formal row reference probability is outside [0,1]")
    _finite(
        parameter["max_abs_delta"],
        label="formal row maximum parameter delta",
        nonnegative=True,
    )
    if type(parameter["changed"]) is not bool:
        raise ValueError("formal row parameter changed flag must be bool")


def validate_formal_matrix(
    rows: Sequence[Mapping[str, object]],
    *,
    formal_suite: Mapping[str, object],
    formal_campaign: Mapping[str, object],
    replicate_role: str,
    replicate_index: int,
) -> ValidatedFormalMatrix:
    """Authenticate one exact role-specific 40-cell formal matrix."""

    if type(replicate_role) is not str or type(replicate_index) is not int:
        raise ValueError("replicate role/index have wrong exact types")
    role_key = (replicate_role, replicate_index)
    if role_key not in _ROLES:
        raise ValueError("replicate role/index is not registered")
    stage, eligible = _ROLES[role_key]
    suite_payload, suite_tasks = _validate_formal_suite(_mapping(formal_suite, label="formal suite"))
    campaign_payload, primary_cells, repeat_cells, pairs = _validate_campaign(_mapping(formal_campaign, label="formal campaign"))
    raw_rows = _sequence(rows, label="formal rows")
    if len(raw_rows) != 40:
        raise ValueError("formal matrix must contain exactly forty rows")
    cells = primary_cells if replicate_role == "primary" else repeat_cells
    normalized_rows: list[dict[str, object]] = []
    stage_identity: dict[str, object] | None = None
    campaign_identity: dict[str, object] | None = None
    suite_identity: dict[str, object] | None = None
    common_master: object | None = None
    common_source: object | None = None
    common_weight: object | None = None
    expected_cells = [(task, variant) for task in _TASKS for variant in _VARIANTS]
    try:
        from experiments.run_soft_loss_v3 import RunSpec

        run_spec_keys = set(RunSpec.__dataclass_fields__)
    except (ImportError, AttributeError) as error:  # pragma: no cover - upstream invariant
        raise ValueError("Task 11 RunSpec contract is unavailable") from error
    for position, (raw_row, raw_cell, expected_cell) in enumerate(zip(raw_rows, cells, expected_cells)):
        row = _mapping(raw_row, label=f"formal row[{position}]")
        cell = _mapping(raw_cell, label=f"campaign cell[{position}]")
        _exact_keys(row, _ROW_KEYS, label=f"formal row[{position}]")
        task, variant = expected_cell
        if (row["task"], row["variant"]) != expected_cell:
            raise ValueError("formal row order differs from campaign order")
        if row["schema"] != "pcraft_soft_loss_v3_collected_row_v1" or row["stage"] != stage:
            raise ValueError("formal row schema/stage is invalid")
        if (row["replicate_role"], row["replicate_index"], row["inference_eligible"]) != (replicate_role, replicate_index, eligible):
            raise ValueError("formal row role/index/eligibility is invalid")
        if type(row["resolution"]) is not int or row["resolution"] != 256 or type(row["steps"]) is not int or row["steps"] != 2000:
            raise ValueError("formal row protocol resolution/steps changed")
        for key in ("run_id", "path", "task", "variant", "replicate_role", "replicate_index", "inference_eligible", "scientific_config_hash"):
            if row[key] != cell[key]:
                raise ValueError(f"formal row differs from campaign cell: {key}")
        run_id = _string(row["run_id"], label="formal row run_id")
        path = Path(_string(row["path"], label="formal row path"))
        _canonical_path(path, label="formal row path")
        expected_run_id = f"{stage}/steps_2000/{task}/{variant}/replicate_{replicate_index}/seed_17"
        if run_id != expected_run_id or not str(path).endswith(f"/{expected_run_id}"):
            raise ValueError("formal row run ID/path hierarchy is noncanonical")
        scientific = _digest(row["scientific_config_hash"], label="formal row scientific hash")
        expected_config = _digest(row["expected_config_hash"], label="formal row expected config hash")
        run_spec = _mapping(row["run_spec"], label="formal row RunSpec")
        _exact_keys(run_spec, run_spec_keys, label="formal row RunSpec")
        if normalize_builtin(run_spec) != normalize_builtin(cell["run_spec"]):
            raise ValueError("formal row RunSpec differs from campaign")
        if run_spec["stage"] != stage or run_spec["run_id"] != run_id or run_spec["output_path"] != str(path):
            raise ValueError("formal row RunSpec identity is inconsistent")
        if normalize_builtin(run_spec["inputs_xyz"]) != suite_tasks[task]:
            raise ValueError("formal row x/y/z inputs differ from formal suite")
        current_suite = _identity(run_spec["suite_identity"], label="RunSpec formal suite identity")
        if current_suite["semantic_sha256"] != suite_payload["semantic_sha256"]:
            raise ValueError("RunSpec formal suite semantic identity is stale")
        if suite_identity is None:
            suite_identity = current_suite
        elif current_suite != suite_identity:
            raise ValueError("formal matrix mixes formal-suite identities")
        for name, previous in (("master_identity", common_master), ("source_identity", common_source), ("weight_identity", common_weight)):
            current = normalize_builtin(run_spec[name])
            if previous is None:
                if name == "master_identity":
                    common_master = current
                elif name == "source_identity":
                    common_source = current
                else:
                    common_weight = current
            elif current != previous:
                raise ValueError(f"formal matrix mixes {name}")
        current_stage = _identity(row["stage_manifest"], label="formal row stage manifest")
        if Path(str(current_stage["path"])).name != f"{stage}.json":
            raise ValueError("formal row stage-manifest path is wrong")
        if stage_identity is None:
            stage_identity = current_stage
        elif current_stage != stage_identity:
            raise ValueError("formal matrix mixes stage manifests")
        current_campaign = _identity(row["formal_campaign"], label="formal row campaign identity")
        if Path(str(current_campaign["path"])).name != "formal_campaign.json" or current_campaign["semantic_sha256"] != campaign_payload["semantic_sha256"]:
            raise ValueError("formal row campaign identity is stale")
        if campaign_identity is None:
            campaign_identity = current_campaign
        elif current_campaign != campaign_identity:
            raise ValueError("formal matrix mixes campaign roots/identities")
        status = _mapping(row["status"], label="formal row STATUS")
        _exact_keys(status, {"path", "sha256", "payload"}, label="formal row STATUS")
        if status["path"] != str(path / "STATUS.json"):
            raise ValueError("formal row STATUS path is noncanonical")
        _digest(status["sha256"], label="formal row STATUS sha256")
        status_payload = _mapping(status["payload"], label="formal row STATUS payload")
        _exact_keys(status_payload, {"schema", "state", "config_hash", "completed_updates", "artifact_manifest_sha256"}, label="formal row STATUS payload")
        if status_payload["schema"] != "pcraft_soft_loss_v3_status_v1" or status_payload["state"] != "completed" or status_payload["config_hash"] != expected_config or status_payload["completed_updates"] != 2000 or type(status_payload["completed_updates"]) is not int:
            raise ValueError("formal row STATUS is not exact final completion")
        artifact = _mapping(row["artifact_manifest"], label="formal row artifact manifest")
        _exact_keys(artifact, {"path", "sha256", "payload"}, label="formal row artifact manifest")
        if artifact["path"] != str(path / "artifact_manifest.json") or artifact["sha256"] != status_payload["artifact_manifest_sha256"]:
            raise ValueError("formal row artifact-manifest identity is inconsistent")
        _digest(artifact["sha256"], label="formal row artifact-manifest sha256")
        artifact_payload = _mapping(artifact["payload"], label="formal row artifact payload")
        _exact_keys(artifact_payload, {"schema", "config_hash", "artifacts_sha256"}, label="formal row artifact payload")
        if artifact_payload["schema"] != "pcraft_soft_loss_v3_artifacts_v1" or artifact_payload["config_hash"] != expected_config:
            raise ValueError("formal row artifact payload is stale")
        artifacts = _mapping(artifact_payload["artifacts_sha256"], label="formal row artifact hashes")
        metrics_record = _mapping(row["metrics"], label="formal row metrics")
        _exact_keys(metrics_record, {"path", "sha256", "payload"}, label="formal row metrics")
        if metrics_record["path"] != str(path / "metrics.json") or artifacts.get("metrics.json") != metrics_record["sha256"]:
            raise ValueError("formal row metrics identity is inconsistent")
        _digest(metrics_record["sha256"], label="formal row metrics sha256")
        _validate_metrics(metrics_record["payload"], label="formal row metrics payload", expected_config=expected_config, expected_variant=variant)
        _validate_collected_auxiliaries(row, path=path)
        normalized_rows.append(normalize_builtin(row))
    assert stage_identity is not None and campaign_identity is not None and suite_identity is not None
    matrix_payload = {
        "replicate_role": replicate_role,
        "replicate_index": replicate_index,
        "inference_eligible": eligible,
        "tasks": list(_TASKS),
        "variants": list(_VARIANTS),
        "rows": normalized_rows,
        "formal_suite_identity": suite_identity,
        "formal_campaign_identity": campaign_identity,
        "stage_manifest_sha256": stage_identity["sha256"],
    }
    return ValidatedFormalMatrix._validated(
        _MATRIX_TOKEN,
        replicate_role=replicate_role,
        replicate_index=replicate_index,
        inference_eligible=eligible,
        rows=normalized_rows,
        pairs=[normalize_builtin(pair) for pair in pairs],
        matrix_sha256=canonical_hash(matrix_payload),
        suite_identity=suite_identity,
        campaign_identity=campaign_identity,
        stage_manifest_sha256=str(stage_identity["sha256"]),
    )


def _matrix_rows(matrix: ValidatedFormalMatrix) -> tuple[Mapping[str, object], ...]:
    if not isinstance(matrix, ValidatedFormalMatrix):
        raise ValueError("expected a ValidatedFormalMatrix")
    return matrix._rows


def _row_endpoints(row: Mapping[str, object]) -> tuple[float, float]:
    metrics = _mapping(_mapping(row["metrics"], label="metrics")["payload"], label="metrics payload")
    return (
        _finite(metrics["task_mscd_px"], label="task MSCD"),
        _finite(metrics["worst_threshold_iou"], label="worst-threshold IoU"),
    )


def factorial_task_contrasts(matrix: ValidatedFormalMatrix) -> list[FactorialTaskContrast]:
    rows = _matrix_rows(matrix)
    result: list[FactorialTaskContrast] = []
    for task_index, task in enumerate(_TASKS):
        task_rows = rows[task_index * 4 : (task_index + 1) * 4]
        endpoints = {variant: _row_endpoints(row) for variant, row in zip(_VARIANTS, task_rows)}
        b_mscd, b_iou = endpoints["l12_bal"]
        d_mscd, d_iou = endpoints["dic_only"]
        t_mscd, t_iou = endpoints["dts_only"]
        f_mscd, f_iou = endpoints["v3_full"]
        dic_first = b_mscd - d_mscd
        dic_second = t_mscd - f_mscd
        dts_first = t_iou - b_iou
        dts_second = f_iou - d_iou
        result.append(
            FactorialTaskContrast(
                task=task,
                dic_marginal_mscd_improvement_px=0.5 * (dic_first + dic_second),
                dts_marginal_worst_threshold_iou_improvement=0.5 * (dts_first + dts_second),
                m_d_interaction_px=-f_mscd + d_mscd + t_mscd - b_mscd,
                m_t_interaction=f_iou - d_iou - t_iou + b_iou,
                dic_base_to_dic_mscd_improvement_px=dic_first,
                dic_dts_to_full_mscd_improvement_px=dic_second,
                dts_base_to_dts_worst_threshold_iou_improvement=dts_first,
                dts_dic_to_full_worst_threshold_iou_improvement=dts_second,
            )
        )
    return result


def _numeric_sequence(values: Sequence[float], *, label: str, minimum: int = 1, maximum: int | None = None) -> np.ndarray:
    items = _sequence(values, label=label)
    if len(items) < minimum or (maximum is not None and len(items) > maximum):
        raise ValueError(f"{label} has invalid length")
    return np.asarray([_finite(item, label=f"{label}[{index}]") for index, item in enumerate(items)], dtype=np.float64)


def exact_sign_flip_greater(differences: Sequence[float]) -> float:
    values = _numeric_sequence(differences, label="sign-flip differences", minimum=1, maximum=20)
    observed = float(values.mean(dtype=np.float64))
    count = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = float(np.multiply(values, np.asarray(signs, dtype=np.float64)).mean(dtype=np.float64))
        if permuted >= observed - 1e-15:
            count += 1
    return float(count / (2 ** len(values)))


def _hf7(sorted_values: np.ndarray, q: float) -> float:
    if sorted_values.ndim != 1 or sorted_values.size == 0:
        raise ValueError("HF7 requires a nonempty one-dimensional array")
    probability = _finite(q, label="quantile probability")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability is outside [0,1]")
    h = (sorted_values.size - 1) * probability
    lower = int(math.floor(h))
    fraction = h - lower
    upper = min(lower + 1, sorted_values.size - 1)
    return float((1.0 - fraction) * sorted_values[lower] + fraction * sorted_values[upper])


def _bootstrap_index_matrix(*, draws: int, n_tasks: int, seed: int) -> np.ndarray:
    count = _exact_int(draws, label="bootstrap draws", positive=True)
    tasks = _exact_int(n_tasks, label="bootstrap n_tasks", positive=True)
    seed_value = _exact_int(seed, label="bootstrap seed")
    return np.random.default_rng(seed_value).integers(0, tasks, size=(count, tasks))


def paired_bootstrap(
    differences: Sequence[float],
    *,
    draws: int = _LOCKED_DRAWS,
    seed: int = _LOCKED_SEED,
) -> dict[str, float]:
    values = _numeric_sequence(differences, label="bootstrap differences", minimum=1, maximum=20)
    indices = _bootstrap_index_matrix(draws=draws, n_tasks=len(values), seed=seed)
    means = values[indices].mean(axis=1, dtype=np.float64)
    means.sort(kind="mergesort")
    return {
        "mean": float(values.mean(dtype=np.float64)),
        "ci_low": _hf7(means, 0.025),
        "ci_high": _hf7(means, 0.975),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    values = _mapping(p_values, label="Holm p-values")
    _exact_keys(values, {"dic", "dts"}, label="Holm p-values")
    normalized = {name: _finite(values[name], label=f"Holm {name}") for name in ("dic", "dts")}
    if any(not 0.0 <= value <= 1.0 for value in normalized.values()):
        raise ValueError("Holm p-values must lie in [0,1]")
    order = sorted(("dic", "dts"), key=lambda name: (normalized[name], 0 if name == "dic" else 1))
    first, second = order
    first_adjusted = min(1.0, 2.0 * normalized[first])
    second_adjusted = max(first_adjusted, normalized[second])
    unordered = {first: first_adjusted, second: second_adjusted}
    return {"dic": float(unordered["dic"]), "dts": float(unordered["dts"])}


def _bootstrap_records(contrasts: Sequence[FactorialTaskContrast], *, draws: int) -> dict[str, dict[str, float]]:
    if len(contrasts) != 10:
        raise ValueError("formal inference requires exactly ten task contrasts")
    arrays = OrderedDict(
        (
            name,
            np.asarray([getattr(contrast, field) for contrast in contrasts], dtype=np.float64),
        )
        for name, field in (
            ("dic", "dic_marginal_mscd_improvement_px"),
            ("dts", "dts_marginal_worst_threshold_iou_improvement"),
            ("dic_base_to_dic", "dic_base_to_dic_mscd_improvement_px"),
            ("dic_dts_to_full", "dic_dts_to_full_mscd_improvement_px"),
            ("dts_base_to_dts", "dts_base_to_dts_worst_threshold_iou_improvement"),
            ("dts_dic_to_full", "dts_dic_to_full_worst_threshold_iou_improvement"),
            ("m_d", "m_d_interaction_px"),
            ("m_t", "m_t_interaction"),
        )
    )
    indices = _bootstrap_index_matrix(draws=draws, n_tasks=10, seed=_LOCKED_SEED)
    result: dict[str, dict[str, float]] = {}
    for name, values in arrays.items():
        means = values[indices].mean(axis=1, dtype=np.float64)
        means.sort(kind="mergesort")
        result[name] = {
            "mean": float(values.mean(dtype=np.float64)),
            "ci_low": _hf7(means, 0.025),
            "ci_high": _hf7(means, 0.975),
        }
    return result


def _failure_list(*items: tuple[bool, str]) -> list[str]:
    return sorted({code for failed, code in items if failed})


def _formal_inference(matrix: ValidatedFormalMatrix, *, draws: int, require_primary: bool) -> dict[str, object]:
    if require_primary and (matrix.replicate_role, matrix.replicate_index, matrix.inference_eligible) != ("primary", 0, True):
        raise ValueError("formal confirmatory inference accepts only primary/0")
    draw_count = _exact_int(draws, label="formal bootstrap draws", positive=True)
    contrasts = factorial_task_contrasts(matrix)
    boot = _bootstrap_records(contrasts, draws=draw_count)
    effects = {
        "dic": [float(item.dic_marginal_mscd_improvement_px) for item in contrasts],
        "dts": [float(item.dts_marginal_worst_threshold_iou_improvement) for item in contrasts],
    }
    raw_p = {name: exact_sign_flip_greater(values) for name, values in effects.items()}
    adjusted = holm_adjust(raw_p)
    endpoints = {"dic": "task_mscd_px_improvement", "dts": "worst_threshold_iou_improvement"}
    hypotheses: dict[str, object] = {}
    for name in ("dic", "dts"):
        statistic_valid = True
        significant = adjusted[name] < 0.05
        ci_above = boot[name]["ci_low"] > 0.0
        failure_codes = _failure_list(
            (not statistic_valid, "primary_invalid"),
            (not significant, "holm_not_significant"),
            (not ci_above, "ci_not_above_zero"),
        )
        hypotheses[name] = {
            "component": name,
            "endpoint": endpoints[name],
            "task_effects": effects[name],
            "mean_effect": boot[name]["mean"],
            "exact_sign_flip_p_greater": raw_p[name],
            "holm_adjusted_p": adjusted[name],
            "ci_low": boot[name]["ci_low"],
            "ci_high": boot[name]["ci_high"],
            "statistic_valid": statistic_valid,
            "significant": significant,
            "ci_above_zero": ci_above,
            "passed": statistic_valid and significant and ci_above,
            "failure_codes": failure_codes,
        }
    conditional_fields = {
        "dic": (
            ("base_to_dic", "dic_base_to_dic_mscd_improvement_px", "dic_base_to_dic"),
            ("dts_to_full", "dic_dts_to_full_mscd_improvement_px", "dic_dts_to_full"),
        ),
        "dts": (
            ("base_to_dts", "dts_base_to_dts_worst_threshold_iou_improvement", "dts_base_to_dts"),
            ("dic_to_full", "dts_dic_to_full_worst_threshold_iou_improvement", "dts_dic_to_full"),
        ),
    }
    conditional: dict[str, object] = {}
    for component in ("dic", "dts"):
        component_effects: OrderedDict[str, object] = OrderedDict()
        for effect_name, field_name, bootstrap_name in conditional_fields[component]:
            component_effects[effect_name] = {
                "task_effects": [float(getattr(item, field_name)) for item in contrasts],
                **boot[bootstrap_name],
                "statistic_valid": True,
            }
        conditional[component] = component_effects
    interactions: dict[str, object] = {}
    for name, field_name, endpoint, lower, upper in (
        ("m_d", "m_d_interaction_px", "sign_equivalent_mscd_interaction_px", -0.10, 0.10),
        ("m_t", "m_t_interaction", "worst_threshold_iou_interaction", -0.001, 0.001),
    ):
        equivalent = boot[name]["ci_low"] >= lower and boot[name]["ci_high"] <= upper
        interactions[name] = {
            "endpoint": endpoint,
            "task_effects": [float(getattr(item, field_name)) for item in contrasts],
            **boot[name],
            "equivalence_lower": lower,
            "equivalence_upper": upper,
            "equivalent": equivalent,
            "wording": "NO_MATERIAL_INTERACTION" if equivalent else "CONDITIONAL_OR_COUPLED_EFFECTS",
            "statistic_valid": True,
        }
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_formal_inference_v1",
        "matrix_sha256": matrix.matrix_sha256,
        "replicate_role": matrix.replicate_role,
        "replicate_index": matrix.replicate_index,
        "n_tasks": 10,
        "family": ["dic", "dts"],
        "alpha": 0.05,
        "alternative": "greater",
        "assumption": "task_level_symmetry_exchangeability_not_randomized_treatment",
        "bootstrap": {
            "draws": draw_count,
            "seed": _LOCKED_SEED,
            "ci_level": 0.95,
            "method": "hyndman_fan_type_7",
            "joint_task_indices": True,
        },
        "hypotheses": hypotheses,
        "conditional_effects": conditional,
        "interactions": interactions,
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    return payload


def formal_factorial_inference(
    matrix: ValidatedFormalMatrix,
    *,
    draws: int = _LOCKED_DRAWS,
) -> dict[str, object]:
    if not isinstance(matrix, ValidatedFormalMatrix):
        raise ValueError("formal inference requires a validated matrix")
    return _formal_inference(matrix, draws=draws, require_primary=True)


def _aggregate_for(row: Mapping[str, object]) -> Mapping[str, object]:
    metrics = _mapping(_mapping(row["metrics"], label="metrics")["payload"], label="metrics payload")
    thresholds = _mapping(metrics["thresholds"], label="metrics thresholds")
    return _mapping(_mapping(thresholds["0.05"], label="tau=0.05")["aggregate"], label="tau=0.05 aggregate")


def _has_threshold_iou_invalid(row: Mapping[str, object]) -> bool:
    metrics = _mapping(_mapping(row["metrics"], label="metrics")["payload"], label="metrics payload")
    thresholds = _mapping(metrics["thresholds"], label="metrics thresholds")
    return any(
        _mapping(
            _mapping(thresholds[threshold], label=f"tau={threshold}")["aggregate"],
            label=f"tau={threshold} aggregate",
        )["invalid_iou_denominator"]
        is True
        for threshold in ("0.04", "0.05", "0.06")
    )


def conditional_safety_checks(
    matrix: ValidatedFormalMatrix,
    component: str,
) -> dict[str, object]:
    rows = _matrix_rows(matrix)
    if type(component) is not str or component not in {"dic", "dts"}:
        raise ValueError("component must be exactly dic or dts")
    comparisons = (
        (("base_to_dic", "l12_bal", "dic_only"), ("dts_to_full", "dts_only", "v3_full"))
        if component == "dic"
        else (("base_to_dts", "l12_bal", "dts_only"), ("dic_to_full", "dic_only", "v3_full"))
    )
    indexed = {(str(row["task"]), str(row["variant"])): row for row in rows}
    reports: list[dict[str, object]] = []
    all_failures: set[str] = set()
    for name, control_name, candidate_name in comparisons:
        primary_effects: list[float] = []
        iou_effects: list[float] = []
        control_ious: list[float] = []
        candidate_ious: list[float] = []
        recall_effects: list[float] = []
        recall_pairs: list[tuple[float, float]] = []
        fpr_effects: list[float] = []
        control_fprs: list[float] = []
        candidate_fprs: list[float] = []
        invalid: list[str] = []
        recall_valid = True
        fpr_valid = True
        iou_valid = True
        primary_valid = True
        for task in _TASKS:
            control = indexed[(task, control_name)]
            candidate = indexed[(task, candidate_name)]
            control_mscd, control_worst = _row_endpoints(control)
            candidate_mscd, candidate_worst = _row_endpoints(candidate)
            primary_effects.append(control_mscd - candidate_mscd if component == "dic" else candidate_worst - control_worst)
            control_aggregate = _aggregate_for(control)
            candidate_aggregate = _aggregate_for(candidate)
            if component == "dic" and any(bool(item[flag]) for item in (control_aggregate, candidate_aggregate) for flag in ("invalid_predicted_boundary", "invalid_target_boundary")):
                primary_valid = False
                invalid.append(f"{task}:boundary_invalid")
            if component == "dts" and any(
                _has_threshold_iou_invalid(item) for item in (control, candidate)
            ):
                primary_valid = False
                invalid.append(f"{task}:worst_threshold_iou_invalid")
            if any(bool(item["invalid_iou_denominator"]) for item in (control_aggregate, candidate_aggregate)):
                iou_valid = False
                invalid.append(f"{task}:iou_invalid")
                if component == "dts":
                    primary_valid = False
            if iou_valid:
                candidate_iou = _finite(candidate_aggregate["mean_iou"], label="candidate IoU")
                control_iou = _finite(control_aggregate["mean_iou"], label="control IoU")
                candidate_ious.append(candidate_iou)
                control_ious.append(control_iou)
                iou_effects.append(candidate_iou - control_iou)
            if control_aggregate["mean_recall"] is None or candidate_aggregate["mean_recall"] is None or control_aggregate["invalid_foreground_denominator"] or candidate_aggregate["invalid_foreground_denominator"]:
                recall_valid = False
                invalid.append(f"{task}:recall_invalid")
            else:
                candidate_recall = _finite(candidate_aggregate["mean_recall"], label="candidate recall")
                control_recall = _finite(control_aggregate["mean_recall"], label="control recall")
                recall_pairs.append((control_recall, candidate_recall))
                recall_effects.append(candidate_recall - control_recall)
            if control_aggregate["mean_fpr"] is None or candidate_aggregate["mean_fpr"] is None or control_aggregate["invalid_background_denominator"] or candidate_aggregate["invalid_background_denominator"]:
                fpr_valid = False
                invalid.append(f"{task}:fpr_invalid")
            else:
                candidate_fpr = _finite(candidate_aggregate["mean_fpr"], label="candidate FPR")
                control_fpr = _finite(control_aggregate["mean_fpr"], label="control FPR")
                candidate_fprs.append(candidate_fpr)
                control_fprs.append(control_fpr)
                fpr_effects.append(candidate_fpr - control_fpr)
        mean_primary = float(np.mean(primary_effects, dtype=np.float64)) if primary_valid else None
        mean_iou = float(np.mean(iou_effects, dtype=np.float64)) if iou_valid and len(iou_effects) == 10 else None
        min_recall = float(min(recall_effects)) if recall_valid and len(recall_effects) == 10 else None
        mean_fpr = float(np.mean(fpr_effects, dtype=np.float64)) if fpr_valid and len(fpr_effects) == 10 else None
        checks = {
            "primary_beneficial": mean_primary is not None and mean_primary > 0.0,
            "iou_noninferior": (
                mean_iou is not None
                and (
                    sum((Decimal(str(value)) for value in candidate_ious), Decimal(0))
                    - sum((Decimal(str(value)) for value in control_ious), Decimal(0))
                ) / Decimal(10) >= Decimal("-0.001")
            ),
            "recall_noninferior": (
                min_recall is not None
                and all(
                    Decimal(str(candidate)) - Decimal(str(control)) >= Decimal("-0.005")
                    for control, candidate in recall_pairs
                )
            ),
            "fpr_guard": (
                mean_fpr is not None
                and (
                    sum((Decimal(str(value)) for value in candidate_fprs), Decimal(0))
                    - sum((Decimal(str(value)) for value in control_fprs), Decimal(0))
                ) / Decimal(10) <= Decimal("0.001")
            ),
        }
        if not checks["primary_beneficial"]:
            all_failures.add("primary_effect_nonpositive")
        if not checks["iou_noninferior"]:
            all_failures.add("iou_noninferiority_failed")
        if not checks["recall_noninferior"]:
            all_failures.add("recall_noninferiority_failed")
        if not checks["fpr_guard"]:
            all_failures.add("fpr_guard_failed")
        if invalid:
            all_failures.add("relevant_metric_invalid")
        reports.append(
            {
                "name": name,
                "control_variant": control_name,
                "candidate_variant": candidate_name,
                "task_primary_effects": primary_effects,
                "mean_primary_effect": mean_primary,
                "task_iou_0_05_effects": iou_effects if len(iou_effects) == 10 else [],
                "mean_iou_0_05_effect": mean_iou,
                "task_recall_0_05_effects": recall_effects if len(recall_effects) == 10 else [],
                "min_task_recall_0_05_effect": min_recall,
                "task_fpr_0_05_effects": fpr_effects if len(fpr_effects) == 10 else [],
                "mean_fpr_0_05_effect": mean_fpr,
                "checks": checks,
                "invalid_evidence": sorted(set(invalid)),
            }
        )
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_conditional_safety_v1",
        "matrix_sha256": matrix.matrix_sha256,
        "component": component,
        "n_tasks": 10,
        "criteria": {
            "mean_primary_effect_strict_lower": 0.0,
            "mean_iou_0_05_delta_min": -0.001,
            "per_task_recall_0_05_delta_min": -0.005,
            "mean_fpr_0_05_delta_max": 0.001,
        },
        "comparisons": reports,
        "passed": not all_failures,
        "failure_codes": sorted(all_failures),
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    return payload


def _validate_semantic_result(value: object, *, schema: str, keys: set[str], label: str) -> dict[str, object]:
    payload = _mapping(value, label=label)
    _exact_keys(payload, keys, label=label)
    if payload["schema"] != schema:
        raise ValueError(f"{label} schema is invalid")
    _semantic_payload(payload, label=label)
    return normalize_builtin(payload)


def _validate_numeric_record(
    record: Mapping[str, object],
    *,
    sequence_keys: Sequence[str],
    scalar_keys: Sequence[str],
    label: str,
) -> None:
    for key in sequence_keys:
        values = _sequence(record[key], label=f"{label}.{key}")
        if len(values) != 10:
            raise ValueError(f"{label}.{key} must contain exactly ten tasks")
        for index, value in enumerate(values):
            _finite(value, label=f"{label}.{key}[{index}]")
    for key in scalar_keys:
        _finite(record[key], label=f"{label}.{key}")


def _validate_inference_structure(payload: Mapping[str, object]) -> Mapping[str, object]:
    _digest(payload["matrix_sha256"], label="formal inference matrix SHA-256")
    if type(payload["replicate_role"]) is not str or payload["replicate_role"] != "primary":
        raise ValueError("formal inference replicate role is not primary")
    if type(payload["replicate_index"]) is not int or payload["replicate_index"] != 0:
        raise ValueError("formal inference replicate index is not exact integer zero")
    if type(payload["n_tasks"]) is not int or payload["n_tasks"] != 10:
        raise ValueError("formal inference task count is not exact integer ten")
    if _sequence(payload["family"], label="formal inference family") != ["dic", "dts"]:
        raise ValueError("formal inference family is not the registered order")
    if _finite(payload["alpha"], label="formal inference alpha") != 0.05:
        raise ValueError("formal inference alpha changed")
    if payload["alternative"] != "greater":
        raise ValueError("formal inference alternative is unregistered")
    if payload["assumption"] != "task_level_symmetry_exchangeability_not_randomized_treatment":
        raise ValueError("formal inference assumption changed")
    bootstrap = _mapping(payload["bootstrap"], label="formal inference bootstrap")
    _exact_keys(
        bootstrap,
        {"draws", "seed", "ci_level", "method", "joint_task_indices"},
        label="formal inference bootstrap",
    )
    _exact_int(bootstrap["draws"], label="formal inference bootstrap draws", positive=True)
    if _exact_int(bootstrap["seed"], label="formal inference bootstrap seed") != _LOCKED_SEED:
        raise ValueError("formal inference bootstrap seed changed")
    if _finite(bootstrap["ci_level"], label="formal inference bootstrap CI level") != 0.95:
        raise ValueError("formal inference bootstrap CI level changed")
    if bootstrap["method"] != "hyndman_fan_type_7":
        raise ValueError("formal inference bootstrap method changed")
    if bootstrap["joint_task_indices"] is not True:
        raise ValueError("formal inference joint-task flag must be exact true")

    hypotheses = _mapping(payload["hypotheses"], label="inference hypotheses")
    _exact_keys(hypotheses, {"dic", "dts"}, label="inference hypotheses")
    for component in ("dic", "dts"):
        hypothesis = _mapping(hypotheses[component], label=f"{component} hypothesis")
        _exact_keys(hypothesis, _HYPOTHESIS_KEYS, label=f"{component} hypothesis")
        if hypothesis["component"] != component:
            raise ValueError(f"{component} hypothesis component is swapped")
        expected_endpoint = {
            "dic": "task_mscd_px_improvement",
            "dts": "worst_threshold_iou_improvement",
        }[component]
        if hypothesis["endpoint"] != expected_endpoint:
            raise ValueError(f"{component} hypothesis endpoint changed")
        _validate_numeric_record(
            hypothesis,
            sequence_keys=("task_effects",),
            scalar_keys=(
                "mean_effect",
                "exact_sign_flip_p_greater",
                "holm_adjusted_p",
                "ci_low",
                "ci_high",
            ),
            label=f"{component} hypothesis",
        )
        for key in ("exact_sign_flip_p_greater", "holm_adjusted_p"):
            probability = float(hypothesis[key])
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{component} hypothesis probability is outside [0,1]")
        for key in ("statistic_valid", "significant", "ci_above_zero", "passed"):
            if type(hypothesis[key]) is not bool:
                raise ValueError(f"{component} hypothesis {key} must be bool")
        failures = _sequence(
            hypothesis["failure_codes"], label=f"{component} hypothesis failure codes"
        )
        if any(type(code) is not str for code in failures):
            raise ValueError(f"{component} hypothesis failure code must be a string")

    conditional = _mapping(
        payload["conditional_effects"], label="formal inference conditional effects"
    )
    _exact_keys(conditional, {"dic", "dts"}, label="formal inference conditional effects")
    names = {
        "dic": ("base_to_dic", "dts_to_full"),
        "dts": ("base_to_dts", "dic_to_full"),
    }
    for component, expected_names in names.items():
        effects = _mapping(conditional[component], label=f"{component} conditional effects")
        _exact_keys(effects, set(expected_names), label=f"{component} conditional effects")
        if tuple(effects) != expected_names:
            raise ValueError(f"{component} conditional effects are reordered")
        for name in expected_names:
            record = _mapping(effects[name], label=f"{component}.{name} conditional effect")
            _exact_keys(
                record,
                {"task_effects", "mean", "ci_low", "ci_high", "statistic_valid"},
                label=f"{component}.{name} conditional effect",
            )
            _validate_numeric_record(
                record,
                sequence_keys=("task_effects",),
                scalar_keys=("mean", "ci_low", "ci_high"),
                label=f"{component}.{name} conditional effect",
            )
            if type(record["statistic_valid"]) is not bool:
                raise ValueError(f"{component}.{name} statistic-valid flag must be bool")

    interactions = _mapping(payload["interactions"], label="formal inference interactions")
    _exact_keys(interactions, {"m_d", "m_t"}, label="formal inference interactions")
    for name in ("m_d", "m_t"):
        record = _mapping(interactions[name], label=f"{name} interaction")
        _exact_keys(
            record,
            {
                "endpoint",
                "task_effects",
                "mean",
                "ci_low",
                "ci_high",
                "equivalence_lower",
                "equivalence_upper",
                "equivalent",
                "wording",
                "statistic_valid",
            },
            label=f"{name} interaction",
        )
        _string(record["endpoint"], label=f"{name} interaction endpoint")
        _validate_numeric_record(
            record,
            sequence_keys=("task_effects",),
            scalar_keys=("mean", "ci_low", "ci_high", "equivalence_lower", "equivalence_upper"),
            label=f"{name} interaction",
        )
        for key in ("equivalent", "statistic_valid"):
            if type(record[key]) is not bool:
                raise ValueError(f"{name} interaction {key} must be bool")
        if record["wording"] not in {
            "NO_MATERIAL_INTERACTION",
            "CONDITIONAL_OR_COUPLED_EFFECTS",
        }:
            raise ValueError(f"{name} interaction wording is unregistered")
    return hypotheses


def _validate_safety_structure(
    payload: Mapping[str, object], *, component: str, matrix_sha256: str
) -> dict[str, object]:
    result = _validate_semantic_result(
        payload,
        schema="pcraft_soft_loss_v3_conditional_safety_v1",
        keys={
            "schema",
            "matrix_sha256",
            "component",
            "n_tasks",
            "criteria",
            "comparisons",
            "passed",
            "failure_codes",
            "semantic_sha256",
        },
        label=f"{component} safety",
    )
    if (
        result["component"] != component
        or result["matrix_sha256"] != matrix_sha256
        or result["n_tasks"] != 10
        or type(result["n_tasks"]) is not int
    ):
        raise ValueError("safety result is swapped or belongs to another matrix")
    criteria = _mapping(result["criteria"], label=f"{component} safety criteria")
    _exact_keys(criteria, set(_SAFETY_CRITERIA), label=f"{component} safety criteria")
    for name in _SAFETY_CRITERIA:
        _finite(criteria[name], label=f"{component} safety criterion {name}")
    if dict(criteria) != _SAFETY_CRITERIA:
        raise ValueError(f"{component} safety criteria changed")
    comparisons = _sequence(result["comparisons"], label=f"{component} safety comparisons")
    expected = (
        (("base_to_dic", "l12_bal", "dic_only"), ("dts_to_full", "dts_only", "v3_full"))
        if component == "dic"
        else (("base_to_dts", "l12_bal", "dts_only"), ("dic_to_full", "dic_only", "v3_full"))
    )
    if len(comparisons) != 2:
        raise ValueError(f"{component} safety requires exactly two comparisons")
    for index, (raw_record, identity) in enumerate(zip(comparisons, expected)):
        record = _mapping(raw_record, label=f"{component} safety comparison[{index}]")
        _exact_keys(
            record,
            _SAFETY_COMPARISON_KEYS,
            label=f"{component} safety comparison[{index}]",
        )
        if (record["name"], record["control_variant"], record["candidate_variant"]) != identity:
            raise ValueError(f"{component} safety comparison order or identity changed")
        primary = _sequence(
            record["task_primary_effects"],
            label=f"{component} safety comparison[{index}] primary effects",
        )
        if len(primary) != 10:
            raise ValueError("safety primary effects must contain exactly ten tasks")
        for position, value in enumerate(primary):
            _finite(value, label=f"{component} safety primary effect[{position}]")
        if record["mean_primary_effect"] is not None:
            _finite(
                record["mean_primary_effect"],
                label=f"{component} safety comparison[{index}] mean primary effect",
            )
        for sequence_key, summary_key in (
            ("task_iou_0_05_effects", "mean_iou_0_05_effect"),
            ("task_recall_0_05_effects", "min_task_recall_0_05_effect"),
            ("task_fpr_0_05_effects", "mean_fpr_0_05_effect"),
        ):
            values = _sequence(record[sequence_key], label=f"{component} safety {sequence_key}")
            if len(values) not in {0, 10}:
                raise ValueError(f"{component} safety {sequence_key} has invalid length")
            for position, value in enumerate(values):
                _finite(value, label=f"{component} safety {sequence_key}[{position}]")
            if record[summary_key] is None:
                if values:
                    raise ValueError(f"{component} safety {summary_key} unexpectedly null")
            else:
                _finite(record[summary_key], label=f"{component} safety {summary_key}")
                if not values:
                    raise ValueError(f"{component} safety {summary_key} lacks task effects")
        checks = _mapping(record["checks"], label=f"{component} safety checks")
        _exact_keys(checks, _SAFETY_CHECK_KEYS, label=f"{component} safety checks")
        if any(type(value) is not bool for value in checks.values()):
            raise ValueError(f"{component} safety checks must be bool")
        evidence = _sequence(
            record["invalid_evidence"], label=f"{component} safety invalid evidence"
        )
        if any(type(item) is not str for item in evidence):
            raise ValueError(f"{component} safety invalid evidence must contain strings")
    if type(result["passed"]) is not bool:
        raise ValueError(f"{component} safety passed flag must be bool")
    failures = _sequence(result["failure_codes"], label=f"{component} safety failure codes")
    allowed = {
        "primary_effect_nonpositive",
        "iou_noninferiority_failed",
        "recall_noninferiority_failed",
        "fpr_guard_failed",
        "relevant_metric_invalid",
    }
    if any(type(code) is not str or code not in allowed for code in failures):
        raise ValueError(f"{component} safety failure code is unregistered")
    return result


def retention_decision(
    inference: Mapping[str, object],
    safety: Mapping[str, object],
) -> dict[str, object]:
    inference_payload = _validate_semantic_result(
        inference,
        schema="pcraft_soft_loss_v3_formal_inference_v1",
        keys={"schema", "matrix_sha256", "replicate_role", "replicate_index", "n_tasks", "family", "alpha", "alternative", "assumption", "bootstrap", "hypotheses", "conditional_effects", "interactions", "semantic_sha256"},
        label="formal inference",
    )
    if inference_payload["replicate_role"] != "primary" or inference_payload["replicate_index"] != 0 or inference_payload["n_tasks"] != 10 or inference_payload["family"] != ["dic", "dts"]:
        raise ValueError("retention requires primary ten-task inference")
    safety_bundle = _mapping(safety, label="safety bundle")
    _exact_keys(safety_bundle, {"dic", "dts"}, label="safety bundle")
    hypotheses = _validate_inference_structure(inference_payload)
    decisions: dict[str, object] = {}
    retained_components: list[str] = []
    for component in ("dic", "dts"):
        safety_result = _validate_safety_structure(
            _mapping(safety_bundle[component], label=f"{component} safety"),
            component=component,
            matrix_sha256=str(inference_payload["matrix_sha256"]),
        )
        hypothesis = _mapping(hypotheses[component], label=f"{component} hypothesis")
        statistic_valid = hypothesis.get("statistic_valid") is True
        adjusted = _finite(hypothesis.get("holm_adjusted_p"), label=f"{component} adjusted p")
        ci_low = _finite(hypothesis.get("ci_low"), label=f"{component} CI lower")
        significant = adjusted < 0.05
        ci_above = ci_low > 0.0
        safety_passed = safety_result["passed"] is True
        retained = statistic_valid and significant and ci_above and safety_passed
        failures = _failure_list(
            (not statistic_valid, "primary_invalid"),
            (not significant, "holm_not_significant"),
            (not ci_above, "ci_not_above_zero"),
            (not safety_passed, "conditional_safety_failed"),
        )
        decisions[component] = {
            "statistic_valid": statistic_valid,
            "holm_adjusted_p": adjusted,
            "holm_significant": significant,
            "ci_low": ci_low,
            "ci_above_zero": ci_above,
            "conditional_safety_passed": safety_passed,
            "retained": retained,
            "failure_codes": failures,
        }
        if retained:
            retained_components.append(component)
    expression = {
        (): "L0",
        ("dic",): "L0+lambda_I(t)*L_DIC",
        ("dts",): "L0+lambda_T(t)*L_DTS",
        ("dic", "dts"): "L0+lambda_I(t)*L_DIC+lambda_T(t)*L_DTS",
    }[tuple(retained_components)]
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_retention_v1",
        "primary_matrix_sha256": inference_payload["matrix_sha256"],
        "n_tasks": 10,
        "criteria": {
            "holm_alpha": 0.05,
            "holm_adjusted_p_strict_upper": 0.05,
            "bootstrap_ci_low_strict_lower": 0.0,
            "conditional_safety_required": True,
        },
        "components": decisions,
        "retained_components": retained_components,
        "retained_formula": {
            "base": "mean_v[L2_bal+0.35*L1_bal]",
            "dic_enabled": "dic" in retained_components,
            "dts_enabled": "dts" in retained_components,
            "components": retained_components,
            "expression": expression,
        },
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    return payload


def _retention_for_matrix(matrix: ValidatedFormalMatrix, *, draws: int = _LOCKED_DRAWS) -> dict[str, bool]:
    inference = _formal_inference(matrix, draws=draws, require_primary=False)
    # Retention's public validator intentionally accepts only primary identity.
    # Audit-only retention uses the same arithmetic and rules without exporting
    # repeat p-values as confirmatory claims.
    hypotheses = _mapping(inference["hypotheses"], label="audit hypotheses")
    result: dict[str, bool] = {}
    for component in ("dic", "dts"):
        safety = conditional_safety_checks(matrix, component)
        hypothesis = _mapping(hypotheses[component], label="audit hypothesis")
        result[component] = bool(
            hypothesis["statistic_valid"] is True
            and _finite(hypothesis["holm_adjusted_p"], label="audit adjusted p") < 0.05
            and _finite(hypothesis["ci_low"], label="audit CI lower") > 0.0
            and safety["passed"] is True
        )
    return result


def formal_repeat_audit(
    primary: ValidatedFormalMatrix,
    repeat: ValidatedFormalMatrix,
    tolerance: float = 1e-4,
) -> dict[str, object]:
    if not isinstance(primary, ValidatedFormalMatrix) or not isinstance(repeat, ValidatedFormalMatrix):
        raise ValueError("repeat audit requires two validated matrices")
    if (primary.replicate_role, primary.replicate_index, primary.inference_eligible) != ("primary", 0, True):
        raise ValueError("repeat audit primary matrix identity is invalid")
    if (repeat.replicate_role, repeat.replicate_index, repeat.inference_eligible) != ("technical_repeat", 1, False):
        raise ValueError("repeat audit repeat matrix identity is invalid")
    limit = _finite(tolerance, label="repeat tolerance", nonnegative=True)
    if normalize_builtin(primary.formal_campaign_identity) != normalize_builtin(repeat.formal_campaign_identity):
        raise ValueError("primary and repeat matrices belong to different campaigns")
    primary_rows = _matrix_rows(primary)
    repeat_rows = _matrix_rows(repeat)
    per_pair: list[dict[str, object]] = []
    max_mscd = 0.0
    max_iou = 0.0
    failures: set[str] = set()
    for position, (first, second, raw_pair) in enumerate(zip(primary_rows, repeat_rows, primary._campaign_pairs)):
        pair = _mapping(raw_pair, label=f"campaign pair[{position}]")
        if (
            (first["task"], first["variant"]) != (second["task"], second["variant"])
            or pair["primary_run_id"] != first["run_id"]
            or pair["repeat_run_id"] != second["run_id"]
            or first["scientific_config_hash"] != second["scientific_config_hash"]
        ):
            raise ValueError("repeat matrix is not the exact campaign pairing")
        primary_mscd, primary_iou = _row_endpoints(first)
        repeat_mscd, repeat_iou = _row_endpoints(second)
        delta_mscd = repeat_mscd - primary_mscd
        delta_iou = repeat_iou - primary_iou
        mscd_ok = abs(delta_mscd) <= limit
        iou_ok = abs(delta_iou) <= limit
        max_mscd = max(max_mscd, abs(delta_mscd))
        max_iou = max(max_iou, abs(delta_iou))
        if not mscd_ok:
            failures.add("repeat_mscd_tolerance_exceeded")
        if not iou_ok:
            failures.add("repeat_worst_threshold_iou_tolerance_exceeded")
        per_pair.append(
            {
                "task": first["task"],
                "variant": first["variant"],
                "primary_run_id": first["run_id"],
                "repeat_run_id": second["run_id"],
                "task_mscd_px_difference": float(delta_mscd),
                "task_mscd_px_within_tolerance": mscd_ok,
                "worst_threshold_iou_difference": float(delta_iou),
                "worst_threshold_iou_within_tolerance": iou_ok,
            }
        )
    primary_retention = _retention_for_matrix(primary)
    repeat_retention = _retention_for_matrix(repeat)
    for matrix in (primary, repeat):
        for component in ("dic", "dts"):
            if "relevant_metric_invalid" in conditional_safety_checks(matrix, component)["failure_codes"]:
                failures.add("relevant_metric_invalid")
    unchanged = primary_retention == repeat_retention
    if not unchanged:
        failures.add("repeat_retention_changed")
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_repeat_audit_v1",
        "primary_matrix_sha256": primary.matrix_sha256,
        "repeat_matrix_sha256": repeat.matrix_sha256,
        "pair_count": 40,
        "tolerance": limit,
        "metrics": ["task_mscd_px", "worst_threshold_iou"],
        "per_pair": per_pair,
        "max_abs_task_mscd_px": float(max_mscd),
        "max_abs_worst_threshold_iou": float(max_iou),
        "primary_component_retention": primary_retention,
        "repeat_component_retention_audit_only": repeat_retention,
        "retention_unchanged": unchanged,
        "statistical_units": {"primary": 10, "repeat_audit_only": 10, "pooled": None},
        "status": "PASS" if not failures else "FAIL",
        "failure_codes": sorted(failures),
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    return payload


def _json_projection(schema: str, fields: Mapping[str, object]) -> bytes:
    payload: dict[str, object] = {"schema": schema, **normalize_builtin(fields)}
    payload["semantic_sha256"] = canonical_hash(payload)
    return canonical_json_bytes(payload)


def _tsv_bytes(header: Sequence[str], rows: Sequence[Sequence[object]]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(list(header))
    for row in rows:
        rendered = []
        for value in row:
            if value is None:
                rendered.append("")
            elif isinstance(value, bool):
                rendered.append("true" if value else "false")
            else:
                rendered.append(str(value))
        writer.writerow(rendered)
    return stream.getvalue().encode("utf-8")


def _mean_by_variant(matrix: ValidatedFormalMatrix) -> dict[str, object]:
    grouped: dict[str, dict[str, list[float]]] = {variant: {"task_mscd_px": [], "worst_threshold_iou": []} for variant in _VARIANTS}
    for row in _matrix_rows(matrix):
        mscd, iou = _row_endpoints(row)
        variant = str(row["variant"])
        grouped[variant]["task_mscd_px"].append(mscd)
        grouped[variant]["worst_threshold_iou"].append(iou)
    return {
        variant: {
            "mean_task_mscd_px": float(np.mean(values["task_mscd_px"], dtype=np.float64)),
            "mean_worst_threshold_iou": float(np.mean(values["worst_threshold_iou"], dtype=np.float64)),
        }
        for variant, values in grouped.items()
    }


def _build_export_bytes(
    primary: ValidatedFormalMatrix,
    repeat: ValidatedFormalMatrix,
    inference: Mapping[str, object],
    safety: Mapping[str, object],
    retention: Mapping[str, object],
    repeat_audit: Mapping[str, object],
) -> "OrderedDict[str, bytes]":
    if not isinstance(primary, ValidatedFormalMatrix) or not isinstance(repeat, ValidatedFormalMatrix):
        raise ValueError("exports require validated matrices")
    task_header = (
        "replicate_role", "replicate_index", "task", "variant", "run_id", "path",
        "scientific_config_hash", "expected_config_hash", "status_sha256",
        "artifact_manifest_sha256", "metrics_sha256", "task_mscd_px",
        "worst_threshold_iou", "mean_iou_0_05", "mean_recall_0_05", "mean_fpr_0_05",
    )
    task_rows: list[list[object]] = []
    for row in (*_matrix_rows(primary), *_matrix_rows(repeat)):
        metrics = _mapping(_mapping(row["metrics"], label="metrics")["payload"], label="metrics payload")
        aggregate = _aggregate_for(row)
        task_rows.append([
            row["replicate_role"], row["replicate_index"], row["task"], row["variant"], row["run_id"], row["path"],
            row["scientific_config_hash"], row["expected_config_hash"], _mapping(row["status"], label="status")["sha256"],
            _mapping(row["artifact_manifest"], label="artifact manifest")["sha256"], _mapping(row["metrics"], label="metrics")["sha256"],
            metrics["task_mscd_px"], metrics["worst_threshold_iou"], aggregate["mean_iou"], aggregate["mean_recall"], aggregate["mean_fpr"],
        ])
    contrasts = factorial_task_contrasts(primary)
    contrast_header = (
        "task", "dic_marginal_mscd_improvement_px", "dts_marginal_worst_threshold_iou_improvement",
        "m_d_interaction_px", "m_t_interaction", "dic_base_to_dic_mscd_improvement_px",
        "dic_dts_to_full_mscd_improvement_px", "dts_base_to_dts_worst_threshold_iou_improvement",
        "dts_dic_to_full_worst_threshold_iou_improvement",
    )
    contrast_rows = [[getattr(item, name) for name in contrast_header] for item in contrasts]
    inference_map = _mapping(inference, label="inference")
    safety_map = _mapping(safety, label="safety")
    confirmatory_fields = {name: inference_map[name] for name in ("matrix_sha256", "n_tasks", "family", "alpha", "alternative", "assumption", "bootstrap", "hypotheses")}
    conditional_fields = {"matrix_sha256": inference_map["matrix_sha256"], "n_tasks": inference_map["n_tasks"], "effects": inference_map["conditional_effects"]}
    interactions = _mapping(inference_map["interactions"], label="interactions")
    overall = "NO_MATERIAL_INTERACTION" if all(_mapping(interactions[name], label=name)["equivalent"] is True for name in ("m_d", "m_t")) else "CONDITIONAL_OR_COUPLED_EFFECTS"
    interaction_fields = {"matrix_sha256": inference_map["matrix_sha256"], "n_tasks": inference_map["n_tasks"], "interactions": interactions, "overall_wording": overall}
    safety_fields = {"matrix_sha256": inference_map["matrix_sha256"], "dic": safety_map["dic"], "dts": safety_map["dts"], "all_components_passed": safety_map["dic"]["passed"] is True and safety_map["dts"]["passed"] is True}
    aggregate_fields = {
        "primary_n_tasks": 10,
        "repeat_audit_n_tasks": 10,
        "pooled_n": None,
        "primary_by_variant": _mean_by_variant(primary),
        "repeat_audit_by_variant": _mean_by_variant(repeat),
        "inference_source": "primary_only",
        "disclaimer": "repeat_is_technical_audit_not_additional_statistical_units",
    }
    return OrderedDict(
        (
            ("task_rows.tsv", _tsv_bytes(task_header, task_rows)),
            ("factorial_task_contrasts.tsv", _tsv_bytes(contrast_header, contrast_rows)),
            ("confirmatory_tests.json", _json_projection("pcraft_soft_loss_v3_confirmatory_tests_v1", confirmatory_fields)),
            ("conditional_effects.json", _json_projection("pcraft_soft_loss_v3_conditional_effects_v1", conditional_fields)),
            ("interactions.json", _json_projection("pcraft_soft_loss_v3_interactions_v1", interaction_fields)),
            ("safety_checks.json", _json_projection("pcraft_soft_loss_v3_safety_bundle_v1", safety_fields)),
            ("retention_decision.json", canonical_json_bytes(retention)),
            ("repeat_audit.json", canonical_json_bytes(repeat_audit)),
            ("aggregate_metrics.json", _json_projection("pcraft_soft_loss_v3_aggregate_metrics_v1", aggregate_fields)),
        )
    )


def _analysis_root_from_path(path: Path) -> tuple[Path, Path]:
    candidate = _canonical_path(path, label="analysis path")
    if candidate.name not in {"analysis_manifest.json", "formal_conclusion.json"} or candidate.parent.name != "analysis":
        raise ValueError("analysis artifact is not at its exact canonical path")
    return candidate.parent.parent, candidate.parent


def _read_canonical(path: Path, *, label: str, root: Path) -> tuple[dict[str, object], bytes, str]:
    snapshot = registry._snapshot_file(path, label=label, root=root)
    payload = registry._decode_json(snapshot.data, label=label)
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError(f"{label} is not canonical JSON")
    return payload, snapshot.data, snapshot.sha256


def _immutable_file(path: Path, data: bytes) -> Path:
    root, _ = _analysis_root_from_path(path if path.name in {"analysis_manifest.json", "formal_conclusion.json"} else path.parent / "analysis_manifest.json")
    if os.path.lexists(path):
        current = registry._snapshot_file(path, label=f"immutable {path.name}", root=root)
        if current.data != data:
            raise FileExistsError(f"immutable output differs: {path}")
        return path
    registry._publish_bundle({path: data}, output_root=root)
    return path


def _conclusion_payload(
    analysis_manifest: Mapping[str, object],
    retention: Mapping[str, object],
    repeat_audit: Mapping[str, object],
    *,
    analysis_path: Path,
    analysis_sha256: str,
) -> dict[str, object]:
    status = "LOCKED" if repeat_audit["status"] == "PASS" else "FAILED_TECHNICAL_AUDIT"
    interactions_path = analysis_path.parent / "interactions.json"
    interactions: Mapping[str, object] = {}
    if interactions_path.is_file():
        interactions_payload, _, _ = _read_canonical(interactions_path, label="interactions export", root=analysis_path.parent.parent)
        interactions = _mapping(interactions_payload.get("interactions", {}), label="interaction results")
    wording = {
        name: (_mapping(interactions[name], label=name)["wording"] if name in interactions else "CONDITIONAL_OR_COUPLED_EFFECTS")
        for name in ("m_d", "m_t")
    }
    wording["overall"] = "NO_MATERIAL_INTERACTION" if all(value == "NO_MATERIAL_INTERACTION" for value in wording.values()) else "CONDITIONAL_OR_COUPLED_EFFECTS"
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_conclusion_v1",
        "status": status,
        "analysis_manifest": {"path": str(analysis_path), "sha256": analysis_sha256, "semantic_sha256": analysis_manifest["semantic_sha256"]},
        "formal_campaign": analysis_manifest["formal_campaign"],
        "run_evidence": analysis_manifest["run_evidence"],
        "retention_criteria": {
            "holm_alpha": 0.05,
            "holm_adjusted_p_strict_upper": 0.05,
            "bootstrap_ci_low_strict_lower": 0.0,
            "conditional_primary_strict_lower": 0.0,
            "mean_iou_0_05_delta_min": -0.001,
            "per_task_recall_0_05_delta_min": -0.005,
            "mean_fpr_0_05_delta_max": 0.001,
            "interaction_can_rescue": False,
        },
        "component_decisions": retention["components"],
        "retained_formula": retention["retained_formula"],
        "interaction_wording": wording,
        "repeat_audit": {
            name: repeat_audit[name]
            for name in (
                "schema", "primary_matrix_sha256", "repeat_matrix_sha256", "pair_count", "tolerance", "status",
                "max_abs_task_mscd_px", "max_abs_worst_threshold_iou", "retention_unchanged", "semantic_sha256",
            )
        },
        "executed_sources": analysis_manifest["executed_sources"],
        "failure_codes": list(repeat_audit["failure_codes"]),
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    return payload


def write_formal_conclusion(
    path: Path,
    *,
    analysis_manifest: Mapping[str, object],
    retention: Mapping[str, object],
    repeat_audit: Mapping[str, object],
) -> Path:
    destination = _canonical_path(path, label="formal conclusion path")
    root, analysis_dir = _analysis_root_from_path(destination)
    if destination != analysis_dir / "formal_conclusion.json":
        raise ValueError("formal conclusion must use its exact registered path")
    manifest_path = analysis_dir / "analysis_manifest.json"
    manifest_payload, manifest_bytes, manifest_sha = _read_canonical(manifest_path, label="analysis manifest", root=root)
    if normalize_builtin(analysis_manifest) != manifest_payload:
        raise ValueError("passed analysis manifest differs from bound bytes")
    campaign = _identity(manifest_payload["formal_campaign"], label="analysis formal campaign")
    validated = validate_analysis_manifest(manifest_path, expected_campaign_sha256=str(campaign["sha256"]))
    if validated != manifest_payload:
        raise ValueError("analysis validator returned different data")
    retention_path = analysis_dir / "retention_decision.json"
    repeat_path = analysis_dir / "repeat_audit.json"
    if retention_path.is_file():
        current_retention, _, _ = _read_canonical(retention_path, label="retention export", root=root)
        if current_retention != normalize_builtin(retention):
            raise ValueError("passed retention differs from its export")
    if repeat_path.is_file():
        current_repeat, _, _ = _read_canonical(repeat_path, label="repeat audit export", root=root)
        if current_repeat != normalize_builtin(repeat_audit):
            raise ValueError("passed repeat audit differs from its export")
    payload = _conclusion_payload(manifest_payload, normalize_builtin(retention), normalize_builtin(repeat_audit), analysis_path=manifest_path, analysis_sha256=hashlib.sha256(manifest_bytes).hexdigest())
    registry._publish_bundle({destination: canonical_json_bytes(payload)}, output_root=root)
    return destination


def _canonical_identity_from_file(
    path: Path,
    *,
    label: str,
    root: Path,
    semantic_required: bool = True,
) -> tuple[dict[str, object], dict[str, object], object]:
    snapshot = registry._snapshot_file(path, label=label, root=root)
    payload = registry._decode_json(snapshot.data, label=label)
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError(f"{label} is not canonical JSON")
    record: dict[str, object] = {"path": str(snapshot.path), "sha256": snapshot.sha256}
    if semantic_required:
        semantic = _digest(payload.get("semantic_sha256"), label=f"{label}.semantic_sha256")
        record["semantic_sha256"] = semantic
    return record, payload, snapshot


def _analysis_source_records() -> dict[str, object]:
    import experiments.run_soft_loss_v3 as runner
    import experiments.summarize_soft_loss_v3 as summarizer

    logical_paths = OrderedDict(
        (
            ("experiments/soft_loss_v3_statistics.py", Path(__file__).resolve(strict=True)),
            ("experiments/summarize_soft_loss_v3.py", Path(summarizer.__file__).resolve(strict=True)),
            ("experiments/run_soft_loss_v3.py", Path(runner.__file__).resolve(strict=True)),
            ("experiments/soft_loss_v3_registry.py", Path(registry.__file__).resolve(strict=True)),
        )
    )
    project = Path(__file__).resolve(strict=True).parents[1]
    result: dict[str, object] = {}
    for logical, path in logical_paths.items():
        if path != project / logical:
            raise ValueError("analysis source is not the authoritative project file")
        snapshot = registry._snapshot_file(path, label=f"analysis source {logical}", root=project)
        result[logical] = {"path": str(snapshot.path), "sha256": snapshot.sha256}
    return result


def _verify_analysis_dependencies(
    manifest: Mapping[str, object], *, root: Path
) -> None:
    """Revalidate every dependency bound by a pending analysis commit."""

    output_root = _canonical_path(root, label="analysis dependency root")
    fixed_paths = {
        "formal_campaign": output_root / "manifests" / "formal_campaign.json",
        "formal_suite": output_root / "manifests" / "formal_suite.json",
        "master_manifest": output_root / "experiment_manifest.json",
        "frozen_weights": output_root / "analysis" / "weights_frozen.json",
    }
    for name, expected_path in fixed_paths.items():
        record = _identity(manifest[name], label=f"analysis {name}")
        if record["path"] != str(expected_path):
            raise ValueError(f"analysis {name} path changed")
        snapshot = registry._snapshot_file(
            expected_path, label=f"analysis {name}", root=output_root
        )
        if snapshot.sha256 != record["sha256"]:
            raise ValueError(f"analysis dependency changed during publication: {name}")

    specification = _identity(
        manifest["specification"], label="analysis specification"
    )
    specification_path = Path(str(specification["path"]))
    specification_snapshot = registry._snapshot_file(
        specification_path,
        label="analysis specification",
        root=specification_path.parent,
    )
    if specification_snapshot.sha256 != specification["sha256"]:
        raise ValueError("analysis specification changed during publication")

    stages = _mapping(manifest["source_stages"], label="analysis source stages")
    _exact_keys(stages, {"primary", "repeat"}, label="analysis source stages")
    for role, stage in (
        ("primary", "formal256_primary"),
        ("repeat", "formal256_repeat"),
    ):
        record = _identity(stages[role], label=f"analysis {role} stage")
        expected_path = output_root / "manifests" / "stages" / f"{stage}.json"
        if record["path"] != str(expected_path):
            raise ValueError(f"analysis {role} stage path changed")
        snapshot = registry._snapshot_file(
            expected_path, label=f"analysis {role} stage", root=output_root
        )
        if snapshot.sha256 != record["sha256"]:
            raise ValueError(f"analysis {role} stage changed during publication")

    sources = _mapping(manifest["executed_sources"], label="analysis executed sources")
    if normalize_builtin(sources) != _analysis_source_records():
        raise ValueError("analysis executed source changed during publication")

    suite_path = fixed_paths["formal_suite"]
    suite_payload, _, _ = _read_canonical(
        suite_path, label="analysis formal suite", root=output_root
    )
    _, task_inputs = _validate_formal_suite(suite_payload)
    for task in _TASKS:
        for input_record in task_inputs[task]:
            input_path = Path(str(input_record["resolved_path"]))
            snapshot = registry._snapshot_file(
                input_path,
                label=f"analysis input {task} {input_record['axis']}",
                root=input_path.parent,
            )
            if snapshot.sha256 != input_record["sha256"]:
                raise ValueError("analysis formal input changed during publication")

    evidence = _sequence(manifest["run_evidence"], label="analysis run evidence")
    expected_cells = [(task, variant) for task in _TASKS for variant in _VARIANTS]
    if len(evidence) != 40:
        raise ValueError("analysis run evidence must contain exactly forty pairs")
    role_keys = {
        "run_id",
        "path",
        "expected_config_hash",
        "status_sha256",
        "artifact_manifest_sha256",
        "metrics_sha256",
    }
    for index, (raw_pair, (task, variant)) in enumerate(zip(evidence, expected_cells)):
        pair = _mapping(raw_pair, label=f"analysis run evidence[{index}]")
        _exact_keys(
            pair,
            {"task", "variant", "scientific_config_hash", "primary", "repeat"},
            label=f"analysis run evidence[{index}]",
        )
        if (pair["task"], pair["variant"]) != (task, variant):
            raise ValueError("analysis run evidence order changed")
        _digest(pair["scientific_config_hash"], label="analysis scientific config")
        for role, stage, replicate_index in (
            ("primary", "formal256_primary", 0),
            ("repeat", "formal256_repeat", 1),
        ):
            record = _mapping(pair[role], label=f"analysis {role} evidence[{index}]")
            _exact_keys(record, role_keys, label=f"analysis {role} evidence[{index}]")
            run_id = f"{stage}/steps_2000/{task}/{variant}/replicate_{replicate_index}/seed_17"
            run_path = output_root / run_id
            if record["run_id"] != run_id or record["path"] != str(run_path):
                raise ValueError("analysis run evidence identity changed")
            _digest(record["expected_config_hash"], label="analysis expected config")
            for filename, hash_key in (
                ("STATUS.json", "status_sha256"),
                ("artifact_manifest.json", "artifact_manifest_sha256"),
                ("metrics.json", "metrics_sha256"),
            ):
                expected_sha = _digest(
                    record[hash_key], label=f"analysis {role} {filename} sha256"
                )
                snapshot = registry._snapshot_file(
                    run_path / filename,
                    label=f"analysis {role} {filename}",
                    root=output_root,
                )
                if snapshot.sha256 != expected_sha:
                    raise ValueError(
                        f"analysis {role} run evidence changed during publication"
                    )


def _row_evidence(row: Mapping[str, object]) -> dict[str, object]:
    status = _mapping(row["status"], label="analysis STATUS evidence")
    artifact = _mapping(row["artifact_manifest"], label="analysis artifact evidence")
    metrics = _mapping(row["metrics"], label="analysis metrics evidence")
    return {
        "run_id": row["run_id"],
        "path": row["path"],
        "expected_config_hash": row["expected_config_hash"],
        "status_sha256": status["sha256"],
        "artifact_manifest_sha256": artifact["sha256"],
        "metrics_sha256": metrics["sha256"],
    }


def _analysis_material(
    root: Path,
    *,
    draws: int = _LOCKED_DRAWS,
) -> tuple[
    dict[str, object],
    "OrderedDict[str, bytes]",
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Recollect and deterministically rebuild all formal analysis bytes."""

    output_root = _canonical_path(root, label="analysis output root")
    if not output_root.is_dir():
        raise ValueError("analysis output root must be an existing directory")
    draw_count = _exact_int(draws, label="formal analysis draws", positive=True)
    if draw_count != _LOCKED_DRAWS:
        raise ValueError("published formal analysis requires exactly 100000 bootstrap draws")
    import experiments.summarize_soft_loss_v3 as summarizer

    master_record, master, master_snapshot = _canonical_identity_from_file(
        output_root / "experiment_manifest.json",
        label="master manifest",
        root=output_root,
    )
    suite_record, suite, suite_snapshot = _canonical_identity_from_file(
        output_root / "manifests" / "formal_suite.json",
        label="formal suite",
        root=output_root,
    )
    campaign_record, campaign, campaign_snapshot = _canonical_identity_from_file(
        output_root / "manifests" / "formal_campaign.json",
        label="formal campaign",
        root=output_root,
    )
    weights_record, weights, weights_snapshot = _canonical_identity_from_file(
        output_root / "analysis" / "weights_frozen.json",
        label="frozen weights",
        root=output_root,
    )
    primary_stage_record, _, primary_stage_snapshot = _canonical_identity_from_file(
        output_root / "manifests" / "stages" / "formal256_primary.json",
        label="formal primary stage",
        root=output_root,
    )
    repeat_stage_record, _, repeat_stage_snapshot = _canonical_identity_from_file(
        output_root / "manifests" / "stages" / "formal256_repeat.json",
        label="formal repeat stage",
        root=output_root,
    )
    design_spec = _mapping(master.get("design_spec"), label="master design specification")
    _exact_keys(design_spec, {"path", "sha256"}, label="master design specification")
    spec_path = Path(_string(design_spec["path"], label="design specification path"))
    spec_snapshot = registry._snapshot_file(spec_path, label="design specification", root=spec_path.parent)
    if spec_snapshot.sha256 != _digest(design_spec["sha256"], label="design specification sha256"):
        raise ValueError("design specification byte identity is stale")
    specification_record = {
        "path": str(spec_snapshot.path),
        "sha256": spec_snapshot.sha256,
        "semantic_sha256": spec_snapshot.sha256,
    }
    primary_rows = summarizer.collect_stage_rows(output_root, "formal256_primary")
    repeat_rows = summarizer.collect_stage_rows(output_root, "formal256_repeat")
    primary = validate_formal_matrix(
        primary_rows,
        formal_suite=suite,
        formal_campaign=campaign,
        replicate_role="primary",
        replicate_index=0,
    )
    repeat = validate_formal_matrix(
        repeat_rows,
        formal_suite=suite,
        formal_campaign=campaign,
        replicate_role="technical_repeat",
        replicate_index=1,
    )
    inference = formal_factorial_inference(primary, draws=draw_count)
    safety = {
        "dic": conditional_safety_checks(primary, "dic"),
        "dts": conditional_safety_checks(primary, "dts"),
    }
    retention = retention_decision(inference, safety)
    repeat_audit = formal_repeat_audit(primary, repeat, tolerance=1e-4)
    exports = _build_export_bytes(primary, repeat, inference, safety, retention, repeat_audit)
    primary_by_cell = {(str(row["task"]), str(row["variant"])): row for row in _matrix_rows(primary)}
    repeat_by_cell = {(str(row["task"]), str(row["variant"])): row for row in _matrix_rows(repeat)}
    run_evidence = []
    for task in _TASKS:
        for variant in _VARIANTS:
            first = primary_by_cell[(task, variant)]
            second = repeat_by_cell[(task, variant)]
            if first["scientific_config_hash"] != second["scientific_config_hash"]:
                raise ValueError("analysis run pair scientific identity changed")
            run_evidence.append(
                {
                    "task": task,
                    "variant": variant,
                    "scientific_config_hash": first["scientific_config_hash"],
                    "primary": _row_evidence(first),
                    "repeat": _row_evidence(second),
                }
            )
    sources = _analysis_source_records()
    manifest: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_analysis_v1",
        "status": "COMPLETE",
        "formal_campaign": campaign_record,
        "formal_suite": suite_record,
        "master_manifest": master_record,
        "specification": specification_record,
        "frozen_weights": weights_record,
        "source_stages": {"primary": primary_stage_record, "repeat": repeat_stage_record},
        "matrices": {
            "primary": {"replicate_role": "primary", "replicate_index": 0, "row_count": 40, "matrix_sha256": primary.matrix_sha256},
            "repeat": {"replicate_role": "technical_repeat", "replicate_index": 1, "row_count": 40, "matrix_sha256": repeat.matrix_sha256},
        },
        "run_evidence": run_evidence,
        "executed_sources": sources,
        "statistical_policy": {
            "unit": "task",
            "primary_n": 10,
            "repeat_audit_n": 10,
            "pooled_n": None,
            "sign_flip": {
                "alternative": "greater",
                "tie_tolerance": 1e-15,
                "enumerated": True,
                "assumption": "task_level_symmetry_exchangeability",
            },
            "bootstrap": {
                "draws": _LOCKED_DRAWS,
                "seed": _LOCKED_SEED,
                "ci_level": 0.95,
                "method": "hyndman_fan_type_7",
                "joint_task_indices": True,
            },
            "holm": {"family": ["dic", "dts"], "alpha": 0.05},
            "repeat": {"tolerance": 0.0001, "inference_eligible": False},
        },
        "exports_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in exports.items()},
    }
    manifest["semantic_sha256"] = canonical_hash(manifest)
    for snapshot, label, snapshot_root in (
        (master_snapshot, "master manifest", output_root),
        (suite_snapshot, "formal suite", output_root),
        (campaign_snapshot, "formal campaign", output_root),
        (weights_snapshot, "frozen weights", output_root),
        (primary_stage_snapshot, "formal primary stage", output_root),
        (repeat_stage_snapshot, "formal repeat stage", output_root),
        (spec_snapshot, "design specification", spec_path.parent),
    ):
        registry._verify_snapshot_unchanged(snapshot, label=label, root=snapshot_root)
    if _analysis_source_records() != sources:
        raise ValueError("analysis source changed during authenticated recomputation")
    return manifest, exports, retention, repeat_audit, inference


def publish_formal_analysis(
    root: Path,
    analysis_dir: Path,
) -> Mapping[str, object]:
    """Authenticate both matrices and publish conclusion-last formal output."""

    output_root = _canonical_path(root, label="formal output root")
    expected_analysis = output_root / "analysis"
    if not isinstance(analysis_dir, Path) or analysis_dir != expected_analysis:
        raise ValueError("analysis directory is not the exact canonical output path")
    _canonical_path(analysis_dir, label="analysis directory")
    if not analysis_dir.is_dir():
        raise ValueError("analysis directory must already exist with frozen weights")
    manifest, exports, retention, repeat_audit, _ = _analysis_material(output_root)
    manifest_bytes = canonical_json_bytes(manifest)
    files: "OrderedDict[Path, bytes]" = OrderedDict(
        (analysis_dir / name, data) for name, data in exports.items()
    )
    files[analysis_dir / "analysis_manifest.json"] = manifest_bytes

    def precommit() -> None:
        _verify_analysis_dependencies(manifest, root=output_root)

    registry._publish_bundle(files, output_root=output_root, precommit=precommit)
    validated = validate_analysis_manifest(
        analysis_dir / "analysis_manifest.json",
        expected_campaign_sha256=str(_mapping(manifest["formal_campaign"], label="formal campaign")["sha256"]),
    )
    if validated != manifest:
        raise ValueError("published analysis differs after authenticated recomputation")
    conclusion_path = write_formal_conclusion(
        analysis_dir / "formal_conclusion.json",
        analysis_manifest=manifest,
        retention=retention,
        repeat_audit=repeat_audit,
    )
    conclusion_snapshot = registry._snapshot_file(conclusion_path, label="formal conclusion", root=output_root)
    return validate_formal_conclusion(
        conclusion_path,
        expected_analysis_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        expected_campaign_sha256=str(_mapping(manifest["formal_campaign"], label="formal campaign")["sha256"]),
    )


def validate_analysis_manifest(
    path: Path,
    *,
    expected_campaign_sha256: str,
) -> Mapping[str, object]:
    candidate = _canonical_path(path, label="analysis manifest path", require_exists=True)
    root, analysis_dir = _analysis_root_from_path(candidate)
    expected_campaign = _digest(expected_campaign_sha256, label="expected campaign sha256")
    payload, _, _ = _read_canonical(candidate, label="analysis manifest", root=root)
    _exact_keys(
        payload,
        {"schema", "status", "formal_campaign", "formal_suite", "master_manifest", "specification", "frozen_weights", "source_stages", "matrices", "run_evidence", "executed_sources", "statistical_policy", "exports_sha256", "semantic_sha256"},
        label="analysis manifest",
    )
    if payload["schema"] != "pcraft_soft_loss_v3_analysis_v1" or payload["status"] != "COMPLETE":
        raise ValueError("analysis manifest schema/status is invalid")
    _semantic_payload(payload, label="analysis manifest")
    campaign_identity = _identity(payload["formal_campaign"], label="analysis formal campaign")
    if campaign_identity["sha256"] != expected_campaign:
        raise ValueError("analysis manifest campaign SHA differs from caller expectation")
    exports = _mapping(payload["exports_sha256"], label="analysis export hashes")
    _exact_keys(exports, set(_EXPORT_NAMES), label="analysis export hashes")
    actual_export_bytes: dict[str, bytes] = {}
    for name in _EXPORT_NAMES:
        digest = _digest(exports[name], label=f"analysis export {name}")
        snapshot = registry._snapshot_file(analysis_dir / name, label=f"analysis export {name}", root=root)
        if snapshot.sha256 != digest:
            raise ValueError(f"analysis export drift: {name}")
        actual_export_bytes[name] = snapshot.data
    policy = _mapping(payload["statistical_policy"], label="analysis statistical policy")
    bootstrap = _mapping(policy.get("bootstrap"), label="analysis bootstrap policy")
    if type(bootstrap.get("draws")) is not int or bootstrap["draws"] != _LOCKED_DRAWS:
        raise ValueError("analysis bootstrap policy is not the registered 100000 draws")
    rebuilt_manifest, rebuilt_exports, _, _, _ = _analysis_material(root, draws=_LOCKED_DRAWS)
    if rebuilt_manifest != payload:
        raise ValueError("analysis manifest differs from authenticated recomputation")
    for name in _EXPORT_NAMES:
        if rebuilt_exports[name] != actual_export_bytes[name]:
            raise ValueError(f"analysis export differs from authenticated recomputation: {name}")
    return normalize_builtin(payload)


def validate_formal_conclusion(
    path: Path,
    *,
    expected_analysis_sha256: str,
    expected_campaign_sha256: str,
) -> Mapping[str, object]:
    candidate = _canonical_path(path, label="formal conclusion path", require_exists=True)
    root, analysis_dir = _analysis_root_from_path(candidate)
    expected_analysis = _digest(expected_analysis_sha256, label="expected analysis sha256")
    expected_campaign = _digest(expected_campaign_sha256, label="expected campaign sha256")
    payload, _, _ = _read_canonical(candidate, label="formal conclusion", root=root)
    _exact_keys(payload, {"schema", "status", "analysis_manifest", "formal_campaign", "run_evidence", "retention_criteria", "component_decisions", "retained_formula", "interaction_wording", "repeat_audit", "executed_sources", "failure_codes", "semantic_sha256"}, label="formal conclusion")
    if payload["schema"] != "pcraft_soft_loss_v3_conclusion_v1" or payload["status"] not in {"LOCKED", "FAILED_TECHNICAL_AUDIT"}:
        raise ValueError("formal conclusion schema/status is invalid")
    _semantic_payload(payload, label="formal conclusion")
    analysis_identity = _identity(payload["analysis_manifest"], label="conclusion analysis identity")
    campaign_identity = _identity(payload["formal_campaign"], label="conclusion campaign identity")
    if analysis_identity["path"] != str(analysis_dir / "analysis_manifest.json") or analysis_identity["sha256"] != expected_analysis or campaign_identity["sha256"] != expected_campaign:
        raise ValueError("formal conclusion caller-bound identity mismatch")
    manifest = validate_analysis_manifest(analysis_dir / "analysis_manifest.json", expected_campaign_sha256=expected_campaign)
    manifest_snapshot = registry._snapshot_file(analysis_dir / "analysis_manifest.json", label="analysis manifest", root=root)
    if manifest_snapshot.sha256 != expected_analysis or manifest["formal_campaign"] != payload["formal_campaign"] or manifest["run_evidence"] != payload["run_evidence"] or manifest["executed_sources"] != payload["executed_sources"]:
        raise ValueError("formal conclusion differs from current analysis")
    retention, _, _ = _read_canonical(analysis_dir / "retention_decision.json", label="retention export", root=root)
    repeat, _, _ = _read_canonical(analysis_dir / "repeat_audit.json", label="repeat audit export", root=root)
    expected_payload = _conclusion_payload(
        manifest,
        retention,
        repeat,
        analysis_path=analysis_dir / "analysis_manifest.json",
        analysis_sha256=expected_analysis,
    )
    if expected_payload != payload:
        raise ValueError("formal conclusion differs from authenticated recomputation")
    return normalize_builtin(payload)


__all__ = [
    "FactorialTaskContrast",
    "ValidatedFormalMatrix",
    "validate_formal_matrix",
    "factorial_task_contrasts",
    "exact_sign_flip_greater",
    "paired_bootstrap",
    "holm_adjust",
    "formal_factorial_inference",
    "conditional_safety_checks",
    "retention_decision",
    "formal_repeat_audit",
    "write_formal_conclusion",
    "validate_formal_conclusion",
    "validate_analysis_manifest",
    "publish_formal_analysis",
]
