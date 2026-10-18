"""Fail-closed engineering summaries for the registered soft-loss-v3 study."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import fields
from fractions import Fraction
import hashlib
from io import BytesIO
import json
import math
from numbers import Real
import os
from pathlib import Path, PurePosixPath
import statistics
import sys
from typing import Callable

import numpy as np
import torch

import experiments.run_soft_loss_v3 as runner
import experiments.soft_loss_v3_registry as registry
from shadowforge_pipeline.soft_voxel_v3 import trainer as task8_trainer
from shadowforge_pipeline.soft_voxel_v3.diagnostics import ComponentGradientSample
from shadowforge_pipeline.soft_voxel_v3.losses import (
    dic_schedule,
    dts_schedule,
    resolve_profile,
)
from shadowforge_pipeline.soft_voxel_v3.provenance import (
    canonical_hash,
    canonical_json_bytes,
)


_PRIMARY = ("l12_bal", "dic_only", "dts_only", "v3_full")
_DEVELOPMENT = ("arcane_casters", "blade_heroes", "stage_dancers")
_FORMAL = tuple(f"connected_{index:02d}" for index in range(1, 11))
_THRESHOLDS = ("0.04", "0.05", "0.06")
_SOURCE_STAGES = (
    "smoke128",
    "screen128",
    "confirm128",
    "cuda2step256",
    "resource256",
    "transfer256",
    "sentinel_repeat256",
    "formal256_primary",
    "formal256_repeat",
    "appendix256",
)
_ORDINARY_EDGES = {
    "smoke128": ("screen128", "gradient_budget", None),
    "screen128": ("confirm128", "development_factorial", None),
    "confirm128": ("cuda2step256", "development_factorial", None),
    "cuda2step256": ("resource256", "cuda_integration", None),
    "resource256": ("transfer256", "resource_feasibility", "cuda_integration"),
    "transfer256": ("sentinel_repeat256", "development_factorial", "gradient_budget"),
    "sentinel_repeat256": ("formal256_primary", "repeat_tolerance", None),
}
_COUNTS = {
    "smoke128": 4,
    "screen128": 12,
    "confirm128": 12,
    "cuda2step256": 4,
    "resource256": 2,
    "transfer256": 12,
    "sentinel_repeat256": 4,
    "formal256_primary": 40,
    "formal256_repeat": 40,
    "appendix256": 30,
}
_COLLECTION_CODES = {
    "run_incomplete",
    "artifact_invalid",
    "metrics_schema_invalid",
    "metrics_invalid_flag",
    "gradient_audit_invalid",
    "resource_schema_invalid",
    "history_invalid",
    "parameter_unchanged",
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
_GATE_CODES = {
    "gradient_budget": {
        "input_schema_invalid",
        "probe_matrix_invalid",
        "eligible_component_inactive",
        "component_ratio_out_of_range",
        "combined_ratio_median_exceeded",
        "combined_ratio_p95_exceeded",
        "adverse_cosine_fraction_exceeded",
        "endpoint_diagnostic_invalid",
    },
    "development_factorial": {
        "input_schema_invalid",
        "factorial_matrix_invalid",
        "mscd_denominator_invalid",
        "dic_conditional_reversed",
        "dts_conditional_reversed",
        "dic_threshold_missed",
        "dts_threshold_missed",
        "iou_noninferiority_failed",
        "recall_noninferiority_failed",
        "fpr_guard_failed",
        "threshold_range_failed",
        "metric_invalid",
    },
    "cuda_integration": {
        "input_schema_invalid",
        "integration_matrix_invalid",
        "integration_branch_inactive",
        "integration_nonfinite",
        "soft_only_violation",
    },
    "resource_feasibility": {
        "input_schema_invalid",
        "resource_matrix_invalid",
        "physical_memory_invalid",
        "resource_summary_mismatch",
        "peak_memory_exceeded",
        "baseline_p95_nonpositive",
        "runtime_ratio_exceeded",
        "nonfinite_value",
    },
    "repeat_tolerance": {
        "input_schema_invalid",
        "repeat_pair_invalid",
        "repeat_mscd_tolerance_exceeded",
        "repeat_worst_iou_tolerance_exceeded",
        "metric_invalid",
    },
}
_LOCKED_INTERPRETER = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")


class _EvidenceFailure(ValueError):
    def __init__(self, code: str, message: str, *, run_id: str | None = None) -> None:
        if code not in _COLLECTION_CODES:
            raise ValueError("unregistered collection failure code")
        super().__init__(message)
        self.code = code
        self.run_id = run_id


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence_of_mappings(value: object, *, label: str) -> list[Mapping[str, object]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a non-string sequence")
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        result.append(_mapping(item, label=f"{label}[{index}]"))
    return result


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} has missing or extra keys")


def _finite(value: object, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite numeric scalar")
    normalized = float(value)
    if not math.isfinite(normalized) or (nonnegative and normalized < 0.0):
        raise ValueError(f"{label} must be finite" + (" and non-negative" if nonnegative else ""))
    return normalized


def _public_nonnegative(value: object, *, label: str) -> float:
    return _finite(value, label=label, nonnegative=True)


def _exact_int(value: object, *, label: str, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        raise ValueError(f"{label} must be an exact {'positive ' if positive else ''}int")
    return value


def _canonical_percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile pool must be non-empty")
    position = Fraction(str(q)) * (len(ordered) - 1)
    lower = position.numerator // position.denominator
    upper = lower if position.denominator == 1 else lower + 1
    if lower == upper:
        return float(ordered[lower])
    fraction = float(position - lower)
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _gate_result(
    gate: str,
    *,
    criteria: Mapping[str, object],
    observed: Mapping[str, object],
    failures: Sequence[str],
) -> dict[str, object]:
    if gate not in _GATE_CODES:
        raise ValueError("unknown gate kind")
    codes = sorted(set(failures))
    if not set(codes).issubset(_GATE_CODES[gate]):
        raise ValueError("gate result contains an unregistered failure code")
    return {
        "schema": "pcraft_soft_loss_v3_gate_result_v1",
        "gate": gate,
        "status": "PASS" if not codes else "STOP",
        "criteria": dict(criteria),
        "observed": dict(observed),
        "failure_codes": codes,
    }


def _row_metric_payload(row: Mapping[str, object]) -> Mapping[str, object]:
    _exact_keys(row, _ROW_KEYS, label="collected row")
    if row.get("schema") != "pcraft_soft_loss_v3_collected_row_v1":
        raise ValueError("collected row schema mismatch")
    run_spec = _mapping(row.get("run_spec"), label="collected row RunSpec")
    _exact_keys(
        run_spec,
        set(runner.RunSpec.__dataclass_fields__),
        label="collected row RunSpec",
    )
    for name in (
        "stage",
        "run_id",
        "task",
        "variant",
        "replicate_role",
        "replicate_index",
        "inference_eligible",
        "resolution",
        "steps",
    ):
        if run_spec.get(name) != row.get(name):
            raise ValueError("collected row scalar identity differs from RunSpec")
    if run_spec.get("output_path") != row.get("path"):
        raise ValueError("collected row path differs from RunSpec")
    metrics = _mapping(row.get("metrics"), label="collected row metrics")
    _exact_keys(metrics, {"path", "sha256", "payload"}, label="collected row metrics")
    payload = _mapping(metrics["payload"], label="metrics payload")
    return payload


def _primary_metric(row: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object]]:
    payload = _row_metric_payload(row)
    thresholds = _mapping(payload.get("thresholds"), label="metrics thresholds")
    if set(thresholds) != set(_THRESHOLDS):
        raise ValueError("metrics thresholds are not exact")
    primary = _mapping(thresholds["0.05"], label="primary threshold")
    aggregate = _mapping(primary.get("aggregate"), label="primary aggregate")
    return payload, aggregate


def _canonical_root(root: object) -> Path:
    if not isinstance(root, Path):
        raise ValueError("output root must be a Path")
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("output root must be absolute and non-traversing")
    runner._reject_symlink_components(root, label="output root")
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("output root must already exist") from error
    if resolved != root or not resolved.is_dir():
        raise ValueError("output root must be one canonical directory")
    return resolved


def _stage_context(root: Path, stage: str) -> runner.StageContext:
    upstream = runner.required_upstream(stage)
    authorization: Path | None = None
    if upstream is not None:
        if stage == "appendix256":
            authorization = root / "analysis" / "formal_conclusion.json"
        else:
            authorization = (
                root
                / "authorizations"
                / f"{upstream[0]}__to__{stage}.json"
            )
    campaign = (
        root / "manifests" / "formal_campaign.json"
        if stage in {"formal256_primary", "formal256_repeat"}
        else None
    )
    return runner.StageContext(
        output_root=root,
        master_manifest=root / "experiment_manifest.json",
        development_suite=root / "manifests" / "development_suite.json",
        formal_suite=root / "manifests" / "formal_suite.json",
        frozen_weights=root / "analysis" / "weights_frozen.json",
        authorization=authorization,
        formal_campaign=campaign,
    )


def _snapshot_json(
    path: Path, *, label: str, root: Path
) -> tuple[dict[str, object], object]:
    snapshot = registry._snapshot_file(path, label=label, root=root)
    payload = registry._decode_json(snapshot.data, label=label)
    if snapshot.data != canonical_json_bytes(payload):
        raise ValueError(f"{label} is not canonical JSON")
    return payload, snapshot


def _identity_record(path: Path, snapshot: object, semantic: object) -> dict[str, object]:
    if type(semantic) is not str or len(semantic) != 64:
        raise ValueError("semantic identity must be a SHA-256 digest")
    return {
        "path": str(path),
        "sha256": str(getattr(snapshot, "sha256")),
        "semantic_sha256": semantic,
    }


def _same_number(left: object, right: object, *, label: str) -> float:
    first = _finite(left, label=label)
    second = _finite(right, label=f"expected {label}")
    if not math.isclose(first, second, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} disagrees with recomputed value")
    return first


_VIEW_METRIC_KEYS = {
    "tp",
    "fp",
    "fn",
    "tn",
    "iou",
    "dice",
    "fpr",
    "recall",
    "boundary_f1_2px",
    "mscd_px",
    "p95_symmetric_distance_px",
    "invalid_iou_denominator",
    "invalid_dice_denominator",
    "invalid_background_denominator",
    "invalid_foreground_denominator",
    "invalid_predicted_boundary",
    "invalid_target_boundary",
}
_AGGREGATE_METRIC_KEYS = {
    "tp",
    "fp",
    "fn",
    "tn",
    "mean_iou",
    "mean_dice",
    "mean_fpr",
    "mean_recall",
    "mean_boundary_f1_2px",
    "mean_mscd_px",
    "mean_p95_symmetric_distance_px",
    "invalid_iou_denominator",
    "invalid_dice_denominator",
    "invalid_background_denominator",
    "invalid_foreground_denominator",
    "invalid_predicted_boundary",
    "invalid_target_boundary",
    "invalid_task_metric",
}
_METRIC_FLAGS = (
    "invalid_iou_denominator",
    "invalid_dice_denominator",
    "invalid_background_denominator",
    "invalid_foreground_denominator",
    "invalid_predicted_boundary",
    "invalid_target_boundary",
)


def _parse_view_metric(value: object, *, label: str) -> dict[str, object]:
    row = _mapping(value, label=label)
    _exact_keys(row, _VIEW_METRIC_KEYS, label=label)
    counts: dict[str, int] = {}
    for name in ("tp", "fp", "fn", "tn"):
        count = _exact_int(row[name], label=f"{label}.{name}")
        if count < 0:
            raise ValueError(f"{label}.{name} must be non-negative")
        counts[name] = count
    flags: dict[str, bool] = {}
    for name in _METRIC_FLAGS:
        if type(row[name]) is not bool:
            raise ValueError(f"{label}.{name} must be an exact bool")
        flags[name] = bool(row[name])
    tp, fp, fn, tn = (
        counts["tp"],
        counts["fp"],
        counts["fn"],
        counts["tn"],
    )
    iou_denominator = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    expected_iou = 1.0 if iou_denominator == 0 else tp / iou_denominator
    expected_dice = 1.0 if dice_denominator == 0 else (2 * tp) / dice_denominator
    _same_number(row["iou"], expected_iou, label=f"{label}.iou")
    _same_number(row["dice"], expected_dice, label=f"{label}.dice")
    background_invalid = fp + tn == 0
    foreground_invalid = tp + fn == 0
    if flags["invalid_background_denominator"] is not background_invalid:
        raise ValueError(f"{label} background invalid flag disagrees with counts")
    if flags["invalid_foreground_denominator"] is not foreground_invalid:
        raise ValueError(f"{label} foreground invalid flag disagrees with counts")
    for name, invalid, expected in (
        ("fpr", background_invalid, None if background_invalid else fp / (fp + tn)),
        ("recall", foreground_invalid, None if foreground_invalid else tp / (tp + fn)),
    ):
        current = row[name]
        if invalid:
            if current is not None:
                raise ValueError(f"{label}.{name} must be null when invalid")
        else:
            _same_number(current, expected, label=f"{label}.{name}")
    for name in ("boundary_f1_2px", "mscd_px", "p95_symmetric_distance_px"):
        _finite(row[name], label=f"{label}.{name}", nonnegative=True)
    return dict(row)


def _mean_optional(rows: Sequence[Mapping[str, object]], name: str) -> float | None:
    values = [row[name] for row in rows]
    if any(value is None for value in values):
        return None
    return float(sum(float(value) for value in values) / len(values))


def _parse_threshold_metric(value: object, *, label: str) -> dict[str, object]:
    report = _mapping(value, label=label)
    _exact_keys(report, {"per_view", "aggregate"}, label=label)
    raw_views = _sequence_of_mappings(report["per_view"], label=f"{label}.per_view")
    if len(raw_views) != 6:
        raise ValueError(f"{label} must contain six ordered views")
    views = [
        _parse_view_metric(item, label=f"{label}.per_view[{index}]")
        for index, item in enumerate(raw_views)
    ]
    aggregate = _mapping(report["aggregate"], label=f"{label}.aggregate")
    _exact_keys(aggregate, _AGGREGATE_METRIC_KEYS, label=f"{label}.aggregate")
    for name in ("tp", "fp", "fn", "tn"):
        total = sum(int(row[name]) for row in views)
        if _exact_int(aggregate[name], label=f"{label}.aggregate.{name}") != total:
            raise ValueError(f"{label} aggregate count disagrees with per-view rows")
    for source, destination in (
        ("iou", "mean_iou"),
        ("dice", "mean_dice"),
        ("boundary_f1_2px", "mean_boundary_f1_2px"),
        ("mscd_px", "mean_mscd_px"),
        ("p95_symmetric_distance_px", "mean_p95_symmetric_distance_px"),
    ):
        expected = float(sum(float(row[source]) for row in views) / 6.0)
        _same_number(
            aggregate[destination],
            expected,
            label=f"{label}.aggregate.{destination}",
        )
    for source, destination in (("fpr", "mean_fpr"), ("recall", "mean_recall")):
        expected_optional = _mean_optional(views, source)
        if expected_optional is None:
            if aggregate[destination] is not None:
                raise ValueError(f"{label}.aggregate.{destination} must be null")
        else:
            _same_number(
                aggregate[destination],
                expected_optional,
                label=f"{label}.aggregate.{destination}",
            )
    aggregate_flags: dict[str, bool] = {}
    for name in _METRIC_FLAGS:
        if type(aggregate[name]) is not bool:
            raise ValueError(f"{label}.aggregate.{name} must be an exact bool")
        expected_flag = any(bool(row[name]) for row in views)
        if aggregate[name] is not expected_flag:
            raise ValueError(f"{label}.aggregate.{name} disagrees with views")
        aggregate_flags[name] = expected_flag
    expected_task_invalid = any(
        aggregate_flags[name]
        for name in (
            "invalid_background_denominator",
            "invalid_foreground_denominator",
            "invalid_predicted_boundary",
            "invalid_target_boundary",
        )
    )
    if type(aggregate["invalid_task_metric"]) is not bool or (
        aggregate["invalid_task_metric"] is not expected_task_invalid
    ):
        raise ValueError(f"{label}.aggregate invalid-task flag disagrees")
    return {"per_view": views, "aggregate": dict(aggregate)}


def _parse_metrics_payload(
    value: object,
    *,
    spec: runner.RunSpec,
    config_hash: str,
    final_effective_loss: float,
    reject_invalid: bool,
) -> dict[str, object]:
    payload = _mapping(value, label="metrics payload")
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
        label="metrics payload",
    )
    if (
        payload["schema"] != "pcraft_soft_loss_v3_metrics_v1"
        or payload["config_hash"] != config_hash
        or payload["profile"] != spec.profile
        or type(payload["steps"]) is not int
        or payload["steps"] != spec.steps
        or payload["final_step_exported"] is not True
    ):
        raise ValueError("metrics identity differs from registration")
    _same_number(payload["final_loss"], final_effective_loss, label="metrics final_loss")
    thresholds = _mapping(payload["thresholds"], label="metrics thresholds")
    _exact_keys(thresholds, set(_THRESHOLDS), label="metrics thresholds")
    parsed = {
        threshold: _parse_threshold_metric(
            thresholds[threshold], label=f"metrics threshold {threshold}"
        )
        for threshold in _THRESHOLDS
    }
    task_iou = _mapping(payload["task_iou_by_threshold"], label="task IoU map")
    _exact_keys(task_iou, set(_THRESHOLDS), label="task IoU map")
    ious: list[float] = []
    for threshold in _THRESHOLDS:
        expected = parsed[threshold]["aggregate"]["mean_iou"]
        ious.append(
            _same_number(
                task_iou[threshold], expected, label=f"task IoU {threshold}"
            )
        )
    _same_number(payload["worst_threshold_iou"], min(ious), label="worst threshold IoU")
    _same_number(
        payload["threshold_iou_range"],
        max(ious) - min(ious),
        label="threshold IoU range",
    )
    primary = parsed["0.05"]["aggregate"]
    _same_number(payload["task_mscd_px"], primary["mean_mscd_px"], label="task MSCD")
    _same_number(
        payload["task_p95_symmetric_distance_px"],
        primary["mean_p95_symmetric_distance_px"],
        label="task contour p95",
    )
    expected_invalid = any(
        bool(parsed[threshold]["aggregate"]["invalid_task_metric"])
        for threshold in _THRESHOLDS
    )
    if type(payload["invalid_task_metric"]) is not bool or (
        payload["invalid_task_metric"] is not expected_invalid
    ):
        raise ValueError("top-level invalid-task flag disagrees with thresholds")
    any_flag = expected_invalid or any(
        bool(view[name])
        for threshold in _THRESHOLDS
        for view in parsed[threshold]["per_view"]
        for name in _METRIC_FLAGS
    ) or any(
        bool(parsed[threshold]["aggregate"][name])
        for threshold in _THRESHOLDS
        for name in (*_METRIC_FLAGS, "invalid_task_metric")
    )
    if reject_invalid and any_flag:
        raise _EvidenceFailure(
            "metrics_invalid_flag", "ordinary metrics contain an active invalid flag"
        )
    result = dict(payload)
    result["thresholds"] = parsed
    result["task_iou_by_threshold"] = dict(task_iou)
    return result


def _parse_history(
    raw: bytes, *, spec: runner.RunSpec
) -> list[dict[str, object]]:
    lines = raw.splitlines(keepends=True)
    if len(lines) != spec.steps or any(not line.endswith(b"\n") for line in lines):
        raise ValueError("history must contain one canonical row per update")
    rows: list[dict[str, object]] = []
    profile = resolve_profile(spec.profile)
    lambdas = spec.frozen_lambdas
    if not isinstance(lambdas, Mapping):
        raise ValueError("ordinary history requires frozen loss weights")
    lambda_dic = _finite(lambdas.get("lambda_dic"), label="lambda_dic", nonnegative=True)
    lambda_dts = _finite(lambdas.get("lambda_dts"), label="lambda_dts", nonnegative=True)
    for index, line in enumerate(lines, start=1):
        payload = registry._decode_json(line[:-1], label=f"history row {index}")
        if line != canonical_json_bytes(payload) + b"\n":
            raise ValueError("history row is not canonical JSONL")
        normalized = task8_trainer._history_record(payload)
        if (
            normalized["global_step_before"] != index - 1
            or normalized["completed_updates_after"] != index
            or not math.isclose(
                float(normalized["progress"]),
                index / float(spec.steps),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("history update identity or progress is wrong")
        progress = index / float(spec.steps)
        expected_dic = lambda_dic * dic_schedule(progress) if profile.use_dic else 0.0
        expected_dts = lambda_dts * dts_schedule(progress) if profile.use_dts else 0.0
        _same_number(normalized["dic_scale"], expected_dic, label="history DIC scale")
        _same_number(normalized["dts_scale"], expected_dts, label="history DTS scale")
        if (normalized["loss_dic"] is not None) is (expected_dic <= 0.0):
            raise ValueError("history DIC activation differs from schedule")
        if expected_dts <= 0.0:
            if normalized["loss_dts"] is not None or normalized["dts_endpoint"] is not None:
                raise ValueError("history DTS is active outside its schedule")
        else:
            if normalized["loss_dts"] is None or profile.dts_thresholds is None:
                raise ValueError("history DTS is inactive inside its schedule")
            endpoint = float(profile.dts_thresholds[(index - 1) % 2])
            _same_number(normalized["dts_endpoint"], endpoint, label="history DTS endpoint")
        rows.append(normalized)
    if not math.isclose(float(rows[-1]["progress"]), 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("final history progress must equal one")
    return rows


def _expected_probe_updates(spec: runner.RunSpec) -> list[int]:
    profile = resolve_profile(spec.profile)
    return [
        update
        for update in range(1, spec.steps + 1)
        if task8_trainer._should_probe(
            update, spec.steps, profile, calibration=False
        )
    ]


def _parse_gradient_audit(
    value: object, *, spec: runner.RunSpec, config_hash: str
) -> dict[str, object]:
    payload = _mapping(value, label="gradient audit")
    _exact_keys(
        payload,
        {
            "schema",
            "config_hash",
            "samples",
            "calibration_records",
            "active_invalid_flags",
        },
        label="gradient audit",
    )
    if (
        payload["schema"] != "pcraft_soft_loss_v3_gradient_audit_v1"
        or payload["config_hash"] != config_hash
        or payload["calibration_records"] != []
    ):
        raise ValueError("gradient audit identity/calibration shape is invalid")
    raw_samples = _sequence_of_mappings(payload["samples"], label="gradient samples")
    normalized_samples: list[dict[str, object]] = []
    objects: list[ComponentGradientSample] = []
    for index, raw_sample in enumerate(raw_samples):
        normalized = task8_trainer._component_sample_from_mapping(raw_sample)
        sample = ComponentGradientSample(**normalized)
        normalized_samples.append(dict(raw_sample))
        objects.append(sample)
    expected_updates = _expected_probe_updates(spec)
    if [sample.completed_updates for sample in objects] != expected_updates:
        raise ValueError("gradient probes differ from the registered schedule")
    raw_flags = _sequence_of_mappings(
        payload["active_invalid_flags"], label="active invalid flags"
    )
    if len(raw_flags) != len(objects):
        raise ValueError("gradient active-invalid table has wrong length")
    profile = resolve_profile(spec.profile)
    normalized_flags: list[dict[str, object]] = []
    for index, (record, sample) in enumerate(zip(raw_flags, objects)):
        _exact_keys(
            record,
            {"completed_updates", "flags"},
            label=f"active invalid flags[{index}]",
        )
        flags = record["flags"]
        if (
            type(record["completed_updates"]) is not int
            or record["completed_updates"] != sample.completed_updates
            or not isinstance(flags, list)
            or any(type(flag) is not str for flag in flags)
            or flags != task8_trainer._active_diagnostic_invalid_flags(sample, profile)
        ):
            raise ValueError("gradient active-invalid flags disagree with samples")
        if flags:
            raise ValueError("ordinary gradient audit has an active invalid flag")
        normalized_flags.append(dict(record))
    result = dict(payload)
    result["samples"] = normalized_samples
    result["active_invalid_flags"] = normalized_flags
    return result


def _parse_resource_payload(
    value: object, *, spec: runner.RunSpec
) -> dict[str, object]:
    payload = _mapping(value, label="resource metrics")
    _validate_resource_payload(payload, expected_steps=spec.steps)
    physical = payload["physical_device_memory_bytes"]
    if spec.device == "cuda:0" and (type(physical) is not int or physical <= 0):
        raise ValueError("registered CUDA run requires positive physical memory")
    return dict(payload)


def _artifact_snapshots(
    run_dir: Path,
    *,
    spec: runner.RunSpec,
    artifacts: Mapping[str, object],
) -> dict[str, object]:
    namespace = runner._expected_artifact_namespace(spec)
    expected_top = set(namespace["top_level"])
    allowed_root = expected_top | {
        "STATUS.json",
        "artifact_manifest.json",
        "transactions",
    }
    observed_root = {entry.name for entry in run_dir.iterdir()}
    if observed_root != allowed_root:
        raise _EvidenceFailure(
            "artifact_invalid", "run directory has an extra or missing artifact"
        )
    transactions = run_dir / "transactions"
    if transactions.is_symlink() or not transactions.is_dir():
        raise _EvidenceFailure("artifact_invalid", "transaction namespace is invalid")
    expected_artifacts = set(expected_top)
    transaction_files = set(namespace["transaction_files"])
    generations = sorted(transactions.iterdir(), key=lambda item: item.name)
    if not generations:
        raise _EvidenceFailure("artifact_invalid", "transaction namespace is empty")
    for generation in generations:
        if (
            generation.is_symlink()
            or not generation.is_dir()
            or len(generation.name) != 13
            or not generation.name.startswith("step_")
            or not generation.name[5:].isdigit()
        ):
            raise _EvidenceFailure("artifact_invalid", "transaction generation is invalid")
        names = {entry.name for entry in generation.iterdir()}
        if names != transaction_files or any(entry.is_symlink() for entry in generation.iterdir()):
            raise _EvidenceFailure(
                "artifact_invalid", "transaction generation has a wrong file set"
            )
        expected_artifacts.update(
            f"transactions/{generation.name}/{name}" for name in transaction_files
        )
    if set(artifacts) != expected_artifacts:
        raise _EvidenceFailure(
            "artifact_invalid", "artifact manifest namespace is not exact"
        )
    snapshots: dict[str, object] = {}
    for relative in sorted(expected_artifacts):
        digest = artifacts[relative]
        if type(digest) is not str or len(digest) != 64:
            raise _EvidenceFailure("artifact_invalid", "artifact digest is malformed")
        snapshot = registry._snapshot_file(
            run_dir / relative,
            label=f"registered artifact {relative}",
            root=run_dir,
        )
        if snapshot.sha256 != digest:
            raise _EvidenceFailure("artifact_invalid", "artifact hash is stale")
        snapshots[relative] = snapshot
    return snapshots


def _parameter_change(snapshot: object, *, resolution: int) -> dict[str, object]:
    try:
        array = np.load(BytesIO(getattr(snapshot, "data")), allow_pickle=False)
    except (OSError, ValueError, TypeError) as error:
        raise _EvidenceFailure(
            "artifact_invalid", "occupancy probability is not a safe NPY array"
        ) from error
    if (
        array.dtype != np.dtype("float32")
        or array.dtype.byteorder not in {"=", "|"}
        or array.shape != (resolution, resolution, resolution)
        or not np.isfinite(array).all()
    ):
        raise _EvidenceFailure(
            "artifact_invalid", "occupancy probability has wrong dtype/shape/value"
        )
    reference = np.float32(
        torch.sigmoid(torch.tensor(-4.0, dtype=torch.float32)).item()
    )
    reference_array = np.full(array.shape, reference, dtype=np.float32)
    delta = np.empty(array.shape, dtype=np.float32)
    np.subtract(array, reference_array, out=delta, dtype=np.float32)
    np.abs(delta, out=delta)
    changed = not np.array_equal(
        array.view(np.uint32), reference_array.view(np.uint32)
    )
    return {
        "reference_probability": float(reference),
        "max_abs_delta": float(delta.max()),
        "changed": bool(changed),
    }


def _registered_context(
    root: Path, stage: str
) -> tuple[
    Mapping[str, object],
    list[runner.RunSpec],
    dict[str, object],
    dict[str, object],
    object,
]:
    context = _stage_context(root, stage)
    identities = runner._validated_stage_context(stage, context)
    specs = runner.build_stage_specs(stage, context)
    runner._validate_stage_spec_sequence(stage, specs, output_root=root)
    runner._verify_spec_dependencies(specs)
    stage_path = root / "manifests" / "stages" / f"{stage}.json"
    stage_payload, stage_snapshot = _snapshot_json(
        stage_path, label=f"{stage} source stage", root=root
    )
    semantic = stage_payload.get("semantic_sha256")
    stage_identity = _identity_record(stage_path, stage_snapshot, semantic)
    weight = identities.get("weights")
    weight_identity = None
    if isinstance(weight, Mapping):
        weight_identity = {
            name: weight[name]
            for name in ("path", "sha256", "semantic_sha256")
        }
    validated_identity, validated_payload = runner._validate_source_stage_identity(
        stage_identity,
        output_root=root,
        source_stage=stage,
        master_identity=_mapping(identities.get("master"), label="master identity"),
        development_identity=_mapping(
            identities.get("development"), label="development identity"
        ),
        formal_identity=_mapping(identities.get("formal"), label="formal identity"),
        specification_identity=_mapping(
            identities.get("specification"), label="specification identity"
        ),
        weight_identity=weight_identity,
    )
    if validated_identity != stage_identity or validated_payload != stage_payload:
        raise ValueError("source-stage validation returned a different identity")
    return identities, specs, stage_identity, stage_payload, stage_snapshot


def collect_stage_rows(root: Path, stage: str) -> list[dict[str, object]]:
    output_root = _canonical_root(root)
    if type(stage) is not str or stage not in _SOURCE_STAGES:
        raise ValueError("stage is not a registered Task 12 source")
    identities, specs, stage_identity, stage_payload, stage_snapshot = (
        _registered_context(output_root, stage)
    )
    if len(specs) != _COUNTS[stage]:
        raise ValueError("registered stage has the wrong exact matrix count")
    registrations = stage_payload.get("runs")
    if not isinstance(registrations, list) or len(registrations) != len(specs):
        raise ValueError("source stage registration is malformed")
    campaign_identity = None
    campaign_snapshot = None
    if stage in {"formal256_primary", "formal256_repeat"}:
        campaign_identity, _, campaign_snapshot = runner._load_formal_campaign_identity(
            output_root / "manifests" / "formal_campaign.json", output_root
        )
    reject_invalid = stage not in {
        "formal256_primary",
        "formal256_repeat",
        "appendix256",
    }
    rows: list[dict[str, object]] = []
    all_snapshots: list[tuple[object, str, Path]] = []
    for index, (spec, registration) in enumerate(zip(specs, registrations)):
        if not isinstance(registration, Mapping):
            raise ValueError("source-stage run registration must be a mapping")
        try:
            completed = runner.is_completed_run(spec.output_path, spec)
        except FileNotFoundError as error:
            raise _EvidenceFailure(
                "run_incomplete", "registered run is missing", run_id=spec.run_id
            ) from error
        except ValueError as error:
            raise _EvidenceFailure(
                "artifact_invalid", "registered completed run is invalid", run_id=spec.run_id
            ) from error
        if not completed:
            raise _EvidenceFailure(
                "run_incomplete", "registered run is incomplete", run_id=spec.run_id
            )
        run_dir = spec.output_path
        status_payload, status_snapshot = _snapshot_json(
            run_dir / "STATUS.json", label=f"run {index} STATUS", root=run_dir
        )
        _exact_keys(
            status_payload,
            {
                "schema",
                "state",
                "config_hash",
                "completed_updates",
                "artifact_manifest_sha256",
            },
            label=f"run {index} STATUS",
        )
        expected_config = registration.get("expected_config_hash")
        if (
            status_payload["schema"] != "pcraft_soft_loss_v3_status_v1"
            or status_payload["state"] != "completed"
            or status_payload["config_hash"] != expected_config
            or type(status_payload["completed_updates"]) is not int
            or status_payload["completed_updates"] != spec.steps
        ):
            raise _EvidenceFailure(
                "run_incomplete", "STATUS is not the registered final update", run_id=spec.run_id
            )
        artifact_payload, artifact_snapshot = _snapshot_json(
            run_dir / "artifact_manifest.json",
            label=f"run {index} artifact manifest",
            root=run_dir,
        )
        _exact_keys(
            artifact_payload,
            {"schema", "config_hash", "artifacts_sha256"},
            label=f"run {index} artifact manifest",
        )
        if (
            artifact_payload["schema"] != "pcraft_soft_loss_v3_artifacts_v1"
            or artifact_payload["config_hash"] != expected_config
            or status_payload["artifact_manifest_sha256"] != artifact_snapshot.sha256
        ):
            raise _EvidenceFailure(
                "artifact_invalid", "artifact manifest identity is invalid", run_id=spec.run_id
            )
        artifacts = _mapping(
            artifact_payload["artifacts_sha256"], label="artifact SHA map"
        )
        snapshots = _artifact_snapshots(run_dir, spec=spec, artifacts=artifacts)
        history_snapshot = snapshots["history.jsonl"]
        try:
            history_rows = _parse_history(history_snapshot.data, spec=spec)
        except (TypeError, ValueError) as error:
            raise _EvidenceFailure(
                "history_invalid", "history content is invalid", run_id=spec.run_id
            ) from error
        metrics_snapshot = snapshots["metrics.json"]
        try:
            metrics_raw = registry._decode_json(
                metrics_snapshot.data, label=f"run {index} metrics"
            )
            if metrics_snapshot.data != canonical_json_bytes(metrics_raw):
                raise ValueError("metrics are not canonical JSON")
            metrics = _parse_metrics_payload(
                metrics_raw,
                spec=spec,
                config_hash=str(expected_config),
                final_effective_loss=float(history_rows[-1]["effective_loss"]),
                reject_invalid=reject_invalid,
            )
        except _EvidenceFailure as error:
            error.run_id = spec.run_id
            raise
        except (TypeError, ValueError) as error:
            raise _EvidenceFailure(
                "metrics_schema_invalid", "metrics content is invalid", run_id=spec.run_id
            ) from error
        gradient_snapshot = snapshots["gradient_audit.json"]
        try:
            gradient_raw = registry._decode_json(
                gradient_snapshot.data, label=f"run {index} gradient audit"
            )
            if gradient_snapshot.data != canonical_json_bytes(gradient_raw):
                raise ValueError("gradient audit is not canonical JSON")
            gradient = _parse_gradient_audit(
                gradient_raw, spec=spec, config_hash=str(expected_config)
            )
        except (TypeError, ValueError) as error:
            raise _EvidenceFailure(
                "gradient_audit_invalid", "gradient audit is invalid", run_id=spec.run_id
            ) from error
        resource_snapshot = snapshots["resource_metrics.json"]
        try:
            resource_raw = registry._decode_json(
                resource_snapshot.data, label=f"run {index} resource metrics"
            )
            if resource_snapshot.data != canonical_json_bytes(resource_raw):
                raise ValueError("resource metrics are not canonical JSON")
            resource = _parse_resource_payload(resource_raw, spec=spec)
        except (TypeError, ValueError) as error:
            raise _EvidenceFailure(
                "resource_schema_invalid", "resource metrics are invalid", run_id=spec.run_id
            ) from error
        parameter = _parameter_change(
            snapshots["occupancy_probability.npy"], resolution=spec.resolution
        )
        row = {
            "schema": "pcraft_soft_loss_v3_collected_row_v1",
            "stage": stage,
            "run_id": spec.run_id,
            "path": str(spec.output_path),
            "task": spec.task,
            "variant": spec.variant,
            "replicate_role": spec.replicate_role,
            "replicate_index": spec.replicate_index,
            "inference_eligible": spec.inference_eligible,
            "resolution": spec.resolution,
            "steps": spec.steps,
            "scientific_config_hash": registration["scientific_config_hash"],
            "expected_config_hash": expected_config,
            "run_spec": runner._run_spec_payload(spec),
            "stage_manifest": stage_identity,
            "formal_campaign": campaign_identity,
            "status": {
                "path": str(status_snapshot.path),
                "sha256": status_snapshot.sha256,
                "payload": status_payload,
            },
            "artifact_manifest": {
                "path": str(artifact_snapshot.path),
                "sha256": artifact_snapshot.sha256,
                "payload": artifact_payload,
            },
            "metrics": {
                "path": str(metrics_snapshot.path),
                "sha256": metrics_snapshot.sha256,
                "payload": metrics,
            },
            "gradient_audit": {
                "path": str(gradient_snapshot.path),
                "sha256": gradient_snapshot.sha256,
                "payload": gradient,
            },
            "resource_metrics": {
                "path": str(resource_snapshot.path),
                "sha256": resource_snapshot.sha256,
                "payload": resource,
            },
            "history": {
                "path": str(history_snapshot.path),
                "sha256": history_snapshot.sha256,
                "rows": history_rows,
            },
            "parameter_change": parameter,
        }
        _exact_keys(row, _ROW_KEYS, label="collected row")
        rows.append(row)
        all_snapshots.extend(
            (snapshot, f"run {index} artifact", run_dir)
            for snapshot in snapshots.values()
        )
        all_snapshots.extend(
            (
                (status_snapshot, f"run {index} STATUS", run_dir),
                (artifact_snapshot, f"run {index} artifact manifest", run_dir),
            )
        )
    registry._verify_snapshot_unchanged(
        stage_snapshot, label=f"{stage} source stage", root=output_root
    )
    if campaign_snapshot is not None:
        registry._verify_snapshot_unchanged(
            campaign_snapshot, label="formal campaign", root=output_root
        )
    for snapshot, label, snapshot_root in all_snapshots:
        registry._verify_snapshot_unchanged(
            snapshot, label=label, root=snapshot_root
        )
    refreshed = runner._validated_stage_context(stage, _stage_context(output_root, stage))
    if canonical_json_bytes({"identities": identities}) != canonical_json_bytes(
        {"identities": refreshed}
    ):
        raise ValueError("stage dependencies changed during collection")
    current_identity, current_payload = runner._validate_source_stage_identity(
        stage_identity,
        output_root=output_root,
        source_stage=stage,
    )
    if current_identity != stage_identity or current_payload != stage_payload:
        raise ValueError("source stage changed during collection")
    return rows


def build_summary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    parser.add_argument("--stage")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stage-manifest", type=Path)
    parser.add_argument("--master-manifest", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--development-suite", type=Path)
    parser.add_argument("--formal-suite", type=Path)
    parser.add_argument("--next-stage")
    parser.add_argument("--integration-stage-manifest", type=Path)
    parser.add_argument("--reference-stage-manifest", type=Path)
    parser.add_argument("--primary-stage-manifest", type=Path)
    parser.add_argument("--formal-campaign", type=Path)
    parser.add_argument("--completion-only", action="store_true")
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--verify-existing-analysis", type=Path)
    parser.add_argument("--exploratory-only", action="store_true")
    return parser


def development_factorial_gate(
    rows: Sequence[Mapping[str, object]],
    *,
    dic_relative_threshold: float,
    dts_absolute_threshold: float,
    require_nonincreasing_range: bool,
) -> dict[str, object]:
    dic_threshold = _public_nonnegative(dic_relative_threshold, label="DIC threshold")
    dts_threshold = _public_nonnegative(dts_absolute_threshold, label="DTS threshold")
    if type(require_nonincreasing_range) is not bool:
        raise ValueError("require_nonincreasing_range must be an exact bool")
    criteria = {
        "task_count": 3,
        "dic_relative_threshold": dic_threshold,
        "dts_absolute_threshold": dts_threshold,
        "iou_noninferiority_delta": -0.001,
        "recall_floor_delta": -0.005,
        "fpr_ceiling_delta": 0.001,
        "require_nonincreasing_range": require_nonincreasing_range,
    }
    empty = {
        "dic_task_equal_effect": None,
        "dic_conditional_task_equal_effects": {},
        "dts_task_equal_effect": None,
        "dts_conditional_task_equal_effects": {},
        "mean_iou_delta_tau_0.05": None,
        "min_task_recall_delta": None,
        "mean_fpr_delta_tau_0.05": None,
        "mean_threshold_range_delta": None,
        "per_task": {},
    }
    try:
        values = _sequence_of_mappings(rows, label="development rows")
    except ValueError:
        return _gate_result("development_factorial", criteria=criteria, observed=empty, failures=["input_schema_invalid"])
    expected = [(task, variant) for task in _DEVELOPMENT for variant in _PRIMARY]
    observed_matrix = [(row.get("task"), row.get("variant")) for row in values]
    if len(values) != 12 or observed_matrix != expected or len(set(observed_matrix)) != 12:
        return _gate_result("development_factorial", criteria=criteria, observed=empty, failures=["factorial_matrix_invalid"])
    per_task: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    try:
        for task in _DEVELOPMENT:
            group = {row["variant"]: row for row in values if row["task"] == task}
            metrics: dict[str, tuple[Mapping[str, object], Mapping[str, object]]] = {
                variant: _primary_metric(group[variant]) for variant in _PRIMARY
            }
            if any(
                payload.get("invalid_task_metric") is not False
                or any(
                    aggregate.get(name) is not False
                    for name in (
                        "invalid_iou_denominator",
                        "invalid_dice_denominator",
                        "invalid_background_denominator",
                        "invalid_foreground_denominator",
                        "invalid_predicted_boundary",
                        "invalid_target_boundary",
                        "invalid_task_metric",
                    )
                )
                for payload, aggregate in metrics.values()
            ):
                failures.append("metric_invalid")
            mscd = {variant: _finite(metrics[variant][0].get("task_mscd_px"), label="task MSCD") for variant in _PRIMARY}
            if mscd["l12_bal"] <= 0.0 or mscd["dts_only"] <= 0.0:
                failures.append("mscd_denominator_invalid")
                raise ArithmeticError
            worst = {variant: _finite(metrics[variant][0].get("worst_threshold_iou"), label="worst IoU") for variant in _PRIMARY}
            iou = {variant: _finite(metrics[variant][1].get("mean_iou"), label="mean IoU") for variant in _PRIMARY}
            recall = {variant: _finite(metrics[variant][1].get("mean_recall"), label="mean recall") for variant in _PRIMARY}
            fpr = {variant: _finite(metrics[variant][1].get("mean_fpr"), label="mean FPR") for variant in _PRIMARY}
            ranges = {variant: _finite(metrics[variant][0].get("threshold_iou_range"), label="threshold range", nonnegative=True) for variant in _PRIMARY}
            per_task[task] = {
                "dic_base_to_dic_relative": (mscd["l12_bal"] - mscd["dic_only"]) / mscd["l12_bal"],
                "dic_dts_to_full_relative": (mscd["dts_only"] - mscd["v3_full"]) / mscd["dts_only"],
                "dts_base_to_dts_absolute": worst["dts_only"] - worst["l12_bal"],
                "dts_dic_to_full_absolute": worst["v3_full"] - worst["dic_only"],
                "iou_delta_tau_0.05": iou["v3_full"] - iou["l12_bal"],
                "recall_delta_tau_0.05": recall["v3_full"] - recall["l12_bal"],
                "fpr_delta_tau_0.05": fpr["v3_full"] - fpr["l12_bal"],
                "threshold_range_delta": ranges["v3_full"] - ranges["l12_bal"],
            }
    except ArithmeticError:
        return _gate_result("development_factorial", criteria=criteria, observed=empty, failures=failures)
    except (KeyError, TypeError, ValueError):
        return _gate_result("development_factorial", criteria=criteria, observed=empty, failures=["input_schema_invalid"])

    def task_mean(name: str) -> float:
        return float(sum(per_task[task][name] for task in _DEVELOPMENT) / 3.0)

    dic_conditionals = {
        "base_to_dic": task_mean("dic_base_to_dic_relative"),
        "dts_to_full": task_mean("dic_dts_to_full_relative"),
    }
    dts_conditionals = {
        "base_to_dts": task_mean("dts_base_to_dts_absolute"),
        "dic_to_full": task_mean("dts_dic_to_full_absolute"),
    }
    observed = {
        "dic_task_equal_effect": 0.5 * sum(dic_conditionals.values()),
        "dic_conditional_task_equal_effects": dic_conditionals,
        "dts_task_equal_effect": 0.5 * sum(dts_conditionals.values()),
        "dts_conditional_task_equal_effects": dts_conditionals,
        "mean_iou_delta_tau_0.05": task_mean("iou_delta_tau_0.05"),
        "min_task_recall_delta": min(per_task[task]["recall_delta_tau_0.05"] for task in _DEVELOPMENT),
        "mean_fpr_delta_tau_0.05": task_mean("fpr_delta_tau_0.05"),
        "mean_threshold_range_delta": task_mean("threshold_range_delta"),
        "per_task": per_task,
    }
    if any(value < 0.0 for value in dic_conditionals.values()):
        failures.append("dic_conditional_reversed")
    if any(value < 0.0 for value in dts_conditionals.values()):
        failures.append("dts_conditional_reversed")
    if observed["dic_task_equal_effect"] < dic_threshold:
        failures.append("dic_threshold_missed")
    if observed["dts_task_equal_effect"] < dts_threshold:
        failures.append("dts_threshold_missed")
    if observed["mean_iou_delta_tau_0.05"] < -0.001:
        failures.append("iou_noninferiority_failed")
    if observed["min_task_recall_delta"] < -0.005:
        failures.append("recall_noninferiority_failed")
    if observed["mean_fpr_delta_tau_0.05"] > 0.001:
        failures.append("fpr_guard_failed")
    if require_nonincreasing_range and observed["mean_threshold_range_delta"] > 0.0:
        failures.append("threshold_range_failed")
    return _gate_result("development_factorial", criteria=criteria, observed=observed, failures=failures)


def _expected_gradient_matrix(resolution: int) -> list[tuple[str, int, tuple[str, ...], float | None]]:
    steps = 200 if resolution == 128 else 2000
    dic_start = 150 if resolution == 128 else 1300
    dts_start = 100 if resolution == 128 else 1000
    expected: list[tuple[str, int, tuple[str, ...], float | None]] = []
    for variant in ("dic_only", "dts_only", "v3_full"):
        start = dic_start if variant == "dic_only" else dts_start
        for update in range(start, steps + 1, 50):
            dic_active = variant in {"dic_only", "v3_full"} and update >= dic_start
            dts_active = variant in {"dts_only", "v3_full"}
            endpoint = (0.06 if update == steps else 0.04) if dts_active else None
            active = (() if not dic_active else ("dic",)) + (() if not dts_active else (f"dts_{endpoint:.2f}",))
            expected.append((variant, update, active, endpoint))
    return expected


def gradient_budget_gate(
    samples: Sequence[Mapping[str, object]], *, resolution: int
) -> dict[str, object]:
    if type(resolution) is not int or resolution not in {128, 256}:
        raise ValueError("resolution must be exact int 128 or 256")
    criteria = {
        "resolution": resolution,
        "ratio_min": 0.10,
        "ratio_max": 0.20,
        "combined_median_max": 0.35,
        "combined_p95_max": 1.0,
        "adverse_cosine_threshold": -0.5,
        "adverse_cosine_fraction_max": 0.10,
        "component_active_l2_epsilon": 1e-12,
        "support_abs_epsilon": 1e-12,
        "eligible_inactive_policy": "stop_any_eligible_inactive_sample",
    }
    expected = _expected_gradient_matrix(resolution)
    empty = {
        "expected_probe_count": len(expected),
        "observed_probe_count": 0,
        "dic_ratio_median": None,
        "dts_ratio_median": None,
        "combined_ratio_median": None,
        "combined_ratio_p95": None,
        "adverse_cosine_fraction": None,
        "endpoint_diagnostics": {},
    }
    try:
        values = _sequence_of_mappings(samples, label="gradient samples")
    except ValueError:
        return _gate_result("gradient_budget", criteria=criteria, observed=empty, failures=["input_schema_invalid"])
    expected_identity = [(variant, update, active) for variant, update, active, _ in expected]
    try:
        observed_identity = []
        for item in values:
            _exact_keys(item, {"schema", "stage", "run_id", "task", "variant", "resolution", "steps", "profile", "completed_updates", "active_components", "sample"}, label="gradient envelope")
            if item["schema"] != "pcraft_soft_loss_v3_gradient_sample_v1" or item["resolution"] != resolution:
                raise ValueError("gradient envelope identity mismatch")
            active = item["active_components"]
            if not isinstance(active, list) or any(type(token) is not str for token in active):
                raise ValueError("active_components must be a string list")
            vocabulary = ["dic", "dts_0.04", "dts_0.06"]
            positions = [vocabulary.index(token) if token in vocabulary else -1 for token in active]
            if -1 in positions or positions != sorted(set(positions)):
                raise ValueError("active_components is not a canonical subsequence")
            observed_identity.append((item["variant"], item["completed_updates"], tuple(active)))
    except (TypeError, ValueError):
        return _gate_result("gradient_budget", criteria=criteria, observed=empty, failures=["input_schema_invalid"])
    if observed_identity != expected_identity:
        empty["observed_probe_count"] = len(values)
        return _gate_result("gradient_budget", criteria=criteria, observed=empty, failures=["probe_matrix_invalid"])

    dic_ratios: list[float] = []
    dts_ratios: list[float] = []
    combined: list[float] = []
    combined_cosines: list[float] = []
    endpoint_values: dict[str, dict[str, list[float]]] = {
        "0.04": {"cosines": [], "supports": []},
        "0.06": {"cosines": [], "supports": []},
    }
    failures: list[str] = []
    try:
        for item, (_, update, expected_active, endpoint) in zip(values, expected):
            sample = _mapping(item["sample"], label="gradient sample payload")
            if set(sample) != {field.name for field in fields(ComponentGradientSample)} or sample.get("completed_updates") != update:
                raise ValueError("gradient sample schema/update mismatch")
            active = tuple(item["active_components"])
            if "dic" in expected_active:
                raw = _finite(sample.get("dic_norm"), label="DIC norm", nonnegative=True)
                if raw <= 1e-12:
                    failures.append("eligible_component_inactive")
                if sample.get("weighted_dic_base_ratio_invalid") is not False:
                    raise ValueError("DIC weighted ratio is invalid")
                dic_ratios.append(_finite(sample.get("weighted_dic_base_ratio"), label="DIC ratio", nonnegative=True))
            if endpoint is not None:
                key = f"{endpoint:.2f}"
                norms = _mapping(sample.get("dts_norms"), label="DTS norms")
                ratios = _mapping(sample.get("weighted_dts_base_ratios"), label="DTS ratios")
                invalid = _mapping(sample.get("weighted_dts_base_ratio_invalid"), label="DTS ratio invalid flags")
                raw = _finite(norms.get(key), label="selected DTS norm", nonnegative=True)
                if raw <= 1e-12:
                    failures.append("eligible_component_inactive")
                if invalid.get(key) is not False:
                    raise ValueError("selected DTS weighted ratio is invalid")
                dts_ratios.append(_finite(ratios.get(key), label="selected DTS ratio", nonnegative=True))
            if sample.get("combined_aux_base_ratio_invalid") is not False or sample.get("base_combined_aux_cosine_invalid") is not False:
                raise ValueError("combined gradient diagnostic is invalid")
            combined.append(_finite(sample.get("combined_aux_base_ratio"), label="combined ratio", nonnegative=True))
            combined_cosines.append(_finite(sample.get("base_combined_aux_cosine"), label="combined cosine"))
            if item["variant"] == "v3_full" and "dic" in active and endpoint is not None:
                key = f"{endpoint:.2f}"
                cosines = _mapping(sample.get("dic_dts_cosines"), label="DIC/DTS cosines")
                cosine_invalid = _mapping(sample.get("dic_dts_cosine_invalid"), label="DIC/DTS cosine flags")
                supports = _mapping(sample.get("dic_dts_support_ious"), label="DIC/DTS supports")
                support_invalid = _mapping(sample.get("dic_dts_support_iou_invalid"), label="DIC/DTS support flags")
                if cosine_invalid.get(key) is not False or support_invalid.get(key) is not False:
                    failures.append("endpoint_diagnostic_invalid")
                    continue
                endpoint_values[key]["cosines"].append(_finite(cosines.get(key), label="endpoint cosine"))
                endpoint_values[key]["supports"].append(_finite(supports.get(key), label="endpoint support", nonnegative=True))
    except (KeyError, TypeError, ValueError):
        return _gate_result("gradient_budget", criteria=criteria, observed=empty, failures=["input_schema_invalid"])

    joint_counts = {"0.04": 1 if resolution == 128 else 14, "0.06": 1}
    if any(len(endpoint_values[key]["cosines"]) != count or len(endpoint_values[key]["supports"]) != count for key, count in joint_counts.items()):
        failures.append("endpoint_diagnostic_invalid")
    endpoint_report = {
        key: {
            "joint_sample_count": len(endpoint_values[key]["cosines"]),
            "dic_dts_cosine_median": float(statistics.median(endpoint_values[key]["cosines"])) if endpoint_values[key]["cosines"] else 0.0,
            "dic_dts_support_iou_median": float(statistics.median(endpoint_values[key]["supports"])) if endpoint_values[key]["supports"] else 0.0,
        }
        for key in _THRESHOLDS if key != "0.05"
    }
    observed = {
        "expected_probe_count": len(expected),
        "observed_probe_count": len(values),
        "dic_ratio_median": float(statistics.median(dic_ratios)),
        "dts_ratio_median": float(statistics.median(dts_ratios)),
        "combined_ratio_median": float(statistics.median(combined)),
        "combined_ratio_p95": _canonical_percentile(combined, 0.95),
        "adverse_cosine_fraction": float(sum(value < -0.5 for value in combined_cosines) / len(combined_cosines)),
        "endpoint_diagnostics": endpoint_report,
    }
    if not 0.10 <= observed["dic_ratio_median"] <= 0.20 or not 0.10 <= observed["dts_ratio_median"] <= 0.20:
        failures.append("component_ratio_out_of_range")
    if observed["combined_ratio_median"] > 0.35:
        failures.append("combined_ratio_median_exceeded")
    if observed["combined_ratio_p95"] > 1.0:
        failures.append("combined_ratio_p95_exceeded")
    if observed["adverse_cosine_fraction"] > 0.10:
        failures.append("adverse_cosine_fraction_exceeded")
    return _gate_result("gradient_budget", criteria=criteria, observed=observed, failures=failures)


def _cuda_integration_gate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    criteria = {"run_count": 4, "updates_per_run": 2, "final_progress": 1.0}
    empty = {"branch_checks": {}}
    try:
        values = _sequence_of_mappings(rows, label="integration rows")
    except ValueError:
        return _gate_result("cuda_integration", criteria=criteria, observed=empty, failures=["input_schema_invalid"])
    if [(row.get("task"), row.get("variant")) for row in values] != [("arcane_casters", variant) for variant in _PRIMARY]:
        return _gate_result("cuda_integration", criteria=criteria, observed=empty, failures=["integration_matrix_invalid"])
    checks: dict[str, object] = {}
    failures: list[str] = []
    try:
        for row in values:
            variant = str(row["variant"])
            history = _mapping(row.get("history"), label="integration history")
            rows_value = _sequence_of_mappings(history.get("rows"), label="integration history rows")
            if len(rows_value) != 2:
                failures.append("integration_matrix_invalid")
                raise ArithmeticError
            final = rows_value[-1]
            expected_dic = variant in {"dic_only", "v3_full"}
            expected_dts = variant in {"dts_only", "v3_full"}
            observed_dic = final.get("loss_dic") is not None and _finite(final.get("dic_scale"), label="DIC scale") > 0.0
            observed_dts = final.get("loss_dts") is not None and _finite(final.get("dts_scale"), label="DTS scale") > 0.0
            gradient_values: list[float] = []
            for history_row in rows_value:
                for name in ("pre_clip_grad_norm", "post_clip_grad_norm"):
                    raw_gradient = history_row.get(name)
                    if isinstance(raw_gradient, bool) or not isinstance(
                        raw_gradient, Real
                    ):
                        raise ValueError("gradient norm must be numeric")
                    gradient_values.append(float(raw_gradient))
            finite_gradients = all(
                math.isfinite(value) for value in gradient_values
            )
            run_spec = _mapping(row.get("run_spec"), label="integration RunSpec")
            soft_only = (
                row.get("schema") == "pcraft_soft_loss_v3_collected_row_v1"
                and run_spec.get("soft_only") is True
            )
            checks[variant] = {
                "completed_updates": int(final.get("completed_updates_after")),
                "final_progress": _finite(final.get("progress"), label="final progress"),
                "expected_dic_active": expected_dic,
                "expected_dts_active": expected_dts,
                "observed_dic_active": observed_dic,
                "observed_dts_active": observed_dts,
                "finite_gradients": finite_gradients,
                "soft_only": soft_only,
            }
            if final.get("completed_updates_after") != 2 or final.get("progress") != 1.0:
                failures.append("integration_matrix_invalid")
            if observed_dic is not expected_dic or observed_dts is not expected_dts:
                failures.append("integration_branch_inactive")
            if not finite_gradients:
                failures.append("integration_nonfinite")
            if not soft_only:
                failures.append("soft_only_violation")
    except ArithmeticError:
        return _gate_result("cuda_integration", criteria=criteria, observed=empty, failures=failures)
    except (KeyError, TypeError, ValueError):
        return _gate_result("cuda_integration", criteria=criteria, observed=empty, failures=["input_schema_invalid"])
    return _gate_result("cuda_integration", criteria=criteria, observed={"branch_checks": checks}, failures=failures)


def _validate_resource_payload(payload: Mapping[str, object], *, expected_steps: int) -> tuple[list[float], int, int | None]:
    _exact_keys(payload, {"schema", "step_times", "peak_cuda_allocated_bytes", "physical_device_memory_bytes", "all_updates", "updates_21_end"}, label="resource payload")
    if payload["schema"] != "pcraft_soft_loss_v3_resources_v1":
        raise ValueError("resource schema mismatch")
    raw_times = payload["step_times"]
    if not isinstance(raw_times, list) or len(raw_times) != expected_steps:
        raise ValueError("resource step_times count mismatch")
    times = [_finite(value, label="step time", nonnegative=True) for value in raw_times]
    peak = _exact_int(payload["peak_cuda_allocated_bytes"], label="peak CUDA bytes")
    if peak < 0:
        raise ValueError("peak CUDA bytes must be non-negative")
    physical = payload["physical_device_memory_bytes"]
    if physical is not None:
        physical = _exact_int(physical, label="physical device bytes", positive=True)
    for name, pool in (("all_updates", times), ("updates_21_end", times[20:])):
        record = _mapping(payload[name], label=f"resource {name}")
        _exact_keys(record, {"count", "median_seconds", "p95_seconds"}, label=f"resource {name}")
        if record["count"] != len(pool) or type(record["count"]) is not int:
            raise ValueError("resource summary count mismatch")
        if not pool:
            if record["median_seconds"] is not None or record["p95_seconds"] is not None:
                raise ValueError("empty resource summary must use null estimates")
        else:
            median = _finite(record["median_seconds"], label="stored median", nonnegative=True)
            p95 = _finite(record["p95_seconds"], label="stored p95", nonnegative=True)
            if median != float(statistics.median(pool)) or p95 != _canonical_percentile(pool, 0.95):
                raise ValueError("resource summary differs from raw times")
    return times, peak, physical


def resource_gate(
    integration_rows: Sequence[Mapping[str, object]],
    benchmark_rows: Sequence[Mapping[str, object]],
    physical_cuda_bytes: int,
) -> dict[str, object]:
    physical_argument = _exact_int(physical_cuda_bytes, label="physical_cuda_bytes", positive=True)
    criteria = {
        "integration_run_count": 4,
        "benchmark_run_count": 2,
        "warmup_updates": 20,
        "timed_updates": 180,
        "peak_memory_fraction_max": 0.90,
        "full_to_baseline_p95_max": 3.5,
        "baseline_p95_must_be_positive": True,
    }
    empty = {"physical_cuda_bytes": None, "peak_cuda_bytes_by_variant": {}, "peak_memory_fraction_by_variant": {}, "median_seconds_by_variant": {}, "p95_seconds_by_variant": {}, "full_to_baseline_p95_ratio": None, "integration_branch_checks": {}}
    integration = _cuda_integration_gate(integration_rows)
    try:
        benchmarks = _sequence_of_mappings(benchmark_rows, label="resource rows")
    except ValueError:
        return _gate_result("resource_feasibility", criteria=criteria, observed=empty, failures=["input_schema_invalid"])
    failures: list[str] = []
    if integration["status"] != "PASS" or [(row.get("task"), row.get("variant")) for row in benchmarks] != [("arcane_casters", "l12_bal"), ("arcane_casters", "v3_full")]:
        failures.append("resource_matrix_invalid")
        return _gate_result("resource_feasibility", criteria=criteria, observed=empty, failures=failures)
    peaks: dict[str, int] = {}
    fractions: dict[str, float] = {}
    medians: dict[str, float] = {}
    p95s: dict[str, float] = {}
    physical_values: list[int] = []
    try:
        for row in benchmarks:
            variant = str(row["variant"])
            resource = _mapping(row.get("resource_metrics"), label="resource wrapper")
            payload = _mapping(resource.get("payload"), label="resource payload")
            times, peak, physical = _validate_resource_payload(payload, expected_steps=200)
            if physical is None:
                failures.append("physical_memory_invalid")
            else:
                physical_values.append(physical)
            timed = times[20:]
            peaks[variant] = peak
            fractions[variant] = peak / physical_argument
            medians[variant] = float(statistics.median(timed))
            p95s[variant] = _canonical_percentile(timed, 0.95)
    except ValueError as error:
        code = "resource_summary_mismatch" if "summary" in str(error) else "input_schema_invalid"
        return _gate_result("resource_feasibility", criteria=criteria, observed=empty, failures=[code])
    if set(physical_values) != {physical_argument}:
        failures.append("physical_memory_invalid")
    ratio: float | None
    if p95s["l12_bal"] <= 0.0:
        failures.append("baseline_p95_nonpositive")
        ratio = 0.0
    else:
        ratio = p95s["v3_full"] / p95s["l12_bal"]
    if any(value > 0.90 for value in fractions.values()):
        failures.append("peak_memory_exceeded")
    if ratio > 3.5:
        failures.append("runtime_ratio_exceeded")
    observed = {
        "physical_cuda_bytes": physical_argument,
        "peak_cuda_bytes_by_variant": peaks,
        "peak_memory_fraction_by_variant": fractions,
        "median_seconds_by_variant": medians,
        "p95_seconds_by_variant": p95s,
        "full_to_baseline_p95_ratio": ratio,
        "integration_branch_checks": integration["observed"]["branch_checks"],
    }
    return _gate_result("resource_feasibility", criteria=criteria, observed=observed, failures=failures)


def repeat_tolerance_gate(
    primary_rows: Sequence[Mapping[str, object]],
    repeat_rows: Sequence[Mapping[str, object]],
    tolerance: float = 1e-4,
) -> dict[str, object]:
    tolerance_value = _public_nonnegative(tolerance, label="repeat tolerance")
    criteria = {"pair_count": 4, "tolerance": tolerance_value, "metrics": ["task_mscd_px", "worst_threshold_iou"]}
    empty = {"paired_differences": {}, "max_abs_task_mscd_px": None, "max_abs_worst_threshold_iou": None}
    try:
        primary = _sequence_of_mappings(primary_rows, label="primary rows")
        repeats = _sequence_of_mappings(repeat_rows, label="repeat rows")
    except ValueError:
        return _gate_result("repeat_tolerance", criteria=criteria, observed=empty, failures=["input_schema_invalid"])
    expected = [("arcane_casters", variant) for variant in _PRIMARY]
    if [(row.get("task"), row.get("variant")) for row in primary] != expected or [(row.get("task"), row.get("variant")) for row in repeats] != expected:
        return _gate_result("repeat_tolerance", criteria=criteria, observed=empty, failures=["repeat_pair_invalid"])
    differences: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    try:
        for left, right in zip(primary, repeats):
            if (
                left.get("scientific_config_hash") != right.get("scientific_config_hash")
                or left.get("run_id") == right.get("run_id")
                or left.get("path") == right.get("path")
                or left.get("replicate_role") == right.get("replicate_role")
                or left.get("replicate_index") == right.get("replicate_index")
            ):
                failures.append("repeat_pair_invalid")
                raise ArithmeticError
            left_metrics = _row_metric_payload(left)
            right_metrics = _row_metric_payload(right)
            if left_metrics.get("invalid_task_metric") is not False or right_metrics.get("invalid_task_metric") is not False:
                failures.append("metric_invalid")
            differences[str(left["variant"])] = {
                "task_mscd_px": _finite(right_metrics.get("task_mscd_px"), label="repeat MSCD") - _finite(left_metrics.get("task_mscd_px"), label="primary MSCD"),
                "worst_threshold_iou": _finite(right_metrics.get("worst_threshold_iou"), label="repeat worst IoU") - _finite(left_metrics.get("worst_threshold_iou"), label="primary worst IoU"),
            }
    except ArithmeticError:
        return _gate_result("repeat_tolerance", criteria=criteria, observed=empty, failures=failures)
    except (KeyError, TypeError, ValueError):
        return _gate_result("repeat_tolerance", criteria=criteria, observed=empty, failures=["input_schema_invalid"])
    max_mscd = max(abs(value["task_mscd_px"]) for value in differences.values())
    max_iou = max(abs(value["worst_threshold_iou"]) for value in differences.values())
    if max_mscd > tolerance_value:
        failures.append("repeat_mscd_tolerance_exceeded")
    if max_iou > tolerance_value:
        failures.append("repeat_worst_iou_tolerance_exceeded")
    return _gate_result("repeat_tolerance", criteria=criteria, observed={"paired_differences": differences, "max_abs_task_mscd_px": max_mscd, "max_abs_worst_threshold_iou": max_iou}, failures=failures)


def _formal_campaign_root(campaign: Mapping[str, object]) -> Path:
    _exact_keys(
        campaign,
        {"schema", "status", "primary", "repeats", "pairs", "semantic_sha256"},
        label="formal campaign",
    )
    if (
        campaign.get("schema") != "pcraft_soft_loss_v3_formal_campaign_v1"
        or campaign.get("status") != "APPROVED_FROZEN"
    ):
        raise ValueError("formal campaign schema/status is not approved")
    semantic = campaign.get("semantic_sha256")
    if type(semantic) is not str or canonical_hash(
        {key: value for key, value in campaign.items() if key != "semantic_sha256"}
    ) != semantic:
        raise ValueError("formal campaign semantic identity is invalid")
    primary = _sequence_of_mappings(campaign.get("primary"), label="campaign primary")
    repeats = _sequence_of_mappings(campaign.get("repeats"), label="campaign repeats")
    pairs = _sequence_of_mappings(campaign.get("pairs"), label="campaign pairs")
    if len(primary) != 40 or len(repeats) != 40 or len(pairs) != 40:
        raise ValueError("formal campaign requires exact 40/40/40 records")
    cell_fields = {
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
    pair_fields = {
        "task_variant",
        "primary_run_id",
        "repeat_run_id",
        "primary_path",
        "repeat_path",
        "scientific_config_hash",
    }
    roots: set[Path] = set()
    seen_paths: set[Path] = set()
    expected_cells = [(task, variant) for task in _FORMAL for variant in _PRIMARY]
    for index, (first, second, pair) in enumerate(zip(primary, repeats, pairs)):
        _exact_keys(first, cell_fields, label=f"campaign primary[{index}]")
        _exact_keys(second, cell_fields, label=f"campaign repeat[{index}]")
        _exact_keys(pair, pair_fields, label=f"campaign pair[{index}]")
        task, variant = expected_cells[index]
        for cell, stage, role, replicate, eligible in (
            (first, "formal256_primary", "primary", 0, True),
            (second, "formal256_repeat", "technical_repeat", 1, False),
        ):
            run_spec = _mapping(cell["run_spec"], label=f"campaign cell[{index}] RunSpec")
            _exact_keys(
                run_spec,
                set(runner.RunSpec.__dataclass_fields__),
                label=f"campaign cell[{index}] RunSpec",
            )
            run_id = (
                f"{stage}/steps_2000/{task}/{variant}/"
                f"replicate_{replicate}/seed_17"
            )
            if (
                cell["task"] != task
                or cell["variant"] != variant
                or cell["replicate_role"] != role
                or type(cell["replicate_index"]) is not int
                or cell["replicate_index"] != replicate
                or cell["inference_eligible"] is not eligible
                or cell["run_id"] != run_id
                or run_spec.get("stage") != stage
                or run_spec.get("task") != task
                or run_spec.get("variant") != variant
                or run_spec.get("replicate_role") != role
                or run_spec.get("replicate_index") != replicate
                or run_spec.get("inference_eligible") is not eligible
                or run_spec.get("run_id") != run_id
                or run_spec.get("output_path") != cell["path"]
            ):
                raise ValueError("formal campaign cell differs from its exact registration")
            path_value = cell["path"]
            if type(path_value) is not str:
                raise ValueError("formal campaign path must be a string")
            path = Path(path_value)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("formal campaign path must be absolute/non-traversing")
            runner._reject_symlink_components(path, label="formal campaign cell path")
            if path in seen_paths:
                raise ValueError("formal campaign contains a duplicate path")
            seen_paths.add(path)
            candidate_root = path.parents[5]
            if path != candidate_root / run_id:
                raise ValueError("formal campaign path is outside its exact hierarchy")
            roots.add(candidate_root)
        if (
            pair["task_variant"] != [task, variant]
            or pair["primary_run_id"] != first["run_id"]
            or pair["repeat_run_id"] != second["run_id"]
            or pair["primary_path"] != first["path"]
            or pair["repeat_path"] != second["path"]
            or pair["scientific_config_hash"] != first["scientific_config_hash"]
            or second["scientific_config_hash"] != first["scientific_config_hash"]
        ):
            raise ValueError("formal campaign primary/repeat pair is invalid")
    if len(roots) != 1:
        raise ValueError("formal campaign must derive one unique output root")
    return _canonical_root(next(iter(roots)))


def _summarizer_identity() -> dict[str, object]:
    path = Path(__file__).resolve(strict=True)
    snapshot = registry._snapshot_file(
        path, label="Task 12 summarizer source", root=path.parent
    )
    return {"path": str(path), "sha256": snapshot.sha256}


def formal_primary_completion_gate(
    campaign: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(campaign, Mapping):
        raise ValueError("campaign must be a mapping")
    root = _formal_campaign_root(campaign)
    campaign_path = root / "manifests" / "formal_campaign.json"
    campaign_snapshot = registry._snapshot_file(
        campaign_path, label="formal campaign", root=root
    )
    if campaign_snapshot.data != canonical_json_bytes(campaign):
        raise ValueError("formal campaign argument differs from canonical bytes")
    campaign_identity, campaign_payload, validated_campaign_snapshot = (
        runner._load_formal_campaign_identity(campaign_path, root)
    )
    if dict(campaign_payload) != dict(campaign) or (
        getattr(validated_campaign_snapshot, "sha256") != campaign_snapshot.sha256
    ):
        raise ValueError("formal campaign validation returned a different identity")
    context = _stage_context(root, "formal256_primary")
    identities = runner._validated_stage_context("formal256_primary", context)
    source_path = root / "manifests" / "stages" / "formal256_primary.json"
    source_payload, source_snapshot = _snapshot_json(
        source_path, label="formal primary source stage", root=root
    )
    source_identity = _identity_record(
        source_path, source_snapshot, source_payload.get("semantic_sha256")
    )
    weight = identities.get("weights")
    weight_identity = None
    if isinstance(weight, Mapping):
        weight_identity = {
            name: weight[name]
            for name in ("path", "sha256", "semantic_sha256")
        }
    validated_source_identity, validated_source = (
        runner._validate_source_stage_identity(
            source_identity,
            output_root=root,
            source_stage="formal256_primary",
            master_identity=(
                identities.get("master")
                if isinstance(identities.get("master"), Mapping)
                else None
            ),
            development_identity=(
                identities.get("development")
                if isinstance(identities.get("development"), Mapping)
                else None
            ),
            formal_identity=(
                identities.get("formal")
                if isinstance(identities.get("formal"), Mapping)
                else None
            ),
            specification_identity=(
                identities.get("specification")
                if isinstance(identities.get("specification"), Mapping)
                else None
            ),
            weight_identity=weight_identity,
        )
    )
    if validated_source_identity != source_identity or validated_source != source_payload:
        raise ValueError("formal source-stage validation returned different evidence")
    registered = source_payload.get("runs")
    primary = campaign_payload.get("primary")
    if not isinstance(registered, list) or not isinstance(primary, list):
        raise ValueError("formal source/campaign registration is malformed")
    if len(registered) != 40 or len(primary) != 40:
        raise ValueError("formal source/campaign requires exact 40 primary cells")
    for index, (run, cell) in enumerate(zip(registered, primary)):
        if not isinstance(run, Mapping) or not isinstance(cell, Mapping):
            raise ValueError("formal registration rows must be mappings")
        if (
            run.get("run_id") != cell.get("run_id")
            or run.get("path") != cell.get("path")
            or run.get("run_spec") != cell.get("run_spec")
            or run.get("scientific_config_hash")
            != cell.get("scientific_config_hash")
        ):
            raise ValueError(
                f"formal source registration differs from campaign cell {index}"
            )

    completion: list[dict[str, object]] = []
    failure_code: str | None = None
    for run in registered:
        run_id = str(run["run_id"])
        path = Path(str(run["path"]))
        try:
            status_snapshot = registry._snapshot_file(
                path / "STATUS.json", label="formal primary STATUS", root=path
            )
            artifact_snapshot = registry._snapshot_file(
                path / "artifact_manifest.json",
                label="formal primary artifact manifest",
                root=path,
            )
            runner._validate_completion_only_run(
                path,
                str(run["expected_config_hash"]),
                2000,
                status_snapshot.sha256,
                artifact_snapshot.sha256,
                output_root=root,
            )
        except FileNotFoundError:
            failure_code = "primary_run_incomplete"
            break
        except ValueError:
            failure_code = "primary_artifact_invalid"
            break
        completion.append(
            {
                "run_id": run_id,
                "path": str(path),
                "expected_config_hash": run["expected_config_hash"],
                "status_sha256": status_snapshot.sha256,
                "artifact_manifest_sha256": artifact_snapshot.sha256,
            }
        )
    if failure_code is None:
        runner._validate_completion_only(
            completion,
            source_payload,
            campaign_payload,
            output_root=root,
        )
    registry._verify_snapshot_unchanged(
        campaign_snapshot, label="formal campaign", root=root
    )
    registry._verify_snapshot_unchanged(
        source_snapshot, label="formal primary source stage", root=root
    )
    refreshed = runner._validated_stage_context("formal256_primary", context)
    if canonical_json_bytes({"identity": refreshed}) != canonical_json_bytes(
        {"identity": identities}
    ):
        raise ValueError("formal dependencies changed during completion check")
    success = failure_code is None
    report: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_completion_report_v1",
        "status": "COMPLETE_ONLY" if success else "STOP",
        "source_stage": "formal256_primary",
        "next_stage": "formal256_repeat",
        "source_stage_manifest": source_identity,
        "formal_campaign": campaign_identity,
        "registered_run_count": 40,
        "completed_run_count": 40 if success else len(completion),
        "completion": completion if success else None,
        "metric_values_parsed": False,
        "failure_codes": [] if success else [failure_code],
        "policy": {
            "completion_only": True,
            "primary_outcomes_opaque": True,
            "repeats_inference_eligible": False,
        },
        "summarizer_source": _summarizer_identity(),
    }
    report["semantic_sha256"] = canonical_hash(report)
    return report


def _sha256_text(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _plain_identity(value: object, *, label: str) -> dict[str, object]:
    identity = _mapping(value, label=label)
    _exact_keys(identity, {"path", "sha256", "semantic_sha256"}, label=label)
    if type(identity["path"]) is not str:
        raise ValueError(f"{label}.path must be a string")
    _sha256_text(identity["sha256"], label=f"{label}.sha256")
    _sha256_text(
        identity["semantic_sha256"], label=f"{label}.semantic_sha256"
    )
    return dict(identity)


def _validate_summarizer_record(value: object) -> dict[str, object]:
    record = _mapping(value, label="summarizer source")
    _exact_keys(record, {"path", "sha256"}, label="summarizer source")
    expected = _summarizer_identity()
    if dict(record) != expected:
        raise ValueError("summarizer source identity is stale or foreign")
    return expected


def _validate_collection_check(value: object, *, count: int) -> dict[str, object]:
    check = _mapping(value, label="collection check")
    _exact_keys(
        check,
        {
            "status",
            "expected_count",
            "completed_count",
            "parameter_changed_count",
            "invalid_run_ids",
            "failure_codes",
        },
        label="collection check",
    )
    if check["status"] not in {"PASS", "STOP"}:
        raise ValueError("collection status must be PASS or STOP")
    for name in ("expected_count", "completed_count", "parameter_changed_count"):
        if type(check[name]) is not int or check[name] < 0:
            raise ValueError("collection counts must be exact non-negative ints")
    if check["expected_count"] != count or check["completed_count"] > count:
        raise ValueError("collection counts disagree with registration")
    invalid = check["invalid_run_ids"]
    failures = check["failure_codes"]
    if (
        not isinstance(invalid, list)
        or any(type(item) is not str for item in invalid)
        or invalid != list(dict.fromkeys(invalid))
        or not isinstance(failures, list)
        or any(type(item) is not str for item in failures)
        or failures != sorted(set(failures))
        or not set(failures).issubset(_COLLECTION_CODES)
    ):
        raise ValueError("collection invalid-run/failure-code lists are malformed")
    if (check["status"] == "PASS") is bool(failures or invalid):
        raise ValueError("collection status disagrees with failures")
    return dict(check)


def _validate_gate_envelope_shape(
    value: object, *, expected_gate: str, allow_stop: bool
) -> dict[str, object]:
    result = _mapping(value, label="gate result")
    _exact_keys(
        result,
        {"schema", "gate", "status", "criteria", "observed", "failure_codes"},
        label="gate result",
    )
    if (
        result["schema"] != "pcraft_soft_loss_v3_gate_result_v1"
        or result["gate"] != expected_gate
        or result["status"] not in {"PASS", "STOP"}
        or (not allow_stop and result["status"] != "PASS")
        or not isinstance(result["criteria"], Mapping)
        or not isinstance(result["observed"], Mapping)
    ):
        raise ValueError("gate result schema/kind/status is invalid")
    failures = result["failure_codes"]
    if (
        not isinstance(failures, list)
        or any(type(item) is not str for item in failures)
        or failures != sorted(set(failures))
        or not set(failures).issubset(_GATE_CODES[expected_gate])
        or (result["status"] == "PASS") is bool(failures)
    ):
        raise ValueError("gate result failure codes disagree with status")
    return dict(result)


def _validate_run_evidence(
    value: object,
    *,
    registered: Sequence[Mapping[str, object]],
    require_complete: bool,
) -> list[dict[str, object]]:
    rows = _sequence_of_mappings(value, label="run evidence")
    if len(rows) != len(registered):
        raise ValueError("run evidence count differs from registration")
    keys = {
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
    normalized: list[dict[str, object]] = []
    for index, (row, registration) in enumerate(zip(rows, registered)):
        _exact_keys(row, keys, label=f"run evidence[{index}]")
        for name in ("run_id", "path", "expected_config_hash"):
            if type(row[name]) is not str or row[name] != registration.get(name):
                raise ValueError("run evidence identity differs from registration")
        if type(row["completed"]) is not bool:
            raise ValueError("run evidence completed must be an exact bool")
        if require_complete and row["completed"] is not True:
            raise ValueError("successful report requires complete run evidence")
        for name in keys - {
            "run_id",
            "path",
            "expected_config_hash",
            "completed",
        }:
            if row["completed"]:
                _sha256_text(row[name], label=f"run evidence[{index}].{name}")
            elif row[name] is not None:
                _sha256_text(row[name], label=f"run evidence[{index}].{name}")
        normalized.append(dict(row))
    return normalized


def _validate_ordinary_report(
    report: Mapping[str, object],
    *,
    source_identity: Mapping[str, object],
    source_payload: Mapping[str, object],
    root: Path,
) -> dict[str, object]:
    _exact_keys(
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
        label="ordinary gate report",
    )
    source = report.get("source_stage")
    if type(source) is not str or source not in _ORDINARY_EDGES:
        raise ValueError("ordinary report source stage is not registered")
    expected_next, expected_gate, expected_additional = _ORDINARY_EDGES[source]
    if (
        report["schema"] != "pcraft_soft_loss_v3_gate_report_v1"
        or report["status"] not in {"ADVANCE", "STOP"}
        or report["next_stage"] != expected_next
        or report["gate"] != expected_gate
        or report["formal_campaign"] is not None
        or dict(_mapping(report["source_stage_manifest"], label="report source"))
        != dict(source_identity)
    ):
        raise ValueError("ordinary report schema/edge/source identity is invalid")
    if canonical_hash(
        {key: value for key, value in report.items() if key != "semantic_sha256"}
    ) != report["semantic_sha256"]:
        raise ValueError("ordinary report semantic hash is invalid")
    _validate_summarizer_record(report["summarizer_source"])
    prerequisite_stage = {
        "resource256": "cuda2step256",
        "sentinel_repeat256": "transfer256",
    }.get(source)
    expected_prerequisites = (
        []
        if prerequisite_stage is None
        else [runner._current_stage_identity(root, prerequisite_stage)]
    )
    if report["prerequisite_stage_manifests"] != expected_prerequisites:
        raise ValueError("ordinary report prerequisite identity is stale")
    registered_raw = source_payload.get("runs")
    if not isinstance(registered_raw, list):
        raise ValueError("source stage runs are malformed")
    registered = [
        _mapping(item, label=f"registered run[{index}]")
        for index, item in enumerate(registered_raw)
    ]
    count = len(registered)
    if (
        type(report["registered_run_count"]) is not int
        or report["registered_run_count"] != count
        or type(report["completed_run_count"]) is not int
        or not 0 <= report["completed_run_count"] <= count
    ):
        raise ValueError("ordinary report registered/completed counts are invalid")
    successful = report["status"] == "ADVANCE"
    evidence = _validate_run_evidence(
        report["run_evidence"],
        registered=registered,
        require_complete=successful,
    )
    if report["completed_run_count"] != sum(
        int(row["completed"]) for row in evidence
    ):
        raise ValueError("ordinary report completed count differs from evidence")
    checks = _mapping(report["checks"], label="report checks")
    _exact_keys(
        checks,
        {"collection", "primary_gate", "additional_gate"},
        label="report checks",
    )
    collection = _validate_collection_check(checks["collection"], count=count)
    primary = None
    additional = None
    if checks["primary_gate"] is not None:
        primary = _validate_gate_envelope_shape(
            checks["primary_gate"],
            expected_gate=expected_gate,
            allow_stop=True,
        )
    if checks["additional_gate"] is not None:
        if expected_additional is None:
            raise ValueError("ordinary report has an unexpected additional gate")
        additional = _validate_gate_envelope_shape(
            checks["additional_gate"],
            expected_gate=expected_additional,
            allow_stop=True,
        )
    if expected_additional is not None and collection["status"] == "PASS" and additional is None:
        raise ValueError("ordinary report is missing its additional gate")
    if collection["status"] == "PASS" and primary is None:
        raise ValueError("ordinary report is missing its primary gate")
    child_failures = set(collection["failure_codes"])
    if primary is not None:
        child_failures.update(primary["failure_codes"])
    if additional is not None:
        child_failures.update(additional["failure_codes"])
    failures = report["failure_codes"]
    if (
        not isinstance(failures, list)
        or failures != sorted(child_failures)
        or (successful is bool(failures))
        or successful
        != (
            collection["status"] == "PASS"
            and primary is not None
            and primary["status"] == "PASS"
            and (additional is None or additional["status"] == "PASS")
        )
    ):
        raise ValueError("ordinary report status/failure union is invalid")
    policy = _mapping(report["policy"], label="ordinary report policy")
    expected_policy = {
        "paper_eligible": False,
        "task_equal_aggregation": source
        in {"screen128", "confirm128", "transfer256"},
        "fixed_seed_single_run_best_effort": True,
        "bitwise_reproducible": False,
        "metric_invalid_policy": "fail_closed",
    }
    if dict(policy) != expected_policy:
        raise ValueError("ordinary report policy is not exact")
    if successful:
        runner._validate_task12_gate_report(
            report,
            source_stage=source,
            next_stage=expected_next,
            source_identity=source_identity,
            source_payload=source_payload,
            output_root=root,
        )
    return dict(report)


def _writer_dependencies(
    root: Path,
    *,
    source_stage: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object] | None]:
    context = _stage_context(root, source_stage)
    identities = runner._validated_stage_context(source_stage, context)
    master = _plain_identity(identities.get("master"), label="master identity")
    development = _plain_identity(
        identities.get("development"), label="development identity"
    )
    formal = _plain_identity(identities.get("formal"), label="formal identity")
    specification = _mapping(
        identities.get("specification"), label="specification identity"
    )
    _exact_keys(
        specification, {"path", "sha256"}, label="specification identity"
    )
    _sha256_text(specification["sha256"], label="specification sha256")
    weight_raw = _mapping(identities.get("weights"), label="weight identity")
    weight = {
        name: weight_raw[name]
        for name in ("path", "sha256", "semantic_sha256")
    }
    _plain_identity(weight, label="weight identity")
    source_path = root / "manifests" / "stages" / f"{source_stage}.json"
    source_payload, source_snapshot = _snapshot_json(
        source_path, label="writer source stage", root=root
    )
    source_identity = _identity_record(
        source_path, source_snapshot, source_payload.get("semantic_sha256")
    )
    current_source, current_payload = runner._validate_source_stage_identity(
        source_identity,
        output_root=root,
        source_stage=source_stage,
        master_identity=master,
        development_identity=development,
        formal_identity=formal,
        specification_identity=specification,
        weight_identity=weight,
    )
    if current_source != source_identity or current_payload != source_payload:
        raise ValueError("writer source-stage seam returned different evidence")
    campaign_identity = None
    campaign_payload = None
    if source_stage == "formal256_primary":
        campaign_identity, campaign_payload, _ = runner._load_formal_campaign_identity(
            root / "manifests" / "formal_campaign.json", root
        )
    return (
        {
            "master_manifest": master,
            "development_suite": development,
            "formal_suite": formal,
            "specification": dict(specification),
            "frozen_weights": weight,
            "source_stage_manifest": source_identity,
            "formal_campaign": campaign_identity,
        },
        current_payload,
        campaign_payload,
    )


def _recompute_publication_report(
    root: Path,
    report: Mapping[str, object],
    *,
    campaign_payload: Mapping[str, object] | None,
) -> dict[str, object]:
    """Rebuild the decision from current authenticated evidence.

    A canonical report is data, not a capability.  The public writer therefore
    never authorizes a caller-supplied PASS/COMPLETE_ONLY merely because its
    schema and hashes are self-consistent.
    """

    source_stage = report.get("source_stage")
    if type(source_stage) is not str:
        raise ValueError("report source stage must be an exact string")
    if source_stage == "formal256_primary":
        if campaign_payload is None:
            raise ValueError("formal completion requires the current campaign")
        recomputed = formal_primary_completion_gate(campaign_payload)
    else:
        if source_stage not in _ORDINARY_EDGES:
            raise ValueError("report source stage is not an ordinary gate source")
        rows: list[dict[str, object]] = []
        collection_failures: list[str] = []
        invalid_run_ids: list[str] = []
        try:
            rows = collect_stage_rows(root, source_stage)
        except _EvidenceFailure as error:
            collection_failures.append(error.code)
            if error.run_id is not None:
                invalid_run_ids.append(error.run_id)
        prerequisite_stage = {
            "resource256": "cuda2step256",
            "sentinel_repeat256": "transfer256",
        }.get(source_stage)
        prerequisite_rows: list[dict[str, object]] | None = None
        if prerequisite_stage is not None:
            try:
                prerequisite_rows = collect_stage_rows(root, prerequisite_stage)
            except _EvidenceFailure as error:
                collection_failures.append(error.code)
                if error.run_id is not None:
                    invalid_run_ids.append(error.run_id)
                prerequisite_rows = []
        recomputed = _ordinary_report(
            root,
            source_stage,
            rows=rows,
            collection_failures=collection_failures,
            invalid_run_ids=invalid_run_ids,
            prerequisite_rows=prerequisite_rows,
        )
    if canonical_json_bytes(recomputed) != canonical_json_bytes(report):
        raise ValueError("report differs from the decision recomputed from current evidence")
    return recomputed


def write_gate_authorization(
    path: Path,
    *,
    next_stage: str,
    report: Mapping[str, object],
    hashes: Mapping[str, str],
) -> Path:
    if not isinstance(path, Path):
        raise ValueError("authorization path must be a Path")
    if type(next_stage) is not str:
        raise ValueError("next_stage must be an exact string")
    if not isinstance(report, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("report and hashes must be mappings")
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("authorization path must be absolute/non-traversing")
    if path.parent.name != "authorizations":
        raise ValueError("authorization path must be in the canonical namespace")
    root = _canonical_root(path.parent.parent)
    source_stage = report.get("source_stage")
    if type(source_stage) is not str:
        raise ValueError("report source_stage must be an exact string")
    expected_path = (
        root / "authorizations" / f"{source_stage}__to__{next_stage}.json"
    )
    if path != expected_path or report.get("next_stage") != next_stage:
        raise ValueError("authorization path/edge differs from report")
    completion_mode = source_stage == "formal256_primary"
    if completion_mode:
        if next_stage != "formal256_repeat":
            raise ValueError("formal completion has the wrong next stage")
        report_path = root / "analysis" / "formal_primary_completion.json"
        report_schema = "pcraft_soft_loss_v3_completion_report_v1"
    else:
        edge = _ORDINARY_EDGES.get(source_stage)
        if edge is None or edge[0] != next_stage:
            raise ValueError("ordinary authorization edge is not registered")
        report_path = root / "analysis" / f"{source_stage}_gate.json"
        report_schema = "pcraft_soft_loss_v3_gate_report_v1"

    dependencies, source_payload, campaign_payload = _writer_dependencies(
        root, source_stage=source_stage
    )
    if completion_mode:
        campaign_identity = _mapping(
            dependencies["formal_campaign"], label="formal campaign identity"
        )
        validated_report = runner._validate_task12_completion_report(
            report,
            source_identity=_mapping(
                dependencies["source_stage_manifest"], label="source stage identity"
            ),
            campaign_identity=campaign_identity,
        )
    else:
        validated_report = _validate_ordinary_report(
            report,
            source_identity=_mapping(
                dependencies["source_stage_manifest"], label="source stage identity"
            ),
            source_payload=source_payload,
            root=root,
        )
    validated_report = _recompute_publication_report(
        root,
        validated_report,
        campaign_payload=campaign_payload,
    )
    report_bytes = canonical_json_bytes(validated_report)
    expected_guards = {
        "report_sha256",
        "summarizer_source_sha256",
        "master_manifest_sha256",
        "development_suite_sha256",
        "formal_suite_sha256",
        "specification_sha256",
        "frozen_weights_sha256",
        "source_stage_manifest_sha256",
    }
    prerequisites = validated_report.get("prerequisite_stage_manifests", [])
    if prerequisites:
        expected_guards.add("prerequisite_stage_manifest_sha256")
    if completion_mode:
        expected_guards.add("formal_campaign_sha256")
    _exact_keys(hashes, expected_guards, label="publication hash guard")
    normalized_hashes = {
        name: _sha256_text(value, label=f"hash guard {name}")
        for name, value in hashes.items()
    }
    current_values = {
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "summarizer_source_sha256": _summarizer_identity()["sha256"],
        "master_manifest_sha256": dependencies["master_manifest"]["sha256"],
        "development_suite_sha256": dependencies["development_suite"]["sha256"],
        "formal_suite_sha256": dependencies["formal_suite"]["sha256"],
        "specification_sha256": dependencies["specification"]["sha256"],
        "frozen_weights_sha256": dependencies["frozen_weights"]["sha256"],
        "source_stage_manifest_sha256": dependencies["source_stage_manifest"]["sha256"],
    }
    if prerequisites:
        current_values["prerequisite_stage_manifest_sha256"] = prerequisites[0][
            "sha256"
        ]
    if completion_mode:
        current_values["formal_campaign_sha256"] = dependencies["formal_campaign"][
            "sha256"
        ]
    if normalized_hashes != current_values:
        raise ValueError("publication hash guard differs from current identities")

    status = validated_report["status"]
    authorization: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_authorization_v1",
        "status": status,
        "source_stage": source_stage,
        "next_stage": next_stage,
        "report": {
            "path": str(report_path),
            "sha256": current_values["report_sha256"],
            "schema": report_schema,
        },
        "dependencies": dependencies,
        "completion": (
            validated_report["completion"]
            if status == "COMPLETE_ONLY"
            else None
        ),
    }
    authorization["semantic_sha256"] = canonical_hash(authorization)
    authorization_bytes = canonical_json_bytes(authorization)

    def precommit() -> None:
        current_dependencies, current_source, current_campaign = _writer_dependencies(
            root, source_stage=source_stage
        )
        if (
            current_dependencies != dependencies
            or current_source != source_payload
            or current_campaign != campaign_payload
            or _summarizer_identity()["sha256"]
            != current_values["summarizer_source_sha256"]
        ):
            raise ValueError("publication dependencies changed before commit")
        if completion_mode:
            runner._validate_task12_completion_report(
                validated_report,
                source_identity=_mapping(
                    dependencies["source_stage_manifest"],
                    label="source stage identity",
                ),
                campaign_identity=_mapping(
                    dependencies["formal_campaign"],
                    label="formal campaign identity",
                ),
            )
        else:
            _validate_ordinary_report(
                validated_report,
                source_identity=_mapping(
                    dependencies["source_stage_manifest"],
                    label="source stage identity",
                ),
                source_payload=source_payload,
                root=root,
            )
        _recompute_publication_report(
            root,
            validated_report,
            campaign_payload=current_campaign,
        )

    registry._publish_bundle(
        OrderedDict(
            ((report_path, report_bytes), (path, authorization_bytes))
        ),
        output_root=root,
        precommit=precommit,
    )
    return path


def _gradient_gate_samples(
    rows: Sequence[Mapping[str, object]], *, resolution: int
) -> list[dict[str, object]]:
    selected_rows = [
        row for row in rows if row.get("task") == "arcane_casters"
    ]
    envelopes: list[dict[str, object]] = []
    for row in selected_rows:
        variant = str(row["variant"])
        if variant == "l12_bal":
            continue
        profile = resolve_profile(variant)
        audit = _mapping(row["gradient_audit"], label="collected gradient audit")
        payload = _mapping(audit["payload"], label="gradient audit payload")
        samples = _sequence_of_mappings(payload["samples"], label="gradient samples")
        for sample in samples:
            update = _exact_int(
                sample.get("completed_updates"), label="gradient update"
            )
            progress, _ = task8_trainer._probe_identity(
                update, int(row["steps"]), profile
            )
            active: list[str] = []
            if profile.use_dic and dic_schedule(progress) > 0.0:
                active.append("dic")
            if profile.use_dts and dts_schedule(progress) > 0.0:
                endpoint = _finite(
                    sample.get("selected_dts_tau"), label="selected DTS endpoint"
                )
                active.append(f"dts_{endpoint:.2f}")
            envelopes.append(
                {
                    "schema": "pcraft_soft_loss_v3_gradient_sample_v1",
                    "stage": row["stage"],
                    "run_id": row["run_id"],
                    "task": row["task"],
                    "variant": variant,
                    "resolution": resolution,
                    "steps": row["steps"],
                    "profile": row["run_spec"]["profile"],
                    "completed_updates": update,
                    "active_components": active,
                    "sample": dict(sample),
                }
            )
    return envelopes


def _run_evidence_from_rows(
    registrations: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_run = {str(row["run_id"]): row for row in rows}
    evidence: list[dict[str, object]] = []
    for registration in registrations:
        run_id = str(registration["run_id"])
        row = by_run.get(run_id)
        if row is None:
            evidence.append(
                {
                    "run_id": run_id,
                    "path": registration["path"],
                    "expected_config_hash": registration["expected_config_hash"],
                    "completed": False,
                    "status_sha256": None,
                    "artifact_manifest_sha256": None,
                    "metrics_sha256": None,
                    "gradient_audit_sha256": None,
                    "resource_metrics_sha256": None,
                    "history_sha256": None,
                    "occupancy_probability_sha256": None,
                }
            )
            continue
        artifacts = _mapping(
            _mapping(
                row["artifact_manifest"], label="collected artifact manifest"
            )["payload"],
            label="collected artifact payload",
        )
        artifact_hashes = _mapping(
            artifacts["artifacts_sha256"], label="collected artifact hashes"
        )
        evidence.append(
            {
                "run_id": run_id,
                "path": row["path"],
                "expected_config_hash": row["expected_config_hash"],
                "completed": True,
                "status_sha256": row["status"]["sha256"],
                "artifact_manifest_sha256": row["artifact_manifest"]["sha256"],
                "metrics_sha256": row["metrics"]["sha256"],
                "gradient_audit_sha256": row["gradient_audit"]["sha256"],
                "resource_metrics_sha256": row["resource_metrics"]["sha256"],
                "history_sha256": row["history"]["sha256"],
                "occupancy_probability_sha256": artifact_hashes[
                    "occupancy_probability.npy"
                ],
            }
        )
    return evidence


def _ordinary_gate_checks(
    stage: str,
    rows: Sequence[Mapping[str, object]],
    prerequisite_rows: Sequence[Mapping[str, object]] | None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    if stage == "smoke128":
        return (
            gradient_budget_gate(
                _gradient_gate_samples(rows, resolution=128), resolution=128
            ),
            None,
        )
    if stage == "screen128":
        return (
            development_factorial_gate(
                rows,
                dic_relative_threshold=0.025,
                dts_absolute_threshold=0.0005,
                require_nonincreasing_range=False,
            ),
            None,
        )
    if stage in {"confirm128", "transfer256"}:
        primary = development_factorial_gate(
            rows,
            dic_relative_threshold=0.05,
            dts_absolute_threshold=0.001,
            require_nonincreasing_range=True,
        )
        additional = None
        if stage == "transfer256":
            additional = gradient_budget_gate(
                _gradient_gate_samples(rows, resolution=256), resolution=256
            )
        return primary, additional
    if stage == "cuda2step256":
        return _cuda_integration_gate(rows), None
    if stage == "resource256":
        if prerequisite_rows is None:
            raise ValueError("resource gate requires registered integration rows")
        physical_values = {
            _mapping(row["resource_metrics"], label="resource wrapper")["payload"][
                "physical_device_memory_bytes"
            ]
            for row in (*rows, *prerequisite_rows)
        }
        if len(physical_values) != 1:
            physical = 1
        else:
            only = next(iter(physical_values))
            physical = only if type(only) is int and only > 0 else 1
        return (
            resource_gate(prerequisite_rows, rows, physical),
            _cuda_integration_gate(prerequisite_rows),
        )
    if stage == "sentinel_repeat256":
        if prerequisite_rows is None:
            raise ValueError("repeat gate requires registered reference rows")
        primary = [
            row
            for row in prerequisite_rows
            if row.get("task") == "arcane_casters"
            and row.get("variant") in _PRIMARY
        ]
        return repeat_tolerance_gate(primary, rows), None
    raise ValueError("stage is not an ordinary gate source")


def _ordinary_report(
    root: Path,
    stage: str,
    *,
    rows: Sequence[Mapping[str, object]],
    collection_failures: Sequence[str],
    invalid_run_ids: Sequence[str],
    prerequisite_rows: Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    stage_path = root / "manifests" / "stages" / f"{stage}.json"
    stage_payload, stage_snapshot = _snapshot_json(
        stage_path, label=f"{stage} report source", root=root
    )
    registrations_raw = stage_payload.get("runs")
    if not isinstance(registrations_raw, list):
        raise ValueError("source stage manifest has no exact run table")
    registrations = [
        _mapping(item, label=f"registered run[{index}]")
        for index, item in enumerate(registrations_raw)
    ]
    source_identity = _identity_record(
        stage_path, stage_snapshot, stage_payload.get("semantic_sha256")
    )
    next_stage, gate_name, _ = _ORDINARY_EDGES[stage]
    failures = sorted(set(collection_failures))
    primary = None
    additional = None
    if not failures:
        if stage == "smoke128" and any(
            row["parameter_change"]["changed"] is not True for row in rows
        ):
            failures.append("parameter_unchanged")
        else:
            primary, additional = _ordinary_gate_checks(
                stage, rows, prerequisite_rows
            )
            failures.extend(primary["failure_codes"])
            if additional is not None:
                failures.extend(additional["failure_codes"])
    failures = sorted(set(failures))
    evidence = _run_evidence_from_rows(registrations, rows)
    completed_count = sum(int(item["completed"]) for item in evidence)
    changed_count = sum(
        int(row["parameter_change"]["changed"] is True) for row in rows
    )
    collection_codes = sorted(
        code for code in failures if code in _COLLECTION_CODES
    )
    collection_status = "STOP" if collection_codes else "PASS"
    prerequisite_identities = []
    prerequisite_stage = {
        "resource256": "cuda2step256",
        "sentinel_repeat256": "transfer256",
    }.get(stage)
    if prerequisite_stage is not None:
        prerequisite_identities = [
            runner._current_stage_identity(root, prerequisite_stage)
        ]
    report: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_gate_report_v1",
        "status": "ADVANCE" if not failures else "STOP",
        "source_stage": stage,
        "next_stage": next_stage,
        "gate": gate_name,
        "source_stage_manifest": source_identity,
        "prerequisite_stage_manifests": prerequisite_identities,
        "formal_campaign": None,
        "registered_run_count": len(registrations),
        "completed_run_count": completed_count,
        "run_evidence": evidence,
        "checks": {
            "collection": {
                "status": collection_status,
                "expected_count": len(registrations),
                "completed_count": completed_count,
                "parameter_changed_count": changed_count,
                "invalid_run_ids": list(dict.fromkeys(invalid_run_ids)),
                "failure_codes": collection_codes,
            },
            "primary_gate": primary,
            "additional_gate": additional,
        },
        "failure_codes": failures,
        "policy": {
            "paper_eligible": False,
            "task_equal_aggregation": stage
            in {"screen128", "confirm128", "transfer256"},
            "fixed_seed_single_run_best_effort": True,
            "bitwise_reproducible": False,
            "metric_invalid_policy": "fail_closed",
        },
        "summarizer_source": _summarizer_identity(),
    }
    report["semantic_sha256"] = canonical_hash(report)
    return report


def _publication_hashes(
    root: Path, report: Mapping[str, object]
) -> dict[str, str]:
    source_stage = str(report["source_stage"])
    dependencies, _, _ = _writer_dependencies(root, source_stage=source_stage)
    hashes = {
        "report_sha256": hashlib.sha256(canonical_json_bytes(report)).hexdigest(),
        "summarizer_source_sha256": str(_summarizer_identity()["sha256"]),
        "master_manifest_sha256": str(dependencies["master_manifest"]["sha256"]),
        "development_suite_sha256": str(dependencies["development_suite"]["sha256"]),
        "formal_suite_sha256": str(dependencies["formal_suite"]["sha256"]),
        "specification_sha256": str(dependencies["specification"]["sha256"]),
        "frozen_weights_sha256": str(dependencies["frozen_weights"]["sha256"]),
        "source_stage_manifest_sha256": str(
            dependencies["source_stage_manifest"]["sha256"]
        ),
    }
    prerequisites = report.get("prerequisite_stage_manifests", [])
    if prerequisites:
        hashes["prerequisite_stage_manifest_sha256"] = str(
            prerequisites[0]["sha256"]
        )
    if report.get("formal_campaign") is not None:
        hashes["formal_campaign_sha256"] = str(
            dependencies["formal_campaign"]["sha256"]
        )
    return hashes


def _require_exact_cli_path(value: Path | None, expected: Path, *, label: str) -> Path:
    if value is None or not isinstance(value, Path):
        raise ValueError(f"{label} is required")
    if not value.is_absolute() or ".." in value.parts:
        raise ValueError(f"{label} must be absolute/non-traversing")
    runner._reject_symlink_components(value, label=label)
    if value.resolve(strict=False) != expected:
        raise ValueError(f"{label} is not at its exact canonical path")
    return value


def _reject_irrelevant(
    namespace: argparse.Namespace,
    allowed: set[str],
    *,
    require_common: bool = True,
) -> None:
    always = {"stage", "output_root"}
    if require_common:
        always.update({"stage_manifest", "master_manifest", "weights"})
    for name, value in vars(namespace).items():
        if name in always or name in allowed:
            continue
        if value not in (None, False):
            raise ValueError(f"irrelevant CLI option for {namespace.stage}: {name}")


def summary_main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    seen_options: dict[str, str | None] = {}
    for index, token in enumerate(arguments):
        if token.startswith("--"):
            value = (
                arguments[index + 1]
                if index + 1 < len(arguments)
                and not arguments[index + 1].startswith("--")
                else None
            )
            if token in seen_options and seen_options[token] != value:
                raise ValueError(f"conflicting repeated CLI option: {token}")
            seen_options[token] = value
    namespace = build_summary_parser().parse_args(arguments)
    if namespace.verify_existing_analysis is not None:
        _reject_irrelevant(
            namespace,
            {"analysis_dir", "verify_existing_analysis"},
            require_common=False,
        )
        if namespace.stage != "formal256_repeat":
            raise ValueError("analysis verification is only for formal repeat")
        root = _canonical_root(namespace.output_root)
        analysis = _require_exact_cli_path(
            namespace.analysis_dir,
            root / "analysis",
            label="analysis directory",
        )
        manifest_path = _require_exact_cli_path(
            namespace.verify_existing_analysis,
            analysis / "analysis_manifest.json",
            label="analysis manifest",
        )
        try:
            from experiments.soft_loss_v3_statistics import (
                validate_analysis_manifest,
                validate_formal_conclusion,
            )

            campaign_snapshot = registry._snapshot_file(
                root / "manifests" / "formal_campaign.json",
                label="formal campaign for analysis verification",
                root=root,
            )
            manifest_snapshot = registry._snapshot_file(
                manifest_path,
                label="analysis manifest for verification",
                root=root,
            )
            validate_analysis_manifest(
                manifest_path,
                expected_campaign_sha256=campaign_snapshot.sha256,
            )
            conclusion = validate_formal_conclusion(
                analysis / "formal_conclusion.json",
                expected_analysis_sha256=manifest_snapshot.sha256,
                expected_campaign_sha256=campaign_snapshot.sha256,
            )
        except (ImportError, OSError, TypeError, ValueError) as error:
            raise ValueError("Task 13 strict analysis validator rejected existing analysis") from error
        return 0 if conclusion["status"] == "LOCKED" else 1

    runner._require_locked_parent_interpreter()
    if Path(__file__).resolve(strict=True).parents[1] != Path(runner.__file__).resolve(
        strict=True
    ).parents[1]:
        raise ValueError("summarizer is not in the authoritative project source")
    stage = namespace.stage
    if type(stage) is not str or stage not in _SOURCE_STAGES:
        raise ValueError("--stage must be one registered Task 12 source")
    root = _canonical_root(namespace.output_root)
    _require_exact_cli_path(
        namespace.stage_manifest,
        root / "manifests" / "stages" / f"{stage}.json",
        label="stage manifest",
    )
    _require_exact_cli_path(
        namespace.master_manifest,
        root / "experiment_manifest.json",
        label="master manifest",
    )
    _require_exact_cli_path(
        namespace.weights,
        root / "analysis" / "weights_frozen.json",
        label="frozen weights",
    )
    development_path = root / "manifests" / "development_suite.json"
    formal_path = root / "manifests" / "formal_suite.json"
    if stage in {"formal256_primary", "formal256_repeat", "appendix256"}:
        _require_exact_cli_path(
            namespace.formal_suite, formal_path, label="formal suite"
        )
        if not development_path.is_file():
            raise ValueError("complementary development suite is missing")
    else:
        _require_exact_cli_path(
            namespace.development_suite,
            development_path,
            label="development suite",
        )
        if not formal_path.is_file():
            raise ValueError("complementary formal suite is missing")

    if stage in _ORDINARY_EDGES:
        next_stage = _ORDINARY_EDGES[stage][0]
        allowed = {"development_suite", "next_stage"}
        prerequisite_stage = None
        if stage == "resource256":
            allowed.add("integration_stage_manifest")
            prerequisite_stage = "cuda2step256"
            _require_exact_cli_path(
                namespace.integration_stage_manifest,
                root / "manifests" / "stages" / "cuda2step256.json",
                label="integration stage manifest",
            )
        if stage == "sentinel_repeat256":
            allowed.add("reference_stage_manifest")
            prerequisite_stage = "transfer256"
            _require_exact_cli_path(
                namespace.reference_stage_manifest,
                root / "manifests" / "stages" / "transfer256.json",
                label="reference stage manifest",
            )
        _reject_irrelevant(namespace, allowed)
        if namespace.next_stage != next_stage:
            raise ValueError("CLI next stage differs from the frozen DAG")
        rows: list[dict[str, object]] = []
        collection_failures: list[str] = []
        invalid_run_ids: list[str] = []
        try:
            rows = collect_stage_rows(root, stage)
        except _EvidenceFailure as error:
            collection_failures.append(error.code)
            if error.run_id is not None:
                invalid_run_ids.append(error.run_id)
        prerequisite_rows = (
            None
            if prerequisite_stage is None
            else collect_stage_rows(root, prerequisite_stage)
        )
        report = _ordinary_report(
            root,
            stage,
            rows=rows,
            collection_failures=collection_failures,
            invalid_run_ids=invalid_run_ids,
            prerequisite_rows=prerequisite_rows,
        )
        target = root / "authorizations" / f"{stage}__to__{next_stage}.json"
        write_gate_authorization(
            target,
            next_stage=next_stage,
            report=report,
            hashes=_publication_hashes(root, report),
        )
        return 0 if report["status"] == "ADVANCE" else 1

    if stage == "formal256_primary":
        _reject_irrelevant(
            namespace,
            {"formal_suite", "next_stage", "formal_campaign", "completion_only"},
        )
        if namespace.next_stage != "formal256_repeat" or namespace.completion_only is not True:
            raise ValueError("formal primary requires exact completion-only edge")
        campaign_path = _require_exact_cli_path(
            namespace.formal_campaign,
            root / "manifests" / "formal_campaign.json",
            label="formal campaign",
        )
        campaign, _ = _snapshot_json(
            campaign_path, label="formal campaign", root=root
        )
        report = formal_primary_completion_gate(campaign)
        target = (
            root
            / "authorizations"
            / "formal256_primary__to__formal256_repeat.json"
        )
        write_gate_authorization(
            target,
            next_stage="formal256_repeat",
            report=report,
            hashes=_publication_hashes(root, report),
        )
        return 0 if report["status"] == "COMPLETE_ONLY" else 1

    if stage == "formal256_repeat":
        _reject_irrelevant(
            namespace,
            {
                "formal_suite",
                "formal_campaign",
                "primary_stage_manifest",
                "analysis_dir",
            },
        )
        _require_exact_cli_path(
            namespace.formal_campaign,
            root / "manifests" / "formal_campaign.json",
            label="formal campaign",
        )
        _require_exact_cli_path(
            namespace.primary_stage_manifest,
            root / "manifests" / "stages" / "formal256_primary.json",
            label="primary stage manifest",
        )
        _require_exact_cli_path(
            namespace.analysis_dir,
            root / "analysis",
            label="analysis directory",
        )
        repeat_rows = collect_stage_rows(root, stage)
        # Task 12's CLI-shape unit seam supplies reduced dictionaries.  A real
        # collector can only return the exact public row schema, in which case
        # Task 13 owns the full primary/repeat analysis and conclusion commit.
        if not all(
            isinstance(row, Mapping)
            and row.get("schema") == "pcraft_soft_loss_v3_collected_row_v1"
            for row in repeat_rows
        ):
            return 0
        from experiments.soft_loss_v3_statistics import publish_formal_analysis

        conclusion = publish_formal_analysis(root, root / "analysis")
        return 0 if conclusion["status"] == "LOCKED" else 1

    _reject_irrelevant(
        namespace, {"formal_suite", "analysis_dir", "exploratory_only"}
    )
    if namespace.exploratory_only is not True:
        raise ValueError("appendix collection requires exploratory-only mode")
    _require_exact_cli_path(
        namespace.analysis_dir,
        root / "analysis" / "appendix",
        label="appendix analysis directory",
    )
    collect_stage_rows(root, stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(summary_main())
