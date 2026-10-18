"""Read-only analysis for the preregistered Soft Loss v4 ten-task ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Mapping, Sequence


if __package__ in (None, ""):  # Permit the documented direct script launcher.
    _BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
    if str(_BOOTSTRAP_ROOT) not in sys.path:
        sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from shadowforge_pipeline.soft_voxel_v4.config import (
    canonical_trainer_argv,
    validate_resolved_config,
)
from shadowforge_pipeline.soft_voxel_v4.provenance import (
    canonical_hash,
    canonical_json_bytes,
    revalidate_snapshot,
    snapshot_regular_file,
)
from shadowforge_pipeline.soft_voxel_v4.trainer import validate_completed_run


_TASKS = tuple(f"connected_{index:02d}" for index in range(1, 11))
_PROFILES = ("l12_bal", "dic_only", "dts_only", "v4_full")
_AUXILIARY_PROFILES = _PROFILES[1:]
_EXPECTED_MATRIX = tuple((task, profile) for task in _TASKS for profile in _PROFILES)
_MANIFEST_KEYS = {
    "schema",
    "suite_identity",
    "specification_identity",
    "unit_coefficient_identity",
    "candidate_registry_identity",
    "candidate_row",
    "constants",
    "matrix",
    "decision_policy",
    "semantic_sha256",
}
_MATRIX_KEYS = {
    "task_id",
    "profile",
    "run_id",
    "run_dir",
    "config_hash",
    "fresh_argv",
    "resume_argv",
    "resolved_config",
}
_CONSTANTS = {
    "seed": 17,
    "optimizer": "Adam",
    "learning_rate": 0.007,
    "resolution": 128,
    "export_resolution": 128,
    "steps": 2000,
    "stage": "ablation10_128",
    "replicate_role": "primary",
    "replicate_index": 0,
}
_DECISION_POLICY = {
    "threshold": 0.05,
    "view_aggregation": "six_view_equal_weight",
    "statistical_unit": "task",
    "task_count": 10,
    "task_aggregation": "ordered_task_deltas_then_sum_divide_10_no_tolerance",
    "contrasts": {
        "dic": {"treatment": "v4_full", "control": "dts_only"},
        "dts": {"treatment": "v4_full", "control": "dic_only"},
        "full_vs_base": {"treatment": "v4_full", "control": "l12_bal"},
    },
    "component_key": {
        "mean_delta_iou_min": 0.01,
        "mean_delta_dice_min": 0.005,
        "positive_task_count_min": 8,
        "task_delta_iou_min": -0.005,
        "test": "one_sided_exact_sign_flip_task_equal_mean_inclusive_tail",
        "correction": "holm_dic_dts_iou_only",
        "adjusted_p_strict_max": 0.05,
    },
    "overall_quality": {
        "full_mean_iou_min": 0.95,
        "full_task_iou_min": 0.94,
        "full_base_mean_iou_min": -0.001,
        "full_base_mean_dice_min": -0.001,
    },
    "safety": {
        "profiles_vs_base": ["v4_full", "dic_only", "dts_only"],
        "task_recall_delta_min": -0.005,
        "mean_fpr_delta_max": 0.001,
    },
    "engineering": {
        "completed_runs": 40,
        "auxiliary_eligible_slots_min": 1,
        "engineering_gate_status": "PASS",
        "projected_zero_fraction_max": 0.10,
        "each_enabled_component_has_positive_final_norm": True,
        "required_components": {
            "dic_only": ["dic"],
            "dts_only": ["dts"],
            "v4_full": ["dic", "dts"],
        },
    },
    "labels": [
        "BOTH_KEY",
        "DIC_KEY",
        "DTS_KEY",
        "NEITHER_KEY",
        "HARMFUL_OR_ANTAGONISTIC",
        "INCONCLUSIVE",
    ],
    "invalid_evidence": "raise_without_publication",
    "below_definition": "mean_iou_below_mcid_or_mean_dice_below_mcid",
    "ambiguous_definition": "not_key_and_not_below",
    "label_precedence": [
        "engineering_stop_or_excess_zero_or_no_activation=>INCONCLUSIVE",
        "quality_or_safety_failure_or_negative_mean_iou_or_dice=>HARMFUL_OR_ANTAGONISTIC",
        "component_ambiguous=>INCONCLUSIVE",
        "both_key=>BOTH_KEY",
        "dic_key_and_dts_below=>DIC_KEY",
        "dts_key_and_dic_below=>DTS_KEY",
        "both_below=>NEITHER_KEY",
    ],
    "fixed_final_step": True,
    "best_checkpoint": False,
    "candidate_selection": "fixed_r08_r08_for_auxiliary_profiles",
    "candidate_search": False,
    "stage_dag": False,
    "repeats": False,
    "appendix": False,
    "resolution_256": False,
    "outcome_read": False,
}
_CSV_FIELDS = (
    "task_id",
    "profile",
    "run_id",
    "config_hash",
    "metrics_sha256",
    "gradient_audit_sha256",
    "mean_iou",
    "mean_dice",
    "mean_recall",
    "mean_fpr",
    "eligible_slots",
    "active_slots",
    "fully_projected_zero_slots",
    "fully_projected_zero_fraction",
    "engineering_status",
)
_METRICS_KEYS = {
    "schema",
    "config_hash",
    "thresholds",
    "task_iou_by_threshold",
    "worst_threshold_iou",
    "threshold_iou_range",
    "task_mscd_px",
    "task_p95_symmetric_distance_px",
    "invalid_task_metric",
}
_VIEW_KEYS = {
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
_AGGREGATE_KEYS = {
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
_AUDIT_KEYS = {
    "schema",
    "config_hash",
    "profile",
    "total_steps",
    "step_config",
    "eligible_slots",
    "samples",
    "engineering_gate",
    "redundancy_gate",
}
_ENGINEERING_KEYS = {
    "status",
    "eligible_slots",
    "fully_projected_zero_slots",
    "fully_projected_zero_fraction",
    "failure_codes",
}


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact mapping")
    return value


def _exact_keys(value: Mapping[str, object], keys: set[str], *, label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} has wrong exact keys")


def _exact_int(value: object, *, label: str, nonnegative: bool = False) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an exact integer")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _finite(value: object, *, label: str, unit: bool = False, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{label} must be finite")
    if unit and not 0.0 <= scalar <= 1.0:
        raise ValueError(f"{label} must lie in [0,1]")
    if nonnegative and scalar < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return scalar


def _exact_bool(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be an exact bool")
    return value


def _ordered_mean(values: Sequence[float]) -> float:
    if len(values) != 10:
        raise ValueError("task aggregation requires exactly ten values")
    return float(sum(values) / 10.0)


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """One-sided exact paired sign-flip p-value over exactly ten tasks."""

    if not isinstance(differences, Sequence) or len(differences) != 10:
        raise ValueError("exact sign-flip requires exactly ten differences")
    checked = [_finite(value, label=f"difference[{index}]") for index, value in enumerate(differences)]
    observed = _ordered_mean(checked)
    tail = 0
    for signs in itertools.product((-1.0, 1.0), repeat=10):
        permuted = float(sum(sign * value for sign, value in zip(signs, checked)) / 10.0)
        if permuted >= observed:
            tail += 1
    return float(tail / 1024.0)


def holm_adjust_two(dic_pvalue: float, dts_pvalue: float) -> dict[str, float]:
    """Holm-adjust exactly the registered two IoU hypotheses."""

    values = {
        "dic": _finite(dic_pvalue, label="DIC p-value", unit=True),
        "dts": _finite(dts_pvalue, label="DTS p-value", unit=True),
    }
    order = {"dic": 0, "dts": 1}
    ranked = sorted(values, key=lambda name: (values[name], order[name]))
    first, second = ranked
    first_adjusted = min(1.0, 2.0 * values[first])
    second_adjusted = max(first_adjusted, values[second])
    return {
        "dic": float(first_adjusted if first == "dic" else second_adjusted),
        "dts": float(first_adjusted if first == "dts" else second_adjusted),
    }


def _canonical_root(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate != Path(os.path.abspath(candidate)):
        raise ValueError("output root must be absolute and canonical")
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise ValueError("output root is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("output root must be a canonical directory")
    if candidate.resolve(strict=True) != candidate:
        raise ValueError("output root must contain no symlink component")
    return candidate


def _decode_canonical_json(snapshot: object, *, label: str) -> dict[str, object]:
    payload_bytes = getattr(snapshot, "payload", None)
    if type(payload_bytes) is not bytes:
        raise ValueError(f"{label} snapshot has no immutable bytes")
    try:
        decoded = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    payload = _mapping(decoded, label=label)
    try:
        expected = canonical_json_bytes(payload) + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not finite canonical JSON") from error
    if payload_bytes != expected:
        raise ValueError(f"{label} bytes are not canonical JSON")
    return payload


def _same_canonical(left: object, right: object, *, label: str) -> None:
    if type(left) is not dict or type(right) is not dict:
        raise ValueError(f"{label} must be an exact mapping")
    if canonical_json_bytes(left) != canonical_json_bytes(right):
        raise ValueError(f"{label} differs from the frozen registration")


def _argv_from_config(config: Mapping[str, object], *, resume: bool) -> tuple[str, ...]:
    run_identity = _mapping(config.get("run_identity"), label="resolved run identity")
    output_root = _mapping(config.get("output_root_identity"), label="output-root identity")
    profile = _mapping(config.get("profile"), label="resolved profile")
    shadows = config.get("shadow_inputs_xyz")
    if type(shadows) is not list or len(shadows) != 3:
        raise ValueError("resolved shadow input identity is malformed")
    shadow_files: list[str] = []
    for index, raw in enumerate(shadows):
        item = _mapping(raw, label=f"shadow input {index}")
        if type(item.get("path")) is not str:
            raise ValueError("resolved shadow path is malformed")
        shadow_files.append(item["path"])
    suite = _mapping(config.get("suite_manifest"), label="suite identity")
    specification = _mapping(config.get("specification"), label="specification identity")
    unit = config.get("unit_coefficient_identity")
    registry = config.get("candidate_registry_identity")
    budget = config.get("budget_identity")
    namespace = argparse.Namespace(
        run_kind=config.get("run_kind"),
        device=config.get("device"),
        output_root=output_root.get("path"),
        run_id=run_identity.get("run_id"),
        stage=run_identity.get("stage"),
        task_id=run_identity.get("task_id"),
        replicate_role=run_identity.get("replicate_role"),
        replicate_index=run_identity.get("replicate_index"),
        steps=config.get("steps"),
        learning_rate=config.get("learning_rate"),
        shadow_files=shadow_files,
        resolution=config.get("resolution"),
        export_resolution=config.get("export_resolution"),
        profile=profile.get("name"),
        suite_manifest=suite.get("path"),
        specification=specification.get("path"),
        unit_coefficients=None if unit is None else _mapping(unit, label="unit identity").get("path"),
        candidate_registry=None if registry is None else _mapping(registry, label="registry identity").get("path"),
        candidate_id=None if budget is None else _mapping(budget, label="budget identity").get("candidate_id"),
        resume=False,
        export_final_step=True,
        soft_only_eval=True,
    )
    try:
        return canonical_trainer_argv(namespace, resume=resume)
    except (TypeError, ValueError) as error:
        raise ValueError("resolved config cannot reproduce canonical trainer argv") from error


def _paired_scientific_identity(config: Mapping[str, object]) -> dict[str, object]:
    """Remove only the fields that define which ablation arm is enabled."""

    shared = dict(config)
    for name in (
        "profile",
        "step_config",
        "budget_identity",
        "unit_coefficient_identity",
        "candidate_registry_identity",
    ):
        shared.pop(name, None)
    run_identity = dict(
        _mapping(shared.get("run_identity"), label="paired run identity")
    )
    run_identity.pop("run_id", None)
    run_identity.pop("run_dir", None)
    shared["run_identity"] = run_identity
    return shared


def _validate_manifest(root: Path) -> tuple[dict[str, object], object, list[dict[str, object]]]:
    manifest_path = root / "manifests" / "ablation10.json"
    snapshot = snapshot_regular_file(manifest_path, root=root)
    payload = _decode_canonical_json(snapshot, label="ablation10 manifest")
    _exact_keys(payload, _MANIFEST_KEYS, label="ablation10 manifest")
    if payload["schema"] != "pcraft_soft_loss_v4_ablation10_v1":
        raise ValueError("ablation10 manifest schema mismatch")
    semantic = payload["semantic_sha256"]
    if type(semantic) is not str or len(semantic) != 64:
        raise ValueError("ablation10 semantic SHA is malformed")
    body = dict(payload)
    del body["semantic_sha256"]
    if canonical_hash(body) != semantic:
        raise ValueError("ablation10 semantic SHA mismatch")
    _same_canonical(payload["constants"], _CONSTANTS, label="ablation10 constants")
    _same_canonical(payload["decision_policy"], _DECISION_POLICY, label="ablation10 decision policy")
    _same_canonical(
        payload["candidate_row"],
        {
            "candidate_id": "r08_r08",
            "order_index": 1,
            "dic_max_ratio": 0.08,
            "dts_max_ratio": 0.08,
        },
        label="frozen r08_r08 candidate row",
    )

    matrix = payload["matrix"]
    if type(matrix) is not list or len(matrix) != 40:
        raise ValueError("ablation10 manifest must contain exactly forty cells")
    normalized: list[dict[str, object]] = []
    resolved_devices: list[str] = []
    paired_identities: dict[str, list[dict[str, object]]] = {
        task: [] for task in _TASKS
    }
    for index, ((task, profile), raw) in enumerate(zip(_EXPECTED_MATRIX, matrix)):
        row = _mapping(raw, label=f"ablation10 matrix[{index}]")
        _exact_keys(row, _MATRIX_KEYS, label=f"ablation10 matrix[{index}]")
        run_id = f"runs/ablation10/{task}/{profile}"
        run_dir = root.joinpath(*run_id.split("/"))
        if (
            row["task_id"] != task
            or row["profile"] != profile
            or row["run_id"] != run_id
            or row["run_dir"] != str(run_dir)
        ):
            raise ValueError("ablation10 matrix order or run identity drifted")
        config = _mapping(row["resolved_config"], label=f"resolved config {run_id}")
        try:
            validated_config = validate_resolved_config(dict(config))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{run_id} resolved config schema is invalid") from error
        if canonical_json_bytes(validated_config) != canonical_json_bytes(config):
            raise ValueError(f"{run_id} resolved config normalization drifted")
        device = config.get("device")
        if type(device) is not str:
            raise ValueError(f"{run_id} resolved device is malformed")
        resolved_devices.append(device)
        paired_identities[task].append(_paired_scientific_identity(config))
        config_hash = row["config_hash"]
        if type(config_hash) is not str or canonical_hash(config) != config_hash:
            raise ValueError("ablation10 resolved config hash mismatch")
        run_identity = _mapping(config.get("run_identity"), label="resolved run identity")
        config_profile = _mapping(config.get("profile"), label="resolved profile")
        output_identity = _mapping(config.get("output_root_identity"), label="output root identity")
        if (
            run_identity.get("run_id") != run_id
            or run_identity.get("run_dir") != str(run_dir)
            or run_identity.get("stage") != "ablation10_128"
            or run_identity.get("task_id") != task
            or run_identity.get("replicate_role") != "primary"
            or run_identity.get("replicate_index") != 0
            or config_profile.get("name") != profile
            or output_identity.get("path") != str(root)
            or config.get("steps") != 2000
            or config.get("learning_rate") != 0.007
            or config.get("resolution") != 128
            or config.get("export_resolution") != 128
            or config.get("seed") != 17
        ):
            raise ValueError("ablation10 resolved scientific identity drifted")
        _same_canonical(config.get("suite_manifest"), payload["suite_identity"], label="suite identity")
        _same_canonical(config.get("specification"), payload["specification_identity"], label="specification identity")
        if profile == "l12_bal":
            if any(
                config.get(name) is not None
                for name in (
                    "unit_coefficient_identity",
                    "candidate_registry_identity",
                    "budget_identity",
                )
            ):
                raise ValueError("baseline cell carries an auxiliary dependency")
        else:
            _same_canonical(config.get("unit_coefficient_identity"), payload["unit_coefficient_identity"], label="unit identity")
            _same_canonical(config.get("candidate_registry_identity"), payload["candidate_registry_identity"], label="candidate registry identity")
            _same_canonical(config.get("budget_identity"), payload["candidate_row"], label="candidate row")
        for key, expected in (
            ("fresh_argv", _argv_from_config(config, resume=False)),
            ("resume_argv", _argv_from_config(config, resume=True)),
        ):
            argv = row[key]
            if type(argv) is not list or any(type(item) is not str for item in argv):
                raise ValueError(f"{run_id} {key} is malformed")
            if argv != list(expected):
                raise ValueError(f"{run_id} {key} differs from canonical Task 3 argv")
        normalized.append(dict(row))
    if len({(row["task_id"], row["profile"]) for row in normalized}) != 40:
        raise ValueError("ablation10 matrix contains a duplicate cell")
    if len(set(resolved_devices)) != 1:
        raise ValueError("ablation10 paired matrix uses more than one device")
    for task, identities in paired_identities.items():
        if len(identities) != 4:
            raise ValueError(f"{task} paired scientific matrix is incomplete")
        reference = canonical_json_bytes(identities[0])
        if any(canonical_json_bytes(identity) != reference for identity in identities[1:]):
            raise ValueError(
                f"{task} paired non-arm scientific/input identity differs across arms"
            )
    return payload, snapshot, normalized


def _same_number(value: object, expected: float, *, label: str) -> float:
    scalar = _finite(value, label=label)
    if scalar != float(expected):
        raise ValueError(f"{label} disagrees with the six-view evidence")
    return scalar


def _parse_view(value: object, *, label: str) -> dict[str, object]:
    row = _mapping(value, label=label)
    _exact_keys(row, _VIEW_KEYS, label=label)
    counts = {
        name: _exact_int(row[name], label=f"{label}.{name}", nonnegative=True)
        for name in ("tp", "fp", "fn", "tn")
    }
    tp, fp, fn, tn = (counts[name] for name in ("tp", "fp", "fn", "tn"))
    iou_denominator = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    expected_iou = 1.0 if iou_denominator == 0 else tp / iou_denominator
    expected_dice = 1.0 if dice_denominator == 0 else (2 * tp) / dice_denominator
    _same_number(row["iou"], expected_iou, label=f"{label}.iou")
    _same_number(row["dice"], expected_dice, label=f"{label}.dice")
    background_invalid = fp + tn == 0
    foreground_invalid = tp + fn == 0
    for name, expected in (
        ("invalid_background_denominator", background_invalid),
        ("invalid_foreground_denominator", foreground_invalid),
    ):
        if _exact_bool(row[name], label=f"{label}.{name}") is not expected:
            raise ValueError(f"{label}.{name} disagrees with counts")
    if background_invalid:
        if row["fpr"] is not None:
            raise ValueError(f"{label}.fpr must be null")
    else:
        _same_number(row["fpr"], fp / (fp + tn), label=f"{label}.fpr")
    if foreground_invalid:
        if row["recall"] is not None:
            raise ValueError(f"{label}.recall must be null")
    else:
        _same_number(row["recall"], tp / (tp + fn), label=f"{label}.recall")
    for name in ("boundary_f1_2px", "mscd_px", "p95_symmetric_distance_px"):
        _finite(row[name], label=f"{label}.{name}", nonnegative=True)
    for name in _METRIC_FLAGS:
        _exact_bool(row[name], label=f"{label}.{name}")
    return dict(row)


def _parse_primary_threshold(value: object) -> dict[str, object]:
    report = _mapping(value, label="metrics threshold 0.05")
    _exact_keys(report, {"per_view", "aggregate"}, label="metrics threshold 0.05")
    raw_views = report["per_view"]
    if type(raw_views) is not list or len(raw_views) != 6:
        raise ValueError("metrics threshold 0.05 must contain exactly six views")
    views = [
        _parse_view(item, label=f"metrics threshold 0.05 view[{index}]")
        for index, item in enumerate(raw_views)
    ]
    aggregate = _mapping(report["aggregate"], label="metrics threshold 0.05 aggregate")
    _exact_keys(aggregate, _AGGREGATE_KEYS, label="metrics threshold 0.05 aggregate")
    for name in ("tp", "fp", "fn", "tn"):
        total = sum(int(view[name]) for view in views)
        if _exact_int(aggregate[name], label=f"aggregate.{name}", nonnegative=True) != total:
            raise ValueError(f"aggregate.{name} differs from six-view counts")
    for source, destination in (
        ("iou", "mean_iou"),
        ("dice", "mean_dice"),
        ("fpr", "mean_fpr"),
        ("recall", "mean_recall"),
        ("boundary_f1_2px", "mean_boundary_f1_2px"),
        ("mscd_px", "mean_mscd_px"),
        ("p95_symmetric_distance_px", "mean_p95_symmetric_distance_px"),
    ):
        if any(view[source] is None for view in views):
            if aggregate[destination] is not None:
                raise ValueError(f"aggregate.{destination} must be null")
            continue
        expected = float(sum(float(view[source]) for view in views) / 6.0)
        _same_number(aggregate[destination], expected, label=f"aggregate.{destination}")
    for name in _METRIC_FLAGS:
        expected = any(bool(view[name]) for view in views)
        if _exact_bool(aggregate[name], label=f"aggregate.{name}") is not expected:
            raise ValueError(f"aggregate.{name} differs from six-view flags")
    expected_invalid = any(
        bool(aggregate[name])
        for name in (
            "invalid_background_denominator",
            "invalid_foreground_denominator",
            "invalid_predicted_boundary",
            "invalid_target_boundary",
        )
    )
    if _exact_bool(aggregate["invalid_task_metric"], label="aggregate.invalid_task_metric") is not expected_invalid:
        raise ValueError("aggregate invalid-task flag differs from six-view flags")
    for name in ("mean_iou", "mean_dice", "mean_recall", "mean_fpr"):
        _finite(aggregate[name], label=f"aggregate.{name}", unit=True)
    return {"per_view": views, "aggregate": dict(aggregate)}


def _parse_metrics(payload: object, *, config_hash: str) -> dict[str, float]:
    metrics = _mapping(payload, label="metrics")
    _exact_keys(metrics, _METRICS_KEYS, label="metrics")
    if metrics["schema"] != "pcraft_soft_loss_v4_metrics_v1" or metrics["config_hash"] != config_hash:
        raise ValueError("metrics schema or config identity mismatch")
    thresholds = _mapping(metrics["thresholds"], label="metrics thresholds")
    _exact_keys(thresholds, {"0.04", "0.05", "0.06"}, label="metrics thresholds")
    for key in ("0.04", "0.05", "0.06"):
        _mapping(thresholds[key], label=f"metrics threshold {key}")
    primary = _parse_primary_threshold(thresholds["0.05"])
    aggregate = primary["aggregate"]
    iou_map = _mapping(metrics["task_iou_by_threshold"], label="task IoU map")
    _exact_keys(iou_map, {"0.04", "0.05", "0.06"}, label="task IoU map")
    _same_number(iou_map["0.05"], float(aggregate["mean_iou"]), label="task IoU 0.05")
    return {
        "mean_iou": float(aggregate["mean_iou"]),
        "mean_dice": float(aggregate["mean_dice"]),
        "mean_recall": float(aggregate["mean_recall"]),
        "mean_fpr": float(aggregate["mean_fpr"]),
    }


def _parse_audit(payload: object, *, config_hash: str, profile: str) -> dict[str, object]:
    audit = _mapping(payload, label="gradient audit")
    _exact_keys(audit, _AUDIT_KEYS, label="gradient audit")
    if (
        audit["schema"] != "pcraft_soft_loss_v4_gradient_audit_v1"
        or audit["config_hash"] != config_hash
        or audit["profile"] != profile
        or audit["total_steps"] != 2000
    ):
        raise ValueError("gradient audit identity mismatch")
    eligible = _exact_int(audit["eligible_slots"], label="eligible slots", nonnegative=True)
    samples = audit["samples"]
    if type(samples) is not list or len(samples) != eligible:
        raise ValueError("gradient audit samples differ from eligible slots")
    active_components: set[str] = set()
    active_slots = 0
    for index, raw in enumerate(samples):
        sample = _mapping(raw, label=f"gradient sample[{index}]")
        component = sample.get("component")
        if component not in {"dic", "dts"}:
            raise ValueError("gradient sample component is malformed")
        norm = _finite(
            sample.get("final_auxiliary_gradient_norm"),
            label=f"gradient sample[{index}] final norm",
            nonnegative=True,
        )
        if norm > 0.0:
            active_slots += 1
            active_components.add(component)
    if profile == "l12_bal":
        if eligible != 0 or samples or audit["engineering_gate"] is not None:
            raise ValueError("baseline gradient audit carries auxiliary evidence")
        if audit["redundancy_gate"] is not None:
            raise ValueError("baseline gradient audit carries redundancy evidence")
        return {
            "eligible_slots": 0,
            "active_slots": 0,
            "active_components": [],
            "fully_projected_zero_slots": 0,
            "fully_projected_zero_fraction": None,
            "engineering_status": None,
        }
    gate = _mapping(audit["engineering_gate"], label="engineering gate")
    _exact_keys(gate, _ENGINEERING_KEYS, label="engineering gate")
    status = gate["status"]
    if status not in {"PASS", "STOP"}:
        raise ValueError("engineering gate status is malformed")
    gate_eligible = _exact_int(gate["eligible_slots"], label="engineering eligible slots", nonnegative=True)
    zero_slots = _exact_int(
        gate["fully_projected_zero_slots"], label="fully projected zero slots", nonnegative=True
    )
    fraction = _finite(
        gate["fully_projected_zero_fraction"], label="fully projected zero fraction", unit=True
    )
    if gate_eligible != eligible or zero_slots > eligible:
        raise ValueError("engineering gate slot counts disagree with audit")
    expected_fraction = zero_slots / eligible if eligible else 0.0
    if fraction != expected_fraction:
        raise ValueError("engineering projected-zero fraction disagrees with counts")
    failure_codes = gate["failure_codes"]
    if type(failure_codes) is not list or any(type(item) is not str for item in failure_codes):
        raise ValueError("engineering failure codes are malformed")
    required = _DECISION_POLICY["engineering"]["required_components"][profile]
    return {
        "eligible_slots": eligible,
        "active_slots": active_slots,
        "active_components": sorted(active_components, key=("dic", "dts").index),
        "required_components": list(required),
        "fully_projected_zero_slots": zero_slots,
        "fully_projected_zero_fraction": fraction,
        "engineering_status": status,
    }


def _authenticated_identity(value: object) -> tuple[object, ...]:
    try:
        return (
            getattr(value, "run_dir"),
            canonical_json_bytes(getattr(value, "config")),
            getattr(value, "config_hash"),
            getattr(value, "status_sha256"),
            getattr(value, "artifact_manifest_sha256"),
            canonical_json_bytes(getattr(value, "artifacts_sha256")),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Task 3 completed-run authentication result is malformed") from error


def _collect_rows(
    root: Path, matrix: Sequence[Mapping[str, object]], manifest_snapshot: object
) -> tuple[list[dict[str, object]], list[tuple[object, object, object, Mapping[str, object], Path]]]:
    rows: list[dict[str, object]] = []
    evidence: list[tuple[object, object, object, Mapping[str, object], Path]] = []
    for registration in matrix:
        run_dir = Path(registration["run_dir"])
        config = _mapping(registration["resolved_config"], label="registered resolved config")
        authenticated = validate_completed_run(run_dir, expected_config=config)
        if (
            getattr(authenticated, "run_dir", None) != str(run_dir)
            or getattr(authenticated, "config_hash", None) != registration["config_hash"]
            or canonical_json_bytes(getattr(authenticated, "config", None))
            != canonical_json_bytes(config)
        ):
            raise ValueError("Task 3 authentication differs from the registered cell")
        artifacts = getattr(authenticated, "artifacts_sha256", None)
        if type(artifacts) is not dict:
            raise ValueError("authenticated artifact hash mapping is malformed")
        metrics_snapshot = snapshot_regular_file(run_dir / "metrics.json", root=run_dir)
        audit_snapshot = snapshot_regular_file(run_dir / "gradient_audit.json", root=run_dir)
        if (
            artifacts.get("metrics.json") != metrics_snapshot.sha256
            or artifacts.get("gradient_audit.json") != audit_snapshot.sha256
        ):
            raise ValueError("authenticated metrics or gradient artifact hash mismatch")
        metrics_payload = _decode_canonical_json(metrics_snapshot, label="metrics")
        audit_payload = _decode_canonical_json(audit_snapshot, label="gradient audit")
        metrics = _parse_metrics(metrics_payload, config_hash=str(registration["config_hash"]))
        engineering = _parse_audit(
            audit_payload,
            config_hash=str(registration["config_hash"]),
            profile=str(registration["profile"]),
        )
        revalidate_snapshot(metrics_snapshot, root=run_dir)
        revalidate_snapshot(audit_snapshot, root=run_dir)
        rows.append(
            {
                "task_id": registration["task_id"],
                "profile": registration["profile"],
                "run_id": registration["run_id"],
                "config_hash": registration["config_hash"],
                "metrics_sha256": metrics_snapshot.sha256,
                "gradient_audit_sha256": audit_snapshot.sha256,
                **metrics,
                "eligible_slots": engineering["eligible_slots"],
                "active_slots": engineering["active_slots"],
                "active_components": engineering["active_components"],
                "required_components": engineering.get("required_components", []),
                "fully_projected_zero_slots": engineering["fully_projected_zero_slots"],
                "fully_projected_zero_fraction": engineering["fully_projected_zero_fraction"],
                "engineering_status": engineering["engineering_status"],
            }
        )
        evidence.append((authenticated, metrics_snapshot, audit_snapshot, config, run_dir))
    return rows, evidence


def _component_gate(
    iou_deltas: Sequence[float], dice_deltas: Sequence[float], *, adjusted_p: float
) -> dict[str, object]:
    if len(iou_deltas) != 10 or len(dice_deltas) != 10:
        raise ValueError("component gate requires exactly ten task deltas")
    iou = [_finite(value, label=f"IoU delta[{index}]") for index, value in enumerate(iou_deltas)]
    dice = [_finite(value, label=f"Dice delta[{index}]") for index, value in enumerate(dice_deltas)]
    pvalue = _finite(adjusted_p, label="adjusted p-value", unit=True)
    observed = {
        "mean_delta_iou": _ordered_mean(iou),
        "mean_delta_dice": _ordered_mean(dice),
        "positive_task_count": sum(value > 0.0 for value in iou),
        "min_task_delta_iou": min(iou),
        "holm_adjusted_pvalue": pvalue,
    }
    failures: list[str] = []
    if observed["mean_delta_iou"] < 0.01:
        failures.append("mean_delta_iou_min")
    if observed["mean_delta_dice"] < 0.005:
        failures.append("mean_delta_dice_min")
    if observed["positive_task_count"] < 8:
        failures.append("positive_task_count_min")
    if observed["min_task_delta_iou"] < -0.005:
        failures.append("task_delta_iou_min")
    if observed["holm_adjusted_pvalue"] >= 0.05:
        failures.append("adjusted_p_strict_max")
    below = observed["mean_delta_iou"] < 0.01 or observed["mean_delta_dice"] < 0.005
    passed = not failures
    return {
        "criteria": dict(_DECISION_POLICY["component_key"]),
        "observed": observed,
        "passed": passed,
        "below": below,
        "ambiguous": not passed and not below,
        "failures": failures,
    }


def _quality_gate(
    full_mean_iou: float,
    full_task_iou_min: float,
    full_base_mean_iou: float,
    full_base_mean_dice: float,
) -> dict[str, object]:
    observed = {
        "full_mean_iou": _finite(full_mean_iou, label="full mean IoU", unit=True),
        "full_task_iou_min": _finite(full_task_iou_min, label="minimum full task IoU", unit=True),
        "full_base_mean_iou": _finite(full_base_mean_iou, label="full/base mean IoU"),
        "full_base_mean_dice": _finite(full_base_mean_dice, label="full/base mean Dice"),
    }
    failures = [
        name
        for name, value, threshold in (
            ("full_mean_iou_min", observed["full_mean_iou"], 0.95),
            ("full_task_iou_min", observed["full_task_iou_min"], 0.94),
            ("full_base_mean_iou_min", observed["full_base_mean_iou"], -0.001),
            ("full_base_mean_dice_min", observed["full_base_mean_dice"], -0.001),
        )
        if value < threshold
    ]
    return {
        "criteria": dict(_DECISION_POLICY["overall_quality"]),
        "observed": observed,
        "passed": not failures,
        "failures": failures,
    }


def _safety_profile_gate(
    recall_deltas: Sequence[float], fpr_deltas: Sequence[float]
) -> dict[str, object]:
    if len(recall_deltas) != 10 or len(fpr_deltas) != 10:
        raise ValueError("safety gate requires exactly ten task deltas")
    recall = [_finite(value, label=f"recall delta[{index}]") for index, value in enumerate(recall_deltas)]
    fpr = [_finite(value, label=f"FPR delta[{index}]") for index, value in enumerate(fpr_deltas)]
    observed = {
        "min_task_recall_delta": min(recall),
        "mean_fpr_delta": _ordered_mean(fpr),
    }
    failures: list[str] = []
    if observed["min_task_recall_delta"] < -0.005:
        failures.append("task_recall_delta_min")
    if observed["mean_fpr_delta"] > 0.001:
        failures.append("mean_fpr_delta_max")
    return {"observed": observed, "passed": not failures, "failures": failures}


def _engineering_row_gate(
    *,
    eligible_slots: int,
    active_slots: int,
    status: object,
    projected_zero_fraction: object,
    active_components: Sequence[str] | None = None,
    required_components: Sequence[str] | None = None,
) -> dict[str, object]:
    eligible = _exact_int(eligible_slots, label="eligible slots", nonnegative=True)
    active = _exact_int(active_slots, label="active slots", nonnegative=True)
    fraction = _finite(projected_zero_fraction, label="projected-zero fraction", unit=True)
    observed_components = set(active_components or (() if active == 0 else ("registered",)))
    required = set(required_components or (() if active == 0 else ("registered",)))
    failures: list[str] = []
    if eligible < 1:
        failures.append("auxiliary_eligible_slots_min")
    if status != "PASS":
        failures.append("engineering_gate_status")
    if fraction > 0.10:
        failures.append("projected_zero_fraction_max")
    if active < 1 or not required.issubset(observed_components):
        failures.append("each_enabled_component_has_positive_final_norm")
    return {
        "observed": {
            "eligible_slots": eligible,
            "active_slots": active,
            "status": status,
            "projected_zero_fraction": fraction,
            "active_components": sorted(observed_components),
            "required_components": sorted(required),
        },
        "passed": not failures,
        "failures": failures,
    }


def _validate_decision_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    if type(rows) not in (list, tuple) or len(rows) != 40:
        raise ValueError("decision requires the exact forty-row matrix")
    if [(row.get("task_id"), row.get("profile")) for row in rows] != list(_EXPECTED_MATRIX):
        raise ValueError("decision rows differ from the registered matrix")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("decision row must be a mapping")
        for name in ("mean_iou", "mean_dice", "mean_recall", "mean_fpr"):
            _finite(row.get(name), label=f"row[{index}].{name}", unit=True)
    return list(rows)


def _decision_from_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    checked = _validate_decision_rows(rows)
    by_task: dict[str, dict[str, Mapping[str, object]]] = {
        task: {row["profile"]: row for row in checked if row["task_id"] == task}
        for task in _TASKS
    }
    contrasts: list[dict[str, object]] = []
    dic_iou: list[float] = []
    dic_dice: list[float] = []
    dts_iou: list[float] = []
    dts_dice: list[float] = []
    full_base_iou: list[float] = []
    full_base_dice: list[float] = []
    for task in _TASKS:
        group = by_task[task]
        full = group["v4_full"]
        dts = group["dts_only"]
        dic = group["dic_only"]
        base = group["l12_bal"]
        values = {
            "dic_delta_iou": float(full["mean_iou"] - dts["mean_iou"]),
            "dic_delta_dice": float(full["mean_dice"] - dts["mean_dice"]),
            "dts_delta_iou": float(full["mean_iou"] - dic["mean_iou"]),
            "dts_delta_dice": float(full["mean_dice"] - dic["mean_dice"]),
            "full_base_delta_iou": float(full["mean_iou"] - base["mean_iou"]),
            "full_base_delta_dice": float(full["mean_dice"] - base["mean_dice"]),
        }
        contrasts.append({"task_id": task, **values})
        dic_iou.append(values["dic_delta_iou"])
        dic_dice.append(values["dic_delta_dice"])
        dts_iou.append(values["dts_delta_iou"])
        dts_dice.append(values["dts_delta_dice"])
        full_base_iou.append(values["full_base_delta_iou"])
        full_base_dice.append(values["full_base_delta_dice"])
    raw_p = {
        "dic": exact_sign_flip_pvalue(dic_iou),
        "dts": exact_sign_flip_pvalue(dts_iou),
    }
    adjusted = holm_adjust_two(raw_p["dic"], raw_p["dts"])
    dic_gate = _component_gate(dic_iou, dic_dice, adjusted_p=adjusted["dic"])
    dts_gate = _component_gate(dts_iou, dts_dice, adjusted_p=adjusted["dts"])
    full_ious = [float(by_task[task]["v4_full"]["mean_iou"]) for task in _TASKS]
    quality = _quality_gate(
        _ordered_mean(full_ious),
        min(full_ious),
        _ordered_mean(full_base_iou),
        _ordered_mean(full_base_dice),
    )
    safety_profiles: dict[str, object] = {}
    for profile in ("v4_full", "dic_only", "dts_only"):
        recall_deltas = [
            float(by_task[task][profile]["mean_recall"] - by_task[task]["l12_bal"]["mean_recall"])
            for task in _TASKS
        ]
        fpr_deltas = [
            float(by_task[task][profile]["mean_fpr"] - by_task[task]["l12_bal"]["mean_fpr"])
            for task in _TASKS
        ]
        safety_profiles[profile] = _safety_profile_gate(recall_deltas, fpr_deltas)
    safety = {
        "criteria": dict(_DECISION_POLICY["safety"]),
        "profiles": safety_profiles,
        "passed": all(bool(gate["passed"]) for gate in safety_profiles.values()),
        "failures": [
            f"{profile}:{failure}"
            for profile, gate in safety_profiles.items()
            for failure in gate["failures"]
        ],
    }
    engineering_runs: list[dict[str, object]] = []
    for row in checked:
        if row["profile"] == "l12_bal":
            continue
        gate = _engineering_row_gate(
            eligible_slots=row.get("eligible_slots"),
            active_slots=row.get("active_slots"),
            status=row.get("engineering_status"),
            projected_zero_fraction=row.get("fully_projected_zero_fraction"),
            active_components=row.get("active_components"),
            required_components=row.get("required_components"),
        )
        engineering_runs.append({"run_id": row["run_id"], **gate})
    engineering = {
        "criteria": dict(_DECISION_POLICY["engineering"]),
        "runs": engineering_runs,
        "passed": all(bool(item["passed"]) for item in engineering_runs),
        "failures": [
            f"{item['run_id']}:{failure}"
            for item in engineering_runs
            for failure in item["failures"]
        ],
    }
    negative_component_mean = any(
        gate["observed"][name] < 0.0
        for gate in (dic_gate, dts_gate)
        for name in ("mean_delta_iou", "mean_delta_dice")
    )
    if not engineering["passed"]:
        label = "INCONCLUSIVE"
    elif not quality["passed"] or not safety["passed"] or negative_component_mean:
        label = "HARMFUL_OR_ANTAGONISTIC"
    elif dic_gate["ambiguous"] or dts_gate["ambiguous"]:
        label = "INCONCLUSIVE"
    elif dic_gate["passed"] and dts_gate["passed"]:
        label = "BOTH_KEY"
    elif dic_gate["passed"] and dts_gate["below"]:
        label = "DIC_KEY"
    elif dts_gate["passed"] and dic_gate["below"]:
        label = "DTS_KEY"
    elif dic_gate["below"] and dts_gate["below"]:
        label = "NEITHER_KEY"
    else:  # Defensive: the registered state machine is intended to be exhaustive.
        raise ValueError("complete decision evidence did not match a registered label")
    return {
        "task_contrasts": contrasts,
        "p_values": {
            "dic": {"raw": raw_p["dic"], "holm_adjusted": adjusted["dic"]},
            "dts": {"raw": raw_p["dts"], "holm_adjusted": adjusted["dts"]},
        },
        "gates": {
            "dic": dic_gate,
            "dts": dts_gate,
            "quality": quality,
            "safety": safety,
            "engineering": engineering,
        },
        "label": label,
        "supports_both_crucial": label == "BOTH_KEY",
    }


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name) for name in _CSV_FIELDS})
    return buffer.getvalue().encode("utf-8")


def _safe_analysis_directory(root: Path, *, create: bool) -> Path:
    directory = root / "analysis"
    if not os.path.lexists(directory):
        if not create:
            raise FileNotFoundError(directory)
        directory.mkdir()
    metadata = directory.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("analysis path must be a canonical directory")
    if directory.resolve(strict=True) != directory:
        raise ValueError("analysis directory contains a symlink component")
    return directory


def _existing_bytes(path: Path, *, root: Path) -> bytes | None:
    if not os.path.lexists(path):
        return None
    return snapshot_regular_file(path, root=root).payload


def _publish_bytes_once(path: Path, payload: bytes, *, root: Path) -> None:
    existing = _existing_bytes(path, root=root)
    if existing is not None:
        if existing != payload:
            raise FileExistsError(f"immutable analysis differs: {path}")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def analyze_ablation_manifest(output_root: Path, *, verify: bool = False) -> dict[str, object]:
    """Authenticate all forty runs, then create or byte-verify the frozen analysis."""

    if type(verify) is not bool:
        raise ValueError("verify must be an exact bool")
    root = _canonical_root(Path(output_root))
    manifest, manifest_snapshot, matrix = _validate_manifest(root)
    rows, evidence = _collect_rows(root, matrix, manifest_snapshot)
    decision = _decision_from_rows(rows)
    rows_bytes = _csv_bytes(rows)
    report: dict[str, object] = {
        "schema": "pcraft_soft_loss_v4_ablation10_report_v1",
        "manifest_identity": {
            "path": str(root / "manifests" / "ablation10.json"),
            "sha256": manifest_snapshot.sha256,
            "semantic_sha256": manifest["semantic_sha256"],
        },
        "decision_policy": manifest["decision_policy"],
        "row_count": 40,
        "rows_sha256": hashlib.sha256(rows_bytes).hexdigest(),
        **decision,
    }
    report["semantic_sha256"] = canonical_hash(report)
    report_bytes = canonical_json_bytes(report) + b"\n"

    revalidate_snapshot(manifest_snapshot, root=root)
    for authenticated, metrics_snapshot, audit_snapshot, config, run_dir in evidence:
        confirmed = validate_completed_run(run_dir, expected_config=config)
        if _authenticated_identity(confirmed) != _authenticated_identity(authenticated):
            raise ValueError("completed-run identity changed before analysis publication")
        revalidate_snapshot(metrics_snapshot, root=run_dir)
        revalidate_snapshot(audit_snapshot, root=run_dir)
    revalidate_snapshot(manifest_snapshot, root=root)

    analysis_dir = _safe_analysis_directory(root, create=not verify)
    rows_path = analysis_dir / "ablation10_rows.csv"
    report_path = analysis_dir / "ablation10_report.json"
    if verify:
        if _existing_bytes(rows_path, root=analysis_dir) != rows_bytes:
            raise ValueError("existing ablation10 rows differ from recomputation")
        if _existing_bytes(report_path, root=analysis_dir) != report_bytes:
            raise ValueError("existing ablation10 report differs from recomputation")
        return report

    for path, payload in ((rows_path, rows_bytes), (report_path, report_bytes)):
        existing = _existing_bytes(path, root=analysis_dir)
        if existing is not None and existing != payload:
            raise FileExistsError(f"immutable analysis differs: {path}")
    _publish_bytes_once(rows_path, rows_bytes, root=analysis_dir)
    _publish_bytes_once(report_path, report_bytes, root=analysis_dir)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze the frozen Soft Loss v4 ablation10 matrix",
        allow_abbrev=False,
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze_ablation_manifest(Path(args.output_root), verify=args.verify)
    print(f"ablation10: {report['label']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
