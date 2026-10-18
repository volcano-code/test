from __future__ import annotations

from copy import deepcopy
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from fractions import Fraction
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pytest
import torch

import experiments.run_soft_loss_v3 as runner_module
import experiments.soft_loss_v3_registry as registry_module
import experiments.summarize_soft_loss_v3 as summary_module
from experiments.summarize_soft_loss_v3 import (
    _cuda_integration_gate,
    build_summary_parser,
    collect_stage_rows,
    development_factorial_gate,
    formal_primary_completion_gate,
    gradient_budget_gate,
    repeat_tolerance_gate,
    resource_gate,
    summary_main,
    write_gate_authorization,
)
from shadowforge_pipeline.soft_voxel_v3.diagnostics import ComponentGradientSample
from shadowforge_pipeline.soft_voxel_v3.provenance import canonical_hash


PRIMARY = ("l12_bal", "dic_only", "dts_only", "v3_full")
DEVELOPMENT = ("arcane_casters", "blade_heroes", "stage_dancers")
FORMAL = tuple(f"connected_{index:02d}" for index in range(1, 11))
ROW_KEYS = {
    "schema", "stage", "run_id", "path", "task", "variant",
    "replicate_role", "replicate_index", "inference_eligible", "resolution",
    "steps", "scientific_config_hash", "expected_config_hash", "run_spec",
    "stage_manifest", "formal_campaign", "status", "artifact_manifest",
    "metrics", "gradient_audit", "resource_metrics", "history",
    "parameter_change",
}

GATE_FAILURE_VOCABULARY = {
    "gradient_budget": {
        "input_schema_invalid", "probe_matrix_invalid",
        "eligible_component_inactive", "component_ratio_out_of_range",
        "combined_ratio_median_exceeded", "combined_ratio_p95_exceeded",
        "adverse_cosine_fraction_exceeded", "endpoint_diagnostic_invalid",
    },
    "development_factorial": {
        "input_schema_invalid", "factorial_matrix_invalid",
        "mscd_denominator_invalid", "dic_conditional_reversed",
        "dts_conditional_reversed", "dic_threshold_missed",
        "dts_threshold_missed", "iou_noninferiority_failed",
        "recall_noninferiority_failed", "fpr_guard_failed",
        "threshold_range_failed", "metric_invalid",
    },
    "cuda_integration": {
        "input_schema_invalid", "integration_matrix_invalid",
        "integration_branch_inactive", "integration_nonfinite",
        "soft_only_violation",
    },
    "resource_feasibility": {
        "input_schema_invalid", "resource_matrix_invalid",
        "physical_memory_invalid", "resource_summary_mismatch",
        "peak_memory_exceeded", "baseline_p95_nonpositive",
        "runtime_ratio_exceeded", "nonfinite_value",
    },
    "repeat_tolerance": {
        "input_schema_invalid", "repeat_pair_invalid",
        "repeat_mscd_tolerance_exceeded",
        "repeat_worst_iou_tolerance_exceeded", "metric_invalid",
    },
}


def _sha(char: str = "a") -> str:
    return char * 64


def _identity(path: str = "/tmp/item", char: str = "a", *, semantic: bool = False) -> dict[str, object]:
    result: dict[str, object] = {"path": path, "sha256": _sha(char)}
    if semantic:
        result["semantic_sha256"] = _sha(chr(ord(char) + 1))
    return result


def _counts_for_metrics(iou: float, recall: float, fpr: float) -> tuple[int, int, int, int]:
    iou_q = Fraction(str(iou))
    recall_q = Fraction(str(recall))
    fpr_q = Fraction(str(fpr))
    if not (0 < iou_q <= 1 and 0 < recall_q <= 1 and 0 < fpr_q < 1):
        raise ValueError("fixture ratios must have positive denominators")
    fn_factor = (1 - recall_q) / recall_q
    fp_factor = 1 / iou_q - 1 - fn_factor
    if fp_factor <= 0:
        raise ValueError("fixture IoU/recall combination has no positive FP count")
    tn_factor = fp_factor * (1 - fpr_q) / fpr_q
    denominator = math.lcm(
        fn_factor.denominator,
        fp_factor.denominator,
        tn_factor.denominator,
    )
    tp = denominator
    fn = int(fn_factor * denominator)
    fp = int(fp_factor * denominator)
    tn = int(tn_factor * denominator)
    return tp, fp, fn, tn


def _view_record(*, iou: float, recall: float = 0.9, fpr: float = 0.01, mscd: float = 1.0) -> dict[str, object]:
    tp, fp, fn, tn = _counts_for_metrics(iou, recall, fpr)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": tp / (tp + fp + fn),
        "dice": (2 * tp) / (2 * tp + fp + fn),
        "fpr": fp / (fp + tn),
        "recall": tp / (tp + fn),
        "boundary_f1_2px": 0.8,
        "mscd_px": mscd,
        "p95_symmetric_distance_px": 2.0,
        "invalid_iou_denominator": False,
        "invalid_dice_denominator": False,
        "invalid_background_denominator": False,
        "invalid_foreground_denominator": False,
        "invalid_predicted_boundary": False,
        "invalid_target_boundary": False,
    }


def _aggregate(*, iou: float, recall: float, fpr: float, mscd: float) -> dict[str, object]:
    view = _view_record(iou=iou, recall=recall, fpr=fpr, mscd=mscd)
    return {
        "tp": 6 * int(view["tp"]),
        "fp": 6 * int(view["fp"]),
        "fn": 6 * int(view["fn"]),
        "tn": 6 * int(view["tn"]),
        "mean_iou": view["iou"],
        "mean_dice": view["dice"],
        "mean_fpr": view["fpr"],
        "mean_recall": view["recall"],
        "mean_boundary_f1_2px": 0.8,
        "mean_mscd_px": mscd,
        "mean_p95_symmetric_distance_px": 2.0,
        "invalid_iou_denominator": False,
        "invalid_dice_denominator": False,
        "invalid_background_denominator": False,
        "invalid_foreground_denominator": False,
        "invalid_predicted_boundary": False,
        "invalid_target_boundary": False,
        "invalid_task_metric": False,
    }


def _metrics_payload(
    *,
    profile: str,
    steps: int,
    config_hash: str,
    iou: float,
    worst_iou: float,
    recall: float,
    fpr: float,
    mscd: float,
    threshold_range: float,
    final_loss: float = 1.0,
) -> dict[str, object]:
    task_iou = {"0.04": worst_iou, "0.05": iou, "0.06": worst_iou + threshold_range}
    thresholds: dict[str, object] = {}
    for threshold, threshold_iou in task_iou.items():
        views = [_view_record(iou=threshold_iou, recall=recall, fpr=fpr, mscd=mscd) for _ in range(6)]
        thresholds[threshold] = {
            "per_view": views,
            "aggregate": _aggregate(iou=threshold_iou, recall=recall, fpr=fpr, mscd=mscd),
        }
    return {
        "schema": "pcraft_soft_loss_v3_metrics_v1",
        "config_hash": config_hash,
        "profile": profile,
        "steps": steps,
        "final_step_exported": True,
        "final_loss": final_loss,
        "thresholds": thresholds,
        "task_iou_by_threshold": task_iou,
        "worst_threshold_iou": min(task_iou.values()),
        "threshold_iou_range": max(task_iou.values()) - min(task_iou.values()),
        "task_mscd_px": mscd,
        "task_p95_symmetric_distance_px": 2.0,
        "invalid_task_metric": False,
    }


def _history_row(update: int, steps: int, *, dic: bool, dts: bool) -> dict[str, object]:
    progress = update / steps
    dic_active = dic and progress > 0.65
    dts_active = dts and progress > 0.50
    endpoint = None if not dts_active else (0.04 if (update - 1) % 2 == 0 else 0.06)
    loss_base = 1.0
    loss_dic = 0.5 if dic_active else None
    loss_dts = 0.25 if dts_active else None
    dic_scale = 0.1 if dic_active else 0.0
    dts_scale = 0.2 if dts_active else 0.0
    effective = loss_base + (0.0 if loss_dic is None else loss_dic * dic_scale) + (0.0 if loss_dts is None else loss_dts * dts_scale)
    return {
        "global_step_before": update - 1,
        "completed_updates_after": update,
        "progress": progress,
        "dts_endpoint": endpoint,
        "loss_base": loss_base,
        "loss_dic": loss_dic,
        "loss_dts": loss_dts,
        "dic_scale": dic_scale,
        "dts_scale": dts_scale,
        "pre_clip_grad_norm": 1.0,
        "post_clip_grad_norm": 1.0,
        "effective_loss": effective,
    }


def _gradient_sample(update: int, *, dic: bool, dts: bool, endpoint: float | None) -> dict[str, object]:
    dts_map = {} if not dts else {"0.04": 1.0, "0.06": 1.0}
    selected_map = {} if endpoint is None else {"0.04": 0.0, "0.06": 0.0}
    invalid_map = {} if endpoint is None else {"0.04": False, "0.06": False}
    support_map = {} if not (dic and dts) else {"0.04": 0.5, "0.06": 0.5}
    support_invalid = {} if not (dic and dts) else {"0.04": False, "0.06": False}
    return {
        "completed_updates": update,
        "base_norm": 1.0,
        "dic_norm": 1.0 if dic else None,
        "dts_norms": dts_map,
        "base_dic_cosine": 0.0 if dic else None,
        "base_dic_cosine_invalid": False if dic else True,
        "base_dts_cosines": selected_map,
        "base_dts_cosine_invalid": invalid_map,
        "dic_dts_cosines": support_map,
        "dic_dts_cosine_invalid": support_invalid,
        "dts_pair_cosines": ({"0.04": {"0.06": 0.0}} if dts else {}),
        "dts_pair_cosine_invalid": ({"0.04": {"0.06": False}} if dts else {}),
        "dic_dts_support_ious": support_map,
        "dic_dts_support_iou_invalid": support_invalid,
        "weighted_dic_base_ratio": 0.15 if dic else None,
        "weighted_dic_base_ratio_invalid": False if dic else True,
        "weighted_dts_base_ratios": ({"0.04": 0.15, "0.06": 0.15} if dts else {}),
        "weighted_dts_base_ratio_invalid": ({"0.04": False, "0.06": False} if dts else {}),
        "selected_dts_tau": endpoint,
        "combined_aux_base_ratio": (0.30 if dic and dts else 0.15),
        "combined_aux_base_ratio_invalid": False,
        "base_combined_aux_cosine": 0.0,
        "base_combined_aux_cosine_invalid": False,
    }


def _collected_row(
    *,
    stage: str,
    task: str,
    variant: str,
    resolution: int,
    steps: int,
    mscd: float = 1.0,
    worst_iou: float = 0.7,
    iou: float = 0.8,
    recall: float = 0.9,
    fpr: float = 0.01,
    threshold_range: float = 0.02,
    role: str = "development",
    replicate_index: int = 0,
    inference_eligible: bool = False,
    step_times: list[float] | None = None,
) -> dict[str, object]:
    config_hash = _sha("c")
    use_dic = variant in {"dic_only", "v3_full"}
    use_dts = variant in {"dts_only", "v3_full"}
    history = [_history_row(update, steps, dic=use_dic, dts=use_dts) for update in range(1, steps + 1)]
    metrics = _metrics_payload(
        profile=variant,
        steps=steps,
        config_hash=config_hash,
        iou=iou,
        worst_iou=worst_iou,
        recall=recall,
        fpr=fpr,
        mscd=mscd,
        threshold_range=threshold_range,
        final_loss=float(history[-1]["effective_loss"]),
    )
    times = [0.01] * steps if step_times is None else step_times
    run_id = f"{stage}/steps_{steps}/{task}/{variant}/replicate_{replicate_index}/seed_17"
    profile_values = {
        "l12_bal": (0.35, 1.0, False, False, None),
        "dic_only": (0.35, 1.0, True, False, None),
        "dts_only": (0.35, 1.0, False, True, (0.04, 0.06)),
        "v3_full": (0.35, 1.0, True, True, (0.04, 0.06)),
    }[variant]
    inputs = tuple(
        {
            "axis": axis,
            "source_path": f"/tmp/source/{task}_{axis}.png",
            "resolved_path": f"/tmp/source/{task}_{axis}.png",
            "sha256": _sha(str(index + 4)),
        }
        for index, axis in enumerate("xyz")
    )
    spec = runner_module.RunSpec(
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
        output_path=Path(f"/tmp/root/{run_id}"),
        run_id=run_id,
        device="cuda:0",
        optimizer="Adam",
        learning_rate=0.007,
        initialization_logit=-4.0,
        gradient_clip=1.0,
        seed=17,
        l1_weight=profile_values[0],
        l2_weight=profile_values[1],
        use_dic=profile_values[2],
        use_dts=profile_values[3],
        dic_threshold=0.05,
        dts_thresholds=profile_values[4],
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
        frozen_lambdas={"lambda_dic": 0.25, "lambda_dts": 0.5},
        master_identity=_identity("/tmp/root/experiment_manifest.json", "7", semantic=True),
        suite_identity=_identity("/tmp/root/manifests/development_suite.json", "8", semantic=True),
        specification_identity=_identity("/tmp/spec.md", "9"),
        source_identity={"schema": "pcraft_soft_loss_v3_runner_source_v1", "bundle_sha256": _sha("a")},
        weight_identity={**_identity("/tmp/root/analysis/weights_frozen.json", "b", semantic=True), "status": "APPROVED_FROZEN", "lambdas": {"lambda_dic": 0.25, "lambda_dts": 0.5}},
        inference_eligible=inference_eligible,
    )
    run_spec = runner_module._run_spec_payload(spec)
    expected_namespace = runner_module._expected_artifact_namespace(spec)
    transaction_updates = [0, *range(100, steps + 1, 100)]
    if transaction_updates[-1] != steps:
        transaction_updates.append(steps)
    artifact_names = set(expected_namespace["top_level"])
    for update in transaction_updates:
        prefix = f"transactions/step_{update:08d}"
        artifact_names.update(f"{prefix}/{name}" for name in expected_namespace["transaction_files"])
    artifact_hashes = {name: _sha("d") for name in sorted(artifact_names)}
    row = {
        "schema": "pcraft_soft_loss_v3_collected_row_v1",
        "stage": stage,
        "run_id": run_id,
        "path": f"/tmp/root/{run_id}",
        "task": task,
        "variant": variant,
        "replicate_role": role,
        "replicate_index": replicate_index,
        "inference_eligible": inference_eligible,
        "resolution": resolution,
        "steps": steps,
        "scientific_config_hash": runner_module.scientific_config_hash(spec),
        "expected_config_hash": config_hash,
        "run_spec": run_spec,
        "stage_manifest": _identity("/tmp/root/stage.json", "d", semantic=True),
        "formal_campaign": None,
        "status": {"path": f"/tmp/root/{run_id}/STATUS.json", "sha256": _sha("e"), "payload": {"schema": "pcraft_soft_loss_v3_status_v1", "state": "completed", "config_hash": config_hash, "completed_updates": steps, "artifact_manifest_sha256": _sha("f")}},
        "artifact_manifest": {"path": f"/tmp/root/{run_id}/artifact_manifest.json", "sha256": _sha("f"), "payload": {"schema": "pcraft_soft_loss_v3_artifacts_v1", "config_hash": config_hash, "artifacts_sha256": artifact_hashes}},
        "metrics": {"path": f"/tmp/root/{run_id}/metrics.json", "sha256": _sha("1"), "payload": metrics},
        "gradient_audit": {"path": f"/tmp/root/{run_id}/gradient_audit.json", "sha256": _sha("2"), "payload": {"schema": "pcraft_soft_loss_v3_gradient_audit_v1", "config_hash": config_hash, "samples": [], "calibration_records": [], "active_invalid_flags": []}},
        "resource_metrics": {"path": f"/tmp/root/{run_id}/resource_metrics.json", "sha256": _sha("3"), "payload": {"schema": "pcraft_soft_loss_v3_resources_v1", "step_times": times, "peak_cuda_allocated_bytes": 100, "physical_device_memory_bytes": 1000, "all_updates": {"count": steps, "median_seconds": float(np.median(times)), "p95_seconds": float(np.percentile(times, 95))}, "updates_21_end": {"count": max(steps - 20, 0), "median_seconds": None if steps <= 20 else float(np.median(times[20:])), "p95_seconds": None if steps <= 20 else float(np.percentile(times[20:], 95))}}},
        "history": {"path": f"/tmp/root/{run_id}/history.jsonl", "sha256": _sha("4"), "rows": history},
        "parameter_change": {"reference_probability": float(np.float32(1 / (1 + np.exp(np.float32(4.0))))), "max_abs_delta": 0.01, "changed": True},
    }
    assert set(row) == ROW_KEYS
    return row


def _development_rows(*, stage: str = "confirm128", resolution: int = 128, steps: int = 2000) -> list[dict[str, object]]:
    rows = []
    for task in DEVELOPMENT:
        rows.extend(
            [
                _collected_row(stage=stage, task=task, variant="l12_bal", resolution=resolution, steps=steps, mscd=10.0, worst_iou=0.70, iou=0.80, recall=0.90, fpr=0.010, threshold_range=0.04),
                _collected_row(stage=stage, task=task, variant="dic_only", resolution=resolution, steps=steps, mscd=9.0, worst_iou=0.70, iou=0.80, recall=0.90, fpr=0.010, threshold_range=0.04),
                _collected_row(stage=stage, task=task, variant="dts_only", resolution=resolution, steps=steps, mscd=10.0, worst_iou=0.71, iou=0.80, recall=0.90, fpr=0.010, threshold_range=0.04),
                _collected_row(stage=stage, task=task, variant="v3_full", resolution=resolution, steps=steps, mscd=9.0, worst_iou=0.72, iou=0.80, recall=0.90, fpr=0.010, threshold_range=0.03),
            ]
        )
    return rows


def _replace_metrics(row: dict[str, object], **updates: float) -> None:
    current = row["metrics"]["payload"]
    primary = current["thresholds"]["0.05"]["aggregate"]
    values = {
        "iou": float(primary["mean_iou"]),
        "worst_iou": float(current["worst_threshold_iou"]),
        "recall": float(primary["mean_recall"]),
        "fpr": float(primary["mean_fpr"]),
        "mscd": float(current["task_mscd_px"]),
        "threshold_range": float(current["threshold_iou_range"]),
    }
    values.update(updates)
    row["metrics"]["payload"] = _metrics_payload(
        profile=str(row["variant"]),
        steps=int(row["steps"]),
        config_hash=str(row["expected_config_hash"]),
        final_loss=float(row["history"]["rows"][-1]["effective_loss"]),
        **values,
    )


def _gradient_envelopes(resolution: int) -> list[dict[str, object]]:
    assert resolution in {128, 256}
    stage = "smoke128" if resolution == 128 else "transfer256"
    steps = 200 if resolution == 128 else 2000
    envelopes: list[dict[str, object]] = []
    schedules = {
        "dic_only": list(range(150 if resolution == 128 else 1300, steps + 1, 50)),
        "dts_only": list(range(100 if resolution == 128 else 1000, steps + 1, 50)),
        "v3_full": list(range(100 if resolution == 128 else 1000, steps + 1, 50)),
    }
    for variant in ("dic_only", "dts_only", "v3_full"):
        for update in schedules[variant]:
            dic = variant in {"dic_only", "v3_full"} and update >= (150 if resolution == 128 else 1300)
            dts = variant in {"dts_only", "v3_full"}
            endpoint = None if not dts else (0.06 if update == steps else 0.04)
            active = []
            if dic:
                active.append("dic")
            if dts:
                active.append(f"dts_{endpoint:.2f}")
            envelopes.append(
                {
                    "schema": "pcraft_soft_loss_v3_gradient_sample_v1",
                    "stage": stage,
                    "run_id": f"{stage}/{variant}",
                    "task": "arcane_casters",
                    "variant": variant,
                    "resolution": resolution,
                    "steps": steps,
                    "profile": variant,
                    "completed_updates": update,
                    "active_components": active,
                    "sample": _gradient_sample(update, dic=dic, dts=dts, endpoint=endpoint),
                }
            )
    return envelopes


def _assert_gate_envelope(result: Mapping[str, object], gate: str, status: str) -> None:
    assert set(result) == {"schema", "gate", "status", "criteria", "observed", "failure_codes"}
    assert result["schema"] == "pcraft_soft_loss_v3_gate_result_v1"
    assert result["gate"] == gate
    assert result["status"] == status
    assert result["failure_codes"] == sorted(set(result["failure_codes"]))
    assert set(result["failure_codes"]).issubset(GATE_FAILURE_VOCABULARY[gate])


def _parameter_shape(function: object) -> list[tuple[str, inspect._ParameterKind, object]]:
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    ]


def test_summary_module_exposes_exact_public_signatures_and_parser_surface() -> None:
    assert _parameter_shape(build_summary_parser) == []
    assert _parameter_shape(collect_stage_rows) == [
        ("root", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("stage", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ]
    assert _parameter_shape(gradient_budget_gate) == [
        ("samples", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("resolution", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
    ]
    assert _parameter_shape(development_factorial_gate) == [
        ("rows", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("dic_relative_threshold", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
        ("dts_absolute_threshold", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
        ("require_nonincreasing_range", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
    ]
    assert _parameter_shape(resource_gate) == [
        ("integration_rows", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("benchmark_rows", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("physical_cuda_bytes", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ]
    assert _parameter_shape(repeat_tolerance_gate) == [
        ("primary_rows", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("repeat_rows", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("tolerance", inspect.Parameter.POSITIONAL_OR_KEYWORD, 1e-4),
    ]
    assert _parameter_shape(formal_primary_completion_gate) == [
        ("campaign", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ]
    assert _parameter_shape(write_gate_authorization) == [
        ("path", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("next_stage", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
        ("report", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
        ("hashes", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
    ]
    assert _parameter_shape(summary_main) == [
        ("argv", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
    ]
    parser = build_summary_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.allow_abbrev is False
    observed = {option for action in parser._actions for option in action.option_strings if option != "--help"}
    assert observed == {
        "--stage", "--output-root", "--stage-manifest", "--master-manifest",
        "--weights", "--development-suite", "--formal-suite", "--next-stage",
        "--integration-stage-manifest", "--reference-stage-manifest",
        "--primary-stage-manifest", "--formal-campaign", "--completion-only",
        "--analysis-dir", "--verify-existing-analysis", "--exploratory-only",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(["--out", "/tmp/no-abbrev"])
    with pytest.raises(SystemExit):
        parser.parse_args(["all"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dic_relative_threshold": True, "dts_absolute_threshold": 0.001, "require_nonincreasing_range": True},
        {"dic_relative_threshold": -0.1, "dts_absolute_threshold": 0.001, "require_nonincreasing_range": True},
        {"dic_relative_threshold": 0.05, "dts_absolute_threshold": math.inf, "require_nonincreasing_range": True},
        {"dic_relative_threshold": 0.05, "dts_absolute_threshold": 0.001, "require_nonincreasing_range": 1},
    ],
)
def test_development_public_argument_errors_raise_valueerror(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        development_factorial_gate(_development_rows(), **kwargs)


def test_development_factorial_is_task_equal_and_checks_both_conditionals() -> None:
    result = development_factorial_gate(
        _development_rows(),
        dic_relative_threshold=0.05,
        dts_absolute_threshold=0.001,
        require_nonincreasing_range=True,
    )
    _assert_gate_envelope(result, "development_factorial", "PASS")
    assert set(result["criteria"]) == {
        "task_count", "dic_relative_threshold", "dts_absolute_threshold",
        "iou_noninferiority_delta", "recall_floor_delta", "fpr_ceiling_delta",
        "require_nonincreasing_range",
    }
    assert set(result["observed"]) == {
        "dic_task_equal_effect", "dic_conditional_task_equal_effects",
        "dts_task_equal_effect", "dts_conditional_task_equal_effects",
        "mean_iou_delta_tau_0.05", "min_task_recall_delta",
        "mean_fpr_delta_tau_0.05", "mean_threshold_range_delta", "per_task",
    }
    observed = result["observed"]
    assert observed["dic_task_equal_effect"] == pytest.approx(0.10)
    assert observed["dic_conditional_task_equal_effects"] == pytest.approx({"base_to_dic": 0.10, "dts_to_full": 0.10})
    assert observed["dts_task_equal_effect"] == pytest.approx(0.015)
    assert observed["dts_conditional_task_equal_effects"] == pytest.approx({"base_to_dts": 0.01, "dic_to_full": 0.02})
    assert list(observed["per_task"]) == list(DEVELOPMENT)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("dic_base_reverse", "dic_conditional_reversed"),
        ("dic_dts_reverse", "dic_conditional_reversed"),
        ("dts_base_reverse", "dts_conditional_reversed"),
        ("dts_dic_reverse", "dts_conditional_reversed"),
        ("iou", "iou_noninferiority_failed"),
        ("recall", "recall_noninferiority_failed"),
        ("fpr", "fpr_guard_failed"),
        ("range", "threshold_range_failed"),
        ("invalid_metric", "metric_invalid"),
        ("zero_denominator", "mscd_denominator_invalid"),
        ("missing_row", "factorial_matrix_invalid"),
    ],
)
def test_development_gate_fails_each_independent_clause(mutation: str, code: str) -> None:
    rows = _development_rows()
    by = {(row["task"], row["variant"]): row for row in rows}
    task = DEVELOPMENT[0]
    if mutation == "dic_base_reverse":
        _replace_metrics(by[(task, "dic_only")], mscd=14.0)
    elif mutation == "dic_dts_reverse":
        _replace_metrics(by[(task, "v3_full")], mscd=14.0)
    elif mutation == "dts_base_reverse":
        _replace_metrics(by[(task, "dts_only")], worst_iou=0.30)
    elif mutation == "dts_dic_reverse":
        _replace_metrics(by[(task, "v3_full")], worst_iou=0.30)
    elif mutation == "iou":
        for task_name in DEVELOPMENT:
            _replace_metrics(by[(task_name, "v3_full")], iou=0.798)
    elif mutation == "recall":
        _replace_metrics(by[(task, "v3_full")], recall=0.894)
    elif mutation == "fpr":
        for task_name in DEVELOPMENT:
            _replace_metrics(by[(task_name, "v3_full")], fpr=0.012)
    elif mutation == "range":
        for task_name in DEVELOPMENT:
            _replace_metrics(by[(task_name, "v3_full")], threshold_range=0.11)
    elif mutation == "invalid_metric":
        by[(task, "v3_full")]["metrics"]["payload"]["invalid_task_metric"] = True
    elif mutation == "zero_denominator":
        _replace_metrics(by[(task, "l12_bal")], mscd=0.0)
    elif mutation == "missing_row":
        rows.pop()
    result = development_factorial_gate(
        rows,
        dic_relative_threshold=0.05,
        dts_absolute_threshold=0.001,
        require_nonincreasing_range=True,
    )
    _assert_gate_envelope(result, "development_factorial", "STOP")
    assert code in result["failure_codes"]


def test_development_exact_threshold_equality_passes_and_screen_can_skip_range() -> None:
    rows = _development_rows(stage="screen128", steps=1000)
    for row in rows:
        payload = row["metrics"]["payload"]
        if row["variant"] == "dic_only":
            _replace_metrics(row, mscd=9.75)
        if row["variant"] == "v3_full":
            _replace_metrics(row, mscd=9.75, worst_iou=0.701, threshold_range=0.11)
        if row["variant"] == "dts_only":
            _replace_metrics(row, worst_iou=0.700)
    result = development_factorial_gate(
        rows,
        dic_relative_threshold=0.025,
        dts_absolute_threshold=0.0005,
        require_nonincreasing_range=False,
    )
    assert result["status"] == "PASS"


@pytest.mark.parametrize("bad_rows", ["rows", [1], {"not": "a sequence"}])
def test_public_gate_collections_reject_or_stop_on_nonmapping_sequences(bad_rows: object) -> None:
    development = development_factorial_gate(
        bad_rows,
        dic_relative_threshold=0.05,
        dts_absolute_threshold=0.001,
        require_nonincreasing_range=True,
    )
    _assert_gate_envelope(development, "development_factorial", "STOP")
    assert development["failure_codes"] == ["input_schema_invalid"]
    assert development["observed"] == {
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


@pytest.mark.parametrize("resolution", [128, 256])
def test_gradient_budget_uses_exact_registered_probe_and_joint_pools(resolution: int) -> None:
    result = gradient_budget_gate(_gradient_envelopes(resolution), resolution=resolution)
    _assert_gate_envelope(result, "gradient_budget", "PASS")
    assert set(result["criteria"]) == {
        "resolution", "ratio_min", "ratio_max", "combined_median_max",
        "combined_p95_max", "adverse_cosine_threshold",
        "adverse_cosine_fraction_max", "component_active_l2_epsilon",
        "support_abs_epsilon", "eligible_inactive_policy",
    }
    assert set(result["observed"]) == {
        "expected_probe_count", "observed_probe_count", "dic_ratio_median",
        "dts_ratio_median", "combined_ratio_median", "combined_ratio_p95",
        "adverse_cosine_fraction", "endpoint_diagnostics",
    }
    expected_probes = 8 if resolution == 128 else 57
    expected_joint = {"0.04": 1 if resolution == 128 else 14, "0.06": 1}
    assert result["observed"]["expected_probe_count"] == expected_probes
    assert result["observed"]["observed_probe_count"] == expected_probes
    assert {key: value["joint_sample_count"] for key, value in result["observed"]["endpoint_diagnostics"].items()} == expected_joint


@pytest.mark.parametrize("resolution", [True, 64, 512])
def test_gradient_budget_rejects_bool_and_unsupported_resolution(resolution: object) -> None:
    with pytest.raises(ValueError):
        gradient_budget_gate([], resolution=resolution)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("missing", "probe_matrix_invalid"),
        ("duplicate_component", "input_schema_invalid"),
        ("reordered_component", "input_schema_invalid"),
        ("inactive", "eligible_component_inactive"),
        ("ratio", "component_ratio_out_of_range"),
        ("combined_median", "combined_ratio_median_exceeded"),
        ("combined_p95", "combined_ratio_p95_exceeded"),
        ("cosine", "adverse_cosine_fraction_exceeded"),
        ("support", "endpoint_diagnostic_invalid"),
    ],
)
def test_gradient_gate_fails_closed_on_each_registered_policy(mutation: str, code: str) -> None:
    samples = _gradient_envelopes(128)
    if mutation == "missing":
        samples.pop()
    elif mutation == "duplicate_component":
        samples[-1]["active_components"] = ["dic", "dic", "dts_0.06"]
    elif mutation == "reordered_component":
        samples[-1]["active_components"] = ["dts_0.06", "dic"]
    elif mutation == "inactive":
        samples[-1]["sample"]["dic_norm"] = 1e-13
    elif mutation == "ratio":
        for item in samples:
            if "dic" in item["active_components"]:
                item["sample"]["weighted_dic_base_ratio"] = 0.25
    elif mutation == "combined_median":
        for item in samples:
            item["sample"]["combined_aux_base_ratio"] = 0.40
    elif mutation == "combined_p95":
        samples[-1]["sample"]["combined_aux_base_ratio"] = 20.0
    elif mutation == "cosine":
        for item in samples:
            item["sample"]["base_combined_aux_cosine"] = -0.75
    elif mutation == "support":
        samples[-1]["sample"]["dic_dts_support_ious"]["0.06"] = None
        samples[-1]["sample"]["dic_dts_support_iou_invalid"]["0.06"] = True
    result = gradient_budget_gate(samples, resolution=128)
    _assert_gate_envelope(result, "gradient_budget", "STOP")
    assert code in result["failure_codes"]


def test_gradient_joint_pool_ignores_single_profile_and_unselected_endpoint_diagnostics() -> None:
    samples = _gradient_envelopes(128)
    for item in samples:
        sample = item["sample"]
        if item["variant"] in {"dic_only", "dts_only"}:
            sample["dic_dts_cosines"] = {}
            sample["dic_dts_cosine_invalid"] = {}
            sample["dic_dts_support_ious"] = {}
            sample["dic_dts_support_iou_invalid"] = {}
        if item["variant"] == "v3_full" and item["completed_updates"] == 150:
            sample["dic_dts_cosines"]["0.06"] = -999.0
            sample["dic_dts_support_ious"]["0.06"] = 999.0
    result = gradient_budget_gate(samples, resolution=128)
    _assert_gate_envelope(result, "gradient_budget", "PASS")
    assert result["observed"]["endpoint_diagnostics"]["0.04"] == {
        "joint_sample_count": 1,
        "dic_dts_cosine_median": 0.5,
        "dic_dts_support_iou_median": 0.5,
    }


def test_cuda_integration_exact_schema_and_each_failure_mode() -> None:
    result = _cuda_integration_gate(_integration_rows())
    _assert_gate_envelope(result, "cuda_integration", "PASS")
    assert result["criteria"] == {
        "run_count": 4, "updates_per_run": 2, "final_progress": 1.0,
    }
    assert set(result["observed"]) == {"branch_checks"}
    assert set(result["observed"]["branch_checks"]) == set(PRIMARY)

    missing = _cuda_integration_gate(_integration_rows()[:-1])
    assert missing["failure_codes"] == ["integration_matrix_invalid"]
    assert missing["observed"] == {"branch_checks": {}}

    inactive_rows = _integration_rows()
    inactive_rows[-1]["history"]["rows"][-1]["dic_scale"] = 0.0
    inactive_rows[-1]["history"]["rows"][-1]["loss_dic"] = None
    assert "integration_branch_inactive" in _cuda_integration_gate(inactive_rows)["failure_codes"]

    nonfinite_rows = _integration_rows()
    nonfinite_rows[0]["history"]["rows"][-1]["pre_clip_grad_norm"] = math.inf
    assert "integration_nonfinite" in _cuda_integration_gate(nonfinite_rows)["failure_codes"]

    downstream_rows = _integration_rows()
    downstream_rows[0]["run_spec"]["soft_only"] = False
    assert "soft_only_violation" in _cuda_integration_gate(downstream_rows)["failure_codes"]


def _integration_rows() -> list[dict[str, object]]:
    return [_collected_row(stage="cuda2step256", task="arcane_casters", variant=variant, resolution=256, steps=2, step_times=[0.0, 0.01]) for variant in PRIMARY]


def _resource_rows(*, baseline: list[float] | None = None, full: list[float] | None = None) -> list[dict[str, object]]:
    base = [0.01] * 200 if baseline is None else baseline
    v3 = [0.02] * 200 if full is None else full
    return [
        _collected_row(stage="resource256", task="arcane_casters", variant="l12_bal", resolution=256, steps=200, step_times=base),
        _collected_row(stage="resource256", task="arcane_casters", variant="v3_full", resolution=256, steps=200, step_times=v3),
    ]


def test_resource_gate_recomputes_raw_type7_timings_and_accepts_zero_integration_time() -> None:
    baseline = [0.0] * 20 + [float(index) / 1000.0 for index in range(1, 181)]
    full = [0.0] * 20 + [float(index) / 500.0 for index in range(1, 181)]
    result = resource_gate(_integration_rows(), _resource_rows(baseline=baseline, full=full), 1000)
    _assert_gate_envelope(result, "resource_feasibility", "PASS")
    assert set(result["criteria"]) == {
        "integration_run_count", "benchmark_run_count", "warmup_updates",
        "timed_updates", "peak_memory_fraction_max",
        "full_to_baseline_p95_max", "baseline_p95_must_be_positive",
    }
    assert set(result["observed"]) == {
        "physical_cuda_bytes", "peak_cuda_bytes_by_variant",
        "peak_memory_fraction_by_variant", "median_seconds_by_variant",
        "p95_seconds_by_variant", "full_to_baseline_p95_ratio",
        "integration_branch_checks",
    }
    observed = result["observed"]
    assert observed["p95_seconds_by_variant"]["l12_bal"] == pytest.approx(float(np.percentile(np.asarray(baseline[20:]), 95)))
    assert observed["full_to_baseline_p95_ratio"] == pytest.approx(2.0)
    assert observed["integration_branch_checks"]["v3_full"]["observed_dic_active"] is True


@pytest.mark.parametrize("physical", [True, 0, -1, 1.5])
def test_resource_gate_rejects_invalid_physical_memory_argument(physical: object) -> None:
    with pytest.raises(ValueError):
        resource_gate(_integration_rows(), _resource_rows(), physical)


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("missing_integration", "resource_matrix_invalid"),
        ("missing_benchmark", "resource_matrix_invalid"),
        ("zero_baseline", "baseline_p95_nonpositive"),
        ("runtime", "runtime_ratio_exceeded"),
        ("memory", "peak_memory_exceeded"),
        ("summary", "resource_summary_mismatch"),
        ("physical", "physical_memory_invalid"),
    ],
)
def test_resource_gate_fails_each_resource_clause(mutation: str, code: str) -> None:
    integration = _integration_rows()
    benchmarks = _resource_rows()
    if mutation == "missing_integration":
        integration.pop()
    elif mutation == "missing_benchmark":
        benchmarks.pop()
    elif mutation == "zero_baseline":
        benchmarks = _resource_rows(baseline=[0.0] * 200)
    elif mutation == "runtime":
        benchmarks = _resource_rows(full=[0.1] * 200)
    elif mutation == "memory":
        benchmarks[1]["resource_metrics"]["payload"]["peak_cuda_allocated_bytes"] = 901
    elif mutation == "summary":
        benchmarks[0]["resource_metrics"]["payload"]["updates_21_end"]["p95_seconds"] = 99.0
    elif mutation == "physical":
        benchmarks[1]["resource_metrics"]["payload"]["physical_device_memory_bytes"] = 999
    result = resource_gate(integration, benchmarks, 1000)
    _assert_gate_envelope(result, "resource_feasibility", "STOP")
    assert code in result["failure_codes"]


def test_resource_exact_type7_even_odd_and_equal_limits_pass() -> None:
    baseline_tail = [float(index) for index in range(1, 181)]
    full_tail = [3.5 * value for value in baseline_tail]
    rows = _resource_rows(
        baseline=[0.0] * 20 + baseline_tail,
        full=[0.0] * 20 + full_tail,
    )
    rows[0]["resource_metrics"]["payload"]["peak_cuda_allocated_bytes"] = 900
    rows[1]["resource_metrics"]["payload"]["peak_cuda_allocated_bytes"] = 900

    def type7(values: list[float]) -> float:
        position = Fraction("0.95") * (len(values) - 1)
        lower = position.numerator // position.denominator
        upper = lower if position.denominator == 1 else lower + 1
        if lower == upper:
            return float(values[lower])
        return float(
            values[lower]
            + float(position - lower) * (values[upper] - values[lower])
        )

    for row in rows:
        payload = row["resource_metrics"]["payload"]
        values = payload["step_times"]
        tail = values[20:]
        payload["all_updates"] = {
            "count": len(values),
            "median_seconds": float(np.median(values)),
            "p95_seconds": type7(values),
        }
        payload["updates_21_end"] = {
            "count": len(tail),
            "median_seconds": float(np.median(tail)),
            "p95_seconds": type7(tail),
        }
    result = resource_gate(_integration_rows(), rows, 1000)
    _assert_gate_envelope(result, "resource_feasibility", "PASS")
    assert result["observed"]["p95_seconds_by_variant"]["l12_bal"] == 171.05
    assert result["observed"]["full_to_baseline_p95_ratio"] == pytest.approx(3.5)
    assert result["observed"]["peak_memory_fraction_by_variant"] == {
        "l12_bal": 0.9, "v3_full": 0.9,
    }


def _repeat_rows(delta: float = 0.0) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    primary = [_collected_row(stage="transfer256", task="arcane_casters", variant=variant, resolution=256, steps=2000) for variant in PRIMARY]
    repeat = [_collected_row(stage="sentinel_repeat256", task="arcane_casters", variant=variant, resolution=256, steps=2000, mscd=1.0 + delta, worst_iou=0.7 + delta, role="technical_repeat", replicate_index=1) for variant in PRIMARY]
    for left, right in zip(primary, repeat):
        right["scientific_config_hash"] = left["scientific_config_hash"]
    return primary, repeat


def test_repeat_tolerance_equality_passes_and_preserves_signed_differences() -> None:
    primary, repeat = _repeat_rows(1e-4)
    result = repeat_tolerance_gate(primary, repeat)
    _assert_gate_envelope(result, "repeat_tolerance", "PASS")
    assert set(result["criteria"]) == {"pair_count", "tolerance", "metrics"}
    assert set(result["observed"]) == {
        "paired_differences", "max_abs_task_mscd_px",
        "max_abs_worst_threshold_iou",
    }
    assert result["observed"]["max_abs_task_mscd_px"] == pytest.approx(1e-4)
    assert result["observed"]["paired_differences"]["v3_full"]["worst_threshold_iou"] == pytest.approx(1e-4)


@pytest.mark.parametrize("tolerance", [True, -1.0, math.nan])
def test_repeat_tolerance_rejects_invalid_public_tolerance(tolerance: object) -> None:
    primary, repeat = _repeat_rows()
    with pytest.raises(ValueError):
        repeat_tolerance_gate(primary, repeat, tolerance=tolerance)


@pytest.mark.parametrize(("mutation", "code"), [("large_mscd", "repeat_mscd_tolerance_exceeded"), ("large_iou", "repeat_worst_iou_tolerance_exceeded"), ("pair", "repeat_pair_invalid"), ("invalid", "metric_invalid")])
def test_repeat_gate_fails_closed_on_tolerance_pair_and_metric(mutation: str, code: str) -> None:
    primary, repeat = _repeat_rows()
    if mutation == "large_mscd":
        _replace_metrics(repeat[0], mscd=1.001)
    elif mutation == "large_iou":
        _replace_metrics(repeat[0], worst_iou=0.701)
    elif mutation == "pair":
        repeat[0]["scientific_config_hash"] = _sha("9")
    else:
        repeat[0]["metrics"]["payload"]["invalid_task_metric"] = True
    result = repeat_tolerance_gate(primary, repeat)
    _assert_gate_envelope(result, "repeat_tolerance", "STOP")
    assert code in result["failure_codes"]


@pytest.mark.parametrize(
    "mutation",
    ["task", "variant", "role", "replicate_index", "run_id", "path", "order"],
)
def test_repeat_pairing_rejects_every_registered_identity_mismatch(mutation: str) -> None:
    primary, repeat = _repeat_rows()
    if mutation == "task":
        repeat[0]["task"] = "blade_heroes"
    elif mutation == "variant":
        repeat[0]["variant"] = "dic_only"
    elif mutation == "role":
        repeat[0]["replicate_role"] = "development"
    elif mutation == "replicate_index":
        repeat[0]["replicate_index"] = 0
    elif mutation == "run_id":
        repeat[0]["run_id"] = primary[0]["run_id"]
    elif mutation == "path":
        repeat[0]["path"] = primary[0]["path"]
    else:
        repeat[0], repeat[1] = repeat[1], repeat[0]
    result = repeat_tolerance_gate(primary, repeat)
    _assert_gate_envelope(result, "repeat_tolerance", "STOP")
    assert "repeat_pair_invalid" in result["failure_codes"]


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner_module.canonical_json_bytes(payload))


def _collector_history(profile_name: str, steps: int) -> list[dict[str, object]]:
    profile = summary_module.resolve_profile(profile_name)
    rows: list[dict[str, object]] = []
    for update in range(1, steps + 1):
        progress = update / float(steps)
        dic_scale = 0.25 * summary_module.dic_schedule(progress) if profile.use_dic else 0.0
        dts_scale = 0.5 * summary_module.dts_schedule(progress) if profile.use_dts else 0.0
        loss_dic = 0.5 if dic_scale > 0.0 else None
        loss_dts = 0.25 if dts_scale > 0.0 else None
        endpoint = None
        if loss_dts is not None:
            assert profile.dts_thresholds is not None
            endpoint = float(profile.dts_thresholds[(update - 1) % 2])
        effective = 1.0
        if loss_dic is not None:
            effective += dic_scale * loss_dic
        if loss_dts is not None:
            effective += dts_scale * loss_dts
        rows.append(
            {
                "global_step_before": update - 1,
                "completed_updates_after": update,
                "progress": progress,
                "dts_endpoint": endpoint,
                "loss_base": 1.0,
                "loss_dic": loss_dic,
                "loss_dts": loss_dts,
                "dic_scale": float(dic_scale),
                "dts_scale": float(dts_scale),
                "pre_clip_grad_norm": 1.0,
                "post_clip_grad_norm": 1.0,
                "effective_loss": float(effective),
            }
        )
    return rows


def _run_spec_from_fixture(row: Mapping[str, object], root: Path) -> runner_module.RunSpec:
    values = json.loads(runner_module.canonical_json_bytes(row["run_spec"]))
    values["output_path"] = root / str(values["run_id"])
    values["inputs_xyz"] = tuple(values["inputs_xyz"])
    for name in ("dic_schedule", "dts_schedule", "evaluation_thresholds"):
        values[name] = tuple(values[name])
    if values["dts_thresholds"] is not None:
        values["dts_thresholds"] = tuple(values["dts_thresholds"])
    return runner_module.RunSpec(**values)


def _write_registered_run(
    spec: runner_module.RunSpec,
    *,
    changed: bool,
) -> dict[str, object]:
    run_dir = spec.output_path
    run_dir.mkdir(parents=True)
    config_hash = _sha("c")
    history = _collector_history(spec.profile, spec.steps)
    metrics = _metrics_payload(
        profile=spec.profile,
        steps=spec.steps,
        config_hash=config_hash,
        iou=0.80,
        worst_iou=0.70,
        recall=0.90,
        fpr=0.01,
        mscd=1.0,
        threshold_range=0.02,
        final_loss=float(history[-1]["effective_loss"]),
    )
    samples = [
        deepcopy(item["sample"])
        for item in _gradient_envelopes(spec.resolution)
        if item["variant"] == spec.variant
    ]
    for sample in samples:
        if spec.variant == "v3_full" and sample["completed_updates"] < (
            150 if spec.resolution == 128 else 1300
        ):
            sample["dic_norm"] = 1.0
            sample["base_dic_cosine"] = 0.0
            sample["base_dic_cosine_invalid"] = False
            sample["weighted_dic_base_ratio"] = 0.15
            sample["weighted_dic_base_ratio_invalid"] = False
    gradient = {
        "schema": "pcraft_soft_loss_v3_gradient_audit_v1",
        "config_hash": config_hash,
        "samples": samples,
        "calibration_records": [],
        "active_invalid_flags": [
            {"completed_updates": int(sample["completed_updates"]), "flags": []}
            for sample in samples
        ],
    }
    times = [0.0] + [0.01] * (spec.steps - 1)
    resource = {
        "schema": "pcraft_soft_loss_v3_resources_v1",
        "step_times": times,
        "peak_cuda_allocated_bytes": 100,
        "physical_device_memory_bytes": 1000,
        "all_updates": {
            "count": spec.steps,
            "median_seconds": float(np.median(times)),
            "p95_seconds": float(np.percentile(times, 95)),
        },
        "updates_21_end": {
            "count": max(spec.steps - 20, 0),
            "median_seconds": float(np.median(times[20:])),
            "p95_seconds": float(np.percentile(times[20:], 95)),
        },
    }
    _write_canonical(run_dir / "metrics.json", metrics)
    _write_canonical(run_dir / "gradient_audit.json", gradient)
    _write_canonical(run_dir / "resource_metrics.json", resource)
    history_bytes = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
        for row in history
    )
    (run_dir / "history.jsonl").write_bytes(history_bytes)

    reference = np.float32(torch.sigmoid(torch.tensor(-4.0, dtype=torch.float32)).item())
    occupancy = np.full((spec.resolution,) * 3, reference, dtype=np.float32)
    if changed:
        occupancy.flat[0] = np.float32(reference + np.float32(0.01))
    with (run_dir / "occupancy_probability.npy").open("wb") as handle:
        np.save(handle, occupancy, allow_pickle=False)

    namespace = runner_module._expected_artifact_namespace(spec)
    for name in namespace["top_level"]:
        target = run_dir / name
        if not target.exists():
            target.write_bytes(f"fixture:{name}".encode("utf-8"))
    transaction_updates = (0, 100, 200)
    artifact_names = set(namespace["top_level"])
    for update in transaction_updates:
        for name in namespace["transaction_files"]:
            relative = f"transactions/step_{update:08d}/{name}"
            target = run_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"fixture:{relative}".encode("utf-8"))
            artifact_names.add(relative)
    artifacts = {
        name: _file_sha(run_dir / name)
        for name in sorted(artifact_names)
    }
    manifest = {
        "schema": "pcraft_soft_loss_v3_artifacts_v1",
        "config_hash": config_hash,
        "artifacts_sha256": artifacts,
    }
    _write_canonical(run_dir / "artifact_manifest.json", manifest)
    status = {
        "schema": "pcraft_soft_loss_v3_status_v1",
        "state": "completed",
        "config_hash": config_hash,
        "completed_updates": spec.steps,
        "artifact_manifest_sha256": _file_sha(run_dir / "artifact_manifest.json"),
    }
    _write_canonical(run_dir / "STATUS.json", status)
    return {
        "run_id": spec.run_id,
        "path": str(spec.output_path),
        "run_spec": runner_module._run_spec_payload(spec),
        "scientific_config_hash": runner_module.scientific_config_hash(spec),
        "expected_config_hash": config_hash,
        "expected_artifacts": namespace,
    }


def _refresh_registered_artifact(run_dir: Path, relative: str) -> None:
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts_sha256"][relative] = _file_sha(run_dir / relative)
    _write_canonical(manifest_path, manifest)
    status_path = run_dir / "STATUS.json"
    status = json.loads(status_path.read_bytes())
    status["artifact_manifest_sha256"] = _file_sha(manifest_path)
    _write_canonical(status_path, status)


def _collector_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    changed: bool = True,
) -> tuple[Path, list[runner_module.RunSpec], dict[str, object], list[tuple[str, str]]]:
    root = (tmp_path / "root").resolve()
    root.mkdir()
    specs: list[runner_module.RunSpec] = []
    registrations: list[dict[str, object]] = []
    for variant in PRIMARY:
        row = _collected_row(
            stage="smoke128",
            task="arcane_casters",
            variant=variant,
            resolution=128,
            steps=200,
        )
        spec = _run_spec_from_fixture(row, root)
        specs.append(spec)
        registrations.append(_write_registered_run(spec, changed=changed))
    stage: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_stage_v1",
        "status": "APPROVED_FROZEN",
        "stage": "smoke128",
        "authorization": None,
        "formal_campaign": None,
        "run_count": 4,
        "runs": registrations,
    }
    stage["semantic_sha256"] = canonical_hash(stage)
    stage_path = root / "manifests" / "stages" / "smoke128.json"
    _write_canonical(stage_path, stage)
    persisted_stage = json.loads(stage_path.read_bytes())
    stage_identity = {
        "path": str(stage_path),
        "sha256": _file_sha(stage_path),
        "semantic_sha256": stage["semantic_sha256"],
    }
    context = {
        "output_root": str(root),
        "tasks": {},
        "master": _identity(str(root / "experiment_manifest.json"), "1", semantic=True),
        "development": _identity(str(root / "manifests/development_suite.json"), "2", semantic=True),
        "formal": _identity(str(root / "manifests/formal_suite.json"), "3", semantic=True),
        "specification": _identity(str(tmp_path / "spec.md"), "4"),
        "weights": {**_identity(str(root / "analysis/weights_frozen.json"), "5", semantic=True), "status": "APPROVED_FROZEN", "lambdas": {"lambda_dic": 0.25, "lambda_dts": 0.5}},
        "authorization": None,
        "campaign_payload": None,
        "implementation": runner_module._ordinary_source_identity(),
    }
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda stage_name, supplied: (calls.append(("context", stage_name)) or context))
    monkeypatch.setattr(runner_module, "build_stage_specs", lambda stage_name, supplied: (calls.append(("build", stage_name)) or specs))
    monkeypatch.setattr(runner_module, "_validate_stage_spec_sequence", lambda stage_name, values, **kwargs: calls.append(("matrix", stage_name)))
    monkeypatch.setattr(runner_module, "_verify_spec_dependencies", lambda values: calls.append(("dependencies", "smoke128")))
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: _sha("c"))
    monkeypatch.setattr(
        runner_module,
        "_validate_source_stage_identity",
        lambda value, **kwargs: (
            calls.append(("source", kwargs["source_stage"]))
            or (dict(value), persisted_stage)
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "is_completed_run",
        lambda path, spec: (calls.append(("completed", spec.variant)) or True),
    )
    return root, specs, stage, calls


def test_collector_returns_exact_registered_rows_and_parses_all_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, specs, stage, calls = _collector_root(tmp_path, monkeypatch)
    rows = collect_stage_rows(root, "smoke128")
    assert len(rows) == 4
    assert [row["variant"] for row in rows] == list(PRIMARY)
    assert all(set(row) == ROW_KEYS for row in rows)
    assert all(row["run_spec"] == runner_module._run_spec_payload(spec) for row, spec in zip(rows, specs))
    assert all(row["stage_manifest"]["semantic_sha256"] == stage["semantic_sha256"] for row in rows)
    assert all(row["formal_campaign"] is None for row in rows)
    assert all(row["parameter_change"]["changed"] is True for row in rows)
    assert all(row["parameter_change"]["max_abs_delta"] > 0.0 for row in rows)
    assert all(len(row["history"]["rows"]) == 200 for row in rows)
    assert all(row["metrics"]["payload"]["final_loss"] == row["history"]["rows"][-1]["effective_loss"] for row in rows)
    assert [(kind, name) for kind, name in calls if kind == "completed"] == [
        ("completed", variant) for variant in PRIMARY
    ]
    assert {kind for kind, _ in calls} >= {"context", "build", "matrix", "dependencies", "source", "completed"}


def test_collector_uses_float32_bitwise_change_and_rejects_arbitrary_extra_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _, _ = _collector_root(tmp_path, monkeypatch, changed=False)
    rows = collect_stage_rows(root, "smoke128")
    assert [row["parameter_change"]["changed"] for row in rows] == [False] * 4
    assert [row["parameter_change"]["max_abs_delta"] for row in rows] == [0.0] * 4
    extra = Path(rows[0]["path"]) / "innocent_extra.bin"
    extra.write_bytes(b"not registered")
    with pytest.raises(ValueError, match="artifact|namespace|extra"):
        collect_stage_rows(root, "smoke128")


@pytest.mark.parametrize("mutation", ["metrics_loss", "metric_count", "history_loss"])
def test_collector_rejects_internal_metric_and_history_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root, specs, _, _ = _collector_root(tmp_path, monkeypatch)
    run_dir = specs[0].output_path
    if mutation in {"metrics_loss", "metric_count"}:
        path = run_dir / "metrics.json"
        payload = json.loads(path.read_bytes())
        if mutation == "metrics_loss":
            payload["final_loss"] += 1.0
        else:
            payload["thresholds"]["0.05"]["per_view"][0]["tp"] += 1
        _write_canonical(path, payload)
        _refresh_registered_artifact(run_dir, "metrics.json")
    else:
        path = run_dir / "history.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["effective_loss"] += 1.0
        path.write_bytes(b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows))
        _refresh_registered_artifact(run_dir, "history.jsonl")
    with pytest.raises(ValueError, match="metric|history|loss|count"):
        collect_stage_rows(root, "smoke128")


def test_collector_stable_revalidation_detects_stage_toctou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _, _ = _collector_root(tmp_path, monkeypatch)
    stage_path = root / "manifests" / "stages" / "smoke128.json"
    original = runner_module._validate_source_stage_identity
    calls = 0

    def mutate_after_auth(value: object, **kwargs: object) -> object:
        nonlocal calls
        result = original(value, **kwargs)
        calls += 1
        if calls == 1:
            stage_path.write_bytes(stage_path.read_bytes() + b" ")
        return result

    monkeypatch.setattr(runner_module, "_validate_source_stage_identity", mutate_after_auth)
    with pytest.raises(ValueError, match="changed|snapshot|stage|canonical"):
        collect_stage_rows(root, "smoke128")


def _formal_campaign(root: Path) -> dict[str, object]:
    primary = []
    repeats = []
    pairs = []
    for task in FORMAL:
        for variant in PRIMARY:
            primary_id = f"formal256_primary/steps_2000/{task}/{variant}/replicate_0/seed_17"
            repeat_id = f"formal256_repeat/steps_2000/{task}/{variant}/replicate_1/seed_17"
            l1, l2, use_dic, use_dts, dts_thresholds = {
                "l12_bal": (0.35, 1.0, False, False, None),
                "dic_only": (0.35, 1.0, True, False, None),
                "dts_only": (0.35, 1.0, False, True, (0.04, 0.06)),
                "v3_full": (0.35, 1.0, True, True, (0.04, 0.06)),
            }[variant]
            common = {
                "task": task,
                "variant": variant,
                "profile": variant,
                "inputs_xyz": tuple({"axis": axis, "source_path": f"/tmp/source/{task}_{axis}.png", "resolved_path": f"/tmp/source/{task}_{axis}.png", "sha256": _sha(str(index + 1))} for index, axis in enumerate("xyz")),
                "resolution": 256,
                "export_resolution": 256,
                "steps": 2000,
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
                "frozen_lambdas": {"lambda_dic": 0.25, "lambda_dts": 0.5},
                "master_identity": _identity(str(root / "experiment_manifest.json"), "4", semantic=True),
                "suite_identity": _identity(str(root / "manifests/formal_suite.json"), "5", semantic=True),
                "specification_identity": _identity("/tmp/spec.md", "6"),
                "source_identity": {"schema": "pcraft_soft_loss_v3_runner_source_v1", "bundle_sha256": _sha("7")},
                "weight_identity": {**_identity(str(root / "analysis/weights_frozen.json"), "8", semantic=True), "status": "APPROVED_FROZEN", "lambdas": {"lambda_dic": 0.25, "lambda_dts": 0.5}},
            }
            primary_spec = runner_module.RunSpec(stage="formal256_primary", replicate_role="primary", replicate_index=0, output_path=root / primary_id, run_id=primary_id, inference_eligible=True, **common)
            repeat_spec = runner_module.RunSpec(stage="formal256_repeat", replicate_role="technical_repeat", replicate_index=1, output_path=root / repeat_id, run_id=repeat_id, inference_eligible=False, **common)
            first = runner_module._campaign_cell(primary_spec)
            second = runner_module._campaign_cell(repeat_spec)
            scientific = str(first["scientific_config_hash"])
            assert second["scientific_config_hash"] == scientific
            primary.append(first)
            repeats.append(second)
            pairs.append({"task_variant": [task, variant], "primary_run_id": primary_id, "repeat_run_id": repeat_id, "primary_path": first["path"], "repeat_path": second["path"], "scientific_config_hash": scientific})
    payload: dict[str, object] = {"schema": "pcraft_soft_loss_v3_formal_campaign_v1", "status": "APPROVED_FROZEN", "primary": primary, "repeats": repeats, "pairs": pairs}
    payload["semantic_sha256"] = canonical_hash(payload)
    return payload


def _formal_collection_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    *,
    shared_root: Path | None = None,
) -> tuple[Path, list[runner_module.RunSpec], dict[str, object], list[bool]]:
    """Build a tiny-content, full-registration formal collector fixture.

    The real collector still traverses the exact registered matrix, snapshots
    every required artifact and builds the public rows.  Only the expensive
    256-cubed payload decoder and 2,000-row numeric decoders are replaced by
    deterministic content-parser seams; smoke tests above exercise those real
    parsers byte-for-byte.
    """

    root = (
        (tmp_path / f"{stage}-root").resolve()
        if shared_root is None
        else shared_root.resolve()
    )
    root.mkdir(exist_ok=True)
    campaign = _formal_campaign(root)
    campaign_path = root / "manifests" / "formal_campaign.json"
    _write_canonical(campaign_path, campaign)
    persisted_campaign = json.loads(campaign_path.read_bytes())
    campaign_snapshot = registry_module._snapshot_file(
        campaign_path, label="formal campaign", root=root
    )
    campaign_identity = {
        "path": str(campaign_path),
        "sha256": campaign_snapshot.sha256,
        "semantic_sha256": campaign["semantic_sha256"],
    }

    if stage in {"formal256_primary", "formal256_repeat"}:
        cells = campaign["primary" if stage == "formal256_primary" else "repeats"]
        specs = [
            _run_spec_from_fixture(
                {
                    "run_spec": json.loads(
                        runner_module.canonical_json_bytes(cell["run_spec"])
                    )
                },
                root,
            )
            for cell in cells
        ]
    else:
        assert stage == "appendix256"
        primary_specs = [
            _run_spec_from_fixture(
                {
                    "run_spec": json.loads(
                        runner_module.canonical_json_bytes(cell["run_spec"])
                    )
                },
                root,
            )
            for cell in campaign["primary"]
            if cell["variant"] == "v3_full"
        ]
        appendix_profiles = {
            "v3_zero_margin": (0.35, 1.0, (0.05, 0.05)),
            "v3_no_l1": (0.0, 1.0, (0.04, 0.06)),
            "v3_no_l2": (0.35, 0.0, (0.04, 0.06)),
        }
        specs = []
        for base in primary_specs:
            for variant, (l1_weight, l2_weight, dts_thresholds) in appendix_profiles.items():
                run_id = (
                    f"appendix256/steps_2000/{base.task}/{variant}/"
                    "replicate_0/seed_17"
                )
                specs.append(
                    replace(
                        base,
                        stage="appendix256",
                        variant=variant,
                        profile=variant,
                        replicate_role="exploratory",
                        replicate_index=0,
                        output_path=root / run_id,
                        run_id=run_id,
                        l1_weight=l1_weight,
                        l2_weight=l2_weight,
                        dts_thresholds=dts_thresholds,
                        inference_eligible=False,
                    )
                )

    registrations: list[dict[str, object]] = []
    artifact_names = (
        "metrics.json",
        "gradient_audit.json",
        "resource_metrics.json",
        "history.jsonl",
        "occupancy_probability.npy",
    )
    for index, spec in enumerate(specs):
        run_dir = spec.output_path
        run_dir.mkdir(parents=True)
        for name in artifact_names:
            target = run_dir / name
            if name == "history.jsonl":
                target.write_bytes(b"{}\n")
            elif name == "occupancy_probability.npy":
                target.write_bytes(b"tiny formal fixture")
            else:
                _write_canonical(target, {"fixture": name, "index": index})
        artifact_hashes = {
            name: _file_sha(run_dir / name) for name in artifact_names
        }
        artifact_payload = {
            "schema": "pcraft_soft_loss_v3_artifacts_v1",
            "config_hash": _sha("c"),
            "artifacts_sha256": artifact_hashes,
        }
        _write_canonical(run_dir / "artifact_manifest.json", artifact_payload)
        status_payload = {
            "schema": "pcraft_soft_loss_v3_status_v1",
            "state": "completed",
            "config_hash": _sha("c"),
            "completed_updates": 2000,
            "artifact_manifest_sha256": _file_sha(
                run_dir / "artifact_manifest.json"
            ),
        }
        _write_canonical(run_dir / "STATUS.json", status_payload)
        registrations.append(
            {
                "run_id": spec.run_id,
                "path": str(spec.output_path),
                "run_spec": runner_module._run_spec_payload(spec),
                "scientific_config_hash": runner_module.scientific_config_hash(spec),
                "expected_config_hash": _sha("c"),
                "expected_artifacts": runner_module._expected_artifact_namespace(spec),
            }
        )

    stage_payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_stage_v1",
        "status": "APPROVED_FROZEN",
        "stage": stage,
        "authorization": {"fixture": True},
        "formal_campaign": (
            campaign_identity
            if stage in {"formal256_primary", "formal256_repeat"}
            else None
        ),
        "run_count": len(specs),
        "runs": registrations,
    }
    stage_payload["semantic_sha256"] = canonical_hash(stage_payload)
    stage_path = root / "manifests" / "stages" / f"{stage}.json"
    _write_canonical(stage_path, stage_payload)
    persisted_stage_payload = json.loads(stage_path.read_bytes())
    stage_snapshot = registry_module._snapshot_file(
        stage_path, label="formal stage", root=root
    )
    stage_identity = {
        "path": str(stage_path),
        "sha256": stage_snapshot.sha256,
        "semantic_sha256": stage_payload["semantic_sha256"],
    }
    context = {
        "master": _identity(str(root / "experiment_manifest.json"), "1", semantic=True),
        "development": _identity(str(root / "manifests/development_suite.json"), "2", semantic=True),
        "formal": _identity(str(root / "manifests/formal_suite.json"), "3", semantic=True),
        "specification": _identity(str(tmp_path / "spec.md"), "4"),
        "weights": {
            **_identity(str(root / "analysis/weights_frozen.json"), "5", semantic=True),
            "status": "APPROVED_FROZEN",
            "lambdas": {"lambda_dic": 0.25, "lambda_dts": 0.5},
        },
    }
    reject_invalid_calls: list[bool] = []

    monkeypatch.setattr(
        runner_module,
        "_validated_stage_context",
        lambda stage_name, supplied: context,
    )
    monkeypatch.setattr(
        runner_module, "build_stage_specs", lambda stage_name, supplied: specs
    )
    monkeypatch.setattr(
        runner_module, "_validate_stage_spec_sequence", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(runner_module, "_verify_spec_dependencies", lambda specs: None)
    monkeypatch.setattr(
        runner_module,
        "_validate_source_stage_identity",
        lambda value, **kwargs: (dict(value), persisted_stage_payload),
    )
    monkeypatch.setattr(runner_module, "is_completed_run", lambda path, spec: True)
    monkeypatch.setattr(
        runner_module,
        "_load_formal_campaign_identity",
        lambda path, output_root: (
            campaign_identity,
            campaign,
            campaign_snapshot,
        ),
    )

    def snapshots(run_dir: Path, **kwargs: object) -> dict[str, object]:
        return {
            name: registry_module._snapshot_file(
                run_dir / name, label=name, root=run_dir
            )
            for name in artifact_names
        }

    monkeypatch.setattr(summary_module, "_artifact_snapshots", snapshots)
    monkeypatch.setattr(
        summary_module,
        "_parse_history",
        lambda data, *, spec: [{"effective_loss": 1.0}],
    )

    def parse_metrics(
        raw: object,
        *,
        spec: runner_module.RunSpec,
        config_hash: str,
        final_effective_loss: float,
        reject_invalid: bool,
    ) -> dict[str, object]:
        reject_invalid_calls.append(reject_invalid)
        return {
            "invalid_task_metric": True,
            "task_mscd_px": 1.0,
            "worst_threshold_iou": 0.7,
        }

    monkeypatch.setattr(summary_module, "_parse_metrics_payload", parse_metrics)
    monkeypatch.setattr(
        summary_module, "_parse_gradient_audit", lambda raw, **kwargs: {"samples": []}
    )
    monkeypatch.setattr(
        summary_module,
        "_parse_resource_payload",
        lambda raw, **kwargs: {"physical_device_memory_bytes": 1000},
    )
    monkeypatch.setattr(
        summary_module,
        "_parameter_change",
        lambda snapshot, *, resolution: {
            "reference_probability": 0.01,
            "max_abs_delta": 0.1,
            "changed": True,
        },
    )
    return root, specs, campaign_identity, reject_invalid_calls


@pytest.mark.parametrize(
    ("stage", "count", "role", "replicate_index", "eligible"),
    [
        ("formal256_primary", 40, "primary", 0, True),
        ("formal256_repeat", 40, "technical_repeat", 1, False),
        ("appendix256", 30, "exploratory", 0, False),
    ],
)
def test_formal_and_appendix_collectors_preserve_exact_handoff_roles_and_invalid_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    count: int,
    role: str,
    replicate_index: int,
    eligible: bool,
) -> None:
    root, specs, campaign_identity, reject_invalid_calls = _formal_collection_fixture(
        tmp_path, monkeypatch, stage
    )
    rows = collect_stage_rows(root, stage)
    assert len(rows) == count
    assert [row["run_id"] for row in rows] == [spec.run_id for spec in specs]
    assert all(set(row) == ROW_KEYS for row in rows)
    assert all(row["replicate_role"] == role for row in rows)
    assert all(row["replicate_index"] == replicate_index for row in rows)
    assert all(row["inference_eligible"] is eligible for row in rows)
    assert all(row["metrics"]["payload"]["invalid_task_metric"] is True for row in rows)
    assert reject_invalid_calls == [False] * count
    if stage in {"formal256_primary", "formal256_repeat"}:
        assert all(row["formal_campaign"] == campaign_identity for row in rows)
    else:
        assert all(row["formal_campaign"] is None for row in rows)


def test_task13_handoff_keeps_primary_and_repeat_as_two_distinct_paired_matrices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_root = (tmp_path / "formal-handoff-root").resolve()
    primary_root, primary_specs, _, _ = _formal_collection_fixture(
        tmp_path,
        monkeypatch,
        "formal256_primary",
        shared_root=shared_root,
    )
    primary = collect_stage_rows(primary_root, "formal256_primary")
    monkeypatch.undo()
    repeat_root, repeat_specs, _, _ = _formal_collection_fixture(
        tmp_path,
        monkeypatch,
        "formal256_repeat",
        shared_root=shared_root,
    )
    repeat = collect_stage_rows(repeat_root, "formal256_repeat")
    assert len(primary) == len(repeat) == 40
    assert [row["run_id"] for row in primary] == [spec.run_id for spec in primary_specs]
    assert [row["run_id"] for row in repeat] == [spec.run_id for spec in repeat_specs]
    for left, right in zip(primary, repeat):
        assert (left["task"], left["variant"]) == (right["task"], right["variant"])
        assert left["scientific_config_hash"] == right["scientific_config_hash"]
        assert (left["replicate_role"], left["replicate_index"], left["inference_eligible"]) == ("primary", 0, True)
        assert (right["replicate_role"], right["replicate_index"], right["inference_eligible"]) == ("technical_repeat", 1, False)


def _plain_campaign(campaign: Mapping[str, object]) -> dict[str, object]:
    return json.loads(runner_module.canonical_json_bytes(campaign))


def _resign_campaign(campaign: dict[str, object]) -> None:
    campaign["semantic_sha256"] = canonical_hash(
        {key: value for key, value in campaign.items() if key != "semantic_sha256"}
    )


def test_formal_completion_rejects_bare_fake_and_true_mixed_root_campaign(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        formal_primary_completion_gate({})
    root = (tmp_path / "root").resolve()
    campaign = _plain_campaign(_formal_campaign(root))
    foreign = (tmp_path / "foreign").resolve()
    for repeat, pair in zip(campaign["repeats"], campaign["pairs"]):
        new_path = foreign / repeat["run_id"]
        repeat["path"] = str(new_path)
        repeat["run_spec"]["output_path"] = str(new_path)
        pair["repeat_path"] = str(new_path)
    _resign_campaign(campaign)
    with pytest.raises(ValueError, match="root|hierarchy"):
        formal_primary_completion_gate(campaign)


@pytest.mark.parametrize("mutation", ["incomplete", "copied_primary_as_repeat"])
def test_formal_completion_rejects_exact_campaign_cardinality_and_role_mutations(
    tmp_path: Path, mutation: str
) -> None:
    root = (tmp_path / mutation).resolve()
    campaign = _plain_campaign(_formal_campaign(root))
    if mutation == "incomplete":
        campaign["primary"].pop()
        campaign["repeats"].pop()
        campaign["pairs"].pop()
    else:
        campaign["repeats"][0] = deepcopy(campaign["primary"][0])
    _resign_campaign(campaign)
    with pytest.raises(ValueError, match="40|role|registration|cell|pair"):
        formal_primary_completion_gate(campaign)


def test_formal_completion_rejects_symlinked_cell_hierarchy(tmp_path: Path) -> None:
    root = (tmp_path / "symlink-root").resolve()
    root.mkdir()
    real_stage = root / "real-primary"
    real_stage.mkdir()
    (root / "formal256_primary").symlink_to(real_stage, target_is_directory=True)
    campaign = _formal_campaign(root)
    with pytest.raises(ValueError, match="symlink"):
        formal_primary_completion_gate(campaign)


def test_formal_completion_rejects_argument_that_differs_from_current_canonical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "altered-bytes-root").resolve()
    argument = _plain_campaign(_formal_campaign(root))
    on_disk = deepcopy(argument)
    on_disk["primary"][0]["run_spec"]["learning_rate"] = 0.008
    _resign_campaign(on_disk)
    _write_canonical(root / "manifests" / "formal_campaign.json", on_disk)
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("Task 11 campaign seam reached after byte mismatch")

    monkeypatch.setattr(runner_module, "_load_formal_campaign_identity", forbidden)
    with pytest.raises(ValueError, match="canonical|bytes|differs"):
        formal_primary_completion_gate(argument)
    assert called is False


def _authenticated_completion_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    root = (tmp_path / "authenticated-formal-root").resolve()
    campaign = _plain_campaign(_formal_campaign(root))
    campaign_path = root / "manifests" / "formal_campaign.json"
    _write_canonical(campaign_path, campaign)
    campaign_snapshot = registry_module._snapshot_file(
        campaign_path, label="campaign", root=root
    )
    campaign_identity = {
        "path": str(campaign_path),
        "sha256": campaign_snapshot.sha256,
        "semantic_sha256": campaign["semantic_sha256"],
    }
    runs = []
    for cell in campaign["primary"]:
        run_dir = Path(cell["path"])
        run_dir.mkdir(parents=True)
        (run_dir / "STATUS.json").write_bytes(b"opaque status")
        (run_dir / "artifact_manifest.json").write_bytes(b"opaque artifacts")
        runs.append(
            {
                "run_id": cell["run_id"],
                "path": cell["path"],
                "run_spec": cell["run_spec"],
                "scientific_config_hash": cell["scientific_config_hash"],
                "expected_config_hash": _sha("c"),
                "expected_artifacts": {},
            }
        )
    stage_payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_stage_v1",
        "status": "APPROVED_FROZEN",
        "stage": "formal256_primary",
        "authorization": {"fixture": True},
        "formal_campaign": campaign_identity,
        "run_count": 40,
        "runs": runs,
    }
    stage_payload["semantic_sha256"] = canonical_hash(stage_payload)
    stage_path = root / "manifests" / "stages" / "formal256_primary.json"
    _write_canonical(stage_path, stage_payload)
    persisted_stage = json.loads(stage_path.read_bytes())
    stage_snapshot = registry_module._snapshot_file(
        stage_path, label="formal stage", root=root
    )
    stage_identity = {
        "path": str(stage_path),
        "sha256": stage_snapshot.sha256,
        "semantic_sha256": persisted_stage["semantic_sha256"],
    }
    monkeypatch.setattr(
        runner_module,
        "_load_formal_campaign_identity",
        lambda path, output_root: (
            campaign_identity,
            campaign,
            campaign_snapshot,
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "_validated_stage_context",
        lambda stage, context: {
            "master": {},
            "development": {},
            "formal": {},
            "specification": {},
            "weights": None,
        },
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_source_stage_identity",
        lambda value, **kwargs: (dict(value), persisted_stage),
    )
    monkeypatch.setattr(
        runner_module, "_validate_completion_only_run", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runner_module, "_validate_completion_only", lambda *args, **kwargs: None
    )
    return root, campaign, campaign_identity, persisted_stage, stage_identity


@pytest.mark.parametrize(
    "mutation", ["foreign_campaign", "foreign_stage", "registration_mismatch"]
)
def test_formal_completion_rejects_authenticated_foreign_or_mismatched_evidence_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root, campaign, campaign_identity, stage_payload, stage_identity = (
        _authenticated_completion_fixture(tmp_path, monkeypatch)
    )
    calls: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_validate_completion_only_run",
        lambda *args, **kwargs: calls.append("cell"),
    )
    if mutation == "foreign_campaign":
        foreign = deepcopy(campaign)
        foreign["primary"][0]["run_spec"]["learning_rate"] = 0.008
        _resign_campaign(foreign)
        snapshot = registry_module._snapshot_file(
            root / "manifests" / "formal_campaign.json",
            label="campaign",
            root=root,
        )
        monkeypatch.setattr(
            runner_module,
            "_load_formal_campaign_identity",
            lambda path, output_root: (campaign_identity, foreign, snapshot),
        )
    elif mutation == "foreign_stage":
        foreign_stage = deepcopy(stage_payload)
        foreign_stage["stage"] = "formal256_repeat"
        monkeypatch.setattr(
            runner_module,
            "_validate_source_stage_identity",
            lambda value, **kwargs: (stage_identity, foreign_stage),
        )
    else:
        mismatched = deepcopy(stage_payload)
        mismatched["runs"][0]["scientific_config_hash"] = _sha("0")
        _resign_campaign(mismatched)
        stage_path = root / "manifests" / "stages" / "formal256_primary.json"
        _write_canonical(stage_path, mismatched)
        persisted_mismatch = json.loads(stage_path.read_bytes())
        monkeypatch.setattr(
            runner_module,
            "_validate_source_stage_identity",
            lambda value, **kwargs: (dict(value), persisted_mismatch),
        )
    with pytest.raises(ValueError, match="campaign|source|registration|different"):
        formal_primary_completion_gate(campaign)
    assert calls == []


def test_formal_completion_uses_only_completion_seams_and_never_metric_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = (tmp_path / "root").resolve()
    campaign = _formal_campaign(root)
    campaign_path = root / "manifests" / "formal_campaign.json"
    stage_path = root / "manifests" / "stages" / "formal256_primary.json"
    campaign_path.parent.mkdir(parents=True)
    stage_path.parent.mkdir(parents=True)
    campaign_path.write_bytes(runner_module.canonical_json_bytes(campaign))
    campaign = json.loads(campaign_path.read_bytes())
    runs = []
    for cell in campaign["primary"]:
        run = {"run_id": cell["run_id"], "path": cell["path"], "run_spec": cell["run_spec"], "scientific_config_hash": cell["scientific_config_hash"], "expected_config_hash": _sha("c"), "expected_artifacts": {}}
        runs.append(run)
        path = Path(cell["path"])
        path.mkdir(parents=True)
        (path / "STATUS.json").write_bytes(b"status")
        (path / "artifact_manifest.json").write_bytes(b"artifact")
        (path / "metrics.json").write_bytes(b"this is deliberately not JSON")
    stage_payload: dict[str, object] = {"schema": "pcraft_soft_loss_v3_stage_v1", "status": "APPROVED_FROZEN", "stage": "formal256_primary", "authorization": None, "formal_campaign": _identity(str(campaign_path), "a", semantic=True), "run_count": 40, "runs": runs}
    stage_payload["semantic_sha256"] = canonical_hash(stage_payload)
    stage_path.write_bytes(runner_module.canonical_json_bytes(stage_payload))
    stage_payload = json.loads(stage_path.read_bytes())
    monkeypatch.setattr(runner_module, "_load_formal_campaign_identity", lambda path, output_root: ({"path": str(campaign_path), "sha256": registry_module._snapshot_file(campaign_path, label="campaign", root=root).sha256, "semantic_sha256": campaign["semantic_sha256"]}, campaign, registry_module._snapshot_file(campaign_path, label="campaign", root=root)))
    monkeypatch.setattr(runner_module, "_validate_source_stage_identity", lambda value, **kwargs: (dict(value), stage_payload))
    monkeypatch.setattr(
        runner_module,
        "_validated_stage_context",
        lambda stage, context: {
            "master": {}, "development": {}, "formal": {},
            "specification": {}, "weights": None,
        },
    )
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        runner_module,
        "_validate_completion_only_run",
        lambda path, config_hash, total_steps, status_sha256,
        artifact_manifest_sha256, *, output_root: calls.append(
            (
                "cell", path, config_hash, total_steps, status_sha256,
                artifact_manifest_sha256, output_root,
            )
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_completion_only",
        lambda completion, source, frozen, *, output_root: calls.append(
            ("aggregate", deepcopy(completion), source, frozen, output_root)
        ),
    )
    monkeypatch.setattr(runner_module, "is_completed_run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ordinary completion seam called")))
    from shadowforge_pipeline.soft_voxel_v3 import trainer as trainer_module
    monkeypatch.setattr(trainer_module, "_validate_completed_run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("metric-parsing completion called")))
    report = formal_primary_completion_gate(campaign)
    assert report["status"] == "COMPLETE_ONLY"
    assert report["metric_values_parsed"] is False
    assert len(calls) == 41 and calls[-1][0] == "aggregate"
    assert len(report["completion"]) == 40
    assert set(report) == {
        "schema", "status", "source_stage", "next_stage",
        "source_stage_manifest", "formal_campaign", "registered_run_count",
        "completed_run_count", "completion", "metric_values_parsed",
        "failure_codes", "policy", "summarizer_source", "semantic_sha256",
    }
    for index, cell in enumerate(campaign["primary"]):
        call = calls[index]
        assert call == (
            "cell",
            Path(cell["path"]),
            _sha("c"),
            2000,
            _file_sha(Path(cell["path"]) / "STATUS.json"),
            _file_sha(Path(cell["path"]) / "artifact_manifest.json"),
            root,
        )
    assert calls[-1][1] == report["completion"]
    assert calls[-1][2] == stage_payload
    assert calls[-1][3] == campaign
    assert calls[-1][4] == root
    assert all("metric" not in item for item in report["completion"])


def test_formal_completion_authenticated_incomplete_returns_exact_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "root").resolve()
    campaign = _formal_campaign(root)
    campaign_path = root / "manifests" / "formal_campaign.json"
    stage_path = root / "manifests" / "stages" / "formal256_primary.json"
    _write_canonical(campaign_path, campaign)
    campaign = json.loads(campaign_path.read_bytes())
    runs = []
    for cell in campaign["primary"]:
        run_dir = Path(cell["path"])
        run_dir.mkdir(parents=True)
        (run_dir / "STATUS.json").write_bytes(b"status")
        (run_dir / "artifact_manifest.json").write_bytes(b"artifact")
        runs.append(
            {
                "run_id": cell["run_id"], "path": cell["path"],
                "run_spec": cell["run_spec"],
                "scientific_config_hash": cell["scientific_config_hash"],
                "expected_config_hash": _sha("c"), "expected_artifacts": {},
            }
        )
    stage_payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_stage_v1",
        "status": "APPROVED_FROZEN", "stage": "formal256_primary",
        "authorization": None,
        "formal_campaign": _identity(str(campaign_path), "a", semantic=True),
        "run_count": 40, "runs": runs,
    }
    stage_payload["semantic_sha256"] = canonical_hash(stage_payload)
    _write_canonical(stage_path, stage_payload)
    stage_payload = json.loads(stage_path.read_bytes())
    campaign_snapshot = registry_module._snapshot_file(campaign_path, label="campaign", root=root)
    monkeypatch.setattr(
        runner_module,
        "_load_formal_campaign_identity",
        lambda path, output_root: (
            {
                "path": str(campaign_path), "sha256": campaign_snapshot.sha256,
                "semantic_sha256": campaign["semantic_sha256"],
            },
            campaign,
            campaign_snapshot,
        ),
    )
    monkeypatch.setattr(runner_module, "_validate_source_stage_identity", lambda value, **kwargs: (dict(value), stage_payload))
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda stage, context: {"master": {}, "development": {}, "formal": {}, "specification": {}, "weights": None})
    count = 0

    def incomplete(*args: object, **kwargs: object) -> None:
        nonlocal count
        count += 1
        if count == 3:
            raise FileNotFoundError("authenticated primary is incomplete")

    monkeypatch.setattr(runner_module, "_validate_completion_only_run", incomplete)
    monkeypatch.setattr(runner_module, "_validate_completion_only", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("aggregate called after incomplete cell")))
    report = formal_primary_completion_gate(campaign)
    assert report["status"] == "STOP"
    assert report["registered_run_count"] == 40
    assert report["completed_run_count"] == 2
    assert report["completion"] is None
    assert report["failure_codes"] == ["primary_run_incomplete"]
    assert report["metric_values_parsed"] is False
    assert set(report["failure_codes"]) <= {
        "primary_run_incomplete", "primary_artifact_invalid",
    }


def _semantic_fixture(path: Path, name: str) -> dict[str, object]:
    payload: dict[str, object] = {"schema": f"fixture_{name}_v1", "status": "APPROVED_FROZEN"}
    payload["semantic_sha256"] = canonical_hash(payload)
    _write_canonical(path, payload)
    return {
        "path": str(path), "sha256": _file_sha(path),
        "semantic_sha256": payload["semantic_sha256"],
    }


def _publication_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, object], dict[str, str], dict[str, object], dict[str, object]]:
    root = (tmp_path / "publication-root").resolve()
    (root / "manifests" / "stages").mkdir(parents=True)
    (root / "analysis").mkdir()
    (root / "authorizations").mkdir()
    master = _semantic_fixture(root / "experiment_manifest.json", "master")
    development = _semantic_fixture(root / "manifests" / "development_suite.json", "development")
    formal = _semantic_fixture(root / "manifests" / "formal_suite.json", "formal")
    weights = _semantic_fixture(root / "analysis" / "weights_frozen.json", "weights")
    specification_path = tmp_path / "specification.md"
    specification_path.write_bytes(b"# frozen fixture specification\n")
    specification = {"path": str(specification_path.resolve()), "sha256": _file_sha(specification_path)}

    rows = _development_rows(stage="screen128", resolution=128, steps=1000)
    runs = [
        {
            "run_id": row["run_id"], "path": row["path"],
            "run_spec": row["run_spec"],
            "scientific_config_hash": row["scientific_config_hash"],
            "expected_config_hash": row["expected_config_hash"],
            "expected_artifacts": runner_module._expected_artifact_namespace(
                _run_spec_from_fixture(row, root)
            ),
        }
        for row in rows
    ]
    stage_payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_stage_v1",
        "status": "APPROVED_FROZEN", "stage": "screen128",
        "authorization": {"fixture": True}, "formal_campaign": None,
        "run_count": 12, "runs": runs,
    }
    stage_payload["semantic_sha256"] = canonical_hash(stage_payload)
    stage_path = root / "manifests" / "stages" / "screen128.json"
    _write_canonical(stage_path, stage_payload)
    persisted_stage_payload = json.loads(stage_path.read_bytes())
    source_identity = {
        "path": str(stage_path), "sha256": _file_sha(stage_path),
        "semantic_sha256": stage_payload["semantic_sha256"],
    }

    gate = development_factorial_gate(
        rows,
        dic_relative_threshold=0.025,
        dts_absolute_threshold=0.0005,
        require_nonincreasing_range=False,
    )
    assert gate["status"] == "PASS"
    evidence = [
        {
            "run_id": row["run_id"], "path": row["path"],
            "expected_config_hash": row["expected_config_hash"],
            "completed": True,
            "status_sha256": row["status"]["sha256"],
            "artifact_manifest_sha256": row["artifact_manifest"]["sha256"],
            "metrics_sha256": row["metrics"]["sha256"],
            "gradient_audit_sha256": row["gradient_audit"]["sha256"],
            "resource_metrics_sha256": row["resource_metrics"]["sha256"],
            "history_sha256": row["history"]["sha256"],
            "occupancy_probability_sha256": row["artifact_manifest"]["payload"]["artifacts_sha256"]["occupancy_probability.npy"],
        }
        for row in rows
    ]
    summarizer_path = Path("experiments/summarize_soft_loss_v3.py").resolve()
    report: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_gate_report_v1",
        "status": "ADVANCE", "source_stage": "screen128",
        "next_stage": "confirm128", "gate": "development_factorial",
        "source_stage_manifest": source_identity,
        "prerequisite_stage_manifests": [], "formal_campaign": None,
        "registered_run_count": 12, "completed_run_count": 12,
        "run_evidence": evidence,
        "checks": {
            "collection": {
                "status": "PASS", "expected_count": 12,
                "completed_count": 12, "parameter_changed_count": 12,
                "invalid_run_ids": [], "failure_codes": [],
            },
            "primary_gate": gate, "additional_gate": None,
        },
        "failure_codes": [],
        "policy": {
            "paper_eligible": False, "task_equal_aggregation": True,
            "fixed_seed_single_run_best_effort": True,
            "bitwise_reproducible": False,
            "metric_invalid_policy": "fail_closed",
        },
        "summarizer_source": {
            "path": str(summarizer_path), "sha256": _file_sha(summarizer_path),
        },
    }
    report["semantic_sha256"] = canonical_hash(report)
    report_bytes = runner_module.canonical_json_bytes(report)
    hashes = {
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "summarizer_source_sha256": _file_sha(summarizer_path),
        "master_manifest_sha256": str(master["sha256"]),
        "development_suite_sha256": str(development["sha256"]),
        "formal_suite_sha256": str(formal["sha256"]),
        "specification_sha256": str(specification["sha256"]),
        "frozen_weights_sha256": str(weights["sha256"]),
        "source_stage_manifest_sha256": str(source_identity["sha256"]),
    }
    context = {
        "master": master, "development": development, "formal": formal,
        "specification": specification,
        "weights": {**weights, "status": "APPROVED_FROZEN", "lambdas": {"lambda_dic": 0.25, "lambda_dts": 0.5}},
    }
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda stage, supplied: context)
    monkeypatch.setattr(
        runner_module,
        "_validate_source_stage_identity",
        lambda value, **kwargs: (dict(value), persisted_stage_payload),
    )
    monkeypatch.setattr(
        summary_module,
        "collect_stage_rows",
        lambda current_root, stage: rows
        if current_root == root and stage == "screen128"
        else (_ for _ in ()).throw(ValueError("unexpected collector request")),
    )
    authorization = root / "authorizations" / "screen128__to__confirm128.json"
    return root, authorization, report, hashes, context, persisted_stage_payload


def _task11_read_authorization(
    root: Path,
    authorization: Path,
    context: Mapping[str, object],
) -> dict[str, object]:
    return runner_module._validate_authorization(
        "confirm128",
        runner_module.StageContext(
            output_root=root,
            master_manifest=root / "experiment_manifest.json",
            development_suite=root / "manifests" / "development_suite.json",
            formal_suite=root / "manifests" / "formal_suite.json",
            frozen_weights=root / "analysis" / "weights_frozen.json",
            authorization=authorization,
            formal_campaign=None,
        ),
        output_root=root,
        master_identity=context["master"],
        development_identity=context["development"],
        formal_identity=context["formal"],
        specification_identity=context["specification"],
        weight_identity={
            key: context["weights"][key]
            for key in ("path", "sha256", "semantic_sha256")
        },
        campaign_identity=None,
    )


def test_writer_publishes_report_first_authorization_last_and_task11_accepts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, target, report, hashes, context, _ = _publication_fixture(tmp_path, monkeypatch)
    observed_order: list[Path] = []
    original = registry_module._publish_bundle

    def capture(files: Mapping[Path, bytes], **kwargs: object) -> None:
        observed_order.extend(files)
        original(files, **kwargs)

    monkeypatch.setattr(registry_module, "_publish_bundle", capture)
    result = write_gate_authorization(
        target, next_stage="confirm128", report=report, hashes=hashes
    )
    assert result == target
    assert observed_order == [
        root / "analysis" / "screen128_gate.json", target,
    ]
    payload = json.loads(target.read_bytes())
    assert set(payload) == {
        "schema", "status", "source_stage", "next_stage", "report",
        "dependencies", "completion", "semantic_sha256",
    }
    assert payload["status"] == "ADVANCE"
    assert set(payload["dependencies"]) == {
        "master_manifest", "development_suite", "formal_suite",
        "specification", "frozen_weights", "source_stage_manifest",
        "formal_campaign",
    }
    accepted = _task11_read_authorization(root, target, context)
    assert accepted["status"] == "ADVANCE"
    assert write_gate_authorization(
        target, next_stage="confirm128", report=report, hashes=hashes
    ) == target


@pytest.mark.parametrize(
    "guard",
    [
        "report_sha256", "summarizer_source_sha256", "master_manifest_sha256",
        "development_suite_sha256", "formal_suite_sha256",
        "specification_sha256", "frozen_weights_sha256",
        "source_stage_manifest_sha256",
    ],
)
def test_writer_rejects_each_hash_guard_with_zero_final_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, guard: str
) -> None:
    root, target, report, hashes, _, _ = _publication_fixture(tmp_path, monkeypatch)
    hashes[guard] = _sha("0")
    with pytest.raises(ValueError, match="hash|guard|identity|current"):
        write_gate_authorization(
            target, next_stage="confirm128", report=report, hashes=hashes
        )
    assert not target.exists()
    assert not (root / "analysis" / "screen128_gate.json").exists()


def test_writer_rejects_rehashed_forged_pass_that_disagrees_with_current_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, target, report, hashes, _, _ = _publication_fixture(tmp_path, monkeypatch)
    forged = deepcopy(report)
    observed = forged["checks"]["primary_gate"]["observed"]
    observed["dic_task_equal_effect"] = -999.0
    observed["min_task_recall_delta"] = -999.0
    observed["mean_fpr_delta_tau_0.05"] = 999.0
    forged["semantic_sha256"] = canonical_hash(
        {key: value for key, value in forged.items() if key != "semantic_sha256"}
    )
    forged_hashes = dict(hashes)
    forged_hashes["report_sha256"] = hashlib.sha256(
        runner_module.canonical_json_bytes(forged)
    ).hexdigest()
    with pytest.raises(ValueError, match="current|recompute|evidence|report"):
        write_gate_authorization(
            target,
            next_stage="confirm128",
            report=forged,
            hashes=forged_hashes,
        )
    assert not target.exists()
    assert not (root / "analysis" / "screen128_gate.json").exists()


def test_reader_rejects_report_only_window_and_publication_rollback_is_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, target, report, hashes, context, _ = _publication_fixture(tmp_path, monkeypatch)
    report_path = root / "analysis" / "screen128_gate.json"
    report_path.write_bytes(runner_module.canonical_json_bytes(report))
    with pytest.raises((FileNotFoundError, ValueError)):
        _task11_read_authorization(root, target, context)
    report_path.unlink()

    original_link = registry_module.os.link

    def fail_authorization_link(source: object, destination: object, *args: object, **kwargs: object) -> None:
        if destination == target.name:
            raise OSError("injected second-link failure")
        original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(registry_module.os, "link", fail_authorization_link)
    with pytest.raises(OSError, match="second-link"):
        write_gate_authorization(
            target, next_stage="confirm128", report=report, hashes=hashes
        )
    assert not report_path.exists()
    assert not target.exists()
    monkeypatch.setattr(registry_module.os, "link", original_link)
    assert write_gate_authorization(
        target, next_stage="confirm128", report=report, hashes=hashes
    ) == target


def test_two_identical_writers_converge_and_changed_decision_cannot_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, target, report, hashes, _, _ = _publication_fixture(tmp_path, monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: write_gate_authorization(
                    target, next_stage="confirm128", report=report, hashes=hashes
                ),
                range(2),
            )
        )
    assert results == [target, target]
    changed = deepcopy(report)
    changed["checks"]["primary_gate"]["status"] = "STOP"
    changed["checks"]["primary_gate"]["failure_codes"] = ["dic_threshold_missed"]
    changed["status"] = "STOP"
    changed["failure_codes"] = ["dic_threshold_missed"]
    changed["semantic_sha256"] = canonical_hash(
        {key: value for key, value in changed.items() if key != "semantic_sha256"}
    )
    changed_hashes = dict(hashes)
    changed_hashes["report_sha256"] = hashlib.sha256(
        runner_module.canonical_json_bytes(changed)
    ).hexdigest()
    with pytest.raises((FileExistsError, ValueError)):
        write_gate_authorization(
            target, next_stage="confirm128", report=changed, hashes=changed_hashes
        )
    assert json.loads(target.read_bytes())["status"] == "ADVANCE"


def test_completion_writer_publishes_exact_complete_only_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "completion-root").resolve()
    (root / "manifests" / "stages").mkdir(parents=True)
    (root / "analysis").mkdir()
    (root / "authorizations").mkdir()
    master = _semantic_fixture(root / "experiment_manifest.json", "master")
    development = _semantic_fixture(root / "manifests" / "development_suite.json", "development")
    formal = _semantic_fixture(root / "manifests" / "formal_suite.json", "formal")
    weights = _semantic_fixture(root / "analysis" / "weights_frozen.json", "weights")
    specification_path = tmp_path / "formal-spec.md"
    specification_path.write_bytes(b"# formal fixture\n")
    specification = {"path": str(specification_path.resolve()), "sha256": _file_sha(specification_path)}
    campaign = _formal_campaign(root)
    campaign_path = root / "manifests" / "formal_campaign.json"
    _write_canonical(campaign_path, campaign)
    persisted_campaign = json.loads(campaign_path.read_bytes())
    campaign_identity = {
        "path": str(campaign_path), "sha256": _file_sha(campaign_path),
        "semantic_sha256": campaign["semantic_sha256"],
    }
    completion = []
    runs = []
    for cell in campaign["primary"]:
        run_dir = Path(cell["path"])
        run_dir.mkdir(parents=True)
        status = run_dir / "STATUS.json"
        artifact = run_dir / "artifact_manifest.json"
        status.write_bytes(b"opaque status")
        artifact.write_bytes(b"opaque artifact")
        completion.append(
            {
                "run_id": cell["run_id"], "path": cell["path"],
                "expected_config_hash": _sha("c"),
                "status_sha256": _file_sha(status),
                "artifact_manifest_sha256": _file_sha(artifact),
            }
        )
        runs.append(
            {
                "run_id": cell["run_id"], "path": cell["path"],
                "run_spec": cell["run_spec"],
                "scientific_config_hash": cell["scientific_config_hash"],
                "expected_config_hash": _sha("c"), "expected_artifacts": {},
            }
        )
    source_payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_stage_v1",
        "status": "APPROVED_FROZEN", "stage": "formal256_primary",
        "authorization": {"fixture": True},
        "formal_campaign": campaign_identity,
        "run_count": 40, "runs": runs,
    }
    source_payload["semantic_sha256"] = canonical_hash(source_payload)
    source_path = root / "manifests" / "stages" / "formal256_primary.json"
    _write_canonical(source_path, source_payload)
    persisted_source_payload = json.loads(source_path.read_bytes())
    source_identity = {
        "path": str(source_path), "sha256": _file_sha(source_path),
        "semantic_sha256": source_payload["semantic_sha256"],
    }
    summarizer_path = Path("experiments/summarize_soft_loss_v3.py").resolve()
    report: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_completion_report_v1",
        "status": "COMPLETE_ONLY", "source_stage": "formal256_primary",
        "next_stage": "formal256_repeat",
        "source_stage_manifest": source_identity,
        "formal_campaign": campaign_identity,
        "registered_run_count": 40, "completed_run_count": 40,
        "completion": completion, "metric_values_parsed": False,
        "failure_codes": [],
        "policy": {
            "completion_only": True, "primary_outcomes_opaque": True,
            "repeats_inference_eligible": False,
        },
        "summarizer_source": {
            "path": str(summarizer_path), "sha256": _file_sha(summarizer_path),
        },
    }
    report["semantic_sha256"] = canonical_hash(report)
    report_bytes = runner_module.canonical_json_bytes(report)
    hashes = {
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "summarizer_source_sha256": _file_sha(summarizer_path),
        "master_manifest_sha256": str(master["sha256"]),
        "development_suite_sha256": str(development["sha256"]),
        "formal_suite_sha256": str(formal["sha256"]),
        "specification_sha256": str(specification["sha256"]),
        "frozen_weights_sha256": str(weights["sha256"]),
        "source_stage_manifest_sha256": str(source_identity["sha256"]),
        "formal_campaign_sha256": str(campaign_identity["sha256"]),
    }
    context = {
        "master": master, "development": development, "formal": formal,
        "specification": specification,
        "weights": {**weights, "status": "APPROVED_FROZEN", "lambdas": {"lambda_dic": 0.25, "lambda_dts": 0.5}},
    }
    campaign_snapshot = registry_module._snapshot_file(campaign_path, label="campaign", root=root)
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda stage, supplied: context)
    monkeypatch.setattr(runner_module, "_validate_source_stage_identity", lambda value, **kwargs: (dict(value), persisted_source_payload))
    monkeypatch.setattr(runner_module, "_load_formal_campaign_identity", lambda path, output_root: (campaign_identity, persisted_campaign, campaign_snapshot))
    monkeypatch.setattr(runner_module, "_validate_completion_only_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner_module, "_validate_completion_only", lambda *args, **kwargs: None)
    target = root / "authorizations" / "formal256_primary__to__formal256_repeat.json"
    forged = deepcopy(report)
    forged["completion"][0]["status_sha256"] = _sha("0")
    forged["semantic_sha256"] = canonical_hash(
        {key: value for key, value in forged.items() if key != "semantic_sha256"}
    )
    forged_hashes = dict(hashes)
    forged_hashes["report_sha256"] = hashlib.sha256(
        runner_module.canonical_json_bytes(forged)
    ).hexdigest()
    with pytest.raises(ValueError, match="current|recompute|completion|report"):
        write_gate_authorization(
            target,
            next_stage="formal256_repeat",
            report=forged,
            hashes=forged_hashes,
        )
    assert not target.exists()
    assert not (root / "analysis" / "formal_primary_completion.json").exists()
    live_status = Path(completion[0]["path"]) / "STATUS.json"
    original_status = live_status.read_bytes()
    live_status.write_bytes(original_status + b" changed")
    with pytest.raises(ValueError, match="current|recompute|completion|report"):
        write_gate_authorization(
            target,
            next_stage="formal256_repeat",
            report=report,
            hashes=hashes,
        )
    assert not target.exists()
    assert not (root / "analysis" / "formal_primary_completion.json").exists()
    live_status.write_bytes(original_status)
    assert write_gate_authorization(
        target, next_stage="formal256_repeat", report=report, hashes=hashes
    ) == target
    authorization = json.loads(target.read_bytes())
    assert authorization["status"] == "COMPLETE_ONLY"
    assert authorization["completion"] == completion
    accepted = runner_module._validate_authorization(
        "formal256_repeat",
        runner_module.StageContext(
            output_root=root,
            master_manifest=root / "experiment_manifest.json",
            development_suite=root / "manifests" / "development_suite.json",
            formal_suite=root / "manifests" / "formal_suite.json",
            frozen_weights=root / "analysis" / "weights_frozen.json",
            authorization=target,
            formal_campaign=campaign_path,
        ),
        output_root=root,
        master_identity=master,
        development_identity=development,
        formal_identity=formal,
        specification_identity=specification,
        weight_identity={key: context["weights"][key] for key in ("path", "sha256", "semantic_sha256")},
        campaign_identity=campaign_identity,
        campaign_payload=campaign,
    )
    assert accepted["status"] == "COMPLETE_ONLY"


def test_gate_writer_rejects_programmer_types_before_any_write(tmp_path: Path) -> None:
    target = tmp_path / "auth.json"
    with pytest.raises(ValueError):
        write_gate_authorization(target, next_stage=1, report={}, hashes={})
    assert not target.exists()


def test_collect_stage_rows_rejects_wrong_types_unregistered_stage_and_relative_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        collect_stage_rows(str(tmp_path), "smoke128")
    with pytest.raises(ValueError):
        collect_stage_rows(tmp_path, "all")
    with pytest.raises(ValueError):
        collect_stage_rows(Path("relative"), "smoke128")
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(canonical, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|canonical"):
        collect_stage_rows(alias.absolute(), "smoke128")


def test_cli_verify_existing_analysis_fails_closed_until_task13_and_writes_nothing(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    analysis = root / "analysis"
    analysis.mkdir()
    manifest = analysis / "analysis_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="Task 13|not installed|validator"):
        summary_main(["--stage", "formal256_repeat", "--output-root", str(root), "--analysis-dir", str(analysis), "--verify-existing-analysis", str(manifest)])
    after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert after == before


def _cli_row(stage: str, index: int) -> dict[str, object]:
    variant = PRIMARY[index % len(PRIMARY)]
    steps = {
        "smoke128": 200,
        "screen128": 1000,
        "confirm128": 2000,
        "cuda2step256": 2,
        "resource256": 200,
        "transfer256": 2000,
        "sentinel_repeat256": 2000,
        "formal256_primary": 2000,
        "formal256_repeat": 2000,
        "appendix256": 2000,
    }[stage]
    return {
        "stage": stage,
        "run_id": f"{stage}/run_{index:02d}",
        "path": f"/tmp/{stage}/run_{index:02d}",
        "task": "arcane_casters",
        "variant": variant,
        "scientific_config_hash": _sha(str((index % 9) + 1)),
        "expected_config_hash": _sha("c"),
        "steps": steps,
        "run_spec": {"profile": variant},
        "parameter_change": {
            "reference_probability": 0.01, "max_abs_delta": 0.1,
            "changed": True,
        },
        "status": {"sha256": _sha("1")},
        "artifact_manifest": {
            "sha256": _sha("2"),
            "payload": {"artifacts_sha256": {"occupancy_probability.npy": _sha("3")}},
        },
        "metrics": {"sha256": _sha("4"), "payload": {}},
        "gradient_audit": {"sha256": _sha("5"), "payload": {"samples": []}},
        "resource_metrics": {
            "sha256": _sha("6"),
            "payload": {"physical_device_memory_bytes": 1000},
        },
        "history": {"sha256": _sha("7"), "rows": []},
    }


def _cli_files(root: Path) -> dict[str, Path]:
    paths = {
        "master": root / "experiment_manifest.json",
        "weights": root / "analysis" / "weights_frozen.json",
        "development": root / "manifests" / "development_suite.json",
        "formal": root / "manifests" / "formal_suite.json",
        "campaign": root / "manifests" / "formal_campaign.json",
    }
    for path in paths.values():
        _write_canonical(path, {})
    counts = {
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
    for stage, count in counts.items():
        registrations = [
            {
                "run_id": row["run_id"],
                "path": row["path"],
                "expected_config_hash": row["expected_config_hash"],
            }
            for row in (_cli_row(stage, index) for index in range(count))
        ]
        payload: dict[str, object] = {
            "schema": "pcraft_soft_loss_v3_stage_v1",
            "status": "APPROVED_FROZEN",
            "stage": stage,
            "runs": registrations,
        }
        payload["semantic_sha256"] = canonical_hash(payload)
        _write_canonical(
            root / "manifests" / "stages" / f"{stage}.json",
            payload,
        )
    (root / "analysis" / "appendix").mkdir(parents=True, exist_ok=True)
    return paths


def _ordinary_cli_argv(
    root: Path, paths: Mapping[str, Path], stage: str, next_stage: str
) -> list[str]:
    argv = [
        "--stage", stage,
        "--output-root", str(root),
        "--stage-manifest", str(root / "manifests" / "stages" / f"{stage}.json"),
        "--master-manifest", str(paths["master"]),
        "--weights", str(paths["weights"]),
        "--development-suite", str(paths["development"]),
        "--next-stage", next_stage,
    ]
    if stage == "resource256":
        argv.extend(
            [
                "--integration-stage-manifest",
                str(root / "manifests" / "stages" / "cuda2step256.json"),
            ]
        )
    if stage == "sentinel_repeat256":
        argv.extend(
            [
                "--reference-stage-manifest",
                str(root / "manifests" / "stages" / "transfer256.json"),
            ]
        )
    return argv


@pytest.mark.parametrize(
    ("stage", "next_stage", "extra", "expected_write"),
    [
        ("smoke128", "screen128", [], True),
        ("screen128", "confirm128", [], True),
        ("confirm128", "cuda2step256", [], True),
        ("cuda2step256", "resource256", [], True),
        ("resource256", "transfer256", ["integration"], True),
        ("transfer256", "sentinel_repeat256", [], True),
        ("sentinel_repeat256", "formal256_primary", ["reference"], True),
        ("formal256_primary", "formal256_repeat", ["completion"], True),
        ("formal256_repeat", None, ["repeat_collection"], False),
        ("appendix256", None, ["exploratory"], False),
    ],
)
def test_cli_legal_stage_shapes_dispatch_without_auto_chaining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    next_stage: str | None,
    extra: list[str],
    expected_write: bool,
) -> None:
    root = (tmp_path / "cli-root").resolve()
    root.mkdir()
    paths = _cli_files(root)
    counts = {
        "smoke128": 4, "screen128": 12, "confirm128": 12,
        "cuda2step256": 4, "resource256": 2, "transfer256": 12,
        "sentinel_repeat256": 4, "formal256_primary": 40,
        "formal256_repeat": 40, "appendix256": 30,
    }
    collections: list[str] = []

    def collect(root_arg: Path, stage_arg: str) -> list[dict[str, object]]:
        collections.append(stage_arg)
        return [_cli_row(stage_arg, index) for index in range(counts[stage_arg])]

    def passing(gate: str) -> dict[str, object]:
        return {
            "schema": "pcraft_soft_loss_v3_gate_result_v1",
            "gate": gate, "status": "PASS", "criteria": {},
            "observed": {}, "failure_codes": [],
        }

    monkeypatch.setattr(summary_module, "collect_stage_rows", collect)
    monkeypatch.setattr(summary_module, "gradient_budget_gate", lambda *args, **kwargs: passing("gradient_budget"))
    monkeypatch.setattr(summary_module, "development_factorial_gate", lambda *args, **kwargs: passing("development_factorial"))
    monkeypatch.setattr(summary_module, "_cuda_integration_gate", lambda *args, **kwargs: passing("cuda_integration"))
    monkeypatch.setattr(summary_module, "resource_gate", lambda *args, **kwargs: passing("resource_feasibility"))
    monkeypatch.setattr(summary_module, "repeat_tolerance_gate", lambda *args, **kwargs: passing("repeat_tolerance"))
    completion_report = {
        "schema": "pcraft_soft_loss_v3_completion_report_v1",
        "status": "COMPLETE_ONLY", "source_stage": "formal256_primary",
        "next_stage": "formal256_repeat", "source_stage_manifest": {},
        "formal_campaign": {}, "registered_run_count": 40,
        "completed_run_count": 40, "completion": [],
        "metric_values_parsed": False, "failure_codes": [], "policy": {},
        "summarizer_source": {}, "semantic_sha256": _sha("a"),
    }
    monkeypatch.setattr(summary_module, "formal_primary_completion_gate", lambda campaign: completion_report)
    writes: list[tuple[Path, str, dict[str, object]]] = []
    monkeypatch.setattr(summary_module, "_publication_hashes", lambda root, report: {})
    monkeypatch.setattr(
        summary_module,
        "write_gate_authorization",
        lambda path, *, next_stage, report, hashes: (
            writes.append((path, next_stage, deepcopy(report))) or path
        ),
    )

    argv = [
        "--stage", stage,
        "--output-root", str(root),
        "--stage-manifest", str(root / "manifests" / "stages" / f"{stage}.json"),
        "--master-manifest", str(paths["master"]),
        "--weights", str(paths["weights"]),
    ]
    if stage in {"formal256_primary", "formal256_repeat", "appendix256"}:
        argv.extend(["--formal-suite", str(paths["formal"])])
    else:
        argv.extend(["--development-suite", str(paths["development"])])
    if next_stage is not None:
        argv.extend(["--next-stage", next_stage])
    if "integration" in extra:
        argv.extend(["--integration-stage-manifest", str(root / "manifests" / "stages" / "cuda2step256.json")])
    if "reference" in extra:
        argv.extend(["--reference-stage-manifest", str(root / "manifests" / "stages" / "transfer256.json")])
    if "completion" in extra:
        argv.extend(["--formal-campaign", str(paths["campaign"]), "--completion-only"])
    if "repeat_collection" in extra:
        argv.extend(
            [
                "--formal-campaign", str(paths["campaign"]),
                "--primary-stage-manifest", str(root / "manifests" / "stages" / "formal256_primary.json"),
                "--analysis-dir", str(root / "analysis"),
            ]
        )
    if "exploratory" in extra:
        argv.extend(["--analysis-dir", str(root / "analysis" / "appendix"), "--exploratory-only"])
    assert summary_main(argv) == 0
    assert bool(writes) is expected_write
    assert len(collections) <= 2
    if stage not in {"formal256_primary"}:
        assert stage in collections
    if stage in {
        "smoke128", "screen128", "confirm128", "cuda2step256",
        "resource256", "transfer256", "sentinel_repeat256",
    }:
        expected_binding = {
            "smoke128": ("gradient_budget", "gradient_budget", None),
            "screen128": ("development_factorial", "development_factorial", None),
            "confirm128": ("development_factorial", "development_factorial", None),
            "cuda2step256": ("cuda_integration", "cuda_integration", None),
            "resource256": ("resource_feasibility", "resource_feasibility", "cuda_integration"),
            "transfer256": ("development_factorial", "development_factorial", "gradient_budget"),
            "sentinel_repeat256": ("repeat_tolerance", "repeat_tolerance", None),
        }[stage]
        report = writes[0][2]
        primary = report["checks"]["primary_gate"]
        additional = report["checks"]["additional_gate"]
        assert report["gate"] == expected_binding[0]
        assert primary["gate"] == expected_binding[1]
        assert (
            None if additional is None else additional["gate"]
        ) == expected_binding[2]
        assert report["source_stage"] == stage
        assert report["next_stage"] == next_stage
        assert report["status"] == "ADVANCE"
        assert report["registered_run_count"] == counts[stage]
        assert report["completed_run_count"] == counts[stage]
        assert report["checks"]["collection"] == {
            "status": "PASS",
            "expected_count": counts[stage],
            "completed_count": counts[stage],
            "parameter_changed_count": counts[stage],
            "invalid_run_ids": [],
            "failure_codes": [],
        }
    elif stage == "formal256_primary":
        assert writes[0][2] == completion_report


@pytest.mark.parametrize(
    ("case", "stage", "next_stage", "failed_gate", "failure_code"),
    [
        ("primary", "smoke128", "screen128", "gradient_budget", "component_ratio_out_of_range"),
        ("primary", "screen128", "confirm128", "development_factorial", "dic_threshold_missed"),
        ("primary", "confirm128", "cuda2step256", "development_factorial", "dts_threshold_missed"),
        ("primary", "cuda2step256", "resource256", "cuda_integration", "integration_branch_inactive"),
        ("primary", "resource256", "transfer256", "resource_feasibility", "runtime_ratio_exceeded"),
        ("additional", "resource256", "transfer256", "cuda_integration", "integration_nonfinite"),
        ("primary", "transfer256", "sentinel_repeat256", "development_factorial", "iou_noninferiority_failed"),
        ("additional", "transfer256", "sentinel_repeat256", "gradient_budget", "combined_ratio_p95_exceeded"),
        ("primary", "sentinel_repeat256", "formal256_primary", "repeat_tolerance", "repeat_mscd_tolerance_exceeded"),
    ],
)
def test_cli_all_seven_ordinary_edges_publish_stop_when_any_required_conjunct_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    stage: str,
    next_stage: str,
    failed_gate: str,
    failure_code: str,
) -> None:
    root = (tmp_path / f"{stage}-{case}").resolve()
    root.mkdir()
    paths = _cli_files(root)
    counts = {
        "smoke128": 4,
        "screen128": 12,
        "confirm128": 12,
        "cuda2step256": 4,
        "resource256": 2,
        "transfer256": 12,
        "sentinel_repeat256": 4,
    }

    def collect(root_arg: Path, stage_arg: str) -> list[dict[str, object]]:
        return [_cli_row(stage_arg, index) for index in range(counts[stage_arg])]

    def outcome(gate: str, *, position: str = "primary") -> dict[str, object]:
        should_stop = gate == failed_gate and position == case
        return {
            "schema": "pcraft_soft_loss_v3_gate_result_v1",
            "gate": gate,
            "status": "STOP" if should_stop else "PASS",
            "criteria": {},
            "observed": {},
            "failure_codes": [failure_code] if should_stop else [],
        }

    monkeypatch.setattr(summary_module, "collect_stage_rows", collect)
    monkeypatch.setattr(
        summary_module,
        "development_factorial_gate",
        lambda *args, **kwargs: outcome("development_factorial"),
    )
    monkeypatch.setattr(
        summary_module,
        "gradient_budget_gate",
        lambda *args, **kwargs: outcome(
            "gradient_budget",
            position="additional" if stage == "transfer256" else "primary",
        ),
    )
    monkeypatch.setattr(
        summary_module,
        "_cuda_integration_gate",
        lambda *args, **kwargs: outcome(
            "cuda_integration",
            position="additional" if stage == "resource256" else "primary",
        ),
    )
    monkeypatch.setattr(
        summary_module,
        "resource_gate",
        lambda *args, **kwargs: outcome("resource_feasibility"),
    )
    monkeypatch.setattr(
        summary_module,
        "repeat_tolerance_gate",
        lambda *args, **kwargs: outcome("repeat_tolerance"),
    )
    monkeypatch.setattr(summary_module, "_publication_hashes", lambda root, report: {})
    writes: list[dict[str, object]] = []
    monkeypatch.setattr(
        summary_module,
        "write_gate_authorization",
        lambda path, *, next_stage, report, hashes: (
            writes.append(deepcopy(report)) or path
        ),
    )
    assert summary_main(_ordinary_cli_argv(root, paths, stage, next_stage)) == 1
    assert len(writes) == 1
    report = writes[0]
    assert report["status"] == "STOP"
    assert report["failure_codes"] == [failure_code]
    assert report["registered_run_count"] == counts[stage]
    assert report["completed_run_count"] == counts[stage]
    primary = report["checks"]["primary_gate"]
    additional = report["checks"]["additional_gate"]
    stopped = additional if case == "additional" else primary
    assert stopped["gate"] == failed_gate
    assert stopped["status"] == "STOP"


def _registered_report_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> tuple[Path, dict[str, object]]:
    root = (tmp_path / f"recompute-{stage}").resolve()
    root.mkdir()
    prerequisite_rows: list[dict[str, object]] | None = None
    if stage == "smoke128":
        rows = [
            _collected_row(
                stage=stage,
                task="arcane_casters",
                variant=variant,
                resolution=128,
                steps=200,
            )
            for variant in PRIMARY
        ]
        probes = _gradient_envelopes(128)
        for row in rows:
            row["gradient_audit"]["payload"]["samples"] = [
                deepcopy(item["sample"])
                for item in probes
                if item["variant"] == row["variant"]
            ]
    elif stage in {"screen128", "confirm128", "transfer256"}:
        resolution = 256 if stage == "transfer256" else 128
        steps = 1000 if stage == "screen128" else 2000
        rows = _development_rows(stage=stage, resolution=resolution, steps=steps)
        if stage == "transfer256":
            probes = _gradient_envelopes(256)
            for row in rows:
                row["gradient_audit"]["payload"]["samples"] = [
                    deepcopy(item["sample"])
                    for item in probes
                    if item["variant"] == row["variant"]
                ]
    elif stage == "cuda2step256":
        rows = _integration_rows()
    elif stage == "resource256":
        rows = _resource_rows()
        prerequisite_rows = _integration_rows()
    else:
        assert stage == "sentinel_repeat256"
        primary, rows = _repeat_rows()
        prerequisite_rows = primary

    registrations = [
        {
            "run_id": row["run_id"],
            "path": row["path"],
            "expected_config_hash": row["expected_config_hash"],
        }
        for row in rows
    ]
    stage_payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_stage_v1",
        "status": "APPROVED_FROZEN",
        "stage": stage,
        "runs": registrations,
    }
    stage_payload["semantic_sha256"] = canonical_hash(stage_payload)
    _write_canonical(
        root / "manifests" / "stages" / f"{stage}.json", stage_payload
    )
    monkeypatch.setattr(
        runner_module,
        "_current_stage_identity",
        lambda current_root, prerequisite: _identity(
            str(current_root / "manifests" / "stages" / f"{prerequisite}.json"),
            "9",
            semantic=True,
        ),
    )

    def collect(current_root: Path, requested: str) -> list[dict[str, object]]:
        assert current_root == root
        if requested == stage:
            return rows
        if stage == "resource256" and requested == "cuda2step256":
            assert prerequisite_rows is not None
            return prerequisite_rows
        if stage == "sentinel_repeat256" and requested == "transfer256":
            assert prerequisite_rows is not None
            return prerequisite_rows
        raise AssertionError(f"unexpected collection {requested}")

    monkeypatch.setattr(summary_module, "collect_stage_rows", collect)
    return root, summary_module._ordinary_report(
        root,
        stage,
        rows=rows,
        collection_failures=[],
        invalid_run_ids=[],
        prerequisite_rows=prerequisite_rows,
    )


@pytest.mark.parametrize(
    ("stage", "slot", "path", "forged_value"),
    [
        ("smoke128", "primary_gate", ("observed", "dic_ratio_median"), -999.0),
        ("screen128", "primary_gate", ("observed", "dic_task_equal_effect"), -999.0),
        ("confirm128", "primary_gate", ("observed", "dts_task_equal_effect"), -999.0),
        ("cuda2step256", "primary_gate", ("observed", "branch_checks", "l12_bal", "finite_gradients"), False),
        ("resource256", "primary_gate", ("observed", "full_to_baseline_p95_ratio"), 999.0),
        ("resource256", "additional_gate", ("observed", "branch_checks", "l12_bal", "finite_gradients"), False),
        ("transfer256", "primary_gate", ("observed", "min_task_recall_delta"), -999.0),
        ("transfer256", "additional_gate", ("observed", "combined_ratio_p95"), 999.0),
        ("sentinel_repeat256", "primary_gate", ("observed", "max_abs_task_mscd_px"), 999.0),
    ],
)
def test_current_evidence_recomputation_rejects_forged_pass_for_every_gate_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    slot: str,
    path: tuple[str, ...],
    forged_value: object,
) -> None:
    root, report = _registered_report_fixture(tmp_path, monkeypatch, stage)
    assert report["status"] == "ADVANCE"
    forged = deepcopy(report)
    cursor: dict[str, object] = forged["checks"][slot]
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = forged_value
    forged["semantic_sha256"] = canonical_hash(
        {key: value for key, value in forged.items() if key != "semantic_sha256"}
    )
    with pytest.raises(ValueError, match="current|recomputed|evidence|report"):
        summary_module._recompute_publication_report(
            root, forged, campaign_payload=None
        )


def test_cli_rejects_irrelevant_options_and_wrong_next_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = (tmp_path / "cli-invalid").resolve()
    root.mkdir()
    paths = _cli_files(root)
    writes: list[object] = []
    monkeypatch.setattr(summary_module, "write_gate_authorization", lambda *args, **kwargs: writes.append(args))
    base = [
        "--stage", "smoke128", "--output-root", str(root),
        "--stage-manifest", str(root / "manifests" / "stages" / "smoke128.json"),
        "--master-manifest", str(paths["master"]), "--weights", str(paths["weights"]),
        "--development-suite", str(paths["development"]),
    ]
    with pytest.raises(ValueError, match="next|stage|option"):
        summary_main([*base, "--next-stage", "transfer256"])
    with pytest.raises(ValueError, match="irrelevant|option|formal"):
        summary_main([*base, "--next-stage", "screen128", "--formal-campaign", str(paths["campaign"])])
    assert writes == []


def _screen_cli_argv(root: Path) -> list[str]:
    return [
        "--stage",
        "screen128",
        "--output-root",
        str(root),
        "--stage-manifest",
        str(root / "manifests" / "stages" / "screen128.json"),
        "--master-manifest",
        str(root / "experiment_manifest.json"),
        "--weights",
        str(root / "analysis" / "weights_frozen.json"),
        "--development-suite",
        str(root / "manifests" / "development_suite.json"),
        "--next-stage",
        "confirm128",
    ]


def test_cli_publishes_authenticated_scientific_stop_but_auth_failure_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, target, _, _, context, _ = _publication_fixture(tmp_path, monkeypatch)
    rows = _development_rows(stage="screen128", resolution=128, steps=1000)
    for row in rows:
        if row["variant"] == "dic_only":
            row["metrics"]["payload"]["task_mscd_px"] = 12.0
    scientific = development_factorial_gate(
        rows,
        dic_relative_threshold=0.025,
        dts_absolute_threshold=0.0005,
        require_nonincreasing_range=False,
    )
    assert scientific["status"] == "STOP"
    assert "dic_conditional_reversed" in scientific["failure_codes"]
    monkeypatch.setattr(summary_module, "collect_stage_rows", lambda root, stage: rows)

    assert summary_main(_screen_cli_argv(root)) == 1
    report_path = root / "analysis" / "screen128_gate.json"
    assert report_path.is_file() and target.is_file()
    report = json.loads(report_path.read_bytes())
    authorization = json.loads(target.read_bytes())
    assert report["status"] == authorization["status"] == "STOP"
    assert report["checks"]["primary_gate"] == scientific
    assert report["failure_codes"] == scientific["failure_codes"]
    with pytest.raises(ValueError, match="STOP|status|advance"):
        _task11_read_authorization(root, target, context)
    frozen_bytes = (report_path.read_bytes(), target.read_bytes())
    assert summary_main(_screen_cli_argv(root)) == 1
    assert (report_path.read_bytes(), target.read_bytes()) == frozen_bytes

    monkeypatch.undo()
    auth_tmp = tmp_path / "authentication-case"
    auth_tmp.mkdir()
    auth_root, auth_target, _, _, _, _ = _publication_fixture(
        auth_tmp, monkeypatch
    )
    monkeypatch.setattr(
        summary_module,
        "collect_stage_rows",
        lambda root, stage: (_ for _ in ()).throw(
            ValueError("authenticated lineage mismatch")
        ),
    )
    with pytest.raises(ValueError, match="lineage"):
        summary_main(_screen_cli_argv(auth_root))
    assert not auth_target.exists()
    assert not (auth_root / "analysis" / "screen128_gate.json").exists()


def test_source_scope_has_no_data_trainer_execution_or_downstream_imports() -> None:
    source = Path("experiments/summarize_soft_loss_v3.py").read_text(encoding="utf-8")
    assert "/data" not in source
    forbidden = ("subprocess", "marching_cubes", "guarded_carving", "core_shell", "PINO", "FEM")
    for token in forbidden:
        assert token not in source
