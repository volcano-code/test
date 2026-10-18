from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, fields
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

import experiments.soft_loss_v3_registry as registry
from shadowforge_pipeline.soft_voxel_v3.provenance import (
    canonical_hash,
    canonical_json_bytes,
)

from experiments import soft_loss_v3_statistics as statistics
from experiments.summarize_soft_loss_v3 import summary_main


TASKS = tuple(f"connected_{index:02d}" for index in range(1, 11))
VARIANTS = ("l12_bal", "dic_only", "dts_only", "v3_full")
ROW_KEYS = {
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


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(path: str, seed: str, *, semantic: bool = True) -> dict[str, object]:
    result: dict[str, object] = {"path": path, "sha256": _sha(seed)}
    if semantic:
        result["semantic_sha256"] = _sha(f"semantic:{seed}")
    return result


def _formal_suite(root: Path) -> dict[str, object]:
    source = _identity(str(root / "benchmark_manifest.json"), "benchmark")
    source["source_path"] = source.pop("path")
    source["resolved_path"] = source["source_path"]
    tasks = []
    for task_index, task in enumerate(TASKS):
        inputs = []
        for axis_index, axis in enumerate("xyz"):
            path = root / "inputs" / f"{task}_view_{axis}.png"
            inputs.append(
                {
                    "axis": axis,
                    "source_path": str(path),
                    "resolved_path": str(path),
                    "sha256": _sha(f"input:{task_index}:{axis_index}"),
                }
            )
        tasks.append({"case_id": task, "inputs": inputs})
    payload: dict[str, object] = {
        "schema": registry.SUITE_SCHEMA,
        "status": registry.STATUS,
        "role": "formal",
        "source_manifest": source,
        "tasks": tasks,
        "task_count": 10,
        "input_count": 30,
        "input_order": ["x", "y", "z"],
        "protocol": registry._protocol(),
        "primary_variants": list(VARIANTS),
        "appendix_controls": registry._appendix_controls(),
        "formal_run_roles": registry._formal_run_roles(),
    }
    payload["semantic_sha256"] = canonical_hash(registry._scientific_suite(payload))
    return payload


def _run_spec(
    root: Path,
    suite: dict[str, object],
    *,
    task: str,
    variant: str,
    role: str,
    index: int,
) -> dict[str, object]:
    stage = "formal256_primary" if role == "primary" else "formal256_repeat"
    run_id = f"{stage}/steps_2000/{task}/{variant}/replicate_{index}/seed_17"
    profile = {
        "l12_bal": (0.35, 1.0, False, False, None),
        "dic_only": (0.35, 1.0, True, False, None),
        "dts_only": (0.35, 1.0, False, True, [0.04, 0.06]),
        "v3_full": (0.35, 1.0, True, True, [0.04, 0.06]),
    }[variant]
    task_inputs = next(item["inputs"] for item in suite["tasks"] if item["case_id"] == task)
    return {
        "stage": stage,
        "task": task,
        "variant": variant,
        "profile": variant,
        "replicate_role": role,
        "replicate_index": index,
        "inputs_xyz": copy.deepcopy(task_inputs),
        "resolution": 256,
        "export_resolution": 256,
        "steps": 2000,
        "output_path": str(root / run_id),
        "run_id": run_id,
        "device": "cuda:0",
        "optimizer": "Adam",
        "learning_rate": 0.007,
        "initialization_logit": -4.0,
        "gradient_clip": 1.0,
        "seed": 17,
        "l1_weight": profile[0],
        "l2_weight": profile[1],
        "use_dic": profile[2],
        "use_dts": profile[3],
        "dic_threshold": 0.05,
        "dts_thresholds": profile[4],
        "dic_schedule": [0.65, 0.80],
        "dts_schedule": [0.50, 0.65],
        "evaluation_thresholds": [0.04, 0.05, 0.06],
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
        "master_identity": _identity(str(root / "experiment_manifest.json"), "master"),
        "suite_identity": {
            "path": str(root / "manifests" / "formal_suite.json"),
            "sha256": _sha("suite-bytes"),
            "semantic_sha256": suite["semantic_sha256"],
        },
        "specification_identity": _identity(str(root / "revised-design.md"), "spec", semantic=False),
        "source_identity": {
            "schema": "pcraft_soft_loss_v3_runner_source_v1",
            "bundle_sha256": _sha("source"),
        },
        "weight_identity": {
            **_identity(str(root / "analysis" / "weights_frozen.json"), "weights"),
            "status": "APPROVED_FROZEN",
            "lambdas": {"lambda_dic": 0.25, "lambda_dts": 0.5},
        },
        "inference_eligible": role == "primary",
    }


def _campaign(root: Path, suite: dict[str, object]) -> dict[str, object]:
    primary = []
    repeats = []
    pairs = []
    for task in TASKS:
        for variant in VARIANTS:
            primary_spec = _run_spec(root, suite, task=task, variant=variant, role="primary", index=0)
            repeat_spec = _run_spec(root, suite, task=task, variant=variant, role="technical_repeat", index=1)
            scientific = _sha(f"scientific:{task}:{variant}")
            first = {
                "run_id": primary_spec["run_id"],
                "path": primary_spec["output_path"],
                "task": task,
                "variant": variant,
                "replicate_role": "primary",
                "replicate_index": 0,
                "inference_eligible": True,
                "scientific_config_hash": scientific,
                "run_spec": primary_spec,
            }
            second = {
                "run_id": repeat_spec["run_id"],
                "path": repeat_spec["output_path"],
                "task": task,
                "variant": variant,
                "replicate_role": "technical_repeat",
                "replicate_index": 1,
                "inference_eligible": False,
                "scientific_config_hash": scientific,
                "run_spec": repeat_spec,
            }
            primary.append(first)
            repeats.append(second)
            pairs.append(
                {
                    "task_variant": [task, variant],
                    "primary_run_id": first["run_id"],
                    "repeat_run_id": second["run_id"],
                    "primary_path": first["path"],
                    "repeat_path": second["path"],
                    "scientific_config_hash": scientific,
                }
            )
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_formal_campaign_v1",
        "status": "APPROVED_FROZEN",
        "primary": primary,
        "repeats": repeats,
        "pairs": pairs,
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    return payload


def _aggregate(*, iou: float, recall: float | None, fpr: float | None) -> dict[str, object]:
    return {
        "tp": 10,
        "fp": 1,
        "fn": 1,
        "tn": 10,
        "mean_iou": iou,
        "mean_dice": iou,
        "mean_fpr": fpr,
        "mean_recall": recall,
        "mean_boundary_f1_2px": 0.9,
        "mean_mscd_px": 1.0,
        "mean_p95_symmetric_distance_px": 2.0,
        "invalid_iou_denominator": False,
        "invalid_dice_denominator": False,
        "invalid_background_denominator": fpr is None,
        "invalid_foreground_denominator": recall is None,
        "invalid_predicted_boundary": False,
        "invalid_target_boundary": False,
        "invalid_task_metric": False,
    }


def _metrics(
    variant: str,
    *,
    mscd: float,
    worst: float,
    iou: float = 0.80,
    recall: float | None = 0.90,
    fpr: float | None = 0.01,
) -> dict[str, object]:
    thresholds = {
        key: {"per_view": [], "aggregate": _aggregate(iou=iou, recall=recall, fpr=fpr)}
        for key in ("0.04", "0.05", "0.06")
    }
    return {
        "schema": "pcraft_soft_loss_v3_metrics_v1",
        "config_hash": _sha(f"config:{variant}"),
        "profile": variant,
        "steps": 2000,
        "final_step_exported": True,
        "final_loss": 1.0,
        "thresholds": thresholds,
        "task_iou_by_threshold": {"0.04": worst, "0.05": iou, "0.06": worst},
        "worst_threshold_iou": worst,
        "threshold_iou_range": iou - worst,
        "task_mscd_px": mscd,
        "task_p95_symmetric_distance_px": mscd + 1.0,
        "invalid_task_metric": False,
    }


def _row(
    root: Path,
    campaign: dict[str, object],
    *,
    task: str,
    variant: str,
    role: str,
    index: int,
    mscd: float,
    worst: float,
    iou: float = 0.80,
    recall: float | None = 0.90,
    fpr: float | None = 0.01,
) -> dict[str, object]:
    cells = campaign["primary"] if role == "primary" else campaign["repeats"]
    cell = next(item for item in cells if item["task"] == task and item["variant"] == variant)
    stage = "formal256_primary" if role == "primary" else "formal256_repeat"
    expected = _sha(f"expected:{role}:{task}:{variant}")
    metrics = _metrics(variant, mscd=mscd, worst=worst, iou=iou, recall=recall, fpr=fpr)
    metrics["config_hash"] = expected
    run_path = Path(str(cell["path"]))
    row = {
        "schema": "pcraft_soft_loss_v3_collected_row_v1",
        "stage": stage,
        "run_id": cell["run_id"],
        "path": cell["path"],
        "task": task,
        "variant": variant,
        "replicate_role": role,
        "replicate_index": index,
        "inference_eligible": role == "primary",
        "resolution": 256,
        "steps": 2000,
        "scientific_config_hash": cell["scientific_config_hash"],
        "expected_config_hash": expected,
        "run_spec": copy.deepcopy(cell["run_spec"]),
        "stage_manifest": _identity(str(root / "manifests" / "stages" / f"{stage}.json"), f"stage:{stage}"),
        "formal_campaign": {
            "path": str(root / "manifests" / "formal_campaign.json"),
            "sha256": _sha("campaign-bytes"),
            "semantic_sha256": campaign["semantic_sha256"],
        },
        "status": {
            "path": str(run_path / "STATUS.json"),
            "sha256": _sha(f"status:{role}:{task}:{variant}"),
            "payload": {
                "schema": "pcraft_soft_loss_v3_status_v1",
                "state": "completed",
                "config_hash": expected,
                "completed_updates": 2000,
                "artifact_manifest_sha256": _sha(f"artifacts:{role}:{task}:{variant}"),
            },
        },
        "artifact_manifest": {
            "path": str(run_path / "artifact_manifest.json"),
            "sha256": _sha(f"artifacts:{role}:{task}:{variant}"),
            "payload": {
                "schema": "pcraft_soft_loss_v3_artifacts_v1",
                "config_hash": expected,
                "artifacts_sha256": {"metrics.json": _sha(f"metrics:{role}:{task}:{variant}")},
            },
        },
        "metrics": {
            "path": str(run_path / "metrics.json"),
            "sha256": _sha(f"metrics:{role}:{task}:{variant}"),
            "payload": metrics,
        },
        "gradient_audit": {
            "path": str(run_path / "gradient_audit.json"),
            "sha256": _sha(f"gradient:{role}:{task}:{variant}"),
            "payload": {},
        },
        "resource_metrics": {
            "path": str(run_path / "resource_metrics.json"),
            "sha256": _sha(f"resource:{role}:{task}:{variant}"),
            "payload": {},
        },
        "history": {
            "path": str(run_path / "history.jsonl"),
            "sha256": _sha(f"history:{role}:{task}:{variant}"),
            "rows": [],
        },
        "parameter_change": {
            "reference_probability": 0.01798621006309986,
            "max_abs_delta": 0.1,
            "changed": True,
        },
    }
    assert set(row) == ROW_KEYS
    return row


def _formal_fixture(root: Path) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    suite = _formal_suite(root)
    campaign = _campaign(root, suite)
    endpoint = {
        "l12_bal": (10.0, 0.70),
        "dic_only": (8.0, 0.71),
        "dts_only": (9.0, 0.74),
        "v3_full": (6.0, 0.78),
    }
    primary = []
    repeat = []
    for task in TASKS:
        for variant in VARIANTS:
            mscd, worst = endpoint[variant]
            primary.append(_row(root, campaign, task=task, variant=variant, role="primary", index=0, mscd=mscd, worst=worst))
            repeat.append(_row(root, campaign, task=task, variant=variant, role="technical_repeat", index=1, mscd=mscd, worst=worst))
    return suite, campaign, primary, repeat


def _matrix(root: Path, *, repeat: bool = False) -> statistics.ValidatedFormalMatrix:
    suite, campaign, primary, repeats = _formal_fixture(root)
    return statistics.validate_formal_matrix(
        repeats if repeat else primary,
        formal_suite=suite,
        formal_campaign=campaign,
        replicate_role="technical_repeat" if repeat else "primary",
        replicate_index=1 if repeat else 0,
    )


def test_formal_matrix_accepts_only_exact_registered_40_rows_and_is_deep_frozen(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    suite, campaign, rows, _ = _formal_fixture(root)
    matrix = statistics.validate_formal_matrix(
        rows,
        formal_suite=suite,
        formal_campaign=campaign,
        replicate_role="primary",
        replicate_index=0,
    )
    assert matrix.row_count == 40
    assert matrix.tasks == TASKS
    assert matrix.variants == VARIANTS
    before = statistics.factorial_task_contrasts(matrix)[0]
    rows[0]["metrics"]["payload"]["task_mscd_px"] = 999.0
    assert statistics.factorial_task_contrasts(matrix)[0] == before
    with pytest.raises((AttributeError, FrozenInstanceError)):
        matrix.row_count = 39
    with pytest.raises(ValueError, match="construct|validator"):
        statistics.ValidatedFormalMatrix()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "extra",
        "reordered",
        "wrong_role",
        "wrong_index",
        "wrong_stage",
        "wrong_eligibility",
        "wrong_path",
        "wrong_scientific",
        "wrong_config",
        "wrong_input",
        "mixed_root",
        "extra_key",
    ],
)
def test_formal_matrix_rejects_every_registration_and_evidence_mutation(tmp_path: Path, mutation: str) -> None:
    root = tmp_path.resolve()
    suite, campaign, rows, _ = _formal_fixture(root)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "extra":
        rows.append(copy.deepcopy(rows[-1]))
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "wrong_role":
        rows[0]["replicate_role"] = "technical_repeat"
    elif mutation == "wrong_index":
        rows[0]["replicate_index"] = 1
    elif mutation == "wrong_stage":
        rows[0]["stage"] = "formal256_repeat"
    elif mutation == "wrong_eligibility":
        rows[0]["inference_eligible"] = False
    elif mutation == "wrong_path":
        rows[0]["path"] = str(root / "elsewhere")
    elif mutation == "wrong_scientific":
        rows[0]["scientific_config_hash"] = _sha("wrong")
    elif mutation == "wrong_config":
        rows[0]["status"]["payload"]["config_hash"] = _sha("wrong")
    elif mutation == "wrong_input":
        rows[0]["run_spec"]["inputs_xyz"][0]["sha256"] = _sha("wrong")
    elif mutation == "mixed_root":
        rows[1]["formal_campaign"]["path"] = str(root / "other" / "formal_campaign.json")
    else:
        rows[0]["invented"] = True
    with pytest.raises(ValueError):
        statistics.validate_formal_matrix(
            rows,
            formal_suite=suite,
            formal_campaign=campaign,
            replicate_role="primary",
            replicate_index=0,
        )


def test_formal_matrix_rejects_extra_nested_collected_row_wrapper_key(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    suite, campaign, rows, _ = _formal_fixture(root)
    rows[0]["gradient_audit"]["invented"] = True
    with pytest.raises(ValueError, match="extra|missing|keys"):
        statistics.validate_formal_matrix(rows, formal_suite=suite, formal_campaign=campaign, replicate_role="primary", replicate_index=0)


def test_factorial_contrasts_use_absolute_pixel_improvement_and_exact_interactions(tmp_path: Path) -> None:
    contrast = statistics.factorial_task_contrasts(_matrix(tmp_path.resolve()))[0]
    assert [field.name for field in fields(contrast)] == [
        "task",
        "dic_marginal_mscd_improvement_px",
        "dts_marginal_worst_threshold_iou_improvement",
        "m_d_interaction_px",
        "m_t_interaction",
        "dic_base_to_dic_mscd_improvement_px",
        "dic_dts_to_full_mscd_improvement_px",
        "dts_base_to_dts_worst_threshold_iou_improvement",
        "dts_dic_to_full_worst_threshold_iou_improvement",
    ]
    assert contrast.dic_base_to_dic_mscd_improvement_px == 2.0
    assert contrast.dic_dts_to_full_mscd_improvement_px == 3.0
    assert contrast.dic_marginal_mscd_improvement_px == 2.5
    assert contrast.m_d_interaction_px == 1.0
    assert contrast.dts_base_to_dts_worst_threshold_iou_improvement == pytest.approx(0.04)
    assert contrast.dts_dic_to_full_worst_threshold_iou_improvement == pytest.approx(0.07)
    assert contrast.dts_marginal_worst_threshold_iou_improvement == pytest.approx(0.055)
    assert contrast.m_t_interaction == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 2, 3, 4], 0.0625), ([-1, -2, -3, -4], 1.0), ([0, 0, 0], 1.0)],
)
def test_exact_sign_flip_is_one_sided_enumeration(values: list[float], expected: float) -> None:
    assert statistics.exact_sign_flip_greater(values) == expected


@pytest.mark.parametrize("values", [[], [True], [math.inf], [1.0] * 21])
def test_exact_sign_flip_rejects_invalid_domain(values: list[object]) -> None:
    with pytest.raises(ValueError):
        statistics.exact_sign_flip_greater(values)


def test_paired_bootstrap_uses_fixed_rng_and_handwritten_hf7() -> None:
    result = statistics.paired_bootstrap(range(1, 11), draws=11, seed=20260826)
    assert result == {"mean": 5.5, "ci_low": 3.625, "ci_high": 6.425}
    with pytest.raises(ValueError):
        statistics.paired_bootstrap([1.0, True])


def test_holm_is_exact_two_member_family_with_fixed_tie_order() -> None:
    assert statistics.holm_adjust({"dic": 0.01, "dts": 0.03}) == {"dic": 0.02, "dts": 0.03}
    assert statistics.holm_adjust({"dic": 0.03, "dts": 0.03}) == {"dic": 0.06, "dts": 0.06}
    with pytest.raises(ValueError):
        statistics.holm_adjust({"dic": 0.01})
    with pytest.raises(ValueError):
        statistics.holm_adjust({"dic": False, "dts": 0.2})


def test_formal_inference_generates_one_joint_index_matrix_for_all_eight_estimands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    original = statistics._bootstrap_index_matrix

    def counted(*, draws: int, n_tasks: int, seed: int) -> np.ndarray:
        calls.append((draws, n_tasks, seed))
        return original(draws=draws, n_tasks=n_tasks, seed=seed)

    monkeypatch.setattr(statistics, "_bootstrap_index_matrix", counted)
    inference = statistics.formal_factorial_inference(_matrix(tmp_path.resolve()), draws=257)
    assert calls == [(257, 10, 20260826)]
    assert inference["schema"] == "pcraft_soft_loss_v3_formal_inference_v1"
    assert inference["n_tasks"] == 10
    assert inference["family"] == ["dic", "dts"]
    assert set(inference["hypotheses"]) == {"dic", "dts"}
    assert set(inference["conditional_effects"]) == {"dic", "dts"}
    assert set(inference["interactions"]) == {"m_d", "m_t"}
    assert inference["hypotheses"]["dic"]["mean_effect"] == 2.5
    assert inference["hypotheses"]["dts"]["mean_effect"] == pytest.approx(0.055)
    assert inference["interactions"]["m_d"]["wording"] == "CONDITIONAL_OR_COUPLED_EFFECTS"


def test_formal_inference_rejects_repeat_matrix_and_adjusted_p_equal_alpha_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="primary"):
        statistics.formal_factorial_inference(_matrix(tmp_path.resolve(), repeat=True), draws=31)
    inference = statistics.formal_factorial_inference(_matrix((tmp_path / "primary").resolve()), draws=31)
    inference["hypotheses"]["dic"]["holm_adjusted_p"] = 0.05
    inference["hypotheses"]["dic"]["passed"] = False
    inference["hypotheses"]["dic"]["failure_codes"] = ["holm_not_significant"]
    inference["semantic_sha256"] = canonical_hash(
        {key: value for key, value in inference.items() if key != "semantic_sha256"}
    )
    safety = {
        component: statistics.conditional_safety_checks(_matrix((tmp_path / "primary").resolve()), component)
        for component in ("dic", "dts")
    }
    decision = statistics.retention_decision(inference, safety)
    assert decision["components"]["dic"]["retained"] is False


def test_conditional_safety_uses_two_simple_effects_and_inclusive_guard_boundaries(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    suite, campaign, rows, _ = _formal_fixture(root)
    for row in rows:
        aggregate = row["metrics"]["payload"]["thresholds"]["0.05"]["aggregate"]
        if row["variant"] in {"dic_only", "dts_only"}:
            aggregate["mean_iou"] -= 0.001
            aggregate["mean_recall"] -= 0.005
            aggregate["mean_fpr"] += 0.001
        if row["variant"] == "v3_full":
            aggregate["mean_iou"] -= 0.002
            aggregate["mean_recall"] -= 0.010
            aggregate["mean_fpr"] += 0.002
    matrix = statistics.validate_formal_matrix(rows, formal_suite=suite, formal_campaign=campaign, replicate_role="primary", replicate_index=0)
    assert statistics.conditional_safety_checks(matrix, "dic")["passed"] is True
    assert statistics.conditional_safety_checks(matrix, "dts")["passed"] is True


def test_safety_next_float_failures_negative_fpr_and_single_bad_recall(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    suite, campaign, rows, _ = _formal_fixture(root)
    first_dic = next(row for row in rows if row["task"] == TASKS[0] and row["variant"] == "dic_only")
    first_dic["metrics"]["payload"]["thresholds"]["0.05"]["aggregate"]["mean_recall"] = float(np.nextafter(0.895, -math.inf))
    for row in rows:
        if row["variant"] in {"dic_only", "v3_full"}:
            row["metrics"]["payload"]["thresholds"]["0.05"]["aggregate"]["mean_fpr"] = 0.0
    matrix = statistics.validate_formal_matrix(rows, formal_suite=suite, formal_campaign=campaign, replicate_role="primary", replicate_index=0)
    result = statistics.conditional_safety_checks(matrix, "dic")
    assert result["passed"] is False
    assert "recall_noninferiority_failed" in result["failure_codes"]
    assert "fpr_guard_failed" not in result["failure_codes"]


def test_related_invalid_metric_fails_only_related_component(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    suite, campaign, rows, _ = _formal_fixture(root)
    row = next(item for item in rows if item["task"] == TASKS[0] and item["variant"] == "dic_only")
    aggregate = row["metrics"]["payload"]["thresholds"]["0.05"]["aggregate"]
    aggregate["invalid_predicted_boundary"] = True
    matrix = statistics.validate_formal_matrix(rows, formal_suite=suite, formal_campaign=campaign, replicate_role="primary", replicate_index=0)
    assert statistics.conditional_safety_checks(matrix, "dic")["passed"] is False
    assert statistics.conditional_safety_checks(matrix, "dts")["passed"] is True

    inference = statistics.formal_factorial_inference(matrix, draws=257)
    safety = {
        component: statistics.conditional_safety_checks(matrix, component)
        for component in ("dic", "dts")
    }
    decision = statistics.retention_decision(inference, safety)
    assert decision["components"]["dic"]["retained"] is False
    assert decision["components"]["dts"]["retained"] is True


def test_iou_invalid_fails_both_components_and_repeat_audit(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    suite, campaign, primary_rows, repeat_rows = _formal_fixture(root)
    for rows in (primary_rows, repeat_rows):
        row = next(item for item in rows if item["task"] == TASKS[0] and item["variant"] == "dic_only")
        aggregate = row["metrics"]["payload"]["thresholds"]["0.05"]["aggregate"]
        aggregate["invalid_iou_denominator"] = True
    primary = statistics.validate_formal_matrix(primary_rows, formal_suite=suite, formal_campaign=campaign, replicate_role="primary", replicate_index=0)
    repeat = statistics.validate_formal_matrix(repeat_rows, formal_suite=suite, formal_campaign=campaign, replicate_role="technical_repeat", replicate_index=1)
    assert statistics.conditional_safety_checks(primary, "dic")["passed"] is False
    audit = statistics.formal_repeat_audit(primary, repeat)
    assert audit["status"] == "FAIL"
    assert "relevant_metric_invalid" in audit["failure_codes"]


def test_noncentral_threshold_iou_invalid_fails_only_dts_primary_endpoint(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    suite, campaign, rows, _ = _formal_fixture(root)
    row = next(item for item in rows if item["task"] == TASKS[0] and item["variant"] == "dts_only")
    row["metrics"]["payload"]["thresholds"]["0.04"]["aggregate"]["invalid_iou_denominator"] = True
    matrix = statistics.validate_formal_matrix(
        rows,
        formal_suite=suite,
        formal_campaign=campaign,
        replicate_role="primary",
        replicate_index=0,
    )
    assert statistics.conditional_safety_checks(matrix, "dic")["passed"] is True
    dts = statistics.conditional_safety_checks(matrix, "dts")
    assert dts["passed"] is False
    assert "relevant_metric_invalid" in dts["failure_codes"]


def test_retention_has_strict_ci_and_primary_rules_and_exact_formula(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path.resolve())
    inference = statistics.formal_factorial_inference(matrix, draws=257)
    safety = {name: statistics.conditional_safety_checks(matrix, name) for name in ("dic", "dts")}
    retained = statistics.retention_decision(inference, safety)
    assert retained["retained_components"] == ["dic", "dts"]
    assert retained["retained_formula"]["expression"] == "L0+lambda_I(t)*L_DIC+lambda_T(t)*L_DTS"
    inference["hypotheses"]["dic"]["ci_low"] = 0.0
    inference["hypotheses"]["dic"]["ci_above_zero"] = False
    inference["hypotheses"]["dic"]["passed"] = False
    inference["hypotheses"]["dic"]["failure_codes"] = ["ci_not_above_zero"]
    inference["semantic_sha256"] = canonical_hash({key: value for key, value in inference.items() if key != "semantic_sha256"})
    retained = statistics.retention_decision(inference, safety)
    assert retained["components"]["dic"]["retained"] is False
    assert retained["retained_formula"]["expression"] == "L0+lambda_T(t)*L_DTS"


def test_retention_rejects_extra_nested_inference_and_safety_keys(tmp_path: Path) -> None:
    matrix = _matrix(tmp_path.resolve())
    inference = statistics.formal_factorial_inference(matrix, draws=257)
    safety = {name: statistics.conditional_safety_checks(matrix, name) for name in ("dic", "dts")}
    forged = copy.deepcopy(inference)
    forged["hypotheses"]["dic"]["invented"] = True
    forged["semantic_sha256"] = canonical_hash({key: value for key, value in forged.items() if key != "semantic_sha256"})
    with pytest.raises(ValueError, match="extra|missing|keys"):
        statistics.retention_decision(forged, safety)
    forged_safety = copy.deepcopy(safety)
    forged_safety["dic"]["comparisons"][0]["invented"] = True
    forged_safety["dic"]["semantic_sha256"] = canonical_hash({key: value for key, value in forged_safety["dic"].items() if key != "semantic_sha256"})
    with pytest.raises(ValueError, match="extra|missing|keys"):
        statistics.retention_decision(inference, forged_safety)


def test_interaction_equivalence_changes_wording_but_cannot_rescue_or_remove_retention(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    suite, campaign, rows, _ = _formal_fixture(root)
    for row in rows:
        if row["variant"] == "v3_full":
            row["metrics"]["payload"]["task_mscd_px"] = 7.0
            row["metrics"]["payload"]["worst_threshold_iou"] = 0.75
    matrix = statistics.validate_formal_matrix(rows, formal_suite=suite, formal_campaign=campaign, replicate_role="primary", replicate_index=0)
    inference = statistics.formal_factorial_inference(matrix, draws=257)
    safety = {name: statistics.conditional_safety_checks(matrix, name) for name in ("dic", "dts")}
    retained = statistics.retention_decision(inference, safety)
    assert inference["interactions"]["m_d"]["equivalent"] is True
    assert inference["interactions"]["m_t"]["equivalent"] is True
    assert retained["retained_components"] == ["dic", "dts"]


def test_repeat_audit_uses_ten_plus_ten_without_pooling_and_tolerance_is_inclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = _matrix((tmp_path / "audit").resolve())
    repeat = _matrix((tmp_path / "audit").resolve(), repeat=True)
    lengths = []
    original = statistics.exact_sign_flip_greater

    def counted(values: object) -> float:
        lengths.append(len(values))
        return original(values)

    monkeypatch.setattr(statistics, "exact_sign_flip_greater", counted)
    audit = statistics.formal_repeat_audit(primary, repeat, tolerance=1e-4)
    assert audit["status"] == "PASS"
    assert audit["statistical_units"] == {"primary": 10, "repeat_audit_only": 10, "pooled": None}
    assert lengths and set(lengths) == {10}


def test_repeat_audit_fails_next_tolerance_float_and_retention_flip(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    suite, campaign, primary_rows, repeat_rows = _formal_fixture(root)
    repeat_rows[0]["metrics"]["payload"]["task_mscd_px"] += 0.0001001
    primary = statistics.validate_formal_matrix(primary_rows, formal_suite=suite, formal_campaign=campaign, replicate_role="primary", replicate_index=0)
    repeat = statistics.validate_formal_matrix(repeat_rows, formal_suite=suite, formal_campaign=campaign, replicate_role="technical_repeat", replicate_index=1)
    audit = statistics.formal_repeat_audit(primary, repeat)
    assert audit["status"] == "FAIL"
    assert "repeat_mscd_tolerance_exceeded" in audit["failure_codes"]

    second_root = (tmp_path / "retention_flip").resolve()
    suite, campaign, primary_rows, repeat_rows = _formal_fixture(second_root)
    for row in repeat_rows:
        if row["variant"] in {"dic_only", "v3_full"}:
            row["metrics"]["payload"]["task_mscd_px"] += 5.0
    primary = statistics.validate_formal_matrix(
        primary_rows,
        formal_suite=suite,
        formal_campaign=campaign,
        replicate_role="primary",
        replicate_index=0,
    )
    repeat = statistics.validate_formal_matrix(
        repeat_rows,
        formal_suite=suite,
        formal_campaign=campaign,
        replicate_role="technical_repeat",
        replicate_index=1,
    )
    audit = statistics.formal_repeat_audit(primary, repeat)
    assert audit["primary_component_retention"]["dic"] is True
    assert audit["repeat_component_retention_audit_only"]["dic"] is False
    assert audit["retention_unchanged"] is False
    assert "repeat_retention_changed" in audit["failure_codes"]


def test_export_bundle_has_exact_nine_files_and_no_pooled_inference(tmp_path: Path) -> None:
    primary = _matrix(tmp_path.resolve())
    repeat = _matrix(tmp_path.resolve(), repeat=True)
    inference = statistics.formal_factorial_inference(primary, draws=257)
    safety = {name: statistics.conditional_safety_checks(primary, name) for name in ("dic", "dts")}
    retention = statistics.retention_decision(inference, safety)
    audit = statistics.formal_repeat_audit(primary, repeat)
    exports = statistics._build_export_bytes(primary, repeat, inference, safety, retention, audit)
    assert list(exports) == [
        "task_rows.tsv",
        "factorial_task_contrasts.tsv",
        "confirmatory_tests.json",
        "conditional_effects.json",
        "interactions.json",
        "safety_checks.json",
        "retention_decision.json",
        "repeat_audit.json",
        "aggregate_metrics.json",
    ]
    assert exports["task_rows.tsv"].count(b"\n") == 81
    assert exports["factorial_task_contrasts.tsv"].count(b"\n") == 11
    aggregate = json.loads(exports["aggregate_metrics.json"])
    assert aggregate["primary_n_tasks"] == 10
    assert aggregate["repeat_audit_n_tasks"] == 10
    assert aggregate["pooled_n"] is None
    assert aggregate["inference_source"] == "primary_only"


def test_analysis_publication_precommit_revalidates_full_dependency_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    analysis = root / "analysis"
    analysis.mkdir(parents=True)
    identity = {
        "path": str(root / "placeholder.json"),
        "sha256": _sha("bytes"),
        "semantic_sha256": _sha("semantic"),
    }
    manifest = {
        "formal_campaign": copy.deepcopy(identity),
        "formal_suite": copy.deepcopy(identity),
        "master_manifest": copy.deepcopy(identity),
        "frozen_weights": copy.deepcopy(identity),
    }
    monkeypatch.setattr(
        statistics,
        "_analysis_material",
        lambda output_root: (manifest, {"task_rows.tsv": b"x\n"}, {}, {}, {}),
    )
    monkeypatch.setattr(statistics, "_analysis_source_records", lambda: {})
    calls: list[tuple[object, Path]] = []
    monkeypatch.setattr(
        statistics,
        "_verify_analysis_dependencies",
        lambda payload, *, root: calls.append((payload, root)),
        raising=False,
    )

    def interrupted_publish(files: object, *, output_root: Path, precommit: object) -> None:
        assert callable(precommit)
        precommit()
        precommit()
        raise RuntimeError("stop after precommit probes")

    monkeypatch.setattr(registry, "_publish_bundle", interrupted_publish)
    monkeypatch.setattr(
        registry,
        "_snapshot_file",
        lambda path, *, label, root: type("Snapshot", (), {"sha256": _sha("bytes")})(),
    )
    with pytest.raises(RuntimeError, match="precommit probes"):
        statistics.publish_formal_analysis(root, analysis)
    assert calls == [(manifest, root), (manifest, root)]


def test_conclusion_status_is_derived_from_repeat_and_identical_bytes_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    analysis = root / "analysis"
    analysis.mkdir(parents=True)
    campaign_path = root / "manifests" / "formal_campaign.json"
    campaign_path.parent.mkdir(parents=True)
    campaign_path.write_bytes(b"campaign")
    manifest_path = analysis / "analysis_manifest.json"
    retention_path = analysis / "retention_decision.json"
    repeat_path = analysis / "repeat_audit.json"
    manifest = {
        "schema": "pcraft_soft_loss_v3_analysis_v1",
        "status": "COMPLETE",
        "formal_campaign": {"path": str(campaign_path), "sha256": _sha("campaign"), "semantic_sha256": _sha("campaign-semantic")},
        "run_evidence": [],
        "executed_sources": {},
        "semantic_sha256": _sha("analysis-semantic"),
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    retention = {
        "schema": "pcraft_soft_loss_v3_retention_v1",
        "components": {"dic": {"retained": False}, "dts": {"retained": False}},
        "retained_formula": {"base": "mean_v[L2_bal+0.35*L1_bal]", "dic_enabled": False, "dts_enabled": False, "components": [], "expression": "L0"},
    }
    repeat = {
        "schema": "pcraft_soft_loss_v3_repeat_audit_v1",
        "primary_matrix_sha256": _sha("primary"),
        "repeat_matrix_sha256": _sha("repeat"),
        "pair_count": 40,
        "tolerance": 0.0001,
        "status": "PASS",
        "max_abs_task_mscd_px": 0.0,
        "max_abs_worst_threshold_iou": 0.0,
        "retention_unchanged": True,
        "failure_codes": [],
        "semantic_sha256": _sha("repeat-semantic"),
    }
    retention_path.write_bytes(canonical_json_bytes(retention))
    repeat_path.write_bytes(canonical_json_bytes(repeat))
    monkeypatch.setattr(statistics, "validate_analysis_manifest", lambda path, *, expected_campaign_sha256: manifest)
    conclusion_path = analysis / "formal_conclusion.json"
    statistics.write_formal_conclusion(conclusion_path, analysis_manifest=manifest, retention=retention, repeat_audit=repeat)
    first = conclusion_path.read_bytes()
    payload = json.loads(first)
    assert payload["status"] == "LOCKED"
    assert payload["retained_formula"]["expression"] == "L0"
    statistics.write_formal_conclusion(conclusion_path, analysis_manifest=manifest, retention=retention, repeat_audit=repeat)
    assert conclusion_path.read_bytes() == first
    changed = copy.deepcopy(repeat)
    changed["status"] = "FAIL"
    changed["failure_codes"] = ["repeat_retention_changed"]
    with pytest.raises((FileExistsError, ValueError)):
        statistics.write_formal_conclusion(conclusion_path, analysis_manifest=manifest, retention=retention, repeat_audit=changed)


def test_public_numeric_inputs_reject_bool_nonfinite_extra_and_noncanonical_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        statistics.paired_bootstrap([1.0], draws=True)
    with pytest.raises(ValueError):
        statistics.holm_adjust({"dic": 0.1, "dts": float("nan")})
    with pytest.raises(ValueError):
        statistics.validate_formal_conclusion(
            Path("relative.json"),
            expected_analysis_sha256=_sha("a"),
            expected_campaign_sha256=_sha("b"),
        )

    matrix = _matrix(tmp_path.resolve())
    inference = statistics.formal_factorial_inference(matrix, draws=31)
    safety = {
        component: statistics.conditional_safety_checks(matrix, component)
        for component in ("dic", "dts")
    }
    forged_inference = copy.deepcopy(inference)
    forged_inference["replicate_index"] = False
    forged_inference["semantic_sha256"] = canonical_hash(
        {key: value for key, value in forged_inference.items() if key != "semantic_sha256"}
    )
    with pytest.raises(ValueError):
        statistics.retention_decision(forged_inference, safety)

    forged_safety = copy.deepcopy(safety)
    forged_safety["dic"]["criteria"]["mean_primary_effect_strict_lower"] = False
    forged_safety["dic"]["semantic_sha256"] = canonical_hash(
        {key: value for key, value in forged_safety["dic"].items() if key != "semantic_sha256"}
    )
    with pytest.raises(ValueError):
        statistics.retention_decision(inference, forged_safety)


def test_verify_existing_analysis_rejects_every_irrelevant_common_cli_option(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    analysis = root / "analysis"
    analysis.mkdir()
    for option, value in (
        ("--stage-manifest", root / "manifests" / "stages" / "formal256_repeat.json"),
        ("--master-manifest", root / "experiment_manifest.json"),
        ("--weights", analysis / "weights_frozen.json"),
    ):
        with pytest.raises(ValueError, match="irrelevant"):
            summary_main(
                [
                    "--stage",
                    "formal256_repeat",
                    "--output-root",
                    str(root),
                    "--analysis-dir",
                    str(analysis),
                    "--verify-existing-analysis",
                    str(analysis / "analysis_manifest.json"),
                    option,
                    str(value),
                ]
            )
