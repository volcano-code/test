"""Contracts for the read-only Soft Loss v4 ablation10 analyzer.

Every expectation in this file is hand-derived from the preregistered policy.
The Task 3 completion authenticator is replaced only at the expensive 40-run
boundary; the analyzer's manifest, artifact, metric, decision, and publication
behaviour remains real.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.analyze_soft_loss_v4_ablation10 as analyzer
from shadowforge_pipeline.soft_voxel_v4.provenance import canonical_hash, canonical_json_bytes
from shadowforge_pipeline.soft_voxel_v4.config import canonical_trainer_argv


TASKS = tuple(f"connected_{index:02d}" for index in range(1, 11))
PROFILES = ("l12_bal", "dic_only", "dts_only", "v4_full")
POLICY = {
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


def _write_json(path: Path, payload: dict[str, object], *, canonical: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        canonical_json_bytes(payload) + b"\n"
        if canonical
        else json.dumps(payload, allow_nan=True).encode("utf-8")
    )
    path.write_bytes(encoded)


def _view(tp: int, fp: int, fn: int, tn: int) -> dict[str, object]:
    iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    dice = 1.0 if 2 * tp + fp + fn == 0 else (2 * tp) / (2 * tp + fp + fn)
    fpr = None if fp + tn == 0 else fp / (fp + tn)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": float(iou),
        "dice": float(dice),
        "fpr": None if fpr is None else float(fpr),
        "recall": None if recall is None else float(recall),
        "boundary_f1_2px": 0.9,
        "mscd_px": 0.2,
        "p95_symmetric_distance_px": 0.4,
        "invalid_iou_denominator": False,
        "invalid_dice_denominator": False,
        "invalid_background_denominator": fpr is None,
        "invalid_foreground_denominator": recall is None,
        "invalid_predicted_boundary": False,
        "invalid_target_boundary": False,
    }


def _threshold(views: list[dict[str, object]]) -> dict[str, object]:
    def mean(name: str) -> float:
        return float(sum(float(item[name]) for item in views) / 6.0)

    def optional_mean(name: str) -> float | None:
        if any(item[name] is None for item in views):
            return None
        return mean(name)

    flags = {
        name: any(bool(item[name]) for item in views)
        for name in (
            "invalid_iou_denominator",
            "invalid_dice_denominator",
            "invalid_background_denominator",
            "invalid_foreground_denominator",
            "invalid_predicted_boundary",
            "invalid_target_boundary",
        )
    }
    aggregate = {
        "tp": sum(int(item["tp"]) for item in views),
        "fp": sum(int(item["fp"]) for item in views),
        "fn": sum(int(item["fn"]) for item in views),
        "tn": sum(int(item["tn"]) for item in views),
        "mean_iou": mean("iou"),
        "mean_dice": mean("dice"),
        "mean_fpr": optional_mean("fpr"),
        "mean_recall": optional_mean("recall"),
        "mean_boundary_f1_2px": mean("boundary_f1_2px"),
        "mean_mscd_px": mean("mscd_px"),
        "mean_p95_symmetric_distance_px": mean("p95_symmetric_distance_px"),
        **flags,
        "invalid_task_metric": any(
            flags[name]
            for name in (
                "invalid_background_denominator",
                "invalid_foreground_denominator",
                "invalid_predicted_boundary",
                "invalid_target_boundary",
            )
        ),
    }
    return {"per_view": views, "aggregate": aggregate}


COUNTS = {
    "l12_bal": (940, 40, 60, 960),
    "dic_only": (970, 30, 30, 970),
    "dts_only": (965, 25, 35, 975),
    "v4_full": (980, 20, 20, 980),
}


def _metrics(config_hash: str, profile: str, *, views: list[dict[str, object]] | None = None) -> dict[str, object]:
    primary_views = views or [_view(*COUNTS[profile]) for _ in range(6)]
    reports = {key: _threshold(copy.deepcopy(primary_views)) for key in ("0.04", "0.05", "0.06")}
    task_iou = {key: float(value["aggregate"]["mean_iou"]) for key, value in reports.items()}
    return {
        "schema": "pcraft_soft_loss_v4_metrics_v1",
        "config_hash": config_hash,
        "thresholds": reports,
        "task_iou_by_threshold": task_iou,
        "worst_threshold_iou": min(task_iou.values()),
        "threshold_iou_range": max(task_iou.values()) - min(task_iou.values()),
        "task_mscd_px": float(reports["0.05"]["aggregate"]["mean_mscd_px"]),
        "task_p95_symmetric_distance_px": float(
            reports["0.05"]["aggregate"]["mean_p95_symmetric_distance_px"]
        ),
        "invalid_task_metric": False,
    }


def _audit(config_hash: str, profile: str, *, active: bool = True, fraction: float = 0.0, status: str = "PASS") -> dict[str, object]:
    auxiliary = profile != "l12_bal"
    components = {
        "l12_bal": [],
        "dic_only": ["dic"],
        "dts_only": ["dts"],
        "v4_full": ["dic", "dts"],
    }[profile]
    eligible = len(components)
    gate = None
    samples: list[dict[str, object]] = []
    if auxiliary:
        zero_slots = int(fraction > 0.0)
        gate = {
            "status": status,
            "eligible_slots": eligible,
            "fully_projected_zero_slots": zero_slots,
            "fully_projected_zero_fraction": fraction,
            "failure_codes": [] if status == "PASS" else ["EXCESS_PROJECTION_ZERO"],
        }
        samples = [
            {
                "component": component,
                "final_auxiliary_gradient_norm": 0.01 if active else 0.0,
            }
            for component in components
        ]
    return {
        "schema": "pcraft_soft_loss_v4_gradient_audit_v1",
        "config_hash": config_hash,
        "profile": profile,
        "total_steps": 2000,
        "step_config": {"registered": True},
        "eligible_slots": eligible,
        "samples": samples,
        "engineering_gate": gate,
        "redundancy_gate": None,
    }


def _resolved_config(root: Path, task: str, profile: str) -> dict[str, object]:
    run_id = f"runs/ablation10/{task}/{profile}"
    auxiliary = profile != "l12_bal"
    return {
        "run_kind": "experiment",
        "device": "cpu",
        "output_root_identity": {"path": str(root)},
        "run_identity": {
            "run_id": run_id,
            "run_dir": str(root / run_id),
            "stage": "ablation10_128",
            "task_id": task,
            "replicate_role": "primary",
            "replicate_index": 0,
        },
        "profile": {"name": profile},
        "steps": 2000,
        "learning_rate": 0.007,
        "resolution": 128,
        "export_resolution": 128,
        "seed": 17,
        "shadow_inputs_xyz": [
            {"axis": axis, "path": f"/{task}_{axis}.png", "sha256": str(index) * 64}
            for index, axis in enumerate("xyz", start=1)
        ],
        "suite_manifest": {"path": "/formal.json", "sha256": "a" * 64, "semantic_sha256": "b" * 64, "role": "formal"},
        "specification": {"path": "/spec.md", "sha256": "c" * 64},
        "unit_coefficient_identity": ({"path": "/unit.json", "sha256": "d" * 64, "semantic_sha256": "e" * 64} if auxiliary else None),
        "candidate_registry_identity": ({"path": "/registry.json", "sha256": "f" * 64, "semantic_sha256": "0" * 64} if auxiliary else None),
        "budget_identity": ({"candidate_id": "r08_r08", "order_index": 1, "dic_max_ratio": 0.08, "dts_max_ratio": 0.08} if auxiliary else None),
    }


def _canonical_argv(config: dict[str, object], *, resume: bool) -> list[str]:
    run_identity = config["run_identity"]
    profile = config["profile"]["name"]
    auxiliary = profile != "l12_bal"
    args = argparse.Namespace(
        run_kind=config["run_kind"],
        device=config["device"],
        output_root=config["output_root_identity"]["path"],
        run_id=run_identity["run_id"],
        stage=run_identity["stage"],
        task_id=run_identity["task_id"],
        replicate_role=run_identity["replicate_role"],
        replicate_index=run_identity["replicate_index"],
        steps=config["steps"],
        learning_rate=config["learning_rate"],
        shadow_files=[item["path"] for item in config["shadow_inputs_xyz"]],
        resolution=config["resolution"],
        export_resolution=config["export_resolution"],
        profile=profile,
        suite_manifest=config["suite_manifest"]["path"],
        specification=config["specification"]["path"],
        unit_coefficients=config["unit_coefficient_identity"]["path"] if auxiliary else None,
        candidate_registry=config["candidate_registry_identity"]["path"] if auxiliary else None,
        candidate_id=config["budget_identity"]["candidate_id"] if auxiliary else None,
        resume=False,
        export_final_step=True,
        soft_only_eval=True,
    )
    return list(canonical_trainer_argv(args, resume=resume))


def _campaign(tmp_path: Path) -> tuple[Path, dict[str, dict[str, object]]]:
    root = tmp_path / "campaign"
    root.mkdir(parents=True)
    matrix: list[dict[str, object]] = []
    configs: dict[str, dict[str, object]] = {}
    for task in TASKS:
        for profile in PROFILES:
            run_id = f"runs/ablation10/{task}/{profile}"
            config = _resolved_config(root, task, profile)
            config_hash = canonical_hash(config)
            matrix.append(
                {
                    "task_id": task,
                    "profile": profile,
                    "run_id": run_id,
                    "run_dir": str(root / run_id),
                    "config_hash": config_hash,
                    "fresh_argv": _canonical_argv(config, resume=False),
                    "resume_argv": _canonical_argv(config, resume=True),
                    "resolved_config": config,
                }
            )
            configs[run_id] = config
            run_dir = root / run_id
            _write_json(run_dir / "metrics.json", _metrics(config_hash, profile))
            _write_json(run_dir / "gradient_audit.json", _audit(config_hash, profile))
    manifest: dict[str, object] = {
        "schema": "pcraft_soft_loss_v4_ablation10_v1",
        "suite_identity": {"path": "/formal.json", "sha256": "a" * 64, "semantic_sha256": "b" * 64, "role": "formal"},
        "specification_identity": {"path": "/spec.md", "sha256": "c" * 64},
        "unit_coefficient_identity": {"path": "/unit.json", "sha256": "d" * 64, "semantic_sha256": "e" * 64},
        "candidate_registry_identity": {"path": "/registry.json", "sha256": "f" * 64, "semantic_sha256": "0" * 64},
        "candidate_row": {"candidate_id": "r08_r08", "order_index": 1, "dic_max_ratio": 0.08, "dts_max_ratio": 0.08},
        "constants": {
            "seed": 17,
            "optimizer": "Adam",
            "learning_rate": 0.007,
            "resolution": 128,
            "export_resolution": 128,
            "steps": 2000,
            "stage": "ablation10_128",
            "replicate_role": "primary",
            "replicate_index": 0,
        },
        "matrix": matrix,
        "decision_policy": copy.deepcopy(POLICY),
    }
    manifest["semantic_sha256"] = canonical_hash(manifest)
    _write_json(root / "manifests" / "ablation10.json", manifest)
    return root, configs


def _install_authenticator(monkeypatch: pytest.MonkeyPatch, *, stale: str | None = None) -> list[tuple[Path, dict[str, object]]]:
    calls: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(
        analyzer,
        "validate_resolved_config",
        lambda value: copy.deepcopy(value),
        raising=False,
    )

    def authenticate(run_dir: Path, *, expected_config: dict[str, object]) -> SimpleNamespace:
        calls.append((Path(run_dir), expected_config))
        metrics = Path(run_dir) / "metrics.json"
        audit = Path(run_dir) / "gradient_audit.json"
        hashes = {
            "metrics.json": hashlib.sha256(metrics.read_bytes()).hexdigest(),
            "gradient_audit.json": hashlib.sha256(audit.read_bytes()).hexdigest(),
        }
        if stale is not None and Path(run_dir).as_posix().endswith(stale):
            hashes["metrics.json"] = "9" * 64
        return SimpleNamespace(
            run_dir=str(run_dir),
            config=copy.deepcopy(expected_config),
            config_hash=canonical_hash(expected_config),
            status_sha256="1" * 64,
            artifact_manifest_sha256="2" * 64,
            artifacts_sha256=hashes,
        )

    monkeypatch.setattr(analyzer, "validate_completed_run", authenticate)
    return calls


def _decision_rows(
    *,
    dic_iou: list[float] | None = None,
    dic_dice: list[float] | None = None,
    dts_iou: list[float] | None = None,
    dts_dice: list[float] | None = None,
) -> list[dict[str, object]]:
    dic_iou = dic_iou or [0.012] * 10
    dic_dice = dic_dice or [0.006] * 10
    dts_iou = dts_iou or [0.011] * 10
    dts_dice = dts_dice or [0.006] * 10
    rows: list[dict[str, object]] = []
    for index, task in enumerate(TASKS):
        full_iou = 0.96
        full_dice = 0.98
        values = {
            "l12_bal": (0.94, 0.97),
            "dic_only": (full_iou - dts_iou[index], full_dice - dts_dice[index]),
            "dts_only": (full_iou - dic_iou[index], full_dice - dic_dice[index]),
            "v4_full": (full_iou, full_dice),
        }
        for profile in PROFILES:
            iou, dice = values[profile]
            auxiliary = profile != "l12_bal"
            rows.append(
                {
                    "task_id": task,
                    "profile": profile,
                    "run_id": f"runs/ablation10/{task}/{profile}",
                    "config_hash": str(index) * 64,
                    "metrics_sha256": "a" * 64,
                    "gradient_audit_sha256": "b" * 64,
                    "mean_iou": iou,
                    "mean_dice": dice,
                    "mean_recall": 0.98,
                    "mean_fpr": 0.002,
                    "eligible_slots": 1 if auxiliary else 0,
                    "active_slots": 1 if auxiliary else 0,
                    "fully_projected_zero_slots": 0,
                    "fully_projected_zero_fraction": 0.0 if auxiliary else None,
                    "engineering_status": "PASS" if auxiliary else None,
                }
            )
    return rows


def test_exact_sign_flip_uses_all_1024_assignments_and_inclusive_tail() -> None:
    """Break caught: the exact test uses a normal approximation, pseudo-count, or excludes ties."""

    assert analyzer.exact_sign_flip_pvalue([1.0] * 10) == 1.0 / 1024.0
    assert analyzer.exact_sign_flip_pvalue([1.0] * 9 + [0.0]) == 2.0 / 1024.0
    assert analyzer.exact_sign_flip_pvalue([0.0] * 10) == 1.0
    with pytest.raises(ValueError, match="exactly ten"):
        analyzer.exact_sign_flip_pvalue([1.0] * 9)
    with pytest.raises(ValueError, match="finite"):
        analyzer.exact_sign_flip_pvalue([1.0] * 9 + [math.nan])


def test_holm_adjusts_only_two_named_iou_hypotheses_with_deterministic_ties() -> None:
    """Break caught: correction includes Dice or does not enforce Holm monotonicity."""

    assert analyzer.holm_adjust_two(0.01, 0.04) == {"dic": 0.02, "dts": 0.04}
    assert analyzer.holm_adjust_two(0.04, 0.01) == {"dic": 0.04, "dts": 0.02}
    assert analyzer.holm_adjust_two(0.02, 0.02) == {"dic": 0.04, "dts": 0.04}
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        analyzer.holm_adjust_two(-0.01, 0.2)


def test_component_gate_uses_inclusive_mcids_strict_positivity_and_strict_p() -> None:
    """Break caught: any preregistered component boundary is rounded or made strict/inclusive wrongly."""

    exact = analyzer._component_gate(
        [0.02] * 8 + [0.0, -0.005], [0.05] + [0.0] * 9,
        adjusted_p=math.nextafter(0.05, 0.0),
    )
    assert exact["passed"] is True
    assert exact["observed"]["positive_task_count"] == 8
    cases = [
        ([math.nextafter(0.01, 0.0)] * 10, [0.005] * 10, 0.001, "mean_delta_iou_min"),
        ([0.01] * 10, [math.nextafter(0.005, 0.0)] * 10, 0.001, "mean_delta_dice_min"),
        ([0.02] * 7 + [0.0] * 3, [0.006] * 10, 0.001, "positive_task_count_min"),
        ([0.02] * 9 + [math.nextafter(-0.005, -math.inf)], [0.006] * 10, 0.001, "task_delta_iou_min"),
        ([0.02] * 10, [0.006] * 10, 0.05, "adjusted_p_strict_max"),
    ]
    for iou, dice, adjusted, failure in cases:
        gate = analyzer._component_gate(iou, dice, adjusted_p=adjusted)
        assert gate["passed"] is False
        assert failure in gate["failures"]


def test_quality_safety_and_engineering_next_float_boundaries() -> None:
    """Break caught: full quality, noninferiority, safety, or projection limits use tolerances."""

    assert analyzer._quality_gate(0.95, 0.94, -0.001, -0.001)["passed"] is True
    for values, failure in [
        ((math.nextafter(0.95, 0.0), 0.94, 0.0, 0.0), "full_mean_iou_min"),
        ((0.95, math.nextafter(0.94, 0.0), 0.0, 0.0), "full_task_iou_min"),
        ((0.95, 0.94, math.nextafter(-0.001, -math.inf), 0.0), "full_base_mean_iou_min"),
        ((0.95, 0.94, 0.0, math.nextafter(-0.001, -math.inf)), "full_base_mean_dice_min"),
    ]:
        gate = analyzer._quality_gate(*values)
        assert failure in gate["failures"]

    assert analyzer._safety_profile_gate(
        [-0.005] * 10, [0.01] + [0.0] * 9
    )["passed"] is True
    assert "task_recall_delta_min" in analyzer._safety_profile_gate(
        [-0.005] * 9 + [math.nextafter(-0.005, -math.inf)], [0.001] * 10
    )["failures"]
    assert "mean_fpr_delta_max" in analyzer._safety_profile_gate(
        [-0.005] * 10, [math.nextafter(0.01, math.inf)] + [0.0] * 9
    )["failures"]
    assert analyzer._engineering_row_gate(
        eligible_slots=1, active_slots=1, status="PASS", projected_zero_fraction=0.10
    )["passed"] is True
    assert "projected_zero_fraction_max" in analyzer._engineering_row_gate(
        eligible_slots=1,
        active_slots=1,
        status="PASS",
        projected_zero_fraction=math.nextafter(0.10, math.inf),
    )["failures"]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "BOTH_KEY"),
        ({"dts_iou": [0.004] * 10, "dts_dice": [0.003] * 10}, "DIC_KEY"),
        ({"dic_iou": [0.004] * 10, "dic_dice": [0.003] * 10}, "DTS_KEY"),
        (
            {
                "dic_iou": [0.004] * 10,
                "dic_dice": [0.003] * 10,
                "dts_iou": [0.004] * 10,
                "dts_dice": [0.003] * 10,
            },
            "NEITHER_KEY",
        ),
        ({"dic_iou": [-0.001] * 10, "dic_dice": [0.002] * 10}, "HARMFUL_OR_ANTAGONISTIC"),
        ({"dic_iou": [0.018] * 7 + [0.0] * 3, "dic_dice": [0.006] * 10}, "INCONCLUSIVE"),
    ],
)
def test_decision_emits_exactly_the_six_registered_labels(
    overrides: dict[str, list[float]], expected: str
) -> None:
    """Break caught: complete evidence is relabeled or a seventh outcome is invented."""

    result = analyzer._decision_from_rows(_decision_rows(**overrides))
    assert result["label"] == expected
    assert set(result["p_values"]) == {"dic", "dts"}


def test_deletion_contrasts_are_full_minus_the_matching_single_arm() -> None:
    """Break caught: DIC and DTS controls are swapped or subtraction is reversed."""

    decision = analyzer._decision_from_rows(
        _decision_rows(dic_iou=[0.013] * 10, dts_iou=[0.021] * 10)
    )
    first = decision["task_contrasts"][0]
    assert first["dic_delta_iou"] == pytest.approx(0.013)
    assert first["dts_delta_iou"] == pytest.approx(0.021)
    assert first["full_base_delta_iou"] == pytest.approx(0.02)


def test_task_effect_aggregates_ordered_deltas_not_separate_arm_means() -> None:
    """Break caught: floating-point arm means are subtracted before paired task effects."""

    controls = [0.96, 0.96, 0.8, 0.94, 0.97, 0.96, 0.94, 0.97, 0.95, 0.93]
    rows = _decision_rows()
    for index, task in enumerate(TASKS):
        group = {row["profile"]: row for row in rows if row["task_id"] == task}
        group["dts_only"]["mean_iou"] = controls[index]
        group["v4_full"]["mean_iou"] = controls[index] + 0.01
    decision = analyzer._decision_from_rows(rows)
    paired = sum((value + 0.01) - value for value in controls) / 10.0
    separate = sum(value + 0.01 for value in controls) / 10.0 - sum(controls) / 10.0
    assert paired >= 0.01
    assert separate < 0.01
    assert decision["gates"]["dic"]["observed"]["mean_delta_iou"] == paired
    assert "mean_delta_iou_min" not in decision["gates"]["dic"]["failures"]


def test_complete_engineering_failure_is_inconclusive_not_a_scientific_success() -> None:
    """Break caught: STOP, inactive, or zero-slot auxiliary evidence can produce a KEY label."""

    rows = _decision_rows()
    rows[1]["engineering_status"] = "STOP"
    assert analyzer._decision_from_rows(rows)["label"] == "INCONCLUSIVE"
    rows = _decision_rows()
    rows[1]["active_slots"] = 0
    assert analyzer._decision_from_rows(rows)["label"] == "INCONCLUSIVE"


def test_quality_and_safety_failures_take_harmful_precedence() -> None:
    """Break caught: a component KEY result hides a failed full-quality or safety gate."""

    rows = _decision_rows()
    for row in rows:
        row["mean_iou"] = float(row["mean_iou"] - 0.03)
    assert analyzer._decision_from_rows(rows)["label"] == "HARMFUL_OR_ANTAGONISTIC"

    rows = _decision_rows()
    base_recall = rows[0]["mean_recall"]
    rows[1]["mean_recall"] = float(base_recall - math.nextafter(0.005, math.inf))
    assert analyzer._decision_from_rows(rows)["label"] == "HARMFUL_OR_ANTAGONISTIC"


def test_full_requires_positive_dic_and_dts_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: one active full-model component is mistaken for activation of both."""

    root, _ = _campaign(tmp_path)
    audit_path = root / "runs/ablation10/connected_01/v4_full/gradient_audit.json"
    audit = json.loads(audit_path.read_bytes())
    assert [sample["component"] for sample in audit["samples"]] == ["dic", "dts"]
    audit["samples"][1]["final_auxiliary_gradient_norm"] = 0.0
    _write_json(audit_path, audit)
    _install_authenticator(monkeypatch)
    report = analyzer.analyze_ablation_manifest(root)
    assert report["label"] == "INCONCLUSIVE"
    assert report["gates"]["engineering"]["passed"] is False


def test_analyzer_authenticates_exact_matrix_and_reads_nested_six_view_means(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: analysis trusts STATUS/config hashes or pools pixels instead of equal-weighting views."""

    root, configs = _campaign(tmp_path)
    first_run = root / "runs/ablation10/connected_01/l12_bal"
    unequal = [
        _view(1, 0, 0, 9999),
        _view(1, 1, 0, 1),
        _view(1, 2, 0, 1),
        _view(1, 3, 0, 1),
        _view(1, 4, 0, 1),
        _view(1, 5, 0, 1),
    ]
    config_hash = canonical_hash(configs["runs/ablation10/connected_01/l12_bal"])
    _write_json(first_run / "metrics.json", _metrics(config_hash, "l12_bal", views=unequal))
    calls = _install_authenticator(monkeypatch)

    report = analyzer.analyze_ablation_manifest(root)

    assert len(calls) == 80
    assert [(path, expected) for path, expected in calls[:40]] == [
        (root / run_id, configs[run_id])
        for task in TASKS
        for profile in PROFILES
        for run_id in [f"runs/ablation10/{task}/{profile}"]
    ]
    with (root / "analysis/ablation10_rows.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_equal_view_iou = sum(float(item["iou"]) for item in unequal) / 6.0
    pooled_iou = 6.0 / (6.0 + 15.0)
    assert float(rows[0]["mean_iou"]) == expected_equal_view_iou
    assert float(rows[0]["mean_iou"]) != pytest.approx(pooled_iou)
    assert len(rows) == 40
    assert report["row_count"] == 40


def _replace_threshold_with_scientifically_irrelevant_invalid_diagnostics(
    metrics_path: Path, threshold: str
) -> None:
    payload = json.loads(metrics_path.read_bytes())
    invalid_report = _threshold([_view(1, 0, 0, 0) for _ in range(6)])
    payload["thresholds"][threshold] = invalid_report
    payload["task_iou_by_threshold"][threshold] = 1.0
    ious = list(payload["task_iou_by_threshold"].values())
    payload["worst_threshold_iou"] = min(ious)
    payload["threshold_iou_range"] = max(ious) - min(ious)
    payload["invalid_task_metric"] = True
    _write_json(metrics_path, payload)


def test_authenticated_point04_point06_invalid_diagnostics_do_not_change_science_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: non-primary thresholds become hidden scientific/safety gates."""

    reference_root, _ = _campaign(tmp_path / "reference")
    _install_authenticator(monkeypatch)
    reference = analyzer.analyze_ablation_manifest(reference_root)

    altered_root, _ = _campaign(tmp_path / "altered")
    metrics_path = altered_root / "runs/ablation10/connected_01/l12_bal/metrics.json"
    _replace_threshold_with_scientifically_irrelevant_invalid_diagnostics(
        metrics_path, "0.04"
    )
    _replace_threshold_with_scientifically_irrelevant_invalid_diagnostics(
        metrics_path, "0.06"
    )
    _install_authenticator(monkeypatch)
    altered = analyzer.analyze_ablation_manifest(altered_root)

    for key in (
        "task_contrasts",
        "p_values",
        "gates",
        "label",
        "supports_both_crucial",
    ):
        assert altered[key] == reference[key]


def test_primary_point05_null_invalid_safety_metric_still_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: ignoring irrelevant endpoints also stops validating primary Recall/FPR."""

    root, _ = _campaign(tmp_path)
    metrics_path = root / "runs/ablation10/connected_01/l12_bal/metrics.json"
    _replace_threshold_with_scientifically_irrelevant_invalid_diagnostics(
        metrics_path, "0.05"
    )
    _install_authenticator(monkeypatch)
    with pytest.raises(ValueError, match="0.05|fpr|aggregate|finite"):
        analyzer.analyze_ablation_manifest(root)
    assert not (root / "analysis").exists()


@pytest.mark.parametrize(
    "breakage",
    ["missing", "semantic", "policy", "metrics_schema", "nonfinite", "auth"],
)
def test_incomplete_corrupt_or_nonfinite_inputs_raise_before_any_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, breakage: str
) -> None:
    """Break caught: fail-closed evidence problems leave a partial or INCONCLUSIVE report."""

    root, _ = _campaign(tmp_path)
    manifest_path = root / "manifests/ablation10.json"
    manifest = json.loads(manifest_path.read_bytes())
    if breakage == "missing":
        manifest["matrix"].pop()
        body = dict(manifest)
        body.pop("semantic_sha256")
        manifest["semantic_sha256"] = canonical_hash(body)
        _write_json(manifest_path, manifest)
        _install_authenticator(monkeypatch)
    elif breakage == "semantic":
        manifest["semantic_sha256"] = "9" * 64
        _write_json(manifest_path, manifest)
        _install_authenticator(monkeypatch)
    elif breakage == "policy":
        manifest["decision_policy"]["component_key"]["mean_delta_iou_min"] = 0.0
        body = dict(manifest)
        body.pop("semantic_sha256")
        manifest["semantic_sha256"] = canonical_hash(body)
        _write_json(manifest_path, manifest)
        _install_authenticator(monkeypatch)
    elif breakage == "metrics_schema":
        path = root / "runs/ablation10/connected_01/l12_bal/metrics.json"
        payload = json.loads(path.read_bytes())
        payload["unexpected"] = True
        _write_json(path, payload)
        _install_authenticator(monkeypatch)
    elif breakage == "nonfinite":
        path = root / "runs/ablation10/connected_01/l12_bal/metrics.json"
        payload = json.loads(path.read_bytes())
        payload["thresholds"]["0.05"]["per_view"][0]["iou"] = math.nan
        _write_json(path, payload, canonical=False)
        _install_authenticator(monkeypatch)
    else:
        monkeypatch.setattr(
            analyzer,
            "validate_completed_run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("not completed")),
        )
    with pytest.raises((FileNotFoundError, ValueError)):
        analyzer.analyze_ablation_manifest(root)
    assert not (root / "analysis").exists()


def test_manifest_rejects_unregistered_resolved_config_before_touching_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: malformed resolved configs reach run paths before Task 3 schema validation."""

    root, _ = _campaign(tmp_path)
    auth_calls = _install_authenticator(monkeypatch)
    monkeypatch.setattr(
        analyzer,
        "validate_resolved_config",
        lambda _value: (_ for _ in ()).throw(ValueError("invalid resolved schema")),
        raising=False,
    )
    with pytest.raises(ValueError, match="resolved|schema"):
        analyzer.analyze_ablation_manifest(root)
    assert auth_calls == []
    assert not (root / "analysis").exists()


def test_coherently_rehashed_non_r08_candidate_manifest_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a valid but unregistered .12/.12 budget replaces frozen r08_r08."""

    root, _ = _campaign(tmp_path)
    path = root / "manifests/ablation10.json"
    manifest = json.loads(path.read_bytes())
    changed_candidate = {
        "candidate_id": "r12_r12",
        "order_index": 2,
        "dic_max_ratio": 0.12,
        "dts_max_ratio": 0.12,
    }
    manifest["candidate_row"] = changed_candidate
    for cell in manifest["matrix"]:
        if cell["profile"] == "l12_bal":
            continue
        config = cell["resolved_config"]
        config["budget_identity"] = copy.deepcopy(changed_candidate)
        config_hash = canonical_hash(config)
        cell["config_hash"] = config_hash
        cell["fresh_argv"] = _canonical_argv(config, resume=False)
        cell["resume_argv"] = _canonical_argv(config, resume=True)
        run_dir = Path(cell["run_dir"])
        metrics = json.loads((run_dir / "metrics.json").read_bytes())
        metrics["config_hash"] = config_hash
        _write_json(run_dir / "metrics.json", metrics)
        audit = json.loads((run_dir / "gradient_audit.json").read_bytes())
        audit["config_hash"] = config_hash
        _write_json(run_dir / "gradient_audit.json", audit)
    body = dict(manifest)
    body.pop("semantic_sha256")
    manifest["semantic_sha256"] = canonical_hash(body)
    _write_json(path, manifest)
    _install_authenticator(monkeypatch)
    with pytest.raises(ValueError, match="r08|candidate|budget"):
        analyzer.analyze_ablation_manifest(root)
    assert not (root / "analysis").exists()


def _rehash_changed_cell(cell: dict[str, object]) -> None:
    config = cell["resolved_config"]
    config_hash = canonical_hash(config)
    cell["config_hash"] = config_hash
    cell["fresh_argv"] = _canonical_argv(config, resume=False)
    cell["resume_argv"] = _canonical_argv(config, resume=True)
    run_dir = Path(cell["run_dir"])
    metrics = json.loads((run_dir / "metrics.json").read_bytes())
    metrics["config_hash"] = config_hash
    _write_json(run_dir / "metrics.json", metrics)
    audit = json.loads((run_dir / "gradient_audit.json").read_bytes())
    audit["config_hash"] = config_hash
    _write_json(run_dir / "gradient_audit.json", audit)


def _republish_manifest_semantic(path: Path, manifest: dict[str, object]) -> None:
    body = dict(manifest)
    body.pop("semantic_sha256")
    manifest["semantic_sha256"] = canonical_hash(body)
    _write_json(path, manifest)


def test_coherently_rehashed_single_arm_device_drift_fails_before_run_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: one arm silently runs on a different device than the paired matrix."""

    root, _ = _campaign(tmp_path)
    path = root / "manifests/ablation10.json"
    manifest = json.loads(path.read_bytes())
    changed = manifest["matrix"][1]
    assert changed["profile"] == "dic_only"
    changed["resolved_config"]["device"] = "cuda:0"
    _rehash_changed_cell(changed)
    _republish_manifest_semantic(path, manifest)
    auth_calls = _install_authenticator(monkeypatch)
    with pytest.raises(ValueError, match="device|paired|shared|scientific"):
        analyzer.analyze_ablation_manifest(root)
    assert auth_calls == []
    assert not (root / "analysis").exists()


def test_coherently_rehashed_single_arm_input_identity_drift_fails_before_run_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: one arm receives a different ordered shadow input than its paired arms."""

    root, _ = _campaign(tmp_path)
    path = root / "manifests/ablation10.json"
    manifest = json.loads(path.read_bytes())
    changed = manifest["matrix"][2]
    assert changed["profile"] == "dts_only"
    changed_input = changed["resolved_config"]["shadow_inputs_xyz"][0]
    changed_input["path"] = "/coherently_changed_x.png"
    changed_input["sha256"] = "9" * 64
    _rehash_changed_cell(changed)
    _republish_manifest_semantic(path, manifest)
    auth_calls = _install_authenticator(monkeypatch)
    with pytest.raises(ValueError, match="input|paired|shared|scientific"):
        analyzer.analyze_ablation_manifest(root)
    assert auth_calls == []
    assert not (root / "analysis").exists()


def test_stale_authenticated_artifact_hash_is_rejected_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: metrics are read after authentication without comparing manifest-bound bytes."""

    root, _ = _campaign(tmp_path)
    _install_authenticator(monkeypatch, stale="connected_01/l12_bal")
    with pytest.raises(ValueError, match="artifact|hash|metrics"):
        analyzer.analyze_ablation_manifest(root)
    assert not (root / "analysis").exists()


def test_create_once_is_idempotent_and_refuses_changed_existing_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: reruns overwrite an immutable scientific conclusion."""

    root, _ = _campaign(tmp_path)
    _install_authenticator(monkeypatch)
    first = analyzer.analyze_ablation_manifest(root)
    rows_path = root / "analysis/ablation10_rows.csv"
    report_path = root / "analysis/ablation10_report.json"
    before = (rows_path.read_bytes(), report_path.read_bytes())
    assert analyzer.analyze_ablation_manifest(root) == first
    assert (rows_path.read_bytes(), report_path.read_bytes()) == before
    report_path.write_bytes(report_path.read_bytes() + b"tampered")
    with pytest.raises((FileExistsError, ValueError), match="immutable|differs|canonical"):
        analyzer.analyze_ablation_manifest(root)


def test_verify_recomputes_byte_for_byte_and_performs_zero_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: --verify repairs, creates, or merely trusts existing analysis bytes."""

    root, _ = _campaign(tmp_path)
    _install_authenticator(monkeypatch)
    expected = analyzer.analyze_ablation_manifest(root)
    before = {
        path.relative_to(root): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert analyzer.analyze_ablation_manifest(root, verify=True) == expected
    after = {
        path.relative_to(root): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before

    missing_root, _ = _campaign(tmp_path / "missing")
    _install_authenticator(monkeypatch)
    with pytest.raises(FileNotFoundError):
        analyzer.analyze_ablation_manifest(missing_root, verify=True)
    assert not (missing_root / "analysis").exists()


def test_report_binds_manifest_rows_hash_contrasts_gates_and_semantic_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the conclusion cannot be reproduced from its registered identities and rows."""

    root, _ = _campaign(tmp_path)
    _install_authenticator(monkeypatch)
    report = analyzer.analyze_ablation_manifest(root)
    payload = json.loads((root / "analysis/ablation10_report.json").read_bytes())
    body = dict(payload)
    semantic = body.pop("semantic_sha256")
    assert canonical_hash(body) == semantic
    assert payload == report
    assert payload["manifest_identity"]["semantic_sha256"] == json.loads(
        (root / "manifests/ablation10.json").read_bytes()
    )["semantic_sha256"]
    assert payload["rows_sha256"] == hashlib.sha256(
        (root / "analysis/ablation10_rows.csv").read_bytes()
    ).hexdigest()
    assert len(payload["task_contrasts"]) == 10
    assert set(payload["gates"]) == {"dic", "dts", "quality", "safety", "engineering"}
    assert payload["supports_both_crucial"] is True


def test_cli_has_only_output_root_and_verify_and_imports_no_training_runtime() -> None:
    """Break caught: analyzer accepts outcome-tuning switches or exposes a training mode."""

    parser = analyzer.build_parser()
    args = parser.parse_args(["--output-root", "/tmp/campaign", "--verify"])
    assert vars(args) == {"output_root": "/tmp/campaign", "verify": True}
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-root", "/tmp/campaign", "--threshold", "0.06"])
