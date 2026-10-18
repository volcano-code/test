from __future__ import annotations

import math
from types import MappingProxyType

import pytest

from shadowforge_pipeline.soft_voxel_v7.statistics import (
    evaluate_formal_outcome,
    exact_sign_flip_pvalue,
    holm_adjust_two,
    ordered_task_mean,
)


TASKS = tuple(f"v7_formal_{index:02d}" for index in range(1, 11))
PROFILES = ("base", "hit_only", "barrier_only", "full")


def _semantic(row: dict[str, object]) -> dict[str, object]:
    from shadowforge_pipeline.soft_voxel_v7.provenance import canonical_sha256
    row["semantic_sha256"] = canonical_sha256(row)
    return row


def rows(*, full_iou: float = .96, hit_effect: float = .02,
         barrier_effect: float = .02) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics: list[dict[str, object]] = []
    engineering: list[dict[str, object]] = []
    for task in TASKS:
        values = {
            "base": (.93, .94, .94, .02),
            "hit_only": (full_iou - barrier_effect, .97 - barrier_effect, .96, .02),
            "barrier_only": (full_iou - hit_effect, .97 - hit_effect, .94, .01),
            "full": (full_iou, .97, .97, .01),
        }
        for profile in PROFILES:
            index = len(metrics)
            iou, dice, recall, fpr = values[profile]
            metrics.append(_semantic({
                "schema": "pcraft_soft_loss_v7_metric_row_v1", "stage": "formal-3000",
                "logical_index": index, "task_id": task, "budget_id": None if profile == "base" else "budget_010",
                "profile": profile, "config_sha256": f"{index:064x}",
                "run_path": f"runs/formal/{task}/{profile}/seed_17",
                "completed_updates": 3000, "checkpoint_sha256": f"{index+100:064x}",
                "metrics_path": f"runs/formal/{task}/{profile}/seed_17/artifacts/metrics/step_3000.json",
                "metrics_sha256": f"{index+200:064x}",
                "iou": float(iou), "dice": float(dice), "recall": float(recall), "fpr": float(fpr),
            }))
            engineering.append(_semantic({
                "schema": "pcraft_soft_loss_v7_engineering_row_v1", "stage": "formal-3000",
                "physical_index": index, "task_id": task, "budget_id": None if profile == "base" else "budget_010",
                "profile": profile, "config_sha256": f"{index:064x}",
                "run_path": f"runs/formal/{task}/{profile}/seed_17", "completed_updates": 3000,
                "checkpoint_sha256": f"{index+100:064x}", "status": "PASS", "failure_code": "NONE",
                "device": "cuda:0", "loss_finite": True, "gradient_finite": True,
                "parameters_finite": True, "cuda_executed": True, "clip_count": 3000,
                "optimizer_step_count": 3000, "slot_valid": True, "disabled_not_constructed": True,
                "step_time_count": 3000, "step_times_finite": True, "min_step_seconds": .1,
                "peak_allocated_bytes": 1, "device_total_bytes": 2,
                "hit_eligible_slots": 1450 if profile in ("hit_only", "full") else 0,
                "hit_active_slots": 1450 if profile in ("hit_only", "full") else 0,
                "barrier_eligible_slots": 1450 if profile in ("barrier_only", "full") else 0,
                "barrier_active_slots": 1450 if profile in ("barrier_only", "full") else 0,
                "failure_record_path": None,
                "failure_record_sha256": None,
            }))
    return metrics, engineering


def test_ordered_mean_and_exact_sign_flip_are_frozen() -> None:
    values = tuple(float(index) for index in range(10))
    assert ordered_task_mean(values) == sum(values) / 10.0
    assert exact_sign_flip_pvalue((1.0,) * 10) == 1 / 1024
    assert exact_sign_flip_pvalue((0.0,) * 10) == 1.0
    for bad in (values[:-1], values + (10.0,), (True,) + values[1:], (math.nan,) + values[1:]):
        with pytest.raises(ValueError):
            ordered_task_mean(bad)


def test_holm_tie_is_hit_first_and_result_immutable() -> None:
    adjusted = holm_adjust_two(.02, .02)
    assert adjusted == {"hit": .04, "barrier": .04}
    assert isinstance(adjusted, MappingProxyType)
    with pytest.raises(TypeError):
        adjusted["hit"] = .2  # type: ignore[index]
    with pytest.raises(ValueError): holm_adjust_two(-.1, .2)
    with pytest.raises(ValueError): holm_adjust_two(True, .2)


def test_formal_outcome_exact_schema_order_and_passing_labels() -> None:
    metric, engineering = rows()
    result = evaluate_formal_outcome(metric, engineering)
    assert result["schema"] == "pcraft_soft_loss_v7_formal_outcome_v1"
    assert result["task_ids"] == TASKS
    assert result["labels"] == {"SOFT_PASS": "PASS", "BOTH_KEY": "PASS", "claim_allowed": True}
    assert result["hit"]["iou_differences"] == (.96 - .94,) * 10
    assert result["barrier"]["iou_differences"] == (.96 - .94,) * 10
    assert tuple(g["code"] for g in result["gates"]) == (
        "FULL_MEAN_IOU","FULL_WORST_IOU","HIT_MEAN_IOU","HIT_MEAN_DICE",
        "HIT_POSITIVE_COUNT","HIT_MINIMUM_IOU","HIT_HOLM","HIT_RECALL_ROLE",
        "BARRIER_MEAN_IOU","BARRIER_MEAN_DICE","BARRIER_POSITIVE_COUNT",
        "BARRIER_MINIMUM_IOU","BARRIER_HOLM","BARRIER_FPR_MEAN_ROLE",
        "BARRIER_FPR_TASK_CEILING","HIT_SINGLE_IOU","HIT_SINGLE_DICE",
        "BARRIER_SINGLE_IOU","BARRIER_SINGLE_DICE",
    )
    assert isinstance(result, MappingProxyType)
    assert isinstance(result["hit"], MappingProxyType)


@pytest.mark.parametrize("field,boundary", [
    ("hit_iou", .010), ("hit_dice", .005), ("barrier_iou", .010),
    ("barrier_dice", .005), ("full_mean", .95), ("full_worst", .94),
])
def test_registered_boundaries_use_exact_comparisons(field: str, boundary: float) -> None:
    metric, engineering = rows()
    def mutate(value: float) -> None:
        for index, row in enumerate(metric):
            profile = row["profile"]
            if field == "hit_iou" and profile == "barrier_only": row["iou"] = row["iou"] - value + .02
            if field == "hit_dice" and profile == "barrier_only": row["dice"] = row["dice"] - value + .02
            if field == "barrier_iou" and profile == "hit_only": row["iou"] = row["iou"] - value + .02
            if field == "barrier_dice" and profile == "hit_only": row["dice"] = row["dice"] - value + .02
            if field in {"full_mean", "full_worst"} and profile == "full": row["iou"] = value
            unsigned = dict(row); unsigned.pop("semantic_sha256")
            row["semantic_sha256"] = __import__("shadowforge_pipeline.soft_voxel_v7.provenance", fromlist=["canonical_sha256"]).canonical_sha256(unsigned)
    mutate(boundary)
    exact = evaluate_formal_outcome(metric, engineering)
    code = {"hit_iou":"HIT_MEAN_IOU","hit_dice":"HIT_MEAN_DICE","barrier_iou":"BARRIER_MEAN_IOU","barrier_dice":"BARRIER_MEAN_DICE","full_mean":"FULL_MEAN_IOU","full_worst":"FULL_WORST_IOU"}[field]
    assert next(g for g in exact["gates"] if g["code"] == code)["passed"]


def test_rejects_wrong_order_alias_bool_nonfinite_range_and_engineering_fail() -> None:
    metric, engineering = rows()
    variants = []
    bad = [dict(row) for row in metric]; bad[0], bad[1] = bad[1], bad[0]; variants.append((bad, engineering))
    bad = [dict(row) for row in metric]; bad[0]["task_id"] = "alias"; variants.append((bad, engineering))
    bad = [dict(row) for row in metric]; bad[0]["iou"] = True; variants.append((bad, engineering))
    bad = [dict(row) for row in metric]; bad[0]["iou"] = math.nan; variants.append((bad, engineering))
    bad = [dict(row) for row in metric]; bad[0]["iou"] = 1.1; variants.append((bad, engineering))
    bad_eng = [dict(row) for row in engineering]; bad_eng[0]["status"] = "FAIL"; variants.append((metric, bad_eng))
    for metrics, eng in variants:
        with pytest.raises(ValueError): evaluate_formal_outcome(metrics, eng)


def test_direction_count_minimum_fpr_ceiling_and_single_arm_floors() -> None:
    metric, engineering = rows()
    # Exactly seven positive Hit tasks fails count and Holm cannot rescue it.
    for task_index in range(3):
        row = metric[task_index * 4 + 2]
        row["iou"] = metric[task_index * 4 + 3]["iou"]
        unsigned = dict(row); unsigned.pop("semantic_sha256"); row["semantic_sha256"] = __import__("shadowforge_pipeline.soft_voxel_v7.provenance", fromlist=["canonical_sha256"]).canonical_sha256(unsigned)
    result = evaluate_formal_outcome(metric, engineering)
    assert not next(g for g in result["gates"] if g["code"] == "HIT_POSITIVE_COUNT")["passed"]
    assert result["labels"]["BOTH_KEY"] == "FAIL"


@pytest.mark.parametrize(
    ("profile", "field", "value"),
    [
        ("base", "hit_active_slots", -1),
        ("base", "hit_eligible_slots", 1),
        ("hit_only", "hit_eligible_slots", 1449),
        ("hit_only", "hit_active_slots", 1451),
        ("hit_only", "barrier_eligible_slots", 1),
        ("barrier_only", "barrier_eligible_slots", 1451),
        ("barrier_only", "barrier_active_slots", -1),
        ("full", "hit_eligible_slots", 0),
        ("full", "barrier_active_slots", 1451),
    ],
)
def test_resigned_impossible_engineering_slot_rows_reject(
    profile: str, field: str, value: int
) -> None:
    metric, engineering = rows()
    index = PROFILES.index(profile)
    row = engineering[index]
    # First make the fixture itself exact under the registered profile contract.
    row["hit_eligible_slots"] = 1450 if profile in ("hit_only", "full") else 0
    row["hit_active_slots"] = 1450 if profile in ("hit_only", "full") else 0
    row["barrier_eligible_slots"] = 1450 if profile in ("barrier_only", "full") else 0
    row["barrier_active_slots"] = 1450 if profile in ("barrier_only", "full") else 0
    row[field] = value
    unsigned = dict(row); unsigned.pop("semantic_sha256")
    row["semantic_sha256"] = __import__(
        "shadowforge_pipeline.soft_voxel_v7.provenance", fromlist=["canonical_sha256"]
    ).canonical_sha256(unsigned)
    with pytest.raises(ValueError):
        evaluate_formal_outcome(metric, engineering)


def test_valid_fixture_uses_exact_registered_engineering_slots() -> None:
    metric, engineering = rows()
    for row in engineering:
        profile = row["profile"]
        row["hit_eligible_slots"] = 1450 if profile in ("hit_only", "full") else 0
        row["hit_active_slots"] = row["hit_eligible_slots"]
        row["barrier_eligible_slots"] = 1450 if profile in ("barrier_only", "full") else 0
        row["barrier_active_slots"] = row["barrier_eligible_slots"]
        unsigned = dict(row); unsigned.pop("semantic_sha256")
        row["semantic_sha256"] = __import__(
            "shadowforge_pipeline.soft_voxel_v7.provenance", fromlist=["canonical_sha256"]
        ).canonical_sha256(unsigned)
    assert evaluate_formal_outcome(metric, engineering)["labels"]["claim_allowed"] is True


def _resign(row: dict[str, object]) -> None:
    unsigned=dict(row); unsigned.pop("semantic_sha256")
    row["semantic_sha256"] = __import__(
        "shadowforge_pipeline.soft_voxel_v7.provenance",fromlist=["canonical_sha256"]
    ).canonical_sha256(unsigned)


@pytest.mark.parametrize(
    ("code", "profile", "metric_name", "threshold"),
    [
        ("HIT_MEAN_IOU","barrier_only","iou",.010),
        ("HIT_MEAN_DICE","barrier_only","dice",.005),
        ("BARRIER_MEAN_IOU","hit_only","iou",.010),
        ("BARRIER_MEAN_DICE","hit_only","dice",.005),
    ],
)
def test_mcid_nextafter_below_fails(code: str, profile: str, metric_name: str,
                                    threshold: float) -> None:
    metric,engineering=rows()
    below=math.nextafter(threshold,-math.inf)
    for task_index in range(10):
        control=metric[task_index*4+PROFILES.index(profile)]
        full=metric[task_index*4+3]
        control[metric_name]=0.0; full[metric_name]=below
        _resign(control); _resign(full)
    result=evaluate_formal_outcome(metric,engineering)
    assert next(row for row in result["gates"] if row["code"]==code)["passed"] is False


def test_floor_quality_fpr_and_holm_nextafter_sides() -> None:
    metric,engineering=rows()
    # One Hit task immediately below the -.005 reverse floor.
    control=metric[2]; full=metric[3]
    control["iou"]=full["iou"]-math.nextafter(-.005,-math.inf); _resign(control)
    result=evaluate_formal_outcome(metric,engineering)
    assert next(row for row in result["gates"] if row["code"]=="HIT_MINIMUM_IOU")["passed"] is False

    metric,engineering=rows()
    hit=metric[1]; full=metric[3]
    full["fpr"]=hit["fpr"]+math.nextafter(.001,math.inf); _resign(full)
    result=evaluate_formal_outcome(metric,engineering)
    assert next(row for row in result["gates"] if row["code"]=="BARRIER_FPR_TASK_CEILING")["passed"] is False

    adjusted=holm_adjust_two(.025,.025)
    assert adjusted["hit"]==.05 and adjusted["barrier"]==.05
    assert holm_adjust_two(math.nextafter(.025,-math.inf),math.nextafter(.025,-math.inf))["hit"] < .05


@pytest.mark.parametrize(("profile","code","metric_name"),[
    ("hit_only","HIT_SINGLE_IOU","iou"),("hit_only","HIT_SINGLE_DICE","dice"),
    ("barrier_only","BARRIER_SINGLE_IOU","iou"),("barrier_only","BARRIER_SINGLE_DICE","dice"),
])
def test_single_arm_floor_nextafter_below_fails(profile: str, code: str,
                                                metric_name: str) -> None:
    metric,engineering=rows()
    below=math.nextafter(-.001,-math.inf)
    for task_index in range(10):
        arm=metric[task_index*4+PROFILES.index(profile)]; base=metric[task_index*4]
        arm[metric_name]=base[metric_name]+below; _resign(arm)
    result=evaluate_formal_outcome(metric,engineering)
    assert next(row for row in result["gates"] if row["code"]==code)["passed"] is False


def test_exactly_eight_positive_tasks_passes_count_and_zero_roles_fail() -> None:
    metric,engineering=rows()
    for task_index in range(2):
        barrier_only=metric[task_index*4+2]; full=metric[task_index*4+3]
        barrier_only["iou"]=full["iou"]; _resign(barrier_only)
    result=evaluate_formal_outcome(metric,engineering)
    assert next(row for row in result["gates"] if row["code"]=="HIT_POSITIVE_COUNT")["passed"] is True

    metric,engineering=rows()
    for task_index in range(10):
        full=metric[task_index*4+3]; barrier_only=metric[task_index*4+2]
        full["recall"]=barrier_only["recall"]; _resign(full)
    result=evaluate_formal_outcome(metric,engineering)
    assert next(row for row in result["gates"] if row["code"]=="HIT_RECALL_ROLE")["passed"] is False

    metric,engineering=rows()
    for task_index in range(10):
        full=metric[task_index*4+3]; hit_only=metric[task_index*4+1]
        hit_only["fpr"]=full["fpr"]; _resign(hit_only)
    result=evaluate_formal_outcome(metric,engineering)
    assert next(row for row in result["gates"] if row["code"]=="BARRIER_FPR_MEAN_ROLE")["passed"] is False


@pytest.mark.parametrize(("code","mode"),[("FULL_MEAN_IOU","mean"),("FULL_WORST_IOU","worst")])
def test_quality_nextafter_below_fails(code: str, mode: str) -> None:
    metric,engineering=rows()
    if mode=="mean":
        for task_index in range(10):
            full=metric[task_index*4+3]; full["iou"]=math.nextafter(.95,-math.inf); _resign(full)
    else:
        full=metric[3]; full["iou"]=math.nextafter(.94,-math.inf); _resign(full)
    result=evaluate_formal_outcome(metric,engineering)
    assert next(row for row in result["gates"] if row["code"]==code)["passed"] is False


def test_holm_exact_point_zero_five_fails_strict_boundary() -> None:
    adjusted=holm_adjust_two(.025,.025)
    assert adjusted == {"hit":.05,"barrier":.05}
    assert not (adjusted["hit"] < .05) and not (adjusted["barrier"] < .05)
