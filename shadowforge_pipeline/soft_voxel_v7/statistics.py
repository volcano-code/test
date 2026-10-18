"""Pure preregistered formal statistics for P-CRAFT v7.

This module deliberately has no project, NumPy, or Torch imports: the producer
loads it only after all training evidence has been authenticated.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType


_TASK_IDS = tuple(f"v7_formal_{index:02d}" for index in range(1, 11))
_PROFILES = ("base", "hit_only", "barrier_only", "full")
_METRIC_KEYS = {
    "schema", "stage", "logical_index", "task_id", "budget_id", "profile",
    "config_sha256", "run_path", "completed_updates", "checkpoint_sha256",
    "metrics_path", "metrics_sha256", "iou", "dice", "recall", "fpr",
    "semantic_sha256",
}
_ENGINEERING_KEYS = {
    "schema", "stage", "physical_index", "task_id", "budget_id", "profile",
    "config_sha256", "run_path", "completed_updates", "checkpoint_sha256",
    "status", "failure_code", "device", "loss_finite", "gradient_finite",
    "parameters_finite", "cuda_executed", "clip_count", "optimizer_step_count",
    "slot_valid", "disabled_not_constructed", "step_time_count",
    "step_times_finite", "min_step_seconds", "peak_allocated_bytes",
    "device_total_bytes", "hit_eligible_slots", "hit_active_slots",
    "barrier_eligible_slots", "barrier_active_slots", "failure_record_path",
    "failure_record_sha256", "semantic_sha256",
}
_HEX = frozenset("0123456789abcdef")
_GATE_CODES = (
    "FULL_MEAN_IOU", "FULL_WORST_IOU",
    "HIT_MEAN_IOU", "HIT_MEAN_DICE", "HIT_POSITIVE_COUNT",
    "HIT_MINIMUM_IOU", "HIT_HOLM", "HIT_RECALL_ROLE",
    "BARRIER_MEAN_IOU", "BARRIER_MEAN_DICE", "BARRIER_POSITIVE_COUNT",
    "BARRIER_MINIMUM_IOU", "BARRIER_HOLM", "BARRIER_FPR_MEAN_ROLE",
    "BARRIER_FPR_TASK_CEILING", "HIT_SINGLE_IOU", "HIT_SINGLE_DICE",
    "BARRIER_SINGLE_IOU", "BARRIER_SINGLE_DICE",
)


def _immutable(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _immutable(nested) for key, nested in value.items()})
    if type(value) in (list, tuple):
        return tuple(_immutable(nested) for nested in value)
    return value


def _finite_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be an exact finite float")
    return value


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in _HEX for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _semantic(row: Mapping[str, object]) -> str:
    unsigned = dict(row)
    claimed = unsigned.pop("semantic_sha256", None)
    raw = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    actual = hashlib.sha256(raw).hexdigest()
    if claimed != actual:
        raise ValueError("row semantic SHA-256 mismatch")
    return actual


def ordered_task_mean(values: Sequence[float]) -> float:
    """Return the frozen left-to-right arithmetic mean of ten task values."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 10:
        raise ValueError("ordered task mean requires exactly ten values")
    checked = tuple(_finite_float(value, label=f"value[{index}]") for index, value in enumerate(values))
    return sum(checked) / 10.0


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """Exact one-sided inclusive sign-flip p-value over all 2^10 assignments."""

    if not isinstance(differences, Sequence) or isinstance(differences, (str, bytes)) or len(differences) != 10:
        raise ValueError("exact sign flip requires exactly ten differences")
    checked = tuple(_finite_float(value, label=f"difference[{index}]") for index, value in enumerate(differences))
    observed = ordered_task_mean(checked)
    tail = 0
    for signs in itertools.product((-1.0, 1.0), repeat=10):
        assigned = sum(sign * value for sign, value in zip(signs, checked)) / 10.0
        if assigned >= observed:
            tail += 1
    return tail / 1024.0


def holm_adjust_two(hit_pvalue: float, barrier_pvalue: float) -> Mapping[str, float]:
    """Holm-adjust exactly Hit and Barrier, resolving ties Hit-first."""

    raw = {
        "hit": _finite_float(hit_pvalue, label="Hit p-value"),
        "barrier": _finite_float(barrier_pvalue, label="Barrier p-value"),
    }
    if any(value < 0.0 or value > 1.0 for value in raw.values()):
        raise ValueError("p-values must lie in [0,1]")
    rank = {"hit": 0, "barrier": 1}
    first, second = sorted(raw, key=lambda name: (raw[name], rank[name]))
    first_adjusted = min(1.0, 2.0 * raw[first])
    second_adjusted = max(first_adjusted, raw[second])
    return MappingProxyType({
        "hit": first_adjusted if first == "hit" else second_adjusted,
        "barrier": first_adjusted if first == "barrier" else second_adjusted,
    })


def _validate_rows(
    metric_rows: Sequence[Mapping[str, object]],
    engineering_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, dict[str, float]]]:
    if type(metric_rows) not in (list, tuple) or len(metric_rows) != 40:
        raise ValueError("formal metric rows must be an exact 40-row sequence")
    if type(engineering_rows) not in (list, tuple) or len(engineering_rows) != 40:
        raise ValueError("formal engineering rows must be an exact 40-row sequence")
    expected = tuple((task, profile) for task in _TASK_IDS for profile in _PROFILES)
    cells: dict[str, dict[str, dict[str, float]]] = {}
    selected_budget: str | None = None
    for index, (metric_raw, engineering_raw, identity) in enumerate(
        zip(metric_rows, engineering_rows, expected, strict=True)
    ):
        if type(metric_raw) is not dict or set(metric_raw) != _METRIC_KEYS:
            raise ValueError(f"formal metric row {index} has the wrong exact schema")
        if type(engineering_raw) is not dict or set(engineering_raw) != _ENGINEERING_KEYS:
            raise ValueError(f"formal engineering row {index} has the wrong exact schema")
        task_id, profile = identity
        shared = (task_id, profile, index)
        if (
            (metric_raw["task_id"], metric_raw["profile"], metric_raw["logical_index"]) != shared
            or (engineering_raw["task_id"], engineering_raw["profile"], engineering_raw["physical_index"]) != shared
            or metric_raw["schema"] != "pcraft_soft_loss_v7_metric_row_v1"
            or engineering_raw["schema"] != "pcraft_soft_loss_v7_engineering_row_v1"
            or metric_raw["stage"] != "formal-3000"
            or engineering_raw["stage"] != "formal-3000"
            or metric_raw["completed_updates"] != 3000
            or engineering_raw["completed_updates"] != 3000
            or metric_raw["budget_id"] != engineering_raw["budget_id"]
            or metric_raw["config_sha256"] != engineering_raw["config_sha256"]
            or metric_raw["run_path"] != engineering_raw["run_path"]
            or metric_raw["checkpoint_sha256"] != engineering_raw["checkpoint_sha256"]
        ):
            raise ValueError("formal rows are missing, reordered, or cross-bound")
        expected_run = f"runs/formal/{task_id}/{profile}/seed_17"
        budget = metric_raw["budget_id"]
        if profile == "base":
            if budget is not None:
                raise ValueError("formal Base must have null budget")
        else:
            if budget not in ("budget_010", "budget_015", "budget_020"):
                raise ValueError("formal nonbase row has an unregistered budget")
            if selected_budget is None:
                selected_budget = budget
            elif budget != selected_budget:
                raise ValueError("formal rows do not share one selected budget")
        if (
            metric_raw["run_path"] != expected_run
            or metric_raw["metrics_path"] != f"{expected_run}/artifacts/metrics/step_3000.json"
        ):
            raise ValueError("formal row path is not canonical")
        if (
            engineering_raw["status"] != "PASS"
            or engineering_raw["failure_code"] != "NONE"
            or engineering_raw["failure_record_path"] is not None
            or engineering_raw["failure_record_sha256"] is not None
        ):
            raise ValueError("formal engineering evidence must contain exactly 40 PASS rows")
        for key in ("logical_index", "completed_updates"):
            if type(metric_raw[key]) is not int:
                raise ValueError(f"metric {key} must be exact int")
        for key in ("physical_index", "completed_updates", "clip_count", "optimizer_step_count",
                    "step_time_count", "peak_allocated_bytes", "device_total_bytes",
                    "hit_eligible_slots", "hit_active_slots", "barrier_eligible_slots",
                    "barrier_active_slots"):
            if type(engineering_raw[key]) is not int:
                raise ValueError(f"engineering {key} must be exact int")
        for key in ("loss_finite", "gradient_finite", "parameters_finite", "cuda_executed",
                    "slot_valid", "disabled_not_constructed", "step_times_finite"):
            if type(engineering_raw[key]) is not bool or not engineering_raw[key]:
                raise ValueError(f"engineering {key} must be exact true")
        if engineering_raw["device"] != "cuda:0":
            raise ValueError("formal engineering device must be cuda:0")
        if (
            engineering_raw["clip_count"] != 3000
            or engineering_raw["optimizer_step_count"] != 3000
            or engineering_raw["step_time_count"] != 3000
            or type(engineering_raw["min_step_seconds"]) is not float
            or not math.isfinite(engineering_raw["min_step_seconds"])
            or engineering_raw["min_step_seconds"] <= 0.0
            or engineering_raw["peak_allocated_bytes"] <= 0
            or engineering_raw["device_total_bytes"] <= engineering_raw["peak_allocated_bytes"]
        ):
            raise ValueError("formal engineering PASS counters/resources are invalid")
        expected_hit = 1450 if profile in ("hit_only", "full") else 0
        expected_barrier = 1450 if profile in ("barrier_only", "full") else 0
        hit_eligible = engineering_raw["hit_eligible_slots"]
        hit_active = engineering_raw["hit_active_slots"]
        barrier_eligible = engineering_raw["barrier_eligible_slots"]
        barrier_active = engineering_raw["barrier_active_slots"]
        if (
            hit_eligible != expected_hit
            or barrier_eligible != expected_barrier
            or hit_active < 0 or hit_active > hit_eligible
            or barrier_active < 0 or barrier_active > barrier_eligible
        ):
            raise ValueError("formal engineering slot evidence is impossible")
        for key in ("config_sha256", "checkpoint_sha256", "metrics_sha256"):
            _digest(metric_raw[key], label=f"metric {key}")
        _digest(engineering_raw["semantic_sha256"], label="engineering semantic SHA")
        _semantic(metric_raw)
        _semantic(engineering_raw)
        values = {
            name: _finite_float(metric_raw[name], label=f"{task_id}/{profile}/{name}")
            for name in ("iou", "dice", "recall", "fpr")
        }
        if any(value < 0.0 or value > 1.0 for value in values.values()):
            raise ValueError("formal metrics must lie in [0,1]")
        cells.setdefault(task_id, {})[profile] = values
    return cells


def evaluate_formal_outcome(
    metric_rows: Sequence[Mapping[str, object]],
    engineering_rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Evaluate the exact registered v7 formal outcome and return immutable data."""

    cells = _validate_rows(metric_rows, engineering_rows)

    def values(profile: str, metric: str) -> tuple[float, ...]:
        return tuple(cells[task][profile][metric] for task in _TASK_IDS)

    def difference(treatment: str, control: str, metric: str) -> tuple[float, ...]:
        return tuple(
            cells[task][treatment][metric] - cells[task][control][metric]
            for task in _TASK_IDS
        )

    full_iou = values("full", "iou")
    hit_iou = difference("full", "barrier_only", "iou")
    hit_dice = difference("full", "barrier_only", "dice")
    hit_recall = difference("full", "barrier_only", "recall")
    barrier_iou = difference("full", "hit_only", "iou")
    barrier_dice = difference("full", "hit_only", "dice")
    barrier_fpr_improvement = difference("hit_only", "full", "fpr")
    barrier_fpr_harm = difference("full", "hit_only", "fpr")
    hit_single_iou = difference("hit_only", "base", "iou")
    hit_single_dice = difference("hit_only", "base", "dice")
    barrier_single_iou = difference("barrier_only", "base", "iou")
    barrier_single_dice = difference("barrier_only", "base", "dice")

    hit_raw = exact_sign_flip_pvalue(hit_iou)
    barrier_raw = exact_sign_flip_pvalue(barrier_iou)
    adjusted = holm_adjust_two(hit_raw, barrier_raw)
    hit_mean_iou, hit_mean_dice = ordered_task_mean(hit_iou), ordered_task_mean(hit_dice)
    barrier_mean_iou = ordered_task_mean(barrier_iou)
    barrier_mean_dice = ordered_task_mean(barrier_dice)
    hit_positive = sum(value > 0.0 for value in hit_iou)
    barrier_positive = sum(value > 0.0 for value in barrier_iou)
    full_mean, full_worst = ordered_task_mean(full_iou), min(full_iou)
    hit_recall_mean = ordered_task_mean(hit_recall)
    barrier_fpr_mean = ordered_task_mean(barrier_fpr_improvement)
    maximum_fpr_harm = max(barrier_fpr_harm)
    single = {
        "hit_iou_vs_base": ordered_task_mean(hit_single_iou),
        "hit_dice_vs_base": ordered_task_mean(hit_single_dice),
        "barrier_iou_vs_base": ordered_task_mean(barrier_single_iou),
        "barrier_dice_vs_base": ordered_task_mean(barrier_single_dice),
    }
    decisions = (
        full_mean >= .95, full_worst >= .94,
        hit_mean_iou >= .010, hit_mean_dice >= .005, hit_positive >= 8,
        min(hit_iou) >= -.005, adjusted["hit"] < .05, hit_recall_mean > 0.0,
        barrier_mean_iou >= .010, barrier_mean_dice >= .005,
        barrier_positive >= 8, min(barrier_iou) >= -.005,
        adjusted["barrier"] < .05, barrier_fpr_mean > 0.0,
        maximum_fpr_harm <= .001,
        single["hit_iou_vs_base"] >= -.001,
        single["hit_dice_vs_base"] >= -.001,
        single["barrier_iou_vs_base"] >= -.001,
        single["barrier_dice_vs_base"] >= -.001,
    )
    gates = tuple({"code": code, "passed": passed} for code, passed in zip(_GATE_CODES, decisions, strict=True))
    soft_pass = decisions[0] and decisions[1]
    both_key = all(decisions[2:])
    result = {
        "schema": "pcraft_soft_loss_v7_formal_outcome_v1",
        "task_ids": _TASK_IDS,
        "hit": {
            "iou_differences": hit_iou, "dice_differences": hit_dice,
            "recall_differences": hit_recall, "mean_iou": hit_mean_iou,
            "mean_dice": hit_mean_dice, "positive_iou_count": hit_positive,
            "minimum_iou": min(hit_iou), "raw_pvalue": hit_raw,
            "holm_pvalue": adjusted["hit"], "recall_role_mean": hit_recall_mean,
        },
        "barrier": {
            "iou_differences": barrier_iou, "dice_differences": barrier_dice,
            "fpr_improvements": barrier_fpr_improvement, "mean_iou": barrier_mean_iou,
            "mean_dice": barrier_mean_dice, "positive_iou_count": barrier_positive,
            "minimum_iou": min(barrier_iou), "raw_pvalue": barrier_raw,
            "holm_pvalue": adjusted["barrier"], "fpr_role_mean": barrier_fpr_mean,
            "maximum_fpr_harm": maximum_fpr_harm,
        },
        "single_arm": single,
        "quality": {"full_task_iou": full_iou, "full_mean_iou": full_mean, "full_worst_iou": full_worst},
        "gates": gates,
        "failure_codes": tuple(code for code, passed in zip(_GATE_CODES, decisions, strict=True) if not passed),
        "labels": {"SOFT_PASS": "PASS" if soft_pass else "FAIL",
                   "BOTH_KEY": "PASS" if both_key else "FAIL",
                   "claim_allowed": bool(soft_pass and both_key)},
    }
    return _immutable(result)  # type: ignore[return-value]


__all__ = [
    "ordered_task_mean", "exact_sign_flip_pvalue", "holm_adjust_two",
    "evaluate_formal_outcome",
]
