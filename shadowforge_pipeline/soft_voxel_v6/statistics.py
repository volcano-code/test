"""Pure preregistered statistics for the v6 ten-task soft ablation."""

from __future__ import annotations

import itertools
import math
from typing import Mapping, Sequence

from .core import PROFILE_ORDER, TASK_ORDER


_ROW_KEYS = {"task", "arm", "mean_iou", "mean_dice", "mean_recall", "mean_fpr"}


def _finite(value: object, *, label: str) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def ordered_mean(values: Sequence[float]) -> float:
    """Return the frozen task-equal ordered mean over exactly ten values."""

    if not isinstance(values, Sequence) or len(values) != 10:
        raise ValueError("ordered task mean requires exactly ten values")
    checked = [_finite(value, label=f"value[{index}]") for index, value in enumerate(values)]
    return float(sum(checked) / 10.0)


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """One-sided inclusive exact sign-flip p-value over all 1024 assignments."""

    if not isinstance(differences, Sequence) or len(differences) != 10:
        raise ValueError("exact sign flip requires exactly ten differences")
    checked = [_finite(value, label=f"difference[{index}]") for index, value in enumerate(differences)]
    observed = ordered_mean(checked)
    tail = 0
    for signs in itertools.product((-1.0, 1.0), repeat=10):
        statistic = float(sum(sign * value for sign, value in zip(signs, checked)) / 10.0)
        if statistic >= observed:
            tail += 1
    return float(tail / 1024.0)


def holm_adjust_two(j_pvalue: float, h_pvalue: float) -> dict[str, float]:
    """Holm-adjust exactly J and H, breaking raw-p ties in J-then-H order."""

    raw = {
        "j": _finite(j_pvalue, label="J p-value"),
        "h": _finite(h_pvalue, label="H p-value"),
    }
    if any(not 0.0 <= value <= 1.0 for value in raw.values()):
        raise ValueError("p-values must lie in [0,1]")
    order = {"j": 0, "h": 1}
    first, second = sorted(raw, key=lambda name: (raw[name], order[name]))
    first_adjusted = min(1.0, 2.0 * raw[first])
    second_adjusted = max(first_adjusted, raw[second])
    return {
        "j": float(first_adjusted if first == "j" else second_adjusted),
        "h": float(first_adjusted if first == "h" else second_adjusted),
    }


def _validated_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    if not isinstance(rows, Sequence) or len(rows) != 40:
        raise ValueError("soft analysis requires exactly 40 rows")
    expected = tuple((task, arm) for task in TASK_ORDER for arm in PROFILE_ORDER)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for index, (raw, identity) in enumerate(zip(rows, expected)):
        if type(raw) is not dict or set(raw) != _ROW_KEYS:
            raise ValueError(f"row {index} has the wrong exact schema")
        task, arm = identity
        if raw["task"] != task or raw["arm"] != arm:
            raise ValueError("rows are missing, duplicated, or out of canonical order")
        metrics = {
            name: _finite(raw[f"mean_{name}"], label=f"{task}/{arm}/{name}")
            for name in ("iou", "dice", "recall", "fpr")
        }
        if any(not 0.0 <= value <= 1.0 for value in metrics.values()):
            raise ValueError("all task metrics must lie in [0,1]")
        result.setdefault(task, {})[arm] = metrics
    return result


def _validate_activity(
    activity: Mapping[str, Mapping[str, Mapping[str, Mapping[str, int]]]]
) -> tuple[bool, list[dict[str, object]]]:
    if type(activity) is not dict or tuple(activity) != TASK_ORDER:
        raise ValueError("activity must contain the exact ordered ten tasks")
    required = {
        "base": (),
        "j_only": ("j",),
        "h_only": ("h",),
        "full": ("j", "h"),
    }
    passed = True
    rows: list[dict[str, object]] = []
    for task in TASK_ORDER:
        by_arm = activity[task]
        if type(by_arm) is not dict or tuple(by_arm) != PROFILE_ORDER:
            raise ValueError("activity arms are missing, extra, or reordered")
        for arm in PROFILE_ORDER:
            components = by_arm[arm]
            if type(components) is not dict or tuple(components) != required[arm]:
                raise ValueError("activity components violate the frozen arm branches")
            for component in required[arm]:
                record = components[component]
                if type(record) is not dict or set(record) != {"eligible", "zero", "active"}:
                    raise ValueError("activity record has the wrong exact schema")
                if any(type(record[name]) is not int for name in record):
                    raise ValueError("activity counters must be exact integers")
                eligible = record["eligible"]
                zero = record["zero"]
                active = record["active"]
                valid = (
                    eligible == 2100
                    and 0 <= zero <= eligible
                    and active == eligible - zero
                    and active > 0
                    and zero <= 210
                )
                passed = passed and valid
                rows.append(
                    {
                        "task": task,
                        "arm": arm,
                        "component": component,
                        "eligible": eligible,
                        "zero": zero,
                        "active": active,
                        "zero_fraction": zero / 2100.0,
                        "pass": valid,
                    }
                )
    return passed, rows


def evaluate_soft_outcome(
    rows: Sequence[Mapping[str, object]],
    activity: Mapping[str, Mapping[str, Mapping[str, Mapping[str, int]]]],
) -> dict[str, object]:
    """Evaluate every frozen v6 soft-quality, role, safety, and activity gate."""

    cells = _validated_rows(rows)
    activity_pass, activity_rows = _validate_activity(activity)

    def values(arm: str, metric: str) -> list[float]:
        return [cells[task][arm][metric] for task in TASK_ORDER]

    def deltas(treatment: str, control: str, metric: str) -> list[float]:
        return [
            cells[task][treatment][metric] - cells[task][control][metric]
            for task in TASK_ORDER
        ]

    full_iou = values("full", "iou")
    j_iou = deltas("full", "h_only", "iou")
    j_dice = deltas("full", "h_only", "dice")
    h_iou = deltas("full", "j_only", "iou")
    h_fpr = deltas("full", "j_only", "fpr")
    raw_p = {
        "j": exact_sign_flip_pvalue(j_iou),
        "h": exact_sign_flip_pvalue(h_iou),
    }
    adjusted = holm_adjust_two(raw_p["j"], raw_p["h"])
    j_mean = ordered_mean(j_iou)
    h_mean = ordered_mean(h_iou)
    gates: dict[str, bool] = {
        "quality_full_mean_iou": ordered_mean(full_iou) >= 0.95,
        "quality_full_worst_iou": min(full_iou) >= 0.94,
        "j_iou_mean": j_mean > 0.0,
        "j_iou_positive_count": sum(value > 0.0 for value in j_iou) >= 8,
        "j_iou_reverse_floor": min(j_iou) >= -0.005,
        "j_iou_holm": adjusted["j"] < 0.05,
        "j_dice_role": ordered_mean(j_dice) >= 0.0,
        "h_iou_mean": h_mean > 0.0,
        "h_iou_positive_count": sum(value > 0.0 for value in h_iou) >= 8,
        "h_iou_reverse_floor": min(h_iou) >= -0.005,
        "h_iou_holm": adjusted["h"] < 0.05,
        "h_fpr_role": ordered_mean(h_fpr) < 0.0,
        "engineering_activity": activity_pass,
    }
    recall_deltas = {
        arm: deltas(arm, "base", "recall") for arm in ("j_only", "h_only", "full")
    }
    gates["recall_safety"] = all(
        delta >= -0.005 for arm in recall_deltas.values() for delta in arm
    )
    quality = gates["quality_full_mean_iou"] and gates["quality_full_worst_iou"]
    j_iou_gate = all(gates[name] for name in (
        "j_iou_mean", "j_iou_positive_count", "j_iou_reverse_floor", "j_iou_holm"
    ))
    h_iou_gate = all(gates[name] for name in (
        "h_iou_mean", "h_iou_positive_count", "h_iou_reverse_floor", "h_iou_holm"
    ))
    soft_pass = quality and gates["recall_safety"] and gates["engineering_activity"]
    both_confirmed = j_iou_gate and gates["j_dice_role"] and h_iou_gate and gates["h_fpr_role"]

    failures: list[str] = []
    code_map = (
        ("quality_full_mean_iou", "QUALITY_FULL_MEAN_IOU"),
        ("quality_full_worst_iou", "QUALITY_FULL_WORST_IOU"),
        ("j_iou_mean", "J_IOU_MEAN"),
        ("j_iou_positive_count", "J_IOU_POSITIVE_COUNT"),
        ("j_iou_reverse_floor", "J_IOU_REVERSE"),
        ("j_iou_holm", "J_IOU_SIGN_FLIP_HOLM"),
        ("j_dice_role", "J_DICE_ROLE"),
        ("h_iou_mean", "H_IOU_MEAN"),
        ("h_iou_positive_count", "H_IOU_POSITIVE_COUNT"),
        ("h_iou_reverse_floor", "H_IOU_REVERSE"),
        ("h_iou_holm", "H_IOU_SIGN_FLIP_HOLM"),
        ("h_fpr_role", "H_FPR_ROLE"),
        ("recall_safety", "RECALL_SAFETY"),
        ("engineering_activity", "ENGINEERING_ACTIVITY"),
    )
    failures.extend(code for gate, code in code_map if not gates[gate])
    return {
        "task_order": list(TASK_ORDER),
        "arm_order": list(PROFILE_ORDER),
        "contrasts": {
            "j": {
                "iou_deltas": j_iou,
                "dice_deltas": j_dice,
                "mean_iou": j_mean,
                "mean_dice": ordered_mean(j_dice),
                "positive_iou_tasks": sum(value > 0.0 for value in j_iou),
                "minimum_iou": min(j_iou),
                "raw_p": raw_p["j"],
                "holm_adjusted_p": adjusted["j"],
            },
            "h": {
                "iou_deltas": h_iou,
                "fpr_deltas": h_fpr,
                "mean_iou": h_mean,
                "mean_fpr": ordered_mean(h_fpr),
                "positive_iou_tasks": sum(value > 0.0 for value in h_iou),
                "minimum_iou": min(h_iou),
                "raw_p": raw_p["h"],
                "holm_adjusted_p": adjusted["h"],
            },
        },
        "quality": {"full_mean_iou": ordered_mean(full_iou), "full_worst_iou": min(full_iou)},
        "recall_deltas": recall_deltas,
        "activity": activity_rows,
        "gates": gates,
        "SOFT_PASS": soft_pass,
        "BOTH_CONFIRMED": both_confirmed,
        "claim_allowed": soft_pass and both_confirmed,
        "failure_codes": failures,
    }


__all__ = ["ordered_mean", "exact_sign_flip_pvalue", "holm_adjust_two", "evaluate_soft_outcome"]
