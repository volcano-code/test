"""Shared Raw-PINO ranking and FEM-shortlist policy.

PINO orders fabrication-valid candidates.  FEM remains the authority for every
candidate admitted to the structurally eligible shortlist.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def pino_material_rank_key(row: dict[str, Any], threshold_sf: float) -> tuple:
    """Prefer predicted-safe candidates, then lower material use."""
    sf_pino = float(row["sf_pino"])
    return (
        sf_pino < float(threshold_sf),
        int(row["material"]),
        -sf_pino,
        str(row.get("candidate", row.get("candidate_id", ""))),
    )


def rank_pino_candidates(
    rows: list[dict[str, Any]], threshold_sf: float
) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: pino_material_rank_key(row, threshold_sf))


def verify_ranked_prefix(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    threshold_sf: float,
    fem_evaluate: Callable[[dict[str, Any]], dict[str, Any]],
    fallback: str = "expand",
) -> dict[str, Any]:
    """Verify a PINO-ranked prefix and return an auditable policy summary.

    ``expand`` checks another ``top_k`` rows only when the current prefix has no
    FEM-safe candidate. ``full`` verifies all rows. ``none`` keeps a fixed
    prefix even when it contains no safe candidate.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if fallback not in {"none", "expand", "full"}:
        raise ValueError(f"Unsupported fallback: {fallback}")

    ordered = rank_pino_candidates(rows, threshold_sf)
    limit = len(ordered) if fallback == "full" else min(top_k, len(ordered))
    checked: list[dict[str, Any]] = []
    while len(checked) < limit:
        row = ordered[len(checked)]
        row["fem_rank"] = len(checked) + 1
        row["fem_result"] = fem_evaluate(row)
        checked.append(row)

    if fallback == "expand":
        while checked and not any(bool(row["fem_result"].get("safe")) for row in checked):
            if len(checked) >= len(ordered):
                break
            limit = min(len(checked) + top_k, len(ordered))
            while len(checked) < limit:
                row = ordered[len(checked)]
                row["fem_rank"] = len(checked) + 1
                row["fem_result"] = fem_evaluate(row)
                checked.append(row)

    checked_ids = {id(row) for row in checked}
    for rank, row in enumerate(ordered, start=1):
        row["pino_rank"] = rank
        row["fem_checked"] = id(row) in checked_ids

    safe_checked = [row for row in checked if bool(row["fem_result"].get("safe"))]
    selected = min(
        safe_checked,
        key=lambda row: (int(row["material"]), int(row["pino_rank"])),
        default=None,
    )
    return {
        "schema": "pino_ranked_fem_shortlist_v1",
        "ranking": "predicted-safe first, then material ascending, predicted SF descending",
        "top_k": int(top_k),
        "fallback": fallback,
        "candidate_count": len(ordered),
        "fem_calls": len(checked),
        "fem_call_reduction_rate": 1.0 - len(checked) / max(len(ordered), 1),
        "safe_checked_count": len(safe_checked),
        "shortlist_lightest_safe_candidate": (
            selected.get("candidate", selected.get("candidate_id")) if selected else None
        ),
    }


def verify_trajectory_rollback(
    rows: list[dict[str, Any]],
    *,
    threshold_sf: float,
    safety_margin: float,
    fem_evaluate: Callable[[dict[str, Any]], dict[str, Any]],
    max_rollbacks: int = -1,
) -> dict[str, Any]:
    """Use PINO to stop material reduction and FEM to verify/rollback.

    Each trajectory must be ordered from its heavier scaffold toward lighter
    carving checkpoints through ``trajectory_step``.  PINO is evaluated before
    this function and supplied as ``sf_pino``.  A cumulative-minimum envelope
    makes the predicted safety sequence conservative when individual estimates
    fluctuate.  FEM first checks the last checkpoint before the predicted
    boundary and walks toward heavier predecessors only when verification fails.
    """
    if safety_margin < 0.0:
        raise ValueError("safety_margin must be non-negative")
    if max_rollbacks < -1:
        raise ValueError("max_rollbacks must be -1 (unlimited) or non-negative")

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        trajectory = str(row.get("trajectory_id", "candidate_pool"))
        groups.setdefault(trajectory, []).append(row)

    decision_threshold = float(threshold_sf) + float(safety_margin)
    trajectories: list[dict[str, Any]] = []
    total_fem_calls = 0
    selected_ids: list[str] = []

    for trajectory_id, group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda row: (
                int(row.get("trajectory_step", 0)),
                -int(row.get("material", 0)),
                str(row.get("candidate", row.get("candidate_id", ""))),
            ),
        )
        monotone_sf = float("inf")
        first_predicted_unsafe: int | None = None
        for index, row in enumerate(ordered):
            monotone_sf = min(monotone_sf, float(row["sf_pino"]))
            row["trajectory_rank"] = index + 1
            row["monotone_sf_pino"] = monotone_sf
            row["pino_predicted_safe"] = bool(monotone_sf >= decision_threshold)
            row["fem_checked"] = False
            if first_predicted_unsafe is None and not row["pino_predicted_safe"]:
                first_predicted_unsafe = index

        if first_predicted_unsafe is None:
            start_index = len(ordered) - 1
        else:
            start_index = max(first_predicted_unsafe - 1, 0)

        checked: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        index = start_index
        rollbacks = 0
        while index >= 0:
            row = ordered[index]
            row["fem_result"] = fem_evaluate(row)
            row["fem_checked"] = True
            checked.append(row)
            total_fem_calls += 1
            if bool(row["fem_result"].get("safe", False)):
                selected = row
                break
            if max_rollbacks >= 0 and rollbacks >= max_rollbacks:
                break
            index -= 1
            rollbacks += 1

        selected_id = None
        if selected is not None:
            selected_id = str(selected.get("candidate", selected.get("candidate_id")))
            selected_ids.append(selected_id)
        trajectories.append({
            "trajectory_id": trajectory_id,
            "candidate_count": len(ordered),
            "decision_threshold_sf": decision_threshold,
            "first_predicted_unsafe_step": (
                int(ordered[first_predicted_unsafe].get("trajectory_step", first_predicted_unsafe))
                if first_predicted_unsafe is not None else None
            ),
            "fem_start_candidate": str(
                ordered[start_index].get("candidate", ordered[start_index].get("candidate_id"))
            ),
            "fem_checked_candidates": [
                str(row.get("candidate", row.get("candidate_id"))) for row in checked
            ],
            "rollback_count": max(len(checked) - 1, 0),
            "selected_fem_safe_candidate": selected_id,
        })

    return {
        "schema": "pino_trajectory_rollback_fem_v1",
        "policy": "PINO cumulative-minimum stopping with heavier-candidate FEM rollback",
        "threshold_sf": float(threshold_sf),
        "safety_margin": float(safety_margin),
        "decision_threshold_sf": decision_threshold,
        "max_rollbacks": int(max_rollbacks),
        "trajectory_count": len(trajectories),
        "candidate_count": len(rows),
        "fem_calls": total_fem_calls,
        "fem_call_reduction_rate": 1.0 - total_fem_calls / max(len(rows), 1),
        "selected_fem_safe_candidates": selected_ids,
        "trajectories": trajectories,
    }
