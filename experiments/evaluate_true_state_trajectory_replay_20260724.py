#!/usr/bin/env python3
"""Evaluate structural-query policies on fixed, sequential carving states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def select_fem_every_state(rows: list[dict], threshold: float) -> tuple[dict | None, int]:
    selected = None
    calls = 0
    for row in rows:
        calls += 1
        if float(row["sf_fem"]) < threshold:
            break
        selected = row
    return selected, calls


def predicted_boundary(
    rows: list[dict], threshold: float, margin: float
) -> tuple[dict | None, int, int]:
    """Return the last PINO-safe state and its index on a heavy-to-light path."""
    monotone_sf = float("inf")
    selected = None
    selected_index = -1
    queries = 0
    for index, row in enumerate(rows):
        queries += 1
        monotone_sf = min(monotone_sf, float(row["sf_pino"]))
        if monotone_sf < threshold + margin:
            break
        selected = row
        selected_index = index
    return selected, selected_index, queries


def select_hybrid(
    rows: list[dict], threshold: float, margin: float
) -> tuple[dict | None, int, int, list[int]]:
    """FEM-check the PINO boundary and roll back on the same fixed path."""
    predicted, index, pino_queries = predicted_boundary(rows, threshold, margin)
    if predicted is None:
        index = 0
    calls = 0
    fem_indices: list[int] = []
    while index >= 0:
        calls += 1
        fem_indices.append(index)
        row = rows[index]
        if float(row["sf_fem"]) >= threshold:
            return row, calls, pino_queries, fem_indices
        index -= 1
    return None, calls, pino_queries, fem_indices


def elapsed_sum(rows: list[dict], key: str, indices: list[int]) -> float | None:
    values = [rows[index].get(key) for index in indices]
    if not values or any(value is None for value in values):
        return None
    return float(sum(float(value) for value in values))


def selected_detail(
    *,
    case_id: str,
    policy: str,
    selected: dict | None,
    oracle: dict,
    fem_calls: int,
    pino_calls: int,
    runtime_s: float | None,
    threshold: float,
) -> dict:
    return {
        "case_id": case_id,
        "policy": policy,
        "oracle_step": int(oracle["step"]),
        "selected_step": int(selected["step"]) if selected is not None else None,
        "oracle_agreement": bool(
            selected is not None and int(selected["step"]) == int(oracle["step"])
        ),
        "fem_safe": bool(
            selected is not None and float(selected["sf_fem"]) >= threshold
        ),
        "saving": float(selected["saving"]) if selected is not None else 0.0,
        "mean_iou": float(selected["mean_iou"]) if selected is not None else 0.0,
        "sf_pino": float(selected["sf_pino"]) if selected is not None else None,
        "sf_fem": float(selected["sf_fem"]) if selected is not None else None,
        "fem_calls": int(fem_calls),
        "pino_calls": int(pino_calls),
        "structural_screen_runtime_s": runtime_s,
    }


def summarize(
    policy: str,
    rows: list[dict],
    oracle_calls: float,
    oracle_runtime_s: float | None,
) -> dict:
    calls = mean_std([float(row["fem_calls"]) for row in rows])
    runtimes = [
        float(row["structural_screen_runtime_s"])
        for row in rows
        if row["structural_screen_runtime_s"] is not None
    ]
    runtime = mean_std(runtimes) if len(runtimes) == len(rows) else None
    return {
        "policy": policy,
        "trajectories": len(rows),
        "oracle_agreement": float(np.mean([row["oracle_agreement"] for row in rows])),
        "fem_safe_rate": float(np.mean([row["fem_safe"] for row in rows])),
        "saving": mean_std([float(row["saving"]) for row in rows]),
        "fem_calls_per_trajectory": calls,
        "fem_call_reduction": float(1.0 - calls["mean"] / oracle_calls),
        "fem_call_speedup": (
            float(oracle_calls / calls["mean"]) if calls["mean"] > 0 else None
        ),
        "structural_screen_runtime_s": runtime,
        "wall_clock_speedup": (
            float(oracle_runtime_s / runtime["mean"])
            if oracle_runtime_s is not None
            and runtime is not None
            and runtime["mean"] > 0
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sf-threshold", type=float, default=1.3)
    parser.add_argument("--pino-margin", type=float, default=0.05)
    args = parser.parse_args()

    sources = sorted(args.root.glob("connected_*/boundary_trajectory_rows.json"))
    if not sources:
        raise FileNotFoundError(f"No trajectory rows below {args.root}")

    details: list[dict] = []
    excluded: list[dict] = []
    policy_rows: dict[str, list[dict]] = {
        "FEM at every state": [],
        "PINO only": [],
        "PINO + selective FEM": [],
    }
    audit = {
        "joint_states": 0,
        "agreement": 0,
        "false_safe": 0,
        "false_unsafe": 0,
        "by_case": [],
    }

    for source in sources:
        payload = json.loads(source.read_text())
        case_id = str(payload["protocol"]["case_id"])
        all_joint_rows = sorted(
            [
                row
                for row in payload["rows"]
                if row.get("sf_fem") is not None
                and row.get("sf_pino") is not None
            ],
            key=lambda row: int(row["step"]),
        )
        case_agreement = case_false_safe = case_false_unsafe = 0
        for row in all_joint_rows:
            fem_safe = float(row["sf_fem"]) >= args.sf_threshold
            pino_safe = float(row["sf_pino"]) >= args.sf_threshold
            case_agreement += int(fem_safe == pino_safe)
            case_false_safe += int(pino_safe and not fem_safe)
            case_false_unsafe += int(fem_safe and not pino_safe)
        audit["joint_states"] += len(all_joint_rows)
        audit["agreement"] += case_agreement
        audit["false_safe"] += case_false_safe
        audit["false_unsafe"] += case_false_unsafe
        audit["by_case"].append({
            "case_id": case_id,
            "joint_states": len(all_joint_rows),
            "agreement": case_agreement,
            "false_safe": case_false_safe,
            "false_unsafe": case_false_unsafe,
        })
        rows = all_joint_rows
        if not rows:
            excluded.append({"case_id": case_id, "reason": "no jointly evaluated states"})
            continue

        oracle, oracle_calls = select_fem_every_state(rows, args.sf_threshold)
        if oracle is None:
            excluded.append({"case_id": case_id, "reason": "initial state FEM-unsafe"})
            continue

        pino, _, pino_queries = predicted_boundary(
            rows, args.sf_threshold, args.pino_margin
        )
        hybrid, hybrid_calls, hybrid_pino_queries, hybrid_fem_indices = select_hybrid(
            rows, args.sf_threshold, args.pino_margin
        )
        oracle_indices = list(range(oracle_calls))
        selections = {
            "FEM at every state": (
                oracle,
                oracle_calls,
                0,
                elapsed_sum(rows, "fem_elapsed_s", oracle_indices),
            ),
            "PINO only": (
                pino,
                0,
                pino_queries,
                elapsed_sum(
                    rows, "pino_elapsed_s", list(range(pino_queries))
                ),
            ),
            "PINO + selective FEM": (
                hybrid,
                hybrid_calls,
                hybrid_pino_queries,
                (
                    None
                    if (
                        elapsed_sum(
                            rows,
                            "pino_elapsed_s",
                            list(range(hybrid_pino_queries)),
                        )
                        is None
                        or elapsed_sum(rows, "fem_elapsed_s", hybrid_fem_indices)
                        is None
                    )
                    else elapsed_sum(
                        rows,
                        "pino_elapsed_s",
                        list(range(hybrid_pino_queries)),
                    )
                    + elapsed_sum(rows, "fem_elapsed_s", hybrid_fem_indices)
                ),
            ),
        }
        for policy, (selected, calls, pino_calls, runtime_s) in selections.items():
            detail = selected_detail(
                case_id=case_id,
                policy=policy,
                selected=selected,
                oracle=oracle,
                fem_calls=calls,
                pino_calls=pino_calls,
                runtime_s=runtime_s,
                threshold=args.sf_threshold,
            )
            details.append(detail)
            policy_rows[policy].append(detail)

    if not policy_rows["FEM at every state"]:
        raise RuntimeError("No trajectory has an FEM-safe initial state")
    oracle_call_mean = float(
        np.mean(
            [
                row["fem_calls"]
                for row in policy_rows["FEM at every state"]
            ]
        )
    )
    oracle_runtime_values = [
        float(row["structural_screen_runtime_s"])
        for row in policy_rows["FEM at every state"]
        if row["structural_screen_runtime_s"] is not None
    ]
    oracle_runtime_mean = (
        float(np.mean(oracle_runtime_values))
        if len(oracle_runtime_values) == len(policy_rows["FEM at every state"])
        else None
    )
    audit["agreement_rate"] = (
        float(audit["agreement"] / audit["joint_states"])
        if audit["joint_states"]
        else None
    )
    order = ["FEM at every state", "PINO only", "PINO + selective FEM"]
    result = {
        "schema": "pcraft_true_state_trajectory_replay_v1",
        "protocol": {
            "source": str(args.root),
            "ordered_states": "repaired scaffold followed by sequential guarded-carving phases",
            "trajectory_generation": "frozen before structural screening",
            "sf_threshold": args.sf_threshold,
            "pino_margin": args.pino_margin,
            "pino_role": "structural-query substitution only; no removal choice or trajectory reordering",
            "pino_only_audit": "zero deployment FEM calls; selected state is audited offline with FEM",
            "hybrid_policy": "FEM verifies the PINO-predicted boundary and rolls back to heavier states after failure",
            "speedup_definition": "wall-clock time of all structural queries made along one complete trajectory",
        },
        "eligible_trajectories": len(policy_rows["FEM at every state"]),
        "excluded": excluded,
        "all_joint_state_audit": audit,
        "rows": [
            summarize(
                policy,
                policy_rows[policy],
                oracle_call_mean,
                oracle_runtime_mean,
            )
            for policy in order
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
