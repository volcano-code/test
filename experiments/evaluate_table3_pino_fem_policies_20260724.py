#!/usr/bin/env python3
"""Evaluate FEM, PINO-only, and PINO-guided selective-FEM policies."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def candidate_material(row: dict) -> int:
    for key in ("delivery_occupied", "proxy_occupied", "occupied_voxels"):
        value = row.get(key)
        if value is not None:
            return int(value)
    raise KeyError(f"No material field in row: {row}")


def ranked_candidates(candidates: list[dict], threshold_sf: float) -> list[dict]:
    return sorted(
        candidates,
        key=lambda row: (
            row["sf_pino"] < threshold_sf,
            row["material"],
            -row["sf_pino"],
            row["row_index"],
        ),
    )


def summarize_policy(name: str, selections: list[dict]) -> dict:
    count = len(selections)
    exact = sum(row["selected_path"] == row["oracle_path"] for row in selections)
    safe = sum(row["selected_safe"] for row in selections)
    return {
        "policy": name,
        "tasks": count,
        "oracle_match_rate": exact / count,
        "safe_selection_rate": safe / count,
        "saving_vs_densest_candidate": mean_std(
            [row["saving_vs_densest_candidate"] for row in selections]
        ),
        "fem_calls_per_task": mean_std([row["fem_calls"] for row in selections]),
        "total_fem_calls": int(sum(row["fem_calls"] for row in selections)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="sealed_test")
    parser.add_argument("--threshold-sf", type=float, default=1.3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Use the first N lexicographically sorted tasks; 0 uses all tasks.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.index.read_text())
    rows = [row for row in payload["rows"] if row["split"] == args.split]
    predictions = np.load(args.predictions)
    pred_log = np.asarray(predictions[args.split], dtype=np.float64)
    if len(rows) != len(pred_log):
        raise RuntimeError(f"Prediction count {len(pred_log)} != row count {len(rows)}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for index, (source, score) in enumerate(zip(rows, np.power(10.0, pred_log))):
        row = dict(source)
        row["row_index"] = index
        row["sf_pino"] = float(score)
        row["sf_fem"] = float(source["sf"])
        row["fem_safe"] = row["sf_fem"] >= args.threshold_sf
        row["material"] = candidate_material(source)
        grouped[str(source["task"])].append(row)

    task_names = sorted(
        task for task, candidates in grouped.items()
        if any(row["fem_safe"] for row in candidates)
    )
    if args.max_tasks > 0:
        task_names = task_names[: args.max_tasks]

    details: list[dict] = []
    by_policy: dict[str, list[dict]] = defaultdict(list)
    for task in task_names:
        candidates = grouped[task]
        fem_safe = [row for row in candidates if row["fem_safe"]]
        oracle = min(
            fem_safe, key=lambda row: (row["material"], row["row_index"])
        )
        densest = max(row["material"] for row in candidates)
        ordered = ranked_candidates(candidates, args.threshold_sf)

        # Exhaustive FEM evaluates every candidate and chooses the lightest safe one.
        selections = {
            "FEM at every state": (oracle, len(candidates)),
            # PINO-only deploys the first ranked candidate. FEM is used only for
            # the offline audit reported in this experiment.
            "PINO only": (ordered[0], 0),
        }

        # Verify a frozen top-k prefix. If it contains no safe candidate, expand
        # one candidate at a time until FEM finds a safe rollback point.
        checked = ordered[: min(args.top_k, len(ordered))]
        cursor = len(checked)
        while not any(row["fem_safe"] for row in checked) and cursor < len(ordered):
            checked.append(ordered[cursor])
            cursor += 1
        checked_safe = [row for row in checked if row["fem_safe"]]
        selective = min(
            checked_safe, key=lambda row: (row["material"], row["row_index"])
        )
        selections["PINO + selective FEM"] = (selective, len(checked))

        for policy, (selected, fem_calls) in selections.items():
            detail = {
                "task": task,
                "policy": policy,
                "oracle_path": oracle["path"],
                "selected_path": selected["path"],
                "selected_safe": bool(selected["fem_safe"]),
                "selected_sf_pino": float(selected["sf_pino"]),
                "selected_sf_fem": float(selected["sf_fem"]),
                "selected_material": int(selected["material"]),
                "densest_material": int(densest),
                "saving_vs_densest_candidate": float(
                    1.0 - selected["material"] / max(densest, 1)
                ),
                "fem_calls": int(fem_calls),
            }
            details.append(detail)
            by_policy[policy].append(detail)

    policy_order = [
        "FEM at every state",
        "PINO only",
        "PINO + selective FEM",
    ]
    summaries = [summarize_policy(name, by_policy[name]) for name in policy_order]
    full_calls = summaries[0]["fem_calls_per_task"]["mean"]
    for row in summaries:
        calls = row["fem_calls_per_task"]["mean"]
        row["fem_call_reduction_rate"] = float(1.0 - calls / full_calls)
        row["fem_call_speedup"] = float(full_calls / calls) if calls > 0 else None

    result = {
        "schema": "pcraft_table3_pino_fem_policy_v1",
        "protocol": {
            "split": args.split,
            "task_selection": (
                f"first {args.max_tasks} lexicographically sorted eligible tasks"
                if args.max_tasks > 0
                else "all eligible task-disjoint sealed tasks"
            ),
            "tasks": len(task_names),
            "sf_threshold": args.threshold_sf,
            "frozen_top_k": args.top_k,
            "material_definition": (
                "occupied proxy voxels; saving is relative to the densest "
                "candidate within the same task"
            ),
            "pino_only_fem_usage": "zero at deployment; FEM labels used only for audit",
        },
        "rows": summaries,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
