#!/usr/bin/env python3
"""Compare post-solidification selectors on identical candidate trajectories.

The audit is read-only.  Every policy sees the candidates stored in the same
MANIFEST.json, so differences can only come from candidate selection.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


REMOVED_RE = re.compile(r"carve_phase_\d+_removed_(\d+)")


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def compact(candidate: dict, best_iou: float) -> dict:
    return {
        key: candidate.get(key)
        for key in (
            "candidate", "mean_iou", "min_iou", "mean_dice", "saving",
            "components", "watertight", "eligible", "fabrication_pass",
        )
    } | {"fidelity_regret": float(best_iou - candidate["mean_iou"])}


def last_effective_carve(candidates: list[dict]) -> dict:
    effective = []
    for candidate in candidates:
        match = REMOVED_RE.search(candidate.get("candidate", ""))
        if match and int(match.group(1)) > 0:
            effective.append(candidate)
    return effective[-1] if effective else candidates[-1]


def summarize(rows: list[dict]) -> dict:
    return {
        "n_tasks": len(rows),
        "mean_iou": mean_std([r["mean_iou"] for r in rows]),
        "worst_view_iou": mean_std([r["min_iou"] for r in rows]),
        "mean_saving": mean_std([r["saving"] for r in rows]),
        "fidelity_regret": mean_std([r["fidelity_regret"] for r in rows]),
        "delivery_success_rate": float(np.mean([bool(r["eligible"]) for r in rows])),
        "single_body_rate": float(np.mean([r["components"] == 1 for r in rows])),
        "watertight_rate": float(np.mean([bool(r["watertight"]) for r in rows])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--iou-tolerance", type=float, default=0.01)
    ap.add_argument(
        "--variant", default="full",
        help="Only audit manifests whose parent directory has this name; use '*' for all.",
    )
    args = ap.parse_args()

    policy_rows: dict[str, list[dict]] = {
        "best_fidelity_candidate": [],
        "terminal_effective_carve": [],
        "lightest_eligible_candidate": [],
        "fidelity_band_selector": [],
    }
    per_task = []
    manifests = sorted(args.root.glob("**/MANIFEST.json"))
    if args.variant != "*":
        manifests = [path for path in manifests if path.parent.name == args.variant]

    for path in manifests:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        candidates = data.get("candidates", [])
        if not candidates:
            continue

        eligible = [row for row in candidates if row.get("eligible", False)]
        pool = eligible or candidates
        best_iou = max(row["mean_iou"] for row in candidates)

        # The fidelity reference keeps the least-carved member when projection
        # scores tie, making its material cost an interpretable upper anchor.
        best_fidelity = max(
            candidates,
            key=lambda row: (row["mean_iou"], row["min_iou"], -row["saving"]),
        )
        terminal = last_effective_carve(candidates)
        lightest = max(pool, key=lambda row: (row["saving"], row["mean_iou"], row["min_iou"]))
        pool_best_iou = max(row["mean_iou"] for row in pool)
        band = [
            row for row in pool
            if row["mean_iou"] >= pool_best_iou - args.iou_tolerance
        ]
        fidelity_band = max(
            band,
            key=lambda row: (row.get("eligible", False), row["saving"], row["min_iou"]),
        )

        selected = {
            "best_fidelity_candidate": best_fidelity,
            "terminal_effective_carve": terminal,
            "lightest_eligible_candidate": lightest,
            "fidelity_band_selector": fidelity_band,
        }
        task = str(path.parent.relative_to(args.root))
        task_row = {"task": task, "manifest": str(path), "choices": {}}
        for policy, candidate in selected.items():
            row = compact(candidate, best_iou)
            task_row["choices"][policy] = row
            policy_rows[policy].append(row)
        per_task.append(task_row)

    if not per_task:
        raise RuntimeError(f"No candidate manifests found under {args.root}")

    policies = list(policy_rows)
    choice_disagreement_rate = float(np.mean([
        len({row["choices"][policy]["candidate"] for policy in policies}) > 1
        for row in per_task
    ]))
    metric_disagreement_rate = float(np.mean([
        len({
            (
                round(row["choices"][policy]["mean_iou"], 12),
                round(row["choices"][policy]["saving"], 12),
                bool(row["choices"][policy]["eligible"]),
            )
            for policy in policies
        }) > 1
        for row in per_task
    ]))

    payload = {
        "source": str(args.root),
        "scope": "matched-trajectory selector audit",
        "variant": args.variant,
        "iou_tolerance": args.iou_tolerance,
        "n_tasks": len(per_task),
        "choice_disagreement_rate": choice_disagreement_rate,
        "metric_disagreement_rate": metric_disagreement_rate,
        "per_task": per_task,
        "aggregate": {name: summarize(rows) for name, rows in policy_rows.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "n_tasks": payload["n_tasks"],
        "choice_disagreement_rate": choice_disagreement_rate,
        "metric_disagreement_rate": metric_disagreement_rate,
        "aggregate": payload["aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
