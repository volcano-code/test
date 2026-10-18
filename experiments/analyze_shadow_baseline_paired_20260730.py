#!/usr/bin/env python3
"""Task-paired inference for the frozen ten-task shadow benchmark.

The task, rather than a view, seed, or checkpoint, is the independent unit.
For each baseline and metric, the script reports P-CRAFT minus baseline:
the paired mean, a task bootstrap percentile interval, an exact two-sided
sign-flip randomization p-value, and win/tie/loss counts.  Holm adjustment is
applied jointly to the six baseline-by-metric tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "shadowforge_task_paired_baseline_audit_v1"
METRICS = ("mean_iou", "worst_iou")
REFERENCE = "P-CRAFT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_sign_flip_pvalue(delta: np.ndarray) -> float:
    observed = abs(float(delta.mean()))
    signed_means = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(delta)):
        signed_means.append(float(np.mean(delta * np.asarray(signs))))
    extreme = sum(abs(value) >= observed - 1.0e-15 for value in signed_means)
    return float(extreme / len(signed_means))


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(np.asarray(pvalues, dtype=float))
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * pvalues[int(index)])
        running = max(running, value)
        adjusted[int(index)] = running
    return adjusted.tolist()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.bootstrap_reps < 1000:
        raise ValueError("--bootstrap-reps must be at least 1000")
    payload = json.loads(source.read_text(encoding="utf-8"))
    per_case = payload["per_case"]
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for row in per_case:
        by_method.setdefault(str(row["method"]), {})[str(row["case_id"])] = row
    if REFERENCE not in by_method:
        raise ValueError(f"Missing {REFERENCE} rows")
    reference_cases = set(by_method[REFERENCE])
    baselines = [method for method in by_method if method != REFERENCE]
    for method in baselines:
        if set(by_method[method]) != reference_cases:
            raise ValueError(f"{method}: task IDs differ from {REFERENCE}")

    rng = np.random.default_rng(args.seed)
    results: list[dict[str, Any]] = []
    for method in baselines:
        case_ids = sorted(reference_cases)
        for metric in METRICS:
            reference = np.asarray(
                [float(by_method[REFERENCE][case][metric]) for case in case_ids]
            )
            baseline = np.asarray(
                [float(by_method[method][case][metric]) for case in case_ids]
            )
            delta = reference - baseline
            indices = rng.integers(
                0, len(delta), size=(args.bootstrap_reps, len(delta))
            )
            bootstrap = delta[indices].mean(axis=1)
            wins = int(np.count_nonzero(delta > 1.0e-12))
            losses = int(np.count_nonzero(delta < -1.0e-12))
            ties = int(len(delta) - wins - losses)
            results.append(
                {
                    "baseline": method,
                    "metric": metric,
                    "n_tasks": len(delta),
                    "case_ids": case_ids,
                    "paired_differences": delta.tolist(),
                    "mean_delta_pcraft_minus_baseline": float(delta.mean()),
                    "median_delta_pcraft_minus_baseline": float(np.median(delta)),
                    "bootstrap_95_percentile_ci": [
                        float(np.quantile(bootstrap, 0.025)),
                        float(np.quantile(bootstrap, 0.975)),
                    ],
                    "exact_two_sided_sign_flip_p": exact_sign_flip_pvalue(delta),
                    "wins_ties_losses": [wins, ties, losses],
                }
            )

    adjusted = holm_adjust(
        [float(row["exact_two_sided_sign_flip_p"]) for row in results]
    )
    for row, value in zip(results, adjusted):
        row["holm_adjusted_p_across_six_tests"] = float(value)

    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "input": str(source),
        "input_sha256": sha256_file(source),
        "reference_method": REFERENCE,
        "independent_unit": "task",
        "n_tasks": len(reference_cases),
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.seed,
        "randomization_test": (
            "exact two-sided paired sign-flip test over all 2^10 assignments"
        ),
        "multiplicity": "Holm adjustment across 3 baselines x 2 metrics",
        "results": results,
    }
    write_json(output / "paired_results.json", result)
    fields = (
        "baseline",
        "metric",
        "n_tasks",
        "mean_delta_pcraft_minus_baseline",
        "median_delta_pcraft_minus_baseline",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "exact_two_sided_sign_flip_p",
        "holm_adjusted_p_across_six_tests",
        "wins",
        "ties",
        "losses",
    )
    with (output / "paired_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "baseline": row["baseline"],
                    "metric": row["metric"],
                    "n_tasks": row["n_tasks"],
                    "mean_delta_pcraft_minus_baseline": row[
                        "mean_delta_pcraft_minus_baseline"
                    ],
                    "median_delta_pcraft_minus_baseline": row[
                        "median_delta_pcraft_minus_baseline"
                    ],
                    "bootstrap_ci_low": row["bootstrap_95_percentile_ci"][0],
                    "bootstrap_ci_high": row["bootstrap_95_percentile_ci"][1],
                    "exact_two_sided_sign_flip_p": row[
                        "exact_two_sided_sign_flip_p"
                    ],
                    "holm_adjusted_p_across_six_tests": row[
                        "holm_adjusted_p_across_six_tests"
                    ],
                    "wins": row["wins_ties_losses"][0],
                    "ties": row["wins_ties_losses"][1],
                    "losses": row["wins_ties_losses"][2],
                }
            )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
