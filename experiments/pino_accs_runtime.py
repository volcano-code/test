#!/usr/bin/env python3
"""Runtime and fallback accounting for the paper-facing PINO+ACCS screen."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]


def finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(int(np.ceil((len(ordered) + 1) * (1 - alpha))) - 1, len(ordered) - 1)
    return float(ordered[max(rank, 0)])


def summarize(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean_s": None, "median_s": None, "total_s": None, "n": 0}
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "total_s": float(arr.sum()),
        "n": int(arr.size),
    }


def load_fem_times(paths: list[Path]) -> list[float]:
    times: list[float] = []
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            rows = []
            for value in payload.values():
                if isinstance(value, list):
                    rows.extend(value)
            for row in rows:
                if isinstance(row, dict) and row.get("fem_solve_time_s") is not None:
                    times.append(float(row["fem_solve_time_s"]))
    return times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=ROOT / "certification/pino_vs_fem_hard450.json")
    parser.add_argument("--baseline_results", type=Path, default=ROOT / "results/baselines_fem_light5_20260623/results_combined.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pino_accs_runtime_20260623.json")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--sf_threshold", type=float, default=1.3)
    parser.add_argument("--splits", type=int, default=50)
    parser.add_argument("--assumed_pino_ms", type=float, default=8.0)
    args = parser.parse_args()

    payload = json.loads(args.pairs.read_text())
    fem = np.asarray(payload["fem_sf"], dtype=float)
    pino = np.asarray(payload["pino_sf"], dtype=float)
    valid = np.isfinite(fem) & np.isfinite(pino) & (fem > 0) & (pino > 0)
    fem, pino = fem[valid], pino[valid]
    x, y = np.log10(pino), np.log10(fem)
    tau = float(np.log10(args.sf_threshold))

    decision_times: list[float] = []
    fallback_rates: list[float] = []
    false_accept_rates: list[float] = []
    accepted_rates: list[float] = []
    safe_coverages: list[float] = []
    for seed in range(args.splits):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(y))
        calibration, test = indices[: len(indices) // 2], indices[len(indices) // 2 :]
        inner = np.random.default_rng(seed + 10000).permutation(calibration)
        iso_idx, conf_idx = np.array_split(inner, 2)

        t0 = time.perf_counter()
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[iso_idx], y[iso_idx])
        q = finite_sample_quantile(iso.predict(x[conf_idx]) - y[conf_idx], args.alpha)
        pred = iso.predict(x[test]) - q
        accepted = pred >= tau
        decision_times.append(time.perf_counter() - t0)

        safe = y[test] >= tau
        fallback_rates.append(float((~accepted).mean()))
        accepted_rates.append(float(accepted.mean()))
        false_accept_rates.append(float(np.logical_and(accepted, ~safe).sum() / max(accepted.sum(), 1)))
        safe_coverages.append(float(np.logical_and(accepted, safe).sum() / max(safe.sum(), 1)))

    fem_times = load_fem_times([args.baseline_results])
    fem_time = float(np.mean(fem_times)) if fem_times else None
    pino_time = float(args.assumed_pino_ms / 1000.0)
    accs_decision_time = float(np.mean(decision_times) / max(len(y) / 2, 1))
    fallback_rate = float(np.mean(fallback_rates))

    if fem_time is not None:
        pure_fea_mean_s = fem_time
        hybrid_mean_s = pino_time + accs_decision_time + fallback_rate * fem_time
        speedup = pure_fea_mean_s / max(hybrid_mean_s, 1e-12)
    else:
        pure_fea_mean_s = None
        hybrid_mean_s = None
        speedup = None

    result = {
        "schema_version": "pino_accs_runtime_v1",
        "n_cases": int(len(y)),
        "method": "PINO+ACCS+FEA-fallback",
        "iou": None,
        "volume_material": None,
        "safety_fem": {
            "sf_threshold": float(args.sf_threshold),
            "false_discovery_rate_mean": float(np.mean(false_accept_rates)),
            "safe_coverage_mean": float(np.mean(safe_coverages)),
        },
        "runtime": {
            "pino_inference_assumed_s": pino_time,
            "accs_decision_mean_s": accs_decision_time,
            "fem_solve_time": summarize(fem_times),
            "fallback_rate_mean": fallback_rate,
            "accepted_rate_mean": float(np.mean(accepted_rates)),
            "pure_fea_mean_s_per_case": pure_fea_mean_s,
            "hybrid_mean_s_per_case": hybrid_mean_s,
            "estimated_speedup_vs_pure_fea": speedup,
            "note": "PINO time is a conservative configured estimate; FEM time is read from cached FEM-backed baseline rows.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
