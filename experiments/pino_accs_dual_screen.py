#!/usr/bin/env python3
"""Dual-sided ACS screening: certify-safe, certify-unsafe, fallback to FEM."""

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
        return {"mean": None, "std": None, "max": None}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "max": float(arr.max())}


def load_fem_times(path: Path) -> list[float]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    times: list[float] = []
    for value in payload.values():
        if isinstance(value, list):
            for row in value:
                if isinstance(row, dict) and row.get("fem_solve_time_s") is not None:
                    times.append(float(row["fem_solve_time_s"]))
    return times


def decision_counts(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, threshold: float) -> dict:
    safe = y >= threshold
    certify_safe = lower >= threshold
    certify_unsafe = upper < threshold
    fallback = ~(certify_safe | certify_unsafe)

    false_pass = np.logical_and(certify_safe, ~safe)
    false_fail = np.logical_and(certify_unsafe, safe)
    return {
        "certified_safe_count": int(certify_safe.sum()),
        "certified_unsafe_count": int(certify_unsafe.sum()),
        "fallback_count": int(fallback.sum()),
        "false_pass_count": int(false_pass.sum()),
        "false_fail_count": int(false_fail.sum()),
        "safe_count": int(safe.sum()),
        "unsafe_count": int((~safe).sum()),
        "autonomous_rate": float((certify_safe | certify_unsafe).mean()),
        "fallback_rate": float(fallback.mean()),
        "safe_accept_rate": float(certify_safe.mean()),
        "unsafe_reject_rate": float(certify_unsafe.mean()),
        "safe_coverage": float(np.logical_and(certify_safe, safe).sum() / max(safe.sum(), 1)),
        "unsafe_coverage": float(np.logical_and(certify_unsafe, ~safe).sum() / max((~safe).sum(), 1)),
        "safe_false_discovery_rate": float(false_pass.sum() / max(certify_safe.sum(), 1)),
        "unsafe_false_discovery_rate": float(false_fail.sum() / max(certify_unsafe.sum(), 1)),
        "system_error_rate": float((false_pass.sum() + false_fail.sum()) / max((certify_safe | certify_unsafe).sum(), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=ROOT / "certification/pino_vs_fem_hard450.json")
    parser.add_argument("--baseline_results", type=Path, default=ROOT / "results/baselines_fem20_20260623/results_combined.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pino_acs_dual_screen_20260624.json")
    parser.add_argument("--alpha_safe", type=float, default=0.10)
    parser.add_argument("--alpha_unsafe", type=float, default=0.10)
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
    threshold = float(np.log10(args.sf_threshold))

    rows: list[dict] = []
    decision_times: list[float] = []
    for seed in range(args.splits):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(y))
        calibration, test = indices[: len(indices) // 2], indices[len(indices) // 2 :]
        iso_idx, residual_idx = np.array_split(np.random.default_rng(seed + 10000).permutation(calibration), 2)

        t0 = time.perf_counter()
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[iso_idx], y[iso_idx])
        pred_cal = iso.predict(x[residual_idx])
        q_safe = finite_sample_quantile(pred_cal - y[residual_idx], args.alpha_safe)
        q_unsafe = finite_sample_quantile(y[residual_idx] - pred_cal, args.alpha_unsafe)
        pred_test = iso.predict(x[test])
        lower = pred_test - q_safe
        upper = pred_test + q_unsafe
        decision_times.append(time.perf_counter() - t0)
        rows.append(decision_counts(y[test], lower, upper, threshold))

    fem_times = load_fem_times(args.baseline_results)
    fem_time = float(np.mean(fem_times)) if fem_times else None
    pino_time = float(args.assumed_pino_ms / 1000.0)
    acs_time = float(np.mean(decision_times) / max(len(y) / 2, 1))
    fallback_rate = float(np.mean([r["fallback_rate"] for r in rows]))
    if fem_time is not None:
        hybrid_time = pino_time + acs_time + fallback_rate * fem_time
        speedup = fem_time / max(hybrid_time, 1e-12)
    else:
        hybrid_time = None
        speedup = None

    result = {
        "schema_version": "pino_acs_dual_screen_v1",
        "n_cases": int(len(y)),
        "method": "dual-sided-PINO+ACS+FEA-fallback",
        "threshold_sf": float(args.sf_threshold),
        "alpha_safe": float(args.alpha_safe),
        "alpha_unsafe": float(args.alpha_unsafe),
        "decision_policy": "certify safe if lower_bound>=tau; certify unsafe if upper_bound<tau; otherwise FEM fallback",
        "metrics": {
            key: summarize([float(r[key]) for r in rows])
            for key in [
                "autonomous_rate",
                "fallback_rate",
                "safe_accept_rate",
                "unsafe_reject_rate",
                "safe_coverage",
                "unsafe_coverage",
                "safe_false_discovery_rate",
                "unsafe_false_discovery_rate",
                "system_error_rate",
                "certified_safe_count",
                "certified_unsafe_count",
                "fallback_count",
                "false_pass_count",
                "false_fail_count",
            ]
        },
        "runtime": {
            "pino_inference_assumed_s": pino_time,
            "acs_decision_mean_s": acs_time,
            "fem_mean_s": fem_time,
            "hybrid_mean_s_per_case": hybrid_time,
            "estimated_speedup_vs_pure_fea": speedup,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
