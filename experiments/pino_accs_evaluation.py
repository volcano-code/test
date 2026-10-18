#!/usr/bin/env python3
"""Paper-facing PINO + ACCS evaluation from cached FEM/PINO pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[1]


def finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(int(np.ceil((len(ordered) + 1) * (1 - alpha))) - 1, len(ordered) - 1)
    return float(ordered[max(rank, 0)])


def mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "max": None}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "max": float(arr.max())}


def decision_metrics(accepted: np.ndarray, safe: np.ndarray) -> dict:
    fp = int(np.logical_and(accepted, ~safe).sum())
    tp = int(np.logical_and(accepted, safe).sum())
    accepted_n = int(accepted.sum())
    safe_n = int(safe.sum())
    return {
        "false_accept_count": fp,
        "true_accept_count": tp,
        "accepted_count": accepted_n,
        "safe_count": safe_n,
        "safe_coverage": float(tp / max(safe_n, 1)),
        "false_discovery_rate": float(fp / max(accepted_n, 1)),
    }


def evaluate_split(x: np.ndarray, y: np.ndarray, variant: str, alpha: float, seed: int, threshold: float) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    calibration, test = indices[: len(indices) // 2], indices[len(indices) // 2 :]
    safe = y[test] >= threshold

    if variant == "raw":
        accepted = x[test] >= threshold
    elif variant == "isotonic":
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[calibration], y[calibration])
        accepted = iso.predict(x[test]) >= threshold
    elif variant == "conformal":
        q = finite_sample_quantile(x[calibration] - y[calibration], alpha)
        accepted = x[test] - q >= threshold
    elif variant == "accs":
        rng_inner = np.random.default_rng(seed + 10000)
        inner = rng_inner.permutation(calibration)
        iso_idx, conf_idx = np.array_split(inner, 2)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[iso_idx], y[iso_idx])
        q = finite_sample_quantile(iso.predict(x[conf_idx]) - y[conf_idx], alpha)
        accepted = iso.predict(x[test]) - q >= threshold
    else:
        raise ValueError(variant)
    return decision_metrics(accepted, safe)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=ROOT / "certification/pino_vs_fem_hard450.json")
    parser.add_argument("--boundary_pairs", type=Path, default=ROOT / "data/boundary_pairs.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pino_accs_evaluation.json")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--sf_threshold", type=float, default=1.3)
    parser.add_argument("--splits", type=int, default=50)
    args = parser.parse_args()

    payload = json.loads(args.pairs.read_text())
    fem = np.asarray(payload["fem_sf"], dtype=float)
    pino = np.asarray(payload["pino_sf"], dtype=float)
    valid = np.isfinite(fem) & np.isfinite(pino) & (fem > 0) & (pino > 0)
    fem, pino = fem[valid], pino[valid]
    x, y = np.log10(pino), np.log10(fem)
    threshold = float(np.log10(args.sf_threshold))

    raw_safe = fem >= args.sf_threshold
    raw_accepted = pino >= args.sf_threshold
    result = {
        "schema_version": "pino_accs_evaluation_v1",
        "n_pairs": int(len(y)),
        "sf_threshold": float(args.sf_threshold),
        "alpha": float(args.alpha),
        "pino_raw": {
            "log10_mae": float(np.mean(np.abs(x - y))),
            "log10_rmse": float(np.sqrt(np.mean((x - y) ** 2))),
            "spearman_rho": float(spearmanr(x, y).statistic),
            **decision_metrics(raw_accepted, raw_safe),
        },
    }

    variants = {}
    for variant in ("raw", "isotonic", "conformal", "accs"):
        rows = [evaluate_split(x, y, variant, args.alpha, seed, threshold) for seed in range(args.splits)]
        variants[variant] = {
            "false_accept_count": mean_std([r["false_accept_count"] for r in rows]),
            "safe_coverage": mean_std([r["safe_coverage"] for r in rows]),
            "false_discovery_rate": mean_std([r["false_discovery_rate"] for r in rows]),
            "accepted_count": mean_std([r["accepted_count"] for r in rows]),
        }
    result["component_ablation"] = variants

    bins = [(1.3, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, np.inf)]
    stratified = {}
    for lo, hi in bins:
        coverages = []
        for seed in range(args.splits):
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(y))
            calibration, test = indices[: len(indices) // 2], indices[len(indices) // 2 :]
            inner = np.random.default_rng(seed + 10000).permutation(calibration)
            iso_idx, conf_idx = np.array_split(inner, 2)
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[iso_idx], y[iso_idx])
            q = finite_sample_quantile(iso.predict(x[conf_idx]) - y[conf_idx], args.alpha)
            accepted = iso.predict(x[test]) - q >= threshold
            selected = (fem[test] >= lo) & (fem[test] < hi)
            if selected.any():
                coverages.append(float(accepted[selected].mean()))
        key = f"[{lo},{hi})" if np.isfinite(hi) else f"[{lo},inf)"
        stratified[key] = {
            "n_full_dataset": int(((fem >= lo) & (fem < hi)).sum()),
            "coverage": mean_std(coverages),
        }
    result["coverage_by_true_sf"] = stratified

    boundary = json.loads(args.boundary_pairs.read_text())
    bx = np.log10(np.asarray([r["pino_sf"] for r in boundary], dtype=float) * (12.0 / 50.0))
    by = np.log10(np.asarray([r["fem_sf"] for r in boundary], dtype=float) * (12.0 / 50.0))
    valid_b = np.isfinite(bx) & np.isfinite(by)
    bx, by = bx[valid_b], by[valid_b]
    shift_weights = np.exp(-((bx - threshold) / 0.15) ** 2)
    iid_fdr, shifted_fdr, weighted_fdr = [], [], []
    for seed in range(args.splits):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(by))
        calibration, test = indices[: len(indices) // 2], indices[len(indices) // 2 :]
        iso_idx, conf_idx = np.array_split(rng.permutation(calibration), 2)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(bx[iso_idx], by[iso_idx])
        residual = iso.predict(bx[conf_idx]) - by[conf_idx]
        q = finite_sample_quantile(residual, 0.05)

        probs = shift_weights[test] / shift_weights[test].sum()
        shifted = rng.choice(test, size=len(test), replace=True, p=probs)
        iid_fdr.append(decision_metrics(iso.predict(bx[test]) - q >= threshold, by[test] >= threshold)["false_discovery_rate"])
        shifted_fdr.append(decision_metrics(iso.predict(bx[shifted]) - q >= threshold, by[shifted] >= threshold)["false_discovery_rate"])

        order = np.argsort(residual)
        weights = shift_weights[conf_idx][order]
        cumulative = np.cumsum(weights) / max(weights.sum(), 1e-12)
        weighted_q = float(residual[order][min(np.searchsorted(cumulative, 0.95), len(order) - 1)])
        weighted_fdr.append(decision_metrics(iso.predict(bx[shifted]) - weighted_q >= threshold, by[shifted] >= threshold)["false_discovery_rate"])
    result["optimization_shift"] = {
        "n_pairs": int(len(by)),
        "alpha": 0.05,
        "iid_false_discovery_rate": mean_std(iid_fdr),
        "naive_shifted_false_discovery_rate": mean_std(shifted_fdr),
        "shift_weighted_false_discovery_rate": mean_std(weighted_fdr),
        "note": "Importance-weighted boundary stress test; not an IID conformal guarantee.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
