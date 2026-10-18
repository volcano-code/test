#!/usr/bin/env python3
"""Evaluate stricter ACS and candidate hard-negative calibration on saved candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
TAU = 1.3


def finite_sample_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    rank = min(int(np.ceil((len(values) + 1) * (1 - alpha))) - 1, len(values) - 1)
    return float(values[max(rank, 0)])


def fit_hard450(alpha: float):
    payload = json.loads((ROOT / "certification/pino_vs_fem_hard450.json").read_text())
    fem = np.asarray(payload["fem_sf"], dtype=float)
    pino = np.asarray(payload["pino_sf"], dtype=float)
    valid = np.isfinite(fem) & np.isfinite(pino) & (fem > 0) & (pino > 0)
    x, y = np.log10(pino[valid]), np.log10(fem[valid])
    rng = np.random.default_rng(20260624)
    idx = rng.permutation(len(y))
    iso_idx, conf_idx = np.array_split(idx, 2)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[iso_idx], y[iso_idx])
    pred = iso.predict(x[conf_idx])
    return iso, finite_sample_quantile(pred - y[conf_idx], alpha), finite_sample_quantile(y[conf_idx] - pred, alpha)


def fit_candidate(rows: list[dict], alpha: float):
    x = np.asarray([math.log10(max(r["sf_pino"], 1e-12)) for r in rows], dtype=float)
    y = np.asarray([math.log10(max(r["sf_fem"], 1e-12)) for r in rows], dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    rng = np.random.default_rng(20260624)
    idx = rng.permutation(len(y))
    iso_idx, conf_idx = np.array_split(idx, 2)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[iso_idx], y[iso_idx])
    pred = iso.predict(x[conf_idx])
    return iso, finite_sample_quantile(pred - y[conf_idx], alpha), finite_sample_quantile(y[conf_idx] - pred, alpha)


def with_acs(row: dict, iso, q_safe: float, q_unsafe: float, safe_margin: float) -> dict:
    out = dict(row)
    pred = float(iso.predict([math.log10(max(row["sf_pino"], 1e-12))])[0])
    safe_bar = math.log10(TAU * safe_margin)
    tau_log = math.log10(TAU)
    out["acs_lower_sf"] = float(10 ** (pred - q_safe))
    out["acs_upper_sf"] = float(10 ** (pred + q_unsafe))
    out["acs_certified_safe"] = bool(pred - q_safe >= safe_bar)
    out["acs_certified_unsafe"] = bool(pred + q_unsafe < tau_log)
    return out


def select(rows: list[dict], policy: str) -> tuple[dict | None, int]:
    printable = [r for r in rows if r["mean_iou"] >= 0.95 and r["component_count_6"] == 1 and r["watertight"]]
    ordered = sorted(printable, key=lambda r: r["occupied_voxels"])
    if not ordered:
        return None, 0
    if policy == "acs_safe":
        safe = [r for r in ordered if r["acs_certified_safe"]]
        return (safe[0] if safe else None), 0
    if policy == "dual_acs_fem_fallback":
        fallbacks = 0
        for row in ordered:
            if row["acs_certified_safe"]:
                return row, fallbacks
            if row["acs_certified_unsafe"]:
                continue
            fallbacks += 1
            if row["fem_safe"]:
                return row, fallbacks
        return None, fallbacks
    raise ValueError(policy)


def summarize(cases: list[dict], iso, q_safe: float, q_unsafe: float, safe_margin: float) -> dict:
    policies = ["acs_safe", "dual_acs_fem_fallback"]
    selected = {p: [] for p in policies}
    for case in cases:
        rows = [with_acs(r, iso, q_safe, q_unsafe, safe_margin) for r in case["candidates"]]
        for p in policies:
            chosen, fallbacks = select(rows, p)
            if chosen:
                selected[p].append(
                    {
                        "case_id": case["case_id"],
                        "triple": case["triple"],
                        "candidate": chosen["candidate"],
                        "safe": bool(chosen["fem_safe"]),
                        "saving": chosen["material_saved_vs_hull"],
                        "fallbacks": fallbacks,
                        "sf_fem": chosen["sf_fem"],
                        "sf_pino": chosen["sf_pino"],
                        "acs_lower_sf": chosen["acs_lower_sf"],
                    }
                )
    out = {}
    n_cases = len(cases)
    for p, rows in selected.items():
        out[p] = {
            "n_selected": len(rows),
            "selection_rate": len(rows) / n_cases if n_cases else 0.0,
            "safe_rate": float(np.mean([r["safe"] for r in rows])) if rows else None,
            "mean_saving": float(np.mean([r["saving"] for r in rows])) if rows else None,
            "mean_fem_fallbacks": float(np.mean([r["fallbacks"] for r in rows])) if rows else None,
            "false_selected_count": int(sum(1 for r in rows if not r["safe"])),
            "false_selected": [r for r in rows if not r["safe"]],
        }
    return out


def split_cases(cases: list[dict]):
    train, test = [], []
    for i, case in enumerate(cases):
        (train if i % 2 == 0 else test).append(case)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--free", type=Path, default=ROOT / "results/candidate_selection_batch_free50_20260624.json")
    parser.add_argument("--fixed", type=Path, default=ROOT / "results/candidate_selection_batch_fixed50_20260624.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/candidate_acs_variant_eval_20260624.json")
    args = parser.parse_args()

    datasets = {
        "free": json.loads(args.free.read_text())["cases"],
        "fixed": json.loads(args.fixed.read_text())["cases"],
    }

    results = {
        "schema_version": "candidate_acs_variant_eval_v1",
        "sources": {"free": str(args.free), "fixed": str(args.fixed)},
        "variants": {},
    }

    for alpha in [0.10, 0.05, 0.02, 0.01]:
        iso, q_safe, q_unsafe = fit_hard450(alpha)
        for margin in [1.0, 1.05, 1.10, 1.20]:
            name = f"hard450_alpha{alpha:.2f}_margin{margin:.2f}"
            results["variants"][name] = {
                "calibration": "hard450",
                "alpha": alpha,
                "safe_margin": margin,
                "branches": {
                    b: summarize(cases, iso, q_safe, q_unsafe, margin) for b, cases in datasets.items()
                },
            }

    for alpha in [0.10, 0.05, 0.02]:
        for branch, cases in datasets.items():
            train_cases, test_cases = split_cases(cases)
            train_rows = [r for c in train_cases for r in c["candidates"]]
            iso, q_safe, q_unsafe = fit_candidate(train_rows, alpha)
            name = f"{branch}_hardneg_alpha{alpha:.2f}_heldout"
            results["variants"][name] = {
                "calibration": f"{branch}_candidate_hard_negative_train_half",
                "alpha": alpha,
                "safe_margin": 1.0,
                "branches": {branch: summarize(test_cases, iso, q_safe, q_unsafe, 1.0)},
                "n_train_cases": len(train_cases),
                "n_test_cases": len(test_cases),
            }

    for alpha in [0.10, 0.05, 0.02]:
        train_rows = []
        test_by_branch = {}
        n_train_cases = 0
        for branch, cases in datasets.items():
            train_cases, test_cases = split_cases(cases)
            n_train_cases += len(train_cases)
            train_rows.extend(r for c in train_cases for r in c["candidates"])
            test_by_branch[branch] = test_cases
        iso, q_safe, q_unsafe = fit_candidate(train_rows, alpha)
        name = f"combined_hardneg_alpha{alpha:.2f}_heldout"
        results["variants"][name] = {
            "calibration": "free_fixed_candidate_hard_negative_train_half",
            "alpha": alpha,
            "safe_margin": 1.0,
            "branches": {b: summarize(cases, iso, q_safe, q_unsafe, 1.0) for b, cases in test_by_branch.items()},
            "n_train_cases": n_train_cases,
            "n_test_cases": sum(len(cases) for cases in test_by_branch.values()),
        }

    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps({"output": str(args.output), "n_variants": len(results["variants"])}, indent=2))


if __name__ == "__main__":
    main()
