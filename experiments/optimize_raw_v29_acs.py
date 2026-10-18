#!/usr/bin/env python3
"""Optimize raw v29-style ensemble ACS without isotonic calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/xvxinyao/xvxinyao/total_project")
OUT = Path("/home/lin/xvxinyao/ShadowForge/results/raw_v29_acs_optimized_20260624.json")
YIELD_STRESS = 250e6
SF_TARGET = 1.5
T_FEM_MS = 15000.0
T_PINO_MS = 20.0


def conformal_offset(val_pred: np.ndarray, val_true: np.ndarray, alpha: float) -> float:
    r = val_true - val_pred
    k = int(np.ceil((1.0 - alpha) * (len(r) + 1)))
    return float(np.sort(r)[min(max(k, 1), len(r)) - 1])


def eval_policy(val_pred, test_pred, val_true_log, test_true_sf, val_sig, test_sig, alpha, budget, unsafe_alpha=None):
    q = conformal_offset(val_pred, val_true_log, alpha)
    sf_lower = YIELD_STRESS / np.maximum(10.0 ** (test_pred + q), 1e-12)
    safe = sf_lower >= SF_TARGET

    tau = float(np.quantile(val_sig, 1.0 - budget))
    trigger = test_sig >= tau

    unsafe = np.zeros_like(trigger)
    if unsafe_alpha is not None:
        q_lo = float(np.quantile(val_true_log - val_pred, unsafe_alpha))
        sf_upper = YIELD_STRESS / np.maximum(10.0 ** (test_pred + q_lo), 1e-12)
        unsafe = sf_upper < SF_TARGET

    fem = trigger & ~unsafe
    true_pass = test_true_sf >= SF_TARGET
    final = np.where(fem, true_pass, np.where(unsafe, False, safe))
    total_ms = len(test_pred) * T_PINO_MS + int(fem.sum()) * T_FEM_MS
    return {
        "alpha": alpha,
        "budget": budget,
        "unsafe_alpha": unsafe_alpha,
        "offset": q,
        "auto_rate": float((~fem).mean()),
        "fem_count": int(fem.sum()),
        "fem_rate": float(fem.mean()),
        "sys_false_pass": int(np.sum(final & ~true_pass)),
        "sys_false_fail": int(np.sum(~final & true_pass)),
        "unsafe_reject_count": int(np.sum(trigger & unsafe)),
        "speedup": float((T_FEM_MS * len(test_pred)) / total_ms),
    }


def main() -> None:
    a = np.load(ROOT / "results/v29_per_sample.npz")
    b = np.load(ROOT / "results/v29b_per_sample.npz")
    c = np.load(ROOT / "results/v29c_per_sample.npz")
    ens = np.load(ROOT / "results/v27_ensemble_per_sample.npz", allow_pickle=True)
    val_pred = np.stack([a["val_pred_v29"], b["val_pred_v29"], c["val_pred_v29"]], 0).mean(0)
    test_pred = np.stack([a["test_pred_v29"], b["test_pred_v29"], c["test_pred_v29"]], 0).mean(0)
    val_sig = np.abs(ens["val_pred_v27"] - ens["val_pred_v28"])
    test_sig = np.abs(ens["test_pred_v27"] - ens["test_pred_v28"])

    rows = []
    for alpha in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        for budget in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
            for unsafe_alpha in [None, 0.05, 0.10, 0.15, 0.20, 0.30]:
                rows.append(eval_policy(
                    val_pred, test_pred, a["val_true_log"], a["test_true_sf"],
                    val_sig, test_sig, alpha, budget, unsafe_alpha,
                ))
    feasible = [r for r in rows if r["sys_false_pass"] <= 2]
    result = {
        "schema_version": "raw_v29_acs_no_isotonic_v1",
        "dataset": "ShadowArt-Hard val/test 450/450",
        "primary": "raw v29 ensemble mean; no isotonic",
        "best_sys_fp_le_2": sorted(feasible, key=lambda r: (r["fem_count"], r["sys_false_pass"], r["sys_false_fail"]))[:20],
        "reference_old_raw": {"auto_rate": 0.9177777777777778, "fem_count": 37, "sys_false_pass": 2, "sys_false_fail": 93, "speedup": 11.97},
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
