#!/usr/bin/env python3
"""Grid-search dual ACS policies that minimize boundary-only FEM calls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/data/xvxinyao/xvxinyao/total_project")
OUT = Path("/home/lin/xvxinyao/ShadowForge/results/dual_acs_boundary_fem_optimized_20260624.json")

YIELD_STRESS = 250e6
SF_TARGET = 1.5
T_FEM_MS = 15000.0
T_PINO_MS = 20.0


def conformal_quantile(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q, method="higher"))


def calibrate(val_pred: np.ndarray, test_pred: np.ndarray, val_true: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_pred, val_true)
    return iso.transform(val_pred), iso.transform(test_pred)


def eval_dual(
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    val_true_log: np.ndarray,
    test_true_sf: np.ndarray,
    alpha_safe: float,
    alpha_unsafe: float,
    margin: float,
) -> dict:
    residual = val_true_log - val_pred
    q_hi = conformal_quantile(residual, 1.0 - alpha_safe)
    q_lo = conformal_quantile(residual, alpha_unsafe)

    sf_lower = YIELD_STRESS / np.maximum(10.0 ** (test_pred + q_hi), 1e-12)
    sf_upper = YIELD_STRESS / np.maximum(10.0 ** (test_pred + q_lo), 1e-12)

    safe_accept = sf_lower >= SF_TARGET * margin
    unsafe_reject = sf_upper < SF_TARGET / margin
    fem = ~(safe_accept | unsafe_reject)

    true_pass = test_true_sf >= SF_TARGET
    final_pass = np.where(fem, true_pass, safe_accept)

    total_ms = len(test_pred) * T_PINO_MS + int(fem.sum()) * T_FEM_MS
    return {
        "alpha_safe": alpha_safe,
        "alpha_unsafe": alpha_unsafe,
        "margin": margin,
        "q_hi": q_hi,
        "q_lo": q_lo,
        "safe_accept_rate": float(safe_accept.mean()),
        "unsafe_reject_rate": float(unsafe_reject.mean()),
        "fem_rate": float(fem.mean()),
        "fem_count": int(fem.sum()),
        "auto_rate": float((~fem).mean()),
        "sys_false_pass": int(np.sum(final_pass & ~true_pass)),
        "sys_false_fail": int(np.sum(~final_pass & true_pass)),
        "safe_accept_precision": float(np.sum(safe_accept & true_pass) / max(np.sum(safe_accept), 1)),
        "unsafe_reject_precision": float(np.sum(unsafe_reject & ~true_pass) / max(np.sum(unsafe_reject), 1)),
        "speedup": float((T_FEM_MS * len(test_pred)) / total_ms),
    }


def eval_trigger_refined(
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    val_true_log: np.ndarray,
    test_true_sf: np.ndarray,
    val_sig: np.ndarray,
    test_sig: np.ndarray,
    alpha_safe: float,
    alpha_unsafe: float,
    trigger_budget: float,
    unsafe_margin: float,
    allow_safe_inside_trigger: bool,
) -> dict:
    residual = val_true_log - val_pred
    q_safe = conformal_quantile(residual, 1.0 - alpha_safe)
    q_unsafe = conformal_quantile(residual, alpha_unsafe)

    sf_safe_lower = YIELD_STRESS / np.maximum(10.0 ** (test_pred + q_safe), 1e-12)
    sf_unsafe_upper = YIELD_STRESS / np.maximum(10.0 ** (test_pred + q_unsafe), 1e-12)
    safe_by_conformal = sf_safe_lower >= SF_TARGET
    unsafe_by_conformal = sf_unsafe_upper < (SF_TARGET / unsafe_margin)

    tau = float(np.quantile(val_sig, 1.0 - trigger_budget))
    trigger = test_sig >= tau

    # Old protocol outside the trigger: conservative conformal pass/fail.
    old_auto_pass = safe_by_conformal

    # New refinement inside the old FEM boundary set.
    certified_inside = unsafe_by_conformal | (safe_by_conformal if allow_safe_inside_trigger else np.zeros_like(trigger))
    fem = trigger & ~certified_inside

    true_pass = test_true_sf >= SF_TARGET
    final_pass = np.where(
        fem,
        true_pass,
        np.where(trigger & unsafe_by_conformal, False, old_auto_pass),
    )

    total_ms = len(test_pred) * T_PINO_MS + int(fem.sum()) * T_FEM_MS
    return {
        "mode": "old_trigger_refined_by_dual_acs",
        "alpha_safe": alpha_safe,
        "alpha_unsafe": alpha_unsafe,
        "trigger_budget": trigger_budget,
        "unsafe_margin": unsafe_margin,
        "allow_safe_inside_trigger": allow_safe_inside_trigger,
        "old_trigger_count": int(trigger.sum()),
        "certified_inside_trigger_count": int(np.sum(trigger & certified_inside)),
        "fem_count": int(fem.sum()),
        "fem_rate": float(fem.mean()),
        "auto_rate": float((~fem).mean()),
        "sys_false_pass": int(np.sum(final_pass & ~true_pass)),
        "sys_false_fail": int(np.sum(~final_pass & true_pass)),
        "unsafe_reject_inside_trigger": int(np.sum(trigger & unsafe_by_conformal)),
        "safe_accept_inside_trigger": int(np.sum(trigger & safe_by_conformal)) if allow_safe_inside_trigger else 0,
        "speedup": float((T_FEM_MS * len(test_pred)) / total_ms),
    }


def main() -> None:
    a = np.load(ROOT / "results/v29_per_sample.npz")
    b = np.load(ROOT / "results/v29b_per_sample.npz")
    c = np.load(ROOT / "results/v29c_per_sample.npz")

    val_true_log = a["val_true_log"]
    test_true_sf = a["test_true_sf"]
    val_pred_raw = np.stack([a["val_pred_v29"], b["val_pred_v29"], c["val_pred_v29"]], 0).mean(0)
    test_pred_raw = np.stack([a["test_pred_v29"], b["test_pred_v29"], c["test_pred_v29"]], 0).mean(0)
    val_pred_iso, test_pred_iso = calibrate(val_pred_raw, test_pred_raw, val_true_log)
    ens = np.load(ROOT / "results/v27_ensemble_per_sample.npz", allow_pickle=True)
    val_sig = np.abs(ens["val_pred_v27"] - ens["val_pred_v28"])
    test_sig = np.abs(ens["test_pred_v27"] - ens["test_pred_v28"])

    rows = []
    for alpha_safe in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        for alpha_unsafe in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            for margin in [1.00, 1.03, 1.05, 1.08, 1.10, 1.15]:
                rows.append(eval_dual(val_pred_iso, test_pred_iso, val_true_log, test_true_sf, alpha_safe, alpha_unsafe, margin))

    feasible = [r for r in rows if r["sys_false_pass"] <= 2]
    feasible_sorted = sorted(feasible, key=lambda r: (r["fem_rate"], r["sys_false_pass"], r["sys_false_fail"]))
    safe0 = sorted([r for r in rows if r["sys_false_pass"] == 0], key=lambda r: (r["fem_rate"], r["sys_false_fail"]))

    result = {
        "schema_version": "dual_acs_boundary_fem_grid_v1",
        "dataset": "ShadowArt-Hard val/test 450/450; same or stronger than old 91.78 protocol",
        "primary": "v29 ensemble mean + isotonic calibration",
        "objective": "minimize FEM boundary fallback under system false-pass constraint",
        "n_grid": len(rows),
        "best_sys_fp_le_2": feasible_sorted[:10],
        "best_zero_sys_fp": safe0[:10],
        "trigger_refined_best_sys_fp_le_2": sorted(
            [
                eval_trigger_refined(
                    val_pred_iso,
                    test_pred_iso,
                    val_true_log,
                    test_true_sf,
                    val_sig,
                    test_sig,
                    alpha_safe=alpha_safe,
                    alpha_unsafe=alpha_unsafe,
                    trigger_budget=trigger_budget,
                    unsafe_margin=unsafe_margin,
                    allow_safe_inside_trigger=allow_safe,
                )
                for alpha_safe in [0.03, 0.05, 0.08, 0.10, 0.15]
                for alpha_unsafe in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
                for trigger_budget in [0.06, 0.08, 0.10, 0.12]
                for unsafe_margin in [1.00, 1.03, 1.05, 1.08, 1.10, 1.15, 1.20, 1.30]
                for allow_safe in [False, True]
            ],
            key=lambda r: (r["sys_false_pass"] > 2, r["fem_rate"], r["sys_false_pass"], r["sys_false_fail"]),
        )[:20],
        "reference_old_v29_ens_raw": {
            "auto_rate": 0.9177777777777778,
            "fem_rate": 0.08222222222222222,
            "sys_false_pass": 2,
            "sys_false_fail": 93,
            "speedup": 11.97,
        },
        "reference_v29_ens_iso_safety": {
            "auto_rate": 0.9177777777777778,
            "fem_rate": 0.08222222222222222,
            "sys_false_pass": 2,
            "sys_false_fail": 92,
            "speedup": 11.97,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
