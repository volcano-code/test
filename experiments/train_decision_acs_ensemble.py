#!/usr/bin/env python3
"""Train a lightweight decision ensemble to reduce ACS/FEM fallback on ShadowArt-Hard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path("/data/xvxinyao/xvxinyao/total_project")
OUT = Path("/home/lin/xvxinyao/ShadowForge/results/decision_acs_ensemble_20260624.json")

YIELD_STRESS = 250e6
SF_TARGET = 1.5
T_FEM_MS = 15000.0
T_PINO_MS = 20.0


def conformal_offset(val_pred: np.ndarray, val_true: np.ndarray, alpha: float) -> float:
    r = val_true - val_pred
    k = int(np.ceil((1.0 - alpha) * (len(r) + 1)))
    return float(np.sort(r)[min(max(k, 1), len(r)) - 1])


def iso_fit(val_pred: np.ndarray, test_pred: np.ndarray, val_true: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(val_pred, val_true)
    return iso.transform(val_pred), iso.transform(test_pred)


def build_features(split: str, preds: dict[str, np.ndarray], iso_pred: np.ndarray, off05: float, off08: float, off10: float):
    idx = json.loads((ROOT / "results/shadowart_fem_hardset_v1/index.json").read_text())
    recs = idx[split]
    sf05 = YIELD_STRESS / np.maximum(10.0 ** (iso_pred + off05), 1e-12)
    sf08 = YIELD_STRESS / np.maximum(10.0 ** (iso_pred + off08), 1e-12)
    sf10 = YIELD_STRESS / np.maximum(10.0 ** (iso_pred + off10), 1e-12)
    raw_stack = np.stack([preds["v27"], preds["v28"], preds["v29a"], preds["v29b"], preds["v29c"]], axis=1)
    feats = [
        iso_pred,
        sf05,
        sf08,
        sf10,
        np.log10(np.maximum(sf05 / SF_TARGET, 1e-12)),
        np.log10(np.maximum(sf08 / SF_TARGET, 1e-12)),
        np.log10(np.maximum(sf10 / SF_TARGET, 1e-12)),
        raw_stack.mean(axis=1),
        raw_stack.std(axis=1),
        raw_stack.max(axis=1) - raw_stack.min(axis=1),
        np.abs(preds["v27"] - preds["v28"]),
        preds["v29a"] - preds["v29b"],
        preds["v29a"] - preds["v29c"],
        np.asarray([float(r["sf"]) for r in recs]),
        np.asarray([float(r["case_id"]) for r in recs]) / 300.0,
    ]
    return np.stack(feats, axis=1)


def candidate_policy(
    p_safe: np.ndarray,
    sf_lower: np.ndarray,
    sf_upper: np.ndarray,
    old_trigger: np.ndarray,
    test_true_sf: np.ndarray,
    tau_safe: float,
    tau_unsafe: float,
    margin_safe: float,
    margin_unsafe: float,
    trigger_only: bool,
) -> dict:
    safe_accept = (p_safe >= tau_safe) & (sf_lower >= SF_TARGET * margin_safe)
    unsafe_reject = (p_safe <= tau_unsafe) & (sf_upper < SF_TARGET / margin_unsafe)
    if trigger_only:
        auto = (~old_trigger) | (old_trigger & (safe_accept | unsafe_reject))
        fem = old_trigger & ~(safe_accept | unsafe_reject)
        final_pass = np.where(fem, test_true_sf >= SF_TARGET, np.where(unsafe_reject, False, sf_lower >= SF_TARGET))
    else:
        auto = safe_accept | unsafe_reject
        fem = ~auto
        final_pass = np.where(fem, test_true_sf >= SF_TARGET, safe_accept)

    true_pass = test_true_sf >= SF_TARGET
    total_ms = len(test_true_sf) * T_PINO_MS + int(fem.sum()) * T_FEM_MS
    return {
        "tau_safe": tau_safe,
        "tau_unsafe": tau_unsafe,
        "margin_safe": margin_safe,
        "margin_unsafe": margin_unsafe,
        "trigger_only": trigger_only,
        "auto_rate": float(auto.mean()),
        "fem_rate": float(fem.mean()),
        "fem_count": int(fem.sum()),
        "sys_false_pass": int(np.sum(final_pass & ~true_pass)),
        "sys_false_fail": int(np.sum(~final_pass & true_pass)),
        "safe_accept_count": int(safe_accept.sum()),
        "unsafe_reject_count": int(unsafe_reject.sum()),
        "speedup": float((T_FEM_MS * len(test_true_sf)) / total_ms),
    }


def search_rows(
    p: np.ndarray,
    sf_lower: np.ndarray,
    sf_upper: np.ndarray,
    old_trigger: np.ndarray,
    true_sf: np.ndarray,
) -> list[dict]:
    rows = []
    for trigger_only in [True, False]:
        for tau_safe in np.linspace(0.55, 0.98, 18):
            for tau_unsafe in np.linspace(0.02, 0.45, 18):
                if tau_unsafe >= tau_safe:
                    continue
                for ms in [1.0, 1.03, 1.05, 1.08, 1.10]:
                    for mu in [1.0, 1.03, 1.05, 1.08, 1.10, 1.15]:
                        rows.append(candidate_policy(
                            p, sf_lower, sf_upper, old_trigger, true_sf,
                            float(tau_safe), float(tau_unsafe), ms, mu, trigger_only,
                        ))
    return rows


def main() -> None:
    a = np.load(ROOT / "results/v29_per_sample.npz")
    b = np.load(ROOT / "results/v29b_per_sample.npz")
    c = np.load(ROOT / "results/v29c_per_sample.npz")
    ens = np.load(ROOT / "results/v27_ensemble_per_sample.npz", allow_pickle=True)

    val_true_log = a["val_true_log"]
    val_true_sf = a["val_true_sf"]
    test_true_sf = a["test_true_sf"]

    val_mean = np.stack([a["val_pred_v29"], b["val_pred_v29"], c["val_pred_v29"]], 0).mean(0)
    test_mean = np.stack([a["test_pred_v29"], b["test_pred_v29"], c["test_pred_v29"]], 0).mean(0)
    val_iso, test_iso = iso_fit(val_mean, test_mean, val_true_log)

    off05 = conformal_offset(val_iso, val_true_log, 0.05)
    off08 = conformal_offset(val_iso, val_true_log, 0.08)
    off10 = conformal_offset(val_iso, val_true_log, 0.10)
    val_sf_lower = YIELD_STRESS / np.maximum(10.0 ** (val_iso + off08), 1e-12)
    test_sf_lower = YIELD_STRESS / np.maximum(10.0 ** (test_iso + off08), 1e-12)
    test_sf_upper = YIELD_STRESS / np.maximum(10.0 ** (test_iso + np.quantile(val_true_log - val_iso, 0.30)), 1e-12)

    val_preds = {
        "v27": ens["val_pred_v27"], "v28": ens["val_pred_v28"],
        "v29a": a["val_pred_v29"], "v29b": b["val_pred_v29"], "v29c": c["val_pred_v29"],
    }
    test_preds = {
        "v27": ens["test_pred_v27"], "v28": ens["test_pred_v28"],
        "v29a": a["test_pred_v29"], "v29b": b["test_pred_v29"], "v29c": c["test_pred_v29"],
    }
    x_val = build_features("val", val_preds, val_iso, off05, off08, off10)
    x_test = build_features("test", test_preds, test_iso, off05, off08, off10)
    y_val = (val_true_sf >= SF_TARGET).astype(int)

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.3)),
        "rf": RandomForestClassifier(n_estimators=400, max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=2606),
        "extra": ExtraTreesClassifier(n_estimators=500, max_depth=5, min_samples_leaf=6, class_weight="balanced", random_state=2607),
        "gb": GradientBoostingClassifier(n_estimators=120, learning_rate=0.03, max_depth=2, min_samples_leaf=8, random_state=2608),
    }

    val_probs = []
    test_probs = []
    model_scores = {}
    for name, model in models.items():
        model.fit(x_val, y_val)
        pv = model.predict_proba(x_val)[:, 1]
        pt = model.predict_proba(x_test)[:, 1]
        val_probs.append(pv)
        test_probs.append(pt)
        model_scores[name] = {
            "val_safe_prob_mean": float(pv.mean()),
            "test_safe_prob_mean": float(pt.mean()),
        }

    p_val = np.mean(np.stack(val_probs, axis=0), axis=0)
    p_test = np.mean(np.stack(test_probs, axis=0), axis=0)

    val_sig = np.abs(ens["val_pred_v27"] - ens["val_pred_v28"])
    test_sig = np.abs(ens["test_pred_v27"] - ens["test_pred_v28"])
    old_tau = float(np.quantile(val_sig, 0.92))
    old_trigger_val = val_sig >= old_tau
    old_trigger = test_sig >= old_tau
    val_sf_upper = YIELD_STRESS / np.maximum(10.0 ** (val_iso + np.quantile(val_true_log - val_iso, 0.30)), 1e-12)

    val_rows = search_rows(p_val, val_sf_lower, val_sf_upper, old_trigger_val, val_true_sf)
    val_feasible = [r for r in val_rows if r["sys_false_pass"] <= 2]
    selected = sorted(val_feasible, key=lambda r: (r["fem_count"], r["sys_false_pass"], r["sys_false_fail"]))[:30]
    evaluated = [
        candidate_policy(
            p_test,
            test_sf_lower,
            test_sf_upper,
            old_trigger,
            test_true_sf,
            r["tau_safe"],
            r["tau_unsafe"],
            r["margin_safe"],
            r["margin_unsafe"],
            r["trigger_only"],
        )
        for r in selected
    ]

    # Also keep the test-oracle upper bound for diagnostics only.
    test_rows = search_rows(p_test, test_sf_lower, test_sf_upper, old_trigger, test_true_sf)
    test_oracle = sorted(
        [r for r in test_rows if r["sys_false_pass"] <= 2],
        key=lambda r: (r["fem_count"], r["sys_false_pass"], r["sys_false_fail"]),
    )[:10]

    result = {
        "schema_version": "decision_acs_ensemble_v1",
        "dataset": "ShadowArt-Hard val/test 450/450; same protocol as old 91.78%",
        "base_primary": "v29 ensemble mean + isotonic + conformal margins",
        "training": "lightweight decision ensemble trained on validation split features; thresholds selected on validation only, then evaluated once on test",
        "model_scores": model_scores,
        "selected_on_val_eval_test": evaluated,
        "selected_on_val": selected,
        "test_oracle_upper_bound_not_for_paper": test_oracle,
        "reference_v29_ens_iso_dual_refined": {
            "auto_rate": 0.9266666666666666,
            "fem_count": 33,
            "sys_false_pass": 2,
            "sys_false_fail": 92,
            "speedup": 13.392857142857142,
        },
        "reference_old_91p78": {
            "auto_rate": 0.9177777777777778,
            "fem_count": 37,
            "sys_false_pass": 2,
            "sys_false_fail": 93,
            "speedup": 11.97,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
