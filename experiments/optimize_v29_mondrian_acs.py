#!/usr/bin/env python3
"""Explore Mondrian ACS policies for the frozen v29 ensemble.

The script does not retrain models and does not modify the authoritative
pipeline. It reuses cached v29 predictions, selects policies on candidate-domain
validation groups, and audits the selected policy on untouched candidate groups
and the external scaleup set.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_candidate_pino_field import load_field_map, load_scaleup_rows

YIELD_STRESS = 50e6
SF_TARGET = 1.3
SEED = 20260703


def load_candidate_rows() -> list[dict]:
    rows = []
    field_map = load_field_map()
    for branch_id, path in enumerate((ROOT / "data/aug.npz", ROOT / "data/aug_fixed.npz")):
        payload = np.load(path)
        branch = "free" if branch_id == 0 else "fixed"
        for index in range(len(payload["dens"])):
            group = index // 16
            geometry = (index % 16) // 4
            record = field_map.get((branch_id, group, geometry))
            if record is None or not record["valid"]:
                continue
            density = payload["dens"][index].astype(np.float32)
            rows.append({
                "branch": branch,
                "group": group,
                "geometry": geometry,
                "density": density,
                "volume_fraction": float(density.mean()),
                "multiplier": float(payload["mult"][index]),
                "log_vm": float(np.log10(max(float(payload["vm"][index]), 1e-12))),
                "sf": float(payload["sf"][index]),
            })
    return rows


def split_groups() -> dict[str, list[int]]:
    groups = np.arange(40)
    np.random.default_rng(SEED).shuffle(groups)
    return {
        "calibration": sorted(int(value) for value in groups[:16]),
        "validation": sorted(int(value) for value in groups[16:24]),
        "heldout": sorted(int(value) for value in groups[24:]),
    }


def subset_indices(rows: list[dict], groups: list[int]) -> np.ndarray:
    allowed = set(groups)
    return np.asarray([i for i, row in enumerate(rows) if row["group"] in allowed], dtype=int)


def prediction_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = np.load(ROOT / "results/v29_ensemble_candidate_predictions_20260703.npz")
    candidate_mean = np.mean(
        np.stack([cache[f"candidate_{name}"] for name in ("v29a", "v29b", "v29c")]),
        axis=0,
    )
    candidate_disagreement = np.abs(cache["candidate_v27"] - cache["candidate_v28"])
    external_mean = np.mean(
        np.stack([cache[f"external_{name}"] for name in ("v29a", "v29b", "v29c")]),
        axis=0,
    )
    external_disagreement = np.abs(cache["external_v27"] - cache["external_v28"])
    return candidate_mean, candidate_disagreement, external_mean, external_disagreement


def bucket_values(rows: list[dict], pred: np.ndarray, disagreement: np.ndarray, scheme: str, cuts=None) -> tuple[np.ndarray, list[float]]:
    if scheme == "global":
        return np.zeros(len(rows), dtype=int), []
    if scheme == "branch":
        return np.asarray([0 if row.get("branch") == "free" else 1 for row in rows], dtype=int), []
    if scheme == "volume3":
        values = np.asarray([row["volume_fraction"] for row in rows], dtype=float)
        if cuts is None:
            cuts = np.quantile(values, [1 / 3, 2 / 3]).tolist()
        return np.digitize(values, cuts).astype(int), list(cuts)
    if scheme == "load3":
        values = np.asarray([row["multiplier"] for row in rows], dtype=float)
        if cuts is None:
            cuts = np.quantile(values, [1 / 3, 2 / 3]).tolist()
        return np.digitize(values, cuts).astype(int), list(cuts)
    if scheme == "pred3":
        if cuts is None:
            cuts = np.quantile(pred, [1 / 3, 2 / 3]).tolist()
        return np.digitize(pred, cuts).astype(int), list(cuts)
    if scheme == "disagreement3":
        if cuts is None:
            cuts = np.quantile(disagreement, [1 / 3, 2 / 3]).tolist()
        return np.digitize(disagreement, cuts).astype(int), list(cuts)
    raise ValueError(f"Unknown bucket scheme: {scheme}")


def conformal_high(residual: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(residual, dtype=float))
    rank = int(np.ceil((1.0 - alpha) * (len(ordered) + 1)))
    return float(ordered[min(max(rank, 1), len(ordered)) - 1])


def bucket_calibration(
    pred: np.ndarray,
    disagreement: np.ndarray,
    rows: list[dict],
    cal_idx: np.ndarray,
    scheme: str,
    alpha: float,
    unsafe_alpha: float,
    fem_budget: float,
    min_bucket: int,
) -> dict:
    cal_rows = [rows[i] for i in cal_idx]
    cal_pred = pred[cal_idx]
    cal_dis = disagreement[cal_idx]
    cal_bucket, cuts = bucket_values(cal_rows, cal_pred, cal_dis, scheme)
    true_log = np.asarray([row["log_vm"] for row in cal_rows], dtype=float)
    residual = true_log - cal_pred
    global_values = {
        "q_safe": conformal_high(residual, alpha),
        "q_unsafe": float(np.quantile(residual, unsafe_alpha)),
        "trigger": float(np.quantile(cal_dis, 1.0 - fem_budget)),
    }
    bucket_params = {}
    for bucket in sorted(set(int(value) for value in cal_bucket)):
        mask = cal_bucket == bucket
        if int(mask.sum()) < min_bucket:
            bucket_params[str(bucket)] = {**global_values, "n": int(mask.sum()), "fallback_global": True}
            continue
        bucket_params[str(bucket)] = {
            "q_safe": conformal_high(residual[mask], alpha),
            "q_unsafe": float(np.quantile(residual[mask], unsafe_alpha)),
            "trigger": float(np.quantile(cal_dis[mask], 1.0 - fem_budget)),
            "n": int(mask.sum()),
            "fallback_global": False,
        }
    return {
        "scheme": scheme,
        "cuts": cuts,
        "alpha": alpha,
        "unsafe_alpha": unsafe_alpha,
        "fem_budget": fem_budget,
        "min_bucket": min_bucket,
        "global": global_values,
        "buckets": bucket_params,
    }


def evaluate_policy(
    pred: np.ndarray,
    disagreement: np.ndarray,
    rows: list[dict],
    idx: np.ndarray,
    policy: dict,
    external: bool = False,
) -> dict:
    selected_rows = [rows[i] for i in idx]
    selected_pred = pred[idx]
    selected_dis = disagreement[idx]
    if external and policy["scheme"] in {"branch", "load3"}:
        bucket = np.zeros(len(selected_rows), dtype=int)
    else:
        bucket, _ = bucket_values(selected_rows, selected_pred, selected_dis, policy["scheme"], policy["cuts"])

    q_safe = np.empty(len(selected_rows), dtype=float)
    q_unsafe = np.empty(len(selected_rows), dtype=float)
    trigger_threshold = np.empty(len(selected_rows), dtype=float)
    for i, value in enumerate(bucket):
        params = policy["buckets"].get(str(int(value)), policy["global"])
        q_safe[i] = params["q_safe"]
        q_unsafe[i] = params["q_unsafe"]
        trigger_threshold[i] = params["trigger"]

    true_sf = np.asarray([row["sf"] for row in selected_rows], dtype=float)
    true_safe = true_sf >= SF_TARGET
    lower_sf = YIELD_STRESS / np.maximum(10.0 ** (selected_pred + q_safe), 1e-12)
    upper_sf = YIELD_STRESS / np.maximum(10.0 ** (selected_pred + q_unsafe), 1e-12)
    predicted_safe = lower_sf >= SF_TARGET
    predicted_unsafe = upper_sf < SF_TARGET
    trigger = selected_dis >= trigger_threshold
    fem = trigger & ~predicted_unsafe & ~predicted_safe
    decided = ~fem
    final_safe = np.where(fem, true_safe, np.where(predicted_unsafe, False, predicted_safe))
    false_pass = final_safe & ~true_safe
    false_fail = ~final_safe & true_safe
    return {
        "n": len(selected_rows),
        "safe": int(true_safe.sum()),
        "unsafe": int((~true_safe).sum()),
        "autonomous_count": int(decided.sum()),
        "autonomous_rate": float(decided.mean()),
        "fem_count": int(fem.sum()),
        "system_false_pass": int(false_pass.sum()),
        "system_false_fail": int(false_fail.sum()),
        "safe_accept_count": int(final_safe.sum()),
        "true_safe_count": int(true_safe.sum()),
        "safe_accept_rate": float((final_safe & true_safe).sum() / max(true_safe.sum(), 1)),
        "unsafe_reject_count": int(predicted_unsafe.sum()),
        "predicted_safe_count": int(predicted_safe.sum()),
    }


def score(metrics: dict) -> tuple:
    zero_fp = metrics["system_false_pass"] == 0
    return (
        1 if zero_fp else 0,
        metrics["safe_accept_rate"] if zero_fp else -metrics["system_false_pass"],
        -metrics["fem_count"],
        -metrics["system_false_fail"],
    )


def main() -> None:
    candidate_rows = load_candidate_rows()
    candidate_mean, candidate_disagreement, external_mean, external_disagreement = prediction_cache()
    if len(candidate_rows) != len(candidate_mean):
        raise RuntimeError(f"Row/prediction mismatch: {len(candidate_rows)} rows vs {len(candidate_mean)} predictions")
    external_rows = load_scaleup_rows()
    for row in external_rows:
        row["volume_fraction"] = float(row["density"].mean())
        row["multiplier"] = 1.0

    splits = split_groups()
    cal_idx = subset_indices(candidate_rows, splits["calibration"])
    val_idx = subset_indices(candidate_rows, splits["validation"])
    heldout_idx = subset_indices(candidate_rows, splits["heldout"])
    external_idx = np.arange(len(external_rows), dtype=int)

    schemes = ["global", "branch", "volume3", "load3", "pred3", "disagreement3"]
    alphas = [0.02, 0.05, 0.10, 0.15, 0.20]
    unsafe_alphas = [0.10, 0.20, 0.30, 0.40]
    fem_budgets = [0.00, 0.04, 0.08, 0.12]
    rows = []
    for scheme in schemes:
        for alpha in alphas:
            for unsafe_alpha in unsafe_alphas:
                for fem_budget in fem_budgets:
                    policy = bucket_calibration(
                        candidate_mean, candidate_disagreement, candidate_rows,
                        cal_idx, scheme, alpha, unsafe_alpha, fem_budget, min_bucket=32,
                    )
                    validation = evaluate_policy(candidate_mean, candidate_disagreement, candidate_rows, val_idx, policy)
                    rows.append({"policy": policy, "validation": validation})

    zero_fp = [row for row in rows if row["validation"]["system_false_pass"] == 0]
    selected = max(zero_fp or rows, key=lambda row: score(row["validation"]))
    selected["candidate_heldout"] = evaluate_policy(
        candidate_mean, candidate_disagreement, candidate_rows, heldout_idx, selected["policy"]
    )
    selected["external_scaleup"] = evaluate_policy(
        external_mean, external_disagreement, external_rows, external_idx, selected["policy"], external=True
    )
    selected["selection_rule"] = (
        "selected on candidate validation only: zero false pass first, then max safe-accept rate, "
        "then fewer FEM fallbacks and false fails"
    )
    selected["promotion_gate"] = {
        "candidate_heldout_zero_false_pass": selected["candidate_heldout"]["system_false_pass"] == 0,
        "external_zero_false_pass": selected["external_scaleup"]["system_false_pass"] == 0,
        "candidate_safe_accept_at_least_50pct": selected["candidate_heldout"]["safe_accept_rate"] >= 0.50,
        "external_safe_accept_at_least_50pct": selected["external_scaleup"]["safe_accept_rate"] >= 0.50,
    }
    selected["promotion_eligible_for_fem_replacement_claim"] = all(selected["promotion_gate"].values())
    result = {
        "schema": "v29_mondrian_acs_search_v1",
        "created": "2026-07-03",
        "split_seed": SEED,
        "splits": splits,
        "num_policies": len(rows),
        "num_zero_fp_validation": len(zero_fp),
        "selected": selected,
        "top_validation_zero_fp": sorted(
            [{"policy": row["policy"], "validation": row["validation"]} for row in zero_fp],
            key=lambda row: score(row["validation"]),
            reverse=True,
        )[:10],
        "limitations": [
            "Policy selection uses candidate validation groups only; external scaleup is audit-only.",
            "Labels are voxel-FEM references, not physical measurements.",
            "This script does not modify checkpoints or the authoritative pipeline.",
        ],
    }
    output = ROOT / "results/v29_mondrian_acs_search_20260703.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "selected_policy": selected["policy"],
        "validation": selected["validation"],
        "candidate_heldout": selected["candidate_heldout"],
        "external_scaleup": selected["external_scaleup"],
        "promotion_eligible_for_fem_replacement_claim": selected["promotion_eligible_for_fem_replacement_claim"],
    }, indent=2))


if __name__ == "__main__":
    main()
