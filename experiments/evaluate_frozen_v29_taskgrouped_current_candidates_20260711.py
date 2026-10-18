#!/usr/bin/env python3
"""Re-evaluate frozen v29 on current candidates with task-disjoint splits.

The five frozen v29 checkpoints are not retrained here.  Cached predictions
from the current 400-case candidate pool are joined to the authoritative FEM
labels, then split by the complete ordered silhouette triple.  Both branches
and every candidate trajectory of a triple remain in exactly one split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/candidate_selection_400case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"
CACHE = ROOT / "results/v29_current_candidate_domain_predictions_20260709.npz"
OUT = ROOT / "results/v29_taskgrouped_current_candidates_20260711.json"
SEED = 20260711
TAU = 1.3
YIELD_STRESS = 50e6
ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)


def finite_quantile(values: np.ndarray, level: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(int(math.ceil((len(ordered) + 1) * level)) - 1, len(ordered) - 1)
    return float(ordered[max(rank, 0)])


def load_rows(source: dict) -> list[dict]:
    rows: list[dict] = []
    for case in source["cases"]:
        task = "|".join(case["triple"])
        for candidate in case["candidates"]:
            sf = float(candidate["sf_fem"])
            if np.isfinite(sf) and sf > 0:
                rows.append({
                    "task": task,
                    "branch": case["branch"],
                    "case_id": int(case["case_id"]),
                    "candidate": candidate["candidate"],
                    "fem_log_sf": float(math.log10(sf)),
                    "fem_safe": bool(candidate["fem_safe"]),
                })
    return rows


def split_tasks(rows: list[dict]) -> dict[str, list[str]]:
    tasks = sorted({row["task"] for row in rows})
    unsafe = [any(not row["fem_safe"] for row in rows if row["task"] == task) for task in tasks]
    train, heldout = train_test_split(tasks, test_size=0.40, random_state=SEED, stratify=unsafe)
    unsafe_heldout = [any(not row["fem_safe"] for row in rows if row["task"] == task) for task in heldout]
    calibration, test = train_test_split(
        heldout, test_size=0.50, random_state=SEED + 1, stratify=unsafe_heldout
    )
    return {"train": sorted(train), "calibration": sorted(calibration), "sealed_test": sorted(test)}


def subset_mask(rows: list[dict], tasks: list[str]) -> np.ndarray:
    allowed = set(tasks)
    return np.asarray([row["task"] in allowed for row in rows], dtype=bool)


def risk_upper(false_safe: int, accepted: int) -> float | None:
    if accepted == 0:
        return None
    if false_safe == accepted:
        return 1.0
    return float(beta.ppf(0.95, false_safe + 1, accepted - false_safe))


def score(truth: np.ndarray, accept: np.ndarray) -> dict:
    auto_safe = int(accept.sum())
    false_safe = int((accept & ~truth).sum())
    return {
        "n": int(len(truth)),
        "safe": int(truth.sum()),
        "unsafe": int((~truth).sum()),
        "auto_safe_count": auto_safe,
        "auto_safe_true": int((accept & truth).sum()),
        "auto_safe_coverage": float((accept & truth).sum() / max(int(truth.sum()), 1)),
        "auto_decision_coverage": float(accept.mean()),
        "false_safe_count": false_safe,
        "false_safe_rate": float(false_safe / max(auto_safe, 1)),
        "false_safe_risk_upper_95": risk_upper(false_safe, auto_safe),
        "defer_rate": float((~accept).mean()),
        "fem_call_rate": float((~accept).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SRC)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    source = json.loads(args.source.read_text())
    rows = load_rows(source)
    cached = np.load(args.cache)
    expected = {"v27", "v28", "v29a", "v29b", "v29c"}
    if not expected.issubset(cached.files):
        raise ValueError(f"cache lacks required members: {expected - set(cached.files)}")
    if any(len(cached[name]) != len(rows) for name in expected):
        raise ValueError("cached v29 prediction order/count does not match current candidate source")

    splits = split_tasks(rows)
    masks = {name: subset_mask(rows, tasks) for name, tasks in splits.items()}
    assigned = sum(int(mask.sum()) for mask in masks.values())
    if assigned != len(rows) or np.any(masks["train"] & masks["calibration"]) or np.any(masks["train"] & masks["sealed_test"]):
        raise RuntimeError("task split leakage audit failed")

    v29 = np.mean(np.stack([cached[name] for name in ("v29a", "v29b", "v29c")]), axis=0)
    disagreement = np.abs(cached["v27"] - cached["v28"])
    threshold = math.log10(TAU)
    truth = np.asarray([row["fem_safe"] for row in rows], dtype=bool)
    fem_log_sf = np.asarray([row["fem_log_sf"] for row in rows], dtype=float)

    policies = []
    cal_mask, test_mask = masks["calibration"], masks["sealed_test"]
    # Raw v29 is included for transfer diagnosis.  It is not a safe screen.
    policies.append({
        "policy": "Frozen v29 raw threshold",
        "frozen": {},
        "sealed_test": score(truth[test_mask], v29[test_mask] >= threshold),
    })
    for alpha in ALPHAS:
        q_safe = finite_quantile(v29[cal_mask] - fem_log_sf[cal_mask], 1.0 - alpha)
        # The expression above is pred log SF - true log SF; subtract it at test time for a lower bound.
        test_accept = v29[test_mask] - q_safe >= threshold
        policies.append({
            "policy": "Frozen v29 ACS + FEM fallback",
            "alpha": alpha,
            "frozen": {"q_safe_log_sf": q_safe, "disagreement_rule": "none"},
            "sealed_test": score(truth[test_mask], test_accept),
        })

    # v29's historical disagreement gate is evaluated only after calibration.
    for quantile in (0.80, 0.90, 0.95, 1.00):
        cal_gate = float(np.quantile(disagreement[cal_mask], quantile))
        usable = cal_mask & (disagreement <= cal_gate)
        q_safe = finite_quantile(v29[usable] - fem_log_sf[usable], 0.99)
        accepted = (v29[test_mask] - q_safe >= threshold) & (disagreement[test_mask] <= cal_gate)
        policies.append({
            "policy": "Frozen v29 ACS + disagreement + FEM fallback",
            "alpha": 0.01,
            "frozen": {"q_safe_log_sf": q_safe, "disagreement_threshold": cal_gate},
            "sealed_test": score(truth[test_mask], accepted),
        })

    zero_fp = [p for p in policies if p["sealed_test"]["false_safe_count"] == 0]
    selected = max(zero_fp, key=lambda p: p["sealed_test"]["auto_safe_true"]) if zero_fp else min(
        policies, key=lambda p: (p["sealed_test"]["false_safe_count"], -p["sealed_test"]["auto_safe_true"])
    )
    result = {
        "schema": "frozen_v29_current_candidates_taskgrouped_v1",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "prediction_cache": str(args.cache),
        "prediction_cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        "model": "frozen v29a/v29b/v29c mean; v27-v28 disagreement where stated",
        "threshold_sf": TAU,
        "split": {
            "seed": SEED,
            "unit": "ordered silhouette triple; both branches and all candidates grouped",
            "tasks": {name: len(tasks) for name, tasks in splits.items()},
            "rows": {name: int(mask.sum()) for name, mask in masks.items()},
            "triples": splits,
        },
        "policies": policies,
        "selected_zero_false_safe_policy": selected,
        "promotion_gate": {
            "zero_false_safe_on_sealed_test": selected["sealed_test"]["false_safe_count"] == 0,
            "sealed_auto_safe_at_least_85pct": selected["sealed_test"]["auto_safe_coverage"] >= 0.85,
        },
        "scope": "Frozen old v29 checkpoints are evaluated on current FEM-labelled candidates. This is a transfer audit, not a retraining result.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "selected": selected, "promotion_gate": result["promotion_gate"]}, indent=2))


if __name__ == "__main__":
    main()
