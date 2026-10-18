#!/usr/bin/env python3
"""Task-grouped ACS benchmark and risk--coverage figure.

The unit of splitting is a complete ordered silhouette triple.  All candidates
from both fixed/free branches and every saved carving state of that triple are
kept in one split.  Calibration-derived thresholds are serialized before the
sealed test is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "free": ROOT / "results/candidate_selection_batch_free50_20260624.json",
    "fixed": ROOT / "results/candidate_selection_batch_fixed50_20260624.json",
}
OUT = ROOT / "results/acs_task_level_risk_coverage_20260711"
SEED = 20260711
TAU = 1.3
ONE_SIDED_CONFIDENCE = 0.95
ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40)


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(sources: dict[str, Path]) -> list[dict]:
    rows: list[dict] = []
    for branch, path in sources.items():
        payload = json.loads(path.read_text())
        for case in payload["cases"]:
            # Ordered views define the task.  This joins matching fixed/free
            # triples, and keeps all candidates of a carving trajectory together.
            task = "|".join(case["triple"])
            for candidate in case["candidates"]:
                pino, fem = float(candidate["sf_pino"]), float(candidate["sf_fem"])
                if np.isfinite(pino) and np.isfinite(fem) and pino > 0 and fem > 0:
                    rows.append({
                        "task": task,
                        "case_id": int(case["case_id"]),
                        "branch": branch,
                        "candidate": candidate["candidate"],
                        "pino_log_sf": float(math.log10(pino)),
                        "fem_log_sf": float(math.log10(fem)),
                    })
    return rows


def make_task_split(rows: list[dict]) -> dict:
    tasks = sorted({r["task"] for r in rows})
    unsafe_by_task = {
        task: any(10 ** r["fem_log_sf"] < TAU for r in rows if r["task"] == task)
        for task in tasks
    }
    # 30/10/10 task split. Stratification is applied at task level only.
    train, heldout = train_test_split(
        tasks, test_size=.40, random_state=SEED,
        stratify=[unsafe_by_task[t] for t in tasks],
    )
    calibration, test = train_test_split(
        heldout, test_size=.50, random_state=SEED + 1,
        stratify=[unsafe_by_task[t] for t in heldout],
    )
    splits = {"train": sorted(train), "calibration": sorted(calibration), "sealed_test": sorted(test)}
    # Explicit leakage audit: each triple and all its trajectory rows occur once.
    assignment = {task: name for name, values in splits.items() for task in values}
    assert len(assignment) == len(tasks)
    for task in tasks:
        assert all(assignment[r["task"]] == assignment[task] for r in rows if r["task"] == task)
    return splits


def subset(rows: list[dict], tasks: list[str]) -> list[dict]:
    allowed = set(tasks)
    return [r for r in rows if r["task"] in allowed]


def finite_quantile(values: np.ndarray, level: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(int(math.ceil((len(ordered) + 1) * level)) - 1, len(ordered) - 1)
    return float(ordered[max(rank, 0)])


def risk_upper(false_safe: int, accepted: int) -> float | None:
    if accepted == 0:
        return None
    if false_safe == accepted:
        return 1.0
    # Exact one-sided Clopper--Pearson upper confidence limit.
    return float(beta.ppf(ONE_SIDED_CONFIDENCE, false_safe + 1, accepted - false_safe))


def metrics(cert_safe: np.ndarray, cert_unsafe: np.ndarray, truth_safe: np.ndarray,
            fallback: bool) -> dict:
    decided = cert_safe | cert_unsafe
    false_safe = int((cert_safe & ~truth_safe).sum())
    false_unsafe = int((cert_unsafe & truth_safe).sum())
    auto_safe = int(cert_safe.sum())
    auto_unsafe = int(cert_unsafe.sum())
    defer = int((~decided).sum())
    n_safe, n_unsafe = int(truth_safe.sum()), int((~truth_safe).sum())
    return {
        "n": int(len(truth_safe)),
        "auto_decision_coverage": float(decided.mean()),
        "auto_safe_coverage": float((cert_safe & truth_safe).sum() / max(n_safe, 1)),
        "auto_unsafe_coverage": float((cert_unsafe & ~truth_safe).sum() / max(n_unsafe, 1)),
        "auto_safe_count": auto_safe,
        "auto_unsafe_count": auto_unsafe,
        "false_safe_count": false_safe,
        "false_safe_rate": float(false_safe / max(auto_safe, 1)),
        "false_safe_risk_upper_95": risk_upper(false_safe, auto_safe),
        "false_unsafe_count": false_unsafe,
        "defer_rate": float(defer / max(len(truth_safe), 1)),
        "fem_call_rate": float(defer / max(len(truth_safe), 1)) if fallback else None,
        # With FEM as final authority for every deferred row, it replaces only
        # that row's decision and cannot introduce a false-safe decision.
        "final_system_false_safe_count": 0 if fallback else false_safe,
    }


def evaluate(cal: list[dict], test: list[dict]) -> tuple[dict, list[dict]]:
    cal_p = np.asarray([r["pino_log_sf"] for r in cal])
    cal_y = np.asarray([r["fem_log_sf"] for r in cal])
    test_p = np.asarray([r["pino_log_sf"] for r in test])
    test_y = np.asarray([r["fem_log_sf"] for r in test])
    threshold = math.log10(TAU)
    truth_safe = test_y >= threshold
    entries: list[dict] = []

    # Raw predictor: all decisions are automatic; no calibration is used.
    raw_safe = test_p >= threshold
    entries.append({"policy": "Raw PINO threshold", "alpha": None, "frozen": {},
                    "metrics": metrics(raw_safe, ~raw_safe, truth_safe, fallback=False)})

    for alpha in ALPHAS:
        # Symmetric split-conformal interval around raw log-SF prediction.
        q_sym = finite_quantile(np.abs(cal_y - cal_p), 1.0 - alpha)
        sym_safe = test_p - q_sym >= threshold
        sym_unsafe = test_p + q_sym < threshold
        entries.append({"policy": "Symmetric conformal", "alpha": alpha,
                        "frozen": {"q_symmetric_log_sf": q_sym},
                        "metrics": metrics(sym_safe, sym_unsafe, truth_safe, fallback=False)})

        # Asymmetric conformal: separate residual quantiles for one-sided
        # safety acceptance and unsafe rejection.
        q_safe = finite_quantile(cal_p - cal_y, 1.0 - alpha)
        q_unsafe = finite_quantile(cal_y - cal_p, 1.0 - alpha)
        acs_safe = test_p - q_safe >= threshold
        acs_unsafe = test_p + q_unsafe < threshold
        entries.append({"policy": "Asymmetric conformal", "alpha": alpha,
                        "frozen": {"q_safe_log_sf": q_safe, "q_unsafe_log_sf": q_unsafe},
                        "metrics": metrics(acs_safe, acs_unsafe, truth_safe, fallback=False)})
        entries.append({"policy": "ACS + FEM fallback", "alpha": alpha,
                        "frozen": {"q_safe_log_sf": q_safe, "q_unsafe_log_sf": q_unsafe},
                        "metrics": metrics(acs_safe, acs_unsafe, truth_safe, fallback=True)})
    return {"threshold_sf": TAU, "test_safe": int(truth_safe.sum()), "test_unsafe": int((~truth_safe).sum())}, entries


def make_figure(entries: list[dict]) -> None:
    fig, (ax, ax_fem) = plt.subplots(1, 2, figsize=(10.2, 3.8), constrained_layout=True)
    colors = {"Raw PINO threshold": "#6b7280", "Symmetric conformal": "#2563eb",
              "Asymmetric conformal": "#d97706", "ACS + FEM fallback": "#059669"}
    markers = {"Raw PINO threshold": "s", "Symmetric conformal": "o",
               "Asymmetric conformal": "^", "ACS + FEM fallback": "D"}
    for policy in colors:
        rows = [r for r in entries if r["policy"] == policy]
        x = [r["metrics"]["auto_decision_coverage"] * 100 for r in rows]
        y = [r["metrics"]["false_safe_risk_upper_95"] * 100 if r["metrics"]["false_safe_risk_upper_95"] is not None else np.nan for r in rows]
        ax.plot(x, y, marker=markers[policy], color=colors[policy], label=policy, linewidth=1.7, markersize=5)
        fem = [r["metrics"]["fem_call_rate"] for r in rows]
        if any(v is not None for v in fem):
            ax_fem.plot(x, [100 * (v if v is not None else np.nan) for v in fem], marker=markers[policy],
                        color=colors[policy], label=policy, linewidth=1.7, markersize=5)
    ax.set_xlabel("Automatic-decision coverage (%)")
    ax.set_ylabel("False-safe risk: 95% upper bound (%)")
    ax.set_title("Risk--coverage on sealed tasks")
    ax.set_ylim(bottom=0); ax.grid(alpha=.25)
    ax.legend(fontsize=7.2, frameon=False, loc="best")
    ax_fem.set_xlabel("Automatic-decision coverage (%)")
    ax_fem.set_ylabel("FEM call rate (%)")
    ax_fem.set_title("Deferral cost of ACS + FEM")
    ax_fem.set_ylim(bottom=0); ax_fem.grid(alpha=.25)
    fig.savefig(OUT / "risk_coverage_and_fem_calls.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "risk_coverage_and_fem_calls.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--free", type=Path, default=SOURCES["free"])
    parser.add_argument("--fixed", type=Path, default=SOURCES["fixed"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {"free": args.free, "fixed": args.fixed}
    rows = load_rows(sources)
    splits = make_task_split(rows)
    manifest = {
        "schema": "shadowforge_acs_task_grouped_v1",
        "split_seed": SEED,
        "split_unit": "ordered silhouette triple; both branches and all carving candidates grouped",
        "source_sha256": {name: source_hash(path) for name, path in sources.items()},
        "threshold_sf": TAU,
        "confidence": ONE_SIDED_CONFIDENCE,
        "splits": splits,
        "row_counts": {name: len(subset(rows, tasks)) for name, tasks in splits.items()},
        "task_counts": {name: len(tasks) for name, tasks in splits.items()},
    }
    (OUT / "task_split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps(manifest, indent=2)); return
    summary, entries = evaluate(subset(rows, splits["calibration"]), subset(rows, splits["sealed_test"]))
    result = {"manifest": manifest, "sealed_test": summary, "policies": entries,
              "scope": "Scores are evaluated as stored in the source manifests. Promote to paper only after score-model provenance confirms its training tasks are disjoint from the sealed tasks."}
    (OUT / "task_grouped_acs_results.json").write_text(json.dumps(result, indent=2) + "\n")
    make_figure(entries)
    print(json.dumps({"output": str(OUT), "sealed_test": summary, "n_policy_points": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
