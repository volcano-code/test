#!/usr/bin/env python3
"""Plot a reproducible 100-safe/100-unsafe diagnostic for frozen dual ACS.

The subset is selected by a stable hash of task and path, never by prediction
score or error status.  ACS quantiles remain those fitted on calibration tasks.
This is a balanced diagnostic and must not replace the full sealed-test audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TAU = 1.3


def stable_key(row: dict, seed: str) -> str:
    token = f"{seed}|{row.get('task')}|{row.get('path')}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def decisions(pred: np.ndarray, q_safe: float, q_unsafe: float):
    threshold = math.log10(TAU)
    auto_safe = pred - q_safe >= threshold
    auto_unsafe = pred + q_unsafe < threshold
    conflict = auto_safe & auto_unsafe
    return auto_safe & ~conflict, auto_unsafe & ~conflict


def evaluate(rows: list[dict], q_safe: float, q_unsafe: float) -> dict:
    pred = np.log10(np.asarray([row["sf_pino"] for row in rows], dtype=float))
    truth_safe = np.asarray([row["truth_safe"] for row in rows], dtype=bool)
    auto_safe, auto_unsafe = decisions(pred, q_safe, q_unsafe)
    defer = ~(auto_safe | auto_unsafe)
    false_safe = auto_safe & ~truth_safe
    false_unsafe = auto_unsafe & truth_safe
    correct_safe = auto_safe & truth_safe
    correct_unsafe = auto_unsafe & ~truth_safe
    n_safe = int(truth_safe.sum())
    n_unsafe = int((~truth_safe).sum())
    return {
        "auto_decision_coverage": float((~defer).mean()),
        "correct_auto_rate": float((correct_safe | correct_unsafe).mean()),
        "auto_safe_rate": float(correct_safe.sum() / n_safe),
        "auto_unsafe_rate": float(correct_unsafe.sum() / n_unsafe),
        "false_safe_count": int(false_safe.sum()),
        "false_safe_rate_over_unsafe": float(false_safe.sum() / n_unsafe),
        "false_unsafe_count": int(false_unsafe.sum()),
        "defer_rate": float(defer.mean()),
        "n_safe": n_safe,
        "n_unsafe": n_unsafe,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acs-result", type=Path, required=True)
    parser.add_argument("--sealed-audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", default="20260719-balanced-acs")
    parser.add_argument("--per-class", type=int, default=100)
    args = parser.parse_args()

    result = json.loads(args.acs_result.read_text())
    audit = json.loads(args.sealed_audit.read_text())
    safe = sorted((row for row in audit if row["truth_safe"]),
                  key=lambda row: stable_key(row, args.seed))
    unsafe = sorted((row for row in audit if not row["truth_safe"]),
                    key=lambda row: stable_key(row, args.seed))
    if min(len(safe), len(unsafe)) < args.per_class:
        raise ValueError("The sealed audit does not contain enough rows per class.")
    subset = safe[:args.per_class] + unsafe[:args.per_class]

    selected = result["selected_policy"]
    selected_metrics = evaluate(
        subset, selected["q_safe_log_sf"], selected["q_unsafe_log_sf"]
    )
    frontier = []
    for policy in result["all_calibration_policies"]:
        metrics = evaluate(
            subset, policy["q_safe_log_sf"], policy["q_unsafe_log_sf"]
        )
        frontier.append({
            "alpha_safe": policy["alpha_safe"],
            "alpha_unsafe": policy["alpha_unsafe"],
            **metrics,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "balanced diagnostic; not a replacement for full sealed-test",
        "selection": "stable SHA-256 hash of task and path; independent of scores/errors",
        "seed": args.seed,
        "selected_policy": {
            "alpha_safe": selected["alpha_safe"],
            "alpha_unsafe": selected["alpha_unsafe"],
            "q_safe_log_sf": selected["q_safe_log_sf"],
            "q_unsafe_log_sf": selected["q_unsafe_log_sf"],
        },
        "metrics": selected_metrics,
        "rows": subset,
        "policy_frontier": frontier,
    }
    (args.out_dir / "balanced_acs_diagnostic.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    selected_rows = []
    pred = np.log10(np.asarray([row["sf_pino"] for row in subset], dtype=float))
    auto_safe, auto_unsafe = decisions(
        pred, selected["q_safe_log_sf"], selected["q_unsafe_log_sf"]
    )
    truth_safe = np.asarray([row["truth_safe"] for row in subset], dtype=bool)
    for mask in (truth_safe, ~truth_safe):
        selected_rows.append([
            100 * float(auto_safe[mask].mean()),
            100 * float(auto_unsafe[mask].mean()),
            100 * float((~(auto_safe | auto_unsafe))[mask].mean()),
        ])

    ax = axes[0]
    groups = ["True safe", "True unsafe"]
    colors = ["#00897B", "#FFB300", "#455A64"]
    labels_decision = ["SAFE", "UNSAFE", "DEFER"]
    left = np.zeros(2)
    values_by_decision = np.asarray(selected_rows).T
    for values, color, label in zip(values_by_decision, colors, labels_decision):
        bars = ax.barh(groups, values, left=left, color=color, height=0.52, label=label)
        for bar, value in zip(bars, values):
            if value >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.0f}%", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
        left += values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Decision composition (%)")
    ax.set_title("(a) Frozen ACS decisions")
    ax.grid(axis="x", color="#ECEFF1", linewidth=0.8)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, -0.34))

    ax = axes[1]
    labels = ["Correct auto", "Auto-safe", "Auto-unsafe", "FEM calls"]
    values = [
        selected_metrics["correct_auto_rate"],
        selected_metrics["auto_safe_rate"],
        selected_metrics["auto_unsafe_rate"],
        selected_metrics["defer_rate"],
    ]
    colors = ["#4F83CC", "#00897B", "#FFB300", "#455A64"]
    bars = ax.bar(labels, 100 * np.asarray(values), color=colors, width=0.68)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Rate (%)")
    ax.set_title("(b) Balanced diagnostic metrics")
    ax.tick_params(axis="x", rotation=24)
    ax.grid(axis="y", color="#ECEFF1", linewidth=0.8)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, 100 * value + 2,
                f"{100 * value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(
        0.02, 0.98,
        f"100 safe + 100 unsafe\nfalse-safe = {selected_metrics['false_safe_count']}/100",
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CFD8DC"},
    )

    fig.savefig(args.out_dir / "balanced_acs_risk_coverage.png", dpi=400, facecolor="white")
    fig.savefig(args.out_dir / "balanced_acs_risk_coverage.pdf", facecolor="white")
    print(json.dumps(selected_metrics, indent=2))


if __name__ == "__main__":
    main()
