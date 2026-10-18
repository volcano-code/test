#!/usr/bin/env python3
"""Render the paper risk--coverage figure from a frozen ACS evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


TEAL = "#00897B"
AMBER = "#E69F00"
BLUE = "#2F6FDB"
RED = "#C44E52"
SLATE = "#455A64"
GRID = "#DCE3E7"
TEXT = "#263238"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operating-alpha", type=float, default=0.10)
    return parser.parse_args()


def percent(value: float) -> float:
    return 100.0 * value


def rows_for(data: dict, policy: str) -> list[dict]:
    rows = [row for row in data["policies"] if row["policy"] == policy]
    return sorted(rows, key=lambda row: float(row["alpha"]))


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(SLATE)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    args = parse_args()
    data = json.loads(args.input.read_text())
    symmetric = rows_for(data, "Symmetric conformal")
    asymmetric = rows_for(data, "Asymmetric conformal")
    fallback = rows_for(data, "ACS + FEM fallback")
    raw = next(row for row in data["policies"] if row["policy"] == "Raw PINO threshold")
    chosen = next(row for row in fallback if float(row["alpha"]) == args.operating_alpha)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "text.color": TEXT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), constrained_layout=True)
    ax_risk, ax_cost = axes

    for rows, label, color, marker in (
        (symmetric, "Symmetric conformal", BLUE, "o"),
        (asymmetric, "Asymmetric ACS", AMBER, "^"),
    ):
        x = [percent(row["metrics"]["auto_decision_coverage"]) for row in rows]
        y = [percent(row["metrics"]["false_safe_risk_upper_95"]) for row in rows]
        ax_risk.plot(x, y, color=color, marker=marker, markersize=4.5,
                     linewidth=1.5, label=label)

    raw_metrics = raw["metrics"]
    ax_risk.scatter(
        percent(raw_metrics["auto_decision_coverage"]),
        percent(raw_metrics["false_safe_risk_upper_95"]),
        color=SLATE, marker="s", s=32, label="Raw PINO threshold", zorder=4,
    )

    chosen_x = percent(chosen["metrics"]["auto_decision_coverage"])
    chosen_risk = percent(chosen["metrics"]["false_safe_risk_upper_95"])
    ax_risk.scatter(chosen_x, chosen_risk, marker="*", s=115, color=TEAL,
                    edgecolor="white", linewidth=0.7, zorder=6,
                    label=r"Selected ACS ($\alpha=0.10$)")
    ax_risk.annotate(
        "0 observed false-safe\n(0 / 100 unsafe)",
        xy=(chosen_x, chosen_risk), xytext=(55, 1.08),
        arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.0},
        fontsize=7.2, color=TEAL,
    )
    ax_risk.set_title("(a) Calibrated screening risk", fontsize=10, weight="semibold")
    ax_risk.set_xlabel("Automatic-decision coverage (%)")
    ax_risk.set_ylabel("95% upper bound on false-safe risk (%)")
    ax_risk.set_xlim(10, 102)
    ax_risk.set_ylim(0, 4.0)
    ax_risk.xaxis.set_major_locator(MultipleLocator(20))
    ax_risk.yaxis.set_major_locator(MultipleLocator(1))
    style_axis(ax_risk)
    ax_risk.legend(frameon=False, fontsize=7, loc="upper left")

    x = [percent(row["metrics"]["auto_decision_coverage"]) for row in fallback]
    fem = [percent(row["metrics"]["fem_call_rate"]) for row in fallback]
    unsafe = [percent(row["metrics"]["auto_unsafe_coverage"]) for row in fallback]
    ax_cost.plot(x, fem, color=TEAL, marker="D", markersize=4.5,
                 linewidth=1.6, label="FEM call rate")
    ax_cost.plot(x, unsafe, color=RED, marker="X", markersize=5,
                 linewidth=1.4, label="Auto-unsafe coverage")
    ax_cost.axvline(chosen_x, color=AMBER, linestyle="--", linewidth=1.0)
    ax_cost.scatter(chosen_x, percent(chosen["metrics"]["fem_call_rate"]),
                    marker="*", s=115, color=AMBER, edgecolor="white",
                    linewidth=0.7, zorder=6)
    ax_cost.text(
        0.035, 0.96,
        "Frozen operating point  $\\alpha=0.10$\n"
        f"Auto decision: {chosen_x:.1f}%\n"
        f"Auto-safe: {percent(chosen['metrics']['auto_safe_coverage']):.1f}% of safe\n"
        f"Auto-unsafe: {percent(chosen['metrics']['auto_unsafe_coverage']):.1f}% of unsafe\n"
        f"FEM calls: {percent(chosen['metrics']['fem_call_rate']):.1f}%",
        transform=ax_cost.transAxes, va="top", ha="left", fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": GRID, "linewidth": 0.8},
    )
    ax_cost.set_title("(b) Coverage and fallback cost", fontsize=10, weight="semibold")
    ax_cost.set_xlabel("Automatic-decision coverage (%)")
    ax_cost.set_ylabel("Rate (%)")
    ax_cost.set_xlim(10, 102)
    ax_cost.set_ylim(0, 92)
    ax_cost.xaxis.set_major_locator(MultipleLocator(20))
    ax_cost.yaxis.set_major_locator(MultipleLocator(20))
    style_axis(ax_cost)
    ax_cost.legend(frameon=False, fontsize=7, loc="upper right")

    counts = data["rows"]
    fig.suptitle(
        "Task-disjoint structural triage on the sealed test set",
        fontsize=11.5, weight="semibold", color=TEXT,
    )
    fig.text(
        0.5, -0.015,
        "115 / 37 / 41 disjoint train / calibration / sealed tasks; "
        f"sealed candidates: {counts['sealed_test']} (1,147 safe; 100 unsafe).",
        ha="center", va="top", fontsize=7.2, color=SLATE,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=400, bbox_inches="tight",
                facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
