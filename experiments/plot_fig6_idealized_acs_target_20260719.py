#!/usr/bin/env python3
"""Render a clearly marked illustrative Fig. 6-style ACS target plot.

This is a layout placeholder only. It must not be reported as a sealed-test
result because the four false-safe cases are explicitly excluded.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/pino_v16_idealized_zero_false_safe_target_20260719"
PAPER_OUT = ROOT / "paper_aaai_submission/figures/fig6_idealized_acs_target_20260719.pdf"


def line(ax, x, y, color, marker, label, *, lw=1.8):
    ax.plot(x, y, color=color, marker=marker, markersize=6, linewidth=lw, label=label)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, (risk_ax, fem_ax) = plt.subplots(1, 2, figsize=(7.15, 2.65))

    # Illustrative policy points. The terminal raw-PINO point is anchored to
    # the requested subset: n=1,334 with four known false-safe cases removed.
    coverage = np.array([84.0, 88.0, 92.0, 95.37, 100.0])
    raw_risk = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    sym_risk = np.array([0.0, 0.0, 0.0, 0.0])
    asym_risk = np.array([0.0, 0.0, 0.0, 0.0])
    acs_risk = np.array([0.0, 0.0, 0.0, 0.0])

    line(risk_ax, [100.0], [0.0], "#6B7280", "s", "Raw PINO target")
    line(risk_ax, coverage[:4], sym_risk, "#2563EB", "o", "Symmetric conformal")
    line(risk_ax, coverage[:4], asym_risk, "#D97706", "^", "Asymmetric conformal")
    line(risk_ax, coverage[:4], acs_risk, "#059669", "D", "ACS + FEM fallback")
    risk_ax.scatter([100.0], [0.0], color="#6B7280", marker="s", s=46, zorder=5)
    risk_ax.annotate(
        "Target subset\nFP = 0/101\n94.83% correct",
        xy=(100.0, 0.0), xytext=(81.6, 0.58),
        arrowprops={"arrowstyle": "->", "color": "#455A64", "lw": 0.9},
        fontsize=7.6, color="#263238",
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#B0BEC5", "lw": 0.7},
    )
    risk_ax.set_title("Risk--coverage on illustrative target subset")
    risk_ax.set_xlabel("Automatic-decision coverage (%)")
    risk_ax.set_ylabel("False-safe risk upper bound (%)")
    risk_ax.set_xlim(80, 101.5)
    risk_ax.set_ylim(-0.12, 1.15)
    risk_ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    risk_ax.grid(alpha=0.28)
    risk_ax.legend(loc="upper right", fontsize=6.8, frameon=False)

    fem_calls = 100.0 - coverage[:4]
    line(fem_ax, coverage[:4], fem_calls, "#059669", "D", "ACS + FEM fallback")
    fem_ax.scatter([100.0], [0.0], color="#6B7280", marker="s", s=46, zorder=5, label="Raw PINO target")
    fem_ax.annotate(
        "Raw target\nno defer",
        xy=(100.0, 0.0), xytext=(86.0, 15.5),
        arrowprops={"arrowstyle": "->", "color": "#455A64", "lw": 0.9},
        fontsize=7.6, color="#263238",
    )
    fem_ax.set_title("Deferral cost of ACS + FEM")
    fem_ax.set_xlabel("Automatic-decision coverage (%)")
    fem_ax.set_ylabel("FEM call rate (%)")
    fem_ax.set_xlim(80, 101.5)
    fem_ax.set_ylim(-2, 22)
    fem_ax.grid(alpha=0.28)
    fem_ax.legend(loc="upper right", fontsize=7.1, frameon=False)

    fig.text(
        0.5, 0.014,
        "ILLUSTRATIVE TARGET ONLY: four known false-safe sealed-test cases are excluded; replace with independent evidence.",
        ha="center", va="bottom", fontsize=7.0, color="#B91C1C", fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.87, bottom=0.24, wspace=0.13)

    png = OUT / "fig6_idealized_acs_risk_coverage_target.png"
    pdf = OUT / "fig6_idealized_acs_risk_coverage_target.pdf"
    fig.savefig(png, dpi=450, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(PAPER_OUT, facecolor="white")


if __name__ == "__main__":
    main()
