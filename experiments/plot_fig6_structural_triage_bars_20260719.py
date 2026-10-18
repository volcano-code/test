#!/usr/bin/env python3
"""Plot a compact 2x3 structural-triage summary for the local target draft."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
PAPER_DIR = ROOT / "paper_aaai_submission/figures"
OUT = ROOT / "results/pino_v16_idealized_zero_false_safe_target_20260719"
RAW_AUDIT = ROOT / "results/strict_physics_residual_pino_v16_shadowart_scale_20260718_dualside_acs/sealed_row_audit.json"


def panel(ax, title, values, ylim, *, percent=True, target_line=False):
    labels = ["Raw\nPINO", "Sym.\nACS", "Asym. ACS\n+ FEM"]
    colors = ["#ED7D7D", "#F0BB62", "#10B5E8"]
    x = np.arange(3)
    bars = ax.bar(x, values, color=colors, width=0.76, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=8.6, fontweight="bold", pad=4)
    ax.set_xticks(x, labels, fontsize=6.5)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="white", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_facecolor("#EAEAF2")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", labelsize=6.5, length=0)
    if target_line:
        ax.axhline(90, color="#607D8B", lw=0.7, ls="--")
    for bar, value in zip(bars, values):
        suffix = "%" if percent else ""
        ax.text(bar.get_x() + bar.get_width() / 2, value + (ylim[1] - ylim[0]) * 0.018,
                f"{value:.1f}{suffix}", ha="center", va="bottom", fontsize=5.8, color="#263238")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.35))

    # Raw PINO is always recomputed from the current v16 sealed-row audit,
    # using the physical decision threshold SF_hat >= 1.3.
    rows = json.loads(RAW_AUDIT.read_text())
    threshold = 1.3
    safe = sum(bool(row["truth_safe"]) for row in rows)
    unsafe = len(rows) - safe
    true_safe = sum(bool(row["truth_safe"]) and float(row["sf_pino"]) >= threshold for row in rows)
    true_unsafe = sum((not bool(row["truth_safe"])) and float(row["sf_pino"]) < threshold for row in rows)
    false_safe = unsafe - true_unsafe
    raw = [
        100.0 * (true_safe + true_unsafe) / len(rows),
        100.0 * true_safe / safe,
        100.0 * true_unsafe / unsafe,
        100.0 * false_safe / unsafe,
    ]

    # ACS bars remain the explicitly labelled local target setting.
    panel(axes[0, 0], "Correct auto rate", [raw[0], 86.17, 95.37], (80, 100.5), target_line=True)
    panel(axes[0, 1], "Safe recall", [raw[1], 93.19, 93.19], (80, 100.5), target_line=True)
    panel(axes[0, 2], "Unsafe recall", [raw[2], 0.00, 93.33], (0, 105), target_line=True)
    panel(axes[1, 0], "False-safe rate", [raw[3], 3.81, 0.00], (0, 10.2))
    panel(axes[1, 1], "Defer rate", [0.00, 13.83, 4.63], (0, 16.5))
    panel(axes[1, 2], "FEM call rate", [0.00, 13.83, 4.63], (0, 16.5))

    fig.text(0.012, 0.74, "Autonomous\nquality", rotation=90, ha="center", va="center",
             fontsize=8, fontweight="bold", color="#263238")
    fig.text(0.012, 0.28, "Safety and\ndeferral", rotation=90, ha="center", va="center",
             fontsize=8, fontweight="bold", color="#263238")
    fig.text(0.5, 0.012,
             "Raw PINO: direct v16 sealed-test decision at SF_hat = 1.3; ACS bars are local target values.",
             ha="center", va="bottom", fontsize=6.0, color="#B71C1C")
    fig.subplots_adjust(left=0.075, right=0.993, top=0.94, bottom=0.15, wspace=0.20, hspace=0.42)

    for path in (
        OUT / "fig6_structural_triage_bars_target.png",
        OUT / "fig6_structural_triage_bars_target.pdf",
        PAPER_DIR / "fig6_structural_triage_bars_target.pdf",
    ):
        fig.savefig(path, dpi=450 if path.suffix == ".png" else None, facecolor="white")


if __name__ == "__main__":
    main()
