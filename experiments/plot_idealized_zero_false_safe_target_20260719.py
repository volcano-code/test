#!/usr/bin/env python3
"""Plot an explicitly illustrative target after excluding four false-safe rows."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(
    "/home/lin/xvxinyao/ShadowForge/results/"
    "pino_v16_idealized_zero_false_safe_target_20260719"
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Source diagnostic: 1,233 safe and 105 unsafe rows. At the selected
    # sealed-label threshold, 1,164 safe and 101 unsafe rows are correct.
    # This illustrative subset excludes the four false-safe unsafe rows.
    n_safe = 1233
    n_unsafe = 101
    correct_safe = 1164
    correct_unsafe = 101
    total = n_safe + n_unsafe

    values = np.array(
        [
            (correct_safe + correct_unsafe) / total,
            correct_safe / n_safe,
            correct_unsafe / n_unsafe,
        ]
    ) * 100.0
    labels = ["Overall correct", "Safe recall", "Unsafe recall"]
    colors = ["#455A64", "#00897B", "#FFB300"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 2.75))
    bars = ax.bar(np.arange(3), values, width=0.58, color=colors, edgecolor="white")

    ax.axhline(90, color="#90A4AE", linewidth=1.0, linestyle="--")
    ax.text(2.48, 90.7, "90% target", color="#607D8B", fontsize=7.5, ha="right")
    ax.set_ylim(80, 102.5)
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(np.arange(3), labels, rotation=8, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#ECEFF1", linewidth=0.8)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.45,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#263238",
        )

    ax.set_title("Illustrative zero-false-safe target", fontsize=10, fontweight="bold", color="#263238")
    ax.text(
        0.5,
        -0.30,
        "n = 1,334; FP = 0/101. Excludes four known false-safe cases;\n"
        "illustrative target only, not sealed-test evidence.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.2,
        color="#B71C1C",
    )

    fig.subplots_adjust(left=0.20, right=0.98, top=0.88, bottom=0.31)

    fig.savefig(OUT / "idealized_zero_false_safe_target.png", dpi=450, facecolor="white")
    fig.savefig(OUT / "idealized_zero_false_safe_target.pdf", facecolor="white")


if __name__ == "__main__":
    main()
