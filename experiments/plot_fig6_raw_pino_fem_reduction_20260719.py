#!/usr/bin/env python3
"""Plot sealed task-level Raw-PINO/FEM reduction results for the paper."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SUMMARY = ROOT / "results/pino_ranked_fem_taskdisjoint_v16_20260720/summary.json"
OUT = ROOT / "results/pino_ranked_fem_taskdisjoint_v16_20260720"
PAPER_FIGURE = ROOT / "paper_aaai_submission/figures/fig6_raw_pino_fem_reduction_20260719.pdf"


def lookup(rows: list[dict], method: str, top_k: int) -> dict:
    return next(row for row in rows if row["method"] == method and row["top_k"] == top_k)


def add_labels(ax, bars: list, values: list[float], suffix: str, offset: float) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#263238",
        )


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    audit = summary["sealed_test"]
    k = int(summary["policy"]["top_k"])
    source = json.loads((ROOT / "results/raw_pino_fem_reduction_sealed_v16_20260719/summary.json").read_text())
    rows = source["task_topk"]
    material = lookup(rows, "material_only", k)

    labels = ["Full FEM", f"Material only\n+ Top-{k}", f"Raw PINO\n+ Top-{k}"]
    agreement = [100.0, 100.0 * material["exact_selection_agreement"],
                 100.0 * audit["exact_selection_agreement"]]
    call_rate = [100.0, 100.0 * material["fem_calls"] / material["full_fem_calls"],
                 100.0 * audit["fem_calls"] / audit["full_fem_calls"]]
    colors = ["#455A64", "#B0BEC5", "#00897B"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.35))
    x = np.arange(len(labels))

    for ax, values, title, ylabel, ylim, suffix, offset in (
        (axes[0], agreement, "Agreement with full-FEM selection", "Tasks matched (%)", (90, 101.8), "%", 0.35),
        (axes[1], call_rate, "Voxel-FEM budget", "Candidates evaluated by FEM (%)", (0, 108), "%", 3.0),
    ):
        bars = ax.bar(x, values, color=colors, width=0.7, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=9.2, fontweight="bold", pad=5)
        ax.set_ylabel(ylabel, fontsize=7.8)
        ax.set_xticks(x, labels, fontsize=6.8)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#CFD8DC", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", labelsize=7, length=0)
        add_labels(ax, list(bars), values, suffix, offset)

    fig.text(
        0.5,
        0.01,
        f"Top-{k} frozen on 37 calibration tasks; evaluated once on 41 task-disjoint sealed tasks.",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#455A64",
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.85, bottom=0.25, wspace=0.38)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (OUT / "fig6_raw_pino_fem_reduction_20260719.png",
                 OUT / "fig6_raw_pino_fem_reduction_20260719.pdf",
                 PAPER_FIGURE):
        fig.savefig(path, dpi=450 if path.suffix == ".png" else None, facecolor="white")


if __name__ == "__main__":
    main()
