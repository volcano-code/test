#!/usr/bin/env python3
"""Plot PINO screening capability and task-level FEM budget reduction."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SUMMARY = ROOT / "results/raw_pino_fem_reduction_sealed_v16_20260719/summary.json"
OUT_DIR = ROOT / "results/pino_capability_budget_20260721"
PAPER_DIR = ROOT / "paper_aaai_submission/figures"


def rows_for(rows: list[dict], method: str) -> list[dict]:
    return sorted(
        (row for row in rows if row["method"] == method),
        key=lambda row: int(row["top_k"]),
    )


def budget_curve(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([100.0 * row["fem_calls"] / row["full_fem_calls"] for row in rows])
    y = np.asarray([100.0 * row["exact_selection_agreement"] for row in rows])
    return x, y


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    topk = summary["task_topk"]
    pino = rows_for(topk, "raw_pino_safe_first_then_material")
    material = rows_for(topk, "material_only")
    pino_x, pino_y = budget_curve(pino)
    material_x, material_y = budget_curve(material)

    # Frozen balanced-diagnostic results reported in Table 4(a).
    methods = ["Geom. MLP", "3D CNN", "Field U-FNO", "Dual w/o phys.", "Full PINO"]
    accuracy = np.asarray([50.0, 88.1, 50.0, 84.8, 94.8])
    unsafe_recall = np.asarray([0.0, 76.2, 0.0, 69.5, 91.4])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelcolor": "#263238",
            "xtick.color": "#455A64",
            "ytick.color": "#455A64",
        }
    )
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(3.42, 4.15), gridspec_kw={"hspace": 0.5})

    y = np.arange(len(methods))
    ax0.hlines(y, unsafe_recall, accuracy, color="#CFD8DC", linewidth=2.0, zorder=1)
    ax0.scatter(accuracy, y, s=31, color="#00897B", marker="o", label="Accuracy", zorder=3)
    ax0.scatter(
        unsafe_recall,
        y,
        s=34,
        color="#FFB300",
        edgecolor="#8A6100",
        linewidth=0.45,
        marker="^",
        label="Unsafe recall",
        zorder=4,
    )
    ax0.set_yticks(y, methods)
    ax0.set_ylim(len(methods) - 0.55, -0.85)
    ax0.set_xlim(-2, 103)
    ax0.set_xticks([0, 25, 50, 75, 100])
    ax0.set_xlabel("Balanced diagnostic performance (%)", labelpad=3)
    ax0.grid(axis="x", color="#E3E9EC", linewidth=0.65)
    ax0.set_axisbelow(True)
    ax0.text(0.015, 0.965, "(a)", transform=ax0.transAxes, fontsize=8.4, fontweight="bold", va="top")
    ax0.legend(
        loc="upper right",
        ncol=2,
        fontsize=6.7,
        frameon=True,
        facecolor="white",
        edgecolor="#CFD8DC",
        borderpad=0.35,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    ax1.plot(
        material_x,
        material_y,
        color="#90A4AE",
        marker="o",
        markersize=4.0,
        linewidth=1.35,
        label="Material-only",
        zorder=3,
    )
    ax1.plot(
        pino_x,
        pino_y,
        color="#00897B",
        marker="o",
        markersize=4.0,
        linewidth=1.7,
        label="PINO-guided",
        zorder=4,
    )
    frozen = next(i for i, row in enumerate(pino) if int(row["top_k"]) == 3)
    ax1.scatter(
        [pino_x[frozen]],
        [pino_y[frozen]],
        marker="*",
        s=88,
        facecolor="#FFB300",
        edgecolor="#7A5200",
        linewidth=0.6,
        zorder=6,
        label="Frozen Top-3",
    )
    ax1.annotate(
        "Top-3\n100% agreement, 9.2% FEM",
        xy=(pino_x[frozen], pino_y[frozen]),
        xytext=(25, 98.35),
        fontsize=6.5,
        arrowprops={"arrowstyle": "->", "color": "#607D8B", "lw": 0.75},
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#CFD8DC", "lw": 0.65},
    )
    ax1.scatter(
        [100.0],
        [100.0],
        marker="s",
        s=29,
        facecolor="#455A64",
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
        label="Exhaustive FEM",
    )
    ax1.set_xlabel("Candidates evaluated by voxel FEM (%)", labelpad=3)
    ax1.set_ylabel("Selection agreement (%)", labelpad=3)
    ax1.set_xlim(0, 103)
    ax1.set_ylim(94.2, 100.7)
    ax1.set_xticks([0, 10, 20, 40, 60, 80, 100])
    ax1.set_yticks([95, 96, 97, 98, 99, 100])
    ax1.grid(color="#E3E9EC", linewidth=0.65)
    ax1.set_axisbelow(True)
    ax1.text(0.015, 0.965, "(b)", transform=ax1.transAxes, fontsize=8.4, fontweight="bold", va="top")
    ax1.legend(
        loc="lower right",
        fontsize=6.4,
        frameon=True,
        facecolor="white",
        edgecolor="#CFD8DC",
        borderpad=0.35,
        handlelength=1.65,
    )

    for ax in (ax0, ax1):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#B0BEC5")
            ax.spines[spine].set_linewidth(0.8)

    fig.subplots_adjust(left=0.28, right=0.985, top=0.985, bottom=0.095)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    outputs = (
        OUT_DIR / "fig5_pino_capability_budget_20260721.png",
        OUT_DIR / "fig5_pino_capability_budget_20260721.pdf",
        PAPER_DIR / "fig5_pino_capability_budget_20260721.pdf",
    )
    for output in outputs:
        fig.savefig(output, dpi=600 if output.suffix == ".png" else None, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
