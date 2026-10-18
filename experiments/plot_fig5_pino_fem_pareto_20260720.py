#!/usr/bin/env python3
"""Plot the sealed task-level ranking experiment under a varying FEM budget."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SUMMARY = ROOT / "results/raw_pino_fem_reduction_sealed_v16_20260719/summary.json"
OUT_DIR = ROOT / "results/pino_ranked_fem_taskdisjoint_v16_20260720"
PAPER_DIR = ROOT / "paper_aaai_submission/figures"


def rows_for(rows: list[dict], method: str) -> list[dict]:
    return sorted(
        (row for row in rows if row["method"] == method),
        key=lambda row: int(row["top_k"]),
    )


def coordinates(rows: list[dict]) -> tuple[list[float], list[float]]:
    budget = [100.0 * row["fem_calls"] / row["full_fem_calls"] for row in rows]
    agreement = [100.0 * row["exact_selection_agreement"] for row in rows]
    return budget, agreement


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    rows = summary["task_topk"]
    pino = rows_for(rows, "raw_pino_safe_first_then_material")
    material = rows_for(rows, "material_only")
    score_only = rows_for(rows, "raw_pino_sf_descending")
    pino_x, pino_y = coordinates(pino)
    material_x, material_y = coordinates(material)
    score_x, score_y = coordinates(score_only)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelcolor": "#263238",
            "xtick.color": "#455A64",
            "ytick.color": "#455A64",
        }
    )
    fig, ax = plt.subplots(figsize=(3.42, 2.72))

    ax.plot(
        score_x,
        score_y,
        color="#7E57C2",
        marker="^",
        markersize=4.2,
        linewidth=1.35,
        label="PINO score only",
        zorder=2,
    )
    ax.plot(
        material_x,
        material_y,
        color="#90A4AE",
        marker="o",
        markersize=4.5,
        linewidth=1.5,
        label="Material first",
        zorder=3,
    )
    ax.plot(
        pino_x,
        pino_y,
        color="#00897B",
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        label="PINO safe-first + material",
        zorder=4,
    )

    selected = next(i for i, row in enumerate(pino) if int(row["top_k"]) == 3)
    ax.scatter(
        [pino_x[selected]],
        [pino_y[selected]],
        marker="*",
        s=105,
        facecolor="#FFB300",
        edgecolor="#7A5200",
        linewidth=0.7,
        zorder=6,
        label="Frozen Top-3",
    )
    ax.annotate(
        "Frozen Top-3\n100% agreement\n9.2% FEM budget",
        xy=(pino_x[selected], pino_y[selected]),
        xytext=(21.5, 87.0),
        fontsize=6.8,
        color="#263238",
        arrowprops={"arrowstyle": "->", "color": "#607D8B", "lw": 0.8},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#CFD8DC", "lw": 0.7},
    )

    ax.scatter(
        [100.0],
        [100.0],
        marker="s",
        s=34,
        facecolor="#455A64",
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="Exhaustive FEM",
    )

    ax.set_xlabel("Candidates evaluated by voxel FEM (%)", labelpad=4)
    ax.set_ylabel("Lightest-safe selection agreement (%)", labelpad=4)
    ax.set_xlim(0, 103)
    ax.set_ylim(-3, 104)
    ax.set_xticks([0, 10, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(color="#E3E9EC", linewidth=0.65)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#B0BEC5")
        ax.spines[spine].set_linewidth(0.8)

    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#CFD8DC",
        framealpha=1.0,
        fontsize=6.4,
        borderpad=0.45,
        handlelength=1.8,
    )
    fig.tight_layout(pad=0.45)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    outputs = (
        OUT_DIR / "fig5_pino_fem_pareto_20260720.png",
        OUT_DIR / "fig5_pino_fem_pareto_20260720.pdf",
        PAPER_DIR / "fig5_pino_fem_pareto_20260720.pdf",
    )
    for output in outputs:
        fig.savefig(
            output,
            dpi=600 if output.suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
