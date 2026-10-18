#!/usr/bin/env python3
"""Render publication-ready summaries for the new trajectory and soft ablations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_summary(directory: Path) -> list[dict]:
    path = directory / "summary.json"
    return json.loads(path.read_text()) if path.is_file() else []


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_pino(rows: list[dict], out: Path) -> None:
    fields = ["policy", "tasks", "mean_iou", "mean_saving", "fem_safe_selection_rate",
              "exact_exhaustive_fem_agreement", "mean_fem_budget"]
    write_tsv(out / "table3_pino_trajectory.tsv", rows, fields)
    labels = {
        "material_only": "Material only",
        "exhaustive_fem": "Exhaustive FEM",
        "pino_top3": "PINO Top-3",
        "pino_rollback_m0p00": "Trajectory, m=0.00",
        "pino_rollback_m0p10": "Trajectory, m=0.10",
        "pino_rollback_m0p20": "Trajectory, m=0.20",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.18, 0.82, len(rows)))
    for color, row in zip(colors, rows):
        name = labels.get(row["policy"], row["policy"])
        budget = 100 * row["mean_fem_budget"]
        agreement = 100 * row["exact_exhaustive_fem_agreement"]
        axes[0].scatter(budget, agreement, s=54, color=color, edgecolor="white", linewidth=0.6)
        axes[0].annotate(name, (budget, agreement), xytext=(4, 3), textcoords="offset points", fontsize=7)
    axes[0].set_xlabel("Voxel-FEM budget (%)")
    axes[0].set_ylabel("Agreement with exhaustive FEM (%)")
    axes[0].grid(alpha=0.25)

    x = np.arange(len(rows))
    axes[1].bar(x - 0.18, [100 * r["fem_safe_selection_rate"] for r in rows], 0.36,
                label="FEM-safe selection", color="#00897B")
    axes[1].bar(x + 0.18, [100 * r["mean_saving"] for r in rows], 0.36,
                label="Material saving", color="#FFB300")
    axes[1].set_xticks(x, [labels.get(r["policy"], r["policy"]) for r in rows], rotation=28, ha="right")
    axes[1].set_ylabel("Rate (%)")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7)
    fig.savefig(out / "fig_pino_trajectory_stopping.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_pino_trajectory_stopping.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_soft(rows: list[dict], out: Path) -> None:
    fields = ["variant", "n", "mean_iou_mean", "mean_iou_std", "worst_view_iou_mean",
              "worst_view_iou_std", "mean_dice_mean", "mean_dice_std", "fp_leak_mean",
              "fp_leak_std", "boundary_f1_mean", "boundary_f1_std"]
    write_tsv(out / "table2a_soft_loss_ablation.tsv", rows, fields)
    labels = {"full": "Full", "w_o_Ldice": r"w/o $L_{dice}$",
              "w_o_Lout": r"w/o $L_{out}$", "w_o_Lssim": r"w/o $L_{ssim}$"}
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
    axes[0].bar(x - 0.27, [100 * r["mean_iou_mean"] for r in rows], 0.27, label="Mean IoU", color="#00897B")
    axes[0].bar(x, [100 * r["worst_view_iou_mean"] for r in rows], 0.27, label="Worst-view IoU", color="#55C5C9")
    axes[0].bar(x + 0.27, [100 * r["mean_dice_mean"] for r in rows], 0.27, label="Dice", color="#4F83CC")
    axes[0].set_ylabel("Projection score (%)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].bar(x - 0.18, [100 * r["boundary_f1_mean"] for r in rows], 0.36,
                label="Boundary F1", color="#00897B")
    axes[1].bar(x + 0.18, [100 * r["fp_leak_mean"] for r in rows], 0.36,
                label="FP leakage", color="#D95F59")
    axes[1].set_ylabel("Rate (%)")
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.set_xticks(x, [labels.get(r["variant"], r["variant"]) for r in rows], rotation=18, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(labelsize=7)
    fig.savefig(out / "fig_soft_loss_ablation.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_soft_loss_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pino-dir", type=Path, required=True)
    parser.add_argument("--soft-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pino = load_summary(args.pino_dir)
    soft = load_summary(args.soft_dir)
    if pino:
        render_pino(pino, args.out_dir)
    if soft:
        render_soft(soft, args.out_dir)
    (args.out_dir / "STATUS.json").write_text(json.dumps({
        "pino_rows": len(pino), "soft_rows": len(soft),
        "pino_source": str(args.pino_dir), "soft_source": str(args.soft_dir),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
