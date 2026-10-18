#!/usr/bin/env python3
"""Create paper-ready geometry-ablation table and Pareto plot from saved data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = ROOT / "results/geometry_alternatives_ablation_20260710/geometry_ablation_results.json"
OUT = ROOT / "results/geometry_alternatives_ablation_20260710"


def main() -> None:
    rows = json.loads(SRC.read_text())["aggregate"]
    tex = [
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Variant & IoU & Dice & Saving & Single & Water & Wall \\", r"\midrule",
    ]
    for row in rows:
        label = row["variant"].replace("w/o", r"w/o")
        values = (
            f"{row['mean_iou']:.3f}", f"{row['mean_dice']:.3f}",
            f"{100*row['mean_saving']:.1f}\\%", f"{100*row['single_body_rate']:.1f}\\%",
            f"{100*row['watertight_rate']:.1f}\\%", f"{100*row['wall_pass_rate']:.1f}\\%",
        )
        if row["variant"] == "Full geometry pipeline":
            label = r"\textbf{Full geometry pipeline}"
            values = tuple(r"\textbf{" + value + "}" for value in values)
        tex.append(label + " & " + " & ".join(values) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table_geometry_ablation.tex").write_text("\n".join(tex) + "\n")

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=220)
    for row in rows:
        x, y = 100 * row["mean_saving"], row["mean_iou"]
        valid = row["single_body_rate"] * row["watertight_rate"] * row["wall_pass_rate"]
        full = row["variant"] == "Full geometry pipeline"
        ax.scatter(x, y, s=170 if full else 85, marker="*" if full else "o",
                   color="#167C5A" if full else plt.cm.RdYlGn(valid), edgecolor="#202A35", zorder=3)
        ax.annotate(row["variant"], (x, y), xytext=(5, 5), textcoords="offset points", fontsize=7.5)
    ax.axvspan(40, 60, color="#DCEFE7", alpha=0.55, label="deployment saving band")
    ax.set_xlabel("Material saving relative to visual hull (%)")
    ax.set_ylabel("Mean silhouette IoU")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "geometry_pareto.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "geometry_pareto.pdf", bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
