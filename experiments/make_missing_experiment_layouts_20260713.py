#!/usr/bin/env python3
"""Render draft-only experiment protocol figures without invented results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_aaai_submission"


def box(ax, x, y, w, h, title, body, fill, edge="#243447"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
                           linewidth=1.4, edgecolor=edge, facecolor=fill)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * .67, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color="#18212b")
    ax.text(x + w / 2, y + h * .31, body, ha="center", va="center", fontsize=8.2,
            color="#364152", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.5, color="#425466"))


def candidate_pool_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.15), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.02, .95, "(a) Candidate pool and fixed selection", fontsize=13, fontweight="bold")
    specs = [
        (.03, .42, .16, .31, "Fixed silhouettes", "two wall masks\n+and view assignment", "#fff6db"),
        (.26, .42, .16, .31, "Soft proposal", "differentiable\noccupancy field", "#e8f1fb"),
        (.49, .42, .16, .31, "Threshold pool", "multiple seeds\nfrom one proposal", "#edf7ed"),
        (.72, .42, .16, .31, "Repair + carve", "ray restoration\n6-connected bridge", "#fbecee"),
    ]
    for x, y, w, h, t, b, c in specs:
        box(ax, x, y, w, h, t, b, c)
    for x in (.19, .42, .65):
        arrow(ax, x, .575, x + .065, .575)
    box(ax, .72, .08, .16, .20, "Selected", "lightest eligible\nwithin fidelity band", "#d9f5e5", edge="#16804b")
    arrow(ax, .80, .42, .80, .285)
    ax.text(.04, .17, "Shared gates: delivered IoU, wall survival, single body, watertight mesh", fontsize=9,
            color="#364152")

    ax = axes[1]
    ax.set_title("(b) Audit reported over the candidate pool", fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_color("#9aa5b1")
    ax.add_patch(Rectangle((.06, .58), .88, .30, facecolor="#eaf6ed", edgecolor="none"))
    ax.add_patch(Rectangle((.06, .12), .88, .30, facecolor="#fff0f0", edgecolor="none"))
    ax.axhline(.50, color="#4a5568", lw=1.2, ls="--")
    ax.text(.08, .83, "eligible region", fontsize=9, color="#237144")
    ax.text(.08, .17, "FEM fallback / rejected region", fontsize=9, color="#a33a3a")
    for x, y, c, m in [(.20,.70,"#5b6470","o"),(.38,.77,"#2f9e5d","o"),(.54,.64,"#2f9e5d","o"),
                        (.72,.73,"#2f9e5d","o"),(.30,.31,"#d34a4a","x"),(.63,.27,"#d34a4a","x")]:
        ax.scatter(x, y, s=54, c=c, marker=m, zorder=3)
    ax.scatter(.79, .68, s=210, marker="*", c="#16804b", edgecolors="#142b20", zorder=4)
    ax.text(.72, .92, "saving", fontsize=9); ax.text(.01, .47, "structural evidence", fontsize=9, rotation=90, va="center")
    fig.tight_layout(pad=.6)
    fig.savefig(PAPER / "fig4_candidate_pool_audit_layout.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def resolution_figure():
    fig, ax = plt.subplots(figsize=(10.8, 2.95))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.02, .91, "Resolution-decoupled delivery protocol", fontsize=13.5, fontweight="bold")
    items = [
        (.03, "48$^3$ structural core", "PINO / ACS / FEM\nstructural proxy", "#e8f1fb"),
        (.29, "Nearest-neighbor lift", "clip to $H_{256}$", "#f3f5f7"),
        (.53, "256$^3$ silhouette shell", "recover fine projected\nboundaries", "#fff6db"),
        (.78, "Delivered mesh", "IoU, QA, slicing\nand physical evaluation", "#d9f5e5"),
    ]
    for x, title, body, color in items:
        box(ax, x, .28, .18, .38, title, body, color)
    for x in (.21, .47, .72): arrow(ax, x, .47, x + .07, .47)
    ax.text(.03, .10, "Structural labels apply to the delivered-object proxy; final delivery is rechecked for projection and mesh validity.",
            fontsize=9.2, color="#364152")
    fig.tight_layout(pad=.5)
    fig.savefig(PAPER / "fig6_resolution_delivery_layout.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    PAPER.mkdir(parents=True, exist_ok=True)
    candidate_pool_figure()
    resolution_figure()


if __name__ == "__main__":
    main()
