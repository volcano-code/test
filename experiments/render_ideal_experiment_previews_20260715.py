"""Render data-honest layout previews for the revised ShadowForge experiments.

The previews use local shadow/geometry assets where they exist.  Any evidence
that still requires a completed benchmark, sealed test, or print photograph is
explicitly marked TBD instead of being replaced with invented numbers.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
ASSETS = ROOT / "paper_aaai_submission/pipeline_reference_assets_20260714"
OUT = ROOT / "results/ideal_experiment_previews_20260715"

TEAL = "#00897B"
AMBER = "#FFB300"
SLATE = "#455A64"
PALE = "#ECEFF1"
INK = "#263238"
LIGHT = "#CFD8DC"


def setup(figsize):
    fig = plt.figure(figsize=figsize, facecolor="white")
    return fig


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def image(path):
    return Image.open(path).convert("RGB")


def title(fig, text, note):
    fig.text(0.02, 0.965, text, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.925, note, fontsize=8.5, color=SLATE, va="top")


def panel(ax, label, color=LIGHT):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(1.1)
    ax.set_facecolor("white")
    ax.set_title(label, fontsize=8, color=INK, pad=4)


def draw_tbd(ax, label, color=LIGHT):
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.06, 0.10), 0.88, 0.80, boxstyle="round,pad=0.02,rounding_size=0.035",
                                transform=ax.transAxes, facecolor="#FAFBFB", edgecolor=color, linewidth=1.1))
    ax.text(0.5, 0.58, "TBD", ha="center", va="center", fontsize=14, fontweight="bold", color=SLATE,
            transform=ax.transAxes)
    ax.text(0.5, 0.37, label, ha="center", va="center", fontsize=7.2, color=SLATE,
            transform=ax.transAxes, wrap=True)


def fig3_controlled_comparison():
    fig = setup((13.8, 7.1))
    title(fig, "Controlled Comparison with Shadow-Art Baselines", "Layout preview only: replace every TBD cell with a same-protocol rendering and measured score.")
    rows = ["Target shadows", "Mitra-style", "Sadekar mesh", "Sadekar voxel", "ShadowForge"]
    cols = ["Task A: fixed-wall pair", "Task B: fixed-wall pair"]
    left, right, bottom, top = 0.13, 0.98, 0.07, 0.88
    width, height = (right-left)/2, (top-bottom)/5
    targets = [image(ASSETS / "01_target_x.png"), image(ASSETS / "02_target_y.png")]
    for r, row in enumerate(rows):
        fig.text(0.015, top-(r+0.5)*height, row, ha="left", va="center", fontsize=10,
                 fontweight="bold" if r in (0, 4) else "normal", color=INK)
        for c, col in enumerate(cols):
            ax = fig.add_axes([left+c*width+0.01, top-(r+1)*height+0.01, width-0.02, height-0.02])
            if r == 0:
                ax.imshow(targets[c])
                panel(ax, col, TEAL)
            else:
                draw_tbd(ax, f"{col}\nrender + IoU / Dice")
                if r == 4:
                    for spine in ax.spines.values():
                        spine.set_color(TEAL)
    fig.text(0.13, 0.015, "Required final evidence: identical targets, cameras, binary renderer, resolution, mesh extraction, and QA for all rows.", fontsize=8.2, color=SLATE)
    save(fig, "fig3_controlled_comparison_preview")


def fig4_conversion_comparison():
    fig = setup((14.2, 6.0))
    title(fig, "Conversion of a Fixed Differentiable Occupancy Proposal", "Illustrative target layout: the same soft field must be converted and evaluated by every column under one renderer and QA protocol.")

    input_ax = fig.add_axes([0.02, 0.33, 0.18, 0.48])
    input_ax.imshow(image(ASSETS / "05_soft_occupancy.png"))
    panel(input_ax, "Fixed soft occupancy", TEAL)
    fig.text(0.11, 0.23, "same density field", ha="center", fontsize=8, color=SLATE)
    fig.text(0.215, 0.57, "→", fontsize=25, color=SLATE, ha="center", va="center")

    variants = [
        ("Direct marching cubes", ASSETS / "06_scaffold_3d.png", LIGHT,
         "surface extraction only"),
        ("SDF / morphology-only", ASSETS / "08_carved_3d.png", "#90A4AE",
         "local smoothing and cleanup"),
        ("ShadowForge conversion", ASSETS / "10_final_model.png", AMBER,
         "ray restoration + guarded carving + bridge repair"),
    ]
    for i, (name, path, color, detail) in enumerate(variants):
        x = 0.25 + i * 0.245
        ax = fig.add_axes([x, 0.42, 0.205, 0.39])
        ax.imshow(image(path))
        panel(ax, name, color)
        fig.text(x + 0.102, 0.355, detail, ha="center", va="center", fontsize=7.2, color=SLATE, wrap=True)

        cmp_ax = fig.add_axes([x + 0.03, 0.095, 0.145, 0.20])
        cmp_ax.imshow(image(ASSETS / "11_final_comparison.png"))
        panel(cmp_ax, "aligned projections", color)
        fig.text(x + 0.102, 0.045, "IoU / Dice / single body / watertight: TBD", ha="center", fontsize=6.8, color=INK)

    save(fig, "fig4_conversion_comparison_preview")


def fig5_risk_coverage():
    fig = setup((12.8, 5.3))
    title(fig, "Task-Disjoint Structural Triage", "Final curves must be produced once from frozen calibration and a sealed task-level test split.")
    ax = fig.add_axes([0.08, 0.18, 0.42, 0.66])
    ax.set_xlim(0, 100); ax.set_ylim(0, 8)
    ax.set_xlabel("automatic decision coverage (%)", color=INK)
    ax.set_ylabel("false-safe risk upper bound (%)", color=INK)
    ax.axhspan(0, 1, color="#E0F2F1", alpha=0.9)
    ax.axhline(1, color=TEAL, linestyle="--", linewidth=1)
    ax.text(4, 0.25, "target risk region", fontsize=8, color=TEAL)
    ax.text(50, 4.4, "sealed-test curve\nTBD", ha="center", va="center", fontsize=15, color=SLATE)
    ax.grid(color=PALE, linewidth=0.8)
    ax2 = fig.add_axes([0.58, 0.18, 0.34, 0.66])
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 100)
    ax2.set_xlabel("automatic decision coverage (%)", color=INK)
    ax2.set_ylabel("FEM-call rate (%)", color=INK)
    ax2.text(50, 50, "FEM-call trade-off\nTBD", ha="center", va="center", fontsize=15, color=SLATE)
    ax2.grid(color=PALE, linewidth=0.8)
    fig.text(0.08, 0.06, "Report raw PINO threshold, symmetric conformal, ACS, and ACS + FEM fallback; annotate false-safe denominator.", fontsize=8.2, color=SLATE)
    save(fig, "fig5_risk_coverage_preview")


def fig6_physical_realization():
    fig = setup((13.6, 5.1))
    title(fig, "Physical Realization", "Final figure needs three independently printed cases; this preview deliberately does not substitute renders for physical photographs.")
    labels = ["Printed object", "Observed two-wall shadows", "Simulated shadows"]
    for row, label in enumerate(labels):
        fig.text(0.018, 0.77-row*0.29, label, fontsize=9.5, color=INK, va="center",
                 fontweight="bold" if row == 0 else "normal")
        for col in range(3):
            ax = fig.add_axes([0.14+col*0.28, 0.66-row*0.29, 0.22, 0.22])
            draw_tbd(ax, f"Case {col+1}\nmeasured evidence")
    fig.text(0.14, 0.05, "For each case: print scale, material, orientation, layer height, support policy, and corresponding simulation must be recorded.", fontsize=8.2, color=SLATE)
    save(fig, "fig6_physical_realization_preview")


def table_png(name, title_text, columns, rows, note):
    fig = setup((12.8, max(2.6, 0.42*len(rows)+1.5)))
    title(fig, title_text, note)
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.78])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8.2); table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(LIGHT); cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor(PALE); cell.set_text_props(weight="bold", color=INK)
        elif r % 2 == 0:
            cell.set_facecolor("#FAFBFB")
    save(fig, name)


def tables():
    table_png("table1_controlled_benchmark_preview", "Table 1. Controlled Comparison with Shadow-Art Baselines",
              ["Method", "IoU ↑", "Dice ↑", "Saving ↑", "Single", "Watertight"],
              [["Mitra-style"] + ["TBD"]*5,
               ["Sadekar voxel"] + ["TBD"]*5,
               ["Sadekar mesh"] + ["TBD"]*5,
               ["ShadowForge"] + ["TBD"]*5],
              "Same tasks, camera protocol, renderer, mesh extraction, and QA. No cross-paper reported scores.")
    table_png("table2_two_stage_ablation_preview", "Table 2. Two-Stage Ablation",
              ["Stage", "Variant", "IoU", "Dice", "Saving", "Single", "Watertight"],
              [["Soft", "Full / w/o each added L", "TBD", "TBD", "—", "—", "—"],
               ["Delivery", "Full delivery", "TBD", "TBD", "TBD", "TBD", "TBD"],
               ["Delivery", "w/o ray restoration", "TBD", "TBD", "TBD", "TBD", "TBD"],
               ["Delivery", "w/o bridge repair", "TBD", "TBD", "TBD", "TBD", "TBD"],
               ["Delivery", "w/o guarded carving", "TBD", "TBD", "TBD", "TBD", "TBD"],
               ["Delivery", "w/o delivery shell", "TBD", "TBD", "TBD", "TBD", "TBD"]],
              "Soft rows evaluate raw voxels only. Delivery rows evaluate final exported geometry.")
    table_png("table3_material_delivery_preview", "Table 3. Material Selection and Resolution-Decoupled Delivery",
              ["Panel", "Policy / setting", "IoU@256", "Dice@256", "Saving", "Single", "Watertight"],
              [["Selection", "Visual hull", "TBD", "TBD", "TBD", "TBD", "TBD"],
               ["Selection", "Terminal export", "TBD", "TBD", "TBD", "TBD", "TBD"],
               ["Selection", "Lightest fabrication-valid", "TBD", "TBD", "TBD", "TBD", "TBD"],
               ["Selection", "Verified selector", "TBD", "TBD", "TBD", "TBD", "TBD"],
               ["Delivery", "48 core", "TBD", "TBD", "TBD", "TBD", "TBD"],
               ["Delivery", "48 core + 256 delivery", "TBD", "TBD", "TBD", "TBD", "TBD"]],
              "Selection and resolution are separate panels over their respective common task sets.")
    table_png("table4_structural_triage_preview", "Table 4. Structural Surrogate and ACS–FEM Triage",
              ["Panel", "Method / policy", "SF / stress error", "Auto-safe", "False-safe", "Defer", "FEM calls"],
              [["Surrogate", "Geometry MLP", "TBD", "—", "—", "—", "—"],
               ["Surrogate", "3D CNN / U-Net", "TBD", "—", "—", "—", "—"],
               ["Surrogate", "FNO", "TBD", "—", "—", "—", "—"],
               ["Surrogate", "PINO", "TBD", "—", "—", "—", "—"],
               ["Triage", "Raw PINO", "—", "TBD", "TBD", "TBD", "TBD"],
               ["Triage", "ACS + FEM fallback", "—", "TBD", "TBD", "TBD", "TBD"]],
              "All results use a frozen calibration protocol and task-disjoint sealed test. False-safe must include its unsafe denominator.")


def illustrative_tables():
    """Non-evidentiary target layouts. These values must never enter a submission."""
    tag = "ILLUSTRATIVE TARGETS ONLY -- NOT MEASURED RESULTS"
    table_png("table1_illustrative_ideal_preview", "Table 1. Controlled Comparison with Shadow-Art Baselines",
              ["Method", "IoU ↑", "Dice ↑", "Saving ↑", "Single", "Watertight"],
              [["Mitra-style", "0.884", "0.938", "5.2%", "58.3%", "50.0%"],
               ["Sadekar voxel", "0.912", "0.954", "8.1%", "41.7%", "33.3%"],
               ["Sadekar mesh", "0.927", "0.961", "9.4%", "50.0%", "41.7%"],
               ["ShadowForge", "0.987", "0.993", "45.9%", "100.0%", "100.0%"]],
              tag + ". Same-protocol benchmark; describe n=12 in text, not in the table.")
    table_png("table2_illustrative_ideal_preview", "Table 2. Two-Stage Ablation",
              ["Stage", "Variant", "IoU", "Dice", "Saving", "Single", "Watertight"],
              [["Soft", "Full", "0.976", "0.988", "—", "—", "—"],
               ["Soft", "w/o Ldice", "0.958", "0.979", "—", "—", "—"],
               ["Soft", "w/o Lfp", "0.966", "0.983", "—", "—", "—"],
               ["Soft", "w/o Lconn", "0.968", "0.984", "—", "—", "—"],
               ["Delivery", "Full", "0.989", "0.995", "45.9%", "100.0%", "100.0%"],
               ["Delivery", "w/o ray restoration", "0.982", "0.991", "43.5%", "91.7%", "100.0%"],
               ["Delivery", "w/o bridge repair", "0.988", "0.994", "47.1%", "66.7%", "100.0%"],
               ["Delivery", "w/o guarded carving", "0.989", "0.995", "23.4%", "100.0%", "100.0%"],
               ["Delivery", "w/o delivery shell", "0.961", "0.980", "47.8%", "100.0%", "75.0%"]],
              tag + ". Soft and delivery rows are separate evidence, not a common aggregate.")
    table_png("table3_illustrative_ideal_preview", "Table 3. Material Selection and Resolution-Decoupled Delivery",
              ["Panel", "Policy / setting", "IoU@256", "Dice@256", "Saving", "Single", "Watertight"],
              [["Selection", "Visual hull", "0.990", "0.995", "0.0%", "100.0%", "100.0%"],
               ["Selection", "Terminal export", "0.981", "0.990", "25.4%", "100.0%", "100.0%"],
               ["Selection", "Lightest fabrication-valid", "0.977", "0.988", "51.8%", "100.0%", "100.0%"],
               ["Selection", "Verified selector", "0.988", "0.994", "45.9%", "100.0%", "100.0%"],
               ["Delivery", "48 core", "0.938", "0.967", "46.1%", "100.0%", "100.0%"],
               ["Delivery", "48 core + 256 delivery", "0.989", "0.995", "45.9%", "100.0%", "100.0%"]],
              tag + ". Selection and resolution are two panels with different controls.")
    table_png("table4_illustrative_ideal_preview", "Table 4. Structural Surrogate and ACS--FEM Triage",
              ["Panel", "Method / policy", "SF / stress error", "Auto-safe", "False-safe", "Defer", "FEM calls"],
              [["Surrogate", "Geometry MLP", "0.31", "—", "—", "—", "—"],
               ["Surrogate", "3D CNN / U-Net", "0.23", "—", "—", "—", "—"],
               ["Surrogate", "FNO", "0.18", "—", "—", "—", "—"],
               ["Surrogate", "PINO", "0.12", "—", "—", "—", "—"],
               ["Triage", "Raw PINO", "—", "92.4%", "0.8%", "6.8%", "6.8%"],
               ["Triage", "ACS + FEM fallback", "—", "91.8%", "0.0%", "8.2%", "8.2%"]],
              tag + ". Sealed task-level split and frozen calibration are mandatory for final values.")


if __name__ == "__main__":
    fig3_controlled_comparison()
    fig4_conversion_comparison()
    fig5_risk_coverage()
    fig6_physical_realization()
    tables()
    illustrative_tables()
    print(OUT)
