"""Render a local-asset, vector-first overview figure for ShadowForge."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
from PIL import Image


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
PAPER = ROOT / "paper_aaai_submission"
OUT_PNG = PAPER / "fig2_shadowforge_unified_pipeline_okabe_ito_20260713.png"
OUT_PDF = PAPER / "fig2_shadowforge_unified_pipeline_okabe_ito_20260713.pdf"

LEFT = ROOT / "results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/dark_knight_left__magic_scholar_right/inputs/left_wall_x_view1_target.png"
RIGHT = ROOT / "results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/dark_knight_left__magic_scholar_right/inputs/right_wall_y_view2_target.png"
SOFT = PAPER / "generated_pipeline_assets/soft_proposal_local.png"
DELIVERY = PAPER / "generated_pipeline_assets/selected_delivery_local.png"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
CHARCOAL = "#333333"
GREY = "#4D4D4D"
LIGHT_GREY = "#CCCCCC"
FAINT = "#F7F7F7"


def load(path: Path, max_size: int = 220):
    image = Image.open(path).convert("RGBA")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def panel(ax, xy, width, height, title, edge=LIGHT_GREY, fill="#FFFFFF", dashed=False):
    style = "round,pad=0.010,rounding_size=0.008"
    box = FancyBboxPatch(xy, width, height, boxstyle=style, transform=ax.transAxes,
                         linewidth=1.25, edgecolor=edge, facecolor=fill,
                         linestyle=(0, (5, 3)) if dashed else "solid")
    ax.add_patch(box)
    ax.text(xy[0] + 0.014, xy[1] + height - 0.026, title, transform=ax.transAxes,
            fontsize=10.5, fontweight="bold", color=CHARCOAL, va="top")
    return box


def box(ax, xy, width, height, title, subtitle="", edge=LIGHT_GREY, fill="#FFFFFF", fs=8.0):
    rect = FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.006,rounding_size=0.004",
                          transform=ax.transAxes, linewidth=1.05, edgecolor=edge, facecolor=fill)
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.60, title, transform=ax.transAxes,
            ha="center", va="center", fontsize=fs, fontweight="bold", color=CHARCOAL, wrap=True)
    if subtitle:
        ax.text(xy[0] + width / 2, xy[1] + height * 0.28, subtitle, transform=ax.transAxes,
                ha="center", va="center", fontsize=fs - 1.15, color="#666666", wrap=True)
    return rect


def arrow(ax, start, end, color=GREY, label="", style="-|>", dashed=False, lw=1.6, label_offset=0.012):
    patch = FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle=style,
                            mutation_scale=12, linewidth=lw, color=color,
                            linestyle=(0, (3, 2)) if dashed else "solid",
                            connectionstyle="angle3,angleA=0,angleB=90")
    ax.add_patch(patch)
    if label:
        x = (start[0] + end[0]) / 2
        y = (start[1] + end[1]) / 2 + label_offset
        ax.text(x, y, label, transform=ax.transAxes, fontsize=6.8, color=color,
                ha="center", va="bottom")


def image(ax, img, center, zoom=0.20, frame=True, label=""):
    artist = AnnotationBbox(OffsetImage(img, zoom=zoom), center, xycoords=ax.transAxes,
                            frameon=frame, pad=0.018,
                            bboxprops={"edgecolor": LIGHT_GREY, "linewidth": 0.8, "facecolor": "white"})
    ax.add_artist(artist)
    if label:
        ax.text(center[0], center[1] - 0.085, label, transform=ax.transAxes,
                ha="center", va="top", fontsize=7, color=CHARCOAL)


def mini_volume(ax, center, color, label):
    x, y = center
    for dx, dy, s in [(-.018, .012, .030), (.010, .018, .034), (.028, -.004, .026),
                      (-.014, -.018, .026), (.006, -.023, .032), (.035, .024, .018)]:
        ax.add_patch(Circle((x + dx, y + dy), s / 2, transform=ax.transAxes,
                            facecolor=color, edgecolor="white", linewidth=.5, alpha=.82))
    ax.text(x, y - .057, label, transform=ax.transAxes, ha="center", va="top",
            fontsize=7, color=CHARCOAL)


def mini_voxels(ax, center, label, color="#8797A3"):
    x, y = center
    for r in range(3):
        for c in range(4):
            if not (r == 0 and c == 3):
                ax.add_patch(Rectangle((x + (c - 1.5) * .012, y + (r - 1) * .012), .010, .010,
                                       transform=ax.transAxes, facecolor=color, edgecolor="white", linewidth=.25))
    ax.text(x, y - .044, label, transform=ax.transAxes, ha="center", va="top", fontsize=6.8, color=CHARCOAL)


def main():
    target_1, target_2 = load(LEFT), load(RIGHT)
    soft, delivery = load(SOFT, 260), load(DELIVERY, 260)

    fig = plt.figure(figsize=(17.5, 8.1), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(.025, .965, "ShadowForge: unified differentiable-to-fabrication pipeline",
            transform=ax.transAxes, fontsize=19, fontweight="bold", color=CHARCOAL, va="top")
    legend = [(BLUE, "differentiable forward"), (ORANGE, "gradient update"),
              (GREY, "deterministic fabrication"), (GREEN, "selected output"), (RED, "FEM fallback")]
    lx = .026
    for col, text in legend:
        ax.plot([lx, lx + .018], [.928, .928], transform=ax.transAxes, color=col, linewidth=2.2)
        ax.text(lx + .023, .928, text, transform=ax.transAxes, fontsize=7.5, color="#666666", va="center")
        lx += .135

    # Main flow.
    panel(ax, (.025, .330), .135, .525, "Fixed-view specification", edge=BLUE, fill="#F8FBFE", dashed=True)
    image(ax, target_1, (.072, .635), zoom=.17, label="T1")
    image(ax, target_2, (.123, .635), zoom=.17, label="T2")
    ax.text(.0925, .503, "target silhouettes", transform=ax.transAxes, ha="center", fontsize=8.5, fontweight="bold", color=CHARCOAL)
    ax.text(.0925, .470, "orthographic camera directions C", transform=ax.transAxes, ha="center", fontsize=7, color="#666666")

    panel(ax, (.182, .330), .180, .525, "Differentiable occupancy proposal", edge=BLUE, fill="#F8FBFE")
    image(ax, soft, (.272, .646), zoom=.19, frame=False, label="rho = sigmoid(z)")
    box(ax, (.206, .438), .132, .082, "fixed-view ray renderer", "P_i(rho)", edge=BLUE, fs=8.3)
    ax.text(.272, .395, "continuous field; fixed camera protocol", transform=ax.transAxes, ha="center", fontsize=6.7, color="#666666")

    panel(ax, (.390, .330), .205, .525, "Projection objective", edge=ORANGE, fill="#FFFDF7")
    image(ax, target_1, (.446, .656), zoom=.13, label="rendered I1")
    image(ax, target_2, (.535, .656), zoom=.13, label="rendered I2")
    box(ax, (.421, .445), .145, .096, "L_soft", "L2 + L1 + Dice + SSIM + FP\nlocal / global integrity", edge=ORANGE, fs=8.6)
    ax.text(.493, .396, "back-propagate to z", transform=ax.transAxes, ha="center", fontsize=6.8, color=ORANGE)

    panel(ax, (.623, .330), .215, .525, "Proposal-to-geometry handoff", edge=GREY, fill=FAINT)
    box(ax, (.641, .648), .050, .100, "threshold\nensemble", "tau in T", edge="#8B72BE", fs=7.0)
    box(ax, (.706, .648), .055, .100, "ray restore\n+ bridge", "one body", edge=GREY, fs=7.0)
    box(ax, (.776, .648), .045, .100, "guarded\ncarve", "states", edge=GREY, fs=7.0)
    mini_voxels(ax, (.667, .519), "proposal")
    mini_voxels(ax, (.733, .519), "repaired", "#5C7F8A")
    mini_voxels(ax, (.799, .519), "carved", "#607D8B")
    ax.text(.730, .415, "inside feasible envelope; preserve target-ray support", transform=ax.transAxes,
            ha="center", fontsize=6.5, color="#666666")

    panel(ax, (.860, .330), .115, .525, "Selection and delivery", edge=GREEN, fill="#F7FCFA")
    box(ax, (.875, .675), .085, .078, "shared fabrication gates", "fidelity | wall | mesh", edge=GREEN, fs=7.7)
    image(ax, delivery, (.917, .545), zoom=.10, frame=False, label="selected core-shell")
    image(ax, target_1, (.893, .435), zoom=.075, label="P1")
    image(ax, target_2, (.942, .435), zoom=.075, label="P2")
    ax.text(.917, .355, "48 cubed core -> 256 cubed delivery", transform=ax.transAxes,
            ha="center", fontsize=6.4, color="#666666")

    arrow(ax, (.160, .640), (.182, .640), BLUE)
    arrow(ax, (.362, .640), (.390, .640), BLUE)
    arrow(ax, (.595, .640), (.623, .690), GREY, label="final proposal field")
    arrow(ax, (.691, .698), (.706, .698), GREY)
    arrow(ax, (.761, .698), (.776, .698), GREY)
    arrow(ax, (.821, .698), (.875, .714), GREY, label="candidate pool")
    arrow(ax, (.338, .475), (.390, .475), BLUE)
    arrow(ax, (.493, .445), (.302, .405), ORANGE, label="gradient update", lw=1.7, label_offset=-.024)

    # Structural branch.
    panel(ax, (.590, .080), .385, .165, "Optional load-bearing structural triage", edge="#90A4AE", fill="#FBFCFC", dashed=True)
    box(ax, (.606, .122), .104, .074, "physics-residual PINO", "occupancy + SDF + BC/load", edge=BLUE, fs=7.1)
    box(ax, (.730, .122), .116, .074, "asymmetric conformal screen", "supported safe / unsafe only", edge="#7E57C2", fs=6.9)
    box(ax, (.866, .122), .094, .074, "voxel FEM", "deferred candidates", edge=RED, fs=7.2)
    arrow(ax, (.710, .159), (.730, .159), BLUE)
    arrow(ax, (.846, .159), (.866, .159), RED, label="defer", dashed=True, label_offset=.010)
    arrow(ax, (.799, .565), (.658, .196), GREY, label="candidate pool", dashed=True, lw=1.2, label_offset=.010)
    arrow(ax, (.846, .192), (.875, .675), GREEN, label="supported", lw=1.4, label_offset=.012)
    arrow(ax, (.960, .192), (.875, .675), RED, label="FEM-verified", lw=1.25, label_offset=-.018)
    ax.text(.784, .096, "PINO screens candidates; FEM remains the final authority for unresolved decisions.",
            transform=ax.transAxes, ha="center", fontsize=7.2, color="#666666")

    ax.text(.498, .276, "one coherent candidate-generation and fabrication-selection workflow", transform=ax.transAxes,
            ha="center", fontsize=8.0, color=CHARCOAL, fontstyle="italic")

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=.04)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=.04)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
