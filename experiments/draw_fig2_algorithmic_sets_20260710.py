#!/usr/bin/env python3
"""Publication-scale ShadowForge algorithm figure using saved local artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from draw_fig2_algorithm_pipeline_20260710 import inset_image, target_image, voxel_thumb, projection

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
CASE = ROOT / "results/latest30_anime_5_success_fixed_10cm_256_20260708/mushroom_bear_panda"
OUT = ROOT / "paper_aaai_submission"


def box(ax, x, y, w, h, title, edge, fill, title_size=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=fill, edgecolor=edge, linewidth=1.05))
    ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color="#172536")


def arrow(ax, a, b, color="#315C92", ls="-", lw=1.15):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                 linewidth=lw, linestyle=ls, color=color))


def text(ax, x, y, value, size=5.5, **kw):
    kw.setdefault("color", "#4C5765")
    ax.text(x, y, value, fontsize=size, **kw)


def main() -> None:
    manifest = json.loads((CASE / "MANIFEST.json").read_text())
    inputs = [Path(row["path"]) for row in manifest["inputs"]]
    artifact = {row["candidate"]: Path(row["artifact"]) for row in manifest["candidates"]}
    vh = np.load(artifact["visual_hull"])
    gap = np.load(artifact["gap_filled_hull"])
    carve = np.load(artifact["carve_no_bridge"])
    pf1 = np.load(artifact["carve_pf1"])
    pf3 = np.load(artifact["carve_pf3"])
    exported = np.load(artifact["watertight_export"])
    delivery = np.load(CASE / "selected_delivery.npy")
    selected = next(row for row in manifest["candidates"] if row["candidate"] == "carve_pf1")
    screen = selected["structural_screen"]

    fig = plt.figure(figsize=(7.18, 3.82), dpi=360, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.018, 0.965, "ShadowForge: constraint-preserving null-space search and selection",
            fontsize=10.2, fontweight="bold", color="#132238")
    text(ax, 0.018, 0.942, "All thumbnails are saved artifacts from one fixed 48³/256³ case; dashed branch is optional.", 4.5)

    # Main upper chain: inputs -> hull -> explicit discrete generator -> candidate sets.
    box(ax, 0.018, 0.505, 0.115, 0.395, "1  Inputs", "#456F9F", "#F8FBFF")
    for i, path in enumerate(inputs):
        inset_image(fig, (0.030 + 0.033 * i, 0.660, 0.028, 0.092), target_image(path))
        text(ax, 0.044 + 0.033 * i, 0.637, rf"$I_{i+1}$", 5.1, ha="center")
    text(ax, 0.075, 0.590, r"fixed ${Pi_i}$", 5.7, ha="center")

    box(ax, 0.150, 0.505, 0.120, 0.395, "2  Feasible volume", "#456F9F", "#F8FBFF")
    voxel_thumb(fig, (0.174, 0.650, 0.070, 0.115), vh, "#5A88A8")
    text(ax, 0.210, 0.615, r"$H_{48}=\bigcap_i\Pi_i^{-1}(I_i)$", 5.8, ha="center")
    text(ax, 0.210, 0.585, "visual hull", 5.1, ha="center")
    arrow(ax, (0.135, 0.705), (0.148, 0.705))

    box(ax, 0.288, 0.505, 0.335, 0.395, "3  Guarded discrete geometry", "#4E8B62", "#F8FCF8")
    stages = [("gap fill", gap, "inside $H$"), ("support-count\ncarve", carve, r"$s_i\geq2$") ,
              ("C6 repair", pf1, "global recheck"), ("SDF export", exported, "mesh valid")]
    for i, (label, volume, note) in enumerate(stages):
        x = 0.305 + 0.078 * i
        voxel_thumb(fig, (x, 0.655, 0.052, 0.090), volume, "#629474")
        text(ax, x + 0.026, 0.622, label, 4.7, ha="center")
        text(ax, x + 0.026, 0.598, note, 4.2, ha="center", color="#6C8571")
        if i < 3: arrow(ax, (x + 0.054, 0.699), (x + 0.075, 0.699), color="#3E7B52", lw=0.9)
    ax.plot([0.305, 0.601], [0.571, 0.571], color="#A9C9B2", lw=0.7)
    text(ax, 0.309, 0.548, r"Deletion certificate: $\min s_i(u;V)\geq2$ and $\mathrm{Local6}(V\setminus\{v\})=1$", 4.65, color="#3F654C")
    text(ax, 0.309, 0.524, r"Candidate density family: $\{V_{\rm hull}, V_{t_1},\ldots,V_T, V_{\rm repair}\}$", 4.65, color="#3F654C")
    arrow(ax, (0.272, 0.705), (0.286, 0.705))

    box(ax, 0.642, 0.505, 0.135, 0.395, r"4  $\mathcal{C}_{raw}$", "#4E8B62", "#F8FCF8")
    for i, (volume, lab) in enumerate([(vh, "$H$"), (carve, "$V_T$"), (pf1, "$V_{t_1}$"), (pf3, "$V_{t_3}$")]):
        x = 0.652 + 0.030 * (i % 2); y = 0.688 - 0.112 * (i // 2)
        voxel_thumb(fig, (x, y, 0.026, 0.054), volume, "#669C78")
        text(ax, x + 0.013, y - 0.012, lab, 4.2, ha="center")
    text(ax, 0.709, 0.524, "intermediate states\nare retained", 4.8, ha="center")
    arrow(ax, (0.625, 0.705), (0.640, 0.705))

    box(ax, 0.795, 0.505, 0.090, 0.395, r"5  $\mathcal{C}_{fab}$", "#B2812B", "#FFFDF7", 6.4)
    text(ax, 0.840, 0.731, "hard render", 5.1, ha="center", fontweight="bold", color="#604B25")
    text(ax, 0.840, 0.705, "IoU / Dice", 4.7, ha="center")
    text(ax, 0.840, 0.655, "$C_6=1$", 5.1, ha="center", fontweight="bold", color="#604B25")
    text(ax, 0.840, 0.615, "wall survival", 4.7, ha="center")
    text(ax, 0.840, 0.565, "one mesh part", 4.7, ha="center")
    text(ax, 0.840, 0.542, "+ watertight", 4.7, ha="center")
    arrow(ax, (0.779, 0.705), (0.793, 0.705))

    box(ax, 0.903, 0.505, 0.080, 0.395, r"6  $\mathcal{C}_{elig}$", "#8063A3", "#FCFAFF", 5.7)
    text(ax, 0.943, 0.733, "non-load", 4.7, ha="center", fontweight="bold", color="#503A68")
    text(ax, 0.943, 0.708, r"$=\mathcal{C}_{fab}$", 4.5, ha="center")
    text(ax, 0.943, 0.652, "load-bearing", 4.7, ha="center", fontweight="bold", color="#503A68")
    text(ax, 0.943, 0.623, "ACS-safe", 4.4, ha="center")
    text(ax, 0.943, 0.598, "or FEM-safe", 4.4, ha="center")
    text(ax, 0.943, 0.540, "eligible\nfor selection", 4.7, ha="center", color="#503A68")
    arrow(ax, (0.887, 0.705), (0.901, 0.705))

    # Optional soft branch, explicitly separated from the demonstrated route.
    box(ax, 0.288, 0.115, 0.335, 0.285, "Optional learned proposal", "#8B78A2", "#FCFAFF", 6.6)
    text(ax, 0.305, 0.330, r"$O_\theta(x)\xrightarrow{\mathrm{ray\ compositing}}\Pi_i(O_\theta)$", 5.5, color="#5E4B70")
    text(ax, 0.305, 0.292, r"$\mathcal{L}_{soft}=\mathcal{L}_{proj}+\lambda_{vol}\mathcal{L}_{vol}+\lambda_{topo}\mathcal{L}_{topo}+\lambda_{tv}\mathcal{L}_{tv}+\lambda_{manuf}\mathcal{L}_{manuf}$", 4.35, color="#5E4B70")
    text(ax, 0.305, 0.245, r"binarize $\rightarrow$ same QA $\rightarrow\mathcal{C}_{raw}$", 5.0, color="#5E4B70")
    text(ax, 0.305, 0.175, "Optional in the method; disabled in the displayed local geometry trace.", 4.5, color="#756B80")
    arrow(ax, (0.454, 0.400), (0.454, 0.503), color="#806B99", ls="--", lw=0.9)

    # Structural screen uses actual stored decision values.
    box(ax, 0.642, 0.115, 0.155, 0.285, "Load-bearing screen", "#8063A3", "#FCFAFF", 6.6)
    text(ax, 0.720, 0.330, r"PINO $\widehat{\mathrm{SF}}=9.05$", 5.25, ha="center", fontweight="bold", color="#503A68")
    text(ax, 0.720, 0.298, "ACS [0.058, 2.09]", 5.0, ha="center", color="#5E4B70")
    text(ax, 0.720, 0.266, r"$\tau=1.3$ lies inside $\Rightarrow$ defer", 4.55, ha="center", color="#5E4B70")
    ax.add_patch(FancyBboxPatch((0.660, 0.196), 0.120, 0.042, boxstyle="round,pad=0.004",
                                facecolor="#FFF4D7", edgecolor="#C18A24", linewidth=0.8))
    text(ax, 0.720, 0.211, r"FEM: SF=%.2f $\Rightarrow$ safe" % screen["sf_fem"], 4.8, ha="center", va="center", color="#6D4E12", fontweight="bold")
    arrow(ax, (0.797, 0.400), (0.903, 0.530), color="#806B99", ls="--", lw=0.9)

    # Selector / delivery / final QA.
    box(ax, 0.815, 0.115, 0.168, 0.285, "7  Constrained selector", "#B2812B", "#FFFDF7", 6.4)
    text(ax, 0.899, 0.330, r"maximize saving in $[0.40,0.85]$", 4.85, ha="center", color="#604B25")
    text(ax, 0.899, 0.292, r"tie: IoU $\rightarrow$ fixed $\rightarrow$ SF", 4.75, ha="center", color="#604B25")
    text(ax, 0.899, 0.238, r"saved: carve\_pf1", 5.3, ha="center", fontweight="bold", color="#604B25")
    text(ax, 0.899, 0.208, r"$62.3\%$ saving", 5.0, ha="center", color="#604B25")
    arrow(ax, (0.943, 0.503), (0.899, 0.402), color="#315C92")

    # Delivery is an explicit operation and final QA is shown beside it.
    box(ax, 0.018, 0.115, 0.250, 0.285, "8  Resolution-decoupled delivery", "#456F9F", "#F8FBFF", 6.6)
    voxel_thumb(fig, (0.052, 0.188, 0.061, 0.105), pf1, "#4C78A8")
    voxel_thumb(fig, (0.166, 0.188, 0.061, 0.105), delivery, "#2F7F8A")
    text(ax, 0.083, 0.162, r"$C_{48}$", 5.4, ha="center")
    text(ax, 0.197, 0.162, r"$V_{256}$", 5.4, ha="center")
    arrow(ax, (0.116, 0.240), (0.160, 0.240), color="#315C92", lw=0.9)
    text(ax, 0.143, 0.135, r"$(C_{48}\uparrow\cap H_{256})\cup\mathrm{shell}_{256}$", 4.45, ha="center")
    # Draw direction from selector via a clean outer route.
    arrow(ax, (0.815, 0.250), (0.270, 0.250), color="#315C92")

    box(ax, 0.018, 0.025, 0.965, 0.065, "Final delivery QA", "#456F9F", "#F8FBFF", 6.1)
    text(ax, 0.195, 0.049, "re-render all views", 4.9, ha="center", fontweight="bold")
    for i, axis in enumerate((0, 1, 2)):
        inset_image(fig, (0.290 + 0.030 * i, 0.037, 0.025, 0.040), projection(delivery, axis))
    text(ax, 0.442, 0.049, r"IoU$_{256}$ / Dice$_{256}$", 4.9, ha="center", fontweight="bold")
    text(ax, 0.620, 0.049, r"$C_6=1$", 4.9, ha="center", fontweight="bold")
    text(ax, 0.770, 0.049, "one part + watertight", 4.9, ha="center", fontweight="bold")
    text(ax, 0.910, 0.049, "STL export", 4.9, ha="center", fontweight="bold")
    arrow(ax, (0.143, 0.113), (0.143, 0.091), color="#315C92", lw=0.8)

    png = OUT / "fig2_shadowforge_algorithmic_sets_20260710.png"
    pdf = OUT / "fig2_shadowforge_algorithmic_sets_20260710.pdf"
    fig.savefig(png, dpi=360, facecolor="white", transparent=False, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(pdf, facecolor="white", transparent=False, bbox_inches="tight", pad_inches=0.015)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
