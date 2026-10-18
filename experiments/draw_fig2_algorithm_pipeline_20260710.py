#!/usr/bin/env python3
"""Draw the ShadowForge method overview from saved local experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
CASE = ROOT / "results/latest30_anime_5_success_fixed_10cm_256_20260708/mushroom_bear_panda"
OUT = ROOT / "paper_aaai_submission"
S6 = generate_binary_structure(3, 1)


def panel(ax, x, y, w, h, title, edge="#4C78A8", fill="#F8FBFF", fs=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.016",
                         lw=1.5, edgecolor=edge, facecolor=fill, zorder=0)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.025, title, ha="center", va="top",
            fontsize=fs, weight="bold", color="#182535")
    return box


def arrow(ax, a, b, color="#315C92", lw=1.8, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12,
                                 lw=lw, color=color, linestyle=ls,
                                 connectionstyle="arc3,rad=0"))


def target_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    # Saved inputs are white-on-black; figures use black-on-white.
    if arr.mean() < 127:
        arr = 255 - arr
    return arr


def inset_image(fig, xywh, img, cmap="gray", border="#9AA5B1"):
    x, y, w, h = xywh
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(border); spine.set_linewidth(0.8)
    return ax


def voxel_thumb(fig, xywh, volume, color="#4C78A8"):
    """Render a saved occupancy volume as a compact isometric surface cloud."""
    x, y, w, h = xywh
    ax = fig.add_axes([x, y, w, h], projection="3d")
    v = volume.astype(bool)
    step = max(1, int(np.ceil(v.shape[0] / 48)))
    v = v[::step, ::step, ::step]
    surface = v & ~binary_erosion(v, S6, border_value=0)
    pts = np.argwhere(surface)
    if len(pts) > 11000:
        pts = pts[:: max(1, len(pts) // 11000)]
    ax.scatter(pts[:, 2], pts[:, 1], pts[:, 0], s=1.8, c=color, alpha=0.9, depthshade=True)
    n = v.shape[0]
    ax.set_xlim(0, n); ax.set_ylim(0, n); ax.set_zlim(0, n)
    ax.view_init(elev=20, azim=-55)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    ax.patch.set_alpha(0)
    return ax


def projection(volume: np.ndarray, axis: int) -> np.ndarray:
    return (~volume.astype(bool).any(axis=axis)).astype(np.uint8) * 255


def main() -> None:
    manifest = json.loads((CASE / "MANIFEST.json").read_text())
    inputs = [Path(row["path"]) for row in manifest["inputs"]]
    candidates = {row["candidate"]: Path(row["artifact"]) for row in manifest["candidates"]}
    hull = np.load(candidates["visual_hull"])
    gap_filled = np.load(candidates["gap_filled_hull"])
    carved = np.load(candidates["carve_no_bridge"])
    intermediate = np.load(candidates["carve_pf3"])
    repaired = np.load(candidates["watertight_export"])
    delivered = np.load(CASE / "selected_delivery.npy")
    selected = next(row for row in manifest["candidates"] if row["candidate"] == "carve_pf1")
    screen = selected["structural_screen"]

    fig = plt.figure(figsize=(16.2, 8.4), dpi=220, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.03, 0.965, "ShadowForge: fabrication-aware search over a shadow null-space",
            fontsize=18, weight="bold", color="#132238")
    ax.text(0.03, 0.938, "Local artifact trace: mushroom / bear / panda fixed case; every shown candidate is a saved geometry artifact.",
            fontsize=8.5, color="#536273")

    # Input and feasible volume.
    panel(ax, 0.025, 0.60, 0.13, 0.29, "1. Target silhouettes", edge="#456F9F")
    for i, path in enumerate(inputs):
        inset_image(fig, (0.036 + 0.040 * i, 0.665, 0.034, 0.106), target_image(path))
        ax.text(0.053 + 0.040 * i, 0.645, rf"$I_{i+1}$", ha="center", fontsize=8)
    ax.text(0.090, 0.618, r"fixed cameras $\{\Pi_i\}$", ha="center", fontsize=8)

    panel(ax, 0.18, 0.60, 0.135, 0.29, "2. Feasible set", edge="#456F9F")
    voxel_thumb(fig, (0.205, 0.650, 0.085, 0.15), hull, "#5A88A8")
    ax.text(0.248, 0.635, r"$H=\bigcap_i\Pi_i^{-1}(I_i)$", ha="center", fontsize=9)
    ax.text(0.248, 0.615, "visual hull / shadow null-space", ha="center", fontsize=7.5, color="#536273")
    arrow(ax, (0.157, 0.745), (0.177, 0.745))

    # Candidate generation, real saved intermediate volumes.
    panel(ax, 0.34, 0.50, 0.265, 0.39, "3. Discrete null-space search", edge="#4E8B62", fill="#F8FCF8")
    ax.text(0.355, 0.835, "Deterministic trajectory retained at multiple material levels", fontsize=8.5, weight="bold", color="#245839")
    ax.text(0.355, 0.812, r"$V_0=\mathrm{FillGaps}(H)$; delete $v$ only if $s_i(u;V)\geq2$ and $\mathrm{Local6}=1$", fontsize=6.8, color="#3C634B")
    for x, name, vol, note in [
        (0.350, "gap-filled\nscaffold", gap_filled, r"$V_0$"),
        (0.410, "raw carve", carved, r"$V_{t_1}$"),
        (0.470, "guarded\nintermediate", intermediate, r"$V_{t_2}$"),
        (0.530, "repaired +\nvalidated", repaired, r"$\widetilde V$"),
    ]:
        voxel_thumb(fig, (x, 0.650, 0.050, 0.105), vol, "#6B9C7B")
        ax.text(x + 0.025, 0.620, name, ha="center", fontsize=6.9)
        ax.text(x + 0.025, 0.593, note, ha="center", fontsize=6.7, color="#637565")
    for x0, x1 in ((0.401, 0.408), (0.461, 0.468), (0.521, 0.528)):
        arrow(ax, (x0, 0.705), (x1, 0.705), color="#3E7B52", lw=1.2)
    ax.plot([0.355, 0.590], [0.570, 0.570], color="#95B6A0", lw=0.8)
    ax.text(0.472, 0.542, r"$\mathcal{C}_{raw}=\{H,V_0,V_{t_1},\ldots,V_T,\widetilde V\}$",
            ha="center", fontsize=8.1, weight="bold", color="#245839")
    ax.text(0.472, 0.515, "heavier states retain margin; lighter states improve material efficiency",
            ha="center", fontsize=6.7, color="#526C59")
    arrow(ax, (0.317, 0.745), (0.337, 0.745))

    # Shared gates with explicit tests.
    panel(ax, 0.63, 0.50, 0.15, 0.39, "4. Shared hard gates", edge="#B2812B", fill="#FFFDF7")
    ax.text(0.705, 0.825, r"$C_{raw}\rightarrow C_{fab}$", ha="center", fontsize=8.5, weight="bold", color="#6D531B")
    gate_lines = [
        ("Binary render", r"$\mathrm{IoU}(\Pi_i(V),I_i)\geq\epsilon$"),
        ("Connectivity", r"$\#\mathrm{CC}_6(V)=1$"),
        ("Wall survival", r"$\mathrm{erode}(V,t)\neq\varnothing$"),
        ("Mesh validation", "one part + watertight"),
    ]
    for i, (name, formula) in enumerate(gate_lines):
        yy = 0.785 - i * 0.072
        ax.text(0.648, yy, name, fontsize=8.2, weight="bold", color="#57421F")
        ax.text(0.648, yy - 0.021, formula, fontsize=7.2, color="#68593C")
        if i < len(gate_lines) - 1: ax.plot([0.645, 0.765], [yy - 0.040, yy - 0.040], color="#DFCFA8", lw=0.8)
    arrow(ax, (0.607, 0.745), (0.627, 0.745))

    # Selector is the continuation of the main geometric path.
    panel(ax, 0.80, 0.50, 0.175, 0.39, "5. Fixed selector", edge="#B2812B", fill="#FFFDF7")
    ax.text(0.887, 0.783, r"$V^*=\arg\min_{V\in\mathcal{C}_{elig}} |V|$", ha="center", fontsize=10, weight="bold", color="#4B3920")
    ax.text(0.887, 0.735, r"$C_{fab}\rightarrow C_{elig}$", ha="center", fontsize=8.5, weight="bold", color="#6D531B")
    ax.text(0.887, 0.700, "minimum-volume candidate satisfying\nall applicable hard gates", ha="center", fontsize=8.0, color="#5F513A")
    ax.plot([0.817, 0.958], [0.657, 0.657], color="#DFCFA8", lw=0.8)
    ax.text(0.887, 0.625, r"saved choice: carve\_pf1", ha="center", fontsize=8, weight="bold", color="#6B5935")
    ax.text(0.887, 0.599, r"$62.3\%$ material reduction", ha="center", fontsize=7.5, color="#6B5935")
    ax.text(0.887, 0.550, "load-bearing mode additionally\nrequires structural eligibility", ha="center", fontsize=7.0, color="#6B5935")
    arrow(ax, (0.782, 0.745), (0.797, 0.745))

    # Structural branch uses the stored decision record for this same case.
    panel(ax, 0.055, 0.115, 0.275, 0.265, "Load-bearing side branch: PINO + ACS + FEM", edge="#8063A3", fill="#FCFAFF", fs=9.2)
    ax.text(0.192, 0.305, r"PINO: $\widehat{\mathrm{SF}}=9.05$", ha="center", fontsize=9, weight="bold", color="#46315E")
    ax.text(0.192, 0.274, r"ACS interval: $[0.058,2.09]$", ha="center", fontsize=8, color="#5C4674")
    ax.text(0.192, 0.245, r"$\mathrm{lower}\leq\tau\leq\mathrm{upper}$  $\Rightarrow$ defer", ha="center", fontsize=7.5, color="#5C4674")
    ax.add_patch(FancyBboxPatch((0.088, 0.181), 0.208, 0.045, boxstyle="round,pad=0.006",
                                facecolor="#FFF4D7", edgecolor="#C18A24", lw=1.0))
    ax.text(0.192, 0.203, r"FEM fallback: SF = %.2f  $\Rightarrow$ eligible" % screen["sf_fem"], ha="center", va="center", fontsize=7.5, weight="bold", color="#6D4E12")
    ax.text(0.192, 0.145, "automatic decisions only when\ncalibrated bounds support them", ha="center", fontsize=6.9, color="#645872")
    arrow(ax, (0.705, 0.500), (0.330, 0.362), color="#8063A3", lw=1.1, ls="--")

    panel(ax, 0.37, 0.115, 0.275, 0.265, "6. Resolution-decoupled delivery", edge="#456F9F")
    voxel_thumb(fig, (0.390, 0.178, 0.090, 0.135), repaired, "#4C78A8")
    voxel_thumb(fig, (0.535, 0.178, 0.090, 0.135), delivered, "#2F7F8A")
    ax.text(0.435, 0.155, r"$C_{48}$ structural core", ha="center", fontsize=8)
    ax.text(0.580, 0.155, r"$V_{256}=(C_{48}\uparrow\cap H_{256})\cup\mathrm{shell}_{256}$", ha="center", fontsize=7.2)
    arrow(ax, (0.485, 0.245), (0.528, 0.245), color="#315C92")
    # Route the serpentine main-flow arrow through the whitespace above the
    # lower panels, keeping it clear of the delivered-output panel.
    arrow(ax, (0.887, 0.500), (0.887, 0.405), color="#315C92")
    arrow(ax, (0.887, 0.405), (0.652, 0.405), color="#315C92")
    arrow(ax, (0.652, 0.405), (0.638, 0.300), color="#315C92")

    panel(ax, 0.685, 0.115, 0.285, 0.265, "7. Delivered model and re-rendered shadows", edge="#456F9F")
    voxel_thumb(fig, (0.700, 0.165, 0.090, 0.150), delivered, "#2F7F8A")
    for i, axis in enumerate((0, 1, 2)):
        inset_image(fig, (0.810 + 0.050 * i, 0.205, 0.043, 0.085), projection(delivered, axis))
    ax.text(0.748, 0.148, "saved 256$^3$ delivery", ha="center", fontsize=7.2)
    ax.text(0.885, 0.175, r"$\Pi_1(V_{256}),\Pi_2(V_{256}),\Pi_3(V_{256})$", ha="center", fontsize=7.2)
    arrow(ax, (0.647, 0.250), (0.682, 0.250))

    ax.text(0.03, 0.055, "Solid arrows show the deployed geometry path. The triage panel shows the stored FEM-fallback record for this local case.",
            fontsize=7.2, color="#536273")
    png = OUT / "fig2_shadowforge_algorithm_pipeline_20260710.png"
    pdf = OUT / "fig2_shadowforge_algorithm_pipeline_20260710.pdf"
    fig.savefig(png, dpi=260, bbox_inches="tight", pad_inches=0.03,
                facecolor="white", transparent=False)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03,
                facecolor="white", transparent=False)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
