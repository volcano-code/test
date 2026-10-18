#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pyvista as pv


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
EVIDENCE = ROOT / "paper_aaai_submission/p_craft_pipeline_evidence_20260721"
OUT_DIR = ROOT / "paper_aaai_submission/figures"
FORMAL_ROWS = ROOT / "results/formal_material_trajectory_fixed3_20260722/trajectory_rows.json"
FAILED_STL = (
    ROOT
    / "results/formal_material_trajectory_fixed3_20260722/connected_10/exhaustive_fem"
    / "selected_display_fixed.stl"
)

STAGES = [
    (
        "Repaired scaffold",
        EVIDENCE / "06_repaired_scaffold_3d.png",
        EVIDENCE / "metrics_soft_guided_full_hull_scaffold_core48_delivery256.json",
    ),
    (
        "Carve phase 1",
        EVIDENCE / "07_carve_heavy_3d.png",
        EVIDENCE / "metrics_soft_guided_carve_phase_1_removed_2544_core48_delivery256.json",
    ),
    (
        "Carve phase 2",
        EVIDENCE / "08_carve_middle_3d.png",
        EVIDENCE / "metrics_soft_guided_carve_phase_2_removed_1045_core48_delivery256.json",
    ),
    (
        "Carve phase 3",
        EVIDENCE / "09_carve_light_3d.png",
        EVIDENCE / "metrics_soft_guided_carve_phase_3_removed_37_core48_delivery256.json",
    ),
]


def cropped_rgba(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    rgba = np.asarray(image)
    alpha = rgba[..., 3]
    rgb = rgba[..., :3]
    nonwhite = (alpha > 8) & (np.min(rgb, axis=2) < 247)
    ys, xs = np.where(nonwhite)
    if not len(xs):
        return image
    pad = 12
    box = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(image.width, int(xs.max()) + pad + 1),
        min(image.height, int(ys.max()) + pad + 1),
    )
    return image.crop(box)


def render_failed_candidate(path: Path) -> Image.Image:
    destination = OUT_DIR / "material_trajectory_rejected_candidate.png"
    if not destination.exists():
        mesh = pv.read(path)
        plotter = pv.Plotter(off_screen=True, window_size=(820, 720))
        plotter.set_background("white")
        plotter.add_mesh(
            mesh,
            color="#AFC5CE",
            smooth_shading=True,
            specular=0.18,
            ambient=0.24,
        )
        plotter.camera_position = "iso"
        plotter.enable_parallel_projection()
        plotter.reset_camera()
        plotter.camera.zoom(1.18)
        plotter.show(screenshot=str(destination), auto_close=True)
    return cropped_rgba(destination)


def main() -> None:
    rows = [json.loads(metrics.read_text()) for _, _, metrics in STAGES]
    formal_rows = json.loads(FORMAL_ROWS.read_text())
    failed = next(
        row
        for row in formal_rows
        if row["case_id"] == "connected_10" and "t0.55" in row["candidate"]
    )
    labels = [name for name, _, _ in STAGES]
    saving = np.array([100.0 * row["saving"] for row in rows])
    mean_iou = np.array([100.0 * row["mean_iou"] for row in rows])
    min_iou = np.array([100.0 * row["min_iou"] for row in rows])
    voxels = np.array([row["occupied_voxels"] for row in rows])
    removed = np.r_[0, voxels[:-1] - voxels[1:]]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(11.2, 5.65), facecolor="white")
    grid = fig.add_gridspec(
        2, 5, height_ratios=[1.62, 1.0],
        left=0.06, right=0.97, bottom=0.12, top=0.95,
        hspace=0.10, wspace=0.13,
    )

    teal = "#00897B"
    amber = "#E69F00"
    blue = "#4F83CC"
    dark = "#263238"
    pale = "#F4F7F8"
    selected = "#00796B"
    rejected = "#C62828"
    selected_index = 1

    for index, ((label, image_path, _), row) in enumerate(zip(STAGES, rows)):
        ax = fig.add_subplot(grid[0, index])
        ax.set_facecolor("#EDF8F6" if index == selected_index else pale)
        ax.imshow(cropped_rgba(image_path))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.7 if index == selected_index else 0.8)
            spine.set_edgecolor(selected if index == selected_index else "#CFD8DC")
        ax.set_title(label, color=dark, pad=5, fontweight="semibold")
        ax.text(
            0.5, -0.04,
            f"{row['occupied_voxels'] / 1000:.0f}k voxels  |  save {100 * row['saving']:.1f}%",
            transform=ax.transAxes, ha="center", va="top", color=dark, fontsize=9.1,
        )
        if index:
            fig.text(
                ax.get_position().x0 - 0.018,
                ax.get_position().y0 + 0.52 * ax.get_position().height,
                f"−{removed[index] / 1000:.0f}k\nvoxels",
                ha="center", va="center", color=amber, fontsize=8.6,
                fontweight="semibold",
            )
        if index == selected_index:
            ax.text(
                0.96, 0.96, "selected delivery",
                transform=ax.transAxes, ha="right", va="top",
                color="white", fontsize=8.5, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.25", "fc": selected, "ec": "none"},
            )

    failed_ax = fig.add_subplot(grid[0, 4])
    failed_ax.set_facecolor("#FFF2F2")
    failed_ax.imshow(render_failed_candidate(FAILED_STL))
    failed_ax.set_xticks([])
    failed_ax.set_yticks([])
    for spine in failed_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor(rejected)
    failed_ax.set_title("Matched unsafe candidate", color="#7F1D1D", pad=5, fontweight="semibold")
    failed_ax.text(
        0.5,
        0.03,
        (
            f"save {100 * failed['saving']:.1f}%\n"
            f"$\\widehat{{SF}}_{{PINO}}$={failed['sf_pino']:.2f}; "
            f"$SF_{{FEM}}$={failed['sf_fem']:.2f} < 1.30"
        ),
        transform=failed_ax.transAxes,
        ha="center",
        va="bottom",
        color="#7F1D1D",
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#F0B6B6", "alpha": 0.94},
    )
    failed_ax.text(
        0.95,
        0.95,
        "rejected",
        transform=failed_ax.transAxes,
        ha="right",
        va="top",
        color="white",
        fontsize=8.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": rejected, "ec": "none"},
    )

    ax = fig.add_subplot(grid[1, :])
    x = np.arange(len(rows))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4EAED", linewidth=0.8)
    ax.fill_between(x, saving, color=amber, alpha=0.13)
    line_saving, = ax.plot(
        x, saving, color=amber, marker="o", linewidth=2.2,
        markersize=6, label="Material saving",
    )
    ax.set_ylabel("Material saving vs. shadow hull (%)", color="#8A5A00")
    ax.set_ylim(0, 43)
    ax.tick_params(axis="y", colors="#8A5A00")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", pad=7)

    ax_iou = ax.twinx()
    line_mean, = ax_iou.plot(
        x, mean_iou, color=teal, marker="s", linewidth=2.0,
        markersize=5.5, label="Mean IoU",
    )
    line_min, = ax_iou.plot(
        x, min_iou, color=blue, marker="D", linewidth=1.7,
        markersize=5, label="Worst-view IoU",
    )
    ax_iou.set_ylim(98.72, 99.08)
    ax_iou.tick_params(axis="y", colors=dark)

    for i, value in enumerate(saving):
        ax.annotate(
            f"{value:.1f}%",
            (i, value), xytext=(0, 8), textcoords="offset points",
            ha="center", color="#7A5000", fontsize=8.8, fontweight="semibold",
        )

    ax.axvline(selected_index, color=selected, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.annotate(
        "$SF_{FEM}=5.35$",
        (selected_index, saving[selected_index]),
        xytext=(30, -31),
        textcoords="offset points",
        ha="left",
        color=selected,
        fontsize=9.0,
        fontweight="semibold",
        arrowprops={"arrowstyle": "-", "color": selected, "linewidth": 1.0},
    )
    ax.legend(
        [line_saving, line_mean, line_min],
        ["Material saving", "Mean IoU", "Worst-view IoU"],
        loc="lower right", frameon=False, ncol=3,
        bbox_to_anchor=(1.0, 1.01), borderaxespad=0,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("top", "left"):
        ax_iou.spines[side].set_visible(False)

    fig.text(
        0.50, 0.025,
        "Phase 1 is the lightest fidelity-band candidate passing fabrication gates and FEM verification. "
        "Red is a matched unsafe counterexample.",
        ha="center", va="center", color="#455A64", fontsize=9.5,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "dark_knight_material_trajectory_visual_20260723.png"
    pdf = OUT_DIR / "dark_knight_material_trajectory_visual_20260723.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
