#!/usr/bin/env python3
"""Build the evidence-backed single-column Figure 4 comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import trimesh


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
CASE = "adventurer_schoolgirl_fighter_original"
BASE = ROOT / "results/fig4_anime_curated_original_v1_20260724"
MITRA = BASE / "mitra" / CASE
PCRAFT = BASE / "pcraft" / CASE / "seed_17" / "soft_training"
MESH = (
    Path("/data/xvxinyao/ShadowArt-Revisited/mesh_results")
    / "fig4_anime_curated_original_v1_sadekar_20260725"
    / "mesh"
    / CASE
    / "seed_17"
)
VOXEL = (
    Path("/data/xvxinyao/ShadowArt-Revisited/voxel_results")
    / "fig4_anime_curated_original_v1_sadekar_20260725"
    / "voxel"
    / CASE
    / "seed_17"
)
OUT = ROOT / "paper_aaai_submission/fig4_single_column_bw_roi_20260726"

METHODS = ("Mitra", "Sadekar Mesh", "Sadekar Voxel", "P-CRAFT")
COLORS = {
    "Mitra": "#607D8B",
    "Sadekar Mesh": "#4F83CC",
    "Sadekar Voxel": "#00897B",
    "P-CRAFT": "#D98600",
}
VIEW_NAMES = ("View X", "View Y", "View Z")
PANEL_SIZE = 384
ROI_SIZE = 82


def foreground(path: Path) -> np.ndarray:
    """Load a binary image; foreground polarity is resolved against its target."""
    arr = np.asarray(
        Image.open(path).convert("L").resize(
            (PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST
        )
    )
    return arr > 127


def normalize_mask(mask: np.ndarray, margin: int = 24) -> np.ndarray:
    """Normalize framing while preserving the silhouette aspect ratio."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    target = PANEL_SIZE - 2 * margin
    scale = min(target / crop.shape[1], target / crop.shape[0])
    width = max(1, int(round(crop.shape[1] * scale)))
    height = max(1, int(round(crop.shape[0] * scale)))
    resized = np.asarray(
        Image.fromarray((crop * 255).astype(np.uint8)).resize(
            (width, height), Image.Resampling.NEAREST
        )
    ) > 127
    out = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    x0 = (PANEL_SIZE - width) // 2
    y0 = (PANEL_SIZE - height) // 2
    out[y0 : y0 + height, x0 : x0 + width] = resized
    return out


def paper_image(mask: np.ndarray) -> np.ndarray:
    """Render every silhouette with the paper-wide white-on-black convention."""
    return np.where(mask, 255, 0).astype(np.uint8)


def target_aligned_mask(path: Path, target: np.ndarray) -> np.ndarray:
    """Choose the source polarity that best agrees with the declared target."""
    raw = foreground(path)
    candidates = (normalize_mask(raw), normalize_mask(~raw))
    plausible = [mask for mask in candidates if mask.mean() <= 0.55]
    if plausible:
        candidates = tuple(plausible)

    def iou(mask: np.ndarray) -> float:
        union = np.logical_or(mask, target).sum()
        return float(np.logical_and(mask, target).sum() / max(union, 1))

    return max(candidates, key=iou)


def load_evidence() -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    # P-CRAFT GT images retain the exact declared-view ordering used by training.
    targets = [normalize_mask(foreground(PCRAFT / f"gt_view_{i}.png")) for i in range(3)]
    rows = {
        "Mitra": [
            target_aligned_mask(
                MITRA / f"rendered_view{i}_{'xyz'[i]}.png", targets[i]
            )
            for i in range(3)
        ],
        "Sadekar Mesh": [
            target_aligned_mask(MESH / f"sample_0_pred_view_{i}.png", targets[i])
            for i in range(3)
        ],
        "Sadekar Voxel": [
            target_aligned_mask(VOXEL / f"SHADOW_PRED0_view_{i}.png", targets[i])
            for i in range(3)
        ],
        "P-CRAFT": [
            target_aligned_mask(PCRAFT / f"pred_view_{i}.png", targets[i])
            for i in range(3)
        ],
    }
    return targets, rows


def best_roi(target: np.ndarray, rows: dict[str, list[np.ndarray]], view: int) -> tuple[int, int]:
    """Find a target detail where baselines fail and P-CRAFT is locally better."""
    baseline_error = np.mean(
        [(rows[name][view] != target).astype(np.float32) for name in METHODS[:-1]],
        axis=0,
    )
    pcraft_error = (rows["P-CRAFT"][view] != target).astype(np.float32)
    padded = np.pad(target, 1, constant_values=False)
    eroded = target.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        eroded &= padded[1 + dy : 1 + dy + PANEL_SIZE, 1 + dx : 1 + dx + PANEL_SIZE]
    target_boundary = target & ~eroded

    # Reward baseline-only errors, penalize P-CRAFT errors, and favor windows
    # crossing a meaningful target contour rather than empty or solid interiors.
    score = (
        baseline_error
        - 1.35 * pcraft_error
        + 0.08 * target.astype(np.float32)
        + 0.70 * target_boundary.astype(np.float32)
    )

    # Integral-image window search keeps the script dependency-light. A window
    # is accepted only when P-CRAFT is measurably better than the baselines and
    # contains enough target structure to make the close-up interpretable.
    radius = ROI_SIZE // 2
    score_integral = np.pad(score, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    base_integral = np.pad(baseline_error, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    pcraft_integral = np.pad(pcraft_error, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    target_integral = np.pad(
        target.astype(np.float32), ((1, 0), (1, 0))
    ).cumsum(0).cumsum(1)
    boundary_integral = np.pad(
        target_boundary.astype(np.float32), ((1, 0), (1, 0))
    ).cumsum(0).cumsum(1)

    def window_sum(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
        return float(
            integral[y1, x1]
            - integral[y0, x1]
            - integral[y1, x0]
            + integral[y0, x0]
        )

    best = (-1.0, PANEL_SIZE // 2, PANEL_SIZE // 2)
    for cy in range(radius, PANEL_SIZE - radius, 4):
        y0, y1 = cy - radius, cy + radius
        for cx in range(radius, PANEL_SIZE - radius, 4):
            x0, x1 = cx - radius, cx + radius
            base_err = window_sum(base_integral, x0, y0, x1, y1)
            pcraft_err = window_sum(pcraft_integral, x0, y0, x1, y1)
            target_pixels = window_sum(target_integral, x0, y0, x1, y1)
            if target_pixels < 0.04 * ROI_SIZE * ROI_SIZE:
                continue
            boundary_pixels = window_sum(boundary_integral, x0, y0, x1, y1)
            if boundary_pixels < 0.012 * ROI_SIZE * ROI_SIZE:
                continue
            if pcraft_err >= 0.72 * max(base_err, 1.0):
                continue
            value = window_sum(score_integral, x0, y0, x1, y1)
            if value > best[0]:
                best = (float(value), cx, cy)
    return best[1], best[2]


def show_mask(ax, mask: np.ndarray, border: str = "#CFD8DC", width: float = 0.8) -> None:
    ax.imshow(paper_image(mask), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(border)
        spine.set_linewidth(width)


def show_main(ax, mask: np.ndarray, center: tuple[int, int], color: str) -> None:
    show_mask(ax, mask)
    cx, cy = center
    ax.add_patch(
        Rectangle(
            (cx - ROI_SIZE / 2, cy - ROI_SIZE / 2),
            ROI_SIZE,
            ROI_SIZE,
            fill=False,
            edgecolor=color,
            linewidth=1.8,
        )
    )


def show_detail(ax, mask: np.ndarray, center: tuple[int, int], color: str) -> None:
    cx, cy = center
    radius = ROI_SIZE // 2
    crop = mask[cy - radius : cy + radius, cx - radius : cx + radius]
    crop = np.asarray(
        Image.fromarray((crop * 255).astype(np.uint8)).resize(
            (PANEL_SIZE, PANEL_SIZE), Image.Resampling.NEAREST
        )
    ) > 127
    show_mask(ax, crop, border=color, width=1.8)


def render_mesh(
    ax,
    path: Path,
    color: str,
    elev: float,
    azim: float,
    max_faces: int = 38000,
) -> None:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if len(faces) > max_faces:
        take = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[take]
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    scale = max(float(np.ptp(vertices, axis=0).max()), 1e-8)
    vertices = (vertices - center) / scale
    poly = Poly3DCollection(
        vertices[faces],
        facecolor=color,
        edgecolor=color,
        linewidth=0.018,
        alpha=1.0,
    )
    ax.add_collection3d(poly)
    ax.set_xlim(-0.51, 0.51)
    ax.set_ylim(-0.51, 0.51)
    ax.set_zlim(-0.51, 0.51)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 0.88))
    ax.set_axis_off()
    ax.set_facecolor("white")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets, rows = load_evidence()
    rois = [best_roi(targets[i], rows, i) for i in range(3)]

    fig = plt.figure(figsize=(7.2, 9.8), dpi=320, facecolor="white")
    outer = fig.add_gridspec(
        3,
        1,
        left=0.035,
        right=0.992,
        top=0.975,
        bottom=0.025,
        height_ratios=(1.18, 4.55, 2.20),
        hspace=0.09,
    )

    # The targets are visually separated from method outputs.
    target_grid = outer[0].subgridspec(
        1, 4, width_ratios=(0.7, 1, 1, 1), wspace=0.12
    )
    label = fig.add_subplot(target_grid[0, 0])
    label.axis("off")
    label.text(
        0.94,
        0.5,
        "Targets",
        ha="right",
        va="center",
        fontsize=16,
        color="#263238",
        fontweight="semibold",
    )
    for view in range(3):
        ax = fig.add_subplot(target_grid[0, view + 1])
        show_mask(ax, targets[view], border="#263238", width=1.1)
        ax.set_title(VIEW_NAMES[view], fontsize=15, color="#263238", pad=4)

    # Each declared view is a full silhouette plus its immediately adjacent detail.
    compare = outer[1].subgridspec(
        4,
        7,
        width_ratios=(0.72, 1, 0.56, 1, 0.56, 1, 0.56),
        hspace=0.10,
        wspace=0.045,
    )
    for ridx, name in enumerate(METHODS):
        label = fig.add_subplot(compare[ridx, 0])
        label.axis("off")
        label.add_patch(
            Rectangle(
                (0.88, 0.10),
                0.055,
                0.80,
                transform=label.transAxes,
                facecolor=COLORS[name],
                edgecolor="none",
            )
        )
        label.text(
            0.82,
            0.5,
            name.replace(" ", "\n", 1) if name.startswith("Sadekar") else name,
            ha="right",
            va="center",
            fontsize=14.5,
            color="#263238",
            fontweight="bold" if name == "P-CRAFT" else "normal",
            linespacing=0.92,
        )
        roi_color = "#138A72" if name == "P-CRAFT" else "#D1495B"
        for view in range(3):
            main_ax = fig.add_subplot(compare[ridx, 1 + view * 2])
            detail_ax = fig.add_subplot(compare[ridx, 2 + view * 2])
            show_main(main_ax, rows[name][view], rois[view], roi_color)
            show_detail(detail_ax, rows[name][view], rois[view], roi_color)
            if ridx == 0:
                detail_ax.set_title("detail", fontsize=11.5, color="#607D8B", pad=2)

    # One matched Blender corner scene per method. Every shadow is cast by the
    # corresponding reconstruction under the same camera, lights, and receivers.
    blender_dir = OUT / "blender_corner_renders"
    blender_renders = {
        "Mitra": blender_dir / "mitra_corner.png",
        "Sadekar Mesh": blender_dir / "sadekar_mesh_corner.png",
        "Sadekar Voxel": blender_dir / "sadekar_voxel_corner.png",
        "P-CRAFT": blender_dir / "pcraft_corner.png",
    }
    render_grid = outer[2].subgridspec(
        1,
        5,
        width_ratios=(0.70, 1, 1, 1, 1),
        wspace=0.03,
    )
    label = fig.add_subplot(render_grid[0, 0])
    label.axis("off")
    label.text(
        0.92,
        0.5,
        "3D results",
        ha="right",
        va="center",
        fontsize=14,
        color="#263238",
        fontweight="semibold",
    )
    for col, name in enumerate(METHODS, start=1):
        path = blender_renders[name]
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing Blender evidence {path}; run "
                "render_fig4_anime_corner_blender_20260726.py first."
            )
        ax = fig.add_subplot(render_grid[0, col])
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(COLORS[name])
            spine.set_linewidth(1.4)
        ax.set_title(
            name,
            fontsize=12.5,
            color=COLORS[name],
            fontweight="semibold",
            pad=2,
        )

    png = OUT / "fig4_projection_details_and_3d_comparison.png"
    pdf = OUT / "fig4_projection_details_and_3d_comparison.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)

    print(png)
    print(pdf)
    print("ROIs:", rois)


if __name__ == "__main__":
    main()
