#!/usr/bin/env python3
"""Build Figure 4 from the strongest evidence-backed Table 1 case."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from skimage.measure import marching_cubes


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
CASE = "connected_06"
PCRAFT = (
    ROOT
    / "results/fixed3_mr1_connected10_pcraft_20260722"
    / CASE
    / "seed_17"
)
MITRA = ROOT / "results/fixed3_mr1_connected10_mitra_20260722" / CASE
MESH = (
    Path("/data/xvxinyao/ShadowArt-Revisited/mesh_results")
    / "fixed3_mr1_connected10_sadekar_20260722"
    / "mesh"
    / CASE
    / "seed_17"
)
VOXEL = (
    Path("/data/xvxinyao/ShadowArt-Revisited/voxel_results")
    / "fixed3_mr1_connected10_sadekar_20260722"
    / "voxel"
    / CASE
    / "seed_17"
)
OUT = ROOT / "paper_aaai_submission/fig4_table1_connected06_20260726"

METHODS = ("Mitra", "Sadekar Mesh", "Sadekar Voxel", "P-CRAFT")
COLORS = {
    "Mitra": "#607D8B",
    "Sadekar Mesh": "#4F83CC",
    "Sadekar Voxel": "#00897B",
    "P-CRAFT": "#D98600",
}
# Method-native render order mapped to canonical target columns (X, Y, Z).
# Sadekar stores the first two fixed-camera views in Y/X order.
VIEW_SOURCE_ORDER = {
    "Mitra": (0, 1, 2),
    "Sadekar Mesh": (1, 0, 2),
    "Sadekar Voxel": (1, 0, 2),
    "P-CRAFT": (0, 1, 2),
}
METRICS = {
    "Mitra": {"iou": 0.8173998685662701, "dice": 0.8972819507002548},
    "Sadekar Mesh": {"iou": 0.8385823991116409, "dice": 0.9103815688184765},
    "Sadekar Voxel": {"iou": 0.8669621202932024, "dice": 0.9252192367390486},
    "P-CRAFT": {"iou": 0.9649994497192215, "dice": 0.9820091463939132},
}
VIEW_NAMES = ("View X", "View Y", "View Z")
SIZE = 512
ROI_SIZE = 112


def load_gray(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (SIZE, SIZE):
        image = image.resize((SIZE, SIZE), Image.Resampling.NEAREST)
    return np.asarray(image)


def best_polarity(path: Path, target: np.ndarray) -> np.ndarray:
    """Resolve only image polarity; never crop, recenter, or register a result."""
    raw = load_gray(path) > 127
    candidates = (raw, ~raw)

    def iou(mask: np.ndarray) -> float:
        union = np.logical_or(mask, target).sum()
        return float(np.logical_and(mask, target).sum() / max(union, 1))

    return max(candidates, key=iou)


def align_views(
    paths: list[Path],
    targets: list[np.ndarray],
    source_for_target: tuple[int, ...],
) -> tuple[list[np.ndarray], tuple[int, ...]]:
    """Map method-native view indices to the canonical X/Y/Z target columns."""
    raw = [load_gray(path) > 127 for path in paths]

    def score(mask: np.ndarray, target: np.ndarray) -> float:
        union = np.logical_or(mask, target).sum()
        return float(np.logical_and(mask, target).sum() / max(union, 1))

    aligned = []
    for target_idx, source_idx in enumerate(source_for_target):
        mask = raw[source_idx]
        target = targets[target_idx]
        aligned.append(
            max(
                (mask, ~mask),
                key=lambda candidate: score(candidate, target),
            )
        )
    return aligned, source_for_target


def load_evidence() -> tuple[
    list[np.ndarray], dict[str, list[np.ndarray]], dict[str, tuple[int, ...]]
]:
    targets = [
        load_gray(PCRAFT / "projections" / f"target_view_{i}.png") > 127
        for i in range(3)
    ]
    method_paths = {
        "Mitra": [
            MITRA / f"rendered_view{i}_{'xyz'[i]}.png" for i in range(3)
        ],
        "Sadekar Mesh": [
            MESH / f"sample_0_pred_view_{i}.png" for i in range(3)
        ],
        "Sadekar Voxel": [
            VOXEL / f"SHADOW_PRED0_view_{i}.png" for i in range(3)
        ],
        "P-CRAFT": [
            PCRAFT / "projections" / f"projection_view_{i}.png"
            for i in range(3)
        ],
    }
    rows = {}
    mappings = {}
    for name, paths in method_paths.items():
        rows[name], mappings[name] = align_views(
            paths, targets, VIEW_SOURCE_ORDER[name]
        )
    return targets, rows, mappings


def local_sum(array: np.ndarray, window: int) -> np.ndarray:
    padded = np.pad(array, ((1, 0), (1, 0)))
    integral = padded.cumsum(0).cumsum(1)
    out = np.zeros_like(array, dtype=np.float32)
    radius = window // 2
    for cy in range(radius, SIZE - radius, 3):
        y0, y1 = cy - radius, cy + radius
        for cx in range(radius, SIZE - radius, 3):
            x0, x1 = cx - radius, cx + radius
            out[cy, cx] = (
                integral[y1, x1]
                - integral[y0, x1]
                - integral[y1, x0]
                + integral[y0, x0]
            )
    return out


def choose_roi(
    target: np.ndarray, rows: dict[str, list[np.ndarray]], view: int
) -> tuple[int, int]:
    """Choose one shared ROI where baseline errors exceed P-CRAFT errors."""
    baseline_error = np.mean(
        [(rows[name][view] != target).astype(np.float32) for name in METHODS[:-1]],
        axis=0,
    )
    pcraft_error = (rows["P-CRAFT"][view] != target).astype(np.float32)
    padded = np.pad(target, 1, constant_values=False)
    eroded = target.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        eroded &= padded[1 + dy : 1 + dy + SIZE, 1 + dx : 1 + dx + SIZE]
    boundary = target & ~eroded

    score = (
        1.0 * baseline_error
        - 1.8 * pcraft_error
        + 0.75 * boundary.astype(np.float32)
        + 0.04 * target.astype(np.float32)
    )
    window_score = local_sum(score, ROI_SIZE)
    target_pixels = local_sum(target.astype(np.float32), ROI_SIZE)
    baseline_pixels = local_sum(baseline_error, ROI_SIZE)
    pcraft_pixels = local_sum(pcraft_error, ROI_SIZE)

    valid = (
        (target_pixels > 0.035 * ROI_SIZE**2)
        & (baseline_pixels > 1.35 * np.maximum(pcraft_pixels, 1.0))
    )
    window_score[~valid] = -np.inf
    cy, cx = np.unravel_index(np.argmax(window_score), window_score.shape)
    if not np.isfinite(window_score[cy, cx]):
        cy, cx = SIZE // 2, SIZE // 2
    return int(cx), int(cy)


def paper_image(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, 255, 0).astype(np.uint8)


def show_mask(ax, mask: np.ndarray, border: str = "#B0BEC5", width: float = 0.7) -> None:
    ax.imshow(paper_image(mask), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(border)
        spine.set_linewidth(width)


def show_main(ax, mask: np.ndarray, roi: tuple[int, int], color: str) -> None:
    show_mask(ax, mask)
    cx, cy = roi
    ax.add_patch(
        Rectangle(
            (cx - ROI_SIZE / 2, cy - ROI_SIZE / 2),
            ROI_SIZE,
            ROI_SIZE,
            fill=False,
            edgecolor=color,
            linewidth=1.45,
        )
    )


def show_detail(ax, mask: np.ndarray, roi: tuple[int, int], color: str) -> None:
    cx, cy = roi
    radius = ROI_SIZE // 2
    crop = mask[cy - radius : cy + radius, cx - radius : cx + radius]
    show_mask(ax, crop, border=color, width=1.45)


def make_mitra_surface() -> Path:
    """Extract a display surface from the baseline occupancy without altering it."""
    out = OUT / "mitra_connected06_surface_for_visualization.obj"
    if out.is_file():
        return out
    volume = np.load(MITRA / "mitra_pauly2009_shadow_hull.npy").astype(np.float32)
    vertices, faces, _, _ = marching_cubes(
        volume, level=0.5, step_size=2, allow_degenerate=False
    )
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(out)
    return out


def render_mesh(
    ax, path: Path, color: str, elev: float = 22, azim: float = -48
) -> None:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if len(faces) > 45000:
        keep = np.linspace(0, len(faces) - 1, 45000, dtype=np.int64)
        faces = faces[keep]
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    scale = max(float(np.ptp(vertices, axis=0).max()), 1e-8)
    vertices = (vertices - center) / scale
    collection = Poly3DCollection(
        vertices[faces],
        facecolor=color,
        edgecolor=color,
        linewidth=0.012,
        alpha=1.0,
    )
    ax.add_collection3d(collection)
    ax.set_xlim(-0.52, 0.52)
    ax.set_ylim(-0.52, 0.52)
    ax.set_zlim(-0.52, 0.52)
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 0.9))
    ax.set_axis_off()
    ax.set_facecolor("white")


def exact_iou(mask: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(mask, target).sum()
    return float(np.logical_and(mask, target).sum() / max(union, 1))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets, rows, view_mappings = load_evidence()
    rois = [choose_roi(targets[i], rows, i) for i in range(3)]

    fig = plt.figure(figsize=(7.15, 8.35), dpi=360, facecolor="white")
    outer = fig.add_gridspec(
        3,
        1,
        left=0.025,
        right=0.992,
        top=0.972,
        bottom=0.018,
        height_ratios=(1.02, 4.1, 1.72),
        hspace=0.055,
    )

    targets_grid = outer[0].subgridspec(
        1, 4, width_ratios=(0.9, 1, 1, 1), wspace=0.09
    )
    label_ax = fig.add_subplot(targets_grid[0, 0])
    label_ax.axis("off")
    label_ax.text(
        0.95,
        0.5,
        "Targets",
        ha="right",
        va="center",
        fontsize=12.2,
        fontweight="semibold",
        color="#263238",
    )
    for view in range(3):
        ax = fig.add_subplot(targets_grid[0, view + 1])
        show_mask(ax, targets[view], border="#263238", width=0.9)
        ax.set_title(VIEW_NAMES[view], fontsize=10.8, color="#263238", pad=1)

    compare = outer[1].subgridspec(
        4,
        7,
        width_ratios=(1.0, 1, 0.52, 1, 0.52, 1, 0.52),
        hspace=0.065,
        wspace=0.03,
    )
    for ridx, name in enumerate(METHODS):
        label_ax = fig.add_subplot(compare[ridx, 0])
        label_ax.axis("off")
        label_ax.add_patch(
            Rectangle(
                (0.91, 0.12),
                0.038,
                0.76,
                transform=label_ax.transAxes,
                color=COLORS[name],
                linewidth=0,
            )
        )
        display_name = {
            "Mitra": "Mitra",
            "Sadekar Mesh": "Sadekar\nMesh",
            "Sadekar Voxel": "Sadekar\nVoxel",
            "P-CRAFT": "P-CRAFT",
        }[name]
        label_ax.text(
            0.86,
            0.50,
            display_name,
            ha="right",
            va="center",
            fontsize=10.2,
            color="#263238",
            fontweight="bold" if name == "P-CRAFT" else "normal",
            linespacing=0.92,
        )
        roi_color = COLORS[name]
        for view in range(3):
            main_ax = fig.add_subplot(compare[ridx, 1 + 2 * view])
            detail_ax = fig.add_subplot(compare[ridx, 2 + 2 * view])
            show_main(main_ax, rows[name][view], rois[view], roi_color)
            show_detail(detail_ax, rows[name][view], rois[view], roi_color)
            if ridx == 0:
                detail_ax.set_title(
                    "ROI zoom", fontsize=8.2, color="#607D8B", pad=1
                )

    blender_renders = {
        "Mitra": OUT / "blender_renders/mitra_view_a.png",
        "Sadekar Mesh": OUT / "blender_renders/sadekar_mesh_view_a.png",
        "Sadekar Voxel": OUT / "blender_renders/sadekar_voxel_view_a.png",
        "P-CRAFT": OUT / "blender_renders/pcraft_view_a.png",
    }
    missing_renders = [str(path) for path in blender_renders.values() if not path.is_file()]
    if missing_renders:
        raise FileNotFoundError(
            "Missing fixed-camera Blender renders:\n" + "\n".join(missing_renders)
        )
    render_grid = outer[2].subgridspec(
        1, 5, width_ratios=(0.92, 1, 1, 1, 1), wspace=0.025
    )
    label_ax = fig.add_subplot(render_grid[0, 0])
    label_ax.axis("off")
    label_ax.text(
        0.94,
        0.5,
        "Fixed-camera\n3D results",
        ha="right",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
        color="#263238",
        linespacing=0.95,
    )
    for col, name in enumerate(METHODS, start=1):
        ax = fig.add_subplot(render_grid[0, col])
        ax.imshow(plt.imread(blender_renders[name]))
        ax.axis("off")
        render_name = {
            "Mitra": "Mitra",
            "Sadekar Mesh": "Sadekar\nMesh",
            "Sadekar Voxel": "Sadekar\nVoxel",
            "P-CRAFT": "P-CRAFT",
        }[name]
        ax.set_title(
            render_name,
            fontsize=8.9,
            color=COLORS[name],
            fontweight="semibold" if name == "P-CRAFT" else "normal",
            pad=-3,
        )

    png = OUT / "fig4_connected06_projection_details_and_3d.png"
    pdf = OUT / "fig4_connected06_projection_details_and_3d.pdf"
    final_png = OUT / "fig4_connected06_single_column_final.png"
    final_pdf = OUT / "fig4_connected06_single_column_final.pdf"
    fig.savefig(png, dpi=360, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(final_png, dpi=360, facecolor="white")
    fig.savefig(final_pdf, facecolor="white")
    plt.close(fig)

    audit = {
        "case_id": CASE,
        "selection_reason": "largest P-CRAFT IoU margin over the best Table 1 baseline",
        "shared_rois_xy": rois,
        "metrics": METRICS,
        "measured_display_ious": {
            name: [exact_iou(rows[name][i], targets[i]) for i in range(3)]
            for name in METHODS
        },
        "source_view_for_target_xyz": {
            name: list(mapping) for name, mapping in view_mappings.items()
        },
        "sources": {
            "targets_and_pcraft": str(PCRAFT),
            "mitra": str(MITRA),
            "sadekar_mesh": str(MESH),
            "sadekar_voxel": str(VOXEL),
            "pcraft_display_stl": str(PCRAFT / "selected_display_fixed.stl"),
        },
        "note": (
            "All silhouettes preserve the stored pixel coordinate system. "
            "Method-native view indices are remapped to X/Y/Z using the stored "
            "fixed-camera protocol. Only polarity and nearest-neighbor "
            "resolution matching are applied."
        ),
    }
    (OUT / "EVIDENCE.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(png)
    print(pdf)
    print(final_png)
    print(final_pdf)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
