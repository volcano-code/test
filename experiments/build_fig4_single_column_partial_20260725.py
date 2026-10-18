#!/usr/bin/env python3
"""Build the evidence-backed, single-column partial Figure 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageOps
import trimesh


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
CASE = "adventurer_schoolgirl_fighter_original"
MITRA = ROOT / "results/fig4_anime_curated_original_v1_20260724/mitra" / CASE
VOXEL = Path(
    "/data/xvxinyao/ShadowArt-Revisited/voxel_results/"
    "fig4_anime_curated_original_v1_sadekar_20260725/voxel"
) / CASE / "seed_17"
MESH = Path(
    "/data/xvxinyao/ShadowArt-Revisited/mesh_results/"
    "fig4_anime_curated_original_v1_sadekar_20260725/mesh"
) / CASE / "seed_17"
OUT = ROOT / "paper_aaai_submission/fig4_single_column_partial_20260725"

COLORS = {
    "Target": "#263238",
    "Mitra": "#607D8B",
    "Sadekar Mesh": "#4F83CC",
    "Sadekar Voxel": "#00897B",
}


def binary_panel(path: Path, invert: bool = False) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L").resize((512, 512), Image.Resampling.NEAREST))
    if invert:
        arr = 255 - arr
    mask = arr > 127
    # Consistent paper convention: black foreground on white.
    return np.where(mask, 0, 255).astype(np.uint8)


def load_rows() -> list[tuple[str, list[np.ndarray]]]:
    targets = [
        binary_panel(MITRA / "target_view0_x.png", invert=True),
        binary_panel(MITRA / "target_view1_y.png", invert=True),
        binary_panel(MITRA / "target_view2_z.png", invert=True),
    ]
    mitra = [
        binary_panel(MITRA / "rendered_view0_x.png", invert=True),
        binary_panel(MITRA / "rendered_view1_y.png", invert=True),
        binary_panel(MITRA / "rendered_view2_z.png", invert=True),
    ]
    sadekar_mesh = [
        binary_panel(MESH / f"sample_0_pred_view_{i}.png", invert=True)
        for i in range(3)
    ]
    sadekar_voxel = [
        binary_panel(VOXEL / f"SHADOW_PRED0_view_{i}.png", invert=False)
        for i in range(3)
    ]
    return [
        ("Target", targets),
        ("Mitra", mitra),
        ("Sadekar Mesh", sadekar_mesh),
        ("Sadekar Voxel", sadekar_voxel),
    ]


def show_silhouette(ax, image: np.ndarray) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CFD8DC")
        spine.set_linewidth(0.8)


def show_zoom(ax, image: np.ndarray) -> None:
    # The upper body in the X view exposes both missing geometry and mesh artifacts.
    crop = image[72:344, 74:438]
    crop = np.asarray(
        ImageOps.contain(
            Image.fromarray(crop),
            (512, 512),
            Image.Resampling.NEAREST,
        )
    )
    show_silhouette(ax, crop)


def render_mesh(ax, path: Path, color: str, max_faces: int = 45000) -> None:
    mesh = trimesh.load(path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
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
        linewidth=0.025,
        alpha=1.0,
    )
    ax.add_collection3d(poly)
    ax.set_xlim(-0.50, 0.50)
    ax.set_ylim(-0.50, 0.50)
    ax.set_zlim(-0.50, 0.50)
    ax.view_init(elev=22, azim=-48)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 0.8))
    ax.set_axis_off()
    ax.set_facecolor("white")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()

    fig = plt.figure(figsize=(7.0, 8.6), dpi=320, facecolor="white")
    grid = fig.add_gridspec(
        6,
        5,
        left=0.045,
        right=0.985,
        top=0.955,
        bottom=0.035,
        width_ratios=(0.72, 1, 1, 1, 1.08),
        height_ratios=(0.18, 1, 1, 1, 1, 1.16),
        hspace=0.105,
        wspace=0.075,
    )

    for col, title in enumerate(("View X", "View Y", "View Z", "Detail")):
        ax = fig.add_subplot(grid[0, col + 1])
        ax.axis("off")
        ax.text(
            0.5,
            0.2,
            title,
            ha="center",
            va="bottom",
            fontsize=9.4,
            color="#263238",
            fontweight="semibold",
        )

    for ridx, (name, images) in enumerate(rows, start=1):
        label_ax = fig.add_subplot(grid[ridx, 0])
        label_ax.axis("off")
        label_ax.add_patch(
            Rectangle(
                (0.82, 0.08),
                0.07,
                0.84,
                transform=label_ax.transAxes,
                facecolor=COLORS[name],
                edgecolor="none",
            )
        )
        label_ax.text(
            0.73,
            0.5,
            name,
            ha="right",
            va="center",
            fontsize=8.7,
            color="#263238",
            fontweight="semibold" if name == "Target" else "normal",
            linespacing=0.95,
        )
        for cidx, image in enumerate(images):
            ax = fig.add_subplot(grid[ridx, cidx + 1])
            show_silhouette(ax, image)
            if cidx == 0:
                ax.add_patch(
                    Rectangle(
                        (74, 72),
                        364,
                        272,
                        fill=False,
                        edgecolor="#D1495B",
                        linewidth=1.0,
                    )
                )
        zoom_ax = fig.add_subplot(grid[ridx, 4])
        show_zoom(zoom_ax, images[0])
        for spine in zoom_ax.spines.values():
            spine.set_color("#D1495B")
            spine.set_linewidth(1.15)

    mesh_specs = [
        (
            "Mitra",
            MITRA / "mitra_pauly2009_shadow_hull.obj",
            "#78909C",
        ),
        (
            "Sadekar Mesh",
            MESH / "sample_0_output.obj",
            "#5C8FD4",
        ),
        (
            "Sadekar Voxel",
            VOXEL / "sample_0_output.obj",
            "#26A69A",
        ),
    ]
    bottom_label = fig.add_subplot(grid[5, 0])
    bottom_label.axis("off")
    bottom_label.text(
        0.73,
        0.5,
        "3D output",
        ha="right",
        va="center",
        fontsize=8.7,
        color="#263238",
        fontweight="semibold",
    )
    for cidx, (name, path, color) in enumerate(mesh_specs, start=1):
        ax = fig.add_subplot(grid[5, cidx], projection="3d")
        render_mesh(ax, path, color)
        ax.text2D(
            0.5,
            -0.02,
            name,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.2,
            color="#263238",
        )
        ax.patch.set_edgecolor(COLORS[name])
        ax.patch.set_linewidth(1.25)
    note_ax = fig.add_subplot(grid[5, 4])
    note_ax.set_xticks([])
    note_ax.set_yticks([])
    note_ax.set_facecolor("white")
    for spine in note_ax.spines.values():
        spine.set_color("#FFB300")
        spine.set_linewidth(1.25)
    note_ax.text(
        0.5,
        0.5,
        "P-CRAFT\npending",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#B26A00",
        fontweight="semibold",
        linespacing=1.15,
    )

    png = OUT / "fig4_partial_target_mitra_sadekar_single_column.png"
    pdf = OUT / "fig4_partial_target_mitra_sadekar_single_column.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
