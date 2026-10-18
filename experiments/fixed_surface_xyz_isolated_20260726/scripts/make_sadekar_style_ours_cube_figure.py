#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from PIL import Image, ImageDraw, ImageOps


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/sadekar_style_main_comparison_20260705"
BLENDER = OUT / "blender_simulation"
BENCH = Path("/data/xvxinyao/xvxinyao/total_project/mesh_combined_globalconsistent_tfs_20260614/results/aaai2027_paper_20260615/e1_benchmark_final")
METRICS = OUT / "figure_metrics_with_mitra_selected_6cc.tsv"

SETS = [
    ("setA_owl_lobster_ant", "(a) owl / lobster / ant"),
    ("setB_lobster_beetle_bat", "(b) lobster / beetle / bat"),
    ("setC_turtle_scorpion_eagle", "(c) turtle / scorpion / eagle"),
    ("setD_dragon_trex_scorpion", "(d) dragon / T-rex / scorpion"),
]

METHODS = [
    ("Target shadows", "target"),
    ("Projection-only carve", "mitra"),
    ("Sadekar mesh", "sadekar_mesh"),
    ("Sadekar voxel", "sadekar_voxel"),
    ("ShadowForge selected", "ours"),
]


def read_metrics():
    rows = {}
    with open(METRICS, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[(r["set"], r["method"])] = r
    return rows


def load_bw(path, size=80):
    img = Image.open(path).convert("L")
    img = ImageOps.autocontrast(img)
    arr = np.array(img)
    # Convert both black-on-white and white-on-black masks to black foreground.
    corner = np.r_[arr[:6, :6].ravel(), arr[:6, -6:].ravel(), arr[-6:, :6].ravel(), arr[-6:, -6:].ravel()]
    bg_dark = np.median(corner) < 128
    mask = arr > 128 if bg_dark else arr < 128
    ys, xs = np.where(mask)
    if len(xs):
        pad = 3
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
        mask = mask[y0:y1, x0:x1]
    im = Image.fromarray((~mask * 255).astype(np.uint8)).resize((size, size), Image.Resampling.LANCZOS)
    return np.array(im)


def projection_from_voxels(path, view, size=80):
    v = np.load(path) > 0
    axes = [(0, 1), (0, 2), (1, 2)]
    p = v.max(axis=view)
    p = np.rot90(p)
    ys, xs = np.where(p)
    if len(xs):
        pad = 2
        p = p[max(ys.min()-pad, 0):min(ys.max()+pad+1, p.shape[0]),
              max(xs.min()-pad, 0):min(xs.max()+pad+1, p.shape[1])]
    im = Image.fromarray((~p * 255).astype(np.uint8)).resize((size, size), Image.Resampling.LANCZOS)
    return np.array(im)


def image_for(set_name, method, view):
    base = BENCH / set_name
    if method == "target":
        return load_bw(base / "targets" / f"input_view{view}.png")
    if method == "mitra":
        return load_bw(OUT / "mitra_style_reimplementation_A-D" / set_name / f"pred_view{view}.png")
    if method in {"sadekar_mesh", "sadekar_voxel"}:
        return load_bw(base / method / "shadows" / f"pred_view{view}.png")
    if method == "ours":
        return load_bw(base / "ours_fixed" / "shadows" / f"pred_view{view}.png")
    raise ValueError(method)


def add_shadow_image(ax, arr, xy, wh, alpha=0.42):
    x, y = xy
    w, h = wh
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=float)
    fg = arr < 180
    rgba[..., 0:3] = 0.05
    rgba[..., 3] = fg.astype(float) * alpha
    ax.imshow(rgba, extent=(x, x + w, y, y + h), origin="upper", zorder=3)


def draw_voxel_thumb(ax, vox_path, center=(0.50, 0.42), scale=0.105):
    v = np.load(vox_path) > 0
    coords = np.argwhere(v)
    if len(coords) == 0:
        return
    # Deterministic sparse surface-ish thumbnail. Small enough not to hide the floor shadow.
    rng = np.random.default_rng(0)
    if len(coords) > 900:
        coords = coords[rng.choice(len(coords), 900, replace=False)]
    c = coords.astype(float)
    c = (c - c.min(0)) / np.maximum(np.ptp(c, axis=0), 1)
    x = center[0] + scale * (c[:, 2] - c[:, 0]) * 0.82
    y = center[1] + scale * (0.48 * (c[:, 2] + c[:, 0]) - c[:, 1]) * 0.72
    depth = c[:, 0] + c[:, 1] + c[:, 2]
    order = np.argsort(depth)
    ax.scatter(x[order], y[order], s=2.0, c="#2459c6", alpha=0.52, linewidths=0, zorder=6)


def perspective_coeffs(src, dst):
    a = []
    b = []
    for (x, y), (u, v) in zip(dst, src):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.extend([u, v])
    return np.linalg.solve(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def silhouette_rgba(path, size, alpha=205):
    arr = load_bw(path, size)
    mask = arr < 128
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[mask, :3] = 24
    rgba[mask, 3] = alpha
    return Image.fromarray(rgba, "RGBA")


def voxel_projection_rgba(voxels, view, size, alpha=130):
    mask = np.rot90((voxels > 0).max(axis=view))
    ys, xs = np.where(mask)
    if len(xs):
        pad = 2
        mask = mask[max(ys.min() - pad, 0):min(ys.max() + pad + 1, mask.shape[0]),
                    max(xs.min() - pad, 0):min(xs.max() + pad + 1, mask.shape[1])]
    img = Image.fromarray((mask * 255).astype(np.uint8)).resize((size, size), Image.Resampling.LANCZOS)
    m = np.array(img) > 0
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[m, :3] = 54
    rgba[m, 3] = alpha
    return Image.fromarray(rgba, "RGBA")


def paste_perspective(canvas, img, quad):
    w, h = canvas.size
    src = [(0, 0), (img.width, 0), (img.width, img.height), (0, img.height)]
    coeffs = perspective_coeffs(src, quad)
    warped = img.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    canvas.alpha_composite(warped)


def draw_projected_voxel_shadow(draw, pts_norm, proj, plane, fill=(18, 18, 18, 225)):
    if plane == "left":
        q = [(0.0, float(p[1]), float(p[0])) for p in pts_norm]
    elif plane == "back":
        q = [(float(p[2]), 0.0, float(p[0])) for p in pts_norm]
    elif plane == "floor":
        q = [(float(p[2]), float(p[1]), 0.0) for p in pts_norm]
    else:
        raise ValueError(plane)
    xy = np.array([proj(p) for p in q])
    # Small squares make a solid-looking raster shadow while still being generated
    # from projected occupied voxels, not pasted silhouette images.
    r = 1.8 if plane != "floor" else 1.5
    for x, y in xy:
        draw.rectangle((x - r, y - r, x + r, y + r), fill=fill)


def draw_cube_scene(ax, set_name, mean_iou, mean_ds):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    blender_img = BLENDER / f"{set_name}_blender_corner_sim.png"
    if blender_img.exists():
        img_rgba = Image.open(blender_img).convert("RGBA")
        arr = np.array(img_rgba)
        # Blender renders transparent outside the simulated corner. Crop by alpha
        # and composite on white so no black renderer background remains.
        visible = arr[..., 3] > 8
        ys, xs = np.where(visible)
        if len(xs):
            pad = 10
            img_rgba = img_rgba.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                                      min(xs.max() + pad + 1, img_rgba.width),
                                      min(ys.max() + pad + 1, img_rgba.height)))
        white = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
        white.alpha_composite(img_rgba)
        img = white.convert("RGB")
        ax.imshow(img, extent=(0.08, 0.92, 0.16, 0.88), origin="upper")
        ax.text(0.50, 0.93, "ShadowForge 3D sculpture", ha="center", va="top", fontsize=7)
        ax.text(0.50, 0.055, f"IoU {mean_iou:.3f}   DS {mean_ds:.3f}", ha="center", va="bottom", fontsize=7)
        return

    W, H = 900, 680
    scene = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    draw = ImageDraw.Draw(scene, "RGBA")

    def proj(p):
        # Strict axonometric projection of a cube corner.
        # left wall: x=0, back/right wall: y=0, floor: z=0.
        x, y, z = p
        cx, cy = 448.0, 392.0
        sx, sy, sz = 178.0, 74.0, 244.0
        return (cx + sx * (x - y), cy + sy * (x + y) - sz * z)

    corner_top = proj((0, 0, 1))
    corner_bottom = proj((0, 0, 0))
    left_top = proj((0, 1, 1))
    left_bottom = proj((0, 1, 0))
    right_top = proj((1, 0, 1))
    right_bottom = proj((1, 0, 0))
    floor_front = proj((1, 1, 0))

    left = [left_top, corner_top, corner_bottom, left_bottom]
    right = [corner_top, right_top, right_bottom, corner_bottom]
    floor = [left_bottom, corner_bottom, right_bottom, floor_front]

    draw.polygon(floor, fill=(202, 212, 222, 220), outline=(104, 112, 122, 255))
    draw.polygon(left, fill=(232, 238, 166, 230), outline=(100, 108, 82, 255))
    draw.polygon(right, fill=(176, 139, 200, 230), outline=(100, 76, 118, 255))
    for a, b in [
        (corner_top, corner_bottom),
        (left_top, corner_top), (corner_top, right_top),
        (left_bottom, corner_bottom), (corner_bottom, right_bottom),
        (left_top, left_bottom), (right_top, right_bottom),
        (left_bottom, floor_front), (right_bottom, floor_front),
    ]:
        draw.line([a, b], fill=(78, 78, 78, 255), width=3)

    vox = OUT / "shadowforge_selected_6cc_cleaned" / set_name / "shadowforge_selected_6cc_cleaned.npy"
    v = np.load(vox) > 0
    pts = np.argwhere(v)
    if len(pts):
        rng = np.random.default_rng(0)
        pts = pts.astype(float)
        pts = (pts - pts.min(0)) / np.maximum(np.ptp(pts, axis=0), 1)
        paste_perspective(scene, voxel_projection_rgba(v, 0, 140, 135),
                          [proj((0, 0.18, 0.74)), proj((0, 0.52, 0.74)),
                           proj((0, 0.52, 0.43)), proj((0, 0.18, 0.43))])
        paste_perspective(scene, voxel_projection_rgba(v, 1, 140, 135),
                          [proj((0.48, 0, 0.74)), proj((0.82, 0, 0.74)),
                           proj((0.82, 0, 0.43)), proj((0.48, 0, 0.43))])
        paste_perspective(scene, voxel_projection_rgba(v, 2, 115, 105),
                          [proj((0.36, 0.58, 0)), proj((0.64, 0.58, 0)),
                           proj((0.64, 0.84, 0)), proj((0.36, 0.84, 0))])

        if len(pts) > 1500:
            pts = pts[rng.choice(len(pts), 1500, replace=False)]
        x3 = 0.50 + (pts[:, 2] - 0.5) * 0.18
        y3 = 0.50 + (pts[:, 1] - 0.5) * 0.18
        z3 = 0.20 + pts[:, 0] * 0.22
        xy = np.array([proj(p) for p in zip(x3, y3, z3)])
        xs, ys = xy[:, 0], xy[:, 1]
        order = np.argsort(pts[:, 0] + pts[:, 1] + pts[:, 2])
        d = ImageDraw.Draw(scene, "RGBA")
        for x, y in zip(xs[order], ys[order]):
            d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(185, 54, 32, 238))

    alpha = np.array(scene)[..., 3]
    ys, xs = np.where(alpha > 0)
    if len(xs):
        pad = 28
        box = (
            max(int(xs.min()) - pad, 0),
            max(int(ys.min()) - pad, 0),
            min(int(xs.max()) + pad + 1, W),
            min(int(ys.max()) + pad + 1, H),
        )
        scene = scene.crop(box)
    ax.imshow(scene, extent=(0.12, 0.88, 0.12, 0.90), origin="upper")
    ax.text(0.50, 0.96, "ShadowForge 3D sculpture", ha="center", va="top", fontsize=7)
    ax.text(0.50, 0.035, f"IoU {mean_iou:.3f}   DS {mean_ds:.3f}", ha="center", va="bottom", fontsize=7)


def draw_triplet(ax, set_name, method):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    for j in range(3):
        x0 = 0.05 + j * 0.315
        ax.add_patch(Rectangle((x0, 0.22), 0.255, 0.66, fill=False, lw=0.55, ec="#777777"))
        ax.imshow(image_for(set_name, method, j), cmap="gray", vmin=0, vmax=255,
                  extent=(x0 + 0.012, x0 + 0.243, 0.24, 0.86), origin="upper",
                  aspect="auto")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics()

    fig = plt.figure(figsize=(13.8, 8.0), dpi=220)
    gs = fig.add_gridspec(
        nrows=6, ncols=5,
        width_ratios=[0.58, 1, 1, 1, 1],
        height_ratios=[1.62, 1.08, 1.08, 1.08, 1.08, 1.08],
        left=0.045, right=0.995, top=0.965, bottom=0.045,
        hspace=0.075, wspace=0.035,
    )

    for c, (set_name, title) in enumerate(SETS, start=1):
        ax_title = fig.add_subplot(gs[0, c])
        sf = metrics[(set_name, "ShadowForge selected 6CC-cleaned")]
        draw_cube_scene(ax_title, set_name, float(sf["mean_iou"]), float(sf["mean_dice"]))
        ax_title.set_title(title, fontsize=10, fontweight="bold", pad=4)

    for r, (label, method) in enumerate(METHODS, start=1):
        lab = fig.add_subplot(gs[r, 0])
        lab.axis("off")
        lab.text(0.98, 0.5, label, ha="right", va="center", fontsize=10, fontweight="bold")
        for c, (set_name, _title) in enumerate(SETS, start=1):
            ax = fig.add_subplot(gs[r, c])
            draw_triplet(ax, set_name, method)
            if method != "target":
                key = {
                    "mitra": "Projection-only carve",
                    "sadekar_mesh": "Sadekar mesh",
                    "sadekar_voxel": "Sadekar voxel",
                    "ours": "ShadowForge selected 6CC-cleaned",
                }[method]
                m = metrics[(set_name, key)]
                ax.text(0.50, 0.08, f"IoU {float(m['mean_iou']):.3f}   DS {float(m['mean_dice']):.3f}",
                        ha="center", va="center", fontsize=7.5)

    fig.text(0.52, 0.013,
             "Sadekar-style comparison using local artifacts: top row shows only the selected ShadowForge model in a three-face cube scene; rows below show rendered shadows.",
             ha="center", va="bottom", fontsize=7)
    png = OUT / "sadekar_style_ours_cube_scene_small_model.png"
    pdf = OUT / "sadekar_style_ours_cube_scene_small_model.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
