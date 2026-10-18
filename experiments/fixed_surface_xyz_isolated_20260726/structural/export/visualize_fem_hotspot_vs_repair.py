"""
Visualize FEM stress hotspots against actual repaired voxels.

This script creates a paper-friendly explanation figure for the traditional
FEM repair pipeline by overlaying:
- original occupancy
- high-stress hotspot voxels from FEM
- voxels added by repair
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import ndimage as ndi

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _load_binary(path: Path, threshold: float = 0.5) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 4:
        arr = arr[0]
    return arr > threshold


def _overlay_from_masks(
    solid_2d: np.ndarray,
    hotspot_2d: np.ndarray,
    added_2d: np.ndarray,
) -> np.ndarray:
    img = np.zeros((*solid_2d.shape, 3), dtype=np.float32)
    img[..., :] = 0.08
    img[solid_2d > 0] = (0.80, 0.80, 0.80)
    img[hotspot_2d > 0] = (1.00, 0.62, 0.12)
    img[added_2d > 0] = (0.90, 0.10, 0.10)
    overlap = hotspot_2d & added_2d
    img[overlap > 0] = (0.75, 0.0, 0.85)
    return img


def _bbox_from_mask(mask: np.ndarray):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = coords.mean(axis=0)
    return mins, maxs, center


def _mip(mask: np.ndarray, axis: int) -> np.ndarray:
    return mask.max(axis=axis)


def create_hotspot_vs_repair_figure(
    original: np.ndarray,
    repaired: np.ndarray,
    von_mises: np.ndarray,
    output_path: Path,
    hotspot_percentile: float = 99.5,
) -> dict:
    solid = original > 0
    added = repaired & (~original)
    vm_vals = von_mises[solid]
    threshold = float(np.percentile(vm_vals, hotspot_percentile))
    hotspot = solid & (von_mises >= threshold)

    overlap = hotspot & added
    hotspot_dilated = ndi.binary_dilation(hotspot, iterations=2)
    overlap_dilated = hotspot_dilated & added

    hotspot_bbox = _bbox_from_mask(hotspot)
    added_bbox = _bbox_from_mask(added)

    if hotspot_bbox is not None:
        _, _, hotspot_center = hotspot_bbox
        center = np.round(hotspot_center).astype(int)
    else:
        center = np.array(original.shape) // 2

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.8))
    axis_specs = [
        ("X Projection", 0),
        ("Y Projection", 1),
        ("Z Projection", 2),
    ]

    for col, (title, axis) in enumerate(axis_specs):
        solid_mip = _mip(solid, axis)
        hotspot_mip = _mip(hotspot, axis)
        added_mip = _mip(added, axis)
        overlay = _overlay_from_masks(solid_mip, hotspot_mip, added_mip)
        axes[0, col].imshow(overlay)
        axes[0, col].set_title(f"{title}\nMIP overlay")
        axes[0, col].axis("off")

    slice_specs = [
        ("X Slice", original[center[0]], hotspot[center[0]], added[center[0]]),
        ("Y Slice", original[:, center[1], :], hotspot[:, center[1], :], added[:, center[1], :]),
        ("Z Slice", original[:, :, center[2]], hotspot[:, :, center[2]], added[:, :, center[2]]),
    ]
    for col, (title, solid_sl, hotspot_sl, added_sl) in enumerate(slice_specs):
        overlay = _overlay_from_masks(solid_sl, hotspot_sl, added_sl)
        axes[1, col].imshow(overlay)
        axes[1, col].set_title(f"{title}\nthrough hotspot center")
        axes[1, col].axis("off")

    legend_items = [
        Patch(facecolor=(0.80, 0.80, 0.80), edgecolor="none", label="Original solid"),
        Patch(facecolor=(1.00, 0.62, 0.12), edgecolor="none", label=f"High stress hotspot (top {100-hotspot_percentile:.1f}%)"),
        Patch(facecolor=(0.90, 0.10, 0.10), edgecolor="none", label="Added repair voxels"),
        Patch(facecolor=(0.75, 0.0, 0.85), edgecolor="none", label="Hotspot ∩ added"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))

    summary_lines = [
        f"Hotspot threshold (von Mises): {threshold:.3e} Pa",
        f"Hotspot voxels: {int(hotspot.sum())}",
        f"Added voxels: {int(added.sum())}",
        f"Direct overlap: {int(overlap.sum())}",
        f"Dilated-hotspot overlap: {int(overlap_dilated.sum())}",
    ]
    fig.text(
        0.02,
        0.02,
        "\n".join(summary_lines),
        fontsize=10,
        family="DejaVu Sans Mono",
        va="bottom",
    )

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    result = {
        "hotspot_percentile": hotspot_percentile,
        "hotspot_threshold": threshold,
        "hotspot_voxels": int(hotspot.sum()),
        "added_voxels": int(added.sum()),
        "direct_overlap_voxels": int(overlap.sum()),
        "dilated_hotspot_overlap_voxels": int(overlap_dilated.sum()),
        "hotspot_bbox": None if hotspot_bbox is None else {
            "min": hotspot_bbox[0].tolist(),
            "max": hotspot_bbox[1].tolist(),
            "center": hotspot_bbox[2].tolist(),
        },
        "added_bbox": None if added_bbox is None else {
            "min": added_bbox[0].tolist(),
            "max": added_bbox[1].tolist(),
            "center": added_bbox[2].tolist(),
        },
        "figure_path": str(output_path),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize FEM hotspot vs repaired voxels")
    parser.add_argument("--original", required=True, help="Original low-res density .npy")
    parser.add_argument("--repaired", required=True, help="Repaired low-res density .npy")
    parser.add_argument("--von_mises", required=True, help="FEM von Mises field .npy")
    parser.add_argument("--output", required=True, help="Output figure path")
    parser.add_argument("--summary_json", default=None, help="Optional summary json path")
    parser.add_argument("--hotspot_percentile", type=float, default=99.5)
    args = parser.parse_args()

    original = _load_binary(Path(args.original))
    repaired = _load_binary(Path(args.repaired))
    vm = np.load(args.von_mises)
    result = create_hotspot_vs_repair_figure(
        original=original,
        repaired=repaired,
        von_mises=vm,
        output_path=Path(args.output),
        hotspot_percentile=float(args.hotspot_percentile),
    )
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
