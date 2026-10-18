"""
Map a low-resolution FEM repair result back to a high-resolution voxel volume.

Two remap modes are supported:
1. structural_preserving:
   preserve the repaired low-resolution structure when projected back to high
   resolution, and enforce consistency under the same downsampling used by the
   FEM verification script.
2. display_safe:
   keep the outer contour as unchanged as possible for presentation-only assets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, zoom

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from structural.export.voxel_to_mesh import voxel_to_obj


def _load_binary_volume(path: Path, threshold: float = 0.5) -> np.ndarray:
    vol = np.load(path)
    if vol.ndim == 4:
        vol = vol[0]
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D or 4D voxel array at {path}, got shape {vol.shape}")
    return vol > threshold


def _nearest_upsample(binary_volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    factors = tuple(t / s for s, t in zip(binary_volume.shape, target_shape))
    if not all(abs(round(f) - f) < 1e-6 for f in factors):
        raise ValueError(
            f"Only integer-ratio remapping is supported, got source {binary_volume.shape} -> target {target_shape}"
        )
    up = zoom(binary_volume.astype(np.float32), zoom=factors, order=0)
    return up > 0.5


def _downsample_like_fem_verify(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    factors = tuple(t / s for s, t in zip(volume.shape, target_shape))
    down = zoom(volume.astype(np.float32), zoom=factors, order=1)
    down = np.clip(down, 0.0, 1.0)
    return down > 0.5


def _block_slices(low_idx: Tuple[int, int, int], ratio: int) -> Tuple[slice, slice, slice]:
    return tuple(slice(i * ratio, (i + 1) * ratio) for i in low_idx)


def _projection_masks(binary_volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    proj_x = binary_volume.max(axis=0)  # yz
    proj_y = binary_volume.max(axis=1)  # xz
    proj_z = binary_volume.max(axis=2)  # xy
    return proj_x, proj_y, proj_z


def _silhouette_safe_add_region(binary_volume: np.ndarray) -> np.ndarray:
    proj_x, proj_y, proj_z = _projection_masks(binary_volume)
    # Any added voxel that remains covered in all three orthographic projections
    # does not alter the x/y/z silhouettes.
    return proj_x[None, :, :] & proj_y[:, None, :] & proj_z[:, :, None]


def remap_repaired_volume_display(
    original_high: np.ndarray,
    repaired_low: np.ndarray,
    proximity_voxels: int = 4,
) -> Dict[str, np.ndarray]:
    up_repaired = _nearest_upsample(repaired_low, original_high.shape)
    prox_mask = binary_dilation(original_high, iterations=int(proximity_voxels))
    safe_add_mask = _silhouette_safe_add_region(original_high)

    add_mask = (~original_high) & up_repaired & prox_mask & safe_add_mask
    repaired_high = original_high | add_mask

    return {
        "repaired_high": repaired_high,
        "upsampled_repaired_low": up_repaired,
        "add_mask": add_mask,
        "remove_mask": original_high & (~repaired_high),
        "safe_add_mask": safe_add_mask,
        "proximity_mask": prox_mask,
    }


def remap_repaired_volume_structural(
    original_high: np.ndarray,
    repaired_low: np.ndarray,
) -> Dict[str, np.ndarray]:
    target_shape = repaired_low.shape
    if any(h % l != 0 for h, l in zip(original_high.shape, target_shape)):
        raise ValueError(
            f"High-res shape {original_high.shape} must be an integer multiple of low-res shape {target_shape}"
        )
    ratios = [h // l for h, l in zip(original_high.shape, target_shape)]
    if len(set(ratios)) != 1:
        raise ValueError(f"Only isotropic ratios are supported, got {ratios}")
    ratio = ratios[0]

    original_low = _downsample_like_fem_verify(original_high, repaired_low.shape)
    repaired_high = original_high.copy()

    low_add = repaired_low & (~original_low)
    low_remove = original_low & (~repaired_low)
    low_same = repaired_low == original_low

    changed_coords = np.argwhere(low_add | low_remove)
    for coord in changed_coords:
        coord_t = tuple(int(x) for x in coord)
        slices = _block_slices(coord_t, ratio)
        repaired_high[slices] = repaired_low[coord_t]

    # Keep unchanged low-res cells exactly as the original high-res detail.
    unchanged_coords = np.argwhere(low_same)
    for coord in unchanged_coords:
        coord_t = tuple(int(x) for x in coord)
        slices = _block_slices(coord_t, ratio)
        repaired_high[slices] = original_high[slices]

    add_mask = repaired_high & (~original_high)
    remove_mask = original_high & (~repaired_high)

    downsampled_back = _downsample_like_fem_verify(repaired_high, repaired_low.shape)
    low_consistency_iou = _axis_iou(downsampled_back, repaired_low)

    return {
        "repaired_high": repaired_high,
        "original_low": original_low,
        "downsampled_repaired_high": downsampled_back,
        "low_add_mask": low_add,
        "low_remove_mask": low_remove,
        "add_mask": add_mask,
        "remove_mask": remove_mask,
        "low_consistency_iou": low_consistency_iou,
        "low_add_voxels_preserved": int((downsampled_back & low_add).sum()),
        "low_add_voxels_total": int(low_add.sum()),
    }


def remap_repaired_volume(
    original_high: np.ndarray,
    repaired_low: np.ndarray,
    mode: str = "structural_preserving",
    proximity_voxels: int = 4,
) -> Dict[str, np.ndarray]:
    if mode == "display_safe":
        result = remap_repaired_volume_display(
            original_high=original_high,
            repaired_low=repaired_low,
            proximity_voxels=proximity_voxels,
        )
        result["mode"] = "display_safe"
        return result

    result = remap_repaired_volume_structural(
        original_high=original_high,
        repaired_low=repaired_low,
    )
    result["mode"] = "structural_preserving"
    return result


def _axis_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def _axis_dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def evaluate_silhouette_changes(original_high: np.ndarray, repaired_high: np.ndarray) -> Dict[str, object]:
    axes = {
        "view_x": (original_high.max(axis=0), repaired_high.max(axis=0)),
        "view_y": (original_high.max(axis=1), repaired_high.max(axis=1)),
        "view_z": (original_high.max(axis=2), repaired_high.max(axis=2)),
    }
    per_view = {}
    ious = []
    dices = []
    for name, (orig_proj, rep_proj) in axes.items():
        iou = _axis_iou(orig_proj, rep_proj)
        dice = _axis_dice(orig_proj, rep_proj)
        added = int(np.logical_and(~orig_proj, rep_proj).sum())
        removed = int(np.logical_and(orig_proj, ~rep_proj).sum())
        per_view[name] = {
            "iou": iou,
            "dice": dice,
            "added_pixels": added,
            "removed_pixels": removed,
            "changed_pixels": added + removed,
        }
        ious.append(iou)
        dices.append(dice)
    return {
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "per_view": per_view,
    }


def _mip(volume: np.ndarray, axis: int) -> np.ndarray:
    return volume.max(axis=axis).astype(np.uint8)


def save_difference_panel(
    original_high: np.ndarray,
    repaired_high: np.ndarray,
    add_mask: np.ndarray,
    remove_mask: np.ndarray,
    out_path: Path,
) -> None:
    axis_specs = [
        ("X Projection", 0),
        ("Y Projection", 1),
        ("Z Projection", 2),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for col, (title, axis) in enumerate(axis_specs):
        orig_img = _mip(original_high, axis)
        rep_img = _mip(repaired_high, axis)
        add_img = _mip(add_mask, axis)
        rem_img = _mip(remove_mask, axis)

        overlay = np.zeros((*orig_img.shape, 3), dtype=np.float32)
        overlay[..., :] = 0.08
        overlay[orig_img > 0] = (0.75, 0.75, 0.75)
        overlay[rep_img > 0] = np.maximum(overlay[rep_img > 0], np.array([0.95, 0.95, 0.95]))
        overlay[add_img > 0] = (0.90, 0.15, 0.15)
        overlay[rem_img > 0] = (0.15, 0.35, 0.95)

        axes[0, col].imshow(orig_img, cmap="gray")
        axes[0, col].set_title(f"{title}\nOriginal")
        axes[1, col].imshow(rep_img, cmap="gray")
        axes[1, col].set_title("Remapped Repair")
        axes[2, col].imshow(overlay)
        axes[2, col].set_title("Difference Overlay\nred=added, blue=removed")

        for row in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_slice_panel(
    original_high: np.ndarray,
    repaired_high: np.ndarray,
    add_mask: np.ndarray,
    remove_mask: np.ndarray,
    out_path: Path,
) -> None:
    changed = add_mask | remove_mask
    if changed.any():
        coords = np.argwhere(changed)
        center = np.median(coords, axis=0).astype(int)
    else:
        center = np.array(original_high.shape) // 2

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    slice_specs = [
        ("X Slice", original_high[center[0]], repaired_high[center[0]], add_mask[center[0]], remove_mask[center[0]]),
        ("Y Slice", original_high[:, center[1], :], repaired_high[:, center[1], :], add_mask[:, center[1], :], remove_mask[:, center[1], :]),
        ("Z Slice", original_high[:, :, center[2]], repaired_high[:, :, center[2]], add_mask[:, :, center[2]], remove_mask[:, :, center[2]]),
    ]

    for col, (title, orig_sl, rep_sl, add_sl, rem_sl) in enumerate(slice_specs):
        overlay = np.zeros((*orig_sl.shape, 3), dtype=np.float32)
        overlay[..., :] = 0.08
        overlay[orig_sl > 0] = (0.75, 0.75, 0.75)
        overlay[rep_sl > 0] = np.maximum(overlay[rep_sl > 0], np.array([0.95, 0.95, 0.95]))
        overlay[add_sl > 0] = (0.90, 0.15, 0.15)
        overlay[rem_sl > 0] = (0.15, 0.35, 0.95)

        axes[0, col].imshow(orig_sl, cmap="gray")
        axes[0, col].set_title(f"{title}\nOriginal")
        axes[1, col].imshow(rep_sl, cmap="gray")
        axes[1, col].set_title("Remapped Repair")
        axes[2, col].imshow(overlay)
        axes[2, col].set_title("Difference Overlay")

        for row in range(3):
            axes[row, col].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_volume_changes(original_high: np.ndarray, repaired_high: np.ndarray, add_mask: np.ndarray) -> Dict[str, object]:
    orig_solid = int(original_high.sum())
    rep_solid = int(repaired_high.sum())
    added = int(add_mask.sum())
    return {
        "original_solid_voxels": orig_solid,
        "repaired_solid_voxels": rep_solid,
        "added_voxels": added,
        "removed_voxels": int((original_high & (~repaired_high)).sum()),
        "volume_ratio": float(rep_solid / max(orig_solid, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap low-res FEM repair to high-res and export visuals/OBJ")
    parser.add_argument("--original_high", required=True, help="Original high-res .npy")
    parser.add_argument("--repaired_low", required=True, help="Repaired low-res .npy")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["structural_preserving", "display_safe"],
        default="structural_preserving",
        help="High-res remap mode",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--proximity_voxels", type=int, default=4)
    parser.add_argument("--obj_voxel_size", type=float, default=1.0)
    args = parser.parse_args()

    original_high = _load_binary_volume(Path(args.original_high), threshold=float(args.threshold))
    repaired_low = _load_binary_volume(Path(args.repaired_low), threshold=float(args.threshold))

    result = remap_repaired_volume(
        original_high=original_high,
        repaired_low=repaired_low,
        mode=str(args.mode),
        proximity_voxels=int(args.proximity_voxels),
    )
    repaired_high = result["repaired_high"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repaired_npy = out_dir / "repaired_highres_density.npy"
    np.save(repaired_npy, repaired_high.astype(np.float32)[None, ...])

    diff_panel = out_dir / "voxel_difference_projection_panel.png"
    slice_panel = out_dir / "voxel_difference_slice_panel.png"
    save_difference_panel(
        original_high=original_high,
        repaired_high=repaired_high,
        add_mask=result["add_mask"],
        remove_mask=result["remove_mask"],
        out_path=diff_panel,
    )
    save_slice_panel(
        original_high=original_high,
        repaired_high=repaired_high,
        add_mask=result["add_mask"],
        remove_mask=result["remove_mask"],
        out_path=slice_panel,
    )

    silhouette_metrics = evaluate_silhouette_changes(original_high, repaired_high)
    volume_metrics = summarize_volume_changes(original_high, repaired_high, result["add_mask"])
    structural_consistency = {}
    if "downsampled_repaired_high" in result:
        repaired_low_back = result["downsampled_repaired_high"]
        structural_consistency = {
            "downsample_low_iou": float(result["low_consistency_iou"]),
            "low_add_voxels_preserved": int(result["low_add_voxels_preserved"]),
            "low_add_voxels_total": int(result["low_add_voxels_total"]),
            "low_add_preservation_rate": (
                float(result["low_add_voxels_preserved"] / max(result["low_add_voxels_total"], 1))
            ),
        }
    summary = {
        "mode": str(args.mode),
        "original_high_shape": list(original_high.shape),
        "repaired_low_shape": list(repaired_low.shape),
        "proximity_voxels": int(args.proximity_voxels),
        "volume_metrics": volume_metrics,
        "silhouette_metrics": silhouette_metrics,
        "structural_consistency": structural_consistency,
        "artifacts": {
            "repaired_highres_density": str(repaired_npy),
            "difference_projection_panel": str(diff_panel),
            "difference_slice_panel": str(slice_panel),
        },
    }

    json_path = out_dir / "remap_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    obj_path = out_dir / "repaired_highres.obj"
    voxel_to_obj(
        repaired_high.astype(np.float32),
        str(obj_path),
        threshold=0.5,
        voxel_size=float(args.obj_voxel_size),
        smooth_iterations=3,
        fill_internal_holes=True,
        keep_largest_component=True,
    )
    print(f"Saved remapped high-res volume -> {repaired_npy}")
    print(f"Saved difference panels -> {diff_panel}, {slice_panel}")
    print(f"Saved silhouette/volume summary -> {json_path}")
    print(f"Saved OBJ -> {obj_path}")


if __name__ == "__main__":
    main()
