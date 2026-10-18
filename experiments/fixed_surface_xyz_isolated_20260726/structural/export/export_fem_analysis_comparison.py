"""
Export before/after FEM comparison visualizations and summary statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _metric_mip(field: np.ndarray, axis: int) -> np.ndarray:
    return field.max(axis=axis)


def _save_scalar_field_comparison(
    before_field: np.ndarray,
    after_field: np.ndarray,
    out_path: Path,
    title_prefix: str,
    cmap: str = "inferno",
    log_scale: bool = False,
) -> None:
    if log_scale:
        before_vis = np.log10(np.maximum(before_field, 1e-8))
        after_vis = np.log10(np.maximum(after_field, 1e-8))
    else:
        before_vis = before_field
        after_vis = after_field

    reduction = before_field - after_field
    vmax = float(max(before_vis.max(), after_vis.max(), 1e-8))
    rvmax = float(max(np.abs(reduction).max(), 1e-8))

    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    axis_specs = [("X Projection", 0), ("Y Projection", 1), ("Z Projection", 2)]
    for col, (title, axis) in enumerate(axis_specs):
        b = _metric_mip(before_vis, axis)
        a = _metric_mip(after_vis, axis)
        r = _metric_mip(reduction, axis)
        axes[0, col].imshow(b, cmap=cmap, vmin=0.0 if not log_scale else None, vmax=vmax if not log_scale else None)
        axes[0, col].set_title(f"{title}\nBefore")
        axes[1, col].imshow(a, cmap=cmap, vmin=0.0 if not log_scale else None, vmax=vmax if not log_scale else None)
        axes[1, col].set_title("After")
        axes[2, col].imshow(r, cmap="coolwarm", vmin=-rvmax, vmax=rvmax)
        axes[2, col].set_title("Before - After")
        for row in range(3):
            axes[row, col].axis("off")

    fig.suptitle(title_prefix, fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def _save_histogram_comparison(
    before_vals: np.ndarray,
    after_vals: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    before_vals = np.asarray(before_vals).ravel()
    after_vals = np.asarray(after_vals).ravel()
    positive = np.concatenate([before_vals[before_vals > 0], after_vals[after_vals > 0]])
    bins = 60
    if len(positive) > 0:
        lo = np.percentile(positive, 1)
        hi = np.percentile(positive, 99.5)
        if hi > lo > 0:
            bins = np.logspace(np.log10(lo), np.log10(hi), 60)
            ax.set_xscale("log")
    ax.hist(before_vals[before_vals > 0], bins=bins, alpha=0.55, label="Before", color="#d95f02")
    ax.hist(after_vals[after_vals > 0], bins=bins, alpha=0.55, label="After", color="#1b9e77")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_hotspot_comparison(
    before_vm: np.ndarray,
    after_vm: np.ndarray,
    out_path: Path,
) -> dict:
    before_thr = float(np.percentile(before_vm[before_vm > 0], 99)) if np.any(before_vm > 0) else 0.0
    after_thr = float(np.percentile(after_vm[after_vm > 0], 99)) if np.any(after_vm > 0) else 0.0
    before_hot = before_vm >= before_thr
    after_hot = after_vm >= after_thr
    reduced_hot = before_hot & (~after_hot)
    persistent_hot = before_hot & after_hot

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axis_specs = [("X Projection", 0), ("Y Projection", 1), ("Z Projection", 2)]
    for col, (title, axis) in enumerate(axis_specs):
        b = _metric_mip(before_hot.astype(np.uint8), axis)
        a = _metric_mip(after_hot.astype(np.uint8), axis)
        overlay = np.zeros((*b.shape, 3), dtype=np.float32)
        overlay[_metric_mip(reduced_hot.astype(np.uint8), axis) > 0] = (0.12, 0.45, 0.95)
        overlay[_metric_mip(persistent_hot.astype(np.uint8), axis) > 0] = (0.88, 0.20, 0.15)
        overlay[_metric_mip(after_hot.astype(np.uint8), axis) > 0] = np.maximum(
            overlay[_metric_mip(after_hot.astype(np.uint8), axis) > 0],
            np.array([0.15, 0.75, 0.25]),
        )
        axes[0, col].imshow(b, cmap="magma")
        axes[0, col].set_title(f"{title}\nBefore hotspots")
        axes[1, col].imshow(overlay)
        axes[1, col].set_title("Overlay\nblue=reduced, red=persistent, green=after")
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "before_hotspot_threshold": before_thr,
        "after_hotspot_threshold": after_thr,
        "before_hotspot_voxels": int(before_hot.sum()),
        "after_hotspot_voxels": int(after_hot.sum()),
        "reduced_hotspot_voxels": int(reduced_hot.sum()),
        "persistent_hotspot_voxels": int(persistent_hot.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export before/after FEM analysis comparison")
    parser.add_argument("--before_dir", required=True, help="Directory containing original fem_verification.json, fem_von_mises.npy, fem_displacement.npy")
    parser.add_argument("--after_dir", required=True, help="Directory containing repaired fem_verification.json, fem_von_mises.npy, fem_displacement.npy")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    before_dir = Path(args.before_dir)
    after_dir = Path(args.after_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    before_json = json.loads((before_dir / "fem_verification.json").read_text())
    after_json = json.loads((after_dir / "fem_verification.json").read_text())
    before_vm = np.load(before_dir / "fem_von_mises.npy")
    after_vm = np.load(after_dir / "fem_von_mises.npy")
    before_disp = np.load(before_dir / "fem_displacement.npy")
    after_disp = np.load(after_dir / "fem_displacement.npy")
    before_disp_mag = np.linalg.norm(before_disp, axis=-1)
    after_disp_mag = np.linalg.norm(after_disp, axis=-1)

    _save_scalar_field_comparison(
        before_field=before_vm,
        after_field=after_vm,
        out_path=out_dir / "stress_vm_comparison_formal.png",
        title_prefix="Von Mises Stress Comparison (formal FEM verification)",
        cmap="inferno",
        log_scale=True,
    )
    _save_scalar_field_comparison(
        before_field=before_disp_mag,
        after_field=after_disp_mag,
        out_path=out_dir / "displacement_comparison_formal.png",
        title_prefix="Displacement Magnitude Comparison (formal FEM verification)",
        cmap="viridis",
        log_scale=False,
    )
    _save_histogram_comparison(
        before_vm[before_vm > 0],
        after_vm[after_vm > 0],
        out_dir / "stress_histogram_comparison_formal.png",
        title="Von Mises Stress Histogram",
        xlabel="Von Mises Stress (Pa)",
    )
    hotspot_summary = _save_hotspot_comparison(
        before_vm=before_vm,
        after_vm=after_vm,
        out_path=out_dir / "stress_hotspot_comparison_formal.png",
    )

    summary = {
        "before": before_json["fem"],
        "after": after_json["fem"],
        "improvement": {
            "sf_gain": float(after_json["fem"]["sf_min"] / max(before_json["fem"]["sf_min"], 1e-8)),
            "vm_max_ratio": float(after_json["fem"]["vm_max"] / max(before_json["fem"]["vm_max"], 1e-8)),
            "disp_max_ratio": float(after_json["fem"]["disp_max"] / max(before_json["fem"]["disp_max"], 1e-8)),
        },
        "hotspot_summary": hotspot_summary,
        "artifacts": {
            "stress_vm_comparison_formal": str(out_dir / "stress_vm_comparison_formal.png"),
            "displacement_comparison_formal": str(out_dir / "displacement_comparison_formal.png"),
            "stress_histogram_comparison_formal": str(out_dir / "stress_histogram_comparison_formal.png"),
            "stress_hotspot_comparison_formal": str(out_dir / "stress_hotspot_comparison_formal.png"),
        },
    }

    (out_dir / "fem_comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
