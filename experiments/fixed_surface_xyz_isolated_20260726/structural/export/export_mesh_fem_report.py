"""
Export a calibration/comparison report between experimental mesh-FEM and voxel-FEM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from structural.mesh_fem.calibration import calibrate_percentile_to_reference, sf_metrics


def _save_hist(mesh_vm: np.ndarray, voxel_vm: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    m = mesh_vm[mesh_vm > 0]
    v = voxel_vm[voxel_vm > 0]
    positive = np.concatenate([m, v]) if len(m) and len(v) else (m if len(m) else v)
    bins = 60
    if len(positive) > 0:
        lo = np.percentile(positive, 1)
        hi = np.percentile(positive, 99.5)
        if hi > lo > 0:
            bins = np.logspace(np.log10(lo), np.log10(hi), 60)
            ax.set_xscale("log")
    ax.hist(v, bins=bins, alpha=0.5, label="Voxel-FEM", color="#1b9e77")
    ax.hist(m, bins=bins, alpha=0.5, label="Mesh-FEM", color="#d95f02")
    ax.set_title("Von Mises Histogram: Voxel-FEM vs Mesh-FEM")
    ax.set_xlabel("Von Mises Stress (Pa)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_sf_bar(voxel_sf: float, mesh_metrics: dict, calib: dict, out_path: Path) -> None:
    labels = ["Voxel-FEM\n(reference)", "Mesh-FEM\n(raw max)", f"Mesh-FEM\ncalib p={calib['best_percentile']:.2f}"]
    values = [voxel_sf, mesh_metrics["sf_max"], calib["best_sf"]]
    colors = ["#1b9e77", "#d95f02", "#7570b3"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values, color=colors)
    ax.axhline(1.5, color="red", linestyle="--", linewidth=1.5, label="Safety threshold")
    ax.set_ylabel("Safety Factor")
    ax.set_title("Safety Factor Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _project_element_field(centroids: np.ndarray, values: np.ndarray, bounds: np.ndarray, bins: int = 160):
    mins = bounds[0]
    maxs = bounds[1]
    spans = np.maximum(maxs - mins, 1e-8)
    norm = (centroids - mins[None, :]) / spans[None, :]
    idx = np.clip(np.floor(norm * (bins - 1)).astype(int), 0, bins - 1)

    projections = {}
    axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}
    for name, (a, b) in axes.items():
        grid = np.zeros((bins, bins), dtype=np.float64)
        for i in range(len(values)):
            ia = idx[i, a]
            ib = idx[i, b]
            if values[i] > grid[ia, ib]:
                grid[ia, ib] = values[i]
        projections[name] = grid
    return projections


def _save_projection_panel(centroids: np.ndarray, values: np.ndarray, out_path: Path, title: str, cmap: str = "inferno") -> None:
    bounds = np.stack([centroids.min(axis=0), centroids.max(axis=0)], axis=0)
    proj = _project_element_field(centroids, values, bounds)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, grid) in zip(axes, proj.items()):
        vis = np.log10(np.maximum(grid, 1e-8)) if cmap == "inferno" else grid
        ax.imshow(vis, cmap=cmap)
        ax.set_title(f"{name.upper()} projection")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export mesh-FEM calibration/comparison report")
    parser.add_argument("--mesh_fem_dir", required=True)
    parser.add_argument("--voxel_fem_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--yield_stress", type=float, default=50.0e6)
    args = parser.parse_args()

    mesh_dir = Path(args.mesh_fem_dir)
    voxel_dir = Path(args.voxel_fem_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_summary = json.loads((mesh_dir / "mesh_fem_verification.json").read_text())
    voxel_summary = json.loads((voxel_dir / "fem_verification.json").read_text())
    mesh_vm = np.load(mesh_dir / "mesh_von_mises.npy")
    mesh_nodes = np.load(mesh_dir / "mesh_nodes.npy")
    mesh_elems = np.load(mesh_dir / "mesh_elems.npy")
    mesh_disp = np.load(mesh_dir / "mesh_displacement.npy")
    voxel_vm = np.load(voxel_dir / "fem_von_mises.npy")

    mesh_metrics = sf_metrics(mesh_vm, yield_stress=float(args.yield_stress))
    voxel_sf = float(voxel_summary["fem"]["sf_min"])
    calib = calibrate_percentile_to_reference(
        von_mises=mesh_vm,
        yield_stress=float(args.yield_stress),
        reference_sf=voxel_sf,
    )

    centroids = mesh_nodes[mesh_elems].mean(axis=1)
    elem_disp = np.linalg.norm(mesh_disp[mesh_elems].mean(axis=1), axis=1)

    _save_hist(mesh_vm, voxel_vm, out_dir / "voxel_vs_mesh_vm_histogram.png")
    _save_sf_bar(voxel_sf, mesh_metrics, calib, out_dir / "voxel_vs_mesh_sf_comparison.png")
    _save_projection_panel(centroids, mesh_vm, out_dir / "mesh_vm_projection_panel.png", "Mesh-FEM Von Mises Element Projection")
    _save_projection_panel(centroids, elem_disp, out_dir / "mesh_disp_projection_panel.png", "Mesh-FEM Displacement Magnitude Projection", cmap="viridis")

    summary = {
        "voxel_reference": voxel_summary["fem"],
        "mesh_raw": mesh_summary["fem"],
        "mesh_sf_metrics": mesh_metrics,
        "calibration": calib,
        "interpretation": {
            "note": "This calibration aligns the experimental mesh-FEM branch to the project's voxel-FEM SF convention via a percentile-based effective stress metric.",
            "recommended_effective_percentile": calib["best_percentile"],
        },
        "artifacts": {
            "histogram": str(out_dir / "voxel_vs_mesh_vm_histogram.png"),
            "sf_bar": str(out_dir / "voxel_vs_mesh_sf_comparison.png"),
            "mesh_vm_projection": str(out_dir / "mesh_vm_projection_panel.png"),
            "mesh_disp_projection": str(out_dir / "mesh_disp_projection_panel.png"),
        },
    }
    (out_dir / "mesh_fem_calibration_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
