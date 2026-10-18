"""
Scan FEM datasets and summarize structural target distributions.

This utility is the first step of the "hard-case data" plan: confirm whether
the current FEM training corpora are dominated by overly safe samples before
augmenting near-failure cases.

Outputs:
  - JSON summary with SF / vm_max / solid_ratio distribution stats
  - Optional CSV with per-sample records

Usage:
    python structural/training/analyze_fem_dataset_distribution.py \
        --data_dirs fem_dataset fem_dataset_v2 fem_dataset_v3 \
        --output_json results/fem_dataset_distribution.json \
        --output_csv results/fem_dataset_distribution.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _safe_stats(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _sf_bins(sf_values: np.ndarray) -> dict:
    return {
        "lt_1.0": int((sf_values < 1.0).sum()),
        "lt_1.5": int((sf_values < 1.5).sum()),
        "lt_2.0": int((sf_values < 2.0).sum()),
        "lt_5.0": int((sf_values < 5.0).sum()),
        "lt_10.0": int((sf_values < 10.0).sum()),
        "gt_50.0": int((sf_values > 50.0).sum()),
        "gt_100.0": int((sf_values > 100.0).sum()),
    }


def _occupied_bbox(density: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int]:
    occ = np.argwhere(density > threshold)
    if occ.size == 0:
        return (0, 0, 0)
    mins = occ.min(axis=0)
    maxs = occ.max(axis=0)
    spans = (maxs - mins + 1).astype(int)
    return int(spans[0]), int(spans[1]), int(spans[2])


def _shape_proxies(density: np.ndarray) -> tuple[float, float]:
    sx, sy, sz = _occupied_bbox(density)
    spans = np.asarray([sx, sy, sz], dtype=np.float64)
    nonzero = spans[spans > 0]
    if nonzero.size == 0:
        return 0.0, 0.0
    thin_shell_proxy = float(nonzero.min() / max(nonzero.max(), 1.0))
    sorted_spans = np.sort(nonzero)
    if sorted_spans.size == 1:
        slenderness_proxy = 1.0
    else:
        slenderness_proxy = float(sorted_spans[-1] / max(sorted_spans[-2], 1.0))
    return thin_shell_proxy, slenderness_proxy


def maybe_write_plots(rows: list[dict], plot_dir: Path | None):
    if plot_dir is None:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    sf = np.asarray([r["sf_gt"] for r in rows], dtype=np.float64) if rows else np.array([])
    vm = np.asarray([r["vm_max"] for r in rows], dtype=np.float64) if rows else np.array([])
    solid = np.asarray([r["solid_ratio"] for r in rows], dtype=np.float64) if rows else np.array([])
    if sf.size:
        plt.figure(figsize=(7, 4))
        plt.hist(sf, bins=40, color="#2a6fbb", alpha=0.9)
        plt.xlabel("SF_gt")
        plt.ylabel("Count")
        plt.title("FEM Safety Factor Distribution")
        plt.tight_layout()
        plt.savefig(plot_dir / "fem_sf_hist.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.hist(np.log10(np.clip(sf, 1e-8, None)), bins=40, color="#1f9d55", alpha=0.9)
        plt.xlabel("log10(SF_gt)")
        plt.ylabel("Count")
        plt.title("FEM Safety Factor Distribution (log scale)")
        plt.tight_layout()
        plt.savefig(plot_dir / "fem_sf_hist_log10.png", dpi=180)
        plt.close()

    if vm.size:
        plt.figure(figsize=(7, 4))
        plt.hist(vm, bins=40, color="#b45309", alpha=0.9)
        plt.xlabel("vm_max")
        plt.ylabel("Count")
        plt.title("Von Mises Max Distribution")
        plt.tight_layout()
        plt.savefig(plot_dir / "fem_vm_hist.png", dpi=180)
        plt.close()

    if solid.size:
        plt.figure(figsize=(7, 4))
        plt.hist(solid, bins=40, color="#7c3aed", alpha=0.9)
        plt.xlabel("solid_ratio")
        plt.ylabel("Count")
        plt.title("Solid Ratio Distribution")
        plt.tight_layout()
        plt.savefig(plot_dir / "fem_solid_ratio_hist.png", dpi=180)
        plt.close()


def scan_dataset(data_dirs: list[str], yield_stress: float) -> tuple[list[dict], dict]:
    rows: list[dict] = []

    for data_dir in data_dirs:
        root = Path(data_dir)
        if not root.exists():
            continue

        files = sorted(root.glob("sample_*.npz"))
        for path in files:
            data = np.load(path)
            density = data["density"].astype(np.float32)
            stress = data["stress"].astype(np.float32)
            solid_ratio = float((density > 0.5).mean())
            thin_shell_proxy, slenderness_proxy = _shape_proxies(density)

            if "von_mises" in data:
                vm = data["von_mises"].astype(np.float32)
            else:
                sxx, syy, szz, syz, szx, sxy = stress
                vm = np.sqrt(
                    0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                    + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2)
                ).astype(np.float32)

            solid_mask = density > 0.1
            if solid_mask.any():
                vm_max = float(vm[solid_mask].max())
                vm_mean = float(vm[solid_mask].mean())
            else:
                vm_max = 0.0
                vm_mean = 0.0

            sf_gt = float(yield_stress / max(vm_max, 1e-12))
            rows.append(
                {
                    "dataset": root.name,
                    "file": path.name,
                    "solid_ratio": solid_ratio,
                    "vm_max": vm_max,
                    "vm_mean": vm_mean,
                    "sf_gt": sf_gt,
                    "thin_shell_proxy": thin_shell_proxy,
                    "slenderness_proxy": slenderness_proxy,
                    "near_failure": bool(sf_gt < 5.0),
                    "sf_bin": (
                        "lt_1.5" if sf_gt < 1.5 else
                        "lt_5" if sf_gt < 5.0 else
                        "lt_10" if sf_gt < 10.0 else
                        "safe"
                    ),
                }
            )

    sf_vals = np.asarray([r["sf_gt"] for r in rows], dtype=np.float64) if rows else np.array([])
    vm_vals = [r["vm_max"] for r in rows]
    solid_vals = [r["solid_ratio"] for r in rows]
    thin_vals = [r["thin_shell_proxy"] for r in rows]
    slender_vals = [r["slenderness_proxy"] for r in rows]

    summary = {
        "n_samples": len(rows),
        "yield_stress": yield_stress,
        "sf_gt": _safe_stats(sf_vals.tolist()),
        "vm_max": _safe_stats(vm_vals),
        "solid_ratio": _safe_stats(solid_vals),
        "thin_shell_proxy": _safe_stats(thin_vals),
        "slenderness_proxy": _safe_stats(slender_vals),
        "sf_bins": _sf_bins(sf_vals) if sf_vals.size else {},
        "near_failure_rate": float((sf_vals < 5.0).mean()) if sf_vals.size else 0.0,
        "datasets": {},
    }

    for dataset_name in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset_name]
        subset_sf = np.asarray([r["sf_gt"] for r in subset], dtype=np.float64)
        summary["datasets"][dataset_name] = {
            "n_samples": len(subset),
            "sf_gt": _safe_stats(subset_sf.tolist()),
            "sf_bins": _sf_bins(subset_sf),
            "near_failure_rate": float((subset_sf < 5.0).mean()) if subset_sf.size else 0.0,
        }

    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--yield_stress", type=float, default=50e6)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--plot_dir", default=None)
    args = parser.parse_args()

    rows, summary = scan_dataset(args.data_dirs, args.yield_stress)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset", "file", "solid_ratio", "vm_max", "vm_mean", "sf_gt",
                    "thin_shell_proxy", "slenderness_proxy", "near_failure", "sf_bin",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    maybe_write_plots(rows, Path(args.plot_dir) if args.plot_dir else None)

    print(f"Scanned {summary['n_samples']} samples")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
