"""
Optimize a structurally safe high-resolution remap of a low-resolution FEM repair.

This script targets the actual project export goal:
1. Preserve the structural repair under the same 256->64 downsampling used by FEM verification.
2. Keep orthographic silhouette changes as small as possible.
3. Export a single-shell OBJ suitable for presentation / downstream use.
4. Export voxel-difference visualizations and low-resolution stress analysis summaries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np
from scipy.ndimage import distance_transform_edt, label, zoom

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from structural.analysis.fem_solver import quick_sf_check
from structural.export.remap_fem_repair_to_highres import (
    _downsample_like_fem_verify,
    _load_binary_volume,
    evaluate_silhouette_changes,
    save_difference_panel,
    save_slice_panel,
    summarize_volume_changes,
)
from structural.export.voxel_to_mesh import voxel_to_obj


def _search_keep_counts(total: int) -> List[int]:
    seeds = [1, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 64, 80, 96, 128, 160, 192, 256, total]
    return sorted({min(total, x) for x in seeds if 0 < x <= total})


def _candidate_counts_around(anchor: int, total: int, radius: int = 12) -> List[int]:
    dense = list(range(max(1, anchor - radius), min(total, anchor + radius) + 1))
    return sorted(set(dense + _search_keep_counts(total)))


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
        before_img = _metric_mip(before_vis, axis)
        after_img = _metric_mip(after_vis, axis)
        reduction_img = _metric_mip(reduction, axis)

        axes[0, col].imshow(before_img, cmap=cmap, vmin=0.0 if not log_scale else None, vmax=vmax if not log_scale else None)
        axes[0, col].set_title(f"{title}\nBefore")
        axes[1, col].imshow(after_img, cmap=cmap, vmin=0.0 if not log_scale else None, vmax=vmax if not log_scale else None)
        axes[1, col].set_title("After")
        axes[2, col].imshow(reduction_img, cmap="coolwarm", vmin=-rvmax, vmax=rvmax)
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


def optimize_low_repair_subset(
    original_low_density: np.ndarray,
    repaired_low_binary: np.ndarray,
    repaired_vm: np.ndarray,
    target_sf: float,
) -> Dict[str, object]:
    original_low_density = np.clip(original_low_density.astype(np.float32), 0.0, 1.0)
    original_low_binary = original_low_density > 0.5
    added_coords = np.argwhere(repaired_low_binary & (~original_low_binary))

    fem_before = quick_sf_check(original_low_density.astype(np.float32), verbose=False)
    if len(added_coords) == 0:
        return {
            "optimized_low_density": original_low_density.copy(),
            "optimized_low_binary": original_low_binary.copy(),
            "selected_coords": np.empty((0, 3), dtype=int),
            "selected_count": 0,
            "full_added_count": 0,
            "search_log": [],
            "fem_before": fem_before,
            "fem_after": fem_before,
            "ranked_coords": np.empty((0, 3), dtype=int),
        }

    weights = repaired_vm[tuple(added_coords.T)]
    ranked_coords = added_coords[np.argsort(weights)[::-1]]
    cache: Dict[int, Dict[str, object]] = {}

    def eval_keep(n: int) -> Dict[str, object]:
        n = int(max(0, min(n, len(ranked_coords))))
        if n in cache:
            return cache[n]
        low = original_low_density.copy()
        if n > 0:
            low[tuple(ranked_coords[:n].T)] = 1.0
        fem = quick_sf_check(low.astype(np.float32), verbose=False)
        cache[n] = {
            "count": n,
            "sf": float(fem["sf_min"]),
            "density": low,
            "binary": low > 0.5,
            "fem": fem,
        }
        return cache[n]

    seed_counts = _search_keep_counts(len(ranked_coords))
    for n in seed_counts:
        eval_keep(n)

    feasible = [cache[n] for n in seed_counts if cache[n]["sf"] >= target_sf]
    if feasible:
        upper = min(item["count"] for item in feasible)
        lower = max(1, upper - 12)
        for n in range(lower, upper):
            eval_keep(n)
    else:
        upper = len(ranked_coords)

    feasible_exact = [item for item in cache.values() if item["sf"] >= target_sf]
    chosen = min(feasible_exact, key=lambda item: item["count"]) if feasible_exact else cache[upper]

    return {
        "optimized_low_density": chosen["density"],
        "optimized_low_binary": chosen["binary"],
        "selected_coords": ranked_coords[: chosen["count"]],
        "selected_count": int(chosen["count"]),
        "full_added_count": int(len(added_coords)),
        "search_log": [{"count": int(v["count"]), "sf": float(v["sf"])} for v in sorted(cache.values(), key=lambda x: x["count"])],
        "fem_before": fem_before,
        "fem_after": chosen["fem"],
        "ranked_coords": ranked_coords,
    }


def _axis_interp(high_size: int, low_size: int) -> List[List[int]]:
    coords = np.linspace(0, high_size - 1, low_size)
    interp: List[List[int]] = []
    for t in coords:
        lo = int(np.floor(t))
        hi = min(high_size - 1, int(np.ceil(t)))
        if lo == hi:
            interp.append([lo])
        else:
            interp.append(sorted(set([lo, hi])))
    return interp


def _selected_low_binary(
    original_low_binary: np.ndarray,
    ranked_coords: np.ndarray,
    keep_n: int,
) -> np.ndarray:
    low = original_low_binary.copy()
    if keep_n > 0:
        low[tuple(ranked_coords[:keep_n].T)] = True
    return low


def _connect_components_to_original(
    high_volume: np.ndarray,
    original_high: np.ndarray,
    allowed_low: np.ndarray,
) -> np.ndarray:
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, n_components = label(high_volume, structure=structure)
    if n_components <= 1:
        return high_volume

    nearest = distance_transform_edt(~original_high, return_indices=True)[1]
    ratio = high_volume.shape[0] // allowed_low.shape[0]
    connected = high_volume.copy()

    original_points = np.argwhere(original_high)
    if len(original_points) == 0:
        return high_volume

    original_component = None
    original_seed = tuple(original_points[0])
    if labels[original_seed] > 0:
        original_component = int(labels[original_seed])

    for comp_idx in range(1, n_components + 1):
        if original_component is not None and comp_idx == original_component:
            continue
        component_pts = np.argwhere(labels == comp_idx)
        if len(component_pts) == 0:
            continue
        if np.any(original_high[tuple(component_pts.T)]):
            continue

        nearest_pts = nearest[:, component_pts[:, 0], component_pts[:, 1], component_pts[:, 2]].T
        d2 = ((component_pts - nearest_pts) ** 2).sum(axis=1)
        p = component_pts[np.argmin(d2)]
        q = nearest[:, p[0], p[1], p[2]]

        steps = int(np.max(np.abs(q - p))) + 1
        xs = np.round(np.linspace(p[0], q[0], steps)).astype(int)
        ys = np.round(np.linspace(p[1], q[1], steps)).astype(int)
        zs = np.round(np.linspace(p[2], q[2], steps)).astype(int)
        for x, y, z in zip(xs, ys, zs):
            li = min(allowed_low.shape[0] - 1, max(0, x // ratio))
            lj = min(allowed_low.shape[1] - 1, max(0, y // ratio))
            lk = min(allowed_low.shape[2] - 1, max(0, z // ratio))
            if allowed_low[li, lj, lk]:
                connected[x, y, z] = True
    return connected


def build_8corner_connected_high_remap(
    original_high: np.ndarray,
    optimized_low_binary: np.ndarray,
    selected_coords: np.ndarray,
) -> Dict[str, np.ndarray]:
    interp_x = _axis_interp(original_high.shape[0], optimized_low_binary.shape[0])
    interp_y = _axis_interp(original_high.shape[1], optimized_low_binary.shape[1])
    interp_z = _axis_interp(original_high.shape[2], optimized_low_binary.shape[2])

    high = original_high.copy()
    for coord in selected_coords:
        i, j, k = (int(coord[0]), int(coord[1]), int(coord[2]))
        for x in interp_x[i]:
            for y in interp_y[j]:
                for z in interp_z[k]:
                    high[x, y, z] = True

    high = _connect_components_to_original(
        high_volume=high,
        original_high=original_high,
        allowed_low=optimized_low_binary,
    )

    back_low_density = zoom(
        high.astype(np.float32),
        tuple(t / s for s, t in zip(high.shape, optimized_low_binary.shape)),
        order=1,
    )
    back_low_density = np.clip(back_low_density, 0.0, 1.0)
    back_low_binary = back_low_density > 0.5
    add_mask = high & (~original_high)
    remove_mask = original_high & (~high)

    return {
        "high": high,
        "back_low_density": back_low_density,
        "back_low_binary": back_low_binary,
        "add_mask": add_mask,
        "remove_mask": remove_mask,
    }


def _changed_pixel_count(silhouette_metrics: Dict[str, object]) -> int:
    return int(sum(view["changed_pixels"] for view in silhouette_metrics["per_view"].values()))


def _quick_high_candidate_eval(
    original_high: np.ndarray,
    original_low_binary: np.ndarray,
    ranked_coords: np.ndarray,
    keep_n: int,
    target_sf: float,
) -> Dict[str, object]:
    optimized_low_binary = _selected_low_binary(original_low_binary, ranked_coords, keep_n)
    selected_coords = ranked_coords[:keep_n]
    remap = build_8corner_connected_high_remap(
        original_high=original_high,
        optimized_low_binary=optimized_low_binary,
        selected_coords=selected_coords,
    )
    fem = quick_sf_check(remap["back_low_density"].astype(np.float32), verbose=False)
    silhouette = evaluate_silhouette_changes(original_high, remap["high"])
    volume = summarize_volume_changes(original_high, remap["high"], remap["add_mask"])
    consistency = {
        "downsample_low_iou": float(
            np.logical_and(remap["back_low_binary"], optimized_low_binary).sum()
            / max(np.logical_or(remap["back_low_binary"], optimized_low_binary).sum(), 1)
        ),
        "missing_low_voxels": int((optimized_low_binary & (~remap["back_low_binary"])).sum()),
        "extra_low_voxels": int((remap["back_low_binary"] & (~optimized_low_binary)).sum()),
    }
    return {
        "count": int(keep_n),
        "sf": float(fem["sf_min"]),
        "fem": fem,
        "optimized_low_binary": optimized_low_binary,
        "selected_coords": selected_coords,
        "remap": remap,
        "silhouette": silhouette,
        "volume": volume,
        "consistency": consistency,
        "changed_pixels": _changed_pixel_count(silhouette),
        "added_voxels": int(remap["add_mask"].sum()),
        "safe": bool(fem["sf_min"] >= target_sf),
    }


def select_best_high_candidate(
    original_high: np.ndarray,
    original_low_binary: np.ndarray,
    ranked_coords: np.ndarray,
    low_safe_count: int,
    target_sf: float,
) -> Dict[str, object]:
    total = len(ranked_coords)
    if total == 0:
        return _quick_high_candidate_eval(
            original_high=original_high,
            original_low_binary=original_low_binary,
            ranked_coords=ranked_coords,
            keep_n=0,
            target_sf=target_sf,
        )

    tried: Dict[int, Dict[str, object]] = {}
    search_band = _candidate_counts_around(low_safe_count, total, radius=12)
    current_max = max(search_band) if search_band else low_safe_count

    while True:
        for n in search_band:
            if n not in tried:
                tried[n] = _quick_high_candidate_eval(
                    original_high=original_high,
                    original_low_binary=original_low_binary,
                    ranked_coords=ranked_coords,
                    keep_n=n,
                    target_sf=target_sf,
                )

        safe = [item for item in tried.values() if item["safe"]]
        if safe:
            return min(safe, key=lambda item: (item["changed_pixels"], item["added_voxels"], item["count"]))

        if current_max >= total:
            return max(tried.values(), key=lambda item: item["sf"])

        next_max = min(total, current_max + 12)
        search_band = list(range(current_max + 1, next_max + 1))
        current_max = next_max


def _mesh_part_count(obj_path: Path) -> int:
    try:
        import trimesh
    except Exception:
        return -1
    mesh = trimesh.load_mesh(obj_path, process=False)
    return len(mesh.split(only_watertight=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize a structure-safe high-res remap with minimal silhouette change"
    )
    parser.add_argument("--original_high", required=True)
    parser.add_argument("--repaired_low", required=True)
    parser.add_argument("--repaired_vm", default=None, help="Optional FEM von Mises field for the repaired low-res volume")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target_sf", type=float, default=1.5)
    parser.add_argument("--obj_voxel_size", type=float, default=1.0)
    args = parser.parse_args()

    original_high = _load_binary_volume(Path(args.original_high), threshold=float(args.threshold))
    repaired_low = _load_binary_volume(Path(args.repaired_low), threshold=float(args.threshold))
    original_low_density = zoom(
        original_high.astype(np.float32),
        tuple(t / s for s, t in zip(original_high.shape, repaired_low.shape)),
        order=1,
    )
    original_low_density = np.clip(original_low_density, 0.0, 1.0)
    original_low_binary = original_low_density > 0.5

    if args.repaired_vm:
        repaired_vm = np.load(args.repaired_vm)
    else:
        repaired_vm = quick_sf_check(repaired_low.astype(np.float32), verbose=False)["von_mises"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    low_opt = optimize_low_repair_subset(
        original_low_density=original_low_density,
        repaired_low_binary=repaired_low,
        repaired_vm=repaired_vm,
        target_sf=float(args.target_sf),
    )

    np.save(out_dir / "optimized_low_density.npy", low_opt["optimized_low_density"].astype(np.float32))
    np.save(out_dir / "optimized_low_binary.npy", low_opt["optimized_low_binary"].astype(np.uint8))

    best_high = select_best_high_candidate(
        original_high=original_high,
        original_low_binary=original_low_binary,
        ranked_coords=low_opt["ranked_coords"],
        low_safe_count=int(low_opt["selected_count"]),
        target_sf=float(args.target_sf),
    )

    repaired_high = best_high["remap"]["high"]
    repaired_npy = out_dir / "repaired_highres_density.npy"
    np.save(repaired_npy, repaired_high.astype(np.float32)[None, ...])

    diff_panel = out_dir / "voxel_difference_projection_panel.png"
    slice_panel = out_dir / "voxel_difference_slice_panel.png"
    save_difference_panel(
        original_high=original_high,
        repaired_high=repaired_high,
        add_mask=best_high["remap"]["add_mask"],
        remove_mask=best_high["remap"]["remove_mask"],
        out_path=diff_panel,
    )
    save_slice_panel(
        original_high=original_high,
        repaired_high=repaired_high,
        add_mask=best_high["remap"]["add_mask"],
        remove_mask=best_high["remap"]["remove_mask"],
        out_path=slice_panel,
    )

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
    mesh_parts = _mesh_part_count(obj_path)

    np.save(out_dir / "fem_vm_before.npy", low_opt["fem_before"]["von_mises"])
    np.save(out_dir / "fem_vm_after_quick.npy", best_high["fem"]["von_mises"])
    before_disp_mag = np.linalg.norm(low_opt["fem_before"]["displacement"], axis=-1)
    after_disp_mag = np.linalg.norm(best_high["fem"]["displacement"], axis=-1)
    np.save(out_dir / "fem_disp_mag_before.npy", before_disp_mag)
    np.save(out_dir / "fem_disp_mag_after_quick.npy", after_disp_mag)

    _save_scalar_field_comparison(
        before_field=low_opt["fem_before"]["von_mises"],
        after_field=best_high["fem"]["von_mises"],
        out_path=out_dir / "stress_vm_comparison_quick.png",
        title_prefix="Von Mises Stress Comparison (quick FEM @64^3)",
        cmap="inferno",
        log_scale=True,
    )
    _save_scalar_field_comparison(
        before_field=before_disp_mag,
        after_field=after_disp_mag,
        out_path=out_dir / "displacement_comparison_quick.png",
        title_prefix="Displacement Magnitude Comparison (quick FEM @64^3)",
        cmap="viridis",
        log_scale=False,
    )
    solid_before = low_opt["fem_before"]["von_mises"][original_low_density > 0.3]
    solid_after = best_high["fem"]["von_mises"][best_high["remap"]["back_low_density"] > 0.3]
    _save_histogram_comparison(
        solid_before,
        solid_after,
        out_dir / "stress_histogram_comparison_quick.png",
        title="Solid-Region Von Mises Stress Histogram",
        xlabel="Von Mises Stress (Pa)",
    )

    summary = {
        "target_sf": float(args.target_sf),
        "low_optimization": {
            "full_added_voxels": int(low_opt["full_added_count"]),
            "selected_added_voxels": int(low_opt["selected_count"]),
            "search_log": low_opt["search_log"],
        },
        "high_optimization": {
            "selected_added_voxels": int(best_high["count"]),
            "sf_after_quick": float(best_high["sf"]),
            "changed_pixels": int(best_high["changed_pixels"]),
            "added_high_voxels": int(best_high["added_voxels"]),
        },
        "fem_metrics_quick": {
            "sf_before": float(low_opt["fem_before"]["sf_min"]),
            "sf_after": float(best_high["fem"]["sf_min"]),
            "vm_max_before": float(low_opt["fem_before"]["von_mises"][original_low_density > 0.3].max()),
            "vm_max_after": float(best_high["fem"]["von_mises"][best_high["remap"]["back_low_density"] > 0.3].max()),
            "disp_max_before": float(np.abs(low_opt["fem_before"]["displacement"]).max()),
            "disp_max_after": float(np.abs(best_high["fem"]["displacement"]).max()),
        },
        "volume_metrics": best_high["volume"],
        "silhouette_metrics": best_high["silhouette"],
        "structural_consistency": best_high["consistency"],
        "mesh": {
            "obj_path": str(obj_path),
            "mesh_parts": int(mesh_parts),
        },
        "artifacts": {
            "optimized_low_density": str(out_dir / "optimized_low_density.npy"),
            "optimized_low_binary": str(out_dir / "optimized_low_binary.npy"),
            "repaired_highres_density": str(repaired_npy),
            "difference_projection_panel": str(diff_panel),
            "difference_slice_panel": str(slice_panel),
            "stress_vm_comparison_quick": str(out_dir / "stress_vm_comparison_quick.png"),
            "displacement_comparison_quick": str(out_dir / "displacement_comparison_quick.png"),
            "stress_histogram_comparison_quick": str(out_dir / "stress_histogram_comparison_quick.png"),
        },
    }

    with open(out_dir / "optimization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
