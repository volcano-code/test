#!/usr/bin/env python3
"""
Run baseline methods and attach true voxel-FEM verification results.

This is a heavier evaluation path than results/baselines_fixed because it
re-runs each baseline to recover the optimized density, then verifies the final
voxel with the project FEM solver. The output schema is intentionally close to
run_all_baselines.py so that paper aggregation can merge it transparently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "structural"))

from run_all_baselines import ALL_METHODS, YIELD_STRESS, evaluate_metrics, load_pino, run_method
from structural.analysis.fem_solver import voxel_fem_solve
from structural.analysis.load_generator import LoadGenerator

BASELINE_FEM_SCHEMA_VERSION = "baseline_fem_backed_v2"


def ensure_resolution(arr: np.ndarray, target_resolution: int) -> np.ndarray:
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 5:
        arr = arr[0, 0]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] == target_resolution:
        return np.clip(arr, 0.0, 1.0)
    scale = target_resolution / arr.shape[0]
    return np.clip(zoom(arr, scale, order=1), 0.0, 1.0)


def load_force_and_support(model_dir: Path, density_orig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    loads_file = model_dir / "loads.npz"
    if loads_file.exists():
        loads = np.load(loads_file)
        return loads["force_standing"].astype(np.float32), loads["fixed_mask_standing"].astype(np.float32)
    ld = LoadGenerator().standing_load(density_orig)
    return ld["force"].astype(np.float32), ld["fixed_mask"].astype(np.float32)


def fem_verify_density(
    density: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    yield_stress: float,
) -> dict:
    fem = voxel_fem_solve(
        density=density,
        force_field=force,
        fixed_mask=fixed_mask,
        verbose=False,
    )
    solid_mask = density > 0.3
    if solid_mask.any():
        vm_solid = fem["von_mises"][solid_mask]
        vm_max = float(vm_solid.max())
        vm_mean = float(vm_solid.mean())
        sf = float(yield_stress / max(vm_max, 1e-12))
    else:
        vm_max = 0.0
        vm_mean = 0.0
        sf = float("inf")
    return {
        "sf_fem": sf,
        "vm_max_fem": vm_max,
        "vm_mean_fem": vm_mean,
        "disp_max_fem": float(np.abs(fem["displacement"]).max()),
        "fem_solve_time_s": float(fem["total_time_s"]),
        "fem_n_elements": int(fem["n_elements"]),
        "fem_safe": bool(sf >= 1.5),
    }


def save_density(output_dir: Path, dataset_name: str, method: str, model_id: str, density: np.ndarray):
    out = output_dir / "method_outputs" / dataset_name / method / model_id
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "density.npy", density.astype(np.float32))


def classify_failure(row: dict) -> str:
    if row.get("constraint_satisfied", False):
        return ""
    if not row.get("fem_safe", False):
        return "fem_unsafe"
    if float(row.get("vol_ratio", 999.0)) > 1.3:
        return "volume_budget_exceeded"
    if not row.get("constraint_satisfied_pino", True):
        return "surrogate_constraint_unsatisfied"
    return "constraint_unsatisfied"


def summarize_results(results: dict[str, list[dict]]) -> dict:
    summary = {}
    for method, rows in results.items():
        valid = [r for r in rows if "sf_fem" in r]
        summary[method] = {
            "n_rows": len(rows),
            "n_valid": len(valid),
            "n_failures": sum(1 for r in rows if r.get("failure_reason")),
            "paper_eligible": sum(
                1
                for r in valid
                if bool(r.get("constraint_satisfied", False))
                and bool(r.get("fem_safe", False))
            ),
            "mean_sf_fem": float(np.mean([r["sf_fem"] for r in valid])) if valid else None,
            "mean_iou_3d": float(np.mean([r["iou_3d"] for r in valid if "iou_3d" in r])) if valid else None,
            "mean_vol_ratio": float(np.mean([r["vol_ratio"] for r in valid if "vol_ratio" in r])) if valid else None,
        }
    return summary


def run_dataset(
    dataset_dir: Path,
    methods: list[str],
    pino_model,
    device: torch.device,
    output_dir: Path,
    n_steps: int,
    limit_models: int | None,
    start_index: int,
    target_resolution: int,
    yield_stress: float,
    save_outputs: bool,
) -> dict[str, list[dict]]:
    meta = json.loads((dataset_dir / "dataset_meta.json").read_text())
    source = meta.get("source", dataset_dir.name)
    models = meta.get("models", [])
    if start_index:
        models = models[start_index:]
    if limit_models is not None:
        models = models[:limit_models]

    all_results = {m: [] for m in methods}
    for idx, m_info in enumerate(models):
        model_id = m_info["model_id"]
        model_dir = dataset_dir / model_id
        voxel_file = model_dir / "voxels.npy"
        if not voxel_file.exists():
            continue

        density_orig = ensure_resolution(np.load(voxel_file), target_resolution)
        force, fixed_mask = load_force_and_support(model_dir, density_orig)
        print(f"[{source} {idx+1}/{len(models)}] {model_id}")

        for method in methods:
            t0 = time.time()
            density_out, elapsed_baseline = run_method(
                method=method,
                density_orig=density_orig,
                force=force,
                fixed_mask=fixed_mask,
                pino_model=pino_model,
                device=device,
                n_steps=n_steps,
            )
            density_out = ensure_resolution(density_out, target_resolution)
            if save_outputs:
                save_density(output_dir, source, method, model_id, density_out)

            pino_metrics = evaluate_metrics(density_out, density_orig, force, pino_model, device)
            fem_metrics = fem_verify_density(density_out, force, fixed_mask, yield_stress)
            total_elapsed = time.time() - t0

            row = {
                **pino_metrics,
                **fem_metrics,
                "constraint_satisfied_pino": bool(pino_metrics.get("constraint_satisfied", False)),
                "constraint_satisfied": bool(
                    fem_metrics["fem_safe"] and float(pino_metrics.get("vol_ratio", 999.0)) <= 1.3
                ),
                "model_id": model_id,
                "category": m_info.get("category", "unknown"),
                "source": source,
                "method": method,
                "elapsed_baseline_s": float(elapsed_baseline),
                "elapsed_s": float(total_elapsed),
                "pino_checkpoint": getattr(pino_model, "_checkpoint_path", None),
                "n_steps": int(n_steps),
                "target_resolution": int(target_resolution),
                "yield_stress_physical": float(yield_stress),
                "metric_source": "sf_fem",
                "fem_backed": True,
                "paper_eligible": bool(
                    fem_metrics["fem_safe"] and float(pino_metrics.get("vol_ratio", 999.0)) <= 1.3
                ),
                "output_schema_version": BASELINE_FEM_SCHEMA_VERSION,
            }
            row["failure_reason"] = classify_failure(row)
            all_results[method].append(row)
            print(
                f"  {method:12s} FEM-SF={row['sf_fem']:.3f} "
                f"IoU={row['iou_3d']:.3f} Vol={row['vol_ratio']:.3f} time={row['elapsed_s']:.1f}s"
            )
    return all_results


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pino_checkpoint", default="/home/lin/xvxinyao/total_project/checkpoints/pino/best_model.pt")
    parser.add_argument(
        "--dataset_dirs",
        nargs="+",
        default=[
            "/home/lin/xvxinyao/total_project/test_datasets/modelnet40",
            "/home/lin/xvxinyao/total_project/test_datasets/thingi10k",
        ],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["stava", "weighted_sum", "simp", "ntopo"],
        choices=ALL_METHODS,
    )
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--output_dir", default="/home/lin/xvxinyao/total_project/results/baselines_fem_backed")
    parser.add_argument("--device", default=None)
    parser.add_argument("--target_resolution", type=int, default=64)
    parser.add_argument("--yield_stress", type=float, default=YIELD_STRESS)
    parser.add_argument("--limit_models", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--save_outputs", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_manifest.json",
        {
            "output_schema_version": BASELINE_FEM_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "pino_checkpoint": str(args.pino_checkpoint),
            "device": str(device),
            "dataset_dirs": [str(x) for x in args.dataset_dirs],
            "methods": list(args.methods),
            "n_steps": int(args.n_steps),
            "target_resolution": int(args.target_resolution),
            "yield_stress_physical": float(args.yield_stress),
            "save_outputs": bool(args.save_outputs),
            "notes": [
                "Rows in these outputs are FEM-backed and preferred for paper-facing comparisons.",
                "paper_eligible requires FEM-safe final structure and volume-budget satisfaction.",
            ],
        },
    )

    pino_model = load_pino(args.pino_checkpoint, device)
    pino_model._checkpoint_path = str(args.pino_checkpoint)
    combined = {m: [] for m in args.methods}

    for dataset in args.dataset_dirs:
        dataset_dir = Path(dataset)
        results = run_dataset(
            dataset_dir=dataset_dir,
            methods=args.methods,
            pino_model=pino_model,
            device=device,
            output_dir=output_dir,
            n_steps=args.n_steps,
            limit_models=args.limit_models,
            start_index=args.start_index,
            target_resolution=args.target_resolution,
            yield_stress=args.yield_stress,
            save_outputs=args.save_outputs,
        )
        source = json.loads((dataset_dir / "dataset_meta.json").read_text()).get("source", dataset_dir.name)
        write_json(output_dir / f"results_{source}.json", results)
        for m, rows in results.items():
            combined[m].extend(rows)

    write_json(output_dir / "results_combined.json", combined)
    write_json(output_dir / "results_summary.json", summarize_results(combined))
    print(f"Saved FEM-backed baseline results to {output_dir}")


if __name__ == "__main__":
    main()
