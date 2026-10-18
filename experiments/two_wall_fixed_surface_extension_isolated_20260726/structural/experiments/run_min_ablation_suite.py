#!/usr/bin/env python3
"""
Run the minimum ablation suite needed for paper readiness.

Priority variants:
- Stage A only
- Ours w/o ALM
- Ours w/o Calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from structural.analysis.load_generator import LoadGenerator
from structural.analysis.stress_eval import get_yield_stress_for_model
from structural.experiments.ablation import (
    run_ablation_no_alm,
    run_ablation_no_calibration,
)
from structural.models.load_model import load_surrogate, forward_surrogate_physical


def load_models(dataset_dir: Path, model_filter: list[str] | None, n_models: int) -> list[dict]:
    meta = json.loads((dataset_dir / "dataset_meta.json").read_text())
    models = meta.get("models", [])
    if model_filter:
        wanted = set(model_filter)
        models = [m for m in models if m.get("model_id") in wanted]
    return models[:n_models]


def load_density(voxel_path: Path, target: int = 64) -> np.ndarray:
    density = np.load(str(voxel_path))
    if density.ndim == 4:
        density = density[0]
    if density.ndim == 5:
        density = density[0, 0]
    if density.shape != (target, target, target):
        density = zoom(density, target / density.shape[0], order=1)
        density = np.clip(density, 0, 1)
    return density.astype(np.float32)


def evaluate_density(pino, density: np.ndarray, force: np.ndarray, yield_stress: float, device: torch.device) -> dict:
    density_t = torch.from_numpy(density).float().unsqueeze(0).unsqueeze(0).to(device)
    force_t = torch.from_numpy(force).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = forward_surrogate_physical(pino, density_t, force_t)
        vm = out["von_mises"][0, 0]
        solid = density_t[0, 0] > 0.3
        vm_solid = vm[solid]
        vm_max = float(vm_solid.max().item()) if vm_solid.numel() else 0.0
        sf = float(yield_stress / max(vm_max, 1e-12))
    return {
        "sf_pred": sf,
        "vm_max_pred": vm_max,
        "vol_ratio": float((density > 0.5).mean()),
    }


def run_stage_a_only(density_original: np.ndarray, **_: object) -> np.ndarray:
    return density_original.copy()


ABLATION_RUNNERS = {
    "Stage A only": run_stage_a_only,
    "Ours w/o ALM": run_ablation_no_alm,
    "Ours w/o Calibration": run_ablation_no_calibration,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/home/lin/xvxinyao/total_project/test_datasets/modelnet40")
    parser.add_argument("--pino_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/home/lin/xvxinyao/total_project/results/ablations_min")
    parser.add_argument("--n_models", type=int, default=3)
    parser.add_argument("--model_filter", nargs="*", default=None)
    parser.add_argument("--yield_stress", type=float, default=50e6)
    parser.add_argument("--n_steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pino, _ = load_surrogate(args.pino_checkpoint, device)
    pino.eval()

    models = load_models(dataset_dir, args.model_filter, args.n_models)
    load_gen = LoadGenerator(material_density=1.24e3)

    rows: list[dict] = []
    for model in models:
        model_id = model["model_id"]
        voxel_path = dataset_dir / model_id / "voxels.npy"
        if not voxel_path.exists():
            continue
        density = load_density(voxel_path)
        load = load_gen.standing_load(density)
        force = load["force"]
        fixed_mask = load["fixed_mask"]

        for method_name, fn in ABLATION_RUNNERS.items():
            if method_name == "Stage A only":
                density_out = fn(density_original=density)
            else:
                density_out = fn(
                    density_original=density,
                    force=force,
                    fixed_mask=fixed_mask,
                    pino_model=pino,
                    yield_stress=args.yield_stress,
                    device=device,
                    n_steps=args.n_steps,
                    lr=args.lr,
                )
            metrics = evaluate_density(pino, density_out, force, args.yield_stress, device)
            row = {
                "dataset_dir": str(dataset_dir),
                "model_id": model_id,
                "method": method_name,
                "yield_stress": float(args.yield_stress),
                "n_steps": int(args.n_steps),
                **metrics,
            }
            rows.append(row)

            model_out = output_dir / method_name.replace(" ", "_").replace("/", "_") / model_id
            model_out.mkdir(parents=True, exist_ok=True)
            np.save(model_out / "density.npy", density_out)
            (model_out / "summary.json").write_text(json.dumps(row, indent=2))

    summary: dict[str, dict] = {}
    for method_name in sorted({r["method"] for r in rows}):
        subset = [r for r in rows if r["method"] == method_name]
        summary[method_name] = {
            "n": len(subset),
            "sf_pred_mean": float(sum(r["sf_pred"] for r in subset) / len(subset)) if subset else None,
            "vm_max_pred_mean": float(sum(r["vm_max_pred"] for r in subset) / len(subset)) if subset else None,
            "vol_ratio_mean": float(sum(r["vol_ratio"] for r in subset) / len(subset)) if subset else None,
        }

    manifest = {
        "schema_version": "ablation_min_v1",
        "dataset_dir": str(dataset_dir),
        "pino_checkpoint": str(args.pino_checkpoint),
        "yield_stress": float(args.yield_stress),
        "n_models": int(args.n_models),
        "methods": list(ABLATION_RUNNERS.keys()),
        "output_dir": str(output_dir),
        "effective_yield_for_model": float(get_yield_stress_for_model(pino, args.yield_stress)),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "results.json").write_text(json.dumps(rows, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved ablation outputs to {output_dir}")


if __name__ == "__main__":
    main()
