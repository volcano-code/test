"""
Generate continuous-density FEM training samples from existing voxel datasets.

Why:
  Stage C optimizes continuous densities in [0, 1], but the legacy FEM
  datasets are mostly near-binary. This script creates intermediate-density
  variants and re-solves FEM so PINO can learn meaningful responses on those
  states.

Usage:
  python structural/data/generate_continuous_density_fem_dataset.py \
    --source_dirs fem_dataset,fem_dataset_v3,fem_dataset_v3b \
    --output_dir fem_dataset_continuous \
    --variants_per_sample 2 \
    --limit_samples 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from structural.analysis.fem_solver import voxel_fem_solve


def _continuous_variant(
    density: np.ndarray,
    rng: np.random.Generator,
    sigma_range: tuple[float, float],
    blend_range: tuple[float, float],
    simp_powers: list[float],
) -> tuple[np.ndarray, dict]:
    base = np.clip(density.astype(np.float32), 0.0, 1.0)
    sigma = float(rng.uniform(*sigma_range))
    blend = float(rng.uniform(*blend_range))
    simp_p = float(rng.choice(simp_powers))

    blurred = gaussian_filter(base, sigma=sigma)
    if float(blurred.max()) > 1e-8:
        blurred = blurred / float(blurred.max())
    mixed = blend * base + (1.0 - blend) * blurred
    mixed = np.clip(mixed, 0.0, 1.0)
    variant = np.power(mixed, simp_p).astype(np.float32)
    variant = np.clip(variant, 0.0, 1.0)

    meta = {
        "augment_type": "gaussian_blur_plus_simp",
        "gaussian_sigma": sigma,
        "blend_alpha": blend,
        "simp_power": simp_p,
        "solid_ratio_03": float((variant > 0.3).mean()),
        "solid_ratio_05": float((variant > 0.5).mean()),
    }
    return variant, meta


def _adapt_force_to_density(force: np.ndarray, density_orig: np.ndarray, density_new: np.ndarray) -> np.ndarray:
    """Approximate load transfer for continuous density variants.

    We keep the spatial load/support pattern but scale per-voxel body load with
    the new density ratio. This is not meant to replace full load regeneration
    for all scenarios, but is a much better approximation than keeping binary
    loads unchanged while changing voxel mass/stiffness.
    """
    force = force.astype(np.float32)
    ratio = np.zeros_like(density_new, dtype=np.float32)
    mask = density_orig > 1e-4
    ratio[mask] = density_new[mask] / np.maximum(density_orig[mask], 1e-4)
    ratio = np.clip(ratio, 0.0, 3.0)
    return force * ratio[None, ...]


def generate_continuous_density_dataset(
    source_dirs: list[str],
    output_dir: str,
    variants_per_sample: int = 2,
    limit_samples: int = 0,
    start_idx: int = 0,
    seed: int = 42,
    sigma_min: float = 1.0,
    sigma_max: float = 3.0,
    blend_min: float = 0.35,
    blend_max: float = 0.8,
    simp_powers: str = "1.0,1.5,2.0,3.0",
    density_threshold: float = 0.05,
) -> dict:
    rng = np.random.default_rng(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for src in source_dirs:
        files.extend(sorted(Path(src).glob("sample_*.npz")))
    if limit_samples > 0:
        files = files[:limit_samples]

    simp_power_list = [float(x) for x in simp_powers.split(",") if x.strip()]
    sigma_range = (sigma_min, sigma_max)
    blend_range = (blend_min, blend_max)

    saved = 0
    failed = 0
    records = []
    next_idx = start_idx

    pbar = tqdm(files, desc="Generating continuous-density FEM dataset")
    for src_path in pbar:
        data = np.load(src_path)
        density = data["density"].astype(np.float32)
        force = data["force"].astype(np.float32)
        fixed_mask = data["fixed_mask"].astype(np.float32)
        if fixed_mask.ndim == 4:
            fixed_mask_3d = fixed_mask[0]
        else:
            fixed_mask_3d = fixed_mask

        for _ in range(variants_per_sample):
            density_variant, meta = _continuous_variant(
                density, rng, sigma_range, blend_range, simp_power_list
            )
            force_variant = _adapt_force_to_density(force, density, density_variant)

            try:
                fem = voxel_fem_solve(
                    density=density_variant,
                    force_field=force_variant,
                    fixed_mask=fixed_mask_3d,
                    E=3.5e9,
                    nu=0.36,
                    density_threshold=density_threshold,
                    verbose=False,
                )
            except Exception:
                failed += 1
                continue

            if not np.all(np.isfinite(fem["displacement"])) or not np.all(np.isfinite(fem["stress"])):
                failed += 1
                continue
            if fem.get("vm_max", 0.0) <= 0:
                failed += 1
                continue

            out_path = out_dir / f"sample_{next_idx:05d}.npz"
            np.savez_compressed(
                out_path,
                density=density_variant.astype(np.float32),
                force=force_variant.astype(np.float32),
                fixed_mask=fixed_mask.astype(np.float32),
                displacement=fem["displacement"].astype(np.float32),
                stress=fem["stress"].astype(np.float32),
                von_mises=fem["von_mises"].astype(np.float32),
                source_sample=str(src_path),
                density_mode="continuous",
                augment_type=meta["augment_type"],
                gaussian_sigma=np.float32(meta["gaussian_sigma"]),
                blend_alpha=np.float32(meta["blend_alpha"]),
                simp_power=np.float32(meta["simp_power"]),
            )

            records.append(
                {
                    "sample_idx": next_idx,
                    "source_sample": str(src_path),
                    **meta,
                    "vm_max": float(fem["vm_max"]),
                    "sf_min": float(fem["sf_min"]),
                }
            )
            saved += 1
            next_idx += 1

        pbar.set_postfix(saved=saved, failed=failed)

    summary = {
        "schema_version": "continuous_density_fem_dataset_v1",
        "source_dirs": source_dirs,
        "output_dir": str(out_dir),
        "variants_per_sample": variants_per_sample,
        "limit_samples": limit_samples,
        "seed": seed,
        "density_threshold": density_threshold,
        "saved": saved,
        "failed": failed,
        "records": records[: min(len(records), 200)],
    }
    (out_dir / "metadata.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate continuous-density FEM dataset")
    parser.add_argument("--source_dirs", required=True, help="Comma-separated source dataset dirs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variants_per_sample", type=int, default=2)
    parser.add_argument("--limit_samples", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma_min", type=float, default=1.0)
    parser.add_argument("--sigma_max", type=float, default=3.0)
    parser.add_argument("--blend_min", type=float, default=0.35)
    parser.add_argument("--blend_max", type=float, default=0.80)
    parser.add_argument("--simp_powers", type=str, default="1.0,1.5,2.0,3.0")
    parser.add_argument("--density_threshold", type=float, default=0.05)
    args = parser.parse_args()

    summary = generate_continuous_density_dataset(
        source_dirs=[d.strip() for d in args.source_dirs.split(",") if d.strip()],
        output_dir=args.output_dir,
        variants_per_sample=args.variants_per_sample,
        limit_samples=args.limit_samples,
        start_idx=args.start_idx,
        seed=args.seed,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        blend_min=args.blend_min,
        blend_max=args.blend_max,
        simp_powers=args.simp_powers,
        density_threshold=args.density_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
