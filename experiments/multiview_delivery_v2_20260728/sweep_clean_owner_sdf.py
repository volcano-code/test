#!/usr/bin/env python3
"""Tune SDF reconstruction for an already-clean multi-view geometry owner.

This diagnostic is intentionally isolated from the mature two-wall pipeline.
It evaluates reconstructed binary volumes with the exact recorded camera
protocol before exporting a small set of watertight exterior-shell meshes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
ISOLATED = ROOT / "experiments/fixed_surface_xyz_isolated_20260726"
for module_path in (
    ISOLATED / "shadowforge_pipeline",
    ISOLATED / "generation",
    ISOLATED,
    ROOT,
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import unified_run as pipeline
from voxel_ops import sdf_watertight_mesh

S6 = ndimage.generate_binary_structure(3, 1)
S26 = ndimage.generate_binary_structure(3, 3)


def parse_values(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-artifact-dir", type=Path, required=True)
    parser.add_argument("--source-volume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sigmas", default="0.6,0.8,1.0,1.2")
    parser.add_argument("--levels", default="0.0,0.25,0.5,0.75")
    parser.add_argument("--min-mean-iou", type=float, default=0.95)
    parser.add_argument("--min-view-iou", type=float, default=0.93)
    parser.add_argument("--export-count", type=int, default=6)
    parser.add_argument("--print-mm", type=float, default=100.0)
    return parser.parse_args()


def largest_component(volume: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(volume, structure=S6)
    if count == 0:
        return np.zeros_like(volume, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def surface_fraction(volume: np.ndarray) -> float:
    occupied = np.asarray(volume, dtype=bool)
    count = int(occupied.sum())
    if count == 0:
        return 1.0
    interior = ndimage.binary_erosion(occupied, structure=S6)
    return float((count - int(interior.sum())) / count)


def make_protocol(metrics: dict, device: str) -> SimpleNamespace:
    trainer_args = metrics.get("args", {})
    return SimpleNamespace(
        device=device,
        silhouettes=[
            Path(path) for path in trainer_args.get("shadow_files", [])
        ],
        mr=int(trainer_args.get("mirror_mode", 0)),
        tsf=int(trainer_args.get("tsf_mode", 0)),
        print_mm=100.0,
    )


def exterior_shell(volume: np.ndarray) -> tuple[trimesh.Trimesh, dict]:
    mesh = sdf_watertight_mesh(volume, smooth_sigma=0.0)
    parts = [part for part in mesh.split(only_watertight=False) if part.is_watertight]
    if not parts:
        raise RuntimeError("SDF reconstruction produced no watertight shell")
    parts.sort(key=lambda part: abs(float(part.volume)), reverse=True)
    shell = parts[0].copy()
    trimesh.repair.fix_normals(shell, multibody=False)
    total_volume = sum(abs(float(part.volume)) for part in parts)
    return shell, {
        "shells": int(len(parts)),
        "exterior_fraction": float(
            abs(float(shell.volume)) / max(total_volume, 1e-12)
        ),
        "triangles": int(len(shell.faces)),
        "watertight": bool(shell.is_watertight),
    }


def scale_for_print(mesh: trimesh.Trimesh, print_mm: float) -> trimesh.Trimesh:
    result = mesh.copy()
    extent = float(np.max(result.extents))
    if extent <= 0:
        raise ValueError("Cannot scale an empty mesh")
    result.apply_scale(float(print_mm) / extent)
    center = result.bounds.mean(axis=0)
    result.apply_translation([-center[0], -center[1], -result.bounds[0, 2]])
    return result


def main() -> None:
    args = parse_args()
    soft_dir = args.soft_artifact_dir.resolve()
    source_path = args.source_volume.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(
        (soft_dir / "metrics.json").read_text(encoding="utf-8")
    )
    protocol = make_protocol(metrics, args.device)
    soft_info = {"metrics": metrics}
    proposal = pipeline._canonical_xyz_from_trainer_zyx(
        np.load(soft_dir / "occupancy_soft_proposal.npy").astype(bool)
    )
    targets = pipeline._generic_targets(soft_dir, int(proposal.shape[0]))
    reference_voxels = int(proposal.sum())

    source = largest_component(np.load(source_path).astype(bool))
    source = ndimage.binary_fill_holes(source, structure=S26)
    signed_distance = (
        ndimage.distance_transform_edt(~source)
        - ndimage.distance_transform_edt(source)
    ).astype(np.float32)

    rows: list[dict] = []
    volumes: dict[str, np.ndarray] = {}
    for sigma in parse_values(args.sigmas):
        smoothed = ndimage.gaussian_filter(signed_distance, sigma=sigma)
        for level in parse_values(args.levels):
            name = f"s{sigma:g}_l{level:g}"
            reconstructed = smoothed <= level
            reconstructed = ndimage.binary_fill_holes(
                reconstructed, structure=S26
            )
            reconstructed = largest_component(reconstructed)
            candidate_metrics = pipeline._generic_prescreen_metrics(
                reconstructed,
                targets,
                reference_voxels,
                protocol,
                soft_info,
            )
            row = {
                "candidate": name,
                "sigma": float(sigma),
                "level": float(level),
                "occupied_voxels": int(reconstructed.sum()),
                "surface_fraction": surface_fraction(reconstructed),
                **candidate_metrics,
            }
            row["projection_pass"] = bool(
                row["mean_iou"] >= args.min_mean_iou
                and row["min_iou"] >= args.min_view_iou
                and row["components"] == 1
            )
            rows.append(row)
            volumes[name] = reconstructed
            print(json.dumps(row), flush=True)

    eligible = [row for row in rows if row["projection_pass"]]
    eligible.sort(
        key=lambda row: (
            row["surface_fraction"],
            -row["min_iou"],
            -row["mean_iou"],
        )
    )

    exports: list[dict] = []
    for row in eligible[: args.export_count]:
        name = row["candidate"]
        volume = volumes[name]
        np.save(output / f"{name}.npy", volume.astype(np.uint8))
        shell, shell_audit = exterior_shell(volume)
        shell = scale_for_print(shell, args.print_mm)
        stl_path = output / f"{name}_external_shell_100mm.stl"
        shell.export(stl_path)
        exports.append(
            {
                "candidate": name,
                "stl": str(stl_path),
                **shell_audit,
            }
        )

    manifest = {
        "source_volume": str(source_path),
        "rows": rows,
        "exports": exports,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
