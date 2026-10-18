#!/usr/bin/env python3
"""Prune attached voxel noise from a clean multi-view geometry owner.

This diagnostic stays isolated from the mature two-wall implementation.  It
cuts thin attachments in volume space, keeps the largest connected body, and
accepts a result only after replaying the recorded projection protocol.
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
S18 = ndimage.generate_binary_structure(3, 2)
S26 = ndimage.generate_binary_structure(3, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-artifact-dir", type=Path, required=True)
    parser.add_argument("--source-volume", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-mean-iou", type=float, default=0.95)
    parser.add_argument("--min-view-iou", type=float, default=0.93)
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
        silhouettes=[Path(path) for path in trainer_args.get("shadow_files", [])],
        mr=int(trainer_args.get("mirror_mode", 0)),
        tsf=int(trainer_args.get("tsf_mode", 0)),
        print_mm=100.0,
    )


def scale_for_print(mesh: trimesh.Trimesh, print_mm: float) -> trimesh.Trimesh:
    result = mesh.copy()
    extent = float(np.max(result.extents))
    if extent <= 0:
        raise ValueError("Cannot scale an empty mesh")
    result.apply_scale(float(print_mm) / extent)
    center = result.bounds.mean(axis=0)
    result.apply_translation([-center[0], -center[1], -result.bounds[0, 2]])
    return result


def exterior_shell(volume: np.ndarray, sigma: float) -> tuple[trimesh.Trimesh, dict]:
    mesh = sdf_watertight_mesh(volume, smooth_sigma=sigma)
    parts = [part for part in mesh.split(only_watertight=False) if part.is_watertight]
    if not parts:
        raise RuntimeError("No watertight SDF shell was produced")
    parts.sort(key=lambda part: abs(float(part.volume)), reverse=True)
    shell = parts[0].copy()
    trimesh.repair.fix_normals(shell, multibody=False)
    total = sum(abs(float(part.volume)) for part in parts)
    return shell, {
        "mesh_shells": int(len(parts)),
        "mesh_exterior_fraction": float(
            abs(float(shell.volume)) / max(total, 1e-12)
        ),
        "triangles": int(len(shell.faces)),
        "watertight": bool(shell.is_watertight),
    }


def regularized_candidates(source: np.ndarray) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {"source": source}

    for structure_name, structure in (("s6", S6), ("s18", S18)):
        opened = ndimage.binary_opening(source, structure=structure, iterations=1)
        candidates[f"open_{structure_name}"] = largest_component(opened)

        opened_closed = ndimage.binary_closing(
            opened, structure=S6, iterations=1
        )
        candidates[f"open_{structure_name}_close_s6"] = largest_component(
            opened_closed
        )

    neighbor_count = ndimage.convolve(
        source.astype(np.uint8),
        np.ones((3, 3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    for minimum in (3, 4, 5, 6, 8, 10):
        pruned = source & (neighbor_count >= minimum)
        candidates[f"prune_n{minimum}"] = largest_component(pruned)
        closed = ndimage.binary_closing(pruned, structure=S6, iterations=1)
        candidates[f"prune_n{minimum}_close_s6"] = largest_component(closed)

    return candidates


def main() -> None:
    args = parse_args()
    soft_dir = args.soft_artifact_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((soft_dir / "metrics.json").read_text(encoding="utf-8"))
    protocol = make_protocol(metrics, args.device)
    soft_info = {"metrics": metrics}
    proposal = pipeline._canonical_xyz_from_trainer_zyx(
        np.load(soft_dir / "occupancy_soft_proposal.npy").astype(bool)
    )
    targets = pipeline._generic_targets(soft_dir, int(proposal.shape[0]))
    reference_voxels = int(proposal.sum())

    source = largest_component(np.load(args.source_volume).astype(bool))
    rows: list[dict] = []
    volumes: dict[str, np.ndarray] = {}
    for name, candidate in regularized_candidates(source).items():
        candidate = largest_component(candidate)
        candidate_metrics = pipeline._generic_prescreen_metrics(
            candidate,
            targets,
            reference_voxels,
            protocol,
            soft_info,
        )
        row = {
            "candidate": name,
            "occupied_voxels": int(candidate.sum()),
            "surface_fraction": surface_fraction(candidate),
            **candidate_metrics,
        }
        row["projection_pass"] = bool(
            row["mean_iou"] >= args.min_mean_iou
            and row["min_iou"] >= args.min_view_iou
            and row["components"] == 1
        )
        rows.append(row)
        volumes[name] = candidate
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
    for row in eligible[:6]:
        name = row["candidate"]
        volume = volumes[name]
        np.save(output / f"{name}.npy", volume.astype(np.uint8))
        for sigma in (0.6, 0.8):
            shell, audit = exterior_shell(volume, sigma)
            shell = scale_for_print(shell, args.print_mm)
            stl_path = output / f"{name}_sdf{sigma:g}_100mm.stl"
            shell.export(stl_path)
            exports.append(
                {
                    "candidate": name,
                    "sdf_sigma": sigma,
                    "stl": str(stl_path),
                    **audit,
                }
            )

    manifest = {
        "source_volume": str(args.source_volume.resolve()),
        "rows": rows,
        "exports": exports,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
