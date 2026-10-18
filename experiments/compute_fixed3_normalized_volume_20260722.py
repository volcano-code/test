#!/usr/bin/env python3
"""Compare material volume after isotropic size normalization.

For every output, the longest bounding-box extent is normalized to one.  The
reported dimensionless volume is therefore V / L_max^3.  This compares material
use at a common maximum print dimension without assigning a method-specific
notion of "saving" to baselines that do not perform carving.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


ROOT = Path("/home/lin/xvxinyao/ShadowForge")


def voxel_measure(path: Path, threshold: float | None = None) -> dict:
    volume = np.squeeze(np.load(path))
    occupied = volume.astype(bool) if threshold is None else volume > threshold
    points = np.argwhere(occupied)
    if len(points) == 0:
        raise ValueError(f"Empty voxel result: {path}")
    extents = points.max(axis=0) - points.min(axis=0) + 1
    longest = int(extents.max())
    count = int(occupied.sum())
    return {
        "representation": "voxel",
        "occupied_voxels": count,
        "bbox_extents": [int(value) for value in extents],
        "longest_extent": longest,
        "normalized_volume": float(count / longest**3),
        "watertight_for_volume": None,
    }


def mesh_measure(path: Path) -> dict:
    mesh = trimesh.load(path, force="mesh", process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    extents = np.asarray(mesh.extents, dtype=np.float64)
    longest = float(extents.max())
    if not np.isfinite(longest) or longest <= 0:
        raise ValueError(f"Degenerate mesh bounds: {path}")
    physical_volume = float(abs(mesh.volume))
    if not np.isfinite(physical_volume):
        raise ValueError(f"Invalid mesh volume: {path}")
    return {
        "representation": "mesh",
        "physical_volume_native_units": physical_volume,
        "bbox_extents": [float(value) for value in extents],
        "longest_extent": longest,
        "normalized_volume": float(physical_volume / longest**3),
        "watertight_for_volume": bool(mesh.is_watertight),
        "mesh_components": int(len(mesh.split(only_watertight=False))),
    }


def mask_bbox_extents(path: Path) -> tuple[int, int]:
    mask = np.asarray(Image.open(path).convert("L")) > 127
    points = np.argwhere(mask)
    if len(points) == 0:
        raise ValueError(f"Empty projection: {path}")
    extents = points.max(axis=0) - points.min(axis=0) + 1
    return int(extents[0]), int(extents[1])


def pcraft_measure(run_dir: Path, hull_path: Path, manifest: dict) -> dict:
    """Recover selected candidate volume without integrating internal STL shells."""
    hull_voxels = int(np.asarray(np.load(hull_path), dtype=bool).sum())
    selected_voxels = float(hull_voxels * (1.0 - float(manifest["selected"]["saving"])))
    hull_resolution = int(np.load(hull_path, mmap_mode="r").shape[0])
    projection_paths = [
        run_dir / "projections" / f"projection_view_{index}.png" for index in range(3)
    ]
    projection_extents = [mask_bbox_extents(path) for path in projection_paths]
    image_resolution = int(Image.open(projection_paths[0]).size[0])
    scale = hull_resolution / image_resolution
    grid_extents = [tuple(value * scale for value in pair) for pair in projection_extents]
    longest = max(value for pair in grid_extents for value in pair)
    return {
        "representation": "selected voxel candidate (volume recovered from recorded saving)",
        "occupied_voxels": selected_voxels,
        "projection_image_extents": [list(pair) for pair in projection_extents],
        "bbox_extents_in_delivery_grid": [list(pair) for pair in grid_extents],
        "longest_extent": float(longest),
        "normalized_volume": float(selected_voxels / longest**3),
        "watertight_for_volume": bool(manifest["selected"]["watertight"]),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/fixed3_mr1_connected10_pending_20260722/benchmark_manifest.json",
    )
    parser.add_argument(
        "--mitra",
        type=Path,
        default=ROOT / "results/fixed3_mr1_connected10_mitra_20260722",
    )
    parser.add_argument(
        "--sadekar-summary",
        type=Path,
        default=ROOT / "results/fixed3_mr1_connected10_sadekar_20260722/baseline_run_summary.json",
    )
    parser.add_argument(
        "--pcraft",
        type=Path,
        default=ROOT / "results/fixed3_mr1_connected10_pcraft_20260722",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/fixed3_mr1_connected10_normalized_volume_20260722",
    )
    args = parser.parse_args()

    benchmark = json.loads(args.manifest.read_text())
    case_ids = [row["case_id"] for row in benchmark["cases"]]
    sadekar = json.loads(args.sadekar_summary.read_text())
    seeds = [int(seed) for seed in benchmark["protocol"]["seeds"]]
    sad_index = {
        (row["case_id"], row["kind"], int(row["seed"])): Path(row["artifact"])
        for row in sadekar["runs"]
    }

    rows: list[dict] = []
    for case_id in case_ids:
        mitra_path = args.mitra / case_id / "mitra_pauly2009_shadow_hull.npy"
        rows.append({
            "case_id": case_id,
            "method": "Mitra-style",
            "seed": None,
            "artifact": str(mitra_path),
            **voxel_measure(mitra_path),
        })

        for seed in seeds:
            voxel_path = sad_index[(case_id, "voxel", seed)]
            rows.append({
                "case_id": case_id,
                "method": "Sadekar voxel",
                "seed": seed,
                "artifact": str(voxel_path),
                **voxel_measure(voxel_path, threshold=0.05),
            })

            mesh_path = sad_index[(case_id, "mesh", seed)]
            rows.append({
                "case_id": case_id,
                "method": "Sadekar mesh",
                "seed": seed,
                "artifact": str(mesh_path),
                **mesh_measure(mesh_path),
            })

            manifest_path = args.pcraft / case_id / f"seed_{seed}" / "MANIFEST.json"
            manifest = json.loads(manifest_path.read_text())
            stl_path = Path(manifest["display_stl"])
            rows.append({
                "case_id": case_id,
                "method": "P-CRAFT",
                "seed": seed,
                "artifact": str(stl_path),
                **pcraft_measure(manifest_path.parent, mitra_path, manifest),
            })

    by_case_method: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_case_method[(row["case_id"], row["method"])].append(row)

    per_case = []
    for (case_id, method), group in sorted(by_case_method.items()):
        mean, std = mean_std([row["normalized_volume"] for row in group])
        per_case.append({
            "case_id": case_id,
            "method": method,
            "runs": len(group),
            "normalized_volume_mean": mean,
            "normalized_volume_std_across_seeds": std,
            "all_meshes_watertight": all(
                row["watertight_for_volume"] is not False for row in group
            ),
        })

    aggregate = []
    for method in ("Mitra-style", "Sadekar mesh", "Sadekar voxel", "P-CRAFT"):
        task_values = [
            row["normalized_volume_mean"] for row in per_case if row["method"] == method
        ]
        mean, std = mean_std(task_values)
        run_values = [row["normalized_volume"] for row in rows if row["method"] == method]
        aggregate.append({
            "method": method,
            "tasks": len(task_values),
            "runs": len(run_values),
            "normalized_volume_mean": mean,
            "normalized_volume_std_across_tasks": std,
            "normalized_volume_min": float(np.min(task_values)),
            "normalized_volume_max": float(np.max(task_values)),
            "all_meshes_watertight": all(
                row["watertight_for_volume"] is not False
                for row in rows if row["method"] == method
            ),
        })

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric": "isotropic normalized solid volume",
        "definition": "V_norm = V / L_max^3 after setting each output's longest bounding-box extent to one",
        "interpretation": "lower is less material at the same maximum physical print dimension",
        "protocol": benchmark["protocol"],
        "aggregate": aggregate,
        "per_case": per_case,
        "per_run": rows,
    }
    (args.output / "normalized_volume_results.json").write_text(json.dumps(payload, indent=2) + "\n")

    for filename, table in (
        ("normalized_volume_aggregate.tsv", aggregate),
        ("normalized_volume_per_case.tsv", per_case),
    ):
        with (args.output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(table)

    print(json.dumps({"output": str(args.output), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
