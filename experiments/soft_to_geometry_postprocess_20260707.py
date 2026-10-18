#!/usr/bin/env python3
"""Test whether geometry export/post-processing fixes soft-branch mesh failures.

This script uses the saved mesh_combined2 soft result and reports only quantities
that can be checked from local artifacts: voxel occupancy, 6-connected
components, largest-component mass, and mesh watertightness/component count.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, label, generate_binary_structure

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path.insert(0, str(ROOT))

from generation.voxel_ops import sdf_watertight_mesh


SRC = ROOT / "results/mesh_combined2_best_soft_ablation_20260706"
OUT = ROOT / "results/soft_geometry_postprocess_20260707"


def keep_largest(vol: np.ndarray) -> tuple[np.ndarray, int, float]:
    s6 = generate_binary_structure(3, 1)
    lab, n = label(vol, structure=s6)
    if n == 0:
        return vol.astype(bool), 0, 0.0
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    largest = int(counts.argmax())
    ratio = float(counts[largest] / max(int(vol.sum()), 1))
    return (lab == largest), int(n), ratio


def mesh_stats(mesh) -> dict:
    mesh.merge_vertices()
    mesh.process()
    parts = mesh.split(only_watertight=False)
    return {
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_components": int(len(parts)),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
    }


def summarize_variant(name: str, vol: np.ndarray) -> dict:
    vol = vol.astype(bool)
    largest, comps, largest_ratio = keep_largest(vol)
    mesh = sdf_watertight_mesh(vol, smooth_sigma=0.8)
    mesh_path = OUT / f"{name}.ply"
    stats = mesh_stats(mesh)
    mesh.export(mesh_path)
    stats.update(
        {
            "variant": name,
            "occupied_voxels": int(vol.sum()),
            "component_count_6conn": int(comps),
            "largest_component_mass": float(largest_ratio),
            "mesh_path": str(mesh_path),
        }
    )
    return stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_path = SRC / "rows.json"
    rows = json.loads(rows_path.read_text())
    variants = []
    sources = []
    for row in rows:
        metrics = row.get("metrics", {})
        files = metrics.get("files", {})
        vol_name = files.get("voxels")
        if not vol_name or not Path(vol_name).exists():
            sources.append({
                "variant": row.get("variant"),
                "status": "skipped",
                "reason": "missing voxel file",
            })
            continue
        vol_path = Path(vol_name)
        raw = np.squeeze(np.load(vol_path)).astype(bool)
        closed = binary_fill_holes(
            binary_closing(raw, structure=generate_binary_structure(3, 1), iterations=1)
        )
        largest, _, _ = keep_largest(closed)
        variant = str(row["variant"])
        variants.extend([
            summarize_variant(f"{variant}_raw_sdf", raw),
            summarize_variant(f"{variant}_closed_sdf", closed),
            summarize_variant(f"{variant}_closed_largest_sdf", largest),
        ])
        sources.append({
            "variant": variant,
            "status": "processed",
            "source_voxels": str(vol_path),
            "source_export_mean_iou": metrics.get("export_metrics", {}).get("mean_iou"),
            "source_export_min_iou": metrics.get("export_metrics", {}).get("min_iou"),
            "source_export_component_count": metrics.get("export_metrics", {}).get("component_count"),
            "source_mesh_watertight": metrics.get("mesh_watertight"),
            "source_mesh_split_components": metrics.get("mesh_split_components"),
        })

    source = {
        "source_rows_json": str(rows_path),
        "processed_variants": sources,
        "note": "Projection IoU after SDF export is not re-reported here because the soft run used a two-silhouette -tsf protocol; this script audits geometry/export failure modes only.",
    }
    (OUT / "summary.json").write_text(json.dumps({"source": source, "variants": variants}, indent=2))

    with (OUT / "summary.tsv").open("w", newline="") as f:
        fieldnames = [
            "variant",
            "occupied_voxels",
            "component_count_6conn",
            "largest_component_mass",
            "mesh_watertight",
            "mesh_components",
            "vertices",
            "faces",
            "mesh_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in variants:
            writer.writerow({k: row[k] for k in fieldnames})

    print(json.dumps({"source": source, "summary_tsv": str(OUT / "summary.tsv")}, indent=2))


if __name__ == "__main__":
    main()
