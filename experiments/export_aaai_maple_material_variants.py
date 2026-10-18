#!/usr/bin/env python3
"""Export comparable 70 mm AAAI/maple material variants from voxel volumes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import generate_binary_structure, label

from build_projection_guarded_local_reinforcement import (
    erosion_survival,
    project,
    support_metrics,
)
from shadowforge_pipeline.unified_run import export_stl_at_print_scale


S6 = generate_binary_structure(3, 1)


def variant_report(
    name: str,
    volume_path: Path,
    volume: np.ndarray,
    hull_voxels: int,
    target_mm: float,
    output_dir: Path,
    source_display_stl: Path | None = None,
    source_print_stl: Path | None = None,
) -> dict:
    print_volume = np.transpose(volume, (1, 2, 0))
    size_label = f"{target_mm:g}mm"
    display_name = f"{name}_display_{size_label}.stl"
    print_name = f"{name}_print_x_up_{size_label}.stl"

    if source_display_stl is not None:
        display_mesh = trimesh.load_mesh(source_display_stl, process=False)
        display_mesh.apply_scale(target_mm / float(display_mesh.extents.max()))
        display_mesh.export(output_dir / display_name)
    else:
        display_mesh = export_stl_at_print_scale(
            volume, output_dir / display_name, target_mm
        )

    if source_print_stl is not None:
        print_mesh = trimesh.load_mesh(source_print_stl, process=False)
        print_mesh.apply_scale(target_mm / float(print_mesh.extents.max()))
        print_mesh.export(output_dir / print_name)
    else:
        print_mesh = export_stl_at_print_scale(
            print_volume, output_dir / print_name, target_mm
        )

    return {
        "variant": name,
        "source_volume": str(volume_path.resolve()),
        "occupied_voxels": int(volume.sum()),
        "saving_vs_common_visual_hull": float(
            1.0 - int(volume.sum()) / max(hull_voxels, 1)
        ),
        "relative_material_vs_sparse": None,
        "components_6": int(label(volume, structure=S6)[1]),
        "erosion_survival": erosion_survival(volume),
        "print_proxy": support_metrics(print_volume, target_mm),
        "display_mesh": {
            "path": display_name,
            "watertight": bool(display_mesh.is_watertight),
            "mesh_parts": int(len(display_mesh.split(only_watertight=False))),
            "volume_mm3": float(abs(display_mesh.volume)),
            "extents_mm": [float(value) for value in display_mesh.extents],
        },
        "print_mesh": {
            "path": print_name,
            "watertight": bool(print_mesh.is_watertight),
            "mesh_parts": int(len(print_mesh.split(only_watertight=False))),
            "volume_mm3": float(abs(print_mesh.volume)),
            "extents_mm": [float(value) for value in print_mesh.extents],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--reinforced", type=Path, required=True)
    parser.add_argument("--thickened", type=Path, required=True)
    parser.add_argument("--sparse-display-stl", type=Path)
    parser.add_argument("--sparse-print-stl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-mm", type=float, default=70.0)
    args = parser.parse_args()

    sources = [
        ("01_sparse_delivery", args.sparse),
        ("02_local_reinforced", args.reinforced),
        ("03_global_thickened", args.thickened),
    ]
    volumes: list[tuple[str, Path, np.ndarray]] = []
    for name, path in sources:
        volume = np.load(path).astype(bool)
        if volume.shape != (256, 256, 256) or not volume.any():
            raise ValueError(f"{path}: expected non-empty 256^3 volume, got {volume.shape}")
        volumes.append((name, path, volume))

    sparse = volumes[0][2]
    projection_x, projection_y = project(sparse)
    common_hull = projection_x[None, :, :] & projection_y[:, None, :]
    hull_voxels = int(common_hull.sum())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, (name, path, volume) in enumerate(volumes):
        reports.append(
            variant_report(
                name,
                path,
                volume,
                hull_voxels,
                args.target_mm,
                args.output_dir,
                source_display_stl=args.sparse_display_stl if index == 0 else None,
                source_print_stl=args.sparse_print_stl if index == 0 else None,
            )
        )
    sparse_voxels = reports[0]["occupied_voxels"]
    sparse_mesh_volume = reports[0]["print_mesh"]["volume_mm3"]
    for report in reports:
        report["relative_material_vs_sparse"] = float(
            report["occupied_voxels"] / max(sparse_voxels, 1)
        )
        report["relative_mesh_volume_vs_sparse"] = float(
            report["print_mesh"]["volume_mm3"] / max(sparse_mesh_volume, 1e-9)
        )

    result = {
        "schema": "pcraft_aaai_maple_material_variants_v1",
        "target_longest_extent_mm": float(args.target_mm),
        "common_visual_hull_voxels": hull_voxels,
        "comparison_rule": (
            "All variants use the same 256^3 canonical frame, the same longest "
            "extent, the production delivery exporter, and print frame [Y,Z,X]. "
            "The sparse reference reuses the exact selected pipeline mesh."
        ),
        "variants": reports,
    }
    (args.output_dir / "MATERIAL_VARIANTS_QA.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
