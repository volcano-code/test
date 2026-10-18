from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from structural.export.remap_fem_repair_to_highres import evaluate_silhouette_changes


def _load_bin(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 5:
        arr = arr[0, 0]
    return arr > 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an alignment summary using silhouette IoU as the appearance metric")
    parser.add_argument("--original", required=True)
    parser.add_argument("--repaired", required=True)
    parser.add_argument("--revoxelized", required=True)
    parser.add_argument("--orig_fem_json", required=True)
    parser.add_argument("--repaired_fem_json", required=True)
    parser.add_argument("--revox_fem_json", required=True)
    parser.add_argument("--mesh_fem_json", required=False, default=None)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    original = _load_bin(Path(args.original))
    repaired = _load_bin(Path(args.repaired))
    revox = _load_bin(Path(args.revoxelized))

    orig_fem = json.load(open(args.orig_fem_json))
    repaired_fem = json.load(open(args.repaired_fem_json))
    revox_fem = json.load(open(args.revox_fem_json))
    mesh_fem = json.load(open(args.mesh_fem_json)) if args.mesh_fem_json else None

    summary = {
        "recommended_unified_standard": {
            "primary_structural_metric": "voxel-FEM",
            "primary_appearance_metric": "multi-view silhouette IoU",
            "secondary_metric": "direct mesh-FEM as auxiliary evidence only",
            "reason": "Silhouette IoU matches the shadow-art appearance objective and remains comparable across voxel and OBJ branches, while direct mesh-FEM is still sensitive to geometry scale and support-thickness choices.",
        },
        "structural_metrics": {
            "original_voxel_fem": orig_fem["fem"],
            "repaired_voxel_fem": repaired_fem["fem"],
            "revoxelized_obj_voxel_fem": revox_fem["fem"],
            "direct_mesh_fem": mesh_fem["fem"] if mesh_fem else None,
        },
        "appearance_metrics": {
            "original_vs_repaired_voxel": evaluate_silhouette_changes(original, repaired),
            "repaired_voxel_vs_revoxelized_obj": evaluate_silhouette_changes(repaired, revox),
            "original_vs_revoxelized_obj": evaluate_silhouette_changes(original, revox),
        },
    }

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
