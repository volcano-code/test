#!/usr/bin/env python3
"""Freeze the two-wall physical-print benchmark used for local baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/physical_print_controlled_baselines_v2_20260715"

CASES = [
    {
        "case_id": "coord_calibration_archer_heroine",
        "split": "coordinate_calibration",
        "source": ROOT / "results/fem_two_wall_print_candidate_search_20260714/anime/H05_archer_heroine",
        "report": False,
    },
    {
        "case_id": "dark_knight_magic_scholar",
        "split": "test",
        "source": ROOT / "results/dark_knight_fdm_7cm_evidence_20260713",
        "report": True,
        "physical_status": "printed_successfully",
    },
    {
        "case_id": "sword_fox",
        "split": "test",
        "source": ROOT / "results/physical_print_soft_geometry_20260714/H02_sword_fox",
        "report": True,
    },
    {
        "case_id": "future_mech",
        "split": "test",
        "source": ROOT / "results/geometry_delivery_ablation_pilot_20260714/future_mech/full",
        "report": True,
    },
    {
        "case_id": "parrot_snail",
        "split": "test",
        "source": ROOT / "results/geometry_delivery_ablation_pilot_20260714/parrot_snail/full",
        "report": True,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    rows = []
    for spec in CASES:
        source = spec["source"]
        manifest_path = source / "MANIFEST.json"
        payload = json.loads(manifest_path.read_text())
        # The source manifest retains provenance paths to the preprocessed
        # silhouettes.  The actual fixed-wall task is the frozen pair written
        # beside each delivered candidate after canonical crop/scale/orientation
        # normalization; baselines must consume this exact pair.
        inputs = [
            source / "inputs" / "left_wall_x_target.png",
            source / "inputs" / "right_wall_y_target.png",
        ]
        stl = resolve(ROOT, payload["display_stl"])
        voxels = source / "selected_display_voxels256.npy"
        if spec["report"]:
            required = [*inputs, stl, voxels]
        else:
            required = [*inputs]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{spec['case_id']}: missing {missing}")
        row = {
            "case_id": spec["case_id"],
            "split": spec["split"],
            "report": spec["report"],
            "input_paths": [str(path) for path in inputs],
            "source_manifest": str(manifest_path),
            "shadowforge_voxels256": str(voxels) if voxels.is_file() else None,
            "shadowforge_stl": str(stl) if stl.is_file() else None,
            "shadowforge_candidate": payload.get("selected", {}).get("candidate"),
            "physical_status": spec.get("physical_status", "candidate"),
            "checksums": {str(path): sha256(path) for path in required},
        }
        rows.append(row)

    manifest = {
        "schema": "shadowforge_two_wall_physical_baseline_v1",
        "methods": [
            "Mitra-style carving (local reimplementation)",
            "Sadekar voxel-based (author code)",
            "Sadekar mesh-based (author code)",
            "ShadowForge",
        ],
        "protocol": {
            "task": "fixed orthographic two-wall, no-floor",
            "views": ["left wall / x-axis", "right wall / y-axis"],
            "target_resolution": [256, 256],
            "evaluation_grid": [256, 256, 256],
            "mirror_mode": True,
            "tsf_mode": False,
            "camera_angles_degrees": [0, 90, 180, 270],
            "common_evaluator": "hard two-axis occupancy projection at 256",
            "single_body": "one 6-connected occupied component",
            "watertight": "trimesh.is_watertight on the exported mesh",
            "saving": "1 - occupied(V256) / occupied(two-wall visual hull V256)",
            "sadekar_seeds": [17, 29, 43],
            "sadekar_voxel": {"iterations": 600, "lr": 0.01, "threshold": 0.05},
            "sadekar_mesh": {"iterations": 2000, "lr": 0.15},
            "coordinate_normalization": "one global transform selected only on the held-out coordinate-calibration task",
        },
        "cases": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "benchmark_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
