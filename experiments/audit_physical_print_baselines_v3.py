#!/usr/bin/env python3
"""Audit why a two-wall baseline result may not be protocol-comparable.

This script is intentionally diagnostic.  It does not rewrite any scores or
rerun a method.  It records the facts that must be resolved before the v2
physical-print baseline can be used as a controlled comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path("/home/lin/xvxinyao/ShadowForge")


def mesh_bounds(path: Path) -> dict:
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return {
        "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
        "extents": np.asarray(mesh.extents, dtype=float).tolist(),
        "watertight": bool(mesh.is_watertight),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=ROOT / "results/physical_print_controlled_baselines_v2_20260715",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.baseline_root
    manifest = json.loads((root / "benchmark_manifest.json").read_text())
    runs = json.loads((root / "baseline_run_summary.json").read_text())

    examples = []
    for run in runs.get("runs", []):
        if run.get("kind") != "mesh" or run.get("seed") != 17:
            continue
        mesh = Path(run["run_dir"]) / "sample_0_output.obj"
        if mesh.is_file():
            examples.append({"case_id": run["case_id"], "mesh": str(mesh), **mesh_bounds(mesh)})

    report = {
        "status": "NOT_A_CONTROLLED_COMPARISON",
        "baseline_root": str(root),
        "observed_protocol": manifest["protocol"],
        "findings": [
            {
                "id": "mitra_not_original",
                "severity": "blocking",
                "fact": "The local routine is a deterministic projection-preserving voxel carve. The original Mitra and Pauly implementation is not present in this workspace.",
                "required_action": "Report it only as Projection-only carving (local deterministic control), not as an exact reproduction of Mitra and Pauly.",
            },
            {
                "id": "sadekar_renderer_mismatch",
                "severity": "blocking",
                "fact": "Sadekar runs optimize the author renderer, while v2 scores hard axis-aligned occupancy projections.",
                "required_action": "Use one frozen two-view camera implementation for training, rendering, and final scoring, or report Sadekar native-renderer fidelity separately from common mesh QA.",
            },
            {
                "id": "sadekar_mesh_adaptive_scale",
                "severity": "blocking",
                "fact": "The v2 evaluator centers and rescales every exported mesh independently before voxelization.",
                "required_action": "Map every mesh with one frozen world extent and origin; do not fit a separate scale per output mesh.",
            },
            {
                "id": "coordinate_calibration_not_closed",
                "severity": "blocking",
                "fact": "The v2 coordinate transform is chosen on one calibration task but its recorded calibration fidelity is low.",
                "required_action": "Validate the frozen transform using native rendered masks before evaluating held-out tasks.",
            },
        ],
        "mesh_world_coordinate_examples": examples,
        "next_protocol": {
            "label": "fixed two-wall common protocol",
            "views": 2,
            "mirror_mode": False,
            "tsf_mode": False,
            "world_extent": 1.7,
            "camera": "one frozen orthographic implementation shared by all methods",
            "mesh_voxelization": "one fixed origin and voxel pitch for all methods",
            "report": "mean and standard deviation across frozen seeds for stochastic baselines",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
