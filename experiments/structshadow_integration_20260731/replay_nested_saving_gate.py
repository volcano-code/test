#!/usr/bin/env python3
"""Replay the corrected nested material gate from cached geometry evidence."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_anime3_integrated as integrated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", choices=("fixed", "free"), default="free")
    parser.add_argument("--saving-min", type=float, default=0.30)
    parser.add_argument("--saving-max", type=float, default=0.50)
    parser.add_argument("--saving-target", type=float, default=0.40)
    parser.add_argument("--fidelity-band", type=float, default=0.001)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--evaluation-resolution", type=int, default=256)
    parser.add_argument("--print-mm", type=float, default=210.0)
    parser.add_argument("--smooth-sigma", type=float, default=0.6)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    return integrated.json_ready(value)


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_results = json.loads((source / "results.json").read_text())
    branch_name = f"pcraft_{args.calibration}"
    branch = source_results[branch_name]
    calibration = branch["calibration"]
    calibration["axis_for_view"] = {
        int(key): value for key, value in calibration["axis_for_view"].items()
    }
    calibration["ops"] = {
        int(key): value for key, value in calibration["ops"].items()
    }

    input_paths = [Path(path) for path in source_results["inputs"]]
    geometry_resolution = int(source_results["protocol"]["geometry_resolution"])
    masks = integrated.load_masks(input_paths, geometry_resolution)
    anchor_path = (
        source
        / "candidates"
        / f"{branch_name}_minimal_watertight"
        / "volume.npy"
    )
    owner_path = source / "candidates" / f"{branch_name}_repaired" / "volume.npy"
    anchor = np.load(anchor_path).astype(bool)
    owner = np.load(owner_path).astype(bool)
    hull_voxels = int(branch["hull_voxels"])

    trajectory, trajectory_metadata = integrated.build_controlled_material_trajectory(
        anchor,
        owner,
        hull_voxels,
        args.saving_min,
        args.max_steps,
    )
    states = {"anchor": anchor, **trajectory}
    evaluation_args = SimpleNamespace(
        evaluation_resolution=int(args.evaluation_resolution),
        print_mm=float(args.print_mm),
        smooth_sigma=float(args.smooth_sigma),
    )

    records = []
    for name, volume in states.items():
        metrics = integrated.evaluate_candidate(
            name,
            volume,
            hull_voxels,
            masks,
            calibration,
            output,
            evaluation_args,
        )
        record = {
            "state": name,
            "occupied_voxels": int(metrics["occupied_voxels"]),
            "saving_vs_hull": float(metrics["saving_vs_hull"]),
            "mean_iou": float(metrics["projection"]["mean_iou"]),
            "min_iou": float(metrics["projection"]["min_iou"]),
            "mean_dice": float(metrics["projection"]["mean_dice"]),
            "component_count_6": int(metrics["component_count_6"]),
            "watertight": bool(metrics["mesh"]["watertight"]),
            "winding_consistent": bool(metrics["mesh"]["winding_consistent"]),
            "mesh_path": metrics["mesh"]["path"],
        }
        record["geometry_eligible"] = bool(
            record["component_count_6"] == 1
            and record["watertight"]
            and record["winding_consistent"]
        )
        records.append(record)

    anchor_iou = records[0]["mean_iou"]
    eligible = [
        record
        for record in records
        if record["geometry_eligible"]
        and args.saving_min <= record["saving_vs_hull"] <= args.saving_max
        and record["mean_iou"] >= anchor_iou - args.fidelity_band
    ]
    if not eligible:
        raise RuntimeError("No nested state satisfies geometry, saving, and fidelity gates")
    selected = min(
        eligible,
        key=lambda record: (
            abs(record["saving_vs_hull"] - args.saving_target),
            -record["min_iou"],
            -record["mean_iou"],
        ),
    )
    selected_stl = output / "selected_delivery.stl"
    shutil.copy2(selected["mesh_path"], selected_stl)
    normalized_mesh = trimesh.load_mesh(selected_stl, force="mesh", process=True)
    normalized_parts = normalized_mesh.split(only_watertight=False)
    normalized_mesh_audit = {
        "path": str(selected_stl),
        "vertices": int(len(normalized_mesh.vertices)),
        "faces": int(len(normalized_mesh.faces)),
        "body_count": int(len(normalized_parts)),
        "watertight": bool(normalized_mesh.is_watertight),
        "winding_consistent": bool(normalized_mesh.is_winding_consistent),
        "volume": (
            float(abs(normalized_mesh.volume))
            if normalized_mesh.is_volume
            else None
        ),
    }
    if not (
        normalized_mesh_audit["body_count"] == 1
        and normalized_mesh_audit["watertight"]
        and normalized_mesh_audit["winding_consistent"]
    ):
        raise RuntimeError(
            "Selected STL failed normalized external topology validation: "
            f"{normalized_mesh_audit}"
        )

    result = {
        "schema": "nested_saving_gate_replay_v1",
        "source_results": str(source / "results.json"),
        "source_anchor": str(anchor_path),
        "source_owner": str(owner_path),
        "calibration": args.calibration,
        "saving_gate": {
            "min": float(args.saving_min),
            "max": float(args.saving_max),
            "target": float(args.saving_target),
        },
        "fidelity_band": float(args.fidelity_band),
        "trajectory_metadata": trajectory_metadata,
        "states": records,
        "selected": selected,
        "selected_normalized_mesh": normalized_mesh_audit,
    }
    results_path = output / "results.json"
    results_path.write_text(
        json.dumps(json_ready(result), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "results": str(results_path),
        "selected_state": selected["state"],
        "selected_stl": str(selected_stl),
        "selected_source_stl": selected["mesh_path"],
        "saving_vs_hull": selected["saving_vs_hull"],
        "mean_iou": selected["mean_iou"],
        "min_iou": selected["min_iou"],
        "mean_dice": selected["mean_dice"],
        "connected": normalized_mesh_audit["body_count"] == 1,
        "watertight": normalized_mesh_audit["watertight"],
        "winding_consistent": normalized_mesh_audit["winding_consistent"],
        "normalized_mesh": normalized_mesh_audit,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
