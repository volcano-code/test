#!/usr/bin/env python3
"""Replay the archived fixed-three-view C48 -> V256 delivery path in isolation.

This entry point intentionally uses a copied pipeline snapshot and a frozen soft
artifact. It does not import or modify the active ShadowForge pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
HERE = ROOT / "experiments/fixed_surface_xyz_isolated_20260726"
PIPELINE = HERE / "shadowforge_pipeline/unified_run.py"
PYTHON = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", default="connected_01")
    parser.add_argument("--soft-artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    benchmark = json.loads(args.manifest.read_text(encoding="utf-8"))
    case = next(row for row in benchmark["cases"] if row["case_id"] == args.case)
    args.output.mkdir(parents=True, exist_ok=True)

    command = [
        str(PYTHON),
        str(PIPELINE),
        "--silhouettes",
        *case["target_paths_xyz"],
        "--output",
        str(args.output),
        "--branch",
        "soft-geometry",
        "--tsf",
        "1",
        "--mr",
        "1",
        "--soft-artifact-dir",
        str(args.soft_artifact_dir),
        "--soft-resolution",
        "256",
        "--soft-control-res",
        "0",
        "--soft-proposal-mode",
        "proposal",
        "--core-resolution",
        "48",
        "--soft-seed-thresholds",
        "0.05,0.10,0.15,0.20,0.25,0.35,0.45,0.55,0.65",
        "--bridge-radius",
        "2",
        "--bridge-passes",
        "48",
        "--disable-guarded-carve",
        "--delivery-shell-voxels",
        "2",
        "--min-iou",
        "0.95",
        "--min-view-iou",
        "0.90",
        "--min-saving",
        "0.000001",
        "--max-saving",
        "0.85",
        "--selection-iou-tolerance",
        "0.01",
        "--selection-policy",
        "material",
        "--screen",
        "skip",
        "--save-candidate-evidence",
        "--device",
        "cuda:0",
    ]
    (args.output / "REPLAY_COMMAND.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(command, cwd=ROOT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
