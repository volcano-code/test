#!/usr/bin/env python3
"""Run the isolated three-fixed-surface extension without touching two-wall code."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIPELINE = HERE / "shadowforge_pipeline_snapshot" / "unified_run.py"
DEFAULT_PYTHON = Path(
    "/home/lin/anaconda3/envs/shadowart-revisited/bin/python"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--silhouettes",
        nargs=3,
        type=Path,
        metavar=("VIEW_X", "VIEW_Y", "VIEW_Z"),
        required=True,
        help="Exactly three fixed-surface target silhouettes in X/Y/Z order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--soft-artifact-dir", type=Path)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mr", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--screen",
        choices=("skip", "fem_only", "pino_rollback_fem"),
        default="fem_only",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not PIPELINE.is_file():
        raise FileNotFoundError(f"Isolated pipeline snapshot not found: {PIPELINE}")
    missing = [str(path) for path in args.silhouettes if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing silhouettes: " + ", ".join(missing))

    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python),
        "-u",
        str(PIPELINE),
        "--silhouettes",
        *(str(path.resolve()) for path in args.silhouettes),
        "--output",
        str(args.output.resolve()),
        "--branch",
        "soft-geometry",
        "--tsf",
        "1",
        "--mr",
        str(args.mr),
        "--soft-resolution",
        "256",
        "--soft-control-res",
        "0",
        "--soft-proposal-mode",
        "proposal",
        "--soft-iterations",
        str(args.iterations),
        "--soft-lr",
        "0.007",
        "--soft-save-every",
        "100",
        "--soft-seed",
        str(args.seed),
        "--core-resolution",
        "48",
        # One seed is sufficient for the requested protocol smoke test.
        "--soft-seed-thresholds",
        "0.15",
        "--bridge-radius",
        "1",
        "--bridge-passes",
        "16",
        "--delivery-shell-voxels",
        "2",
        # Promote only the selected C48 trajectory state to V256 delivery.
        "--delivery-shortlist-size",
        "1",
        "--delivery-bridge-passes",
        "8",
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
        args.screen,
        "--save-candidate-evidence",
        "--export-best-rejected",
        "--device",
        args.device,
    ]
    if args.soft_artifact_dir is not None:
        command.extend(
            ["--soft-artifact-dir", str(args.soft_artifact_dir.resolve())]
        )

    metadata = {
        "scope": "isolated fixed-surface X/Y/Z extension",
        "pipeline_snapshot": str(PIPELINE),
        "production_two_wall_modified": False,
        "v256_geometry_owner": False,
        "delivery_shortlist_size": 1,
        "command": command,
    }
    (args.output / "RUN_COMMAND.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
