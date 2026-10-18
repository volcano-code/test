#!/usr/bin/env python3
"""Run a controlled voxel re-parameterization pilot for mesh8.1.7.

The legacy mesh8.1.7 optimizer deforms an icosphere with PyTorch3D.  This
pilot keeps one archived mesh8 target set and camera flags, but optimizes a
voxel occupancy field through mesh_combined2.  It is deliberately a separate
experiment: it must not be described as an equivalent replacement unless the
same-protocol re-rendered metrics support that claim.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
MESH8_CASE = Path(
    "/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/"
    "shadow-project2/mesh/mesh8.1.7/35.5"
)
VOXEL_SOURCE = Path(
    "/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/"
    "shadow-project2/mesh/mesh_combined2"
)
OUT = Path(
    "/data/xvxinyao/xvxinyao/ShadowForge_runs/"
    "mesh8_voxel_reparameterization_pilot_20260711"
)
VOXEL_PYTHON = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Start voxel optimization after writing provenance.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.007)
    args = parser.parse_args()

    targets = [MESH8_CASE / f"sample_1_gt_view_{i}.png" for i in range(6)]
    missing = [str(path) for path in targets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing archived mesh8 targets: {missing}")
    if not (VOXEL_SOURCE / "train.py").is_file():
        raise FileNotFoundError(f"Missing voxel trainer: {VOXEL_SOURCE}")
    if not VOXEL_PYTHON.is_file():
        raise FileNotFoundError(f"Missing voxel environment: {VOXEL_PYTHON}")

    OUT.mkdir(parents=True, exist_ok=True)
    provenance = {
        "experiment": "mesh8_voxel_reparameterization_pilot_v1",
        "legacy_representation": "PyTorch3D icosphere vertex deformation",
        "voxel_representation": "learned occupancy VolumeModel",
        "legacy_case_dir": str(MESH8_CASE),
        "legacy_mesh": str(MESH8_CASE / "sample_1_output_cleaned.obj"),
        "targets": [str(path) for path in targets],
        "camera_protocol": {"camera": "ortho", "mirror": 1, "tsf": 0, "tsf4": 0},
        "train_resolution": 48,
        "delivery_resolution": 256,
        "important_limit": (
            "This is a re-parameterization experiment, not an equivalence claim. "
            "Mesh spectral/edge regularizers are replaced by voxel connectivity, "
            "minimum-thickness, and morphology regularizers."
        ),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    command = [
        str(VOXEL_PYTHON), str(VOXEL_SOURCE / "train.py"), args.device,
        str(OUT / "voxel_run"), str(args.iterations), str(args.lr), "1",
        "-sdlist", *map(str, targets),
        "-mr", "1", "-tsf", "0", "-tsf4", "0", "-cam", "ortho",
        "-imsz", "256", "-res", "48", "-eres", "256", "-vh", "1",
        "-zdist", "2.5", "--post_export_bundle", "1",
        "--enable_sparse_teacher", "0", "--ablation_profile", "conservative",
    ]
    (OUT / "command.txt").write_text(" ".join(command) + "\n")
    print(f"Prepared: {OUT}")
    print("Command:", " ".join(command))
    if args.run:
        return subprocess.call(command, cwd=VOXEL_SOURCE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
