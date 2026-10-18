#!/usr/bin/env python3
"""Rerun only Sadekar mesh/voxel baselines for Fig. 3(a)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SAR = Path("/data/xvxinyao/ShadowArt-Revisited")
PY = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")
DEVICE = "cuda:0"
IMG_SIZE = 256
TAG = "fig3_a_rerun_20260710"
CASE = "case4B_shrimp_turtle_owl"
OUT = ROOT / "results" / TAG

INPUTS = [
    ROOT / "results/nsa_style_fixed_required_10cm_256/case4_white_paper_sweep/case4B_shrimp_turtle_owl/inputs/input0_sea_shrimp.png",
    ROOT / "results/nsa_style_fixed_required_10cm_256/case4_white_paper_sweep/case4B_shrimp_turtle_owl/inputs/input1_reptiles_turtle.png",
    ROOT / "results/nsa_style_fixed_required_10cm_256/case4_white_paper_sweep/case4B_shrimp_turtle_owl/inputs/input2_birds_owl.png",
]


def run_one(kind: str, cmd: list[str], cwd: Path, expected: Path) -> dict:
    log = OUT / f"{kind}.log"
    t0 = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        subprocess.run(cmd, cwd=str(cwd), check=True, stdout=f, stderr=subprocess.STDOUT)
    return {
        "kind": kind,
        "seconds": time.time() - t0,
        "log": str(log),
        "expected_output": str(expected),
        "exists": expected.exists(),
    }


def main() -> None:
    for p in INPUTS:
        if not p.exists():
            raise FileNotFoundError(p)

    paths = [str(p) for p in INPUTS]
    voxel_sub = f"{TAG}/voxel/{CASE}"
    mesh_sub = f"{TAG}/mesh/{CASE}"
    voxel_dir = SAR / "voxel_results" / voxel_sub
    mesh_dir = SAR / "mesh_results" / mesh_sub
    voxel_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    voxel_cmd = [
        str(PY), "val.py", DEVICE, voxel_sub, "600", "0.01",
        "-swt", "10.0", "-l1wt", "10.0", "-mwt", "0.0",
        "-th", "0.05", "-zd", "1.7", "-imsz", str(IMG_SIZE),
        "-mr", "true", "-tsf", "true", "-sdlist", *paths,
    ]
    mesh_cmd = [
        str(PY), "val.py", DEVICE, mesh_sub, "600", "0.01", "0",
        "-swt", "1.6", "-l1wt", "1.6", "-mwt", "0.0",
        "-ewt", "1.2", "-nwt", "0.03", "-lwt", "1.2",
        "-imsz", str(IMG_SIZE), "-mr", "true", "-tsf", "true",
        "-sdlist", *paths,
    ]

    summary = {
        "tag": TAG,
        "case": CASE,
        "inputs": [str(p) for p in INPUTS],
        "runs": [
            run_one("sadekar_voxel", voxel_cmd, SAR / "voxel", voxel_dir / "sample_0_final_voxels.npy"),
            run_one("sadekar_mesh", mesh_cmd, SAR / "mesh", mesh_dir / "sample_0_output.obj"),
        ],
        "voxel_dir": str(voxel_dir),
        "mesh_dir": str(mesh_dir),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
