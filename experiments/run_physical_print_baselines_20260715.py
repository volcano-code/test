#!/usr/bin/env python3
"""Run Sadekar author baselines for the frozen two-wall benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/physical_print_controlled_baselines_v2_20260715"
MANIFEST = OUT / "benchmark_manifest.json"
SAR = Path("/data/xvxinyao/ShadowArt-Revisited")
PY = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")
TAG = "shadowforge_two_wall_physical_v2_20260715"


def seeded_command(seed: int, args: list[str]) -> list[str]:
    code = (
        "import os,random,runpy,numpy as np,torch;"
        f"seed={seed};os.environ['PYTHONHASHSEED']=str(seed);"
        "random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);"
        "torch.cuda.manual_seed_all(seed);"
        "runpy.run_path('val.py',run_name='__main__')"
    )
    return [str(PY), "-c", code, *args]


def run_one(kind: str, case: dict, seed: int, force: bool) -> dict:
    case_id = case["case_id"]
    inputs = case["input_paths"]
    sub = f"{TAG}/{kind}/{case_id}/seed_{seed}"
    run_dir = SAR / f"{kind}_results" / sub
    expected = run_dir / ("sample_0_final_voxels.npy" if kind == "voxel" else "sample_0_output.obj")
    log = OUT / "logs" / case_id / f"sadekar_{kind}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if expected.is_file() and not force:
        return {"case_id": case_id, "kind": kind, "seed": seed, "status": "reused", "run_dir": str(run_dir)}
    run_dir.mkdir(parents=True, exist_ok=True)
    if kind == "voxel":
        args = [
            "cuda:0", sub, "600", "0.01", "-swt", "10.0", "-l1wt", "10.0",
            "-mwt", "0.0", "-th", "0.05", "-zd", "1.7", "-imsz", "256",
            "-mr", "true", "-tsf", "false", "-sdlist", *inputs,
        ]
    else:
        args = [
            "cuda:0", sub, "2000", "0.15", "0", "-swt", "1.6", "-l1wt", "1.6",
            "-mwt", "0.0", "-i2vwt", "0.0", "-ewt", "1.6", "-nwt", "0.6",
            "-lwt", "1.2", "-imsz", "256", "-mr", "true", "-tsf", "false",
            "-sdlist", *inputs,
        ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    started = time.time()
    with log.open("w") as handle:
        subprocess.run(
            seeded_command(seed, args), cwd=str(SAR / kind), env=env,
            stdout=handle, stderr=subprocess.STDOUT, check=True,
        )
    if not expected.is_file():
        raise RuntimeError(f"missing expected output: {expected}")
    return {
        "case_id": case_id, "kind": kind, "seed": seed, "status": "completed",
        "seconds": time.time() - started, "run_dir": str(run_dir), "log": str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--single-seed", type=int)
    args = parser.parse_args()
    payload = json.loads(MANIFEST.read_text())
    seeds = [args.single_seed] if args.single_seed is not None else payload["protocol"]["sadekar_seeds"]
    summary = {"manifest": str(MANIFEST), "started": time.time(), "runs": []}
    for case in payload["cases"]:
        for seed in seeds:
            for kind in ("voxel", "mesh"):
                try:
                    record = run_one(kind, case, seed, args.force)
                except Exception as exc:
                    record = {"case_id": case["case_id"], "kind": kind, "seed": seed, "status": "failed", "error": str(exc)}
                    summary["runs"].append(record)
                    (OUT / "baseline_run_summary.json").write_text(json.dumps(summary, indent=2))
                    raise
                summary["runs"].append(record)
                (OUT / "baseline_run_summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(record), flush=True)
    summary["finished"] = time.time()
    summary["elapsed_s"] = summary["finished"] - summary["started"]
    (OUT / "baseline_run_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
