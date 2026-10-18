#!/usr/bin/env python3
"""Run frozen Mitra and author-code Sadekar baselines without touching PINO."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/controlled_benchmark_geometry_only_20260710"
MANIFEST = OUT / "benchmark_manifest.json"
SAR = Path("/data/xvxinyao/ShadowArt-Revisited")
PY = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")
TAG = "shadowforge_controlled_20260710"


def pino_active() -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "list-units", "--type=service", "--state=running", "--no-legend"],
        text=True, capture_output=True, check=False,
    )
    return any("pino" in line.lower() for line in result.stdout.splitlines())


def seeded_command(seed: int, args: list[str]) -> list[str]:
    code = (
        "import os,random,runpy,numpy as np,torch;"
        f"seed={seed};os.environ['PYTHONHASHSEED']=str(seed);"
        "random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);"
        "torch.cuda.manual_seed_all(seed);"
        "runpy.run_path('val.py',run_name='__main__')"
    )
    return [str(PY), "-c", code, *args]


def run_command(kind: str, case: dict, seed: int, force: bool) -> dict:
    case_id = case["case_id"]
    paths = case["input_paths"]
    if kind == "voxel":
        sub = f"{TAG}/voxel/{case_id}/seed_{seed}"
        run_dir = SAR / "voxel_results" / sub
        expected = run_dir / "sample_0_final_voxels.npy"
        args = [
            "cuda:0", sub, "600", "0.01", "-swt", "10.0", "-l1wt", "10.0",
            "-mwt", "0.0", "-th", "0.05", "-zd", "1.7", "-imsz", "256",
            "-mr", "true", "-tsf", "true", "-sdlist", *paths,
        ]
    else:
        sub = f"{TAG}/mesh/{case_id}/seed_{seed}"
        run_dir = SAR / "mesh_results" / sub
        expected = run_dir / "sample_0_output.obj"
        args = [
            "cuda:0", sub, "2000", "0.15", "0", "-swt", "1.6", "-l1wt", "1.6",
            "-mwt", "0.0", "-i2vwt", "0.0", "-ewt", "1.6", "-nwt", "0.6",
            "-lwt", "1.2", "-imsz", "256", "-mr", "true", "-tsf", "true",
            "-sdlist", *paths,
        ]
    log = OUT / "logs" / case_id / f"sadekar_{kind}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if expected.exists() and not force:
        return {"kind": kind, "seed": seed, "status": "reused", "run_dir": str(run_dir), "log": str(log)}
    if pino_active():
        raise RuntimeError("A PINO service is active; refusing to start a competing GPU run")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    started = time.time()
    with log.open("w") as f:
        subprocess.run(
            seeded_command(seed, args), cwd=str(SAR / kind), env=env,
            stdout=f, stderr=subprocess.STDOUT, check=True,
        )
    if not expected.exists():
        raise RuntimeError(f"missing expected output: {expected}")
    return {
        "kind": kind, "seed": seed, "status": "completed",
        "seconds": time.time() - started, "run_dir": str(run_dir), "log": str(log),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split", choices=["all", "validation", "test"], default="all")
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    seeds = manifest["protocol"]["sadekar_seeds"]
    cases = [c for c in manifest["cases"] if args.split == "all" or c["split"] == args.split]
    summary = {"manifest": str(MANIFEST), "started": time.time(), "runs": []}
    for case in cases:
        for seed in seeds:
            for kind in ("voxel", "mesh"):
                rec = {"case_id": case["case_id"], "split": case["split"]}
                try:
                    rec.update(run_command(kind, case, seed, args.force))
                except Exception as exc:
                    rec.update({"kind": kind, "seed": seed, "status": "failed", "error": str(exc)})
                    summary["runs"].append(rec)
                    (OUT / "baseline_run_summary.json").write_text(json.dumps(summary, indent=2))
                    raise
                summary["runs"].append(rec)
                (OUT / "baseline_run_summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(rec), flush=True)
    summary["finished"] = time.time()
    summary["elapsed_s"] = summary["finished"] - summary["started"]
    (OUT / "baseline_run_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
