#!/usr/bin/env python3
"""Run the formal ShadowForge pipeline on a frozen fixed3/mr1 benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
PY = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    benchmark = json.loads(args.manifest.read_text())
    protocol = benchmark["protocol"]
    if int(protocol.get("views", 0)) != 3 or int(protocol.get("mr", 0)) != 1:
        raise ValueError("This runner requires exactly three silhouettes and mr=1")
    cases = [row for row in benchmark["cases"] if not args.case or row["case_id"] in args.case]
    seeds = args.seed or protocol.get("seeds", [17, 29, 43])
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        for seed in seeds:
            run = args.output / case["case_id"] / f"seed_{seed}"
            manifest = run / "MANIFEST.json"
            if manifest.is_file() and not args.force:
                payload = json.loads(manifest.read_text())
                rows.append({"case_id": case["case_id"], "seed": seed, "status": "reused", "manifest": str(manifest), "selected": payload.get("selected")})
                continue
            run.mkdir(parents=True, exist_ok=True)
            command = [
                str(PY), str(ROOT / "shadowforge_pipeline/unified_run.py"),
                "--silhouettes", *case["target_paths_xyz"],
                "--output", str(run), "--branch", "soft-geometry",
                "--tsf", "1", "--mr", "1",
                "--soft-resolution", "256", "--soft-control-res", "0",
                "--soft-proposal-mode", "proposal", "--soft-iterations", str(args.iterations),
                "--soft-lr", "0.007", "--soft-save-every", "100", "--soft-seed", str(seed),
                "--soft-export-final-step", "--core-resolution", "48",
                "--v256-geometry-owner",
                "--soft-seed-thresholds", "0.05,0.10,0.15,0.20,0.25,0.35,0.45,0.55,0.65",
                "--bridge-radius", "2", "--bridge-passes", "48",
                "--disable-guarded-carve",
                "--delivery-shell-voxels", "2", "--min-iou", "0.95",
                "--min-view-iou", "0.90", "--min-saving", "0.000001", "--max-saving", "0.85",
                "--selection-iou-tolerance", "0.01", "--selection-policy", "material",
                "--screen", "skip", "--save-candidate-evidence", "--device", "cuda:0",
            ]
            with (run / "RUN_OUTPUT.txt").open("w") as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            payload = json.loads(manifest.read_text()) if manifest.is_file() else {}
            row = {"case_id": case["case_id"], "seed": seed, "returncode": completed.returncode,
                   "status": payload.get("status", "ERROR"), "manifest": str(manifest),
                   "selected": payload.get("selected")}
            rows.append(row)
            (args.output / "SUMMARY.partial.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    (args.output / "SUMMARY.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
