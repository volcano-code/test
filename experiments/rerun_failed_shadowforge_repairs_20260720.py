#!/usr/bin/env python3
"""Expand geometry repair candidates for failed frozen ShadowForge tasks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
PY = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")
SOURCE = ROOT / "results/mitra_stratified_low6_high6_fixed3_20260716/benchmark_manifest.json"
BASE = ROOT / "results/mitra_stratified_low6_high6_shadowforge_unified_20260718"
OUT = ROOT / "results/mitra_stratified_low6_high6_shadowforge_repair_expanded_20260720"


def main() -> None:
    cases = json.loads(SOURCE.read_text())["cases"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        base_manifest = BASE / case["case_id"] / "MANIFEST.json"
        if not base_manifest.is_file():
            continue
        base = json.loads(base_manifest.read_text())
        if base.get("status") != "BEST_EFFORT":
            continue
        artifact = BASE / case["case_id"] / "soft_training"
        out = OUT / case["case_id"]
        manifest = out / "MANIFEST.json"
        if manifest.is_file():
            result = json.loads(manifest.read_text())
            rows.append({"case_id": case["case_id"], "status": "reused", "selected": result.get("selected", {})})
            continue
        command = [
            str(PY), str(ROOT / "shadowforge_pipeline/unified_run.py"),
            "--silhouettes", *case["target_paths_xyz"],
            "--output", str(out), "--branch", "soft-geometry",
            "--tsf", "1", "--mr", "1", "--soft-resolution", "256",
            "--soft-artifact-dir", str(artifact), "--core-resolution", "48",
            "--delivery-shell-voxels", "2", "--soft-seed-thresholds",
            "0.05,0.10,0.15,0.20,0.25,0.35,0.45,0.55,0.65",
            "--bridge-radius", "2", "--bridge-passes", "48",
            "--min-iou", "0.95", "--min-view-iou", "0.90",
            "--min-saving", "0.000001", "--max-saving", "0.85",
            "--selection-iou-tolerance", "0.01", "--selection-policy", "material",
            "--screen", "skip", "--save-candidate-evidence", "--device", "cuda:0",
        ]
        out.mkdir(parents=True, exist_ok=True)
        with (out / "RUN_OUTPUT.txt").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads(manifest.read_text()) if manifest.is_file() else {}
        row = {"case_id": case["case_id"], "returncode": completed.returncode,
               "status": result.get("status", "ERROR"), "selected": result.get("selected", {})}
        rows.append(row)
        (OUT / "SUMMARY.partial.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    (OUT / "SUMMARY.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
