#!/usr/bin/env python3
"""Run the current unified ShadowForge pipeline on the frozen 12-task benchmark."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
PY = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")
SOURCE = ROOT / "results/mitra_stratified_low6_high6_fixed3_20260716/benchmark_manifest.json"
OUT = ROOT / "results/mitra_stratified_low6_high6_shadowforge_unified_20260718"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in source["cases"]:
        case_out = OUT / case["case_id"]
        manifest = case_out / "MANIFEST.json"
        if manifest.is_file():
            payload = json.loads(manifest.read_text())
            selected = payload.get("selected") or {}
            rows.append({
                "case_id": case["case_id"], "status": "reused", "manifest": str(manifest), "selected": selected,
                "mean_iou": selected.get("mean_iou"), "mean_dice": selected.get("mean_dice"),
                "saving": selected.get("saving"), "components6": selected.get("components"),
                "watertight": selected.get("watertight"),
            })
            continue
        command = [
            str(PY), str(ROOT / "shadowforge_pipeline/unified_run.py"),
            "--silhouettes", *case["target_paths_xyz"],
            "--output", str(case_out), "--branch", "soft-geometry",
            "--tsf", "1", "--mr", "1", "--soft-resolution", "256",
            "--soft-control-res", "0", "--soft-iterations", "2000",
            "--soft-seed", "42", "--core-resolution", "48",
            "--delivery-shell-voxels", "2", "--min-iou", "0.95",
            "--min-view-iou", "0.90", "--min-saving", "0.20",
            "--max-saving", "0.85", "--selection-iou-tolerance", "0.01",
            "--selection-policy", "material", "--screen", "skip",
            "--save-candidate-evidence", "--device", "cuda:0",
        ]
        case_out.mkdir(parents=True, exist_ok=True)
        log_path = case_out / "RUN_OUTPUT.txt"
        with log_path.open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = {"case_id": case["case_id"], "returncode": completed.returncode, "manifest": str(manifest), "log": str(log_path)}
        if manifest.is_file():
            payload = json.loads(manifest.read_text())
            selected = payload.get("selected") or {}
            row.update({
                "status": payload.get("status"), "selected": selected, "display_stl": payload.get("display_stl"),
                "mean_iou": selected.get("mean_iou"), "mean_dice": selected.get("mean_dice"),
                "saving": selected.get("saving"), "components6": selected.get("components"),
                "watertight": selected.get("watertight"),
            })
        rows.append(row)
        (OUT / "SUMMARY.partial.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    (OUT / "SUMMARY.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
