#!/usr/bin/env python3
"""Run authoritative ShadowForge on the same frozen 12-task TFS benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/mitra_stratified_low6_high6_fixed3_20260716/benchmark_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/mitra_stratified_low6_high6_shadowforge_20260717",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--soft-iters", type=int, default=2000)
    args = parser.parse_args()

    source = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in source["cases"]:
        case_out = args.output / case["case_id"]
        manifest_path = case_out / "MANIFEST.json"
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text())
            if payload.get("delivery", {}).get("artifact"):
                rows.append({"case_id": case["case_id"], "status": "reused", "manifest": str(manifest_path)})
                continue
        command = [
            sys.executable,
            str(ROOT / "shadowforge_pipeline/run.py"),
            "--silhouettes", *case["target_paths_xyz"],
            "--output", str(case_out),
            "--branches", "fixed",
            "--soft", "train",
            "--soft-iters", str(args.soft_iters),
            "--soft-resolution", "256",
            "--cert-resolution", "48",
            "--delivery-resolution", "256",
            "--screen", "fem_only",
            "--device", args.device,
            "--min-iou", "0.95",
            "--min-saving", "0.20",
            "--max-saving", "0.85",
            "--selection-saving-target", "0.45",
            "--shell-voxels", "2",
            "--allow-internal-cavity-mesh-parts",
        ]
        case_out.mkdir(parents=True, exist_ok=True)
        log_path = case_out / "RUN_OUTPUT.txt"
        with log_path.open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        row = {"case_id": case["case_id"], "returncode": completed.returncode, "manifest": str(manifest_path), "log": str(log_path)}
        if manifest_path.is_file():
            payload = json.loads(manifest_path.read_text())
            delivery = payload.get("delivery") or {}
            row.update({
                "status": payload.get("status"),
                "artifact": delivery.get("artifact"),
                "stl": (delivery.get("mesh") or {}).get("artifact"),
                "mean_iou": delivery.get("mean_iou"),
                "mean_dice": delivery.get("mean_dice"),
                "saving": delivery.get("material_saved_vs_hull"),
                "components6": delivery.get("component_count_6"),
                "watertight": delivery.get("watertight"),
            })
        rows.append(row)
        (args.output / "SUMMARY.partial.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
    (args.output / "SUMMARY.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
