#!/usr/bin/env python3
"""Regenerate the controlled benchmark's ShadowForge row with the authoritative pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
BENCH = ROOT / "results/controlled_benchmark_geometry_only_20260710"
MANIFEST = BENCH / "benchmark_manifest.json"
OUT = BENCH / "shadowforge_authoritative_fixed_shell2"
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]
from shadowforge_pipeline.run import mesh_for_print


def main() -> None:
    source = json.loads(MANIFEST.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in source["cases"]:
        case_out = OUT / case["case_id"]
        cmd = [
            sys.executable, str(ROOT / "shadowforge_pipeline/run.py"),
            "--silhouettes", *case["input_paths"],
            "--output", str(case_out),
            "--branches", "fixed", "--soft", "off",
            "--screen", "fem_only", "--device", "cpu",
            "--cert-resolution", "48", "--delivery-resolution", "256",
            "--min-iou", "0.95", "--min-saving", "0.40", "--max-saving", "0.85",
            "--selection-saving-target", "0.55", "--shell-voxels", "2",
            "--allow-internal-cavity-mesh-parts",
        ]
        print("RUN", case["case_id"], flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        case_out.mkdir(parents=True, exist_ok=True)
        (case_out / "RUN_OUTPUT.txt").write_text(proc.stdout, encoding="utf-8")
        manifest_path = case_out / "MANIFEST.json"
        row = {"case_id": case["case_id"], "split": case["split"], "returncode": proc.returncode,
               "manifest": str(manifest_path)}
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text())
            delivery = payload.get("delivery") or {}
            geometry_stl = None
            delivery_path = Path(delivery["artifact"]) if delivery.get("artifact") else None
            if delivery_path and delivery_path.exists() and delivery.get("watertight"):
                mesh = mesh_for_print(np.load(delivery_path).astype(bool), 100.0)
                geometry_stl = case_out / "shadowforge_geometry_only.stl"
                mesh.export(geometry_stl)
            row.update({
                "status": payload.get("status"), "defer_reason": payload.get("defer_reason"),
                "selected_core": (payload.get("selected_core") or {}).get("candidate_id"),
                "delivery": delivery.get("artifact"), "stl": str(geometry_stl) if geometry_stl else (delivery.get("mesh") or {}).get("artifact"),
                "mean_iou": delivery.get("mean_iou"), "min_iou": delivery.get("min_iou"),
                "saving": delivery.get("material_saved_vs_hull"), "components6": delivery.get("component_count_6"),
                "watertight": delivery.get("watertight"), "mesh_parts": delivery.get("mesh_parts"),
                "wall": delivery.get("erosion_survival"),
            })
        rows.append(row)
        (OUT / "SUMMARY.partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print("DONE", json.dumps(row), flush=True)
    (OUT / "SUMMARY.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
