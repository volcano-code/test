#!/usr/bin/env python3
"""Probe fixed geometry-only replacements for the unsuitable spider triple."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = ROOT / "results/nsa_style_fixed_required_10cm_256/MANIFEST.json"
OUT = ROOT / "results/controlled_benchmark_geometry_only_20260710/replacement_probe"
NAMES = ["case1B_eagle_frog_trex", "case2B_bat_trex_snail", "case3B_lobster_squid_ladybug"]

payload = json.loads(SRC.read_text())
index = {model["example"]: model for group in payload["cases"] for model in group["models"]}
OUT.mkdir(parents=True, exist_ok=True)
rows = []
for name in NAMES:
    model = index[name]
    inputs = sorted(Path(model["voxels"]).parent.joinpath("inputs").glob("input*.png"))
    out = OUT / name
    cmd = [
        sys.executable, str(ROOT / "shadowforge_pipeline/run.py"), "--silhouettes", *map(str, inputs),
        "--output", str(out), "--branches", "fixed", "--soft", "off", "--screen", "fem_only", "--device", "cpu",
        "--cert-resolution", "48", "--delivery-resolution", "256", "--shell-voxels", "2",
        "--min-iou", "0.95", "--min-saving", "0.40", "--max-saving", "0.85",
        "--selection-saving-target", "0.55", "--allow-internal-cavity-mesh-parts",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "RUN_OUTPUT.txt").write_text(proc.stdout)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    delivery = manifest.get("delivery") or {}
    rows.append({"case_id": name, "returncode": proc.returncode, "status": manifest.get("status"),
                 "saving": delivery.get("material_saved_vs_hull"), "iou": delivery.get("mean_iou"),
                 "min_iou": delivery.get("min_iou"), "watertight": delivery.get("watertight"),
                 "components": delivery.get("component_count_6"), "mesh_parts": delivery.get("mesh_parts")})
    print(rows[-1], flush=True)
(OUT / "SUMMARY.json").write_text(json.dumps(rows, indent=2))
