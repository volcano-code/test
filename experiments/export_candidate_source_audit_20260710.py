#!/usr/bin/env python3
"""Export an explicit source/provenance row for every selector-audit candidate."""

import csv
import json
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = ROOT / "results/candidate_selection_100case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"
OUT = ROOT / "results/controlled_benchmark_geometry_only_20260710/candidate_source_manifest.csv"

payload = json.loads(SRC.read_text())
fields = [
    "case_id", "candidate_id", "candidate_source", "branch", "mean_iou", "saving",
    "single_body", "watertight", "fem_label",
]
with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for case in payload["cases"]:
        for candidate in case["candidates"]:
            writer.writerow({
                "case_id": case["case_id"], "candidate_id": candidate["candidate"],
                "candidate_source": "geometry", "branch": case["branch"],
                "mean_iou": candidate["mean_iou"], "saving": candidate["material_saved_vs_hull"],
                "single_body": candidate["component_count_6"] == 1,
                "watertight": candidate["watertight"], "fem_label": candidate["fem_safe"],
            })
print(OUT)
