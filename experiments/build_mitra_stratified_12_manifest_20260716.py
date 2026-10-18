#!/usr/bin/env python3
"""Freeze a 6-low / 6-high Mitra-stratified diagnostic suite.

This is a deliberately stratified diagnostic suite, not a random benchmark.
It preserves the exact target paths and declares the physical x/y/z axes so
every downstream method can be evaluated under one frozen protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument(
        "--high",
        type=Path,
        default=ROOT / "results/mitra_pauly2009_fixed3view_20260716/mitra_difficulty_strata.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    low = json.loads((args.low / "MANIFEST.json").read_text())
    strata = json.loads(args.high.read_text())
    low_rows = low["rows"]
    high_rows = strata["high_mitra_cases"]
    if len(low_rows) != 6 or len(high_rows) != 6:
        raise RuntimeError(f"Expected 6+6 tasks, got {len(low_rows)}+{len(high_rows)}")

    rows = []
    for row in low_rows:
        by_axis = dict(zip(row["axes"], row["source_paths"]))
        rows.append({
            "case_id": f"low__{row['case_id']}",
            "stratum": "Mitra-low",
            "target_paths_xyz": [by_axis["x"], by_axis["y"], by_axis["z"]],
            "sadekar_tfs_targets": [by_axis["y"], by_axis["x"], by_axis["z"]],
            "mitra_mean_iou": row["mean_iou"],
            "mitra_worst_iou": row["worst_iou"],
        })
    for row in high_rows:
        case_root = ROOT / "results/nsa_style_fixed_required_10cm_256"
        inputs = next(case_root.glob(f"**/{row['case_id']}/inputs"))
        x = next(inputs.glob("input1_*.png"))
        y = next(inputs.glob("input2_*.png"))
        z = next(inputs.glob("input0_*.png"))
        rows.append({
            "case_id": f"high__{row['case_id']}",
            "stratum": "Mitra-high",
            "target_paths_xyz": [str(x), str(y), str(z)],
            "sadekar_tfs_targets": [str(y), str(x), str(z)],
            "mitra_mean_iou": row["mean_iou"],
            "mitra_worst_iou": row["worst_iou"],
        })

    payload = {
        "schema": "mitra_stratified_fixed3_v1",
        "purpose": "diagnostic 6-low/6-high Mitra-stratified comparison; report strata and all-case aggregate separately",
        "warning": "Task selection is conditioned on Mitra performance and must not be presented as a randomly sampled primary benchmark.",
        "protocol": {
            "views": 3,
            "axes": ["x", "y", "z"],
            "camera": "fixed orthographic TFS, mirror_mode=false, zdist=1.7",
            "resolution": 256,
            "evaluation": "one common fixed-origin renderer; no independent reorientation or rescaling",
        },
        "cases": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "benchmark_manifest.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out": str(out), "cases": len(rows), "low": len(low_rows), "high": len(high_rows)}, indent=2))


if __name__ == "__main__":
    main()
