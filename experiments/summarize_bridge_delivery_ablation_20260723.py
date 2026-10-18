#!/usr/bin/env python3
"""Summarize frozen-proposal delivery ablations without mixing soft-stage effects."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=["full", "w_o_ray_restoration", "w_o_bridge_repair", "w_o_delivery_shell"])
    args = parser.parse_args()

    rows = []
    for variant in args.variants:
        manifest = args.root / variant / "MANIFEST.json"
        data = json.loads(manifest.read_text()) if manifest.is_file() else {}
        chosen = data.get("selected") or {}
        rows.append({
            "variant": variant,
            "status": data.get("status", "MISSING"),
            "mean_iou": chosen.get("mean_iou"),
            "worst_view_iou": chosen.get("min_iou"),
            "mean_dice": chosen.get("mean_dice"),
            "saving": chosen.get("saving"),
            "components": chosen.get("components"),
            "single_body": chosen.get("components") == 1 if chosen else None,
            "watertight": chosen.get("watertight"),
            "mesh_outer_parts": chosen.get("mesh_outer_parts"),
            "eligible": chosen.get("eligible"),
            "candidate": chosen.get("candidate"),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    with args.output.with_suffix(".tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
