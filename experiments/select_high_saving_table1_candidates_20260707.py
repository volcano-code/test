#!/usr/bin/env python3
"""Select replacement examples whose ShadowForge saving is in [40%, 70%]."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/table1_high_saving_replacement_20260707"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "results/print_10cm_256/MANIFEST.json"
    data = json.loads(manifest.read_text())
    rows = []
    for item in data["examples"]:
        saving = float(item["saving"])
        if not (0.4 <= saving <= 0.7):
            continue
        example = item["example"]
        case_dir = ROOT / "results/print_10cm_256" / example
        stl = next(case_dir.glob("*.stl"), None)
        vox = next(case_dir.glob("*.npy"), None)
        rows.append({
            "example": example,
            "silhouettes": ",".join(item["silhouettes"]),
            "iou": item["iou"],
            "saving": saving,
            "saving_pct": f"{100*saving:.1f}%",
            "single": item["single"],
            "watertight": item["watertight"],
            "candidate": item["candidate"],
            "resolution": item["resolution"],
            "print_mm": item["print_mm"],
            "stl": str(stl) if stl else "",
            "voxels": str(vox) if vox else "",
            "source_manifest": str(manifest),
        })
    rows.sort(key=lambda r: (-r["saving"], r["example"]))

    with (OUT / "selected_print_10cm_256.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    anime = json.loads((ROOT / "results/anime30_adjusted_full_screen_20260704/SUMMARY.json").read_text())
    anime_rows = []
    for item in anime["top"]:
        saving = float(item["saving"])
        if 0.4 <= saving <= 0.7 and item.get("single") and float(item["mean_iou"]) >= 0.97:
            anime_rows.append({
                "mode": item["mode"],
                "silhouettes": ",".join(item["silhouettes"]),
                "mean_iou": item["mean_iou"],
                "min_iou": item["min_iou"],
                "saving": saving,
                "saving_pct": f"{100*saving:.1f}%",
                "single": item["single"],
                "res": item["res"],
                "candidate": item["candidate"],
                "paths": "|".join(item["paths"]),
                "status": "screened_res48_not_10cm_256_export",
            })
    anime_rows.sort(key=lambda r: (-float(r["mean_iou"]), -r["saving"]))
    with (OUT / "anime30_screened_candidates.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(anime_rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(anime_rows[:30])

    summary = {
        "recommended_replacement_set": "print_10cm_256",
        "reason": "All four examples already have 10cm/256 exports, IoU, saving, single-body, and watertight evidence.",
        "target_saving_range": [0.4, 0.7],
        "selected_count": len(rows),
        "selected_tsv": str(OUT / "selected_print_10cm_256.tsv"),
        "anime_screened_count": len(anime_rows),
        "anime_tsv": str(OUT / "anime30_screened_candidates.tsv"),
        "anime_status": "usable as future replacement only after formal 10cm/256 rebuild/export",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
