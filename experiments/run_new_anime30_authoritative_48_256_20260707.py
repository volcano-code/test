#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = Path("/home/lin/下载/original_anime_silhouettes_30(1) (2)/original_anime_silhouettes_30")
OUT = ROOT / "results/new_anime30_authoritative_48core_256delivery_20260707"

EXAMPLES = [
    ("blade_heroes", "Blade heroes", ["01", "11", "25"]),
    ("arcane_casters", "Arcane casters", ["02", "14", "27"]),
    ("stage_dancers", "Stage dancers", ["08", "16", "22"]),
    ("mechanic_companions", "Mechanic companions", ["12", "13", "30"]),
    ("beast_spirits", "Beast spirits", ["15", "17", "26"]),
]


def path_for(idx: str) -> Path:
    matches = sorted(SRC.glob(f"{idx}_*.png"))
    if len(matches) != 1:
        raise FileNotFoundError((idx, matches))
    return matches[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for slug, title, ids in EXAMPLES:
        case_out = OUT / slug
        cmd = [
            sys.executable,
            str(ROOT / "shadowforge_pipeline/run.py"),
            "--silhouettes",
            *[str(path_for(i)) for i in ids],
            "--output",
            str(case_out),
            "--branches",
            "fixed",
            "--soft",
            "off",
            "--screen",
            "pino_acs_fem",
            "--cert-resolution",
            "48",
            "--delivery-resolution",
            "256",
            "--min-iou",
            "0.95",
            "--min-saving",
            "0.40",
            "--max-saving",
            "0.85",
            "--print-mm",
            "100",
            "--nozzle-mm",
            "0.4",
            "--wall-safety",
            "2.0",
            "--device",
            "cpu",
        ]
        print("RUN", slug, flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (case_out / "RUN_OUTPUT.txt").write_text(proc.stdout, encoding="utf-8")
        manifest_path = case_out / "MANIFEST.json"
        rec = {
            "example": slug,
            "group_name": title,
            "ids": ids,
            "returncode": proc.returncode,
            "manifest": str(manifest_path),
        }
        if manifest_path.exists():
            manifest = json.load(open(manifest_path, encoding="utf-8"))
            rec.update({
                "status": manifest.get("status"),
                "defer_reason": manifest.get("defer_reason"),
                "selected_core": manifest.get("selected_core", {}).get("candidate_id") if manifest.get("selected_core") else None,
                "candidate_counts": manifest.get("candidate_counts"),
                "delivery": manifest.get("delivery"),
                "config": manifest.get("config"),
            })
        else:
            rec["error_tail"] = proc.stdout[-2000:]
        rows.append(rec)
        (OUT / "SUMMARY.partial.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print("DONE", slug, rec.get("status"), rec.get("defer_reason"), flush=True)
    summary = {
        "schema": "new_anime30_authoritative_48core_256delivery_v1",
        "source_dir": str(SRC),
        "output": str(OUT),
        "n_examples": len(rows),
        "rows": rows,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
