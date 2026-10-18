#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
MASK_ROOT = ROOT / "results/anime30_adjusted_full_screen_20260704/adjusted_masks/thick1"
OUT = ROOT / "results/old_anime30_authoritative_v29_48core_256delivery_20260707"

EXAMPLES = [
    ("unique1_08_15_28", ["anime30_08", "anime30_15", "anime30_28"]),
    ("unique2_13_14_27", ["anime30_13", "anime30_14", "anime30_27"]),
    ("unique3_05_07_20", ["anime30_05", "anime30_07", "anime30_20"]),
    ("unique4_01_04_06", ["anime30_01", "anime30_04", "anime30_06"]),
    ("unique5_02_16_30", ["anime30_02", "anime30_16", "anime30_30"]),
]


def path_for(name: str) -> str:
    path = MASK_ROOT / f"{name}_thick1.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for slug, ids in EXAMPLES:
        case = OUT / slug
        cmd = [
            sys.executable,
            str(ROOT / "shadowforge_pipeline/run.py"),
            "--silhouettes",
            *[path_for(i) for i in ids],
            "--output",
            str(case),
            "--branches",
            "fixed",
            "--soft",
            "off",
            "--screen",
            "v29_ensemble_acs_fem",
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
            "--device",
            "cpu",
        ]
        print("RUN", slug, flush=True)
        proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        case.mkdir(parents=True, exist_ok=True)
        (case / "RUN_OUTPUT.txt").write_text(proc.stdout, encoding="utf-8")
        mp = case / "MANIFEST.json"
        rec = {"example": slug, "ids": ids, "returncode": proc.returncode, "manifest": str(mp)}
        if mp.exists():
            manifest = json.load(open(mp, encoding="utf-8"))
            delivery = manifest.get("delivery") or {}
            screen = delivery.get("structural_screen") or {}
            rec.update({
                "status": manifest.get("status"),
                "defer_reason": manifest.get("defer_reason"),
                "selected_core": manifest.get("selected_core", {}).get("candidate_id") if manifest.get("selected_core") else None,
                "candidate_counts": manifest.get("candidate_counts"),
                "mean_iou": delivery.get("mean_iou"),
                "min_iou": delivery.get("min_iou"),
                "saving": delivery.get("material_saved_vs_hull"),
                "mesh_parts": delivery.get("mesh_parts"),
                "watertight": delivery.get("watertight"),
                "manufacturing_pass": delivery.get("manufacturing_pass"),
                "struct_decided_by": screen.get("decided_by"),
                "struct_safe": screen.get("safe"),
                "needs_fem": screen.get("needs_fem"),
                "sf_pino": screen.get("sf_pino"),
                "acs_lower_sf": screen.get("acs_lower_sf"),
                "sf_fem": screen.get("sf_fem"),
                "mesh": (delivery.get("mesh") or {}).get("artifact"),
                "delivery_artifact": delivery.get("artifact"),
            })
        rows.append(rec)
        (OUT / "SUMMARY.partial.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print("DONE", slug, rec.get("status"), rec.get("mean_iou"), rec.get("saving"), rec.get("mesh_parts"), rec.get("struct_decided_by"), flush=True)
    (OUT / "SUMMARY.json").write_text(json.dumps({"schema": "old_anime30_authoritative_v29_48_256_v1", "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
