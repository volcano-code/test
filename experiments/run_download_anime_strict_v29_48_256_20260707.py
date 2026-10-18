#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = Path("/home/lin/xvxinyao/download_anime_character_silhouettes_white_on_black_20260707/images")
OUT = ROOT / "results/download_anime_strict_v29_48core_256delivery_20260707"

EXAMPLES = [
    ("hero_magic_android", [
        "0011_0011_original_anime_silhouettes_100_2_001_hot_blooded_hero.png",
        "0012_0012_original_anime_silhouettes_100_2_002_magic_girl.png",
        "0020_0020_original_anime_silhouettes_100_2_010_space_android.png",
    ]),
    ("ninja_archer_mascot", [
        "0014_0014_original_anime_silhouettes_100_2_004_stealth_ninja.png",
        "0019_0019_original_anime_silhouettes_100_2_009_ranger_archer.png",
        "0018_0018_original_anime_silhouettes_100_2_008_mascot_creature.png",
    ]),
    ("swordsman_cat_mecha", [
        "0013_0013_original_anime_silhouettes_100_2_003_cool_swordsman.png",
        "0016_0016_original_anime_silhouettes_100_2_006_cat_ear_girl.png",
        "0015_0015_original_anime_silhouettes_100_2_005_mecha_pilot.png",
    ]),
    ("school_mage_traveler", [
        "0007_0007_anime_silhouettes_connected_batch_2.png",
        "0004_0004_anime_silhouettes_connected_batch_2.png",
        "0005_0005_anime_silhouettes_connected_batch_2.png",
    ]),
]


def path_for(name: str) -> str:
    path = SRC / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for slug, names in EXAMPLES:
        case = OUT / slug
        cmd = [
            sys.executable,
            str(ROOT / "shadowforge_pipeline/run.py"),
            "--silhouettes",
            *[path_for(name) for name in names],
            "--output",
            str(case),
            "--branches",
            "free",
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
        rec = {"example": slug, "silhouettes": names, "returncode": proc.returncode, "manifest": str(mp)}
        if mp.exists():
            manifest = json.load(open(mp, encoding="utf-8"))
            delivery = manifest.get("delivery") or {}
            screen = delivery.get("structural_screen") or {}
            mesh = delivery.get("mesh") or {}
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
                "mesh": mesh.get("artifact"),
                "delivery_artifact": delivery.get("artifact"),
            })
        rows.append(rec)
        (OUT / "SUMMARY.partial.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print("DONE", slug, rec.get("status"), rec.get("mean_iou"), rec.get("saving"), rec.get("mesh_parts"), rec.get("struct_decided_by"), flush=True)
    (OUT / "SUMMARY.json").write_text(json.dumps({"schema": "download_anime_strict_v29_48_256_v1", "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
