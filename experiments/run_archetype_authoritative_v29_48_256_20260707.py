#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = ROOT / "data/silhouettes/anime_archetype_v1"
OUT = ROOT / "results/archetype_authoritative_v29_48core_256delivery_20260707"

EXAMPLES = [
    ("hero_team", ["anime_caped_hero", "anime_spiky_swordsman", "anime_winged_idol"]),
    ("mascot_mecha", ["anime_chibi_mascot", "anime_mecha_robot", "anime_magic_beast"]),
    ("stealth_warriors", ["anime_masked_thief", "anime_ninja_runner", "anime_samurai_mask"]),
    ("magic_stage", ["anime_sailor_warrior", "anime_twin_tail_magical", "anime_umbrella_girl"]),
]


def path_for(name: str) -> str:
    path = SRC / f"{name}.png"
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
    (OUT / "SUMMARY.json").write_text(json.dumps({"schema": "archetype_authoritative_v29_48_256_v1", "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
