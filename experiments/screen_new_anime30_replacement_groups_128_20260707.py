#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, generate_binary_structure

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
for p in [ROOT, ROOT / "generation", ROOT / "experiments"]:
    sys.path.insert(0, str(p))

from candidate_selection_batch import make_candidates, n6
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume

SRC = Path("/home/lin/下载/original_anime_silhouettes_30(1) (2)/original_anime_silhouettes_30")
OUT = ROOT / "results/new_anime30_replacement_groups_128_20260707"
R = 128
PRINT_MM = 100.0
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0
S6 = generate_binary_structure(3, 1)

GROUPS = [
    ("stealth_rangers", "Stealth rangers", ["04", "09", "19"]),
    ("future_commanders", "Future commanders", ["06", "18", "28"]),
    ("mechanic_companions", "Mechanic companions", ["12", "13", "30"]),
    ("wandering_mystics", "Wandering mystics", ["05", "07", "29"]),
    ("urban_catwalk", "Urban catwalk", ["03", "21", "23"]),
    ("ninja_samurai_archer", "Ninja samurai archer", ["04", "09", "29"]),
    ("commander_warriors", "Commander warriors", ["06", "18", "21"]),
]


def file_for(idx: str) -> Path:
    matches = sorted(SRC.glob(f"{idx}_*.png"))
    if len(matches) != 1:
        raise FileNotFoundError((idx, matches))
    return matches[0]


def erosion_survival(volume: np.ndarray) -> float:
    vol = np.asarray(volume).astype(bool)
    return float(binary_erosion(vol, S6).sum() / max(int(vol.sum()), 1))


def eval_group(slug: str, title: str, ids: list[str]) -> dict:
    paths = [str(file_for(x)) for x in ids]
    sils, _ = load_data_from_list(paths, True, (R, R), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils[:3]]
    cands, aux = make_candidates(masks, R, True, print_mm=PRINT_MM, nozzle_mm=NOZZLE_MM, wall_safety=WALL_SAFETY)
    gt6 = masks + [np.fliplr(m) for m in masks]
    hull_vox = max(int(aux["hull"].sum()), 1)
    rows = []
    for cname, vol in cands.items():
        vol = vol.astype(bool)
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=384)
        rows.append({
            "candidate": cname,
            "mean_iou": float(ev["mean_iou"]),
            "min_iou": float(min(ev["view_ious"][:3])),
            "view_ious6": [float(x) for x in ev["view_ious"][:6]],
            "saving": float(1.0 - int(vol.sum()) / hull_vox),
            "single": bool(n6(vol) == 1),
            "erosion_survival": erosion_survival(vol),
        })
    gate = [
        r for r in rows
        if r["mean_iou"] >= 0.970
        and r["min_iou"] >= 0.950
        and 0.40 <= r["saving"] <= 0.60
        and r["single"]
        and r["erosion_survival"] >= 0.15
    ]
    chosen = max(gate, key=lambda r: (-abs(r["saving"] - 0.50), r["mean_iou"])) if gate else max(rows, key=lambda r: (r["mean_iou"], r["single"], r["saving"]))
    return {"slug": slug, "title": title, "ids": ids, "paths": paths, "accepted128": bool(gate), "selected128": chosen, "all_candidates128": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = [eval_group(*g) for g in GROUPS]
    (OUT / "SCREEN128.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))
    for r in records:
        s = r["selected128"]
        print(r["slug"], r["title"], r["ids"], "accepted128=", r["accepted128"], s["candidate"], round(s["mean_iou"], 4), round(s["min_iou"], 4), round(s["saving"], 4), s["single"])


if __name__ == "__main__":
    main()
