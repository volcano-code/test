#!/usr/bin/env python3
from __future__ import annotations

import itertools
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

R = 128
PRINT_MM = 100.0
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0
MASK_ROOT = ROOT / "results/anime30_adjusted_full_screen_20260704/adjusted_masks/thick1"
OUT = ROOT / "results/anime30_unique15_replacement_screen128_20260707"
S6 = generate_binary_structure(3, 1)

USED_ACCEPTED = {
    "anime30_08", "anime30_15", "anime30_28",
    "anime30_13", "anime30_14", "anime30_27",
    "anime30_05", "anime30_07", "anime30_20",
    "anime30_02", "anime30_16", "anime30_30",
}
POOL = [f"anime30_{i:02d}" for i in range(1, 31) if f"anime30_{i:02d}" not in USED_ACCEPTED]


def path_for(name: str) -> str:
    return str(MASK_ROOT / f"{name}_thick1.png")


def erosion_survival(volume: np.ndarray) -> float:
    vol = np.asarray(volume).astype(bool)
    return float(binary_erosion(vol, S6).sum() / max(int(vol.sum()), 1))


def eval_triplet(names: tuple[str, str, str]) -> dict:
    sils, _ = load_data_from_list([path_for(n) for n in names], True, (R, R), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils[:3]]
    cands, aux = make_candidates(masks, R, True, print_mm=PRINT_MM, nozzle_mm=NOZZLE_MM, wall_safety=WALL_SAFETY)
    gt6 = masks + [np.fliplr(m) for m in masks]
    hull_vox = max(int(aux["hull"].sum()), 1)
    best = None
    rows = []
    for cname, vol in cands.items():
        vol = vol.astype(bool)
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=384)
        row = {
            "candidate": cname,
            "mean_iou": float(ev["mean_iou"]),
            "min_iou": float(min(ev["view_ious"][:3])),
            "saving": float(1.0 - int(vol.sum()) / hull_vox),
            "single": bool(n6(vol) == 1),
            "erosion_survival": erosion_survival(vol),
        }
        rows.append(row)
        key = (row["mean_iou"] >= 0.965, row["saving"] >= 0.25, row["single"], row["mean_iou"], row["saving"])
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    return {"silhouettes": list(names), "selected": best[1], "all_candidates": rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, names in enumerate(itertools.combinations(POOL, 3), 1):
        rec = eval_triplet(names)
        records.append(rec)
        if idx % 25 == 0:
            (OUT / "SCREEN.partial.json").write_text(json.dumps(records, indent=2))
            print("done", idx, flush=True)
    records.sort(key=lambda r: (
        r["selected"]["mean_iou"] >= 0.965,
        r["selected"]["saving"] >= 0.25,
        r["selected"]["single"],
        r["selected"]["mean_iou"],
        r["selected"]["saving"],
    ), reverse=True)
    (OUT / "SCREEN.json").write_text(json.dumps({"pool": POOL, "records": records}, indent=2))
    for r in records[:20]:
        s = r["selected"]
        print("+".join(r["silhouettes"]), s["candidate"], round(s["mean_iou"], 4), round(s["min_iou"], 4), round(s["saving"], 4), s["single"], flush=True)


if __name__ == "__main__":
    main()
