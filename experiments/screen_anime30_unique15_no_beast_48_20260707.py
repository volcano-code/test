#!/usr/bin/env python3
"""Screen no-beast anime30 triples for five disjoint fixed examples."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from candidate_selection_batch import make_candidates, n6
from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume

R = 48
OUT = ROOT / "results/anime30_unique15_no_beast_screen48_20260707"
MASK_ROOT = ROOT / "results/anime30_adjusted_full_screen_20260704/adjusted_masks/thick1"
EXCLUDE = {"anime30_05", "anime30_08", "anime30_18", "anime30_29"}
NAMES = [f"anime30_{i:02d}" for i in range(1, 31) if f"anime30_{i:02d}" not in EXCLUDE]


def paths_for(names: tuple[str, str, str]) -> list[str]:
    return [str(MASK_ROOT / f"{n}_thick1.png") for n in names]


def run_one(names: tuple[str, str, str]) -> dict | None:
    sils, _ = load_data_from_list(paths_for(names), True, (R, R), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils[:3]]
    cands, aux = make_candidates(masks, R, True, print_mm=100.0, nozzle_mm=0.4, wall_safety=2.0)
    gt6 = masks + [np.fliplr(m) for m in masks]
    hull_vox = max(int(aux["hull"].sum()), 1)
    best = None
    for cname, vol in cands.items():
        vol = vol.astype(bool)
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=256)
        saving = 1.0 - int(vol.sum()) / hull_vox
        row = {
            "names": list(names),
            "candidate": cname,
            "mean_iou": float(ev["mean_iou"]),
            "min_iou": float(min(ev["view_ious"][:3])),
            "saving": float(saving),
            "single": bool(n6(vol) == 1),
        }
        if best is None or (row["mean_iou"], row["single"], -abs(row["saving"] - 0.55)) > (
            best["mean_iou"], best["single"], -abs(best["saving"] - 0.55)
        ):
            best = row
    return best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    combos = list(itertools.combinations(NAMES, 3))
    for idx, names in enumerate(combos, 1):
        row = run_one(names)
        if row and row["mean_iou"] >= 0.955 and row["min_iou"] >= 0.94 and row["single"]:
            rows.append(row)
        if idx % 100 == 0:
            (OUT / "partial.json").write_text(json.dumps(rows, indent=2))
            print("screened", idx, "kept", len(rows), flush=True)
    rows = sorted(rows, key=lambda r: (-r["mean_iou"], abs(r["saving"] - 0.55)))
    selected = []
    used = set()
    for r in rows:
        names = set(r["names"])
        if names & used:
            continue
        selected.append(r)
        used |= names
        if len(selected) == 5:
            break
    out = {"schema": "unique15_no_beast_screen48_v1", "exclude": sorted(EXCLUDE), "rows": rows, "selected": selected}
    (OUT / "SCREEN.json").write_text(json.dumps(out, indent=2))
    print("selected", len(selected), sorted(used), flush=True)
    for r in selected:
        print(r, flush=True)


if __name__ == "__main__":
    main()
