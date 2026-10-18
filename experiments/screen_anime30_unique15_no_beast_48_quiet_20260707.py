#!/usr/bin/env python3
"""Quiet 48^3 screen for five disjoint no-beast anime30 triples."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from candidate_selection_batch import make_candidates, n6
from voxel_ops import evaluate_tfs_volume

R = 48
OUT = ROOT / "results/anime30_unique15_no_beast_screen48_quiet_20260707"
MASK_ROOT = ROOT / "results/anime30_adjusted_full_screen_20260704/adjusted_masks/thick1"
EXCLUDE = {"anime30_05", "anime30_08", "anime30_18", "anime30_29"}
NAMES = [f"anime30_{i:02d}" for i in range(1, 31) if f"anime30_{i:02d}" not in EXCLUDE]


def load_mask(name: str) -> np.ndarray:
    path = MASK_ROOT / f"{name}_thick1.png"
    img = cv2.imread(str(path), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.resize(img, (R, R), interpolation=cv2.INTER_NEAREST)
    return img.astype(np.float32)


def run_one(names: tuple[str, str, str], mask_cache: dict[str, np.ndarray]) -> dict | None:
    masks = [mask_cache[n] for n in names]
    try:
        cands, aux = make_candidates(masks, R, True, print_mm=100.0, nozzle_mm=0.4, wall_safety=2.0)
    except Exception:
        return None
    gt6 = masks + [np.fliplr(m) for m in masks]
    hull_vox = max(int(aux["hull"].sum()), 1)
    best_gate = None
    best_any = None
    for cname, vol in cands.items():
        vol = vol.astype(bool)
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=256)
        row = {
            "names": list(names),
            "candidate": cname,
            "mean_iou": float(ev["mean_iou"]),
            "min_iou": float(min(ev["view_ious"][:3])),
            "saving": float(1.0 - int(vol.sum()) / hull_vox),
            "single": bool(n6(vol) == 1),
        }
        key = (row["mean_iou"], row["min_iou"], row["single"], -abs(row["saving"] - 0.55))
        if best_any is None or key > (best_any["mean_iou"], best_any["min_iou"], best_any["single"], -abs(best_any["saving"] - 0.55)):
            best_any = row
        if row["mean_iou"] >= 0.955 and row["min_iou"] >= 0.94 and row["single"]:
            if best_gate is None or key > (best_gate["mean_iou"], best_gate["min_iou"], best_gate["single"], -abs(best_gate["saving"] - 0.55)):
                best_gate = row
    return best_gate or best_any


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mask_cache = {n: load_mask(n) for n in NAMES}
    rows = []
    combos = list(itertools.combinations(NAMES, 3))
    for idx, names in enumerate(combos, 1):
        row = run_one(names, mask_cache)
        if row and row["mean_iou"] >= 0.95 and row["min_iou"] >= 0.93 and row["single"]:
            rows.append(row)
        if idx % 250 == 0:
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
    out = {"schema": "unique15_no_beast_screen48_quiet_v1", "exclude": sorted(EXCLUDE), "rows": rows, "selected": selected}
    (OUT / "SCREEN.json").write_text(json.dumps(out, indent=2))
    print("selected", len(selected), sorted(used), flush=True)
    for r in selected:
        print(r, flush=True)


if __name__ == "__main__":
    main()
