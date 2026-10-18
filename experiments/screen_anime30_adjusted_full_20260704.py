#!/usr/bin/env python3
"""Screen adjusted anime30 silhouettes for 10cm physical-print candidates.

This is a search-only script. It does not edit source silhouettes. It creates
closed/thickened copies, evaluates all C(30,3) triplets at low resolution, and
promotes the best candidates for later 256^3 construction.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, generate_binary_structure

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from candidate_selection_batch import make_candidates, n6  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from voxel_ops import evaluate_tfs_volume  # noqa: E402

SRC = ROOT / "data/silhouettes/original_anime_30_20260704"
OUT = ROOT / "results/anime30_adjusted_full_screen_20260704"
S2 = generate_binary_structure(2, 1)


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 127


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(path)


def adjusted(mask: np.ndarray, mode: str) -> np.ndarray:
    m = mask.astype(bool)
    if mode == "orig":
        return m
    if mode == "close1":
        return binary_fill_holes(binary_closing(m, S2, iterations=1))
    if mode == "close2":
        return binary_fill_holes(binary_closing(m, S2, iterations=2))
    if mode == "thick1":
        return binary_fill_holes(binary_dilation(binary_closing(m, S2, iterations=1), S2, iterations=1))
    raise ValueError(mode)


def build_adjusted_library(modes: list[str]) -> dict[str, list[dict]]:
    source = sorted(SRC.glob("*.png"))
    if len(source) != 30:
        raise RuntimeError(f"expected 30 masks in {SRC}, found {len(source)}")
    lib: dict[str, list[dict]] = {}
    for mode in modes:
        d = OUT / "adjusted_masks" / mode
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for p in source:
            sid = p.stem
            m = adjusted(load_mask(p), mode)
            dst = d / f"{sid}_{mode}.png"
            save_mask(m, dst)
            rows.append({"id": sid, "mode": mode, "path": str(dst), "fg": float(m.mean())})
        lib[mode] = rows
    return lib


def score_triplet(paths: list[str], res: int) -> dict | None:
    sils, _ = load_data_from_list(paths, True, (res, res), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils]
    candidates, aux = make_candidates(masks, res, True, print_mm=100.0, nozzle_mm=0.4, wall_safety=2.0)
    gt6 = masks + [np.fliplr(m) for m in masks]
    hull = aux["hull"].astype(bool)
    best = None
    for cname, vol0 in candidates.items():
        vol = vol0.astype(bool)
        if n6(vol) != 1:
            continue
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=256)
        view_ious = [float(x) for x in ev["view_ious"][:3]]
        saving = float(1.0 - int(vol.sum()) / max(int(hull.sum()), 1))
        rec = {
            "candidate": cname,
            "mean_iou": float(np.mean(view_ious)),
            "min_iou": float(min(view_ious)),
            "view_ious": view_ious,
            "saving": saving,
            "single": True,
            "quick_pass": bool(np.mean(view_ious) >= 0.975 and min(view_ious) >= 0.95 and 0.40 <= saving <= 0.85),
        }
        key = (rec["quick_pass"], rec["mean_iou"], rec["min_iou"], 0.40 <= saving <= 0.85, saving)
        old = None if best is None else (best["quick_pass"], best["mean_iou"], best["min_iou"], 0.40 <= best["saving"] <= 0.85, best["saving"])
        if old is None or key > old:
            best = rec
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=48)
    parser.add_argument("--modes", default="close1,close2,thick1")
    parser.add_argument("--topk", type=int, default=120)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    lib = build_adjusted_library(modes)
    all_rows = []
    for mode in modes:
        rows = lib[mode]
        for ids in itertools.combinations(range(len(rows)), 3):
            paths = [rows[i]["path"] for i in ids]
            names = [rows[i]["id"] for i in ids]
            try:
                best = score_triplet(paths, args.res)
                if best:
                    all_rows.append({"mode": mode, "silhouettes": names, "paths": paths, "res": args.res, **best})
            except Exception as exc:
                all_rows.append({"mode": mode, "silhouettes": names, "paths": paths, "res": args.res, "error": str(exc), "quick_pass": False})
        all_rows.sort(key=lambda r: (r.get("quick_pass", False), r.get("mean_iou", -1), r.get("min_iou", -1), r.get("saving", -1)), reverse=True)
        (OUT / "screen_partial.json").write_text(json.dumps({"rows": all_rows[: args.topk]}, indent=2, ensure_ascii=False), encoding="utf-8")

    all_rows.sort(key=lambda r: (r.get("quick_pass", False), r.get("mean_iou", -1), r.get("min_iou", -1), r.get("saving", -1)), reverse=True)
    summary = {
        "schema": "anime30_adjusted_full_screen_v1",
        "source": str(SRC),
        "res": args.res,
        "modes": modes,
        "triplets_per_mode": 4060,
        "total_records": len(all_rows),
        "quick_pass_count": sum(1 for r in all_rows if r.get("quick_pass")),
        "top": all_rows[: args.topk],
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT / "top.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["mode", "silhouettes", "candidate", "mean_iou", "min_iou", "saving", "quick_pass", "res"])
        wr.writeheader()
        for r in all_rows[: args.topk]:
            wr.writerow({
                "mode": r.get("mode"),
                "silhouettes": " ".join(r.get("silhouettes", [])),
                "candidate": r.get("candidate", ""),
                "mean_iou": r.get("mean_iou", ""),
                "min_iou": r.get("min_iou", ""),
                "saving": r.get("saving", ""),
                "quick_pass": r.get("quick_pass", False),
                "res": r.get("res"),
            })
    print(json.dumps({k: summary[k] for k in ("quick_pass_count", "total_records", "modes")}, indent=2))


if __name__ == "__main__":
    main()
