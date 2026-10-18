#!/usr/bin/env python3
"""Create task-disjoint full-field FEM data for physics-aware PINO adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "generation")]

from candidate_selection_batch import TAU, make_candidates  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from structural.analysis.fem_solver import voxel_fem_solve  # noqa: E402
from structural.analysis.load_generator import LoadGenerator  # noqa: E402


SRC = ROOT / "results/candidate_selection_400case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"
SIL_DIR = ROOT / "data/silhouettes/bench3"
OUT = ROOT / "data/current_candidate_fullfield_fem_20260711"


def task_key(triple: list[str]) -> str:
    return "|".join(sorted(map(str, triple)))


def split_tasks(cases: list[dict], seed: int) -> dict[str, set[str]]:
    tasks = sorted({task_key(case["triple"]) for case in cases})
    rng = np.random.default_rng(seed)
    rng.shuffle(tasks)
    n_train = int(round(0.60 * len(tasks)))
    n_cal = int(round(0.20 * len(tasks)))
    return {
        "train": set(tasks[:n_train]),
        "val": set(tasks[n_train:n_train + n_cal]),
        "sealed_test": set(tasks[n_train + n_cal:]),
    }


def select_rows(cases: list[dict], allowed: set[str], count: int) -> list[dict]:
    """Balance safe/unsafe samples nearest to the real safety threshold."""
    safe, unsafe = [], []
    for case in cases:
        if task_key(case["triple"]) not in allowed:
            continue
        for row in case["candidates"]:
            sf = float(row.get("sf_fem") or 0.0)
            if not np.isfinite(sf) or sf <= 0:
                continue
            item = {
                "branch": str(case["branch"]),
                "case_id": int(case["case_id"]),
                "triple": list(case["triple"]),
                "candidate": str(row["candidate"]),
                "sf": sf,
                "fem_safe": bool(row.get("fem_safe", sf >= TAU)),
                "distance": abs(math.log10(sf / TAU)),
            }
            (safe if item["fem_safe"] else unsafe).append(item)
    # Keep at most two variants per silhouette triple. Both branches of the
    # same triple must remain a single statistical task.
    def take(pool: list[dict], n: int) -> list[dict]:
        chosen, per_task = [], {}
        for item in sorted(pool, key=lambda x: (x["distance"], x["candidate"])):
            key = task_key(item["triple"])
            if per_task.get(key, 0) >= 2:
                continue
            chosen.append(item)
            per_task[key] = per_task.get(key, 0) + 1
            if len(chosen) >= n:
                return chosen
        return chosen
    n_safe = count // 2
    return take(safe, n_safe) + take(unsafe, count - n_safe)


def load_masks(triple: list[str], resolution: int) -> list[np.ndarray]:
    paths = [SIL_DIR / f"{name}.png" for name in triple]
    tensors, _ = load_data_from_list(
        [str(path) for path in paths], True, (resolution, resolution), torch.device("cpu"), 3, False
    )
    return [np.asarray(mask).squeeze().astype(np.float32) for mask in tensors]


def stable_name(row: dict) -> str:
    raw = f"{row['branch']}_{row['case_id']:03d}_{row['candidate']}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{raw}_{digest}.npz"


def export_row(row: dict, resolution: int, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / stable_name(row)
    if path.exists():
        return {**row, "path": str(path.relative_to(output_dir.parent)), "cached": True}
    masks = load_masks(row["triple"], resolution)
    candidates, _ = make_candidates(masks, resolution, row["branch"] == "fixed")
    if row["candidate"] not in candidates:
        raise KeyError(f"candidate unavailable after reconstruction: {row['candidate']}")
    density = candidates[row["candidate"]].astype(np.float32)
    load = LoadGenerator().standing_load(density)
    fem = voxel_fem_solve(density, load["force"], load["fixed_mask"], verbose=False)
    solid = density > 0.5
    vm_max = float(np.max(fem["von_mises"][solid])) if np.any(solid) else 0.0
    sf = 50e6 / max(vm_max, 1e-12)
    np.savez_compressed(
        path,
        density=density,
        force=load["force"].astype(np.float32),
        fixed_mask=load["fixed_mask"].astype(np.float32),
        displacement=fem["displacement"].astype(np.float32),
        stress=fem["stress"].astype(np.float32),
        von_mises=fem["von_mises"].astype(np.float32),
        sf=np.float32(sf),
    )
    return {
        **row,
        "path": str(path.relative_to(output_dir.parent)),
        "sf_recomputed": float(sf),
        "fem_safe_recomputed": bool(sf >= TAU),
        "fem_total_time_s": float(fem["total_time_s"]),
        "cached": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SRC)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--train-count", type=int, default=160)
    parser.add_argument("--val-count", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers != 1:
        raise ValueError("Use one FEM worker: the sparse solver already uses CPU parallelism.")
    source = json.loads(args.src.read_text())
    cases = source["cases"]
    resolution = int(source["resolution"])
    splits = split_tasks(cases, args.seed)
    chosen = {
        "train": select_rows(cases, splits["train"], args.train_count),
        "val": select_rows(cases, splits["val"], args.val_count),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    manifest = {
        "schema": "shadowforge_current_candidate_fullfield_fem_v1",
        "source": str(args.src), "resolution": resolution, "tau": TAU, "seed": args.seed,
        "split_unit": "ordered silhouette triple; branches are assigned together",
        "task_counts": {name: len(values) for name, values in splits.items()},
        "train": [], "val": [], "sealed_test_tasks": sorted(splits["sealed_test"]),
    }
    for split in ("train", "val"):
        out_dir = args.out / split
        for row in chosen[split]:
            try:
                result = export_row(row, resolution, out_dir)
                manifest[split].append(result)
                print(json.dumps({"split": split, **result}, ensure_ascii=False), flush=True)
            except Exception as exc:
                print(json.dumps({"split": split, "row": row, "error": repr(exc)}), flush=True)
            (args.out / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["elapsed_s"] = time.time() - t0
    manifest["counts"] = {name: len(manifest[name]) for name in ("train", "val")}
    (args.out / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
