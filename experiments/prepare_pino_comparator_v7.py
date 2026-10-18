#!/usr/bin/env python3
"""Prepare an immutable task-disjoint benchmark manifest for PINO baselines.

This does not train models.  It converts the delivered v7 index into the
single protocol shared by Geometry-MLP, 3D-CNN, FNO, PINO without physics, and
full PINO.  Training launchers can consume the emitted manifest after the v7
FEM data builder finishes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-res", type=int, default=48, choices=[48])
    args = parser.parse_args()

    source = json.loads(args.index.read_text())
    rows = source.get("rows", [])
    if not rows:
        raise RuntimeError("Expected a row-based task-grouped source index")
    counts = {name: 0 for name in ("train", "calibration", "sealed_test")}
    tasks = {name: set() for name in counts}
    missing = []
    for row in rows:
        split = row.get("split")
        if split not in counts:
            raise RuntimeError(f"Unexpected split: {split}")
        path = args.index.parent / row["path"]
        if not path.is_file():
            missing.append(str(path))
        counts[split] += 1
        tasks[split].add(row["task"])
    if missing:
        raise RuntimeError(f"Source index has {len(missing)} missing files")
    if any(not tasks[name] for name in tasks):
        raise RuntimeError(f"Incomplete task splits: {counts}")

    task_owner = {}
    for split, split_tasks in tasks.items():
        for task in split_tasks:
            if task in task_owner:
                raise RuntimeError(f"Task leakage: {task} appears in {task_owner[task]} and {split}")
            task_owner[task] = split
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "rows": rows,
        "protocol": {
            "source_index": str(args.index.resolve()),
            "split_unit": "complete silhouette-triple task",
            "target_resolution": args.target_res,
            "input": "delivered structural proxy C48 with occupancy, loads, SDF, and fixed-support mask",
            "labels": "FEM displacement, stress, von-Mises stress, peak stress, and FEM safety factor",
            "models": ["geometry_mlp", "cnn3d", "ufno", "ufno_dual_no_physics", "ufno_dual_pino"],
            "calibration": "frozen calibration-only ACS; sealed test remains untouched until final evaluation",
        },
        "counts": {split: {"rows": counts[split], "tasks": len(tasks[split])} for split in counts},
        "task_digest": hashlib.sha256("\n".join(sorted(task_owner)).encode()).hexdigest(),
    }
    args.output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"out": str(args.output), "counts": manifest["counts"],
                      "task_digest": manifest["task_digest"]}, indent=2))


if __name__ == "__main__":
    main()
