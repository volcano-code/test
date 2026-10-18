#!/usr/bin/env python3
"""Build a small task-disjoint index for official-operator integration pilots."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def select_tasks(
    rows: list[dict],
    split: str,
    task_limit: int,
    row_limit: int,
    rng: random.Random,
) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            by_task[row["task"]].append(row)

    tasks = list(by_task)
    rng.shuffle(tasks)
    chosen = tasks[:task_limit]
    selected = [row for task in chosen for row in by_task[task]]
    rng.shuffle(selected)
    return selected[:row_limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-tasks", type=int, default=8)
    parser.add_argument("--calibration-tasks", type=int, default=4)
    parser.add_argument("--train-rows", type=int, default=96)
    parser.add_argument("--calibration-rows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    rng = random.Random(args.seed)
    train = select_tasks(
        payload["rows"], "train", args.train_tasks, args.train_rows, rng
    )
    calibration = select_tasks(
        payload["rows"],
        "calibration",
        args.calibration_tasks,
        args.calibration_rows,
        rng,
    )
    selected = train + calibration
    train_tasks = {row["task"] for row in train}
    calibration_tasks = {row["task"] for row in calibration}
    if train_tasks & calibration_tasks:
        raise RuntimeError("Pilot index is not task-disjoint.")

    output = {
        "schema": payload.get("schema"),
        "split_unit": "task",
        "source_index": str(args.input.resolve()),
        "purpose": "official GINO/Transolver integration pilot only",
        "seed": args.seed,
        "counts": {
            "train": len(train),
            "calibration": len(calibration),
            "train_tasks": len(train_tasks),
            "calibration_tasks": len(calibration_tasks),
        },
        "rows": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(output["counts"], indent=2))


if __name__ == "__main__":
    main()
