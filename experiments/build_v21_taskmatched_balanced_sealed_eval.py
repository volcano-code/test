#!/usr/bin/env python3
"""Freeze a task-matched balanced sealed diagnostic from existing FEM labels."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--danger-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    base = json.loads(args.base_index.read_text())
    danger = json.loads(args.danger_index.read_text())
    safe_by_task: dict[str, list[dict]] = defaultdict(list)
    danger_by_task: dict[str, list[dict]] = defaultdict(list)
    for row in base["rows"]:
        if row["split"] == "sealed_test" and bool(row["fem_safe"]):
            safe_by_task[row["task"]].append(dict(row))
    for row in danger["rows"]:
        if row["split"] == "sealed_test" and not bool(row["fem_safe"]):
            danger_by_task[row["task"]].append(dict(row))

    rng = random.Random(args.seed)
    selected_safe: list[dict] = []
    selected_danger: list[dict] = []
    per_task = []
    for task in sorted(danger_by_task):
        safe = safe_by_task[task]
        unsafe = danger_by_task[task]
        n = min(len(safe), len(unsafe))
        if not n:
            continue
        rng.shuffle(safe)
        rng.shuffle(unsafe)
        selected_safe.extend(safe[:n])
        selected_danger.extend(unsafe[:n])
        per_task.append({"task": task, "safe": n, "unsafe": n})

    def absolute(rows: list[dict], root: Path) -> list[dict]:
        out = []
        for row in rows:
            record = dict(row)
            record["path"] = str((root / record["path"]).resolve())
            out.append(record)
        return out

    rows = absolute(selected_safe, args.base_index.parent) + absolute(selected_danger, args.danger_index.parent)
    rng.shuffle(rows)
    payload = {
        "schema": "shadowforge_taskmatched_balanced_sealed_diagnostic_v21",
        "purpose": "frozen evaluation only; no row may be used for training or calibration",
        "base_index": str(args.base_index.resolve()),
        "danger_index": str(args.danger_index.resolve()),
        "split": "sealed_test",
        "threshold_sf": 1.3,
        "seed": args.seed,
        "safe": len(selected_safe),
        "unsafe": len(selected_danger),
        "per_task": per_task,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "safe": len(selected_safe), "unsafe": len(selected_danger), "total": len(rows), "tasks": len(per_task)}, indent=2))


if __name__ == "__main__":
    main()
