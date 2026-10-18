#!/usr/bin/env python3
"""Freeze an equal-class diagnostic index without altering the source test set."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TAU = 1.3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-index", type=Path, required=True)
    p.add_argument("--split", default="sealed_test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-class", type=int, default=105)
    p.add_argument("--seed", type=int, default=20260720)
    p.add_argument("--match-unsafe-tasks", action="store_true",
                   help="sample safe rows from the same task distribution as unsafe rows")
    p.add_argument("--restrict-safe-to-unsafe-task-set", action="store_true",
                   help="sample safe rows only from tasks that also contain unsafe rows")
    args = p.parse_args()
    source = json.loads(args.source_index.read_text())
    rows = [dict(r) for r in source["rows"] if r["split"] == args.split]
    safe = [r for r in rows if float(r["sf"]) >= TAU]
    unsafe = [r for r in rows if float(r["sf"]) < TAU]
    n = min(args.per_class, len(safe), len(unsafe))
    rng = random.Random(args.seed)
    # Keep unsafe ordering reproducible but stratified around the boundary.
    unsafe.sort(key=lambda r: (abs(float(r["sf"]) - TAU), str(r["path"])))
    unsafe = unsafe[:n]
    unsafe_tasks = {row["task"] for row in unsafe}
    if args.restrict_safe_to_unsafe_task_set:
        safe = [row for row in safe if row["task"] in unsafe_tasks]
    if args.match_unsafe_tasks:
        safe_by_task: dict[str, list[dict]] = {}
        for row in safe:
            safe_by_task.setdefault(row["task"], []).append(row)
        unsafe_by_task: dict[str, list[dict]] = {}
        for row in unsafe:
            unsafe_by_task.setdefault(row["task"], []).append(row)
        selected_safe: list[dict] = []
        for task, task_unsafe in sorted(unsafe_by_task.items()):
            candidates = safe_by_task.get(task, [])
            rng.shuffle(candidates)
            if len(candidates) < len(task_unsafe):
                raise RuntimeError(f"task {task} has insufficient safe rows for matched diagnostic")
            selected_safe.extend(candidates[:len(task_unsafe)])
        safe = selected_safe
    else:
        rng.shuffle(safe)
        safe = safe[:n]
    chosen = safe + unsafe
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema": "balanced_taskdisjoint_pino_diagnostic_v1",
        "source": str(args.source_index.resolve()), "source_split": args.split,
        "purpose": "equal-class diagnostic only; source natural-distribution audit remains primary",
        "task_matching": bool(args.match_unsafe_tasks),
        "safe_task_set_restricted": bool(args.restrict_safe_to_unsafe_task_set),
        "threshold": TAU, "safe": n, "unsafe": n, "rows": chosen,
    }, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "safe": n, "unsafe": n, "total": len(chosen)}, indent=2))


if __name__ == "__main__":
    main()
