#!/usr/bin/env python3
"""Create FEM-labelled dangerous variants while preserving task-level splits.

Unlike the earlier v13 augmentation builder, this utility can augment a chosen
split.  It is intended for a *new* sealed evaluation extension: generated rows
inherit the source task split, and are never added to model training.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from build_dangerous_fem_v13_20260716 import (  # noqa: E402
    bin_index,
    parse_bins,
    resolve_row_path,
    stable_name,
    variants,
    volume_digest,
)
from build_taskgrouped_decision_fem_dataset_20260711 import save_one  # noqa: E402
from structural.analysis.load_generator import LoadGenerator  # noqa: E402


def select_seeds(payload: dict, low: float, high: float, per_task: int) -> list[dict]:
    """Task-interleaved source rows; the caller has already frozen one split."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in payload["rows"]:
        sf = float(row.get("sf", float("nan")))
        if low <= sf <= high:
            by_task[row["task"]].append(row)
    selected: list[dict] = []
    for rank in range(per_task):
        for task in sorted(by_task):
            rows = sorted(by_task[task], key=lambda r: (abs(float(r["sf"]) - 1.3), str(r.get("candidate"))))
            if rank < len(rows):
                selected.append(rows[rank])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--source-split", choices=("train", "calibration", "sealed_test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sf-low", type=float, default=0.90)
    parser.add_argument("--source-sf-high", type=float, default=3.50)
    parser.add_argument("--target-bins", required=True)
    parser.add_argument("--max-per-task", type=int, default=32)
    parser.add_argument("--seeds-per-task", type=int, default=24)
    parser.add_argument("--max-evaluations", type=int, default=20000)
    args = parser.parse_args()

    source = json.loads(args.source_index.read_text())
    # Reuse the original task-balanced ordering, but restrict it to the split
    # whose task ownership we are extending.
    split_rows = {**source, "rows": [r for r in source["rows"] if r["split"] == args.source_split]}
    bins = parse_bins(args.target_bins)
    target_total = sum(int(b["target"]) for b in bins)
    args.output.mkdir(parents=True, exist_ok=True)
    out_index = args.output / "index.json"
    payload = {
        "schema": "shadowforge_dangerous_split_extension_v1",
        "source": str(args.source_index.resolve()),
        "source_split": args.source_split,
        "split_rule": "generated rows retain source task ownership; no task crosses splits",
        "label_protocol": "fresh full-field Hex8 FEM under the standing self-weight protocol",
        "target_bins": bins,
        "rows": [],
    }
    if out_index.is_file():
        payload = json.loads(out_index.read_text())
        bins = payload["target_bins"]
        target_total = sum(int(b["target"]) for b in bins)

    accepted_per_task: dict[str, int] = {}
    seen: set[str] = set()
    done: set[tuple] = set()
    counts = [0] * len(bins)
    for row in payload["rows"]:
        accepted_per_task[row["task"]] = accepted_per_task.get(row["task"], 0) + 1
        seen.add(str(row["volume_hash"]))
        done.add((row["task"], row.get("case_id"), row.get("source_candidate"), row["operation"]))
        i = bin_index(float(row["sf"]), bins)
        if i is not None:
            counts[i] += 1

    seeds = select_seeds(split_rows, args.source_sf_low, args.source_sf_high, args.seeds_per_task)
    generator = LoadGenerator()
    evaluations = 0
    for seed in seeds:
        if len(payload["rows"]) >= target_total or evaluations >= args.max_evaluations:
            break
        if accepted_per_task.get(seed["task"], 0) >= args.max_per_task:
            continue
        with __import__("numpy").load(resolve_row_path(args.source_index, seed)) as archive:
            density = archive["density"] > 0.5
        for operation, volume, ratio in variants(density):
            if len(payload["rows"]) >= target_total or evaluations >= args.max_evaluations:
                break
            if accepted_per_task.get(seed["task"], 0) >= args.max_per_task:
                break
            key = (seed["task"], seed.get("case_id"), seed.get("candidate"), operation)
            digest = volume_digest(volume)
            if key in done or digest in seen:
                continue
            evaluations += 1
            path = args.output / args.source_split / stable_name(seed, operation)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                values = save_one(volume, path, generator, store_nodal_displacement=True)
                i = bin_index(float(values["sf"]), bins)
                keep = i is not None and counts[i] < int(bins[i]["target"]) and int(values["n_disconnected"]) == 0 and int(volume.sum()) >= 256
                if keep:
                    row = {
                        "split": args.source_split, "task": seed["task"], "triple": list(seed.get("triple", [])),
                        "branch": seed.get("branch"), "case_id": seed.get("case_id"),
                        "candidate": f"{seed.get('candidate')}__danger_{operation}",
                        "parent_candidate": seed.get("candidate"), "source_candidate": seed.get("candidate"),
                        "operation": operation, "source_sf": float(seed["sf"]),
                        "volume_ratio_to_source": ratio, "volume_hash": digest,
                        "augmentation_generation": "v20_split_extension", "is_active_variant": True,
                        "structural_representation": seed.get("structural_representation", "conservative_delivered_proxy"),
                        "path": str(path.relative_to(args.output)), "occupied_voxels": int(volume.sum()), **values,
                    }
                    payload["rows"].append(row)
                    accepted_per_task[seed["task"]] = accepted_per_task.get(seed["task"], 0) + 1
                    counts[i] += 1
                    seen.add(digest); done.add(key)
                    print(json.dumps({"status": "saved", "task": seed["task"], "sf": values["sf"], "bin": i}), flush=True)
                else:
                    path.unlink(missing_ok=True)
                    print(json.dumps({
                        "status": "rejected", "task": seed["task"], "sf": float(values["sf"]),
                        "bin": i, "components": int(values["n_disconnected"]),
                        "voxels": int(volume.sum()),
                    }), flush=True)
            except Exception as exc:
                path.unlink(missing_ok=True)
                print(json.dumps({"status": "failed", "task": seed["task"], "error": repr(exc)}), flush=True)
            payload["counts"] = {"rows": len(payload["rows"]), "tasks": len({r["task"] for r in payload["rows"]}), "bins": [{**b, "accepted": counts[j]} for j, b in enumerate(bins)]}
            payload["evaluations"] = evaluations
            out_index.write_text(json.dumps(payload, indent=2) + "\n")

    payload["complete"] = len(payload["rows"]) >= target_total or evaluations >= args.max_evaluations
    payload["counts"] = {"rows": len(payload["rows"]), "tasks": len({r["task"] for r in payload["rows"]}), "bins": [{**b, "accepted": counts[j]} for j, b in enumerate(bins)]}
    payload["evaluations"] = evaluations
    out_index.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(out_index), "counts": payload["counts"], "evaluations": evaluations, "complete": payload["complete"]}, indent=2))


if __name__ == "__main__":
    main()
