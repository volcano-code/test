#!/usr/bin/env python3
"""Add task-disjoint, FEM-verified near-boundary unsafe training samples.

The source index is never modified.  Only rows owned by the training split are
used as seeds, and every retained variant receives a fresh full-field Hex8 FEM
label.  Calibration and sealed-test tasks are therefore unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure, label

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from build_taskgrouped_decision_fem_dataset_20260711 import save_one  # noqa: E402
from structural.analysis.load_generator import LoadGenerator  # noqa: E402

S6 = generate_binary_structure(3, 1)


def largest_component(volume: np.ndarray) -> np.ndarray:
    labels, count = label(volume.astype(bool), structure=S6)
    if count <= 1:
        return volume.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def axis_structure(axis: int) -> np.ndarray:
    structure = np.zeros((3, 3, 3), dtype=bool)
    index = [1, 1, 1]
    for offset in range(3):
        index[axis] = offset
        structure[tuple(index)] = True
    return structure


def variants(volume: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Generate deterministic thinning steps near a fabrication trajectory."""
    base = largest_component(volume)
    out: list[tuple[str, np.ndarray]] = []
    for axis, axis_name in enumerate("xyz"):
        structure = axis_structure(axis)
        for steps in (1, 2):
            changed = binary_erosion(base, structure=structure, iterations=steps)
            changed = largest_component(changed)
            if changed.any() and not np.array_equal(changed, base):
                out.append((f"axis_{axis_name}_erode{steps}", changed))
    changed = largest_component(binary_erosion(base, structure=S6, iterations=1))
    if changed.any() and not np.array_equal(changed, base):
        out.append(("cross_erode1", changed))
    return out


def resolve_row_path(index: Path, row: dict) -> Path:
    path = Path(row["path"])
    return path if path.is_absolute() else index.parent / path


def stable_name(row: dict, operation: str) -> str:
    raw = "|".join(map(str, (
        row["task"], row.get("branch"), row.get("case_id"),
        row.get("candidate"), operation,
    )))
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"{row.get('branch', 'fixed')}_{int(row.get('case_id', 0)):03d}_{operation}_{digest}.npz"


def volume_digest(volume: np.ndarray) -> str:
    packed = np.packbits(volume.astype(np.uint8), axis=None)
    return hashlib.sha1(packed.tobytes()).hexdigest()


def source_rows(payload: dict, low: float, high: float, seeds_per_task: int) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in payload["rows"]:
        sf = float(row.get("sf", np.nan))
        if row.get("split") == "train" and np.isfinite(sf) and low <= sf <= high:
            by_task[row["task"]].append(row)

    # Round-robin task ordering gives tasks without an existing unsafe example
    # first priority and prevents a few dense candidate pools from dominating.
    unsafe_tasks = {
        row["task"] for row in payload["rows"]
        if row.get("split") == "train" and float(row.get("sf", np.inf)) < 1.3
    }
    ordered: list[dict] = []
    for task in sorted(by_task, key=lambda key: (key in unsafe_tasks, key)):
        rows = sorted(by_task[task], key=lambda row: abs(float(row["sf"]) - 1.3))
        ordered.extend(rows[:seeds_per_task])
    return ordered


def write_index(path: Path, payload: dict, started: float) -> None:
    payload["elapsed_s"] = time.time() - started
    payload["counts"] = {
        "rows": len(payload["rows"]),
        "tasks": len({row["task"] for row in payload["rows"]}),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sf-low", type=float, default=1.05)
    parser.add_argument("--source-sf-high", type=float, default=2.20)
    parser.add_argument("--target-sf-low", type=float, default=0.75)
    parser.add_argument("--target-sf-high", type=float, default=1.30)
    parser.add_argument("--target-count", type=int, default=300)
    parser.add_argument("--max-per-task", type=int, default=4)
    parser.add_argument("--seeds-per-task", type=int, default=3)
    parser.add_argument("--max-evaluations", type=int, default=1800)
    args = parser.parse_args()
    if not (0 < args.target_sf_low < args.target_sf_high <= 1.3):
        raise ValueError("The retained interval must be on the unsafe side of SF=1.3")

    source = json.loads(args.source_index.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.json"
    payload = {
        "schema": "shadowforge_near_boundary_unsafe_fullfield_fem_v12",
        "source": str(args.source_index.resolve()),
        "split": "train-only augmentation",
        "split_rule": "source task ownership is preserved; calibration and sealed_test are untouched",
        "label_protocol": "fresh full-field Hex8 FEM under the standing self-weight protocol",
        "source_sf_interval": [args.source_sf_low, args.source_sf_high],
        "target_sf_interval": [args.target_sf_low, args.target_sf_high],
        "rows": [],
    }
    if index_path.is_file():
        payload = json.loads(index_path.read_text())
    done = {
        (row["task"], row.get("source_candidate"), row.get("operation"))
        for row in payload["rows"]
    }
    accepted_per_task: dict[str, int] = defaultdict(int)
    seen_volumes: set[str] = set()
    evaluated_volumes: set[str] = set()
    for row in payload["rows"]:
        accepted_per_task[row["task"]] += 1
        digest = row.get("volume_hash")
        if digest:
            seen_volumes.add(str(digest))
            evaluated_volumes.add(str(digest))

    seeds = source_rows(source, args.source_sf_low, args.source_sf_high, args.seeds_per_task)
    generator = LoadGenerator()
    started = time.time()
    evaluations = 0
    for seed in seeds:
        if len(payload["rows"]) >= args.target_count or evaluations >= args.max_evaluations:
            break
        if accepted_per_task[seed["task"]] >= args.max_per_task:
            continue
        source_path = resolve_row_path(args.source_index, seed)
        with np.load(source_path) as archive:
            density = np.asarray(archive["density"]) > 0.5
        for operation, volume in variants(density):
            if len(payload["rows"]) >= args.target_count or evaluations >= args.max_evaluations:
                break
            key = (seed["task"], seed.get("candidate"), operation)
            if key in done or accepted_per_task[seed["task"]] >= args.max_per_task:
                continue
            digest = volume_digest(volume)
            if digest in evaluated_volumes:
                continue
            evaluated_volumes.add(digest)
            evaluations += 1
            path = args.output / "train" / stable_name(seed, operation)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                values = save_one(volume, path, generator, store_nodal_displacement=True)
                sf = float(values["sf"])
                keep = (args.target_sf_low <= sf < args.target_sf_high
                        and int(values["n_disconnected"]) == 0)
                event = {
                    "status": "saved" if keep else "rejected",
                    "evaluation": evaluations,
                    "task": seed["task"],
                    "source_candidate": seed.get("candidate"),
                    "operation": operation,
                    "source_sf": float(seed["sf"]),
                    "sf": sf,
                    "n_disconnected": int(values["n_disconnected"]),
                }
                if keep:
                    row = {
                        "split": "train",
                        "task": seed["task"],
                        "triple": list(seed.get("triple", [])),
                        "branch": seed.get("branch"),
                        "case_id": seed.get("case_id"),
                        "candidate": f"{seed.get('candidate')}__nbunsafe_{operation}",
                        "parent_candidate": seed.get("candidate"),
                        "source_candidate": seed.get("candidate"),
                        "operation": operation,
                        "source_sf": float(seed["sf"]),
                        "volume_hash": digest,
                        "is_active_variant": True,
                        "structural_representation": seed.get("structural_representation", "conservative_delivered_proxy"),
                        "path": str(path.relative_to(args.output)),
                        "occupied_voxels": int(volume.sum()),
                        **values,
                    }
                    payload["rows"].append(row)
                    accepted_per_task[seed["task"]] += 1
                    seen_volumes.add(digest)
                    done.add(key)
                    write_index(index_path, payload, started)
                else:
                    path.unlink(missing_ok=True)
                print(json.dumps(event), flush=True)
            except Exception as exc:
                path.unlink(missing_ok=True)
                print(json.dumps({
                    "status": "failed", "evaluation": evaluations,
                    "task": seed["task"], "operation": operation,
                    "error": repr(exc),
                }), flush=True)

    payload["evaluations"] = evaluations
    payload["complete"] = len(payload["rows"]) >= args.target_count or evaluations >= args.max_evaluations
    write_index(index_path, payload, started)
    print(json.dumps({
        "output": str(index_path), "accepted": len(payload["rows"]),
        "tasks": len({row["task"] for row in payload["rows"]}),
        "evaluations": evaluations,
        "elapsed_s": payload["elapsed_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
