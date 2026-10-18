#!/usr/bin/env python3
"""Build a task-balanced, FEM-labelled unsafe training augmentation.

Only train-owned source tasks are used. Calibration and sealed-test files are
never modified. Every retained density receives a fresh full-field Hex8 FEM
label under the standing self-weight protocol.
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


def local_mask(shape: tuple[int, int, int], axis: int, start: float, stop: float) -> np.ndarray:
    lo = max(0, min(shape[axis], int(round(start * shape[axis]))))
    hi = max(lo + 1, min(shape[axis], int(round(stop * shape[axis]))))
    slices = [slice(None)] * 3
    slices[axis] = slice(lo, hi)
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(slices)] = True
    return mask


def variants(volume: np.ndarray):
    """Yield global and local thinning variants along a fabrication trajectory."""
    base = largest_component(volume)
    base_count = int(base.sum())
    seen: set[str] = set()

    def emit(name: str, changed: np.ndarray):
        changed = largest_component(changed)
        if not changed.any() or np.array_equal(changed, base):
            return None
        ratio = float(changed.sum()) / max(base_count, 1)
        if ratio < 0.22:
            return None
        digest = volume_digest(changed)
        if digest in seen:
            return None
        seen.add(digest)
        return name, changed, ratio

    for erosion_axis, erosion_name in enumerate("xyz"):
        structure = axis_structure(erosion_axis)
        for steps in (1, 2, 3):
            item = emit(
                f"global_{erosion_name}_erode{steps}",
                binary_erosion(base, structure=structure, iterations=steps),
            )
            if item:
                yield item

    for steps in (1, 2):
        item = emit(
            f"global_cross_erode{steps}",
            binary_erosion(base, structure=S6, iterations=steps),
        )
        if item:
            yield item

    # Local thinning creates neck/support-sensitive examples without reducing
    # the entire object to an out-of-domain sparse remnant.
    windows = ((0.0, 0.35, "lo35"), (0.0, 0.5, "lo50"),
               (0.5, 1.0, "hi50"), (0.65, 1.0, "hi35"),
               (0.3, 0.7, "mid40"))
    for mask_axis, mask_name in enumerate("xyz"):
        for start, stop, window_name in windows:
            mask = local_mask(base.shape, mask_axis, start, stop)
            for erosion_axis, erosion_name in enumerate("xyz"):
                structure = axis_structure(erosion_axis)
                for steps in (1, 2):
                    eroded = binary_erosion(base, structure=structure, iterations=steps)
                    changed = base.copy()
                    changed[mask] = eroded[mask]
                    item = emit(
                        f"local_{mask_name}{window_name}_{erosion_name}erode{steps}",
                        changed,
                    )
                    if item:
                        yield item


def volume_digest(volume: np.ndarray) -> str:
    return hashlib.sha1(np.packbits(volume.astype(np.uint8), axis=None).tobytes()).hexdigest()


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


def parse_bins(spec: str) -> list[dict]:
    bins = []
    for item in spec.split(","):
        low, high, target = item.split(":")
        bins.append({"low": float(low), "high": float(high), "target": int(target)})
    if not bins or any(not (0 < row["low"] < row["high"] <= 1.3) for row in bins):
        raise ValueError("All target bins must lie on the unsafe side of SF=1.3")
    return bins


def bin_index(sf: float, bins: list[dict]) -> int | None:
    for index, row in enumerate(bins):
        if row["low"] <= sf < row["high"]:
            return index
    return None


def source_rows(payload: dict, low: float, high: float, per_task: int) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in payload["rows"]:
        sf = float(row.get("sf", np.nan))
        if row.get("split") == "train" and np.isfinite(sf) and low <= sf <= high:
            by_task[row["task"]].append(row)
    selected: list[dict] = []
    # Interleave ranks across tasks so broad task coverage precedes extra seeds.
    for rank in range(per_task):
        for task in sorted(by_task):
            rows = sorted(
                by_task[task],
                key=lambda row: (abs(float(row["sf"]) - 1.3), str(row.get("candidate"))),
            )
            if rank < len(rows):
                selected.append(rows[rank])
    return selected


def write_index(path: Path, payload: dict, started: float, bin_counts: list[int]) -> None:
    payload["elapsed_s"] = time.time() - started
    payload["counts"] = {
        "rows": len(payload["rows"]),
        "tasks": len({row["task"] for row in payload["rows"]}),
        "bins": [
            {**spec, "accepted": bin_counts[index]}
            for index, spec in enumerate(payload["target_bins"])
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--prior-augmentation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sf-low", type=float, default=0.90)
    parser.add_argument("--source-sf-high", type=float, default=3.50)
    parser.add_argument("--target-bins", default="0.35:0.75:300,0.75:0.95:300,0.95:1.15:450,1.15:1.30:450")
    parser.add_argument("--max-per-task", type=int, default=16)
    parser.add_argument("--seeds-per-task", type=int, default=12)
    parser.add_argument("--max-evaluations", type=int, default=8000)
    args = parser.parse_args()

    bins = parse_bins(args.target_bins)
    target_count = sum(row["target"] for row in bins)
    source = json.loads(args.source_index.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.json"
    payload = {
        "schema": "shadowforge_dangerous_fullfield_fem_v13",
        "source": str(args.source_index.resolve()),
        "split": "train-only augmentation",
        "split_rule": "source task ownership is preserved; calibration and sealed_test are untouched",
        "label_protocol": "fresh full-field Hex8 FEM under the standing self-weight protocol",
        "source_sf_interval": [args.source_sf_low, args.source_sf_high],
        "target_bins": bins,
        "rows": [],
    }
    if index_path.is_file():
        payload = json.loads(index_path.read_text())
        bins = payload["target_bins"]
        target_count = sum(row["target"] for row in bins)
    elif args.prior_augmentation and args.prior_augmentation.is_file():
        prior = json.loads(args.prior_augmentation.read_text())
        for row in prior["rows"]:
            copied = dict(row)
            copied["path"] = str(resolve_row_path(args.prior_augmentation, row).resolve())
            copied["augmentation_generation"] = "v12"
            payload["rows"].append(copied)

    accepted_per_task: dict[str, int] = defaultdict(int)
    seen_volumes: set[str] = set()
    done: set[tuple] = set()
    bin_counts = [0] * len(bins)
    for row in payload["rows"]:
        accepted_per_task[row["task"]] += 1
        if row.get("volume_hash"):
            seen_volumes.add(str(row["volume_hash"]))
        done.add((row["task"], row.get("case_id"), row.get("source_candidate"), row.get("operation")))
        index = bin_index(float(row["sf"]), bins)
        if index is not None:
            bin_counts[index] += 1

    seeds = source_rows(source, args.source_sf_low, args.source_sf_high, args.seeds_per_task)
    generator = LoadGenerator()
    started = time.time()
    evaluations = 0
    for seed in seeds:
        if len(payload["rows"]) >= target_count or evaluations >= args.max_evaluations:
            break
        if accepted_per_task[seed["task"]] >= args.max_per_task:
            continue
        source_path = resolve_row_path(args.source_index, seed)
        with np.load(source_path) as archive:
            density = np.asarray(archive["density"]) > 0.5
        source_count = int(density.sum())
        for operation, volume, volume_ratio in variants(density):
            if len(payload["rows"]) >= target_count or evaluations >= args.max_evaluations:
                break
            if accepted_per_task[seed["task"]] >= args.max_per_task:
                break
            key = (seed["task"], seed.get("case_id"), seed.get("candidate"), operation)
            if key in done:
                continue
            digest = volume_digest(volume)
            if digest in seen_volumes:
                continue
            seen_volumes.add(digest)
            evaluations += 1
            path = args.output / "train" / stable_name(seed, operation)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                values = save_one(volume, path, generator, store_nodal_displacement=True)
                sf = float(values["sf"])
                index = bin_index(sf, bins)
                keep = (
                    index is not None
                    and bin_counts[index] < int(bins[index]["target"])
                    and int(values["n_disconnected"]) == 0
                    and int(volume.sum()) >= 256
                )
                event = {
                    "status": "saved" if keep else "rejected",
                    "evaluation": evaluations,
                    "task": seed["task"],
                    "operation": operation,
                    "source_sf": float(seed["sf"]),
                    "sf": sf,
                    "volume_ratio": volume_ratio,
                    "n_disconnected": int(values["n_disconnected"]),
                }
                if keep:
                    row = {
                        "split": "train",
                        "task": seed["task"],
                        "triple": list(seed.get("triple", [])),
                        "branch": seed.get("branch"),
                        "case_id": seed.get("case_id"),
                        "candidate": f"{seed.get('candidate')}__danger_{operation}",
                        "parent_candidate": seed.get("candidate"),
                        "source_candidate": seed.get("candidate"),
                        "operation": operation,
                        "source_sf": float(seed["sf"]),
                        "source_occupied_voxels": source_count,
                        "volume_ratio_to_source": volume_ratio,
                        "volume_hash": digest,
                        "augmentation_generation": "v13",
                        "is_active_variant": True,
                        "structural_representation": seed.get(
                            "structural_representation", "conservative_delivered_proxy"
                        ),
                        "path": str(path.relative_to(args.output)),
                        "occupied_voxels": int(volume.sum()),
                        **values,
                    }
                    payload["rows"].append(row)
                    accepted_per_task[seed["task"]] += 1
                    bin_counts[index] += 1
                    done.add(key)
                    write_index(index_path, payload, started, bin_counts)
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
    payload["complete"] = (
        all(bin_counts[index] >= int(row["target"]) for index, row in enumerate(bins))
        or evaluations >= args.max_evaluations
    )
    write_index(index_path, payload, started, bin_counts)
    print(json.dumps({
        "output": str(index_path),
        "accepted": len(payload["rows"]),
        "tasks": len({row["task"] for row in payload["rows"]}),
        "evaluations": evaluations,
        "bins": payload["counts"]["bins"],
        "elapsed_s": payload["elapsed_s"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
