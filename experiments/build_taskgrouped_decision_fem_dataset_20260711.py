#!/usr/bin/env python3
"""Create a frozen, task-grouped FEM data set for safe/unsafe decisions.

All candidates derived from the same ordered silhouette triple are assigned to
one split.  Base candidates are regenerated from the current generator and
re-solved with voxel FEM.  Near the safety threshold, controlled erosion and
dilation variants are added and also re-solved.  Every saved sample carries the
exact Hex8 nodal displacement field; no continuous PDE residual is assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, label

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "generation")]

from candidate_selection_batch import TAU, make_candidates  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from voxel_ops import build_tfs_visual_hull, calibrate_tfs_projection  # noqa: E402
from structural.analysis.fem_solver import voxel_fem_solve  # noqa: E402
from structural.analysis.load_generator import LoadGenerator  # noqa: E402

SRC = ROOT / "results/candidate_selection_400case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"
SIL_DIR = ROOT / "data/silhouettes/bench3"
S6 = generate_binary_structure(3, 1)


def task_key(triple: list[str]) -> str:
    return "|".join(map(str, triple))


def largest_component(volume: np.ndarray) -> np.ndarray:
    labels, n = label(volume.astype(bool), structure=S6)
    if n <= 1:
        return volume.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def load_masks(triple: list[str], resolution: int) -> list[np.ndarray]:
    tensors, _ = load_data_from_list(
        [str(SIL_DIR / f"{name}.png") for name in triple],
        True,
        (resolution, resolution),
        torch.device("cpu"),
        3,
        False,
    )
    return [np.asarray(x).squeeze().astype(np.float32) for x in tensors]


def stratified_task_splits(cases: list[dict], seed: int) -> dict[str, set[str]]:
    """Split triples, stratified by whether their pool straddles SF=tau."""
    stats: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        stats[task_key(case["triple"])].extend(float(row["sf_fem"]) for row in case["candidates"])
    buckets: dict[str, list[str]] = defaultdict(list)
    for task, sf_values in stats.items():
        sf = np.asarray(sf_values, dtype=float)
        near_safe = np.any((sf >= TAU) & (sf <= 2.0 * TAU))
        near_unsafe = np.any((sf >= 0.5 * TAU) & (sf < TAU))
        buckets[f"{int(near_safe)}{int(near_unsafe)}"].append(task)
    rng = np.random.default_rng(seed)
    out = {"train": set(), "calibration": set(), "sealed_test": set()}
    for tasks in buckets.values():
        tasks = list(tasks)
        rng.shuffle(tasks)
        n_train = int(round(0.60 * len(tasks)))
        n_cal = int(round(0.20 * len(tasks)))
        out["train"].update(tasks[:n_train])
        out["calibration"].update(tasks[n_train:n_train + n_cal])
        out["sealed_test"].update(tasks[n_train + n_cal:])
    assert not (out["train"] & out["calibration"] or out["train"] & out["sealed_test"] or out["calibration"] & out["sealed_test"])
    return out


def source_near_boundary(case: dict, band: float) -> list[str]:
    names = []
    for row in case["candidates"]:
        sf = float(row["sf_fem"])
        if sf > 0 and abs(math.log10(sf / TAU)) <= band:
            names.append(str(row["candidate"]))
    return names


def variants(name: str, volume: np.ndarray, hull: np.ndarray, expected_safe: bool,
             iterations: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Perturb toward and across the safety boundary while retaining the hull."""
    base = largest_component(volume & hull)
    out = {}
    op = binary_erosion if expected_safe else binary_dilation
    for n_steps in iterations:
        changed = op(base, structure=S6, iterations=n_steps)
        changed = largest_component(changed & hull)
        if changed.any() and not np.array_equal(changed, base):
            direction = "erode" if expected_safe else "dilate"
            out[f"{name}__{direction}{n_steps}"] = changed
    return out


def file_name(task: str, branch: str, case_id: int, candidate: str) -> str:
    digest = hashlib.sha1(f"{task}|{branch}|{case_id}|{candidate}".encode()).hexdigest()[:12]
    return f"{branch}_{case_id:03d}_{candidate}_{digest}.npz"


def upsample_nearest(core: np.ndarray, target: int) -> np.ndarray:
    idx = np.floor(np.arange(target, dtype=np.float32) * (core.shape[0] / target)).astype(np.int64)
    idx = np.clip(idx, 0, core.shape[0] - 1)
    return core.astype(bool)[np.ix_(idx, idx, idx)]


def conservative_pool(volume: np.ndarray, target: int) -> np.ndarray:
    """Map delivered occupancy to a coarse structural proxy without dropping material."""
    edges = np.rint(np.linspace(0, volume.shape[0], target + 1)).astype(np.int64)
    pooled = np.maximum.reduceat(volume.astype(np.uint8), edges[:-1], axis=0)
    pooled = np.maximum.reduceat(pooled, edges[:-1], axis=1)
    pooled = np.maximum.reduceat(pooled, edges[:-1], axis=2)
    return pooled.astype(bool)


def delivered_proxy(core: np.ndarray, hull256: np.ndarray, shell_voxels: int, proxy_resolution: int) -> tuple[np.ndarray, dict]:
    core_up = upsample_nearest(core, hull256.shape[0]) & hull256
    shell = hull256 & ~binary_erosion(hull256, structure=S6, iterations=shell_voxels)
    delivered = largest_component(core_up | shell)
    proxy = conservative_pool(delivered, proxy_resolution)
    return proxy, {"core_occupied": int(core.sum()), "delivery_occupied": int(delivered.sum()),
                   "proxy_occupied": int(proxy.sum())}


def save_one(volume: np.ndarray, path: Path, generator: LoadGenerator, store_nodal_displacement: bool) -> dict:
    density = volume.astype(np.float32)
    load = generator.standing_load(density)
    fem = voxel_fem_solve(
        density,
        load["force"].astype(np.float32),
        load["fixed_mask"].astype(np.float32),
        return_nodal_displacement=store_nodal_displacement,
        verbose=False,
    )
    solid = density > 0.5
    vm_solid = fem["von_mises"][solid]
    vm_max = float(np.max(vm_solid)) if vm_solid.size else float("inf")
    sf = float(50e6 / max(vm_max, 1e-12))
    payload = {
        "density": density,
        "force": load["force"].astype(np.float32),
        "fixed_mask": load["fixed_mask"].astype(np.float32),
        "sf": np.float32(sf),
        "vm_max": np.float32(vm_max),
    }
    if store_nodal_displacement:
        payload.update({
            "nodal_displacement": fem["nodal_displacement"],
            "nodal_active_mask": fem["nodal_active_mask"],
            "displacement": fem["displacement"].astype(np.float32),
            "stress": fem["stress"].astype(np.float32),
            "von_mises": fem["von_mises"].astype(np.float32),
        })
    np.savez_compressed(path, **payload)
    return {
        "sf": sf,
        "fem_safe": bool(sf >= TAU),
        "vm_max": vm_max,
        "fem_total_time_s": float(fem["total_time_s"]),
        "n_elements": int(fem["n_elements"]),
        "n_disconnected": int(fem["n_disconnected"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SRC)
    parser.add_argument("--out", type=Path, default=ROOT / "data/taskgrouped_decision_fem_v2_20260711")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--near-log-band", type=float, default=0.42)
    parser.add_argument("--max-variants-per-case", type=int, default=4)
    parser.add_argument("--variant-iterations", default="1,2",
                        help="Comma-separated erosion/dilation radii for near-threshold candidates.")
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--store-nodal-displacement", action="store_true",
                        help="Store full nodal and voxel fields; requires substantially more disk space.")
    parser.add_argument("--delivered-proxy", action="store_true",
                        help="Label a conservative 48^3 proxy of the delivered 256^3 core--shell solid.")
    parser.add_argument("--delivery-resolution", type=int, default=256)
    parser.add_argument("--shell-voxels", type=int, default=2)
    args = parser.parse_args()
    variant_iterations = tuple(sorted({int(x) for x in args.variant_iterations.split(",") if x.strip()}))
    if not variant_iterations or min(variant_iterations) < 1:
        raise ValueError("--variant-iterations must contain positive integers")

    source = json.loads(args.src.read_text())
    cases = source["cases"][: args.case_limit or None]
    resolution = int(source["resolution"])
    splits = stratified_task_splits(cases, args.seed)
    split_for_task = {task: split for split, tasks in splits.items() for task in tasks}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "index.json"
    manifest = {
        "schema": "shadowforge_taskgrouped_decision_fem_v2",
        "source": str(args.src),
        "resolution": resolution,
        "tau": TAU,
        "seed": args.seed,
        "split_unit": "ordered silhouette triple; all branches, candidates, and perturbations remain together",
        "label_protocol": ("fresh voxel Hex8 FEM labels on conservative delivered core--shell proxies"
                           if args.delivered_proxy else "fresh voxel Hex8 FEM labels")
                          + (" with saved nodal displacement" if args.store_nodal_displacement else " with scalar safety-factor supervision"),
        "near_boundary_protocol": "near-source candidates receive erosion/dilation variants and fresh FEM labels",
        "tasks": {name: sorted(values) for name, values in splits.items()},
        "rows": [],
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    done = {(row["task"], row["branch"], int(row["case_id"]), row["candidate"]) for row in manifest["rows"]}
    generator = LoadGenerator()
    start = time.time()

    for case in cases:
        task = task_key(case["triple"])
        split = split_for_task[task]
        masks = load_masks(case["triple"], resolution)
        candidates, aux = make_candidates(masks, resolution, case["branch"] == "fixed")
        hull256 = None
        if args.delivered_proxy:
            masks256 = load_masks(case["triple"], args.delivery_resolution)
            cal256 = calibrate_tfs_projection(masks256, search_res=min(96, args.delivery_resolution),
                                               fix_axis=case["branch"] == "fixed")
            hull256, _ = build_tfs_visual_hull(masks256, args.delivery_resolution, calibration=cal256)
            hull256 = hull256.astype(bool)
        recorded = {str(row["candidate"]): row for row in case["candidates"]}
        pool = {name: largest_component(vol.astype(bool)) for name, vol in candidates.items() if name in recorded}
        near_names = source_near_boundary(case, args.near_log_band)
        added = 0
        for name in near_names:
            if added >= args.max_variants_per_case or name not in pool:
                continue
            extra = variants(name, pool[name], aux["hull"].astype(bool),
                             bool(recorded[name].get("fem_safe", False)), variant_iterations)
            for variant_name, variant in extra.items():
                if added >= args.max_variants_per_case:
                    break
                pool[variant_name] = variant
                added += 1
        for name, volume in pool.items():
            key = (task, str(case["branch"]), int(case["case_id"]), name)
            if key in done:
                continue
            path = args.out / split / file_name(task, str(case["branch"]), int(case["case_id"]), name)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                fem_volume, delivery_meta = ((delivered_proxy(volume, hull256, args.shell_voxels, resolution))
                                             if args.delivered_proxy else (volume, {}))
                values = save_one(fem_volume, path, generator, args.store_nodal_displacement)
                row = {
                    "split": split,
                    "task": task,
                    "triple": list(case["triple"]),
                    "branch": str(case["branch"]),
                    "case_id": int(case["case_id"]),
                    "candidate": name,
                    "parent_candidate": name.split("__", 1)[0],
                    "is_active_variant": "__" in name,
                    "structural_representation": "conservative_delivered_proxy" if args.delivered_proxy else "core",
                    **delivery_meta,
                    "path": str(path.relative_to(args.out)),
                    "occupied_voxels": int(volume.sum()),
                    **values,
                }
                manifest["rows"].append(row)
                print(json.dumps({"status": "saved", **row}), flush=True)
            except Exception as exc:
                print(json.dumps({"status": "failed", "task": task, "candidate": name, "error": repr(exc)}), flush=True)
            manifest["elapsed_s"] = time.time() - start
            manifest["counts"] = {name: int(sum(row["split"] == name for row in manifest["rows"])) for name in ("train", "calibration", "sealed_test")}
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["elapsed_s"] = time.time() - start
    manifest["counts"] = {name: int(sum(row["split"] == name for row in manifest["rows"])) for name in ("train", "calibration", "sealed_test")}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(manifest_path), "counts": manifest["counts"], "elapsed_s": manifest["elapsed_s"]}, indent=2))


if __name__ == "__main__":
    main()
