#!/usr/bin/env python3
"""Build true near-boundary FEM labels around current-domain false-safe cases."""

from __future__ import annotations

import json
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, generate_binary_structure, label

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "generation")]

from candidate_selection_batch import TAU, fem_sf, make_candidates  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from structural.analysis.load_generator import LoadGenerator  # noqa: E402
from voxel_ops import compute_dice_np, compute_iou_np, evaluate_tfs_volume, resize_binary_mask  # noqa: E402


SRC = ROOT / "results/candidate_selection_400case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"
OUT = ROOT / "data/near_boundary_current_candidate_fem_20260710"
SIL_DIR = ROOT / "data/silhouettes/bench3"
HARD_KEYS = [
    ("free", 196),
    ("free", 164),
    ("free", 49),
    ("fixed", 2),
    ("fixed", 24),
]
S6 = generate_binary_structure(3, 1)


def load_masks(triple: list[str], resolution: int) -> list[np.ndarray]:
    paths = [SIL_DIR / f"{name}.png" for name in triple]
    tensors, _ = load_data_from_list(
        [str(p) for p in paths],
        True,
        (resolution, resolution),
        torch.device("cpu"),
        3,
        False,
    )
    return [np.asarray(x).squeeze().astype(np.float32) for x in tensors]


def keep_largest(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(bool)
    lab, count = label(vol, structure=S6)
    if count <= 1:
        return vol
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def project_scores(volume: np.ndarray, masks: list[np.ndarray], calibration: dict) -> dict:
    metrics = evaluate_tfs_volume(volume.astype(bool), masks, calibration)
    view_ious, view_dices = [], []
    for pred, gt in zip(metrics["pred_masks"], masks[: len(metrics["pred_masks"])]):
        pr = resize_binary_mask(pred, gt.shape[0])
        gr = resize_binary_mask(gt, gt.shape[0])
        view_ious.append(float(compute_iou_np(pr, gr)))
        view_dices.append(float(compute_dice_np(pr, gr)))
    return {
        "mean_iou": float(np.mean(view_ious)),
        "mean_dice": float(np.mean(view_dices)),
        "view_ious": view_ious,
        "view_dices": view_dices,
    }


def variants(base_name: str, base: np.ndarray, hull: np.ndarray) -> dict[str, np.ndarray]:
    hb = hull.astype(bool)
    b = base.astype(bool) & hb
    out: dict[str, np.ndarray] = {f"{base_name}__base": b}
    if base_name in {"visual_hull", "gap_filled_hull"}:
        e = binary_erosion(b, S6, iterations=1) & hb
        if e.any():
            out[f"{base_name}__erode1"] = e
    else:
        for it in (1, 2, 3, 4):
            out[f"{base_name}__dilate{it}"] = binary_dilation(b, S6, iterations=it) & hb
    return {k: keep_largest(v) for k, v in out.items() if v.any()}


def select_hard_keys(source: dict, count: int) -> list[tuple[str, int]]:
    """Choose current-domain cases carrying both sides of the FEM boundary."""
    ranked = []
    for case in source["cases"]:
        sf = np.asarray([float(row["sf_fem"]) for row in case["candidates"]])
        near_safe = int(((sf >= TAU) & (sf < 2.0 * TAU)).sum())
        near_unsafe = int(((sf >= 0.5 * TAU) & (sf < TAU)).sum())
        if near_safe and near_unsafe:
            ranked.append((near_safe + near_unsafe, near_safe, near_unsafe,
                           str(case["branch"]), int(case["case_id"])))
    ranked.sort(reverse=True)
    return [(branch, case_id) for _, _, _, branch, case_id in ranked[:count]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=SRC)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-cases", type=int, default=0,
                        help="Automatically select this many mixed near-boundary cases.")
    args = parser.parse_args()
    source = json.loads(args.src.read_text())
    resolution = int(source["resolution"])
    cases = {(row["branch"], int(row["case_id"])): row for row in source["cases"]}
    hard_keys = select_hard_keys(source, args.n_cases) if args.n_cases else HARD_KEYS
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"
    rows = []
    done = set()
    if index_path.exists():
        old = json.loads(index_path.read_text())
        rows = list(old.get("rows", []))
        done = {(r["branch"], int(r["case_id"]), r["candidate"]) for r in rows}
    generator = LoadGenerator()
    t0 = time.time()

    for branch, case_id in hard_keys:
        case = cases[(branch, case_id)]
        masks = load_masks(case["triple"], resolution)
        candidates, aux = make_candidates(masks, resolution, branch == "fixed")
        hull = aux["hull"].astype(bool)
        calibration = aux["calibration"]
        case_dir = args.out / f"{branch}_{case_id:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        expanded = {}
        for name, vol in candidates.items():
            if name in {"visual_hull", "gap_filled_hull", "carve_no_bridge", "carve_thick2", "carve_thick3", "min_printable_bridge_r1", "watertight_export"}:
                expanded.update(variants(name, vol, hull))

        seen = set()
        for name, vol in expanded.items():
            if (branch, int(case_id), name) in done:
                continue
            packed = np.packbits(vol.astype(np.uint8)).tobytes()
            if packed in seen:
                continue
            seen.add(packed)
            force = generator.standing_load(vol.astype(np.float32))["force"].astype(np.float32)
            try:
                fem = fem_sf(vol, 1.0)
                qa = project_scores(vol, masks, calibration)
            except Exception as exc:
                print(json.dumps({
                    "status": "skipped",
                    "branch": branch,
                    "case_id": int(case_id),
                    "candidate": name,
                    "error": repr(exc),
                }), flush=True)
                continue
            npz = case_dir / f"{name}.npz"
            np.savez_compressed(npz, density=vol.astype(np.float32), force=force, sf=np.float32(fem["sf_fem"] or -1.0))
            rows.append({
                "source": "near_boundary_current_candidate_fem_20260710",
                "group": f"{branch}:{case_id}",
                "branch": branch,
                "case_id": int(case_id),
                "triple": case["triple"],
                "candidate": name,
                "density_path": str(npz),
                "sf": float(fem["sf_fem"] or -1.0),
                "fem_safe": bool(fem["safe"]),
                "fem_elapsed_s": float(fem["elapsed_s"]),
                "mean_iou": qa["mean_iou"],
                "mean_dice": qa["mean_dice"],
                "occupied_voxels": int(vol.sum()),
            })
            print(json.dumps(rows[-1]), flush=True)
            payload = {
                "schema": "near_boundary_current_candidate_fem_v1",
                "source_candidates": str(SRC),
                "resolution": resolution,
                "tau": TAU,
                "hard_keys": [{"branch": b, "case_id": c} for b, c in hard_keys],
                "n_rows": len(rows),
                "safe": int(sum(r["fem_safe"] for r in rows)),
                "unsafe": int(sum(not r["fem_safe"] for r in rows)),
                "elapsed_s": time.time() - t0,
                "rows": rows,
            }
            index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    payload = {
        "schema": "near_boundary_current_candidate_fem_v1",
        "source_candidates": str(SRC),
        "resolution": resolution,
        "tau": TAU,
        "hard_keys": [{"branch": b, "case_id": c} for b, c in hard_keys],
        "n_rows": len(rows),
        "safe": int(sum(r["fem_safe"] for r in rows)),
        "unsafe": int(sum(not r["fem_safe"] for r in rows)),
        "elapsed_s": time.time() - t0,
        "rows": rows,
    }
    index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("n_rows", "safe", "unsafe", "elapsed_s")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
