#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure, label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
for p in [ROOT, ROOT / "generation", ROOT / "experiments"]:
    sys.path.insert(0, str(p))

from datasets import load_data_from_list
from voxel_ops import build_tfs_visual_hull, calibrate_tfs_projection, evaluate_tfs_volume, sdf_watertight_mesh

SRC_ROOT = ROOT / "results/new_anime30_authoritative_v29_48core_256delivery_20260707"
OUT = ROOT / "results/new_anime30_delivery_repair_scan_20260707"
S6 = generate_binary_structure(3, 1)


def keep_largest(volume: np.ndarray) -> np.ndarray:
    labels, count = label(volume.astype(bool), structure=S6)
    if count <= 1:
        return volume.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def load_masks(paths: list[str], resolution: int):
    sils, _ = load_data_from_list(paths[:3], True, (resolution, resolution), torch.device("cpu"), 3, False)
    unique = [np.asarray(x).squeeze().astype(np.float32) for x in sils[:3]]
    return unique + [np.fliplr(x) for x in unique]


def mesh_parts(volume: np.ndarray) -> tuple[bool, int]:
    try:
        mesh = sdf_watertight_mesh(volume.astype(np.uint8), smooth_sigma=0.6)
        return bool(mesh.is_watertight and mesh.is_winding_consistent), int(len(mesh.split(only_watertight=False)))
    except Exception:
        return False, 0


def variants(volume: np.ndarray):
    yield "raw", volume.astype(bool)
    for it in [1, 2, 3]:
        yield f"close{it}", binary_closing(volume, structure=S6, iterations=it)
    for it in [1, 2]:
        closed = binary_closing(volume, structure=S6, iterations=it)
        yield f"close{it}_fill", binary_fill_holes(closed)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for manifest_path in sorted(SRC_ROOT.glob("*/MANIFEST.json")):
        manifest = json.load(open(manifest_path, encoding="utf-8"))
        delivery = manifest.get("delivery") or {}
        if not delivery.get("artifact"):
            continue
        case = manifest_path.parent.name
        volume = np.load(delivery["artifact"]) > 0
        input_paths = [row["path"] for row in manifest["inputs"]]
        masks = load_masks(input_paths, 256)
        cal = calibrate_tfs_projection(masks, search_res=96, fix_axis=True)
        hull, _ = build_tfs_visual_hull(masks, 256, calibration=cal)
        hull = hull.astype(bool)
        case_dir = OUT / case
        case_dir.mkdir(parents=True, exist_ok=True)
        for name, candidate in variants(volume):
            candidate = keep_largest(candidate & hull)
            ev = evaluate_tfs_volume(candidate.astype(np.float32), masks, cal, image_size=512)
            wt, parts = mesh_parts(candidate)
            row = {
                "case": case,
                "variant": name,
                "mean_iou": float(ev["mean_iou"]),
                "min_iou": float(ev["min_iou"]),
                "saving": float(1.0 - int(candidate.sum()) / max(int(hull.sum()), 1)),
                "component_count_6": 1,
                "watertight": wt,
                "mesh_parts": parts,
                "pass_print_gate": bool(
                    ev["mean_iou"] >= 0.95
                    and int(parts) == 1
                    and wt
                    and 0.40 <= (1.0 - int(candidate.sum()) / max(int(hull.sum()), 1)) <= 0.85
                ),
                "artifact": str(case_dir / f"{name}.npy"),
            }
            np.save(row["artifact"], candidate.astype(np.uint8))
            rows.append(row)
            print(case, name, "pass", row["pass_print_gate"], "iou", round(row["mean_iou"], 4), "min", round(row["min_iou"], 4), "saving", round(row["saving"], 4), "parts", parts, "wt", wt, flush=True)
    (OUT / "SUMMARY.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
