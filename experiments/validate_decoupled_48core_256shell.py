#!/usr/bin/env python3
"""Validate the 48-core / 256-shell resolution-decoupled export rule.

Rule:
  1. Take the certified low-resolution structural core C_low.
  2. Nearest-neighbor upsample to 256 and clip to the 256 visual hull H256.
  3. Build a thin 256 shell: H256 & ~erode(H256, t).
  4. Union clipped core and shell, then keep the largest 6-connected component.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, generate_binary_structure, label, zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generation"))
sys.path.insert(0, str(ROOT / "experiments"))

from datasets import load_data_from_list
from voxel_ops import build_tfs_visual_hull, calibrate_tfs_projection, evaluate_tfs_volume


S6 = generate_binary_structure(3, 1)


def keep_largest(volume: np.ndarray) -> np.ndarray:
    vol = volume.astype(bool)
    lab, count = label(vol, structure=S6)
    if count <= 1:
        return vol
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def n6(volume: np.ndarray) -> int:
    _, count = label(volume.astype(bool), structure=S6)
    return int(count)


def upsample_nearest(core: np.ndarray, target: int = 256) -> np.ndarray:
    core = core.astype(bool)
    factors = [target / s for s in core.shape]
    up = zoom(core.astype(np.uint8), zoom=factors, order=0) > 0
    return up[:target, :target, :target]


def ordered_pngs(input_dir: Path) -> list[Path]:
    paths = list(input_dir.glob("*.png"))
    if any(p.name.startswith("view") for p in paths):
        return sorted(paths)
    preferred = ["spiderman", "superman", "batman"]
    rank = {name: i for i, name in enumerate(preferred)}
    return sorted(paths, key=lambda p: rank.get(p.stem.lower(), 100 + paths.index(p)))


def load_masks(input_dir: Path, resolution: int) -> tuple[list[np.ndarray], list[str]]:
    paths = ordered_pngs(input_dir)
    if len(paths) < 3:
        raise ValueError(f"Need at least 3 input masks in {input_dir}, found {len(paths)}")
    sils, _ = load_data_from_list(
        [str(p) for p in paths[:3]],
        True,
        (resolution, resolution),
        torch.device("cpu"),
        3,
        False,
    )
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils]
    return masks, [p.name for p in paths[:3]]


def validate_case(case_dir: Path, shell_t: int, target: int) -> dict:
    npys = sorted(case_dir.glob("*.npy"))
    if not npys:
        raise ValueError(f"No core npy found in {case_dir}")
    core_path = npys[0]
    core = np.load(core_path) > 0
    masks, mask_names = load_masks(case_dir / "inputs", target)
    fix_axis = "_free" not in case_dir.name
    cal = calibrate_tfs_projection(masks, search_res=96, fix_axis=fix_axis)
    h256, _ = build_tfs_visual_hull(masks, target, calibration=cal)
    h256 = h256.astype(bool)

    core_up = upsample_nearest(core, target)
    clipped = core_up & h256
    eroded = binary_erosion(h256, structure=S6, iterations=int(shell_t))
    shell = h256 & ~eroded
    union = clipped | shell
    final = keep_largest(union)

    core_up_count = int(core_up.sum())
    clipped_count = int(clipped.sum())
    shell_count = int(shell.sum())
    attached_shell = shell & final
    final_count = int(final.sum())
    gt6 = masks + [np.fliplr(x) for x in masks]
    ev = evaluate_tfs_volume(final.astype(np.float32), gt6, cal, image_size=target)

    out_dir = case_dir / "decoupled_256_validation"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "core_up_clipped_256.npy", clipped.astype(np.uint8))
    np.save(out_dir / "shell_256.npy", shell.astype(np.uint8))
    np.save(out_dir / "final_256.npy", final.astype(np.uint8))

    return {
        "case": case_dir.name,
        "core_path": str(core_path),
        "input_masks": mask_names,
        "fix_axis": bool(fix_axis),
        "core_resolution": list(core.shape),
        "target_resolution": target,
        "shell_t": int(shell_t),
        "h256_voxels": int(h256.sum()),
        "core_up_voxels": core_up_count,
        "core_clipped_voxels": clipped_count,
        "core_retention": float(clipped_count / max(core_up_count, 1)),
        "shell_voxels": shell_count,
        "shell_attached_rate": float(int(attached_shell.sum()) / max(shell_count, 1)),
        "final_voxels": final_count,
        "final_components_6": n6(final),
        "final_mean_iou_256": float(ev["mean_iou"]),
        "material_saved_vs_h256": float(1.0 - final_count / max(int(h256.sum()), 1)),
        "artifacts": {
            "core_up_clipped": str(out_dir / "core_up_clipped_256.npy"),
            "shell": str(out_dir / "shell_256.npy"),
            "final": str(out_dir / "final_256.npy"),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "PRINT_DELIVERY_256sdf"))
    ap.add_argument("--out", default=str(ROOT / "results/decoupled_48core_256shell_validation.json"))
    ap.add_argument("--shell_t", type=int, default=2)
    ap.add_argument("--target", type=int, default=256)
    args = ap.parse_args()

    rows = []
    for case_dir in sorted(Path(args.root).iterdir()):
        if not case_dir.is_dir() or not (case_dir / "inputs").exists():
            continue
        try:
            rows.append(validate_case(case_dir, shell_t=args.shell_t, target=args.target))
        except Exception as exc:
            rows.append({"case": case_dir.name, "status": "error", "error": repr(exc)})

    summary = {
        "schema": "decoupled_48core_256shell_validation_v1",
        "rule": "V = keep_largest((C_low nearest-upsampled to 256 intersect H256) union (H256 minus erode(H256,t)))",
        "n_cases": len(rows),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
