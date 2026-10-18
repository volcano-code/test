#!/usr/bin/env python3
"""Add projection-preserving, layerwise FDM support paths to two-wall models."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = ROOT / "results/anime_two_wall_fixed_no_floor_physical_axes_qualityfirst_10cm_256_20260710"
OUT = ROOT / "results/anime_two_wall_fdm_self_supporting_10cm_256_20260710"
BASE = ROOT / "experiments/build_anime_n2_fixed_display_10cm_256_20260709.py"

spec = importlib.util.spec_from_file_location("n2", str(BASE))
n2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(n2)

SUPPORT_FOOTPRINT = np.ones((3, 3), dtype=bool)


def load_mask(path: str) -> np.ndarray:
    image = Image.open(path).convert("L").resize((256, 256), Image.Resampling.NEAREST)
    array = np.asarray(image)
    if array.mean() > 127:
        array = 255 - array
    return array > 127


def physical_hull(left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
    left_yz = np.flipud(left_image).T
    right_xz = np.flipud(right_image).T
    xyz = left_yz[None, :, :] & right_xz[:, None, :]
    return np.transpose(xyz, (2, 1, 0)).copy()  # [Z,Y,X], export order


def support_closure(source: np.ndarray, hull: np.ndarray) -> tuple[np.ndarray, dict]:
    """Route every layer to the layer below within a one-voxel 45-degree cone."""
    result = source.copy()
    unresolved_total = 0
    added_total = 0
    for z in range(result.shape[0] - 1, 0, -1):
        supported = binary_dilation(result[z - 1], structure=SUPPORT_FOOTPRINT)
        remaining = result[z] & ~supported
        if not remaining.any():
            continue
        # Direct vertical parents are preferred; diagonal parents are added
        # only for pixels whose direct parent lies outside the visual hull.
        direct = remaining & hull[z - 1]
        result[z - 1] |= direct
        remaining &= ~binary_dilation(result[z - 1], structure=SUPPORT_FOOTPRINT)
        if remaining.any():
            diagonal = binary_dilation(remaining, structure=SUPPORT_FOOTPRINT) & hull[z - 1]
            result[z - 1] |= diagonal
            remaining &= ~binary_dilation(result[z - 1], structure=SUPPORT_FOOTPRINT)
        unresolved_total += int(remaining.sum())
    added_total = int(result.sum() - source.sum())
    violations = 0
    for z in range(1, result.shape[0]):
        violations += int((result[z] & ~binary_dilation(result[z - 1], structure=SUPPORT_FOOTPRINT)).sum())
    return result, {
        "added_voxels": added_total,
        "unresolved_during_routing": unresolved_total,
        "final_unsupported_voxels_45deg": violations,
        "max_required_bridge_span_voxels": 0 if violations == 0 else None,
    }


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(int((a | b).sum()), 1))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * (a & b).sum() / max(int(a.sum() + b.sum()), 1))


def main() -> None:
    manifest = json.loads((SRC / "MANIFEST.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for source_record in manifest["cases"]:
        name = source_record["case"]
        left = load_mask(source_record["input_paths"][0])
        right = load_mask(source_record["input_paths"][1])
        hull = physical_hull(left, right)
        source = np.load(source_record["voxels"]).astype(bool)
        result, support_metrics = support_closure(source, hull)
        destination = OUT / name
        destination.mkdir(parents=True, exist_ok=True)
        stl = destination / f"{name}_fdm_self_supporting_10cm_256.stl"
        voxels = destination / f"{name}_fdm_self_supporting_voxels256.npy"
        mesh = n2.export_stl(result, stl)
        np.save(voxels, result.astype(np.uint8))
        left_projection = np.flipud(result.any(axis=2))
        right_projection = np.flipud(result.any(axis=1))
        hull_voxels = int(hull.sum())
        record = {
            "case": name,
            "accepted": bool(
                support_metrics["final_unsupported_voxels_45deg"] == 0
                and n2.component_count(result) == 1
                and len(mesh.split(only_watertight=False)) == 1
                and mesh.is_watertight
            ),
            "source": source_record["stl"],
            "stl": str(stl),
            "voxels": str(voxels),
            "mean_iou": float(np.mean([iou(left_projection, left), iou(right_projection, right)])),
            "min_iou": float(min(iou(left_projection, left), iou(right_projection, right))),
            "mean_dice": float(np.mean([dice(left_projection, left), dice(right_projection, right)])),
            "saving_vs_two_wall_hull": float(1.0 - result.sum() / max(hull_voxels, 1)),
            "components_6conn": n2.component_count(result),
            "mesh_parts": len(mesh.split(only_watertight=False)),
            "mesh_watertight": bool(mesh.is_watertight),
            "self_support_angle_deg": 45,
            "external_support_reachability": "not required only when final_unsupported_voxels_45deg is zero",
            **support_metrics,
        }
        if not record["accepted"]:
            record["rejection_reason"] = (
                "strict projection-preserving support closure left unsupported voxels "
                "or disconnected permanent support material"
            )
        (destination / "FDM_QA.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    payload = {"schema": "two_wall_fdm_self_supporting_v1", "cases": records}
    (OUT / "MANIFEST.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
