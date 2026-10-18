#!/usr/bin/env python3
"""Create print-only detachable carriers for FDM-aware two-wall candidates.

The carrier is not part of the display model: a thin bed disk is connected by
small vertical breakaway pins at the model's lowest surface.  It increases bed
adhesion while keeping the fixed-display STL unchanged.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = ROOT / "results/anime_two_wall_fdm_qualityfirst_10cm_256_20260710"
OUT = ROOT / "results/anime_two_wall_fdm_qualityfirst_with_temporary_carrier_10cm_256_20260710"
BASE = ROOT / "experiments/build_anime_two_wall_fixed_no_floor_10cm_256_20260709.py"
R, PRINT_MM = 256, 100.0
BASE_RADIUS_MM, BASE_THICKNESS_MM = 18.0, 1.6
PIN_RADIUS_VOX, PIN_COUNT = 1, 4
S2 = np.ones((3, 3), dtype=bool)

spec = importlib.util.spec_from_file_location("n2base", str(BASE))
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)
n2 = base.n2


def support_metrics(v: np.ndarray) -> dict:
    coords = np.where(v)
    z0, z1 = int(coords[2].min()), int(coords[2].max())
    unsupported = 0
    for z in range(z0 + 1, z1 + 1):
        unsupported += int((v[:, :, z] & ~binary_dilation(v[:, :, z - 1], structure=S2)).sum())
    voxel_mm = PRINT_MM / R
    return {
        "unsupported_voxels": unsupported,
        "unsupported_ratio": float(unsupported / max(int(v.sum()), 1)),
        "bed_area_mm2": float(v[:, :, z0].sum() * voxel_mm * voxel_mm),
    }


def anchor_points(bottom: np.ndarray, count: int) -> list[tuple[int, int]]:
    pts = np.argwhere(bottom)
    center = pts.mean(axis=0)
    chosen = [pts[np.argmin(np.sum((pts - center) ** 2, axis=1))]]
    while len(chosen) < min(count, len(pts)):
        distances = np.min([np.sum((pts - c) ** 2, axis=1) for c in chosen], axis=0)
        chosen.append(pts[np.argmax(distances)])
    return [tuple(map(int, p)) for p in chosen]


def add_carrier(print_volume: np.ndarray) -> tuple[np.ndarray, dict]:
    v = print_volume.astype(bool)
    z0 = int(np.where(v)[2].min())
    thickness = max(2, int(round(BASE_THICKNESS_MM * R / PRINT_MM)))
    shift = thickness + 2 - z0
    if shift < 0:
        shift = 0
    out = np.zeros((R, R, R + thickness + 4), dtype=bool)
    out[:, :, shift:shift + R] = v
    new_z0 = int(np.where(out)[2].min())
    bottom = out[:, :, new_z0]
    anchors = anchor_points(bottom, PIN_COUNT)
    cx, cy = np.mean(np.argwhere(bottom), axis=0)
    yy, xx = np.ogrid[:R, :R]
    radius = int(round(BASE_RADIUS_MM * R / PRINT_MM))
    disk = (xx - cy) ** 2 + (yy - cx) ** 2 <= radius * radius
    for z in range(thickness):
        out[:, :, z] |= disk
    for ax, ay in anchors:
        out[max(0, ax - PIN_RADIUS_VOX):min(R, ax + PIN_RADIUS_VOX + 1),
            max(0, ay - PIN_RADIUS_VOX):min(R, ay + PIN_RADIUS_VOX + 1),
            thickness:new_z0 + 1] = True
    return out[:, :, :R].astype(bool), {
        "carrier": "temporary disk with vertical breakaway pins",
        "disk_radius_mm": BASE_RADIUS_MM,
        "disk_thickness_mm": thickness * PRINT_MM / R,
        "pin_radius_mm": PIN_RADIUS_VOX * PRINT_MM / R,
        "pin_count": len(anchors),
        "anchors_xy_voxels": anchors,
    }


def main():
    manifest = json.loads((SRC / "MANIFEST.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for rec in manifest["cases"]:
        if not rec.get("accepted"):
            continue
        case = rec["case"]
        src_dir = SRC / case
        volume_path = next(src_dir.glob("print_*_voxels256.npy"))
        carrier, carrier_info = add_carrier(np.load(volume_path).astype(bool))
        dst = OUT / case
        dst.mkdir(parents=True, exist_ok=True)
        np.save(dst / "print_with_temporary_carrier_voxels256.npy", carrier.astype(np.uint8))
        stl = dst / f"{case}_print_with_temporary_carrier_10cm_256.stl"
        mesh = n2.export_stl(np.transpose(carrier, (2, 1, 0)), stl)
        qa = {
            "case": case,
            "display_stl": rec["display_stl"],
            "print_source_stl": rec["print_stl"],
            "print_carrier_stl": str(stl),
            "carrier_must_be_removed_after_print": True,
            "display_model_is_unchanged": True,
            "carrier_info": carrier_info,
            "source_fdm_proxy": rec["fdm_proxy"],
            "carrier_fdm_proxy": support_metrics(carrier),
            "mesh_parts": int(len(mesh.split(only_watertight=False))),
            "watertight": bool(mesh.is_watertight),
            "required_validation": "slice at 50 mm first; inspect support reachability and remove carrier before wall-corner display",
        }
        (dst / "TEMPORARY_CARRIER_QA.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n")
        records.append(qa)
        print(json.dumps(qa, ensure_ascii=False), flush=True)
    (OUT / "MANIFEST.json").write_text(json.dumps({"schema": "temporary_print_carrier_v1", "cases": records}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
