#!/usr/bin/env python3
"""Build the current paper Fig.1 overview models as n=2 10 cm assets.

The current Fig.1 overview is generated from
make_fig1_aaai_corner_blender_shadowart_original.py and uses two target
silhouettes per model: one back-wall letter view and one floor object view.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure, label
from skimage import measure

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
PAPER_SCRIPT = ROOT / "paper_aaai_submission/make_fig1_aaai_corner_blender_shadowart_original.py"
OUT = ROOT / "results/fig1_overview_n2_10cm_256"
RES = 256
PRINT_MM = 100.0
S6 = generate_binary_structure(3, 1)

spec = importlib.util.spec_from_file_location("fig1_overview_src", PAPER_SCRIPT)
src = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(src)
src.SHADOWART_SRC = ROOT / "generation"


def resize_mask(mask: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255, "L").resize((RES, RES), Image.Resampling.NEAREST)) > 127


def visual_hull_back_floor(back_xz: np.ndarray, floor_xy: np.ndarray) -> np.ndarray:
    return back_xz[:, None, :] & floor_xy[None, :, :]


def project_back(vol: np.ndarray) -> np.ndarray:
    return vol.any(axis=1)


def project_floor(vol: np.ndarray) -> np.ndarray:
    return vol.any(axis=0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    return float((a & b).sum() / max(int((a | b).sum()), 1))


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(path)


def export_stl(vol: np.ndarray, path: Path) -> trimesh.Trimesh:
    padded = np.pad(vol.astype(np.float32), 1)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts = (verts - 1) / (RES - 1)
    verts -= 0.5
    mesh = trimesh.Trimesh(vertices=verts[:, [2, 1, 0]], faces=faces, process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    trimesh.repair.fill_holes(mesh)
    if mesh.volume < 0:
        mesh.invert()
    trimesh.repair.fix_normals(mesh)
    mesh.apply_scale(PRINT_MM / max(float(mesh.extents.max()), 1e-9))
    mesh.export(path)
    return mesh


def component_count(vol: np.ndarray) -> int:
    _, n = label(vol.astype(bool), structure=S6)
    return int(n)


def erosion_survival(vol: np.ndarray) -> float:
    return float(binary_erosion(vol.astype(bool), S6).sum() / max(int(vol.sum()), 1))


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    cases = [
        ("model_A1_spade", "A", "spade"),
        ("model_A2_house", "A", "house"),
        ("model_A3_teddy", "A", "teddy"),
        ("model_I_sword", "I", "sword"),
    ]
    records = []
    for name, ch, partner in cases:
        d = OUT / name
        (d / "inputs").mkdir(parents=True, exist_ok=True)
        (d / "projections").mkdir(parents=True, exist_ok=True)
        back = resize_mask(src.letter_mask(ch, RES))
        floor = resize_mask(src.partner_mask(partner, RES))
        vol = visual_hull_back_floor(back, floor)
        pred_back = project_back(vol)
        pred_floor = project_floor(vol)
        npy = d / f"{name}_n2_voxels256.npy"
        stl = d / f"{name}_n2_10cm_256.stl"
        np.save(npy, vol.astype(np.uint8))
        save_mask(back, d / f"inputs/input0_back_{ch}.png")
        save_mask(floor, d / f"inputs/input1_floor_{partner}.png")
        save_mask(pred_back, d / f"projections/projection0_back_{ch}.png")
        save_mask(pred_floor, d / f"projections/projection1_floor_{partner}.png")
        mesh = export_stl(vol, stl)
        rec = {
            "example": name,
            "n_inputs": 2,
            "targets": [ch, partner],
            "views": ["back_wall_xz", "floor_xy"],
            "iou_mean": round((iou(pred_back, back) + iou(pred_floor, floor)) / 2, 6),
            "view_ious": [round(iou(pred_back, back), 6), round(iou(pred_floor, floor), 6)],
            "voxel_count": int(vol.sum()),
            "components_6conn": component_count(vol),
            "erosion_survival": round(erosion_survival(vol), 6),
            "watertight": bool(mesh.is_watertight),
            "mesh_faces": int(len(mesh.faces)),
            "mesh_extents_mm": [round(float(x), 3) for x in mesh.extents],
            "stl": str(stl),
            "voxels": str(npy),
            "inputs": [str(d / f"inputs/input0_back_{ch}.png"), str(d / f"inputs/input1_floor_{partner}.png")],
            "projections": [str(d / f"projections/projection0_back_{ch}.png"), str(d / f"projections/projection1_floor_{partner}.png")],
        }
        records.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    (OUT / "MANIFEST.json").write_text(json.dumps({
        "schema": "fig1_overview_n2_10cm_256_v1",
        "source_definition": str(PAPER_SCRIPT),
        "note": "Current Fig.1 overview models; each model uses exactly two silhouettes, no support/third view.",
        "resolution": RES,
        "print_scale_mm": PRINT_MM,
        "n_examples": len(records),
        "examples": records,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
