#!/usr/bin/env python3
"""Rebuild current Fig.1 n=2 models with a material-saving pipeline.

Inputs are the four current Fig.1 overview cases:
  A+spade, A+house, A+teddy, I+sword.

For each case:
  two-view visual hull -> projection-preserving sparse carve -> scale-aware
  thickening candidates -> manufacturing gates -> minimum-volume feasible STL.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, label
from skimage import measure

import sys

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path.insert(0, str(ROOT / "generation"))
from voxel_ops import sdf_watertight_mesh
PAPER_SCRIPT = ROOT / "paper_aaai_submission/make_fig1_aaai_corner_blender_shadowart_original.py"
OUT = ROOT / "results/fig1_overview_n2_full_pipeline_10cm_256"
RES = 256
PRINT_MM = 100.0
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0
IOU_MIN = 0.95
SAVE_MIN = 0.40
SAVE_MAX = 0.85
EROSION_MIN = 0.15
S6 = generate_binary_structure(3, 1)

spec = importlib.util.spec_from_file_location("fig1_overview_src", PAPER_SCRIPT)
src = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(src)
src.SHADOWART_SRC = ROOT / "generation"


def resize_mask(mask: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255, "L").resize((RES, RES), Image.Resampling.NEAREST)) > 127


def visual_hull(back_xz: np.ndarray, floor_xy: np.ndarray) -> np.ndarray:
    return back_xz[:, None, :] & floor_xy[None, :, :]


def project_back(vol: np.ndarray) -> np.ndarray:
    return vol.any(axis=1)


def project_floor(vol: np.ndarray) -> np.ndarray:
    return vol.any(axis=0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    return float((a & b).sum() / max(int((a | b).sum()), 1))


def keep_largest(vol: np.ndarray) -> np.ndarray:
    lab, n = label(vol.astype(bool), structure=S6)
    if n <= 1:
        return vol.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def component_count(vol: np.ndarray) -> int:
    _, n = label(vol.astype(bool), structure=S6)
    return int(n)


def erosion_survival(vol: np.ndarray) -> float:
    return float(binary_erosion(vol.astype(bool), S6).sum() / max(int(vol.sum()), 1))


def projection_preserving_seed(back_xz: np.ndarray, floor_xy: np.ndarray) -> np.ndarray:
    """Create a sparse subset of the two-view hull that covers both projections."""
    seed = np.zeros((RES, RES, RES), dtype=bool)  # z, y, x
    for x in range(RES):
        z_ids = np.flatnonzero(back_xz[:, x])
        y_ids = np.flatnonzero(floor_xy[:, x])
        if z_ids.size == 0 or y_ids.size == 0:
            continue
        n = max(z_ids.size, y_ids.size)
        z_pick = z_ids[np.floor(np.linspace(0, z_ids.size - 1, n)).astype(int)]
        y_pick = y_ids[np.floor(np.linspace(0, y_ids.size - 1, n)).astype(int)]
        seed[z_pick, y_pick, x] = True
        # Add a mild center rail inside the feasible hull to improve 6-connectivity.
        zc = int(z_ids[len(z_ids) // 2])
        yc = int(y_ids[len(y_ids) // 2])
        seed[zc, y_ids, x] = True
        seed[z_ids, yc, x] = True
    return seed


def make_candidates(back_xz: np.ndarray, floor_xy: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    hull = visual_hull(back_xz, floor_xy)
    seed = projection_preserving_seed(back_xz, floor_xy) & hull
    floor_vox = int(np.ceil(WALL_SAFETY * NOZZLE_MM * RES / PRINT_MM))
    d_floor = max(int(np.ceil((floor_vox - 1) / 2.0)), 1)
    candidates = {
        "visual_hull": hull,
        "seed": seed,
    }
    for i in range(9):
        candidates[f"carve_pf{i+1}"] = binary_dilation(seed, S6, iterations=d_floor + i) & hull
    # Repair-oriented candidates: add progressively larger interior layers of
    # the visual hull. This keeps both projections unchanged, but can collapse
    # multi-shell SDF exports into a single printable mesh while retaining
    # material saving.
    for i in range(1, 10):
        base = binary_dilation(seed, S6, iterations=d_floor + i) & hull
        for e in range(2, 18, 2):
            inner = binary_erosion(hull, S6, iterations=e)
            if not inner.any():
                continue
            candidates[f"pf{i}_inner{e}"] = (base | inner) & hull
    for i in range(2, 8):
        base = binary_dilation(seed, S6, iterations=d_floor + i) & hull
        closed = binary_fill_holes(binary_closing(base, S6, iterations=2)) & hull
        candidates[f"pf{i}_closed2"] = closed
    return {k: keep_largest(v) for k, v in candidates.items() if v.any()}, hull


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(path)


def export_stl(vol: np.ndarray, path: Path) -> trimesh.Trimesh:
    mesh = sdf_watertight_mesh(vol.astype(bool), smooth_sigma=0.6)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    trimesh.repair.fill_holes(mesh)
    if mesh.volume < 0:
        mesh.invert()
    trimesh.repair.fix_normals(mesh)
    mesh.apply_scale(PRINT_MM / max(float(mesh.extents.max()), 1e-9))
    mesh.export(path)
    return mesh


def mesh_metrics(vol: np.ndarray) -> tuple[bool, int]:
    try:
        pad = np.pad(vol.astype(np.float32), 1, mode="constant")
        verts, faces, _, _ = measure.marching_cubes(pad, level=0.5, step_size=2)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        return bool(mesh.is_watertight and mesh.is_winding_consistent), int(len(mesh.split(only_watertight=False)))
    except Exception:
        return False, 0


def evaluate_candidate(name: str, vol: np.ndarray, hull: np.ndarray, back: np.ndarray, floor: np.ndarray, with_mesh: bool = False) -> dict:
    pb = project_back(vol)
    pf = project_floor(vol)
    mesh_ok, mesh_parts = mesh_metrics(vol) if with_mesh else (None, None)
    row = {
        "candidate": name,
        "iou": float((iou(pb, back) + iou(pf, floor)) / 2.0),
        "view_ious": [float(iou(pb, back)), float(iou(pf, floor))],
        "occupied_voxels": int(vol.sum()),
        "hull_voxels": int(hull.sum()),
        "material_saved_vs_hull": float(1.0 - int(vol.sum()) / max(int(hull.sum()), 1)),
        "component_count_6": component_count(vol),
        "watertight": mesh_ok,
        "mesh_parts": mesh_parts,
        "erosion_survival": erosion_survival(vol),
    }
    row["manufacturing_pass"] = bool(
        row["iou"] >= IOU_MIN
        and row["component_count_6"] == 1
        and row["watertight"] is True
        and row["mesh_parts"] == 1
        and row["erosion_survival"] >= EROSION_MIN
        and SAVE_MIN <= row["material_saved_vs_hull"] <= SAVE_MAX
    )
    return row


def run_one(example: str, ch: str, partner: str) -> dict:
    d = OUT / example
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "projections").mkdir(parents=True, exist_ok=True)
    (d / "candidates").mkdir(parents=True, exist_ok=True)
    back = resize_mask(src.letter_mask(ch, RES))
    floor = resize_mask(src.partner_mask(partner, RES))
    save_mask(back, d / f"inputs/input0_back_{ch}.png")
    save_mask(floor, d / f"inputs/input1_floor_{partner}.png")
    candidates, hull = make_candidates(back, floor)
    quick_rows = []
    volumes = {}
    for cname, vol in candidates.items():
        row = evaluate_candidate(cname, vol, hull, back, floor, with_mesh=False)
        quick_rows.append(row)
        volumes[cname] = vol
        np.save(d / "candidates" / f"{cname}.npy", vol.astype(np.uint8))
    shortlist = sorted(
        quick_rows,
        key=lambda r: (
            r["iou"] >= IOU_MIN,
            r["component_count_6"] == 1,
            r["erosion_survival"] >= EROSION_MIN,
            SAVE_MIN <= r["material_saved_vs_hull"] <= SAVE_MAX,
            r["material_saved_vs_hull"],
            r["iou"],
        ),
        reverse=True,
    )[:24]
    rows = []
    checked = {r["candidate"] for r in shortlist}
    for row in quick_rows:
        if row["candidate"] in checked:
            rows.append(evaluate_candidate(row["candidate"], volumes[row["candidate"]], hull, back, floor, with_mesh=True))
        else:
            rows.append(row)
    feasible = sorted(
        [r for r in rows if r["manufacturing_pass"]],
        key=lambda r: (r["material_saved_vs_hull"], r["iou"]),
        reverse=True,
    )
    fallback = sorted(
        [r for r in rows if not r["manufacturing_pass"]],
        key=lambda r: (r["mesh_parts"] == 1, r["iou"], 0.40 <= r["material_saved_vs_hull"] <= 0.85, r["material_saved_vs_hull"]),
        reverse=True,
    )
    selected = None
    mesh = None
    export_parts = 0
    final_pass = False
    for candidate in feasible + fallback[:1]:
        vol = volumes[candidate["candidate"]]
        npy = d / f"{example}_full_pipeline_voxels256.npy"
        stl = d / f"{example}_full_pipeline_10cm_256.stl"
        np.save(npy, vol.astype(np.uint8))
        mesh = export_stl(vol, stl)
        export_parts = int(len(mesh.split(only_watertight=False)))
        final_pass = bool(candidate["manufacturing_pass"] and mesh.is_watertight and export_parts == 1)
        selected = candidate
        if final_pass:
            break
    assert selected is not None and mesh is not None
    vol = volumes[selected["candidate"]]
    save_mask(project_back(vol), d / f"projections/projection0_back_{ch}.png")
    save_mask(project_floor(vol), d / f"projections/projection1_floor_{partner}.png")
    npy = d / f"{example}_full_pipeline_voxels256.npy"
    stl = d / f"{example}_full_pipeline_10cm_256.stl"
    rec = {
        "example": example,
        "n_inputs": 2,
        "targets": [ch, partner],
        "views": ["back_wall_xz", "floor_xy"],
        "accepted": final_pass,
        "selected": {
            **selected,
            "stl": str(stl),
            "voxels": str(npy),
            "mesh_faces": int(len(mesh.faces)),
            "mesh_extents_mm": [round(float(x), 3) for x in mesh.extents],
            "export_watertight": bool(mesh.is_watertight),
            "export_mesh_parts": export_parts,
            "final_pass": final_pass,
        },
        "all_candidates": sorted(rows, key=lambda r: (-r["material_saved_vs_hull"], -r["iou"])),
    }
    return rec


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
    for case in cases:
        rec = run_one(*case)
        records.append(rec)
        s = rec["selected"]
        print(
            f"{rec['example']}: accepted={rec['accepted']} {s['candidate']} "
            f"IoU={s['iou']:.4f} saving={s['material_saved_vs_hull']:.4f} "
            f"single={s['component_count_6']==1} watertight={s['watertight']}",
            flush=True,
        )
    manifest = {
        "schema": "fig1_overview_n2_full_pipeline_10cm_256_v1",
        "source_definition": str(PAPER_SCRIPT),
        "note": "Current Fig.1 n=2 models rebuilt with projection-preserving material-saving candidates and manufacturing gates.",
        "resolution": RES,
        "print_mm": PRINT_MM,
        "gate": {
            "iou_min": IOU_MIN,
            "saving_range": [SAVE_MIN, SAVE_MAX],
            "component_count_6": 1,
            "watertight": True,
            "mesh_parts": 1,
            "erosion_survival_min": EROSION_MIN,
        },
        "n_examples": len(records),
        "n_accepted": sum(r["accepted"] for r in records),
        "examples": records,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
