#!/usr/bin/env python3
"""Build two-view fixed anime shadow-art print candidates.

This experiment is intentionally separate from the three-view wall-corner
pipeline.  It tests the display setting discussed on 2026-07-09: two fixed
projected silhouettes plus an optional independent display base.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, generate_binary_structure, label
from skimage import measure

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/anime_n2_fixed_display_10cm_256_20260709"
SOURCES = [
    Path("/home/lin/下载/original_anime_silhouettes_100 (2)"),
    Path("/home/lin/下载/original_anime_silhouettes_30 (4)"),
    Path("/home/lin/下载/all_silhouettes_single_connected_unique/silhouettes"),
]
RES_SCREEN = 128
RES_EXPORT = 256
PRINT_MM = 100.0
S6 = generate_binary_structure(3, 1)


def load_mask(path: Path, res: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((res, res), Image.Resampling.NEAREST)
    arr = np.asarray(img)
    # Support both black-on-white and white-on-black sources.
    if arr.mean() > 127:
        arr = 255 - arr
    return arr > 127


def save_mask(mask: np.ndarray, path: Path, scale: int = 512) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(mask.astype(np.uint8) * 255, "L").resize((scale, scale), Image.Resampling.NEAREST)
    img.save(path)


def visual_hull_n2(front_xz: np.ndarray, side_yz: np.ndarray) -> np.ndarray:
    # Coordinates are [x,y,z].  front_xz constrains y-projection; side_yz
    # constrains x-projection.
    return front_xz[:, None, :] & side_yz[None, :, :]


def project_front_xz(vol: np.ndarray) -> np.ndarray:
    return vol.any(axis=1)


def project_side_yz(vol: np.ndarray) -> np.ndarray:
    return vol.any(axis=0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    return float((a & b).sum() / max(int((a | b).sum()), 1))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    return float(2 * (a & b).sum() / max(int(a.sum() + b.sum()), 1))


def face_occupancy(volume: np.ndarray) -> dict:
    v = np.asarray(volume, dtype=bool)
    coords = np.argwhere(v)
    if coords.size == 0:
        return {"bbox_fill": 0.0, "max_face_occupancy": 1.0, "mean_face_occupancy": 1.0}
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    sub = v[mins[0] : maxs[0] + 1, mins[1] : maxs[1] + 1, mins[2] : maxs[2] + 1]
    faces = [
        float(sub[0].mean()),
        float(sub[-1].mean()),
        float(sub[:, 0, :].mean()),
        float(sub[:, -1, :].mean()),
        float(sub[:, :, 0].mean()),
        float(sub[:, :, -1].mean()),
    ]
    return {"bbox_fill": float(sub.mean()), "max_face_occupancy": float(max(faces)), "mean_face_occupancy": float(np.mean(faces))}


def component_count(vol: np.ndarray) -> int:
    _, n = label(vol.astype(bool), structure=S6)
    return int(n)


def keep_largest(vol: np.ndarray) -> np.ndarray:
    lab, n = label(vol.astype(bool), structure=S6)
    if n <= 1:
        return vol.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def erosion_survival(vol: np.ndarray) -> float:
    return float(binary_erosion(vol.astype(bool), S6).sum() / max(int(vol.sum()), 1))


def smooth_candidate(hull: np.ndarray, mode: str) -> np.ndarray:
    """Create conservative shape variants without using slab templates."""
    if mode == "hull":
        return keep_largest(hull.astype(bool))
    if mode == "center_band":
        coords = np.indices(hull.shape)
        y = coords[1].astype(np.float32)
        center = (hull.shape[1] - 1) * 0.5
        band = np.abs(y - center) <= hull.shape[1] * 0.34
        vol = hull & band
    elif mode == "front_side_union_core":
        # Keep a central core plus boundary shells implied by both silhouettes.
        er = binary_erosion(hull, S6, iterations=2)
        coords = np.indices(hull.shape)
        y = coords[1].astype(np.float32)
        center = (hull.shape[1] - 1) * 0.5
        band = np.abs(y - center) <= hull.shape[1] * 0.28
        vol = er | (hull & band)
    else:
        raise ValueError(mode)
    # Repair projection false negatives by adding the original hull back on
    # missing projection rays.
    vol = keep_largest(vol)
    f0 = project_front_xz(vol)
    s0 = project_side_yz(vol)
    miss_f = np.argwhere(project_front_xz(hull) & ~f0)
    for x, z in miss_f:
        ys = np.where(hull[x, :, z])[0]
        if ys.size:
            vol[x, int(ys[len(ys) // 2]), z] = True
    miss_s = np.argwhere(project_side_yz(hull) & ~s0)
    for y, z in miss_s:
        xs = np.where(hull[:, y, z])[0]
        if xs.size:
            vol[int(xs[len(xs) // 2]), y, z] = True
    return keep_largest(vol)


def metrics(vol: np.ndarray, front: np.ndarray, side: np.ndarray, hull_vox: int) -> dict:
    pf = project_front_xz(vol)
    ps = project_side_yz(vol)
    m = {
        "view_ious": [iou(pf, front), iou(ps, side)],
        "view_dices": [dice(pf, front), dice(ps, side)],
        "mean_iou": 0.0,
        "mean_dice": 0.0,
        "min_iou": 0.0,
        "saving": float(1.0 - int(vol.sum()) / max(hull_vox, 1)),
        "components": component_count(vol),
        "erosion_survival": erosion_survival(vol),
        **face_occupancy(vol),
    }
    m["mean_iou"] = float(np.mean(m["view_ious"]))
    m["mean_dice"] = float(np.mean(m["view_dices"]))
    m["min_iou"] = float(np.min(m["view_ious"]))
    return m


def export_stl(vol: np.ndarray, path: Path) -> trimesh.Trimesh:
    padded = np.pad(vol.astype(np.float32), 1)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts = (verts - 1) / (vol.shape[0] - 1)
    verts -= 0.5
    mesh = trimesh.Trimesh(vertices=verts[:, [2, 1, 0]], faces=faces, process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    trimesh.repair.fill_holes(mesh)
    if mesh.volume < 0:
        mesh.invert()
    trimesh.repair.fix_normals(mesh)
    parts = mesh.split(only_watertight=False)
    if parts:
        mesh = max(parts, key=lambda p: len(p.faces))
    mesh.apply_scale(PRINT_MM / max(float(mesh.extents.max()), 1e-9))
    mesh.export(path)
    return mesh


def export_base(path: Path) -> trimesh.Trimesh:
    base = trimesh.creation.box(extents=(110.0, 76.0, 5.0))
    base.apply_translation((0, 0, -2.5))
    slot = trimesh.creation.box(extents=(72.0, 8.0, 3.0))
    slot.apply_translation((0, 0, 2.5))
    # Store a simple display plinth; the slot is represented by a raised locator
    # instead of boolean subtraction to avoid requiring external CSG backends.
    marker = slot.copy()
    marker.visual.face_colors = [30, 30, 30, 255]
    scene = trimesh.util.concatenate([base, marker])
    scene.export(path)
    return scene


def contact_sheet(title: str, front: np.ndarray, side: np.ndarray, pf: np.ndarray, ps: np.ndarray, out: Path) -> None:
    cell = 156
    canvas = Image.new("RGB", (2 * cell + 48, 2 * cell + 74), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 10), title, fill=(0, 0, 0))
    items = [("target front", front), ("target side", side), ("projection front", pf), ("projection side", ps)]
    for idx, (label_text, mask) in enumerate(items):
        x = 16 + (idx % 2) * (cell + 16)
        y = 44 + (idx // 2) * (cell + 20)
        img = Image.fromarray(mask.astype(np.uint8) * 255, "L").resize((cell, cell), Image.Resampling.NEAREST)
        canvas.paste(Image.merge("RGB", (img, img, img)), (x, y))
        draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(150, 150, 150))
        draw.text((x, y + cell + 2), label_text, fill=(45, 45, 45))
    canvas.save(out)


def source_files() -> list[Path]:
    files: list[Path] = []
    for src in SOURCES:
        if src.exists():
            files.extend(sorted(src.glob("*.png")))
    # Prefer named, distinctive categories while keeping paths unique.
    seen = set()
    unique = []
    for p in files:
        key = p.name
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    files = source_files()
    pairs = []
    keywords = ("hero", "ninja", "magic", "swordsman", "android", "archer", "cat", "wizard", "武", "巫", "忍", "滑板", "战士", "魔法")
    preferred = [p for p in files if any(k.lower() in p.name.lower() for k in keywords)]
    pool = (preferred if len(preferred) >= 12 else files)[:36]
    for i, a in enumerate(pool):
        for b in pool[i + 1 :]:
            if a.stem.split("_")[-1] == b.stem.split("_")[-1]:
                continue
            front = load_mask(a, RES_SCREEN)
            side = load_mask(b, RES_SCREEN)
            hull = visual_hull_n2(front, side)
            if hull.sum() < 100:
                continue
            hvox = int(hull.sum())
            best = None
            for mode in ("hull", "center_band"):
                vol = smooth_candidate(hull, mode)
                m = metrics(vol, front, side, hvox)
                score = (
                    m["mean_iou"],
                    m["components"] == 1,
                    m["erosion_survival"],
                    -m["max_face_occupancy"],
                    -abs(m["saving"] - 0.25),
                )
                if best is None or score > best[0]:
                    best = (score, mode, m)
            assert best is not None
            mode, m = best[1], best[2]
            accepted_screen = (
                m["mean_iou"] >= 0.975
                and m["min_iou"] >= 0.95
                and m["components"] == 1
                and m["erosion_survival"] >= 0.08
                and m["max_face_occupancy"] <= 0.18
            )
            if accepted_screen:
                pairs.append({"front": a, "side": b, "mode": mode, **m})
    pairs.sort(key=lambda r: (r["mean_iou"], -r["max_face_occupancy"], r["erosion_survival"]), reverse=True)
    records = []
    exported_idx = 0
    for row in pairs:
        if exported_idx >= 5:
            break
        idx = exported_idx + 1
        name = f"anime_n2_{idx:02d}_{row['front'].stem[:18]}__{row['side'].stem[:18]}"
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        d = OUT / safe_name
        (d / "inputs").mkdir(parents=True, exist_ok=True)
        (d / "projections").mkdir(parents=True, exist_ok=True)
        front = load_mask(row["front"], RES_EXPORT)
        side = load_mask(row["side"], RES_EXPORT)
        hull = visual_hull_n2(front, side)
        vol = smooth_candidate(hull, row["mode"])
        m = metrics(vol, front, side, int(hull.sum()))
        final_ok = (
            m["mean_iou"] >= 0.95
            and m["min_iou"] >= 0.90
            and m["components"] == 1
            and m["erosion_survival"] >= 0.08
            and m["max_face_occupancy"] <= 0.18
        )
        if not final_ok:
            continue
        stl = d / f"{safe_name}_10cm_256.stl"
        npy = d / f"{safe_name}_voxels256.npy"
        mesh = export_stl(vol, stl)
        base = d / "display_base_110x76mm.stl"
        export_base(base)
        np.save(npy, vol.astype(np.uint8))
        pf = project_front_xz(vol)
        ps = project_side_yz(vol)
        save_mask(front, d / "inputs/front_target.png")
        save_mask(side, d / "inputs/side_target.png")
        save_mask(pf, d / "projections/front_projection.png")
        save_mask(ps, d / "projections/side_projection.png")
        contact_sheet(safe_name, front, side, pf, ps, d / "input_vs_projection.png")
        rec = {
            "case": safe_name,
            "setting": "two-view fixed display",
            "input_paths": [str(row["front"]), str(row["side"])],
            "candidate_mode": row["mode"],
            "mean_iou": round(m["mean_iou"], 6),
            "min_iou": round(m["min_iou"], 6),
            "mean_dice": round(m["mean_dice"], 6),
            "view_ious": [round(float(x), 6) for x in m["view_ious"]],
            "view_dices": [round(float(x), 6) for x in m["view_dices"]],
            "saving_vs_n2_hull": round(m["saving"], 6),
            "components_6conn": int(m["components"]),
            "mesh_parts": int(len(mesh.split(only_watertight=False))),
            "mesh_watertight": bool(mesh.is_watertight),
            "erosion_survival": round(m["erosion_survival"], 6),
            "max_face_occupancy": round(m["max_face_occupancy"], 6),
            "bbox_fill": round(m["bbox_fill"], 6),
            "stl": str(stl),
            "voxels": str(npy),
            "display_base_stl": str(base),
            "preview": str(d / "input_vs_projection.png"),
        }
        records.append(rec)
        exported_idx += 1
        (d / "CASE_MANIFEST.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    manifest = {
        "schema": "anime_n2_fixed_display_10cm_256_20260709",
        "sources": [str(s) for s in SOURCES],
        "screen_resolution": RES_SCREEN,
        "export_resolution": RES_EXPORT,
        "print_scale_mm": PRINT_MM,
        "screened_pairs": len(pairs),
        "n_exported": len(records),
        "selection_note": "Two-view fixed projection; slab-like candidates are screened by max_face_occupancy at the 128 screening stage.",
        "cases": records,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DONE {OUT}")


if __name__ == "__main__":
    main()
