#!/usr/bin/env python3
"""Build the Figure-1 AAAI n=2 models as 10 cm printable assets.

This intentionally uses two target silhouettes per model, matching
paper_aaai_submission/make_fig1_aaai_four_n2_models.py.  It does not add a
third support silhouette.
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, generate_binary_structure, label
from skimage import measure

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
PAPER_SCRIPT = ROOT / "paper_aaai_submission/make_fig1_aaai_four_n2_models.py"
OUT = ROOT / "results/fig1_aaai_n2_10cm_256"
RES = 256
PRINT_MM = 100.0
S6 = generate_binary_structure(3, 1)


spec = importlib.util.spec_from_file_location("fig1_n2", PAPER_SCRIPT)
fig1_n2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fig1_n2)


def resize_mask(mask: np.ndarray) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    return np.array(img.resize((RES, RES), Image.Resampling.NEAREST)) > 127


def visual_hull(letter: np.ndarray, partner: np.ndarray) -> np.ndarray:
    return letter[:, None, :] & partner[:, :, None]


def project_letter(vol: np.ndarray) -> np.ndarray:
    return vol.any(axis=1)


def project_partner(vol: np.ndarray) -> np.ndarray:
    return vol.any(axis=2)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    return float((a & b).sum() / max(int((a | b).sum()), 1))


def save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), "L").save(path)


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


def erosion_survival(vol: np.ndarray) -> float:
    return float(binary_erosion(vol.astype(bool), S6).sum() / max(int(vol.sum()), 1))


def component_count(vol: np.ndarray) -> int:
    _, n = label(vol.astype(bool), structure=S6)
    return int(n)


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def preview_tile(name: str, ch: str, partner: str, d: Path) -> Image.Image:
    tile = Image.new("RGB", (360, 360), "white")
    draw = ImageDraw.Draw(tile)
    draw.text((14, 10), name, fill=(20, 20, 20), font=font(20))
    labels = [(f"target: {ch}", "inputs/input0_letter.png"), (f"target: {partner}", "inputs/input1_partner.png"),
              (f"proj: {ch}", "projections/projection0_letter.png"), (f"proj: {partner}", "projections/projection1_partner.png")]
    for i, (caption, rel) in enumerate(labels):
        x = 24 + (i % 2) * 168
        y = 54 + (i // 2) * 146
        img = Image.open(d / rel).convert("L").resize((112, 112), Image.Resampling.NEAREST)
        tile.paste(Image.merge("RGB", (img, img, img)), (x, y))
        draw.rectangle([x, y, x + 111, y + 111], outline=(185, 185, 185), width=1)
        draw.text((x, y + 116), caption, fill=(70, 70, 70), font=font(13))
    return tile


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    cases = [
        ("fig1_A_star", "A", "star"),
        ("fig1_A_heart", "A", "heart"),
        ("fig1_A_gear", "A", "gear"),
        ("fig1_I_tree", "I", "tree"),
    ]
    records = []
    tiles = []

    for name, ch, partner_name in cases:
        d = OUT / name
        (d / "inputs").mkdir(parents=True, exist_ok=True)
        (d / "projections").mkdir(parents=True, exist_ok=True)
        letter = resize_mask(fig1_n2.letter_mask(ch))
        partner = resize_mask(fig1_n2.partner_mask(partner_name))
        vol = visual_hull(letter, partner)
        proj0 = project_letter(vol)
        proj1 = project_partner(vol)
        npy = d / f"{name}_n2_voxels256.npy"
        stl = d / f"{name}_n2_10cm_256.stl"
        np.save(npy, vol.astype(np.uint8))
        save_mask(letter, d / "inputs/input0_letter.png")
        save_mask(partner, d / "inputs/input1_partner.png")
        save_mask(proj0, d / "projections/projection0_letter.png")
        save_mask(proj1, d / "projections/projection1_partner.png")
        mesh = export_stl(vol, stl)
        rec = {
            "example": name,
            "n_inputs": 2,
            "targets": [ch, partner_name],
            "views": ["front_xz", "side_yz"],
            "iou_mean": round((iou(proj0, letter) + iou(proj1, partner)) / 2, 6),
            "view_ious": [round(iou(proj0, letter), 6), round(iou(proj1, partner), 6)],
            "voxel_count": int(vol.sum()),
            "components_6conn": component_count(vol),
            "erosion_survival": round(erosion_survival(vol), 6),
            "watertight": bool(mesh.is_watertight),
            "mesh_faces": int(len(mesh.faces)),
            "mesh_extents_mm": [round(float(x), 3) for x in mesh.extents],
            "stl": str(stl),
            "voxels": str(npy),
            "inputs": [str(d / "inputs/input0_letter.png"), str(d / "inputs/input1_partner.png")],
            "projections": [str(d / "projections/projection0_letter.png"), str(d / "projections/projection1_partner.png")],
        }
        records.append(rec)
        tiles.append(preview_tile(name, ch, partner_name, d))
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    sheet = Image.new("RGB", (720, 720), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * 360, (i // 2) * 360))
    sheet.save(OUT / "fig1_aaai_n2_10cm_256_contact_sheet.png")

    manifest = {
        "schema": "fig1_aaai_n2_10cm_256_v1",
        "source_definition": str(PAPER_SCRIPT),
        "note": "Each Fig.1 AAAI model uses exactly two target silhouettes; no support or third view is added.",
        "print_scale_mm": PRINT_MM,
        "resolution": RES,
        "n_examples": len(records),
        "examples": records,
        "contact_sheet": str(OUT / "fig1_aaai_n2_10cm_256_contact_sheet.png"),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
