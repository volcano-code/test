#!/usr/bin/env python3
"""Generate 12 NSA-style display models with the current 3-plane pipeline.

Each display case corresponds to one visual setup in the NSA example image.
For each case we generate two different 10 cm / 256^3 models using the existing
fixed-axis TFS pipeline:
  view0 -> floor z projection, view1 -> left-wall x projection, view2 -> right-wall y projection.

Exact same-wall multi-light NSA layouts are recorded as approximations when the
current orthogonal three-plane pipeline cannot reproduce them literally.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, generate_binary_structure

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
for p in [ROOT, ROOT / "generation", ROOT / "experiments"]:
    sys.path.insert(0, str(p))

from datasets import load_data_from_list
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh
from candidate_selection_batch import make_candidates, n6

_spec = importlib.util.spec_from_file_location(
    "export_high_quality_mesh", str(ROOT / "generation/export_high_quality_mesh.py")
)
ehqm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ehqm)

R = 256
PRINT_MM = 100.0
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0
CANDIDATE = "carve_pf5"
BENCH = ROOT / "data/silhouettes/bench3"
OUT = ROOT / "results/nsa_style_12_10cm_256"
S6 = generate_binary_structure(3, 1)

CASES = [
    {
        "case_id": "case1_corner_clear_plate",
        "nsa_panel": "top-left",
        "setup": "90-degree two-wall corner, black table, transparent acrylic plate; show left/right wall shadows, optional floor shadow.",
        "caution": "Use model-only STL for shadow validation; any opaque plate/base is display-only and must stay outside the measured shadow windows.",
        "examples": [
            ("case1A_owl_bat_turtle", ["birds_owl", "x_bat", "reptiles_turtle"]),
            ("case1B_eagle_frog_trex", ["birds_eagle", "reptiles_frog", "x_trex"]),
        ],
    },
    {
        "case_id": "case2_corner_suspended_object",
        "nsa_panel": "top-middle",
        "setup": "90-degree two-wall corner; suspend/hold the model with transparent line or thin acrylic rod from above; keep fixture outside the wall-shadow regions.",
        "caution": "This is the cleanest physical setup when a model has poor bottom contact; fixture is not part of the algorithmic result.",
        "examples": [
            ("case2A_parrot_dragon_snail", ["birds_parrot", "x_dragon", "sea_snail"]),
            ("case2B_bat_trex_snail", ["x_bat", "x_trex", "sea_snail"]),
        ],
    },
    {
        "case_id": "case3_single_wall_like",
        "nsa_panel": "top-right",
        "setup": "Approximation: use two vertical white screens in a shallow V or a 90-degree corner and photograph nearly front-on so the two wall shadows read like one wall display.",
        "caution": "The current pipeline is orthogonal three-plane TFS, not arbitrary same-wall multi-light projection; exact NSA same-wall reproduction needs a non-orthogonal projection operator.",
        "examples": [
            ("case3A_spider_scorpion_beetle", ["x_spider", "x_scorpion", "bugs_beetle"]),
            ("case3B_lobster_squid_ladybug", ["x_lobster", "sea_squid", "x_ladybug"]),
        ],
    },
    {
        "case_id": "case4_white_paper_sweep",
        "nsa_panel": "bottom-left",
        "setup": "White seamless paper sweep from wall to floor; use one horizontal wall light and one overhead/floor light, with the third wall optional.",
        "caution": "A curved sweep is for photography; metric validation still uses the three orthogonal digital projections.",
        "examples": [
            ("case4A_parrot_frog_peacock", ["birds_parrot", "reptiles_frog", "x_peacock"]),
            ("case4B_shrimp_turtle_owl", ["sea_shrimp", "reptiles_turtle", "birds_owl"]),
        ],
    },
    {
        "case_id": "case5_acrylic_center_display",
        "nsa_panel": "bottom-middle",
        "setup": "Transparent acrylic block/plate at the corner center; three projection planes can be shown, but take photos with the fixture nearly invisible.",
        "caution": "If using translucent resin, report it as a physical display choice; geometry metrics are computed on the opaque binary model.",
        "examples": [
            ("case5A_ant_peacock_frog", ["bugs_ant", "x_peacock", "reptiles_frog"]),
            ("case5B_beetle_owl_frog", ["bugs_beetle", "birds_owl", "reptiles_frog"]),
        ],
    },
    {
        "case_id": "case6_plinth_like_display",
        "nsa_panel": "bottom-right",
        "setup": "Use an independent plinth or transparent pedestal only for presentation; place/crop it so plinth shadow is not evaluated as part of the three silhouettes.",
        "caution": "Do not merge the plinth into the shadow-art STL; an opaque connected base would break at least one target silhouette.",
        "examples": [
            ("case6A_lizard_snail_bat", ["reptiles_lizard", "sea_snail", "x_bat"]),
            ("case6B_ladybug_frog_peacock", ["x_ladybug", "reptiles_frog", "x_peacock"]),
        ],
    },
]


def save_binary_image(mask: np.ndarray, path: Path) -> None:
    img = ((np.asarray(mask) > 0.5).astype(np.uint8) * 255)
    Image.fromarray(img).resize((512, 512), Image.Resampling.NEAREST).save(path)


def mesh_clean_scale(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    parts = mesh.split(only_watertight=False)
    if parts:
        mesh = max(parts, key=lambda x: len(x.faces))
    trimesh.repair.fill_holes(mesh)
    if mesh.volume < 0:
        mesh.invert()
    trimesh.repair.fix_normals(mesh)
    mesh.apply_scale(PRINT_MM / max(float(np.asarray(mesh.extents).max()), 1e-9))
    mesh.export(str(path))
    return mesh


def erosion_survival(volume: np.ndarray) -> float:
    return float(binary_erosion(volume.astype(bool), S6).sum() / max(int(volume.sum()), 1))


def support_proxy(volume: np.ndarray) -> dict:
    v = volume.astype(bool)
    if not v.any():
        return {"bed_contact_area_mm2": 0.0}
    z_min = int(np.where(v)[2].min())
    voxel_mm = PRINT_MM / R
    return {
        "bed_contact_voxels": int(v[:, :, z_min].sum()),
        "bed_contact_area_mm2": float(int(v[:, :, z_min].sum()) * voxel_mm * voxel_mm),
    }


def make_contact_sheet(rows: list[dict]) -> None:
    thumbs = []
    for rec in rows:
        d = OUT / rec["case_id"] / rec["example"] / "inputs"
        panels = [Image.open(d / f"input{i}_{name}.png").convert("RGB").resize((160, 160)) for i, name in enumerate(rec["silhouettes"])]
        tile = Image.new("RGB", (480, 200), "white")
        for i, img in enumerate(panels):
            tile.paste(img, (i * 160, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((4, 164), rec["example"], fill=(0, 0, 0))
        draw.text((4, 182), f"IoU={rec['iou']} save={rec['saving']}", fill=(0, 0, 0))
        thumbs.append(tile)
    sheet = Image.new("RGB", (960, 6 * 200), "white")
    for idx, tile in enumerate(thumbs):
        x = (idx % 2) * 480
        y = (idx // 2) * 200
        sheet.paste(tile, (x, y))
    sheet.save(OUT / "INPUT_CONTACT_SHEET.png")


def run_one(case: dict, example: str, names: list[str]) -> dict:
    t0 = time.time()
    paths = [BENCH / f"{name}.png" for name in names]
    sils, _ = load_data_from_list([str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils]
    candidates, aux = make_candidates(
        masks,
        R,
        True,
        print_mm=PRINT_MM,
        nozzle_mm=NOZZLE_MM,
        wall_safety=WALL_SAFETY,
    )
    if CANDIDATE not in candidates:
        raise RuntimeError(f"{example}: missing {CANDIDATE}; got {sorted(candidates)}")
    hull = aux["hull"].astype(bool)
    vol = candidates[CANDIDATE].astype(bool)
    gt6 = masks + [np.fliplr(m) for m in masks]
    ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=512)
    watertight = bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight)

    d = OUT / case["case_id"] / example
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "silhouettes").mkdir(parents=True, exist_ok=True)
    for i, (src, name) in enumerate(zip(paths, names)):
        img = np.array(Image.open(src).convert("L"))
        save_binary_image(img > 127, d / "inputs" / f"input{i}_{name}.png")
        save_binary_image(ev["pred_masks"][i], d / "silhouettes" / f"shadow{i}_{name}.png")

    npy = d / f"{example}_voxels256_pf5.npy"
    stl = d / f"{example}_10cm_256_pf5.stl"
    np.save(npy, vol.astype(np.uint8))
    ehqm.export_marching_cubes(
        vol.astype(bool),
        stl,
        smooth_sigma=0.6,
        taubin_iterations=3,
        upsample_factor=1,
    )
    mesh = mesh_clean_scale(stl)
    rec = {
        "case_id": case["case_id"],
        "nsa_panel": case["nsa_panel"],
        "example": example,
        "silhouettes": names,
        "physical_order": {
            "floor_z_view0": names[0],
            "left_wall_x_view1": names[1],
            "right_wall_y_view2": names[2],
        },
        "resolution": R,
        "print_mm": PRINT_MM,
        "candidate": CANDIDATE,
        "iou": round(float(ev["mean_iou"]), 4),
        "view_ious": [round(float(x), 4) for x in ev["view_ious"][:3]],
        "saving": round(float(1.0 - int(vol.sum()) / max(int(hull.sum()), 1)), 4),
        "single": bool(n6(vol) == 1),
        "watertight": watertight,
        "wall_mm_median_proxy": round(erosion_survival(vol), 4),
        "support_proxy": support_proxy(vol),
        "mesh_extents_mm": [round(float(x), 1) for x in mesh.extents],
        "mesh_faces": int(len(mesh.faces)),
        "stl": str(stl),
        "voxels": str(npy),
        "sec": round(time.time() - t0, 1),
    }
    return rec


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    case_rows = []
    for case in CASES:
        per_case = {k: case[k] for k in ("case_id", "nsa_panel", "setup", "caution")}
        per_case["models"] = []
        for example, names in case["examples"]:
            rec = run_one(case, example, names)
            rows.append(rec)
            per_case["models"].append(rec)
            print(
                f"{example}: IoU={rec['iou']} saving={rec['saving']} "
                f"single={rec['single']} watertight={rec['watertight']} "
                f"bed={rec['support_proxy']['bed_contact_area_mm2']:.1f}",
                flush=True,
            )
        case_rows.append(per_case)
    make_contact_sheet(rows)
    summary = {
        "schema": "nsa_style_12_10cm_256_v1",
        "resolution": R,
        "print_mm": PRINT_MM,
        "candidate": CANDIDATE,
        "projection_model": "fixed-axis TFS: view0->z/floor, view1->x/left wall, view2->y/right wall",
        "important_limit": "Exact same-wall multi-light NSA layouts require a non-orthogonal projection extension; this set uses the current orthogonal three-plane pipeline.",
        "n_cases": len(case_rows),
        "n_models": len(rows),
        "cases": case_rows,
        "flat_examples": rows,
    }
    with open(OUT / "MANIFEST.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(OUT / "BATCH_MANIFEST.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
