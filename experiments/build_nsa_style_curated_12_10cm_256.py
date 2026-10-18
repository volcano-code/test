#!/usr/bin/env python3
"""Regenerate 12 NSA/Sadekar-style physical-print examples with stronger silhouettes.

The set uses the current 256^3, 10 cm pipeline. Examples that were already
validated as fixed-axis remain fixed; symbolic/complex replacements use the free
branch when fixed-axis screening is not adequate. The set emphasizes the two
shadow-art references' visual vocabulary: complex objects, characters/symbols,
and recognizable animal/character-like silhouettes.
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
from PIL import Image

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
for p in [ROOT, ROOT / "generation", ROOT / "experiments"]:
    sys.path.insert(0, str(p))

BASE = ROOT / "experiments/build_nsa_style_12_10cm_256.py"
spec = importlib.util.spec_from_file_location("nsa_base", str(BASE))
nsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nsa)

from candidate_selection_batch import make_candidates, n6  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh  # noqa: E402

OUT = ROOT / "results/nsa_style_shadowart_refs_12_10cm_256"
R = 256
PRINT_MM = 100.0
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0


def sil_path(name: str) -> Path:
    for sub in ("bench3", "curated_symbolic_v1", "complex_v2_filled"):
        p = ROOT / "data/silhouettes" / sub / f"{name}.png"
        if p.exists():
            return p
    raise FileNotFoundError(name)


CASES = [
    {
        "case_id": "case1_corner_clear_plate",
        "nsa_panel": "top-left",
        "setup": "90-degree two-wall corner, black table, transparent acrylic plate; show left/right wall shadows, optional floor shadow.",
        "caution": "Use model-only STL for shadow validation; any opaque plate/base is display-only and must stay outside the measured shadow windows.",
        "examples": [
            {"example": "case1A_dragon_shadow_A", "names": ["complex_dragon", "han_ying_shadow", "aaai_A"], "branch": "free", "candidate": "auto_quality"},
            {"example": "case1B_phoenix_robot_light", "names": ["complex_phoenix", "complex_robot", "han_guang_light"], "branch": "free", "candidate": "auto_quality"},
        ],
    },
    {
        "case_id": "case2_corner_suspended_object",
        "nsa_panel": "top-middle",
        "setup": "90-degree two-wall corner; suspend/hold the model with transparent line or thin acrylic rod from above; keep fixture outside the wall-shadow regions.",
        "caution": "This is the cleanest physical setup when a model has poor bottom contact; fixture is not part of the algorithmic result.",
        "examples": [
            {"example": "case2A_castle_ship_A", "names": ["complex_castle", "complex_ship", "aaai_A"], "branch": "free", "candidate": "auto_quality"},
            {"example": "case2B_cathedral_bicycle_mind", "names": ["complex_cathedral", "complex_bicycle", "han_zhi_intelligence"], "branch": "free", "candidate": "auto_quality"},
        ],
    },
    {
        "case_id": "case3_single_wall_like",
        "nsa_panel": "top-right",
        "setup": "Approximation: use two vertical white screens in a shallow V or a 90-degree corner and photograph nearly front-on so the two wall shadows read like one wall display.",
        "caution": "The current pipeline is orthogonal three-plane TFS, not arbitrary same-wall multi-light projection; exact NSA same-wall reproduction needs a non-orthogonal projection operator.",
        "examples": [
            {"example": "case3A_owl_gear_dream", "names": ["complex_owl_totem", "aaai_gear", "han_meng_dream"], "branch": "free", "candidate": "auto_quality"},
            {"example": "case3B_violin_stag_A", "names": ["complex_violin", "complex_stag", "aaai_A"], "branch": "free", "candidate": "auto_quality"},
        ],
    },
    {
        "case_id": "case4_white_paper_sweep",
        "nsa_panel": "bottom-left",
        "setup": "White seamless paper sweep from wall to floor; use one horizontal wall light and one overhead/floor light, with the third wall optional.",
        "caution": "A curved sweep is for photography; metric validation still uses the three orthogonal digital projections.",
        "examples": [
            {"example": "case4A_scorpion_peacock_gear", "names": ["x_scorpion", "x_peacock", "aaai_gear"], "branch": "free", "candidate": "auto_quality"},
            {"example": "case4B_lobster_squid_dragon", "names": ["x_lobster", "sea_squid", "x_dragon"], "branch": "fixed", "candidate": "auto_quality"},
        ],
    },
    {
        "case_id": "case5_acrylic_center_display",
        "nsa_panel": "bottom-middle",
        "setup": "Transparent acrylic block/plate at the corner center; three projection planes can be shown, but take photos with the fixture nearly invisible.",
        "caution": "If using translucent resin, report it as a physical display choice; geometry metrics are computed on the opaque binary model.",
        "examples": [
            {"example": "case5A_beetle_eagle_shadow", "names": ["bugs_beetle", "birds_eagle", "han_ying_shadow"], "branch": "fixed", "candidate": "auto_quality"},
            {"example": "case5B_parrot_trex_light", "names": ["birds_parrot", "x_trex", "han_guang_light"], "branch": "fixed", "candidate": "auto_quality"},
        ],
    },
    {
        "case_id": "case6_plinth_like_display",
        "nsa_panel": "bottom-right",
        "setup": "Use an independent plinth or transparent pedestal only for presentation; place/crop it so plinth shadow is not evaluated as part of the three silhouettes.",
        "caution": "Do not merge the plinth into the shadow-art STL; an opaque connected base would break at least one target silhouette.",
        "examples": [
            {"example": "case6A_bat_trex_A", "names": ["x_bat", "x_trex", "aaai_A"], "branch": "fixed", "candidate": "auto_quality"},
            {"example": "case6B_ladybug_frog_peacock", "names": ["x_ladybug", "reptiles_frog", "x_peacock"], "branch": "fixed", "candidate": "auto_quality"},
        ],
    },
]


def run_one(case: dict, spec: dict) -> dict:
    t0 = time.time()
    names = spec["names"]
    paths = [sil_path(name) for name in names]
    sils, _ = load_data_from_list([str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils]
    candidates, aux = make_candidates(
        masks,
        R,
        spec["branch"] == "fixed",
        print_mm=PRINT_MM,
        nozzle_mm=NOZZLE_MM,
        wall_safety=WALL_SAFETY,
    )
    hull = aux["hull"].astype(bool)
    gt6 = masks + [np.fliplr(m) for m in masks]
    scored = []
    for cname, cvol in candidates.items():
        cvol = cvol.astype(bool)
        ev_c = evaluate_tfs_volume(cvol.astype(np.float32), gt6, aux["calibration"], image_size=512)
        saving_c = float(1.0 - int(cvol.sum()) / max(int(hull.sum()), 1))
        if n6(cvol) != 1 or not (0.40 <= saving_c <= 0.85):
            continue
        try:
            watertight_c = bool(sdf_watertight_mesh(cvol.astype(np.uint8), smooth_sigma=0.6).is_watertight)
        except Exception:
            watertight_c = False
        scored.append({
            "candidate": cname,
            "volume": cvol,
            "eval": ev_c,
            "saving": saving_c,
            "watertight": watertight_c,
            "mean_iou": float(ev_c["mean_iou"]),
            "min_iou": float(min(ev_c["view_ious"])),
        })
    feasible = [r for r in scored if r["mean_iou"] >= 0.975 and r["watertight"]]
    if not feasible:
        details = [
            {k: r[k] for k in ("candidate", "mean_iou", "min_iou", "saving", "watertight")}
            for r in scored
        ]
        raise RuntimeError(f"{spec['example']}: no quality candidate; scored={details}")
    chosen = sorted(feasible, key=lambda r: (-r["mean_iou"], -r["saving"]))[0]
    vol = chosen["volume"]
    ev = chosen["eval"]
    watertight = chosen["watertight"]

    d = OUT / case["case_id"] / spec["example"]
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "silhouettes").mkdir(parents=True, exist_ok=True)
    for i, (src, name) in enumerate(zip(paths, names)):
        img = np.array(Image.open(src).convert("L"))
        nsa.save_binary_image(img > 127, d / "inputs" / f"input{i}_{name}.png")
        nsa.save_binary_image(ev["pred_masks"][i], d / "silhouettes" / f"shadow{i}_{name}.png")

    npy = d / f"{spec['example']}_voxels256_{spec['branch']}_{chosen['candidate']}.npy"
    stl = d / f"{spec['example']}_10cm_256_{spec['branch']}_{chosen['candidate']}.stl"
    np.save(npy, vol.astype(np.uint8))
    nsa.ehqm.export_marching_cubes(vol, stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
    mesh = nsa.mesh_clean_scale(stl)
    return {
        "case_id": case["case_id"],
        "nsa_panel": case["nsa_panel"],
        "example": spec["example"],
        "silhouettes": names,
        "physical_order": {
            "floor_z_view0": names[0],
            "left_wall_x_view1": names[1],
            "right_wall_y_view2": names[2],
        },
        "resolution": R,
        "print_mm": PRINT_MM,
        "selected_branch": spec["branch"],
        "candidate": chosen["candidate"],
        "iou": round(float(ev["mean_iou"]), 4),
        "view_ious": [round(float(x), 4) for x in ev["view_ious"][:3]],
        "saving": round(float(chosen["saving"]), 4),
        "single": bool(n6(vol) == 1),
        "watertight": watertight,
        "wall_mm_median_proxy": round(nsa.erosion_survival(vol), 4),
        "support_proxy": nsa.support_proxy(vol),
        "mesh_extents_mm": [round(float(x), 1) for x in mesh.extents],
        "mesh_faces": int(len(mesh.faces)),
        "stl": str(stl),
        "voxels": str(npy),
        "sec": round(time.time() - t0, 1),
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cases_out = []
    for case in CASES:
        out_case = {k: case[k] for k in ("case_id", "nsa_panel", "setup", "caution")}
        out_case["models"] = []
        for spec in case["examples"]:
            rec = run_one(case, spec)
            rows.append(rec)
            out_case["models"].append(rec)
            print(
                f"{rec['example']}: branch={rec['selected_branch']} cand={rec['candidate']} "
                f"IoU={rec['iou']} saving={rec['saving']} single={rec['single']} watertight={rec['watertight']}",
                flush=True,
            )
        cases_out.append(out_case)
    nsa.OUT = OUT
    nsa.make_contact_sheet(rows)
    summary = {
        "schema": "nsa_style_shadowart_refs_12_10cm_256_v1",
        "selection_note": "Curated from fixed/free candidates with NSA/Sadekar-style complex object, symbol, character, and animal silhouettes; symbolic replacements use free branch when fixed screening is inadequate.",
        "resolution": R,
        "print_mm": PRINT_MM,
        "projection_model": "TFS: view0->z/floor, view1->x/left wall, view2->y/right wall",
        "n_cases": len(cases_out),
        "n_models": len(rows),
        "cases": cases_out,
        "flat_examples": rows,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "BATCH_MANIFEST.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
