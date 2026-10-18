#!/usr/bin/env python3
"""Build 12 print-scale models from public-safe anime archetype silhouettes."""
from __future__ import annotations

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

import export_high_quality_mesh as ehqm  # noqa: E402
from candidate_selection_batch import make_candidates, n6  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh  # noqa: E402

OUT = ROOT / "results/anime_archetype_12_10cm_256"
SIL = ROOT / "data/silhouettes/anime_archetype_v1"
R = 256
PRINT_MM = 100.0
NOZZLE_MM = 0.4
WALL_SAFETY = 2.0


CASES = [
    ("case1A_shonen_magic_mecha", "top-left", ["anime_spiky_swordsman", "anime_twin_tail_magical", "anime_mecha_robot"], False),
    ("case1B_ninja_idol_beast", "top-left", ["anime_ninja_runner", "anime_winged_idol", "anime_magic_beast"], False),
    ("case2A_samurai_thief_umbrella", "top-middle", ["anime_samurai_mask", "anime_masked_thief", "anime_umbrella_girl"], False),
    ("case2B_sailor_chibi_caped", "top-middle", ["anime_sailor_warrior", "anime_chibi_mascot", "anime_caped_hero"], False),
    ("case3A_spiky_ninja_samurai", "top-right", ["anime_spiky_swordsman", "anime_ninja_runner", "anime_samurai_mask"], False),
    ("case3B_mecha_beast_thief", "top-right", ["anime_mecha_robot", "anime_magic_beast", "anime_masked_thief"], False),
    ("case4A_magical_sailor_idol", "bottom-left", ["anime_twin_tail_magical", "anime_sailor_warrior", "anime_winged_idol"], False),
    ("case4B_umbrella_chibi_caped", "bottom-left", ["anime_umbrella_girl", "anime_chibi_mascot", "anime_caped_hero"], False),
    ("case5A_fixed_mecha_samurai_spiky", "bottom-middle", ["anime_mecha_robot", "anime_samurai_mask", "anime_spiky_swordsman"], True),
    ("case5B_fixed_beast_ninja_idol", "bottom-middle", ["anime_magic_beast", "anime_ninja_runner", "anime_winged_idol"], True),
    ("case6A_fixed_thief_sailor_umbrella", "bottom-right", ["anime_masked_thief", "anime_sailor_warrior", "anime_umbrella_girl"], True),
    ("case6B_fixed_chibi_magical_robot", "bottom-right", ["anime_chibi_mascot", "anime_twin_tail_magical", "anime_mecha_robot"], True),
]


def save_binary(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), "L").save(path)


def mesh_clean_scale(path: Path):
    import trimesh

    mesh = trimesh.load(path, force="mesh")
    scale = PRINT_MM / max(float(mesh.extents.max()), 1e-8)
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.bounds[0])
    mesh.export(path)
    return trimesh.load(path, force="mesh")


def run_one(example: str, panel: str, names: list[str], fixed: bool) -> dict:
    t0 = time.time()
    paths = [SIL / f"{name}.png" for name in names]
    sils, _ = load_data_from_list([str(p) for p in paths], True, (R, R), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils]
    candidates, aux = make_candidates(masks, R, fixed, print_mm=PRINT_MM, nozzle_mm=NOZZLE_MM, wall_safety=WALL_SAFETY)
    hull = aux["hull"].astype(bool)
    gt6 = masks + [np.fliplr(m) for m in masks]

    scored = []
    for cname, vol0 in candidates.items():
        vol = vol0.astype(bool)
        ev = evaluate_tfs_volume(vol.astype(np.float32), gt6, aux["calibration"], image_size=512)
        saving = float(1.0 - int(vol.sum()) / max(int(hull.sum()), 1))
        if n6(vol) != 1 or not (0.40 <= saving <= 0.85):
            continue
        try:
            watertight = bool(sdf_watertight_mesh(vol.astype(np.uint8), smooth_sigma=0.6).is_watertight)
        except Exception:
            watertight = False
        scored.append({
            "candidate": cname,
            "volume": vol,
            "eval": ev,
            "saving": saving,
            "single": n6(vol) == 1,
            "watertight": watertight,
            "mean_iou": float(ev["mean_iou"]),
            "min_iou": float(min(ev["view_ious"])),
        })

    feasible = [r for r in scored if r["mean_iou"] >= 0.975 and r["watertight"]]
    if not feasible:
        details = [{k: r[k] for k in ("candidate", "mean_iou", "min_iou", "saving", "watertight")} for r in scored]
        raise RuntimeError(f"{example}: no feasible candidate; scored={details}")
    chosen = sorted(feasible, key=lambda r: (-r["mean_iou"], -r["saving"]))[0]
    vol = chosen["volume"]
    ev = chosen["eval"]

    d = OUT / panel / example
    (d / "inputs").mkdir(parents=True, exist_ok=True)
    (d / "silhouettes").mkdir(parents=True, exist_ok=True)
    for i, (src, name) in enumerate(zip(paths, names)):
        mask = np.asarray(Image.open(src).convert("L")) > 127
        save_binary(mask, d / "inputs" / f"input{i}_{name}.png")
        save_binary(ev["pred_masks"][i], d / "silhouettes" / f"shadow{i}_{name}.png")

    tag = "fixed" if fixed else "free"
    npy = d / f"{example}_voxels256_{tag}_{chosen['candidate']}.npy"
    stl = d / f"{example}_10cm_256_{tag}_{chosen['candidate']}.stl"
    np.save(npy, vol.astype(np.uint8))
    ehqm.export_marching_cubes(vol, stl, smooth_sigma=0.6, taubin_iterations=3, upsample_factor=1)
    mesh = mesh_clean_scale(stl)
    return {
        "example": example,
        "nsa_panel": panel,
        "silhouettes": names,
        "source": "anime_archetype_v1_original_non_ip",
        "resolution": R,
        "print_mm": PRINT_MM,
        "selected_branch": tag,
        "candidate": chosen["candidate"],
        "iou": round(float(chosen["mean_iou"]), 4),
        "min_iou": round(float(chosen["min_iou"]), 4),
        "view_ious": [round(float(x), 4) for x in ev["view_ious"][:3]],
        "saving": round(float(chosen["saving"]), 4),
        "single": bool(n6(vol) == 1),
        "watertight": bool(chosen["watertight"]),
        "mesh_extents_mm": [round(float(x), 1) for x in mesh.extents],
        "mesh_faces": int(len(mesh.faces)),
        "stl": str(stl),
        "voxels": str(npy),
        "sec": round(time.time() - t0, 1),
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows = []
    for example, panel, names, fixed in CASES:
        row = run_one(example, panel, names, fixed)
        rows.append(row)
        print(f"{example}: IoU={row['iou']} saving={row['saving']} {row['selected_branch']}/{row['candidate']} watertight={row['watertight']}", flush=True)
    summary = {
        "schema": "anime_archetype_12_10cm_256_v1",
        "note": "Original public-safe anime archetype silhouettes; no copyrighted character silhouettes are copied.",
        "quality_gate": {"iou_min": 0.975, "saving_range": [0.40, 0.85], "single": True, "watertight": True},
        "n_models": len(rows),
        "models": rows,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "BATCH_MANIFEST.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
