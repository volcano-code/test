#!/usr/bin/env python3
"""Build a 33-record v13 soft ablation table with final printability checks."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage


PROJECT = Path("/data/xvxinyao/xvxinyao/total_project/mesh_combined_fusion_generalizable_ready_20260528")
PYTHON = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")
OUT_ROOT = PROJECT / "runs_codex_v13_soft_ablation_printability_33rows_20260705"
PREV_ROOT = PROJECT / "runs_codex_v13_soft_ablation_printability_20260705"
TIMEOUT_S = 900
EXPORT_STABLE_S = 20

CASES = {
    "case00_spider_armadillo_batman": ["Spider-Man.png", "armadillo_0.png", "batman.png"],
    "case01_bunny_duck_hand": ["bunny_0.png", "duck.png", "hand.png"],
    "case02_heart_bunny135_armadillo": ["heart.png", "bunny_135.png", "armadillo_0.png"],
}

ABLATIONS = {
    "full": [],
    "no_asym": ["--loss_asym_scale", "0"],
    "no_fn": ["--loss_fn_scale", "0"],
    "no_iso": ["--loss_iso_scale", "0"],
    "no_view_reweight": ["--view_weight_power", "0.0"],
    "no_weak_polish": ["--weak_polish_iters", "0"],
    "no_reg": ["--loss_reg_scale", "0"],
    "no_silh": ["--loss_silh_scale", "0"],
    "no_l1": ["--loss_l1_scale", "0"],
    "no_ssim": ["--loss_ssim_scale", "0"],
    "no_iou_dice": ["--loss_iou_dice_scale", "0"],
}

BASE_PREFIX = [
    str(PYTHON),
    "train.py",
    "cuda:0",
    "{out}",
    "1600",
    "0.03",
    "0",
]

BASE_SUFFIX = [
    "-mr",
    "0",
    "-tsf",
    "0",
    "-tsf4",
    "0",
    "-imsz",
    "128",
    "-res",
    "128",
    "-vh",
    "1",
    "--skip_final_export",
    "0",
    "--export_res",
    "256",
    "--view_weight_power",
    "3.0",
    "--view_weight_floor",
    "0.5",
    "--weak_polish_iters",
    "300",
    "--weak_view_threshold",
    "0.9",
    "--weak_focus_weight",
    "4.0",
    "--protect_good_weight",
    "4.0",
    "--weak_polish_lr_scale",
    "0.015",
]


def load_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def projection_metrics(run_dir: Path) -> dict:
    data = load_json(run_dir / "metrics_history.json")
    final = data.get("final_metrics", {}) if isinstance(data, dict) else {}
    renderer = final.get("renderer_metrics", {})
    view_ious = renderer.get("view_ious") or []
    return {
        "metrics_exists": bool(data),
        "renderer_mean_iou": renderer.get("mean_iou") or final.get("mean_iou"),
        "renderer_min_iou": min(view_ious) if view_ious else None,
        "renderer_mean_dice": renderer.get("mean_dice") or final.get("mean_dice"),
    }


def check_export(run_dir: Path) -> dict:
    voxels = sorted(run_dir.glob("*final_voxels.npy"))
    objs = sorted(run_dir.glob("*output.obj"))
    out = {
        "voxel_exists": bool(voxels),
        "obj_exists": bool(objs),
        "voxel_path": str(voxels[0]) if voxels else None,
        "obj_path": str(objs[0]) if objs else None,
    }
    if voxels:
        arr = np.squeeze(np.load(voxels[0]))
        occ = arr > 0.5
        struct = np.zeros((3, 3, 3), dtype=bool)
        struct[1, 1, 1] = True
        struct[0, 1, 1] = struct[2, 1, 1] = True
        struct[1, 0, 1] = struct[1, 2, 1] = True
        struct[1, 1, 0] = struct[1, 1, 2] = True
        labels, count = ndimage.label(occ, structure=struct)
        sizes = np.bincount(labels.ravel()) if count else np.array([occ.size])
        comps = sorted([int(x) for x in sizes[1:]], reverse=True)
        occupied = int(occ.sum())
        out.update(
            {
                "occupied_voxels": occupied,
                "occupancy_ratio": float(occ.mean()),
                "component_count_6": int(count),
                "largest_component_ratio": float(comps[0] / occupied) if comps and occupied else 0.0,
                "small_component_voxels": int(sum(comps[1:])) if len(comps) > 1 else 0,
            }
        )
    if objs:
        mesh = trimesh.load(objs[0], force="mesh", process=False)
        out.update(
            {
                "mesh_is_watertight": bool(mesh.is_watertight),
                "mesh_is_winding_consistent": bool(mesh.is_winding_consistent),
                "mesh_body_count_split": int(len(mesh.split(only_watertight=False))),
                "mesh_vertices": int(len(mesh.vertices)),
                "mesh_faces": int(len(mesh.faces)),
            }
        )
    out["printability_proxy_pass"] = bool(
        out.get("component_count_6") == 1
        and out.get("small_component_voxels") == 0
        and out.get("obj_exists")
        and out.get("mesh_is_watertight")
        and out.get("mesh_body_count_split") == 1
        and (out.get("occupied_voxels") or 0) > 1000
    )
    return out


def import_previous_case00() -> list[dict]:
    prev = load_json(PREV_ROOT / "summary.json")
    if not isinstance(prev, dict):
        return []
    rows = []
    for row in prev.get("rows", []):
        copied = dict(row)
        copied["case"] = "case00_spider_armadillo_batman"
        copied["silhouettes"] = CASES["case00_spider_armadillo_batman"]
        copied["source"] = str(PREV_ROOT / "summary.json")
        copied["record_id"] = f"{copied['case']}::{copied['name']}"
        if (copied.get("occupied_voxels") or 0) <= 1000:
            copied["printability_proxy_pass"] = False
            copied["degenerate_export"] = True
        rows.append(copied)
    return rows


def run_one(case: str, silhouettes: list[str], ablation: str, flags: list[str]) -> dict:
    run_dir = OUT_ROOT / case / ablation
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [part.format(out=str(run_dir)) for part in BASE_PREFIX]
    cmd += ["-sdlist", *silhouettes]
    cmd += BASE_SUFFIX + flags
    (run_dir / "command.json").write_text(
        json.dumps({"case": case, "ablation": ablation, "silhouettes": silhouettes, "cmd": cmd}, indent=2),
        encoding="utf-8",
    )
    start = time.time()
    timed_out = False
    harvested_after_export = False
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT), stdout=log, stderr=subprocess.STDOUT)
        stable_since = None
        last_sizes = None
        while True:
            returncode = proc.poll()
            if returncode is not None:
                break
            voxels = sorted(run_dir.glob("*final_voxels.npy"))
            objs = sorted(run_dir.glob("*output.obj"))
            if voxels and objs:
                sizes = (voxels[0].stat().st_size, objs[0].stat().st_size)
                now = time.time()
                if sizes == last_sizes:
                    stable_since = stable_since or now
                else:
                    stable_since = now
                    last_sizes = sizes
                if now - stable_since >= EXPORT_STABLE_S:
                    harvested_after_export = True
                    proc.kill()
                    returncode = proc.wait()
                    break
            if time.time() - start > TIMEOUT_S:
                timed_out = True
                proc.kill()
                returncode = proc.wait()
                break
            time.sleep(5)
    row = {
        "record_id": f"{case}::{ablation}",
        "case": case,
        "name": ablation,
        "silhouettes": silhouettes,
        "flags": flags,
        "run_dir": str(run_dir),
        "returncode": returncode,
        "timed_out": timed_out,
        "harvested_after_export": harvested_after_export,
        "elapsed_s": time.time() - start,
    }
    row.update(projection_metrics(run_dir))
    row.update(check_export(run_dir))
    (run_dir / "codex_ablation_printability_check.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = import_previous_case00()
    (OUT_ROOT / "summary_partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for case, silhouettes in list(CASES.items())[1:]:
        for ablation, flags in ABLATIONS.items():
            rows.append(run_one(case, silhouettes, ablation, flags))
            (OUT_ROOT / "summary_partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    out = {
        "schema": "v13_soft_ablation_printability_33_records_v1",
        "created": "2026-07-05",
        "record_count": len(rows),
        "case_count": len(CASES),
        "ablation_count": len(ABLATIONS),
        "note": "33 records = 3 silhouette triplets x 11 ablation settings. Printability is a proxy: non-degenerate final voxels, one 6-connected component, no small components, OBJ exists, watertight mesh, and one mesh body.",
        "cases": CASES,
        "rows": rows,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
