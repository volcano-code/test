#!/usr/bin/env python3
"""Run final-export connectivity/printability checks for v13 soft branch."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

PROJECT = Path("/data/xvxinyao/xvxinyao/total_project/mesh_combined_fusion_generalizable_ready_20260528")
PYTHON = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")
OUT_ROOT = PROJECT / "runs_codex_v13_export_printability_20260705"

BASE = [
    str(PYTHON), "train.py", "cuda:0", "{out}", "1600", "0.03", "0",
    "-sdlist", "Spider-Man.png", "armadillo_0.png", "batman.png",
    "-mr", "0", "-tsf", "0", "-tsf4", "0",
    "-imsz", "128", "-res", "128", "-vh", "1",
    "--skip_final_export", "0",
    "--export_res", "256",
    "--view_weight_power", "3.0", "--view_weight_floor", "0.5",
    "--weak_polish_iters", "300", "--weak_view_threshold", "0.9",
    "--weak_focus_weight", "4.0", "--protect_good_weight", "4.0",
    "--weak_polish_lr_scale", "0.015",
]

RUNS = {
    "full_export": [],
    "no_iso_export": ["--loss_iso_scale", "0"],
    "no_view_reweight_export": ["--view_weight_power", "0.0"],
    "no_hard_connected_export": ["--hard_connected_export", "0"],
}


def load_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics_history.json"
    if not path.exists():
        return {"metrics_exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    final = data.get("final_metrics", {})
    renderer = final.get("renderer_metrics", {})
    export = final.get("export_metrics", {})
    obj_path = export.get("obj_path") or export.get("raw_export_obj")
    voxel_path = export.get("voxel_path") or export.get("raw_export_voxels")
    return {
        "metrics_exists": True,
        "renderer_mean_iou": renderer.get("mean_iou"),
        "renderer_min_iou": min(renderer.get("view_ious", []) or [None]),
        "export_strategy": export.get("export_strategy") or final.get("selected_strategy"),
        "export_mean_iou": export.get("mean_iou"),
        "export_min_iou": export.get("min_iou"),
        "component_count": export.get("component_count"),
        "largest_component_ratio": export.get("largest_component_ratio"),
        "small_component_voxels": export.get("small_component_voxels"),
        "occupied_voxels": export.get("occupied_voxels"),
        "bridge_voxels_added": export.get("bridge_voxels_added"),
        "repair_voxels_added": export.get("repair_voxels_added"),
        "export_res": export.get("export_res"),
        "obj_path": obj_path,
        "obj_exists": bool(obj_path and (PROJECT / obj_path).exists()),
        "voxel_path": voxel_path,
        "voxel_exists": bool(voxel_path and (PROJECT / voxel_path).exists()),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, flags in RUNS.items():
        run_dir = OUT_ROOT / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [part.format(out=str(run_dir)) for part in BASE] + flags
        (run_dir / "command.json").write_text(json.dumps({"cmd": cmd, "cwd": str(PROJECT)}, indent=2), encoding="utf-8")
        start = time.time()
        with (run_dir / "run.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(PROJECT), stdout=log, stderr=subprocess.STDOUT)
        row = {"name": name, "returncode": proc.returncode, "elapsed_s": time.time() - start, "run_dir": str(run_dir), **load_metrics(run_dir)}
        rows.append(row)
        (OUT_ROOT / "summary_partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if proc.returncode != 0:
            break
    out = {
        "schema": "v13_export_printability_check_v1",
        "created": "2026-07-05",
        "note": "Final export enabled. Printability proxy is single connected component plus exported OBJ/voxel artifacts; watertightness is not asserted unless explicitly reported by the exporter.",
        "rows": rows,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
