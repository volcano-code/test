#!/usr/bin/env python3
"""Run loss ablations for the effective v13 soft ShadowArt branch."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

PROJECT = Path("/data/xvxinyao/xvxinyao/total_project/mesh_combined_fusion_generalizable_ready_20260528")
PYTHON = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")
OUT_ROOT = PROJECT / "runs_codex_v13_loss_ablation_formal_20260705"

BASE_CMD = [
    str(PYTHON),
    "train.py",
    "cuda:0",
    "{out}",
    "1600",
    "0.03",
    "0",
    "-sdlist",
    "Spider-Man.png",
    "armadillo_0.png",
    "batman.png",
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
    "1",
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

ABLATIONS = {
    "full": {},
    "no_silh": {"--loss_silh_scale": "0"},
    "no_l1": {"--loss_l1_scale": "0"},
    "no_ssim": {"--loss_ssim_scale": "0"},
    "no_iou_dice": {"--loss_iou_dice_scale": "0"},
    "no_asym": {"--loss_asym_scale": "0"},
    "no_fn": {"--loss_fn_scale": "0"},
    "no_reg": {"--loss_reg_scale": "0"},
}


def load_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics_history.json"
    if not path.exists():
        return {"metrics_exists": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    final = data.get("final_metrics", {})
    renderer = final.get("renderer_metrics", {})
    export = final.get("export_metrics", {})
    return {
        "metrics_exists": True,
        "mean_iou": final.get("mean_iou"),
        "mean_dice": final.get("mean_dice"),
        "renderer_mean_iou": renderer.get("mean_iou"),
        "renderer_min_iou": min(renderer.get("view_ious", []) or [None]),
        "renderer_view_ious": renderer.get("view_ious"),
        "export_strategy": export.get("export_strategy"),
        "export_mean_iou": export.get("mean_iou"),
        "export_min_iou": export.get("min_iou"),
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, flags in ABLATIONS.items():
        run_dir = OUT_ROOT / name
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [part.format(out=str(run_dir)) for part in BASE_CMD]
        for key, value in flags.items():
            cmd += [key, value]
        command_record = {
            "name": name,
            "flags": flags,
            "cmd": cmd,
            "cwd": str(PROJECT),
        }
        (run_dir / "command.json").write_text(json.dumps(command_record, indent=2), encoding="utf-8")
        log_path = run_dir / "run.log"
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=str(PROJECT), stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - started
        row = {
            "name": name,
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "run_dir": str(run_dir),
            **load_metrics(run_dir),
        }
        rows.append(row)
        (OUT_ROOT / "summary_partial.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if proc.returncode != 0:
            break
    result = {
        "schema": "v13_soft_loss_ablation_formal_v1",
        "created": "2026-07-05",
        "project": str(PROJECT),
        "base_case": ["Spider-Man.png", "armadillo_0.png", "batman.png"],
        "active_losses": [
            "silhouette",
            "l1",
            "ms_ssim",
            "iou_dice",
            "asymmetric_silhouette",
            "false_negative_only",
            "tv_isolation_regularization",
        ],
        "inactive_in_this_v13_command": [
            "boundary_loss_weight",
            "focal_tversky_weight",
            "mirror_consistency_weight",
        ],
        "rows": rows,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
