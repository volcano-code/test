#!/usr/bin/env python3
"""Formal loss ablation for the verified mesh_combined2 soft branch.

Each run is an isolated copy of the historical project.  The source tree is
never modified. All variants use the same shortened protocol; a single
advanced loss is disabled through the project's public CLI flag.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


SOURCE = Path(
    "/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/"
    "shadow-project2/mesh/mesh_combined2"
)
DATA = SOURCE.parents[1] / "data"
# Each ablation is an isolated project copy. Keep this high-volume workspace on
# /data; the ShadowForge volume is intentionally reserved for paper artifacts.
OUT = Path("/data/xvxinyao/xvxinyao/ShadowForge_runs/mesh_combined2_loss_ablation_2k_20260711")
PYTHON = "/home/lin/anaconda3/envs/shadowart-revisited/bin/python"
PHASE1_ITERATIONS = 2000

# Fixed ablation protocol: 2,000 Phase-1 updates, lr=0.007,
# TSF plus mirrored views, 128^3 search and 256^3 delivery.
BASE_FLAGS = [
    "-tsf", "1", "-mr", "1", "-tsf4", "0", "-cam", "ortho",
    "-imsz", "256", "-res", "128", "-eres", "256", "-vh", "1",
    "-zdist", "2.5", "--post_export_bundle", "0",
    "--ablation_profile", "conservative",
    "--enable_sparse_teacher", "0",
]

TASKS = {
    "heroes": ["Spider-Man.png", "superman.png", "batman.png"],
    "creatures": ["puma.png", "bunny_0.png", "duck.png"],
    "mixed_icons": ["armadillo_0.png", "teddy2.png", "hand.png"],
}

# These are precisely the optional loss terms exposed by this source version.
VARIANTS = {
    "full": None,
    "w_o_multiscale": "--enable_multiscale_loss",
    "w_o_worst_view": "--enable_worst_view_loss",
    "w_o_soft_morphology": "--enable_soft_morphology",
    "w_o_distance_band": "--enable_distance_band_teacher",
    "w_o_cohesion": "--enable_cohesion_prior",
    "w_o_soft_connectivity": "--enable_soft_connectivity",
    "w_o_contour": "--enable_contour_projection_loss",
    "w_o_minimum_thickness": "--enable_minimum_thickness_prior",
}


def build_command(project: Path, run_dir: Path, files: list[str], disabled: str | None) -> list[str]:
    command = [
        PYTHON, "train.py", "cuda:0", str(run_dir), str(PHASE1_ITERATIONS), "0.007", "0",
        "-sdlist", *[str(DATA / name) for name in files], *BASE_FLAGS,
    ]
    if disabled:
        command.extend([disabled, "0"])
    return command


def copy_source(project: Path) -> None:
    if project.exists():
        shutil.rmtree(project)
    shutil.copytree(
        SOURCE,
        project,
        ignore=shutil.ignore_patterns("output", "__pycache__", "*.pyc", ".git"),
    )
    # Persist phase-end metrics in the isolated copy while it is training.
    train = project / "train.py"
    text = train.read_text(encoding="utf-8")
    marker = "metrics_history = {'history': [], 'final_metrics': {}}"
    replacement = (
        "metrics_history = {'history': [], 'final_metrics': {}, "
        "'_phase_metrics_live_path': os.path.join(output_dir, 'phase_metrics_live.json')}"
    )
    if marker not in text:
        raise RuntimeError("could not inject live phase metric path")
    train.write_text(text.replace(marker, replacement, 1), encoding="utf-8")

    runtime = project / "phase_runtime.py"
    text = runtime.read_text(encoding="utf-8")
    marker = "    phase_reports.append(report)\n    return end_mean_iou, end_mean_dice\n"
    replacement = (
        "    phase_reports.append(report)\n"
        "    live_path = metrics_history.get('_phase_metrics_live_path')\n"
        "    if live_path:\n"
        "        payload = dict(metrics_history)\n"
        "        payload['phase_reports'] = list(phase_reports)\n"
        "        with open(live_path, 'w', encoding='utf-8') as handle:\n"
        "            json.dump(payload, handle, indent=2)\n"
        "    return end_mean_iou, end_mean_dice\n"
    )
    if marker not in text:
        raise RuntimeError("could not inject live phase persistence")
    runtime.write_text(text.replace(marker, replacement, 1), encoding="utf-8")


def metric_value(metrics: dict, name: str) -> float | None:
    value = metrics.get(name)
    return None if value is None else float(value)


def read_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics_history.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    final = payload.get("final_metrics", {})
    export = final.get("export_metrics", {})
    renderer = final.get("renderer_metrics", {})
    components = int(export.get("component_count", 0) or 0)
    mesh_candidates = sorted(run_dir.glob("*.obj"))
    watertight = None
    mesh_parts = None
    try:
        import trimesh

        if mesh_candidates:
            mesh = trimesh.load(mesh_candidates[-1], force="mesh")
            watertight = bool(mesh.is_watertight)
            mesh_parts = int(len(mesh.split(only_watertight=False)))
    except Exception:
        pass
    return {
        "renderer_iou": metric_value(renderer, "mean_iou"),
        "renderer_dice": metric_value(renderer, "mean_dice"),
        "delivered_iou_256": metric_value(export, "mean_iou"),
        "delivered_dice_256": metric_value(export, "mean_dice"),
        "delivered_min_iou_256": metric_value(export, "min_iou"),
        "occupied_voxels_256": export.get("occupied_voxels"),
        "component_count": components,
        "single_body": components == 1,
        "mesh_watertight": watertight,
        "mesh_parts": mesh_parts,
        "export_strategy": final.get("export_strategy"),
    }


def write_summary(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    columns = [
        "task", "variant", "returncode", "elapsed_minutes", "renderer_iou",
        "renderer_dice", "delivered_iou_256", "delivered_dice_256",
        "delivered_min_iou_256", "occupied_voxels_256", "component_count",
        "single_body", "mesh_watertight", "mesh_parts", "export_strategy", "run_dir",
    ]
    with (OUT / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_one(task: str, files: list[str], variant: str, disabled: str | None) -> dict:
    project = OUT / "projects" / task / variant
    run_dir = OUT / "runs" / task / variant
    row_path = run_dir / "row.json"
    if row_path.exists():
        return json.loads(row_path.read_text(encoding="utf-8"))
    copy_source(project)
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(project, run_dir, files, disabled)
    manifest = {
        "task": task,
        "input_files": files,
        "variant": variant,
        "disabled_loss_flag": disabled,
        "protocol": {
            "phase1_iterations": PHASE1_ITERATIONS,
            "base_lr": 0.007,
            "tsf_mode": 1,
            "mirror_mode": 1,
            "voxel_resolution": 128,
            "export_resolution": 256,
            "visual_hull": 1,
            "loss_profile": "conservative",
            "all_other_optional_losses_enabled": True,
        },
        "command": command,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.time()
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=project, stdout=log, stderr=subprocess.STDOUT, text=True)
    row = {
        "task": task,
        "variant": variant,
        "returncode": process.returncode,
        "elapsed_minutes": round((time.time() - started) / 60.0, 2),
        "run_dir": str(run_dir),
        **read_metrics(run_dir),
    }
    row_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> None:
    if not SOURCE.is_dir() or not Path(PYTHON).is_file():
        raise RuntimeError("source project or shadowart-revisited Python is unavailable")
    missing = [name for files in TASKS.values() for name in files if not (DATA / name).is_file()]
    if missing:
        raise RuntimeError(f"missing input silhouettes: {missing}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "experiment_manifest.json").write_text(
        json.dumps({"source": str(SOURCE), "tasks": TASKS, "variants": VARIANTS, "base_flags": BASE_FLAGS}, indent=2),
        encoding="utf-8",
    )
    rows: list[dict] = []
    for task, files in TASKS.items():
        for variant, disabled in VARIANTS.items():
            row = run_one(task, files, variant, disabled)
            rows.append(row)
            write_summary(rows)
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
