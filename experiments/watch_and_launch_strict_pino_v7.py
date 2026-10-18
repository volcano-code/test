#!/usr/bin/env python3
"""Launch strict PINO training only after the v7 dataset is complete.

The builder writes its index before all FEM samples have been materialized.  This
watcher therefore validates every indexed file and split before it starts the
next training job.  It is safe to leave running under ``nohup``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data/taskgrouped_strict_pino_delivered_v7_boundary_20260715/index.json"
DEFAULT_OUTPUT = ROOT / "checkpoints/strict_physics_residual_pino_v10_v7boundary_20260716"
DEFAULT_LOG = ROOT / "logs/strict_physics_residual_pino_v10_v7boundary_20260716.log"
DEFAULT_STATUS = ROOT / "results/strict_physics_residual_pino_v10_v7boundary_20260716_launch.json"


def validate_index(index_path: Path) -> dict:
    payload = json.loads(index_path.read_text())
    rows = payload.get("rows", [])
    if not rows:
        raise RuntimeError("index has no rows")
    counts = {split: 0 for split in ("train", "calibration", "sealed_test")}
    missing = []
    for row in rows:
        split = row.get("split")
        if split not in counts:
            continue
        counts[split] += 1
        path = index_path.parent / row["path"]
        if not path.is_file():
            missing.append(str(path))
    if any(counts[split] == 0 for split in counts):
        raise RuntimeError(f"incomplete task splits: {counts}")
    if missing:
        raise RuntimeError(f"{len(missing)} indexed FEM files are not ready")
    return {"rows": len(rows), "splits": counts}


def builder_is_running(pattern: str) -> bool:
    completed = subprocess.run(["pgrep", "-f", pattern], stdout=subprocess.DEVNULL)
    return completed.returncode == 0


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_unix_s"] = time.time()
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if (args.output_dir / "best_model.pt").is_file():
        write_status(args.status, {"state": "already_complete", "output_dir": str(args.output_dir)})
        print("A completed checkpoint already exists; nothing to launch.", flush=True)
        return

    pattern = "build_taskgrouped_decision_fem_dataset_20260711.py"
    while True:
        try:
            summary = validate_index(args.index)
            if not builder_is_running(pattern):
                break
            write_status(args.status, {"state": "waiting_for_builder", "index": str(args.index), **summary})
            print("Dataset index is readable, but FEM builder is still active.", flush=True)
        except (FileNotFoundError, json.JSONDecodeError, RuntimeError) as exc:
            write_status(args.status, {"state": "waiting_for_valid_dataset", "detail": str(exc), "index": str(args.index)})
            print(f"Waiting for valid dataset: {exc}", flush=True)
        time.sleep(args.poll_seconds)

    command = [
        args.python, "structural/training/pretrain_pino_v3.py",
        "--index_file", str(args.index), "--target_res", "48", "--include_fixed_mask",
        "--arch", "ufno_dual", "--epochs", "180", "--batch_size", "1",
        "--accum_steps", "4", "--lr", "1e-5", "--warmup_epochs", "20",
        "--no_amp", "--disable_train_augment", "--hard_case_replay_ratio", "0.35",
        "--boundary_replay_ratio", "0.70", "--boundary_log_band", "0.12",
        "--lambda_disp", "1.0", "--lambda_stress_log", "3.0",
        "--lambda_vm_asym", "2.0", "--lambda_vm_peak", "1.0", "--lambda_bc", "1.0",
        "--lambda_pde", "0.03", "--lambda_constitutive", "0.03", "--lambda_scale", "3.0",
        "--lambda_decision", "1.0", "--lambda_decision_margin", "1.0",
        "--decision_temperature", "0.12", "--decision_margin_log", "0.06",
        "--underestimate_weight", "6.0", "--overestimate_weight", "0.3",
        "--output_dir", str(args.output_dir), "--version_tag", args.output_dir.name,
    ]
    if args.dry_run:
        print(" ".join(command), flush=True)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_status(args.status, {"state": "training_exec", "pid": os.getpid(),
                               "index": str(args.index), "output_dir": str(args.output_dir),
                               "log": str(args.log), "command": command, **summary})
    # Replace the watcher process rather than spawning a child.  This is
    # essential under systemd: a child that outlives a transient watcher unit
    # is otherwise terminated with the unit control group.
    print(f"Executing PINO training in watcher PID {os.getpid()}", flush=True)
    os.chdir(ROOT)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
