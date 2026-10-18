#!/usr/bin/env python3
"""
Batch Stage C runner for pilot and verified experiments.

Wraps train_joint.py in a dataset-level loop so that Stage C can be launched
immediately once a PINO v3 checkpoint is considered usable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import csv
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def find_shadow_files(model_dir: Path) -> list[str]:
    sil_dir = model_dir / "silhouettes"
    return sorted(str(p) for p in sil_dir.glob("view_*.png"))


def load_history(history_path: Path) -> dict:
    if not history_path.exists():
        return {}
    try:
        data = json.loads(history_path.read_text())
    except Exception:
        return {}
    if isinstance(data, list) and data:
        return data[-1]
    if isinstance(data, dict):
        return data
    return {}


def load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_last_list_entry(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    if isinstance(data, list) and data:
        last = data[-1]
        return last if isinstance(last, dict) else {}
    if isinstance(data, dict):
        return data
    return {}


def load_first_existing_json(paths: list[Path]) -> dict:
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


def apply_benchmark_list(models: list[dict], dataset_dir: Path, benchmark_list_path: Path) -> list[dict]:
    try:
        payload = json.loads(benchmark_list_path.read_text())
    except Exception:
        return models
    if not isinstance(payload, dict):
        return models
    selected_ids = set()
    for dataset in payload.get("datasets", []):
        if not isinstance(dataset, dict):
            continue
        if Path(dataset.get("dataset_dir", "")) != dataset_dir:
            continue
        for entry in dataset.get("entries", []):
            model_id = entry.get("model_id")
            if model_id:
                selected_ids.add(model_id)
    if not selected_ids:
        return models
    return [m for m in models if m.get("model_id") in selected_ids]


def classify_stage_c_failure(proc_returncode: int, final_voxel_exists: bool, last_safety: dict, final_fem: dict) -> str:
    if proc_returncode != 0:
        return "optimizer_failed"
    if not final_voxel_exists:
        return "no_final_voxels"
    selected = last_safety.get("selected_state")
    if selected not in {"best_fem_feasible"}:
        return "no_fem_feasible_state"
    if final_fem and final_fem.get("fem_safe_final") is False:
        return "final_fem_unsafe"
    return ""


def paper_acceptance(row: dict) -> bool:
    return bool(
        row.get("returncode") == 0
        and row.get("final_voxels_exists")
        and row.get("selected_state") == "best_fem_feasible"
        and row.get("sf_fem_final") is not None
        and row.get("fem_safe_final") is not None
    )


def summarize_stage_c_results(rows: list[dict]) -> dict:
    accepted = [r for r in rows if paper_acceptance(r)]
    return {
        "n_rows": len(rows),
        "n_returncode_ok": sum(1 for r in rows if r.get("returncode") == 0),
        "n_best_fem_feasible": sum(1 for r in rows if r.get("selected_state") == "best_fem_feasible"),
        "n_paper_accepted": len(accepted),
        "failure_reason_counts": {
            reason: sum(1 for r in rows if r.get("failure_reason") == reason)
            for reason in sorted({r.get("failure_reason", "") for r in rows if r.get("failure_reason")})
        },
        "mean_sf_fem_final": (
            sum(float(r["sf_fem_final"]) for r in accepted) / len(accepted) if accepted else None
        ),
        "mean_iou_final": (
            sum(float(r["iou_final"]) for r in accepted if r.get("iou_final") is not None) / len(accepted)
            if accepted and all(r.get("iou_final") is not None for r in accepted) else None
        ),
    }


def write_stage_c_breakdown(rows: list[dict], out_root: Path) -> None:
    def primary_gap(row: dict) -> str:
        if row.get("paper_accepted"):
            return "paper_accepted"
        if row.get("returncode") != 0:
            return "optimizer_failed"
        if not row.get("final_voxels_exists"):
            return "no_final_voxels"
        if row.get("selected_state") != "best_fem_feasible":
            return "no_fem_feasible_state"
        sf = row.get("sf_fem_final")
        if sf is None:
            return "missing_final_fem"
        if row.get("fem_safe_final") is False or float(sf) < 1.5:
            return "final_fem_unsafe"
        vol = row.get("vol_ratio_final")
        if vol is not None and float(vol) > 1.3:
            return "volume_budget_exceeded"
        iou = row.get("iou_final")
        if iou is not None and float(iou) < 0.99:
            return "appearance_drop"
        return row.get("failure_reason") or "unclassified"

    enriched = []
    for row in rows:
        item = dict(row)
        item["primary_gap"] = primary_gap(row)
        enriched.append(item)

    gap_counts: dict[str, int] = {}
    for row in enriched:
        gap = row["primary_gap"]
        gap_counts[gap] = gap_counts.get(gap, 0) + 1

    breakdown = {
        "schema_version": "stage_c_failure_breakdown_v1",
        "n_rows": len(enriched),
        "n_paper_accepted": sum(1 for row in enriched if row.get("paper_accepted")),
        "primary_gap_counts": dict(sorted(gap_counts.items())),
        "rows": enriched,
    }
    (out_root / "stage_c_failure_breakdown.json").write_text(json.dumps(breakdown, indent=2))

    csv_path = out_root / "stage_c_failure_breakdown.csv"
    cols = [
        "model_id",
        "category",
        "returncode",
        "selected_state",
        "paper_accepted",
        "primary_gap",
        "failure_reason",
        "sf_fem_final",
        "fem_safe_final",
        "vol_ratio_final",
        "iou_final",
        "elapsed_s",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in enriched:
            writer.writerow({k: row.get(k) for k in cols})


def main():
    parser = argparse.ArgumentParser(description="Batch Stage C pilot/verified runner")
    parser.add_argument("--dataset_dir", default="/home/lin/xvxinyao/total_project/test_datasets/modelnet40")
    parser.add_argument("--pino_checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_dir", default="/home/lin/xvxinyao/total_project/results/stage_c_verified")
    parser.add_argument("--n_models", type=int, default=5)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--sgldbench_src",
        default="/data/xvxinyao/xvxinyao/total_project/SGLDBench/python/src",
        help="Directory containing the sgldbench Python package",
    )
    parser.add_argument("--model_filter", nargs="*", default=None)
    parser.add_argument("--benchmark_list", default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    meta = json.loads((dataset_dir / "dataset_meta.json").read_text())
    models = meta.get("models", [])
    if args.benchmark_list:
        models = apply_benchmark_list(models, dataset_dir, Path(args.benchmark_list))
    if args.model_filter:
        wanted = set(args.model_filter)
        models = [m for m in models if m.get("model_id") in wanted]
    if args.start_index:
        models = models[args.start_index:]
    models = models[: args.n_models]

    results = []
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_manifest.json").write_text(
        json.dumps(
            {
                "output_schema_version": "stage_c_batch_v3",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_dir": str(dataset_dir),
                "pino_checkpoint": str(args.pino_checkpoint),
                "config_path": str(args.config) if args.config else None,
                "device": str(args.device) if args.device else None,
                "n_models_requested": int(args.n_models),
                "start_index": int(args.start_index),
                "benchmark_list": str(args.benchmark_list) if args.benchmark_list else None,
                "acceptance_rule": {
                    "returncode": 0,
                    "final_voxels_exists": True,
                    "selected_state": "best_fem_feasible",
                    "requires_final_fem_fields": ["sf_fem_final", "fem_safe_final"],
                },
            },
            indent=2,
        )
    )

    for idx, model_info in enumerate(models):
        model_id = model_info["model_id"]
        model_dir = dataset_dir / model_id
        shadow_files = find_shadow_files(model_dir)
        voxel_path = model_dir / "voxels.npy"
        if not shadow_files or not voxel_path.exists():
            results.append({"model_id": model_id, "status": "skipped", "reason": "missing_inputs"})
            continue

        model_out = out_root / model_id
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "structural" / "training" / "train_joint.py"),
            "--shadow_files",
            *shadow_files,
            "--pino_checkpoint",
            args.pino_checkpoint,
            "--output_dir",
            str(model_out),
            "--skip_stage_a",
            "--voxel_path",
            str(voxel_path),
        ]
        if args.config:
            cmd.extend(["--config", args.config])

        env = dict(**__import__("os").environ)
        if args.sgldbench_src:
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = args.sgldbench_src + ((":" + current_pythonpath) if current_pythonpath else "")
        if args.device:
            if args.device.startswith("cuda:"):
                env["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]

        print(f"[{idx+1}/{len(models)}] Running Stage C on {model_id}")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
        elapsed = time.time() - t0

        history_path = model_out / "stage_c" / "history.json"
        final_voxel = model_out / "stage_c" / "final_voxels.npy"
        safety_log_path = model_out / "stage_c" / "safety_bound_log.json"
        routing_stats_path = model_out / "stage_c" / "routing_stats.json"
        alm_state_path = model_out / "stage_c" / "alm_state.json"
        lyapunov_summary_path = model_out / "stage_c" / "lyapunov_summary.json"
        last_history = load_history(history_path)
        last_safety = load_last_list_entry(safety_log_path)
        routing_stats = load_json_dict(routing_stats_path)
        alm_state = load_json_dict(alm_state_path)
        lyapunov_summary = load_json_dict(lyapunov_summary_path)
        final_fem = load_first_existing_json(
            [
                model_out / "stage_c" / "final_fem_eval.json",
                model_out / "stage_c" / "final_fem_verification.json",
                model_out / "stage_c" / "fem_verification.json",
                model_out / "fem_verify" / "fem_verification.json",
            ]
        )
        final_fem_summary = final_fem.get("summary", final_fem if isinstance(final_fem, dict) else {})
        row = {
            "model_id": model_id,
            "category": model_info.get("category", "unknown"),
            "returncode": int(proc.returncode),
            "elapsed_s": float(elapsed),
            "config_path": str(args.config) if args.config else None,
            "pino_checkpoint": str(args.pino_checkpoint),
            "final_voxels_exists": final_voxel.exists(),
            "history_exists": history_path.exists(),
            "history_path": str(history_path) if history_path.exists() else None,
            "safety_bound_log_exists": safety_log_path.exists(),
            "routing_stats_exists": routing_stats_path.exists(),
            "alm_state_exists": alm_state_path.exists(),
            "sf_surrogate_last": last_history.get("sf_surrogate"),
            "routing_mode_last": last_history.get("routing_mode"),
            "surrogate_feasible_last": last_history.get("surrogate_feasible"),
            "vol_ratio_last": last_history.get("vol_ratio"),
            "u_max_last": last_history.get("u_max"),
            "selected_state": last_safety.get("selected_state"),
            "sf_pino_final": last_safety.get("sf_pino"),
            "sf_lower_bound_final": last_safety.get("sf_lower_bound", last_safety.get("sf_lower_bound_global")),
            "sf_fem_last_probe": last_safety.get("sf_fem"),
            "fem_safe_last_probe": last_safety.get("fem_safe"),
            "u_fem_last_probe": last_safety.get("u_fem"),
            "sf_fem_final": final_fem_summary.get("sf_fem_final", final_fem_summary.get("sf_fem")),
            "fem_safe_final": final_fem_summary.get("fem_safe_final", final_fem_summary.get("safe")),
            "vol_ratio_final": final_fem_summary.get("vol_ratio_final", last_history.get("vol_ratio")),
            "iou_final": final_fem_summary.get("iou_final"),
            "alm_feasible_final": alm_state.get("feasible"),
            "alm_rho_final": alm_state.get("rho"),
            "lyapunov_monotone_rate": lyapunov_summary.get("monotone_decrease_rate"),
            "final_fem_eval_exists": bool(final_fem),
            "output_schema_version": "stage_c_batch_v3",
        }
        row["failure_reason"] = classify_stage_c_failure(proc.returncode, final_voxel.exists(), last_safety, row)
        row["paper_accepted"] = paper_acceptance(row)
        results.append(
            row
        )

        (out_root / "stage_c_batch_results.json").write_text(json.dumps(results, indent=2))
        (out_root / "stage_c_batch_summary.json").write_text(json.dumps(summarize_stage_c_results(results), indent=2))
        write_stage_c_breakdown(results, out_root)

    print(f"Saved Stage C batch summary to {out_root / 'stage_c_batch_results.json'}")


if __name__ == "__main__":
    main()
