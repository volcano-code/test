#!/usr/bin/env python3
"""Run missing full-config soft ablations sequentially.

Each variant calls experiments/soft_branch_ablation_full_config.py and writes a
separate result directory. Existing completed summaries are skipped unless
--force is supplied.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
PYTHON = sys.executable
DEFAULT_OUT = ROOT / "results/soft_full_loss_ablation_20260704"


VARIANTS: dict[str, dict[str, float]] = {
    "full": {},
    "no_topo_connectivity": {"loss_topo_connectivity_scale": 0.0, "conn_w": 0.0},
    "no_volume": {"loss_volume_scale": 0.0, "vol_min": 0.0},
    "no_support": {"loss_support_scale": 0.0},
    "no_manufacturability": {"loss_manufacturability_scale": 0.0},
    "no_export_consistency": {"loss_export_consistency_scale": 0.0},
    "no_main_structure": {"loss_main_structure_scale": 0.0},
    "no_export_distill": {"loss_export_distill_scale": 0.0},
    "no_tfs_projection": {"loss_tfs_projection_scale": 0.0},
}


def read_summary(path: Path) -> dict | None:
    summary = path / "soft_branch_ablation_full_config.json"
    if not summary.exists():
        return None
    try:
        return json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--niter", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=0.007)
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--iou_gate", type=float, default=0.95)
    parser.add_argument("--vol_min", type=float, default=0.35)
    parser.add_argument("--conn_w", type=float, default=1.0)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    status_path = args.out / "RUN_STATUS.json"
    report: dict[str, dict] = {}
    variants = args.only if args.only else list(VARIANTS)

    for name in variants:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant: {name}")
        variant_dir = args.out / name
        existing = read_summary(variant_dir)
        if existing is not None and not args.force:
            report[name] = {
                "status": "skipped_existing",
                "result": str(variant_dir / "soft_branch_ablation_full_config.json"),
                "aggregate": existing.get("aggregate"),
                "config": existing.get("config"),
            }
            status_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            continue

        cfg = {
            "vol_min": args.vol_min,
            "conn_w": args.conn_w,
            "loss_export_consistency_scale": 1.0,
            "loss_support_scale": 1.0,
            "loss_main_structure_scale": 1.0,
            "loss_manufacturability_scale": 1.0,
            "loss_export_distill_scale": 1.0,
            "loss_tfs_projection_scale": 1.0,
            "loss_volume_scale": 1.0,
            "loss_topo_connectivity_scale": 1.0,
        }
        cfg.update(VARIANTS[name])
        cmd = [
            PYTHON,
            str(ROOT / "experiments/soft_branch_ablation_full_config.py"),
            "--out",
            str(variant_dir),
            "--niter",
            str(args.niter),
            "--lr",
            str(args.lr),
            "--res",
            str(args.res),
            "--iou_gate",
            str(args.iou_gate),
            "--vol_min",
            str(cfg["vol_min"]),
            "--conn_w",
            str(cfg["conn_w"]),
            "--loss_export_consistency_scale",
            str(cfg["loss_export_consistency_scale"]),
            "--loss_support_scale",
            str(cfg["loss_support_scale"]),
            "--loss_main_structure_scale",
            str(cfg["loss_main_structure_scale"]),
            "--loss_manufacturability_scale",
            str(cfg["loss_manufacturability_scale"]),
            "--loss_export_distill_scale",
            str(cfg["loss_export_distill_scale"]),
            "--loss_tfs_projection_scale",
            str(cfg["loss_tfs_projection_scale"]),
            "--loss_volume_scale",
            str(cfg["loss_volume_scale"]),
            "--loss_topo_connectivity_scale",
            str(cfg["loss_topo_connectivity_scale"]),
        ]
        if args.force:
            cmd.append("--force")

        report[name] = {
            "status": "running",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "command": cmd,
            "config": cfg,
        }
        status_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log_path = variant_dir / "variant_runner.log"
        variant_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

        summary = read_summary(variant_dir)
        report[name].update({
            "status": "ok" if proc.returncode == 0 and summary is not None else "failed",
            "returncode": proc.returncode,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "log": str(log_path),
            "result": str(variant_dir / "soft_branch_ablation_full_config.json"),
            "aggregate": summary.get("aggregate") if summary else None,
        })
        status_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
