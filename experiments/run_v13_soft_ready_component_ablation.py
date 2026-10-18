from __future__ import annotations

import json
import subprocess
from pathlib import Path
from statistics import mean


OLD_ROOT = Path("/data/xvxinyao/xvxinyao/total_project/mesh_combined_fusion_generalizable_ready_20260528")
PYTHON = "/home/lin/anaconda3/envs/shadowart-revisited/bin/python"
CASE_FILE = OLD_ROOT / "soft_teacher_seed_cases_v13.jsonl"


VARIANTS = {
    "full_v13": {
        "iters": 1600,
        "lr": 0.03,
        "vh": 1,
        "view_weight_power": 3.0,
        "weak_polish_iters": 300,
        "weak_view_threshold": 0.9,
        "weak_focus_weight": 4.0,
        "protect_good_weight": 4.0,
        "weak_polish_lr_scale": 0.015,
    },
    "no_weak_polish": {
        "iters": 1600,
        "lr": 0.03,
        "vh": 1,
        "view_weight_power": 3.0,
        "weak_polish_iters": 0,
        "weak_view_threshold": 0.9,
        "weak_focus_weight": 4.0,
        "protect_good_weight": 4.0,
        "weak_polish_lr_scale": 0.015,
    },
    "no_view_reweight": {
        "iters": 1600,
        "lr": 0.03,
        "vh": 1,
        "view_weight_power": 0.0,
        "weak_polish_iters": 300,
        "weak_view_threshold": 0.9,
        "weak_focus_weight": 4.0,
        "protect_good_weight": 4.0,
        "weak_polish_lr_scale": 0.015,
    },
    "no_view_reweight_no_polish": {
        "iters": 1600,
        "lr": 0.03,
        "vh": 1,
        "view_weight_power": 0.0,
        "weak_polish_iters": 0,
        "weak_view_threshold": 0.9,
        "weak_focus_weight": 4.0,
        "protect_good_weight": 4.0,
        "weak_polish_lr_scale": 0.015,
    },
    "no_visual_hull": {
        "iters": 1600,
        "lr": 0.03,
        "vh": 0,
        "view_weight_power": 3.0,
        "weak_polish_iters": 300,
        "weak_view_threshold": 0.9,
        "weak_focus_weight": 4.0,
        "protect_good_weight": 4.0,
        "weak_polish_lr_scale": 0.015,
    },
}


def read_cases() -> list[dict]:
    cases = []
    with CASE_FILE.open() as f:
        for line in f:
            item = json.loads(line)
            cases.append(
                {
                    "files": item["files"],
                    "mr": int(item.get("mr", 0)),
                    "tsf": int(item.get("tsf", 0)),
                    "tsf4": int(item.get("tsf4", 0)),
                }
            )
    return cases


def metric_entry(metrics_path: Path) -> dict:
    data = json.loads(metrics_path.read_text())
    final = data["final_metrics"]
    renderer = final.get("renderer_metrics", {})
    export = final.get("export_metrics", {})
    return {
        "renderer_mean_iou": renderer.get("mean_iou"),
        "renderer_min_iou": min(renderer.get("view_ious", []) or [None]),
        "export_mean_iou": export.get("mean_iou"),
        "export_min_iou": export.get("min_iou"),
        "export_strategy": export.get("export_strategy"),
    }


def aggregate(entries: list[dict]) -> dict:
    ok = [e for e in entries if e["status"] == "ok"]

    def avg(key: str):
        vals = [e[key] for e in ok if isinstance(e.get(key), (int, float))]
        return mean(vals) if vals else None

    def minv(key: str):
        vals = [e[key] for e in ok if isinstance(e.get(key), (int, float))]
        return min(vals) if vals else None

    return {
        "num_ok": len(ok),
        "num_failed": len(entries) - len(ok),
        "renderer_mean_iou": avg("renderer_mean_iou"),
        "renderer_worst_view_iou": minv("renderer_min_iou"),
        "export_mean_iou": avg("export_mean_iou"),
        "export_worst_view_iou": minv("export_min_iou"),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--indices", default="0,1,2")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cases = read_cases()
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    report = {}

    for variant, cfg in VARIANTS.items():
        entries = []
        variant_dir = out_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        for idx in indices:
            case = cases[idx]
            sample_dir = variant_dir / f"sample_{idx:03d}_ready"
            log_path = variant_dir / f"sample_{idx:03d}_ready.log"
            metrics_path = sample_dir / "metrics_history.json"
            if metrics_path.exists() and not args.overwrite:
                entry = {"index": idx, "status": "ok", **metric_entry(metrics_path)}
                entries.append(entry)
                continue
            cmd = [
                PYTHON,
                "train.py",
                args.device,
                str(sample_dir),
                str(cfg["iters"]),
                str(cfg["lr"]),
                "0",
                "-sdlist",
                *case["files"],
                "-mr",
                str(case["mr"]),
                "-tsf",
                str(case["tsf"]),
                "-tsf4",
                str(case["tsf4"]),
                "-imsz",
                "128",
                "-res",
                "128",
                "-vh",
                str(cfg["vh"]),
                "--skip_final_export",
                "1",
                "--view_weight_power",
                str(cfg["view_weight_power"]),
                "--view_weight_floor",
                "0.5",
                "--weak_polish_iters",
                str(cfg["weak_polish_iters"]),
                "--weak_view_threshold",
                str(cfg["weak_view_threshold"]),
                "--weak_focus_weight",
                str(cfg["weak_focus_weight"]),
                "--protect_good_weight",
                str(cfg["protect_good_weight"]),
                "--weak_polish_lr_scale",
                str(cfg["weak_polish_lr_scale"]),
            ]
            with log_path.open("w") as log:
                log.write("RUN: " + " ".join(cmd) + "\n")
                log.flush()
                proc = subprocess.run(cmd, cwd=OLD_ROOT, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode == 0 and metrics_path.exists():
                entry = {"index": idx, "status": "ok", **metric_entry(metrics_path)}
            else:
                entry = {"index": idx, "status": "failed", "returncode": proc.returncode}
            entries.append(entry)
            (out_root / "ablation_report.json").write_text(json.dumps(report, indent=2))
        report[variant] = {
            "config": cfg,
            "aggregate": aggregate(entries),
            "entries": entries,
        }
        (out_root / "ablation_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
