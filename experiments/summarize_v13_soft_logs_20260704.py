#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/v13_soft_component_ablation_20260704"

FINAL_RE = re.compile(r"Final Model: IOU=([0-9.]+), DICE=([0-9.]+)")
VIEW_RE = re.compile(r"View\s+(\d+): IOU=([0-9.]+), DICE=([0-9.]+)")
BEST_RE = re.compile(r"Final best IOU: ([0-9.]+)")


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    finals = FINAL_RE.findall(text)
    bests = BEST_RE.findall(text)
    views = VIEW_RE.findall(text)
    if finals:
        iou, dice = finals[-1]
        view_tail = views[-3:] if len(views) >= 3 else views
        view_ious = [float(v[1]) for v in view_tail]
        return {
            "status": "ok",
            "renderer_mean_iou": float(iou),
            "renderer_dice": float(dice),
            "renderer_min_iou": min(view_ious) if view_ious else None,
            "view_ious": view_ious,
            "final_best_iou": float(bests[-1]) if bests else float(iou),
        }
    return {
        "status": "running_or_incomplete",
        "final_best_iou": float(bests[-1]) if bests else None,
    }


def aggregate(entries: list[dict]) -> dict:
    ok = [e for e in entries if e.get("status") == "ok"]
    vals = lambda key: [e[key] for e in ok if isinstance(e.get(key), (int, float))]
    return {
        "num_ok": len(ok),
        "num_total": len(entries),
        "renderer_mean_iou": mean(vals("renderer_mean_iou")) if vals("renderer_mean_iou") else None,
        "renderer_worst_view_iou": min(vals("renderer_min_iou")) if vals("renderer_min_iou") else None,
        "final_best_iou_mean": mean(vals("final_best_iou")) if vals("final_best_iou") else None,
    }


def main() -> None:
    report = {}
    for variant_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        logs = sorted(variant_dir.glob("sample_*_ready.log"))
        entries = []
        for log in logs:
            entry = {"index": int(log.stem.split("_")[1]), "log": str(log), **parse_log(log)}
            entries.append(entry)
        report[variant_dir.name] = {"aggregate": aggregate(entries), "entries": entries}
    dest = OUT / "ablation_report_from_logs.json"
    dest.write_text(json.dumps({"schema": "v13_soft_log_summary_v1", "report": report}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
