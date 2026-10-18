#!/usr/bin/env python3
"""Summarize v13 soft ablation logs produced with --skip_final_export."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/v13_soft_loss_ablation_20260703"
OUT_JSON = OUT_ROOT / "soft_log_ablation_summary.json"

FINAL_RE = re.compile(r"Final Model: IOU=([0-9.]+), DICE=([0-9.]+)")
VIEW_RE = re.compile(r"View\s+(\d+): IOU=([0-9.]+), DICE=([0-9.]+)")


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    final_matches = FINAL_RE.findall(text)
    view_matches = VIEW_RE.findall(text)
    if not final_matches:
        return {"status": "incomplete", "log": str(path)}
    final_iou, final_dice = final_matches[-1]
    views = [{"view": int(v), "iou": float(iou), "dice": float(dice)} for v, iou, dice in view_matches[-6:]]
    return {
        "status": "ok",
        "log": str(path),
        "final_iou": float(final_iou),
        "final_dice": float(final_dice),
        "min_view_iou": min((v["iou"] for v in views), default=None),
        "views": views,
    }


def aggregate(entries: list[dict]) -> dict:
    ok = [entry for entry in entries if entry["status"] == "ok"]
    min_view_values = [entry["min_view_iou"] for entry in ok if entry["min_view_iou"] is not None]
    return {
        "num_ok": len(ok),
        "num_total": len(entries),
        "mean_final_iou": mean([entry["final_iou"] for entry in ok]) if ok else None,
        "mean_min_view_iou": mean(min_view_values) if min_view_values else None,
        "worst_min_view_iou": min(min_view_values, default=None),
    }


def main() -> None:
    report = {}
    for variant_dir in sorted(path for path in OUT_ROOT.iterdir() if path.is_dir()):
        entries = []
        for log in sorted(variant_dir.glob("sample_*_ready.log")):
            entry = parse_log(log)
            match = re.search(r"sample_(\\d+)_ready", log.name)
            entry["index"] = int(match.group(1)) if match else None
            entries.append(entry)
        report[variant_dir.name] = {
            "aggregate": aggregate(entries),
            "entries": entries,
        }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT_JSON), "variants": report}, indent=2))


if __name__ == "__main__":
    main()
