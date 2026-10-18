#!/usr/bin/env python3
"""
Lightweight monitor for PINO training history.

Watches a `training_history.json` file written by `pretrain_pino_v3.py` and
emits a short analysis snapshot every N epochs to a JSONL file.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def load_history(path: Path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    return data if isinstance(data, list) else []


def summarize_epoch(entry: dict, prev_entry: dict | None):
    val = entry.get("val", {})
    train = entry.get("train", {})
    epoch = int(entry.get("epoch", -1))
    vm_rel = val.get("vm_max_rel_mean")
    sf_ratio = val.get("sf_ratio_mean")
    cons = val.get("conservative_pct")
    vm_phys = val.get("vm_phys_rel_mean")
    sf_phys = val.get("sf_ratio_phys_mean")
    cons_phys = val.get("conservative_pct_phys")
    scale_mae = val.get("scale_mae")
    scale_p95 = val.get("scale_p95")
    disp_rl2 = val.get("disp_rel_l2")
    stress_rl2 = val.get("stress_rel_l2")
    val_total = val.get("total")

    if prev_entry:
        prev_val = prev_entry.get("val", {})
        delta = {
            "val_total": None if val_total is None or prev_val.get("total") is None else float(val_total - prev_val["total"]),
            "vm_rel_mean": None if vm_rel is None or prev_val.get("vm_max_rel_mean") is None else float(vm_rel - prev_val["vm_max_rel_mean"]),
            "vm_phys_rel_mean": None if vm_phys is None or prev_val.get("vm_phys_rel_mean") is None else float(vm_phys - prev_val["vm_phys_rel_mean"]),
            "scale_mae": None if scale_mae is None or prev_val.get("scale_mae") is None else float(scale_mae - prev_val["scale_mae"]),
        }
    else:
        delta = {}

    if scale_mae is None:
        status = "waiting_for_scale_metrics"
    elif scale_mae < 0.30 and vm_phys is not None and vm_phys < 0.20 and sf_phys is not None and 0.8 <= sf_phys <= 1.25:
        status = "scale_recovered_candidate"
    elif scale_mae < 1.0:
        status = "scale_head_partially_reliable"
    else:
        status = "scale_head_not_recovered"

    analysis = (
        f"epoch {epoch}: val={val_total:.4f} vm_rel={vm_rel:.4f} cons={cons:.1f}% "
        f"vm_phys={vm_phys:.4f} sf_phys={sf_phys:.3f} cons_phys={cons_phys:.1f}% "
        f"scale_mae={scale_mae:.3f} scale_p95={scale_p95:.3f}; status={status}"
        if all(x is not None for x in [val_total, vm_rel, cons, vm_phys, sf_phys, cons_phys, scale_mae, scale_p95])
        else f"epoch {epoch}: partial metrics only; status={status}"
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": epoch,
        "train_total": train.get("total"),
        "val_total": val_total,
        "vm_rel_mean": vm_rel,
        "sf_ratio": sf_ratio,
        "conservative_pct": cons,
        "vm_phys_rel_mean": vm_phys,
        "sf_ratio_phys_mean": sf_phys,
        "conservative_pct_phys": cons_phys,
        "disp_rel_l2": disp_rl2,
        "stress_rel_l2": stress_rl2,
        "scale_mae": scale_mae,
        "scale_p95": scale_p95,
        "delta_from_prev_checkpoint": delta,
        "status": status,
        "analysis": analysis,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--poll_seconds", type=int, default=60)
    parser.add_argument("--stop_epoch", type=int, default=200)
    args = parser.parse_args()

    history_path = Path(args.history)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    if output_path.exists():
        try:
            for line in output_path.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if "epoch" in row:
                    seen.add(int(row["epoch"]))
        except Exception:
            pass

    while True:
        history = load_history(history_path)
        if history:
            for idx, entry in enumerate(history):
                epoch = int(entry.get("epoch", -1))
                if epoch <= 0 or epoch % args.every != 0 or epoch in seen:
                    continue
                prev_entry = history[idx - 1] if idx > 0 else None
                row = summarize_epoch(entry, prev_entry)
                with output_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(row["analysis"], flush=True)
                seen.add(epoch)
            if int(history[-1].get("epoch", -1)) >= args.stop_epoch:
                break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
