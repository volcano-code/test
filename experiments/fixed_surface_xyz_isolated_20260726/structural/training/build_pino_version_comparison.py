#!/usr/bin/env python3
"""
Build a compact comparison table across multiple PINO checkpoint directories.

Usage:
  python structural/training/build_pino_version_comparison.py \
      --checkpoints \
          checkpoints/pino_v16_ufno \
          checkpoints/pino_v18_ufno \
          checkpoints/pino_v19_ufno \
          checkpoints/pino_v22_scaleaware_ufno \
      --output_json results/pino_version_comparison.json \
      --output_csv results/pino_version_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_history(checkpoint_dir: Path) -> list[dict]:
    history = load_json(checkpoint_dir / "training_history.json")
    if isinstance(history, list):
        return [x for x in history if isinstance(x, dict)]
    return []


def load_best(checkpoint_dir: Path) -> dict:
    path = checkpoint_dir / "best_model.pt"
    if not path.exists():
        return {}
    try:
        best = torch.load(path, map_location="cpu")
    except Exception:
        best = load_json(path)
    return best if isinstance(best, dict) else {}


def classify_row(best: dict) -> str:
    if not best:
        return "missing"
    sf_phys = best.get("sf_ratio_phys_mean")
    scale_mae = best.get("scale_mae")
    vm_phys = best.get("vm_phys_rel_mean")
    cons_phys = best.get("conservative_pct_phys")
    if sf_phys is None or vm_phys is None:
        return "normalized-only strong"
    if scale_mae is not None and scale_mae > 1.0:
        return "physical-scale failed"
    if sf_phys < 0.8:
        return "over-conservative"
    if 0.8 <= sf_phys <= 1.25 and vm_phys < 0.2 and cons_phys is not None and 40.0 <= cons_phys <= 80.0:
        return "balanced candidate"
    return "physical-scale recovering"


def build_record(checkpoint_dir: Path) -> dict:
    history = load_history(checkpoint_dir)
    best = load_best(checkpoint_dir)
    latest = history[-1] if history else {}
    best_epoch = best.get("epoch")
    return {
        "version": checkpoint_dir.name,
        "checkpoint_dir": str(checkpoint_dir),
        "n_epochs_logged": len(history),
        "best_epoch": best_epoch,
        "best_selection_space": best.get("best_selection_space"),
        "latest_epoch": latest.get("epoch"),
        "latest_val_total": latest.get("val_total"),
        "latest_vm_rel": latest.get("vm_max_rel_mean"),
        "latest_sf_ratio": latest.get("sf_ratio_mean"),
        "latest_conservative_pct": latest.get("conservative_pct"),
        "latest_vm_phys_rel_mean": latest.get("vm_phys_rel_mean"),
        "latest_sf_ratio_phys_mean": latest.get("sf_ratio_phys_mean"),
        "latest_conservative_pct_phys": latest.get("conservative_pct_phys"),
        "latest_scale_mae": latest.get("scale_mae"),
        "best_val_total": best.get("val_loss"),
        "best_vm_rel": best.get("vm_max_rel_mean"),
        "best_p95_vm_error": best.get("p95_vm_error"),
        "best_sf_ratio": best.get("sf_ratio_mean"),
        "best_conservative_pct": best.get("conservative_pct"),
        "best_vm_phys_rel_mean": best.get("vm_phys_rel_mean"),
        "best_sf_ratio_phys_mean": best.get("sf_ratio_phys_mean"),
        "best_conservative_pct_phys": best.get("conservative_pct_phys"),
        "best_scale_mae": best.get("scale_mae"),
        "best_scale_p95": best.get("scale_p95"),
        "classification": classify_row(best),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    rows = [build_record(Path(p)) for p in args.checkpoints]
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"versions": rows}, indent=2))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
