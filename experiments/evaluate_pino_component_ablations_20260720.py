#!/usr/bin/env python3
"""Evaluate one-factor PINO component ablations on the immutable sealed split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import experiments.evaluate_taskdisjoint_five_model_comparator_20260717 as comparator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--no-physics", type=Path, required=True)
    parser.add_argument("--no-scale-fno", type=Path, required=True)
    parser.add_argument("--no-asymmetric", type=Path, required=True)
    parser.add_argument("--no-context", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    comparator.INDEX = args.index
    payload = json.loads(args.index.read_text())
    rows = [row for row in payload["rows"] if row["split"] == "sealed_test"]
    truth = np.log10(np.asarray([float(row["sf"]) for row in rows], dtype=float))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoints = {
        "full_pino": args.full,
        "w_o_physics": args.no_physics,
        "w_o_scale_fno": args.no_scale_fno,
        "w_o_asymmetric_loss": args.no_asymmetric,
        "w_o_structural_context": args.no_context,
    }

    result = {
        "schema": "pino_component_ablation_v1",
        "index": str(args.index),
        "split": "sealed_test",
        "samples": len(rows),
        "threshold_sf": comparator.TAU,
        "models": {},
    }
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {name}: {checkpoint}")
        print(f"Evaluating {name}: {checkpoint}", flush=True)
        pred, elapsed, parameters = comparator.operator_predictions(checkpoint, rows, device)
        result["models"][name] = {
            "checkpoint": str(checkpoint),
            "parameters": parameters,
            "inference_ms_per_sample": 1000.0 * elapsed / len(rows),
            **comparator.raw_summary(pred, truth),
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    columns = [
        "model", "peak_log_stress_mae", "auroc", "auto_rate",
        "auto_safe_rate", "auto_unsafe_rate", "balanced_accuracy",
        "false_safe_count", "false_unsafe_count", "inference_ms_per_sample",
    ]
    lines = ["\t".join(columns)]
    for name, row in result["models"].items():
        lines.append("\t".join(str(row[column]) if column != "model" else name
                               for column in columns))
    (args.out_dir / "results.tsv").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(args.out_dir), "models": list(checkpoints)}, indent=2))


if __name__ == "__main__":
    main()
