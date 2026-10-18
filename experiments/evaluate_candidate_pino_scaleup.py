#!/usr/bin/env python3
"""Compare candidate PINO checkpoints on the separate scale-up dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_candidate_pino_grouped import metrics, predict
from train_pino_aug_hardneg import load_geoattn_model
from structural.analysis.load_generator import LoadGenerator


def load_rows() -> list[dict]:
    payload = np.load(ROOT / "data/scaleup_test.npz")
    generator = LoadGenerator()
    rows = []
    for index in range(len(payload["dens"])):
        density = payload["dens"][index].astype(np.float32)
        force = generator.standing_load(density)["force"] * float(payload["mult"][index])
        rows.append({
            "density": density,
            "force": force.astype(np.float32),
            "log_vm": float(np.log10(max(float(payload["vm"][index]), 1e-12))),
            "sf": float(payload["sf"][index]),
        })
    return rows


def evaluate(checkpoint: Path, rows: list[dict], device: torch.device) -> dict:
    base = ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"
    model, _, _ = load_geoattn_model(device, base, checkpoint)
    result = metrics(predict(model, rows, device, batch_size=16), rows)
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows()
    current_path = ROOT / "checkpoints/pino_aug_hardneg_20260624/scale_geoattn_e8.pt"
    candidate_path = ROOT / "checkpoints/pino_candidate_grouped_20260703/best_grouped_candidate.pt"
    result = {
        "schema": "candidate_pino_scaleup_comparison_v1",
        "dataset": "data/scaleup_test.npz",
        "dataset_provenance": "30 triples generated with seed 99; 4 geometries x 3 load multipliers; not used by either training script",
        "n": len(rows),
        "current_hardneg": evaluate(current_path, rows, device),
        "new_grouped_candidate": evaluate(candidate_path, rows, device),
        "pipeline_modified": False,
    }
    output = ROOT / "results/pino_candidate_scaleup_eval_20260703.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), **result}, indent=2))


if __name__ == "__main__":
    main()
