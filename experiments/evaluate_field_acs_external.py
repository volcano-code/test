#!/usr/bin/env python3
"""Evaluate baseline and field-tail ACS on the untouched scale-up set."""

from __future__ import annotations

import json
import math
import sys
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_candidate_pino_field import load_rows, load_scaleup_rows, predict_log_vm, subset
from train_candidate_pino_grouped import acs_metrics, split_groups
from train_pino_aug_hardneg import load_geoattn_model


def wilson_upper(k: int, n: int, z: float = 1.959963984540054) -> float | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center + half


def attach_intervals(result: dict) -> dict:
    out = dict(result)
    out["false_safe_rate_among_certified_safe"] = (
        out["false_safe"] / out["certified_safe"] if out["certified_safe"] else None
    )
    out["false_safe_rate_wilson95_upper"] = wilson_upper(out["false_safe"], out["certified_safe"])
    out["false_unsafe_rate_among_certified_unsafe"] = (
        out["false_unsafe"] / out["certified_unsafe"] if out["certified_unsafe"] else None
    )
    out["false_unsafe_rate_wilson95_upper"] = wilson_upper(out["false_unsafe"], out["certified_unsafe"])
    return out


def evaluate(model, calibration_rows, external_rows, device) -> dict:
    calibration_pred = predict_log_vm(model, calibration_rows, device)
    external_pred = predict_log_vm(model, external_rows, device)
    return attach_intervals(acs_metrics(
        calibration_pred,
        calibration_rows,
        external_pred,
        external_rows,
        alpha=0.05,
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delta",
        type=Path,
        default=ROOT / "checkpoints/pino_candidate_field_20260703/field_tail_delta.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/pino_field_acs_external_20260703.json",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows()
    calibration_rows = subset(rows, split_groups()["conformal_calibration"])
    external_rows = load_scaleup_rows()
    base = ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"
    scale = ROOT / "checkpoints/pino_aug_hardneg_20260624/scale_geoattn_e8.pt"
    delta = torch.load(args.delta, map_location="cpu", weights_only=False)

    model, _, _ = load_geoattn_model(device, base, scale)
    baseline = evaluate(model, calibration_rows, external_rows, device)
    model.dec1[-1].load_state_dict(delta["dec1_tail"])
    model.stress_head.load_state_dict(delta["stress_head"])
    candidate = evaluate(model, calibration_rows, external_rows, device)

    result = {
        "schema": "field_acs_external_v1",
        "calibration": "five case groups from aug/aug_fixed, excluded from field-tail training",
        "external_test": "data/scaleup_test.npz; 30 new triples, 360 rows",
        "baseline": baseline,
        "field_tail_candidate": candidate,
        "pass_rule": "zero false-safe, zero false-unsafe, and nonzero autonomy on external rows",
        "baseline_pass": baseline["false_safe"] == 0 and baseline["false_unsafe"] == 0 and baseline["autonomous_rate"] > 0,
        "candidate_pass": candidate["false_safe"] == 0 and candidate["false_unsafe"] == 0 and candidate["autonomous_rate"] > 0,
    }
    output = args.output
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), **result}, indent=2))


if __name__ == "__main__":
    main()
