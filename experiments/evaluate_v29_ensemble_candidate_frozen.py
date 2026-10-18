#!/usr/bin/env python3
"""Frozen v29 ensemble + v27/v28 disagreement on grouped candidate data."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from structural.models.load_model import load_surrogate
from train_candidate_pino_field import load_rows, load_scaleup_rows
from train_pino_aug_hardneg import xg

YIELD_STRESS = 50e6
SF_TARGET = 1.3
ALPHA = 0.02
FEM_BUDGET = 0.08
UNSAFE_ALPHA = 0.30
SPLIT_SEED = 20260703

BASE = ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"
CHECKPOINTS = {
    "v27": ROOT / "checkpoints/v27_shadowart_scale_sf_safety/best_scale_sf.pt",
    "v28": ROOT / "checkpoints/v28_shadowart_scale_sf_300case/best_scale_sf.pt",
    "v29a": ROOT / "checkpoints/v29_v27_pinball/best_scale_sf.pt",
    "v29b": ROOT / "checkpoints/v29b_v27_pinball_s2605/best_scale_sf.pt",
    "v29c": ROOT / "checkpoints/v29c_v27_pinball_s2606/best_scale_sf.pt",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path: Path, device: torch.device):
    model, _ = load_surrogate(str(BASE), device=device)
    model.enable_enhanced_scale_head()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.scale_head_v2.load_state_dict(checkpoint["scale_head_v2_state_dict"])
    for name, state in checkpoint["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def predict(model, rows: list[dict], device: torch.device, batch_size: int = 16) -> np.ndarray:
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        inputs = torch.cat([xg(model, row["density"], row["force"], device) for row in batch], dim=0)
        values = model._scale_forward(inputs).squeeze(-1).reshape(-1)
        output.extend(float(value) for value in values.detach().cpu())
    return np.asarray(output, dtype=float)


def conformal_offset(prediction: np.ndarray, truth: np.ndarray, alpha: float) -> float:
    residual = truth - prediction
    rank = int(np.ceil((1.0 - alpha) * (len(residual) + 1)))
    return float(np.sort(residual)[min(max(rank, 1), len(residual)) - 1])


def policy_metrics(
    calibration_prediction: np.ndarray,
    test_prediction: np.ndarray,
    calibration_disagreement: np.ndarray,
    test_disagreement: np.ndarray,
    calibration_rows: list[dict],
    test_rows: list[dict],
) -> dict:
    calibration_true_log = np.asarray([row["log_vm"] for row in calibration_rows], dtype=float)
    test_true_sf = np.asarray([row["sf"] for row in test_rows], dtype=float)
    q_safe = conformal_offset(calibration_prediction, calibration_true_log, ALPHA)
    q_unsafe = float(np.quantile(calibration_true_log - calibration_prediction, UNSAFE_ALPHA))
    trigger_threshold = float(np.quantile(calibration_disagreement, 1.0 - FEM_BUDGET))

    sf_lower = YIELD_STRESS / np.maximum(10.0 ** (test_prediction + q_safe), 1e-12)
    sf_upper = YIELD_STRESS / np.maximum(10.0 ** (test_prediction + q_unsafe), 1e-12)
    predicted_safe = sf_lower >= SF_TARGET
    predicted_unsafe = sf_upper < SF_TARGET
    trigger = test_disagreement >= trigger_threshold
    fem = trigger & ~predicted_unsafe
    true_safe = test_true_sf >= SF_TARGET
    final_safe = np.where(fem, true_safe, np.where(predicted_unsafe, False, predicted_safe))
    auto = ~fem
    return {
        "n": len(test_rows),
        "alpha": ALPHA,
        "fem_budget": FEM_BUDGET,
        "unsafe_alpha": UNSAFE_ALPHA,
        "q_safe": q_safe,
        "q_unsafe": q_unsafe,
        "trigger_threshold": trigger_threshold,
        "autonomous_count": int(auto.sum()),
        "autonomous_rate": float(auto.mean()),
        "fem_count": int(fem.sum()),
        "system_false_pass": int((final_safe & ~true_safe).sum()),
        "system_false_fail": int((~final_safe & true_safe).sum()),
        "safe_accept_count": int(final_safe.sum()),
        "true_safe_count": int(true_safe.sum()),
        "unsafe_reject_count": int(predicted_unsafe.sum()),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_rows = load_rows()
    external_rows = load_scaleup_rows()
    groups = np.arange(40)
    np.random.default_rng(SPLIT_SEED).shuffle(groups)
    calibration_groups = sorted(int(value) for value in groups[:20])
    test_groups = sorted(int(value) for value in groups[20:])
    calibration_rows = [row for row in candidate_rows if row["group"] in set(calibration_groups)]
    test_rows = [row for row in candidate_rows if row["group"] in set(test_groups)]

    predictions = {"candidate": {}, "external": {}}
    for name, path in CHECKPOINTS.items():
        model = load_model(path, device)
        predictions["candidate"][name] = predict(model, candidate_rows, device)
        predictions["external"][name] = predict(model, external_rows, device)
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"completed_model": name}), flush=True)

    candidate_indices = {id(row): index for index, row in enumerate(candidate_rows)}
    cal_indices = np.asarray([candidate_indices[id(row)] for row in calibration_rows], dtype=int)
    test_indices = np.asarray([candidate_indices[id(row)] for row in test_rows], dtype=int)
    candidate_mean = np.mean(
        np.stack([predictions["candidate"][name] for name in ("v29a", "v29b", "v29c")]), axis=0
    )
    candidate_disagreement = np.abs(predictions["candidate"]["v27"] - predictions["candidate"]["v28"])
    external_mean = np.mean(
        np.stack([predictions["external"][name] for name in ("v29a", "v29b", "v29c")]), axis=0
    )
    external_disagreement = np.abs(predictions["external"]["v27"] - predictions["external"]["v28"])

    heldout = policy_metrics(
        candidate_mean[cal_indices], candidate_mean[test_indices],
        candidate_disagreement[cal_indices], candidate_disagreement[test_indices],
        calibration_rows, test_rows,
    )
    external = policy_metrics(
        candidate_mean[cal_indices], external_mean,
        candidate_disagreement[cal_indices], external_disagreement,
        calibration_rows, external_rows,
    )
    pass_rule = {
        "candidate_zero_false_pass": heldout["system_false_pass"] == 0,
        "candidate_autonomy_at_least_80pct": heldout["autonomous_rate"] >= 0.80,
        "external_zero_false_pass": external["system_false_pass"] == 0,
        "external_autonomy_at_least_80pct": external["autonomous_rate"] >= 0.80,
    }
    promotion_eligible = all(pass_rule.values())
    result = {
        "schema": "frozen_v29_candidate_validation_v1",
        "frozen_protocol": {
            "alpha": ALPHA,
            "fem_budget": FEM_BUDGET,
            "unsafe_alpha": UNSAFE_ALPHA,
            "selection_source": "raw_v29_boundary_member_acs_eval_20260624.json; no candidate-test tuning",
        },
        "split": {
            "seed": SPLIT_SEED,
            "calibration_groups": calibration_groups,
            "heldout_groups": test_groups,
            "calibration_rows": len(calibration_rows),
            "heldout_rows": len(test_rows),
        },
        "candidate_heldout": heldout,
        "external_scaleup": external,
        "promotion_gate": pass_rule,
        "promotion_eligible": promotion_eligible,
        "authoritative_pipeline_modified": False,
        "checkpoint_sha256": {name: sha256(path) for name, path in CHECKPOINTS.items()},
        "limitations": [
            "Safety labels are voxel-FEM references, not physical measurements.",
            "The 80% autonomy promotion threshold is predeclared for this validation run.",
        ],
    }
    output = ROOT / "results/v29_ensemble_candidate_frozen_20260703.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    cache = ROOT / "results/v29_ensemble_candidate_predictions_20260703.npz"
    np.savez_compressed(cache, **{
        f"candidate_{name}": values for name, values in predictions["candidate"].items()
    }, **{
        f"external_{name}": values for name, values in predictions["external"].items()
    })
    print(json.dumps({"output": str(output), "prediction_cache": str(cache), **result}, indent=2))


if __name__ == "__main__":
    main()
