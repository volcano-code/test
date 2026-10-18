#!/usr/bin/env python3
"""Audit domain-separated thresholds for the near-boundary safety gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_near_boundary_safety_decision_gate import (
    TAU,
    choose_threshold,
    evaluate,
    load_hardset_rows,
    row_features,
)
from train_candidate_pino_field import load_scaleup_rows
from train_candidate_pino_grouped import split_groups, subset
from train_pino_safe_side_boundary import SEED, load_boundary_rows, load_candidate_rows, predict
from train_pino_safe_side_external_hardneg import split_external
from train_pino_aug_hardneg import load_geoattn_model
import torch


def main() -> None:
    checkpoint = ROOT / "checkpoints/pino_safe_side_external_hardneg_cal240_20260703/safe_side_external_hardneg_scale.pt"
    base = ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, head, _ = load_geoattn_model(device, base, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    for name, state in payload["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(state)
    if "attn_scale_head_state_dict" in payload:
        head.load_state_dict(payload["attn_scale_head_state_dict"])

    candidate_rows = load_candidate_rows()
    splits = split_groups()
    candidate_val = subset(candidate_rows, splits["selection_validation"])
    candidate_test = subset(candidate_rows, splits["heldout_test"])
    external_rows = load_scaleup_rows()
    for row in external_rows:
        row["source"] = "scaleup"
    external_cal, external_heldout, _ = split_external(external_rows, 240, SEED)
    near_sf = {0.8, 1.0, 1.2, 1.5, 2.0}
    hard_train = load_hardset_rows("train", near_sf)
    hard_val = load_hardset_rows("val", near_sf)
    hard_test = load_hardset_rows("test", near_sf)
    boundary = load_boundary_rows(600)
    train_pool = subset(candidate_rows, splits["train"]) + boundary + external_cal + hard_train
    cal_joint = candidate_val + external_cal

    pools = {
        "train": train_pool,
        "candidate_external_cal": cal_joint,
        "candidate_heldout": candidate_test,
        "external_heldout": external_heldout,
        "hardset_cal": hard_val,
        "hardset_test": hard_test,
    }
    pred = {name: predict(model, rows, device, 16) for name, rows in pools.items()}
    feat = {name: row_features(rows, pred[name]) for name, rows in pools.items()}

    gate_path = ROOT / "checkpoints/near_boundary_safety_decision_gate_20260704/gate.joblib"
    clf = joblib.load(gate_path)
    scores = {name: clf.predict_proba(feat[name])[:, 1] for name in pools}
    t_ce = choose_threshold(scores["candidate_external_cal"], cal_joint)
    t_hard = choose_threshold(scores["hardset_cal"], hard_val)
    record = {
        "classifier": str(gate_path),
        "thresholds": {
            "candidate_external": t_ce,
            "hardset": t_hard,
        },
        "candidate_heldout": evaluate(scores["candidate_heldout"], candidate_test, t_ce),
        "external_heldout": evaluate(scores["external_heldout"], external_heldout, t_ce),
        "hardset_test_near_boundary": evaluate(scores["hardset_test"], hard_test, t_hard),
    }
    record["joint_false_pass"] = sum(record[k]["auto_false_pass"] for k in ("candidate_heldout", "external_heldout", "hardset_test_near_boundary"))
    record["joint_auto_safe_true"] = sum(record[k]["auto_safe_true"] for k in ("candidate_heldout", "external_heldout", "hardset_test_near_boundary"))
    path = ROOT / "results/domain_separated_near_boundary_gate_20260704.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
