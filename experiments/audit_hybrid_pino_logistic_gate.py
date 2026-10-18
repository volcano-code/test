#!/usr/bin/env python3
"""Audit a two-threshold hybrid: frozen PINO log-SF and logistic safe gate."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_candidate_pino_field import load_scaleup_rows
from train_candidate_pino_grouped import split_groups, subset
from train_pino_aug_hardneg import load_geoattn_model
from train_pino_safe_side_boundary import SEED, TAU, load_candidate_rows, predict
from train_pino_safe_side_external_hardneg import split_external
from train_selective_safe_accept_gate import YIELD_STRESS, evaluate_accept, geom_features, train_and_score


def eval_two(scores, log_sf, rows, cls_t, pino_t):
    truth = np.asarray([row["sf"] >= TAU for row in rows], dtype=bool)
    accept = (scores >= cls_t) & (log_sf >= pino_t)
    return {
        "classifier_threshold": float(cls_t),
        "pino_threshold": float(pino_t),
        "n": len(rows),
        "safe": int(truth.sum()),
        "unsafe": int((~truth).sum()),
        "auto_safe_accept_count": int(accept.sum()),
        "auto_safe_true": int((accept & truth).sum()),
        "auto_false_pass": int((accept & ~truth).sum()),
        "auto_safe_rate": float((accept & truth).sum() / max(int(truth.sum()), 1)),
    }


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
    train_rows = subset(candidate_rows, splits["train"])
    val_rows = subset(candidate_rows, splits["selection_validation"])
    test_rows = subset(candidate_rows, splits["heldout_test"])
    external_rows = load_scaleup_rows()
    for row in external_rows:
        row["source"] = "scaleup"
    external_cal, external_heldout, _ = split_external(external_rows, 240, SEED)

    train_pool = train_rows + external_cal
    threshold_pool = val_rows + external_cal
    pred_train = predict(model, train_pool, device, 16)
    pred_threshold = predict(model, threshold_pool, device, 16)
    pred_test = predict(model, test_rows, device, 16)
    pred_external = predict(model, external_heldout, device, 16)
    log_sf_threshold = math.log10(YIELD_STRESS) - pred_threshold
    log_sf_test = math.log10(YIELD_STRESS) - pred_test
    log_sf_external = math.log10(YIELD_STRESS) - pred_external

    x_train = geom_features(train_pool, pred_train)
    y_train = np.asarray([row["sf"] >= TAU for row in train_pool], dtype=int)
    x_threshold = geom_features(threshold_pool, pred_threshold)
    x_test = geom_features(test_rows, pred_test)
    x_external = geom_features(external_heldout, pred_external)
    clf, threshold_scores = train_and_score("logistic", x_train, y_train, x_threshold)
    test_scores = clf.predict_proba(x_test)[:, 1]
    external_scores = clf.predict_proba(x_external)[:, 1]

    cls_grid = sorted(set(float(v) for v in threshold_scores), reverse=True)
    pino_grid = sorted(set(float(v) for v in log_sf_threshold), reverse=True)
    best = None
    for cls_t in cls_grid:
        for pino_t in pino_grid:
            threshold_metric = eval_two(threshold_scores, log_sf_threshold, threshold_pool, cls_t, pino_t)
            if threshold_metric["auto_false_pass"] != 0:
                continue
            candidate = eval_two(test_scores, log_sf_test, test_rows, cls_t, pino_t)
            external = eval_two(external_scores, log_sf_external, external_heldout, cls_t, pino_t)
            record = {
                "threshold_pool": threshold_metric,
                "candidate_heldout": candidate,
                "external_heldout": external,
                "joint_false_pass": candidate["auto_false_pass"] + external["auto_false_pass"],
                "joint_auto_safe_true": candidate["auto_safe_true"] + external["auto_safe_true"],
            }
            key = (record["joint_false_pass"], -record["joint_auto_safe_true"])
            if best is None or key < best[0]:
                best = (key, record)
    out = {
        "schema": "hybrid_pino_logistic_gate_v1",
        "selected": best[1],
        "comparison_cal240": {
            "candidate_auto_safe_rate": 0.8,
            "external_auto_safe_rate": 0.734375,
            "joint_false_pass": 0,
        },
    }
    path = ROOT / "results/hybrid_pino_logistic_gate_cal240_20260703.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    joblib.dump(clf, ROOT / "checkpoints/selective_safe_accept_gate_cal240_20260703/hybrid_logistic.joblib")
    print(json.dumps(out["selected"], indent=2))


if __name__ == "__main__":
    main()
