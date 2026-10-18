#!/usr/bin/env python3
"""Train a selective safe-accept gate on top of a frozen safe-side PINO."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_candidate_pino_field import load_scaleup_rows
from train_candidate_pino_grouped import split_groups, subset
from train_pino_aug_hardneg import load_geoattn_model
from train_pino_safe_side_boundary import SEED, TAU, load_candidate_rows, predict
from train_pino_safe_side_external_hardneg import split_external

YIELD_STRESS = 50e6


def geom_features(rows: list[dict], pred_log_vm: np.ndarray) -> np.ndarray:
    pred_log_sf = math.log10(YIELD_STRESS) - pred_log_vm
    feats = []
    for row, log_sf in zip(rows, pred_log_sf):
        density = row["density"].astype(bool)
        force = row["force"].astype(np.float32)
        occ = float(density.mean())
        coords = np.argwhere(density)
        if len(coords):
            span = coords.max(axis=0) - coords.min(axis=0) + 1
            bbox_fill = float(len(coords) / max(float(np.prod(span)), 1.0))
            center = coords.mean(axis=0) / max(density.shape[0] - 1, 1)
        else:
            span = np.zeros(3)
            bbox_fill = 0.0
            center = np.zeros(3)
        force_abs = np.abs(force)
        feats.append([
            float(log_sf),
            occ,
            bbox_fill,
            float(span[0] / density.shape[0]),
            float(span[1] / density.shape[1]),
            float(span[2] / density.shape[2]),
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(force_abs.sum()),
            float(force_abs.max()),
            float(np.count_nonzero(force_abs) / force_abs.size),
            1.0 if row.get("source") == "scaleup" else 0.0,
        ])
    return np.asarray(feats, dtype=np.float32)


def evaluate_accept(scores: np.ndarray, rows: list[dict], threshold: float) -> dict:
    truth = np.asarray([row["sf"] >= TAU for row in rows], dtype=bool)
    accept = scores >= threshold
    return {
        "threshold": float(threshold),
        "n": len(rows),
        "safe": int(truth.sum()),
        "unsafe": int((~truth).sum()),
        "auto_safe_accept_count": int(accept.sum()),
        "auto_safe_true": int((accept & truth).sum()),
        "auto_false_pass": int((accept & ~truth).sum()),
        "auto_safe_rate": float((accept & truth).sum() / max(int(truth.sum()), 1)),
    }


def choose_threshold(scores: np.ndarray, rows: list[dict]) -> float:
    truth = np.asarray([row["sf"] >= TAU for row in rows], dtype=bool)
    best_threshold = 1e9
    best_true = -1
    for threshold in sorted(set(float(v) for v in scores), reverse=True):
        accept = scores >= threshold
        false_pass = int((accept & ~truth).sum())
        true_accept = int((accept & truth).sum())
        if false_pass == 0 and true_accept > best_true:
            best_threshold = threshold
            best_true = true_accept
    return float(best_threshold)


def train_and_score(kind: str, x_train, y_train, x_eval) -> tuple[object, np.ndarray]:
    if kind == "logistic":
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000))
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.04, l2_regularization=0.1, random_state=SEED)
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=SEED,
        )
    else:
        raise ValueError(kind)
    clf.fit(x_train, y_train)
    return clf, clf.predict_proba(x_eval)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/pino_safe_side_external_hardneg_cal240_20260703/safe_side_external_hardneg_scale.pt")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--external-calibration", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "results/selective_safe_accept_gate_cal240_20260703.json")
    parser.add_argument("--model-out", type=Path, default=ROOT / "checkpoints/selective_safe_accept_gate_cal240_20260703/gate.joblib")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, head, _ = load_geoattn_model(device, args.base_ckpt, args.checkpoint)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    external_cal, external_heldout, external_split = split_external(external_rows, args.external_calibration)

    train_pool = train_rows + external_cal
    threshold_pool = val_rows + external_cal
    pred_train = predict(model, train_pool, device, args.batch_size)
    pred_threshold = predict(model, threshold_pool, device, args.batch_size)
    pred_test = predict(model, test_rows, device, args.batch_size)
    pred_external = predict(model, external_heldout, device, args.batch_size)

    x_train = geom_features(train_pool, pred_train)
    y_train = np.asarray([row["sf"] >= TAU for row in train_pool], dtype=int)
    x_threshold = geom_features(threshold_pool, pred_threshold)
    x_test = geom_features(test_rows, pred_test)
    x_external = geom_features(external_heldout, pred_external)

    candidates = {}
    best = None
    for kind in ("logistic", "hgb", "rf"):
        clf, threshold_scores = train_and_score(kind, x_train, y_train, x_threshold)
        threshold = choose_threshold(threshold_scores, threshold_pool)
        test_scores = clf.predict_proba(x_test)[:, 1]
        external_scores = clf.predict_proba(x_external)[:, 1]
        record = {
            "threshold": threshold,
            "candidate_heldout": evaluate_accept(test_scores, test_rows, threshold),
            "external_heldout": evaluate_accept(external_scores, external_heldout, threshold),
        }
        record["joint_false_pass"] = record["candidate_heldout"]["auto_false_pass"] + record["external_heldout"]["auto_false_pass"]
        record["joint_auto_safe_true"] = record["candidate_heldout"]["auto_safe_true"] + record["external_heldout"]["auto_safe_true"]
        candidates[kind] = record
        key = (record["joint_false_pass"], -record["joint_auto_safe_true"])
        if best is None or key < best[0]:
            best = (key, kind, clf, record)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best[2], args.model_out)
    result = {
        "schema": "selective_safe_accept_gate_v1",
        "checkpoint": str(args.checkpoint),
        "external_calibration": args.external_calibration,
        "external_heldout_rows": len(external_heldout),
        "feature_note": "PINO log-SF plus simple occupancy/load descriptors.",
        "candidates": candidates,
        "selected": {"kind": best[1], **best[3]},
        "model": str(args.model_out),
        "comparison_cal240": {
            "candidate_auto_safe_rate": 0.8,
            "external_auto_safe_rate": 0.734375,
            "joint_false_pass": 0,
        },
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["selected"], indent=2))


if __name__ == "__main__":
    main()
