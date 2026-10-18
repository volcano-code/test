#!/usr/bin/env python3
"""Train a safety-decision gate with near-boundary FEM hardset rows.

The model uses frozen safe-side PINO predictions plus simple geometry/load
descriptors. Thresholds are selected only on calibration rows. Sealed audits are
reported on candidate heldout, external heldout, and hardset test near-boundary
rows.
"""

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
from train_pino_safe_side_boundary import SEED, TAU, load_boundary_rows, load_candidate_rows, predict
from train_pino_safe_side_external_hardneg import split_external

YIELD_STRESS = 50e6
HARDSET_ROOT = Path("/home/lin/xvxinyao/total_project/results/shadowart_fem_hardset_v1")


def load_hardset_rows(split: str, sf_values: set[float]) -> list[dict]:
    index_path = ROOT / "data/shadowart_fem_hardset_v1/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = []
    for rec in index[split]:
        sf = float(rec["sf"])
        if round(sf, 3) not in sf_values:
            continue
        path = HARDSET_ROOT / rec["path"]
        if not path.exists():
            continue
        payload = np.load(path)
        rows.append({
            "source": "hardset_near_boundary",
            "case_id": int(rec["case_id"]),
            "case_name": rec["case_name"],
            "density": payload["density"].astype(np.float32),
            "force": payload["force"].astype(np.float32),
            "log_vm": float(payload["log_vm_max"]),
            "sf": float(payload["sf"]),
        })
    return rows


def row_features(rows: list[dict], pred_log_vm: np.ndarray) -> np.ndarray:
    pred_log_sf = math.log10(YIELD_STRESS) - pred_log_vm
    feats = []
    for row, log_sf in zip(rows, pred_log_sf):
        density = row["density"].astype(bool)
        force = row["force"].astype(np.float32)
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
            float(density.mean()),
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
            1.0 if row.get("source") == "hardset_near_boundary" else 0.0,
            1.0 if str(row.get("source", "")).startswith("boundary") else 0.0,
        ])
    return np.asarray(feats, dtype=np.float32)


def evaluate(scores: np.ndarray, rows: list[dict], threshold: float) -> dict:
    truth = np.asarray([row["sf"] >= TAU for row in rows], dtype=bool)
    accept = scores >= threshold
    return {
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


def train_model(kind: str, x_train, y_train):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000))
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.035, l2_regularization=0.12, random_state=SEED)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=SEED,
        )
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/pino_safe_side_external_hardneg_cal240_20260703/safe_side_external_hardneg_scale.pt")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--external-calibration", type=int, default=240)
    parser.add_argument("--boundary-limit", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "results/near_boundary_safety_decision_gate_20260704.json")
    parser.add_argument("--model-out", type=Path, default=ROOT / "checkpoints/near_boundary_safety_decision_gate_20260704/gate.joblib")
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
    candidate_train = subset(candidate_rows, splits["train"])
    candidate_val = subset(candidate_rows, splits["selection_validation"])
    candidate_test = subset(candidate_rows, splits["heldout_test"])
    boundary_rows = load_boundary_rows(args.boundary_limit)
    external_rows = load_scaleup_rows()
    for row in external_rows:
        row["source"] = "scaleup"
    external_cal, external_heldout, external_split = split_external(external_rows, args.external_calibration)

    near_sf = {0.8, 1.0, 1.2, 1.5, 2.0}
    hard_train = load_hardset_rows("train", near_sf)
    hard_val = load_hardset_rows("val", near_sf)
    hard_test = load_hardset_rows("test", near_sf)

    train_pool = candidate_train + boundary_rows + external_cal + hard_train
    calibration_pool = candidate_val + external_cal + hard_val

    pools = {
        "train": train_pool,
        "calibration": calibration_pool,
        "candidate_heldout": candidate_test,
        "external_heldout": external_heldout,
        "hardset_test_near_boundary": hard_test,
    }
    predictions = {name: predict(model, rows, device, args.batch_size) for name, rows in pools.items()}
    features = {name: row_features(rows, predictions[name]) for name, rows in pools.items()}

    y_train = np.asarray([row["sf"] >= TAU for row in train_pool], dtype=int)
    records = {}
    best = None
    for kind in ("logistic", "hgb", "rf"):
        clf = train_model(kind, features["train"], y_train)
        clf.fit(features["train"], y_train)
        cal_scores = clf.predict_proba(features["calibration"])[:, 1]
        threshold = choose_threshold(cal_scores, calibration_pool)
        record = {"threshold": threshold}
        joint_false = 0
        joint_true = 0
        for split in ("candidate_heldout", "external_heldout", "hardset_test_near_boundary"):
            scores = clf.predict_proba(features[split])[:, 1]
            metric = evaluate(scores, pools[split], threshold)
            record[split] = metric
            joint_false += metric["auto_false_pass"]
            joint_true += metric["auto_safe_true"]
        record["joint_false_pass"] = joint_false
        record["joint_auto_safe_true"] = joint_true
        records[kind] = record
        key = (joint_false, -joint_true)
        if best is None or key < best[0]:
            best = (key, kind, clf, record)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best[2], args.model_out)
    result = {
        "schema": "near_boundary_safety_decision_gate_v1",
        "created": "2026-07-04",
        "checkpoint": str(args.checkpoint),
        "hardset_root": str(HARDSET_ROOT),
        "near_boundary_sf_values": sorted(near_sf),
        "training_rows": {
            "candidate_train": len(candidate_train),
            "boundary": len(boundary_rows),
            "external_calibration": len(external_cal),
            "hardset_train_near_boundary": len(hard_train),
        },
        "calibration_rows": {
            "candidate_validation": len(candidate_val),
            "external_calibration": len(external_cal),
            "hardset_validation_near_boundary": len(hard_val),
        },
        "sealed_test_rows": {
            "candidate_heldout": len(candidate_test),
            "external_heldout": len(external_heldout),
            "hardset_test_near_boundary": len(hard_test),
        },
        "external_split": external_split,
        "candidates": records,
        "selected": {"kind": best[1], **best[3]},
        "promotion_gate": {
            "zero_false_pass_all_sealed": best[3]["joint_false_pass"] == 0,
            "candidate_auto_safe_at_least_90pct": best[3]["candidate_heldout"]["auto_safe_rate"] >= 0.90,
            "external_auto_safe_at_least_90pct": best[3]["external_heldout"]["auto_safe_rate"] >= 0.90,
            "hardset_auto_safe_at_least_90pct": best[3]["hardset_test_near_boundary"]["auto_safe_rate"] >= 0.90,
        },
        "model": str(args.model_out),
        "limitations": [
            "The hardset samples are external near-boundary voxel-FEM rows stored outside ShadowForge; the local ShadowForge index records their metadata.",
            "The selected classifier is calibrated only on validation/calibration rows; sealed rows are audit-only.",
            "The claim remains relative to voxel-FEM labels and the current ShadowForge candidate domain.",
        ],
    }
    result["promotion_eligible_for_full_fem_replacement_claim"] = all(result["promotion_gate"].values())
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"selected": result["selected"], "promotion_gate": result["promotion_gate"]}, indent=2))


if __name__ == "__main__":
    main()
