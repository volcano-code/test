#!/usr/bin/env python3
"""Train a safe/unsafe decision gate on corrected near-boundary FEM rows.

This experiment does not overwrite the authoritative PINO checkpoint.  It uses
the merged ShadowArt train-format index whose labels are expressed under the
50MPa decision yield used by the current screening scripts.
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
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
DEFAULT_MERGED_INDEX = Path(
    "/data/xvxinyao/xvxinyao/total_project/results/"
    "shadowart_merged_detect_repair_v1/index_trainfmt.json"
)


def load_merged_rows(index_path: Path, split: str, sf_min: float, sf_max: float) -> list[dict]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = []
    for rec in index[split]:
        sf = float(rec["sf"])
        if sf < sf_min or sf > sf_max:
            continue
        path = Path(rec["path"])
        if not path.exists():
            continue
        payload = np.load(path, allow_pickle=True)
        rows.append({
            "source": "merged_near_boundary_fem",
            "case_id": int(rec.get("case_id", -1)),
            "density": payload["density"].astype(np.float32),
            "force": payload["force"].astype(np.float32),
            "log_vm": float(math.log10(max(YIELD_STRESS / max(sf, 1e-12), 1e-12))),
            "sf": sf,
            "path": str(path),
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
            float(log_sf - math.log10(TAU)),
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
            abs(float(log_sf) - math.log10(TAU)),
            1.0 if row.get("source") == "scaleup" else 0.0,
            1.0 if row.get("source") == "merged_near_boundary_fem" else 0.0,
        ])
    return np.asarray(feats, dtype=np.float32)


def labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([float(row["sf"]) >= TAU for row in rows], dtype=bool)


def evaluate(scores: np.ndarray, rows: list[dict], threshold: float) -> dict:
    truth = labels(rows)
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


def choose_zero_fp_threshold(scores: np.ndarray, rows: list[dict]) -> float:
    truth = labels(rows)
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


def build_model(kind: str):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=0.35, max_iter=3000))
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=260, learning_rate=0.025, l2_regularization=0.18, random_state=SEED)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=700,
            max_depth=9,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=900,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
    raise ValueError(kind)


def score_model(clf, x: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x)[:, 1]
    return clf.decision_function(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-index", type=Path, default=DEFAULT_MERGED_INDEX)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/pino_safe_accept90_cal300_m08_20260704/safe_side_external_hardneg_scale.pt")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--external-calibration", type=int, default=300)
    parser.add_argument("--sf-min", type=float, default=0.8)
    parser.add_argument("--sf-max", type=float, default=2.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "results/merged_near_boundary_decision_gate_20260705.json")
    parser.add_argument("--model-out", type=Path, default=ROOT / "checkpoints/merged_near_boundary_decision_gate_20260705/gate.joblib")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, head, _ = load_geoattn_model(device, args.base_ckpt, args.checkpoint)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "scale_branch_state_dict" in payload:
        for name, state in payload["scale_branch_state_dict"].items():
            getattr(model, name).load_state_dict(state)
    if "attn_scale_head_state_dict" in payload:
        head.load_state_dict(payload["attn_scale_head_state_dict"])

    candidate_rows = load_candidate_rows()
    splits = split_groups()
    candidate_train = subset(candidate_rows, splits["train"])
    candidate_val = subset(candidate_rows, splits["selection_validation"])
    candidate_test = subset(candidate_rows, splits["heldout_test"])

    external_rows = load_scaleup_rows()
    for row in external_rows:
        row["source"] = "scaleup"
    external_cal, external_heldout, external_split = split_external(external_rows, args.external_calibration)

    merged_train = load_merged_rows(args.merged_index, "train", args.sf_min, args.sf_max)
    merged_val = load_merged_rows(args.merged_index, "val", args.sf_min, args.sf_max)
    merged_test = load_merged_rows(args.merged_index, "test", args.sf_min, args.sf_max)

    pools = {
        "train": candidate_train + external_cal + merged_train,
        "calibration": candidate_val + external_cal + merged_val,
        "candidate_heldout": candidate_test,
        "external_heldout": external_heldout,
        "merged_near_boundary_test": merged_test,
    }
    predictions = {name: predict(model, rows, device, args.batch_size) for name, rows in pools.items()}
    features = {name: row_features(rows, predictions[name]) for name, rows in pools.items()}
    y_train = labels(pools["train"]).astype(int)

    records = {}
    best = None
    for kind in ("logistic", "hgb", "rf", "extra"):
        clf = build_model(kind)
        clf.fit(features["train"], y_train)
        cal_scores = score_model(clf, features["calibration"])
        threshold = choose_zero_fp_threshold(cal_scores, pools["calibration"])
        record = {"threshold": threshold}
        joint_false = 0
        joint_true = 0
        for split in ("candidate_heldout", "external_heldout", "merged_near_boundary_test"):
            metric = evaluate(score_model(clf, features[split]), pools[split], threshold)
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
        "schema": "merged_near_boundary_decision_gate_v1",
        "created": "2026-07-05",
        "checkpoint": str(args.checkpoint),
        "merged_index": str(args.merged_index),
        "label_yield_stress": YIELD_STRESS,
        "sf_range": [args.sf_min, args.sf_max],
        "training_rows": {
            "candidate_train": len(candidate_train),
            "external_calibration": len(external_cal),
            "merged_near_boundary_train": len(merged_train),
        },
        "calibration_rows": {
            "candidate_validation": len(candidate_val),
            "external_calibration": len(external_cal),
            "merged_near_boundary_validation": len(merged_val),
        },
        "sealed_test_rows": {
            "candidate_heldout": len(candidate_test),
            "external_heldout": len(external_heldout),
            "merged_near_boundary_test": len(merged_test),
        },
        "external_split": external_split,
        "candidates": records,
        "selected": {"kind": best[1], **best[3]},
        "promotion_gate": {
            "zero_false_pass_all_sealed": best[3]["joint_false_pass"] == 0,
            "candidate_auto_safe_at_least_90pct": best[3]["candidate_heldout"]["auto_safe_rate"] >= 0.90,
            "external_auto_safe_at_least_90pct": best[3]["external_heldout"]["auto_safe_rate"] >= 0.90,
            "merged_near_boundary_auto_safe_at_least_90pct": best[3]["merged_near_boundary_test"]["auto_safe_rate"] >= 0.90,
        },
        "model": str(args.model_out),
        "limitations": [
            "This is a decision-gate experiment; it does not overwrite the current best PINO checkpoint or main pipeline.",
            "Labels are voxel-FEM-derived safe/unsafe labels under the 50MPa screening yield.",
            "Promotion requires zero false pass on all sealed splits plus high auto-safe coverage.",
        ],
    }
    result["promotion_eligible_for_fem_replacement_claim"] = all(result["promotion_gate"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": result["selected"], "promotion_gate": result["promotion_gate"]}, indent=2))


if __name__ == "__main__":
    main()
