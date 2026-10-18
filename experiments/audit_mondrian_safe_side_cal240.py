#!/usr/bin/env python3
"""Audit simple Mondrian safe-side thresholds for the cal240 PINO screen.

The policy may choose different PINO log-SF thresholds for predeclared feature
groups, but thresholds are fitted only on the candidate-validation plus external
calibration pool. Candidate/external heldout rows remain audit-only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_candidate_pino_field import load_scaleup_rows
from train_candidate_pino_grouped import split_groups, subset
from train_pino_aug_hardneg import load_geoattn_model
from train_pino_safe_side_boundary import SEED, TAU, load_candidate_rows, predict
from train_pino_safe_side_external_hardneg import split_external

YIELD_STRESS = 50e6


def row_features(rows: list[dict], pred_log_vm: np.ndarray) -> dict[str, np.ndarray]:
    pred_log_sf = math.log10(YIELD_STRESS) - pred_log_vm
    out: dict[str, list[float]] = {
        "pred_log_sf": [],
        "occ": [],
        "bbox_fill": [],
        "span_z": [],
        "force_sum": [],
    }
    for row, log_sf in zip(rows, pred_log_sf):
        density = row["density"].astype(bool)
        force = np.abs(row["force"].astype(np.float32))
        coords = np.argwhere(density)
        if len(coords):
            span = coords.max(axis=0) - coords.min(axis=0) + 1
            bbox_fill = float(len(coords) / max(float(np.prod(span)), 1.0))
        else:
            span = np.zeros(3)
            bbox_fill = 0.0
        out["pred_log_sf"].append(float(log_sf))
        out["occ"].append(float(density.mean()))
        out["bbox_fill"].append(bbox_fill)
        out["span_z"].append(float(span[2] / density.shape[2]))
        out["force_sum"].append(float(force.sum()))
    return {key: np.asarray(value, dtype=np.float32) for key, value in out.items()}


def metric(log_sf: np.ndarray, rows: list[dict], accept: np.ndarray) -> dict:
    truth = np.asarray([row["sf"] >= TAU for row in rows], dtype=bool)
    return {
        "n": len(rows),
        "safe": int(truth.sum()),
        "unsafe": int((~truth).sum()),
        "auto_safe_accept_count": int(accept.sum()),
        "auto_safe_true": int((accept & truth).sum()),
        "auto_false_pass": int((accept & ~truth).sum()),
        "auto_safe_rate": float((accept & truth).sum() / max(int(truth.sum()), 1)),
    }


def choose_threshold(log_sf: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    values = log_sf[mask]
    labels = truth[mask]
    best_threshold = 1e9
    best_true = -1
    for threshold in sorted(set(float(v) for v in values), reverse=True):
        accept = values >= threshold
        false_pass = int((accept & ~labels).sum())
        true_accept = int((accept & labels).sum())
        if false_pass == 0 and true_accept > best_true:
            best_threshold = threshold
            best_true = true_accept
    return float(best_threshold)


def fit_partition(
    name: str,
    train_log_sf: np.ndarray,
    train_truth: np.ndarray,
    train_feature: np.ndarray,
    test_log_sf: np.ndarray,
    test_feature: np.ndarray,
    external_log_sf: np.ndarray,
    external_feature: np.ndarray,
    quantiles: tuple[float, ...],
) -> dict:
    cuts = np.quantile(train_feature, quantiles)
    cuts = np.unique(cuts)
    train_bins = np.digitize(train_feature, cuts, right=False)
    thresholds = {}
    for group in sorted(set(int(v) for v in train_bins)):
        mask = train_bins == group
        thresholds[str(group)] = choose_threshold(train_log_sf, train_truth, mask)

    def accept(log_sf: np.ndarray, feat: np.ndarray) -> np.ndarray:
        bins = np.digitize(feat, cuts, right=False)
        out = np.zeros(len(log_sf), dtype=bool)
        for group in sorted(set(int(v) for v in bins)):
            out[bins == group] = log_sf[bins == group] >= thresholds.get(str(group), 1e9)
        return out

    return {
        "name": name,
        "cuts": [float(v) for v in cuts],
        "thresholds": thresholds,
        "threshold_pool_accept": accept(train_log_sf, train_feature),
        "candidate_accept": accept(test_log_sf, test_feature),
        "external_accept": accept(external_log_sf, external_feature),
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
    val_rows = subset(candidate_rows, splits["selection_validation"])
    test_rows = subset(candidate_rows, splits["heldout_test"])
    external_rows = load_scaleup_rows()
    for row in external_rows:
        row["source"] = "scaleup"
    external_cal, external_heldout, _ = split_external(external_rows, 240, SEED)
    threshold_rows = val_rows + external_cal

    pred_threshold = predict(model, threshold_rows, device, 16)
    pred_test = predict(model, test_rows, device, 16)
    pred_external = predict(model, external_heldout, device, 16)
    log_sf_threshold = math.log10(YIELD_STRESS) - pred_threshold
    log_sf_test = math.log10(YIELD_STRESS) - pred_test
    log_sf_external = math.log10(YIELD_STRESS) - pred_external
    truth_threshold = np.asarray([row["sf"] >= TAU for row in threshold_rows], dtype=bool)

    feat_threshold = row_features(threshold_rows, pred_threshold)
    feat_test = row_features(test_rows, pred_test)
    feat_external = row_features(external_heldout, pred_external)

    global_t = choose_threshold(log_sf_threshold, truth_threshold, np.ones(len(threshold_rows), dtype=bool))
    baseline = {
        "threshold": global_t,
        "threshold_pool": metric(log_sf_threshold, threshold_rows, log_sf_threshold >= global_t),
        "candidate_heldout": metric(log_sf_test, test_rows, log_sf_test >= global_t),
        "external_heldout": metric(log_sf_external, external_heldout, log_sf_external >= global_t),
    }

    candidates = []
    for feat_name in ("occ", "bbox_fill", "span_z", "force_sum", "pred_log_sf"):
        for quantiles in ((0.5,), (0.33, 0.66), (0.25, 0.5, 0.75)):
            part = fit_partition(
                f"{feat_name}_q{'_'.join(str(q) for q in quantiles)}",
                log_sf_threshold,
                truth_threshold,
                feat_threshold[feat_name],
                log_sf_test,
                feat_test[feat_name],
                log_sf_external,
                feat_external[feat_name],
                quantiles,
            )
            rec = {
                "name": part["name"],
                "cuts": part["cuts"],
                "thresholds": part["thresholds"],
                "threshold_pool": metric(log_sf_threshold, threshold_rows, part["threshold_pool_accept"]),
                "candidate_heldout": metric(log_sf_test, test_rows, part["candidate_accept"]),
                "external_heldout": metric(log_sf_external, external_heldout, part["external_accept"]),
            }
            rec["joint_false_pass"] = rec["candidate_heldout"]["auto_false_pass"] + rec["external_heldout"]["auto_false_pass"]
            rec["joint_auto_safe_true"] = rec["candidate_heldout"]["auto_safe_true"] + rec["external_heldout"]["auto_safe_true"]
            candidates.append(rec)

    zero_fp = [row for row in candidates if row["joint_false_pass"] == 0]
    best = max(zero_fp, key=lambda row: row["joint_auto_safe_true"], default=None)
    out = {
        "schema": "mondrian_safe_side_cal240_audit_v1",
        "checkpoint": str(checkpoint),
        "selection_rule": "thresholds fit on candidate validation plus external calibration only; heldout is audit-only",
        "baseline_global": baseline,
        "best_zero_false_pass": best,
        "all_candidates": candidates,
    }
    path = ROOT / "results/mondrian_safe_side_cal240_audit_20260703.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "best_zero_false_pass": best}, indent=2))


if __name__ == "__main__":
    main()
