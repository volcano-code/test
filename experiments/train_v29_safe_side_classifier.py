#!/usr/bin/env python3
"""Train a safe-side selector on frozen v29 ensemble features.

This is a diagnostic experiment. It trains only a lightweight classifier on
cached v29 predictions and scalar candidate features, then chooses an acceptance
threshold on validation groups with zero false-safe decisions. Held-out candidate
groups and the external scaleup set are audit-only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_candidate_pino_field import load_field_map

YIELD_STRESS = 50e6
SF_TARGET = 1.3
SEED = 20260703


def load_candidate_rows() -> list[dict]:
    rows = []
    field_map = load_field_map()
    for branch_id, path in enumerate((ROOT / "data/aug.npz", ROOT / "data/aug_fixed.npz")):
        payload = np.load(path)
        branch = "free" if branch_id == 0 else "fixed"
        for index in range(len(payload["dens"])):
            group = index // 16
            geometry = (index % 16) // 4
            record = field_map.get((branch_id, group, geometry))
            if record is None or not record["valid"]:
                continue
            density = payload["dens"][index].astype(np.float32)
            rows.append({
                "branch": branch,
                "branch_id": branch_id,
                "group": group,
                "geometry": geometry,
                "volume_fraction": float(density.mean()),
                "multiplier": float(payload["mult"][index]),
                "sf": float(payload["sf"][index]),
            })
    return rows


def load_external_rows() -> list[dict]:
    payload = np.load(ROOT / "data/scaleup_test.npz")
    rows = []
    for index in range(len(payload["dens"])):
        density = payload["dens"][index].astype(np.float32)
        rows.append({
            "branch": "external",
            "branch_id": 0,
            "group": -1,
            "geometry": -1,
            "volume_fraction": float(density.mean()),
            "multiplier": float(payload["mult"][index]) if "mult" in payload.files else 1.0,
            "sf": float(payload["sf"][index]),
        })
    return rows


def split_groups() -> dict[str, list[int]]:
    groups = np.arange(40)
    np.random.default_rng(SEED).shuffle(groups)
    return {
        "train": sorted(int(value) for value in groups[:20]),
        "validation": sorted(int(value) for value in groups[20:28]),
        "heldout": sorted(int(value) for value in groups[28:]),
    }


def indices_for(rows: list[dict], groups: list[int]) -> np.ndarray:
    allowed = set(groups)
    return np.asarray([index for index, row in enumerate(rows) if row["group"] in allowed], dtype=int)


def prediction_cache():
    cache = np.load(ROOT / "results/v29_ensemble_candidate_predictions_20260703.npz")
    candidate_members = np.stack([cache[f"candidate_{name}"] for name in ("v29a", "v29b", "v29c")])
    external_members = np.stack([cache[f"external_{name}"] for name in ("v29a", "v29b", "v29c")])
    return {
        "candidate_mean": candidate_members.mean(axis=0),
        "candidate_std": candidate_members.std(axis=0),
        "candidate_disagreement": np.abs(cache["candidate_v27"] - cache["candidate_v28"]),
        "external_mean": external_members.mean(axis=0),
        "external_std": external_members.std(axis=0),
        "external_disagreement": np.abs(cache["external_v27"] - cache["external_v28"]),
    }


def features(rows: list[dict], mean_log_vm: np.ndarray, std_log_vm: np.ndarray, disagreement: np.ndarray) -> np.ndarray:
    pred_log_sf = math.log10(YIELD_STRESS) - mean_log_vm
    margin = pred_log_sf - math.log10(SF_TARGET)
    volume = np.asarray([row["volume_fraction"] for row in rows], dtype=float)
    multiplier = np.asarray([row["multiplier"] for row in rows], dtype=float)
    branch = np.asarray([row["branch_id"] for row in rows], dtype=float)
    return np.stack([
        pred_log_sf,
        margin,
        mean_log_vm,
        std_log_vm,
        disagreement,
        volume,
        multiplier,
        branch,
        margin * volume,
        margin * disagreement,
    ], axis=1).astype(np.float32)


def labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["sf"] >= SF_TARGET for row in rows], dtype=bool)


def train_model(x_train, y_train, x_val, y_val):
    torch.manual_seed(SEED)
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    xt = torch.as_tensor((x_train - x_mean) / x_std, dtype=torch.float32)
    yt = torch.as_tensor(y_train.astype(np.float32), dtype=torch.float32).unsqueeze(1)
    xv = torch.as_tensor((x_val - x_mean) / x_std, dtype=torch.float32)
    yv = torch.as_tensor(y_val.astype(np.float32), dtype=torch.float32).unsqueeze(1)
    model = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], 24),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.05),
        torch.nn.Linear(24, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    best_state = None
    best_key = None
    history = []
    for epoch in range(1, 1201):
        model.train()
        logits = model(xt)
        bce = F.binary_cross_entropy_with_logits(logits, yt, reduction="none")
        weights = torch.where(yt > 0.5, torch.tensor(1.0), torch.tensor(4.0))
        prob = torch.sigmoid(logits)
        false_safe_penalty = ((1.0 - yt) * prob.square()).mean() * 8.0
        loss = (bce * weights).mean() + false_safe_penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_prob = torch.sigmoid(model(xv)).squeeze(1).numpy()
            metric = threshold_metrics(val_prob, y_val)
            key = (metric["auto_false_pass"], -metric["auto_safe_true"], -metric["auto_safe_rate"])
            history.append({"epoch": epoch, "loss": float(loss.detach()), "validation": metric})
            if best_key is None or key < best_key:
                best_key = key
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, x_mean, x_std, history


def threshold_metrics(prob: np.ndarray, y: np.ndarray, threshold: float | None = None) -> dict:
    if threshold is None:
        candidates = sorted(set(float(value) for value in prob), reverse=True)
        threshold = 1.0
        best_accept = -1
        for value in candidates:
            accept = prob >= value
            false_safe = int((accept & ~y).sum())
            true_accept = int((accept & y).sum())
            if false_safe == 0 and true_accept > best_accept:
                threshold = value
                best_accept = true_accept
    accept = prob >= threshold
    false_safe = int((accept & ~y).sum())
    true_accept = int((accept & y).sum())
    safe_total = int(y.sum())
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "safe": safe_total,
        "unsafe": int((~y).sum()),
        "auto_safe_accept_count": int(accept.sum()),
        "auto_safe_true": true_accept,
        "auto_false_pass": false_safe,
        "auto_safe_rate": float(true_accept / max(safe_total, 1)),
    }


def conservative_threshold(prob: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    unsafe = prob[~y]
    if len(unsafe) == 0:
        return 1.0
    return float(min(1.0, unsafe.max() + eps))


def predict_prob(model, x: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor((x - x_mean) / x_std, dtype=torch.float32)
        return torch.sigmoid(model(xt)).squeeze(1).numpy()


def main() -> None:
    candidate_rows = load_candidate_rows()
    external_rows = load_external_rows()
    cache = prediction_cache()
    x_candidate = features(
        candidate_rows, cache["candidate_mean"], cache["candidate_std"], cache["candidate_disagreement"]
    )
    x_external = features(
        external_rows, cache["external_mean"], cache["external_std"], cache["external_disagreement"]
    )
    y_candidate = labels(candidate_rows)
    y_external = labels(external_rows)
    splits = split_groups()
    train_idx = indices_for(candidate_rows, splits["train"])
    val_idx = indices_for(candidate_rows, splits["validation"])
    heldout_idx = indices_for(candidate_rows, splits["heldout"])

    model, x_mean, x_std, history = train_model(
        x_candidate[train_idx], y_candidate[train_idx],
        x_candidate[val_idx], y_candidate[val_idx],
    )
    val_prob = predict_prob(model, x_candidate[val_idx], x_mean, x_std)
    validation = threshold_metrics(val_prob, y_candidate[val_idx])
    threshold = validation["threshold"]
    heldout = threshold_metrics(
        predict_prob(model, x_candidate[heldout_idx], x_mean, x_std), y_candidate[heldout_idx], threshold
    )
    external = threshold_metrics(
        predict_prob(model, x_external, x_mean, x_std), y_external, threshold
    )
    trainval_idx = np.concatenate([train_idx, val_idx])
    trainval_prob = predict_prob(model, x_candidate[trainval_idx], x_mean, x_std)
    strict_threshold = conservative_threshold(trainval_prob, y_candidate[trainval_idx])
    strict_validation = threshold_metrics(val_prob, y_candidate[val_idx], strict_threshold)
    strict_heldout = threshold_metrics(
        predict_prob(model, x_candidate[heldout_idx], x_mean, x_std), y_candidate[heldout_idx], strict_threshold
    )
    strict_external = threshold_metrics(
        predict_prob(model, x_external, x_mean, x_std), y_external, strict_threshold
    )
    promotion = {
        "candidate_zero_false_pass": heldout["auto_false_pass"] == 0,
        "external_zero_false_pass": external["auto_false_pass"] == 0,
        "candidate_auto_safe_at_least_50pct": heldout["auto_safe_rate"] >= 0.50,
        "external_auto_safe_at_least_50pct": external["auto_safe_rate"] >= 0.50,
    }
    result = {
        "schema": "v29_safe_side_classifier_v1",
        "created": "2026-07-03",
        "splits": splits,
        "features": [
            "pred_log_sf", "margin", "mean_log_vm", "std_log_vm", "disagreement",
            "volume_fraction", "multiplier", "branch_id", "margin_x_volume",
            "margin_x_disagreement",
        ],
        "selection_rule": "threshold chosen on validation for zero false-safe and maximum true safe accepts",
        "validation": validation,
        "candidate_heldout": heldout,
        "external_scaleup": external,
        "strict_trainval_threshold": {
            "threshold": strict_threshold,
            "validation": strict_validation,
            "candidate_heldout": strict_heldout,
            "external_scaleup": strict_external,
        },
        "promotion_gate": promotion,
        "promotion_eligible_for_fem_replacement_claim": all(promotion.values()),
        "history_tail": history[-10:],
        "limitations": [
            "Lightweight classifier uses cached v29 predictions; no voxel field is retrained.",
            "External branch_id is set to free-compatible default because scaleup data has no free/fixed branch.",
            "Labels are voxel-FEM references, not physical measurements.",
        ],
    }
    output = ROOT / "results/v29_safe_side_classifier_20260703.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "validation": validation,
        "candidate_heldout": heldout,
        "external_scaleup": external,
        "strict_trainval_threshold": result["strict_trainval_threshold"],
        "promotion_eligible_for_fem_replacement_claim": result["promotion_eligible_for_fem_replacement_claim"],
    }, indent=2))


if __name__ == "__main__":
    main()
