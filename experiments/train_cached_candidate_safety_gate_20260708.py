#!/usr/bin/env python3
"""Train cached candidate-domain safe/unsafe gates.

This experiment avoids rebuilding voxel candidates. It uses the already
evaluated current candidate-domain JSON, trains small tabular decision models
on PINO/ACS/geometric features, and calibrates fail-closed thresholds on
candidate-type buckets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "results/candidate_selection_400case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"
TAU = 1.3
SEED = 20260708


def candidate_bucket(name: str) -> str:
    if name in {"visual_hull", "gap_filled_hull"}:
        return "hull"
    if name in {"carve_pf1", "carve_pf2", "carve_no_bridge", "min_printable_bridge_r1"}:
        return "early_carve"
    if name.startswith("carve_pf"):
        return "late_carve"
    if name == "watertight_export":
        return "export"
    return "other"


def feature_names(candidates: list[str], buckets: list[str]) -> list[str]:
    names = [
        "sf_pino_log",
        "acs_lower_log",
        "acs_upper_log",
        "acs_width_log",
        "acs_lower_margin",
        "pino_margin",
        "material_saved_vs_hull",
        "mean_iou",
        "mean_dice",
        "occupied_log",
        "component_count_6",
        "watertight",
        "raw_pino_safe",
        "acs_certified_safe",
        "acs_certified_unsafe",
        "branch_fixed",
    ]
    names += [f"cand_{c}" for c in candidates]
    names += [f"bucket_{b}" for b in buckets]
    return names


def build_cached_arrays(src: Path, out_npz: Path, out_meta: Path) -> dict:
    data = json.loads(src.read_text())
    candidate_names = sorted({r["candidate"] for c in data["cases"] for r in c["candidates"]})
    buckets = sorted({candidate_bucket(c) for c in candidate_names})
    cand_index = {c: i for i, c in enumerate(candidate_names)}
    bucket_index = {b: i for i, b in enumerate(buckets)}
    rows = []
    labels = []
    groups = []
    row_meta = []
    fem_times = []
    for case in data["cases"]:
        group = f"{case['branch']}:{int(case['case_id'])}"
        for r in case["candidates"]:
            sf_pino = max(float(r["sf_pino"]), 1e-9)
            acs_lower = max(float(r["acs_lower_sf"]), 1e-9)
            acs_upper = max(float(r["acs_upper_sf"]), 1e-9)
            base = [
                np.log(sf_pino),
                np.log(acs_lower),
                np.log(acs_upper),
                np.log(max(acs_upper - acs_lower, 1e-9)),
                acs_lower / TAU,
                sf_pino / TAU,
                float(r["material_saved_vs_hull"]),
                float(r["mean_iou"]),
                float(r["mean_dice"]),
                np.log1p(float(r["occupied_voxels"])),
                float(r["component_count_6"]),
                float(bool(r["watertight"])),
                float(bool(r["raw_pino_safe"])),
                float(bool(r["acs_certified_safe"])),
                float(bool(r["acs_certified_unsafe"])),
                float(case["branch"] == "fixed"),
            ]
            cand_oh = [0.0] * len(candidate_names)
            cand_oh[cand_index[r["candidate"]]] = 1.0
            bucket = candidate_bucket(r["candidate"])
            bucket_oh = [0.0] * len(buckets)
            bucket_oh[bucket_index[bucket]] = 1.0
            rows.append(base + cand_oh + bucket_oh)
            labels.append(float(bool(r["fem_safe"])))
            groups.append(group)
            fem_times.append(float(r.get("fem_elapsed_s", 0.0)))
            row_meta.append({
                "group": group,
                "branch": case["branch"],
                "case_id": int(case["case_id"]),
                "candidate": r["candidate"],
                "bucket": bucket,
                "sf_fem": float(r["sf_fem"]),
                "sf_pino": float(r["sf_pino"]),
                "acs_lower_sf": float(r["acs_lower_sf"]),
                "fem_safe": bool(r["fem_safe"]),
            })

    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    fem = np.asarray(fem_times, dtype=np.float32)
    group_arr = np.asarray(groups)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, x=x, y=y, groups=group_arr, fem_times=fem)
    meta = {
        "schema": "cached_candidate_safety_gate_v1",
        "source": str(src),
        "threshold_sf": TAU,
        "feature_names": feature_names(candidate_names, buckets),
        "candidate_names": candidate_names,
        "buckets": buckets,
        "rows": row_meta,
        "counts": {
            "n": int(len(y)),
            "safe": int(y.sum()),
            "unsafe": int((1 - y).sum()),
            "groups": int(len(set(groups))),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def split_groups(groups: np.ndarray, train_frac: float, cal_frac: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    n_train = int(round(len(unique) * train_frac))
    n_cal = int(round(len(unique) * cal_frac))
    train = set(unique[:n_train])
    cal = set(unique[n_train:n_train + n_cal])
    test = set(unique[n_train + n_cal:])
    return {
        "train": np.asarray([g in train for g in groups]),
        "calibration": np.asarray([g in cal for g in groups]),
        "sealed_test": np.asarray([g in test for g in groups]),
    }


class Gate(nn.Module):
    def __init__(self, dim: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def standardize(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0
    return (x - mean) / std, mean, std


def zero_fp_threshold(scores: np.ndarray, y: np.ndarray, margin: float) -> float:
    unsafe = scores[y < 0.5]
    if len(unsafe) == 0:
        return float(np.min(scores) - 1e-6 + margin)
    return float(np.max(unsafe) + 1e-6 + margin)


def bucket_thresholds(scores: np.ndarray, y: np.ndarray, buckets: list[str], margin: float) -> dict[str, float]:
    out = {}
    for b in sorted(set(buckets)):
        m = np.asarray([bb == b for bb in buckets], dtype=bool)
        out[b] = zero_fp_threshold(scores[m], y[m], margin)
    out["__global__"] = zero_fp_threshold(scores, y, margin)
    return out


def evaluate(scores: np.ndarray, y: np.ndarray, fem_times: np.ndarray, buckets: list[str], thresholds: dict[str, float]) -> dict:
    accepted = np.zeros_like(y, dtype=bool)
    for i, b in enumerate(buckets):
        accepted[i] = scores[i] >= thresholds.get(b, thresholds["__global__"])
    truth = y > 0.5
    fallback = ~accepted
    return {
        "n": int(len(y)),
        "safe": int(truth.sum()),
        "unsafe": int((~truth).sum()),
        "auto_safe": int(accepted.sum()),
        "auto_safe_true": int((accepted & truth).sum()),
        "false_safe": int((accepted & ~truth).sum()),
        "auto_safe_rate_over_safe": float((accepted & truth).sum() / max(int(truth.sum()), 1)),
        "autonomous_rate": float(accepted.mean()) if len(y) else 0.0,
        "full_fem_time_s": float(fem_times.sum()),
        "fallback_fem_time_s": float(fem_times[fallback].sum()),
        "fem_time_avoided_rate": float(1.0 - fem_times[fallback].sum() / max(fem_times.sum(), 1e-12)),
    }


def train_one(x: np.ndarray, y: np.ndarray, masks: dict[str, np.ndarray], seed: int, args) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Gate(x.shape[1], args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_train = torch.as_tensor(x[masks["train"]], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y[masks["train"]], dtype=torch.float32, device=device)
    unsafe_weight = torch.where(y_train > 0.5, torch.ones_like(y_train) * args.safe_weight, torch.ones_like(y_train) * args.unsafe_weight)
    n = len(y_train)
    history = []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(n)
        total = 0.0
        model.train()
        for start in range(0, n, args.batch_size):
            idx = torch.as_tensor(order[start:start + args.batch_size], dtype=torch.long, device=device)
            logits = model(x_train[idx])
            target = y_train[idx]
            weight = unsafe_weight[idx]
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            prob = torch.sigmoid(logits)
            false_safe_penalty = (1.0 - target) * F.relu(prob - args.unsafe_prob_cap).pow(2)
            safe_miss_penalty = target * F.relu(args.safe_prob_floor - prob).pow(2)
            loss = torch.mean(weight * bce + args.false_safe_weight * false_safe_penalty + args.safe_miss_weight * safe_miss_penalty)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 5, 1) == 0:
            history.append({"epoch": epoch, "train_loss": total / max(n, 1)})
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.as_tensor(x, dtype=torch.float32, device=device))).detach().cpu().numpy()
    return scores, {"seed": seed, "history": history}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--cache", type=Path, default=ROOT / "results/pino_decision_cache_20260708/candidate_domain_cache.npz")
    ap.add_argument("--cache-meta", type=Path, default=ROOT / "results/pino_decision_cache_20260708/candidate_domain_cache_meta.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/cached_candidate_safety_gate_20260708.json")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.60)
    ap.add_argument("--cal-frac", type=float, default=0.20)
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--safe-weight", type=float, default=1.0)
    ap.add_argument("--unsafe-weight", type=float, default=3.0)
    ap.add_argument("--false-safe-weight", type=float, default=35.0)
    ap.add_argument("--safe-miss-weight", type=float, default=1.5)
    ap.add_argument("--unsafe-prob-cap", type=float, default=0.05)
    ap.add_argument("--safe-prob-floor", type=float, default=0.65)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.cache.exists() or not args.cache_meta.exists():
        meta = build_cached_arrays(args.src, args.cache, args.cache_meta)
    else:
        meta = json.loads(args.cache_meta.read_text())
    cached = np.load(args.cache, allow_pickle=True)
    x_raw = cached["x"].astype(np.float32)
    y = cached["y"].astype(np.float32)
    groups = cached["groups"]
    fem_times = cached["fem_times"].astype(np.float32)
    row_buckets = [row["bucket"] for row in meta["rows"]]
    masks = split_groups(groups, args.train_frac, args.cal_frac, SEED)
    x, mean, std = standardize(x_raw, masks["train"])

    model_records = []
    scores_list = []
    for i in range(args.ensemble):
        scores, rec = train_one(x, y, masks, SEED + i * 17, args)
        scores_list.append(scores)
        model_records.append(rec)
        print(json.dumps({"model": i, "last": rec["history"][-1]}, ensure_ascii=False), flush=True)

    scores_arr = np.vstack(scores_list)
    ensemble_scores = scores_arr.min(axis=0)
    cal_idx = masks["calibration"]
    test_idx = masks["sealed_test"]
    train_idx = masks["train"]
    thresholds = bucket_thresholds(
        ensemble_scores[cal_idx],
        y[cal_idx],
        [b for b, m in zip(row_buckets, cal_idx) if m],
        args.margin,
    )
    result = {
        "schema": "cached_candidate_safety_gate_v1",
        "source": str(args.src),
        "cache": str(args.cache),
        "cache_meta": str(args.cache_meta),
        "config": vars(args) | {"src": str(args.src), "cache": str(args.cache), "cache_meta": str(args.cache_meta), "out": str(args.out)},
        "cache_counts": meta["counts"],
        "split_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "split_safe_unsafe": {
            k: {"safe": int(y[v].sum()), "unsafe": int((1 - y[v]).sum())}
            for k, v in masks.items()
        },
        "thresholds": thresholds,
        "train": evaluate(ensemble_scores[train_idx], y[train_idx], fem_times[train_idx], [b for b, m in zip(row_buckets, train_idx) if m], thresholds),
        "calibration": evaluate(ensemble_scores[cal_idx], y[cal_idx], fem_times[cal_idx], [b for b, m in zip(row_buckets, cal_idx) if m], thresholds),
        "sealed_test": evaluate(ensemble_scores[test_idx], y[test_idx], fem_times[test_idx], [b for b, m in zip(row_buckets, test_idx) if m], thresholds),
        "promotion_gate": {},
        "models": model_records,
        "notes": [
            "This is a cached candidate-domain decision gate using existing PINO/ACS/geometric features.",
            "It is not a stress-field predictor and does not overwrite the deployed PINO checkpoint.",
            "Automatic safety is calibrated by bucket on calibration rows and audited on sealed test rows.",
        ],
    }
    result["promotion_gate"] = {
        "zero_false_safe_on_calibration": result["calibration"]["false_safe"] == 0,
        "zero_false_safe_on_sealed_test": result["sealed_test"]["false_safe"] == 0,
        "sealed_test_auto_safe_at_least_50pct": result["sealed_test"]["auto_safe_rate_over_safe"] >= 0.50,
        "sealed_test_auto_safe_at_least_85pct": result["sealed_test"]["auto_safe_rate_over_safe"] >= 0.85,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out),
        "calibration": result["calibration"],
        "sealed_test": result["sealed_test"],
        "promotion_gate": result["promotion_gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
