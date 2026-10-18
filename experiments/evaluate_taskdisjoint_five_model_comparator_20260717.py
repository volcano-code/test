#!/usr/bin/env python3
"""Frozen calibration/sealed evaluation for the task-disjoint five-model run."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.evaluate_strict_pino_acs_v5_20260713 import (
    TAU, finite_quantile, load_model, make_dataset, metrics, predict_log_sf,
)
from experiments.evaluate_strict_pino_dualside_acs_20260717 import decisions
from experiments.train_scalar_structural_baseline_v7 import (
    GeometryMLP, ScalarCNN3D, geometry_features,
)
from structural.training.pretrain_pino_v3 import FEMDatasetV3, collate_v3


INDEX = ROOT / "data/taskgrouped_strict_pino_v13_union_20260716/index.json"
YIELD_STRESS = 50.0e6


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def auc(score: np.ndarray, positive: np.ndarray) -> float | None:
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def scalar_predictions(kind: str, checkpoint: Path, rows: list[dict],
                       device: torch.device) -> tuple[np.ndarray, float, int]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = GeometryMLP() if kind == "geometry_mlp" else ScalarCNN3D()
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device).eval()
    ds = make_dataset(INDEX, rows, 48)
    loader = DataLoader(ds, batch_size=8 if kind == "geometry_mlp" else 4,
                        shuffle=False, num_workers=0, pin_memory=device.type == "cuda",
                        collate_fn=collate_v3)
    # One untimed batch removes CUDA initialization and allocator startup from
    # the per-sample latency shared by all five models.
    first = next(iter(loader))
    with torch.no_grad():
        x = first["x"].to(device)
        _ = model(geometry_features(x) if kind == "geometry_mlp" else x)
    synchronize(device)
    del x, first, _
    out = []
    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            pred_log_vm = model(geometry_features(x) if kind == "geometry_mlp" else x)
            out.extend((math.log10(YIELD_STRESS) - pred_log_vm).cpu().numpy().tolist())
    synchronize(device)
    elapsed = time.perf_counter() - start
    params = sum(p.numel() for p in model.parameters())
    del model
    return np.asarray(out, dtype=float), elapsed, params


def operator_predictions(checkpoint: Path, rows: list[dict],
                         device: torch.device,
                         batch_size: int = 4) -> tuple[np.ndarray, float, int]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    in_channels = int(state["config"]["in_channels"])
    if in_channels not in (4, 5, 6):
        raise ValueError(f"Unsupported structural input width: {in_channels}")

    def dataset(selected_rows: list[dict]):
        ds = FEMDatasetV3(
            data_dirs=[], split="train", augment=False, target_res=48,
            include_sdf=in_channels >= 5, include_fixed_mask=in_channels == 6,
        )
        root = INDEX.resolve().parent
        ds.files = [root / record["path"] for record in selected_rows]
        ds.solid_fractions = [None] * len(ds.files)
        return ds

    model = load_model(checkpoint, device)
    # Warm up with a small prefix, then time the complete immutable split.
    warm_rows = rows[:min(4, len(rows))]
    _ = predict_log_sf(model, dataset(warm_rows), device, batch_size)
    synchronize(device)
    del _
    synchronize(device)
    start = time.perf_counter()
    pred = predict_log_sf(model, dataset(rows), device, batch_size)
    synchronize(device)
    elapsed = time.perf_counter() - start
    params = sum(p.numel() for p in model.parameters())
    del model
    return pred, elapsed, params


def raw_summary(pred: np.ndarray, truth: np.ndarray) -> dict:
    threshold = math.log10(TAU)
    truth_safe = truth >= threshold
    pred_safe = pred >= threshold
    m = metrics(pred_safe, ~pred_safe, truth_safe, False)
    correct = pred_safe == truth_safe
    m.update({
        # Direct Raw-PINO decisions at SF=TAU: no ACS, deferral, or FEM.
        "auto_rate": float(correct.mean()),
        "auto_safe_rate": float(
            (pred_safe & truth_safe).sum() / max(int(truth_safe.sum()), 1)),
        "auto_unsafe_rate": float(
            ((~pred_safe) & (~truth_safe)).sum() / max(int((~truth_safe).sum()), 1)),
        "log_sf_mae": float(np.abs(pred - truth).mean()),
        "log_sf_rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        # With one fixed yield stress, the absolute error in log10(SF) equals
        # the absolute error in log10 peak von-Mises stress.  This is the fair
        # scalar metric available to MLP, CNN, FNO, no-physics, and full PINO.
        "peak_log_stress_mae": float(np.abs(pred - truth).mean()),
        "auroc": auc(pred, truth_safe),
        "predicted_pass_rate": float(pred_safe.mean()),
        "pass_precision": float((pred_safe & truth_safe).sum() / max(int(pred_safe.sum()), 1)),
        "balanced_accuracy": float(0.5 * (
            (pred_safe & truth_safe).sum() / max(int(truth_safe.sum()), 1)
            + ((~pred_safe) & (~truth_safe)).sum() / max(int((~truth_safe).sum()), 1)
        )),
    })
    return m


def frozen_acs(cal_pred: np.ndarray, cal_truth: np.ndarray,
               test_pred: np.ndarray, test_truth: np.ndarray) -> dict:
    threshold = math.log10(TAU)
    residual = cal_pred - cal_truth
    candidates = []
    for a_safe in (0.005, 0.01, 0.02, 0.05, 0.10):
        q_safe = finite_quantile(residual, 1.0 - a_safe)
        for a_unsafe in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
            q_unsafe = finite_quantile(-residual, 1.0 - a_unsafe)
            safe, unsafe, conflict = decisions(cal_pred, threshold, q_safe, q_unsafe)
            m = metrics(safe, unsafe, cal_truth >= threshold, False)
            candidates.append({"alpha_safe": a_safe, "alpha_unsafe": a_unsafe,
                               "q_safe": q_safe, "q_unsafe": q_unsafe,
                               "conflicts": int(conflict.sum()), "metrics": m})
    valid = [x for x in candidates if x["metrics"]["false_safe_count"] == 0]
    chosen = max(valid, key=lambda x: (x["metrics"]["auto_decision_coverage"],
                                       x["metrics"]["auto_unsafe_coverage"],
                                       x["metrics"]["auto_safe_coverage"]))
    safe, unsafe, conflict = decisions(test_pred, threshold, chosen["q_safe"], chosen["q_unsafe"])
    return {"frozen": {k: chosen[k] for k in ("alpha_safe", "alpha_unsafe", "q_safe", "q_unsafe")},
            "calibration": chosen["metrics"],
            "sealed_test": metrics(safe, unsafe, test_truth >= threshold, True),
            "sealed_conflicts": int(conflict.sum())}


def main() -> None:
    global INDEX
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path,
                        help="Optional current full-PINO checkpoint overriding checkpoint-root/full_pino.")
    parser.add_argument("--gino-checkpoint", type=Path,
                        help="Optional official GINO-backbone checkpoint.")
    parser.add_argument("--transolver-checkpoint", type=Path,
                        help="Optional official Transolver-backbone checkpoint.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--raw-only", action="store_true",
                        help="Evaluate direct autonomous predictions without ACS calibration.")
    parser.add_argument(
        "--only-models", nargs="+",
        choices=("geometry_mlp", "cnn3d", "ufno", "ufno_dual_no_physics",
                 "full_pino", "gino_official", "transolver_official"),
        help="Evaluate only the named models instead of every available checkpoint.",
    )
    parser.add_argument(
        "--operator-batch-size", type=int, default=4,
        help="Inference batch size for neural-operator models.",
    )
    args = parser.parse_args()
    INDEX = args.index
    ckpt = args.checkpoint_root
    out_dir = args.out_dir
    payload = json.loads(INDEX.read_text())
    rows = {name: [r for r in payload["rows"] if r["split"] == name]
            for name in ("calibration", "sealed_test")}
    truth = {name: np.log10(np.asarray([float(r["sf"]) for r in split], dtype=float))
             for name, split in rows.items()}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    paths = {
        "geometry_mlp": ckpt / "geometry_mlp/best_model.pt",
        "cnn3d": ckpt / "cnn3d/best_model.pt",
        "ufno": ckpt / "ufno/best_model.pt",
        "ufno_dual_no_physics": ckpt / "ufno_dual_no_physics/best_model.pt",
        "full_pino": args.full_checkpoint or ckpt / "full_pino/best_model.pt",
    }
    if args.gino_checkpoint:
        paths["gino_official"] = args.gino_checkpoint
    if args.transolver_checkpoint:
        paths["transolver_official"] = args.transolver_checkpoint
    if args.only_models:
        requested = set(args.only_models)
        paths = {name: path for name, path in paths.items() if name in requested}
        missing = requested.difference(paths)
        if missing:
            parser.error(f"Requested model checkpoints were not supplied: {sorted(missing)}")
    result = {"schema": "taskdisjoint_five_model_comparator_v1",
              "index": str(INDEX), "threshold_sf": TAU, "models": {}}
    for name, checkpoint in paths.items():
        print(f"Evaluating {name}", flush=True)
        pred, timings, params = {}, {}, None
        evaluated_splits = ("sealed_test",) if args.raw_only else ("calibration", "sealed_test")
        for split in evaluated_splits:
            if name in ("geometry_mlp", "cnn3d"):
                p, elapsed, n_params = scalar_predictions(name, checkpoint, rows[split], device)
            else:
                p, elapsed, n_params = operator_predictions(
                    checkpoint, rows[split], device, args.operator_batch_size)
            pred[split], timings[split], params = p, elapsed / len(rows[split]), n_params
            if device.type == "cuda": torch.cuda.empty_cache()
        model_result = {
            "checkpoint": str(checkpoint), "parameters": params,
            "inference_s_per_sample": timings["sealed_test"],
            "inference_ms_per_sample": 1000.0 * timings["sealed_test"],
            "raw_sealed": raw_summary(pred["sealed_test"], truth["sealed_test"]),
        }
        if not args.raw_only:
            model_result["frozen_dualside_acs"] = frozen_acs(
                pred["calibration"], truth["calibration"],
                pred["sealed_test"], truth["sealed_test"])
        result["models"][name] = model_result
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    keys = ["model", "log_sf_mae", "peak_log_stress_mae", "auroc",
            "auto_rate", "auto_safe_rate", "auto_unsafe_rate",
            "balanced_accuracy", "raw_false_safe", "raw_false_unsafe",
            "inference_ms_per_sample"]
    if not args.raw_only:
        keys[10:10] = ["acs_coverage", "acs_false_safe", "fem_call_rate"]
    lines = ["\t".join(keys)]
    for name, row in result["models"].items():
        raw = row["raw_sealed"]
        values = [name, raw["log_sf_mae"],
            raw["peak_log_stress_mae"], raw["auroc"],
            raw["auto_rate"], raw["auto_safe_rate"], raw["auto_unsafe_rate"],
            raw["balanced_accuracy"], raw["false_safe_count"], raw["false_unsafe_count"],
            row["inference_ms_per_sample"]]
        if not args.raw_only:
            acs = row["frozen_dualside_acs"]["sealed_test"]
            values[10:10] = [acs["auto_decision_coverage"], acs["false_safe_count"],
                             acs["fem_call_rate"]]
        lines.append("\t".join(map(str, values)))
    (out_dir / "results.tsv").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(out_dir), "models": list(paths)}, indent=2))


if __name__ == "__main__":
    main()
