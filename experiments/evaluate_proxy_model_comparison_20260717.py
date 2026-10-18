#!/usr/bin/env python3
"""Audit the five existing structural surrogate checkpoints on one frozen split.

The 2026-07-11 comparator checkpoints share a sample-level split.  This script
does not present that split as task-disjoint evidence; it records the limitation
in the output.  The FNO checkpoint uses its bottleneck scale head, whereas the
two PINO variants use the dual-branch scale head.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.pino_proxy_comparison_20260711 import (
    GeometryMLP,
    ScalarCNN3D,
    geometry_features,
)
from structural.models.pino import PINO3d, UFNO3d, UFNODualBranch3d
from structural.training.pretrain_pino_v3 import (
    FEMDatasetV3,
    collate_v3,
    compute_von_mises,
)


ROOT = Path("results/pino_proxy_model_comparison_20260711")
OUT = Path("results/pino_proxy_model_comparison_eval_20260717")
YIELD_STRESS = 50.0e6
SF_THRESHOLD = 1.3


def dataset_for(records: list[dict]) -> FEMDatasetV3:
    ds = FEMDatasetV3(data_dirs=[], split="train", augment=False,
                      target_res=64, include_fixed_mask=False)
    ds.files = [Path(row["path"]) for row in records]
    ds.solid_fractions = [None] * len(ds.files)
    ds.augment = False
    return ds


def loader(records: list[dict], batch_size: int = 2) -> DataLoader:
    return DataLoader(dataset_for(records), batch_size=batch_size, shuffle=False,
                      num_workers=2, pin_memory=True, persistent_workers=True,
                      collate_fn=collate_v3)


def average_rank_auc(score: np.ndarray, positive: np.ndarray) -> float | None:
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
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
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def decision_metrics(pred_log_vm: np.ndarray, true_log_vm: np.ndarray) -> dict:
    pred_log_sf = math.log10(YIELD_STRESS) - pred_log_vm
    true_log_sf = math.log10(YIELD_STRESS) - true_log_vm
    threshold = math.log10(SF_THRESHOLD)
    pred_safe = pred_log_sf >= threshold
    true_safe = true_log_sf >= threshold
    fp = int((pred_safe & ~true_safe).sum())
    fn = int((~pred_safe & true_safe).sum())
    n_safe = int(true_safe.sum())
    n_unsafe = int((~true_safe).sum())
    return {
        "accuracy": float((pred_safe == true_safe).mean()),
        "balanced_accuracy": float(0.5 * (
            (pred_safe & true_safe).sum() / max(n_safe, 1)
            + ((~pred_safe) & (~true_safe)).sum() / max(n_unsafe, 1)
        )),
        "auroc": average_rank_auc(pred_log_sf, true_safe),
        "false_safe_count": fp,
        "false_unsafe_count": fn,
        "n_safe": n_safe,
        "n_unsafe": n_unsafe,
    }


def load_scalar(kind: str, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(ROOT / kind / "best_model.pt", map_location="cpu",
                            weights_only=False)
    model = GeometryMLP() if kind == "mlp" else ScalarCNN3D()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def load_operator(kind: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(ROOT / kind / "best_model.pt", map_location="cpu",
                            weights_only=False)
    classes = {"pino": PINO3d, "ufno": UFNO3d, "ufno_dual": UFNODualBranch3d}
    model = classes[checkpoint["arch"]](**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def predict_scalar(model: torch.nn.Module, records: list[dict], kind: str,
                   device: torch.device) -> dict:
    pred_log, true_log = [], []
    start = time.perf_counter()
    for batch in loader(records, 4):
        x = batch["x"].to(device, non_blocking=True)
        pred = model(geometry_features(x) if kind == "mlp" else x)
        pred_log.extend(pred.detach().cpu().numpy().tolist())
        true_log.extend(np.asarray(batch["log_vm_max"], dtype=float).tolist())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return summarize(np.asarray(pred_log), np.asarray(true_log), elapsed,
                     field_metrics=None)


@torch.no_grad()
def predict_operator(model: torch.nn.Module, records: list[dict],
                     device: torch.device) -> dict:
    pred_log, true_log = [], []
    disp_rel, stress_rel, vm_peak_rel = [], [], []
    has_scale = hasattr(model, "forward_with_scale")
    start = time.perf_counter()
    for batch in loader(records, 2):
        x = batch["x"].to(device, non_blocking=True)
        if has_scale:
            pred, log_scale = model.forward_with_scale(x)
        else:
            pred, log_scale = model(x), None
        density = batch["density"].to(device)
        true_disp = batch["disp_norm"].to(device)
        true_stress = batch["stress_log"].to(device)
        pred_vm = compute_von_mises(pred[:, 3:].float())
        for i in range(pred.shape[0]):
            disp_rel.append(float(torch.linalg.vector_norm(pred[i, :3] - true_disp[i]) /
                                  torch.linalg.vector_norm(true_disp[i]).clamp_min(1e-12)))
            stress_rel.append(float(torch.linalg.vector_norm(pred[i, 3:] - true_stress[i]) /
                                    torch.linalg.vector_norm(true_stress[i]).clamp_min(1e-12)))
            mask = density[i, 0] > FEMDatasetV3.SOLID_THRESHOLD
            peak = float(pred_vm[i, 0][mask].max()) if mask.any() else 1e-12
            vm_peak_rel.append(abs(peak - 1.0))
            if log_scale is not None:
                pred_log.append(math.log10(max(peak, 1e-12)) + float(log_scale[i, 0]))
            true_log.append(float(batch["log_vm_max"][i]))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    fields = {
        "displacement_rel_l2_mean": float(np.mean(disp_rel)),
        "normalized_stress_rel_l2_mean": float(np.mean(stress_rel)),
        "normalized_vm_peak_rel_mean": float(np.mean(vm_peak_rel)),
    }
    if not has_scale:
        return {
            "n": len(records),
            "absolute_scale_available": False,
            "log_vm_mae": None,
            "log_vm_rmse": None,
            "decision": None,
            "inference_s_per_sample": elapsed / max(len(records), 1),
            "field_metrics": fields,
            "note": "Single-branch FNO predicts normalized fields only; no scale head is present.",
        }
    return summarize(np.asarray(pred_log), np.asarray(true_log), elapsed, fields)


def summarize(pred: np.ndarray, truth: np.ndarray, elapsed: float,
              field_metrics: dict | None) -> dict:
    return {
        "n": int(len(truth)),
        "absolute_scale_available": True,
        "log_vm_mae": float(np.abs(pred - truth).mean()),
        "log_vm_rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "decision": decision_metrics(pred, truth),
        "inference_s_per_sample": elapsed / max(len(truth), 1),
        "field_metrics": field_metrics,
    }


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manifest = json.loads((ROOT / "split_manifest.json").read_text())
    splits = {"calibration": manifest["val"], "sealed_test": manifest["sealed_test"]}
    result = {
        "schema": "shadowforge_proxy_comparator_audit_v1",
        "protocol": {
            "source": str((ROOT / "split_manifest.json").resolve()),
            "split_unit": "sample-level (legacy; not task-disjoint)",
            "threshold_sf": SF_THRESHOLD,
            "yield_stress_pa": YIELD_STRESS,
            "warning": "Use as a diagnostic comparison, not the final leakage-safe paper table.",
        },
        "models": {},
    }
    for kind in ("mlp", "cnn"):
        print(f"Evaluating {kind}", flush=True)
        model = load_scalar(kind, device)
        result["models"][kind] = {
            split: predict_scalar(model, rows, kind, device)
            for split, rows in splits.items()
        }
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    for kind in ("fno", "pino_no_physics", "full_pino"):
        print(f"Evaluating {kind}", flush=True)
        model, checkpoint = load_operator(kind, device)
        result["models"][kind] = {
            "checkpoint_epoch": checkpoint.get("epoch"),
            **{split: predict_operator(model, rows, device)
               for split, rows in splits.items()},
        }
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    columns = ["model", "log_vm_mae", "log_vm_rmse", "auroc", "balanced_accuracy",
               "false_safe", "false_unsafe", "seconds_per_sample"]
    lines = ["\t".join(columns)]
    for name, record in result["models"].items():
        sealed = record["sealed_test"]
        decision = sealed.get("decision") or {}
        lines.append("\t".join(map(str, [
            name, sealed.get("log_vm_mae"), sealed.get("log_vm_rmse"),
            decision.get("auroc"), decision.get("balanced_accuracy"),
            decision.get("false_safe_count"), decision.get("false_unsafe_count"),
            sealed.get("inference_s_per_sample"),
        ])))
    (OUT / "comparison.tsv").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out": str(OUT), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
