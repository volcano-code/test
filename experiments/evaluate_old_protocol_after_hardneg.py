#!/usr/bin/env python3
"""Evaluate the old 91.78% autonomous-rate protocol with a newer PINO checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_v26_shadowart_scale_sf import ShadowArtScaleDataset, collate
from structural.models.load_model import load_surrogate

YIELD_STRESS = 250e6
SF_TARGET = 1.5
T_FEM_MS = 15000.0
T_PINO_MS = 8.0
SCALE_BRANCH = ["scale_lifting", "scale_down1", "scale_enc2", "scale_down2", "scale_bottleneck"]


def load_geoattn_model(device: torch.device, base_ckpt: Path, geo_ckpt: Path):
    spec = importlib.util.spec_from_file_location("geoattn_head", ROOT / "certification/02_geoattn_head.py")
    geo = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(geo)

    model, _ = load_surrogate(str(base_ckpt), device)
    model.enable_enhanced_scale_head()
    ckpt = torch.load(geo_ckpt, map_location=device, weights_only=False)
    for name, state in ckpt["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(state)
    head = geo.patch_pino_with_attn(model, attn_dim=ckpt.get("attn_dim", 64), n_heads=ckpt.get("n_heads", 4))
    head.load_state_dict(ckpt["attn_scale_head_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_split(model, dataset: ShadowArtScaleDataset, device: torch.device, batch_size: int) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    pred, true_log, true_sf, case_id = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        b, _, nx, ny, nz = x.shape
        grid = model._get_grid((nx, ny, nz), x.device).expand(b, -1, -1, -1, -1)
        y = model._scale_forward(torch.cat([x, grid], dim=1)).squeeze(-1)
        pred.extend(float(v) for v in y.detach().cpu().reshape(-1))
        true_log.extend(float(v) for v in batch["log_vm"].detach().cpu().reshape(-1))
        true_sf.extend(float(v) for v in batch["sf"].detach().cpu().reshape(-1))
        case_id.extend(int(v) for v in batch["case_id"])
    return {
        "pred_log": np.asarray(pred, dtype=float),
        "true_log": np.asarray(true_log, dtype=float),
        "true_sf": np.asarray(true_sf, dtype=float),
        "case_id": np.asarray(case_id, dtype=int),
    }


def conformal_offset(val_pred: np.ndarray, val_true: np.ndarray, alpha: float) -> float:
    residual = val_true - val_pred
    n = len(residual)
    k = int(np.ceil((1.0 - alpha) * (n + 1)))
    return float(np.sort(residual)[min(max(k, 1), n) - 1])


def old_protocol(
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    val_true_log: np.ndarray,
    test_true_sf: np.ndarray,
    val_sig: np.ndarray,
    test_sig: np.ndarray,
    alpha: float,
    budget: float,
) -> dict:
    offset = conformal_offset(val_pred, val_true_log, alpha)
    pred_sf = YIELD_STRESS / np.maximum(10.0 ** (test_pred + offset), 1e-12)
    pino_pass = pred_sf >= SF_TARGET
    true_pass = test_true_sf >= SF_TARGET

    tau = float(np.quantile(val_sig, 1.0 - budget))
    fem_trigger = test_sig >= tau
    auto = ~fem_trigger
    final_pass = np.where(fem_trigger, true_pass, pino_pass)

    ratio = pred_sf[auto] / np.maximum(test_true_sf[auto], 1e-12)
    log_err = (
        np.log10(np.maximum(pred_sf[auto], 1e-12))
        - np.log10(np.maximum(test_true_sf[auto], 1e-12))
    )

    total_ms = len(test_pred) * T_PINO_MS + int(fem_trigger.sum()) * T_FEM_MS
    return {
        "alpha": alpha,
        "budget": budget,
        "offset": offset,
        "trigger_threshold": tau,
        "n_test": int(len(test_pred)),
        "fem_trigger_count": int(fem_trigger.sum()),
        "fem_trigger_rate": float(fem_trigger.mean()),
        "auto_count": int(auto.sum()),
        "auto_rate": float(auto.mean()),
        "sys_false_pass": int(np.sum(final_pass & ~true_pass)),
        "sys_false_fail": int(np.sum(~final_pass & true_pass)),
        "raw_pino_false_pass": int(np.sum(pino_pass & ~true_pass)),
        "raw_pino_false_fail": int(np.sum(~pino_pass & true_pass)),
        "pino_win_auto_subset": float(((ratio >= 0.5) & (ratio <= 1.5)).mean()) if auto.any() else None,
        "pino_log_mae_auto_subset": float(np.abs(log_err).mean()) if auto.any() else None,
        "speedup": float((T_FEM_MS * len(test_pred)) / total_ms),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_file", type=Path, default=Path("/home/lin/xvxinyao/ShadowForge/data/shadowart_fem_hardset_v1/index.json"))
    ap.add_argument("--old_results_dir", type=Path, default=Path("/data/xvxinyao/xvxinyao/total_project/results"))
    ap.add_argument("--base_ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    ap.add_argument("--new_ckpt", type=Path, default=ROOT / "checkpoints/pino_aug_hardneg_20260624/scale_geoattn_e8.pt")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, default=ROOT / "results/old_protocol_after_pino_aug_hardneg_20260624.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_geoattn_model(device, args.base_ckpt, args.new_ckpt)

    val_ds = ShadowArtScaleDataset(args.index_file, "val")
    test_ds = ShadowArtScaleDataset(args.index_file, "test")
    val = predict_split(model, val_ds, device, args.batch_size)
    test = predict_split(model, test_ds, device, args.batch_size)

    old_v29 = np.load(args.old_results_dir / "v29_per_sample.npz")
    old_ens = np.load(args.old_results_dir / "v27_ensemble_per_sample.npz", allow_pickle=True)

    checks = {
        "val_true_log_max_abs_diff_vs_old": float(np.max(np.abs(val["true_log"] - old_v29["val_true_log"]))),
        "test_true_log_max_abs_diff_vs_old": float(np.max(np.abs(test["true_log"] - old_v29["test_true_log"]))),
        "val_true_sf_max_abs_diff_vs_old": float(np.max(np.abs(val["true_sf"] - old_v29["val_true_sf"]))),
        "test_true_sf_max_abs_diff_vs_old": float(np.max(np.abs(test["true_sf"] - old_v29["test_true_sf"]))),
    }

    val_disagree = np.abs(old_ens["val_pred_v27"] - old_ens["val_pred_v28"])
    test_disagree = np.abs(old_ens["test_pred_v27"] - old_ens["test_pred_v28"])

    rows = {
        "new_hardneg_pino_same_old_protocol_alpha0.05_budget0.08": old_protocol(
            val["pred_log"],
            test["pred_log"],
            old_v29["val_true_log"],
            old_v29["test_true_sf"],
            val_disagree,
            test_disagree,
            alpha=0.05,
            budget=0.08,
        ),
        "old_reported_reference_v29_ens_raw_G": {
            "alpha": 0.05,
            "budget": 0.08,
            "auto_rate": 0.918,
            "sys_false_pass": 2,
            "sys_false_fail": 93,
            "pino_win_auto_subset": 0.673,
            "pino_log_mae_auto_subset": 0.239,
            "speedup": 11.97,
            "source": "structural/paper/generated/table_hybrid_pipeline.tex",
        },
    }

    result = {
        "schema_version": "old_pino_autonomy_protocol_v1",
        "protocol": "old conformal alpha=0.05 + |v27-v28| disagreement FEM trigger budget=8% on ShadowArt-Hard val/test 450/450",
        "checkpoint": str(args.new_ckpt),
        "index_file": str(args.index_file),
        "order_checks": checks,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
