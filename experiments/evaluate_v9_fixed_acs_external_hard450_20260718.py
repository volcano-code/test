#!/usr/bin/env python3
"""One-shot external audit of the frozen v9 PINO + alpha=0.1 ACS policy.

The checkpoint and conformal corrections are frozen from the strict v9 audit.
No Hard450 row is used for calibration or policy selection.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structural.training.pretrain_pino_v3 import (  # noqa: E402
    FEMDatasetV3,
    PINO3d,
    UFNO3d,
    UFNODualBranch3d,
    collate_v3,
    compute_von_mises,
)

TAU = 1.3
YIELD_STRESS = 250.0e6
Q_SAFE = 0.1952986028043909
Q_UNSAFE = 0.31987571058142183


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location=device, weights_only=False)
    model_cls = {"pino": PINO3d, "ufno": UFNO3d, "ufno_dual": UFNODualBranch3d}[state["arch"]]
    model = model_cls(**state["config"]).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model


def make_dataset(paths: list[Path]) -> FEMDatasetV3:
    dataset = FEMDatasetV3(data_dirs=[], split="train", augment=False,
                           target_res=48, include_fixed_mask=True)
    dataset.files = paths
    dataset.solid_fractions = [None] * len(paths)
    dataset.augment = False
    return dataset


@torch.no_grad()
def predict(model: torch.nn.Module, dataset: FEMDatasetV3, device: torch.device,
            batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda", collate_fn=collate_v3)
    values: list[float] = []
    for step, batch in enumerate(loader, 1):
        x = batch["x"].to(device)
        pred, log_vm_max = model.forward_with_scale(x)
        vm_shape = compute_von_mises(pred[:, 3:].float())
        density = batch["density"].to(device)
        for i in range(pred.shape[0]):
            mask = density[i, 0] > FEMDatasetV3.SOLID_THRESHOLD
            shape_peak = float(vm_shape[i, 0][mask].max().item()) if mask.any() else 1e-12
            scale = 10.0 ** float(log_vm_max[i, 0].item())
            sf = YIELD_STRESS / max(shape_peak * scale, 1e-12)
            values.append(float(math.log10(sf)))
        if step == 1 or step % 50 == 0 or step == len(loader):
            print(f"predicted {step}/{len(loader)} batches", flush=True)
    return np.asarray(values, dtype=float)


def risk_upper(false_safe: int, accepted: int) -> float | None:
    if accepted == 0:
        return None
    if false_safe == accepted:
        return 1.0
    return float(beta.ppf(0.95, false_safe + 1, accepted - false_safe))


def summarize(auto_safe: np.ndarray, auto_unsafe: np.ndarray,
              truth_safe: np.ndarray) -> dict:
    decided = auto_safe | auto_unsafe
    false_safe = int(np.count_nonzero(auto_safe & ~truth_safe))
    false_unsafe = int(np.count_nonzero(auto_unsafe & truth_safe))
    accepted = int(np.count_nonzero(auto_safe))
    n_safe = int(np.count_nonzero(truth_safe))
    n_unsafe = int(np.count_nonzero(~truth_safe))
    return {
        "n": int(len(truth_safe)),
        "n_safe": n_safe,
        "n_unsafe": n_unsafe,
        "auto_decision_coverage": float(decided.mean()),
        "auto_safe_coverage": float(np.count_nonzero(auto_safe & truth_safe) / max(n_safe, 1)),
        "auto_unsafe_coverage": float(np.count_nonzero(auto_unsafe & ~truth_safe) / max(n_unsafe, 1)),
        "auto_safe_precision": float(np.count_nonzero(auto_safe & truth_safe) / max(accepted, 1)),
        "false_safe_count": false_safe,
        "false_unsafe_count": false_unsafe,
        "defer_rate": float(np.count_nonzero(~decided) / len(truth_safe)),
        "fem_call_rate": float(np.count_nonzero(~decided) / len(truth_safe)),
        "false_safe_risk_upper_95": risk_upper(false_safe, accepted),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--hardset-index", type=Path, required=True)
    ap.add_argument("--hardset-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    source = json.loads(args.hardset_index.read_text())
    records = source["test"]
    paths = [args.hardset_root / r["path"] for r in records]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} Hard450 tensors; first={missing[0]}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    pred_log_sf = predict(model, make_dataset(paths), device, args.batch_size)
    true_sf = np.asarray([float(r["sf"]) for r in records], dtype=float)
    true_log_sf = np.log10(true_sf)
    threshold = math.log10(TAU)
    truth_safe = true_log_sf >= threshold
    auto_safe = pred_log_sf - Q_SAFE >= threshold
    auto_unsafe = pred_log_sf + Q_UNSAFE < threshold

    rows = []
    for record, pred, safe, unsafe in zip(records, pred_log_sf, auto_safe, auto_unsafe):
        rows.append({
            "path": record["path"],
            "case_id": record.get("case_id"),
            "case_name": record.get("case_name"),
            "sf_fem": float(record["sf"]),
            "sf_pino": float(10.0 ** pred),
            "truth_safe": bool(float(record["sf"]) >= TAU),
            "decision": "SAFE" if safe else ("UNSAFE" if unsafe else "DEFER"),
        })

    payload = {
        "schema": "v9_fixed_alpha01_external_hard450_audit_v1",
        "protocol": {
            "checkpoint": str(args.checkpoint),
            "policy": "asymmetric ACS",
            "alpha": 0.1,
            "q_safe_log_sf": Q_SAFE,
            "q_unsafe_log_sf": Q_UNSAFE,
            "threshold_sf": TAU,
            "yield_stress_pa": YIELD_STRESS,
            "selection": "checkpoint and ACS corrections frozen before this external audit; no recalibration",
            "external_split": "Hard450 test cases only",
        },
        "metrics": summarize(auto_safe, auto_unsafe, truth_safe),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
