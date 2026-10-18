#!/usr/bin/env python3
"""Task-disjoint ACS audit for strict full-field physics-residual PINO.

The input index must contain rows with train/calibration/sealed_test task splits.
ACS quantiles are estimated from calibration only; the sealed-test split is read
exactly once for final reporting and is never used to choose a threshold.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from structural.models.official_operator_baselines import (  # noqa: E402
    OfficialGINO3d,
    OfficialTransolver3d,
)

YIELD_STRESS = 50.0e6
TAU = 1.3


def finite_quantile(values: np.ndarray, level: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(int(math.ceil((len(ordered) + 1) * level)) - 1, len(ordered) - 1)
    return float(ordered[max(rank, 0)])


def risk_upper(false_safe: int, accepted: int, confidence: float = 0.95) -> float | None:
    if accepted == 0:
        return None
    if false_safe == accepted:
        return 1.0
    return float(beta.ppf(confidence, false_safe + 1, accepted - false_safe))


def make_dataset(index_path: Path, records: list[dict], target_res: int) -> FEMDatasetV3:
    """Reuse the training preprocessing without allowing its train/val split logic."""
    dataset = FEMDatasetV3(data_dirs=[], split="train", augment=False,
                           target_res=target_res, include_fixed_mask=True)
    root = index_path.resolve().parent
    dataset.files = [root / record["path"] for record in records]
    dataset.solid_fractions = [None] * len(dataset.files)
    dataset.augment = False
    return dataset


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state["config"]
    model_map = {
        "pino": PINO3d,
        "ufno": UFNO3d,
        "ufno_dual": UFNODualBranch3d,
        "gino_official": OfficialGINO3d,
        "transolver_official": OfficialTransolver3d,
    }
    model = model_map[state["arch"]](**config).to(device)
    model_state = state["model_state_dict"].copy()
    # Official neuraloperator checkpoints may serialize module metadata as a
    # dictionary entry. It is not a trainable tensor and must not be passed to
    # PyTorch's strict parameter loader.
    model_state.pop("_metadata", None)
    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def predict_log_sf(model: torch.nn.Module, dataset: FEMDatasetV3, device: torch.device,
                   batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda", collate_fn=collate_v3)
    out: list[float] = []
    total = len(loader)
    for step, batch in enumerate(loader, start=1):
        x = batch["x"].to(device)
        if hasattr(model, "forward_with_scale"):
            pred, log_vm_max = model.forward_with_scale(x)
        else:
            pred, log_vm_max = model(x), None
        vm = compute_von_mises(pred[:, 3:].float())
        density = batch["density"].to(device)
        for i in range(pred.shape[0]):
            mask = density[i, 0] > FEMDatasetV3.SOLID_THRESHOLD
            vm_norm_max = float(vm[i, 0][mask].max().item()) if mask.any() else 1e-12
            scale = 1.0 if log_vm_max is None else 10.0 ** float(log_vm_max[i, 0].item())
            sf = YIELD_STRESS / max(vm_norm_max * scale, 1e-12)
            out.append(float(math.log10(sf)))
        # The 48^3 dual-branch model is evaluated over long sealed splits.
        # Explicitly release each batch to prevent CUDA cache growth on shared GPUs.
        del x, pred, log_vm_max, vm, density
        if device.type == "cuda" and (step % 32 == 0 or step == total):
            torch.cuda.empty_cache()
        if step == 1 or step % 100 == 0 or step == total:
            print(f"  predicted {step}/{total} batches", flush=True)
    return np.asarray(out, dtype=float)


def metrics(auto_safe: np.ndarray, auto_unsafe: np.ndarray, truth_safe: np.ndarray,
            fallback: bool) -> dict:
    decided = auto_safe | auto_unsafe
    false_safe = int((auto_safe & ~truth_safe).sum())
    false_unsafe = int((auto_unsafe & truth_safe).sum())
    accepted = int(auto_safe.sum())
    n_safe = int(truth_safe.sum())
    n_unsafe = int((~truth_safe).sum())
    defer = int((~decided).sum())
    return {
        "auto_decision_coverage": float(decided.mean()),
        "auto_safe_coverage": float((auto_safe & truth_safe).sum() / max(n_safe, 1)),
        "auto_unsafe_coverage": float((auto_unsafe & ~truth_safe).sum() / max(n_unsafe, 1)),
        "false_safe_count": false_safe,
        "false_safe_rate": float(false_safe / max(accepted, 1)),
        "false_safe_risk_upper_95": risk_upper(false_safe, accepted),
        "false_unsafe_count": false_unsafe,
        "defer_rate": float(defer / len(truth_safe)),
        "fem_call_rate": float(defer / len(truth_safe)) if fallback else None,
        # FEM resolves only deferred candidates; an erroneous automatic SAFE
        # decision remains a system false-safe decision under this policy.
        "final_system_false_safe_count": false_safe,
        "n_safe": n_safe,
        "n_unsafe": n_unsafe,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--target-res", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--alphas", default="0.01,0.02,0.05,0.10,0.20,0.30")
    ap.add_argument("--freeze-policy-on-calibration", action="store_true",
                    help="Choose one zero-error deployment point using calibration only; never use sealed metrics for policy choice.")
    ap.add_argument(
        "--deployment-constraint",
        choices=("zero_false_safe", "zero_both"),
        default="zero_both",
        help=("Calibration-only deployment constraint. zero_false_safe treats "
              "false-safe decisions as the safety-critical error; zero_both also "
              "forbids conservative false-unsafe decisions."),
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    payload = json.loads(args.index.read_text())
    records = payload["rows"]
    splits = {name: [r for r in records if r["split"] == name]
              for name in ("train", "calibration", "sealed_test")}
    if not splits["calibration"] or not splits["sealed_test"]:
        raise RuntimeError("The index must provide non-empty calibration and sealed_test task splits.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    pred = {name: predict_log_sf(model, make_dataset(args.index, rows, args.target_res),
                                 device, args.batch_size)
            for name, rows in splits.items() if name != "train"}
    truth = {name: np.log10(np.asarray([float(r["sf"]) for r in rows]))
             for name, rows in splits.items() if name != "train"}
    threshold = math.log10(TAU)
    test_safe = truth["sealed_test"] >= threshold
    entries: list[dict] = []

    raw_safe = pred["sealed_test"] >= threshold
    entries.append({"policy": "Raw PINO threshold", "alpha": None, "frozen": {},
                    "metrics": metrics(raw_safe, ~raw_safe, test_safe, False)})
    residual = pred["calibration"] - truth["calibration"]
    calibration_safe = truth["calibration"] >= threshold
    calibration_candidates: list[dict] = []
    for alpha in [float(v) for v in args.alphas.split(",") if v.strip()]:
        q_sym = finite_quantile(np.abs(residual), 1.0 - alpha)
        sym_safe = pred["sealed_test"] - q_sym >= threshold
        sym_unsafe = pred["sealed_test"] + q_sym < threshold
        entries.append({"policy": "Symmetric conformal", "alpha": alpha,
                        "frozen": {"q_symmetric_log_sf": q_sym},
                        "metrics": metrics(sym_safe, sym_unsafe, test_safe, False)})
        q_safe = finite_quantile(residual, 1.0 - alpha)
        q_unsafe = finite_quantile(-residual, 1.0 - alpha)
        acs_safe = pred["sealed_test"] - q_safe >= threshold
        acs_unsafe = pred["sealed_test"] + q_unsafe < threshold
        frozen = {"q_safe_log_sf": q_safe, "q_unsafe_log_sf": q_unsafe}
        entries.append({"policy": "Asymmetric conformal", "alpha": alpha,
                        "frozen": frozen, "metrics": metrics(acs_safe, acs_unsafe, test_safe, False)})
        entries.append({"policy": "ACS + FEM fallback", "alpha": alpha,
                        "frozen": frozen, "metrics": metrics(acs_safe, acs_unsafe, test_safe, True)})

        # Deployment-point selection is intentionally performed only on the
        # calibration split.  This avoids choosing alpha after observing sealed
        # test coverage or false-safe counts.
        cal_sym_safe = pred["calibration"] - q_sym >= threshold
        cal_sym_unsafe = pred["calibration"] + q_sym < threshold
        cal_acs_safe = pred["calibration"] - q_safe >= threshold
        cal_acs_unsafe = pred["calibration"] + q_unsafe < threshold
        calibration_candidates.extend([
            {"policy": "Symmetric conformal", "alpha": alpha,
             "frozen": {"q_symmetric_log_sf": q_sym},
             "metrics": metrics(cal_sym_safe, cal_sym_unsafe, calibration_safe, False)},
            {"policy": "Asymmetric conformal", "alpha": alpha,
             "frozen": frozen,
             "metrics": metrics(cal_acs_safe, cal_acs_unsafe, calibration_safe, False)},
        ])

    deployment_policy = None
    if args.freeze_policy_on_calibration:
        valid = [r for r in calibration_candidates
                 if r["metrics"]["false_safe_count"] == 0]
        if args.deployment_constraint == "zero_both":
            valid = [r for r in valid if r["metrics"]["false_unsafe_count"] == 0]
        if valid:
            selected = max(valid, key=lambda r: (
                r["metrics"]["auto_decision_coverage"],
                r["metrics"]["auto_unsafe_coverage"],
                r["metrics"]["auto_safe_coverage"],
            ))
            match = next(r for r in entries
                         if r["policy"] == selected["policy"] and r["alpha"] == selected["alpha"])
            deployment_policy = {
                "selection_protocol": (
                    "calibration only; maximize autonomous coverage subject to "
                    + ("zero calibration false-safe decisions"
                       if args.deployment_constraint == "zero_false_safe"
                       else "zero calibration false-safe and false-unsafe decisions")
                ),
                "deployment_constraint": args.deployment_constraint,
                "calibration_metrics": selected["metrics"],
                "sealed_metrics": match["metrics"],
                "policy": selected["policy"],
                "alpha": selected["alpha"],
                "frozen": selected["frozen"],
            }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "strict_physics_residual_pino_taskgrouped_acs_v1",
        "checkpoint": str(args.checkpoint),
        "index": str(args.index),
        "threshold_sf": TAU,
        "split_unit": payload.get("split_unit"),
        "calibration_protocol": "ACS quantiles are fit on calibration tasks only.",
        "sealed_protocol": "Sealed tasks are evaluated once after corrections are frozen.",
        "rows": {name: len(rows) for name, rows in splits.items()},
        "policies": entries,
        "deployment_policy": deployment_policy,
    }
    (args.out_dir / "acs_taskgrouped_results.json").write_text(json.dumps(result, indent=2) + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6), constrained_layout=True)
    for policy, color, marker in [("Raw PINO threshold", "#6b7280", "s"),
                                  ("Symmetric conformal", "#2563eb", "o"),
                                  ("Asymmetric conformal", "#d97706", "^"),
                                  ("ACS + FEM fallback", "#059669", "D")]:
        rows = [r for r in entries if r["policy"] == policy]
        x = [100.0 * r["metrics"]["auto_decision_coverage"] for r in rows]
        y = [100.0 * r["metrics"]["false_safe_risk_upper_95"]
             if r["metrics"]["false_safe_risk_upper_95"] is not None else np.nan for r in rows]
        ax1.plot(x, y, marker=marker, color=color, label=policy)
        fem = [r["metrics"]["fem_call_rate"] for r in rows]
        if any(v is not None for v in fem):
            ax2.plot(x, [100.0 * v if v is not None else np.nan for v in fem],
                     marker=marker, color=color, label=policy)
    ax1.set(xlabel="Automatic-decision coverage (%)", ylabel="False-safe risk upper bound (%)",
            title="Risk--coverage on sealed tasks")
    ax2.set(xlabel="Automatic-decision coverage (%)", ylabel="FEM call rate (%)",
            title="Deferral cost of ACS + FEM")
    for ax in (ax1, ax2):
        ax.grid(alpha=.25); ax.legend(fontsize=7, frameon=False)
    fig.savefig(args.out_dir / "risk_coverage_and_fem_calls.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "risk_coverage_and_fem_calls.png", dpi=300, bbox_inches="tight")
    print(json.dumps({"out": str(args.out_dir), "sealed_rows": len(splits["sealed_test"]),
                      "n_policy_points": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
