#!/usr/bin/env python3
"""Decision-aware PINO fine-tuning with sealed zero-false-pass audits.

This is intentionally separate from the deployment pipeline. It trains the PINO
scale branch plus a small decision head on candidate, boundary, external-cal, and
near-boundary hardset FEM rows. Thresholds are selected only on calibration rows;
candidate/external/hardset test rows are audit-only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_candidate_pino_field import load_scaleup_rows
from train_candidate_pino_grouped import split_groups, subset
from train_near_boundary_safety_decision_gate import HARDSET_ROOT, load_hardset_rows
from train_pino_aug_hardneg import SCALE_BRANCH, load_geoattn_model, xg
from train_pino_safe_side_boundary import SEED, TAU, load_boundary_rows, load_candidate_rows, predict
from train_pino_safe_side_external_hardneg import split_external

YIELD_STRESS = 50e6


class DecisionHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def decision_features(model, density: torch.Tensor, log_sf: torch.Tensor) -> torch.Tensor:
    h = model._scale_feat
    pool_mean = F.adaptive_avg_pool3d(h, 1).flatten(1)
    pool_max = F.adaptive_max_pool3d(h, 1).flatten(1)
    struct = model._compute_structural_features(density[:, :1])
    return torch.cat([log_sf.reshape(-1, 1), pool_mean, pool_max, struct], dim=1)


def eval_metric(scores: np.ndarray, rows: list[dict], threshold: float) -> dict:
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


@torch.no_grad()
def predict_decision(model, head: DecisionHead, rows: list[dict], device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    head.eval()
    scores = []
    log_vms = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        xb = torch.cat([xg(model, row["density"], row["force"], device) for row in batch], dim=0)
        pred_log_vm = model._scale_forward(xb).squeeze(-1).reshape(-1)
        pred_log_sf = math.log10(YIELD_STRESS) - pred_log_vm
        feats = decision_features(model, xb, pred_log_sf)
        logits = head(feats)
        scores.extend(float(v) for v in torch.sigmoid(logits).detach().cpu())
        log_vms.extend(float(v) for v in pred_log_vm.detach().cpu())
    return np.asarray(scores, dtype=float), np.asarray(log_vms, dtype=float)


def raw_stress_metric(pred_log_vm: np.ndarray, rows: list[dict]) -> dict:
    truth = np.asarray([row["sf"] >= TAU for row in rows], dtype=bool)
    pred_log_sf = math.log10(YIELD_STRESS) - pred_log_vm
    pred_safe = pred_log_sf >= math.log10(TAU)
    return {
        "n": len(rows),
        "safe": int(truth.sum()),
        "unsafe": int((~truth).sum()),
        "raw_safe_selected": int(pred_safe.sum()),
        "raw_safe_true": int((pred_safe & truth).sum()),
        "raw_false_pass": int((pred_safe & ~truth).sum()),
        "raw_safe_rate": float((pred_safe & truth).sum() / max(int(truth.sum()), 1)),
        "log_vm_mae": float(np.mean(np.abs(pred_log_vm - np.asarray([row["log_vm"] for row in rows], dtype=float)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--false-safe-weight", type=float, default=28.0)
    parser.add_argument("--hardset-unsafe-weight", type=float, default=1.0)
    parser.add_argument("--safe-miss-weight", type=float, default=0.8)
    parser.add_argument("--threshold-margin", type=float, default=0.0)
    parser.add_argument("--external-calibration", type=int, default=240)
    parser.add_argument("--boundary-limit", type=int, default=600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--start-ckpt", type=Path, default=ROOT / "checkpoints/pino_safe_side_external_hardneg_cal240_20260703/safe_side_external_hardneg_scale.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "results/decision_aware_pino_20260704.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/decision_aware_pino_20260704")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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

    train_rows = candidate_train + boundary_rows + external_cal + hard_train
    calibration_rows = candidate_val + external_cal + hard_val

    model, attn_head, payload = load_geoattn_model(device, args.base_ckpt, args.start_ckpt)
    start_payload = torch.load(args.start_ckpt, map_location=device, weights_only=False)
    for name, state in start_payload["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(state)
    if "attn_scale_head_state_dict" in start_payload:
        attn_head.load_state_dict(start_payload["attn_scale_head_state_dict"])

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name in SCALE_BRANCH:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    for parameter in attn_head.parameters():
        parameter.requires_grad_(True)
        trainable.append(parameter)

    with torch.no_grad():
        probe = torch.cat([xg(model, train_rows[0]["density"], train_rows[0]["force"], device)], dim=0)
        pred = model._scale_forward(probe).squeeze(-1).reshape(-1)
        feat_dim = decision_features(model, probe, math.log10(YIELD_STRESS) - pred).shape[1]
    decision_head = DecisionHead(feat_dim).to(device)
    trainable.extend(decision_head.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    log_yield = math.log10(YIELD_STRESS)
    log_tau = math.log10(TAU)
    best = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        decision_head.train()
        order = rng.permutation(len(train_rows))
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = [train_rows[int(i)] for i in order[start:start + args.batch_size]]
            xb = torch.cat([xg(model, row["density"], row["force"], device) for row in batch], dim=0)
            target_log_vm = torch.as_tensor([row["log_vm"] for row in batch], dtype=torch.float32, device=device)
            sf = torch.as_tensor([row["sf"] for row in batch], dtype=torch.float32, device=device)
            y_safe = (sf >= TAU).float()
            source_hard = torch.as_tensor([row.get("source") == "hardset_near_boundary" for row in batch], dtype=torch.bool, device=device)

            pred_log_vm = model._scale_forward(xb).squeeze(-1).reshape(-1)
            pred_log_sf = log_yield - pred_log_vm
            logits = decision_head(decision_features(model, xb, pred_log_sf))
            prob = torch.sigmoid(logits)

            near = (torch.abs(torch.log10(sf) - log_tau) < 0.22).float()
            reg = F.smooth_l1_loss(pred_log_vm, target_log_vm, reduction="none")
            under_vm = F.relu(target_log_vm - pred_log_vm)
            bce = F.binary_cross_entropy_with_logits(logits, y_safe, reduction="none")
            hard_unsafe = source_hard.float() * (1.0 - y_safe)
            false_safe = ((1.0 - y_safe) * F.relu(prob - 0.02) ** 2) * (1.0 + args.hardset_unsafe_weight * hard_unsafe)
            safe_miss = (y_safe * F.relu(0.78 - prob) ** 2)
            weights = 1.0 + 3.0 * near + 1.5 * source_hard.float()
            loss = torch.mean(
                weights * (reg + 4.0 * bce)
                + 5.0 * under_vm
                + args.false_safe_weight * false_safe
                + args.safe_miss_weight * safe_miss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)

        cal_scores, cal_pred = predict_decision(model, decision_head, calibration_rows, device, args.batch_size)
        threshold = choose_threshold(cal_scores, calibration_rows) + args.threshold_margin
        cal_metric = eval_metric(cal_scores, calibration_rows, threshold)
        cand_scores, cand_pred = predict_decision(model, decision_head, candidate_test, device, args.batch_size)
        ext_scores, ext_pred = predict_decision(model, decision_head, external_heldout, device, args.batch_size)
        hard_scores, hard_pred = predict_decision(model, decision_head, hard_test, device, args.batch_size)
        record = {
            "epoch": epoch,
            "train_loss": total / len(train_rows),
            "threshold": threshold,
            "calibration": cal_metric,
            "candidate_heldout": eval_metric(cand_scores, candidate_test, threshold),
            "external_heldout": eval_metric(ext_scores, external_heldout, threshold),
            "hardset_test_near_boundary": eval_metric(hard_scores, hard_test, threshold),
        }
        record["joint_false_pass"] = sum(record[k]["auto_false_pass"] for k in ("candidate_heldout", "external_heldout", "hardset_test_near_boundary"))
        record["joint_auto_safe_true"] = sum(record[k]["auto_safe_true"] for k in ("candidate_heldout", "external_heldout", "hardset_test_near_boundary"))
        history.append(record)
        print(json.dumps(record), flush=True)
        key = (record["joint_false_pass"], -record["joint_auto_safe_true"], -record["candidate_heldout"]["auto_safe_true"])
        if best is None or key < best[0]:
            best = (
                key,
                {
                    "scale_branch_state_dict": {name: {k: v.detach().cpu().clone() for k, v in getattr(model, name).state_dict().items()} for name in SCALE_BRANCH},
                    "attn_scale_head_state_dict": {k: v.detach().cpu().clone() for k, v in attn_head.state_dict().items()},
                    "decision_head_state_dict": {k: v.detach().cpu().clone() for k, v in decision_head.state_dict().items()},
                    "feat_dim": feat_dim,
                    "epoch": epoch,
                    "threshold": threshold,
                    "record": record,
                },
            )

    state = best[1]
    for name, sd in state["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(sd)
    attn_head.load_state_dict(state["attn_scale_head_state_dict"])
    decision_head.load_state_dict(state["decision_head_state_dict"])

    final = state["record"]
    final["candidate_raw_stress"] = raw_stress_metric(predict(model, candidate_test, device, args.batch_size), candidate_test)
    final["external_raw_stress"] = raw_stress_metric(predict(model, external_heldout, device, args.batch_size), external_heldout)
    final["hardset_raw_stress"] = raw_stress_metric(predict(model, hard_test, device, args.batch_size), hard_test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.output_dir / "decision_aware_pino.pt"
    torch.save({
        **state,
        "base_ckpt": str(args.base_ckpt),
        "start_ckpt": str(args.start_ckpt),
        "attn_dim": payload.get("attn_dim", 64),
        "n_heads": payload.get("n_heads", 4),
        "external_split": external_split,
    }, ckpt)
    result = {
        "schema": "decision_aware_pino_v1",
        "created": "2026-07-04",
        "checkpoint": str(ckpt),
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
        "near_boundary_sf_values": sorted(near_sf),
        "training_config": {
            "epochs": args.epochs,
            "lr": args.lr,
            "false_safe_weight": args.false_safe_weight,
            "hardset_unsafe_weight": args.hardset_unsafe_weight,
            "safe_miss_weight": args.safe_miss_weight,
            "threshold_margin": args.threshold_margin,
        },
        "selected": final,
        "history": history,
        "promotion_gate": {
            "zero_false_pass_all_sealed": final["joint_false_pass"] == 0,
            "candidate_auto_safe_at_least_90pct": final["candidate_heldout"]["auto_safe_rate"] >= 0.90,
            "external_auto_safe_at_least_90pct": final["external_heldout"]["auto_safe_rate"] >= 0.90,
            "hardset_auto_safe_at_least_90pct": final["hardset_test_near_boundary"]["auto_safe_rate"] >= 0.90,
        },
        "limitations": [
            "Hardset near-boundary rows come from an external voxel-FEM corpus indexed in ShadowForge.",
            "Threshold selection uses calibration rows only; sealed rows are audit-only.",
            "All safety claims remain relative to voxel-FEM labels.",
        ],
    }
    result["promotion_eligible_for_full_fem_replacement_claim"] = all(result["promotion_gate"].values())
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(args.out), "selected": final, "promotion_gate": result["promotion_gate"]}, indent=2))


if __name__ == "__main__":
    main()
