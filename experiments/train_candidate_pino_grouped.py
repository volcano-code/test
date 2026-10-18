#!/usr/bin/env python3
"""Train a candidate-domain scale PINO with group-isolated evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from train_pino_aug_hardneg import SCALE_BRANCH, load_geoattn_model, xg
from structural.analysis.load_generator import LoadGenerator

YIELD_STRESS = 50e6
TAU = 1.3
SEED = 20260703


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows() -> list[dict]:
    rows = []
    generator = LoadGenerator()
    for branch, path in (("free", ROOT / "data/aug.npz"), ("fixed", ROOT / "data/aug_fixed.npz")):
        payload = np.load(path)
        if len(payload["dens"]) != 640:
            raise ValueError(f"Expected 640 rows in {path}; got {len(payload['dens'])}")
        for index in range(len(payload["dens"])):
            density = payload["dens"][index].astype(np.float32)
            force = generator.standing_load(density)["force"] * float(payload["mult"][index])
            rows.append({
                "branch": branch,
                "group": index // 16,
                "density": density,
                "force": force.astype(np.float32),
                "log_vm": float(np.log10(max(float(payload["vm"][index]), 1e-12))),
                "sf": float(payload["sf"][index]),
            })
    return rows


def split_groups() -> dict[str, list[int]]:
    groups = np.arange(40)
    np.random.default_rng(SEED).shuffle(groups)
    return {
        "train": sorted(int(value) for value in groups[:24]),
        "selection_validation": sorted(int(value) for value in groups[24:30]),
        "conformal_calibration": sorted(int(value) for value in groups[30:35]),
        "heldout_test": sorted(int(value) for value in groups[35:]),
    }


def subset(rows: list[dict], groups: list[int]) -> list[dict]:
    selected = set(groups)
    return [row for row in rows if row["group"] in selected]


@torch.no_grad()
def predict(model, rows: list[dict], device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        inputs = torch.cat([xg(model, row["density"], row["force"], device) for row in batch], dim=0)
        output = model._scale_forward(inputs).squeeze(-1)
        predictions.extend(float(value) for value in output.detach().cpu().reshape(-1))
    return np.asarray(predictions, dtype=float)


def metrics(pred_log_vm: np.ndarray, rows: list[dict]) -> dict:
    true_log_vm = np.asarray([row["log_vm"] for row in rows], dtype=float)
    true_sf = np.asarray([row["sf"] for row in rows], dtype=float)
    pred_sf = YIELD_STRESS / np.maximum(10.0 ** pred_log_vm, 1e-12)
    pred_safe = pred_sf >= TAU
    true_safe = true_sf >= TAU
    false_pass = pred_safe & ~true_safe
    true_pass = pred_safe & true_safe
    return {
        "n": len(rows),
        "safe": int(true_safe.sum()),
        "unsafe": int((~true_safe).sum()),
        "log_vm_mae": float(np.mean(np.abs(pred_log_vm - true_log_vm))),
        "log_vm_corr": float(np.corrcoef(pred_log_vm, true_log_vm)[0, 1]),
        "raw_safe_selected": int(pred_safe.sum()),
        "raw_safe_precision": float(true_pass.sum() / max(pred_safe.sum(), 1)),
        "raw_safe_coverage": float(true_pass.sum() / max(true_safe.sum(), 1)),
        "false_pass_count": int(false_pass.sum()),
    }


def finite_quantile(values: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    rank = min(int(np.ceil((len(ordered) + 1) * (1 - alpha))) - 1, len(ordered) - 1)
    return float(ordered[max(rank, 0)])


def acs_metrics(cal_pred_vm, cal_rows, test_pred_vm, test_rows, alpha: float) -> dict:
    cal_pred_sf = math.log10(YIELD_STRESS) - cal_pred_vm
    cal_true_sf = np.log10(np.asarray([row["sf"] for row in cal_rows], dtype=float))
    q_safe = finite_quantile(cal_pred_sf - cal_true_sf, alpha)
    q_unsafe = finite_quantile(cal_true_sf - cal_pred_sf, alpha)
    test_pred_sf = math.log10(YIELD_STRESS) - test_pred_vm
    test_true_sf = np.asarray([row["sf"] for row in test_rows], dtype=float)
    threshold = math.log10(TAU)
    cert_safe = test_pred_sf - q_safe >= threshold
    cert_unsafe = test_pred_sf + q_unsafe < threshold
    true_safe = test_true_sf >= TAU
    decided = cert_safe | cert_unsafe
    return {
        "alpha": alpha,
        "q_safe_log10": q_safe,
        "q_unsafe_log10": q_unsafe,
        "certified_safe": int(cert_safe.sum()),
        "certified_unsafe": int(cert_unsafe.sum()),
        "fallback": int((~decided).sum()),
        "autonomous_rate": float(decided.mean()),
        "false_safe": int((cert_safe & ~true_safe).sum()),
        "false_unsafe": int((cert_unsafe & true_safe).sum()),
    }


def state_to_cpu(module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--geo-ckpt", type=Path, default=ROOT / "checkpoints/v40_geoattn/best.pt")
    parser.add_argument("--current-hardneg", type=Path, default=ROOT / "checkpoints/pino_aug_hardneg_20260624/scale_geoattn_e8.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/pino_candidate_grouped_20260703")
    parser.add_argument("--result", type=Path, default=ROOT / "results/pino_candidate_grouped_20260703.json")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = load_rows()
    splits = split_groups()
    train_rows = subset(rows, splits["train"])
    val_rows = subset(rows, splits["selection_validation"])
    cal_rows = subset(rows, splits["conformal_calibration"])
    test_rows = subset(rows, splits["heldout_test"])

    current, _, _ = load_geoattn_model(device, args.base_ckpt, args.current_hardneg)
    current_test_pred = predict(current, test_rows, device, args.batch_size)
    current_test = metrics(current_test_pred, test_rows)
    del current
    torch.cuda.empty_cache()

    model, head, geo_payload = load_geoattn_model(device, args.base_ckpt, args.geo_ckpt)
    initial_test = metrics(predict(model, test_rows, device, args.batch_size), test_rows)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name in SCALE_BRANCH:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
        trainable.append(parameter)

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    history = []
    best_key = None
    best_epoch = None
    best_state = None
    rng = np.random.default_rng(SEED)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_rows))
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            batch = [train_rows[index] for index in indices]
            inputs = torch.cat([xg(model, row["density"], row["force"], device) for row in batch], dim=0)
            target = torch.as_tensor([row["log_vm"] for row in batch], dtype=torch.float32, device=device)
            unsafe = torch.as_tensor([row["sf"] < TAU for row in batch], dtype=torch.bool, device=device)
            output = model._scale_forward(inputs).squeeze(-1).reshape(-1)
            base = F.smooth_l1_loss(output, target, reduction="none")
            underestimate = F.relu(target - output)
            predicted_sf = YIELD_STRESS / torch.clamp(10.0 ** output, min=1e-12)
            false_pass = unsafe & (predicted_sf >= TAU)
            near_boundary = torch.abs(
                (math.log10(YIELD_STRESS) - target) - math.log10(TAU)
            ) < 0.25
            weights = torch.where(near_boundary, 2.5, 1.0)
            loss = torch.mean(base * weights + 4.0 * underestimate + 8.0 * false_pass * underestimate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)

        val = metrics(predict(model, val_rows, device, args.batch_size), val_rows)
        key = (val["false_pass_count"], -val["raw_safe_coverage"], val["log_vm_mae"])
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_rows), "validation": val})
        print(json.dumps(history[-1]), flush=True)
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                "scale_branch_state_dict": {name: state_to_cpu(getattr(model, name)) for name in SCALE_BRANCH},
                "attn_scale_head_state_dict": state_to_cpu(head),
            }

    for name, state in best_state["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(state)
    head.load_state_dict(best_state["attn_scale_head_state_dict"])
    cal_pred = predict(model, cal_rows, device, args.batch_size)
    test_pred = predict(model, test_rows, device, args.batch_size)
    candidate_test = metrics(test_pred, test_rows)
    acs = acs_metrics(cal_pred, cal_rows, test_pred, test_rows, alpha=0.05)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "best_grouped_candidate.pt"
    torch.save({
        **best_state,
        "attn_dim": geo_payload.get("attn_dim", 64),
        "n_heads": geo_payload.get("n_heads", 4),
        "base_ckpt": str(args.base_ckpt),
        "best_epoch": best_epoch,
        "split_seed": SEED,
        "source": "case-grouped aug+aug_fixed candidate-domain training",
    }, checkpoint)

    promotion = {
        "current_pipeline_modified": False,
        "eligible": bool(
            candidate_test["false_pass_count"] <= current_test["false_pass_count"]
            and candidate_test["raw_safe_coverage"] > current_test["raw_safe_coverage"]
            and acs["false_safe"] == 0
            and acs["false_unsafe"] == 0
            and acs["autonomous_rate"] > 0
        ),
        "rule": "no worse raw false passes, higher raw safe coverage, and nonzero zero-error ACS autonomy on untouched test",
    }
    result = {
        "schema": "candidate_grouped_pino_training_v1",
        "sources": {
            "data/aug.npz": sha256(ROOT / "data/aug.npz"),
            "data/aug_fixed.npz": sha256(ROOT / "data/aug_fixed.npz"),
        },
        "group_definition": "Each source generator emits 16 consecutive rows per silhouette triple (4 geometries x 4 load multipliers); free/fixed rows with the same group stay together.",
        "splits": splits,
        "row_counts": {name: len(subset(rows, groups)) for name, groups in splits.items()},
        "training": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "best_epoch": best_epoch},
        "current_hardneg_test": current_test,
        "initial_geoattn_test": initial_test,
        "candidate_test": candidate_test,
        "candidate_acs_test": acs,
        "history": history,
        "checkpoint": str(checkpoint),
        "promotion": promotion,
        "limitations": [
            "Test evidence covers five silhouette-triple groups (160 correlated load/geometry rows).",
            "Labels are voxel-FEM outputs, not physical-test measurements.",
            "The scale head is experimental and is not used by the current authoritative field-based pipeline.",
        ],
    }
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    report = ROOT / "results/PINO_CANDIDATE_RETRAIN_20260703.md"
    report.write_text("\n".join([
        "# Candidate PINO Retraining",
        "",
        f"- Best epoch: {best_epoch}/{args.epochs}.",
        f"- Current hard-negative checkpoint test: false pass={current_test['false_pass_count']}, safe coverage={current_test['raw_safe_coverage']:.3f}, correlation={current_test['log_vm_corr']:.3f}.",
        f"- New grouped checkpoint test: false pass={candidate_test['false_pass_count']}, safe coverage={candidate_test['raw_safe_coverage']:.3f}, correlation={candidate_test['log_vm_corr']:.3f}.",
        f"- ACS on untouched test: autonomy={acs['autonomous_rate']:.3f}, false safe={acs['false_safe']}, false unsafe={acs['false_unsafe']}.",
        f"- Promotion eligible: {promotion['eligible']}; authoritative pipeline modified: false.",
    ]) + "\n")
    print(json.dumps({"result": str(args.result), "report": str(report), "checkpoint": str(checkpoint), "promotion": promotion, "candidate_test": candidate_test, "acs": acs}, indent=2))


if __name__ == "__main__":
    main()
