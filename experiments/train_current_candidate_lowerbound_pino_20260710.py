#!/usr/bin/env python3
"""Decision-aware PINO training with a calibrated lower bound on log safety factor."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "generation")]

from audit_pino_union_safe_screen_20260708 import load_screen  # noqa: E402
from train_current_candidate_decision_aware_pino_20260708 import (  # noqa: E402
    DEFAULT_SRC, TAU, YIELD_STRESS, build_rows, labels, split_rows,
)
from train_current_candidate_maxcoverage_pino_20260709 import (  # noqa: E402
    load_extra_fem_rows, set_trainable,
)
from train_pino_aug_hardneg import xg  # noqa: E402
from train_pino_safe_side_boundary import SEED  # noqa: E402


def predict_log_sf(model, rows: list[dict], device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            xb = torch.cat([xg(model, row["density"], row["force"], device) for row in batch], dim=0)
            pred_log_vm = model._scale_forward(xb).squeeze(-1).reshape(-1)
            result.extend((math.log10(YIELD_STRESS) - pred_log_vm).detach().cpu().tolist())
    return np.asarray(result, dtype=float)


def residual_quantile(pred: np.ndarray, rows: list[dict], alpha: float) -> float:
    truth = np.log10(np.asarray([float(row["sf"]) for row in rows], dtype=float))
    # A positive residual means the network overestimated safety.  Subtracting
    # this calibrated upper quantile gives a conservative lower confidence bound.
    return float(np.quantile(pred - truth, 1.0 - alpha, method="higher"))


def lowerbound_metrics(pred: np.ndarray, rows: list[dict], residual_q: float) -> dict:
    truth_safe = labels(rows)
    lower = pred - residual_q
    accept = lower >= math.log10(TAU)
    return {
        "n": int(len(rows)),
        "safe": int(truth_safe.sum()),
        "unsafe": int((~truth_safe).sum()),
        "auto_safe": int(accept.sum()),
        "auto_safe_true": int((accept & truth_safe).sum()),
        "false_safe": int((accept & ~truth_safe).sum()),
        "auto_safe_rate_over_safe": float((accept & truth_safe).sum() / max(int(truth_safe.sum()), 1)),
        "autonomous_rate": float(accept.mean()) if len(rows) else 0.0,
        "residual_q": float(residual_q),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=ROOT / DEFAULT_SRC)
    parser.add_argument("--extra-index", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--trainable", choices=["rear_scale", "full_scale"], default="full_scale")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source = json.loads(args.src.read_text())
    base_rows = build_rows(source)
    splits = split_rows(base_rows, 0.60, 0.20)
    train_groups = {row["group"] for row in splits["train"]}
    extra_rows = [row for row in load_extra_fem_rows(args.extra_index) if row["group"] in train_groups]
    train_rows = splits["train"] + extra_rows
    if not extra_rows:
        raise RuntimeError("No extra FEM rows joined the training split; aborting instead of training on stale data.")

    model = load_screen(
        device,
        ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt",
        ROOT / "checkpoints/pino_safe_accept90_cal300_m08_20260704/safe_side_external_hardneg_scale.pt",
    )
    trainable = set_trainable(model, args.trainable)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=2e-5)
    rng = np.random.default_rng(SEED)
    target_train = torch.as_tensor([math.log10(float(row["sf"])) for row in train_rows], dtype=torch.float32, device=device)
    best = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        order = rng.permutation(len(train_rows))
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            batch = [train_rows[int(i)] for i in idx]
            xb = torch.cat([xg(model, row["density"], row["force"], device) for row in batch], dim=0)
            target = target_train[torch.as_tensor(idx, device=device)]
            pred = math.log10(YIELD_STRESS) - model._scale_forward(xb).squeeze(-1).reshape(-1)
            error = target - pred
            # Lower-quantile regression discourages unsafe overestimation.
            pinball = torch.maximum(0.10 * error, -0.90 * error).mean()
            near = torch.abs(target - math.log10(TAU)) <= 0.16
            near_loss = F.smooth_l1_loss(pred[near], target[near]) if near.any() else pred.sum() * 0.0
            unsafe = target < math.log10(TAU)
            false_safe_penalty = F.relu(pred[unsafe] - target[unsafe]).pow(2).mean() if unsafe.any() else pred.sum() * 0.0
            loss = pinball + 1.5 * near_loss + 8.0 * false_safe_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)

        cal_pred = predict_log_sf(model, splits["calibration"], device, args.batch_size)
        q = residual_quantile(cal_pred, splits["calibration"], args.alpha)
        cal = lowerbound_metrics(cal_pred, splits["calibration"], q)
        test_pred = predict_log_sf(model, splits["sealed_test"], device, args.batch_size)
        test = lowerbound_metrics(test_pred, splits["sealed_test"], q)
        record = {"epoch": epoch, "train_loss": total / len(train_rows), "calibration": cal, "sealed_test": test}
        history.append(record)
        print(json.dumps(record), flush=True)
        key = (cal["false_safe"], -cal["auto_safe_true"], q)
        if best is None or key < best[0]:
            best = (key, epoch, q, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                if k.startswith(("scale_", "attn_scale_head"))
            })

    assert best is not None
    model.load_state_dict(best[3], strict=False)
    cal_pred = predict_log_sf(model, splits["calibration"], device, args.batch_size)
    q = residual_quantile(cal_pred, splits["calibration"], args.alpha)
    test_pred = predict_log_sf(model, splits["sealed_test"], device, args.batch_size)
    result = {
        "schema": "current_candidate_calibrated_lowerbound_pino_v1",
        "source": str(args.src), "extra_index": str(args.extra_index),
        "extra_train_rows": len(extra_rows), "split_sizes": {k: len(v) for k, v in splits.items()},
        "best_epoch": best[1], "alpha": args.alpha, "calibration": lowerbound_metrics(cal_pred, splits["calibration"], q),
        "sealed_test": lowerbound_metrics(test_pred, splits["sealed_test"], q), "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "lowerbound_scale_pino.pt"
    torch.save({"state_dict": best[3], "best_epoch": best[1], "residual_q": q, "alpha": args.alpha}, checkpoint)
    result["checkpoint"] = str(checkpoint)
    result["promotion_gate"] = {
        "sealed_zero_false_safe": result["sealed_test"]["false_safe"] == 0,
        "sealed_auto_safe_at_least_90pct": result["sealed_test"]["auto_safe_rate_over_safe"] >= 0.90,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(args.out), "promotion": result["promotion_gate"]}), flush=True)


if __name__ == "__main__":
    main()
