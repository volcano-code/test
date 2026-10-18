#!/usr/bin/env python3
"""Train a scalar FEM-factor and safety-decision PINO on frozen task splits.

This trainer deliberately contains no continuum PDE residual.  Its supervision
is the fresh voxel Hex8 FEM safety factor in the dataset manifest.  Nodal
displacements are retained in the dataset for a future FEM-discrete field loss,
but are not misused as voxel-average field labels here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_current_candidate_decision_aware_pino_20260708 import (  # noqa: E402
    YIELD_STRESS,
    load_scale_model,
)
from train_decision_aware_pino import DecisionHead, decision_features  # noqa: E402
from train_pino_aug_hardneg import SCALE_BRANCH, xg  # noqa: E402

TAU = 1.3
SEED = 20260711


class ArrayCache:
    def __init__(self, root: Path, capacity: int):
        self.root = root
        self.capacity = capacity
        self.values: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(self, row: dict) -> tuple[np.ndarray, np.ndarray]:
        key = str(row["path"])
        if key in self.values:
            self.values.move_to_end(key)
            return self.values[key]
        with np.load(self.root / key) as payload:
            value = (payload["density"].astype(np.float32), payload["force"].astype(np.float32))
        self.values[key] = value
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return value


def labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([float(row["sf"]) >= TAU for row in rows], dtype=bool)


def evaluate(scores: np.ndarray, rows: list[dict], threshold: float) -> dict:
    truth = labels(rows)
    accept = scores >= threshold
    return {
        "n": len(rows),
        "safe": int(truth.sum()),
        "unsafe": int((~truth).sum()),
        "auto_safe": int(accept.sum()),
        "auto_safe_true": int((accept & truth).sum()),
        "false_safe": int((accept & ~truth).sum()),
        "auto_safe_coverage": float((accept & truth).sum() / max(int(truth.sum()), 1)),
        "autonomous_rate": float(accept.mean()) if len(rows) else 0.0,
        "classification_accuracy": float(((scores >= 0.5) == truth).mean()) if len(rows) else 0.0,
    }


def regression_metrics(log_vms: np.ndarray, rows: list[dict]) -> dict:
    true_log_sf = np.log10(np.maximum(np.asarray([float(row["sf"]) for row in rows]), 1e-12))
    pred_log_sf = math.log10(YIELD_STRESS) - np.asarray(log_vms, dtype=float)
    return {
        "log_sf_mae": float(np.mean(np.abs(pred_log_sf - true_log_sf))),
        "log_sf_rmse": float(np.sqrt(np.mean((pred_log_sf - true_log_sf) ** 2))),
        "log_sf_pearson": float(np.corrcoef(pred_log_sf, true_log_sf)[0, 1]) if len(rows) > 1 else 0.0,
    }


def choose_zero_false_safe_threshold(scores: np.ndarray, rows: list[dict]) -> float:
    unsafe = scores[~labels(rows)]
    return float(np.max(unsafe) + 1e-6) if unsafe.size else float(np.min(scores) - 1e-6)


@torch.no_grad()
def predict(model, head, rows: list[dict], cache: ArrayCache, device: torch.device, batch_size: int, stage: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    head.eval()
    scores, log_vms = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        xb = torch.cat([xg(model, *cache.get(row), device) for row in batch], dim=0)
        pred_log_vm = model._scale_forward(xb).squeeze(-1).reshape(-1)
        pred_log_sf = math.log10(YIELD_STRESS) - pred_log_vm
        if stage == "regression":
            # A monotone provisional safety score; ACS is not calibrated in stage A.
            score = torch.sigmoid(4.0 * (pred_log_sf - math.log10(TAU)))
        else:
            score = torch.sigmoid(head(decision_features(model, xb, pred_log_sf)))
        scores.extend(score.detach().cpu().tolist())
        log_vms.extend(pred_log_vm.detach().cpu().tolist())
    return np.asarray(scores, dtype=float), np.asarray(log_vms, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stage", choices=("regression", "decision", "joint"), default="joint")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Stage-A checkpoint used to initialize stage-B decision training.")
    parser.add_argument("--tasks-per-batch", type=int, default=2)
    parser.add_argument("--max-rows-per-task", type=int, default=16,
                        help="Bound GPU memory while retaining task-grouped supervision.")
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--cache-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=8e-6)
    parser.add_argument("--false-safe-weight", type=float, default=30.0)
    parser.add_argument("--safe-margin-weight", type=float, default=10.0)
    parser.add_argument("--rank-weight", type=float, default=1.5)
    parser.add_argument("--unfreeze-shared-decoder", action="store_true",
                        help="Jointly adapt the late shared decoder with the scale branch.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--start-ckpt", type=Path, default=ROOT / "checkpoints/v29c_v27_pinball_s2606/best_scale_sf.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = json.loads(args.dataset.read_text())
    root = args.dataset.parent
    rows_by_split = {name: [row for row in manifest["rows"] if row["split"] == name] for name in ("train", "calibration", "sealed_test")}
    if any(not rows_by_split[name] for name in rows_by_split):
        raise ValueError("dataset must contain train, calibration, and sealed_test rows")
    cache = ArrayCache(root, args.cache_size)

    model = load_scale_model(device, args.base_ckpt, args.start_ckpt)
    if args.resume is not None:
        blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        for name, state in blob.get("scale_branch_state_dict", {}).items():
            getattr(model, name).load_state_dict(state)
        for name, state in blob.get("shared_decoder_state_dict", {}).items():
            getattr(model, name).load_state_dict(state)
        if "scale_head_v2_state_dict" in blob:
            model.scale_head_v2.load_state_dict(blob["scale_head_v2_state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    params = []
    for name in SCALE_BRANCH:
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(True)
            params.append(parameter)
    decoder_names = ("up2", "dec2_proj", "dec2", "up1", "dec1_proj", "dec1")
    if args.unfreeze_shared_decoder:
        # These late decoder blocks retain spatial stress cues while avoiding a full-network reset.
        for name in decoder_names:
            for parameter in getattr(model, name).parameters():
                parameter.requires_grad_(True)
                params.append(parameter)
    for parameter in model.scale_head_v2.parameters():
        parameter.requires_grad_(True)
        params.append(parameter)
    with torch.no_grad():
        density, force = cache.get(rows_by_split["train"][0])
        probe = xg(model, density, force, device)
        pred = model._scale_forward(probe).squeeze(-1).reshape(-1)
        feat_dim = decision_features(model, probe, math.log10(YIELD_STRESS) - pred).shape[1]
    head = DecisionHead(feat_dim).to(device)
    if args.resume is not None and "decision_head_state_dict" in blob and args.stage != "regression":
        head.load_state_dict(blob["decision_head_state_dict"])
    if args.stage != "regression":
        params.extend(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    log_yield, log_tau = math.log10(YIELD_STRESS), math.log10(TAU)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows_by_split["train"]:
        by_task[row["task"]].append(row)

    def training_rows(task_rows: list[dict]) -> list[dict]:
        """Deterministic task-local subsampling for variable-size candidate pools.

        Evaluation remains on every calibration/sealed candidate. During training,
        each update retains unsafe examples first, then fills the fixed memory
        budget with safe rows. This avoids an oversized candidate trajectory
        causing OOM while preserving the safety decision objective.
        """
        limit = int(args.max_rows_per_task)
        if limit <= 0 or len(task_rows) <= limit:
            return task_rows
        unsafe = [row for row in task_rows if float(row["sf"]) < TAU]
        safe = [row for row in task_rows if float(row["sf"]) >= TAU]
        unsafe = sorted(unsafe, key=lambda row: abs(math.log10(max(float(row["sf"]), 1e-12)) - log_tau))
        safe = sorted(safe, key=lambda row: abs(math.log10(max(float(row["sf"]), 1e-12)) - log_tau))
        # Near-threshold unsafe rows are never silently displaced by easy safe rows.
        kept_unsafe = unsafe[:limit]
        remain = max(limit - len(kept_unsafe), 0)
        return kept_unsafe + safe[:remain]

    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        losses, n_seen = 0.0, 0
        task_order = rng.permutation(sorted(by_task))
        for start in range(0, len(task_order), args.tasks_per_batch):
            batch = [
                row
                for task in task_order[start:start + args.tasks_per_batch]
                for row in training_rows(by_task[str(task)])
            ]
            xb = torch.cat([xg(model, *cache.get(row), device) for row in batch], dim=0)
            sf = torch.as_tensor([float(row["sf"]) for row in batch], dtype=torch.float32, device=device)
            y = (sf >= TAU).float()
            target_log_vm = torch.as_tensor([math.log10(YIELD_STRESS / max(float(row["sf"]), 1e-12)) for row in batch], dtype=torch.float32, device=device)
            pred_log_vm = model._scale_forward(xb).squeeze(-1).reshape(-1)
            pred_log_sf = log_yield - pred_log_vm
            logits = head(decision_features(model, xb, pred_log_sf))
            prob = torch.sigmoid(logits)
            near = (torch.abs(torch.log10(sf) - log_tau) <= 0.35).float()
            clear_safe = y * (sf >= 2.0 * TAU).float()
            reg = F.smooth_l1_loss(pred_log_vm, target_log_vm, reduction="none")
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            scale_false_safe = (1.0 - y) * F.relu(pred_log_sf - log_tau + 0.03).pow(2)
            scale_safe_margin = clear_safe * F.relu(log_tau + 0.18 - pred_log_sf).pow(2)
            decision_false_safe = (1.0 - y) * F.relu(prob - 0.02).pow(2)
            decision_safe_margin = clear_safe * F.relu(0.92 - prob).pow(2)
            rank_terms = []
            for task in sorted(set(row["task"] for row in batch)):
                idx = torch.as_tensor([i for i, row in enumerate(batch) if row["task"] == task], device=device)
                safe_idx, unsafe_idx = idx[y[idx] > 0.5], idx[y[idx] < 0.5]
                if safe_idx.numel() and unsafe_idx.numel():
                    score_gap = prob[safe_idx].reshape(-1, 1) - prob[unsafe_idx].reshape(1, -1)
                    rank_terms.append(F.relu(0.25 - score_gap).pow(2).mean())
            rank = torch.stack(rank_terms).mean() if rank_terms else prob.sum() * 0.0
            sf_rank_terms = []
            for task in sorted(set(row["task"] for row in batch)):
                idx = torch.as_tensor([i for i, row in enumerate(batch) if row["task"] == task], device=device)
                safe_idx, unsafe_idx = idx[y[idx] > 0.5], idx[y[idx] < 0.5]
                if safe_idx.numel() and unsafe_idx.numel():
                    gap = pred_log_sf[safe_idx].reshape(-1, 1) - pred_log_sf[unsafe_idx].reshape(1, -1)
                    sf_rank_terms.append(F.relu(0.10 - gap).pow(2).mean())
            sf_rank = torch.stack(sf_rank_terms).mean() if sf_rank_terms else pred_log_sf.sum() * 0.0
            # The current candidate domain is safety-heavy; balance the decision term so
            # unsafe near-threshold rows cannot be absorbed by the majority class.
            unsafe_ratio = float(y.numel()) / max(float((1.0 - y).sum().item()), 1.0)
            class_weight = torch.where(y > 0.5, torch.ones_like(y), torch.full_like(y, unsafe_ratio))
            weights = class_weight * (1.0 + 3.0 * near + 2.0 * clear_safe)
            if args.stage == "regression":
                loss = torch.mean(weights * reg) + args.rank_weight * sf_rank
            elif args.stage == "decision":
                loss = torch.mean(
                    weights * (0.25 * reg + 3.0 * bce)
                    + args.false_safe_weight * (scale_false_safe + decision_false_safe)
                    + args.safe_margin_weight * (scale_safe_margin + decision_safe_margin)
                ) + args.rank_weight * rank
            else:
                loss = torch.mean(
                    weights * (reg + 3.0 * bce)
                    + args.false_safe_weight * (scale_false_safe + decision_false_safe)
                    + args.safe_margin_weight * (scale_safe_margin + decision_safe_margin)
                ) + args.rank_weight * (rank + sf_rank)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses += float(loss.detach().cpu()) * len(batch)
            n_seen += len(batch)

        predictions = {name: predict(model, head, rows, cache, device, args.eval_batch_size, args.stage) for name, rows in rows_by_split.items()}
        threshold = choose_zero_false_safe_threshold(predictions["calibration"][0], rows_by_split["calibration"])
        record = {
            "epoch": epoch,
            "train_loss": losses / max(n_seen, 1),
            "threshold": threshold,
            **{name: {**evaluate(predictions[name][0], rows, threshold),
                      **regression_metrics(predictions[name][1], rows)} for name, rows in rows_by_split.items()},
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        key = ((record["calibration"]["log_sf_mae"],) if args.stage == "regression" else
               (record["calibration"]["false_safe"], -record["calibration"]["auto_safe_true"], -record["train"]["auto_safe_true"]))
        if best is None or key < best[0]:
            best = (key, {"epoch": epoch, "threshold": threshold, "record": record,
                          "scale_branch_state_dict": {name: {k: v.detach().cpu().clone() for k, v in getattr(model, name).state_dict().items()} for name in SCALE_BRANCH},
                          "scale_head_v2_state_dict": {k: v.detach().cpu().clone() for k, v in model.scale_head_v2.state_dict().items()},
                          "shared_decoder_state_dict": {name: {k: v.detach().cpu().clone() for k, v in getattr(model, name).state_dict().items()} for name in decoder_names if args.unfreeze_shared_decoder},
                          "decision_head_state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, "feat_dim": feat_dim})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.output_dir / "taskgrouped_fem_decision_pino.pt"
    torch.save(best[1], ckpt)
    result = {
        "schema": "taskgrouped_fem_decision_pino_v1",
        "stage": args.stage,
        "dataset": str(args.dataset),
        "checkpoint": str(ckpt),
        "supervision": "fresh voxel Hex8 FEM safety factor and decision labels; no continuum PDE residual",
        "selection_protocol": "checkpoint and threshold selected on calibration only; sealed test is audit only",
        "rows": {name: len(rows) for name, rows in rows_by_split.items()},
        "safe_unsafe": {name: {"safe": int(labels(rows).sum()), "unsafe": int((~labels(rows)).sum())} for name, rows in rows_by_split.items()},
        "selected": best[1]["record"],
        "history": history,
        "promotion_gate": {"sealed_zero_false_safe": best[1]["record"]["sealed_test"]["false_safe"] == 0,
                           "sealed_auto_safe_85pct": best[1]["record"]["sealed_test"]["auto_safe_coverage"] >= 0.85},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "selected": result["selected"], "promotion_gate": result["promotion_gate"]}, indent=2))


if __name__ == "__main__":
    main()
