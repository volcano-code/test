#!/usr/bin/env python3
"""Train a task-grouped decision-aware PINO on the current candidate domain.

This experiment targets the current 100-case / 700-candidate domain used by the
paper tables. It does not overwrite deployment checkpoints.  Splits are made
by the ordered silhouette triple, so all fixed/free variants and all carving
states of one task remain in exactly one of train/calibration/sealed-test.
"""
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

from candidate_selection_batch import make_candidates  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from structural.analysis.load_generator import LoadGenerator  # noqa: E402
from structural.models.load_model import load_surrogate  # noqa: E402
from train_decision_aware_pino import DecisionHead, decision_features, predict_decision  # noqa: E402
from train_pino_aug_hardneg import SCALE_BRANCH, xg  # noqa: E402
from train_pino_safe_side_boundary import SEED, TAU  # noqa: E402

YIELD_STRESS = 50e6
DEFAULT_SRC = ROOT / "results/candidate_selection_100case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"


def load_scale_model(device: torch.device, base_ckpt: Path, checkpoint: Path):
    """Load a scale-head checkpoint without requiring the legacy GeoAttn head."""
    model, _ = load_surrogate(str(base_ckpt), device)
    model.enable_enhanced_scale_head()
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    for name, state in payload.get("scale_branch_state_dict", {}).items():
        getattr(model, name).load_state_dict(state)
    if "scale_head_v2_state_dict" in payload:
        model.scale_head_v2.load_state_dict(payload["scale_head_v2_state_dict"])
    return model


def load_masks(triple: list[str], resolution: int) -> list[np.ndarray]:
    sil_dir = ROOT / "data/silhouettes/bench3"
    paths = [sil_dir / f"{name}.png" for name in triple]
    tensors, _ = load_data_from_list(
        [str(p) for p in paths],
        True,
        (resolution, resolution),
        torch.device("cpu"),
        3,
        False,
    )
    return [np.asarray(x).squeeze().astype(np.float32) for x in tensors]


def build_rows(source: dict) -> list[dict]:
    generator = LoadGenerator()
    rows = []
    for case in source["cases"]:
        resolution = int(source["resolution"])
        masks = load_masks(case["triple"], resolution)
        candidates, _ = make_candidates(masks, resolution, case["branch"] == "fixed")
        task = "|".join(case["triple"])
        recorded_by_name = {row["candidate"]: row for row in case["candidates"]}
        for name, volume in candidates.items():
            recorded = recorded_by_name.get(name)
            if recorded is None:
                continue
            sf = float(recorded["sf_fem"])
            force = generator.standing_load(volume.astype(np.float32))["force"].astype(np.float32)
            rows.append({
                "source": "current_candidate_domain",
                "group": f"{case['branch']}:{case['case_id']}",
                "task": task,
                "branch": case["branch"],
                "case_id": int(case["case_id"]),
                "candidate": name,
                "density": volume.astype(np.float32),
                "force": force,
                "sf": sf,
                "log_vm": float(math.log10(max(YIELD_STRESS / max(sf, 1e-12), 1e-12))),
                "fem_safe": bool(recorded["fem_safe"]),
                "fem_elapsed_s": float(recorded["fem_elapsed_s"]),
                "mean_iou": float(recorded["mean_iou"]),
                "mean_dice": float(recorded["mean_dice"]),
            })
    return rows


def split_rows(rows: list[dict], train_frac: float, cal_frac: float) -> dict[str, list[dict]]:
    rng = np.random.default_rng(SEED)
    tasks = sorted({row["task"] for row in rows})
    rng.shuffle(tasks)
    n_train = int(round(len(tasks) * train_frac))
    n_cal = int(round(len(tasks) * cal_frac))
    train_groups = set(tasks[:n_train])
    cal_groups = set(tasks[n_train:n_train + n_cal])
    test_groups = set(tasks[n_train + n_cal:])
    return {
        "train": [row for row in rows if row["task"] in train_groups],
        "calibration": [row for row in rows if row["task"] in cal_groups],
        "sealed_test": [row for row in rows if row["task"] in test_groups],
    }


def labels(rows: list[dict]) -> np.ndarray:
    return np.asarray([row["sf"] >= TAU for row in rows], dtype=bool)


def choose_zero_fp_threshold(scores: np.ndarray, rows: list[dict], margin: float) -> float:
    truth = labels(rows)
    unsafe_scores = scores[~truth]
    if len(unsafe_scores) == 0:
        return float(np.min(scores) - 1e-6 + margin)
    return float(np.max(unsafe_scores) + 1e-6 + margin)


def stratum_key(row: dict) -> str:
    cand = str(row["candidate"])
    if "hull" in cand:
        ctype = "hull"
    elif "bridge" in cand or "repair" in cand or "export" in cand:
        ctype = "repaired"
    elif "carve" in cand:
        ctype = "carve"
    else:
        ctype = "other"
    iou_bin = "hi_iou" if float(row.get("mean_iou", 0.0)) >= 0.97 else "lo_iou"
    return f"{row['branch']}:{ctype}:{iou_bin}"


def choose_stratified_thresholds(scores: np.ndarray, rows: list[dict], margin: float, min_unsafe: int = 3) -> dict[str, float]:
    global_threshold = choose_zero_fp_threshold(scores, rows, margin)
    out = {"__global__": global_threshold}
    keys = np.asarray([stratum_key(row) for row in rows], dtype=object)
    for key in sorted(set(keys)):
        idx = keys == key
        truth = labels([row for row, keep in zip(rows, idx) if keep])
        if int((~truth).sum()) < min_unsafe:
            continue
        out[key] = choose_zero_fp_threshold(scores[idx], [row for row, keep in zip(rows, idx) if keep], margin)
    return out


def metric_stratified(scores: np.ndarray, rows: list[dict], thresholds: dict[str, float]) -> dict:
    effective = np.asarray([thresholds.get(stratum_key(row), thresholds["__global__"]) for row in rows], dtype=float)
    return metric(scores, rows, 0.0) if False else metric_with_accept(scores >= effective, rows)


def metric_with_accept(accept: np.ndarray, rows: list[dict]) -> dict:
    truth = labels(rows)
    fem_times = np.asarray([float(row["fem_elapsed_s"]) for row in rows], dtype=float)
    fallback = ~accept
    return {
        "n": int(len(rows)),
        "safe": int(truth.sum()),
        "unsafe": int((~truth).sum()),
        "auto_safe": int(accept.sum()),
        "auto_safe_true": int((accept & truth).sum()),
        "false_safe": int((accept & ~truth).sum()),
        "auto_safe_rate_over_safe": float((accept & truth).sum() / max(int(truth.sum()), 1)),
        "autonomous_rate": float(accept.mean()) if len(rows) else 0.0,
        "full_fem_time_s": float(fem_times.sum()),
        "fallback_fem_time_s": float(fem_times[fallback].sum()),
    }


def top_unsafe_scores(scores: np.ndarray, rows: list[dict], k: int = 8) -> list[dict]:
    truth = labels(rows)
    unsafe_idx = np.where(~truth)[0]
    if len(unsafe_idx) == 0:
        return []
    order = unsafe_idx[np.argsort(scores[unsafe_idx])[::-1]][:k]
    out = []
    for idx in order:
        row = rows[int(idx)]
        out.append({
            "score": float(scores[int(idx)]),
            "sf": float(row["sf"]),
            "group": row["group"],
            "branch": row["branch"],
            "case_id": int(row["case_id"]),
            "candidate": row["candidate"],
            "mean_iou": float(row["mean_iou"]),
            "mean_dice": float(row["mean_dice"]),
            "stratum": stratum_key(row),
        })
    return out


def metric(scores: np.ndarray, rows: list[dict], threshold: float) -> dict:
    truth = labels(rows)
    accept = scores >= threshold
    return metric_with_accept(accept, rows)


def pairwise_rank_loss(prob: torch.Tensor, y_safe: torch.Tensor, sf: torch.Tensor, tasks: list[str], max_pairs: int = 128) -> torch.Tensor:
    """Rank safe above unsafe candidates only within the same silhouette task."""
    terms = []
    for task in sorted(set(tasks)):
        idx = torch.as_tensor([i for i, value in enumerate(tasks) if value == task], device=prob.device)
        if idx.numel() < 2:
            continue
        terms.append(_pairwise_rank_loss_one_task(prob[idx], y_safe[idx], sf[idx], max_pairs))
    return torch.stack(terms).mean() if terms else prob.sum() * 0.0


def _pairwise_rank_loss_one_task(prob: torch.Tensor, y_safe: torch.Tensor, sf: torch.Tensor, max_pairs: int = 128) -> torch.Tensor:
    safe_idx = torch.nonzero(y_safe > 0.5, as_tuple=False).flatten()
    unsafe_idx = torch.nonzero(y_safe < 0.5, as_tuple=False).flatten()
    if safe_idx.numel() == 0 or unsafe_idx.numel() == 0:
        return prob.sum() * 0.0
    safe_margin = torch.log10(sf[safe_idx]).detach()
    unsafe_margin = torch.log10(sf[unsafe_idx]).detach()
    # Prefer unambiguous safe rows for ranking; near-boundary rows are mostly for defer.
    order_safe = torch.argsort(safe_margin, descending=True)[: min(safe_idx.numel(), max_pairs)]
    order_unsafe = torch.argsort(unsafe_margin, descending=True)[: min(unsafe_idx.numel(), max_pairs)]
    s = safe_idx[order_safe]
    u = unsafe_idx[order_unsafe]
    diff = prob[s].reshape(-1, 1) - prob[u].reshape(1, -1)
    return F.relu(0.25 - diff).pow(2).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--tasks-per-batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--false-safe-weight", type=float, default=42.0)
    parser.add_argument("--safe-miss-weight", type=float, default=6.0)
    parser.add_argument("--scale-false-safe-weight", type=float, default=18.0)
    parser.add_argument("--scale-safe-margin-weight", type=float, default=6.0)
    parser.add_argument("--threshold-margin", type=float, default=0.02)
    parser.add_argument("--threshold-margins", default="")
    parser.add_argument("--threshold-mode", choices=["global", "stratified"], default="global")
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.60)
    parser.add_argument("--cal-frac", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-ckpt", type=Path, default=ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt")
    parser.add_argument("--start-ckpt", type=Path, default=ROOT / "checkpoints/v29c_v27_pinball_s2606/best_scale_sf.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "results/current_candidate_taskgrouped_decision_pino_20260711.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/current_candidate_taskgrouped_decision_pino_20260711")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source = json.loads(args.src.read_text())
    rows = build_rows(source)
    splits = split_rows(rows, args.train_frac, args.cal_frac)

    model = load_scale_model(device, args.base_ckpt, args.start_ckpt)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for name in SCALE_BRANCH:
        if hasattr(model, name):
            for parameter in getattr(model, name).parameters():
                parameter.requires_grad_(True)
                trainable.append(parameter)
    # v29c's calibrated scalar head is part of the decision model and must be
    # adapted jointly with the scale encoder on the new candidate domain.
    for parameter in model.scale_head_v2.parameters():
        parameter.requires_grad_(True)
        trainable.append(parameter)

    with torch.no_grad():
        probe = xg(model, splits["train"][0]["density"], splits["train"][0]["force"], device)
        pred_log_vm = model._scale_forward(probe).squeeze(-1).reshape(-1)
        feat_dim = decision_features(model, probe, math.log10(YIELD_STRESS) - pred_log_vm).shape[1]
    head = DecisionHead(feat_dim).to(device)
    trainable.extend(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    log_yield = math.log10(YIELD_STRESS)
    log_tau = math.log10(TAU)
    best = None
    history = []

    train_rows = splits["train"]
    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        by_task = {}
        for row in train_rows:
            by_task.setdefault(row["task"], []).append(row)
        task_order = rng.permutation(sorted(by_task))
        total_loss = 0.0
        for start in range(0, len(task_order), args.tasks_per_batch):
            batch = []
            for task in task_order[start:start + args.tasks_per_batch]:
                batch.extend(by_task[str(task)])
            xb = torch.cat([xg(model, row["density"], row["force"], device) for row in batch], dim=0)
            target_log_vm = torch.as_tensor([row["log_vm"] for row in batch], dtype=torch.float32, device=device)
            sf = torch.as_tensor([row["sf"] for row in batch], dtype=torch.float32, device=device)
            y_safe = (sf >= TAU).float()
            pred_log_vm = model._scale_forward(xb).squeeze(-1).reshape(-1)
            pred_log_sf = log_yield - pred_log_vm
            logits = head(decision_features(model, xb, pred_log_sf))
            prob = torch.sigmoid(logits)
            near = (torch.abs(torch.log10(sf) - log_tau) < 0.35).float()
            reg = F.smooth_l1_loss(pred_log_vm, target_log_vm, reduction="none")
            bce = F.binary_cross_entropy_with_logits(logits, y_safe, reduction="none")
            unsafe_accept = (1.0 - y_safe) * F.relu(prob - 0.015).pow(2)
            safe_miss = y_safe * F.relu(0.92 - prob).pow(2)
            clear_safe = y_safe * (sf >= 3.0).float()
            clear_safe_margin = clear_safe * F.relu(0.97 - prob).pow(2)
            # Train the physical safety margin itself, not only a detached
            # classifier.  Unsafe rows are penalized for crossing the FEM
            # safety threshold; clear-safe rows retain a positive margin.
            scale_false_safe = (1.0 - y_safe) * F.relu(pred_log_sf - log_tau + 0.03).pow(2)
            scale_safe_margin = clear_safe * F.relu(log_tau + 0.20 - pred_log_sf).pow(2)
            weights = 1.0 + 4.0 * near + 2.0 * clear_safe
            rank = pairwise_rank_loss(prob, y_safe, sf, [row["task"] for row in batch]) if args.rank_weight > 0 else prob.sum() * 0.0
            loss = torch.mean(
                weights * (reg + 4.0 * bce)
                + args.false_safe_weight * unsafe_accept
                + args.safe_miss_weight * safe_miss
                + 3.0 * clear_safe_margin
                + args.scale_false_safe_weight * scale_false_safe
                + args.scale_safe_margin_weight * scale_safe_margin
            ) + args.rank_weight * rank
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)

        cal_scores, _ = predict_decision(model, head, splits["calibration"], device, args.batch_size)
        margins = [args.threshold_margin]
        if args.threshold_margins.strip():
            margins = [float(x) for x in args.threshold_margins.split(",") if x.strip()]
        test_scores, _ = predict_decision(model, head, splits["sealed_test"], device, args.batch_size)
        train_scores, _ = predict_decision(model, head, splits["train"], device, args.batch_size)
        choices = []
        for margin in margins:
            if args.threshold_mode == "stratified":
                thresholds = choose_stratified_thresholds(cal_scores, splits["calibration"], margin)
                choice = {
                    "margin": margin,
                    "threshold": thresholds["__global__"],
                    "thresholds": thresholds,
                    "train": metric_stratified(train_scores, splits["train"], thresholds),
                    "calibration": metric_stratified(cal_scores, splits["calibration"], thresholds),
                    "sealed_test": metric_stratified(test_scores, splits["sealed_test"], thresholds),
                }
            else:
                threshold = choose_zero_fp_threshold(cal_scores, splits["calibration"], margin)
                choice = {
                    "margin": margin,
                    "threshold": threshold,
                    "thresholds": {"__global__": threshold},
                    "train": metric(train_scores, splits["train"], threshold),
                    "calibration": metric(cal_scores, splits["calibration"], threshold),
                    "sealed_test": metric(test_scores, splits["sealed_test"], threshold),
                }
            choices.append(choice)
        # Calibration determines both threshold and selected epoch.  The sealed
        # task split is audit-only and cannot influence this choice.
        choice = min(
            choices,
            key=lambda item: (
                item["calibration"]["false_safe"],
                -item["calibration"]["auto_safe_true"],
                -item["train"]["auto_safe_true"],
            ),
        )
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_rows), 1),
            "threshold": choice["threshold"],
            "thresholds": choice["thresholds"],
            "threshold_margin": choice["margin"],
            "threshold_mode": args.threshold_mode,
            "calibration_top_unsafe": top_unsafe_scores(cal_scores, splits["calibration"]),
            "sealed_top_unsafe": top_unsafe_scores(test_scores, splits["sealed_test"]),
            "margin_sweep": choices,
            "train": choice["train"],
            "calibration": choice["calibration"],
            "sealed_test": choice["sealed_test"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        key = (
            record["calibration"]["false_safe"],
            -record["calibration"]["auto_safe_true"],
            -record["train"]["auto_safe_true"],
        )
        if best is None or key < best[0]:
            best = (
                key,
                {
                    "epoch": epoch,
                    "threshold": choice["threshold"],
                    "thresholds": choice["thresholds"],
                    "record": record,
                    "scale_branch_state_dict": {
                        name: {k: v.detach().cpu().clone() for k, v in getattr(model, name).state_dict().items()}
                        for name in SCALE_BRANCH if hasattr(model, name)
                    },
                    "scale_head_v2_state_dict": {
                        k: v.detach().cpu().clone() for k, v in model.scale_head_v2.state_dict().items()
                    },
                    "decision_head_state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    "feat_dim": feat_dim,
                },
            )

    state = best[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.output_dir / "decision_aware_pino.pt"
    torch.save({
        **state,
        "base_ckpt": str(args.base_ckpt),
        "start_ckpt": str(args.start_ckpt),
        "source_candidates": str(args.src),
    }, ckpt)
    selected = state["record"]
    full_fem_time = selected["sealed_test"]["full_fem_time_s"]
    fallback_time = selected["sealed_test"]["fallback_fem_time_s"]
    selected["sealed_test"]["fem_time_avoided_rate"] = 1.0 - fallback_time / max(full_fem_time, 1e-12)
    result = {
        "schema": "current_candidate_taskgrouped_decision_pino_v2",
        "source_candidates": str(args.src),
        "checkpoint": str(ckpt),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_protocol": "ordered silhouette triples are disjoint across train, calibration, and sealed test",
        "model_selection_protocol": "epoch and auto-safe threshold selected using calibration only; sealed test is audit-only",
        "split_safe_unsafe": {
            k: {"safe": int(labels(v).sum()), "unsafe": int((~labels(v)).sum())}
            for k, v in splits.items()
        },
        "config": vars(args) | {"src": str(args.src), "base_ckpt": str(args.base_ckpt), "start_ckpt": str(args.start_ckpt), "out": str(args.out), "output_dir": str(args.output_dir)},
        "selected": selected,
        "history": history,
        "promotion_gate": {
            "zero_false_safe_on_calibration": selected["calibration"]["false_safe"] == 0,
            "zero_false_safe_on_sealed_test": selected["sealed_test"]["false_safe"] == 0,
            "sealed_test_auto_safe_at_least_85pct": selected["sealed_test"]["auto_safe_rate_over_safe"] >= 0.85,
            "sealed_test_auto_safe_at_least_90pct": selected["sealed_test"]["auto_safe_rate_over_safe"] >= 0.90,
        },
        "limitations": [
            "This is a current-candidate-domain experiment; promotion still requires a larger sealed domain before broad FEM-replacement claims.",
            "Safety labels are voxel-FEM labels under the 50MPa threshold.",
            "The checkpoint is not installed into the main pipeline by this script.",
        ],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(args.out), "checkpoint": str(ckpt), "selected": selected, "promotion_gate": result["promotion_gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
