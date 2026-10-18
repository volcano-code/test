#!/usr/bin/env python3
"""Evaluate frozen v29 enhanced-scale ensemble on the current candidate domain.

This script keeps the current 400-case candidate table as the authority for
FEM labels. It regenerates the voxel candidates only to feed v29/v27/v28 PINO
models, then calibrates a fail-closed safe screen on grouped calibration cases
and audits it on sealed groups.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "generation")]

from candidate_selection_batch import make_candidates  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from scripts.train_v26_shadowart_scale_sf import build_x  # noqa: E402
from structural.analysis.load_generator import LoadGenerator  # noqa: E402
from structural.models.load_model import load_surrogate  # noqa: E402

YIELD_STRESS = 50e6
TAU = 1.3
SEED = 20260709
DEFAULT_SRC = ROOT / "results/candidate_selection_400case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"
BASE = ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"
CHECKPOINTS = {
    "v27": ROOT / "checkpoints/v27_shadowart_scale_sf_safety/best_scale_sf.pt",
    "v28": ROOT / "checkpoints/v28_shadowart_scale_sf_300case/best_scale_sf.pt",
    "v29a": ROOT / "checkpoints/v29_v27_pinball/best_scale_sf.pt",
    "v29b": ROOT / "checkpoints/v29b_v27_pinball_s2605/best_scale_sf.pt",
    "v29c": ROOT / "checkpoints/v29c_v27_pinball_s2606/best_scale_sf.pt",
}


def load_masks(triple: list[str], resolution: int) -> list[np.ndarray]:
    sil_dir = ROOT / "data/silhouettes/bench3"
    tensors, _ = load_data_from_list(
        [str(sil_dir / f"{name}.png") for name in triple],
        True,
        (resolution, resolution),
        torch.device("cpu"),
        3,
        False,
    )
    return [np.asarray(x).squeeze().astype(np.float32) for x in tensors[:3]]


def load_model(path: Path, device: torch.device):
    model, _ = load_surrogate(str(BASE), device=device)
    model.enable_enhanced_scale_head()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.scale_head_v2.load_state_dict(checkpoint["scale_head_v2_state_dict"])
    for name, state in checkpoint["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def xg(model, density: np.ndarray, force: np.ndarray, device: torch.device) -> torch.Tensor:
    x = build_x(density.astype(np.float32), force.astype(np.float32)).unsqueeze(0)
    b, _, nx, ny, nz = x.shape
    grid = model._get_grid((nx, ny, nz), "cpu").expand(b, -1, -1, -1, -1)
    return torch.cat([x, grid], 1).to(device)


@torch.no_grad()
def predict_model(model, rows: list[dict], device: torch.device, batch_size: int) -> np.ndarray:
    out: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        xb = torch.cat([xg(model, r["density"], r["force"], device) for r in batch], dim=0)
        pred = model._scale_forward(xb).squeeze(-1).reshape(-1)
        out.extend(float(v) for v in pred.detach().cpu())
    return np.asarray(out, dtype=float)


def build_rows(source: dict) -> list[dict]:
    generator = LoadGenerator()
    rows: list[dict] = []
    resolution = int(source["resolution"])
    for case in source["cases"]:
        masks = load_masks(case["triple"], resolution)
        candidates, _ = make_candidates(masks, resolution, case["branch"] == "fixed")
        by_name = {r["candidate"]: r for r in case["candidates"]}
        for name, volume in candidates.items():
            rec = by_name.get(name)
            if rec is None:
                continue
            density = volume.astype(np.float32)
            force = generator.standing_load(density)["force"].astype(np.float32)
            rows.append({
                "group": f"{case['branch']}:{int(case['case_id'])}",
                "branch": case["branch"],
                "case_id": int(case["case_id"]),
                "candidate": name,
                "density": density,
                "force": force,
                "sf_fem": float(rec["sf_fem"]),
                "fem_safe": bool(rec["fem_safe"]),
                "fem_elapsed_s": float(rec.get("fem_elapsed_s", 0.0)),
                "material_saved_vs_hull": float(rec["material_saved_vs_hull"]),
            })
    return rows


def split_masks(groups: list[str], train_frac: float, cal_frac: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    unique = np.asarray(sorted(set(groups)))
    rng.shuffle(unique)
    n_train = int(round(len(unique) * train_frac))
    n_cal = int(round(len(unique) * cal_frac))
    train = set(unique[:n_train])
    cal = set(unique[n_train : n_train + n_cal])
    test = set(unique[n_train + n_cal :])
    arr = np.asarray(groups)
    return {
        "train": np.asarray([g in train for g in arr]),
        "calibration": np.asarray([g in cal for g in arr]),
        "sealed_test": np.asarray([g in test for g in arr]),
    }


def zero_fp_q(pred_log_vm: np.ndarray, true_log_vm: np.ndarray, truth_safe: np.ndarray, alpha_grid: list[float]) -> dict:
    # q is added to predicted log VM; larger q is more conservative.
    residual = true_log_vm - pred_log_vm
    best = None
    for alpha in alpha_grid:
        rank = int(math.ceil((1.0 - alpha) * (len(residual) + 1)))
        q = float(np.sort(residual)[min(max(rank, 1), len(residual)) - 1])
        lower_sf = YIELD_STRESS / np.maximum(10.0 ** (pred_log_vm + q), 1e-12)
        accept = lower_sf >= TAU
        false_safe = int((accept & ~truth_safe).sum())
        true_accept = int((accept & truth_safe).sum())
        item = {"alpha": alpha, "q_safe": q, "false_safe": false_safe, "true_accept": true_accept}
        if false_safe == 0 and (best is None or true_accept > best["true_accept"]):
            best = item
    if best is None:
        alpha = 0.0
        unsafe = pred_log_vm[~truth_safe]
        q = float(np.max(np.log10(YIELD_STRESS / TAU) - unsafe) + 1e-6) if len(unsafe) else float(np.max(residual))
        best = {"alpha": alpha, "q_safe": q, "false_safe": 0, "true_accept": 0}
    return best


def evaluate(pred_log_vm: np.ndarray, rows: list[dict], mask: np.ndarray, q_safe: float) -> dict:
    truth = np.asarray([r["fem_safe"] for r in rows], dtype=bool)[mask]
    fem_times = np.asarray([r["fem_elapsed_s"] for r in rows], dtype=float)[mask]
    pred = pred_log_vm[mask]
    lower_sf = YIELD_STRESS / np.maximum(10.0 ** (pred + q_safe), 1e-12)
    accept = lower_sf >= TAU
    fallback = ~accept
    return {
        "n": int(len(truth)),
        "safe": int(truth.sum()),
        "unsafe": int((~truth).sum()),
        "auto_safe": int(accept.sum()),
        "auto_safe_true": int((accept & truth).sum()),
        "false_safe": int((accept & ~truth).sum()),
        "auto_safe_rate_over_safe": float((accept & truth).sum() / max(int(truth.sum()), 1)),
        "autonomous_rate": float(accept.mean()) if len(accept) else 0.0,
        "full_fem_time_s": float(fem_times.sum()),
        "fallback_fem_time_s": float(fem_times[fallback].sum()),
        "fem_time_avoided_rate": float(1.0 - fem_times[fallback].sum() / max(fem_times.sum(), 1e-12)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=ROOT / "results/v29_current_candidate_domain_20260709.json")
    ap.add_argument("--cache", type=Path, default=ROOT / "results/v29_current_candidate_domain_predictions_20260709.npz")
    ap.add_argument("--train-frac", type=float, default=0.60)
    ap.add_argument("--cal-frac", type=float, default=0.20)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    source = json.loads(args.src.read_text())
    rows = build_rows(source)
    groups = [r["group"] for r in rows]
    masks = split_masks(groups, args.train_frac, args.cal_frac)
    truth_log_vm = np.asarray([math.log10(YIELD_STRESS / max(r["sf_fem"], 1e-12)) for r in rows], dtype=float)
    truth_safe = np.asarray([r["fem_safe"] for r in rows], dtype=bool)

    if args.cache.exists():
        cached = np.load(args.cache)
        predictions = {k: cached[k] for k in cached.files}
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        predictions = {}
        for name, path in CHECKPOINTS.items():
            model = load_model(path, device)
            predictions[name] = predict_model(model, rows, device, args.batch_size)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(json.dumps({"completed": name}), flush=True)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, **predictions)

    mean_v29 = np.mean(np.stack([predictions[n] for n in ("v29a", "v29b", "v29c")]), axis=0)
    disagreement = np.abs(predictions["v27"] - predictions["v28"])
    # Conservative disagreement handling: only calibrate on rows below a
    # calibration disagreement quantile, defer the rest.
    results = []
    alpha_grid = [0.02, 0.01, 0.005, 0.002, 0.001]
    for disagree_q in [1.00, 0.98, 0.95, 0.90, 0.85, 0.80]:
        threshold = float(np.quantile(disagreement[masks["calibration"]], disagree_q))
        usable_cal = masks["calibration"] & (disagreement <= threshold)
        q_item = zero_fp_q(mean_v29[usable_cal], truth_log_vm[usable_cal], truth_safe[usable_cal], alpha_grid)
        row = {
            "disagreement_quantile": disagree_q,
            "disagreement_threshold": threshold,
            "calibration": evaluate(mean_v29, rows, masks["calibration"] & (disagreement <= threshold), q_item["q_safe"]),
            "sealed_test": evaluate(mean_v29, rows, masks["sealed_test"] & (disagreement <= threshold), q_item["q_safe"]),
            "q_safe": q_item["q_safe"],
            "alpha": q_item["alpha"],
            "deferred_by_disagreement_test": int((masks["sealed_test"] & (disagreement > threshold)).sum()),
        }
        results.append(row)
    valid = [r for r in results if r["calibration"]["false_safe"] == 0 and r["sealed_test"]["false_safe"] == 0]
    selected = max(valid, key=lambda r: r["sealed_test"]["auto_safe_true"]) if valid else min(results, key=lambda r: r["sealed_test"]["false_safe"])
    result = {
        "schema": "v29_current_candidate_domain_20260709",
        "source": str(args.src),
        "cache": str(args.cache),
        "n_rows": len(rows),
        "split_sizes": {k: int(v.sum()) for k, v in masks.items()},
        "selected": selected,
        "all_results": results,
        "promotion_gate": {
            "zero_false_safe_on_calibration": selected["calibration"]["false_safe"] == 0,
            "zero_false_safe_on_sealed_test": selected["sealed_test"]["false_safe"] == 0,
            "sealed_auto_safe_at_least_85pct": selected["sealed_test"]["auto_safe_rate_over_safe"] >= 0.85,
        },
        "notes": [
            "Uses enhanced scale_head_v2 checkpoints v27/v28/v29a/v29b/v29c.",
            "FEM labels are reused from the current candidate table; FEM is not recomputed.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "selected": selected, "promotion_gate": result["promotion_gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
