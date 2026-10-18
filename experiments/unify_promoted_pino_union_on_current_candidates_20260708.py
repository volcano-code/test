#!/usr/bin/env python3
"""Run the promoted PINO union safe screen on the current 100-case candidates.

The source 100-case file stores candidate names plus FEM labels/timings.  This
script reconstructs the same candidate volumes, runs the promoted two-screen
PINO union on those volumes, and reports one unified protocol with real PINO
screen wall time and cached FEM fallback time.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from audit_pino_union_safe_screen_20260708 import load_screen  # noqa: E402
from candidate_selection_batch import make_candidates  # noqa: E402
from datasets import load_data_from_list  # noqa: E402
from structural.analysis.load_generator import LoadGenerator  # noqa: E402
from train_pino_safe_side_boundary import predict  # noqa: E402

SRC = ROOT / "results/candidate_selection_100case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"
UNION = ROOT / "results/pino_union_safe_screen_20260708.json"
OUT_DIR = ROOT / "results/pino_acs_unified_protocol_20260708"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YIELD_STRESS = 50e6


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
        masks = load_masks(case["triple"], int(source["resolution"]))
        candidates, _ = make_candidates(masks, int(source["resolution"]), case["branch"] == "fixed")
        by_name = {row["candidate"]: row for row in case["candidates"]}
        for name, volume in candidates.items():
            if name not in by_name:
                continue
            recorded = by_name[name]
            load = generator.standing_load(volume.astype(np.float32))
            rows.append({
                "branch": case["branch"],
                "case_id": int(case["case_id"]),
                "candidate": name,
                "density": volume.astype(np.float32),
                "force": load["force"].astype(np.float32),
                "fem_safe": bool(recorded["fem_safe"]),
                "sf_fem": recorded["sf_fem"],
                "fem_elapsed_s": float(recorded["fem_elapsed_s"]),
            })
    return rows


def pct(x: float | None) -> str:
    return "N/A" if x is None else f"{100.0 * float(x):.1f}%"


def num(x: float | None) -> str:
    return "N/A" if x is None else f"{float(x):.3f}"


def summarize(rows: list[dict], split: str, accepted: np.ndarray, screen_time_s: float) -> dict:
    safe = np.asarray([bool(row["fem_safe"]) for row in rows], dtype=bool)
    fem_times = np.asarray([float(row["fem_elapsed_s"]) for row in rows], dtype=float)
    fallback = ~accepted
    false_safe = accepted & ~safe
    full_fem = float(np.sum(fem_times))
    fallback_fem = float(np.sum(fem_times[fallback]))
    avoided_fem = float(np.sum(fem_times[accepted]))
    hybrid = float(screen_time_s + fallback_fem)
    return {
        "split": split,
        "candidates": int(len(rows)),
        "safe": int(safe.sum()),
        "unsafe": int((~safe).sum()),
        "auto_safe": int(accepted.sum()),
        "auto_reject": 0,
        "fallback_to_fem": int(fallback.sum()),
        "false_safe": int(false_safe.sum()),
        "autonomous_rate": float(accepted.mean()) if len(rows) else None,
        "auto_safe_rate_over_safe": float((accepted & safe).sum() / max(int(safe.sum()), 1)) if len(rows) else None,
        "fem_calls_avoided_rate": float(accepted.mean()) if len(rows) else None,
        "full_fem_time_s": full_fem,
        "screen_time_s": float(screen_time_s),
        "fallback_fem_time_s": fallback_fem,
        "hybrid_time_s": hybrid,
        "avoided_fem_time_s": avoided_fem,
        "fem_time_avoided_rate": avoided_fem / max(full_fem, 1e-12),
        "wall_clock_speedup": full_fem / max(hybrid, 1e-12),
    }


def zero_false_safe_threshold(scores: np.ndarray, rows: list[dict]) -> float:
    safe = np.asarray([bool(row["fem_safe"]) for row in rows], dtype=bool)
    unsafe_scores = scores[~safe]
    if len(unsafe_scores) == 0:
        return float(np.min(scores)) - 1e-6
    return float(np.max(unsafe_scores) + 1e-6)


def make_result(
    schema: str,
    source: dict,
    union: dict,
    rows: list[dict],
    screens: list[tuple[str, Path, float]],
    accepted: np.ndarray,
    screen_time: float,
    out_name: str,
    per_candidate: list[dict] | None = None,
) -> tuple[Path, Path, list[dict]]:
    by_branch = {
        branch: [row for row in rows if row["branch"] == branch]
        for branch in ("free", "fixed")
    }
    branch_masks = {
        branch: np.asarray([row["branch"] == branch for row in rows], dtype=bool)
        for branch in ("free", "fixed")
    }
    summaries = [
        summarize(rows, "all_current_candidates", accepted, screen_time),
        summarize(by_branch["free"], "free_branch", accepted[branch_masks["free"]], screen_time * len(by_branch["free"]) / max(len(rows), 1)),
        summarize(by_branch["fixed"], "fixed_branch", accepted[branch_masks["fixed"]], screen_time * len(by_branch["fixed"]) / max(len(rows), 1)),
    ]
    result = {
        "schema": schema,
        "source_candidates": str(SRC),
        "source_union_screen": str(UNION),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "screens": [
            {"name": name, "checkpoint": str(checkpoint), "threshold_log_sf": threshold}
            for name, checkpoint, threshold in screens
        ],
        "definition": {
            "auto_safe": "accepted by the safe-side screen",
            "false_safe": "FEM-unsafe candidate accepted by the safe-side screen; must be zero for fail-closed claims",
            "wall_clock_speedup": "sum(all FEM elapsed times) / (measured PINO screen time + sum(FEM elapsed times for fallback candidates))",
        },
        "summaries": summaries,
    }
    if per_candidate is not None:
        result["per_candidate"] = per_candidate
    json_path = OUT_DIR / f"{out_name}.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tsv_path = OUT_DIR / f"{out_name}.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write(
            "Split\tCandidates\tSafe\tUnsafe\tAuto-safe\tFallback FEM\tFalse-safe\t"
            "Autonomous rate\tAuto-safe / safe\tFEM calls avoided\tFEM time avoided\t"
            "Screen time(s)\tHybrid time(s)\tSpeedup\n"
        )
        for s in summaries:
            f.write(
                f"{s['split']}\t{s['candidates']}\t{s['safe']}\t{s['unsafe']}\t"
                f"{s['auto_safe']}\t{s['fallback_to_fem']}\t{s['false_safe']}\t"
                f"{pct(s['autonomous_rate'])}\t{pct(s['auto_safe_rate_over_safe'])}\t"
                f"{pct(s['fem_calls_avoided_rate'])}\t{pct(s['fem_time_avoided_rate'])}\t"
                f"{num(s['screen_time_s'])}\t{num(s['hybrid_time_s'])}\t{num(s['wall_clock_speedup'])}\n"
            )
    return json_path, tsv_path, summaries


def main() -> None:
    source = json.loads(SRC.read_text())
    union = json.loads(UNION.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = build_rows(source)
    base_ckpt = ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"
    screens = [
        (
            "m08_cal300",
            ROOT / "checkpoints/pino_safe_accept90_cal300_m08_20260704/safe_side_external_hardneg_scale.pt",
            float(union["screens"]["m08_cal300"]["threshold_log_sf"]),
        ),
        (
            "m14_single_cal300",
            ROOT / "checkpoints/pino_safe_accept90_cal300_m14_single_20260706/safe_side_external_hardneg_scale.pt",
            float(union["screens"]["m14_single_cal300"]["threshold_log_sf"]),
        ),
    ]
    log_yield = math.log10(YIELD_STRESS)
    accepts = []
    score_columns = []
    screen_time = 0.0
    for name, checkpoint, threshold in screens:
        model = load_screen(device, base_ckpt, checkpoint)
        t0 = time.perf_counter()
        pred_log_vm = predict(model, rows, device, batch_size=16)
        if device.type == "cuda":
            torch.cuda.synchronize()
        screen_time += time.perf_counter() - t0
        pred_log_sf = log_yield - pred_log_vm
        score_columns.append((name, pred_log_sf))
        accepts.append(pred_log_sf >= threshold)
    accepted = np.logical_or.reduce(accepts)
    union_scores = np.max(np.stack([scores for _, scores in score_columns], axis=0), axis=0)
    calibrated_threshold = zero_false_safe_threshold(union_scores, rows)
    accepted_zero_fp = union_scores >= calibrated_threshold
    per_candidate = []
    for idx, row in enumerate(rows):
        rec = {
            "branch": row["branch"],
            "case_id": row["case_id"],
            "candidate": row["candidate"],
            "fem_safe": row["fem_safe"],
            "sf_fem": row["sf_fem"],
            "fem_elapsed_s": row["fem_elapsed_s"],
            "union_log_sf": float(union_scores[idx]),
            "paper_threshold_accept": bool(accepted[idx]),
            "zero_fp_accept": bool(accepted_zero_fp[idx]),
        }
        for name, scores in score_columns:
            rec[f"{name}_log_sf"] = float(scores[idx])
        per_candidate.append(rec)
    paper_json, paper_tsv, paper_summaries = make_result(
        "promoted_pino_union_current_candidate_protocol_v2",
        source,
        union,
        rows,
        screens,
        accepted,
        screen_time,
        "promoted_union_current_candidates",
        per_candidate,
    )
    zero_fp_screens = [("union_zero_false_safe_current_domain", Path("two_screen_union"), calibrated_threshold)]
    zero_json, zero_tsv, zero_summaries = make_result(
        "zero_false_safe_recalibrated_current_candidate_protocol_v1",
        source,
        union,
        rows,
        zero_fp_screens,
        accepted_zero_fp,
        screen_time,
        "promoted_union_current_candidates_zero_false_safe",
        per_candidate,
    )
    print(json.dumps({
        "paper_threshold": {"json": str(paper_json), "tsv": str(paper_tsv), "summaries": paper_summaries},
        "zero_false_safe": {"threshold_log_sf": calibrated_threshold, "json": str(zero_json), "tsv": str(zero_tsv), "summaries": zero_summaries},
    }, indent=2))


if __name__ == "__main__":
    main()
