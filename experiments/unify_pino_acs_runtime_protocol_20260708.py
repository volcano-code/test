#!/usr/bin/env python3
"""Unify PINO/ACS safety and runtime accounting on the current candidate domain.

This script uses the freshly generated 100-case candidate pool with per-candidate
FEM labels/timings.  It reports the current ACS fields stored in that pool under
one protocol: auto-safe, auto-reject, FEM fallback, false-safe, and avoided FEM
time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SRC = ROOT / "results/candidate_selection_100case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"
OUT_DIR = ROOT / "results/pino_acs_unified_protocol_20260708"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pct(x: float | None) -> str:
    return "N/A" if x is None else f"{100.0 * float(x):.1f}%"


def num(x: float | None) -> str:
    return "N/A" if x is None else f"{float(x):.3f}"


def summarize(rows: list[dict], name: str) -> dict:
    n = len(rows)
    safe = np.asarray([bool(r.get("fem_safe", False)) for r in rows], dtype=bool)
    auto_safe = np.asarray([bool(r.get("acs_certified_safe", False)) for r in rows], dtype=bool)
    auto_reject = np.asarray([bool(r.get("acs_certified_unsafe", False)) for r in rows], dtype=bool)
    autonomous = auto_safe | auto_reject
    fallback = ~autonomous
    false_safe = auto_safe & ~safe
    false_reject = auto_reject & safe
    fem_times = np.asarray([float(r.get("fem_elapsed_s", np.nan)) for r in rows], dtype=float)
    finite_times = np.isfinite(fem_times)
    full_fem_time = float(np.nansum(fem_times))
    fallback_fem_time = float(np.nansum(fem_times[fallback & finite_times]))
    avoided_fem_time = float(np.nansum(fem_times[autonomous & finite_times]))
    # PINO/ACS inference time is not recorded by the candidate generation run.
    # We therefore report FEM-time-only speedup and keep wall-clock speedup as N/A.
    fem_only_speedup = full_fem_time / max(fallback_fem_time, 1e-12) if full_fem_time > 0 else None
    return {
        "split": name,
        "candidates": n,
        "safe": int(safe.sum()),
        "unsafe": int((~safe).sum()),
        "auto_safe": int(auto_safe.sum()),
        "auto_reject": int(auto_reject.sum()),
        "fallback_to_fem": int(fallback.sum()),
        "false_safe": int(false_safe.sum()),
        "false_reject": int(false_reject.sum()),
        "autonomous_rate": float(autonomous.mean()) if n else None,
        "auto_safe_rate_over_safe": float((auto_safe & safe).sum() / max(int(safe.sum()), 1)) if n else None,
        "auto_reject_rate_over_unsafe": float((auto_reject & ~safe).sum() / max(int((~safe).sum()), 1)) if n else None,
        "fem_calls_avoided_rate": float(autonomous.mean()) if n else None,
        "full_fem_time_s": full_fem_time,
        "fallback_fem_time_s": fallback_fem_time,
        "avoided_fem_time_s": avoided_fem_time,
        "fem_time_avoided_rate": avoided_fem_time / max(full_fem_time, 1e-12) if full_fem_time > 0 else None,
        "fem_time_only_speedup": fem_only_speedup,
        "wall_clock_speedup": None,
        "wall_clock_speedup_note": "PINO/ACS inference time was not recorded in the source candidate run; report FEM-time-only speedup unless a timed PINO pass is rerun.",
    }


def main() -> None:
    data = json.loads(SRC.read_text())
    all_rows = []
    by_branch = {"free": [], "fixed": []}
    for case in data["cases"]:
        branch = case["branch"]
        for row in case["candidates"]:
            row = dict(row)
            row["branch"] = branch
            row["case_id"] = case["case_id"]
            all_rows.append(row)
            by_branch[branch].append(row)

    summaries = [
        summarize(all_rows, "all_current_candidates"),
        summarize(by_branch["free"], "free_branch"),
        summarize(by_branch["fixed"], "fixed_branch"),
    ]
    result = {
        "schema": "pino_acs_unified_candidate_protocol_v1",
        "source": str(SRC),
        "definition": {
            "autonomous_rate": "(auto_safe + auto_reject) / all candidates",
            "auto_safe_rate_over_safe": "FEM-safe candidates auto-accepted / FEM-safe candidates",
            "false_safe": "FEM-unsafe candidates auto-accepted; must be zero for fail-closed safety claims",
            "fem_time_only_speedup": "sum(all candidate FEM times) / sum(fallback candidate FEM times), excluding unrecorded PINO/ACS inference time",
        },
        "summaries": summaries,
    }
    json_path = OUT_DIR / "unified_pino_acs_runtime_current_candidates.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    tsv_path = OUT_DIR / "unified_pino_acs_runtime_current_candidates.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write(
            "Split\tCandidates\tSafe\tUnsafe\tAuto-safe\tAuto-reject\tFallback FEM\t"
            "False-safe\tAutonomous rate\tAuto-safe / safe\tFEM calls avoided\t"
            "FEM time avoided\tFEM-time-only speedup\tWall-clock speedup\n"
        )
        for s in summaries:
            f.write(
                f"{s['split']}\t{s['candidates']}\t{s['safe']}\t{s['unsafe']}\t"
                f"{s['auto_safe']}\t{s['auto_reject']}\t{s['fallback_to_fem']}\t"
                f"{s['false_safe']}\t{pct(s['autonomous_rate'])}\t"
                f"{pct(s['auto_safe_rate_over_safe'])}\t{pct(s['fem_calls_avoided_rate'])}\t"
                f"{pct(s['fem_time_avoided_rate'])}\t{num(s['fem_time_only_speedup'])}\tN/A\n"
            )
    print(json.dumps({"json": str(json_path), "tsv": str(tsv_path), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
