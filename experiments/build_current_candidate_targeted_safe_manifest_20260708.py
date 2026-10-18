#!/usr/bin/env python3
"""Summarize near-boundary and ACS-margin safe rows in current candidates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "results/candidate_selection_400case_iou_dice_20260708/candidate_selection_100case_iou_dice.json"
DEFAULT_OUT = ROOT / "results/current_candidate_targeted_safe_manifest_20260708.json"


def pct(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ("min", "p05", "p25", "median", "p75", "p95", "max")}
    qs = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])
    return dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"), [float(x) for x in qs]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--near-log-width", type=float, default=0.35)
    parser.add_argument("--acs-lower-margin", type=float, default=1.05)
    args = parser.parse_args()

    source = json.loads(args.src.read_text())
    tau = float(source["threshold_sf"])
    rows = []
    by_case = {}
    for case in source["cases"]:
        gid = f"{case['branch']}:{case['case_id']}"
        case_counts = {"rows": 0, "safe": 0, "near_safe": 0, "acs_margin_safe": 0, "both": 0}
        for row in case["candidates"]:
            fem_safe = float(row["sf_fem"]) >= tau
            log_margin = math.log10(max(float(row["sf_fem"]), 1e-12) / tau)
            acs_lower = float(row.get("acs_lower_sf", 0.0))
            acs_lower_margin = acs_lower / tau
            near_safe = fem_safe and 0.0 <= log_margin <= args.near_log_width
            acs_margin_safe = fem_safe and acs_lower_margin >= args.acs_lower_margin
            item = {
                "group": gid,
                "branch": case["branch"],
                "case_id": int(case["case_id"]),
                "triple": case["triple"],
                "candidate": row["candidate"],
                "sf_fem": float(row["sf_fem"]),
                "acs_lower_sf": acs_lower,
                "fem_safe": fem_safe,
                "log_sf_margin": log_margin,
                "acs_lower_margin": acs_lower_margin,
                "near_boundary_safe": near_safe,
                "acs_lower_margin_safe": acs_margin_safe,
                "targeted_safe": near_safe or acs_margin_safe,
                "both_targets": near_safe and acs_margin_safe,
            }
            rows.append(item)
            case_counts["rows"] += 1
            case_counts["safe"] += int(fem_safe)
            case_counts["near_safe"] += int(near_safe)
            case_counts["acs_margin_safe"] += int(acs_margin_safe)
            case_counts["both"] += int(near_safe and acs_margin_safe)
        by_case[gid] = case_counts

    safe = [r for r in rows if r["fem_safe"]]
    result = {
        "schema": "current_candidate_targeted_safe_manifest_v1",
        "source": str(args.src),
        "threshold_sf": tau,
        "near_log_width": args.near_log_width,
        "acs_lower_margin": args.acs_lower_margin,
        "counts": {
            "rows": len(rows),
            "safe": sum(r["fem_safe"] for r in rows),
            "unsafe": sum(not r["fem_safe"] for r in rows),
            "near_boundary_safe": sum(r["near_boundary_safe"] for r in rows),
            "acs_lower_margin_safe": sum(r["acs_lower_margin_safe"] for r in rows),
            "both_targets": sum(r["both_targets"] for r in rows),
            "targeted_safe_union": sum(r["targeted_safe"] for r in rows),
            "cases_with_targeted_safe": sum(1 for c in by_case.values() if c["near_safe"] or c["acs_margin_safe"]),
        },
        "safe_distributions": {
            "sf_fem": pct([r["sf_fem"] for r in safe]),
            "log_sf_margin": pct([r["log_sf_margin"] for r in safe]),
            "acs_lower_margin": pct([r["acs_lower_margin"] for r in safe]),
        },
        "target_rows": [r for r in rows if r["targeted_safe"]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "counts": result["counts"], "safe_distributions": result["safe_distributions"]}, indent=2))


if __name__ == "__main__":
    main()
