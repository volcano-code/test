#!/usr/bin/env python3
"""Select one Fig. 4 case only from eligible P-CRAFT deliveries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--pcraft-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    table = json.loads(args.table.read_text())
    pcraft = json.loads(args.pcraft_summary.read_text())
    selected_by_case = {
        row["case_id"]: row["selected"]
        for row in pcraft
        if row.get("selected") is not None
    }
    per_case: dict[str, dict[str, dict]] = {}
    for row in table["per_case"]:
        per_case.setdefault(row["case_id"], {})[row["method"]] = row

    candidates = []
    for case_id, rows in per_case.items():
        ours = rows.get("P-CRAFT", {})
        selected = selected_by_case.get(case_id, {})
        eligible = bool(selected.get("eligible", ours.get("eligible", False)))
        if not (eligible and ours.get("single_body") and ours.get("watertight")):
            continue
        baselines = [
            rows[name]["mean_iou"]
            for name in ("Mitra-style", "Sadekar voxel", "Sadekar mesh")
            if name in rows
        ]
        best_baseline = max(baselines)
        candidates.append({
            "case_id": case_id,
            "pcraft_iou": ours["mean_iou"],
            "pcraft_worst_view_iou": ours["worst_iou"],
            "best_baseline_iou": best_baseline,
            "margin": ours["mean_iou"] - best_baseline,
            "selected": selected,
        })
    candidates.sort(key=lambda row: (row["margin"], row["pcraft_worst_view_iou"]), reverse=True)
    payload = {
        "selection_rule": table["selection_policy"],
        "selected": candidates[0] if candidates else None,
        "eligible_candidates": candidates,
        "status": "selected" if candidates else "NO_ELIGIBLE_PCRAFT_CASE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
