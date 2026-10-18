#!/usr/bin/env python3
"""Split one fixed Mitra benchmark into low- and high-Mitra strata.

Every pre-registered task is retained.  The split only supports subgroup
analysis; the aggregate over all tasks remains the primary comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/lin/xvxinyao/ShadowForge")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results/mitra_pauly2009_fixed3view_20260716/results.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = sorted(payload["rows"], key=lambda row: (row["mean_iou"], row["case_id"]))
    if len(rows) < 2 or len(rows) % 2:
        raise RuntimeError("Expected an even number of fixed benchmark tasks")
    middle = len(rows) // 2
    for stratum, subset in (("low_mitra", rows[:middle]), ("high_mitra", rows[middle:])):
        for rank, row in enumerate(subset, start=1):
            row["stratum"] = stratum
            row["stratum_rank"] = rank

    output = {
        "name": "Fixed 12-task benchmark with post-hoc Mitra difficulty strata",
        "protocol": "All 12 pre-registered fixed three-view tasks are retained. Strata are analysis labels derived only after the Mitra core evaluation and must not replace the aggregate primary benchmark.",
        "source_results": str(args.results.resolve()),
        "cases": len(rows),
        "low_mitra_cases": rows[:middle],
        "high_mitra_cases": rows[middle:],
        "aggregate": {
            "mean_iou": float(np.mean([row["mean_iou"] for row in rows])),
            "std_iou": float(np.std([row["mean_iou"] for row in rows], ddof=1)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    tsv = args.output.with_suffix(".tsv")
    with tsv.open("w", newline="", encoding="utf-8") as handle:
        fields = ["stratum", "stratum_rank", "case_id", "mean_iou", "mean_dice", "worst_iou", "components_6", "single_body"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows([{field: row[field] for field in fields} for row in rows])
    print(json.dumps({"output": str(args.output), "tsv": str(tsv), "cases": len(rows), "low_mitra": [row["case_id"] for row in rows[:middle]], "high_mitra": [row["case_id"] for row in rows[middle:]]}, indent=2))


if __name__ == "__main__":
    main()
