#!/usr/bin/env python3
"""Compare two multi-seed Table-1 P-CRAFT case directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = (
    "mean_iou",
    "min_iou",
    "mean_dice",
    "saving",
    "components",
    "watertight",
    "mesh_outer_parts",
)


def load_case(root: Path) -> dict[str, dict]:
    rows = {}
    for manifest in sorted(root.glob("seed_*/MANIFEST.json")):
        payload = json.loads(manifest.read_text())
        selected = payload.get("selected") or {}
        rows[manifest.parent.name] = {
            "status": payload.get("status"),
            "candidate": selected.get("candidate"),
            **{field: selected.get(field) for field in FIELDS},
            "display_stl": payload.get("display_stl"),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = load_case(args.reference)
    rerun = load_case(args.rerun)
    deltas = {}
    for seed in sorted(set(reference) & set(rerun)):
        deltas[seed] = {
            field: (
                rerun[seed][field] - reference[seed][field]
                if isinstance(rerun[seed].get(field), (int, float))
                and not isinstance(rerun[seed].get(field), bool)
                and isinstance(reference[seed].get(field), (int, float))
                and not isinstance(reference[seed].get(field), bool)
                else None
            )
            for field in FIELDS
        }

    result = {
        "reference": reference,
        "rerun": rerun,
        "rerun_minus_reference": deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
