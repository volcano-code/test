#!/usr/bin/env python3
"""Append train-only FEM rows while preserving frozen evaluation splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve(index: Path, row: dict) -> dict:
    item = dict(row)
    path = Path(item["path"])
    item["path"] = str(path if path.is_absolute() else (index.parent / path).resolve())
    if not Path(item["path"]).is_file():
        raise FileNotFoundError(item["path"])
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--augmentation-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = json.loads(args.base_index.read_text())
    augmentation = json.loads(args.augmentation_index.read_text())
    base_label = base.get("schema", "base")
    augmentation_label = augmentation.get("schema", "train_augmentation")
    base_rows = [resolve(args.base_index, row) for row in base["rows"]]
    extra_rows = [resolve(args.augmentation_index, row) for row in augmentation["rows"]]
    if any(row.get("split") != "train" for row in extra_rows):
        raise RuntimeError("Augmentation index must contain train rows only")

    owner = {row["task"]: row["split"] for row in base_rows}
    for row in extra_rows:
        if owner.get(row["task"]) != "train":
            raise RuntimeError(f"Augmentation task is not train-owned: {row['task']}")
        row["union_source"] = augmentation_label
    for row in base_rows:
        row.setdefault("union_source", base_label)

    rows = base_rows + extra_rows
    counts = {
        split: {
            "rows": sum(row["split"] == split for row in rows),
            "tasks": len({row["task"] for row in rows if row["split"] == split}),
            "safe": sum(row["split"] == split and float(row["sf"]) >= 1.3 for row in rows),
            "unsafe": sum(row["split"] == split and float(row["sf"]) < 1.3 for row in rows),
            "near_boundary_unsafe": sum(
                row["split"] == split and 0.75 <= float(row["sf"]) < 1.3 for row in rows
            ),
        }
        for split in ("train", "calibration", "sealed_test")
    }
    output = {
        "schema": "taskgrouped_strict_pino_train_augmentation_union",
        "split_unit": "complete silhouette task",
        "sources": {
            "base": str(args.base_index.resolve()),
            "train_augmentation": str(args.augmentation_index.resolve()),
        },
        "frozen_evaluation": "calibration and sealed_test rows are byte-identical source files from the base index",
        "counts": counts,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
