#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "case",
    "views",
    "category",
    "runner_status",
    "manifest_status",
    "mean_iou",
    "min_iou",
    "mean_dice",
    "saving",
    "components",
    "watertight",
    "structural_pass",
    "structural_status",
    "structural_decided_by",
    "display_stl",
    "print_stl",
]


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def summarize(config_path: Path) -> list[dict]:
    config = load_json(config_path)
    output_root = Path(config["output_root"])
    rows = []
    for case in config["cases"]:
        case_dir = output_root / case["id"]
        runner = load_json(case_dir / "STATUS.json")
        manifest = load_json(case_dir / "MANIFEST.json")
        selected = manifest.get("selected") or {}
        structural = selected.get("structural_screen") or {}
        rows.append(
            {
                "case": case["id"],
                "views": case["views"],
                "category": case["category"],
                "runner_status": runner.get("status", "NOT_STARTED"),
                "manifest_status": manifest.get("status"),
                "mean_iou": selected.get("mean_iou"),
                "min_iou": selected.get("min_iou"),
                "mean_dice": selected.get("mean_dice"),
                "saving": selected.get("saving"),
                "components": selected.get("components"),
                "watertight": selected.get("watertight"),
                "structural_pass": selected.get("structural_pass"),
                "structural_status": structural.get("status"),
                "structural_decided_by": structural.get("decided_by"),
                "display_stl": manifest.get("display_stl"),
                "print_stl": manifest.get("print_stl"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("cases.json"),
    )
    args = parser.parse_args()
    config = load_json(args.config)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    rows = summarize(args.config)

    json_path = output_root / "SUMMARY.json"
    tsv_path = output_root / "SUMMARY.tsv"
    json_path.write_text(
        json.dumps({"batch_id": config["batch_id"], "cases": rows}, indent=2),
        encoding="utf-8",
    )
    with tsv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary_json": str(json_path), "summary_tsv": str(tsv_path), "cases": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
