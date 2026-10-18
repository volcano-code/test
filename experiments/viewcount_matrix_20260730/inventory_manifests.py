#!/usr/bin/env python3
"""Build a compact inventory of reusable P-CRAFT result manifests."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FAMILY_PATTERNS = {
    "anime": re.compile(
        r"anime|advent|school|fighter|warrior|miko|violin|mecha|skater|mage",
        re.IGNORECASE,
    ),
    "ordinary": re.compile(
        r"ordinary|beetle|parrot|eagle|snail|turtle|scorpion|lobster|connected",
        re.IGNORECASE,
    ),
    "mixed": re.compile(
        r"mixed|detect|spider|dark_knight|fox|future",
        re.IGNORECASE,
    ),
}


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def infer_family(path: Path) -> str:
    text = str(path)
    for family, pattern in FAMILY_PATTERNS.items():
        if pattern.search(text):
            return family
    return "unclassified"


def infer_views(data: dict[str, Any], path: Path) -> int | None:
    selected = data.get("selected")
    if not isinstance(selected, dict):
        selected = {}
    metric_blocks = [
        selected,
        data.get("selected_volume_metrics"),
        data.get("delivery_qa"),
        data.get("raw_soft"),
        data,
    ]
    for block in metric_blocks:
        if not isinstance(block, dict):
            continue
        for key in ("view_ious", "per_view", "views"):
            value = block.get(key)
            if not isinstance(value, list):
                continue
            count = len(value)
            if count in (2, 3):
                return count
            if count in (4, 6):
                return count // 2
    text = str(path).lower()
    if "three_view" in text or "3v" in text or "fixed3" in text:
        return 3
    if "two_view" in text or "2v" in text or "two_wall" in text:
        return 2
    return None


def read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    selected = data.get("selected")
    delivery = data.get("delivery_qa")
    volume = data.get("selected_volume_metrics")
    if not isinstance(selected, dict):
        selected = {}
    if not isinstance(delivery, dict):
        delivery = {}
    if not isinstance(volume, dict):
        volume = {}
    metrics = selected or delivery or volume
    mean_iou = first_present(metrics, "mean_iou")
    min_iou = first_present(metrics, "min_iou")
    if min_iou is None:
        view_ious = metrics.get("view_ious")
        if isinstance(view_ious, list) and view_ious:
            min_iou = min(view_ious)
    display_stl = data.get("display_stl")
    if display_stl and not Path(display_stl).is_absolute():
        candidate = Path.cwd() / display_stl
        display_stl = str(candidate if candidate.is_file() else path.parent / display_stl)
    return {
        "family": infer_family(path),
        "views": infer_views(data, path),
        "mean_iou": mean_iou,
        "min_iou": min_iou,
        "mean_dice": first_present(metrics, "mean_dice"),
        "saving": first_present(metrics, "saving"),
        "components": first_present(delivery, "components")
        if delivery
        else first_present(metrics, "components"),
        "watertight": first_present(delivery, "watertight")
        if delivery
        else first_present(metrics, "watertight"),
        "status": data.get("status"),
        "display_stl": display_stl,
        "manifest": str(path),
    }


def reusable(row: dict[str, Any], min_iou: float) -> bool:
    mean_iou = row.get("mean_iou")
    return bool(
        row.get("views") in (2, 3)
        and mean_iou is not None
        and float(mean_iou) >= min_iou
        and row.get("components") in (None, 1)
        and row.get("watertight") in (None, True)
        and row.get("display_stl")
        and Path(row["display_stl"]).is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-iou", type=float, default=0.95)
    parser.add_argument("--limit-per-cell", type=int, default=8)
    args = parser.parse_args()

    rows = []
    for path in args.root.glob("results/**/MANIFEST.json"):
        row = read_manifest(path)
        if row and reusable(row, args.min_iou):
            rows.append(row)
    rows.sort(
        key=lambda row: (
            row["family"],
            row["views"],
            -(float(row["mean_iou"]) if row["mean_iou"] is not None else -1.0),
        )
    )

    selected: list[dict[str, Any]] = []
    counts: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (row["family"], row["views"])
        if counts.get(key, 0) >= args.limit_per_cell:
            continue
        selected.append(row)
        counts[key] = counts.get(key, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2))
    columns = [
        "family",
        "views",
        "mean_iou",
        "min_iou",
        "mean_dice",
        "saving",
        "components",
        "watertight",
        "manifest",
    ]
    print("\t".join(columns))
    for row in selected:
        print("\t".join(str(row.get(column, "")) for column in columns))


if __name__ == "__main__":
    main()
