#!/usr/bin/env python3
"""Summarize proposal and geometry-module ablations without mixing their stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def components(volume_path: Path) -> tuple[int, float]:
    volume = np.load(volume_path).astype(bool)
    labels, count = ndimage.label(volume, structure=ndimage.generate_binary_structure(3, 1))
    if not count:
        return 0, 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return int(count), float(sizes.max() / max(int(volume.sum()), 1))


def proposal_rows(root: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(root.glob("*/*/seed_*/metrics.json")):
        data = json.loads(metrics_path.read_text())
        comp, largest = components(metrics_path.parent / "occupancy_raw_core.npy")
        rows.append({
            "task": metrics_path.parents[2].name,
            "variant": metrics_path.parents[1].name,
            "seed": metrics_path.parent.name,
            "proposal_iou": data["raw_soft_iou"],
            "proposal_dice": data["core_mean_dice"],
            "components": comp,
            "largest_component_fraction": largest,
            "single_body": int(comp == 1),
        })
    return rows


def geometry_rows(root: Path) -> list[dict]:
    rows = []
    for manifest_path in sorted(root.glob("*/*/MANIFEST.json")):
        data = json.loads(manifest_path.read_text())
        selected = data.get("selected", {})
        if selected:
            rows.append({
                "task": manifest_path.parents[1].name,
                "variant": manifest_path.parent.name,
                "delivery_iou": selected.get("mean_iou"),
                "delivery_dice": selected.get("mean_dice"),
                "saving": selected.get("saving"),
                "single_body": int(selected.get("components") == 1),
                "watertight": int(bool(selected.get("watertight"))),
            })
    return rows


def aggregate(rows: list[dict], group: str) -> list[dict]:
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    out = []
    for variant, group_rows in sorted(by_variant.items()):
        result = {"variant": variant, "n": len(group_rows)}
        for key in group_rows[0]:
            if key not in {"task", "variant", "seed"} and isinstance(group_rows[0][key], (int, float)):
                values = [r[key] for r in group_rows if r[key] is not None]
                result[key] = float(np.mean(values)) if values else None
        out.append(result)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "proposal_raw_rows": proposal_rows(args.proposal_root),
        "geometry_raw_rows": geometry_rows(args.geometry_root),
    }
    payload["proposal_aggregate"] = aggregate(payload["proposal_raw_rows"], "proposal")
    payload["geometry_aggregate"] = aggregate(payload["geometry_raw_rows"], "geometry")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "proposal_rows": len(payload["proposal_raw_rows"]),
                      "geometry_rows": len(payload["geometry_raw_rows"])}, indent=2))


if __name__ == "__main__":
    main()
