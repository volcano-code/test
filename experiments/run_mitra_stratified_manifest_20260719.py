#!/usr/bin/env python3
"""Evaluate the reproducible Mitra--Pauly shadow hull on a frozen task manifest."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import generate_binary_structure, label


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path.insert(0, str(ROOT / "experiments"))

from mitra_pauly2009_shadow_hull import build_shadow_hull, dice, iou, project_axis, read_mask, save_binary, save_obj


S6 = generate_binary_structure(3, 1)


def run_case(case: dict, output: Path, resolution: int, export_obj: bool = True) -> dict:
    masks = [read_mask(Path(path), resolution) for path in case["target_paths_xyz"]]
    axes = ["x", "y", "z"]
    volume = build_shadow_hull(masks, axes)
    _, components = label(volume, structure=S6)
    case_dir = output / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "mitra_pauly2009_shadow_hull.npy", volume.astype(np.uint8))
    if export_obj:
        save_obj(volume, case_dir / "mitra_pauly2009_shadow_hull.obj")

    views = []
    for index, (axis, target) in enumerate(zip(axes, masks)):
        rendered = project_axis(volume, axis)
        save_binary(target, case_dir / f"target_view{index}_{axis}.png")
        save_binary(rendered, case_dir / f"rendered_view{index}_{axis}.png")
        views.append({"axis": axis, "iou": iou(rendered, target), "dice": dice(rendered, target)})
    row = {
        "case_id": case["case_id"],
        "stratum": case.get("stratum", "unspecified"),
        "mean_iou": float(np.mean([view["iou"] for view in views])),
        "mean_dice": float(np.mean([view["dice"] for view in views])),
        "worst_iou": float(np.min([view["iou"] for view in views])),
        "occupied_voxels": int(volume.sum()),
        "components_6": int(components),
        "single_body": bool(components == 1),
        "digital_watertight": True,
        "views": views,
    }
    (case_dir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--skip-obj",
        action="store_true",
        help="Skip optional marching-cubes visualization; voxel metrics are unchanged.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse per-case metrics when present; useful for isolated 256^3 runs.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        case for case in manifest["cases"]
        if not args.case or case["case_id"] in set(args.case)
    ]
    rows = []
    for case in cases:
        metrics = args.output / case["case_id"] / "metrics.json"
        if args.reuse and metrics.is_file():
            rows.append(json.loads(metrics.read_text()))
        else:
            rows.append(run_case(case, args.output, args.resolution, export_obj=not args.skip_obj))
    aggregate = {
        "method": "Mitra--Pauly (2009) shadow-hull core reproduction",
        "task_count": len(rows),
        "mean_iou": float(np.mean([row["mean_iou"] for row in rows])),
        "std_iou": float(np.std([row["mean_iou"] for row in rows])),
        "mean_dice": float(np.mean([row["mean_dice"] for row in rows])),
        "single_body_rate": float(np.mean([row["single_body"] for row in rows])),
    }
    (args.output / "results.json").write_text(json.dumps({"aggregate": aggregate, "rows": rows}, indent=2) + "\n")
    with (args.output / "summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "stratum", "mean_iou", "mean_dice", "worst_iou", "components_6", "single_body"])
        writer.writeheader()
        writer.writerows([{name: row[name] for name in writer.fieldnames} for row in rows])
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
