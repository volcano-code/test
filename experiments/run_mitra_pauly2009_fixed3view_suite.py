#!/usr/bin/env python3
"""Run the reproducible Mitra--Pauly shadow-hull core on fixed three-view tasks.

The suite is deliberately restricted to manifest rows that declare the physical
axis assignment of all three silhouettes.  It evaluates the conservative hull
before any interactive editing, which is the only fully reproducible stage of
Mitra and Pauly (2009) without the unavailable original application.
"""

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
VIEW_TO_AXIS = {
    "floor_z_view0": "z",
    "left_wall_x_view1": "x",
    "right_wall_y_view2": "y",
}


def task_rows(manifest: dict) -> list[dict]:
    rows = []
    for case in manifest["cases"]:
        for model in case["models"]:
            order = model.get("physical_order")
            if not order or set(order) != set(VIEW_TO_AXIS):
                continue
            model_path = Path(model["stl"])
            root = model_path.parent
            inputs = {
                "floor_z_view0": next(root.glob("inputs/input0_*.png"), None),
                "left_wall_x_view1": next(root.glob("inputs/input1_*.png"), None),
                "right_wall_y_view2": next(root.glob("inputs/input2_*.png"), None),
            }
            if not all(inputs.values()):
                continue
            rows.append(
                {
                    "case_id": model["example"],
                    "input_paths": [inputs[key] for key in VIEW_TO_AXIS],
                    "axes": [VIEW_TO_AXIS[key] for key in VIEW_TO_AXIS],
                }
            )
    return rows


def run_one(task: dict, output: Path, resolution: int) -> dict:
    masks = [read_mask(path, resolution) for path in task["input_paths"]]
    hull = build_shadow_hull(masks, task["axes"])
    labels, components = label(hull, structure=S6)
    case_dir = output / task["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "mitra_pauly2009_shadow_hull.npy", hull.astype(np.uint8))
    save_obj(hull, case_dir / "mitra_pauly2009_shadow_hull.obj")

    view_rows = []
    for index, (mask, axis) in enumerate(zip(masks, task["axes"])):
        rendered = project_axis(hull, axis)
        save_binary(mask, case_dir / f"target_view{index}_{axis}.png")
        save_binary(rendered, case_dir / f"rendered_view{index}_{axis}.png")
        view_rows.append({"axis": axis, "iou": iou(rendered, mask), "dice": dice(rendered, mask)})
    row = {
        "case_id": task["case_id"],
        "axes": task["axes"],
        "mean_iou": float(np.mean([item["iou"] for item in view_rows])),
        "mean_dice": float(np.mean([item["dice"] for item in view_rows])),
        "worst_iou": float(np.min([item["iou"] for item in view_rows])),
        "occupied_voxels": int(hull.sum()),
        "components_6": int(components),
        "single_body": bool(components == 1),
        "saving_vs_own_hull": 0.0,
        "views": view_rows,
    }
    (case_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/nsa_style_fixed_required_10cm_256/MANIFEST.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="0 runs all valid manifest rows")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = task_rows(manifest)
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        raise RuntimeError("No fixed three-view tasks with explicit physical axes were found")

    args.output.mkdir(parents=True, exist_ok=True)
    rows = [run_one(task, args.output, args.resolution) for task in tasks]
    aggregate = {
        "method": "Mitra--Pauly (2009) shadow-hull core reproduction",
        "scope": "fixed orthographic three-view conservative hull; no unavailable interactive ARAP/editing",
        "cases": len(rows),
        "resolution": args.resolution,
        "mean_iou": float(np.mean([row["mean_iou"] for row in rows])),
        "mean_dice": float(np.mean([row["mean_dice"] for row in rows])),
        "worst_iou": float(np.min([row["worst_iou"] for row in rows])),
        "single_body_rate": float(np.mean([row["single_body"] for row in rows])),
        "mean_components_6": float(np.mean([row["components_6"] for row in rows])),
        "mean_saving_vs_own_hull": 0.0,
    }
    (args.output / "results.json").write_text(json.dumps({"aggregate": aggregate, "rows": rows}, indent=2), encoding="utf-8")
    with (args.output / "summary.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "mean_iou", "mean_dice", "worst_iou", "occupied_voxels", "components_6", "single_body"])
        writer.writeheader()
        writer.writerows([{key: row[key] for key in writer.fieldnames} for row in rows])
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
