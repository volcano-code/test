#!/usr/bin/env python3
"""Construct explicitly labelled low-fidelity stress tasks for a Mitra shadow hull.

The tasks are composed only from silhouettes whose physical x/y/z assignments
are declared by an existing fixed-view manifest.  Candidate triples are first
screened at a low resolution, then re-evaluated under the requested resolution.
This is a diagnostic stress suite, *not* a randomly sampled benchmark and must
not be mixed into a controlled main-table average.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import generate_binary_structure, label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path.insert(0, str(ROOT / "experiments"))

from mitra_pauly2009_shadow_hull import build_shadow_hull, dice, iou, project_axis, read_mask, save_binary, save_obj

S6 = generate_binary_structure(3, 1)
AXIS_INPUT = {"z": "input0_*.png", "x": "input1_*.png", "y": "input2_*.png"}
AXIS_ORDER = ("x", "y", "z")


def source_pool(manifest: dict) -> dict[str, list[dict]]:
    """Collect unique, axis-declared source masks from existing fixed tasks."""
    pool: dict[str, dict[str, dict]] = {axis: {} for axis in AXIS_ORDER}
    for case in manifest["cases"]:
        for model in case["models"]:
            order = model.get("physical_order")
            if not order or set(order) != {"floor_z_view0", "left_wall_x_view1", "right_wall_y_view2"}:
                continue
            root = Path(model["stl"]).parent
            key_for_axis = {"z": "floor_z_view0", "x": "left_wall_x_view1", "y": "right_wall_y_view2"}
            for axis, physical_key in key_for_axis.items():
                path = next(root.glob(f"inputs/{AXIS_INPUT[axis]}"), None)
                if path is None:
                    continue
                record = {
                    "label": str(order[physical_key]),
                    "source_case": str(model["example"]),
                    "path": str(path.resolve()),
                }
                pool[axis].setdefault(record["path"], record)
    return {axis: sorted(records.values(), key=lambda item: (item["label"], item["path"])) for axis, records in pool.items()}


def score(masks: list[np.ndarray]) -> tuple[float, float, np.ndarray]:
    hull = build_shadow_hull(masks, list(AXIS_ORDER))
    views = [project_axis(hull, axis) for axis in AXIS_ORDER]
    values = [iou(view, target) for view, target in zip(views, masks)]
    return float(np.mean(values)), float(np.min(values)), hull


def run_one(task: tuple[dict, dict, dict], output: Path, resolution: int, screen_iou: float) -> dict:
    masks = [read_mask(Path(record["path"]), resolution) for record in task]
    mean_iou, worst_iou, hull = score(masks)
    labels, component_count = label(hull, structure=S6)
    label_prefix = "__".join(record["label"].replace("/", "_") for record in task)
    source_digest = hashlib.sha1("|".join(record["path"] for record in task).encode("utf-8")).hexdigest()[:10]
    case_id = f"{label_prefix}__{source_digest}"
    case_dir = output / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "mitra_pauly2009_shadow_hull.npy", hull.astype(np.uint8))
    save_obj(hull, case_dir / "mitra_pauly2009_shadow_hull.obj")

    views = []
    for index, (target, axis) in enumerate(zip(masks, AXIS_ORDER)):
        rendered = project_axis(hull, axis)
        save_binary(target, case_dir / f"target_view{index}_{axis}.png")
        save_binary(rendered, case_dir / f"rendered_view{index}_{axis}.png")
        views.append({"axis": axis, "iou": iou(rendered, target), "dice": dice(rendered, target)})
    row = {
        "case_id": case_id,
        "source_labels": [record["label"] for record in task],
        "source_paths": [record["path"] for record in task],
        "source_cases": [record["source_case"] for record in task],
        "axes": list(AXIS_ORDER),
        "screen_mean_iou": screen_iou,
        "mean_iou": mean_iou,
        "mean_dice": float(np.mean([view["dice"] for view in views])),
        "worst_iou": worst_iou,
        "occupied_voxels": int(hull.sum()),
        "components_6": int(component_count),
        "single_body": bool(component_count == 1),
        "views": views,
    }
    (case_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/nsa_style_fixed_required_10cm_256/MANIFEST.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--min-iou", type=float, default=0.0, help="Final mean-IoU lower bound for selected cases")
    parser.add_argument("--threshold", type=float, default=0.90, help="Final mean-IoU upper bound for selected cases")
    parser.add_argument("--target-iou", type=float, default=None, help="Prefer candidates nearest this mean IoU during screening")
    parser.add_argument("--screen-resolution", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-final-evals", type=int, default=96)
    parser.add_argument(
        "--disjoint-source-labels",
        action="store_true",
        help="Require no source silhouette to appear in more than one selected stress task.",
    )
    args = parser.parse_args()
    if not 0 <= args.min_iou < args.threshold <= 1:
        parser.error("Require 0 <= --min-iou < --threshold <= 1")
    if args.target_iou is not None and not args.min_iou <= args.target_iou <= args.threshold:
        parser.error("--target-iou must lie within [--min-iou, --threshold]")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pool = source_pool(manifest)
    if any(not pool[axis] for axis in AXIS_ORDER):
        raise RuntimeError("The manifest did not provide all three declared physical axes")

    cached_masks = {
        axis: {record["path"]: read_mask(Path(record["path"]), args.screen_resolution) for record in pool[axis]}
        for axis in AXIS_ORDER
    }
    screened = []
    for task in itertools.product(pool["x"], pool["y"], pool["z"]):
        labels = [record["label"] for record in task]
        if len(set(labels)) != 3:
            continue
        masks = [cached_masks[axis][record["path"]] for axis, record in zip(AXIS_ORDER, task)]
        mean_iou, worst_iou, _ = score(masks)
        if args.min_iou <= mean_iou < args.threshold:
            screened.append((mean_iou, worst_iou, task))
    if args.target_iou is None:
        screened.sort(key=lambda item: (item[0], item[1], tuple(record["label"] for record in item[2])))
    else:
        screened.sort(key=lambda item: (abs(item[0] - args.target_iou), item[1], tuple(record["label"] for record in item[2])))

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    evaluated = 0
    used_labels: set[str] = set()
    selected_label_triples: set[tuple[str, str, str]] = set()
    for screen_iou, _screen_worst, task in screened:
        if evaluated >= args.max_final_evals or len(rows) >= args.count:
            break
        labels = {record["label"] for record in task}
        label_triple = tuple(record["label"] for record in task)
        if label_triple in selected_label_triples:
            continue
        if args.disjoint_source_labels and labels & used_labels:
            continue
        evaluated += 1
        row = run_one(task, args.output, args.resolution, screen_iou)
        if args.min_iou <= row["mean_iou"] < args.threshold:
            rows.append(row)
            used_labels |= labels
            selected_label_triples.add(label_triple)

    result = {
        "suite": "Mitra shadow-hull conflict stress suite",
        "warning": "Tasks were selected because the deterministic shadow-hull core has mean IoU below the declared threshold. Do not combine with a randomly sampled controlled benchmark.",
        "method_scope": "fixed orthographic conservative hull; no unavailable interactive ARAP/editing",
        "selection": {
            "threshold": args.threshold,
            "min_iou": args.min_iou,
            "target_iou": args.target_iou,
            "screen_resolution": args.screen_resolution,
            "final_resolution": args.resolution,
            "candidate_space": {axis: len(pool[axis]) for axis in AXIS_ORDER},
            "screened_below_threshold": len(screened),
            "final_evaluated": evaluated,
            "disjoint_source_labels": args.disjoint_source_labels,
        },
        "cases": len(rows),
        "rows": rows,
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output / "summary.tsv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["case_id", "mean_iou", "mean_dice", "worst_iou", "components_6", "single_body", "screen_mean_iou"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fields} for row in rows])
    print(json.dumps({key: result[key] for key in ("suite", "cases", "selection")}, indent=2))


if __name__ == "__main__":
    main()
