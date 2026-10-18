#!/usr/bin/env python3
"""Audit and re-export projection evidence in the evaluator's coordinate frame.

This tool is intentionally read-only with respect to the source experiment. It
loads each selected occupancy volume, reconstructs the exact masks used by the
TFS evaluator, and writes canonical target/projection pairs to a new directory.
It also measures the legacy saved PNG pairs to expose coordinate-frame drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
DEFAULT_SOURCE = ROOT / "results/anime30_top4_10cm_256_20260707"
DEFAULT_OUTPUT = ROOT / "results/anime30_top4_10cm_256_20260707_canonical_audit_20260731"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generation"))

from datasets import load_data_from_list  # noqa: E402
from voxel_ops import (  # noqa: E402
    calibrate_tfs_projection,
    compute_dice_np,
    compute_iou_np,
    evaluate_tfs_volume,
    resize_binary_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-scale-error", type=float, default=0.02)
    parser.add_argument("--max-centroid-shift-px", type=float, default=2.0)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def load_binary(path: Path, size: int | None = None) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 127
    if size is not None:
        mask = resize_binary_mask(mask, size)
    return mask.astype(bool)


def save_binary(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def mask_geometry(mask: np.ndarray) -> dict[str, Any]:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return {
            "area": 0,
            "bbox": None,
            "bbox_width": 0,
            "bbox_height": 0,
            "centroid_x": None,
            "centroid_y": None,
        }
    x0, x1 = int(xx.min()), int(xx.max())
    y0, y1 = int(yy.min()), int(yy.max())
    return {
        "area": int(len(xx)),
        "bbox": [x0, y0, x1, y1],
        "bbox_width": int(x1 - x0 + 1),
        "bbox_height": int(y1 - y0 + 1),
        "centroid_x": float(xx.mean()),
        "centroid_y": float(yy.mean()),
    }


def pair_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    target = target.astype(bool)
    prediction = prediction.astype(bool)
    target_geo = mask_geometry(target)
    pred_geo = mask_geometry(prediction)
    if target_geo["bbox_width"] and target_geo["bbox_height"]:
        width_ratio = pred_geo["bbox_width"] / target_geo["bbox_width"]
        height_ratio = pred_geo["bbox_height"] / target_geo["bbox_height"]
        scale_error = max(abs(width_ratio - 1.0), abs(height_ratio - 1.0))
    else:
        width_ratio = None
        height_ratio = None
        scale_error = None
    if target_geo["centroid_x"] is not None and pred_geo["centroid_x"] is not None:
        centroid_shift = math.hypot(
            pred_geo["centroid_x"] - target_geo["centroid_x"],
            pred_geo["centroid_y"] - target_geo["centroid_y"],
        )
    else:
        centroid_shift = None
    return {
        "iou": compute_iou_np(prediction, target),
        "dice": compute_dice_np(prediction, target),
        "target": target_geo,
        "prediction": pred_geo,
        "bbox_width_ratio": width_ratio,
        "bbox_height_ratio": height_ratio,
        "max_bbox_scale_error": scale_error,
        "centroid_shift_px": centroid_shift,
    }


def overlay_image(target: np.ndarray, prediction: np.ndarray) -> Image.Image:
    canvas = np.full((*target.shape, 3), 255, dtype=np.uint8)
    target_only = target & ~prediction
    prediction_only = prediction & ~target
    overlap = target & prediction
    canvas[target_only] = (0, 114, 178)
    canvas[prediction_only] = (213, 94, 0)
    canvas[overlap] = (0, 158, 115)
    return Image.fromarray(canvas, mode="RGB")


def build_panel(targets: list[np.ndarray], predictions: list[np.ndarray], path: Path) -> None:
    cell = targets[0].shape[0]
    header = 32
    gap = 12
    width = 3 * cell + 4 * gap
    height = len(targets) * (cell + header) + (len(targets) + 1) * gap
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    labels = ("Canonical target", "Canonical projection", "Overlay")
    for row, (target, prediction) in enumerate(zip(targets, predictions)):
        y0 = gap + row * (cell + header + gap)
        images = (
            Image.fromarray(target.astype(np.uint8) * 255, mode="L").convert("RGB"),
            Image.fromarray(prediction.astype(np.uint8) * 255, mode="L").convert("RGB"),
            overlay_image(target, prediction),
        )
        for col, (label, image) in enumerate(zip(labels, images)):
            x0 = gap + col * (cell + gap)
            draw.text((x0, y0), f"View {row}: {label}", fill=(38, 50, 56), font=font)
            panel.paste(image, (x0, y0 + header))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def find_legacy_mask(directory: Path, prefix: str, view_idx: int) -> Path:
    matches = sorted(directory.glob(f"{prefix}{view_idx}_*.png"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {prefix}{view_idx} PNG in {directory}, found {len(matches)}")
    return matches[0]


def audit_example(
    record: dict[str, Any],
    source: Path,
    output: Path,
    image_size: int,
    max_scale_error: float,
    max_centroid_shift_px: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    example = record["example"]
    source_dir = source / example
    output_dir = output / example
    target_dir = output_dir / "canonical_targets"
    projection_dir = output_dir / "canonical_projections"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = [str(path) for path in record["input_paths"]]
    loaded, _ = load_data_from_list(
        input_paths,
        True,
        (int(record["resolution"]), int(record["resolution"])),
        torch.device("cpu"),
        3,
        False,
    )
    masks = [np.asarray(mask).astype(np.float32) for mask in loaded]
    calibration = calibrate_tfs_projection(
        masks,
        search_res=min(96, int(record["resolution"])),
        fix_axis=bool(record.get("fixed", True)),
    )
    volume = np.load(record["selected"]["voxels"]).astype(bool)
    evaluation = evaluate_tfs_volume(volume, masks, calibration, image_size=image_size)

    canonical_targets = [mask.astype(bool) for mask in evaluation["gt_masks"][:3]]
    canonical_predictions = [mask.astype(bool) for mask in evaluation["pred_masks"][:3]]
    rows: list[dict[str, Any]] = []
    view_reports = []
    for view_idx, (target, prediction) in enumerate(zip(canonical_targets, canonical_predictions)):
        target_path = target_dir / f"target_view_{view_idx}.png"
        prediction_path = projection_dir / f"projection_view_{view_idx}.png"
        save_binary(target, target_path)
        save_binary(prediction, prediction_path)

        canonical = pair_metrics(target, prediction)
        raw_source_path = Path(input_paths[view_idx])
        raw_source = load_binary(raw_source_path, image_size)
        raw_to_canonical = pair_metrics(raw_source, target)
        legacy_target_path = find_legacy_mask(source_dir / "inputs", "input", view_idx)
        legacy_prediction_path = find_legacy_mask(source_dir / "projections", "projection", view_idx)
        legacy = pair_metrics(
            load_binary(legacy_target_path, image_size),
            load_binary(legacy_prediction_path, image_size),
        )
        scale_error = canonical["max_bbox_scale_error"]
        centroid_shift = canonical["centroid_shift_px"]
        alignment_pass = bool(
            scale_error is not None
            and centroid_shift is not None
            and scale_error <= max_scale_error
            and centroid_shift <= max_centroid_shift_px
        )
        evaluator_iou = float(evaluation["view_ious"][view_idx])
        evaluator_dice = float(evaluation["view_dices"][view_idx])
        metric_consistent = bool(
            abs(canonical["iou"] - evaluator_iou) <= 1e-12
            and abs(canonical["dice"] - evaluator_dice) <= 1e-12
        )
        view_report = {
            "view": view_idx,
            "canonical_target": str(target_path),
            "canonical_projection": str(prediction_path),
            "evaluator_iou": evaluator_iou,
            "evaluator_dice": evaluator_dice,
            "canonical": canonical,
            "raw_source_to_canonical_target": raw_to_canonical,
            "legacy_evidence": legacy,
            "metric_consistent": metric_consistent,
            "alignment_pass": alignment_pass,
        }
        view_reports.append(view_report)
        rows.append({
            "example": example,
            "view": view_idx,
            "canonical_iou": canonical["iou"],
            "canonical_dice": canonical["dice"],
            "legacy_evidence_iou": legacy["iou"],
            "legacy_evidence_dice": legacy["dice"],
            "raw_source_to_canonical_iou": raw_to_canonical["iou"],
            "raw_source_bbox_width_ratio": raw_to_canonical["bbox_width_ratio"],
            "raw_source_bbox_height_ratio": raw_to_canonical["bbox_height_ratio"],
            "raw_source_centroid_shift_px": raw_to_canonical["centroid_shift_px"],
            "target_bbox_width": canonical["target"]["bbox_width"],
            "target_bbox_height": canonical["target"]["bbox_height"],
            "projection_bbox_width": canonical["prediction"]["bbox_width"],
            "projection_bbox_height": canonical["prediction"]["bbox_height"],
            "bbox_width_ratio": canonical["bbox_width_ratio"],
            "bbox_height_ratio": canonical["bbox_height_ratio"],
            "centroid_shift_px": canonical["centroid_shift_px"],
            "alignment_pass": alignment_pass,
            "metric_consistent": metric_consistent,
        })

    build_panel(canonical_targets, canonical_predictions, output_dir / "canonical_evidence_panel.png")
    recomputed_mean_iou = float(np.mean(evaluation["view_ious"]))
    report = {
        "example": example,
        "source_directory": str(source_dir),
        "selected_volume": record["selected"]["voxels"],
        "calibration": jsonable(calibration),
        "reported_selected": record["selected"],
        "recomputed": {
            "mean_iou_all_6_views": recomputed_mean_iou,
            "mean_dice_all_6_views": float(np.mean(evaluation["view_dices"])),
            "view_ious_all_6": [float(value) for value in evaluation["view_ious"]],
            "view_dices_all_6": [float(value) for value in evaluation["view_dices"]],
            "reported_mean_iou_delta": recomputed_mean_iou - float(record["selected"]["mean_iou"]),
        },
        "views": view_reports,
        "all_metrics_consistent": all(view["metric_consistent"] for view in view_reports),
        "all_alignment_checks_pass": all(view["alignment_pass"] for view in view_reports),
    }
    (output_dir / "AUDIT.json").write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
    return report, rows


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(summary: dict[str, Any], path: Path) -> None:
    failed = [
        f"- `{row['example']}`, view {row['view']}: "
        f"width ratio {row['bbox_width_ratio']:.4f}, "
        f"height ratio {row['bbox_height_ratio']:.4f}, "
        f"centroid shift {row['centroid_shift_px']:.3f} px"
        for row in summary["rows"]
        if not row["alignment_pass"]
    ]
    failure_text = "\n".join(failed) if failed else "- None."
    text = f"""# Canonical Projection Evidence Audit

This directory is an isolated, read-only audit of `{summary['source']}`. The
source experiment and its legacy evidence files were not modified.

## Result

- Evaluated examples: {summary['n_examples']}
- Evaluated canonical views: {summary['n_views']}
- Canonical target/projection mean IoU: {summary['canonical_mean_iou']:.6f}
- Legacy raw-target/canonical-projection mean IoU: {summary['legacy_evidence_mean_iou']:.6f}
- Recomputed metrics exactly match the evaluator: {summary['all_metrics_consistent']}
- All strict scale/centroid checks pass: {summary['all_alignment_checks_pass']}

The high reported IoU is reproducible only in the evaluator's canonical mask
frame. `generation/datasets.py::get_props` independently scales each source
mask's foreground bounding box to 70% of the square canvas in x and y, then
centers it. Consequently, a raw source PNG and a canonical evaluator projection
are not a valid metric or evidence pair.

## Strict alignment exceptions

{failure_text}

The strict check uses a maximum bbox scale error of
{summary['thresholds']['max_scale_error']:.2%} and a maximum centroid shift of
{summary['thresholds']['max_centroid_shift_px']:.1f} px. An exception does not
invalidate the full-canvas IoU; it flags a visible geometric discrepancy for
manual review.

## Authoritative files

For every example:

- `canonical_targets/target_view_*.png`: exact evaluator-frame targets.
- `canonical_projections/projection_view_*.png`: exact evaluator projections.
- `canonical_evidence_panel.png`: target, projection, and error overlay.
- `AUDIT.json`: metrics, geometry, and provenance for each view.

At the root:

- `AUDIT.tsv`: flat per-view audit table.
- `AUDIT.json`: aggregate report.

Use these canonical pairs when showing or reporting evaluator IoU. If raw input
silhouettes are shown elsewhere, label them as source inputs and do not present
their side-by-side appearance as the evaluated pixel correspondence.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_manifest = args.source / "MANIFEST.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    if args.output.resolve() == args.source.resolve():
        raise ValueError("Output must differ from source; source results are immutable")

    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    rows: list[dict[str, Any]] = []
    for record in payload["examples"]:
        report, example_rows = audit_example(
            record,
            args.source,
            args.output,
            args.image_size,
            args.max_scale_error,
            args.max_centroid_shift_px,
        )
        reports.append(report)
        rows.extend(example_rows)
        print(
            f"{report['example']}: canonical_mean_iou="
            f"{report['recomputed']['mean_iou_all_6_views']:.6f} "
            f"metrics_ok={report['all_metrics_consistent']} "
            f"alignment_ok={report['all_alignment_checks_pass']}",
            flush=True,
        )

    write_tsv(rows, args.output / "AUDIT.tsv")
    summary = {
        "schema": "canonical_projection_evidence_audit_v1",
        "source": str(args.source),
        "output": str(args.output),
        "image_size": args.image_size,
        "thresholds": {
            "max_scale_error": args.max_scale_error,
            "max_centroid_shift_px": args.max_centroid_shift_px,
        },
        "n_examples": len(reports),
        "n_views": len(rows),
        "all_metrics_consistent": all(report["all_metrics_consistent"] for report in reports),
        "all_alignment_checks_pass": all(report["all_alignment_checks_pass"] for report in reports),
        "canonical_mean_iou": float(np.mean([row["canonical_iou"] for row in rows])),
        "legacy_evidence_mean_iou": float(np.mean([row["legacy_evidence_iou"] for row in rows])),
        "raw_source_to_canonical_mean_iou": float(
            np.mean([row["raw_source_to_canonical_iou"] for row in rows])
        ),
        "examples": reports,
    }
    (args.output / "AUDIT.json").write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    write_readme({**summary, "rows": rows}, args.output / "README.md")
    print(json.dumps({
        "output": str(args.output),
        "n_examples": summary["n_examples"],
        "canonical_mean_iou": summary["canonical_mean_iou"],
        "legacy_evidence_mean_iou": summary["legacy_evidence_mean_iou"],
        "raw_source_to_canonical_mean_iou": summary["raw_source_to_canonical_mean_iou"],
        "all_metrics_consistent": summary["all_metrics_consistent"],
        "all_alignment_checks_pass": summary["all_alignment_checks_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
