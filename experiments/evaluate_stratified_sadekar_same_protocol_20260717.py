#!/usr/bin/env python3
"""Evaluate Sadekar author outputs on the frozen 12-task fixed-TFS suite.

The author renderer was invoked with the same three target masks, orthographic
TFS camera, image resolution, mirror setting, and z distance for every run.
Consequently, silhouette metrics are computed directly from its saved binary
renders; no per-case pose search, rescaling, or post-hoc registration is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import generate_binary_structure, label


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
S6 = generate_binary_structure(3, 1)


def binary_mask(path: Path, resolution: int = 256) -> np.ndarray:
    image = Image.open(path).convert("L").resize((resolution, resolution), Image.Resampling.NEAREST)
    array = np.asarray(image)
    corners = np.r_[array[:5, :5].ravel(), array[-5:, -5:].ravel()]
    return array > 127 if np.median(corners) < 127 else array < 127


def author_training_mask(path: Path, resolution: int = 256) -> np.ndarray:
    """Reproduce the target normalization applied by Sadekar's data loader."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    corners = np.r_[image[:5, :5].ravel(), image[-5:, -5:].ravel()]
    if np.median(corners) > 127:
        image = 255 - image
    image = cv2.resize(image, (resolution, resolution))
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No foreground contour in {path}")
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    target_extent = int(resolution * 0.7)
    center = resolution // 2
    scale_x = target_extent / width
    scale_y = target_extent / height
    contour_center_x = x + width / 2
    contour_center_y = y + height / 2
    transform = np.float32([
        [scale_x, 0, center * (1 - scale_x) + (center - contour_center_x)],
        [0, scale_y, center * (1 - scale_y) + (center - contour_center_y)],
    ])
    normalized = cv2.warpAffine(image, transform, (resolution, resolution))
    return normalized > 127


def scores(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict:
    ious, dices = [], []
    for prediction, target in zip(predictions, targets, strict=True):
        intersection = int(np.logical_and(prediction, target).sum())
        union = int(np.logical_or(prediction, target).sum())
        denominator = int(prediction.sum() + target.sum())
        ious.append(intersection / max(union, 1))
        dices.append(2 * intersection / max(denominator, 1))
    return {
        "mean_iou": float(np.mean(ious)),
        "min_view_iou": float(np.min(ious)),
        "mean_dice": float(np.mean(dices)),
        "view_ious": ious,
        "view_dices": dices,
    }


def mesh_qa(path: Path) -> dict:
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    parts = int(len(mesh.split(only_watertight=False)))
    return {
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_parts": parts,
        "single_body": bool(parts == 1),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
    }


def voxel_qa(path: Path) -> dict:
    volume = np.squeeze(np.load(path, mmap_mode="r")) > 0.05
    _, components = label(volume, structure=S6)
    return {
        "occupied_voxels": int(volume.sum()),
        "components6": int(components),
        "single_body": bool(components == 1),
    }


def predicted_paths(kind: str, artifact: Path, count: int) -> list[Path]:
    folder = artifact.parent
    pattern = "sample_0view_{}_silh.png" if kind == "voxel" else "sample_0_pred_view_{}.png"
    return [folder / pattern.format(index) for index in range(count)]


def aggregate(rows: list[dict], method: str, prefix: str = "") -> dict:
    selected = [row for row in rows if row["method"] == method]
    cases = sorted({row["case_id"] for row in selected})
    per_case = []
    for case_id in cases:
        case_rows = [row for row in selected if row["case_id"] == case_id]
        per_case.append({
            "case_id": case_id,
            "mean_iou": float(np.mean([row[f"{prefix}mean_iou"] for row in case_rows])),
            "mean_dice": float(np.mean([row[f"{prefix}mean_dice"] for row in case_rows])),
            "min_view_iou": float(np.min([row[f"{prefix}min_view_iou"] for row in case_rows])),
            "single_body": float(np.mean([row.get("single_body", False) for row in case_rows])),
            "watertight": float(np.mean([row.get("mesh_watertight", False) for row in case_rows])),
        })
    return {
        "method": method,
        "tasks": len(cases),
        "seeds_per_task": len(selected) // max(len(cases), 1),
        "mean_iou": float(np.mean([row["mean_iou"] for row in per_case])),
        "std_iou_across_tasks": float(np.std([row["mean_iou"] for row in per_case])),
        "mean_dice": float(np.mean([row["mean_dice"] for row in per_case])),
        "std_dice_across_tasks": float(np.std([row["mean_dice"] for row in per_case])),
        "worst_view_iou": float(np.min([row["min_view_iou"] for row in per_case])),
        "single_body_rate": float(np.mean([row["single_body"] for row in per_case])),
        "watertight_rate": float(np.mean([row["watertight"] for row in per_case])),
        "per_case_seed_averages": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/mitra_stratified_low6_high6_fixed3_20260716/benchmark_manifest.json",
    )
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=ROOT / "results/mitra_stratified_low6_high6_author_tsf_20260716/baseline_run_summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/mitra_stratified_low6_high6_unified_eval_20260717/sadekar",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Evaluate only the named benchmark case(s); the default evaluates the full manifest.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    summary = json.loads(args.run_summary.read_text())
    mirror = bool(manifest.get("protocol", {}).get("mr", False))
    base_views = int(manifest.get("protocol", {}).get("views", 3))
    declared_views = base_views * (2 if mirror else 1)
    selected_cases = [
        row for row in manifest["cases"] if not args.case or row["case_id"] in set(args.case)
    ]
    if args.case and len(selected_cases) != len(set(args.case)):
        known = {row["case_id"] for row in manifest["cases"]}
        missing_cases = sorted(set(args.case) - known)
        raise ValueError(f"Unknown case id(s): {missing_cases}")
    cases = {row["case_id"]: row for row in selected_cases}
    runs = [
        row for row in summary["runs"]
        if row.get("status") in {"completed", "reused"} and row["case_id"] in cases
    ]
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for run in runs:
        case = cases[run["case_id"]]
        artifact = Path(run["artifact"])
        render_paths = predicted_paths(run["kind"], artifact, declared_views)
        required = [artifact, *render_paths]
        absent = [str(path) for path in required if not path.is_file()]
        if absent:
            missing.extend(absent)
            continue
        original_targets = [binary_mask(Path(path)) for path in case["sadekar_tfs_targets"]]
        author_targets = [author_training_mask(Path(path)) for path in case["sadekar_tfs_targets"]]
        if mirror:
            original_targets += [np.fliplr(mask) for mask in original_targets]
            author_targets += [np.fliplr(mask) for mask in author_targets]
        predictions = [binary_mask(path) for path in render_paths]
        qa = voxel_qa(artifact) if run["kind"] == "voxel" else {}
        output_mesh = artifact.parent / "sample_0_output.obj"
        if output_mesh.is_file():
            qa.update(mesh_qa(output_mesh))
        original_scores = scores(predictions, original_targets)
        author_scores = scores(predictions, author_targets)
        rows.append({
            "case_id": run["case_id"],
            "stratum": case.get("stratum", "unspecified"),
            "method": f"Sadekar {run['kind']}",
            "seed": int(run["seed"]),
            "artifact": str(artifact),
            **original_scores,
            **{f"author_target_{key}": value for key, value in author_scores.items()},
            **qa,
        })

    seeds = manifest.get("protocol", {}).get("seeds", [17, 29, 43])
    expected = len(cases) * len(seeds) * 2
    if missing or len(rows) != expected:
        raise RuntimeError(f"Incomplete evaluation: rows={len(rows)}/{expected}, missing={len(missing)}")

    aggregates = [aggregate(rows, "Sadekar voxel"), aggregate(rows, "Sadekar mesh")]
    author_target_aggregates = [
        aggregate(rows, "Sadekar voxel", "author_target_"),
        aggregate(rows, "Sadekar mesh", "author_target_"),
    ]
    protocol = {
        **manifest["protocol"],
        "seeds": seeds,
        "mirror_mode": mirror,
        "declared_rendered_views": declared_views,
        "metric_source": "author-code saved binary renders under the frozen camera",
        "target_warning": (
            "Sadekar applies anisotropic 70%-extent centering to every input mask. "
            "Scores against original targets and author-preprocessed targets are reported separately."
        ),
        "registration": "none",
        "aggregation": "average seeds within each task, then average the 12 task means",
    }
    payload = {
        "protocol": protocol,
        "aggregate_against_original_targets": aggregates,
        "aggregate_against_author_preprocessed_targets": author_target_aggregates,
        "aggregate": aggregates,
        "rows": rows,
    }
    (args.output / "unified_metrics.json").write_text(json.dumps(payload, indent=2))
    with (args.output / "per_run_metrics.csv").open("w", newline="") as handle:
        fields = ["case_id", "stratum", "method", "seed", "mean_iou", "min_view_iou", "mean_dice", "single_body", "mesh_watertight", "mesh_parts", "artifact"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "aggregate": aggregates}, indent=2))


if __name__ == "__main__":
    main()
