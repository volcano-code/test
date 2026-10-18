#!/usr/bin/env python3
"""Compute matched Dice and binary-mask SSIM for the connected fixed-3 suite."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


ROOT = Path("/home/lin/xvxinyao/ShadowForge")


def mask(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"))
    corners = np.r_[image[:5, :5].ravel(), image[-5:, -5:].ravel()]
    return image > 127 if np.median(corners) < 127 else image < 127


def pair_scores(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    intersection = np.logical_and(prediction, target).sum()
    dice = 2.0 * intersection / max(prediction.sum() + target.sum(), 1)
    ssim = structural_similarity(
        target.astype(np.float32), prediction.astype(np.float32), data_range=1.0
    )
    return float(dice), float(ssim)


def summarize(task_values: dict[str, list[tuple[float, float]]]) -> dict:
    per_task = {
        case: np.mean(np.asarray(values, dtype=np.float64), axis=0)
        for case, values in task_values.items()
    }
    values = np.asarray(list(per_task.values()), dtype=np.float64)
    return {
        "tasks": len(per_task),
        "mean_dice": float(values[:, 0].mean()),
        "std_dice": float(values[:, 0].std()),
        "mean_ssim": float(values[:, 1].mean()),
        "std_ssim": float(values[:, 1].std()),
        "per_task": {
            case: {"dice": float(value[0]), "ssim": float(value[1])}
            for case, value in sorted(per_task.items())
        },
    }


def score_image_pairs(targets: list[Path], predictions: list[Path]) -> tuple[float, float]:
    scores = [pair_scores(mask(pred), mask(target)) for target, pred in zip(targets, predictions, strict=True)]
    return tuple(np.mean(np.asarray(scores), axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/fixed3_mr1_connected10_comparison_20260722/dice_ssim.json",
    )
    args = parser.parse_args()

    methods: dict[str, dict[str, list[tuple[float, float]]]] = {
        name: defaultdict(list)
        for name in ("Mitra-style", "Sadekar voxel", "Sadekar mesh", "P-CRAFT")
    }

    mitra = ROOT / "results/fixed3_mr1_connected10_mitra_20260722"
    for case_dir in sorted(mitra.glob("connected_*")):
        targets = sorted(case_dir.glob("target_view*.png"))
        predictions = sorted(case_dir.glob("rendered_view*.png"))
        methods["Mitra-style"][case_dir.name].append(score_image_pairs(targets, predictions))

    sad_json = ROOT / "results/fixed3_mr1_connected10_sadekar_20260722/unified_eval/unified_metrics.json"
    sad = json.loads(sad_json.read_text())
    manifest_path = ROOT / "results/fixed3_mr1_connected10_pending_20260722/benchmark_manifest.json"
    cases = {row["case_id"]: row for row in json.loads(manifest_path.read_text())["cases"]}
    for row in sad["rows"]:
        artifact = Path(row["artifact"])
        kind = row["method"].split()[-1]
        pattern = "sample_0view_{}_silh.png" if kind == "voxel" else "sample_0_pred_view_{}.png"
        predictions = [artifact.parent / pattern.format(index) for index in range(6)]
        base_targets = [Path(path) for path in cases[row["case_id"]]["sadekar_tfs_targets"]]
        base_masks = [mask(path) for path in base_targets]
        target_masks = base_masks + [np.fliplr(value) for value in base_masks]
        scores = [
            pair_scores(mask(prediction), target)
            for prediction, target in zip(predictions, target_masks, strict=True)
        ]
        methods[row["method"]][row["case_id"]].append(tuple(np.mean(scores, axis=0)))

    pcraft = ROOT / "results/fixed3_mr1_connected10_pcraft_20260722"
    for case_dir in sorted(pcraft.glob("connected_*")):
        for seed_dir in sorted(case_dir.glob("seed_*")):
            projection_dir = seed_dir / "projections"
            targets = [projection_dir / f"target_view_{index}.png" for index in range(6)]
            predictions = [projection_dir / f"projection_view_{index}.png" for index in range(6)]
            methods["P-CRAFT"][case_dir.name].append(score_image_pairs(targets, predictions))

    result = {method: summarize(values) for method, values in methods.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
