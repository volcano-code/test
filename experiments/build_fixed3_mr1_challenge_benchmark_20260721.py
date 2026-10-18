#!/usr/bin/env python3
"""Freeze a difficult three-silhouette, mirror-enabled benchmark.

Selection uses input masks only.  No method output, projection score, mesh QA,
or eligibility status participates in choosing the tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
COMPLEX = ROOT / "data/silhouettes/complex_v2_filled"
ANIME = Path("/home/lin/xvxinyao/shadow/anime_shadow_processed_black_bg_light_simplified")


def load_mask(path: Path, resolution: int = 256) -> np.ndarray:
    image = Image.open(path).convert("L").resize((resolution, resolution), Image.Resampling.NEAREST)
    mask = np.asarray(image) >= 128
    if mask.mean() > 0.5:
        mask = ~mask
    return mask


def metrics(path: Path) -> dict[str, float | int]:
    mask = load_mask(path)
    structure = np.ones((3, 3), dtype=bool)
    labels, count = ndimage.label(mask, structure)
    component_sizes = np.bincount(labels.ravel())[1:]
    significant_components = int(np.count_nonzero(component_sizes >= 16))
    filled = ndimage.binary_fill_holes(mask)
    hole_labels, _ = ndimage.label(filled & ~mask, structure)
    hole_sizes = np.bincount(hole_labels.ravel())[1:]
    holes = int(np.count_nonzero(hole_sizes >= 16))
    eroded = ndimage.binary_erosion(mask, structure=structure)
    boundary = mask & ~eroded
    distance = ndimage.distance_transform_edt(mask)
    area = int(mask.sum())
    mirrored = np.fliplr(mask)
    mirror_union = int(np.logical_or(mask, mirrored).sum())
    mirror_iou = float(np.logical_and(mask, mirrored).sum() / max(mirror_union, 1))
    return {
        "foreground_fraction": float(mask.mean()),
        "components_8_ge16": significant_components,
        "holes_8_ge16": holes,
        "boundary_over_sqrt_area": float(boundary.sum() / np.sqrt(max(area, 1))),
        "thin_fraction_r2": float(np.logical_and(mask, distance <= 2.0).sum() / max(area, 1)),
        "mirror_asymmetry": float(1.0 - mirror_iou),
    }


def robust_normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(values, [10, 90])
    return np.clip((values - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)


def rank_pool(paths: list[Path]) -> list[dict]:
    rows = [{"path": str(path.resolve()), "metrics": metrics(path)} for path in paths]
    names = ("boundary_over_sqrt_area", "thin_fraction_r2", "mirror_asymmetry", "components_8_ge16", "holes_8_ge16")
    normalized = {
        name: robust_normalize(np.asarray([float(row["metrics"][name]) for row in rows]))
        for name in names
    }
    for index, row in enumerate(rows):
        m = row["metrics"]
        area_ok = 0.035 <= float(m["foreground_fraction"]) <= 0.42
        row["eligible_input"] = bool(area_ok and int(m["components_8_ge16"]) <= 8)
        row["difficulty_score"] = float(
            0.30 * normalized["boundary_over_sqrt_area"][index]
            + 0.25 * normalized["thin_fraction_r2"][index]
            + 0.20 * normalized["mirror_asymmetry"][index]
            + 0.15 * normalized["components_8_ge16"][index]
            + 0.10 * normalized["holes_8_ge16"][index]
        )
    return sorted(rows, key=lambda row: (-row["eligible_input"], -row["difficulty_score"], row["path"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/fixed3_mr1_challenge_frozen_20260721/benchmark_manifest.json",
    )
    args = parser.parse_args()

    complex_rows = rank_pool(sorted(COMPLEX.glob("*.png")))
    anime_rows = [row for row in rank_pool(sorted(ANIME.glob("*.png"))) if row["eligible_input"]]
    if len(complex_rows) < args.tasks or len(anime_rows) < 2 * args.tasks:
        raise RuntimeError("Insufficient input silhouettes for the requested frozen benchmark")

    # Interleave the upper difficulty range rather than taking adjacent files.
    # This preserves high complexity while avoiding nearly duplicate anime masks.
    anime_pick = anime_rows[: 3 * args.tasks :]
    left = anime_pick[::2][: args.tasks]
    right = anime_pick[1::2][: args.tasks]
    if len(left) < args.tasks or len(right) < args.tasks:
        left = anime_rows[:args.tasks]
        right = anime_rows[args.tasks : 2 * args.tasks]

    cases = []
    for index in range(args.tasks):
        selected = [complex_rows[index], left[index], right[index]]
        xyz = [row["path"] for row in selected]
        cases.append({
            "case_id": f"challenge_{index + 1:02d}",
            "stratum": "input-only-fabrication-challenging",
            "target_paths_xyz": xyz,
            # Sadekar TFS uses side/front/top ordering Y, X, Z.
            "sadekar_tfs_targets": [xyz[1], xyz[0], xyz[2]],
            "input_only_selection": {
                "difficulty_score": float(np.mean([row["difficulty_score"] for row in selected])),
                "per_view": [{"path": row["path"], **row["metrics"], "score": row["difficulty_score"]} for row in selected],
            },
        })

    payload = {
        "schema": "fixed3_mr1_challenge_v1",
        "purpose": "method-independent challenge benchmark and Figure 4 evidence source",
        "selection_policy": (
            "Frozen before method execution using only foreground fraction, boundary complexity, "
            "thin-feature fraction, significant components, holes, and mirror asymmetry."
        ),
        "protocol": {
            "views": 3,
            "declared_rendered_views": 6,
            "axes": ["x", "y", "z"],
            "camera": "fixed orthographic TFS",
            "tsf": 1,
            "mr": 1,
            "zdist": 1.7,
            "resolution": 256,
            "seeds": [17, 29, 43],
            "evaluation": "shared original masks plus their prescribed mirror views; no registration or per-method alignment",
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "tasks": len(cases),
        "mean_input_difficulty": float(np.mean([case["input_only_selection"]["difficulty_score"] for case in cases])),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
