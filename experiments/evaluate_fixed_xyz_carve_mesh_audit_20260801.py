#!/usr/bin/env python3
"""Evaluate independently rendered STL silhouettes against saved targets."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_target(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127


def load_alpha(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.uint8)[..., 3] > 127


def normalize_bbox(mask: np.ndarray, size: int = 512, margin: int = 20) -> np.ndarray:
    ys, xs = np.where(mask)
    if not len(xs):
        return np.zeros((size, size), dtype=bool)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    usable = size - 2 * margin
    scale = min(usable / crop.shape[0], usable / crop.shape[1])
    new_h = max(1, int(round(crop.shape[0] * scale)))
    new_w = max(1, int(round(crop.shape[1] * scale)))
    resized = np.asarray(
        Image.fromarray(crop.astype(np.uint8) * 255).resize(
            (new_w, new_h),
            Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    ) > 127
    canvas = np.zeros((size, size), dtype=bool)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def d4(mask: np.ndarray) -> list[np.ndarray]:
    rotations = [np.rot90(mask, k) for k in range(4)]
    flipped = np.fliplr(mask)
    return rotations + [np.rot90(flipped, k) for k in range(4)]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / max(int(union), 1))


def best_pair(render: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    render = normalize_bbox(render)
    target = normalize_bbox(target)
    scores = [iou(candidate, target) for candidate in d4(render)]
    best = int(np.argmax(scores))
    return {"iou": float(scores[best]), "d4": best}


def evaluate_case(case_dir: Path) -> dict[str, object]:
    metadata = json.loads((case_dir / "render_metadata.json").read_text(encoding="utf-8"))
    targets = [load_target(Path(path)) for path in metadata["targets"]]
    axes = ["x", "y", "z"]
    renders = [load_alpha(case_dir / f"mesh_axis_{axis}.png") for axis in axes]

    best_result = None
    for permutation in itertools.permutations(range(3)):
        pairs = [best_pair(renders[permutation[index]], targets[index]) for index in range(3)]
        mean_iou = float(np.mean([pair["iou"] for pair in pairs]))
        min_iou = float(np.min([pair["iou"] for pair in pairs]))
        result = {
            "axis_assignment": [axes[index] for index in permutation],
            "views": pairs,
            "mean_iou": mean_iou,
            "min_iou": min_iou,
        }
        if best_result is None or (mean_iou, min_iou) > (
            best_result["mean_iou"],
            best_result["min_iou"],
        ):
            best_result = result

    assert best_result is not None
    return {
        "case": case_dir.name,
        "mesh": metadata["mesh"],
        "registration": "best axis permutation plus per-view D4 and bbox normalization",
        **best_result,
    }


def main() -> None:
    args = parse_args()
    rows = []
    for case_dir in sorted(args.renders.iterdir()):
        if (case_dir / "render_metadata.json").is_file():
            rows.append(evaluate_case(case_dir))
    rows.sort(key=lambda row: row["mean_iou"], reverse=True)
    payload = {"cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['case']}\tmean={row['mean_iou']:.4f}"
            f"\tmin={row['min_iou']:.4f}\t{','.join(row['axis_assignment'])}"
        )


if __name__ == "__main__":
    main()
