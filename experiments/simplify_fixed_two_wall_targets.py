#!/usr/bin/env python3
"""Create conservative, fixed-camera silhouette copies for physical delivery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def foreground_mask(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return the minority binary region and whether foreground is bright."""
    bright = image > 127
    foreground_is_bright = bool(bright.mean() < 0.5)
    return (bright if foreground_is_bright else ~bright), foreground_is_bright


def simplify(image: np.ndarray, close_radius: int) -> tuple[np.ndarray, dict]:
    mask, foreground_is_bright = foreground_mask(image)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    main = labels == largest_label

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1)
    )
    repaired = cv2.morphologyEx(main.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    repaired_count, repaired_labels, repaired_stats, _ = cv2.connectedComponentsWithStats(repaired, 8)
    repaired_largest = 1 + int(np.argmax(repaired_stats[1:, cv2.CC_STAT_AREA]))
    repaired = repaired_labels == repaired_largest

    output = np.full(image.shape, 255 if foreground_is_bright else 0, dtype=np.uint8)
    output[repaired] = 0 if foreground_is_bright else 255
    report = {
        "input_components": int(count - 1),
        "input_foreground_pixels": int(mask.sum()),
        "retained_largest_component_pixels": int(main.sum()),
        "output_components": int(repaired_count - 1),
        "output_foreground_pixels": int(repaired.sum()),
        "removed_foreground_pixels": int(mask.sum() - main.sum()),
        "close_radius_px": close_radius,
        "foreground_is_bright": foreground_is_bright,
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--close-radius", type=int, default=1)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for source in map(Path, args.inputs):
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(source)
        output, info = simplify(image, args.close_radius)
        target = out / source.name
        if not cv2.imwrite(str(target), output):
            raise RuntimeError(f"Could not write {target}")
        report[source.name] = {"source": str(source.resolve()), "output": str(target.resolve()), **info}
    (out / "SIMPLIFICATION_MANIFEST.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
