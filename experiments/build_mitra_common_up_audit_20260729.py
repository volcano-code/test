#!/usr/bin/env python3
"""Build an isolated Mitra shadow hull in a shared-Z-up camera frame.

The baseline mask arrays use image row/column order, while the Blender corner
scene uses a shared physical Z-up convention. Rotating each square receiver
mask counter-clockwise maps image rows to physical receiver coordinates before
the generalized-cone intersection. The source silhouettes are not modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mitra_pauly2009_shadow_hull import (
    build_shadow_hull,
    read_mask,
    save_binary,
    save_obj,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silhouettes", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    source_masks = [
        read_mask(path, args.resolution) for path in args.silhouettes
    ]
    calibrated_masks = [np.rot90(mask, 1) for mask in source_masks]
    hull = build_shadow_hull(calibrated_masks, ["x", "y", "z"])

    for index, (source, calibrated) in enumerate(
        zip(source_masks, calibrated_masks)
    ):
        save_binary(source, args.output / f"source_view_{index}.png")
        save_binary(calibrated, args.output / f"calibrated_view_{index}.png")

    np.save(args.output / "mitra_common_up_hull.npy", hull.astype(np.uint8))
    save_obj(hull, args.output / "mitra_common_up_hull.obj")
    metadata = {
        "purpose": "isolated Blender orientation audit",
        "source_silhouettes": [str(path.resolve()) for path in args.silhouettes],
        "axes": ["x", "y", "z"],
        "mask_transform": "90 degrees counter-clockwise into shared-Z-up receiver frame",
        "occupied_voxels": int(hull.sum()),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
