#!/usr/bin/env python3
"""Screen one selected delivered occupancy with a frozen PINO/ACS pair."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shadowforge_pipeline import unified_run as unified
from shadowforge_pipeline.run import StructuralScreen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--acs-policy", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.run / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    selected = manifest["selected"]
    volume = np.load(args.run / "selected_display_voxels256.npy").astype(bool)

    left_img = unified.n2.load_mask(args.run / "inputs/left_wall_x_target.png", 256)
    right_img = unified.n2.load_mask(args.run / "inputs/right_wall_y_target.png", 256)
    left = unified.tw.image_to_wall_axes(left_img)
    right = unified.tw.image_to_wall_axes(right_img)
    hull = unified.tw.tw.visual_hull_two_wall(left, right)

    orientation = selected["print_orientation"]
    oriented_volume = dict(unified.print_orientations(volume))[orientation]
    oriented_hull = dict(unified.print_orientations(hull))[orientation]
    proxy, proxy_hull = unified.structural_proxy(oriented_volume, oriented_hull, 48)

    screen_args = Namespace(
        screen="pino_acs_fem",
        device=args.device,
        pino_checkpoint=args.checkpoint.resolve(),
        acs_policy=args.acs_policy.resolve(),
        sf_threshold=1.3,
        load_multiplier=1.0,
        alpha=0.01,
    )
    screen = StructuralScreen(screen_args)
    result = screen.evaluate(proxy, proxy_hull)
    result.update({
        "candidate": selected["candidate"],
        "source_run": str(args.run.resolve()),
        "print_orientation": orientation,
        "structural_resolution": 48,
        "structural_representation": "max-pooled delivered print orientation",
        "checkpoint": str(args.checkpoint.resolve()),
        "acs_policy_file": str(args.acs_policy.resolve()),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
