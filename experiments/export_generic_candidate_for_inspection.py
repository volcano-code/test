#!/usr/bin/env python3
"""Rebuild and export one generic-protocol candidate for visual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadowforge_pipeline import unified_run as unified

n2 = unified.n2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-dir", type=Path, required=True)
    parser.add_argument("--silhouettes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--core-resolution", type=int, default=48)
    parser.add_argument("--delivery-shell-voxels", type=int, default=1)
    parser.add_argument("--bridge-radius", type=int, default=2)
    parser.add_argument("--bridge-passes", type=int, default=48)
    parser.add_argument("--print-mm", type=float, default=100.0)
    parser.add_argument("--tsf", type=int, choices=(0, 1), default=1)
    parser.add_argument("--mr", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def contact_sheet(
    targets: list[np.ndarray], renders: list[np.ndarray], path: Path
) -> None:
    size = 256
    margin = 20
    label_height = 28
    canvas = Image.new(
        "RGB",
        (len(targets) * size, 2 * (size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (target, render) in enumerate(zip(targets, renders)):
        x = index * size
        draw.text((x + margin, 6), f"Target {index}", fill="black")
        draw.text((x + margin, size + label_height + 6), f"Render {index}", fill="black")
        target_image = Image.fromarray((target.astype(np.uint8) * 255), mode="L").convert("RGB")
        render_image = Image.fromarray((render.astype(np.uint8) * 255), mode="L").convert("RGB")
        canvas.paste(target_image, (x, label_height))
        canvas.paste(render_image, (x, size + 2 * label_height))
    canvas.save(path)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = SimpleNamespace(
        silhouettes=args.silhouettes,
        tsf=args.tsf,
        mr=args.mr,
        device=args.device,
        print_mm=args.print_mm,
    )

    probability_zyx = np.load(
        args.soft_dir / "occupancy_probability_core.npy"
    ).astype(np.float32)
    # Match canonical_voxel_soft(): the trainer image frame reverses both
    # projected image axes relative to the fixed delivery geometry frame.
    probability = np.flip(
        np.transpose(probability_zyx, (2, 1, 0)), axis=(0, 2)
    )
    targets_256 = unified._generic_targets(args.soft_dir, 256)
    targets_core = unified._generic_targets(args.soft_dir, args.core_resolution)
    hull_256_zyx = unified._generic_visual_hull(targets_256, protocol, 256)
    hull_core_zyx = unified._generic_visual_hull(
        targets_core, protocol, args.core_resolution
    )
    hull_256 = np.transpose(hull_256_zyx, (2, 1, 0))
    hull_core = np.transpose(hull_core_zyx, (2, 1, 0))
    guidance_core = np.transpose(
        unified.downsample_max_to(probability, args.core_resolution), (2, 1, 0)
    )

    seeds = unified._generic_seed_trajectory(
        guidance_core,
        hull_core,
        (args.threshold,),
        args.bridge_radius,
        args.bridge_passes,
        bridge=True,
    )
    seed_name, scaffold = seeds[0]
    trajectory = unified._generic_tfs_carve_trajectory(
        scaffold,
        guidance_core,
        hull_core,
        seed_name,
        len(args.silhouettes),
        args.bridge_radius,
        args.bridge_passes,
    )
    if args.phase >= len(trajectory):
        raise ValueError(
            f"Requested phase {args.phase}, but trajectory has {len(trajectory) - 1} carve phases"
        )
    candidate_name, core = trajectory[args.phase]
    shell = hull_256 & ~binary_erosion(
        hull_256,
        structure=unified.S6,
        iterations=args.delivery_shell_voxels,
    )
    delivered = unified._generic_connected_delivery(
        core, hull_256, shell, include_shell=True
    )
    metrics = unified._generic_metrics(
        delivered, targets_256, int(hull_256.sum()), protocol
    )

    np.save(args.output / "candidate_core48.npy", core.astype(np.uint8))
    np.save(args.output / "candidate_delivery256.npy", delivered.astype(np.uint8))
    unified.export_stl_at_print_scale(
        delivered, args.output / "candidate_display_100mm.stl", args.print_mm
    )
    unified.export_stl_at_print_scale(
        metrics["print_volume"],
        args.output / f"candidate_print_{metrics['print_orientation']}_100mm.stl",
        args.print_mm,
    )
    projection_dir = args.output / "projections"
    projection_dir.mkdir(exist_ok=True)
    for index, (target, render) in enumerate(zip(targets_256, metrics["renders"])):
        n2.save_mask(target, projection_dir / f"target_view_{index}.png")
        n2.save_mask(render, projection_dir / f"projection_view_{index}.png")
    contact_sheet(
        targets_256, metrics["renders"], args.output / "target_vs_projection.png"
    )

    serializable = {
        key: value
        for key, value in metrics.items()
        if key not in {"print_volume", "renders"}
    }
    serializable.update(
        {
            "candidate": candidate_name,
            "inspection_only": True,
            "reason": "Exported despite failed formal eligibility for visual diagnosis.",
        }
    )
    (args.output / "metrics.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
