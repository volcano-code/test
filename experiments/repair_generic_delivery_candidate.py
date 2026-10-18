#!/usr/bin/env python3
"""Probe minimal envelope-constrained repairs for a generic delivery candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_dilation, convolve, generate_binary_structure, label
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadowforge_pipeline import unified_run as unified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--soft-dir", type=Path, required=True)
    parser.add_argument("--silhouettes", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--print-mm", type=float, default=100.0)
    parser.add_argument("--tsf", type=int, choices=(0, 1), default=1)
    parser.add_argument("--mr", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-iou", type=float, default=0.95)
    return parser.parse_args()


def projection_scores(
    volume: np.ndarray,
    targets: list[np.ndarray],
    protocol: SimpleNamespace,
) -> dict:
    renders = unified._generic_render(volume, protocol)
    scores = []
    for pred, target in zip(renders, targets):
        intersection = int(np.logical_and(pred, target).sum())
        union = int(np.logical_or(pred, target).sum())
        scores.append(
            {
                "iou": intersection / max(union, 1),
                "dice": 2.0 * intersection / max(int(pred.sum() + target.sum()), 1),
                "ssim": structural_similarity(
                    target.astype(np.float32),
                    pred.astype(np.float32),
                    data_range=1.0,
                ),
            }
        )
    return {
        "mean_iou": float(np.mean([score["iou"] for score in scores])),
        "min_iou": float(min(score["iou"] for score in scores)),
        "mean_dice": float(np.mean([score["dice"] for score in scores])),
        "mean_ssim": float(np.mean([score["ssim"] for score in scores])),
        "view_ious": [float(score["iou"]) for score in scores],
        "renders": renders,
    }


def ranked_frontier(volume: np.ndarray, hull: np.ndarray) -> np.ndarray:
    frontier = binary_dilation(volume, structure=unified.S6) & hull & ~volume
    coordinates = np.argwhere(frontier)
    if not len(coordinates):
        return coordinates
    neighbor_count = convolve(
        volume.astype(np.uint8),
        unified.S6.astype(np.uint8),
        mode="constant",
        cval=0,
    )
    scores = neighbor_count[
        coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    ]
    return coordinates[np.argsort(-scores, kind="stable")]


def add_fraction(
    volume: np.ndarray,
    ranked: np.ndarray,
    fraction: float,
) -> np.ndarray:
    repaired = volume.copy()
    count = int(np.ceil(len(ranked) * fraction))
    selected = ranked[:count]
    if len(selected):
        repaired[selected[:, 0], selected[:, 1], selected[:, 2]] = True
    return repaired


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
    candidate = np.load(args.candidate).astype(bool)
    targets = unified._generic_targets(args.soft_dir, candidate.shape[0])
    hull_zyx = unified._generic_visual_hull(
        targets, protocol, candidate.shape[0]
    )
    hull = np.transpose(hull_zyx, (2, 1, 0))
    hull_voxels = int(hull.sum())

    from voxel_ops import connect6_min_bridges

    clipped = candidate & hull
    variants: list[tuple[str, np.ndarray]] = [
        ("original", candidate),
        ("clipped", clipped),
    ]
    for radius in (0, 1):
        connected, _ = connect6_min_bridges(
            clipped, hull, max_passes=48, bridge_radius=radius
        )
        variants.append((f"reconnect_r{radius}", connected))
        if radius == 0:
            ranked = ranked_frontier(connected, hull)
            for fraction in (0.125, 0.25, 0.5):
                variants.append(
                    (
                        f"reconnect_r0_local_fraction{fraction:g}",
                        add_fraction(connected, ranked, fraction),
                    )
                )

    screened = []
    eligible_variants: list[tuple[str, np.ndarray, dict]] = []
    for name, volume in variants:
        score = projection_scores(volume, targets, protocol)
        row = {
            "variant": name,
            "occupied_voxels": int(volume.sum()),
            "added_voxels": int(volume.sum() - candidate.sum()),
            "components": int(
                label(volume, structure=generate_binary_structure(3, 1))[1]
            ),
            "saving": float(1.0 - volume.sum() / max(hull_voxels, 1)),
            **{key: value for key, value in score.items() if key != "renders"},
        }
        screened.append(row)
        print(json.dumps({"stage": "screen", **row}), flush=True)
        if score["mean_iou"] >= args.min_iou and row["components"] == 1:
            eligible_variants.append((name, volume, score))

    if not eligible_variants:
        eligible_variants = [max(
            (
                (name, volume, projection_scores(volume, targets, protocol))
                for name, volume in variants
            ),
            key=lambda item: item[2]["mean_iou"],
        )]

    selected_name = ""
    selected_volume = eligible_variants[0][1]
    full = {}
    full_checks = []
    evaluated_variants = []
    for name, volume, _ in eligible_variants:
        candidate_full = unified._generic_metrics(
            volume, targets, hull_voxels, protocol
        )
        candidate_full["mean_ssim"] = projection_scores(
            volume, targets, protocol
        )["mean_ssim"]
        full_checks.append(
            {
                "variant": name,
                **{
                    key: value
                    for key, value in candidate_full.items()
                    if key not in {"print_volume", "renders"}
                },
            }
        )
        evaluated_variants.append((name, volume, candidate_full))
        print(
            json.dumps(
                {
                    "stage": "mesh_qa",
                    "variant": name,
                    "mesh_outer_parts": candidate_full["mesh_outer_parts"],
                    "watertight": candidate_full["watertight"],
                }
            ),
            flush=True,
        )
        if candidate_full["watertight"] and candidate_full["mesh_outer_parts"] == 1:
            selected_name, selected_volume, full = name, volume, candidate_full
            break
    if not full:
        # Diagnostic fallback: prefer fidelity, then the lighter candidate.
        selected_name, selected_volume, full = max(
            evaluated_variants,
            key=lambda item: (
                item[2]["mean_iou"],
                item[2]["min_iou"],
                item[2]["saving"],
            ),
        )
    np.save(args.output / "repaired_delivery256.npy", selected_volume.astype(np.uint8))
    unified.export_stl_at_print_scale(
        selected_volume, args.output / "repaired_display_100mm.stl", args.print_mm
    )
    unified.export_stl_at_print_scale(
        full["print_volume"],
        args.output / f"repaired_print_{full['print_orientation']}_100mm.stl",
        args.print_mm,
    )
    projection_dir = args.output / "projections"
    projection_dir.mkdir(exist_ok=True)
    selected_projection = projection_scores(selected_volume, targets, protocol)
    for index, (target, render) in enumerate(
        zip(targets, selected_projection["renders"], strict=True)
    ):
        unified.n2.save_mask(target, projection_dir / f"target_view_{index}.png")
        unified.n2.save_mask(render, projection_dir / f"projection_view_{index}.png")
    serializable_full = {
        key: value
        for key, value in full.items()
        if key not in {"print_volume", "renders"}
    }
    report = {
        "selected_variant": selected_name,
        "selected": serializable_full,
        "full_checks": full_checks,
        "screened": screened,
    }
    (args.output / "repair_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
