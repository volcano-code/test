#!/usr/bin/env python3
"""Compare C48+V256 delivery with direct V256 export on one frozen soft field."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure, label

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.voxel_ops import connect6_min_bridges
from shadowforge_pipeline.unified_run import (
    _canonical_xyz_from_trainer_zyx,
    _generic_render,
    _generic_targets,
    _generic_visual_hull,
    downsample_max_to,
    export_stl_at_print_scale,
    mesh_qa,
    upsample_nearest_to,
)


S6 = generate_binary_structure(3, 1)


def component_stats(volume: np.ndarray) -> dict:
    labs, count = label(np.asarray(volume, dtype=bool), structure=S6)
    sizes = np.bincount(labs.ravel())[1:]
    sizes = np.sort(sizes)[::-1]
    total = int(sizes.sum())
    significant_cutoff = max(64, int(np.ceil(total * 1.0e-5)))
    return {
        "components_6": int(count),
        "significant_components_6": int(np.count_nonzero(sizes >= significant_cutoff)),
        "significant_component_cutoff": int(significant_cutoff),
        "largest_component_fraction": float(sizes[0] / max(total, 1)) if sizes.size else 0.0,
        "largest_component_voxels": int(sizes[0]) if sizes.size else 0,
    }


def keep_largest(volume: np.ndarray) -> np.ndarray:
    labs, count = label(np.asarray(volume, dtype=bool), structure=S6)
    if count <= 1:
        return np.asarray(volume, dtype=bool)
    sizes = np.bincount(labs.ravel())
    sizes[0] = 0
    return labs == int(sizes.argmax())


def projection_metrics(
    volume: np.ndarray,
    targets: list[np.ndarray],
    render_args: SimpleNamespace,
    soft_info: dict,
) -> tuple[dict, list[np.ndarray]]:
    renders = _generic_render(volume, render_args, soft_info)
    rows = []
    for target, pred in zip(targets, renders):
        inter = int(np.logical_and(target, pred).sum())
        union = int(np.logical_or(target, pred).sum())
        rows.append(
            {
                "iou": float(inter / max(union, 1)),
                "dice": float(2.0 * inter / max(int(target.sum() + pred.sum()), 1)),
                "target_pixels": int(target.sum()),
                "rendered_pixels": int(pred.sum()),
            }
        )
    return (
        {
            "mean_iou": float(np.mean([row["iou"] for row in rows])),
            "min_iou": float(min(row["iou"] for row in rows)),
            "mean_dice": float(np.mean([row["dice"] for row in rows])),
            "per_view": rows,
        },
        renders,
    )


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def export_and_qa(volume: np.ndarray, path: Path, print_mm: float) -> dict:
    mesh = export_stl_at_print_scale(volume, path, print_mm)
    watertight, mesh_parts, mesh_outer_parts = mesh_qa(volume)
    return {
        "path": str(path),
        "watertight": bool(watertight),
        "mesh_parts": int(mesh_parts),
        "mesh_outer_parts": int(mesh_outer_parts),
        "mesh_internal_parts": int(max(mesh_parts - mesh_outer_parts, 0)),
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "volume_mm3": float(abs(mesh.volume)) if mesh.is_volume else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--core-res", type=int, default=48)
    parser.add_argument("--print-mm", type=float, default=100.0)
    parser.add_argument("--min-iou", type=float, default=0.95)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.soft_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    shadow_files = metrics["args"]["shadow_files"]
    render_args = SimpleNamespace(
        device=args.device,
        silhouettes=shadow_files,
        mr=int(metrics["args"].get("mirror_mode", 0)),
        tsf=int(metrics["args"].get("tsf_mode", 0)),
        camera_mode=metrics["args"].get("camera_mode", "ortho"),
    )
    soft_info = {"metrics": metrics, "artifact_dir": str(args.soft_dir)}
    targets = _generic_targets(args.soft_dir, 256)

    probability = _canonical_xyz_from_trainer_zyx(
        np.load(args.soft_dir / "occupancy_probability_core.npy").astype(np.float32)
    )
    proposal_path = args.soft_dir / "occupancy_soft_proposal.npy"
    proposal = (
        _canonical_xyz_from_trainer_zyx(np.load(proposal_path).astype(bool))
        if proposal_path.is_file()
        else probability >= float(metrics["args"].get("sadekar_density_threshold", 0.05))
    )
    hull = _generic_visual_hull(
        targets,
        render_args,
        256,
        soft_info=soft_info,
        preserve_xyz=proposal,
    )

    direct_projection, direct_renders = projection_metrics(
        proposal, targets, render_args, soft_info
    )
    direct_row = {
        "name": "direct_v256_soft_field",
        "occupied_voxels": int(proposal.sum()),
        "saving_vs_protocol_envelope": float(1.0 - proposal.sum() / max(int(hull.sum()), 1)),
        **component_stats(proposal),
        **direct_projection,
    }
    direct_dir = args.output / "direct_v256"
    direct_dir.mkdir(exist_ok=True)
    np.save(direct_dir / "volume.npy", proposal.astype(np.uint8))
    for index, render in enumerate(direct_renders):
        save_mask(direct_dir / f"projection_view_{index}.png", render)
    direct_row["mesh_qa"] = export_and_qa(
        proposal, direct_dir / "direct_v256_soft_field.stl", args.print_mm
    )

    hull48 = downsample_max_to(hull, args.core_res) >= 0.5
    probability48 = downsample_max_to(probability, args.core_res)
    trial_rows = []
    trial_volumes: dict[str, np.ndarray] = {}
    for threshold in (0.05, 0.10, 0.15, 0.20):
        core48 = probability48 >= threshold
        core48, bridge_voxels = connect6_min_bridges(
            core48, hull48, max_passes=12, bridge_radius=1
        )
        lifted_core = upsample_nearest_to(core48, 256).astype(bool) & hull
        variants: list[tuple[str, np.ndarray, int]] = [
            ("full_proposal", lifted_core | proposal, 0)
        ]
        for shell_thickness in (1, 2, 3):
            eroded = binary_erosion(
                proposal,
                structure=S6,
                iterations=shell_thickness,
                border_value=0,
            )
            projection_shell = proposal & ~eroded
            variants.append(
                (f"shell{shell_thickness}", lifted_core | projection_shell, shell_thickness)
            )

        for variant, delivery, shell_thickness in variants:
            projection, renders = projection_metrics(
                delivery, targets, render_args, soft_info
            )
            name = f"core48_t{threshold:.2f}_{variant}"
            row = {
                "name": name,
                "core_threshold": threshold,
                "shell_thickness": shell_thickness,
                "core48_voxels": int(core48.sum()),
                "core48_bridge_voxels": int(bridge_voxels),
                "occupied_voxels": int(delivery.sum()),
                "saving_vs_protocol_envelope": float(
                    1.0 - delivery.sum() / max(int(hull.sum()), 1)
                ),
                **component_stats(delivery),
                **projection,
            }
            trial_rows.append(row)
            trial_volumes[name] = delivery
            row["_renders"] = renders
            print(
                json.dumps(
                    {
                        "candidate": name,
                        "mean_iou": row["mean_iou"],
                        "min_iou": row["min_iou"],
                        "voxels": row["occupied_voxels"],
                    }
                ),
                flush=True,
            )

    eligible = [row for row in trial_rows if row["mean_iou"] >= args.min_iou]
    selection_pool = eligible or trial_rows
    selected = min(
        selection_pool,
        key=lambda row: (
            -row["mean_iou"] if not eligible else row["occupied_voxels"],
            -row["min_iou"],
        ),
    )
    selected_name = selected["name"]
    selected_volume = trial_volumes[selected_name]
    selected_dir = args.output / "core48_v256_shell"
    selected_dir.mkdir(exist_ok=True)
    np.save(selected_dir / "volume.npy", selected_volume.astype(np.uint8))
    for index, render in enumerate(selected.pop("_renders")):
        save_mask(selected_dir / f"projection_view_{index}.png", render)
    selected["mesh_qa"] = export_and_qa(
        selected_volume,
        selected_dir / "core48_v256_shell.stl",
        args.print_mm,
    )
    for row in trial_rows:
        row.pop("_renders", None)

    report = {
        "protocol": {
            "soft_dir": str(args.soft_dir),
            "camera_protocol": metrics.get("camera_protocol"),
            "views": len(targets),
            "core_resolution": args.core_res,
            "delivery_resolution": 256,
            "print_mm": args.print_mm,
            "minimum_mean_iou": args.min_iou,
            "shared_surface_export": "production SDF, smooth_sigma=0.6",
        },
        "direct_v256": direct_row,
        "core48_v256_shell": selected,
        "core_shell_trials": trial_rows,
    }
    (args.output / "comparison_metrics.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "direct_v256": {
                    "mean_iou": direct_row["mean_iou"],
                    "components_6": direct_row["components_6"],
                    "mesh": direct_row["mesh_qa"],
                },
                "selected_core_shell": {
                    "name": selected_name,
                    "mean_iou": selected["mean_iou"],
                    "min_iou": selected["min_iou"],
                    "mesh": selected["mesh_qa"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
