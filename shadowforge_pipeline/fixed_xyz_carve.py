#!/usr/bin/env python3
"""Isolated fixed-axis carved-cube backend for P-CRAFT.

This backend preserves the three orthogonal projection constraints directly at
the delivery resolution.  It is intentionally separate from the standard
48-core/256-shell and structural-screening path, and is enabled only through
``--pipeline-branch fixed_xyz_carve``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, generate_binary_structure

from candidate_selection_batch import make_candidates, n6
from datasets import load_data_from_list
from export_high_quality_mesh import export_marching_cubes
from voxel_ops import evaluate_tfs_volume, sdf_watertight_mesh


S6 = generate_binary_structure(3, 1)


def _load_fixed_masks(paths: list[Path], resolution: int) -> list[np.ndarray]:
    if len(paths) != 3:
        raise ValueError("fixed_xyz_carve requires exactly three orthogonal silhouettes")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing silhouettes: {missing}")
    tensors, _ = load_data_from_list(
        [str(path) for path in paths],
        True,
        (resolution, resolution),
        torch.device("cpu"),
        3,
        False,
    )
    return [
        np.asarray(item).squeeze().astype(np.float32)
        for item in tensors[:3]
    ]


def _erosion_survival(volume: np.ndarray) -> float:
    volume = np.asarray(volume, dtype=bool)
    return float(
        binary_erosion(volume, structure=S6).sum()
        / max(int(volume.sum()), 1)
    )


def _save_mask(mask: np.ndarray, path: Path, size: int = 512) -> None:
    image = Image.fromarray(
        (np.asarray(mask) > 0.5).astype(np.uint8) * 255,
        mode="L",
    )
    image.resize((size, size), Image.Resampling.NEAREST).save(path)


def _save_projection_sheet(
    inputs: list[np.ndarray],
    predictions: list[np.ndarray],
    view_ious: list[float],
    path: Path,
) -> None:
    cell = 220
    gap = 22
    label_height = 28
    width = 3 * cell + 4 * gap
    height = 2 * (cell + label_height) + 3 * gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index in range(3):
        x = gap + index * (cell + gap)
        input_y = gap + label_height
        pred_y = input_y + cell + gap + label_height
        input_image = Image.fromarray(
            (np.asarray(inputs[index]) > 0.5).astype(np.uint8) * 255,
            mode="L",
        ).resize((cell, cell), Image.Resampling.NEAREST)
        pred_image = Image.fromarray(
            (np.asarray(predictions[index]) > 0.5).astype(np.uint8) * 255,
            mode="L",
        ).resize((cell, cell), Image.Resampling.NEAREST)
        canvas.paste(input_image.convert("RGB"), (x, input_y))
        canvas.paste(pred_image.convert("RGB"), (x, pred_y))
        draw.text((x, gap), f"Target {index + 1}", fill="black")
        draw.text(
            (x, input_y + cell + gap),
            f"Projection {index + 1}  IoU={view_ious[index]:.4f}",
            fill="black",
        )
        draw.rectangle(
            (x, input_y, x + cell - 1, input_y + cell - 1),
            outline=(120, 120, 120),
        )
        draw.rectangle(
            (x, pred_y, x + cell - 1, pred_y + cell - 1),
            outline=(120, 120, 120),
        )
    canvas.save(path)


def _mesh_watertight(volume: np.ndarray, smooth_sigma: float) -> bool:
    try:
        mesh = sdf_watertight_mesh(volume, smooth_sigma=smooth_sigma)
        return bool(mesh.is_watertight and mesh.is_winding_consistent)
    except Exception:
        return False


def _candidate_row(
    name: str,
    volume: np.ndarray,
    masks6: list[np.ndarray],
    calibration: dict[str, Any],
    hull_voxels: int,
    resolution: int,
    smooth_sigma: float,
) -> dict[str, Any]:
    evaluation = evaluate_tfs_volume(
        volume.astype(np.float32),
        masks6,
        calibration,
        image_size=512,
    )
    return {
        "candidate": name,
        "mean_iou": float(evaluation["mean_iou"]),
        "mean_dice": float(evaluation["mean_dice"]),
        "min_iou": float(evaluation["min_iou"]),
        "view_ious": [float(value) for value in evaluation["view_ious"]],
        "view_dices": [float(value) for value in evaluation["view_dices"]],
        "occupied_voxels": int(volume.sum()),
        "material_saved_vs_hull": float(
            1.0 - int(volume.sum()) / max(int(hull_voxels), 1)
        ),
        "component_count_6": int(n6(volume)),
        "watertight": _mesh_watertight(volume, smooth_sigma),
        "erosion_survival": _erosion_survival(volume),
        "_volume": volume,
        "_evaluation": evaluation,
    }


def _passes_gate(row: dict[str, Any], args: Any) -> bool:
    return bool(
        row["mean_iou"] >= args.min_iou
        and row.get("min_iou", row["mean_iou"])
        >= getattr(args, "fixed_xyz_min_view_iou", 0.95)
        and row["component_count_6"] == 1
        and row["watertight"]
        and row["erosion_survival"] >= args.min_erosion_survival
        and args.min_saving
        <= row["material_saved_vs_hull"]
        <= args.max_saving
    )


def select_fixed_xyz_candidate(
    rows: list[dict[str, Any]],
    requested: str,
    saving_target: float,
    args: Any,
) -> tuple[dict[str, Any], bool]:
    """Select one carved-cube candidate without invoking PINO or FEM."""
    if not rows:
        raise RuntimeError("fixed_xyz_carve produced no candidates")
    if requested != "auto":
        matches = [row for row in rows if row["candidate"] == requested]
        if not matches:
            available = ", ".join(sorted(row["candidate"] for row in rows))
            raise ValueError(
                f"Unknown fixed-XYZ candidate {requested!r}; available: {available}"
            )
        chosen = matches[0]
        return chosen, _passes_gate(chosen, args)

    gated = [row for row in rows if _passes_gate(row, args)]
    target = 0.50 if saving_target < 0.0 else saving_target
    if gated:
        chosen = min(
            gated,
            key=lambda row: (
                abs(row["material_saved_vs_hull"] - target),
                -row["mean_iou"],
                -row["erosion_survival"],
            ),
        )
    else:
        chosen = max(
            rows,
            key=lambda row: (
                row["mean_iou"],
                row["component_count_6"] == 1,
                row["watertight"],
                row["erosion_survival"],
            ),
        )
    return chosen, bool(gated)


def _export_print_mesh(
    volume: np.ndarray,
    path: Path,
    print_mm: float,
    smooth_sigma: float,
    taubin_iterations: int,
    upsample_factor: int,
) -> trimesh.Trimesh:
    export_marching_cubes(
        volume,
        path,
        smooth_sigma=smooth_sigma,
        taubin_iterations=taubin_iterations,
        upsample_factor=upsample_factor,
    )
    mesh = trimesh.load(str(path), force="mesh")
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    parts = mesh.split(only_watertight=False)
    if parts:
        mesh = max(parts, key=lambda part: len(part.faces))
    trimesh.repair.fill_holes(mesh)
    if mesh.volume < 0:
        mesh.invert()
    trimesh.repair.fix_normals(mesh)
    mesh.apply_scale(
        float(print_mm) / max(float(np.asarray(mesh.extents).max()), 1e-9)
    )
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation(
        [-center[0], -center[1], -float(mesh.bounds[0, 2])]
    )
    mesh.export(str(path))
    return mesh


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("_")
    }


def run_fixed_xyz_carve(args: Any) -> dict[str, Any]:
    """Run the isolated high-resolution fixed-XYZ carved-cube branch."""
    started = time.time()
    resolution = int(args.fixed_xyz_resolution)
    if resolution <= 0:
        raise ValueError("fixed_xyz_resolution must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    input_dir = args.output / "inputs"
    projection_dir = args.output / "projections"
    candidate_dir = args.output / "candidates"
    input_dir.mkdir(exist_ok=True)
    projection_dir.mkdir(exist_ok=True)
    if args.fixed_xyz_save_all_candidates:
        candidate_dir.mkdir(exist_ok=True)

    masks3 = _load_fixed_masks(args.silhouettes, resolution)
    candidates, auxiliary = make_candidates(
        masks3,
        resolution,
        True,
        print_mm=args.print_mm,
        nozzle_mm=args.nozzle_mm,
        wall_safety=args.wall_safety,
    )
    masks6 = masks3 + [np.fliplr(mask) for mask in masks3]
    hull = np.asarray(auxiliary["hull"], dtype=bool)
    calibration = auxiliary["calibration"]

    rows = []
    for name, candidate in candidates.items():
        volume = np.asarray(candidate, dtype=bool)
        row = _candidate_row(
            name,
            volume,
            masks6,
            calibration,
            int(hull.sum()),
            resolution,
            float(args.fixed_xyz_smooth_sigma),
        )
        rows.append(row)
        if args.fixed_xyz_save_all_candidates:
            np.save(
                candidate_dir / f"{name}.npy",
                volume.astype(np.uint8),
            )

    chosen, accepted = select_fixed_xyz_candidate(
        rows,
        args.fixed_xyz_candidate,
        args.selection_saving_target,
        args,
    )
    selected_volume = np.asarray(chosen["_volume"], dtype=bool)
    selected_volume_path = args.output / "selected_volume.npy"
    selected_stl_path = args.output / "selected_display.stl"
    projection_sheet_path = args.output / "selected_projection_audit.png"
    np.save(selected_volume_path, selected_volume.astype(np.uint8))

    evaluation = chosen["_evaluation"]
    for index, mask in enumerate(masks3):
        _save_mask(mask, input_dir / f"target_view_{index}.png")
    for index, mask in enumerate(evaluation["pred_masks"][:3]):
        _save_mask(mask, projection_dir / f"rendered_view_{index}.png")
    _save_projection_sheet(
        masks3,
        evaluation["pred_masks"][:3],
        chosen["view_ious"][:3],
        projection_sheet_path,
    )

    mesh = _export_print_mesh(
        selected_volume,
        selected_stl_path,
        float(args.print_mm),
        float(args.fixed_xyz_smooth_sigma),
        int(args.fixed_xyz_taubin_iterations),
        int(args.fixed_xyz_upsample_factor),
    )
    selected_public = _public_row(chosen)
    selected_public.update({
        "artifact_volume": str(selected_volume_path),
        "artifact_stl": str(selected_stl_path),
        "artifact_projection_audit": str(projection_sheet_path),
        "mesh_watertight": bool(
            mesh.is_watertight and mesh.is_winding_consistent
        ),
        "mesh_parts": int(len(mesh.split(only_watertight=False))),
        "max_dimension_mm": float(mesh.extents.max()),
    })
    manifest = {
        "schema": "pcraft_fixed_xyz_carve_v1",
        "status": "PASS" if accepted else "BEST_EFFORT",
        "pipeline_branch": "fixed_xyz_carve",
        "default_pipeline_unchanged": True,
        "protocol": {
            "view_count": 3,
            "fixed_orthogonal_axes": True,
            "resolution": resolution,
            "candidate_family": (
                "visual hull + target-ray repair + connectivity-preserving "
                "carving + printable thickening + SDF mesh"
            ),
        },
        "inputs": [str(path) for path in args.silhouettes],
        "selection": {
            "requested_candidate": args.fixed_xyz_candidate,
            "saving_target": (
                0.50
                if args.selection_saving_target < 0
                else float(args.selection_saving_target)
            ),
            "manufacturing_gate_pass": accepted,
        },
        "selected": selected_public,
        "all_candidates": [_public_row(row) for row in rows],
        "structural_screen": {
            "status": "NOT_RUN",
            "reason": (
                "fixed_xyz_carve is a visual-geometry branch; run the standard "
                "P-CRAFT branch when PINO/FEM screening is required"
            ),
        },
        "runtime_seconds": float(time.time() - started),
    }
    manifest_path = args.output / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest
