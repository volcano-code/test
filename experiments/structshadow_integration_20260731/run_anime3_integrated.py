#!/usr/bin/env python3
"""Isolated StructShadow-to-P-CRAFT integration for one complex anime triplet.

The script imports an immutable StructShadow snapshot and preserves three
invariants:

1. calibrated high-resolution geometry remains the delivery owner;
2. the frozen soft field guides target-ray restoration but never owns geometry;
3. C48 is used only for the PINO/FEM structural query.

Both fixed and free three-view calibrations can be audited.  Every candidate is
exported with projection, connectivity, material, and SDF-mesh evidence.  No
mature two-view P-CRAFT source file is imported or modified.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import trimesh
from scipy.ndimage import binary_dilation, generate_binary_structure, label


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = SCRIPT_DIR / "structshadow_snapshot"
for path in (ROOT, ROOT / "experiments", SNAPSHOT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets import load_data_from_list
from voxel_ops import (
    build_tfs_visual_hull,
    calibrate_tfs_projection,
    evaluate_tfs_volume,
    make_watertight_printable,
    minimize_material_printable,
    repair_tfs_projection_gaps,
    resize_binary_volume,
    sdf_watertight_mesh,
)
from shadowforge_pipeline.run import decouple_volume, strict_pino_sf
from structural.analysis.load_generator import LoadGenerator
from structural.models.load_model import load_surrogate


S6 = generate_binary_structure(3, 1)
DEFAULT_INPUT_DIR = (
    ROOT / "results/fig4_anime_curated_original_v1_20260724/inputs"
)
DEFAULT_SILHOUETTES = (
    DEFAULT_INPUT_DIR / "adventurer_schoolgirl_fighter_view_x.png",
    DEFAULT_INPUT_DIR / "adventurer_schoolgirl_fighter_view_y.png",
    DEFAULT_INPUT_DIR / "adventurer_schoolgirl_fighter_view_z.png",
)
DEFAULT_OUTPUT = (
    ROOT / "results/structshadow_integrated_anime3_seed17_saving40_20260731"
)
DEFAULT_PINO = (
    ROOT
    / "checkpoints/taskdisjoint_five_model_comparator_20260718/full_pino/best_model.pt"
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--silhouettes",
        type=Path,
        nargs=3,
        metavar=("VIEW_X", "VIEW_Y", "VIEW_Z"),
        default=DEFAULT_SILHOUETTES,
        help="Exactly three unique silhouettes; mirror views are derived.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--geometry-resolution", type=int, default=128)
    parser.add_argument("--evaluation-resolution", type=int, default=256)
    parser.add_argument("--structural-resolution", type=int, default=48)
    parser.add_argument("--print-mm", type=float, default=210.0)
    parser.add_argument("--smooth-sigma", type=float, default=0.6)
    parser.add_argument("--shell-voxels", type=int, default=2)
    parser.add_argument("--connect-iters", type=int, default=1)
    parser.add_argument("--pino-checkpoint", type=Path, default=DEFAULT_PINO)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--guidance-density",
        type=Path,
        help=(
            "Optional frozen soft probability field used only to rank "
            "target-ray restoration locations."
        ),
    )
    parser.add_argument(
        "--guidance-layout",
        choices=("zyx", "xyz"),
        default="zyx",
        help="Axis layout of --guidance-density; trainer artifacts use zyx.",
    )
    parser.add_argument("--skip-pino", action="store_true")
    parser.add_argument(
        "--calibration",
        choices=("fixed", "free", "both"),
        default="both",
        help="Audit the native fixed frame, searched free frame, or both.",
    )
    parser.add_argument(
        "--calibration-search-resolution",
        type=int,
        default=48,
        help="Low-resolution exhaustive search used only by free calibration.",
    )
    parser.add_argument(
        "--fidelity-band",
        type=float,
        default=0.01,
        help="Connected candidates within this mean-IoU distance compete on material.",
    )
    parser.add_argument(
        "--saving-min",
        type=float,
        default=0.30,
        help="Minimum admissible material saving relative to the calibrated hull.",
    )
    parser.add_argument(
        "--saving-max",
        type=float,
        default=0.50,
        help="Maximum admissible material saving relative to the calibrated hull.",
    )
    parser.add_argument(
        "--saving-target",
        type=float,
        default=0.40,
        help="Preferred saving inside the admissible interval.",
    )
    parser.add_argument(
        "--trajectory-max-steps",
        type=int,
        default=8,
        help="Maximum number of nested material-trajectory states to audit.",
    )
    parser.add_argument(
        "--skip-core-shell-diagnostic",
        action="store_true",
        help="Skip the non-owner C48-to-V256 diagnostic reconstruction.",
    )
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_count(volume: np.ndarray) -> int:
    return int(label(np.asarray(volume, dtype=bool), structure=S6)[1])


def load_masks(paths: list[Path], resolution: int) -> list[np.ndarray]:
    images, _ = load_data_from_list(
        [str(path) for path in paths],
        True,
        (resolution, resolution),
        torch.device("cpu"),
        3,
        False,
    )
    masks = [np.asarray(image).squeeze().astype(np.float32) for image in images]
    if len(masks) != 6:
        raise RuntimeError(f"Expected exactly six TFS masks, received {len(masks)}")
    return masks


def load_guidance_density(path: Path, layout: str) -> np.ndarray:
    """Load a frozen soft field without turning its fragments into geometry."""
    guidance = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if guidance.ndim != 3:
        raise ValueError(
            f"Expected a 3D guidance density in {path}, received {guidance.shape}"
        )
    if layout == "zyx":
        guidance = np.flip(np.transpose(guidance, (2, 1, 0)), axis=(0, 2))
    return np.clip(guidance, 0.0, 1.0)


def scale_mesh(mesh: trimesh.Trimesh, print_mm: float) -> trimesh.Trimesh:
    mesh = mesh.copy()
    max_extent = max(float(mesh.extents.max()), 1e-12)
    mesh.apply_scale(float(print_mm) / max_extent)
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation([-center[0], -center[1], -mesh.bounds[0, 2]])
    return mesh


def mesh_stats(mesh: trimesh.Trimesh) -> dict[str, Any]:
    parts = mesh.split(only_watertight=False)
    return {
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "body_count": int(len(parts)),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "volume": float(abs(mesh.volume)) if mesh.is_volume else None,
        "surface_area": float(mesh.area),
        "extents": [float(value) for value in mesh.extents],
    }


def export_mesh(
    volume: np.ndarray,
    path: Path,
    print_mm: float,
    smooth_sigma: float,
) -> dict[str, Any]:
    mesh = sdf_watertight_mesh(np.asarray(volume, dtype=bool), smooth_sigma=smooth_sigma)
    mesh = scale_mesh(mesh, print_mm)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return {"path": str(path), **mesh_stats(mesh)}


def export_dominant_external_shell(
    source_path: Path,
    output_path: Path,
    minimum_fraction: float = 0.999,
) -> dict[str, Any]:
    mesh = trimesh.load_mesh(source_path, force="mesh", process=True)
    mesh.merge_vertices()
    parts = list(mesh.split(only_watertight=False))
    parts.sort(
        key=lambda part: abs(float(part.volume)) if part.is_volume else float(part.area),
        reverse=True,
    )
    if not parts:
        raise RuntimeError(f"No connected mesh bodies found in {source_path}")

    total_volume = sum(abs(float(part.volume)) for part in parts if part.is_volume)
    largest_volume = abs(float(parts[0].volume)) if parts[0].is_volume else 0.0
    largest_fraction = largest_volume / max(total_volume, 1e-12)
    if len(parts) > 1 and largest_fraction < minimum_fraction:
        raise RuntimeError(
            "Refusing external-shell cleanup because the dominant body accounts for "
            f"only {largest_fraction:.6f} of the closed-mesh volume"
        )

    external = parts[0].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    external.export(output_path)
    return {
        "path": str(output_path),
        "source_path": str(source_path),
        "source_body_count": int(len(parts)),
        "dominant_volume_fraction": float(largest_fraction),
        "removed_body_count": int(max(len(parts) - 1, 0)),
        "removed_volume_fraction": float(max(1.0 - largest_fraction, 0.0)),
        **mesh_stats(external),
    }


def reference_mesh_stats(source: Path) -> dict[str, Any]:
    rows = {}
    for path in sorted(source.glob("*.stl")):
        mesh = trimesh.load_mesh(path, force="mesh", process=True)
        rows[path.name] = {"path": str(path), **mesh_stats(mesh)}
    return rows


def save_binary(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255)


def save_projection_evidence(
    name: str,
    volume: np.ndarray,
    masks: list[np.ndarray],
    calibration: dict[str, Any],
    output: Path,
    image_size: int,
) -> dict[str, Any]:
    evidence_dir = output / "projections" / name
    metrics = evaluate_tfs_volume(
        np.asarray(volume, dtype=np.float32),
        masks,
        calibration,
        image_size=image_size,
    )
    tiles = []
    for index, (target, prediction) in enumerate(
        zip(metrics["gt_masks"], metrics["pred_masks"])
    ):
        save_binary(evidence_dir / f"view_{index}_target.png", target)
        save_binary(evidence_dir / f"view_{index}_prediction.png", prediction)
        target_u8 = np.asarray(target, dtype=np.uint8) * 255
        pred_u8 = np.asarray(prediction, dtype=np.uint8) * 255
        separator = np.full((target_u8.shape[0], 8), 127, dtype=np.uint8)
        tiles.append(np.concatenate([target_u8, separator, pred_u8], axis=1))
    row_separator = np.full((8, tiles[0].shape[1]), 127, dtype=np.uint8)
    montage_parts = []
    for index, tile in enumerate(tiles):
        if index:
            montage_parts.append(row_separator)
        montage_parts.append(tile)
    montage = np.concatenate(montage_parts, axis=0)
    cv2.imwrite(str(evidence_dir / "target_vs_prediction.png"), montage)
    return {
        key: json_ready(value)
        for key, value in metrics.items()
        if key not in {"pred_masks", "gt_masks"}
    }


def evaluate_candidate(
    name: str,
    volume: np.ndarray,
    hull_voxels: int,
    masks: list[np.ndarray],
    calibration: dict[str, Any],
    output: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    volume = np.asarray(volume, dtype=bool)
    candidate_dir = output / "candidates" / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    np.save(candidate_dir / "volume.npy", volume.astype(np.uint8))
    projection = save_projection_evidence(
        name,
        volume,
        masks,
        calibration,
        output,
        args.evaluation_resolution,
    )
    mesh = export_mesh(
        volume,
        candidate_dir / f"{name}.stl",
        args.print_mm,
        args.smooth_sigma,
    )
    occupied = int(volume.sum())
    return {
        "name": name,
        "resolution": int(volume.shape[0]),
        "occupied_voxels": occupied,
        "component_count_6": component_count(volume),
        "saving_vs_hull": float(1.0 - occupied / max(int(hull_voxels), 1)),
        "projection": projection,
        "mesh": mesh,
    }


def run_pino(
    volume48: np.ndarray,
    checkpoint: Path,
    device_name: str,
) -> dict[str, Any]:
    if not checkpoint.is_file():
        return {"status": "SKIPPED", "reason": f"missing checkpoint: {checkpoint}"}
    device = torch.device(device_name)
    model, checkpoint_payload = load_surrogate(str(checkpoint), device)
    config = checkpoint_payload.get("config", {})
    if int(config.get("in_channels", 0)) != 6:
        return {
            "status": "SKIPPED",
            "reason": "strict structural query requires a six-channel checkpoint",
            "checkpoint_in_channels": int(config.get("in_channels", 0)),
        }
    load = LoadGenerator().standing_load(np.asarray(volume48, dtype=np.float32))
    started = time.perf_counter()
    metrics = strict_pino_sf(
        model,
        np.asarray(volume48, dtype=np.float32),
        np.asarray(load["force"], dtype=np.float32),
        np.asarray(load["fixed_mask"], dtype=np.float32),
        device,
    )
    elapsed = time.perf_counter() - started
    return {
        "status": "OK",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_version": checkpoint_payload.get("version"),
        "device": str(device),
        "elapsed_s": float(elapsed),
        **json_ready(metrics),
    }


def build_controlled_material_trajectory(
    anchor: np.ndarray,
    owner: np.ndarray,
    hull_voxels: int,
    saving_min: float,
    max_steps: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build nested, connected states between a watertight anchor and owner.

    The high-fidelity watertight anchor is expanded by face-connected dilation
    inside the repaired owner domain. The states remain strict supersets of the
    anchor, so lowering the saving target cannot silently switch to the lower-
    fidelity raw minimal backbone. SDF export supplies the watertight mesh and
    therefore does not require a second, projection-changing thickening pass.
    """
    current = np.asarray(anchor, dtype=bool).copy()
    domain = np.asarray(owner, dtype=bool) | current
    candidates: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    anchor_voxels = int(current.sum())

    for step in range(1, max(int(max_steps), 0) + 1):
        expanded = current | (binary_dilation(current, structure=S6) & domain)
        if np.array_equal(expanded, current):
            break
        current = expanded
        saving = float(
            1.0 - int(current.sum()) / max(hull_voxels, 1)
        )
        name = f"trajectory_{step:02d}_watertight"
        candidates[name] = current.copy()
        metadata[name] = {
            "step": int(step),
            "anchor_voxels": anchor_voxels,
            "occupied_voxels": int(current.sum()),
            "saving_vs_hull": saving,
            "nested_superset_of_anchor": True,
            "mesh_export": "SDF",
        }
        if saving < float(saving_min):
            break

    return candidates, metadata


def branch_geometry(
    branch_name: str,
    masks: list[np.ndarray],
    calibration: dict[str, Any],
    args: argparse.Namespace,
    output: Path,
    use_repair: bool,
    guidance_density: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    resolution = args.geometry_resolution
    hull, _ = build_tfs_visual_hull(masks, resolution, calibration=calibration)
    candidates: dict[str, np.ndarray] = {"hull": hull}
    branch_meta: dict[str, Any] = {}

    owner = hull
    if use_repair:
        repaired, repair_info = repair_tfs_projection_gaps(
            masks,
            resolution,
            calibration=calibration,
            guidance_density=guidance_density,
        )
        candidates["repaired"] = repaired
        branch_meta["repair"] = json_ready(repair_info)
        owner = repaired

    minimal, minimal_info = minimize_material_printable(owner, bridge_radius=1)
    candidates["minimal"] = minimal
    branch_meta["minimal"] = json_ready(minimal_info)

    hull_watertight, hull_watertight_info = make_watertight_printable(hull, hull)
    minimal_watertight, minimal_watertight_info = make_watertight_printable(minimal, owner)
    candidates["hull_watertight"] = hull_watertight
    candidates["minimal_watertight"] = minimal_watertight
    branch_meta["hull_watertight"] = json_ready(hull_watertight_info)
    branch_meta["minimal_watertight"] = json_ready(minimal_watertight_info)

    hull_voxels = int(hull.sum())
    trajectory, trajectory_info = build_controlled_material_trajectory(
        minimal_watertight,
        owner,
        hull_voxels,
        args.saving_min,
        args.trajectory_max_steps,
    )
    candidates.update(trajectory)
    branch_meta["controlled_material_trajectory"] = trajectory_info

    evaluated = {}
    for candidate_name, volume in candidates.items():
        full_name = f"{branch_name}_{candidate_name}"
        evaluated[candidate_name] = evaluate_candidate(
            full_name,
            volume,
            hull_voxels,
            masks,
            calibration,
            output,
            args,
        )
        gc.collect()

    return {
        "branch": branch_name,
        "calibration": json_ready(calibration),
        "hull_voxels": hull_voxels,
        "metadata": branch_meta,
        "candidates": evaluated,
    }, candidates


def select_delivery_owner(
    branches: dict[str, dict[str, Any]],
    branch_volumes: dict[str, dict[str, np.ndarray]],
    fidelity_band: float,
    saving_min: float,
    saving_max: float,
    saving_target: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for calibration_name, branch in branches.items():
        for candidate_name, metrics in branch["candidates"].items():
            mesh = metrics["mesh"]
            projection = metrics["projection"]
            record = {
                "calibration": calibration_name,
                "candidate": candidate_name,
                "connected": int(metrics["component_count_6"]) == 1,
                "mesh_valid": bool(mesh["watertight"])
                and bool(mesh["winding_consistent"]),
                "mean_iou": float(projection["mean_iou"]),
                "min_iou": float(projection["min_iou"]),
                "mean_dice": float(projection["mean_dice"]),
                "occupied_voxels": int(metrics["occupied_voxels"]),
                "saving_vs_hull": float(metrics["saving_vs_hull"]),
            }
            record["geometry_eligible"] = bool(
                record["connected"] and record["mesh_valid"]
            )
            record["saving_in_band"] = bool(
                float(saving_min)
                <= record["saving_vs_hull"]
                <= float(saving_max)
            )
            record["saving_distance_to_target"] = float(
                abs(record["saving_vs_hull"] - float(saving_target))
            )
            records.append(record)

    geometry_eligible = [record for record in records if record["geometry_eligible"]]
    best_geometry_iou = (
        max(record["mean_iou"] for record in geometry_eligible)
        if geometry_eligible
        else None
    )
    for record in records:
        record["fidelity_eligible"] = bool(
            best_geometry_iou is not None
            and record["mean_iou"]
            >= best_geometry_iou - max(float(fidelity_band), 0.0)
        )
        record["eligible"] = bool(
            record["geometry_eligible"]
            and record["saving_in_band"]
            and record["fidelity_eligible"]
        )

    eligible = [record for record in records if record["eligible"]]
    if eligible:
        best_iou = max(record["mean_iou"] for record in eligible)
        contenders = [
            record
            for record in eligible
            if record["mean_iou"] >= best_iou - max(float(fidelity_band), 0.0)
        ]
        selected = min(
            contenders,
            key=lambda record: (
                record["saving_distance_to_target"],
                -record["min_iou"],
                -record["mean_iou"],
                record["occupied_voxels"],
            ),
        )
        selected = {
            **selected,
            "selection_status": "ELIGIBLE",
            "best_eligible_mean_iou": float(best_iou),
            "fidelity_band": float(fidelity_band),
            "saving_gate": {
                "min": float(saving_min),
                "max": float(saving_max),
                "target": float(saving_target),
            },
        }
    else:
        if geometry_eligible:
            best_iou = max(record["mean_iou"] for record in geometry_eligible)
            contenders = [
                record
                for record in geometry_eligible
                if record["mean_iou"]
                >= best_iou - max(float(fidelity_band), 0.0)
            ]
            selected = min(
                contenders,
                key=lambda record: (
                    record["saving_distance_to_target"],
                    -record["min_iou"],
                    -record["mean_iou"],
                ),
            )
            status = "BEST_EFFORT_OUTSIDE_SAVING_GATE"
        else:
            selected = max(
                records,
                key=lambda record: (
                    int(record["connected"]),
                    int(record["mesh_valid"]),
                    record["mean_iou"],
                    record["min_iou"],
                    -record["occupied_voxels"],
                ),
            )
            best_iou = None
            status = "BEST_EFFORT_NO_GEOMETRY_ELIGIBLE_CANDIDATE"
        selected = {
            **selected,
            "selection_status": status,
            "best_eligible_mean_iou": (
                float(best_iou) if best_iou is not None else None
            ),
            "fidelity_band": float(fidelity_band),
            "saving_gate": {
                "min": float(saving_min),
                "max": float(saving_max),
                "target": float(saving_target),
            },
        }

    selected["volume"] = branch_volumes[selected["calibration"]][
        selected["candidate"]
    ]
    return selected, records


def run_core_shell_diagnostic(
    owner: np.ndarray,
    masks: list[np.ndarray],
    calibration: dict[str, Any],
    args: argparse.Namespace,
    output: Path,
) -> dict[str, Any]:
    core48 = resize_binary_volume(owner, args.structural_resolution)
    high_res_hull, _ = build_tfs_visual_hull(
        masks,
        args.evaluation_resolution,
        calibration=calibration,
    )
    delivered = decouple_volume(
        core48,
        high_res_hull,
        args.evaluation_resolution,
        args.shell_voxels,
        connect_iters=args.connect_iters,
        fill_holes=False,
    )
    diagnostic_dir = output / "fusion"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    np.save(diagnostic_dir / "owner_core48.npy", core48.astype(np.uint8))
    np.save(diagnostic_dir / "core_shell_reconstructed256.npy", delivered.astype(np.uint8))
    metrics = evaluate_candidate(
        "core_shell_reconstructed",
        delivered,
        int(high_res_hull.sum()),
        masks,
        calibration,
        output,
        args,
    )
    return {
        "core48_path": str(diagnostic_dir / "owner_core48.npy"),
        "core48_occupied_voxels": int(core48.sum()),
        "delivery": metrics,
    }


def main() -> None:
    args = parse_args()
    if not (
        0.0
        <= float(args.saving_min)
        <= float(args.saving_target)
        <= float(args.saving_max)
        < 1.0
    ):
        raise ValueError(
            "Expected 0 <= saving-min <= saving-target <= saving-max < 1."
        )
    if int(args.trajectory_max_steps) < 1:
        raise ValueError("--trajectory-max-steps must be at least 1.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty experiment folder: {args.output}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    mask_paths = [Path(path).resolve() for path in args.silhouettes]
    missing = [str(path) for path in mask_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing silhouette inputs: {missing}")

    started = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    masks = load_masks(mask_paths, args.geometry_resolution)
    guidance_density = None
    if args.guidance_density is not None:
        if not args.guidance_density.is_file():
            raise FileNotFoundError(
                f"Missing guidance density: {args.guidance_density}"
            )
        guidance_density = load_guidance_density(
            args.guidance_density,
            args.guidance_layout,
        )

    results: dict[str, Any] = {
        "schema": "structshadow_integrated_anime3_v1",
        "output": str(args.output),
        "seed": int(args.seed),
        "inputs": [str(path) for path in mask_paths],
        "protocol": {
            "unique_views": 3,
            "evaluated_masks": 6,
            "mirror_mode": True,
            "geometry_resolution": int(args.geometry_resolution),
            "evaluation_resolution": int(args.evaluation_resolution),
            "structural_resolution": int(args.structural_resolution),
            "smooth_sigma": float(args.smooth_sigma),
            "print_mm": float(args.print_mm),
            "guidance_density": (
                str(args.guidance_density)
                if args.guidance_density is not None
                else None
            ),
            "guidance_layout": (
                args.guidance_layout if args.guidance_density is not None else None
            ),
            "calibration": args.calibration,
            "calibration_search_resolution": int(
                args.calibration_search_resolution
            ),
            "fidelity_band": float(args.fidelity_band),
            "saving_gate": {
                "min": float(args.saving_min),
                "max": float(args.saving_max),
                "target": float(args.saving_target),
            },
            "trajectory_max_steps": int(args.trajectory_max_steps),
            "geometry_owner": "calibrated high-resolution connected volume",
            "structural_proxy": "C48 occupancy for PINO/FEM query only",
        },
        "provenance": {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "structshadow_snapshot": {
                path.name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for path in sorted(SNAPSHOT_DIR.glob("*.py"))
            },
            "inputs": {
                path.name: {"path": str(path), "sha256": sha256_file(path)}
                for path in mask_paths
            },
            "guidance_density": (
                {
                    "path": str(args.guidance_density.resolve()),
                    "sha256": sha256_file(args.guidance_density.resolve()),
                }
                if args.guidance_density is not None
                else None
            ),
        },
    }

    calibration_modes = (
        ("fixed", "free") if args.calibration == "both" else (args.calibration,)
    )
    calibrations: dict[str, dict[str, Any]] = {}
    branches: dict[str, dict[str, Any]] = {}
    branch_volumes: dict[str, dict[str, np.ndarray]] = {}
    for mode_index, mode in enumerate(calibration_modes, start=1):
        print(
            f"[{mode_index}/{len(calibration_modes)}] "
            f"Calibrating and building {mode} high-resolution geometry",
            flush=True,
        )
        calibration = calibrate_tfs_projection(
            masks,
            search_res=min(
                args.calibration_search_resolution,
                args.geometry_resolution,
            ),
            fix_axis=mode == "fixed",
        )
        calibrations[mode] = calibration
        branch, volumes = branch_geometry(
            f"pcraft_{mode}",
            masks,
            calibration,
            args,
            args.output,
            use_repair=True,
            guidance_density=guidance_density if mode == "fixed" else None,
        )
        branches[mode] = branch
        branch_volumes[mode] = volumes
        results[f"pcraft_{mode}"] = branch
        (args.output / "results.partial.json").write_text(
            json.dumps(json_ready(results), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    selected, candidate_audit = select_delivery_owner(
        branches,
        branch_volumes,
        args.fidelity_band,
        args.saving_min,
        args.saving_max,
        args.saving_target,
    )
    owner = np.asarray(selected.pop("volume"), dtype=bool)
    selected_calibration = calibrations[selected["calibration"]]
    selected_branch = branches[selected["calibration"]]
    core48 = resize_binary_volume(owner, args.structural_resolution)
    fusion_dir = args.output / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    np.save(fusion_dir / "owner_preserved_volume.npy", owner.astype(np.uint8))
    np.save(fusion_dir / "owner_preserved_core48.npy", core48.astype(np.uint8))
    owner_metrics = evaluate_candidate(
        "owner_preserved_delivery",
        owner,
        int(selected_branch["hull_voxels"]),
        masks,
        selected_calibration,
        args.output,
        args,
    )
    owner_mesh_path = Path(owner_metrics["mesh"]["path"])
    recommended_stl = owner_mesh_path
    try:
        owner_external_shell = export_dominant_external_shell(
            owner_mesh_path,
            owner_mesh_path.with_name(
                f"{owner_mesh_path.stem}_single_external_shell.stl"
            ),
        )
        owner_metrics["single_external_shell"] = owner_external_shell
        recommended_stl = Path(owner_external_shell["path"])
    except RuntimeError as error:
        owner_metrics["single_external_shell"] = {
            "status": "SKIPPED",
            "reason": str(error),
            "fallback_path": str(owner_mesh_path),
        }
    results["pcraft_fusion"] = {
        "recommended_geometry_owner": "owner_preserved_delivery",
        "selection": selected,
        "candidate_audit": candidate_audit,
        "owner_preserved": owner_metrics,
        "core48_path": str(fusion_dir / "owner_preserved_core48.npy"),
        "core48_occupied_voxels": int(core48.sum()),
        "interpretation": (
            "The high-resolution projection geometry remains the delivery owner; "
            "C48 is a structural proxy only."
        ),
    }

    print("[structural] Running C48 structural proxy query", flush=True)
    if args.skip_pino:
        results["pcraft_fusion"]["pino"] = {"status": "SKIPPED", "reason": "--skip-pino"}
    else:
        results["pcraft_fusion"]["pino"] = run_pino(
            core48,
            args.pino_checkpoint,
            args.device,
        )

    if not args.skip_core_shell_diagnostic:
        print(
            "[diagnostic] Running non-owner C48-to-V256 reconstruction",
            flush=True,
        )
        results["pcraft_fusion"]["core_shell_diagnostic"] = run_core_shell_diagnostic(
            owner,
            masks,
            selected_calibration,
            args,
            args.output,
        )
    else:
        results["pcraft_fusion"]["core_shell_diagnostic"] = {
            "status": "SKIPPED",
            "reason": "--skip-core-shell-diagnostic",
        }

    results["elapsed_s"] = float(time.time() - started)
    results_path = args.output / "results.json"
    results_path.write_text(
        json.dumps(json_ready(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "results": str(results_path),
        "recommended_stl": str(recommended_stl),
        "raw_delivery_stl": owner_metrics["mesh"]["path"],
        "recommended_core48": str(fusion_dir / "owner_preserved_core48.npy"),
        "selected_calibration": selected["calibration"],
        "selected_candidate": selected["candidate"],
        "selection_status": selected["selection_status"],
        "elapsed_s": results["elapsed_s"],
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
