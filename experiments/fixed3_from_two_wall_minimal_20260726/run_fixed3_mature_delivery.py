#!/usr/bin/env python3
"""Mature fixed-three-view delivery without modifying the two-wall path.

The adapter starts from a frozen exact Sadekar-TFS V256 proposal and applies
the same classes of deterministic operations used by the validated two-wall
delivery: target-ray restoration, local 6-connected bridging, guarded
material removal, exterior cleanup, and one final SDF mesh reconstruction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
for module_path in (ROOT, ROOT / "shadowforge_pipeline", ROOT / "generation"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import unified_run as pipeline
from voxel_ops import connect6_min_bridges

S6 = ndimage.generate_binary_structure(3, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-mm", type=float, default=100.0)
    parser.add_argument("--bridge-radius", type=int, choices=(0, 1), default=0)
    parser.add_argument("--carve-checkpoints", type=int, default=3)
    parser.add_argument("--carve-voxels", type=int, default=4096)
    parser.add_argument("--fidelity-tolerance", type=float, default=0.008)
    return parser.parse_args()


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def largest_component(volume: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(np.asarray(volume, dtype=bool), structure=S6)
    if count <= 1:
        return np.asarray(volume, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def clean_exterior(volume: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """Remove enclosed cavities and merge one-voxel surface contacts."""
    occupied = ndimage.binary_fill_holes(np.asarray(volume, dtype=bool))
    occupied = ndimage.binary_closing(occupied, structure=S6, iterations=1)
    occupied &= np.asarray(envelope, dtype=bool)
    return largest_component(occupied)


def export_oriented_print_mesh(
    display_mesh: trimesh.Trimesh, orientation: str, path: Path
) -> None:
    rotations = {
        "display_z_up": (0.0, (1.0, 0.0, 0.0)),
        "display_z_down": (np.pi, (1.0, 0.0, 0.0)),
        "display_x_up": (-np.pi / 2.0, (0.0, 1.0, 0.0)),
        "display_x_down": (np.pi / 2.0, (0.0, 1.0, 0.0)),
        "display_y_up": (np.pi / 2.0, (1.0, 0.0, 0.0)),
        "display_y_down": (-np.pi / 2.0, (1.0, 0.0, 0.0)),
    }
    angle, axis = rotations[orientation]
    mesh = display_mesh.copy()
    if angle:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation((-center[0], -center[1], -mesh.bounds[0, 2]))
    mesh.export(path)


def fast_metrics(
    volume: np.ndarray,
    targets: list[np.ndarray],
    hull_voxels: int,
    protocol: SimpleNamespace,
    soft_info: dict,
) -> dict:
    return pipeline._generic_prescreen_metrics(
        volume, targets, hull_voxels, protocol, soft_info
    )


def main() -> None:
    cli = parse_args()
    soft_dir = cli.soft_artifact_dir.resolve()
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    soft_metrics = json.loads(
        (soft_dir / "metrics.json").read_text(encoding="utf-8")
    )
    trainer_args = soft_metrics.get("args", {})
    silhouettes = [Path(path) for path in trainer_args.get("shadow_files", [])]
    if len(silhouettes) != 3:
        raise ValueError(f"Expected three base silhouettes, got {len(silhouettes)}")
    if not soft_metrics.get("tsf_direct"):
        raise ValueError("The frozen artifact must use exact direct Sadekar TFS")

    protocol = SimpleNamespace(
        device=cli.device,
        silhouettes=silhouettes,
        mr=int(trainer_args.get("mirror_mode", 1)),
        tsf=int(trainer_args.get("tsf_mode", 1)),
        print_mm=float(cli.print_mm),
    )
    soft_info = {"metrics": soft_metrics}
    proposal = pipeline._canonical_xyz_from_trainer_zyx(
        np.load(soft_dir / "occupancy_soft_proposal.npy").astype(bool)
    )
    probability = pipeline._canonical_xyz_from_trainer_zyx(
        np.load(soft_dir / "occupancy_probability_core.npy").astype(np.float32)
    )
    resolution = int(proposal.shape[0])
    targets = pipeline._generic_targets(soft_dir, resolution)
    envelope = pipeline._generic_visual_hull(
        targets, protocol, resolution, soft_info, preserve_xyz=proposal
    )

    restored = pipeline.restore_declared_view_rays(
        proposal, envelope, probability, targets, protocol, soft_info
    )
    restored, _ = connect6_min_bridges(
        restored,
        envelope,
        max_passes=16,
        bridge_radius=int(cli.bridge_radius),
    )

    trajectory: list[tuple[str, np.ndarray]] = [("restored", restored.copy())]
    current = restored.copy()
    for checkpoint in range(1, int(cli.carve_checkpoints) + 1):
        carved, removed = pipeline.axis_guarded_carve_step(
            current,
            priority=probability,
            num_axes=3,
            max_removals=int(cli.carve_voxels),
        )
        if removed == 0:
            break
        carved = pipeline.restore_declared_view_rays(
            carved, envelope, probability, targets, protocol, soft_info
        )
        carved, _ = connect6_min_bridges(
            carved,
            envelope,
            max_passes=16,
            bridge_radius=int(cli.bridge_radius),
        )
        trajectory.append((f"carve_{checkpoint}", carved.copy()))
        current = carved

    hull_voxels = int(envelope.sum())
    audit = []
    for name, volume in trajectory:
        metrics = fast_metrics(volume, targets, hull_voxels, protocol, soft_info)
        audit.append({"candidate": name, **metrics})
        print(json.dumps({"stage": "fixed3_trajectory", **audit[-1]}), flush=True)

    best_iou = max(row["mean_iou"] for row in audit)
    admissible = [
        (name, volume, row)
        for (name, volume), row in zip(trajectory, audit)
        if row["components"] == 1
        and row["mean_iou"] >= best_iou - float(cli.fidelity_tolerance)
    ]
    if not admissible:
        admissible = [
            (name, volume, row)
            for (name, volume), row in zip(trajectory, audit)
            if row["components"] == 1
        ]
    selected_name, selected, selected_fast = max(
        admissible, key=lambda item: (item[2]["saving"], item[2]["min_iou"])
    )

    cleaned = clean_exterior(selected, envelope)
    cleaned = pipeline.restore_declared_view_rays(
        cleaned, envelope, probability, targets, protocol, soft_info
    )
    cleaned, _ = connect6_min_bridges(
        cleaned, envelope, max_passes=16, bridge_radius=int(cli.bridge_radius)
    )
    cleaned = clean_exterior(cleaned, envelope)
    cleaned_fast = fast_metrics(cleaned, targets, hull_voxels, protocol, soft_info)
    if (
        cleaned_fast["components"] == 1
        and cleaned_fast["mean_iou"] >= best_iou - float(cli.fidelity_tolerance)
    ):
        selected_name += ":exterior_clean"
        selected = cleaned
        selected_fast = cleaned_fast

    qa = pipeline._generic_metrics(
        selected, targets, hull_voxels, protocol, soft_info
    )
    print_volume = qa.pop("print_volume")
    renders = qa.pop("renders")

    np.save(output / "selected_volume_256.npy", selected.astype(np.uint8))
    core48 = pipeline.downsample_max_to(selected, 48) >= 0.5
    np.save(output / "structural_core_48.npy", core48.astype(np.uint8))
    for index, (target, rendered) in enumerate(zip(targets, renders)):
        save_mask(output / f"target_view_{index}.png", target)
        save_mask(output / f"projection_view_{index}.png", rendered)

    display_stl = output / "selected_display_fixed.stl"
    print_stl = output / f"selected_print_{qa['print_orientation']}.stl"
    display_mesh = pipeline.export_stl_at_print_scale(
        selected, display_stl, cli.print_mm
    )
    export_oriented_print_mesh(display_mesh, qa["print_orientation"], print_stl)

    manifest = {
        "experiment": "fixed3_mature_delivery_isolated",
        "soft_artifact_dir": str(soft_dir),
        "silhouettes": [str(path) for path in silhouettes],
        "selected_candidate": selected_name,
        "trajectory": audit,
        "selected_fast": selected_fast,
        "qa": qa,
        "display_stl": str(display_stl),
        "print_stl": str(print_stl),
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
