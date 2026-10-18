#!/usr/bin/env python3
"""Isolated three-view delivery adapter for a frozen Sadekar-TFS soft proposal.

This experiment deliberately leaves the production two-wall implementation
untouched.  It preserves the validated V256 proposal surface, adds only the
shortest six-connected bridges, derives a C48 structural proxy, evaluates the
exact frozen camera protocol, and exports the result through the production SDF
mesh path.
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
from scipy.ndimage import generate_binary_structure, label

ROOT = Path(__file__).resolve().parents[2]
for module_path in (ROOT, ROOT / "shadowforge_pipeline", ROOT / "generation"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import unified_run as pipeline
from voxel_ops import connect6_min_bridges

S6 = generate_binary_structure(3, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-mm", type=float, default=100.0)
    parser.add_argument("--bridge-radius", type=int, choices=(0, 1), default=0)
    return parser.parse_args()


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def export_oriented_print_mesh(
    display_mesh: trimesh.Trimesh,
    orientation: str,
    path: Path,
) -> None:
    """Rigidly orient the validated display mesh without rebuilding its SDF."""
    rotations = {
        "display_z_up": (0.0, (1.0, 0.0, 0.0)),
        "display_z_down": (np.pi, (1.0, 0.0, 0.0)),
        "display_x_up": (-np.pi / 2.0, (0.0, 1.0, 0.0)),
        "display_x_down": (np.pi / 2.0, (0.0, 1.0, 0.0)),
        "display_y_up": (np.pi / 2.0, (1.0, 0.0, 0.0)),
        "display_y_down": (-np.pi / 2.0, (1.0, 0.0, 0.0)),
    }
    if orientation not in rotations:
        raise ValueError(f"Unsupported print orientation: {orientation}")
    angle, axis = rotations[orientation]
    mesh = display_mesh.copy()
    if angle:
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(angle, axis)
        )
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation((-center[0], -center[1], -mesh.bounds[0, 2]))
    mesh.export(path)


def main() -> None:
    cli = parse_args()
    soft_dir = cli.soft_artifact_dir.resolve()
    output = cli.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metrics_path = soft_dir / "metrics.json"
    proposal_path = soft_dir / "occupancy_soft_proposal.npy"
    if not metrics_path.is_file() or not proposal_path.is_file():
        raise FileNotFoundError(
            "The frozen artifact must contain metrics.json and "
            "occupancy_soft_proposal.npy"
        )

    soft_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    trainer_args = soft_metrics.get("args", {})
    silhouettes = [Path(path) for path in trainer_args.get("shadow_files", [])]
    if len(silhouettes) != 3:
        raise ValueError(
            f"This isolated adapter requires exactly three base views, got {len(silhouettes)}"
        )
    if not (
        soft_metrics.get("tsf_direct")
        and soft_metrics.get("camera_protocol")
        == "sadekar_voxel_tfs_mirror_orthographic"
    ):
        raise ValueError("The frozen artifact does not declare the exact Sadekar TFS protocol")

    protocol_args = SimpleNamespace(
        device=cli.device,
        silhouettes=silhouettes,
        mr=int(trainer_args.get("mirror_mode", 1)),
        tsf=int(trainer_args.get("tsf_mode", 1)),
        print_mm=float(cli.print_mm),
    )
    if protocol_args.mr != 1 or protocol_args.tsf != 1:
        raise ValueError("The current fixed-three-view adapter expects mr=1 and tsf=1")

    soft_info = {"metrics": soft_metrics}
    proposal = pipeline._canonical_xyz_from_trainer_zyx(
        np.load(proposal_path).astype(bool)
    )
    resolution = int(proposal.shape[0])
    if proposal.shape != (resolution, resolution, resolution):
        raise ValueError(f"Expected a cubic proposal, got {proposal.shape}")

    targets = pipeline._generic_targets(soft_dir, resolution)
    if len(targets) != 6:
        raise ValueError(f"Three mirrored TFS views must produce six targets, got {len(targets)}")

    # Unlike the exact axis-aligned two-wall hull, the discrete TFS
    # backprojection is only a repair envelope.  Preserve every soft voxel
    # already validated by the exact renderer instead of clipping it away.
    repair_envelope = pipeline._generic_visual_hull(
        targets,
        protocol_args,
        resolution,
        soft_info,
        preserve_xyz=proposal,
    )

    components_before = int(label(proposal, structure=S6)[1])
    delivered, bridge_voxels = connect6_min_bridges(
        proposal,
        repair_envelope,
        max_passes=2,
        bridge_radius=int(cli.bridge_radius),
    )
    components_after = int(label(delivered, structure=S6)[1])
    if components_after != 1:
        raise RuntimeError(
            f"Minimal bridge repair left {components_after} components"
        )

    # The core is a structural proxy only.  It is not lifted back to replace
    # the high-resolution projection surface.
    core48 = pipeline.downsample_max_to(delivered, 48) >= 0.5
    core48_envelope = pipeline.downsample_max_to(repair_envelope, 48) >= 0.5
    core48, core_bridge_voxels = connect6_min_bridges(
        core48,
        core48_envelope,
        max_passes=2,
        bridge_radius=0,
    )

    qa = pipeline._generic_metrics(
        delivered,
        targets,
        int(repair_envelope.sum()),
        protocol_args,
        soft_info,
    )
    print_volume = qa.pop("print_volume")
    renders = qa.pop("renders")

    np.save(output / "delivered_volume_256.npy", delivered)
    np.save(output / "structural_core_48.npy", core48)
    for index, (target, rendered) in enumerate(zip(targets, renders)):
        save_mask(output / f"target_view_{index}.png", target)
        save_mask(output / f"projection_view_{index}.png", rendered)

    display_stl = output / "selected_display_fixed.stl"
    print_stl = output / f"selected_print_{qa['print_orientation']}.stl"
    display_mesh = pipeline.export_stl_at_print_scale(
        delivered, display_stl, cli.print_mm
    )
    export_oriented_print_mesh(
        display_mesh,
        qa["print_orientation"],
        print_stl,
    )

    manifest = {
        "experiment": "fixed3_from_two_wall_minimal_isolated",
        "scope": (
            "Three fixed Sadekar-TFS views; production two-wall source remains untouched"
        ),
        "soft_artifact_dir": str(soft_dir),
        "silhouettes": [str(path) for path in silhouettes],
        "protocol": {
            "base_views": 3,
            "mirror_mode": protocol_args.mr,
            "rendered_views": len(targets),
            "tsf": protocol_args.tsf,
            "camera_protocol": soft_metrics["camera_protocol"],
        },
        "delivery": {
            "resolution": resolution,
            "proposal_voxels": int(proposal.sum()),
            "components_before": components_before,
            "bridge_voxels": int(bridge_voxels),
            "bridge_radius": int(cli.bridge_radius),
            "components_after": components_after,
            "structural_core_resolution": 48,
            "structural_core_voxels": int(core48.sum()),
            "structural_core_bridge_voxels": int(core_bridge_voxels),
        },
        "qa": qa,
        "display_stl": str(display_stl),
        "print_stl": str(print_stl),
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
