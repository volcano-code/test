#!/usr/bin/env python3
"""Recover a V256 delivery candidate from an already trained fixed-view soft field.

This isolated diagnostic keeps the largest 6-connected V256 component and fills
enclosed cavities before SDF export. C48 is emitted only as a structural proxy;
it never owns delivery geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes, label


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SNAPSHOT = HERE / "shadowforge_pipeline_snapshot"
sys.path[:0] = [str(SNAPSHOT), str(ROOT)]

import unified_run as pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soft-artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--silhouettes", nargs=3, type=Path, required=True)
    parser.add_argument("--mr", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-mm", type=float, default=100.0)
    args = parser.parse_args()

    artifact = args.soft_artifact_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metrics_path = artifact / "metrics.json"
    proposal_path = artifact / "occupancy_soft_proposal.npy"
    if not metrics_path.is_file() or not proposal_path.is_file():
        raise FileNotFoundError(
            "The artifact directory must contain metrics.json and "
            "occupancy_soft_proposal.npy"
        )

    soft_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    soft_info = {"metrics": soft_metrics, "artifact_dir": str(artifact)}
    proposal_zyx = np.load(proposal_path).astype(bool)
    proposal_xyz = pipeline._canonical_xyz_from_trainer_zyx(proposal_zyx)

    components, count = label(proposal_xyz, structure=pipeline.S6)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    connected = components == largest_label
    delivery = binary_fill_holes(connected).astype(bool)

    class ProtocolArgs:
        pass

    protocol = ProtocolArgs()
    protocol.device = args.device
    protocol.silhouettes = list(args.silhouettes)
    protocol.mr = args.mr
    protocol.tsf = 1
    protocol.print_mm = args.print_mm

    targets = pipeline._generic_targets(artifact, 256)
    measured = pipeline._generic_prescreen_metrics(
        delivery,
        targets,
        int(proposal_xyz.sum()),
        protocol,
        soft_info,
    )

    volume_path = output / "delivery_v256_connected_filled.npy"
    proxy_path = output / "structural_proxy_c48.npy"
    stl_path = output / "delivery_v256_connected_filled_sdf_100mm.stl"
    np.save(volume_path, delivery)
    proxy = (
        pipeline.downsample_max_to(delivery.astype(np.float32), 48) >= 0.5
    )
    np.save(proxy_path, proxy)
    mesh = pipeline.export_stl_at_print_scale(
        delivery, stl_path, args.print_mm
    )
    mesh_watertight, mesh_parts, mesh_outer_parts = pipeline.mesh_qa(delivery)

    report = {
        "schema": "pcraft_fixed3_v256_recovery_v1",
        "purpose": (
            "Isolated recovery diagnostic; V256 owns delivery geometry and "
            "C48 is structural-proxy only."
        ),
        "source_artifact": str(artifact),
        "source_components": int(count),
        "source_voxels": int(proposal_xyz.sum()),
        "largest_component_fraction": float(
            connected.sum() / max(int(proposal_xyz.sum()), 1)
        ),
        "delivery_voxels": int(delivery.sum()),
        "mean_iou": measured["mean_iou"],
        "min_iou": measured["min_iou"],
        "view_ious": measured["view_ious"],
        "mean_dice": measured["mean_dice"],
        "voxel_components": measured["components"],
        "mesh_watertight": bool(mesh_watertight and mesh.is_watertight),
        "mesh_parts": int(mesh_parts),
        "mesh_outer_parts": int(mesh_outer_parts),
        "volume_npy": str(volume_path),
        "structural_proxy_c48": str(proxy_path),
        "display_stl": str(stl_path),
    }
    (output / "REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
