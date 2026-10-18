#!/usr/bin/env python3
"""Search compact V256 geometry owners without touching two-wall delivery."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
ISOLATED = ROOT / "experiments/fixed_surface_xyz_isolated_20260726"
for module_path in (
    ISOLATED / "shadowforge_pipeline",
    ISOLATED / "generation",
    ISOLATED,
    ROOT,
):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import unified_run as pipeline
from voxel_ops import sdf_watertight_mesh

S6 = ndimage.generate_binary_structure(3, 1)
S26 = ndimage.generate_binary_structure(3, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-artifact-dir", type=Path, required=True)
    parser.add_argument("--exact-envelope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def largest_component(volume: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(volume, structure=S6)
    if count == 0:
        return np.zeros_like(volume, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def surface_fraction(volume: np.ndarray) -> float:
    occupied = np.asarray(volume, dtype=bool)
    count = int(occupied.sum())
    if count == 0:
        return 1.0
    interior = ndimage.binary_erosion(occupied, structure=S6, iterations=1)
    return float((count - int(interior.sum())) / count)


def fill_slices(volume: np.ndarray, axis: int) -> np.ndarray:
    result = np.asarray(volume, dtype=bool).copy()
    for index in range(result.shape[axis]):
        key = [slice(None)] * 3
        key[axis] = index
        key_tuple = tuple(key)
        result[key_tuple] = ndimage.binary_fill_holes(
            result[key_tuple],
            structure=ndimage.generate_binary_structure(2, 2),
        )
    return result


def make_protocol(metrics: dict, device: str) -> SimpleNamespace:
    trainer_args = metrics.get("args", {})
    return SimpleNamespace(
        device=device,
        silhouettes=[
            Path(path) for path in trainer_args.get("shadow_files", [])
        ],
        mr=int(trainer_args.get("mirror_mode", 0)),
        tsf=int(trainer_args.get("tsf_mode", 0)),
        print_mm=100.0,
    )


def save_preview(mesh: trimesh.Trimesh, path: Path) -> None:
    faces = mesh.faces
    if len(faces) > 50000:
        indices = np.linspace(0, len(faces) - 1, 50000, dtype=np.int64)
        faces = faces[indices]
    triangles = mesh.vertices[faces]
    figure = plt.figure(figsize=(6.4, 6.4), dpi=180)
    axis = figure.add_subplot(111, projection="3d")
    axis.add_collection3d(
        Poly3DCollection(
            triangles,
            facecolor="#d8b46a",
            edgecolor="#725a32",
            linewidth=0.06,
        )
    )
    center = mesh.bounds.mean(axis=0)
    radius = float(np.max(mesh.extents)) * 0.56
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.view_init(elev=24, azim=-46)
    axis.set_box_aspect((1, 1, 1))
    axis.set_axis_off()
    figure.patch.set_alpha(0.0)
    axis.set_facecolor((0, 0, 0, 0))
    figure.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    soft_dir = args.soft_artifact_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metrics = json.loads(
        (soft_dir / "metrics.json").read_text(encoding="utf-8")
    )
    protocol = make_protocol(metrics, args.device)
    soft_info = {"metrics": metrics}
    proposal = pipeline._canonical_xyz_from_trainer_zyx(
        np.load(soft_dir / "occupancy_soft_proposal.npy").astype(bool)
    )
    probability = pipeline._canonical_xyz_from_trainer_zyx(
        np.load(soft_dir / "occupancy_probability_core.npy").astype(np.float32)
    )
    targets = pipeline._generic_targets(soft_dir, int(proposal.shape[0]))
    reference_voxels = int(proposal.sum())
    main = largest_component(proposal)
    envelope = np.load(args.exact_envelope).astype(bool)

    candidates: dict[str, np.ndarray] = {
        "exact_envelope": envelope,
        "soft_main": main,
    }
    for iterations in (1, 2, 3, 4):
        closed = ndimage.binary_closing(
            main, structure=S26, iterations=iterations
        )
        candidates[f"main_close{iterations}"] = largest_component(closed)

    filled_axes = [fill_slices(main, axis) for axis in range(3)]
    fill_votes = sum(item.astype(np.uint8) for item in filled_axes)
    candidates["main_slice_fill_majority"] = largest_component(fill_votes >= 2)
    candidates["main_slice_fill_union"] = largest_component(fill_votes >= 1)

    # Constrain any fill to a one-voxel camera-consistent envelope and retain
    # high-confidence soft support. This is a geometry-owner diagnostic, not a
    # selectable final result.
    envelope1 = ndimage.binary_dilation(envelope, structure=S26, iterations=1)
    confident = probability >= 0.5
    for name in ("main_slice_fill_majority", "main_slice_fill_union"):
        constrained = candidates[name] & envelope1
        constrained |= main & confident
        candidates[f"{name}_camera_bound"] = largest_component(constrained)

    rows: list[dict] = []
    for name, candidate in candidates.items():
        candidate = largest_component(candidate)
        candidate_metrics = pipeline._generic_prescreen_metrics(
            candidate,
            targets,
            reference_voxels,
            protocol,
            soft_info,
        )
        row = {
            "candidate": name,
            "occupied_voxels": int(candidate.sum()),
            "surface_fraction": surface_fraction(candidate),
            **candidate_metrics,
        }
        rows.append(row)
        np.save(output / f"{name}.npy", candidate.astype(np.uint8))
        print(json.dumps(row), flush=True)

    eligible = [
        row
        for row in rows
        if row["mean_iou"] >= 0.95 and row["min_iou"] >= 0.93
    ]
    eligible.sort(
        key=lambda row: (
            row["surface_fraction"],
            -row["min_iou"],
            -row["mean_iou"],
        )
    )
    previews: list[dict] = []
    for row in eligible[:4]:
        name = row["candidate"]
        volume = np.load(output / f"{name}.npy").astype(bool)
        mesh = sdf_watertight_mesh(volume, smooth_sigma=0.5)
        stl_path = output / f"{name}.stl"
        preview_path = output / f"{name}_preview.png"
        mesh.export(stl_path)
        save_preview(mesh, preview_path)
        previews.append(
            {
                "candidate": name,
                "stl": str(stl_path),
                "preview": str(preview_path),
                "triangles": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
            }
        )

    manifest = {"rows": rows, "previews": previews}
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
