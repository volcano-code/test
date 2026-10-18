#!/usr/bin/env python3
"""Fill sealed delivery cavities and rebuild one production SDF surface.

This is deliberately isolated from the mature two-wall implementation.  It
operates on an already delivered V256 volume, preserves every
exterior-connected opening, fills only sealed background regions, and exports
the result through the same SDF mesh path used by the production pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import binary_propagation, generate_binary_structure, label

ROOT = Path(__file__).resolve().parents[2]
for module_path in (ROOT, ROOT / "shadowforge_pipeline", ROOT / "generation"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

import unified_run as pipeline

S6 = generate_binary_structure(3, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-mm", type=float, default=100.0)
    parser.add_argument("--smooth-sigma", type=float, default=0.6)
    return parser.parse_args()


def fill_sealed_cavities(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill background components that cannot reach the padded exterior."""
    solid = np.asarray(volume, dtype=bool)
    padded = np.pad(solid, 1, mode="constant", constant_values=False)
    background = ~padded
    seed = np.zeros_like(background)
    seed[0, :, :] = background[0, :, :]
    seed[-1, :, :] = background[-1, :, :]
    seed[:, 0, :] = background[:, 0, :]
    seed[:, -1, :] = background[:, -1, :]
    seed[:, :, 0] = background[:, :, 0]
    seed[:, :, -1] = background[:, :, -1]
    exterior = binary_propagation(seed, structure=S6, mask=background)
    cavities_padded = background & ~exterior
    filled_padded = padded | cavities_padded
    return filled_padded[1:-1, 1:-1, 1:-1], cavities_padded[1:-1, 1:-1, 1:-1]


def export_oriented_print_mesh(
    display_mesh: trimesh.Trimesh,
    orientation: str,
    path: Path,
) -> None:
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
        mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))
    center = mesh.bounds.mean(axis=0)
    mesh.apply_translation((-center[0], -center[1], -mesh.bounds[0, 2]))
    mesh.export(path)


def mesh_audit(mesh: trimesh.Trimesh) -> dict:
    edge_counts = np.bincount(mesh.edges_unique_inverse)
    parts = mesh.split(only_watertight=False)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "connected_surface_parts": int(len(parts)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(edge_counts > 2)),
        "volume_mm3": float(abs(mesh.volume)),
        "surface_area_mm2": float(mesh.area),
        "extents_mm": [float(v) for v in mesh.extents],
        "bounds_mm": [[float(v) for v in row] for row in mesh.bounds],
    }


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def projection_audit(
    volume: np.ndarray,
    manifest: dict,
    output: Path,
    device: str,
) -> dict:
    soft_dir = Path(manifest["soft_artifact_dir"])
    soft_metrics = json.loads((soft_dir / "metrics.json").read_text(encoding="utf-8"))
    trainer_args = soft_metrics.get("args", {})
    silhouettes = [Path(path) for path in trainer_args.get("shadow_files", [])]
    protocol_args = SimpleNamespace(
        device=device,
        silhouettes=silhouettes,
        mr=int(trainer_args.get("mirror_mode", 1)),
        tsf=int(trainer_args.get("tsf_mode", 1)),
        print_mm=float(manifest.get("qa", {}).get("print_mm", 100.0)),
    )
    targets = pipeline._generic_targets(soft_dir, int(volume.shape[0]))
    renders = pipeline._generic_render(
        volume,
        protocol_args,
        {"metrics": soft_metrics},
    )
    view_ious: list[float] = []
    view_dices: list[float] = []
    for index, (target, rendered) in enumerate(zip(targets, renders)):
        inter = int(np.logical_and(target, rendered).sum())
        union = int(np.logical_or(target, rendered).sum())
        denom = int(target.sum() + rendered.sum())
        view_ious.append(float(inter / max(union, 1)))
        view_dices.append(float(2.0 * inter / max(denom, 1)))
        save_mask(output / f"target_view_{index}.png", target)
        save_mask(output / f"projection_view_{index}.png", rendered)
    return {
        "mean_iou": float(np.mean(view_ious)),
        "min_iou": float(np.min(view_ious)),
        "mean_dice": float(np.mean(view_dices)),
        "view_ious": view_ious,
        "view_dices": view_dices,
    }


def save_preview(mesh: trimesh.Trimesh, path: Path) -> None:
    """Save a lightweight, deterministic preview without an OpenGL context."""
    faces = mesh.faces
    max_faces = 45000
    if len(faces) > max_faces:
        indices = np.linspace(0, len(faces) - 1, max_faces, dtype=np.int64)
        faces = faces[indices]
    triangles = mesh.vertices[faces]
    fig = plt.figure(figsize=(7.2, 7.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        triangles,
        facecolor="#d7b36a",
        edgecolor="#7c6337",
        linewidth=0.08,
        alpha=1.0,
    )
    ax.add_collection3d(collection)
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    radius = float(np.max(mesh.extents)) * 0.56
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=24, azim=-46)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output = (
        args.output.resolve()
        if args.output
        else input_dir / "volume_cavity_sdf_remesh"
    )
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = input_dir / "MANIFEST.json"
    volume_path = input_dir / "delivered_volume_256.npy"
    if not manifest_path.is_file() or not volume_path.is_file():
        raise FileNotFoundError(
            "Input directory must contain MANIFEST.json and delivered_volume_256.npy"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = np.load(volume_path).astype(bool)
    before_components = int(label(original, structure=S6)[1])

    filled, cavity_mask = fill_sealed_cavities(original)
    after_components = int(label(filled, structure=S6)[1])
    cavity_labels, cavity_count = label(cavity_mask, structure=S6)
    cavity_sizes = np.bincount(cavity_labels.ravel())[1:]
    largest_cavities = sorted((int(v) for v in cavity_sizes), reverse=True)[:20]

    np.save(output / "filled_volume_256.npy", filled)
    np.save(output / "cavity_mask_256.npy", cavity_mask)

    display_stl = output / "selected_display_cavity_filled_sdf.stl"
    display_mesh = pipeline.export_stl_at_print_scale(
        filled,
        display_stl,
        float(args.print_mm),
    )
    orientation = manifest.get("qa", {}).get("print_orientation", "display_z_down")
    print_stl = output / f"selected_print_cavity_filled_sdf_{orientation}.stl"
    export_oriented_print_mesh(display_mesh, orientation, print_stl)

    mesh_info = mesh_audit(display_mesh)
    projection = projection_audit(filled, manifest, output, args.device)
    save_preview(
        display_mesh,
        output / "selected_display_cavity_filled_sdf_preview.png",
    )

    report = {
        "operation": "sealed-cavity fill followed by production SDF remeshing",
        "input_dir": str(input_dir),
        "input_volume": str(volume_path),
        "resolution": [int(v) for v in original.shape],
        "volume_qa": {
            "solid_components_before": before_components,
            "solid_components_after": after_components,
            "occupied_voxels_before": int(original.sum()),
            "occupied_voxels_after": int(filled.sum()),
            "filled_cavity_voxels": int(cavity_mask.sum()),
            "filled_cavity_fraction_of_original": float(
                cavity_mask.sum() / max(int(original.sum()), 1)
            ),
            "sealed_cavity_components": int(cavity_count),
            "largest_sealed_cavities_voxels": largest_cavities,
        },
        "projection_qa_before": {
            key: manifest.get("qa", {}).get(key)
            for key in ("mean_iou", "min_iou", "mean_dice", "view_ious", "view_dices")
        },
        "projection_qa_after": projection,
        "mesh_qa": mesh_info,
        "acceptance": {
            "one_solid_component": after_components == 1,
            "one_surface_component": mesh_info["connected_surface_parts"] == 1,
            "watertight": mesh_info["watertight"],
            "zero_boundary_edges": mesh_info["boundary_edges"] == 0,
            "zero_nonmanifold_edges": mesh_info["nonmanifold_edges"] == 0,
        },
        "display_stl": str(display_stl),
        "print_stl": str(print_stl),
        "filled_volume": str(output / "filled_volume_256.npy"),
        "cavity_mask": str(output / "cavity_mask_256.npy"),
        "preview": str(output / "selected_display_cavity_filled_sdf_preview.png"),
    }
    report["accepted"] = bool(all(report["acceptance"].values()))
    report_path = output / "cavity_fill_sdf_remesh_qa.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
