#!/usr/bin/env python3
"""Build one clean exterior SDF shell from an isolated V256 delivery volume.

This utility is intentionally isolated from the mature two-view pipeline. It
regularizes only the supplied three-view delivery volume, evaluates projection
fidelity under the stored fixed-view protocol, and accepts a result only when
the exported mesh has one watertight surface component.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    iterate_structure,
    label,
)
from skimage.measure import marching_cubes


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SNAPSHOT = HERE / "shadowforge_pipeline_snapshot"
sys.path[:0] = [str(SNAPSHOT), str(ROOT)]

import unified_run as pipeline  # noqa: E402


S6 = generate_binary_structure(3, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-volume", type=Path, required=True)
    parser.add_argument("--soft-artifact-dir", type=Path, required=True)
    parser.add_argument("--silhouettes", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mr", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-mm", type=float, default=100.0)
    parser.add_argument("--closing-radii", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--smooth-sigmas", nargs="+", type=float, default=(0.55, 0.8))
    parser.add_argument("--min-mean-iou", type=float, default=0.95)
    parser.add_argument("--min-view-iou", type=float, default=0.90)
    return parser.parse_args()


def largest_component(volume: np.ndarray) -> np.ndarray:
    labels, count = label(volume, structure=S6)
    if count == 0:
        raise RuntimeError("The input volume contains no occupied voxel.")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def regularize_volume(volume: np.ndarray, radius: int) -> np.ndarray:
    """Close sub-voxel channels without joining remote components."""
    padded = np.pad(volume, radius + 2, mode="constant", constant_values=False)
    if radius > 0:
        structure = iterate_structure(S6, radius)
        padded = binary_closing(padded, structure=structure)
    padded = binary_fill_holes(padded)
    cropped = padded[
        radius + 2 : -(radius + 2),
        radius + 2 : -(radius + 2),
        radius + 2 : -(radius + 2),
    ]
    return largest_component(cropped)


def sdf_mesh(volume: np.ndarray, sigma: float, print_mm: float) -> trimesh.Trimesh:
    """Extract a smoothed SDF surface and retain its true exterior shell."""
    padded = np.pad(volume, 3, mode="constant", constant_values=False)
    inside = distance_transform_edt(padded)
    outside = distance_transform_edt(~padded)
    field = (inside - outside).astype(np.float32)
    del inside, outside
    if sigma > 0:
        field = gaussian_filter(field, sigma=sigma)
    vertices, faces, _, _ = marching_cubes(
        field,
        level=0.0,
        spacing=(1.0, 1.0, 1.0),
        allow_degenerate=False,
    )
    vertices -= 3.0
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    parts = list(mesh.split(only_watertight=False))
    if not parts:
        raise RuntimeError("SDF extraction produced no surface.")
    # Internal cavity shells have smaller enclosed volume. Keeping the largest
    # enclosing shell removes them without deleting the exterior body.
    mesh = max(
        parts,
        key=lambda part: (
            bool(part.is_watertight),
            abs(float(part.volume)) if part.is_watertight else 0.0,
            float(part.area),
        ),
    )
    trimesh.repair.fix_normals(mesh, multibody=False)
    mesh.remove_unreferenced_vertices()
    scale = float(print_mm) / max(float(mesh.extents.max()), 1e-8)
    mesh.apply_scale(scale)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def mesh_audit(mesh: trimesh.Trimesh) -> dict[str, object]:
    edge_counts = np.bincount(mesh.edges_unique_inverse)
    parts = mesh.split(only_watertight=False)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "surface_components": int(len(parts)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(edge_counts > 2)),
        "volume_mm3": float(abs(mesh.volume)),
        "area_mm2": float(mesh.area),
    }


def projection_audit(
    volume: np.ndarray,
    soft_dir: Path,
    silhouettes: list[Path],
    mr: int,
    device: str,
) -> dict[str, object]:
    metrics = json.loads((soft_dir / "metrics.json").read_text(encoding="utf-8"))
    protocol = SimpleNamespace(
        device=device,
        silhouettes=silhouettes,
        mr=mr,
        tsf=1,
        print_mm=100.0,
    )
    targets = pipeline._generic_targets(soft_dir, int(volume.shape[0]))
    renders = pipeline._generic_render(volume, protocol, {"metrics": metrics})
    ious: list[float] = []
    dices: list[float] = []
    for target, rendered in zip(targets, renders):
        intersection = int(np.logical_and(target, rendered).sum())
        union = int(np.logical_or(target, rendered).sum())
        ious.append(float(intersection / max(union, 1)))
        dices.append(
            float(2 * intersection / max(int(target.sum() + rendered.sum()), 1))
        )
    return {
        "mean_iou": float(np.mean(ious)),
        "min_iou": float(np.min(ious)),
        "mean_dice": float(np.mean(dices)),
        "view_ious": ious,
        "view_dices": dices,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    soft_dir = args.soft_artifact_dir.resolve()
    volume = np.load(args.input_volume.resolve()).astype(bool)
    volume = largest_component(volume)

    candidates: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    for radius in args.closing_radii:
        regularized = regularize_volume(volume, radius)
        projection = projection_audit(
            regularized,
            soft_dir,
            [path.resolve() for path in args.silhouettes],
            args.mr,
            args.device,
        )
        np.save(output / f"regularized_r{radius}.npy", regularized)
        for sigma in args.smooth_sigmas:
            slug = f"r{radius}_s{sigma:.2f}".replace(".", "p")
            mesh = sdf_mesh(regularized, sigma, args.print_mm)
            mesh_info = mesh_audit(mesh)
            stl_path = output / f"candidate_{slug}.stl"
            mesh.export(stl_path)
            accepted = bool(
                mesh_info["surface_components"] == 1
                and mesh_info["watertight"]
                and mesh_info["boundary_edges"] == 0
                and mesh_info["nonmanifold_edges"] == 0
                and projection["mean_iou"] >= args.min_mean_iou
                and projection["min_iou"] >= args.min_view_iou
            )
            record = {
                "closing_radius": radius,
                "smooth_sigma": sigma,
                "projection": projection,
                "mesh": mesh_info,
                "accepted": accepted,
                "stl": str(stl_path),
                "volume": str(output / f"regularized_r{radius}.npy"),
            }
            candidates.append(record)
            if accepted and (
                selected is None
                or projection["mean_iou"] > selected["projection"]["mean_iou"]
            ):
                selected = record
            print(json.dumps(record, ensure_ascii=False), flush=True)

    report = {
        "schema": "pcraft_isolated_fixed3_single_shell_sdf_v1",
        "input_volume": str(args.input_volume.resolve()),
        "soft_artifact_dir": str(soft_dir),
        "acceptance": {
            "min_mean_iou": args.min_mean_iou,
            "min_view_iou": args.min_view_iou,
            "one_surface_component": True,
            "watertight": True,
            "zero_boundary_edges": True,
            "zero_nonmanifold_edges": True,
        },
        "selected": selected,
        "candidates": candidates,
        "status": "ACCEPTED" if selected else "NO_ACCEPTED_CANDIDATE",
    }
    if selected:
        source = Path(selected["stl"])
        final_path = output / "selected_display_single_shell_sdf_100mm.stl"
        shutil.copy2(source, final_path)
        selected["stl"] = str(final_path)
    (output / "REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
