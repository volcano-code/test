#!/usr/bin/env python3
"""Isolated V256-owned delivery for fixed-surface multi-view shadow art.

This adapter deliberately does not modify the mature two-wall implementation.
It consumes a frozen soft-stage artifact, keeps the high-resolution proposal as
the geometry owner, evaluates every state with the exact recorded cameras, and
uses C48 only as a structural-query proxy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymeshlab
import trimesh
import vtk
from PIL import Image
from scipy import ndimage
from vtk.util.numpy_support import (
    numpy_to_vtk,
    numpy_to_vtkIdTypeArray,
    vtk_to_numpy,
)

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
    parser = argparse.ArgumentParser(
        description="Gate-driven V256 delivery for fixed-surface multi-view tasks."
    )
    parser.add_argument("--soft-artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-mm", type=float, default=100.0)
    parser.add_argument("--min-mean-iou", type=float, default=0.95)
    parser.add_argument("--min-view-iou", type=float, default=0.95)
    parser.add_argument("--carve-checkpoints", type=int, default=0)
    parser.add_argument("--carve-voxels", type=int, default=2048)
    parser.add_argument("--max-repair-components", type=int, default=64)
    parser.add_argument("--max-repair-gap", type=float, default=6.0)
    parser.add_argument("--max-repair-fraction", type=float, default=0.02)
    parser.add_argument(
        "--carve-mode",
        choices=("target_rays", "occupied_rays"),
        default="target_rays",
        help=(
            "Preserve only declared target rays by default. The legacy "
            "occupied-rays mode also preserves false-positive envelope rays."
        ),
    )
    parser.add_argument(
        "--sdf-sigmas",
        default="0.0,0.3,0.5,0.8,1.0,1.2",
        help="Comma-separated SDF smoothing candidates.",
    )
    parser.add_argument(
        "--sdf-dilations",
        default="0,1,2",
        help="Comma-separated pre-SDF dilation candidates.",
    )
    parser.add_argument("--sdf-level", type=float, default=0.0)
    parser.add_argument("--max-surface-fraction", type=float, default=0.35)
    parser.add_argument("--max-mesh-triangles", type=int, default=2000000)
    parser.add_argument("--min-exterior-shell-fraction", type=float, default=0.98)
    parser.add_argument(
        "--allow-bounded-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").save(path)


def component_labels(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, count = ndimage.label(np.asarray(volume, dtype=bool), structure=S6)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    if sizes.size:
        sizes[0] = 0
    return labels, sizes


def largest_component(volume: np.ndarray) -> tuple[np.ndarray, dict]:
    labels, sizes = component_labels(volume)
    count = max(int(sizes.size - 1), 0)
    occupied = int(np.asarray(volume, dtype=bool).sum())
    if count == 0:
        return np.zeros_like(volume, dtype=bool), {
            "components_before": 0,
            "retained_voxels": 0,
            "retained_fraction": 0.0,
        }
    root = int(np.argmax(sizes))
    kept = labels == root
    return kept, {
        "components_before": count,
        "retained_voxels": int(kept.sum()),
        "retained_fraction": float(kept.sum() / max(occupied, 1)),
    }


def gate_pass(metrics: dict, args: argparse.Namespace) -> bool:
    return bool(
        metrics["components"] == 1
        and metrics["mean_iou"] >= args.min_mean_iou
        and metrics["min_iou"] >= args.min_view_iou
    )


def parse_number_list(value: str, cast: type) -> list:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value.")
    return [cast(item) for item in items]


def make_protocol(
    soft_metrics: dict, device: str, print_mm: float
) -> tuple[SimpleNamespace, list[Path]]:
    trainer_args = soft_metrics.get("args", {})
    silhouettes = [Path(path) for path in trainer_args.get("shadow_files", [])]
    if not silhouettes:
        raise ValueError("The soft artifact does not record input silhouettes.")
    if len(silhouettes) > 3 and soft_metrics.get("tsf_direct"):
        raise ValueError(
            "The recorded direct Sadekar TFS protocol supports one to three "
            "base views (up to six including mirrored views)."
        )
    protocol = SimpleNamespace(
        device=device,
        silhouettes=silhouettes,
        mr=int(trainer_args.get("mirror_mode", 0)),
        tsf=int(trainer_args.get("tsf_mode", 0)),
        print_mm=float(print_mm),
    )
    return protocol, silhouettes


def prescreen(
    volume: np.ndarray,
    targets: list[np.ndarray],
    reference_voxels: int,
    protocol: SimpleNamespace,
    soft_info: dict,
) -> dict:
    return pipeline._generic_prescreen_metrics(
        volume, targets, reference_voxels, protocol, soft_info
    )


def line_voxels(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    steps = int(np.max(np.abs(end - start))) + 1
    line = np.rint(
        np.stack(
            [
                np.linspace(start[axis], end[axis], steps)
                for axis in range(3)
            ],
            axis=1,
        )
    ).astype(np.int64)
    return np.unique(line, axis=0)


def bounded_connect_to_root(
    source: np.ndarray,
    root_seed: np.ndarray,
    max_components: int,
    max_gap: float,
    max_added_voxels: int,
) -> tuple[np.ndarray, dict]:
    """Attach only nearby source components under an explicit voxel budget."""
    source = np.asarray(source, dtype=bool)
    root = np.asarray(root_seed, dtype=bool).copy()
    labels, sizes = component_labels(source)
    if not root.any() or sizes.size <= 2:
        return root, {
            "considered_components": 0,
            "attached_components": 0,
            "added_voxels": 0,
        }

    # One distance transform replaces tens of thousands of unconstrained
    # component-to-component bridge attempts.
    distances, nearest = ndimage.distance_transform_edt(
        ~root, return_indices=True
    )
    root_labels = labels[root]
    root_labels = root_labels[root_labels > 0]
    root_label = (
        int(np.bincount(root_labels).argmax()) if root_labels.size else -1
    )
    candidates: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
    for label_id in range(1, sizes.size):
        if label_id == root_label or sizes[label_id] == 0:
            continue
        coords = np.argwhere(labels == label_id)
        if not len(coords):
            continue
        component_distances = distances[tuple(coords.T)]
        local_index = int(np.argmin(component_distances))
        distance = float(component_distances[local_index])
        if distance > max_gap:
            continue
        start = coords[local_index]
        end = nearest[:, start[0], start[1], start[2]]
        candidates.append((distance, -int(sizes[label_id]), label_id, start, end))

    candidates.sort(key=lambda item: (item[0], item[1]))
    attached = 0
    before = int(root.sum())
    for _, _, label_id, start, end in candidates[:max_components]:
        component = labels == label_id
        bridge = line_voxels(start, end)
        proposed_added = int(np.logical_and(component, ~root).sum())
        proposed_added += int(
            np.count_nonzero(~root[tuple(bridge.T)])
        )
        if int(root.sum()) - before + proposed_added > max_added_voxels:
            continue
        root |= component
        root[tuple(bridge.T)] = True
        attached += 1

    connected, audit = largest_component(root)
    return connected, {
        "considered_components": int(len(candidates)),
        "attached_components": int(attached),
        "added_voxels": int(connected.sum()) - before,
        "components_after": int(
            ndimage.label(connected, structure=S6)[1]
        ),
        **audit,
    }


def attempt_bounded_repair(
    current: np.ndarray,
    proposal: np.ndarray,
    probability: np.ndarray,
    targets: list[np.ndarray],
    protocol: SimpleNamespace,
    soft_info: dict,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
    """Recover declared rays using only the frozen proposal and a small budget."""
    recovered = pipeline.restore_declared_view_rays(
        current,
        proposal,
        probability,
        targets,
        protocol,
        soft_info,
    )
    budget = max(1, int(proposal.sum() * args.max_repair_fraction))
    return bounded_connect_to_root(
        recovered,
        current,
        max_components=args.max_repair_components,
        max_gap=args.max_repair_gap,
        max_added_voxels=budget,
    )


def target_ray_guarded_carve_step(
    volume_xyz: np.ndarray,
    priority_xyz: np.ndarray,
    targets: list[np.ndarray],
    soft_info: dict,
    max_removals: int | None,
) -> tuple[np.ndarray, int]:
    """Delete low-priority boundary voxels while preserving declared target rays.

    The legacy generic carve protects every currently occupied ray, including
    false-positive rays from the soft envelope. This direct-TFS variant protects
    only rays required by the normalized targets and therefore permits attached
    grid-like envelope material to be removed.
    """
    if not pipeline._uses_sadekar_direct_tfs(soft_info):
        raise RuntimeError(
            "Target-ray carving currently requires exact direct Sadekar TFS metadata."
        )

    occ = pipeline._trainer_zyx_from_canonical_xyz(
        np.asarray(volume_xyz, dtype=bool)
    ).copy()
    priority = pipeline._trainer_zyx_from_canonical_xyz(
        np.asarray(priority_xyz, dtype=np.float32)
    )
    covers = {axis: occ.sum(axis=axis) for axis in range(3)}
    view_specs: list[tuple[int, np.ndarray]] = []
    for index, declared_target in enumerate(targets):
        axis, transform = pipeline._sadekar_tfs_view_spec(index)
        target = pipeline._transform_sadekar_tfs_target(
            declared_target, transform
        )
        expected = tuple(
            occ.shape[dimension]
            for dimension in range(3)
            if dimension != axis
        )
        if target.shape != expected:
            raise ValueError(
                f"Target {index} has shape {target.shape}, expected {expected} "
                f"for direct-TFS ray axis {axis}."
            )
        view_specs.append((axis, target))

    pad = np.pad(occ, 1)
    interior = (
        pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1]
        & pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1]
        & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
    )
    boundary = np.argwhere(occ & ~interior)
    if not len(boundary):
        return np.asarray(volume_xyz, dtype=bool).copy(), 0

    # Remove boundary voxels that support no declared target ray first, then
    # follow the frozen soft probability within each support tier.
    support_count = np.zeros(len(boundary), dtype=np.uint8)
    for axis, target in view_specs:
        plane_coordinates = tuple(
            boundary[:, dimension]
            for dimension in range(3)
            if dimension != axis
        )
        support_count += target[plane_coordinates].astype(np.uint8)
    probability_scores = priority[tuple(boundary.T)]
    order = np.lexsort((probability_scores, support_count))
    boundary = boundary[order]

    removed = 0
    for coordinate in boundary:
        point = tuple(int(value) for value in coordinate)
        if not occ[point]:
            continue
        protected = False
        for axis, target in view_specs:
            ray_index = tuple(
                point[dimension]
                for dimension in range(3)
                if dimension != axis
            )
            if target[ray_index] and covers[axis][ray_index] <= 1:
                protected = True
                break
        if protected or not pipeline.local_simple_point(occ, *point):
            continue
        occ[point] = False
        for axis in covers:
            ray_index = tuple(
                point[dimension]
                for dimension in range(3)
                if dimension != axis
            )
            covers[axis][ray_index] -= 1
        removed += 1
        if max_removals is not None and removed >= max_removals:
            break

    return pipeline._canonical_xyz_from_trainer_zyx(occ), removed


def surface_fraction(volume: np.ndarray) -> float:
    occupied = np.asarray(volume, dtype=bool)
    total = int(occupied.sum())
    if total == 0:
        return 1.0
    eroded = ndimage.binary_erosion(occupied, structure=S6, iterations=1)
    return float((total - int(eroded.sum())) / total)


def reconstruct_sdf_volume(
    volume: np.ndarray,
    dilation: int,
    smooth_sigma: float,
    level: float,
) -> tuple[np.ndarray, dict]:
    """Create a solid reconstruction volume before the final mesh extraction."""
    source = np.asarray(volume, dtype=bool)
    if dilation > 0:
        source = ndimage.binary_dilation(
            source, structure=S6, iterations=dilation
        )
    # Treat diagonal empty passages as exterior-connected. The default
    # 6-neighbour fill incorrectly seals the narrow diagonal openings common in
    # multi-view shadow geometry and turns them into large false-positive slabs.
    source = ndimage.binary_fill_holes(source, structure=S26)
    sdf = (
        ndimage.distance_transform_edt(~source)
        - ndimage.distance_transform_edt(source)
    ).astype(np.float32)
    if smooth_sigma > 0:
        sdf = ndimage.gaussian_filter(sdf, sigma=smooth_sigma)
    reconstructed = sdf <= float(level)
    reconstructed = ndimage.binary_fill_holes(
        reconstructed, structure=S26
    )
    reconstructed, component_audit = largest_component(reconstructed)
    return reconstructed, {
        "dilation": int(dilation),
        "smooth_sigma": float(smooth_sigma),
        "level": float(level),
        "occupied_voxels": int(reconstructed.sum()),
        "surface_fraction": surface_fraction(reconstructed),
        **component_audit,
    }


def select_external_shell(volume: np.ndarray) -> tuple[trimesh.Trimesh, dict]:
    """Mesh a solid reconstruction and audit whether one shell dominates."""
    mesh = sdf_watertight_mesh(
        np.asarray(volume, dtype=bool), smooth_sigma=0.0
    )
    parts = list(mesh.split(only_watertight=False))
    watertight_parts = [part for part in parts if part.is_watertight]
    if not watertight_parts:
        raise RuntimeError("SDF reconstruction produced no watertight shell.")

    def shell_key(part: trimesh.Trimesh) -> tuple[float, float]:
        try:
            volume_score = abs(float(part.volume))
        except Exception:
            volume_score = 0.0
        return volume_score, float(part.area)

    shell_volumes = np.asarray(
        [max(shell_key(part)[0], 0.0) for part in watertight_parts],
        dtype=np.float64,
    )
    exterior_index = int(np.argmax(shell_volumes))
    exterior = watertight_parts[exterior_index].copy()
    exterior_fraction = float(
        shell_volumes[exterior_index] / max(float(shell_volumes.sum()), 1e-12)
    )
    # `Trimesh.process(validate=True)` can merge coincident vertices across
    # already valid marching-cubes seams and turn a watertight shell into a
    # non-manifold one. Keep the validated SDF topology intact.
    trimesh.repair.fix_normals(exterior, multibody=False)
    if not exterior.is_watertight:
        raise RuntimeError("The selected exterior SDF shell is not watertight.")
    return exterior, {
        "sdf_shells_total": int(len(parts)),
        "sdf_shells_watertight": int(len(watertight_parts)),
        "exterior_shell_fraction": exterior_fraction,
        "retained_shell_volume": float(abs(exterior.volume)),
        "retained_shell_area": float(exterior.area),
        "retained_shell_triangles": int(len(exterior.faces)),
        "retained_shell_watertight": bool(exterior.is_watertight),
        "retained_shell_winding_consistent": bool(
            exterior.is_winding_consistent
        ),
    }


def simplify_external_shell(
    mesh: trimesh.Trimesh, target_faces: int
) -> tuple[trimesh.Trimesh, dict]:
    """Reduce a watertight shell without changing its topology."""
    source_faces = int(len(mesh.faces))
    if source_faces <= target_faces:
        return mesh.copy(), {
            "simplification_applied": False,
            "source_triangles": source_faces,
            "target_triangles": int(target_faces),
            "result_triangles": source_faces,
        }

    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        )
    )
    mesh_set.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
        qualitythr=0.3,
        autoclean=True,
    )
    simplified_ml = mesh_set.current_mesh()
    simplified = trimesh.Trimesh(
        vertices=np.asarray(simplified_ml.vertex_matrix(), dtype=np.float64),
        faces=np.asarray(simplified_ml.face_matrix(), dtype=np.int64),
        process=True,
    )
    trimesh.repair.fix_normals(simplified, multibody=False)
    return simplified, {
        "simplification_applied": True,
        "source_triangles": source_faces,
        "target_triangles": int(target_faces),
        "result_triangles": int(len(simplified.faces)),
    }


def mesh_to_voxel_volume(
    mesh: trimesh.Trimesh,
    shape: tuple[int, int, int],
    coordinate_offset: float = 1.0,
) -> np.ndarray:
    """Rasterize the actual exported shell back into canonical XYZ voxels."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64) - float(
        coordinate_offset
    )
    faces = np.asarray(mesh.faces, dtype=np.int64)

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(vertices, deep=True))
    cells = np.empty((len(faces), 4), dtype=np.int64)
    cells[:, 0] = 3
    cells[:, 1:] = faces
    polygons = vtk.vtkCellArray()
    polygons.SetCells(
        len(faces),
        numpy_to_vtkIdTypeArray(cells.ravel(), deep=True),
    )
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polygons)

    image = vtk.vtkImageData()
    image.SetDimensions(*map(int, shape))
    image.SetSpacing(1.0, 1.0, 1.0)
    image.SetOrigin(0.0, 0.0, 0.0)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    vtk_to_numpy(image.GetPointData().GetScalars())[:] = 1

    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(polydata)
    stencil.SetOutputOrigin(image.GetOrigin())
    stencil.SetOutputSpacing(image.GetSpacing())
    stencil.SetOutputWholeExtent(image.GetExtent())
    stencil.Update()

    apply_stencil = vtk.vtkImageStencil()
    apply_stencil.SetInputData(image)
    apply_stencil.SetStencilConnection(stencil.GetOutputPort())
    apply_stencil.ReverseStencilOff()
    apply_stencil.SetBackgroundValue(0)
    apply_stencil.Update()

    zyx = vtk_to_numpy(
        apply_stencil.GetOutput().GetPointData().GetScalars()
    ).reshape((shape[2], shape[1], shape[0]))
    return zyx.transpose(2, 1, 0).astype(bool)


def in_memory_mesh_audit(mesh: trimesh.Trimesh) -> dict:
    parts = list(mesh.split(only_watertight=False))
    return {
        "components": int(len(parts)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "triangles": int(len(mesh.faces)),
        "volume": float(abs(mesh.volume)),
    }


def reload_mesh_audit(path: Path) -> dict:
    loaded = trimesh.load(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"Exported file {path} did not reload as one mesh.")
    parts = list(loaded.split(only_watertight=False))
    return {
        "reloaded_components": int(len(parts)),
        "reloaded_watertight": bool(loaded.is_watertight),
        "reloaded_winding_consistent": bool(loaded.is_winding_consistent),
        "reloaded_triangles": int(len(loaded.faces)),
        "reloaded_volume": float(abs(loaded.volume)),
        "reloaded_extents": [float(value) for value in loaded.extents],
    }


def scale_display_mesh(
    mesh: trimesh.Trimesh, print_mm: float
) -> trimesh.Trimesh:
    result = mesh.copy()
    extent = float(np.max(result.extents))
    if extent <= 0:
        raise ValueError("Cannot scale an empty mesh.")
    result.apply_scale(float(print_mm) / extent)
    center = result.bounds.mean(axis=0)
    result.apply_translation((-center[0], -center[1], -result.bounds[0, 2]))
    return result


def orient_print_mesh(
    display_mesh: trimesh.Trimesh, orientation: str
) -> trimesh.Trimesh:
    rotations = {
        "display_z_up": (0.0, (1.0, 0.0, 0.0)),
        "display_z_down": (np.pi, (1.0, 0.0, 0.0)),
        "display_x_up": (-np.pi / 2.0, (0.0, 1.0, 0.0)),
        "display_x_down": (np.pi / 2.0, (0.0, 1.0, 0.0)),
        "display_y_up": (np.pi / 2.0, (1.0, 0.0, 0.0)),
        "display_y_down": (-np.pi / 2.0, (1.0, 0.0, 0.0)),
    }
    result = display_mesh.copy()
    angle, axis = rotations[orientation]
    if angle:
        result.apply_transform(
            trimesh.transformations.rotation_matrix(angle, axis)
        )
    center = result.bounds.mean(axis=0)
    result.apply_translation((-center[0], -center[1], -result.bounds[0, 2]))
    return result


def main() -> None:
    args = parse_args()
    soft_dir = args.soft_artifact_dir.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metrics_path = soft_dir / "metrics.json"
    soft_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    protocol, silhouettes = make_protocol(
        soft_metrics, args.device, args.print_mm
    )
    soft_info = {"metrics": soft_metrics}

    proposal = pipeline._canonical_xyz_from_trainer_zyx(
        np.load(soft_dir / "occupancy_soft_proposal.npy").astype(bool)
    )
    probability = pipeline._canonical_xyz_from_trainer_zyx(
        np.load(soft_dir / "occupancy_probability_core.npy").astype(np.float32)
    )
    targets = pipeline._generic_targets(soft_dir, int(proposal.shape[0]))
    expected_views = len(silhouettes) * (2 if protocol.mr else 1)
    if len(targets) != expected_views:
        raise RuntimeError(
            f"Artifact contains {len(targets)} targets, protocol declares "
            f"{expected_views} views."
        )

    reference_voxels = int(proposal.sum())
    raw_metrics = prescreen(
        proposal, targets, reference_voxels, protocol, soft_info
    )
    current, component_audit = largest_component(proposal)
    current_metrics = prescreen(
        current, targets, reference_voxels, protocol, soft_info
    )
    print(
        json.dumps(
            {
                "stage": "v256_main_component",
                **component_audit,
                **current_metrics,
            }
        ),
        flush=True,
    )

    repair_audit: dict | None = None
    if not gate_pass(current_metrics, args) and args.allow_bounded_repair:
        repaired, repair_audit = attempt_bounded_repair(
            current,
            proposal,
            probability,
            targets,
            protocol,
            soft_info,
            args,
        )
        repaired_metrics = prescreen(
            repaired, targets, reference_voxels, protocol, soft_info
        )
        print(
            json.dumps(
                {
                    "stage": "bounded_projection_repair",
                    **repair_audit,
                    **repaired_metrics,
                }
            ),
            flush=True,
        )
        if (
            repaired_metrics["components"] == 1
            and repaired_metrics["mean_iou"] >= current_metrics["mean_iou"]
        ):
            current, current_metrics = repaired, repaired_metrics

    trajectory = [
        {
            "state": "V0",
            "accepted": bool(gate_pass(current_metrics, args)),
            **current_metrics,
        }
    ]
    for checkpoint in range(1, max(args.carve_checkpoints, 0) + 1):
        if args.carve_mode == "target_rays":
            candidate, removed = target_ray_guarded_carve_step(
                current,
                priority_xyz=probability,
                targets=targets,
                soft_info=soft_info,
                max_removals=args.carve_voxels,
            )
        else:
            candidate, removed = pipeline.axis_guarded_carve_step(
                current,
                priority=probability,
                num_axes=min(len(silhouettes), 3),
                max_removals=args.carve_voxels,
            )
        if removed == 0:
            break
        candidate, candidate_components = largest_component(candidate)
        candidate_metrics = prescreen(
            candidate, targets, reference_voxels, protocol, soft_info
        )
        accepted = gate_pass(candidate_metrics, args)
        row = {
            "state": f"V{checkpoint}",
            "removed_voxels": int(removed),
            "accepted": bool(accepted),
            "component_audit": candidate_components,
            **candidate_metrics,
        }
        trajectory.append(row)
        print(json.dumps({"stage": "guarded_checkpoint", **row}), flush=True)
        if not accepted:
            break
        current, current_metrics = candidate, candidate_metrics

    # The accepted carving state is not yet a deliverable. Search a small,
    # explicit reconstruction family and re-evaluate every reconstructed volume.
    # This prevents a connected grid envelope from being mistaken for a clean
    # external shell merely because its pre-SDF projections pass.
    np.save(
        output / "accepted_carved_volume_256.npy",
        current.astype(np.uint8),
    )
    reconstruction_search: list[dict] = []
    reconstruction_candidates: list[
        tuple[
            tuple[float, float, float, float],
            np.ndarray,
            trimesh.Trimesh,
            dict,
            dict,
        ]
    ] = []
    diagnostic_candidates: list[
        tuple[
            tuple[float, float, float, float],
            np.ndarray,
            trimesh.Trimesh,
            dict,
            dict,
        ]
    ] = []
    sigmas = parse_number_list(args.sdf_sigmas, float)
    dilations = parse_number_list(args.sdf_dilations, int)
    for dilation in dilations:
        for smooth_sigma in sigmas:
            label = f"d{dilation}_s{smooth_sigma:g}"
            try:
                reconstructed, reconstruction_audit = reconstruct_sdf_volume(
                    current,
                    dilation=dilation,
                    smooth_sigma=smooth_sigma,
                    level=args.sdf_level,
                )
                reconstructed_metrics = prescreen(
                    reconstructed,
                    targets,
                    reference_voxels,
                    protocol,
                    soft_info,
                )
                exterior, shell_audit = select_external_shell(reconstructed)
                regularized_mesh = exterior
                regularization_audit = {
                    "simplification_applied": False,
                    "source_triangles": int(len(exterior.faces)),
                    "target_triangles": int(args.max_mesh_triangles),
                    "result_triangles": int(len(exterior.faces)),
                }
                exported_volume = reconstructed
                exported_metrics = reconstructed_metrics
                exported_component_count = int(
                    reconstructed_metrics["components"]
                )
                exported_mesh_audit = in_memory_mesh_audit(regularized_mesh)

                # Only spend the expensive mesh audit on candidates that first
                # pass the projection gate. The final decision is made from the
                # regularized shell re-rasterized into the recorded protocol,
                # never from the pre-export binary volume alone.
                if gate_pass(reconstructed_metrics, args):
                    regularized_mesh, regularization_audit = (
                        simplify_external_shell(
                            exterior,
                            target_faces=args.max_mesh_triangles,
                        )
                    )
                    exported_mesh_audit = in_memory_mesh_audit(
                        regularized_mesh
                    )
                    exported_volume = mesh_to_voxel_volume(
                        regularized_mesh,
                        shape=tuple(map(int, reconstructed.shape)),
                    )
                    _, exported_component_count = ndimage.label(
                        exported_volume, structure=S6
                    )
                    exported_metrics = prescreen(
                        exported_volume,
                        targets,
                        reference_voxels,
                        protocol,
                        soft_info,
                    )

                mesh_clean = bool(
                    surface_fraction(exported_volume)
                    <= args.max_surface_fraction
                    and exported_mesh_audit["triangles"]
                    <= args.max_mesh_triangles
                    and shell_audit["exterior_shell_fraction"]
                    >= args.min_exterior_shell_fraction
                    and exported_mesh_audit["components"] == 1
                    and exported_mesh_audit["watertight"]
                    and exported_mesh_audit["winding_consistent"]
                    and exported_component_count == 1
                )
                accepted = bool(
                    gate_pass(exported_metrics, args) and mesh_clean
                )
                record = {
                    "candidate": label,
                    "accepted": accepted,
                    "mesh_clean": mesh_clean,
                    "projection_gate": bool(
                        gate_pass(reconstructed_metrics, args)
                    ),
                    **reconstruction_audit,
                    **reconstructed_metrics,
                    "shell": shell_audit,
                    "mesh_regularization": regularization_audit,
                    "exported_mesh": exported_mesh_audit,
                    "exported_volume_components": int(
                        exported_component_count
                    ),
                    "exported_volume_surface_fraction": surface_fraction(
                        exported_volume
                    ),
                    "exported_volume_metrics": exported_metrics,
                }
                reconstruction_search.append(record)
                print(
                    json.dumps(
                        {"stage": "sdf_reconstruction_candidate", **record}
                    ),
                    flush=True,
                )
                accepted_score = (
                    float(exported_metrics["min_iou"]),
                    float(exported_metrics["mean_iou"]),
                    -float(surface_fraction(exported_volume)),
                    -float(exported_mesh_audit["triangles"]),
                )
                diagnostic_score = (
                    float(mesh_clean),
                    float(reconstructed_metrics["min_iou"]),
                    float(reconstructed_metrics["mean_iou"]),
                    -float(reconstruction_audit["surface_fraction"]),
                )
                item = (
                    accepted_score,
                    exported_volume,
                    regularized_mesh,
                    record,
                    shell_audit,
                )
                diagnostic_candidates.append(
                    (
                        diagnostic_score,
                        exported_volume,
                        regularized_mesh,
                        record,
                        shell_audit,
                    )
                )
                if accepted:
                    reconstruction_candidates.append(item)
            except Exception as error:
                record = {
                    "candidate": label,
                    "accepted": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                reconstruction_search.append(record)
                print(
                    json.dumps(
                        {"stage": "sdf_reconstruction_candidate", **record}
                    ),
                    flush=True,
                )

    if not reconstruction_candidates:
        diagnostic_path: Path | None = None
        diagnostic_record: dict | None = None
        if diagnostic_candidates:
            (
                _,
                diagnostic_volume,
                diagnostic_mesh,
                diagnostic_record,
                _,
            ) = max(diagnostic_candidates, key=lambda item: item[0])
            np.save(
                output / "diagnostic_failed_volume_256.npy",
                diagnostic_volume.astype(np.uint8),
            )
            diagnostic_path = (
                output / "diagnostic_failed_external_shell_NOT_SELECTED.stl"
            )
            scale_display_mesh(
                diagnostic_mesh, args.print_mm
            ).export(diagnostic_path)
        failure_manifest = {
            "status": "FAILED_RECONSTRUCTION_GATE",
            "experiment": "multiview_delivery_v2_isolated",
            "reason": (
                "No reconstructed candidate simultaneously passed final "
                "projection, exterior-shell, surface-complexity, and mesh gates."
            ),
            "camera_contract": soft_metrics.get("camera_protocol"),
            "base_view_count": int(len(silhouettes)),
            "evaluated_view_count": int(len(targets)),
            "trajectory": trajectory,
            "accepted_carved_volume": str(
                output / "accepted_carved_volume_256.npy"
            ),
            "reconstruction_search": reconstruction_search,
            "best_failed_candidate": diagnostic_record,
            "diagnostic_stl": (
                str(diagnostic_path) if diagnostic_path is not None else None
            ),
        }
        (output / "MANIFEST.json").write_text(
            json.dumps(failure_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "status": failure_manifest["status"],
                    "diagnostic_stl": failure_manifest["diagnostic_stl"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        raise SystemExit(2)

    (
        _,
        selected_volume,
        exterior,
        selected_reconstruction,
        shell_audit,
    ) = max(reconstruction_candidates, key=lambda item: item[0])
    selected_metrics = prescreen(
        selected_volume,
        targets,
        reference_voxels,
        protocol,
        soft_info,
    )
    qa = pipeline._generic_metrics(
        selected_volume,
        targets,
        reference_voxels,
        protocol,
        soft_info,
    )
    renders = qa.pop("renders")
    qa.pop("print_volume")

    np.save(
        output / "selected_volume_256.npy",
        selected_volume.astype(np.uint8),
    )
    core48 = pipeline.downsample_max_to(selected_volume, 48) >= 0.5
    np.save(output / "structural_core_48.npy", core48.astype(np.uint8))
    for index, (target, rendered) in enumerate(zip(targets, renders)):
        save_mask(output / f"target_view_{index}.png", target)
        save_mask(output / f"projection_view_{index}.png", rendered)

    display_mesh = scale_display_mesh(exterior, args.print_mm)
    display_path = output / "selected_display_external_shell.stl"
    display_mesh.export(display_path)
    display_reload = reload_mesh_audit(display_path)
    print_mesh = orient_print_mesh(display_mesh, qa["print_orientation"])
    print_path = output / f"selected_print_{qa['print_orientation']}.stl"
    print_mesh.export(print_path)
    print_reload = reload_mesh_audit(print_path)

    exported_mesh_pass = bool(
        display_reload["reloaded_components"] == 1
        and display_reload["reloaded_watertight"]
        and display_reload["reloaded_winding_consistent"]
        and display_reload["reloaded_triangles"]
        <= args.max_mesh_triangles
        and print_reload["reloaded_components"] == 1
        and print_reload["reloaded_watertight"]
        and print_reload["reloaded_winding_consistent"]
    )
    final_pass = bool(
        gate_pass(selected_metrics, args)
        and qa["components"] == 1
        and selected_reconstruction["mesh_clean"]
        and shell_audit["retained_shell_watertight"]
        and exported_mesh_pass
    )
    manifest = {
        "status": "SUCCESS" if final_pass else "FAILED_GATE",
        "experiment": "multiview_delivery_v2_isolated",
        "geometry_owner": "V256 frozen soft proposal and accepted V256 states",
        "structural_proxy": "C48 = adaptive max-pool(V256); never lifted back",
        "camera_contract": soft_metrics.get("camera_protocol"),
        "base_view_count": int(len(silhouettes)),
        "evaluated_view_count": int(len(targets)),
        "mirror_mode": int(protocol.mr),
        "silhouettes": [str(path) for path in silhouettes],
        "soft_artifact_dir": str(soft_dir),
        "gates": {
            "min_mean_iou": float(args.min_mean_iou),
            "min_view_iou": float(args.min_view_iou),
            "single_component": True,
            "single_watertight_external_shell": True,
            "max_surface_fraction": float(args.max_surface_fraction),
            "max_mesh_triangles": int(args.max_mesh_triangles),
            "min_exterior_shell_fraction": float(
                args.min_exterior_shell_fraction
            ),
        },
        "raw_soft": raw_metrics,
        "main_component": {
            **component_audit,
            **trajectory[0],
        },
        "bounded_repair": repair_audit,
        "trajectory": trajectory,
        "reconstruction_search": reconstruction_search,
        "selected_reconstruction": selected_reconstruction,
        "selected_volume_metrics": selected_metrics,
        "delivery_qa": qa,
        "external_shell": shell_audit,
        "exported_display_mesh": display_reload,
        "exported_print_mesh": print_reload,
        "exported_mesh_pass": exported_mesh_pass,
        "display_stl": str(display_path),
        "print_stl": str(print_path),
        "accepted_carved_volume": str(
            output / "accepted_carved_volume_256.npy"
        ),
        "selected_volume": str(output / "selected_volume_256.npy"),
        "structural_core": str(output / "structural_core_48.npy"),
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "status": manifest["status"],
                "mean_iou": selected_metrics["mean_iou"],
                "min_iou": selected_metrics["min_iou"],
                "components": qa["components"],
                "watertight_external_shell": shell_audit[
                    "retained_shell_watertight"
                ],
                "surface_fraction": selected_reconstruction[
                    "surface_fraction"
                ],
                "mesh_triangles": display_reload["reloaded_triangles"],
                "exported_mesh_pass": exported_mesh_pass,
                "display_stl": str(display_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not final_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
