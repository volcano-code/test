#!/usr/bin/env python3
"""Audit z-up COM/support-polygon stability for ordinary two-view deliveries.

The primary object is ``selected_display_fixed.stl`` in each case directory.
It is evaluated in its existing display coordinates with +z up; the script
never rotates a mesh to search for a favorable pose.

If both of the following optional assets exist, a separate rigid-assembly
audit is also performed:

``display_delivery/display_model_on_plinth.stl``
``display_delivery/display_plinth_separate.stl``

The assembly COM is mass weighted from the two closed solids, and its outer
tipping polygon is obtained only from the assembly's global ground-contact
band.  That large outer plinth polygon is deliberately *not* reused for either
the bare-model check or the model-side interface check.  The latter uses the
lowest band of the placed model and is only a geometric load-transfer proxy;
it does not certify friction, fasteners, clamps, or joint mechanics.

The frozen contact-band factors are 0.5h, h, and 2h, with h primary.  Unless
``--voxel-pitch-mm`` is supplied, h is the bare model's maximum extent divided
by ``--delivery-resolution``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import scipy
from scipy.spatial import ConvexHull, QhullError
import trimesh


SCHEMA = "shadowforge_ordinary_two_view_com_v1"
CONTACT_BAND_FACTORS = (0.5, 1.0, 2.0)
PRIMARY_CONTACT_BAND_FACTOR = 1.0
DEFAULT_DELIVERY_RESOLUTION = 256
DEFAULT_MODEL_DENSITY = 1.0
DEFAULT_PLINTH_DENSITY = 1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_output_paths(
    *,
    output_json: Path | None,
    output_dir: Path | None,
) -> dict[str, Path]:
    if (output_json is None) == (output_dir is None):
        raise ValueError("Specify exactly one of --output or --output-dir")
    if output_json is not None:
        summary = output_json.resolve()
        if summary.suffix.lower() != ".json":
            raise ValueError("--output must name a .json file")
        stem = summary.stem
        return {
            "summary": summary,
            "per_case": summary.with_name(f"{stem}_per_case.csv"),
            "per_band": summary.with_name(f"{stem}_per_band.csv"),
            "run_manifest": summary.with_name(f"{stem}_RUN_MANIFEST.json"),
        }
    assert output_dir is not None
    directory = output_dir.resolve()
    return {
        "summary": directory / "summary.json",
        "per_case": directory / "per_case.csv",
        "per_band": directory / "per_band.csv",
        "run_manifest": directory / "RUN_MANIFEST.json",
    }


def prepare_output_files(paths: dict[str, Path], overwrite: bool) -> None:
    parents = {path.parent for path in paths.values()}
    if len(parents) != 1:
        raise ValueError("All output files must share one parent directory")
    parent = next(iter(parents))
    if parent.exists() and not parent.is_dir():
        raise FileExistsError(f"Output parent is not a directory: {parent}")
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output files: "
            f"{[str(path) for path in existing]}. Pass --overwrite explicitly."
        )
    for path in existing:
        if not path.is_file():
            raise FileExistsError(f"Output target is not a regular file: {path}")
    parent.mkdir(parents=True, exist_ok=True)


def load_closed_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(
            f"Expected Trimesh, received {type(loaded).__name__}: {path}"
        )
    mesh = loaded
    mesh.remove_unreferenced_vertices()
    source_signed_volume = float(mesh.volume)
    inverted_for_analysis = False
    if (
        mesh.vertices.size
        and mesh.faces.size
        and mesh.is_watertight
        and mesh.is_winding_consistent
        and math.isfinite(source_signed_volume)
        and source_signed_volume < 0.0
    ):
        # Some legacy exporters produced consistently inward-facing closed
        # surfaces.  Reorient the in-memory analysis copy; never alter the STL.
        mesh.invert()
        inverted_for_analysis = True
    volume = float(mesh.volume)
    if (
        mesh.vertices.size == 0
        or mesh.faces.size == 0
        or not mesh.is_watertight
        or not mesh.is_winding_consistent
        or not mesh.is_volume
        or not math.isfinite(volume)
        or volume <= 0.0
    ):
        raise ValueError(
            "Mesh failed closed-volume QA: "
            f"watertight={mesh.is_watertight}, "
            f"winding={mesh.is_winding_consistent}, "
            f"is_volume={mesh.is_volume}, volume={volume}: {path}"
        )
    center_mass = np.asarray(mesh.center_mass, dtype=np.float64)
    if center_mass.shape != (3,) or not np.all(np.isfinite(center_mass)):
        raise ValueError(f"Mesh has a non-finite volume centroid: {path}")
    mesh.metadata["source_signed_volume_mm3"] = source_signed_volume
    mesh.metadata["inverted_winding_for_analysis"] = inverted_for_analysis
    return mesh


def mesh_record(path: Path, mesh: trimesh.Trimesh) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_signed_volume_mm3": mesh.metadata.get(
            "source_signed_volume_mm3"
        ),
        "inverted_winding_for_analysis": bool(
            mesh.metadata.get("inverted_winding_for_analysis", False)
        ),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "volume_mm3": float(mesh.volume),
        "surface_area_mm2": float(mesh.area),
        "bounds_mm": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "extents_mm": np.asarray(mesh.extents, dtype=np.float64).tolist(),
        "volume_centroid_mm": np.asarray(
            mesh.center_mass, dtype=np.float64
        ).tolist(),
    }


def numerical_tolerance_mm(max_extent_mm: float) -> float:
    return max(1e-6 * float(max_extent_mm), 1e-4)


def lower_band_points(
    mesh: trimesh.Trimesh,
    *,
    ground_z_mm: float,
    contact_band_mm: float,
    numerical_tolerance: float,
) -> np.ndarray:
    """Return XY points from a fixed global lower band and its upper slice.

    The explicit ``ground_z_mm`` is essential for assemblies: a model located
    above a plinth must not contribute its own local bottom as ground contact.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    plane_z = float(ground_z_mm + contact_band_mm)
    below = vertices[:, 2] <= plane_z + numerical_tolerance
    parts: list[np.ndarray] = []
    if np.any(below):
        parts.append(vertices[below, :2])

    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    start = vertices[edges[:, 0]]
    end = vertices[edges[:, 1]]
    dz_start = start[:, 2] - plane_z
    dz_end = end[:, 2] - plane_z
    crosses = (
        ((dz_start < -numerical_tolerance) & (dz_end > numerical_tolerance))
        | ((dz_end < -numerical_tolerance) & (dz_start > numerical_tolerance))
    )
    if np.any(crosses):
        a = start[crosses]
        b = end[crosses]
        alpha = (plane_z - a[:, 2]) / (b[:, 2] - a[:, 2])
        intersections = a[:, :2] + alpha[:, None] * (b[:, :2] - a[:, :2])
        parts.append(intersections)

    if not parts:
        return np.empty((0, 2), dtype=np.float64)
    points = np.vstack(parts)
    return np.unique(np.round(points, decimals=10), axis=0)


def combined_lower_band_points(
    meshes: Sequence[trimesh.Trimesh],
    *,
    ground_z_mm: float,
    contact_band_mm: float,
    numerical_tolerance: float,
) -> np.ndarray:
    parts = [
        lower_band_points(
            mesh,
            ground_z_mm=ground_z_mm,
            contact_band_mm=contact_band_mm,
            numerical_tolerance=numerical_tolerance,
        )
        for mesh in meshes
    ]
    nonempty = [part for part in parts if len(part)]
    if not nonempty:
        return np.empty((0, 2), dtype=np.float64)
    return np.unique(np.round(np.vstack(nonempty), decimals=10), axis=0)


def point_to_segments_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    vectors = ends - starts
    denominator = np.einsum("ij,ij->i", vectors, vectors)
    safe = denominator > np.finfo(np.float64).eps
    alpha = np.zeros(len(starts), dtype=np.float64)
    alpha[safe] = (
        np.einsum(
            "ij,ij->i",
            point[None, :] - starts[safe],
            vectors[safe],
        )
        / denominator[safe]
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    closest = starts + alpha[:, None] * vectors
    return float(np.linalg.norm(closest - point[None, :], axis=1).min())


def support_polygon_metrics(
    *,
    contact_points: np.ndarray,
    projected_com: np.ndarray,
    com_height_mm: float,
    max_extent_mm: float,
    numerical_tolerance: float,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "contact_point_count": int(len(contact_points)),
        "numerical_tolerance_mm": float(numerical_tolerance),
    }
    if len(contact_points) < 3:
        return {
            **common,
            "support_polygon_valid": False,
            "support_failure": "fewer_than_three_contact_points",
            "stability_class": "invalid_support",
            "stable_under_vertical_gravity": False,
        }
    centered = contact_points - contact_points.mean(axis=0)
    if int(np.linalg.matrix_rank(centered)) < 2:
        return {
            **common,
            "support_polygon_valid": False,
            "support_failure": "collinear_contact_points",
            "stability_class": "invalid_support",
            "stable_under_vertical_gravity": False,
        }
    try:
        hull = ConvexHull(contact_points)
    except QhullError as exc:
        return {
            **common,
            "support_polygon_valid": False,
            "support_failure": f"qhull_error:{type(exc).__name__}",
            "stability_class": "invalid_support",
            "stable_under_vertical_gravity": False,
        }

    polygon = contact_points[hull.vertices]
    halfspace = (
        hull.equations[:, :2] @ projected_com + hull.equations[:, 2]
    )
    inside_or_boundary = bool(np.all(halfspace <= numerical_tolerance))
    boundary_distance = point_to_segments_distance(projected_com, polygon)
    signed_margin = boundary_distance if inside_or_boundary else -boundary_distance
    if signed_margin > numerical_tolerance:
        stability_class = "stable"
    elif signed_margin < -numerical_tolerance:
        stability_class = "unstable"
    else:
        stability_class = "boundary_sensitive"

    support_diagonal = float(np.linalg.norm(np.ptp(polygon, axis=0)))
    support_area = float(hull.volume)
    equivalent_radius = float(math.sqrt(support_area / math.pi))
    signed_angle = (
        float(math.degrees(math.atan2(signed_margin, com_height_mm)))
        if com_height_mm > numerical_tolerance
        else None
    )
    return {
        **common,
        "support_polygon_valid": True,
        "support_failure": "",
        "support_vertex_count": int(len(polygon)),
        "support_area_mm2": support_area,
        "support_diagonal_mm": support_diagonal,
        "support_equivalent_radius_mm": equivalent_radius,
        "com_projection_inside_or_boundary": inside_or_boundary,
        "signed_margin_mm": float(signed_margin),
        "normalized_margin_max_extent": (
            float(signed_margin / max_extent_mm)
            if max_extent_mm > numerical_tolerance
            else None
        ),
        "normalized_margin_support_diagonal": (
            float(signed_margin / support_diagonal)
            if support_diagonal > numerical_tolerance
            else None
        ),
        "normalized_margin_equivalent_radius": (
            float(signed_margin / equivalent_radius)
            if equivalent_radius > numerical_tolerance
            else None
        ),
        "com_height_above_support_mm": float(com_height_mm),
        "signed_stability_angle_deg": signed_angle,
        "critical_tilt_angle_deg": (
            signed_angle if stability_class == "stable" else None
        ),
        "stability_class": stability_class,
        "stable_under_vertical_gravity": stability_class == "stable",
    }


def analyze_support(
    *,
    case_id: str,
    role: str,
    support_source: str,
    meshes_for_contact: Sequence[trimesh.Trimesh],
    center_mass: np.ndarray,
    ground_z_mm: float,
    max_extent_mm: float,
    h_mm: float,
) -> list[dict[str, Any]]:
    tolerance = numerical_tolerance_mm(max_extent_mm)
    projected_com = np.asarray(center_mass[:2], dtype=np.float64)
    com_height = float(center_mass[2] - ground_z_mm)
    rows: list[dict[str, Any]] = []
    for factor in CONTACT_BAND_FACTORS:
        band = float(factor * h_mm)
        contact_points = combined_lower_band_points(
            meshes_for_contact,
            ground_z_mm=ground_z_mm,
            contact_band_mm=band,
            numerical_tolerance=tolerance,
        )
        metrics = support_polygon_metrics(
            contact_points=contact_points,
            projected_com=projected_com,
            com_height_mm=com_height,
            max_extent_mm=max_extent_mm,
            numerical_tolerance=tolerance,
        )
        rows.append(
            {
                "case_id": case_id,
                "analysis_role": role,
                "support_source": support_source,
                "pose": "fixed_display_z_up",
                "gravity_axis": "-z",
                "density_model": "piecewise_uniform_closed_solids",
                "h_mm": float(h_mm),
                "contact_band_factor_h": float(factor),
                "contact_band_mm": band,
                "is_primary_contact_band": math.isclose(
                    factor,
                    PRIMARY_CONTACT_BAND_FACTOR,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "ground_z_mm": float(ground_z_mm),
                "com_x_mm": float(center_mass[0]),
                "com_y_mm": float(center_mass[1]),
                "com_z_mm": float(center_mass[2]),
                **metrics,
            }
        )
    return rows


def primary_row(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    primary = [row for row in rows if row["is_primary_contact_band"]]
    if len(primary) != 1:
        raise RuntimeError(f"Expected one primary row, received {len(primary)}")
    return primary[0]


def primary_projection(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    keys = (
        "support_polygon_valid",
        "support_failure",
        "stability_class",
        "stable_under_vertical_gravity",
        "signed_margin_mm",
        "normalized_margin_max_extent",
        "normalized_margin_support_diagonal",
        "signed_stability_angle_deg",
        "critical_tilt_angle_deg",
        "support_area_mm2",
        "support_diagonal_mm",
        "contact_band_mm",
        "numerical_tolerance_mm",
    )
    return {f"{prefix}_{key}": row.get(key) for key in keys}


def resolve_delivery_assets(case_dir: Path) -> tuple[Path | None, Path | None]:
    delivery = case_dir / "display_delivery"
    model = delivery / "display_model_on_plinth.stl"
    plinth = delivery / "display_plinth_separate.stl"
    present = (model.is_file(), plinth.is_file())
    if present == (False, False):
        return None, None
    if present != (True, True):
        raise FileNotFoundError(
            "Display delivery is incomplete; both assets are required or both "
            f"must be absent: model={model.is_file()}, plinth={plinth.is_file()}, "
            f"case={case_dir}"
        )
    return model.resolve(), plinth.resolve()


def analyze_case(
    *,
    case_dir: Path,
    case_id: str,
    delivery_resolution: int,
    voxel_pitch_mm: float | None,
    model_density: float,
    plinth_density: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    bare_path = (case_dir / "selected_display_fixed.stl").resolve()
    if not bare_path.is_file():
        raise FileNotFoundError(
            f"Missing required bare display mesh: {bare_path}"
        )
    bare = load_closed_mesh(bare_path)
    bare_max_extent = float(np.max(bare.extents))
    h_mm = (
        float(voxel_pitch_mm)
        if voxel_pitch_mm is not None
        else float(bare_max_extent / delivery_resolution)
    )
    if not math.isfinite(h_mm) or h_mm <= 0.0:
        raise ValueError(f"Invalid voxel pitch h={h_mm} for {case_id}")

    bare_ground = float(bare.bounds[0, 2])
    bare_rows = analyze_support(
        case_id=case_id,
        role="bare_model",
        support_source="bare_model_lower_band",
        meshes_for_contact=(bare,),
        center_mass=np.asarray(bare.center_mass, dtype=np.float64),
        ground_z_mm=bare_ground,
        max_extent_mm=bare_max_extent,
        h_mm=h_mm,
    )
    all_rows = list(bare_rows)
    bare_primary = primary_row(bare_rows)
    case_record: dict[str, Any] = {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "pose": "fixed_display_z_up",
        "gravity_axis": "-z",
        "delivery_resolution": int(delivery_resolution),
        "h_mm": h_mm,
        "h_source": (
            "explicit_voxel_pitch_mm"
            if voxel_pitch_mm is not None
            else "bare_model_max_extent/delivery_resolution"
        ),
        "bare_model": {
            "mesh": mesh_record(bare_path, bare),
            "support_definition": "bare_model_lower_band",
            "primary": bare_primary,
            "sensitivity": bare_rows,
        },
        "display_delivery_available": False,
        "outer_support_substitutes_for_bare_or_interface": False,
    }
    flat_record: dict[str, Any] = {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "h_mm": h_mm,
        "h_source": case_record["h_source"],
        "display_delivery_available": False,
        **primary_projection(bare_primary, "bare"),
    }

    placed_path, plinth_path = resolve_delivery_assets(case_dir)
    if placed_path is None or plinth_path is None:
        return case_record, all_rows, flat_record

    placed = load_closed_mesh(placed_path)
    plinth = load_closed_mesh(plinth_path)
    placed_volume = float(placed.volume)
    plinth_volume = float(plinth.volume)
    model_mass = float(model_density * placed_volume)
    plinth_mass = float(plinth_density * plinth_volume)
    total_mass = model_mass + plinth_mass
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError(f"Invalid assembly mass for {case_id}: {total_mass}")
    placed_com = np.asarray(placed.center_mass, dtype=np.float64)
    plinth_com = np.asarray(plinth.center_mass, dtype=np.float64)
    assembly_com = (
        model_mass * placed_com + plinth_mass * plinth_com
    ) / total_mass
    assembly_bounds = np.vstack((placed.bounds, plinth.bounds))
    assembly_min = np.min(assembly_bounds, axis=0)
    assembly_max = np.max(assembly_bounds, axis=0)
    assembly_extents = assembly_max - assembly_min
    assembly_max_extent = float(np.max(assembly_extents))
    assembly_ground = float(assembly_min[2])

    interface_ground = float(placed.bounds[0, 2])
    interface_rows = analyze_support(
        case_id=case_id,
        role="model_to_plinth_interface_proxy",
        support_source="placed_model_lower_band_not_plinth_outer_envelope",
        meshes_for_contact=(placed,),
        center_mass=placed_com,
        ground_z_mm=interface_ground,
        max_extent_mm=float(np.max(placed.extents)),
        h_mm=h_mm,
    )
    outer_rows = analyze_support(
        case_id=case_id,
        role="rigid_assembly_outer_tipping",
        support_source="global_ground_band_of_model_plus_plinth",
        meshes_for_contact=(placed, plinth),
        center_mass=assembly_com,
        ground_z_mm=assembly_ground,
        max_extent_mm=assembly_max_extent,
        h_mm=h_mm,
    )
    all_rows.extend(interface_rows)
    all_rows.extend(outer_rows)
    interface_primary = primary_row(interface_rows)
    outer_primary = primary_row(outer_rows)

    volume_rel_diff = abs(placed_volume - float(bare.volume)) / float(bare.volume)
    extents_rel_diff = float(
        np.max(
            np.abs(np.asarray(placed.extents) - np.asarray(bare.extents))
            / np.maximum(np.asarray(bare.extents), 1e-12)
        )
    )
    placed_consistent = volume_rel_diff <= 1e-5 and extents_rel_diff <= 1e-5
    two_stage_pass = bool(
        interface_primary.get("stable_under_vertical_gravity", False)
        and outer_primary.get("stable_under_vertical_gravity", False)
    )
    case_record["display_delivery_available"] = True
    case_record["display_delivery"] = {
        "model_on_plinth_mesh": mesh_record(placed_path, placed),
        "plinth_mesh": mesh_record(plinth_path, plinth),
        "model_density_relative": float(model_density),
        "plinth_density_relative": float(plinth_density),
        "model_mass_relative": model_mass,
        "plinth_mass_relative": plinth_mass,
        "assembly_mass_relative": total_mass,
        "mass_weighted_assembly_com_mm": assembly_com.tolist(),
        "assembly_bounds_mm": [assembly_min.tolist(), assembly_max.tolist()],
        "assembly_extents_mm": assembly_extents.tolist(),
        "placed_model_rigid_geometry_consistency": {
            "volume_relative_difference_from_bare": volume_rel_diff,
            "extents_max_relative_difference_from_bare": extents_rel_diff,
            "consistent_with_rigid_translation": placed_consistent,
        },
        "interface_proxy": {
            "definition": (
                "placed model COM against the placed model's own lowest-band "
                "support polygon; the plinth outer envelope is excluded"
            ),
            "joint_mechanics_certified": False,
            "primary": interface_primary,
            "sensitivity": interface_rows,
        },
        "rigid_assembly_outer_tipping": {
            "definition": (
                "mass-weighted model-plus-plinth COM against the global "
                "ground-contact support polygon"
            ),
            "primary": outer_primary,
            "sensitivity": outer_rows,
        },
        "two_stage_geometric_pass": two_stage_pass,
        "two_stage_definition": (
            "interface proxy and rigid-assembly outer tipping both stable; "
            "does not certify slip or joint mechanics"
        ),
        "outer_support_substitutes_for_interface": False,
    }
    flat_record.update(
        {
            "display_delivery_available": True,
            "placed_model_consistent_with_rigid_translation": placed_consistent,
            "model_density_relative": float(model_density),
            "plinth_density_relative": float(plinth_density),
            "assembly_com_x_mm": float(assembly_com[0]),
            "assembly_com_y_mm": float(assembly_com[1]),
            "assembly_com_z_mm": float(assembly_com[2]),
            "two_stage_geometric_pass": two_stage_pass,
            **primary_projection(interface_primary, "interface"),
            **primary_projection(outer_primary, "assembly_outer"),
        }
    )
    return case_record, all_rows, flat_record


def unique_case_ids(case_dirs: Sequence[Path]) -> list[str]:
    ids = [path.name for path in case_dirs]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(
            "Case directory basenames must be unique; duplicate case IDs: "
            f"{duplicates}"
        )
    return ids


def discover_case_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    if (root / "selected_display_fixed.stl").is_file():
        return [root]
    case_dirs = sorted(
        path.parent.resolve()
        for path in root.glob("*/selected_display_fixed.stl")
        if path.is_file()
    )
    if not case_dirs:
        raise FileNotFoundError(
            "No immediate case subdirectories containing "
            f"selected_display_fixed.stl were found under: {root}"
        )
    return case_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--case-dir",
        type=Path,
        action="append",
        help=(
            "Case directory containing selected_display_fixed.stl; repeat for "
            "multiple fixed tasks."
        ),
    )
    inputs.add_argument(
        "--root",
        type=Path,
        help=(
            "Root whose immediate case subdirectories contain "
            "selected_display_fixed.stl; cases are discovered in sorted order."
        ),
    )
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument(
        "--output",
        type=Path,
        help=(
            "Summary JSON path. CSVs and the run manifest are written beside "
            "it using the same filename stem."
        ),
    )
    outputs.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory-mode compatibility interface: writes summary.json, "
            "per_case.csv, per_band.csv, and RUN_MANIFEST.json."
        ),
    )
    parser.add_argument(
        "--delivery-resolution",
        type=int,
        default=DEFAULT_DELIVERY_RESOLUTION,
        help="Resolution used to derive h from the bare mesh maximum extent.",
    )
    parser.add_argument(
        "--voxel-pitch-mm",
        type=float,
        default=None,
        help="Explicit h shared by all cases; otherwise derive h per case.",
    )
    parser.add_argument(
        "--model-density",
        type=float,
        default=DEFAULT_MODEL_DENSITY,
        help="Relative uniform density of the model for assembly COM.",
    )
    parser.add_argument(
        "--plinth-density",
        type=float,
        default=DEFAULT_PLINTH_DENSITY,
        help="Relative uniform density of the plinth for assembly COM.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Explicitly replace this invocation's JSON/CSV/manifest targets; "
            "unrelated neighboring files are never removed."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.delivery_resolution <= 1:
        raise ValueError("--delivery-resolution must exceed 1")
    if args.voxel_pitch_mm is not None and args.voxel_pitch_mm <= 0.0:
        raise ValueError("--voxel-pitch-mm must be positive")
    if args.model_density <= 0.0 or args.plinth_density <= 0.0:
        raise ValueError("Both relative densities must be positive")

    case_dirs = (
        discover_case_dirs(args.root)
        if args.root is not None
        else [path.resolve() for path in args.case_dir]
    )
    for case_dir in case_dirs:
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory does not exist: {case_dir}")
    case_ids = unique_case_ids(case_dirs)
    output_paths = resolve_output_paths(
        output_json=args.output,
        output_dir=args.output_dir,
    )
    prepare_output_files(output_paths, bool(args.overwrite))

    run_manifest_path = output_paths["run_manifest"]
    run_record: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "command": " ".join(sys.argv),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "case_dirs": [str(path) for path in case_dirs],
        "output_paths": {
            name: str(path) for name, path in output_paths.items()
        },
        "protocol": {
            "pose": "fixed_display_z_up",
            "gravity_axis": "-z",
            "com": "closed-solid volume centroid",
            "contact_band_factors_h": list(CONTACT_BAND_FACTORS),
            "primary_contact_band_factor_h": PRIMARY_CONTACT_BAND_FACTOR,
            "delivery_resolution": int(args.delivery_resolution),
            "voxel_pitch_mm": args.voxel_pitch_mm,
            "numerical_tolerance": "max(1e-6 * max_extent_mm, 1e-4 mm)",
            "stable_rule": "signed_margin_mm > numerical_tolerance_mm",
            "boundary_rule": (
                "abs(signed_margin_mm) <= numerical_tolerance_mm is "
                "boundary_sensitive, not stable"
            ),
            "assembly_com": "relative-density-times-volume mass weighting",
            "model_density_relative": float(args.model_density),
            "plinth_density_relative": float(args.plinth_density),
            "outer_support_substitutes_for_bare_or_interface": False,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "trimesh": trimesh.__version__,
        },
    }
    write_json(run_manifest_path, run_record)

    try:
        cases: list[dict[str, Any]] = []
        per_band_rows: list[dict[str, Any]] = []
        per_case_rows: list[dict[str, Any]] = []
        for index, (case_id, case_dir) in enumerate(
            zip(case_ids, case_dirs, strict=True),
            start=1,
        ):
            print(
                f"[{index:02d}/{len(case_dirs):02d}] "
                f"{case_id}: fixed display z-up",
                flush=True,
            )
            case_record, band_rows, flat_record = analyze_case(
                case_dir=case_dir,
                case_id=case_id,
                delivery_resolution=int(args.delivery_resolution),
                voxel_pitch_mm=args.voxel_pitch_mm,
                model_density=float(args.model_density),
                plinth_density=float(args.plinth_density),
            )
            cases.append(case_record)
            per_band_rows.extend(band_rows)
            per_case_rows.append(flat_record)

        summary = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "claim_scope": (
                "Digital geometric tipping stability in the fixed z-up display "
                "pose under vertical gravity and the declared density model."
            ),
            "not_certified": [
                "friction or sliding",
                "model-to-plinth joint mechanics",
                "fasteners, clamps, or adhesive strength",
                "fracture or yielding",
                "manufacturing defects",
                "physical printed-part stability",
            ],
            "protocol": run_record["protocol"],
            "n_cases": len(cases),
            "n_cases_with_display_delivery": int(
                sum(case["display_delivery_available"] for case in cases)
            ),
            "n_bare_primary_stable": int(
                sum(
                    case["bare_model"]["primary"][
                        "stable_under_vertical_gravity"
                    ]
                    for case in cases
                )
            ),
            "n_outer_assembly_primary_stable": int(
                sum(
                    case.get("display_delivery", {})
                    .get("rigid_assembly_outer_tipping", {})
                    .get("primary", {})
                    .get("stable_under_vertical_gravity", False)
                    for case in cases
                )
            ),
            "n_interface_proxy_primary_stable": int(
                sum(
                    case.get("display_delivery", {})
                    .get("interface_proxy", {})
                    .get("primary", {})
                    .get("stable_under_vertical_gravity", False)
                    for case in cases
                )
            ),
            "n_two_stage_geometric_pass": int(
                sum(
                    case.get("display_delivery", {}).get(
                        "two_stage_geometric_pass", False
                    )
                    for case in cases
                )
            ),
            "cases": cases,
        }
        write_json(output_paths["summary"], summary)
        write_csv(output_paths["per_case"], per_case_rows)
        write_csv(output_paths["per_band"], per_band_rows)
        run_record.update(
            {
                "status": "COMPLETE",
                "completed_at_utc": utc_now(),
                "outputs": {
                    "summary": str(output_paths["summary"]),
                    "per_case": str(output_paths["per_case"]),
                    "per_band": str(output_paths["per_band"]),
                },
                "counts": {
                    "cases": len(cases),
                    "per_case_rows": len(per_case_rows),
                    "per_band_rows": len(per_band_rows),
                },
            }
        )
        write_json(run_manifest_path, run_record)
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "output": str(output_paths["summary"]),
                    "per_case_csv": str(output_paths["per_case"]),
                    "per_band_csv": str(output_paths["per_band"]),
                    "n_cases": summary["n_cases"],
                    "n_cases_with_display_delivery": summary[
                        "n_cases_with_display_delivery"
                    ],
                    "n_bare_primary_stable": summary[
                        "n_bare_primary_stable"
                    ],
                    "n_outer_assembly_primary_stable": summary[
                        "n_outer_assembly_primary_stable"
                    ],
                    "n_interface_proxy_primary_stable": summary[
                        "n_interface_proxy_primary_stable"
                    ],
                    "n_two_stage_geometric_pass": summary[
                        "n_two_stage_geometric_pass"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        run_record.update(
            {
                "status": "FAILED",
                "completed_at_utc": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(run_manifest_path, run_record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
