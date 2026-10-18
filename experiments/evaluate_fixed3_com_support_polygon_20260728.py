#!/usr/bin/env python3
"""Uniform-density COM/support-polygon audit for the frozen connected-10 set.

The primary geometry is the fixed-display STL referenced by each run's
MANIFEST.json because placement stability concerns the pose in which the
sculpture realizes its prescribed shadows.  The print-oriented STL is audited
separately as a build-pose diagnostic.  Each closed mesh is treated as a
homogeneous solid; its volume centroid is projected onto the horizontal plane
and measured against the convex hull of a finite bottom contact band.

This is a digital rigid-body tipping audit under vertical gravity.  It is not a
fracture, sliding, print-process, or physical-test certification.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.spatial import ConvexHull, QhullError
import trimesh


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK = (
    ROOT
    / "results/fixed3_mr1_connected10_pending_20260722/benchmark_manifest.json"
)
DEFAULT_COMPARISON = (
    ROOT / "results/fixed3_mr1_connected10_comparison_20260722/table1_real.json"
)
DEFAULT_RUNS = ROOT / "results/fixed3_mr1_connected10_pcraft_20260722"
DEFAULT_OUTPUT = ROOT / "results/fixed3_mr1_connected10_com_audit_dualpose_20260728"
DEFAULT_CONTACT_FRACTIONS = (0.0025, 0.005, 0.01)
DEFAULT_PRIMARY_FRACTION = 0.005
DEFAULT_POSES = ("display", "print")
DEFAULT_PRIMARY_POSE = "display"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_geometry_sha256(mesh: trimesh.Trimesh) -> str:
    """Hash geometry independent of STL facet and vertex ordering."""
    vertices = np.round(np.asarray(mesh.vertices, dtype=np.float64), decimals=8)
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    canonical_vertices = np.ascontiguousarray(vertices[order], dtype="<f8")
    canonical_faces = np.sort(inverse[np.asarray(mesh.faces, dtype=np.int64)], axis=1)
    face_order = np.lexsort(
        (canonical_faces[:, 2], canonical_faces[:, 1], canonical_faces[:, 0])
    )
    canonical_faces = np.ascontiguousarray(canonical_faces[face_order], dtype="<i8")
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical_vertices.shape, dtype="<i8").tobytes())
    digest.update(canonical_vertices.tobytes())
    digest.update(np.asarray(canonical_faces.shape, dtype="<i8").tobytes())
    digest.update(canonical_faces.tobytes())
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
                seen.add(key)
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_fractions(text: str) -> tuple[float, ...]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values or any(value <= 0.0 or value >= 0.1 for value in values):
        raise argparse.ArgumentTypeError(
            "Contact-band fractions must be comma-separated values in (0, 0.1)"
        )
    return tuple(values)


def parse_poses(text: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(item.strip() for item in text.split(",") if item.strip()))
    if not values or any(value not in {"display", "print"} for value in values):
        raise argparse.ArgumentTypeError(
            "Poses must be a comma-separated subset of: display,print"
        )
    return values


def resolve_geometry(
    manifest_path: Path, payload: dict[str, Any], pose: str
) -> Path:
    field = "display_stl" if pose == "display" else "print_stl"
    raw = payload.get(field)
    if not raw:
        raise ValueError(f"Missing {field} in {manifest_path}")
    path = Path(raw)
    if not path.is_file():
        local = manifest_path.parent / path.name
        if local.is_file():
            path = local
    if not path.is_file():
        raise FileNotFoundError(f"Manifest {field} does not exist: {path}")

    orientation = payload.get("selected", {}).get("print_orientation")
    expected_name = (
        "selected_display_fixed.stl"
        if pose == "display"
        else f"selected_print_{orientation}.stl"
    )
    if (pose == "print" and not orientation) or path.name != expected_name:
        raise ValueError(
            f"{pose}-geometry mismatch in {manifest_path}: "
            f"orientation={orientation!r}, file={path.name!r}, "
            f"expected={expected_name!r}"
        )
    resolved = path.resolve()
    if resolved.parent != manifest_path.parent.resolve():
        raise ValueError(
            f"Geometry path escapes its run directory: {resolved} "
            f"(manifest {manifest_path})"
        )
    return resolved


def load_closed_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, received {type(loaded).__name__}: {path}")
    mesh = loaded
    mesh.remove_unreferenced_vertices()
    if (
        mesh.vertices.size == 0
        or mesh.faces.size == 0
        or not mesh.is_watertight
        or not mesh.is_winding_consistent
        or not mesh.is_volume
        or not np.isfinite(mesh.volume)
        or mesh.volume <= 0.0
    ):
        raise ValueError(
            "Mesh failed closed-volume QA: "
            f"watertight={mesh.is_watertight}, "
            f"winding={mesh.is_winding_consistent}, "
            f"is_volume={mesh.is_volume}, volume={mesh.volume}: {path}"
        )
    return mesh


def lower_contact_band_points(
    mesh: trimesh.Trimesh, epsilon_mm: float
) -> np.ndarray:
    """Project the lower epsilon band, including edge/plane intersections."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    z_min = float(vertices[:, 2].min())
    plane_z = z_min + float(epsilon_mm)
    numerical_tol = max(float(mesh.extents.max()) * 1e-10, 1e-10)

    points = [vertices[vertices[:, 2] <= plane_z + numerical_tol, :2]]
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    start = vertices[edges[:, 0]]
    end = vertices[edges[:, 1]]
    dz_start = start[:, 2] - plane_z
    dz_end = end[:, 2] - plane_z
    crosses = (
        ((dz_start < -numerical_tol) & (dz_end > numerical_tol))
        | ((dz_end < -numerical_tol) & (dz_start > numerical_tol))
    )
    if np.any(crosses):
        a = start[crosses]
        b = end[crosses]
        alpha = (plane_z - a[:, 2]) / (b[:, 2] - a[:, 2])
        intersections = a[:, :2] + alpha[:, None] * (b[:, :2] - a[:, :2])
        points.append(intersections)

    combined = np.vstack([part for part in points if len(part)])
    # Rounding removes duplicated STL vertices without changing any reported
    # distance at the scale of the experiment.
    return np.unique(np.round(combined, decimals=10), axis=0)


def point_to_segments_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    vectors = ends - starts
    denominator = np.einsum("ij,ij->i", vectors, vectors)
    safe = denominator > np.finfo(np.float64).eps
    alpha = np.zeros(len(starts), dtype=np.float64)
    alpha[safe] = (
        np.einsum("ij,ij->i", point[None, :] - starts[safe], vectors[safe])
        / denominator[safe]
    )
    alpha = np.clip(alpha, 0.0, 1.0)
    closest = starts + alpha[:, None] * vectors
    return float(np.linalg.norm(closest - point[None, :], axis=1).min())


def support_polygon_metrics(
    contact_points: np.ndarray,
    projected_com: np.ndarray,
    scale_mm: float,
) -> dict[str, Any]:
    if len(contact_points) < 3:
        return {
            "support_polygon_valid": False,
            "support_failure": "fewer_than_three_contact_points",
        }
    centered = contact_points - contact_points.mean(axis=0)
    if int(np.linalg.matrix_rank(centered)) < 2:
        return {
            "support_polygon_valid": False,
            "support_failure": "collinear_contact_points",
        }
    try:
        hull = ConvexHull(contact_points)
    except QhullError as exc:
        return {
            "support_polygon_valid": False,
            "support_failure": f"qhull_error:{type(exc).__name__}",
        }

    polygon = contact_points[hull.vertices]
    boundary_distance = point_to_segments_distance(projected_com, polygon)
    tolerance = max(scale_mm * 1e-10, 1e-10)
    halfspace = (
        hull.equations[:, :2] @ projected_com + hull.equations[:, 2]
    )
    inside = bool(np.all(halfspace <= tolerance))
    signed_margin = boundary_distance if inside else -boundary_distance
    diagonal = float(np.linalg.norm(np.ptp(polygon, axis=0)))
    equivalent_radius = float(math.sqrt(float(hull.volume) / math.pi))
    return {
        "support_polygon_valid": True,
        "support_failure": "",
        "support_vertex_count": int(len(polygon)),
        "support_area_mm2": float(hull.volume),
        "support_diagonal_mm": diagonal,
        "support_equivalent_radius_mm": equivalent_radius,
        "com_projection_inside": inside,
        "signed_margin_mm": signed_margin,
        "normalized_margin_diagonal": (
            signed_margin / diagonal if diagonal > tolerance else None
        ),
        "normalized_margin_equivalent_radius": (
            signed_margin / equivalent_radius
            if equivalent_radius > tolerance
            else None
        ),
    }


def mean_std(values: Iterable[float]) -> tuple[float | None, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return None, None
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def bootstrap_mean_ci(
    values: list[float], samples: int, seed: int
) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return [float(low), float(high)]


def analyze_mesh(
    *,
    case_id: str,
    seed: int,
    manifest_path: Path,
    payload: dict[str, Any],
    mesh_path: Path,
    pose: str,
    contact_fractions: tuple[float, ...],
) -> list[dict[str, Any]]:
    mesh = load_closed_mesh(mesh_path)
    center_mass = np.asarray(mesh.center_mass, dtype=np.float64)
    if center_mass.shape != (3,) or not np.all(np.isfinite(center_mass)):
        raise ValueError(f"Non-finite volume centroid: {mesh_path}")

    z_min = float(mesh.bounds[0, 2])
    max_extent = float(mesh.extents.max())
    com_height = float(center_mass[2] - z_min)
    projected_com = center_mass[:2]
    mesh_hash = sha256_file(mesh_path)
    canonical_hash = canonical_geometry_sha256(mesh)
    manifest_hash = sha256_file(manifest_path)
    selected = payload.get("selected", {})
    common = {
        "case_id": case_id,
        "seed": int(seed),
        "pose": pose,
        "pose_role": (
            "fixed_display_primary"
            if pose == "display"
            else "fdm_build_orientation_secondary"
        ),
        "run_status": payload.get("status"),
        "selection_eligible": bool(selected.get("eligible", False)),
        "fabrication_pass": bool(selected.get("fabrication_pass", False)),
        "candidate": selected.get("candidate"),
        "print_orientation": selected.get("print_orientation"),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "geometry_stl": str(mesh_path),
        "geometry_stl_raw_sha256": mesh_hash,
        "geometry_canonical_sha256": canonical_hash,
        "mesh_watertight": bool(mesh.is_watertight),
        "mesh_winding_consistent": bool(mesh.is_winding_consistent),
        "mesh_is_volume": bool(mesh.is_volume),
        "mesh_volume_mm3": float(mesh.volume),
        "bbox_x_mm": float(mesh.extents[0]),
        "bbox_y_mm": float(mesh.extents[1]),
        "bbox_z_mm": float(mesh.extents[2]),
        "max_extent_mm": max_extent,
        "ground_z_original_mm": z_min,
        "com_x_mm": float(center_mass[0]),
        "com_y_mm": float(center_mass[1]),
        "com_z_original_mm": float(center_mass[2]),
        "com_height_mm": com_height,
        "density_model": "uniform_closed_solid",
        "gravity_axis": "-z",
        "structural_screen_status": selected.get("structural_screen", {}).get(
            "status"
        ),
    }

    rows: list[dict[str, Any]] = []
    for fraction in contact_fractions:
        epsilon = float(fraction * max_extent)
        contact_points = lower_contact_band_points(mesh, epsilon)
        metrics = support_polygon_metrics(contact_points, projected_com, max_extent)
        row = {
            **common,
            "contact_band_fraction_of_max_extent": float(fraction),
            "contact_band_mm": epsilon,
            "contact_point_count": int(len(contact_points)),
            **metrics,
        }
        margin = row.get("signed_margin_mm")
        row["stable_under_vertical_gravity"] = bool(
            row.get("support_polygon_valid")
            and row.get("com_projection_inside")
            and margin is not None
            and float(margin) > max(max_extent * 1e-10, 1e-10)
        )
        row["signed_stability_angle_deg"] = (
            float(math.degrees(math.atan2(float(margin), com_height)))
            if margin is not None and com_height > 0.0
            else None
        )
        row["critical_tilt_angle_deg"] = (
            float(row["signed_stability_angle_deg"])
            if row["stable_under_vertical_gravity"]
            else None
        )
        row["normalized_margin_max_extent"] = (
            float(margin) / max_extent if margin is not None else None
        )
        rows.append(row)
    return rows


def aggregate_tasks(
    primary_rows: list[dict[str, Any]], expected_seeds: list[int]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary_rows:
        grouped[str(row["case_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for case_id in sorted(grouped):
        rows = sorted(grouped[case_id], key=lambda item: int(item["seed"]))
        margins = [
            float(row["signed_margin_mm"])
            for row in rows
            if row.get("signed_margin_mm") is not None
        ]
        normalized = [
            float(row["normalized_margin_diagonal"])
            for row in rows
            if row.get("normalized_margin_diagonal") is not None
        ]
        normalized_extent = [
            float(row["normalized_margin_max_extent"])
            for row in rows
            if row.get("normalized_margin_max_extent") is not None
        ]
        signed_angles = [
            float(row["signed_stability_angle_deg"])
            for row in rows
            if row.get("signed_stability_angle_deg") is not None
        ]
        margin_mean, margin_std = mean_std(margins)
        norm_mean, norm_std = mean_std(normalized)
        norm_extent_mean, norm_extent_std = mean_std(normalized_extent)
        summaries.append(
            {
                "case_id": case_id,
                "seeds": ",".join(str(row["seed"]) for row in rows),
                "expected_seeds": ",".join(str(seed) for seed in expected_seeds),
                "n_runs": int(len(rows)),
                "n_unique_geometry_hashes": int(
                    len({row["geometry_canonical_sha256"] for row in rows})
                ),
                "run_statuses": ",".join(
                    sorted({str(row["run_status"]) for row in rows})
                ),
                "all_runs_success": bool(
                    len(rows) == len(expected_seeds)
                    and all(row["run_status"] == "SUCCESS" for row in rows)
                ),
                "all_runs_eligible": bool(
                    len(rows) == len(expected_seeds)
                    and all(row["selection_eligible"] for row in rows)
                ),
                "valid_support_rate": float(
                    np.mean([bool(row["support_polygon_valid"]) for row in rows])
                ),
                "stable_seed_rate": float(
                    np.mean(
                        [
                            bool(row["stable_under_vertical_gravity"])
                            for row in rows
                        ]
                    )
                ),
                "task_pass_all_seeds": bool(
                    len(rows) == len(expected_seeds)
                    and all(
                        row["stable_under_vertical_gravity"] for row in rows
                    )
                ),
                "signed_margin_mm_mean": margin_mean,
                "signed_margin_mm_std": margin_std,
                "signed_margin_mm_min": min(margins) if margins else None,
                "normalized_margin_diagonal_mean": norm_mean,
                "normalized_margin_diagonal_std": norm_std,
                "normalized_margin_diagonal_min": (
                    min(normalized) if normalized else None
                ),
                "normalized_margin_max_extent_mean": norm_extent_mean,
                "normalized_margin_max_extent_std": norm_extent_std,
                "normalized_margin_max_extent_min": (
                    min(normalized_extent) if normalized_extent else None
                ),
                "signed_stability_angle_deg_min": (
                    min(signed_angles) if signed_angles else None
                ),
                "critical_tilt_angle_deg_min": min(
                    [
                        float(row["critical_tilt_angle_deg"])
                        for row in rows
                        if row.get("critical_tilt_angle_deg") is not None
                    ],
                    default=None,
                ),
            }
        )
    return summaries


def summarize(
    sensitivity_rows: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    primary_pose: str,
    primary_fraction: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    sensitivity: list[dict[str, Any]] = []
    poses = sorted({str(row["pose"]) for row in sensitivity_rows})
    fractions = sorted(
        {
            float(row["contact_band_fraction_of_max_extent"])
            for row in sensitivity_rows
        }
    )
    for pose in poses:
        for fraction in fractions:
            rows = [
                row
                for row in sensitivity_rows
                if row["pose"] == pose
                and math.isclose(
                    float(row["contact_band_fraction_of_max_extent"]),
                    fraction,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ]
            if not rows:
                continue
            margins = [
                float(row["signed_margin_mm"])
                for row in rows
                if row.get("signed_margin_mm") is not None
            ]
            task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                task_groups[str(row["case_id"])].append(row)
            sensitivity.append(
                {
                    "pose": pose,
                    "contact_band_fraction_of_max_extent": fraction,
                    "contact_band_mm_median": float(
                        np.median([row["contact_band_mm"] for row in rows])
                    ),
                    "n_runs": len(rows),
                    "n_tasks": len(task_groups),
                    "valid_support_polygons": int(
                        sum(bool(row["support_polygon_valid"]) for row in rows)
                    ),
                    "stable_runs": int(
                        sum(
                            bool(row["stable_under_vertical_gravity"])
                            for row in rows
                        )
                    ),
                    "tasks_passing_all_seeds": int(
                        sum(
                            all(
                                candidate["stable_under_vertical_gravity"]
                                for candidate in task_group
                            )
                            for task_group in task_groups.values()
                        )
                    ),
                    "worst_signed_margin_mm": min(margins) if margins else None,
                }
            )

    task_norm_means = [
        float(row["normalized_margin_diagonal_mean"])
        for row in task_rows
        if row.get("normalized_margin_diagonal_mean") is not None
    ]
    task_margin_means = [
        float(row["signed_margin_mm_mean"])
        for row in task_rows
        if row.get("signed_margin_mm_mean") is not None
    ]
    task_extent_means = [
        float(row["normalized_margin_max_extent_mean"])
        for row in task_rows
        if row.get("normalized_margin_max_extent_mean") is not None
    ]
    task_norm_mean, task_norm_std = mean_std(task_norm_means)
    task_margin_mean, task_margin_std = mean_std(task_margin_means)
    task_extent_mean, task_extent_std = mean_std(task_extent_means)
    finite_primary = [
        row for row in primary_rows if row.get("signed_margin_mm") is not None
    ]
    worst = (
        min(finite_primary, key=lambda row: float(row["signed_margin_mm"]))
        if finite_primary
        else None
    )
    success_tasks = [row for row in task_rows if row["all_runs_success"]]
    return {
        "claim_scope": (
            "Uniform-density rigid-body geometric tipping stability under "
            "vertical gravity on an ideal planar support."
        ),
        "not_certified": [
            "fracture or yielding",
            "sliding or friction",
            "layer adhesion and process defects",
            "warping",
            "impact or handling loads",
            "physical printed-part stability",
        ],
        "primary_pose": primary_pose,
        "primary_pose_definition": (
            "fixed display/projection coordinates"
            if primary_pose == "display"
            else "FDM build orientation"
        ),
        "primary_contact_band_fraction_of_max_extent": primary_fraction,
        "n_runs": len(primary_rows),
        "n_tasks": len(task_rows),
        "n_unique_raw_geometry_files": len(
            {row["geometry_stl_raw_sha256"] for row in primary_rows}
        ),
        "n_unique_canonical_geometries": len(
            {row["geometry_canonical_sha256"] for row in primary_rows}
        ),
        "run_status_counts": {
            status: int(sum(row["run_status"] == status for row in primary_rows))
            for status in sorted({str(row["run_status"]) for row in primary_rows})
        },
        "valid_support_polygons": int(
            sum(bool(row["support_polygon_valid"]) for row in primary_rows)
        ),
        "stable_runs": int(
            sum(bool(row["stable_under_vertical_gravity"]) for row in primary_rows)
        ),
        "task_passes_all_seeds": int(
            sum(bool(row["task_pass_all_seeds"]) for row in task_rows)
        ),
        "success_only_task_passes_all_seeds": int(
            sum(bool(row["task_pass_all_seeds"]) for row in success_tasks)
        ),
        "success_only_task_count": len(success_tasks),
        "task_level_signed_margin_mm_mean": task_margin_mean,
        "task_level_signed_margin_mm_std": task_margin_std,
        "task_level_normalized_margin_diagonal_mean": task_norm_mean,
        "task_level_normalized_margin_diagonal_std": task_norm_std,
        "task_level_normalized_margin_max_extent_mean": task_extent_mean,
        "task_level_normalized_margin_max_extent_std": task_extent_std,
        "task_level_normalized_margin_diagonal_mean_95pct_bootstrap_ci": (
            bootstrap_mean_ci(
                task_norm_means, bootstrap_samples, bootstrap_seed
            )
        ),
        "worst_primary_run": (
            {
                "case_id": worst["case_id"],
                "seed": worst["seed"],
                "signed_margin_mm": worst["signed_margin_mm"],
                "normalized_margin_diagonal": worst[
                    "normalized_margin_diagonal"
                ],
                "normalized_margin_max_extent": worst[
                    "normalized_margin_max_extent"
                ],
                "signed_stability_angle_deg": worst[
                    "signed_stability_angle_deg"
                ],
                "critical_tilt_angle_deg": worst["critical_tilt_angle_deg"],
            }
            if worst
            else None
        ),
        "contact_band_sensitivity": sensitivity,
        "statistical_unit": (
            "task; the three seeds are aggregated within each task and are not "
            "treated as 30 independent tasks"
        ),
    }


def render_report(summary: dict[str, Any]) -> str:
    worst = summary.get("worst_primary_run") or {}
    ci = summary.get(
        "task_level_normalized_margin_diagonal_mean_95pct_bootstrap_ci"
    )
    ci_text = (
        f"[{ci[0]:.6f}, {ci[1]:.6f}]" if ci is not None else "not available"
    )
    return "\n".join(
        [
            "# Frozen connected-10 COM/support-polygon audit",
            "",
            "Scope: " + summary["claim_scope"],
            "",
            f"- Primary pose: {summary['primary_pose']} "
            f"({summary['primary_pose_definition']}).",
            f"- Primary contact band: "
            f"{100.0 * summary['primary_contact_band_fraction_of_max_extent']:.3f}% "
            "of the mesh's maximum extent.",
            f"- Runs: {summary['n_runs']} across {summary['n_tasks']} tasks; "
            f"{summary['n_unique_canonical_geometries']} unique canonical geometries.",
            f"- Valid support polygons: {summary['valid_support_polygons']}/"
            f"{summary['n_runs']}.",
            f"- Stable runs: {summary['stable_runs']}/{summary['n_runs']}.",
            f"- Tasks passing for all seeds: "
            f"{summary['task_passes_all_seeds']}/{summary['n_tasks']}.",
            f"- SUCCESS-only tasks passing for all seeds: "
            f"{summary['success_only_task_passes_all_seeds']}/"
            f"{summary['success_only_task_count']}.",
            f"- Task-level normalized-margin mean ± SD: "
            f"{summary['task_level_normalized_margin_diagonal_mean']:.6f} ± "
            f"{summary['task_level_normalized_margin_diagonal_std']:.6f}; "
            f"95% task-bootstrap CI {ci_text}.",
            f"- Worst primary run: {worst.get('case_id')}, seed "
            f"{worst.get('seed')}, margin {worst.get('signed_margin_mm')} mm.",
            "",
            "The result is a digital uniform-density tipping audit. It does not "
            "certify fracture resistance, sliding, manufacturing defects, or "
            "the behavior of a physical printed part.",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--comparison-table", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--contact-band-fracs",
        type=parse_fractions,
        default=DEFAULT_CONTACT_FRACTIONS,
        help="Comma-separated fractions of the mesh maximum extent.",
    )
    parser.add_argument(
        "--primary-contact-band-frac",
        type=float,
        default=DEFAULT_PRIMARY_FRACTION,
    )
    parser.add_argument(
        "--poses",
        type=parse_poses,
        default=DEFAULT_POSES,
        help="Comma-separated poses to audit: display,print.",
    )
    parser.add_argument(
        "--primary-pose",
        choices=("display", "print"),
        default=DEFAULT_PRIMARY_POSE,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Restrict to a case_id; repeat for multiple cases.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a subset of the frozen case/seed grid (for smoke tests only).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    fractions = tuple(args.contact_band_fracs)
    poses = tuple(args.poses)
    primary_pose = str(args.primary_pose)
    if primary_pose not in poses:
        raise ValueError("--primary-pose must be included in --poses")
    primary = float(args.primary_contact_band_frac)
    if primary <= 0.0 or primary >= 0.1:
        raise ValueError("--primary-contact-band-frac must be in (0, 0.1)")
    if not any(math.isclose(primary, value, abs_tol=1e-12) for value in fractions):
        fractions = tuple(sorted({*fractions, primary}))
    if args.bootstrap_samples < 1000:
        raise ValueError("--bootstrap-samples must be at least 1000")

    benchmark_path = args.benchmark_manifest.resolve()
    comparison_path = args.comparison_table.resolve()
    runs_root = args.runs_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output / "RUN_MANIFEST.json"
    run_record: dict[str, Any] = {
        "schema": "shadowforge_com_support_polygon_v2",
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "command": " ".join(sys.argv),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "benchmark_manifest": str(benchmark_path),
        "benchmark_manifest_sha256": sha256_file(benchmark_path),
        "comparison_table": str(comparison_path),
        "comparison_table_sha256": sha256_file(comparison_path),
        "runs_root": str(runs_root),
        "output": str(output),
        "poses": list(poses),
        "primary_pose": primary_pose,
        "contact_band_fractions_of_max_extent": list(fractions),
        "primary_contact_band_fraction_of_max_extent": primary,
        "density_model": "uniform_closed_solid",
        "gravity_axis": "-z",
        "support_definition": (
            "2D convex hull of the lower contact band of the manifest-selected, "
            "closed STL; fixed-display pose is primary and FDM build pose is a "
            "secondary diagnostic"
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "trimesh": trimesh.__version__,
        },
    }
    write_json(run_manifest_path, run_record)

    try:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark.get("status") != "APPROVED_FROZEN":
            raise ValueError(
                "Benchmark is not APPROVED_FROZEN: "
                f"{benchmark.get('status')!r}"
            )
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison.get("status") != "complete":
            raise ValueError(
                f"Comparison table is not complete: {comparison.get('status')!r}"
            )
        pcraft_aggregate = next(
            (
                row
                for row in comparison.get("aggregate", [])
                if row.get("method") == "P-CRAFT"
            ),
            None,
        )
        if not pcraft_aggregate or int(pcraft_aggregate.get("tasks", -1)) != 10:
            raise ValueError("Comparison table lacks the frozen 10-task P-CRAFT row")
        expected_cases = [str(case["case_id"]) for case in benchmark["cases"]]
        expected_seeds = [int(seed) for seed in benchmark["protocol"]["seeds"]]
        selected_cases = args.cases or expected_cases
        unknown = sorted(set(selected_cases) - set(expected_cases))
        if unknown:
            raise ValueError(f"Cases are absent from frozen benchmark: {unknown}")
        if not args.allow_partial and selected_cases != expected_cases:
            raise ValueError(
                "Full audit requires every frozen case in manifest order; "
                "use --allow-partial only for a smoke test"
            )

        inputs: list[tuple[str, int, Path, dict[str, Any], str, Path]] = []
        for case_id in selected_cases:
            for seed in expected_seeds:
                manifest_path = (
                    runs_root / case_id / f"seed_{seed}" / "MANIFEST.json"
                )
                if not manifest_path.is_file():
                    raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                for pose in poses:
                    mesh_path = resolve_geometry(manifest_path, payload, pose)
                    inputs.append(
                        (case_id, seed, manifest_path, payload, pose, mesh_path)
                    )
        expected_count = len(selected_cases) * len(expected_seeds) * len(poses)
        if len(inputs) != expected_count:
            raise RuntimeError(
                f"Resolved {len(inputs)} runs, expected {expected_count}"
            )

        sensitivity_rows: list[dict[str, Any]] = []
        for index, (
            case_id,
            seed,
            manifest_path,
            payload,
            pose,
            mesh_path,
        ) in enumerate(inputs, start=1):
            print(
                f"[{index:02d}/{len(inputs):02d}] {case_id} seed={seed} "
                f"pose={pose} mesh={mesh_path.name}",
                flush=True,
            )
            sensitivity_rows.extend(
                analyze_mesh(
                    case_id=case_id,
                    seed=seed,
                    manifest_path=manifest_path,
                    payload=payload,
                    mesh_path=mesh_path,
                    pose=pose,
                    contact_fractions=fractions,
                )
            )

        primary_rows = [
            row
            for row in sensitivity_rows
            if row["pose"] == primary_pose
            and math.isclose(
                float(row["contact_band_fraction_of_max_extent"]),
                primary,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        pose_primary_band_rows = [
            row
            for row in sensitivity_rows
            if math.isclose(
                float(row["contact_band_fraction_of_max_extent"]),
                primary,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        task_rows = aggregate_tasks(primary_rows, expected_seeds)
        summary = summarize(
            sensitivity_rows,
            primary_rows,
            task_rows,
            primary_pose,
            primary,
            args.bootstrap_samples,
            args.bootstrap_seed,
        )

        write_csv(output / "per_run_sensitivity.csv", sensitivity_rows)
        write_csv(output / "per_run_primary.csv", primary_rows)
        write_csv(output / "per_pose_primary_band.csv", pose_primary_band_rows)
        write_csv(output / "task_summary.csv", task_rows)
        write_json(output / "summary.json", summary)
        (output / "REPORT.md").write_text(
            render_report(summary), encoding="utf-8"
        )

        run_record.update(
            {
                "status": "COMPLETE",
                "completed_at_utc": utc_now(),
                "n_resolved_run_records": len(selected_cases) * len(expected_seeds),
                "n_geometry_evaluations": len(inputs),
                "outputs": {
                    "summary": str(output / "summary.json"),
                    "report": str(output / "REPORT.md"),
                    "per_run_primary": str(output / "per_run_primary.csv"),
                    "per_run_sensitivity": str(
                        output / "per_run_sensitivity.csv"
                    ),
                    "per_pose_primary_band": str(
                        output / "per_pose_primary_band.csv"
                    ),
                    "task_summary": str(output / "task_summary.csv"),
                },
            }
        )
        write_json(run_manifest_path, run_record)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
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
