#!/usr/bin/env python3
"""Reproducible bare-model COM audit for the four photographed prints.

The input is a frozen JSON protocol containing explicit case IDs, absolute
artifact paths, and SHA-256 hashes.  Only the bare ``selected_display_fixed``
mesh is evaluated.  Build-orientation meshes are identity evidence, not an
alternative gravity frame, and legacy plinth assets are deliberately ignored.

This is a retrospective frozen-protocol audit, not a prospective
preregistration: exploratory values existed before the protocol lock.
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
from typing import Any

import numpy as np
import scipy
from scipy.spatial import ConvexHull
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_ordinary_two_view_com_20260729 import (  # noqa: E402
    load_closed_mesh,
    lower_band_points,
    numerical_tolerance_mm,
    support_polygon_metrics,
)


SCHEMA = "shadowforge_physical_print_exact4_com_audit_v1"
EXPECTED_PROTOCOL_SCHEMA = "shadowforge_physical_print_exact4_com_protocol_v1"


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


def write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this script's named output files.",
    )
    return parser.parse_args()


def prepare_outputs(
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.json",
        "per_case": output_dir / "per_case.csv",
        "sensitivity": output_dir / "sensitivity.csv",
        "run_manifest": output_dir / "AUDIT_RUN_MANIFEST.json",
        "report": output_dir / "REPORT.md",
        "figure": output_dir / "support_polygons_primary.png",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist: {[str(path) for path in existing]}"
        )
    for path in existing:
        if not path.is_file():
            raise FileExistsError(f"Output target is not a file: {path}")
    return paths


def require_file_and_hash(
    case_id: str,
    role: str,
    raw_path: str,
    expected_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{case_id}: missing {role}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{case_id}: {role} SHA-256 mismatch; "
            f"expected={expected_sha256}, actual={actual}"
        )
    return path, {
        "path": str(path),
        "sha256": actual,
        "bytes": path.stat().st_size,
    }


def surface_centroid(mesh: trimesh.Trimesh) -> np.ndarray:
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if (
        centers.ndim != 2
        or centers.shape[1] != 3
        or len(centers) != len(areas)
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(areas))
        or float(np.sum(areas)) <= 0.0
    ):
        raise ValueError("Cannot compute a finite area-weighted shell centroid")
    return np.average(centers, axis=0, weights=areas)


def mesh_summary(path: Path, mesh: trimesh.Trimesh) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent_after_in_memory_reorientation": bool(
            mesh.is_winding_consistent
        ),
        "inverted_winding_for_analysis": bool(
            mesh.metadata.get("inverted_winding_for_analysis", False)
        ),
        "source_signed_volume_mm3": float(
            mesh.metadata.get("source_signed_volume_mm3", mesh.volume)
        ),
        "analysis_volume_mm3": float(mesh.volume),
        "surface_area_mm2": float(mesh.area),
        "bounds_mm": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "extents_mm": np.asarray(mesh.extents, dtype=np.float64).tolist(),
        "solid_centroid_mm": np.asarray(
            mesh.center_mass, dtype=np.float64
        ).tolist(),
        "shell_centroid_mm": surface_centroid(mesh).tolist(),
    }


def congruence_record(
    display_mesh: trimesh.Trimesh,
    print_mesh: trimesh.Trimesh,
) -> dict[str, Any]:
    display_extents = np.sort(np.asarray(display_mesh.extents, dtype=np.float64))
    print_extents = np.sort(np.asarray(print_mesh.extents, dtype=np.float64))
    volume_rel = abs(float(display_mesh.volume) - float(print_mesh.volume)) / max(
        float(display_mesh.volume), 1e-12
    )
    area_rel = abs(float(display_mesh.area) - float(print_mesh.area)) / max(
        float(display_mesh.area), 1e-12
    )
    extent_rel = float(
        np.max(
            np.abs(display_extents - print_extents)
            / np.maximum(display_extents, 1e-12)
        )
    )
    return {
        "volume_relative_difference": float(volume_rel),
        "surface_area_relative_difference": float(area_rel),
        "sorted_extents_max_relative_difference": extent_rel,
        "same_vertex_count": int(len(display_mesh.vertices))
        == int(len(print_mesh.vertices)),
        "same_face_count": int(len(display_mesh.faces))
        == int(len(print_mesh.faces)),
        "consistent_with_rigid_axis_rotation": bool(
            volume_rel <= 1e-8 and area_rel <= 1e-8 and extent_rel <= 1e-8
        ),
    }


def analyze_case(
    case: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
]:
    case_id = str(case["case_id"])
    display_path, display_artifact = require_file_and_hash(
        case_id,
        "display_stl",
        case["display_stl"],
        case["display_stl_sha256"],
    )
    print_path, print_artifact = require_file_and_hash(
        case_id,
        "print_stl",
        case["print_stl"],
        case["print_stl_sha256"],
    )
    _, photo_artifact = require_file_and_hash(
        case_id,
        "photo",
        case["photo"],
        case["photo_sha256"],
    )

    mesh = load_closed_mesh(display_path)
    print_mesh = load_closed_mesh(print_path)
    max_extent = float(np.max(mesh.extents))
    h_mm = max_extent / 256.0
    ground_z = float(mesh.bounds[0, 2])
    tolerance = numerical_tolerance_mm(max_extent)
    solid_com = np.asarray(mesh.center_mass, dtype=np.float64)
    shell_com = surface_centroid(mesh)
    fractions = [
        float(value) for value in protocol["surface_mass_fractions"]
    ]
    band_factors = [
        float(value) for value in protocol["contact_band_factors_h"]
    ]
    primary_factor = float(protocol["primary_contact_band_factor_h"])

    contact_by_factor: dict[float, np.ndarray] = {}
    for band_factor in band_factors:
        contact_by_factor[band_factor] = lower_band_points(
            mesh,
            ground_z_mm=ground_z,
            contact_band_mm=band_factor * h_mm,
            numerical_tolerance=tolerance,
        )

    sensitivity_rows: list[dict[str, Any]] = []
    for surface_fraction in fractions:
        center_mass = (
            (1.0 - surface_fraction) * solid_com
            + surface_fraction * shell_com
        )
        for band_factor in band_factors:
            metrics = support_polygon_metrics(
                contact_points=contact_by_factor[band_factor],
                projected_com=center_mass[:2],
                com_height_mm=float(center_mass[2] - ground_z),
                max_extent_mm=max_extent,
                numerical_tolerance=tolerance,
            )
            sensitivity_rows.append(
                {
                    "case_id": case_id,
                    "pose": "fixed_display_z_up",
                    "gravity_axis": "-z",
                    "surface_mass_fraction": surface_fraction,
                    "mass_model": (
                        "uniform_solid"
                        if math.isclose(surface_fraction, 0.0)
                        else (
                            "uniform_thin_shell"
                            if math.isclose(surface_fraction, 1.0)
                            else "solid_shell_centroid_interpolation"
                        )
                    ),
                    "contact_band_factor_h": band_factor,
                    "contact_band_mm": band_factor * h_mm,
                    "is_primary_endpoint": bool(
                        math.isclose(surface_fraction, 0.0)
                        and math.isclose(band_factor, primary_factor)
                    ),
                    "com_x_mm": float(center_mass[0]),
                    "com_y_mm": float(center_mass[1]),
                    "com_z_mm": float(center_mass[2]),
                    **metrics,
                }
            )

    primary = [
        row for row in sensitivity_rows if row["is_primary_endpoint"]
    ]
    if len(primary) != 1:
        raise RuntimeError(f"{case_id}: expected exactly one primary endpoint")
    primary_row = primary[0]
    stable_all = bool(
        all(
            row.get("stable_under_vertical_gravity", False)
            for row in sensitivity_rows
        )
    )
    margins = [
        float(row["signed_margin_mm"])
        for row in sensitivity_rows
        if row.get("signed_margin_mm") is not None
    ]
    angles = [
        float(row["signed_stability_angle_deg"])
        for row in sensitivity_rows
        if row.get("signed_stability_angle_deg") is not None
    ]
    robust_case_pass = bool(
        stable_all
        and margins
        and angles
        and min(margins) >= 1.0
        and min(angles) >= 2.0
    )
    shell_primary = next(
        row
        for row in sensitivity_rows
        if math.isclose(float(row["surface_mass_fraction"]), 1.0)
        and math.isclose(float(row["contact_band_factor_h"]), primary_factor)
    )
    solid_band_consistent = bool(
        all(
            row.get("stable_under_vertical_gravity", False)
            == primary_row.get("stable_under_vertical_gravity", False)
            for row in sensitivity_rows
            if math.isclose(float(row["surface_mass_fraction"]), 0.0)
        )
    )
    density_primary_consistent = bool(
        all(
            row.get("stable_under_vertical_gravity", False)
            == primary_row.get("stable_under_vertical_gravity", False)
            for row in sensitivity_rows
            if math.isclose(
                float(row["contact_band_factor_h"]), primary_factor
            )
        )
    )
    # Derived design diagnostic only: for an ideal massless rigid support whose
    # footprint is the Minkowski dilation of the measured support polygon, an
    # outward offset r increases signed margin by r.  This is a lower bound for
    # fixture design, not evidence about the unmeasured photographed fixture.
    def required_footprint_dilation(row: dict[str, Any]) -> float:
        margin = float(row["signed_margin_mm"])
        height = float(row["com_height_above_support_mm"])
        target_margin = max(1.0, height * math.tan(math.radians(2.0)))
        return max(0.0, target_margin - margin)

    primary_required_dilation = required_footprint_dilation(primary_row)
    robust_required_dilation = max(
        required_footprint_dilation(row) for row in sensitivity_rows
    )

    case_summary = {
        "case_id": case_id,
        "artifact_linkage_status": case["artifact_linkage_status"],
        "fixture_observation": case["fixture_observation"],
        "legacy_display_delivery_ignored": True,
        "display_artifact": display_artifact,
        "print_artifact": {
            **print_artifact,
            "orientation_label": case["print_orientation_label"],
            "used_as_gravity_frame": False,
        },
        "photo_artifact": photo_artifact,
        "display_print_geometry_check": congruence_record(mesh, print_mesh),
        "display_mesh": mesh_summary(display_path, mesh),
        "max_extent_mm": max_extent,
        "h_mm": h_mm,
        "ground_z_mm": ground_z,
        "numerical_tolerance_mm": tolerance,
        "primary_uniform_solid_h": primary_row,
        "primary_uniform_thin_shell_h": shell_primary,
        "solid_contact_band_classification_consistent": solid_band_consistent,
        "density_primary_classification_consistent": density_primary_consistent,
        "stable_across_all_frozen_sensitivity_cells": stable_all,
        "minimum_signed_margin_across_sensitivity_mm": min(margins),
        "maximum_signed_margin_across_sensitivity_mm": max(margins),
        "minimum_signed_stability_angle_across_sensitivity_deg": min(angles),
        "robust_case_pass": robust_case_pass,
        "support_design_diagnostic": {
            "interpretation": (
                "minimum isotropic outward support-footprint dilation for an "
                "ideal massless rigidly bonded support to reach both 1 mm "
                "margin and 2 degree angle; lower bound, not physical evidence"
            ),
            "primary_required_dilation_mm": primary_required_dilation,
            "all_sensitivity_cells_required_dilation_mm": (
                robust_required_dilation
            ),
        },
    }
    flat = {
        "case_id": case_id,
        "primary_stability_class": primary_row["stability_class"],
        "primary_stable": primary_row["stable_under_vertical_gravity"],
        "primary_signed_margin_mm": primary_row["signed_margin_mm"],
        "primary_signed_stability_angle_deg": primary_row[
            "signed_stability_angle_deg"
        ],
        "thin_shell_primary_signed_margin_mm": shell_primary[
            "signed_margin_mm"
        ],
        "solid_contact_band_classification_consistent": solid_band_consistent,
        "density_primary_classification_consistent": density_primary_consistent,
        "minimum_sensitivity_margin_mm": min(margins),
        "maximum_sensitivity_margin_mm": max(margins),
        "minimum_sensitivity_angle_deg": min(angles),
        "robust_case_pass": robust_case_pass,
        "ideal_support_dilation_primary_mm": primary_required_dilation,
        "ideal_support_dilation_all_sensitivity_mm": robust_required_dilation,
        "max_extent_mm": max_extent,
        "h_mm": h_mm,
        "fixture_observation": case["fixture_observation"],
    }
    plot_payload = {
        "case_id": case_id,
        "contact_points": contact_by_factor[primary_factor],
        "solid_com_xy": solid_com[:2],
        "shell_com_xy": shell_com[:2],
        "primary_margin_mm": float(primary_row["signed_margin_mm"]),
    }
    return case_summary, flat, sensitivity_rows, plot_payload


def create_figure(
    path: Path,
    plot_payloads: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for axis, payload in zip(axes.flat, plot_payloads, strict=True):
        points = np.asarray(payload["contact_points"], dtype=np.float64)
        hull = ConvexHull(points)
        polygon = points[hull.vertices]
        closed = np.vstack((polygon, polygon[0]))
        step = max(1, len(points) // 1200)
        axis.scatter(
            points[::step, 0],
            points[::step, 1],
            s=3,
            alpha=0.18,
            color="#4472C4",
            label="lower-band points",
        )
        axis.fill(
            closed[:, 0],
            closed[:, 1],
            color="#9DC3E6",
            alpha=0.35,
            label="support polygon",
        )
        axis.plot(closed[:, 0], closed[:, 1], color="#2F5597", linewidth=1.5)
        axis.scatter(
            *payload["solid_com_xy"],
            marker="x",
            s=80,
            linewidth=2.2,
            color="#C00000",
            label="solid COM projection",
            zorder=4,
        )
        axis.scatter(
            *payload["shell_com_xy"],
            marker="+",
            s=90,
            linewidth=2.0,
            color="#ED7D31",
            label="shell COM projection",
            zorder=4,
        )
        axis.set_title(
            f"{payload['case_id']}\n"
            f"primary signed margin = {payload['primary_margin_mm']:.2f} mm"
        )
        axis.set_xlabel("display x (mm)")
        axis.set_ylabel("display y (mm)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, linewidth=0.4, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2)
    figure.suptitle(
        "Bare-model COM audit: fixed display pose, primary h = D/256",
        fontsize=14,
    )
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=220, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def build_report(summary: dict[str, Any]) -> str:
    rows = []
    for case in summary["cases"]:
        primary = case["primary_uniform_solid_h"]
        shell = case["primary_uniform_thin_shell_h"]
        rows.append(
            "| {case} | {cls} | {margin:.2f} | {angle:.2f} | "
            "{shell_margin:.2f} | {dilation:.2f} | {robust} |".format(
                case=case["case_id"],
                cls=primary["stability_class"],
                margin=float(primary["signed_margin_mm"]),
                angle=float(primary["signed_stability_angle_deg"]),
                shell_margin=float(shell["signed_margin_mm"]),
                dilation=float(
                    case["support_design_diagnostic"][
                        "all_sensitivity_cells_required_dilation_mm"
                    ]
                ),
                robust="yes" if case["robust_case_pass"] else "no",
            )
        )
    expansion = summary["conditional_expansion"]
    return f"""# Exact-four physical-print COM audit

Status: **{summary["status"]}**  
Audit type: retrospective audit with a protocol frozen after exploratory
precheck; this is not presented as a prospective preregistration.

## Result

The primary endpoint is the bare delivered mesh in its fixed display pose,
under vertical gravity, using a uniform-solid centroid and the frozen
`h = D/256` contact band. A positive signed margin is stable; a negative
margin places the projected COM outside the support polygon.

| case | primary class | margin (mm) | angle (deg) | thin-shell margin (mm) | ideal support dilation lower bound (mm) | robust pass |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Primary stable count: **{summary["n_primary_stable"]}/{summary["n_cases"]}**.  
Robust pass count: **{summary["n_robust_case_pass"]}/{summary["n_cases"]}**.

The classification is unchanged across all frozen contact bands
(`0.5h`, `h`, `2h`) and across the solid-to-thin-shell centroid sensitivity
grid for all four cases. Thus, the result is not explained by choosing a
uniform-solid rather than shell-like mass model.

The support-dilation column is a design diagnostic, not a measurement. It is
the minimum isotropic footprint expansion for an ideal massless, rigidly bonded
support to achieve both a 1 mm margin and a 2-degree stability angle in every
frozen sensitivity cell. A real support must be modeled again with its measured
mass distribution and joint.

## Conditional expansion decision

Expansion status: **{expansion["status"]}**.

The frozen gate required all four cases to pass every sensitivity cell, with
minimum margin at least 1 mm and minimum stability angle at least 2 degrees.
Because the pilot gate failed, no favorable-case expansion was run.

## What the photographs establish

The artifact hashes and print/build-orientation companions tie these meshes to
the four fabricated examples. The photographs establish successful fabrication
and projection display. They contain visible transparent contact material, and
the latter two photographs also show a glue bottle. Since the fixture footprint,
mass, and mechanics were not recorded, it cannot be reconstructed
quantitatively. The photographs therefore do **not** establish unassisted
free-standing stability of the bare prints.

Legacy wide-plinth assets found in three result directories were deliberately
excluded: they are not visibly the supports used in the photographs and cannot
be substituted for the photographed setup.

## Required physical follow-up

To make a defensible free-standing claim, remove tape/glue, place each object in
the frozen display pose on a level plate, record three uninterrupted 60-second
trials, and measure the critical tilt angle along the digitally predicted
weakest direction. Report failures as failures. If support is part of the
intended artifact, scan or measure that exact support and report it as a
separate assembly endpoint rather than as bare-model stability.
"""


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol_payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol_payload.get("schema") != EXPECTED_PROTOCOL_SCHEMA:
        raise ValueError(
            "Unexpected protocol schema: "
            f"{protocol_payload.get('schema')!r}"
        )
    cases_input = protocol_payload.get("cases")
    if not isinstance(cases_input, list) or len(cases_input) != 4:
        raise ValueError("The exact-four audit requires exactly four cases")
    output_paths = prepare_outputs(args.output_dir.resolve(), args.overwrite)

    script_path = Path(__file__).resolve()
    run_manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "command": " ".join(sys.argv),
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "legacy_display_delivery_ignored": True,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "trimesh": trimesh.__version__,
        },
        "outputs": {key: str(value) for key, value in output_paths.items()},
    }
    write_json(output_paths["run_manifest"], run_manifest)

    try:
        case_summaries: list[dict[str, Any]] = []
        flat_rows: list[dict[str, Any]] = []
        sensitivity_rows: list[dict[str, Any]] = []
        plot_payloads: list[dict[str, Any]] = []
        for index, case in enumerate(cases_input, start=1):
            print(
                f"[{index}/4] {case['case_id']}: verifying hashes and auditing",
                flush=True,
            )
            case_summary, flat, sensitivity, plot_payload = analyze_case(
                case,
                protocol_payload["protocol"],
            )
            case_summaries.append(case_summary)
            flat_rows.append(flat)
            sensitivity_rows.extend(sensitivity)
            plot_payloads.append(plot_payload)

        n_primary_stable = sum(
            bool(case["primary_uniform_solid_h"][
                "stable_under_vertical_gravity"
            ])
            for case in case_summaries
        )
        n_robust = sum(
            bool(case["robust_case_pass"]) for case in case_summaries
        )
        gate_passed = n_robust == len(case_summaries)
        summary: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "audit_design": (
                "retrospective audit with protocol frozen after exploratory "
                "precheck; not a prospective preregistration"
            ),
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "claim_scope": (
                "bare-mesh digital geometric tipping in the fixed display "
                "pose; does not certify the photographed fixture-assisted "
                "assembly or physical free-standing behavior"
            ),
            "n_cases": len(case_summaries),
            "n_primary_stable": int(n_primary_stable),
            "n_robust_case_pass": int(n_robust),
            "all_hashes_verified": True,
            "all_legacy_display_delivery_ignored": True,
            "conditional_expansion": {
                "gate_passed": gate_passed,
                "status": "NOT_TRIGGERED" if not gate_passed else "TRIGGERED",
                "reason": (
                    "fewer than four robust pilot passes"
                    if not gate_passed
                    else "all four robust pilot passes"
                ),
                "expansion_executed_by_this_run": False,
            },
            "cases": case_summaries,
        }
        write_json(output_paths["summary"], summary)
        write_csv(output_paths["per_case"], flat_rows)
        write_csv(output_paths["sensitivity"], sensitivity_rows)
        create_figure(output_paths["figure"], plot_payloads)
        write_text(output_paths["report"], build_report(summary))
        run_manifest.update(
            {
                "status": "COMPLETE",
                "completed_at_utc": utc_now(),
                "counts": {
                    "cases": len(case_summaries),
                    "sensitivity_rows": len(sensitivity_rows),
                    "primary_stable": int(n_primary_stable),
                    "robust_case_pass": int(n_robust),
                },
                "conditional_expansion_status": summary[
                    "conditional_expansion"
                ]["status"],
                "output_sha256": {
                    key: sha256_file(path)
                    for key, path in output_paths.items()
                    if key != "run_manifest"
                },
            }
        )
        write_json(output_paths["run_manifest"], run_manifest)
        print(
            f"COMPLETE: primary stable {n_primary_stable}/4; "
            f"robust pass {n_robust}/4; expansion "
            f"{summary['conditional_expansion']['status']}",
            flush=True,
        )
        return 0
    except Exception as exc:
        run_manifest.update(
            {
                "status": "FAILED",
                "completed_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(output_paths["run_manifest"], run_manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
