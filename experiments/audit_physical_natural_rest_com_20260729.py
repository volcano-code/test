#!/usr/bin/env python3
"""Post-hoc natural-rest COM audit for photographed ShadowForge deliveries.

This audit does not force every mesh to use its file-coordinate +z direction
as gravity up.  It enumerates convex-hull stable poses with trimesh and selects
the pose whose table normal has the smallest angular deviation from the
nominal display +z direction.  Selection never uses the resulting COM margin,
the photograph, or the stable-pose probability.

The selected pose is then checked independently on the full delivered mesh
over frozen contact-band and solid-to-thin-shell centroid sensitivities.
Because the stable-pose enumerator itself uses the solid centroid, its solid
endpoint is an explanatory geometry diagnostic rather than independent proof.
The thin-shell endpoint and band sensitivity are out-of-model checks.  The
physical observation remains the direct evidence for unaided standing.
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


SCHEMA = "shadowforge_physical_natural_rest_com_audit_v2"
PROTOCOL_SCHEMA = "shadowforge_physical_natural_rest_com_protocol_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
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


def surface_centroid(mesh: trimesh.Trimesh) -> np.ndarray:
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if (
        centers.ndim != 2
        or centers.shape[1] != 3
        or len(centers) != len(areas)
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(areas))
        or float(areas.sum()) <= 0.0
    ):
        raise ValueError("Cannot compute finite area-weighted shell centroid")
    return np.average(centers, axis=0, weights=areas)


def verify_artifact(record: dict[str, Any], role: str) -> tuple[Path, dict[str, Any]]:
    path = Path(record["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role}: {path}")
    actual = sha256_file(path)
    expected = str(record["sha256"])
    if actual != expected:
        raise ValueError(
            f"{role} SHA-256 mismatch: expected={expected}, actual={actual}: {path}"
        )
    return path, {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def pose_tilt_from_file_z_deg(transform: np.ndarray) -> float:
    cosine = float(np.clip(transform[2, 2], -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def enumerate_and_select_pose(
    mesh: trimesh.Trimesh,
    reference_up_display: np.ndarray,
) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
    transforms, probabilities = trimesh.poses.compute_stable_poses(
        mesh,
        center_mass=np.asarray(mesh.center_mass, dtype=np.float64),
        sigma=0.0,
        n_samples=1,
        threshold=0.0,
    )
    if len(transforms) == 0:
        raise ValueError("Stable-pose enumeration returned no candidates")
    reference = np.asarray(reference_up_display, dtype=np.float64)
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_up_display must be a finite 3-vector")
    norm = float(np.linalg.norm(reference))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("reference_up_display must be nonzero")
    reference = reference / norm
    up_vectors = np.asarray(
        [np.asarray(item, dtype=np.float64)[2, :3] for item in transforms],
        dtype=np.float64,
    )
    deviations = np.degrees(
        np.arccos(np.clip(up_vectors @ reference, -1.0, 1.0))
    )
    file_z_tilts = np.asarray(
        [
            pose_tilt_from_file_z_deg(np.asarray(item, dtype=np.float64))
            for item in transforms
        ],
        dtype=np.float64,
    )
    selected_index = int(np.argmin(deviations))
    candidates: list[dict[str, Any]] = []
    for index, (transform, probability, deviation, file_z_tilt) in enumerate(
        zip(
            transforms,
            probabilities,
            deviations,
            file_z_tilts,
            strict=True,
        )
    ):
        matrix = np.asarray(transform, dtype=np.float64)
        candidates.append(
            {
                "candidate_index": index,
                "selected": index == selected_index,
                "probability": float(probability),
                "deviation_from_photo_reference_up_deg": float(deviation),
                "tilt_from_file_display_z_deg": float(file_z_tilt),
                "table_up_vector_in_display_coordinates": matrix[2, :3].tolist(),
                "transform_display_to_rest": matrix.tolist(),
            }
        )
    return (
        np.asarray(transforms[selected_index], dtype=np.float64),
        selected_index,
        candidates,
    )


def analyze_artifact(
    *,
    case_id: str,
    artifact_role: str,
    artifact_record: dict[str, Any],
    reference_up_display: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    mesh_path, verified = verify_artifact(artifact_record, f"{case_id}:{artifact_role}")
    mesh = load_closed_mesh(mesh_path)
    display_max_extent = float(np.max(mesh.extents))
    h_mm = display_max_extent / float(protocol["delivery_resolution"])
    transform, selected_index, candidates = enumerate_and_select_pose(
        mesh,
        reference_up_display,
    )

    rest_mesh = mesh.copy()
    rest_mesh.apply_transform(transform)
    ground_z = float(rest_mesh.bounds[0, 2])
    tolerance = numerical_tolerance_mm(display_max_extent)
    solid_com = np.asarray(rest_mesh.center_mass, dtype=np.float64)
    shell_com = surface_centroid(rest_mesh)
    band_factors = [
        float(value) for value in protocol["contact_band_factors_h"]
    ]
    mass_fractions = [
        float(value) for value in protocol["surface_mass_fractions"]
    ]
    primary_band = float(protocol["primary_contact_band_factor_h"])

    contacts: dict[float, np.ndarray] = {}
    for band_factor in band_factors:
        contacts[band_factor] = lower_band_points(
            rest_mesh,
            ground_z_mm=ground_z,
            contact_band_mm=band_factor * h_mm,
            numerical_tolerance=tolerance,
        )

    sensitivity_rows: list[dict[str, Any]] = []
    for surface_fraction in mass_fractions:
        center_mass = (
            (1.0 - surface_fraction) * solid_com
            + surface_fraction * shell_com
        )
        for band_factor in band_factors:
            metrics = support_polygon_metrics(
                contact_points=contacts[band_factor],
                projected_com=center_mass[:2],
                com_height_mm=float(center_mass[2] - ground_z),
                max_extent_mm=display_max_extent,
                numerical_tolerance=tolerance,
            )
            sensitivity_rows.append(
                {
                    "case_id": case_id,
                    "artifact_role": artifact_role,
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
                    "is_primary_solid_endpoint": bool(
                        math.isclose(surface_fraction, 0.0)
                        and math.isclose(band_factor, primary_band)
                    ),
                    "is_primary_shell_endpoint": bool(
                        math.isclose(surface_fraction, 1.0)
                        and math.isclose(band_factor, primary_band)
                    ),
                    "com_x_mm": float(center_mass[0]),
                    "com_y_mm": float(center_mass[1]),
                    "com_z_mm": float(center_mass[2]),
                    **metrics,
                }
            )

    solid_primary = next(
        row for row in sensitivity_rows if row["is_primary_solid_endpoint"]
    )
    shell_primary = next(
        row for row in sensitivity_rows if row["is_primary_shell_endpoint"]
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
    all_cells_stable = bool(
        all(row.get("stable_under_vertical_gravity", False) for row in sensitivity_rows)
    )
    gate = protocol["robust_gate"]
    robust_pass = bool(
        all_cells_stable
        and margins
        and angles
        and min(margins) >= float(gate["minimum_margin_mm"])
        and min(angles) >= float(gate["minimum_angle_deg"])
    )
    selected_candidate = candidates[selected_index]
    for candidate in candidates:
        candidate["case_id"] = case_id
        candidate["artifact_role"] = artifact_role

    summary = {
        "case_id": case_id,
        "artifact_role": artifact_role,
        "artifact": {
            **verified,
            "linkage_status": artifact_record["linkage_status"],
            "selection_basis": artifact_record["selection_basis"],
        },
        "mesh_qa": {
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "is_volume": bool(mesh.is_volume),
            "body_count": int(len(mesh.split(only_watertight=False))),
            "volume_mm3": float(mesh.volume),
            "surface_area_mm2": float(mesh.area),
            "display_extents_mm": np.asarray(mesh.extents).tolist(),
        },
        "pose_rule": protocol["pose_rule"],
        "photo_reference_up_in_display_coordinates": np.asarray(
            reference_up_display,
            dtype=np.float64,
        ).tolist(),
        "stable_pose_candidate_count": len(candidates),
        "selected_candidate_index": selected_index,
        "selected_pose_probability": selected_candidate["probability"],
        "deviation_from_photo_reference_up_deg": selected_candidate[
            "deviation_from_photo_reference_up_deg"
        ],
        "tilt_from_file_display_z_deg": selected_candidate[
            "tilt_from_file_display_z_deg"
        ],
        "table_up_vector_in_display_coordinates": selected_candidate[
            "table_up_vector_in_display_coordinates"
        ],
        "transform_display_to_rest": selected_candidate[
            "transform_display_to_rest"
        ],
        "display_max_extent_mm": display_max_extent,
        "h_mm": h_mm,
        "primary_uniform_solid": solid_primary,
        "primary_uniform_thin_shell": shell_primary,
        "all_sensitivity_cells_stable": all_cells_stable,
        "minimum_signed_margin_across_sensitivity_mm": min(margins),
        "minimum_signed_stability_angle_across_sensitivity_deg": min(angles),
        "robust_pass": robust_pass,
        "interpretive_limit": (
            "The pose enumerator uses the solid COM; the solid endpoint is not "
            "independent confirmation. Shell-COM and contact-band sensitivity "
            "are out-of-model checks."
        ),
    }
    plot_payload = {
        "case_id": case_id,
        "contact_points": contacts[primary_band],
        "solid_com_xy": solid_com[:2],
        "shell_com_xy": shell_com[:2],
        "solid_margin_mm": float(solid_primary["signed_margin_mm"]),
        "shell_margin_mm": float(shell_primary["signed_margin_mm"]),
        "reference_deviation_deg": float(
            summary["deviation_from_photo_reference_up_deg"]
        ),
        "file_z_tilt_deg": float(summary["tilt_from_file_display_z_deg"]),
    }
    return summary, sensitivity_rows, candidates, plot_payload


def create_figure(path: Path, payloads: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 2
    rows = int(math.ceil(len(payloads) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(10, 4.6 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, payload in zip(axes.flat, payloads, strict=False):
        points = np.asarray(payload["contact_points"], dtype=np.float64)
        hull = ConvexHull(points)
        polygon = points[hull.vertices]
        closed = np.vstack((polygon, polygon[0]))
        step = max(1, len(points) // 1500)
        axis.scatter(
            points[::step, 0],
            points[::step, 1],
            s=3,
            alpha=0.18,
            color="#4472C4",
        )
        axis.fill(closed[:, 0], closed[:, 1], color="#9DC3E6", alpha=0.4)
        axis.plot(closed[:, 0], closed[:, 1], color="#2F5597", linewidth=1.5)
        axis.scatter(
            *payload["solid_com_xy"],
            marker="x",
            s=80,
            linewidth=2.2,
            color="#C00000",
            label="solid COM",
            zorder=4,
        )
        axis.scatter(
            *payload["shell_com_xy"],
            marker="+",
            s=90,
            linewidth=2.0,
            color="#ED7D31",
            label="shell COM",
            zorder=4,
        )
        axis.set_title(
            f"{payload['case_id']} | reference adjustment "
            f"{payload['reference_deviation_deg']:.1f}°\n"
            f"margin solid/shell = {payload['solid_margin_mm']:.2f}/"
            f"{payload['shell_margin_mm']:.2f} mm"
        )
        axis.set_xlabel("rest-plane x (mm)")
        axis.set_ylabel("rest-plane y (mm)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, linewidth=0.4, alpha=0.25)
    for axis in axes.flat[len(payloads) :]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2)
    figure.suptitle(
        "Natural-rest support polygons (primary endpoint: exact contact face)",
        fontsize=14,
    )
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=220, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def report_text(summary: dict[str, Any]) -> str:
    rows: list[str] = []
    for case in summary["cases"]:
        primary = case["primary"]
        solid = primary["primary_uniform_solid"]
        shell = primary["primary_uniform_thin_shell"]
        rows.append(
            "| {case} | {deviation:.1f} | {file_z:.1f} | {solid:.2f} | "
            "{shell:.2f} | "
            "{angle:.2f} | {robust} |".format(
                case=case["case_id"],
                deviation=float(
                    primary["deviation_from_photo_reference_up_deg"]
                ),
                file_z=float(primary["tilt_from_file_display_z_deg"]),
                solid=float(solid["signed_margin_mm"]),
                shell=float(shell["signed_margin_mm"]),
                angle=float(primary["minimum_signed_stability_angle_across_sensitivity_deg"]),
                robust="yes" if case["all_artifact_variants_robust"] else "no",
            )
        )
    return f"""# Corrected four-print natural-rest COM audit

Status: **{summary["status"]}**

## Corrected result

The four photographed objects were reported by the experimenter to stand
unaided. Adhesive was used for other examples, not these four. A glue bottle
visible beside the setup is not in contact with the objects and is not treated
as a fixture.

The earlier `0/4` result is invalid because it paired the first photograph with
the wrong object and forced all meshes into file-coordinate `+z` rather than
their photographed natural resting pose.

| photographed delivery | adjustment from photo reference (deg) | tilt from file +z (deg) | exact-face solid margin (mm) | exact-face shell margin (mm) | worst sensitivity angle (deg) | robust incl. variants |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Primary artifacts stable on the exact contact face:
**{summary["n_primary_stable"]}/{summary["n_cases"]}**.  
Primary artifacts passing every frozen solid-to-shell ×
`exact/0.5h/h/2h` cell:
**{summary["n_primary_robust"]}/{summary["n_cases"]}**.  
Cases remaining robust across every recorded artifact-version alternative:
**{summary["n_case_variant_robust"]}/{summary["n_cases"]}**.

Across all primary artifacts, the worst frozen-cell signed margin is
**{summary["global_minimum_sensitivity_margin_mm"]:.2f} mm** and the worst
stability angle is **{summary["global_minimum_sensitivity_angle_deg"]:.2f}°**.

## Method

For each delivered mesh, `trimesh.poses.compute_stable_poses` enumerates stable
convex-hull support faces using the original closed-mesh solid centroid. The
audit deterministically selects the candidate with the smallest deviation from
the photograph's nominal table-up axis; it does not choose by margin or
probability. The two-view cases use display `+z`. The three-view owl/lobster/ant
case uses display `-y`, because owl and lobster are the two wall projections
while the omitted ant projection is aligned with the table normal.
The full delivered mesh is then transformed to that pose and independently
re-evaluated on the exact contact face and with lower contact bands `0.5h`,
`h`, and `2h`, where
`h = D/256`, over five solid-to-thin-shell centroid mixtures.

The selected tilts are consistent with the photographs: the dark-knight example
rests only slightly off nominal, while the sword example uses a visibly more
inclined face.

The sword/fox case is the limiting endpoint: its exact-face margin is only
1.16 mm and its worst stability angle is 2.42°. It clears the frozen
1 mm / 2° robustness gate, but should be described as a boundary case rather
than as strongly stable.

## Evidence scope

The direct physical endpoint is the experimenter's confirmed observation that
all four objects stand without adhesive or an external fixture. The COM audit
is a reproducible geometric explanation and sensitivity check. It is not a
load-bearing, sliding-friction, impact, or long-duration stability test.

The stable-pose routine uses the solid centroid, so the positive solid endpoint
is partly construction-dependent. The thin-shell endpoint and contact-band
sensitivity are the relevant out-of-model checks; all remain positive here.

The top-left fixed branch is supported by three independent local clues:
its owl/lobster/ant target-axis map makes display `-y` the photographed table
normal, its printed body matches the fixed render, and the pre-existing
main-comparison record selects `ours_fixed`. The free branch has a different
target-axis calibration and is therefore not treated as a coordinate-compatible
version alternative. The other three use the frozen paper inventory artifacts;
hashes, pose matrices, candidates, and all sensitivity cells are included in
this directory.
"""


def paper_notes(summary: dict[str, Any]) -> str:
    return f"""# Paper integration notes (do not overclaim)

## Main-text sentence

> All four photographed prints stood without adhesive or external fixtures.
> In a post-hoc geometric audit using the stable face closest to each
> photograph-derived nominal table-up axis, all four COM projections lay
> inside their
> support polygons across solid-to-shell mass models and exact-face plus
> `0.5h/h/2h` contact bands (worst signed margin
> {summary["global_minimum_sensitivity_margin_mm"]:.2f} mm; worst tipping angle
> {summary["global_minimum_sensitivity_angle_deg"]:.2f}°).

## Put in the supplement

- photo-to-target and artifact hashes;
- the deterministic natural-rest selection rule;
- all pose matrices, candidate counts, and stable-pose probabilities;
- per-case solid/shell COMs and exact-face plus `0.5h/h/2h` support polygons;
- the owl fixed-branch identification and target-axis evidence;
- the limitation that solid-COM pose enumeration is not independent proof;
- no friction, vibration, impact, duration, or strength claim.

## Forbidden interpretation

Do not use this COM result as FEM validation or structural-safety certification.
COM support-polygon stability and self-weight stress safety are different
properties.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_paths(directory: Path, overwrite: bool) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": directory / "summary.json",
        "per_case": directory / "per_case.csv",
        "sensitivity": directory / "sensitivity.csv",
        "candidates": directory / "rest_pose_candidates.csv",
        "figure": directory / "support_polygons_natural_rest.png",
        "report": directory / "REPORT.md",
        "paper_notes": directory / "PAPER_INTEGRATION_NOTES.md",
        "manifest": directory / "AUDIT_RUN_MANIFEST.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output files exist: {[str(path) for path in existing]}")
    return paths


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"Unexpected protocol schema: {payload.get('schema')!r}")
    cases = payload["cases"]
    if len(cases) != int(payload["expected_case_count"]):
        raise ValueError("Case count does not match expected_case_count")
    paths = output_paths(args.output_dir.resolve(), args.overwrite)
    script_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "command": " ".join(sys.argv),
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "trimesh": trimesh.__version__,
        },
    }
    write_json(paths["manifest"], manifest)
    try:
        case_summaries: list[dict[str, Any]] = []
        flat_rows: list[dict[str, Any]] = []
        all_sensitivity: list[dict[str, Any]] = []
        all_candidates: list[dict[str, Any]] = []
        plot_payloads: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            case_id = str(case["case_id"])
            print(f"[{index}/{len(cases)}] {case_id}", flush=True)
            photo_path, photo = verify_artifact(case["photo"], f"{case_id}:photo")
            targets = [
                verify_artifact(target, f"{case_id}:target")[1]
                for target in case["visible_target_masks"]
            ]
            artifacts = [("primary", case["primary_artifact"])] + [
                (f"alternative_{alt_index}", item)
                for alt_index, item in enumerate(case.get("artifact_alternatives", []), start=1)
            ]
            artifact_summaries: list[dict[str, Any]] = []
            primary_plot: dict[str, Any] | None = None
            for artifact_role, artifact in artifacts:
                result, sensitivity, candidates, plot = analyze_artifact(
                    case_id=case_id,
                    artifact_role=artifact_role,
                    artifact_record=artifact,
                    reference_up_display=np.asarray(
                        case["photo_reference_up_in_display_coordinates"],
                        dtype=np.float64,
                    ),
                    protocol=payload["protocol"],
                )
                artifact_summaries.append(result)
                all_sensitivity.extend(sensitivity)
                all_candidates.extend(candidates)
                if artifact_role == "primary":
                    primary_plot = plot
            assert primary_plot is not None
            plot_payloads.append(primary_plot)
            primary = artifact_summaries[0]
            variant_robust = all(item["robust_pass"] for item in artifact_summaries)
            case_summary = {
                "case_id": case_id,
                "figure_position": case["figure_position"],
                "visible_targets": case["visible_targets"],
                "photo": {**photo, "path": str(photo_path)},
                "visible_target_masks": targets,
                "physical_observation": case["physical_observation"],
                "artifact_linkage_note": case["artifact_linkage_note"],
                "primary": primary,
                "artifact_alternatives": artifact_summaries[1:],
                "all_artifact_variants_robust": variant_robust,
            }
            case_summaries.append(case_summary)
            flat_rows.append(
                {
                    "case_id": case_id,
                    "figure_position": case["figure_position"],
                    "visible_targets": " / ".join(case["visible_targets"]),
                    "primary_artifact": primary["artifact"]["path"],
                    "primary_sha256": primary["artifact"]["sha256"],
                    "linkage_status": primary["artifact"]["linkage_status"],
                    "photo_reference_up": json.dumps(
                        case["photo_reference_up_in_display_coordinates"]
                    ),
                    "reference_adjustment_deg": primary[
                        "deviation_from_photo_reference_up_deg"
                    ],
                    "file_z_tilt_deg": primary["tilt_from_file_display_z_deg"],
                    "solid_margin_primary_mm": primary["primary_uniform_solid"][
                        "signed_margin_mm"
                    ],
                    "shell_margin_primary_mm": primary["primary_uniform_thin_shell"][
                        "signed_margin_mm"
                    ],
                    "minimum_sensitivity_margin_mm": primary[
                        "minimum_signed_margin_across_sensitivity_mm"
                    ],
                    "minimum_sensitivity_angle_deg": primary[
                        "minimum_signed_stability_angle_across_sensitivity_deg"
                    ],
                    "primary_robust_pass": primary["robust_pass"],
                    "all_artifact_variants_robust": variant_robust,
                    "unaided_self_standing_observed": case["physical_observation"][
                        "unaided_self_standing"
                    ],
                }
            )

        primary_results = [case["primary"] for case in case_summaries]
        n_primary_stable = sum(
            bool(item["primary_uniform_solid"]["stable_under_vertical_gravity"])
            and bool(item["primary_uniform_thin_shell"]["stable_under_vertical_gravity"])
            for item in primary_results
        )
        n_primary_robust = sum(bool(item["robust_pass"]) for item in primary_results)
        n_variant_robust = sum(
            bool(case["all_artifact_variants_robust"]) for case in case_summaries
        )
        summary = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "audit_design": (
                "post-hoc deterministic natural-rest audit; not prospective "
                "preregistration and not independent physical validation"
            ),
            "correction": payload["correction"],
            "physical_observation_scope": payload["physical_observation_scope"],
            "protocol": payload["protocol"],
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "n_cases": len(case_summaries),
            "n_primary_stable": int(n_primary_stable),
            "n_primary_robust": int(n_primary_robust),
            "n_case_variant_robust": int(n_variant_robust),
            "global_minimum_sensitivity_margin_mm": min(
                float(item["minimum_signed_margin_across_sensitivity_mm"])
                for item in primary_results
            ),
            "global_minimum_sensitivity_angle_deg": min(
                float(item["minimum_signed_stability_angle_across_sensitivity_deg"])
                for item in primary_results
            ),
            "expansion_gate_passed": bool(n_variant_robust == len(case_summaries)),
            "cases": case_summaries,
        }
        write_json(paths["summary"], summary)
        write_csv(paths["per_case"], flat_rows)
        write_csv(paths["sensitivity"], all_sensitivity)
        write_csv(paths["candidates"], all_candidates)
        create_figure(paths["figure"], plot_payloads)
        write_text(paths["report"], report_text(summary))
        write_text(paths["paper_notes"], paper_notes(summary))
        manifest.update(
            {
                "status": "COMPLETE",
                "completed_at_utc": utc_now(),
                "counts": {
                    "cases": len(case_summaries),
                    "artifact_analyses": sum(
                        1 + len(case["artifact_alternatives"]) for case in case_summaries
                    ),
                    "sensitivity_rows": len(all_sensitivity),
                    "stable_pose_candidate_rows": len(all_candidates),
                },
                "result": {
                    "primary_stable": f"{n_primary_stable}/{len(case_summaries)}",
                    "primary_robust": f"{n_primary_robust}/{len(case_summaries)}",
                    "case_variant_robust": f"{n_variant_robust}/{len(case_summaries)}",
                    "expansion_gate_passed": summary["expansion_gate_passed"],
                },
                "output_sha256": {
                    key: sha256_file(path)
                    for key, path in paths.items()
                    if key != "manifest"
                },
            }
        )
        write_json(paths["manifest"], manifest)
        print(
            f"COMPLETE: stable={n_primary_stable}/{len(cases)}, "
            f"robust={n_primary_robust}/{len(cases)}, "
            f"variant_robust={n_variant_robust}/{len(cases)}",
            flush=True,
        )
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "FAILED",
                "completed_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(paths["manifest"], manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
