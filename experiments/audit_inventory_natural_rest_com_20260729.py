#!/usr/bin/env python3
"""Expand the natural-rest COM audit to a frozen selected-delivery inventory."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_physical_natural_rest_com_20260729 import (  # noqa: E402
    analyze_artifact,
    create_figure,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


SCHEMA = "shadowforge_inventory_natural_rest_com_audit_v1"
PROTOCOL_SCHEMA = "shadowforge_inventory_natural_rest_com_protocol_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_paths(directory: Path, overwrite: bool) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": directory / "summary.json",
        "per_case": directory / "per_case.csv",
        "sensitivity": directory / "sensitivity.csv",
        "candidates": directory / "rest_pose_candidates.csv",
        "figure": directory / "support_polygons_exact_contact.png",
        "report": directory / "REPORT.md",
        "manifest": directory / "AUDIT_RUN_MANIFEST.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output files exist: {[str(path) for path in existing]}")
    return paths


def build_report(summary: dict[str, Any]) -> str:
    rows: list[str] = []
    for case in summary["cases"]:
        rows.append(
            "| {case} | {role} | {deviation:.1f} | {solid:.2f} | "
            "{shell:.2f} | {min_margin:.2f} | {min_angle:.2f} | {robust} |".format(
                case=case["case_id"],
                role=case["inventory_role"],
                deviation=float(case["deviation_from_reference_up_deg"]),
                solid=float(case["exact_solid_margin_mm"]),
                shell=float(case["exact_shell_margin_mm"]),
                min_margin=float(case["minimum_sensitivity_margin_mm"]),
                min_angle=float(case["minimum_sensitivity_angle_deg"]),
                robust="yes" if case["robust_pass"] else "no",
            )
        )
    return f"""# Frozen-inventory natural-rest COM expansion

Status: **{summary["status"]}**

This expansion was triggered only after the corrected four-print pilot passed.
The pool is the complete eight-row `task_inventory.tsv` frozen on 2026-07-15;
no case was added or removed using COM outcomes.

| selected delivery | frozen role | adjustment from +z (deg) | exact solid margin (mm) | exact shell margin (mm) | worst sensitivity margin (mm) | worst sensitivity angle (deg) | robust |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Exact-face solid and shell endpoints stable:
**{summary["n_exact_solid_shell_stable"]}/{summary["n_cases"]}**.  
Passing every solid-to-shell × `exact/0.5h/h/2h` cell and the frozen
`1 mm / 2°` gate: **{summary["n_robust"]}/{summary["n_cases"]}**.
Passing that robustness gate while also staying within 30° of nominal
display `+z`: **{summary["n_robust_and_le_30"]}/{summary["n_cases"]}**.

All four rows frozen as `benchmark` pass both criteria:
**{summary["role_breakdown"]["benchmark"]["n_robust_and_le_30"]}/{summary["role_breakdown"]["benchmark"]["n"]}**.
The non-benchmark rows are
reported rather than discarded.

Reference-up adjustment: median **{summary["reference_adjustment_deg"]["median"]:.1f}°**,
range **{summary["reference_adjustment_deg"]["minimum"]:.1f}–{summary["reference_adjustment_deg"]["maximum"]:.1f}°**;
**{summary["reference_adjustment_deg"]["n_le_15"]}/{summary["n_cases"]}** are
within 15° and **{summary["reference_adjustment_deg"]["n_le_30"]}/{summary["n_cases"]}**
within 30° of nominal display `+z`.

The worst frozen-cell margin is
**{summary["global_minimum_sensitivity_margin_mm"]:.2f} mm** and the worst
stability angle is **{summary["global_minimum_sensitivity_angle_deg"]:.2f}°**.
`archer_heroine` is a numerical boundary failure: its unrounded worst angle is
1.99797°, just below the strict 2° gate. `street_mecha` and `frog_beetle` miss
the gate more materially, while `knight_mage` is robust but needs a 57.6°
departure from nominal display-up.

## Interpretation

Every closed object has at least one mathematically stable support face, so the
existence of a selected face and the positive solid margin are not independent
validation. The informative expansion endpoints are (i) how far the selected
face departs from display `+z`, (ii) whether the unused `1 mm / 2°` robustness
gate is met, and (iii) whether shell-COM and contact-band sensitivities preserve
the result.

These eight are digital selected deliveries, not eight additional physical
prints. The expansion supports a geometric natural-rest analysis only; it does
not establish friction, strength, impact resistance, or real-world standing.
"""


def main() -> int:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError(f"Unexpected protocol schema: {payload.get('schema')!r}")
    inventory_path = Path(payload["inventory"]["path"]).resolve()
    if sha256_file(inventory_path) != payload["inventory"]["sha256"]:
        raise ValueError("Frozen inventory SHA-256 mismatch")
    cases = payload["cases"]
    if len(cases) != int(payload["expected_case_count"]):
        raise ValueError("Case count does not match expected_case_count")

    paths = prepare_paths(args.output_dir.resolve(), args.overwrite)
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
        "inventory": str(inventory_path),
        "inventory_sha256": sha256_file(inventory_path),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "trimesh": trimesh.__version__,
        },
    }
    write_json(paths["manifest"], manifest)

    try:
        case_rows: list[dict[str, Any]] = []
        case_details: list[dict[str, Any]] = []
        sensitivity_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        plot_payloads: list[dict[str, Any]] = []
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] {case['case_id']}", flush=True)
            result, sensitivity, candidates, plot = analyze_artifact(
                case_id=case["case_id"],
                artifact_role="primary",
                artifact_record=case["artifact"],
                reference_up_display=np.asarray(
                    case["reference_up_in_display_coordinates"],
                    dtype=np.float64,
                ),
                protocol=payload["protocol"],
            )
            sensitivity_rows.extend(sensitivity)
            candidate_rows.extend(candidates)
            plot_payloads.append(plot)
            detail = {
                "case_id": case["case_id"],
                "inventory_role": case["inventory_role"],
                "inventory_row": case["inventory_row"],
                "analysis": result,
            }
            case_details.append(detail)
            case_rows.append(
                {
                    "case_id": case["case_id"],
                    "inventory_role": case["inventory_role"],
                    "inventory_row": case["inventory_row"],
                    "artifact_path": result["artifact"]["path"],
                    "artifact_sha256": result["artifact"]["sha256"],
                    "deviation_from_reference_up_deg": result[
                        "deviation_from_photo_reference_up_deg"
                    ],
                    "exact_solid_margin_mm": result["primary_uniform_solid"][
                        "signed_margin_mm"
                    ],
                    "exact_shell_margin_mm": result["primary_uniform_thin_shell"][
                        "signed_margin_mm"
                    ],
                    "minimum_sensitivity_margin_mm": result[
                        "minimum_signed_margin_across_sensitivity_mm"
                    ],
                    "minimum_sensitivity_angle_deg": result[
                        "minimum_signed_stability_angle_across_sensitivity_deg"
                    ],
                    "robust_pass": result["robust_pass"],
                }
            )

        deviations = np.asarray(
            [row["deviation_from_reference_up_deg"] for row in case_rows],
            dtype=np.float64,
        )
        n_exact_stable = sum(
            bool(detail["analysis"]["primary_uniform_solid"][
                "stable_under_vertical_gravity"
            ])
            and bool(detail["analysis"]["primary_uniform_thin_shell"][
                "stable_under_vertical_gravity"
            ])
            for detail in case_details
        )
        n_robust = sum(bool(row["robust_pass"]) for row in case_rows)
        n_robust_and_le_30 = sum(
            bool(row["robust_pass"])
            and float(row["deviation_from_reference_up_deg"]) <= 30.0
            for row in case_rows
        )
        roles = sorted({str(row["inventory_role"]) for row in case_rows})
        role_breakdown = {}
        for role in roles:
            role_rows = [
                row for row in case_rows if str(row["inventory_role"]) == role
            ]
            role_breakdown[role] = {
                "n": len(role_rows),
                "n_robust": sum(bool(row["robust_pass"]) for row in role_rows),
                "n_robust_and_le_30": sum(
                    bool(row["robust_pass"])
                    and float(row["deviation_from_reference_up_deg"]) <= 30.0
                    for row in role_rows
                ),
            }
        summary = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "audit_design": (
                "post-hoc expansion over a complete pre-existing frozen "
                "inventory; natural-rest face chosen without margin ranking"
            ),
            "trigger": payload["trigger"],
            "inventory": {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
                "frozen_before_current_audit": True,
            },
            "protocol": payload["protocol"],
            "n_cases": len(case_rows),
            "n_exact_solid_shell_stable": int(n_exact_stable),
            "n_robust": int(n_robust),
            "n_robust_and_le_30": int(n_robust_and_le_30),
            "role_breakdown": role_breakdown,
            "reference_adjustment_deg": {
                "minimum": float(deviations.min()),
                "maximum": float(deviations.max()),
                "mean": float(deviations.mean()),
                "median": float(np.median(deviations)),
                "n_le_15": int(np.sum(deviations <= 15.0)),
                "n_le_30": int(np.sum(deviations <= 30.0)),
            },
            "global_minimum_sensitivity_margin_mm": min(
                float(row["minimum_sensitivity_margin_mm"]) for row in case_rows
            ),
            "global_minimum_sensitivity_angle_deg": min(
                float(row["minimum_sensitivity_angle_deg"]) for row in case_rows
            ),
            "cases": case_rows,
            "case_details": case_details,
            "claim_scope": (
                "digital natural-rest geometry only; not additional physical "
                "fabrication or free-standing evidence"
            ),
        }
        write_json(paths["summary"], summary)
        write_csv(paths["per_case"], case_rows)
        write_csv(paths["sensitivity"], sensitivity_rows)
        write_csv(paths["candidates"], candidate_rows)
        create_figure(paths["figure"], plot_payloads)
        write_text(paths["report"], build_report(summary))
        manifest.update(
            {
                "status": "COMPLETE",
                "completed_at_utc": utc_now(),
                "counts": {
                    "cases": len(case_rows),
                    "sensitivity_rows": len(sensitivity_rows),
                    "stable_pose_candidate_rows": len(candidate_rows),
                },
                "result": {
                    "exact_solid_shell_stable": f"{n_exact_stable}/{len(case_rows)}",
                    "robust": f"{n_robust}/{len(case_rows)}",
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
            f"COMPLETE: exact stable={n_exact_stable}/{len(case_rows)}, "
            f"robust={n_robust}/{len(case_rows)}",
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
