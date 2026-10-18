#!/usr/bin/env python3
"""Audit whether an exact-max-pooled Q48 proxy is conservative against Q96.

The audit compares the same native 96^3 delivery at two voxel-FEM
discretizations over a fixed span of 48 solver length units:

* Q96 is the native binary occupancy with h=0.5.
* Q48 is exact non-overlapping 2x2x2 occupancy max pooling with h=1.0.
* Gravity acts along array/spatial axis 1.
* The clamped support has the same physical thickness: 3 Q48 layers and
  6 Q96 layers.

The repository's Hex8 solver natively accepts ``cell_size``.  It uses that
value in both element stiffness assembly and center-point stress recovery.
Body force is constructed here as rho * g * h^3 per occupied voxel.  No
post-hoc or equivalent stress scaling is applied.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural.analysis.fem_solver import voxel_fem_solve  # noqa: E402


SCHEMA = "shadowforge_q48_q96_cross_resolution_voxel_fem_audit_v1"
DEFAULT_INPUT_GLOB = "PRINT_DELIVERY_COMPLETE/*/*_voxels96.npy"
EXPECTED_INPUT_COUNT = 5

PHYSICAL_SPAN = 48.0
GRAVITY_AXIS = 1
H_BY_RESOLUTION = {48: 1.0, 96: 0.5}
SUPPORT_LAYERS_BY_RESOLUTION = {48: 3, 96: 6}

YOUNGS_MODULUS_PA = 3.5e9
POISSON_RATIO = 0.36
MATERIAL_DENSITY_KG_M3 = 1.24e3
GRAVITY_M_S2 = -9.81
YIELD_STRESS_PA = 50.0e6
TAU = 1.3
DENSITY_THRESHOLD = 0.3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("q48_q96_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(output_dir / "audit.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def discover_inputs(pattern: str, case: str | None = None) -> list[Path]:
    if Path(pattern).is_absolute():
        raise ValueError("--input-glob must be relative to the repository root")
    paths = sorted(ROOT.glob(pattern))
    if len(paths) != EXPECTED_INPUT_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_INPUT_COUNT} native Q96 inputs for "
            f"{pattern!r}, found {len(paths)}"
        )
    if case is not None:
        selected = [path for path in paths if path.parent.name == case]
        if len(selected) != 1:
            available = ", ".join(path.parent.name for path in paths)
            raise RuntimeError(
                f"--case must exactly match one input parent directory; got "
                f"{case!r}. Available cases: {available}"
            )
        paths = selected
    return [path.resolve() for path in paths]


def load_native_q96(path: Path) -> np.ndarray:
    raw = np.load(path, allow_pickle=False)
    if raw.shape != (96, 96, 96):
        raise ValueError(f"{path}: expected shape (96, 96, 96), got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: occupancy contains non-finite values")
    unique = np.unique(raw)
    if not np.all(np.isin(unique, (0, 1))):
        raise ValueError(f"{path}: native occupancy must be binary, got {unique[:12]}")
    occupancy = np.ascontiguousarray(raw > 0)
    if not occupancy.any():
        raise ValueError(f"{path}: native occupancy is empty")
    return occupancy


def exact_max_pool_2x(occupancy96: np.ndarray) -> np.ndarray:
    """Non-overlapping 2x2x2 logical-OR pooling; no interpolation is used."""
    if occupancy96.shape != (96, 96, 96):
        raise ValueError(f"Expected Q96 shape, got {occupancy96.shape}")
    pooled = occupancy96.reshape(48, 2, 48, 2, 48, 2).max(axis=(1, 3, 5))
    pooled = np.ascontiguousarray(pooled, dtype=bool)

    # Every native occupied subvoxel must map to an occupied parent.
    parent_reexpanded = np.repeat(
        np.repeat(np.repeat(pooled, 2, axis=0), 2, axis=1), 2, axis=2
    )
    if np.any(occupancy96 & ~parent_reexpanded):
        raise AssertionError("Exact max pooling lost native occupied subvoxels")
    return pooled


def make_self_weight_load(
    occupancy: np.ndarray,
    cell_size: float,
    support_layers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build axis-1 gravity and a global bottom-band clamped support."""
    resolution = occupancy.shape[0]
    if occupancy.shape != (resolution, resolution, resolution):
        raise ValueError(f"Expected cubic occupancy, got {occupancy.shape}")
    if support_layers <= 0:
        raise ValueError("support_layers must be positive")

    density = occupancy.astype(np.float32)
    cell_volume = float(cell_size**3)
    force = np.zeros((3, resolution, resolution, resolution), dtype=np.float32)
    force[GRAVITY_AXIS] = (
        density
        * np.float32(MATERIAL_DENSITY_KG_M3)
        * np.float32(GRAVITY_M_S2)
        * np.float32(cell_volume)
    )

    y_nonempty = np.flatnonzero(occupancy.any(axis=(0, 2)))
    if y_nonempty.size == 0:
        raise ValueError("Cannot support an empty occupancy")
    y_start = int(y_nonempty[0])
    y_stop = min(resolution, y_start + support_layers)
    fixed_mask = np.zeros((1, resolution, resolution, resolution), dtype=np.float32)
    fixed_mask[0, :, y_start:y_stop, :] = occupancy[:, y_start:y_stop, :]

    fixed_voxels = int(np.count_nonzero(fixed_mask))
    if fixed_voxels == 0:
        raise AssertionError("Support mask has no fixed occupied voxels")

    occupied_volume = float(np.count_nonzero(occupancy) * cell_volume)
    load_meta = {
        "gravity_axis": GRAVITY_AXIS,
        "gravity_m_s2": GRAVITY_M_S2,
        "cell_size": float(cell_size),
        "cell_volume": cell_volume,
        "support_layers": int(support_layers),
        "support_physical_thickness": float(support_layers * cell_size),
        "support_y_start_index": y_start,
        "support_y_stop_exclusive": y_stop,
        "support_fixed_voxels": fixed_voxels,
        "occupied_voxels": int(np.count_nonzero(occupancy)),
        "occupied_volume": occupied_volume,
        "total_mass_kg_under_declared_units": float(
            MATERIAL_DENSITY_KG_M3 * occupied_volume
        ),
        "total_gravity_force_n": float(force[GRAVITY_AXIS].sum(dtype=np.float64)),
    }
    return force, fixed_mask, load_meta


def solve_grid(
    occupancy: np.ndarray,
    resolution: int,
    solver: str,
    iterative_rtol: float,
    iterative_maxiter: int,
    verbose_fem: bool,
) -> dict[str, Any]:
    expected_shape = (resolution, resolution, resolution)
    if occupancy.shape != expected_shape:
        raise ValueError(f"Q{resolution}: expected {expected_shape}, got {occupancy.shape}")
    cell_size = H_BY_RESOLUTION[resolution]
    support_layers = SUPPORT_LAYERS_BY_RESOLUTION[resolution]
    force, fixed_mask, load_meta = make_self_weight_load(
        occupancy, cell_size, support_layers
    )

    density = occupancy.astype(np.float32)
    started = time.time()
    fem = voxel_fem_solve(
        density=density,
        force_field=force,
        fixed_mask=fixed_mask,
        E=YOUNGS_MODULUS_PA,
        nu=POISSON_RATIO,
        cell_size=cell_size,
        density_threshold=DENSITY_THRESHOLD,
        solver=solver,
        iterative_rtol=iterative_rtol,
        iterative_maxiter=iterative_maxiter,
        verbose=verbose_fem,
    )
    wall_time = time.time() - started

    peak_stress_pa = float(fem.get("vm_max", 0.0))
    if not np.isfinite(peak_stress_pa) or peak_stress_pa <= 0.0:
        raise RuntimeError(
            f"Q{resolution}: invalid peak von Mises stress {peak_stress_pa!r}"
        )
    safety_factor = float(YIELD_STRESS_PA / peak_stress_pa)
    if not np.isfinite(safety_factor) or safety_factor <= 0.0:
        raise RuntimeError(f"Q{resolution}: invalid safety factor {safety_factor!r}")

    free_dofs = int(fem["n_dofs"])
    if solver == "auto":
        effective_solver = "iterative_amg_cg" if free_dofs > 80000 else "direct_spsolve"
    elif solver == "iterative":
        effective_solver = "iterative_amg_cg"
    else:
        effective_solver = "direct_spsolve"

    result = {
        "resolution": resolution,
        "grid_shape": list(expected_shape),
        **load_meta,
        "youngs_modulus_pa": YOUNGS_MODULUS_PA,
        "poisson_ratio": POISSON_RATIO,
        "material_density_kg_m3": MATERIAL_DENSITY_KG_M3,
        "yield_stress_pa": YIELD_STRESS_PA,
        "density_threshold": DENSITY_THRESHOLD,
        "solver_requested": solver,
        "solver_effective": effective_solver,
        "iterative_rtol": iterative_rtol,
        "iterative_maxiter": iterative_maxiter,
        "peak_von_mises_pa": peak_stress_pa,
        "peak_von_mises_mpa": float(peak_stress_pa / 1.0e6),
        "safety_factor": safety_factor,
        "safe_at_tau": bool(safety_factor >= TAU),
        "tau": TAU,
        "n_elements_connected": int(fem["n_elements"]),
        "n_elements_total": int(fem.get("n_elements_total", fem["n_elements"])),
        "n_disconnected_elements": int(fem.get("n_disconnected", 0)),
        "n_free_dofs": free_dofs,
        "linear_solve_time_s": float(fem["solve_time_s"]),
        "fem_total_time_s": float(fem["total_time_s"]),
        "wall_time_s": float(wall_time),
    }

    del fem, force, fixed_mask, density
    gc.collect()
    return result


def paired_result(
    case_name: str,
    input_path: Path,
    input_sha256: str,
    q48: dict[str, Any],
    q96: dict[str, Any],
) -> dict[str, Any]:
    sf48 = float(q48["safety_factor"])
    sf96 = float(q96["safety_factor"])
    delta = sf96 - sf48
    conservative = bool(sf48 <= sf96)
    false_safe = bool(sf48 >= TAU and sf96 < TAU)
    return {
        "case": case_name,
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "q48_derivation": "exact non-overlapping 2x2x2 occupancy max pooling from native Q96",
        "q48": q48,
        "q96": q96,
        "delta_sf96_minus_sf48": float(delta),
        "strict_conservative_sf48_le_sf96": conservative,
        "q48_false_safe_at_tau_1p3": false_safe,
        "classification_at_tau_1p3": {
            "q48_safe": bool(sf48 >= TAU),
            "q96_safe": bool(sf96 >= TAU),
        },
    }


CSV_FIELDS = [
    "case",
    "input_path",
    "occupied_voxels_q48",
    "occupied_voxels_q96",
    "occupied_volume_q48",
    "occupied_volume_q96",
    "peak_stress_q48_pa",
    "peak_stress_q96_pa",
    "peak_stress_q48_mpa",
    "peak_stress_q96_mpa",
    "sf_q48",
    "sf_q96",
    "delta_sf96_minus_sf48",
    "strict_conservative_sf48_le_sf96",
    "q48_safe_at_tau_1p3",
    "q96_safe_at_tau_1p3",
    "q48_false_safe_at_tau_1p3",
    "support_layers_q48",
    "support_layers_q96",
    "support_thickness_q48",
    "support_thickness_q96",
    "n_free_dofs_q48",
    "n_free_dofs_q96",
    "wall_time_q48_s",
    "wall_time_q96_s",
]


def csv_row(row: dict[str, Any]) -> dict[str, Any]:
    q48 = row["q48"]
    q96 = row["q96"]
    cls = row["classification_at_tau_1p3"]
    return {
        "case": row["case"],
        "input_path": row["input_path"],
        "occupied_voxels_q48": q48["occupied_voxels"],
        "occupied_voxels_q96": q96["occupied_voxels"],
        "occupied_volume_q48": q48["occupied_volume"],
        "occupied_volume_q96": q96["occupied_volume"],
        "peak_stress_q48_pa": q48["peak_von_mises_pa"],
        "peak_stress_q96_pa": q96["peak_von_mises_pa"],
        "peak_stress_q48_mpa": q48["peak_von_mises_mpa"],
        "peak_stress_q96_mpa": q96["peak_von_mises_mpa"],
        "sf_q48": q48["safety_factor"],
        "sf_q96": q96["safety_factor"],
        "delta_sf96_minus_sf48": row["delta_sf96_minus_sf48"],
        "strict_conservative_sf48_le_sf96": row[
            "strict_conservative_sf48_le_sf96"
        ],
        "q48_safe_at_tau_1p3": cls["q48_safe"],
        "q96_safe_at_tau_1p3": cls["q96_safe"],
        "q48_false_safe_at_tau_1p3": row["q48_false_safe_at_tau_1p3"],
        "support_layers_q48": q48["support_layers"],
        "support_layers_q96": q96["support_layers"],
        "support_thickness_q48": q48["support_physical_thickness"],
        "support_thickness_q96": q96["support_physical_thickness"],
        "n_free_dofs_q48": q48["n_free_dofs"],
        "n_free_dofs_q96": q96["n_free_dofs"],
        "wall_time_q48_s": q48["wall_time_s"],
        "wall_time_q96_s": q96["wall_time_s"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_row(row))


def build_summary(
    rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    total_inputs: int,
) -> dict[str, Any]:
    return {
        "n_inputs": total_inputs,
        "n_completed_pairs": len(rows),
        "n_errors": len(errors),
        "n_strictly_conservative": sum(
            bool(row["strict_conservative_sf48_le_sf96"]) for row in rows
        ),
        "all_completed_pairs_strictly_conservative": bool(rows)
        and all(bool(row["strict_conservative_sf48_le_sf96"]) for row in rows),
        "n_q48_false_safe_at_tau_1p3": sum(
            bool(row["q48_false_safe_at_tau_1p3"]) for row in rows
        ),
        "any_q48_false_safe_at_tau_1p3": any(
            bool(row["q48_false_safe_at_tau_1p3"]) for row in rows
        ),
        "mean_delta_sf96_minus_sf48": (
            float(np.mean([row["delta_sf96_minus_sf48"] for row in rows]))
            if rows
            else None
        ),
        "min_delta_sf96_minus_sf48": (
            float(min(row["delta_sf96_minus_sf48"] for row in rows))
            if rows
            else None
        ),
    }


def metadata(args: argparse.Namespace, inputs: list[Path]) -> dict[str, Any]:
    package_versions: dict[str, str | None] = {}
    for name in ("numpy", "scipy", "pyamg"):
        try:
            module = __import__(name)
            package_versions[name] = str(module.__version__)
        except Exception:
            package_versions[name] = None
    return {
        "schema": SCHEMA,
        "started_at_utc": utc_now(),
        "repository_root": str(ROOT),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "package_versions": package_versions,
        "input_glob": args.input_glob,
        "case_filter": args.case,
        "inputs": [str(path) for path in inputs],
        "protocol": {
            "physical_span": PHYSICAL_SPAN,
            "length_unit_note": (
                "Declared normalized solver-length protocol: both grids span 48; "
                "h48=1 and h96=0.5."
            ),
            "h_by_resolution": {str(k): v for k, v in H_BY_RESOLUTION.items()},
            "q48_derivation": (
                "exact non-overlapping 2x2x2 occupancy max pooling from native Q96"
            ),
            "gravity_axis": GRAVITY_AXIS,
            "gravity_m_s2": GRAVITY_M_S2,
            "support_layers_by_resolution": {
                str(k): v for k, v in SUPPORT_LAYERS_BY_RESOLUTION.items()
            },
            "support_physical_thickness": 3.0,
            "support_definition": (
                "occupied voxels in the first global nonempty y band; "
                "3 layers at Q48 and 6 layers at Q96"
            ),
            "youngs_modulus_pa": YOUNGS_MODULUS_PA,
            "poisson_ratio": POISSON_RATIO,
            "material_density_kg_m3": MATERIAL_DENSITY_KG_M3,
            "yield_stress_pa": YIELD_STRESS_PA,
            "tau": TAU,
            "peak_metric": "maximum element-center von Mises stress",
        },
        "voxel_size_handling": {
            "api_supports_cell_size": True,
            "native_api_parameter": "voxel_fem_solve(..., cell_size=h)",
            "stiffness_evidence": (
                "hex8_stiffness(E, nu, h) uses h in the Jacobian, B matrix, "
                "and integration determinant"
            ),
            "stress_recovery_evidence": "_compute_B_center(h) uses the same h",
            "load_evidence": "body force per occupied voxel is rho*g*h^3",
            "equivalent_posthoc_scaling_used": False,
        },
        "solver": {
            "requested": args.solver,
            "iterative_rtol": args.iterative_rtol,
            "iterative_maxiter": args.iterative_maxiter,
            "repository_auto_rule": (
                "direct spsolve at <=80000 free DOFs; AMG-preconditioned CG above"
            ),
        },
    }


def run_smoke(args: argparse.Namespace) -> None:
    """Exercise native-input pooling plus a small physical Hex8 paired solve."""
    input_checks = []
    for path in discover_inputs(args.input_glob, args.case):
        q96 = load_native_q96(path)
        q48 = exact_max_pool_2x(q96)
        input_checks.append(
            {
                "case": path.parent.name,
                "q96_shape": list(q96.shape),
                "q48_shape": list(q48.shape),
                "q96_occupied": int(q96.sum()),
                "q48_occupied": int(q48.sum()),
            }
        )

    coarse = np.zeros((8, 8, 8), dtype=bool)
    coarse[2:6, 1:7, 2:6] = True
    fine = np.repeat(np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1), 2, axis=2)
    if not np.array_equal(
        fine.reshape(8, 2, 8, 2, 8, 2).max(axis=(1, 3, 5)), coarse
    ):
        raise AssertionError("Synthetic exact pooling round trip failed")

    smoke_h = {8: 1.0, 16: 0.5}
    smoke_layers = {8: 3, 16: 6}
    smoke_results = {}
    for resolution, occupancy in ((8, coarse), (16, fine)):
        force, fixed_mask, load_meta = make_self_weight_load(
            occupancy,
            cell_size=smoke_h[resolution],
            support_layers=smoke_layers[resolution],
        )
        fem = voxel_fem_solve(
            occupancy.astype(np.float32),
            force,
            fixed_mask,
            E=YOUNGS_MODULUS_PA,
            nu=POISSON_RATIO,
            cell_size=smoke_h[resolution],
            density_threshold=DENSITY_THRESHOLD,
            solver="direct",
            verbose=False,
        )
        peak = float(fem.get("vm_max", 0.0))
        if not np.isfinite(peak) or peak <= 0:
            raise RuntimeError(f"Synthetic Q{resolution} smoke solve failed: {peak}")
        smoke_results[f"q{resolution}"] = {
            "peak_von_mises_pa": peak,
            "safety_factor": float(YIELD_STRESS_PA / peak),
            "support_physical_thickness": load_meta[
                "support_physical_thickness"
            ],
            "n_free_dofs": int(fem["n_dofs"]),
        }
    print(
        json.dumps(
            {
                "status": "SMOKE_PASS",
                "input_checks": input_checks,
                "synthetic_paired_fem": smoke_results,
                "api_native_cell_size": True,
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_audit(args: argparse.Namespace) -> None:
    inputs = discover_inputs(args.input_glob, args.case)
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir()
    logger = setup_logger(output_dir)

    meta = metadata(args, inputs)
    write_json(output_dir / "metadata.json", meta)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.time()

    def checkpoint(status: str, current_case: str | None = None) -> None:
        summary = build_summary(rows, errors, len(inputs))
        payload = {
            "schema": SCHEMA,
            "status": status,
            "current_case": current_case,
            "updated_at_utc": utc_now(),
            "elapsed_s": float(time.time() - started),
            "summary": summary,
            "cases": rows,
            "errors": errors,
        }
        write_json(output_dir / "audit_results.json", payload)
        write_json(
            output_dir / "status.json",
            {
                "status": status,
                "current_case": current_case,
                "updated_at_utc": payload["updated_at_utc"],
                "elapsed_s": payload["elapsed_s"],
                "summary": summary,
            },
        )
        write_csv(output_dir / "audit_results.csv", rows)

    checkpoint("RUNNING")
    logger.info(
        "Starting %s with %d inputs; output=%s",
        SCHEMA,
        len(inputs),
        output_dir,
    )

    for index, input_path in enumerate(inputs, start=1):
        case_name = input_path.parent.name
        case_dir = cases_dir / case_name
        case_dir.mkdir()
        logger.info("[%d/%d] %s: loading native Q96", index, len(inputs), case_name)
        checkpoint("RUNNING", case_name)
        try:
            occupancy96 = load_native_q96(input_path)
            occupancy48 = exact_max_pool_2x(occupancy96)
            input_record = {
                "case": case_name,
                "input_path": str(input_path),
                "input_sha256": sha256_file(input_path),
                "q96_shape": list(occupancy96.shape),
                "q48_shape": list(occupancy48.shape),
                "q96_occupied": int(occupancy96.sum()),
                "q48_occupied": int(occupancy48.sum()),
                "q48_derivation": (
                    "exact non-overlapping 2x2x2 occupancy max pooling"
                ),
            }
            write_json(case_dir / "input.json", input_record)

            logger.info("[%d/%d] %s: solving Q48", index, len(inputs), case_name)
            q48 = solve_grid(
                occupancy48,
                48,
                args.solver,
                args.iterative_rtol,
                args.iterative_maxiter,
                args.verbose_fem,
            )
            write_json(case_dir / "q48_fem.json", q48)
            logger.info(
                "[%d/%d] %s: Q48 peak=%.6g Pa SF=%.6g",
                index,
                len(inputs),
                case_name,
                q48["peak_von_mises_pa"],
                q48["safety_factor"],
            )

            logger.info("[%d/%d] %s: solving native Q96", index, len(inputs), case_name)
            q96 = solve_grid(
                occupancy96,
                96,
                args.solver,
                args.iterative_rtol,
                args.iterative_maxiter,
                args.verbose_fem,
            )
            write_json(case_dir / "q96_fem.json", q96)
            row = paired_result(
                case_name,
                input_path,
                input_record["input_sha256"],
                q48,
                q96,
            )
            write_json(case_dir / "paired_result.json", row)
            rows.append(row)
            logger.info(
                "[%d/%d] %s: Q96 peak=%.6g Pa SF=%.6g delta=%.6g "
                "conservative=%s false_safe=%s",
                index,
                len(inputs),
                case_name,
                q96["peak_von_mises_pa"],
                q96["safety_factor"],
                row["delta_sf96_minus_sf48"],
                row["strict_conservative_sf48_le_sf96"],
                row["q48_false_safe_at_tau_1p3"],
            )
            del occupancy96, occupancy48
            gc.collect()
            checkpoint("RUNNING", case_name)
        except Exception as exc:
            error = {
                "case": case_name,
                "input_path": str(input_path),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "time_utc": utc_now(),
            }
            errors.append(error)
            write_json(case_dir / "error.json", error)
            logger.exception("[%d/%d] %s failed", index, len(inputs), case_name)
            checkpoint("RUNNING_WITH_ERRORS", case_name)

    final_status = "COMPLETED" if not errors else "COMPLETED_WITH_ERRORS"
    checkpoint(final_status)
    logger.info(
        "Finished: status=%s completed=%d/%d conservative=%d false_safe=%d "
        "elapsed=%.1fs",
        final_status,
        len(rows),
        len(inputs),
        sum(bool(row["strict_conservative_sf48_le_sf96"]) for row in rows),
        sum(bool(row["q48_false_safe_at_tau_1p3"]) for row in rows),
        time.time() - started,
    )
    if errors:
        raise RuntimeError(f"{len(errors)} case(s) failed; see {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help="Repository-relative glob resolving to exactly five native Q96 inputs.",
    )
    parser.add_argument(
        "--case",
        default=None,
        help=(
            "Optionally run exactly one case by its input parent-directory name. "
            "Omitting this option preserves the default five-case audit."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/q48_cross_resolution_audit_20260729",
        help="New output directory. Existing directories are never overwritten.",
    )
    parser.add_argument(
        "--solver",
        choices=("auto", "direct", "iterative"),
        default="auto",
        help="Voxel-FEM linear solver selection.",
    )
    parser.add_argument("--iterative-rtol", type=float, default=1.0e-8)
    parser.add_argument("--iterative-maxiter", type=int, default=4000)
    parser.add_argument("--verbose-fem", action="store_true")
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Validate inputs, pooling, support thickness, and a tiny paired FEM solve.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_only:
        run_smoke(args)
    else:
        run_audit(args)


if __name__ == "__main__":
    main()
