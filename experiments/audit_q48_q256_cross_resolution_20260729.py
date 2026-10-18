#!/usr/bin/env python3
"""Minimal paired Q48--Q256 voxel-FEM audit for a delivered ShadowForge model.

The input is a native Q256 delivery.  For a pipeline output, a supplied
manifest transforms ``selected_display_voxels256.npy`` to the selected print
orientation exactly as the structural screening path does.  Reconstructed
sealed-test states may instead declare ``canonical_y_standing``: their arrays
already use the historical analysis frame with gravity along negative axis 1,
so no display transform is applied.  Q48 is then obtained with PyTorch
adaptive max pooling, matching
``shadowforge_pipeline.unified_run.downsample_max_to``.

Only the spatial discretization changes:

* normalized physical span = 48;
* h48 = 1 and h256 = 48/256;
* gravity uses the repository voxel-FEM convention, array axis 1;
* the clamped support is physically matched: 3 Q48 layers and 16 Q256 layers;
* material, gravity, yield stress, solver tolerance, and safety threshold are
  identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural.analysis.fem_solver import voxel_fem_solve  # noqa: E402


SCHEMA = "shadowforge_q48_q256_cross_resolution_voxel_fem_audit_v1"
PHYSICAL_SPAN = 48.0
GRAVITY_AXIS = 1
H48 = 1.0
H256 = PHYSICAL_SPAN / 256.0
SUPPORT_LAYERS_48 = 3
SUPPORT_LAYERS_256 = 16
E_PA = 3.5e9
NU = 0.36
MATERIAL_DENSITY = 1.24e3
GRAVITY = -9.81
YIELD_PA = 50.0e6
TAU = 1.3
DENSITY_THRESHOLD = 0.3
CORNER_OFFSETS = (
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
)
MAX_DOF_COUPLINGS = 81
MEMORY_HEADROOM_FACTOR = 1.25
AMG_HIERARCHY_CSR_MULTIPLIER = 3.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_binary_occupancy(occupancy: np.ndarray) -> str:
    """Hash a canonical uint8 payload, independent of NPY container metadata."""
    payload = np.ascontiguousarray(occupancy, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(payload.shape).encode("ascii"))
    digest.update(b"|uint8|")
    digest.update(memoryview(payload))
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_binary_q256(path: Path) -> np.ndarray:
    raw = np.load(path, allow_pickle=False)
    if raw.shape != (256, 256, 256):
        raise ValueError(f"Expected a 256^3 volume, got {raw.shape}: {path}")
    if not np.isfinite(raw).all():
        raise ValueError(f"Volume contains non-finite values: {path}")
    volume = np.ascontiguousarray(raw > 0.5)
    if not volume.any():
        raise ValueError(f"Volume is empty: {path}")
    return volume


def selected_orientation(manifest_path: Path | None, explicit: str) -> str:
    manifest_orientation: str | None = None
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = manifest.get("selected")
        if not isinstance(selected, dict):
            raise ValueError(
                f"Manifest selected record is not an object: {manifest_path}"
            )
        value = selected.get("print_orientation")
        if value:
            manifest_orientation = str(value)

    if explicit == "auto":
        if manifest_path is None:
            raise ValueError("--orientation auto requires --manifest")
        if manifest_orientation is None:
            raise ValueError(
                f"Manifest has no selected.print_orientation: {manifest_path}"
            )
        return manifest_orientation

    if (
        manifest_orientation is not None
        and explicit != manifest_orientation
    ):
        raise ValueError(
            f"Explicit orientation {explicit!r} conflicts with manifest "
            f"selected.print_orientation {manifest_orientation!r}: {manifest_path}"
        )
    return explicit


def orient_for_print(volume: np.ndarray, orientation: str) -> np.ndarray:
    mappings = {
        "canonical_y_standing": lambda v: v,
        "display_z_up": lambda v: v,
        "display_z_down": lambda v: v[:, :, ::-1],
        "display_x_up": lambda v: np.transpose(v, (1, 2, 0)),
        "display_x_down": lambda v: np.transpose(v, (1, 2, 0))[:, :, ::-1],
        "display_y_up": lambda v: np.transpose(v, (0, 2, 1)),
        "display_y_down": lambda v: np.transpose(v, (0, 2, 1))[:, :, ::-1],
    }
    if orientation not in mappings:
        raise ValueError(
            f"Unsupported print orientation {orientation!r}; "
            f"expected one of {sorted(mappings)}"
        )
    return np.ascontiguousarray(mappings[orientation](volume), dtype=bool)


def adaptive_max_pool_q48(volume256: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(volume256, dtype=np.float32)
    tensor = torch.from_numpy(source)[None, None]
    pooled = torch.nn.functional.adaptive_max_pool3d(
        tensor, output_size=(48, 48, 48)
    )[0, 0]
    return np.ascontiguousarray(pooled.numpy() >= 0.5)


def support_bounds(
    occupancy: np.ndarray,
    support_layers: int,
) -> tuple[int, int]:
    if occupancy.ndim != 3 or len(set(occupancy.shape)) != 1:
        raise ValueError(f"Expected a cubic occupancy, got {occupancy.shape}")
    if support_layers <= 0:
        raise ValueError("support_layers must be positive")
    nonempty_y = np.flatnonzero(occupancy.any(axis=(0, 2)))
    if nonempty_y.size == 0:
        raise ValueError("Cannot support an empty occupancy")
    y0 = int(nonempty_y[0])
    return y0, min(occupancy.shape[1], y0 + support_layers)


def element_node_mask(occupancy: np.ndarray) -> np.ndarray:
    """Return the exact active Hex8 node mask without assembling stiffness."""
    nx, ny, nz = occupancy.shape
    nodes = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    for di, dj, dk in CORNER_OFFSETS:
        nodes[
            di : di + nx,
            dj : dj + ny,
            dk : dk + nz,
        ] |= occupancy
    return nodes


def make_load(
    occupancy: np.ndarray,
    cell_size: float,
    support_layers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    resolution = occupancy.shape[0]
    density = occupancy.astype(np.float32)
    force = np.zeros((3, resolution, resolution, resolution), dtype=np.float32)
    force[GRAVITY_AXIS] = (
        density
        * np.float32(MATERIAL_DENSITY)
        * np.float32(GRAVITY)
        * np.float32(cell_size**3)
    )

    y0, y1 = support_bounds(occupancy, support_layers)
    fixed = np.zeros((1, resolution, resolution, resolution), dtype=np.float32)
    fixed[0, :, y0:y1, :] = occupancy[:, y0:y1, :]
    if not np.count_nonzero(fixed):
        raise RuntimeError("The matched support mask is empty")

    occupied = int(np.count_nonzero(occupancy))
    return force, fixed, {
        "resolution": resolution,
        "cell_size": float(cell_size),
        "cell_volume": float(cell_size**3),
        "gravity_axis": GRAVITY_AXIS,
        "support_layers": int(support_layers),
        "support_physical_thickness": float(support_layers * cell_size),
        "support_y_start": y0,
        "support_y_stop_exclusive": y1,
        "fixed_voxels": int(np.count_nonzero(fixed)),
        "occupied_voxels": occupied,
        "occupied_physical_volume": float(occupied * cell_size**3),
        "material_mass_under_declared_units": float(
            occupied * cell_size**3 * MATERIAL_DENSITY
        ),
        "total_gravity_force_n": float(
            force[GRAVITY_AXIS].sum(dtype=np.float64)
        ),
    }


def solve_one(
    occupancy: np.ndarray,
    cell_size: float,
    support_layers: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    force, fixed, load_meta = make_load(occupancy, cell_size, support_layers)
    started = time.time()
    result = voxel_fem_solve(
        density=occupancy.astype(np.float32),
        force_field=force,
        fixed_mask=fixed,
        E=E_PA,
        nu=NU,
        cell_size=cell_size,
        density_threshold=DENSITY_THRESHOLD,
        solver=args.solver,
        iterative_rtol=args.iterative_rtol,
        iterative_maxiter=args.iterative_maxiter,
        verbose=args.verbose_fem,
    )
    wall_time = time.time() - started
    peak = float(result.get("vm_max", 0.0))
    if not np.isfinite(peak) or peak <= 0:
        raise RuntimeError(f"Invalid peak von Mises stress: {peak!r}")
    sf = float(YIELD_PA / peak)
    n_free_dofs = int(result["n_dofs"])
    effective_solver = (
        "iterative_amg_cg"
        if args.solver == "iterative"
        or (args.solver == "auto" and n_free_dofs > 80000)
        else "direct_spsolve"
    )
    return {
        **load_meta,
        "youngs_modulus_pa": E_PA,
        "poisson_ratio": NU,
        "material_density": MATERIAL_DENSITY,
        "gravity_m_s2": GRAVITY,
        "yield_stress_pa": YIELD_PA,
        "density_threshold": DENSITY_THRESHOLD,
        "tau": TAU,
        "peak_metric": "maximum element-center von Mises stress",
        "peak_von_mises_pa": peak,
        "peak_von_mises_mpa": float(peak / 1.0e6),
        "safety_factor": sf,
        "safe_at_tau": bool(sf >= TAU),
        "n_elements_connected": int(result["n_elements"]),
        "n_elements_total": int(result.get("n_elements_total", result["n_elements"])),
        "n_disconnected_elements": int(result.get("n_disconnected", 0)),
        "n_free_dofs": n_free_dofs,
        "solver_requested": args.solver,
        "solver_effective": effective_solver,
        "iterative_rtol": float(args.iterative_rtol),
        "iterative_maxiter": int(args.iterative_maxiter),
        "solver_converged": True,
        "linear_solve_time_s": float(result["solve_time_s"]),
        "fem_total_time_s": float(result["total_time_s"]),
        "wall_time_s": float(wall_time),
    }


def memory_preflight(volume256: np.ndarray) -> dict[str, Any]:
    resolution = volume256.shape[0]
    active = int(np.count_nonzero(volume256))
    y0, y1 = support_bounds(volume256, SUPPORT_LAYERS_256)

    active_node_mask = element_node_mask(volume256)
    n_active_nodes = int(np.count_nonzero(active_node_mask))
    fixed_voxels = np.zeros_like(volume256, dtype=bool)
    fixed_voxels[:, y0:y1, :] = volume256[:, y0:y1, :]
    fixed_node_mask = element_node_mask(fixed_voxels)
    n_fixed_nodes = int(np.count_nonzero(fixed_node_mask & active_node_mask))
    del fixed_node_mask, fixed_voxels, active_node_mask

    total_dofs = 3 * n_active_nodes
    free_dofs = 3 * (n_active_nodes - n_fixed_nodes)

    # fem_solver.py materializes rows, cols, and values, each with 576 entries
    # per active Hex8 element and 8-byte storage.
    assembly_triplet_bytes = active * 576 * 8 * 3
    element_index_bytes = active * (8 * 8 * 2 + 24 * 8)
    node_map_bytes = (resolution + 1) ** 3 * 8

    # Each displacement DOF couples to at most 3 components on the 27 nodes in
    # its local 3x3x3 node neighborhood. SciPy uses float64 data and, for these
    # dimensions, int32 CSR indices/indptr.
    csr_total_upper_bytes = (
        total_dofs * MAX_DOF_COUPLINGS * (8 + 4)
        + (total_dofs + 1) * 4
    )
    csr_free_upper_bytes = (
        free_dofs * MAX_DOF_COUPLINGS * (8 + 4)
        + (free_dofs + 1) * 4
    )

    # Explicit dense arrays known to coexist at different solver stages:
    # density, 3-vector force, fixed mask, solid/component labels, and the
    # returned 3-displacement + 6-stress + von-Mises fields. This is still an
    # approximation and intentionally does not count Python/SciPy allocator
    # fragmentation.
    n_voxels = resolution**3
    dense_grid_planning_bytes = n_voxels * (
        4 + 3 * 4 + 4 + 1 + 4 + (3 + 6 + 1) * 4
    )

    # COO->CSR conversion can require workspace comparable to the COO
    # triplets. PyAMG hierarchy size is matrix- and coarsening-dependent, so a
    # declared multiplier is used for planning rather than presented as an
    # exact peak.
    assembly_stage_planning_bytes = (
        2 * assembly_triplet_bytes
        + csr_total_upper_bytes
        + element_index_bytes
        + node_map_bytes
        + dense_grid_planning_bytes
    )
    iterative_stage_planning_bytes = (
        csr_total_upper_bytes
        + csr_free_upper_bytes
        + int(AMG_HIERARCHY_CSR_MULTIPLIER * csr_free_upper_bytes)
        + element_index_bytes
        + node_map_bytes
        + dense_grid_planning_bytes
    )
    planning_peak_bytes = max(
        assembly_stage_planning_bytes,
        iterative_stage_planning_bytes,
    )
    recommended_available_bytes = int(
        np.ceil(planning_peak_bytes * MEMORY_HEADROOM_FACTOR)
    )

    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0]) * 1024
    mem_available = meminfo.get("MemAvailable", 0)
    memory_recommended = bool(
        mem_available > 0 and mem_available >= recommended_available_bytes
    )
    return {
        "active_q256_voxels": active,
        "active_q256_nodes": n_active_nodes,
        "fixed_q256_nodes": n_fixed_nodes,
        "total_q256_dofs": total_dofs,
        "estimated_free_q256_dofs": free_dofs,
        "support_y_start": y0,
        "support_y_stop_exclusive": y1,
        "estimated_rows_cols_values_bytes": assembly_triplet_bytes,
        "estimated_rows_cols_values_gib": assembly_triplet_bytes / 1024**3,
        "estimated_element_index_bytes": element_index_bytes,
        "estimated_node_map_bytes": node_map_bytes,
        "estimated_dense_grid_planning_bytes": dense_grid_planning_bytes,
        "estimated_csr_total_upper_bytes": csr_total_upper_bytes,
        "estimated_csr_free_upper_bytes": csr_free_upper_bytes,
        "assembly_stage_planning_bytes": assembly_stage_planning_bytes,
        "iterative_stage_planning_bytes": iterative_stage_planning_bytes,
        "planning_peak_bytes": planning_peak_bytes,
        "planning_peak_gib": planning_peak_bytes / 1024**3,
        "memory_headroom_factor": MEMORY_HEADROOM_FACTOR,
        "recommended_available_bytes": recommended_available_bytes,
        "recommended_available_gib": recommended_available_bytes / 1024**3,
        "mem_available_bytes_at_start": mem_available,
        "mem_available_gib_at_start": mem_available / 1024**3,
        "swap_free_bytes_at_start": meminfo.get("SwapFree", 0),
        "memory_recommended_to_start_q256": memory_recommended,
        "note": (
            "The rows/cols/values estimate is an exact lower bound for the "
            "current vectorized assembler. Planning peaks use CSR sparsity "
            "upper bounds, one triplet-sized conversion workspace, and a "
            f"{AMG_HIERARCHY_CSR_MULTIPLIER:g}x free-CSR AMG hierarchy "
            "allowance; they are conservative planning estimates, not measured "
            "peak RSS. Swap is reported but never treated as safe headroom."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--orientation",
        default="auto",
        choices=(
            "auto",
            "canonical_y_standing",
            "display_z_up",
            "display_z_down",
            "display_x_up",
            "display_x_down",
            "display_y_up",
            "display_y_down",
        ),
    )
    parser.add_argument(
        "--solver", choices=("auto", "direct", "iterative"), default="iterative"
    )
    parser.add_argument("--iterative-rtol", type=float, default=1.0e-8)
    parser.add_argument("--iterative-maxiter", type=int, default=12000)
    parser.add_argument("--verbose-fem", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--q48-only",
        action="store_true",
        help=(
            "Run and record only the deployment-consistent Q48 solve. "
            "This is a diagnostic label and not a completed paired audit."
        ),
    )
    parser.add_argument(
        "--allow-memory-risk",
        action="store_true",
        help=(
            "Permit a full run when the conservative Q256 memory preflight "
            "exceeds currently available RAM. Never affects --preflight-only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else None
    output = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if manifest_path is not None and not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if args.iterative_rtol <= 0:
        raise ValueError("--iterative-rtol must be positive")
    if args.iterative_maxiter <= 0:
        raise ValueError("--iterative-maxiter must be positive")
    if args.preflight_only and args.q48_only:
        raise ValueError("--preflight-only and --q48-only are mutually exclusive")
    if not np.isclose(48 * H48, 256 * H256, rtol=0.0, atol=1.0e-12):
        raise AssertionError("Q48 and Q256 physical spans do not match")
    if not np.isclose(
        SUPPORT_LAYERS_48 * H48,
        SUPPORT_LAYERS_256 * H256,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("Q48 and Q256 support thicknesses do not match")

    display256 = load_binary_q256(input_path)
    orientation = selected_orientation(manifest_path, args.orientation)
    print256 = orient_for_print(display256, orientation)
    reconstructed_canonical = orientation == "canonical_y_standing"
    input_coordinate_frame = (
        "canonical historical structural-analysis frame"
        if reconstructed_canonical
        else "display"
    )
    analysis_coordinate_frame = (
        "canonical standing orientation; gravity along negative array axis 1"
        if reconstructed_canonical
        else "manifest-selected or explicitly declared print orientation"
    )
    q48 = adaptive_max_pool_q48(print256)
    preflight = memory_preflight(print256)
    print256_payload_sha256 = sha256_binary_occupancy(print256)
    q48_payload_sha256 = sha256_binary_occupancy(q48)

    if args.preflight_only:
        memory_ok = preflight["memory_recommended_to_start_q256"]
        print(
            json.dumps(
                {
                    "status": (
                        "PREFLIGHT_MEMORY_PASS_WITH_CAVEATS"
                        if memory_ok
                        else "PREFLIGHT_MEMORY_RISK"
                    ),
                    "case": args.case,
                    "orientation": orientation,
                    "physical_span": PHYSICAL_SPAN,
                    "cell_sizes": {"48": H48, "256": H256},
                    "gravity_axis": GRAVITY_AXIS,
                    "support_layers": {
                        "48": SUPPORT_LAYERS_48,
                        "256": SUPPORT_LAYERS_256,
                    },
                    "support_physical_thickness": {
                        "48": SUPPORT_LAYERS_48 * H48,
                        "256": SUPPORT_LAYERS_256 * H256,
                    },
                    "q48_occupied": int(q48.sum()),
                    "q256_occupied": int(print256.sum()),
                    "oriented_q256_payload_sha256": print256_payload_sha256,
                    "q48_payload_sha256": q48_payload_sha256,
                    "memory_preflight": preflight,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if (
        not args.q48_only
        and not preflight["memory_recommended_to_start_q256"]
        and not args.allow_memory_risk
    ):
        raise RuntimeError(
            "Q256 memory preflight exceeds currently available RAM; refusing "
            "to start FEM. Re-run --preflight-only after freeing memory, or "
            "explicitly pass --allow-memory-risk to acknowledge the risk."
        )

    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    metadata = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "run_scope": "Q48_ONLY" if args.q48_only else "PAIRED_Q48_Q256",
        "started_at_utc": utc_now(),
        "case": args.case,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "manifest": str(manifest_path) if manifest_path else None,
        "manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
        "print_orientation": orientation,
        "input_coordinate_frame": input_coordinate_frame,
        "analysis_coordinate_frame": analysis_coordinate_frame,
        "oriented_q256_occupied_voxels": int(print256.sum()),
        "q48_occupied_voxels": int(q48.sum()),
        "oriented_q256_payload_sha256": print256_payload_sha256,
        "q48_payload_sha256": q48_payload_sha256,
        "q48_derivation": (
            "torch adaptive_max_pool3d of the analysis-oriented native Q256 "
            "delivery, matching shadowforge_pipeline.unified_run.downsample_max_to"
        ),
        "protocol": {
            "physical_span": PHYSICAL_SPAN,
            "cell_sizes": {"48": H48, "256": H256},
            "gravity_axis": GRAVITY_AXIS,
            "support_layers": {"48": SUPPORT_LAYERS_48, "256": SUPPORT_LAYERS_256},
            "support_physical_thickness": 3.0,
            "youngs_modulus_pa": E_PA,
            "poisson_ratio": NU,
            "material_density": MATERIAL_DENSITY,
            "gravity": GRAVITY,
            "yield_stress_pa": YIELD_PA,
            "tau": TAU,
            "peak_metric": "maximum element-center von Mises stress",
            "solver": args.solver,
            "iterative_rtol": args.iterative_rtol,
            "iterative_maxiter": args.iterative_maxiter,
        },
        "memory_preflight": preflight,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    write_json(output / "metadata.json", metadata)
    write_json(
        output / "status.json",
        {"status": "RUNNING", "stage": "Q48", "updated_at_utc": utc_now()},
    )

    try:
        q48_result = solve_one(q48, H48, SUPPORT_LAYERS_48, args)
        write_json(output / "q48_fem.json", q48_result)
        if args.q48_only:
            metadata.update(
                {
                    "status": "Q48_ONLY_COMPLETE",
                    "completed_at_utc": utc_now(),
                    "q48_result": q48_result,
                    "interpretation": (
                        "Deployment-consistent Q48 diagnostic only; no Q256 "
                        "solve was requested and no cross-resolution endpoint "
                        "can be computed from this run."
                    ),
                }
            )
            write_json(output / "metadata.json", metadata)
            write_json(
                output / "status.json",
                {
                    "status": "Q48_ONLY_COMPLETE",
                    "stage": "DONE",
                    "updated_at_utc": utc_now(),
                },
            )
            print(
                json.dumps(
                    {
                        "status": "Q48_ONLY_COMPLETE",
                        "case": args.case,
                        "q48": q48_result,
                        "output": str(output),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return
        write_json(
            output / "status.json",
            {"status": "RUNNING", "stage": "Q256", "updated_at_utc": utc_now()},
        )

        q256_result = solve_one(print256, H256, SUPPORT_LAYERS_256, args)
        write_json(output / "q256_fem.json", q256_result)

        sf48 = float(q48_result["safety_factor"])
        sf256 = float(q256_result["safety_factor"])
        paired = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "completed_at_utc": utc_now(),
            "elapsed_s": float(time.time() - started),
            "case": args.case,
            "q48": q48_result,
            "q256": q256_result,
            "delta_sf256_minus_sf48": float(sf256 - sf48),
            "strict_conservative_sf48_le_sf256": bool(sf48 <= sf256),
            "q48_false_safe_at_tau_1p3": bool(sf48 >= TAU and sf256 < TAU),
            "classification_at_tau_1p3": {
                "q48_safe": bool(sf48 >= TAU),
                "q256_safe": bool(sf256 >= TAU),
            },
        }
        write_json(output / "paired_result.json", paired)
        metadata.update(
            {
                "status": "COMPLETE",
                "completed_at_utc": paired["completed_at_utc"],
                "elapsed_s": paired["elapsed_s"],
                "q48_result": q48_result,
                "q256_result": q256_result,
                "paired_result": paired,
            }
        )
        write_json(output / "metadata.json", metadata)
        write_json(
            output / "status.json",
            {
                "status": "COMPLETE",
                "stage": None,
                "updated_at_utc": utc_now(),
                "elapsed_s": paired["elapsed_s"],
                "sf48": sf48,
                "sf256": sf256,
                "strict_conservative": paired[
                    "strict_conservative_sf48_le_sf256"
                ],
                "false_safe": paired["q48_false_safe_at_tau_1p3"],
            },
        )
        print(json.dumps(paired, indent=2, ensure_ascii=False), flush=True)
    except Exception as exc:
        failure = {
            "status": "FAILED",
            "stage": json.loads(
                (output / "status.json").read_text(encoding="utf-8")
            ).get("stage"),
            "updated_at_utc": utc_now(),
            "elapsed_s": float(time.time() - started),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output / "status.json", failure)
        metadata.update(
            {
                "status": "FAILED",
                "failed_at_utc": failure["updated_at_utc"],
                "failed_stage": failure["stage"],
                "elapsed_s": failure["elapsed_s"],
                "error_type": failure["error_type"],
                "error": failure["error"],
            }
        )
        write_json(output / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()
