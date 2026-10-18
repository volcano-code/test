#!/usr/bin/env python3
"""Reconstruct a native-Q256 delivery behind an archived raw-PINO false-safe.

This is a provenance/audit utility, not a FEM runner.  It reproduces the exact
candidate and delivered-proxy construction used by
``build_taskgrouped_decision_fem_dataset_20260711.py``:

1. reload the frozen source case and silhouettes;
2. regenerate the requested Q48 candidate with ``make_candidates``;
3. rebuild the calibrated Q256 visual hull;
4. nearest-neighbor lift the Q48 core, add the two-voxel delivery shell, and
   keep the largest 6-connected component;
5. compare both the historical rounded-edge conservative pool and the current
   deployed PyTorch adaptive max pool against the archived Q48 density.

The script saves the native Q256 delivery and a JSON provenance record.  It
deliberately performs no FEM solve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import binary_erosion


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "generation")]

from build_taskgrouped_decision_fem_dataset_20260711 import (  # noqa: E402
    S6,
    conservative_pool,
    largest_component,
    load_masks,
    upsample_nearest,
)
from candidate_selection_batch import make_candidates  # noqa: E402
from voxel_ops import build_tfs_visual_hull, calibrate_tfs_projection  # noqa: E402


SCHEMA = "shadowforge_raw_pino_false_safe_q256_reconstruction_v1"
DEFAULT_SOURCE = (
    ROOT
    / "results/candidate_selection_400case_iou_dice_20260708"
    / "candidate_selection_100case_iou_dice.json"
)
DEFAULT_INDEX = (
    ROOT / "data/taskgrouped_strict_pino_delivered_v7_boundary_20260715/index.json"
)
DEFAULT_PINO_AUDIT = (
    ROOT
    / "results/strict_physics_residual_pino_v16_shadowart_scale_20260718_dualside_acs"
    / "sealed_row_audit.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_binary_occupancy(volume: np.ndarray) -> str:
    """Hash canonical binary payload plus shape, independent of file format."""
    payload = np.ascontiguousarray(volume, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(payload.shape).encode("ascii"))
    digest.update(b"|uint8|")
    digest.update(memoryview(payload))
    return digest.hexdigest()


def sha256_array_payload(array: np.ndarray) -> str:
    """Hash exact C-order array bytes together with shape and dtype."""
    payload = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(payload.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(payload.dtype.str.encode("ascii"))
    digest.update(b"|")
    digest.update(memoryview(payload))
    return digest.hexdigest()


def difference_summary(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {
            "shape_equal": False,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "bitwise_equal": False,
        }
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    only_a = int(np.count_nonzero(aa & ~bb))
    only_b = int(np.count_nonzero(bb & ~aa))
    different = only_a + only_b
    union = int(np.count_nonzero(aa | bb))
    return {
        "shape_equal": True,
        "bitwise_equal": bool(different == 0),
        "different_voxels": different,
        "only_a_voxels": only_a,
        "only_b_voxels": only_b,
        "fraction_of_grid_different": float(different / aa.size),
        "iou": float(np.count_nonzero(aa & bb) / union) if union else 1.0,
    }


def adaptive_max_pool_q48(volume256: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(volume256, dtype=np.float32)
    pooled = torch.nn.functional.adaptive_max_pool3d(
        torch.from_numpy(source)[None, None],
        output_size=(48, 48, 48),
    )[0, 0]
    return np.ascontiguousarray(pooled.numpy() >= 0.5)


def mem_available_bytes() -> int | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def regenerate_candidate(
    candidates: dict[str, np.ndarray],
    hull48: np.ndarray,
    candidate_name: str,
) -> np.ndarray:
    """Recover a base candidate or a deterministic erosion/dilation variant."""
    if candidate_name in candidates:
        return largest_component(candidates[candidate_name].astype(bool))
    if "__" not in candidate_name:
        raise KeyError(
            f"Candidate {candidate_name!r} was not regenerated; "
            f"available={sorted(candidates)}"
        )
    parent, operation = candidate_name.split("__", 1)
    if parent not in candidates:
        raise KeyError(
            f"Parent candidate {parent!r} was not regenerated; "
            f"available={sorted(candidates)}"
        )
    if operation.startswith("erode"):
        transform = binary_erosion
        steps_text = operation.removeprefix("erode")
    elif operation.startswith("dilate"):
        from scipy.ndimage import binary_dilation

        transform = binary_dilation
        steps_text = operation.removeprefix("dilate")
    else:
        raise ValueError(f"Unsupported archived variant operation: {operation!r}")
    steps = int(steps_text)
    if steps < 1:
        raise ValueError(f"Invalid archived variant iteration count: {steps}")
    changed = transform(
        largest_component(candidates[parent].astype(bool)),
        structure=S6,
        iterations=steps,
    )
    changed = largest_component(changed & hull48.astype(bool))
    if not changed.any():
        raise RuntimeError(f"Archived variant {candidate_name!r} regenerated as empty")
    return changed


def select_unique_source_case(
    payload: dict[str, Any],
    task: str,
    branch: str,
    case_id: int,
) -> dict[str, Any]:
    found = [
        case
        for case in payload["cases"]
        if "|".join(map(str, case["triple"])) == task
        and str(case["branch"]) == branch
        and int(case["case_id"]) == case_id
    ]
    if len(found) != 1:
        raise RuntimeError(
            f"Expected exactly one source case for task={task!r}, "
            f"branch={branch!r}, case_id={case_id}; found {len(found)}"
        )
    return found[0]


def select_unique_index_row(
    payload: dict[str, Any],
    task: str,
    branch: str,
    case_id: int,
    candidate: str,
) -> dict[str, Any]:
    found = [
        row
        for row in payload["rows"]
        if str(row["task"]) == task
        and str(row["branch"]) == branch
        and int(row["case_id"]) == case_id
        and str(row["candidate"]) == candidate
    ]
    if len(found) != 1:
        raise RuntimeError(
            f"Expected exactly one index row for task={task!r}, "
            f"branch={branch!r}, case_id={case_id}, candidate={candidate!r}; "
            f"found {len(found)}"
        )
    return found[0]


def select_unique_pino_row(
    payload: list[dict[str, Any]],
    task: str,
    candidate: str,
    archive_path: Path,
) -> dict[str, Any]:
    found = [
        row
        for row in payload
        if str(row["task"]) == task
        and str(row["candidate"]) == candidate
        and Path(str(row["path"])).resolve() == archive_path
    ]
    if len(found) != 1:
        raise RuntimeError(
            f"Expected exactly one PINO audit row for task={task!r}, "
            f"candidate={candidate!r}, archive={archive_path}; "
            f"found {len(found)}"
        )
    return found[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--pino-audit", type=Path, default=DEFAULT_PINO_AUDIT)
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help=(
            "Optional explicit archived NPZ. By default the path is resolved "
            "from the unique frozen index row for the requested candidate."
        ),
    )
    parser.add_argument("--task", default="sea_snail|x_ladybug|x_lobster")
    parser.add_argument("--branch", choices=("free", "fixed"), default="free")
    parser.add_argument("--case-id", type=int, default=115)
    parser.add_argument("--candidate", default="min_printable_bridge_r1")
    parser.add_argument("--delivery-resolution", type=int, default=256)
    parser.add_argument("--shell-voxels", type=int, default=2)
    parser.add_argument(
        "--minimum-mem-available-gib",
        type=float,
        default=3.0,
        help=(
            "Refuse reconstruction if Linux MemAvailable is below this value; "
            "the default prevents contention with a concurrent Q256 FEM solve."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Fresh output directory. By default a candidate-specific path "
            "under results/ is used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.source.resolve()
    index_path = args.index.resolve()
    pino_audit_path = args.pino_audit.resolve()
    candidate_slug = "".join(
        char if char.isalnum() or char in "_-" else "_" for char in args.candidate
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else (
            ROOT
            / (
                "results/raw_pino_boundary_q256_reconstruction_"
                f"case{args.case_id:03d}_{candidate_slug}_20260730"
            )
        ).resolve()
    )

    for path in (source_path, index_path, pino_audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.delivery_resolution != 256:
        raise ValueError(
            "This frozen reconstruction targets the original native Q256 delivery"
        )
    if args.shell_voxels != 2:
        raise ValueError(
            "The archived delivered-proxy data were generated with shell_voxels=2"
        )
    if args.minimum_mem_available_gib < 0:
        raise ValueError("--minimum-mem-available-gib must be nonnegative")
    mem_available = mem_available_bytes()
    minimum_memory = int(args.minimum_mem_available_gib * 1024**3)
    if mem_available is not None and mem_available < minimum_memory:
        raise RuntimeError(
            "Insufficient MemAvailable for non-competing reconstruction: "
            f"{mem_available / 1024**3:.3f} GiB < "
            f"{args.minimum_mem_available_gib:.3f} GiB"
        )

    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    pino_audit_payload = json.loads(pino_audit_path.read_text(encoding="utf-8"))
    if not isinstance(pino_audit_payload, list):
        raise TypeError("Expected the sealed PINO audit to be a JSON row list")
    source_case = select_unique_source_case(
        source_payload, args.task, args.branch, args.case_id
    )
    index_row = select_unique_index_row(
        index_payload, args.task, args.branch, args.case_id, args.candidate
    )
    archived_relpath = Path(str(index_row["path"]))
    indexed_archive_path = (
        archived_relpath
        if archived_relpath.is_absolute()
        else index_path.parent / archived_relpath
    ).resolve()
    archive_path = (
        args.archive.resolve() if args.archive is not None else indexed_archive_path
    )
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    pino_row = select_unique_pino_row(
        pino_audit_payload,
        args.task,
        args.candidate,
        archive_path,
    )
    source_resolution = int(source_payload["resolution"])
    if source_resolution != int(index_payload["resolution"]):
        raise RuntimeError(
            "Source and archived index resolutions disagree: "
            f"{source_resolution} vs {index_payload['resolution']}"
        )

    if indexed_archive_path != archive_path:
        raise RuntimeError(
            "Explicit archive does not match the frozen index row: "
            f"{archive_path} vs {indexed_archive_path}"
        )

    with np.load(archive_path, allow_pickle=False) as archived_npz:
        archived_density_raw = np.asarray(archived_npz["density"])
        archived_sf = float(archived_npz["sf"])
        archived_vm_max = float(archived_npz["vm_max"])
    if archived_density_raw.shape != (source_resolution,) * 3:
        raise RuntimeError(
            f"Unexpected archived density shape: {archived_density_raw.shape}"
        )
    archived_density = np.ascontiguousarray(archived_density_raw > 0.5)

    triple = list(map(str, source_case["triple"]))
    fix_axis = args.branch == "fixed"
    masks48 = load_masks(triple, source_resolution)
    candidates48, aux48 = make_candidates(masks48, source_resolution, fix_axis)
    core48 = regenerate_candidate(
        candidates48,
        aux48["hull"].astype(bool),
        args.candidate,
    )

    masks256 = load_masks(triple, args.delivery_resolution)
    calibration256 = calibrate_tfs_projection(
        masks256,
        search_res=min(96, args.delivery_resolution),
        fix_axis=fix_axis,
    )
    hull256, _ = build_tfs_visual_hull(
        masks256,
        args.delivery_resolution,
        calibration=calibration256,
    )
    hull256 = np.ascontiguousarray(hull256, dtype=bool)

    core_up256 = upsample_nearest(core48, args.delivery_resolution) & hull256
    shell256 = hull256 & ~binary_erosion(
        hull256,
        structure=S6,
        iterations=args.shell_voxels,
    )
    delivery256 = np.ascontiguousarray(
        largest_component(core_up256 | shell256),
        dtype=bool,
    )

    historical_pool48 = np.ascontiguousarray(
        conservative_pool(delivery256, source_resolution),
        dtype=bool,
    )
    deployed_pool48 = adaptive_max_pool_q48(delivery256)

    historical_vs_archive = difference_summary(historical_pool48, archived_density)
    deployed_vs_archive = difference_summary(deployed_pool48, archived_density)
    deployed_vs_historical = difference_summary(deployed_pool48, historical_pool48)
    historical_as_archived_dtype = np.ascontiguousarray(
        historical_pool48.astype(archived_density_raw.dtype)
    )
    deployed_as_archived_dtype = np.ascontiguousarray(
        deployed_pool48.astype(archived_density_raw.dtype)
    )
    historical_raw_bitwise_equal = bool(
        historical_as_archived_dtype.tobytes(order="C")
        == np.ascontiguousarray(archived_density_raw).tobytes(order="C")
    )
    deployed_raw_bitwise_equal = bool(
        deployed_as_archived_dtype.tobytes(order="C")
        == np.ascontiguousarray(archived_density_raw).tobytes(order="C")
    )

    expected_counts = {
        "core_occupied": int(index_row["core_occupied"]),
        "delivery_occupied": int(index_row["delivery_occupied"]),
        "proxy_occupied": int(index_row["proxy_occupied"]),
    }
    observed_counts = {
        "core_occupied": int(np.count_nonzero(core48)),
        "core_up_intersect_hull_occupied": int(np.count_nonzero(core_up256)),
        "shell_occupied": int(np.count_nonzero(shell256)),
        "delivery_occupied": int(np.count_nonzero(delivery256)),
        "historical_proxy_occupied": int(np.count_nonzero(historical_pool48)),
        "deployed_proxy_occupied": int(np.count_nonzero(deployed_pool48)),
        "archived_proxy_occupied": int(np.count_nonzero(archived_density)),
    }
    count_checks = {
        "core_matches_index": (
            observed_counts["core_occupied"] == expected_counts["core_occupied"]
        ),
        "delivery_matches_index": (
            observed_counts["delivery_occupied"] == expected_counts["delivery_occupied"]
        ),
        "historical_proxy_matches_index": (
            observed_counts["historical_proxy_occupied"]
            == expected_counts["proxy_occupied"]
        ),
    }

    strict_historical_reconstruction = bool(
        all(count_checks.values())
        and historical_vs_archive["bitwise_equal"]
        and historical_raw_bitwise_equal
    )
    strict_deployed_pair = bool(deployed_vs_archive["bitwise_equal"])

    output.mkdir(parents=True, exist_ok=False)
    delivery_path = output / "native_delivery_q256.npy"
    core_path = output / "regenerated_core_q48.npy"
    historical_path = output / "historical_conservative_pool_q48.npy"
    deployed_path = output / "deployed_adaptive_max_pool_q48.npy"
    np.save(delivery_path, delivery256.astype(np.uint8), allow_pickle=False)
    np.save(core_path, core48.astype(np.uint8), allow_pickle=False)
    np.save(
        historical_path,
        historical_pool48.astype(np.uint8),
        allow_pickle=False,
    )
    np.save(
        deployed_path,
        deployed_pool48.astype(np.uint8),
        allow_pickle=False,
    )

    record = {
        "schema": SCHEMA,
        "created_at_utc": utc_now(),
        "status": "COMPLETE",
        "performs_fem": False,
        "case": {
            "task": args.task,
            "triple": triple,
            "branch": args.branch,
            "case_id": args.case_id,
            "candidate": args.candidate,
            "split": index_row["split"],
            "source_resolution": source_resolution,
            "delivery_resolution": args.delivery_resolution,
            "shell_voxels": args.shell_voxels,
            "structural_representation": index_row["structural_representation"],
        },
        "archived_label": {
            "sf": archived_sf,
            "vm_max_pa": archived_vm_max,
            "fem_safe_at_tau": bool(archived_sf >= float(index_payload["tau"])),
            "tau": float(index_payload["tau"]),
        },
        "pino_audit": {
            "source": str(pino_audit_path),
            "source_sha256": sha256_file(pino_audit_path),
            "row": int(pino_row["row"]),
            "sf_fem": float(pino_row["sf_fem"]),
            "sf_pino": float(pino_row["sf_pino"]),
            "truth_safe": bool(pino_row["truth_safe"]),
            "raw_pino_safe_at_tau": bool(
                float(pino_row["sf_pino"]) >= float(index_payload["tau"])
            ),
            "raw_pino_false_safe": bool(
                float(pino_row["sf_fem"]) < float(index_payload["tau"])
                and float(pino_row["sf_pino"]) >= float(index_payload["tau"])
            ),
            "selective_policy_decision": str(pino_row["decision"]),
        },
        "inputs": {
            "source": str(source_path),
            "source_sha256": sha256_file(source_path),
            "index": str(index_path),
            "index_sha256": sha256_file(index_path),
            "archive": str(archive_path),
            "archive_file_sha256": sha256_file(archive_path),
            "silhouettes": [
                {
                    "path": str(
                        (ROOT / "data/silhouettes/bench3" / f"{name}.png").resolve()
                    ),
                    "sha256": sha256_file(
                        ROOT / "data/silhouettes/bench3" / f"{name}.png"
                    ),
                }
                for name in triple
            ],
        },
        "generation": {
            "candidate_function": (
                "experiments.candidate_selection_batch.make_candidates"
            ),
            "delivery_function_equivalent": (
                "experiments.build_taskgrouped_decision_fem_dataset_20260711."
                "delivered_proxy"
            ),
            "historical_pool": ("rounded-edge np.maximum.reduceat conservative_pool"),
            "deployed_pool": "torch.nn.functional.adaptive_max_pool3d",
            "calibration256": calibration256,
            "largest_component_connectivity": 6,
        },
        "expected_counts_from_index": expected_counts,
        "observed_counts": observed_counts,
        "count_checks": count_checks,
        "payload_hashes": {
            "core_q48": sha256_binary_occupancy(core48),
            "delivery_q256": sha256_binary_occupancy(delivery256),
            "historical_pool_q48": sha256_binary_occupancy(historical_pool48),
            "deployed_pool_q48": sha256_binary_occupancy(deployed_pool48),
            "archived_density_q48": sha256_binary_occupancy(archived_density),
            "historical_pool_as_archived_dtype": sha256_array_payload(
                historical_as_archived_dtype
            ),
            "deployed_pool_as_archived_dtype": sha256_array_payload(
                deployed_as_archived_dtype
            ),
            "archived_density_exact_array": sha256_array_payload(archived_density_raw),
        },
        "comparisons": {
            "historical_pool_vs_archived": {
                **historical_vs_archive,
                "exact_dtype": str(archived_density_raw.dtype),
                "exact_bytes_equal_after_dtype_match": (historical_raw_bitwise_equal),
            },
            "deployed_pool_vs_archived": {
                **deployed_vs_archive,
                "exact_dtype": str(archived_density_raw.dtype),
                "exact_bytes_equal_after_dtype_match": (deployed_raw_bitwise_equal),
            },
            "deployed_pool_vs_historical": deployed_vs_historical,
        },
        "audit_eligibility": {
            "historical_dataset_pair_reconstructed_exactly": (
                strict_historical_reconstruction
            ),
            "archived_q48_is_exact_deployed_pool_of_q256": strict_deployed_pair,
            "strict_pair_for_historical_dataset_protocol": (
                strict_historical_reconstruction
            ),
            "native_q256_and_generated_deployed_q48_form_strict_geometry_pair": (
                strict_historical_reconstruction
            ),
            "eligible_for_fresh_current_protocol_q48_q256_fem_audit": (
                strict_historical_reconstruction
            ),
            "archived_q48_fem_label_reusable_in_current_pool_protocol": (
                strict_historical_reconstruction and strict_deployed_pair
            ),
            "interpretation": (
                "The reconstructed Q256 delivery is a strict native/coarse pair "
                "for the historical dataset protocol if the historical equality "
                "flag is true. Its newly generated adaptive-pool Q48 proxy and "
                "native Q256 delivery can then receive fresh matched FEM solves. "
                "The archived Q48 FEM/PINO label may be reused under the current "
                "pool protocol only if the adaptive proxy also equals the archive."
            ),
        },
        "outputs": {
            "native_delivery_q256": str(delivery_path),
            "native_delivery_q256_file_sha256": sha256_file(delivery_path),
            "regenerated_core_q48": str(core_path),
            "regenerated_core_q48_file_sha256": sha256_file(core_path),
            "historical_conservative_pool_q48": str(historical_path),
            "historical_conservative_pool_q48_file_sha256": sha256_file(
                historical_path
            ),
            "deployed_adaptive_max_pool_q48": str(deployed_path),
            "deployed_adaptive_max_pool_q48_file_sha256": sha256_file(deployed_path),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mem_available_bytes_at_start": mem_available,
        },
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": record["status"],
                "output": str(output),
                "counts": observed_counts,
                "historical_pool_vs_archive": record["comparisons"][
                    "historical_pool_vs_archived"
                ],
                "deployed_pool_vs_archive": record["comparisons"][
                    "deployed_pool_vs_archived"
                ],
                "audit_eligibility": record["audit_eligibility"],
                "provenance": str(provenance_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
