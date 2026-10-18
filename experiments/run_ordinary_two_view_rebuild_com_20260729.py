#!/usr/bin/env python3
"""Preregister, rebuild, and COM-audit three ordinary two-view examples.

The three cases, normalized two-wall inputs, frozen soft fields, downstream
thresholds, delivery gates, and display-fixture dimensions are fixed in this
file.  Rebuilds are serial and begin only when Linux ``MemAvailable`` reaches
the requested threshold.  The output root must not exist: interrupted or
completed runs are preserved rather than silently resumed or overwritten.

By default the runner expects
``experiments/evaluate_ordinary_two_view_com_20260729.py``.  Use ``--skip-com``
only when intentionally producing the preregistered rebuilds without the COM
audit.  ``--preflight-only`` validates all frozen inputs and command-line
contracts without creating an output directory or starting reconstruction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(
    "/home/lin/anaconda3/envs/shadowart-revisited/bin/python"
)
DEFAULT_UNIFIED_RUN = ROOT / "shadowforge_pipeline/unified_run.py"
DEFAULT_COM_EVALUATOR = (
    ROOT / "experiments/evaluate_ordinary_two_view_com_20260729.py"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "results/ordinary_two_view_rebuild_com_20260729"
)

STRICT_GATES = {
    "min_iou": 0.975,
    "min_view_iou": 0.95,
    "min_saving": 0.30,
    "max_saving": 0.85,
    "min_erosion_survival": 0.08,
}
FIXTURE = {
    "margin_mm": 8.0,
    "base_thickness_mm": 4.0,
    "clearance_mm": 0.8,
    "retainer_height_mm": 1.5,
}


@dataclass(frozen=True)
class Case:
    case_id: str
    source_left: Path
    source_right: Path
    left_input: Path
    right_input: Path
    frozen_soft: Path
    threshold: str
    proposal_mode: str


CASES = (
    Case(
        case_id="parrot_snail",
        source_left=ROOT / "data/silhouettes/bench3/birds_parrot.png",
        source_right=ROOT / "data/silhouettes/bench3/sea_snail.png",
        left_input=(
            ROOT
            / "results/fem_two_wall_print_candidate_search_20260714"
            / "simple/L01_parrot_snail/inputs/left_wall_x_target.png"
        ),
        right_input=(
            ROOT
            / "results/fem_two_wall_print_candidate_search_20260714"
            / "simple/L01_parrot_snail/inputs/right_wall_y_target.png"
        ),
        frozen_soft=(
            ROOT
            / "results/soft_novel_loss_ablations_v3_20260715"
            / "parrot_snail/full/seed_17"
        ),
        threshold="0.55",
        proposal_mode="regularized",
    ),
    Case(
        case_id="owl_turtle",
        source_left=ROOT / "data/silhouettes/bench3/birds_owl.png",
        source_right=ROOT / "data/silhouettes/bench3/reptiles_turtle.png",
        left_input=(
            ROOT
            / "results/fem_two_wall_print_candidate_search_20260714"
            / "simple/L02_owl_turtle/inputs/left_wall_x_target.png"
        ),
        right_input=(
            ROOT
            / "results/fem_two_wall_print_candidate_search_20260714"
            / "simple/L02_owl_turtle/inputs/right_wall_y_target.png"
        ),
        frozen_soft=(
            ROOT
            / "results/soft_novel_loss_ablations_v3_20260715"
            / "owl_turtle/full/seed_17"
        ),
        threshold="0.55",
        proposal_mode="regularized",
    ),
    Case(
        case_id="frog_beetle",
        source_left=ROOT / "data/silhouettes/bench3/reptiles_frog.png",
        source_right=ROOT / "data/silhouettes/bench3/bugs_beetle.png",
        left_input=(
            ROOT
            / "results/fem_two_wall_print_candidate_search_20260714"
            / "simple/L03_frog_beetle/inputs/left_wall_x_target.png"
        ),
        right_input=(
            ROOT
            / "results/fem_two_wall_print_candidate_search_20260714"
            / "simple/L03_frog_beetle/inputs/right_wall_y_target.png"
        ),
        frozen_soft=(
            ROOT
            / "results/table2_two_stage_pilot_true256_3tasks_seed17_20260722"
            / "soft/L03_frog_beetle/full/seed_17"
        ),
        threshold="0.45",
        proposal_mode="proposal",
    ),
)

FROZEN_ARTIFACT_NAMES = (
    "metrics.json",
    "occupancy_probability_core.npy",
    "occupancy_soft_proposal.npy",
    "gt_view_0.png",
    "gt_view_1.png",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def paths_equal(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left.resolve())) == os.path.normcase(
        str(right.resolve())
    )


def memory_gib() -> tuple[float, float]:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
        for line in handle:
            key, raw = line.split(":", 1)
            if key in {"MemAvailable", "MemTotal"}:
                values[key] = int(raw.strip().split()[0])
    if set(values) != {"MemAvailable", "MemTotal"}:
        raise RuntimeError("Could not read MemAvailable/MemTotal from /proc/meminfo")
    scale = 1024.0 * 1024.0
    return values["MemAvailable"] / scale, values["MemTotal"] / scale


def log_line(handle: TextIO, message: str) -> None:
    record = f"{utc_now()} {message}"
    print(record, flush=True)
    handle.write(record + "\n")
    handle.flush()


def wait_for_memory(
    minimum_gib: float,
    poll_seconds: float,
    timeout_seconds: float,
    log_handle: TextIO,
) -> None:
    started = time.monotonic()
    while True:
        available, total = memory_gib()
        log_line(
            log_handle,
            (
                f"memory-check available={available:.2f} GiB "
                f"required={minimum_gib:.2f} GiB total={total:.2f} GiB"
            ),
        )
        if available >= minimum_gib:
            return
        elapsed = time.monotonic() - started
        if timeout_seconds > 0 and elapsed + poll_seconds > timeout_seconds:
            raise TimeoutError(
                f"MemAvailable stayed below {minimum_gib:.2f} GiB for "
                f"{elapsed:.1f} seconds"
            )
        time.sleep(poll_seconds)


def frozen_files(case: Case) -> tuple[Path, ...]:
    return tuple(case.frozen_soft / name for name in FROZEN_ARTIFACT_NAMES)


def source_geometry_manifest(case: Case) -> Path:
    left_parent = case.left_input.parent
    if left_parent != case.right_input.parent or left_parent.name != "inputs":
        raise ValueError(f"Unexpected normalized-input layout for {case.case_id}")
    return left_parent.parent / "MANIFEST.json"


def validate_case(case: Case) -> dict[str, Any]:
    required = (
        case.source_left,
        case.source_right,
        case.left_input,
        case.right_input,
        *frozen_files(case),
        source_geometry_manifest(case),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{case.case_id} is missing frozen inputs/artifacts: {missing}"
        )

    metrics_path = case.frozen_soft / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    frozen_args = metrics.get("args", {})
    shadow_files = [Path(item) for item in frozen_args.get("shadow_files", [])]
    expected_inputs = [case.left_input, case.right_input]
    if len(shadow_files) != 2 or any(
        not paths_equal(observed, expected)
        for observed, expected in zip(shadow_files, expected_inputs)
    ):
        raise ValueError(
            f"{case.case_id}: frozen soft field was not trained from the "
            f"declared normalized input pair: {shadow_files}"
        )
    expected_frozen = {
        "proposal_mode": case.proposal_mode,
        "res": 256,
        "export_res": 256,
        "mirror_mode": 0,
        "tsf_mode": 0,
        "seed": 17,
    }
    mismatches = {
        key: {"expected": expected, "observed": frozen_args.get(key)}
        for key, expected in expected_frozen.items()
        if frozen_args.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"{case.case_id}: frozen soft metadata mismatch: {mismatches}"
        )

    provenance_manifest_path = source_geometry_manifest(case)
    provenance = json.loads(
        provenance_manifest_path.read_text(encoding="utf-8")
    )
    source_inputs = [Path(item) for item in provenance.get("inputs", [])]
    expected_sources = [case.source_left, case.source_right]
    if len(source_inputs) != 2:
        raise ValueError(
            f"{case.case_id}: source geometry manifest does not declare two inputs"
        )
    for observed, expected in zip(source_inputs, expected_sources):
        if observed.name != expected.name or sha256_file(observed) != sha256_file(
            expected
        ):
            raise ValueError(
                f"{case.case_id}: bench3 source does not match recorded source "
                f"{observed}"
            )

    return {
        "case_id": case.case_id,
        "source_silhouettes": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in expected_sources
        ],
        "pipeline_inputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in expected_inputs
        ],
        "source_geometry_manifest": {
            "path": str(provenance_manifest_path.resolve()),
            "sha256": sha256_file(provenance_manifest_path),
        },
        "frozen_soft": {
            "directory": str(case.frozen_soft.resolve()),
            "proposal_mode": case.proposal_mode,
            "threshold": case.threshold,
            "files": [
                {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for path in frozen_files(case)
            ],
        },
    }


def validate_unified_run(python: Path, unified_run: Path) -> str:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"Python executable is unavailable: {python}")
    if not unified_run.is_file():
        raise FileNotFoundError(f"Unified runner is unavailable: {unified_run}")
    completed = subprocess.run(
        [str(python), str(unified_run), "--help"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    required_flags = (
        "--silhouettes",
        "--branch",
        "--soft-artifact-dir",
        "--soft-seed-thresholds",
        "--display-fixture",
        "--min-iou",
        "--min-view-iou",
        "--min-saving",
        "--max-saving",
        "--min-erosion-survival",
        "--tsf",
        "--mr",
    )
    missing = [flag for flag in required_flags if flag not in completed.stdout]
    if missing:
        raise RuntimeError(
            f"Unified runner help is missing required flags: {missing}"
        )
    return completed.stdout


def build_rebuild_command(
    python: Path,
    unified_run: Path,
    case: Case,
    output: Path,
) -> list[str]:
    return [
        str(python.resolve()),
        str(unified_run.resolve()),
        "--silhouettes",
        str(case.left_input.resolve()),
        str(case.right_input.resolve()),
        "--branch",
        "soft-geometry",
        "--output",
        str(output.resolve()),
        "--tsf",
        "0",
        "--mr",
        "0",
        "--print-mm",
        "100",
        "--soft-resolution",
        "256",
        "--soft-control-res",
        "0",
        "--soft-proposal-mode",
        case.proposal_mode,
        "--soft-artifact-dir",
        str(case.frozen_soft.resolve()),
        "--soft-seed-thresholds",
        case.threshold,
        "--core-resolution",
        "48",
        "--delivery-shell-voxels",
        "2",
        "--min-iou",
        str(STRICT_GATES["min_iou"]),
        "--min-view-iou",
        str(STRICT_GATES["min_view_iou"]),
        "--min-saving",
        str(STRICT_GATES["min_saving"]),
        "--max-saving",
        str(STRICT_GATES["max_saving"]),
        "--min-erosion-survival",
        str(STRICT_GATES["min_erosion_survival"]),
        "--selection-iou-tolerance",
        "0.01",
        "--selection-policy",
        "material",
        "--screen",
        "skip",
        "--display-fixture",
        "--fixture-margin-mm",
        str(FIXTURE["margin_mm"]),
        "--fixture-base-thickness-mm",
        str(FIXTURE["base_thickness_mm"]),
        "--fixture-clearance-mm",
        str(FIXTURE["clearance_mm"]),
        "--fixture-retainer-height-mm",
        str(FIXTURE["retainer_height_mm"]),
        "--device",
        "cpu",
    ]


def run_streamed(
    command: list[str],
    local_log: Path,
    master_log: TextIO,
    label: str,
) -> None:
    log_line(master_log, f"{label} command={shlex.join(command)}")
    with local_log.open("x", encoding="utf-8") as case_log:
        case_log.write(f"{utc_now()} command={shlex.join(command)}\n")
        case_log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            case_log.write(line)
            case_log.flush()
            master_log.write(f"[{label}] {line}")
            master_log.flush()
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    log_line(master_log, f"{label} completed returncode=0")


def ensure_close(observed: Any, expected: float, label: str) -> None:
    if not math.isclose(
        float(observed), expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{label}: expected {expected}, observed {observed}")


def resolve_manifest_file(run_dir: Path, raw: Any, label: str) -> Path:
    if not raw:
        raise ValueError(f"Run manifest is missing {label}")
    path = Path(str(raw))
    if not path.is_file():
        local = run_dir / path.name
        if local.is_file():
            path = local
    if not path.is_file():
        raise FileNotFoundError(f"Run manifest {label} does not exist: {path}")
    resolved = path.resolve()
    if resolved.parent != run_dir.resolve():
        raise ValueError(
            f"Run manifest {label} escapes the case directory: {resolved}"
        )
    return resolved


def validate_rebuild(case: Case, run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{case.case_id}: unified run did not write MANIFEST.json"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "SUCCESS":
        raise ValueError(
            f"{case.case_id}: strict rebuild status is "
            f"{payload.get('status')!r}, not 'SUCCESS'"
        )
    if (
        payload.get("view_model") != "fixed_two_wall"
        or payload.get("n_views") != 2
        or payload.get("branches") != "soft-geometry"
    ):
        raise ValueError(
            f"{case.case_id}: unexpected protocol fields in run manifest"
        )
    observed_inputs = [Path(item) for item in payload.get("inputs", [])]
    expected_inputs = [case.left_input, case.right_input]
    if len(observed_inputs) != 2 or any(
        not paths_equal(observed, expected)
        for observed, expected in zip(observed_inputs, expected_inputs)
    ):
        raise ValueError(
            f"{case.case_id}: output manifest changed the input pair"
        )
    soft_dir = Path(payload.get("soft", {}).get("artifact_dir", ""))
    if not paths_equal(soft_dir, case.frozen_soft):
        raise ValueError(
            f"{case.case_id}: output manifest changed the frozen soft field"
        )

    standards = payload.get("standards", {})
    for key, expected in STRICT_GATES.items():
        ensure_close(standards.get(key), expected, f"{case.case_id}:{key}")

    selected = payload.get("selected") or {}
    if (
        float(selected.get("mean_iou", -1.0)) < STRICT_GATES["min_iou"]
        or float(selected.get("min_iou", -1.0))
        < STRICT_GATES["min_view_iou"]
        or float(selected.get("saving", -1.0)) < STRICT_GATES["min_saving"]
        or float(selected.get("saving", 2.0)) > STRICT_GATES["max_saving"]
        or float(selected.get("erosion_survival", -1.0))
        < STRICT_GATES["min_erosion_survival"]
        or int(selected.get("components", -1)) != 1
        or not bool(selected.get("watertight"))
        or int(selected.get("mesh_outer_parts", -1)) != 1
        or not bool(payload.get("mesh_watertight"))
    ):
        raise ValueError(
            f"{case.case_id}: selected delivery failed a preregistered gate"
        )

    display_stl = resolve_manifest_file(
        run_dir, payload.get("display_stl"), "display_stl"
    )
    print_stl = resolve_manifest_file(
        run_dir, payload.get("print_stl"), "print_stl"
    )
    display_voxels = run_dir / "selected_display_voxels256.npy"
    if not display_voxels.is_file():
        raise FileNotFoundError(
            f"{case.case_id}: missing selected_display_voxels256.npy"
        )
    delivery = payload.get("display_delivery")
    if not isinstance(delivery, dict):
        raise ValueError(
            f"{case.case_id}: --display-fixture did not produce delivery metadata"
        )
    display_model = Path(str(delivery.get("display_model", "")))
    separate_plinth = Path(str(delivery.get("separate_plinth", "")))
    for label, path in (
        ("display_model", display_model),
        ("separate_plinth", separate_plinth),
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"{case.case_id}: missing fixture {label}: {path}"
            )
        if path.resolve().parent != (run_dir / "display_delivery").resolve():
            raise ValueError(
                f"{case.case_id}: fixture {label} escapes display_delivery"
            )
    fixture_manifest = run_dir / "display_delivery/DISPLAY_DELIVERY.json"
    if not fixture_manifest.is_file():
        raise FileNotFoundError(
            f"{case.case_id}: missing DISPLAY_DELIVERY.json"
        )

    artifact_paths = {
        "manifest": manifest_path,
        "display_voxels": display_voxels,
        "display_stl": display_stl,
        "print_stl": print_stl,
        "display_model_on_plinth": display_model,
        "separate_plinth": separate_plinth,
        "display_delivery_manifest": fixture_manifest,
    }
    return {
        "case_id": case.case_id,
        "status": "SUCCESS",
        "run_dir": str(run_dir.resolve()),
        "selected_metrics": {
            key: selected.get(key)
            for key in (
                "candidate",
                "mean_iou",
                "min_iou",
                "mean_dice",
                "saving",
                "components",
                "erosion_survival",
                "watertight",
                "mesh_outer_parts",
                "print_orientation",
            )
        },
        "artifacts": {
            key: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for key, path in artifact_paths.items()
        },
    }


def wait_for_com_evaluator(
    requested: Path,
    wait_seconds: float,
    poll_seconds: float,
) -> Path:
    candidate = requested.resolve()
    started = time.monotonic()
    while not candidate.is_file():
        elapsed = time.monotonic() - started
        if wait_seconds <= 0 or elapsed >= wait_seconds:
            raise FileNotFoundError(
                f"Ordinary two-view COM evaluator is unavailable: {candidate}. "
                "Provide --com-evaluator, wait with "
                "--wait-for-com-evaluator-seconds, or explicitly use --skip-com."
            )
        remaining = max(wait_seconds - elapsed, 0.0)
        print(
            f"Waiting for COM evaluator {candidate} "
            f"({remaining:.0f}s remaining)",
            flush=True,
        )
        time.sleep(min(poll_seconds, remaining))
    return candidate


def evaluator_help(python: Path, evaluator: Path) -> str:
    completed = subprocess.run(
        [str(python), str(evaluator), "--help"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout


def output_flag(help_text: str) -> str:
    if "--output" in help_text:
        return "--output"
    if "--out-dir" in help_text:
        return "--out-dir"
    raise RuntimeError("COM evaluator exposes neither --output nor --out-dir")


def build_com_commands(
    python: Path,
    evaluator: Path,
    help_text: str,
    rebuild_index: Path,
    rebuilds_root: Path,
    com_output: Path,
    records: list[dict[str, Any]],
    extra_args: list[str],
) -> tuple[str, list[tuple[str, list[str]]]]:
    out_flag = output_flag(help_text)
    base = [str(python.resolve()), str(evaluator.resolve())]
    if "--root" in help_text:
        return (
            "root",
            [
                (
                    "com-batch",
                    base
                    + [
                        "--root",
                        str(rebuilds_root.resolve()),
                        out_flag,
                        str(com_output.resolve()),
                    ]
                    + extra_args,
                )
            ],
        )
    if "--batch-manifest" in help_text:
        return (
            "batch-manifest",
            [
                (
                    "com-batch",
                    base
                    + [
                        "--batch-manifest",
                        str(rebuild_index.resolve()),
                        out_flag,
                        str(com_output.resolve()),
                    ]
                    + extra_args,
                )
            ],
        )
    if "--run-index" in help_text:
        return (
            "run-index",
            [
                (
                    "com-batch",
                    base
                    + [
                        "--run-index",
                        str(rebuild_index.resolve()),
                        out_flag,
                        str(com_output.resolve()),
                    ]
                    + extra_args,
                )
            ],
        )
    if "--runs-root" in help_text:
        return (
            "runs-root",
            [
                (
                    "com-batch",
                    base
                    + [
                        "--runs-root",
                        str(rebuilds_root.resolve()),
                        out_flag,
                        str(com_output.resolve()),
                    ]
                    + extra_args,
                )
            ],
        )
    per_case_flags = ("--input", "--manifest", "--case")
    if all(flag in help_text for flag in per_case_flags):
        commands: list[tuple[str, list[str]]] = []
        for record in records:
            artifacts = record["artifacts"]
            case_id = str(record["case_id"])
            commands.append(
                (
                    f"com-{case_id}",
                    base
                    + [
                        "--input",
                        artifacts["display_voxels"]["path"],
                        "--manifest",
                        artifacts["manifest"]["path"],
                        "--case",
                        case_id,
                        out_flag,
                        str((com_output / case_id).resolve()),
                    ]
                    + extra_args,
                )
            )
        return "per-case", commands
    raise RuntimeError(
        "Unsupported COM evaluator CLI. Expected one of --batch-manifest, "
        "--run-index, --runs-root, or the per-case "
        "--input/--manifest/--case contract."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--unified-run", type=Path, default=DEFAULT_UNIFIED_RUN
    )
    parser.add_argument(
        "--wait-for-memory-gib",
        type=float,
        default=32.0,
        help="Minimum Linux MemAvailable before each serial rebuild.",
    )
    parser.add_argument(
        "--memory-poll-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--memory-timeout-seconds",
        type=float,
        default=0.0,
        help="0 waits indefinitely; otherwise fail after this many seconds.",
    )
    parser.add_argument(
        "--com-evaluator",
        type=Path,
        default=DEFAULT_COM_EVALUATOR,
    )
    parser.add_argument(
        "--wait-for-com-evaluator-seconds",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--com-poll-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--com-extra-arg",
        action="append",
        default=[],
        help=(
            "Append one argument to every COM evaluator invocation. For values "
            "beginning with '--', use --com-extra-arg=--flag."
        ),
    )
    parser.add_argument(
        "--skip-com",
        action="store_true",
        help="Explicitly stop after the three preregistered rebuilds.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate frozen inputs and CLIs without creating output or running.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    python = args.python.resolve()
    unified_run = args.unified_run.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite or resume existing output root: {output_root}"
        )
    if args.wait_for_memory_gib < 0:
        raise ValueError("--wait-for-memory-gib must be non-negative")
    if args.memory_poll_seconds <= 0 or args.com_poll_seconds <= 0:
        raise ValueError("Polling intervals must be positive")
    if args.memory_timeout_seconds < 0:
        raise ValueError("--memory-timeout-seconds must be non-negative")
    if args.wait_for_com_evaluator_seconds < 0:
        raise ValueError(
            "--wait-for-com-evaluator-seconds must be non-negative"
        )

    unified_help = validate_unified_run(python, unified_run)
    del unified_help
    case_provenance = [validate_case(case) for case in CASES]
    available_gib, total_gib = memory_gib()
    if args.wait_for_memory_gib > total_gib:
        raise ValueError(
            f"Requested memory threshold {args.wait_for_memory_gib:.2f} GiB "
            f"exceeds MemTotal {total_gib:.2f} GiB"
        )

    com_evaluator: Path | None = None
    com_help: str | None = None
    if not args.skip_com:
        com_evaluator = wait_for_com_evaluator(
            args.com_evaluator,
            args.wait_for_com_evaluator_seconds,
            args.com_poll_seconds,
        )
        com_help = evaluator_help(python, com_evaluator)

    rebuilds_root = output_root / "rebuilds"
    logs_root = output_root / "logs"
    commands = {
        case.case_id: build_rebuild_command(
            python,
            unified_run,
            case,
            rebuilds_root / case.case_id,
        )
        for case in CASES
    }
    preregistration: dict[str, Any] = {
        "schema": "shadowforge_ordinary_two_view_rebuild_com_prereg_v1",
        "status": "FROZEN_BEFORE_EXECUTION",
        "created_at_utc": utc_now(),
        "objective": (
            "Serially replay three ordinary fixed-two-wall reconstructions from "
            "their matched frozen soft fields, export separate display fixtures, "
            "enforce strict delivery gates, and optionally run a separately "
            "declared COM evaluator."
        ),
        "analysis_scope": {
            "cases": [case.case_id for case in CASES],
            "case_selection": "fixed before execution; no replacements",
            "n_views": 2,
            "view_model": "fixed_two_wall",
            "view_0": "left wall; projection along physical X; image maps to [Y,Z]",
            "view_1": "right wall; projection along physical Y; image maps to [X,Z]",
            "image_top": "physical +Z",
            "display_pose": "selected_display_fixed; physical Z-up",
            "print_pose": "secondary FDM build orientation only",
        },
        "execution": {
            "serial": True,
            "overwrite_policy": "output root must not exist",
            "output_root": str(output_root),
            "wait_for_memory_gib": float(args.wait_for_memory_gib),
            "memory_poll_seconds": float(args.memory_poll_seconds),
            "memory_timeout_seconds": float(args.memory_timeout_seconds),
            "available_memory_at_preflight_gib": available_gib,
            "total_memory_gib": total_gib,
        },
        "reconstruction_protocol": {
            "branch": "soft-geometry",
            "tsf": 0,
            "mr": 0,
            "print_mm": 100.0,
            "soft_resolution": 256,
            "core_resolution": 48,
            "delivery_shell_voxels": 2,
            "selection_iou_tolerance": 0.01,
            "selection_policy": "material",
            "screen": "skip",
            "strict_gates": STRICT_GATES,
            "required_topology": {
                "components_6conn": 1,
                "watertight": True,
                "exterior_mesh_parts": 1,
            },
            "display_fixture": FIXTURE,
            "fixture_metric_policy": (
                "separate display fixture is exported after selection and remains "
                "excluded from silhouette fidelity and material-saving metrics"
            ),
        },
        "cases": [
            {
                **provenance,
                "output": str((rebuilds_root / case.case_id).resolve()),
                "command": commands[case.case_id],
                "command_shell": shlex.join(commands[case.case_id]),
            }
            for case, provenance in zip(CASES, case_provenance)
        ],
        "software": {
            "python_executable": str(python),
            "python_version": platform.python_version(),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "unified_run": str(unified_run),
            "unified_run_sha256": sha256_file(unified_run),
        },
        "com": {
            "skipped_by_explicit_flag": bool(args.skip_com),
            "evaluator": (
                str(com_evaluator.resolve()) if com_evaluator is not None else None
            ),
            "evaluator_sha256": (
                sha256_file(com_evaluator) if com_evaluator is not None else None
            ),
            "extra_args": list(args.com_extra_arg),
            "primary_pose": "fixed display",
            "gravity_axis": "-Z",
            "fixture_policy": (
                "the COM evaluator must declare whether it audits the bare fixed "
                "display mesh or the as-delivered placed model plus separate "
                "plinth; the rebuild wrapper never substitutes one silently"
            ),
        },
    }

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_OK",
                    "output_root_available": str(output_root),
                    "case_ids": [case.case_id for case in CASES],
                    "available_memory_gib": available_gib,
                    "required_memory_gib": args.wait_for_memory_gib,
                    "com": preregistration["com"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output_root.mkdir(parents=True, exist_ok=False)
    rebuilds_root.mkdir()
    logs_root.mkdir()
    preregistration_path = output_root / "PREREGISTRATION.json"
    status_path = output_root / "STATUS.json"
    rebuild_index_path = output_root / "REBUILD_INDEX.json"
    master_log_path = output_root / "RUN.log"
    write_json_atomic(preregistration_path, preregistration)
    status: dict[str, Any] = {
        "schema": "shadowforge_ordinary_two_view_rebuild_com_status_v1",
        "status": "RUNNING",
        "started_at_utc": utc_now(),
        "preregistration": str(preregistration_path),
        "cases": {
            case.case_id: {"status": "PENDING"} for case in CASES
        },
        "com": {"status": "PENDING" if not args.skip_com else "SKIPPED"},
    }
    write_json_atomic(status_path, status)

    records: list[dict[str, Any]] = []
    with master_log_path.open("x", encoding="utf-8") as master_log:
        try:
            log_line(
                master_log,
                (
                    "preregistration-frozen "
                    f"sha256={sha256_file(preregistration_path)}"
                ),
            )
            for case in CASES:
                case_status = status["cases"][case.case_id]
                case_status.update(
                    {"status": "WAITING_FOR_MEMORY", "updated_at_utc": utc_now()}
                )
                write_json_atomic(status_path, status)
                wait_for_memory(
                    args.wait_for_memory_gib,
                    args.memory_poll_seconds,
                    args.memory_timeout_seconds,
                    master_log,
                )
                run_dir = rebuilds_root / case.case_id
                if run_dir.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite case output: {run_dir}"
                    )
                case_status.update(
                    {
                        "status": "RUNNING",
                        "started_at_utc": utc_now(),
                        "command": commands[case.case_id],
                    }
                )
                write_json_atomic(status_path, status)
                run_streamed(
                    commands[case.case_id],
                    logs_root / f"{case.case_id}.log",
                    master_log,
                    f"rebuild-{case.case_id}",
                )
                record = validate_rebuild(case, run_dir)
                records.append(record)
                case_status.update(
                    {
                        "status": "SUCCESS",
                        "completed_at_utc": utc_now(),
                        "manifest": record["artifacts"]["manifest"]["path"],
                        "display_voxels": record["artifacts"][
                            "display_voxels"
                        ]["path"],
                        "display_stl": record["artifacts"]["display_stl"]["path"],
                    }
                )
                write_json_atomic(status_path, status)

            rebuild_index = {
                "schema": "shadowforge_ordinary_two_view_rebuild_index_v1",
                "status": "COMPLETE",
                "created_at_utc": utc_now(),
                "preregistration": str(preregistration_path),
                "primary_pose": "fixed display",
                "gravity_axis": "-Z",
                "cases": records,
            }
            write_json_atomic(rebuild_index_path, rebuild_index)
            status["status"] = "REBUILDS_COMPLETE"
            status["rebuild_index"] = str(rebuild_index_path)
            status["rebuilds_completed_at_utc"] = utc_now()
            write_json_atomic(status_path, status)

            if args.skip_com:
                log_line(master_log, "COM audit explicitly skipped")
                status["status"] = "COMPLETE_COM_SKIPPED"
                status["completed_at_utc"] = utc_now()
                write_json_atomic(status_path, status)
                return 0

            assert com_evaluator is not None and com_help is not None
            com_output = output_root / "com_summary.json"
            contract, com_commands = build_com_commands(
                python=python,
                evaluator=com_evaluator,
                help_text=com_help,
                rebuild_index=rebuild_index_path,
                rebuilds_root=rebuilds_root,
                com_output=com_output,
                records=records,
                extra_args=list(args.com_extra_arg),
            )
            status["com"].update(
                {
                    "status": "RUNNING",
                    "started_at_utc": utc_now(),
                    "evaluator": str(com_evaluator),
                    "contract": contract,
                    "commands": [command for _, command in com_commands],
                }
            )
            write_json_atomic(status_path, status)
            for label, command in com_commands:
                run_streamed(
                    command,
                    logs_root / f"{label}.log",
                    master_log,
                    label,
                )
            status["com"].update(
                {"status": "SUCCESS", "completed_at_utc": utc_now()}
            )
            status["status"] = "COMPLETE"
            status["completed_at_utc"] = utc_now()
            write_json_atomic(status_path, status)
            return 0
        except KeyboardInterrupt:
            status["status"] = "INTERRUPTED"
            status["completed_at_utc"] = utc_now()
            status["error"] = "KeyboardInterrupt"
            write_json_atomic(status_path, status)
            log_line(master_log, "run interrupted")
            raise
        except Exception as exc:
            status["status"] = "FAILED"
            status["completed_at_utc"] = utc_now()
            status["error"] = repr(exc)
            status["traceback"] = traceback.format_exc()
            write_json_atomic(status_path, status)
            log_line(master_log, f"run failed error={exc!r}")
            raise


if __name__ == "__main__":
    raise SystemExit(main())
