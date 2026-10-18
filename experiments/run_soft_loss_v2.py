#!/usr/bin/env python3
"""Pre-registered staged runner for the P-CRAFT soft-loss-v2 experiment.

This script is orchestration only.  It invokes the project-local soft trainer
and never imports or launches repair, carving, delivery, meshing, PINO, or FEM.
Every run is addressed by stage/steps/task/variant/seed and is resumable only
when its resolved trainer configuration is byte-for-byte compatible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT = Path("/home/lin/xvxinyao/P-CRAFT")
PYTHON = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")
OUTPUT_ROOT = Path(
    "/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/"
    "shadow-project2/mesh/voxel/runs/soft_loss_v2_20260826"
)
BENCHMARK = PROJECT / "results/fixed3_mr1_connected10_pending_20260722/benchmark_manifest.json"
ANIME_ROOT = PROJECT / "results/table2_anime_fixed3_inputs_20260722"
APPROVED_STATUS = "APPROVED_FROZEN"
SEED = 17
LEARNING_RATE = 0.007
ANIME_TASKS = ("arcane_casters", "blade_heroes", "stage_dancers")
STAGES = (
    "manifest", "smoke128", "dev128_screen", "dev128_confirm",
    "transfer256", "formal256", "all",
)


@dataclass(frozen=True)
class TaskInputs:
    case_id: str
    targets: tuple[str, str, str]
    input_order: str = "xyz"


@dataclass(frozen=True)
class RunSpec:
    stage: str
    task: str
    variant: str
    targets: tuple[str, str, str]
    resolution: int
    export_resolution: int
    steps: int
    profile: str
    output_root: Path
    seed: int = SEED
    learning_rate: float = LEARNING_RATE
    experiment_id: int = 1
    device: str = "cuda:0"
    control_resolution: int = 0
    lambda_g: float = 0.25
    lambda_b: float = 0.10
    late_ssim_weight: float = 0.0
    view_aggregation: str | None = None
    disable_proxy: bool = False
    disable_gate: bool = False
    disable_boundary: bool = False
    disable_l1: bool = False
    disable_l2: bool = False
    disable_ssim: bool = False
    soft_only: bool = True
    final_step_export: bool = True
    paper_eligible: bool = False
    frozen_config_hash: str | None = None
    frozen_status: str | None = None
    input_order: str = "xyz"

    @property
    def run_dir(self) -> Path:
        return (
            Path(self.output_root) / self.stage / f"steps_{self.steps}" /
            self.task / self.variant / f"seed_{self.seed}"
        )


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialized_json(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(_serialized_json(payload))
    os.replace(temporary, path)


def immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create a registration exactly once; only an identical rerun may reuse it."""

    expected = _serialized_json(payload)
    expected_semantics = json.loads(expected.decode("utf-8"))
    if path.exists():
        try:
            existing_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileExistsError(f"refusing to replace unreadable registration: {path}") from exc
        if existing_payload != expected_semantics or path.read_bytes() != expected:
            raise FileExistsError(f"refusing to overwrite changed registration: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(expected)
    try:
        # Exclusive final creation avoids replacing a registration created by
        # a concurrent process between the existence check and this point.
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        immutable_json(path, payload)
        return
    temporary.unlink()


def _rewrite_project_path(path_text: str) -> str:
    old_prefix = "/home/lin/xvxinyao/ShadowForge/"
    if path_text.startswith(old_prefix):
        return str(PROJECT / path_text[len(old_prefix):])
    return str(Path(path_text))


def load_frozen_benchmark(path: Path = BENCHMARK, *, require_files: bool = True) -> list[TaskInputs]:
    """Load the approved benchmark in P-CRAFT x,y,z order.

    The historical ``sadekar_tfs_targets`` field swaps x/y and is deliberately
    ignored.  Only ``target_paths_xyz`` is authoritative for this experiment.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != APPROVED_STATUS:
        raise ValueError(f"benchmark is not {APPROVED_STATUS}: {path}")
    if payload.get("protocol", {}).get("axes") != ["x", "y", "z"]:
        raise ValueError("frozen benchmark does not declare x,y,z input axes")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("frozen benchmark must contain exactly ten cases")
    loaded: list[TaskInputs] = []
    for case in cases:
        case_id = str(case.get("case_id", ""))
        expected_id = f"connected_{len(loaded) + 1:02d}"
        if case_id != expected_id:
            raise ValueError(f"expected ordered case {expected_id}, found {case_id!r}")
        raw_targets = case.get("target_paths_xyz")
        if not isinstance(raw_targets, list) or len(raw_targets) != 3:
            raise ValueError(f"{case_id}: target_paths_xyz must contain x,y,z")
        targets = tuple(_rewrite_project_path(str(value)) for value in raw_targets)
        names = [Path(value).stem.lower() for value in targets]
        if not all(f"view_{axis}" in name for name, axis in zip(names, "xyz")):
            raise ValueError(f"{case_id}: target_paths_xyz is not in x,y,z order")
        if require_files:
            missing = [value for value in targets if not Path(value).is_file()]
            if missing:
                raise FileNotFoundError(f"{case_id}: missing frozen inputs: {missing}")
        loaded.append(TaskInputs(case_id=case_id, targets=targets))
    return loaded


def load_anime_tasks(*, require_files: bool = True) -> list[TaskInputs]:
    loaded = []
    for task in ANIME_TASKS:
        targets = tuple(str(ANIME_ROOT / task / f"view_{index}.png") for index in range(3))
        if require_files:
            missing = [value for value in targets if not Path(value).is_file()]
            if missing:
                raise FileNotFoundError(f"{task}: missing inputs: {missing}")
        loaded.append(TaskInputs(case_id=task, targets=targets))
    return loaded


def _selected_candidate(path: Path) -> dict[str, float]:
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"development selection is required before confirmation: {path}"
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = payload.get("selected_candidate", payload.get("selected", payload.get("candidate")))
    if not isinstance(selected, Mapping):
        raise ValueError("dev128 selection has no selected_candidate mapping")
    result = {
        "lambda_g": float(selected["lambda_g"]),
        "lambda_b": float(selected["lambda_b"]),
        "late_ssim_weight": float(selected.get("late_ssim_weight", 0.0)),
    }
    allowed_gb = {(0.10, 0.05), (0.25, 0.05), (0.25, 0.10), (0.50, 0.20)}
    if (result["lambda_g"], result["lambda_b"]) not in allowed_gb:
        raise ValueError("selected Gate/Boundary weights were not screened")
    if result["late_ssim_weight"] not in (0.0, 0.05):
        raise ValueError("selected late SSIM weight was not screened")
    return result


def load_frozen_config(path: Path, *, expected_source: str | None = None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("an explicit approved --frozen-config is required")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "pcraft_soft_loss_v2_frozen_config_v1":
        raise ValueError("unrecognized frozen configuration schema")
    if payload.get("status") != APPROVED_STATUS:
        raise ValueError(f"frozen configuration is not {APPROVED_STATUS}")
    if expected_source is not None and payload.get("source_stage") != expected_source:
        raise ValueError(
            f"frozen configuration for this stage must come from {expected_source}; "
            f"found {payload.get('source_stage')!r}"
        )
    values = {
        "lambda_g": float(payload["lambda_g"]),
        "lambda_b": float(payload["lambda_b"]),
        "late_ssim_weight": float(payload.get("late_ssim_weight", 0.0)),
    }
    if values["lambda_g"] < 0 or values["lambda_b"] < 0 or values["late_ssim_weight"] < 0:
        raise ValueError("frozen loss weights must be non-negative")
    payload = dict(payload)
    payload.update(values)
    payload["config_hash"] = _payload_hash({key: payload[key] for key in sorted(payload) if key != "config_hash"})
    return payload


def _base_spec(
    *, stage: str, task: TaskInputs, output_root: Path, resolution: int,
    steps: int, paper_eligible: bool = False,
) -> RunSpec:
    return RunSpec(
        stage=stage, task=task.case_id, variant="", targets=task.targets,
        resolution=resolution, export_resolution=resolution, steps=steps,
        profile="new_full", output_root=Path(output_root),
        paper_eligible=paper_eligible, input_order=task.input_order,
    )


def _unique_specs(specs: Sequence[RunSpec]) -> list[RunSpec]:
    result = list(specs)
    keys = [(spec.stage, spec.steps, spec.task, spec.variant, spec.seed) for spec in result]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"duplicate registered runs: {duplicates}")
    return result


def _fixed_variants(base: RunSpec, frozen: Mapping[str, Any], *, include_ssim_ablation: bool) -> list[RunSpec]:
    weights = {
        "lambda_g": float(frozen["lambda_g"]),
        "lambda_b": float(frozen["lambda_b"]),
        "late_ssim_weight": float(frozen["late_ssim_weight"]),
        "frozen_config_hash": str(frozen["config_hash"]),
        "frozen_status": str(frozen["status"]),
    }
    variants = [
        replace(base, variant="legacy_full", profile="legacy_full", late_ssim_weight=0.0, **{k: v for k, v in weights.items() if k != "late_ssim_weight"}),
        replace(base, variant="l12_plain", profile="l12_plain", late_ssim_weight=0.0, **{k: v for k, v in weights.items() if k != "late_ssim_weight"}),
        replace(base, variant="l12_bal", profile="l12_bal", late_ssim_weight=0.0, **{k: v for k, v in weights.items() if k != "late_ssim_weight"}),
        replace(base, variant="new_full", profile="new_full", **weights),
        replace(base, variant="new_full_no_proxy", profile="new_full", disable_proxy=True, **weights),
        replace(base, variant="new_full_no_gate", profile="new_full", disable_gate=True, **weights),
        replace(base, variant="new_full_no_boundary", profile="new_full", disable_boundary=True, **weights),
        replace(base, variant="new_full_mean", profile="new_full", view_aggregation="mean", **weights),
        replace(base, variant="new_full_no_l1", profile="new_full", disable_l1=True, **weights),
        replace(base, variant="new_full_no_l2", profile="new_full", disable_l2=True, **weights),
    ]
    if include_ssim_ablation and float(frozen["late_ssim_weight"]) > 0:
        variants.insert(8, replace(base, variant="new_full_no_ssim", profile="new_full", disable_ssim=True, **weights))
    return variants


def build_stage_specs(
    stage: str,
    *,
    output_root: Path = OUTPUT_ROOT,
    benchmark_path: Path = BENCHMARK,
    frozen_config_path: Path | None = None,
    selection_path: Path | None = None,
    require_files: bool = True,
) -> list[RunSpec]:
    if stage not in STAGES or stage in {"manifest", "all"}:
        raise ValueError(f"stage {stage!r} does not directly define run specs")
    output_root = Path(output_root)
    anime = load_anime_tasks(require_files=require_files)

    if stage == "smoke128":
        base = _base_spec(stage=stage, task=anime[0], output_root=output_root, resolution=128, steps=200)
        return _unique_specs([
            replace(base, variant="legacy_full", profile="legacy_full"),
            replace(base, variant="l12_plain", profile="l12_plain"),
            replace(base, variant="l12_bal", profile="l12_bal"),
            replace(base, variant="new_full", profile="new_full"),
            replace(base, variant="new_full_no_proxy", profile="new_full", disable_proxy=True),
            replace(base, variant="new_full_no_gate", profile="new_full", disable_gate=True),
            replace(base, variant="new_full_no_boundary", profile="new_full", disable_boundary=True),
            replace(base, variant="new_full_mean", profile="new_full", view_aggregation="mean"),
        ])

    if stage == "dev128_screen":
        specs: list[RunSpec] = []
        weights = ((0.10, 0.05), (0.25, 0.05), (0.25, 0.10), (0.50, 0.20))
        for task in anime:
            base = _base_spec(stage=stage, task=task, output_root=output_root, resolution=128, steps=1000)
            specs.append(replace(base, variant="l12_bal", profile="l12_bal"))
            for lambda_g, lambda_b in weights:
                for ssim in (0.0, 0.05):
                    label = f"new_full_g{lambda_g:.2f}_b{lambda_b:.2f}_ssim{ssim:.2f}".replace(".", "p")
                    specs.append(
                        replace(base, variant=label, profile="new_full", lambda_g=lambda_g,
                                lambda_b=lambda_b, late_ssim_weight=ssim)
                    )
        return _unique_specs(specs)

    if stage == "dev128_confirm":
        selected_path = Path(selection_path) if selection_path else output_root / "analysis/dev128_selection.json"
        selected = _selected_candidate(selected_path)
        specs = []
        for task in anime:
            base = _base_spec(stage=stage, task=task, output_root=output_root, resolution=128, steps=2000)
            specs.extend([
                replace(base, variant="l12_bal", profile="l12_bal"),
                replace(
                    base, variant="new_full_no_ssim", profile="new_full",
                    lambda_g=selected["lambda_g"], lambda_b=selected["lambda_b"],
                    late_ssim_weight=0.0,
                ),
                replace(
                    base, variant="new_full_ssim", profile="new_full",
                    lambda_g=selected["lambda_g"], lambda_b=selected["lambda_b"],
                    late_ssim_weight=0.05,
                ),
            ])
        return _unique_specs(specs)

    default_frozen_name = "transfer_config.json" if stage == "transfer256" else "frozen_config.json"
    frozen_path = (
        Path(frozen_config_path)
        if frozen_config_path
        else output_root / "analysis" / default_frozen_name
    )
    frozen = load_frozen_config(
        frozen_path,
        expected_source="dev128_confirm" if stage == "transfer256" else "transfer256",
    )
    if stage == "transfer256":
        specs = []
        for task in anime:
            base = _base_spec(stage=stage, task=task, output_root=output_root, resolution=256, steps=2000)
            common = {
                "lambda_g": frozen["lambda_g"], "lambda_b": frozen["lambda_b"],
                "frozen_config_hash": frozen["config_hash"], "frozen_status": frozen["status"],
            }
            specs.extend([
                replace(base, variant="l12_bal", profile="l12_bal", **common),
                replace(base, variant="new_full", profile="new_full", late_ssim_weight=0.0, **common),
                replace(base, variant="new_full_ssim", profile="new_full", late_ssim_weight=0.05, **common),
            ])
        return _unique_specs(specs)

    if stage == "formal256":
        connected = load_frozen_benchmark(Path(benchmark_path), require_files=require_files)
        specs = []
        for task in connected:
            base = _base_spec(
                stage=stage, task=task, output_root=output_root,
                resolution=256, steps=2000, paper_eligible=True,
            )
            specs.extend(_fixed_variants(base, frozen, include_ssim_ablation=True))
        return _unique_specs(specs)
    raise AssertionError(stage)


def trainer_argv(spec: RunSpec, *, resume: bool = False) -> list[str]:
    if not spec.soft_only or not spec.final_step_export:
        raise ValueError("runner accepts soft-only final-step experiments only")
    argv = [
        spec.device, str(spec.run_dir), str(spec.steps), str(spec.learning_rate),
        str(spec.experiment_id), "-sdlist", *spec.targets,
        "-mr", "1", "-tsf", "1", "-tsf4", "0", "-cam", "ortho",
        "-imsz", "256", "--res", str(spec.resolution), "-eres",
        str(spec.export_resolution), "--control-res", str(spec.control_resolution),
        "--sadekar-zdist", "1.7", "--sadekar-volume-extent", "1.7",
        "--sadekar-npts-per-ray", "150", "--sadekar-density-threshold", "0.05",
        "--seed", str(spec.seed), "--profile", spec.profile,
        "--lambda-g", str(spec.lambda_g), "--lambda-b", str(spec.lambda_b),
        "--late-ssim-weight", str(spec.late_ssim_weight),
        "--export-final-step", "--soft-only-eval",
    ]
    if spec.view_aggregation:
        argv.extend(["--view-aggregation", spec.view_aggregation])
    for enabled, flag in (
        (spec.disable_proxy, "--disable-proxy"),
        (spec.disable_gate, "--disable-gate"),
        (spec.disable_boundary, "--disable-boundary"),
        (spec.disable_l1, "--disable-l1"),
        (spec.disable_l2, "--disable-l2"),
        (spec.disable_ssim, "--disable-ssim"),
    ):
        if enabled:
            argv.append(flag)
    if resume:
        argv.append("--resume")
    return argv


def trainer_command(spec: RunSpec, *, resume: bool = False) -> list[str]:
    return [str(PYTHON), "-m", "shadowforge_pipeline.soft_voxel_v2.trainer", *trainer_argv(spec, resume=resume)]


def expected_config_hash(spec: RunSpec) -> str:
    from shadowforge_pipeline.soft_voxel_v2.trainer import build_parser, config_hash, resolved_config

    args = build_parser().parse_args(trainer_argv(spec))
    return config_hash(resolved_config(args))


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _required_final_artifacts(run_dir: Path) -> tuple[Path, ...]:
    artifacts = [
        run_dir / "checkpoint_final.pt",
        run_dir / "occupancy_probability.npy",
        *(run_dir / f"occupancy_binary_tau_{label}.npy" for label in ("0p04", "0p05", "0p06")),
    ]
    for view in range(6):
        artifacts.extend(
            [
                run_dir / f"gt_view_{view}.png",
                run_dir / f"continuous_view_{view}.png",
                *(run_dir / f"binary_tau_{label}_view_{view}.png" for label in ("0p04", "0p05", "0p06")),
            ]
        )
    return tuple(artifacts)


def _metrics_are_complete(metrics: Mapping[str, Any]) -> bool:
    per_view_fields = {
        "tp", "fp", "fn", "tn", "continuous_iou", "binary_iou", "dice",
        "ssim", "fpr", "recall", "boundary_f1_2px", "soft_binary_iou_gap",
    }
    aggregate_fields = {
        "mean_binary_iou", "worst_view_binary_iou", "mean_fpr", "mean_recall",
        "mean_dice", "mean_ssim", "mean_boundary_f1_2px",
        "mean_soft_binary_iou_gap",
    }
    try:
        if metrics.get("density_threshold_semantics") is not True:
            return False
        if not _equal_numeric(metrics.get("primary_threshold"), 0.05):
            return False
        if not all(
            math.isfinite(float(metrics[field]))
            for field in ("final_loss", "final_gradient_norm_pre_clip")
        ):
            return False
        thresholds = metrics["thresholds"]
        for threshold in ("0.04", "0.05", "0.06"):
            report = thresholds[threshold]
            views = report["per_view"]
            aggregate = report["aggregate"]
            if len(views) != 6 or any(not per_view_fields.issubset(view) for view in views):
                return False
            if not aggregate_fields.issubset(aggregate):
                return False
            for view in views:
                if not all(math.isfinite(float(view[field])) for field in per_view_fields):
                    return False
            if not all(math.isfinite(float(aggregate[field])) for field in aggregate_fields):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _equal_numeric(value: Any, expected: float) -> bool:
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def is_completed_run(run_dir: Path, expected_hash: str) -> bool:
    run_dir = Path(run_dir)
    command = _read_json(run_dir / "COMMAND.json")
    status = _read_json(run_dir / "STATUS.json")
    metrics = _read_json(run_dir / "metrics.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    artifacts_complete = all(
        path.is_file() and path.stat().st_size > 0
        for path in _required_final_artifacts(run_dir)
    )
    return bool(
        command and status and metrics and manifest and artifacts_complete
        and _metrics_are_complete(metrics)
        and command.get("config_hash") == expected_hash
        and status.get("state") == "completed"
        and status.get("config_hash") == expected_hash
        and metrics.get("config_hash") == expected_hash
        and manifest.get("config_hash") == expected_hash
        and metrics.get("schema") == "pcraft_soft_loss_v2_metrics_v1"
        and manifest.get("schema") == "pcraft_soft_loss_v2_run_v1"
        and metrics.get("final_step_exported") is True
        and status.get("last_completed_step") == metrics.get("steps")
        and manifest.get("scope", {}).get("soft_only") is True
        and manifest.get("scope", {}).get("final_step_export_only") is True
        and manifest.get("protocol", {}).get("density_threshold_semantics") is True
        and manifest.get("protocol", {}).get("density_thresholds") == [0.04, 0.05, 0.06]
    )


def runner_log_path(spec: RunSpec) -> Path:
    """Return an orchestration log path outside the trainer artifact allowlist."""

    filename = (
        f"{spec.stage}__steps_{spec.steps}__{spec.task}__{spec.variant}__"
        f"seed_{spec.seed}.log"
    )
    return Path(spec.output_root) / "logs" / filename


def _spec_record(spec: RunSpec) -> dict[str, Any]:
    record = asdict(spec)
    record["output_root"] = str(spec.output_root)
    record["run_dir"] = str(spec.run_dir.resolve())
    record["expected_config_hash"] = expected_config_hash(spec)
    record["input_sha256"] = [
        {"path": path, "sha256": _file_sha256(Path(path))}
        for path in spec.targets
    ]
    return record


def write_stage_manifest(
    stage: str,
    specs: Sequence[RunSpec],
    output_root: Path,
    *,
    benchmark_path: Path = BENCHMARK,
    frozen_config_path: Path | None = None,
) -> Path:
    frozen_hashes = {spec.frozen_config_hash for spec in specs if spec.frozen_config_hash}
    if len(frozen_hashes) > 1:
        raise ValueError(f"stage contains multiple frozen configurations: {sorted(frozen_hashes)}")
    frozen_hash = next(iter(frozen_hashes), None)
    payload = {
        "schema": "pcraft_soft_loss_v2_stage_manifest_v1",
        "stage": stage,
        "paper_eligible": stage == "formal256",
        "soft_only": True,
        "seed": SEED,
        "project": str(PROJECT),
        "python": str(PYTHON),
        "benchmark_path": str(Path(benchmark_path).resolve()),
        "benchmark_sha256": _file_sha256(Path(benchmark_path)),
        "frozen_config_hash": frozen_hash,
        "frozen_config_path": (
            str(Path(frozen_config_path).resolve()) if frozen_config_path is not None else None
        ),
        "frozen_config_file_sha256": (
            _file_sha256(Path(frozen_config_path)) if frozen_config_path is not None else None
        ),
        "determinism": "fixed-seed single-run best-effort",
        "bitwise_reproducible": False,
        "nondeterminism_reason": (
            "PyTorch3D CUDA grid_sampler_3d backward has no deterministic implementation."
        ),
        "runs": [_spec_record(spec) for spec in specs],
    }
    payload["manifest_hash"] = _payload_hash(payload)
    destination = Path(output_root) / stage / "stage_manifest.json"
    immutable_json(destination, payload)
    return destination


def write_master_manifest(output_root: Path, benchmark_path: Path) -> Path:
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    if benchmark.get("status") != APPROVED_STATUS:
        raise ValueError("cannot register an unapproved benchmark")
    input_hashes = []
    for case in benchmark.get("cases", []):
        case_id = str(case.get("case_id", ""))
        targets = case.get("target_paths_xyz")
        if not isinstance(targets, list) or len(targets) != 3:
            raise ValueError(f"{case_id}: benchmark target_paths_xyz is incomplete")
        for axis, path_text in zip("xyz", targets):
            path = Path(_rewrite_project_path(str(path_text)))
            input_hashes.append(
                {"task": case_id, "axis": axis, "path": str(path), "sha256": _file_sha256(path)}
            )
    payload = {
        "schema": "pcraft_soft_loss_v2_experiment_manifest_v1",
        "status": "PREREGISTERED",
        "created_for": "soft_loss_v2_20260826",
        "project": str(PROJECT),
        "output_root": str(output_root),
        "benchmark_path": str(Path(benchmark_path).resolve()),
        "benchmark_sha256": _file_sha256(Path(benchmark_path)),
        "benchmark_input_sha256": input_hashes,
        "benchmark_status": benchmark["status"],
        "input_order": "target_paths_xyz:x,y,z",
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "paper_eligible_stage": "formal256",
        "development_stages_paper_eligible": False,
        "determinism": "fixed-seed single-run best-effort",
        "bitwise_reproducible": False,
        "nondeterminism_reason": (
            "PyTorch3D CUDA grid_sampler_3d backward has no deterministic implementation."
        ),
    }
    payload["manifest_hash"] = _payload_hash(payload)
    destination = Path(output_root) / "experiment_manifest.json"
    immutable_json(destination, payload)
    return destination


def run_specs(
    specs: Sequence[RunSpec], *, dry_run: bool, resume: bool, max_runs: int | None,
) -> dict[str, int]:
    counters = {"planned": len(specs), "completed": 0, "skipped": 0, "executed": 0}
    executed = 0
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(PROJECT),
            "PYTHONHASHSEED": str(SEED),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    for spec in specs:
        digest = expected_config_hash(spec)
        if is_completed_run(spec.run_dir, digest):
            counters["skipped"] += 1
            counters["completed"] += 1
            continue
        if max_runs is not None and executed >= max_runs:
            break
        has_partial = spec.run_dir.exists() and any(spec.run_dir.iterdir())
        can_resume = has_partial and (spec.run_dir / "checkpoint_latest.pt").is_file()
        if has_partial and not (resume and can_resume):
            raise RuntimeError(
                f"partial/incompatible run exists; pass --resume only if it has a checkpoint: {spec.run_dir}"
            )
        command = trainer_command(spec, resume=bool(resume and can_resume))
        if dry_run:
            print(_canonical_json({"run_dir": str(spec.run_dir), "command": command, "config_hash": digest}))
            executed += 1
            counters["executed"] += 1
            continue
        log_path = runner_log_path(spec)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Do not create the run directory: the trainer owns it and deliberately
        # refuses a non-empty fresh target.  Resume reuses the existing target.
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=PROJECT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        executed += 1
        counters["executed"] += 1
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command)
        if not is_completed_run(spec.run_dir, digest):
            raise RuntimeError(f"trainer returned success without a complete matching artifact set: {spec.run_dir}")
        counters["completed"] += 1
    return counters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The 'all' stage is not an automatic model-selection loop. It requires "
            "an existing dev128 selection and an explicitly APPROVED_FROZEN config; "
            "use summarize_soft_loss_v2.py at the decision gates."
        ),
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument(
        "--transfer-config", type=Path,
        help="APPROVED dev128_confirm configuration used only to launch transfer256.",
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_runs is not None and args.max_runs < 0:
        raise SystemExit("--max-runs must be non-negative")
    write_master_manifest(args.output_root, args.benchmark)
    if args.stage == "manifest":
        return 0
    if args.stage == "all":
        selection = args.selection or args.output_root / "analysis/dev128_selection.json"
        transfer_config = args.transfer_config or args.output_root / "analysis/transfer_config.json"
        frozen = args.frozen_config or args.output_root / "analysis/frozen_config.json"
        if (
            not Path(selection).is_file()
            or not Path(transfer_config).is_file()
            or not Path(frozen).is_file()
        ):
            raise SystemExit(
                "stage 'all' is fail-closed at human decision gates: first summarize "
                "dev128_screen/dev128_confirm/transfer256, then provide --selection, "
                "an APPROVED --transfer-config, and an APPROVED_FROZEN --frozen-config"
            )
    stages = (
        ("smoke128", "dev128_screen", "dev128_confirm", "transfer256", "formal256")
        if args.stage == "all" else (args.stage,)
    )
    remaining = args.max_runs
    for stage in stages:
        if stage == "transfer256":
            stage_frozen_path = (
                args.transfer_config
                or (args.frozen_config if args.stage != "all" else None)
                or args.output_root / "analysis/transfer_config.json"
            )
        elif stage == "formal256":
            stage_frozen_path = args.frozen_config or args.output_root / "analysis/frozen_config.json"
        else:
            stage_frozen_path = None
        specs = build_stage_specs(
            stage, output_root=args.output_root, benchmark_path=args.benchmark,
            frozen_config_path=stage_frozen_path, selection_path=args.selection,
        )
        write_stage_manifest(
            stage, specs, args.output_root, benchmark_path=args.benchmark,
            frozen_config_path=stage_frozen_path,
        )
        counters = run_specs(specs, dry_run=args.dry_run, resume=args.resume, max_runs=remaining)
        print(_canonical_json({"stage": stage, **counters}))
        if remaining is not None:
            remaining = max(0, remaining - counters["executed"])
            if remaining == 0:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
