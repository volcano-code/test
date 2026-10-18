"""Frozen runner for the preregistered Soft Loss v4 ten-task ablation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import uuid
from typing import Iterator, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.soft_loss_v4_registry import load_v4_suite_identity
from shadowforge_pipeline.soft_voxel_v4.config import (
    build_parser as _task3_parser,
    canonical_trainer_argv,
    resolve_run_configuration,
)
from shadowforge_pipeline.soft_voxel_v4.provenance import (
    canonical_hash,
    canonical_json_bytes,
    publish_json_once,
)
from shadowforge_pipeline.soft_voxel_v4.trainer import (
    validate_completed_run,
    validate_resumable_run,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FORMAL = tuple(f"connected_{index:02d}" for index in range(1, 11))
_PROFILES = ("l12_bal", "dic_only", "dts_only", "v4_full")
_STAGE = "ablation10_128"
_CANDIDATE_ID = "r08_r08"
_FROZEN_CONSTANTS = {
    "seed": 17,
    "optimizer": "Adam",
    "learning_rate": 0.007,
    "resolution": 128,
    "export_resolution": 128,
    "steps": 2000,
    "stage": _STAGE,
    "replicate_role": "primary",
    "replicate_index": 0,
}


@dataclass(frozen=True)
class AblationRunSpec:
    """One resolved, identity-bound primary cell in the frozen 10x4 matrix."""

    task_id: str
    profile: str
    stage: str
    run_id: str
    run_dir: Path
    namespace: argparse.Namespace
    resolved_config: dict[str, object]
    config_hash: str
    fresh_argv: tuple[str, ...]
    resume_argv: tuple[str, ...]


@contextmanager
def _controlled_pythonpath() -> Iterator[None]:
    """Match the registered Task 3 environment to the locked child environment."""

    prior = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(_PROJECT_ROOT)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = prior


def _resolve_task3(namespace: argparse.Namespace) -> object:
    """Resolve one Task 3 namespace under the exact child Python path."""

    with _controlled_pythonpath():
        return resolve_run_configuration(namespace)


def _task3_namespace(
    *,
    output_root: Path,
    formal_suite: Path,
    specification: Path,
    unit_coefficients: Path,
    candidate_registry: Path,
    device: str,
    task_id: str,
    profile: str,
    shadow_files: tuple[str, str, str],
) -> argparse.Namespace:
    """Construct the exact Task 3 parser namespace for one approved cell."""

    run_id = f"runs/ablation10/{task_id}/{profile}"
    argv = [
        "--run-kind", "experiment",
        "--device", device,
        "--output-root", str(output_root),
        "--run-id", run_id,
        "--stage", _STAGE,
        "--task-id", task_id,
        "--replicate-role", "primary",
        "--replicate-index", "0",
        "--steps", "2000",
        "--learning-rate", "0.007",
        "--shadow-files", *shadow_files,
        "--resolution", "128",
        "--export-resolution", "128",
        "--profile", profile,
        "--suite-manifest", str(formal_suite),
        "--specification", str(specification),
    ]
    if profile != "l12_bal":
        argv.extend(
            (
                "--unit-coefficients", str(unit_coefficients),
                "--candidate-registry", str(candidate_registry),
                "--candidate-id", _CANDIDATE_ID,
            )
        )
    argv.extend(("--export-final-step", "--soft-only-eval"))
    return _task3_parser().parse_args(argv)


def _validated_formal_tasks(formal_suite: Path) -> tuple[object, ...]:
    suite = load_v4_suite_identity(Path(formal_suite), "formal")
    tasks = tuple(suite.tasks)
    if suite.role != "formal" or tuple(task.case_id for task in tasks) != _FORMAL:
        raise ValueError("formal suite task order/count differs from ablation10 registration")
    for task in tasks:
        if len(task.inputs) != 3 or tuple(item.axis for item in task.inputs) != ("x", "y", "z"):
            raise ValueError("formal suite task input identity/order is malformed")
    return tasks


def build_ablation_specs(
    *,
    output_root: Path,
    formal_suite: Path,
    specification: Path,
    unit_coefficients: Path,
    candidate_registry: Path,
    device: str,
) -> tuple[AblationRunSpec, ...]:
    """Resolve all forty preregistered cells through Task 3's public surface."""

    root = Path(output_root)
    tasks = _validated_formal_tasks(Path(formal_suite))
    specs: list[AblationRunSpec] = []
    for task in tasks:
        shadows = tuple(item.resolved_path for item in task.inputs)
        for profile in _PROFILES:
            namespace = _task3_namespace(
                output_root=root,
                formal_suite=Path(formal_suite),
                specification=Path(specification),
                unit_coefficients=Path(unit_coefficients),
                candidate_registry=Path(candidate_registry),
                device=device,
                task_id=task.case_id,
                profile=profile,
                shadow_files=shadows,
            )
            resolved = _resolve_task3(namespace)
            config = json.loads(canonical_json_bytes(resolved.config))
            if canonical_hash(config) != resolved.config_hash:
                raise ValueError("Task 3 resolved configuration hash drifted")
            run_id = f"runs/ablation10/{task.case_id}/{profile}"
            if config.get("run_identity", {}).get("run_id") != run_id:
                raise ValueError("Task 3 resolved run ID differs from ablation10 registration")
            specs.append(
                AblationRunSpec(
                    task_id=task.case_id,
                    profile=profile,
                    stage=_STAGE,
                    run_id=run_id,
                    run_dir=root.joinpath(*run_id.split("/")),
                    namespace=namespace,
                    resolved_config=config,
                    config_hash=resolved.config_hash,
                    fresh_argv=canonical_trainer_argv(namespace, resume=False),
                    resume_argv=canonical_trainer_argv(namespace, resume=True),
                )
            )
    _validate_specs(specs)
    return tuple(specs)


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{label} must be an exact mapping")
    return value


def _no_symlink_components(path: Path, *, label: str) -> None:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate != Path(os.path.abspath(candidate)):
        raise ValueError(f"{label} must be absolute and canonical")
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink component: {current}")


def _canonical_output_root(value: object) -> Path:
    if type(value) is not str:
        raise ValueError("output root must be an exact string")
    root = Path(value)
    _no_symlink_components(root, label="output root")
    try:
        metadata = root.lstat()
    except OSError as error:
        raise ValueError("output root is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("output root must be a directory")
    if root.resolve(strict=True) != root:
        raise ValueError("output root is not canonical")
    return root


def _cell_paths(spec: AblationRunSpec, *, root: Path) -> tuple[Path, Path]:
    expected_run = root / "runs" / "ablation10" / spec.task_id / spec.profile
    expected_log = root / "logs" / "ablation10" / spec.task_id / spec.profile
    if spec.run_dir != expected_run:
        raise ValueError("ablation10 run directory is outside its canonical cell path")
    for path, label in ((expected_run, "run path"), (expected_log, "log path")):
        try:
            path.relative_to(root)
        except ValueError as error:  # pragma: no cover - defensive after literals above
            raise ValueError(f"{label} escapes output root") from error
        _no_symlink_components(path, label=label)
    return expected_run, expected_log


def _validate_specs(specs: Sequence[AblationRunSpec]) -> Path:
    expected = [(task_id, profile) for task_id in _FORMAL for profile in _PROFILES]
    if [(item.task_id, item.profile) for item in specs] != expected:
        raise ValueError("ablation10 matrix order/count differs from registration")
    root: Path | None = None
    suite_identity: object | None = None
    specification_identity: object | None = None
    unit_identity: object | None = None
    registry_identity: object | None = None
    candidate_row: object | None = None
    for spec in specs:
        if not isinstance(spec, AblationRunSpec):
            raise ValueError("ablation10 spec has the wrong type")
        if type(spec.namespace) is not argparse.Namespace or type(spec.resolved_config) is not dict:
            raise ValueError("ablation10 spec namespace/config has the wrong type")
        if spec.stage != _STAGE or spec.run_id != f"runs/ablation10/{spec.task_id}/{spec.profile}":
            raise ValueError("ablation10 run identity drifted")
        spec_root = _canonical_output_root(spec.namespace.output_root)
        if root is None:
            root = spec_root
        elif spec_root != root:
            raise ValueError("ablation10 specs belong to different output roots")
        if spec.run_dir != spec_root.joinpath(*spec.run_id.split("/")):
            raise ValueError("ablation10 run directory drifted")
        _cell_paths(spec, root=spec_root)
        if canonical_hash(spec.resolved_config) != spec.config_hash:
            raise ValueError("ablation10 resolved configuration hash drifted")
        config = spec.resolved_config
        run_identity = _mapping(config.get("run_identity"), label="run identity")
        if run_identity.get("run_id") != spec.run_id:
            raise ValueError("resolved configuration has the wrong run ID")
        if run_identity.get("run_dir") != str(spec.run_dir):
            raise ValueError("resolved configuration has the wrong run directory")
        output_identity = _mapping(
            config.get("output_root_identity"), label="output root identity"
        )
        if output_identity.get("path") != str(spec_root):
            raise ValueError("resolved configuration has the wrong output root")
        if spec.namespace.resume is not False:
            raise ValueError("ablation10 namespace must be fresh while frozen")
        if spec.fresh_argv != canonical_trainer_argv(spec.namespace, resume=False):
            raise ValueError("ablation10 fresh argv differs from Task 3 canonical argv")
        if spec.resume_argv != canonical_trainer_argv(spec.namespace, resume=True):
            raise ValueError("ablation10 resume argv differs from Task 3 canonical argv")
        current_suite = _mapping(config.get("suite_manifest"), label="suite identity")
        current_specification = _mapping(config.get("specification"), label="specification identity")
        if suite_identity is None:
            suite_identity = current_suite
            specification_identity = current_specification
        elif current_suite != suite_identity or current_specification != specification_identity:
            raise ValueError("suite/specification identity differs across ablation10 cells")
        if spec.profile == "l12_bal":
            if (
                spec.namespace.unit_coefficients is not None
                or spec.namespace.candidate_registry is not None
                or spec.namespace.candidate_id is not None
                or config.get("unit_coefficient_identity") is not None
                or config.get("candidate_registry_identity") is not None
                or config.get("budget_identity") is not None
            ):
                raise ValueError("shared baseline must carry no unit, registry, or candidate identity")
            continue
        if spec.namespace.candidate_id != _CANDIDATE_ID:
            raise ValueError("auxiliary profile candidate is not frozen r08_r08")
        current_unit = _mapping(config.get("unit_coefficient_identity"), label="unit identity")
        current_registry = _mapping(config.get("candidate_registry_identity"), label="registry identity")
        current_candidate = _mapping(config.get("budget_identity"), label="candidate row")
        if current_candidate.get("candidate_id") != _CANDIDATE_ID:
            raise ValueError("auxiliary resolved candidate is not frozen r08_r08")
        if unit_identity is None:
            unit_identity = current_unit
            registry_identity = current_registry
            candidate_row = current_candidate
        elif (
            current_unit != unit_identity
            or current_registry != registry_identity
            or current_candidate != candidate_row
        ):
            raise ValueError("auxiliary dependency identity differs across ablation10 cells")
    if root is None:  # pragma: no cover - exact matrix check prevents this
        raise ValueError("ablation10 matrix is empty")
    _validate_specs_against_frozen_task3_cells(specs, root=root)
    return root


def _validate_specs_against_frozen_task3_cells(
    specs: Sequence[AblationRunSpec], *, root: Path
) -> None:
    """Resolve the immutable forty-cell matrix anew and compare every public field."""

    first = specs[0]
    first_auxiliary = next(spec for spec in specs if spec.profile != "l12_bal")
    try:
        formal_suite = Path(first.namespace.suite_manifest)
        specification = Path(first.namespace.specification)
        unit_coefficients = Path(first_auxiliary.namespace.unit_coefficients)
        candidate_registry = Path(first_auxiliary.namespace.candidate_registry)
        device = first.namespace.device
    except (AttributeError, TypeError) as error:
        raise ValueError("ablation10 namespace lacks the frozen Task 3 context") from error
    tasks = _validated_formal_tasks(formal_suite)
    for task_index, task in enumerate(tasks):
        shadows = tuple(item.resolved_path for item in task.inputs)
        for profile_index, profile in enumerate(_PROFILES):
            spec = specs[(task_index * len(_PROFILES)) + profile_index]
            expected_namespace = _task3_namespace(
                output_root=root,
                formal_suite=formal_suite,
                specification=specification,
                unit_coefficients=unit_coefficients,
                candidate_registry=candidate_registry,
                device=device,
                task_id=task.case_id,
                profile=profile,
                shadow_files=shadows,
            )
            if canonical_json_bytes(vars(spec.namespace)) != canonical_json_bytes(
                vars(expected_namespace)
            ):
                raise ValueError("ablation10 namespace differs from the frozen cell registration")
            expected = _resolve_task3(expected_namespace)
            expected_config = json.loads(canonical_json_bytes(expected.config))
            expected_hash = canonical_hash(expected_config)
            if expected_hash != expected.config_hash:
                raise ValueError("Task 3 expected configuration hash drifted")
            if canonical_json_bytes(spec.resolved_config) != canonical_json_bytes(expected_config):
                raise ValueError(
                    "ablation10 resolved configuration differs from the frozen cell registration"
                )
            if spec.config_hash != expected_hash:
                raise ValueError("ablation10 configuration hash differs from the frozen cell registration")
            if spec.fresh_argv != canonical_trainer_argv(expected_namespace, resume=False):
                raise ValueError("ablation10 fresh argv differs from the frozen cell registration")
            if spec.resume_argv != canonical_trainer_argv(expected_namespace, resume=True):
                raise ValueError("ablation10 resume argv differs from the frozen cell registration")


def _manifest_constants(specs: Sequence[AblationRunSpec]) -> dict[str, object]:
    """Derive the published constants from resolved cells and fail on any mismatch."""

    def constants_for(spec: AblationRunSpec) -> dict[str, object]:
        config = spec.resolved_config
        run_identity = _mapping(config.get("run_identity"), label="run identity")
        optimizer = _mapping(config.get("optimizer"), label="optimizer identity")
        return {
            "seed": config.get("seed"),
            "optimizer": optimizer.get("name"),
            "learning_rate": config.get("learning_rate"),
            "resolution": config.get("resolution"),
            "export_resolution": config.get("export_resolution"),
            "steps": config.get("steps"),
            "stage": run_identity.get("stage"),
            "replicate_role": run_identity.get("replicate_role"),
            "replicate_index": run_identity.get("replicate_index"),
        }

    derived = constants_for(specs[0])
    if canonical_json_bytes(derived) != canonical_json_bytes(_FROZEN_CONSTANTS):
        raise ValueError("ablation10 derived constants differ from the frozen registration")
    for spec in specs[1:]:
        if canonical_json_bytes(constants_for(spec)) != canonical_json_bytes(derived):
            raise ValueError("ablation10 derived constants differ across matrix cells")
    return derived


def _revalidate_spec(spec: AblationRunSpec) -> None:
    """Rebuild and byte-compare a cell before/after lifecycle transitions."""

    resolved = _resolve_task3(spec.namespace)
    current = json.loads(canonical_json_bytes(resolved.config))
    if canonical_json_bytes(current) != canonical_json_bytes(spec.resolved_config):
        raise ValueError("Task 3 resolved configuration drifted from the ablation10 manifest")
    if resolved.config_hash != spec.config_hash:
        raise ValueError("Task 3 resolved configuration hash drifted from the ablation10 manifest")
    if canonical_trainer_argv(spec.namespace, resume=False) != spec.fresh_argv:
        raise ValueError("Task 3 fresh argv drifted from the ablation10 manifest")
    if canonical_trainer_argv(spec.namespace, resume=True) != spec.resume_argv:
        raise ValueError("Task 3 resume argv drifted from the ablation10 manifest")


def _revalidate_specs(specs: Sequence[AblationRunSpec]) -> None:
    for spec in specs:
        _revalidate_spec(spec)


def _manifest_payload(specs: Sequence[AblationRunSpec]) -> dict[str, object]:
    _validate_specs(specs)
    first = specs[0]
    first_auxiliary = next(spec for spec in specs if spec.profile != "l12_bal")
    matrix = [
        {
            "task_id": spec.task_id,
            "profile": spec.profile,
            "run_id": spec.run_id,
            "run_dir": str(spec.run_dir),
            "config_hash": spec.config_hash,
            "fresh_argv": list(spec.fresh_argv),
            "resume_argv": list(spec.resume_argv),
            "resolved_config": spec.resolved_config,
        }
        for spec in specs
    ]
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v4_ablation10_v1",
        "suite_identity": first.resolved_config["suite_manifest"],
        "specification_identity": first.resolved_config["specification"],
        "unit_coefficient_identity": first_auxiliary.resolved_config["unit_coefficient_identity"],
        "candidate_registry_identity": first_auxiliary.resolved_config["candidate_registry_identity"],
        "candidate_row": first_auxiliary.resolved_config["budget_identity"],
        "constants": _manifest_constants(specs),
        "matrix": matrix,
        "decision_policy": {
            "threshold": 0.05,
            "view_aggregation": "six_view_equal_weight",
            "statistical_unit": "task",
            "task_count": 10,
            "task_aggregation": "ordered_task_deltas_then_sum_divide_10_no_tolerance",
            "contrasts": {
                "dic": {"treatment": "v4_full", "control": "dts_only"},
                "dts": {"treatment": "v4_full", "control": "dic_only"},
                "full_vs_base": {"treatment": "v4_full", "control": "l12_bal"},
            },
            "component_key": {
                "mean_delta_iou_min": 0.01,
                "mean_delta_dice_min": 0.005,
                "positive_task_count_min": 8,
                "task_delta_iou_min": -0.005,
                "test": "one_sided_exact_sign_flip_task_equal_mean_inclusive_tail",
                "correction": "holm_dic_dts_iou_only",
                "adjusted_p_strict_max": 0.05,
            },
            "overall_quality": {
                "full_mean_iou_min": 0.95,
                "full_task_iou_min": 0.94,
                "full_base_mean_iou_min": -0.001,
                "full_base_mean_dice_min": -0.001,
            },
            "safety": {
                "profiles_vs_base": ["v4_full", "dic_only", "dts_only"],
                "task_recall_delta_min": -0.005,
                "mean_fpr_delta_max": 0.001,
            },
            "engineering": {
                "completed_runs": 40,
                "auxiliary_eligible_slots_min": 1,
                "engineering_gate_status": "PASS",
                "projected_zero_fraction_max": 0.10,
                "each_enabled_component_has_positive_final_norm": True,
                "required_components": {
                    "dic_only": ["dic"],
                    "dts_only": ["dts"],
                    "v4_full": ["dic", "dts"],
                },
            },
            "labels": [
                "BOTH_KEY",
                "DIC_KEY",
                "DTS_KEY",
                "NEITHER_KEY",
                "HARMFUL_OR_ANTAGONISTIC",
                "INCONCLUSIVE",
            ],
            "invalid_evidence": "raise_without_publication",
            "below_definition": "mean_iou_below_mcid_or_mean_dice_below_mcid",
            "ambiguous_definition": "not_key_and_not_below",
            "label_precedence": [
                "engineering_stop_or_excess_zero_or_no_activation=>INCONCLUSIVE",
                "quality_or_safety_failure_or_negative_mean_iou_or_dice=>HARMFUL_OR_ANTAGONISTIC",
                "component_ambiguous=>INCONCLUSIVE",
                "both_key=>BOTH_KEY",
                "dic_key_and_dts_below=>DIC_KEY",
                "dts_key_and_dic_below=>DTS_KEY",
                "both_below=>NEITHER_KEY",
            ],
            "fixed_final_step": True,
            "best_checkpoint": False,
            "candidate_selection": "fixed_r08_r08_for_auxiliary_profiles",
            "candidate_search": False,
            "stage_dag": False,
            "repeats": False,
            "appendix": False,
            "resolution_256": False,
            "outcome_read": False,
        },
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    return payload


def write_ablation_manifest(output_root: Path, specs: Sequence[AblationRunSpec]) -> Path:
    """Create-once publish the complete, semantic-hashed ablation registration."""

    root = _canonical_output_root(str(Path(output_root)))
    spec_root = _validate_specs(specs)
    if root != spec_root:
        raise ValueError("manifest output root differs from the resolved ablation10 spec root")
    manifest = root / "manifests" / "ablation10.json"
    _no_symlink_components(manifest, label="manifest path")
    _revalidate_specs(specs)
    payload = _manifest_payload(specs)
    publish_json_once(manifest, payload)
    return manifest


def _authenticate_completed(spec: AblationRunSpec) -> object:
    return validate_completed_run(spec.run_dir, expected_config=spec.resolved_config)


def _attempt_logs(spec: AblationRunSpec) -> tuple[object, object]:
    root = _canonical_output_root(spec.namespace.output_root)
    _, directory = _cell_paths(spec, root=root)
    directory.mkdir(parents=True, exist_ok=True)
    _no_symlink_components(directory, label="log path")
    attempt = f"attempt_{uuid.uuid4().hex}"
    stdout = (directory / f"{attempt}.stdout.log").open("xb")
    try:
        stderr = (directory / f"{attempt}.stderr.log").open("xb")
    except BaseException:
        stdout.close()
        raise
    return stdout, stderr


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_PROJECT_ROOT)
    return environment


def run_specs(specs: Sequence[AblationRunSpec], *, resume: bool = False) -> dict[str, int]:
    """Sequentially launch only fresh or Task 3-authenticated resumable cells."""

    root = _validate_specs(specs)
    _revalidate_specs(specs)
    launched = skipped = resumed = 0
    for spec in specs:
        _cell_paths(spec, root=root)
        _revalidate_spec(spec)
        child_argv: tuple[str, ...]
        if os.path.lexists(spec.run_dir):
            try:
                _authenticate_completed(spec)
            except ValueError:
                if not resume:
                    raise
                validate_resumable_run(spec.run_dir, expected_config=spec.resolved_config)
                child_argv = spec.resume_argv
                resumed += 1
            else:
                skipped += 1
                _revalidate_spec(spec)
                continue
        else:
            child_argv = spec.fresh_argv
        stdout, stderr = _attempt_logs(spec)
        try:
            child = subprocess.run(
                child_argv,
                cwd=_PROJECT_ROOT,
                env=_child_environment(),
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        finally:
            stdout.close()
            stderr.close()
        if child.returncode != 0:
            raise RuntimeError(f"ablation10 child {spec.run_id} exited with exit code {child.returncode}")
        _authenticate_completed(spec)
        launched += 1
        _revalidate_spec(spec)
    return {"launched": launched, "skipped": skipped, "resumed": resumed}


def build_parser() -> argparse.ArgumentParser:
    """Build the small orchestration-only command line parser."""

    parser = argparse.ArgumentParser(
        description="Run the frozen Soft Loss v4 ablation10 matrix", allow_abbrev=False
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--formal-suite", required=True)
    parser.add_argument("--specification", required=True)
    parser.add_argument("--unit-coefficients", required=True)
    parser.add_argument("--candidate-registry", required=True)
    parser.add_argument("--device", required=True)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--dry-run", action="store_true")
    selector.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = build_ablation_specs(
        output_root=Path(args.output_root),
        formal_suite=Path(args.formal_suite),
        specification=Path(args.specification),
        unit_coefficients=Path(args.unit_coefficients),
        candidate_registry=Path(args.candidate_registry),
        device=args.device,
    )
    write_ablation_manifest(Path(args.output_root), specs)
    if args.dry_run:
        print("dry-run: 40 dry cells")
        return 0
    result = run_specs(specs, resume=args.resume)
    print(
        "ablation10: "
        f"launched={result['launched']} skipped={result['skipped']} resumed={result['resumed']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
