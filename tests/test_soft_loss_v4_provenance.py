"""Strict configuration and provenance contracts for soft-loss v4 Task 3."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

import cv2
import numpy as np
import pytest
import torch

import shadowforge_pipeline.soft_voxel_v4.config as config_module
import shadowforge_pipeline.soft_voxel_v4.provenance as provenance_module
from shadowforge_pipeline.soft_voxel_v4.config import (
    BoundFileSnapshot,
    V4BudgetCandidate,
    bootstrap_v4_root,
    build_parser,
    canonical_trainer_argv,
    resolve_run_configuration,
    resolved_config,
    step_config_from_resolved,
    v4_root_manifest_payload,
    validate_resolved_config,
)
from shadowforge_pipeline.soft_voxel_v4.provenance import (
    atomic_checkpoint,
    atomic_json,
    canonical_hash,
    canonical_json_bytes,
    environment_manifest,
    implementation_source_manifest,
    interpreter_manifest,
    publish_json_once,
    sha256_file,
    snapshot_implementation_sources,
    snapshot_regular_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECIFICATION = (
    PROJECT_ROOT
    / ".superpowers/sdd/2026-08-27-soft-loss-v4-implementation/spec.md"
)
SOURCE_FILES = (
    "shadowforge_pipeline/__init__.py",
    "shadowforge_pipeline/soft_voxel_v4/__init__.py",
    "shadowforge_pipeline/soft_voxel_v4/config.py",
    "shadowforge_pipeline/soft_voxel_v4/provenance.py",
    "shadowforge_pipeline/soft_voxel_v4/trainer.py",
    "shadowforge_pipeline/soft_voxel_v4/trainer_step.py",
    "shadowforge_pipeline/soft_voxel_v4/diagnostics.py",
    "shadowforge_pipeline/soft_voxel_v4/gradients.py",
    "shadowforge_pipeline/soft_voxel_v4/losses.py",
    "shadowforge_pipeline/soft_voxel_v3/__init__.py",
    "shadowforge_pipeline/soft_voxel_v3/contours.py",
    "shadowforge_pipeline/soft_voxel_v3/data.py",
    "shadowforge_pipeline/soft_voxel_v3/diagnostics.py",
    "shadowforge_pipeline/soft_voxel_v3/losses.py",
    "shadowforge_pipeline/soft_voxel_v3/lovasz.py",
    "shadowforge_pipeline/soft_voxel_v3/metrics.py",
    "shadowforge_pipeline/soft_voxel_v3/protocol.py",
    "shadowforge_pipeline/soft_voxel_v3/provenance.py",
    "shadowforge_pipeline/soft_voxel_v3/thresholding.py",
)


def _local_canonical_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _published(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_published(payload))


def _semantic(payload: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(payload)
    result["semantic_sha256"] = _local_canonical_hash(result)
    return result


def _write_inputs(root: Path, *, role: str = "development") -> dict[str, object]:
    bootstrap_v4_root(root)
    inputs = root / "inputs"
    inputs.mkdir()
    shadows: list[Path] = []
    for axis, offset in zip("xyz", (0, 1, 2)):
        image = np.zeros((7, 7), dtype=np.uint8)
        image[1:6, 1:6] = 255
        image[offset + 1, offset + 1] = 64 + 32 * offset
        path = inputs / f"{axis}.png"
        assert cv2.imwrite(str(path), image)
        shadows.append(path)
    source_suite = root / "source_suite.json"
    source_suite.write_bytes(b"approved source suite\n")
    task_inputs = [
        {
            "axis": axis,
            "source_path": str(path),
            "resolved_path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for axis, path in zip("xyz", shadows)
    ]
    suite = _semantic(
        {
            "schema": "pcraft_soft_loss_v4_suite_v1",
            "status": "REGISTERED",
            "role": role,
            "source_suite": {
                "path": str(source_suite),
                "sha256": hashlib.sha256(source_suite.read_bytes()).hexdigest(),
                "semantic_sha256": "1" * 64,
            },
            "tasks": [{"case_id": "toy_task", "inputs": task_inputs}],
            "task_count": 1,
            "input_count": 3,
            "input_order": ["x", "y", "z"],
            "protocol": {
                "views": 3,
                "rendered_views": 6,
                "axes": ["x", "y", "z"],
                "mirror_enabled": True,
                "camera": "fixed orthographic TFS",
                "image_size": 256,
                "zdist": 1.7,
                "volume_extent": 1.7,
                "samples_per_ray": 150,
                "seed": 17,
                "density_thresholds": [0.04, 0.05, 0.06],
            },
            "profiles": [
                "l12_bal",
                "dic_only",
                "dts_only",
                "v4_full",
                "v4_no_l1",
                "v4_no_l2",
            ],
        }
    )
    suite_path = root / "manifests" / f"{role}_suite.json"
    _write_json(suite_path, suite)
    specification = root / "spec.md"
    specification.write_bytes(SPECIFICATION.read_bytes())
    return {
        "root": root,
        "shadows": shadows,
        "suite": suite_path,
        "suite_payload": suite,
        "specification": specification,
    }


def _args(
    fixture: dict[str, object],
    *,
    run_kind: str = "experiment",
    profile: str = "l12_bal",
    role: str = "primary",
    steps: int = 200,
    resolution: int = 128,
    resume: bool = False,
    auxiliary: tuple[Path, Path, str] | None = None,
) -> argparse.Namespace:
    argv = [
        "--run-kind",
        run_kind,
        "--device",
        "cpu",
        "--output-root",
        str(fixture["root"]),
        "--run-id",
        "smoke/toy_task/l12_bal/primary_0",
        "--stage",
        "smoke",
        "--task-id",
        "toy_task",
        "--replicate-role",
        role,
        "--replicate-index",
        "0" if role != "technical_repeat" else "1",
        "--steps",
        str(steps),
        "--learning-rate",
        "0.007",
        "--shadow-files",
        *(str(path) for path in fixture["shadows"]),
        "--resolution",
        str(resolution),
        "--export-resolution",
        str(resolution),
        "--profile",
        profile,
        "--suite-manifest",
        str(fixture["suite"]),
        "--specification",
        str(fixture["specification"]),
    ]
    if auxiliary is not None:
        unit, registry, candidate_id = auxiliary
        argv.extend(
            (
                "--unit-coefficients",
                str(unit),
                "--candidate-registry",
                str(registry),
                "--candidate-id",
                candidate_id,
            )
        )
    if resume:
        argv.append("--resume")
    argv.extend(("--export-final-step", "--soft-only-eval"))
    return build_parser().parse_args(argv)


def test_canonical_json_requires_exact_json_builtins_and_string_keys() -> None:
    """Break caught: foreign aliases, nonfinite values, or key coercion enter hashes."""

    first = {"b": [2, 3.0], "a": {"flag": True, "none": None}}
    second = {"a": {"none": None, "flag": True}, "b": [2, 3.0]}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_hash({"x": 1}) != canonical_hash({"x": 1.0})

    class StrAlias(str):
        pass

    class IntAlias(int):
        pass

    invalid = (
        {1: "coerced"},
        {StrAlias("x"): 1},
        {"x": IntAlias(1)},
        {"x": np.int64(1)},
        {"x": np.bool_(True)},
        {"x": float("nan")},
        {"x": torch.tensor(1.0)},
        {"x": object()},
    )
    for payload in invalid:
        with pytest.raises((TypeError, ValueError)):
            canonical_json_bytes(payload)  # type: ignore[arg-type]


def test_snapshot_regular_file_is_nofollow_stable_and_revalidates_identity(
    tmp_path: Path,
) -> None:
    """Break caught: a symlink or replaced inode is trusted after its bytes are bound."""

    root = tmp_path / "root"
    root.mkdir()
    path = root / "input.bin"
    path.write_bytes(b"first")
    snapshot = snapshot_regular_file(path, root=root)
    assert snapshot == BoundFileSnapshot(
        path=str(path),
        sha256=hashlib.sha256(b"first").hexdigest(),
        payload=b"first",
    )
    provenance_module.revalidate_snapshot(snapshot, root=root)

    replacement = root / "replacement.bin"
    replacement.write_bytes(b"first")
    os.replace(replacement, path)
    with pytest.raises(ValueError, match="identity|changed"):
        provenance_module.revalidate_snapshot(snapshot, root=root)

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = root / "link.bin"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        snapshot_regular_file(link, root=root)


def test_source_manifest_is_exact_ordered_whitelist_and_future_file_is_irrelevant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: rglob includes future orchestration or v3 trainer source."""

    manifest, snapshots = snapshot_implementation_sources()
    assert manifest["schema"] == "pcraft_soft_loss_v4_source_manifest_v1"
    assert manifest["algorithm"] == "sha256"
    assert list(manifest["files_sha256"]) == list(SOURCE_FILES)
    assert tuple(snapshot.path for snapshot in snapshots) == tuple(
        str(PROJECT_ROOT / relative) for relative in SOURCE_FILES
    )
    assert "shadowforge_pipeline/soft_voxel_v3/trainer.py" not in manifest[
        "files_sha256"
    ]
    assert all("test" not in relative for relative in manifest["files_sha256"])

    toy_root = tmp_path / "source_root"
    toy_root.mkdir()
    listed = toy_root / "listed.py"
    listed.write_text("LISTED = True\n", encoding="utf-8")
    monkeypatch.setattr(provenance_module, "PROJECT_ROOT", toy_root)
    monkeypatch.setattr(
        provenance_module, "IMPLEMENTATION_SOURCE_FILES", ("listed.py",)
    )
    toy_manifest = implementation_source_manifest()
    (toy_root / "future_unused.py").write_text("UNUSED = True\n", encoding="utf-8")
    assert implementation_source_manifest() == toy_manifest


def test_atomic_and_create_once_publication_are_canonical_and_fail_closed(
    tmp_path: Path,
) -> None:
    """Break caught: create-once publication overwrites foreign bytes or follows links."""

    path = tmp_path / "record.json"
    publish_json_once(path, {"b": 2, "a": 1})
    first = path.read_bytes()
    assert first == b'{"a":1,"b":2}\n'
    publish_json_once(path, {"a": 1, "b": 2})
    assert path.read_bytes() == first
    with pytest.raises(FileExistsError):
        publish_json_once(path, {"a": 2, "b": 2})
    assert path.read_bytes() == first

    mutable = tmp_path / "mutable.json"
    atomic_json(mutable, {"x": 1})
    atomic_json(mutable, {"x": 2})
    assert mutable.read_bytes() == b'{"x":2}\n'
    target = tmp_path / "target.json"
    target.write_bytes(b"outside")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        atomic_json(link, {"x": 3})
    assert target.read_bytes() == b"outside"


def test_root_bootstrap_is_create_once_idempotent_and_concurrent(tmp_path: Path) -> None:
    """Break caught: two binders create different roots or foreign roots are adopted."""

    root = tmp_path / "campaign"
    outcomes: list[object] = []

    def worker() -> None:
        try:
            outcomes.append(bootstrap_v4_root(root))
        except BaseException as error:  # pragma: no cover - diagnostic branch
            outcomes.append(error)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes == [root / "V4_ROOT.json"] * 4
    assert json.loads((root / "V4_ROOT.json").read_text()) == v4_root_manifest_payload()
    first = (root / "V4_ROOT.json").read_bytes()
    assert bootstrap_v4_root(root) == root / "V4_ROOT.json"
    assert (root / "V4_ROOT.json").read_bytes() == first

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "unexpected").write_bytes(b"foreign")
    with pytest.raises(ValueError, match="foreign|non-empty"):
        bootstrap_v4_root(foreign)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        bootstrap_v4_root(link)


def test_environment_and_interpreter_are_locked_allowlisted_and_secret_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: provenance leaks arbitrary environment or overclaims determinism."""

    monkeypatch.setenv("PYTHONHASHSEED", "17")
    monkeypatch.setenv("OPENAI_API_KEY", "never-serialize")
    payload = environment_manifest(torch.device("cpu"))
    assert payload["schema"] == "pcraft_soft_loss_v4_environment_v1"
    assert payload["environment"] == {"PYTHONHASHSEED": "17", "PYTHONPATH": os.environ["PYTHONPATH"]}
    assert "never-serialize" not in json.dumps(payload)
    assert payload["fixed_seed"] is True
    assert payload["determinism"] == "best_effort"
    assert payload["bitwise_reproducible"] is False
    assert "grid_sampler_3d_backward" in payload["nondeterminism_reason"]
    interpreter = interpreter_manifest()
    assert interpreter["path"] == str(Path(sys.executable).resolve())
    assert interpreter["sha256"] == sha256_file(Path(sys.executable).resolve())


def test_parser_rejects_abbreviation_aliases_extra_keys_and_ambiguous_device(
    tmp_path: Path,
) -> None:
    """Break caught: argparse coercion or direct Namespace drift changes identity."""

    fixture = _write_inputs(tmp_path / "campaign")
    args = _args(fixture)
    assert set(vars(args)) == set(config_module.PARSER_NAMESPACE_KEYS)
    assert args.learning_rate == 0.007
    assert canonical_trainer_argv(args, resume=False)[0:3] == (
        config_module.LOCKED_INTERPRETER,
        "-m",
        "shadowforge_pipeline.soft_voxel_v4.trainer",
    )
    assert "--resume" not in canonical_trainer_argv(args, resume=False)
    assert "--resume" in canonical_trainer_argv(args, resume=True)

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(("--run-k", "experiment"))
    with pytest.raises(SystemExit):
        parser.parse_args(("--spec", str(fixture["specification"])))
    malformed = argparse.Namespace(**vars(args), unexpected=True)
    with pytest.raises(ValueError, match="Namespace"):
        resolved_config(malformed)
    for field, value in (
        ("steps", True),
        ("replicate_index", np.int64(0)),
        ("learning_rate", float("nan")),
        ("device", "cuda"),
        ("run_id", "../escape"),
    ):
        bad = argparse.Namespace(**vars(args))
        setattr(bad, field, value)
        with pytest.raises(ValueError):
            resolved_config(bad)


def test_base_and_calibration_configuration_matrix_and_exact_nested_schema(
    tmp_path: Path,
) -> None:
    """Break caught: private inert coefficients leak or role/profile rules drift."""

    fixture = _write_inputs(tmp_path / "campaign")
    base_args = _args(fixture)
    resolved = resolve_run_configuration(base_args)
    payload = resolved.config
    assert resolved_config(base_args) == payload
    assert resolved.config_hash == canonical_hash(payload)
    assert payload["schema"] == "pcraft_soft_loss_v4_config_v1"
    assert payload["stack_version"] == 4
    assert payload["profile"] == {
        "name": "l12_bal",
        "l1_weight": 0.35,
        "l2_weight": 1.0,
        "use_dic": False,
        "use_dts": False,
    }
    assert payload["step_config"] is None
    assert payload["budget_identity"] is None
    assert payload["unit_coefficient_identity"] is None
    assert payload["candidate_registry_identity"] is None
    assert payload["objective"]["formula"] == "l2_weight*L2 + l1_weight*L1"
    assert payload["schedule"]["ramp_start"] == 0.60
    assert payload["schedule"]["ramp_full"] == 0.75
    assert [slot["component"] for slot in payload["schedule"]["slot_cycle"]] == [
        "dic",
        "dts",
        "dic",
        "dts",
    ]
    assert payload["evaluation_thresholds"] == [0.04, 0.05, 0.06]
    assert step_config_from_resolved(payload) is None
    assert validate_resolved_config(payload) == payload
    extra = copy.deepcopy(payload)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="schema|keys"):
        validate_resolved_config(extra)

    calibration = _args(
        fixture,
        run_kind="calibration_source",
        profile="l12_bal",
        role="calibration",
        steps=2000,
        resolution=128,
    )
    calibration.run_id = "calibration/toy_task/source"
    calibration.stage = "calibration"
    cal = resolve_run_configuration(calibration)
    assert cal.config["checkpoint_policy"] == {
        "name": "unit_calibration",
        "commit_interval": 100,
        "retained_updates": [],
    }
    assert cal.config["step_config"] is None
    assert cal.step_config.dic_unit_coefficient == 1.0

    for mutation in (
        ("profile", "dic_only"),
        ("steps", 1000),
        ("resolution", 256),
        ("replicate_role", "primary"),
        ("replicate_index", 1),
    ):
        bad = argparse.Namespace(**vars(calibration))
        setattr(bad, mutation[0], mutation[1])
        with pytest.raises(ValueError):
            resolve_run_configuration(bad)


def test_public_dependency_revalidation_rejects_self_consistently_rehashed_foreign_root(
    tmp_path: Path,
) -> None:
    """Break caught: a recorded digest substitutes for the exact v4 root marker payload."""

    fixture = _write_inputs(tmp_path / "foreign_root")
    resolved = resolve_run_configuration(_args(fixture))
    marker = Path(fixture["root"]) / "V4_ROOT.json"
    _write_json(
        marker,
        {
            "schema": "pcraft_soft_loss_v3_root_v1",
            "namespace": "soft_loss_v3",
            "stack_version": 3,
        },
    )
    forged = copy.deepcopy(resolved.config)
    forged["output_root_identity"]["marker_sha256"] = sha256_file(marker)
    with pytest.raises(ValueError, match="root|marker|namespace|v4"):
        config_module.revalidate_resolved_dependencies(forged)


def test_suite_schema_role_membership_and_xyz_path_byte_cross_binding(
    tmp_path: Path,
) -> None:
    """Break caught: self-consistent hashes allow a substituted task/path/order."""

    fixture = _write_inputs(tmp_path / "campaign")
    args = _args(fixture)
    baseline = resolve_run_configuration(args)
    assert baseline.config["suite_manifest"]["role"] == "development"

    reordered = argparse.Namespace(**vars(args))
    reordered.shadow_files = list(reversed(args.shadow_files))
    with pytest.raises(ValueError, match="x/y/z|suite|ordered|membership"):
        resolve_run_configuration(reordered)

    substituted = fixture["root"] / "inputs" / "substitute.png"
    substituted.write_bytes(Path(args.shadow_files[0]).read_bytes())
    suite = copy.deepcopy(fixture["suite_payload"])
    suite["tasks"][0]["inputs"][0]["source_path"] = str(substituted)
    suite["tasks"][0]["inputs"][0]["resolved_path"] = str(substituted)
    suite.pop("semantic_sha256")
    suite = _semantic(suite)
    _write_json(fixture["suite"], suite)
    with pytest.raises(ValueError, match="source_path|membership|resolved_path"):
        resolve_run_configuration(args)


def test_bound_shadow_bytes_survive_path_replacement_but_revalidation_detects_drift(
    tmp_path: Path,
) -> None:
    """Break caught: runtime reopens targets or publication ignores dependency drift."""

    fixture = _write_inputs(tmp_path / "campaign")
    resolved = resolve_run_configuration(_args(fixture))
    before = tuple(snapshot.payload for snapshot in resolved.shadow_snapshots)
    replacement = np.zeros((7, 7), dtype=np.uint8)
    replacement[3, :] = 255
    assert cv2.imwrite(str(fixture["shadows"][0]), replacement)
    assert tuple(snapshot.payload for snapshot in resolved.shadow_snapshots) == before
    with pytest.raises(ValueError, match="changed|identity|hash"):
        config_module.revalidate_resolved_run(resolved)


def test_auxiliary_unit_registry_candidate_schema_and_cross_root_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: candidates recalibrate, duplicate coefficients, or cross campaigns."""

    fixture = _write_inputs(tmp_path / "campaign")
    root = fixture["root"]
    marker_hash = sha256_file(root / "V4_ROOT.json")
    calibration_run = root / "runs" / "calibration"
    calibration_run.mkdir(parents=True)
    stage = _semantic({"schema": "task4_calibration_stage", "task_id": "toy_task"})
    stage_path = root / "analysis" / "calibration_stage.json"
    _write_json(stage_path, stage)
    calibration_config = {
        "run_kind": "calibration_source",
        "run_identity": {"task_id": "toy_task"},
        "output_root_identity": {
            "path": str(root),
            "marker_path": str(root / "V4_ROOT.json"),
            "marker_sha256": marker_hash,
        },
        "suite_manifest": {
            "path": str(fixture["suite"]),
            "sha256": sha256_file(fixture["suite"]),
            "semantic_sha256": fixture["suite_payload"]["semantic_sha256"],
            "role": "development",
        },
    }

    from shadowforge_pipeline.soft_voxel_v4.trainer import AuthenticatedCompletedRun

    authenticated = AuthenticatedCompletedRun(
        run_dir=str(calibration_run),
        output_root_identity=calibration_config["output_root_identity"],
        config=calibration_config,
        config_hash="2" * 64,
        status_sha256="3" * 64,
        artifact_manifest_sha256="4" * 64,
        artifacts_sha256={},
        transaction_checkpoints=(),
    )
    import shadowforge_pipeline.soft_voxel_v4.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "validate_completed_run", lambda *args, **kwargs: authenticated)
    unit = _semantic(
        {
            "schema": "pcraft_soft_loss_v4_unit_coefficients_v1",
            "status": "APPROVED_FROZEN",
            "calibration_run": {
                "path": str(calibration_run),
                "run_id": "calibration/toy_task/source",
                "config_hash": "2" * 64,
                "status_sha256": "3" * 64,
                "artifact_manifest_sha256": "4" * 64,
            },
            "calibration_stage": {
                "path": str(stage_path),
                "sha256": sha256_file(stage_path),
                "semantic_sha256": stage["semantic_sha256"],
            },
            "calibration_task_identity": {"task_id": "toy_task"},
            "development_suite_identity": {
                "path": str(fixture["suite"]),
                "sha256": sha256_file(fixture["suite"]),
                "semantic_sha256": fixture["suite_payload"]["semantic_sha256"],
                "role": "development",
            },
            "estimator": {
                "name": "registered_task4_estimator",
                "completed_updates": [100, 200],
                "dic_endpoint": 0.05,
                "dts_endpoints": [0.04, 0.06],
                "candidate_independent": True,
            },
            "coefficients": {"dic": 1.25, "dts": 0.75},
        }
    )
    unit_path = root / "analysis" / "unit_coefficients.json"
    _write_json(unit_path, unit)
    candidates = [
        ("r04_r04", 0, 0.04, 0.04),
        ("r08_r08", 1, 0.08, 0.08),
        ("r12_r12", 2, 0.12, 0.12),
        ("r12_r08", 3, 0.12, 0.08),
        ("r08_r12", 4, 0.08, 0.12),
    ]
    registry = _semantic(
        {
            "schema": "pcraft_soft_loss_v4_candidate_registry_v1",
            "status": "REGISTERED",
            "unit_coefficients": {
                "path": str(unit_path),
                "sha256": sha256_file(unit_path),
                "semantic_sha256": unit["semantic_sha256"],
            },
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "order_index": order,
                    "dic_max_ratio": dic,
                    "dts_max_ratio": dts,
                }
                for candidate_id, order, dic, dts in candidates
            ],
        }
    )
    registry_path = root / "manifests" / "candidate_registry.json"
    _write_json(registry_path, registry)
    args = _args(
        fixture,
        profile="v4_full",
        auxiliary=(unit_path, registry_path, "r12_r08"),
    )
    args.run_id = "smoke/toy_task/v4_full/r12_r08/primary_0"
    resolved = resolve_run_configuration(args)
    assert resolved.step_config.dic_unit_coefficient == 1.25
    assert resolved.step_config.dts_unit_coefficient == 0.75
    assert resolved.step_config.dic_max_ratio == 0.12
    assert resolved.step_config.dts_max_ratio == 0.08
    assert resolved.config["budget_identity"] == asdict(
        V4BudgetCandidate("r12_r08", 3, 0.12, 0.08)
    )

    # A preregistered plan is a distinct pre-run artifact and can never be
    # substituted for the evidence-bearing post-run calibration stage.
    plan_path = root / "manifests" / "calibration_plan.json"
    _write_json(plan_path, stage)
    wrong_stage_unit = copy.deepcopy(unit)
    wrong_stage_unit["calibration_stage"] = {
        "path": str(plan_path),
        "sha256": sha256_file(plan_path),
        "semantic_sha256": stage["semantic_sha256"],
    }
    wrong_stage_unit.pop("semantic_sha256")
    wrong_stage_unit = _semantic(wrong_stage_unit)
    _write_json(unit_path, wrong_stage_unit)
    wrong_stage_registry = copy.deepcopy(registry)
    wrong_stage_registry["unit_coefficients"] = {
        "path": str(unit_path),
        "sha256": sha256_file(unit_path),
        "semantic_sha256": wrong_stage_unit["semantic_sha256"],
    }
    wrong_stage_registry.pop("semantic_sha256")
    _write_json(registry_path, _semantic(wrong_stage_registry))
    with pytest.raises(ValueError, match="post-run|calibration_stage|canonical"):
        resolve_run_configuration(args)

    _write_json(unit_path, unit)
    _write_json(registry_path, registry)

    attacked = copy.deepcopy(unit)
    attacked["candidate_id"] = "r12_r08"
    attacked.pop("semantic_sha256")
    attacked = _semantic(attacked)
    _write_json(unit_path, attacked)
    with pytest.raises(ValueError, match="exact|candidate|schema"):
        resolve_run_configuration(args)


def test_every_scientific_or_byte_identity_change_changes_config_hash(tmp_path: Path) -> None:
    """Break caught: one nested scientific field is omitted from canonical identity."""

    fixture = _write_inputs(tmp_path / "campaign")
    payload = resolve_run_configuration(_args(fixture)).config
    original = canonical_hash(payload)
    mutations = (
        ("steps", 1000),
        ("learning_rate", 0.008),
        ("seed", 18),
        ("evaluation_thresholds", [0.04, 0.05]),
        ("profile", {**payload["profile"], "l1_weight": 0.0}),
        ("implementation", {**payload["implementation"], "bundle_sha256": "f" * 64}),
    )
    for key, value in mutations:
        changed = copy.deepcopy(payload)
        changed[key] = value
        assert canonical_hash(changed) != original


def test_public_records_are_frozen_exact_and_snapshot_bytes_never_serialize() -> None:
    """Break caught: snapshot payloads leak into scientific public mappings."""

    assert [field.name for field in fields(BoundFileSnapshot)] == [
        "path",
        "sha256",
        "payload",
    ]
    assert [field.name for field in fields(V4BudgetCandidate)] == [
        "candidate_id",
        "order_index",
        "dic_max_ratio",
        "dts_max_ratio",
    ]
    snapshot = BoundFileSnapshot("/tmp/a", "0" * 64, b"secret-bytes")
    with pytest.raises(TypeError):
        canonical_json_bytes({"snapshot": snapshot})
