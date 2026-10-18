"""Fail-closed contracts for v4 unit-gradient calibration."""

from __future__ import annotations

import argparse
import ast
from dataclasses import FrozenInstanceError
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys

import pytest
import torch

import shadowforge_pipeline.soft_voxel_v4.calibration as calibration
from shadowforge_pipeline.soft_voxel_v4.calibration import (
    UnitGradientMeasurement,
    build_calibration_parser,
    build_calibration_plan,
    build_calibration_stage,
    calibration_main,
    collect_unit_gradient_measurements,
    derive_unit_coefficients,
    publish_calibration_bundle,
    validate_calibration_plan,
    validate_calibration_stage,
    validate_unit_coefficients,
    write_calibration_plan,
)
from shadowforge_pipeline.soft_voxel_v4.provenance import canonical_hash, canonical_json_bytes
from shadowforge_pipeline.soft_voxel_v4.trainer_step import V4StepConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPDATES = (1200, 1500, 1800)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measurements() -> tuple[UnitGradientMeasurement, ...]:
    values = (
        (10.0, 1.0, 100.0, 2.0),
        (20.0, 4.0, 200.0, 3.0),
        (30.0, 10.0, 300.0, 4.0),
    )
    return tuple(
        UnitGradientMeasurement(
            completed_updates=update,
            transaction=f"transactions/step_{update:08d}",
            transaction_manifest_sha256=f"{index + 1:x}" * 64,
            checkpoint_path=f"transactions/step_{update:08d}/checkpoint.pt",
            checkpoint_sha256=f"{index + 4:x}" * 64,
            g_base=base,
            g_dic=dic,
            g_dts_004=g004,
            g_dts_006=g006,
        )
        for index, (update, (base, dic, g004, g006)) in enumerate(zip(UPDATES, values))
    )


def test_measurement_dataclass_is_exact_frozen_and_formula_uses_six_dts_values() -> None:
    """Break caught: calibration changes identity fields or drops an endpoint."""

    measurements = _measurements()
    assert tuple(measurements[0].__dataclass_fields__) == (
        "completed_updates", "transaction", "transaction_manifest_sha256",
        "checkpoint_path", "checkpoint_sha256", "g_base", "g_dic",
        "g_dts_004", "g_dts_006",
    )
    with pytest.raises(FrozenInstanceError):
        measurements[0].g_base = 99.0
    result = derive_unit_coefficients(measurements)
    assert result == {
        "dic": median([10.0, 20.0, 30.0]) / (median([1.0, 4.0, 10.0]) + 1e-12),
        "dts": median([10.0, 20.0, 30.0]) / (
            median([100.0, 2.0, 200.0, 3.0, 300.0, 4.0]) + 1e-12
        ),
    }
    assert all(type(value) is float for value in result.values())


def test_formula_next_float_boundaries_are_not_clipped_or_rounded() -> None:
    """Break caught: coefficients are rounded, capped, or use a candidate ratio."""

    rows = list(_measurements())
    payload = dict(rows[1].__dict__)
    payload["g_dic"] = math.nextafter(4.0, math.inf)
    rows[1] = UnitGradientMeasurement(**payload)
    result = derive_unit_coefficients(rows)
    expected = 20.0 / (math.nextafter(4.0, math.inf) + 1e-12)
    assert result["dic"] == expected
    assert result["dic"] != 0.04 * expected


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.reverse(),
        lambda rows: rows.__setitem__(1, rows[0]),
        lambda rows: rows.__setitem__(0, UnitGradientMeasurement(**{**rows[0].__dict__, "completed_updates": True})),
        lambda rows: rows.__setitem__(0, UnitGradientMeasurement(**{**rows[0].__dict__, "g_base": "1"})),
        lambda rows: rows.__setitem__(0, UnitGradientMeasurement(**{**rows[0].__dict__, "g_dic": 0.0})),
        lambda rows: rows.__setitem__(0, UnitGradientMeasurement(**{**rows[0].__dict__, "g_dts_004": float("nan")})),
        lambda rows: rows.__setitem__(0, UnitGradientMeasurement(**{**rows[0].__dict__, "g_dts_006": float("inf")})),
    ],
)
def test_formula_rejects_wrong_identity_types_and_nonpositive_norms(mutation: object) -> None:
    """Break caught: malformed raw evidence is sorted/coerced/defaulted."""

    rows = list(_measurements())
    mutation(rows)
    with pytest.raises((TypeError, ValueError)):
        derive_unit_coefficients(rows)


def test_formula_rejects_overflow_and_duplicate_checkpoint_identity() -> None:
    """Break caught: overflow or duplicated transaction bytes silently survive."""

    rows = list(_measurements())
    rows[1] = UnitGradientMeasurement(**{**rows[1].__dict__, "checkpoint_path": rows[0].checkpoint_path})
    with pytest.raises(ValueError, match="duplicate|checkpoint"):
        derive_unit_coefficients(rows)
    rows = list(_measurements())
    for index in range(3):
        rows[index] = UnitGradientMeasurement(**{**rows[index].__dict__, "g_base": 1e308, "g_dic": 5e-324})
    with pytest.raises(ValueError, match="finite|positive|overflow"):
        derive_unit_coefficients(rows)


def test_raw_gradient_helper_matches_four_task2_branches_and_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Break caught: calibration applies anchor/schedule/candidate/clip or mutates logits."""

    # Keep sigmoid(logits) near the registered endpoints so the exact
    # temperature-.01 threshold STE is not mathematically saturated.
    logits = torch.tensor([[[[-2.9, -3.3], [-3.0, -3.5]]]], dtype=torch.float64, requires_grad=True)
    before = logits.detach().clone()
    target = torch.tensor([[[True, False], [True, False]]])
    bands = type("Bands", (), {
        "inner": target,
        "outer": ~target,
    })()
    class Protocol:
        def render_all(self, density: torch.Tensor) -> torch.Tensor:
            return density[0, 0].unsqueeze(0)
    monkeypatch.setattr(calibration, "_contour_bands", lambda _: bands)
    norms = calibration._raw_gradient_norms(
        logits=logits,
        protocol=Protocol(),
        targets=target.to(dtype=torch.float64),
    )
    assert set(norms) == {"g_base", "g_dic", "g_dts_004", "g_dts_006"}
    assert all(type(value) is float and math.isfinite(value) and value > 0.0 for value in norms.values())
    assert torch.equal(logits.detach(), before)
    assert logits.grad is None


@pytest.mark.parametrize("kind", ["disconnected", "zero", "nan", "wrong_dtype"])
def test_raw_gradient_helper_rejects_bad_graph_norm_and_dtype(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Break caught: unusable calibration gradients become a plausible coefficient."""

    logits = torch.zeros((1, 1, 2, 2), dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[[True, False], [True, False]]], dtype=torch.float64)
    class Protocol:
        def render_all(self, density: torch.Tensor) -> torch.Tensor:
            if kind == "disconnected":
                return torch.ones((1, 2, 2), dtype=density.dtype)
            if kind == "zero":
                return density.sum() * 0.0 + torch.ones((1, 2, 2), dtype=density.dtype)
            if kind == "nan":
                return density[0, 0].unsqueeze(0) * float("nan")
            return density[0, 0].unsqueeze(0).to(torch.float32)
    monkeypatch.setattr(calibration, "_contour_bands", lambda _: type("Bands", (), {"inner": target.bool(), "outer": ~target.bool()})())
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        calibration._raw_gradient_norms(logits=logits, protocol=Protocol(), targets=target)


def test_collection_authenticates_exact_completed_transaction_steps_and_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: calibration guesses retained checkpoints or uses another run/task."""

    root = tmp_path / "root"
    run = root / "runs/calibration"
    run.mkdir(parents=True)
    plan_path = root / "manifests/calibration_plan.json"
    suite = root / "manifests/development_suite.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(b"plan")
    suite.write_bytes(b"suite")
    expected_config = {"schema": "synthetic-config"}
    plan = {
        "resolved_config": expected_config,
        "development_suite_identity": {
            "path": str(suite), "sha256": _sha(suite), "semantic_sha256": "a" * 64,
            "role": "development",
        },
    }
    monkeypatch.setattr(calibration, "validate_calibration_plan", lambda path: plan)
    monkeypatch.setattr(calibration, "_validate_development_suite_identity", lambda path, identity: None)
    calls: list[tuple[Path, int, object]] = []
    snapshots = []
    for update in UPDATES:
        path = run / f"step-{update}.pt"
        path.write_bytes(str(update).encode())
        identity = type("Identity", (), {
            "completed_updates": update,
            "transaction": f"transactions/step_{update:08d}",
            "transaction_manifest_sha256": f"{update // 100:x}"[-1] * 64,
            "checkpoint_path": f"step-{update}.pt",
            "checkpoint_sha256": _sha(path),
        })()
        snapshots.append(type("Snapshot", (), {"identity": identity, "checkpoint": type("File", (), {"path": str(path), "sha256": _sha(path), "payload": path.read_bytes()})()})())
    def authenticate(path: Path, *, completed_updates: int, expected_config: object) -> object:
        calls.append((path, completed_updates, expected_config))
        return snapshots[UPDATES.index(completed_updates)]
    monkeypatch.setattr(calibration, "authenticate_transaction_checkpoint", authenticate)
    monkeypatch.setattr(
        calibration,
        "_measure_authenticated_checkpoint",
        lambda snapshot, config, development_suite: {
            "g_base": float(snapshot.identity.completed_updates), "g_dic": 2.0,
            "g_dts_004": 3.0, "g_dts_006": 4.0,
        },
    )
    measured = collect_unit_gradient_measurements(run, plan_path, suite)
    assert [item.completed_updates for item in measured] == list(UPDATES)
    assert calls == [(run, update, expected_config) for update in UPDATES]
    assert [item.transaction for item in measured] == [f"transactions/step_{update:08d}" for update in UPDATES]


@pytest.mark.parametrize("attack", ["wrong_step", "duplicate_path", "hash", "transaction"])
def test_collection_rejects_checkpoint_identity_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    """Break caught: authenticated identity is detached from the measured payload."""

    root = tmp_path / "root"
    run = root / "run"
    run.mkdir(parents=True)
    plan_path = root / "plan.json"
    suite = root / "suite.json"
    plan_path.write_bytes(b"plan")
    suite.write_bytes(b"suite")
    plan = {"resolved_config": {}, "development_suite_identity": {"path": str(suite), "sha256": _sha(suite), "semantic_sha256": "a" * 64, "role": "development"}}
    monkeypatch.setattr(calibration, "validate_calibration_plan", lambda path: plan)
    monkeypatch.setattr(calibration, "_validate_development_suite_identity", lambda path, identity: None)
    items = []
    for index, update in enumerate(UPDATES):
        path = run / f"{update}.pt"
        path.write_bytes(bytes((index,)))
        completed = 1199 if attack == "wrong_step" and index == 0 else update
        checkpoint_path = str(UPDATES[0]) + ".pt" if attack == "duplicate_path" and index == 1 else path.name
        digest = "0" * 64 if attack == "hash" and index == 0 else _sha(path)
        transaction = "wrong" if attack == "transaction" and index == 0 else f"transactions/step_{update:08d}"
        identity = type("Identity", (), {"completed_updates": completed, "transaction": transaction, "transaction_manifest_sha256": "b" * 64, "checkpoint_path": checkpoint_path, "checkpoint_sha256": digest})()
        file = type("File", (), {"path": str(path), "sha256": _sha(path), "payload": path.read_bytes()})()
        items.append(type("Snapshot", (), {"identity": identity, "checkpoint": file})())
    monkeypatch.setattr(calibration, "authenticate_transaction_checkpoint", lambda path, *, completed_updates, expected_config: items[UPDATES.index(completed_updates)])
    monkeypatch.setattr(calibration, "_measure_authenticated_checkpoint", lambda *args, **kwargs: {"g_base": 1.0, "g_dic": 2.0, "g_dts_004": 3.0, "g_dts_006": 4.0})
    with pytest.raises(ValueError, match="checkpoint|transaction|update|identity|hash"):
        collect_unit_gradient_measurements(run, plan_path, suite)


def _identity(path: Path, semantic: str | None = None, role: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path.resolve()), "sha256": _sha(path)}
    if semantic is not None:
        result["semantic_sha256"] = semantic
    if role is not None:
        result["role"] = role
    return result


def _fake_resolved(root: Path, development: Path, spec: Path) -> dict[str, object]:
    return {
        "schema": "pcraft_soft_loss_v4_config_v1",
        "run_kind": "calibration_source",
        "run_identity": {
            "stage": "calibration", "task_id": "arcane_casters", "replicate_role": "calibration",
            "replicate_index": 0, "run_id": "calibration/source", "run_dir": str(root / "calibration/source"),
        },
        "output_root_identity": {"path": str(root), "marker_path": str(root / "V4_ROOT.json"), "marker_sha256": _sha(root / "V4_ROOT.json")},
        "suite_manifest": {"path": str(development), "sha256": _sha(development), "semantic_sha256": "d" * 64, "role": "development"},
        "specification": {"path": str(spec), "sha256": _sha(spec)},
        "steps": 2000, "resolution": 128, "export_resolution": 128,
        "profile": {"name": "l12_bal"}, "seed": 17,
        "implementation": {"schema": "pcraft_soft_loss_v4_source_manifest_v1", "algorithm": "sha256", "files_sha256": {"trainer.py": "e" * 64}, "bundle_sha256": "f" * 64},
    }


def test_plan_builder_calls_resolver_once_and_binds_exact_commands_and_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: plan reconstructs config/argv or omits source/suite authority."""

    root = tmp_path / "root"
    root.mkdir()
    (root / "V4_ROOT.json").write_bytes(b"marker")
    development = root / "manifests/development_suite.json"
    formal = root / "manifests/formal_suite.json"
    development.parent.mkdir()
    development.write_bytes(b"development")
    formal.write_bytes(b"formal")
    spec = tmp_path / "spec.md"
    spec.write_bytes(b"spec")
    resolved = _fake_resolved(root, development, spec)
    calls = 0
    class Resolved:
        config = resolved
        config_hash = "a" * 64
    def resolve(args: argparse.Namespace) -> object:
        nonlocal calls
        calls += 1
        return Resolved()
    monkeypatch.setattr(calibration, "resolve_run_configuration", resolve)
    monkeypatch.setattr(calibration, "canonical_trainer_argv", lambda args, resume: ("resume",) if resume else ("fresh",))
    monkeypatch.setattr(calibration, "_load_suites", lambda d, f: (
        type("Suite", (), {"path": str(d), "sha256": _sha(d), "semantic_sha256": "d" * 64, "role": "development", "tasks": (type("Task", (), {"case_id": "arcane_casters"})(),)})(),
        type("Suite", (), {"path": str(f), "sha256": _sha(f), "semantic_sha256": "c" * 64, "role": "formal", "tasks": ()})(),
    ))
    monkeypatch.setattr(calibration, "_source_bundle", lambda names: {"schema": "pcraft_soft_loss_v4_source_bundle_v1", "algorithm": "sha256_canonical_ordered_path_digest_rows_v1", "files_sha256": {name: "b" * 64 for name in names}, "bundle_sha256": "c" * 64})
    payload = build_calibration_plan(output_root=root, trainer_args=argparse.Namespace(), development_suite=development, formal_suite=formal)
    assert calls == 1
    assert payload["resolved_config"] == resolved
    assert payload["fresh_argv"] == ["fresh"]
    assert payload["resume_argv"] == ["resume"]
    assert payload["calibration_task_identity"] == {"task_id": "arcane_casters"}
    assert payload["estimator"] == {
        "name": "median_raw_gradient_norm_ratio_v1", "completed_updates": [1200, 1500, 1800],
        "dic_endpoint": 0.05, "dts_endpoints": [0.04, 0.06], "epsilon": 1e-12,
        "candidate_independent": True,
    }
    assert payload["semantic_sha256"] == canonical_hash({k: v for k, v in payload.items() if k != "semantic_sha256"})


def test_plan_argv_accepts_canonical_locked_symlink_for_resolved_interpreter() -> None:
    """Break caught: lexical locked argv and resolved interpreter identity are compared as strings."""

    args = calibration.build_trainer_parser().parse_args(
        [
            "--run-kind", "calibration_source",
            "--device", "cuda:0",
            "--output-root", "/tmp/v4-root",
            "--run-id", "runs/calibration/source",
            "--stage", "calibration_source128",
            "--task-id", "arcane_casters",
            "--replicate-role", "calibration",
            "--replicate-index", "0",
            "--steps", "2000",
            "--learning-rate", "0.007",
            "--shadow-files", "/tmp/x.png", "/tmp/y.png", "/tmp/z.png",
            "--resolution", "128",
            "--export-resolution", "128",
            "--profile", "l12_bal",
            "--suite-manifest", "/tmp/development_suite.json",
            "--specification", "/tmp/spec.md",
            "--export-final-step",
            "--soft-only-eval",
        ]
    )
    fresh = list(calibration.canonical_trainer_argv(args, resume=False))
    resume = list(calibration.canonical_trainer_argv(args, resume=True))
    assert Path(fresh[0]).resolve(strict=True) != Path(fresh[0])
    config = {
        "run_kind": args.run_kind,
        "device": args.device,
        "output_root_identity": {"path": args.output_root},
        "run_identity": {
            "run_id": args.run_id,
            "stage": args.stage,
            "task_id": args.task_id,
            "replicate_role": args.replicate_role,
            "replicate_index": args.replicate_index,
        },
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "shadow_inputs_xyz": [
            {"path": path} for path in args.shadow_files
        ],
        "resolution": args.resolution,
        "export_resolution": args.export_resolution,
        "profile": {"name": args.profile},
        "suite_manifest": {"path": args.suite_manifest},
        "specification": {"path": args.specification},
        "unit_coefficient_identity": None,
        "candidate_registry_identity": None,
        "budget_identity": None,
        "interpreter": {"path": str(Path(fresh[0]).resolve(strict=True))},
    }
    calibration._validate_argv(fresh, resume, config)


def test_source_bundle_uses_documented_ordered_rows_and_rejects_every_identity_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: source order/schema/digest/bundle hash becomes decorative."""

    names = ("z_source.py", "a_source.py")
    for index, name in enumerate(names):
        (tmp_path / name).write_bytes(f"source-{index}".encode())
    monkeypatch.setattr(calibration, "PROJECT_ROOT", tmp_path)
    bundle = calibration._source_bundle(names)
    rows = [{"path": name, "sha256": _sha(tmp_path / name)} for name in names]
    expected = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert bundle == {
        "schema": "pcraft_soft_loss_v4_source_bundle_v1",
        "algorithm": "sha256_canonical_ordered_path_digest_rows_v1",
        "files_sha256": {name: _sha(tmp_path / name) for name in names},
        "bundle_sha256": expected,
    }
    for mutate in (
        lambda value: value.__setitem__("schema", "forged"),
        lambda value: value.__setitem__("algorithm", "forged"),
        lambda value: value["files_sha256"].__setitem__(names[0], "0" * 64),
        lambda value: value.__setitem__("bundle_sha256", "0" * 64),
    ):
        forged = json.loads(json.dumps(bundle))
        mutate(forged)
        with pytest.raises(ValueError):
            calibration._validate_source_bundle(forged, names, label="test source")


def test_plan_and_stage_are_distinct_exact_schemas_and_semantic_hash_every_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: post-run evidence rewrites the preregistered plan or excludes fields."""

    plan_path = tmp_path / "manifests/calibration_plan.json"
    plan_path.parent.mkdir()
    plan = {
        "schema": "pcraft_soft_loss_v4_calibration_plan_v1", "status": "REGISTERED",
        "output_root_identity": {"path": str(tmp_path), "marker_path": str(tmp_path / "V4_ROOT.json"), "marker_sha256": "a" * 64},
        "resolved_config": {"run_identity": {"task_id": "arcane_casters", "run_id": "calibration/source", "run_dir": str(tmp_path / "calibration/source")}},
        "config_hash": "b" * 64, "fresh_argv": ["fresh"], "resume_argv": ["resume"],
        "calibration_task_identity": {"task_id": "arcane_casters"},
        "development_suite_identity": {"path": str(tmp_path / "manifests/development_suite.json"), "sha256": "c" * 64, "semantic_sha256": "d" * 64, "role": "development"},
        "formal_suite_identity": {"path": str(tmp_path / "manifests/formal_suite.json"), "sha256": "e" * 64, "semantic_sha256": "f" * 64, "role": "formal"},
        "specification_identity": {"path": str(tmp_path / "spec"), "sha256": "1" * 64},
        "estimator": {"name": "median_raw_gradient_norm_ratio_v1", "completed_updates": [1200,1500,1800], "dic_endpoint": 0.05, "dts_endpoints": [0.04,0.06], "epsilon": 1e-12, "candidate_independent": True},
        "binder_source_bundle": {"schema": "x", "algorithm": "y", "files_sha256": {}, "bundle_sha256": "2" * 64},
        "calibrator_source_bundle": {"schema": "x", "algorithm": "y", "files_sha256": {}, "bundle_sha256": "3" * 64},
        "trainer_source_bundle": {"schema": "x", "algorithm": "sha256", "files_sha256": {}, "bundle_sha256": "4" * 64},
        "publication": {"policy": "create_once_before_calibration_process"},
    }
    plan["semantic_sha256"] = canonical_hash(plan)
    plan_path.write_bytes(canonical_json_bytes(plan))
    run = tmp_path / "calibration/source"
    run.mkdir(parents=True)
    authenticated = type("Run", (), {
        "run_dir": str(run), "config_hash": "b" * 64, "status_sha256": "5" * 64,
        "artifact_manifest_sha256": "6" * 64,
        "config": {"run_identity": {"run_id": "calibration/source", "task_id": "arcane_casters"}},
    })()
    monkeypatch.setattr(calibration, "validate_calibration_plan", lambda path: plan)
    monkeypatch.setattr(calibration, "validate_completed_run", lambda path, expected_config=None: authenticated)
    stage = build_calibration_stage(calibration_plan=plan_path, calibration_run=run, measurements=_measurements())
    assert stage["schema"] == "pcraft_soft_loss_v4_calibration_stage_v1"
    assert stage["status"] == "APPROVED_FROZEN"
    assert stage["publication"] == {"policy": "create_once_after_completed_calibration_source"}
    assert [row["completed_updates"] for row in stage["checkpoints"]] == list(UPDATES)
    assert [row["completed_updates"] for row in stage["measurements"]] == list(UPDATES)
    assert stage["semantic_sha256"] == canonical_hash({k: v for k, v in stage.items() if k != "semantic_sha256"})
    assert plan_path != tmp_path / "analysis/calibration_stage.json"


def test_plan_writer_refuses_posthoc_registration_when_run_lexists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: calibration authority is created after observing trajectory bytes."""

    root = tmp_path / "root"
    run = root / "calibration/source"
    run.mkdir(parents=True)
    monkeypatch.setattr(calibration, "build_calibration_plan", lambda **kwargs: {"resolved_config": {"run_identity": {"run_dir": str(run)}}})
    with pytest.raises((FileExistsError, ValueError), match="run|exists|preregister"):
        write_calibration_plan(root, trainer_args=argparse.Namespace(), development_suite=Path("/d"), formal_suite=Path("/f"))


def test_parser_has_only_root_and_plan_without_abbreviation_or_override() -> None:
    """Break caught: CLI gains checkpoint/metric/candidate/coefficient selection."""

    parser = build_calibration_parser()
    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    assert set(actions) == {"output_root", "calibration_plan"}
    assert all(action.required for action in actions.values())
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "/x", "--calibration-plan", "/p"])


def test_candidate_consumer_compatibility_uses_one_coefficient_pair() -> None:
    """Break caught: each candidate silently recalibrates its own coefficients."""

    coefficients = derive_unit_coefficients(_measurements())
    ratios = ((.04,.04),(.08,.08),(.12,.12),(.12,.08),(.08,.12))
    configs = [V4StepConfig(coefficients["dic"], coefficients["dts"], dic, dts) for dic, dts in ratios]
    assert {item.dic_unit_coefficient for item in configs} == {coefficients["dic"]}
    assert {item.dts_unit_coefficient for item in configs} == {coefficients["dts"]}


def test_unit_and_registry_validation_perform_no_renderer_backward_or_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: downstream validation replays nondeterministic GPU backward."""

    forbidden = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("forbidden replay"))
    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    path = tmp_path / "unit.json"
    monkeypatch.setattr(calibration, "_validate_unit_payload", lambda payload, path: payload)
    payload = {"schema": "synthetic"}
    path.write_bytes(canonical_json_bytes(payload))
    assert validate_unit_coefficients(path) == payload


def test_calibration_import_launcher_and_ast_scope_are_side_effect_free(tmp_path: Path) -> None:
    """Break caught: calibration imports training or scans quality/data outputs."""

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(PROJECT_ROOT))
    before = set(tmp_path.rglob("*"))
    subprocess.run([sys.executable, "-c", "import shadowforge_pipeline.soft_voxel_v4.calibration; import experiments.calibrate_soft_loss_v4"], cwd=tmp_path, env=env, check=True)
    assert set(tmp_path.rglob("*")) == before
    for relative in ("shadowforge_pipeline/soft_voxel_v4/calibration.py", "experiments/calibrate_soft_loss_v4.py"):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        assert not any(name.endswith(("soft_voxel_v3.calibration", "soft_voxel_v3.trainer", "run_soft_loss_v4")) for name in imports)
        assert "/data" not in source
        assert ".glob(" not in source and ".rglob(" not in source


def test_calibration_launcher_help_is_thin() -> None:
    """Break caught: launcher implements a second mutable scientific parser."""

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "experiments/calibrate_soft_loss_v4.py"), "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "--output-root" in result.stdout and "--calibration-plan" in result.stdout
