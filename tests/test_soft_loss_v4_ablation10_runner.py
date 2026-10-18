"""Contracts for the frozen Soft Loss v4 ten-task ablation runner."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import experiments.run_soft_loss_v4_ablation10 as runner
from experiments.soft_loss_v4_registry import V4BoundInput, V4SuiteIdentity, V4TaskInputs
from shadowforge_pipeline.soft_voxel_v4.config import (
    bootstrap_v4_root,
    canonical_trainer_argv,
)
from shadowforge_pipeline.soft_voxel_v4.losses import registered_profiles, resolve_profile
from shadowforge_pipeline.soft_voxel_v4.provenance import canonical_hash, canonical_json_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL = tuple(f"connected_{index:02d}" for index in range(1, 11))
PROFILES = ("l12_bal", "dic_only", "dts_only", "v4_full")
DECISION_POLICY = {
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
        "mean_delta_iou_min": 0.010,
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
}


def _assert_exact_builtin_types(value: object) -> None:
    if isinstance(value, dict):
        assert type(value) is dict
        for key, nested in value.items():
            assert type(key) is str
            _assert_exact_builtin_types(nested)
    elif isinstance(value, list):
        assert type(value) is list
        for nested in value:
            _assert_exact_builtin_types(nested)
    else:
        assert type(value) in (str, bool, int, float)


def _suite(
    tmp_path: Path, *, task_ids: tuple[str, ...] = FORMAL, role: str = "formal"
) -> V4SuiteIdentity:
    tasks = []
    for task_index, task_id in enumerate(task_ids):
        inputs = []
        for axis_index, axis in enumerate("xyz"):
            path = tmp_path / "inputs" / task_id / f"{axis}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes((task_index, axis_index)))
            inputs.append(
                V4BoundInput(
                    axis=axis,
                    source_path=str(path),
                    resolved_path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        tasks.append(V4TaskInputs(task_id, tuple(inputs), role))
    return V4SuiteIdentity(
        path=str(tmp_path / "manifests" / f"{role}_suite.json"),
        sha256="a" * 64,
        semantic_sha256="b" * 64,
        role=role,
        tasks=tuple(tasks),
    )


def _fake_resolver(
    calls: list[argparse.Namespace], *, state: dict[str, object] | None = None
):
    state = {} if state is None else state

    def resolve(args: argparse.Namespace) -> SimpleNamespace:
        calls.append(args)
        assert set(vars(args)) == {
            "run_kind", "device", "output_root", "run_id", "stage", "task_id",
            "replicate_role", "replicate_index", "steps", "learning_rate",
            "shadow_files", "resolution", "export_resolution", "profile",
            "suite_manifest", "specification", "unit_coefficients",
            "candidate_registry", "candidate_id", "resume", "export_final_step",
            "soft_only_eval",
        }
        auxiliary = args.profile != "l12_bal"
        profile = resolve_profile(args.profile)
        candidates = {
            "r04_r04": {"candidate_id": "r04_r04", "order_index": 0, "dic_max_ratio": 0.04, "dts_max_ratio": 0.04},
            "r08_r08": {"candidate_id": "r08_r08", "order_index": 1, "dic_max_ratio": 0.08, "dts_max_ratio": 0.08},
            "r12_r12": {"candidate_id": "r12_r12", "order_index": 2, "dic_max_ratio": 0.12, "dts_max_ratio": 0.12},
        }
        candidate = candidates[args.candidate_id] if auxiliary else None
        suite_role = (
            "development"
            if Path(args.suite_manifest).name == "development_suite.json"
            else "formal"
        )
        config: dict[str, object] = {
            "run_kind": args.run_kind,
            "run_identity": {
                "stage": args.stage,
                "task_id": args.task_id,
                "replicate_role": args.replicate_role,
                "replicate_index": args.replicate_index,
                "run_id": args.run_id,
                "run_dir": str(Path(args.output_root) / args.run_id),
            },
            "output_root_identity": {
                "path": args.output_root,
                "marker_path": str(Path(args.output_root) / "V4_ROOT.json"),
                "marker_sha256": "1" * 64,
            },
            "device": args.device,
            "suite_manifest": {"path": args.suite_manifest, "sha256": "a" * 64, "semantic_sha256": "b" * 64, "role": suite_role},
            "specification": {"path": args.specification, "sha256": "c" * 64},
            "shadow_inputs_xyz": [
                {"axis": axis, "path": path, "sha256": f"{index + 2:x}" * 64}
                for index, (axis, path) in enumerate(zip("xyz", args.shadow_files))
            ],
            "unit_coefficient_identity": ({"path": args.unit_coefficients, "sha256": "d" * 64, "semantic_sha256": "e" * 64} if auxiliary else None),
            "candidate_registry_identity": ({"path": args.candidate_registry, "sha256": "f" * 64, "semantic_sha256": "0" * 64} if auxiliary else None),
            "step_config": ({"dic_unit_coefficient": 1.0, "dts_unit_coefficient": 1.0, "dic_max_ratio": candidate["dic_max_ratio"], "dts_max_ratio": candidate["dts_max_ratio"]} if candidate is not None else None),
            "budget_identity": dict(candidate) if candidate is not None else None,
            "interpreter": {"path": "/locked/python", "sha256": "7" * 64},
            "implementation": {"files_sha256": {"task3.py": "8" * 64}, "bundle_sha256": "9" * 64},
            "determinism": {"environment": {"PYTHONPATH": os.environ.get("PYTHONPATH")}},
            "profile": {
                "name": profile.name,
                "l1_weight": profile.l1_weight,
                "l2_weight": profile.l2_weight,
                "use_dic": profile.use_dic,
                "use_dts": profile.use_dts,
            },
            "objective": {
                "density_parameterization": "sigmoid(logits)",
                "view_reduction": "mean_over_six_views",
                "formula": "l2_weight*L2 + l1_weight*L1",
                "l1_weight": profile.l1_weight,
                "l2_weight": profile.l2_weight,
            },
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "resolution": args.resolution,
            "export_resolution": args.export_resolution,
            "optimizer": {"name": "Adam", "initialization_logit": -4.0, "gradient_clip_norm": 1.0},
            "seed": 17,
            "checkpoint_policy": (
                {"name": "convergence", "commit_interval": 100, "retained_updates": [1600, 1800]}
                if args.steps == 2000
                else {"name": "final_only", "commit_interval": 100, "retained_updates": []}
            ),
            "soft_only": args.soft_only_eval,
            "final_step_export_only": args.export_final_step,
        }
        if state.get("dependency_drift") is not None:
            config["dependency_probe"] = state["dependency_drift"]
        return SimpleNamespace(config=config, config_hash=canonical_hash(config))

    return resolve


def _build_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    state: dict[str, object] | None = None,
) -> tuple[runner.AblationRunSpec, ...]:
    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    suite = _suite(tmp_path)
    calls: list[argparse.Namespace] = []
    monkeypatch.setattr(runner, "load_v4_suite_identity", lambda *_: suite)
    monkeypatch.setattr(runner, "resolve_run_configuration", _fake_resolver(calls, state=state))
    specs = runner.build_ablation_specs(
        output_root=root,
        formal_suite=Path(suite.path),
        specification=tmp_path / "spec.md",
        unit_coefficients=root / "analysis" / "unit_coefficients.json",
        candidate_registry=root / "manifests" / "candidate_registry.json",
        device="cpu",
    )
    assert len(calls) >= 40
    return specs


def _coherently_rebuilt_spec(
    spec: runner.AblationRunSpec,
    *,
    namespace_updates: dict[str, object] | None = None,
    config_mutator: object | None = None,
) -> runner.AblationRunSpec:
    """Rebuild one attack spec so its namespace, config hash, and argv agree."""

    namespace = argparse.Namespace(**(vars(spec.namespace) | (namespace_updates or {})))
    resolved = runner._resolve_task3(namespace)
    config = json.loads(canonical_json_bytes(resolved.config))
    if config_mutator is not None:
        config_mutator(config)
    return replace(
        spec,
        namespace=namespace,
        run_dir=Path(config["run_identity"]["run_dir"]),
        resolved_config=config,
        config_hash=canonical_hash(config),
        fresh_argv=canonical_trainer_argv(namespace, resume=False),
        resume_argv=canonical_trainer_argv(namespace, resume=True),
    )


def _real_formal_baseline_fixture(tmp_path: Path) -> tuple[Path, Path, tuple[str, str, str]]:
    """Build a byte-only, formal Task 3 fixture without images, training, or /data."""

    root = (tmp_path / "real-campaign").resolve()
    bootstrap_v4_root(root)
    source = (tmp_path / "registered-source-suite.json").resolve()
    source.write_bytes(b"registered formal suite source\n")
    tasks: list[dict[str, object]] = []
    first_paths: tuple[str, str, str] | None = None
    for task_index, task_id in enumerate(FORMAL):
        inputs: list[dict[str, str]] = []
        for axis_index, axis in enumerate("xyz"):
            path = (root / "inputs" / task_id / f"{axis}.bin").resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes((task_index, axis_index, 17)))
            inputs.append(
                {
                    "axis": axis,
                    "source_path": str(path),
                    "resolved_path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        if first_paths is None:
            first_paths = tuple(item["resolved_path"] for item in inputs)  # type: ignore[assignment]
        tasks.append({"case_id": task_id, "inputs": inputs})
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v4_suite_v1",
        "status": "REGISTERED",
        "role": "formal",
        "source_suite": {
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "semantic_sha256": "1" * 64,
        },
        "tasks": tasks,
        "task_count": 10,
        "input_count": 30,
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
        "profiles": list(registered_profiles()),
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    suite = root / "manifests" / "formal_suite.json"
    suite.parent.mkdir(exist_ok=True)
    suite.write_bytes(canonical_json_bytes(payload) + b"\n")
    specification = (tmp_path / "v4-specification.md").resolve()
    specification.write_bytes(
        (PROJECT_ROOT / ".superpowers/sdd/2026-08-27-soft-loss-v4-implementation/spec.md").read_bytes()
    )
    assert first_paths is not None
    return root, specification, first_paths


def test_builds_the_frozen_ordered_matrix_and_exact_task3_namespaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a swapped, duplicate, or scientifically different cell is launched."""

    specs = _build_specs(tmp_path, monkeypatch)
    assert [(item.task_id, item.profile) for item in specs] == [
        (task_id, profile) for task_id in FORMAL for profile in PROFILES
    ]
    assert all(item.stage == "ablation10_128" for item in specs)
    assert all(item.resolved_config["steps"] == 2000 for item in specs)
    assert all(item.resolved_config["learning_rate"] == 0.007 for item in specs)
    assert all(item.resolved_config["resolution"] == 128 for item in specs)
    assert all(item.resolved_config["export_resolution"] == 128 for item in specs)
    for spec in specs:
        assert spec.run_id == f"runs/ablation10/{spec.task_id}/{spec.profile}"
        assert spec.fresh_argv == canonical_trainer_argv(spec.namespace, resume=False)
        assert spec.resume_argv == canonical_trainer_argv(spec.namespace, resume=True)
        if spec.profile == "l12_bal":
            assert spec.namespace.unit_coefficients is None
            assert spec.namespace.candidate_registry is None
            assert spec.namespace.candidate_id is None
            assert spec.resolved_config["unit_coefficient_identity"] is None
            assert spec.resolved_config["candidate_registry_identity"] is None
            assert spec.resolved_config["budget_identity"] is None
        else:
            assert spec.namespace.candidate_id == "r08_r08"
            assert spec.resolved_config["budget_identity"]["candidate_id"] == "r08_r08"


def test_build_rejects_malformed_or_duplicate_formal_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a malformed suite silently changes the preregistered ten tasks."""

    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    monkeypatch.setattr(runner, "resolve_run_configuration", _fake_resolver([]))
    for task_ids in (("connected_01",) * 10, FORMAL[:-1]):
        monkeypatch.setattr(runner, "load_v4_suite_identity", lambda *_: _suite(tmp_path / str(len(task_ids)), task_ids=task_ids))
        with pytest.raises(ValueError, match="formal suite.*order|formal suite"):
            runner.build_ablation_specs(
                output_root=root, formal_suite=root / "manifests/formal_suite.json",
                specification=tmp_path / "spec.md", unit_coefficients=root / "analysis/unit_coefficients.json",
                candidate_registry=root / "manifests/candidate_registry.json", device="cpu",
            )


def test_manifest_is_canonical_create_once_and_binds_all_resolved_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a changed matrix/config overwrites an existing preregistration."""

    specs = _build_specs(tmp_path, monkeypatch)
    manifest = runner.write_ablation_manifest(tmp_path / "campaign", specs)
    first = manifest.read_bytes()
    payload = json.loads(first)
    semantic = dict(payload)
    assert canonical_hash(semantic | {"semantic_sha256": None}) != payload["semantic_sha256"]
    del semantic["semantic_sha256"]
    assert canonical_hash(semantic) == payload["semantic_sha256"]
    assert payload["constants"] == {
        "seed": 17, "optimizer": "Adam", "learning_rate": 0.007,
        "resolution": 128, "export_resolution": 128, "steps": 2000,
        "stage": "ablation10_128", "replicate_role": "primary", "replicate_index": 0,
    }
    assert len(payload["matrix"]) == 40
    assert payload["matrix"][0]["resolved_config"] == specs[0].resolved_config
    assert manifest == tmp_path / "campaign/manifests/ablation10.json"
    assert runner.write_ablation_manifest(tmp_path / "campaign", specs).read_bytes() == first
    manifest.write_bytes(b'{"foreign":true}\n')
    with pytest.raises(FileExistsError, match="immutable JSON differs"):
        runner.write_ablation_manifest(tmp_path / "campaign", specs)


def test_manifest_rejects_dependency_or_config_drift_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: inconsistent suite/spec/unit identities enter a single manifest."""

    specs = _build_specs(tmp_path, monkeypatch)
    broken = dict(specs[1].resolved_config)
    broken["suite_manifest"] = dict(broken["suite_manifest"], sha256="9" * 64)
    with pytest.raises(ValueError, match="suite|identity|configuration"):
        runner.write_ablation_manifest(tmp_path / "campaign", (specs[0], replace(specs[1], resolved_config=broken), *specs[2:]))


def test_manifest_rejects_a_resolved_run_directory_that_disagrees_with_its_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a manifest binds a config whose Task 3 run directory names another cell."""

    specs = _build_specs(tmp_path, monkeypatch)
    broken = dict(specs[0].resolved_config)
    broken["run_identity"] = dict(broken["run_identity"], run_dir=str(tmp_path / "elsewhere"))
    changed = replace(specs[0], resolved_config=broken, config_hash=canonical_hash(broken))
    with pytest.raises(ValueError, match="run directory|run ID|run identity"):
        runner.write_ablation_manifest(tmp_path / "campaign", (changed, *specs[1:]))


@pytest.mark.parametrize(
    ("label", "cell_index", "namespace_updates"),
    (
        ("steps_1000", 0, {"steps": 1000}),
        ("resolution_256", 0, {"resolution": 256, "export_resolution": 256}),
        ("learning_rate", 0, {"learning_rate": 0.001}),
        ("run_kind", 0, {"run_kind": "calibration_source"}),
        ("stage", 0, {"stage": "other_stage"}),
        ("task_id", 0, {"task_id": "connected_02"}),
        ("replicate_role_and_index", 0, {"replicate_role": "technical_repeat", "replicate_index": 1}),
        ("profile_row_correspondence", 1, {"profile": "dts_only"}),
        ("device", 1, {"device": "cuda:0"}),
        ("scope_acknowledgements", 0, {"export_final_step": False, "soft_only_eval": False}),
    ),
)
def test_validate_specs_rejects_coherently_rebuilt_frozen_namespace_and_config_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    cell_index: int,
    namespace_updates: dict[str, object],
) -> None:
    """Break caught: a self-consistent Task 3 cell changes a frozen ablation science field."""

    specs = _build_specs(tmp_path, monkeypatch)
    changed = list(specs)
    changed[cell_index] = _coherently_rebuilt_spec(
        specs[cell_index], namespace_updates=namespace_updates
    )
    assert changed[cell_index].fresh_argv == canonical_trainer_argv(
        changed[cell_index].namespace, resume=False
    )
    assert changed[cell_index].resume_argv == canonical_trainer_argv(
        changed[cell_index].namespace, resume=True
    )
    with pytest.raises(ValueError, match="frozen|namespace|configuration|ablation10|matrix"):
        runner._validate_specs(tuple(changed))


@pytest.mark.parametrize(
    ("label", "config_mutator"),
    (
        ("root_marker_identity", lambda config: config["output_root_identity"].__setitem__("marker_sha256", "9" * 64)),
        ("seed", lambda config: config.__setitem__("seed", 18)),
        ("optimizer", lambda config: config["optimizer"].__setitem__("gradient_clip_norm", 0.5)),
        ("profile_semantics", lambda config: config["profile"].__setitem__("l1_weight", 0.0)),
        ("final_step_scope", lambda config: config.__setitem__("final_step_export_only", False)),
    ),
)
def test_validate_specs_rejects_forged_resolved_config_identities_and_constants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    config_mutator: object,
) -> None:
    """Break caught: config bytes/hash are forged coherently but differ from Task 3's frozen result."""

    specs = _build_specs(tmp_path, monkeypatch)
    changed = list(specs)
    changed[0] = _coherently_rebuilt_spec(specs[0], config_mutator=config_mutator)
    with pytest.raises(ValueError, match="frozen|namespace|configuration|ablation10|config"):
        runner._validate_specs(tuple(changed))


def test_validate_specs_rejects_all_auxiliary_r08_ratio_forgery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: all auxiliary rows self-consistently change r08 ratios while retaining the r08 label."""

    specs = _build_specs(tmp_path, monkeypatch)

    def alter_ratios(config: dict[str, object]) -> None:
        config["budget_identity"] = {
            "candidate_id": "r08_r08",
            "order_index": 1,
            "dic_max_ratio": 0.04,
            "dts_max_ratio": 0.04,
        }
        config["step_config"] = {
            "dic_unit_coefficient": 1.0,
            "dts_unit_coefficient": 1.0,
            "dic_max_ratio": 0.04,
            "dts_max_ratio": 0.04,
        }

    changed = tuple(
        _coherently_rebuilt_spec(spec, config_mutator=alter_ratios)
        if spec.profile != "l12_bal"
        else spec
        for spec in specs
    )
    with pytest.raises(ValueError, match="frozen|configuration|ablation10|candidate"):
        runner._validate_specs(changed)


def test_validate_specs_rejects_non_r08_candidate_id_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: an auxiliary cell changes the candidate identity away from the preregistered r08 row."""

    specs = _build_specs(tmp_path, monkeypatch)
    changed = list(specs)
    changed[1] = _coherently_rebuilt_spec(
        specs[1], namespace_updates={"candidate_id": "r04_r04"}
    )
    with pytest.raises(ValueError, match="candidate|r08|ablation10"):
        runner._validate_specs(tuple(changed))


def test_write_and_run_reject_a_coherent_1000_step_spec_before_manifest_or_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a 1000-step Task 3 spec registers a manifest that falsely claims 2000 constants."""

    specs = _build_specs(tmp_path, monkeypatch)
    changed = (_coherently_rebuilt_spec(specs[0], namespace_updates={"steps": 1000}), *specs[1:])
    with pytest.raises(ValueError, match="frozen|namespace|configuration|ablation10|steps"):
        runner.write_ablation_manifest(tmp_path / "campaign", changed)
    assert not (tmp_path / "campaign/manifests/ablation10.json").exists()
    launched: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda argv, **_kwargs: launched.append(argv) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(runner, "_authenticate_completed", lambda _spec: object())
    with pytest.raises(ValueError, match="frozen|namespace|configuration|ablation10|steps"):
        runner.run_specs(changed)
    assert launched == []
    assert not (tmp_path / "campaign/logs").exists()


def test_write_rejects_a_coherent_development_suite_role_for_every_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: all rows replace the formal suite identity with a development suite before publication."""

    formal = _suite(tmp_path)
    development = _suite(tmp_path / "development", role="development")
    specs = _build_specs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "load_v4_suite_identity",
        lambda path, _role: development if Path(path) == Path(development.path) else formal,
    )
    changed = tuple(
        _coherently_rebuilt_spec(
            spec, namespace_updates={"suite_manifest": development.path}
        )
        for spec in specs
    )
    with pytest.raises(ValueError, match="formal suite|formal|frozen|ablation10"):
        runner.write_ablation_manifest(tmp_path / "campaign", changed)


def test_dry_run_only_publishes_manifest_and_reports_all_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Break caught: dry-run creates runs, logs, or invokes a trainer child."""

    specs = _build_specs(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "build_ablation_specs", lambda **_: specs)
    calls: list[object] = []
    monkeypatch.setattr(runner, "run_specs", lambda *args, **kwargs: calls.append((args, kwargs)))
    result = runner.main([
        "--output-root", str(tmp_path / "campaign"), "--formal-suite", str(tmp_path / "campaign/manifests/formal_suite.json"),
        "--specification", str(tmp_path / "spec.md"), "--unit-coefficients", str(tmp_path / "campaign/analysis/unit_coefficients.json"),
        "--candidate-registry", str(tmp_path / "campaign/manifests/candidate_registry.json"), "--device", "cpu", "--dry-run",
    ])
    assert result == 0
    assert calls == []
    assert not (tmp_path / "campaign/runs").exists()
    assert not (tmp_path / "campaign/logs").exists()
    assert "40 dry cells" in capsys.readouterr().out


def _single_spec(tmp_path: Path) -> runner.AblationRunSpec:
    root = tmp_path / "campaign"
    root.mkdir()
    namespace = argparse.Namespace(
        run_kind="experiment", device="cpu", output_root=str(root), run_id="runs/ablation10/connected_01/l12_bal",
        stage="ablation10_128", task_id="connected_01", replicate_role="primary", replicate_index=0,
        steps=2000, learning_rate=0.007, shadow_files=["/x", "/y", "/z"], resolution=128,
        export_resolution=128, profile="l12_bal", suite_manifest="/suite", specification="/spec",
        unit_coefficients=None, candidate_registry=None, candidate_id=None, resume=False,
        export_final_step=True, soft_only_eval=True,
    )
    config = {"run_identity": {"run_dir": str(root / namespace.run_id)}}
    return runner.AblationRunSpec(
        task_id="connected_01", profile="l12_bal", stage="ablation10_128", run_id=namespace.run_id,
        run_dir=root / namespace.run_id, namespace=namespace, resolved_config=config,
        config_hash=canonical_hash(config), fresh_argv=("locked", "fresh"), resume_argv=("locked", "resume"),
    )


def test_run_specs_fresh_launch_logs_then_authenticates_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a fresh child runs without controlled launch/log/completion checks."""

    specs = _build_specs(tmp_path, monkeypatch)
    launched: list[object] = []
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **kwargs: launched.append((argv, kwargs)) or SimpleNamespace(returncode=0))
    authenticated: list[object] = []
    monkeypatch.setattr(runner, "_authenticate_completed", lambda item: authenticated.append(item))
    outcome = runner.run_specs(specs, resume=False)
    assert outcome == {"launched": 40, "skipped": 0, "resumed": 0}
    assert len(launched) == 40
    assert launched[0][0] == specs[0].fresh_argv
    assert launched[0][1]["cwd"] == PROJECT_ROOT
    assert launched[0][1]["env"]["PYTHONPATH"] == str(PROJECT_ROOT)
    assert authenticated == list(specs)
    stdout = list((tmp_path / "campaign/logs/ablation10/connected_01/l12_bal").glob("*.stdout.log"))
    stderr = list((tmp_path / "campaign/logs/ablation10/connected_01/l12_bal").glob("*.stderr.log"))
    assert len(stdout) == len(stderr) == 1


def test_run_specs_skips_only_authenticated_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: an existing but unauthenticated run is treated as complete."""

    specs = _build_specs(tmp_path, monkeypatch)
    for spec in specs:
        spec.run_dir.mkdir(parents=True)
    authenticated: list[tuple[Path, dict[str, object]]] = []
    monkeypatch.setattr(
        runner,
        "validate_completed_run",
        lambda path, *, expected_config: authenticated.append((path, expected_config)) or object(),
    )
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must skip"))
    assert runner.run_specs(specs, resume=False) == {"launched": 0, "skipped": 40, "resumed": 0}
    assert authenticated == [(spec.run_dir, spec.resolved_config) for spec in specs]


def test_resume_requires_task3_authenticated_compatible_running_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: --resume launches a fresh, corrupt, or incompatible directory."""

    specs = _build_specs(tmp_path, monkeypatch)
    for spec in specs:
        spec.run_dir.mkdir(parents=True)
    seen: list[object] = []
    monkeypatch.setattr(runner, "validate_resumable_run", lambda path, *, expected_config: seen.append((path, expected_config)) or object())
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    authentication_calls: dict[str, int] = {}

    def authenticate(item: object) -> None:
        spec = item
        authentication_calls[spec.run_id] = authentication_calls.get(spec.run_id, 0) + 1
        if authentication_calls[spec.run_id] == 1:
            raise ValueError("running")

    monkeypatch.setattr(runner, "_authenticate_completed", authenticate)
    assert runner.run_specs(specs, resume=True) == {"launched": 40, "skipped": 0, "resumed": 40}
    assert seen == [(spec.run_dir, spec.resolved_config) for spec in specs]
    monkeypatch.setattr(runner, "validate_resumable_run", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("corrupt")))
    monkeypatch.setattr(runner, "_authenticate_completed", lambda *_: (_ for _ in ()).throw(ValueError("running")))
    with pytest.raises(ValueError, match="corrupt"):
        runner.run_specs(specs, resume=True)


def test_resume_finishes_a_mixed_completed_running_and_missing_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: --resume stops at the first missing cell instead of completing the interrupted matrix."""

    specs = _build_specs(tmp_path, monkeypatch)
    specs[0].run_dir.mkdir(parents=True)
    specs[1].run_dir.mkdir(parents=True)
    authenticated: dict[str, int] = {}

    def complete(spec: runner.AblationRunSpec) -> object:
        authenticated[spec.run_id] = authenticated.get(spec.run_id, 0) + 1
        if spec is specs[1] and authenticated[spec.run_id] == 1:
            raise ValueError("running")
        return object()

    resumable: list[tuple[Path, dict[str, object]]] = []
    launched: list[tuple[str, ...]] = []
    monkeypatch.setattr(runner, "_authenticate_completed", complete)
    monkeypatch.setattr(
        runner,
        "validate_resumable_run",
        lambda path, *, expected_config: resumable.append((path, expected_config)) or object(),
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda argv, **_kwargs: launched.append(argv) or SimpleNamespace(returncode=0),
    )

    assert runner.run_specs(specs, resume=True) == {
        "launched": 39,
        "skipped": 1,
        "resumed": 1,
    }
    assert resumable == [(specs[1].run_dir, specs[1].resolved_config)]
    assert launched[0] == specs[1].resume_argv
    assert launched[1:] == [spec.fresh_argv for spec in specs[2:]]


def test_resume_launches_all_missing_cells_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: --resume incorrectly requires every cell directory to preexist."""

    specs = _build_specs(tmp_path, monkeypatch)
    launched: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda argv, **_kwargs: launched.append(argv) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(runner, "_authenticate_completed", lambda _spec: object())
    assert runner.run_specs(specs, resume=True) == {
        "launched": 40,
        "skipped": 0,
        "resumed": 0,
    }
    assert launched == [spec.fresh_argv for spec in specs]


def test_subprocess_failure_aborts_without_advancing_to_later_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a failed trainer child permits the runner to continue the matrix."""

    specs = _build_specs(tmp_path, monkeypatch)
    calls: list[object] = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: calls.append(1) or SimpleNamespace(returncode=9))
    with pytest.raises(RuntimeError, match="exit code 9"):
        runner.run_specs(specs, resume=False)
    assert calls == [1]


def test_import_has_no_filesystem_or_training_side_effect(tmp_path: Path) -> None:
    """Break caught: importing the runner creates a campaign artifact or starts work."""

    before = sorted(tmp_path.rglob("*"))
    completed = subprocess.run(
        [sys.executable, "-c", "import experiments.run_soft_loss_v4_ablation10"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert sorted(tmp_path.rglob("*")) == before


def test_manifest_preregisters_the_exact_scientific_decision_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the analyzer receives mutable orchestration flags instead of scientific rules."""

    specs = _build_specs(tmp_path, monkeypatch)
    manifest = runner.write_ablation_manifest(tmp_path / "campaign", specs)
    policy = json.loads(manifest.read_text(encoding="utf-8"))["decision_policy"]
    assert policy == DECISION_POLICY
    _assert_exact_builtin_types(policy)


def test_build_uses_the_same_controlled_pythonpath_as_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: parent resolution records a different environment than the locked child."""

    monkeypatch.delenv("PYTHONPATH", raising=False)
    specs = _build_specs(tmp_path, monkeypatch)
    assert "PYTHONPATH" not in os.environ
    assert {
        item.resolved_config["determinism"]["environment"]["PYTHONPATH"]
        for item in specs
    } == {str(PROJECT_ROOT)}
    assert runner._child_environment()["PYTHONPATH"] == str(PROJECT_ROOT)


def test_real_task3_resolution_without_parent_pythonpath_matches_the_locked_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the registered real Task 3 config/hash differs from its locked child environment."""

    root, specification, shadows = _real_formal_baseline_fixture(tmp_path)
    namespace = runner._task3_namespace(
        output_root=root,
        formal_suite=root / "manifests" / "formal_suite.json",
        specification=specification,
        unit_coefficients=root / "analysis" / "unit_coefficients.json",
        candidate_registry=root / "manifests" / "candidate_registry.json",
        device="cpu",
        task_id="connected_01",
        profile="l12_bal",
        shadow_files=shadows,
    )
    monkeypatch.delenv("PYTHONPATH", raising=False)
    parent = runner._resolve_task3(namespace)
    assert "PYTHONPATH" not in os.environ
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys; "
                "from shadowforge_pipeline.soft_voxel_v4.config import build_parser,resolve_run_configuration; "
                "resolved=resolve_run_configuration(build_parser().parse_args(json.loads(sys.argv[1]))); "
                "print(json.dumps({'config':resolved.config,'config_hash':resolved.config_hash},sort_keys=True))"
            ),
            json.dumps(list(canonical_trainer_argv(namespace, resume=False)[3:])),
        ],
        cwd=PROJECT_ROOT,
        env=runner._child_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    payload = json.loads(child.stdout)
    assert payload["config"] == parent.config
    assert payload["config_hash"] == parent.config_hash


@pytest.mark.parametrize(
    "dependency",
    ("suite", "shadow", "specification", "unit", "registry", "interpreter", "task3_source"),
)
def test_post_manifest_dependency_drift_aborts_before_any_log_or_trainer_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dependency: str
) -> None:
    """Break caught: frozen Task 3 inputs change after preregistration but a child still starts."""

    state: dict[str, object] = {}
    specs = _build_specs(tmp_path, monkeypatch, state=state)
    runner.write_ablation_manifest(tmp_path / "campaign", specs)
    state["dependency_drift"] = dependency
    launched: list[object] = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: launched.append((args, kwargs)) or SimpleNamespace(returncode=0),
    )
    with pytest.raises(ValueError, match="drift|configuration|dependency|manifest"):
        runner.run_specs(specs)
    assert launched == []
    assert not (tmp_path / "campaign/logs").exists()
    assert not (tmp_path / "campaign/runs").exists()


def test_dependency_drift_between_cells_stops_before_the_next_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a later cell starts after a prior child changes a bound dependency."""

    state: dict[str, object] = {}
    specs = _build_specs(tmp_path, monkeypatch, state=state)
    runner.write_ablation_manifest(tmp_path / "campaign", specs)
    launched: list[tuple[str, ...]] = []

    def child(argv: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        launched.append(argv)
        state["dependency_drift"] = "shadow"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", child)
    monkeypatch.setattr(runner, "_authenticate_completed", lambda _: object())
    with pytest.raises(ValueError, match="drift|configuration|dependency"):
        runner.run_specs(specs)
    assert launched == [specs[0].fresh_argv]


def test_dependency_drift_after_a_completed_skip_is_revalidated_before_next_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: skipping a completed cell bypasses the post-cell dependency fence."""

    state: dict[str, object] = {}
    specs = _build_specs(tmp_path, monkeypatch, state=state)
    runner.write_ablation_manifest(tmp_path / "campaign", specs)
    for spec in specs:
        spec.run_dir.mkdir(parents=True)
    seen: list[tuple[Path, dict[str, object]]] = []

    def completed(path: Path, *, expected_config: dict[str, object]) -> object:
        seen.append((path, expected_config))
        if len(seen) == 1:
            state["dependency_drift"] = "registry"
        return object()

    monkeypatch.setattr(runner, "validate_completed_run", completed)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="drift|configuration|dependency"):
        runner.run_specs(specs)
    assert seen == [(specs[0].run_dir, specs[0].resolved_config)]


def test_final_cell_is_revalidated_after_its_child_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the final cell has no following launch, so post-child drift goes unchecked."""

    state: dict[str, object] = {}
    specs = _build_specs(tmp_path, monkeypatch, state=state)
    runner.write_ablation_manifest(tmp_path / "campaign", specs)
    launched = 0

    def child(_argv: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        nonlocal launched
        launched += 1
        if launched == len(specs):
            state["dependency_drift"] = "task3_source"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", child)
    monkeypatch.setattr(runner, "_authenticate_completed", lambda _: object())
    with pytest.raises(ValueError, match="drift|configuration|dependency"):
        runner.run_specs(specs)
    assert launched == len(specs)


def test_run_specs_rejects_subsets_and_mutated_specs_before_any_log_or_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: callers bypass the registered 40-cell manifest with a subset or forged argv."""

    specs = _build_specs(tmp_path, monkeypatch)
    launched: list[object] = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: launched.append((args, kwargs)))
    for invalid in (specs[:-1], (replace(specs[0], fresh_argv=("forged",)), *specs[1:])):
        with pytest.raises(ValueError, match="matrix|argv|canonical|spec"):
            runner.run_specs(invalid)
    assert launched == []
    assert not (tmp_path / "campaign/logs").exists()


def test_manifest_root_must_equal_each_spec_root_and_never_create_a_foreign_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a manifest is published under a root other than the resolved cell roots."""

    specs = _build_specs(tmp_path, monkeypatch)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    with pytest.raises(ValueError, match="root|output"):
        runner.write_ablation_manifest(foreign, specs)
    assert not (foreign / "manifests/ablation10.json").exists()


@pytest.mark.parametrize("linked", ("logs", "runs", "runs/ablation10"))
def test_output_run_and_log_symlink_components_reject_before_writes_or_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked: str
) -> None:
    """Break caught: a symlinked output hierarchy redirects logs or run lifecycle files."""

    specs = _build_specs(tmp_path, monkeypatch)
    root = tmp_path / "campaign"
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / linked
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runner, "_authenticate_completed", lambda _: (_ for _ in ()).throw(ValueError("not completed")))
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not spawn"))
    with pytest.raises(ValueError, match="symlink|canonical|path"):
        runner.run_specs(specs)
    assert list(outside.rglob("*")) == []


def test_dry_run_build_allows_read_only_nvidia_discovery_but_never_trainer_or_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Break caught: dry-run forbids Task 3 discovery or accidentally invokes a training child."""

    root = tmp_path / "campaign"
    (root / "manifests").mkdir(parents=True)
    suite = _suite(tmp_path)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(runner, "load_v4_suite_identity", lambda *_: suite)

    def discover_then_resolve(args: argparse.Namespace) -> SimpleNamespace:
        runner.subprocess.run(("nvidia-smi", "--query-gpu=driver_version"), check=False)
        return _fake_resolver([])(args)

    monkeypatch.setattr(runner, "resolve_run_configuration", discover_then_resolve)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(tuple(argv)) or SimpleNamespace(returncode=0, stdout=""),
    )
    assert runner.main([
        "--output-root", str(root), "--formal-suite", str(suite.path),
        "--specification", str(tmp_path / "spec.md"),
        "--unit-coefficients", str(root / "analysis/unit_coefficients.json"),
        "--candidate-registry", str(root / "manifests/candidate_registry.json"),
        "--device", "cpu", "--dry-run",
    ]) == 0
    assert calls and all(call[0] == "nvidia-smi" for call in calls)
    assert not (root / "runs").exists()
    assert not (root / "logs").exists()
    assert "40 dry cells" in capsys.readouterr().out


def test_import_and_direct_launcher_help_leave_the_actual_project_and_root_unwritten(
    tmp_path: Path,
) -> None:
    """Break caught: importing or direct execution writes the project/cwd rather than only parsing CLI."""

    root = tmp_path / "actual-root"
    root.mkdir()
    project_before = {
        path.relative_to(PROJECT_ROOT): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (PROJECT_ROOT / "experiments").glob("run_soft_loss_v4_ablation10.py")
    }
    root_before = set(root.rglob("*"))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    imported = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import experiments.run_soft_loss_v4_ablation10", str(PROJECT_ROOT)],
        cwd=root, env=environment, capture_output=True, text=True, check=False,
    )
    assert imported.returncode == 0, imported.stderr
    launched = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "experiments/run_soft_loss_v4_ablation10.py"), "--help"],
        cwd=root, env=environment, capture_output=True, text=True, check=False,
    )
    assert launched.returncode == 0, launched.stderr
    assert "--output-root" in launched.stdout
    assert set(root.rglob("*")) == root_before
    assert {
        path.relative_to(PROJECT_ROOT): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (PROJECT_ROOT / "experiments").glob("run_soft_loss_v4_ablation10.py")
    } == project_before
