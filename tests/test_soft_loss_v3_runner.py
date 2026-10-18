"""Fail-closed contracts for the immutable soft-loss-v3 stage runner."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from dataclasses import replace
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import pytest

import experiments.run_soft_loss_v3 as runner_module
import experiments.soft_loss_v3_registry as registry_module
from experiments.run_soft_loss_v3 import (
    RunSpec,
    StageContext,
    build_runner_parser,
    build_stage_specs,
    expected_config_hash,
    is_completed_run,
    required_upstream,
    runner_main,
    run_specs,
    write_formal_campaign_manifest,
    write_stage_manifest,
    scientific_config_hash,
    trainer_argv,
)
from shadowforge_pipeline.soft_voxel_v3.provenance import (
    canonical_hash,
    implementation_source_manifest,
)
from shadowforge_pipeline.soft_voxel_v3 import trainer as trainer_module
from shadowforge_pipeline.soft_voxel_v3 import calibration as calibration_module
from experiments.soft_loss_v3_registry import (
    SuiteIdentity,
    binding_main,
    load_bound_suite_identity,
)


PRIMARY = ("l12_bal", "dic_only", "dts_only", "v3_full")
APPENDIX = ("v3_zero_margin", "v3_no_l1", "v3_no_l2")
DEVELOPMENT = ("arcane_casters", "blade_heroes", "stage_dancers")
FORMAL = tuple(f"connected_{index:02d}" for index in range(1, 11))


STAGES = {
    "bind_manifests": ((), (), "development", 0, 0, 0),
    "calibrate128": (("arcane_casters",), ("l12_bal",), "calibration", 0, 128, 2000),
    "smoke128": (("arcane_casters",), PRIMARY, "development", 0, 128, 200),
    "screen128": (DEVELOPMENT, PRIMARY, "development", 0, 128, 1000),
    "confirm128": (DEVELOPMENT, PRIMARY, "development", 0, 128, 2000),
    "cuda2step256": (("arcane_casters",), PRIMARY, "integration", 0, 256, 2),
    "resource256": (("arcane_casters",), ("l12_bal", "v3_full"), "resource", 0, 256, 200),
    "transfer256": (DEVELOPMENT, PRIMARY, "development", 0, 256, 2000),
    "sentinel_repeat256": (("arcane_casters",), PRIMARY, "technical_repeat", 1, 256, 2000),
    "formal256_primary": (FORMAL, PRIMARY, "primary", 0, 256, 2000),
    "formal256_repeat": (FORMAL, PRIMARY, "technical_repeat", 1, 256, 2000),
    "appendix256": (FORMAL, APPENDIX, "exploratory", 0, 256, 2000),
}


UPSTREAM = {
    "bind_manifests": None,
    "calibrate128": None,
    "smoke128": None,
    "screen128": ("smoke128", "ADVANCE"),
    "confirm128": ("screen128", "ADVANCE"),
    "cuda2step256": ("confirm128", "ADVANCE"),
    "resource256": ("cuda2step256", "ADVANCE"),
    "transfer256": ("resource256", "ADVANCE"),
    "sentinel_repeat256": ("transfer256", "ADVANCE"),
    "formal256_primary": ("sentinel_repeat256", "ADVANCE"),
    "formal256_repeat": ("formal256_primary", "COMPLETE_ONLY"),
    "appendix256": ("formal_conclusion", "LOCKED"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_fixture(tmp_path: Path) -> dict[str, object]:
    tasks: dict[str, dict[str, object]] = {}
    for task_index, task in enumerate((*DEVELOPMENT, *FORMAL)):
        inputs = []
        for axis_index, axis in enumerate("xyz"):
            target = tmp_path / "inputs" / task / f"view_{axis}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"runner-input\0" + bytes([task_index, axis_index]))
            inputs.append(
                {
                    "axis": axis,
                    "source_path": str(target.resolve()),
                    "resolved_path": str(target.resolve()),
                    "sha256": _sha(target),
                }
            )
        tasks[task] = {"case_id": task, "inputs": inputs}
    files = {}
    for name in ("master", "development", "formal", "spec", "authorization", "campaign"):
        path = tmp_path / f"{name}.json"
        path.write_bytes(f"{name}-identity".encode("ascii"))
        files[name] = {"path": str(path.resolve()), "sha256": _sha(path)}
    weight_path = tmp_path / "weights.json"
    weight_path.write_text(
        json.dumps(
            {
                "schema": "pcraft_soft_loss_v3_weights_v1",
                "status": "APPROVED_FROZEN",
                "lambdas": {"lambda_dic": 0.25, "lambda_dts": 0.5},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    files["weights"] = {"path": str(weight_path.resolve()), "sha256": _sha(weight_path)}
    files["development"]["semantic_sha256"] = "d" * 64
    files["formal"]["semantic_sha256"] = "f" * 64
    files["master"]["semantic_sha256"] = "a" * 64
    return {
        "tasks": tasks,
        "master": files["master"],
        "development": files["development"],
        "formal": files["formal"],
        "specification": files["spec"],
        "weights": {
            **files["weights"],
            "status": "APPROVED_FROZEN",
            "semantic_sha256": "b" * 64,
            "lambdas": {"lambda_dic": 0.25, "lambda_dts": 0.5},
        },
        "authorization": files["authorization"],
        "formal_campaign": files["campaign"],
        "implementation": runner_module._ordinary_source_identity(),
    }


def _context(tmp_path: Path, *, calibration: bool) -> StageContext:
    output = tmp_path / "output"
    output.mkdir()
    paths = {}
    for name in ("master", "development", "formal", "weights", "authorization", "campaign"):
        path = tmp_path / f"{name}.json"
        if not path.exists():
            path.write_bytes(name.encode("ascii"))
        paths[name] = path
    return StageContext(
        output_root=output.resolve(),
        master_manifest=paths["master"].resolve(),
        development_suite=paths["development"].resolve(),
        formal_suite=paths["formal"].resolve(),
        frozen_weights=None if calibration else paths["weights"].resolve(),
        authorization=None if calibration else paths["authorization"].resolve(),
        formal_campaign=None if calibration else paths["campaign"].resolve(),
    )


def test_required_upstream_is_the_exact_frozen_dag() -> None:
    """Break caught: a stage is auto-chained, skips a gate, or accepts a wrong status."""

    assert {stage: required_upstream(stage) for stage in STAGES} == UPSTREAM
    for invalid in ("all", "", "formal", "unknown"):
        with pytest.raises(ValueError, match="stage|registered|unknown"):
            required_upstream(invalid)


@pytest.mark.parametrize("stage", tuple(STAGES))
def test_every_stage_builds_the_exact_full_cartesian_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """Break caught: count-preserving duplicate/hidden cells replace a registered row."""

    tasks, variants, role, replicate, resolution, steps = STAGES[stage]
    fixture = _identity_fixture(tmp_path)
    if stage in {"bind_manifests", "calibrate128"}:
        fixture["weights"] = None
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda *_: fixture)
    specs = build_stage_specs(
        stage,
        _context(tmp_path, calibration=stage in {"bind_manifests", "calibrate128"}),
    )
    expected = {
        (task, variant, role, replicate)
        for task in tasks
        for variant in variants
    }
    observed = {
        (spec.task, spec.variant, spec.replicate_role, spec.replicate_index)
        for spec in specs
    }
    assert observed == expected
    assert len(specs) == len(expected)
    assert len({spec.run_id for spec in specs}) == len(specs)
    assert len({spec.output_path for spec in specs}) == len(specs)
    assert all(spec.stage == stage for spec in specs)
    assert all(spec.resolution == resolution and spec.export_resolution == resolution for spec in specs)
    assert all(spec.steps == steps for spec in specs)
    assert all(spec.inference_eligible is (stage == "formal256_primary") for spec in specs)
    assert all(tuple(item["axis"] for item in spec.inputs_xyz) == ("x", "y", "z") for spec in specs)
    if stage == "calibrate128":
        assert specs[0].frozen_lambdas is None
        assert specs[0].weight_identity is None
    elif specs:
        assert all(spec.frozen_lambdas == {"lambda_dic": 0.25, "lambda_dts": 0.5} for spec in specs)
        assert all(spec.weight_identity is not None for spec in specs)


def test_runspec_and_stage_context_are_frozen(tmp_path: Path) -> None:
    """Break caught: orchestration mutates an already registered cell or context."""

    context = _context(tmp_path, calibration=True)
    with pytest.raises(FrozenInstanceError):
        context.output_root = tmp_path


def test_runspec_is_recursively_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: frozen RunSpec still exposes mutable nested science identities."""

    spec = _one_spec(tmp_path, monkeypatch)
    with pytest.raises(TypeError):
        spec.inputs_xyz[0]["sha256"] = "0" * 64
    with pytest.raises(TypeError):
        spec.master_identity["sha256"] = "0" * 64
    with pytest.raises(TypeError):
        spec.source_identity["bundle_sha256"] = "0" * 64
    assert spec.frozen_lambdas is not None
    with pytest.raises(TypeError):
        spec.frozen_lambdas["lambda_dic"] = 99.0


def _one_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage: str = "smoke128",
) -> RunSpec:
    fixture = _identity_fixture(tmp_path)
    if stage == "calibrate128":
        fixture["weights"] = None
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda *_: fixture)
    # These narrow unit fixtures intentionally use opaque identity bytes rather
    # than Task 9 manifests.  Tests of the public writer's exact Task 9 row
    # binding use _real_smoke_specs below and leave this validator intact.
    monkeypatch.setattr(runner_module, "_verify_registered_inputs", lambda spec: None)
    return build_stage_specs(
        stage,
        _context(tmp_path, calibration=stage == "calibrate128"),
    )[0]


def test_runner_parser_exposes_only_the_registered_single_stage_interface() -> None:
    """Break caught: CLI abbreviation, `all`, or an unregistered tuning knob is accepted."""

    parser = build_runner_parser()
    assert parser.allow_abbrev is False
    assert {action.dest for action in parser._actions} == {
        "help",
        "stage",
        "output_root",
        "master_manifest",
        "development_suite",
        "formal_suite",
        "weights",
        "authorization",
        "formal_campaign",
        "resume",
        "dry_run",
        "max_runs",
    }
    stage_action = next(action for action in parser._actions if action.dest == "stage")
    assert set(stage_action.choices) == set(STAGES)
    assert "all" not in stage_action.choices
    required = {
        action.dest
        for action in parser._actions
        if action.required
    }
    assert required == {
        "stage",
        "output_root",
        "master_manifest",
        "development_suite",
        "formal_suite",
    }
    valid = [
        "--stage", "smoke128",
        "--output-root", "/tmp/out",
        "--master-manifest", "/tmp/master",
        "--development-suite", "/tmp/development",
        "--formal-suite", "/tmp/formal",
    ]
    for abbreviation in ("--stag", "--output-r", "--master-m", "--development-s", "--formal-s"):
        argv = list(valid)
        index = next(index for index, item in enumerate(argv) if item.startswith(abbreviation))
        argv[index] = abbreviation
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    with pytest.raises(SystemExit):
        parser.parse_args([*valid, "--threshold", "0.1"])
    for value in ("0", "-1", "true"):
        with pytest.raises(SystemExit):
            parser.parse_args([*valid, "--max-runs", value])


def _runner_cli(tmp_path: Path, stage: str, *extra: str) -> list[str]:
    return [
        "--stage",
        stage,
        "--output-root",
        str((tmp_path / "output").resolve()),
        "--master-manifest",
        str((tmp_path / "master.json").resolve()),
        "--development-suite",
        str((tmp_path / "development.json").resolve()),
        "--formal-suite",
        str((tmp_path / "formal.json").resolve()),
        *extra,
    ]


@pytest.mark.parametrize(
    ("stage", "extra", "message"),
    [
        ("calibrate128", ("--weights", "/tmp/weights"), "weight|irrelevant"),
        ("smoke128", (), "weight|required"),
        ("screen128", ("--weights", "/tmp/weights"), "authorization|required"),
        (
            "screen128",
            ("--weights", "/tmp/weights", "--authorization", "/tmp/auth", "--formal-campaign", "/tmp/campaign"),
            "campaign|irrelevant",
        ),
        ("formal256_primary", ("--weights", "/tmp/weights", "--authorization", "/tmp/auth"), "campaign|required"),
    ],
)
def test_runner_main_rejects_missing_or_irrelevant_stage_options_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    extra: tuple[str, ...],
    message: str,
) -> None:
    """Break caught: the CLI silently defaults or accepts outcome-changing path options."""

    monkeypatch.setattr(
        runner_module,
        "build_stage_specs",
        lambda *args, **kwargs: pytest.fail("invalid stage options reached spec construction"),
    )
    with pytest.raises(ValueError, match=message):
        runner_main(_runner_cli(tmp_path, stage, *extra))


def test_runner_main_freezes_formal_campaign_before_primary_run_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: primary outcomes can alter the pre-registered repeat matrix."""

    primary = _fake_specs(tmp_path, monkeypatch, "formal256_primary")
    order: list[tuple[str, object]] = []
    monkeypatch.setattr(runner_module, "build_stage_specs", lambda stage, context: primary)

    def freeze(first: object, repeats: object, path: Path) -> Path:
        assert len(first) == len(repeats) == 40
        assert all(spec.stage == "formal256_repeat" for spec in repeats)
        order.append(("campaign", path))
        return path

    monkeypatch.setattr(runner_module, "write_formal_campaign_manifest", freeze)
    monkeypatch.setattr(
        runner_module,
        "run_specs",
        lambda specs, **kwargs: order.append(("runs", len(specs))) or {"registered": 40},
    )
    campaign = primary[0].output_path.parents[5] / "manifests" / "formal_campaign.json"
    argv = _runner_cli(
        tmp_path,
        "formal256_primary",
        "--weights",
        str((tmp_path / "weights.json").resolve()),
        "--authorization",
        str((tmp_path / "auth.json").resolve()),
        "--formal-campaign",
        str(campaign),
        "--dry-run",
    )
    assert runner_main(argv) == 0
    assert order == [("campaign", campaign), ("runs", 40)]


def test_runner_main_bind_manifests_validates_and_returns_zero_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: bind mode rebinding files or passing an empty sequence to run_specs."""

    calls: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "build_stage_specs",
        lambda stage, context: calls.append(stage) or [],
    )
    monkeypatch.setattr(
        runner_module,
        "run_specs",
        lambda *args, **kwargs: pytest.fail("bind_manifests invoked run_specs"),
    )
    assert runner_main(_runner_cli(tmp_path, "bind_manifests", "--dry-run")) == 0
    assert calls == ["bind_manifests"]


def test_scientific_hash_changes_for_science_but_not_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: blacklist hashing omits science or includes primary/repeat wrappers."""

    spec = _one_spec(tmp_path, monkeypatch)
    baseline = scientific_config_hash(spec)
    included = {
        "task": "different-task",
        "variant": "different-variant",
        "profile": "different-profile",
        "inputs_xyz": tuple(reversed(spec.inputs_xyz)),
        "resolution": spec.resolution + 1,
        "export_resolution": spec.export_resolution + 1,
        "steps": spec.steps + 1,
        "device": "cpu",
        "optimizer": "SGD",
        "learning_rate": 0.008,
        "initialization_logit": -3.0,
        "gradient_clip": 2.0,
        "seed": 18,
        "l1_weight": 0.0,
        "l2_weight": 0.0,
        "use_dic": not spec.use_dic,
        "use_dts": not spec.use_dts,
        "dic_threshold": 0.07,
        "dts_thresholds": (0.03, 0.07),
        "dic_schedule": (0.1, 0.9),
        "dts_schedule": (0.2, 0.8),
        "evaluation_thresholds": (0.03, 0.05, 0.07),
        "mirror_mode": 0,
        "tsf_mode": 0,
        "tsf4_mode": 1,
        "camera_mode": "perspective",
        "image_size": 128,
        "control_resolution": 1,
        "zdist": 1.8,
        "volume_extent": 1.8,
        "samples_per_ray": 151,
        "final_step_export": False,
        "soft_only": False,
        "frozen_lambdas": {"lambda_dic": 0.125, "lambda_dts": 0.5},
        "master_identity": {**spec.master_identity, "sha256": "1" * 64},
        "suite_identity": {**spec.suite_identity, "sha256": "2" * 64},
        "specification_identity": {
            **spec.specification_identity,
            "sha256": "3" * 64,
        },
        "source_identity": {**spec.source_identity, "bundle_sha256": "0" * 64},
        "weight_identity": {**spec.weight_identity, "sha256": "4" * 64},
    }
    for field, value in included.items():
        assert scientific_config_hash(replace(spec, **{field: value})) != baseline, field
    excluded = {
        "stage": "formal256_repeat",
        "run_id": "other/orchestration/id",
        "replicate_role": "technical_repeat",
        "replicate_index": 1,
        "output_path": tmp_path / "other-output",
        "inference_eligible": not spec.inference_eligible,
    }
    for field, value in excluded.items():
        assert scientific_config_hash(replace(spec, **{field: value})) == baseline, field


def test_ordinary_source_identity_binds_task8_and_both_task11_drivers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: ordinary registrations omit the runner/registry bytes they execute."""

    source = runner_module._ordinary_source_identity()
    assert set(source) == {"schema", "task8", "drivers", "bundle_sha256"}
    assert source["schema"] == "pcraft_soft_loss_v3_runner_source_v1"
    assert source["task8"] == implementation_source_manifest()
    assert set(source["drivers"]) == {
        "experiments/run_soft_loss_v3.py",
        "experiments/soft_loss_v3_registry.py",
    }
    for logical, record in source["drivers"].items():
        path = Path(record["path"])
        assert path == Path(__file__).resolve().parents[1] / logical
        assert record["sha256"] == _sha(path)
    assert source["bundle_sha256"] == canonical_hash(
        {"task8": source["task8"], "drivers": source["drivers"]}
    )

    specs = _real_smoke_specs(tmp_path, monkeypatch)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: "9" * 64)
    changed = json.loads(runner_module.canonical_json_bytes(source))
    changed["drivers"]["experiments/run_soft_loss_v3.py"]["sha256"] = "0" * 64
    changed["bundle_sha256"] = canonical_hash(
        {"task8": changed["task8"], "drivers": changed["drivers"]}
    )
    monkeypatch.setattr(runner_module, "_ordinary_source_identity", lambda: changed)
    with pytest.raises(ValueError, match="source|driver|implementation|changed"):
        write_stage_manifest("smoke128", specs, specs[0].output_path.parents[5])


def test_formal_primary_repeat_pairs_have_identical_scientific_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: technical-repeat orchestration changes a paired scientific cell."""

    fixture = _identity_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda *_: fixture)
    context = _context(tmp_path, calibration=False)
    primary = build_stage_specs("formal256_primary", context)
    repeat = build_stage_specs("formal256_repeat", context)
    paired = {
        (spec.task, spec.variant): scientific_config_hash(spec)
        for spec in primary
    }
    assert {
        (spec.task, spec.variant): scientific_config_hash(spec)
        for spec in repeat
    } == paired


def test_trainer_argv_is_exact_and_calibration_is_the_only_zero_applied_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: argv reorders xyz, changes raw weights, or omits soft final export."""

    normal = _one_spec(tmp_path / "normal", monkeypatch)
    argv = trainer_argv(normal)
    assert argv[:3] == [
        "/home/lin/anaconda3/envs/shadowart-revisited/bin/python",
        "-m",
        "shadowforge_pipeline.soft_voxel_v3.trainer",
    ]
    assert argv[3:7] == ["cuda:0", str(normal.output_path), "200", "0.007"]
    shadow_index = argv.index("--shadow-files")
    assert argv[shadow_index + 1 : shadow_index + 4] == [
        item["resolved_path"] for item in normal.inputs_xyz
    ]
    assert argv[argv.index("--lambda-dic") + 1] == "0.25"
    assert argv[argv.index("--lambda-dts") + 1] == "0.5"
    assert "--weight-config" in argv
    assert "--calibration-probe" not in argv
    assert argv[-2:] == ["--export-final-step", "--soft-only-eval"]
    resumed = trainer_argv(normal, resume=True)
    assert resumed[-1] == "--resume"
    assert resumed.count("--resume") == 1

    calibration = _one_spec(tmp_path / "calibration", monkeypatch, stage="calibrate128")
    calibration_argv = trainer_argv(calibration)
    assert calibration_argv[calibration_argv.index("--lambda-dic") + 1] == "0.0"
    assert calibration_argv[calibration_argv.index("--lambda-dts") + 1] == "0.0"
    assert "--calibration-probe" in calibration_argv
    assert "--weight-config" not in calibration_argv


def test_expected_config_hash_delegates_to_real_task8_parser_and_resolver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: runner maintains a hand-written config schema that drifts from Task 8."""

    spec = _one_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        trainer_module,
        "environment_manifest",
        lambda _device: {"nondeterminism_reason": "fixture", "environment": {}},
    )
    parsed = trainer_module.build_parser().parse_args(trainer_argv(spec)[3:])
    expected = canonical_hash(trainer_module.resolved_config(parsed))
    assert expected_config_hash(spec) == expected


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_bound_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    project = tmp_path / "project"
    development = project / "development"
    formal = project / "formal"
    development.mkdir(parents=True)
    formal.mkdir(parents=True)
    development_tasks = []
    for task_index, task in enumerate(DEVELOPMENT):
        views = []
        for view_index in range(3):
            path = development / task / f"view_{view_index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"development\0" + bytes([task_index, view_index]))
            views.append(str(path.resolve()))
        development_tasks.append({"task": task, "views": views})
    _write_json(
        development / "MANIFEST.json",
        {
            "protocol": "fixed-3-view TFS, mr=1, 256^3",
            "foreground": "white-on-black",
            "tasks": development_tasks,
        },
    )
    inputs = formal / "inputs"
    inputs.mkdir()
    formal_cases = []
    for task_index, task in enumerate(FORMAL):
        targets = []
        for axis_index, axis in enumerate("xyz"):
            path = inputs / f"{task}_view_{axis}.png"
            path.write_bytes(b"formal\0" + bytes([task_index, axis_index]))
            targets.append(str(path.resolve()))
        formal_cases.append({"case_id": task, "target_paths_xyz": targets})
    formal_manifest = formal / "benchmark_manifest.json"
    _write_json(
        formal_manifest,
        {
            "schema": "fixed3_mr1_connected10_v1",
            "status": "APPROVED_FROZEN",
            "protocol": {
                "views": 3,
                "declared_rendered_views": 6,
                "axes": ["x", "y", "z"],
                "camera": "fixed orthographic TFS",
                "tsf": 1,
                "mr": 1,
                "zdist": 1.7,
                "resolution": 256,
                "seeds": [17, 29, 43],
                "evaluation": "shared canonical masks and prescribed mirrors; no registration",
            },
            "cases": formal_cases,
        },
    )
    spec = project / "revised-design.md"
    spec.write_text("# frozen runner fixture\n", encoding="utf-8")
    output = tmp_path / "bound"
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", project)
    assert binding_main(
        [
            "--output-root", str(output),
            "--development-root", str(development),
            "--formal-benchmark", str(formal_manifest),
            "--spec", str(spec),
        ]
    ) == 0
    return {
        "output": output,
        "master": output / "experiment_manifest.json",
        "development": output / "manifests" / "development_suite.json",
        "formal": output / "manifests" / "formal_suite.json",
        "spec": spec,
    }


def _task10_weight_fixture(
    paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    *,
    lineage_root: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    """Install a complete Task 10-shaped weight with selectable causal root."""

    output = paths["output"].resolve()
    causal_root = output if lineage_root is None else lineage_root.resolve()
    stage = causal_root / "manifests" / "stages" / "calibrate128.json"
    stage.parent.mkdir(parents=True, exist_ok=True)
    calibration_run = (
        causal_root
        / "calibrate128"
        / "steps_2000"
        / "arcane_casters"
        / "l12_bal"
        / "replicate_0"
        / "seed_17"
    )
    run_record = {
        "path": str(calibration_run),
        "task": "arcane_casters",
        "profile": "l12_bal",
        "resolution": 128,
        "export_resolution": 128,
        "steps": 2000,
        "seed": 17,
        "soft_only": True,
        "calibration_probe": True,
        "config_hash": "1" * 64,
        "run_manifest_sha256": "2" * 64,
        "gradient_audit_sha256": "3" * 64,
        "artifact_manifest_sha256": "4" * 64,
        "status_sha256": "5" * 64,
    }
    stage_payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_calibration_stage_v1",
        "status": "APPROVED_FROZEN",
        "calibration_run": run_record,
    }
    stage_payload["semantic_sha256"] = canonical_hash(stage_payload)
    stage.write_bytes(runner_module.canonical_json_bytes(stage_payload))
    master_payload = json.loads(paths["master"].read_bytes())
    development_payload = json.loads(paths["development"].read_bytes())
    development = load_bound_suite_identity(paths["development"], "development")
    provenance = {
        "stage_manifest": _record(
            stage, semantic=stage_payload["semantic_sha256"]
        ),
        "calibration_run": run_record,
        "master_manifest": _record(
            paths["master"], semantic=master_payload["semantic_sha256"]
        ),
        "development_suite": _record(
            paths["development"], semantic=development_payload["semantic_sha256"]
        ),
        "specification": _record(paths["spec"]),
        "inputs_xyz": [
            {
                "axis": item.axis,
                "source_path": item.source_path,
                "resolved_path": str(item.resolved_path),
                "sha256": item.sha256,
            }
            for item in development.tasks[0].inputs
        ],
        "implementation": implementation_source_manifest(),
        "interpreter": {
            "path": str(runner_module._LOCKED_INTERPRETER),
            "sha256": _sha(runner_module._LOCKED_INTERPRETER),
            "python": "3.10",
        },
        "executed_source_bundle": runner_module._calibration_executed_source_bundle()[0],
    }
    payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_weights_v1",
        "status": "APPROVED_FROZEN",
        "lambdas": {"lambda_dic": 0.25, "lambda_dts": 0.5},
        "calibration": {
            "factor": 0.15,
            "epsilon": 1e-12,
            "formula_dic": "0.15 * median(g0) / (median(g_dic) + 1e-12)",
            "formula_dts": "0.15 * median(g0) / (median(g_dts_004,g_dts_006 by ordered update) + 1e-12)",
        },
        "measurements": [],
        "provenance": provenance,
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    weight = output / "analysis" / "weights_frozen.json"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(runner_module.canonical_json_bytes(payload))
    monkeypatch.setattr(
        runner_module,
        "validate_frozen_weight_config",
        lambda candidate: payload,
    )
    return weight, payload


def test_suite_identity_loader_returns_one_frozen_ordered_hash_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: runner path-only loads then reopens/reorders suite JSON independently."""

    paths = _make_bound_registry(tmp_path, monkeypatch)
    development = load_bound_suite_identity(paths["development"], "development")
    formal = load_bound_suite_identity(paths["formal"], "formal")
    assert isinstance(development, SuiteIdentity)
    assert development.path == paths["development"].resolve()
    assert development.sha256 == _sha(paths["development"])
    assert [task.case_id for task in development.tasks] == list(DEVELOPMENT)
    assert [task.case_id for task in formal.tasks] == list(FORMAL)
    assert [item.axis for item in formal.tasks[0].inputs] == ["x", "y", "z"]
    assert [item.resolved_path.name for item in formal.tasks[0].inputs] == [
        "connected_01_view_x.png",
        "connected_01_view_y.png",
        "connected_01_view_z.png",
    ]
    with pytest.raises(FrozenInstanceError):
        development.sha256 = "0" * 64


def test_suite_identity_loader_rehashes_current_input_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: suite identity trusts stored hashes after an x/y/z input drifts."""

    paths = _make_bound_registry(tmp_path, monkeypatch)
    identity = load_bound_suite_identity(paths["development"], "development")
    target = identity.tasks[0].inputs[0].resolved_path
    target.write_bytes(target.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="input|sha256|changed"):
        load_bound_suite_identity(paths["development"], "development")


def test_real_context_binds_master_suites_spec_weight_and_current_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: real matrix construction bypasses Task 9/10 strict identities."""

    paths = _make_bound_registry(tmp_path, monkeypatch)
    weights, _ = _task10_weight_fixture(paths, monkeypatch)
    context = StageContext(
        output_root=paths["output"].resolve(),
        master_manifest=paths["master"].resolve(),
        development_suite=paths["development"].resolve(),
        formal_suite=paths["formal"].resolve(),
        frozen_weights=weights.resolve(),
        authorization=None,
        formal_campaign=None,
    )
    specs = build_stage_specs("smoke128", context)
    assert len(specs) == 4
    assert [Path(item["resolved_path"]) for item in specs[0].inputs_xyz] == [
        item.resolved_path
        for item in load_bound_suite_identity(paths["development"], "development").tasks[0].inputs
    ]
    assert specs[0].master_identity["sha256"] == _sha(paths["master"])
    assert specs[0].suite_identity["sha256"] == _sha(paths["development"])
    assert specs[0].specification_identity["sha256"] == _sha(paths["spec"])
    assert specs[0].weight_identity["sha256"] == _sha(weights)


def test_context_requires_task9_trio_at_exact_paths_under_same_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: runner accepts a valid Task9 trio relocated from another output root."""

    paths = _make_bound_registry(tmp_path, monkeypatch)
    context = StageContext(
        output_root=(tmp_path / "different-output").resolve(),
        master_manifest=paths["master"].resolve(),
        development_suite=paths["development"].resolve(),
        formal_suite=paths["formal"].resolve(),
        frozen_weights=None,
        authorization=None,
        formal_campaign=None,
    )
    with pytest.raises(ValueError, match="master|suite|output|registered|path"):
        build_stage_specs("calibrate128", context)


def test_context_recomputes_exact_task9_master_payload_not_only_semantic_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a coherently rehashed alteration of the frozen Task9 protocol passes."""

    paths = _make_bound_registry(tmp_path, monkeypatch)
    master = json.loads(paths["master"].read_bytes())
    master["determinism"]["policy"] = "coherently-rehashed-but-unapproved"
    master["semantic_sha256"] = canonical_hash(
        {
            field: master[field]
            for field in registry_module._MASTER_SCIENTIFIC_FIELDS
        }
    )
    paths["master"].write_bytes(runner_module.canonical_json_bytes(master))
    context = StageContext(
        output_root=paths["output"].resolve(),
        master_manifest=paths["master"].resolve(),
        development_suite=paths["development"].resolve(),
        formal_suite=paths["formal"].resolve(),
        frozen_weights=None,
        authorization=None,
        formal_campaign=None,
    )
    with pytest.raises(ValueError, match="master|exact|approved|payload|determinism"):
        build_stage_specs("calibrate128", context)


def test_normal_weight_must_use_exact_output_root_analysis_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a valid weight is accepted from an unregistered filename/root."""

    paths = _make_bound_registry(tmp_path, monkeypatch)
    registered, payload = _task10_weight_fixture(paths, monkeypatch)
    foreign = tmp_path / "foreign-weight.json"
    foreign.write_bytes(registered.read_bytes())
    monkeypatch.setattr(
        runner_module, "validate_frozen_weight_config", lambda candidate: payload
    )
    context = StageContext(
        output_root=paths["output"].resolve(),
        master_manifest=paths["master"].resolve(),
        development_suite=paths["development"].resolve(),
        formal_suite=paths["formal"].resolve(),
        frozen_weights=foreign.resolve(),
        authorization=None,
        formal_campaign=None,
    )
    with pytest.raises(ValueError, match="weight|analysis|exact|registered|path"):
        build_stage_specs("smoke128", context)


def test_normal_weight_must_bind_current_calibration_causal_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: copied valid Task 10 bytes authorize a different calibration lineage."""

    paths = _make_bound_registry(tmp_path, monkeypatch)
    weight, _ = _task10_weight_fixture(
        paths,
        monkeypatch,
        lineage_root=tmp_path / "foreign-campaign",
    )
    context = StageContext(
        output_root=paths["output"].resolve(),
        master_manifest=paths["master"].resolve(),
        development_suite=paths["development"].resolve(),
        formal_suite=paths["formal"].resolve(),
        frozen_weights=weight.resolve(),
        authorization=None,
        formal_campaign=None,
    )
    with pytest.raises(ValueError, match="weight|provenance|calibrat|lineage|root|stage"):
        build_stage_specs("smoke128", context)


def test_context_rejects_calibration_weight_and_postcalibration_missing_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: calibration consumes frozen weights or later stages silently default them."""

    paths = _make_bound_registry(tmp_path, monkeypatch)
    weight = tmp_path / "weight.json"
    weight.write_text("unused", encoding="utf-8")
    base = dict(
        output_root=paths["output"].resolve(),
        master_manifest=paths["master"].resolve(),
        development_suite=paths["development"].resolve(),
        formal_suite=paths["formal"].resolve(),
        authorization=None,
        formal_campaign=None,
    )
    with pytest.raises(ValueError, match="calibrat|weight|irrelevant"):
        build_stage_specs("calibrate128", StageContext(**base, frozen_weights=weight.resolve()))
    with pytest.raises(ValueError, match="weight|required"):
        build_stage_specs("smoke128", StageContext(**base, frozen_weights=None))


def _fake_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> list[RunSpec]:
    fixture = _identity_fixture(tmp_path)
    if stage == "calibrate128":
        fixture["weights"] = None
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda *_: fixture)
    # Execution-loop unit tests isolate launch/resume behavior from the real
    # Task 9 registry fixtures exercised by the provenance tests below.
    monkeypatch.setattr(runner_module, "_verify_registered_inputs", lambda spec: None)
    specs = build_stage_specs(
        stage,
        _context(tmp_path, calibration=stage == "calibrate128"),
    )
    edge = UPSTREAM[stage]
    if specs and edge is not None and edge[1] != "LOCKED":
        source, status = edge
        root = specs[0].output_path.parents[5]
        authorization: dict[str, object] = {
            "schema": "pcraft_soft_loss_v3_authorization_v1",
            "status": status,
            "source_stage": source,
            "next_stage": stage,
            "report": {},
            "dependencies": {},
            "completion": None,
        }
        authorization["semantic_sha256"] = canonical_hash(authorization)
        path = root / "authorizations" / f"{source}__to__{stage}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(runner_module.canonical_json_bytes(authorization))
    return specs


def test_general_stage_manifest_is_complete_canonical_create_once_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: outcomes can add/remove cells or rewrite a pre-registered matrix."""

    context, _, _ = _make_authorized_screen_context(tmp_path, monkeypatch)
    specs = build_stage_specs("screen128", context)
    expected_hashes = [expected_config_hash(spec) for spec in specs]
    output_root = context.output_root
    path = write_stage_manifest("screen128", specs, output_root)
    payload = json.loads(path.read_bytes())
    assert path.read_bytes() == runner_module.canonical_json_bytes(payload)
    assert path == output_root / "manifests" / "stages" / "screen128.json"
    assert payload["schema"] == "pcraft_soft_loss_v3_stage_v1"
    assert payload["status"] == "APPROVED_FROZEN"
    assert payload["stage"] == "screen128"
    assert payload["authorization"]["status"] == "ADVANCE"
    assert payload["authorization"]["source_stage"] == "smoke128"
    assert payload["formal_campaign"] is None
    assert len(payload["runs"]) == 12
    assert [record["run_id"] for record in payload["runs"]] == [spec.run_id for spec in specs]
    assert [record["expected_config_hash"] for record in payload["runs"]] == expected_hashes
    assert all(record["expected_artifacts"]["status"] == "STATUS.json" for record in payload["runs"])
    assert payload["semantic_sha256"] == canonical_hash(
        {key: value for key, value in payload.items() if key != "semantic_sha256"}
    )
    accepted = path.read_bytes()
    assert write_stage_manifest("screen128", specs, output_root) == path
    assert path.read_bytes() == accepted

    changed = list(specs)
    changed[0] = replace(changed[0], learning_rate=0.008)
    with pytest.raises(ValueError, match="constant|frozen|RunSpec|matrix"):
        write_stage_manifest("screen128", changed, output_root)
    assert path.read_bytes() == accepted

    authorization = Path(payload["authorization"]["path"])
    authorization.write_bytes(authorization.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="authorization|canonical|semantic|changed"):
        write_stage_manifest("screen128", specs, output_root)
    assert path.read_bytes() == accepted


def test_public_stage_writer_rejects_count_preserving_profile_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: dic_only can be registered with the baseline science under its old label."""

    specs = _fake_specs(tmp_path, monkeypatch, "smoke128")
    dic_only = specs[1]
    specs[1] = replace(dic_only, profile="l12_bal", use_dic=False)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: "9" * 64)
    with pytest.raises(ValueError, match="profile|constant|frozen|RunSpec|matrix"):
        write_stage_manifest("smoke128", specs, specs[0].output_path.parents[5])


def test_public_stage_writer_rejects_current_but_unregistered_input_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a current hashed input from another suite row replaces registered x."""

    specs = _real_smoke_specs(tmp_path, monkeypatch)
    root = specs[0].output_path.parents[5]
    suite = load_bound_suite_identity(
        root / "manifests" / "development_suite.json", "development"
    )
    blade_x = suite.tasks[1].inputs[0]
    replacement = {
        "axis": blade_x.axis,
        "source_path": blade_x.source_path,
        "resolved_path": str(blade_x.resolved_path),
        "sha256": blade_x.sha256,
    }
    altered = list(specs)
    altered[0] = replace(
        altered[0],
        inputs_xyz=(replacement, *altered[0].inputs_xyz[1:]),
    )
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: "9" * 64)
    with pytest.raises(ValueError, match="input|suite|registered|task|x/y/z"):
        write_stage_manifest("smoke128", altered, root)
    assert not root.joinpath("manifests", "stages", "smoke128.json").exists()


def test_stage_manifest_rejects_count_preserving_duplicate_hidden_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: duplicate cells preserve count while hiding a required variant."""

    specs = _fake_specs(tmp_path, monkeypatch, "smoke128")
    malformed = [specs[0], specs[0], specs[2], specs[3]]
    with pytest.raises(ValueError, match="matrix|duplicate|cell|variant"):
        write_stage_manifest("smoke128", malformed, specs[0].output_path.parents[5])
    assert not (
        specs[0].output_path.parents[5] / "manifests" / "stages" / "smoke128.json"
    ).exists()


def _materialize_calibration_stage_artifacts(spec: RunSpec) -> None:
    spec.output_path.mkdir(parents=True)
    for name in (
        "run_manifest.json",
        "gradient_audit.json",
        "artifact_manifest.json",
        "STATUS.json",
        "calibration_checkpoint_step_00001000.pt",
        "calibration_checkpoint_step_00001300.pt",
        "calibration_checkpoint_step_00001600.pt",
    ):
        spec.output_path.joinpath(name).write_bytes(f"current:{name}".encode("ascii"))


def test_calibration_stage_is_postcompletion_and_exactly_task10_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: calibration gets a generic pre-run stage that Task 10 cannot consume."""

    spec = _one_spec(tmp_path, monkeypatch, stage="calibrate128")
    root = spec.output_path.parents[5]
    with pytest.raises(ValueError, match="completed|calibrat"):
        write_stage_manifest("calibrate128", [spec], root)
    assert not root.joinpath("manifests", "stages", "calibrate128.json").exists()

    _materialize_calibration_stage_artifacts(spec)
    monkeypatch.setattr(runner_module, "is_completed_run", lambda path, value: True)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda value: "8" * 64)
    written = write_stage_manifest("calibrate128", [spec], root)
    payload = json.loads(written.read_bytes())
    assert written.read_bytes() == runner_module.canonical_json_bytes(payload)
    assert set(payload) == {
        "schema",
        "status",
        "calibration_run",
        "checkpoints",
        "master_manifest",
        "development_suite",
        "specification",
        "inputs_xyz",
        "executed_source_bundle",
        "publication",
        "semantic_sha256",
    }
    assert payload["schema"] == "pcraft_soft_loss_v3_calibration_stage_v1"
    assert payload["calibration_run"]["status_sha256"] == payload["publication"]["completed_status_sha256"]
    assert [item["path"] for item in payload["checkpoints"]] == [
        "calibration_checkpoint_step_00001000.pt",
        "calibration_checkpoint_step_00001300.pt",
        "calibration_checkpoint_step_00001600.pt",
    ]
    assert set(payload["executed_source_bundle"]["files"]) == set(
        calibration_module._CALIBRATION_EXECUTED_SOURCE_NAMES
    )
    validated, _ = calibration_module._validate_stage(
        calibration_module._snapshot_file(written, label="Task 11 calibration stage")
    )
    assert validated == payload


def test_formal_campaign_freezes_all_eighty_cells_and_one_to_one_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: primary outcomes create, remove, or scientifically alter repeat cells."""

    fixture = _identity_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda *_: fixture)
    monkeypatch.setattr(runner_module, "_verify_registered_inputs", lambda spec: None)
    context = _context(tmp_path, calibration=False)
    primary = build_stage_specs("formal256_primary", context)
    repeats = build_stage_specs("formal256_repeat", context)
    path = tmp_path / "output" / "manifests" / "formal_campaign.json"
    written = write_formal_campaign_manifest(primary, repeats, path)
    payload = json.loads(written.read_bytes())
    assert written.read_bytes() == runner_module.canonical_json_bytes(payload)
    assert payload["schema"] == "pcraft_soft_loss_v3_formal_campaign_v1"
    assert payload["status"] == "APPROVED_FROZEN"
    assert len(payload["primary"]) == 40
    assert len(payload["repeats"]) == 40
    assert len(payload["pairs"]) == 40
    assert [pair["task_variant"] for pair in payload["pairs"]] == [
        [spec.task, spec.variant] for spec in primary
    ]
    assert all(pair["scientific_config_hash"] == scientific_config_hash(primary[index]) for index, pair in enumerate(payload["pairs"]))
    assert payload["semantic_sha256"] == canonical_hash(
        {key: value for key, value in payload.items() if key != "semantic_sha256"}
    )
    accepted = written.read_bytes()
    assert write_formal_campaign_manifest(primary, repeats, path) == written
    assert written.read_bytes() == accepted

    identity, validated, _ = runner_module._load_formal_campaign_identity(
        written,
        path.parents[1],
    )
    assert identity == {
        "path": str(written),
        "sha256": _sha(written),
        "semantic_sha256": payload["semantic_sha256"],
    }
    assert validated == payload


def test_formal_campaign_writer_requires_the_exact_registered_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: an authorization can point at an alternate campaign wrapper."""

    fixture = _identity_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda *_: fixture)
    context = _context(tmp_path, calibration=False)
    primary = build_stage_specs("formal256_primary", context)
    repeats = build_stage_specs("formal256_repeat", context)
    wrong = tmp_path / "output" / "manifests" / "alternate_campaign.json"
    with pytest.raises(ValueError, match="campaign|exact|path|registered"):
        write_formal_campaign_manifest(primary, repeats, wrong)


def test_formal_campaign_loader_rejects_stale_semantics_and_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: formal execution trusts campaign bytes or builds different cells."""

    fixture = _identity_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda *_: fixture)
    monkeypatch.setattr(runner_module, "_verify_registered_inputs", lambda spec: None)
    context = _context(tmp_path, calibration=False)
    primary = build_stage_specs("formal256_primary", context)
    repeats = build_stage_specs("formal256_repeat", context)
    path = tmp_path / "output" / "manifests" / "formal_campaign.json"
    write_formal_campaign_manifest(primary, repeats, path)
    payload = json.loads(path.read_bytes())
    payload["pairs"][0]["scientific_config_hash"] = "0" * 64
    path.write_bytes(runner_module.canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="semantic|campaign|pair"):
        runner_module._load_formal_campaign_identity(path, path.parents[1])

    payload["semantic_sha256"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "semantic_sha256"}
    )
    fixture["campaign_payload"] = payload
    with pytest.raises(ValueError, match="campaign|differ|frozen"):
        build_stage_specs("formal256_repeat", context)


def test_first_formal_campaign_publication_requires_all_eighty_paths_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: formal cells are retrospectively registered after an outcome exists."""

    fixture = _identity_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda *_: fixture)
    monkeypatch.setattr(runner_module, "_verify_registered_inputs", lambda spec: None)
    context = _context(tmp_path, calibration=False)
    primary = build_stage_specs("formal256_primary", context)
    repeats = build_stage_specs("formal256_repeat", context)
    repeats[-1].output_path.mkdir(parents=True)
    destination = context.output_root / "manifests" / "formal_campaign.json"

    with pytest.raises(ValueError, match="campaign|first publication|run path|absent|retrospective"):
        write_formal_campaign_manifest(primary, repeats, destination)
    assert not destination.exists()


def test_first_technical_repeat_stage_publication_precedes_run_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a technical repeat is registered only after its directory exists."""

    specs = _fake_specs(tmp_path, monkeypatch, "sentinel_repeat256")
    root = specs[0].output_path.parents[5]
    specs[0].output_path.mkdir(parents=True)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: "8" * 64)
    monkeypatch.setattr(
        runner_module,
        "_stage_wrapper_identities",
        lambda stage, output_root, values: ({"synthetic": "validated gate"}, None),
    )
    destination = root / "manifests" / "stages" / "sentinel_repeat256.json"

    with pytest.raises(ValueError, match="repeat|first publication|run path|absent|retrospective"):
        write_stage_manifest("sentinel_repeat256", specs, root)
    assert not destination.exists()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "scientific"])
def test_formal_campaign_rejects_malformed_pair_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: repeat pairing accepts a missing, duplicate, or changed scientific cell."""

    fixture = _identity_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda *_: fixture)
    context = _context(tmp_path, calibration=False)
    primary = build_stage_specs("formal256_primary", context)
    repeats = build_stage_specs("formal256_repeat", context)
    if mutation == "missing":
        repeats.pop()
    elif mutation == "duplicate":
        repeats[-1] = repeats[0]
    else:
        repeats[0] = replace(repeats[0], learning_rate=0.008)
    target = context.output_root / "manifests" / "formal_campaign.json"
    with pytest.raises(ValueError, match="40|pair|duplicate|scientific|matrix|frozen"):
        write_formal_campaign_manifest(primary, repeats, target)
    assert not target.exists()


def _record(path: Path, *, semantic: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": str(path.resolve()), "sha256": _sha(path)}
    if semantic is not None:
        value["semantic_sha256"] = semantic
    return value


def _collected_rows_for_report_evidence(
    evidence: list[dict[str, object]],
    *,
    changed: tuple[bool, ...],
) -> list[dict[str, object]]:
    assert len(evidence) == len(changed)
    return [
        {
            "run_id": item["run_id"],
            "path": item["path"],
            "expected_config_hash": item["expected_config_hash"],
            "status": {"sha256": item["status_sha256"]},
            "artifact_manifest": {
                "sha256": item["artifact_manifest_sha256"],
                "payload": {
                    "artifacts_sha256": {
                        "occupancy_probability.npy": item[
                            "occupancy_probability_sha256"
                        ]
                    }
                },
            },
            "metrics": {"sha256": item["metrics_sha256"]},
            "gradient_audit": {"sha256": item["gradient_audit_sha256"]},
            "resource_metrics": {"sha256": item["resource_metrics_sha256"]},
            "history": {"sha256": item["history_sha256"]},
            "parameter_change": {"changed": flag},
        }
        for item, flag in zip(evidence, changed)
    ]


def _cuda_branch_checks_fixture() -> dict[str, object]:
    expected = {
        "l12_bal": (False, False),
        "dic_only": (True, False),
        "dts_only": (False, True),
        "v3_full": (True, True),
    }
    return {
        variant: {
            "completed_updates": 2,
            "final_progress": 1.0,
            "expected_dic_active": dic_active,
            "expected_dts_active": dts_active,
            "observed_dic_active": dic_active,
            "observed_dts_active": dts_active,
            "finite_gradients": True,
            "soft_only": True,
        }
        for variant, (dic_active, dts_active) in expected.items()
    }


def _task12_gate_result_fixture(gate: str, source_stage: str) -> dict[str, object]:
    if gate == "gradient_budget":
        resolution = 128 if source_stage == "smoke128" else 256
        probe_count = 8 if resolution == 128 else 57
        joint_counts = (1, 1) if resolution == 128 else (14, 1)
        criteria: dict[str, object] = {
            "resolution": resolution,
            "ratio_min": 0.10,
            "ratio_max": 0.20,
            "combined_median_max": 0.35,
            "combined_p95_max": 1.0,
            "adverse_cosine_threshold": -0.5,
            "adverse_cosine_fraction_max": 0.10,
            "component_active_l2_epsilon": 1e-12,
            "support_abs_epsilon": 1e-12,
            "eligible_inactive_policy": "stop_any_eligible_inactive_sample",
        }
        observed: dict[str, object] = {
            "expected_probe_count": probe_count,
            "observed_probe_count": probe_count,
            "dic_ratio_median": 0.15,
            "dts_ratio_median": 0.15,
            "combined_ratio_median": 0.30,
            "combined_ratio_p95": 0.40,
            "adverse_cosine_fraction": 0.0,
            "endpoint_diagnostics": {
                endpoint: {
                    "joint_sample_count": count,
                    "dic_dts_cosine_median": 0.0,
                    "dic_dts_support_iou_median": 0.5,
                }
                for endpoint, count in zip(("0.04", "0.06"), joint_counts)
            },
        }
    elif gate == "development_factorial":
        screen = source_stage == "screen128"
        criteria = {
            "task_count": 3,
            "dic_relative_threshold": 0.025 if screen else 0.05,
            "dts_absolute_threshold": 0.0005 if screen else 0.001,
            "iou_noninferiority_delta": -0.001,
            "recall_floor_delta": -0.005,
            "fpr_ceiling_delta": 0.001,
            "require_nonincreasing_range": not screen,
        }
        task_record = {
            "dic_base_to_dic_relative": 0.10,
            "dic_dts_to_full_relative": 0.10,
            "dts_base_to_dts_absolute": 0.01,
            "dts_dic_to_full_absolute": 0.01,
            "iou_delta_tau_0.05": 0.0,
            "recall_delta_tau_0.05": 0.0,
            "fpr_delta_tau_0.05": 0.0,
            "threshold_range_delta": -0.01,
        }
        observed = {
            "dic_task_equal_effect": 0.10,
            "dic_conditional_task_equal_effects": {
                "base_to_dic": 0.10,
                "dts_to_full": 0.10,
            },
            "dts_task_equal_effect": 0.01,
            "dts_conditional_task_equal_effects": {
                "base_to_dts": 0.01,
                "dic_to_full": 0.01,
            },
            "mean_iou_delta_tau_0.05": 0.0,
            "min_task_recall_delta": 0.0,
            "mean_fpr_delta_tau_0.05": 0.0,
            "mean_threshold_range_delta": -0.01,
            "per_task": {
                task: dict(task_record)
                for task in ("arcane_casters", "blade_heroes", "stage_dancers")
            },
        }
    elif gate == "cuda_integration":
        criteria = {
            "run_count": 4,
            "updates_per_run": 2,
            "final_progress": 1.0,
        }
        observed = {"branch_checks": _cuda_branch_checks_fixture()}
    elif gate == "resource_feasibility":
        criteria = {
            "integration_run_count": 4,
            "benchmark_run_count": 2,
            "warmup_updates": 20,
            "timed_updates": 180,
            "peak_memory_fraction_max": 0.90,
            "full_to_baseline_p95_max": 3.5,
            "baseline_p95_must_be_positive": True,
        }
        observed = {
            "physical_cuda_bytes": 1_000_000,
            "peak_cuda_bytes_by_variant": {"l12_bal": 100_000, "v3_full": 200_000},
            "peak_memory_fraction_by_variant": {"l12_bal": 0.10, "v3_full": 0.20},
            "median_seconds_by_variant": {"l12_bal": 0.01, "v3_full": 0.02},
            "p95_seconds_by_variant": {"l12_bal": 0.02, "v3_full": 0.04},
            "full_to_baseline_p95_ratio": 2.0,
            "integration_branch_checks": _cuda_branch_checks_fixture(),
        }
    elif gate == "repeat_tolerance":
        criteria = {
            "pair_count": 4,
            "tolerance": 1e-4,
            "metrics": ["task_mscd_px", "worst_threshold_iou"],
        }
        observed = {
            "paired_differences": {
                variant: {
                    "task_mscd_px": 0.0,
                    "worst_threshold_iou": 0.0,
                }
                for variant in PRIMARY
            },
            "max_abs_task_mscd_px": 0.0,
            "max_abs_worst_threshold_iou": 0.0,
        }
    else:
        raise AssertionError(f"unknown test gate: {gate}")
    return {
        "schema": "pcraft_soft_loss_v3_gate_result_v1",
        "gate": gate,
        "status": "PASS",
        "criteria": criteria,
        "observed": observed,
        "failure_codes": [],
    }


def _task12_gate_report_fixture(
    tmp_path: Path,
    source_stage: str,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object] | None,
    list[dict[str, object]],
]:
    mapping = {
        "smoke128": ("screen128", "gradient_budget", None, 4),
        "screen128": ("confirm128", "development_factorial", None, 12),
        "confirm128": ("cuda2step256", "development_factorial", None, 12),
        "cuda2step256": ("resource256", "cuda_integration", None, 4),
        "resource256": ("transfer256", "resource_feasibility", "cuda_integration", 2),
        "transfer256": (
            "sentinel_repeat256",
            "development_factorial",
            "gradient_budget",
            12,
        ),
        "sentinel_repeat256": ("formal256_primary", "repeat_tolerance", None, 4),
    }
    next_stage, primary_gate, additional_gate, count = mapping[source_stage]
    output = (tmp_path / "runs").resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_identity = {
        "path": str(output / "manifests" / "stages" / f"{source_stage}.json"),
        "sha256": "1" * 64,
        "semantic_sha256": "2" * 64,
    }
    runs = [
        {
            "run_id": f"{source_stage}-{index:02d}",
            "path": str(output / source_stage / f"run-{index:02d}"),
            "expected_config_hash": f"{index % 10}" * 64,
        }
        for index in range(count)
    ]
    source_payload: dict[str, object] = {"runs": runs}
    evidence = [
        {
            "run_id": row["run_id"],
            "path": row["path"],
            "expected_config_hash": row["expected_config_hash"],
            "completed": True,
            "status_sha256": "3" * 64,
            "artifact_manifest_sha256": "4" * 64,
            "metrics_sha256": "5" * 64,
            "gradient_audit_sha256": "6" * 64,
            "resource_metrics_sha256": "7" * 64,
            "history_sha256": "8" * 64,
            "occupancy_probability_sha256": "9" * 64,
        }
        for row in runs
    ]
    prerequisite = None
    if source_stage in {"resource256", "sentinel_repeat256"}:
        prerequisite = {
            "path": str(output / "manifests" / "stages" / "prerequisite.json"),
            "sha256": "a" * 64,
            "semantic_sha256": "b" * 64,
        }
    report: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_gate_report_v1",
        "status": "ADVANCE",
        "source_stage": source_stage,
        "next_stage": next_stage,
        "gate": primary_gate,
        "source_stage_manifest": source_identity,
        "prerequisite_stage_manifests": [] if prerequisite is None else [prerequisite],
        "formal_campaign": None,
        "registered_run_count": count,
        "completed_run_count": count,
        "run_evidence": evidence,
        "checks": {
            "collection": {
                "status": "PASS",
                "expected_count": count,
                "completed_count": count,
                "parameter_changed_count": count,
                "invalid_run_ids": [],
                "failure_codes": [],
            },
            "primary_gate": _task12_gate_result_fixture(primary_gate, source_stage),
            "additional_gate": (
                None
                if additional_gate is None
                else _task12_gate_result_fixture(additional_gate, source_stage)
            ),
        },
        "failure_codes": [],
        "policy": {
            "paper_eligible": False,
            "task_equal_aggregation": source_stage
            in {"screen128", "confirm128", "transfer256"},
            "fixed_seed_single_run_best_effort": True,
            "bitwise_reproducible": False,
            "metric_invalid_policy": "fail_closed",
        },
        "summarizer_source": {"path": str(tmp_path / "summarizer.py"), "sha256": "c" * 64},
    }
    report["semantic_sha256"] = canonical_hash(report)
    collector_rows = _collected_rows_for_report_evidence(
        evidence, changed=tuple(True for _ in evidence)
    )
    return report, source_identity, source_payload, prerequisite, collector_rows


@pytest.mark.parametrize(
    "source_stage",
    [
        "smoke128",
        "screen128",
        "confirm128",
        "cuda2step256",
        "resource256",
        "transfer256",
        "sentinel_repeat256",
    ],
)
def test_task12_report_accepts_only_the_frozen_seven_stage_gate_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_stage: str,
) -> None:
    """Break caught: Task 11 still expects stage aliases instead of gate kinds."""

    report, source_identity, source_payload, prerequisite, collector_rows = (
        _task12_gate_report_fixture(tmp_path, source_stage)
    )
    monkeypatch.setattr(
        runner_module, "_validate_current_summarizer_identity", lambda value: None
    )
    if prerequisite is not None:
        monkeypatch.setattr(
            runner_module,
            "_current_stage_identity",
            lambda *args, **kwargs: prerequisite,
        )
    monkeypatch.setattr(
        runner_module,
        "_task12_collect_stage_rows",
        lambda root, stage, *, source_identity: collector_rows,
        raising=False,
    )

    validated = runner_module._validate_task12_gate_report(
        report,
        source_stage=source_stage,
        next_stage=str(report["next_stage"]),
        source_identity=source_identity,
        source_payload=source_payload,
        output_root=(tmp_path / "runs").resolve(),
    )
    assert validated["gate"] == report["checks"]["primary_gate"]["gate"]


@pytest.mark.parametrize(
    ("gate", "source_stage"),
    [
        ("gradient_budget", "smoke128"),
        ("development_factorial", "screen128"),
        ("cuda_integration", "cuda2step256"),
        ("resource_feasibility", "resource256"),
        ("repeat_tolerance", "sentinel_repeat256"),
    ],
)
def test_each_task12_gate_rejects_nested_keys_fixed_values_and_type_coercion(
    gate: str,
    source_stage: str,
) -> None:
    """Break caught: a generic mapping shell stands in for exact gate schemas."""

    valid = _task12_gate_result_fixture(gate, source_stage)
    runner_module._validate_gate_result(
        valid,
        label="exact Task 12 child",
        require_pass=True,
        expected_gate=gate,
        source_stage=source_stage,
    )
    for mutation in (
        "criteria_key",
        "observed_key",
        "nested_key",
        "fixed_value",
        "exact_type",
    ):
        child = json.loads(json.dumps(valid))
        criteria = child["criteria"]
        observed = child["observed"]
        if mutation == "criteria_key":
            criteria["invented_criterion"] = 1
        elif mutation == "observed_key":
            observed["invented_observation"] = 0.0
        elif mutation == "nested_key":
            if gate == "gradient_budget":
                observed["endpoint_diagnostics"]["0.04"]["invented"] = 0.0
            elif gate == "development_factorial":
                observed["dic_conditional_task_equal_effects"]["invented"] = 0.0
            elif gate == "cuda_integration":
                observed["branch_checks"]["l12_bal"]["invented"] = True
            elif gate == "resource_feasibility":
                observed["peak_cuda_bytes_by_variant"]["invented"] = 1
            else:
                observed["paired_differences"]["l12_bal"]["invented"] = 0.0
        elif mutation == "fixed_value":
            fixed_key = {
                "gradient_budget": "ratio_min",
                "development_factorial": "dic_relative_threshold",
                "cuda_integration": "updates_per_run",
                "resource_feasibility": "warmup_updates",
                "repeat_tolerance": "tolerance",
            }[gate]
            criteria[fixed_key] = 999
        else:
            exact_int_key = {
                "gradient_budget": "resolution",
                "development_factorial": "task_count",
                "cuda_integration": "run_count",
                "resource_feasibility": "integration_run_count",
                "repeat_tolerance": "pair_count",
            }[gate]
            criteria[exact_int_key] = True
        with pytest.raises(
            ValueError, match="gate|criteria|observed|field|type|exact|schema|endpoint"
        ):
            runner_module._validate_gate_result(
                child,
                label="mutated Task 12 child",
                require_pass=True,
                expected_gate=gate,
                source_stage=source_stage,
            )


@pytest.mark.parametrize("child_gate", ["", "invented_gate", "smoke_gradient"])
def test_task12_child_rejects_empty_invented_and_stage_alias_gate_names(
    child_gate: str,
) -> None:
    child = _task12_gate_result_fixture("gradient_budget", "smoke128")
    child["gate"] = child_gate
    with pytest.raises(ValueError, match="gate|identity|expected|exact"):
        runner_module._validate_gate_result(
            child,
            label="mutated Task 12 child",
            require_pass=True,
            expected_gate="gradient_budget",
            source_stage="smoke128",
        )


def test_resource_report_rejects_wrong_additional_child_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: resource accepts any PASS-shaped additional child."""

    report, source_identity, source_payload, prerequisite, _ = (
        _task12_gate_report_fixture(tmp_path, "resource256")
    )
    assert prerequisite is not None
    report["checks"]["additional_gate"] = _task12_gate_result_fixture(
        "gradient_budget", "transfer256"
    )
    report["semantic_sha256"] = canonical_hash(
        {key: value for key, value in report.items() if key != "semantic_sha256"}
    )
    monkeypatch.setattr(
        runner_module, "_validate_current_summarizer_identity", lambda value: None
    )
    monkeypatch.setattr(
        runner_module,
        "_current_stage_identity",
        lambda *args, **kwargs: prerequisite,
    )
    with pytest.raises(ValueError, match="additional|gate|cuda|binding"):
        runner_module._validate_task12_gate_report(
            report,
            source_stage="resource256",
            next_stage="transfer256",
            source_identity=source_identity,
            source_payload=source_payload,
            output_root=(tmp_path / "runs").resolve(),
        )


@pytest.mark.parametrize(
    ("source_stage", "expected"),
    [
        ("smoke128", False),
        ("screen128", True),
        ("confirm128", True),
        ("cuda2step256", False),
        ("resource256", False),
        ("transfer256", True),
        ("sentinel_repeat256", False),
    ],
)
def test_task_equal_policy_is_bound_exactly_to_the_frozen_stage_set(
    source_stage: str,
    expected: bool,
) -> None:
    policy = {
        "paper_eligible": False,
        "task_equal_aggregation": expected,
        "fixed_seed_single_run_best_effort": True,
        "bitwise_reproducible": False,
        "metric_invalid_policy": "fail_closed",
    }
    runner_module._validate_task12_gate_policy(policy, source_stage=source_stage)
    policy["task_equal_aggregation"] = not expected
    with pytest.raises(ValueError, match="policy|task.equal|stage"):
        runner_module._validate_task12_gate_policy(policy, source_stage=source_stage)


def test_smoke_advance_recomputes_four_parameter_changes_from_bound_collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: smoke trusts a reported count without calling Task 12 collection."""

    report, source_identity, source_payload, _, collector_rows = (
        _task12_gate_report_fixture(tmp_path, "smoke128")
    )
    calls: list[tuple[Path, str]] = []

    def collect(
        root: Path, stage: str, *, source_identity: object
    ) -> list[dict[str, object]]:
        calls.append((root, stage))
        return collector_rows

    monkeypatch.setattr(
        runner_module, "_validate_current_summarizer_identity", lambda value: None
    )
    monkeypatch.setattr(
        runner_module, "_task12_collect_stage_rows", collect, raising=False
    )
    runner_module._validate_task12_gate_report(
        report,
        source_stage="smoke128",
        next_stage="screen128",
        source_identity=source_identity,
        source_payload=source_payload,
        output_root=(tmp_path / "runs").resolve(),
    )
    assert calls == [((tmp_path / "runs").resolve(), "smoke128")]


@pytest.mark.parametrize(
    "mutation",
    ["reported_count", "collector_count", "evidence_hash", "changed_type"],
)
def test_smoke_advance_rejects_nonfour_or_nonmatching_collector_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    report, source_identity, source_payload, _, collector_rows = (
        _task12_gate_report_fixture(tmp_path, "smoke128")
    )
    rows = json.loads(json.dumps(collector_rows))
    if mutation == "reported_count":
        report["checks"]["collection"]["parameter_changed_count"] = 3
    elif mutation == "collector_count":
        rows[-1]["parameter_change"]["changed"] = False
    elif mutation == "evidence_hash":
        rows[-1]["status"]["sha256"] = "f" * 64
    else:
        rows[-1]["parameter_change"]["changed"] = 1
    report["semantic_sha256"] = canonical_hash(
        {key: value for key, value in report.items() if key != "semantic_sha256"}
    )
    monkeypatch.setattr(
        runner_module, "_validate_current_summarizer_identity", lambda value: None
    )
    monkeypatch.setattr(
        runner_module,
        "_task12_collect_stage_rows",
        lambda root, stage, *, source_identity: rows,
        raising=False,
    )
    with pytest.raises(ValueError, match="parameter|collector|evidence|changed|smoke|hash"):
        runner_module._validate_task12_gate_report(
            report,
            source_stage="smoke128",
            next_stage="screen128",
            source_identity=source_identity,
            source_payload=source_payload,
            output_root=(tmp_path / "runs").resolve(),
        )


def test_smoke_collector_executes_the_exact_report_bound_source_not_cached_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: a stale cached function is trusted because __file__ still matches."""

    report, source_identity, source_payload, _, collector_rows = (
        _task12_gate_report_fixture(tmp_path, "smoke128")
    )
    summarizer = tmp_path / "summarize_soft_loss_v3.py"
    summarizer.write_text(
        "def collect_stage_rows(root, stage):\n"
        f"    return {collector_rows!r}\n",
        encoding="utf-8",
    )
    report["summarizer_source"] = _record(summarizer)
    report["semantic_sha256"] = canonical_hash(
        {key: value for key, value in report.items() if key != "semantic_sha256"}
    )
    cached_calls: list[tuple[Path, str]] = []

    def cached_collect(root: Path, stage: str) -> list[dict[str, object]]:
        cached_calls.append((root, stage))
        return collector_rows

    cached_module = types.SimpleNamespace(
        __file__=str(summarizer),
        collect_stage_rows=cached_collect,
    )
    monkeypatch.setattr(runner_module, "_SUMMARIZER_SOURCE", summarizer.resolve())
    monkeypatch.setattr(
        runner_module.importlib,
        "import_module",
        lambda name: cached_module,
    )

    runner_module._validate_task12_gate_report(
        report,
        source_stage="smoke128",
        next_stage="screen128",
        source_identity=source_identity,
        source_payload=source_payload,
        output_root=(tmp_path / "runs").resolve(),
    )
    assert cached_calls == []


def test_smoke_collector_source_swap_during_call_fails_stable_postcheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: collector bytes can change after their one-time identity check."""

    report, source_identity, source_payload, _, collector_rows = (
        _task12_gate_report_fixture(tmp_path, "smoke128")
    )
    summarizer = tmp_path / "summarize_soft_loss_v3.py"
    summarizer.write_text(
        "from pathlib import Path\n"
        f"ROWS = {collector_rows!r}\n"
        "def collect_stage_rows(root, stage):\n"
        "    Path(__file__).write_bytes(b'# swapped collector source\\n')\n"
        "    return ROWS\n",
        encoding="utf-8",
    )
    report["summarizer_source"] = _record(summarizer)
    report["semantic_sha256"] = canonical_hash(
        {key: value for key, value in report.items() if key != "semantic_sha256"}
    )

    def cached_collect(root: Path, stage: str) -> list[dict[str, object]]:
        summarizer.write_bytes(b"# swapped collector source\n")
        return collector_rows

    cached_module = types.SimpleNamespace(
        __file__=str(summarizer),
        collect_stage_rows=cached_collect,
    )
    monkeypatch.setattr(runner_module, "_SUMMARIZER_SOURCE", summarizer.resolve())
    monkeypatch.setattr(
        runner_module.importlib,
        "import_module",
        lambda name: cached_module,
    )

    with pytest.raises(ValueError, match="summarizer|collector|source|changed|snapshot|stable"):
        runner_module._validate_task12_gate_report(
            report,
            source_stage="smoke128",
            next_stage="screen128",
            source_identity=source_identity,
            source_payload=source_payload,
            output_root=(tmp_path / "runs").resolve(),
        )


def _make_authorized_screen_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[StageContext, Path, dict[str, object]]:
    paths = _make_bound_registry(tmp_path, monkeypatch)
    output = paths["output"]
    weight, weight_payload = _task10_weight_fixture(paths, monkeypatch)
    weight_semantic = weight_payload["semantic_sha256"]
    source_specs = build_stage_specs(
        "smoke128",
        StageContext(
            output_root=output.resolve(),
            master_manifest=paths["master"].resolve(),
            development_suite=paths["development"].resolve(),
            formal_suite=paths["formal"].resolve(),
            frozen_weights=weight.resolve(),
            authorization=None,
            formal_campaign=None,
        ),
    )
    source_stage = output / "manifests" / "stages" / "smoke128.json"
    source_stage_payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_stage_v1",
        "status": "APPROVED_FROZEN",
        "stage": "smoke128",
        "authorization": None,
        "formal_campaign": None,
        "run_count": 4,
        "runs": [
            {
                "run_id": spec.run_id,
                "path": str(spec.output_path),
                "run_spec": runner_module._run_spec_payload(spec),
                "scientific_config_hash": scientific_config_hash(spec),
                "expected_config_hash": expected_config_hash(spec),
                "expected_artifacts": runner_module._expected_artifact_namespace(spec),
            }
            for index, spec in enumerate(source_specs)
        ],
    }
    source_stage_payload["semantic_sha256"] = canonical_hash(source_stage_payload)
    source_stage.parent.mkdir(parents=True, exist_ok=True)
    source_stage.write_bytes(runner_module.canonical_json_bytes(source_stage_payload))
    report = output / "analysis" / "smoke128_gate.json"
    summarizer = tmp_path / "authoritative-summarize-soft-loss-v3.py"
    summarizer.write_bytes(b"# tmp-only Task 12 summarizer identity seam\n")
    monkeypatch.setattr(
        runner_module, "_SUMMARIZER_SOURCE", summarizer.resolve(), raising=False
    )
    source_identity = _record(
        source_stage, semantic=source_stage_payload["semantic_sha256"]
    )
    evidence = [
        {
            "run_id": row["run_id"],
            "path": row["path"],
            "expected_config_hash": row["expected_config_hash"],
            "completed": True,
            "status_sha256": "8" * 64,
            "artifact_manifest_sha256": "9" * 64,
            "metrics_sha256": "a" * 64,
            "gradient_audit_sha256": "b" * 64,
            "resource_metrics_sha256": "c" * 64,
            "history_sha256": "d" * 64,
            "occupancy_probability_sha256": "e" * 64,
        }
        for row in source_stage_payload["runs"]
    ]
    report_payload = {
        "schema": "pcraft_soft_loss_v3_gate_report_v1",
        "status": "ADVANCE",
        "source_stage": "smoke128",
        "next_stage": "screen128",
        "gate": "gradient_budget",
        "source_stage_manifest": source_identity,
        "prerequisite_stage_manifests": [],
        "formal_campaign": None,
        "registered_run_count": 4,
        "completed_run_count": 4,
        "run_evidence": evidence,
        "checks": {
            "collection": {
                "status": "PASS",
                "expected_count": 4,
                "completed_count": 4,
                "parameter_changed_count": 4,
                "invalid_run_ids": [],
                "failure_codes": [],
            },
            "primary_gate": {
                "schema": "pcraft_soft_loss_v3_gate_result_v1",
                "gate": "gradient_budget",
                "status": "PASS",
                "criteria": {
                    "resolution": 128,
                    "ratio_min": 0.10,
                    "ratio_max": 0.20,
                    "combined_median_max": 0.35,
                    "combined_p95_max": 1.0,
                    "adverse_cosine_threshold": -0.5,
                    "adverse_cosine_fraction_max": 0.10,
                    "component_active_l2_epsilon": 1e-12,
                    "support_abs_epsilon": 1e-12,
                    "eligible_inactive_policy": "stop_any_eligible_inactive_sample",
                },
                "observed": {
                    "expected_probe_count": 8,
                    "observed_probe_count": 8,
                    "dic_ratio_median": 0.15,
                    "dts_ratio_median": 0.15,
                    "combined_ratio_median": 0.30,
                    "combined_ratio_p95": 0.30,
                    "adverse_cosine_fraction": 0.0,
                    "endpoint_diagnostics": {
                        "0.04": {
                            "joint_sample_count": 1,
                            "dic_dts_cosine_median": 0.0,
                            "dic_dts_support_iou_median": 0.5,
                        },
                        "0.06": {
                            "joint_sample_count": 1,
                            "dic_dts_cosine_median": 0.0,
                            "dic_dts_support_iou_median": 0.5,
                        },
                    },
                },
                "failure_codes": [],
            },
            "additional_gate": None,
        },
        "failure_codes": [],
        "policy": {
            "paper_eligible": False,
            "task_equal_aggregation": False,
            "fixed_seed_single_run_best_effort": True,
            "bitwise_reproducible": False,
            "metric_invalid_policy": "fail_closed",
        },
        "summarizer_source": _record(summarizer),
    }
    report_payload["semantic_sha256"] = canonical_hash(report_payload)
    report.parent.mkdir(exist_ok=True)
    report.write_bytes(runner_module.canonical_json_bytes(report_payload))
    collector_rows = _collected_rows_for_report_evidence(
        evidence, changed=(True, True, True, True)
    )
    monkeypatch.setattr(
        runner_module,
        "_task12_collect_stage_rows",
        lambda root, stage, *, source_identity: collector_rows,
        raising=False,
    )
    master = json.loads(paths["master"].read_bytes())
    development = json.loads(paths["development"].read_bytes())
    formal = json.loads(paths["formal"].read_bytes())
    authorization: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_authorization_v1",
        "status": "ADVANCE",
        "source_stage": "smoke128",
        "next_stage": "screen128",
        "report": {
            **_record(report),
            "schema": "pcraft_soft_loss_v3_gate_report_v1",
        },
        "dependencies": {
            "master_manifest": _record(paths["master"], semantic=master["semantic_sha256"]),
            "development_suite": _record(paths["development"], semantic=development["semantic_sha256"]),
            "formal_suite": _record(paths["formal"], semantic=formal["semantic_sha256"]),
            "specification": _record(paths["spec"]),
            "frozen_weights": _record(weight, semantic=weight_semantic),
            "source_stage_manifest": source_identity,
            "formal_campaign": None,
        },
        "completion": None,
    }
    authorization["semantic_sha256"] = canonical_hash(authorization)
    auth_path = output / "authorizations" / "smoke128__to__screen128.json"
    auth_path.parent.mkdir()
    auth_path.write_bytes(runner_module.canonical_json_bytes(authorization))
    context = StageContext(
        output_root=output.resolve(),
        master_manifest=paths["master"].resolve(),
        development_suite=paths["development"].resolve(),
        formal_suite=paths["formal"].resolve(),
        frozen_weights=weight.resolve(),
        authorization=auth_path.resolve(),
        formal_campaign=None,
    )
    return context, auth_path, authorization


def _rewrite_authorization(path: Path, payload: dict[str, object]) -> None:
    payload["semantic_sha256"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "semantic_sha256"}
    )
    path.write_bytes(runner_module.canonical_json_bytes(payload))


def test_advance_authorization_validates_exact_edge_and_transitive_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a stage advances without the exact current transitive evidence chain."""

    context, _, _ = _make_authorized_screen_context(tmp_path, monkeypatch)
    specs = build_stage_specs("screen128", context)
    assert len(specs) == 12
    assert not context.output_root.joinpath("screen128").exists()


def test_source_stage_recomputes_each_registered_task8_config_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a coherently rehashed false expected-config row authorizes downstream."""

    context, authorization_path, authorization = _make_authorized_screen_context(
        tmp_path, monkeypatch
    )
    source_path = Path(
        authorization["dependencies"]["source_stage_manifest"]["path"]
    )
    source = json.loads(source_path.read_bytes())
    source["runs"][0]["expected_config_hash"] = "0" * 64
    source["semantic_sha256"] = canonical_hash(
        {key: value for key, value in source.items() if key != "semantic_sha256"}
    )
    source_path.write_bytes(runner_module.canonical_json_bytes(source))
    authorization["dependencies"]["source_stage_manifest"] = _record(
        source_path, semantic=source["semantic_sha256"]
    )
    _rewrite_authorization(authorization_path, authorization)

    with pytest.raises(ValueError, match="config|Task 8|resolved|registered|source stage"):
        build_stage_specs("screen128", context)


@pytest.mark.parametrize("mutation", ["path", "status", "source", "next", "schema"])
def test_authorization_report_has_exact_analysis_path_schema_status_and_edge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: an arbitrary/STOP report can be relabeled as an ADVANCE wrapper."""

    context, auth_path, authorization = _make_authorized_screen_context(
        tmp_path, monkeypatch
    )
    report_path = Path(authorization["report"]["path"])
    report = json.loads(report_path.read_bytes())
    if mutation == "path":
        alternate = context.output_root / "analysis" / "arbitrary.json"
        alternate.write_bytes(report_path.read_bytes())
        authorization["report"] = {
            **_record(alternate),
            "schema": "pcraft_soft_loss_v3_gate_report_v1",
        }
    else:
        if mutation == "status":
            report["status"] = "STOP"
        elif mutation == "source":
            report["source_stage"] = "confirm128"
        elif mutation == "next":
            report["next_stage"] = "confirm128"
        else:
            report["schema"] = "pcraft_soft_loss_v3_completion_report_v1"
            authorization["report"]["schema"] = report["schema"]
        report_path.write_bytes(runner_module.canonical_json_bytes(report))
        authorization["report"]["sha256"] = _sha(report_path)
    _rewrite_authorization(auth_path, authorization)
    with pytest.raises(ValueError, match="report|path|schema|status|edge|ADVANCE"):
        build_stage_specs("screen128", context)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "self_hashed_shell",
        "summarizer_drift",
        "source_identity",
        "count",
        "failure",
        "collection_stop",
        "child_stop",
    ],
)
def test_advance_requires_exact_task12_gate_report_and_current_summarizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: an incomplete/self-hashed Task 12 report shell advances a stage."""

    context, authorization_path, authorization = _make_authorized_screen_context(
        tmp_path, monkeypatch
    )
    report_path = Path(authorization["report"]["path"])
    report = json.loads(report_path.read_bytes())
    if mutation == "extra":
        report["selected_candidate"] = "v3_full"
    elif mutation == "missing":
        report.pop("policy")
    elif mutation == "self_hashed_shell":
        report = {
            "schema": "pcraft_soft_loss_v3_gate_report_v1",
            "status": "ADVANCE",
            "source_stage": "smoke128",
            "next_stage": "screen128",
        }
    elif mutation == "summarizer_drift":
        Path(report["summarizer_source"]["path"]).write_bytes(b"drifted summarizer\n")
    elif mutation == "source_identity":
        report["source_stage_manifest"]["sha256"] = "0" * 64
    elif mutation == "count":
        report["completed_run_count"] = 3
    elif mutation == "failure":
        report["failure_codes"] = ["UNAPPROVED_PASS"]
    elif mutation == "collection_stop":
        report["checks"]["collection"]["status"] = "STOP"
    else:
        report["checks"]["primary_gate"]["status"] = "STOP"
    if mutation != "summarizer_drift":
        report["semantic_sha256"] = canonical_hash(
            {key: value for key, value in report.items() if key != "semantic_sha256"}
        )
        report_path.write_bytes(runner_module.canonical_json_bytes(report))
        authorization["report"]["sha256"] = _sha(report_path)
        _rewrite_authorization(authorization_path, authorization)

    with pytest.raises(
        ValueError,
        match="report|summarizer|source|count|failure|collection|gate|schema|field|semantic",
    ):
        build_stage_specs("screen128", context)


@pytest.mark.parametrize("mutation", ["report", "dependency"])
def test_stage_publication_revalidates_full_authorization_after_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: a coherently rehashed gate replacement after build still publishes."""

    context, auth_path, authorization = _make_authorized_screen_context(
        tmp_path, monkeypatch
    )
    specs = build_stage_specs("screen128", context)
    if mutation == "report":
        report_path = Path(authorization["report"]["path"])
        report = json.loads(report_path.read_bytes())
        report["status"] = "STOP"
        report_path.write_bytes(runner_module.canonical_json_bytes(report))
        authorization["report"]["sha256"] = _sha(report_path)
    else:
        authorization["dependencies"]["master_manifest"]["sha256"] = "0" * 64
    _rewrite_authorization(auth_path, authorization)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: "9" * 64)
    with pytest.raises(ValueError, match="authorization|report|dependency|master|stale|changed"):
        write_stage_manifest("screen128", specs, context.output_root)
    assert not context.output_root.joinpath(
        "manifests", "stages", "screen128.json"
    ).exists()


def test_stage_publication_revalidates_the_full_transitive_authorization_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: an earlier DAG authorization drifts after a later stage is built."""

    screen_context, prior_auth_path, prior_authorization = (
        _make_authorized_screen_context(tmp_path, monkeypatch)
    )
    screen_specs = build_stage_specs("screen128", screen_context)
    screen_stage = write_stage_manifest(
        "screen128", screen_specs, screen_context.output_root
    )
    screen_stage_payload = json.loads(screen_stage.read_bytes())
    report = screen_context.output_root / "analysis" / "screen128_gate.json"
    report_payload = json.loads(
        Path(prior_authorization["report"]["path"]).read_bytes()
    )
    report_payload.update(
        {
            "source_stage": "screen128",
            "next_stage": "confirm128",
            "gate": "development_factorial",
            "source_stage_manifest": _record(
                screen_stage,
                semantic=screen_stage_payload["semantic_sha256"],
            ),
            "registered_run_count": 12,
            "completed_run_count": 12,
            "run_evidence": [
                {
                    "run_id": row["run_id"],
                    "path": row["path"],
                    "expected_config_hash": row["expected_config_hash"],
                    "completed": True,
                    "status_sha256": "8" * 64,
                    "artifact_manifest_sha256": "9" * 64,
                    "metrics_sha256": "a" * 64,
                    "gradient_audit_sha256": "b" * 64,
                    "resource_metrics_sha256": "c" * 64,
                    "history_sha256": "d" * 64,
                    "occupancy_probability_sha256": "e" * 64,
                }
                for row in screen_stage_payload["runs"]
            ],
        }
    )
    report_payload["checks"]["collection"].update(
        {
            "expected_count": 12,
            "completed_count": 12,
            "parameter_changed_count": 12,
        }
    )
    report_payload["checks"]["primary_gate"] = _task12_gate_result_fixture(
        "development_factorial", "screen128"
    )
    report_payload["policy"]["task_equal_aggregation"] = True
    report_payload["semantic_sha256"] = canonical_hash(
        {
            key: value
            for key, value in report_payload.items()
            if key != "semantic_sha256"
        }
    )
    report.write_bytes(runner_module.canonical_json_bytes(report_payload))
    authorization: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_authorization_v1",
        "status": "ADVANCE",
        "source_stage": "screen128",
        "next_stage": "confirm128",
        "report": {
            **_record(report),
            "schema": "pcraft_soft_loss_v3_gate_report_v1",
        },
        "dependencies": {
            **prior_authorization["dependencies"],
            "source_stage_manifest": _record(
                screen_stage,
                semantic=screen_stage_payload["semantic_sha256"],
            ),
        },
        "completion": None,
    }
    authorization["semantic_sha256"] = canonical_hash(authorization)
    current_auth = (
        screen_context.output_root
        / "authorizations"
        / "screen128__to__confirm128.json"
    )
    current_auth.write_bytes(runner_module.canonical_json_bytes(authorization))
    confirm_context = replace(screen_context, authorization=current_auth)
    confirm_specs = build_stage_specs("confirm128", confirm_context)

    prior_authorization["dependencies"]["master_manifest"]["sha256"] = "0" * 64
    _rewrite_authorization(prior_auth_path, prior_authorization)
    with pytest.raises(ValueError, match="authorization|transitive|master|stale|changed"):
        write_stage_manifest(
            "confirm128", confirm_specs, confirm_context.output_root
        )
    assert not confirm_context.output_root.joinpath(
        "manifests", "stages", "confirm128.json"
    ).exists()


@pytest.mark.parametrize(
    "mutation",
    ["stop", "wrong_source", "wrong_next", "completion", "master_hash", "source_drift", "report_drift"],
)
def test_authorization_stop_wrong_edge_and_stale_dependencies_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: STOP, wrong-next-stage, or stale evidence authorizes a downstream stage."""

    context, auth_path, payload = _make_authorized_screen_context(tmp_path, monkeypatch)
    if mutation == "stop":
        payload["status"] = "STOP"
    elif mutation == "wrong_source":
        payload["source_stage"] = "confirm128"
    elif mutation == "wrong_next":
        payload["next_stage"] = "confirm128"
    elif mutation == "completion":
        payload["completion"] = []
    elif mutation == "master_hash":
        payload["dependencies"]["master_manifest"]["sha256"] = "0" * 64
    elif mutation == "source_drift":
        source = Path(payload["dependencies"]["source_stage_manifest"]["path"])
        source.write_bytes(source.read_bytes() + b"drift")
    else:
        report = Path(payload["report"]["path"])
        report.write_bytes(report.read_bytes() + b"drift")
    if mutation not in {"source_drift", "report_drift"}:
        _rewrite_authorization(auth_path, payload)
    with pytest.raises(ValueError, match="authorization|ADVANCE|STOP|source|next|completion|master|report|hash|changed|stale"):
        build_stage_specs("screen128", context)
    assert not context.output_root.joinpath("screen128").exists()


def test_gated_stage_requires_authorization_and_rejects_irrelevant_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: gated stages default missing evidence or accept outcome-changing extras."""

    context, _, _ = _make_authorized_screen_context(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="authorization|required"):
        build_stage_specs("screen128", replace(context, authorization=None))
    campaign = tmp_path / "campaign.json"
    campaign.write_text("irrelevant", encoding="utf-8")
    with pytest.raises(ValueError, match="campaign|irrelevant"):
        build_stage_specs("screen128", replace(context, formal_campaign=campaign.resolve()))


def test_source_stage_validator_rejects_semantically_rehashed_extra_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a non-Task11 source-stage shape is accepted when self-rehashed."""

    context, _, authorization = _make_authorized_screen_context(tmp_path, monkeypatch)
    source_path = Path(
        authorization["dependencies"]["source_stage_manifest"]["path"]
    )
    source = json.loads(source_path.read_bytes())
    source["outcome_selected_rows"] = ["not-a-registration-field"]
    source["semantic_sha256"] = canonical_hash(
        {key: value for key, value in source.items() if key != "semantic_sha256"}
    )
    source_path.write_bytes(runner_module.canonical_json_bytes(source))
    record = _record(source_path, semantic=source["semantic_sha256"])
    with pytest.raises(ValueError, match="source stage|schema|field|exact|registration"):
        runner_module._validate_source_stage_identity(
            record,
            output_root=context.output_root,
            source_stage="smoke128",
        )


def test_appendix_authorization_delegates_to_strict_locked_conclusion_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: arbitrary/failed conclusion JSON starts exploratory appendix runs."""

    fixture = _identity_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda *_: fixture)
    monkeypatch.setattr(runner_module, "_verify_registered_inputs", lambda spec: None)
    context_for_specs = _context(tmp_path, calibration=False)
    primary = build_stage_specs("formal256_primary", context_for_specs)
    repeats = build_stage_specs("formal256_repeat", context_for_specs)
    output = context_for_specs.output_root
    campaign = output / "manifests" / "formal_campaign.json"
    write_formal_campaign_manifest(primary, repeats, campaign)
    analysis = output / "analysis" / "analysis_manifest.json"
    analysis.parent.mkdir()
    analysis.write_bytes(b"current analysis manifest")
    conclusion = output / "analysis" / "formal_conclusion.json"
    conclusion.write_bytes(b"Task13-owned conclusion bytes")
    status = {"value": "LOCKED"}
    calls: list[tuple[Path, str, str]] = []

    def validate(path: Path, *, expected_analysis_sha256: str, expected_campaign_sha256: str) -> dict[str, object]:
        calls.append((path, expected_analysis_sha256, expected_campaign_sha256))
        return {
            "schema": "pcraft_soft_loss_v3_conclusion_v1",
            "status": status["value"],
            "semantic_sha256": "c" * 64,
        }

    monkeypatch.setitem(
        sys.modules,
        "experiments.soft_loss_v3_statistics",
        types.SimpleNamespace(validate_formal_conclusion=validate),
    )
    appendix_context = StageContext(
        output_root=output,
        master_manifest=tmp_path / "master",
        development_suite=tmp_path / "development",
        formal_suite=tmp_path / "formal",
        frozen_weights=tmp_path / "weights",
        authorization=conclusion,
        formal_campaign=None,
    )
    result = runner_module._validate_authorization(
        "appendix256",
        appendix_context,
        output_root=output,
        master_identity={},
        development_identity={},
        formal_identity={},
        specification_identity={},
        weight_identity={},
        campaign_identity=None,
    )
    assert result is not None and result["status"] == "LOCKED"
    assert calls == [(conclusion, _sha(analysis), _sha(campaign))]
    status["value"] = "FAILED_TECHNICAL_AUDIT"
    with pytest.raises(ValueError, match="LOCKED|conclusion|appendix"):
        runner_module._validate_authorization(
            "appendix256",
            appendix_context,
            output_root=output,
            master_identity={},
            development_identity={},
            formal_identity={},
            specification_identity={},
            weight_identity={},
            campaign_identity=None,
        )


def _completion_only_fixture(
    tmp_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    runs: list[dict[str, object]] = []
    completion: list[dict[str, object]] = []
    for index in range(40):
        run_dir = tmp_path / "formal256_primary" / f"run_{index:02d}"
        run_dir.mkdir(parents=True)
        status = run_dir / "STATUS.json"
        artifact = run_dir / "artifact_manifest.json"
        status.write_bytes(b"current completed status")
        artifact.write_bytes(b"current artifact manifest")
        run_id = f"formal256_primary/run_{index:02d}"
        config_hash = f"{index:064x}"
        runs.append(
            {
                "run_id": run_id,
                "path": str(run_dir.resolve()),
                "expected_config_hash": config_hash,
            }
        )
        completion.append(
            {
                "run_id": run_id,
                "path": str(run_dir.resolve()),
                "expected_config_hash": config_hash,
                "status_sha256": _sha(status),
                "artifact_manifest_sha256": _sha(artifact),
            }
        )
    return completion, {
        "schema": "pcraft_soft_loss_v3_stage_v1",
        "status": "APPROVED_FROZEN",
        "stage": "formal256_primary",
        "run_count": 40,
        "runs": runs,
    }


def _exact_completion_campaign_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object], Path]:
    fixture = _identity_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "_validated_stage_context", lambda *_: fixture)
    context = _context(tmp_path, calibration=False)
    specs = build_stage_specs("formal256_primary", context)
    completion: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        spec.output_path.mkdir(parents=True)
        status = spec.output_path / "STATUS.json"
        artifact = spec.output_path / "artifact_manifest.json"
        status.write_bytes(f"completed:{index}".encode("ascii"))
        artifact.write_bytes(f"artifact:{index}".encode("ascii"))
        config_hash = f"{index + 1:064x}"
        runs.append(
            {
                "run_id": spec.run_id,
                "path": str(spec.output_path),
                "run_spec": runner_module._run_spec_payload(spec),
                "scientific_config_hash": scientific_config_hash(spec),
                "expected_config_hash": config_hash,
                "expected_artifacts": runner_module._expected_artifact_namespace(spec),
            }
        )
        completion.append(
            {
                "run_id": spec.run_id,
                "path": str(spec.output_path),
                "expected_config_hash": config_hash,
                "status_sha256": _sha(status),
                "artifact_manifest_sha256": _sha(artifact),
            }
        )
    stage: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_stage_v1",
        "status": "APPROVED_FROZEN",
        "stage": "formal256_primary",
        "authorization": {"bound": True},
        "formal_campaign": {"bound": True},
        "run_count": 40,
        "runs": runs,
    }
    stage["semantic_sha256"] = canonical_hash(stage)
    campaign: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_formal_campaign_v1",
        "status": "APPROVED_FROZEN",
        "primary": [runner_module._campaign_cell(spec) for spec in specs],
        "repeats": [],
        "pairs": [],
    }
    campaign["semantic_sha256"] = canonical_hash(campaign)
    return completion, stage, campaign, context.output_root


@pytest.mark.parametrize(
    "mutation",
    ["campaign_order", "source_run_shape", "campaign_run_spec", "path_escape"],
)
def test_completion_only_is_the_exact_campaign_primary_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: any unrelated 40 completed rows can authorize the formal repeat."""

    completion, stage, campaign, output_root = _exact_completion_campaign_fixture(
        tmp_path, monkeypatch
    )
    if mutation == "campaign_order":
        campaign["primary"][0], campaign["primary"][1] = (
            campaign["primary"][1],
            campaign["primary"][0],
        )
    elif mutation == "source_run_shape":
        stage["runs"][0].pop("run_spec")
    elif mutation == "campaign_run_spec":
        campaign["primary"][0]["run_spec"]["variant"] = "dic_only"
    else:
        outside = tmp_path / "outside" / "formal-cell"
        outside.mkdir(parents=True)
        outside.joinpath("STATUS.json").write_bytes(b"completed:0")
        outside.joinpath("artifact_manifest.json").write_bytes(b"artifact:0")
        outside_text = str(outside.resolve())
        completion[0]["path"] = outside_text
        stage["runs"][0]["path"] = outside_text
        stage["runs"][0]["run_spec"]["output_path"] = outside_text
        campaign["primary"][0]["path"] = outside_text
        campaign["primary"][0]["run_spec"]["output_path"] = outside_text
    monkeypatch.setattr(
        runner_module,
        "_validate_completion_only_run",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="completion|campaign|source|run|path|output|registered"):
        runner_module._validate_completion_only(
            completion,
            stage,
            campaign,
            output_root=output_root,
        )


def test_completion_only_aggregate_is_structural_and_does_not_revalidate_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the aggregate helper internally repeats all 40 live checks."""

    completion, stage, campaign, output_root = _exact_completion_campaign_fixture(
        tmp_path, monkeypatch
    )
    calls: list[tuple[Path, str, int]] = []
    monkeypatch.setattr(trainer_module, "_preflight_reserved_top_level_paths", lambda path: None)
    monkeypatch.setattr(
        trainer_module,
        "_status",
        lambda path: {
            "schema": "pcraft_soft_loss_v3_status_v1",
            "state": "completed",
            "config_hash": completion[int(path.parent.name[-2:])]["expected_config_hash"],
            "completed_updates": 2000,
            "artifact_manifest_sha256": completion[int(path.parent.name[-2:])]["artifact_manifest_sha256"],
        },
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_completion_only_run",
        lambda path, config_hash, total_steps, status_sha256, artifact_manifest_sha256, *, output_root: calls.append(
            (path, config_hash, total_steps)
        ),
    )
    runner_module._validate_completion_only(
        completion, stage, campaign, output_root=output_root
    )
    assert calls == []
    assert all("metric" not in set(record) for record in completion)


def test_formal_repeat_authorization_invokes_completion_only_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: COMPLETE_ONLY is schema-checked but its 40 runs are never validated."""

    output = tmp_path / "runs"
    output.mkdir()
    completion, source_payload = _completion_only_fixture(tmp_path)
    source_payload["semantic_sha256"] = canonical_hash(source_payload)
    source = output / "manifests" / "stages" / "formal256_primary.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(runner_module.canonical_json_bytes(source_payload))
    report = output / "analysis" / "formal_primary_completion.json"
    report.parent.mkdir()
    summarizer = tmp_path / "authoritative-formal-summarizer.py"
    summarizer.write_bytes(b"# tmp-only Task 12 completion identity seam\n")
    monkeypatch.setattr(
        runner_module, "_SUMMARIZER_SOURCE", summarizer.resolve(), raising=False
    )
    source_identity = _record(
        source, semantic=source_payload["semantic_sha256"]
    )
    report_payload = {
        "schema": "pcraft_soft_loss_v3_completion_report_v1",
        "status": "COMPLETE_ONLY",
        "source_stage": "formal256_primary",
        "next_stage": "formal256_repeat",
        "source_stage_manifest": source_identity,
        "formal_campaign": None,
        "registered_run_count": 40,
        "completed_run_count": 40,
        "completion": completion,
        "metric_values_parsed": False,
        "failure_codes": [],
        "policy": {
            "completion_only": True,
            "primary_outcomes_opaque": True,
            "repeats_inference_eligible": False,
        },
        "summarizer_source": _record(summarizer),
    }
    master = {"path": str((tmp_path / "master.json").resolve()), "sha256": "1" * 64}
    development = {"path": str((tmp_path / "development.json").resolve()), "sha256": "2" * 64}
    formal = {"path": str((tmp_path / "formal.json").resolve()), "sha256": "3" * 64}
    spec = {"path": str((tmp_path / "spec.md").resolve()), "sha256": "4" * 64}
    weight = {"path": str((tmp_path / "weights.json").resolve()), "sha256": "5" * 64}
    campaign = {
        "path": str((output / "manifests" / "formal_campaign.json").resolve()),
        "sha256": "6" * 64,
        "semantic_sha256": "7" * 64,
    }
    report_payload["formal_campaign"] = campaign
    report_payload["semantic_sha256"] = canonical_hash(report_payload)
    report.write_bytes(runner_module.canonical_json_bytes(report_payload))
    authorization: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_authorization_v1",
        "status": "COMPLETE_ONLY",
        "source_stage": "formal256_primary",
        "next_stage": "formal256_repeat",
        "report": {**_record(report), "schema": report_payload["schema"]},
        "dependencies": {
            "master_manifest": master,
            "development_suite": development,
            "formal_suite": formal,
            "specification": spec,
            "frozen_weights": weight,
            "source_stage_manifest": source_identity,
            "formal_campaign": campaign,
        },
        "completion": completion,
    }
    authorization["semantic_sha256"] = canonical_hash(authorization)
    auth = output / "authorizations" / "formal256_primary__to__formal256_repeat.json"
    auth.parent.mkdir()
    auth.write_bytes(runner_module.canonical_json_bytes(authorization))
    observed: list[tuple[object, object]] = []
    per_run_calls: list[tuple[Path, str, int, str, str, Path]] = []
    monkeypatch.setattr(
        runner_module,
        "_validate_completion_only_run",
        lambda path, config_hash, total_steps, status_sha256, artifact_manifest_sha256, *, output_root: per_run_calls.append(
            (
                path,
                config_hash,
                total_steps,
                status_sha256,
                artifact_manifest_sha256,
                output_root,
            )
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_completion_only",
        lambda values, stage, campaign, *, output_root: observed.append(
            (values, stage)
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_source_stage_identity",
        lambda *args, **kwargs: (
            dict(authorization["dependencies"]["source_stage_manifest"]),
            source_payload,
        ),
    )
    context = StageContext(
        output_root=output.resolve(),
        master_manifest=Path(master["path"]),
        development_suite=Path(development["path"]),
        formal_suite=Path(formal["path"]),
        frozen_weights=Path(weight["path"]),
        authorization=auth.resolve(),
        formal_campaign=Path(campaign["path"]),
    )
    result = runner_module._validate_authorization(
        "formal256_repeat",
        context,
        output_root=output.resolve(),
        master_identity=master,
        development_identity=development,
        formal_identity=formal,
        specification_identity=spec,
        weight_identity=weight,
        campaign_identity=campaign,
        campaign_payload={"primary": []},
    )
    assert result is not None and result["status"] == "COMPLETE_ONLY"
    assert observed == [(completion, source_payload)]
    assert per_run_calls == [
        (
            Path(record["path"]),
            record["expected_config_hash"],
            2000,
            record["status_sha256"],
            record["artifact_manifest_sha256"],
            output.resolve(),
        )
        for record in completion
    ]


def test_complete_only_report_completion_is_exactly_authorization_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: report and authorization bind different 40-cell completion lists."""

    output = tmp_path / "runs"
    output.mkdir()
    completion, source_payload = _completion_only_fixture(tmp_path)
    source_payload["semantic_sha256"] = canonical_hash(source_payload)
    source = output / "manifests" / "stages" / "formal256_primary.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(runner_module.canonical_json_bytes(source_payload))
    campaign_path = output / "manifests" / "formal_campaign.json"
    campaign_path.write_bytes(b"current campaign")
    campaign = {
        "path": str(campaign_path.resolve()),
        "sha256": _sha(campaign_path),
        "semantic_sha256": "7" * 64,
    }
    summarizer = tmp_path / "authoritative-completion-summarizer.py"
    summarizer.write_bytes(b"# tmp-only Task 12 completion identity seam\n")
    monkeypatch.setattr(
        runner_module, "_SUMMARIZER_SOURCE", summarizer.resolve(), raising=False
    )
    source_identity = _record(source, semantic=source_payload["semantic_sha256"])
    report_payload: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_completion_report_v1",
        "status": "COMPLETE_ONLY",
        "source_stage": "formal256_primary",
        "next_stage": "formal256_repeat",
        "source_stage_manifest": source_identity,
        "formal_campaign": campaign,
        "registered_run_count": 40,
        "completed_run_count": 40,
        "completion": completion,
        "metric_values_parsed": False,
        "failure_codes": [],
        "policy": {
            "completion_only": True,
            "primary_outcomes_opaque": True,
            "repeats_inference_eligible": False,
        },
        "summarizer_source": _record(summarizer),
    }
    report_payload["semantic_sha256"] = canonical_hash(report_payload)
    report = output / "analysis" / "formal_primary_completion.json"
    report.parent.mkdir()
    report.write_bytes(runner_module.canonical_json_bytes(report_payload))
    identities = {
        "master_manifest": {"path": str(tmp_path / "master"), "sha256": "1" * 64},
        "development_suite": {"path": str(tmp_path / "development"), "sha256": "2" * 64},
        "formal_suite": {"path": str(tmp_path / "formal"), "sha256": "3" * 64},
        "specification": {"path": str(tmp_path / "spec"), "sha256": "4" * 64},
        "frozen_weights": {"path": str(tmp_path / "weight"), "sha256": "5" * 64},
    }
    authorization_completion = [dict(item) for item in completion]
    authorization_completion[0]["status_sha256"] = "f" * 64
    authorization: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_authorization_v1",
        "status": "COMPLETE_ONLY",
        "source_stage": "formal256_primary",
        "next_stage": "formal256_repeat",
        "report": {**_record(report), "schema": report_payload["schema"]},
        "dependencies": {
            **identities,
            "source_stage_manifest": source_identity,
            "formal_campaign": campaign,
        },
        "completion": authorization_completion,
    }
    authorization["semantic_sha256"] = canonical_hash(authorization)
    auth = output / "authorizations" / "formal256_primary__to__formal256_repeat.json"
    auth.parent.mkdir()
    auth.write_bytes(runner_module.canonical_json_bytes(authorization))
    monkeypatch.setattr(
        runner_module,
        "_validate_source_stage_identity",
        lambda *args, **kwargs: (source_identity, source_payload),
    )
    monkeypatch.setattr(runner_module, "_validate_completion_only", lambda *args, **kwargs: None)
    context = StageContext(
        output_root=output.resolve(),
        master_manifest=Path(identities["master_manifest"]["path"]),
        development_suite=Path(identities["development_suite"]["path"]),
        formal_suite=Path(identities["formal_suite"]["path"]),
        frozen_weights=Path(identities["frozen_weights"]["path"]),
        authorization=auth.resolve(),
        formal_campaign=campaign_path.resolve(),
    )

    with pytest.raises(ValueError, match="report|completion|equivalent|40|mismatch"):
        runner_module._validate_authorization(
            "formal256_repeat",
            context,
            output_root=output.resolve(),
            master_identity=identities["master_manifest"],
            development_identity=identities["development_suite"],
            formal_identity=identities["formal_suite"],
            specification_identity=identities["specification"],
            weight_identity=identities["frozen_weights"],
            campaign_identity=campaign,
            campaign_payload={"primary": []},
        )


@pytest.mark.parametrize("completed_count", [0, 17, 39])
def test_completion_stop_report_accepts_exact_strict_partial_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed_count: int,
) -> None:
    """Break caught: a valid STOP count is incorrectly forced to zero or forty."""

    summarizer = tmp_path / "authoritative-completion-summarizer.py"
    summarizer.write_bytes(b"# tmp-only Task 12 completion identity seam\n")
    monkeypatch.setattr(
        runner_module, "_SUMMARIZER_SOURCE", summarizer.resolve(), raising=False
    )
    source_identity = {
        "path": str((tmp_path / "formal256_primary.json").resolve()),
        "sha256": "1" * 64,
        "semantic_sha256": "2" * 64,
    }
    campaign_identity = {
        "path": str((tmp_path / "formal_campaign.json").resolve()),
        "sha256": "3" * 64,
        "semantic_sha256": "4" * 64,
    }
    report: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_completion_report_v1",
        "status": "STOP",
        "source_stage": "formal256_primary",
        "next_stage": "formal256_repeat",
        "source_stage_manifest": source_identity,
        "formal_campaign": campaign_identity,
        "registered_run_count": 40,
        "completed_run_count": completed_count,
        "completion": None,
        "metric_values_parsed": False,
        "failure_codes": ["primary_run_incomplete"],
        "policy": {
            "completion_only": True,
            "primary_outcomes_opaque": True,
            "repeats_inference_eligible": False,
        },
        "summarizer_source": _record(summarizer),
    }
    report["semantic_sha256"] = canonical_hash(report)

    validated = runner_module._validate_task12_completion_report(
        report,
        source_identity=source_identity,
        campaign_identity=campaign_identity,
    )
    assert validated["completed_run_count"] == completed_count


@pytest.mark.parametrize("completed_count", [-1, 40, True])
def test_completion_stop_report_rejects_nonpartial_or_boolean_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed_count: object,
) -> None:
    """Break caught: STOP accepts a non-strict, success-sized, or boolean count."""

    summarizer = tmp_path / "authoritative-completion-summarizer.py"
    summarizer.write_bytes(b"# tmp-only Task 12 completion identity seam\n")
    monkeypatch.setattr(
        runner_module, "_SUMMARIZER_SOURCE", summarizer.resolve(), raising=False
    )
    source_identity = {
        "path": str((tmp_path / "formal256_primary.json").resolve()),
        "sha256": "1" * 64,
        "semantic_sha256": "2" * 64,
    }
    campaign_identity = {
        "path": str((tmp_path / "formal_campaign.json").resolve()),
        "sha256": "3" * 64,
        "semantic_sha256": "4" * 64,
    }
    report: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_completion_report_v1",
        "status": "STOP",
        "source_stage": "formal256_primary",
        "next_stage": "formal256_repeat",
        "source_stage_manifest": source_identity,
        "formal_campaign": campaign_identity,
        "registered_run_count": 40,
        "completed_run_count": completed_count,
        "completion": None,
        "metric_values_parsed": False,
        "failure_codes": ["primary_run_incomplete"],
        "policy": {
            "completion_only": True,
            "primary_outcomes_opaque": True,
            "repeats_inference_eligible": False,
        },
        "summarizer_source": _record(summarizer),
    }
    report["semantic_sha256"] = canonical_hash(report)

    with pytest.raises(ValueError, match="completed|count|STOP|partial|integer"):
        runner_module._validate_task12_completion_report(
            report,
            source_identity=source_identity,
            campaign_identity=campaign_identity,
        )


@pytest.mark.parametrize(
    "failure_codes",
    [
        ["INCOMPLETE_PRIMARY_RUNS"],
        ["authorization_invalid"],
        ["primary_run_incomplete", "primary_run_unknown"],
    ],
)
def test_completion_stop_rejects_every_code_outside_frozen_two_code_vocabulary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_codes: list[str],
) -> None:
    """Break caught: any nonempty STOP code list is accepted as authorization."""

    summarizer = tmp_path / "authoritative-completion-summarizer.py"
    summarizer.write_bytes(b"# tmp-only Task 12 completion identity seam\n")
    monkeypatch.setattr(
        runner_module, "_SUMMARIZER_SOURCE", summarizer.resolve(), raising=False
    )
    source_identity = {
        "path": str((tmp_path / "formal256_primary.json").resolve()),
        "sha256": "1" * 64,
        "semantic_sha256": "2" * 64,
    }
    campaign_identity = {
        "path": str((tmp_path / "formal_campaign.json").resolve()),
        "sha256": "3" * 64,
        "semantic_sha256": "4" * 64,
    }
    report: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_completion_report_v1",
        "status": "STOP",
        "source_stage": "formal256_primary",
        "next_stage": "formal256_repeat",
        "source_stage_manifest": source_identity,
        "formal_campaign": campaign_identity,
        "registered_run_count": 40,
        "completed_run_count": 17,
        "completion": None,
        "metric_values_parsed": False,
        "failure_codes": failure_codes,
        "policy": {
            "completion_only": True,
            "primary_outcomes_opaque": True,
            "repeats_inference_eligible": False,
        },
        "summarizer_source": _record(summarizer),
    }
    report["semantic_sha256"] = canonical_hash(report)

    with pytest.raises(ValueError, match="failure|code|STOP|vocabulary"):
        runner_module._validate_task12_completion_report(
            report,
            source_identity=source_identity,
            campaign_identity=campaign_identity,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra_metric", "reordered"])
def test_completion_only_aggregate_rejects_nonexact_structural_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: malformed structural evidence can authorize formal repeats."""

    completion, stage, campaign, output_root = _exact_completion_campaign_fixture(
        tmp_path, monkeypatch
    )
    if mutation == "missing":
        completion.pop()
    elif mutation == "extra_metric":
        completion[0]["metric_value"] = 0.9
    elif mutation == "reordered":
        completion[0], completion[1] = completion[1], completion[0]
    with pytest.raises(ValueError, match="completion|40|ordered|schema|metric|stale|status"):
        runner_module._validate_completion_only(
            completion, stage, campaign, output_root=output_root
        )


def test_completed_run_reuses_task8_strict_status_chain_and_artifact_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: STATUS existence/count substitutes for Task 8 semantic artifact validation."""

    spec = _one_spec(tmp_path, monkeypatch)
    run_dir = spec.output_path
    run_dir.mkdir(parents=True)
    expected = "8" * 64
    calls: list[tuple[Path, int, bool]] = []
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda _spec: expected)
    monkeypatch.setattr(trainer_module, "_preflight_reserved_top_level_paths", lambda path: None)
    monkeypatch.setattr(
        trainer_module,
        "_status",
        lambda path: {
            "schema": "pcraft_soft_loss_v3_status_v1",
            "state": "completed",
            "config_hash": expected,
            "completed_updates": spec.steps,
            "artifact_manifest_sha256": "a" * 64,
        },
    )

    def strict_validate(path: Path, status: object, config: str, steps: int, *, calibration: bool = False) -> dict[str, object]:
        assert config == expected
        calls.append((path, steps, calibration))
        return {"strict": True}

    monkeypatch.setattr(trainer_module, "_validate_completed_run", strict_validate)
    command = {
        "schema": "pcraft_soft_loss_v3_command_v1",
        "argv": trainer_argv(spec),
        "resolved_config": {"fixture": "strictly validated by the Task 8 seam"},
        "config_hash": expected,
    }
    # Exercise the real Task 8 producer contract.  Task 8 deliberately writes
    # sorted, indented JSON with one trailing newline via atomic_json; a compact
    # registry manifest is not representative of a completed trainer run.
    trainer_module.atomic_json(run_dir / "COMMAND.json", command)
    assert is_completed_run(run_dir, spec) is True
    assert calls == [(run_dir.resolve(), spec.steps, False)]


def test_completed_run_returns_false_only_for_compatible_running_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: stale/corrupt state is treated as resumable or completed."""

    spec = _one_spec(tmp_path, monkeypatch)
    spec.output_path.mkdir(parents=True)
    expected = "8" * 64
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda _spec: expected)
    monkeypatch.setattr(trainer_module, "_preflight_reserved_top_level_paths", lambda path: None)
    state = {
        "schema": "pcraft_soft_loss_v3_status_v1",
        "state": "running",
        "config_hash": expected,
        "completed_updates": 0,
    }
    monkeypatch.setattr(trainer_module, "_status", lambda path: dict(state))
    assert is_completed_run(spec.output_path, spec) is False
    state["config_hash"] = "0" * 64
    with pytest.raises(ValueError, match="config|stale|mismatch"):
        is_completed_run(spec.output_path, spec)
    state["config_hash"] = expected
    state["state"] = "failed"
    with pytest.raises(ValueError, match="failed|terminal|state"):
        is_completed_run(spec.output_path, spec)


def test_run_specs_dry_run_registers_but_never_launches_or_creates_run_or_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: dry-run starts training or mutates a fresh run/log directory."""

    specs = _fake_specs(tmp_path, monkeypatch, "smoke128")
    registrations: list[tuple[str, int, Path]] = []
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: "8" * 64)
    monkeypatch.setattr(
        runner_module,
        "_revalidate_runtime_registration",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner_module,
        "write_stage_manifest",
        lambda stage, values, root: registrations.append((stage, len(values), root)) or root / "stage.json",
    )
    monkeypatch.setattr(runner_module, "is_completed_run", lambda path, spec: False)
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dry-run launched a trainer"),
    )
    counts = run_specs(specs, dry_run=True, resume=False, max_runs=None)
    assert counts == {
        "registered": 4,
        "completed": 0,
        "launched": 0,
        "resumed": 0,
        "dry_run": 4,
        "limited": 0,
    }
    assert registrations == [("smoke128", 4, specs[0].output_path.parents[5])]
    assert all(not spec.output_path.exists() for spec in specs)
    assert not specs[0].output_path.parents[5].joinpath("logs").exists()


def test_run_specs_skips_strict_complete_limits_launches_and_keeps_logs_external(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: max-runs counts skips, launch order changes, or stdout enters run dirs."""

    specs = _fake_specs(tmp_path, monkeypatch, "smoke128")
    completed = {specs[0].run_id}
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: "8" * 64)
    monkeypatch.setattr(
        runner_module,
        "_revalidate_runtime_registration",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(runner_module, "write_stage_manifest", lambda *args, **kwargs: tmp_path / "stage.json")
    monkeypatch.setattr(
        runner_module,
        "is_completed_run",
        lambda path, spec: spec.run_id in completed,
    )
    launches: list[tuple[str, str, str]] = []

    def launch(argv: list[str], **kwargs: object) -> object:
        output = Path(argv[4])
        assert not output.exists()
        output.mkdir(parents=True)
        completed.add(output.relative_to(specs[0].output_path.parents[5]).as_posix())
        stdout = kwargs["stdout"]
        stderr = kwargs["stderr"]
        assert not Path(stdout.name).is_relative_to(output)
        assert not Path(stderr.name).is_relative_to(output)
        launches.append((argv[4], stdout.name, stderr.name))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(runner_module.subprocess, "run", launch)
    counts = run_specs(specs, dry_run=False, resume=False, max_runs=2)
    assert counts == {
        "registered": 4,
        "completed": 1,
        "launched": 2,
        "resumed": 0,
        "dry_run": 0,
        "limited": 1,
    }
    assert [Path(item[0]) for item in launches] == [specs[1].output_path, specs[2].output_path]
    assert all("logs" in Path(item[1]).parts for item in launches)


def test_run_specs_resumes_only_compatible_incomplete_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: resume repairs stale/corrupt state or omits Task 8's resume flag."""

    spec = _fake_specs(tmp_path, monkeypatch, "calibrate128")[0]
    spec.output_path.mkdir(parents=True)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda value: "8" * 64)
    monkeypatch.setattr(
        runner_module,
        "_revalidate_runtime_registration",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(runner_module, "write_stage_manifest", lambda *args, **kwargs: tmp_path / "stage.json")
    sequence = iter((False, True))
    monkeypatch.setattr(runner_module, "is_completed_run", lambda path, value: next(sequence))
    compatible: list[Path] = []
    monkeypatch.setattr(
        runner_module,
        "_validate_compatible_incomplete",
        lambda path, value: compatible.append(path),
    )

    def launch(argv: list[str], **kwargs: object) -> object:
        assert argv[-1] == "--resume"
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(runner_module.subprocess, "run", launch)
    counts = run_specs([spec], dry_run=False, resume=True, max_runs=None)
    assert compatible == [spec.output_path]
    assert counts["launched"] == 1
    assert counts["resumed"] == 1

    monkeypatch.setattr(runner_module, "is_completed_run", lambda path, value: False)
    monkeypatch.setattr(
        runner_module,
        "_validate_compatible_incomplete",
        lambda path, value: (_ for _ in ()).throw(ValueError("stale config")),
    )
    with pytest.raises(ValueError, match="stale"):
        run_specs([spec], dry_run=False, resume=True, max_runs=None)


def test_running_resume_preflight_validates_authoritative_transaction_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: compatible audit files bypass a corrupt running pointer/chain."""

    spec = _one_spec(tmp_path, monkeypatch, stage="calibrate128")
    spec.output_path.mkdir(parents=True)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda value: "8" * 64)
    monkeypatch.setattr(trainer_module, "_preflight_reserved_top_level_paths", lambda path: None)
    monkeypatch.setattr(
        trainer_module,
        "environment_manifest",
        lambda device: {"nondeterminism_reason": "fixture", "environment": {}},
    )
    digest_seen: list[str] = []

    def audits(path: Path, args: object, config: object, digest: str) -> dict[str, object]:
        digest_seen.append(digest)
        return {
            "schema": "pcraft_soft_loss_v3_status_v1",
            "state": "running",
            "config_hash": digest,
            "completed_updates": 0,
        }

    monkeypatch.setattr(trainer_module, "_validate_existing_audits", audits)
    observed: list[tuple[Path, str, object]] = []

    def verify_chain(path: Path, **kwargs: object) -> object:
        observed.append(
            (path, str(kwargs["expected_config_hash"]), kwargs.get("expected_final_step"))
        )
        return types.SimpleNamespace(completed_updates=37)

    monkeypatch.setattr(trainer_module, "_verify_transaction_chain", verify_chain)
    runner_module._validate_compatible_incomplete(spec.output_path, spec)
    assert observed == [(spec.output_path.resolve(), digest_seen[0], None)]


@pytest.mark.parametrize("chain_update", [-1, 2000, 2001, "37"])
def test_running_resume_rejects_malformed_or_nonincomplete_chain_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chain_update: object,
) -> None:
    """Break caught: STATUS zero hides a malformed/ahead authoritative chain head."""

    spec = _one_spec(tmp_path, monkeypatch, stage="calibrate128")
    spec.output_path.mkdir(parents=True)
    monkeypatch.setattr(trainer_module, "_preflight_reserved_top_level_paths", lambda path: None)
    monkeypatch.setattr(
        trainer_module,
        "_validate_existing_audits",
        lambda path, args, config, digest: {
            "schema": "pcraft_soft_loss_v3_status_v1",
            "state": "running",
            "config_hash": digest,
            "completed_updates": 0,
        },
    )
    monkeypatch.setattr(
        trainer_module,
        "_verify_transaction_chain",
        lambda *args, **kwargs: types.SimpleNamespace(completed_updates=chain_update),
    )
    with pytest.raises(ValueError, match="transaction|chain|progress|completed|step"):
        runner_module._validate_compatible_incomplete(spec.output_path, spec)


def test_corrupt_running_transaction_chain_blocks_resume_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a corrupt running chain reaches trainer spawn."""

    spec = _fake_specs(tmp_path, monkeypatch, "calibrate128")[0]
    spec.output_path.mkdir(parents=True)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda value: "8" * 64)
    monkeypatch.setattr(
        runner_module,
        "_revalidate_runtime_registration",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(runner_module, "is_completed_run", lambda path, value: False)
    monkeypatch.setattr(trainer_module, "_preflight_reserved_top_level_paths", lambda path: None)
    monkeypatch.setattr(
        trainer_module,
        "environment_manifest",
        lambda device: {"nondeterminism_reason": "fixture", "environment": {}},
    )
    monkeypatch.setattr(
        trainer_module,
        "_validate_existing_audits",
        lambda path, args, config, digest: {
            "state": "running",
            "config_hash": digest,
            "completed_updates": 0,
        },
    )
    monkeypatch.setattr(
        trainer_module,
        "_verify_transaction_chain",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("corrupt transaction chain")),
    )
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("corrupt chain spawned trainer"),
    )
    with pytest.raises(ValueError, match="corrupt|transaction|chain"):
        run_specs([spec], dry_run=False, resume=True, max_runs=None)


@pytest.mark.parametrize("operation", ["complete", "resume"])
def test_reuse_rejects_symlinked_ancestor_before_task8_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Break caught: completed/resume reads escape through a symlinked stage ancestor."""

    spec = _one_spec(tmp_path, monkeypatch, stage="calibrate128")
    root = spec.output_path.parents[5]
    external_stage = tmp_path / "external-stage"
    external_run = external_stage.joinpath(*spec.output_path.relative_to(root).parts[1:])
    external_run.mkdir(parents=True)
    (root / spec.stage).symlink_to(external_stage, target_is_directory=True)
    reads: list[Path] = []
    monkeypatch.setattr(
        trainer_module,
        "_preflight_reserved_top_level_paths",
        lambda path: reads.append(Path(path)),
    )
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda value: "8" * 64)
    monkeypatch.setattr(
        trainer_module,
        "_status",
        lambda path: {
            "schema": "pcraft_soft_loss_v3_status_v1",
            "state": "completed",
            "config_hash": "8" * 64,
            "completed_updates": spec.steps,
            "artifact_manifest_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(trainer_module, "_validate_completed_run", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        trainer_module,
        "_validate_existing_audits",
        lambda path, args, config, digest: {
            "schema": "pcraft_soft_loss_v3_status_v1",
            "state": "running",
            "config_hash": digest,
            "completed_updates": 0,
        },
    )
    monkeypatch.setattr(
        trainer_module,
        "_verify_transaction_chain",
        lambda *args, **kwargs: types.SimpleNamespace(completed_updates=37),
    )

    with pytest.raises(ValueError, match="symlink|ancestor|path|escape|registered"):
        if operation == "complete":
            is_completed_run(spec.output_path, spec)
        else:
            runner_module._validate_compatible_incomplete(spec.output_path, spec)
    assert reads == []


@pytest.mark.parametrize("stage", ["smoke128", "sentinel_repeat256", "formal256_repeat"])
def test_completed_reuse_command_binds_exact_registered_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    """Break caught: copied completed bytes masquerade as a distinct repeat path."""

    spec = _fake_specs(tmp_path, monkeypatch, stage)[0]
    spec.output_path.mkdir(parents=True)
    wrong = tmp_path / "copied-primary-output"
    argv = trainer_argv(spec)
    argv[4] = str(wrong.resolve())
    command = {
        "schema": "pcraft_soft_loss_v3_command_v1",
        "argv": argv,
        "resolved_config": {"fixture": "already validated by Task 8"},
        "config_hash": "8" * 64,
    }
    spec.output_path.joinpath("COMMAND.json").write_bytes(
        runner_module.canonical_json_bytes(command)
    )
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda value: "8" * 64)
    monkeypatch.setattr(trainer_module, "_preflight_reserved_top_level_paths", lambda path: None)
    monkeypatch.setattr(
        trainer_module,
        "_status",
        lambda path: {
            "schema": "pcraft_soft_loss_v3_status_v1",
            "state": "completed",
            "config_hash": "8" * 64,
            "completed_updates": spec.steps,
            "artifact_manifest_sha256": "9" * 64,
        },
    )
    monkeypatch.setattr(trainer_module, "_validate_completed_run", lambda *args, **kwargs: {})

    with pytest.raises(ValueError, match="COMMAND|command|output|path|copied|registered"):
        is_completed_run(spec.output_path, spec)


def test_attempt_scoped_logs_preserve_history_across_multiple_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the first historical stdout.log makes a later resume impossible."""

    spec = _fake_specs(tmp_path, monkeypatch, "calibrate128")[0]
    spec.output_path.mkdir(parents=True)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda value: "8" * 64)
    monkeypatch.setattr(
        runner_module,
        "_revalidate_runtime_registration",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        runner_module,
        "write_stage_manifest",
        lambda *args, **kwargs: tmp_path / "calibration-stage.json",
    )
    states = iter((False, True, False, True))
    monkeypatch.setattr(
        runner_module,
        "is_completed_run",
        lambda path, value: next(states),
    )
    monkeypatch.setattr(
        runner_module,
        "_validate_compatible_incomplete",
        lambda path, value: None,
    )
    attempts = 0

    def launch(argv: list[str], **kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        kwargs["stdout"].write(f"stdout-{attempts}".encode("ascii"))
        kwargs["stderr"].write(f"stderr-{attempts}".encode("ascii"))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(runner_module.subprocess, "run", launch)
    assert run_specs([spec], dry_run=False, resume=True, max_runs=None)["resumed"] == 1
    assert run_specs([spec], dry_run=False, resume=True, max_runs=None)["resumed"] == 1
    log_root = runner_module._log_paths(spec, spec.output_path.parents[5])[0].parent
    assert log_root.joinpath("attempt_000001", "stdout.log").read_bytes() == b"stdout-1"
    assert log_root.joinpath("attempt_000002", "stdout.log").read_bytes() == b"stdout-2"
    assert log_root.joinpath("attempt_000001", "stderr.log").read_bytes() == b"stderr-1"
    assert log_root.joinpath("attempt_000002", "stderr.log").read_bytes() == b"stderr-2"


def test_single_run_orchestration_lock_rejects_concurrent_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: two runner processes can resume the same run concurrently."""

    spec = _fake_specs(tmp_path, monkeypatch, "calibrate128")[0]
    spec.output_path.mkdir(parents=True)
    root = spec.output_path.parents[5]
    log_root = runner_module._log_paths(spec, root)[0].parent
    log_root.mkdir(parents=True)
    lock_path = log_root / ".orchestration.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda value: "8" * 64)
    monkeypatch.setattr(
        runner_module,
        "_revalidate_runtime_registration",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(runner_module, "is_completed_run", lambda path, value: False)
    monkeypatch.setattr(
        runner_module,
        "_validate_compatible_incomplete",
        lambda path, value: None,
    )
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("concurrent resume spawned trainer"),
    )
    try:
        with pytest.raises(ValueError, match="lock|concurrent|orchestration"):
            run_specs([spec], dry_run=False, resume=True, max_runs=None)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _real_smoke_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[RunSpec]:
    paths = _make_bound_registry(tmp_path, monkeypatch)
    weight, _ = _task10_weight_fixture(paths, monkeypatch)
    return build_stage_specs(
        "smoke128",
        StageContext(
            output_root=paths["output"].resolve(),
            master_manifest=paths["master"].resolve(),
            development_suite=paths["development"].resolve(),
            formal_suite=paths["formal"].resolve(),
            frozen_weights=weight.resolve(),
            authorization=None,
            formal_campaign=None,
        ),
    )


def _real_formal_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[RunSpec], list[RunSpec]]:
    paths = _make_bound_registry(tmp_path, monkeypatch)
    weight, _ = _task10_weight_fixture(paths, monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "_validate_authorization",
        lambda *args, **kwargs: {
            "path": str(paths["output"] / "synthetic-authorization.json"),
            "sha256": "6" * 64,
            "semantic_sha256": "7" * 64,
            "status": "ADVANCE",
            "source_stage": "sentinel_repeat256",
            "next_stage": "formal256_primary",
        },
    )
    primary = build_stage_specs(
        "formal256_primary",
        StageContext(
            output_root=paths["output"].resolve(),
            master_manifest=paths["master"].resolve(),
            development_suite=paths["development"].resolve(),
            formal_suite=paths["formal"].resolve(),
            frozen_weights=weight.resolve(),
            authorization=(paths["output"] / "synthetic-authorization.json").resolve(),
            formal_campaign=(
                paths["output"] / "manifests" / "formal_campaign.json"
            ).resolve(),
        ),
    )
    return primary, runner_module._formal_repeat_specs(primary, paths["output"].resolve())


@pytest.mark.parametrize("writer", ["stage", "campaign"])
@pytest.mark.parametrize("identity", ["master", "suite", "specification", "weight"])
def test_public_writers_reject_self_consistent_foreign_scientific_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer: str,
    identity: str,
) -> None:
    """Break caught: public publication trusts foreign but byte-current identities."""

    if writer == "stage":
        first = _real_smoke_specs(tmp_path, monkeypatch)
        second: list[RunSpec] | None = None
    else:
        first, second = _real_formal_specs(tmp_path, monkeypatch)
    root = first[0].output_path.parents[5]
    field = {
        "master": "master_identity",
        "suite": "suite_identity",
        "specification": "specification_identity",
        "weight": "weight_identity",
    }[identity]
    original = getattr(first[0], field)
    assert original is not None
    foreign = tmp_path / f"foreign-{writer}-{identity}.json"
    foreign.write_bytes(Path(str(original["path"])).read_bytes())
    replacement = {**dict(original), "path": str(foreign.resolve()), "sha256": _sha(foreign)}
    first = [replace(spec, **{field: replacement}) for spec in first]
    if second is not None:
        second = [replace(spec, **{field: replacement}) for spec in second]

    target = (
        root / "manifests" / "stages" / "smoke128.json"
        if writer == "stage"
        else root / "manifests" / "formal_campaign.json"
    )
    with pytest.raises(ValueError, match="master|suite|specification|weight|foreign|authoritative|registered|path"):
        if writer == "stage":
            write_stage_manifest("smoke128", first, root)
        else:
            assert second is not None
            write_formal_campaign_manifest(first, second, target)
    assert not target.exists()


@pytest.mark.parametrize("branch", ["skip", "dry"])
def test_every_cell_revalidates_dependencies_before_skip_and_dry_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
) -> None:
    """Break caught: dependency drift after cell one is ignored by skip/dry cells."""

    specs = _real_smoke_specs(tmp_path, monkeypatch)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: "8" * 64)
    calls = 0

    def completed(path: Path, spec: RunSpec) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            target = Path(specs[1].inputs_xyz[0]["resolved_path"])
            target.write_bytes(target.read_bytes() + b"between-cells-drift")
        return branch == "skip"

    monkeypatch.setattr(runner_module, "is_completed_run", completed)
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dependency drift launched a trainer"),
    )
    with pytest.raises(ValueError, match="input|dependency|changed|registration|sha256"):
        run_specs(
            specs,
            dry_run=branch == "dry",
            resume=False,
            max_runs=None,
        )
    assert calls == 1


@pytest.mark.parametrize("drift", ["input", "config"])
def test_postprocess_revalidation_blocks_cell_two_after_cell_one_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    """Break caught: cell two launches after cell one changes an input/config identity."""

    specs = _real_smoke_specs(tmp_path, monkeypatch)
    config_state = {"drift": False}
    monkeypatch.setattr(
        runner_module,
        "expected_config_hash",
        lambda spec: ("9" if config_state["drift"] else "8") * 64,
    )
    completed: set[str] = set()
    monkeypatch.setattr(
        runner_module,
        "is_completed_run",
        lambda path, spec: spec.run_id in completed,
    )
    launches: list[str] = []

    def launch(argv: list[str], **kwargs: object) -> object:
        launches.append(argv[4])
        completed.add(specs[0].run_id)
        if drift == "input":
            target = Path(specs[1].inputs_xyz[0]["resolved_path"])
            target.write_bytes(target.read_bytes() + b"post-cell-drift")
        else:
            config_state["drift"] = True
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(runner_module.subprocess, "run", launch)
    with pytest.raises(ValueError, match="input|config|dependency|registration|changed"):
        run_specs(specs, dry_run=False, resume=False, max_runs=None)
    assert launches == [str(specs[0].output_path)]


@pytest.mark.parametrize("branch", ["completed", "dry", "limited"])
def test_final_cell_branch_revalidates_before_continue_and_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    branch: str,
) -> None:
    """Break caught: dependency drift in the final continue branch escapes the loop."""

    specs = _real_smoke_specs(tmp_path, monkeypatch)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: "8" * 64)
    completed_ids: set[str] = set()
    initial_checks: dict[str, int] = {}

    def completed(path: Path, spec: RunSpec) -> bool:
        initial_checks[spec.run_id] = initial_checks.get(spec.run_id, 0) + 1
        if spec is specs[-1] and initial_checks[spec.run_id] == 1:
            target = Path(spec.inputs_xyz[0]["resolved_path"])
            target.write_bytes(target.read_bytes() + b"final-branch-drift")
            return branch == "completed"
        if branch == "limited" and spec in specs[1:3]:
            return True
        return spec.run_id in completed_ids

    def launch(argv: list[str], **kwargs: object) -> object:
        output = Path(argv[4])
        output.mkdir(parents=True)
        completed_ids.add(output.relative_to(specs[0].output_path.parents[5]).as_posix())
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(runner_module, "is_completed_run", completed)
    monkeypatch.setattr(runner_module.subprocess, "run", launch)
    with pytest.raises(ValueError, match="input|dependency|registration|changed|sha256"):
        run_specs(
            specs,
            dry_run=branch == "dry",
            resume=False,
            max_runs=1 if branch == "limited" else None,
        )


def test_run_specs_rejects_unlocked_parent_before_registration_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: an ambient interpreter publishes a manifest for a different child."""

    specs = _real_smoke_specs(tmp_path, monkeypatch)
    root = specs[0].output_path.parents[5]
    foreign = tmp_path / "ambient-python"
    foreign.write_bytes(b"not the locked interpreter")
    monkeypatch.setattr(runner_module.sys, "executable", str(foreign.resolve()))
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: "8" * 64)
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("unlocked parent launched a trainer"),
    )

    with pytest.raises(ValueError, match="interpreter|locked|sys.executable|parent"):
        run_specs(specs, dry_run=True, resume=False, max_runs=None)
    assert not root.joinpath("manifests", "stages", "smoke128.json").exists()


def test_expected_config_and_child_share_explicit_allowlisted_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: ambient PYTHONPATH/secrets reach child or alter registered config."""

    spec = _one_spec(tmp_path, monkeypatch, stage="calibrate128")
    monkeypatch.setattr(trainer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        trainer_module,
        "environment_manifest",
        lambda device: {
            "nondeterminism_reason": "synthetic environment fixture",
            "environment": {
                name: os.environ[name]
                for name in (
                    "PYTHONPATH",
                    "CUDA_VISIBLE_DEVICES",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "TORCH_DETERMINISTIC",
                    "TORCH_USE_CUDA_DSA",
                    "NVIDIA_TF32_OVERRIDE",
                )
                if name in os.environ
            },
        },
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "shadow-a"))
    monkeypatch.setenv("TASK11_OUTCOME_SCORE", "0.999")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-propagate")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    first = expected_config_hash(spec)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "shadow-b"))
    monkeypatch.setenv("TASK11_OUTCOME_SCORE", "changed")
    second = expected_config_hash(spec)
    child = runner_module._controlled_launch_environment()

    assert first == second
    assert child["PYTHONPATH"] == str(runner_module._PROJECT_ROOT)
    assert child["PYTHONNOUSERSITE"] == "1"
    assert child["PYTHONSAFEPATH"] == "1"
    assert child["CUDA_VISIBLE_DEVICES"] == "7"
    assert "TASK11_OUTCOME_SCORE" not in child
    assert "AWS_SECRET_ACCESS_KEY" not in child


@pytest.mark.parametrize("escape", ["run", "log"])
def test_last_prelaunch_boundary_rejects_run_and_log_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    escape: str,
) -> None:
    """Break caught: a registered run/log component follows a symlink outside output."""

    specs = _real_smoke_specs(tmp_path, monkeypatch)
    root = specs[0].output_path.parents[5]
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / ("smoke128" if escape == "run" else "logs")
    link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: "8" * 64)
    monkeypatch.setattr(runner_module, "is_completed_run", lambda path, spec: False)
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("symlink escape launched a trainer"),
    )
    with pytest.raises(ValueError, match="symlink|escape|run|log|path"):
        run_specs(specs, dry_run=False, resume=False, max_runs=1)


def test_launch_uses_locked_interpreter_project_cwd_and_controlled_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: ambient interpreter/cwd/PYTHONPATH can shadow recorded v3 code."""

    specs = _real_smoke_specs(tmp_path, monkeypatch)
    monkeypatch.setattr(runner_module, "expected_config_hash", lambda spec: "8" * 64)
    monkeypatch.setenv("PYTHONPATH", "/tmp/evil-shadow-package")
    completed: set[str] = set()
    monkeypatch.setattr(
        runner_module,
        "is_completed_run",
        lambda path, spec: spec.run_id in completed,
    )
    observed: list[tuple[list[str], object, object]] = []

    def launch(argv: list[str], **kwargs: object) -> object:
        observed.append((argv, kwargs.get("cwd"), kwargs.get("env")))
        completed.add(specs[0].run_id)
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(runner_module.subprocess, "run", launch)
    counts = run_specs(specs, dry_run=False, resume=False, max_runs=1)
    assert counts["launched"] == 1
    argv, cwd, environment = observed[0]
    assert argv[0] == "/home/lin/anaconda3/envs/shadowart-revisited/bin/python"
    assert cwd == runner_module._PROJECT_ROOT
    assert environment["PYTHONPATH"] == str(runner_module._PROJECT_ROOT)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "/tmp/evil-shadow-package" not in environment["PYTHONPATH"]


def test_runner_source_scope_has_no_data_training_analysis_or_metric_value_reader() -> None:
    """Break caught: Task 11 grows into training, analysis, geometry, or outcome selection."""

    source_path = Path(runner_module.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert '"/data"' not in source and "'/data'" not in source
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        name.endswith(("metrics", "statistics", "summarize_soft_loss_v3"))
        for name in imported
    )
    assert '(run_dir / "metrics.json").read' not in source
    assert "torch" not in imported
