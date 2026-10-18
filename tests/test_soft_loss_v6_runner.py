"""Fail-closed contracts for the exact v6 10x4 runner."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import experiments.run_soft_loss_v6_ablation10_256 as runner
from experiments.soft_loss_v3_registry import bind_formal_suite
from experiments.soft_loss_v4_registry import bind_v4_suite
from shadowforge_pipeline.soft_voxel_v4.provenance import canonical_json_bytes
from shadowforge_pipeline.soft_voxel_v6.core import PROFILE_ORDER, TASK_ORDER
from shadowforge_pipeline.soft_voxel_v6.provenance import canonical_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = PROJECT_ROOT / ".superpowers/sdd/2026-08-28-soft-loss-v6-256-ablation/spec.md"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _write_v6_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical_json_bytes(payload))


def _torch_bytes(payload: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _engineering_checkpoint_payload(*, peak: int, seconds: float) -> dict[str, object]:
    return {
        "step_times": [seconds] * 200,
        "history": [{"peak_cuda_allocated_bytes": peak}] * 200,
    }


def _write_running_engineering_chain(run_dir: Path, current: int) -> str:
    predecessor: str | None = None
    latest = ""
    for completed in range(200, current + 1, 200):
        payload = (
            _engineering_checkpoint_payload(peak=13, seconds=0.1)
            if completed == 200
            else {}
        )
        payload["predecessor_sha256"] = predecessor
        path = run_dir / "checkpoints" / f"step_{completed:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_torch_bytes(payload))
        latest = _sha(path)
        predecessor = latest
    return latest


def _loaded_v6_execution_sources() -> set[str]:
    names = {
        "experiments.run_soft_loss_v6_ablation10_256",
        "experiments.soft_loss_v3_registry",
        "experiments.soft_loss_v4_registry",
    }
    prefixes = (
        "shadowforge_pipeline.soft_voxel_v3",
        "shadowforge_pipeline.soft_voxel_v4",
        "shadowforge_pipeline.soft_voxel_v6",
    )
    relatives: set[str] = set()
    for name, module in tuple(sys.modules.items()):
        if name not in names and not name.startswith(prefixes):
            continue
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            continue
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            continue
        if path.suffix == ".py":
            relatives.add(relative)
    return relatives


def _formal_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import experiments.soft_loss_v3_registry as v3_registry

    project = tmp_path / "project"
    formal = project / "formal"
    cases: list[dict[str, object]] = []
    for case_index, case_id in enumerate(TASK_ORDER):
        paths: list[str] = []
        for axis_index, axis in enumerate(("x", "y", "z")):
            image = np.zeros((9 + axis_index, 11 + case_index % 3), dtype=np.uint8)
            image[1 : 5 + axis_index, 2 : 6 + case_index % 3] = 255
            image[0, 0] = case_index + axis_index + 1
            path = formal / "inputs" / f"{case_id}_view_{axis}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(path), image)
            paths.append(str(path.resolve()))
        cases.append({"case_id": case_id, "target_paths_xyz": paths})
    source_manifest = formal / "benchmark_manifest.json"
    _write_json(
        source_manifest,
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
            "cases": cases,
        },
    )
    monkeypatch.setattr(v3_registry, "PROJECT_ROOT", project)
    v3_payload = bind_formal_suite(formal, source_manifest)
    v3_path = tmp_path / "v3" / "formal_suite.json"
    _write_json(v3_path, v3_payload)
    v4_root = tmp_path / "v4"
    output = v4_root / "manifests/formal_suite.json"
    bind_v4_suite(v3_path, output, "formal")
    return output


@pytest.fixture()
def manifest_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    suite = _formal_suite(tmp_path, monkeypatch)
    root = tmp_path / "output"
    manifest = runner._build_manifest_for_test(
        output_root=root,
        formal_suite=suite,
        device="cpu",
        resolution=8,
    )
    return root, suite, manifest


def test_manifest_exact_schema_matrix_order_paths_hash_and_cross_arm_fairness(
    manifest_bundle,
) -> None:
    root, suite, manifest = manifest_bundle
    assert set(manifest) == {
        "schema", "status", "output_root", "formal_suite", "specification",
        "implementation", "protocol", "objective", "schedule", "decision_policy",
        "tasks", "profiles", "cells", "semantic_sha256",
    }
    assert manifest["status"] == "REGISTERED"
    assert manifest["output_root"] == str(root.resolve())
    assert manifest["formal_suite"]["path"] == str(suite.resolve())
    assert manifest["semantic_sha256"] == canonical_hash(
        {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    )
    assert [row["task"] for row in manifest["tasks"]] == list(TASK_ORDER)
    assert manifest["profiles"] == list(PROFILE_ORDER)
    expected = [(task, arm) for task in TASK_ORDER for arm in PROFILE_ORDER]
    assert [(cell["task"], cell["arm"]) for cell in manifest["cells"]] == expected
    assert len(manifest["cells"]) == 40
    for cell in manifest["cells"]:
        expected_path = root / "runs" / cell["task"] / cell["arm"] / "seed_17"
        assert cell["run_path"] == str(expected_path.resolve())
        assert cell["config_sha256"] == canonical_hash(cell["resolved_config"])
    for task in TASK_ORDER:
        configs = [cell["resolved_config"] for cell in manifest["cells"] if cell["task"] == task]
        shared = [
            (
                config["inputs"], config["targets"], config["cameras"],
                config["initialization"], config["hull"]
            )
            for config in configs
        ]
        assert all(identity == shared[0] for identity in shared)
        assert len({canonical_hash(config) for config in configs}) == 4
    runner._validate_manifest_for_test(manifest, resolution=8)


def test_manifest_binds_every_loaded_local_execution_module(manifest_bundle) -> None:
    _, _, manifest = manifest_bundle
    bundled = set(manifest["implementation"]["files_sha256"])
    required = {
        "experiments/run_soft_loss_v6_ablation10_256.py",
        "experiments/soft_loss_v3_registry.py",
        "experiments/soft_loss_v4_registry.py",
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
        "shadowforge_pipeline/soft_voxel_v4/__init__.py",
        "shadowforge_pipeline/soft_voxel_v4/config.py",
        "shadowforge_pipeline/soft_voxel_v4/diagnostics.py",
        "shadowforge_pipeline/soft_voxel_v4/gradients.py",
        "shadowforge_pipeline/soft_voxel_v4/losses.py",
        "shadowforge_pipeline/soft_voxel_v4/provenance.py",
        "shadowforge_pipeline/soft_voxel_v4/trainer_step.py",
        "shadowforge_pipeline/soft_voxel_v6/__init__.py",
        "shadowforge_pipeline/soft_voxel_v6/core.py",
        "shadowforge_pipeline/soft_voxel_v6/provenance.py",
        "shadowforge_pipeline/soft_voxel_v6/trainer.py",
    }
    assert len(required) == 24
    assert set(runner.SOURCE_FILES) == required
    assert bundled == required
    assert _loaded_v6_execution_sources() <= bundled


def test_static_revalidation_rejects_loaded_training_module_from_unbundled_path(
    manifest_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, manifest = manifest_bundle
    config = manifest["cells"][0]["resolved_config"]
    monkeypatch.setattr(runner, "__file__", str(Path(__file__).resolve()))
    with pytest.raises(ValueError, match="loaded|unbundled|source"):
        runner.revalidate_static_identities(config)


@pytest.mark.parametrize(
    "path",
    [
        ("formal_suite", "sha256"),
        ("specification", "sha256"),
        ("implementation", "bundle_sha256"),
        ("protocol", "density_threshold"),
        ("objective", "j_budget"),
        ("schedule", "updates"),
        ("decision_policy", "primary_contrast"),
        ("tasks", 0, "inputs", 0, "sha256"),
        ("cells", 0, "resolved_config", "cameras", "rotation_sha256"),
        ("cells", 0, "resolved_config", "hull", "sha256"),
        ("cells", 0, "resolved_config", "initialization", "logits_sha256"),
        ("cells", 0, "resolved_config", "environment", "torch"),
    ],
)
def test_manifest_deep_scientific_mutation_even_rehashed_is_rejected(
    manifest_bundle, path: tuple[object, ...]
) -> None:
    _, _, manifest = manifest_bundle
    forged = copy.deepcopy(manifest)
    cursor: object = forged
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    key = path[-1]
    old = cursor[key]  # type: ignore[index]
    cursor[key] = (old + 1 if type(old) in (int, float) else "f" * 64)  # type: ignore[index]
    for cell in forged["cells"]:
        cell["config_sha256"] = canonical_hash(cell["resolved_config"])
    forged["semantic_sha256"] = canonical_hash(
        {name: value for name, value in forged.items() if name != "semantic_sha256"}
    )
    with pytest.raises(ValueError):
        runner._validate_manifest_for_test(forged, resolution=8)


def test_independent_mirror_normalization_and_identity_hashes(manifest_bundle) -> None:
    _, _, manifest = manifest_bundle
    task = manifest["tasks"][0]
    assert task["targets"]["raw_flip_then_normalize"] is True
    assert task["targets"]["base_sha256"] != task["targets"]["mirror_sha256"]
    assert task["targets"]["mapped_six_sha256"] not in {
        task["targets"]["base_sha256"], task["targets"]["mirror_sha256"]
    }
    for name in ("rotation_sha256", "translation_sha256", "payload_sha256"):
        assert len(task["cameras"][name]) == 64
    for name in ("sha256",):
        assert len(task["hull"][name]) == 64
    assert len(task["initialization"]["logits_sha256"]) == 64
    assert len(task["initialization"]["step_zero_hard_sha256"]) == 64


def test_parser_is_exact_nonabbreviating_and_mutually_exclusive() -> None:
    parser = runner.build_parser()
    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    assert set(actions) == {"output_root", "formal_suite", "device", "dry_run", "resume"}
    assert actions["device"].default == "cuda:0"
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "/tmp/out", "--formal-suite", "/tmp/s"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--output-root", "/tmp/o", "--formal-suite", "/tmp/s", "--dry-run", "--resume"]
        )


def test_dry_run_keeps_root_absent_and_never_trains_subprocesses_or_touches_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _formal_suite(tmp_path, monkeypatch)
    root = tmp_path / "absent"
    calls = {"train": 0, "subprocess": 0, "cuda": 0}

    def forbidden(name: str):
        def fail(*args: object, **kwargs: object):
            calls[name] += 1
            raise AssertionError(name)
        return fail

    monkeypatch.setattr(runner, "train_cell", forbidden("train"))
    monkeypatch.setattr(subprocess, "run", forbidden("subprocess"))
    monkeypatch.setattr(runner.torch.cuda, "is_available", forbidden("cuda"))
    assert runner.main(
        [
            "--output-root", str(root), "--formal-suite", str(suite),
            "--device", "cuda:0", "--dry-run",
        ]
    ) == 0
    assert not root.exists() and calls == {"train": 0, "subprocess": 0, "cuda": 0}


def _fake_result(cell: dict[str, object], completed: int):
    return SimpleNamespace(
        run_dir=cell["run_path"], config_sha256=cell["config_sha256"],
        completed_updates=completed,
    )


def test_three_pass_launch_order_fresh_and_engineering_gate(manifest_bundle, monkeypatch) -> None:
    root, _, manifest = manifest_bundle
    launches: list[tuple[str, str, bool, int | None]] = []
    states: dict[tuple[str, str], tuple[str, int]] = {}

    def fake_train(config, *, resume=False, max_updates=None):
        launches.append((config["task"], config["arm"], resume, max_updates))
        states[(config["task"], config["arm"])] = (
            "completed" if max_updates == 3000 else "running", int(max_updates)
        )
        return _fake_result(
            next(cell for cell in manifest["cells"] if cell["resolved_config"] == config),
            int(max_updates),
        )

    monkeypatch.setattr(runner, "train_cell", fake_train)
    monkeypatch.setattr(
        runner,
        "_inspect_cell",
        lambda cell: states.get((cell["task"], cell["arm"])),
    )
    monkeypatch.setattr(runner, "validate_resumable_run", lambda config, expected_completed_updates: SimpleNamespace(completed_updates=expected_completed_updates))
    monkeypatch.setattr(runner, "validate_completed_run", lambda config: SimpleNamespace(completed_updates=3000))
    monkeypatch.setattr(runner, "validate_engineering_gate", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "revalidate_static_identities", lambda config: None)
    runner._execute_manifest_for_test(manifest, fresh=True)
    expected_two = [(task, arm, False, 2) for task in TASK_ORDER for arm in PROFILE_ORDER]
    expected_two_hundred = [
        ("connected_01", "base", True, 200),
        ("connected_01", "full", True, 200),
    ]
    expected_final = [(task, arm, True, 3000) for task in TASK_ORDER for arm in PROFILE_ORDER]
    assert launches == expected_two + expected_two_hundred + expected_final


def test_running_at_3000_is_resumed_once_to_publish_terminal_completion(
    manifest_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, manifest = manifest_bundle
    cell = manifest["cells"][0]
    marker = SimpleNamespace(completed_updates=3000)
    calls = {"train": [], "resumable": [], "completed": 0}
    monkeypatch.setattr(runner, "revalidate_static_identities", lambda config: None)
    monkeypatch.setattr(runner, "_inspect_cell", lambda observed: ("running", 3000))

    def train(config, *, resume=False, max_updates=None):
        calls["train"].append((resume, max_updates))
        return marker

    def resumable(config, *, expected_completed_updates=None):
        calls["resumable"].append(expected_completed_updates)
        return SimpleNamespace(completed_updates=expected_completed_updates)

    def completed(config):
        calls["completed"] += 1
        return marker

    monkeypatch.setattr(runner, "train_cell", train)
    monkeypatch.setattr(runner, "validate_resumable_run", resumable)
    monkeypatch.setattr(runner, "validate_completed_run", completed)
    result = runner._advance(cell, target=3000, fresh_mode=False)
    assert result is marker
    assert calls == {
        "train": [(True, 3000)],
        "resumable": [],
        "completed": 1,
    }


def test_engineering_gate_rejects_validate_then_load_checkpoint_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "running"
    checkpoint = run_dir / "checkpoints/step_0200.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(_torch_bytes(_engineering_checkpoint_payload(peak=11, seconds=0.1)))
    authenticated_digest = _sha(checkpoint)
    _write_v6_json(
        run_dir / "STATUS.json",
        {
            "schema": "pcraft_soft_loss_v6_status_v1",
            "state": "running",
            "config_sha256": "c" * 64,
            "completed_updates": 200,
        },
    )

    def validate_then_swap(config, *, expected_completed_updates):
        assert expected_completed_updates == 200
        checkpoint.write_bytes(
            _torch_bytes(_engineering_checkpoint_payload(peak=12, seconds=0.2))
        )
        return SimpleNamespace(
            run_dir=str(run_dir),
            config_sha256="c" * 64,
            completed_updates=200,
            checkpoint_sha256=authenticated_digest,
        )

    monkeypatch.setattr(runner, "validate_resumable_run", validate_then_swap)
    monkeypatch.setattr(
        runner.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=1024),
    )
    config = {"run": {"path": str(run_dir)}, "device": "cuda:0"}
    with pytest.raises(ValueError, match="checkpoint|digest|changed"):
        runner.validate_engineering_gate(config)


def test_engineering_gate_authenticates_step_200_from_completed_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "completed"
    checkpoint = run_dir / "checkpoints/step_0200.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(_torch_bytes(_engineering_checkpoint_payload(peak=13, seconds=0.1)))
    config_hash = "d" * 64
    artifact = {
        "schema": "pcraft_soft_loss_v6_artifacts_v1",
        "config_sha256": config_hash,
        "files_sha256": {"checkpoints/step_0200.pt": _sha(checkpoint)},
    }
    artifact_path = run_dir / "artifacts/artifacts.json"
    _write_v6_json(artifact_path, artifact)
    _write_v6_json(
        run_dir / "STATUS.json",
        {
            "schema": "pcraft_soft_loss_v6_status_v1",
            "state": "completed",
            "config_sha256": config_hash,
            "completed_updates": 3000,
            "artifacts_sha256": _sha(artifact_path),
        },
    )
    completed = runner.CompletedRun(
        str(run_dir), config_hash, 3000, _sha(artifact_path), {}
    )
    calls = {"completed": 0, "resumable": 0}

    def validate_completed(config):
        calls["completed"] += 1
        return completed

    def forbid_resumable(*args: object, **kwargs: object):
        calls["resumable"] += 1
        raise AssertionError("completed gate must not require running@200")

    monkeypatch.setattr(runner, "validate_completed_run", validate_completed)
    monkeypatch.setattr(runner, "validate_resumable_run", forbid_resumable)
    monkeypatch.setattr(
        runner.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=1024),
    )
    runner.validate_engineering_gate(
        {"run": {"path": str(run_dir)}, "device": "cuda:0"}
    )
    assert calls == {"completed": 1, "resumable": 0}


@pytest.mark.parametrize(
    ("arm", "current"),
    [("base", 400), ("full", 1200), ("base", 3000), ("full", 3000)],
)
def test_engineering_gate_authenticates_step_200_from_later_running_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    current: int,
) -> None:
    run_dir = tmp_path / arm / str(current)
    latest_digest = _write_running_engineering_chain(run_dir, current)
    _write_v6_json(
        run_dir / "STATUS.json",
        {
            "schema": "pcraft_soft_loss_v6_status_v1",
            "state": "running",
            "config_sha256": "e" * 64,
            "completed_updates": current,
        },
    )
    calls: list[int | None] = []

    def validate(config, *, expected_completed_updates=None):
        calls.append(expected_completed_updates)
        return runner.ResumableRun(
            str(run_dir), "e" * 64, current, latest_digest
        )

    monkeypatch.setattr(runner, "validate_resumable_run", validate)
    monkeypatch.setattr(
        runner.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=1024),
    )
    runner.validate_engineering_gate(
        {
            "run": {"path": str(run_dir)},
            "arm": arm,
            "device": "cuda:0",
        }
    )
    assert calls == [None]


def test_resume_skips_completed_resumes_running_and_starts_missing_later(
    manifest_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, manifest = manifest_bundle
    states: dict[tuple[str, str], tuple[str, int] | None] = {}
    for index, cell in enumerate(manifest["cells"]):
        identity = (cell["task"], cell["arm"])
        states[identity] = ("completed", 3000) if index == 0 else (("running", 2) if index == 1 else None)
    calls: list[tuple[str, str, bool, int]] = []

    monkeypatch.setattr(runner, "_inspect_cell", lambda cell: states[(cell["task"], cell["arm"])])

    def fake_train(config, *, resume=False, max_updates=None):
        calls.append((config["task"], config["arm"], resume, int(max_updates)))
        states[(config["task"], config["arm"])] = (
            "completed" if max_updates == 3000 else "running", int(max_updates)
        )
        return SimpleNamespace(completed_updates=max_updates)

    monkeypatch.setattr(runner, "train_cell", fake_train)
    monkeypatch.setattr(runner, "validate_resumable_run", lambda config, expected_completed_updates: SimpleNamespace(completed_updates=expected_completed_updates))
    monkeypatch.setattr(runner, "validate_completed_run", lambda config: SimpleNamespace(completed_updates=3000))
    monkeypatch.setattr(runner, "validate_engineering_gate", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "revalidate_static_identities", lambda config: None)
    runner._execute_manifest_for_test(manifest, fresh=False)
    assert not any(row[:2] == ("connected_01", "base") for row in calls)
    assert any(row[:3] == ("connected_01", "j_only", True) for row in calls)
    assert any(row[:3] == ("connected_01", "h_only", False) for row in calls)


def test_corrupt_existing_cell_stops_before_launch(manifest_bundle, monkeypatch) -> None:
    _, _, manifest = manifest_bundle
    calls = 0

    def corrupt(cell):
        raise ValueError("corrupt existing evidence")

    def train(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1

    monkeypatch.setattr(runner, "_inspect_cell", corrupt)
    monkeypatch.setattr(runner, "train_cell", train)
    with pytest.raises(ValueError, match="corrupt"):
        runner._execute_manifest_for_test(manifest, fresh=False)
    assert calls == 0


def test_fresh_publish_is_create_once_and_resume_requires_byte_identical_manifest(
    manifest_bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, suite, manifest = manifest_bundle
    runner._publish_manifest_for_test(manifest)
    path = root / "manifests/ablation10.json"
    before = path.read_bytes()
    runner._publish_manifest_for_test(manifest)
    assert path.read_bytes() == before
    path.write_bytes(before + b" ")
    with pytest.raises(ValueError, match="canonical|manifest|bytes"):
        runner._load_resume_manifest_for_test(root, suite, "cpu", resolution=8)


def test_source_input_and_spec_stale_bytes_fail_static_revalidation(manifest_bundle) -> None:
    _, suite, manifest = manifest_bundle
    config = manifest["cells"][0]["resolved_config"]
    runner.revalidate_static_identities(config)
    input_path = Path(config["inputs"]["raw_xyz"][0]["path"])
    original = input_path.read_bytes()
    input_path.write_bytes(original + b"stale")
    with pytest.raises(ValueError, match="input|hash|changed"):
        runner.revalidate_static_identities(config)
    input_path.write_bytes(original)
    suite.write_bytes(suite.read_bytes() + b"stale")
    with pytest.raises(ValueError, match="suite|hash|changed"):
        runner.revalidate_static_identities(config)


def test_real_cli_help_has_no_runpy_warning() -> None:
    completed = subprocess.run(
        [
            sys.executable, "-W", "error::RuntimeWarning", "-m",
            "experiments.run_soft_loss_v6_ablation10_256", "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "soft-loss-v6" in completed.stdout.lower()
    assert "RuntimeWarning" not in completed.stderr


def test_protected_59_file_inventory_remains_exact() -> None:
    inventory = PROJECT_ROOT / ".superpowers/sdd/2026-08-28-soft-loss-v6-256-ablation/v3-v4-v5-baseline.sha256"
    rows = [line.split(maxsplit=1) for line in inventory.read_text().splitlines() if line]
    assert len(rows) == 59
    for expected, relative in rows:
        assert hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() == expected
