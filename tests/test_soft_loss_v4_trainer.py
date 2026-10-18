"""Trainer lifecycle, transactions, checkpoints, and artifacts for soft-loss v4."""

from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Mapping

import cv2
import numpy as np
import pytest
import torch

import shadowforge_pipeline.soft_voxel_v4.trainer as trainer_module
from shadowforge_pipeline.soft_voxel_v3.contours import ContourBands
from shadowforge_pipeline.soft_voxel_v4.config import (
    bootstrap_v4_root,
    build_parser,
    canonical_trainer_argv,
    resolve_run_configuration,
)
from shadowforge_pipeline.soft_voxel_v4.losses import resolve_profile
from shadowforge_pipeline.soft_voxel_v4.provenance import (
    atomic_checkpoint,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from shadowforge_pipeline.soft_voxel_v4.trainer import (
    AuthenticatedCheckpointSnapshot,
    AuthenticatedCompletedRun,
    AuthenticatedResumableRun,
    AuthenticatedTransactionCheckpoint,
    ResumePreflightError,
    ResumeState,
    V4TrainingStepError,
    authenticate_transaction_checkpoint,
    build_checkpoint_payload,
    commit_training_transaction,
    restore_checkpoint_payload,
    restore_latest_transaction,
    run,
    validate_completed_run,
    validate_resumable_run,
)
from shadowforge_pipeline.soft_voxel_v4.trainer_step import (
    V4StepConfig,
    V4StepResult,
    train_one_v4_update,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECIFICATION = (
    PROJECT_ROOT
    / ".superpowers/sdd/2026-08-27-soft-loss-v4-implementation/spec.md"
)
THRESHOLDS = (0.04, 0.05, 0.06)


def _hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _semantic(payload: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(payload)
    result["semantic_sha256"] = _hash(result)
    return result


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "campaign"
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
    source = root / "source_suite.json"
    source.write_bytes(b"source\n")
    suite = _semantic(
        {
            "schema": "pcraft_soft_loss_v4_suite_v1",
            "status": "REGISTERED",
            "role": "development",
            "source_suite": {
                "path": str(source),
                "sha256": sha256_file(source),
                "semantic_sha256": "1" * 64,
            },
            "tasks": [
                {
                    "case_id": "toy_task",
                    "inputs": [
                        {
                            "axis": axis,
                            "source_path": str(path),
                            "resolved_path": str(path),
                            "sha256": sha256_file(path),
                        }
                        for axis, path in zip("xyz", shadows)
                    ],
                }
            ],
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
    suite_path = root / "manifests" / "development_suite.json"
    _write_json(suite_path, suite)
    specification = root / "spec.md"
    specification.write_bytes(SPECIFICATION.read_bytes())
    return {
        "root": root,
        "shadows": shadows,
        "suite": suite_path,
        "specification": specification,
    }


def _args(
    fixture: Mapping[str, object],
    *,
    run_id: str = "smoke/toy_task/l12_bal/primary_0",
    resume: bool = False,
) -> argparse.Namespace:
    argv = [
        "--run-kind",
        "experiment",
        "--device",
        "cpu",
        "--output-root",
        str(fixture["root"]),
        "--run-id",
        run_id,
        "--stage",
        "smoke",
        "--task-id",
        "toy_task",
        "--replicate-role",
        "primary",
        "--replicate-index",
        "0",
        "--steps",
        "200",
        "--learning-rate",
        "0.007",
        "--shadow-files",
        *(str(path) for path in fixture["shadows"]),
        "--resolution",
        "128",
        "--export-resolution",
        "128",
        "--profile",
        "l12_bal",
        "--suite-manifest",
        str(fixture["suite"]),
        "--specification",
        str(fixture["specification"]),
    ]
    if resume:
        argv.append("--resume")
    argv.extend(("--export-final-step", "--soft-only-eval"))
    return build_parser().parse_args(argv)


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": [],
    }


def _optimizer_pair() -> tuple[torch.Tensor, torch.optim.Optimizer]:
    logits = torch.linspace(-1.2, 1.3, 8, dtype=torch.float64).reshape(
        1, 1, 2, 2, 2
    ).requires_grad_()
    return logits, torch.optim.Adam([logits], lr=0.007)


def _base_result(completed: int, total: int = 200) -> V4StepResult:
    return V4StepResult(
        global_step_before=completed - 1,
        completed_updates_after=completed,
        progress=completed / float(total),
        slot_component=None,
        slot_endpoint=None,
        slot_multiplier=0.0,
        loss_base=0.5,
        loss_anchor=None,
        loss_auxiliary=None,
        auxiliary_scale=0.0,
        pre_clip_gradient_norm=0.25,
        post_clip_gradient_norm=0.25,
        slot_audit=None,
    )


def _history_row(result: V4StepResult) -> dict[str, object]:
    return {"schema": "pcraft_soft_loss_v4_history_row_v1", **asdict(result)}


def _checkpoint(
    config: Mapping[str, object],
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    completed: int,
) -> dict[str, object]:
    return build_checkpoint_payload(
        logits=logits,
        optimizer=optimizer,
        completed_updates=completed,
        last_result=None if completed == 0 else _base_result(completed),
        rng_state=_rng_state(),
        step_times=() if completed == 0 else (0.0,) * completed,
        peak_cuda_bytes=0,
        slot_audits=(),
        resolved_config=config,
    )


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for one, two in zip(left, right):
            _assert_nested_equal(one, two)
    else:
        assert left == right


def _tree(root: Path) -> dict[str, tuple[str, object]]:
    result: dict[str, tuple[str, object]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            result[relative] = ("file", path.read_bytes())
        elif path.is_dir():
            result[relative] = ("directory", None)
    return result


class _ToyProtocol:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def render_all(self, density: torch.Tensor) -> torch.Tensor:
        view = density[0, 0].mean(dim=0)
        return view.unsqueeze(0).repeat(6, 1, 1)

    @staticmethod
    def binary_mask(projection: torch.Tensor) -> torch.Tensor:
        maximum = projection.detach().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return projection >= 0.8 * maximum


def _toy_runtime(resolved, device: torch.device, profile):
    logits, optimizer = _optimizer_pair()
    logits = logits.to(device).detach().requires_grad_()
    optimizer = torch.optim.Adam([logits], lr=0.007)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64, device=device)
    targets = target.unsqueeze(0).repeat(6, 1, 1)
    bands = None
    if profile.use_dic:
        binary = targets >= 0.5
        bands = ContourBands(inner=binary, outer=~binary)
    return trainer_module._Runtime(
        logits=logits,
        optimizer=optimizer,
        protocol=_ToyProtocol(device),
        targets=targets,
        bands=bands,
    )


class _Interruption(BaseException):
    pass


def _install_toy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trainer_module, "_build_runtime", _toy_runtime)
    monkeypatch.setattr(trainer_module, "_timer", lambda: 0.0)


def _running_run(
    fixture: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    *,
    interrupted_at: int,
    run_id: str = "smoke/toy_task/l12_bal/primary_0",
) -> tuple[Path, dict[str, object]]:
    """Create an authentic running run whose pointer is at update 0 or 100."""

    _install_toy(monkeypatch)
    args = _args(fixture, run_id=run_id)
    config = resolve_run_configuration(args).config

    def interrupt(completed: int) -> None:
        if completed == interrupted_at:
            raise _Interruption

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_Interruption):
        run(args)
    monkeypatch.setattr(trainer_module, "_post_update_hook", lambda unused: None)
    return Path(fixture["root"]) / run_id, config


def _forbid_runtime_or_restore(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*unused: object, **unused_keywords: object) -> object:
        raise AssertionError("read-only resumable authentication entered runtime/restore")

    monkeypatch.setattr(trainer_module, "_build_runtime", forbidden)
    monkeypatch.setattr(trainer_module, "restore_latest_transaction", forbidden)
    monkeypatch.setattr(trainer_module, "restore_checkpoint_payload", forbidden)
    monkeypatch.setattr(trainer_module, "_restore_rng_state", forbidden)


def _rewrite_status_as_running(run_dir: Path) -> None:
    status = json.loads((run_dir / "STATUS.json").read_text())
    atomic_json(
        run_dir / "STATUS.json",
        {
            "schema": "pcraft_soft_loss_v4_status_v1",
            "state": "running",
            "config_hash": status["config_hash"],
            "completed_updates": 0,
        },
    )


def _forge_step_one_pointer(run_dir: Path, config: Mapping[str, object]) -> None:
    """Publish a self-consistent 0 -> 1 chain without using the guarded publisher."""

    pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
    generation = run_dir / "transactions/step_00000001"
    generation.mkdir()
    logits, optimizer = _optimizer_pair()
    atomic_checkpoint(
        generation / "checkpoint.pt", _checkpoint(config, logits, optimizer, 1)
    )
    _write_json(generation / "history_segment.jsonl", _history_row(_base_result(1)))
    manifest = {
        "schema": "pcraft_soft_loss_v4_transaction_v1",
        "config_hash": canonical_hash(config),
        "completed_updates": 1,
        "checkpoint_sha256": sha256_file(generation / "checkpoint.pt"),
        "history_segment_sha256": sha256_file(
            generation / "history_segment.jsonl"
        ),
        "history_interval": {
            "first_completed_update": 1,
            "last_completed_update": 1,
            "row_count": 1,
        },
        "previous_transaction": pointer["transaction"],
        "previous_transaction_sha256": pointer["manifest_sha256"],
    }
    _write_json(generation / "transaction_manifest.json", manifest)
    atomic_json(
        run_dir / "checkpoint_latest.json",
        {
            "schema": "pcraft_soft_loss_v4_checkpoint_pointer_v1",
            "transaction": "transactions/step_00000001",
            "manifest_sha256": sha256_file(
                generation / "transaction_manifest.json"
            ),
        },
    )


def test_checkpoint_round_trip_restores_full_config_rng_optimizer_and_result(
    tmp_path: Path,
) -> None:
    """Break caught: restore trusts a digest alone or mutates before full validation."""

    fixture = _fixture(tmp_path)
    config = resolve_run_configuration(_args(fixture)).config
    logits, optimizer = _optimizer_pair()
    logits.square().sum().backward()
    optimizer.step()
    payload = _checkpoint(config, logits, optimizer, 1)
    assert set(payload) == {
        "schema",
        "resolved_config",
        "config_hash",
        "completed_updates",
        "logits",
        "optimizer",
        "last_result",
        "rng_state",
        "step_times",
        "peak_cuda_bytes",
        "slot_audits",
    }
    path = tmp_path / "checkpoint.pt"
    atomic_checkpoint(path, payload)
    restored_logits, restored_optimizer = _optimizer_pair()
    state = restore_checkpoint_payload(
        path,
        expected_config=config,
        logits=restored_logits,
        optimizer=restored_optimizer,
    )
    assert state.completed_updates == 1
    assert state.last_result == _base_result(1)
    assert state.step_times == (0.0,)
    assert torch.equal(restored_logits, logits)
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())


@pytest.mark.parametrize(
    "mutation",
    ["extra", "config", "last", "times", "rng", "optimizer"],
)
def test_checkpoint_corruption_rejects_before_live_mutation(
    tmp_path: Path, mutation: str
) -> None:
    """Break caught: corrupt parity/types partially alter logits, optimizer, or RNG."""

    fixture = _fixture(tmp_path / mutation)
    config = resolve_run_configuration(_args(fixture)).config
    logits, optimizer = _optimizer_pair()
    payload = _checkpoint(config, logits, optimizer, 1)
    if mutation == "extra":
        payload["extra"] = True
    elif mutation == "config":
        payload["resolved_config"] = copy.deepcopy(config)
        payload["resolved_config"]["seed"] = 18
    elif mutation == "last":
        payload["last_result"]["global_step_before"] = 1
    elif mutation == "times":
        payload["step_times"] = []
    elif mutation == "rng":
        payload["rng_state"] = {"python": None}
    else:
        payload["optimizer"] = {"forged": True}
    path = tmp_path / f"bad-{mutation}.pt"
    atomic_checkpoint(path, payload)
    live_logits, live_optimizer = _optimizer_pair()
    logits_before = live_logits.detach().clone()
    optimizer_before = copy.deepcopy(live_optimizer.state_dict())
    rng_before = torch.random.get_rng_state().clone()
    with pytest.raises((ValueError, ResumePreflightError)):
        restore_checkpoint_payload(
            path,
            expected_config=config,
            logits=live_logits,
            optimizer=live_optimizer,
        )
    assert torch.equal(live_logits, logits_before)
    _assert_nested_equal(live_optimizer.state_dict(), optimizer_before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_auxiliary_audit_prefix_reconstructs_every_eligible_slot_and_phase() -> None:
    """Break caught: resume validates audit count only and resets the four-slot phase."""

    profile = resolve_profile("dic_only")
    logits, optimizer = _optimizer_pair()
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    targets = target.unsqueeze(0).repeat(6, 1, 1)
    bands = ContourBands(inner=targets >= 0.5, outer=targets < 0.5)
    result = train_one_v4_update(
        logits=logits,
        optimizer=optimizer,
        protocol=_ToyProtocol(torch.device("cpu")),
        targets=targets,
        bands=bands,
        profile=profile,
        global_step_before=120,
        total_steps=200,
        config=V4StepConfig(1.0, 1.0, 0.04, 0.04),
    )
    assert result.slot_audit is not None
    trainer_module._validate_slot_audit_prefix(
        (result.slot_audit,), profile=profile, total_steps=200, completed_updates=121
    )
    shifted = copy.deepcopy(asdict(result.slot_audit))
    shifted["global_step_before"] = 121
    with pytest.raises(ValueError, match="audit|slot|step"):
        trainer_module._validate_slot_audit_prefix(
            (shifted,), profile=profile, total_steps=200, completed_updates=121
        )


def test_transaction_step_zero_boundary_and_restore_are_append_only(tmp_path: Path) -> None:
    """Break caught: a checkpoint/history pair is mixed across predecessor boundaries."""

    fixture = _fixture(tmp_path)
    config = resolve_run_configuration(_args(fixture)).config
    run_dir = tmp_path / "transactions"
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    zero = commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(config, logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    pointer0 = json.loads((run_dir / "checkpoint_latest.json").read_text())
    assert zero.relative_to(run_dir).as_posix() == "transactions/step_00000000"
    one = commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(config, logits, optimizer, 100),
        history_segment=tuple(
            _history_row(_base_result(completed)) for completed in range(1, 101)
        ),
        previous_transaction_sha256=pointer0["manifest_sha256"],
    )
    manifest = json.loads((one / "transaction_manifest.json").read_text())
    assert set(manifest) == {
        "schema",
        "config_hash",
        "completed_updates",
        "checkpoint_sha256",
        "history_segment_sha256",
        "history_interval",
        "previous_transaction",
        "previous_transaction_sha256",
    }
    assert manifest["history_interval"] == {
        "first_completed_update": 1,
        "last_completed_update": 100,
        "row_count": 100,
    }
    live_logits, live_optimizer = _optimizer_pair()
    state = restore_latest_transaction(
        run_dir,
        expected_config=config,
        logits=live_logits,
        optimizer=live_optimizer,
    )
    assert state.completed_updates == 100
    assert state.committed_history == tuple(
        _history_row(_base_result(completed)) for completed in range(1, 101)
    )
    assert state.committed_transaction == "transactions/step_00000100"


def test_transaction_publication_rejects_noncadence_generation(tmp_path: Path) -> None:
    """Break caught: a running pointer exposes an unauthorized update-1 generation."""

    fixture = _fixture(tmp_path)
    config = resolve_run_configuration(_args(fixture)).config
    run_dir = tmp_path / "noncadence"
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(config, logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
    before = (run_dir / "checkpoint_latest.json").read_bytes()
    with pytest.raises(ValueError, match="cadence|schedule|interval"):
        commit_training_transaction(
            run_dir,
            checkpoint_payload=_checkpoint(config, logits, optimizer, 1),
            history_segment=(_history_row(_base_result(1)),),
            previous_transaction_sha256=pointer["manifest_sha256"],
        )
    assert (run_dir / "checkpoint_latest.json").read_bytes() == before


def test_transaction_publication_requires_every_frozen_commit_generation(
    tmp_path: Path,
) -> None:
    """Break caught: pointer publication accepts an on-cadence 0-to-200 skip."""

    fixture = _fixture(tmp_path)
    config = resolve_run_configuration(_args(fixture)).config
    run_dir = tmp_path / "skipped_schedule"
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(config, logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
    before = (run_dir / "checkpoint_latest.json").read_bytes()
    with pytest.raises(ValueError, match="schedule|generation|interval|cadence"):
        commit_training_transaction(
            run_dir,
            checkpoint_payload=_checkpoint(config, logits, optimizer, 200),
            history_segment=tuple(
                _history_row(_base_result(completed)) for completed in range(1, 201)
            ),
            previous_transaction_sha256=pointer["manifest_sha256"],
        )
    assert (run_dir / "checkpoint_latest.json").read_bytes() == before


def test_stale_predecessor_or_corrupt_chain_never_moves_pointer(tmp_path: Path) -> None:
    """Break caught: concurrent stale commits roll back or a rehashed cycle is accepted."""

    fixture = _fixture(tmp_path)
    config = resolve_run_configuration(_args(fixture)).config
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(config, logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    pointer0 = json.loads((run_dir / "checkpoint_latest.json").read_text())
    commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(config, logits, optimizer, 100),
        history_segment=tuple(
            _history_row(_base_result(completed)) for completed in range(1, 101)
        ),
        previous_transaction_sha256=pointer0["manifest_sha256"],
    )
    pointer1 = (run_dir / "checkpoint_latest.json").read_bytes()
    with pytest.raises(ValueError, match="predecessor|pointer"):
        commit_training_transaction(
            run_dir,
            checkpoint_payload=_checkpoint(config, logits, optimizer, 100),
            history_segment=tuple(
                _history_row(_base_result(completed)) for completed in range(1, 101)
            ),
            previous_transaction_sha256=pointer0["manifest_sha256"],
        )
    assert (run_dir / "checkpoint_latest.json").read_bytes() == pointer1

    generation = run_dir / "transactions" / "step_00000100"
    checkpoint = generation / "checkpoint.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    live_logits, live_optimizer = _optimizer_pair()
    with pytest.raises(ValueError, match="checkpoint|hash"):
        restore_latest_transaction(
            run_dir,
            expected_config=config,
            logits=live_logits,
            optimizer=live_optimizer,
        )


@pytest.mark.parametrize(
    "component", ["generation", "checkpoint.pt", "history_segment.jsonl", "transaction_manifest.json"]
)
def test_transaction_chain_rejects_symlinked_components(
    tmp_path: Path, component: str
) -> None:
    """Break caught: transaction authentication follows a link outside the run."""

    fixture = _fixture(tmp_path / component.replace(".", "_"))
    config = resolve_run_configuration(_args(fixture)).config
    run_dir = tmp_path / f"run-{component.replace('.', '_')}"
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    generation = commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(config, logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    outside = tmp_path / f"outside-{component.replace('.', '_')}"
    if component == "generation":
        generation.rename(outside)
        generation.symlink_to(outside, target_is_directory=True)
    else:
        source = generation / component
        source.rename(outside)
        source.symlink_to(outside)
    live_logits, live_optimizer = _optimizer_pair()
    with pytest.raises(ValueError, match="symlink"):
        restore_latest_transaction(
            run_dir,
            expected_config=config,
            logits=live_logits,
            optimizer=live_optimizer,
        )


def test_real_toy_run_publishes_exact_artifacts_and_public_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: terminal validation trusts STATUS without chain/artifact evidence."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)
    metrics = run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    assert set(metrics) == {
        "schema",
        "config_hash",
        "thresholds",
        "task_iou_by_threshold",
        "worst_threshold_iou",
        "threshold_iou_range",
        "task_mscd_px",
        "task_p95_symmetric_distance_px",
        "invalid_task_metric",
    }
    common = {
        "COMMAND.json",
        "run_manifest.json",
        "environment.json",
        "history.jsonl",
        "occupancy_probability.npy",
        "occupancy_binary_tau_0p04.npy",
        "occupancy_binary_tau_0p05.npy",
        "occupancy_binary_tau_0p06.npy",
        "metrics.json",
        "gradient_audit.json",
        "resource_metrics.json",
        "checkpoint_latest.json",
        "checkpoint_final.pt",
        "artifact_manifest.json",
        "STATUS.json",
    }
    common.update(
        f"binary_tau_{label}_view_{view}.png"
        for label in ("0p04", "0p05", "0p06")
        for view in range(6)
    )
    assert common == {path.name for path in run_dir.iterdir() if path.is_file()}
    assert not (run_dir / "checkpoint_step_00001600.pt").exists()
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert len(history) == 200
    assert all("effective_loss" not in row for row in history)
    assert [row["completed_updates_after"] for row in history] == list(range(1, 201))
    audit = json.loads((run_dir / "gradient_audit.json").read_text())
    assert audit["samples"] == []
    assert audit["engineering_gate"] is None
    assert audit["redundancy_gate"] is None
    resources = json.loads((run_dir / "resource_metrics.json").read_text())
    assert resources["all_updates"]["count"] == 200
    assert resources["updates_21_to_end"]["count"] == 180
    authenticated = validate_completed_run(run_dir)
    assert isinstance(authenticated, AuthenticatedCompletedRun)
    assert authenticated.config_hash == canonical_hash(authenticated.config)
    assert [item.completed_updates for item in authenticated.transaction_checkpoints] == [
        0,
        100,
        200,
    ]
    snapshot = authenticate_transaction_checkpoint(run_dir, completed_updates=100)
    assert isinstance(snapshot, AuthenticatedCheckpointSnapshot)
    assert snapshot.identity.completed_updates == 100
    assert snapshot.checkpoint.payload == (
        run_dir / "transactions/step_00000100/checkpoint.pt"
    ).read_bytes()


def test_completed_reuse_is_zero_write_and_rejects_tampered_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: completed skip repairs or ignores changed immutable evidence."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)
    run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    before = _tree(run_dir)
    resume_args = _args(fixture, resume=True)
    assert run(resume_args) == json.loads((run_dir / "metrics.json").read_text())
    assert _tree(run_dir) == before

    metrics = run_dir / "metrics.json"
    metrics.write_bytes(metrics.read_bytes() + b"tampered")
    tampered = _tree(run_dir)
    with pytest.raises(ValueError, match="artifact|hash"):
        run(resume_args)
    assert _tree(run_dir) == tampered


def test_completed_authentication_rejects_self_consistently_rehashed_command_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: immutable COMMAND vectors are treated as unauthenticated strings."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)
    run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    command = json.loads((run_dir / "COMMAND.json").read_text())
    command["fresh_argv"] = ["forged", "fresh"]
    command["resume_argv"] = ["forged", "resume"]
    _write_json(run_dir / "COMMAND.json", command)
    artifact_manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    artifact_manifest["artifacts_sha256"]["COMMAND.json"] = sha256_file(
        run_dir / "COMMAND.json"
    )
    _write_json(run_dir / "artifact_manifest.json", artifact_manifest)
    status = json.loads((run_dir / "STATUS.json").read_text())
    status["artifact_manifest_sha256"] = sha256_file(
        run_dir / "artifact_manifest.json"
    )
    _write_json(run_dir / "STATUS.json", status)
    with pytest.raises(ValueError, match="COMMAND|argv|canonical"):
        validate_completed_run(run_dir)


@pytest.mark.parametrize(
    ("interrupted_at", "committed_at"), [(1, 0), (99, 0), (100, 100), (101, 100)]
)
def test_interruption_resume_preserves_history_logits_optimizer_rng_audits_and_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_at: int,
    committed_at: int,
) -> None:
    """Break caught: uncommitted updates survive or resume shifts global slot identity."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    baseline_args = _args(fixture, run_id="baseline")
    run(baseline_args)
    resumed_args = _args(fixture, run_id=f"resume_{interrupted_at}")

    def interrupt(completed: int) -> None:
        if completed == interrupted_at:
            raise _Interruption

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_Interruption):
        run(resumed_args)
    resumed_dir = Path(fixture["root"]) / resumed_args.run_id
    assert json.loads((resumed_dir / "STATUS.json").read_text())["state"] == "running"
    assert json.loads((resumed_dir / "checkpoint_latest.json").read_text())[
        "transaction"
    ].endswith(f"step_{committed_at:08d}")
    monkeypatch.setattr(trainer_module, "_post_update_hook", lambda unused: None)
    resumed_args.resume = True
    run(resumed_args)
    baseline_dir = Path(fixture["root"]) / baseline_args.run_id
    assert (baseline_dir / "history.jsonl").read_bytes() == (
        resumed_dir / "history.jsonl"
    ).read_bytes()
    baseline_audit = json.loads((baseline_dir / "gradient_audit.json").read_text())
    resumed_audit = json.loads((resumed_dir / "gradient_audit.json").read_text())
    assert baseline_audit.pop("config_hash") != resumed_audit.pop("config_hash")
    assert baseline_audit == resumed_audit
    base_checkpoint = torch.load(
        baseline_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False
    )
    resumed_checkpoint = torch.load(
        resumed_dir / "checkpoint_final.pt", map_location="cpu", weights_only=False
    )
    for key in ("logits", "optimizer", "rng_state", "slot_audits"):
        _assert_nested_equal(base_checkpoint[key], resumed_checkpoint[key])


def test_scientific_failure_is_terminal_and_pending_diagnostic_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a computation failure is resumed or a pending diagnostic is ignored."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)
    original = trainer_module.train_one_v4_update

    def fail(**unused):
        raise V4TrainingStepError("nonfinite injected")

    monkeypatch.setattr(trainer_module, "train_one_v4_update", fail)
    with pytest.raises(V4TrainingStepError):
        run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    status = json.loads((run_dir / "STATUS.json").read_text())
    assert status == {
        "schema": "pcraft_soft_loss_v4_status_v1",
        "state": "failed",
        "config_hash": status["config_hash"],
        "completed_updates": 0,
        "diagnostic": "failure.json",
    }
    assert json.loads((run_dir / "failure.json").read_text())["completed_updates"] == 0
    before = _tree(run_dir)
    resume = _args(fixture, resume=True)
    monkeypatch.setattr(trainer_module, "train_one_v4_update", original)
    with pytest.raises(ValueError, match="failed"):
        run(resume)
    assert _tree(run_dir) == before


def test_dependency_toctou_before_pointer_leaves_previous_commit_authoritative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: dependency changes after generation write but before pointer exposure."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)
    changed = {"done": False}

    def mutate(run_dir: Path, completed: int) -> None:
        if completed == 100 and not changed["done"]:
            changed["done"] = True
            Path(fixture["shadows"][0]).write_bytes(b"changed-after-snapshot")

    monkeypatch.setattr(trainer_module, "_before_pointer_publication_hook", mutate)
    with pytest.raises(ValueError, match="dependency|snapshot|changed|hash"):
        run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
    assert pointer["transaction"] == "transactions/step_00000000"
    assert json.loads((run_dir / "STATUS.json").read_text())["state"] == "running"


def test_dependency_toctou_after_artifact_publication_prevents_completed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: terminal STATUS publishes after a bound input changes post-manifest."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)

    def mutate_dependency(unused_run_dir: Path) -> None:
        Path(fixture["shadows"][0]).write_bytes(b"changed-after-artifact")

    monkeypatch.setattr(
        trainer_module, "_after_artifact_publication_hook", mutate_dependency
    )
    with pytest.raises(ValueError, match="dependency|snapshot|changed|hash"):
        run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    assert (run_dir / "artifact_manifest.json").is_file()
    assert json.loads((run_dir / "STATUS.json").read_text())["state"] == "running"


def test_dependency_toctou_rehashes_full_chain_before_pointer_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a changed ancestor is ignored while a new pointer is exposed."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)
    changed = {"done": False}

    def mutate_ancestor(run_dir: Path, completed: int) -> None:
        if completed == 100 and not changed["done"]:
            changed["done"] = True
            checkpoint = run_dir / "transactions/step_00000000/checkpoint.pt"
            checkpoint.write_bytes(checkpoint.read_bytes() + b"changed-ancestor")

    monkeypatch.setattr(trainer_module, "_before_pointer_publication_hook", mutate_ancestor)
    with pytest.raises((ValueError, ResumePreflightError), match="transaction|checkpoint|hash"):
        run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
    assert pointer["transaction"] == "transactions/step_00000000"
    assert json.loads((run_dir / "STATUS.json").read_text())["state"] == "running"


def test_pending_completed_terminalization_authenticates_without_runtime_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: valid final artifacts are retrained instead of terminalized in place."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)
    expected_metrics = run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    completed_status = json.loads((run_dir / "STATUS.json").read_text())
    atomic_json(
        run_dir / "STATUS.json",
        {
            "schema": "pcraft_soft_loss_v4_status_v1",
            "state": "running",
            "config_hash": completed_status["config_hash"],
            "completed_updates": 0,
        },
    )
    before = _tree(run_dir)

    def forbidden_runtime(*unused, **unused_keywords):
        raise AssertionError("pending terminalization must precede runtime construction")

    monkeypatch.setattr(trainer_module, "_build_runtime", forbidden_runtime)
    resume = _args(fixture, resume=True)
    assert run(resume) == expected_metrics
    after = _tree(run_dir)
    changed = {path for path in before if before[path] != after[path]}
    assert changed == {"STATUS.json"}
    assert json.loads((run_dir / "STATUS.json").read_text()) == completed_status
    validate_completed_run(run_dir)


def test_pending_terminalization_rejects_rehashed_static_manifest_before_status_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: forged static evidence is terminalized before semantic validation."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)
    run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    completed_status = json.loads((run_dir / "STATUS.json").read_text())
    atomic_json(
        run_dir / "STATUS.json",
        {
            "schema": "pcraft_soft_loss_v4_status_v1",
            "state": "running",
            "config_hash": completed_status["config_hash"],
            "completed_updates": 0,
        },
    )
    environment = json.loads((run_dir / "environment.json").read_text())
    environment["schema"] = "forged_environment"
    _write_json(run_dir / "environment.json", environment)
    artifact_manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    artifact_manifest["artifacts_sha256"]["environment.json"] = sha256_file(
        run_dir / "environment.json"
    )
    _write_json(run_dir / "artifact_manifest.json", artifact_manifest)
    before = _tree(run_dir)

    def forbidden_runtime(*unused, **unused_keywords):
        raise AssertionError("invalid pending evidence must fail before runtime")

    monkeypatch.setattr(trainer_module, "_build_runtime", forbidden_runtime)
    with pytest.raises(ValueError, match="environment|manifest|identity|schema"):
        run(_args(fixture, resume=True))
    assert _tree(run_dir) == before


def test_convergence_run_retains_authenticated_1600_and_1800_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a 2,000-update endpoint omits or substitutes convergence evidence."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture, run_id="confirmation/toy_task/l12_bal/primary_0")
    args.stage = "confirmation"
    args.steps = 2000
    run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    completed = validate_completed_run(run_dir)
    assert [item.completed_updates for item in completed.transaction_checkpoints] == list(
        range(0, 2001, 100)
    )
    by_step = {
        item.completed_updates: item for item in completed.transaction_checkpoints
    }
    for update in (1600, 1800):
        retained = run_dir / f"checkpoint_step_{update:08d}.pt"
        assert sha256_file(retained) == by_step[update].checkpoint_sha256
        authenticated = authenticate_transaction_checkpoint(
            run_dir, completed_updates=update
        )
        assert authenticated.checkpoint.payload == retained.read_bytes()


def test_reserved_artifact_symlink_rejects_resume_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: dangling reserved links evade preflight and are overwritten."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)

    def interrupt(completed: int) -> None:
        if completed == 1:
            raise _Interruption

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_Interruption):
        run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    link = run_dir / "artifact_manifest.json"
    link.symlink_to(tmp_path / "missing-outside.json")
    before = _tree(run_dir)
    resume = _args(fixture, resume=True)
    monkeypatch.setattr(trainer_module, "_post_update_hook", lambda unused: None)
    with pytest.raises(ValueError, match="symlink|reserved|artifact"):
        run(resume)
    assert _tree(run_dir) == before


@pytest.mark.parametrize(("interrupted_at", "committed_at"), [(1, 0), (100, 100)])
def test_public_resumable_authenticates_running_step_zero_or_hundred_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_at: int,
    committed_at: int,
) -> None:
    """Break caught: Task 5 preflight constructs runtime or trusts running STATUS progress."""

    fixture = _fixture(tmp_path)
    run_dir, config = _running_run(
        fixture, monkeypatch, interrupted_at=interrupted_at
    )
    before = _tree(run_dir)
    rng_before = torch.random.get_rng_state().clone()
    _forbid_runtime_or_restore(monkeypatch)
    authenticated = validate_resumable_run(run_dir, expected_config=config)
    assert isinstance(authenticated, AuthenticatedResumableRun)
    assert authenticated.run_dir == str(run_dir)
    assert authenticated.output_root_identity == config["output_root_identity"]
    assert authenticated.config == config
    assert authenticated.config_hash == canonical_hash(config)
    assert authenticated.status_sha256 == sha256_file(run_dir / "STATUS.json")
    assert authenticated.completed_updates == committed_at
    assert authenticated.latest_transaction.completed_updates == committed_at
    assert authenticated.latest_transaction.transaction.endswith(
        f"step_{committed_at:08d}"
    )
    assert authenticated.pending_terminalization is None
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert _tree(run_dir) == before


@pytest.mark.parametrize(
    "mutation", ["config", "argv", "dependency", "noncadence", "orphan", "symlink"]
)
def test_public_resumable_rejects_wrong_identity_or_chain_with_zero_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: pre-spawn authentication accepts forged identity or transaction state."""

    fixture = _fixture(tmp_path / "subject")
    run_dir, config = _running_run(fixture, monkeypatch, interrupted_at=1)
    expected = config
    if mutation == "config":
        other = _fixture(tmp_path / "other-root")
        expected = resolve_run_configuration(_args(other)).config
    elif mutation == "argv":
        command = json.loads((run_dir / "COMMAND.json").read_text())
        command["fresh_argv"] = ["forged", "fresh"]
        command["resume_argv"] = ["forged", "resume"]
        _write_json(run_dir / "COMMAND.json", command)
    elif mutation == "dependency":
        Path(fixture["shadows"][0]).write_bytes(b"stale-after-run")
    elif mutation == "noncadence":
        _forge_step_one_pointer(run_dir, config)
    elif mutation == "orphan":
        (run_dir / "transactions/step_00000099").mkdir()
    else:
        (run_dir / "artifact_manifest.json").symlink_to(
            tmp_path / "outside-artifact-manifest.json"
        )
    before = _tree(run_dir)
    _forbid_runtime_or_restore(monkeypatch)
    with pytest.raises((ValueError, ResumePreflightError)):
        validate_resumable_run(run_dir, expected_config=expected)
    assert _tree(run_dir) == before


@pytest.mark.parametrize("pending", ["completed", "failed"])
def test_public_resumable_authenticates_pending_terminal_evidence_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pending: str,
) -> None:
    """Break caught: a pending terminal classification is returned before strict evidence auth."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)
    config = resolve_run_configuration(args).config
    if pending == "completed":
        run(args)
    else:
        original = trainer_module.train_one_v4_update

        def fail(**unused: object) -> object:
            raise V4TrainingStepError("pending failure")

        monkeypatch.setattr(trainer_module, "train_one_v4_update", fail)
        with pytest.raises(V4TrainingStepError):
            run(args)
        monkeypatch.setattr(trainer_module, "train_one_v4_update", original)
    run_dir = Path(fixture["root"]) / args.run_id
    _rewrite_status_as_running(run_dir)
    before = _tree(run_dir)
    _forbid_runtime_or_restore(monkeypatch)
    authenticated = validate_resumable_run(run_dir, expected_config=config)
    assert authenticated.pending_terminalization == pending
    assert authenticated.completed_updates == (config["steps"] if pending == "completed" else 0)
    assert authenticated.latest_transaction.completed_updates == authenticated.completed_updates
    assert _tree(run_dir) == before


@pytest.mark.parametrize("corruption", ["partial_completed", "corrupt_failed", "dual"])
def test_public_resumable_rejects_corrupt_or_dual_pending_evidence_zero_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    """Break caught: partial, malformed, or ambiguous terminal evidence is repaired or accepted."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)
    config = resolve_run_configuration(args).config
    if corruption == "partial_completed":
        run_dir, config = _running_run(fixture, monkeypatch, interrupted_at=1)
        atomic_json(
            run_dir / "artifact_manifest.json",
            {
                "schema": "pcraft_soft_loss_v4_artifacts_v1",
                "config_hash": canonical_hash(config),
                "artifacts_sha256": {},
            },
        )
    elif corruption == "corrupt_failed":
        def fail(**unused: object) -> object:
            raise V4TrainingStepError("pending failure")

        monkeypatch.setattr(trainer_module, "train_one_v4_update", fail)
        with pytest.raises(V4TrainingStepError):
            run(args)
        run_dir = Path(fixture["root"]) / args.run_id
        _rewrite_status_as_running(run_dir)
        failure = json.loads((run_dir / "failure.json").read_text())
        failure["completed_updates"] = True
        _write_json(run_dir / "failure.json", failure)
    else:
        run(args)
        run_dir = Path(fixture["root"]) / args.run_id
        _rewrite_status_as_running(run_dir)
        atomic_json(
            run_dir / "failure.json",
            {
                "schema": "pcraft_soft_loss_v4_failure_v1",
                "config_hash": canonical_hash(config),
                "completed_updates": config["steps"],
                "error_type": "InjectedFailure",
                "error": "dual pending evidence",
            },
        )
    before = _tree(run_dir)
    _forbid_runtime_or_restore(monkeypatch)
    with pytest.raises((ValueError, ResumePreflightError)):
        validate_resumable_run(run_dir, expected_config=config)
    assert _tree(run_dir) == before


def test_public_resumable_rejects_pending_completed_dangling_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: dangling unexpected links disappear from file/dir namespace sets."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)
    config = resolve_run_configuration(args).config
    run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    _rewrite_status_as_running(run_dir)
    (run_dir / "unexpected-dangling-link").symlink_to(tmp_path / "missing-target")
    before = _tree(run_dir)
    _forbid_runtime_or_restore(monkeypatch)
    with pytest.raises((ValueError, ResumePreflightError), match="namespace|symlink"):
        validate_resumable_run(run_dir, expected_config=config)
    assert _tree(run_dir) == before


def test_public_resumable_rejects_forged_frozen_environment_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: environment authentication validates only its schema and allowlist."""

    fixture = _fixture(tmp_path)
    run_dir, config = _running_run(fixture, monkeypatch, interrupted_at=1)
    environment = json.loads((run_dir / "environment.json").read_text())
    environment.update(
        {
            "pytorch": "forged",
            "fixed_seed": False,
            "determinism": "forged",
            "bitwise_reproducible": True,
            "nondeterminism_reason": "forged",
        }
    )
    _write_json(run_dir / "environment.json", environment)
    before = _tree(run_dir)
    _forbid_runtime_or_restore(monkeypatch)
    with pytest.raises((ValueError, ResumePreflightError), match="environment"):
        validate_resumable_run(run_dir, expected_config=config)
    assert _tree(run_dir) == before


@pytest.mark.parametrize("manifest_name", ["environment.json", "run_manifest.json"])
def test_public_resumable_rejects_boolean_integer_static_manifest_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_name: str
) -> None:
    """Break caught: Python mapping equality treats JSON booleans as integer aliases."""

    fixture = _fixture(tmp_path)
    run_dir, config = _running_run(fixture, monkeypatch, interrupted_at=1)
    path = run_dir / manifest_name
    payload = json.loads(path.read_text())
    if manifest_name == "environment.json":
        payload["fixed_seed"] = 1
        payload["bitwise_reproducible"] = 0
    else:
        payload["scope"]["soft_only"] = 1
        payload["scope"]["final_step_export_only"] = 1
        payload["scope"]["downstream_geometry"] = 0
    _write_json(path, payload)
    before = _tree(run_dir)
    _forbid_runtime_or_restore(monkeypatch)
    with pytest.raises((ValueError, ResumePreflightError), match="manifest|environment"):
        validate_resumable_run(run_dir, expected_config=config)
    assert _tree(run_dir) == before


@pytest.mark.parametrize("mutation", ["completed_bool", "interval_endpoint_bool"])
def test_public_resumable_rejects_boolean_transaction_manifest_integers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Break caught: Python bool aliases authenticate as frozen integer transaction fields."""

    fixture = _fixture(tmp_path)
    interrupted_at = 1 if mutation == "completed_bool" else 100
    run_dir, config = _running_run(
        fixture, monkeypatch, interrupted_at=interrupted_at
    )
    pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
    manifest_path = run_dir / pointer["transaction"] / "transaction_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "completed_bool":
        manifest["completed_updates"] = False
    else:
        manifest["history_interval"]["first_completed_update"] = True
    _write_json(manifest_path, manifest)
    pointer["manifest_sha256"] = sha256_file(manifest_path)
    _write_json(run_dir / "checkpoint_latest.json", pointer)
    before = _tree(run_dir)
    _forbid_runtime_or_restore(monkeypatch)
    with pytest.raises((ValueError, ResumePreflightError), match="integer|transaction|interval"):
        validate_resumable_run(run_dir, expected_config=config)
    assert _tree(run_dir) == before


def test_public_resumable_rechecks_chain_namespace_at_final_linearization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: an orphan inserted after initial chain auth survives final checks."""

    fixture = _fixture(tmp_path)
    run_dir, config = _running_run(fixture, monkeypatch, interrupted_at=1)
    before = _tree(run_dir)
    original = trainer_module._authenticate_running_chain

    def authenticate_then_inject(
        root: Path, expected: Mapping[str, object]
    ) -> object:
        chain = original(root, expected)
        orphan = root / "transactions/step_00000099"
        if not os.path.lexists(orphan):
            orphan.mkdir()
        return chain

    monkeypatch.setattr(
        trainer_module, "_authenticate_running_chain", authenticate_then_inject
    )
    _forbid_runtime_or_restore(monkeypatch)
    with pytest.raises((ValueError, ResumePreflightError), match="orphan|namespace"):
        validate_resumable_run(run_dir, expected_config=config)
    after = _tree(run_dir)
    assert after.pop("transactions/step_00000099") == ("directory", None)
    assert after == before


def test_public_resumable_requires_expected_config_and_rejects_terminal_statuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: callers omit the expected identity or authenticate terminal runs as resumable."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    completed_args = _args(fixture, run_id="completed")
    completed_config = resolve_run_configuration(completed_args).config
    run(completed_args)
    completed_dir = Path(fixture["root"]) / completed_args.run_id
    with pytest.raises(TypeError):
        validate_resumable_run(completed_dir)  # type: ignore[call-arg]
    completed_before = _tree(completed_dir)
    with pytest.raises(ValueError, match="running|resumable|terminal"):
        validate_resumable_run(completed_dir, expected_config=completed_config)
    assert _tree(completed_dir) == completed_before

    failed_args = _args(fixture, run_id="failed")
    failed_config = resolve_run_configuration(failed_args).config

    def fail(**unused: object) -> object:
        raise V4TrainingStepError("terminal failure")

    monkeypatch.setattr(trainer_module, "train_one_v4_update", fail)
    with pytest.raises(V4TrainingStepError):
        run(failed_args)
    failed_dir = Path(fixture["root"]) / failed_args.run_id
    failed_before = _tree(failed_dir)
    with pytest.raises(ValueError, match="running|resumable|terminal"):
        validate_resumable_run(failed_dir, expected_config=failed_config)
    assert _tree(failed_dir) == failed_before


def test_public_completed_and_checkpoint_authentication_reject_nonterminal_or_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: calibration consumes running, failed, or uncommitted checkpoint bytes."""

    _install_toy(monkeypatch)
    fixture = _fixture(tmp_path)
    args = _args(fixture)

    def interrupt(completed: int) -> None:
        if completed == 1:
            raise _Interruption

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_Interruption):
        run(args)
    run_dir = Path(fixture["root"]) / args.run_id
    with pytest.raises(ValueError, match="completed"):
        validate_completed_run(run_dir)
    with pytest.raises(ValueError, match="completed"):
        authenticate_transaction_checkpoint(run_dir, completed_updates=0)


def test_dataclass_schemas_are_exact_and_completed_run_retains_only_checkpoint_metadata() -> None:
    """Break caught: completed validation retains every multi-GB checkpoint payload."""

    assert [field.name for field in fields(ResumeState)] == [
        "completed_updates",
        "last_result",
        "rng_state",
        "step_times",
        "peak_cuda_bytes",
        "slot_audits",
        "committed_transaction",
        "committed_transaction_sha256",
        "committed_history",
    ]
    assert [field.name for field in fields(AuthenticatedTransactionCheckpoint)] == [
        "completed_updates",
        "transaction",
        "transaction_manifest_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
    ]
    assert "payload" not in {field.name for field in fields(AuthenticatedTransactionCheckpoint)}
    assert "checkpoint" in {field.name for field in fields(AuthenticatedCheckpointSnapshot)}
    assert "transaction_checkpoints" in {field.name for field in fields(AuthenticatedCompletedRun)}
    assert [field.name for field in fields(AuthenticatedResumableRun)] == [
        "run_dir",
        "output_root_identity",
        "config",
        "config_hash",
        "status_sha256",
        "completed_updates",
        "latest_transaction",
        "pending_terminalization",
    ]
    assert "validate_resumable_run" in trainer_module.__all__


def _qualified(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts)).lower()


def test_v4_soft_only_ast_boundary_and_no_v3_trainer_import() -> None:
    """Break caught: lifecycle crosses into geometry or imports a v3 experiment entrypoint."""

    forbidden = {
        "repair",
        "carve",
        "carving",
        "delivery_geometry",
        "core_shell",
        "mesh",
        "meshing",
        "generate_mesh",
        "marching_cubes",
        "obj",
        "stl",
        "pino",
        "fem",
    }
    for path in sorted((PROJECT_ROOT / "shadowforge_pipeline/soft_voxel_v4").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.lower() for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [
                    f"{node.module or ''}.{alias.name}".lower() for alias in node.names
                ]
            elif isinstance(node, ast.Call):
                names = [_qualified(node.func)]
            else:
                names = []
            for name in names:
                assert "soft_voxel_v3.trainer" not in name
                assert "soft_voxel_v3.calibration" not in name
                assert "soft_voxel_v3.registry" not in name
                components = set(name.replace("-", "_").split("."))
                assert not (components & forbidden), (path, name)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                assert lowered != "/data" and not lowered.startswith("/data/")
                assert not lowered.endswith((".obj", ".stl"))
