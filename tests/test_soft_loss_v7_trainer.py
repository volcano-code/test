"""Fail-closed contracts for the authenticated v7 trainer lifecycle."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, asdict, fields
import gc
import hashlib
import io
import inspect
import json
import math
import os
from pathlib import Path
import random
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch

from shadowforge_pipeline.soft_voxel_v7.provenance import canonical_json_bytes, canonical_sha256

import shadowforge_pipeline.soft_voxel_v7.trainer as trainer
from shadowforge_pipeline.soft_voxel_v7.core import (
    compose_v7_gradient,
    v7_auxiliary_multiplier,
    v7_auxiliary_slot,
    v7_learning_rate,
)
from shadowforge_pipeline.soft_voxel_v7.trainer import (
    AuthenticatedCheckpoint,
    AuthenticatedEndpointMetrics,
    CompletedRun,
    FailedRun,
    ResumableRun,
    StepTelemetry,
    evaluate_six_view_metrics,
    required_checkpoint_steps,
    required_diagnostic_steps,
)


def _tree(root: Path) -> dict[str, tuple[str, object]]:
    result: dict[str, tuple[str, object]] = {}
    if not root.exists():
        return result
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            result[relative] = ("file", path.read_bytes())
        else:
            result[relative] = ("directory", None)
    return result


class _ToyProtocol:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.render_inputs: list[torch.Tensor] = []

    def render_all(self, density: torch.Tensor) -> torch.Tensor:
        self.render_inputs.append(density)
        view = density[0, 0].mean(dim=0)
        return view.unsqueeze(0).repeat(6, 1, 1)


class _CountingAdam(torch.optim.Adam):
    def __init__(self, parameter: torch.Tensor) -> None:
        super().__init__([parameter], lr=0.007)
        self.zero_calls = 0
        self.step_calls = 0

    def zero_grad(self, *args: object, **kwargs: object) -> None:
        self.zero_calls += 1
        super().zero_grad(*args, **kwargs)

    def step(self, *args: object, **kwargs: object) -> object:
        self.step_calls += 1
        return super().step(*args, **kwargs)


def _toy_state(profile: str = "full", completed: int = 300):
    logits = torch.linspace(-1.1, 1.2, 64, dtype=torch.float64).reshape(
        1, 1, 4, 4, 4
    ).requires_grad_()
    target = torch.zeros((6, 4, 4), dtype=torch.bool)
    target[:, 1:3, 1:3] = True
    return logits, _CountingAdam(logits), _ToyProtocol(), target, profile, completed


def _test_config(run_dir: Path, *, profile: str = "full", total: int = 4):
    return trainer._make_test_resolved_config(
        run_dir=run_dir,
        profile=profile,
        total_updates=total,
        resolution=4,
    )


def _fix5_stage_matrices(config: Mapping[str, object]):
    base = trainer._test_stage_matrix(config)

    def build(
        stage: str, endpoint: int, authorization_path: str | None,
        authorization_sha256: str | None,
    ):
        payload = trainer._plain(base.payload)
        payload.update({
            "stage": stage,
            "endpoint": endpoint,
            "authorization_path": authorization_path,
            "authorization_sha256": authorization_sha256,
        })
        payload["semantic_sha256"] = canonical_sha256({
            key: value for key, value in payload.items()
            if key != "semantic_sha256"
        })
        return type(base)(
            stage, endpoint, base.physical_rows, base.logical_rows,
            payload["semantic_sha256"], trainer._deep_immutable(payload),
            None, None,
        )

    historical = build("development-0300", 300, None, None)
    later = build(
        "development-1000", 1000,
        "authorizations/development_0300.json", "a" * 64,
    )
    return historical, later


def _install_fix5_stage_authentication(
    monkeypatch: pytest.MonkeyPatch, config: Mapping[str, object],
    historical: object, later: object,
) -> None:
    matrices = (historical, later)

    def validate_test_matrix(cfg: Mapping[str, object], supplied: object):
        assert cfg["config_sha256"] == config["config_sha256"]
        payload = canonical_json_bytes(trainer._stage_payload(supplied))
        matches = [
            matrix for matrix in matrices
            if canonical_json_bytes(trainer._plain(matrix.payload)) == payload
        ]
        if len(matches) != 1:
            raise ValueError("test stage matrix is unpublished, foreign, or a loser")
        return matches[0]

    def checkpoint_grant(
        cfg: Mapping[str, object], current: object, grant: object,
    ):
        assert cfg["config_sha256"] == config["config_sha256"]
        matches = []
        for matrix in matrices:
            expected = trainer._grant(matrix)
            if (
                all(grant[key] == expected[key] for key in expected if key != "endpoint")
                and grant["endpoint"] in (matrix.endpoint, config["updates"])
            ):
                matches.append(matrix)
        if len(matches) != 1:
            raise trainer.TrainingStateError("checkpoint grant is unpublished or foreign")
        return matches[0]

    original_matrix_for_validator = trainer._matrix_for_validator

    def matrix_for_validator(
        cfg: Mapping[str, object], supplied: object | None, run: Path,
    ):
        if supplied is not None:
            return validate_test_matrix(cfg, supplied)
        status = json.loads((run / "STATUS.json").read_text())
        if status["status"] == "running":
            stage_sha = status["stage_matrix_sha256"]
        elif status["status"] == "failed":
            stage_sha = json.loads(
                (run / "failure/observation.json").read_text()
            )["stage_matrix_sha256"]
        elif status["status"] == "completed":
            history = json.loads(
                (run / "artifacts/manifest.json").read_text()
            )["grant_history"]
            stage_sha = history[-1]["stage_matrix_sha256"]
        else:
            return original_matrix_for_validator(cfg, supplied, run)
        matches = [matrix for matrix in matrices if matrix.semantic_sha256 == stage_sha]
        if len(matches) != 1:
            raise trainer.TrainingStateError("live grant is unpublished or foreign")
        return matches[0]

    monkeypatch.setattr(trainer, "_validate_test_matrix", validate_test_matrix)
    monkeypatch.setattr(trainer, "_matrix_for_checkpoint_grant", checkpoint_grant)
    monkeypatch.setattr(trainer, "_matrix_for_validator", matrix_for_validator)
    monkeypatch.setattr(
        trainer, "_endpoint_steps",
        lambda records, cfg: tuple(
            step for step in (300, 1000)
            if any(row.completed_updates == step for row in records)
        ),
    )


def _rewrite_json(path: Path, row: dict[str, object]) -> str:
    raw = canonical_json_bytes(row)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _semantic(row: dict[str, object]) -> dict[str, object]:
    unsigned = dict(row); unsigned.pop("semantic_sha256", None)
    unsigned["semantic_sha256"] = canonical_sha256(unsigned)
    return unsigned


def _rewrite_checkpoint_and_pointers(
    config: Mapping[str, object], step: int, mutate: object,
) -> None:
    run = Path(config["run_path"])
    checkpoint = run / f"checkpoints/step_{step:04d}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mutate(payload)  # type: ignore[operator]
    buffer = io.BytesIO(); torch.save(payload, buffer); checkpoint.write_bytes(buffer.getvalue())
    digest = hashlib.sha256(buffer.getvalue()).hexdigest()
    latest_path = run / "checkpoints/LATEST.json"
    latest = json.loads(latest_path.read_text()); latest["checkpoint_sha256"] = digest
    _rewrite_json(latest_path, _semantic(latest))
    status_path = run / "STATUS.json"
    status = json.loads(status_path.read_text())
    key = "last_checkpoint_sha256" if status["status"] == "failed" else "current_checkpoint_sha256"
    status[key] = digest
    _rewrite_json(status_path, _semantic(status))


def test_public_records_are_exact_frozen_and_recursively_immutable() -> None:
    assert [field.name for field in fields(StepTelemetry)] == [
        "completed_updates", "learning_rate", "multiplier", "slot", "component",
        "base_loss", "aux_loss", "auxiliary_constructed", "base_gradient_norm",
        "raw_auxiliary_gradient_norm", "final_auxiliary_gradient_norm",
        "base_auxiliary_dot", "effective_ratio", "clip_count",
        "optimizer_step_count", "step_seconds", "cuda_executed",
        "peak_allocated_bytes", "diagnostic_sha256",
    ]
    assert [field.name for field in fields(AuthenticatedCheckpoint)] == [
        "run_dir", "config_sha256", "completed_updates", "checkpoint_path",
        "checkpoint_sha256", "predecessor_checkpoint_sha256", "stage",
        "stage_matrix_sha256", "authorization_path", "authorization_sha256",
        "history_length",
    ]
    assert [field.name for field in fields(AuthenticatedEndpointMetrics)] == [
        "path", "sha256", "completed_updates", "checkpoint_sha256",
        "stage_matrix_sha256", "per_view", "aggregate",
    ]
    assert [field.name for field in fields(ResumableRun)] == [
        "run_dir", "config_sha256", "status_path", "status_sha256",
        "current_checkpoint", "endpoint_metrics",
    ]
    assert [field.name for field in fields(CompletedRun)] == [
        "run_dir", "config_sha256", "status_path", "status_sha256",
        "current_checkpoint", "artifact_manifest_path", "artifact_manifest_sha256",
        "artifacts", "endpoint_metrics",
    ]
    assert [field.name for field in fields(FailedRun)] == [
        "run_dir", "config_sha256", "status_path", "status_sha256",
        "last_checkpoint", "observation_path", "observation_sha256",
        "failure_record_path", "failure_record_sha256", "endpoint_metrics",
    ]
    row = StepTelemetry(
        1, 0.007, 0.0, "hit", "base", 1.0, None, False, 1.0, 0.0,
        0.0, 0.0, 0.0, 1, 1, 0.01, False, 0, "0" * 64,
    )
    with pytest.raises(FrozenInstanceError):
        row.component = "hit"
    frozen = trainer._deep_immutable({"a": [{"b": 1}]})
    assert isinstance(frozen, MappingProxyType) and isinstance(frozen["a"], tuple)
    with pytest.raises(TypeError):
        frozen["a"] = ()


def test_public_signatures_are_exact_and_test_seams_are_private() -> None:
    assert tuple(inspect.signature(trainer.train_cell).parameters) == (
        "resolved_config", "stage_matrix", "resume", "max_updates"
    )
    assert tuple(inspect.signature(trainer.validate_resumable_run).parameters) == (
        "resolved_config", "stage_matrix", "expected_completed_updates"
    )
    assert tuple(inspect.signature(trainer.recover_interrupted_checkpoint_publication).parameters) == (
        "resolved_config", "stage_matrix"
    )
    assert "_test" not in inspect.signature(trainer.train_cell).parameters


@pytest.mark.parametrize(
    ("total", "expected"),
    [
        (1, (0, 1)),
        (2, (0, 2)),
        (300, (0, 2, 200, 300)),
        (1000, (0, 2, 200, 300, 400, 600, 800, 1000)),
        (3000, tuple(sorted({0, 2, 300, 1000, 3000, 2400, 2700, *range(200, 3001, 200)}))),
    ],
)
def test_registered_checkpoint_schedule(total: int, expected: tuple[int, ...]) -> None:
    assert required_checkpoint_steps(total) == expected
    assert required_diagnostic_steps(total) == ((2400, 2700, 3000) if total == 3000 else ())
    for bad in (True, 0, -1, 3.0, "3000"):
        with pytest.raises(ValueError):
            required_checkpoint_steps(bad)  # type: ignore[arg-type]


def test_six_view_metrics_use_raw_projection_inclusive_threshold_and_view_means() -> None:
    projection = torch.zeros((6, 3, 4), dtype=torch.float32)
    target = torch.zeros((6, 3, 4), dtype=torch.bool)
    foreground = (1, 2, 3, 4, 5, 6)
    true_positive = (1, 1, 1, 2, 1, 3)
    false_positive = (0, 1, 0, 2, 0, 3)
    for view, (fg, tp, fp) in enumerate(zip(foreground, true_positive, false_positive)):
        target[view].view(-1)[:fg] = True
        projection[view].view(-1)[:tp] = 1.0
        projection[view].view(-1)[fg : fg + fp] = 0.8
        projection[view].view(-1)[-1] = 1.0
    report = evaluate_six_view_metrics(projection, target)
    assert tuple(report) == ("per_view", "aggregate")
    assert len(report["per_view"]) == 6
    assert report["per_view"][1]["fp"] >= 1  # equality at .8*max is delivered
    assert report["aggregate"]["mean_recall"] == sum(
        row["recall"] for row in report["per_view"]
    ) / 6.0
    assert report["aggregate"]["mean_recall"] != pytest.approx(
        sum(true_positive) / sum(foreground)
    )
    with pytest.raises(TypeError):
        report["aggregate"]["mean_iou"] = 0.0
    for bad_projection, bad_target in (
        (projection.double(), target),
        (projection.clone().fill_(math.nan), target),
        (projection[:5], target[:5]),
        (torch.zeros_like(projection), target),
        (projection, target.float()),
    ):
        with pytest.raises(ValueError):
            evaluate_six_view_metrics(bad_projection, bad_target)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("profile", "completed", "component", "auxiliary", "renders"),
    [
        ("full", 99, "base", False, 1),
        ("full", 100, "hit", True, 2),
        ("full", 101, "barrier", True, 2),
        ("hit_only", 101, "base", False, 1),
        ("barrier_only", 100, "base", False, 1),
        ("base", 300, "base", False, 1),
    ],
)
def test_one_update_constructs_only_registered_fresh_graph(
    profile: str, completed: int, component: str, auxiliary: bool, renders: int,
) -> None:
    logits, optimizer, protocol, target, _, _ = _toy_state(profile, completed)
    row = trainer._train_one_update(
        logits=logits,
        optimizer=optimizer,
        protocol=protocol,
        targets=target,
        profile=profile,
        budget_ratio=0.10 if profile != "base" else None,
        completed_updates=completed,
        render_thresholded_fn=trainer._test_render_thresholded,
        allow_test_size=True,
        timer=trainer._DeterministicTestTimer(),
    )
    assert len(protocol.render_inputs) == renders
    assert len({id(value) for value in protocol.render_inputs}) == renders
    assert row.slot == ("hit" if completed % 4 in (0, 2) else "barrier")
    assert row.component == component
    assert row.auxiliary_constructed is auxiliary
    assert (row.aux_loss is not None) is auxiliary
    assert optimizer.zero_calls == optimizer.step_calls == 1
    assert row.clip_count == row.optimizer_step_count == 1
    assert row.learning_rate == v7_learning_rate(completed)
    assert row.multiplier == v7_auxiliary_multiplier(completed)
    assert row.base_auxiliary_dot >= 0.0
    assert all(math.isfinite(value) for value in asdict(row).values() if type(value) is float)


def test_base_gradient_bytes_and_single_full_component_are_identical() -> None:
    observed: dict[str, StepTelemetry] = {}
    for profile in ("hit_only", "full"):
        logits, optimizer, protocol, target, _, _ = _toy_state(profile, 100)
        observed[profile] = trainer._train_one_update(
            logits=logits,
            optimizer=optimizer,
            protocol=protocol,
            targets=target,
            profile=profile,
            budget_ratio=0.15,
            completed_updates=100,
            render_thresholded_fn=trainer._test_render_thresholded,
            allow_test_size=True,
            timer=trainer._DeterministicTestTimer(),
        )
    assert observed["hit_only"].aux_loss == observed["full"].aux_loss
    assert observed["hit_only"].raw_auxiliary_gradient_norm == observed["full"].raw_auxiliary_gradient_norm
    base = torch.tensor([1.0, -2.0], dtype=torch.float64)
    composed = compose_v7_gradient(base, None, 0.10, 0.0)
    assert torch.equal(composed, base) and composed.data_ptr() != base.data_ptr()


def test_one_update_rejects_nonfinite_loss_gradient_or_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    logits, optimizer, protocol, target, _, _ = _toy_state("base", 0)
    monkeypatch.setattr(trainer, "balanced_lp_loss", lambda *a, **k: protocol.render_inputs[-1].sum() * 0 + math.inf)
    with pytest.raises(trainer.TrainingStateError, match="finite"):
        trainer._train_one_update(
            logits=logits, optimizer=optimizer, protocol=protocol, targets=target,
            profile="base", budget_ratio=None, completed_updates=0,
            render_thresholded_fn=trainer._test_render_thresholded,
            allow_test_size=True, timer=trainer._DeterministicTestTimer(),
        )


def test_private_lifecycle_fresh_two_resume_and_completion_are_equivalent(tmp_path: Path) -> None:
    uninterrupted = _test_config(tmp_path / "uninterrupted" / "run")
    resumed = _test_config(tmp_path / "resumed" / "run")
    full = trainer._train_cell_for_test(uninterrupted, max_updates=4)
    partial = trainer._train_cell_for_test(resumed, max_updates=2)
    assert isinstance(partial, ResumableRun)
    assert partial.current_checkpoint.completed_updates == 2
    assert trainer.validate_resumable_run(
        resumed, stage_matrix=trainer._test_stage_matrix(resumed),
        expected_completed_updates=2,
    ) == partial
    resumed_final = trainer._train_cell_for_test(resumed, resume=True, max_updates=4)
    assert isinstance(full, CompletedRun) and isinstance(resumed_final, CompletedRun)
    full_logits = np.load(Path(full.run_dir) / "artifacts/final_logits.npy", allow_pickle=False)
    resumed_logits = np.load(Path(resumed_final.run_dir) / "artifacts/final_logits.npy", allow_pickle=False)
    assert np.array_equal(full_logits, resumed_logits)
    full_state = torch.load(full.current_checkpoint.checkpoint_path, map_location="cpu", weights_only=False)
    resumed_state = torch.load(resumed_final.current_checkpoint.checkpoint_path, map_location="cpu", weights_only=False)
    assert torch.equal(full_state["optimizer"]["exp_avg"], resumed_state["optimizer"]["exp_avg"])
    assert torch.equal(full_state["optimizer"]["exp_avg_sq"], resumed_state["optimizer"]["exp_avg_sq"])
    assert full_state["history"] == resumed_state["history"]
    assert trainer.validate_completed_run(
        resumed, stage_matrix=trainer._test_stage_matrix(resumed)
    ) == resumed_final


def test_private_running_and_completed_namespaces_are_exact(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run")
    result = trainer._train_cell_for_test(config, max_updates=2)
    run = Path(result.run_dir)
    assert set(path.name for path in run.iterdir()) == {
        "CONFIG.json", "STATUS.json", "checkpoints", "artifacts"
    }
    assert set(path.name for path in (run / "checkpoints").iterdir()) == {
        "step_0000.pt", "step_0002.pt", "LATEST.json"
    }
    assert not any("best" in path.name.lower() for path in run.rglob("*"))
    complete = trainer._train_cell_for_test(config, resume=True, max_updates=4)
    assert isinstance(complete, CompletedRun)
    assert set(path.name for path in run.iterdir()) == {
        "CONFIG.json", "STATUS.json", "checkpoints", "artifacts"
    }
    assert set(path.name for path in (run / "artifacts").iterdir()) == {
        "metrics", "final_logits.npy", "final_density.npy", "final_hard_005.npy",
        "audit.json", "resource.json", "manifest.json",
    }


@pytest.mark.parametrize(
    "relative",
    [
        "CONFIG.json", "STATUS.json", "checkpoints/LATEST.json",
        "checkpoints/step_0002.pt", "artifacts/final_logits.npy",
        "artifacts/metrics/step_0004.json", "artifacts/manifest.json",
    ],
)
def test_read_only_validators_reject_tampering_with_zero_writes(
    tmp_path: Path, relative: str,
) -> None:
    config = _test_config(tmp_path / relative.replace("/", "_") / "run")
    trainer._train_cell_for_test(config, max_updates=4)
    path = Path(config["run_path"]) / relative
    path.write_bytes(path.read_bytes() + b"tamper")
    before = _tree(Path(config["run_path"]))
    with pytest.raises((ValueError, trainer.TrainingStateError)):
        trainer.validate_completed_run(config, stage_matrix=trainer._test_stage_matrix(config))
    assert _tree(Path(config["run_path"])) == before


def test_validator_rejects_extra_symlink_and_hardlink_namespaces(tmp_path: Path) -> None:
    for attack in ("extra", "symlink", "hardlink"):
        config = _test_config(tmp_path / attack / "run")
        trainer._train_cell_for_test(config, max_updates=2)
        run = Path(config["run_path"])
        if attack == "extra":
            (run / "foreign").write_text("x")
        elif attack == "symlink":
            (run / "evil").symlink_to(run / "CONFIG.json")
        else:
            os.link(run / "CONFIG.json", run / "evil")
        before = _tree(run)
        with pytest.raises((ValueError, trainer.TrainingStateError)):
            trainer.validate_resumable_run(
                config, stage_matrix=trainer._test_stage_matrix(config),
                expected_completed_updates=2,
            )
        assert _tree(run) == before


def test_recovery_adopts_exact_successor_and_rejects_ambiguous_orphan(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run")
    trainer._train_cell_for_test(config, max_updates=2)
    stage = trainer._test_stage_matrix(config)
    trainer._test_publish_authenticated_successor_without_pointer(config, stage_matrix=stage)
    recovered = trainer.recover_interrupted_checkpoint_publication(config, stage_matrix=stage)
    assert recovered.current_checkpoint.completed_updates == 3
    trainer._test_publish_authenticated_successor_without_pointer(config, stage_matrix=stage)
    trainer._test_duplicate_successor_orphan(config)
    before = _tree(Path(config["run_path"]))
    with pytest.raises(trainer.TrainingStateError, match="ambiguous|orphan"):
        trainer.recover_interrupted_checkpoint_publication(config, stage_matrix=stage)
    assert _tree(Path(config["run_path"])) == before


def test_registered_caps_and_stage_grants_fail_before_writes(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run")
    stage = trainer._test_stage_matrix(config)
    for cap in (True, 0, 1, 3, 200, 3001, "2"):
        before = _tree(tmp_path)
        with pytest.raises(ValueError):
            trainer._validate_absolute_cap(config, stage, cap)  # type: ignore[arg-type]
        assert _tree(tmp_path) == before
    forged = trainer._plain(stage.payload)
    forged["semantic_sha256"] = "0" * 64
    before = _tree(tmp_path)
    with pytest.raises(ValueError):
        trainer._train_cell_for_test(config, stage_matrix=forged, max_updates=2)
    assert _tree(tmp_path) == before


def test_failure_record_is_derived_terminal_and_immutable(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run")
    failed = trainer._fail_cell_for_test(
        config,
        observation={"cuda_executed": False, "device_total_bytes": 100, "peak_allocated_bytes": 1},
    )
    assert isinstance(failed, FailedRun)
    assert trainer.validate_failed_run(
        config, stage_matrix=trainer._test_stage_matrix(config)
    ) == failed
    with pytest.raises(trainer.TrainingStateError, match="failed"):
        trainer._train_cell_for_test(config, resume=True, max_updates=2)
    run = Path(failed.run_dir)
    assert set(path.name for path in run.iterdir()) == {
        "CONFIG.json", "STATUS.json", "checkpoints", "artifacts", "failure"
    }
    assert set(path.name for path in (run / "failure").iterdir()) == {
        "observation.json", "failure.json"
    }


def test_endpoint_projection_rows_bind_authenticated_checkpoint_and_metrics(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run")
    trainer._train_cell_for_test(config, max_updates=4)
    matrix = trainer._test_stage_matrix(config)
    engineering = trainer.project_engineering_row(config, stage_matrix=matrix, physical_index=0)
    metric = trainer.project_metric_row(config, stage_matrix=matrix, logical_index=0)
    assert engineering["checkpoint_sha256"] == metric["checkpoint_sha256"]
    assert engineering["status"] == "PASS" and engineering["failure_code"] == "NONE"
    assert metric["completed_updates"] == matrix.endpoint
    assert metric["metrics_path"].endswith("artifacts/metrics/step_0004.json")
    with pytest.raises(ValueError):
        trainer.project_metric_row(config, stage_matrix=matrix, logical_index=True)


def test_production_api_rejects_private_config_and_never_creates_run(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run")
    before = _tree(tmp_path)
    with pytest.raises(ValueError, match="production"):
        trainer.train_cell(config, stage_matrix=trainer._test_stage_matrix(config), max_updates=2)
    assert _tree(tmp_path) == before


def test_fix1_run_transaction_rejects_bound_ancestor_and_leaf_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"; (root / "runs/development/task/base").mkdir(parents=True)
    config = {
        "root_path": str(root), "run_path": "runs/development/task/base/seed_17",
    }

    class _Root:
        def __enter__(self): return self
        def __exit__(self, *args: object): return None
        def ensure_registered_directory(self, name: str) -> None: (root / name).mkdir(exist_ok=True)
        def revalidate_root_marker(self) -> None: return None

    monkeypatch.setattr(trainer, "open_v7_root_transaction", lambda path: _Root())
    with trainer._V7RunTransaction(config) as transaction:
        original = root / "runs/development/task"
        moved = root / "runs/development/task.old"
        original.rename(moved); original.mkdir()
        with pytest.raises(trainer.TrainingStateError, match="ancestor|changed|replacement"):
            transaction.revalidate()

    (root / "runs/development/task").rmdir()
    moved.rename(root / "runs/development/task")
    leaf = root / "runs/development/task/base/seed_17"; leaf.mkdir()
    with trainer._V7RunTransaction(config) as transaction:
        moved_leaf = leaf.with_name("seed_17.old"); leaf.rename(moved_leaf); leaf.mkdir()
        with pytest.raises(trainer.TrainingStateError, match="leaf|changed|replacement"):
            transaction.revalidate()


def test_fix1_transaction_publication_is_fd_relative_and_noreplace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"; (root / "runs/development/task/base").mkdir(parents=True)
    config = {"root_path": str(root), "run_path": "runs/development/task/base/seed_17"}

    class _Root:
        def __enter__(self): return self
        def __exit__(self, *args: object): return None
        def ensure_registered_directory(self, name: str) -> None: (root / name).mkdir(exist_ok=True)
        def revalidate_root_marker(self) -> None: return None

    monkeypatch.setattr(trainer, "open_v7_root_transaction", lambda path: _Root())
    with trainer._V7RunTransaction(config) as transaction:
        transaction.create_run_leaf()
        transaction.publish(Path("CONFIG.json"), b"first", create_once=True)
        leaf = root / config["run_path"]
        moved = leaf.with_name("seed_17.old"); leaf.rename(moved); leaf.mkdir()
        with pytest.raises(trainer.TrainingStateError):
            transaction.publish(Path("STATUS.json"), b"must-not-land", create_once=False)
        assert not (leaf / "STATUS.json").exists()
        assert not (moved / "STATUS.json").exists()
        with pytest.raises(trainer.TrainingStateError):
            transaction.publish(Path("CONFIG.json"), b"replacement", create_once=True)
        assert (moved / "CONFIG.json").read_bytes() == b"first"


def test_fix1_checkpoint_digest_is_authenticated_before_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(tmp_path / "run")
    trainer._train_cell_for_test(config, max_updates=2)
    latest_path = Path(config["run_path"]) / "checkpoints/LATEST.json"
    latest = json.loads(latest_path.read_text()); latest["checkpoint_sha256"] = "0" * 64
    _rewrite_json(latest_path, _semantic(latest))
    called = False

    def forbidden(*args: object, **kwargs: object):
        nonlocal called; called = True
        raise AssertionError("untrusted pickle executed")

    monkeypatch.setattr(torch, "load", forbidden)
    before = _tree(Path(config["run_path"]))
    with pytest.raises(trainer.TrainingStateError, match="digest|SHA|LATEST"):
        trainer.validate_resumable_run(config, stage_matrix=trainer._test_stage_matrix(config))
    assert called is False and _tree(Path(config["run_path"])) == before


def test_fix1_recovery_repairs_pointer_ahead_status(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run")
    trainer._train_cell_for_test(config, max_updates=2); matrix = trainer._test_stage_matrix(config)
    trainer._test_publish_authenticated_successor_without_pointer(config, stage_matrix=matrix)
    run = Path(config["run_path"]); successor = run / "checkpoints/step_0003.pt"
    payload = torch.load(successor, map_location="cpu", weights_only=False)
    digest = hashlib.sha256(successor.read_bytes()).hexdigest()
    latest = trainer._latest_payload(config["config_sha256"], 3, digest, payload["predecessor_checkpoint_sha256"])
    _rewrite_json(run / "checkpoints/LATEST.json", latest)
    recovered = trainer.recover_interrupted_checkpoint_publication(config, stage_matrix=matrix)
    assert recovered.current_checkpoint.completed_updates == 3
    assert json.loads((run / "STATUS.json").read_text())["current_step"] == 3


def test_fix1_recovery_completes_manifest_before_status(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run"); matrix = trainer._test_stage_matrix(config)
    completed = trainer._train_cell_for_test(config, max_updates=4)
    run = Path(config["run_path"])
    running = trainer._running_status(config["config_sha256"], completed.current_checkpoint)
    _rewrite_json(run / "STATUS.json", running)
    recovered = trainer.recover_interrupted_checkpoint_publication(config, stage_matrix=matrix)
    assert isinstance(recovered, CompletedRun)
    assert json.loads((run / "STATUS.json").read_text())["status"] == "completed"


def test_fix1_recovery_completes_observation_before_record_and_status(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run"); matrix = trainer._test_stage_matrix(config)
    failed = trainer._fail_cell_for_test(config, observation={"cuda_executed": False})
    run = Path(config["run_path"]); (run / "failure/failure.json").unlink()
    _rewrite_json(run / "STATUS.json", trainer._running_status(config["config_sha256"], failed.last_checkpoint))
    recovered = trainer.recover_interrupted_checkpoint_publication(config, stage_matrix=matrix)
    assert isinstance(recovered, FailedRun)
    assert (run / "failure/failure.json").is_file()


def test_fix1_recovery_rebuilds_missing_endpoint_metric_and_terminalizes(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run"); matrix = trainer._test_stage_matrix(config)
    completed = trainer._train_cell_for_test(config, max_updates=4)
    run = Path(config["run_path"])
    for path in list((run / "artifacts").iterdir()):
        if path.name == "metrics":
            for child in path.iterdir(): child.unlink()
        elif path.is_file(): path.unlink()
    _rewrite_json(run / "STATUS.json", trainer._running_status(config["config_sha256"], completed.current_checkpoint))
    recovered = trainer.recover_interrupted_checkpoint_publication(config, stage_matrix=matrix)
    assert isinstance(recovered, CompletedRun)
    assert (run / "artifacts/metrics/step_0004.json").is_file()


def test_fix1_nonfinite_update_publishes_derived_failed_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(tmp_path / "run"); original = trainer._train_one_update

    def nonfinite(**kwargs: object):
        if kwargs["completed_updates"] == 1:
            raise trainer.TrainingStateError("composed gradient must be finite")
        return original(**kwargs)

    monkeypatch.setattr(trainer, "_train_one_update", nonfinite)
    result = trainer._train_cell_for_test(config, max_updates=2)
    assert isinstance(result, FailedRun)
    record = json.loads(result.failure_record_path.read_text())
    assert record["failure_code"] == "NONFINITE_TRAINING"


def test_fix1_failed_validator_rejects_type_and_identity_forgery(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run"); matrix = trainer._test_stage_matrix(config)
    failed = trainer._fail_cell_for_test(config, observation={"cuda_executed": False})
    observation = json.loads(failed.observation_path.read_text())
    observation["cuda_executed"] = 0
    observation_sha = _rewrite_json(failed.observation_path, _semantic(observation))
    status = json.loads(failed.status_path.read_text()); status["observation_sha256"] = observation_sha
    _rewrite_json(failed.status_path, _semantic(status))
    before = _tree(Path(config["run_path"]))
    with pytest.raises(trainer.TrainingStateError, match="boolean|type|observation"):
        trainer.validate_failed_run(config, stage_matrix=matrix)
    assert _tree(Path(config["run_path"])) == before


def test_fix1_step_zero_optimizer_and_rng_semantics_are_deep(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run"); matrix = trainer._test_stage_matrix(config)
    trainer._train_cell_for_test(config, max_updates=2)
    step0 = torch.load(Path(config["run_path"]) / "checkpoints/step_0000.pt", map_location="cpu", weights_only=False)
    assert torch.equal(step0["logits"], torch.full_like(step0["logits"], 2.0))
    assert torch.count_nonzero(step0["optimizer"]["exp_avg"]) == 0
    assert torch.count_nonzero(step0["optimizer"]["exp_avg_sq"]) == 0

    def negative_square(payload: dict[str, object]) -> None:
        payload["optimizer"]["exp_avg_sq"].view(-1)[0] = -1.0

    _rewrite_checkpoint_and_pointers(config, 2, negative_square)
    with pytest.raises(trainer.TrainingStateError, match="optimizer|nonnegative"):
        trainer.validate_resumable_run(config, stage_matrix=matrix)

    fresh = _test_config(tmp_path / "rng" / "run"); trainer._train_cell_for_test(fresh, max_updates=2)
    def bad_numpy(payload: dict[str, object]) -> None:
        payload["rng"]["numpy"] = ("MT19937", np.array([1], dtype=np.uint32), 625, 0, 0.0)
    _rewrite_checkpoint_and_pointers(fresh, 2, bad_numpy)
    with pytest.raises(trainer.TrainingStateError, match="RNG|NumPy"):
        trainer.validate_resumable_run(fresh, stage_matrix=trainer._test_stage_matrix(fresh))


@pytest.mark.parametrize("terminal", ["completed", "failed"])
def test_fix1_terminal_tree_rejects_extra_empty_directories(tmp_path: Path, terminal: str) -> None:
    config = _test_config(tmp_path / terminal / "run"); matrix = trainer._test_stage_matrix(config)
    if terminal == "completed":
        trainer._train_cell_for_test(config, max_updates=4); validator = trainer.validate_completed_run
    else:
        trainer._fail_cell_for_test(config, observation={"cuda_executed": False}); validator = trainer.validate_failed_run
    (Path(config["run_path"]) / "artifacts/foreign_empty").mkdir()
    before = _tree(Path(config["run_path"]))
    with pytest.raises(trainer.TrainingStateError, match="tree|namespace|artifact"):
        validator(config, stage_matrix=matrix)
    assert _tree(Path(config["run_path"])) == before


def test_fix1_final_density_and_hard_are_exact_float32_checkpoint_functions(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run"); matrix = trainer._test_stage_matrix(config)
    completed = trainer._train_cell_for_test(config, max_updates=4); run = Path(config["run_path"])
    logits = torch.load(completed.current_checkpoint.checkpoint_path, map_location="cpu", weights_only=False)["logits"][0, 0]
    density = np.load(run / "artifacts/final_density.npy", allow_pickle=False)
    hard = np.load(run / "artifacts/final_hard_005.npy", allow_pickle=False)
    assert density.dtype == np.float32
    assert np.array_equal(density, torch.sigmoid(logits.float()).numpy())
    assert hard.dtype == np.bool_ and np.array_equal(hard, density >= 0.05)


def test_fix1_dependency_failure_before_publication_leaves_no_new_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(tmp_path / "run")
    calls: list[int] = []

    def reject(config_row: Mapping[str, object], matrix: object, completed: int) -> None:
        calls.append(completed)
        if completed == 2:
            raise trainer.TrainingStateError("input identity changed")

    monkeypatch.setattr(trainer, "_reauthenticate_before_publication", reject)
    with pytest.raises(trainer.TrainingStateError, match="identity changed"):
        trainer._train_cell_for_test(config, max_updates=2)
    run = Path(config["run_path"])
    assert calls == [0, 2]
    assert not (run / "checkpoints/step_0002.pt").exists()
    assert json.loads((run / "checkpoints/LATEST.json").read_text())["completed_updates"] == 0


def test_fix1_historical_metric_uses_checkpoint_bound_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(tmp_path / "run"); current = trainer._test_stage_matrix(config)
    completed = trainer._train_cell_for_test(config, max_updates=4)
    historical = copy.copy(current)
    object.__setattr__(historical, "stage", "test-historical")
    observed: list[str] = []
    original = trainer._validate_endpoint_file

    def capture(path: Path, cfg: Mapping[str, object], matrix: object, checkpoint: object):
        observed.append(matrix.stage)
        return original(path, cfg, current, checkpoint)

    monkeypatch.setattr(trainer, "_matrix_for_checkpoint_record", lambda cfg, matrix, record: historical)
    monkeypatch.setattr(trainer, "_validate_endpoint_file", capture)
    trainer._collect_endpoint_metrics(Path(config["run_path"]), config, current, (completed.current_checkpoint,))
    assert observed == ["test-historical"]


@pytest.mark.parametrize("endpoint", [300, 1000])
def test_fix2_recovery_rebuilds_checkpoint_bound_endpoint_metric_after_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: int,
) -> None:
    config = _test_config(tmp_path / f"step_{endpoint}" / "run", profile="base", total=3000)
    base_matrix = trainer._test_stage_matrix(config)
    matrix_payload = trainer._plain(base_matrix.payload)
    matrix_payload["stage"] = f"test-{endpoint:04d}"
    matrix_payload["endpoint"] = endpoint
    matrix_payload["semantic_sha256"] = canonical_sha256({
        key: value for key, value in matrix_payload.items()
        if key != "semantic_sha256"
    })
    matrix = type(base_matrix)(
        matrix_payload["stage"], endpoint, base_matrix.physical_rows,
        base_matrix.logical_rows, matrix_payload["semantic_sha256"],
        trainer._deep_immutable(matrix_payload), None, None,
    )
    original = trainer._publish_endpoint_metric

    def crash(*args: object, **kwargs: object) -> str:
        raise RuntimeError("crash before endpoint metric publication")

    monkeypatch.setattr(trainer, "_publish_endpoint_metric", crash)
    with pytest.raises(RuntimeError, match="endpoint metric"):
        trainer._train_impl(config, matrix, resume=False, cap=endpoint)
    run = Path(config["run_path"])
    latest = json.loads((run / "checkpoints/LATEST.json").read_text())
    assert latest["completed_updates"] == endpoint
    assert not (run / f"artifacts/metrics/step_{endpoint:04d}.json").exists()

    monkeypatch.setattr(trainer, "_publish_endpoint_metric", original)
    monkeypatch.setattr(trainer, "_validate_test_matrix", lambda cfg, supplied: matrix)
    recovered = trainer.recover_interrupted_checkpoint_publication(
        config, stage_matrix=matrix,
    )
    assert isinstance(recovered, ResumableRun)
    metric = recovered.endpoint_metrics[-1]
    expected = trainer._recompute_endpoint_metrics(
        config, recovered.current_checkpoint, matrix,
    )
    assert metric.completed_updates == endpoint
    assert metric.checkpoint_sha256 == recovered.current_checkpoint.checkpoint_sha256
    assert canonical_json_bytes(trainer._plain({
        "per_view": metric.per_view, "aggregate": metric.aggregate,
    })) == canonical_json_bytes(trainer._plain(expected))


def test_fix2_failed_engineering_projection_copies_authenticated_observation(
    tmp_path: Path,
) -> None:
    from shadowforge_pipeline.soft_voxel_v7 import registry

    config = _test_config(tmp_path / "run")
    matrix = trainer._test_stage_matrix(config)
    failed = trainer._fail_cell_for_test(config, observation={
        "cuda_executed": False,
        "peak_allocated_bytes": 17,
        "device_total_bytes": 123,
    })
    observation = json.loads(failed.observation_path.read_text())
    projected = trainer.project_engineering_row(
        config, stage_matrix=matrix, physical_index=0,
    )
    evidence_fields = (
        "loss_finite", "gradient_finite", "parameters_finite", "cuda_executed",
        "clip_count", "optimizer_step_count", "slot_valid",
        "disabled_not_constructed", "step_time_count", "step_times_finite",
        "min_step_seconds", "peak_allocated_bytes", "device_total_bytes",
        "hit_eligible_slots", "hit_active_slots", "barrier_eligible_slots",
        "barrier_active_slots",
    )
    assert {key: projected[key] for key in evidence_fields} == {
        key: observation[key] for key in evidence_fields
    }
    expected = trainer._plain(matrix.physical_rows[0])
    registry._validate_engineering_row(trainer._plain(projected), expected, matrix)


def test_fix2_recovery_adopts_crashed_genesis_and_removes_empty_failure_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(tmp_path / "genesis" / "run")
    matrix = trainer._test_stage_matrix(config)
    original = trainer._publish_run_json

    def crash_latest(run: Path, relative: Path, payload: Mapping[str, object], **kwargs: object) -> str:
        if relative == Path("checkpoints/LATEST.json"):
            raise RuntimeError("crash after genesis generation")
        return original(run, relative, payload, **kwargs)

    monkeypatch.setattr(trainer, "_publish_run_json", crash_latest)
    with pytest.raises(RuntimeError, match="genesis"):
        trainer._train_cell_for_test(config, max_updates=2)
    run = Path(config["run_path"])
    assert (run / "checkpoints/step_0000.pt").is_file()
    assert not (run / "checkpoints/LATEST.json").exists()
    assert not (run / "STATUS.json").exists()

    monkeypatch.setattr(trainer, "_publish_run_json", original)
    recovered = trainer.recover_interrupted_checkpoint_publication(
        config, stage_matrix=matrix,
    )
    assert isinstance(recovered, ResumableRun)
    assert recovered.current_checkpoint.completed_updates == 0

    (run / "failure").mkdir()
    recovered_again = trainer.recover_interrupted_checkpoint_publication(
        config, stage_matrix=matrix,
    )
    assert isinstance(recovered_again, ResumableRun)
    assert not (run / "failure").exists()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("base loss must be finite", (False, True, True)),
        ("base gradient must be finite", (True, False, True)),
        ("parameters must remain finite", (True, True, False)),
    ],
)
def test_fix2_nonfinite_failure_flags_identify_actual_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    message: str, expected: tuple[bool, bool, bool],
) -> None:
    config = _test_config(tmp_path / message.split()[1] / "run")

    def fail(**kwargs: object):
        raise trainer.TrainingStateError(message)

    monkeypatch.setattr(trainer, "_train_one_update", fail)
    result = trainer._train_cell_for_test(config, max_updates=2)
    assert isinstance(result, FailedRun)
    observation = json.loads(result.observation_path.read_text())
    assert (
        observation["loss_finite"], observation["gradient_finite"],
        observation["parameters_finite"],
    ) == expected


def _seeded_rng(seed: int) -> dict[str, object]:
    return {
        "schema": trainer._RNG_SCHEMA,
        "python": random.Random(seed).getstate(),
        "numpy": np.random.RandomState(seed).get_state(),
        "torch_cpu": torch.Generator(device="cpu").manual_seed(seed).get_state(),
        "torch_cuda": (),
    }


def _rewrite_rng_chain(config: Mapping[str, object], *, genesis_seed: int, successor_seed: int) -> None:
    run = Path(config["run_path"])
    step0_path = run / "checkpoints/step_0000.pt"
    step0 = torch.load(step0_path, map_location="cpu", weights_only=False)
    step0["rng"] = _seeded_rng(genesis_seed)
    buffer0 = io.BytesIO(); torch.save(step0, buffer0); step0_path.write_bytes(buffer0.getvalue())
    sha0 = hashlib.sha256(buffer0.getvalue()).hexdigest()

    step2_path = run / "checkpoints/step_0002.pt"
    step2 = torch.load(step2_path, map_location="cpu", weights_only=False)
    step2["rng"] = _seeded_rng(successor_seed)
    step2["predecessor_checkpoint_sha256"] = sha0
    buffer2 = io.BytesIO(); torch.save(step2, buffer2); step2_path.write_bytes(buffer2.getvalue())
    sha2 = hashlib.sha256(buffer2.getvalue()).hexdigest()
    _rewrite_json(
        run / "checkpoints/LATEST.json",
        trainer._latest_payload(config["config_sha256"], 2, sha2, sha0),
    )
    _rewrite_json(
        run / "STATUS.json",
        trainer._running_status(
            config["config_sha256"],
            trainer._checkpoint_record(run, config["config_sha256"], step2, sha2),
        ),
    )


def test_fix2_rng_chain_binds_seed17_genesis_and_exact_successor_state(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "honest" / "run")
    matrix = trainer._test_stage_matrix(config)
    trainer._train_cell_for_test(config, max_updates=2)
    run = Path(config["run_path"])
    step0 = torch.load(run / "checkpoints/step_0000.pt", map_location="cpu", weights_only=False)
    step2 = torch.load(run / "checkpoints/step_0002.pt", map_location="cpu", weights_only=False)
    expected = _seeded_rng(17)
    assert step0["rng"]["python"] == expected["python"]
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b for a, b in zip(step0["rng"]["numpy"], expected["numpy"]))
    assert torch.equal(step0["rng"]["torch_cpu"], expected["torch_cpu"])
    assert step0["rng"]["torch_cuda"] == expected["torch_cuda"]
    assert step2["rng"]["python"] == step0["rng"]["python"]
    assert torch.equal(step2["rng"]["torch_cpu"], step0["rng"]["torch_cpu"])

    _rewrite_rng_chain(config, genesis_seed=17, successor_seed=99)
    with pytest.raises(trainer.TrainingStateError, match="predecessor|RNG"):
        trainer.validate_resumable_run(config, stage_matrix=matrix)

    coherent = _test_config(tmp_path / "coherent_99" / "run")
    coherent_matrix = trainer._test_stage_matrix(coherent)
    trainer._train_cell_for_test(coherent, max_updates=2)
    _rewrite_rng_chain(coherent, genesis_seed=99, successor_seed=99)
    with pytest.raises(trainer.TrainingStateError, match="seed|genesis|RNG"):
        trainer.validate_resumable_run(coherent, stage_matrix=coherent_matrix)


def test_fix2_run_transaction_enter_closes_every_fd_after_binding_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    (root / "runs/development/task/base").mkdir(parents=True)
    config = {"root_path": str(root), "run_path": "runs/development/task/base/seed_17"}

    class _Root:
        def __enter__(self): return self
        def __exit__(self, *args: object): return None
        def ensure_registered_directory(self, name: str) -> None: return None
        def revalidate_root_marker(self) -> None: return None

    monkeypatch.setattr(trainer, "open_v7_root_transaction", lambda path: _Root())
    original_stat = trainer.os.stat

    def mismatched_stat(path: object, *args: object, **kwargs: object):
        observed = original_stat(path, *args, **kwargs)
        if path == "development" and kwargs.get("dir_fd") is not None:
            return SimpleNamespace(st_dev=observed.st_dev, st_ino=observed.st_ino + 1)
        return observed

    monkeypatch.setattr(trainer.os, "stat", mismatched_stat)
    transaction = trainer._V7RunTransaction(config)
    with pytest.raises(trainer.TrainingStateError, match="ancestor|binding|changed"):
        transaction.__enter__()
    assert transaction._lock_fd is None
    assert transaction._descriptors == []
    assert transaction._bindings == []


def test_fix2_checkpoint_chain_validation_retains_at_most_two_full_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(tmp_path / "run", profile="base", total=3000)
    matrix = trainer._test_stage_matrix(config)
    trainer._train_cell_for_test(config, max_updates=3000)
    original = trainer._load_checkpoint_path

    class TrackingPayload(dict):
        alive = 0
        maximum = 0

        def __init__(self, value: Mapping[str, object]) -> None:
            super().__init__(value)
            type(self).alive += 1
            type(self).maximum = max(type(self).maximum, type(self).alive)

        def __del__(self) -> None:
            type(self).alive -= 1

    def tracked(*args: object, **kwargs: object):
        payload, record = original(*args, **kwargs)
        return TrackingPayload(payload), record

    monkeypatch.setattr(trainer, "_load_checkpoint_path", tracked)
    trainer.validate_completed_run(config, stage_matrix=matrix)
    gc.collect()
    assert TrackingPayload.maximum <= 2


def test_fix3_orphan_rng_extension_is_rejected_before_pointer_or_status_cas(
    tmp_path: Path,
) -> None:
    config = _test_config(tmp_path / "run")
    matrix = trainer._test_stage_matrix(config)
    trainer._train_cell_for_test(config, max_updates=2)
    trainer._test_publish_authenticated_successor_without_pointer(
        config, stage_matrix=matrix,
    )
    run = Path(config["run_path"])
    orphan_path = run / "checkpoints/step_0003.pt"
    orphan = torch.load(orphan_path, map_location="cpu", weights_only=False)
    orphan["rng"] = _seeded_rng(99)
    buffer = io.BytesIO(); torch.save(orphan, buffer); orphan_path.write_bytes(buffer.getvalue())
    latest_before = (run / "checkpoints/LATEST.json").read_bytes()
    status_before = (run / "STATUS.json").read_bytes()
    tree_before = _tree(run)

    with pytest.raises(trainer.TrainingStateError, match="orphan|RNG|predecessor"):
        trainer.recover_interrupted_checkpoint_publication(
            config, stage_matrix=matrix,
        )
    assert (run / "checkpoints/LATEST.json").read_bytes() == latest_before
    assert (run / "STATUS.json").read_bytes() == status_before
    assert _tree(run) == tree_before
    _, current, _ = trainer._load_chain(
        run, config, matrix, allow_orphans=True,
    )
    assert current.completed_updates == 2
    assert json.loads((run / "STATUS.json").read_text())["current_step"] == 2


def test_fix3_crashed_genesis_foreign_artifact_rejects_before_commit_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = trainer._publish_run_json

    for label, relative in (
        ("artifact", Path("artifacts/foreign.bin")),
        ("metric", Path("artifacts/metrics/foreign.json")),
    ):
        config = _test_config(tmp_path / label / "run")
        matrix = trainer._test_stage_matrix(config)

        def crash_latest(run: Path, target: Path, payload: Mapping[str, object], **kwargs: object) -> str:
            if target == Path("checkpoints/LATEST.json"):
                raise RuntimeError("crash after genesis generation")
            return original(run, target, payload, **kwargs)

        monkeypatch.setattr(trainer, "_publish_run_json", crash_latest)
        with pytest.raises(RuntimeError, match="genesis"):
            trainer._train_cell_for_test(config, max_updates=2)
        monkeypatch.setattr(trainer, "_publish_run_json", original)
        run = Path(config["run_path"])
        target = run / relative; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"foreign")
        tree_before = _tree(run)

        with pytest.raises(trainer.TrainingStateError, match="artifact|metric|namespace"):
            trainer.recover_interrupted_checkpoint_publication(
                config, stage_matrix=matrix,
            )
        assert not (run / "checkpoints/LATEST.json").exists()
        assert not (run / "STATUS.json").exists()
        assert _tree(run) == tree_before


@pytest.mark.parametrize("terminal", [False, True])
def test_fix4_engineering_prefix_is_exact_immutable_live_cap_and_zero_write(
    tmp_path: Path, terminal: bool,
) -> None:
    total = 2 if terminal else 4
    config = _test_config(tmp_path / ("completed" if terminal else "running") / "run", total=total)
    matrix = trainer._test_stage_matrix(config)
    state = trainer._train_cell_for_test(config, max_updates=2)
    assert isinstance(state, CompletedRun if terminal else ResumableRun)
    before = _tree(Path(config["run_path"]))

    row = trainer.project_engineering_prefix(
        config, stage_matrix=matrix, physical_index=0, completed_updates=2,
    )
    assert tuple(row) == (
        "schema", "stage", "physical_index", "task_id", "budget_id", "profile",
        "config_sha256", "run_path", "completed_updates", "checkpoint_path",
        "checkpoint_sha256", "stage_matrix_sha256", "authorization_sha256",
        "device", "loss_finite", "gradient_finite", "parameters_finite",
        "cuda_executed", "clip_count", "optimizer_step_count", "slot_valid",
        "disabled_not_constructed", "step_time_count", "step_times_finite",
        "min_step_seconds", "peak_allocated_bytes", "device_total_bytes",
        "hit_eligible_slots", "hit_active_slots", "barrier_eligible_slots",
        "barrier_active_slots", "semantic_sha256",
    )
    assert row["schema"] == "pcraft_soft_loss_v7_engineering_prefix_v1"
    assert row["stage"] == matrix.stage
    assert row["physical_index"] == 0
    assert row["completed_updates"] == 2
    assert row["checkpoint_path"] == f"{config['run_path']}/checkpoints/step_0002.pt"
    assert row["checkpoint_sha256"] == state.current_checkpoint.checkpoint_sha256
    assert row["stage_matrix_sha256"] == matrix.semantic_sha256
    assert row["authorization_sha256"] is None
    assert row["device"] == "cuda:0"
    assert row["loss_finite"] is row["gradient_finite"] is row["parameters_finite"] is True
    assert row["clip_count"] == row["optimizer_step_count"] == row["step_time_count"] == 2
    assert trainer._failure_code({**trainer._plain(row), "_profile": config["profile"]}, 2) == "NONE"
    unsigned = trainer._plain(row); semantic = unsigned.pop("semantic_sha256")
    assert canonical_sha256(unsigned) == semantic
    with pytest.raises(TypeError):
        row["completed_updates"] = 1
    assert _tree(Path(config["run_path"])) == before


def test_fix4_engineering_prefix_rejects_missing_failed_and_unregistered_requests(
    tmp_path: Path,
) -> None:
    running_config = _test_config(tmp_path / "running" / "run")
    running_matrix = trainer._test_stage_matrix(running_config)
    trainer._train_cell_for_test(running_config, max_updates=2)
    running_before = _tree(Path(running_config["run_path"]))
    for physical_index, completed in (
        (True, 2), (1, 2), (0, True), (0, 0), (0, 3), (0, 200),
    ):
        with pytest.raises((ValueError, trainer.TrainingStateError)):
            trainer.project_engineering_prefix(
                running_config, stage_matrix=running_matrix,
                physical_index=physical_index, completed_updates=completed,
            )
        assert _tree(Path(running_config["run_path"])) == running_before

    future_config = _test_config(tmp_path / "future" / "run")
    future_matrix = trainer._test_stage_matrix(future_config)
    trainer._initialize_run(future_config, future_matrix)
    future_before = _tree(Path(future_config["run_path"]))
    with pytest.raises(trainer.TrainingStateError, match="live|cap|update|prefix"):
        trainer.project_engineering_prefix(
            future_config, stage_matrix=future_matrix,
            physical_index=0, completed_updates=2,
        )
    assert _tree(Path(future_config["run_path"])) == future_before

    failed_config = _test_config(tmp_path / "failed" / "run")
    failed_matrix = trainer._test_stage_matrix(failed_config)
    trainer._fail_cell_for_test(failed_config, observation={"cuda_executed": False})
    failed_before = _tree(Path(failed_config["run_path"]))
    with pytest.raises(trainer.TrainingStateError, match="failed|PASS|prefix"):
        trainer.project_engineering_prefix(
            failed_config, stage_matrix=failed_matrix,
            physical_index=0, completed_updates=2,
        )
    assert _tree(Path(failed_config["run_path"])) == failed_before


@pytest.mark.parametrize("terminal", [False, True])
def test_fix4_engineering_prefix_rebuilds_identical_historical_cap_zero_write(
    tmp_path: Path, terminal: bool,
) -> None:
    total = 4 if terminal else 6
    config = _test_config(tmp_path / ("completed" if terminal else "running") / "run", total=total)
    matrix = trainer._test_stage_matrix(config)
    trainer._train_cell_for_test(config, max_updates=2)
    live = trainer.project_engineering_prefix(
        config, stage_matrix=matrix, physical_index=0, completed_updates=2,
    )

    later = (
        trainer._train_cell_for_test(config, resume=True, max_updates=4)
        if terminal
        else trainer._train_impl(config, matrix, resume=True, cap=4)
    )
    assert isinstance(later, CompletedRun if terminal else ResumableRun)
    before = _tree(Path(config["run_path"]))
    historical = trainer.project_engineering_prefix(
        config, stage_matrix=matrix, physical_index=0, completed_updates=2,
    )

    assert tuple(historical) == tuple(live)
    assert trainer._plain(historical) == trainer._plain(live)
    assert historical["checkpoint_sha256"] != later.current_checkpoint.checkpoint_sha256
    assert historical["checkpoint_path"].endswith("/checkpoints/step_0002.pt")
    assert _tree(Path(config["run_path"])) == before


@pytest.mark.parametrize("later_state", ["running", "completed", "failed"])
def test_fix5_historical_projectors_rebuild_live_bytes_across_later_grant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, later_state: str,
) -> None:
    config = _test_config(tmp_path / later_state / "run", profile="base", total=1000)
    historical, later = _fix5_stage_matrices(config)
    _install_fix5_stage_authentication(monkeypatch, config, historical, later)

    trainer._train_impl(config, historical, resume=False, cap=2)
    live_prefix = trainer.project_engineering_prefix(
        config, stage_matrix=historical, physical_index=0, completed_updates=2,
    )
    trainer._train_impl(config, historical, resume=True, cap=300)
    live_engineering = trainer.project_engineering_row(
        config, stage_matrix=historical, physical_index=0,
    )
    live_metric = trainer.project_metric_row(
        config, stage_matrix=historical, logical_index=0,
    )

    cap = 1000 if later_state == "completed" else 400
    later_run = trainer._train_impl(config, later, resume=True, cap=cap)
    if later_state == "failed":
        assert isinstance(later_run, ResumableRun)
        run = Path(config["run_path"])
        payload, checkpoint, _ = trainer._load_chain(run, config, later)
        observation = trainer._observation_from_checkpoint(
            config, later, checkpoint, payload,
        )
        observation["cuda_executed"] = False
        observation["semantic_sha256"] = canonical_sha256({
            key: value for key, value in observation.items()
            if key != "semantic_sha256"
        })
        later_run = trainer._publish_failed_from_observation(
            run, config, later, checkpoint, observation,
        )
        assert isinstance(later_run, FailedRun)
    elif later_state == "completed":
        assert isinstance(later_run, CompletedRun)
    else:
        assert isinstance(later_run, ResumableRun)

    before = _tree(Path(config["run_path"]))
    historical_prefix = trainer.project_engineering_prefix(
        config, stage_matrix=historical, physical_index=0, completed_updates=2,
    )
    historical_engineering = trainer.project_engineering_row(
        config, stage_matrix=historical, physical_index=0,
    )
    historical_metric = trainer.project_metric_row(
        config, stage_matrix=historical, logical_index=0,
    )
    assert trainer._plain(historical_prefix) == trainer._plain(live_prefix)
    assert trainer._plain(historical_engineering) == trainer._plain(live_engineering)
    assert trainer._plain(historical_metric) == trainer._plain(live_metric)
    assert _tree(Path(config["run_path"])) == before

    if later_state == "failed":
        same_stage = trainer.project_engineering_row(
            config, stage_matrix=later, physical_index=0,
        )
        assert same_stage["status"] == "FAIL"
        assert same_stage["failure_code"] == "CUDA_NOT_USED"
        with pytest.raises(trainer.TrainingStateError):
            trainer.project_metric_row(
                config, stage_matrix=later, logical_index=0,
            )
        with pytest.raises(trainer.TrainingStateError):
            trainer.project_engineering_prefix(
                config, stage_matrix=later, physical_index=0,
                completed_updates=2,
            )
        assert _tree(Path(config["run_path"])) == before


def test_fix5_historical_projectors_reject_unpublished_loser_and_foreign_zero_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _test_config(tmp_path / "run", profile="base", total=1000)
    historical, later = _fix5_stage_matrices(config)
    _install_fix5_stage_authentication(monkeypatch, config, historical, later)
    trainer._train_impl(config, historical, resume=False, cap=300)
    trainer._train_impl(config, later, resume=True, cap=400)
    before = _tree(Path(config["run_path"]))

    payload = trainer._plain(historical.payload)
    payload["authorization_sha256"] = "f" * 64
    payload["semantic_sha256"] = canonical_sha256({
        key: value for key, value in payload.items()
        if key != "semantic_sha256"
    })
    foreign = type(historical)(
        historical.stage, historical.endpoint, historical.physical_rows,
        historical.logical_rows, payload["semantic_sha256"],
        trainer._deep_immutable(payload), None, None,
    )
    loser_payload = trainer._plain(historical.payload)
    loser_payload["physical_rows"] = []
    loser_payload["logical_rows"] = []
    loser_payload["semantic_sha256"] = canonical_sha256({
        key: value for key, value in loser_payload.items()
        if key != "semantic_sha256"
    })
    loser = type(historical)(
        historical.stage, historical.endpoint, (), (),
        loser_payload["semantic_sha256"], trainer._deep_immutable(loser_payload),
        None, None,
    )

    for matrix in (foreign, loser):
        for project in (
            lambda: trainer.project_engineering_prefix(
                config, stage_matrix=matrix, physical_index=0,
                completed_updates=2,
            ),
            lambda: trainer.project_engineering_row(
                config, stage_matrix=matrix, physical_index=0,
            ),
            lambda: trainer.project_metric_row(
                config, stage_matrix=matrix, logical_index=0,
            ),
        ):
            with pytest.raises((ValueError, trainer.TrainingStateError)):
                project()
            assert _tree(Path(config["run_path"])) == before


@pytest.mark.parametrize("failure_step", [0, 2, 4])
def test_fix6_same_stage_failed_prefix_depends_on_failure_order(
    tmp_path: Path, failure_step: int,
) -> None:
    config = _test_config(tmp_path / f"step_{failure_step}" / "run", total=6)
    matrix = trainer._test_stage_matrix(config)
    trainer._train_impl(config, matrix, resume=False, cap=failure_step)
    live = None
    if failure_step >= 2:
        live = trainer.project_engineering_prefix(
            config, stage_matrix=matrix, physical_index=0,
            completed_updates=2,
        )
    run = Path(config["run_path"])
    payload, checkpoint, _ = trainer._load_chain(run, config, matrix)
    assert checkpoint.completed_updates == failure_step
    observation = trainer._observation_from_checkpoint(
        config, matrix, checkpoint, payload,
    )
    observation["cuda_executed"] = False
    observation["semantic_sha256"] = canonical_sha256({
        key: value for key, value in observation.items()
        if key != "semantic_sha256"
    })
    failed = trainer._publish_failed_from_observation(
        run, config, matrix, checkpoint, observation,
    )
    assert isinstance(failed, FailedRun)
    before = _tree(run)

    if failure_step >= 2:
        rebuilt = trainer.project_engineering_prefix(
            config, stage_matrix=matrix, physical_index=0,
            completed_updates=2,
        )
        assert live is not None
        assert trainer._plain(rebuilt) == trainer._plain(live)
    else:
        with pytest.raises(trainer.TrainingStateError):
            trainer.project_engineering_prefix(
                config, stage_matrix=matrix, physical_index=0,
                completed_updates=2,
            )
    assert _tree(run) == before


@pytest.mark.parametrize("cap_was_durable", [False, True])
def test_fix7_real_failed_update_preserves_only_an_already_durable_pass_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cap_was_durable: bool,
) -> None:
    config = _test_config(
        tmp_path / ("after_cap" if cap_was_durable else "building_cap") / "run",
        profile="base", total=6,
    )
    matrix = trainer._test_stage_matrix(config)
    live = None
    if cap_was_durable:
        trainer._train_impl(config, matrix, resume=False, cap=2)
        live = trainer.project_engineering_prefix(
            config, stage_matrix=matrix, physical_index=0,
            completed_updates=2,
        )

    def fail_first_update(**kwargs: object) -> object:
        raise trainer.TrainingStateError("base loss must be finite")

    monkeypatch.setattr(trainer, "_train_one_update", fail_first_update)
    failed = trainer._train_impl(
        config, matrix, resume=cap_was_durable,
        cap=4 if cap_was_durable else 2,
    )
    assert isinstance(failed, FailedRun)
    assert failed.last_checkpoint.completed_updates == (2 if cap_was_durable else 0)
    before = _tree(Path(config["run_path"]))

    if cap_was_durable:
        rebuilt = trainer.project_engineering_prefix(
            config, stage_matrix=matrix, physical_index=0,
            completed_updates=2,
        )
        assert live is not None
        assert trainer._plain(rebuilt) == trainer._plain(live)
    else:
        with pytest.raises(trainer.TrainingStateError):
            trainer.project_engineering_prefix(
                config, stage_matrix=matrix, physical_index=0,
                completed_updates=2,
            )
    assert _tree(Path(config["run_path"])) == before
