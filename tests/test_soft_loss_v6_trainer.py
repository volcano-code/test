"""Fail-closed contracts for the authenticated native-256 v6 trainer."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, asdict, fields
import io
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pytest
import torch

import shadowforge_pipeline.soft_voxel_v6 as soft_voxel_v6
import shadowforge_pipeline.soft_voxel_v6.trainer as trainer
from shadowforge_pipeline.soft_voxel_v6.core import (
    PROFILE_ORDER,
    compose_v6_gradient,
    v6_auxiliary_multiplier,
    v6_learning_rate,
)
from shadowforge_pipeline.soft_voxel_v6.provenance import (
    atomic_json,
    canonical_hash,
    canonical_json_bytes,
    sha256_file,
)
from shadowforge_pipeline.soft_voxel_v6.trainer import (
    CompletedRun,
    ResumableRun,
    StepTelemetry,
    evaluate_six_view_metrics,
    train_cell,
    validate_completed_run,
    validate_resumable_run,
)


def _nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping) and set(left) == set(right)
        for key in left:
            _nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for a, b in zip(left, right):
            _nested_equal(a, b)
    else:
        assert left == right


class _ToyProtocol:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.inputs: list[torch.Tensor] = []

    def render_all(self, density: torch.Tensor) -> torch.Tensor:
        self.inputs.append(density)
        view = density[0, 0].mean(0)
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


def _toy_state(profile: str = "full", completed: int = 900):
    logits = torch.linspace(-1.3, 1.2, 64, dtype=torch.float64).reshape(
        1, 1, 4, 4, 4
    ).requires_grad_()
    target = torch.zeros((6, 4, 4), dtype=torch.float64)
    target[:, 1:3, 1:3] = 1.0
    return logits, _CountingAdam(logits), _ToyProtocol(), target, profile, completed


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


def _test_config(run_dir: Path, *, arm: str = "full", total: int = 4) -> dict[str, object]:
    return trainer._make_test_resolved_config(
        run_dir=run_dir,
        arm=arm,
        total_updates=total,
        resolution=4,
    )


def _write_npy(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    path.write_bytes(buffer.getvalue())


def _rehash_completed_artifacts(run_dir: Path, relatives: set[str]) -> None:
    manifest_path = run_dir / "artifacts/artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    for relative in relatives:
        manifest["files_sha256"][relative] = sha256_file(run_dir / relative)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    status_path = run_dir / "STATUS.json"
    status = json.loads(status_path.read_text())
    status["artifacts_sha256"] = sha256_file(manifest_path)
    status_path.write_bytes(canonical_json_bytes(status))


def _rewrite_final_checkpoint(
    run_dir: Path, mutate: object, *, completed: int = 4
) -> None:
    relative = f"checkpoints/step_{completed:04d}.pt"
    path = run_dir / relative
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mutate(payload)  # type: ignore[operator]
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(buffer.getvalue())
    pointer_path = run_dir / "checkpoints/LATEST.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["sha256"] = sha256_file(path)
    pointer_path.write_bytes(canonical_json_bytes(pointer))
    _rehash_completed_artifacts(
        run_dir, {relative, "checkpoints/LATEST.json"}
    )


def _telemetry_row(completed_updates: int) -> StepTelemetry:
    completed_before = completed_updates - 1
    multiplier = v6_auxiliary_multiplier(completed_before)
    return StepTelemetry(
        completed_updates=completed_updates,
        learning_rate=v6_learning_rate(completed_before),
        auxiliary_multiplier=multiplier,
        base_loss=1.0,
        j_loss=1.0 if multiplier > 0.0 else None,
        h_loss=1.0 if multiplier > 0.0 else None,
        base_norm=1.0,
        j_raw_norm=1.0 if multiplier > 0.0 else 0.0,
        h_raw_norm=1.0 if multiplier > 0.0 else 0.0,
        j_ratio=0.2 * multiplier,
        h_ratio=0.1 * multiplier,
        combined_ratio=0.3 * multiplier,
        j_zero=False,
        h_zero=False,
        pre_clip_norm=1.0,
        post_clip_norm=1.0,
        parameter_delta_norm=0.1,
        clip_calls=1,
        optimizer_steps=1,
        elapsed_seconds=0.01,
        peak_cuda_allocated_bytes=1,
        tail_details=(),
    )


def test_public_records_and_signature_are_exact_immutable_and_json_safe() -> None:
    assert [field.name for field in fields(StepTelemetry)] == [
        "completed_updates",
        "learning_rate",
        "auxiliary_multiplier",
        "base_loss",
        "j_loss",
        "h_loss",
        "base_norm",
        "j_raw_norm",
        "h_raw_norm",
        "j_ratio",
        "h_ratio",
        "combined_ratio",
        "j_zero",
        "h_zero",
        "pre_clip_norm",
        "post_clip_norm",
        "parameter_delta_norm",
        "clip_calls",
        "optimizer_steps",
        "elapsed_seconds",
        "peak_cuda_allocated_bytes",
        "tail_details",
    ]
    assert [field.name for field in fields(ResumableRun)] == [
        "run_dir", "config_sha256", "completed_updates", "checkpoint_sha256"
    ]
    assert [field.name for field in fields(CompletedRun)] == [
        "run_dir", "config_sha256", "completed_updates", "artifacts_sha256", "metrics"
    ]
    record = StepTelemetry(
        1, 0.007, 0.0, 1.0, None, None, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, False, False, 1.0, 1.0, 0.1, 1, 1, 0.01, 0, (),
    )
    json.dumps(asdict(record), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        record.learning_rate = 1.0
    import inspect

    assert tuple(inspect.signature(train_cell).parameters) == (
        "resolved_config", "resume", "max_updates"
    )


def test_package_root_exports_authenticated_trainer_api_and_records() -> None:
    assert soft_voxel_v6.train_cell is train_cell
    assert soft_voxel_v6.validate_resumable_run is validate_resumable_run
    assert soft_voxel_v6.validate_completed_run is validate_completed_run
    assert soft_voxel_v6.StepTelemetry is StepTelemetry
    assert soft_voxel_v6.ResumableRun is ResumableRun
    assert soft_voxel_v6.CompletedRun is CompletedRun


def test_canonical_provenance_is_byte_stable_strict_and_create_once(tmp_path: Path) -> None:
    left = {"b": [2, 3], "a": {"flag": True, "integer": 1, "real": 1.0}}
    right = {"a": {"real": 1.0, "integer": 1, "flag": True}, "b": [2, 3]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_hash(left) == canonical_hash(right)
    assert canonical_hash({"v": 1}) != canonical_hash({"v": 1.0})
    for invalid in (math.nan, math.inf, torch.tensor(1), object()):
        with pytest.raises((TypeError, ValueError)):
            canonical_hash({"v": invalid})
    path = tmp_path / "record.json"
    atomic_json(path, left, create_once=True)
    before = path.read_bytes()
    atomic_json(path, right, create_once=True)
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        atomic_json(path, {"different": True}, create_once=True)


def test_one_update_uses_fresh_graphs_one_clip_one_step_lr_before_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits, optimizer, protocol, targets, profile, completed = _toy_state()
    backward_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    clip_calls: list[tuple[list[torch.Tensor], float]] = []
    lr_at_step: list[float] = []
    original_backward = torch.autograd.grad
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = optimizer.step

    def capture_grad(*args: object, **kwargs: object):
        backward_calls.append((args, kwargs))
        assert kwargs.get("retain_graph", False) is False
        return original_backward(*args, **kwargs)

    def capture_clip(parameters: object, maximum: float, *args: object, **kwargs: object):
        values = list(parameters)  # type: ignore[arg-type]
        clip_calls.append((values, maximum))
        return original_clip(values, maximum, *args, **kwargs)

    def capture_step(*args: object, **kwargs: object):
        lr_at_step.append(float(optimizer.param_groups[0]["lr"]))
        return original_step(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", capture_grad)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_clip)
    monkeypatch.setattr(optimizer, "step", capture_step)
    before = logits.detach().clone()
    row = trainer._train_one_update(
        logits=logits,
        optimizer=optimizer,
        protocol=protocol,
        targets=targets,
        arm=profile,
        completed_updates=completed,
        timer=lambda: 1.0,
    )
    assert len(protocol.inputs) == 3
    assert len({id(value) for value in protocol.inputs}) == 3
    assert len(backward_calls) == 3
    assert all(call[1].get("retain_graph", False) is False for call in backward_calls)
    assert clip_calls == [([logits], 1.0)]
    assert optimizer.zero_calls == 1 and optimizer.step_calls == 1
    assert lr_at_step == [v6_learning_rate(completed)]
    assert row.clip_calls == 1 and row.optimizer_steps == 1
    assert row.parameter_delta_norm > 0.0 and not torch.equal(before, logits)
    assert row.auxiliary_multiplier == v6_auxiliary_multiplier(completed)
    assert all(math.isfinite(v) for v in asdict(row).values() if type(v) is float)


@pytest.mark.parametrize(
    ("arm", "completed", "renders"),
    [
        ("base", 900, 1),
        ("j_only", 900, 2),
        ("h_only", 900, 2),
        ("full", 900, 3),
        ("full", 599, 1),
    ],
)
def test_disabled_and_off_ramp_components_are_never_constructed(
    arm: str, completed: int, renders: int
) -> None:
    logits, optimizer, protocol, targets, _, _ = _toy_state(arm, completed)
    row = trainer._train_one_update(
        logits=logits,
        optimizer=optimizer,
        protocol=protocol,
        targets=targets,
        arm=arm,
        completed_updates=completed,
        timer=lambda: 2.0,
    )
    assert len(protocol.inputs) == renders
    assert (row.j_loss is not None) is (completed >= 600 and arm in {"j_only", "full"})
    assert (row.h_loss is not None) is (completed >= 600 and arm in {"h_only", "full"})


def test_same_state_j_h_components_match_single_and_full_profiles() -> None:
    observed: dict[str, tuple[float | None, float | None]] = {}
    for arm in ("j_only", "h_only", "full"):
        logits, optimizer, protocol, targets, _, completed = _toy_state(arm)
        row = trainer._train_one_update(
            logits=logits,
            optimizer=optimizer,
            protocol=protocol,
            targets=targets,
            arm=arm,
            completed_updates=completed,
            timer=lambda: 3.0,
        )
        observed[arm] = (row.j_loss, row.h_loss)
    assert observed["j_only"][0] == observed["full"][0]
    assert observed["h_only"][1] == observed["full"][1]


def test_zero_auxiliary_norm_is_recorded_not_divided(monkeypatch: pytest.MonkeyPatch) -> None:
    logits, optimizer, protocol, targets, _, completed = _toy_state("j_only")
    monkeypatch.setattr(
        trainer,
        "delivery_lovasz_loss",
        lambda score, target: score.sum() * 0.0,
    )
    row = trainer._train_one_update(
        logits=logits,
        optimizer=optimizer,
        protocol=protocol,
        targets=targets,
        arm="j_only",
        completed_updates=completed,
        timer=lambda: 4.0,
    )
    assert row.j_raw_norm == 0.0 and row.j_zero is True and row.j_ratio == 0.0


@pytest.mark.parametrize("failure", ["base", "gradient", "post_step"])
def test_nonfinite_state_fails_closed_without_silent_continuation(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    logits, optimizer, protocol, targets, _, completed = _toy_state("base")
    if failure == "base":
        monkeypatch.setattr(
            trainer,
            "balanced_lp_loss",
            lambda *args, **kwargs: protocol.inputs[-1].sum() * 0.0 + math.inf,
        )
    elif failure == "gradient":
        monkeypatch.setattr(
            trainer,
            "compose_v6_gradient",
            lambda *args, **kwargs: torch.full_like(logits, math.nan),
        )
    else:
        original = optimizer.step

        def bad_step(*args: object, **kwargs: object):
            result = original(*args, **kwargs)
            with torch.no_grad():
                logits.flatten()[0] = math.nan
            return result

        monkeypatch.setattr(optimizer, "step", bad_step)
    with pytest.raises(trainer.TrainingStateError, match="finite"):
        trainer._train_one_update(
            logits=logits,
            optimizer=optimizer,
            protocol=protocol,
            targets=targets,
            arm="base",
            completed_updates=completed,
            timer=lambda: 5.0,
        )


def test_six_view_metrics_are_finite_equal_view_means_not_pooled() -> None:
    target = torch.zeros((6, 3, 4), dtype=torch.bool)
    predicted = torch.zeros_like(target)
    foreground = (1, 2, 3, 4, 5, 6)
    true_positive = (1, 1, 1, 2, 1, 3)
    false_positive = (0, 1, 0, 2, 0, 3)
    for index, (fg, tp, fp) in enumerate(zip(foreground, true_positive, false_positive)):
        target[index].view(-1)[:fg] = True
        predicted[index].view(-1)[:tp] = True
        predicted[index].view(-1)[fg : fg + fp] = True
    report = evaluate_six_view_metrics(predicted, target, completed_updates=3000)
    assert set(report) == {"schema", "completed_updates", "threshold", "per_view", "aggregate"}
    assert len(report["per_view"]) == 6
    assert report["aggregate"]["mean_recall"] == pytest.approx(
        sum(row["recall"] for row in report["per_view"]) / 6.0
    )
    pooled = sum(true_positive) / sum(foreground)
    assert report["aggregate"]["mean_recall"] != pytest.approx(pooled)
    json.dumps(report, allow_nan=False)
    for bad_target, bad_prediction in (
        (torch.ones_like(target), torch.ones_like(predicted)),
        (torch.zeros_like(target), torch.ones_like(predicted)),
        (target[:5], predicted[:5]),
    ):
        with pytest.raises(ValueError, match="denominator|six"):
            evaluate_six_view_metrics(bad_prediction, bad_target, completed_updates=3000)


def test_public_update_caps_are_absolute_and_strict(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run")
    for invalid in (True, 1, 3, 199, 3001):
        with pytest.raises(ValueError, match="max_updates"):
            train_cell(config, max_updates=invalid)
    with pytest.raises(ValueError, match="production"):
        train_cell(config, max_updates=2)


def test_private_small_lifecycle_has_exact_chain_diagnostics_and_no_best(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "run", total=4)
    result = trainer._train_cell_for_test(config, max_updates=2)
    assert isinstance(result, ResumableRun) and result.completed_updates == 2
    run_dir = Path(result.run_dir)
    assert sorted(path.name for path in (run_dir / "checkpoints").glob("step_*.pt")) == [
        "step_0000.pt", "step_0002.pt"
    ]
    assert not any("best" in path.name.lower() for path in run_dir.rglob("*"))
    assert validate_resumable_run(config, expected_completed_updates=2).completed_updates == 2
    completed = trainer._train_cell_for_test(config, resume=True, max_updates=4)
    assert isinstance(completed, CompletedRun) and completed.completed_updates == 4
    expected = {
        "COMMAND.json", "RUN.json", "environment.json", "sources.json", "inputs.json",
        "STATUS.json", "checkpoints/LATEST.json", "checkpoints/step_0000.pt",
        "checkpoints/step_0002.pt", "checkpoints/step_0004.pt",
        "artifacts/final_logits.npy", "artifacts/final_density.npy",
        "artifacts/final_hard_005.npy", "artifacts/metrics.json", "artifacts/audit.json",
        "artifacts/resource.json", "artifacts/artifacts.json",
    }
    assert expected.issubset(_tree(run_dir))
    assert validate_completed_run(config).completed_updates == 4


@pytest.mark.parametrize(
    "target",
    [
        "sources.json", "inputs.json", "environment.json", "RUN.json", "STATUS.json",
        "checkpoints/step_0002.pt", "checkpoints/LATEST.json",
        "artifacts/final_logits.npy", "artifacts/metrics.json", "artifacts/artifacts.json",
    ],
)
def test_resume_and_completion_reject_tampering_without_writes(tmp_path: Path, target: str) -> None:
    config = _test_config(tmp_path / target.replace("/", "_") / "run", total=4)
    trainer._train_cell_for_test(config, max_updates=2)
    run_dir = Path(config["run"]["path"])
    path = run_dir / target
    if not path.exists():
        trainer._train_cell_for_test(config, resume=True, max_updates=4)
    path.write_bytes(path.read_bytes() + b"tamper")
    before = _tree(run_dir)
    validator = validate_completed_run if "artifacts/" in target else validate_resumable_run
    with pytest.raises(ValueError):
        if validator is validate_resumable_run:
            validator(config, expected_completed_updates=2)
        else:
            validator(config)
    assert _tree(run_dir) == before


def test_completed_validator_rejects_unregistered_artifact_entry(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "extra_artifact" / "run", total=4)
    trainer._train_cell_for_test(config, max_updates=4)
    run_dir = Path(config["run"]["path"])
    extra = run_dir / "artifacts/unregistered.json"
    extra.write_bytes(canonical_json_bytes({"foreign": True}))
    before = _tree(run_dir)
    with pytest.raises(ValueError, match="extra|unexpected|unregistered|namespace"):
        validate_completed_run(config)
    assert _tree(run_dir) == before


def test_resume_rejects_symlink_component_and_config_bool_alias(tmp_path: Path) -> None:
    config = _test_config(tmp_path / "real" / "run", total=4)
    trainer._train_cell_for_test(config, max_updates=2)
    real = Path(config["run"]["path"])
    link = tmp_path / "linked"
    link.symlink_to(real.parent, target_is_directory=True)
    forged = copy.deepcopy(config)
    forged["run"]["path"] = str(link / real.name)
    with pytest.raises(ValueError, match="symlink|canonical"):
        validate_resumable_run(forged, expected_completed_updates=2)
    forged = copy.deepcopy(config)
    forged["seed"] = True
    with pytest.raises(ValueError, match="config|type|canonical"):
        trainer._validate_test_resolved_config(forged)


@pytest.mark.parametrize("cut", [2, 3])
def test_interrupted_resume_matches_uninterrupted_logits_adam_rng_audit_metrics(
    tmp_path: Path, cut: int
) -> None:
    base = _test_config(tmp_path / f"base_{cut}", total=4)
    split = _test_config(tmp_path / f"split_{cut}", total=4)
    trainer._train_cell_for_test(base, max_updates=4)
    trainer._train_cell_for_test(split, max_updates=cut)
    trainer._train_cell_for_test(split, resume=True, max_updates=4)
    base_dir = Path(base["run"]["path"])
    split_dir = Path(split["run"]["path"])
    for name in (
        "final_logits.npy", "final_density.npy", "final_hard_005.npy", "metrics.json", "audit.json"
    ):
        assert (base_dir / "artifacts" / name).read_bytes() == (
            split_dir / "artifacts" / name
        ).read_bytes()
    base_checkpoint = torch.load(
        base_dir / "checkpoints/step_0004.pt", map_location="cpu", weights_only=False
    )
    split_checkpoint = torch.load(
        split_dir / "checkpoints/step_0004.pt", map_location="cpu", weights_only=False
    )
    for payload in (base_checkpoint, split_checkpoint):
        payload.pop("config_sha256")
        payload.pop("predecessor_sha256")
    _nested_equal(base_checkpoint, split_checkpoint)


def test_resume_recovers_authenticated_generation_created_before_pointer_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _test_config(tmp_path / "baseline", total=4)
    interrupted = _test_config(tmp_path / "interrupted", total=4)
    trainer._train_cell_for_test(baseline, max_updates=4)
    trainer._train_cell_for_test(interrupted, max_updates=2)
    run_dir = Path(interrupted["run"]["path"])
    original_atomic_json = trainer.atomic_json

    def crash_before_pointer(path, payload, *, create_once=False):
        if (
            Path(path).name == "LATEST.json"
            and payload.get("completed_updates") == 3
        ):
            raise RuntimeError("injected crash before checkpoint pointer publication")
        return original_atomic_json(path, payload, create_once=create_once)

    monkeypatch.setattr(trainer, "atomic_json", crash_before_pointer)
    with pytest.raises(RuntimeError, match="pointer publication"):
        trainer._train_cell_for_test(interrupted, resume=True, max_updates=3)
    assert (run_dir / "checkpoints/step_0003.pt").is_file()
    assert json.loads((run_dir / "checkpoints/LATEST.json").read_text())["completed_updates"] == 2
    assert json.loads((run_dir / "STATUS.json").read_text())["completed_updates"] == 2
    monkeypatch.setattr(trainer, "atomic_json", original_atomic_json)
    result = trainer._train_cell_for_test(interrupted, resume=True, max_updates=4)
    assert isinstance(result, CompletedRun) and result.completed_updates == 4
    baseline_dir = Path(baseline["run"]["path"])
    for name in (
        "final_logits.npy", "final_density.npy", "final_hard_005.npy",
        "metrics.json", "audit.json", "resource.json",
    ):
        assert (run_dir / "artifacts" / name).read_bytes() == (
            baseline_dir / "artifacts" / name
        ).read_bytes()


def test_resume_rejects_corrupt_unpublished_checkpoint_generation_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _test_config(tmp_path / "corrupt_orphan" / "run", total=4)
    trainer._train_cell_for_test(config, max_updates=2)
    run_dir = Path(config["run"]["path"])
    original_atomic_json = trainer.atomic_json

    def crash_before_pointer(path, payload, *, create_once=False):
        if (
            Path(path).name == "LATEST.json"
            and payload.get("completed_updates") == 3
        ):
            raise RuntimeError("injected crash before checkpoint pointer publication")
        return original_atomic_json(path, payload, create_once=create_once)

    monkeypatch.setattr(trainer, "atomic_json", crash_before_pointer)
    with pytest.raises(RuntimeError, match="pointer publication"):
        trainer._train_cell_for_test(config, resume=True, max_updates=3)
    orphan = run_dir / "checkpoints/step_0003.pt"
    orphan.write_bytes(orphan.read_bytes() + b"corrupt")
    monkeypatch.setattr(trainer, "atomic_json", original_atomic_json)
    before = _tree(run_dir)
    with pytest.raises(ValueError, match="checkpoint|generation|orphan|payload"):
        trainer._train_cell_for_test(config, resume=True, max_updates=4)
    assert _tree(run_dir) == before


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("adam", lambda payload: payload.__setitem__("optimizer", {"state": {}, "param_groups": []})),
        ("rng", lambda payload: payload.__setitem__("rng", {})),
        ("audit", lambda payload: payload["audit"].__setitem__("clip_calls", 0)),
        ("history_prefix", lambda payload: payload["history"][0].__setitem__("base_loss", 123.0)),
        ("telemetry_times", lambda payload: payload["step_times"].__setitem__(0, 123.0)),
        ("final_logits", lambda payload: payload.__setitem__("logits", torch.zeros_like(payload["logits"]))),
    ],
)
def test_completed_validator_rejects_coherently_rehashed_forged_checkpoint_state(
    tmp_path: Path, name: str, mutate: object
) -> None:
    config = _test_config(tmp_path / name / "run", total=4)
    trainer._train_cell_for_test(config, max_updates=4)
    run_dir = Path(config["run"]["path"])
    _rewrite_final_checkpoint(run_dir, mutate)
    with pytest.raises(ValueError, match="checkpoint|Adam|RNG|audit|history|telemetry|logits"):
        validate_completed_run(config)


@pytest.mark.parametrize(
    "mutation",
    [
        "non_npy",
        "logits_relation",
        "density_relation",
        "hard_relation",
        "shell_metrics",
        "recomputed_metrics",
        "audit",
        "resource",
    ],
)
def test_completed_validator_reconstructs_terminal_evidence_not_only_hashes(
    tmp_path: Path, mutation: str
) -> None:
    config = _test_config(tmp_path / mutation / "run", total=4)
    trainer._train_cell_for_test(config, max_updates=4)
    run_dir = Path(config["run"]["path"])
    changed: set[str]
    if mutation == "non_npy":
        relative = "artifacts/final_density.npy"
        (run_dir / relative).write_bytes(b"not-an-npy")
        changed = {relative}
    elif mutation == "logits_relation":
        relative = "artifacts/final_logits.npy"
        observed = np.load(run_dir / relative, allow_pickle=False)
        _write_npy(run_dir / relative, np.zeros_like(observed, dtype=np.float32))
        changed = {relative}
    elif mutation == "density_relation":
        relative = "artifacts/final_density.npy"
        observed = np.load(run_dir / relative, allow_pickle=False)
        _write_npy(run_dir / relative, np.zeros_like(observed, dtype=np.float32))
        changed = {relative}
    elif mutation == "hard_relation":
        relative = "artifacts/final_hard_005.npy"
        observed = np.load(run_dir / relative, allow_pickle=False)
        _write_npy(run_dir / relative, np.zeros_like(observed, dtype=np.uint8))
        changed = {relative}
    elif mutation == "shell_metrics":
        relative = "artifacts/metrics.json"
        (run_dir / relative).write_bytes(
            canonical_json_bytes(
                {"completed_updates": 4, "per_view": [{}, {}, {}, {}, {}, {}]}
            )
        )
        changed = {relative}
    elif mutation == "recomputed_metrics":
        relative = "artifacts/metrics.json"
        metrics = json.loads((run_dir / relative).read_text())
        metrics["per_view"][0]["tp"] += 1
        (run_dir / relative).write_bytes(canonical_json_bytes(metrics))
        changed = {relative}
    elif mutation == "audit":
        relative = "artifacts/audit.json"
        (run_dir / relative).write_bytes(canonical_json_bytes({}))
        changed = {relative}
    else:
        relative = "artifacts/resource.json"
        (run_dir / relative).write_bytes(canonical_json_bytes({}))
        changed = {relative}
    _rehash_completed_artifacts(run_dir, changed)
    with pytest.raises(ValueError, match="artifact|NPY|logits|density|hard|metrics|audit|resource"):
        validate_completed_run(config)


def test_full_production_checkpoint_schedule_and_diagnostics_are_frozen() -> None:
    assert trainer.required_checkpoint_steps(3000) == (
        0, 2, *tuple(range(200, 3001, 200))
    )
    assert trainer.required_diagnostic_steps(3000) == (2400, 2700, 3000)
    assert 2700 not in trainer.required_checkpoint_steps(3000)


def test_full_ramp_audit_counts_actual_updates_901_through_3000_exactly() -> None:
    audit = trainer._initial_audit("full")
    for completed in range(1, 3001):
        trainer._update_audit(audit, _telemetry_row(completed), "full")
    assert audit["eligible_updates"] == 2400
    assert audit["j_full_ramp_eligible"] == 2100
    assert audit["h_full_ramp_eligible"] == 2100


def test_scientific_failure_publishes_terminal_failed_at_last_durable_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _test_config(tmp_path / "failed", total=4)

    def fail(**kwargs: object) -> StepTelemetry:
        raise trainer.TrainingStateError("injected nonfinite state")

    monkeypatch.setattr(trainer, "_train_one_update", fail)
    with pytest.raises(trainer.TrainingStateError, match="nonfinite"):
        trainer._train_cell_for_test(config, max_updates=2)
    status = json.loads((Path(config["run"]["path"]) / "STATUS.json").read_text())
    assert status == {
        "schema": "pcraft_soft_loss_v6_status_v1",
        "state": "failed",
        "config_sha256": canonical_hash(config),
        "completed_updates": 0,
        "error": "TrainingStateError: injected nonfinite state",
    }
    with pytest.raises(ValueError, match="running|failed"):
        validate_resumable_run(config)
