"""Fail-closed lifecycle contracts for the isolated soft-loss-v3 trainer."""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, fields
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Mapping

import cv2
import numpy as np
import pytest
import torch

import shadowforge_pipeline.soft_voxel_v3 as v3_public
import shadowforge_pipeline.soft_voxel_v3.provenance as provenance_module
import shadowforge_pipeline.soft_voxel_v3.trainer as trainer_module
from shadowforge_pipeline.soft_voxel_v3.contours import ContourBands
from shadowforge_pipeline.soft_voxel_v3.data import load_targets
from shadowforge_pipeline.soft_voxel_v3.diagnostics import (
    ComponentGradientSample,
    measure_component_gradients,
)
from shadowforge_pipeline.soft_voxel_v3.losses import resolve_profile
from shadowforge_pipeline.soft_voxel_v3.provenance import (
    atomic_checkpoint,
    atomic_json,
    canonical_hash,
    implementation_source_manifest,
)
from shadowforge_pipeline.soft_voxel_v3.trainer import (
    ResumeState,
    SoftTrainingStepError,
    StepRecord,
    build_checkpoint_payload,
    build_parser,
    commit_training_transaction,
    resolved_config,
    restore_checkpoint_payload,
    restore_latest_transaction,
    run,
)


THRESHOLDS = (0.04, 0.05, 0.06)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    shadows: list[Path] = []
    for axis, offset in zip("xyz", (0, 1, 2)):
        image = np.zeros((5, 5), dtype=np.uint8)
        image[1:4, 1 + (offset % 2) : 4] = 255
        path = tmp_path / f"{axis}.png"
        assert cv2.imwrite(str(path), image)
        shadows.append(path)
    suite = tmp_path / "suite.json"
    spec = tmp_path / "spec.md"
    weights = tmp_path / "weights.json"
    suite.write_text('{"schema":"toy_suite","tasks":["toy"]}\n', encoding="utf-8")
    spec.write_text("frozen toy specification\n", encoding="utf-8")
    weights.write_text(
        json.dumps(
            {
                "schema": "pcraft_soft_loss_v3_weights_v1",
                "status": "APPROVED_FROZEN",
                "lambdas": {"lambda_dic": 0.2, "lambda_dts": 0.3},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"shadows": shadows, "suite": suite, "spec": spec, "weights": weights}


def _args(
    tmp_path: Path,
    output: Path,
    *,
    steps: int = 3,
    profile: str = "v3_full",
    calibration: bool = False,
    resume: bool = False,
):
    inputs = _write_inputs(tmp_path)
    argv = [
        "cpu",
        str(output),
        str(steps),
        "0.007",
        "--shadow-files",
        *(str(path) for path in inputs["shadows"]),
        "--res",
        "128",
        "--export-res",
        "128",
        "--profile",
        profile,
        "--lambda-dic",
        "0.2" if not calibration else "0.0",
        "--lambda-dts",
        "0.3" if not calibration else "0.0",
        "--suite-manifest",
        str(inputs["suite"]),
        "--spec",
        str(inputs["spec"]),
        "--soft-only-eval",
    ]
    if calibration:
        argv.extend(("--calibration-probe",))
    else:
        argv.extend(("--weight-config", str(inputs["weights"])))
    if resume:
        argv.extend(("--resume",))
    return build_parser().parse_args(argv), inputs


def _step_record(completed: int, *, active: bool = True) -> StepRecord:
    before = completed - 1
    return StepRecord(
        global_step_before=before,
        completed_updates_after=completed,
        progress=completed / 200.0,
        dts_endpoint=(0.04 if before % 2 == 0 else 0.06) if active else None,
        loss_base=1.0 / completed,
        loss_dic=0.2 if active else None,
        loss_dts=0.3 if active else None,
        dic_scale=0.1 if active else 0.0,
        dts_scale=0.2 if active else 0.0,
        pre_clip_grad_norm=0.7,
        post_clip_grad_norm=0.7,
    )


def _history_row(record: StepRecord) -> dict[str, object]:
    row = asdict(record)
    row["effective_loss"] = (
        record.loss_base
        + record.dic_scale * (record.loss_dic or 0.0)
        + record.dts_scale * (record.loss_dts or 0.0)
    )
    return row


def _sample(
    logits: torch.Tensor, completed_updates: int = 100
) -> ComponentGradientSample:
    def builder(gradient: list[float]):
        value = torch.as_tensor(gradient, dtype=logits.dtype).reshape_as(logits)
        return lambda: (logits * value).sum()

    return measure_component_gradients(
        logits,
        completed_updates=completed_updates,
        base_builder=builder([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        dic_builder=builder([0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        dts_builders={
            0.04: builder([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            0.06: builder([0.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        },
        dic_scale=0.2,
        dts_scale=0.3,
        selected_dts_tau=0.04,
    )


def _calibration_record_for(sample: ComponentGradientSample) -> dict[str, object]:
    assert sample.dic_norm is not None
    return {
        "completed_updates": sample.completed_updates,
        "g0": sample.base_norm,
        "g_dic": sample.dic_norm,
        "g_dts_004": sample.dts_norms[0.04],
        "g_dts_006": sample.dts_norms[0.06],
    }


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": [],
    }


def _checkpoint(
    logits: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    completed: int,
    *,
    config_hash: str = "a" * 64,
    history_active: bool = True,
) -> dict[str, object]:
    last = None if completed == 0 else _step_record(completed, active=history_active)
    return build_checkpoint_payload(
        logits=logits,
        optimizer=optimizer,
        completed_updates=completed,
        last_record=last,
        rng_state=_rng_state(),
        gradient_samples=() if completed < 100 else (_sample(logits),),
        step_times=() if completed == 0 else (0.125,) * completed,
        peak_cuda_bytes=0,
        calibration_records=(),
        config_hash=config_hash,
    )


def _optimizer_pair() -> tuple[torch.Tensor, torch.optim.Optimizer]:
    logits = torch.linspace(-0.4, 0.3, 8, dtype=torch.float64).reshape(
        1, 1, 2, 2, 2
    ).requires_grad_()
    optimizer = torch.optim.Adam([logits], lr=0.01)
    return logits, optimizer


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
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def _tree_snapshot(root: Path) -> dict[str, tuple[str, object]]:
    snapshot: dict[str, tuple[str, object]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes())
        elif path.is_dir():
            snapshot[relative] = ("directory", None)
    return snapshot


def _terminal_transition_worker(
    run_dir: str,
    state: str,
    result_queue,
) -> None:
    try:
        kwargs = (
            {"artifact_manifest_sha256": "b" * 64}
            if state == "completed"
            else {"diagnostic": f"../.{Path(run_dir).name}.failure.json"}
        )
        trainer_module._cas_terminal_status(
            Path(run_dir),
            config_hash="a" * 64,
            state=state,
            completed_updates=1,
            **kwargs,
        )
    except BaseException as error:
        result_queue.put((state, "error", type(error).__name__, str(error)))
    else:
        result_queue.put(
            (state, "success", (Path(run_dir) / "STATUS.json").read_bytes())
        )


def _transaction_commit_worker(
    run_dir: str,
    checkpoint: Mapping[str, object],
    rows: tuple[Mapping[str, object], ...],
    previous_digest: str | None,
    result_queue,
) -> None:
    try:
        generation = commit_training_transaction(
            Path(run_dir),
            checkpoint_payload=checkpoint,
            history_segment=rows,
            previous_transaction_sha256=previous_digest,
        )
    except BaseException as error:
        result_queue.put(("error", type(error).__name__, str(error)))
    else:
        result_queue.put(("success", generation.name))


def _fresh_run_worker(
    args: SimpleNamespace,
    result_queue,
    before_initial_status,
    winner_entered,
    completed_event,
) -> None:
    trainer_module._build_runtime = _toy_runtime
    trainer_module._timer = lambda: 0.0
    trainer_module.torch.cuda.is_available = lambda: False
    original_atomic_json = trainer_module.atomic_json

    def synchronized_initial_status(path: Path, payload: Mapping[str, object]) -> None:
        if Path(path).name == "COMMAND.json":
            if multiprocessing.current_process().name == "fresh-winner":
                winner_entered.set()
            try:
                before_initial_status.wait(timeout=0.4)
            except threading.BrokenBarrierError:
                pass
        if (
            Path(path).name == "STATUS.json"
            and payload.get("state") == "running"
            and multiprocessing.current_process().name == "fresh-delayed"
        ):
            assert completed_event.wait(timeout=5)
        original_atomic_json(path, payload)

    trainer_module.atomic_json = synchronized_initial_status
    try:
        run(args, command_argv=("toy-fresh-race",))
    except BaseException as error:
        result_queue.put(("error", type(error).__name__, str(error)))
    else:
        if multiprocessing.current_process().name == "fresh-winner":
            completed_event.set()
        result_queue.put(("success",))


def test_canonical_hash_is_stable_strict_and_rejects_unsupported_values() -> None:
    """Break caught: provenance stringifies tensors/nonfinites or depends on key order."""

    assert canonical_hash({"b": [2, 3], "a": {"x": True}}) == canonical_hash(
        {"a": {"x": True}, "b": [2, 3]}
    )
    assert canonical_hash({"x": 1}) != canonical_hash({"x": 1.0})
    for value in (float("nan"), float("inf"), torch.tensor(1.0), object()):
        with pytest.raises((TypeError, ValueError)):
            canonical_hash({"value": value})


def test_source_manifest_hashes_only_authoritative_python_and_detects_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: source hashing omits a production module or includes tests/cache."""

    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 2\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.py").write_text("C = 3\n", encoding="utf-8")
    (tmp_path / "test_a.py.txt").write_text("ignored\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.pyc").write_bytes(b"ignored")
    (tmp_path / "__pycache__" / "hidden.py").write_text("ignored\n")
    (nested / "__pycache__").mkdir()
    (nested / "__pycache__" / "hidden.py").write_text("ignored\n")
    monkeypatch.setattr(provenance_module, "AUTHORITATIVE_SOURCE_ROOT", tmp_path)

    first = implementation_source_manifest()
    assert set(first["files_sha256"]) == {"a.py", "b.py", "nested/c.py"}
    (nested / "c.py").write_text("C = 4\n", encoding="utf-8")
    second = implementation_source_manifest()
    assert first["bundle_sha256"] != second["bundle_sha256"]


def test_resolved_config_binds_xyz_suite_spec_weight_interpreter_and_byte_drift(
    tmp_path: Path,
) -> None:
    """Break caught: a scientific input changes without changing the run hash."""

    args, inputs = _args(tmp_path, tmp_path / "run")
    config = resolved_config(args)
    first_hash = canonical_hash(config)

    assert [Path(item["path"]).name for item in config["shadow_inputs_xyz"]] == [
        "x.png",
        "y.png",
        "z.png",
    ]
    assert config["interpreter"]["path"] == str(Path(sys.executable).resolve())
    assert config["interpreter"]["sha256"] == _sha(Path(sys.executable).resolve())
    assert config["soft_only"] is True
    assert config["determinism"]["fixed_seed"] is True
    assert config["determinism"]["bitwise_reproducible"] is False

    inputs["suite"].write_text("changed suite bytes\n", encoding="utf-8")
    assert canonical_hash(resolved_config(args)) != first_hash


def test_environment_capture_is_allowlisted_secret_safe_and_explicitly_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: environment.json leaks credentials or overclaims determinism."""

    monkeypatch.setenv("PYTHONHASHSEED", "17")
    monkeypatch.setenv("OPENAI_API_KEY", "never-serialize-this")
    monkeypatch.setenv("MY_PASSWORD", "never-serialize-this-either")
    payload = provenance_module.environment_manifest(torch.device("cpu"))
    serialized = json.dumps(payload, allow_nan=False)

    assert payload["environment"] == {
        key: os.environ[key]
        for key in payload["environment"]
    }
    assert set(payload["environment"]).issubset(provenance_module.ALLOWED_ENVIRONMENT)
    assert "never-serialize" not in serialized
    assert payload["fixed_seed"] is True
    assert payload["determinism"] == "best_effort"
    assert payload["bitwise_reproducible"] is False
    assert "grid_sampler_3d_backward" in payload["nondeterminism_reason"]


def test_cuda_driver_provenance_uses_a_controlled_nvidia_smi_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: missing torch CUDA API silently records a null driver."""

    calls: list[tuple[object, ...]] = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return SimpleNamespace(returncode=0, stdout="580.173.02\n", stderr="")

    monkeypatch.setattr(torch.cuda, "driver_version", lambda: None, raising=False)
    monkeypatch.setattr(
        provenance_module,
        "subprocess",
        SimpleNamespace(run=fake_run),
        raising=False,
    )
    assert provenance_module._cuda_driver_version() == "580.173.02"
    assert calls and calls[0][0] == (
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host required")
def test_cuda_environment_manifest_records_nonempty_host_driver() -> None:
    """Break caught: an available CUDA device emits null driver provenance."""

    payload = provenance_module.environment_manifest(torch.device("cuda"))
    assert isinstance(payload["cuda_driver"], str)
    assert payload["cuda_driver"].strip()


def test_parser_locks_protocol_profile_weights_calibration_and_soft_only(tmp_path: Path) -> None:
    """Break caught: unknown/drifted protocols or unfrozen normal weights reach a run dir."""

    valid, _ = _args(tmp_path, tmp_path / "valid")
    assert resolved_config(valid)["protocol"] == {
        "mirror_mode": 1,
        "tsf_mode": 1,
        "tsf4_mode": 0,
        "camera_mode": "ortho",
        "image_size": 256,
        "control_res": 0,
        "zdist": 1.7,
        "volume_extent": 1.7,
        "samples_per_ray": 150,
    }

    drifted, _ = _args(tmp_path / "drift", tmp_path / "drift" / "run")
    drifted.image_size = 255
    with pytest.raises(ValueError, match="image"):
        resolved_config(drifted)
    assert not Path(drifted.output_dir).exists()

    normal_without_weight, _ = _args(tmp_path / "nw", tmp_path / "nw" / "run")
    normal_without_weight.weight_config = None
    with pytest.raises(ValueError, match="weight"):
        resolved_config(normal_without_weight)

    bad_calibration, _ = _args(
        tmp_path / "cal", tmp_path / "cal" / "run", profile="v3_full"
    )
    bad_calibration.calibration_probe = True
    bad_calibration.weight_config = None
    with pytest.raises(ValueError, match="l12_bal"):
        resolved_config(bad_calibration)


def test_learning_rate_is_frozen_to_exact_0p007_before_run_directory_write(
    tmp_path: Path,
) -> None:
    """Break caught: an arbitrary positive optimizer learning rate enters the run."""

    valid, _ = _args(tmp_path / "valid", tmp_path / "valid" / "run")
    assert resolved_config(valid)["learning_rate"] == 0.007
    drifted, _ = _args(tmp_path / "drift", tmp_path / "drift" / "run")
    drifted.lr = 0.123

    with pytest.raises(ValueError, match="0.007"):
        resolved_config(drifted)

    assert not Path(drifted.output_dir).exists()


def test_bound_shadow_snapshot_matches_real_loader_and_survives_path_replacement(
    tmp_path: Path,
) -> None:
    """Break caught: runtime reopens shadow paths after their bytes were hashed."""

    args, inputs = _args(tmp_path, tmp_path / "run")
    expected = load_targets(
        args.shadow_files,
        True,
        args.image_size,
        torch.device("cpu"),
    )
    resolved_config(args)
    bound_before = trainer_module._load_bound_targets(args, torch.device("cpu"))
    assert torch.equal(bound_before, expected)
    for index, path in enumerate(inputs["shadows"]):
        replacement = np.zeros((5, 5), dtype=np.uint8)
        replacement[index + 1, :] = 255
        assert cv2.imwrite(str(path), replacement)

    bound_after = trainer_module._load_bound_targets(args, torch.device("cpu"))
    reopened = load_targets(
        args.shadow_files,
        True,
        args.image_size,
        torch.device("cpu"),
    )
    assert torch.equal(bound_after, expected)
    assert not torch.equal(reopened, expected)


def test_real_runtime_consumes_bound_shadow_snapshot_not_reopened_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the real runtime bypasses the byte-snapshot decoder."""

    args, inputs = _args(tmp_path, tmp_path / "run", profile="l12_bal")
    expected = load_targets(
        args.shadow_files,
        True,
        args.image_size,
        torch.device("cpu"),
    )
    original_seed = trainer_module._seed_everything
    original_build = trainer_module._build_runtime
    observed: dict[str, torch.Tensor] = {}

    def replace_after_config(seed: int) -> None:
        original_seed(seed)
        for index, path in enumerate(inputs["shadows"]):
            replacement = np.zeros((5, 5), dtype=np.uint8)
            replacement[:, index + 1] = 255
            assert cv2.imwrite(str(path), replacement)

    def capture_runtime(*positional, **keywords):
        runtime = original_build(*positional, **keywords)
        observed["targets"] = runtime.targets.detach().clone()
        raise _InjectedInterruption

    monkeypatch.setattr(trainer_module, "_seed_everything", replace_after_config)
    monkeypatch.setattr(
        trainer_module,
        "SadekarTFSProtocol",
        lambda **keywords: _ToyProtocol(keywords["device"]),
    )
    monkeypatch.setattr(trainer_module, "_build_runtime", capture_runtime)
    with pytest.raises(_InjectedInterruption):
        run(args, command_argv=("real-runtime-probe",))

    assert torch.equal(observed["targets"], expected)
    assert not Path(args.output_dir).exists()


def test_real_module_cli_help_has_one_trainer_module_and_no_runpy_warning() -> None:
    """Break caught: eager package exports preload and then re-execute trainer."""

    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::RuntimeWarning",
            "-m",
            "shadowforge_pipeline.soft_voxel_v3.trainer",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "P-CRAFT soft-loss-v3 trainer" in completed.stdout
    assert "RuntimeWarning" not in completed.stderr
    assert v3_public.ResumeState is ResumeState
    assert v3_public.SoftTrainingStepError is SoftTrainingStepError
    assert v3_public.StepRecord is StepRecord
    assert v3_public.build_parser is build_parser
    assert v3_public.run is run
    assert v3_public.train_one_update is trainer_module.train_one_update


def test_checkpoint_round_trip_restores_dataclasses_rng_optimizer_and_float_keys(
    tmp_path: Path,
) -> None:
    """Break caught: JSON key coercion or resume omits exact optimizer/RNG state."""

    logits, optimizer = _optimizer_pair()
    (logits.square().sum()).backward()
    optimizer.step()
    record = _step_record(100)
    sample_payload = json.loads(json.dumps(asdict(_sample(logits))))
    sample = ComponentGradientSample(**trainer_module._component_sample_from_mapping(sample_payload))
    payload = build_checkpoint_payload(
        logits=logits,
        optimizer=optimizer,
        completed_updates=100,
        last_record=record,
        rng_state=_rng_state(),
        gradient_samples=(sample,),
        step_times=(0.1,) * 100,
        peak_cuda_bytes=123,
        calibration_records=(),
        config_hash="b" * 64,
    )
    path = tmp_path / "checkpoint.pt"
    atomic_checkpoint(path, payload)

    restored_logits, restored_optimizer = _optimizer_pair()
    state = restore_checkpoint_payload(
        path,
        expected_config_hash="b" * 64,
        logits=restored_logits,
        optimizer=restored_optimizer,
    )

    assert isinstance(state, ResumeState)
    assert state.completed_updates == 100
    assert state.last_record == record
    assert state.gradient_samples[0] == sample
    assert set(state.gradient_samples[0].dts_norms) == {0.04, 0.06}
    assert set(state.gradient_samples[0].dts_pair_cosines[0.04]) == {0.06}
    assert state.step_times == (0.1,) * 100
    assert state.peak_cuda_bytes == 123
    assert torch.equal(restored_logits, logits)
    assert list(restored_optimizer.param_groups[0]["params"]) == [restored_logits]
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())


@pytest.mark.parametrize(
    "malformation",
    [
        "extra",
        "missing",
        "nan",
        "negative",
        "duplicate",
        "unregistered",
        "future",
        "mismatch",
    ],
)
def test_calibration_records_are_exact_registered_finite_and_sample_consistent(
    tmp_path: Path, malformation: str
) -> None:
    """Break caught: malformed calibration telemetry survives build or mutates restore."""

    logits, optimizer = _optimizer_pair()
    valid_sample = _sample(logits, 1000)
    samples = [valid_sample]
    records = [_calibration_record_for(valid_sample)]
    if malformation == "extra":
        records[0]["extra"] = 1
    elif malformation == "missing":
        records[0].pop("g_dic")
    elif malformation == "nan":
        records[0]["g0"] = float("nan")
    elif malformation == "negative":
        records[0]["g_dts_004"] = -1.0
    elif malformation == "duplicate":
        records.append(dict(records[0]))
    elif malformation == "unregistered":
        samples = [_sample(logits, 999)]
        records = [_calibration_record_for(samples[0])]
    elif malformation == "future":
        records[0]["completed_updates"] = 1300
    elif malformation == "mismatch":
        records[0]["g0"] = float(records[0]["g0"]) + 1.0

    with pytest.raises(ValueError):
        build_checkpoint_payload(
            logits=logits,
            optimizer=optimizer,
            completed_updates=1000,
            last_record=_step_record(1000),
            rng_state=_rng_state(),
            gradient_samples=samples,
            step_times=(0.1,) * 1000,
            peak_cuda_bytes=0,
            calibration_records=records,
            config_hash="9" * 64,
        )

    valid_payload = build_checkpoint_payload(
        logits=logits,
        optimizer=optimizer,
        completed_updates=1000,
        last_record=_step_record(1000),
        rng_state=_rng_state(),
        gradient_samples=(valid_sample,),
        step_times=(0.1,) * 1000,
        peak_cuda_bytes=0,
        calibration_records=(_calibration_record_for(valid_sample),),
        config_hash="9" * 64,
    )
    valid_payload["gradient_samples"] = [asdict(sample) for sample in samples]
    valid_payload["calibration_records"] = records
    path = tmp_path / f"bad-{malformation}.pt"
    atomic_checkpoint(path, valid_payload)
    restored_logits, restored_optimizer = _optimizer_pair()
    logits_before = restored_logits.detach().clone()
    optimizer_before = copy.deepcopy(restored_optimizer.state_dict())

    with pytest.raises(ValueError):
        restore_checkpoint_payload(
            path,
            expected_config_hash="9" * 64,
            logits=restored_logits,
            optimizer=restored_optimizer,
        )

    assert torch.equal(restored_logits, logits_before)
    _assert_nested_equal(restored_optimizer.state_dict(), optimizer_before)


@pytest.mark.parametrize(
    ("completed", "last"),
    [(0, _step_record(1)), (1, None), (2, _step_record(1))],
)
def test_checkpoint_rejects_step_zero_positive_last_record_mismatch(
    completed: int, last: StepRecord | None
) -> None:
    """Break caught: resume fabricates, omits, or shifts the last completed update."""

    logits, optimizer = _optimizer_pair()
    with pytest.raises(ValueError, match="last_record"):
        build_checkpoint_payload(
            logits=logits,
            optimizer=optimizer,
            completed_updates=completed,
            last_record=last,
            rng_state=_rng_state(),
            gradient_samples=(),
            step_times=(),
            peak_cuda_bytes=0,
            calibration_records=(),
            config_hash="c" * 64,
        )


def test_checkpoint_corruption_and_foreign_optimizer_fail_before_mutation(
    tmp_path: Path,
) -> None:
    """Break caught: corrupt state partially mutates live logits or a foreign optimizer."""

    logits, optimizer = _optimizer_pair()
    payload = _checkpoint(logits, optimizer, 0, config_hash="d" * 64)
    payload["unknown"] = 1
    path = tmp_path / "bad.pt"
    atomic_checkpoint(path, payload)
    restored_logits, restored_optimizer = _optimizer_pair()
    before = restored_logits.detach().clone()
    with pytest.raises(ValueError):
        restore_checkpoint_payload(
            path,
            expected_config_hash="d" * 64,
            logits=restored_logits,
            optimizer=restored_optimizer,
        )
    assert torch.equal(restored_logits, before)

    foreign = torch.zeros_like(restored_logits, requires_grad=True)
    foreign_optimizer = torch.optim.Adam([foreign], lr=0.01)
    with pytest.raises(ValueError, match="exact live logits"):
        trainer_module._validate_optimizer_owns_logits(restored_optimizer, foreign)
    with pytest.raises(ValueError, match="exact live logits"):
        trainer_module._validate_optimizer_owns_logits(foreign_optimizer, restored_logits)


def test_checkpoint_rejects_primary_endpoint_parity_drift_before_mutation(
    tmp_path: Path,
) -> None:
    """Break caught: the restored 0.04/0.06 cycle silently resets after interruption."""

    logits, optimizer = _optimizer_pair()
    payload = _checkpoint(logits, optimizer, 1, config_hash="e" * 64)
    payload["last_record"]["dts_endpoint"] = 0.06
    path = tmp_path / "wrong-parity.pt"
    atomic_checkpoint(path, payload)
    restored_logits, restored_optimizer = _optimizer_pair()
    before = restored_logits.detach().clone()

    with pytest.raises(ValueError, match="parity"):
        restore_checkpoint_payload(
            path,
            expected_config_hash="e" * 64,
            logits=restored_logits,
            optimizer=restored_optimizer,
        )

    assert torch.equal(restored_logits, before)


def test_transaction_step_zero_and_boundary_chain_are_append_only_and_verifiable(
    tmp_path: Path,
) -> None:
    """Break caught: checkpoints and history can be mixed across commit boundaries."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    step0 = commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    pointer0 = json.loads((run_dir / "checkpoint_latest.json").read_text())
    assert step0.relative_to(run_dir).as_posix() == "transactions/step_00000000"
    assert pointer0["transaction"] == "transactions/step_00000000"

    rows = [_history_row(_step_record(step)) for step in range(1, 101)]
    step100 = commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(logits, optimizer, 100),
        history_segment=rows,
        previous_transaction_sha256=pointer0["manifest_sha256"],
    )
    manifest = json.loads((step100 / "transaction_manifest.json").read_text())
    assert manifest["history_interval"] == {
        "first_completed_update": 1,
        "last_completed_update": 100,
        "row_count": 100,
    }
    assert manifest["previous_transaction_sha256"] == pointer0["manifest_sha256"]

    restored_logits, restored_optimizer = _optimizer_pair()
    state = restore_latest_transaction(
        run_dir,
        expected_config_hash="a" * 64,
        logits=restored_logits,
        optimizer=restored_optimizer,
    )
    assert state.completed_updates == 100
    assert len(state.committed_history) == 100
    assert state.committed_history[-1]["completed_updates_after"] == 100
    assert state.committed_transaction == "transactions/step_00000100"


def test_transaction_rejects_malformed_checkpoint_before_generation_write(
    tmp_path: Path,
) -> None:
    """Break caught: commit persists a payload that restore must later reject."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    malformed = _checkpoint(logits, optimizer, 0)
    malformed["unknown"] = "not part of the checkpoint schema"

    with pytest.raises(ValueError, match="checkpoint"):
        commit_training_transaction(
            run_dir,
            checkpoint_payload=malformed,
            history_segment=(),
            previous_transaction_sha256=None,
        )

    assert not (run_dir / "checkpoint_latest.json").exists()
    assert not (run_dir / "transactions").exists()


@pytest.mark.parametrize("corruption", ["checkpoint_hash", "interval", "predecessor"])
def test_transaction_chain_hash_interval_and_predecessor_corruption_fail_closed(
    tmp_path: Path, corruption: str
) -> None:
    """Break caught: resume accepts a tampered linked transaction generation."""

    run_dir = tmp_path / corruption
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    old_pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
    generation = commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(logits, optimizer, 1),
        history_segment=(_history_row(_step_record(1)),),
        previous_transaction_sha256=old_pointer["manifest_sha256"],
    )
    if corruption == "checkpoint_hash":
        (generation / "checkpoint.pt").write_bytes(b"tampered")
    else:
        manifest_path = generation / "transaction_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if corruption == "interval":
            manifest["history_interval"]["first_completed_update"] = 0
        else:
            manifest["previous_transaction_sha256"] = "f" * 64
        atomic_json(manifest_path, manifest)
        pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
        pointer["manifest_sha256"] = _sha(manifest_path)
        atomic_json(run_dir / "checkpoint_latest.json", pointer)

    restored_logits, restored_optimizer = _optimizer_pair()
    before = restored_logits.detach().clone()
    with pytest.raises(ValueError):
        restore_latest_transaction(
            run_dir,
            expected_config_hash="a" * 64,
            logits=restored_logits,
            optimizer=restored_optimizer,
        )
    assert torch.equal(restored_logits, before)


def test_unreferenced_complete_and_temporary_generations_are_ignored(tmp_path: Path) -> None:
    """Break caught: an orphan generation is mistaken for committed progress."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    orphan = run_dir / "transactions" / "step_00000100"
    orphan.mkdir()
    (orphan / "transaction_manifest.json").write_text("corrupt orphan\n")
    temporary = run_dir / "transactions" / ".step_00000200.tmp.injected"
    temporary.mkdir()
    (temporary / "partial").write_bytes(b"partial")

    restored_logits, restored_optimizer = _optimizer_pair()
    state = restore_latest_transaction(
        run_dir,
        expected_config_hash="a" * 64,
        logits=restored_logits,
        optimizer=restored_optimizer,
    )
    assert state.completed_updates == 0
    assert state.committed_history == ()


def test_terminal_status_cas_is_process_safe_and_keeps_first_terminal_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: completed and failed writers both win one running STATUS race."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    atomic_json(
        run_dir / "STATUS.json",
        {
            "schema": "pcraft_soft_loss_v3_status_v1",
            "state": "running",
            "config_hash": "a" * 64,
            "completed_updates": 0,
        },
    )
    atomic_json(
        run_dir.parent / f".{run_dir.name}.failure.json",
        {
            "schema": "pcraft_soft_loss_v3_failure_v1",
            "config_hash": "a" * 64,
            "completed_updates": 1,
            "error_type": "SoftTrainingStepError",
            "error": "concurrent terminal transition",
        },
    )
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    original_atomic_json = trainer_module.atomic_json

    def synchronize_terminal_replace(path: Path, payload: Mapping[str, object]) -> None:
        if Path(path).name == "STATUS.json" and payload.get("state") in {
            "completed",
            "failed",
        }:
            try:
                barrier.wait(timeout=0.35)
            except threading.BrokenBarrierError:
                pass
        original_atomic_json(path, payload)

    monkeypatch.setattr(trainer_module, "atomic_json", synchronize_terminal_replace)
    results = context.Queue()
    processes = [
        context.Process(
            target=_terminal_transition_worker,
            args=(str(run_dir), state, results),
            name=f"terminal-{state}",
        )
        for state in ("completed", "failed")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    observed = [results.get(timeout=1) for _ in processes]

    winners = [item for item in observed if item[1] == "success"]
    losers = [item for item in observed if item[1] == "error"]
    assert len(winners) == 1, observed
    assert len(losers) == 1, observed
    assert (run_dir / "STATUS.json").read_bytes() == winners[0][2]


def test_concurrent_transaction_commit_cannot_publish_a_stale_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a delayed step 1 writer rolls the pointer back from step 2."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    predecessor = json.loads((run_dir / "checkpoint_latest.json").read_text())[
        "manifest_sha256"
    ]
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    newer_entered = context.Event()
    original_read_pointer = trainer_module._read_pointer

    def synchronized_predecessor_read(path: Path) -> dict[str, str]:
        pointer = original_read_pointer(path)
        process_name = multiprocessing.current_process().name
        if process_name == "newer":
            newer_entered.set()
        try:
            barrier.wait(timeout=0.4)
        except threading.BrokenBarrierError:
            pass
        if process_name == "stale":
            time.sleep(0.2)
        return pointer

    monkeypatch.setattr(trainer_module, "_read_pointer", synchronized_predecessor_read)
    results = context.Queue()
    newer = context.Process(
        target=_transaction_commit_worker,
        args=(
            str(run_dir),
            _checkpoint(logits, optimizer, 2),
            tuple(_history_row(_step_record(step)) for step in (1, 2)),
            predecessor,
            results,
        ),
        name="newer",
    )
    stale = context.Process(
        target=_transaction_commit_worker,
        args=(
            str(run_dir),
            _checkpoint(logits, optimizer, 1),
            (_history_row(_step_record(1)),),
            predecessor,
            results,
        ),
        name="stale",
    )
    newer.start()
    assert newer_entered.wait(timeout=2)
    stale.start()
    for process in (newer, stale):
        process.join(timeout=5)
        assert process.exitcode == 0
    observed = [results.get(timeout=1) for _ in range(2)]

    assert sorted(item[0] for item in observed) == ["error", "success"], observed
    assert ("success", "step_00000002") in observed
    pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
    assert pointer["transaction"] == "transactions/step_00000002"


def test_concurrent_step_zero_commit_has_exactly_one_winner(tmp_path: Path) -> None:
    """Break caught: two creators both publish an authoritative step-zero pointer."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    checkpoint = _checkpoint(logits, optimizer, 0)
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    writers = [
        context.Process(
            target=_transaction_commit_worker,
            args=(str(run_dir), checkpoint, (), None, results),
            name=f"step-zero-{index}",
        )
        for index in range(2)
    ]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=5)
        assert writer.exitcode == 0
    observed = [results.get(timeout=1) for _ in writers]

    assert sorted(item[0] for item in observed) == ["error", "success"]
    assert json.loads((run_dir / "checkpoint_latest.json").read_text())[
        "transaction"
    ] == "transactions/step_00000000"


def test_concurrent_fresh_runs_serialize_initialization_and_loser_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a delayed initializer overwrites a completed STATUS with running."""

    _install_toy(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    baseline_args, _ = _args(
        tmp_path / "inputs", tmp_path / "baseline" / "run", steps=2
    )
    run(baseline_args, command_argv=("toy-fresh-race",))
    baseline = _tree_snapshot(Path(baseline_args.output_dir))

    concurrent_args = SimpleNamespace(**vars(baseline_args))
    concurrent_args.output_dir = str(tmp_path / "concurrent" / "run")
    context = multiprocessing.get_context("spawn")
    before_initial_status = context.Barrier(2)
    winner_entered = context.Event()
    winner_completed = context.Event()
    results = context.Queue()
    winner = context.Process(
        target=_fresh_run_worker,
        args=(
            concurrent_args,
            results,
            before_initial_status,
            winner_entered,
            winner_completed,
        ),
        name="fresh-winner",
    )
    delayed = context.Process(
        target=_fresh_run_worker,
        args=(
            concurrent_args,
            results,
            before_initial_status,
            winner_entered,
            winner_completed,
        ),
        name="fresh-delayed",
    )
    winner.start()
    assert winner_entered.wait(timeout=5)
    delayed.start()
    for process in (winner, delayed):
        process.join(timeout=10)
        assert process.exitcode == 0
    observed = [results.get(timeout=1) for _ in range(2)]

    assert sorted(item[0] for item in observed) == ["error", "success"], observed
    concurrent_dir = Path(concurrent_args.output_dir)
    assert json.loads((concurrent_dir / "STATUS.json").read_text())["state"] == "completed"
    assert _tree_snapshot(concurrent_dir) == baseline


@pytest.mark.parametrize(
    "bad_name", ["step_0000000", "step_000000000", "step_deadbeef"]
)
def test_transaction_pointer_rejects_noncanonical_generation_names(
    tmp_path: Path, bad_name: str
) -> None:
    """Break caught: startswith('step_') accepts a noncanonical generation identity."""

    run_dir = tmp_path / bad_name
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    generation = commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    renamed = generation.with_name(bad_name)
    generation.rename(renamed)
    manifest_path = renamed / "transaction_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["transaction"] = f"transactions/{bad_name}"
    atomic_json(manifest_path, manifest)
    pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
    pointer["transaction"] = f"transactions/{bad_name}"
    pointer["manifest_sha256"] = _sha(manifest_path)
    atomic_json(run_dir / "checkpoint_latest.json", pointer)

    restored_logits, restored_optimizer = _optimizer_pair()
    with pytest.raises(ValueError, match="canonical|unsafe"):
        restore_latest_transaction(
            run_dir,
            expected_config_hash="a" * 64,
            logits=restored_logits,
            optimizer=restored_optimizer,
        )


def test_transaction_chain_rejects_predecessor_outside_run_directory(
    tmp_path: Path,
) -> None:
    """Break caught: a manifest predecessor traverses outside the run directory."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    step0 = commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    pointer0 = json.loads((run_dir / "checkpoint_latest.json").read_text())
    step1 = commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(logits, optimizer, 1),
        history_segment=(_history_row(_step_record(1)),),
        previous_transaction_sha256=pointer0["manifest_sha256"],
    )
    outside = tmp_path / "outside_generation"
    shutil.copytree(step0, outside)
    outside_manifest = outside / "transaction_manifest.json"
    outside_payload = json.loads(outside_manifest.read_text())
    outside_payload["transaction"] = "../outside_generation"
    atomic_json(outside_manifest, outside_payload)
    step1_manifest = step1 / "transaction_manifest.json"
    step1_payload = json.loads(step1_manifest.read_text())
    step1_payload["previous_transaction"] = "../outside_generation"
    step1_payload["previous_transaction_sha256"] = _sha(outside_manifest)
    atomic_json(step1_manifest, step1_payload)
    pointer1 = json.loads((run_dir / "checkpoint_latest.json").read_text())
    pointer1["manifest_sha256"] = _sha(step1_manifest)
    atomic_json(run_dir / "checkpoint_latest.json", pointer1)

    restored_logits, restored_optimizer = _optimizer_pair()
    with pytest.raises(ValueError, match="canonical|unsafe|outside"):
        restore_latest_transaction(
            run_dir,
            expected_config_hash="a" * 64,
            logits=restored_logits,
            optimizer=restored_optimizer,
        )


@pytest.mark.parametrize(
    "component", ["generation", "checkpoint.pt", "history_segment.jsonl", "transaction_manifest.json"]
)
def test_transaction_chain_rejects_symlinked_generation_or_payload_component(
    tmp_path: Path, component: str
) -> None:
    """Break caught: transaction validation follows a symlink outside its generation."""

    run_dir = tmp_path / component.replace(".", "_")
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    generation = commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    outside = tmp_path / f"outside_{component.replace('.', '_')}"
    if component == "generation":
        generation.rename(outside)
        generation.symlink_to(outside, target_is_directory=True)
    else:
        source = generation / component
        source.rename(outside)
        source.symlink_to(outside)

    restored_logits, restored_optimizer = _optimizer_pair()
    with pytest.raises(ValueError, match="symlink"):
        restore_latest_transaction(
            run_dir,
            expected_config_hash="a" * 64,
            logits=restored_logits,
            optimizer=restored_optimizer,
        )


def test_quarantine_rejects_symlink_without_moving_it_or_touching_target(
    tmp_path: Path,
) -> None:
    """Break caught: quarantine renames an attacker-controlled generation symlink."""

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    logits, optimizer = _optimizer_pair()
    commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(logits, optimizer, 0),
        history_segment=(),
        previous_transaction_sha256=None,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_bytes(b"outside-bytes")
    orphan = run_dir / "transactions" / "step_00000100"
    orphan.symlink_to(outside, target_is_directory=True)
    before = _tree_snapshot(run_dir)
    outside_before = _tree_snapshot(outside)

    with pytest.raises(ValueError, match="symlink"):
        trainer_module._quarantine_unreferenced_transactions(run_dir)

    assert _tree_snapshot(run_dir) == before
    assert _tree_snapshot(outside) == outside_before


class _ToyProtocol:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def render_all(self, density: torch.Tensor) -> torch.Tensor:
        view = density[0, 0].mean(dim=0)
        return view.unsqueeze(0).repeat(6, 1, 1)

    @staticmethod
    def binary_mask(projection: torch.Tensor) -> torch.Tensor:
        maximum = projection.detach().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return projection >= maximum * 0.8


def _toy_runtime(args, device: torch.device, profile):
    logits = torch.tensor(
        [-1.2, -0.8, -0.4, 0.1, 0.4, 0.7, 1.0, 1.3],
        dtype=torch.float64,
        device=device,
    ).reshape(1, 1, 2, 2, 2).requires_grad_()
    optimizer = torch.optim.Adam([logits], lr=float(args.lr))
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64, device=device)
    targets = target.unsqueeze(0).repeat(6, 1, 1)
    bands = None
    if profile.use_dic or bool(args.calibration_probe):
        binary = targets >= 0.5
        bands = ContourBands(inner=binary, outer=~binary)
    return trainer_module._Runtime(
        logits=logits,
        optimizer=optimizer,
        protocol=_ToyProtocol(device),
        targets=targets,
        bands=bands,
    )


class _InjectedInterruption(BaseException):
    pass


def _install_toy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trainer_module, "_build_runtime", _toy_runtime)
    monkeypatch.setattr(trainer_module, "_timer", lambda: 0.0)


def _load_checkpoint(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.parametrize(
    ("interrupted_at", "committed_at"), [(1, 0), (99, 0), (100, 100), (101, 100)]
)
def test_cpu_toy_interruption_resume_matches_uninterrupted_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_at: int,
    committed_at: int,
) -> None:
    """Break caught: resume duplicates/omits history or continues uncommitted state."""

    _install_toy(monkeypatch)
    uninterrupted_args, _ = _args(tmp_path / "inputs", tmp_path / "base" / "run", steps=103)
    run(uninterrupted_args, command_argv=("toy",))

    resumed_args = SimpleNamespace(**vars(uninterrupted_args))
    resumed_args.output_dir = str(tmp_path / "resume" / "run")

    def interrupt(completed_updates: int) -> None:
        if completed_updates == interrupted_at:
            raise _InjectedInterruption(str(completed_updates))

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_InjectedInterruption):
        run(resumed_args, command_argv=("toy",))
    status = json.loads((Path(resumed_args.output_dir) / "STATUS.json").read_text())
    assert status["state"] == "running"
    pointer = json.loads(
        (Path(resumed_args.output_dir) / "checkpoint_latest.json").read_text()
    )
    assert pointer["transaction"].endswith(f"step_{committed_at:08d}")

    monkeypatch.setattr(trainer_module, "_post_update_hook", lambda unused: None)
    resumed_args.resume = "__OUTPUT_LATEST__"
    run(resumed_args, command_argv=("toy", "--resume"))

    base_dir = Path(uninterrupted_args.output_dir)
    resumed_dir = Path(resumed_args.output_dir)
    assert (base_dir / "history.jsonl").read_bytes() == (
        resumed_dir / "history.jsonl"
    ).read_bytes()
    assert (base_dir / "metrics.json").read_bytes() == (
        resumed_dir / "metrics.json"
    ).read_bytes()
    assert (base_dir / "checkpoint_final.pt").read_bytes() == (
        resumed_dir / "checkpoint_final.pt"
    ).read_bytes()
    _assert_nested_equal(
        _load_checkpoint(base_dir / "checkpoint_final.pt"),
        _load_checkpoint(resumed_dir / "checkpoint_final.pt"),
    )
    assert np.array_equal(
        np.load(base_dir / "occupancy_probability.npy"),
        np.load(resumed_dir / "occupancy_probability.npy"),
    )
    base_audit = json.loads((base_dir / "gradient_audit.json").read_text())
    resumed_audit = json.loads((resumed_dir / "gradient_audit.json").read_text())
    assert resumed_audit == base_audit


def test_final_artifacts_hashes_binary_exports_nested_primary_and_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: final export performs a second readout or omits audit payloads."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=3)
    metrics = run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    required = {
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
        "artifact_manifest.json",
        "checkpoint_latest.json",
        "checkpoint_final.pt",
        "STATUS.json",
    }
    required.update(
        f"binary_tau_{label}_view_{view}.png"
        for label in ("0p04", "0p05", "0p06")
        for view in range(6)
    )
    assert required.issubset({path.name for path in run_dir.iterdir() if path.is_file()})
    assert "primary" not in metrics
    assert metrics["thresholds"]["0.05"]["aggregate"]["mean_iou"] >= 0.0
    for threshold in THRESHOLDS:
        label = f"{threshold:.2f}".replace(".", "p")
        occupancy = np.load(run_dir / f"occupancy_binary_tau_{label}.npy")
        assert occupancy.dtype == np.uint8
        assert set(np.unique(occupancy)).issubset({0, 1})
        for view in range(6):
            png = cv2.imread(
                str(run_dir / f"binary_tau_{label}_view_{view}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            assert png is not None
            assert set(np.unique(png)).issubset({0, 255})

    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert [row["completed_updates_after"] for row in history] == [1, 2, 3]
    assert history[-1]["effective_loss"] == pytest.approx(
        history[-1]["loss_base"]
        + history[-1]["dic_scale"] * history[-1]["loss_dic"]
        + history[-1]["dts_scale"] * history[-1]["loss_dts"]
    )
    resources = json.loads((run_dir / "resource_metrics.json").read_text())
    assert resources["step_times"] == [0.0, 0.0, 0.0]
    assert resources["updates_21_end"]["count"] == 0

    artifact_manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    listed = artifact_manifest["artifacts_sha256"]
    assert "artifact_manifest.json" not in listed
    assert "STATUS.json" not in listed
    assert "transactions/step_00000000/checkpoint.pt" in listed
    assert "transactions/step_00000003/transaction_manifest.json" in listed
    for relative, digest in listed.items():
        assert _sha(run_dir / relative) == digest
    status = json.loads((run_dir / "STATUS.json").read_text())
    assert status["state"] == "completed"
    assert status["artifact_manifest_sha256"] == _sha(run_dir / "artifact_manifest.json")


def test_probe_identities_disabled_nonconstruction_and_invalid_flags_are_conditional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: resume shifts probes or inactive diagnostics become failures."""

    _install_toy(monkeypatch)
    calls = {"bands": 0, "gradients": 0}
    original_measure = trainer_module.measure_component_gradients

    def recording_measure(*args, **kwargs):
        calls["gradients"] += 1
        if kwargs["dic_builder"] is not None:
            calls["bands"] += 1
        return original_measure(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "measure_component_gradients", recording_measure)
    args, _ = _args(
        tmp_path / "inactive", tmp_path / "inactive" / "run", steps=100, profile="l12_bal"
    )
    run(args, command_argv=("toy",))
    assert calls == {"bands": 0, "gradients": 0}

    sample = SimpleNamespace(
        base_dic_cosine_invalid=True,
        weighted_dic_base_ratio_invalid=True,
        base_dts_cosine_invalid={0.04: True},
        weighted_dts_base_ratio_invalid={0.04: True},
        dic_dts_cosine_invalid={},
        dic_dts_support_iou_invalid={},
        combined_aux_base_ratio_invalid=False,
        base_combined_aux_cosine_invalid=True,
    )
    assert trainer_module._active_diagnostic_invalid_flags(sample, resolve_profile("l12_bal")) == []
    assert trainer_module._probe_identity(99, 200, resolve_profile("v3_full")) == (
        0.5,
        0.06,
    )
    assert trainer_module._probe_identity(200, 200, resolve_profile("v3_full")) == (
        1.0,
        0.06,
    )


def test_regular_probe_uses_the_selected_profiles_baseline_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: appendix baseline diagnostics silently use l12_bal weights."""

    profile = resolve_profile("v3_no_l1")
    args = SimpleNamespace(lr=0.01, calibration_probe=False)
    runtime = _toy_runtime(args, torch.device("cpu"), profile)
    original = trainer_module.compute_baseline_branch
    observed: list[str] = []

    def recording_baseline(projections, targets, selected_profile):
        observed.append(selected_profile.name)
        return original(projections, targets, selected_profile)

    monkeypatch.setattr(trainer_module, "compute_baseline_branch", recording_baseline)
    trainer_module._gradient_probe(
        runtime,
        completed_updates=100,
        total_steps=100,
        profile=profile,
        lambda_dic=0.2,
        lambda_dts=0.3,
        calibration=False,
    )

    assert observed == ["v3_no_l1"]


@pytest.mark.parametrize(
    ("profile_name", "expects_dic", "expected_dts"),
    [
        ("dic_only", True, set()),
        ("dts_only", False, {0.04, 0.06}),
    ],
)
def test_regular_probe_zeros_the_inactive_frozen_component_scale(
    profile_name: str,
    expects_dic: bool,
    expected_dts: set[float],
) -> None:
    """Break caught: raw frozen lambda activates a profile-disabled diagnostic."""

    profile = resolve_profile(profile_name)
    args = SimpleNamespace(lr=0.01, calibration_probe=False)
    runtime = _toy_runtime(args, torch.device("cpu"), profile)

    sample = trainer_module._gradient_probe(
        runtime,
        completed_updates=150,
        total_steps=200,
        profile=profile,
        lambda_dic=0.2,
        lambda_dts=0.3,
        calibration=False,
    )

    assert (sample.dic_norm is not None) is expects_dic
    assert set(sample.dts_norms) == expected_dts
    assert sample.selected_dts_tau == (0.04 if expected_dts else None)


def test_calibration_probe_records_exact_steps_both_endpoints_and_retained_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: calibration probes read outcomes, shift steps, or lose checkpoint bytes."""

    _install_toy(monkeypatch)
    args, _ = _args(
        tmp_path,
        tmp_path / "run",
        steps=2000,
        profile="l12_bal",
        calibration=True,
    )
    run(args, command_argv=("toy-calibration",))
    run_dir = Path(args.output_dir)
    audit = json.loads((run_dir / "gradient_audit.json").read_text())

    assert [item["completed_updates"] for item in audit["samples"]] == [
        1000,
        1300,
        1600,
    ]
    assert [item["completed_updates"] for item in audit["calibration_records"]] == [
        1000,
        1300,
        1600,
    ]
    for record in audit["samples"]:
        assert set(record["dts_norms"]) == {"0.04", "0.06"}
        assert record["dic_norm"] is not None
        assert record["selected_dts_tau"] == 0.04
    artifacts = json.loads((run_dir / "artifact_manifest.json").read_text())[
        "artifacts_sha256"
    ]
    for update in (1000, 1300, 1600):
        name = f"calibration_checkpoint_step_{update:08d}.pt"
        assert name in artifacts
        record = next(
            item
            for item in audit["calibration_records"]
            if item["completed_updates"] == update
        )
        assert record["checkpoint"] == name
        assert record["checkpoint_sha256"] == artifacts[name]
        assert (run_dir / name).read_bytes() == (
            run_dir / "transactions" / f"step_{update:08d}" / "checkpoint.pt"
        ).read_bytes()


def test_resume_preflight_failure_preserves_every_existing_audit_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: corrupt resume rewrites COMMAND, manifest, history, or STATUS."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)

    def interrupt(step: int) -> None:
        if step == 1:
            raise _InjectedInterruption

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_InjectedInterruption):
        run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    before = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    args.resume = "__OUTPUT_LATEST__"
    args.lr = 0.02
    with pytest.raises(ValueError):
        run(args, command_argv=("toy", "--resume"))
    after = {path.relative_to(run_dir): path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    assert after == before


@pytest.mark.parametrize(
    "forgery",
    [
        "command_schema",
        "command_extra",
        "run_scope",
        "run_extra",
        "environment_secret",
        "environment_extra",
    ],
)
def test_resume_rejects_forged_existing_audit_exact_schema_without_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forgery: str
) -> None:
    """Break caught: compatible hashes mask forged scope/schema/environment content."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path / forgery, tmp_path / forgery / "run", steps=2)

    def interrupt(step: int) -> None:
        if step == 1:
            raise _InjectedInterruption

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_InjectedInterruption):
        run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    if forgery.startswith("command"):
        path = run_dir / "COMMAND.json"
        payload = json.loads(path.read_text())
        if forgery == "command_schema":
            payload["schema"] = "forged"
        else:
            payload["extra"] = "forged"
    elif forgery.startswith("run"):
        path = run_dir / "run_manifest.json"
        payload = json.loads(path.read_text())
        if forgery == "run_scope":
            payload["scope"] = {
                "soft_only": False,
                "final_step_export_only": False,
                "downstream_geometry": True,
            }
        else:
            payload["extra"] = "forged"
    else:
        path = run_dir / "environment.json"
        payload = json.loads(path.read_text())
        if forgery == "environment_secret":
            payload["environment"]["OPENAI_API_KEY"] = "must-not-be-accepted"
        else:
            payload["extra"] = "forged"
    atomic_json(path, payload)
    before = _tree_snapshot(run_dir)
    args.resume = "__OUTPUT_LATEST__"

    with pytest.raises(ValueError, match="COMMAND|manifest|environment"):
        run(args, command_argv=("toy", "--resume"))

    assert _tree_snapshot(run_dir) == before


def test_ahead_checkpoint_and_orphan_fail_before_quarantine_or_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: resume quarantines an orphan before rejecting ahead progress."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=1)

    def interrupt(step: int) -> None:
        if step == 1:
            raise _InjectedInterruption

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_InjectedInterruption):
        run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    config_hash = json.loads((run_dir / "COMMAND.json").read_text())["config_hash"]
    pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
    logits, optimizer = _optimizer_pair()
    step2 = commit_training_transaction(
        run_dir,
        checkpoint_payload=_checkpoint(
            logits, optimizer, 2, config_hash=config_hash
        ),
        history_segment=(_history_row(_step_record(2)),),
        previous_transaction_sha256=pointer["manifest_sha256"],
    )
    shutil.copytree(step2, run_dir / "transactions" / "step_00000003")
    before = _tree_snapshot(run_dir)
    args.resume = "__OUTPUT_LATEST__"

    with pytest.raises(ValueError, match="ahead"):
        run(args, command_argv=("toy", "--resume"))

    assert _tree_snapshot(run_dir) == before


def test_invalid_resume_probe_identities_fail_before_quarantine_or_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: resume accepts shifted probes and mutates orphan layout first."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)

    def interrupt(step: int) -> None:
        if step == 2:
            raise _InjectedInterruption

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_InjectedInterruption):
        run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
    generation = run_dir / pointer["transaction"]
    checkpoint_path = generation / "checkpoint.pt"
    checkpoint = _load_checkpoint(checkpoint_path)
    sample_logits, _ = _optimizer_pair()
    checkpoint["gradient_samples"] = [asdict(_sample(sample_logits, 1))]
    atomic_checkpoint(checkpoint_path, checkpoint)
    manifest_path = generation / "transaction_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoint_sha256"] = _sha(checkpoint_path)
    atomic_json(manifest_path, manifest)
    pointer["manifest_sha256"] = _sha(manifest_path)
    atomic_json(run_dir / "checkpoint_latest.json", pointer)
    shutil.copytree(generation, run_dir / "transactions" / "step_00000003")
    before = _tree_snapshot(run_dir)
    args.resume = "__OUTPUT_LATEST__"

    with pytest.raises(ValueError, match="probe"):
        run(args, command_argv=("toy", "--resume"))

    assert _tree_snapshot(run_dir) == before


def test_running_completed_failed_status_cas_and_completed_reuse_immutability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: terminal runs are overwritten or failed runs resume."""

    _install_toy(monkeypatch)
    completed_args, _ = _args(tmp_path / "done", tmp_path / "done" / "run", steps=2)
    run(completed_args, command_argv=("toy",))
    completed_dir = Path(completed_args.output_dir)
    before = {path.relative_to(completed_dir): path.read_bytes() for path in completed_dir.rglob("*") if path.is_file()}
    completed_args.resume = "__OUTPUT_LATEST__"
    reused = run(completed_args, command_argv=("toy", "--resume"))
    after = {path.relative_to(completed_dir): path.read_bytes() for path in completed_dir.rglob("*") if path.is_file()}
    assert after == before
    assert reused == json.loads((completed_dir / "metrics.json").read_text())

    failed_args, _ = _args(tmp_path / "failed", tmp_path / "failed" / "run", steps=2)
    original_train = trainer_module.train_one_update

    def fail_scientifically(**kwargs):
        raise SoftTrainingStepError("injected nonfinite")

    monkeypatch.setattr(trainer_module, "train_one_update", fail_scientifically)
    with pytest.raises(SoftTrainingStepError):
        run(failed_args, command_argv=("toy",))
    failed_dir = Path(failed_args.output_dir)
    failed_status = (failed_dir / "STATUS.json").read_bytes()
    assert json.loads(failed_status)["state"] == "failed"
    assert (failed_dir.parent / f".{failed_dir.name}.failure.json").is_file()
    failed_args.resume = "__OUTPUT_LATEST__"
    monkeypatch.setattr(trainer_module, "train_one_update", original_train)
    with pytest.raises(ValueError, match="failed"):
        run(failed_args, command_argv=("toy", "--resume"))
    assert (failed_dir / "STATUS.json").read_bytes() == failed_status


def test_pending_scientific_failure_terminalizes_on_resume_without_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a crash after diagnostic publication later resumes to completed."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)
    original_train = trainer_module.train_one_update
    original_cas = trainer_module._cas_terminal_status_locked

    def fail_scientifically(**unused) -> StepRecord:
        raise SoftTrainingStepError("pending scientific failure")

    def interrupt_after_diagnostic(run_dir: Path, **keywords) -> None:
        if keywords.get("state") == "failed":
            raise _InjectedInterruption("after diagnostic")
        original_cas(run_dir, **keywords)

    monkeypatch.setattr(trainer_module, "train_one_update", fail_scientifically)
    monkeypatch.setattr(
        trainer_module, "_cas_terminal_status_locked", interrupt_after_diagnostic
    )
    with pytest.raises(_InjectedInterruption):
        run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    diagnostic = run_dir.parent / f".{run_dir.name}.failure.json"
    assert diagnostic.is_file()
    assert json.loads((run_dir / "STATUS.json").read_text())["state"] == "running"
    before = _tree_snapshot(run_dir)
    diagnostic_before = diagnostic.read_bytes()
    args.resume = "__OUTPUT_LATEST__"
    monkeypatch.setattr(trainer_module, "train_one_update", original_train)
    monkeypatch.setattr(trainer_module, "_cas_terminal_status_locked", original_cas)

    with pytest.raises(ValueError, match="pending|scientific failure|failed"):
        run(args, command_argv=("toy", "--resume"))

    after = _tree_snapshot(run_dir)
    assert {key: value for key, value in after.items() if key != "STATUS.json"} == {
        key: value for key, value in before.items() if key != "STATUS.json"
    }
    terminal = json.loads((run_dir / "STATUS.json").read_text())
    assert terminal["state"] == "failed"
    assert terminal["completed_updates"] == 0
    assert diagnostic.read_bytes() == diagnostic_before
    assert not (run_dir / "artifact_manifest.json").exists()
    assert not (run_dir / "checkpoint_final.pt").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "status_path",
        "status_count",
        "diagnostic_schema",
        "diagnostic_config",
        "diagnostic_count",
        "diagnostic_extra",
    ],
)
def test_failed_status_and_diagnostic_require_exact_consistent_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Break caught: failed terminal state does not authenticate its diagnostic."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)

    def fail_scientifically(**unused) -> StepRecord:
        raise SoftTrainingStepError("authenticated scientific failure")

    monkeypatch.setattr(trainer_module, "train_one_update", fail_scientifically)
    with pytest.raises(SoftTrainingStepError):
        run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    status_path = run_dir / "STATUS.json"
    status = json.loads(status_path.read_text())
    diagnostic_path = run_dir.parent / f".{run_dir.name}.failure.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    if mutation == "status_path":
        status["diagnostic"] = "../forged.failure.json"
        atomic_json(status_path, status)
    elif mutation == "status_count":
        status["completed_updates"] = 1
        atomic_json(status_path, status)
    elif mutation == "diagnostic_schema":
        diagnostic["schema"] = "forged"
        atomic_json(diagnostic_path, diagnostic)
    elif mutation == "diagnostic_config":
        diagnostic["config_hash"] = "f" * 64
        atomic_json(diagnostic_path, diagnostic)
    elif mutation == "diagnostic_count":
        diagnostic["completed_updates"] = 1
        atomic_json(diagnostic_path, diagnostic)
    else:
        diagnostic["unexpected"] = True
        atomic_json(diagnostic_path, diagnostic)
    before = _tree_snapshot(tmp_path)
    args.resume = "__OUTPUT_LATEST__"

    with pytest.raises(ValueError, match="diagnostic|failed STATUS|failure"):
        run(args, command_argv=("toy", "--resume"))

    assert _tree_snapshot(tmp_path) == before


@pytest.mark.parametrize("target_exists", [True, False])
def test_failed_resume_rejects_sibling_diagnostic_symlink_with_zero_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_exists: bool,
) -> None:
    """Break caught: failed reuse follows or ignores a sibling diagnostic symlink."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)

    def fail_scientifically(**unused) -> StepRecord:
        raise SoftTrainingStepError("symlinked scientific failure")

    monkeypatch.setattr(trainer_module, "train_one_update", fail_scientifically)
    with pytest.raises(SoftTrainingStepError):
        run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    diagnostic_path = run_dir.parent / f".{run_dir.name}.failure.json"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_target = outside_dir / "failure.json"
    if target_exists:
        diagnostic_path.rename(outside_target)
    else:
        diagnostic_path.unlink()
    diagnostic_path.symlink_to(outside_target)
    before = _tree_snapshot(tmp_path)
    args.resume = "__OUTPUT_LATEST__"

    with pytest.raises(ValueError, match="diagnostic.*symlink"):
        run(args, command_argv=("toy", "--resume"))

    assert _tree_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "mutation",
    ["schema", "config", "completed_updates", "extra"],
)
def test_malformed_pending_failure_rejects_resume_with_zero_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Break caught: a malformed pending diagnostic is ignored and training resumes."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)

    def interrupt(step: int) -> None:
        if step == 1:
            raise _InjectedInterruption

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_InjectedInterruption):
        run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    config_hash = canonical_hash(resolved_config(args))
    diagnostic_path = run_dir.parent / f".{run_dir.name}.failure.json"
    diagnostic = {
        "schema": "pcraft_soft_loss_v3_failure_v1",
        "config_hash": config_hash,
        "completed_updates": 0,
        "error_type": "SoftTrainingStepError",
        "error": "pending",
    }
    if mutation == "schema":
        diagnostic["schema"] = "forged"
    elif mutation == "config":
        diagnostic["config_hash"] = "f" * 64
    elif mutation == "completed_updates":
        diagnostic["completed_updates"] = 3
    else:
        diagnostic["unexpected"] = True
    atomic_json(diagnostic_path, diagnostic)
    before = _tree_snapshot(tmp_path)
    args.resume = "__OUTPUT_LATEST__"
    monkeypatch.setattr(trainer_module, "_post_update_hook", lambda unused: None)

    with pytest.raises(ValueError, match="diagnostic|failure"):
        run(args, command_argv=("toy", "--resume"))

    assert _tree_snapshot(tmp_path) == before


@pytest.mark.parametrize("mutation", ["extra", "missing", "wrong_type"])
def test_running_status_requires_exact_state_schema_before_resume_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Break caught: malformed running STATUS bytes are accepted and then overwritten."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)

    def interrupt(step: int) -> None:
        if step == 1:
            raise _InjectedInterruption

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_InjectedInterruption):
        run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    status_path = run_dir / "STATUS.json"
    status = json.loads(status_path.read_text())
    if mutation == "extra":
        status["unexpected"] = "forged"
    elif mutation == "missing":
        del status["completed_updates"]
    else:
        status["completed_updates"] = "0"
    atomic_json(status_path, status)
    before = _tree_snapshot(run_dir)
    args.resume = "__OUTPUT_LATEST__"
    monkeypatch.setattr(trainer_module, "_post_update_hook", lambda unused: None)

    with pytest.raises(ValueError, match="STATUS|status"):
        run(args, command_argv=("toy", "--resume"))

    assert _tree_snapshot(run_dir) == before


def test_completed_reuse_rehashes_every_artifact_before_returning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a completed run is reused after a listed artifact changes."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)
    run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    status_before = (run_dir / "STATUS.json").read_bytes()
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_bytes(metrics_path.read_bytes() + b" ")
    tampered = metrics_path.read_bytes()
    args.resume = "__OUTPUT_LATEST__"

    with pytest.raises(ValueError, match="artifact hash"):
        run(args, command_argv=("toy", "--resume"))

    assert metrics_path.read_bytes() == tampered
    assert (run_dir / "STATUS.json").read_bytes() == status_before


def test_completed_reuse_rejects_symlinked_top_level_artifact_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: completed validation follows a top-level artifact symlink."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path / "inputs", tmp_path / "run", steps=2)
    run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    metrics_path = run_dir / "metrics.json"
    outside = tmp_path / "outside-metrics.json"
    metrics_path.rename(outside)
    metrics_path.symlink_to(outside)
    before = _tree_snapshot(run_dir)
    outside_before = outside.read_bytes()
    args.resume = "__OUTPUT_LATEST__"

    with pytest.raises(ValueError, match="artifact|symlink|unsafe"):
        run(args, command_argv=("toy", "--resume"))

    assert _tree_snapshot(run_dir) == before
    assert outside.read_bytes() == outside_before


def test_completed_reuse_rejects_rehashed_traversal_artifact_key_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a rehashed extra ../ artifact key escapes the run directory."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path / "inputs", tmp_path / "run", steps=2)
    run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside-but-rehashed")
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts_sha256"]["../outside.bin"] = _sha(outside)
    atomic_json(manifest_path, manifest)
    status_path = run_dir / "STATUS.json"
    status = json.loads(status_path.read_text())
    status["artifact_manifest_sha256"] = _sha(manifest_path)
    atomic_json(status_path, status)
    before = _tree_snapshot(run_dir)
    outside_before = outside.read_bytes()
    args.resume = "__OUTPUT_LATEST__"

    with pytest.raises(ValueError, match="artifact|canonical|unsafe|extra"):
        run(args, command_argv=("toy", "--resume"))

    assert _tree_snapshot(run_dir) == before
    assert outside.read_bytes() == outside_before


def test_linked_transaction_semantics_are_verified_before_manifest_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: terminalization blesses bytes inconsistent with transaction metadata."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)
    original_artifact_manifest = trainer_module._artifact_manifest

    def tamper_before_manifest(run_dir: Path, *positional, **keywords):
        pointer = json.loads((Path(run_dir) / "checkpoint_latest.json").read_text())
        checkpoint = Path(run_dir) / pointer["transaction"] / "checkpoint.pt"
        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered-before-manifest")
        return original_artifact_manifest(run_dir, *positional, **keywords)

    monkeypatch.setattr(trainer_module, "_artifact_manifest", tamper_before_manifest)
    with pytest.raises(ValueError, match="transaction checkpoint hash"):
        run(args, command_argv=("toy",))

    run_dir = Path(args.output_dir)
    assert not (run_dir / "artifact_manifest.json").exists()
    assert json.loads((run_dir / "STATUS.json").read_text())["state"] == "running"


def test_terminal_publication_rechecks_linked_payload_after_final_artifact_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: linked bytes change after their last hash but before completed CAS."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)
    original_sha256_file = trainer_module.sha256_file
    observations = {"final_checkpoint_reads": 0, "mutated": False}

    def mutate_after_final_hash(path: Path) -> str:
        candidate = Path(path)
        digest = original_sha256_file(candidate)
        if (
            candidate.name == "checkpoint.pt"
            and candidate.parent.name == "step_00000002"
            and candidate.parent.parent.name == "transactions"
        ):
            observations["final_checkpoint_reads"] += 1
            if observations["final_checkpoint_reads"] == 3:
                candidate.write_bytes(
                    candidate.read_bytes() + b"mutated-after-final-artifact-hash"
                )
                observations["mutated"] = True
        return digest

    monkeypatch.setattr(trainer_module, "sha256_file", mutate_after_final_hash)

    with pytest.raises(ValueError, match="transaction|artifact|checkpoint"):
        run(args, command_argv=("toy",))

    run_dir = Path(args.output_dir)
    assert observations["mutated"] is True
    assert observations["final_checkpoint_reads"] >= 3
    assert json.loads((run_dir / "STATUS.json").read_text())["state"] != "completed"


def test_pending_manifest_recovery_rechecks_linked_payload_under_terminal_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: crash recovery validates outside the lock and publishes stale bytes."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)
    run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    atomic_json(
        run_dir / "STATUS.json",
        {
            "schema": "pcraft_soft_loss_v3_status_v1",
            "state": "running",
            "config_hash": canonical_hash(resolved_config(args)),
            "completed_updates": 0,
        },
    )
    original_sha256_file = trainer_module.sha256_file
    observations = {"final_checkpoint_reads": 0, "mutated": False}

    def mutate_after_recovery_hash(path: Path) -> str:
        candidate = Path(path)
        digest = original_sha256_file(candidate)
        if (
            candidate.name == "checkpoint.pt"
            and candidate.parent.name == "step_00000002"
            and candidate.parent.parent.name == "transactions"
        ):
            observations["final_checkpoint_reads"] += 1
            if observations["final_checkpoint_reads"] == 2:
                candidate.write_bytes(
                    candidate.read_bytes() + b"pending-recovery-linked-mutation"
                )
                observations["mutated"] = True
        return digest

    monkeypatch.setattr(trainer_module, "sha256_file", mutate_after_recovery_hash)
    args.resume = "__OUTPUT_LATEST__"

    with pytest.raises(ValueError, match="transaction|artifact|checkpoint"):
        run(args, command_argv=("toy", "--resume"))

    assert observations["mutated"] is True
    assert json.loads((run_dir / "STATUS.json").read_text())["state"] == "running"


def test_completed_reuse_rejects_rehashed_artifacts_with_broken_transaction_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: artifact hashes alone bless a checkpoint/manifest disagreement."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)
    run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    pointer = json.loads((run_dir / "checkpoint_latest.json").read_text())
    transaction_checkpoint = run_dir / pointer["transaction"] / "checkpoint.pt"
    transaction_checkpoint.write_bytes(
        transaction_checkpoint.read_bytes() + b"semantically-unbound"
    )
    artifact_path = run_dir / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text())
    for relative in (pointer["transaction"] + "/checkpoint.pt", "checkpoint_final.pt"):
        artifact["artifacts_sha256"][relative] = _sha(run_dir / relative)
    atomic_json(artifact_path, artifact)
    status_path = run_dir / "STATUS.json"
    status = json.loads(status_path.read_text())
    status["artifact_manifest_sha256"] = _sha(artifact_path)
    atomic_json(status_path, status)
    before = _tree_snapshot(run_dir)
    args.resume = "__OUTPUT_LATEST__"

    with pytest.raises(ValueError, match="transaction checkpoint hash"):
        run(args, command_argv=("toy", "--resume"))

    assert _tree_snapshot(run_dir) == before


def test_running_terminalization_with_existing_manifest_never_repairs_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: resume overwrites a bad immutable artifact after manifest creation."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path, tmp_path / "run", steps=2)
    run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    atomic_json(
        run_dir / "STATUS.json",
        {
            "schema": "pcraft_soft_loss_v3_status_v1",
            "state": "running",
            "config_hash": canonical_hash(resolved_config(args)),
            "completed_updates": 0,
        },
    )
    metrics = run_dir / "metrics.json"
    metrics.write_bytes(metrics.read_bytes() + b"tampered")
    tampered = metrics.read_bytes()
    running_status = (run_dir / "STATUS.json").read_bytes()
    args.resume = "__OUTPUT_LATEST__"

    with pytest.raises(ValueError, match="artifact hash"):
        run(args, command_argv=("toy", "--resume"))

    assert metrics.read_bytes() == tampered
    assert (run_dir / "STATUS.json").read_bytes() == running_status


def test_resume_rejects_broken_reserved_artifact_symlink_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: Path.exists misses a broken manifest symlink until finalization."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path / "inputs", tmp_path / "run", steps=2)

    def interrupt(step: int) -> None:
        if step == 1:
            raise _InjectedInterruption

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_InjectedInterruption):
        run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    outside = tmp_path / "missing-outside-artifact-manifest.json"
    manifest_path = run_dir / "artifact_manifest.json"
    manifest_path.symlink_to(outside)
    before = _tree_snapshot(run_dir)
    assert not os.path.lexists(outside)
    args.resume = "__OUTPUT_LATEST__"
    monkeypatch.setattr(trainer_module, "_post_update_hook", lambda unused: None)

    with pytest.raises(ValueError, match="artifact|reserved|symlink"):
        run(args, command_argv=("toy", "--resume"))

    assert _tree_snapshot(run_dir) == before
    assert not os.path.lexists(outside)


@pytest.mark.parametrize(
    ("reserved_name", "target_exists"),
    [
        ("STATUS.json", True),
        ("STATUS.json", False),
        ("COMMAND.json", True),
        ("COMMAND.json", False),
    ],
)
def test_resume_lstat_preflights_status_and_audit_symlinks_with_zero_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reserved_name: str,
    target_exists: bool,
) -> None:
    """Break caught: valid or dangling reserved audit symlinks evade read preflight."""

    _install_toy(monkeypatch)
    args, _ = _args(tmp_path / "inputs", tmp_path / "run", steps=2)

    def interrupt(step: int) -> None:
        if step == 1:
            raise _InjectedInterruption

    monkeypatch.setattr(trainer_module, "_post_update_hook", interrupt)
    with pytest.raises(_InjectedInterruption):
        run(args, command_argv=("toy",))
    run_dir = Path(args.output_dir)
    reserved_path = run_dir / reserved_name
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_target = outside_dir / reserved_name
    if target_exists:
        reserved_path.rename(outside_target)
    else:
        reserved_path.unlink()
    reserved_path.symlink_to(outside_target)
    before = _tree_snapshot(tmp_path)
    args.resume = "__OUTPUT_LATEST__"
    monkeypatch.setattr(trainer_module, "_post_update_hook", lambda unused: None)

    with pytest.raises(ValueError, match="reserved artifact path.*symlink"):
        run(args, command_argv=("toy", "--resume"))

    assert _tree_snapshot(tmp_path) == before


def _qualified_ast_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts)).lower()


def _is_forbidden_soft_only_symbol(symbol: str) -> bool:
    forbidden_families = {
        "carve",
        "carving",
        "core_shell",
        "delivery_geometry",
        "dolfin",
        "fem",
        "fenics",
        "generate_mesh",
        "marching_cubes",
        "mesh",
        "mesh_generation",
        "meshing",
        "meshio",
        "open3d",
        "pino",
        "pymesh",
        "repair",
        "trimesh",
    }
    for component in (part.lower() for part in symbol.split(".") if part):
        delimited = f"_{component}_"
        for family in forbidden_families:
            if f"_{family}_" in delimited:
                return True
    return False


def _assert_soft_only_ast(source: str, *, label: str) -> None:
    forbidden_suffixes = (".obj", ".stl", ".ply", ".mesh")
    tree = ast.parse(source, filename=label)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not _is_forbidden_soft_only_symbol(alias.name), (
                    label,
                    alias.name,
                )
        elif isinstance(node, ast.ImportFrom):
            assert not _is_forbidden_soft_only_symbol(node.module or ""), (
                label,
                node.module,
            )
            for alias in node.names:
                assert not _is_forbidden_soft_only_symbol(alias.name), (
                    label,
                    alias.name,
                )
        elif isinstance(node, ast.Call):
            call_name = _qualified_ast_name(node.func)
            assert not _is_forbidden_soft_only_symbol(call_name), (label, call_name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.lower().strip()
            assert not value.endswith(forbidden_suffixes), (label, value)


def test_soft_only_ast_imports_calls_and_artifact_allowlist() -> None:
    """Break caught: v3 crosses into downstream geometry or writes an unregistered suffix."""

    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "shadowforge_pipeline" / "soft_voxel_v3").rglob("*.py"))
    sources.extend(sorted((root / "experiments").glob("*soft_loss_v3*.py")))
    assert sources
    for path in sources:
        _assert_soft_only_ast(path.read_text(encoding="utf-8"), label=str(path))


@pytest.mark.parametrize(
    "snippet",
    [
        "import shadowforge_pipeline.soft_voxel_v3.repair as r",
        "from shadowforge_pipeline.soft_voxel_v3 import mesh as m",
        "geometry.repair()",
        "solver.pino()",
        "fem.solve()",
        "ops.carving()",
        "from shadowforge_pipeline import delivery_geometry",
        "pipeline.delivery_geometry()",
        "from shadowforge_pipeline import meshing",
        "pipeline.mesh_generation()",
        "generate_mesh()",
        "from shadowforge_pipeline import benign_carving_adapter",
        "pipeline.benign_delivery_geometry_adapter()",
        "from shadowforge_pipeline import benign_meshing_adapter",
        "pipeline.benign_mesh_generation_adapter()",
        "open('artifact.obj', 'wb')",
        "Path('nested/output.stl')",
    ],
)
def test_soft_only_ast_guard_rejects_internal_aliases_calls_and_path_literals(
    snippet: str,
) -> None:
    """Break caught: qualified internal symbols and ordinary artifact paths evade AST."""

    with pytest.raises(AssertionError):
        _assert_soft_only_ast(snippet, label="adversarial.py")
    _assert_soft_only_ast("Path('/mesh/voxel/runs')", label="approved-path.py")
