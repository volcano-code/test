"""Authenticated, candidate-independent unit-gradient calibration for v4."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from statistics import median

import cv2
import numpy as np
import torch

from experiments.soft_loss_v4_registry import (
    CANDIDATES,
    V4SuiteIdentity,
    _candidate_registry_payload,
    _publish_bundle,
    assert_v4_suites_disjoint,
    load_v4_suite_identity,
)
from shadowforge_pipeline.soft_voxel_v3.contours import (
    ContourBands,
    make_contour_bands,
)
from shadowforge_pipeline.soft_voxel_v3.data import normalize_target
from shadowforge_pipeline.soft_voxel_v3.protocol import SadekarTFSProtocol
from shadowforge_pipeline.soft_voxel_v3.thresholding import (
    rendered_margin,
    straight_through_density,
)

from .config import (
    build_parser as build_trainer_parser,
    canonical_trainer_argv,
    resolve_run_configuration,
    revalidate_resolved_dependencies,
    validate_resolved_config,
)
from .losses import (
    asymmetric_dts_loss,
    compute_baseline_branch,
    directional_dic_loss,
    resolve_profile,
)
from .provenance import (
    canonical_hash,
    canonical_json_bytes,
    revalidate_snapshot,
    snapshot_regular_file,
)
from .trainer import (
    authenticate_transaction_checkpoint,
    validate_completed_run,
)
from .trainer_step import V4StepConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "pcraft_soft_loss_v4_calibration_plan_v1"
STAGE_SCHEMA = "pcraft_soft_loss_v4_calibration_stage_v1"
UNIT_SCHEMA = "pcraft_soft_loss_v4_unit_coefficients_v1"
SOURCE_SCHEMA = "pcraft_soft_loss_v4_source_bundle_v1"
SOURCE_ALGORITHM = "sha256_canonical_ordered_path_digest_rows_v1"
UPDATES = (1200, 1500, 1800)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_TRANSACTION = re.compile(r"transactions/step_([0-9]{8})\Z")
_BINDER_FILES = (
    "experiments/soft_loss_v4_registry.py",
    "experiments/bind_soft_loss_v4_suites.py",
    "experiments/soft_loss_v3_registry.py",
)
_CALIBRATOR_FILES = (
    "shadowforge_pipeline/soft_voxel_v4/calibration.py",
    "experiments/calibrate_soft_loss_v4.py",
    "experiments/soft_loss_v4_registry.py",
)
_ESTIMATOR = {
    "name": "median_raw_gradient_norm_ratio_v1",
    "completed_updates": list(UPDATES),
    "dic_endpoint": 0.05,
    "dts_endpoints": [0.04, 0.06],
    "epsilon": 1e-12,
    "candidate_independent": True,
}


@dataclass(frozen=True)
class UnitGradientMeasurement:
    completed_updates: int
    transaction: str
    transaction_manifest_sha256: str
    checkpoint_path: str
    checkpoint_sha256: str
    g_base: float
    g_dic: float
    g_dts_004: float
    g_dts_006: float


def _decode(raw: bytes, *, label: str) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant in {label}: {value}")

    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}") from error
    if type(payload) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _mapping(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} has the wrong exact schema")
    return value


def _exact_value(value: object, expected: object, *, label: str) -> None:
    if type(value) is not type(expected):
        raise ValueError(f"{label} has the wrong exact scalar/container type")
    if type(expected) is dict:
        if set(value) != set(expected):
            raise ValueError(f"{label} has the wrong exact keys")
        for key, expected_item in expected.items():
            _exact_value(value[key], expected_item, label=f"{label}.{key}")
    elif type(expected) is list:
        if len(value) != len(expected):
            raise ValueError(f"{label} has the wrong exact list length")
        for index, (item, expected_item) in enumerate(zip(value, expected)):
            _exact_value(item, expected_item, label=f"{label}[{index}]")
    elif value != expected:
        raise ValueError(f"{label} differs from the frozen exact value")


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact_str(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty exact string")
    return value


def _canonical_path(path: Path, *, label: str, exists: bool = True) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate != Path(os.path.abspath(candidate)):
        raise ValueError(f"{label} must be an absolute canonical path")
    if ".." in candidate.parts:
        raise ValueError(f"{label} contains traversal")
    if exists:
        try:
            if candidate.resolve(strict=True) != candidate:
                raise ValueError(f"{label} is not canonical")
        except OSError as error:
            raise ValueError(f"{label} is missing") from error
    return candidate


def _semantic(payload: Mapping[str, object], *, label: str) -> str:
    recorded = _digest(payload.get("semantic_sha256"), label=f"{label} semantic_sha256")
    scientific = dict(payload)
    scientific.pop("semantic_sha256", None)
    if canonical_hash(scientific) != recorded:
        raise ValueError(f"{label} semantic hash mismatch")
    return recorded


def _json_value_bytes(value: object) -> bytes:
    """Canonical built-in JSON for the documented top-level source-row list."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _source_bundle(names: Sequence[str]) -> dict[str, object]:
    if type(names) not in (tuple, list) or any(type(name) is not str for name in names):
        raise ValueError("source whitelist must be an exact ordered string sequence")
    rows: list[dict[str, str]] = []
    files: dict[str, str] = {}
    snapshots = []
    for relative in names:
        path = PROJECT_ROOT / relative
        snapshot = snapshot_regular_file(path, root=PROJECT_ROOT)
        files[relative] = snapshot.sha256
        rows.append({"path": relative, "sha256": snapshot.sha256})
        snapshots.append(snapshot)
    payload: dict[str, object] = {
        "schema": SOURCE_SCHEMA,
        "algorithm": SOURCE_ALGORITHM,
        "files_sha256": files,
        "bundle_sha256": hashlib.sha256(_json_value_bytes(rows)).hexdigest(),
    }
    for snapshot in snapshots:
        revalidate_snapshot(snapshot, root=PROJECT_ROOT)
    return payload


def _validate_source_bundle(value: object, names: Sequence[str], *, label: str) -> None:
    bundle = _mapping(
        value,
        {"schema", "algorithm", "files_sha256", "bundle_sha256"},
        label=label,
    )
    if bundle["schema"] != SOURCE_SCHEMA or bundle["algorithm"] != SOURCE_ALGORITHM:
        raise ValueError(f"{label} schema/algorithm drifted")
    files = bundle["files_sha256"]
    if type(files) is not dict or set(files) != set(names):
        raise ValueError(f"{label} source whitelist drifted")
    rows: list[dict[str, str]] = []
    snapshots = []
    for relative in names:
        digest = _digest(files.get(relative), label=f"{label} source digest")
        snapshot = snapshot_regular_file(PROJECT_ROOT / relative, root=PROJECT_ROOT)
        if snapshot.sha256 != digest:
            raise ValueError(f"{label} source bytes changed: {relative}")
        snapshots.append(snapshot)
        rows.append({"path": relative, "sha256": digest})
    expected = hashlib.sha256(_json_value_bytes(rows)).hexdigest()
    if _digest(bundle["bundle_sha256"], label=f"{label} bundle sha256") != expected:
        raise ValueError(f"{label} ordered bundle hash mismatch")
    for snapshot in snapshots:
        revalidate_snapshot(snapshot, root=PROJECT_ROOT)


def _suite_record(suite: V4SuiteIdentity) -> dict[str, object]:
    return {
        "path": suite.path,
        "sha256": suite.sha256,
        "semantic_sha256": suite.semantic_sha256,
        "role": suite.role,
    }


def _load_suites(
    development_suite: Path, formal_suite: Path
) -> tuple[V4SuiteIdentity, V4SuiteIdentity]:
    development = load_v4_suite_identity(Path(development_suite), "development")
    formal = load_v4_suite_identity(Path(formal_suite), "formal")
    if Path(development.path).parent.parent != Path(formal.path).parent.parent:
        raise ValueError("development and formal suites must share one v4 root")
    assert_v4_suites_disjoint(development, formal)
    return development, formal


def _identity_equal(actual: V4SuiteIdentity, expected: object, *, label: str) -> None:
    identity = _mapping(
        expected, {"path", "sha256", "semantic_sha256", "role"}, label=label
    )
    if _suite_record(actual) != identity:
        raise ValueError(f"{label} differs from current authenticated suite")


def _validate_development_suite_identity(
    path: Path, identity: object
) -> V4SuiteIdentity:
    current = load_v4_suite_identity(Path(path), "development")
    _identity_equal(current, identity, label="development suite identity")
    return current


def _measurement_rows(
    measurements: Sequence[UnitGradientMeasurement],
) -> tuple[UnitGradientMeasurement, UnitGradientMeasurement, UnitGradientMeasurement]:
    if type(measurements) not in (tuple, list) or len(measurements) != 3:
        raise ValueError("measurements must contain exactly three ordered rows")
    rows = tuple(measurements)
    if any(type(row) is not UnitGradientMeasurement for row in rows):
        raise TypeError("every measurement must be an exact UnitGradientMeasurement")
    if tuple(row.completed_updates for row in rows) != UPDATES:
        raise ValueError("measurement update order must be exactly 1200/1500/1800")
    transactions: set[str] = set()
    transaction_digests: set[str] = set()
    checkpoints: set[str] = set()
    checkpoint_digests: set[str] = set()
    for row in rows:
        if type(row.completed_updates) is not int:
            raise TypeError("completed_updates must be an exact integer")
        transaction = _exact_str(row.transaction, label="measurement transaction")
        checkpoint = _exact_str(row.checkpoint_path, label="measurement checkpoint path")
        transaction_digest = _digest(
            row.transaction_manifest_sha256,
            label="measurement transaction manifest sha256",
        )
        checkpoint_digest = _digest(
            row.checkpoint_sha256, label="measurement checkpoint sha256"
        )
        if transaction in transactions or transaction_digest in transaction_digests:
            raise ValueError("duplicate transaction identity in measurements")
        if checkpoint in checkpoints or checkpoint_digest in checkpoint_digests:
            raise ValueError("duplicate checkpoint identity in measurements")
        transactions.add(transaction)
        transaction_digests.add(transaction_digest)
        checkpoints.add(checkpoint)
        checkpoint_digests.add(checkpoint_digest)
        for name in ("g_base", "g_dic", "g_dts_004", "g_dts_006"):
            value = getattr(row, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be an exact finite positive Python float")
    return rows[0], rows[1], rows[2]


def derive_unit_coefficients(
    measurements: Sequence[UnitGradientMeasurement],
) -> dict[str, float]:
    """Apply the one frozen median-ratio estimator to authenticated raw norms."""

    rows = _measurement_rows(measurements)
    base = float(median([row.g_base for row in rows]))
    dic_denominator = float(median([row.g_dic for row in rows])) + 1e-12
    dts_values = [
        value
        for row in rows
        for value in (row.g_dts_004, row.g_dts_006)
    ]
    dts_denominator = float(median(dts_values)) + 1e-12
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (base, dic_denominator, dts_denominator)
    ):
        raise ValueError("coefficient medians must be finite and positive")
    dic = float(base / dic_denominator)
    dts = float(base / dts_denominator)
    if not all(math.isfinite(value) and value > 0.0 for value in (dic, dts)):
        raise ValueError("coefficient result overflowed or is not finite and positive")
    return {"dic": dic, "dts": dts}


def _contour_bands(target_binary: torch.Tensor) -> ContourBands:
    return make_contour_bands(target_binary, radius_px=2.0)


def _threshold_score(protocol: object, density: torch.Tensor, endpoint: float) -> torch.Tensor:
    occupancy = straight_through_density(density, endpoint)
    projection = protocol.render_all(occupancy)
    if not isinstance(projection, torch.Tensor) or projection.dtype != density.dtype:
        raise TypeError("renderer output must preserve the calibrated floating dtype")
    if projection.device != density.device:
        raise ValueError("renderer output and logits must share a device")
    _, score = rendered_margin(projection)
    return score


def _raw_gradient_norms(
    *, logits: torch.Tensor, protocol: object, targets: torch.Tensor
) -> dict[str, float]:
    """Measure four unweighted Task-2 branch gradients without mutating logits."""

    if (
        not isinstance(logits, torch.Tensor)
        or not logits.is_floating_point()
        or not logits.is_leaf
        or not logits.requires_grad
        or not torch.isfinite(logits).all().item()
    ):
        raise ValueError("logits must be one finite floating gradient-enabled leaf")
    if logits.dtype not in (torch.float32, torch.float64):
        raise TypeError("calibration logits must use float32 or float64")
    if (
        not isinstance(targets, torch.Tensor)
        or not targets.is_floating_point()
        or targets.ndim != 3
        or any(size == 0 for size in targets.shape)
        or targets.dtype != logits.dtype
        or targets.device != logits.device
        or not torch.isfinite(targets).all().item()
    ):
        raise ValueError("targets must be finite [views,H,W] and match logits dtype/device")
    if not callable(getattr(protocol, "render_all", None)):
        raise ValueError("protocol must expose render_all")
    before = logits.detach().clone()
    target_binary = targets >= 0.5
    raw_bands = _contour_bands(target_binary)
    if isinstance(raw_bands, ContourBands):
        bands = raw_bands
    else:
        try:
            bands = ContourBands(inner=raw_bands.inner, outer=raw_bands.outer)
        except (AttributeError, TypeError) as error:
            raise ValueError("contour-band construction returned a malformed value") from error

    def norm(loss: torch.Tensor, *, label: str) -> float:
        if (
            not isinstance(loss, torch.Tensor)
            or loss.ndim != 0
            or not loss.is_floating_point()
            or loss.dtype != logits.dtype
            or loss.device != logits.device
            or not torch.isfinite(loss).item()
        ):
            raise RuntimeError(f"{label} loss must be one finite dtype/device-matched scalar")
        try:
            (gradient,) = torch.autograd.grad(
                loss, logits, retain_graph=False, create_graph=False
            )
        except RuntimeError as error:
            raise RuntimeError(f"{label} gradient is disconnected") from error
        if (
            not isinstance(gradient, torch.Tensor)
            or gradient.shape != logits.shape
            or gradient.dtype != logits.dtype
            or gradient.device != logits.device
            or not torch.isfinite(gradient).all().item()
        ):
            raise RuntimeError(f"{label} gradient is malformed or non-finite")
        value = float(torch.linalg.vector_norm(gradient.to(dtype=torch.float64)).item())
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"{label} gradient norm must be finite and positive")
        return value

    base_loss = compute_baseline_branch(
        protocol.render_all(torch.sigmoid(logits)), targets, resolve_profile("l12_bal")
    )
    g_base = norm(base_loss, label="baseline")
    dic_loss = directional_dic_loss(
        _threshold_score(protocol, torch.sigmoid(logits), 0.05), bands
    ).mean()
    g_dic = norm(dic_loss, label="DIC")
    dts_004_loss = asymmetric_dts_loss(
        _threshold_score(protocol, torch.sigmoid(logits), 0.04),
        target_binary,
        0.04,
    ).mean()
    g_dts_004 = norm(dts_004_loss, label="DTS-0.04")
    dts_006_loss = asymmetric_dts_loss(
        _threshold_score(protocol, torch.sigmoid(logits), 0.06),
        target_binary,
        0.06,
    ).mean()
    g_dts_006 = norm(dts_006_loss, label="DTS-0.06")
    if not torch.equal(logits.detach(), before) or logits.grad is not None:
        raise RuntimeError("raw-gradient measurement mutated checkpoint logits")
    return {
        "g_base": g_base,
        "g_dic": g_dic,
        "g_dts_004": g_dts_004,
        "g_dts_006": g_dts_006,
    }


def _decode_shadow(payload: bytes, *, path: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"authenticated shadow is not a grayscale image: {path}")
    return (
        normalize_target(image, size, False),
        normalize_target(cv2.flip(image, 1), size, False),
    )


def _checkpoint_mapping(snapshot: object, expected_config: Mapping[str, object]) -> tuple[dict[str, object], torch.Tensor]:
    try:
        payload = torch.load(
            io.BytesIO(snapshot.checkpoint.payload),
            map_location="cpu",
            weights_only=False,
        )
    except Exception as error:
        raise ValueError("authenticated checkpoint payload cannot be decoded") from error
    if type(payload) is not dict:
        raise ValueError("checkpoint payload must be an exact mapping")
    required = {
        "schema", "resolved_config", "config_hash", "completed_updates", "logits",
        "optimizer", "last_result", "rng_state", "step_times", "peak_cuda_bytes",
        "slot_audits",
    }
    if set(payload) != required or payload["schema"] != "pcraft_soft_loss_v4_checkpoint_v1":
        raise ValueError("checkpoint has the wrong exact schema")
    config = validate_resolved_config(payload["resolved_config"])
    expected = validate_resolved_config(dict(expected_config))
    if canonical_json_bytes(config) != canonical_json_bytes(expected):
        raise ValueError("checkpoint resolved config differs from calibration plan")
    if payload["config_hash"] != canonical_hash(config):
        raise ValueError("checkpoint config hash mismatch")
    if payload["completed_updates"] != snapshot.identity.completed_updates:
        raise ValueError("checkpoint completed update identity mismatch")
    logits = payload["logits"]
    resolution = config["resolution"]
    if (
        not isinstance(logits, torch.Tensor)
        or not logits.is_floating_point()
        or logits.shape != (1, 1, resolution, resolution, resolution)
        or not torch.isfinite(logits).all().item()
    ):
        raise ValueError("checkpoint logits are malformed")
    return config, logits.detach().clone().requires_grad_()


def _measure_authenticated_checkpoint(
    snapshot: object,
    config: Mapping[str, object],
    development_suite: V4SuiteIdentity,
) -> dict[str, float]:
    checked, logits = _checkpoint_mapping(snapshot, config)
    if development_suite.role != "development" or not development_suite.tasks:
        raise ValueError("calibration requires the frozen development suite")
    task = development_suite.tasks[0]
    if task.case_id != "arcane_casters" or checked["run_identity"]["task_id"] != task.case_id:
        raise ValueError("calibration checkpoint is not the first development task")
    declared = checked["shadow_inputs_xyz"]
    if [item.resolved_path for item in task.inputs] != [item["path"] for item in declared]:
        raise ValueError("calibration checkpoint shadow paths differ from development task")
    if [item.sha256 for item in task.inputs] != [item["sha256"] for item in declared]:
        raise ValueError("calibration checkpoint shadow hashes differ from development task")
    shadow_snapshots = []
    decoded = []
    for item in declared:
        source = snapshot_regular_file(Path(item["path"]))
        if source.sha256 != item["sha256"]:
            raise ValueError("calibration shadow bytes changed")
        shadow_snapshots.append(source)
        decoded.append(
            _decode_shadow(
                source.payload, path=source.path, size=checked["protocol"]["image_size"]
            )
        )
    ordered = [pair[0] for pair in decoded] + [pair[1] for pair in decoded]
    targets = torch.from_numpy(np.stack(ordered).astype(np.float64) / 255.0)
    if logits.dtype == torch.float32:
        targets = targets.to(dtype=torch.float32)
    protocol = SadekarTFSProtocol(
        device=torch.device("cpu"),
        image_size=checked["protocol"]["image_size"],
        zdist=checked["protocol"]["zdist"],
        volume_extent_world=checked["protocol"]["volume_extent"],
        n_pts_per_ray=checked["protocol"]["samples_per_ray"],
    )
    result = _raw_gradient_norms(logits=logits, protocol=protocol, targets=targets)
    for source in shadow_snapshots:
        revalidate_snapshot(source)
    revalidate_snapshot(snapshot.checkpoint, root=Path(checked["run_identity"]["run_dir"]))
    revalidate_resolved_dependencies(checked)
    return result


def _coherent_checkpoint(snapshot: object, run: Path, update: int) -> None:
    identity = snapshot.identity
    checkpoint = snapshot.checkpoint
    if type(identity.completed_updates) is not int or identity.completed_updates != update:
        raise ValueError("authenticated checkpoint update identity mismatch")
    if identity.transaction != f"transactions/step_{update:08d}":
        raise ValueError("authenticated checkpoint transaction identity mismatch")
    _digest(identity.transaction_manifest_sha256, label="transaction manifest sha256")
    relative = PurePosixPath(_exact_str(identity.checkpoint_path, label="checkpoint path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("authenticated checkpoint path escapes its run")
    expected_path = run.joinpath(*relative.parts)
    if checkpoint.path != str(expected_path):
        raise ValueError("authenticated checkpoint path identity mismatch")
    if checkpoint.sha256 != identity.checkpoint_sha256:
        raise ValueError("authenticated checkpoint hash identity mismatch")
    _digest(identity.checkpoint_sha256, label="checkpoint sha256")


def collect_unit_gradient_measurements(
    calibration_run: Path,
    calibration_plan: Path,
    development_suite: Path,
) -> tuple[UnitGradientMeasurement, UnitGradientMeasurement, UnitGradientMeasurement]:
    """Authenticate and measure exactly one checkpoint payload at a time."""

    run = _canonical_path(Path(calibration_run), label="calibration run")
    plan = validate_calibration_plan(Path(calibration_plan))
    config = plan["resolved_config"]
    run_identity = config.get("run_identity")
    if type(run_identity) is dict and str(run) != run_identity.get("run_dir"):
        raise ValueError("calibration run differs from preregistered plan")
    suite_path = _canonical_path(Path(development_suite), label="development suite")
    development = _validate_development_suite_identity(
        suite_path, plan["development_suite_identity"]
    )
    measured: list[UnitGradientMeasurement] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for update in UPDATES:
        authenticated = authenticate_transaction_checkpoint(
            run, completed_updates=update, expected_config=config
        )
        _coherent_checkpoint(authenticated, run, update)
        if authenticated.identity.checkpoint_path in seen_paths:
            raise ValueError("duplicate authenticated checkpoint path")
        if authenticated.identity.checkpoint_sha256 in seen_hashes:
            raise ValueError("duplicate authenticated checkpoint hash")
        seen_paths.add(authenticated.identity.checkpoint_path)
        seen_hashes.add(authenticated.identity.checkpoint_sha256)
        norms = _measure_authenticated_checkpoint(authenticated, config, development)
        _coherent_checkpoint(authenticated, run, update)
        validate_calibration_plan(Path(calibration_plan))
        _validate_development_suite_identity(
            suite_path, plan["development_suite_identity"]
        )
        measured.append(
            UnitGradientMeasurement(
                completed_updates=update,
                transaction=authenticated.identity.transaction,
                transaction_manifest_sha256=authenticated.identity.transaction_manifest_sha256,
                checkpoint_path=authenticated.identity.checkpoint_path,
                checkpoint_sha256=authenticated.identity.checkpoint_sha256,
                g_base=norms["g_base"],
                g_dic=norms["g_dic"],
                g_dts_004=norms["g_dts_004"],
                g_dts_006=norms["g_dts_006"],
            )
        )
        del authenticated
    rows = _measurement_rows(measured)
    return rows


def build_calibration_plan(
    *,
    output_root: Path,
    trainer_args: argparse.Namespace,
    development_suite: Path,
    formal_suite: Path,
) -> Mapping[str, object]:
    """Construct the immutable authority before the source run exists."""

    development, formal = _load_suites(development_suite, formal_suite)
    resolved = resolve_run_configuration(trainer_args)
    config = copy.deepcopy(resolved.config)
    root = _canonical_path(Path(output_root), label="output root")
    if Path(development.path).parent.parent != root or Path(formal.path).parent.parent != root:
        raise ValueError("calibration suites must be canonical children of output root")
    identity = config.get("run_identity", {})
    if (
        config.get("run_kind") != "calibration_source"
        or identity.get("task_id") != "arcane_casters"
        or identity.get("replicate_role") != "calibration"
        or identity.get("replicate_index") != 0
        or config.get("steps") != 2000
        or config.get("resolution") != 128
        or config.get("export_resolution") != 128
        or config.get("seed") != 17
        or config.get("profile", {}).get("name") != "l12_bal"
    ):
        raise ValueError("resolved calibration-source matrix drifted")
    if config.get("output_root_identity", {}).get("path") != str(root):
        raise ValueError("resolved calibration root differs from requested root")
    _identity_equal(development, config.get("suite_manifest"), label="resolved development suite")
    if development.tasks[0].case_id != "arcane_casters":
        raise ValueError("first development task is not the frozen calibration task")
    specification = _mapping(
        config.get("specification"), {"path", "sha256"}, label="resolved specification"
    )
    payload: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "status": "REGISTERED",
        "output_root_identity": copy.deepcopy(config["output_root_identity"]),
        "resolved_config": config,
        "config_hash": resolved.config_hash,
        "fresh_argv": list(canonical_trainer_argv(trainer_args, resume=False)),
        "resume_argv": list(canonical_trainer_argv(trainer_args, resume=True)),
        "calibration_task_identity": {"task_id": "arcane_casters"},
        "development_suite_identity": _suite_record(development),
        "formal_suite_identity": _suite_record(formal),
        "specification_identity": copy.deepcopy(specification),
        "estimator": copy.deepcopy(_ESTIMATOR),
        "binder_source_bundle": _source_bundle(_BINDER_FILES),
        "calibrator_source_bundle": _source_bundle(_CALIBRATOR_FILES),
        "trainer_source_bundle": copy.deepcopy(config["implementation"]),
        "publication": {"policy": "create_once_before_calibration_process"},
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    return payload


_PLAN_KEYS = {
    "schema", "status", "output_root_identity", "resolved_config", "config_hash",
    "fresh_argv", "resume_argv", "calibration_task_identity",
    "development_suite_identity", "formal_suite_identity", "specification_identity",
    "estimator", "binder_source_bundle", "calibrator_source_bundle",
    "trainer_source_bundle", "publication", "semantic_sha256",
}


def _validate_argv(fresh: object, resume: object, config: Mapping[str, object]) -> None:
    if (
        type(fresh) is not list
        or type(resume) is not list
        or any(type(item) is not str or not item for item in fresh + resume)
    ):
        raise ValueError("plan argv values must be exact non-empty string arrays")
    if len(fresh) < 4 or fresh[-2:] != ["--export-final-step", "--soft-only-eval"]:
        raise ValueError("fresh argv acknowledgement tail drifted")
    if resume != fresh[:-2] + ["--resume"] + fresh[-2:]:
        raise ValueError("resume argv is not the exact canonical resume command")
    forbidden = {"--unit-coefficients", "--candidate-registry", "--candidate-id"}
    if forbidden.intersection(fresh) or forbidden.intersection(resume):
        raise ValueError("calibration argv contains downstream candidate options")
    try:
        argv_interpreter = Path(fresh[0]).resolve(strict=True)
        config_interpreter = Path(config["interpreter"]["path"]).resolve(strict=True)
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError("plan argv interpreter identity is invalid") from error
    if argv_interpreter != config_interpreter:
        raise ValueError("plan argv interpreter differs from resolved config")
    if fresh[1:3] != ["-m", "shadowforge_pipeline.soft_voxel_v4.trainer"]:
        raise ValueError("plan argv trainer module drifted")
    fresh_args = build_trainer_parser().parse_args(fresh[3:])
    resume_args = build_trainer_parser().parse_args(resume[3:])
    if fresh_args.resume is not False or resume_args.resume is not True:
        raise ValueError("plan argv resume selectors drifted")
    if {
        key: value for key, value in vars(fresh_args).items() if key != "resume"
    } != {
        key: value for key, value in vars(resume_args).items() if key != "resume"
    }:
        raise ValueError("fresh/resume argv scientific arguments differ")
    if fresh != list(canonical_trainer_argv(fresh_args, resume=False)):
        raise ValueError("fresh argv is not Task-3 canonical bytes")
    if resume != list(canonical_trainer_argv(resume_args, resume=True)):
        raise ValueError("resume argv is not Task-3 canonical bytes")
    projection = {
        "run_kind": fresh_args.run_kind,
        "device": fresh_args.device,
        "output_root": fresh_args.output_root,
        "run_id": fresh_args.run_id,
        "stage": fresh_args.stage,
        "task_id": fresh_args.task_id,
        "replicate_role": fresh_args.replicate_role,
        "replicate_index": fresh_args.replicate_index,
        "steps": fresh_args.steps,
        "learning_rate": fresh_args.learning_rate,
        "shadow_files": fresh_args.shadow_files,
        "resolution": fresh_args.resolution,
        "export_resolution": fresh_args.export_resolution,
        "profile": fresh_args.profile,
        "suite_manifest": fresh_args.suite_manifest,
        "specification": fresh_args.specification,
    }
    expected_projection = {
        "run_kind": config["run_kind"],
        "device": config["device"],
        "output_root": config["output_root_identity"]["path"],
        "run_id": config["run_identity"]["run_id"],
        "stage": config["run_identity"]["stage"],
        "task_id": config["run_identity"]["task_id"],
        "replicate_role": config["run_identity"]["replicate_role"],
        "replicate_index": config["run_identity"]["replicate_index"],
        "steps": config["steps"],
        "learning_rate": config["learning_rate"],
        "shadow_files": [item["path"] for item in config["shadow_inputs_xyz"]],
        "resolution": config["resolution"],
        "export_resolution": config["export_resolution"],
        "profile": config["profile"]["name"],
        "suite_manifest": config["suite_manifest"]["path"],
        "specification": config["specification"]["path"],
    }
    if projection != expected_projection:
        raise ValueError("plan argv differs from the complete resolved config projection")


def _validate_plan_payload(payload: dict[str, object], path: Path) -> dict[str, object]:
    _mapping(payload, _PLAN_KEYS, label="calibration plan")
    if payload["schema"] != PLAN_SCHEMA or payload["status"] != "REGISTERED":
        raise ValueError("calibration plan schema/status mismatch")
    _semantic(payload, label="calibration plan")
    config = validate_resolved_config(payload["resolved_config"])
    if payload["config_hash"] != canonical_hash(config):
        raise ValueError("calibration plan config hash mismatch")
    root_identity = _mapping(
        payload["output_root_identity"],
        {"path", "marker_path", "marker_sha256"},
        label="plan output root identity",
    )
    root = _canonical_path(Path(root_identity["path"]), label="plan output root")
    if path != root / "manifests" / "calibration_plan.json":
        raise ValueError("calibration plan is not at its canonical root path")
    if config["output_root_identity"] != root_identity:
        raise ValueError("plan root identity differs from resolved config")
    run = config["run_identity"]
    if (
        config["run_kind"] != "calibration_source"
        or run["task_id"] != "arcane_casters"
        or run["replicate_role"] != "calibration"
        or run["replicate_index"] != 0
        or config["steps"] != 2000
        or config["resolution"] != 128
        or config["export_resolution"] != 128
        or config["seed"] != 17
        or config["profile"]["name"] != "l12_bal"
        or config["suite_manifest"]["role"] != "development"
    ):
        raise ValueError("plan calibration-source matrix drifted")
    if payload["calibration_task_identity"] != {"task_id": "arcane_casters"}:
        raise ValueError("plan calibration task identity drifted")
    development, formal = _load_suites(
        Path(payload["development_suite_identity"]["path"]),
        Path(payload["formal_suite_identity"]["path"]),
    )
    if Path(development.path).parent.parent != root or Path(formal.path).parent.parent != root:
        raise ValueError("plan suites must be canonical children of one output root")
    _identity_equal(development, payload["development_suite_identity"], label="plan development suite")
    _identity_equal(formal, payload["formal_suite_identity"], label="plan formal suite")
    if config["suite_manifest"] != payload["development_suite_identity"]:
        raise ValueError("plan config does not bind the development suite")
    specification = _mapping(
        payload["specification_identity"], {"path", "sha256"}, label="plan specification"
    )
    if specification != config["specification"]:
        raise ValueError("plan specification differs from resolved config")
    specification_snapshot = snapshot_regular_file(Path(specification["path"]))
    if specification_snapshot.sha256 != specification["sha256"]:
        raise ValueError("plan specification bytes changed")
    _exact_value(payload["estimator"], _ESTIMATOR, label="plan estimator")
    _validate_source_bundle(payload["binder_source_bundle"], _BINDER_FILES, label="binder source bundle")
    _validate_source_bundle(payload["calibrator_source_bundle"], _CALIBRATOR_FILES, label="calibrator source bundle")
    if payload["trainer_source_bundle"] != config["implementation"]:
        raise ValueError("plan trainer source bundle differs from resolved config")
    _exact_value(
        payload["publication"],
        {"policy": "create_once_before_calibration_process"},
        label="plan publication policy",
    )
    _validate_argv(payload["fresh_argv"], payload["resume_argv"], config)
    revalidate_resolved_dependencies(config)
    revalidate_snapshot(specification_snapshot)
    return payload


def validate_calibration_plan(path: Path) -> Mapping[str, object]:
    absolute = _canonical_path(Path(path), label="calibration plan")
    snapshot = snapshot_regular_file(absolute, root=absolute.parent.parent)
    payload = _decode(snapshot.payload, label="calibration plan")
    if snapshot.payload != canonical_json_bytes(payload):
        raise ValueError("calibration plan is not canonical JSON")
    checked = _validate_plan_payload(payload, absolute)
    revalidate_snapshot(snapshot, root=absolute.parent.parent)
    return copy.deepcopy(checked)


def write_calibration_plan(
    output_root: Path,
    *,
    trainer_args: argparse.Namespace,
    development_suite: Path,
    formal_suite: Path,
) -> Path:
    payload = build_calibration_plan(
        output_root=output_root,
        trainer_args=trainer_args,
        development_suite=development_suite,
        formal_suite=formal_suite,
    )
    root = _canonical_path(Path(output_root), label="output root")
    run = Path(payload["resolved_config"]["run_identity"]["run_dir"])
    if os.path.lexists(run):
        raise FileExistsError("calibration plan must be preregistered before the run exists")
    target = root / "manifests" / "calibration_plan.json"
    encoded = canonical_json_bytes(payload)

    def precommit() -> None:
        if os.path.lexists(run):
            raise FileExistsError("calibration run exists before plan publication")
        _validate_plan_payload(copy.deepcopy(dict(payload)), target)

    _publish_bundle({target: encoded}, output_root=root, precommit=precommit)
    validate_calibration_plan(target)
    return target


def build_calibration_stage(
    *,
    calibration_plan: Path,
    calibration_run: Path,
    measurements: Sequence[UnitGradientMeasurement],
) -> Mapping[str, object]:
    plan_path = _canonical_path(Path(calibration_plan), label="calibration plan")
    plan = validate_calibration_plan(plan_path)
    rows = _measurement_rows(measurements)
    derive_unit_coefficients(rows)
    run_path = _canonical_path(Path(calibration_run), label="calibration run")
    if str(run_path) != plan["resolved_config"]["run_identity"]["run_dir"]:
        raise ValueError("calibration stage run differs from plan")
    completed = validate_completed_run(
        run_path, expected_config=plan["resolved_config"]
    )
    plan_snapshot = snapshot_regular_file(plan_path, root=plan_path.parent.parent)
    run_identity = completed.config["run_identity"]
    payload: dict[str, object] = {
        "schema": STAGE_SCHEMA,
        "status": "APPROVED_FROZEN",
        "output_root_identity": copy.deepcopy(plan["output_root_identity"]),
        "calibration_plan": {
            "path": plan_snapshot.path,
            "sha256": plan_snapshot.sha256,
            "semantic_sha256": plan["semantic_sha256"],
        },
        "calibration_run": {
            "path": completed.run_dir,
            "run_id": run_identity["run_id"],
            "task_id": run_identity["task_id"],
            "config_hash": completed.config_hash,
            "status_sha256": completed.status_sha256,
            "artifact_manifest_sha256": completed.artifact_manifest_sha256,
        },
        "calibration_task_identity": copy.deepcopy(plan["calibration_task_identity"]),
        "development_suite_identity": copy.deepcopy(plan["development_suite_identity"]),
        "formal_suite_identity": copy.deepcopy(plan["formal_suite_identity"]),
        "checkpoints": [
            {
                "completed_updates": row.completed_updates,
                "transaction": row.transaction,
                "transaction_manifest_sha256": row.transaction_manifest_sha256,
                "path": row.checkpoint_path,
                "sha256": row.checkpoint_sha256,
            }
            for row in rows
        ],
        "measurements": [asdict(row) for row in rows],
        "estimator": copy.deepcopy(plan["estimator"]),
        "trainer_source_bundle": copy.deepcopy(plan["trainer_source_bundle"]),
        "binder_source_bundle": copy.deepcopy(plan["binder_source_bundle"]),
        "calibrator_source_bundle": copy.deepcopy(plan["calibrator_source_bundle"]),
        "publication": {"policy": "create_once_after_completed_calibration_source"},
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    revalidate_snapshot(plan_snapshot, root=plan_path.parent.parent)
    return payload


_STAGE_KEYS = {
    "schema", "status", "output_root_identity", "calibration_plan",
    "calibration_run", "calibration_task_identity", "development_suite_identity",
    "formal_suite_identity", "checkpoints", "measurements", "estimator",
    "trainer_source_bundle", "binder_source_bundle", "calibrator_source_bundle",
    "publication", "semantic_sha256",
}


def _rows_from_payload(value: object) -> tuple[UnitGradientMeasurement, UnitGradientMeasurement, UnitGradientMeasurement]:
    if type(value) is not list:
        raise ValueError("stage measurements must be an exact list")
    keys = set(UnitGradientMeasurement.__dataclass_fields__)
    rows = []
    for raw in value:
        item = _mapping(raw, keys, label="stage measurement")
        rows.append(UnitGradientMeasurement(**item))
    return _measurement_rows(rows)


def _validate_stage_payload(payload: dict[str, object], path: Path) -> dict[str, object]:
    _mapping(payload, _STAGE_KEYS, label="calibration stage")
    if payload["schema"] != STAGE_SCHEMA or payload["status"] != "APPROVED_FROZEN":
        raise ValueError("calibration stage schema/status mismatch")
    _semantic(payload, label="calibration stage")
    root_identity = _mapping(
        payload["output_root_identity"],
        {"path", "marker_path", "marker_sha256"},
        label="stage root identity",
    )
    root = _canonical_path(Path(root_identity["path"]), label="stage output root")
    if path != root / "analysis" / "calibration_stage.json":
        raise ValueError("calibration stage is not at its canonical root path")
    plan_identity = _mapping(
        payload["calibration_plan"], {"path", "sha256", "semantic_sha256"}, label="stage plan identity"
    )
    plan_path = _canonical_path(Path(plan_identity["path"]), label="stage calibration plan")
    if plan_path != root / "manifests" / "calibration_plan.json":
        raise ValueError("calibration stage crosses plan/output roots")
    plan_snapshot = snapshot_regular_file(plan_path, root=root)
    if plan_snapshot.sha256 != _digest(plan_identity["sha256"], label="stage plan sha256"):
        raise ValueError("calibration plan bytes changed after stage")
    plan = validate_calibration_plan(plan_path)
    if plan["semantic_sha256"] != plan_identity["semantic_sha256"]:
        raise ValueError("stage plan semantic identity mismatch")
    if payload["output_root_identity"] != plan["output_root_identity"]:
        raise ValueError("stage root differs from calibration plan")
    if payload["calibration_task_identity"] != plan["calibration_task_identity"]:
        raise ValueError("stage task differs from calibration plan")
    for key in ("development_suite_identity", "formal_suite_identity"):
        if payload[key] != plan[key]:
            raise ValueError(f"stage {key} differs from calibration plan")
    if canonical_json_bytes(payload["estimator"]) != canonical_json_bytes(plan["estimator"]):
        raise ValueError("stage estimator differs from calibration plan")
    _exact_value(payload["estimator"], _ESTIMATOR, label="stage estimator")
    for key in ("trainer_source_bundle", "binder_source_bundle", "calibrator_source_bundle"):
        if payload[key] != plan[key]:
            raise ValueError(f"stage {key} differs from calibration plan")
    _exact_value(
        payload["publication"],
        {"policy": "create_once_after_completed_calibration_source"},
        label="stage publication policy",
    )
    run_record = _mapping(
        payload["calibration_run"],
        {"path", "run_id", "task_id", "config_hash", "status_sha256", "artifact_manifest_sha256"},
        label="stage calibration run",
    )
    run_path = _canonical_path(Path(run_record["path"]), label="stage calibration run")
    if not run_path.is_relative_to(root) or str(run_path) != plan["resolved_config"]["run_identity"]["run_dir"]:
        raise ValueError("stage calibration run crosses or differs from plan root")
    completed = validate_completed_run(run_path, expected_config=plan["resolved_config"])
    expected_run = {
        "path": completed.run_dir,
        "run_id": completed.config["run_identity"]["run_id"],
        "task_id": completed.config["run_identity"]["task_id"],
        "config_hash": completed.config_hash,
        "status_sha256": completed.status_sha256,
        "artifact_manifest_sha256": completed.artifact_manifest_sha256,
    }
    if run_record != expected_run:
        raise ValueError("stage completed-run identity changed")
    rows = _rows_from_payload(payload["measurements"])
    derive_unit_coefficients(rows)
    checkpoints = payload["checkpoints"]
    if type(checkpoints) is not list or len(checkpoints) != 3:
        raise ValueError("stage checkpoints must contain three ordered rows")
    for raw, row, update in zip(checkpoints, rows, UPDATES):
        item = _mapping(
            raw,
            {"completed_updates", "transaction", "transaction_manifest_sha256", "path", "sha256"},
            label="stage checkpoint",
        )
        expected = {
            "completed_updates": row.completed_updates,
            "transaction": row.transaction,
            "transaction_manifest_sha256": row.transaction_manifest_sha256,
            "path": row.checkpoint_path,
            "sha256": row.checkpoint_sha256,
        }
        if item != expected or item["completed_updates"] != update:
            raise ValueError("stage checkpoint/measurement cross-binding mismatch")
        authenticated = authenticate_transaction_checkpoint(
            run_path, completed_updates=update, expected_config=plan["resolved_config"]
        )
        _coherent_checkpoint(authenticated, run_path, update)
        identity = authenticated.identity
        if (
            identity.transaction != row.transaction
            or identity.transaction_manifest_sha256 != row.transaction_manifest_sha256
            or identity.checkpoint_path != row.checkpoint_path
            or identity.checkpoint_sha256 != row.checkpoint_sha256
        ):
            raise ValueError("stage checkpoint identity changed")
        del authenticated
    revalidate_snapshot(plan_snapshot, root=root)
    return payload


def validate_calibration_stage(path: Path) -> Mapping[str, object]:
    absolute = _canonical_path(Path(path), label="calibration stage")
    snapshot = snapshot_regular_file(absolute, root=absolute.parent.parent)
    payload = _decode(snapshot.payload, label="calibration stage")
    if snapshot.payload != canonical_json_bytes(payload):
        raise ValueError("calibration stage is not canonical JSON")
    checked = _validate_stage_payload(payload, absolute)
    revalidate_snapshot(snapshot, root=absolute.parent.parent)
    return copy.deepcopy(checked)


def _unit_payload(
    *, stage_path: Path, stage_bytes: bytes, stage: Mapping[str, object]
) -> dict[str, object]:
    coefficients = derive_unit_coefficients(_rows_from_payload(stage["measurements"]))
    for _, _, dic_ratio, dts_ratio in CANDIDATES:
        V4StepConfig(coefficients["dic"], coefficients["dts"], dic_ratio, dts_ratio)
    run = stage["calibration_run"]
    payload: dict[str, object] = {
        "schema": UNIT_SCHEMA,
        "status": "APPROVED_FROZEN",
        "calibration_run": {
            "path": run["path"],
            "run_id": run["run_id"],
            "config_hash": run["config_hash"],
            "status_sha256": run["status_sha256"],
            "artifact_manifest_sha256": run["artifact_manifest_sha256"],
        },
        "calibration_stage": {
            "path": str(stage_path),
            "sha256": hashlib.sha256(stage_bytes).hexdigest(),
            "semantic_sha256": stage["semantic_sha256"],
        },
        "calibration_task_identity": copy.deepcopy(stage["calibration_task_identity"]),
        "development_suite_identity": copy.deepcopy(stage["development_suite_identity"]),
        "estimator": {
            key: copy.deepcopy(stage["estimator"][key])
            for key in (
                "name", "completed_updates", "dic_endpoint", "dts_endpoints",
                "candidate_independent",
            )
        },
        "coefficients": coefficients,
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    return payload


_UNIT_KEYS = {
    "schema", "status", "calibration_run", "calibration_stage",
    "calibration_task_identity", "development_suite_identity", "estimator",
    "coefficients", "semantic_sha256",
}


def _validate_unit_payload(payload: dict[str, object], path: Path) -> dict[str, object]:
    _mapping(payload, _UNIT_KEYS, label="unit coefficients")
    if payload["schema"] != UNIT_SCHEMA or payload["status"] != "APPROVED_FROZEN":
        raise ValueError("unit coefficient schema/status mismatch")
    _semantic(payload, label="unit coefficients")
    root = path.parent.parent
    if path != root / "analysis" / "unit_coefficients.json":
        raise ValueError("unit coefficient artifact is not at its canonical path")
    stage_identity = _mapping(
        payload["calibration_stage"], {"path", "sha256", "semantic_sha256"}, label="unit stage identity"
    )
    stage_path = _canonical_path(Path(stage_identity["path"]), label="unit calibration stage")
    if stage_path != root / "analysis" / "calibration_stage.json":
        raise ValueError("unit coefficient stage crosses roots")
    stage_snapshot = snapshot_regular_file(stage_path, root=root)
    if stage_snapshot.sha256 != _digest(stage_identity["sha256"], label="unit stage sha256"):
        raise ValueError("unit calibration stage bytes changed")
    stage = validate_calibration_stage(stage_path)
    if stage["semantic_sha256"] != stage_identity["semantic_sha256"]:
        raise ValueError("unit calibration stage semantic identity changed")
    expected = _unit_payload(stage_path=stage_path, stage_bytes=stage_snapshot.payload, stage=stage)
    if canonical_json_bytes(expected) != canonical_json_bytes(payload):
        raise ValueError("unit coefficients differ from authenticated raw evidence")
    revalidate_snapshot(stage_snapshot, root=root)
    return payload


def validate_unit_coefficients(path: Path) -> Mapping[str, object]:
    absolute = _canonical_path(Path(path), label="unit coefficients")
    snapshot = snapshot_regular_file(absolute)
    payload = _decode(snapshot.payload, label="unit coefficients")
    if snapshot.payload != canonical_json_bytes(payload):
        raise ValueError("unit coefficient artifact is not canonical JSON")
    checked = _validate_unit_payload(payload, absolute)
    revalidate_snapshot(snapshot)
    return copy.deepcopy(checked)


def _revalidate_publication_dependencies(
    *,
    plan_path: Path,
    plan: Mapping[str, object],
    run_path: Path,
    stage: Mapping[str, object],
) -> None:
    current_plan = validate_calibration_plan(plan_path)
    if canonical_json_bytes(current_plan) != canonical_json_bytes(plan):
        raise ValueError("calibration plan changed before publication")
    completed = validate_completed_run(run_path, expected_config=plan["resolved_config"])
    if stage["calibration_run"] != {
        "path": completed.run_dir,
        "run_id": completed.config["run_identity"]["run_id"],
        "task_id": completed.config["run_identity"]["task_id"],
        "config_hash": completed.config_hash,
        "status_sha256": completed.status_sha256,
        "artifact_manifest_sha256": completed.artifact_manifest_sha256,
    }:
        raise ValueError("completed calibration run changed before publication")
    rows = _rows_from_payload(stage["measurements"])
    for row, update in zip(rows, UPDATES):
        authenticated = authenticate_transaction_checkpoint(
            run_path, completed_updates=update, expected_config=plan["resolved_config"]
        )
        _coherent_checkpoint(authenticated, run_path, update)
        if (
            authenticated.identity.transaction != row.transaction
            or authenticated.identity.transaction_manifest_sha256
            != row.transaction_manifest_sha256
            or authenticated.identity.checkpoint_path != row.checkpoint_path
            or authenticated.identity.checkpoint_sha256 != row.checkpoint_sha256
        ):
            raise ValueError("calibration checkpoint changed before publication")
        del authenticated


def publish_calibration_bundle(
    output_root: Path,
    *,
    calibration_run: Path,
    calibration_plan: Path,
    development_suite: Path,
    formal_suite: Path,
) -> tuple[Path, Path, Path]:
    root = _canonical_path(Path(output_root), label="output root")
    plan_path = _canonical_path(Path(calibration_plan), label="calibration plan")
    plan = validate_calibration_plan(plan_path)
    if plan_path != root / "manifests" / "calibration_plan.json":
        raise ValueError("calibration plan crosses requested output root")
    run_path = _canonical_path(Path(calibration_run), label="calibration run")
    development_path = _canonical_path(Path(development_suite), label="development suite")
    formal_path = _canonical_path(Path(formal_suite), label="formal suite")
    if development_path != Path(plan["development_suite_identity"]["path"]):
        raise ValueError("development suite differs from calibration plan")
    if formal_path != Path(plan["formal_suite_identity"]["path"]):
        raise ValueError("formal suite differs from calibration plan")
    if run_path != Path(plan["resolved_config"]["run_identity"]["run_dir"]):
        raise ValueError("calibration run differs from calibration plan")
    _load_suites(development_path, formal_path)
    stage_path = root / "analysis" / "calibration_stage.json"
    unit_path = root / "analysis" / "unit_coefficients.json"
    registry_path = root / "manifests" / "candidate_registry.json"
    if registry_path.is_file():
        validate_candidate = __import__(
            "experiments.soft_loss_v4_registry", fromlist=["validate_candidate_registry"]
        ).validate_candidate_registry
        validate_candidate(registry_path)
        return stage_path, unit_path, registry_path
    measurements = collect_unit_gradient_measurements(run_path, plan_path, development_path)
    stage = build_calibration_stage(
        calibration_plan=plan_path,
        calibration_run=run_path,
        measurements=measurements,
    )
    stage_bytes = canonical_json_bytes(stage)
    unit = _unit_payload(stage_path=stage_path, stage_bytes=stage_bytes, stage=stage)
    unit_bytes = canonical_json_bytes(unit)
    registry = _candidate_registry_payload(
        unit_path, unit, unit_bytes=unit_bytes
    )
    registry_bytes = canonical_json_bytes(registry)

    def precommit() -> None:
        _revalidate_publication_dependencies(
            plan_path=plan_path, plan=plan, run_path=run_path, stage=stage
        )

    _publish_bundle(
        {stage_path: stage_bytes, unit_path: unit_bytes, registry_path: registry_bytes},
        output_root=root,
        precommit=precommit,
    )
    from experiments.soft_loss_v4_registry import validate_candidate_registry

    validate_candidate_registry(registry_path)
    return stage_path, unit_path, registry_path


def build_calibration_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish authenticated soft-loss-v4 calibration evidence",
        allow_abbrev=False,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration-plan", type=Path, required=True)
    return parser


def calibration_main(argv: Sequence[str] | None = None) -> int:
    args = build_calibration_parser().parse_args(argv)
    plan = validate_calibration_plan(args.calibration_plan)
    root = Path(plan["output_root_identity"]["path"])
    if Path(args.output_root) != root:
        raise ValueError("CLI output root differs from calibration plan")
    publish_calibration_bundle(
        root,
        calibration_run=Path(plan["resolved_config"]["run_identity"]["run_dir"]),
        calibration_plan=Path(args.calibration_plan),
        development_suite=Path(plan["development_suite_identity"]["path"]),
        formal_suite=Path(plan["formal_suite_identity"]["path"]),
    )
    return 0


__all__ = [
    "UnitGradientMeasurement",
    "build_calibration_parser",
    "build_calibration_plan",
    "build_calibration_stage",
    "calibration_main",
    "collect_unit_gradient_measurements",
    "derive_unit_coefficients",
    "publish_calibration_bundle",
    "validate_calibration_plan",
    "validate_calibration_stage",
    "validate_unit_coefficients",
    "write_calibration_plan",
]
