"""Strict read-only analyzer for the registered native-256 v6 ablation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Callable, Mapping, Sequence

if __package__ in (None, ""):
    _BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
    if str(_BOOTSTRAP_ROOT) not in sys.path:
        sys.path.insert(0, str(_BOOTSTRAP_ROOT))

import cv2
import numpy as np
import torch

from experiments.run_soft_loss_v6_ablation10_256 import validate_manifest
from shadowforge_pipeline.soft_voxel_v3.data import normalize_target
from shadowforge_pipeline.soft_voxel_v3.protocol import SadekarTFSProtocol
from shadowforge_pipeline.soft_voxel_v3.thresholding import render_thresholded
from shadowforge_pipeline.soft_voxel_v6.core import (
    PROFILE_FLAGS,
    PROFILE_ORDER,
    TASK_ORDER,
    fixed_camera_payload,
    map_suite_views,
    v6_auxiliary_multiplier,
    v6_learning_rate,
)
from shadowforge_pipeline.soft_voxel_v6.provenance import (
    atomic_bytes,
    canonical_hash,
    canonical_json_bytes,
    decode_canonical_json,
    reject_symlink_components,
    revalidate_snapshot,
    snapshot_regular_file,
)
from shadowforge_pipeline.soft_voxel_v6.statistics import evaluate_soft_outcome
from shadowforge_pipeline.soft_voxel_v6.trainer import (
    StepTelemetry,
    validate_completed_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK2_BINDING = {
    ".superpowers/sdd/2026-08-28-soft-loss-v6-256-ablation/reports/task-2-trainer-runner.md": "960973c73f07a7611d2c390606adae4d6498ebdf81562151d8d988143292d003",
    "shadowforge_pipeline/soft_voxel_v6/__init__.py": "b64a60d343e9df3fee65ac5657140f54800b02587cd9aa4d71cd622a873416e3",
    "shadowforge_pipeline/soft_voxel_v6/provenance.py": "dafeb2be4f3084b61c8d24c94aea04fa406f51e3ecbe9e5ed185e82bf74ebc0f",
    "shadowforge_pipeline/soft_voxel_v6/trainer.py": "54f21bee70618dbc9164ccd166129ea3c3986c374b599a7c60cf732ba3b85f67",
    "experiments/run_soft_loss_v6_ablation10_256.py": "946da49539b917141626d445b2cc010cf6b818a18ae1170baf1d39000312bfcd",
    "tests/test_soft_loss_v6_trainer.py": "d33b8d899db0550e33e8b72108d74a93772c4b2cb6ba5bf77c54b896403780b4",
    "tests/test_soft_loss_v6_runner.py": "2e7cd03963cc1041b16f6c209b01ec06209eb55efd0b7e06c75f6e13e8797798",
}
TASK2_BUNDLE_SHA256 = "1a54526ce5bc8e445638a8390bc6daa849c4415e1c46d9e89eabc08ba67cc81d"
_CSV_FIELDS = (
    "task", "arm", "run_path", "config_sha256", "artifacts_sha256",
    "mean_iou", "mean_dice", "mean_recall", "mean_fpr",
    "j_full_ramp_eligible", "j_full_ramp_zero", "j_full_ramp_active",
    "h_full_ramp_eligible", "h_full_ramp_zero", "h_full_ramp_active",
)


@dataclass(frozen=True)
class EvidenceSnapshot:
    path: str
    sha256: str
    size: int
    payload: bytes | None


def _canonical_root(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate != Path(os.path.abspath(candidate)):
        raise ValueError("output root must be absolute and canonical")
    reject_symlink_components(candidate)
    metadata = candidate.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("output root must be a real directory")
    return candidate


def _snapshot_bytes(path: Path, *, label: str) -> EvidenceSnapshot:
    snapshot = snapshot_regular_file(path)
    payload = Path(snapshot.path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != snapshot.sha256:
        raise ValueError(f"{label} changed during snapshot")
    revalidate_snapshot(snapshot)
    return EvidenceSnapshot(snapshot.path, snapshot.sha256, snapshot.size, payload)


def _snapshot_identity(path: Path, *, label: str) -> EvidenceSnapshot:
    """Snapshot a large authenticated file by streamed digest without retaining it."""

    snapshot = snapshot_regular_file(path)
    revalidate_snapshot(snapshot)
    return EvidenceSnapshot(snapshot.path, snapshot.sha256, snapshot.size, None)


def _revalidate_all(snapshots: Sequence[EvidenceSnapshot]) -> None:
    for item in snapshots:
        current = snapshot_regular_file(Path(item.path))
        if current.sha256 != item.sha256 or current.size != item.size:
            raise ValueError(f"authenticated evidence changed: {item.path}")


def _decode_json(raw: bytes, *, label: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key in {label}: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if type(value) is not dict or raw != canonical_json_bytes(value):
        raise ValueError(f"{label} is not an exact canonical JSON object")
    return value


def _decode_exact_npy(
    raw: bytes, *, dtype: np.dtype, shape: tuple[int, ...], label: str
) -> np.ndarray:
    buffer = io.BytesIO(raw)
    try:
        value = np.load(buffer, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"{label} is not an exact no-pickle NPY") from error
    if buffer.tell() != len(raw):
        raise ValueError(f"{label} has trailing bytes")
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != dtype
        or value.shape != shape
        or not value.flags.c_contiguous
    ):
        raise ValueError(f"{label} dtype/shape/layout mismatch")
    if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(f"{label} is nonfinite")
    return value


def _validate_task2_binding(manifest: Mapping[str, object]) -> list[EvidenceSnapshot]:
    snapshots: list[EvidenceSnapshot] = []
    for relative, expected in TASK2_BINDING.items():
        item = _snapshot_bytes(PROJECT_ROOT / relative, label=relative)
        if item.sha256 != expected:
            raise ValueError(f"approved Task2 byte mismatch: {relative}")
        snapshots.append(item)
    implementation = manifest.get("implementation")
    if type(implementation) is not dict or implementation.get("bundle_sha256") != TASK2_BUNDLE_SHA256:
        raise ValueError("manifest does not bind the approved Task2 source closure")
    return snapshots


def _validate_cell_matrix(root: Path, cells: object) -> None:
    if type(cells) is not list or len(cells) != 40:
        raise ValueError("manifest must contain exactly 40 cells")
    expected = tuple((task, arm) for task in TASK_ORDER for arm in PROFILE_ORDER)
    for row, (task, arm) in zip(cells, expected):
        if type(row) is not dict or row.get("task") != task or row.get("arm") != arm:
            raise ValueError("manifest cells are missing, duplicated, or reordered")
        exact_path = root / "runs" / task / arm / "seed_17"
        config = row.get("resolved_config")
        if (
            type(config) is not dict
            or row.get("run_path") != str(exact_path)
            or config.get("task") != task
            or config.get("arm") != arm
            or config.get("seed") != 17
            or config.get("resolution") != 256
            or config.get("updates") != 3000
            or type(config.get("run")) is not dict
            or config["run"].get("root") != str(root)
            or config["run"].get("path") != str(exact_path)
        ):
            raise ValueError("cell is not the exact registered native-256 endpoint")


def _finite_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be an exact finite float")
    return value


def _initial_audit(arm: str) -> dict[str, object]:
    return {
        "schema": "pcraft_soft_loss_v6_audit_v1", "arm": arm,
        "eligible_updates": 0, "j_zero_updates": 0, "h_zero_updates": 0,
        "j_active_updates": 0, "h_active_updates": 0,
        "parameters_changed_updates": 0, "clip_calls": 0, "optimizer_steps": 0,
        "max_j_ratio": 0.0, "max_h_ratio": 0.0, "max_combined_ratio": 0.0,
        "j_full_ramp_eligible": 0, "h_full_ramp_eligible": 0,
        "j_full_ramp_zero": 0, "h_full_ramp_zero": 0,
    }


def _validate_tail(details: object) -> None:
    keys = {"view", "background_count", "selected_count", "selected_indices", "score_min", "score_mean", "score_max"}
    if type(details) is not tuple or len(details) != 6:
        raise ValueError("active H branch requires six tail-detail records")
    for view, row in enumerate(details):
        if type(row) is not dict or set(row) != keys or row["view"] != view:
            raise ValueError("H tail details have the wrong exact schema")
        background, selected = row["background_count"], row["selected_count"]
        indices = row["selected_indices"]
        if (
            type(background) is not int or background <= 0
            or type(selected) is not int or selected != max(1, math.ceil(0.05 * background))
            or type(indices) is not list or len(indices) != selected
            or any(type(index) is not int or index < 0 for index in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("H tail selection is invalid")
        scores = tuple(_finite_float(row[name], label=name) for name in ("score_min", "score_mean", "score_max"))
        if not scores[0] <= scores[1] <= scores[2]:
            raise ValueError("H tail score order is invalid")


def _derive_engineering(
    history: object, *, arm: str
) -> tuple[dict[str, object], dict[str, dict[str, int]]]:
    if type(history) is not list or len(history) != 3000:
        raise ValueError("terminal history must contain exactly 3000 rows")
    exact_fields = {field.name for field in fields(StepTelemetry)}
    audit = _initial_audit(arm)
    flags = PROFILE_FLAGS[arm]
    for update, row in enumerate(history, start=1):
        if type(row) is not dict or set(row) != exact_fields:
            raise ValueError("telemetry row has the wrong exact schema")
        integer_names = ("completed_updates", "clip_calls", "optimizer_steps", "peak_cuda_allocated_bytes")
        if any(type(row[name]) is not int for name in integer_names):
            raise ValueError("telemetry integer/bool alias")
        if row["completed_updates"] != update or row["clip_calls"] != 1 or row["optimizer_steps"] != 1:
            raise ValueError("telemetry order or one-clip/step contract failed")
        if row["learning_rate"] != v6_learning_rate(update - 1) or row["auxiliary_multiplier"] != v6_auxiliary_multiplier(update - 1):
            raise ValueError("telemetry schedule differs from the frozen functions")
        float_names = exact_fields - set(integer_names) - {"j_zero", "h_zero", "j_loss", "h_loss", "tail_details"}
        for name in float_names:
            _finite_float(row[name], label=name)
        for name in ("j_zero", "h_zero"):
            if type(row[name]) is not bool:
                raise ValueError("telemetry zero flag must be exact bool")
        multiplier = row["auxiliary_multiplier"]
        if (
            row["base_loss"] < 0.0 or row["base_norm"] <= 1e-12
            or row["pre_clip_norm"] <= 0.0 or row["post_clip_norm"] <= 0.0
            or row["post_clip_norm"] > min(row["pre_clip_norm"], 1.0) + 1e-6
            or row["parameter_delta_norm"] < 0.0 or row["elapsed_seconds"] <= 0.0
            or row["peak_cuda_allocated_bytes"] < 0
        ):
            raise ValueError("base/clip/resource telemetry violates the frozen contract")
        for name, cap in (("j", 0.20), ("h", 0.10)):
            active = multiplier > 0.0 and flags[name]
            loss, raw, ratio, zero = row[f"{name}_loss"], row[f"{name}_raw_norm"], row[f"{name}_ratio"], row[f"{name}_zero"]
            if not active:
                if loss is not None or raw != 0.0 or ratio != 0.0 or zero is not False:
                    raise ValueError("disabled auxiliary branch was constructed")
            else:
                _finite_float(loss, label=f"{name} loss")
                if raw < 0.0 or zero is not (raw <= 1e-12):
                    raise ValueError("active auxiliary zero flag differs from raw norm")
                upper = float(np.nextafter(np.float32(multiplier * cap), np.float32(math.inf)))
                if ratio < 0.0 or ratio > upper or (zero and ratio != 0.0):
                    raise ValueError("auxiliary ratio exceeds its frozen cap")
        combined_upper = float(np.nextafter(np.float32(multiplier * 0.30), np.float32(math.inf)))
        if row["combined_ratio"] < 0.0 or row["combined_ratio"] > combined_upper:
            raise ValueError("combined auxiliary ratio exceeds its frozen cap")
        if multiplier > 0.0 and flags["h"]:
            _validate_tail(row["tail_details"])
        elif row["tail_details"] != ():
            raise ValueError("inactive H branch carries tail details")
        audit["clip_calls"] += 1; audit["optimizer_steps"] += 1
        if row["parameter_delta_norm"] > 0.0:
            audit["parameters_changed_updates"] += 1
        audit["max_j_ratio"] = max(audit["max_j_ratio"], row["j_ratio"])
        audit["max_h_ratio"] = max(audit["max_h_ratio"], row["h_ratio"])
        audit["max_combined_ratio"] = max(audit["max_combined_ratio"], row["combined_ratio"])
        if multiplier > 0.0:
            audit["eligible_updates"] += 1
            for name in ("j", "h"):
                if not flags[name]:
                    continue
                audit[f"{name}_{'zero' if row[f'{name}_zero'] else 'active'}_updates"] += 1
                if update >= 901:
                    audit[f"{name}_full_ramp_eligible"] += 1
                    if row[f"{name}_zero"]:
                        audit[f"{name}_full_ramp_zero"] += 1
    activity: dict[str, dict[str, int]] = {}
    for name in ("j", "h"):
        eligible = int(audit[f"{name}_full_ramp_eligible"])
        zero = int(audit[f"{name}_full_ramp_zero"])
        if flags[name]:
            activity[name] = {"eligible": eligible, "zero": zero, "active": eligible - zero}
    return audit, activity


def _extend_terminal_audit(prefix: Mapping[str, object], *, arm: str) -> dict[str, object]:
    if type(prefix) is not dict:
        raise ValueError("checkpoint prefix audit must be an exact mapping")
    terminal = dict(prefix)
    flags = PROFILE_FLAGS[arm]
    for name in ("j", "h"):
        eligible = prefix[f"{name}_full_ramp_eligible"]
        zero = prefix[f"{name}_full_ramp_zero"]
        if type(eligible) is not int or type(zero) is not int:
            raise ValueError("checkpoint prefix audit counters must be exact integers")
        terminal[f"{name}_full_ramp_raw_zero_fraction"] = (
            zero / 2100.0 if flags[name] and eligible == 2100 else None
        )
    return terminal


def _recompute_engineering(
    history: object,
    persisted_audit: object,
    *,
    checkpoint_audit: object,
    arm: str,
) -> tuple[dict[str, object], dict[str, dict[str, int]]]:
    prefix, activity = _derive_engineering(history, arm=arm)
    if type(checkpoint_audit) is not dict or canonical_json_bytes(prefix) != canonical_json_bytes(checkpoint_audit):
        raise ValueError("terminal checkpoint audit differs from history-derived prefix")
    terminal = _extend_terminal_audit(prefix, arm=arm)
    if type(persisted_audit) is not dict or canonical_json_bytes(terminal) != canonical_json_bytes(persisted_audit):
        raise ValueError("checkpoint/artifact/recomputed audit disagreement")
    return terminal, activity


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes(order="C")).hexdigest()


def _evaluate_masks(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
    if (
        not isinstance(predicted, torch.Tensor) or predicted.dtype != torch.bool
        or not isinstance(target, torch.Tensor) or target.dtype != torch.bool
        or predicted.shape != target.shape or predicted.ndim != 3
        or predicted.shape[0] != 6 or any(size == 0 for size in predicted.shape)
    ):
        raise ValueError("metric masks must be equal non-empty boolean [6,H,W]")
    rows: list[dict[str, object]] = []
    for view in range(6):
        p, t = predicted[view], target[view]
        tp = int(torch.count_nonzero(p & t).item())
        fp = int(torch.count_nonzero(p & ~t).item())
        fn = int(torch.count_nonzero(~p & t).item())
        tn = int(torch.count_nonzero(~p & ~t).item())
        union, dice_denominator = tp + fp + fn, 2 * tp + fp + fn
        foreground, background = tp + fn, fp + tn
        if min(union, dice_denominator, foreground, background) <= 0:
            raise ValueError(f"invalid metric denominator in view {view}")
        rows.append({
            "view": view, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "iou": tp / union, "dice": 2 * tp / dice_denominator,
            "recall": tp / foreground, "fpr": fp / background,
        })
    aggregate = {
        f"mean_{name}": float(sum(row[name] for row in rows) / 6.0)
        for name in ("iou", "dice", "recall", "fpr")
    }
    if not all(math.isfinite(value) for value in aggregate.values()):
        raise ValueError("metric aggregation is nonfinite")
    return {
        "schema": "pcraft_soft_loss_v6_metrics_v1", "completed_updates": 3000,
        "threshold": 0.05, "per_view": rows, "aggregate": aggregate,
    }


def _load_targets(config: Mapping[str, object]) -> torch.Tensor:
    resolution = int(config["resolution"])
    base: list[torch.Tensor] = []
    mirrors: list[torch.Tensor] = []
    for row in config["inputs"]["raw_xyz"]:
        snapshot = _snapshot_bytes(Path(row["path"]), label="raw target")
        if snapshot.sha256 != row["sha256"]:
            raise ValueError("raw input hash differs from registration")
        image = cv2.imdecode(np.frombuffer(snapshot.payload, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("raw input is not grayscale")
        base.append(torch.from_numpy(normalize_target(image, resolution, False) > 0))
        mirrors.append(torch.from_numpy(normalize_target(cv2.flip(image, 1), resolution, False) > 0))
    mapped = map_suite_views(tuple(zip("xyz", base)), tuple(zip("xyz", mirrors)))
    identity = config["targets"]
    if (
        _tensor_hash(torch.stack(base)) != identity["base_sha256"]
        or _tensor_hash(torch.stack(mirrors)) != identity["mirror_sha256"]
        or _tensor_hash(mapped) != identity["mapped_six_sha256"]
    ):
        raise ValueError("independently reconstructed targets differ")
    return mapped


def _recompute_cell(
    config: Mapping[str, object], snapshots: Mapping[str, EvidenceSnapshot]
) -> tuple[dict[str, object], dict[str, dict[str, int]]]:
    checkpoint = torch.load(io.BytesIO(snapshots["checkpoint"].payload), map_location="cpu", weights_only=False)
    if type(checkpoint) is not dict or checkpoint.get("completed_updates") != 3000:
        raise ValueError("terminal checkpoint is not update 3000")
    logits_tensor = checkpoint.get("logits")
    resolution = int(config["resolution"])
    if resolution <= 0:
        raise ValueError("analysis resolution must be positive")
    shape = (1, 1, resolution, resolution, resolution)
    if (
        not isinstance(logits_tensor, torch.Tensor) or logits_tensor.dtype != torch.float32
        or logits_tensor.device.type != "cpu" or tuple(logits_tensor.shape) != shape
        or not logits_tensor.is_contiguous() or not torch.isfinite(logits_tensor).all().item()
    ):
        raise ValueError("terminal checkpoint logits are malformed")
    persisted_audit = _decode_json(snapshots["audit"].payload, label="audit")
    step_times = checkpoint.get("step_times")
    history = checkpoint.get("history")
    if (
        type(step_times) is not list or len(step_times) != 3000
        or type(history) is not list or len(history) != 3000
        or any(
            type(value) is not float or value != history[index]["elapsed_seconds"]
            for index, value in enumerate(step_times)
        )
    ):
        raise ValueError("checkpoint step-times are not the exact telemetry prefix")
    _, activity = _recompute_engineering(
        history, persisted_audit,
        checkpoint_audit=checkpoint.get("audit"), arm=str(config["arm"]),
    )
    logits = _decode_exact_npy(snapshots["logits"].payload, dtype=np.dtype("float32"), shape=shape, label="final logits")
    density = _decode_exact_npy(snapshots["density"].payload, dtype=np.dtype("float32"), shape=shape, label="final density")
    hard = _decode_exact_npy(snapshots["hard"].payload, dtype=np.dtype("uint8"), shape=shape, label="final hard")
    if not np.array_equal(logits, logits_tensor.numpy()):
        raise ValueError("final logits differ bit-for-bit from terminal checkpoint")
    device = torch.device(str(config["device"]))
    logits_device = logits_tensor.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        expected_density_tensor = torch.sigmoid(logits_device)
    expected_density = expected_density_tensor.cpu().contiguous().numpy()
    if not np.array_equal(density, expected_density):
        raise ValueError("final density differs from registered-device sigmoid")
    if not np.isin(hard, (0, 1)).all() or not np.array_equal(hard, (density >= 0.05).astype(np.uint8)):
        raise ValueError("final hard differs from density >= .05")
    targets = _load_targets(config).to(device)
    protocol = SadekarTFSProtocol(device=device, image_size=resolution, zdist=1.7, volume_extent_world=1.7, n_pts_per_ray=150)
    camera = fixed_camera_payload()
    if (
        _tensor_hash(protocol.cameras.R.float()) != config["cameras"]["rotation_sha256"]
        or _tensor_hash(protocol.cameras.T.float()) != config["cameras"]["translation_sha256"]
        or camera["payload_sha256"] != config["cameras"]["payload_sha256"]
    ):
        raise ValueError("reconstructed camera R/T differ from registration")
    with torch.no_grad():
        masks = render_thresholded(protocol, expected_density_tensor, 0.05).mask
        metrics = _evaluate_masks(masks, targets >= 0.5)
    persisted_metrics = _decode_json(snapshots["metrics"].payload, label="metrics")
    if canonical_json_bytes(metrics) != canonical_json_bytes(persisted_metrics):
        raise ValueError("persisted metrics differ from independent rerender")
    resource = _decode_json(snapshots["resource"].payload, label="resource")
    expected_resource = {
        "schema": "pcraft_soft_loss_v6_resource_v1",
        "step_times": step_times,
        "all_finite_positive": all(
            type(row["elapsed_seconds"]) is float
            and math.isfinite(row["elapsed_seconds"])
            and row["elapsed_seconds"] > 0.0
            for row in history
        ),
        "peak_cuda_allocated_bytes": max(row["peak_cuda_allocated_bytes"] for row in history),
        "device_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else None,
        "cuda_executed": device.type == "cuda",
    }
    if canonical_json_bytes(resource) != canonical_json_bytes(expected_resource):
        raise ValueError("persisted resource evidence differs from history/device recomputation")
    return metrics, activity


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _render_prevalidated_outputs(
    root: Path,
    metric_rows: Sequence[Mapping[str, object]],
    activity: Mapping[str, Mapping[str, Mapping[str, Mapping[str, int]]]],
    *,
    snapshots: Sequence[EvidenceSnapshot],
    report_context: Mapping[str, object] | None = None,
) -> tuple[bytes, bytes]:
    outcome = evaluate_soft_outcome(metric_rows, activity)
    csv_rows: list[dict[str, object]] = []
    context_rows = [] if report_context is None else list(report_context.get("cells", []))
    for index, row in enumerate(metric_rows):
        task, arm = str(row["task"]), str(row["arm"])
        components = activity[task][arm]
        context = context_rows[index] if context_rows else {}
        csv_rows.append({
            "task": task, "arm": arm,
            "run_path": context.get("run_path", "synthetic"),
            "config_sha256": context.get("config_sha256", "0" * 64),
            "artifacts_sha256": context.get("artifacts_sha256", "0" * 64),
            **{name: row[name] for name in ("mean_iou", "mean_dice", "mean_recall", "mean_fpr")},
            "j_full_ramp_eligible": components.get("j", {}).get("eligible", 0),
            "j_full_ramp_zero": components.get("j", {}).get("zero", 0),
            "j_full_ramp_active": components.get("j", {}).get("active", 0),
            "h_full_ramp_eligible": components.get("h", {}).get("eligible", 0),
            "h_full_ramp_zero": components.get("h", {}).get("zero", 0),
            "h_full_ramp_active": components.get("h", {}).get("active", 0),
        })
    csv_payload = _csv_bytes(csv_rows)
    analyzer_files = {
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        str((PROJECT_ROOT / "shadowforge_pipeline/soft_voxel_v6/statistics.py").resolve()): hashlib.sha256((PROJECT_ROOT / "shadowforge_pipeline/soft_voxel_v6/statistics.py").read_bytes()).hexdigest(),
    }
    report = {
        "schema": "pcraft_soft_loss_v6_soft_report_v1",
        "output_root": str(root),
        "manifest": None if report_context is None else report_context.get("manifest"),
        "task2_approval": {"binding": "CLEAN_2026_08_29_FIX2", "bundle_sha256": TASK2_BUNDLE_SHA256, "files_sha256": TASK2_BINDING},
        "analyzer_files_sha256": analyzer_files,
        "evidence_sha256": {item.path: item.sha256 for item in sorted(snapshots, key=lambda value: value.path)},
        "csv": {"path": str(root / "analysis/soft_rows.csv"), "sha256": hashlib.sha256(csv_payload).hexdigest()},
        "rows": list(metric_rows),
        "cells": [] if report_context is None else list(report_context.get("cells", [])),
        "outcome": outcome,
    }
    return csv_payload, canonical_json_bytes(report)


def _publish_outputs(
    root: Path, csv_payload: bytes, report_payload: bytes, *, revalidate: Callable[[str], None]
) -> None:
    csv_path = root / "analysis/soft_rows.csv"
    report_path = root / "analysis/soft_report.json"
    reject_symlink_components(csv_path, must_exist=False)
    reject_symlink_components(report_path, must_exist=False)
    if report_path.exists():
        raise FileExistsError("committed soft report already exists")
    revalidate("before_csv")
    if csv_path.exists():
        if not csv_path.is_file() or csv_path.read_bytes() != csv_payload:
            raise ValueError("existing recovery CSV differs from recomputation")
    else:
        atomic_bytes(csv_path, csv_payload, create_once=True)
    revalidate("before_report")
    atomic_bytes(report_path, report_payload, create_once=True)


def _verify_outputs(
    root: Path, csv_payload: bytes, report_payload: bytes, *, revalidate: Callable[[str], None]
) -> None:
    csv_path = root / "analysis/soft_rows.csv"
    report_path = root / "analysis/soft_report.json"
    reject_symlink_components(csv_path)
    reject_symlink_components(report_path)
    revalidate("verify_before_compare")
    if not csv_path.is_file() or csv_path.read_bytes() != csv_payload:
        raise ValueError("committed CSV differs from complete recomputation")
    if not report_path.is_file() or report_path.read_bytes() != report_payload:
        raise ValueError("committed report differs from complete recomputation")
    revalidate("verify_after_compare")


def _synthetic_passing_inputs_for_test() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    activity: dict[str, object] = {}
    for task in TASK_ORDER:
        activity[task] = {
            "base": {},
            "j_only": {"j": {"eligible": 2100, "zero": 0, "active": 2100}},
            "h_only": {"h": {"eligible": 2100, "zero": 0, "active": 2100}},
            "full": {
                "j": {"eligible": 2100, "zero": 0, "active": 2100},
                "h": {"eligible": 2100, "zero": 0, "active": 2100},
            },
        }
        for arm, values in {
            "base": (0.94, 0.95, 0.98, 0.01),
            "j_only": (0.95, 0.958, 0.98, 0.011),
            "h_only": (0.95, 0.958, 0.98, 0.009),
            "full": (0.96, 0.959, 0.98, 0.01),
        }.items():
            rows.append(dict(zip(("task", "arm", "mean_iou", "mean_dice", "mean_recall", "mean_fpr"), (task, arm, *values))))
    return rows, activity


def analyze(output_root: Path, *, verify: bool = False) -> dict[str, object]:
    root = _canonical_root(output_root)
    manifest_path = root / "manifests/ablation10.json"
    manifest_snapshot = _snapshot_bytes(manifest_path, label="manifest")
    manifest = _decode_json(manifest_snapshot.payload, label="manifest")
    if manifest.get("output_root") != str(root):
        raise ValueError("CLI root differs from manifest.output_root")
    validate_manifest(manifest)
    _validate_cell_matrix(root, manifest.get("cells"))
    all_snapshots = [manifest_snapshot, *_validate_task2_binding(manifest)]
    all_snapshots.extend(
        [
            _snapshot_bytes(Path(__file__).resolve(), label="analyzer source"),
            _snapshot_bytes(
                PROJECT_ROOT / "shadowforge_pipeline/soft_voxel_v6/statistics.py",
                label="statistics source",
            ),
        ]
    )
    metric_rows: list[dict[str, object]] = []
    activity: dict[str, object] = {task: {} for task in TASK_ORDER}
    cell_context: list[dict[str, object]] = []
    configs: list[dict[str, object]] = []
    for cell in manifest["cells"]:
        config = cell["resolved_config"]
        configs.append(config)
        completed = validate_completed_run(config)
        if completed.completed_updates != 3000:
            raise ValueError("analyzer accepts only completed update-3000 cells")
        run_dir = Path(cell["run_path"])
        names = {
            "status": "STATUS.json", "checkpoint": "checkpoints/step_3000.pt",
            "logits": "artifacts/final_logits.npy", "density": "artifacts/final_density.npy",
            "hard": "artifacts/final_hard_005.npy", "metrics": "artifacts/metrics.json",
            "audit": "artifacts/audit.json", "resource": "artifacts/resource.json",
            "artifact_manifest": "artifacts/artifacts.json", "command": "COMMAND.json",
            "run": "RUN.json", "environment": "environment.json", "sources": "sources.json",
            "inputs": "inputs.json", "latest": "checkpoints/LATEST.json",
        }
        cell_snapshots = {key: _snapshot_bytes(run_dir / relative, label=key) for key, relative in names.items()}
        artifact_index = _decode_json(cell_snapshots["artifact_manifest"].payload, label="artifact manifest")
        if type(artifact_index.get("files_sha256")) is not dict:
            raise ValueError("artifact manifest lacks an exact file index")
        for relative, digest in artifact_index["files_sha256"].items():
            extra = _snapshot_identity(run_dir / relative, label=relative)
            if extra.sha256 != digest:
                raise ValueError("artifact index digest differs from snapped bytes")
            if extra.path not in {item.path for item in cell_snapshots.values()}:
                all_snapshots.append(extra)
        all_snapshots.extend(cell_snapshots.values())
        metrics, component_activity = _recompute_cell(config, cell_snapshots)
        aggregate = metrics["aggregate"]
        metric_rows.append({
            "task": cell["task"], "arm": cell["arm"],
            "mean_iou": aggregate["mean_iou"], "mean_dice": aggregate["mean_dice"],
            "mean_recall": aggregate["mean_recall"], "mean_fpr": aggregate["mean_fpr"],
        })
        activity[cell["task"]][cell["arm"]] = component_activity
        cell_context.append({
            "task": cell["task"], "arm": cell["arm"], "run_path": str(run_dir),
            "config_sha256": completed.config_sha256,
            "artifacts_sha256": completed.artifacts_sha256,
            "terminal_checkpoint_sha256": cell_snapshots["checkpoint"].sha256,
            "metrics": metrics,
            "activity": component_activity,
        })
    # Raw inputs, suite and specification are also explicit evidence snapshots.
    seen = {item.path for item in all_snapshots}
    for config in configs:
        for path in [config["inputs"]["suite"]["path"], config["specification"]["path"]] + [item["path"] for item in config["inputs"]["raw_xyz"]]:
            canonical = str(Path(path))
            if canonical not in seen:
                item = _snapshot_bytes(Path(path), label="static evidence")
                all_snapshots.append(item); seen.add(item.path)
    context = {
        "manifest": {"path": str(manifest_path), "sha256": manifest_snapshot.sha256, "semantic_sha256": manifest["semantic_sha256"]},
        "cells": cell_context,
    }
    csv_payload, report_payload = _render_prevalidated_outputs(root, metric_rows, activity, snapshots=all_snapshots, report_context=context)
    def revalidate(stage: str) -> None:
        for config in configs:
            validate_completed_run(config)
        _revalidate_all(all_snapshots)
    if verify:
        _verify_outputs(root, csv_payload, report_payload, revalidate=revalidate)
    else:
        _publish_outputs(root, csv_payload, report_payload, revalidate=revalidate)
    return _decode_json(report_payload, label="generated report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze the exact P-CRAFT v6 native-256 ablation", allow_abbrev=False)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyze(args.output_root, verify=args.verify)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
