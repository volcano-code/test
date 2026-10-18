"""Exact authenticated 10x4 native-256 soft-loss-v6 experiment runner."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch

from experiments.soft_loss_v4_registry import load_v4_suite_identity
from shadowforge_pipeline.soft_voxel_v3.data import normalize_target
from shadowforge_pipeline.soft_voxel_v6.core import (
    PROFILE_FLAGS,
    PROFILE_ORDER,
    TASK_ORDER,
    build_fixed_visual_hull_logits,
    fixed_camera_payload,
    map_suite_views,
)
from shadowforge_pipeline.soft_voxel_v6.provenance import (
    atomic_json,
    canonical_hash,
    canonical_json_bytes,
    decode_canonical_json,
    environment_identity,
    implementation_bundle,
    reject_symlink_components,
    revalidate_snapshot,
    sha256_file,
    snapshot_regular_file,
)
from shadowforge_pipeline.soft_voxel_v6.trainer import (
    CompletedRun,
    ResumableRun,
    recover_interrupted_checkpoint_publication,
    train_cell,
    validate_completed_run,
    validate_resumable_run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / ".superpowers/sdd/2026-08-28-soft-loss-v6-256-ablation/spec.md"
REAL_OUTPUT_ROOT = Path("/data/xvxinyao/pcraft_soft_loss_v6_ablation10_256_20260828")
MANIFEST_SCHEMA = "pcraft_soft_loss_v6_ablation10_manifest_v1"
CONFIG_SCHEMA = "pcraft_soft_loss_v6_resolved_config_v1"
TRAINING_MODULE_SOURCES = (
    ("experiments.run_soft_loss_v6_ablation10_256", "experiments/run_soft_loss_v6_ablation10_256.py"),
    ("experiments.soft_loss_v3_registry", "experiments/soft_loss_v3_registry.py"),
    ("experiments.soft_loss_v4_registry", "experiments/soft_loss_v4_registry.py"),
    ("shadowforge_pipeline.soft_voxel_v3", "shadowforge_pipeline/soft_voxel_v3/__init__.py"),
    ("shadowforge_pipeline.soft_voxel_v3.contours", "shadowforge_pipeline/soft_voxel_v3/contours.py"),
    ("shadowforge_pipeline.soft_voxel_v3.data", "shadowforge_pipeline/soft_voxel_v3/data.py"),
    ("shadowforge_pipeline.soft_voxel_v3.diagnostics", "shadowforge_pipeline/soft_voxel_v3/diagnostics.py"),
    ("shadowforge_pipeline.soft_voxel_v3.losses", "shadowforge_pipeline/soft_voxel_v3/losses.py"),
    ("shadowforge_pipeline.soft_voxel_v3.lovasz", "shadowforge_pipeline/soft_voxel_v3/lovasz.py"),
    ("shadowforge_pipeline.soft_voxel_v3.metrics", "shadowforge_pipeline/soft_voxel_v3/metrics.py"),
    ("shadowforge_pipeline.soft_voxel_v3.protocol", "shadowforge_pipeline/soft_voxel_v3/protocol.py"),
    ("shadowforge_pipeline.soft_voxel_v3.provenance", "shadowforge_pipeline/soft_voxel_v3/provenance.py"),
    ("shadowforge_pipeline.soft_voxel_v3.thresholding", "shadowforge_pipeline/soft_voxel_v3/thresholding.py"),
    ("shadowforge_pipeline.soft_voxel_v4", "shadowforge_pipeline/soft_voxel_v4/__init__.py"),
    ("shadowforge_pipeline.soft_voxel_v4.config", "shadowforge_pipeline/soft_voxel_v4/config.py"),
    ("shadowforge_pipeline.soft_voxel_v4.diagnostics", "shadowforge_pipeline/soft_voxel_v4/diagnostics.py"),
    ("shadowforge_pipeline.soft_voxel_v4.gradients", "shadowforge_pipeline/soft_voxel_v4/gradients.py"),
    ("shadowforge_pipeline.soft_voxel_v4.losses", "shadowforge_pipeline/soft_voxel_v4/losses.py"),
    ("shadowforge_pipeline.soft_voxel_v4.provenance", "shadowforge_pipeline/soft_voxel_v4/provenance.py"),
    ("shadowforge_pipeline.soft_voxel_v4.trainer_step", "shadowforge_pipeline/soft_voxel_v4/trainer_step.py"),
    ("shadowforge_pipeline.soft_voxel_v6", "shadowforge_pipeline/soft_voxel_v6/__init__.py"),
    ("shadowforge_pipeline.soft_voxel_v6.core", "shadowforge_pipeline/soft_voxel_v6/core.py"),
    ("shadowforge_pipeline.soft_voxel_v6.provenance", "shadowforge_pipeline/soft_voxel_v6/provenance.py"),
    ("shadowforge_pipeline.soft_voxel_v6.trainer", "shadowforge_pipeline/soft_voxel_v6/trainer.py"),
)
SOURCE_FILES = tuple(relative for _, relative in TRAINING_MODULE_SOURCES)


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def _stack_hash(rows: Sequence[torch.Tensor]) -> str:
    return _tensor_hash(torch.stack(tuple(rows)).contiguous())


def _load_normalized(
    input_rows: Sequence[Mapping[str, object]], resolution: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    base: list[torch.Tensor] = []
    mirrors: list[torch.Tensor] = []
    for row in input_rows:
        snapshot = snapshot_regular_file(Path(row["path"]))
        if snapshot.sha256 != row["sha256"]:
            raise ValueError("formal-suite input hash changed")
        payload = Path(snapshot.path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != snapshot.sha256:
            raise ValueError("formal-suite input bytes changed while being captured")
        image = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if image is None:
            raise ValueError("formal-suite input is not a decodable grayscale image")
        base.append(torch.from_numpy(normalize_target(image, resolution, False) > 0))
        mirrors.append(
            torch.from_numpy(
                normalize_target(cv2.flip(image, 1), resolution, False) > 0
            )
        )
        revalidate_snapshot(snapshot)
    return base, mirrors


def _camera_identity() -> dict[str, object]:
    payload = fixed_camera_payload()
    return {
        "schema": payload["metadata"]["schema"],
        "metadata": payload["metadata"],
        "rotation_sha256": payload["rotation_sha256"],
        "translation_sha256": payload["translation_sha256"],
        "payload_sha256": payload["payload_sha256"],
    }


def _task_identity(
    task: object, *, suite_record: Mapping[str, object], resolution: int
) -> dict[str, object]:
    inputs = [
        {
            "axis": item.axis,
            "path": str(Path(item.resolved_path)),
            "source_path": str(Path(item.source_path)),
            "sha256": item.sha256,
        }
        for item in task.inputs
    ]
    base, mirrors = _load_normalized(inputs, resolution)
    labelled_base = tuple(zip(("x", "y", "z"), base))
    labelled_mirrors = tuple(zip(("x", "y", "z"), mirrors))
    mapped = map_suite_views(labelled_base, labelled_mirrors)
    logits = build_fixed_visual_hull_logits(
        labelled_base,
        resolution=resolution,
        allow_test_size=resolution != 256,
        dtype=torch.float32,
    )
    hull = logits == 2.0
    hard = torch.sigmoid(logits) >= 0.05
    if not torch.equal(hull, hard):
        raise ValueError("step-zero .05 occupancy differs from the fixed visual hull")
    cameras = _camera_identity()
    return {
        "task": task.case_id,
        "inputs": inputs,
        "targets": {
            "raw_flip_then_normalize": True,
            "base_order": ["x", "y", "z"],
            "mapped_order": ["y", "x", "z", "mirror_y", "mirror_x", "mirror_z"],
            "dtype": "bool",
            "shape": [6, resolution, resolution],
            "base_sha256": _stack_hash(base),
            "mirror_sha256": _stack_hash(mirrors),
            "mapped_six_sha256": _tensor_hash(mapped),
        },
        "cameras": cameras,
        "hull": {
            "axis_for_view": ["z", "x", "y"],
            "d4_forward_ops": [3, 0, 7],
            "dtype": "bool",
            "shape": [resolution, resolution, resolution],
            "sha256": _tensor_hash(hull[0, 0]),
        },
        "initialization": {
            "values": {"inside": 2.0, "outside": -3.0},
            "dtype": "float32",
            "shape": [1, 1, resolution, resolution, resolution],
            "logits_sha256": _tensor_hash(logits),
            "step_zero_hard_sha256": _tensor_hash(hard),
        },
    }


def _protocol() -> dict[str, object]:
    return {
        "name": "sadekar_tfs_fixed_six_view",
        "image_size": 256,
        "zdist": 1.7,
        "volume_extent": 1.7,
        "samples_per_ray": 150,
        "density_threshold": 0.05,
        "ste_temperature": 0.01,
        "readout": "projection >= 0.8 * detached_per_view_max inclusive",
        "mirrors": "raw-horizontal-flip-then-independent-normalize",
    }


def _objective() -> dict[str, object]:
    return {
        "base": "balanced_L2_plus_0.35_balanced_L1",
        "j": {
            "name": "delivery_lovasz_jaccard",
            "budget": 0.20,
            "score_scale": 0.05,
        },
        "h": {
            "name": "background_top_tail",
            "budget": 0.10,
            "fraction": 0.05,
            "margin": 0.01,
            "temperature": 0.02,
            "stable_row_major": True,
        },
        "j_budget": 0.20,
        "h_budget": 0.10,
        "no_reallocation": True,
    }


def _schedule() -> dict[str, object]:
    return {
        "updates": 3000,
        "optimizer": "Adam",
        "learning_rate": {
            "updates_1_1500": 0.007,
            "updates_1501_2700": "cosine_0.007_to_0.0007_inclusive",
            "updates_2701_3000": 0.0007,
        },
        "auxiliary": {
            "updates_1_600": 0.0,
            "updates_601_900": "linear_1_over_300_to_1",
            "updates_901_3000": 1.0,
        },
        "gradient_clip_norm": 1.0,
        "one_clip_one_step": True,
    }


def _decision_policy() -> dict[str, object]:
    return {
        "primary_contrast": "ten ordered task deltas; views/updates/voxels are not samples",
        "quality": {"full_mean_iou_min": 0.95, "full_min_task_iou_min": 0.94},
        "conditional_effect": {
            "ordered_mean_strictly_positive": True,
            "strict_positive_tasks_min": 8,
            "task_delta_floor": -0.005,
            "sign_flip_assignments": 1024,
            "sign_flip_pseudocount": False,
            "holm_family": ["J_IoU", "H_IoU"],
            "holm_adjusted_p_strict_max": 0.05,
        },
        "roles": {
            "mean_dice_full_minus_h_only_min": 0.0,
            "mean_fpr_full_minus_j_only_strictly_below": 0.0,
            "task_recall_decrement_floor": -0.005,
        },
        "engineering": {"full_ramp_raw_zero_fraction_max": 0.10},
        "soft_pass": "quality && recall_safety && engineering_activity",
        "both_confirmed": "J_IoU && J_Dice && H_IoU && H_FPR",
        "post_hard_native_axis_d4_any_hit": {"pass_iou": 0.97, "stretch_iou": 0.99},
    }


def _checkpoint_policy() -> dict[str, object]:
    return {
        "forced_steps": [0, 2],
        "interval": 200,
        "final_step": 3000,
        "diagnostic_steps": [2400, 2700, 3000],
        "step_2700_optimizer_checkpoint_required": False,
        "final_only_endpoint": True,
        "early_stop": False,
        "best_checkpoint": False,
        "legacy_state": False,
    }


def _cell_config(
    *,
    output_root: Path,
    task: Mapping[str, object],
    arm: str,
    formal_suite: Mapping[str, object],
    specification: Mapping[str, object],
    implementation: Mapping[str, object],
    environment: Mapping[str, object],
    resolution: int,
    device: str,
) -> dict[str, object]:
    run_path = output_root / "runs" / str(task["task"]) / arm / "seed_17"
    return {
        "schema": CONFIG_SCHEMA,
        "run": {
            "root": str(output_root),
            "run_id": f"{task['task']}/{arm}/seed_17",
            "path": str(run_path),
        },
        "task": task["task"],
        "arm": arm,
        "seed": 17,
        "device": device,
        "resolution": resolution,
        "export_resolution": resolution,
        "updates": 3000,
        "inputs": {
            "suite": formal_suite,
            "raw_xyz": copy.deepcopy(task["inputs"]),
            "runtime_order": ["y", "x", "z"],
        },
        "targets": copy.deepcopy(task["targets"]),
        "cameras": copy.deepcopy(task["cameras"]),
        "hull": copy.deepcopy(task["hull"]),
        "initialization": copy.deepcopy(task["initialization"]),
        "objective": {
            **_objective(),
            "flags": dict(PROFILE_FLAGS[arm]),
        },
        "schedule": _schedule(),
        "protocol": _protocol(),
        "checkpoint_policy": _checkpoint_policy(),
        "specification": specification,
        "implementation": implementation,
        "environment": environment,
    }


def _build_manifest(
    *, output_root: Path, formal_suite: Path, device: str, resolution: int
) -> dict[str, object]:
    root = reject_symlink_components(Path(output_root).resolve(), must_exist=False)
    suite_path = reject_symlink_components(Path(formal_suite).resolve())
    suite = load_v4_suite_identity(suite_path, "formal")
    if tuple(task.case_id for task in suite.tasks) != TASK_ORDER:
        raise ValueError("formal suite task order/count differs from connected_01..10")
    suite_record = {
        "schema": "pcraft_soft_loss_v4_suite_v1",
        "path": str(suite_path),
        "sha256": suite.sha256,
        "semantic_sha256": suite.semantic_sha256,
        "role": "formal",
    }
    spec = snapshot_regular_file(SPEC_PATH)
    specification = {
        "path": spec.path,
        "sha256": spec.sha256,
        "status": "FROZEN_FOR_IMPLEMENTATION",
    }
    implementation = implementation_bundle(PROJECT_ROOT, SOURCE_FILES)
    _validate_loaded_training_module_sources(implementation)
    environment = environment_identity(device)
    task_rows = [
        _task_identity(task, suite_record=suite_record, resolution=resolution)
        for task in suite.tasks
    ]
    cells: list[dict[str, object]] = []
    for task in task_rows:
        for arm in PROFILE_ORDER:
            config = _cell_config(
                output_root=root,
                task=task,
                arm=arm,
                formal_suite=suite_record,
                specification=specification,
                implementation=implementation,
                environment=environment,
                resolution=resolution,
                device=device,
            )
            cells.append(
                {
                    "task": task["task"],
                    "arm": arm,
                    "run_path": config["run"]["path"],
                    "config_sha256": canonical_hash(config),
                    "resolved_config": config,
                }
            )
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "status": "REGISTERED",
        "output_root": str(root),
        "formal_suite": suite_record,
        "specification": specification,
        "implementation": implementation,
        "protocol": _protocol(),
        "objective": _objective(),
        "schedule": _schedule(),
        "decision_policy": _decision_policy(),
        "tasks": task_rows,
        "profiles": list(PROFILE_ORDER),
        "cells": cells,
    }
    manifest["semantic_sha256"] = canonical_hash(manifest)
    revalidate_snapshot(spec)
    if sha256_file(suite_path) != suite.sha256:
        raise ValueError("formal suite changed while constructing the manifest")
    return json.loads(canonical_json_bytes(manifest))


def build_manifest(
    *, output_root: Path, formal_suite: Path, device: str = "cuda:0"
) -> dict[str, object]:
    return _build_manifest(
        output_root=output_root, formal_suite=formal_suite, device=device, resolution=256
    )


def _build_manifest_for_test(
    *, output_root: Path, formal_suite: Path, device: str, resolution: int
) -> dict[str, object]:
    return _build_manifest(
        output_root=output_root,
        formal_suite=formal_suite,
        device=device,
        resolution=resolution,
    )


def _validate_manifest(
    manifest: object, *, resolution: int
) -> dict[str, object]:
    if type(manifest) is not dict:
        raise ValueError("manifest must be an exact JSON object")
    exact = {
        "schema", "status", "output_root", "formal_suite", "specification",
        "implementation", "protocol", "objective", "schedule", "decision_policy",
        "tasks", "profiles", "cells", "semantic_sha256",
    }
    if set(manifest) != exact or manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "REGISTERED":
        raise ValueError("manifest has the wrong exact schema/status")
    observed_semantic = manifest["semantic_sha256"]
    scientific = {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    if observed_semantic != canonical_hash(scientific):
        raise ValueError("manifest semantic hash mismatch")
    expected = _build_manifest(
        output_root=Path(manifest["output_root"]),
        formal_suite=Path(manifest["formal_suite"]["path"]),
        device=manifest["cells"][0]["resolved_config"]["device"],
        resolution=resolution,
    )
    if canonical_json_bytes(manifest) != canonical_json_bytes(expected):
        raise ValueError("manifest differs from independent exact reconstruction")
    return expected


def validate_manifest(manifest: object) -> dict[str, object]:
    return _validate_manifest(manifest, resolution=256)


def _validate_manifest_for_test(manifest: object, *, resolution: int) -> dict[str, object]:
    return _validate_manifest(manifest, resolution=resolution)


def rebuild_resolved_config(claimed: Mapping[str, object]) -> dict[str, object]:
    if type(claimed) is not dict:
        raise ValueError("resolved config must be an exact mapping")
    try:
        manifest = build_manifest(
            output_root=Path(claimed["run"]["root"]),
            formal_suite=Path(claimed["inputs"]["suite"]["path"]),
            device=claimed["device"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("resolved config has the wrong exact schema") from error
    matches = [
        cell["resolved_config"]
        for cell in manifest["cells"]
        if cell["task"] == claimed.get("task") and cell["arm"] == claimed.get("arm")
    ]
    if len(matches) != 1:
        raise ValueError("resolved config does not identify one registered cell")
    return matches[0]


def _validate_loaded_training_module_sources(
    implementation: Mapping[str, object],
) -> None:
    files = implementation.get("files_sha256")
    if type(files) is not dict or set(files) != set(SOURCE_FILES):
        raise ValueError("implementation source bundle has the wrong exact training closure")
    for module_name, expected_relative in TRAINING_MODULE_SOURCES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        raw_path = getattr(module, "__file__", None)
        if type(raw_path) is not str:
            raise ValueError(f"loaded training module lacks source identity: {module_name}")
        path = Path(raw_path).resolve()
        try:
            observed_relative = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError as error:
            raise ValueError(
                f"loaded training module source is outside the authenticated project: {module_name}"
            ) from error
        if observed_relative != expected_relative or observed_relative not in files:
            raise ValueError(f"loaded training module has unbundled source: {module_name}")


def revalidate_static_identities(config: Mapping[str, object]) -> None:
    if type(config) is not dict:
        raise ValueError("resolved config must be a mapping")
    for label, record in (
        ("suite", config["inputs"]["suite"]),
        ("specification", config["specification"]),
    ):
        snapshot = snapshot_regular_file(Path(record["path"]))
        if snapshot.sha256 != record["sha256"]:
            raise ValueError(f"{label} hash changed")
    for item in config["inputs"]["raw_xyz"]:
        snapshot = snapshot_regular_file(Path(item["path"]))
        if snapshot.sha256 != item["sha256"]:
            raise ValueError("input hash changed")
    _validate_loaded_training_module_sources(config["implementation"])
    implementation = implementation_bundle(PROJECT_ROOT, SOURCE_FILES)
    if canonical_json_bytes(implementation) != canonical_json_bytes(config["implementation"]):
        raise ValueError("implementation source bundle changed")
    interpreter = snapshot_regular_file(Path(config["environment"]["interpreter"]["path"]))
    if interpreter.sha256 != config["environment"]["interpreter"]["sha256"]:
        raise ValueError("locked interpreter bytes changed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the exact P-CRAFT soft-loss-v6 native-256 10x4 ablation",
        allow_abbrev=False,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--formal-suite", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser


def _manifest_path(root: Path) -> Path:
    return root / "manifests/ablation10.json"


def _publish_manifest_for_test(manifest: Mapping[str, object]) -> Path:
    root = Path(manifest["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(root)
    atomic_json(path, manifest, create_once=True)
    return path


def _load_resume_manifest_for_test(
    root: Path, suite: Path, device: str, *, resolution: int
) -> dict[str, object]:
    canonical_root = reject_symlink_components(Path(root).resolve())
    path = _manifest_path(canonical_root)
    observed = decode_canonical_json(path, label="v6 ablation manifest")
    expected = _build_manifest(
        output_root=canonical_root,
        formal_suite=Path(suite),
        device=device,
        resolution=resolution,
    )
    if path.read_bytes() != canonical_json_bytes(expected):
        raise ValueError("existing manifest bytes are not the exact authenticated manifest")
    return observed


def _inspect_cell(cell: Mapping[str, object]) -> tuple[str, int] | None:
    config = cell["resolved_config"]
    run_dir = Path(cell["run_path"])
    if not run_dir.exists():
        return None
    reject_symlink_components(run_dir)
    recover_interrupted_checkpoint_publication(config)
    status = decode_canonical_json(run_dir / "STATUS.json", label="cell STATUS")
    state = status.get("state")
    if state == "completed":
        result = validate_completed_run(config)
        return ("completed", result.completed_updates)
    if state == "running":
        result = validate_resumable_run(config)
        return ("running", result.completed_updates)
    raise ValueError("failed/corrupt cell is not resumable")


def _stable_authenticated_bytes(path: Path, expected_sha256: str, *, label: str) -> bytes:
    if type(expected_sha256) is not str or len(expected_sha256) != 64:
        raise ValueError(f"{label} digest is invalid")
    snapshot = snapshot_regular_file(path)
    if snapshot.sha256 != expected_sha256:
        raise ValueError(f"{label} digest changed after authentication")
    payload = Path(snapshot.path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{label} bytes changed while being loaded")
    revalidate_snapshot(snapshot)
    return payload


def _engineering_checkpoint_identity(
    config: Mapping[str, object], status: Mapping[str, object]
) -> tuple[Path, str]:
    state = status.get("state")
    if state == "running":
        authenticated = validate_resumable_run(
            config,
            expected_completed_updates=(
                200 if status.get("completed_updates") == 200 else None
            ),
        )
        completed = authenticated.completed_updates
        if type(completed) is not int or completed < 200 or completed % 200 != 0:
            raise ValueError("engineering gate requires an authenticated step-200 chain")
        run_dir = Path(authenticated.run_dir)
        digest = authenticated.checkpoint_sha256
        while completed > 200:
            raw = _stable_authenticated_bytes(
                run_dir / "checkpoints" / f"step_{completed:04d}.pt",
                digest,
                label="engineering checkpoint chain",
            )
            try:
                checkpoint = torch.load(
                    io.BytesIO(raw), map_location="cpu", weights_only=False
                )
            except Exception as error:
                raise ValueError("engineering checkpoint chain is malformed") from error
            if type(checkpoint) is not dict:
                raise ValueError("engineering checkpoint chain is malformed")
            predecessor = checkpoint.get("predecessor_sha256")
            if (
                type(predecessor) is not str
                or len(predecessor) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in predecessor
                )
            ):
                raise ValueError("engineering checkpoint predecessor is invalid")
            digest = predecessor
            completed -= 200
        return run_dir, digest
    if state != "completed":
        raise ValueError("engineering gate requires running or completed evidence")
    authenticated = validate_completed_run(config)
    run_dir = Path(authenticated.run_dir)
    manifest_path = run_dir / "artifacts/artifacts.json"
    manifest_snapshot = snapshot_regular_file(manifest_path)
    if manifest_snapshot.sha256 != authenticated.artifacts_sha256:
        raise ValueError("completed artifact manifest changed after authentication")
    manifest = decode_canonical_json(manifest_path, label="completed artifact manifest")
    revalidate_snapshot(manifest_snapshot)
    if (
        set(manifest) != {"schema", "config_sha256", "files_sha256"}
        or manifest.get("schema") != "pcraft_soft_loss_v6_artifacts_v1"
        or manifest.get("config_sha256") != authenticated.config_sha256
        or type(manifest.get("files_sha256")) is not dict
    ):
        raise ValueError("completed artifact manifest is not authenticated")
    digest = manifest["files_sha256"].get("checkpoints/step_0200.pt")
    if type(digest) is not str or len(digest) != 64:
        raise ValueError("completed chain lacks authenticated step 200")
    return run_dir, digest


def validate_engineering_gate(config: Mapping[str, object]) -> None:
    run_dir = Path(config["run"]["path"])
    status = decode_canonical_json(run_dir / "STATUS.json", label="engineering STATUS")
    authenticated_run_dir, digest = _engineering_checkpoint_identity(config, status)
    raw_checkpoint = _stable_authenticated_bytes(
        authenticated_run_dir / "checkpoints/step_0200.pt",
        digest,
        label="engineering checkpoint",
    )
    try:
        checkpoint = torch.load(
            io.BytesIO(raw_checkpoint), map_location="cpu", weights_only=False
        )
    except Exception as error:
        raise ValueError("engineering checkpoint payload is malformed") from error
    times = checkpoint["step_times"]
    if not times or not all(type(value) is float and math.isfinite(value) and value > 0.0 for value in times):
        raise ValueError("engineering gate requires finite positive step timings")
    if not str(config["device"]).startswith("cuda"):
        raise ValueError("engineering gate requires successful CUDA execution")
    peak = max(int(row["peak_cuda_allocated_bytes"]) for row in checkpoint["history"])
    device = torch.device(config["device"])
    total = int(torch.cuda.get_device_properties(device).total_memory)
    if peak <= 0 or peak >= total:
        raise ValueError("engineering gate peak memory is invalid")


def _advance(
    cell: Mapping[str, object], *, target: int, fresh_mode: bool
) -> ResumableRun | CompletedRun | None:
    config = cell["resolved_config"]
    revalidate_static_identities(config)
    state = None if fresh_mode else _inspect_cell(cell)
    if state is not None and state[0] == "completed":
        validate_completed_run(config)
        revalidate_static_identities(config)
        return None
    final_running = (
        state is not None
        and state[0] == "running"
        and state[1] == target
        and target == int(config["updates"])
    )
    if state is not None and state[1] >= target and not final_running:
        validate_resumable_run(config, expected_completed_updates=state[1])
        revalidate_static_identities(config)
        return None
    result = train_cell(config, resume=state is not None, max_updates=target)
    revalidate_static_identities(config)
    if target == int(config["updates"]):
        validate_completed_run(config)
    else:
        validate_resumable_run(config, expected_completed_updates=target)
    revalidate_static_identities(config)
    return result


def _execute_manifest(manifest: Mapping[str, object], *, fresh: bool) -> None:
    cells = manifest["cells"]
    for cell in cells:
        _advance(cell, target=2, fresh_mode=fresh)
    gates = [
        next(cell for cell in cells if cell["task"] == "connected_01" and cell["arm"] == arm)
        for arm in ("base", "full")
    ]
    for cell in gates:
        _advance(cell, target=200, fresh_mode=False)
        validate_engineering_gate(cell["resolved_config"])
    for cell in cells:
        _advance(cell, target=3000, fresh_mode=False)


def _execute_manifest_for_test(manifest: Mapping[str, object], *, fresh: bool) -> None:
    _execute_manifest(manifest, fresh=fresh)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output_root).resolve()
    if args.dry_run:
        if root.exists():
            raise FileExistsError("dry-run requires an absent output root")
        manifest = build_manifest(
            output_root=root, formal_suite=args.formal_suite, device=args.device
        )
        validate_manifest(manifest)
        for cell in manifest["cells"]:
            print(
                f"{cell['task']}\t{cell['arm']}\t{cell['run_path']}\t{cell['config_sha256']}"
            )
        if root.exists():
            raise RuntimeError("dry-run created the output root")
        return 0
    if root != REAL_OUTPUT_ROOT:
        raise ValueError(f"real execution output root must be exactly {REAL_OUTPUT_ROOT}")
    if args.resume:
        manifest = _load_resume_manifest_for_test(
            root, args.formal_suite, args.device, resolution=256
        )
        validate_manifest(manifest)
        _execute_manifest(manifest, fresh=False)
        return 0
    if root.exists():
        raise FileExistsError("fresh normal execution requires an absent output root")
    manifest = build_manifest(
        output_root=root, formal_suite=args.formal_suite, device=args.device
    )
    validate_manifest(manifest)
    _publish_manifest_for_test(manifest)
    _execute_manifest(manifest, fresh=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
