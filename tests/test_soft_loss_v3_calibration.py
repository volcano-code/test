"""Fail-closed contracts for gradient-only soft-loss-v3 calibration."""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

import experiments.soft_loss_v3_registry as registry_module
import shadowforge_pipeline.soft_voxel_v3.calibration as calibration_module
import shadowforge_pipeline.soft_voxel_v3.provenance as provenance_module
from shadowforge_pipeline.soft_voxel_v3.calibration import (
    CalibrationMeasurement,
    build_calibration_parser,
    calibration_main,
    derive_calibrated_weights,
    load_calibration_measurements,
    validate_frozen_weight_config,
    write_frozen_weight_config,
)
from shadowforge_pipeline.soft_voxel_v3.provenance import (
    canonical_hash,
    canonical_json_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPDATES = (1000, 1300, 1600)
AXES = ("x", "y", "z")
DEVELOPMENT_IDS = ("arcane_casters", "blade_heroes", "stage_dancers")
FORMAL_IDS = tuple(f"connected_{index:02d}" for index in range(1, 11))
EXECUTED_SOURCE_NAMES = (
    "experiments/soft_loss_v3_registry.py",
    "experiments/bind_soft_loss_v3_suites.py",
    "experiments/run_soft_loss_v3.py",
    "experiments/calibrate_soft_loss_v3.py",
    "shadowforge_pipeline/soft_voxel_v3/calibration.py",
    "shadowforge_pipeline/soft_voxel_v3/provenance.py",
    "shadowforge_pipeline/soft_voxel_v3/trainer.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _write_source_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    formal_quality_audit: bool = False,
) -> dict[str, Path]:
    project = tmp_path / "project"
    development = project / "development"
    formal = project / "formal"
    development.mkdir(parents=True)
    formal.mkdir(parents=True)

    development_tasks: list[dict[str, object]] = []
    for task_index, case_id in enumerate(DEVELOPMENT_IDS):
        views: list[str] = []
        task_root = development / case_id
        task_root.mkdir()
        for view_index in range(3):
            target = task_root / f"view_{view_index}.png"
            target.write_bytes(b"development-input\0" + bytes([task_index, view_index]))
            views.append(str(target.resolve()))
        development_tasks.append({"task": case_id, "views": views})
    _write_source_json(
        development / "MANIFEST.json",
        {
            "protocol": "fixed-3-view TFS, mr=1, 256^3",
            "foreground": "white-on-black",
            "tasks": development_tasks,
        },
    )

    formal_inputs = formal / "inputs"
    formal_inputs.mkdir()
    cases: list[dict[str, object]] = []
    for case_index, case_id in enumerate(FORMAL_IDS):
        targets: list[str] = []
        for axis_index, axis in enumerate(AXES):
            target = formal_inputs / f"{case_id}_view_{axis}.png"
            target.write_bytes(b"formal-input\0" + bytes([case_index, axis_index]))
            targets.append(str(target.resolve()))
        case: dict[str, object] = {"case_id": case_id, "target_paths_xyz": targets}
        if formal_quality_audit:
            case.update(
                {
                    "sadekar_tfs_targets": [targets[1], targets[0], targets[2]],
                    "input_connectivity_audit": [{"quality": 1.0}],
                    "visual_hull_audit": {
                        "mean_iou": 0.99,
                        "view_ious": [0.98, 0.99, 1.0],
                        "boundary_audit": {"mean_boundary_f1_2px": 0.97},
                    },
                }
            )
        cases.append(case)
    formal_manifest = formal / "benchmark_manifest.json"
    formal_payload: dict[str, object] = {
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
    }
    if formal_quality_audit:
        formal_payload.update(
            {
                "purpose": "real-shape approved formal binding fixture",
                "selection_policy": "input-only",
                "candidate_pool": {"audited_masks": 30},
            }
        )
    _write_source_json(formal_manifest, formal_payload)
    spec = project / "revised-design.md"
    spec.write_text("# synthetic frozen revised design\n", encoding="utf-8")
    output = tmp_path / "registry"
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", project)
    assert registry_module.binding_main(
        [
            "--output-root",
            str(output),
            "--development-root",
            str(development),
            "--formal-benchmark",
            str(formal_manifest),
            "--spec",
            str(spec),
        ]
    ) == 0
    return {
        "root": output,
        "development": output / "manifests" / "development_suite.json",
        "formal": output / "manifests" / "formal_suite.json",
        "master": output / "experiment_manifest.json",
        "spec": spec,
    }


def _make_source_identities(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object], list[Path]]:
    executed_root = project / "executed-sources"
    authoritative = executed_root / "shadowforge_pipeline" / "soft_voxel_v3"
    source_payloads = {
        "__init__.py": b"# synthetic authoritative package\n",
        "trainer.py": b"# synthetic executed trainer\n",
        "calibration.py": b"# synthetic calibration consumer\n",
        "provenance.py": b"# synthetic provenance primitives\n",
        "metrics.py": b"# synthetic unimported metrics implementation\n",
        "nested/diagnostics.py": b"# synthetic transitive local dependency\n",
    }
    for relative, payload in source_payloads.items():
        path = authoritative / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    monkeypatch.setattr(provenance_module, "AUTHORITATIVE_SOURCE_ROOT", authoritative)
    implementation = provenance_module.implementation_source_manifest()

    executed_files: dict[str, object] = {}
    source_paths = [authoritative / relative for relative in source_payloads]
    for index, logical_name in enumerate(EXECUTED_SOURCE_NAMES):
        package_prefix = "shadowforge_pipeline/soft_voxel_v3/"
        relative = logical_name.removeprefix(package_prefix)
        if logical_name.startswith(package_prefix) and relative in source_payloads:
            path = authoritative / relative
        else:
            path = executed_root / logical_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                f"# executed source {index}: {logical_name}\n".encode("utf-8")
            )
        executed_files[logical_name] = {
            "path": str(path.resolve()),
            "sha256": _sha(path),
        }
        source_paths.append(path)
    executed = {
        "algorithm": "sha256",
        "files": executed_files,
        "bundle_sha256": canonical_hash(executed_files),
    }
    return implementation, executed, source_paths


def _gradient_sample(
    update: int,
    values: tuple[float, float, float, float],
) -> dict[str, object]:
    g0, g_dic, g004, g006 = values
    endpoints = {"0.04": 0.1, "0.06": 0.2}
    endpoint_flags = {"0.04": False, "0.06": False}
    return {
        "completed_updates": update,
        "base_norm": g0,
        "dic_norm": g_dic,
        "dts_norms": {"0.04": g004, "0.06": g006},
        "base_dic_cosine": 0.1,
        "base_dic_cosine_invalid": False,
        "base_dts_cosines": endpoints,
        "base_dts_cosine_invalid": endpoint_flags,
        "dic_dts_cosines": endpoints,
        "dic_dts_cosine_invalid": endpoint_flags,
        "dts_pair_cosines": {"0.04": {"0.06": 0.1}},
        "dts_pair_cosine_invalid": {"0.04": {"0.06": False}},
        "dic_dts_support_ious": {"0.04": 0.25, "0.06": 0.5},
        "dic_dts_support_iou_invalid": endpoint_flags,
        "weighted_dic_base_ratio": 0.0,
        "weighted_dic_base_ratio_invalid": False,
        "weighted_dts_base_ratios": {"0.04": 0.0, "0.06": 0.0},
        "weighted_dts_base_ratio_invalid": endpoint_flags,
        "selected_dts_tau": 0.04,
        "combined_aux_base_ratio": 0.0,
        "combined_aux_base_ratio_invalid": False,
        "base_combined_aux_cosine": None,
        "base_combined_aux_cosine_invalid": True,
    }


def _make_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    formal_quality_audit: bool = False,
) -> dict[str, object]:
    registry = _make_registry(
        tmp_path,
        monkeypatch,
        formal_quality_audit=formal_quality_audit,
    )
    implementation, executed_sources, source_paths = _make_source_identities(
        registry["root"].parent / "project",
        monkeypatch,
    )
    development_payload = json.loads(registry["development"].read_bytes())
    inputs = development_payload["tasks"][0]["inputs"]
    interpreter = tmp_path / "locked-python"
    interpreter.write_bytes(b"synthetic-locked-interpreter")
    config: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_config_v1",
        "device": "cuda:0",
        "steps": 2000,
        "learning_rate": 0.007,
        "resolution": 128,
        "export_resolution": 128,
        "profile": "l12_bal",
        "lambda_dic": 0.0,
        "lambda_dts": 0.0,
        "seed": 17,
        "shadow_inputs_xyz": [
            {"path": item["resolved_path"], "sha256": item["sha256"]}
            for item in inputs
        ],
        "suite_manifest": {
            "path": str(registry["development"].resolve()),
            "sha256": _sha(registry["development"]),
        },
        "specification": {
            "path": str(registry["spec"].resolve()),
            "sha256": _sha(registry["spec"]),
        },
        "weight_config": None,
        "calibration_probe": True,
        "protocol": {
            "mirror_mode": 1,
            "tsf_mode": 1,
            "tsf4_mode": 0,
            "camera_mode": "ortho",
            "image_size": 256,
            "control_res": 0,
            "zdist": 1.7,
            "volume_extent": 1.7,
            "samples_per_ray": 150,
        },
        "density_thresholds": [0.04, 0.05, 0.06],
        "final_step_export_only": True,
        "soft_only": True,
        "implementation": implementation,
        "interpreter": {
            "path": str(interpreter.resolve()),
            "sha256": _sha(interpreter),
            "python": "3.synthetic",
        },
        "determinism": {
            "fixed_seed": True,
            "mode": "best_effort",
            "bitwise_reproducible": False,
            "reason": "synthetic best-effort CUDA fixture",
            "environment": {},
        },
    }
    config_hash = canonical_hash(config)
    run = tmp_path / "calibration-run"
    run.mkdir()
    run_manifest = {
        "schema": "pcraft_soft_loss_v3_run_v1",
        "config_hash": config_hash,
        "config": config,
        "profile": "l12_bal",
        "protocol": {
            "name": "sadekar_tfs_fixed_six_view",
            "camera_azimuth_deg": [0.0, 90.0, 0.0, 180.0, 270.0, 0.0],
            "camera_elevation_deg": [0.0, 0.0, 90.0, 0.0, 0.0, 270.0],
            "projection_binary_rule": "protocol.binary_mask",
            "density_thresholds": [0.04, 0.05, 0.06],
        },
        "scope": {
            "soft_only": True,
            "final_step_export_only": True,
            "downstream_geometry": False,
        },
    }
    run_manifest_path = run / "run_manifest.json"
    _write_canonical(run_manifest_path, run_manifest)

    gradients = (
        (10.0, 1.0, 1.0, 100.0),
        (20.0, 4.0, 2.0, 200.0),
        (30.0, 10.0, 3.0, 300.0),
    )
    calibration_records: list[dict[str, object]] = []
    checkpoint_records: list[dict[str, object]] = []
    artifacts: dict[str, str] = {}
    for update, values in zip(UPDATES, gradients):
        name = f"calibration_checkpoint_step_{update:08d}.pt"
        checkpoint = run / name
        checkpoint.write_bytes(b"opaque-checkpoint\0" + str(update).encode("ascii"))
        digest = _sha(checkpoint)
        calibration_records.append(
            {
                "completed_updates": update,
                "g0": values[0],
                "g_dic": values[1],
                "g_dts_004": values[2],
                "g_dts_006": values[3],
                "checkpoint": name,
                "checkpoint_sha256": digest,
            }
        )
        checkpoint_records.append(
            {
                "completed_updates": update,
                "path": name,
                "sha256": digest,
            }
        )
        artifacts[name] = digest
    gradient_audit = {
        "schema": "pcraft_soft_loss_v3_gradient_audit_v1",
        "config_hash": config_hash,
        "samples": [
            _gradient_sample(update, values)
            for update, values in zip(UPDATES, gradients)
        ],
        "calibration_records": calibration_records,
        "active_invalid_flags": [
            {"completed_updates": update, "flags": []}
            for update in UPDATES
        ],
    }
    gradient_audit_path = run / "gradient_audit.json"
    _write_canonical(gradient_audit_path, gradient_audit)
    artifacts["run_manifest.json"] = _sha(run_manifest_path)
    artifacts["gradient_audit.json"] = _sha(gradient_audit_path)
    metrics_path = run / "metrics.json"
    metrics_path.write_bytes(b'{"mean_iou":0.999}')
    resource_metrics_path = run / "resource_metrics.json"
    resource_metrics_path.write_bytes(b'{"elapsed_seconds":1.0}')
    artifacts["metrics.json"] = _sha(metrics_path)
    artifacts["resource_metrics.json"] = _sha(resource_metrics_path)
    artifact_manifest = {
        "schema": "pcraft_soft_loss_v3_artifacts_v1",
        "config_hash": config_hash,
        "artifacts_sha256": artifacts,
    }
    artifact_manifest_path = run / "artifact_manifest.json"
    _write_canonical(artifact_manifest_path, artifact_manifest)
    status = {
        "schema": "pcraft_soft_loss_v3_status_v1",
        "state": "completed",
        "config_hash": config_hash,
        "completed_updates": 2000,
        "artifact_manifest_sha256": _sha(artifact_manifest_path),
    }
    status_path = run / "STATUS.json"
    _write_canonical(status_path, status)

    master_payload = json.loads(registry["master"].read_bytes())
    stage: dict[str, object] = {
        "schema": "pcraft_soft_loss_v3_calibration_stage_v1",
        "status": "APPROVED_FROZEN",
        "calibration_run": {
            "path": str(run.resolve()),
            "task": "arcane_casters",
            "profile": "l12_bal",
            "resolution": 128,
            "export_resolution": 128,
            "steps": 2000,
            "seed": 17,
            "soft_only": True,
            "calibration_probe": True,
            "config_hash": config_hash,
            "run_manifest_sha256": _sha(run_manifest_path),
            "gradient_audit_sha256": _sha(gradient_audit_path),
            "artifact_manifest_sha256": _sha(artifact_manifest_path),
            "status_sha256": _sha(status_path),
        },
        "checkpoints": checkpoint_records,
        "master_manifest": {
            "path": str(registry["master"].resolve()),
            "sha256": _sha(registry["master"]),
            "semantic_sha256": master_payload["semantic_sha256"],
        },
        "development_suite": {
            "path": str(registry["development"].resolve()),
            "sha256": _sha(registry["development"]),
            "semantic_sha256": development_payload["semantic_sha256"],
        },
        "specification": config["specification"],
        "inputs_xyz": inputs,
        "executed_source_bundle": executed_sources,
        "publication": {
            "policy": "create_once_after_completed_calibrate128",
            "completed_status_sha256": _sha(status_path),
        },
    }
    stage["semantic_sha256"] = canonical_hash(stage)
    stage_path = tmp_path / "calibration_stage.json"
    _write_canonical(stage_path, stage)
    return {
        "run": run,
        "run_manifest": run_manifest_path,
        "gradient_audit": gradient_audit_path,
        "artifact_manifest": artifact_manifest_path,
        "metrics": metrics_path,
        "resource_metrics": resource_metrics_path,
        "status": status_path,
        "stage": stage_path,
        "master": registry["master"],
        "development": registry["development"],
        "formal": registry["formal"],
        "spec": registry["spec"],
        "interpreter": interpreter,
        "inputs": [Path(item["resolved_path"]) for item in inputs],
        "source_paths": source_paths,
        "executed_source_bundle": executed_sources,
        "output": tmp_path / "weights.json",
    }


def _cli_args(context: dict[str, object]) -> list[str]:
    return [
        "--calibration-run",
        str(context["run"]),
        "--stage-manifest",
        str(context["stage"]),
        "--master-manifest",
        str(context["master"]),
        "--development-suite",
        str(context["development"]),
        "--output",
        str(context["output"]),
    ]


def _measurement_set(tmp_path: Path) -> tuple[CalibrationMeasurement, ...]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    values = (
        (10.0, 1.0, 1.0, 100.0),
        (20.0, 4.0, 2.0, 200.0),
        (30.0, 10.0, 3.0, 300.0),
    )
    result = []
    for update, gradients in zip(UPDATES, values):
        checkpoint = tmp_path / f"calibration_checkpoint_step_{update:08d}.pt"
        checkpoint.write_bytes(f"checkpoint-{update}".encode("ascii"))
        result.append(
            CalibrationMeasurement(
                completed_updates=update,
                checkpoint=checkpoint.resolve(),
                checkpoint_sha256=_sha(checkpoint),
                g0=gradients[0],
                g_dic=gradients[1],
                g_dts_004=gradients[2],
                g_dts_006=gradients[3],
            )
        )
    return tuple(result)


def _rewrite_stage(path: Path, mutate: object) -> None:
    payload = json.loads(path.read_bytes())
    payload.pop("semantic_sha256")
    mutate(payload)
    payload["semantic_sha256"] = canonical_hash(payload)
    _write_canonical(path, payload)


def _rewrite_output(path: Path, mutate: object, *, rehash: bool = True) -> None:
    payload = json.loads(path.read_bytes())
    mutate(payload)
    if rehash:
        payload["semantic_sha256"] = canonical_hash(
            {key: value for key, value in payload.items() if key != "semantic_sha256"}
        )
    _write_canonical(path, payload)


def _synchronize_completed_run_hashes(context: dict[str, object]) -> None:
    artifact = json.loads(context["artifact_manifest"].read_bytes())
    artifact["artifacts_sha256"]["gradient_audit.json"] = _sha(context["gradient_audit"])
    _write_canonical(context["artifact_manifest"], artifact)
    status = json.loads(context["status"].read_bytes())
    status["artifact_manifest_sha256"] = _sha(context["artifact_manifest"])
    _write_canonical(context["status"], status)

    def synchronize(stage: dict[str, object]) -> None:
        stage["calibration_run"]["gradient_audit_sha256"] = _sha(
            context["gradient_audit"]
        )
        stage["calibration_run"]["artifact_manifest_sha256"] = _sha(
            context["artifact_manifest"]
        )
        stage["calibration_run"]["status_sha256"] = _sha(context["status"])
        stage["publication"]["completed_status_sha256"] = _sha(context["status"])

    _rewrite_stage(context["stage"], synchronize)


def _rewrite_run_identity(
    context: dict[str, object],
    *,
    mutate_config: object | None = None,
    mutate_manifest: object | None = None,
) -> None:
    run_manifest = json.loads(context["run_manifest"].read_bytes())
    if mutate_config is not None:
        mutate_config(run_manifest["config"])
    if mutate_manifest is not None:
        mutate_manifest(run_manifest)
    config_hash = canonical_hash(run_manifest["config"])
    run_manifest["config_hash"] = config_hash
    _write_canonical(context["run_manifest"], run_manifest)

    audit = json.loads(context["gradient_audit"].read_bytes())
    audit["config_hash"] = config_hash
    _write_canonical(context["gradient_audit"], audit)
    artifact = json.loads(context["artifact_manifest"].read_bytes())
    artifact["config_hash"] = config_hash
    artifact["artifacts_sha256"]["run_manifest.json"] = _sha(context["run_manifest"])
    artifact["artifacts_sha256"]["gradient_audit.json"] = _sha(
        context["gradient_audit"]
    )
    _write_canonical(context["artifact_manifest"], artifact)
    status = json.loads(context["status"].read_bytes())
    status["config_hash"] = config_hash
    status["artifact_manifest_sha256"] = _sha(context["artifact_manifest"])
    _write_canonical(context["status"], status)

    def synchronize(stage: dict[str, object]) -> None:
        run = stage["calibration_run"]
        run["config_hash"] = config_hash
        run["run_manifest_sha256"] = _sha(context["run_manifest"])
        run["gradient_audit_sha256"] = _sha(context["gradient_audit"])
        run["artifact_manifest_sha256"] = _sha(context["artifact_manifest"])
        run["status_sha256"] = _sha(context["status"])
        stage["publication"]["completed_status_sha256"] = _sha(context["status"])

    _rewrite_stage(context["stage"], synchronize)


def test_exact_median_formulas_use_all_six_dts_endpoints(tmp_path: Path) -> None:
    """Break caught: a mean, per-step DTS summary, or one endpoint replaces the medians."""

    weights = derive_calibrated_weights(_measurement_set(tmp_path))

    assert weights == {
        "lambda_dic": 0.15 * 20.0 / (4.0 + 1e-12),
        "lambda_dts": 0.15 * 20.0 / (51.5 + 1e-12),
    }


def test_measurement_is_frozen_and_requires_exact_order_count_and_types(tmp_path: Path) -> None:
    """Break caught: callers mutate records or calibration silently sorts/coerces identity."""

    measurements = _measurement_set(tmp_path)
    with pytest.raises(FrozenInstanceError):
        measurements[0].g0 = 99.0
    for bad in (measurements[:2], tuple(reversed(measurements))):
        with pytest.raises(ValueError, match="three|order|1000|1300|1600"):
            derive_calibrated_weights(bad)
    bad_update = list(measurements)
    bad_update[0] = CalibrationMeasurement(
        True,
        bad_update[0].checkpoint,
        bad_update[0].checkpoint_sha256,
        10.0,
        1.0,
        1.0,
        100.0,
    )
    with pytest.raises(ValueError, match="integer|update|type"):
        derive_calibrated_weights(bad_update)
    bad_float = list(measurements)
    bad_float[0] = CalibrationMeasurement(
        1000,
        bad_float[0].checkpoint,
        bad_float[0].checkpoint_sha256,
        10,
        1.0,
        1.0,
        100.0,
    )
    with pytest.raises(ValueError, match="float|type"):
        derive_calibrated_weights(bad_float)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("g0", True, "float|type"),
        ("g0", -1.0, "non-negative"),
        ("g0", float("nan"), "finite"),
        ("g_dic", float("inf"), "finite"),
        ("g_dts_004", -1e-12, "non-negative"),
    ],
)
def test_gradient_values_reject_bool_negative_and_nonfinite(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """Break caught: Python numeric equality admits invalid scientific gradients."""

    measurements = list(_measurement_set(tmp_path))
    payload = measurements[0].__dict__.copy()
    payload[field] = value
    measurements[0] = CalibrationMeasurement(**payload)
    with pytest.raises(ValueError, match=message):
        derive_calibrated_weights(measurements)


@pytest.mark.parametrize("median", ["baseline", "dic", "dts"])
def test_zero_median_fails_instead_of_fallback(tmp_path: Path, median: str) -> None:
    """Break caught: a zero calibration denominator or baseline gets clipped/defaulted."""

    measurements = list(_measurement_set(tmp_path))
    for index, measurement in enumerate(measurements):
        payload = measurement.__dict__.copy()
        if median == "baseline":
            payload["g0"] = 0.0
        elif median == "dic":
            payload["g_dic"] = 0.0
        else:
            payload["g_dts_004"] = 0.0
            payload["g_dts_006"] = 0.0
        measurements[index] = CalibrationMeasurement(**payload)
    with pytest.raises(ValueError, match="median|positive|zero"):
        derive_calibrated_weights(measurements)


def test_tiny_positive_medians_remain_unclipped(tmp_path: Path) -> None:
    """Break caught: small but valid positive gradients are replaced by a manual floor."""

    measurements = []
    for measurement in _measurement_set(tmp_path):
        payload = measurement.__dict__.copy()
        payload.update(g0=1e-300, g_dic=1e-300, g_dts_004=1e-300, g_dts_006=1e-300)
        measurements.append(CalibrationMeasurement(**payload))
    weights = derive_calibrated_weights(measurements)
    expected = 0.15 * 1e-300 / (1e-300 + 1e-12)
    assert weights == {"lambda_dic": expected, "lambda_dts": expected}
    assert all(value > 0.0 for value in weights.values())


def test_checkpoint_name_hash_current_bytes_and_duplicates_are_enforced(tmp_path: Path) -> None:
    """Break caught: derivation trusts a stored path/hash or reuses one checkpoint."""

    measurements = list(_measurement_set(tmp_path))
    wrong_name = tmp_path / "checkpoint.pt"
    wrong_name.write_bytes(b"wrong-name")
    payload = measurements[0].__dict__.copy()
    payload.update(checkpoint=wrong_name.resolve(), checkpoint_sha256=_sha(wrong_name))
    with pytest.raises(ValueError, match="name|checkpoint"):
        derive_calibrated_weights([CalibrationMeasurement(**payload), *measurements[1:]])

    measurements = list(_measurement_set(tmp_path / "hash"))
    measurements[0].checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash|changed|sha256"):
        derive_calibrated_weights(measurements)

    measurements = list(_measurement_set(tmp_path / "duplicate"))
    payload = measurements[1].__dict__.copy()
    payload.update(
        checkpoint=measurements[0].checkpoint,
        checkpoint_sha256=measurements[0].checkpoint_sha256,
    )
    with pytest.raises(ValueError, match="duplicate"):
        derive_calibrated_weights(
            [measurements[0], CalibrationMeasurement(**payload), measurements[2]]
        )


def test_loader_parses_exact_task8_audit_and_resolves_contained_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the Task 8 seven-field audit record is guessed or left relative."""

    context = _make_context(tmp_path, monkeypatch)
    loaded = load_calibration_measurements(context["gradient_audit"])
    assert [item.completed_updates for item in loaded] == list(UPDATES)
    assert all(item.checkpoint.is_absolute() for item in loaded)
    assert [item.g_dts_006 for item in loaded] == [100.0, 200.0, 300.0]


@pytest.mark.parametrize("mutation", ["extra", "missing", "order", "coercible", "escape", "symlink"])
def test_loader_rejects_malformed_audit_and_unsafe_checkpoint_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: malformed JSON or a checkpoint escape/link becomes a measurement."""

    context = _make_context(tmp_path, monkeypatch)
    audit_path = context["gradient_audit"]
    payload = json.loads(audit_path.read_bytes())
    if mutation == "extra":
        payload["calibration_records"][0]["extra"] = 1
    elif mutation == "missing":
        del payload["calibration_records"][0]["g_dic"]
    elif mutation == "order":
        payload["calibration_records"].reverse()
    elif mutation == "coercible":
        payload["calibration_records"][0]["completed_updates"] = 1000.0
    elif mutation == "escape":
        payload["calibration_records"][0]["checkpoint"] = "../outside.pt"
    else:
        name = payload["calibration_records"][0]["checkpoint"]
        target = context["run"] / name
        outside = tmp_path / "outside.pt"
        outside.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(outside)
    _write_canonical(audit_path, payload)
    with pytest.raises(ValueError, match="schema|order|integer|checkpoint|contain|symlink|canonical"):
        load_calibration_measurements(audit_path)


@pytest.mark.parametrize("mutation", ["mismatch", "missing", "order", "extra"])
def test_task8_samples_are_exact_and_cross_checked_with_calibration_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: seven-field records are trusted without their Task 8 probe samples."""

    context = _make_context(tmp_path, monkeypatch)
    audit = json.loads(context["gradient_audit"].read_bytes())
    if mutation == "mismatch":
        audit["samples"][0]["base_norm"] = 11.0
    elif mutation == "missing":
        audit["samples"].pop()
        audit["active_invalid_flags"].pop()
    elif mutation == "order":
        audit["samples"].reverse()
        audit["active_invalid_flags"].reverse()
    else:
        audit["samples"][0]["unregistered"] = 1
    _write_canonical(context["gradient_audit"], audit)
    _synchronize_completed_run_hashes(context)
    with pytest.raises(ValueError, match="sample|Task 8|order|three|schema"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


@pytest.mark.parametrize("source_kind", ["authoritative", "executed"])
def test_current_source_bytes_are_recomputed_before_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
) -> None:
    """Break caught: stored source digests are accepted after current local code drifts."""

    context = _make_context(tmp_path, monkeypatch)
    index = 0 if source_kind == "authoritative" else -1
    source = context["source_paths"][index]
    source.write_bytes(source.read_bytes() + b"# source drift\n")
    with pytest.raises(ValueError, match="source|implementation|bundle|changed|identity"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


def test_calibration_stage_is_a_post_completion_create_once_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: Task 11 emits a generic pre-run stage instead of the post-run schema."""

    context = _make_context(tmp_path, monkeypatch)
    _rewrite_stage(
        context["stage"],
        lambda stage: stage["publication"].__setitem__("policy", "pre_run"),
    )
    with pytest.raises(ValueError, match="publication|completed|create_once|policy"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


def test_cli_publishes_exact_frozen_schema_formula_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the CLI omits a scientific identity or stores an unapproved formula."""

    context = _make_context(tmp_path, monkeypatch)
    assert calibration_main(_cli_args(context)) == 0
    payload = json.loads(context["output"].read_bytes())
    assert context["output"].read_bytes() == canonical_json_bytes(payload)
    assert set(payload) == {
        "schema",
        "status",
        "lambdas",
        "calibration",
        "measurements",
        "provenance",
        "semantic_sha256",
    }
    assert payload["schema"] == "pcraft_soft_loss_v3_weights_v1"
    assert payload["status"] == "APPROVED_FROZEN"
    assert payload["lambdas"] == {
        "lambda_dic": 0.15 * 20.0 / (4.0 + 1e-12),
        "lambda_dts": 0.15 * 20.0 / (51.5 + 1e-12),
    }
    assert payload["calibration"] == {
        "factor": 0.15,
        "epsilon": 1e-12,
        "formula_dic": "0.15 * median(g0) / (median(g_dic) + 1e-12)",
        "formula_dts": "0.15 * median(g0) / (median(g_dts_004,g_dts_006 by ordered update) + 1e-12)",
    }
    assert [item["completed_updates"] for item in payload["measurements"]] == list(UPDATES)
    assert set(payload["provenance"]) == {
        "stage_manifest",
        "calibration_run",
        "master_manifest",
        "development_suite",
        "specification",
        "inputs_xyz",
        "implementation",
        "interpreter",
        "executed_source_bundle",
    }
    assert payload["semantic_sha256"] == canonical_hash(
        {key: value for key, value in payload.items() if key != "semantic_sha256"}
    )
    validated = validate_frozen_weight_config(context["output"])
    assert validated["lambdas"] == payload["lambdas"]
    with pytest.raises(TypeError):
        validated["status"] = "changed"


def test_stage_and_run_bind_exact_task_config_audit_and_artifact_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a different run, task, audit, or unregistered checkpoint is calibrated."""

    context = _make_context(tmp_path, monkeypatch)
    original = context["stage"].read_bytes()
    mutations = (
        lambda value: value["calibration_run"].__setitem__("task", "blade_heroes"),
        lambda value: value["calibration_run"].__setitem__("config_hash", "b" * 64),
        lambda value: value["checkpoints"].pop(),
        lambda value: value["checkpoints"][0].__setitem__("sha256", "c" * 64),
    )
    for mutate in mutations:
        context["stage"].write_bytes(original)
        _rewrite_stage(context["stage"], mutate)
        with pytest.raises(ValueError, match="task|config|checkpoint|three|hash|identity"):
            calibration_main(_cli_args(context))
        assert not context["output"].exists()

    context["stage"].write_bytes(original)
    artifact = json.loads(context["artifact_manifest"].read_bytes())
    del artifact["artifacts_sha256"]["calibration_checkpoint_step_00001000.pt"]
    _write_canonical(context["artifact_manifest"], artifact)
    with pytest.raises(ValueError, match="artifact|checkpoint|hash|STATUS|stage"):
        calibration_main(_cli_args(context))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("calibration_probe", False),
        ("resolution", 256),
        ("export_resolution", 256),
        ("steps", 1000),
        ("seed", 29),
        ("soft_only", False),
        ("lambda_dic", 0.1),
        ("lambda_dts", 0.1),
        ("weight_config", {"path": "/tmp/weights", "sha256": "a" * 64}),
        ("final_step_export_only", False),
        ("profile", "dic_only"),
    ],
)
def test_run_config_rejects_every_non_calibration_identity_even_when_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    """Break caught: a coherently rehashed but scientifically different run is calibrated."""

    context = _make_context(tmp_path, monkeypatch)
    _rewrite_run_identity(
        context,
        mutate_config=lambda config: config.__setitem__(field, value),
    )
    with pytest.raises(ValueError, match="run config|calibration|profile|frozen|scope"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


def test_run_scope_cli_path_and_arcane_input_identity_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: a non-soft scope, guessed run path, or other development task is used."""

    scope_context = _make_context(tmp_path / "scope", monkeypatch)
    _rewrite_run_identity(
        scope_context,
        mutate_manifest=lambda manifest: manifest["scope"].__setitem__(
            "soft_only", False
        ),
    )
    with pytest.raises(ValueError, match="scope|soft"):
        calibration_main(_cli_args(scope_context))

    path_context = _make_context(tmp_path / "path", monkeypatch)
    alias = tmp_path / "different-run"
    alias.mkdir()
    args = _cli_args(path_context)
    args[args.index("--calibration-run") + 1] = str(alias)
    with pytest.raises((FileNotFoundError, ValueError), match="run|stage|manifest"):
        calibration_main(args)

    input_context = _make_context(tmp_path / "input", monkeypatch)
    development = json.loads(input_context["development"].read_bytes())
    blade_inputs = development["tasks"][1]["inputs"]

    def replace_inputs(config: dict[str, object]) -> None:
        config["shadow_inputs_xyz"] = [
            {"path": item["resolved_path"], "sha256": item["sha256"]}
            for item in blade_inputs
        ]

    _rewrite_run_identity(input_context, mutate_config=replace_inputs)
    _rewrite_stage(
        input_context["stage"],
        lambda stage: stage.__setitem__("inputs_xyz", blade_inputs),
    )
    with pytest.raises(ValueError, match="arcane|input|path|identity"):
        calibration_main(_cli_args(input_context))


@pytest.mark.parametrize("dependency", ["master", "development", "spec", "input", "interpreter"])
def test_current_master_suite_spec_input_and_interpreter_bytes_are_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    """Break caught: a stored provenance digest is trusted after dependency bytes drift."""

    context = _make_context(tmp_path, monkeypatch)
    target = {
        "master": context["master"],
        "development": context["development"],
        "spec": context["spec"],
        "input": context["inputs"][0],
        "interpreter": context["interpreter"],
    }[dependency]
    target.write_bytes(target.read_bytes() + b"drift")
    with pytest.raises(
        ValueError,
        match="changed|hash|sha256|canonical|dependency|identity|JSON",
    ):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


def test_master_and_development_cross_identity_mismatches_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: stage/master/run may bind different development suite identities."""

    first = _make_context(tmp_path / "first", monkeypatch)
    second = _make_context(tmp_path / "second", monkeypatch)
    args = _cli_args(first)
    args[args.index("--master-manifest") + 1] = str(second["master"])
    with pytest.raises(ValueError, match="master|stage|identity|hash"):
        calibration_main(args)
    args = _cli_args(first)
    args[args.index("--development-suite") + 1] = str(second["development"])
    with pytest.raises(ValueError, match="development|stage|suite|identity|hash"):
        calibration_main(args)
    assert not first["output"].exists()


@pytest.mark.parametrize(
    ("location", "key", "value"),
    [
        ("audit", "iou", 0.9),
        ("stage", "dice", 0.9),
        ("stage", "checkpoint_selection", "best"),
        ("stage", "threshold_quality_summary", {}),
        ("stage", "reference", "metrics.json"),
    ],
)
def test_quality_metric_keys_and_references_are_rejected_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
    key: str,
    value: object,
) -> None:
    """Break caught: task outcomes or a quality-selected checkpoint influence weights."""

    context = _make_context(tmp_path, monkeypatch)
    path = context["gradient_audit"] if location == "audit" else context["stage"]
    payload = json.loads(path.read_bytes())
    payload[key] = value
    if location == "stage":
        payload["semantic_sha256"] = canonical_hash(
            {name: item for name, item in payload.items() if name != "semantic_sha256"}
        )
    _write_canonical(path, payload)
    if location == "audit":
        _synchronize_completed_run_hashes(context)
    with pytest.raises(ValueError, match="quality|metric|forbidden|selection|reference"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


@pytest.mark.parametrize(
    ("location", "key", "value"),
    [
        ("audit", "Worst-View IoU", 0.9),
        ("audit", "background_normalized_FPR", 0.1),
        ("stage", "Foreground Recall", 0.9),
        ("stage", "PRECISION", 0.9),
        ("stage", "Boundary-F1", 0.9),
        ("stage", "MSCD_pixels", 1.0),
        ("stage", "nested_rank", 1),
        ("stage", "artifact", "nested/path/METRICS.JSON"),
    ],
)
def test_quality_boundary_is_recursive_case_and_punctuation_insensitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
    key: str,
    value: object,
) -> None:
    """Break caught: a nested or differently spelled task-quality signal bypasses calibration."""

    context = _make_context(tmp_path, monkeypatch)
    path = context["gradient_audit"] if location == "audit" else context["stage"]
    payload = json.loads(path.read_bytes())
    payload.setdefault("nested", {})["deeper"] = {key: value}
    if location == "stage":
        payload["semantic_sha256"] = canonical_hash(
            {name: item for name, item in payload.items() if name != "semantic_sha256"}
        )
    _write_canonical(path, payload)
    if location == "audit":
        _synchronize_completed_run_hashes(context)
    with pytest.raises(ValueError, match="quality|metric|forbidden|artifact"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


def test_public_writer_rejects_nested_quality_provenance_before_schema_matching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: direct callers inject a quality selector into otherwise valid provenance."""

    context = _make_context(tmp_path, monkeypatch)
    loaded = calibration_module._load_context(
        calibration_run=context["run"],
        stage_manifest=context["stage"],
        master_manifest=context["master"],
        development_suite=context["development"],
    )
    provenance = json.loads(canonical_json_bytes(loaded.provenance))
    provenance["calibration_run"]["nested"] = {"selection score": 0.99}
    with pytest.raises(ValueError, match="quality|metric|forbidden|score"):
        write_frozen_weight_config(
            tmp_path / "direct.json",
            measurements=loaded.measurements,
            provenance=provenance,
        )


def test_calibration_io_never_opens_a_task_quality_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: a hidden outcome file is opened even though no CLI flag names it."""

    context = _make_context(tmp_path, monkeypatch)
    opened: list[str] = []
    original = calibration_module.os.open

    def record_open(path: object, *args: object, **kwargs: object) -> int:
        opened.append(os.fspath(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(calibration_module.os, "open", record_open)
    assert calibration_main(_cli_args(context)) == 0
    assert opened
    assert all(Path(path).name.casefold() != "metrics.json" for path in opened)


def test_task8_artifact_integrity_map_allows_reserved_metrics_without_opening_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: integrity-only metric names are rejected or opened as selection data."""

    context = _make_context(tmp_path, monkeypatch)
    metrics = context["run"] / "metrics.json"
    resources = context["run"] / "resource_metrics.json"
    metrics.write_bytes(b'{"mean_iou":0.999}')
    resources.write_bytes(b'{"elapsed_seconds":1.0}')
    artifact = json.loads(context["artifact_manifest"].read_bytes())
    artifact["artifacts_sha256"]["metrics.json"] = _sha(metrics)
    artifact["artifacts_sha256"]["resource_metrics.json"] = _sha(resources)
    realistic_integrity_names = {
        "COMMAND.json",
        "environment.json",
        "history.jsonl",
        "occupancy_probability.npy",
        "checkpoint_latest.json",
        "checkpoint_final.pt",
        *{
            f"occupancy_binary_tau_{threshold}.npy"
            for threshold in ("0p04", "0p05", "0p06")
        },
        *{
            f"binary_tau_{threshold}_view_{view}.png"
            for threshold in ("0p04", "0p05", "0p06")
            for view in range(6)
        },
        *{
            f"transactions/step_{update:08d}/{name}"
            for update in range(0, 2001, 100)
            for name in (
                "checkpoint.pt",
                "history_segment.jsonl",
                "transaction_manifest.json",
            )
        },
    }
    for name in realistic_integrity_names:
        artifact["artifacts_sha256"][name] = hashlib.sha256(
            name.encode("utf-8")
        ).hexdigest()
    _write_canonical(context["artifact_manifest"], artifact)
    _synchronize_completed_run_hashes(context)

    opened: list[str] = []
    original_open = calibration_module.os.open

    def record_open(path: object, *args: object, **kwargs: object) -> int:
        opened.append(os.fspath(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(calibration_module.os, "open", record_open)
    assert calibration_main(_cli_args(context)) == 0
    validate_frozen_weight_config(context["output"])
    assert context["output"].exists()
    assert all(Path(path).name not in {"metrics.json", "resource_metrics.json"} for path in opened)
    frozen = context["output"].read_text(encoding="utf-8")
    assert "metrics.json" not in frozen
    assert _sha(metrics) not in frozen


def test_task9_approved_formal_quality_audit_is_hash_bound_but_not_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: approved formal audit fields are rejected or copied into weights."""

    context = _make_context(
        tmp_path,
        monkeypatch,
        formal_quality_audit=True,
    )
    assert calibration_main(_cli_args(context)) == 0
    frozen = context["output"].read_text(encoding="utf-8")
    for forbidden in (
        "mean_iou",
        "view_ious",
        "boundary_audit",
        "input_connectivity_audit",
        "candidate_pool",
    ):
        assert forbidden not in frozen


def test_stage_and_run_source_bundles_cannot_disagree_on_overlapping_v3_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: stage claims different calibration code than the run config."""

    context = _make_context(tmp_path, monkeypatch)
    alternate = (
        tmp_path
        / "alternate"
        / "shadowforge_pipeline"
        / "soft_voxel_v3"
        / "calibration.py"
    )
    alternate.parent.mkdir(parents=True)
    alternate.write_bytes(b"# different calibration implementation\n")

    def replace_overlap(stage: dict[str, object]) -> None:
        bundle = stage["executed_source_bundle"]
        files = bundle["files"]
        files["shadowforge_pipeline/soft_voxel_v3/calibration.py"] = {
            "path": str(alternate.resolve()),
            "sha256": _sha(alternate),
        }
        bundle["bundle_sha256"] = canonical_hash(files)

    _rewrite_stage(context["stage"], replace_overlap)
    with pytest.raises(ValueError, match="source|bundle|overlap|implementation|path|sha256"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


def test_run_implementation_may_bind_unimported_metrics_source_as_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: a source filename is mistaken for quality-selection input."""

    context = _make_context(tmp_path, monkeypatch)
    assert calibration_main(_cli_args(context)) == 0
    frozen = json.loads(context["output"].read_bytes())
    assert "metrics.py" in frozen["provenance"]["implementation"]["files_sha256"]


def test_coherently_rehashed_run_config_cannot_carry_nested_quality_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: free-form determinism metadata becomes a quality side channel."""

    context = _make_context(tmp_path, monkeypatch)
    _rewrite_run_identity(
        context,
        mutate_config=lambda config: config["determinism"]["environment"].__setitem__(
            "quality_metric", "0.999"
        ),
    )

    with pytest.raises(ValueError, match="quality|metric|forbidden"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


def test_coherently_rehashed_run_manifest_cannot_carry_selection_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: run-level metadata carries a quality-driven selection side channel."""

    context = _make_context(tmp_path, monkeypatch)
    _rewrite_run_identity(
        context,
        mutate_manifest=lambda manifest: manifest.__setitem__(
            "selection.audit", {"mean/iou": 0.999}
        ),
    )
    with pytest.raises(ValueError, match="quality|selection|iou|forbidden"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


@pytest.mark.parametrize(
    "key",
    ["quality.metric", "mean/iou", "Boundary:F1", "FPR%"],
)
def test_strict_run_config_quality_keys_cannot_bypass_with_punctuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    """Break caught: separator spelling bypasses the run selection-data boundary."""

    context = _make_context(tmp_path, monkeypatch)
    _rewrite_run_identity(
        context,
        mutate_config=lambda config: config["determinism"]["environment"].__setitem__(
            key, "0.999"
        ),
    )
    with pytest.raises(ValueError, match="quality|metric|forbidden"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


@pytest.mark.parametrize(
    "key",
    [
        "dic_dts_support_ious",
        "support-IoUs",
        "supportIoU",
        "meanIoU",
        "BoundaryF1",
        "qualityMetric",
        "checkpointSelection",
        "SUPPORT.IOUS",
        "Mean-IoU",
        "boundary/F1",
        "Quality:Metric",
        "Checkpoint.Selection",
    ],
)
def test_run_config_rejects_plural_camelcase_and_compound_quality_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    """Break caught: token normalization misses plural or CamelCase quality aliases."""

    context = _make_context(tmp_path, monkeypatch)
    _rewrite_run_identity(
        context,
        mutate_config=lambda config: config["determinism"]["environment"].__setitem__(
            key, "0.999"
        ),
    )
    with pytest.raises(ValueError, match="quality|metric|iou|selection|forbidden"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


@pytest.mark.parametrize(
    "key",
    [
        "scores",
        "rankings",
        "recalls",
        "precisions",
        "dices",
        "ssims",
        "FPRs",
        "MSCDs",
        "checkpointSelections",
    ],
)
def test_run_config_rejects_plural_quality_and_selection_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    """Break caught: a plural quality selector bypasses singular token checks."""

    context = _make_context(tmp_path, monkeypatch)
    _rewrite_run_identity(
        context,
        mutate_config=lambda config: config["determinism"]["environment"].__setitem__(
            key, "0.999"
        ),
    )
    with pytest.raises(ValueError, match="quality|metric|selection|forbidden"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


@pytest.mark.parametrize(
    "field",
    ["dic_dts_support_ious", "dic_dts_support_iou_invalid"],
)
def test_task8_support_iou_exceptions_are_illegal_outside_exact_sample_position(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Break caught: either Task 8 sample-only exception becomes a global bypass."""

    context = _make_context(tmp_path, monkeypatch)
    _rewrite_run_identity(
        context,
        mutate_config=lambda config: config["determinism"]["environment"].__setitem__(
            field, "false"
        ),
    )
    with pytest.raises(ValueError, match="quality|iou|forbidden|position"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


@pytest.mark.parametrize("missing", ["metrics.json", "resource_metrics.json"])
def test_task8_artifact_manifest_requires_both_integrity_only_metric_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    """Break caught: calibration accepts a non-Task8 artifact map missing a required entry."""

    context = _make_context(tmp_path, monkeypatch)
    artifact = json.loads(context["artifact_manifest"].read_bytes())
    artifact["artifacts_sha256"].pop(missing)
    _write_canonical(context["artifact_manifest"], artifact)
    _synchronize_completed_run_hashes(context)
    with pytest.raises(
        ValueError,
        match="required.*(metrics|resource)|(metrics|resource).*required|must contain",
    ):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


def test_parser_has_only_five_required_nonabbreviated_paths() -> None:
    """Break caught: calibration exposes a metric, selection, task, threshold, or weight knob."""

    parser = build_calibration_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.allow_abbrev is False
    assert {action.dest for action in parser._actions} == {
        "help",
        "calibration_run",
        "stage_manifest",
        "master_manifest",
        "development_suite",
        "output",
    }
    assert all(action.required for action in parser._actions if action.dest != "help")
    valid = ["run", "stage", "master", "development", "output"]
    flags = [
        "--calibration-run",
        "--stage-manifest",
        "--master-manifest",
        "--development-suite",
        "--output",
    ]
    for abbreviation in ("--cal", "--stage", "--master", "--development", "--out"):
        argv = [item for pair in zip(flags, valid) for item in pair]
        index = next(index for index, item in enumerate(argv) if item.startswith(abbreviation))
        argv[index] = abbreviation
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    for forbidden in (
        "--metrics-path",
        "--selection-criterion",
        "--threshold",
        "--task",
        "--lambda-dic",
        "--checkpoint",
    ):
        argv = [item for pair in zip(flags, valid) for item in pair] + [forbidden, "x"]
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


@pytest.mark.parametrize(
    "mutation",
    ["lambda", "measurement", "provenance", "status", "extra", "semantic"],
)
def test_validator_recomputes_every_scientific_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: validation trusts stored lambdas, identity, status, or semantic hash."""

    context = _make_context(tmp_path, monkeypatch)
    assert calibration_main(_cli_args(context)) == 0

    def mutate(payload: dict[str, object]) -> None:
        if mutation == "lambda":
            payload["lambdas"]["lambda_dic"] *= 2.0
        elif mutation == "measurement":
            payload["measurements"][0]["g0"] = 999.0
        elif mutation == "provenance":
            payload["provenance"]["calibration_run"]["config_hash"] = "f" * 64
        elif mutation == "status":
            payload["status"] = "PENDING"
        elif mutation == "extra":
            payload["unregistered"] = True
        else:
            payload["semantic_sha256"] = "0" * 64

    _rewrite_output(context["output"], mutate, rehash=mutation != "semantic")
    with pytest.raises(ValueError, match="lambda|measurement|provenance|status|schema|semantic|extra|identity"):
        validate_frozen_weight_config(context["output"])


@pytest.mark.parametrize(
    "dependency",
    [
        "checkpoint",
        "stage",
        "run_manifest",
        "gradient_audit",
        "artifact_manifest",
        "status_file",
        "master",
        "development",
        "formal",
        "spec",
        "input",
        "interpreter",
        "authoritative_source",
        "executed_source",
    ],
)
def test_validator_rehashes_every_external_dependency_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    """Break caught: validation checks only stored weight bytes after a dependency drifts."""

    context = _make_context(tmp_path, monkeypatch)
    assert calibration_main(_cli_args(context)) == 0
    target = {
        "checkpoint": context["run"] / "calibration_checkpoint_step_00001000.pt",
        "stage": context["stage"],
        "run_manifest": context["run_manifest"],
        "gradient_audit": context["gradient_audit"],
        "artifact_manifest": context["artifact_manifest"],
        "status_file": context["status"],
        "master": context["master"],
        "development": context["development"],
        "formal": context["formal"],
        "spec": context["spec"],
        "input": context["inputs"][0],
        "interpreter": context["interpreter"],
        "authoritative_source": context["source_paths"][0],
        "executed_source": context["source_paths"][-1],
    }[dependency]
    target.write_bytes(target.read_bytes() + b"dependency-drift")
    with pytest.raises(
        ValueError,
        match="changed|hash|identity|canonical|source|checkpoint|dependency|JSON",
    ):
        validate_frozen_weight_config(context["output"])


def test_identical_write_is_idempotent_but_nonidentical_and_symlink_targets_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a frozen weight file is overwritten or a target symlink is followed."""

    context = _make_context(tmp_path, monkeypatch)
    assert calibration_main(_cli_args(context)) == 0
    accepted = context["output"].read_bytes()
    assert calibration_main(_cli_args(context)) == 0
    assert context["output"].read_bytes() == accepted

    context["output"].write_bytes(b"different-existing-bytes")
    with pytest.raises((FileExistsError, ValueError), match="immutable|differ|canonical"):
        calibration_main(_cli_args(context))
    assert context["output"].read_bytes() == b"different-existing-bytes"

    other = _make_context(tmp_path / "symlink", monkeypatch)
    outside = tmp_path / "outside-weight.json"
    outside.write_bytes(b"outside-sentinel")
    other["output"].symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        calibration_main(_cli_args(other))
    assert outside.read_bytes() == b"outside-sentinel"


@pytest.mark.parametrize(
    "dependency",
    ["audit", "stage", "master", "development", "spec", "run", "output_parent"],
)
def test_every_path_boundary_rejects_symlink_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    """Break caught: no-follow protection is applied only to retained checkpoints."""

    context = _make_context(tmp_path, monkeypatch)
    if dependency == "run":
        real_run = tmp_path / "real-calibration-run"
        context["run"].rename(real_run)
        context["run"].symlink_to(real_run, target_is_directory=True)
    elif dependency == "output_parent":
        real_parent = tmp_path / "real-output-parent"
        real_parent.mkdir()
        link_parent = tmp_path / "linked-output-parent"
        link_parent.symlink_to(real_parent, target_is_directory=True)
        context["output"] = link_parent / "weights.json"
    else:
        target = {
            "audit": context["gradient_audit"],
            "stage": context["stage"],
            "master": context["master"],
            "development": context["development"],
            "spec": context["spec"],
        }[dependency]
        outside = tmp_path / f"outside-{dependency}"
        outside.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        calibration_main(_cli_args(context))
    assert not os.path.lexists(context["output"])


@pytest.mark.parametrize("malformation", ["duplicate", "nonfinite", "noncanonical"])
def test_json_dependencies_reject_duplicate_nonfinite_and_noncanonical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    """Break caught: a permissive JSON parser normalizes ambiguous dependency bytes."""

    context = _make_context(tmp_path, monkeypatch)
    if malformation == "duplicate":
        context["gradient_audit"].write_bytes(
            b'{"schema":"a","schema":"b"}'
        )
    elif malformation == "nonfinite":
        context["gradient_audit"].write_bytes(b'{"value":NaN}')
    else:
        payload = json.loads(context["stage"].read_bytes())
        context["stage"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if malformation in {"duplicate", "nonfinite"}:
        _synchronize_completed_run_hashes(context)
    with pytest.raises(ValueError, match="JSON|duplicate|finite|canonical"):
        calibration_main(_cli_args(context))
    assert not context["output"].exists()


def test_stable_snapshot_rejects_inode_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: bytes from an unlinked fd are attributed to a replacement path."""

    target = tmp_path / "dependency.json"
    target.write_bytes(b"stable-original")
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b"replacement")
    original_read = calibration_module.os.read
    state = {"mutated": False}

    def replace_after_read(descriptor: int, size: int) -> bytes:
        data = original_read(descriptor, size)
        if data and not state["mutated"]:
            os.replace(replacement, target)
            state["mutated"] = True
        return data

    monkeypatch.setattr(calibration_module.os, "read", replace_after_read)
    with pytest.raises(ValueError, match="path changed|snapshot|changed"):
        calibration_module._snapshot_file(target, label="racing dependency")
    assert state["mutated"] is True


def test_scientific_hash_excludes_output_location_but_binds_scientific_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: orchestration output paths perturb, or gradients fail to perturb, identity."""

    context = _make_context(tmp_path, monkeypatch)
    first = tmp_path / "one" / "weights.json"
    second = tmp_path / "two" / "weights.json"
    args = _cli_args(context)
    args[args.index("--output") + 1] = str(first)
    assert calibration_main(args) == 0
    args[args.index("--output") + 1] = str(second)
    assert calibration_main(args) == 0
    assert first.read_bytes() == second.read_bytes()

    payload = json.loads(first.read_bytes())
    original_semantic = payload["semantic_sha256"]
    payload["measurements"][0]["g0"] = 11.0
    changed_semantic = canonical_hash(
        {key: payload[key] for key in (
            "schema",
            "status",
            "lambdas",
            "calibration",
            "measurements",
            "provenance",
        )}
    )
    assert changed_semantic != original_semantic


def test_public_writer_and_validator_are_recursive_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: direct publication diverges from CLI or nested validated state is mutable."""

    context = _make_context(tmp_path, monkeypatch)
    loaded = calibration_module._load_context(
        calibration_run=context["run"],
        stage_manifest=context["stage"],
        master_manifest=context["master"],
        development_suite=context["development"],
    )
    direct = tmp_path / "direct" / "weights.json"
    assert write_frozen_weight_config(
        direct,
        measurements=loaded.measurements,
        provenance=loaded.provenance,
    ) == direct
    validated = validate_frozen_weight_config(direct)
    with pytest.raises(TypeError):
        validated["lambdas"]["lambda_dic"] = 0.0
    with pytest.raises(TypeError):
        validated["measurements"][0]["g0"] = 0.0
    with pytest.raises(TypeError):
        validated["provenance"]["calibration_run"]["task"] = "changed"


def test_prepublication_and_postpublication_dependency_drift_roll_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a stale provenance snapshot commits before or after the final link."""

    for phase in ("before", "after"):
        context = _make_context(tmp_path / phase, monkeypatch)
        original_link = calibration_module.os.link
        state = {"mutated": False, "observed_target": None}

        def mutate_at_link(source: object, target: object, *args: object, **kwargs: object) -> None:
            if phase == "before":
                state["observed_target"] = os.path.lexists(context["output"])
                context["spec"].write_bytes(context["spec"].read_bytes() + b"race-drift")
                state["mutated"] = True
                original_link(source, target, *args, **kwargs)
            else:
                original_link(source, target, *args, **kwargs)
                state["observed_target"] = os.path.lexists(context["output"])
                context["spec"].write_bytes(context["spec"].read_bytes() + b"race-drift")
                state["mutated"] = True

        monkeypatch.setattr(calibration_module.os, "link", mutate_at_link)
        with pytest.raises(ValueError, match="changed|snapshot|dependency|drift"):
            calibration_main(_cli_args(context))
        assert state["mutated"] is True
        assert state["observed_target"] is (phase == "after")
        assert not context["output"].exists()
        monkeypatch.setattr(calibration_module.os, "link", original_link)


@pytest.mark.parametrize(
    "replacement_kind",
    ["different_bytes", "same_bytes_different_inode", "symlink", "missing"],
)
def test_publish_rejects_target_replaced_after_link_without_deleting_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    """Break caught: link succeeds, target is swapped, and publication reports success."""

    target = tmp_path / "weights.json"
    payload = b"canonical-weight-bytes"
    dependencies = calibration_module._DependencySnapshot(())
    original_link = calibration_module.os.link
    attacker = tmp_path / "attacker.json"
    attacker_identity: tuple[int, int] | None = None

    def replace_after_link(source: object, destination: object, *args: object, **kwargs: object) -> None:
        nonlocal attacker_identity
        original_link(source, destination, *args, **kwargs)
        destination_path = Path(os.fspath(destination))
        if replacement_kind == "different_bytes":
            attacker.write_bytes(b"attacker-controlled")
            metadata = attacker.stat()
            attacker_identity = (metadata.st_dev, metadata.st_ino)
            os.replace(attacker, destination_path)
        elif replacement_kind == "same_bytes_different_inode":
            attacker.write_bytes(payload)
            metadata = attacker.stat()
            attacker_identity = (metadata.st_dev, metadata.st_ino)
            os.replace(attacker, destination_path)
        elif replacement_kind == "symlink":
            attacker.write_bytes(b"attacker-controlled")
            destination_path.unlink()
            destination_path.symlink_to(attacker)
        else:
            destination_path.unlink()

    monkeypatch.setattr(calibration_module.os, "link", replace_after_link)
    with pytest.raises(
        (FileNotFoundError, ValueError),
        match="publish|target|identity|bytes|changed|snapshot|symlink|missing|exist",
    ):
        calibration_module._publish_weight_bytes(
            target,
            payload,
            dependencies=dependencies,
        )

    if replacement_kind in {"different_bytes", "same_bytes_different_inode"}:
        assert target.exists()
        metadata = target.stat()
        assert (metadata.st_dev, metadata.st_ino) == attacker_identity
        assert target.read_bytes() == (
            b"attacker-controlled" if replacement_kind == "different_bytes" else payload
        )
    elif replacement_kind == "symlink":
        assert target.is_symlink()
        assert attacker.read_bytes() == b"attacker-controlled"
    else:
        assert not os.path.lexists(target)


def test_concurrent_weight_publish_is_idempotent_or_single_winner(tmp_path: Path) -> None:
    """Break caught: concurrent writers overwrite, tear, or both win different bytes."""

    dependencies = calibration_module._DependencySnapshot(())
    same_target = tmp_path / "same.json"

    def publish(payload: bytes, target: Path) -> object:
        try:
            return calibration_module._publish_weight_bytes(
                target,
                payload,
                dependencies=dependencies,
            )
        except Exception as error:  # assertions inspect the exact terminal race
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        same_results = list(pool.map(lambda _: publish(b"same", same_target), range(2)))
    assert same_results == [same_target, same_target]
    assert same_target.read_bytes() == b"same"

    different_target = tmp_path / "different.json"
    with ThreadPoolExecutor(max_workers=2) as pool:
        different_results = list(
            pool.map(
                lambda payload: publish(payload, different_target),
                (b"left", b"right"),
            )
        )
    assert sum(isinstance(result, Path) for result in different_results) == 1
    assert sum(isinstance(result, FileExistsError) for result in different_results) == 1
    assert different_target.read_bytes() in {b"left", b"right"}


def test_import_and_ast_scope_excludes_training_geometry_globs_and_quality_file_access(
    tmp_path: Path,
) -> None:
    """Break caught: calibration imports execution code, scans outputs, or names a quality file."""

    paths = [
        PROJECT_ROOT / "shadowforge_pipeline" / "soft_voxel_v3" / "calibration.py",
        PROJECT_ROOT / "experiments" / "calibrate_soft_loss_v3.py",
    ]
    forbidden_import_parts = {
        "trainer",
        "metrics",
        "mesh",
        "geometry",
        "fem",
        "pino",
        "runner",
    }
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "/data/" not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name.lower() for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported = [(node.module or "").lower()]
            else:
                imported = []
            assert not any(
                part in name.replace("-", "_").split(".")
                for name in imported
                for part in forbidden_import_parts
            )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"glob", "rglob"}
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                    assert node.right.value.casefold() != "metrics.json"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                else:
                    call_name = ""
                if call_name in {
                    "open",
                    "Path",
                    "_snapshot_file",
                    "read_text",
                    "read_bytes",
                    "resolve",
                    "stat",
                    "lstat",
                }:
                    literal_arguments = {
                        child.value.casefold()
                        for argument in (*node.args, *[kw.value for kw in node.keywords])
                        for child in ast.walk(argument)
                        if isinstance(child, ast.Constant) and isinstance(child.value, str)
                    }
                    assert "metrics.json" not in literal_arguments

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; before=list(Path('.').iterdir()); "
                "import shadowforge_pipeline.soft_voxel_v3.calibration; "
                "import experiments.calibrate_soft_loss_v3; "
                "assert list(Path('.').iterdir()) == before"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_direct_launcher_help_has_no_import_time_execution() -> None:
    """Break caught: the thin script launches work on import or fails as a direct path."""

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "calibrate_soft_loss_v3.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--calibration-run" in completed.stdout
    assert "--stage-manifest" in completed.stdout
