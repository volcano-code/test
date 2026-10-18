from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import experiments.analyze_soft_loss_v6_ablation10_256 as analysis
from shadowforge_pipeline.soft_voxel_v6.provenance import FileSnapshot, canonical_json_bytes
from shadowforge_pipeline.soft_voxel_v6.core import (
    PROFILE_FLAGS,
    fixed_camera_payload,
    map_suite_views,
    v6_auxiliary_multiplier,
    v6_learning_rate,
)
from shadowforge_pipeline.soft_voxel_v3.data import normalize_target


def test_cli_is_exact_and_disallows_abbreviations() -> None:
    parser = analysis.build_parser()
    parsed = parser.parse_args(["--output-root", "/tmp/exact", "--verify"])
    assert parsed.output_root == Path("/tmp/exact")
    assert parsed.verify is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "/tmp/exact"])


def test_canonical_root_rejects_relative_symlink_and_missing(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert analysis._canonical_root(root) == root
    with pytest.raises(ValueError):
        analysis._canonical_root(Path("relative"))
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        analysis._canonical_root(alias)


def test_csv_is_published_before_report_and_report_is_commit_marker(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    csv_bytes = b"task,arm\nconnected_01,base\n"
    report_bytes = b'{"schema":"report"}\n'
    calls: list[str] = []

    def revalidate(stage: str) -> None:
        calls.append(stage)

    analysis._publish_outputs(root, csv_bytes, report_bytes, revalidate=revalidate)
    assert calls == ["before_csv", "before_report"]
    assert (root / "analysis/soft_rows.csv").read_bytes() == csv_bytes
    assert (root / "analysis/soft_report.json").read_bytes() == report_bytes
    with pytest.raises(FileExistsError):
        analysis._publish_outputs(root, csv_bytes, report_bytes, revalidate=revalidate)


def test_crash_after_csv_allows_only_byte_identical_recovery(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "analysis").mkdir(parents=True)
    csv_bytes = b"task,arm\nconnected_01,base\n"
    (root / "analysis/soft_rows.csv").write_bytes(csv_bytes)
    analysis._publish_outputs(root, csv_bytes, b'{}\n', revalidate=lambda stage: None)
    assert (root / "analysis/soft_report.json").is_file()
    other = tmp_path / "other"
    (other / "analysis").mkdir(parents=True)
    (other / "analysis/soft_rows.csv").write_bytes(b"foreign")
    with pytest.raises(ValueError, match="CSV"):
        analysis._publish_outputs(other, csv_bytes, b'{}\n', revalidate=lambda stage: None)


def test_failure_before_csv_writes_nothing_and_between_leaves_only_csv(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(RuntimeError):
        analysis._publish_outputs(
            root, b"csv", b"report", revalidate=lambda stage: (_ for _ in ()).throw(RuntimeError(stage))
        )
    assert not (root / "analysis").exists()

    calls = 0
    def second(stage: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError(stage)
    with pytest.raises(RuntimeError):
        analysis._publish_outputs(root, b"csv", b"report", revalidate=second)
    assert (root / "analysis/soft_rows.csv").read_bytes() == b"csv"
    assert not (root / "analysis/soft_report.json").exists()


def test_verify_mode_is_byte_identity_and_zero_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    (root / "analysis").mkdir(parents=True)
    csv_bytes = b"csv"
    report_bytes = b"report"
    csv_path = root / "analysis/soft_rows.csv"
    report_path = root / "analysis/soft_report.json"
    csv_path.write_bytes(csv_bytes)
    report_path.write_bytes(report_bytes)
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in (csv_path, report_path)}
    monkeypatch.setattr(analysis, "atomic_bytes", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("write")))
    analysis._verify_outputs(root, csv_bytes, report_bytes, revalidate=lambda stage: None)
    assert before == {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in (csv_path, report_path)}
    report_path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="report"):
        analysis._verify_outputs(root, csv_bytes, report_bytes, revalidate=lambda stage: None)


def test_publication_and_verify_reject_analysis_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "root"; (root / "analysis").mkdir(parents=True)
    foreign = tmp_path / "foreign"; foreign.write_bytes(b"csv")
    (root / "analysis/soft_rows.csv").symlink_to(foreign)
    with pytest.raises(ValueError, match="symlink"):
        analysis._publish_outputs(root, b"csv", b"report", revalidate=lambda stage: None)
    with pytest.raises(ValueError, match="symlink"):
        analysis._verify_outputs(root, b"csv", b"report", revalidate=lambda stage: None)


def test_snapshot_revalidation_detects_post_snapshot_swap(tmp_path: Path) -> None:
    path = tmp_path / "evidence"
    path.write_bytes(b"old")
    snapshot = analysis._snapshot_bytes(path, label="evidence")
    assert snapshot.payload == b"old"
    path.write_bytes(b"new")
    with pytest.raises(ValueError, match="changed"):
        analysis._revalidate_all((snapshot,))


def test_exact_npy_loader_rejects_trailing_dtype_shape_nonfinite_and_pickle() -> None:
    import io
    import numpy as np

    valid = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
    buffer = io.BytesIO(); np.save(buffer, valid, allow_pickle=False)
    raw = buffer.getvalue()
    loaded = analysis._decode_exact_npy(raw, dtype=np.dtype("float32"), shape=valid.shape, label="x")
    assert loaded.flags.c_contiguous
    for bad in (raw + b"x",):
        with pytest.raises(ValueError):
            analysis._decode_exact_npy(bad, dtype=np.dtype("float32"), shape=valid.shape, label="x")
    with pytest.raises(ValueError):
        analysis._decode_exact_npy(raw, dtype=np.dtype("uint8"), shape=valid.shape, label="x")
    bad = valid.copy(); bad.flat[0] = np.nan
    buffer = io.BytesIO(); np.save(buffer, bad, allow_pickle=False)
    with pytest.raises(ValueError):
        analysis._decode_exact_npy(buffer.getvalue(), dtype=np.dtype("float32"), shape=valid.shape, label="x")
    buffer = io.BytesIO(); np.save(buffer, np.array([{"x": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        analysis._decode_exact_npy(buffer.getvalue(), dtype=np.dtype("float32"), shape=valid.shape, label="x")


def test_metric_recomputation_uses_six_view_equal_mean_not_pooled_counts() -> None:
    import torch

    target = torch.zeros((6, 10, 10), dtype=torch.bool)
    predicted = torch.zeros_like(target)
    target[0, 0, 0] = True
    predicted[0, 0, 0] = True
    for view in range(1, 6):
        target[view, :5, :] = True
        predicted[view, :5, :] = True
        predicted[view, 5, :] = True
    metrics = analysis._evaluate_masks(predicted, target)
    expected = (1.0 + 5 * (50 / 60)) / 6
    pooled = 251 / (251 + 50)
    assert metrics["aggregate"]["mean_iou"] == expected
    assert metrics["aggregate"]["mean_iou"] != pooled
    assert len(metrics["per_view"]) == 6


def test_metric_recomputation_rejects_invalid_denominators() -> None:
    import torch

    empty = torch.zeros((6, 2, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="denominator"):
        analysis._evaluate_masks(empty, empty)


def test_manifest_matrix_requires_exact_order_paths_and_endpoint(tmp_path: Path) -> None:
    root = tmp_path / "root"; root.mkdir()
    cells = []
    for task_index in range(1, 11):
        task = f"connected_{task_index:02d}"
        for arm in ("base", "j_only", "h_only", "full"):
            path = root / "runs" / task / arm / "seed_17"
            cells.append({"task": task, "arm": arm, "run_path": str(path), "resolved_config": {"run": {"root": str(root), "path": str(path)}, "updates": 3000, "resolution": 256, "seed": 17, "arm": arm, "task": task}})
    analysis._validate_cell_matrix(root, cells)
    for mutation in (
        lambda value: value.pop(),
        lambda value: value.append(dict(value[-1])),
        lambda value: value.reverse(),
        lambda value: value[0]["resolved_config"].__setitem__("updates", 2700),
    ):
        copy = [dict(row, resolved_config={**row["resolved_config"], "run": dict(row["resolved_config"]["run"])}) for row in cells]
        mutation(copy)
        with pytest.raises(ValueError):
            analysis._validate_cell_matrix(root, copy)


def test_synthetic_40_prevalidated_analysis_is_deterministic(tmp_path: Path) -> None:
    rows, activity = analysis._synthetic_passing_inputs_for_test()
    first = analysis._render_prevalidated_outputs(tmp_path, rows, activity, snapshots=())
    second = analysis._render_prevalidated_outputs(tmp_path, rows, activity, snapshots=())
    assert first == second
    assert b"claim_allowed" in first[1]


def _history(
    arm: str, *, full_ramp_zeros: int = 0, peak_cuda_allocated_bytes: int = 1
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    zero_updates = set(range(901, 901 + full_ramp_zeros))
    tail = tuple(
        {
            "view": view, "background_count": 20, "selected_count": 1,
            "selected_indices": [view], "score_min": 0.0,
            "score_mean": 0.0, "score_max": 0.0,
        }
        for view in range(6)
    )
    for update in range(1, 3001):
        multiplier = v6_auxiliary_multiplier(update - 1)
        enabled_j = multiplier > 0.0 and PROFILE_FLAGS[arm]["j"]
        enabled_h = multiplier > 0.0 and PROFILE_FLAGS[arm]["h"]
        j_zero = enabled_j and update in zero_updates
        h_zero = enabled_h and update in zero_updates
        rows.append(
            {
                "completed_updates": update,
                "learning_rate": v6_learning_rate(update - 1),
                "auxiliary_multiplier": multiplier,
                "base_loss": 1.0,
                "j_loss": 1.0 if enabled_j else None,
                "h_loss": 1.0 if enabled_h else None,
                "base_norm": 1.0,
                "j_raw_norm": 0.0 if not enabled_j or j_zero else 1.0,
                "h_raw_norm": 0.0 if not enabled_h or h_zero else 1.0,
                "j_ratio": 0.0 if not enabled_j or j_zero else multiplier * 0.20,
                "h_ratio": 0.0 if not enabled_h or h_zero else multiplier * 0.10,
                "combined_ratio": 0.0,
                "j_zero": bool(j_zero),
                "h_zero": bool(h_zero),
                "pre_clip_norm": 1.0,
                "post_clip_norm": 1.0,
                "parameter_delta_norm": 1.0,
                "clip_calls": 1,
                "optimizer_steps": 1,
                "elapsed_seconds": 0.01,
                "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
                "tail_details": tail if enabled_h else (),
            }
        )
    return rows


@pytest.mark.parametrize(
    ("arm", "components"),
    [("base", ()), ("j_only", ("j",)), ("h_only", ("h",)), ("full", ("j", "h"))],
)
def test_engineering_recomputation_enforces_exact_601_3000_and_901_3000_ranges(
    arm: str, components: tuple[str, ...]
) -> None:
    audit, activity = analysis._derive_engineering(_history(arm), arm=arm)
    assert audit["eligible_updates"] == 2400
    assert audit["clip_calls"] == 3000
    assert audit["optimizer_steps"] == 3000
    assert tuple(activity) == components
    for component in components:
        assert activity[component] == {"eligible": 2100, "zero": 0, "active": 2100}


def test_engineering_zero_boundary_and_persisted_audit_disagreement() -> None:
    passing_prefix, passing = analysis._derive_engineering(_history("full", full_ramp_zeros=210), arm="full")
    _, failing = analysis._derive_engineering(_history("full", full_ramp_zeros=211), arm="full")
    assert passing["j"] == {"eligible": 2100, "zero": 210, "active": 1890}
    assert failing["h"] == {"eligible": 2100, "zero": 211, "active": 1889}
    assert "j_full_ramp_raw_zero_fraction" not in passing_prefix
    terminal = analysis._extend_terminal_audit(passing_prefix, arm="full")
    assert terminal["j_full_ramp_raw_zero_fraction"] == 0.1
    analysis._recompute_engineering(
        _history("full", full_ramp_zeros=210), terminal,
        checkpoint_audit=passing_prefix, arm="full",
    )
    forged = dict(terminal); forged["clip_calls"] = 2999
    with pytest.raises(ValueError, match="audit disagreement"):
        analysis._recompute_engineering(
            _history("full", full_ramp_zeros=210), forged,
            checkpoint_audit=passing_prefix, arm="full",
        )
    forged_prefix = dict(passing_prefix); forged_prefix["clip_calls"] = 2999
    with pytest.raises(ValueError, match="checkpoint audit"):
        analysis._recompute_engineering(
            _history("full", full_ramp_zeros=210), terminal,
            checkpoint_audit=forged_prefix, arm="full",
        )


def test_engineering_rejects_bool_alias_schedule_and_disabled_branch() -> None:
    for mutate in (
        lambda rows: rows[0].__setitem__("clip_calls", True),
        lambda rows: rows[1499].__setitem__("learning_rate", 0.0007),
        lambda rows: rows[700].__setitem__("j_loss", 1.0),
    ):
        rows = _history("base")
        mutate(rows)
        with pytest.raises(ValueError):
            analysis._derive_engineering(rows, arm="base")


class _TinyProtocol:
    def __init__(self, *, bad_camera: bool = False) -> None:
        camera = fixed_camera_payload()
        rotation = np.frombuffer(camera["rotation_bytes"], dtype=np.float32).reshape(6, 3, 3).copy()
        translation = np.frombuffer(camera["translation_bytes"], dtype=np.float32).reshape(6, 3).copy()
        if bad_camera:
            rotation[0, 0, 0] += 1.0
        self.cameras = SimpleNamespace(R=torch.from_numpy(rotation), T=torch.from_numpy(translation))


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _tiny_recompute_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], dict[str, analysis.EvidenceSnapshot], torch.Tensor]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    resolution = 4
    input_rows = []
    base: list[torch.Tensor] = []
    mirrors: list[torch.Tensor] = []
    raw_images = (
        np.array([[0, 0, 0, 0], [0, 255, 255, 0], [0, 255, 0, 0], [0, 0, 0, 0]], np.uint8),
        np.array([[0, 0, 0, 0], [0, 0, 255, 0], [0, 255, 255, 0], [0, 0, 0, 0]], np.uint8),
        np.array([[0, 0, 0, 0], [255, 255, 0, 0], [0, 255, 0, 0], [0, 0, 0, 0]], np.uint8),
    )
    for axis, image in zip("xyz", raw_images):
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        path = tmp_path / f"{axis}.png"
        path.write_bytes(encoded.tobytes())
        input_rows.append({"axis": axis, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        base.append(torch.from_numpy(normalize_target(image, resolution, False) > 0))
        mirrors.append(torch.from_numpy(normalize_target(cv2.flip(image, 1), resolution, False) > 0))
    mapped = map_suite_views(tuple(zip("xyz", base)), tuple(zip("xyz", mirrors)))
    camera = fixed_camera_payload()
    config: dict[str, object] = {
        "arm": "base", "device": "cpu", "resolution": resolution,
        "inputs": {"raw_xyz": input_rows},
        "targets": {
            "base_sha256": analysis._tensor_hash(torch.stack(base)),
            "mirror_sha256": analysis._tensor_hash(torch.stack(mirrors)),
            "mapped_six_sha256": analysis._tensor_hash(mapped),
        },
        "cameras": {
            "rotation_sha256": camera["rotation_sha256"],
            "translation_sha256": camera["translation_sha256"],
            "payload_sha256": camera["payload_sha256"],
        },
    }
    history = _history("base", peak_cuda_allocated_bytes=0)
    prefix, _ = analysis._derive_engineering(history, arm="base")
    terminal = analysis._extend_terminal_audit(prefix, arm="base")
    logits = torch.linspace(-3.0, 2.0, resolution**3, dtype=torch.float32).reshape(1, 1, resolution, resolution, resolution)
    density = torch.sigmoid(logits).numpy().astype(np.float32, copy=False)
    hard = (density >= 0.05).astype(np.uint8)
    render_mask = mapped.clone()
    metrics = analysis._evaluate_masks(render_mask, mapped)
    resource = {
        "schema": "pcraft_soft_loss_v6_resource_v1",
        "step_times": [0.01] * 3000,
        "all_finite_positive": True,
        "peak_cuda_allocated_bytes": 0,
        "device_total_memory_bytes": None,
        "cuda_executed": False,
    }
    checkpoint_buffer = io.BytesIO()
    torch.save({
            "completed_updates": 3000, "logits": logits, "history": history,
            "step_times": [0.01] * 3000, "audit": prefix,
        }, checkpoint_buffer)
    payloads: dict[str, bytes] = {
        "checkpoint": checkpoint_buffer.getvalue(),
        "audit": canonical_json_bytes(terminal),
        "logits": _npy_bytes(logits.numpy()),
        "density": _npy_bytes(density),
        "hard": _npy_bytes(hard),
        "metrics": canonical_json_bytes(metrics),
        "resource": canonical_json_bytes(resource),
    }
    snapshots = {
        name: analysis.EvidenceSnapshot(
            str(tmp_path / name), hashlib.sha256(payload).hexdigest(), len(payload), payload
        )
        for name, payload in payloads.items()
    }
    monkeypatch.setattr(analysis, "SadekarTFSProtocol", lambda **kwargs: _TinyProtocol())
    monkeypatch.setattr(
        analysis,
        "render_thresholded",
        lambda protocol, density_tensor, threshold: SimpleNamespace(mask=render_mask.to(density_tensor.device)),
    )
    return config, snapshots, mapped


def test_recompute_cell_small_cpu_seam_runs_full_independent_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, snapshots, _ = _tiny_recompute_fixture(tmp_path, monkeypatch)
    metrics, activity = analysis._recompute_cell(config, snapshots)
    assert metrics["completed_updates"] == 3000
    assert metrics["aggregate"]["mean_iou"] == 1.0
    assert activity == {}


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ("logits", "checkpoint"),
        ("density", "sigmoid"),
        ("hard", "density >= .05"),
        ("metrics", "rerender"),
        ("audit", "audit"),
    ],
)
def test_recompute_cell_rejects_each_terminal_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str, message: str
) -> None:
    config, snapshots, _ = _tiny_recompute_fixture(tmp_path, monkeypatch)
    if artifact in {"logits", "density"}:
        array = np.load(io.BytesIO(snapshots[artifact].payload), allow_pickle=False).copy()
        array.flat[0] = np.nextafter(array.flat[0], np.float32(np.inf))
        payload = _npy_bytes(array)
    elif artifact == "hard":
        array = np.load(io.BytesIO(snapshots[artifact].payload), allow_pickle=False).copy()
        array.flat[0] = 1 - array.flat[0]
        payload = _npy_bytes(array)
    else:
        decoded = json.loads(snapshots[artifact].payload)
        if artifact == "metrics":
            decoded["aggregate"]["mean_iou"] = 0.5
        else:
            decoded["clip_calls"] = 2999
        payload = canonical_json_bytes(decoded)
    snapshots[artifact] = analysis.EvidenceSnapshot("mutated", hashlib.sha256(payload).hexdigest(), len(payload), payload)
    with pytest.raises(ValueError, match=message):
        analysis._recompute_cell(config, snapshots)


def test_recompute_cell_rejects_raw_target_camera_and_rerender_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, snapshots, mapped = _tiny_recompute_fixture(tmp_path, monkeypatch)
    input_path = Path(config["inputs"]["raw_xyz"][0]["path"])
    input_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="raw input hash"):
        analysis._recompute_cell(config, snapshots)

    config, snapshots, mapped = _tiny_recompute_fixture(tmp_path / "camera", monkeypatch)
    monkeypatch.setattr(analysis, "SadekarTFSProtocol", lambda **kwargs: _TinyProtocol(bad_camera=True))
    with pytest.raises(ValueError, match="camera"):
        analysis._recompute_cell(config, snapshots)

    config, snapshots, mapped = _tiny_recompute_fixture(tmp_path / "render", monkeypatch)
    wrong = mapped.clone(); wrong[0, 0, 0] = ~wrong[0, 0, 0]
    monkeypatch.setattr(
        analysis, "render_thresholded",
        lambda protocol, density_tensor, threshold: SimpleNamespace(mask=wrong.to(density_tensor.device)),
    )
    with pytest.raises(ValueError, match="rerender"):
        analysis._recompute_cell(config, snapshots)


def test_task2_binding_checks_current_approved_bytes_and_rejects_wrong_bundle_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = {"implementation": {"bundle_sha256": analysis.TASK2_BUNDLE_SHA256}}
    snapshots = analysis._validate_task2_binding(manifest)
    assert len(snapshots) == len(analysis.TASK2_BINDING)
    with pytest.raises(ValueError, match="source closure"):
        analysis._validate_task2_binding({"implementation": {"bundle_sha256": "0" * 64}})
    forged = dict(analysis.TASK2_BINDING)
    first = next(iter(forged)); forged[first] = "0" * 64
    monkeypatch.setattr(analysis, "TASK2_BINDING", forged)
    with pytest.raises(ValueError, match="byte mismatch"):
        analysis._validate_task2_binding(manifest)


def _minimal_analyze_tree(root: Path) -> dict[str, object]:
    cells = []
    for task in analysis.TASK_ORDER:
        for arm in analysis.PROFILE_ORDER:
            run_dir = root / "runs" / task / arm / "seed_17"
            for relative in (
                "STATUS.json", "checkpoints/step_3000.pt", "artifacts/final_logits.npy",
                "artifacts/final_density.npy", "artifacts/final_hard_005.npy",
                "artifacts/metrics.json", "artifacts/audit.json", "artifacts/resource.json",
                "COMMAND.json", "RUN.json", "environment.json", "sources.json", "inputs.json",
                "checkpoints/LATEST.json",
            ):
                path = run_dir / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"x")
            (run_dir / "artifacts/artifacts.json").write_bytes(canonical_json_bytes({"files_sha256": {}}))
            config = {
                "run": {"root": str(root), "path": str(run_dir)}, "task": task, "arm": arm,
                "seed": 17, "resolution": 256, "updates": 3000,
                "inputs": {"suite": {"path": str(root / "suite")}, "raw_xyz": []},
                "specification": {"path": str(root / "spec")},
            }
            cells.append({"task": task, "arm": arm, "run_path": str(run_dir), "resolved_config": config})
    (root / "suite").write_bytes(b"suite"); (root / "spec").write_bytes(b"spec")
    manifest = {
        "output_root": str(root), "cells": cells, "semantic_sha256": "s" * 64,
        "implementation": {"bundle_sha256": analysis.TASK2_BUNDLE_SHA256},
    }
    path = root / "manifests/ablation10.json"; path.parent.mkdir(parents=True); path.write_bytes(canonical_json_bytes(manifest))
    return manifest


def test_analyze_orchestrates_exact40_csv_report_then_verify_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"; root.mkdir(); manifest = _minimal_analyze_tree(root)
    calls = {"manifest": 0, "binding": 0, "completed": 0, "recompute": 0}
    monkeypatch.setattr(analysis, "validate_manifest", lambda value: calls.__setitem__("manifest", calls["manifest"] + 1) or value)
    monkeypatch.setattr(analysis, "_validate_task2_binding", lambda value: calls.__setitem__("binding", calls["binding"] + 1) or [])
    monkeypatch.setattr(
        analysis, "validate_completed_run",
        lambda config: calls.__setitem__("completed", calls["completed"] + 1)
        or SimpleNamespace(completed_updates=3000, config_sha256="c" * 64, artifacts_sha256="a" * 64),
    )
    def recompute(config, snapshots):
        calls["recompute"] += 1
        arm = config["arm"]
        metric = {"base": 0.94, "j_only": 0.95, "h_only": 0.95, "full": 0.96}[arm]
        activity = {
            "base": {},
            "j_only": {"j": {"eligible": 2100, "zero": 0, "active": 2100}},
            "h_only": {"h": {"eligible": 2100, "zero": 0, "active": 2100}},
            "full": {"j": {"eligible": 2100, "zero": 0, "active": 2100}, "h": {"eligible": 2100, "zero": 0, "active": 2100}},
        }[arm]
        return {
            "aggregate": {"mean_iou": metric, "mean_dice": metric, "mean_recall": 0.98, "mean_fpr": 0.01 if arm != "j_only" else 0.011},
            "per_view": [],
        }, activity
    monkeypatch.setattr(analysis, "_recompute_cell", recompute)
    report = analysis.analyze(root)
    assert report["outcome"]["claim_allowed"] is True
    assert calls["manifest"] == 1 and calls["binding"] == 1 and calls["recompute"] == 40
    assert (root / "analysis/soft_rows.csv").is_file()
    assert (root / "analysis/soft_report.json").is_file()
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in (root / "analysis/soft_rows.csv", root / "analysis/soft_report.json")}
    monkeypatch.setattr(analysis, "atomic_bytes", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("verify wrote")))
    analysis.analyze(root, verify=True)
    assert before == {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in before}
    wrong = dict(manifest); wrong["output_root"] = str(tmp_path / "other")
    (root / "manifests/ablation10.json").write_bytes(canonical_json_bytes(wrong))
    with pytest.raises(ValueError, match="manifest.output_root"):
        analysis.analyze(root)
