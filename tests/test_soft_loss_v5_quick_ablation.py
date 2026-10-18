from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import shadowforge_pipeline.soft_voxel_v5.quick_ablation as v5

from shadowforge_pipeline.soft_voxel_v5.quick_ablation import (
    PROFILE_ORDER,
    TASK_ORDER,
    analyze_ablation,
    background_hard_negative_tail_loss,
    build_matrix,
    classify_summary,
    compose_v5_gradient,
    delivery_lovasz_loss,
    evaluate_binary_masks,
    one_sided_sign_flip_pvalue,
    train_cell,
    v5_ramp,
    validate_v4_base_cell,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_tail_selects_stable_top_fraction_and_has_background_only_support() -> None:
    score = torch.tensor(
        [[[0.2, 0.2, -0.3, 0.1], [0.0, -0.2, -0.1, -0.4]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.tensor(
        [[[False, False, False, False], [False, True, True, True]]]
    )
    loss, details = background_hard_negative_tail_loss(
        score,
        target,
        fraction=0.4,
        margin=0.01,
        temperature=0.02,
        return_details=True,
    )
    loss.backward()
    # Five background pixels -> ceil(.4*5)=2; row-major breaks the 0.2 tie.
    assert details[0]["selected_indices"] == [0, 1]
    assert details[0]["selected_count"] == 2
    assert score.grad is not None
    assert torch.count_nonzero(score.grad[0, 0, :2]).item() == 2
    assert torch.count_nonzero(score.grad[target]).item() == 0


def test_tail_uses_one_pixel_for_small_background_and_stays_finite_at_extremes() -> None:
    score = torch.tensor([[[-1.0e6, 1.0e6]]], dtype=torch.float64)
    target = torch.tensor([[[False, True]]])
    loss, details = background_hard_negative_tail_loss(
        score,
        target,
        fraction=0.05,
        margin=0.01,
        temperature=0.02,
        return_details=True,
    )
    assert details[0]["selected_count"] == 1
    assert torch.isfinite(loss)


def test_ramp_has_exact_boundaries_and_next_float_behavior() -> None:
    below_20 = math.nextafter(0.20, 0.0)
    above_20 = math.nextafter(0.20, 1.0)
    below_30 = math.nextafter(0.30, 0.0)
    above_30 = math.nextafter(0.30, 1.0)
    assert v5_ramp(below_20) == 0.0
    assert v5_ramp(0.20) == 0.0
    assert v5_ramp(above_20) > 0.0
    assert v5_ramp(below_30) < 1.0
    assert v5_ramp(0.30) == 1.0
    assert v5_ramp(above_30) == 1.0


def test_component_scaling_is_exact_fair_and_base_is_not_mutated() -> None:
    base = torch.tensor([3.0, 4.0], dtype=torch.float64)
    jaccard = torch.tensor([0.0, 2.0], dtype=torch.float64)
    tail = torch.tensor([4.0, 0.0], dtype=torch.float64)
    before = base.clone()
    full, full_diag = compose_v5_gradient(
        base, jaccard, tail, progress=0.30, profile="v5_full", return_diagnostics=True
    )
    j_only, j_diag = compose_v5_gradient(
        base, jaccard, None, progress=0.30, profile="jaccard_only", return_diagnostics=True
    )
    h_only, h_diag = compose_v5_gradient(
        base, None, tail, progress=0.30, profile="fp_tail_only", return_diagnostics=True
    )
    assert torch.equal(base, before)
    assert torch.equal(full - h_only, j_only - base)
    assert torch.equal(full - j_only, h_only - base)
    assert j_diag["jaccard_ratio"] == pytest.approx(0.20)
    assert h_diag["tail_ratio"] == pytest.approx(0.10)
    assert full_diag["combined_ratio"] <= 0.30 + 1e-12


def test_zero_auxiliary_norm_records_zero_without_division() -> None:
    result, diagnostics = compose_v5_gradient(
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        None,
        progress=1.0,
        profile="jaccard_only",
        return_diagnostics=True,
    )
    assert torch.equal(result, torch.tensor([1.0]))
    assert diagnostics["jaccard_zero"] is True


def test_matrix_order_is_task_major_profile_minor() -> None:
    matrix = build_matrix(TASK_ORDER, PROFILE_ORDER)
    assert len(matrix) == 40
    assert matrix[:4] == tuple(("connected_01", profile) for profile in PROFILE_ORDER)
    assert matrix[-1] == ("connected_10", "v5_full")


def _fake_v4_cell(root: Path, *, task: str = "connected_01") -> tuple[Path, list[str]]:
    inputs = []
    for axis in "xyz":
        path = root / "inputs" / f"{task}_{axis}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(axis.encode())
        inputs.append(str(path))
    run_dir = root / "runs" / "ablation10" / task / "l12_bal"
    config = {
        "run_identity": {"task_id": task},
        "profile": {"name": "l12_bal"},
        "resolution": 128,
        "steps": 2000,
        "seed": 17,
        "learning_rate": 0.007,
        "shadow_inputs_xyz": [
            {"axis": axis, "path": path} for axis, path in zip("xyz", inputs)
        ],
    }
    _write_json(run_dir / "resolved_config.json", config)
    _write_json(run_dir / "STATUS.json", {"state": "completed", "completed_updates": 2000})
    views = [
        {"tp": 9, "fp": 1, "fn": 1, "tn": 9, "iou": 9 / 11, "dice": 0.9, "recall": 0.9, "fpr": 0.1}
        for _ in range(6)
    ]
    _write_json(
        run_dir / "metrics.json",
        {"thresholds": {"0.05": {"per_view": views, "aggregate": {"mean_iou": 9 / 11, "mean_dice": 0.9, "mean_recall": 0.9, "mean_fpr": 0.1}}}},
    )
    manifest = {
        "matrix": [
            {
                "task_id": task,
                "profile": "l12_bal",
                "run_dir": str(run_dir),
                "resolved_config": config,
            }
        ]
    }
    manifest_path = root / "manifests" / "ablation10.json"
    _write_json(manifest_path, manifest)
    return manifest_path, inputs


def _fake_authenticated_v4(root: Path, config: dict[str, object]) -> SimpleNamespace:
    run_dir = root / "runs/ablation10/connected_01/l12_bal"
    metric_path = run_dir / "metrics.json"
    status_path = run_dir / "STATUS.json"
    return SimpleNamespace(
        config=config,
        config_hash="1" * 64,
        status_sha256=hashlib.sha256(status_path.read_bytes()).hexdigest(),
        artifact_manifest_sha256="2" * 64,
        artifacts_sha256={"metrics.json": hashlib.sha256(metric_path.read_bytes()).hexdigest()},
    )


def test_v4_reuse_validation_checks_exact_config_status_inputs_and_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, inputs = _fake_v4_cell(tmp_path)
    manifest_config = json.loads(manifest.read_text())["matrix"][0]["resolved_config"]
    calls: list[tuple[Path, dict[str, object]]] = []

    def authenticate(run_dir: Path, *, expected_config: dict[str, object]) -> SimpleNamespace:
        calls.append((run_dir, expected_config))
        return _fake_authenticated_v4(tmp_path, expected_config)

    monkeypatch.setattr(v5, "_validate_v4_completed", authenticate, raising=False)
    row = validate_v4_base_cell(tmp_path, manifest, "connected_01")
    metric_path = tmp_path / "runs/ablation10/connected_01/l12_bal/metrics.json"
    assert calls == [(metric_path.parent, manifest_config)]
    assert row["input_paths"] == inputs
    assert row["source_metric_path"] == str(metric_path)
    assert row["source_metric_sha256"] == hashlib.sha256(metric_path.read_bytes()).hexdigest()
    assert row["source_status_sha256"] == hashlib.sha256((metric_path.parent / "STATUS.json").read_bytes()).hexdigest()
    assert row["source_artifact_manifest_sha256"] == "2" * 64
    config_path = tmp_path / "runs/ablation10/connected_01/l12_bal/resolved_config.json"
    config = json.loads(config_path.read_text())
    config["learning_rate"] = 0.008
    _write_json(config_path, config)
    with pytest.raises(ValueError, match="learning_rate"):
        validate_v4_base_cell(tmp_path, manifest, "connected_01")


def test_v4_reuse_propagates_existing_authenticator_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _ = _fake_v4_cell(tmp_path)

    def reject(*args: object, **kwargs: object) -> None:
        raise ValueError("authenticated v4 failure")

    monkeypatch.setattr(v5, "_validate_v4_completed", reject, raising=False)
    with pytest.raises(ValueError, match="authenticated v4 failure"):
        validate_v4_base_cell(tmp_path, manifest, "connected_01")


class _FakeProtocol:
    device = torch.device("cpu")

    def render_all(self, density: torch.Tensor) -> torch.Tensor:
        # Six differentiable 2x2 views from a 2x2x2 volume.
        volume = density[0, 0]
        base = volume.mean(dim=0)
        views = [base, base.transpose(0, 1), volume.mean(dim=1)]
        return torch.stack(views + [torch.flip(view, (-1,)) for view in views])


def _tiny_config(profile: str, steps: int = 2) -> dict[str, object]:
    return {
        "task": "connected_01",
        "profile": profile,
        "resolution": 2,
        "steps": steps,
        "seed": 17,
        "learning_rate": 0.007,
        "device": "cpu",
        "input_paths": ["x", "y", "z"],
    }


def _tiny_targets() -> torch.Tensor:
    base = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return torch.stack([base, base.T, base, torch.flip(base, (-1,)), base.T, base])


class _ZeroAuxProtocol(_FakeProtocol):
    def render_all(self, density: torch.Tensor) -> torch.Tensor:
        if torch.all((density == 0.0) | (density == 0.999)).item():
            linked_zero = density.sum() * 0.0
            return linked_zero + torch.zeros((6, 2, 2), dtype=density.dtype, device=density.device)
        return super().render_all(density)


def test_fake_protocol_training_update_and_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "cell"
    partial = train_cell(
        run_dir,
        _tiny_config("v5_full"),
        protocol=_FakeProtocol(),
        targets=_tiny_targets(),
        max_updates=1,
    )
    assert partial["state"] == "running"
    before = torch.load(run_dir / "checkpoint_latest.pt", map_location="cpu")["logits"]
    completed = train_cell(
        run_dir,
        _tiny_config("v5_full"),
        protocol=_FakeProtocol(),
        targets=_tiny_targets(),
        resume=True,
    )
    after = torch.load(run_dir / "checkpoint_latest.pt", map_location="cpu")["logits"]
    assert completed["state"] == "completed"
    assert not torch.equal(before, after)
    assert set(path.name for path in run_dir.iterdir()) == {
        "checkpoint_latest.pt", "metrics.json", "audit.json", "STATUS.json"
    }
    audit = json.loads((run_dir / "audit.json").read_text())
    assert audit["finite"] is True
    assert audit["active_after_ramp"] is True


def test_well_formed_inactive_auxiliary_completes_for_inconclusive_analysis(tmp_path: Path) -> None:
    run_dir = tmp_path / "inactive"
    completed = train_cell(
        run_dir,
        _tiny_config("jaccard_only", steps=2),
        protocol=_ZeroAuxProtocol(),
        targets=_tiny_targets(),
    )
    assert completed["state"] == "completed"
    audit = json.loads((run_dir / "audit.json").read_text())
    assert audit["finite"] is True
    assert audit["active_after_ramp"] is False
    assert classify_summary(
        {
            "engineering_valid": False,
            "safety_pass": True,
            "mean_delta_iou": 0.01,
            "mean_delta_dice": 0.01,
            "all_success_gates": False,
        }
    ) == "INCONCLUSIVE"


def test_activity_requires_positive_gradient_at_full_ramp() -> None:
    audit = {
        "jaccard_active_updates": 3,
        "tail_active_updates": 0,
        "jaccard_full_ramp_active_updates": 0,
        "tail_full_ramp_active_updates": 0,
    }
    assert v5._active_at_full_ramp(audit, use_jaccard=True, use_tail=False) is False
    audit["jaccard_full_ramp_active_updates"] = 1
    assert v5._active_at_full_ramp(audit, use_jaccard=True, use_tail=False) is True


def _frozen_config(
    *, task: str = "connected_01", profile: str = "v5_full", steps: int = 200,
    resolution: int = 128, seed: int = 17, learning_rate: float = 0.007,
    device: str = "cuda:0", input_paths: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema": "pcraft_soft_loss_v5_config_v1",
        "task": task,
        "profile": profile,
        "resolution": resolution,
        "views": 6,
        "image_size": 256,
        "steps": steps,
        "seed": seed,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "initial_logit": -4.0,
        "gradient_clip": 1.0,
        "threshold": 0.05,
        "input_paths": input_paths or ["/inputs/x.png", "/inputs/y.png", "/inputs/z.png"],
        "device": device,
        "inference_scope": v5.INFERENCE_SCOPE,
    }


def _write_completed_v5(run_dir: Path, config: dict[str, object], *, active: bool = True) -> None:
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    metrics = {
        "schema": "pcraft_soft_loss_v5_metrics_v1",
        "config_hash": config_hash,
        "task": config["task"],
        "profile": config["profile"],
        "steps": config["steps"],
        "mean_iou": 0.9,
        "mean_dice": 0.95,
        "mean_recall": 0.99,
        "mean_fpr": 0.01,
    }
    audit = {
        "schema": "pcraft_soft_loss_v5_audit_v1",
        "config_hash": config_hash,
        "finite": True,
        "active_after_ramp": active,
    }
    run_dir.mkdir(parents=True)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "audit.json", audit)
    torch.save(
        {
            "schema": "pcraft_soft_loss_v5_checkpoint_v1",
            "config": config,
            "config_hash": config_hash,
            "completed_updates": config["steps"],
        },
        run_dir / "checkpoint_latest.pt",
    )
    _write_json(
        run_dir / "STATUS.json",
        {
            "schema": "pcraft_soft_loss_v5_status_v1",
            "state": "completed",
            "config_hash": config_hash,
            "completed_updates": config["steps"],
            "metrics_sha256": hashlib.sha256((run_dir / "metrics.json").read_bytes()).hexdigest(),
            "audit_sha256": hashlib.sha256((run_dir / "audit.json").read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256((run_dir / "checkpoint_latest.pt").read_bytes()).hexdigest(),
        },
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"resolution": 64},
        {"seed": 18},
        {"learning_rate": 0.008},
        {"device": "cuda:1"},
        {"input_paths": ["/other/x.png", "/other/y.png", "/other/z.png"]},
        {"task": "connected_02"},
        {"profile": "jaccard_only"},
        {"steps": 201},
    ],
)
def test_completed_v5_rejects_coherently_rehashed_wrong_frozen_config(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    expected = _frozen_config()
    wrong = {**expected, **mutation}
    run_dir = tmp_path / "cell"
    _write_completed_v5(run_dir, wrong)
    with pytest.raises(ValueError, match="expected config"):
        v5._load_completed_v5(run_dir, expected)


def test_completed_v5_requires_all_artifacts_to_share_expected_config_hash(tmp_path: Path) -> None:
    expected = _frozen_config()
    run_dir = tmp_path / "cell"
    _write_completed_v5(run_dir, expected)
    audit = json.loads((run_dir / "audit.json").read_text())
    audit["config_hash"] = "f" * 64
    _write_json(run_dir / "audit.json", audit)
    status = json.loads((run_dir / "STATUS.json").read_text())
    status["audit_sha256"] = hashlib.sha256((run_dir / "audit.json").read_bytes()).hexdigest()
    _write_json(run_dir / "STATUS.json", status)
    with pytest.raises(ValueError, match="config_hash"):
        v5._load_completed_v5(run_dir, expected)


def test_binary_metric_formulas_store_per_view_counts() -> None:
    target = torch.tensor([[[True, True], [False, False]]]).expand(6, -1, -1)
    mask = torch.tensor([[[True, False], [True, False]]]).expand(6, -1, -1)
    result = evaluate_binary_masks(mask, target)
    assert result["per_view"][0] == {
        "tp": 1, "fp": 1, "fn": 1, "tn": 1,
        "iou": 1 / 3, "dice": 0.5, "recall": 0.5, "fpr": 0.5,
    }
    assert result["mean_iou"] == pytest.approx(1 / 3)


def test_exact_sign_flip_is_inclusive_and_has_no_pseudocount() -> None:
    # All-positive equal deltas: only the all-positive assignment reaches observed.
    assert one_sided_sign_flip_pvalue([1.0] * 10) == 1 / 1024
    assert one_sided_sign_flip_pvalue([0.0] * 10) == 1.0


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"engineering_valid": False}, "INCONCLUSIVE"),
        ({"mean_delta_iou": -0.01}, "HARMFUL"),
        ({"safety_pass": False, "mean_delta_iou": 0.01, "mean_delta_dice": 0.01}, "POSITIVE_BUT_SUBSTANTIVE_FAIL"),
        ({"mean_delta_iou": 0.02, "mean_delta_dice": 0.01, "all_success_gates": True}, "V5_SUCCESS"),
        ({"mean_delta_iou": 0.001, "mean_delta_dice": 0.001}, "POSITIVE_BUT_SUBSTANTIVE_FAIL"),
    ],
)
def test_exact_labels(updates: dict[str, object], expected: str) -> None:
    summary = {
        "engineering_valid": True,
        "safety_pass": True,
        "mean_delta_iou": 0.0,
        "mean_delta_dice": 0.0,
        "all_success_gates": False,
    }
    summary.update(updates)
    assert classify_summary(summary) == expected


def test_delivery_lovasz_matches_existing_per_view_mean() -> None:
    score = torch.tensor([[[0.05, -0.05], [0.10, -0.10]]], dtype=torch.float64)
    labels = torch.tensor([[[True, False], [False, True]]])
    value = delivery_lovasz_loss(score, labels)
    assert value.item() == pytest.approx(2.0)


def test_analyzer_refuses_incomplete_matrix_without_writing_report(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest, _ = _fake_v4_cell(source)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="40"):
        analyze_ablation(output, source, manifest_path=manifest)
    assert not (output / "analysis" / "report.json").exists()
