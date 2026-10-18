"""Contract tests for the project-local P-CRAFT soft-only trainer."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from shadowforge_pipeline.soft_voxel_v2.data import load_targets, normalize_target
from shadowforge_pipeline.soft_voxel_v2.metrics import evaluate_rendered_views
from shadowforge_pipeline.soft_voxel_v2.losses import (
    boundary_charbonnier_loss,
    boundary_weights,
    resolve_profile,
)
from shadowforge_pipeline.soft_voxel_v2.trainer import (
    ALLOWED_ARTIFACT_PATTERNS,
    _manifest,
    build_checkpoint_payload,
    build_parser,
    config_hash,
    legacy_full_loss,
    proxy_execution_semantics,
    training_auxiliary_projection,
    validate_args,
)


def _valid_argv(tmp_path: Path) -> list[str]:
    images = [str(tmp_path / f"view_{index}.png") for index in range(3)]
    for index, path in enumerate(images):
        image = np.full((8, 8), 32 + 32 * index, dtype=np.uint8)
        assert cv2.imwrite(path, image)
    return [
        "cpu", str(tmp_path / "run"), "2", "0.01", "7",
        "-sdlist", *images,
        "-mr", "1", "-tsf", "1", "-tsf4", "0",
        "-cam", "ortho", "-imsz", "256",
        "--res", "128", "-eres", "128", "--control-res", "0",
        "--sadekar-zdist", "1.7", "--sadekar-volume-extent", "1.7",
        "--sadekar-npts-per-ray", "150",
        "--sadekar-density-threshold", "0.05", "--seed", "17",
        "--profile", "new_full", "--soft-only-eval",
    ]


def test_cli_fails_closed_without_soft_only_eval(tmp_path):
    parser = build_parser()
    argv = _valid_argv(tmp_path)
    argv.remove("--soft-only-eval")
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


@pytest.mark.parametrize(
    ("flag", "invalid"),
    [
        ("-mr", "0"),
        ("-tsf", "0"),
        ("-tsf4", "1"),
        ("-imsz", "128"),
        ("--control-res", "16"),
        ("--sadekar-zdist", "2.5"),
        ("--sadekar-volume-extent", "2.0"),
        ("--sadekar-npts-per-ray", "64"),
        ("--sadekar-density-threshold", "0.1"),
        ("--seed", "42"),
    ],
)
def test_protocol_validation_rejects_non_preregistered_values(tmp_path, flag, invalid):
    argv = _valid_argv(tmp_path)
    argv[argv.index(flag) + 1] = invalid
    args = build_parser().parse_args(argv)
    with pytest.raises(ValueError):
        validate_args(args)


def test_protocol_requires_three_inputs_and_matching_export_resolution(tmp_path):
    args = build_parser().parse_args(_valid_argv(tmp_path))
    validate_args(args)

    args.shadow_files = args.shadow_files[:2]
    with pytest.raises(ValueError):
        validate_args(args)

    args = build_parser().parse_args(_valid_argv(tmp_path))
    args.export_res = 256
    with pytest.raises(ValueError):
        validate_args(args)


def test_mirror_targets_are_raw_flips_then_independently_normalized(monkeypatch):
    from shadowforge_pipeline.soft_voxel_v2 import data

    originals = {
        "a": np.pad(np.full((3, 1), 255, np.uint8), ((1, 2), (1, 4))),
        "b": np.pad(np.full((2, 2), 255, np.uint8), ((2, 2), (1, 3))),
        "c": np.pad(np.full((1, 3), 255, np.uint8), ((3, 2), (2, 1))),
    }
    monkeypatch.setattr(data.cv2, "imread", lambda path, _: originals[path].copy())

    targets = load_targets(["a", "b", "c"], mirror=True, size=16, device=torch.device("cpu"))
    expected = [normalize_target(originals[name], 16) for name in ("a", "b", "c")]
    expected += [
        normalize_target(data.cv2.flip(originals[name], 1), 16)
        for name in ("a", "b", "c")
    ]
    expected_tensor = torch.from_numpy(np.stack(expected).astype(np.float32) / 255.0)

    assert targets.shape == (6, 16, 16)
    assert torch.equal(targets, expected_tensor)


def test_legacy_full_formula_is_exact_mean_view_control():
    target = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    prediction = torch.tensor([[[0.8, 0.2]], [[0.3, 0.6]]], requires_grad=True)
    result = legacy_full_loss(prediction, target, disable_l1=False, disable_l2=False, disable_ssim=False)

    per_l2 = (prediction - target).square().mean(dim=(1, 2))
    per_l1 = (prediction - target).abs().mean(dim=(1, 2))
    per_dice = 1.0 - (
        2.0 * (prediction * target).sum(dim=(1, 2)) + 1.0
    ) / (prediction.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1.0)
    per_fp = (prediction * (1.0 - target)).mean(dim=(1, 2))
    expected = (per_l2 + 0.35 * per_l1 + 0.80 * per_dice + 0.18 * result.per_view_ssim + 0.35 * per_fp).mean()

    assert torch.allclose(result.total, expected)
    assert torch.allclose(result.per_view_total, per_l2 + 0.35 * per_l1 + 0.80 * per_dice + 0.18 * result.per_view_ssim + 0.35 * per_fp)


def test_metrics_use_density_threshold_renders_not_projection_cutoffs():
    target = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    raw = torch.tensor([[[0.9, 0.4], [0.2, 0.1]]])
    proxy = torch.tensor([[[0.8, 0.3], [0.1, 0.0]]])
    rendered = {
        0.04: torch.tensor([[[True, True], [False, False]]]),
        0.05: torch.tensor([[[True, False], [False, False]]]),
        0.06: torch.tensor([[[False, False], [False, False]]]),
    }
    report = evaluate_rendered_views(raw, target, rendered, proxy_prediction=proxy)

    assert report["density_threshold_semantics"] is True
    assert report["thresholds"]["0.04"]["per_view"][0]["fp"] == 1
    assert report["thresholds"]["0.05"]["per_view"][0]["binary_iou"] == pytest.approx(1.0)
    assert report["thresholds"]["0.06"]["per_view"][0]["fn"] == 1
    assert "mean_raw_continuous_iou" in report["aggregate"]
    assert "mean_proxy_continuous_iou" in report["aggregate"]
    json.dumps(report)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_density_threshold_metrics_reject_invalid_thresholds(threshold):
    image = torch.zeros((1, 2, 2))
    with pytest.raises(ValueError):
        evaluate_rendered_views(image, image, {threshold: image.bool()}, primary_threshold=threshold)


def test_density_threshold_metric_keys_must_be_unique_after_serialization():
    image = torch.zeros((1, 2, 2))
    with pytest.raises(ValueError):
        evaluate_rendered_views(
            image,
            image,
            {0.039: image.bool(), 0.041: image.bool()},
            primary_threshold=0.041,
        )


def test_boundary_uses_inner_contour_and_empty_contour_contributes_zero():
    target = torch.zeros((2, 9, 9))
    target[0, 3:6, 3:6] = 1.0
    weights = boundary_weights(target, sigma_px=2.0)
    # Row 3 is the inner target contour; row 2 is outside and must not itself
    # be labeled as contour (the previous XOR morphology did that).
    assert weights[0, 3, 4] == pytest.approx(1.0)
    assert weights[0, 2, 4] < 1.0
    assert torch.count_nonzero(weights[1]) == 0

    prediction = torch.ones_like(target)
    per_view = boundary_charbonnier_loss(prediction, target, weights=weights)
    assert per_view[0] > 0
    assert per_view[1] == pytest.approx(0.0)


def test_disable_proxy_really_removes_q_and_routes_auxiliary_terms_to_raw(tmp_path):
    argv = _valid_argv(tmp_path) + ["--disable-proxy"]
    args = build_parser().parse_args(argv)
    validate_args(args)
    semantics = proxy_execution_semantics(args)
    assert semantics == {
        "threshold_proxy_enabled": False,
        "proxy_blend_enabled": False,
        "auxiliary_proxy_required": False,
        "auxiliary_domain": "raw_projection",
    }

    class RendererThatMustNotRun:
        def render_all(self, _density):
            raise AssertionError("Q renderer ran despite --disable-proxy")

    density = torch.full((1, 1, 2, 2, 2), 0.1)
    raw = torch.full((6, 2, 2), 0.2)
    auxiliary = training_auxiliary_projection(
        density, raw, args=args, protocol=RendererThatMustNotRun(), progress=0.5
    )
    assert auxiliary is raw


@pytest.mark.parametrize("profile", ["legacy_full", "l12_plain", "l12_bal"])
def test_non_new_profiles_never_construct_diagnostic_q(tmp_path, profile):
    args = build_parser().parse_args(_valid_argv(tmp_path))
    args.profile = profile
    semantics = proxy_execution_semantics(args)
    assert semantics == {
        "threshold_proxy_enabled": False,
        "proxy_blend_enabled": False,
        "auxiliary_proxy_required": False,
        "auxiliary_domain": "none",
    }


def test_legacy_manifest_records_exact_control_weights(tmp_path):
    args = build_parser().parse_args(_valid_argv(tmp_path))
    args.profile = "legacy_full"
    profile = resolve_profile("legacy_full")
    manifest = _manifest(args, {}, "f" * 64, proxy_execution_semantics(args), profile)
    assert manifest["loss_weights"] == {
        "l2": 1.0,
        "l1": 0.35,
        "dice": 0.80,
        "ssim": 0.18,
        "false_positive": 0.35,
        "gate": 0.0,
        "boundary": 0.0,
        "late_ssim": 0.0,
    }


def test_formal_metric_api_is_fixed_to_two_pixel_boundary_tolerance():
    image = torch.zeros((1, 2, 2))
    with pytest.raises(ValueError):
        evaluate_rendered_views(
            image,
            image,
            {0.05: image.bool()},
            primary_threshold=0.05,
            boundary_tolerance_px=1.0,
        )


def test_config_hash_is_order_stable_and_sensitive():
    first = config_hash({"profile": "new_full", "res": 128, "nested": {"b": 2, "a": 1}})
    reordered = config_hash({"nested": {"a": 1, "b": 2}, "res": 128, "profile": "new_full"})
    changed = config_hash({"profile": "l12_bal", "res": 128, "nested": {"a": 1, "b": 2}})
    assert first == reordered
    assert first != changed
    assert len(first) == 64


def test_resolved_config_hash_changes_when_authoritative_source_changes(
    tmp_path, monkeypatch
):
    """Catch resumes/skips that ignore a changed local soft trainer."""

    from shadowforge_pipeline.soft_voxel_v2 import trainer

    source_root = tmp_path / "soft_voxel_v2"
    source_root.mkdir()
    (source_root / "trainer.py").write_text("version = 1\n", encoding="utf-8")
    (source_root / "losses.py").write_text("loss = 'v1'\n", encoding="utf-8")
    monkeypatch.setattr(trainer, "AUTHORITATIVE_SOURCE_ROOT", source_root)
    args = build_parser().parse_args(_valid_argv(tmp_path))

    first = trainer.resolved_config(args)
    first_digest = config_hash(first)
    expected_trainer_digest = hashlib.sha256(b"version = 1\n").hexdigest()
    assert first["implementation"]["files_sha256"]["trainer.py"] == expected_trainer_digest

    (source_root / "trainer.py").write_text("version = 2\n", encoding="utf-8")
    second = trainer.resolved_config(args)
    assert second["implementation"]["bundle_sha256"] != first["implementation"]["bundle_sha256"]
    assert config_hash(second) != first_digest


def test_resolved_config_hashes_shadow_inputs_in_declared_order(tmp_path):
    """Catch content-changing inputs that retain the same absolute paths."""

    from shadowforge_pipeline.soft_voxel_v2 import trainer

    args = build_parser().parse_args(_valid_argv(tmp_path))
    first = trainer.resolved_config(args)
    expected = [
        hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in args.shadow_files
    ]
    assert first["shadow_files_sha256"] == expected

    Path(args.shadow_files[1]).write_bytes(b"changed input bytes")
    second = trainer.resolved_config(args)
    assert second["shadow_files_sha256"][0] == first["shadow_files_sha256"][0]
    assert second["shadow_files_sha256"][1] != first["shadow_files_sha256"][1]
    assert config_hash(second) != config_hash(first)


def test_resolved_config_rejects_missing_shadow_input(tmp_path):
    from shadowforge_pipeline.soft_voxel_v2 import trainer

    args = build_parser().parse_args(_valid_argv(tmp_path))
    missing = Path(args.shadow_files[2])
    missing.unlink()
    with pytest.raises(FileNotFoundError, match=missing.name):
        trainer.resolved_config(args)


def test_trainer_has_no_forbidden_pipeline_imports_or_outputs():
    forbidden = ("repair", "carv", "core_shell", "delivery", "mesh", "pino", "fem")
    package_root = Path(__file__).parents[1] / "shadowforge_pipeline" / "soft_voxel_v2"
    sources = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in package_root.glob("*.py")
        if path.name in {"trainer.py", "data.py", "protocol.py"}
    )
    for token in forbidden:
        assert f"import {token}" not in sources
        assert f"from {token}" not in sources
    assert all(not pattern.lower().endswith((".obj", ".stl")) for pattern in ALLOWED_ARTIFACT_PATTERNS)
    assert not any(name in sys.modules for name in ("trimesh", "pymeshfix", "pino_ranked_fem"))


def test_resume_checkpoint_payload_contains_reproducibility_state():
    logits = torch.zeros((1, 1, 2, 2, 2), requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.1)
    payload = build_checkpoint_payload(
        logits=logits,
        optimizer=optimizer,
        step=3,
        resolved_config={"profile": "new_full", "res": 128},
        config_digest="a" * 64,
    )
    assert {"logits", "optimizer", "step", "config", "config_hash", "rng_state"} <= set(payload)
    assert {"python", "numpy", "torch"} <= set(payload["rng_state"])
    assert payload["step"] == 3


def test_resume_restores_last_record_for_final_step_export():
    """Catch a final-step resume that emits null loss/gradient metrics."""

    from shadowforge_pipeline.soft_voxel_v2 import trainer

    saved_logits = torch.full((1, 1, 2, 2, 2), 0.75, requires_grad=True)
    saved_optimizer = torch.optim.Adam([saved_logits], lr=0.1)
    config = {"profile": "new_full", "res": 128}
    digest = "b" * 64
    last_record = {
        "step": 2000,
        "loss": 0.125,
        "gradient_norm_pre_clip": 0.03125,
    }
    payload = build_checkpoint_payload(
        logits=saved_logits,
        optimizer=saved_optimizer,
        step=2000,
        resolved_config=config,
        config_digest=digest,
        last_record=last_record,
    )

    resumed_logits = torch.zeros_like(saved_logits, requires_grad=True)
    resumed_optimizer = torch.optim.Adam([resumed_logits], lr=0.1)
    start_step, restored_record = trainer.restore_checkpoint_payload(
        payload,
        logits=resumed_logits,
        optimizer=resumed_optimizer,
        expected_config=config,
        expected_digest=digest,
    )

    assert start_step == 2001
    assert restored_record == last_record
    assert torch.equal(resumed_logits, saved_logits)


def test_incompatible_resume_preserves_existing_audit_files_byte_for_byte(tmp_path):
    """Catch preflight failures that overwrite the original run audit trail."""

    from shadowforge_pipeline.soft_voxel_v2 import trainer

    argv = _valid_argv(tmp_path)
    output = Path(argv[1])
    output.mkdir()
    sentinels = {
        "COMMAND.json": b'{"original":"command"}\n',
        "run_manifest.json": b'{"original":"manifest"}\n',
        "STATUS.json": b'{"original":"status"}\n',
    }
    for name, content in sentinels.items():
        (output / name).write_bytes(content)

    logits = torch.zeros((1, 1, 2, 2, 2), requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.1)
    incompatible = build_checkpoint_payload(
        logits=logits,
        optimizer=optimizer,
        step=1,
        resolved_config={"profile": "different"},
        config_digest="c" * 64,
        last_record={"step": 1, "loss": 1.0, "gradient_norm_pre_clip": 1.0},
    )
    torch.save(incompatible, output / "checkpoint_latest.pt")

    with pytest.raises(ValueError, match="configuration does not match"):
        trainer.main([*argv, "--resume"])

    for name, content in sentinels.items():
        assert (output / name).read_bytes() == content
