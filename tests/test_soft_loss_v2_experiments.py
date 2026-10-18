"""Contract tests for the staged soft-loss-v2 experiment and statistics tools."""

from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.run_soft_loss_v2 as runner
from experiments.run_soft_loss_v2 import (
    APPROVED_STATUS,
    BENCHMARK,
    RunSpec,
    build_stage_specs,
    is_completed_run,
    load_frozen_benchmark,
    runner_log_path,
    trainer_argv,
    write_master_manifest,
    write_stage_manifest,
)
from experiments.summarize_soft_loss_v2 import (
    assess_component,
    choose_transfer_configuration,
    confirm_development_gate,
    exact_sign_flip_test,
    export_paper_table,
    holm_adjust,
    paired_bootstrap,
    transfer_gate,
    validate_formal_rows,
    validate_formal_provenance,
    validate_registered_input_hashes,
    validate_stage_registration,
    write_dev_selection,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _frozen_manifest(tmp_path: Path) -> Path:
    cases = []
    for index in range(1, 11):
        case_id = f"connected_{index:02d}"
        old = "/home/lin/xvxinyao/ShadowForge/results/frozen/inputs"
        cases.append(
            {
                "case_id": case_id,
                "target_paths_xyz": [
                    f"{old}/{case_id}_view_x.png",
                    f"{old}/{case_id}_view_y.png",
                    f"{old}/{case_id}_view_z.png",
                ],
                # Deliberately swapped: this must never be selected by v2.
                "sadekar_tfs_targets": [
                    f"{old}/{case_id}_view_y.png",
                    f"{old}/{case_id}_view_x.png",
                    f"{old}/{case_id}_view_z.png",
                ],
            }
        )
    path = tmp_path / "benchmark.json"
    _write_json(
        path,
        {
            "schema": "fixture",
            "status": APPROVED_STATUS,
            "protocol": {"axes": ["x", "y", "z"]},
            "cases": cases,
        },
    )
    return path


def _frozen_config(
    tmp_path: Path, *, ssim: float = 0.05, source: str = "transfer256",
) -> Path:
    path = tmp_path / "frozen_config.json"
    _write_json(
        path,
        {
            "schema": "pcraft_soft_loss_v2_frozen_config_v1",
            "status": APPROVED_STATUS,
            "lambda_g": 0.25,
            "lambda_b": 0.10,
            "late_ssim_weight": ssim,
            "source_stage": source,
        },
    )
    return path


def test_frozen_benchmark_uses_xyz_not_sadekar_order_and_rewrites_project(tmp_path):
    loaded = load_frozen_benchmark(_frozen_manifest(tmp_path), require_files=False)
    first = loaded[0]
    assert first.case_id == "connected_01"
    assert [Path(path).name for path in first.targets] == [
        "connected_01_view_x.png",
        "connected_01_view_y.png",
        "connected_01_view_z.png",
    ]
    assert all("/P-CRAFT/" in path for path in first.targets)
    assert all("/ShadowForge/" not in path for path in first.targets)


def test_stage_matrix_has_preregistered_protocol_and_variants(tmp_path):
    frozen = _frozen_config(tmp_path)
    benchmark = _frozen_manifest(tmp_path)
    smoke = build_stage_specs(
        "smoke128", output_root=tmp_path / "out", benchmark_path=benchmark
    )
    assert {spec.variant for spec in smoke} >= {
        "legacy_full",
        "l12_plain",
        "l12_bal",
        "new_full",
        "new_full_no_proxy",
        "new_full_no_gate",
        "new_full_no_boundary",
    }
    assert len(smoke) == 8
    assert len({(spec.task, spec.variant) for spec in smoke}) == len(smoke)
    assert {spec.task for spec in smoke} == {"arcane_casters"}
    assert all((spec.resolution, spec.steps, spec.seed) == (128, 200, 17) for spec in smoke)

    formal = build_stage_specs(
        "formal256",
        output_root=tmp_path / "out",
        benchmark_path=benchmark,
        frozen_config_path=frozen,
        require_files=False,
    )
    variants = {spec.variant for spec in formal}
    assert {
        "legacy_full", "l12_plain", "l12_bal", "new_full",
        "new_full_no_proxy", "new_full_no_gate", "new_full_no_boundary",
        "new_full_mean", "new_full_no_ssim", "new_full_no_l1", "new_full_no_l2",
    } == variants
    assert len(formal) == 10 * len(variants)
    assert len({spec.run_dir for spec in formal}) == len(formal)
    assert all(spec.paper_eligible for spec in formal)
    assert all((spec.resolution, spec.export_resolution, spec.steps) == (256, 256, 2000) for spec in formal)
    assert all(spec.control_resolution == 0 and spec.soft_only for spec in formal)
    assert formal[0].run_dir == (
        tmp_path / "out" / "formal256" / "steps_2000" /
        formal[0].task / formal[0].variant / "seed_17"
    )


def test_registered_stage_counts_and_run_directories_are_exact(tmp_path):
    benchmark = _frozen_manifest(tmp_path)
    selection = tmp_path / "selection.json"
    _write_json(selection, {"selected_candidate": {
        "lambda_g": .25, "lambda_b": .10, "late_ssim_weight": 0.0,
    }})
    transfer_config = _frozen_config(tmp_path, source="dev128_confirm")
    expected = {
        "smoke128": 8, "dev128_screen": 27,
        "dev128_confirm": 9, "transfer256": 9,
    }
    for stage, count in expected.items():
        specs = build_stage_specs(
            stage, output_root=tmp_path / "out", benchmark_path=benchmark,
            selection_path=selection, frozen_config_path=transfer_config,
        )
        assert len(specs) == count
        assert len({spec.run_dir for spec in specs}) == count


def test_manifests_are_content_addressed_immutable_and_disclose_best_effort(tmp_path):
    benchmark_copy = tmp_path / "benchmark.json"
    benchmark_copy.write_bytes(BENCHMARK.read_bytes())
    output = tmp_path / "out"
    master = write_master_manifest(output, benchmark_copy)
    first_bytes = master.read_bytes()
    first = json.loads(first_bytes)
    assert len(first["benchmark_input_sha256"]) == 30
    assert len(first["benchmark_sha256"]) == 64
    assert first["determinism"] == "fixed-seed single-run best-effort"
    assert first["bitwise_reproducible"] is False
    assert "grid_sampler_3d backward" in first["nondeterminism_reason"]
    assert write_master_manifest(output, benchmark_copy) == master
    assert master.read_bytes() == first_bytes

    changed = json.loads(benchmark_copy.read_text())
    changed["purpose"] = "semantic drift"
    _write_json(benchmark_copy, changed)
    with pytest.raises(FileExistsError):
        write_master_manifest(output, benchmark_copy)

    specs = build_stage_specs("smoke128", output_root=tmp_path / "stage-out")
    stage_path = write_stage_manifest(
        "smoke128", specs, tmp_path / "stage-out", benchmark_path=BENCHMARK,
    )
    stage_bytes = stage_path.read_bytes()
    assert write_stage_manifest(
        "smoke128", specs, tmp_path / "stage-out", benchmark_path=BENCHMARK,
    ) == stage_path
    assert stage_path.read_bytes() == stage_bytes
    stage = json.loads(stage_path.read_text())
    assert stage["benchmark_sha256"]
    assert stage["frozen_config_hash"] is None
    assert all(len(run["input_sha256"]) == 3 for run in stage["runs"])
    assert stage["determinism"] == "fixed-seed single-run best-effort"
    with pytest.raises(FileExistsError):
        write_stage_manifest(
            "smoke128", [replace(specs[0], learning_rate=.008), *specs[1:]],
            tmp_path / "stage-out", benchmark_path=BENCHMARK,
        )


def test_summary_rehashes_inputs_and_benchmark_instead_of_trusting_stored_hashes(tmp_path):
    inputs = []
    for axis in "xyz":
        path = tmp_path / f"view_{axis}.png"
        path.write_bytes(f"input-{axis}".encode())
        inputs.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    registration = {"input_sha256": inputs}
    validate_registered_input_hashes(registration)
    (tmp_path / "view_y.png").write_bytes(b"replaced in place")
    with pytest.raises(ValueError, match="input SHA256"):
        validate_registered_input_hashes(registration)

    benchmark = tmp_path / "benchmark.json"
    benchmark.write_bytes(b'{"status":"APPROVED_FROZEN"}\n')
    digest = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    master = {"benchmark_path": str(benchmark), "benchmark_sha256": digest}
    stage = {"benchmark_path": str(benchmark), "benchmark_sha256": digest}
    validate_formal_provenance(master, stage, benchmark)
    benchmark.write_bytes(b'{"status":"changed"}\n')
    with pytest.raises(ValueError, match="benchmark SHA256"):
        validate_formal_provenance(master, stage, benchmark)


def test_development_confirmation_and_transfer_fail_closed_without_selection_or_freeze(tmp_path):
    benchmark = _frozen_manifest(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_stage_specs(
            "dev128_confirm", output_root=tmp_path / "out", benchmark_path=benchmark
        )
    with pytest.raises((FileNotFoundError, ValueError)):
        build_stage_specs(
            "transfer256", output_root=tmp_path / "out", benchmark_path=benchmark
        )


def test_confirmation_compares_both_ssim_choices_and_transfer_never_duplicates_them(tmp_path):
    benchmark = _frozen_manifest(tmp_path)
    selection = tmp_path / "selection.json"
    _write_json(
        selection,
        {
            "selected_candidate": {
                "lambda_g": 0.25, "lambda_b": 0.10, "late_ssim_weight": 0.0,
            }
        },
    )
    confirm = build_stage_specs(
        "dev128_confirm", output_root=tmp_path / "out", benchmark_path=benchmark,
        selection_path=selection,
    )
    assert {spec.variant for spec in confirm} == {
        "l12_bal", "new_full_no_ssim", "new_full_ssim",
    }
    assert {spec.late_ssim_weight for spec in confirm if spec.profile == "new_full"} == {0.0, 0.05}

    frozen = _frozen_config(tmp_path, ssim=0.0, source="dev128_confirm")
    transfer = build_stage_specs(
        "transfer256", output_root=tmp_path / "out", benchmark_path=benchmark,
        frozen_config_path=frozen,
    )
    by_variant = {spec.variant: spec for spec in transfer if spec.task == "arcane_casters"}
    assert by_variant["new_full"].late_ssim_weight == 0.0
    assert by_variant["new_full_ssim"].late_ssim_weight == 0.05


def test_formal_rejects_config_frozen_before_256_transfer(tmp_path):
    with pytest.raises(ValueError, match="transfer256"):
        build_stage_specs(
            "formal256", output_root=tmp_path / "out",
            benchmark_path=_frozen_manifest(tmp_path),
            frozen_config_path=_frozen_config(tmp_path, source="dev128_confirm"),
            require_files=False,
        )


def test_trainer_command_contains_fixed_protocol_and_ablation_flags(tmp_path):
    spec = RunSpec(
        stage="formal256", task="connected_01", variant="new_full_no_gate",
        targets=("x.png", "y.png", "z.png"), resolution=256,
        export_resolution=256, steps=2000, profile="new_full",
        output_root=tmp_path, disable_gate=True, lambda_g=0.25, lambda_b=0.1,
        late_ssim_weight=0.05, paper_eligible=True,
    )
    argv = trainer_argv(spec)
    assert argv[:5] == ["cuda:0", str(spec.run_dir), "2000", "0.007", "1"]
    assert argv[argv.index("-sdlist") + 1 : argv.index("-sdlist") + 4] == [
        "x.png", "y.png", "z.png"
    ]
    for pair in (
        ("-mr", "1"), ("-tsf", "1"), ("-tsf4", "0"),
        ("-imsz", "256"), ("--res", "256"), ("-eres", "256"),
        ("--control-res", "0"), ("--sadekar-zdist", "1.7"),
        ("--seed", "17"),
    ):
        assert argv[argv.index(pair[0]) + 1] == pair[1]
    assert "--soft-only-eval" in argv and "--export-final-step" in argv
    assert "--disable-gate" in argv


def test_resume_skip_requires_command_status_metrics_and_same_hash(tmp_path):
    run = tmp_path / "run"
    digest = "a" * 64
    _write_json(run / "COMMAND.json", {"config_hash": digest})
    _write_json(
        run / "STATUS.json",
        {"state": "completed", "config_hash": digest, "last_completed_step": 2000},
    )
    per_view = [
        {
            "tp": 1, "fp": 0, "fn": 0, "tn": 1,
            "continuous_iou": 1.0, "binary_iou": 1.0, "dice": 1.0,
            "ssim": 1.0, "fpr": 0.0, "recall": 1.0,
            "boundary_f1_2px": 1.0, "soft_binary_iou_gap": 0.0,
        }
        for _ in range(6)
    ]
    aggregate = {
        "mean_binary_iou": 1.0, "worst_view_binary_iou": 1.0,
        "mean_fpr": 0.0, "mean_recall": 1.0, "mean_dice": 1.0,
        "mean_ssim": 1.0, "mean_boundary_f1_2px": 1.0,
        "mean_continuous_iou": 1.0, "mean_soft_binary_iou_gap": 0.0,
    }
    metrics = {
        "schema": "pcraft_soft_loss_v2_metrics_v1", "config_hash": digest,
        "steps": 2000, "final_step_exported": True, "final_loss": 0.1,
        "final_gradient_norm_pre_clip": 0.2,
        "density_threshold_semantics": True, "primary_threshold": 0.05,
        "per_view": per_view, "aggregate": aggregate,
        "thresholds": {
            threshold: {"per_view": per_view, "aggregate": aggregate}
            for threshold in ("0.04", "0.05", "0.06")
        },
    }
    _write_json(run / "metrics.json", metrics)
    _write_json(
        run / "run_manifest.json",
        {
            "schema": "pcraft_soft_loss_v2_run_v1", "config_hash": digest,
            "scope": {"soft_only": True, "final_step_export_only": True},
            "protocol": {
                "density_threshold_semantics": True,
                "density_thresholds": [0.04, 0.05, 0.06],
            },
        },
    )
    for name in (
        "checkpoint_final.pt", "occupancy_probability.npy",
        "occupancy_binary_tau_0p04.npy", "occupancy_binary_tau_0p05.npy",
        "occupancy_binary_tau_0p06.npy",
    ):
        (run / name).write_bytes(b"artifact")
    for view in range(6):
        for name in (
            f"gt_view_{view}.png", f"continuous_view_{view}.png",
            f"binary_tau_0p04_view_{view}.png", f"binary_tau_0p05_view_{view}.png",
            f"binary_tau_0p06_view_{view}.png",
        ):
            (run / name).write_bytes(b"png")
    assert is_completed_run(run, digest)
    del metrics["thresholds"]["0.06"]
    _write_json(run / "metrics.json", metrics)
    assert not is_completed_run(run, digest)
    _write_json(run / "metrics.json", {**metrics, "thresholds": {
        **metrics["thresholds"], "0.06": {"per_view": per_view, "aggregate": aggregate}
    }})
    (run / "checkpoint_final.pt").unlink()
    assert not is_completed_run(run, digest)


def test_runner_logs_outside_artifact_directory_and_does_not_precreate_run(monkeypatch, tmp_path):
    spec = RunSpec(
        stage="smoke128", task="arcane_casters", variant="l12_bal",
        targets=("x", "y", "z"), resolution=128, export_resolution=128,
        steps=200, profile="l12_bal", output_root=tmp_path,
    )
    checks = iter((False, True))
    monkeypatch.setattr(runner, "expected_config_hash", lambda _: "a" * 64)
    monkeypatch.setattr(runner, "is_completed_run", lambda *_: next(checks))

    def fake_run(*args, **kwargs):
        assert not spec.run_dir.exists()
        assert Path(kwargs["stdout"].name) == runner_log_path(spec)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.run_specs([spec], dry_run=False, resume=False, max_runs=None)
    assert not spec.run_dir.exists()
    assert runner_log_path(spec).is_file()


def test_transfer_gate_implements_all_preregistered_alternatives():
    baseline = {
        "a": {"mean_binary_iou": .900, "recall": .980, "worst_view_iou": .800, "fpr": .020, "boundary_f1": .850},
        "b": {"mean_binary_iou": .910, "recall": .990, "worst_view_iou": .810, "fpr": .020, "boundary_f1": .860},
    }
    candidate = {
        "a": {"mean_binary_iou": .900, "recall": .978, "worst_view_iou": .803, "fpr": .019, "boundary_f1": .850},
        "b": {"mean_binary_iou": .910, "recall": .988, "worst_view_iou": .813, "fpr": .019, "boundary_f1": .860},
    }
    report = transfer_gate(candidate, baseline)
    assert report["passed"] is True
    assert report["primary_improvement"]["worst_view_iou"] >= .002
    candidate["a"]["recall"] = .974
    assert transfer_gate(candidate, baseline)["passed"] is False


def test_dev_selection_parses_multi_character_ssim_token(tmp_path):
    rows = []
    for task in ("a", "b", "c"):
        baseline = {
            "task": task, "variant": "l12_bal", "mean_binary_iou": .90,
            "recall": .98, "worst_view_iou": .80, "fpr": .02,
            "boundary_f1": .85,
        }
        candidate = {
            **baseline,
            "variant": "new_full_g0p25_b0p10_ssim0p05",
            "worst_view_iou": .803,
        }
        rows.extend((baseline, candidate))
    selection = write_dev_selection(rows, tmp_path / "selection.json")
    assert selection["status"] == "SELECTED_FOR_CONFIRMATION"
    assert selection["screen_improvement_gate_passed"] is True
    assert selection["selected_candidate"] == {
        "variant": "new_full_g0p25_b0p10_ssim0p05",
        "lambda_g": .25, "lambda_b": .10, "late_ssim_weight": .05,
    }


def test_dev_screen_allows_noninferior_candidate_to_reach_confirmation(tmp_path):
    rows = []
    for task in ("a", "b", "c"):
        baseline = {
            "task": task, "variant": "l12_bal", "mean_binary_iou": .90,
            "recall": .98, "worst_view_iou": .80, "fpr": .02,
            "boundary_f1": .85,
        }
        candidate = {
            **baseline,
            "variant": "new_full_g0p25_b0p10_ssim0p00",
            # Non-inferior, but no preregistered improvement reaches its gate.
            "worst_view_iou": .801,
        }
        rows.extend((baseline, candidate))
    selection = write_dev_selection(rows, tmp_path / "selection.json")
    assert selection["status"] == "SELECTED_NONINFERIOR_FOR_CONFIRMATION"
    assert selection["screen_improvement_gate_passed"] is False
    assert selection["selected_candidate"]["lambda_g"] == .25
    assert selection["selected_candidate"]["lambda_b"] == .10


def test_dev_screen_rejects_when_every_candidate_fails_noninferiority(tmp_path):
    rows = []
    for task in ("a", "b", "c"):
        baseline = {
            "task": task, "variant": "l12_bal", "mean_binary_iou": .90,
            "recall": .98, "worst_view_iou": .80, "fpr": .02,
            "boundary_f1": .85,
        }
        candidate = {
            **baseline,
            "variant": "new_full_g0p25_b0p10_ssim0p00",
            "mean_binary_iou": .898, "recall": .974,
        }
        rows.extend((baseline, candidate))
    selection = write_dev_selection(rows, tmp_path / "selection.json")
    assert selection["status"] == "NO_ELIGIBLE_CANDIDATE"
    assert selection["screen_improvement_gate_passed"] is False
    assert "selected_candidate" not in selection


def test_dev_screen_score_tolerance_prefers_smaller_gate_boundary_weights(tmp_path):
    rows = []
    for task in ("a", "b", "c"):
        baseline = {
            "task": task, "variant": "l12_bal", "mean_binary_iou": .90,
            "recall": .98, "worst_view_iou": .80, "fpr": .02,
            "boundary_f1": .85,
        }
        rows.extend(
            (
                baseline,
                {
                    **baseline, "variant": "new_full_g0p10_b0p05_ssim0p00",
                    "worst_view_iou": .8010000,
                },
                {
                    **baseline, "variant": "new_full_g0p50_b0p20_ssim0p00",
                    "worst_view_iou": .8010010,
                },
            )
        )
    selection = write_dev_selection(rows, tmp_path / "selection.json")
    assert selection["selected_candidate"]["lambda_g"] == .10
    assert selection["selected_candidate"]["lambda_b"] == .05


def test_transfer_decision_evaluates_both_ssim_candidates_and_freezes_winner():
    rows = []
    for task in ("a", "b", "c"):
        common = {
            "task": task, "mean_binary_iou": .90, "recall": .98,
            "fpr": .02, "boundary_f1": .85,
        }
        rows.extend(
            [
                {**common, "variant": "l12_bal", "worst_view_iou": .80},
                {**common, "variant": "new_full", "worst_view_iou": .802},
                {**common, "variant": "new_full_ssim", "worst_view_iou": .804},
            ]
        )
    decision = choose_transfer_configuration(rows)
    assert set(decision["candidate_reports"]) == {"new_full", "new_full_ssim"}
    assert decision["selected_variant"] == "new_full_ssim"
    assert decision["late_ssim_weight"] == .05


def test_ssim_tie_or_numerical_tolerance_prefers_simpler_no_ssim():
    rows = []
    for task in ("a", "b", "c"):
        base = {
            "task": task, "mean_binary_iou": .90, "recall": .98,
            "fpr": .02, "boundary_f1": .85,
        }
        rows.extend(
            [
                {**base, "variant": "l12_bal", "worst_view_iou": .80},
                {**base, "variant": "new_full", "worst_view_iou": .8030000},
                {**base, "variant": "new_full_ssim", "worst_view_iou": .8030005},
            ]
        )
    assert choose_transfer_configuration(rows)["selected_variant"] == "new_full"


def test_confirmation_must_pass_gate_before_approved_freeze():
    rows = []
    for task in ("a", "b", "c"):
        baseline = {
            "task": task, "variant": "l12_bal", "mean_binary_iou": .90,
            "recall": .98, "worst_view_iou": .80, "fpr": .02,
            "boundary_f1": .85,
        }
        rows.extend(
            [
                baseline,
                {**baseline, "variant": "new_full_no_ssim", "worst_view_iou": .799},
                {**baseline, "variant": "new_full_ssim", "worst_view_iou": .799},
            ]
        )
    assert confirm_development_gate(rows)["passed"] is False


def test_stage_registration_requires_exact_matrix_and_registered_ssim():
    runs = []
    for task in ("a", "b", "c"):
        for variant in ("l12_bal", "new_full", "new_full_ssim"):
            runs.append({"task": task, "variant": variant, "run_dir": f"/{task}/{variant}"})
    registration = {
        "schema": "pcraft_soft_loss_v2_stage_manifest_v1",
        "stage": "transfer256", "soft_only": True, "runs": runs,
    }
    matrix = validate_stage_registration(registration)
    assert matrix["required_variants"] == ["l12_bal", "new_full", "new_full_ssim"]
    registration["runs"] = [run for run in runs if run["variant"] != "new_full_ssim"]
    with pytest.raises(ValueError):
        validate_stage_registration(registration)


def test_paired_statistics_are_deterministic_exact_and_holm_corrected():
    diffs = [1.0, 2.0, 3.0, 4.0]
    first = paired_bootstrap(diffs, draws=2000, seed=20260826)
    second = paired_bootstrap(diffs, draws=2000, seed=20260826)
    assert first == second
    assert first["ci_low"] > 0
    assert exact_sign_flip_test(diffs) == pytest.approx(0.125)
    adjusted = holm_adjust({"a": .01, "b": .03, "c": .20})
    assert adjusted == pytest.approx({"a": .03, "b": .06, "c": .20})


def test_component_effect_requires_ci_holm_and_safety():
    effective = assess_component(
        component="boundary", mean_difference=.01, ci_low=.002, ci_high=.02,
        holm_p=.04, direction="higher", safety_ok=True,
    )
    assert effective["effective"] is True
    assert not assess_component(
        component="boundary", mean_difference=.01, ci_low=-.001, ci_high=.02,
        holm_p=.04, direction="higher", safety_ok=True,
    )["effective"]
    assert not assess_component(
        component="gate", mean_difference=.01, ci_low=.002, ci_high=.02,
        holm_p=.04, direction="higher", safety_ok=False,
    )["effective"]


def test_formal_validation_is_fail_closed_and_128_cannot_be_paper_exported(tmp_path):
    rows = []
    for index in range(1, 11):
        rows.append(
            {
                "task": f"connected_{index:02d}", "variant": "new_full",
                "stage": "formal256", "resolution": 256, "export_resolution": 256,
                "control_resolution": 0, "steps": 2000, "seed": 17,
                "final_step_exported": True, "soft_only": True,
                "frozen_status": APPROVED_STATUS, "input_order": "xyz",
            }
        )
    validate_formal_rows(rows, required_variants=("new_full",))
    rows.append({**rows[0], "variant": "unregistered_extra"})
    with pytest.raises(ValueError):
        validate_formal_rows(rows, required_variants=("new_full",))
    rows.pop()
    rows.pop()
    with pytest.raises(ValueError):
        validate_formal_rows(rows, required_variants=("new_full",))
    with pytest.raises(ValueError):
        export_paper_table(
            [{"stage": "dev128_screen", "paper_eligible": False, "resolution": 128}],
            tmp_path / "paper.tsv",
        )
    complete = []
    for variant in ("new_full", "new_full_no_ssim"):
        complete.extend({**row, "variant": variant} for row in rows)
    with pytest.raises(ValueError):
        export_paper_table(
            [row for row in complete if row["variant"] == "new_full"],
            tmp_path / "paper.tsv",
            required_variants=("new_full", "new_full_no_ssim"),
        )
