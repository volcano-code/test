"""Fail-closed contracts for the frozen soft-loss-v7 registry."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import shadowforge_pipeline.soft_voxel_v7.registry as registry
from shadowforge_pipeline.soft_voxel_v7.provenance import canonical_json_bytes, canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIGEST = "a" * 64


def _source_project(tmp_path: Path, *, complete: bool = True) -> Path:
    root = tmp_path / "project"
    for relative in registry.expected_training_source_paths(root):
        if not complete and relative.as_posix().endswith("soft_voxel_v7/trainer.py"):
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative.as_posix()}\n", encoding="utf-8")
    return root


def _campaign(tmp_path: Path) -> dict[str, object]:
    project = _source_project(tmp_path)
    bundle = registry.build_training_source_bundle(project)
    return registry._build_campaign_manifest_for_test(
        tmp_path / "root", source_bundle=bundle, device="cuda:0"
    )


def _rehash_campaign_from_core(payload: dict[str, object]) -> None:
    core = payload["core"]
    core_sha = canonical_sha256(core)
    payload["campaign_core_sha256"] = core_sha
    tasks = {row["task_id"]: row for row in core["tasks"]}
    rebuilt = []
    for old in payload["development_configs"]:
        rebuilt.append(
            registry._config(
                core, tasks[old["task_id"]], cohort="development",
                profile=old["profile"], budget_id=old["budget_id"],
                campaign_core_sha256=core_sha,
            )
        )
    payload["development_configs"] = rebuilt
    payload["semantic_sha256"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "semantic_sha256"}
    )


def _rehash_authorization(payload: dict[str, object]) -> None:
    payload["semantic_sha256"] = canonical_sha256(
        {k: v for k, v in payload.items() if k != "semantic_sha256"}
    )


def _matrix(campaign: dict[str, object], stage: str, authorization=None):
    return registry.resolve_stage_matrix(campaign, stage=stage, authorization=authorization)


def _metric(matrix, logical_index: int, **values: float) -> dict[str, object]:
    row = dict(matrix.logical_rows[logical_index])
    endpoint = matrix.endpoint
    identity_index = row["physical_index"]
    payload = {
        "schema": "pcraft_soft_loss_v7_metric_row_v1",
        "stage": matrix.stage,
        "logical_index": logical_index,
        "task_id": row["task_id"],
        "budget_id": row["budget_id"],
        "profile": row["profile"],
        "config_sha256": row["config_sha256"],
        "run_path": row["run_path"],
        "completed_updates": endpoint,
        "checkpoint_sha256": hashlib.sha256(f"cp-{identity_index}".encode()).hexdigest(),
        "metrics_path": f"{row['run_path']}/artifacts/metrics/step_{endpoint:04d}.json",
        "metrics_sha256": hashlib.sha256(f"metrics-{identity_index}".encode()).hexdigest(),
        "iou": values.get("iou", 0.96),
        "dice": values.get("dice", 0.98),
        "recall": values.get("recall", 0.98),
        "fpr": values.get("fpr", 0.001),
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload


def _engineering(matrix, physical_index: int, *, status: str = "PASS") -> dict[str, object]:
    row = dict(matrix.physical_rows[physical_index])
    endpoint = matrix.endpoint
    enabled_hit = row["profile"] in ("hit_only", "full")
    enabled_barrier = row["profile"] in ("barrier_only", "full")
    eligible = {2: 0, 200: 50, 300: 100, 1000: 450, 3000: 1450}[endpoint]
    failure = status == "FAIL"
    payload = {
        "schema": "pcraft_soft_loss_v7_engineering_row_v1",
        "stage": matrix.stage,
        "physical_index": physical_index,
        "task_id": row["task_id"],
        "budget_id": row["budget_id"],
        "profile": row["profile"],
        "config_sha256": row["config_sha256"],
        "run_path": row["run_path"],
        "completed_updates": endpoint,
        "checkpoint_sha256": hashlib.sha256(f"cp-{physical_index}".encode()).hexdigest(),
        "status": status,
        "failure_code": "STEP_CONTRACT" if failure else "NONE",
        "device": "cuda:0",
        "loss_finite": True,
        "gradient_finite": True,
        "parameters_finite": True,
        "cuda_executed": True,
        "clip_count": endpoint if not failure else max(endpoint - 1, 0),
        "optimizer_step_count": endpoint,
        "slot_valid": not failure,
        "disabled_not_constructed": True,
        "step_time_count": endpoint,
        "step_times_finite": True,
        "min_step_seconds": 0.01,
        "peak_allocated_bytes": 1,
        "device_total_bytes": 2,
        "hit_eligible_slots": eligible if enabled_hit else 0,
        "hit_active_slots": eligible if enabled_hit else 0,
        "barrier_eligible_slots": eligible if enabled_barrier else 0,
        "barrier_active_slots": eligible if enabled_barrier else 0,
        "failure_record_path": f"{row['run_path']}/failure/failure.json" if failure else None,
        "failure_record_sha256": hashlib.sha256(b"failure").hexdigest() if failure else None,
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload


def _failure_prefix_row(
    campaign: dict[str, object], matrix, physical_index: int, completed_updates: int,
) -> dict[str, object]:
    row = _engineering(matrix, physical_index, status="FAIL")
    row["completed_updates"] = completed_updates
    row["clip_count"] = completed_updates
    row["optimizer_step_count"] = completed_updates
    row["step_time_count"] = completed_updates
    row["min_step_seconds"] = None if completed_updates == 0 else .01
    active_updates = max(0, completed_updates - 100)
    expected_hit = (active_updates + 1) // 2
    expected_barrier = active_updates // 2
    enabled_hit = row["profile"] in ("hit_only", "full")
    enabled_barrier = row["profile"] in ("barrier_only", "full")
    row["hit_eligible_slots"] = expected_hit if enabled_hit else 0
    row["hit_active_slots"] = row["hit_eligible_slots"]
    row["barrier_eligible_slots"] = expected_barrier if enabled_barrier else 0
    row["barrier_active_slots"] = row["barrier_eligible_slots"]
    if completed_updates == 0:
        row["failure_code"] = "RESOURCE_INVALID"
    else:
        row["cuda_executed"] = False
        row["failure_code"] = "CUDA_NOT_USED"
    root = Path(campaign["core"]["root_path"])
    checkpoint_relative = Path(
        f"{row['run_path']}/checkpoints/step_{completed_updates:04d}.pt"
    )
    checkpoint_path = root / checkpoint_relative
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(f"checkpoint-{completed_updates}".encode())
    row["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    failure = {
        "schema": "pcraft_soft_loss_v7_failure_record_v1",
        "config_sha256": row["config_sha256"],
        "stage_matrix_sha256": matrix.semantic_sha256,
        "authorization_sha256": matrix.payload["authorization_sha256"],
        "last_checkpoint_path": checkpoint_relative.as_posix(),
        "last_checkpoint_sha256": row["checkpoint_sha256"],
        "completed_updates": completed_updates,
        "failure_code": row["failure_code"],
        "training_source_bundle_sha256": campaign["core"]["source_bundle"]["semantic_sha256"],
    }
    failure["semantic_sha256"] = canonical_sha256(failure)
    failure_path = root / row["failure_record_path"]
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_bytes(canonical_json_bytes(failure))
    row["failure_record_sha256"] = hashlib.sha256(failure_path.read_bytes()).hexdigest()
    row["semantic_sha256"] = canonical_sha256(
        {k: v for k, v in row.items() if k != "semantic_sha256"}
    )
    return row


def _stage1_rows(matrix, *, full=0.97, hit=0.94, barrier=0.945, base=0.95):
    values = {"base": base, "hit_only": hit, "barrier_only": barrier, "full": full}
    return [_metric(matrix, i, iou=values[row["profile"]]) for i, row in enumerate(matrix.logical_rows)]


def _stage2_rows(matrix, *, good_budgets=("budget_010", "budget_015")):
    rows = []
    for i, row in enumerate(matrix.logical_rows):
        profile = row["profile"]
        good = row["budget_id"] in good_budgets
        iou = {"base": .95, "hit_only": .951, "barrier_only": .952, "full": .965 if good else .94}[profile]
        recall = .97 if profile == "barrier_only" else .985
        fpr = .003 if profile == "hit_only" else .001
        rows.append(_metric(matrix, i, iou=iou, dice=min(1.0, iou + .02), recall=recall, fpr=fpr))
    return rows


def _stage3_rows(matrix, *, hit_delta=.011, barrier_delta=.012, full=.96):
    rows = []
    for i, row in enumerate(matrix.logical_rows):
        profile = row["profile"]
        iou = {"base": .94, "hit_only": full - barrier_delta, "barrier_only": full - hit_delta, "full": full}[profile]
        dice = {"base": .97, "hit_only": .98 - .006, "barrier_only": .98 - .006, "full": .98}[profile]
        rows.append(_metric(matrix, i, iou=iou, dice=dice, recall=.98, fpr=.001))
    return rows


def test_source_closure_is_exact_and_missing_future_files_fail_without_mutation(tmp_path: Path) -> None:
    expected = (
        "shadowforge_pipeline/__init__.py",
        "shadowforge_pipeline/soft_voxel_v7/__init__.py",
        "shadowforge_pipeline/soft_voxel_v7/core.py",
        "shadowforge_pipeline/soft_voxel_v7/protocol.py",
        "shadowforge_pipeline/soft_voxel_v7/provenance.py",
        "shadowforge_pipeline/soft_voxel_v7/suite.py",
        "shadowforge_pipeline/soft_voxel_v7/registry.py",
        "shadowforge_pipeline/soft_voxel_v7/trainer.py",
        "experiments/run_soft_loss_v7_hit_barrier_256.py",
        "experiments/soft_loss_v3_registry.py",
        "shadowforge_pipeline/soft_voxel_v3/__init__.py",
        "shadowforge_pipeline/soft_voxel_v3/contours.py",
        "shadowforge_pipeline/soft_voxel_v3/data.py",
        "shadowforge_pipeline/soft_voxel_v3/diagnostics.py",
        "shadowforge_pipeline/soft_voxel_v3/losses.py",
        "shadowforge_pipeline/soft_voxel_v3/lovasz.py",
        "shadowforge_pipeline/soft_voxel_v3/protocol.py",
        "shadowforge_pipeline/soft_voxel_v3/provenance.py",
        "shadowforge_pipeline/soft_voxel_v3/thresholding.py",
    )
    assert tuple(path.as_posix() for path in registry.expected_training_source_paths(tmp_path)) == expected
    project = _source_project(tmp_path, complete=False)
    before = sorted(path.relative_to(project) for path in project.rglob("*"))
    with pytest.raises(ValueError, match="missing|trainer"):
        registry.build_training_source_bundle(project)
    assert sorted(path.relative_to(project) for path in project.rglob("*")) == before


def test_source_bundle_rejects_symlink_and_hardlink_sources(tmp_path: Path) -> None:
    project = _source_project(tmp_path / "symlink")
    target = project / registry.expected_training_source_paths(project)[0]
    original = target.with_name("original.py")
    target.rename(original)
    target.symlink_to(original.name)
    with pytest.raises(ValueError):
        registry.build_training_source_bundle(project)
    project = _source_project(tmp_path / "hardlink")
    target = project / registry.expected_training_source_paths(project)[0]
    target.with_name("alias.py").hardlink_to(target)
    with pytest.raises(ValueError):
        registry.build_training_source_bundle(project)


def test_source_bundle_exact_schema_hashes_types_and_revalidation(tmp_path: Path) -> None:
    project = _source_project(tmp_path)
    bundle = registry.build_training_source_bundle(project)
    assert set(bundle) == {"schema", "project_root", "files_sha256", "semantic_sha256"}
    assert len(bundle["files_sha256"]) == 19
    plain_bundle = registry._plain(bundle)
    assert bundle["semantic_sha256"] == canonical_sha256({k: v for k, v in plain_bundle.items() if k != "semantic_sha256"})
    registry.validate_training_source_bundle(bundle, project)
    forged = copy.deepcopy(plain_bundle)
    forged["files_sha256"][next(iter(forged["files_sha256"]))] = DIGEST
    forged["semantic_sha256"] = canonical_sha256({k: v for k, v in forged.items() if k != "semantic_sha256"})
    with pytest.raises(ValueError):
        registry.validate_training_source_bundle(forged, project)
    alias = copy.deepcopy(plain_bundle)
    alias["files_sha256"] = list(alias["files_sha256"].items())
    with pytest.raises(ValueError):
        registry.validate_training_source_bundle(alias, project)
    round_trip = json.loads(canonical_json_bytes(plain_bundle))
    registry.validate_training_source_bundle(round_trip, project)


def test_campaign_exact_30_universe_paths_hashes_and_recursive_immutability(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    assert set(campaign) == {"schema", "core", "campaign_core_sha256", "development_configs", "semantic_sha256"}
    assert campaign["schema"] == "pcraft_soft_loss_v7_campaign_v1"
    assert campaign["core"]["device"] == "cuda:0"
    assert len(campaign["development_configs"]) == 30
    configs = campaign["development_configs"]
    expected = []
    for task in registry.DEVELOPMENT_TASKS:
        expected.append((task, None, "base"))
        for budget in registry.BUDGET_IDS:
            for profile in registry.PROFILES[1:]:
                expected.append((task, budget, profile))
    assert [(r["task_id"], r["budget_id"], r["profile"]) for r in configs] == expected
    for row in configs:
        expected_path = f"runs/development/{row['task_id']}/base/seed_17" if row["profile"] == "base" else f"runs/development/{row['task_id']}/{row['budget_id']}/{row['profile']}/seed_17"
        assert row["run_path"] == expected_path
        plain_row = registry._plain(row)
        assert row["config_sha256"] == canonical_sha256({k: v for k, v in plain_row.items() if k != "config_sha256"})
    with pytest.raises(TypeError):
        campaign["core"]["device"] = "cpu"


def test_campaign_device_and_rebuild_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cuda:0"):
        registry._build_campaign_manifest_for_test(tmp_path / "x", source_bundle={}, device="cpu")
    campaign = _campaign(tmp_path)
    claimed = dict(campaign["development_configs"][0])
    rebuilt = registry.rebuild_resolved_config(claimed, campaign)
    assert canonical_json_bytes(registry._plain(rebuilt)) == canonical_json_bytes(registry._plain(claimed))
    forged = copy.deepcopy(registry._plain(claimed))
    forged["seed"] = 17.0
    with pytest.raises(ValueError):
        registry.rebuild_resolved_config(forged, campaign)


@pytest.mark.parametrize(
    "path,value",
    [
        (("core", "scientific_policy", "resolution"), 255),
        (("core", "scientific_policy", "seed"), 17.0),
        (("core", "cameras", "scale_sha256"), DIGEST),
        (("core", "decision_policy", "development_3000", "deletion_iou_min"), .009),
        (("development_configs", 0, "flags", "use_hit"), True),
    ],
)
def test_campaign_deep_mutation_rehashed_is_rejected(
    tmp_path: Path, path: tuple[object, ...], value: object
) -> None:
    forged = copy.deepcopy(registry._plain(_campaign(tmp_path)))
    cursor = forged
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    if path[0] == "core":
        forged["campaign_core_sha256"] = canonical_sha256(forged["core"])
    if path[0] == "development_configs":
        row = forged["development_configs"][path[1]]
        row["config_sha256"] = canonical_sha256({k: v for k, v in row.items() if k != "config_sha256"})
    forged["semantic_sha256"] = canonical_sha256({k: v for k, v in forged.items() if k != "semantic_sha256"})
    with pytest.raises(ValueError):
        registry._validate_campaign_payload(forged)


def test_stage_matrices_exact_shapes_orders_and_shared_base(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    stage1 = _matrix(campaign, "development-0300")
    assert (stage1.endpoint, len(stage1.physical_rows), len(stage1.logical_rows)) == (300, 10, 10)
    rows = _stage1_rows(stage1)
    eng = [_engineering(stage1, i) for i in range(10)]
    decision1 = registry.evaluate_development_decision("development-0300", rows, eng)
    auth1 = registry._build_stage_authorization_for_test(campaign, stage="development-0300", matrix=stage1, metric_rows=rows, engineering_rows=eng)
    assert decision1.status == "ADVANCE"
    stage2 = _matrix(campaign, "development-1000", auth1)
    assert (stage2.endpoint, len(stage2.physical_rows), len(stage2.logical_rows)) == (1000, 21, 24)
    assert [r["task_id"] for r in stage2.physical_rows[:7]] == [registry.DEVELOPMENT_TASKS[0]] * 7
    for task in registry.DEVELOPMENT_TASKS:
        logical_bases = [r for r in stage2.logical_rows if r["task_id"] == task and r["profile"] == "base"]
        assert len(logical_bases) == 2
        assert len({(r["physical_index"], r["config_sha256"], r["run_path"]) for r in logical_bases}) == 1
    rows2 = _stage2_rows(stage2)
    eng2 = [_engineering(stage2, i) for i in range(21)]
    auth2 = registry._build_stage_authorization_for_test(campaign, stage="development-1000", matrix=stage2, metric_rows=rows2, engineering_rows=eng2)
    stage3 = _matrix(campaign, "development-3000", auth2)
    assert (stage3.endpoint, len(stage3.physical_rows), len(stage3.logical_rows)) == (3000, 12, 12)
    reordered = _stage1_rows(stage1)
    reordered[1], reordered[2] = reordered[2], reordered[1]
    with pytest.raises(ValueError):
        registry.evaluate_development_decision("development-0300", reordered, eng)


def test_stage300_ranking_tie_and_engineering_elimination(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    matrix = _matrix(campaign, "development-0300")
    rows = _stage1_rows(matrix)
    eng = [_engineering(matrix, i) for i in range(10)]
    decision = registry.evaluate_development_decision("development-0300", rows, eng)
    assert decision.ranked_budget_ids == registry.BUDGET_IDS
    assert decision.surviving_budget_ids == registry.BUDGET_IDS
    assert decision.authorized_budget_ids == registry.BUDGET_IDS[:2]
    failed = copy.deepcopy(eng)
    failed[1] = _engineering(matrix, 1, status="FAIL")
    reduced_rows = [row for row in rows if row["budget_id"] != "budget_010"]
    decision = registry.evaluate_development_decision("development-0300", reduced_rows, failed)
    assert decision.status == "ADVANCE"
    assert "budget_010" not in decision.surviving_budget_ids
    failed[0] = _engineering(matrix, 0, status="FAIL")
    assert registry.evaluate_development_decision("development-0300", [], failed).status == "MECHANISM_NOT_CONFIRMED"


def test_stage1000_gates_strict_and_next_float(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    s1 = _matrix(campaign, "development-0300")
    a1 = registry._build_stage_authorization_for_test(campaign, stage="development-0300", matrix=s1, metric_rows=_stage1_rows(s1), engineering_rows=[_engineering(s1, i) for i in range(10)])
    s2 = _matrix(campaign, "development-1000", a1)
    rows = _stage2_rows(s2)
    eng = [_engineering(s2, i) for i in range(21)]
    assert registry.evaluate_development_decision("development-1000", rows, eng).status == "ADVANCE"
    bad = copy.deepcopy(rows)
    for row in bad:
        if row["budget_id"] == "budget_010" and row["profile"] == "full":
            row["iou"] = next(r["iou"] for r in bad if r["budget_id"] == "budget_010" and r["task_id"] == row["task_id"] and r["profile"] == "barrier_only")
            row["semantic_sha256"] = canonical_sha256({k: v for k, v in row.items() if k != "semantic_sha256"})
    decision = registry.evaluate_development_decision("development-1000", bad, eng)
    assert "budget_010" not in decision.surviving_budget_ids


def test_stage3000_all_mcid_quality_and_next_float_boundaries(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    s1 = _matrix(campaign, "development-0300")
    a1 = registry._build_stage_authorization_for_test(campaign, stage="development-0300", matrix=s1, metric_rows=_stage1_rows(s1), engineering_rows=[_engineering(s1, i) for i in range(10)])
    s2 = _matrix(campaign, "development-1000", a1)
    a2 = registry._build_stage_authorization_for_test(campaign, stage="development-1000", matrix=s2, metric_rows=_stage2_rows(s2), engineering_rows=[_engineering(s2, i) for i in range(21)])
    s3 = _matrix(campaign, "development-3000", a2)
    eng = [_engineering(s3, i) for i in range(12)]
    rows = _stage3_rows(
        s3, hit_delta=.01, barrier_delta=.01,
        full=math.nextafter(.95, math.inf),
    )
    assert registry.evaluate_development_decision("development-3000", rows, eng).status == "PASS"
    for delta in (math.nextafter(.01, -math.inf),):
        failed = _stage3_rows(s3, hit_delta=delta, barrier_delta=.01, full=.95)
        assert registry.evaluate_development_decision("development-3000", failed, eng).status == "MECHANISM_NOT_CONFIRMED"
    low_quality = _stage3_rows(s3, hit_delta=.01, barrier_delta=.01, full=math.nextafter(.94, -math.inf))
    assert registry.evaluate_development_decision("development-3000", low_quality, eng).status == "MECHANISM_NOT_CONFIRMED"


def test_evidence_rejects_nonfinite_bool_alias_wrong_metric_path_and_duplicate_base(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    s1 = _matrix(campaign, "development-0300")
    rows = _stage1_rows(s1)
    eng = [_engineering(s1, i) for i in range(10)]
    for mutation in ("nan", "bool", "path"):
        bad = copy.deepcopy(rows)
        if mutation == "nan":
            bad[0]["iou"] = float("nan")
        elif mutation == "bool":
            bad[0]["completed_updates"] = True
        else:
            bad[0]["metrics_path"] += ".wrong"
        with pytest.raises(ValueError):
            registry.evaluate_development_decision("development-0300", bad, eng)


def test_engineering_failure_code_is_derived_and_pass_contract_exact(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    matrix = _matrix(campaign, "development-0300")
    rows = _stage1_rows(matrix)
    eng = [_engineering(matrix, i) for i in range(10)]
    bad = copy.deepcopy(eng)
    bad[0]["cuda_executed"] = False
    bad[0]["failure_code"] = "NONE"
    bad[0]["status"] = "PASS"
    with pytest.raises(ValueError):
        registry.evaluate_development_decision("development-0300", rows, bad)
    alias = copy.deepcopy(eng)
    alias[0]["clip_count"] = True
    with pytest.raises(ValueError):
        registry.evaluate_development_decision("development-0300", rows, alias)
    invalid_failure = copy.deepcopy(eng)
    invalid_failure[1] = _engineering(matrix, 1, status="FAIL")
    invalid_failure[1]["step_time_count"] = 0
    invalid_failure[1]["min_step_seconds"] = .01
    invalid_failure[1]["semantic_sha256"] = canonical_sha256(
        {k: v for k, v in invalid_failure[1].items() if k != "semantic_sha256"}
    )
    with pytest.raises(ValueError):
        registry.evaluate_development_decision(
            "development-0300",
            [row for row in rows if row["budget_id"] != "budget_010"],
            invalid_failure,
        )


def test_authorization_recomputes_decision_and_formal_requires_dev_pass(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    s1 = _matrix(campaign, "development-0300")
    evidence = registry._build_stage_evidence_for_test(s1, _stage1_rows(s1), [_engineering(s1, i) for i in range(10)])
    authorization = registry.build_stage_authorization(campaign, stage="development-0300", evidence=evidence)
    assert authorization.status == "ADVANCE"
    with pytest.raises(ValueError):
        registry.build_formal_matrix_manifest(campaign, authorization)
    forged = copy.deepcopy(registry._plain(evidence))
    forged["metric_rows"][1]["iou"] = .1
    forged["metric_rows"][1]["semantic_sha256"] = canonical_sha256(
        {k: v for k, v in forged["metric_rows"][1].items() if k != "semantic_sha256"}
    )
    forged["evidence_sha256"] = canonical_sha256(
        {"engineering_rows": forged["engineering_rows"], "metric_rows": forged["metric_rows"]}
    )
    forged["semantic_sha256"] = canonical_sha256(
        {k: v for k, v in forged.items() if k != "semantic_sha256"}
    )
    changed = registry.build_stage_authorization(
        campaign, stage="development-0300", evidence=forged
    )
    assert changed.payload["evidence_sha256"] == forged["evidence_sha256"]
    illegal = copy.deepcopy(registry._plain(authorization.payload))
    illegal["status"] = "MECHANISM_NOT_CONFIRMED"
    illegal["next_stage"] = None
    illegal["semantic_sha256"] = canonical_sha256(
        {k: v for k, v in illegal.items() if k != "semantic_sha256"}
    )
    with pytest.raises(ValueError):
        registry._validate_authorization_payload(
            illegal, expected_stage="development-0300", campaign=campaign
        )


def test_formal_40_matrix_and_stage_projection_require_terminal_pass(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    s1 = _matrix(campaign, "development-0300")
    a1 = registry._build_stage_authorization_for_test(campaign, stage="development-0300", matrix=s1, metric_rows=_stage1_rows(s1), engineering_rows=[_engineering(s1, i) for i in range(10)])
    s2 = _matrix(campaign, "development-1000", a1)
    a2 = registry._build_stage_authorization_for_test(campaign, stage="development-1000", matrix=s2, metric_rows=_stage2_rows(s2), engineering_rows=[_engineering(s2, i) for i in range(21)])
    s3 = _matrix(campaign, "development-3000", a2)
    a3 = registry._build_stage_authorization_for_test(
        campaign, stage="development-3000", matrix=s3,
        metric_rows=_stage3_rows(s3, hit_delta=.011, barrier_delta=.012, full=.96),
        engineering_rows=[_engineering(s3, i) for i in range(12)],
    )
    assert a3.status == "PASS"
    formal = registry._build_formal_matrix_manifest_for_test(campaign, a3)
    assert formal.payload["schema"] == "pcraft_soft_loss_v7_formal_matrix_v1"
    assert len(formal.payload["formal_configs"]) == 40
    assert [(r["task_id"], r["profile"]) for r in formal.payload["formal_configs"]] == [
        (task, profile) for task in registry.FORMAL_TASKS for profile in registry.PROFILES
    ]
    for row in formal.payload["formal_configs"]:
        if row["profile"] == "base":
            assert row["budget_id"] is None and row["budget_ratio"] is None
        else:
            assert row["budget_id"] == a3.selected_budget_id
            assert row["budget_ratio"] == registry.BUDGET_RATIOS[a3.selected_budget_id]
    stage = registry._resolve_formal_stage_matrix_for_test(campaign, a3, formal)
    assert (stage.endpoint, len(stage.physical_rows), len(stage.logical_rows)) == (3000, 40, 40)


@pytest.mark.parametrize("foreign", ["development-0300", "development-1000", "formal-3000"])
def test_wrong_stage_authorization_rejected(tmp_path: Path, foreign: str) -> None:
    campaign = _campaign(tmp_path)
    with pytest.raises(ValueError):
        registry.resolve_stage_matrix(campaign, stage="development-1000", authorization={"stage": foreign})


def test_canonical_json_duplicate_nonfinite_and_extra_fields_fail(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    extra = copy.deepcopy(registry._plain(campaign))
    extra["extra"] = None
    with pytest.raises(ValueError):
        registry._validate_campaign_payload(extra)
    raw = b'{"a":1,"a":2}'
    with pytest.raises(ValueError):
        registry._decode_registry_json(raw, label="duplicate")
    foreign_key = registry._plain(campaign)
    foreign_key[1] = foreign_key.pop("schema")
    with pytest.raises((TypeError, ValueError)):
        registry._validate_campaign_payload(foreign_key)


def test_fix1_fabricated_authorizations_cannot_advance_or_unlock_formal(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    s1 = _matrix(campaign, "development-0300")
    a1 = registry._build_stage_authorization_for_test(
        campaign, stage="development-0300", matrix=s1,
        metric_rows=_stage1_rows(s1),
        engineering_rows=[_engineering(s1, i) for i in range(10)],
    )
    config_index = registry._config_index(campaign)
    expected_physical_grant = []
    for task in registry.DEVELOPMENT_TASKS:
        expected_physical_grant.append(config_index[(task, None, "base")]["config_sha256"])
        for budget in a1.ranked_budget_ids[:2]:
            for profile in registry.PROFILES[1:]:
                expected_physical_grant.append(
                    config_index[(task, budget, profile)]["config_sha256"]
                )
    assert a1.payload["authorized_physical_config_sha256"] == tuple(expected_physical_grant)
    forged_a1 = copy.deepcopy(registry._plain(a1.payload))
    forged_a1["evidence_sha256"] = "b" * 64
    forged_a1["authorized_physical_config_sha256"].reverse()
    _rehash_authorization(forged_a1)
    with pytest.raises(ValueError):
        registry._validate_authorization_payload(
            forged_a1, expected_stage="development-0300", campaign=campaign
        )
    with pytest.raises(ValueError):
        registry.resolve_stage_matrix(
            campaign, stage="development-1000", authorization=forged_a1
        )

    s2 = _matrix(campaign, "development-1000", a1)
    a2 = registry._build_stage_authorization_for_test(
        campaign, stage="development-1000", matrix=s2,
        metric_rows=_stage2_rows(s2),
        engineering_rows=[_engineering(s2, i) for i in range(21)],
    )
    s3 = _matrix(campaign, "development-3000", a2)
    a3 = registry._build_stage_authorization_for_test(
        campaign, stage="development-3000", matrix=s3,
        metric_rows=_stage3_rows(s3, hit_delta=.011, barrier_delta=.012, full=.96),
        engineering_rows=[_engineering(s3, i) for i in range(12)],
    )
    forged_a3 = copy.deepcopy(registry._plain(a3.payload))
    forged_a3["stage_matrix_sha256"] = "c" * 64
    forged_a3["evidence_sha256"] = "d" * 64
    _rehash_authorization(forged_a3)
    with pytest.raises(ValueError):
        registry._validate_authorization_payload(
            forged_a3, expected_stage="development-3000", campaign=campaign
        )
    formal = registry._build_formal_matrix_manifest_for_test(campaign, a3)
    with pytest.raises(ValueError):
        registry._resolve_formal_stage_matrix_for_test(campaign, forged_a3, formal)


def test_fix1_failure_record_and_checkpoint_cross_row_binding(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    matrix = _matrix(campaign, "development-0300")
    metrics = _stage1_rows(matrix)
    engineering = [_engineering(matrix, i) for i in range(10)]
    mismatched = copy.deepcopy(metrics)
    mismatched[2]["checkpoint_sha256"] = "e" * 64
    mismatched[2]["semantic_sha256"] = canonical_sha256(
        {k: v for k, v in mismatched[2].items() if k != "semantic_sha256"}
    )
    with pytest.raises(ValueError):
        registry._build_stage_evidence_for_test(matrix, mismatched, engineering)

    failed = copy.deepcopy(engineering)
    failed[1] = _engineering(matrix, 1, status="FAIL")
    surviving_metrics = [row for row in metrics if row["budget_id"] != "budget_010"]
    with pytest.raises(ValueError):
        registry._build_stage_evidence_for_test(matrix, surviving_metrics, failed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("specification", {"path": "/tmp/forged-spec.md", "sha256": "a" * 64}),
        ("protected_inventory", {"path": "/tmp/forged.sha256", "sha256": "b" * 64, "entry_count": 157}),
    ],
)
def test_fix1_campaign_rejects_self_hashed_foreign_spec_or_inventory(
    tmp_path: Path, field: str, value: object
) -> None:
    campaign = copy.deepcopy(registry._plain(_campaign(tmp_path)))
    campaign["core"][field] = value
    _rehash_campaign_from_core(campaign)
    with pytest.raises(ValueError):
        registry._validate_campaign_payload(campaign)


def test_fix1_campaign_reconstructs_task_records_from_bound_suite(tmp_path: Path) -> None:
    campaign = copy.deepcopy(registry._plain(_campaign(tmp_path)))
    campaign["core"]["tasks"][0]["inputs"][0]["raw_sha256"] = "f" * 64
    _rehash_campaign_from_core(campaign)
    with pytest.raises(ValueError):
        registry._validate_campaign_payload(campaign)

    foreign_suite = copy.deepcopy(registry._plain(_campaign(tmp_path / "suite")))
    foreign_suite["core"]["development_suite"]["path"] = str(
        tmp_path / "elsewhere" / "development_suite.json"
    )
    _rehash_campaign_from_core(foreign_suite)
    with pytest.raises(ValueError):
        registry._validate_campaign_payload(foreign_suite)


def test_fix1_formal_reconstructs_tasks_and_binds_manifest_schema(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    s1 = _matrix(campaign, "development-0300")
    a1 = registry._build_stage_authorization_for_test(campaign, stage="development-0300", matrix=s1, metric_rows=_stage1_rows(s1), engineering_rows=[_engineering(s1, i) for i in range(10)])
    s2 = _matrix(campaign, "development-1000", a1)
    a2 = registry._build_stage_authorization_for_test(campaign, stage="development-1000", matrix=s2, metric_rows=_stage2_rows(s2), engineering_rows=[_engineering(s2, i) for i in range(21)])
    s3 = _matrix(campaign, "development-3000", a2)
    a3 = registry._build_stage_authorization_for_test(campaign, stage="development-3000", matrix=s3, metric_rows=_stage3_rows(s3, hit_delta=.011, barrier_delta=.012, full=.96), engineering_rows=[_engineering(s3, i) for i in range(12)])
    formal = registry._build_formal_matrix_manifest_for_test(campaign, a3)
    for config in formal.payload["formal_configs"]:
        assert config["formal_manifest_schema"] == registry.FORMAL_MATRIX_SCHEMA
    forged = copy.deepcopy(registry._plain(formal.payload))
    forged["core"]["formal_tasks"][0]["inputs"][0]["raw_sha256"] = "9" * 64
    forged["formal_core_sha256"] = canonical_sha256(forged["core"])
    tasks = {row["task_id"]: row for row in forged["core"]["formal_tasks"]}
    forged["formal_configs"] = [
        registry._config(
            forged["core"], tasks[row["task_id"]], cohort="formal",
            profile=row["profile"], budget_id=row["budget_id"],
            campaign_core_sha256=campaign["campaign_core_sha256"],
            formal_core_sha256=forged["formal_core_sha256"],
            development_authorization_sha256=forged["core"]["development_3000_authorization_sha256"],
        )
        for row in forged["formal_configs"]
    ]
    forged["semantic_sha256"] = canonical_sha256(
        {k: v for k, v in forged.items() if k != "semantic_sha256"}
    )
    with pytest.raises(ValueError):
        registry._validate_formal_payload(forged, campaign)


def test_fix1_publishers_reject_foreign_root_and_unpublished_formal_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = _campaign(tmp_path)
    suites = {
        kind: SimpleNamespace(
            kind=kind,
            path=campaign["core"][f"{kind}_suite"]["path"],
            sha256=campaign["core"][f"{kind}_suite"]["sha256"],
            semantic_sha256=campaign["core"][f"{kind}_suite"]["semantic_sha256"],
        )
        for kind in ("development", "formal")
    }
    monkeypatch.setattr(
        registry, "validate_v7_suite",
        lambda path, expected_kind: suites[expected_kind],
    )
    monkeypatch.setattr(
        registry, "validate_training_source_bundle",
        lambda payload, project_root: payload,
    )

    class FakeTransaction:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def ensure_registered_directory(self, name): pass
        def revalidate_root_marker(self): pass
        def fsync(self, path): pass
        def publish(self, path, payload): pass
        def snapshot(self, path):
            kind = "development" if "development_suite" in str(path) else "formal"
            return SimpleNamespace(sha256=suites[kind].sha256)

    monkeypatch.setattr(registry, "open_v7_root_transaction", lambda root: FakeTransaction())
    with pytest.raises(ValueError, match="campaign root argument"):
        registry.publish_campaign_manifest(tmp_path / "foreign-root", campaign)

    s1 = _matrix(campaign, "development-0300")
    a1 = registry._build_stage_authorization_for_test(campaign, stage="development-0300", matrix=s1, metric_rows=_stage1_rows(s1), engineering_rows=[_engineering(s1, i) for i in range(10)])
    s2 = _matrix(campaign, "development-1000", a1)
    a2 = registry._build_stage_authorization_for_test(campaign, stage="development-1000", matrix=s2, metric_rows=_stage2_rows(s2), engineering_rows=[_engineering(s2, i) for i in range(21)])
    s3 = _matrix(campaign, "development-3000", a2)
    a3 = registry._build_stage_authorization_for_test(campaign, stage="development-3000", matrix=s3, metric_rows=_stage3_rows(s3, hit_delta=.011, barrier_delta=.012, full=.96), engineering_rows=[_engineering(s3, i) for i in range(12)])
    formal = registry._build_formal_matrix_manifest_for_test(campaign, a3)
    monkeypatch.setattr(registry, "validate_campaign_manifest", lambda path: campaign)
    with pytest.raises(ValueError, match="published development-3000 PASS"):
        registry.publish_formal_matrix_manifest(
            Path(campaign["core"]["root_path"]), formal
        )


def test_fix1_loaded_monitored_module_without_file_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _source_project(tmp_path)
    bundle = registry.build_training_source_bundle(project)
    name = "shadowforge_pipeline.soft_voxel_v7.unexpected_runtime"
    module = ModuleType(name)
    monkeypatch.setattr(registry, "sys", SimpleNamespace(modules={name: module}))
    with pytest.raises(ValueError):
        registry._validate_loaded_training_sources(bundle, project)


def test_fix1_loaded_source_pathname_race_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = _source_project(tmp_path)
    bundle = registry.build_training_source_bundle(project)
    relative = "shadowforge_pipeline/soft_voxel_v7/trainer.py"
    path = project / relative
    original = path.read_bytes()
    path.write_bytes(b"# changed after bundle\n")
    module = ModuleType("shadowforge_pipeline.soft_voxel_v7.trainer")
    module.__file__ = str(path)
    monkeypatch.setattr(
        registry, "sys", SimpleNamespace(modules={module.__name__: module})
    )
    original_read = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes",
        lambda self: original if self == path else original_read(self),
    )
    with pytest.raises(ValueError):
        registry._validate_loaded_training_sources(bundle, project)


def test_fix1_environment_requires_exact_ambient_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError):
        registry._environment("cuda:0")
    monkeypatch.setenv("PYTHONHASHSEED", "17")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "0")
    identity = registry._environment("cuda:0")
    assert identity["environment"] == {
        "PYTHONHASHSEED": "17",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_VISIBLE_DEVICES": "0",
    }
    assert set(identity["packages"]) >= {"torch", "numpy", "opencv", "pytorch3d"}


def _terminal_authorizations(campaign: dict[str, object]):
    s1 = _matrix(campaign, "development-0300")
    a1 = registry._build_stage_authorization_for_test(
        campaign, stage="development-0300", matrix=s1,
        metric_rows=_stage1_rows(s1),
        engineering_rows=[_engineering(s1, i) for i in range(10)],
    )
    s2 = _matrix(campaign, "development-1000", a1)
    a2 = registry._build_stage_authorization_for_test(
        campaign, stage="development-1000", matrix=s2,
        metric_rows=_stage2_rows(s2),
        engineering_rows=[_engineering(s2, i) for i in range(21)],
    )
    s3 = _matrix(campaign, "development-3000", a2)
    a3 = registry._build_stage_authorization_for_test(
        campaign, stage="development-3000", matrix=s3,
        metric_rows=_stage3_rows(s3, hit_delta=.011, barrier_delta=.012, full=.96),
        engineering_rows=[_engineering(s3, i) for i in range(12)],
    )
    return s3, a3


def test_fix2_formal_resolution_binds_exact_pass_sha_budget_and_rejects_mnc_reference(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    s3, pass_authorization = _terminal_authorizations(campaign)
    formal = registry._build_formal_matrix_manifest_for_test(
        campaign, pass_authorization
    )

    other_pass = registry._build_stage_authorization_for_test(
        campaign, stage="development-3000", matrix=s3,
        metric_rows=_stage3_rows(
            s3, hit_delta=.011, barrier_delta=.012, full=.961
        ),
        engineering_rows=[_engineering(s3, i) for i in range(12)],
    )
    assert other_pass.status == "PASS"
    assert other_pass.selected_budget_id == pass_authorization.selected_budget_id
    assert other_pass.raw_sha256 != pass_authorization.raw_sha256
    with pytest.raises(ValueError):
        registry._resolve_formal_stage_matrix_for_test(
            campaign, other_pass, formal
        )

    mnc = registry._build_stage_authorization_for_test(
        campaign, stage="development-3000", matrix=s3,
        metric_rows=_stage3_rows(
            s3, hit_delta=.001, barrier_delta=.001, full=.96
        ),
        engineering_rows=[_engineering(s3, i) for i in range(12)],
    )
    assert mnc.status == "MECHANISM_NOT_CONFIRMED"
    forged = copy.deepcopy(registry._plain(formal.payload))
    forged["core"]["development_3000_authorization_sha256"] = mnc.raw_sha256
    forged["formal_core_sha256"] = canonical_sha256(forged["core"])
    tasks = {row["task_id"]: row for row in forged["core"]["formal_tasks"]}
    forged["formal_configs"] = [
        registry._config(
            forged["core"], tasks[row["task_id"]], cohort="formal",
            profile=row["profile"], budget_id=row["budget_id"],
            campaign_core_sha256=campaign["campaign_core_sha256"],
            formal_core_sha256=forged["formal_core_sha256"],
            development_authorization_sha256=mnc.raw_sha256,
        )
        for row in forged["formal_configs"]
    ]
    forged["semantic_sha256"] = canonical_sha256(
        {k: v for k, v in forged.items() if k != "semantic_sha256"}
    )
    forged_manifest = registry.V7FormalMatrixManifest(
        forged["semantic_sha256"], registry._immutable(forged)
    )
    with pytest.raises(ValueError):
        registry._resolve_formal_stage_matrix_for_test(
            campaign, pass_authorization, forged_manifest
        )


@pytest.mark.parametrize("completed_updates", [0, 137])
def test_fix2_engineering_fail_authenticates_durable_prefix_not_endpoint(
    tmp_path: Path, completed_updates: int,
) -> None:
    campaign = _campaign(tmp_path / str(completed_updates))
    matrix = _matrix(campaign, "development-0300")
    metrics = _stage1_rows(matrix)
    engineering = [_engineering(matrix, i) for i in range(10)]
    failed = _failure_prefix_row(campaign, matrix, 1, completed_updates)
    failed["slot_valid"] = True
    failed["semantic_sha256"] = canonical_sha256(
        {k: v for k, v in failed.items() if k != "semantic_sha256"}
    )
    engineering[1] = failed
    surviving_metrics = [
        row for row in metrics if row["budget_id"] != "budget_010"
    ]
    evidence = registry._build_stage_evidence_for_test(
        matrix, surviving_metrics, engineering
    )
    assert evidence["engineering_rows"][1]["completed_updates"] == completed_updates
    assert evidence["engineering_rows"][1]["failure_code"] == (
        "RESOURCE_INVALID" if completed_updates == 0 else "CUDA_NOT_USED"
    )


def test_fix2_resolved_config_and_static_identity_are_formal_cohort_aware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign(tmp_path)
    _, pass_authorization = _terminal_authorizations(campaign)
    formal = registry._build_formal_matrix_manifest_for_test(
        campaign, pass_authorization
    )
    claimed = formal.payload["formal_configs"][0]
    monkeypatch.setattr(
        registry, "validate_formal_matrix_manifest", lambda path, campaign: formal
    )
    rebuilt = registry.rebuild_resolved_config(claimed, campaign)
    assert canonical_json_bytes(registry._plain(rebuilt)) == canonical_json_bytes(
        registry._plain(claimed)
    )

    swapped_payload = copy.deepcopy(registry._plain(formal.payload))
    swapped_payload["formal_configs"] = swapped_payload["formal_configs"][1:]
    swapped = registry.V7FormalMatrixManifest(
        formal.semantic_sha256, registry._immutable(swapped_payload)
    )
    monkeypatch.setattr(
        registry, "validate_formal_matrix_manifest", lambda path, campaign: swapped
    )
    with pytest.raises(ValueError):
        registry.rebuild_resolved_config(claimed, campaign)

    monkeypatch.setattr(registry, "validate_campaign_manifest", lambda path: campaign)
    monkeypatch.setattr(
        registry, "validate_formal_matrix_manifest", lambda path, campaign: formal
    )
    monkeypatch.setattr(
        registry, "validate_training_source_bundle", lambda payload, root: payload
    )
    input_hashes = {
        item["resolved_path"]: item["raw_sha256"] for item in claimed["inputs"]
    }
    original_snapshot = registry.snapshot_regular_file
    monkeypatch.setattr(
        registry, "snapshot_regular_file",
        lambda path, **kwargs: (
            SimpleNamespace(sha256=input_hashes[str(path)])
            if str(path) in input_hashes
            else original_snapshot(path, **kwargs)
        ),
    )
    registry.revalidate_static_identities(claimed)
