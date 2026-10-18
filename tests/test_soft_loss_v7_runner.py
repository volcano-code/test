from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, SimpleNamespace

import pytest

from experiments import run_soft_loss_v7_hit_barrier_256 as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def locked_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in runner.REQUIRED_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)


def _plain(value: object) -> object:
    if isinstance(value, dict) or isinstance(value, MappingProxyType):
        return {key: _plain(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(nested) for nested in value]
    return value


def _row_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _matrix(stage: str, rows: list[dict[str, object]], *, endpoint: int, auth: str | None = None):
    logical = []
    for index, row in enumerate(rows):
        logical.append({"logical_index": index, **row})
    payload = {
        "schema": "pcraft_soft_loss_v7_stage_matrix_v1",
        "stage": stage,
        "campaign_sha256": "c" * 64,
        "endpoint": endpoint,
        "authorization_path": None if auth is None else "authorizations/x.json",
        "authorization_sha256": auth,
        "physical_rows": rows,
        "logical_rows": logical,
    }
    payload["semantic_sha256"] = runner.canonical_sha256(payload)
    return SimpleNamespace(
        stage=stage,
        endpoint=endpoint,
        physical_rows=tuple(MappingProxyType(row) for row in rows),
        logical_rows=tuple(MappingProxyType(row) for row in logical),
        semantic_sha256=payload["semantic_sha256"],
        payload=MappingProxyType(payload),
    )


def _development_rows(root: Path, count: int = 10) -> list[dict[str, object]]:
    identities = [("arcane_casters", None, "base")]
    for budget in runner.BUDGET_IDS:
        for profile in runner.NONBASE_PROFILES:
            identities.append(("arcane_casters", budget, profile))
    identities = identities[:count]
    rows = []
    for index, (task, budget, profile) in enumerate(identities):
        run_path = runner._run_path(root, "development", task, budget, profile)
        rows.append(
            {
                "physical_index": index,
                "task_id": task,
                "budget_id": budget,
                "profile": profile,
                "config_sha256": hashlib.sha256(f"{task}/{budget}/{profile}".encode()).hexdigest(),
                "run_path": run_path,
            }
        )
    return rows


def _prefix(row: dict[str, object], matrix: object, cap: int) -> dict[str, object]:
    result = {
        "schema": "pcraft_soft_loss_v7_engineering_prefix_v1",
        "stage": matrix.stage,
        "physical_index": row["physical_index"],
        "task_id": row["task_id"],
        "budget_id": row["budget_id"],
        "profile": row["profile"],
        "config_sha256": row["config_sha256"],
        "run_path": row["run_path"],
        "completed_updates": cap,
        "checkpoint_path": f"{row['run_path']}/checkpoints/step_{cap:04d}.pt",
        "checkpoint_sha256": hashlib.sha256(f"{row['config_sha256']}/{cap}".encode()).hexdigest(),
        "stage_matrix_sha256": matrix.semantic_sha256,
        "authorization_sha256": matrix.payload["authorization_sha256"],
        "device": "cuda:0",
        "loss_finite": True,
        "gradient_finite": True,
        "parameters_finite": True,
        "cuda_executed": True,
        "clip_count": cap,
        "optimizer_step_count": cap,
        "slot_valid": True,
        "disabled_not_constructed": True,
        "step_time_count": cap,
        "step_times_finite": True,
        "min_step_seconds": 0.01,
        "peak_allocated_bytes": 10,
        "device_total_bytes": 100,
        "hit_eligible_slots": 0 if row["profile"] not in ("hit_only", "full") else max(0, cap - 100 + 1) // 2,
        "hit_active_slots": 0,
        "barrier_eligible_slots": 0 if row["profile"] not in ("barrier_only", "full") else max(0, cap - 100) // 2,
        "barrier_active_slots": 0,
    }
    result["semantic_sha256"] = runner.canonical_sha256(result)
    return result


def test_parser_exact_surface_and_usage_errors(tmp_path: Path) -> None:
    parser = runner._build_parser()
    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    assert set(actions) == {"output_root", "device", "dry_run", "resume"}
    assert actions["device"].default == "cuda:0"
    assert parser.parse_args(["--output-root", str(tmp_path / "out")]).resume is False
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", str(tmp_path / "out")])
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-root", str(tmp_path / "out"), "--device", "cpu"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--output-root", str(tmp_path / "out"), "--dry-run", "--resume"])


def test_environment_interpreter_and_executable_bytes_are_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = SimpleNamespace(
        path=str(runner.LOCKED_INTERPRETER), sha256=runner.LOCKED_INTERPRETER_SHA256
    )
    monkeypatch.setattr(runner, "snapshot_regular_file", lambda *a, **k: snapshot)
    monkeypatch.setattr(runner, "revalidate_snapshot", lambda value, *, root=None: None)
    monkeypatch.setattr(runner.sys, "executable", str(runner.LOCKED_INTERPRETER))
    authority = runner._validate_runtime_authority()
    runner._revalidate_runtime_authority(authority)
    monkeypatch.setenv("PYTHONHASHSEED", "18")
    with pytest.raises(RuntimeError, match="environment"):
        runner._validate_runtime_authority()
    monkeypatch.setenv("PYTHONHASHSEED", "17")
    monkeypatch.setattr(runner.sys, "executable", "/usr/bin/python3")
    with pytest.raises(RuntimeError, match="interpreter"):
        runner._validate_runtime_authority()


def test_dry_run_exact_75_lines_70_unique_cells_and_zero_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "absent"
    calls = {"suite": 0, "trainer": 0, "subprocess": 0, "cuda": 0}
    monkeypatch.setattr(runner, "_validate_runtime_authority", lambda: SimpleNamespace())
    monkeypatch.setattr(runner, "_revalidate_runtime_authority", lambda value: None)
    monkeypatch.setattr(runner, "_validate_development_source", lambda: None)
    monkeypatch.setattr(
        runner,
        "select_v7_formal_tasks",
        lambda project: tuple(SimpleNamespace(case_id=f"v7_formal_{i:02d}") for i in range(1, 11)),
    )
    monkeypatch.setattr(runner, "write_v7_suite_bundle", lambda *a, **k: calls.__setitem__("suite", 1))
    monkeypatch.setattr(runner, "train_cell", lambda *a, **k: calls.__setitem__("trainer", 1))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.__setitem__("subprocess", 1))
    assert runner.main(["--output-root", str(root), "--dry-run"]) == 0
    assert not root.exists()
    assert calls == {"suite": 0, "trainer": 0, "subprocess": 0, "cuda": 0}
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 75
    assert rows[0] == {
        "schema": "pcraft_soft_loss_v7_dry_summary_v1",
        "output_root": str(root),
        "device": "cuda:0",
        "seed": 17,
        "selector_seed": 20260829,
        "development_universe_count": 30,
        "formal_count": 40,
        "optimizer_updates": 0,
        "root_was_absent": True,
    }
    cells = rows[1:71]
    assert len({(r["cohort"], r["task_id"], r["budget_id"], r["profile"]) for r in cells}) == 70
    assert [r["schema"] for r in rows[71:]] == ["pcraft_soft_loss_v7_dry_stage_v1"] * 4
    assert [r["physical_count"] for r in rows[71:]] == [10, 21, 12, 40]
    assert [r["logical_count"] for r in rows[71:]] == [10, 24, 12, 40]
    assert all(row["budget_id"] is None for row in cells[30:])


def test_dry_rejects_existing_or_noncanonical_root_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(runner, "_validate_runtime_authority", lambda: SimpleNamespace())
    monkeypatch.setattr(runner, "_revalidate_runtime_authority", lambda value: None)
    called = 0
    def selector(project):
        nonlocal called
        called += 1
    monkeypatch.setattr(runner, "select_v7_formal_tasks", selector)
    with pytest.raises(RuntimeError, match="absent"):
        runner.main(["--output-root", str(root), "--dry-run"])
    assert called == 0
    with pytest.raises(ValueError, match="canonical"):
        runner.main(["--output-root", str(root / ".." / "other"), "--dry-run"])


def test_gate_payload_exact_schema_hashes_and_predicates(tmp_path: Path) -> None:
    matrix = _matrix("development-0300", _development_rows(tmp_path), endpoint=300)
    covered = list(matrix.physical_rows)
    prefixes = [_prefix(dict(row), matrix, 2) for row in covered]
    payload = runner._gate_payload(matrix, cap=2, covered_rows=covered, engineering_rows=prefixes)
    assert set(payload) == {
        "schema", "stage", "cap", "campaign_sha256", "stage_matrix_sha256",
        "authorization_sha256", "ordered_config_sha256", "engineering_rows", "semantic_sha256",
    }
    assert payload["schema"] == "pcraft_soft_loss_v7_runner_engineering_gate_v1"
    assert payload["ordered_config_sha256"] == [r["config_sha256"] for r in covered]
    unsigned = dict(payload); digest = unsigned.pop("semantic_sha256")
    assert digest == runner.canonical_sha256(unsigned)
    bad = copy.deepcopy(prefixes); bad[0]["cuda_executed"] = False
    with pytest.raises(RuntimeError, match="PASS"):
        runner._gate_payload(matrix, cap=2, covered_rows=covered, engineering_rows=bad)
    bad = copy.deepcopy(prefixes); bad[0]["peak_allocated_bytes"] = 100
    with pytest.raises(RuntimeError, match="PASS"):
        runner._gate_payload(matrix, cap=2, covered_rows=covered, engineering_rows=bad)


def test_stage_evidence_exact_public_schema_and_canonical_hashes(tmp_path: Path) -> None:
    matrix = _matrix("development-0300", _development_rows(tmp_path), endpoint=300)
    engineering = []
    metrics = []
    for physical in matrix.physical_rows:
        row = {"physical_index": physical["physical_index"], "status": "PASS"}
        engineering.append(row)
    for logical in matrix.logical_rows:
        metrics.append({"logical_index": logical["logical_index"], "iou": 0.9})
    payload = runner._stage_evidence(matrix, engineering, metrics)
    assert payload["physical_matrix_sha256"] == runner.canonical_sha256([dict(r) for r in matrix.physical_rows])
    assert payload["logical_matrix_sha256"] == runner.canonical_sha256([dict(r) for r in matrix.logical_rows])
    assert payload["evidence_sha256"] == runner.canonical_sha256(
        {"engineering_rows": engineering, "metric_rows": metrics}
    )
    unsigned = dict(payload); semantic = unsigned.pop("semantic_sha256")
    assert semantic == runner.canonical_sha256(unsigned)


def test_advance_absent_running_completed_failed_and_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _development_rows(tmp_path, 1)[0]
    config = {"root_path": str(tmp_path), "config_sha256": row["config_sha256"], "run_path": row["run_path"]}
    matrix = _matrix("development-0300", [row], endpoint=300)
    events: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "_event", lambda **kw: events.append(kw))
    calls = []
    monkeypatch.setattr(runner, "train_cell", lambda c, **kw: calls.append(kw) or SimpleNamespace(current_checkpoint=SimpleNamespace(completed_updates=kw["max_updates"])))
    result = runner._advance(config, matrix, row, target=2)
    assert result.kind == "running" and calls == [{"stage_matrix": matrix, "resume": False, "max_updates": 2}]
    (tmp_path / str(row["run_path"])).mkdir(parents=True)
    monkeypatch.setattr(runner, "_discover", lambda *a: runner._RunState("running", SimpleNamespace(current_checkpoint=SimpleNamespace(completed_updates=2))))
    result = runner._advance(config, matrix, row, target=200)
    assert result.kind == "running" and calls[-1]["resume"] is True
    monkeypatch.setattr(runner, "_discover", lambda *a: runner._RunState("completed", SimpleNamespace(current_checkpoint=SimpleNamespace(completed_updates=3000))))
    result = runner._advance(config, matrix, row, target=3000)
    assert result.kind == "completed" and events[-1]["status"] == "SKIPPED"
    failed = SimpleNamespace(last_checkpoint=SimpleNamespace(completed_updates=137))
    monkeypatch.setattr(runner, "_discover", lambda *a: runner._RunState("failed", failed))
    assert runner._advance(config, matrix, row, target=300).kind == "failed"
    monkeypatch.setattr(runner, "_discover", lambda *a: (_ for _ in ()).throw(RuntimeError("corrupt")))
    before = len(calls)
    with pytest.raises(RuntimeError, match="corrupt"):
        runner._advance(config, matrix, row, target=300)
    assert len(calls) == before


def test_cell_event_reports_terminal_trainer_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _development_rows(tmp_path, 1)[0]
    config = {"root_path": str(tmp_path), "config_sha256": row["config_sha256"], "run_path": row["run_path"]}
    matrix = _matrix("development-0300", [row], endpoint=300)
    events = []
    monkeypatch.setattr(runner, "_event", lambda **kw: events.append(kw))
    failed = SimpleNamespace(last_checkpoint=SimpleNamespace(completed_updates=137))
    monkeypatch.setattr(runner, "train_cell", lambda *a, **k: failed)
    state = runner._advance(config, matrix, row, target=300)
    assert state.kind == "failed"
    assert events[-1]["status"] == "FAILED"


def test_stage2_fresh_split_and_execution_order(tmp_path: Path) -> None:
    rows = []
    index = 0
    for task in runner.DEVELOPMENT_TASKS:
        rows.append({"physical_index": index, "task_id": task, "budget_id": None, "profile": "base", "config_sha256": f"{index:064x}", "run_path": runner._run_path(tmp_path, "development", task, None, "base")}); index += 1
        for budget in runner.BUDGET_IDS[:2]:
            for profile in runner.NONBASE_PROFILES:
                rows.append({"physical_index": index, "task_id": task, "budget_id": budget, "profile": profile, "config_sha256": f"{index:064x}", "run_path": runner._run_path(tmp_path, "development", task, budget, profile)}); index += 1
    matrix = _matrix("development-1000", rows, endpoint=1000, auth="a" * 64)
    fresh = runner._stage_gate_rows(matrix, cap=2)
    assert len(matrix.physical_rows) == 21
    assert len(fresh) == 14
    assert {row["task_id"] for row in fresh} == {"blade_heroes", "stage_dancers"}
    assert all(row["task_id"] != "arcane_casters" for row in fresh)


def test_gate_missing_after_later_progress_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    matrix = _matrix("development-0300", _development_rows(tmp_path), endpoint=300)
    config_by_sha = {row["config_sha256"]: {"run_path": row["run_path"], "config_sha256": row["config_sha256"]} for row in matrix.physical_rows}
    monkeypatch.setattr(runner, "_gate_exists", lambda *a: False)
    monkeypatch.setattr(runner, "_discover", lambda *a: runner._RunState("running", SimpleNamespace(current_checkpoint=SimpleNamespace(completed_updates=200))))
    calls = 0
    def project(*args, **kwargs):
        nonlocal calls; calls += 1
    monkeypatch.setattr(runner, "project_engineering_prefix", project)
    with pytest.raises(RuntimeError, match="backfill|beyond"):
        runner._ensure_gate(tmp_path, {"core": {"source_bundle": {}}}, matrix, config_by_sha, cap=2)
    assert calls == 0


@pytest.mark.parametrize("terminal_kind", ["failed", "later", "corrupt"])
def test_missing_gate_preflights_all_covered_states_before_any_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_kind: str
) -> None:
    rows = _development_rows(tmp_path, 2)
    matrix = _matrix("development-0300", rows, endpoint=300)
    configs = {
        row["config_sha256"]: {
            "root_path": str(tmp_path),
            "run_path": row["run_path"],
            "config_sha256": row["config_sha256"],
        }
        for row in rows
    }
    monkeypatch.setattr(runner, "_stage_gate_rows", lambda matrix, *, cap: tuple(rows))
    monkeypatch.setattr(runner, "_gate_exists", lambda *a: False)
    discoveries = 0
    def discover(config, observed):
        nonlocal discoveries
        discoveries += 1
        if discoveries == 1:
            return runner._RunState("absent", None)
        if terminal_kind == "failed":
            return runner._RunState(
                "failed", SimpleNamespace(last_checkpoint=SimpleNamespace(completed_updates=1))
            )
        if terminal_kind == "later":
            return runner._RunState(
                "running", SimpleNamespace(current_checkpoint=SimpleNamespace(completed_updates=200))
            )
        raise RuntimeError("corrupt existing evidence")
    monkeypatch.setattr(runner, "_discover", discover)
    advances = []
    monkeypatch.setattr(runner, "_advance", lambda *a, **k: advances.append((a, k)))
    if terminal_kind == "failed":
        assert runner._ensure_gate(tmp_path, {"core": {}}, matrix, configs, cap=2) is False
    else:
        with pytest.raises(RuntimeError, match="backfill|corrupt"):
            runner._ensure_gate(tmp_path, {"core": {}}, matrix, configs, cap=2)
    assert discoveries == 2
    assert advances == []


def test_dependency_revalidation_reconstructs_exact_stage_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _development_rows(tmp_path)
    matrix = _matrix("development-0300", rows, endpoint=300)
    campaign = {"core": {"source_bundle": {}}}
    monkeypatch.setattr(runner, "validate_campaign_manifest", lambda path: campaign)
    monkeypatch.setattr(runner, "validate_training_source_bundle", lambda *a, **k: None)
    forged = _matrix("development-0300", rows[:-1], endpoint=300)
    monkeypatch.setattr(runner, "resolve_stage_matrix", lambda *a, **k: forged)
    with pytest.raises(RuntimeError, match="matrix"):
        runner._validate_dependencies(tmp_path, campaign, matrix)


def test_existing_old_stage_gate_uses_historical_prefix_projector_even_after_later_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = _development_rows(tmp_path, 1)[0]
    matrix = _matrix("development-0300", [row], endpoint=300)
    config = {
        "root_path": str(tmp_path), "run_path": row["run_path"],
        "config_sha256": row["config_sha256"],
    }
    prefix = _prefix(row, matrix, 2)
    gate = runner._gate_payload(matrix, cap=2, covered_rows=[row], engineering_rows=[prefix])
    checkpoint_snapshot = SimpleNamespace(
        sha256=prefix["checkpoint_sha256"], payload=b"checkpoint"
    )
    gate_snapshot = SimpleNamespace(
        sha256=hashlib.sha256(runner.canonical_json_bytes(gate)).hexdigest(),
        payload=runner.canonical_json_bytes(gate),
    )
    class Transaction:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def ensure_registered_directory(self, name): assert name == "runs"
        def snapshot(self, path):
            return gate_snapshot if path == runner._GATE_PATHS[(matrix.stage, 2)] else checkpoint_snapshot
        def publish(self, *args): raise AssertionError("existing gate cannot republish")
        def fsync(self, *args): raise AssertionError("existing gate cannot fsync")
    monkeypatch.setattr(runner, "_gate_exists", lambda *a: True)
    monkeypatch.setattr(runner, "_validate_dependencies", lambda *a: None)
    monkeypatch.setattr(runner, "open_v7_root_transaction", lambda root: Transaction())
    monkeypatch.setattr(runner, "_event", lambda **kw: None)
    monkeypatch.setattr(runner, "_discover", lambda *a: (_ for _ in ()).throw(AssertionError("later FailedRun must not be interpreted as old-gate failure")))
    calls = []
    def historical(observed_config, *, stage_matrix, physical_index, completed_updates):
        calls.append((stage_matrix, physical_index, completed_updates))
        return prefix
    monkeypatch.setattr(runner, "project_engineering_prefix", historical)
    assert runner._ensure_gate(
        tmp_path, {"core": {}}, matrix,
        {row["config_sha256"]: config}, cap=2,
    ) is True
    assert calls == [(matrix, 0, 2)]


def test_lock_rejects_second_owner_and_releases_after_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / "runner.lock"
    monkeypatch.setattr(runner, "RUNNER_LOCK", lock)
    with runner._runner_lock():
        with pytest.raises(RuntimeError, match="runner"):
            with runner._runner_lock():
                pass
    with runner._runner_lock():
        pass
    with pytest.raises(RuntimeError):
        with runner._runner_lock():
            raise RuntimeError("boom")
    with runner._runner_lock():
        pass


def test_bootstrap_fresh_resume_prefixes_and_campaign_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"
    campaign = MappingProxyType({"core": MappingProxyType({
        "root_path": str(root),
        "source_bundle": MappingProxyType({"semantic_sha256": "s" * 64}),
    })})
    suites = {
        "development": SimpleNamespace(path=root / "manifests/development_suite.json"),
        "formal": SimpleNamespace(path=root / "manifests/formal_suite.json"),
    }
    calls = []
    def write(output_root, *, development_source, project_root):
        calls.append("suite")
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        (root / "V7_ROOT.json").write_text("{}")
        for name in ("development_suite.json", "formal_suite.json"):
            (root / "manifests" / name).write_text("{}")
    monkeypatch.setattr(runner, "write_v7_suite_bundle", write)
    monkeypatch.setattr(runner, "validate_v7_suite", lambda path, *, expected_kind: suites[expected_kind])
    monkeypatch.setattr(runner, "build_training_source_bundle", lambda project: MappingProxyType({"semantic_sha256": "s" * 64}))
    monkeypatch.setattr(runner, "build_campaign_manifest", lambda *a, **k: campaign)
    def publish(output_root, payload):
        calls.append("campaign")
        (root / "manifests/campaign.json").write_text("{}")
    monkeypatch.setattr(runner, "publish_campaign_manifest", publish)
    monkeypatch.setattr(runner, "validate_campaign_manifest", lambda path: campaign)
    monkeypatch.setattr(runner, "validate_training_source_bundle", lambda payload, project: payload)
    assert runner._bootstrap(root, resume=False) is campaign
    assert calls == ["suite", "campaign"]
    calls.clear()
    assert runner._bootstrap(root, resume=True) is campaign
    assert calls == []
    (root / "manifests/campaign.json").unlink()
    calls.clear()
    assert runner._bootstrap(root, resume=True) is campaign
    assert calls == ["suite", "campaign"]


def test_bootstrap_rejects_illegal_prefixes_before_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "root"; (root / "manifests").mkdir(parents=True)
    (root / "V7_ROOT.json").write_text("{}")
    (root / "manifests/formal_suite.json").write_text("{}")
    calls = 0
    def forbidden(*a, **k):
        nonlocal calls; calls += 1
    monkeypatch.setattr(runner, "write_v7_suite_bundle", forbidden)
    with pytest.raises(RuntimeError, match="formal-without-development"):
        runner._bootstrap(root, resume=True)
    assert calls == 0


def test_committed_authorizations_must_be_one_valid_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    (root / "authorizations").mkdir(parents=True)
    (root / "authorizations/development_1000.json").write_text("{}")
    calls = 0
    def validate(*args, **kwargs):
        nonlocal calls; calls += 1
    monkeypatch.setattr(runner, "validate_stage_authorization", validate)
    with pytest.raises(RuntimeError, match="prefix"):
        runner._validate_committed_prefix(root, {"core": {}})
    assert calls == 0


def test_negative_authorization_forbids_later_and_formal_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    (root / "authorizations").mkdir(parents=True)
    (root / "manifests").mkdir()
    (root / "authorizations/development_0300.json").write_text("{}")
    (root / "manifests/formal_matrix.json").write_text("{}")
    negative = SimpleNamespace(status="MECHANISM_NOT_CONFIRMED")
    monkeypatch.setattr(runner, "validate_stage_authorization", lambda *a, **k: negative)
    with pytest.raises(RuntimeError, match="negative|formal"):
        runner._validate_committed_prefix(root, {"core": {}})


def test_execute_stops_mnc_without_later_stage_or_formal_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = {"core": {}}
    monkeypatch.setattr(runner, "_bootstrap", lambda *a, **k: campaign)
    events = []
    monkeypatch.setattr(runner, "_event", lambda **kw: events.append(kw))
    calls = []
    negative = SimpleNamespace(status="MECHANISM_NOT_CONFIRMED")
    def stage(root, observed, *, stage, predecessor):
        calls.append(stage)
        return negative
    monkeypatch.setattr(runner, "_run_development_stage", stage)
    monkeypatch.setattr(runner, "_run_formal", lambda *a, **k: (_ for _ in ()).throw(AssertionError("formal")))
    assert runner._execute(tmp_path, resume=False) == 3
    assert calls == ["development-0300"]
    assert events[-1]["event"] == "stop" and events[-1]["status"] == "MECHANISM_NOT_CONFIRMED"


def test_execute_exact_three_stage_chain_then_formal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = {"core": {}}
    monkeypatch.setattr(runner, "_bootstrap", lambda *a, **k: campaign)
    monkeypatch.setattr(runner, "_event", lambda **kw: None)
    calls = []
    auths = [
        SimpleNamespace(status="ADVANCE"),
        SimpleNamespace(status="ADVANCE"),
        SimpleNamespace(status="PASS"),
    ]
    def stage(root, observed, *, stage, predecessor):
        calls.append((stage, predecessor))
        return auths[len(calls) - 1]
    monkeypatch.setattr(runner, "_run_development_stage", stage)
    monkeypatch.setattr(runner, "_run_formal", lambda root, observed, authorization: calls.append(("formal", authorization)) or 0)
    assert runner._execute(tmp_path, resume=True) == 0
    assert [row[0] for row in calls] == [
        "development-0300", "development-1000", "development-3000", "formal"
    ]
    assert calls[1][1] is auths[0] and calls[2][1] is auths[1] and calls[3][1] is auths[2]


def test_resume_existing_development_authorization_revalidates_all_stage_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = {"token": "same"}
    payload = {"token": "same"}
    raw_sha = hashlib.sha256(runner.canonical_json_bytes(payload)).hexdigest()
    auth = SimpleNamespace(
        status="ADVANCE", evidence=evidence, payload=payload, raw_sha256=raw_sha,
    )
    matrix = _matrix("development-0300", _development_rows(tmp_path), endpoint=300)
    monkeypatch.setattr(runner, "_authorization_if_present", lambda *a, **k: auth)
    monkeypatch.setattr(runner, "resolve_stage_matrix", lambda *a, **k: matrix)
    monkeypatch.setattr(runner, "_config_index", lambda *a, **k: {})
    monkeypatch.setattr(runner, "_event", lambda **kw: None)
    caps = []
    monkeypatch.setattr(runner, "_ensure_gate", lambda *a, cap, **k: caps.append(cap) or True)
    monkeypatch.setattr(runner, "_endpoint_evidence", lambda *a, **k: ([], []))
    monkeypatch.setattr(runner, "_stage_evidence", lambda *a, **k: evidence)
    monkeypatch.setattr(runner, "build_stage_authorization", lambda *a, **k: auth)
    observed = runner._run_development_stage(
        tmp_path, {"core": {}}, stage="development-0300", predecessor=None
    )
    assert observed is auth
    assert caps == [2, 200]


@pytest.mark.parametrize(
    ("stage", "status", "predecessor"),
    [
        ("development-0300", "ADVANCE", None),
        ("development-3000", "PASS", SimpleNamespace(status="ADVANCE")),
    ],
)
def test_existing_authorization_rebuilds_exact_endpoint_evidence_and_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    status: str,
    predecessor: object | None,
) -> None:
    matrix = _matrix(stage, _development_rows(tmp_path), endpoint=300 if stage.endswith("0300") else 3000)
    evidence = {"schema": "pcraft_soft_loss_v7_stage_evidence_v1", "token": stage}
    payload = {"schema": "pcraft_soft_loss_v7_stage_authorization_v1", "stage": stage, "status": status}
    raw_sha = hashlib.sha256(runner.canonical_json_bytes(payload)).hexdigest()
    existing = SimpleNamespace(
        status=status, evidence=evidence, payload=payload, raw_sha256=raw_sha,
    )
    rebuilt = SimpleNamespace(
        status=status, evidence=evidence, payload=payload, raw_sha256=raw_sha,
    )
    monkeypatch.setattr(runner, "_authorization_if_present", lambda *a, **k: existing)
    monkeypatch.setattr(runner, "resolve_stage_matrix", lambda *a, **k: matrix)
    monkeypatch.setattr(runner, "_config_index", lambda *a, **k: {})
    monkeypatch.setattr(runner, "_ensure_gate", lambda *a, **k: True)
    calls = []
    monkeypatch.setattr(runner, "_endpoint_evidence", lambda observed, configs: calls.append("endpoint") or ([{"e": 1}], [{"m": 1}]))
    monkeypatch.setattr(runner, "_stage_evidence", lambda observed, engineering, metrics: calls.append("evidence") or evidence)
    monkeypatch.setattr(runner, "build_stage_authorization", lambda campaign, *, stage, evidence: calls.append("authorization") or rebuilt)
    monkeypatch.setattr(runner, "_event", lambda **kw: None)
    observed = runner._run_development_stage(
        tmp_path, {"core": {}}, stage=stage, predecessor=predecessor
    )
    assert observed is existing
    assert calls == ["endpoint", "evidence", "authorization"]


def test_existing_authorization_rejects_endpoint_evidence_or_raw_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = "development-3000"
    matrix = _matrix(stage, _development_rows(tmp_path), endpoint=3000)
    existing = SimpleNamespace(
        status="PASS", evidence={"token": "old"}, payload={"token": "old"},
        raw_sha256="a" * 64,
    )
    monkeypatch.setattr(runner, "_authorization_if_present", lambda *a, **k: existing)
    monkeypatch.setattr(runner, "resolve_stage_matrix", lambda *a, **k: matrix)
    monkeypatch.setattr(runner, "_config_index", lambda *a, **k: {})
    monkeypatch.setattr(runner, "_endpoint_evidence", lambda *a, **k: ([], []))
    monkeypatch.setattr(runner, "_stage_evidence", lambda *a, **k: {"token": "new"})
    monkeypatch.setattr(runner, "build_stage_authorization", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must reject evidence before auth")))
    monkeypatch.setattr(runner, "_event", lambda **kw: None)
    with pytest.raises(RuntimeError, match="evidence"):
        runner._run_development_stage(
            tmp_path, {"core": {}}, stage=stage,
            predecessor=SimpleNamespace(status="ADVANCE"),
        )


def test_endpoint_evidence_eliminates_failed_budget_and_reuses_shared_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _development_rows(tmp_path)
    matrix = _matrix("development-0300", rows, endpoint=300)
    configs = {r["config_sha256"]: {"config_sha256": r["config_sha256"]} for r in rows}
    engineering = []
    for row in rows:
        engineering.append({
            **row,
            "status": "FAIL" if row["budget_id"] == "budget_010" and row["profile"] == "hit_only" else "PASS",
        })
    monkeypatch.setattr(runner, "project_engineering_row", lambda config, *, stage_matrix, physical_index: engineering[physical_index])
    metric_calls = []
    monkeypatch.setattr(runner, "project_metric_row", lambda config, *, stage_matrix, logical_index: metric_calls.append(logical_index) or {"logical_index": logical_index})
    observed_engineering, metrics = runner._endpoint_evidence(matrix, configs)
    assert len(observed_engineering) == 10
    assert len(metrics) == 7
    assert metric_calls == list(range(0, 1)) + list(range(4, 10))


def test_formal_failure_stops_before_later_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path
    auth = SimpleNamespace(status="PASS")
    campaign = {"core": {}}
    rows = []
    for index, profile in enumerate(runner.PROFILES):
        rows.append({"physical_index": index, "task_id": "v7_formal_01", "budget_id": None, "profile": profile, "config_sha256": f"{index:064x}", "run_path": f"runs/formal/v7_formal_01/{profile}/seed_17"})
    matrix = _matrix("formal-3000", rows, endpoint=3000, auth="a" * 64)
    formal = SimpleNamespace(payload={"formal_configs": [{"config_sha256": r["config_sha256"]} for r in rows]})
    monkeypatch.setattr(runner, "_exists", lambda path: True)
    monkeypatch.setattr(runner, "validate_formal_matrix_manifest", lambda *a, **k: formal)
    monkeypatch.setattr(runner, "resolve_stage_matrix", lambda *a, **k: matrix)
    monkeypatch.setattr(runner, "_ensure_gate", lambda *a, **k: True)
    monkeypatch.setattr(runner, "_event", lambda **kw: None)
    calls = []
    failed = runner._RunState("failed", SimpleNamespace(last_checkpoint=SimpleNamespace(completed_updates=201)))
    def advance(config, observed, physical, *, target):
        calls.append(physical["physical_index"])
        return failed
    monkeypatch.setattr(runner, "_advance", advance)
    assert runner._run_formal(root, campaign, auth) == 4
    assert calls == [0]


def test_source_has_no_private_registry_or_trainer_import_and_no_state_json_parsing() -> None:
    source = (PROJECT_ROOT / "experiments/run_soft_loss_v7_hit_barrier_256.py").read_text()
    assert "from shadowforge_pipeline.soft_voxel_v7.registry import _" not in source
    assert "from shadowforge_pipeline.soft_voxel_v7.trainer import _" not in source
    assert "torch.load" not in source
    assert '"STATUS.json"' not in source
    assert "subprocess" not in source
    assert "evaluate_development_decision" not in source


def test_training_source_closure_binds_exact_runner_and_rejects_rehashed_drift() -> None:
    bundle = runner.build_training_source_bundle(PROJECT_ROOT)
    relative = "experiments/run_soft_loss_v7_hit_barrier_256.py"
    assert bundle["files_sha256"][relative] == hashlib.sha256(
        (PROJECT_ROOT / relative).read_bytes()
    ).hexdigest()
    assert len(bundle["files_sha256"]) == 19
    forged = _plain(bundle)
    forged["files_sha256"][relative] = "f" * 64
    forged["semantic_sha256"] = runner.canonical_sha256({
        key: value for key, value in forged.items() if key != "semantic_sha256"
    })
    with pytest.raises(ValueError, match="reconstruction|differs"):
        runner.validate_training_source_bundle(forged, PROJECT_ROOT)


def test_real_cli_help_warning_clean() -> None:
    completed = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-m", "experiments.run_soft_loss_v7_hit_barrier_256", "--help"],
        cwd=PROJECT_ROOT,
        env={**os.environ, **runner.REQUIRED_ENVIRONMENT},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "hit" in completed.stdout.lower() and "barrier" in completed.stdout.lower()
    assert "RuntimeWarning" not in completed.stderr


def test_protected_inventory_and_approved_task_hashes_unchanged() -> None:
    inventory = PROJECT_ROOT / ".superpowers/sdd/2026-08-29-soft-loss-v7-hit-barrier/v3-v6-baseline.sha256"
    rows = [line.split(maxsplit=1) for line in inventory.read_text().splitlines() if line]
    assert len(rows) == 157
    assert all(hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() == digest for digest, relative in rows)
    expected = {
        "shadowforge_pipeline/soft_voxel_v7/__init__.py": "79c04de43816d694eec901bbc6a9b6a0dae88ca6220b46373bb5196af4dddcb3",
        "shadowforge_pipeline/soft_voxel_v7/core.py": "25bfd4844e3c0a378d1289e8eb626a333f039c839847669d3ab4e4e632214fcf",
        "shadowforge_pipeline/soft_voxel_v7/protocol.py": "8e627eb4d897893fafcc66c607b444891f2e6888696c2f410760460cf2da5ce3",
        "shadowforge_pipeline/soft_voxel_v7/provenance.py": "fcbe876a14b26fe2d8858cdf099c2a8a17834548a4365311421685871ae94bb4",
        "shadowforge_pipeline/soft_voxel_v7/suite.py": "9fe008acd94ba7c7da75293a8cb8899ac909e0bfc382ecc7b9390b2880d163e5",
        "shadowforge_pipeline/soft_voxel_v7/registry.py": "7dafd85c05df6a98b066f02bb83cbc3347b8af9b150ca92cbee894c3be420be0",
        "shadowforge_pipeline/soft_voxel_v7/trainer.py": "e0accc9735a77fc565e8a2e44aa6df3c78140bbbbf564ec34be48060e3bd8895",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() == digest
