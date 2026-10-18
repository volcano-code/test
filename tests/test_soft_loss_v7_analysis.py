from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "experiments/analyze_soft_loss_v7.py"
PYTHON = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python3.10")


def _load():
    spec = importlib.util.spec_from_file_location("pcraft_v7_analyzer_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_analyzer_module_is_lazy_and_cli_is_closed() -> None:
    sys.modules.pop("shadowforge_pipeline.soft_voxel_v7.statistics", None)
    analysis = _load()
    assert "shadowforge_pipeline.soft_voxel_v7.statistics" not in sys.modules
    parser = analysis.build_parser()
    assert parser.allow_abbrev is False
    parsed = parser.parse_args(["--output-root", "/tmp/example"])
    assert parsed.output_root == Path("/tmp/example") and parsed.verify is False
    for forbidden in ("--device", "--threshold", "--resume", "--overwrite"):
        with pytest.raises(SystemExit): parser.parse_args(["--output-root", "/tmp/example", forbidden, "x"])


def test_public_record_exact_fields_and_immutable_payload(tmp_path: Path) -> None:
    analysis = _load()
    record = analysis.AuthenticatedSoftReport(
        root=tmp_path, report_path=tmp_path / "r", report_sha256="a" * 64,
        csv_path=tmp_path / "c", csv_sha256="b" * 64,
        analysis_status="FORMAL_COMPLETE", soft_pass="PASS", both_key="PASS",
        claim_allowed=True, payload={"x": [1]},
    )
    assert tuple(record.__dataclass_fields__) == (
        "root","report_path","report_sha256","csv_path","csv_sha256",
        "analysis_status","soft_pass","both_key","claim_allowed","payload",
    )
    assert type(record.payload).__name__ == "mappingproxy"
    assert record.payload["x"] == (1,)


def test_second_analyze_call_fails_before_phase_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analysis = _load()
    analysis._ANALYZE_USED = True
    monkeypatch.setattr(analysis, "_phase_a", lambda *a, **k: (_ for _ in ()).throw(AssertionError("phase A called")))
    with pytest.raises(RuntimeError, match="one-shot"):
        analysis.analyze(tmp_path)


def test_fresh_process_missing_root_is_zero_write_and_statistics_remains_lazy(tmp_path: Path) -> None:
    root = tmp_path / "absent"
    code = (
        "import importlib.util,sys,pathlib;"
        f"p=pathlib.Path({str(SCRIPT)!r});s=importlib.util.spec_from_file_location('a',p);m=importlib.util.module_from_spec(s);sys.modules['a']=m;s.loader.exec_module(m);"
        "assert 'shadowforge_pipeline.soft_voxel_v7.statistics' not in sys.modules;"
        f"\ntry:m.analyze(pathlib.Path({str(root)!r}))\nexcept Exception:pass\nelse:raise AssertionError('accepted missing root');"
        "assert 'shadowforge_pipeline.soft_voxel_v7.statistics' not in sys.modules"
    )
    env = dict(os.environ, PYTHONPATH=str(PROJECT), CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="")
    result = subprocess.run([str(PYTHON), "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not root.exists()


def test_csv_projection_and_canonical_report_bytes(tmp_path: Path) -> None:
    analysis = _load()
    metric = {
        "logical_index": 0, "task_id": "v7_formal_01", "profile": "base",
        "run_path": "runs/formal/v7_formal_01/base/seed_17", "config_sha256": "a"*64,
        "checkpoint_sha256": "b"*64, "iou": .9, "dice": .91, "recall": .92, "fpr": .01,
    }
    cell = {"artifact_manifest_sha256":"c"*64,"final_density_sha256":"d"*64,
            "target_stack_sha256":"e"*64,"metric_row":metric}
    row = analysis._csv_row(cell)
    assert tuple(row) == analysis.CSV_FIELDS
    assert row["mean_iou"] == .9 and row["row_index"] == 0
    payload = {"schema":"x", "semantic_sha256":"0"*64}
    raw = analysis._sealed_json(payload)
    decoded = json.loads(raw)
    assert decoded["semantic_sha256"] != "0"*64


def test_namespace_states_and_verify_are_zero_write(tmp_path: Path) -> None:
    analysis = _load()
    root = tmp_path / "root"; root.mkdir()
    assert analysis._analysis_namespace(root) == "absent"
    (root / "analysis").mkdir()
    assert analysis._analysis_namespace(root) == "empty"
    (root / "analysis/soft_rows.csv").write_bytes(b"x")
    assert analysis._analysis_namespace(root) == "csv_only"
    (root / "analysis/soft_report.json").write_bytes(b"y")
    assert analysis._analysis_namespace(root) == "committed"
    (root / "analysis/extra").write_bytes(b"z")
    with pytest.raises(ValueError): analysis._analysis_namespace(root)


def test_synthetic_formal40_phase_b_publish_and_verify_is_zero_write(tmp_path: Path) -> None:
    import runpy
    analysis=_load()
    provenance=__import__("shadowforge_pipeline.soft_voxel_v7.provenance",fromlist=["bootstrap_v7_root"])
    root=tmp_path / "synthetic40"; provenance.bootstrap_v7_root(root)
    metric,engineering=runpy.run_path(str(PROJECT / "tests/test_soft_loss_v7_statistics.py"))["rows"]()
    cells=[]
    for m,e in zip(metric,engineering,strict=True):
        cells.append({
            "row_index":m["logical_index"],"task_id":m["task_id"],"profile":m["profile"],
            "run_path":m["run_path"],"config_sha256":m["config_sha256"],
            "checkpoint_sha256":m["checkpoint_sha256"],"artifact_manifest_sha256":"1"*64,
            "final_logits_sha256":"2"*64,"final_density_sha256":"3"*64,
            "final_hard_005_sha256":"4"*64,"target_stack_sha256":"5"*64,
            "endpoint_metrics_sha256":m["metrics_sha256"],"per_view_metrics":[],
            "engineering_row":e,"metric_row":m,
        })
    root_seals=(analysis._snapshot_compact(
        provenance,root / "V7_ROOT.json",root=root,exclusive=True
    ),)
    external=[
        analysis._snapshot_compact(provenance,SCRIPT,exclusive=True),
        analysis._snapshot_compact(
            provenance,PROJECT / "shadowforge_pipeline/soft_voxel_v7/statistics.py",exclusive=True
        ),
    ]
    external.extend(analysis._snapshot_selector_universe(provenance))
    plain={
        "campaign_record":{"path":"manifests/campaign.json","raw_sha256":"a"*64,
                           "semantic_sha256":"b"*64,
                           "training_source_bundle_semantic_sha256":"c"*64},
        "development_authorizations":[],"formal_matrix":None,"runner_gates":[],
        "terminal_failure":None,"terminal_run_states":[],"cells":cells,
    }
    phase=analysis._PhaseA(root,"FORMAL_COMPLETE",plain,root_seals,tuple(external))
    published=analysis._phase_b(phase,verify=False)
    before={p.relative_to(root).as_posix():(p.stat().st_mtime_ns,p.read_bytes())
            for p in root.rglob("*") if p.is_file()}
    verified=analysis._phase_b(phase,verify=True)
    after={p.relative_to(root).as_posix():(p.stat().st_mtime_ns,p.read_bytes())
           for p in root.rglob("*") if p.is_file()}
    assert published.report_sha256 == verified.report_sha256
    assert published.csv_sha256 == verified.csv_sha256
    assert before == after


def test_downstream_validator_never_calls_phase_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analysis = _load()
    monkeypatch.setattr(analysis, "_phase_a", lambda *a, **k: (_ for _ in ()).throw(AssertionError("training validator called")))
    with pytest.raises(ValueError):
        analysis.validate_sealed_soft_report(tmp_path / "missing.json")


def test_main_exit_and_exact_runtime_rejection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analysis = _load()
    monkeypatch.setattr(analysis, "analyze", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad")))
    assert analysis.main(["--output-root", str(tmp_path)]) == 2


def _report_shell(root: Path, *, status: str, cells: list[dict[str, object]],
                  statistics: object, labels: dict[str, object]) -> dict[str, object]:
    analysis = _load()
    return {
        "schema":"pcraft_soft_loss_v7_soft_report_v1", "analysis_status":status,
        "root_path":str(root),
        "campaign":{"path":"manifests/campaign.json","raw_sha256":"a"*64,
                    "semantic_sha256":"b"*64,
                    "training_source_bundle_semantic_sha256":"c"*64},
        "development_authorizations":[], "formal_matrix":None, "runner_gates":[],
        "terminal_failure":None, "cells":cells,
        "csv":{"path":"analysis/soft_rows.csv",
               "raw_sha256":"0"*64,"row_count":len(cells),
               "header":list(analysis.CSV_FIELDS)},
        "statistics":statistics, "labels":labels,
        "inference_scope":{"task_ids":list(analysis.TASK_IDS),"views_per_task":6,
                           "statistical_units":10,"aggregation":"task_equal",
                           "dice_role":"corroborative","iou_hypotheses":2,
                           "postprocess_included":False},
        "implementation":{"analyzer_path":str(analysis.SCRIPT if hasattr(analysis,"SCRIPT") else SCRIPT),
                          "analyzer_sha256":"d"*64,
                          "statistics_path":str(PROJECT / "shadowforge_pipeline/soft_voxel_v7/statistics.py"),
                          "statistics_sha256":"e"*64},
        "terminal_run_states":[], "root_evidence_files":[],
        "external_evidence_files":[], "semantic_sha256":"",
    }


def _publish_forged_report(root: Path, report: dict[str, object], csv_raw: bytes) -> Path:
    analysis = _load()
    directory = root / "analysis"; directory.mkdir(parents=True)
    report["csv"]["raw_sha256"] = __import__("hashlib").sha256(csv_raw).hexdigest()
    (directory / "soft_rows.csv").write_bytes(csv_raw)
    (directory / "soft_report.json").write_bytes(analysis._sealed_json(report))
    return directory / "soft_report.json"


def test_downstream_rejects_self_signed_unknown_pass_with_empty_evidence(tmp_path: Path) -> None:
    analysis = _load(); root = tmp_path / "root"; root.mkdir()
    csv_raw = analysis._csv_bytes(())
    report = _report_shell(
        root, status="ARBITRARY_PASS", cells=[], statistics=None,
        labels={"SOFT_PASS":"PASS","BOTH_KEY":"PASS","claim_allowed":True},
    )
    path = _publish_forged_report(root, report, csv_raw)
    with pytest.raises(ValueError): analysis.validate_sealed_soft_report(path)


def test_downstream_rejects_self_signed_formal40_with_empty_evidence(tmp_path: Path) -> None:
    import runpy
    analysis = _load(); root = tmp_path / "root"; root.mkdir()
    metric, engineering = runpy.run_path(str(PROJECT / "tests/test_soft_loss_v7_statistics.py"))["rows"]()
    # Normalize slot evidence to the exact registered contract expected after Fix1.
    for row in engineering:
        profile = row["profile"]
        row["hit_eligible_slots"] = 1450 if profile in ("hit_only","full") else 0
        row["hit_active_slots"] = row["hit_eligible_slots"]
        row["barrier_eligible_slots"] = 1450 if profile in ("barrier_only","full") else 0
        row["barrier_active_slots"] = row["barrier_eligible_slots"]
        unsigned=dict(row); unsigned.pop("semantic_sha256")
        row["semantic_sha256"] = __import__("shadowforge_pipeline.soft_voxel_v7.provenance",fromlist=["canonical_sha256"]).canonical_sha256(unsigned)
    cells=[]
    for m,e in zip(metric,engineering,strict=True):
        cells.append({"row_index":m["logical_index"],"task_id":m["task_id"],"profile":m["profile"],
                      "run_path":m["run_path"],"config_sha256":m["config_sha256"],
                      "checkpoint_sha256":m["checkpoint_sha256"],"artifact_manifest_sha256":"1"*64,
                      "final_logits_sha256":"2"*64,"final_density_sha256":"3"*64,
                      "final_hard_005_sha256":"4"*64,"target_stack_sha256":"5"*64,
                      "endpoint_metrics_sha256":m["metrics_sha256"],"per_view_metrics":[],
                      "engineering_row":e,"metric_row":m})
    outcome=__import__("shadowforge_pipeline.soft_voxel_v7.statistics",fromlist=["evaluate_formal_outcome"]).evaluate_formal_outcome(metric,engineering)
    report=_report_shell(root,status="FORMAL_COMPLETE",cells=cells,
                         statistics=analysis._plain(outcome),labels=analysis._plain(outcome["labels"]))
    path=_publish_forged_report(root,report,analysis._csv_bytes(cells))
    with pytest.raises(ValueError): analysis.validate_sealed_soft_report(path)


def test_phase_a_is_compact_and_never_retains_snapshot_payload(tmp_path: Path) -> None:
    analysis = _load()
    snapshot = __import__("shadowforge_pipeline.soft_voxel_v7.provenance",fromlist=["AuthenticatedFileSnapshot"]).AuthenticatedFileSnapshot(
        str(tmp_path / "x"), "a"*64, b"x" * 1024, 1, 2, 1, 1024, 3, 4, True)
    with pytest.raises(TypeError, match="compact seals"):
        analysis._PhaseA(tmp_path, "DEVELOPMENT_NOT_CONFIRMED", {}, (snapshot,), ())


@pytest.mark.parametrize("location", ["plain", "nested_seal"])
def test_phase_a_rejects_payload_bytes_recursively(
    tmp_path: Path, location: str,
) -> None:
    analysis = _load()
    compact = {
        "path":str(tmp_path / "x"), "sha256":"a"*64, "size":1,
        "device":1, "inode":2, "nlink":1, "mtime_ns":3, "ctime_ns":4,
        "exclusive":True,
    }
    plain: object = {"rows":[{"payload":b"forbidden"}]} if location == "plain" else {}
    seals: object = (compact,) if location == "plain" else ({**compact, "nested":b"forbidden"},)
    with pytest.raises((TypeError, ValueError), match="payload|compact|bytes"):
        analysis._PhaseA(tmp_path, "DEVELOPMENT_NOT_CONFIRMED", plain, seals, ())


def test_phase_a_plain_rows_are_recursively_immutable_and_40_cells_are_compact(
    tmp_path: Path,
) -> None:
    analysis = _load()
    seals=[]
    cells=[]
    for index in range(40):
        seals.append({
            "path":str(tmp_path / f"cell-{index}"), "sha256":f"{index:064x}",
            "size":1024, "device":1, "inode":index+2, "nlink":1,
            "mtime_ns":3, "ctime_ns":4, "exclusive":True,
        })
        cells.append({"row_index":index,"per_view":[{"iou":.95}]})
    phase=analysis._PhaseA(tmp_path,"FORMAL_COMPLETE",{"cells":cells},tuple(seals),())
    assert not any(isinstance(value, bytes) or hasattr(value,"payload") for value in _walk(phase))
    with pytest.raises(TypeError):
        phase.plain["cells"][0]["row_index"] = 99


def _walk(value: object):
    yield value
    if isinstance(value, dict) or type(value).__name__ == "mappingproxy":
        for nested in value.values():
            yield from _walk(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk(nested)


def test_analyzer_source_has_no_private_runner_seams_or_pathname_evidence_reads() -> None:
    source = SCRIPT.read_text("utf-8")
    for forbidden in (
        "runner._stage_gate_rows", "runner._gate_payload", "runner._stage_evidence",
        ".failure_record_path.read_text", "Path(sys.executable).read_bytes",
    ):
        assert forbidden not in source


def test_exact_terminal_vector_classifier_is_total_and_rejects_wrong_barriers() -> None:
    analysis = _load()
    assert hasattr(analysis, "_classify_terminal_vector")
    with pytest.raises(ValueError):
        analysis._classify_terminal_vector(
            stage="formal-3000", gates=(),
            states=({"state":"FAILED","completed_updates":200},),
        )


def test_evidence_closure_requires_namespace_and_selector_universe_enumeration() -> None:
    analysis = _load()
    assert hasattr(analysis, "_snapshot_selector_universe")
    assert hasattr(analysis, "_validate_exact_reached_namespace")
    assert hasattr(analysis, "_reconstruct_expected_evidence_membership")


def _make_selector_tree(root: Path) -> None:
    for relative in (
        "data/silhouettes/bench3", "data/silhouettes/complex_v2_filled",
        "data/silhouettes/original_anime_30_20260704",
        "data/silhouettes/anime_sheet_100_20260704",
    ):
        (root / relative).mkdir(parents=True)


def test_selector_closure_detects_unselected_ineligible_and_added_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis=_load(); provenance=__import__(
        "shadowforge_pipeline.soft_voxel_v7.provenance",fromlist=["snapshot_regular_file"]
    )
    _make_selector_tree(tmp_path)
    unselected=tmp_path / "data/silhouettes/bench3/unselected.png"
    ineligible=tmp_path / "data/silhouettes/complex_v2_filled/rejected.webp"
    unselected.write_bytes(b"unselected"); ineligible.write_bytes(b"rejected")
    monkeypatch.setattr(analysis,"PROJECT_ROOT",tmp_path)
    before=analysis._snapshot_selector_universe(provenance)
    unselected.write_bytes(b"mutated-unselected")
    assert analysis._plain(before) != analysis._plain(analysis._snapshot_selector_universe(provenance))
    unselected.write_bytes(b"unselected")
    before=analysis._snapshot_selector_universe(provenance)
    ineligible.write_bytes(b"mutated-rejected")
    assert analysis._plain(before) != analysis._plain(analysis._snapshot_selector_universe(provenance))
    before=analysis._snapshot_selector_universe(provenance)
    (tmp_path / "data/silhouettes/anime_sheet_100_20260704/extra.jpg").write_bytes(b"extra")
    assert analysis._plain(before) != analysis._plain(analysis._snapshot_selector_universe(provenance))


def _minimal_run_tree(root: Path, run_path: str, current: int) -> None:
    run=root / run_path
    (run / "checkpoints").mkdir(parents=True)
    (run / "artifacts/metrics").mkdir(parents=True)
    for relative in ("CONFIG.json","STATUS.json","checkpoints/LATEST.json"):
        (run / relative).write_bytes(b"x")
    for step in (0,2):
        if step <= current:
            (run / f"checkpoints/step_{step:04d}.pt").write_bytes(b"x")


def test_downstream_run_namespace_rejects_extra_generation_artifact_and_failure(
    tmp_path: Path,
) -> None:
    analysis=_load(); run_path="runs/formal/v7_formal_01/base/seed_17"
    _minimal_run_tree(tmp_path,run_path,2)
    state=_state(0,"RUNNING",2); state["run_path"]=run_path
    analysis._validate_downstream_run_namespaces(tmp_path,[state],[])
    attacks=(
        ("checkpoints/step_0001.pt",b"orphan"),
        ("artifacts/foreign.bin",b"artifact"),
        ("failure/failure.json",b"failure"),
    )
    for relative,payload in attacks:
        target=tmp_path / run_path / relative
        target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(payload)
        with pytest.raises(ValueError,match="namespace|generation|artifact|failure"):
            analysis._validate_downstream_run_namespaces(tmp_path,[state],[])
        target.unlink()


def test_evidence_membership_is_derived_from_terminal_state_not_actual_run_tree() -> None:
    analysis=_load(); state=_state(0,"RUNNING",200)
    state["run_path"]="runs/formal/v7_formal_01/base/seed_17"
    expected=analysis._expected_run_relative_files(state)
    assert expected == {
        f"{state['run_path']}/CONFIG.json", f"{state['run_path']}/STATUS.json",
        f"{state['run_path']}/checkpoints/LATEST.json",
        f"{state['run_path']}/checkpoints/step_0000.pt",
        f"{state['run_path']}/checkpoints/step_0002.pt",
        f"{state['run_path']}/checkpoints/step_0200.pt",
    }
    assert not any("foreign" in relative for relative in expected)
    closure=analysis._expected_reached_run_files([state])
    assert "runs/.pcraft-v7-trainer.lock" in closure
    assert closure == expected | {"runs/.pcraft-v7-trainer.lock"}


def test_terminal_state_paths_are_derived_from_run_and_completed_update() -> None:
    analysis=_load(); run="runs/formal/v7_formal_01/base/seed_17"
    state=_state(0,"RUNNING",200); state.update({
        "run_path":run,"status_path":f"{run}/STATUS.json","status_sha256":"a"*64,
        "checkpoint_path":f"{run}/checkpoints/step_0200.pt","checkpoint_sha256":"b"*64,
    })
    analysis._validate_terminal_state_paths(state)
    state["checkpoint_path"]=f"{run}/checkpoints/step_0002.pt"
    with pytest.raises(ValueError,match="checkpoint path"):
        analysis._validate_terminal_state_paths(state)


def test_terminal_failure_must_exactly_cross_link_unique_failed_state() -> None:
    analysis=_load(); run="runs/formal/v7_formal_01/base/seed_17"
    state=_state(0,"FAILED",200); state.update({
        "run_path":run,"status_path":f"{run}/STATUS.json","status_sha256":"a"*64,
        "checkpoint_path":f"{run}/checkpoints/step_0200.pt","checkpoint_sha256":"b"*64,
        "observation_path":f"{run}/failure/observation.json","observation_sha256":"c"*64,
        "failure_record_path":f"{run}/failure/failure.json","failure_record_sha256":"d"*64,
    })
    terminal={
        "cohort":"formal","stage":"formal-3000","phase":"GATE_STEP_0200",
        "target_updates":200,"task_id":state["task_id"],"profile":state["profile"],
        "budget_id":state["budget_id"],"run_path":run,"completed_updates":200,
        "failure_code":"CUDA_NOT_USED","status_path":state["status_path"],
        "status_sha256":state["status_sha256"],"checkpoint_path":state["checkpoint_path"],
        "checkpoint_sha256":state["checkpoint_sha256"],
        "observation_path":state["observation_path"],"observation_sha256":state["observation_sha256"],
        "failure_record_path":state["failure_record_path"],
        "failure_record_sha256":state["failure_record_sha256"],
    }
    analysis._validate_terminal_failure_crosslink(terminal,[state],target=200)
    terminal["checkpoint_sha256"]="e"*64
    with pytest.raises(ValueError,match="cross-link"):
        analysis._validate_terminal_failure_crosslink(terminal,[state],target=200)
    state.update({
        "state":"FAILED","checkpoint_path":f"{run}/checkpoints/step_0200.pt",
        "observation_path":f"{run}/failure/observation.json","observation_sha256":"c"*64,
        "failure_record_path":f"{run}/failure/failure.json","failure_record_sha256":"d"*64,
    })
    analysis._validate_terminal_state_paths(state)
    state["failure_record_path"]=f"{run}/failure/other.json"
    with pytest.raises(ValueError,match="failure path"):
        analysis._validate_terminal_state_paths(state)


def test_exact_reached_namespace_rejects_unreferenced_and_aba_progress(tmp_path: Path) -> None:
    analysis=_load(); (tmp_path / "runs").mkdir()
    expected={"V7_ROOT.json"}; (tmp_path / "V7_ROOT.json").write_bytes(b"root")
    analysis._validate_exact_reached_namespace(tmp_path,expected_root_files=expected)
    extra=tmp_path / "runs/development/foreign/base/seed_17/CONFIG.json"
    extra.parent.mkdir(parents=True); extra.write_bytes(b"x")
    with pytest.raises(ValueError,match="namespace"):
        analysis._validate_exact_reached_namespace(tmp_path,expected_root_files=expected)
    extra.unlink(); extra.parent.rmdir()
    later=tmp_path / "runs/formal/v7_formal_01/full/seed_17/STATUS.json"
    later.parent.mkdir(parents=True); later.write_bytes(b"later")
    with pytest.raises(ValueError,match="namespace"):
        analysis._validate_exact_reached_namespace(tmp_path,expected_root_files=expected)


def test_publication_contract_has_terminal_revalidation_after_report_publish() -> None:
    source = SCRIPT.read_text("utf-8")
    report_publish = source.index('transaction.publish(Path("analysis/soft_report.json")')
    terminal = source.find("_terminal_revalidate", report_publish)
    assert terminal > report_publish


def _state(index: int, state: str, updates: int | None, *, profile: str = "full") -> dict[str, object]:
    return {"order_index":index,"cohort":"formal","stage":"formal-3000",
            "task_id":f"t{index}","profile":profile,"budget_id":"budget_010",
            "run_path":f"runs/r{index}","state":state,"completed_updates":updates,
            "status_path":None,"status_sha256":None,"checkpoint_path":None,
            "checkpoint_sha256":None,"observation_path":None,"observation_sha256":None,
            "failure_record_path":None,"failure_record_sha256":None}


@pytest.mark.parametrize("failed_index", [0, 19, 39])
def test_formal_step2_failure_exact_runner_vector(failed_index: int) -> None:
    analysis=_load()
    states=[_state(i,"RUNNING",2) if i<failed_index else
            _state(i,"FAILED",2) if i==failed_index else _state(i,"ABSENT",None)
            for i in range(40)]
    assert analysis._classify_terminal_vector(stage="formal-3000",gates=(),states=states) == (
        "FORMAL_ENGINEERING_FAILED",failed_index,2)
    states[-1]=_state(39,"RUNNING",2)
    with pytest.raises(ValueError): analysis._classify_terminal_vector(stage="formal-3000",gates=(),states=states)


@pytest.mark.parametrize("failed_index", [0, 20, 39])
def test_formal_step200_failure_exact_runner_vector(failed_index: int) -> None:
    analysis=_load(); gates=({"stage":"formal-3000","cap":2},)
    states=[_state(i,"RUNNING",200) if i<failed_index else
            _state(i,"FAILED",200) if i==failed_index else _state(i,"RUNNING",2)
            for i in range(40)]
    assert analysis._classify_terminal_vector(stage="formal-3000",gates=gates,states=states)[2] == 200
    states[-1]=_state(39,"ABSENT",None)
    with pytest.raises(ValueError): analysis._classify_terminal_vector(stage="formal-3000",gates=gates,states=states)


@pytest.mark.parametrize("durable", [200, 1400, 3000])
def test_formal_endpoint_failure_exact_registered_durable_set(durable: int) -> None:
    analysis=_load(); gates=({"stage":"formal-3000","cap":2},{"stage":"formal-3000","cap":200})
    failed_index=17
    states=[_state(i,"COMPLETED",3000) if i<failed_index else
            _state(i,"FAILED",durable) if i==failed_index else _state(i,"RUNNING",200)
            for i in range(40)]
    assert analysis._classify_terminal_vector(stage="formal-3000",gates=gates,states=states)[2] == 3000
    states[failed_index]=_state(failed_index,"FAILED",1500)
    with pytest.raises(ValueError): analysis._classify_terminal_vector(stage="formal-3000",gates=gates,states=states)


def test_formal_complete_requires_both_formal_gates_and_40_completed() -> None:
    analysis=_load(); states=[_state(i,"COMPLETED",3000) for i in range(40)]
    gates=({"stage":"formal-3000","cap":2},{"stage":"formal-3000","cap":200})
    assert analysis._classify_terminal_vector(stage="formal-3000",gates=gates,states=states)[0] == "FORMAL_COMPLETE"
    with pytest.raises(ValueError): analysis._classify_terminal_vector(stage="formal-3000",gates=gates[:1],states=states)


def test_development_gate_failure_vectors_cover_stage1_both_barriers_and_stage2() -> None:
    analysis=_load()
    stage1=[_state(i,"RUNNING",2) if i<4 else _state(i,"FAILED",2) if i==4 else _state(i,"ABSENT",None)
            for i in range(10)]
    assert analysis._classify_terminal_vector(stage="development-0300",gates=(),states=stage1)[2] == 2
    profiles=("base","hit_only","barrier_only","full","hit_only","barrier_only","full","hit_only","barrier_only","full")
    stage1_200=[_state(i,"RUNNING",2,profile=profile) for i,profile in enumerate(profiles)]
    covered=[i for i,row in enumerate(stage1_200) if row["profile"] in ("base","full")]
    failed=covered[1]
    stage1_200[covered[0]]=_state(covered[0],"RUNNING",200,profile=profiles[covered[0]])
    stage1_200[failed]=_state(failed,"FAILED",200,profile=profiles[failed])
    assert analysis._classify_terminal_vector(stage="development-0300",gates=({"stage":"development-0300","cap":2},),states=stage1_200)[2] == 200
    stage2=[_state(i,"RUNNING",300) if i<7 else _state(i,"RUNNING",2) if i<12 else
            _state(i,"FAILED",2) if i==12 else _state(i,"ABSENT",None) for i in range(21)]
    assert analysis._classify_terminal_vector(stage="development-1000",gates=(),states=stage2)[2] == 2


def test_fix2_authorization_evidence_path_is_inseparable_for_all_stages() -> None:
    analysis=_load()
    expected={
        "development-0300":"authorizations/development_0300.evidence.json",
        "development-1000":"authorizations/development_1000.evidence.json",
        "development-3000":"authorizations/development_3000.evidence.json",
    }
    assert {stage:analysis._authorization_evidence_path(stage) for stage in expected} == expected
    with pytest.raises(ValueError): analysis._authorization_evidence_path("formal-3000")


def _minimal_stage_evidence(analysis, *, stage: str="development-0300",
                            campaign_sha: str="a"*64, matrix_sha: str="b"*64):
    provenance=__import__("shadowforge_pipeline.soft_voxel_v7.provenance",fromlist=["canonical_sha256"])
    payload={
        "schema":"pcraft_soft_loss_v7_stage_evidence_v1","stage":stage,
        "campaign_sha256":campaign_sha,"endpoint":300,
        "physical_matrix_sha256":"c"*64,"logical_matrix_sha256":"d"*64,
        "engineering_rows":[],"metric_rows":[],
        "evidence_sha256":provenance.canonical_sha256({"engineering_rows":[],"metric_rows":[]}),
    }
    payload["semantic_sha256"]=provenance.canonical_sha256(payload)
    return payload


def test_fix2_local_evidence_schema_semantics_and_auth_crosslink() -> None:
    analysis=_load(); evidence=_minimal_stage_evidence(analysis)
    validated=analysis._validate_local_stage_evidence(
        evidence,stage="development-0300",campaign_sha="a"*64,endpoint=300,
    )
    assert validated["evidence_sha256"] == evidence["evidence_sha256"]
    for attack in ("stage","campaign_sha256","evidence_sha256","semantic_sha256"):
        bad=copy.deepcopy(evidence)
        bad[attack]="development-1000" if attack=="stage" else "f"*64
        with pytest.raises(ValueError):
            analysis._validate_local_stage_evidence(
                bad,stage="development-0300",campaign_sha="a"*64,endpoint=300,
            )


@pytest.mark.parametrize(
    ("status","stage","target","expected"),
    [
        ("DEVELOPMENT_ENGINEERING_FAILED","development-0300",2,()),
        ("DEVELOPMENT_ENGINEERING_FAILED","development-0300",200,(("development-0300",2),)),
        ("DEVELOPMENT_ENGINEERING_FAILED","development-1000",2,(("development-0300",2),("development-0300",200))),
        ("DEVELOPMENT_NOT_CONFIRMED","development-0300",None,(("development-0300",2),("development-0300",200))),
        ("DEVELOPMENT_NOT_CONFIRMED","development-1000",None,(("development-0300",2),("development-0300",200),("development-1000",2))),
        ("DEVELOPMENT_NOT_CONFIRMED","development-3000",None,(("development-0300",2),("development-0300",200),("development-1000",2))),
        ("FORMAL_ENGINEERING_FAILED","formal-3000",2,(("development-0300",2),("development-0300",200),("development-1000",2))),
        ("FORMAL_ENGINEERING_FAILED","formal-3000",200,(("development-0300",2),("development-0300",200),("development-1000",2),("formal-3000",2))),
        ("FORMAL_ENGINEERING_FAILED","formal-3000",3000,(("development-0300",2),("development-0300",200),("development-1000",2),("formal-3000",2),("formal-3000",200))),
        ("FORMAL_COMPLETE","formal-3000",None,(("development-0300",2),("development-0300",200),("development-1000",2),("formal-3000",2),("formal-3000",200))),
    ],
)
def test_fix2_required_gate_table_is_exact(
    status: str, stage: str, target: int | None,
    expected: tuple[tuple[str,int],...],
) -> None:
    analysis=_load()
    assert analysis._required_gate_sequence(status=status,stage=stage,target=target) == expected


def _prefix_row(analysis, *, cap: int, failure: bool=False) -> dict[str,object]:
    provenance=__import__("shadowforge_pipeline.soft_voxel_v7.provenance",fromlist=["canonical_sha256"])
    row={
        "schema":"pcraft_soft_loss_v7_engineering_prefix_v1","stage":"development-0300",
        "physical_index":0,"task_id":"arcane_casters","budget_id":None,"profile":"base",
        "config_sha256":"a"*64,"run_path":"runs/development/arcane_casters/base/seed_17",
        "completed_updates":cap,"checkpoint_path":f"runs/development/arcane_casters/base/seed_17/checkpoints/step_{cap:04d}.pt",
        "checkpoint_sha256":"b"*64,"stage_matrix_sha256":"c"*64,"authorization_sha256":None,
        "device":"cuda:0","loss_finite":not failure,"gradient_finite":True,
        "parameters_finite":True,"cuda_executed":True,"clip_count":cap,
        "optimizer_step_count":cap,"slot_valid":True,"disabled_not_constructed":True,
        "step_time_count":cap,"step_times_finite":True,"min_step_seconds":.1,
        "peak_allocated_bytes":1,"device_total_bytes":2,"hit_eligible_slots":0,
        "hit_active_slots":0,"barrier_eligible_slots":0,"barrier_active_slots":0,
    }
    row["semantic_sha256"]=provenance.canonical_sha256(row)
    return row


@pytest.mark.parametrize("exception", ["training","oserror","valueerror","pass"])
def test_fix2_cap_equality_accepts_only_public_training_state_error_after_exact_fail(
    exception: str,
) -> None:
    analysis=_load()
    class PublicTrainingStateError(RuntimeError): pass
    calls=[]
    run_path="runs/development/arcane_casters/base/seed_17"
    failed=SimpleNamespace(
        last_checkpoint=SimpleNamespace(completed_updates=2,checkpoint_sha256="b"*64),
        failure_record_path=Path(f"{run_path}/failure/failure.json"),failure_record_sha256="d"*64,
    )
    fail_row={
        "status":"FAIL","failure_code":"NONFINITE_TRAINING","completed_updates":2,
        "checkpoint_sha256":"b"*64,"failure_record_path":f"{run_path}/failure/failure.json",
        "failure_record_sha256":"d"*64,"config_sha256":"a"*64,"physical_index":0,
    }
    class Public:
        TrainingStateError=PublicTrainingStateError
        def validate_failed_run(self,*args,**kwargs): calls.append("validate"); return failed
        def project_engineering_row(self,*args,**kwargs): calls.append("row"); return fail_row
        def project_engineering_prefix(self,*args,**kwargs):
            calls.append("prefix")
            if exception=="training": raise PublicTrainingStateError("opaque")
            if exception=="oserror": raise OSError("opaque")
            if exception=="valueerror": raise ValueError("opaque")
            return _prefix_row(analysis,cap=2)
    config={"config_sha256":"a"*64,"run_path":run_path}
    matrix=SimpleNamespace(stage="development-0300",semantic_sha256="c"*64,
                           physical_rows=({"config_sha256":"a"*64},))
    if exception=="training":
        assert analysis._cap_equality_gate_failure(
            Public(),config,matrix,physical_index=0,cap=2,failed=failed,
        ) is True
    elif exception=="pass":
        assert analysis._cap_equality_gate_failure(
            Public(),config,matrix,physical_index=0,cap=2,failed=failed,
        ) is False
    else:
        with pytest.raises(ValueError,match="corruption"):
            analysis._cap_equality_gate_failure(
                Public(),config,matrix,physical_index=0,cap=2,failed=failed,
            )
    assert calls == ["validate","row","prefix"]


def test_fix2_task2_closure_includes_binder_and_all_39_anchors() -> None:
    analysis=_load()
    suite=__import__("shadowforge_pipeline.soft_voxel_v7.suite",fromlist=["RAW_ANCHOR_ROWS"])
    paths=analysis._task2_dependency_paths(
        {"selector":{"anchor_paths":[row[0] for row in suite.RAW_ANCHOR_ROWS],
                     "executed_sources":{relative:"0"*64 for relative in suite.EXECUTED_SOURCE_RELATIVES}}}
    )
    assert str(PROJECT / "experiments/bind_soft_loss_v7_suites.py") in paths
    anchors={str((PROJECT / path) if not Path(path).is_absolute() else Path(path))
             for path,_ in suite.RAW_ANCHOR_ROWS}
    assert anchors.issubset(paths) and len(anchors)==39


def test_fix2_downstream_helpers_for_full_semantic_crosslinks_exist() -> None:
    analysis=_load()
    for name in (
        "_validate_local_authorization_chain","_validate_local_formal_matrix",
        "_validate_local_gate_payload","_validate_local_terminal_matrix",
        "_validate_local_failure_payloads","_validate_local_cell_payloads",
        "_validate_local_artifact_manifest",
    ):
        assert hasattr(analysis,name), name


def test_fix2_task2_fixed_dependencies_are_snapshotted_before_public_validators(
    tmp_path: Path,
) -> None:
    analysis=_load()
    provenance=__import__("shadowforge_pipeline.soft_voxel_v7.provenance",fromlist=["snapshot_regular_file"])
    binder=tmp_path / "experiments/bind_soft_loss_v7_suites.py"
    anchor=tmp_path / "anchors/old.json"
    binder.parent.mkdir(parents=True); anchor.parent.mkdir(parents=True)
    binder.write_bytes(b"binder-v1\n"); anchor.write_bytes(b"anchor-v1\n")
    rows=((str(anchor.relative_to(tmp_path)),__import__("hashlib").sha256(anchor.read_bytes()).hexdigest()),)
    before=analysis._snapshot_fixed_task2_dependencies(
        provenance,project_root=tmp_path,
        executed_relatives=(str(binder.relative_to(tmp_path)),),anchor_rows=rows,
    )
    binder.write_bytes(b"binder-v2\n")
    after=analysis._snapshot_fixed_task2_dependencies(
        provenance,project_root=tmp_path,
        executed_relatives=(str(binder.relative_to(tmp_path)),),anchor_rows=rows,
    )
    assert tuple(map(analysis._plain,before)) != tuple(map(analysis._plain,after))
    binder.write_bytes(b"binder-v1\n"); anchor.write_bytes(b"anchor-v2\n")
    with pytest.raises(ValueError,match="Task2"):
        analysis._snapshot_fixed_task2_dependencies(
            provenance,project_root=tmp_path,
            executed_relatives=(str(binder.relative_to(tmp_path)),),anchor_rows=rows,
        )
    assert len(before)==2


def test_fix2_phase_a_does_not_derive_allowed_run_files_from_live_rglob() -> None:
    source=SCRIPT.read_text(encoding="utf-8")
    phase=source[source.index("def _phase_a("):source.index("def _failure_payload(")]
    assert "for path in run.rglob" not in phase


@pytest.mark.parametrize(
    ("mutation","expected"),
    [
        ({"loss_finite":False},"NONFINITE_TRAINING"),
        ({"clip_count":1},"STEP_CONTRACT"),
        ({"cuda_executed":False},"CUDA_NOT_USED"),
        ({"min_step_seconds":0.0},"RESOURCE_INVALID"),
        ({},"NONE"),
    ],
)
def test_fix2_downstream_independently_derives_failure_code(
    mutation: dict[str, object], expected: str,
) -> None:
    analysis=_load(); completed=2
    observation={
        "loss_finite":True,"gradient_finite":True,"parameters_finite":True,
        "cuda_executed":True,"clip_count":completed,"optimizer_step_count":completed,
        "step_time_count":completed,"slot_valid":True,"disabled_not_constructed":True,
        "hit_eligible_slots":0,"hit_active_slots":0,
        "barrier_eligible_slots":0,"barrier_active_slots":0,
        "step_times_finite":True,"min_step_seconds":0.01,
        "peak_allocated_bytes":1,"device_total_bytes":2,
    }
    observation.update(mutation)
    assert analysis._derive_local_failure_code(
        observation,completed_updates=completed,profile="base",
    ) == expected


def test_fix2_downstream_rejects_self_signed_formal_config_schema(
    tmp_path: Path,
) -> None:
    import runpy
    analysis=_load()
    helpers=runpy.run_path(str(PROJECT / "tests/test_soft_loss_v7_registry.py"))
    campaign=helpers["_campaign"](tmp_path)
    _,authorization=helpers["_terminal_authorizations"](campaign)
    registry=__import__("shadowforge_pipeline.soft_voxel_v7.registry",fromlist=["_build_formal_matrix_manifest_for_test"])
    formal=copy.deepcopy(registry._plain(
        registry._build_formal_matrix_manifest_for_test(campaign,authorization).payload
    ))
    formal["formal_configs"][0]["attacker_extra"]=True
    unsigned=dict(formal["formal_configs"][0]); unsigned.pop("config_sha256")
    formal["formal_configs"][0]["config_sha256"]=__import__("hashlib").sha256(
        analysis._local_canonical_bytes(unsigned)
    ).hexdigest()
    formal["semantic_sha256"]=analysis._local_semantic_sha(formal)
    with pytest.raises(ValueError,match="formal config"):
        analysis._validate_local_formal_matrix(
            formal,root=Path(campaign["core"]["root_path"]),campaign=campaign,
            final_authorization=authorization.payload,
            final_authorization_sha=authorization.raw_sha256,
        )


def test_fix2_endpoint_per_view_schema_and_aggregate_are_independently_checked() -> None:
    analysis=_load()
    per_view=[{"view":index,"tp":9,"fp":1,"fn":1,"tn":89,
               "iou":.8181818181818182,"dice":.9,"recall":.9,"fpr":.011111111111111112}
              for index in range(6)]
    aggregate={f"mean_{key}":sum(row[key] for row in per_view)/6.0
               for key in ("iou","dice","recall","fpr")}
    analysis._validate_local_endpoint_metric_values(per_view,aggregate)
    forged=copy.deepcopy(per_view); forged[0]["attacker_extra"]=True
    with pytest.raises(ValueError,match="per-view"):
        analysis._validate_local_endpoint_metric_values(forged,aggregate)
    forged=copy.deepcopy(per_view); forged[0]["iou"]=.99
    with pytest.raises(ValueError,match="counts"):
        analysis._validate_local_endpoint_metric_values(forged,aggregate)
    with pytest.raises(ValueError,match="aggregate"):
        analysis._validate_local_endpoint_metric_values(per_view,{**aggregate,"mean_iou":.99})


def test_fix2_downstream_reads_authorization_evidence_sha_from_public_field() -> None:
    source=SCRIPT.read_text(encoding="utf-8")
    downstream=source[source.index("def validate_sealed_soft_report("):]
    assert 'decoded.get("evidence",{}).get("evidence_sha256")' not in downstream
    assert 'decoded.get("evidence_sha256") != record["evidence_sha256"]' in downstream


def test_fix2_reached_namespace_derives_prior_development_runs_from_authorized_evidence() -> None:
    analysis=_load()
    run="runs/development/arcane_casters/base/seed_17"
    evidence={}
    for stage,endpoint in (("development-0300",300),("development-1000",1000),
                           ("development-3000",3000)):
        evidence[stage]={"endpoint":endpoint,"engineering_rows":[{
            "task_id":"arcane_casters","profile":"base","budget_id":None,
            "config_sha256":"a"*64,"run_path":run,"status":"PASS",
            "completed_updates":endpoint,
        }]}
    states=analysis._development_terminal_states_from_evidence(evidence)
    assert len(states)==1 and states[0]["state"]=="COMPLETED"
    expected=analysis._expected_reached_run_files(states)
    assert f"{run}/checkpoints/step_3000.pt" in expected
    assert f"{run}/artifacts/manifest.json" in expected


def _install_fix2_full40_fixture(tmp_path: Path, analysis, monkeypatch: pytest.MonkeyPatch):
    import hashlib
    import runpy
    provenance=__import__("shadowforge_pipeline.soft_voxel_v7.provenance",fromlist=["bootstrap_v7_root"])
    registry=__import__("shadowforge_pipeline.soft_voxel_v7.registry",fromlist=["resolve_stage_matrix"])
    suite_real=__import__("shadowforge_pipeline.soft_voxel_v7.suite",fromlist=["RAW_ANCHOR_ROWS"])
    helpers=runpy.run_path(str(PROJECT / "tests/test_soft_loss_v7_registry.py"))
    root=tmp_path / "root"; provenance.bootstrap_v7_root(root)
    manifests=root / "manifests"; manifests.mkdir(exist_ok=True)
    source_suite=tmp_path / "development-source.json"; source_suite.write_bytes(b"{}")
    executed={relative:hashlib.sha256((PROJECT/relative).read_bytes()).hexdigest()
              for relative in suite_real.EXECUTED_SOURCE_RELATIVES}
    selector={"executed_sources":executed,
              "anchor_paths":[row[0] for row in suite_real.RAW_ANCHOR_ROWS],
              "anchor_sha256":[row[1] for row in suite_real.RAW_ANCHOR_ROWS]}
    suite_payloads={
        "development":{"schema":"fixture-suite","kind":"development",
                       "source_suite":{"path":str(source_suite)}},
        "formal":{"schema":"fixture-suite","kind":"formal","selector":selector},
    }
    identities={}
    for kind,payload in suite_payloads.items():
        payload["semantic_sha256"]=provenance.canonical_sha256(payload)
        path=manifests / f"{kind}_suite.json"; path.write_bytes(provenance.canonical_json_bytes(payload))
        identities[kind]={"schema":"pcraft_soft_loss_v7_suite_v1","kind":kind,
                          "path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                          "semantic_sha256":payload["semantic_sha256"]}
    source_project=helpers["_source_project"](tmp_path)
    bundle=registry.build_training_source_bundle(source_project)
    campaign=registry._build_campaign_from_records(
        root,identities["development"],identities["formal"],bundle,
        registry._private_development_task_records(),device="cuda:0",_test_mode=True,
    )
    (manifests/"campaign.json").write_bytes(provenance.canonical_json_bytes(registry._plain(campaign)))
    s1=registry.resolve_stage_matrix(campaign,stage="development-0300")
    a1=registry._build_stage_authorization_for_test(
        campaign,stage="development-0300",matrix=s1,
        metric_rows=helpers["_stage1_rows"](s1),
        engineering_rows=[helpers["_engineering"](s1,i) for i in range(10)],
    )
    s2=registry.resolve_stage_matrix(campaign,stage="development-1000",authorization=a1)
    a2=registry._build_stage_authorization_for_test(
        campaign,stage="development-1000",matrix=s2,
        metric_rows=helpers["_stage2_rows"](s2),
        engineering_rows=[helpers["_engineering"](s2,i) for i in range(21)],
    )
    s3=registry.resolve_stage_matrix(campaign,stage="development-3000",authorization=a2)
    a3=registry._build_stage_authorization_for_test(
        campaign,stage="development-3000",matrix=s3,
        metric_rows=helpers["_stage3_rows"](s3,hit_delta=.011,barrier_delta=.012,full=.96),
        engineering_rows=[helpers["_engineering"](s3,i) for i in range(12)],
    )
    auths={a.stage:a for a in (a1,a2,a3)}
    for stage,a in auths.items():
        (root/analysis._EVIDENCE[stage]).parent.mkdir(exist_ok=True)
        (root/analysis._EVIDENCE[stage]).write_bytes(provenance.canonical_json_bytes(registry._plain(a.evidence)))
        (root/analysis._AUTH[stage]).write_bytes(provenance.canonical_json_bytes(registry._plain(a.payload)))
    formal=registry._build_formal_matrix_manifest_for_test(campaign,a3)
    (manifests/"formal_matrix.json").write_bytes(provenance.canonical_json_bytes(registry._plain(formal.payload)))
    (root/"runs").mkdir(exist_ok=True); (root/"runs/.pcraft-v7-trainer.lock").write_bytes(b"lock")
    physical=[registry._physical_row(index,config) for index,config in enumerate(formal.payload["formal_configs"])]
    logical=[registry._logical_row(index,row) for index,row in enumerate(physical)]
    fm_payload={"schema":"pcraft_soft_loss_v7_stage_matrix_v1","stage":"formal-3000",
                "campaign_sha256":hashlib.sha256((manifests/"campaign.json").read_bytes()).hexdigest(),
                "endpoint":3000,"authorization_path":"authorizations/development_3000.json",
                "authorization_sha256":a3.raw_sha256,"physical_rows":physical,"logical_rows":logical}
    fm_payload["semantic_sha256"]=provenance.canonical_sha256(fm_payload)
    fm=registry.V7StageMatrix("formal-3000",3000,tuple(physical),tuple(logical),
                              fm_payload["semantic_sha256"],fm_payload,campaign,None)

    def make_run_files(state):
        run=root/state["run_path"]; run.mkdir(parents=True,exist_ok=True)
        (run/"checkpoints").mkdir(); (run/"artifacts/metrics").mkdir(parents=True)
        (run/"CONFIG.json").write_bytes(b"config")
        (run/"STATUS.json").write_bytes(provenance.canonical_json_bytes({"status":"completed" if state["state"]=="COMPLETED" else "running"}))
        for step in analysis._registered_checkpoint_steps_at(state["completed_updates"]):
            (run/f"checkpoints/step_{step:04d}.pt").write_bytes(f"cp-{step}".encode())
        (run/"checkpoints/LATEST.json").write_bytes(b"latest")
        for step in analysis._registered_metric_steps(state):
            (run/f"artifacts/metrics/step_{step:04d}.json").write_bytes(b"metric")
        if state["state"]=="COMPLETED":
            for name in ("final_logits.npy","final_density.npy","final_hard_005.npy","audit.json","resource.json","manifest.json"):
                (run/f"artifacts/{name}").write_bytes(name.encode())
    dev_evidence={stage:registry._plain(a.evidence) for stage,a in auths.items()}
    for state in analysis._development_terminal_states_from_evidence(dev_evidence): make_run_files(state)

    per_profile={
        "base":(94,2,4,900),"hit_only":(97,3,2,898),
        "barrier_only":(95,1,4,900),"full":(97,1,2,900),
    }
    formal_eng=[]; formal_metric=[]; completed={}; cells={}
    for index,row in enumerate(fm.physical_rows):
        config=formal.payload["formal_configs"][index]; run=root/row["run_path"]
        state={**dict(row),"cohort":"formal","state":"COMPLETED","completed_updates":3000}
        make_run_files(state)
        cp=run/"checkpoints/step_3000.pt"; cp.write_bytes(f"cp-{index}".encode()); cp_sha=hashlib.sha256(cp.read_bytes()).hexdigest()
        pv=[]; tp,fp,fn,tn=per_profile[row["profile"]]
        for view in range(6):
            pv.append({"view":view,"tp":tp,"fp":fp,"fn":fn,"tn":tn,
                       "iou":tp/(tp+fp+fn),"dice":2*tp/(2*tp+fp+fn),
                       "recall":tp/(tp+fn),"fpr":fp/(fp+tn)})
        agg={f"mean_{key}":sum(v[key] for v in pv)/6.0 for key in ("iou","dice","recall","fpr")}
        endpoint={"schema":"pcraft_soft_loss_v7_endpoint_metrics_v1",
                  "config_sha256":config["config_sha256"],"stage":"formal-3000",
                  "stage_matrix_sha256":fm.semantic_sha256,"authorization_sha256":fm.payload["authorization_sha256"],
                  "completed_updates":3000,"threshold":.05,"checkpoint_path":"checkpoints/step_3000.pt",
                  "checkpoint_sha256":cp_sha,"target_stack_sha256":config["target_stack_sha256"],
                  "camera_R_sha256":config["cameras"]["rotation_sha256"],
                  "camera_T_sha256":config["cameras"]["translation_sha256"],
                  "camera_scale_sha256":config["cameras"]["scale_sha256"],
                  "camera_K_sha256":config["cameras"]["projection_sha256"],
                  "per_view":pv,"aggregate":agg}
        endpoint["semantic_sha256"]=provenance.canonical_sha256(endpoint)
        metric_path=run/"artifacts/metrics/step_3000.json"; metric_path.write_bytes(provenance.canonical_json_bytes(endpoint))
        metric=helpers["_metric"](fm,index,iou=agg["mean_iou"],dice=agg["mean_dice"],recall=agg["mean_recall"],fpr=agg["mean_fpr"])
        metric["checkpoint_sha256"]=cp_sha; metric["metrics_sha256"]=hashlib.sha256(metric_path.read_bytes()).hexdigest()
        metric["semantic_sha256"]=provenance.canonical_sha256({k:v for k,v in metric.items() if k!="semantic_sha256"})
        eng=helpers["_engineering"](fm,index); eng["checkpoint_sha256"]=cp_sha
        eng["semantic_sha256"]=provenance.canonical_sha256({k:v for k,v in eng.items() if k!="semantic_sha256"})
        artifact_files={}
        for name in ("final_logits.npy","final_density.npy","final_hard_005.npy","audit.json","resource.json"):
            path=run/f"artifacts/{name}"; artifact_files[f"artifacts/{name}"]=hashlib.sha256(path.read_bytes()).hexdigest()
        artifact_files["artifacts/metrics/step_3000.json"]=hashlib.sha256(metric_path.read_bytes()).hexdigest()
        artifacts=[{"path":path,"size":(run/path).stat().st_size,"sha256":digest}
                   for path,digest in sorted(artifact_files.items())]
        manifest={"schema":"pcraft_soft_loss_v7_artifact_manifest_v1","config_sha256":config["config_sha256"],
                  "terminal_checkpoint_sha256":cp_sha,"grant_history":[],"artifacts":artifacts}
        manifest["semantic_sha256"]=provenance.canonical_sha256(manifest)
        manifest_path=run/"artifacts/manifest.json"; manifest_path.write_bytes(provenance.canonical_json_bytes(manifest))
        status=run/"STATUS.json"
        obj=SimpleNamespace(run_dir=run,status_path=status,status_sha256=hashlib.sha256(status.read_bytes()).hexdigest(),
            current_checkpoint=SimpleNamespace(completed_updates=3000,checkpoint_path=cp,checkpoint_sha256=cp_sha),
            artifact_manifest_path=manifest_path,artifact_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),artifacts=artifact_files)
        completed[config["config_sha256"]]=obj; formal_eng.append(eng); formal_metric.append(metric)
        cells[config["config_sha256"]]={"row_index":index,"task_id":row["task_id"],"profile":row["profile"],
            "run_path":row["run_path"],"config_sha256":config["config_sha256"],"checkpoint_sha256":cp_sha,
            "artifact_manifest_sha256":obj.artifact_manifest_sha256,
            "final_logits_sha256":artifact_files["artifacts/final_logits.npy"],
            "final_density_sha256":artifact_files["artifacts/final_density.npy"],
            "final_hard_005_sha256":artifact_files["artifacts/final_hard_005.npy"],
            "target_stack_sha256":config["target_stack_sha256"],"endpoint_metrics_sha256":metric["metrics_sha256"],
            "per_view_metrics":pv,"engineering_row":eng,"metric_row":metric}

    def prefix(matrix,row,cap):
        run=root/row["run_path"]; cp=run/f"checkpoints/step_{cap:04d}.pt"
        active=max(0,cap-100); hit=(active+1)//2; barrier=active//2
        value={"schema":"pcraft_soft_loss_v7_engineering_prefix_v1","stage":matrix.stage,
               **{k:row[k] for k in ("physical_index","task_id","budget_id","profile","config_sha256","run_path")},
               "completed_updates":cap,"checkpoint_path":f"{row['run_path']}/checkpoints/step_{cap:04d}.pt",
               "checkpoint_sha256":hashlib.sha256(cp.read_bytes()).hexdigest(),"stage_matrix_sha256":matrix.semantic_sha256,
               "authorization_sha256":matrix.payload["authorization_sha256"],"device":"cuda:0",
               "loss_finite":True,"gradient_finite":True,"parameters_finite":True,"cuda_executed":True,
               "clip_count":cap,"optimizer_step_count":cap,"slot_valid":True,"disabled_not_constructed":True,
               "step_time_count":cap,"step_times_finite":True,"min_step_seconds":.01,
               "peak_allocated_bytes":1,"device_total_bytes":2,
               "hit_eligible_slots":hit if row["profile"] in ("hit_only","full") else 0,
               "hit_active_slots":hit if row["profile"] in ("hit_only","full") else 0,
               "barrier_eligible_slots":barrier if row["profile"] in ("barrier_only","full") else 0,
               "barrier_active_slots":barrier if row["profile"] in ("barrier_only","full") else 0}
        value["semantic_sha256"]=provenance.canonical_sha256(value); return value
    matrices={"development-0300":s1,"development-1000":s2,"development-3000":s3,"formal-3000":fm}
    for stage,cap,path in analysis._GATES:
        matrix=matrices[stage]; covered=analysis._stage_gate_rows(matrix,cap=cap)
        payload=analysis._gate_payload(matrix,cap=cap,covered_rows=covered,
            engineering_rows=[prefix(matrix,row,cap) for row in covered],canonical_sha256=provenance.canonical_sha256)
        (root/path).parent.mkdir(exist_ok=True); (root/path).write_bytes(provenance.canonical_json_bytes(payload))

    dev_eng={stage:{row["config_sha256"]:row for row in a.evidence["engineering_rows"]} for stage,a in auths.items()}
    dev_metric={stage:list(a.evidence["metric_rows"]) for stage,a in auths.items()}
    class FakeRegistry:
        validate_campaign_manifest=lambda self,path:campaign
        validate_training_source_bundle=lambda self,payload,project:payload
        validate_stage_authorization=lambda self,path,expected_stage,campaign:auths[expected_stage]
        resolve_stage_matrix=lambda self,campaign,stage,authorization=None:(fm if stage=="formal-3000" else registry.resolve_stage_matrix(campaign,stage=stage,authorization=authorization))
        build_stage_authorization=lambda self,campaign,stage,evidence:registry.build_stage_authorization(campaign,stage=stage,evidence=evidence)
        validate_formal_matrix_manifest=lambda self,path,campaign:formal
    class FakeTrainer:
        class TrainingStateError(RuntimeError): pass
        def project_engineering_row(self,config,stage_matrix,physical_index):
            return formal_eng[physical_index] if stage_matrix.stage=="formal-3000" else dev_eng[stage_matrix.stage][config["config_sha256"]]
        def project_metric_row(self,config,stage_matrix,logical_index):
            return formal_metric[logical_index] if stage_matrix.stage=="formal-3000" else dev_metric[stage_matrix.stage][logical_index]
        def project_engineering_prefix(self,config,stage_matrix,physical_index,completed_updates):
            return prefix(stage_matrix,stage_matrix.physical_rows[physical_index],completed_updates)
        def validate_completed_run(self,config,stage_matrix): return completed[config["config_sha256"]]
        def validate_resumable_run(self,*a,**k): raise AssertionError("unexpected resumable")
        def validate_failed_run(self,*a,**k): raise AssertionError("unexpected failed")
    dev_tasks=tuple(SimpleNamespace(inputs=tuple(SimpleNamespace(resolved_path=item["resolved_path"])
                    for item in task["inputs"])) for task in campaign["core"]["tasks"])
    suite_objects={kind:SimpleNamespace(tasks=dev_tasks if kind=="development" else (),
                    payload=payload,path=manifests/f"{kind}_suite.json") for kind,payload in suite_payloads.items()}
    class FakeSuite:
        EXECUTED_SOURCE_RELATIVES=suite_real.EXECUTED_SOURCE_RELATIVES
        RAW_ANCHOR_ROWS=suite_real.RAW_ANCHOR_ROWS
        def validate_v7_suite(self,path,expected_kind): return suite_objects[expected_kind]
    protected_digests={line.split("  ",1)[1]:line.split("  ",1)[0]
                       for line in (PROJECT/".superpowers/sdd/2026-08-29-soft-loss-v7-hit-barrier/v3-v6-baseline.sha256").read_text().splitlines()
                       if line and line.split("  ",1)[1].startswith("/data/")}
    protected_digests.update({item["resolved_path"]:item["raw_sha256"]
                              for task in campaign["core"]["tasks"] for item in task["inputs"]})
    class FakeProvenance:
        def __getattr__(self,name): return getattr(provenance,name)
        def snapshot_regular_file(self,path,root=None,exclusive=True):
            path=Path(path)
            if str(path).startswith(("/data/","/test/")):
                digest=protected_digests[str(path)]
                inode=int(hashlib.sha256(str(path).encode()).hexdigest()[:12],16)
                return SimpleNamespace(path=path,payload=b"",sha256=digest,size=0,device=1,
                    inode=inode,nlink=1,mtime_ns=1,ctime_ns=1,exclusive=exclusive)
            return provenance.snapshot_regular_file(path,root=root,exclusive=exclusive)
    fake_provenance=FakeProvenance()
    original=analysis.importlib.import_module
    def importer(name):
        if name=="shadowforge_pipeline.soft_voxel_v7.provenance": return fake_provenance
        if name=="shadowforge_pipeline.soft_voxel_v7.registry": return FakeRegistry()
        if name=="shadowforge_pipeline.soft_voxel_v7.trainer": return FakeTrainer()
        if name=="shadowforge_pipeline.soft_voxel_v7.suite": return FakeSuite()
        return original(name)
    monkeypatch.setattr(analysis.importlib,"import_module",importer)
    monkeypatch.setattr(analysis,"_canonical_root",lambda value,production=True:Path(value))
    monkeypatch.setattr(analysis,"_snapshot_selector_universe",lambda provenance:())
    def recompute(root_,config,completed_,engineering,metric,provenance_):
        cell=cells[config["config_sha256"]]; run=root_/config["run_path"]
        paths=(completed_.current_checkpoint.checkpoint_path,completed_.artifact_manifest_path,
               run/"artifacts/final_logits.npy",run/"artifacts/final_density.npy",
               run/"artifacts/final_hard_005.npy",run/"artifacts/metrics/step_3000.json")
        return cell,tuple(analysis._snapshot_compact(provenance_,p,root=root_,exclusive=True) for p in paths)
    monkeypatch.setattr(analysis,"_recompute_cell",recompute)
    for key,value in analysis.REQUIRED_ENVIRONMENT.items(): monkeypatch.setenv(key,value)
    return root


def test_fix2_public_analyze_real_phase_a_phase_b_full40_publish_verify_and_downstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis=_load(); root=_install_fix2_full40_fixture(tmp_path,analysis,monkeypatch)
    sys.modules.pop("shadowforge_pipeline.soft_voxel_v7.statistics",None)
    published=analysis.analyze(root)
    assert published.analysis_status=="FORMAL_COMPLETE" and len(published.payload["cells"])==40
    analysis._ANALYZE_USED=False
    sys.modules.pop("shadowforge_pipeline.soft_voxel_v7.statistics",None)
    verified=analysis.analyze(root,verify=True)
    assert verified.report_sha256==published.report_sha256
    downstream=analysis.validate_sealed_soft_report(root/"analysis/soft_report.json")
    assert downstream.report_sha256==published.report_sha256


def test_fix3_second_statistics_guard_precedes_lazy_import() -> None:
    analysis=_load()
    phase=SimpleNamespace()
    sys.modules[analysis._STATISTICS_MODULE]=SimpleNamespace()
    with pytest.raises(RuntimeError,match="statistics"):
        analysis._phase_b(phase,verify=False)


@pytest.mark.parametrize("attack",("empty","partial","reordered"))
def test_fix3_gate_coverage_is_rebuilt_from_matrix_not_declared_rows(attack: str) -> None:
    analysis=_load()
    physical=[{"physical_index":index,"task_id":f"task{index}","budget_id":None,
               "profile":"base" if index==0 else "full","config_sha256":f"{index:064x}",
               "run_path":f"runs/r{index}"} for index in range(10)]
    covered=analysis._local_gate_covered_rows("development-0300",200,physical)
    assert [row["physical_index"] for row in covered]==[0,1,2,3]
    declared=[] if attack=="empty" else covered[:2] if attack=="partial" else list(reversed(covered))
    with pytest.raises(ValueError,match="coverage"):
        analysis._require_local_gate_coverage(declared,covered)


def test_fix3_selector_authority_is_exact_151_rows_and_frozen_digest() -> None:
    analysis=_load()
    rows=analysis._snapshot_selector_authority(
        __import__("shadowforge_pipeline.soft_voxel_v7.provenance",fromlist=["snapshot_regular_file"])
    )
    assert len(rows)==151
    assert analysis._selector_authority_sha256(rows)=="d2baa7844440b88f345e3116bdc157e28de537cac5e36781cbf748dd16c7f125"


def test_fix3_campaign_and_authorization_rebuilders_are_mandatory() -> None:
    analysis=_load()
    assert callable(analysis._rebuild_local_campaign)
    assert callable(analysis._rebuild_local_development_chain)
    assert callable(analysis._rebuild_local_formal_matrix)


def test_fix3_test_runtime_exists_and_is_default_disabled() -> None:
    analysis=_load()
    assert analysis._TEST_RECOMPUTE_RUNTIME is None
    assert tuple(analysis._AnalyzerTestRecomputeRuntime.__dataclass_fields__)==(
        "artifact_resolution","device","protocol_factory","render_thresholded",
        "synchronize","empty_cache",
    )
