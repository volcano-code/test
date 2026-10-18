"""Fail-closed contracts for immutable soft-loss-v3 suite binding."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import experiments.soft_loss_v3_registry as registry_module
from experiments.soft_loss_v3_registry import (
    TaskInputs,
    assert_suite_disjoint,
    bind_development_suite,
    bind_formal_suite,
    binding_main,
    build_binding_parser,
    load_bound_suite,
)
from shadowforge_pipeline.soft_voxel_v3.provenance import (
    canonical_hash,
    canonical_json_bytes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_IDS = ("arcane_casters", "blade_heroes", "stage_dancers")
FORMAL_IDS = tuple(f"connected_{index:02d}" for index in range(1, 11))
AXES = ("x", "y", "z")
SUITE_FIELDS = (
    "schema",
    "status",
    "role",
    "source_manifest",
    "tasks",
    "task_count",
    "input_count",
    "input_order",
    "protocol",
    "primary_variants",
    "appendix_controls",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _manifest_path(path: Path, project: Path, obsolete_prefix: bool) -> str:
    if not obsolete_prefix:
        return str(path.resolve())
    relative = path.resolve().relative_to(project.resolve()).as_posix()
    return f"/home/lin/xvxinyao/ShadowForge/{relative}"


def _make_inputs(
    tmp_path: Path,
    *,
    obsolete_prefix: bool = False,
) -> dict[str, Path]:
    project = tmp_path / "project"
    development = project / "results" / "development"
    formal = project / "results" / "formal"
    development.mkdir(parents=True)
    formal.mkdir(parents=True)

    development_tasks: list[dict[str, object]] = []
    for task_index, case_id in enumerate(DEVELOPMENT_IDS):
        views: list[str] = []
        task_root = development / case_id
        task_root.mkdir()
        for view_index in range(3):
            target = task_root / f"view_{view_index}.png"
            target.write_bytes(b"png-development\0" + bytes([task_index, view_index]))
            views.append(_manifest_path(target, project, obsolete_prefix))
        development_tasks.append({"task": case_id, "views": views})
    development_manifest = development / "MANIFEST.json"
    _write_json(
        development_manifest,
        {
            "protocol": "fixed-3-view TFS, mr=1, 256^3",
            "foreground": "white-on-black",
            "tasks": development_tasks,
        },
    )

    formal_cases: list[dict[str, object]] = []
    input_root = formal / "inputs"
    input_root.mkdir()
    for case_index, case_id in enumerate(FORMAL_IDS):
        targets: list[str] = []
        for axis_index, axis in enumerate(AXES):
            target = input_root / f"{case_id}_view_{axis}.png"
            target.write_bytes(b"png-formal\0" + bytes([case_index, axis_index]))
            targets.append(_manifest_path(target, project, obsolete_prefix))
        formal_cases.append(
            {
                "case_id": case_id,
                "target_paths_xyz": targets,
                "sadekar_tfs_targets": [targets[1], targets[0], targets[2]],
                "input_connectivity_audit": [{"quality": 1.0}],
                "visual_hull_audit": {"mean_iou": 0.99},
            }
        )
    formal_manifest = formal / "benchmark_manifest.json"
    _write_json(
        formal_manifest,
        {
            "schema": "fixed3_mr1_connected10_v1",
            "status": "APPROVED_FROZEN",
            "purpose": "synthetic binding fixture",
            "selection_policy": "input-only",
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
            "candidate_pool": {"audited_masks": 72},
            "cases": formal_cases,
        },
    )
    spec = project / "docs" / "revised-design.md"
    spec.parent.mkdir()
    spec.write_text("# frozen synthetic revised design\n", encoding="utf-8")
    return {
        "project": project,
        "development": development,
        "development_manifest": development_manifest,
        "formal": formal,
        "formal_manifest": formal_manifest,
        "spec": spec,
    }


def _suite_semantic_projection(payload: dict[str, object]) -> dict[str, object]:
    fields = list(SUITE_FIELDS)
    if payload.get("role") == "formal":
        fields.append("formal_run_roles")
    return {field: payload[field] for field in fields}


def _write_suite(path: Path, payload: dict[str, object]) -> None:
    payload["semantic_sha256"] = canonical_hash(_suite_semantic_projection(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _bound_suites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    obsolete_prefix: bool = False,
) -> tuple[dict[str, Path], Path, Path, dict[str, object], dict[str, object]]:
    paths = _make_inputs(tmp_path, obsolete_prefix=obsolete_prefix)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    development = dict(bind_development_suite(paths["development"]))
    formal = dict(bind_formal_suite(paths["formal"], paths["formal_manifest"]))
    development_path = tmp_path / "bound" / "development_suite.json"
    formal_path = tmp_path / "bound" / "formal_suite.json"
    _write_suite(development_path, development)
    _write_suite(formal_path, formal)
    return paths, development_path, formal_path, development, formal


def _binding_args(paths: dict[str, Path], output: Path) -> list[str]:
    return [
        "--output-root",
        str(output),
        "--development-root",
        str(paths["development"]),
        "--formal-benchmark",
        str(paths["formal_manifest"]),
        "--spec",
        str(paths["spec"]),
    ]


def test_binding_preserves_exact_task_and_xyz_order_and_ignores_sadekar_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a binder sorts, filters, or substitutes the x/y formal inputs."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])

    development = bind_development_suite(paths["development"])
    formal = bind_formal_suite(paths["formal"], paths["formal_manifest"])

    assert [task["case_id"] for task in development["tasks"]] == list(DEVELOPMENT_IDS)
    assert [task["case_id"] for task in formal["tasks"]] == list(FORMAL_IDS)
    assert development["task_count"] == 3
    assert development["input_count"] == 9
    assert formal["task_count"] == 10
    assert formal["input_count"] == 30
    assert development["input_order"] == ["x", "y", "z"]
    assert formal["input_order"] == ["x", "y", "z"]
    first = formal["tasks"][0]
    assert [item["axis"] for item in first["inputs"]] == ["x", "y", "z"]
    assert [Path(item["resolved_path"]).name for item in first["inputs"]] == [
        "connected_01_view_x.png",
        "connected_01_view_y.png",
        "connected_01_view_z.png",
    ]
    assert formal["primary_variants"] == ["l12_bal", "dic_only", "dts_only", "v3_full"]
    assert formal["appendix_controls"]["included_in_primary_factorial"] is False
    assert formal["formal_run_roles"]["repeat_is_independent_sample"] is False


def test_obsolete_prefix_is_rewritten_exactly_and_both_paths_are_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: inherited ShadowForge inputs are guessed or lose their source path."""

    paths, _, _, development, formal = _bound_suites(
        tmp_path, monkeypatch, obsolete_prefix=True
    )
    for payload in (development, formal):
        for task in payload["tasks"]:
            for item in task["inputs"]:
                assert item["source_path"].startswith(
                    "/home/lin/xvxinyao/ShadowForge/"
                )
                resolved = Path(item["resolved_path"])
                assert resolved.is_relative_to(paths["project"])
                assert item["sha256"] == _sha(resolved)


def test_other_missing_prefix_is_not_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: arbitrary obsolete roots are silently mapped into P-CRAFT."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    payload = json.loads(paths["development_manifest"].read_text(encoding="utf-8"))
    payload["tasks"][0]["views"][0] = "/home/lin/xvxinyao/AlmostShadowForge/results/view_0.png"
    _write_json(paths["development_manifest"], payload)

    with pytest.raises((FileNotFoundError, ValueError), match="missing|rewrite|outside"):
        bind_development_suite(paths["development"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["tasks"].reverse(), "ordered|arcane_casters"),
        (lambda payload: payload["tasks"].append(payload["tasks"][0]), "exactly|duplicate"),
        (lambda payload: payload["tasks"][0]["views"].pop(), "three|x/y/z"),
        (lambda payload: payload["tasks"][0]["views"].__setitem__(0, 7), "string"),
        (
            lambda payload: payload["tasks"][0]["views"].__setitem__(
                slice(0, 2), list(reversed(payload["tasks"][0]["views"][:2]))
            ),
            "order|view_0",
        ),
    ],
)
def test_development_manifest_rejects_bad_ids_views_and_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    message: str,
) -> None:
    """Break caught: malformed development manifests are silently normalized."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    payload = json.loads(paths["development_manifest"].read_text(encoding="utf-8"))
    mutation(payload)
    _write_json(paths["development_manifest"], payload)

    with pytest.raises(ValueError, match=message):
        bind_development_suite(paths["development"])


def test_development_rejects_ambiguous_alternative_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a second plausible development manifest is ignored."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    (paths["development"] / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ambiguous"):
        bind_development_suite(paths["development"])


@pytest.mark.parametrize("mutation", ["status", "order", "count", "non_string"])
def test_formal_manifest_rejects_nonfrozen_or_malformed_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: the formal suite accepts a non-frozen or reordered benchmark."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    payload = json.loads(paths["formal_manifest"].read_text(encoding="utf-8"))
    if mutation == "status":
        payload["status"] = "PENDING"
    elif mutation == "order":
        payload["cases"][0], payload["cases"][1] = payload["cases"][1], payload["cases"][0]
    elif mutation == "count":
        payload["cases"][0]["target_paths_xyz"].pop()
    else:
        payload["cases"][0]["target_paths_xyz"][0] = None
    _write_json(paths["formal_manifest"], payload)

    with pytest.raises(ValueError, match="APPROVED_FROZEN|ordered|three|x/y/z|string"):
        bind_formal_suite(paths["formal"], paths["formal_manifest"])


def test_strict_load_rehashes_manifest_and_inputs_and_returns_frozen_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: loading trusts stored hashes or returns mutable/reordered tasks."""

    paths, development_path, formal_path, _, _ = _bound_suites(tmp_path, monkeypatch)
    development = load_bound_suite(development_path, "development")
    formal = load_bound_suite(formal_path, "formal")

    assert [task.case_id for task in development] == list(DEVELOPMENT_IDS)
    assert [task.case_id for task in formal] == list(FORMAL_IDS)
    assert [Path(path).name for path in formal[0].targets] == [
        "connected_01_view_x.png",
        "connected_01_view_y.png",
        "connected_01_view_z.png",
    ]
    assert all(task.role == "development" for task in development)
    with pytest.raises(FrozenInstanceError):
        development[0].role = "formal"

    target = Path(development[0].targets[0])
    original = target.read_bytes()
    target.write_bytes(original + b"drift")
    with pytest.raises(ValueError, match="input.*sha256|changed"):
        load_bound_suite(development_path, "development")
    target.write_bytes(original)
    paths["formal_manifest"].write_bytes(
        paths["formal_manifest"].read_bytes() + b" "
    )
    with pytest.raises(ValueError, match="manifest.*sha256|changed"):
        load_bound_suite(formal_path, "formal")


@pytest.mark.parametrize("mutation", ["extra", "missing", "duplicate_case", "duplicate_path", "duplicate_hash"])
def test_strict_load_rejects_schema_and_duplicate_attacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: plausible but non-exact suite payloads pass strict loading."""

    _, development_path, _, development, _ = _bound_suites(tmp_path, monkeypatch)
    payload = json.loads(canonical_json_bytes(development))
    if mutation == "extra":
        payload["unregistered"] = True
    elif mutation == "missing":
        del payload["protocol"]
    elif mutation == "duplicate_case":
        payload["tasks"][1]["case_id"] = payload["tasks"][0]["case_id"]
    elif mutation == "duplicate_path":
        payload["tasks"][1]["inputs"][0] = dict(payload["tasks"][0]["inputs"][0])
        payload["tasks"][1]["inputs"][0]["axis"] = "x"
    else:
        first = payload["tasks"][0]["inputs"][0]
        second = payload["tasks"][1]["inputs"][0]
        Path(second["resolved_path"]).write_bytes(Path(first["resolved_path"]).read_bytes())
        second["sha256"] = first["sha256"]
    if mutation != "missing":
        payload["semantic_sha256"] = canonical_hash(_suite_semantic_projection(payload))
    development_path.write_bytes(canonical_json_bytes(payload))

    with pytest.raises(ValueError, match="schema|missing|extra|ordered|duplicate"):
        load_bound_suite(development_path, "development")


def test_strict_load_rejects_noncanonical_duplicate_and_nonfinite_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: permissive JSON accepts whitespace, duplicate keys, or NaN."""

    _, development_path, _, development, _ = _bound_suites(tmp_path, monkeypatch)
    development_path.write_bytes(canonical_json_bytes(development) + b"\n")
    with pytest.raises(ValueError, match="canonical"):
        load_bound_suite(development_path, "development")

    development_path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_bound_suite(development_path, "development")

    development_path.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="finite|constant|NaN"):
        load_bound_suite(development_path, "development")


def test_semantic_hash_uses_scientific_whitelist_not_quality_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: formal quality metrics influence scientific suite semantics."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    first = bind_formal_suite(paths["formal"], paths["formal_manifest"])
    raw = json.loads(paths["formal_manifest"].read_text(encoding="utf-8"))
    raw["cases"][0]["visual_hull_audit"]["mean_iou"] = -123.0
    raw["candidate_pool"]["audited_masks"] = 999999
    _write_json(paths["formal_manifest"], raw)
    second = bind_formal_suite(paths["formal"], paths["formal_manifest"])

    assert first["source_manifest"]["sha256"] != second["source_manifest"]["sha256"]
    assert (
        first["source_manifest"]["semantic_sha256"]
        == second["source_manifest"]["semantic_sha256"]
    )


def test_disjointness_rejects_case_path_and_same_bytes_different_path(
    tmp_path: Path,
) -> None:
    """Break caught: suite overlap is checked by only one of its three identities."""

    files = []
    for index in range(12):
        path = tmp_path / f"input_{index}.png"
        path.write_bytes(f"unique-{index}".encode("ascii"))
        files.append(path)
    development = [
        TaskInputs("same", tuple(str(path) for path in files[0:3]), "development")
    ]
    formal = [TaskInputs("same", tuple(str(path) for path in files[3:6]), "formal")]
    with pytest.raises(ValueError, match="case.*overlap|same"):
        assert_suite_disjoint(development, formal)

    formal = [
        TaskInputs("different", (str(files[0]), str(files[6]), str(files[7])), "formal")
    ]
    with pytest.raises(ValueError, match="path.*overlap|input.*overlap"):
        assert_suite_disjoint(development, formal)

    files[9].write_bytes(files[0].read_bytes())
    formal = [
        TaskInputs("different", tuple(str(path) for path in files[9:12]), "formal")
    ]
    with pytest.raises(ValueError, match="sha256|bytes|hash"):
        assert_suite_disjoint(development, formal)


def test_symlink_manifest_input_and_traversal_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: canonical containment can be bypassed with links or dot-dot."""

    paths = _make_inputs(tmp_path / "manifest")
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    manifest_copy = tmp_path / "manifest-copy.json"
    manifest_copy.write_bytes(paths["development_manifest"].read_bytes())
    paths["development_manifest"].unlink()
    paths["development_manifest"].symlink_to(manifest_copy)
    with pytest.raises(ValueError, match="symlink"):
        bind_development_suite(paths["development"])

    paths = _make_inputs(tmp_path / "input")
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    target = paths["development"] / DEVELOPMENT_IDS[0] / "view_0.png"
    target_copy = tmp_path / "target-copy.png"
    target_copy.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(target_copy)
    with pytest.raises(ValueError, match="symlink|outside"):
        bind_development_suite(paths["development"])

    paths = _make_inputs(tmp_path / "traversal")
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    raw = json.loads(paths["development_manifest"].read_text(encoding="utf-8"))
    original = Path(raw["tasks"][0]["views"][0])
    raw["tasks"][0]["views"][0] = str(original.parent / ".." / original.parent.name / original.name)
    _write_json(paths["development_manifest"], raw)
    with pytest.raises(ValueError, match="traversal|canonical|dot-dot"):
        bind_development_suite(paths["development"])


def test_parser_exposes_only_the_four_frozen_binding_inputs() -> None:
    """Break caught: the binder permits post-hoc task, order, metric, or weight choices."""

    parser = build_binding_parser()
    assert {action.dest for action in parser._actions} == {
        "help",
        "output_root",
        "development_root",
        "formal_benchmark",
        "spec",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--output-root", "out",
                "--development-root", "dev",
                "--formal-benchmark", "formal.json",
                "--spec", "spec.md",
                "--selection-metric", "iou",
            ]
        )
    valid_values = ["out", "dev", "formal.json", "spec.md"]
    for abbreviation in ("--out", "--development-r", "--formal-b", "--sp"):
        argv = [
            "--output-root", valid_values[0],
            "--development-root", valid_values[1],
            "--formal-benchmark", valid_values[2],
            "--spec", valid_values[3],
        ]
        option_index = next(
            index for index, value in enumerate(argv) if value.startswith(abbreviation)
        )
        argv[option_index] = abbreviation
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_binding_main_publishes_once_is_idempotent_and_binds_master(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: repeated binding changes bytes or omits the registered matrix/spec."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    output = tmp_path / "published"
    args = _binding_args(paths, output)

    assert binding_main(args) == 0
    targets = [
        output / "manifests" / "development_suite.json",
        output / "manifests" / "formal_suite.json",
        output / "experiment_manifest.json",
    ]
    first_bytes = [path.read_bytes() for path in targets]
    assert binding_main(args) == 0
    assert [path.read_bytes() for path in targets] == first_bytes

    development = load_bound_suite(targets[0], "development")
    formal = load_bound_suite(targets[1], "formal")
    assert_suite_disjoint(development, formal)
    master = json.loads(targets[2].read_bytes())
    assert targets[2].read_bytes() == canonical_json_bytes(master)
    assert master["design_spec"] == {
        "path": str(paths["spec"].resolve()),
        "sha256": _sha(paths["spec"]),
    }
    assert [row["variant"] for row in master["primary_matrix"]] == [
        "l12_bal",
        "dic_only",
        "dts_only",
        "v3_full",
    ]
    assert master["appendix_controls"]["included_in_primary_factorial"] is False
    assert master["protocol"]["seed"] == 17
    assert master["determinism"]["bitwise_reproducible"] is False
    assert master["suites"]["development"]["path"] == "manifests/development_suite.json"
    assert master["suites"]["formal"]["path"] == "manifests/formal_suite.json"


@pytest.mark.parametrize("drift", ["source", "input", "spec"])
def test_rebinding_after_scientific_drift_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    """Break caught: drift silently replaces an accepted immutable registration."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    output = tmp_path / "published"
    args = _binding_args(paths, output)
    assert binding_main(args) == 0
    accepted = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*.json")
    }
    if drift == "source":
        paths["development_manifest"].write_bytes(
            paths["development_manifest"].read_bytes() + b" "
        )
    elif drift == "input":
        target = paths["formal"] / "inputs" / "connected_01_view_x.png"
        target.write_bytes(target.read_bytes() + b"drift")
    else:
        paths["spec"].write_text("changed spec\n", encoding="utf-8")

    with pytest.raises((FileExistsError, ValueError), match="immutable|changed|differ|overwrite"):
        binding_main(args)
    assert {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*.json")
    } == accepted


def test_partial_existing_set_and_output_symlink_fail_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: partial or symlink output state is completed or overwritten."""

    paths = _make_inputs(tmp_path / "partial")
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    output = tmp_path / "partial-output"
    partial = output / "manifests" / "development_suite.json"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"pre-existing")
    with pytest.raises(FileExistsError, match="partial|immutable"):
        binding_main(_binding_args(paths, output))
    assert partial.read_bytes() == b"pre-existing"
    assert not (output / "manifests" / "formal_suite.json").exists()
    assert not (output / "experiment_manifest.json").exists()

    paths = _make_inputs(tmp_path / "symlink")
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    output = tmp_path / "symlink-output"
    manifests = output / "manifests"
    manifests.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"outside-sentinel")
    (manifests / "development_suite.json").symlink_to(outside)
    with pytest.raises((FileExistsError, ValueError), match="symlink|immutable"):
        binding_main(_binding_args(paths, output))
    assert outside.read_bytes() == b"outside-sentinel"
    assert not (manifests / "formal_suite.json").exists()
    assert not (output / "experiment_manifest.json").exists()


def test_publication_failure_rolls_back_every_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a mid-publication failure leaves a plausible partial binding."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    output = tmp_path / "published"
    original_link = registry_module.os.link
    calls = 0

    def fail_second_link(source: object, destination: object, *args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(registry_module.os, "link", fail_second_link)
    with pytest.raises(OSError, match="injected"):
        binding_main(_binding_args(paths, output))
    assert not (output / "manifests" / "development_suite.json").exists()
    assert not (output / "manifests" / "formal_suite.json").exists()
    assert not (output / "experiment_manifest.json").exists()


def test_importing_binding_modules_performs_no_writes(tmp_path: Path) -> None:
    """Break caught: importing either CLI module starts binding or writes files."""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; before=list(Path('.').iterdir()); "
                "import experiments.soft_loss_v3_registry; "
                "import experiments.bind_soft_loss_v3_suites; "
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


def test_direct_binding_entrypoint_exposes_help_without_import_failure() -> None:
    """Break caught: the executable works with -m but not as a script path."""

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "bind_soft_loss_v3_suites.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--development-root" in completed.stdout
    assert "--formal-benchmark" in completed.stdout


def _install_mutation_after_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
) -> dict[str, bool]:
    original_snapshot = getattr(registry_module, "_snapshot_file", None)
    state = {"mutated": False}

    def mutate_after_snapshot(path: Path, *args: object, **kwargs: object) -> object:
        if original_snapshot is None:
            raise AssertionError("stable snapshot seam is missing")
        snapshot = original_snapshot(path, *args, **kwargs)
        if Path(path).resolve() == target.resolve() and not state["mutated"]:
            target.write_bytes(target.read_bytes() + b"reviewer-drift")
            state["mutated"] = True
        return snapshot

    monkeypatch.setattr(
        registry_module,
        "_snapshot_file",
        mutate_after_snapshot,
        raising=False,
    )
    return state


@pytest.mark.parametrize("dependency", ["formal_source", "formal_input"])
def test_strict_loader_rejects_dependency_mutated_after_its_first_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    """Break caught: strict load accepts source/input drift after its first hash/read."""

    paths, _, formal_path, _, _ = _bound_suites(tmp_path, monkeypatch)
    target = (
        paths["formal_manifest"]
        if dependency == "formal_source"
        else paths["formal"] / "inputs" / "connected_01_view_x.png"
    )
    state = _install_mutation_after_snapshot(monkeypatch, target)

    with pytest.raises(ValueError, match="changed|snapshot|drift|stable"):
        load_bound_suite(formal_path, "formal")
    assert state["mutated"] is True


def test_spec_mutated_after_snapshot_prevents_every_final_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: the master publishes a stale spec hash after spec bytes drift."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    output = tmp_path / "published"
    state = _install_mutation_after_snapshot(monkeypatch, paths["spec"])

    with pytest.raises(ValueError, match="changed|snapshot|drift|stable"):
        binding_main(_binding_args(paths, output))
    assert state["mutated"] is True
    assert not (output / "manifests" / "development_suite.json").exists()
    assert not (output / "manifests" / "formal_suite.json").exists()
    assert not (output / "experiment_manifest.json").exists()


def test_input_mutated_during_fd_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: hashing a file while it changes binds a blended byte stream."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    target = paths["development"] / DEVELOPMENT_IDS[0] / "view_0.png"
    target.write_bytes(b"a" * (2 * 1024 * 1024 + 17))
    original_fstat = registry_module.os.fstat
    state = {"target_fstats": 0, "mutated": False}

    def mutate_before_second_fstat(fd: int) -> os.stat_result:
        try:
            opened = Path(f"/proc/self/fd/{fd}").resolve()
        except OSError:
            opened = Path("/")
        if opened == target.resolve():
            state["target_fstats"] += 1
            if state["target_fstats"] == 2:
                target.write_bytes(target.read_bytes() + b"x")
                state["mutated"] = True
        return original_fstat(fd)

    monkeypatch.setattr(registry_module.os, "fstat", mutate_before_second_fstat)
    with pytest.raises(ValueError, match="changed|snapshot|drift|stable"):
        bind_development_suite(paths["development"])
    assert state["mutated"] is True


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [("mirror_enabled", 1), ("views", 3.0), ("seed", 17.0)],
)
def test_bound_suite_protocol_rejects_numeric_equality_type_coercion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: object,
) -> None:
    """Break caught: Python equality accepts bool/int/float protocol substitutions."""

    _, development_path, _, development, _ = _bound_suites(tmp_path, monkeypatch)
    payload = json.loads(canonical_json_bytes(development))
    payload["protocol"][field] = bad_value
    _write_suite(development_path, payload)

    with pytest.raises(ValueError, match="type|exact|protocol"):
        load_bound_suite(development_path, "development")


@pytest.mark.parametrize(
    "mutation",
    [
        "development_protocol",
        "development_foreground",
        "formal_camera",
        "formal_tsf_bool",
        "formal_mr_bool",
        "formal_views_float",
        "formal_seed_bool",
        "formal_evaluation",
    ],
)
def test_authoritative_source_protocol_must_match_frozen_declarations_zero_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Break caught: arbitrary or type-coerced source protocol becomes a frozen suite."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    if mutation.startswith("development"):
        manifest = paths["development_manifest"]
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if mutation == "development_protocol":
            payload["protocol"] = "arbitrary perspective protocol"
        else:
            payload["foreground"] = "black-on-white"
    else:
        manifest = paths["formal_manifest"]
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        protocol = payload["protocol"]
        if mutation == "formal_camera":
            protocol["camera"] = "perspective"
        elif mutation == "formal_tsf_bool":
            protocol["tsf"] = True
        elif mutation == "formal_mr_bool":
            protocol["mr"] = True
        elif mutation == "formal_views_float":
            protocol["views"] = 3.0
        elif mutation == "formal_seed_bool":
            protocol["seeds"] = [True, 29, 43]
        else:
            protocol["evaluation"] = "registration permitted"
    _write_json(manifest, payload)
    output = tmp_path / "published"

    with pytest.raises(ValueError, match="protocol|foreground|camera|TFS|mirror|exact|type"):
        binding_main(_binding_args(paths, output))
    assert not (output / "manifests" / "development_suite.json").exists()
    assert not (output / "manifests" / "formal_suite.json").exists()
    assert not (output / "experiment_manifest.json").exists()


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _run_crashing_publisher(args: list[str]) -> subprocess.CompletedProcess[str]:
    script = r"""
import os
import sys
import experiments.soft_loss_v3_registry as registry

original_link = registry.os.link
count = 0

def crash_after_first_link(source, destination, *args, **kwargs):
    global count
    original_link(source, destination, *args, **kwargs)
    count += 1
    if count == 1:
        os._exit(73)

registry.os.link = crash_after_first_link
raise SystemExit(registry.binding_main(sys.argv[1:]))
"""
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        check=False,
        capture_output=True,
        text=True,
    )


def _leave_complete_owned_stage(
    output: Path,
    files: dict[Path, bytes],
) -> subprocess.CompletedProcess[str]:
    """Crash after terminal validation but before an owned stage is cleaned."""

    encoded = json.dumps(
        [[str(path), payload.hex()] for path, payload in files.items()]
    )
    script = r"""
import json
import os
import sys
from pathlib import Path
import experiments.soft_loss_v3_registry as registry

output = Path(sys.argv[1])
files = {Path(path): bytes.fromhex(payload) for path, payload in json.loads(sys.argv[2])}

def crash_before_stage_cleanup(*args, **kwargs):
    os._exit(73)

registry._remove_owned_stage = crash_before_stage_cleanup
registry._publish_bundle(files, output_root=output)
"""
    return subprocess.run(
        [sys.executable, "-c", script, str(output), encoded],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        check=False,
        capture_output=True,
        text=True,
    )


def _crash_during_stage_lifecycle(
    output: Path,
    files: dict[Path, bytes],
    *,
    crash_point: str,
) -> subprocess.CompletedProcess[str]:
    encoded = json.dumps(
        [[str(path), payload.hex()] for path, payload in files.items()]
    )
    script = r"""
import json
import os
import sys
from pathlib import Path
import experiments.soft_loss_v3_registry as registry

output = Path(sys.argv[1])
files = {Path(path): bytes.fromhex(payload) for path, payload in json.loads(sys.argv[2])}
crash_point = sys.argv[3]

if crash_point in {"mid_manifest", "mid_payload"}:
    original_write = registry.os.write
    victim = (
        "stage_manifest.json" if crash_point == "mid_manifest" else "payload_000.json"
    )
    returncode = 78 if crash_point == "mid_manifest" else 79

    def crash_after_partial_canonical_write(descriptor, data):
        path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        if path.name == victim or path.name.startswith(
            f".pcraft-soft-loss-v3-write-{victim}-"
        ):
            partial = data[: max(1, len(data) // 2)]
            written = original_write(descriptor, partial)
            os.fsync(descriptor)
            if written != len(partial):
                raise AssertionError("partial crash probe did not write its exact prefix")
            os._exit(returncode)
        return original_write(descriptor, data)

    registry.os.write = crash_after_partial_canonical_write
elif crash_point == "cleanup":
    original_unlink = registry.os.unlink

    def crash_after_first_stage_cleanup_unlink(name, *args, **kwargs):
        descriptor = kwargs.get("dir_fd")
        parent = None
        if isinstance(descriptor, int):
            parent = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        original_unlink(name, *args, **kwargs)
        if (
            parent is not None
            and parent.name.startswith(".pcraft-soft-loss-v3-")
        ):
            os._exit(74)

    registry.os.unlink = crash_after_first_stage_cleanup_unlink
elif crash_point == "cleanup_rename":
    original_rename = registry.os.rename

    def crash_after_first_inner_cleanup_rename(source, destination, *args, **kwargs):
        descriptor = kwargs.get("src_dir_fd")
        parent = None
        if isinstance(descriptor, int):
            parent = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        original_rename(source, destination, *args, **kwargs)
        if (
            parent is not None
            and parent.name.startswith(
                (
                    ".pcraft-soft-loss-v3-cleanup-",
                    ".pcraft-soft-loss-v3-retired-",
                )
            )
            and str(destination).startswith(".pcraft-soft-loss-v3-remove-")
        ):
            os._exit(76)

    registry.os.rename = crash_after_first_inner_cleanup_rename
elif crash_point == "write":
    original_write = registry._write_regular_at
    writes = 0

    def crash_after_first_stage_write(*args, **kwargs):
        global writes
        binding = original_write(*args, **kwargs)
        writes += 1
        if writes == 1:
            os._exit(75)
        return binding

    registry._write_regular_at = crash_after_first_stage_write
else:
    raise AssertionError(crash_point)

registry._publish_bundle(files, output_root=output)
"""
    return subprocess.run(
        [sys.executable, "-c", script, str(output), encoded, crash_point],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        check=False,
        capture_output=True,
        text=True,
    )


def _crash_between_target_rollback_rename_and_unlink(
    output: Path,
    files: dict[Path, bytes],
) -> subprocess.CompletedProcess[str]:
    encoded = json.dumps(
        [[str(path), payload.hex()] for path, payload in files.items()]
    )
    script = r"""
import json
import os
import sys
from pathlib import Path
import experiments.soft_loss_v3_registry as registry

output = Path(sys.argv[1])
files = {Path(path): bytes.fromhex(payload) for path, payload in json.loads(sys.argv[2])}
original_unlink = registry.os.unlink
precommits = 0

def fail_after_first_link():
    global precommits
    precommits += 1
    if precommits == 3:
        raise RuntimeError("force rollback after first immutable link")

def crash_before_target_remove_unlink(name, *args, **kwargs):
    descriptor = kwargs.get("dir_fd")
    if isinstance(name, str) and name.startswith(".pcraft-soft-loss-v3-remove-"):
        parent = None
        if isinstance(descriptor, int):
            parent = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        if parent is not None and not parent.name.startswith(
            ".pcraft-soft-loss-v3-retired-"
        ):
            os._exit(93)
    return original_unlink(name, *args, **kwargs)

registry.os.unlink = crash_before_target_remove_unlink
registry._publish_bundle(files, output_root=output, precommit=fail_after_first_link)
"""
    return subprocess.run(
        [sys.executable, "-c", script, str(output), encoded],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        check=False,
        capture_output=True,
        text=True,
    )


def _run_parent_swap_publisher(
    args: list[str], output: Path, outside: Path
) -> subprocess.CompletedProcess[str]:
    script = r"""
import os
import sys
from pathlib import Path
import experiments.soft_loss_v3_registry as registry

output = Path(sys.argv[1])
outside = Path(sys.argv[2])
original_link = registry.os.link
swapped = False

def swap_parent_before_first_final_link(source, destination, *args, **kwargs):
    global swapped
    if not swapped:
        os.rename(output / "manifests", output / "manifests-held")
        outside.mkdir(parents=True)
        os.symlink(outside, output / "manifests", target_is_directory=True)
        swapped = True
    return original_link(source, destination, *args, **kwargs)

registry.os.link = swap_parent_before_first_final_link
raise SystemExit(registry.binding_main(sys.argv[3:]))
"""
    return subprocess.run(
        [sys.executable, "-c", script, str(output), str(outside), *args],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        check=False,
        capture_output=True,
        text=True,
    )


def test_parent_swap_cannot_redirect_publication_or_leave_external_partial(
    tmp_path: Path,
) -> None:
    """Break caught: a validated parent is replaced before link and receives finals outside."""

    paths = _make_inputs(tmp_path / "inputs")
    output = tmp_path / "published"
    outside = tmp_path / "outside"
    completed = _run_parent_swap_publisher(
        _binding_args(paths, output), output, outside
    )

    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert not list(outside.rglob("*"))
    held = output / "manifests-held"
    assert not held.joinpath("development_suite.json").exists()
    assert not held.joinpath("formal_suite.json").exists()
    assert not (output / "experiment_manifest.json").exists()


def test_crash_after_first_final_link_is_recovered_by_identical_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a process death strands an unrecoverable one-of-three bundle."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    output = tmp_path / "published"
    args = _binding_args(paths, output)

    crashed = _run_crashing_publisher(args)
    assert crashed.returncode == 73, crashed.stderr
    finals = [
        output / "manifests" / "development_suite.json",
        output / "manifests" / "formal_suite.json",
        output / "experiment_manifest.json",
    ]
    assert sum(path.exists() for path in finals) == 1

    assert binding_main(args) == 0
    development = load_bound_suite(finals[0], "development")
    formal = load_bound_suite(finals[1], "formal")
    assert_suite_disjoint(development, formal)
    assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()) == [
        "experiment_manifest.json",
        "manifests/development_suite.json",
        "manifests/formal_suite.json",
    ]


def _start_synchronized_publisher(
    *,
    barrier: Path,
    token: str,
    expected: int,
    args: list[str],
) -> subprocess.Popen[str]:
    script = r"""
import sys
import time
from pathlib import Path
import experiments.soft_loss_v3_registry as registry

barrier = Path(sys.argv[1])
token = sys.argv[2]
expected = int(sys.argv[3])
original_publish = registry._publish_bundle

def synchronized_publish(files, *args, **kwargs):
    barrier.mkdir(parents=True, exist_ok=True)
    (barrier / ("ready-" + token)).write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + 10.0
    while len(list(barrier.glob("ready-*"))) < expected:
        if time.monotonic() >= deadline:
            raise RuntimeError("publisher barrier timeout")
        time.sleep(0.01)
    return original_publish(files, *args, **kwargs)

registry._publish_bundle = synchronized_publish
raise SystemExit(registry.binding_main(sys.argv[4:]))
"""
    return subprocess.Popen(
        [sys.executable, "-c", script, str(barrier), token, str(expected), *args],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _communicate(process: subprocess.Popen[str]) -> tuple[int, str, str]:
    stdout, stderr = process.communicate(timeout=20)
    return process.returncode, stdout, stderr


def test_two_simultaneous_identical_publishers_both_converge_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: an identical concurrent binder loses an exclusive-link race."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    output = tmp_path / "published"
    args = _binding_args(paths, output)
    barrier = tmp_path / "barrier"
    processes = [
        _start_synchronized_publisher(
            barrier=barrier,
            token=str(index),
            expected=2,
            args=args,
        )
        for index in range(2)
    ]
    outcomes = [_communicate(process) for process in processes]

    assert [result[0] for result in outcomes] == [0, 0], outcomes
    development = load_bound_suite(
        output / "manifests" / "development_suite.json", "development"
    )
    formal = load_bound_suite(output / "manifests" / "formal_suite.json", "formal")
    assert_suite_disjoint(development, formal)


def test_two_simultaneous_nonidentical_publishers_produce_one_coherent_winner(
    tmp_path: Path,
) -> None:
    """Break caught: competing nonidentical bundles interleave into a mixed registry."""

    first = _make_inputs(tmp_path / "first")
    second = _make_inputs(tmp_path / "second")
    output = tmp_path / "published"
    barrier = tmp_path / "barrier"
    processes = [
        _start_synchronized_publisher(
            barrier=barrier,
            token="first",
            expected=2,
            args=_binding_args(first, output),
        ),
        _start_synchronized_publisher(
            barrier=barrier,
            token="second",
            expected=2,
            args=_binding_args(second, output),
        ),
    ]
    outcomes = [_communicate(process) for process in processes]

    assert sorted(result[0] == 0 for result in outcomes) == [False, True], outcomes
    development = load_bound_suite(
        output / "manifests" / "development_suite.json", "development"
    )
    formal = load_bound_suite(output / "manifests" / "formal_suite.json", "formal")
    assert_suite_disjoint(development, formal)
    accepted_roots = {first["project"].resolve(), second["project"].resolve()}
    observed_root = next(
        root
        for root in accepted_roots
        if all(Path(target).is_relative_to(root) for task in development for target in task.targets)
    )
    assert all(Path(target).is_relative_to(observed_root) for task in formal for target in task.targets)


@pytest.mark.parametrize("dependency", ["formal_source", "formal_input", "spec"])
def test_post_stage_dependency_drift_rejects_before_first_final_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
) -> None:
    """Break caught: bytes can drift after staging but before all three final links."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    target = {
        "formal_source": paths["formal_manifest"],
        "formal_input": paths["formal"] / "inputs" / "connected_01_view_x.png",
        "spec": paths["spec"],
    }[dependency]
    output = tmp_path / "published"
    original_write_stage = registry_module._write_stage
    state = {"mutated": False}

    def mutate_after_stage(*args: object, **kwargs: object) -> object:
        result = original_write_stage(*args, **kwargs)
        target.write_bytes(target.read_bytes() + b"post-stage-reviewer-drift")
        state["mutated"] = True
        return result

    monkeypatch.setattr(registry_module, "_write_stage", mutate_after_stage)
    with pytest.raises(ValueError, match="changed|dependency|snapshot|drift|stable"):
        binding_main(_binding_args(paths, output))

    assert state["mutated"] is True
    assert not (output / "manifests" / "development_suite.json").exists()
    assert not (output / "manifests" / "formal_suite.json").exists()
    assert not (output / "experiment_manifest.json").exists()
    assert not list(output.glob(".pcraft-soft-loss-v3-stage-*"))


def test_post_link_drift_rolls_back_retry_finals_and_preserves_recovery_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: final revalidation is absent or removes a preexisting accepted link."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    output = tmp_path / "published"
    args = _binding_args(paths, output)
    crashed = _run_crashing_publisher(args)
    assert crashed.returncode == 73, crashed.stderr
    finals = [
        output / "manifests" / "development_suite.json",
        output / "manifests" / "formal_suite.json",
        output / "experiment_manifest.json",
    ]
    preexisting = [path for path in finals if path.exists()]
    assert len(preexisting) == 1
    accepted_bytes = preexisting[0].read_bytes()
    original_link = registry_module.os.link
    state = {"links": 0, "mutated": False}

    def mutate_spec_after_last_missing_link(
        source: object,
        destination: object,
        *link_args: object,
        **link_kwargs: object,
    ) -> None:
        original_link(source, destination, *link_args, **link_kwargs)
        state["links"] += 1
        if state["links"] == 2:
            paths["spec"].write_bytes(
                paths["spec"].read_bytes() + b"post-link-reviewer-drift"
            )
            state["mutated"] = True

    monkeypatch.setattr(registry_module.os, "link", mutate_spec_after_last_missing_link)
    with pytest.raises(ValueError, match="changed|dependency|snapshot|drift|stable"):
        binding_main(args)

    assert state == {"links": 2, "mutated": True}
    assert preexisting[0].read_bytes() == accepted_bytes
    assert [path.exists() for path in finals].count(True) == 1
    assert len(list(output.glob(".pcraft-soft-loss-v3-stage-*"))) == 1


@pytest.mark.parametrize("fail_at_precommit", [2, 3])
def test_partial_retry_precommit_failure_preserves_unique_recovery_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at_precommit: int,
) -> None:
    """Break caught: a transient retry failure destroys the only recoverable stage."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    output = tmp_path / "published"
    args = _binding_args(paths, output)
    crashed = _run_crashing_publisher(args)
    assert crashed.returncode == 73, crashed.stderr
    finals = [
        output / "manifests" / "development_suite.json",
        output / "manifests" / "formal_suite.json",
        output / "experiment_manifest.json",
    ]
    preexisting = [path for path in finals if path.exists()]
    stages = list(output.glob(".pcraft-soft-loss-v3-stage-*"))
    assert len(preexisting) == 1
    assert len(stages) == 1
    accepted_path = preexisting[0]
    accepted_bytes = accepted_path.read_bytes()
    stage_path = stages[0]
    stage_bytes = {
        entry.name: entry.read_bytes()
        for entry in stage_path.iterdir()
        if entry.is_file()
    }

    original_publish = registry_module._publish_bundle
    state = {"failed": False}

    def fail_one_retry_precommit(
        files: object,
        *,
        output_root: Path,
        precommit: object,
    ) -> None:
        calls = 0

        def injected_precommit() -> None:
            nonlocal calls
            assert callable(precommit)
            precommit()
            calls += 1
            if not state["failed"] and calls == fail_at_precommit:
                state["failed"] = True
                raise RuntimeError("injected transient retry precommit failure")

        original_publish(
            files,
            output_root=output_root,
            precommit=injected_precommit,
        )

    monkeypatch.setattr(
        registry_module, "_publish_bundle", fail_one_retry_precommit
    )
    with pytest.raises(RuntimeError, match="transient retry precommit"):
        binding_main(args)

    assert state["failed"] is True
    assert accepted_path.read_bytes() == accepted_bytes
    assert [path.exists() for path in finals].count(True) == 1
    assert stage_path.is_dir()
    assert {
        entry.name: entry.read_bytes()
        for entry in stage_path.iterdir()
        if entry.is_file()
    } == stage_bytes

    assert binding_main(args) == 0
    assert all(path.is_file() for path in finals)
    assert not list(output.glob(".pcraft-soft-loss-v3-stage-*"))


def test_stage_replacement_during_cleanup_preserves_replacement_and_outside_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: cleanup follows a replaced stage pathname and deletes foreign bytes."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    output = tmp_path / "published"
    outside = tmp_path / "outside"
    args = _binding_args(paths, output)
    original_publish = registry_module._publish_bundle
    original_link = registry_module.os.link
    state: dict[str, object] = {"links": 0, "stage": None, "held": None}
    replacement_bytes = {
        "payload_000.json": b"foreign replacement payload zero",
        "payload_001.json": b"foreign replacement payload one",
        "payload_002.json": b"foreign replacement payload two",
        "stage_manifest.json": b"foreign replacement manifest",
        "sentinel.txt": b"must survive fail-closed cleanup",
    }

    def fail_postlink_precommit(
        files: object,
        *,
        output_root: Path,
        precommit: object,
    ) -> None:
        calls = 0

        def injected_precommit() -> None:
            nonlocal calls
            assert callable(precommit)
            precommit()
            calls += 1
            if calls == 3:
                raise RuntimeError("injected postlink cleanup failure")

        original_publish(
            files,
            output_root=output_root,
            precommit=injected_precommit,
        )

    def replace_stage_after_open(
        source: object,
        destination: object,
        *link_args: object,
        **link_kwargs: object,
    ) -> None:
        original_link(source, destination, *link_args, **link_kwargs)
        state["links"] = int(state["links"]) + 1
        if state["links"] != 1:
            return
        stage = next(output.glob(".pcraft-soft-loss-v3-stage-*"))
        outside.mkdir()
        held = outside / stage.name
        os.rename(stage, held)
        stage.mkdir(mode=0o700)
        for name, data in replacement_bytes.items():
            stage.joinpath(name).write_bytes(data)
        state["stage"] = stage
        state["held"] = held

    monkeypatch.setattr(
        registry_module, "_publish_bundle", fail_postlink_precommit
    )
    monkeypatch.setattr(registry_module.os, "link", replace_stage_after_open)
    with pytest.raises(
        (RuntimeError, ValueError, OSError),
        match="cleanup|publication|replaced|postlink|[Dd]irectory",
    ):
        binding_main(args)

    stage = state["stage"]
    held = state["held"]
    assert isinstance(stage, Path) and isinstance(held, Path)
    assert {
        entry.name: entry.read_bytes()
        for entry in stage.iterdir()
        if entry.is_file()
    } == replacement_bytes
    assert held.is_dir()
    assert {entry.name for entry in held.iterdir()} == {
        "payload_000.json",
        "payload_001.json",
        "payload_002.json",
        "stage_manifest.json",
    }
    assert not (output / "manifests" / "development_suite.json").exists()
    assert not (output / "manifests" / "formal_suite.json").exists()
    assert not (output / "experiment_manifest.json").exists()


def test_staged_payload_inode_replacement_before_link_cannot_publish_foreign_bytes(
    tmp_path: Path,
) -> None:
    """Break caught: a validated staged name is replaced before its final hard link."""

    output = tmp_path / "published"
    target = output / "manifests" / "accepted.json"
    expected = b'{"accepted":true}\n'
    calls = 0

    def replace_staged_payload_after_load() -> None:
        nonlocal calls
        calls += 1
        if calls != 2:
            return
        stage = next(output.glob(".pcraft-soft-loss-v3-stage-*"))
        payload = stage / "payload_000.json"
        payload.unlink()
        payload.write_bytes(b'{"foreign":true}\n')

    with pytest.raises(
        (ValueError, OSError),
        match="stage|payload|inode|changed|replaced|publication",
    ):
        registry_module._publish_bundle(
            {target: expected},
            output_root=output,
            precommit=replace_staged_payload_after_load,
        )

    assert calls == 2
    assert not os.path.lexists(target)


def test_stage_creation_is_bound_to_locked_root_when_lexical_root_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: pathname stage creation follows a swapped output-root symlink."""

    output = tmp_path / "published"
    held = tmp_path / "published-held"
    outside = tmp_path / "outside"
    target = output / "manifests" / "accepted.json"
    original_uuid4 = registry_module.uuid.uuid4
    state = {"swapped": False}

    def swap_root_before_stage_name() -> object:
        if not state["swapped"]:
            os.rename(output, held)
            outside.mkdir()
            os.symlink(outside, output, target_is_directory=True)
            state["swapped"] = True
        return original_uuid4()

    monkeypatch.setattr(registry_module.uuid, "uuid4", swap_root_before_stage_name)
    with pytest.raises(
        (ValueError, OSError),
        match="root|stage|publication|replaced|changed|missing",
    ):
        registry_module._publish_bundle(
            {target: b'{"accepted":true}\n'},
            output_root=output,
        )

    assert state["swapped"] is True
    assert not list(outside.glob(".pcraft-soft-loss-v3-stage-*"))
    assert not list(held.glob(".pcraft-soft-loss-v3-stage-*"))
    assert not (outside / "manifests" / "accepted.json").exists()
    assert not (held / "manifests" / "accepted.json").exists()


def test_cleanup_stat_unlink_gap_cannot_delete_replacement_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: cleanup validates one inode and unlinks a foreign replacement."""

    output = tmp_path / "published"
    target = output / "accepted.json"
    foreign = b"foreign payload inserted after cleanup stat"
    original_stat = registry_module.os.stat
    original_unlink = registry_module.os.unlink
    state: dict[str, object] = {"stage": None, "replaced": False}

    def replace_payload_after_cleanup_stat(
        path: object,
        *args: object,
        **kwargs: object,
    ) -> os.stat_result:
        observed = original_stat(path, *args, **kwargs)
        if (
            not state["replaced"]
            and path == "payload_000.json"
            and isinstance(kwargs.get("dir_fd"), int)
            and kwargs.get("follow_symlinks") is False
        ):
            descriptor = int(kwargs["dir_fd"])
            stage = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            original_unlink(path, dir_fd=descriptor)
            replacement = os.open(
                str(path),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=descriptor,
            )
            try:
                os.write(replacement, foreign)
                os.fsync(replacement)
            finally:
                os.close(replacement)
            state.update({"stage": stage, "replaced": True})
        return observed

    monkeypatch.setattr(registry_module.os, "stat", replace_payload_after_cleanup_stat)
    with pytest.raises(
        (ValueError, OSError),
        match="stage|payload|inode|changed|replaced|cleanup",
    ):
        registry_module._publish_bundle(
            {target: b'{"accepted":true}\n'},
            output_root=output,
        )

    assert state["replaced"] is True
    stage = state["stage"]
    assert isinstance(stage, Path)
    assert stage.joinpath("payload_000.json").read_bytes() == foreign


def test_rollback_unlinks_created_target_only_if_recorded_inode_still_matches(
    tmp_path: Path,
) -> None:
    """Break caught: rollback unconditionally deletes a foreign final replacement."""

    output = tmp_path / "published"
    first = output / "manifests" / "first.json"
    second = output / "manifests" / "second.json"
    foreign = b"foreign replacement at created final name"
    calls = 0

    def replace_created_final_before_rollback() -> None:
        nonlocal calls
        calls += 1
        if calls != 3:
            return
        first.unlink()
        first.write_bytes(foreign)
        raise RuntimeError("injected postlink rollback")

    with pytest.raises(RuntimeError, match="postlink rollback"):
        registry_module._publish_bundle(
            {
                first: b'{"first":true}\n',
                second: b'{"second":true}\n',
            },
            output_root=output,
            precommit=replace_created_final_before_rollback,
        )

    assert calls == 3
    assert first.read_bytes() == foreign
    assert not second.exists()


@pytest.mark.parametrize("mutation", ["bytes", "inode"])
def test_all_exists_idempotent_terminally_revalidates_final_bytes_and_inode(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Break caught: idempotent publication trusts finals changed at its last precommit."""

    output = tmp_path / "published"
    target = output / "accepted.json"
    expected = b'{"accepted":true}\n'
    registry_module._publish_bundle({target: expected}, output_root=output)
    initial = target.stat()
    calls = 0

    def replace_final_after_last_precommit() -> None:
        nonlocal calls
        calls += 1
        if calls != 2:
            return
        target.unlink()
        target.write_bytes(
            b'{"foreign":true}\n' if mutation == "bytes" else expected
        )

    with pytest.raises(
        (ValueError, OSError),
        match="final|publication|inode|changed|replaced|bytes",
    ):
        registry_module._publish_bundle(
            {target: expected},
            output_root=output,
            precommit=replace_final_after_last_precommit,
        )

    assert calls == 2
    current = target.stat()
    assert (current.st_dev, current.st_ino) != (initial.st_dev, initial.st_ino)
    assert target.read_bytes() == (
        b'{"foreign":true}\n' if mutation == "bytes" else expected
    )


def test_terminal_stage_revalidation_precedes_cleanup_after_last_precommit(
    tmp_path: Path,
) -> None:
    """Break caught: the final audit omits the staged inode immediately before cleanup."""

    output = tmp_path / "published"
    target = output / "accepted.json"
    expected = b'{"accepted":true}\n'
    foreign = b'{"foreign-stage":true}\n'
    calls = 0
    state: dict[str, object] = {"stage": None}

    def replace_stage_after_last_precommit() -> None:
        nonlocal calls
        calls += 1
        if calls != 3:
            return
        stage = next(output.glob(".pcraft-soft-loss-v3-stage-*"))
        payload = stage / "payload_000.json"
        payload.unlink()
        payload.write_bytes(foreign)
        state["stage"] = stage

    with pytest.raises(
        (ValueError, OSError),
        match="stage|payload|inode|changed|replaced|publication",
    ):
        registry_module._publish_bundle(
            {target: expected},
            output_root=output,
            precommit=replace_stage_after_last_precommit,
        )

    assert calls == 3
    stage = state["stage"]
    assert isinstance(stage, Path)
    assert stage.joinpath("payload_000.json").read_bytes() == foreign


@pytest.mark.parametrize("victim_kind", ["preexisting", "created"])
@pytest.mark.parametrize("mutation", ["bytes", "inode"])
def test_partial_recovery_terminally_revalidates_every_final_and_keeps_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    victim_kind: str,
    mutation: str,
) -> None:
    """Break caught: partial recovery accepts a final replaced at last precommit."""

    paths = _make_inputs(tmp_path)
    monkeypatch.setattr(registry_module, "PROJECT_ROOT", paths["project"])
    output = tmp_path / "published"
    args = _binding_args(paths, output)
    crashed = _run_crashing_publisher(args)
    assert crashed.returncode == 73, crashed.stderr
    finals = [
        output / "manifests" / "development_suite.json",
        output / "manifests" / "formal_suite.json",
        output / "experiment_manifest.json",
    ]
    preexisting = [path for path in finals if path.exists()]
    assert len(preexisting) == 1
    victim = (
        preexisting[0]
        if victim_kind == "preexisting"
        else next(path for path in finals if path not in preexisting)
    )
    stage = next(output.glob(".pcraft-soft-loss-v3-stage-*"))
    stage_bytes = {
        entry.name: entry.read_bytes()
        for entry in stage.iterdir()
        if entry.is_file()
    }
    original_publish = registry_module._publish_bundle
    state = {"calls": 0, "replacement": b""}

    def inject_terminal_replacement(
        files: object,
        *,
        output_root: Path,
        precommit: object,
    ) -> None:
        def injected_precommit() -> None:
            assert callable(precommit)
            precommit()
            state["calls"] = int(state["calls"]) + 1
            if state["calls"] != 3:
                return
            accepted = victim.read_bytes()
            replacement = (
                b'{"foreign-final":true}\n'
                if mutation == "bytes"
                else accepted
            )
            victim.unlink()
            victim.write_bytes(replacement)
            state["replacement"] = replacement

        original_publish(
            files,
            output_root=output_root,
            precommit=injected_precommit,
        )

    monkeypatch.setattr(
        registry_module, "_publish_bundle", inject_terminal_replacement
    )
    with pytest.raises(
        (ValueError, OSError),
        match="final|publication|inode|changed|replaced|bytes",
    ):
        binding_main(args)

    assert state["calls"] == 3
    assert victim.read_bytes() == state["replacement"]
    remaining_stages = list(output.glob(".pcraft-soft-loss-v3-stage-*"))
    assert remaining_stages == [stage]
    assert {
        entry.name: entry.read_bytes()
        for entry in stage.iterdir()
        if entry.is_file()
    } == stage_bytes


def test_report_link_is_revalidated_before_authorization_commit_and_recovers(
    tmp_path: Path,
) -> None:
    """Break caught: report and authorization link without a dependency check between."""

    output = tmp_path / "published"
    report = output / "analysis" / "smoke128_gate.json"
    authorization = output / "authorizations" / "smoke128__to__screen128.json"
    dependency = tmp_path / "dependency.json"
    dependency.write_bytes(b"stable dependency")
    report_bytes = b'{"kind":"report"}\n'
    authorization_bytes = b'{"kind":"authorization"}\n'
    state = {"report_only_seen": False}

    def drift_only_in_report_window() -> None:
        if report.exists() and not authorization.exists() and not state["report_only_seen"]:
            state["report_only_seen"] = True
            dependency.write_bytes(b"drift after report link")
            raise RuntimeError("dependency drift in report-only window")

    with pytest.raises(RuntimeError, match="report-only window"):
        registry_module._publish_bundle(
            {
                report: report_bytes,
                authorization: authorization_bytes,
            },
            output_root=output,
            precommit=drift_only_in_report_window,
        )

    assert state["report_only_seen"] is True
    assert not report.exists()
    assert not authorization.exists()
    assert not list(output.glob(".pcraft-soft-loss-v3-stage-*"))

    dependency.write_bytes(b"stable dependency")
    script = r"""
import os
import sys
from pathlib import Path
import experiments.soft_loss_v3_registry as registry

output = Path(sys.argv[1])
report = Path(sys.argv[2])
authorization = Path(sys.argv[3])
original_link = registry.os.link
links = 0

def crash_after_report(source, destination, *args, **kwargs):
    global links
    original_link(source, destination, *args, **kwargs)
    links += 1
    if links == 1:
        os._exit(73)

registry.os.link = crash_after_report
registry._publish_bundle(
    {report: b'{"kind":"report"}\n', authorization: b'{"kind":"authorization"}\n'},
    output_root=output,
    precommit=lambda: None,
)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", script, str(output), str(report), str(authorization)],
        cwd=PROJECT_ROOT,
        env=_subprocess_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert crashed.returncode == 73, crashed.stderr
    assert report.read_bytes() == report_bytes
    assert not authorization.exists()
    assert len(list(output.glob(".pcraft-soft-loss-v3-stage-*"))) == 1

    registry_module._publish_bundle(
        {
            report: report_bytes,
            authorization: authorization_bytes,
        },
        output_root=output,
        precommit=lambda: None,
    )
    assert report.read_bytes() == report_bytes
    assert authorization.read_bytes() == authorization_bytes
    assert not list(output.glob(".pcraft-soft-loss-v3-stage-*"))


@pytest.mark.parametrize("publication_state", ["fresh", "all_exists"])
def test_stage_directory_rmdir_is_conditioned_on_the_recorded_owned_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_state: str,
) -> None:
    """Break caught: stage-name rmdir deletes a replacement after its inode audit."""

    output = tmp_path / "published"
    target = output / "accepted.json"
    expected = b'{"accepted":true}\n'
    if publication_state == "all_exists":
        crashed = _leave_complete_owned_stage(output, {target: expected})
        assert crashed.returncode == 73, crashed.stderr
        assert target.read_bytes() == expected
        assert len(list(output.glob(".pcraft-soft-loss-v3-stage-*"))) == 1

    original_rmdir = registry_module.os.rmdir
    state: dict[str, object] = {
        "swapped": False,
        "replacement": None,
        "held": None,
    }

    def replace_stage_in_rmdir_gap(
        name: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        descriptor = kwargs.get("dir_fd")
        if (
            not state["swapped"]
            and isinstance(name, str)
            and name.startswith(".pcraft-soft-loss-v3-stage-")
            and isinstance(descriptor, int)
        ):
            root = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            held_name = f"{name}.held"
            os.rename(
                name,
                held_name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
            os.mkdir(name, mode=0o700, dir_fd=descriptor)
            state.update(
                {
                    "swapped": True,
                    "replacement": root / name,
                    "held": root / held_name,
                }
            )
        original_rmdir(name, *args, **kwargs)

    monkeypatch.setattr(registry_module.os, "rmdir", replace_stage_in_rmdir_gap)
    error: BaseException | None = None
    try:
        registry_module._publish_bundle(
            {target: expected},
            output_root=output,
        )
    except (ValueError, OSError) as caught:
        error = caught

    if state["swapped"]:
        assert error is not None
        assert re.search(
            "stage|inode|owned|cleanup|replaced|publication",
            str(error),
            re.IGNORECASE,
        )
        replacement = state["replacement"]
        held = state["held"]
        assert isinstance(replacement, Path) and replacement.is_dir()
        assert isinstance(held, Path) and held.is_dir()
    else:
        assert error is None
        assert target.read_bytes() == expected
        assert not list(output.glob(".pcraft-soft-loss-v3-stage-*"))


@pytest.mark.parametrize("publication_state", ["fresh", "all_exists"])
def test_dependency_is_reauthenticated_after_stage_cleanup_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_state: str,
) -> None:
    """Break caught: cleanup can be followed by dependency drift and stale success."""

    output = tmp_path / "published"
    target = output / "accepted.json"
    expected = b'{"accepted":true}\n'
    dependency = tmp_path / "dependency.bin"
    dependency.write_bytes(b"stable")
    if publication_state == "all_exists":
        crashed = _leave_complete_owned_stage(output, {target: expected})
        assert crashed.returncode == 73, crashed.stderr
        assert target.read_bytes() == expected
        assert len(list(output.glob(".pcraft-soft-loss-v3-stage-*"))) == 1

    original_remove = registry_module._remove_owned_stage
    state = {"mutated": False, "precommits": 0}

    def mutate_dependency_after_cleanup(*args: object, **kwargs: object) -> None:
        original_remove(*args, **kwargs)
        if not state["mutated"]:
            dependency.write_bytes(b"drift")
            state["mutated"] = True

    def verify_dependency() -> None:
        state["precommits"] += 1
        if dependency.read_bytes() != b"stable":
            raise RuntimeError("post-cleanup dependency drift")

    monkeypatch.setattr(
        registry_module,
        "_remove_owned_stage",
        mutate_dependency_after_cleanup,
    )
    with pytest.raises(RuntimeError, match="post-cleanup dependency drift"):
        registry_module._publish_bundle(
            {target: expected},
            output_root=output,
            precommit=verify_dependency,
        )

    assert state["mutated"] is True
    if publication_state == "all_exists":
        assert target.read_bytes() == expected
    else:
        assert not target.exists()


@pytest.mark.parametrize(
    ("crash_point", "returncode"),
    [
        ("cleanup", 74),
        ("cleanup_rename", 76),
        ("write", 75),
        ("mid_manifest", 78),
        ("mid_payload", 79),
    ],
)
def test_identical_retry_recovers_from_partial_owned_stage_lifecycle_crash(
    tmp_path: Path,
    crash_point: str,
    returncode: int,
) -> None:
    """Break caught: an interrupted staged write/cleanup permanently wedges retry."""

    output = tmp_path / "published"
    files = {
        output / "analysis" / "report.json": b'{"kind":"report"}\n',
        output / "authorizations" / "authorization.json": (
            b'{"kind":"authorization"}\n'
        ),
    }
    crashed = _crash_during_stage_lifecycle(
        output,
        files,
        crash_point=crash_point,
    )
    assert crashed.returncode == returncode, crashed.stderr
    if crash_point.startswith("cleanup"):
        for target, expected in files.items():
            assert target.read_bytes() == expected
    else:
        assert all(not target.exists() for target in files)

    registry_module._publish_bundle(files, output_root=output)

    for target, expected in files.items():
        assert target.read_bytes() == expected
    transaction_debris = [
        entry
        for entry in output.iterdir()
        if entry.name.startswith(".pcraft-soft-loss-v3-")
        and entry.name != registry_module._RETIREMENT_DIRECTORY
    ]
    assert not transaction_debris
    retirement = output / registry_module._RETIREMENT_DIRECTORY
    if retirement.exists():
        assert retirement.is_dir()
        assert not list(retirement.iterdir())


@pytest.mark.parametrize("publication_state", ("fresh", "recovery", "rollback"))
def test_every_cooperative_cleanup_path_holds_the_stable_output_root_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_state: str,
) -> None:
    """Break caught: a cooperating publisher cleans outside the common root lock."""

    output = tmp_path / "published"
    files = {
        output / "analysis" / "report.json": b'{"kind":"report"}\n',
        output / "authorizations" / "authorization.json": (
            b'{"kind":"authorization"}\n'
        ),
    }
    if publication_state == "recovery":
        crashed = _leave_complete_owned_stage(output, files)
        assert crashed.returncode == 73, crashed.stderr

    original_unlink = registry_module._conditional_unlink_bound_entry
    original_retired = registry_module._remove_retired_directory
    observed = {"unlink": 0, "retired": 0}

    def assert_root_is_exclusively_locked() -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        probe = os.open(output, flags)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe)

    def checked_unlink(*args: object, **kwargs: object) -> None:
        assert_root_is_exclusively_locked()
        observed["unlink"] += 1
        original_unlink(*args, **kwargs)

    def checked_retired(*args: object, **kwargs: object) -> None:
        assert_root_is_exclusively_locked()
        observed["retired"] += 1
        original_retired(*args, **kwargs)

    monkeypatch.setattr(
        registry_module,
        "_conditional_unlink_bound_entry",
        checked_unlink,
    )
    monkeypatch.setattr(
        registry_module,
        "_remove_retired_directory",
        checked_retired,
    )

    if publication_state == "rollback":
        calls = 0

        def fail_between_links() -> None:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("cooperative rollback probe")

        with pytest.raises(RuntimeError, match="cooperative rollback probe"):
            registry_module._publish_bundle(
                files,
                output_root=output,
                precommit=fail_between_links,
            )
        assert all(not target.exists() for target in files)
    else:
        registry_module._publish_bundle(files, output_root=output)
        for target, expected in files.items():
            assert target.read_bytes() == expected

    assert observed["unlink"] > 0
    assert observed["retired"] > 0


@pytest.mark.parametrize("entry_kind", ("foreign_name", "corrupt_owned_shape"))
def test_preexisting_foreign_retirement_entry_is_preserved_and_fails_closed(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    """Break caught: recovery adopts/removes transaction debris present before locking."""

    output = tmp_path / "published"
    retirement = output / registry_module._RETIREMENT_DIRECTORY
    retirement.mkdir(parents=True, mode=0o700)
    foreign = b"preexisting foreign retirement bytes"
    if entry_kind == "foreign_name":
        entry = retirement / "foreign.bin"
        entry.write_bytes(foreign)
        preserved = entry
    else:
        entry = retirement / (
            ".pcraft-soft-loss-v3-retired-" + "0" * 32
        )
        entry.mkdir(mode=0o700)
        preserved = entry / "foreign.bin"
        preserved.write_bytes(foreign)

    target = output / "accepted.json"
    with pytest.raises(ValueError, match="retirement|recovery|foreign"):
        registry_module._publish_bundle(
            {target: b'{"accepted":true}\n'},
            output_root=output,
        )

    assert preserved.read_bytes() == foreign
    assert not target.exists()


def test_unrelated_prior_bundle_output_does_not_block_current_transaction(
    tmp_path: Path,
) -> None:
    """Break caught: a publisher treats unrelated immutable output as its debris."""

    output = tmp_path / "published"
    output.mkdir()
    prior = output / "unrelated_previous_bundle.bin"
    prior_bytes = b"independent immutable publication"
    prior.write_bytes(prior_bytes)
    target = output / "accepted.json"
    expected = b'{"accepted":true}\n'

    registry_module._publish_bundle({target: expected}, output_root=output)

    assert prior.read_bytes() == prior_bytes
    assert target.read_bytes() == expected


def test_retry_recovers_or_eliminates_target_parent_rollback_remove_residue(
    tmp_path: Path,
) -> None:
    """Break caught: rollback death strands a reserved temp that retry ignores."""

    output = tmp_path / "published"
    files = {
        output / "analysis" / "report.json": b'{"kind":"report"}\n',
        output / "authorizations" / "authorization.json": (
            b'{"kind":"authorization"}\n'
        ),
    }
    crashed = _crash_between_target_rollback_rename_and_unlink(output, files)
    # A protocol that removes the target-parent rename entirely exits through
    # the injected rollback error (1); the prior protocol reaches exit 93.
    assert crashed.returncode in {1, 93}, crashed.stderr
    residues_after_crash = list(
        output.rglob(".pcraft-soft-loss-v3-remove-*")
    )
    if crashed.returncode == 93:
        assert len(residues_after_crash) == 1
    else:
        assert not residues_after_crash

    registry_module._publish_bundle(files, output_root=output)

    for target, expected in files.items():
        assert target.read_bytes() == expected
    assert not list(output.rglob(".pcraft-soft-loss-v3-remove-*"))


def test_preexisting_reserved_remove_in_current_target_parent_fails_closed(
    tmp_path: Path,
) -> None:
    """Break caught: a current target parent silently ignores reserved debris."""

    output = tmp_path / "published"
    parent = output / "analysis"
    parent.mkdir(parents=True)
    reserved = parent / (".pcraft-soft-loss-v3-remove-" + "0" * 32)
    foreign = b"preexisting reserved target-parent bytes"
    reserved.write_bytes(foreign)
    target = parent / "accepted.json"

    with pytest.raises(ValueError, match="remove|temporary|foreign|transaction"):
        registry_module._publish_bundle(
            {target: b'{"accepted":true}\n'},
            output_root=output,
        )

    assert reserved.read_bytes() == foreign
    assert not target.exists()
