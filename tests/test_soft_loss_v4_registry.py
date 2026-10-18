"""Fail-closed contracts for v4 suite binding and immutable publication."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import experiments.soft_loss_v3_registry as v3_registry
import experiments.soft_loss_v4_registry as registry
from experiments.soft_loss_v4_registry import (
    V4BoundInput,
    V4SuiteIdentity,
    V4TaskInputs,
    assert_v4_suites_disjoint,
    bind_v4_suite,
    binding_main,
    build_binding_parser,
    load_v4_suite_identity,
    validate_candidate_registry,
)
from shadowforge_pipeline.soft_voxel_v4.config import bootstrap_v4_root
from shadowforge_pipeline.soft_voxel_v4.provenance import canonical_hash, canonical_json_bytes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_IDS = ("arcane_casters", "blade_heroes", "stage_dancers")
FORMAL_IDS = tuple(f"connected_{index:02d}" for index in range(1, 11))
AXES = ("x", "y", "z")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _make_v3_suites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    project = tmp_path / "project"
    development = project / "development"
    formal = project / "formal"
    development.mkdir(parents=True)
    formal.mkdir(parents=True)
    tasks = []
    for task_index, case_id in enumerate(DEVELOPMENT_IDS):
        views = []
        for axis_index in range(3):
            target = development / case_id / f"view_{axis_index}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"development\0" + bytes((task_index, axis_index)))
            views.append(str(target.resolve()))
        tasks.append({"task": case_id, "views": views})
    _write_json(
        development / "MANIFEST.json",
        {
            "protocol": "fixed-3-view TFS, mr=1, 256^3",
            "foreground": "white-on-black",
            "tasks": tasks,
        },
    )
    cases = []
    for case_index, case_id in enumerate(FORMAL_IDS):
        paths = []
        for axis_index, axis in enumerate(AXES):
            target = formal / "inputs" / f"{case_id}_view_{axis}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"formal\0" + bytes((case_index, axis_index)))
            paths.append(str(target.resolve()))
        cases.append({"case_id": case_id, "target_paths_xyz": paths})
    _write_json(
        formal / "benchmark_manifest.json",
        {
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
        },
    )
    monkeypatch.setattr(v3_registry, "PROJECT_ROOT", project)
    dev_payload = v3_registry.bind_development_suite(development)
    formal_payload = v3_registry.bind_formal_suite(formal, formal / "benchmark_manifest.json")
    source = tmp_path / "v3"
    dev_path = source / "development_suite.json"
    formal_path = source / "formal_suite.json"
    dev_path.parent.mkdir()
    dev_path.write_bytes(canonical_json_bytes(dev_payload))
    formal_path.write_bytes(canonical_json_bytes(formal_payload))
    return dev_path, formal_path


def _bind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    dev_source, formal_source = _make_v3_suites(tmp_path, monkeypatch)
    root = tmp_path / "v4"
    assert binding_main(
        [
            "--output-root",
            str(root),
            "--development-source",
            str(dev_source),
            "--formal-source",
            str(formal_source),
        ]
    ) == 0
    return root, root / "manifests/development_suite.json", root / "manifests/formal_suite.json"


def test_public_dataclasses_are_exact_and_frozen() -> None:
    """Break caught: suite identities become mutable or acquire covert fields."""

    item = V4BoundInput("x", "/a", "/a", "a" * 64)
    task = V4TaskInputs("case", (item, item, item), "development")
    suite = V4SuiteIdentity("/suite", "b" * 64, "c" * 64, "development", (task,))
    assert tuple(item.__dataclass_fields__) == ("axis", "source_path", "resolved_path", "sha256")
    assert tuple(task.__dataclass_fields__) == ("case_id", "inputs", "role")
    assert tuple(suite.__dataclass_fields__) == ("path", "sha256", "semantic_sha256", "role", "tasks")
    with pytest.raises(FrozenInstanceError):
        item.axis = "y"


def test_projection_preserves_v3_order_paths_hashes_and_exact_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: projection filters/reorders tasks or rewrites canonical inputs."""

    root, development, formal = _bind(tmp_path, monkeypatch)
    dev = load_v4_suite_identity(development, "development")
    frm = load_v4_suite_identity(formal, "formal")
    assert [task.case_id for task in dev.tasks] == list(DEVELOPMENT_IDS)
    assert [task.case_id for task in frm.tasks] == list(FORMAL_IDS)
    assert sum(len(task.inputs) for task in dev.tasks) == 9
    assert sum(len(task.inputs) for task in frm.tasks) == 30
    assert_v4_suites_disjoint(dev, frm)
    for identity, role, count in ((dev, "development", 3), (frm, "formal", 10)):
        payload = json.loads(Path(identity.path).read_bytes())
        assert set(payload) == {
            "schema", "status", "role", "source_suite", "tasks", "task_count",
            "input_count", "input_order", "protocol", "profiles", "semantic_sha256",
        }
        assert payload["schema"] == "pcraft_soft_loss_v4_suite_v1"
        assert payload["status"] == "REGISTERED"
        assert payload["role"] == role
        assert payload["task_count"] == count
        assert payload["input_count"] == 3 * count
        assert payload["input_order"] == ["x", "y", "z"]
        assert payload["semantic_sha256"] == canonical_hash(
            {key: value for key, value in payload.items() if key != "semantic_sha256"}
        )
        for task in payload["tasks"]:
            assert [item["axis"] for item in task["inputs"]] == list(AXES)
            for item in task["inputs"]:
                assert item["source_path"] == item["resolved_path"]
                assert item["sha256"] == _sha(Path(item["resolved_path"]))
    assert {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} == {
        "V4_ROOT.json", "manifests/development_suite.json", "manifests/formal_suite.json"
    }


@pytest.mark.parametrize("field", ["schema", "status", "role", "source_suite", "tasks", "protocol", "profiles"])
def test_suite_validator_rejects_every_semantic_field_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Break caught: stored semantics or current dependency bytes are trusted."""

    _, development, _ = _bind(tmp_path, monkeypatch)
    payload = json.loads(development.read_bytes())
    if field == "tasks":
        payload[field] = list(reversed(payload[field]))
    elif field in {"source_suite", "protocol"}:
        payload[field] = dict(payload[field])
        payload[field]["extra"] = True
    elif field == "profiles":
        payload[field] = list(reversed(payload[field]))
    else:
        payload[field] = "forged"
    payload["semantic_sha256"] = canonical_hash({k: v for k, v in payload.items() if k != "semantic_sha256"})
    development.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError):
        load_v4_suite_identity(development, "development")


def test_suite_validator_rejects_source_input_and_canonical_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: suite load skips transitive v3/input/current-byte validation."""

    _, development, _ = _bind(tmp_path, monkeypatch)
    payload = json.loads(development.read_bytes())
    development.write_bytes(development.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="canonical|JSON"):
        load_v4_suite_identity(development, "development")
    development.write_bytes(canonical_json_bytes(payload))
    Path(payload["tasks"][0]["inputs"][0]["resolved_path"]).write_bytes(b"drift")
    with pytest.raises(ValueError, match="hash|sha|changed"):
        load_v4_suite_identity(development, "development")


def test_suite_validator_rejects_boolean_integer_numeric_equality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: JSON integer 1 launders the frozen boolean mirror flag."""

    _, development, _ = _bind(tmp_path, monkeypatch)
    payload = json.loads(development.read_bytes())
    payload["protocol"]["mirror_enabled"] = 1
    payload["semantic_sha256"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "semantic_sha256"}
    )
    development.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="type|protocol"):
        load_v4_suite_identity(development, "development")


@pytest.mark.parametrize("axis", ["case", "path", "hash"])
def test_disjointness_rejects_case_path_and_hash_overlap(axis: str) -> None:
    """Break caught: one of the three required disjointness axes is omitted."""

    def item(letter: str, digest: str) -> V4BoundInput:
        return V4BoundInput("x", f"/{letter}", f"/{letter}", digest * 64)
    dev_input = item("a", "a")
    formal_input = item("b", "b")
    dev = V4SuiteIdentity("/d", "c" * 64, "d" * 64, "development", (
        V4TaskInputs("dev", (dev_input, V4BoundInput("y", "/c", "/c", "c" * 64), V4BoundInput("z", "/d", "/d", "d" * 64)), "development"),
    ))
    if axis == "case":
        case_id = "dev"
    else:
        case_id = "formal"
    if axis == "path":
        formal_input = V4BoundInput("x", "/a", "/a", "e" * 64)
    elif axis == "hash":
        formal_input = V4BoundInput("x", "/e", "/e", "a" * 64)
    frm = V4SuiteIdentity("/f", "e" * 64, "f" * 64, "formal", (
        V4TaskInputs(case_id, (formal_input, V4BoundInput("y", "/f", "/f", "f" * 64), V4BoundInput("z", "/0", "/0", "0" * 64)), "formal"),
    ))
    with pytest.raises(ValueError, match="overlap|duplicate"):
        assert_v4_suites_disjoint(dev, frm)


def test_binding_parser_is_exact_nonabbreviating_and_has_no_selection_knobs() -> None:
    """Break caught: binding accepts task/filter/metric/candidate overrides."""

    parser = build_binding_parser()
    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    assert set(actions) == {"output_root", "development_source", "formal_source"}
    assert all(action.required for action in actions.values())
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "/tmp/x", "--development-source", "/d", "--formal-source", "/f"])


def test_binding_is_idempotent_rejects_nonidentical_and_keeps_registry_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: create-once publication overwrites accepted suite bytes."""

    root, development, formal = _bind(tmp_path, monkeypatch)
    first = (_sha(development), _sha(formal))
    dev_source = Path(json.loads(development.read_bytes())["source_suite"]["path"])
    formal_source = Path(json.loads(formal.read_bytes())["source_suite"]["path"])
    assert binding_main(["--output-root", str(root), "--development-source", str(dev_source), "--formal-source", str(formal_source)]) == 0
    assert (_sha(development), _sha(formal)) == first
    development.write_bytes(b"foreign")
    with pytest.raises((FileExistsError, ValueError)):
        binding_main(["--output-root", str(root), "--development-source", str(dev_source), "--formal-source", str(formal_source)])


def test_bundle_publisher_recovers_after_every_link_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a crash leaves partial finals that identical retry cannot recover."""

    root = tmp_path / "publisher"
    root.mkdir()
    files = {
        root / "analysis/stage.json": b"stage",
        root / "analysis/unit.json": b"unit",
        root / "manifests/registry.json": b"registry",
    }
    original = registry.os.link
    calls = 0
    def crash(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        original(*args, **kwargs)
        if calls == 2:
            raise RuntimeError("synthetic crash")
    monkeypatch.setattr(registry.os, "link", crash)
    with pytest.raises(RuntimeError, match="synthetic"):
        registry._publish_bundle(files, output_root=root)
    monkeypatch.setattr(registry.os, "link", original)
    registry._publish_bundle(files, output_root=root)
    assert [path.read_bytes() for path in files] == list(files.values())
    assert not any("pcraft-soft-loss-v4" in path.name for path in root.rglob("*"))


def test_cleanup_recovery_keeps_original_name_when_two_targets_have_identical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: cleanup recovery guesses a payload name from size/SHA."""

    root = tmp_path / "duplicate-payloads"
    root.mkdir()
    files = {
        root / "analysis/first.json": b"identical",
        root / "analysis/second.json": b"identical",
        root / "manifests/registry.json": b"registry",
    }
    original = registry.os.unlink
    remove_unlinks = 0

    def crash_second_inner_cleanup(name: str, *args: object, **kwargs: object) -> None:
        nonlocal remove_unlinks
        if name.startswith(".pcraft-soft-loss-v4-remove-"):
            remove_unlinks += 1
            if remove_unlinks == 2:
                raise RuntimeError("synthetic duplicate-payload cleanup crash")
        original(name, *args, **kwargs)

    monkeypatch.setattr(registry.os, "unlink", crash_second_inner_cleanup)
    with pytest.raises(RuntimeError, match="duplicate-payload cleanup crash"):
        registry._publish_bundle(files, output_root=root)
    monkeypatch.setattr(registry.os, "unlink", original)
    registry._publish_bundle(files, output_root=root)
    assert [path.read_bytes() for path in files] == list(files.values())
    assert not any("pcraft-soft-loss-v4" in path.name for path in root.rglob("*"))


def test_bundle_publisher_serializes_identical_and_nonidentical_callers(tmp_path: Path) -> None:
    """Break caught: concurrent publishers expose a mixed bundle."""

    root = tmp_path / "concurrent"
    root.mkdir()
    first = {root / "a.json": b"A", root / "b.json": b"B"}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: registry._publish_bundle(first, output_root=root), range(2)))
    assert results == [None, None]
    second = {root / "a.json": b"X", root / "b.json": b"Y"}
    with pytest.raises((FileExistsError, ValueError)):
        registry._publish_bundle(second, output_root=root)
    assert (root / "a.json").read_bytes() == b"A"
    assert (root / "b.json").read_bytes() == b"B"


def test_bundle_publisher_revalidates_dependencies_between_links_and_after_cleanup(
    tmp_path: Path
) -> None:
    """Break caught: publication commits bytes after a bound dependency drifts."""

    root = tmp_path / "drift"
    root.mkdir()
    dependency = tmp_path / "dependency"
    dependency.write_bytes(b"stable")
    expected = _sha(dependency)
    calls = 0
    def precommit() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            dependency.write_bytes(b"changed")
        if _sha(dependency) != expected:
            raise ValueError("dependency drift")
    files = {root / "one.json": b"1", root / "two.json": b"2", root / "commit.json": b"3"}
    with pytest.raises(ValueError, match="dependency drift"):
        registry._publish_bundle(files, output_root=root, precommit=precommit)
    assert not (root / "commit.json").exists()


@pytest.mark.parametrize("kind", ["symlink", "directory", "hardlink"])
def test_bundle_publisher_rejects_foreign_target_types_without_deletion(
    tmp_path: Path, kind: str
) -> None:
    """Break caught: create-once publication follows or removes a foreign inode."""

    root = tmp_path / "root"
    root.mkdir()
    target = root / "manifest.json"
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    if kind == "symlink":
        target.symlink_to(outside)
    elif kind == "directory":
        target.mkdir()
    else:
        os.link(outside, target)
    before = outside.read_bytes()
    with pytest.raises((FileExistsError, ValueError)):
        registry._publish_bundle({target: b"wanted"}, output_root=root)
    assert outside.read_bytes() == before
    assert os.path.lexists(target)


def test_registry_commit_marker_is_linked_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Break caught: readers can observe a registry before stage/unit dependencies."""

    root = tmp_path / "root"
    root.mkdir()
    stage = root / "analysis/calibration_stage.json"
    unit = root / "analysis/unit_coefficients.json"
    marker = root / "manifests/candidate_registry.json"
    order: list[str] = []
    original = registry.os.link
    def record(source: str, destination: str, *args: object, **kwargs: object) -> None:
        order.append(destination)
        original(source, destination, *args, **kwargs)
    monkeypatch.setattr(registry.os, "link", record)
    registry._publish_bundle({stage: b"stage", unit: b"unit", marker: b"registry"}, output_root=root)
    assert order[-1] == "candidate_registry.json"


def test_output_root_lock_does_not_block_unrelated_roots(tmp_path: Path) -> None:
    """Break caught: a global lock serializes unrelated campaign roots."""

    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        root.mkdir()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(registry._publish_bundle, {root / "a.json": str(index).encode()}, output_root=root) for index, root in enumerate(roots)]
        for future in futures:
            future.result(timeout=5)
    assert [root.joinpath("a.json").read_bytes() for root in roots] == [b"0", b"1"]


def test_candidate_registry_validator_is_transitive_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: registry accepts forged rows or trusts unit bytes without validation."""

    root = tmp_path / "root"
    bootstrap_v4_root(root)
    unit = root / "analysis/unit_coefficients.json"
    unit.parent.mkdir()
    unit_payload = {"schema": "synthetic", "semantic_sha256": "a" * 64}
    unit.write_bytes(canonical_json_bytes(unit_payload))
    registry_path = root / "manifests/candidate_registry.json"
    registry_path.parent.mkdir()
    rows = [
        {"candidate_id": "r04_r04", "order_index": 0, "dic_max_ratio": 0.04, "dts_max_ratio": 0.04},
        {"candidate_id": "r08_r08", "order_index": 1, "dic_max_ratio": 0.08, "dts_max_ratio": 0.08},
        {"candidate_id": "r12_r12", "order_index": 2, "dic_max_ratio": 0.12, "dts_max_ratio": 0.12},
        {"candidate_id": "r12_r08", "order_index": 3, "dic_max_ratio": 0.12, "dts_max_ratio": 0.08},
        {"candidate_id": "r08_r12", "order_index": 4, "dic_max_ratio": 0.08, "dts_max_ratio": 0.12},
    ]
    payload = {
        "schema": "pcraft_soft_loss_v4_candidate_registry_v1",
        "status": "REGISTERED",
        "unit_coefficients": {"path": str(unit), "sha256": _sha(unit), "semantic_sha256": "a" * 64},
        "candidates": rows,
    }
    payload["semantic_sha256"] = canonical_hash(payload)
    registry_path.write_bytes(canonical_json_bytes(payload))
    called = 0
    def validate(path: Path) -> dict[str, object]:
        nonlocal called
        called += 1
        assert path == unit
        return unit_payload
    import shadowforge_pipeline.soft_voxel_v4.calibration as calibration
    monkeypatch.setattr(calibration, "validate_unit_coefficients", validate)
    checked = validate_candidate_registry(registry_path)
    assert checked["candidates"] == rows
    assert called == 1
    payload["candidates"][0]["dic"] = 1.0
    payload["semantic_sha256"] = canonical_hash({k: v for k, v in payload.items() if k != "semantic_sha256"})
    registry_path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError):
        validate_candidate_registry(registry_path)


def test_imports_launchers_and_ast_scope_are_side_effect_free(tmp_path: Path) -> None:
    """Break caught: imports write files or Task4 reaches training/data/downstream scope."""

    before = set(tmp_path.rglob("*"))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(PROJECT_ROOT))
    subprocess.run(
        [sys.executable, "-c", "import experiments.soft_loss_v4_registry; import experiments.bind_soft_loss_v4_suites"],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    assert set(tmp_path.rglob("*")) == before
    for relative in ("experiments/soft_loss_v4_registry.py", "experiments/bind_soft_loss_v4_suites.py"):
        tree = ast.parse((PROJECT_ROOT / relative).read_text(encoding="utf-8"))
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
        assert not ({"run", "main", "optimizer", "glob", "rglob"} & calls)
        assert "/data" not in (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def test_binding_launcher_help_uses_locked_public_parser() -> None:
    """Break caught: the thin launcher acquires a second permissive CLI."""

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "experiments/bind_soft_loss_v4_suites.py"), "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "--development-source" in result.stdout
    assert "--formal-source" in result.stdout
