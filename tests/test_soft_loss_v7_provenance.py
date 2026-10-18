"""Fail-closed provenance and v7-root contracts for Task 2."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import shadowforge_pipeline.soft_voxel_v7.provenance as provenance
from shadowforge_pipeline.soft_voxel_v7.provenance import (
    AuthenticatedV7Root,
    bootstrap_v7_root,
    canonical_json_bytes,
    canonical_sha256,
    decode_canonical_json,
    sha256_file,
    validate_v7_root,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK1_HASHES = {
    "__init__.py": "79c04de43816d694eec901bbc6a9b6a0dae88ca6220b46373bb5196af4dddcb3",
    "core.py": "25bfd4844e3c0a378d1289e8eb626a333f039c839847669d3ab4e4e632214fcf",
    "protocol.py": "8e627eb4d897893fafcc66c607b444891f2e6888696c2f410760460cf2da5ce3",
}


def test_canonical_json_is_exact_finite_and_duplicate_safe() -> None:
    """Break caught: aliases, key coercion, NaN, or duplicate JSON enter identity."""

    assert canonical_json_bytes({"b": [2, 3.0], "a": True}) == b'{"a":true,"b":[2,3.0]}'
    assert canonical_sha256({"x": 1}) != canonical_sha256({"x": 1.0})
    assert decode_canonical_json(b'{"a":1,"b":[false,null]}', label="fixture") == {
        "a": 1,
        "b": [False, None],
    }
    with pytest.raises(ValueError, match="duplicate"):
        decode_canonical_json(b'{"a":1,"a":2}', label="duplicate")
    with pytest.raises(ValueError, match="finite"):
        decode_canonical_json(b'{"a":NaN}', label="nan")

    class IntAlias(int):
        pass

    class StrAlias(str):
        pass

    invalid = (
        {1: "coerced"},
        {StrAlias("x"): 1},
        {"x": IntAlias(1)},
        {"x": np.int64(1)},
        {"x": np.bool_(True)},
        {"x": float("inf")},
        {"x": object()},
    )
    for value in invalid:
        with pytest.raises((TypeError, ValueError)):
            canonical_json_bytes(value)


def test_sha256_file_requires_canonical_regular_nonsymlink_path(tmp_path: Path) -> None:
    """Break caught: a public digest follows a link or accepts relative/coercible paths."""

    target = tmp_path / "target.bin"
    target.write_bytes(b"bound")
    assert sha256_file(target) == hashlib.sha256(b"bound").hexdigest()
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        sha256_file(link)
    with pytest.raises(ValueError, match="absolute|canonical"):
        sha256_file(Path("relative.bin"))
    for value in (True, 1, str(target)):
        with pytest.raises((TypeError, ValueError)):
            sha256_file(value)  # type: ignore[arg-type]


def test_bootstrap_binds_exact_root_dependencies_and_is_concurrent(tmp_path: Path) -> None:
    """Break caught: concurrent bootstraps disagree or omit a frozen dependency."""

    root = tmp_path / "v7"
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(lambda _: bootstrap_v7_root(root), range(4)))
    assert all(isinstance(item, AuthenticatedV7Root) for item in identities)
    assert all(item == identities[0] for item in identities)
    identity = validate_v7_root(root)
    payload = json.loads((root / "V7_ROOT.json").read_bytes())
    assert set(payload) == {
        "schema", "root_path", "project_root", "experiment_name", "seed",
        "selector_seed", "interpreter", "specification", "task1_sources",
        "semantic_sha256",
    }
    assert payload["schema"] == "pcraft_soft_loss_v7_root_v1"
    assert payload["root_path"] == str(root)
    assert payload["project_root"] == str(PROJECT_ROOT)
    assert payload["experiment_name"] == "pcraft_soft_loss_v7_hit_barrier_256_20260829"
    assert payload["seed"] == 17 and type(payload["seed"]) is int
    assert payload["selector_seed"] == 20260829
    assert payload["task1_sources"] == TASK1_HASHES
    assert payload["specification"]["sha256"] == "d6f3219080f61acc9a3d6c2b435dfd0fd12eac06b68c2f4a301f0b5fedfa74c1"
    assert payload["interpreter"]["path"] == "/home/lin/anaconda3/envs/shadowart-revisited/bin/python3.10"
    assert payload["interpreter"]["sha256"] == "7365743bea8011d4ada1d0379699045fe411de80f620c39e4c4625855f0b1772"
    unsigned = {key: value for key, value in payload.items() if key != "semantic_sha256"}
    assert payload["semantic_sha256"] == canonical_sha256(unsigned)
    assert identity.marker_sha256 == hashlib.sha256((root / "V7_ROOT.json").read_bytes()).hexdigest()
    assert bootstrap_v7_root(root) == identity


@pytest.mark.parametrize("kind", ("foreign", "partial", "symlink"))
def test_bootstrap_preserves_and_rejects_foreign_partial_or_linked_root(
    tmp_path: Path, kind: str
) -> None:
    """Break caught: bootstrap adopts, deletes, or replaces an unauthenticated root."""

    root = tmp_path / "v7"
    outside = tmp_path / "outside"
    if kind == "symlink":
        outside.mkdir()
        root.symlink_to(outside, target_is_directory=True)
    else:
        root.mkdir()
        (root / ("foreign.bin" if kind == "foreign" else "V7_ROOT.json")).write_bytes(
            b"foreign" if kind == "foreign" else b'{"schema":"partial"}'
        )
    before = {path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises((FileExistsError, ValueError)):
        bootstrap_v7_root(root)
    after = {path.relative_to(tmp_path).as_posix(): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_root_validator_rejects_extra_schema_symlink_hardlink_and_zero_writes(
    tmp_path: Path,
) -> None:
    """Break caught: root authentication ignores namespace drift or changes bytes."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    marker = root / "V7_ROOT.json"
    before = marker.read_bytes()
    before_stat = marker.stat()
    assert validate_v7_root(root).payload["schema"] == "pcraft_soft_loss_v7_root_v1"
    after_stat = marker.stat()
    assert marker.read_bytes() == before
    assert (after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns) == (
        before_stat.st_ino, before_stat.st_size, before_stat.st_mtime_ns
    )

    extra = root / "unexpected"
    extra.write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected|namespace"):
        validate_v7_root(root)
    extra.unlink()

    dangling = root / "manifests"
    dangling.symlink_to(tmp_path / "missing-manifests", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|directory"):
        validate_v7_root(root)
    dangling.unlink()

    outside = tmp_path / "marker-hardlink"
    os.link(marker, outside)
    try:
        with pytest.raises(ValueError, match="hardlink|link count"):
            validate_v7_root(root)
    finally:
        outside.unlink()

    payload = json.loads(before)
    payload["extra"] = True
    marker.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="schema|semantic|differs"):
        validate_v7_root(root)


def test_root_revalidates_current_dependency_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: marker self-hash substitutes for current spec/source bytes."""

    project = tmp_path / "project"
    spec = project / ".superpowers/sdd/2026-08-29-soft-loss-v7-hit-barrier/spec.md"
    spec.parent.mkdir(parents=True)
    spec.write_bytes(provenance.SPECIFICATION_PATH.read_bytes())
    package = project / "shadowforge_pipeline/soft_voxel_v7"
    package.mkdir(parents=True)
    for name in TASK1_HASHES:
        shutil_source = PROJECT_ROOT / "shadowforge_pipeline/soft_voxel_v7" / name
        (package / name).write_bytes(shutil_source.read_bytes())
    monkeypatch.setattr(provenance, "PROJECT_ROOT", project)
    monkeypatch.setattr(provenance, "SPECIFICATION_PATH", spec)
    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    spec.write_bytes(spec.read_bytes() + b"\ndrift")
    with pytest.raises(ValueError, match="specification|dependency|identity|changed"):
        validate_v7_root(root)


def test_bootstrap_failure_before_marker_rolls_back_owned_empty_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a controlled pre-marker failure strands an adoptable partial root."""

    root = tmp_path / "v7"
    monkeypatch.setattr(provenance, "_ROOT_PUBLICATION_HOOK", lambda stage: (_ for _ in ()).throw(RuntimeError(stage)))
    with pytest.raises(RuntimeError, match="after_root_create"):
        bootstrap_v7_root(root)
    assert not root.exists()


def test_root_rejects_float_seed_even_when_old_semantic_is_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: Python numeric equality lets 17.0 impersonate exact integer seed 17."""

    monkeypatch.setattr(
        provenance, "EXPECTED_SPECIFICATION_SHA256",
        "d6f3219080f61acc9a3d6c2b435dfd0fd12eac06b68c2f4a301f0b5fedfa74c1",
    )
    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    marker = root / "V7_ROOT.json"
    payload = json.loads(marker.read_bytes())
    retained_semantic = payload["semantic_sha256"]
    payload["seed"] = 17.0
    payload["semantic_sha256"] = retained_semantic
    marker.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="seed|type|semantic|marker|schema"):
        validate_v7_root(root)


def test_public_fd_safe_seams_revalidate_exclusive_link_count(tmp_path: Path) -> None:
    """Break caught: Task 3 must not depend on private helpers or miss new hardlinks."""

    assert provenance.AuthenticatedFileSnapshot.__name__ == "AuthenticatedFileSnapshot"
    target = tmp_path / "owned.bin"
    target.write_bytes(b"immutable")
    snapshot = provenance.snapshot_regular_file(target)
    assert isinstance(snapshot, provenance.AuthenticatedFileSnapshot)
    assert snapshot.nlink == 1
    provenance.revalidate_snapshot(snapshot)
    hardlink = tmp_path / "owned-hardlink.bin"
    os.link(target, hardlink)
    try:
        with pytest.raises(ValueError, match="hardlink|link count|identity|changed"):
            provenance.revalidate_snapshot(snapshot)
    finally:
        hardlink.unlink()
    for name in (
        "AuthenticatedFileSnapshot", "snapshot_regular_file", "revalidate_snapshot",
        "publish_bytes_once", "fsync_directory",
    ):
        assert name in provenance.__all__


def test_snapshot_rejects_ancestor_swap_without_reading_outside_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: lstat-then-absolute-open follows a swapped ancestor symlink."""

    containment = tmp_path / "containment"
    ancestor = containment / "stable"
    ancestor.mkdir(parents=True)
    victim = ancestor / "payload.bin"
    victim.write_bytes(b"inside")
    moved = containment / "moved"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_payload = outside / "payload.bin"
    outside_payload.write_bytes(b"outside-secret")
    original_bind = provenance._bind_directory
    swapped = False

    def swap_after_root_bind(path: Path, *, label: str) -> object:
        nonlocal swapped
        bound = original_bind(path, label=label)
        if path == containment and not swapped:
            ancestor.rename(moved)
            ancestor.symlink_to(outside, target_is_directory=True)
            swapped = True
        return bound

    monkeypatch.setattr(provenance, "_bind_directory", swap_after_root_bind)
    try:
        with pytest.raises(ValueError, match="symlink|identity|changed|containment"):
            provenance._snapshot_regular_file(victim, root=containment)
        assert swapped
        assert outside_payload.read_bytes() == b"outside-secret"
    finally:
        if ancestor.is_symlink():
            ancestor.unlink()
        if moved.exists():
            moved.rename(ancestor)


def test_publication_rejects_ancestor_swap_without_writing_outside_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: absolute temp/create/rename writes through a swapped parent link."""

    containment = tmp_path / "containment"
    parent = containment / "stable"
    parent.mkdir(parents=True)
    moved = containment / "moved"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = parent / "published.bin"
    original_bind = provenance._bind_directory
    swapped = False

    def swap_after_root_bind(path: Path, *, label: str) -> object:
        nonlocal swapped
        result = original_bind(path, label=label)
        if path == parent and not swapped:
            parent.rename(moved)
            parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(provenance, "_bind_directory", swap_after_root_bind)
    try:
        with pytest.raises((ValueError, OSError), match="symlink|identity|changed|directory"):
            provenance._publish_bytes_once(destination, b"inside-only")
        assert swapped
        assert not (outside / destination.name).exists()
    finally:
        outside_target = outside / destination.name
        if outside_target.exists():
            outside_target.unlink()
        if parent.is_symlink():
            parent.unlink()
        if moved.exists():
            moved.rename(parent)


def test_root_accepts_only_registered_lifecycle_top_level_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: suite-era root validation blocks legitimate downstream handoff."""

    monkeypatch.setattr(
        provenance, "EXPECTED_SPECIFICATION_SHA256",
        "d6f3219080f61acc9a3d6c2b435dfd0fd12eac06b68c2f4a301f0b5fedfa74c1",
    )
    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    for name in ("manifests", "authorizations", "runs", "analysis", "postprocess"):
        directory = root / name
        directory.mkdir()
        (directory / "opaque-owned-by-downstream").write_bytes(b"future")
    assert validate_v7_root(root).path == str(root)
    unexpected = root / "other"
    unexpected.mkdir()
    with pytest.raises(ValueError, match="unexpected|namespace"):
        validate_v7_root(root)


def _swap_directory_after_first_verified_binding(
    monkeypatch: pytest.MonkeyPatch, target: Path,
    replacement_files: dict[str, bytes] | None = None,
) -> tuple[Path, list[bool]]:
    """Replace one verified path with an ordinary directory at the attack boundary."""

    displaced = target.with_name(f"{target.name}-original-inode")
    original_verify = provenance._verify_directory_binding
    state = [False]

    def replace_after_verify(descriptor: int, path: Path, *, label: str) -> None:
        original_verify(descriptor, path, label=label)
        if path == target and not state[0]:
            target.rename(displaced)
            target.mkdir()
            for name, payload in (replacement_files or {}).items():
                (target / name).write_bytes(payload)
            state[0] = True

    monkeypatch.setattr(provenance, "_verify_directory_binding", replace_after_verify)
    return displaced, state


def test_snapshot_retains_containment_root_inode_for_entire_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: reopening by absolute path reads a same-name replacement root."""

    root = tmp_path / "root"
    root.mkdir()
    target = root / "payload.bin"
    target.write_bytes(b"original-inode")
    displaced, state = _swap_directory_after_first_verified_binding(
        monkeypatch, root, {"payload.bin": b"replacement-inode"}
    )
    with pytest.raises(ValueError, match="root|directory|identity|changed|binding"):
        provenance.snapshot_regular_file(target, root=root, exclusive=False)
    assert state == [True]
    assert (root / "payload.bin").read_bytes() == b"replacement-inode"
    assert (displaced / "payload.bin").read_bytes() == b"original-inode"


def test_publication_retains_root_inode_and_never_writes_replacement_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: publication reopens a replaced ordinary root by absolute path."""

    root = tmp_path / "root"
    root.mkdir()
    destination = root / "published.bin"
    displaced, state = _swap_directory_after_first_verified_binding(
        monkeypatch, root
    )
    with pytest.raises(ValueError, match="root|directory|identity|changed|binding"):
        provenance.publish_bytes_once(destination, b"bound", root=root)
    assert state == [True]
    assert not (root / "published.bin").exists()
    assert root.is_dir() and displaced.is_dir()


def test_fsync_retains_target_directory_inode_for_entire_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: fsync verifies one directory inode but syncs its replacement."""

    root = tmp_path / "root"
    root.mkdir()
    _, state = _swap_directory_after_first_verified_binding(monkeypatch, root)
    with pytest.raises(ValueError, match="directory|identity|changed|binding"):
        provenance.fsync_directory(root)
    assert state == [True]


def test_bootstrap_retains_parent_inode_and_never_creates_in_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: bootstrap locks one parent then creates through a replacement."""

    parent = tmp_path / "parent"
    parent.mkdir()
    root = parent / "v7"
    displaced, state = _swap_directory_after_first_verified_binding(
        monkeypatch, parent
    )
    with pytest.raises(ValueError, match="parent|directory|identity|changed|binding"):
        bootstrap_v7_root(root)
    assert state == [True]
    assert not root.exists()
    assert displaced.is_dir()


def test_root_marker_is_revalidated_immediately_before_validator_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: marker mutates after the last dependency check and is accepted."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    marker = root / "V7_ROOT.json"
    original_revalidate = provenance.revalidate_snapshot
    mutated = False

    def mutate_after_last_dependency(snapshot: object, *, root: Path | None = None) -> None:
        nonlocal mutated
        original_revalidate(snapshot, root=root)  # type: ignore[arg-type]
        if not mutated and str(getattr(snapshot, "path", "")).endswith("protocol.py"):
            marker.write_bytes(marker.read_bytes() + b"\n")
            mutated = True

    monkeypatch.setattr(provenance, "revalidate_snapshot", mutate_after_last_dependency)
    monkeypatch.setattr(provenance, "_revalidate_snapshot", mutate_after_last_dependency)
    with pytest.raises(ValueError, match="marker|identity|bytes|changed"):
        validate_v7_root(root)
    assert mutated


def test_root_transaction_public_workflow_is_fd_relative_and_create_once(
    tmp_path: Path,
) -> None:
    """Break caught: downstream lifecycle needs private helpers or overwrites bytes."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    transaction = provenance.open_v7_root_transaction(root)
    assert isinstance(transaction, provenance.V7RootTransaction)
    with transaction as active:
        assert active is transaction
        for name in ("manifests", "authorizations", "runs", "analysis", "postprocess"):
            active.ensure_registered_directory(name)
        relative = Path("authorizations/campaign-lock.json")
        active.publish(relative, b'{"status":"BOUND"}')
        snapshot = active.snapshot(relative)
        assert snapshot.payload == b'{"status":"BOUND"}'
        assert snapshot.path == str(root / relative)
        assert snapshot.nlink == 1
        active.fsync(Path("authorizations"))
        active.publish(Path("manifests/campaign.json"), b'{"status":"REGISTERED"}')
        assert active.snapshot(Path("manifests/campaign.json")).payload == b'{"status":"REGISTERED"}'
        active.fsync(Path("manifests"))
        active.revalidate_root_marker()
    with provenance.open_v7_root_transaction(root) as retry:
        retry.publish(Path("authorizations/campaign-lock.json"), b'{"status":"BOUND"}')
        with pytest.raises(FileExistsError, match="differs|immutable|publication"):
            retry.publish(Path("authorizations/campaign-lock.json"), b"conflict")


def test_root_transaction_rejects_nonexact_unregistered_or_escaping_paths(
    tmp_path: Path,
) -> None:
    """Break caught: lifecycle transaction accepts absolute, aliased, or foreign paths."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)

    class PathAlias(type(Path())):
        pass

    with provenance.open_v7_root_transaction(root) as transaction:
        for name in (True, 1, Path("runs"), "foreign"):
            with pytest.raises((TypeError, ValueError), match="name|registered|exact"):
                transaction.ensure_registered_directory(name)  # type: ignore[arg-type]
        transaction.ensure_registered_directory("runs")
        invalid_paths = (
            Path("."), Path("../escape"), Path("runs/../escape"),
            root / "runs/absolute.bin", Path("foreign/payload.bin"),
            "runs/string.bin", True, PathAlias("runs/alias.bin"),
        )
        for value in invalid_paths:
            with pytest.raises((TypeError, ValueError), match="relative|registered|path|exact|component"):
                transaction.snapshot(value)  # type: ignore[arg-type]
            with pytest.raises((TypeError, ValueError), match="relative|registered|path|exact|component"):
                transaction.publish(value, b"x")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="boolean|exclusive"):
            transaction.snapshot(Path("runs/missing.bin"), exclusive=1)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="bytes|payload"):
            transaction.publish(Path("runs/bad.bin"), "not-bytes")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="manifest|registered"):
            transaction.publish(Path("manifests/arbitrary.json"), b"x")


def test_root_transaction_rejects_nested_ownership_post_close_and_fd_leaks(
    tmp_path: Path,
) -> None:
    """Break caught: transaction can be nested/reused or leaks retained descriptors."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    before = len(os.listdir("/proc/self/fd"))
    transaction = provenance.open_v7_root_transaction(root)
    with transaction:
        with pytest.raises((BlockingIOError, RuntimeError, ValueError), match="lock|active|ownership|nested"):
            with provenance.open_v7_root_transaction(root):
                pass
        with pytest.raises((RuntimeError, ValueError), match="active|nested|reuse"):
            with transaction:
                pass
    for operation in (
        lambda: transaction.ensure_registered_directory("runs"),
        lambda: transaction.snapshot(Path("V7_ROOT.json")),
        lambda: transaction.publish(Path("runs/x"), b"x"),
        lambda: transaction.fsync(Path("runs")),
        transaction.revalidate_root_marker,
    ):
        with pytest.raises((RuntimeError, ValueError), match="closed|active|transaction"):
            operation()
    for _ in range(12):
        with provenance.open_v7_root_transaction(root) as active:
            active.revalidate_root_marker()
    assert len(os.listdir("/proc/self/fd")) == before


def test_root_transaction_rejects_same_path_root_replacement_before_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: entry validates an inode then yields a same-name replacement root."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    displaced, state = _swap_directory_after_first_verified_binding(
        monkeypatch, root, {"V7_ROOT.json": (root / "V7_ROOT.json").read_bytes()}
    )
    yielded = False
    with pytest.raises(ValueError, match="root|directory|identity|changed|binding"):
        with provenance.open_v7_root_transaction(root):
            yielded = True
    assert not yielded and state == [True]
    assert displaced.is_dir()


def test_root_transaction_rejects_root_replacement_after_entry_and_mkdir(
    tmp_path: Path,
) -> None:
    """Break caught: mkdir is redirected into a replacement root after lock acquisition."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    displaced = tmp_path / "v7-original-inode"
    with pytest.raises(ValueError, match="root|directory|identity|changed|binding"):
        with provenance.open_v7_root_transaction(root) as transaction:
            root.rename(displaced)
            root.mkdir()
            (root / "V7_ROOT.json").write_bytes(
                (displaced / "V7_ROOT.json").read_bytes()
            )
            transaction.ensure_registered_directory("runs")
    assert not (root / "runs").exists()
    assert not (displaced / "runs").exists()


def test_root_transaction_rejects_registered_directory_same_name_inode_replacement(
    tmp_path: Path,
) -> None:
    """Break caught: publication reopens a same-name replacement child directory."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    with provenance.open_v7_root_transaction(root) as transaction:
        transaction.ensure_registered_directory("runs")
        displaced = tmp_path / "runs-original-inode"
        (root / "runs").rename(displaced)
        (root / "runs").mkdir()
        with pytest.raises(ValueError, match="directory|identity|changed|binding"):
            transaction.publish(Path("runs/replacement.bin"), b"forbidden")
        assert not (root / "runs/replacement.bin").exists()
        (root / "runs").rmdir()
        displaced.rename(root / "runs")


def test_root_transaction_rejects_byte_identical_marker_inode_replacement(
    tmp_path: Path,
) -> None:
    """Break caught: terminal validation accepts a byte-identical new marker inode."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    marker = root / "V7_ROOT.json"
    original_payload = marker.read_bytes()
    original_inode = marker.stat().st_ino
    displaced = tmp_path / "V7_ROOT-original-inode.json"
    with pytest.raises(ValueError, match="marker|identity|inode|changed|snapshot"):
        with provenance.open_v7_root_transaction(root):
            marker.rename(displaced)
            marker.write_bytes(original_payload)
            assert marker.stat().st_ino != original_inode


def test_root_transaction_rejects_swapped_child_ancestors_for_publish_and_snapshot(
    tmp_path: Path,
) -> None:
    """Break caught: child operations follow a registered-directory symlink outside root."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    with provenance.open_v7_root_transaction(root) as transaction:
        transaction.ensure_registered_directory("runs")
        transaction.publish(Path("runs/inside.bin"), b"inside")
        moved = tmp_path / "runs-original"
        (root / "runs").rename(moved)
        (root / "runs").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink|directory|component|identity"):
            transaction.publish(Path("runs/outside.bin"), b"forbidden")
        with pytest.raises(ValueError, match="symlink|directory|component|identity"):
            transaction.snapshot(Path("runs/inside.bin"))
        assert not (outside / "outside.bin").exists()
        (root / "runs").unlink()
        moved.rename(root / "runs")


@pytest.mark.parametrize("second_payload", (b"identical", b"conflicting"))
def test_root_transaction_excludes_concurrent_writer_ownership(
    tmp_path: Path, second_payload: bytes,
) -> None:
    """Break caught: two retained-root writers concurrently own one publication namespace."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    with provenance.open_v7_root_transaction(root) as first:
        first.ensure_registered_directory("analysis")

        def second_writer() -> None:
            with provenance.open_v7_root_transaction(root) as second:
                second.publish(Path("analysis/claim.bin"), second_payload)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(second_writer)
            with pytest.raises((BlockingIOError, RuntimeError, ValueError), match="lock|ownership|active"):
                future.result(timeout=10)
        first.publish(Path("analysis/claim.bin"), b"identical")
    assert (root / "analysis/claim.bin").read_bytes() == b"identical"


def test_root_transaction_crash_prefix_recovers_registered_directory(
    tmp_path: Path,
) -> None:
    """Break caught: exception between mkdir and publication leaves unusable lifecycle state."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        with provenance.open_v7_root_transaction(root) as transaction:
            transaction.ensure_registered_directory("postprocess")
            raise RuntimeError("synthetic crash before publication")
    with provenance.open_v7_root_transaction(root) as recovered:
        recovered.publish(Path("postprocess/commit.json"), b'{"status":"COMMITTED"}')
    assert (root / "postprocess/commit.json").read_bytes() == b'{"status":"COMMITTED"}'


@pytest.mark.parametrize("mutation", ("marker", "namespace"))
def test_root_transaction_terminally_rejects_marker_or_namespace_mutation(
    tmp_path: Path, mutation: str,
) -> None:
    """Break caught: exit releases ownership without terminal root authentication."""

    root = tmp_path / "v7"
    bootstrap_v7_root(root)
    with pytest.raises(ValueError, match="marker|namespace|unexpected|identity|bytes"):
        with provenance.open_v7_root_transaction(root) as transaction:
            if mutation == "marker":
                marker = root / "V7_ROOT.json"
                marker.write_bytes(marker.read_bytes() + b"\n")
            else:
                (root / "unregistered").mkdir()
            if mutation == "marker":
                with pytest.raises(ValueError, match="marker|identity|bytes|canonical"):
                    transaction.revalidate_root_marker()
