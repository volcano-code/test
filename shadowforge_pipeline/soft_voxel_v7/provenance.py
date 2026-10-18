"""Strict canonical provenance and create-once v7 campaign bootstrap."""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPECIFICATION_PATH = PROJECT_ROOT / ".superpowers/sdd/2026-08-29-soft-loss-v7-hit-barrier/spec.md"
LOCKED_INTERPRETER = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python3.10")
EXPERIMENT_NAME = "pcraft_soft_loss_v7_hit_barrier_256_20260829"
ROOT_SCHEMA = "pcraft_soft_loss_v7_root_v1"
SEED = 17
SELECTOR_SEED = 20260829
EXPECTED_SPECIFICATION_SHA256 = "d6f3219080f61acc9a3d6c2b435dfd0fd12eac06b68c2f4a301f0b5fedfa74c1"
TASK1_SOURCE_SHA256 = {
    "__init__.py": "79c04de43816d694eec901bbc6a9b6a0dae88ca6220b46373bb5196af4dddcb3",
    "core.py": "25bfd4844e3c0a378d1289e8eb626a333f039c839847669d3ab4e4e632214fcf",
    "protocol.py": "8e627eb4d897893fafcc66c607b444891f2e6888696c2f410760460cf2da5ce3",
}
EXPECTED_INTERPRETER_SHA256 = "7365743bea8011d4ada1d0379699045fe411de80f620c39e4c4625855f0b1772"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_RENAME_NOREPLACE = 1
_ROOT_PUBLICATION_HOOK = lambda stage: None
ROOT_DIRECTORY_NAMES = frozenset(
    {"manifests", "authorizations", "runs", "analysis", "postprocess"}
)
REGISTERED_MANIFEST_NAMES = frozenset(
    {"development_suite.json", "formal_suite.json", "campaign.json", "formal_matrix.json"}
)


@dataclass(frozen=True)
class AuthenticatedFileSnapshot:
    path: str
    sha256: str
    payload: bytes
    device: int
    inode: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int
    exclusive: bool


_FileSnapshot = AuthenticatedFileSnapshot


@dataclass(frozen=True)
class AuthenticatedV7Root:
    path: str
    marker_path: str
    marker_sha256: str
    semantic_sha256: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class _BoundDirectory:
    path: Path
    descriptor: int
    device: int
    inode: int


def _normalize_json(value: object) -> object:
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("canonical JSON numbers must be finite")
        return value
    if value_type is dict:
        result: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise TypeError("canonical JSON keys must be exact built-in strings")
            if key in result:
                raise ValueError(f"duplicate canonical JSON key: {key!r}")
            result[key] = _normalize_json(nested)
        return result
    if value_type in (list, tuple):
        return [_normalize_json(nested) for nested in value]
    raise TypeError(f"unsupported canonical JSON value: {value_type.__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Encode only exact finite JSON built-ins with deterministic key order."""

    return json.dumps(
        _normalize_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def decode_canonical_json(raw: bytes, *, label: str) -> dict[str, object]:
    """Decode strict UTF-8 JSON while rejecting duplicate keys and constants."""

    if type(raw) is not bytes or type(label) is not str or not label:
        raise TypeError("raw and label must be exact bytes/non-empty str")

    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant in {label}: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    _normalize_json(value)
    return value


def _path_argument(path: Path, *, label: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{label} must be a pathlib.Path")
    candidate = path
    if not candidate.is_absolute() or candidate != Path(os.path.abspath(candidate)):
        raise ValueError(f"{label} must be absolute and canonical")
    if ".." in candidate.parts:
        raise ValueError(f"{label} contains traversal")
    return candidate


def _reject_symlink_components(path: Path, *, allow_missing_leaf: bool = False) -> None:
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if allow_missing_leaf and index == len(path.parts[1:]) - 1:
                return
            raise ValueError(f"path component is unavailable: {current}")
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"path contains a symlink component: {current}")


def _open_directory_fd(path: Path, *, label: str) -> int:
    """Bind an absolute directory by walking every component with O_NOFOLLOW."""

    candidate = _path_argument(path, label=label)
    flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        current = os.open(candidate.anchor, flags)
    except OSError as error:
        raise ValueError(f"{label} root is unavailable") from error
    try:
        for component in candidate.parts[1:]:
            try:
                following = os.open(component, flags, dir_fd=current)
            except OSError as error:
                raise ValueError(f"{label} contains a missing or symlink component") from error
            os.close(current)
            current = following
        metadata = os.fstat(current)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} must be a directory")
        return current
    except BaseException:
        os.close(current)
        raise


def _verify_directory_binding(descriptor: int, path: Path, *, label: str) -> None:
    opened = os.fstat(descriptor)
    try:
        current = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} changed after fd binding") from error
    if (
        not stat.S_ISDIR(opened.st_mode) or not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode) or opened.st_dev != current.st_dev
        or opened.st_ino != current.st_ino
    ):
        raise ValueError(f"{label} identity changed after fd binding")


def _canonical_existing_directory(path: Path, *, label: str) -> Path:
    candidate = _path_argument(path, label=label)
    _reject_symlink_components(candidate)
    descriptor = _open_directory_fd(candidate, label=label)
    try:
        _verify_directory_binding(descriptor, candidate, label=label)
    finally:
        os.close(descriptor)
    return candidate


def _bind_directory(path: Path, *, label: str) -> _BoundDirectory:
    candidate = _path_argument(path, label=label)
    _reject_symlink_components(candidate)
    descriptor = _open_directory_fd(candidate, label=label)
    try:
        metadata = os.fstat(descriptor)
        _verify_directory_binding(descriptor, candidate, label=label)
        return _BoundDirectory(
            candidate, descriptor, metadata.st_dev, metadata.st_ino
        )
    except BaseException:
        os.close(descriptor)
        raise


def _verify_bound_directory(bound: _BoundDirectory, *, label: str) -> None:
    metadata = os.fstat(bound.descriptor)
    if metadata.st_dev != bound.device or metadata.st_ino != bound.inode:
        raise ValueError(f"{label} descriptor identity changed")
    _verify_directory_binding(bound.descriptor, bound.path, label=label)


def _open_relative_directory_fd(
    root_descriptor: int, relative: Path, *, label: str,
) -> int:
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"{label} must be a contained relative directory")
    flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    current = os.dup(root_descriptor)
    try:
        for component in relative.parts:
            try:
                following = os.open(component, flags, dir_fd=current)
            except OSError as error:
                raise ValueError(
                    f"{label} contains a missing, non-directory, or symlink component"
                ) from error
            os.close(current)
            current = following
        return current
    except BaseException:
        os.close(current)
        raise


def _snapshot_regular_file_at(
    candidate: Path, parent_descriptor: int, *, exclusive: bool,
) -> AuthenticatedFileSnapshot:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate.name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise ValueError(
            f"regular file is unavailable or symlinked: {candidate}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("snapshot target must be a regular file")
        if exclusive and before.st_nlink != 1:
            raise ValueError("exclusive publication has invalid hardlink link count")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            final = os.stat(
                candidate.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
        except OSError as error:
            raise ValueError("file identity changed during snapshot") from error
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev, before.st_ino, before.st_nlink, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns,
    )
    if identity != (
        after.st_dev, after.st_ino, after.st_nlink, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    ) or identity != (
        final.st_dev, final.st_ino, final.st_nlink, final.st_size,
        final.st_mtime_ns, final.st_ctime_ns,
    ):
        raise ValueError("file identity changed during snapshot")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise ValueError("file size changed during snapshot")
    return AuthenticatedFileSnapshot(
        str(candidate), hashlib.sha256(payload).hexdigest(), payload,
        before.st_dev, before.st_ino, before.st_nlink, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns, exclusive,
    )


def _snapshot_regular_file_bound(
    path: Path, *, root: _BoundDirectory, exclusive: bool,
) -> AuthenticatedFileSnapshot:
    candidate = _path_argument(path, label="file path")
    try:
        relative = candidate.relative_to(root.path)
    except ValueError as error:
        raise ValueError("file path escapes its declared root") from error
    if not relative.parts:
        raise ValueError("snapshot target must be below its declared root")
    parent_descriptor = _open_relative_directory_fd(
        root.descriptor, Path(*relative.parts[:-1]), label="file parent"
    )
    try:
        snapshot = _snapshot_regular_file_at(
            candidate, parent_descriptor, exclusive=exclusive
        )
    finally:
        os.close(parent_descriptor)
    _verify_bound_directory(root, label="containment root")
    return snapshot


def snapshot_regular_file(
    path: Path, *, root: Path | None = None, exclusive: bool = True
) -> AuthenticatedFileSnapshot:
    """Read one file through an fd-relative no-follow chain and retain inode identity."""

    if type(exclusive) is not bool:
        raise TypeError("exclusive must be an exact boolean")
    candidate = _path_argument(path, label="file path")
    if root is not None:
        containment = _bind_directory(root, label="containment root")
        try:
            return _snapshot_regular_file_bound(
                candidate, root=containment, exclusive=exclusive
            )
        finally:
            os.close(containment.descriptor)
    parent = _bind_directory(candidate.parent, label="file parent")
    try:
        snapshot = _snapshot_regular_file_at(
            candidate, parent.descriptor, exclusive=exclusive
        )
        _verify_bound_directory(parent, label="file parent")
        return snapshot
    finally:
        os.close(parent.descriptor)


def revalidate_snapshot(
    snapshot: AuthenticatedFileSnapshot, *, root: Path | None = None
) -> None:
    if not isinstance(snapshot, AuthenticatedFileSnapshot):
        raise TypeError("snapshot has the wrong type")
    current = snapshot_regular_file(
        Path(snapshot.path), root=root, exclusive=snapshot.exclusive
    )
    if current != snapshot:
        raise ValueError("dependency file identity or bytes changed")


def sha256_file(path: Path) -> str:
    return snapshot_regular_file(path, exclusive=False).sha256


_snapshot_regular_file = snapshot_regular_file
_revalidate_snapshot = revalidate_snapshot


def _dependency_snapshots() -> tuple[dict[str, object], tuple[_FileSnapshot, ...]]:
    project = _canonical_existing_directory(PROJECT_ROOT, label="project root")
    specification = _snapshot_regular_file(SPECIFICATION_PATH, root=project)
    if specification.sha256 != EXPECTED_SPECIFICATION_SHA256:
        raise ValueError("v7 specification bytes differ from the approved SHA-256")
    interpreter = _snapshot_regular_file(LOCKED_INTERPRETER)
    if interpreter.sha256 != EXPECTED_INTERPRETER_SHA256:
        raise ValueError("locked interpreter bytes differ from the approved SHA-256")
    source_snapshots: list[_FileSnapshot] = []
    observed_sources: dict[str, str] = {}
    package = project / "shadowforge_pipeline/soft_voxel_v7"
    for name, expected in TASK1_SOURCE_SHA256.items():
        snapshot = _snapshot_regular_file(package / name, root=project)
        if snapshot.sha256 != expected:
            raise ValueError(f"approved Task 1 source changed: {name}")
        source_snapshots.append(snapshot)
        observed_sources[name] = snapshot.sha256
    evidence = {
        "project_root": str(project),
        "interpreter": {"path": interpreter.path, "sha256": interpreter.sha256},
        "specification": {"path": specification.path, "sha256": specification.sha256},
        "task1_sources": observed_sources,
    }
    return evidence, (interpreter, specification, *source_snapshots)


def _root_payload(root: Path) -> tuple[dict[str, object], tuple[_FileSnapshot, ...]]:
    evidence, snapshots = _dependency_snapshots()
    payload: dict[str, object] = {
        "schema": ROOT_SCHEMA,
        "root_path": str(root),
        "project_root": evidence["project_root"],
        "experiment_name": EXPERIMENT_NAME,
        "seed": SEED,
        "selector_seed": SELECTOR_SEED,
        "interpreter": evidence["interpreter"],
        "specification": evidence["specification"],
        "task1_sources": evidence["task1_sources"],
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload, snapshots


def _rename_noreplace_at(
    source_descriptor: int, source_name: str,
    destination_descriptor: int, destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_descriptor, os.fsencode(source_name), destination_descriptor,
        os.fsencode(destination_name), _RENAME_NOREPLACE,
    )
    if result != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), destination_name)


def fsync_directory(path: Path) -> None:
    bound = _bind_directory(path, label="fsync directory")
    try:
        os.fsync(bound.descriptor)
        _verify_bound_directory(bound, label="fsync directory")
    finally:
        os.close(bound.descriptor)


def _publish_bytes_once_bound(
    path: Path, payload: bytes, *, root: _BoundDirectory,
) -> None:
    if type(payload) is not bytes:
        raise TypeError("publication payload must be exact bytes")
    destination = _path_argument(path, label="publication path")
    try:
        relative = destination.relative_to(root.path)
    except ValueError as error:
        raise ValueError("publication path escapes its declared root") from error
    if not relative.parts:
        raise ValueError("publication target must be below its declared root")
    parent_descriptor = _open_relative_directory_fd(
        root.descriptor, Path(*relative.parts[:-1]), label="publication parent"
    )
    temporary_name = f".pcraft-soft-loss-v7-write-{destination.name}-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_descriptor)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short immutable publication write")
                    view = view[written:]
                os.fsync(descriptor)
                created = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(created.st_mode) or created.st_nlink != 1
                    or created.st_size != len(payload)
                ):
                    raise ValueError("immutable publication temporary identity changed")
            finally:
                os.close(descriptor)
            try:
                _rename_noreplace_at(
                    parent_descriptor, temporary_name,
                    parent_descriptor, destination.name,
                )
            except OSError as error:
                if error.errno != 17:
                    raise
                existing = _snapshot_regular_file_at(
                    destination, parent_descriptor, exclusive=True
                )
                if existing.payload != payload:
                    raise FileExistsError(f"immutable publication differs: {destination}")
            os.fsync(parent_descriptor)
            final = _snapshot_regular_file_at(
                destination, parent_descriptor, exclusive=True
            )
            if final.payload != payload:
                raise ValueError("immutable publication final bytes changed")
            _verify_bound_directory(root, label="publication root")
        finally:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
    finally:
        os.close(parent_descriptor)


def publish_bytes_once(path: Path, payload: bytes, *, root: Path) -> None:
    """Create immutable bytes through one bound parent fd without replacement."""

    containment = _bind_directory(root, label="publication root")
    try:
        _publish_bytes_once_bound(path, payload, root=containment)
    finally:
        os.close(containment.descriptor)


def _publish_bytes_once(path: Path, payload: bytes) -> None:
    publish_bytes_once(path, payload, root=path.parent)


_fsync_directory = fsync_directory


def _deep_read_only(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType({key: _deep_read_only(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_deep_read_only(item) for item in value)
    return value


def _require_exact_json(value: object, expected: object, *, label: str) -> None:
    if type(value) is not type(expected):
        raise ValueError(
            f"{label} has wrong exact type {type(value).__name__}; "
            f"expected {type(expected).__name__}"
        )
    if type(expected) is dict:
        observed = value
        if set(observed) != set(expected):
            raise ValueError(f"{label} has the wrong exact schema")
        for key, nested in expected.items():
            _require_exact_json(observed[key], nested, label=f"{label}.{key}")
    elif type(expected) is list:
        observed = value
        if len(observed) != len(expected):
            raise ValueError(f"{label} has the wrong exact list length")
        for index, (item, nested) in enumerate(zip(observed, expected, strict=True)):
            _require_exact_json(item, nested, label=f"{label}[{index}]")
    elif value != expected:
        raise ValueError(f"{label} differs from the frozen exact value")


def _validate_root_namespace_bound(root: _BoundDirectory) -> None:
    entries = set(os.listdir(root.descriptor))
    allowed = {"V7_ROOT.json", *ROOT_DIRECTORY_NAMES}
    if not entries.issubset(allowed) or "V7_ROOT.json" not in entries:
        raise ValueError("v7 root contains an unexpected namespace entry")
    for name in entries.intersection(ROOT_DIRECTORY_NAMES):
        descriptor = _open_relative_directory_fd(
            root.descriptor, Path(name), label=f"v7 {name} directory"
        )
        os.close(descriptor)


def _validate_root_namespace(root: Path) -> None:
    bound = _bind_directory(root, label="v7 root")
    try:
        _validate_root_namespace_bound(bound)
        _verify_bound_directory(bound, label="v7 root")
    finally:
        os.close(bound.descriptor)


def _validate_v7_root_bound(root: _BoundDirectory) -> AuthenticatedV7Root:
    _validate_root_namespace_bound(root)
    marker = _snapshot_regular_file_bound(
        root.path / "V7_ROOT.json", root=root, exclusive=True
    )
    payload = decode_canonical_json(marker.payload, label="v7 root marker")
    if marker.payload != canonical_json_bytes(payload):
        raise ValueError("v7 root marker is not canonical JSON")
    expected, dependencies = _root_payload(root.path)
    semantic = payload.get("semantic_sha256")
    if type(semantic) is not str or _DIGEST.fullmatch(semantic) is None:
        raise ValueError("v7 root semantic SHA has the wrong exact type")
    unsigned = dict(payload)
    del unsigned["semantic_sha256"]
    if canonical_sha256(unsigned) != semantic:
        raise ValueError("v7 root semantic SHA does not recompute")
    _require_exact_json(payload, expected, label="v7 root marker")
    for snapshot in dependencies:
        revalidate_snapshot(
            snapshot,
            root=PROJECT_ROOT if Path(snapshot.path).is_relative_to(PROJECT_ROOT) else None,
        )
    current_marker = _snapshot_regular_file_bound(
        root.path / "V7_ROOT.json", root=root, exclusive=True
    )
    if current_marker != marker:
        raise ValueError("v7 root marker identity or bytes changed")
    _verify_bound_directory(root, label="v7 root")
    return AuthenticatedV7Root(
        str(root.path), marker.path, marker.sha256, str(payload["semantic_sha256"]),
        _deep_read_only(payload),
    )


def validate_v7_root(output_root: Path) -> AuthenticatedV7Root:
    root = _bind_directory(
        _path_argument(output_root, label="v7 root"), label="v7 root"
    )
    try:
        return _validate_v7_root_bound(root)
    finally:
        os.close(root.descriptor)


def bootstrap_v7_root(output_root: Path) -> AuthenticatedV7Root:
    """Create only an absent v7 root and marker, or authenticate an exact retry."""

    root = _path_argument(output_root, label="v7 root")
    parent = _bind_directory(root.parent, label="v7 root parent")
    created = False
    root_bound: _BoundDirectory | None = None
    try:
        fcntl.flock(parent.descriptor, fcntl.LOCK_EX)
        try:
            metadata = os.stat(
                root.name, dir_fd=parent.descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            metadata = None
        if metadata is not None:
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ValueError("preexisting v7 root must be an ordinary directory")
            descriptor = _open_relative_directory_fd(
                parent.descriptor, Path(root.name), label="v7 root"
            )
            opened = os.fstat(descriptor)
            root_bound = _BoundDirectory(
                root, descriptor, opened.st_dev, opened.st_ino
            )
            identity = _validate_v7_root_bound(root_bound)
            _verify_bound_directory(parent, label="v7 root parent")
            return identity
        os.mkdir(root.name, mode=0o755, dir_fd=parent.descriptor)
        created = True
        descriptor = _open_relative_directory_fd(
            parent.descriptor, Path(root.name), label="v7 root"
        )
        opened = os.fstat(descriptor)
        root_bound = _BoundDirectory(root, descriptor, opened.st_dev, opened.st_ino)
        try:
            _ROOT_PUBLICATION_HOOK("after_root_create")
            payload, dependencies = _root_payload(root)
            for snapshot in dependencies:
                revalidate_snapshot(snapshot, root=PROJECT_ROOT if Path(snapshot.path).is_relative_to(PROJECT_ROOT) else None)
            _publish_bytes_once_bound(
                root / "V7_ROOT.json", canonical_json_bytes(payload), root=root_bound
            )
            for snapshot in dependencies:
                revalidate_snapshot(snapshot, root=PROJECT_ROOT if Path(snapshot.path).is_relative_to(PROJECT_ROOT) else None)
        except BaseException:
            try:
                if not os.listdir(root_bound.descriptor):
                    os.rmdir(root.name, dir_fd=parent.descriptor)
            except OSError:
                pass
            raise
        os.fsync(parent.descriptor)
        _verify_bound_directory(parent, label="v7 root parent")
        return _validate_v7_root_bound(root_bound)
    finally:
        if root_bound is not None:
            os.close(root_bound.descriptor)
        fcntl.flock(parent.descriptor, fcntl.LOCK_UN)
        os.close(parent.descriptor)


class V7RootTransaction:
    """One exclusive, retained-inode lifecycle transaction for a v7 root."""

    __slots__ = (
        "_root_path", "_bound", "_entry_marker", "_directories",
        "_active", "_closed",
    )

    def __init__(self, output_root: Path) -> None:
        self._root_path = _path_argument(output_root, label="v7 root")
        self._bound: _BoundDirectory | None = None
        self._entry_marker: AuthenticatedFileSnapshot | None = None
        self._directories: dict[str, _BoundDirectory] = {}
        self._active = False
        self._closed = False

    def _require_active(self) -> _BoundDirectory:
        if not self._active or self._closed or self._bound is None:
            raise RuntimeError("v7 root transaction is not active or is closed")
        return self._bound

    @staticmethod
    def _verify_registered_directory_mapping(
        root: _BoundDirectory, name: str, directory: _BoundDirectory,
    ) -> None:
        opened = os.fstat(directory.descriptor)
        try:
            named = os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                f"registered directory {name!r} changed after fd binding"
            ) from error
        if (
            not stat.S_ISDIR(opened.st_mode) or not stat.S_ISDIR(named.st_mode)
            or stat.S_ISLNK(named.st_mode) or opened.st_dev != directory.device
            or opened.st_ino != directory.inode or named.st_dev != directory.device
            or named.st_ino != directory.inode
        ):
            raise ValueError(
                f"registered directory {name!r} identity changed after fd binding"
            )

    @staticmethod
    def _bind_registered_directory(
        root: _BoundDirectory, name: str,
    ) -> _BoundDirectory:
        descriptor = _open_relative_directory_fd(
            root.descriptor, Path(name), label=f"registered directory {name!r}"
        )
        try:
            metadata = os.fstat(descriptor)
            directory = _BoundDirectory(
                root.path / name, descriptor, metadata.st_dev, metadata.st_ino
            )
            V7RootTransaction._verify_registered_directory_mapping(
                root, name, directory
            )
            return directory
        except BaseException:
            os.close(descriptor)
            raise

    def _verify_transaction_state(self) -> None:
        root = self._require_active()
        entry_marker = self._entry_marker
        if entry_marker is None:
            raise RuntimeError("v7 root transaction marker binding is unavailable")
        _verify_bound_directory(root, label="v7 root")
        _validate_root_namespace_bound(root)
        current_marker = _snapshot_regular_file_bound(
            root.path / "V7_ROOT.json", root=root, exclusive=True
        )
        if current_marker != entry_marker:
            raise ValueError("v7 root marker identity or snapshot changed")
        entries = set(os.listdir(root.descriptor)).intersection(ROOT_DIRECTORY_NAMES)
        if entries != set(self._directories):
            raise ValueError("registered directory binding set changed")
        for name, directory in self._directories.items():
            self._verify_registered_directory_mapping(root, name, directory)
        _verify_bound_directory(root, label="v7 root")

    def _close_retained_descriptors(self, root: _BoundDirectory) -> None:
        directories = tuple(self._directories.values())
        self._directories.clear()
        self._entry_marker = None
        self._active = False
        self._closed = True
        self._bound = None
        for directory in directories:
            try:
                os.close(directory.descriptor)
            except OSError:
                pass
        try:
            fcntl.flock(root.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(root.descriptor)

    def _bound_registered_path(
        self, relative: Path,
    ) -> tuple[_BoundDirectory, Path]:
        if len(relative.parts) < 2:
            raise ValueError("registered file path must name a descendant")
        directory = self._directories.get(relative.parts[0])
        if directory is None:
            raise ValueError("registered directory has not been retained")
        return directory, directory.path.joinpath(*relative.parts[1:])

    @staticmethod
    def _registered_relative_path(
        value: Path, *, label: str, allow_top_directory: bool = False,
    ) -> Path:
        if type(value) is not type(Path()):
            raise TypeError(f"{label} must be an exact built-in Path")
        if value.is_absolute() or not value.parts:
            raise ValueError(f"{label} must be a non-empty relative path")
        if any(part in ("", ".", "..") for part in value.parts):
            raise ValueError(f"{label} contains a forbidden relative component")
        if value.parts[0] not in ROOT_DIRECTORY_NAMES:
            raise ValueError(f"{label} has an unregistered top-level owner")
        if value.parts[0] == "manifests" and not (
            allow_top_directory and len(value.parts) == 1
        ) and (
            len(value.parts) != 2 or value.parts[1] not in REGISTERED_MANIFEST_NAMES
        ):
            raise ValueError(f"{label} has an unregistered manifest path")
        return value

    def __enter__(self) -> V7RootTransaction:
        if self._active or self._closed:
            raise RuntimeError("v7 root transaction cannot be nested or reused")
        bound = _bind_directory(self._root_path, label="v7 root")
        try:
            try:
                fcntl.flock(bound.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise BlockingIOError(
                    "v7 root transaction lock ownership is already active"
                ) from error
            self._bound = bound
            self._active = True
            _validate_v7_root_bound(bound)
            self._entry_marker = _snapshot_regular_file_bound(
                bound.path / "V7_ROOT.json", root=bound, exclusive=True
            )
            for name in sorted(
                set(os.listdir(bound.descriptor)).intersection(ROOT_DIRECTORY_NAMES)
            ):
                self._directories[name] = self._bind_registered_directory(
                    bound, name
                )
            self._verify_transaction_state()
            return self
        except BaseException:
            if self._active:
                self._close_retained_descriptors(bound)
            else:
                os.close(bound.descriptor)
            raise

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        bound = self._require_active()
        terminal_error: BaseException | None = None
        try:
            _validate_v7_root_bound(bound)
            self._verify_transaction_state()
        except BaseException as error:
            terminal_error = error
        finally:
            self._close_retained_descriptors(bound)
        if terminal_error is not None:
            raise terminal_error
        return False

    def ensure_registered_directory(self, name: str) -> None:
        bound = self._require_active()
        if type(name) is not str:
            raise TypeError("registered directory name must be an exact string")
        if name not in ROOT_DIRECTORY_NAMES:
            raise ValueError("directory name is not registered")
        self._verify_transaction_state()
        try:
            metadata = os.stat(name, dir_fd=bound.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            os.mkdir(name, mode=0o755, dir_fd=bound.descriptor)
            os.fsync(bound.descriptor)
            self._directories[name] = self._bind_registered_directory(bound, name)
        else:
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise ValueError("registered directory entry is not an ordinary directory")
            if name not in self._directories:
                raise ValueError("registered directory has no retained binding")
        self._verify_transaction_state()

    def snapshot(
        self, relative_path: Path, exclusive: bool = True,
    ) -> AuthenticatedFileSnapshot:
        bound = self._require_active()
        if type(exclusive) is not bool:
            raise TypeError("exclusive must be an exact boolean")
        relative = self._registered_relative_path(relative_path, label="snapshot path")
        self._verify_transaction_state()
        directory, candidate = self._bound_registered_path(relative)
        try:
            return _snapshot_regular_file_bound(
                candidate, root=directory, exclusive=exclusive
            )
        finally:
            self._verify_transaction_state()

    def publish(self, relative_path: Path, payload: bytes) -> None:
        bound = self._require_active()
        relative = self._registered_relative_path(relative_path, label="publication path")
        if type(payload) is not bytes:
            raise TypeError("publication payload must be exact bytes")
        self._verify_transaction_state()
        directory, destination = self._bound_registered_path(relative)
        try:
            _publish_bytes_once_bound(destination, payload, root=directory)
        finally:
            self._verify_transaction_state()

    def fsync(self, relative_directory: Path) -> None:
        bound = self._require_active()
        relative = self._registered_relative_path(
            relative_directory, label="fsync directory", allow_top_directory=True
        )
        self._verify_transaction_state()
        directory = self._directories.get(relative.parts[0])
        if directory is None:
            raise ValueError("registered directory has not been retained")
        remainder = Path(*relative.parts[1:])
        descriptor = (
            os.dup(directory.descriptor) if not remainder.parts
            else _open_relative_directory_fd(
                directory.descriptor, remainder, label="fsync directory"
            )
        )
        try:
            before = os.fstat(descriptor)
            os.fsync(descriptor)
            after = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(before.st_mode) or before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
            ):
                raise ValueError("fsync directory identity changed")
        finally:
            os.close(descriptor)
        self._verify_transaction_state()

    def revalidate_root_marker(self) -> None:
        bound = self._require_active()
        _validate_v7_root_bound(bound)
        self._verify_transaction_state()


def open_v7_root_transaction(output_root: Path) -> V7RootTransaction:
    """Return an unopened exclusive lifecycle transaction for one v7 root."""

    return V7RootTransaction(output_root)


__all__ = [
    "AuthenticatedFileSnapshot",
    "AuthenticatedV7Root",
    "V7RootTransaction",
    "bootstrap_v7_root",
    "canonical_json_bytes",
    "canonical_sha256",
    "decode_canonical_json",
    "fsync_directory",
    "open_v7_root_transaction",
    "publish_bytes_once",
    "revalidate_snapshot",
    "sha256_file",
    "snapshot_regular_file",
    "validate_v7_root",
]
