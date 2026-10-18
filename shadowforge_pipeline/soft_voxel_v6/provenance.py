"""Strict canonical provenance and create-once publication for soft-loss v6."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
from typing import Mapping, Sequence

import torch


_DIGEST_LENGTH = 64
ALLOWED_ENVIRONMENT = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
)


@dataclass(frozen=True)
class FileSnapshot:
    path: str
    sha256: str
    size: int


def _builtin(value: object, *, label: str = "value") -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite")
        return value
    if type(value) is list or type(value) is tuple:
        return [_builtin(item, label=f"{label}[]") for item in value]
    if type(value) is dict:
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} keys must be strings")
            result[key] = _builtin(item, label=f"{label}.{key}")
        return result
    raise TypeError(f"{label} contains an unsupported value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    normalized = _builtin(value)
    return (
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute(path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def reject_symlink_components(path: Path, *, must_exist: bool = True) -> Path:
    candidate = _absolute(Path(path))
    if ".." in candidate.parts or candidate != Path(os.path.abspath(candidate)):
        raise ValueError(f"path must be absolute and canonical: {candidate}")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if must_exist:
                raise ValueError(f"required path is missing: {candidate}")
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"path contains a symlink component: {current}")
    if must_exist and candidate.resolve(strict=True) != candidate:
        raise ValueError(f"path is not canonical: {candidate}")
    return candidate


def snapshot_regular_file(path: Path) -> FileSnapshot:
    candidate = reject_symlink_components(Path(path))
    metadata = candidate.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"path must be a regular file: {candidate}")
    return FileSnapshot(str(candidate), sha256_file(candidate), int(metadata.st_size))


def revalidate_snapshot(snapshot: FileSnapshot) -> None:
    if not isinstance(snapshot, FileSnapshot):
        raise ValueError("snapshot has the wrong type")
    current = snapshot_regular_file(Path(snapshot.path))
    if current != snapshot:
        raise ValueError(f"authenticated file changed: {snapshot.path}")


def decode_canonical_json(path: Path, *, label: str) -> dict[str, object]:
    snapshot = snapshot_regular_file(path)

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            Path(snapshot.path).read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {label}: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON in {label}") from error
    if type(payload) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    if Path(snapshot.path).read_bytes() != canonical_json_bytes(payload):
        raise ValueError(f"{label} must use exact canonical JSON bytes")
    revalidate_snapshot(snapshot)
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes, *, create_once: bool = False) -> None:
    target = reject_symlink_components(Path(path), must_exist=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    reject_symlink_components(target.parent)
    if os.path.lexists(target):
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"target is not a safe regular file: {target}")
        if create_once and target.read_bytes() == payload:
            return
        if create_once:
            raise FileExistsError(f"immutable file already exists with different bytes: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if create_once:
            try:
                os.link(temporary, target)
            except FileExistsError:
                if target.is_file() and not target.is_symlink() and target.read_bytes() == payload:
                    return
                raise FileExistsError(
                    f"immutable file concurrently exists with different bytes: {target}"
                )
        else:
            os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_json(path: Path, payload: object, *, create_once: bool = False) -> None:
    atomic_bytes(path, canonical_json_bytes(payload), create_once=create_once)


def relative_regular_file(root: Path, relative: str) -> Path:
    if type(relative) is not str or not relative:
        raise ValueError("relative artifact path must be non-empty")
    posix = PurePosixPath(relative)
    if (
        posix.is_absolute()
        or posix.as_posix() != relative
        or any(part in ("", ".", "..") for part in posix.parts)
    ):
        raise ValueError("artifact path must be canonical and relative")
    base = reject_symlink_components(Path(root))
    candidate = base / relative
    reject_symlink_components(candidate)
    if not candidate.is_file():
        raise ValueError(f"artifact is missing: {relative}")
    return candidate


def implementation_bundle(project_root: Path, relatives: Sequence[str]) -> dict[str, object]:
    files: dict[str, str] = {}
    snapshots: list[FileSnapshot] = []
    root = reject_symlink_components(Path(project_root))
    for relative in relatives:
        if type(relative) is not str or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            raise ValueError("implementation source path must be canonical and relative")
        snapshot = snapshot_regular_file(root / relative)
        files[relative] = snapshot.sha256
        snapshots.append(snapshot)
    payload = {"files_sha256": files}
    payload["bundle_sha256"] = canonical_hash(payload)
    for snapshot in snapshots:
        revalidate_snapshot(snapshot)
    return payload


def environment_identity(device: str) -> dict[str, object]:
    """Capture only CUDA-build metadata; this never initializes a CUDA context."""

    if type(device) is not str or not device:
        raise ValueError("device must be a non-empty string")
    try:
        import pytorch3d

        pytorch3d_version = getattr(pytorch3d, "__version__", "unknown")
    except ImportError:
        pytorch3d_version = "unavailable"
    interpreter = snapshot_regular_file(Path(sys.executable).resolve())
    return {
        "schema": "pcraft_soft_loss_v6_environment_v1",
        "python": sys.version.split()[0],
        "interpreter": {
            "path": interpreter.path,
            "sha256": interpreter.sha256,
        },
        "torch": str(torch.__version__),
        "pytorch3d": str(pytorch3d_version),
        "cuda_build": None if torch.version.cuda is None else str(torch.version.cuda),
        "requested_device": device,
        "driver": os.environ.get("NVIDIA_DRIVER_VERSION", "runtime-verified"),
        "gpu": os.environ.get("NVIDIA_VISIBLE_DEVICES", "runtime-verified"),
        "environment": {
            key: os.environ[key] for key in ALLOWED_ENVIRONMENT if key in os.environ
        },
        "fixed_seed": True,
        "seed": 17,
        "bitwise_reproducible": False,
        "determinism": "authenticated-resume-best-effort",
    }

