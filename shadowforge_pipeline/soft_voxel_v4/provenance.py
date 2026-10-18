"""Strict byte provenance and crash-safe publication for soft-loss v4."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import uuid

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_SOURCE_FILES = (
    "shadowforge_pipeline/__init__.py",
    "shadowforge_pipeline/soft_voxel_v4/__init__.py",
    "shadowforge_pipeline/soft_voxel_v4/config.py",
    "shadowforge_pipeline/soft_voxel_v4/provenance.py",
    "shadowforge_pipeline/soft_voxel_v4/trainer.py",
    "shadowforge_pipeline/soft_voxel_v4/trainer_step.py",
    "shadowforge_pipeline/soft_voxel_v4/diagnostics.py",
    "shadowforge_pipeline/soft_voxel_v4/gradients.py",
    "shadowforge_pipeline/soft_voxel_v4/losses.py",
    "shadowforge_pipeline/soft_voxel_v3/__init__.py",
    "shadowforge_pipeline/soft_voxel_v3/contours.py",
    "shadowforge_pipeline/soft_voxel_v3/data.py",
    "shadowforge_pipeline/soft_voxel_v3/diagnostics.py",
    "shadowforge_pipeline/soft_voxel_v3/losses.py",
    "shadowforge_pipeline/soft_voxel_v3/lovasz.py",
    "shadowforge_pipeline/soft_voxel_v3/metrics.py",
    "shadowforge_pipeline/soft_voxel_v3/protocol.py",
    "shadowforge_pipeline/soft_voxel_v3/provenance.py",
    "shadowforge_pipeline/soft_voxel_v3/thresholding.py",
)
ALLOWED_ENVIRONMENT = frozenset(
    {
        "PYTHONPATH",
        "PYTHONHASHSEED",
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "TORCH_DETERMINISTIC",
        "TORCH_USE_CUDA_DSA",
        "NVIDIA_TF32_OVERRIDE",
    }
)
_SECRET_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_SNAPSHOT_METADATA: dict[int, tuple[int, int, int, str | None]] = {}


def _normalize_builtin(value: object) -> object:
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("canonical values must be finite")
        return value
    if value_type is dict:
        result: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise TypeError("canonical mapping keys must be exact built-in str")
            if key in result:
                raise ValueError(f"duplicate canonical key: {key!r}")
            result[key] = _normalize_builtin(nested)
        return result
    if value_type in (list, tuple):
        return [_normalize_builtin(nested) for nested in value]
    raise TypeError(f"unsupported canonical value type: {value_type.__name__}")


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Encode exact finite JSON built-ins with stable ordering and no coercion."""

    if type(payload) is not dict:
        raise TypeError("canonical payload must be an exact built-in dict")
    normalized = _normalize_builtin(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _canonical_existing_directory(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate != Path(os.path.abspath(candidate)):
        raise ValueError("directory path must be absolute and canonical")
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise ValueError(f"directory is unavailable: {candidate}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("directory must not be a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("path must be a directory")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise ValueError("directory path must contain no symlink component")
    return candidate


def _canonical_regular_path(path: Path, *, root: Path | None) -> tuple[Path, Path | None]:
    candidate = Path(path)
    if not candidate.is_absolute() or candidate != Path(os.path.abspath(candidate)):
        raise ValueError("file path must be absolute and canonical")
    containment = None
    if root is not None:
        containment = _canonical_existing_directory(Path(root))
        try:
            candidate.relative_to(containment)
        except ValueError as error:
            raise ValueError("file path is outside its declared root") from error
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ValueError(f"file path component is unavailable: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("file path must contain no symlink component")
    metadata = candidate.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("snapshot target must be a regular file")
    if candidate.resolve(strict=True) != candidate:
        raise ValueError("file path must be canonical")
    return candidate, containment


def snapshot_regular_file(path: Path, *, root: Path | None = None):
    """Read one regular file once through O_NOFOLLOW and retain stable bytes."""

    from .config import BoundFileSnapshot

    candidate, containment = _canonical_regular_path(Path(path), root=root)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("snapshot descriptor must identify a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ) or len(payload) != before.st_size:
        raise ValueError("file identity changed during snapshot")
    final = candidate.lstat()
    if (final.st_dev, final.st_ino, final.st_size) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
    ):
        raise ValueError("file identity changed during snapshot")
    snapshot = BoundFileSnapshot(
        path=str(candidate),
        sha256=hashlib.sha256(payload).hexdigest(),
        payload=payload,
    )
    _SNAPSHOT_METADATA[id(snapshot)] = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        None if containment is None else str(containment),
    )
    return snapshot


def revalidate_snapshot(snapshot: object, *, root: Path | None = None) -> None:
    """Reopen and compare path identity, inode, size, and digest to a bound snapshot."""

    from .config import BoundFileSnapshot

    if not isinstance(snapshot, BoundFileSnapshot):
        raise ValueError("snapshot must be a BoundFileSnapshot")
    declared_root = root
    metadata = _SNAPSHOT_METADATA.get(id(snapshot))
    if metadata is not None and metadata[3] is not None:
        if root is not None and str(Path(root)) != metadata[3]:
            raise ValueError("snapshot containment root changed")
        declared_root = Path(metadata[3])
    candidate, _ = _canonical_regular_path(Path(snapshot.path), root=declared_root)
    current = candidate.lstat()
    if metadata is not None and (current.st_dev, current.st_ino, current.st_size) != metadata[:3]:
        raise ValueError("snapshot file identity changed")
    replacement = snapshot_regular_file(candidate, root=declared_root)
    if replacement.sha256 != snapshot.sha256 or replacement.payload != snapshot.payload:
        raise ValueError("snapshot file bytes changed")


def sha256_file(path: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(os.path.abspath(candidate))
    snapshot = snapshot_regular_file(candidate)
    return snapshot.sha256


def snapshot_implementation_sources():
    snapshots = tuple(
        snapshot_regular_file(PROJECT_ROOT / relative, root=PROJECT_ROOT)
        for relative in IMPLEMENTATION_SOURCE_FILES
    )
    files = {
        relative: snapshot.sha256
        for relative, snapshot in zip(IMPLEMENTATION_SOURCE_FILES, snapshots)
    }
    manifest: dict[str, object] = {
        "schema": "pcraft_soft_loss_v4_source_manifest_v1",
        "algorithm": "sha256",
        "files_sha256": files,
        "bundle_sha256": canonical_hash(files),
    }
    return manifest, snapshots


def implementation_source_manifest() -> dict[str, object]:
    manifest, _ = snapshot_implementation_sources()
    return manifest


def interpreter_manifest() -> dict[str, object]:
    executable = Path(sys.executable)
    resolved = executable.resolve(strict=True)
    snapshot = snapshot_regular_file(resolved)
    return {
        "path": snapshot.path,
        "sha256": snapshot.sha256,
        "python": platform.python_version(),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _allowed_environment() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in sorted(ALLOWED_ENVIRONMENT):
        if name not in os.environ:
            continue
        result[name] = (
            "<redacted>"
            if any(part in name.upper() for part in _SECRET_NAME_PARTS)
            else os.environ[name]
        )
    return result


def _cuda_driver_version() -> str | None:
    def normalize(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or any(character not in "0123456789." for character in text):
            return None
        return text

    getter = getattr(torch.cuda, "driver_version", None)
    if callable(getter):
        try:
            observed = normalize(getter())
            if observed is not None:
                return observed
        except Exception:
            pass
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return None if not values else normalize(values[0])


def environment_manifest(device: torch.device) -> dict[str, object]:
    resolved = torch.device(device)
    gpu_name = None
    gpu_memory = None
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu_name = str(properties.name)
        gpu_memory = int(properties.total_memory)
    return {
        "schema": "pcraft_soft_loss_v4_environment_v1",
        "interpreter": interpreter_manifest(),
        "pytorch": str(torch.__version__),
        "pytorch3d": _package_version("pytorch3d"),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_driver": _cuda_driver_version(),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": gpu_memory,
        "environment": _allowed_environment(),
        "fixed_seed": True,
        "determinism": "best_effort",
        "bitwise_reproducible": False,
        "nondeterminism_reason": (
            "CUDA grid_sampler_3d_backward used by PyTorch3D may not provide "
            "bitwise deterministic backward execution."
        ),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_safe_destination(path: Path) -> None:
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = parent.resolve(strict=True)
    if resolved_parent != Path(os.path.abspath(parent)):
        raise ValueError("publication parent must contain no symlink component")
    try:
        metadata = Path(path).lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("publication target must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("publication target must be a regular file")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    destination = Path(path)
    _ensure_safe_destination(destination)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_bytes(Path(path), canonical_json_bytes(payload) + b"\n")


def atomic_text(path: Path, text: str) -> None:
    if type(text) is not str:
        raise TypeError("text payload must be an exact str")
    _atomic_bytes(Path(path), text.encode("utf-8"))


def atomic_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    if type(payload) is not dict:
        raise TypeError("checkpoint payload must be an exact dict")
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    _atomic_bytes(Path(path), buffer.getvalue())


def publish_json_once(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    encoded = canonical_json_bytes(payload) + b"\n"
    _ensure_safe_destination(destination)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError:
        _ensure_safe_destination(destination)
        existing = destination.read_bytes()
        if existing != encoded:
            raise FileExistsError(f"immutable JSON differs: {destination}")
        return
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(destination.parent)


__all__ = [
    "atomic_checkpoint",
    "atomic_json",
    "atomic_text",
    "canonical_hash",
    "canonical_json_bytes",
    "environment_manifest",
    "implementation_source_manifest",
    "interpreter_manifest",
    "publish_json_once",
    "sha256_file",
    "snapshot_implementation_sources",
    "snapshot_regular_file",
]
