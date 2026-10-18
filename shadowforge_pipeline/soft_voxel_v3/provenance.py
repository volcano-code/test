"""Canonical provenance and crash-safe primitive writers for soft-loss v3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

import torch


AUTHORITATIVE_SOURCE_ROOT = Path(__file__).resolve().parent
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


def _normalized_key(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("mapping keys must be finite")
        return str(value)
    raise TypeError(f"unsupported canonical mapping key type: {type(value).__name__}")


def normalize_builtin(value: object) -> object:
    """Normalize JSON-compatible built-ins without stringifying foreign objects."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical values must be finite")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = _normalized_key(raw_key)
            if key in result:
                raise ValueError(f"duplicate canonical mapping key: {key!r}")
            result[key] = normalize_builtin(raw_value)
        return result
    if isinstance(value, (list, tuple)):
        return [normalize_builtin(item) for item in value]
    raise TypeError(f"unsupported canonical value type: {type(value).__name__}")


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    if not isinstance(payload, Mapping):
        raise TypeError("canonical payload must be a mapping")
    normalized = normalize_builtin(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Mapping[str, object]) -> str:
    """Return stable SHA-256 over compact, sorted, finite canonical JSON."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_source_manifest() -> dict[str, object]:
    """Hash every authoritative v3 Python module by relative file name."""

    root = Path(AUTHORITATIVE_SOURCE_ROOT).resolve()
    sources = sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.relative_to(root).parts
    )
    if not sources:
        raise RuntimeError(f"no authoritative v3 Python sources found in {root}")
    files = {path.relative_to(root).as_posix(): sha256_file(path) for path in sources}
    return {
        "algorithm": "sha256",
        "files_sha256": files,
        "bundle_sha256": canonical_hash(files),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unique_temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temporary(path)
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace one JSON file after syncing its bytes and directory."""

    normalized = normalize_builtin(payload)
    encoded = (
        json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_bytes(Path(path), encoded)


def atomic_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace one torch checkpoint from deterministic in-memory bytes."""

    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    _atomic_bytes(Path(path), buffer.getvalue())


def atomic_text(path: Path, text: str) -> None:
    _atomic_bytes(Path(path), text.encode("utf-8"))


def interpreter_manifest() -> dict[str, object]:
    path = Path(sys.executable).resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "python": platform.python_version(),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _allowed_environment() -> dict[str, str]:
    captured: dict[str, str] = {}
    for name in sorted(ALLOWED_ENVIRONMENT):
        if name not in os.environ:
            continue
        if any(part in name.upper() for part in _SECRET_NAME_PARTS):
            captured[name] = "<redacted>"
        else:
            captured[name] = os.environ[name]
    return captured


def _cuda_driver_version() -> str | None:
    def normalized(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or any(character not in "0123456789." for character in text):
            return None
        return text

    getter = getattr(torch.cuda, "driver_version", None)
    if callable(getter):
        try:
            value = normalized(getter())
            if value is not None:
                return value
        except Exception:
            pass
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
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
    if not values:
        return None
    return normalized(values[0])


def environment_manifest(device: torch.device) -> dict[str, object]:
    """Capture the strict environment allowlist and explicit determinism limits."""

    resolved = torch.device(device)
    gpu_name = None
    gpu_memory = None
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = resolved.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu_name = str(properties.name)
        gpu_memory = int(properties.total_memory)
    return {
        "schema": "pcraft_soft_loss_v3_environment_v1",
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
