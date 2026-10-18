#!/usr/bin/env python3
"""Contract-validated multiview soft-artifact delivery adapter."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Mapping, Sequence
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VOXEL_ROOT = Path(
    "/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel"
)
VOXEL_PYTHON = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python3.10")
BACKEND_PATH = PROJECT_ROOT / "experiments/multiview_contract_delivery_20260902.py"
BACKEND_ENTRYPOINT = (
    "experiments.multiview_contract_delivery_20260902.run_contract_delivery"
)
TOP_MANIFEST = "MULTIVIEW_EXPORT.json"
CONTRACT_MANIFEST = "CAMERA_CONTRACT.json"
STAGING_MANIFEST = "staging_delivery_manifest.json"
CANONICAL_EXPORTS = {
    "fabrication_stl": "delivery_postprocessed.stl",
    "fabrication_obj": "delivery_postprocessed.obj",
    "outer_preview_stl": "delivery_outer_preview.stl",
    "outer_preview_obj": "delivery_outer_preview.obj",
}


def build_soft_environment(
    base_environment: Mapping[str, str] | None = None
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{existing}" if existing else str(PROJECT_ROOT)
    )
    return environment


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    status: str
    n: int
    requested_tsf: int
    effective_tsf: int
    mirror: int
    trainer_tsf: int
    trainer_tsf4: int
    geometry_tsf: int
    views: tuple[tuple[float, float], ...]
    camera_protocol: str
    tsf_direct: bool
    renderer_family: str
    direct_view_roles: tuple[tuple[int | None, str | None], ...]
    zdist: float | None = None
    alias_of: str | None = None


@dataclass(frozen=True)
class CameraContract:
    protocol_id: str
    status: str
    requested_tsf: int
    effective_tsf: int
    mirror: int
    trainer_tsf: int
    trainer_tsf4: int
    views: tuple[tuple[float, float], ...]
    renderer_family: str
    direct_view_roles: tuple[tuple[int | None, str | None], ...]
    image_size: int
    zdist: float | None
    volume_extent: float | None
    ray_samples: int | None
    input_sha256: tuple[tuple[str, str], ...]
    normalized_target_sha256: tuple[tuple[str, str], ...]
    target_order: tuple[str, ...]
    soft_zyx_to_canonical_xyz: str
    preserve_target_aspect: bool
    no_registration: bool
    artifact_sha256: tuple[tuple[str, str], ...]


class AdapterFailure(RuntimeError):
    def __init__(
        self, status: str, reason: str, details: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(reason)
        self.status, self.reason, self.details = status, reason, dict(details or {})


def resolve_protocol(*, n: int, tsf: int, mirror: str | int = "auto") -> ProtocolSpec:
    if type(n) is not int or n not in (2, 3, 4):
        raise ValueError("n must be one of 2, 3, or 4")
    if type(tsf) is not int or tsf not in (0, 1):
        raise ValueError("tsf must be 0 or 1")
    if not (
        (type(mirror) is str and mirror == "auto")
        or (type(mirror) is int and mirror in (0, 1))
    ):
        raise ValueError("mirror must be auto, 0, or 1")
    mr = 0 if n == 2 and mirror == "auto" else 1 if mirror == "auto" else mirror
    if n in (3, 4) and mr == 0:
        raise ValueError("explicit mirror=0 is only supported for n=2")
    if n == 2:
        if mr == 0:
            views = ((0.0, 0.0), (90.0, 0.0))
            return ProtocolSpec(
                "dark_knight_two_wall",
                "extension",
                2,
                tsf,
                0,
                0,
                0,
                0,
                0,
                views,
                "custom_orthographic",
                False,
                "custom_orthographic",
                tuple((None, None) for _ in views),
                alias_of="dark_knight_two_wall" if tsf else None,
            )
        views = ((0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (270.0, 0.0))
        return ProtocolSpec(
            "dark_knight_two_wall_experimental_mr1",
            "extension",
            2,
            tsf,
            0,
            1,
            0,
            0,
            0,
            views,
            "custom_orthographic",
            False,
            "custom_orthographic",
            tuple((None, None) for _ in views),
            alias_of="dark_knight_two_wall_experimental_mr1" if tsf else None,
        )
    if tsf == 0:
        views = tuple((360.0 * i / (2 * n), 0.0) for i in range(2 * n))
        return ProtocolSpec(
            f"generic_n{n}",
            "extension",
            n,
            0,
            0,
            1,
            0,
            0,
            0,
            views,
            "custom_orthographic",
            False,
            "custom_orthographic",
            tuple((None, None) for _ in views),
        )
    if n == 3:
        views = (
            (0.0, 0.0),
            (90.0, 0.0),
            (0.0, 90.0),
            (180.0, 0.0),
            (270.0, 0.0),
            (0.0, -90.0),
        )
        roles = (
            (0, "ud"),
            (2, "tudlr"),
            (1, "udlr"),
            (0, "udlr"),
            (2, "tlr"),
            (1, "ud"),
        )
        return ProtocolSpec(
            "paper_fixed3_tfs_mr1",
            "paper",
            3,
            1,
            1,
            1,
            1,
            0,
            1,
            views,
            "sadekar_voxel_tfs_mirror_orthographic",
            True,
            "sadekar_voxel_tfs_mirror_orthographic",
            roles,
            zdist=1.7,
        )
    views = (
        (0.0, 0.0),
        (60.0, 0.0),
        (120.0, 0.0),
        (0.0, 90.0),
        (180.0, 0.0),
        (240.0, 0.0),
        (300.0, 0.0),
        (0.0, -90.0),
    )
    return ProtocolSpec(
        "tsf4_n4",
        "extension",
        4,
        1,
        1,
        1,
        0,
        1,
        0,
        views,
        "custom_orthographic",
        False,
        "custom_orthographic",
        tuple((None, None) for _ in views),
    )


def build_soft_command(
    *,
    spec: ProtocolSpec,
    silhouettes: Sequence[Path],
    output: Path,
    device: str,
    iterations: int,
    learning_rate: float,
    resolution: int,
    seed: int,
) -> list[str]:
    if len(silhouettes) != spec.n:
        raise ValueError("silhouette count does not match resolved protocol")
    if resolution != 256:
        raise ValueError("soft resolution is frozen at 256")
    return [
        str(VOXEL_PYTHON),
        "train.py",
        device,
        str(output),
        str(iterations),
        str(learning_rate),
        "1",
        "-sdlist",
        *map(str, silhouettes),
        "-mr",
        str(spec.mirror),
        "-tsf",
        str(spec.trainer_tsf),
        "-tsf4",
        str(spec.trainer_tsf4),
        "-cam",
        "ortho",
        "-imsz",
        "256",
        "--res",
        str(resolution),
        "-eres",
        "256",
        "--control-res",
        "0",
        "--proposal-mode",
        "proposal",
        "--view-update-mode",
        "joint",
        "--disable-losses",
        "volume",
        "--preserve-target-aspect",
        "--connectivity-start-iou",
        "0.60",
        "--fidelity-gate",
        "0.85",
        "--structure-warmup-steps",
        "300",
        "--structure-ramp-steps",
        "300",
        "--soft-proposal-min-iou",
        "0.95",
        "--save-every",
        "100",
        "--seed",
        str(seed),
    ] + (["--sadekar-zdist", str(spec.zdist)] if spec.zdist is not None else [])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_artifact_hashes(path: Path) -> tuple[tuple[str, str], ...]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return tuple(
        (item.relative_to(path).as_posix(), _sha256(item))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    )


def recheck_artifact_hashes(path: Path, snapshot: tuple[tuple[str, str], ...]) -> None:
    if snapshot_artifact_hashes(path) != snapshot:
        raise ValueError("soft artifact mutated after hash snapshot")


def _normalize_target(
    image: object, *, size: int, preserve_aspect: bool = False
) -> object:
    import cv2
    import numpy as np

    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST)
    foreground = cv2.findNonZero(image)
    if foreground is None:
        return image
    x, y, width, height = cv2.boundingRect(foreground)
    if preserve_aspect:
        scale = min(0.7 * size / max(width, 1), 0.7 * size / max(height, 1))
        matrix = np.float32(
            [
                [scale, 0, size / 2 - scale * (x + width / 2)],
                [0, scale, size / 2 - scale * (y + height / 2)],
            ]
        )
    else:
        scale_x = 0.7 * size / max(width, 1)
        scale_y = 0.7 * size / max(height, 1)
        matrix = np.float32(
            [
                [
                    scale_x,
                    0,
                    size / 2 * (1 - scale_x) + (size / 2 - (x + width / 2)),
                ],
                [
                    0,
                    scale_y,
                    size / 2 * (1 - scale_y) + (size / 2 - (y + height / 2)),
                ],
            ]
        )
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_NEAREST)


def _rebuild_normalized_targets(
    silhouettes: Sequence[Path], *, mirror: bool, size: int
) -> tuple[object, ...]:
    import cv2

    raw = []
    for silhouette in silhouettes:
        image = cv2.imread(str(silhouette), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"failed to decode silhouette: {silhouette}")
        raw.append(image)
    targets = [
        _normalize_target(image, size=size, preserve_aspect=True) for image in raw
    ]
    if mirror:
        targets.extend(
            _normalize_target(cv2.flip(image, 1), size=size, preserve_aspect=True)
            for image in raw
        )
    return tuple(targets)


def validate_soft_artifact(
    path: Path,
    *,
    spec: ProtocolSpec,
    resolution: int,
    silhouettes: Sequence[Path],
    allow_legacy_camera: bool = False,
) -> CameraContract:
    if resolution != 256:
        raise ValueError("resolution is frozen at 256")
    required = [
        path / "metrics.json",
        path / "occupancy_probability_core.npy",
        path / "occupancy_soft_proposal.npy",
        *[path / f"gt_view_{i}.png" for i in range(len(spec.views))],
    ]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError("incomplete soft artifact: " + ", ".join(missing))
    import numpy as np

    if len(silhouettes) != spec.n:
        raise ValueError("silhouette count does not match resolved protocol")
    expected_names = {f"gt_view_{i}.png" for i in range(len(spec.views))}
    actual_names = {item.name for item in path.glob("gt_view_*.png") if item.is_file()}
    if actual_names != expected_names:
        raise ValueError("normalized target file set mismatch")
    for name in ("occupancy_probability_core.npy", "occupancy_soft_proposal.npy"):
        if np.load(path / name, mmap_mode="r").shape != (256, 256, 256):
            raise ValueError(f"{name} must have shape (256, 256, 256)")
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    args = metrics.get("args")
    if not isinstance(args, dict):
        raise ValueError("metrics args are missing")
    expected_args: dict[str, object] = {
        "camera_mode": "ortho",
        "mirror_mode": spec.mirror,
        "tsf_mode": spec.trainer_tsf,
        "tsf4_mode": spec.trainer_tsf4,
        "res": 256,
        "export_res": 256,
        "img_size": 256,
        "control_res": 0,
        "proposal_mode": "proposal",
        "view_update_mode": "joint",
        "preserve_target_aspect": True,
        "connectivity_start_iou": 0.60,
        "fidelity_gate": 0.85,
        "structure_warmup_steps": 300,
        "structure_ramp_steps": 300,
        "soft_proposal_min_iou": 0.95,
    }
    for field, expected in expected_args.items():
        actual = args.get(field)
        if type(actual) is not type(expected) or actual != expected:
            label = (
                f"resolution/{field}"
                if field in {"res", "export_res", "img_size"}
                else field
            )
            raise ValueError(
                f"metrics {label} type/value mismatch: {actual!r} != {expected!r}"
            )
    disabled = args.get("disable_losses")
    if type(disabled) is not list or disabled != ["volume"]:
        raise ValueError("metrics disable_losses must equal ['volume']")
    if args.get("no_registration", True) is not True:
        raise ValueError("metrics no_registration must be true")
    target_order = tuple(str(item.resolve()) for item in silhouettes)
    metric_targets = args.get("shadow_files")
    if (
        type(metric_targets) is not list
        or len(metric_targets) != spec.n
        or not all(type(item) is str for item in metric_targets)
    ):
        raise ValueError("metrics shadow_files must be a list of strings")
    metric_views = metrics.get("views")
    if not isinstance(metric_views, list) or len(metric_views) != len(spec.views):
        raise ValueError("metrics camera view count mismatch")

    def angle(row: Mapping[str, object], field: str, index: int) -> float:
        raw = row.get(field)
        if type(raw) not in (int, float) or not math.isfinite(float(raw)):
            raise ValueError(f"metrics view {index} {field} is malformed")
        return float(raw)

    actual_views = []
    for index, row in enumerate(metric_views):
        if not isinstance(row, dict):
            raise ValueError(f"metrics view {index} must be an object")
        if type(row.get("view")) is not int or row["view"] != index:
            raise ValueError(f"metrics view {index} frame mapping mismatch")
        actual_views.append(
            (angle(row, "azimuth_deg", index), angle(row, "elevation_deg", index))
        )
    if tuple(actual_views) != spec.views:
        raise ValueError("metrics camera views mismatch")
    if metrics.get("camera_protocol") != spec.camera_protocol:
        raise ValueError("metrics camera protocol mismatch")
    if (
        type(metrics.get("tsf_direct")) is not bool
        or metrics["tsf_direct"] is not spec.tsf_direct
    ):
        raise ValueError("metrics tsf_direct mismatch")
    zdist = volume_extent = None
    ray_samples = None
    contract_name, contract_status = spec.name, spec.status
    if spec.name == "paper_fixed3_tfs_mr1":
        camera = metrics.get("sadekar_camera")
        if not isinstance(camera, dict):
            raise ValueError("metrics Sadekar camera is missing")

        def real(mapping: Mapping[str, object], field: str):
            try:
                raw = mapping[field]
                if type(raw) not in (int, float) or not math.isfinite(float(raw)):
                    raise TypeError
                return raw
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"metrics Sadekar camera {field} is malformed"
                ) from error

        def integer(mapping: Mapping[str, object], field: str) -> int:
            try:
                raw = mapping[field]
                if type(raw) is not int:
                    raise TypeError
                return raw
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"metrics Sadekar camera {field} is malformed"
                ) from error

        raw_zdist, raw_extent, raw_rays = (
            real(camera, "zdist"),
            real(camera, "volume_extent_world"),
            integer(camera, "n_pts_per_ray"),
        )
        args_zdist, args_extent, args_rays = (
            real(args, "sadekar_zdist"),
            real(args, "sadekar_volume_extent"),
            integer(args, "sadekar_npts_per_ray"),
        )
        if (
            type(raw_zdist) is not type(args_zdist)
            or type(raw_extent) is not type(args_extent)
            or raw_zdist != args_zdist
            or raw_extent != args_extent
            or raw_rays != args_rays
        ):
            raise ValueError("metrics Sadekar camera fields mismatch")
        zdist, volume_extent, ray_samples = (
            float(raw_zdist),
            float(raw_extent),
            raw_rays,
        )
        if (volume_extent, ray_samples) != (1.7, 150):
            raise ValueError("metrics Sadekar camera fields mismatch")
        if zdist == 2.5:
            if not allow_legacy_camera:
                raise ValueError("legacy fixed3 zdist=2.5 requires allow_legacy_camera")
            contract_name, contract_status = "legacy_fixed3_zdist2.5", "legacy"
        elif zdist != 1.7:
            raise ValueError("paper fixed3 zdist must be 1.7")
    targets = tuple(path / f"gt_view_{i}.png" for i in range(len(spec.views)))
    expected_targets = _rebuild_normalized_targets(
        silhouettes, mirror=bool(spec.mirror), size=256
    )
    if len(expected_targets) != len(targets):
        raise ValueError("normalized target count mismatch")
    import cv2

    for index, (target, expected) in enumerate(zip(targets, expected_targets)):
        actual = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
        if actual is None or actual.shape != (256, 256):
            raise ValueError("normalized targets must each have shape 256x256")
        if not np.array_equal(actual, expected):
            raise ValueError(f"normalized target pixel mismatch: gt_view_{index}.png")
    return CameraContract(
        contract_name,
        contract_status,
        spec.requested_tsf,
        spec.effective_tsf,
        spec.mirror,
        spec.trainer_tsf,
        spec.trainer_tsf4,
        spec.views,
        spec.renderer_family,
        spec.direct_view_roles,
        args["img_size"],
        zdist,
        volume_extent,
        ray_samples,
        tuple((str(item.resolve()), _sha256(item)) for item in silhouettes),
        tuple((target.name, _sha256(target)) for target in targets),
        target_order,
        "flip(transpose(zyx,(2,1,0)),axes=(0,2))",
        True,
        True,
        snapshot_artifact_hashes(path),
    )


def _load_delivery_backend():
    """Import Task2 only at the live delivery boundary."""
    from experiments import multiview_contract_delivery_20260902

    return multiview_contract_delivery_20260902


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _json_value(value: object) -> object:
    return json.loads(json.dumps(value, default=_json_default, sort_keys=True))


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                default=_json_default,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _protocol_validation(protocol_id: str, status: str) -> dict[str, object]:
    return {
        "status": "UNVALIDATED_PROTOCOL",
        "supported": False,
        "protocol_id": protocol_id,
        "protocol_status": status,
        "validation_record": None,
        "reason": "no real end-to-end validation record is registered",
    }


def _base_manifest(
    *,
    status: str,
    output: Path,
    spec: ProtocolSpec | None,
    soft_artifact: Path | None,
    contract: CameraContract | None = None,
) -> dict[str, object]:
    effective_id = contract.protocol_id if contract is not None else None
    effective_status = contract.status if contract is not None else None
    payload: dict[str, object] = {
        "schema": "pcraft_contract_multiview_export_v3",
        "status": status,
        "mode": "shadowart_geometry_only",
        "identity": "shadowart_geometry_only",
        "structural_status": "NOT_REQUESTED",
        "publication_status": "PUBLISHED" if status == "SUCCESS" else "NOT_PUBLISHED",
        "output": str(output),
        "soft_artifact": None if soft_artifact is None else str(soft_artifact),
        "protocol_validation": (
            None
            if spec is None and contract is None
            else _protocol_validation(
                effective_id if effective_id is not None else spec.name,
                effective_status if effective_status is not None else spec.status,
            )
        ),
    }
    if spec is not None:
        payload["protocol"] = {
            "requested": asdict(spec),
            "effective": (
                None
                if contract is None
                else {"id": contract.protocol_id, "status": contract.status}
            ),
        }
    if contract is not None:
        payload["camera_contract"] = asdict(contract)
    return payload


def _write_failure_manifest(
    output: Path,
    *,
    status: str,
    reason: str,
    spec: ProtocolSpec | None,
    soft_artifact: Path | None,
    details: Mapping[str, object] | None = None,
    staging_output: Path | None = None,
    contract: CameraContract | None = None,
) -> None:
    payload = _base_manifest(
        status=status,
        output=output,
        spec=spec,
        soft_artifact=soft_artifact,
        contract=contract,
    )
    payload["failure"] = {"reason": reason, **dict(details or {})}
    if staging_output is not None:
        payload["staging_output"] = str(staging_output)
    _atomic_write_json(output / TOP_MANIFEST, payload)


def _output_overlaps_artifact(output: Path, artifact: Path) -> bool:
    resolved_output = output.resolve()
    resolved_artifact = artifact.resolve()
    return (
        resolved_output == resolved_artifact
        or resolved_artifact in resolved_output.parents
    )


def _source_hashes(contract: CameraContract) -> dict[str, dict[str, str]]:
    sources = {
        "adapter": Path(__file__).resolve(),
        "task2_backend": BACKEND_PATH,
        "train.py": VOXEL_ROOT / "train.py",
        "renderer_voxel.py": VOXEL_ROOT / "renderer_voxel.py",
    }
    if contract.renderer_family == "sadekar_voxel_tfs_mirror_orthographic":
        sources["sadekar_protocol.py"] = VOXEL_ROOT / "sadekar_protocol.py"
    result = {}
    for label, source in sources.items():
        if not source.is_file():
            raise AdapterFailure(
                "FAILED_SOURCE_PROVENANCE",
                "required source file is missing",
                {"label": label, "path": str(source)},
            )
        result[label] = {"path": str(source), "sha256": _sha256(source)}
    return result


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise AdapterFailure(
            "FAILED_DELIVERY_RESULT", f"backend {label} must be an object"
        )
    return value


def _schema_error(label: str) -> AdapterFailure:
    return AdapterFailure("FAILED_DELIVERY_RESULT", f"backend {label} is malformed")


def _exact_object(value: object, keys: set[str], label: str) -> Mapping[str, object]:
    mapping = _require_mapping(value, label)
    if set(mapping) != keys:
        raise _schema_error(label)
    return mapping


def _exact_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise _schema_error(label)
    return value


def _exact_float(value: object, label: str, *, minimum: float | None = None) -> float:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or (minimum is not None and value < minimum)
    ):
        raise _schema_error(label)
    return value


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise _schema_error(label)
    return value


def _exact_hash(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _schema_error(label)
    return value


def _validate_delivery_config(config: object) -> dict[str, object]:
    expected: dict[str, object] = {
        "core_resolution": 48,
        "delivery_resolution": 256,
        "mean_iou_gate": 0.95,
        "worst_iou_gate": 0.90,
        "minimum_saving": 1e-6,
        "maximum_saving": 0.85,
        "minimum_erosion_survival": 0.08,
        "max_checkpoints": 256,
        "shell_voxels": 2,
        "sdf_pad": 2,
        "sdf_sigma": 0.4,
        "sdf_level": 0.0,
        "max_faces": 2_000_000,
        "max_dimension_mm": 100.0,
    }
    if not is_dataclass(config):
        raise _schema_error("delivery config")
    actual = asdict(config)
    if set(actual) != set(expected):
        raise _schema_error("delivery config")
    for name, wanted in expected.items():
        value = actual[name]
        if type(value) is not type(wanted) or value != wanted:
            raise _schema_error(f"delivery config.{name}")
    return actual


def _validate_metrics(
    value: object, *, view_count: int, label: str
) -> Mapping[str, object]:
    return _validate_metric_record(
        value,
        view_count=view_count,
        label=label,
        include_saving=True,
        require_delivery_gates=True,
    )


def _validate_metric_record(
    value: object,
    *,
    view_count: int,
    label: str,
    include_saving: bool,
    require_delivery_gates: bool,
) -> Mapping[str, object]:
    keys = {
        "mean_iou",
        "worst_iou",
        "view_ious",
        "component_count_6",
        "occupied_voxels",
        "erosion_survival",
        "surface_fraction",
        "surface_fraction_role",
    }
    if include_saving:
        keys.add("saving")
    metrics = _exact_object(
        value,
        keys,
        label,
    )
    mean = _exact_float(metrics["mean_iou"], f"{label}.mean_iou")
    worst = _exact_float(metrics["worst_iou"], f"{label}.worst_iou")
    erosion = _exact_float(metrics["erosion_survival"], f"{label}.erosion_survival")
    surface = _exact_float(metrics["surface_fraction"], f"{label}.surface_fraction")
    saving = (
        _exact_float(metrics["saving"], f"{label}.saving") if include_saving else None
    )
    views = metrics["view_ious"]
    if (
        type(views) is not list
        or len(views) != view_count
        or any(
            type(item) is not float or not math.isfinite(item) or not 0.0 <= item <= 1.0
            for item in views
        )
    ):
        raise _schema_error(f"{label}.view_ious")
    components = _exact_int(
        metrics["component_count_6"], f"{label}.component_count_6", minimum=1
    )
    _exact_int(metrics["occupied_voxels"], f"{label}.occupied_voxels", minimum=1)
    if (
        not 0.0 <= mean <= 1.0
        or not 0.0 <= worst <= mean
        or min(views) != worst
        or abs(sum(views) / len(views) - mean) > 1e-12
        or not 0.0 <= erosion <= 1.0
        or not 0.0 <= surface <= 1.0
        or metrics["surface_fraction_role"] != "diagnostic_only"
        or type(metrics["surface_fraction_role"]) is not str
    ):
        raise _schema_error(label)
    if include_saving and not 0.0 <= saving <= 1.0:
        raise _schema_error(label)
    if require_delivery_gates and (
        mean < 0.95
        or worst < 0.90
        or components != 1
        or erosion < 0.08
        or saving is None
        or not 1e-6 <= saving <= 0.85
    ):
        raise _schema_error(label)
    return metrics


def _validate_shell_audit(value: object, label: str) -> Mapping[str, object]:
    audit = _exact_object(
        value,
        {
            "shell_count",
            "exterior_root_count",
            "cavity_shell_count",
            "nested_material_shell_count",
            "shell_depths",
            "all_watertight",
            "all_winding_consistent",
            "zero_boundary_edges",
            "zero_nonmanifold_edges",
            "cavity_orientations_valid",
            "parts",
        },
        label,
    )
    count = _exact_int(audit["shell_count"], f"{label}.shell_count", minimum=1)
    if _exact_int(audit["exterior_root_count"], f"{label}.exterior_root_count") != 1:
        raise _schema_error(label)
    cavities = _exact_int(
        audit["cavity_shell_count"], f"{label}.cavity_shell_count", minimum=0
    )
    nested = _exact_int(
        audit["nested_material_shell_count"],
        f"{label}.nested_material_shell_count",
        minimum=0,
    )
    depths, parts = audit["shell_depths"], audit["parts"]
    if (
        type(depths) is not list
        or len(depths) != count
        or any(type(depth) is not int or depth < 0 for depth in depths)
        or depths.count(0) != 1
        or sum(depth % 2 for depth in depths) != cavities
        or sum(depth > 0 and depth % 2 == 0 for depth in depths) != nested
        or type(parts) is not list
        or len(parts) != count
    ):
        raise _schema_error(label)
    for field in (
        "all_watertight",
        "all_winding_consistent",
        "zero_boundary_edges",
        "zero_nonmanifold_edges",
        "cavity_orientations_valid",
    ):
        if _exact_bool(audit[field], f"{label}.{field}") is not True:
            raise _schema_error(label)
    for index, part_value in enumerate(parts):
        part = _exact_object(
            part_value,
            {"faces", "vertices", "depth", "signed_volume"},
            f"{label}.parts[{index}]",
        )
        _exact_int(part["faces"], f"{label}.parts.faces", minimum=1)
        _exact_int(part["vertices"], f"{label}.parts.vertices", minimum=1)
        depth = _exact_int(part["depth"], f"{label}.parts.depth", minimum=0)
        volume = _exact_float(part["signed_volume"], f"{label}.parts.signed_volume")
        if depth != depths[index] or (volume > 0.0) != (depth % 2 == 0):
            raise _schema_error(label)
    return audit


def _validate_reload(
    value: object, *, label: str, expected_faces: int | None = None
) -> Mapping[str, object]:
    reload = _exact_object(
        value,
        {
            "vertices",
            "faces",
            "watertight",
            "winding_consistent",
            "extents",
            "shell_audit",
        },
        label,
    )
    _exact_int(reload["vertices"], f"{label}.vertices", minimum=1)
    faces = _exact_int(reload["faces"], f"{label}.faces", minimum=1)
    if expected_faces is not None and faces != expected_faces:
        raise _schema_error(label)
    if (
        _exact_bool(reload["watertight"], f"{label}.watertight") is not True
        or _exact_bool(reload["winding_consistent"], f"{label}.winding_consistent")
        is not True
    ):
        raise _schema_error(label)
    extents = reload["extents"]
    if (
        type(extents) is not list
        or len(extents) != 3
        or any(
            type(item) is not float or not math.isfinite(item) or item <= 0.0
            for item in extents
        )
        or abs(max(extents) - 100.0) > 1e-3
    ):
        raise _schema_error(f"{label}.extents")
    _validate_shell_audit(reload["shell_audit"], f"{label}.shell_audit")
    return reload


def _validate_serialized_reload(value: object, *, view_count: int, label: str) -> None:
    reload = _exact_object(
        value,
        {"inverse_transform", "shell_audit", "raster_metrics", "raster_shape"},
        label,
    )
    if (
        type(reload["inverse_transform"]) is not str
        or reload["inverse_transform"] != "(display_xyz - translation) / uniform_scale"
        or type(reload["raster_shape"]) is not list
        or reload["raster_shape"] != [256, 256, 256]
        or any(type(item) is not int for item in reload["raster_shape"])
    ):
        raise _schema_error(label)
    _validate_shell_audit(reload["shell_audit"], f"{label}.shell_audit")
    _validate_metrics(
        reload["raster_metrics"], view_count=view_count, label=f"{label}.metrics"
    )


def _validate_stages(value: object) -> None:
    specifications = {
        "probability256": ([256, 256, 256], "float32", "nonzero"),
        "soft_proposal256_read_only": ([256, 256, 256], "bool", "occupied"),
        "core_probability": ([48, 48, 48], "float32", "nonzero"),
        "hull48": ([48, 48, 48], "bool", "occupied"),
        "hull256": ([256, 256, 256], "bool", "occupied"),
        "mesh_raster": ([256, 256, 256], "bool", "occupied"),
        "selected_core": ([48, 48, 48], "bool", "occupied"),
        "premesh_delivery": ([256, 256, 256], "bool", "occupied"),
    }
    stages = _exact_object(value, set(specifications), "stages")
    for name, (shape, dtype, count_name) in specifications.items():
        keys = {"shape", "frame", "dtype", count_name}
        if name == "soft_proposal256_read_only":
            keys.add("mutated")
        stage = _exact_object(stages[name], keys, f"stages.{name}")
        if (
            type(stage["shape"]) is not list
            or stage["shape"] != shape
            or any(type(item) is not int for item in stage["shape"])
            or type(stage["frame"]) is not str
            or stage["frame"] != "canonical_xyz"
            or type(stage["dtype"]) is not str
            or stage["dtype"] != dtype
        ):
            raise _schema_error(f"stages.{name}")
        _exact_int(stage[count_name], f"stages.{name}.{count_name}", minimum=1)
        if name == "soft_proposal256_read_only" and _exact_bool(
            stage["mutated"], "stages.soft_proposal256_read_only.mutated"
        ):
            raise _schema_error("stages.soft_proposal256_read_only")


def _validate_bridge_audit(value: object, label: str) -> None:
    bridge = _exact_object(
        value,
        {"added_voxels", "passes", "component_count_6", "envelope_contained"},
        label,
    )
    _exact_int(bridge["added_voxels"], f"{label}.added_voxels", minimum=0)
    _exact_int(bridge["passes"], f"{label}.passes", minimum=0)
    if _exact_int(bridge["component_count_6"], f"{label}.component_count_6") != 1:
        raise _schema_error(label)
    if (
        _exact_bool(bridge["envelope_contained"], f"{label}.envelope_contained")
        is not True
    ):
        raise _schema_error(label)


def _validate_core_shell_audit(value: object, label: str) -> None:
    shell = _exact_object(
        value,
        {
            "lifted_voxels",
            "shell_added_voxels",
            "local_shell_voxels",
            "ray_restored_voxels",
            "bridge_added_voxels",
            "total_added_voxels",
            "deleted_voxels",
            "delivery_shape",
            "bridge",
        },
        label,
    )
    for name in (
        "lifted_voxels",
        "shell_added_voxels",
        "local_shell_voxels",
        "ray_restored_voxels",
        "bridge_added_voxels",
        "total_added_voxels",
        "deleted_voxels",
    ):
        _exact_int(shell[name], f"{label}.{name}", minimum=0)
    if (
        shell["shell_added_voxels"] != shell["local_shell_voxels"]
        or shell["deleted_voxels"] != 0
        or type(shell["delivery_shape"]) is not list
        or shell["delivery_shape"] != [256, 256, 256]
        or any(type(item) is not int for item in shell["delivery_shape"])
    ):
        raise _schema_error(label)
    if shell["bridge"] is None:
        if shell["bridge_added_voxels"] != 0:
            raise _schema_error(label)
    else:
        _validate_bridge_audit(shell["bridge"], f"{label}.bridge")
        if shell["bridge_added_voxels"] != shell["bridge"]["added_voxels"]:
            raise _schema_error(label)


def _validate_carve_audit(value: object, *, view_count: int, label: str) -> None:
    carve = _exact_object(
        value,
        {
            "accepted_checkpoints",
            "rollback_reason",
            "checkpoints",
            "carving_shape",
        },
        label,
    )
    accepted = _exact_int(
        carve["accepted_checkpoints"], f"{label}.accepted_checkpoints", minimum=0
    )
    if (
        type(carve["rollback_reason"]) is not str
        or carve["rollback_reason"]
        not in {
            "checkpoint_limit",
            "no_simple_boundary_deletions",
            "bridge_failed",
            "core_gate_failed",
        }
        or type(carve["carving_shape"]) is not list
        or carve["carving_shape"] != [48, 48, 48]
        or any(type(item) is not int for item in carve["carving_shape"])
        or type(carve["checkpoints"]) is not list
    ):
        raise _schema_error(label)
    accepted_rows = 0
    for index, checkpoint_value in enumerate(carve["checkpoints"]):
        checkpoint_label = f"{label}.checkpoints[{index}]"
        checkpoint = _require_mapping(checkpoint_value, checkpoint_label)
        is_bridge_failure = set(checkpoint) == {"checkpoint", "accepted", "reason"}
        expected = (
            {"checkpoint", "accepted", "reason"}
            if is_bridge_failure
            else {"checkpoint", "deleted", "accepted", "metrics", "bridge"}
        )
        checkpoint = _exact_object(checkpoint, expected, checkpoint_label)
        _exact_int(
            checkpoint["checkpoint"], f"{checkpoint_label}.checkpoint", minimum=0
        )
        passed = _exact_bool(checkpoint["accepted"], f"{checkpoint_label}.accepted")
        if is_bridge_failure:
            if passed or type(checkpoint["reason"]) is not str:
                raise _schema_error(checkpoint_label)
        else:
            _exact_int(checkpoint["deleted"], f"{checkpoint_label}.deleted", minimum=1)
            _validate_metric_record(
                checkpoint["metrics"],
                view_count=view_count,
                label=f"{checkpoint_label}.metrics",
                include_saving=False,
                require_delivery_gates=False,
            )
            _validate_bridge_audit(checkpoint["bridge"], f"{checkpoint_label}.bridge")
            accepted_rows += int(passed)
    if accepted_rows != accepted:
        raise _schema_error(label)


def _validate_selection_and_trajectories(
    selection_value: object, trajectories_value: object, *, view_count: int
) -> None:
    selection = _exact_object(
        selection_value,
        {
            "absolute_valid_count",
            "near_best_pool_count",
            "selected_threshold",
            "selected_checkpoint",
            "selected_metrics",
        },
        "selection",
    )
    absolute = _exact_int(
        selection["absolute_valid_count"], "selection.absolute_valid_count", minimum=1
    )
    near = _exact_int(
        selection["near_best_pool_count"], "selection.near_best_pool_count", minimum=1
    )
    threshold = _exact_float(
        selection["selected_threshold"], "selection.selected_threshold"
    )
    checkpoint = selection["selected_checkpoint"]
    if (
        not (
            (type(checkpoint) is str and checkpoint == "seed")
            or (type(checkpoint) is int and checkpoint >= 0)
        )
        or near > absolute
    ):
        raise _schema_error("selection")
    _validate_metrics(
        selection["selected_metrics"],
        view_count=view_count,
        label="selection.selected_metrics",
    )
    trajectories = trajectories_value
    if type(trajectories) is not list or not trajectories:
        raise _schema_error("trajectories")
    valid = []
    for index, row_value in enumerate(trajectories):
        label = f"trajectories[{index}]"
        row = _require_mapping(row_value, label)
        status = row.get("status")
        if type(status) is not str:
            raise _schema_error(label)
        schemas = {
            "empty_seed": {"threshold", "status"},
            "seed_bridge_failed": {"threshold", "status", "reason"},
            "seed_core_gate_failed": {"threshold", "status", "metrics"},
            "absolute_gate_failed": {
                "threshold",
                "checkpoint",
                "status",
                "core_shell",
                "delivery_metrics",
                "seed_bridge",
                "carve",
            },
            "absolute_valid": {
                "threshold",
                "checkpoint",
                "status",
                "core_shell",
                "delivery_metrics",
                "seed_bridge",
                "carve",
            },
        }
        if status not in schemas:
            raise _schema_error(f"{label}.status")
        row = _exact_object(row, schemas[status], label)
        row_threshold = _exact_float(row["threshold"], f"{label}.threshold")
        if not 0.0 <= row_threshold <= 1.0:
            raise _schema_error(f"{label}.threshold")
        if status == "seed_bridge_failed":
            if type(row["reason"]) is not str or not row["reason"]:
                raise _schema_error(label)
        elif status == "seed_core_gate_failed":
            _validate_metric_record(
                row["metrics"],
                view_count=view_count,
                label=f"{label}.metrics",
                include_saving=False,
                require_delivery_gates=False,
            )
        elif status in {"absolute_gate_failed", "absolute_valid"}:
            row_checkpoint = row["checkpoint"]
            if not (
                (type(row_checkpoint) is str and row_checkpoint == "seed")
                or (type(row_checkpoint) is int and row_checkpoint >= 0)
            ):
                raise _schema_error(f"{label}.checkpoint")
            _validate_core_shell_audit(row["core_shell"], f"{label}.core_shell")
            _validate_bridge_audit(row["seed_bridge"], f"{label}.seed_bridge")
            _validate_carve_audit(
                row["carve"], view_count=view_count, label=f"{label}.carve"
            )
            _validate_metric_record(
                row["delivery_metrics"],
                view_count=view_count,
                label=f"{label}.delivery_metrics",
                include_saving=True,
                require_delivery_gates=status == "absolute_valid",
            )
            if status == "absolute_valid":
                valid.append(row)
    selected_metrics = selection["selected_metrics"]
    matching = [
        row
        for row in valid
        if row["threshold"] == threshold and row["checkpoint"] == checkpoint
    ]
    if (
        len(valid) != absolute
        or not matching
        or not any(
            _json_value(row["delivery_metrics"]) == _json_value(selected_metrics)
            for row in matching
        )
    ):
        raise _schema_error("selection/trajectories")


def _validate_gates(value: object) -> None:
    gates = _exact_object(
        value,
        {
            "mean_iou",
            "worst_iou",
            "saving_range",
            "erosion_survival",
            "component_count_6",
            "max_faces",
            "surface_fraction",
        },
        "gates",
    )
    expected = {
        "mean_iou": 0.95,
        "worst_iou": 0.90,
        "erosion_survival": 0.08,
        "component_count_6": 1,
        "max_faces": 2_000_000,
        "surface_fraction": "diagnostic_only",
    }
    for name, wanted in expected.items():
        if type(gates[name]) is not type(wanted) or gates[name] != wanted:
            raise _schema_error(f"gates.{name}")
    saving = gates["saving_range"]
    if (
        type(saving) is not list
        or len(saving) != 2
        or any(type(item) is not float for item in saving)
        or saving != [1e-6, 0.85]
    ):
        raise _schema_error("gates.saving_range")


def _validate_mesh_schema(value: object, *, view_count: int) -> Mapping[str, object]:
    mesh = _exact_object(
        value,
        {
            "pre_cap_shell_audit",
            "triangle_cap",
            "post_cap_shell_audit",
            "reraster_metrics",
            "source_vertices_faces_sha256",
            "serialized_reload_contract_qa",
            "exports",
        },
        "mesh",
    )
    _validate_shell_audit(mesh["pre_cap_shell_audit"], "mesh.pre_cap_shell_audit")
    _validate_shell_audit(mesh["post_cap_shell_audit"], "mesh.post_cap_shell_audit")
    cap = _exact_object(
        mesh["triangle_cap"],
        {"simplification_applied", "source_faces", "result_faces", "max_faces"},
        "mesh.triangle_cap",
    )
    applied = _exact_bool(
        cap["simplification_applied"], "mesh.triangle_cap.simplification_applied"
    )
    source_faces = _exact_int(
        cap["source_faces"], "mesh.triangle_cap.source_faces", minimum=1
    )
    result_faces = _exact_int(
        cap["result_faces"], "mesh.triangle_cap.result_faces", minimum=1
    )
    if (
        _exact_int(cap["max_faces"], "mesh.triangle_cap.max_faces") != 2_000_000
        or result_faces > 2_000_000
        or result_faces > source_faces
        or (not applied and result_faces != source_faces)
    ):
        raise _schema_error("mesh.triangle_cap")
    _validate_metrics(
        mesh["reraster_metrics"],
        view_count=view_count,
        label="mesh.reraster_metrics",
    )
    mesh_hash = _exact_hash(
        mesh["source_vertices_faces_sha256"],
        "mesh.source_vertices_faces_sha256",
    )
    reload_qa = _exact_object(
        mesh["serialized_reload_contract_qa"], {"stl", "obj"}, "mesh.reload_qa"
    )
    for format_name in ("stl", "obj"):
        _validate_serialized_reload(
            reload_qa[format_name],
            view_count=view_count,
            label=f"mesh.reload_qa.{format_name}",
        )
    exports = _exact_object(
        mesh["exports"],
        {
            "files",
            "source_vertices_faces_sha256",
            "outer_preview_source_vertices_faces_sha256",
            "source_vertices",
            "source_faces",
            "transform",
            "file_sha256",
            "fabrication_reload",
            "outer_preview_reload",
        },
        "mesh.exports",
    )
    if (
        _exact_hash(exports["source_vertices_faces_sha256"], "mesh.exports.source_hash")
        != mesh_hash
    ):
        raise _schema_error("mesh.exports same-source hash")
    _exact_hash(
        exports["outer_preview_source_vertices_faces_sha256"],
        "mesh.exports.outer_preview_source_hash",
    )
    _exact_int(exports["source_vertices"], "mesh.exports.source_vertices", minimum=1)
    export_faces = _exact_int(
        exports["source_faces"], "mesh.exports.source_faces", minimum=1
    )
    if export_faces != result_faces:
        raise _schema_error("mesh.exports.source_faces")
    transform = _exact_object(
        exports["transform"],
        {"uniform_scale", "translation", "inverse"},
        "mesh.exports.transform",
    )
    _exact_float(
        transform["uniform_scale"], "mesh.exports.transform.uniform_scale", minimum=0.0
    )
    if transform["uniform_scale"] <= 0.0:
        raise _schema_error("mesh.exports.transform.uniform_scale")
    translation = transform["translation"]
    if (
        type(translation) is not list
        or len(translation) != 3
        or any(
            type(item) is not float or not math.isfinite(item) for item in translation
        )
        or type(transform["inverse"]) is not str
        or transform["inverse"] != "(display_xyz - translation) / uniform_scale"
    ):
        raise _schema_error("mesh.exports.transform")
    fabrication = _exact_object(
        exports["fabrication_reload"], {"stl", "obj"}, "fabrication_reload"
    )
    preview = _exact_object(
        exports["outer_preview_reload"], {"stl", "obj"}, "outer_preview_reload"
    )
    for format_name in ("stl", "obj"):
        _validate_reload(
            fabrication[format_name],
            label=f"fabrication_reload.{format_name}",
            expected_faces=export_faces,
        )
    preview_stl = _validate_reload(preview["stl"], label="outer_preview_reload.stl")
    _validate_reload(
        preview["obj"],
        label="outer_preview_reload.obj",
        expected_faces=preview_stl["faces"],
    )
    return exports


@dataclass(frozen=True)
class CandidateBinding:
    path: Path
    device: int
    inode: int
    recorded_sha256: str


@dataclass(frozen=True)
class PublishedBinding:
    path: Path
    device: int
    inode: int
    file_type: int


def _hash_regular_inode(
    path: Path, *, expected_device: int | None = None, expected_inode: int | None = None
) -> tuple[str, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AdapterFailure(
            "FAILED_DELIVERY_RESULT",
            "candidate is not an openable regular file",
            {"path": str(path), "error": str(error)},
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or (expected_device is not None and before.st_dev != expected_device)
            or (expected_inode is not None and before.st_ino != expected_inode)
        ):
            raise AdapterFailure(
                "FAILED_DELIVERY_RESULT", "candidate inode/type binding changed"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise AdapterFailure(
                "FAILED_DELIVERY_RESULT", "candidate changed while hashing"
            )
        return digest.hexdigest(), after
    finally:
        os.close(descriptor)


def _validate_native_records(top: Mapping[str, object], *, view_count: int) -> None:
    native = _exact_object(top["native_repair"], {
        "algorithm", "paper_method_equivalence", "source_resolution", "max_steps", "seed",
        "saving_reference", "saving_comparable_to_paper", "reference_voxels", "core48_failure",
    }, "native_repair")
    if (native["algorithm"] != "experimental_native256_connected_projection_repair_v1"
            or native["paper_method_equivalence"] is not False
            or native["saving_comparable_to_paper"] is not False
            or native["saving_reference"] != "geometric_center_ray_hull_union_immutable_soft_support"
            or _exact_int(native["source_resolution"], "native.source_resolution") != 256
            or _exact_int(native["max_steps"], "native.max_steps") != 60):
        raise _schema_error("native method identity")
    reference = _exact_int(native["reference_voxels"], "native.reference_voxels", minimum=1)
    seed = _exact_object(native["seed"], {"source_components", "retained_components",
        "dropped_source_voxels", "bridge_added_voxels", "max_components"}, "native.seed")
    for name in seed:
        _exact_int(seed[name], f"native.seed.{name}", minimum=0)
    if (seed["max_components"] != 16 or not 1 <= seed["retained_components"] <= 16
            or seed["retained_components"] > seed["source_components"]):
        raise _schema_error("native seed component counts")
    fallback = native["core48_failure"]
    if fallback is not None:
        failure = _exact_object(fallback, {"status", "reason", "details"}, "native.core48_failure")
        if failure["status"] != "NO_VALID_CANDIDATE":
            raise _schema_error("native fallback reason")
    stages = _exact_object(top["stages"], {"probability256", "soft_proposal256_read_only",
        "reference_envelope256", "selected_native_seed", "premesh_delivery", "mesh_raster"}, "native.stages")
    for name, record in stages.items():
        probability = name == "probability256"
        count = "nonzero" if probability else "occupied"
        keys = {"shape", "frame", "dtype", count}
        if name == "soft_proposal256_read_only":
            keys.add("mutated")
        stage = _exact_object(record, keys, f"native.stages.{name}")
        if (type(stage["shape"]) is not list or stage["shape"] != [256, 256, 256]
                or any(type(x) is not int for x in stage["shape"])
                or stage["frame"] != "canonical_xyz"
                or stage["dtype"] != ("float32" if probability else "bool")
                or _exact_int(stage[count], f"native.stages.{name}.{count}", minimum=1) > 256 ** 3
                or ("mutated" in stage and stage["mutated"] is not False)):
            raise _schema_error(f"native.stages.{name}")
    if stages["reference_envelope256"]["occupied"] != reference:
        raise _schema_error("native reference volume")
    selection = _exact_object(top["selection"], {"selected_metrics"}, "native.selection")
    metrics = _validate_metrics(selection["selected_metrics"], view_count=view_count, label="native.selected")
    if (metrics["occupied_voxels"] != stages["premesh_delivery"]["occupied"]
            or abs(metrics["saving"] - (1 - metrics["occupied_voxels"] / reference)) > 1e-12):
        raise _schema_error("native selected volume/saving")
    history = top["trajectories"]
    if type(history) is not list or not 1 <= len(history) <= 61:
        raise _schema_error("native history")
    last_mean = -1.
    last_step = -2
    for i, value in enumerate(history):
        row = _exact_object(value, {"step", "budget", "edit", "metrics"}, "native checkpoint")
        step = row["step"]
        if type(step) is not int or not last_step < step < 60:
            raise _schema_error("native checkpoint step")
        if ((i == 0 and (step != -1 or row["budget"] != 0 or row["edit"] != "seed"))
                or (i > 0 and (row["budget"] not in (1, 4, 16, 64, 256, 1024)
                                or row["edit"] not in ("joint", "remove", "add")))):
            raise _schema_error("native checkpoint edit")
        checkpoint = _validate_metric_record(row["metrics"], view_count=view_count,
            label="native checkpoint metrics", include_saving=False, require_delivery_gates=False)
        if checkpoint["component_count_6"] != 1 or checkpoint["mean_iou"] <= last_mean:
            raise _schema_error("native checkpoint acceptance")
        last_step, last_mean = step, checkpoint["mean_iou"]
    expected = dict(metrics)
    expected.pop("saving")
    if history[-1]["metrics"] != expected:
        raise _schema_error("native selected checkpoint mismatch")


def _validate_backend_result(
    result: object, *, contract: CameraContract, staging_output: Path
) -> tuple[dict[str, CandidateBinding], dict[str, str], dict[str, str]]:
    native = isinstance(result, dict) and "native_repair" in result
    top = _exact_object(
        result,
        {
            "status",
            "publication_eligible",
            "renderer",
            "mode",
            "structural_status",
            "publication_status",
            "camera_contract",
            "stages",
            "trajectories",
            "selection",
            "gates",
            "mesh",
        } | ({"native_repair"} if native else set()),
        "result",
    )
    renderer = _exact_object(top["renderer"], {"mode", "identity"}, "renderer")
    if (
        type(top.get("status")) is not str
        or top.get("status") != "STAGING_CANDIDATE_READY"
        or _exact_bool(top.get("publication_eligible"), "result.publication_eligible")
        is not True
        or type(renderer.get("mode")) is not str
        or renderer.get("mode") != "production_exact_dispatch"
        or type(renderer.get("identity")) is not str
        or renderer.get("identity") != "builtin_exact_contract_dispatch"
    ):
        raise AdapterFailure(
            "FAILED_DELIVERY_RESULT",
            "backend result is not production-publication eligible",
        )
    if any(
        type(top.get(name)) is not str
        for name in ("mode", "structural_status", "publication_status")
    ) or (
        top.get("mode") != "shadowart_geometry_only"
        or top.get("structural_status") != "NOT_REQUESTED"
        or top.get("publication_status") != "NOT_PUBLISHED"
    ):
        raise AdapterFailure(
            "FAILED_DELIVERY_RESULT", "backend result identity/status mismatch"
        )
    if _json_value(top.get("camera_contract")) != _json_value(asdict(contract)):
        raise AdapterFailure(
            "FAILED_DELIVERY_RESULT", "backend returned a different camera contract"
        )
    if native:
        _validate_native_records(top, view_count=len(contract.views))
    else:
        _validate_stages(top["stages"])
        _validate_selection_and_trajectories(
            top["selection"], top["trajectories"], view_count=len(contract.views)
        )
    _validate_gates(top["gates"])
    exports = _validate_mesh_schema(top["mesh"], view_count=len(contract.views))
    files = _exact_object(
        exports["files"], set(CANONICAL_EXPORTS), "mesh.exports.files"
    )
    hashes = _exact_object(
        exports["file_sha256"], set(CANONICAL_EXPORTS), "mesh.exports.file_sha256"
    )
    roles = set(CANONICAL_EXPORTS)
    if set(files) != roles or set(hashes) != roles:
        raise AdapterFailure(
            "FAILED_DELIVERY_RESULT", "backend export role set mismatch"
        )
    if (
        not staging_output.is_dir()
        or staging_output.is_symlink()
        or staging_output.resolve().parent != staging_output.parent.resolve()
    ):
        raise AdapterFailure("FAILED_DELIVERY_RESULT", "staging directory is invalid")
    stage = staging_output.resolve()
    candidates: dict[str, CandidateBinding] = {}
    recorded_hashes: dict[str, str] = {}
    for role in CANONICAL_EXPORTS:
        raw_name, recorded = files[role], hashes[role]
        if type(raw_name) is not str or not raw_name or Path(raw_name).name != raw_name:
            raise AdapterFailure(
                "FAILED_DELIVERY_RESULT", f"malformed backend export metadata: {role}"
            )
        recorded = _exact_hash(recorded, f"mesh.exports.file_sha256.{role}")
        candidate = staging_output / raw_name
        try:
            candidate_stat = candidate.lstat()
        except OSError as error:
            raise AdapterFailure(
                "FAILED_DELIVERY_RESULT",
                f"backend export is missing: {role}",
                {"error": str(error)},
            ) from error
        if (
            candidate.resolve().parent != stage
            or candidate.is_symlink()
            or not stat.S_ISREG(candidate_stat.st_mode)
        ):
            raise AdapterFailure(
                "FAILED_DELIVERY_RESULT",
                f"backend export is not a staged regular file: {role}",
            )
        actual, bound_stat = _hash_regular_inode(candidate)
        if actual != recorded:
            raise AdapterFailure(
                "FAILED_DELIVERY_RESULT", f"backend export hash mismatch: {role}"
            )
        candidates[role] = CandidateBinding(
            path=candidate,
            device=bound_stat.st_dev,
            inode=bound_stat.st_ino,
            recorded_sha256=recorded,
        )
        recorded_hashes[role] = recorded
    manifest_path = staging_output / STAGING_MANIFEST
    try:
        manifest_stat = manifest_path.lstat()
        manifest_result = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterFailure(
            "FAILED_DELIVERY_RESULT", "backend ready manifest is missing or malformed"
        ) from error
    if (
        manifest_path.is_symlink()
        or not stat.S_ISREG(manifest_stat.st_mode)
        or _json_value(manifest_result) != _json_value(top)
    ):
        raise AdapterFailure(
            "FAILED_DELIVERY_RESULT", "backend return and ready manifest differ"
        )
    return (
        candidates,
        recorded_hashes,
        {"name": manifest_path.name, "sha256": _sha256(manifest_path)},
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _binding_matches(path: Path, binding: CandidateBinding) -> bool:
    try:
        current = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(current.st_mode)
        and not path.is_symlink()
        and current.st_dev == binding.device
        and current.st_ino == binding.inode
    )


def _capture_published_binding(path: Path) -> PublishedBinding:
    current = path.lstat()
    return PublishedBinding(
        path=path,
        device=current.st_dev,
        inode=current.st_ino,
        file_type=stat.S_IFMT(current.st_mode),
    )


def _published_binding_matches(binding: PublishedBinding) -> bool:
    try:
        current = binding.path.lstat()
    except OSError:
        return False
    return (
        current.st_dev == binding.device
        and current.st_ino == binding.inode
        and stat.S_IFMT(current.st_mode) == binding.file_type
    )


def _rollback_publication(
    created: Sequence[PublishedBinding],
) -> list[str]:
    errors = []
    for binding in reversed(created):
        if not _published_binding_matches(binding):
            continue
        try:
            binding.path.unlink()
        except Exception as error:
            errors.append(f"{binding.path}: {error}")
    return errors


def _publish(
    *,
    output: Path,
    staging_output: Path,
    candidates: Mapping[str, CandidateBinding],
    recorded_hashes: Mapping[str, str],
    success_manifest: dict[str, object],
    soft_artifact: Path,
    artifact_snapshot: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    created: list[PublishedBinding] = []
    try:
        for role, name in CANONICAL_EXPORTS.items():
            binding, target = candidates[role], output / name
            if _lexists(target):
                raise AdapterFailure(
                    "FAILED_OUTPUT_COLLISION",
                    "canonical output appeared during publication",
                    {"collision": name},
                )
            before_hash, _ = _hash_regular_inode(
                binding.path,
                expected_device=binding.device,
                expected_inode=binding.inode,
            )
            if before_hash != binding.recorded_sha256:
                raise AdapterFailure(
                    "FAILED_DELIVERY_RESULT", "candidate changed before publication"
                )
            try:
                os.link(binding.path, target, follow_symlinks=False)
            except FileExistsError as error:
                raise AdapterFailure(
                    "FAILED_OUTPUT_COLLISION",
                    "canonical output appeared during publication",
                    {"collision": name},
                ) from error
            published = _capture_published_binding(target)
            created.append(published)
            if (
                published.file_type != stat.S_IFREG
                or published.device != binding.device
                or published.inode != binding.inode
            ):
                raise AdapterFailure(
                    "FAILED_DELIVERY_RESULT",
                    "published canonical inode/type differs from validated candidate",
                )
            target_hash, _ = _hash_regular_inode(
                target,
                expected_device=binding.device,
                expected_inode=binding.inode,
            )
            if target_hash != binding.recorded_sha256:
                raise AdapterFailure(
                    "FAILED_DELIVERY_RESULT",
                    "published canonical hash differs from backend record",
                )
            if not _binding_matches(binding.path, binding):
                raise AdapterFailure(
                    "FAILED_DELIVERY_RESULT",
                    "candidate path changed during publication",
                )
        hashes = {}
        for role, canonical_name in CANONICAL_EXPORTS.items():
            binding = candidates[role]
            canonical = output / canonical_name
            if not _binding_matches(canonical, binding):
                raise AdapterFailure(
                    "FAILED_DELIVERY_RESULT",
                    "canonical post-publication regular-file binding failed",
                )
            actual, _ = _hash_regular_inode(
                canonical,
                expected_device=binding.device,
                expected_inode=binding.inode,
            )
            if actual != recorded_hashes[role]:
                raise AdapterFailure(
                    "FAILED_DELIVERY_RESULT", "canonical post-publication hash mismatch"
                )
            hashes[canonical_name] = actual
        try:
            recheck_artifact_hashes(soft_artifact, artifact_snapshot)
        except Exception as error:
            raise AdapterFailure(
                "FAILED_ARTIFACT_MUTATION",
                "soft artifact mutated after publication and before commit",
                {"error": str(error)},
            ) from error
        success_manifest["canonical_sha256"] = hashes
        success_manifest["staging_candidate_cleanup"] = {
            "status": "RETAINED",
            "reason": "validated staging hardlinks are never path-unlinked",
        }
        _atomic_write_json(output / TOP_MANIFEST, success_manifest)
        return hashes
    except AdapterFailure as error:
        rollback_errors = _rollback_publication(created)
        if rollback_errors:
            raise AdapterFailure(
                "FAILED_PUBLICATION_ROLLBACK",
                "publication failed and rollback was incomplete",
                {
                    "error": str(error),
                    "original_status": error.status,
                    "rollback_errors": rollback_errors,
                },
            ) from error
        raise
    except Exception as error:
        rollback_errors = _rollback_publication(created)
        if rollback_errors:
            raise AdapterFailure(
                "FAILED_PUBLICATION_ROLLBACK",
                "publication failed and rollback was incomplete",
                {"error": str(error), "rollback_errors": rollback_errors},
            ) from error
        raise AdapterFailure(
            "FAILED_PUBLICATION",
            "publication failed; moved files were returned to staging",
            {"error": str(error), "staging_output": str(staging_output)},
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--silhouettes", nargs="+", required=True, type=Path)
    parser.add_argument("--tsf", required=True, choices=(0, 1), type=int)
    parser.add_argument("--mirror", choices=("auto", "0", "1"), default="auto")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--soft-artifact-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.007)
    parser.add_argument("--resolution", type=int, choices=(256,), default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-legacy-camera", action="store_true")
    parser.add_argument("--delivery-mode", choices=("auto", "core48", "native256"), default="auto",
                        help="auto retains Core48 and may recover with explicitly experimental native256 geometry")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _dry_run_payload(
    *, spec: ProtocolSpec, soft_command: Sequence[str], output: Path, delivery_mode: str = "auto"
) -> dict[str, object]:
    return {
        "protocol": asdict(spec),
        "soft_command": list(soft_command),
        "immutable_handoff": {
            "validator": "validate_soft_artifact",
            "artifact_snapshot": "sha256_every_file_before_and_after_delivery",
            "camera_contract_copy": CONTRACT_MANIFEST,
        },
        "delivery_profile": {
            "requested_delivery_mode": delivery_mode,
            "native_recovery_is_paper_method": False,
            "backend": BACKEND_ENTRYPOINT,
            "renderer": None,
            "accepted_status": "STAGING_CANDIDATE_READY",
            "publication_eligible": True,
            "renderer_mode": "production_exact_dispatch",
            "mode": "shadowart_geometry_only",
            "structural_status": "NOT_REQUESTED",
        },
        "staging_directory_template": str(output / ".contract-delivery-<unique-id>"),
        "staging_filenames": {
            "manifest": STAGING_MANIFEST,
            "fabrication_stl": "candidate_fabrication.stl",
            "fabrication_obj": "candidate_fabrication.obj",
            "outer_preview_stl": "candidate_outer_preview.stl",
            "outer_preview_obj": "candidate_outer_preview.obj",
        },
        "canonical_filenames": CANONICAL_EXPORTS,
        "protocol_validation": _protocol_validation(spec.name, spec.status),
    }


def main() -> int:
    args = _parser().parse_args()
    silhouettes = tuple(path.resolve() for path in args.silhouettes)
    output = args.output.resolve()
    soft_artifact = (
        args.soft_artifact_dir.resolve()
        if args.soft_artifact_dir is not None
        else output / "soft_training"
    )
    spec: ProtocolSpec | None = None
    contract: CameraContract | None = None
    staging_output: Path | None = None
    committed = False
    try:
        mirror = args.mirror if args.mirror == "auto" else int(args.mirror)
        spec = resolve_protocol(n=len(silhouettes), tsf=args.tsf, mirror=mirror)
        soft_command = build_soft_command(
            spec=spec,
            silhouettes=silhouettes,
            output=soft_artifact,
            device=args.device,
            iterations=args.iterations,
            learning_rate=args.lr,
            resolution=args.resolution,
            seed=args.seed,
        )
        if args.dry_run:
            print(
                json.dumps(
                    _dry_run_payload(
                        spec=spec, soft_command=soft_command, output=output, delivery_mode=args.delivery_mode
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if _output_overlaps_artifact(output, soft_artifact):
            print(
                "FAILED_PATH_OVERLAP: output must not equal or descend from soft artifact",
                file=sys.stderr,
            )
            return 1
        collisions = [
            name
            for name in CANONICAL_EXPORTS.values()
            if (output / name).exists() or (output / name).is_symlink()
        ]
        if collisions:
            raise AdapterFailure(
                "FAILED_OUTPUT_COLLISION",
                "canonical output already exists",
                {"collisions": collisions},
            )
        for silhouette in silhouettes:
            if not silhouette.is_file():
                raise AdapterFailure(
                    "FAILED_INPUT",
                    "silhouette does not exist",
                    {"path": str(silhouette)},
                )
        if args.soft_artifact_dir is None:
            subprocess.run(
                soft_command, cwd=VOXEL_ROOT, env=build_soft_environment(), check=True
            )
        try:
            contract = validate_soft_artifact(
                soft_artifact,
                spec=spec,
                resolution=args.resolution,
                silhouettes=silhouettes,
                allow_legacy_camera=args.allow_legacy_camera,
            )
            snapshot = snapshot_artifact_hashes(soft_artifact)
            if snapshot != tuple(contract.artifact_sha256):
                raise AdapterFailure(
                    "FAILED_ARTIFACT_MUTATION",
                    "artifact changed between contract validation and delivery snapshot",
                )
            _atomic_write_json(output / CONTRACT_MANIFEST, asdict(contract))
        except AdapterFailure:
            raise
        except Exception as error:
            raise AdapterFailure(
                "FAILED_CONTRACT",
                "soft artifact contract validation failed",
                {"error": str(error)},
            ) from error
        staging_output = output / f".contract-delivery-{uuid.uuid4().hex}"
        if staging_output.exists():
            raise AdapterFailure(
                "FAILED_STAGING", "unique staging path unexpectedly exists"
            )
        backend = _load_delivery_backend()
        config = backend.DeliveryConfig()
        config_data = _validate_delivery_config(config)
        try:
            result = backend.run_contract_delivery(
                contract=contract,
                soft_artifact=soft_artifact,
                staging_output=staging_output,
                device=args.device,
                config=config,
                renderer=None,
                delivery_mode=args.delivery_mode,
            )
        except Exception as error:
            try:
                recheck_artifact_hashes(soft_artifact, snapshot)
            except Exception as mutation:
                raise AdapterFailure(
                    "FAILED_ARTIFACT_MUTATION",
                    "soft artifact mutated while backend failed",
                    {"error": str(mutation)},
                ) from error
            raise AdapterFailure(
                "FAILED_DELIVERY_BACKEND",
                "delivery backend failed",
                {
                    "backend_status": str(
                        getattr(error, "status", type(error).__name__)
                    ),
                    "backend_reason": str(getattr(error, "reason", error)),
                    "backend_details": getattr(error, "details", {}),
                },
            ) from error
        try:
            recheck_artifact_hashes(soft_artifact, snapshot)
        except Exception as error:
            raise AdapterFailure(
                "FAILED_ARTIFACT_MUTATION",
                "soft artifact mutated during backend delivery",
                {"error": str(error)},
            ) from error
        candidates, recorded_hashes, backend_manifest = _validate_backend_result(
            result, contract=contract, staging_output=staging_output
        )
        try:
            recheck_artifact_hashes(soft_artifact, snapshot)
        except Exception as error:
            raise AdapterFailure(
                "FAILED_ARTIFACT_MUTATION",
                "soft artifact mutated before publication",
                {"error": str(error)},
            ) from error
        success = _base_manifest(
            status="SUCCESS",
            output=output,
            spec=spec,
            soft_artifact=soft_artifact,
            contract=contract,
        )
        success.update(
            {
                "print_mm": 100.0,
                "protocol": {
                    "requested": asdict(spec),
                    "effective": {
                        "id": contract.protocol_id,
                        "status": contract.status,
                    },
                },
                "camera_contract": asdict(contract),
                "target_order": list(contract.target_order),
                "target_sha256": dict(contract.normalized_target_sha256),
                "input_sha256": dict(contract.input_sha256),
                "artifact_snapshot": snapshot,
                "soft_command": soft_command,
                "environment": {
                    "python_executable": sys.executable,
                    "soft_python_executable": str(VOXEL_PYTHON),
                },
                "source_sha256": _source_hashes(contract),
                "staging_output": str(staging_output),
                "delivery": {
                    "config": config_data,
                    "result": result,
                    "gates": result["gates"],
                    "recorded_export_sha256": recorded_hashes,
                    "backend_manifest": backend_manifest,
                },
            }
        )
        if "native_repair" in result:
            native_source = PROJECT_ROOT / "experiments/multiview_native_delivery_20260905.py"
            success["source_sha256"]["native_delivery"] = {
                "path": str(native_source), "sha256": _sha256(native_source),
            }
            success["paper_method_equivalence"] = False
            success["delivery_method"] = result["native_repair"]["algorithm"]
        try:
            recheck_artifact_hashes(soft_artifact, snapshot)
        except Exception as error:
            raise AdapterFailure(
                "FAILED_ARTIFACT_MUTATION",
                "soft artifact mutated during provenance/manifest assembly",
                {"error": str(error)},
            ) from error
        hashes = _publish(
            output=output,
            staging_output=staging_output,
            candidates=candidates,
            recorded_hashes=recorded_hashes,
            success_manifest=success,
            soft_artifact=soft_artifact,
            artifact_snapshot=snapshot,
        )
        committed = True
        print(
            json.dumps(
                {
                    "status": "SUCCESS",
                    "mode": "shadowart_geometry_only",
                    "structural_status": "NOT_REQUESTED",
                    "canonical_sha256": hashes,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except AdapterFailure as failure:
        if committed:
            return 0
        try:
            _write_failure_manifest(
                output,
                status=failure.status,
                reason=failure.reason,
                spec=spec,
                soft_artifact=soft_artifact,
                details=failure.details,
                staging_output=staging_output,
                contract=contract,
            )
        except Exception as manifest_error:
            print(
                f"{failure.status}: {failure.reason}; failed to write manifest: {manifest_error}",
                file=sys.stderr,
            )
            return 2
        print(f"{failure.status}: {failure.reason}", file=sys.stderr)
        return 1
    except Exception as error:
        if committed:
            return 0
        try:
            _write_failure_manifest(
                output,
                status="FAILED_ADAPTER",
                reason="unexpected adapter failure",
                spec=spec,
                soft_artifact=soft_artifact,
                details={"error": str(error)},
                staging_output=staging_output,
                contract=contract,
            )
        except Exception:
            pass
        print(f"FAILED_ADAPTER: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
