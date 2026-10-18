from __future__ import annotations

import builtins
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from experiments import multiview_export_adapter_20260830 as adapter
from experiments.multiview_export_adapter_20260830 import (
    build_soft_command,
    resolve_protocol,
)


def test_soft_environment_makes_project_generation_importable_from_voxel_root() -> None:
    environment = adapter.build_soft_environment({"PATH": os.environ["PATH"]})

    completed = subprocess.run(
        [
            "/home/lin/anaconda3/envs/shadowart-revisited/bin/python3.10",
            "-c",
            "import generation.voxel_ops; print(generation.voxel_ops.__file__)",
        ],
        cwd=(
            "/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/"
            "shadow-project2/mesh/voxel"
        ),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        completed.stdout.strip() == "/home/lin/xvxinyao/P-CRAFT/generation/voxel_ops.py"
    )


@pytest.mark.parametrize(
    (
        "n",
        "tsf",
        "mirror",
        "name",
        "status",
        "effective_tsf",
        "trainer_tsf",
        "trainer_tsf4",
        "views",
    ),
    [
        (
            2,
            0,
            "auto",
            "dark_knight_two_wall",
            "extension",
            0,
            0,
            0,
            ((0.0, 0.0), (90.0, 0.0)),
        ),
        (
            2,
            1,
            "auto",
            "dark_knight_two_wall",
            "extension",
            0,
            0,
            0,
            ((0.0, 0.0), (90.0, 0.0)),
        ),
        (
            2,
            0,
            0,
            "dark_knight_two_wall",
            "extension",
            0,
            0,
            0,
            ((0.0, 0.0), (90.0, 0.0)),
        ),
        (
            2,
            1,
            0,
            "dark_knight_two_wall",
            "extension",
            0,
            0,
            0,
            ((0.0, 0.0), (90.0, 0.0)),
        ),
        (
            2,
            0,
            1,
            "dark_knight_two_wall_experimental_mr1",
            "extension",
            0,
            0,
            0,
            ((0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (270.0, 0.0)),
        ),
        (
            2,
            1,
            1,
            "dark_knight_two_wall_experimental_mr1",
            "extension",
            0,
            0,
            0,
            ((0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (270.0, 0.0)),
        ),
        (
            3,
            0,
            "auto",
            "generic_n3",
            "extension",
            0,
            0,
            0,
            (
                (0.0, 0.0),
                (60.0, 0.0),
                (120.0, 0.0),
                (180.0, 0.0),
                (240.0, 0.0),
                (300.0, 0.0),
            ),
        ),
        (
            3,
            0,
            1,
            "generic_n3",
            "extension",
            0,
            0,
            0,
            (
                (0.0, 0.0),
                (60.0, 0.0),
                (120.0, 0.0),
                (180.0, 0.0),
                (240.0, 0.0),
                (300.0, 0.0),
            ),
        ),
        (
            3,
            1,
            "auto",
            "paper_fixed3_tfs_mr1",
            "paper",
            1,
            1,
            0,
            (
                (0.0, 0.0),
                (90.0, 0.0),
                (0.0, 90.0),
                (180.0, 0.0),
                (270.0, 0.0),
                (0.0, -90.0),
            ),
        ),
        (
            3,
            1,
            1,
            "paper_fixed3_tfs_mr1",
            "paper",
            1,
            1,
            0,
            (
                (0.0, 0.0),
                (90.0, 0.0),
                (0.0, 90.0),
                (180.0, 0.0),
                (270.0, 0.0),
                (0.0, -90.0),
            ),
        ),
        (
            4,
            0,
            "auto",
            "generic_n4",
            "extension",
            0,
            0,
            0,
            (
                (0.0, 0.0),
                (45.0, 0.0),
                (90.0, 0.0),
                (135.0, 0.0),
                (180.0, 0.0),
                (225.0, 0.0),
                (270.0, 0.0),
                (315.0, 0.0),
            ),
        ),
        (
            4,
            0,
            1,
            "generic_n4",
            "extension",
            0,
            0,
            0,
            (
                (0.0, 0.0),
                (45.0, 0.0),
                (90.0, 0.0),
                (135.0, 0.0),
                (180.0, 0.0),
                (225.0, 0.0),
                (270.0, 0.0),
                (315.0, 0.0),
            ),
        ),
        (
            4,
            1,
            "auto",
            "tsf4_n4",
            "extension",
            1,
            0,
            1,
            (
                (0.0, 0.0),
                (60.0, 0.0),
                (120.0, 0.0),
                (0.0, 90.0),
                (180.0, 0.0),
                (240.0, 0.0),
                (300.0, 0.0),
                (0.0, -90.0),
            ),
        ),
        (
            4,
            1,
            1,
            "tsf4_n4",
            "extension",
            1,
            0,
            1,
            (
                (0.0, 0.0),
                (60.0, 0.0),
                (120.0, 0.0),
                (0.0, 90.0),
                (180.0, 0.0),
                (240.0, 0.0),
                (300.0, 0.0),
                (0.0, -90.0),
            ),
        ),
    ],
)
def test_resolve_protocol_canonicalizes_each_public_route(
    n: int,
    tsf: int,
    mirror: str | int,
    name: str,
    status: str,
    effective_tsf: int,
    trainer_tsf: int,
    trainer_tsf4: int,
    views: tuple[tuple[float, float], ...],
) -> None:
    """Would fail if a route selects the wrong camera protocol or TSF mode."""
    spec = resolve_protocol(n=n, tsf=tsf, mirror=mirror)

    assert (spec.name, spec.status, spec.effective_tsf) == (name, status, effective_tsf)
    assert (spec.trainer_tsf, spec.trainer_tsf4, spec.views) == (
        trainer_tsf,
        trainer_tsf4,
        views,
    )
    assert spec.mirror == (
        0 if n == 2 and mirror == "auto" else int(mirror) if mirror != "auto" else 1
    )
    assert spec.zdist == (1.7 if name == "paper_fixed3_tfs_mr1" else None)


def test_n2_requested_tsf_one_is_recorded_as_a_canonical_alias() -> None:
    """Would fail if a requested n=2 TSF=1 leaks into the trainer protocol."""
    spec = resolve_protocol(n=2, tsf=1, mirror="auto")

    assert spec.requested_tsf == 1
    assert spec.effective_tsf == spec.trainer_tsf == 0
    assert spec.alias_of == "dark_knight_two_wall"


@pytest.mark.parametrize(
    ("n", "tsf", "mirror"),
    [
        (1, 0, "auto"),
        (5, 0, "auto"),
        (2, -1, "auto"),
        (3, 2, "auto"),
        (3, 0, 0),
        (3, 1, 0),
        (4, 0, 0),
        (4, 1, 0),
        (2, 0, 2),
    ],
)
def test_resolve_protocol_rejects_out_of_scope_inputs(
    n: int, tsf: int, mirror: str | int
) -> None:
    with pytest.raises(ValueError):
        resolve_protocol(n=n, tsf=tsf, mirror=mirror)


@pytest.mark.parametrize(
    ("n", "tsf", "mirror"),
    [
        (True, 0, "auto"),
        (2.0, 0, "auto"),
        (2, True, "auto"),
        (2, 1.0, "auto"),
        (2, 0, True),
        (2, 0, False),
        (2, 0, 1.0),
        (2, 0, 0.0),
        (2, 0, "1"),
        (2, 0, "AUTO"),
    ],
)
def test_resolve_protocol_rejects_non_exact_public_api_types(
    n: object,
    tsf: object,
    mirror: object,
) -> None:
    """Would fail if bool/float/string lookalikes crossed the public protocol boundary."""
    with pytest.raises(ValueError):
        resolve_protocol(n=n, tsf=tsf, mirror=mirror)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("n", "tsf", "mirror"),
    [
        (2, 0, "auto"),
        (2, 1, "auto"),
        (3, 0, "auto"),
        (3, 1, "auto"),
        (4, 0, "auto"),
        (4, 1, "auto"),
    ],
)
def test_soft_command_uses_one_resolved_protocol_and_joint_fixed_optimization(
    n: int, tsf: int, mirror: str | int
) -> None:
    """Would fail if the frozen soft curriculum, aspect, or camera pin is omitted."""
    spec = resolve_protocol(n=n, tsf=tsf, mirror=mirror)
    silhouettes = tuple(Path(f"/inputs/view_{index}.png") for index in range(n))
    command = build_soft_command(
        spec=spec,
        silhouettes=silhouettes,
        output=Path("/outputs/soft"),
        device="cuda:0",
        iterations=2000,
        learning_rate=0.007,
        resolution=256,
        seed=42,
    )

    assert command[command.index("-mr") + 1] == str(spec.mirror)
    assert command[command.index("-tsf") + 1] == str(spec.trainer_tsf)
    assert command[command.index("-tsf4") + 1] == str(spec.trainer_tsf4)
    assert command[command.index("--proposal-mode") + 1] == "proposal"
    assert command[command.index("--view-update-mode") + 1] == "joint"
    assert command[command.index("--res") + 1] == "256"
    assert command[command.index("-imsz") + 1] == "256"
    assert command[command.index("-eres") + 1] == "256"
    assert command[command.index("--control-res") + 1] == "0"
    assert "--preserve-target-aspect" in command
    assert command[command.index("--connectivity-start-iou") + 1] == "0.60"
    assert command[command.index("--fidelity-gate") + 1] == "0.85"
    assert command[command.index("--structure-warmup-steps") + 1] == "300"
    assert command[command.index("--structure-ramp-steps") + 1] == "300"
    assert command[command.index("--soft-proposal-min-iou") + 1] == "0.95"
    if spec.zdist is not None:
        assert command[command.index("--sadekar-zdist") + 1] == "1.7"
    shadow_index = command.index("-sdlist") + 1
    assert tuple(command[shadow_index : shadow_index + n]) == tuple(
        map(str, silhouettes)
    )


def test_soft_command_rejects_non_frozen_resolution() -> None:
    """Would fail if callers can bypass the mandated 256-pixel soft contract."""
    with pytest.raises(ValueError, match="256"):
        build_soft_command(
            spec=resolve_protocol(n=2, tsf=0, mirror="auto"),
            silhouettes=(Path("/inputs/a.png"), Path("/inputs/b.png")),
            output=Path("/outputs/soft"),
            device="cpu",
            iterations=1,
            learning_rate=0.01,
            resolution=128,
            seed=42,
        )


def _write_contract_artifact(
    root: Path,
    *,
    spec,
    inputs: tuple[Path, ...],
    zdist: float = 1.7,
    preserve_target_aspect: bool = True,
) -> None:
    """Write a complete frozen 256 artifact that mirrors the trainer's metrics schema."""
    import numpy as np

    views = [
        {"view": index, "azimuth_deg": azimuth, "elevation_deg": elevation}
        for index, (azimuth, elevation) in enumerate(spec.views)
    ]
    metrics = {
        "camera_protocol": (
            "sadekar_voxel_tfs_mirror_orthographic"
            if spec.name == "paper_fixed3_tfs_mr1"
            else "custom_orthographic"
        ),
        "tsf_direct": spec.name == "paper_fixed3_tfs_mr1",
        "args": {
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
            "disable_losses": ["volume"],
            "preserve_target_aspect": preserve_target_aspect,
            "connectivity_start_iou": 0.60,
            "fidelity_gate": 0.85,
            "structure_warmup_steps": 300,
            "structure_ramp_steps": 300,
            "soft_proposal_min_iou": 0.95,
            "sadekar_zdist": zdist,
            "sadekar_volume_extent": 1.7,
            "sadekar_npts_per_ray": 150,
            "shadow_files": [str(path.resolve()) for path in inputs],
        },
        "views": views,
    }
    if spec.name == "paper_fixed3_tfs_mr1":
        metrics["sadekar_camera"] = {
            "zdist": zdist,
            "volume_extent_world": 1.7,
            "n_pts_per_ray": 150,
        }
    (root / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    np.lib.format.open_memmap(
        root / "occupancy_probability_core.npy",
        mode="w+",
        dtype=np.float32,
        shape=(256, 256, 256),
    ).flush()
    np.lib.format.open_memmap(
        root / "occupancy_soft_proposal.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(256, 256, 256),
    ).flush()
    images = [
        cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE) for input_path in inputs
    ]
    normalized = [
        _trainer_reference_normalize(image, size=256, preserve_aspect=True)
        for image in images
    ]
    if spec.mirror:
        normalized.extend(
            _trainer_reference_normalize(
                cv2.flip(image, 1), size=256, preserve_aspect=True
            )
            for image in images
        )
    assert len(normalized) == len(spec.views)
    for index, image in enumerate(normalized):
        assert cv2.imwrite(str(root / f"gt_view_{index}.png"), image)


def _trainer_reference_normalize(
    image: np.ndarray, *, size: int, preserve_aspect: bool
) -> np.ndarray:
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.resize(binary, (size, size), interpolation=cv2.INTER_NEAREST)
    foreground = cv2.findNonZero(binary)
    if foreground is None:
        return binary
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
    return cv2.warpAffine(binary, matrix, (size, size), flags=cv2.INTER_NEAREST)


def _write_rect(path: Path, *, bounds: tuple[int, int, int, int]) -> None:
    _write_gray_rect(path, bounds=bounds, value=255)


def _write_gray_rect(
    path: Path,
    *,
    bounds: tuple[int, int, int, int],
    value: int,
) -> None:
    image = np.zeros((256, 256), dtype=np.uint8)
    left, top, right, bottom = bounds
    image[top:bottom, left:right] = value
    Image.fromarray(image).save(path)


def test_camera_contract_is_immutable_and_binds_real_metrics_targets_and_hashes(
    tmp_path: Path,
) -> None:
    """Would fail if artifact camera, target order, hashes, or transforms are unbound."""
    spec = resolve_protocol(n=3, tsf=1, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(3))
    for path in inputs:
        _write_rect(path, bounds=(3, 4, 11, 12))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)

    contract = adapter.validate_soft_artifact(
        artifact,
        spec=spec,
        resolution=256,
        silhouettes=inputs,
    )

    assert contract.protocol_id == "paper_fixed3_tfs_mr1"
    assert contract.status == "paper"
    assert contract.views == spec.views
    assert contract.zdist == 1.7
    assert contract.target_order == tuple(str(path.resolve()) for path in inputs)
    assert contract.preserve_target_aspect is True
    assert contract.no_registration is True
    assert contract.renderer_family == "sadekar_voxel_tfs_mirror_orthographic"
    assert contract.direct_view_roles == (
        (0, "ud"),
        (2, "tudlr"),
        (1, "udlr"),
        (0, "udlr"),
        (2, "tlr"),
        (1, "ud"),
    )
    assert (
        contract.soft_zyx_to_canonical_xyz == "flip(transpose(zyx,(2,1,0)),axes=(0,2))"
    )
    assert (
        dict(contract.input_sha256)[str(inputs[0].resolve())]
        == hashlib.sha256(inputs[0].read_bytes()).hexdigest()
    )
    with pytest.raises((AttributeError, TypeError)):
        contract.status = "legacy"  # type: ignore[misc]


def test_camera_contract_accepts_relocated_identical_inputs_by_normalized_pixels(
    tmp_path: Path,
) -> None:
    """Catches historical absolute paths being used as relocation authorization."""
    spec = resolve_protocol(n=3, tsf=1, mirror="auto")
    originals = tuple(
        tmp_path / "original" / f"input_{index}.png" for index in range(3)
    )
    originals[0].parent.mkdir()
    for index, path in enumerate(originals):
        image = np.zeros((91 + index, 137 + index), dtype=np.uint8)
        image[10:70, 15 + index : 85 + index] = 255
        image[20:35, 25:40] = 0
        assert cv2.imwrite(str(path), image)
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=originals, zdist=2.5)
    relocated_root = tmp_path / "relocated"
    relocated_root.mkdir()
    relocated = tuple(relocated_root / path.name for path in originals)
    for source, destination in zip(originals, relocated):
        destination.write_bytes(source.read_bytes())

    contract = adapter.validate_soft_artifact(
        artifact,
        spec=spec,
        resolution=256,
        silhouettes=relocated,
        allow_legacy_camera=True,
    )

    assert contract.protocol_id == "legacy_fixed3_zdist2.5"
    assert contract.target_order == tuple(str(path.resolve()) for path in relocated)
    assert tuple(path for path, _ in contract.input_sha256) == contract.target_order
    metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["args"]["shadow_files"] == [
        str(path.resolve()) for path in originals
    ]


def test_camera_contract_rejects_changed_content_at_the_same_input_path(
    tmp_path: Path,
) -> None:
    """Catches path identity authorizing changed silhouette pixels."""
    spec = resolve_protocol(n=2, tsf=0, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(2))
    for index, path in enumerate(inputs):
        image = np.zeros((256, 256), dtype=np.uint8)
        image[40:200, 40:200] = 255
        image[20 + index : 35 + index, 30:50] = 0
        assert cv2.imwrite(str(path), image)
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    changed = cv2.imread(str(inputs[0]), cv2.IMREAD_GRAYSCALE)
    changed[90:120, 100:140] = 0
    assert cv2.imwrite(str(inputs[0]), changed)

    with pytest.raises(ValueError, match="normalized target pixel mismatch"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


def test_camera_contract_rejects_swapped_input_order_by_target_pixels(
    tmp_path: Path,
) -> None:
    """Catches target slots being treated as an unordered content set."""
    spec = resolve_protocol(n=2, tsf=0, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(2))
    for index, path in enumerate(inputs):
        image = np.zeros((256, 256), dtype=np.uint8)
        image[30:220, 25:225] = 255
        image[60:100, 45 + 90 * index : 75 + 90 * index] = 0
        assert cv2.imwrite(str(path), image)
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    metrics["args"]["shadow_files"] = [str(path.resolve()) for path in reversed(inputs)]
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="normalized target pixel mismatch"):
        adapter.validate_soft_artifact(
            artifact,
            spec=spec,
            resolution=256,
            silhouettes=tuple(reversed(inputs)),
        )


def test_camera_contract_rejects_mirrored_target_slot_tampering(
    tmp_path: Path,
) -> None:
    """Catches mirror views that are not raw-flip-then-normalize outputs."""
    spec = resolve_protocol(n=3, tsf=1, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(3))
    for index, path in enumerate(inputs):
        image = np.zeros((256, 256), dtype=np.uint8)
        image[30:210, 20 + 10 * index : 180 + 10 * index] = 255
        image[60:100, 40 + 10 * index : 75 + 10 * index] = 0
        assert cv2.imwrite(str(path), image)
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    (artifact / "gt_view_3.png").write_bytes((artifact / "gt_view_0.png").read_bytes())

    with pytest.raises(ValueError, match="normalized target pixel mismatch"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


@pytest.mark.parametrize("mirror", [False, True])
def test_adapter_normalization_is_pixel_exact_with_external_trainer(
    tmp_path: Path,
    monkeypatch,
    mirror: bool,
) -> None:
    """Catches drift from trainer Otsu/nearest/union-bbox/0.7 affine behavior."""
    raw = np.zeros((83, 151), dtype=np.uint8)
    raw[5:45, 9:37] = 120
    raw[30:78, 91:145] = 240
    candidate = cv2.flip(raw, 1) if mirror else raw
    monkeypatch.syspath_prepend(str(adapter.VOXEL_ROOT))
    trainer_path = adapter.VOXEL_ROOT / "train.py"
    module_spec = importlib.util.spec_from_file_location(
        f"pcraft_external_trainer_{int(mirror)}", trainer_path
    )
    assert module_spec is not None and module_spec.loader is not None
    trainer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(trainer)

    expected = trainer.normalize_target(candidate, 256, True)
    actual = adapter._normalize_target(candidate, size=256, preserve_aspect=True)

    assert np.array_equal(actual, expected)


def test_camera_contract_rejects_legacy_fixed3_unless_opted_in_and_labels_it_legacy(
    tmp_path: Path,
) -> None:
    """Would fail if a 2.5 fixed3 artifact is silently accepted as the paper protocol."""
    spec = resolve_protocol(n=3, tsf=1, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(3))
    for path in inputs:
        _write_rect(path, bounds=(3, 4, 11, 12))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs, zdist=2.5)

    with pytest.raises(ValueError, match="legacy"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )
    legacy = adapter.validate_soft_artifact(
        artifact,
        spec=spec,
        resolution=256,
        silhouettes=inputs,
        allow_legacy_camera=True,
    )
    assert (legacy.protocol_id, legacy.status, legacy.zdist) == (
        "legacy_fixed3_zdist2.5",
        "legacy",
        2.5,
    )


def test_camera_contract_rejects_aspect_and_camera_metric_mismatch(
    tmp_path: Path,
) -> None:
    """Would fail if target stretching or trainer camera flag drift reaches geometry."""
    spec = resolve_protocol(n=2, tsf=0, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(2))
    for path in inputs:
        _write_rect(path, bounds=(2, 3, 12, 8))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    _write_rect(artifact / "gt_view_0.png", bounds=(2, 3, 8, 13))

    with pytest.raises(ValueError, match="normalized target pixel mismatch"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )
    restored = _trainer_reference_normalize(
        cv2.imread(str(inputs[0]), cv2.IMREAD_GRAYSCALE),
        size=256,
        preserve_aspect=True,
    )
    assert cv2.imwrite(str(artifact / "gt_view_0.png"), restored)
    metrics = json.loads((artifact / "metrics.json").read_text())
    metrics["args"]["mirror_mode"] = 1
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="mirror"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


def test_generic_contract_requires_exact_protocol_and_non_direct_marker(
    tmp_path: Path,
) -> None:
    """Would fail if arbitrary generic camera metadata were admitted to geometry."""
    spec = resolve_protocol(n=2, tsf=0, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(2))
    for path in inputs:
        _write_rect(path, bounds=(32, 40, 160, 104))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)

    contract = adapter.validate_soft_artifact(
        artifact,
        spec=spec,
        resolution=256,
        silhouettes=inputs,
    )
    assert contract.renderer_family == "custom_orthographic"
    assert contract.direct_view_roles == ((None, None), (None, None))
    metrics = json.loads((artifact / "metrics.json").read_text())
    metrics["camera_protocol"] = "arbitrary_camera"
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="camera protocol"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )
    metrics["camera_protocol"] = "custom_orthographic"
    metrics["tsf_direct"] = 1
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="tsf_direct"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


def test_contract_requires_frozen_256_resolution_and_target_shape(
    tmp_path: Path,
) -> None:
    """Would fail if a smaller trainer array or normalized target reached the contract."""
    spec = resolve_protocol(n=2, tsf=0, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(2))
    for path in inputs:
        _write_rect(path, bounds=(32, 40, 160, 104))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    metrics = json.loads((artifact / "metrics.json").read_text())
    metrics["args"]["res"] = 128
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="resolution"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )
    metrics["args"]["res"] = 256
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    for target in sorted(artifact.glob("gt_view_*.png")):
        Image.open(target).resize((128, 128)).save(target)
    with pytest.raises(ValueError, match="256"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


def test_contract_requires_exact_normalized_target_file_set(tmp_path: Path) -> None:
    """Would fail if stale or ambiguously numbered gt targets could enter handoff."""
    spec = resolve_protocol(n=2, tsf=0, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(2))
    for path in inputs:
        _write_rect(path, bounds=(32, 40, 160, 104))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    Image.new("L", (256, 256), color=0).save(artifact / "gt_view_stale.png")

    with pytest.raises(ValueError, match="target file set"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


@pytest.mark.parametrize(
    "disable_losses", ["volume", [], ["volume", "edge"], ("volume", "edge")]
)
def test_contract_requires_exactly_the_frozen_volume_disable(
    tmp_path: Path,
    disable_losses: object,
) -> None:
    """Would fail if an omitted, string, or additional disabled loss changed the curriculum."""
    spec = resolve_protocol(n=2, tsf=0, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(2))
    for path in inputs:
        _write_rect(path, bounds=(32, 40, 160, 104))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    metrics = json.loads((artifact / "metrics.json").read_text())
    metrics["args"]["disable_losses"] = disable_losses
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="disable_losses"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


@pytest.mark.parametrize(
    "camera_mutation",
    [
        lambda metrics: metrics.pop("sadekar_camera"),
        lambda metrics: metrics["sadekar_camera"].pop("zdist"),
        lambda metrics: metrics["sadekar_camera"].__setitem__(
            "n_pts_per_ray", "not-an-int"
        ),
    ],
)
def test_contract_rejects_missing_or_malformed_sadekar_camera_fields(
    tmp_path: Path,
    camera_mutation,
) -> None:
    """Would fail if malformed fixed3 camera metrics escaped as TypeError or passed through."""
    spec = resolve_protocol(n=3, tsf=1, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(3))
    for path in inputs:
        _write_rect(path, bounds=(32, 40, 160, 104))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    metrics = json.loads((artifact / "metrics.json").read_text())
    camera_mutation(metrics)
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="Sadekar camera"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


@pytest.mark.parametrize(
    "view_mutation",
    [
        lambda metrics: metrics["views"].__setitem__(0, "not-a-row"),
        lambda metrics: metrics["views"][0].__setitem__("view", 99),
        lambda metrics: metrics["views"][0].__setitem__("view", True),
        lambda metrics: metrics["views"][0].__setitem__("view", 0.0),
        lambda metrics: metrics["views"][0].__setitem__("view", "0"),
        lambda metrics: metrics["views"][0].__setitem__("azimuth_deg", False),
        lambda metrics: metrics["views"][0].__setitem__("azimuth_deg", "0.0"),
        lambda metrics: metrics["views"][0].__setitem__("elevation_deg", "0.0"),
    ],
)
def test_contract_requires_trainer_view_rows_and_exact_frame_mapping(
    tmp_path: Path,
    view_mutation,
) -> None:
    """Would fail if a non-schema view row or reordered frame mapping were accepted."""
    spec = resolve_protocol(n=3, tsf=1, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(3))
    for path in inputs:
        _write_rect(path, bounds=(32, 40, 160, 104))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    metrics = json.loads((artifact / "metrics.json").read_text())
    view_mutation(metrics)
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="view"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


@pytest.mark.parametrize(
    "scalar_mutation",
    [
        lambda metrics: (
            metrics["sadekar_camera"].__setitem__("n_pts_per_ray", 150.9),
            metrics["args"].__setitem__("sadekar_npts_per_ray", 150.9),
        ),
        lambda metrics: (
            metrics["sadekar_camera"].__setitem__("n_pts_per_ray", "150"),
            metrics["args"].__setitem__("sadekar_npts_per_ray", "150"),
        ),
        lambda metrics: (
            metrics["sadekar_camera"].__setitem__("zdist", "1.7"),
            metrics["args"].__setitem__("sadekar_zdist", "1.7"),
        ),
        lambda metrics: (
            metrics["sadekar_camera"].__setitem__("volume_extent_world", float("nan")),
            metrics["args"].__setitem__("sadekar_volume_extent", float("nan")),
        ),
        lambda metrics: (
            metrics["sadekar_camera"].__setitem__("volume_extent_world", float("inf")),
            metrics["args"].__setitem__("sadekar_volume_extent", float("inf")),
        ),
    ],
)
def test_contract_rejects_non_json_or_nonfinite_sadekar_scalars(
    tmp_path: Path,
    scalar_mutation,
) -> None:
    """Would fail if camera scalars were coerced before exact two-locus comparison."""
    spec = resolve_protocol(n=3, tsf=1, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(3))
    for path in inputs:
        _write_rect(path, bounds=(32, 40, 160, 104))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    metrics = json.loads((artifact / "metrics.json").read_text())
    scalar_mutation(metrics)
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match="Sadekar camera"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


def test_contract_uses_trainer_otsu_foreground_for_low_gray_targets(
    tmp_path: Path,
) -> None:
    """Would fail if valid Otsu foreground at gray 100 were treated as empty."""
    spec = resolve_protocol(n=2, tsf=0, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(2))
    for path in inputs:
        _write_gray_rect(path, bounds=(32, 40, 160, 104), value=100)
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)

    assert (
        adapter.validate_soft_artifact(
            artifact,
            spec=spec,
            resolution=256,
            silhouettes=inputs,
        ).protocol_id
        == "dark_knight_two_wall"
    )


def test_normalize_target_matches_trainer_otsu_black_white_polarity() -> None:
    """Would fail if the validator inverted or hard-thresholded trainer target polarity."""
    black = np.zeros((256, 256), dtype=np.uint8)
    assert not np.any(adapter._normalize_target(black, size=256, preserve_aspect=True))
    white_with_black_shape = np.full((256, 256), 255, dtype=np.uint8)
    white_with_black_shape[32:160, 40:104] = 0

    assert np.array_equal(
        adapter._normalize_target(
            white_with_black_shape, size=256, preserve_aspect=True
        ),
        _trainer_reference_normalize(
            white_with_black_shape, size=256, preserve_aspect=True
        ),
    )


def test_contract_defaults_missing_no_registration_but_rejects_explicit_false(
    tmp_path: Path,
) -> None:
    """Would fail if a trainer-absent flag were required or an explicit false slipped through."""
    spec = resolve_protocol(n=2, tsf=0, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(2))
    for path in inputs:
        _write_rect(path, bounds=(32, 40, 160, 104))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)

    assert (
        adapter.validate_soft_artifact(
            artifact,
            spec=spec,
            resolution=256,
            silhouettes=inputs,
        ).no_registration
        is True
    )
    metrics = json.loads((artifact / "metrics.json").read_text())
    metrics["args"]["no_registration"] = False
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="no_registration"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("mirror_mode", True),
        ("tsf_mode", True),
        ("tsf4_mode", False),
        ("res", 256.0),
        ("img_size", 256.0),
        ("control_res", False),
        ("preserve_target_aspect", 1),
        ("structure_warmup_steps", 300.0),
        ("structure_ramp_steps", 300.0),
    ],
)
def test_contract_rejects_reviewer_json_scalar_type_confusions(
    tmp_path: Path,
    field: str,
    wrong_value: object,
) -> None:
    """Would fail if equal Python values hid a frozen metrics JSON type violation."""
    spec = resolve_protocol(n=3, tsf=1, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(3))
    for path in inputs:
        _write_rect(path, bounds=(32, 40, 160, 104))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    metrics = json.loads((artifact / "metrics.json").read_text())
    metrics["args"][field] = wrong_value
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("connectivity_start_iou", "0.60"),
        ("fidelity_gate", "0.85"),
        ("soft_proposal_min_iou", "0.95"),
        ("connectivity_start_iou", 0),
    ],
)
def test_contract_rejects_non_float_curriculum_scalars(
    tmp_path: Path,
    field: str,
    wrong_value: object,
) -> None:
    """Would fail if float-valued frozen curriculum fields accepted ints or strings."""
    spec = resolve_protocol(n=2, tsf=0, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(2))
    for path in inputs:
        _write_rect(path, bounds=(32, 40, 160, 104))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    metrics = json.loads((artifact / "metrics.json").read_text())
    metrics["args"][field] = wrong_value
    (artifact / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


def test_contract_requires_json_list_for_disabled_losses(
    tmp_path: Path, monkeypatch
) -> None:
    """Would fail if a Python tuple bypassed the metrics JSON-list schema."""
    spec = resolve_protocol(n=2, tsf=0, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(2))
    for path in inputs:
        _write_rect(path, bounds=(32, 40, 160, 104))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    decoded = json.loads((artifact / "metrics.json").read_text())
    decoded["args"]["disable_losses"] = ("volume",)
    monkeypatch.setattr(adapter.json, "loads", lambda _: decoded)

    with pytest.raises(ValueError, match="disable_losses"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


def test_contract_requires_json_string_list_for_target_order(
    tmp_path: Path, monkeypatch
) -> None:
    """Would fail if a string container or Path elements were silently coerced to targets."""
    spec = resolve_protocol(n=2, tsf=0, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(2))
    for path in inputs:
        _write_rect(path, bounds=(32, 40, 160, 104))
    artifact = tmp_path / "soft"
    artifact.mkdir()
    _write_contract_artifact(artifact, spec=spec, inputs=inputs)
    decoded = json.loads((artifact / "metrics.json").read_text())
    decoded["args"]["shadow_files"] = decoded["args"]["shadow_files"][0]
    monkeypatch.setattr(adapter.json, "loads", lambda _: decoded)
    with pytest.raises(ValueError, match="shadow_files"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )

    decoded["args"]["shadow_files"] = [
        Path(value) for value in (str(path.resolve()) for path in inputs)
    ]
    with pytest.raises(ValueError, match="shadow_files"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )

    decoded["args"]["shadow_files"] = [str(inputs[0].resolve())]
    with pytest.raises(ValueError, match="shadow_files"):
        adapter.validate_soft_artifact(
            artifact, spec=spec, resolution=256, silhouettes=inputs
        )


def test_artifact_hash_snapshot_recheck_detects_any_handoff_mutation(
    tmp_path: Path,
) -> None:
    """Would fail if the handoff hash API lets mutated soft files pass unnoticed."""
    artifact = tmp_path / "soft"
    artifact.mkdir()
    tracked = artifact / "metrics.json"
    tracked.write_text('{"state": "before"}', encoding="utf-8")

    snapshot = adapter.snapshot_artifact_hashes(artifact)
    assert adapter.recheck_artifact_hashes(artifact, snapshot) is None
    tracked.write_text('{"state": "after"}', encoding="utf-8")
    with pytest.raises(ValueError, match="mutated"):
        adapter.recheck_artifact_hashes(artifact, snapshot)


CANONICAL_EXPORTS = {
    "fabrication_stl": "delivery_postprocessed.stl",
    "fabrication_obj": "delivery_postprocessed.obj",
    "outer_preview_stl": "delivery_outer_preview.stl",
    "outer_preview_obj": "delivery_outer_preview.obj",
}


@dataclass(frozen=True)
class FakeDeliveryConfig:
    core_resolution: int = 48
    delivery_resolution: int = 256
    mean_iou_gate: float = 0.95
    worst_iou_gate: float = 0.90
    minimum_saving: float = 1e-6
    maximum_saving: float = 0.85
    minimum_erosion_survival: float = 0.08
    max_checkpoints: int = 256
    shell_voxels: int = 2
    sdf_pad: int = 2
    sdf_sigma: float = 0.4
    sdf_level: float = 0.0
    max_faces: int = 2_000_000
    max_dimension_mm: float = 100.0


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_handoff(
    tmp_path: Path,
    *,
    n: int = 2,
    tsf: int = 0,
    legacy: bool = False,
) -> tuple[tuple[Path, ...], Path, adapter.ProtocolSpec, adapter.CameraContract]:
    spec = resolve_protocol(n=n, tsf=tsf, mirror="auto")
    inputs = tuple(tmp_path / f"input_{index}.png" for index in range(n))
    for index, path in enumerate(inputs):
        path.write_bytes(f"input-{index}".encode())
    artifact = tmp_path / "soft"
    artifact.mkdir()
    (artifact / "metrics.json").write_text('{"tiny": true}', encoding="utf-8")
    (artifact / "occupancy_probability_core.npy").write_bytes(b"tiny-probability")
    (artifact / "occupancy_soft_proposal.npy").write_bytes(b"tiny-proposal")
    targets = []
    for index in range(len(spec.views)):
        target = artifact / f"gt_view_{index}.png"
        target.write_bytes(f"target-{index}".encode())
        targets.append(target)
    contract = adapter.CameraContract(
        protocol_id="legacy_fixed3_zdist2.5" if legacy else spec.name,
        status="legacy" if legacy else spec.status,
        requested_tsf=spec.requested_tsf,
        effective_tsf=spec.effective_tsf,
        mirror=spec.mirror,
        trainer_tsf=spec.trainer_tsf,
        trainer_tsf4=spec.trainer_tsf4,
        views=spec.views,
        renderer_family=spec.renderer_family,
        direct_view_roles=spec.direct_view_roles,
        image_size=256,
        zdist=2.5 if legacy else spec.zdist,
        volume_extent=1.7 if spec.zdist is not None else None,
        ray_samples=150 if spec.zdist is not None else None,
        input_sha256=tuple((str(path.resolve()), _digest(path)) for path in inputs),
        normalized_target_sha256=tuple((path.name, _digest(path)) for path in targets),
        target_order=tuple(str(path.resolve()) for path in inputs),
        soft_zyx_to_canonical_xyz="flip(transpose(zyx,(2,1,0)),axes=(0,2))",
        preserve_target_aspect=True,
        no_registration=True,
        artifact_sha256=adapter.snapshot_artifact_hashes(artifact),
    )
    return inputs, artifact, spec, contract


def _main_argv(
    inputs: tuple[Path, ...],
    artifact: Path,
    output: Path,
    *,
    tsf: int,
    allow_legacy: bool = False,
) -> list[str]:
    result = [
        "multiview_export_adapter_20260830.py",
        "--silhouettes",
        *(str(path) for path in inputs),
        "--tsf",
        str(tsf),
        "--output",
        str(output),
        "--soft-artifact-dir",
        str(artifact),
        "--device",
        "cpu",
    ]
    if allow_legacy:
        result.append("--allow-legacy-camera")
    return result


def _delivery_metrics(view_count: int) -> dict[str, object]:
    return {
        "mean_iou": 0.97,
        "worst_iou": 0.97,
        "view_ious": [0.97 for _ in range(view_count)],
        "component_count_6": 1,
        "occupied_voxels": 1000,
        "erosion_survival": 0.20,
        "surface_fraction": 0.30,
        "surface_fraction_role": "diagnostic_only",
        "saving": 0.25,
    }


def _shell_audit(*, faces: int = 24, vertices: int = 16) -> dict[str, object]:
    return {
        "shell_count": 1,
        "exterior_root_count": 1,
        "cavity_shell_count": 0,
        "nested_material_shell_count": 0,
        "shell_depths": [0],
        "all_watertight": True,
        "all_winding_consistent": True,
        "zero_boundary_edges": True,
        "zero_nonmanifold_edges": True,
        "cavity_orientations_valid": True,
        "parts": [
            {
                "faces": faces,
                "vertices": vertices,
                "depth": 0,
                "signed_volume": 1.0,
            }
        ],
    }


def _serialized_reload(view_count: int) -> dict[str, object]:
    return {
        "inverse_transform": "(display_xyz - translation) / uniform_scale",
        "shell_audit": _shell_audit(),
        "raster_metrics": _delivery_metrics(view_count),
        "raster_shape": [256, 256, 256],
    }


def _mesh_reload(*, faces: int, vertices: int) -> dict[str, object]:
    return {
        "vertices": vertices,
        "faces": faces,
        "watertight": True,
        "winding_consistent": True,
        "extents": [100.0, 80.0, 60.0],
        "shell_audit": _shell_audit(faces=faces, vertices=vertices),
    }


def _bridge_audit() -> dict[str, object]:
    return {
        "added_voxels": 0,
        "passes": 0,
        "component_count_6": 1,
        "envelope_contained": True,
    }


def _core_shell_audit() -> dict[str, object]:
    return {
        "lifted_voxels": 800,
        "shell_added_voxels": 100,
        "local_shell_voxels": 100,
        "ray_restored_voxels": 50,
        "bridge_added_voxels": 0,
        "total_added_voxels": 150,
        "deleted_voxels": 0,
        "delivery_shape": [256, 256, 256],
        "bridge": None,
    }


def _carve_audit(view_count: int) -> dict[str, object]:
    core_metrics = dict(_delivery_metrics(view_count))
    del core_metrics["saving"]
    return {
        "accepted_checkpoints": 1,
        "rollback_reason": "no_simple_boundary_deletions",
        "checkpoints": [
            {
                "checkpoint": 0,
                "deleted": 10,
                "accepted": True,
                "metrics": core_metrics,
                "bridge": _bridge_audit(),
            }
        ],
        "carving_shape": [48, 48, 48],
    }


def _full_ready_result(
    contract: adapter.CameraContract,
    files: dict[str, str],
    hashes: dict[str, str],
) -> dict[str, object]:
    view_count = len(contract.views)
    metrics = _delivery_metrics(view_count)
    return {
        "status": "STAGING_CANDIDATE_READY",
        "publication_eligible": True,
        "renderer": {
            "mode": "production_exact_dispatch",
            "identity": "builtin_exact_contract_dispatch",
        },
        "mode": "shadowart_geometry_only",
        "structural_status": "NOT_REQUESTED",
        "publication_status": "NOT_PUBLISHED",
        "camera_contract": asdict(contract),
        "stages": {
            "probability256": {
                "shape": [256, 256, 256],
                "frame": "canonical_xyz",
                "dtype": "float32",
                "nonzero": 2000,
            },
            "soft_proposal256_read_only": {
                "shape": [256, 256, 256],
                "frame": "canonical_xyz",
                "dtype": "bool",
                "occupied": 1200,
                "mutated": False,
            },
            "core_probability": {
                "shape": [48, 48, 48],
                "frame": "canonical_xyz",
                "dtype": "float32",
                "nonzero": 500,
            },
            "hull48": {
                "shape": [48, 48, 48],
                "frame": "canonical_xyz",
                "dtype": "bool",
                "occupied": 400,
            },
            "hull256": {
                "shape": [256, 256, 256],
                "frame": "canonical_xyz",
                "dtype": "bool",
                "occupied": 4000,
            },
            "mesh_raster": {
                "shape": [256, 256, 256],
                "frame": "canonical_xyz",
                "dtype": "bool",
                "occupied": 1000,
            },
            "selected_core": {
                "shape": [48, 48, 48],
                "frame": "canonical_xyz",
                "dtype": "bool",
                "occupied": 200,
            },
            "premesh_delivery": {
                "shape": [256, 256, 256],
                "frame": "canonical_xyz",
                "dtype": "bool",
                "occupied": 1000,
            },
        },
        "trajectories": [
            {
                "threshold": 0.25,
                "checkpoint": "seed",
                "status": "absolute_valid",
                "core_shell": _core_shell_audit(),
                "delivery_metrics": metrics,
                "seed_bridge": _bridge_audit(),
                "carve": _carve_audit(view_count),
            }
        ],
        "selection": {
            "absolute_valid_count": 1,
            "near_best_pool_count": 1,
            "selected_threshold": 0.25,
            "selected_checkpoint": "seed",
            "selected_metrics": metrics,
        },
        "gates": {
            "mean_iou": 0.95,
            "worst_iou": 0.90,
            "saving_range": [1e-6, 0.85],
            "erosion_survival": 0.08,
            "component_count_6": 1,
            "max_faces": 2_000_000,
            "surface_fraction": "diagnostic_only",
        },
        "mesh": {
            "pre_cap_shell_audit": _shell_audit(),
            "triangle_cap": {
                "simplification_applied": False,
                "source_faces": 24,
                "result_faces": 24,
                "max_faces": 2_000_000,
            },
            "post_cap_shell_audit": _shell_audit(),
            "reraster_metrics": metrics,
            "source_vertices_faces_sha256": "1" * 64,
            "serialized_reload_contract_qa": {
                "stl": _serialized_reload(view_count),
                "obj": _serialized_reload(view_count),
            },
            "exports": {
                "files": files,
                "source_vertices_faces_sha256": "1" * 64,
                "outer_preview_source_vertices_faces_sha256": "2" * 64,
                "source_vertices": 16,
                "source_faces": 24,
                "transform": {
                    "uniform_scale": 2.0,
                    "translation": [1.0, 2.0, 3.0],
                    "inverse": "(display_xyz - translation) / uniform_scale",
                },
                "file_sha256": hashes,
                "fabrication_reload": {
                    "stl": _mesh_reload(faces=24, vertices=16),
                    "obj": _mesh_reload(faces=24, vertices=16),
                },
                "outer_preview_reload": {
                    "stl": _mesh_reload(faces=12, vertices=8),
                    "obj": _mesh_reload(faces=12, vertices=8),
                },
            },
        },
    }


def _ready_backend(
    captured: dict[str, object],
    *,
    mutate_result=None,
    after_ready=None,
):
    def run_contract_delivery(**kwargs):
        captured.update(kwargs)
        assert kwargs["renderer"] is None
        staging = Path(kwargs["staging_output"])
        assert not staging.exists()
        staging.mkdir(parents=True)
        files: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for role in CANONICAL_EXPORTS:
            candidate = staging / f"candidate_{role}.mesh"
            candidate.write_bytes(f"mesh:{role}".encode())
            files[role] = candidate.name
            hashes[role] = _digest(candidate)
        result = _full_ready_result(kwargs["contract"], files, hashes)
        if mutate_result is not None:
            mutate_result(result, staging)
        (staging / "staging_delivery_manifest.json").write_text(
            json.dumps(result, sort_keys=True), encoding="utf-8"
        )
        if after_ready is not None:
            after_ready(staging)
        return result

    return SimpleNamespace(
        DeliveryConfig=FakeDeliveryConfig,
        run_contract_delivery=run_contract_delivery,
    )


def _install_tiny_boundaries(
    monkeypatch,
    contract: adapter.CameraContract,
    backend: object,
) -> dict[str, object]:
    validation: dict[str, object] = {}

    def validate(path, **kwargs):
        validation["path"] = path
        validation.update(kwargs)
        return contract

    monkeypatch.setattr(adapter, "validate_soft_artifact", validate)
    monkeypatch.setattr(adapter, "_load_delivery_backend", lambda: backend)
    return validation


def _manifest(output: Path) -> dict[str, object]:
    return json.loads((output / "MULTIVIEW_EXPORT.json").read_text(encoding="utf-8"))


def test_main_lazily_invokes_production_backend_and_atomically_publishes_four_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Would fail if main reused the old geometry chain or published before Task2 gates."""
    inputs, artifact, spec, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    captured: dict[str, object] = {}
    validation = _install_tiny_boundaries(
        monkeypatch, contract, _ready_backend(captured)
    )
    before = adapter.snapshot_artifact_hashes(artifact)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() == 0

    assert captured["contract"] is contract
    assert captured["soft_artifact"] == artifact.resolve()
    assert captured["device"] == "cpu"
    assert isinstance(captured["config"], FakeDeliveryConfig)
    assert captured["renderer"] is None
    assert validation["spec"] == spec
    assert adapter.snapshot_artifact_hashes(artifact) == before
    for role, canonical_name in CANONICAL_EXPORTS.items():
        assert (output / canonical_name).read_bytes() == f"mesh:{role}".encode()
    assert not (output / "geometry_input").exists()
    assert not (output / "soft_initial").exists()
    assert not (output / "postprocessed").exists()

    report = _manifest(output)
    assert report["status"] == "SUCCESS"
    assert report["mode"] == "shadowart_geometry_only"
    assert report["structural_status"] == "NOT_REQUESTED"
    assert report["print_mm"] == 100.0
    assert report["camera_contract"] == json.loads(json.dumps(asdict(contract)))
    assert report["protocol"]["requested"] == json.loads(json.dumps(asdict(spec)))
    assert report["protocol"]["effective"]["id"] == spec.name
    assert report["protocol_validation"]["status"] == "UNVALIDATED_PROTOCOL"
    assert report["protocol_validation"]["supported"] is False
    assert report["artifact_snapshot"] == [list(row) for row in before]
    assert report["delivery"]["result"]["renderer"]["mode"] == "production_exact_dispatch"
    assert report["delivery"]["backend_manifest"]["name"] == "staging_delivery_manifest.json"
    assert set(report["canonical_sha256"]) == set(CANONICAL_EXPORTS.values())
    assert report["staging_candidate_cleanup"] == {
        "status": "RETAINED",
        "reason": "validated staging hardlinks are never path-unlinked",
    }
    assert set(report["source_sha256"]) == {
        "adapter", "task2_backend", "train.py", "renderer_voxel.py",
    }
    assert report["environment"]["python_executable"] == sys.executable
    assert json.loads((output / "CAMERA_CONTRACT.json").read_text()) == json.loads(json.dumps(asdict(contract)))


def _as_native_ready(result, staging):
    metrics = result["selection"]["selected_metrics"]
    metrics["saving"] = .75
    result["native_repair"] = {
        "algorithm": "experimental_native256_connected_projection_repair_v1",
        "paper_method_equivalence": False, "source_resolution": 256, "max_steps": 60,
        "seed": {"source_components": 3, "retained_components": 2,
                 "dropped_source_voxels": 1, "bridge_added_voxels": 5, "max_components": 16},
        "saving_reference": "geometric_center_ray_hull_union_immutable_soft_support",
        "saving_comparable_to_paper": False, "reference_voxels": 4000,
        "core48_failure": None,
    }
    stages = result["stages"]
    stages.pop("core_probability")
    stages.pop("hull48")
    stages.pop("selected_core")
    stages["reference_envelope256"] = stages.pop("hull256")
    stages["selected_native_seed"] = dict(stages["premesh_delivery"])
    result["selection"] = {"selected_metrics": metrics}
    trace_metric = dict(metrics)
    trace_metric.pop("saving")
    result["trajectories"] = [{"step": -1, "budget": 0, "edit": "seed", "metrics": trace_metric}]


def test_native_publication_is_explicitly_nonpaper_and_keeps_all_mesh_gates(tmp_path, monkeypatch):
    inputs, artifact, spec, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "native"
    captured = {}
    _install_tiny_boundaries(monkeypatch, contract, _ready_backend(captured, mutate_result=_as_native_ready))
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0)
                        + ["--delivery-mode", "native256"])
    assert adapter.main() == 0
    assert captured["delivery_mode"] == "native256"
    result = _manifest(output)["delivery"]["result"]
    assert result["native_repair"]["paper_method_equivalence"] is False
    assert "selected_core" not in result["stages"]
    assert (output / "delivery_postprocessed.obj").is_file()


@pytest.mark.parametrize("tamper", ["paper_claim", "low_iou", "disconnected", "wrong_resolution"])
def test_native_claim_or_gate_tampering_never_publishes(tmp_path, monkeypatch, tamper):
    inputs, artifact, spec, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "native_fail"
    def mutate(result, staging):
        _as_native_ready(result, staging)
        if tamper == "paper_claim":
            result["native_repair"]["paper_method_equivalence"] = True
        elif tamper == "wrong_resolution":
            result["stages"]["premesh_delivery"]["shape"] = [48, 48, 48]
        elif tamper == "disconnected":
            result["selection"]["selected_metrics"]["component_count_6"] = 2
        else:
            result["mesh"]["reraster_metrics"]["mean_iou"] = .89
    _install_tiny_boundaries(monkeypatch, contract, _ready_backend({}, mutate_result=mutate))
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0)
                        + ["--delivery-mode", "native256"])
    assert adapter.main() == 1
    assert not (output / "delivery_postprocessed.obj").exists()


def test_legacy_validator_contract_is_the_only_contract_passed_to_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Would fail if main reconstructed paper zdist=1.7 after accepting legacy 2.5."""
    inputs, artifact, _, contract = _tiny_handoff(tmp_path, n=3, tsf=1, legacy=True)
    output = tmp_path / "out"
    captured: dict[str, object] = {}
    _install_tiny_boundaries(monkeypatch, contract, _ready_backend(captured))
    monkeypatch.setattr(
        sys,
        "argv",
        _main_argv(inputs, artifact, output, tsf=1, allow_legacy=True),
    )

    assert adapter.main() == 0
    assert captured["contract"] is contract
    assert captured["contract"].zdist == 2.5
    report = _manifest(output)
    assert report["protocol"]["effective"] == {
        "id": "legacy_fixed3_zdist2.5",
        "status": "legacy",
    }
    assert report["protocol_validation"]["protocol_id"] == "legacy_fixed3_zdist2.5"
    assert report["protocol_validation"]["protocol_status"] == "legacy"
    assert report["camera_contract"]["zdist"] == 2.5
    assert "sadekar_protocol.py" in report["source_sha256"]


class FakeDeliveryFailure(RuntimeError):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.details = {"diagnostic": True}


def test_backend_failure_writes_top_failure_manifest_and_no_canonical_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"

    def fail(**kwargs):
        staging = Path(kwargs["staging_output"])
        staging.mkdir(parents=True)
        (staging / "staging_delivery_failure.json").write_text(
            '{"status":"NO_VALID_CANDIDATE"}', encoding="utf-8"
        )
        raise FakeDeliveryFailure("NO_VALID_CANDIDATE", "no candidate")

    backend = SimpleNamespace(
        DeliveryConfig=FakeDeliveryConfig, run_contract_delivery=fail
    )
    _install_tiny_boundaries(monkeypatch, contract, backend)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() != 0
    assert not any((output / name).exists() for name in CANONICAL_EXPORTS.values())
    failure = _manifest(output)
    assert failure["status"] == "FAILED_DELIVERY_BACKEND"
    assert failure["failure"]["backend_status"] == "NO_VALID_CANDIDATE"
    assert failure["structural_status"] == "NOT_REQUESTED"


@pytest.mark.parametrize("backend_raises", [False, True])
def test_artifact_mutation_after_backend_always_overrides_delivery_result(
    tmp_path: Path,
    monkeypatch,
    backend_raises: bool,
) -> None:
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"

    def mutate(_staging: Path) -> None:
        (artifact / "metrics.json").write_text("mutated", encoding="utf-8")
        if backend_raises:
            raise FakeDeliveryFailure("NO_VALID_CANDIDATE", "after mutation")

    if backend_raises:

        def run(**kwargs):
            staging = Path(kwargs["staging_output"])
            staging.mkdir(parents=True)
            mutate(staging)

    else:
        run = _ready_backend({}, after_ready=mutate).run_contract_delivery
    backend = SimpleNamespace(
        DeliveryConfig=FakeDeliveryConfig, run_contract_delivery=run
    )
    _install_tiny_boundaries(monkeypatch, contract, backend)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() != 0
    assert _manifest(output)["status"] == "FAILED_ARTIFACT_MUTATION"
    assert not any((output / name).exists() for name in CANONICAL_EXPORTS.values())


def _mutate_ready_result(case: str):
    def mutate(result: dict[str, object], staging: Path) -> None:
        if case == "status":
            result["status"] = "SUCCESS"
        elif case == "renderer":
            result["renderer"]["mode"] = "injected_test_only"
        elif case == "contract":
            result["camera_contract"]["protocol_id"] = "wrong-contract"
        elif case == "escape":
            result["mesh"]["exports"]["files"]["fabrication_stl"] = "../escape.stl"
        elif case == "hash":
            result["mesh"]["exports"]["file_sha256"]["fabrication_obj"] = "0" * 64
        elif case == "not_regular":
            name = result["mesh"]["exports"]["files"]["outer_preview_obj"]
            path = staging / name
            path.unlink()
            path.mkdir()
        elif case == "stages":
            result["stages"] = {}
        elif case == "trajectories":
            result["trajectories"] = []
        elif case == "selection":
            result["selection"]["absolute_valid_count"] = 0
        elif case == "selection_type":
            result["selection"]["absolute_valid_count"] = True
        elif case == "gates":
            result["gates"] = {}
        elif case == "pre_shell":
            result["mesh"]["pre_cap_shell_audit"] = {}
        elif case == "triangle_cap":
            result["mesh"]["triangle_cap"]["max_faces"] = 2_000_001
        elif case == "post_shell":
            result["mesh"]["post_cap_shell_audit"]["all_watertight"] = False
        elif case == "reraster":
            result["mesh"]["reraster_metrics"]["mean_iou"] = 0.1
        elif case == "mesh_hash":
            result["mesh"]["source_vertices_faces_sha256"] = "bad"
        elif case == "reload_qa":
            result["mesh"]["serialized_reload_contract_qa"] = {}
        elif case == "export_same_source":
            result["mesh"]["exports"]["source_vertices_faces_sha256"] = "3" * 64
        elif case == "export_transform":
            result["mesh"]["exports"]["transform"] = {}
        elif case == "fabrication_reload":
            result["mesh"]["exports"]["fabrication_reload"] = {}
        elif case == "preview_reload":
            result["mesh"]["exports"]["outer_preview_reload"] = {}
        elif case == "renderer_identity":
            result["renderer"]["identity"] = "forged"
        elif case == "unknown_trajectory_status":
            result["trajectories"].append(
                {"threshold": 0.10, "status": "invented_status"}
            )
        elif case == "trajectory_status_schema":
            result["trajectories"].append(
                {
                    "threshold": 0.10,
                    "status": "empty_seed",
                    "reason": "not part of the empty_seed schema",
                }
            )
        elif case == "impossible_nested_count":
            result["mesh"]["pre_cap_shell_audit"]["nested_material_shell_count"] = 1
        elif case == "view_iou_above_one":
            metrics = result["selection"]["selected_metrics"]
            metrics["mean_iou"] = 1.0
            metrics["worst_iou"] = 0.9
            metrics["view_ious"] = [0.9, 1.1]
        elif case == "selected_metrics_disagree":
            selected = dict(result["selection"]["selected_metrics"])
            selected["occupied_voxels"] = 999
            result["selection"]["selected_metrics"] = selected
        else:  # pragma: no cover - test table owns the cases
            raise AssertionError(case)

    return mutate


@pytest.mark.parametrize(
    "case",
    [
        "status",
        "renderer",
        "renderer_identity",
        "contract",
        "escape",
        "hash",
        "not_regular",
        "stages",
        "trajectories",
        "selection",
        "selection_type",
        "gates",
        "pre_shell",
        "triangle_cap",
        "post_shell",
        "reraster",
        "mesh_hash",
        "reload_qa",
        "export_same_source",
        "export_transform",
        "fabrication_reload",
        "preview_reload",
        "unknown_trajectory_status",
        "trajectory_status_schema",
        "impossible_nested_count",
        "view_iou_above_one",
        "selected_metrics_disagree",
    ],
)
def test_malformed_backend_result_path_or_hash_is_rejected_without_publication(
    tmp_path: Path,
    monkeypatch,
    case: str,
) -> None:
    """Would fail if untrusted backend metadata could select arbitrary publication files."""
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    backend = _ready_backend({}, mutate_result=_mutate_ready_result(case))
    _install_tiny_boundaries(monkeypatch, contract, backend)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() != 0
    assert _manifest(output)["status"] == "FAILED_DELIVERY_RESULT"
    assert not any((output / name).exists() for name in CANONICAL_EXPORTS.values())


def test_existing_canonical_collision_fails_before_validation_or_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs, artifact, _, _ = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    stale = output / "delivery_postprocessed.stl"
    stale.write_bytes(b"old-success")
    broken = output / "delivery_outer_preview.obj"
    broken.symlink_to(output / "missing-preview.obj")

    def forbidden(*args, **kwargs):
        raise AssertionError("collision must stop before validation/backend")

    monkeypatch.setattr(adapter, "validate_soft_artifact", forbidden)
    monkeypatch.setattr(adapter, "_load_delivery_backend", forbidden)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() != 0
    assert stale.read_bytes() == b"old-success"
    report = _manifest(output)
    assert report["status"] == "FAILED_OUTPUT_COLLISION"
    assert report["failure"]["collisions"] == [
        "delivery_postprocessed.stl",
        "delivery_outer_preview.obj",
    ]


def test_partial_publication_exception_rolls_every_moved_file_back_to_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    captured: dict[str, object] = {}
    _install_tiny_boundaries(monkeypatch, contract, _ready_backend(captured))
    real_link = adapter.os.link
    failed = False

    def fail_third_candidate(source, destination, *args, **kwargs):
        nonlocal failed
        destination = Path(destination)
        if destination.name == "delivery_outer_preview.stl" and not failed:
            failed = True
            raise OSError("injected publication failure")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(adapter.os, "link", fail_third_candidate)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() != 0
    assert not any((output / name).exists() for name in CANONICAL_EXPORTS.values())
    report = _manifest(output)
    assert report["status"] == "FAILED_PUBLICATION"
    staging = Path(report["staging_output"])
    assert staging.is_dir()
    assert all(
        (staging / f"candidate_{role}.mesh").is_file() for role in CANONICAL_EXPORTS
    )


def test_candidate_hash_change_during_move_rolls_back_all_canonical_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Would fail if a post-validation candidate race could publish unrecorded bytes."""
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    _install_tiny_boundaries(monkeypatch, contract, _ready_backend({}))
    real_link = adapter.os.link
    changed = False

    def change_first_candidate(source, destination, *args, **kwargs):
        nonlocal changed
        source = Path(source)
        destination = Path(destination)
        if destination.name == "delivery_postprocessed.stl" and not changed:
            source.write_bytes(b"changed-after-validation")
            changed = True
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(adapter.os, "link", change_first_candidate)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() != 0
    assert _manifest(output)["status"] == "FAILED_DELIVERY_RESULT"
    assert not any((output / name).exists() for name in CANONICAL_EXPORTS.values())


def test_same_bytes_regular_inode_swap_after_hash_leaves_no_canonical(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Catches cleanup that owns only the pre-link candidate inode."""
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    _install_tiny_boundaries(monkeypatch, contract, _ready_backend({}))
    real_link = adapter.os.link
    swapped = False

    def swap_regular_inode(source, destination, *args, **kwargs):
        nonlocal swapped
        source = Path(source)
        destination = Path(destination)
        if destination.name == "delivery_postprocessed.stl" and not swapped:
            swapped = True
            payload = source.read_bytes()
            held = source.with_name(f"{source.name}.held-original")
            real_link(source, held, follow_symlinks=False)
            source.unlink()
            source.write_bytes(payload)
            assert source.stat().st_ino != held.stat().st_ino
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(adapter.os, "link", swap_regular_inode)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() != 0
    assert _manifest(output)["status"] == "FAILED_DELIVERY_RESULT"
    assert all(
        not os.path.lexists(output / name) for name in CANONICAL_EXPORTS.values()
    )


def test_staging_replacement_after_binding_check_is_never_unlinked(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Catches check-then-unlink of a staging pathname after publication."""
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    captured: dict[str, object] = {}
    _install_tiny_boundaries(monkeypatch, contract, _ready_backend(captured))
    real_matches = adapter._binding_matches
    swapped = False
    replacement = b"concurrent-staging-owner"

    def swap_after_true(path, binding):
        nonlocal swapped
        path = Path(path)
        matches = real_matches(path, binding)
        if matches and not swapped and path.name == "candidate_fabrication_stl.mesh":
            swapped = True
            held = path.with_name(f"{path.name}.held-original")
            path.rename(held)
            path.write_bytes(replacement)
        return matches

    monkeypatch.setattr(adapter, "_binding_matches", swap_after_true)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() == 0
    staging = Path(captured["staging_output"])
    assert (staging / "candidate_fabrication_stl.mesh").read_bytes() == replacement
    assert (staging / "candidate_fabrication_stl.mesh.held-original").is_file()
    assert all((output / name).is_file() for name in CANONICAL_EXPORTS.values())


def test_wrong_exact_delivery_config_type_is_rejected_before_backend_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    backend = _ready_backend({})
    backend.DeliveryConfig = lambda: FakeDeliveryConfig(core_resolution=True)
    _install_tiny_boundaries(monkeypatch, contract, backend)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() != 0
    assert _manifest(output)["status"] == "FAILED_DELIVERY_RESULT"
    assert not any((output / name).exists() for name in CANONICAL_EXPORTS.values())


@pytest.mark.parametrize("collision_kind", ["regular", "broken_symlink"])
def test_canonical_appearing_at_publication_step_is_never_replaced(
    tmp_path: Path,
    monkeypatch,
    collision_kind: str,
) -> None:
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    _install_tiny_boundaries(monkeypatch, contract, _ready_backend({}))
    real_link = adapter.os.link
    injected = False

    def inject_collision(source, destination, *args, **kwargs):
        nonlocal injected
        destination = Path(destination)
        if not injected:
            injected = True
            if collision_kind == "regular":
                destination.write_bytes(b"concurrent-owner")
            else:
                destination.symlink_to(destination.with_name("missing-race-target"))
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(adapter.os, "link", inject_collision)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() != 0
    first = output / "delivery_postprocessed.stl"
    if collision_kind == "regular":
        assert first.read_bytes() == b"concurrent-owner"
    else:
        assert first.is_symlink()
    assert not (output / "delivery_postprocessed.obj").exists()
    assert _manifest(output)["status"] == "FAILED_OUTPUT_COLLISION"


def test_validated_candidate_swapped_to_external_symlink_is_not_published(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    _install_tiny_boundaries(monkeypatch, contract, _ready_backend({}))
    external = tmp_path / "external.mesh"
    external.write_bytes(b"mesh:fabrication_stl")
    real_link = adapter.os.link
    swapped = False

    def swap_before_link(source, destination, *args, **kwargs):
        nonlocal swapped
        source = Path(source)
        if not swapped:
            swapped = True
            source.unlink()
            source.symlink_to(external)
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(adapter.os, "link", swap_before_link)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() != 0
    assert _manifest(output)["status"] == "FAILED_DELIVERY_RESULT"
    assert not any((output / name).exists() for name in CANONICAL_EXPORTS.values())


def test_output_inside_soft_artifact_fails_before_any_adapter_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    before = adapter.snapshot_artifact_hashes(artifact)
    backend = _ready_backend({})
    _install_tiny_boundaries(monkeypatch, contract, backend)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, artifact, tsf=0))

    assert adapter.main() != 0
    assert adapter.snapshot_artifact_hashes(artifact) == before
    assert not (artifact / "CAMERA_CONTRACT.json").exists()
    assert not (artifact / "MULTIVIEW_EXPORT.json").exists()
    assert adapter._output_overlaps_artifact(artifact, artifact) is True
    assert adapter._output_overlaps_artifact(artifact / "nested", artifact) is True
    assert (
        adapter._output_overlaps_artifact(
            tmp_path / "out", tmp_path / "out" / "soft_training"
        )
        is False
    )


@pytest.mark.parametrize("failure_kind", ["backend", "malformed_result"])
def test_legacy_failure_manifest_keeps_full_effective_contract_identity(
    tmp_path: Path,
    monkeypatch,
    failure_kind: str,
) -> None:
    inputs, artifact, spec, contract = _tiny_handoff(tmp_path, n=3, tsf=1, legacy=True)
    output = tmp_path / "out"
    if failure_kind == "backend":

        def fail(**kwargs):
            Path(kwargs["staging_output"]).mkdir(parents=True)
            raise FakeDeliveryFailure("NO_VALID_CANDIDATE", "legacy failure")

        backend = SimpleNamespace(
            DeliveryConfig=FakeDeliveryConfig, run_contract_delivery=fail
        )
    else:
        backend = _ready_backend({}, mutate_result=_mutate_ready_result("gates"))
    _install_tiny_boundaries(monkeypatch, contract, backend)
    monkeypatch.setattr(
        sys,
        "argv",
        _main_argv(inputs, artifact, output, tsf=1, allow_legacy=True),
    )

    assert adapter.main() != 0
    report = _manifest(output)
    assert report["camera_contract"] == json.loads(json.dumps(asdict(contract)))
    assert report["protocol"]["requested"] == json.loads(json.dumps(asdict(spec)))
    assert report["protocol"]["effective"] == {
        "id": "legacy_fixed3_zdist2.5",
        "status": "legacy",
    }
    assert report["protocol_validation"]["protocol_id"] == "legacy_fixed3_zdist2.5"


def test_artifact_mutation_during_provenance_assembly_blocks_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    _install_tiny_boundaries(monkeypatch, contract, _ready_backend({}))
    real_sources = adapter._source_hashes

    def mutate_after_sources(value):
        result = real_sources(value)
        (artifact / "metrics.json").write_text("late mutation", encoding="utf-8")
        return result

    monkeypatch.setattr(adapter, "_source_hashes", mutate_after_sources)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() != 0
    assert _manifest(output)["status"] == "FAILED_ARTIFACT_MUTATION"
    assert not any((output / name).exists() for name in CANONICAL_EXPORTS.values())


def test_artifact_mutation_after_last_move_rolls_back_before_success_marker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    _install_tiny_boundaries(monkeypatch, contract, _ready_backend({}))
    real_link = adapter.os.link
    linked = 0

    def mutate_after_last_link(source, destination, *args, **kwargs):
        nonlocal linked
        result = real_link(source, destination, *args, **kwargs)
        linked += 1
        if linked == 4:
            (artifact / "metrics.json").write_text(
                "post-move mutation", encoding="utf-8"
            )
        return result

    monkeypatch.setattr(adapter.os, "link", mutate_after_last_link)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() != 0
    assert _manifest(output)["status"] == "FAILED_ARTIFACT_MUTATION"
    assert not any((output / name).exists() for name in CANONICAL_EXPORTS.values())


def test_broken_stdout_after_success_commit_does_not_rewrite_success_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs, artifact, _, contract = _tiny_handoff(tmp_path)
    output = tmp_path / "out"
    _install_tiny_boundaries(monkeypatch, contract, _ready_backend({}))
    real_print = builtins.print
    broken = False

    def break_once(*args, **kwargs):
        nonlocal broken
        if not broken:
            broken = True
            raise BrokenPipeError("closed stdout")
        return real_print(*args, **kwargs)

    monkeypatch.setattr(builtins, "print", break_once)
    monkeypatch.setattr(sys, "argv", _main_argv(inputs, artifact, output, tsf=0))

    assert adapter.main() == 0
    assert _manifest(output)["status"] == "SUCCESS"
    assert all((output / name).is_file() for name in CANONICAL_EXPORTS.values())


@pytest.mark.parametrize(("n", "tsf"), [(2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)])
def test_dry_run_six_routes_describes_only_new_backend_and_honest_validation(
    tmp_path: Path,
    monkeypatch,
    capsys,
    n: int,
    tsf: int,
) -> None:
    inputs = tuple(tmp_path / f"view_{index}.png" for index in range(n))
    output = tmp_path / f"out-{n}-{tsf}"
    argv = [
        "multiview_export_adapter_20260830.py",
        "--silhouettes",
        *(str(path) for path in inputs),
        "--tsf",
        str(tsf),
        "--output",
        str(output),
        "--dry-run",
    ]
    monkeypatch.setattr(
        adapter,
        "_load_delivery_backend",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run imported backend")),
    )
    monkeypatch.setattr(sys, "argv", argv)

    assert adapter.main() == 0
    raw = capsys.readouterr().out
    payload = json.loads(raw)
    assert payload["protocol"]["status"] == (
        "paper" if (n, tsf) == (3, 1) else "extension"
    )
    assert payload["delivery_profile"]["backend"].endswith(".run_contract_delivery")
    assert payload["delivery_profile"]["renderer"] is None
    assert payload["delivery_profile"]["requested_delivery_mode"] == "auto"
    assert payload["delivery_profile"]["native_recovery_is_paper_method"] is False
    assert payload["protocol_validation"]["status"] == "UNVALIDATED_PROTOCOL"
    assert set(payload["canonical_filenames"].values()) == set(
        CANONICAL_EXPORTS.values()
    )
    assert "staging_delivery_manifest.json" == payload["staging_filenames"]["manifest"]
    lowered = raw.lower()
    assert "geometry_command" not in lowered
    assert "geometry_input" not in lowered
    assert "unified_run" not in lowered
    assert "build_views" not in lowered
    assert '"free"' not in lowered


def test_dangerous_legacy_geometry_helpers_are_not_callable() -> None:
    """Would fail if the adapter retained a callable route into the rejected pipeline."""
    forbidden = {
        "build_geometry_argv",
        "select_connected_levelset",
        "evaluate_exact_protocol",
        "write_watertight_mesh",
        "_load_isolated_unified_runner",
        "prepare_connected_geometry_artifact",
        "finalize_postprocessed_delivery",
        "_export_obj",
        "_export_connected_soft_levelset",
        "_run_geometry",
    }

    assert forbidden.isdisjoint(vars(adapter))
