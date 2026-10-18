"""Parity contracts for the isolated soft-loss v3 target/rendering protocol."""

from __future__ import annotations

from pathlib import Path
import warnings

import cv2
import numpy as np
import pytest
import torch

from shadowforge_pipeline.soft_voxel_v2 import data as v2_data
from shadowforge_pipeline.soft_voxel_v2.protocol import SadekarTFSProtocol as V2Protocol
from shadowforge_pipeline.soft_voxel_v3 import data as v3_data
from shadowforge_pipeline.soft_voxel_v3.protocol import (
    CAMERA_AZIMUTH_DEG as V3_CAMERA_AZIMUTH_DEG,
    CAMERA_ELEVATION_DEG as V3_CAMERA_ELEVATION_DEG,
    SadekarTFSProtocol as V3Protocol,
)


def _write_three_targets(tmp_path: Path) -> tuple[Path, Path, Path]:
    outputs = []
    for index in range(3):
        image = np.zeros((19, 23), dtype=np.uint8)
        image[2 + index : 13 + index, 4 + index : 15 + index] = 255
        path = tmp_path / f"target_{index}.png"
        assert cv2.imwrite(str(path), image)
        outputs.append(path)
    return tuple(outputs)


def test_v3_target_loading_is_exactly_v2(tmp_path: Path) -> None:
    """Break caught: v3 changes raw-flip/re-normalize target semantics."""

    paths = _write_three_targets(tmp_path)
    old = v2_data.load_targets(paths, True, 256, torch.device("cpu"))
    new = v3_data.load_targets(paths, True, 256, torch.device("cpu"))

    assert torch.equal(new, old)
    assert new.shape == (6, 256, 256)


def test_v3_formal_mask_matches_v2_and_zero_is_background() -> None:
    """Break caught: an empty projection becomes foreground in the v3 mask."""

    prediction = torch.zeros(6, 4, 4)

    assert torch.equal(V3Protocol.binary_mask(prediction), V2Protocol.binary_mask(prediction))
    assert not V3Protocol.binary_mask(prediction).any()


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else []),
    ids=lambda device: device.type,
)
@pytest.mark.filterwarnings(
    "ignore:NDCGridRaysampler is deprecated,:PendingDeprecationWarning:"
    "pytorch3d\\.renderer\\.implicit\\.raysampling"
)
def test_v3_sadekar_cameras_and_render_all_are_exactly_v2(device: torch.device) -> None:
    """Break caught: v3 drifts from the frozen cameras or red-channel rendering."""

    assert V3_CAMERA_AZIMUTH_DEG == (0.0, 90.0, 0.0, 180.0, 270.0, 0.0)
    assert V3_CAMERA_ELEVATION_DEG == (0.0, 0.0, 90.0, 0.0, 0.0, 270.0)
    old = V2Protocol(device, image_size=8, n_pts_per_ray=8)
    new = V3Protocol(device, image_size=8, n_pts_per_ray=8)
    density = torch.linspace(0.01, 0.64, 8**3, device=device).reshape(1, 1, 8, 8, 8)

    assert torch.equal(new.cameras.R, old.cameras.R)
    assert torch.equal(new.cameras.T, old.cameras.T)
    assert torch.equal(new.render_all(density), old.render_all(density))


def test_renderer_warning_filter_does_not_hide_unrelated_warnings() -> None:
    """Break caught: a broad deprecation filter hides unrelated warnings."""

    message = "unrelated soft-loss-v3 pending deprecation"
    with warnings.catch_warnings(record=True) as captured:
        warnings.filterwarnings(
            "ignore",
            message="NDCGridRaysampler is deprecated,",
            category=PendingDeprecationWarning,
            module=r"pytorch3d\.renderer\.implicit\.raysampling",
        )
        try:
            warnings.warn(message, PendingDeprecationWarning)
        except PendingDeprecationWarning:
            warning_raised = True
        else:
            warning_raised = False

    assert warning_raised or [str(item.message) for item in captured] == [message]
