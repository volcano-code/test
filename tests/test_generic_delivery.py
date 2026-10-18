from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import generate_binary_structure, label

from shadowforge_pipeline.unified_run import (
    _generic_connected_delivery,
    _generic_render,
    _generic_visual_hull,
    _trainer_zyx_from_canonical_xyz,
    _uses_sadekar_direct_tfs,
    restore_declared_view_rays,
)


def _protocol_args(n_views: int, *, tsf: bool, mr: bool):
    return SimpleNamespace(
        silhouettes=[Path(f"view_{index}.png") for index in range(n_views)],
        tsf=int(tsf),
        mr=int(mr),
        device="cpu",
    )


def _sadekar_soft_info():
    """Minimal persisted metadata emitted by a strict Sadekar soft stage."""
    return {
        "metrics": {
            "tsf_direct": True,
            "camera_protocol": "sadekar_voxel_tfs_mirror_orthographic",
            "sadekar_camera": {
                "zdist": 2.5,
                "volume_extent_world": 1.7,
                "n_pts_per_ray": 32,
            },
        }
    }


def _asymmetric_truth(resolution: int = 12) -> np.ndarray:
    truth = np.zeros((resolution, resolution, resolution), dtype=bool)
    truth[2:9, 3:8, 2:7] = True
    truth[7:11, 6:10, 6:10] = True
    truth[3:6, 1:4, 8:11] = True
    return truth


def _assert_target_coverage(volume, targets, args, soft_info=None):
    renders = _generic_render(volume, args, soft_info)
    assert len(renders) == len(targets)
    for rendered, target in zip(renders, targets):
        # Restoration only writes inside the visual envelope, so every
        # prescribed foreground ray must be present without exterior leakage.
        assert np.all(rendered[target])
        assert not np.any(rendered & ~target)


def test_trainer_frame_roundtrip_preserves_voxels():
    trainer_zyx = np.zeros((7, 8, 9), dtype=np.uint8)
    trainer_zyx[1:5, 2:7, 3:8] = 1
    trainer_zyx[5, 1, 7] = 1
    canonical_xyz = np.flip(
        np.transpose(trainer_zyx, (2, 1, 0)),
        axis=(0, 2),
    )

    restored = _trainer_zyx_from_canonical_xyz(canonical_xyz)

    assert np.array_equal(restored, trainer_zyx)


def test_exact_tfs_renderer_is_enabled_only_for_matching_soft_protocol():
    matching = {
        "metrics": {
            "tsf_direct": True,
            "camera_protocol": "sadekar_voxel_tfs_mirror_orthographic",
        }
    }
    custom = {
        "metrics": {
            "tsf_direct": False,
            "camera_protocol": "custom_orthographic",
        }
    }

    assert _uses_sadekar_direct_tfs(matching)
    assert not _uses_sadekar_direct_tfs(custom)
    assert not _uses_sadekar_direct_tfs(None)


def test_fixed_three_surface_delivery_preserves_exact_soft_camera_protocol():
    pytest.importorskip("pytorch3d")
    import torch
    from sadekar_protocol import SadekarTFSProtocol

    truth_zyx = np.zeros((12, 12, 12), dtype=np.float32)
    truth_zyx[2:10, 3:9, 1:7] = 1.0
    truth_zyx[6:11, 1:5, 7:11] = 1.0
    truth = np.flip(
        np.transpose(truth_zyx, (2, 1, 0)),
        axis=(0, 2),
    )
    args = _protocol_args(3, tsf=True, mr=True)
    matching = {
        "metrics": {
            "tsf_direct": True,
            "camera_protocol": "sadekar_voxel_tfs_mirror_orthographic",
            "sadekar_camera": {
                "zdist": 2.5,
                "volume_extent_world": 1.7,
                "n_pts_per_ray": 32,
            },
        }
    }

    protocol = SadekarTFSProtocol(
        device="cpu",
        image_size=12,
        zdist=2.5,
        volume_extent_world=1.7,
        n_pts_per_ray=32,
    )
    density = torch.from_numpy(truth_zyx)[None, None].float() * 0.999
    expected = [
        protocol.binary_mask(protocol.render(density, index)).numpy()
        for index in range(6)
    ]
    rendered = _generic_render(truth, args, matching)

    assert len(rendered) == 6
    assert all(
        np.array_equal(actual, reference)
        for actual, reference in zip(rendered, expected)
    )


def test_exact_three_view_repair_envelope_never_clips_valid_soft_proposal():
    pytest.importorskip("pytorch3d")
    truth = _asymmetric_truth()
    args = _protocol_args(3, tsf=True, mr=True)
    matching = {
        "metrics": {
            "tsf_direct": True,
            "camera_protocol": "sadekar_voxel_tfs_mirror_orthographic",
            "sadekar_camera": {
                "zdist": 2.5,
                "volume_extent_world": 1.7,
                "n_pts_per_ray": 32,
            },
        }
    }
    targets = _generic_render(truth, args, matching)

    envelope = _generic_visual_hull(
        targets,
        args,
        truth.shape[0],
        soft_info=matching,
        preserve_xyz=truth,
    )

    assert not np.any(truth & ~envelope)
    retained = truth & envelope
    rendered = _generic_render(retained, args, matching)
    assert all(
        np.array_equal(actual, reference)
        for actual, reference in zip(rendered, targets)
    )


def test_two_view_soft_protocol_preserves_trainer_binary_projections():
    pytest.importorskip("pytorch3d")
    from sadekar_protocol import SadekarTFSProtocol

    truth_zyx = np.zeros((12, 12, 12), dtype=np.float32)
    truth_zyx[2:10, 3:9, 1:7] = 1.0
    truth_zyx[6:11, 1:5, 7:11] = 1.0
    truth_xyz = np.flip(
        np.transpose(truth_zyx, (2, 1, 0)),
        axis=(0, 2),
    )
    args = _protocol_args(2, tsf=True, mr=True)
    matching = {
        "metrics": {
            "tsf_direct": True,
            "camera_protocol": "sadekar_voxel_tfs_mirror_orthographic",
            "sadekar_camera": {
                "zdist": 2.5,
                "volume_extent_world": 1.7,
                "n_pts_per_ray": 32,
            },
        }
    }

    protocol = SadekarTFSProtocol(
        device="cpu",
        image_size=12,
        zdist=2.5,
        volume_extent_world=1.7,
        n_pts_per_ray=32,
    )
    density = (
        pytest.importorskip("torch")
        .from_numpy(truth_zyx)[None, None]
        .float()
        * 0.999
    )
    expected = [
        protocol.binary_mask(protocol.render(density, index)).numpy()
        for index in range(4)
    ]

    rendered = _generic_render(truth_xyz, args, matching)

    assert all(
        np.array_equal(actual, reference)
        for actual, reference in zip(rendered, expected)
    )
def test_generic_delivery_connects_detached_shell_fragments():
    core = np.zeros((4, 4, 4), dtype=bool)
    core[1:3, 1:3, 1:3] = True
    hull = np.ones((8, 8, 8), dtype=bool)
    shell = np.zeros_like(hull)
    shell[0, 0, 0] = True
    shell[7, 7, 7] = True

    delivery = _generic_connected_delivery(core, hull, shell, include_shell=True)

    assert delivery[0, 0, 0]
    assert delivery[7, 7, 7]
    assert label(delivery, structure=generate_binary_structure(3, 1))[1] == 1


def test_generic_delivery_without_shell_remains_connected():
    core = np.zeros((4, 4, 4), dtype=bool)
    core[1:3, 1:3, 1:3] = True
    hull = np.ones((8, 8, 8), dtype=bool)

    delivery = _generic_connected_delivery(
        core, hull, np.zeros_like(hull), include_shell=False
    )

    assert label(delivery, structure=generate_binary_structure(3, 1))[1] == 1


def test_generic_delivery_preserves_native_delivery_resolution_geometry():
    geometry = np.zeros((8, 8, 8), dtype=bool)
    geometry[1:7, 2:6, 3:5] = True
    geometry[4, 6, 4] = True
    hull = np.ones_like(geometry)

    delivery = _generic_connected_delivery(
        geometry, hull, np.zeros_like(hull), include_shell=False
    )

    assert np.array_equal(delivery, geometry)


@pytest.mark.parametrize("n_base", [1, 2, 3])
@pytest.mark.parametrize("mirror", [False, True])
def test_tfs_declared_view_restoration_covers_all_protocol_rays(n_base, mirror):
    pytest.importorskip("pytorch3d")
    args = _protocol_args(n_base, tsf=True, mr=mirror)
    truth = _asymmetric_truth()
    soft_info = _sadekar_soft_info()
    targets = _generic_render(truth, args, soft_info)
    hull = _generic_visual_hull(
        targets, args, truth.shape[0], soft_info=soft_info, preserve_xyz=truth
    )
    guidance = np.where(truth, 1.0, 0.1).astype(np.float32)
    # The delivery path retains the converged soft proposal before adding
    # protocol-compliant recovery rays. Test that this preservation survives
    # every declared Sadekar view rather than treating raster backprojection
    # as an exact inverse on a 12^3 toy volume.
    seed = truth.copy()

    restored = restore_declared_view_rays(
        seed, hull, guidance, targets, args, soft_info=soft_info
    )

    _assert_target_coverage(restored, targets, args, soft_info)


@pytest.mark.parametrize("mirror", [False, True])
def test_tfs_three_view_hull_preserves_renderer_axis_convention(mirror):
    pytest.importorskip("pytorch3d")
    args = _protocol_args(3, tsf=True, mr=mirror)
    truth = _asymmetric_truth()
    soft_info = _sadekar_soft_info()
    targets = _generic_render(truth, args, soft_info)

    hull = _generic_visual_hull(
        targets, args, truth.shape[0], soft_info=soft_info, preserve_xyz=truth
    )
    rendered = _generic_render(hull, args, soft_info)

    assert all(np.array_equal(actual, expected) for actual, expected in zip(rendered, targets))


@pytest.mark.parametrize("mirror", [False, True])
def test_generic_arbitrary_view_restoration_improves_declared_coverage(mirror):
    args = _protocol_args(4, tsf=False, mr=mirror)
    truth = _asymmetric_truth()
    targets = _generic_render(truth, args)
    hull = _generic_visual_hull(targets, args, truth.shape[0])
    guidance = np.where(truth, 1.0, 0.1).astype(np.float32)
    seed = truth & (np.indices(truth.shape).sum(axis=0) % 3 == 0)

    before = _generic_render(seed, args)
    restored = restore_declared_view_rays(
        seed, hull, guidance, targets, args
    )
    after = _generic_render(restored, args)

    before_recall = np.mean([
        np.logical_and(pred, target).sum() / max(int(target.sum()), 1)
        for pred, target in zip(before, targets)
    ])
    after_recall = np.mean([
        np.logical_and(pred, target).sum() / max(int(target.sum()), 1)
        for pred, target in zip(after, targets)
    ])
    assert after_recall > before_recall
    assert after_recall >= 0.95
    for rendered, target in zip(after, targets):
        assert not np.any(rendered & ~target)
