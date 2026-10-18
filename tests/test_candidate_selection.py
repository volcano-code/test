import numpy as np

from generation import voxel_ops


def _volume(voxels):
    volume = np.zeros((4, 4, 4), dtype=bool)
    volume.reshape(-1)[:voxels] = True
    return volume


def test_soft_and_geometry_candidates_share_strict_pool():
    candidates = {
        "learned_printable": _volume(30),
        "printable_minimal": _volume(25),
    }
    metrics = {
        "learned_printable": {"mean_iou": 0.982, "occupied_voxels": 30, "component_count": 1},
        "printable_minimal": {"mean_iou": 0.978, "occupied_voxels": 25, "component_count": 1},
    }

    pool, report = voxel_ops.constrain_printable_candidate_pool(
        candidates,
        metrics,
        reference_voxels=64,
        min_mean_iou=0.975,
        min_material_saving=0.40,
        max_material_saving=0.85,
    )

    assert report["mode"] == "strict"
    assert set(pool) == {"learned_printable", "printable_minimal"}
    assert report["candidates"]["learned_printable"]["source_branch"] == "soft"
    assert report["candidates"]["printable_minimal"]["source_branch"] == "geometry"


def test_soft_candidate_can_win_after_shared_gate(monkeypatch):
    candidates = {
        "learned_printable": _volume(30),
        "printable_minimal": _volume(25),
    }
    metrics_by_voxels = {
        30: {"mean_iou": 0.982, "min_iou": 0.976, "mean_recall": 0.99,
             "occupied_voxels": 30, "component_count": 1, "largest_component_ratio": 1.0},
        25: {"mean_iou": 0.978, "min_iou": 0.974, "mean_recall": 0.99,
             "occupied_voxels": 25, "component_count": 1, "largest_component_ratio": 1.0},
    }
    monkeypatch.setattr(
        voxel_ops,
        "evaluate_tfs_volume",
        lambda volume, *_args, **_kwargs: metrics_by_voxels[int(volume.sum())],
    )

    name, _, _ = voxel_ops.choose_best_tfs_candidate(
        candidates,
        silhouettes=[],
        calibration={},
        image_size=4,
    )

    assert name == "learned_printable"


def test_gate_relaxation_is_reported_when_material_range_has_no_candidate():
    candidates = {"learned_printable": _volume(60)}
    metrics = {
        "learned_printable": {"mean_iou": 0.982, "occupied_voxels": 60, "component_count": 1},
    }

    pool, report = voxel_ops.constrain_printable_candidate_pool(
        candidates,
        metrics,
        reference_voxels=64,
        min_mean_iou=0.975,
        min_material_saving=0.40,
        max_material_saving=0.85,
    )

    assert set(pool) == {"learned_printable"}
    assert report["mode"] == "material_range_relaxed"
    assert not report["candidates"]["learned_printable"]["material_ok"]


def test_strict_pool_excludes_failed_watertight_postprocess():
    candidates = {
        "learned_printable": _volume(30),
        "printable_minimal": _volume(25),
    }
    metrics = {
        "learned_printable": {
            "mean_iou": 0.982,
            "occupied_voxels": 30,
            "component_count": 1,
            "printability_ok": False,
        },
        "printable_minimal": {
            "mean_iou": 0.978,
            "occupied_voxels": 25,
            "component_count": 1,
            "printability_ok": True,
        },
    }

    pool, report = voxel_ops.constrain_printable_candidate_pool(
        candidates,
        metrics,
        reference_voxels=64,
        min_mean_iou=0.975,
        min_material_saving=0.40,
        max_material_saving=0.85,
    )

    assert set(pool) == {"printable_minimal"}
    assert report["mode"] == "strict"
    assert not report["candidates"]["learned_printable"]["strict_eligible"]
