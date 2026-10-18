from dataclasses import replace

import numpy as np
import trimesh
import pytest

from experiments import multiview_contract_delivery_20260902 as d
from tests.test_multiview_contract_delivery_20260902 import LiteralContract


def fixed3_contract(resolution=32):
    return replace(
        LiteralContract(), protocol_id="paper_fixed3_tfs_mr1",
        image_size=resolution,
        renderer_family=d.DIRECT,
        views=((0., 0.), (90., 0.), (0., 90.), (180., 0.), (270., 0.), (0., -90.)),
        direct_view_roles=((0, "ud"), (2, "tudlr"), (1, "udlr"),
                           (0, "udlr"), (2, "tlr"), (1, "ud")),
        zdist=1.7, volume_extent=1.7, ray_samples=150,
    )


def test_real_camera_backprojection_does_not_shrink_centered_box():
    # The previous pixel extrusion shrank a physical box a second time by .85.
    volume = np.zeros((32, 32, 32), bool)
    volume[8:24, 8:24, 8:24] = True
    contract = fixed3_contract()
    targets = d.render_contract(volume, contract)
    hull = d.backproject_contract(targets, contract, 32)
    assert (hull & volume).sum() / volume.sum() >= .85
    assert not hull[:5].any()
    assert not hull[27:].any()


def test_generic_renderer_inverts_the_declared_artifact_frame():
    import torch
    # Asymmetric fixture catches the extra X/Z flip hidden by symmetric tests.
    raw = np.zeros((24, 24, 24), bool)
    raw[3:10, 7:16, 13:20] = True
    contract = replace(LiteralContract(), image_size=24,
                       views=((0., 0.), (90., 0.)))
    canonical = d.trainer_zyx_to_canonical_xyz(raw)
    external = d._external("renderer_voxel.py", "generic_frame_reference")
    density = torch.tensor(raw.astype(np.float32)[None, None]) * .999
    expected = [(external.render_orthographic(density, az, el) >= .5).numpy()
                for az, el in contract.views]
    predictions = d.render_contract(canonical, contract)
    assert all(np.array_equal(a, b) for a, b in zip(predictions, expected))
    hull = d.backproject_contract(expected, contract, 24)
    assert np.all(hull[canonical])


def test_many_cavities_rasterize_in_one_pass_without_filling_them(monkeypatch):
    outer = trimesh.creation.box(extents=[20, 20, 20])
    outer.apply_translation([12, 12, 12])
    parts = [outer]
    centers = [(x, y, z) for x in (7, 17) for y in (7, 17) for z in (7, 17)]
    for center in centers:
        cavity = trimesh.creation.box(extents=[3, 3, 3])
        cavity.apply_translation(center)
        cavity.invert()
        parts.append(cavity)
    mesh = trimesh.util.concatenate(parts)
    calls = []
    original = d._raster_part
    def counted(part, shape):
        calls.append(len(part.faces))
        return original(part, shape)
    monkeypatch.setattr(d, "_raster_part", counted)
    volume = d.rasterize_mesh_with_cavities(mesh, (25, 25, 25))
    assert volume[12, 12, 12]
    assert all(not volume[center] for center in centers)
    assert not volume[0, 0, 0]
    assert len(calls) == 1


def test_containment_batches_children_per_parent(monkeypatch):
    outer = trimesh.creation.box(extents=[20, 20, 20])
    parts = [outer]
    for x in (-5, 5):
        inner = trimesh.creation.box(extents=[2, 2, 2])
        inner.apply_translation([x, 0, 0])
        inner.invert()
        parts.append(inner)
    calls = []
    original = d._contains_points
    def counted(mesh, points):
        calls.append(len(points))
        return original(mesh, points)
    monkeypatch.setattr(d, "_contains_points", counted)
    result = d.audit_shell_containment(trimesh.util.concatenate(parts))
    assert result["exterior_root_count"] == 1
    assert result["cavity_shell_count"] == 2
    assert calls == [2]


def test_containment_point_query_distinguishes_inside_and_outside():
    box = trimesh.creation.box(extents=[10, 10, 10])
    result = d._contains_points(box, np.array([[0., 0., 0.], [6., 0., 0.], [-4., 4., 4.]]))
    assert result.tolist() == [True, False, True]


def test_native_seed_connects_retained_fragments_without_mutating_soft():
    from experiments import multiview_native_delivery_20260905 as n
    soft = np.zeros((32, 32, 32), bool)
    soft[6:14, 6:20, 6:20] = True
    soft[16:20, 9:14, 9:14] = True
    soft[28, 28, 28] = True
    before = soft.copy()
    seed, audit = n.connected_seed(soft, max_components=2)
    assert d._components(seed)[1] == 1
    assert seed[18, 10, 10]
    assert not seed[28, 28, 28]
    assert audit["source_components"] == 3
    assert audit["retained_components"] == 2
    assert np.array_equal(soft, before)


def test_native_gradient_repair_is_monotone_and_connected_with_real_camera():
    from experiments import multiview_native_delivery_20260905 as n
    target_volume = np.zeros((32, 32, 32), bool)
    target_volume[8:24, 8:24, 8:24] = True
    contract = fixed3_contract()
    targets = d.render_contract(target_volume, contract)
    seed = target_volume.copy()
    seed[7, 10:20, 10:20] = True
    result, history = n.refine_connected(seed, targets, contract, device="cpu", steps=4)
    before = d.evaluate_contract(seed, targets, contract)["mean_iou"]
    after = d.evaluate_contract(result, targets, contract)["mean_iou"]
    assert after > before
    assert d._components(result)[1] == 1
    means = [record["metrics"]["mean_iou"] for record in history]
    assert means == sorted(means)


def test_native_refinement_respects_manufacturing_guard():
    from experiments import multiview_native_delivery_20260905 as n
    target = np.zeros((32, 32, 32), bool)
    target[8:24, 8:24, 8:24] = True
    contract = fixed3_contract()
    targets = d.render_contract(target, contract)
    seed = target.copy()
    seed[7, 10:20, 10:20] = True
    result, history = n.refine_connected(seed, targets, contract, device="cpu", steps=2,
                                        acceptance_gate=lambda v: False)
    assert np.array_equal(result, seed)
    assert len(history) == 1


def test_native_refinement_does_not_delete_protected_material():
    from experiments import multiview_native_delivery_20260905 as n
    target = np.zeros((32, 32, 32), bool)
    target[8:24, 8:24, 8:24] = True
    contract = fixed3_contract()
    targets = d.render_contract(target, contract)
    seed = target.copy()
    seed[7, 10:20, 10:20] = True
    protected = seed & ~target
    result, history = n.refine_connected(seed, targets, contract, device="cpu", steps=2,
                                        protected=protected)
    assert np.all(result[protected])


def test_native_failed_gate_publishes_no_mesh(tmp_path):
    from experiments import multiview_native_delivery_20260905 as n
    volume = np.zeros((32, 32, 32), bool)
    volume[8:24, 8:24, 8:24] = True
    targets = [np.zeros((32, 32), bool)] * 6
    with pytest.raises(d.DeliveryFailure, match="native candidate failed"):
        n.finalize_native(volume, targets, fixed3_contract(), tmp_path, "cpu",
                          d.DeliveryConfig(), hull_voxels=6000)
    assert not list(tmp_path.glob("*.obj"))
    assert not list(tmp_path.glob("*.stl"))


@pytest.mark.parametrize("status, expected", [("NO_VALID_CANDIDATE", True), ("ARTIFACT_MUTATED", False)])
def test_auto_only_recovers_candidate_failure_not_artifact_failure(tmp_path, monkeypatch, status, expected):
    from experiments import multiview_native_delivery_20260905 as n
    called = []
    def fail_core(*args):
        raise d.DeliveryFailure(status, "core failure")
    def fail_native(*args, **kwargs):
        called.append(kwargs["fallback"]["status"])
        raise d.DeliveryFailure("NATIVE_GATE_FAILED", "native failure")
    monkeypatch.setattr(d, "_run", fail_core)
    monkeypatch.setattr(n, "run_native", fail_native)
    with pytest.raises(d.DeliveryFailure):
        d.run_contract_delivery(contract=fixed3_contract(256), soft_artifact=tmp_path / "source",
                                staging_output=tmp_path / "stage", device="cpu", delivery_mode="auto")
    assert bool(called) is expected
    assert not list((tmp_path / "stage").glob("*.obj"))
