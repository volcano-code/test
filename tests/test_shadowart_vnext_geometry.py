"""Real geometry fixtures catch cavity filling, stretch and unsafe repair."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh


def module():
    source = Path(__file__).parents[1] / "experiments/shadowart_vnext/geometry.py"
    assert source.is_file(), "native geometry implementation is missing"
    return importlib.import_module("experiments.shadowart_vnext.geometry")


def camera():
    return SimpleNamespace(name="front", target_index=0, origin=(0., 0., 2.),
                           right=(1., 0., 0.), up=(0., 1., 0.),
                           direction=(0., 0., -1.), scale=2.)


def test_padded_extraction_preserves_a_legal_cavity_and_world_axes():
    v = np.zeros((32,) * 3, bool)
    v[3:29, 5:27, 7:25] = True
    v[10:22, 11:21, 12:20] = False
    mesh = module().occupancy_to_mesh(v, extent=3.2)
    assert mesh.is_watertight
    assert np.allclose(mesh.extents, [2.6, 2.2, 1.8])
    assert not mesh.contains([[0., 0., 0.]])[0]
    assert mesh.contains([[1., 0., 0.]])[0]
    report = module().mesh_audit(mesh)
    assert report["passed"], report
    assert report["outer_shells"] == 1
    assert report["legal_cavities"] == 1


def test_mesh_audit_rejects_crossing_and_disconnected_closed_shells():
    first = trimesh.creation.box(extents=(2., 2., 2.))
    second = first.copy()
    second.apply_translation([1., .4, .3])
    crossing = module().mesh_audit(trimesh.util.concatenate([first, second]))
    assert not crossing["passed"]
    assert crossing["self_intersection"]["status"] == "FAIL"
    second.apply_translation([4., 0., 0.])
    disconnected = module().mesh_audit(trimesh.util.concatenate([first, second]))
    assert not disconnected["passed"]
    assert disconnected["outer_shells"] == 2


def test_mesh_audit_fails_closed_when_intersection_backend_is_unavailable(monkeypatch):
    real_import = importlib.import_module
    def missing(name, *args, **kwargs):
        if name == "pymeshlab":
            raise ImportError("deliberately absent")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", missing)
    report = module().mesh_audit(trimesh.creation.box())
    assert not report["passed"]
    assert report["self_intersection"]["status"] == "REVIEW_REQUIRED"


def test_uniform_transform_keeps_shape_and_maps_y_up_to_bed_z():
    original = trimesh.creation.box(extents=(2., 4., 1.))
    original.apply_translation([3., 5., -2.])
    printed, matrix = module().uniform_print_transform(original, 100.)
    assert np.allclose(np.linalg.svd(matrix[:3, :3])[1], [25., 25., 25.])
    assert np.linalg.det(matrix[:3, :3]) > 0
    assert np.allclose(printed.extents, [50., 25., 100.])
    assert printed.bounds[0, 2] == pytest.approx(0.)
    assert np.allclose((printed.bounds[0, :2] + printed.bounds[1, :2]) / 2, [0., 0.])
    assert np.allclose(original.extents, [2., 4., 1.])
    assert np.allclose(matrix[:3, :3] @ [0., 1., 0.], [0., 0., 25.])


def test_repair_preserves_cavity_and_does_not_mutate_inputs():
    r = 20
    v = np.zeros((r,) * 3, bool)
    v[3:17, 3:17, 3:17] = True
    v[7:13, 7:13, 7:13] = False
    original = v.copy()
    from experiments.shadowart_vnext.rendering import render_binary
    targets = render_binary(v, [camera()], r)
    result, trace = module().repair_geometry(v, v.astype(float), targets, [camera()],
                                            longest_mm=10., max_rounds=1)
    assert np.array_equal(v, original)
    assert not result[7:13, 7:13, 7:13].any()
    assert trace["final"]["quality"]["mean_iou"] == 1.


def test_failed_fidelity_fin_removal_is_rolled_back():
    v = np.zeros((28,) * 3, bool)
    v[3:15, 5:23, 5:23] = True
    v[15:25, 10:18, 13:14] = True
    from experiments.shadowart_vnext.rendering import render_binary
    targets = render_binary(v, [camera()], 56)
    result, trace = module().repair_geometry(v, v.astype(float), targets, [camera()],
                                            longest_mm=5., max_rounds=1)
    assert result[18:24, 11:17, 13].all()
    assert trace["final"]["quality"]["mean_iou"] > .98
    assert any(row.get("reason") == "QUALITY_GATE" for row in trace["attempts"])
    assert trace["status"] == "UNRESOLVED"


def test_disconnected_body_bridge_uses_a_physical_cross_section():
    v = np.zeros((24,) * 3, bool)
    v[4:10, 7:17, 7:17] = True
    v[14:20, 7:17, 7:17] = True
    cam = SimpleNamespace(name="side", target_index=0, origin=(2., 0., 0.),
                          right=(0., 0., -1.), up=(0., 1., 0.),
                          direction=(-1., 0., 0.), scale=2.)
    from experiments.shadowart_vnext.rendering import render_binary
    targets = render_binary(v, [cam], 24)
    result, trace = module().repair_geometry(v, v.astype(float), targets, [cam],
                                            longest_mm=8., max_rounds=2)
    assert trace["final"]["manufacturing"]["component_count_6"] == 1
    assert result[11].sum() >= 9
    assert trace["final"]["quality"]["mean_iou"] == 1.


def test_exact_mean_threshold_is_not_a_pass():
    v = np.zeros((10,) * 3, bool)
    v[:, :, 5] = True
    v[0, :2, 5] = False
    result, trace = module().repair_geometry(v, v.astype(float),
                                            np.ones((1, 10, 10), bool), [camera()],
                                            extent=2., max_rounds=0)
    assert trace["final"]["quality"]["mean_iou"] == pytest.approx(.98)
    assert not trace["final"]["quality"]["passed"]
    assert trace["status"] == "UNRESOLVED"


def test_missing_projection_ray_is_restored_using_measured_contribution():
    complete = np.zeros((20,) * 3, bool)
    complete[4:16, 4:16, 4:16] = True
    v = complete.copy()
    v[10, 10, 4:16] = False
    from experiments.shadowart_vnext.rendering import render_binary
    targets = render_binary(complete, [camera()], 40)
    result, trace = module().repair_geometry(v, complete.astype(float), targets,
                                            [camera()], longest_mm=10., max_rounds=2)
    assert trace["final"]["quality"]["mean_iou"] > trace["initial"]["quality"]["mean_iou"]
    assert trace["final"]["quality"]["mean_iou"] == 1.
    assert any(row["edit"] == "projection_restoration" and row["accepted"] for row in trace["attempts"])
    assert json.loads(json.dumps(trace))["publication_eligible"] is False


def test_empty_native_source_remains_unresolved_without_inventing_material():
    v = np.zeros((8,) * 3, bool)
    result, trace = module().repair_geometry(v, v.astype(float),
                                            np.ones((1, 8, 8), bool), [camera()])
    assert not result.any()
    assert trace["status"] == "UNRESOLVED"
    assert not trace["final"]["quality"]["passed"]


def test_complete_ray_counts_identify_exclusive_foreground_hits():
    v = np.zeros((8,) * 3, bool)
    v[2, 3, 1] = v[2, 3, 5] = True
    v[4, 4, 3] = True
    target = np.zeros((1, 8, 8), bool)
    target[0, 4, 2] = target[0, 3, 4] = True
    constraints = module().ray_constraints(v, target, [camera()], extent=2.)
    assert constraints["counts"][0, 4, 2] == 2
    assert constraints["counts"][0, 3, 4] == 1
    assert constraints["exclusive_hits"][4, 4, 3]
    assert not constraints["exclusive_hits"][2, 3, 1]
    assert not constraints["exclusive_hits"][2, 3, 5]


def test_one_round_batch_restores_many_separate_missing_rays():
    complete = np.zeros((32,) * 3, bool)
    complete[3:29, 3:29, 3:29] = True
    v = complete.copy()
    for x in (6, 12, 18, 24):
        for y in (6, 12, 18, 24):
            v[x, y, 3:29] = False
    from experiments.shadowart_vnext.rendering import render_binary
    targets = render_binary(complete, [camera()], 64, extent=2.)
    result, trace = module().repair_geometry(v, complete.astype(float), targets,
                                            [camera()], extent=2., longest_mm=13., max_rounds=1)
    assert trace["final"]["quality"]["mean_iou"] == 1.
    assert sum(row.get("patches", 0) for row in trace["attempts"] if row["accepted"]) >= 8


def test_weighted_thick_bridge_avoids_background_leakage_corridor():
    v = np.zeros((26,) * 3, bool)
    v[3:8, 7:12, 8:13] = True
    v[18:23, 7:12, 8:13] = True
    desired = v.copy()
    desired[4:8, 9:18, 8:13] = True
    desired[18:22, 9:18, 8:13] = True
    desired[4:22, 14:18, 8:13] = True
    from experiments.shadowart_vnext.rendering import render_binary
    targets = render_binary(desired, [camera()], 26, extent=2.)
    constraints = module().ray_constraints(v, targets, [camera()], extent=2.)
    candidate = module()._thick_bridge(v, 2, guidance=desired.astype(float),
                                        background_votes=constraints["background_votes"])
    prediction = render_binary(candidate, [camera()], 26, extent=2.)
    assert not np.any(prediction & ~targets)
    assert candidate[12, 14:18, 8:13].any()
    assert not candidate[12, 7:12, 8:13].any()


def test_batch_pruning_removes_redundant_islands_without_losing_exclusive_rays():
    v = np.zeros((28,) * 3, bool)
    v[5:23, 5:23, 5:10] = True
    for x in (7, 11, 15, 19):
        for y in (7, 11, 15, 19):
            v[x, y, 18] = True
    from experiments.shadowart_vnext.rendering import render_binary
    targets = render_binary(v, [camera()], 28, extent=2.)
    result, trace = module().repair_geometry(v, v.astype(float), targets, [camera()],
                                            extent=2., longest_mm=9., max_rounds=1)
    assert trace["final"]["manufacturing"]["component_count_6"] == 1
    assert trace["final"]["quality"]["mean_iou"] == 1.
    assert np.array_equal(render_binary(result, [camera()], 28, extent=2.), targets)


def test_oblique_ray_incidence_and_local_box_counts_agree_with_full_renderer():
    angle = .47
    cam = SimpleNamespace(origin=(2*np.sin(angle), 0., 2*np.cos(angle)),
                          right=(np.cos(angle), 0., -np.sin(angle)), up=(0., 1., 0.),
                          direction=(-np.sin(angle), 0., -np.cos(angle)), scale=2.)
    v = np.random.default_rng(52).random((9,) * 3) > .6
    from experiments.shadowart_vnext.rendering import render_binary
    targets = render_binary(v, [cam], 23, extent=1.7)
    constraints = module().ray_constraints(v, targets, [cam], extent=1.7)
    assert np.array_equal(constraints["counts"] > 0, targets)
    touched = module()._deletion_counts(np.argwhere(v), [cam], 23, 9, 1.7)
    reconstructed = np.zeros_like(constraints["counts"])
    for pixel, count in touched[0].items():
        reconstructed[0].ravel()[pixel] = count
    assert np.array_equal(reconstructed, constraints["counts"])


@pytest.mark.parametrize("azimuth,elevation", [(90., 0.), (0., 90.)])
def test_opposite_cardinal_local_counts_share_renderer_grid_plane_ownership(azimuth, elevation):
    from dataclasses import replace
    from experiments.shadowart_vnext.contracts import angle_camera
    from experiments.shadowart_vnext.rendering import render_binary
    base = angle_camera(0, azimuth, elevation)
    cam = replace(base, origin=tuple(-np.array(base.origin)),
                  right=tuple(-np.array(base.right)), direction=tuple(-np.array(base.direction)))
    v = np.random.default_rng(1).random((16,) * 3) > .9
    targets = render_binary(v, [cam], 32)
    constraints = module().ray_constraints(v, targets, [cam])
    touched = module()._deletion_counts(np.argwhere(v), [cam], 32, 16, 1.7)
    reconstructed = np.zeros_like(constraints["counts"])
    for pixel, count in touched[0].items():
        reconstructed[0].ravel()[pixel] = count
    assert np.array_equal(reconstructed, constraints["counts"])


def test_local_sdf_candidate_repairs_real_saddle_topology_with_mesh_qa():
    # Minimal saddle pattern extracted from the failed native pilot; external
    # connections make one six-connected body without changing the saddle.
    v = np.zeros((9,) * 3, bool)
    v[2, 1:4, 2] = v[3, 1:4, 3] = True
    v[3, 4:7, 2] = v[4, 4:7, 3] = True
    v[1:6, 1, 1:5] = v[1:6, 6, 1:5] = v[1, 1:7, 1:5] = True
    before = module().occupancy_to_mesh(v, extent=2.)
    assert not before.is_watertight
    from experiments.shadowart_vnext.rendering import render_mesh, voxelize_mesh
    targets = render_mesh(before, [camera()], 9)
    mesh, report = module().refine_surface_mesh(v, targets, [camera()], extent=2., longest_mm=3.)
    assert report["attempts"], report
    assert any(row["mesh_audit"]["passed"] for row in report["attempts"]), report
    assert report["accepted"], report
    assert mesh.is_watertight
    assert module().mesh_audit(mesh)["passed"]
    assert np.array_equal(voxelize_mesh(mesh, 9, extent=2.), v)
    assert np.array_equal(render_mesh(mesh, [camera()], 9), targets)
    assert report["publication_eligible"] is False


def test_surface_refinement_preserves_valid_cavity_without_global_smoothing():
    v = np.zeros((18,) * 3, bool)
    v[2:16, 2:16, 2:16] = True
    v[6:12, 6:12, 6:12] = False
    from experiments.shadowart_vnext.rendering import render_binary
    targets = render_binary(v, [camera()], 36)
    mesh, report = module().refine_surface_mesh(v, targets, [camera()])
    assert report["status"] == "NOT_NEEDED"
    assert not mesh.contains([[0., 0., 0.]])[0]
    assert report["initial_mesh_audit"]["legal_cavities"] == 1


@pytest.mark.parametrize("disabled", [("reinforcment",), ("fin_removal", "fin_removal"), "reinforcement"])
def test_geometry_ablation_rejects_unknown_duplicate_or_string_switches(disabled):
    v = np.zeros((10,) * 3, bool); v[2:8, 2:8, 2:8] = True
    from experiments.shadowart_vnext.rendering import render_binary
    with pytest.raises(ValueError, match="disabled_stages"):
        module().repair_geometry(v, v.astype(float), render_binary(v, [camera()], 10),
                                 [camera()], max_rounds=0, disabled_stages=disabled)


def test_disabled_geometry_stages_remove_all_removal_and_reinforcement_proposals():
    v = np.zeros((24,) * 3, bool)
    v[3:10, 5:19, 5:19] = True
    v[14:20, 7:17, 10:11] = True
    from experiments.shadowart_vnext.rendering import render_binary
    target = render_binary(v, [camera()], 48)
    result, trace = module().repair_geometry(v, v.astype(float), target, [camera()],
        longest_mm=6., max_rounds=1, disabled_stages=("reinforcement", "fin_removal"))
    assert np.array_equal(result, v)
    assert not trace["attempts"]
    assert trace["disabled_stages"] == ["reinforcement", "fin_removal"]
    assert "guarded_surface_batch" in trace["stage_switch_semantics"]["fin_removal"]


def test_geometry_ablation_keeps_projection_restoration_enabled():
    complete = np.zeros((20,) * 3, bool); complete[4:16, 4:16, 4:16] = True
    v = complete.copy(); v[10, 10, 4:16] = False
    from experiments.shadowart_vnext.rendering import render_binary
    result, trace = module().repair_geometry(v, complete.astype(float),
        render_binary(complete, [camera()], 40), [camera()], longest_mm=10., max_rounds=1,
        disabled_stages=("reinforcement", "fin_removal"))
    assert trace["final"]["quality"]["mean_iou"] == 1.
    assert any(row["edit"] == "projection_restoration" and row["accepted"] for row in trace["attempts"])


def test_fin_removal_cannot_open_an_original_closed_cavity():
    v = np.zeros((32,) * 3, bool)
    v[3:29, 3:29, 3:29] = True
    v[4:23, 8:24, 8:24] = False
    protected = module()._enclosed_voids(v)
    assert protected.sum() == 4864
    from experiments.shadowart_vnext.rendering import render_binary
    result, trace = module().repair_geometry(v, v.astype(float),
        render_binary(v, [camera()], 64, extent=2.), [camera()], extent=2.,
        longest_mm=10., max_rounds=1)
    assert not np.any(result & protected)
    assert np.all(module()._enclosed_voids(result)[protected])
    assert any(row.get("reason") == "ORIGINAL_CAVITY_OPENED" for row in trace["attempts"])


def test_surface_batch_skips_articulation_then_keeps_safe_deletions():
    from scipy import ndimage as ndi
    v = np.zeros((12,) * 3, bool)
    v[2:5, 4:8, 4:8] = True
    v[6:9, 4:8, 4:8] = True
    v[5, 5, 5] = True
    guidance = np.ones(v.shape)
    guidance[5, 5, 5] = 0.  # First removal proposal is the sole connecting voxel.
    target = np.zeros((1, 24, 24), bool)
    constraints = module().ray_constraints(v, target, [camera()], extent=2.)
    result, removed, kind = module()._batch_prune(v, guidance, target, [camera()],
                                                constraints, 2., 8)
    assert result[5, 5, 5]
    assert removed > 0
    assert ndi.label(result, module().FACE6)[1] == 1
    assert kind == "guarded_surface_batch"
    assert v.sum() == 97


def fixture_candidates(v, *, pitch=.25):
    from experiments.shadowart_vnext.rendering import render_binary
    target = render_binary(v, [camera()], 48, extent=2.)
    constraints = module().ray_constraints(v, target, [camera()], extent=2.)
    return list(module()._candidates(v, v.astype(float), target, [camera()], target,
        2., pitch, .9, 1.35, constraints, 8, module()._enclosed_voids(v)))


def test_thin_region_candidates_include_eight_largest_distinct_regions():
    v = np.zeros((32,) * 3, bool)
    v[3:27, 3:27, 3:10] = True
    for i, (x, y) in enumerate((x, y) for x in (7, 14, 21) for y in (7, 14, 21)):
        v[x, y, 10:13+i] = True
    rows = fixture_candidates(v)
    removals = [(a, extra) for kind, a, extra in rows if kind == "fin_removal"]
    assert len(removals) == 8
    assert [int((v & ~a).sum()) for a, _ in removals] == [11, 10, 9, 8, 7, 6, 5, 4]
    assert len({extra["thin_region_id"] for _, extra in removals}) == 8
    assert len([kind for kind, _, _ in rows if kind == "fin_shortening"]) == 8
    assert len([kind for kind, _, _ in rows if kind == "local_fin_thickening"]) == 8


def test_bridge_candidates_include_eight_distinct_component_endpoints():
    from scipy import ndimage as ndi
    v = np.zeros((40,) * 3, bool)
    v[3:11, 3:11, 3:11] = True
    for x, y in ((x, y) for x in (16, 24, 32) for y in (7, 15, 23)):
        v[x:x+3, y:y+3, 7:10] = True
    rows = [(a, extra) for kind, a, extra in fixture_candidates(v, pitch=.7)
            if kind == "physical_component_bridge"]
    assert len(rows) == 8
    assert len({extra["bridge_component_id"] for _, extra in rows}) == 8
    assert all(ndi.label(a, module().FACE6)[1] < 10 for a, _ in rows)
    assert all(np.all(a[v]) for a, _ in rows)


def test_geometry_trace_records_finite_regional_and_endpoint_budgets():
    v = np.zeros((10,) * 3, bool); v[2:8, 2:8, 2:8] = True
    from experiments.shadowart_vnext.rendering import render_binary
    _, trace = module().repair_geometry(v, v.astype(float), render_binary(v, [camera()], 20),
        [camera()], max_rounds=0, max_patches_per_round=16)
    assert trace["candidate_budgets"] == {"thin_regions_per_round":8,
        "bridge_endpoints_per_round":8,"surface_points_considered_per_round":1024,
        "accepted_patches_per_batch":16}
