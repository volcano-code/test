from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from experiments.multiview_contract_delivery_20260902 import (
    DeliveryConfig,
    DeliveryFailure,
    audit_reloaded_fabrication_contract,
    audit_shell_containment,
    backproject_contract,
    bridge6_restricted,
    canonical_xyz_to_trainer_zyx,
    carve_core_trajectory,
    core_shell_deliver,
    enforce_triangle_cap,
    evaluate_delivery_trajectory_states,
    export_same_source_mesh,
    padded_sdf_mesh,
    rasterize_mesh_with_cavities,
    render_contract,
    restore_contract_rays,
    run_contract_delivery,
    select_delivery_candidate,
    trainer_zyx_to_canonical_xyz,
)


@dataclass(frozen=True)
class LiteralContract:
    protocol_id: str = "literal_generic"
    status: str = "extension"
    requested_tsf: int = 0
    effective_tsf: int = 0
    mirror: int = 0
    trainer_tsf: int = 0
    trainer_tsf4: int = 0
    views: tuple[tuple[float, float], ...] = ((10.0, 20.0), (30.0, 40.0))
    renderer_family: str = "custom_orthographic"
    direct_view_roles: tuple[tuple[int | None, str | None], ...] = (
        (None, None),
        (None, None),
    )
    image_size: int = 8
    zdist: float | None = None
    volume_extent: float | None = None
    ray_samples: int | None = None
    input_sha256: tuple[tuple[str, str], ...] = ()
    normalized_target_sha256: tuple[tuple[str, str], ...] = ()
    target_order: tuple[str, ...] = ()
    soft_zyx_to_canonical_xyz: str = "flip(transpose(zyx,(2,1,0)),axes=(0,2))"
    preserve_target_aspect: bool = True
    no_registration: bool = True
    artifact_sha256: tuple[tuple[str, str], ...] = ()


def _literal_renderer(calls: list[tuple[object, ...]]):
    def render(
        *, volume_xyz, family, view_index, azimuth_deg, elevation_deg, contract, device
    ):
        calls.append(
            (
                family,
                view_index,
                azimuth_deg,
                elevation_deg,
                tuple(volume_xyz.shape),
                device,
                contract.protocol_id,
            )
        )
        result = np.zeros(volume_xyz.shape[:2], dtype=bool)
        result[view_index, view_index + 1] = True
        return result

    return render


def test_frame_mapping_round_trip_is_exact_and_contiguous() -> None:
    trainer_bool = np.zeros((3, 4, 5), dtype=bool)
    trainer_bool[0, 1, 4] = True
    trainer_bool[2, 3, 0] = True
    xyz_bool = trainer_zyx_to_canonical_xyz(trainer_bool)
    assert xyz_bool.shape == (5, 4, 3)
    assert xyz_bool.flags.c_contiguous
    assert np.argwhere(xyz_bool).tolist() == [[0, 1, 2], [4, 3, 0]]
    assert np.array_equal(canonical_xyz_to_trainer_zyx(xyz_bool), trainer_bool)

    trainer_float = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    round_trip = canonical_xyz_to_trainer_zyx(
        trainer_zyx_to_canonical_xyz(trainer_float)
    )
    assert round_trip.dtype == np.float32
    assert round_trip.flags.c_contiguous
    assert np.array_equal(round_trip, trainer_float)


def test_render_contract_dispatches_literal_ordered_views() -> None:
    calls: list[tuple[object, ...]] = []
    contract = LiteralContract()
    rendered = render_contract(
        np.ones((8, 8, 8), dtype=bool),
        contract,
        renderer=_literal_renderer(calls),
        device="cpu",
    )
    assert calls == [
        ("custom_orthographic", 0, 10.0, 20.0, (8, 8, 8), "cpu", "literal_generic"),
        ("custom_orthographic", 1, 30.0, 40.0, (8, 8, 8), "cpu", "literal_generic"),
    ]
    assert [np.argwhere(view).tolist() for view in rendered] == [
        [[0, 1]],
        [[1, 2]],
    ]


def test_bridge6_connects_only_through_envelope() -> None:
    volume = np.zeros((7, 7, 7), dtype=bool)
    volume[1, 3, 3] = True
    volume[5, 3, 3] = True
    envelope = np.zeros_like(volume)
    envelope[1:6, 3, 3] = True

    connected, audit = bridge6_restricted(volume, envelope, max_passes=4, radius=0)
    assert np.argwhere(connected).tolist() == [
        [1, 3, 3],
        [2, 3, 3],
        [3, 3, 3],
        [4, 3, 3],
        [5, 3, 3],
    ]
    assert not np.any(connected & ~envelope)
    assert audit["component_count_6"] == 1

    blocked_envelope = envelope.copy()
    blocked_envelope[3, 3, 3] = False
    with pytest.raises(DeliveryFailure, match="no in-envelope Manhattan route"):
        bridge6_restricted(volume, blocked_envelope, max_passes=4, radius=0)


def test_first_failed_core_checkpoint_rolls_back_to_prior_state() -> None:
    seed = np.zeros((8, 8, 8), dtype=bool)
    seed[2:6, 2:6, 2:6] = True
    probability = np.ones(seed.shape, dtype=np.float32)
    envelope = np.ones_like(seed)
    target = np.ones((8, 8), dtype=bool)
    render_shapes: list[tuple[int, int, int]] = []

    def renderer(**kwargs):
        volume = kwargs["volume_xyz"]
        render_shapes.append(tuple(volume.shape))
        if int(volume.sum()) == int(seed.sum()):
            return target.copy()
        return np.zeros_like(target)

    accepted, audit = carve_core_trajectory(
        seed,
        probability,
        envelope,
        [target, target],
        LiteralContract(),
        renderer=renderer,
        device="cpu",
        max_checkpoints=3,
        deletion_budget=1,
        bridge_max_passes=4,
        bridge_radius=0,
    )
    assert np.array_equal(accepted, seed)
    assert audit["accepted_checkpoints"] == 0
    assert audit["rollback_reason"] == "core_gate_failed"
    assert len(audit["accepted_states"]) == 1
    assert audit["accepted_states"][0]["checkpoint"] == "seed"
    assert np.array_equal(audit["accepted_states"][0]["volume"], seed)
    assert render_shapes and set(render_shapes) == {(8, 8, 8)}


def test_core_trajectory_retains_seed_and_every_accepted_checkpoint() -> None:
    seed = np.zeros((8, 8, 8), dtype=bool)
    seed[2:6, 2:6, 2:6] = True
    probability = np.ones(seed.shape, dtype=np.float32)
    envelope = seed.copy()
    target = np.ones((8, 8), dtype=bool)

    def renderer(**kwargs):
        return target.copy()

    accepted, audit = carve_core_trajectory(
        seed,
        probability,
        envelope,
        [target, target],
        # Isolate checkpoint retention from oblique interpolation/restoration.
        replace(LiteralContract(), views=((0., 0.), (0., 0.))),
        renderer=renderer,
        max_checkpoints=2,
        deletion_budget=1,
        bridge_max_passes=4,
        bridge_radius=0,
    )
    states = audit["accepted_states"]
    assert [state["checkpoint"] for state in states] == ["seed", 0, 1]
    assert np.array_equal(states[0]["volume"], seed)
    assert np.array_equal(states[-1]["volume"], accepted)
    assert [int(state["volume"].sum()) for state in states] == [64, 63, 62]


def test_core_shell_delivery_never_deletes_lifted_material() -> None:
    core = np.zeros((4, 4, 4), dtype=bool)
    core[1, 1, 1] = True
    envelope = np.ones((8, 8, 8), dtype=bool)
    probability = np.ones(envelope.shape, dtype=np.float32)
    target = np.ones((8, 8), dtype=bool)

    def renderer(**kwargs):
        return target.copy()

    delivered, audit = core_shell_deliver(
        core,
        envelope,
        probability,
        [target, target],
        LiteralContract(),
        renderer=renderer,
        device="cpu",
        shell_voxels=2,
        bridge_max_passes=4,
        bridge_radius=0,
    )
    expected_lift = np.zeros_like(envelope)
    expected_lift[2:4, 2:4, 2:4] = True
    assert np.all(delivered[expected_lift])
    assert int(delivered.sum()) >= 8
    assert audit["deleted_voxels"] == 0
    assert audit["lifted_voxels"] == 8
    assert audit["shell_added_voxels"] == audit["local_shell_voxels"]
    assert audit["total_added_voxels"] == (
        audit["local_shell_voxels"]
        + audit["ray_restored_voxels"]
        + audit["bridge_added_voxels"]
    )


def test_core_shell_uses_only_local_h_boundary_not_ordinary_thickening() -> None:
    core = np.zeros((4, 4, 4), dtype=bool)
    core[1, 1, 1] = True
    envelope = np.ones((16, 16, 16), dtype=bool)
    probability = np.ones(envelope.shape, dtype=np.float32)
    empty_target = np.zeros((16, 16), dtype=bool)
    delivered, audit = core_shell_deliver(
        core,
        envelope,
        probability,
        [empty_target, empty_target],
        replace(LiteralContract(), image_size=16),
        renderer=_literal_renderer([]),
        device="cpu",
        shell_voxels=2,
        bridge_max_passes=4,
        bridge_radius=0,
    )
    expected_lift = np.zeros_like(envelope)
    expected_lift[4:8, 4:8, 4:8] = True
    assert np.array_equal(delivered, expected_lift)
    assert audit["shell_added_voxels"] == 0
    assert audit["local_shell_voxels"] == 0
    assert audit["ray_restored_voxels"] == 0
    assert audit["bridge_added_voxels"] == 0
    assert audit["total_added_voxels"] == 0


def _nested_box_mesh() -> trimesh.Trimesh:
    outer = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    outer.apply_translation((7.5, 7.5, 7.5))
    cavity = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    cavity.apply_translation((7.5, 7.5, 7.5))
    cavity.invert()
    return trimesh.util.concatenate((outer, cavity))


def test_nested_shell_is_legal_and_cavity_survives_rasterization() -> None:
    mesh = _nested_box_mesh()
    audit = audit_shell_containment(mesh)
    assert audit["exterior_root_count"] == 1
    assert audit["cavity_shell_count"] == 1
    assert audit["all_watertight"] is True
    assert audit["all_winding_consistent"] is True

    raster = rasterize_mesh_with_cavities(mesh, (16, 16, 16))
    assert raster[4, 4, 4]
    assert not raster[8, 8, 8]
    assert int(raster.sum()) > 0


def test_padded_sdf_preserves_internal_cavity_without_fill() -> None:
    volume = np.zeros((12, 12, 12), dtype=bool)
    volume[1:11, 1:11, 1:11] = True
    volume[4:8, 4:8, 4:8] = False
    mesh = padded_sdf_mesh(volume, pad=2, sigma=0.0, level=0.0)
    audit = audit_shell_containment(mesh)
    assert audit["exterior_root_count"] == 1
    assert audit["cavity_shell_count"] == 1
    raster = rasterize_mesh_with_cavities(mesh, volume.shape)
    assert raster[2, 2, 2]
    assert not raster[6, 6, 6]


def test_triangle_cap_fails_closed_without_simplifier() -> None:
    mesh = trimesh.creation.box()
    with pytest.raises(DeliveryFailure) as caught:
        enforce_triangle_cap(mesh, max_faces=10, simplifier=None)
    assert caught.value.status == "TRIANGLE_CAP_FAILED"
    assert caught.value.details["source_faces"] == 12
    assert caught.value.details["max_faces"] == 10


def test_stl_obj_and_preview_are_exported_from_one_transformed_source(
    tmp_path: Path,
) -> None:
    mesh = _nested_box_mesh()
    result = export_same_source_mesh(mesh, tmp_path, max_dimension_mm=100.0)
    assert set(result["files"]) == {
        "fabrication_stl",
        "fabrication_obj",
        "outer_preview_stl",
        "outer_preview_obj",
    }
    for path in result["files"].values():
        assert Path(path).is_file()
    assert result["source_vertices_faces_sha256"]
    assert result["outer_preview_source_vertices_faces_sha256"]
    assert result["transform"]["uniform_scale"] == 10.0
    assert len(result["transform"]["translation"]) == 3
    assert set(result["file_sha256"]) == set(result["files"])
    assert all(len(value) == 64 for value in result["file_sha256"].values())
    assert result["fabrication_reload"]["stl"]["faces"] == 24
    assert result["fabrication_reload"]["obj"]["faces"] == 24
    assert result["outer_preview_reload"]["stl"]["faces"] == 12
    assert result["outer_preview_reload"]["obj"]["faces"] == 12
    stl_mesh = trimesh.load(result["files"]["fabrication_stl"], force="mesh")
    obj_mesh = trimesh.load(result["files"]["fabrication_obj"], force="mesh")
    assert np.allclose(stl_mesh.extents, (100.0, 100.0, 100.0), atol=1e-5)
    assert np.allclose(obj_mesh.extents, stl_mesh.extents, atol=1e-5)
    assert abs(float(stl_mesh.bounds[0, 2])) < 1e-6
    assert abs(float(obj_mesh.bounds[0, 2])) < 1e-6


def test_delivery_config_is_frozen_and_uses_paper_gates() -> None:
    config = DeliveryConfig()
    assert config.core_resolution == 48
    assert config.delivery_resolution == 256
    assert config.mean_iou_gate == 0.95
    assert config.worst_iou_gate == 0.90
    assert config.max_faces == 2_000_000
    with pytest.raises(Exception):
        config.max_faces = 5  # type: ignore[misc]
    invalid_overrides = (
        {"mean_iou_gate": -5.0},
        {"worst_iou_gate": -5.0},
        {"minimum_saving": -9.0},
        {"maximum_saving": 9.0},
        {"minimum_erosion_survival": -1.0},
        {"shell_voxels": 1},
        {"max_faces": 10},
        {"max_dimension_mm": 0.0},
        {"sdf_sigma": -0.1},
    )
    for override in invalid_overrides:
        with pytest.raises(ValueError):
            DeliveryConfig(**override)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("core_resolution", True),
        ("delivery_resolution", 256.0),
        ("max_checkpoints", 256.0),
        ("shell_voxels", 2.0),
        ("sdf_pad", True),
        ("max_faces", 2_000_000.0),
        ("mean_iou_gate", np.float64(0.95)),
        ("sdf_sigma", 0),
        ("sdf_sigma", float("inf")),
        ("sdf_level", float("nan")),
        ("max_dimension_mm", 100),
    ),
)
def test_delivery_config_requires_exact_finite_field_types(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        DeliveryConfig(**{field: value})


def test_all_accepted_core_states_reach_delivery_gate_without_early_break() -> None:
    contract = replace(
        LiteralContract(),
        views=((0.0, 0.0),),
        direct_view_roles=((None, None),),
        image_size=16,
    )
    seed = np.ones((4, 4, 4), dtype=bool)
    checkpoint = np.zeros_like(seed)
    checkpoint[0:3, 0:3, 0:3] = True
    target = np.zeros((16, 16), dtype=bool)
    target[0, 0] = True

    def renderer(**kwargs):
        return target.copy()

    rows, valid = evaluate_delivery_trajectory_states(
        [
            {"checkpoint": "seed", "volume": seed},
            {"checkpoint": 0, "volume": checkpoint},
        ],
        threshold=0.5,
        envelope_xyz=np.ones((16, 16, 16), dtype=bool),
        probability_xyz=np.ones((16, 16, 16), dtype=np.float32),
        targets=[target],
        contract=contract,
        config=DeliveryConfig(
            core_resolution=4,
            delivery_resolution=16,
            max_checkpoints=1,
            sdf_sigma=0.0,
        ),
        renderer=renderer,
        device="cpu",
        bridge_max_passes=4,
        bridge_radius=0,
    )
    assert [row["status"] for row in rows] == [
        "absolute_gate_failed",
        "absolute_valid",
    ]
    assert rows[0]["delivery_metrics"]["saving"] < 1e-6
    assert len(valid) == 1
    assert valid[0][2]["checkpoint"] == 0


def test_generic_backproject_and_restore_use_literal_view() -> None:
    contract = replace(
        LiteralContract(),
        views=((0.0, 0.0),),
        direct_view_roles=((None, None),),
        image_size=5,
    )
    target = np.zeros((5, 5), dtype=bool)
    target[1, 2] = True
    hull = backproject_contract([target], contract, 5, device="cpu")
    assert np.argwhere(hull).tolist() == [
        [0, 2, 3],
        [1, 2, 3],
        [2, 2, 3],
        [3, 2, 3],
        [4, 2, 3],
    ]
    guidance = np.zeros_like(hull, dtype=np.float32)
    # raw Z=1 maps to canonical Z=3 under the declared ZYX->XYZ frame.
    guidance[4, 2, 3] = 9.0
    restored = restore_contract_rays(
        np.zeros_like(hull),
        hull,
        guidance,
        [target],
        contract,
        device="cpu",
        rounds=2,
    )
    assert np.argwhere(restored).tolist() == [[4, 2, 3]]


def test_direct_backproject_restore_uses_contract_axis_transform_without_fallback() -> (
    None
):
    contract = replace(
        LiteralContract(),
        renderer_family="sadekar_voxel_tfs_mirror_orthographic",
        views=((0.0, 0.0),),
        direct_view_roles=((0, "ud"),),
        image_size=5,
        zdist=1.7,
        volume_extent=1.7,
        ray_samples=150,
    )
    target = np.zeros((5, 5), dtype=bool)
    target[1, 2] = True
    hull = backproject_contract([target], contract, 5)
    assert np.argwhere(hull).tolist() == [
        [2, 3, 0],
        [2, 3, 1],
        [2, 3, 2],
        [2, 3, 3],
        [2, 3, 4],
    ]
    guidance = np.zeros_like(hull, dtype=np.float32)
    guidance[2, 3, 1] = 7.0
    restored = restore_contract_rays(
        np.zeros_like(hull), hull, guidance, [target], contract
    )
    assert np.argwhere(restored).tolist() == [[2, 3, 1]]

    invalid = replace(contract, direct_view_roles=((None, None),))
    with pytest.raises(DeliveryFailure, match="direct ray roles malformed"):
        backproject_contract([target], invalid, 5)


def test_candidate_selection_uses_near_best_fidelity_band() -> None:
    a = np.zeros((2, 2, 2), dtype=bool)
    b = np.ones((2, 2, 2), dtype=bool)
    c = np.ones((2, 2, 2), dtype=bool)
    selected, metrics = select_delivery_candidate(
        [
            (a, {"mean_iou": 0.990, "saving": 0.100, "occupied_voxels": 100}),
            (b, {"mean_iou": 0.985, "saving": 0.200, "occupied_voxels": 80}),
            (c, {"mean_iou": 0.950, "saving": 0.800, "occupied_voxels": 10}),
        ]
    )
    assert selected is b
    assert metrics == {"mean_iou": 0.985, "saving": 0.200, "occupied_voxels": 80}


def test_run_failure_writes_truthful_staging_only_manifest(tmp_path: Path) -> None:
    output = tmp_path / "staging"
    with pytest.raises(DeliveryFailure) as caught:
        run_contract_delivery(
            contract=LiteralContract(),
            soft_artifact=tmp_path / "missing",
            staging_output=output,
            device="cpu",
            config=DeliveryConfig(core_resolution=4, delivery_resolution=8),
            renderer=_literal_renderer([]),
        )
    assert caught.value.status == "ARTIFACT_MUTATED"
    failure = json.loads((output / "staging_delivery_failure.json").read_text())
    assert failure["mode"] == "shadowart_geometry_only"
    assert failure["structural_status"] == "NOT_REQUESTED"
    assert failure["publication_status"] == "NOT_PUBLISHED"
    assert not list(output.glob("*.stl"))
    assert not list(output.glob("*.obj"))
    assert not list(tmp_path.glob(".staging.tmp-*"))


def test_shell_audit_rejects_boundary_and_nonmanifold_edges() -> None:
    open_box = trimesh.creation.box()
    open_box.update_faces(np.arange(len(open_box.faces) - 1))
    with pytest.raises(DeliveryFailure) as boundary:
        audit_shell_containment(open_box)
    assert boundary.value.details["boundary_edges"] > 0

    box = trimesh.creation.box()
    duplicate = trimesh.Trimesh(
        vertices=box.vertices.copy(),
        faces=np.vstack((box.faces, box.faces[0])),
        process=False,
    )
    with pytest.raises(DeliveryFailure) as nonmanifold:
        audit_shell_containment(duplicate)
    assert nonmanifold.value.details["nonmanifold_edges"] > 0


def test_oversize_controllable_simplification_is_reaudited() -> None:
    source = trimesh.creation.icosphere(subdivisions=2)
    calls: list[tuple[int, int]] = []

    def simplifier(mesh, target_faces):
        calls.append((len(mesh.faces), target_faces))
        return trimesh.creation.box()

    simplified, cap = enforce_triangle_cap(source, max_faces=20, simplifier=simplifier)
    assert calls == [(320, 20)]
    assert cap == {
        "simplification_applied": True,
        "source_faces": 320,
        "result_faces": 12,
        "max_faces": 20,
    }
    assert audit_shell_containment(simplified)["exterior_root_count"] == 1


def test_simplification_rejects_changed_cavity_topology() -> None:
    source = _nested_box_mesh()

    def destroys_cavity(mesh, target_faces):
        assert len(mesh.faces) == 24
        return trimesh.creation.box(extents=(10.0, 10.0, 10.0))

    with pytest.raises(DeliveryFailure) as caught:
        enforce_triangle_cap(source, max_faces=20, simplifier=destroys_cavity)
    assert caught.value.status == "TRIANGLE_TOPOLOGY_CHANGED"


def test_shell_audit_rejects_wrong_cavity_orientation_and_export_does_not_repair(
    tmp_path: Path,
) -> None:
    outer = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    outer.apply_translation((7.5, 7.5, 7.5))
    wrong_cavity = trimesh.creation.box(extents=(4.0, 4.0, 4.0))
    wrong_cavity.apply_translation((7.5, 7.5, 7.5))
    wrong = trimesh.util.concatenate((outer, wrong_cavity))
    with pytest.raises(DeliveryFailure) as caught:
        audit_shell_containment(wrong)
    assert caught.value.status == "SHELL_ORIENTATION_FAILED"
    with pytest.raises(DeliveryFailure):
        export_same_source_mesh(wrong, tmp_path)


def test_triangle_cap_and_public_checkpoint_helper_propagate_process_control() -> None:
    source = trimesh.creation.icosphere(subdivisions=2)

    def interrupted(mesh, target_faces):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        enforce_triangle_cap(source, max_faces=20, simplifier=interrupted)

    seed = np.ones((4, 4, 4), dtype=bool)
    with pytest.raises(ValueError, match="max_checkpoints"):
        carve_core_trajectory(
            seed,
            seed.astype(np.float32),
            seed,
            [np.ones((4, 4), dtype=bool)] * 2,
            replace(LiteralContract(), image_size=4),
            renderer=_literal_renderer([]),
            max_checkpoints=257,
            bridge_max_passes=1,
            bridge_radius=0,
        )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_snapshot(path: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (item.relative_to(path).as_posix(), _file_hash(item))
        for item in sorted(path.rglob("*"))
        if item.is_file()
    )


def _small_soft_artifact(tmp_path: Path) -> tuple[Path, LiteralContract, np.ndarray]:
    from PIL import Image

    artifact = tmp_path / "soft"
    artifact.mkdir()
    resolution = 16
    target = np.zeros((resolution, resolution), dtype=bool)
    target[4:12, 4:12] = True
    target_path = artifact / "gt_view_0.png"
    Image.fromarray(target.astype(np.uint8) * 255).save(target_path)
    probability_xyz = np.zeros((resolution,) * 3, dtype=np.float32)
    probability_xyz[6:10, 4:12, 4:12] = 1.0
    np.save(
        artifact / "occupancy_probability_core.npy",
        canonical_xyz_to_trainer_zyx(probability_xyz),
    )
    np.save(
        artifact / "occupancy_soft_proposal.npy",
        canonical_xyz_to_trainer_zyx(probability_xyz > 0),
    )
    contract = replace(
        LiteralContract(),
        views=((0.0, 0.0),),
        direct_view_roles=((None, None),),
        image_size=resolution,
        normalized_target_sha256=((target_path.name, _file_hash(target_path)),),
    )
    return (
        artifact,
        replace(contract, artifact_sha256=_artifact_snapshot(artifact)),
        target,
    )


def _small_delivery_config() -> DeliveryConfig:
    return DeliveryConfig(
        core_resolution=4,
        delivery_resolution=16,
        max_checkpoints=1,
        sdf_sigma=0.0,
    )


def test_no_valid_failure_atomically_preserves_only_whitelisted_diagnostics(
    tmp_path: Path,
) -> None:
    artifact, contract, target = _small_soft_artifact(tmp_path)
    output = tmp_path / "failed"

    def renderer(**kwargs):
        volume = kwargs["volume_xyz"]
        return np.zeros(volume.shape[:2], dtype=bool)

    with pytest.raises(DeliveryFailure) as caught:
        run_contract_delivery(
            contract=contract,
            soft_artifact=artifact,
            staging_output=output,
            device="cpu",
            config=_small_delivery_config(),
            renderer=renderer,
        )
    assert caught.value.status == "NO_VALID_CANDIDATE"
    assert sorted(path.name for path in output.iterdir()) == [
        "diagnostic_hull48.npy",
        "staging_delivery_failure.json",
    ]
    assert not list(output.rglob("*.stl"))
    assert not list(output.rglob("*.obj"))
    assert not list(tmp_path.glob(".failed.*"))


def test_mesh_raster_failure_preserves_diagnostic_without_candidate_artifacts(
    tmp_path: Path,
) -> None:
    artifact, contract, target = _small_soft_artifact(tmp_path)
    output = tmp_path / "mesh-failed"
    seen_delivery_checkpoint = 0

    def renderer(**kwargs):
        nonlocal seen_delivery_checkpoint
        volume = kwargs["volume_xyz"]
        rendered = np.transpose(volume.any(axis=0), (1, 0))
        if volume.shape == (16, 16, 16) and int(volume.sum()) == 448:
            seen_delivery_checkpoint += 1
            if seen_delivery_checkpoint > 9:
                return np.zeros_like(target)
        return rendered

    with pytest.raises(DeliveryFailure) as caught:
        run_contract_delivery(
            contract=contract,
            soft_artifact=artifact,
            staging_output=output,
            device="cpu",
            config=_small_delivery_config(),
            renderer=renderer,
        )
    assert caught.value.status == "MESH_RASTER_GATE_FAILED"
    assert sorted(path.name for path in output.iterdir()) == [
        "diagnostic_mesh_raster.npy",
        "staging_delivery_failure.json",
    ]
    assert not list(output.rglob("*.stl"))
    assert not list(output.rglob("*.obj"))
    assert not (output / "staging_delivery_manifest.json").exists()
    assert not list(tmp_path.glob(".mesh-failed.*"))


def test_full_run_with_injected_renderer_is_test_only_and_writes_no_mesh_candidate(
    tmp_path: Path,
) -> None:
    from PIL import Image

    artifact = tmp_path / "soft"
    output = tmp_path / "staging"
    artifact.mkdir()
    resolution = 16
    target = np.zeros((resolution, resolution), dtype=bool)
    target[4:12, 4:12] = True
    target_path = artifact / "gt_view_0.png"
    Image.fromarray(target.astype(np.uint8) * 255).save(target_path)
    probability_xyz = np.zeros((resolution,) * 3, dtype=np.float32)
    probability_xyz[6:10, 4:12, 4:12] = 1.0
    np.save(
        artifact / "occupancy_probability_core.npy",
        canonical_xyz_to_trainer_zyx(probability_xyz),
    )
    np.save(
        artifact / "occupancy_soft_proposal.npy",
        canonical_xyz_to_trainer_zyx(probability_xyz > 0),
    )
    contract = replace(
        LiteralContract(),
        views=((0.0, 0.0),),
        direct_view_roles=((None, None),),
        image_size=resolution,
        normalized_target_sha256=((target_path.name, _file_hash(target_path)),),
    )
    contract = replace(contract, artifact_sha256=_artifact_snapshot(artifact))

    def renderer(**kwargs):
        volume = kwargs["volume_xyz"]
        return np.transpose(volume.any(axis=0), (1, 0))

    result = run_contract_delivery(
        contract=contract,
        soft_artifact=artifact,
        staging_output=output,
        device="cpu",
        config=DeliveryConfig(
            core_resolution=4,
            delivery_resolution=resolution,
            max_checkpoints=1,
            sdf_sigma=0.0,
        ),
        renderer=renderer,
    )
    assert result["status"] == "STAGING_TEST_ONLY"
    assert result["publication_eligible"] is False
    assert result["renderer"]["mode"] == "injected_test_only"
    assert result["renderer"]["identity"].endswith("renderer")
    assert result["mode"] == "shadowart_geometry_only"
    assert result["structural_status"] == "NOT_REQUESTED"
    assert result["publication_status"] == "NOT_PUBLISHED"
    reraster = result["mesh"]["reraster_metrics"]
    assert reraster["mean_iou"] >= 0.95
    assert reraster["worst_iou"] >= 0.90
    assert 1e-6 <= reraster["saving"] <= 0.85
    assert reraster["erosion_survival"] >= 0.08
    assert reraster["component_count_6"] == 1
    assert reraster["surface_fraction_role"] == "diagnostic_only"
    assert (output / "staging_delivery_manifest.json").is_file()
    assert not list(output.glob("*.stl"))
    assert not list(output.glob("*.obj"))
    assert "selected_core" in result["stages"]
    assert "premesh_delivery" in result["stages"]
    assert result["selection"]["selected_checkpoint"] in ("seed", 0)
    assert not (output / "delivery.stl").exists()
    assert not list(tmp_path.glob(".staging.tmp-*"))


def test_run_rejects_artifact_staging_overlap_before_any_write(tmp_path: Path) -> None:
    artifact = tmp_path / "soft"
    artifact.mkdir()
    staging = artifact / "staging"
    with pytest.raises(DeliveryFailure) as caught:
        run_contract_delivery(
            contract=LiteralContract(),
            soft_artifact=artifact,
            staging_output=staging,
            device="cpu",
            config=DeliveryConfig(core_resolution=4, delivery_resolution=8),
            renderer=_literal_renderer([]),
        )
    assert caught.value.status == "PATH_OVERLAP"
    assert not staging.exists()
    assert list(artifact.iterdir()) == []


def test_run_rejects_existing_ready_directory_without_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "soft"
    artifact.mkdir()
    ready = tmp_path / "ready"
    ready.mkdir()
    marker = ready / "keep.txt"
    marker.write_text("existing-ready")
    with pytest.raises(DeliveryFailure) as caught:
        run_contract_delivery(
            contract=LiteralContract(),
            soft_artifact=artifact,
            staging_output=ready,
            device="cpu",
            config=DeliveryConfig(core_resolution=4, delivery_resolution=8),
            renderer=_literal_renderer([]),
        )
    assert caught.value.status == "STAGING_CONFLICT"
    assert marker.read_text() == "existing-ready"
    assert sorted(path.name for path in ready.iterdir()) == ["keep.txt"]


def test_production_ready_requires_frozen_48_256_resolutions(tmp_path: Path) -> None:
    with pytest.raises(DeliveryFailure) as caught:
        run_contract_delivery(
            contract=LiteralContract(),
            soft_artifact=tmp_path / "soft",
            staging_output=tmp_path / "staging",
            device="cpu",
            config=DeliveryConfig(core_resolution=4, delivery_resolution=8),
            renderer=None,
        )
    assert caught.value.status == "PRODUCTION_CONFIG_INVALID"
    assert not (tmp_path / "staging").exists()


@pytest.mark.parametrize(
    "override",
    (
        {"sdf_pad": 3},
        {"sdf_sigma": 0.0},
        {"sdf_level": 1.0},
    ),
)
def test_production_ready_requires_frozen_mesh_configuration(
    tmp_path: Path, override: dict[str, object]
) -> None:
    with pytest.raises(DeliveryFailure) as caught:
        run_contract_delivery(
            contract=LiteralContract(),
            soft_artifact=tmp_path / "soft",
            staging_output=tmp_path / "staging",
            device="cpu",
            config=DeliveryConfig(**override),
            renderer=None,
        )
    assert caught.value.status == "PRODUCTION_CONFIG_INVALID"
    assert not (tmp_path / "staging").exists()


def test_real_generic_export_reload_inverse_raster_passes_same_contract(
    tmp_path: Path,
) -> None:
    contract = replace(
        LiteralContract(),
        views=((0.0, 0.0),),
        direct_view_roles=((None, None),),
        image_size=16,
    )
    volume = np.zeros((16, 16, 16), dtype=bool)
    volume[3:13, 4:12, 4:12] = True
    mesh = padded_sdf_mesh(volume, pad=2, sigma=0.0)
    reference = rasterize_mesh_with_cavities(mesh, volume.shape)
    targets = render_contract(reference, contract, renderer=None, device="cpu")
    exports = export_same_source_mesh(mesh, tmp_path)
    audits = audit_reloaded_fabrication_contract(
        exports,
        volume.shape,
        targets,
        contract,
        "cpu",
        int(reference.sum() * 2),
        DeliveryConfig(core_resolution=4, delivery_resolution=16, sdf_sigma=0.0),
    )
    assert set(audits) == {"stl", "obj"}
    for audit in audits.values():
        assert audit["raster_metrics"]["mean_iou"] >= 0.95
        assert audit["raster_metrics"]["component_count_6"] == 1


@pytest.mark.filterwarnings(
    "ignore:NDCGridRaysampler is deprecated:PendingDeprecationWarning"
)
def test_real_direct_sadekar_renderer_none_branch_is_deterministic() -> None:
    contract = replace(
        LiteralContract(),
        renderer_family="sadekar_voxel_tfs_mirror_orthographic",
        views=(
            (0.0, 0.0),
            (90.0, 0.0),
            (0.0, 90.0),
            (180.0, 0.0),
            (270.0, 0.0),
            (0.0, -90.0),
        ),
        direct_view_roles=(
            (0, "ud"),
            (2, "tudlr"),
            (1, "udlr"),
            (0, "udlr"),
            (2, "tlr"),
            (1, "ud"),
        ),
        image_size=8,
        zdist=1.7,
        volume_extent=1.7,
        ray_samples=150,
    )
    volume = np.zeros((8, 8, 8), dtype=bool)
    volume[1:6, 2:7, 1:5] = True
    first = render_contract(volume, contract, renderer=None, device="cpu")
    second = render_contract(volume, contract, renderer=None, device="cpu")
    assert len(first) == 6
    assert all(mask.shape == (8, 8) and mask.dtype == bool for mask in first)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
