"""Continuous-mesh delivery protocol tests using real geometry where practical."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh
from scipy.spatial import Delaunay


def _target(tmp_path):
    pixels = np.zeros((32, 32), np.uint8)
    pixels[8:24, 8:24] = 255
    path = tmp_path / "square.png"
    Image.fromarray(pixels).save(path)
    return path


def _audit(tmp_path, mesh, protocol):
    from experiments.shadowart_vnext.contracts import TargetSet, build_contract
    from experiments.shadowart_vnext.delivery import audit_serialized_mesh

    return audit_serialized_mesh(
        mesh,
        build_contract(1, resolution=128, image_size=256),
        TargetSet.from_paths([_target(tmp_path)]),
        tmp_path / "audit",
        geometry_settings={
            "fixture": "analytic_cube",
            "acceptance_protocol": protocol,
            "longest_mm": 100.0,
            "min_wall_mm": 0.9,
            "min_neck_mm": 1.35,
        },
    )[0]


def _cross_prism_and_target(tmp_path):
    """Watertight extrusion of a literal 9-cell cross; no production helper reuse."""
    half, arm = 0.7, 0.7 / 3.0
    polygon = np.array([
        [-half, -arm], [-arm, -arm], [-arm, -half], [arm, -half],
        [arm, -arm], [half, -arm], [half, arm], [arm, arm],
        [arm, half], [-arm, half], [-arm, arm], [-half, arm],
    ])
    triangles = Delaunay(polygon).simplices
    centers = polygon[triangles].mean(axis=1)
    # Literal cross-membership test retains only triangles inside the concave polygon.
    triangles = triangles[(np.abs(centers[:, 0]) <= arm) |
                          (np.abs(centers[:, 1]) <= arm)]
    mesh = trimesh.creation.extrude_triangulation(polygon, triangles, height=1.4)
    mesh.apply_translation([0, 0, -0.7])
    pixels = np.zeros((9, 9), np.uint8)
    pixels[3:6, :] = 255
    pixels[:, 3:6] = 255
    path = tmp_path / "cross.png"
    Image.fromarray(pixels).save(path)
    return mesh, path


def _audit_cross(tmp_path, protocol):
    from experiments.shadowart_vnext.contracts import TargetSet, build_contract
    from experiments.shadowart_vnext.delivery import audit_serialized_mesh

    mesh, target = _cross_prism_and_target(tmp_path)
    return audit_serialized_mesh(
        mesh,
        build_contract(1, resolution=128, image_size=256),
        TargetSet.from_paths([target]),
        tmp_path / "audit",
        geometry_settings={
            "fixture": "analytic_cross_prism",
            "acceptance_protocol": protocol,
            "longest_mm": 100.0,
            "min_wall_mm": 0.9,
            "min_neck_mm": 1.35,
        },
    )[0]


def test_continuous_mesh_protocol_accepts_real_cross_when_native128_raster_fails(tmp_path):
    """Catches accidentally retaining native raster IoU as the continuous gate."""
    report = _audit_cross(tmp_path, "continuous_mesh_v1")

    assert report["passed"], report
    assert report["acceptance_protocol"] == "continuous_mesh_v1"
    assert report["serialized_mesh_iou_authoritative"] is True
    assert report["raster_iou_is_gate"] is False
    assert report["native_raster_resolution"] == 128
    for row in report["serialized"].values():
        assert row["quality"]["passed"] is True
        assert row["raster_quality"]["passed"] is False
        assert row["raster_iou_is_gate"] is False
        assert row["component_count_6"] == 1
        assert row["native_thickness"]["authoritative"] is False
        fine = row["fine_mesh_revoxelized_512"]
        assert fine["status"] == "PASS"
        assert fine["authoritative"] is True
        assert fine["resolution"] == 512
        assert fine["component_count_6"] == 1
        assert fine["thickness"]["passed"] is True
        assert row["passed"] is True
        assert row["mesh_minus_native_raster_iou"]["diagnostic_scope"] == (
            "geometry_discretization_loss_vs_actual_serialized_mesh"
        )
        assert row["mesh_minus_native_raster_iou"]["per_view"]


def test_same_real_cross_remains_failed_under_default_strict_protocol(tmp_path):
    """Catches relaxing the missing-key legacy protocol."""
    report = _audit_cross(tmp_path, "strict_voxel_v1")
    assert report["passed"] is False
    assert report["acceptance_protocol"] == "strict_voxel_v1"
    assert report["raster_iou_is_gate"] is True
    assert any(not row["raster_quality"]["passed"] for row in report["serialized"].values())


@pytest.mark.parametrize("kind", ["open", "disconnected", "thin"])
def test_continuous_protocol_cannot_waive_real_topology_or_physical_failures(tmp_path, kind):
    """Catches treating continuous IoU as a waiver for topology/6CC/thickness."""
    if kind == "open":
        mesh = trimesh.creation.box(extents=[1.4] * 3)
        mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    elif kind == "disconnected":
        first = trimesh.creation.box(extents=[0.65] * 3)
        first.apply_translation([-0.38, 0, 0])
        second = trimesh.creation.box(extents=[0.65] * 3)
        second.apply_translation([0.38, 0, 0])
        mesh = trimesh.util.concatenate([first, second])
    else:
        mesh = trimesh.creation.box(extents=[1.4, 1.4, 0.005])

    report = _audit(tmp_path, mesh, "continuous_mesh_v1")
    assert report["passed"] is False
    if kind == "open":
        assert any(not row["topology"]["passed"] for row in report["serialized"].values())
        assert all(row["fine_mesh_revoxelized_512"]["status"] == "NOT_EVALUATED"
                   for row in report["serialized"].values())
    elif kind == "disconnected":
        assert any("MULTIPLE_OUTER_SHELLS" in row["topology"]["reasons"]
                   for row in report["serialized"].values())
        assert all(row["fine_mesh_revoxelized_512"]["status"] == "NOT_EVALUATED"
                   for row in report["serialized"].values())
    else:
        assert any(not row["fine_mesh_revoxelized_512"]["thickness"]["passed"]
                   for row in report["serialized"].values())


@pytest.mark.parametrize("protocol", ["continuous_mesh", "STRICT", 1, None])
def test_serialized_audit_rejects_unknown_explicit_protocol(tmp_path, protocol):
    """Catches silently treating malformed protocol labels as strict or continuous."""
    from experiments.shadowart_vnext.contracts import TargetSet, build_contract
    from experiments.shadowart_vnext.delivery import audit_serialized_mesh

    with pytest.raises(ValueError, match="UNKNOWN_ACCEPTANCE_PROTOCOL"):
        audit_serialized_mesh(
            trimesh.creation.box(extents=[1.4] * 3),
            build_contract(1, resolution=128, image_size=256),
            TargetSet.from_paths([_target(tmp_path)]),
            tmp_path / "audit",
            geometry_settings={"acceptance_protocol": protocol},
        )


def _publish_fixture(tmp_path, *, protocol="continuous_mesh_v1"):
    from experiments.shadowart_vnext.contracts import digest_json, file_hash

    files = {}
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    for ext in ("obj", "stl"):
        path = tmp_path / f"candidate.{ext}"
        mesh.export(path)
        files[ext] = path
    geometry = {"acceptance_protocol": protocol, "longest_mm": 100.0,
                "min_wall_mm": 0.9, "min_neck_mm": 1.35}
    topology = {"passed": True, "watertight": True, "winding_consistent": True,
                "self_intersection": {"status": "PASS"}}
    evaluations = [{"image_size": size, "per_view_iou": [0.99]} for size in (256, 512)]
    serialized = {
        ext: {
            "passed": True,
            "sha256": file_hash(path),
            "topology": topology,
            "quality": {"passed": True},
            "raster_quality": {"passed": False},
            "raster_evaluations": [{"image_size": 256, "per_view_iou": [0.95]}],
            "raster_iou_is_gate": False,
            "native_nonempty": True,
            "component_count_6": 1,
            "native_thickness": {"authoritative": False, "passed": True},
            "fine_mesh_revoxelized_512": {
                "status": "PASS", "passed": True, "authoritative": True,
                "resolution": 512, "nonempty": True, "component_count_6": 1,
                "thickness": {"passed": True},
            },
            "mesh_minus_native_raster_iou": {
                "diagnostic_scope": "geometry_discretization_loss_vs_actual_serialized_mesh",
                "per_view": [{"image_size": 256, "mesh_minus_raster_per_view_iou": [0.04]}],
            },
        }
        for ext, path in files.items()
    }
    qa = {
        "passed": True,
        "acceptance_protocol": protocol,
        "serialized_mesh_iou_authoritative": True,
        "raster_iou_is_gate": False,
        "native_resolution": 128,
        "native_optimization_resolution": 128,
        "image_size": 256,
        "independent_serialized_mesh": True,
        "evaluations": evaluations,
        "serialized": serialized,
        "mesh_sha256": serialized["stl"]["sha256"],
        "camera_digest": "camera",
        "target_hashes": ["target"],
        "geometry_digest": digest_json(geometry),
        "geometry_settings": geometry,
    }
    checks = {key: True for key in (
        "one_body", "watertight", "winding_consistent", "self_intersections_clear",
        "thickness_pass", "slicer_pass", "support_removal_pass",
    )}
    manufacturing = {
        "passed": True,
        "checks": checks,
        "mesh_sha256": qa["mesh_sha256"],
        "camera_digest": qa["camera_digest"],
        "target_hashes": qa["target_hashes"],
        "geometry_digest": qa["geometry_digest"],
        "acceptance_protocol": protocol,
        "native_optimization_resolution": 128,
    }
    return qa, manufacturing, files


def test_publish_accepts_bound_native128_continuous_protocol_and_declares_scope(tmp_path):
    """Catches the old hard-coded native-256 publication guard."""
    from experiments.shadowart_vnext.delivery import publish_delivery

    (tmp_path / "diagnostics").mkdir()
    qa, manufacturing, files = _publish_fixture(tmp_path)
    result = publish_delivery(tmp_path, qa, manufacturing, files)
    report = json.loads((Path(result["directory"]) / "DELIVERY.json").read_text())
    assert report["acceptance_protocol"] == "continuous_mesh_v1"
    assert report["native_optimization_resolution"] == 128
    assert report["physical_print_validated"] is False


@pytest.mark.parametrize("mutation", [
    "missing_flags", "geometry_relabel", "manufacturing_mismatch", "bad_image_size",
])
def test_publish_rejects_fabricated_or_relabelled_continuous_qa(tmp_path, mutation):
    """Catches promotion from passed-only or protocol-relabelled QA dictionaries."""
    from experiments.shadowart_vnext.contracts import digest_json
    from experiments.shadowart_vnext.delivery import publish_delivery

    qa, manufacturing, files = _publish_fixture(tmp_path)
    if mutation == "missing_flags":
        for key in ("serialized_mesh_iou_authoritative", "raster_iou_is_gate"):
            qa.pop(key)
    elif mutation == "geometry_relabel":
        qa["geometry_settings"] = {"acceptance_protocol": "strict_voxel_v1"}
        qa["geometry_digest"] = digest_json(qa["geometry_settings"])
        manufacturing["geometry_digest"] = qa["geometry_digest"]
    elif mutation == "manufacturing_mismatch":
        manufacturing["acceptance_protocol"] = "strict_voxel_v1"
    else:
        qa["image_size"] = 128
    with pytest.raises(ValueError):
        publish_delivery(tmp_path, qa, manufacturing, files)
    assert not (tmp_path / "delivery").exists()


def test_cli_defaults_strict_and_continuous_mode_is_explicit_and_bounded(tmp_path):
    """Catches an implicit default change or acceptance at unsupported settings."""
    from experiments.run_shadowart_vnext import parse_args
    from experiments.run_shadowart_fidelity_probe import parse_args as parse_probe

    base = ["--silhouettes", "x.png", "--output", str(tmp_path / "run")]
    assert parse_args(base).delivery_protocol == "strict_voxel_v1"
    assert parse_args(base + ["--delivery-protocol", "continuous_mesh_v1",
                              "--resolution", "128", "--image-size", "256"]).delivery_protocol == (
        "continuous_mesh_v1"
    )
    # Noncanonical settings remain parsable for diagnostic stages, but cannot publish.
    assert parse_args(base + ["--delivery-protocol", "continuous_mesh_v1",
                              "--resolution", "64", "--image-size", "128"]).resolution == 64
    probe = ["--silhouettes", "x.png", "--soft-artifact-dir", str(tmp_path / "soft"),
             "--output", str(tmp_path / "probe")]
    assert parse_probe(probe).delivery_protocol == "strict_voxel_v1"
    assert parse_probe(probe + ["--delivery-protocol", "continuous_mesh_v1"]).delivery_protocol == (
        "continuous_mesh_v1"
    )


def test_native128_continuous_run_records_experimental_scope_from_preflight(tmp_path):
    """Catches omitting experimental/non-paper scope before the delivery stage."""
    from experiments.run_shadowart_vnext import main

    target = _target(tmp_path)
    output = tmp_path / "run"
    assert main(["--silhouettes", str(target), "--output", str(output),
                 "--stage", "preflight", "--device", "cpu", "--layout-candidates", "1",
                 "--resolution", "128", "--image-size", "256",
                 "--delivery-protocol", "continuous_mesh_v1"]) == 0
    run = json.loads((output / "RUN.json").read_text())
    assert run["acceptance_protocol"] == "continuous_mesh_v1"
    assert run["experimental_vnext"] is True
    assert run["engineering_only"] is False
    assert run["structural_status"] == "NOT_REQUESTED"
    assert run["paper_method_equivalence"] is False


def test_finalizer_rejects_protocol_relabel_before_fresh_audit(tmp_path, monkeypatch):
    """Catches finalization deriving its protocol from new CLI state or stale QA alone."""
    from tests.test_shadowart_vnext_finalize import fixture, install_gates
    from experiments.shadowart_vnext.contracts import digest_json
    import experiments.shadowart_vnext.finalize as finalizer

    args, qa, hashes = fixture(tmp_path)
    geometry = json.loads((args.finalize_run / "GEOMETRY_CONFIG.json").read_text())
    geometry["acceptance_protocol"] = "continuous_mesh_v1"
    (args.finalize_run / "GEOMETRY_CONFIG.json").write_text(json.dumps(geometry))
    # Relabel the old QA but omit continuous protocol evidence: this must not be promotable.
    qa.update(acceptance_protocol="continuous_mesh_v1", native_resolution=128,
              native_optimization_resolution=128, image_size=256,
              geometry_digest=digest_json(geometry))
    (args.finalize_run / "QA.json").write_text(json.dumps(qa))
    manufacturing = json.loads((args.finalize_run / "MANUFACTURING.json").read_text())
    manufacturing["geometry_digest"] = qa["geometry_digest"]
    (args.finalize_run / "MANUFACTURING.json").write_text(json.dumps(manufacturing))
    events = []
    install_gates(monkeypatch, finalizer, qa, hashes, events)

    assert finalizer.finalize_existing(
        args, prepared_output=args.output, source_hashes={"finalize.py": "a" * 64}
    ) == 2
    assert "mesh" not in events
