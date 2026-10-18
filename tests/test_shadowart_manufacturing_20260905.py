import json

import numpy as np
import pytest
from scipy import ndimage as ndi


def dumbbell():
    v = np.zeros((40, 40, 40), bool)
    v[3:15, 10:30, 10:30] = True
    v[25:37, 10:30, 10:30] = True
    v[15:25, 20, 20] = True
    return v


def test_local_neck_gate_catches_large_lobes_despite_high_global_survival():
    from experiments import shadowart_manufacturing_20260905 as m
    report = m.neck_audit(dumbbell(), pitch_mm=.5, minimum_lobe_mm3=10.)
    assert report["component_count_6"] == 1
    assert report["erosion_survival"] > .5
    assert report["significant_eroded_components"] == 2
    assert report["robust_neck_gate"] is False


def test_local_thickening_connects_eroded_lobes_without_global_fill_or_source_mutation():
    from experiments import shadowart_manufacturing_20260905 as m
    v = dumbbell()
    v[8:10, 15:17, 15:17] = False  # Preserve a closed internal cavity.
    source = v.copy()
    repaired, locked, audit = m.fortify_necks(v, pitch_mm=.5, minimum_lobe_mm3=10.)
    assert np.array_equal(v, source)
    assert np.all(repaired[v])
    assert not repaired[8:10, 15:17, 15:17].any()
    assert m.neck_audit(repaired, pitch_mm=.5, minimum_lobe_mm3=10.)["robust_neck_gate"]
    assert np.all(repaired[locked])
    assert audit["added_voxels"] > 0


def test_profile_resolution_expands_inheritance_and_gcode_includes(tmp_path):
    from experiments import shadowart_manufacturing_20260905 as m
    (tmp_path / "base.json").write_text(json.dumps({"name": "base", "type": "machine", "speed": "12", "nested": ["a"]}))
    (tmp_path / "start.json").write_text(json.dumps({"machine_start_gcode": "G28"}))
    (tmp_path / "a1.json").write_text(json.dumps({"name": "a1", "type": "machine", "inherits": "base", "include": ["start"], "speed": "20"}))
    config, hashes = m.resolve_profile(tmp_path, "a1")
    assert config["speed"] == "20"
    assert config["nested"] == ["a"]
    assert config["machine_start_gcode"] == "G28"
    assert "inherits" not in config and "include" not in config
    assert set(hashes) == {"base.json", "start.json", "a1.json"}


def test_profile_cycle_fails_closed(tmp_path):
    from experiments import shadowart_manufacturing_20260905 as m
    for name, parent in (("a", "b"), ("b", "a")):
        (tmp_path / f"{name}.json").write_text(json.dumps({"inherits": parent}))
    with pytest.raises(ValueError, match="cycle"):
        m.resolve_profile(tmp_path, "a")


def test_zero_exit_with_floating_regions_is_not_print_ready():
    from experiments import shadowart_manufacturing_20260905 as m
    report = {"return_code": 0, "sliced_plates": [{"warning_message": "object has floating regions"}]}
    result = m.slicing_audit(report)
    assert result["slicing_completed"] is True
    assert result["print_preparation_passed"] is False


def test_candidate_selection_refuses_thin_best_iou():
    from experiments import shadowart_manufacturing_20260905 as m
    rows = [
        {"id": "thin", "mean_iou": .98, "worst_iou": .96, "robust_neck_gate": False, "voxels": 100},
        {"id": "robust", "mean_iou": .96, "worst_iou": .94, "robust_neck_gate": True, "voxels": 200},
        {"id": "bad_shadow", "mean_iou": .89, "worst_iou": .85, "robust_neck_gate": True, "voxels": 80},
    ]
    assert m.select_manufacturing_candidate(rows)["id"] == "robust"


def test_prepared_slicer_profiles_enable_external_support_and_keep_material_density(tmp_path):
    from experiments import shadowart_manufacturing_20260905 as m
    names = {"machine": "Bambu Lab A1 0.4 nozzle", "process": "0.20mm Standard @BBL A1",
             "filament": "Bambu PLA Basic @BBL A1"}
    for kind, name in names.items():
        folder = tmp_path / "profiles" / kind
        folder.mkdir(parents=True)
        data = {"name": name, "type": kind, "filament_density": ["1.26"]} if kind == "filament" else {"name": name, "type": kind}
        (folder / f"{name}.json").write_text(json.dumps(data))
    out = tmp_path / "resolved"
    paths, report = m.prepare_slicer_profiles(tmp_path / "profiles", out)
    process = json.loads(paths["process"].read_text())
    assert process["enable_support"] == "1"
    assert process["support_on_build_plate_only"] == "1"
    assert process["layer_height"] == "0.2"
    assert json.loads(paths["filament"].read_text())["filament_density"] == ["1.26"]
    assert report["support_policy"] == "external_build_plate_only_removable"


def test_effective_slicer_config_must_match_authorized_printer_and_support():
    from experiments import shadowart_manufacturing_20260905 as m
    config = {"printer_model": "A1", "nozzle_diameter": ["0.4"],
              "layer_height": "0.2", "enable_support": "1",
              "support_on_build_plate_only": "1", "support_type": "tree(auto)",
              "filament_density": ["1.26"], "filament_type": ["PLA"],
              "wall_loops": "3", "machine_max_acceleration_x": ["6000", "6000"]}
    assert m.audit_effective_settings(config)["passed"]
    assert m.audit_effective_settings(dict(config, printer_model="Bambu Lab A1"))["passed"]
    assert not m.audit_effective_settings(dict(config, filament_density=["0"]))["passed"]
    assert not m.audit_effective_settings(dict(config, enable_support="0"))["passed"]
    assert not m.audit_effective_settings(dict(config, printer_model="X1C"))["passed"]


def test_failed_preparation_cannot_publish_canonical_files(tmp_path):
    from experiments import shadowart_manufacturing_20260905 as m
    stage = tmp_path / "staging"
    stage.mkdir()
    (stage / "candidate_fabrication.obj").write_text("diagnostic only")
    with pytest.raises(d_exception()):
        m.publish_preparation(stage, tmp_path, {"status": "FAILED"})
    assert not (tmp_path / "delivery_postprocessed.obj").exists()


def d_exception():
    from experiments.multiview_contract_delivery_20260902 import DeliveryFailure
    return DeliveryFailure


def test_geometry_replay_refuses_unsuccessful_run_before_loading_mesh(tmp_path):
    from experiments import run_shadowart_manufacturing_20260905 as runner
    (tmp_path / "MULTIVIEW_EXPORT.json").write_text(json.dumps({"status": "FAILED"}))
    with pytest.raises(d_exception(), match="successful"):
        runner.load_geometry(tmp_path, device="cpu")


def test_cli_requires_explicit_external_support_authorization():
    from experiments import run_shadowart_manufacturing_20260905 as runner
    with pytest.raises(SystemExit):
        runner.parse_args(["--geometry-run", "/tmp/run", "--output", "/tmp/new"])


def test_sliced_archive_must_contain_checked_gcode_and_real_support(tmp_path):
    import hashlib
    import zipfile
    from experiments import shadowart_manufacturing_20260905 as m
    def archive(name, content):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("Metadata/plate_1.gcode", content)
            z.writestr("Metadata/plate_1.gcode.md5", hashlib.md5(content).hexdigest())
        return path
    path = archive("supported.3mf", b"; FEATURE: Support\nG1 X12 E3\n")
    assert m.audit_sliced_archive(path)["support_toolpaths_present"]
    with pytest.raises(d_exception()):
        m.audit_sliced_archive(archive("unsupported.3mf", b"G1 X1 E1\n"))


def publication_fixture(tmp_path):
    from experiments import multiview_contract_delivery_20260902 as d
    stage = tmp_path / "stage"
    output = tmp_path / "out"
    stage.mkdir()
    output.mkdir()
    files = {role: f"candidate_{role}.{'obj' if role.endswith('obj') else 'stl'}"
             for role in ("fabrication_obj", "fabrication_stl", "outer_preview_obj", "outer_preview_stl")}
    for name in files.values():
        (stage / name).write_bytes(b"isolated transaction fixture")
    (stage / "slicer").mkdir()
    sliced = stage / "slicer" / "shadowart_supported.gcode.3mf"
    sliced.write_bytes(b"isolated transaction fixture")
    report = {"status": "PRINT_PREPARATION_VALIDATED", "mesh_neck": {"robust_neck_gate": True},
              "slicing": {"print_preparation_passed": True}, "effective_settings": {"passed": True},
              "mesh": {"serialized_reload_contract_qa": {"obj": {}, "stl": {}},
                       "exports": {"files": files, "file_sha256": {r: d._hash(stage / p) for r, p in files.items()}}},
              "sliced_3mf_sha256": d._hash(sliced)}
    return stage, output, report


def test_partial_copy_failure_rolls_back_attempted_canonical_file(tmp_path, monkeypatch):
    import shutil
    from experiments import shadowart_manufacturing_20260905 as m
    stage, output, report = publication_fixture(tmp_path)
    def partial_copy(source, destination):
        destination.write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(shutil, "copy2", partial_copy)
    with pytest.raises(OSError, match="disk full"):
        m.publish_preparation(stage, output, report)
    assert list(output.iterdir()) == []


def test_manifest_write_failure_rolls_back_all_canonical_files(tmp_path, monkeypatch):
    from experiments import shadowart_manufacturing_20260905 as m
    stage, output, report = publication_fixture(tmp_path)
    def failed_manifest(path, data):
        path.write_bytes(b"partial manifest")
        raise OSError("manifest write failed")
    monkeypatch.setattr(m.d, "_write", failed_manifest)
    with pytest.raises(OSError, match="manifest write failed"):
        m.publish_preparation(stage, output, report)
    assert list(output.iterdir()) == []


def test_successful_publication_commits_matching_manifest(tmp_path):
    from experiments import shadowart_manufacturing_20260905 as m
    stage, output, report = publication_fixture(tmp_path)
    expected = m.publish_preparation(stage, output, report)
    manifest = json.loads((output / "MANUFACTURING.json").read_text())
    assert manifest["status"] == "PRINT_PREPARATION_VALIDATED"
    assert manifest["canonical_sha256"] == expected
    assert all(m.d._hash(output / name) == digest for name, digest in expected.items())


def test_interrupt_during_publication_removes_partial_delivery(tmp_path, monkeypatch):
    import shutil
    from experiments import shadowart_manufacturing_20260905 as m
    stage, output, report = publication_fixture(tmp_path)
    def interrupted_copy(source, destination):
        destination.write_bytes(b"partial")
        raise KeyboardInterrupt()
    monkeypatch.setattr(shutil, "copy2", interrupted_copy)
    with pytest.raises(KeyboardInterrupt):
        m.publish_preparation(stage, output, report)
    assert list(output.iterdir()) == []


def test_interrupt_just_after_manifest_rename_rolls_back_success_marker(tmp_path, monkeypatch):
    from pathlib import Path
    from experiments import shadowart_manufacturing_20260905 as m
    stage, output, report = publication_fixture(tmp_path)
    original = Path.replace
    def interrupted_replace(source, destination):
        original(source, destination)
        raise KeyboardInterrupt()
    monkeypatch.setattr(Path, "replace", interrupted_replace)
    with pytest.raises(KeyboardInterrupt):
        m.publish_preparation(stage, output, report)
    assert list(output.iterdir()) == []
