"""Synthetic real G-code/3MF artifacts exercise audit behavior, not CLI mocks."""
import hashlib
import importlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import trimesh


def module():
    source = Path(__file__).parents[1] / "experiments/shadowart_vnext/slicer.py"
    assert source.is_file(), "strict slicer audit implementation is missing"
    return importlib.import_module("experiments.shadowart_vnext.slicer")


def settings():
    return {"printer_model": "Bambu Lab A1", "nozzle_diameter": ["0.4"],
            "filament_type": ["PLA"], "filament_diameter": ["1.75"],
            "line_width": "0.42", "initial_layer_line_width": "0.5",
            "outer_wall_line_width": "0.42", "inner_wall_line_width": "0.45",
            "top_surface_line_width": "0.42", "sparse_infill_line_width": "0.45",
            "internal_solid_infill_line_width": "0.42", "support_line_width": "0.42",
            "layer_height": "0.2", "initial_layer_print_height": "0.2", "wall_loops": "3",
            "wall_generator": "arachne", "brim_type": "outer_only", "brim_width": "8",
            "sparse_infill_density": "15%", "sparse_infill_pattern": "grid",
            "enable_support": "1", "support_type": "tree(auto)",
            "support_on_build_plate_only": "1"}


def gcode():
    config = "\n".join(f"; {k} = {v[0] if isinstance(v, list) else v}" for k, v in settings().items())
    return f"; CONFIG_BLOCK_START\n{config}\n; CONFIG_BLOCK_END\n" + """G21
G90
M83
; Z_HEIGHT: 0.2
; LAYER_HEIGHT: 0.2
; FEATURE: Outer wall
G0 X19.21 Y29.21 Z0.2
G1 X20.79 E0.05
G1 Y30.79 E0.05
G1 X19.21 E0.05
G1 Y29.21 E0.05
; FEATURE: Support interface
G0 X21.1 Y29.3
G1 Y30.7 E0.05
; Z_HEIGHT: 0.4
; FEATURE: Outer wall
G0 X19.21 Y29.21 Z0.4
G1 X20.79 E0.05
G1 Y30.79 E0.05
G1 X19.21 E0.05
G1 Y29.21 E0.05
"""


def artifact(tmp_path, *, transform="1 0 0 0 1 0 0 0 1 20 30 0", warning="", corrupt=False):
    mesh = trimesh.creation.box(extents=(2., 2., .4))
    mesh.apply_translation([0., 0., .2])
    source = tmp_path / "source.stl"
    mesh.export(source)
    vertices = "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in mesh.vertices)
    triangles = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in mesh.faces)
    model = f'''<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources><object id="1" type="model"><mesh><vertices>{vertices}</vertices><triangles>{triangles}</triangles></mesh></object></resources><build><item objectid="1" transform="{transform}"/></build></model>'''
    text = gcode()
    archive = tmp_path / "test.gcode.3mf"
    with ZipFile(archive, "w") as z:
        z.writestr("3D/3dmodel.model", model)
        z.writestr("Metadata/plate_1.gcode", text)
        z.writestr("Metadata/plate_1.gcode.md5", "0" * 32 if corrupt else hashlib.md5(text.encode()).hexdigest())
        z.writestr("Metadata/project_settings.config", json.dumps(settings()))
        z.writestr("Metadata/slice_info.config", f'<config><plate>{warning}</plate></config>')
    effective = tmp_path / "effective.json"
    effective.write_text(json.dumps(settings()))
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"return_code": 0, "sliced_plates": [{"warning_message": ""}]}))
    return source, archive, effective, result


def test_effective_profile_checks_widths_infill_and_every_required_control():
    assert module().audit_effective_settings(settings())["passed"]
    changed = settings()
    changed["outer_wall_line_width"] = "0.3"
    assert not module().audit_effective_settings(changed)["passed"]
    del changed["sparse_infill_density"]
    report = module().audit_effective_settings(changed)
    assert not report["checks"]["sparse_infill_density"]


def test_absolute_extrusion_retractions_resets_and_purges_are_not_deposit_paths():
    text = """G90
M82
G92 E0
; FEATURE: Outer wall
G0 X0 Y0 Z0.2
G1 X1 E1
G1 X2 E0.5
G92 E0
G1 E1
G1 X3 E1
G1 X4 E1.1
"""
    report = module().audit_gcode(text)
    assert report["positive_model_segments"] == 2
    assert report["commanded_model_extrusion_mm"] == pytest.approx(1.1)
    assert report["stationary_positive_extrusions"] == 1


def test_positive_extrusion_arcs_are_tessellated_and_unknown_arc_modes_reviewed():
    text = "G90\nM83\n; FEATURE: Outer wall\nG0 X1 Y0 Z0.2\nG3 X0 Y1 I-1 J0 E0.05\n"
    report = module().audit_gcode(text)
    assert report["positive_model_segments"] > 1
    assert report["model_path_length_mm"] == pytest.approx(np.pi / 2, abs=.03)
    unknown = module().audit_gcode(text.replace("I-1 J0", "R1"))
    assert unknown["status"] == "REVIEW_REQUIRED"


def test_moving_unretraction_is_not_counted_as_material_coverage():
    text = "G90\nM83\n; FEATURE: Outer wall\nG0 X0 Y0 Z0.2\nG1 E-0.5\nG1 X1 E0.3\nG1 X2 E0.4\n"
    report = module().audit_gcode(text)
    assert report["positive_model_segments"] == 1
    assert report["commanded_model_extrusion_mm"] == pytest.approx(.2)


def test_archive_geometry_and_source_are_compared_with_explicit_build_transform(tmp_path):
    args = artifact(tmp_path)
    report = module().audit_sliced_artifacts(*args)
    assert report["checks"]["archive_integrity"]["passed"]
    assert report["checks"]["archive_model"]["passed"]
    assert np.allclose(report["checks"]["archive_model"]["build_transform"][:3],
                       [[1, 0, 0, 20], [0, 1, 0, 30], [0, 0, 1, 0]])
    assert not report["passed"]
    assert report["support_removal_pass"] is False
    assert report["checks"]["support_removal"]["status"] == "REVIEW_REQUIRED"


def test_stretch_in_archive_and_corrupt_gcode_hash_fail(tmp_path):
    report = module().audit_sliced_artifacts(*artifact(tmp_path, transform="2 0 0 0 1 0 0 0 1 20 30 0", corrupt=True))
    assert not report["checks"]["archive_integrity"]["passed"]
    assert not report["checks"]["archive_model"]["passed"]
    assert report["status"] == "FAIL"


def test_unknown_warnings_are_never_silently_discarded(tmp_path):
    args = artifact(tmp_path, warning='<warning msg="novel_geometry_condition" level="2"/>')
    report = module().audit_sliced_artifacts(*args)
    assert report["checks"]["warnings"]["status"] == "REVIEW_REQUIRED"
    assert any("novel_geometry_condition" in row["message"] for row in report["checks"]["warnings"]["warnings"])


def test_support_review_is_artifact_bound_and_cannot_override_unknown_correspondence():
    valid = {"schema": "shadowart_support_review_v1", "reviewer": "Fixture Reviewer",
             "reviewed_at": "2026-09-06T12:00:00+08:00", "mesh_sha256": "a" * 64,
             "gcode_sha256": "b" * 64, "archive_sha256": "c" * 64,
             "supports_accessible": True, "supports_removable": True,
             "all_contacts_reviewed": True, "decision": "accessible_and_removable",
             "notes": "All interface contacts have an external removal route."}
    assert module().audit_support_review(valid, "a" * 64, "b" * 64, "c" * 64)["passed"]
    assert not module().audit_support_review(valid, "d" * 64, "b" * 64, "c" * 64)["passed"]
    assert not module().audit_support_review(None, "a" * 64, "b" * 64, "c" * 64)["passed"]
    rejected = {**valid, "supports_removable": False}
    assert module().audit_support_review(rejected, "a" * 64, "b" * 64, "c" * 64)["status"] == "FAIL"


def test_lost_thin_model_island_is_detected_from_mesh_sections_and_real_paths():
    primary = trimesh.creation.box(extents=(2., 2., .4))
    primary.apply_translation([20., 30., .2])
    missing = trimesh.creation.box(extents=(.5, .5, .4))
    missing.apply_translation([24., 30., .2])
    report = module().audit_toolpath_correspondence(trimesh.util.concatenate([primary, missing]), gcode(), settings())
    assert not report["passed"]
    assert report["lost_model_islands"] > 0


def test_missing_slicer_executable_fails_closed_without_launch(tmp_path):
    mesh_path = tmp_path / "source.stl"
    trimesh.creation.box().export(mesh_path)
    result = module().run_slicer(mesh_path, tmp_path / "stage", appimage=tmp_path / "absent.AppImage")
    assert result["status"] == "FAIL"
    assert not result["passed"]


def perimeter_fixture(*, gap=False, height=.4, cavity=False, internal_support=False):
    """Commanded E gives an independently specified .42mm nominal bead width."""
    mesh=trimesh.creation.box(extents=(4.,4.,height));mesh.apply_translation((2,2,height/2))
    if cavity:
        void=trimesh.creation.box(extents=(2.,2.,height));void.apply_translation((2,2,height/2));void.invert()
        mesh=trimesh.util.concatenate([mesh,void])
    lines=['G21','G90','M83','; LAYER_HEIGHT: 0.2']
    def path(points,z,feature='Outer wall'):
        lines.extend([f'; FEATURE: {feature}',f'G0 X{points[0][0]} Y{points[0][1]} Z{z}'])
        for start,end in zip(points,points[1:]):
            distance=np.linalg.norm(np.array(end)-start)
            extrusion=distance*.2*(.42-.2*(1-np.pi/4))/(np.pi*1.75**2/4)
            lines.append(f'G1 X{end[0]} Y{end[1]} E{extrusion:.9f}')
    for z in np.arange(.2,height+.001,.2):
        if gap:
            path([(.21,.21),(3.79,.21),(3.79,3.79),(.21,3.79),(.21,2.6)],z)
            path([(.21,1.4),(.21,.21)],z)
        else:
            path([(.21,.21),(3.79,.21),(3.79,3.79),(.21,3.79),(.21,.21)],z)
        if cavity:
            path([(.79,.79),(3.21,.79),(3.21,3.21),(.79,3.21),(.79,.79)],z)
        if internal_support:
            path([(1.5,2),(2.5,2)],z,'Support')
    return mesh,'\n'.join(lines)


def test_digital_perimeter_correspondence_can_pass_without_solid_infill():
    mesh,text=perimeter_fixture()
    report=module().audit_toolpath_correspondence(mesh,text,settings())
    assert report['passed'],report
    assert report['status']=='PASS' and report['physical_print_validated'] is False
    assert report['convergence']['passed']
    assert report['sampling_pitches_mm']==[.1,.05]
    assert report['interior_policy']=='configured_infill_not_solid_occupancy'


def test_partial_boundary_gap_fails_even_when_model_island_is_present():
    mesh,text=perimeter_fixture(gap=True)
    report=module().audit_toolpath_correspondence(mesh,text,settings())
    assert report['status']=='FAIL'
    assert report['lost_model_islands']==0
    assert 'SOURCE_BOUNDARY_GAP' in report['failures']


def test_partially_lost_attached_appendage_is_detected():
    mesh,text=perimeter_fixture()
    appendage=trimesh.creation.box(extents=(2.,.5,.4));appendage.apply_translation((5,2,.2))
    report=module().audit_toolpath_correspondence(trimesh.util.concatenate([mesh,appendage]),text,settings())
    assert report['status']=='FAIL' and report['lost_model_islands']==0
    assert 'SOURCE_BOUNDARY_GAP' in report['failures']


def test_cavity_is_preserved_but_support_inside_cavity_fails():
    mesh,text=perimeter_fixture(cavity=True)
    assert module().audit_toolpath_correspondence(mesh,text,settings())['passed']
    mesh,text=perimeter_fixture(cavity=True,internal_support=True)
    report=module().audit_toolpath_correspondence(mesh,text,settings())
    assert report['status']=='FAIL'
    assert 'INTERNAL_CAVITY_SUPPORT' in report['failures']


def test_extra_model_material_and_missing_middle_layer_fail():
    mesh,text=perimeter_fixture(height=.6)
    outside=text+'\n; FEATURE: Outer wall\nG0 X5 Y0 Z0.2\nG1 X6 E0.04\n'
    assert 'MODEL_PATH_OUTSIDE_SOURCE' in module().audit_toolpath_correspondence(mesh,outside,settings())['failures']
    # Keep the upper layer, so a maximum-height-only check cannot catch this.
    text=text.replace('Z0.4','Z0.2')
    assert 'MISSING_LAYER_GAP' in module().audit_toolpath_correspondence(mesh,text,settings())['failures']


def test_unresolved_motion_does_not_receive_digital_pass():
    mesh,text=perimeter_fixture()
    report=module().audit_toolpath_correspondence(mesh,'G20\n'+text,settings())
    assert report['status']=='REVIEW_REQUIRED' and not report['passed']


def test_complete_but_floating_upper_model_island_fails_anchoring():
    import re
    bottom,_=perimeter_fixture(height=.2)
    upper=bottom.copy();upper.apply_translation((6,0,.2))
    _,text=perimeter_fixture(height=.4)
    cut=text.index('G0 X0.21 Y0.21 Z0.4')
    upper_text=re.sub(r'X([-+\d.]+)',lambda m:'X'+str(float(m[1])+6),text[cut:])
    report=module().audit_toolpath_correspondence(trimesh.util.concatenate([bottom,upper]),text[:cut]+upper_text,settings())
    assert report['status']=='FAIL'
    assert 'UNANCHORED_DEPOSITION_ISLANDS' in report['failures']
    assert report['lost_model_islands']==0


def test_unclassified_positive_motion_is_ambiguous_even_with_valid_model_paths():
    mesh,text=perimeter_fixture()
    text+='\n; FEATURE: Unknown custom feature\nG0 X5 Y5 Z0.2\nG1 X6 E0.04\n'
    report=module().audit_toolpath_correspondence(mesh,text,settings())
    assert report['status']=='REVIEW_REQUIRED'
    assert any('unclassified positive deposition' in p for p in report['unresolved'])


def test_sampling_gate_disagreement_requires_review_instead_of_best_resolution():
    mesh,text=perimeter_fixture(gap=True)
    # A narrower actual gap changes end-segment widths as commanded E is held;
    # both remain within the supported width range. Coarse/fine gates disagree.
    text=text.replace('Y2.6','Y2.355').replace('Y1.4','Y1.645')
    report=module().audit_toolpath_correspondence(mesh,text,settings())
    assert report['status']=='REVIEW_REQUIRED'
    assert not report['convergence']['same_gate_decisions']
    assert 'SAMPLING_NOT_CONVERGED' in report['unresolved']
@pytest.mark.parametrize('injected',['M221 S0','G0 X10\nG92 X0'])
def test_unmodelled_flow_or_coordinate_offset_never_passes(injected):
    mesh,text=perimeter_fixture()
    changed=text.replace('G90','G90\n'+injected)
    report=module().audit_toolpath_correspondence(mesh,changed,settings())
    assert not report['passed']
    assert report['parse']['unresolved']
