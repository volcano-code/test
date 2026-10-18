import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_cli_explicit_camera_cannot_mix_with_preset(tmp_path):
    from experiments.run_shadowart_vnext import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--silhouettes','x.png','--camera-config','c.json','--tsf','1',
                    '--output',str(tmp_path/'run')])


def test_failed_quality_cannot_publish_canonical_files(tmp_path):
    from experiments.shadowart_vnext.delivery import publish_delivery
    with pytest.raises(ValueError,match='GATE'):
        publish_delivery(tmp_path, {'passed':False}, {'passed':True}, {})
    assert not (tmp_path/'delivery_postprocessed.obj').exists()
    assert not (tmp_path/'delivery_postprocessed.stl').exists()


def test_success_string_without_real_audits_does_not_publish(tmp_path):
    from experiments.shadowart_vnext.delivery import publish_delivery
    with pytest.raises(ValueError):
        publish_delivery(tmp_path, {'passed':True}, {'passed':True}, {})
    assert not (tmp_path/'delivery_postprocessed.obj').exists()


def test_image_resolution_disagreement_cannot_pass_on_good_mean_only():
    from experiments.shadowart_vnext.delivery import converged_quality
    assert not converged_quality([{'image_size':256,'per_view_iou':[.995,.995]},
                                  {'image_size':512,'per_view_iou':[.979,.991]}])['passed']
    assert converged_quality([{'image_size':256,'per_view_iou':[.989,.991]},
                              {'image_size':512,'per_view_iou':[.989,.991]}])['passed']


def test_resolution_evidence_must_be_distinct_and_iou_bounded():
    from experiments.shadowart_vnext.delivery import converged_quality
    assert not converged_quality([{'image_size':512,'per_view_iou':[.99]}]*2)['passed']
    assert not converged_quality([{'image_size':256,'per_view_iou':[1.1]},
                                  {'image_size':512,'per_view_iou':[1.1]}])['passed']
    assert not converged_quality([{'image_size':256,'per_view_iou':[.97]},
                                  {'image_size':512,'per_view_iou':[.99]},
                                  {'image_size':1024,'per_view_iou':[.99]}])['passed']


def test_aggregation_preserves_escalation_in_either_format():
    from experiments.shadowart_vnext.delivery import aggregate_serialized_quality
    rows={'obj':{'evaluations':[{'image_size':256,'per_view_iou':[.982]},
                               {'image_size':512,'per_view_iou':[.986]},
                               {'image_size':1024,'per_view_iou':[.986]}]},
          'stl':{'evaluations':[{'image_size':256,'per_view_iou':[.986]},
                               {'image_size':512,'per_view_iou':[.986]}]}}
    result=aggregate_serialized_quality(rows)
    assert result['requires_resolution_for']=={'stl':[1024]}
    assert result['passed'] is False


def test_open_mesh_keeps_diagnostic_qa_instead_of_losing_failure_to_voxelizer(tmp_path):
    import trimesh
    from experiments.shadowart_vnext.contracts import TargetSet,build_contract
    from experiments.shadowart_vnext.delivery import audit_serialized_mesh
    raw=np.zeros((32,32),np.uint8)
    raw[8:24,8:24]=255
    source=tmp_path/'target.png'
    Image.fromarray(raw).save(source)
    mesh=trimesh.creation.box(extents=[1.4]*3)
    mesh.update_faces(np.arange(len(mesh.faces))!=0)
    report,files=audit_serialized_mesh(mesh,build_contract(1,resolution=16),
        TargetSet.from_paths([source]),tmp_path/'audit',geometry_settings={})
    assert report['passed'] is False
    assert 'NONMANIFOLD_OR_OPEN' in report['serialized']['stl']['topology']['reasons']
    assert (tmp_path/'audit'/'QA.json').exists()


def test_preflight_cli_creates_frozen_inputs_but_no_delivery(tmp_path):
    source=tmp_path/'mask.png'
    raw=np.zeros((32,32),np.uint8)
    raw[8:24,8:24]=255
    Image.fromarray(raw).save(source)
    root=Path(__file__).resolve().parents[1]
    out=tmp_path/'run'
    command=[sys.executable,str(root/'experiments/run_shadowart_vnext.py'),
             '--silhouettes',str(source),'--output',str(out),'--stage','preflight',
             '--resolution','16','--image-size','16','--layout-candidates','1','--device','cpu']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr+result.stdout
    run=json.loads((out/'RUN.json').read_text())
    assert run['status']=='PREFLIGHT_COMPLETED'
    assert run['structural_status']=='NOT_REQUESTED'
    assert run['canonical_delivery_published'] is False
    assert (out/'CAMERA_CONTRACT.json').exists()
    environment=json.loads((out/'ENVIRONMENT.json').read_text())
    assert 'numpy' in environment['distribution_versions']
    assert environment['python_executable']
    assert not (out/'delivery_postprocessed.stl').exists()


def test_native_cube_actual_obj_stl_roundtrip_can_pass_geometry_gate(tmp_path):
    import trimesh
    from experiments.shadowart_vnext.contracts import TargetSet,build_contract
    from experiments.shadowart_vnext.delivery import audit_serialized_mesh
    source=tmp_path/'square.png'
    raw=np.zeros((32,32),np.uint8);raw[8:24,8:24]=255
    Image.fromarray(raw).save(source)
    # Grid-aligned cell-union cube. The unsnapped 1.4 cube loses a border at
    # native revoxelization (IoU .9779 at256), correctly failing the strict gate.
    report,_=audit_serialized_mesh(trimesh.creation.box(extents=[212*1.7/256]*3),
        build_contract(1,resolution=256),TargetSet.from_paths([source]),tmp_path/'audit',
        geometry_settings={'fixture':'analytic_cube'})
    assert report['passed'],report
    assert all(row['component_count_6']==1 for row in report['serialized'].values())
    assert report['physical_print_validated'] is False


def test_existing_serialized_audit_never_reexports_source_bytes(tmp_path):
    import trimesh
    from experiments.shadowart_vnext.contracts import TargetSet,build_contract,file_hash
    from experiments.shadowart_vnext.delivery import audit_serialized_mesh
    raw=np.zeros((32,32),np.uint8);raw[8:24,8:24]=255
    source=tmp_path/'mask.png';Image.fromarray(raw).save(source)
    mesh=trimesh.creation.box(extents=[1.4]*3)
    mesh.update_faces(np.arange(len(mesh.faces))!=0)
    paths={ext:tmp_path/('original.'+ext) for ext in ('obj','stl')}
    for path in paths.values():mesh.export(path)
    original={ext:file_hash(path) for ext,path in paths.items()}
    report,copied=audit_serialized_mesh(None,build_contract(1,resolution=16),
        TargetSet.from_paths([source]),tmp_path/'audit',geometry_settings={},
        serialized_source=paths,physical_transform=np.eye(4))
    assert not report['passed']
    for ext,path in paths.items():
        assert file_hash(path)==file_hash(copied[ext])==original[ext]
