"""Finalizer orchestration tests; mocked gate events are NOT print/QA evidence."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from PIL import Image
import pytest
import trimesh


def module():
    spec=importlib.util.find_spec('experiments.shadowart_vnext.finalize')
    assert spec is not None,'existing-artifact finalizer is missing'
    return importlib.import_module('experiments.shadowart_vnext.finalize')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(tmp_path):
    from dataclasses import replace
    from experiments.shadowart_vnext.contracts import build_contract,TargetSet,mask_hash,digest_json
    source=tmp_path/'original';source.mkdir(); image=tmp_path/'input.png'
    pixels=np.zeros((16,16),np.uint8);pixels[3:13,3:13]=255;Image.fromarray(pixels).save(image)
    targets=TargetSet.from_paths([image]);contract=build_contract(1)
    masks=targets.render(contract.cameras,256)
    contract=replace(contract,input_hashes=targets.input_hashes,target_hashes=tuple(mask_hash(m) for m in masks))
    geometry={'longest_mm':100.,'min_wall_mm':.9,'min_neck_mm':1.35}
    mesh_dir=source/'diagnostics/mesh';mesh_dir.mkdir(parents=True)
    mesh=trimesh.creation.box(extents=(100,100,100));mesh.apply_translation((0,0,50))
    for ext in ('obj','stl'):mesh.export(mesh_dir/f'candidate_full.{ext}')
    topology={'passed':True,'watertight':True,'winding_consistent':True,'self_intersection':{'status':'PASS'}}
    rows={ext:{'passed':True,'sha256':sha(mesh_dir/f'candidate_full.{ext}'),'topology':topology,
               'component_count_6':1,'thickness':{'passed':True}} for ext in ('obj','stl')}
    qa={'passed':True,'status':'PASS','mesh_sha256':rows['stl']['sha256'],'serialized':rows,
        'camera_digest':contract.digest,'target_hashes':list(contract.target_hashes),
        'geometry_digest':digest_json(geometry),'native_resolution':256,
        'independent_serialized_mesh':True,'canonical_to_print_mm':[[100,0,0,0],[0,0,-100,0],[0,100,0,50],[0,0,0,1]],
        'evaluations':[{'image_size':s,'per_view_iou':[.99]} for s in (256,512)]}
    snapshot=source/'source_snapshot';snapshot.mkdir();(snapshot/'pipeline.py').write_text('# captured fixture source\n')
    run={'status':'MANUFACTURING_REVIEW_REQUIRED','camera_digest':contract.digest,
         'source_execution':'captured_source_snapshot','source_sha256':{'pipeline.py':sha(snapshot/'pipeline.py')},
         'arguments':{'silhouettes':[str(image)]},'engineering_only':False}
    write(source/'RUN.json',run);write(source/'QA.json',qa);write(source/'CAMERA_CONTRACT.json',contract.to_dict())
    write(source/'GEOMETRY_CONFIG.json',geometry)
    write(source/'TARGETS.json',targets.diagnostics(contract.cameras,256,contract.layout))
    fabrication=source/'diagnostics/fabrication/slicer';fabrication.mkdir(parents=True)
    archive=fabrication/'shadowart_supported.gcode.3mf';gcode=b'; existing fixture, never resliced\n'
    with ZipFile(archive,'w') as z:z.writestr('Metadata/plate_1.gcode',gcode)
    write(fabrication/'effective_settings.json',{});write(fabrication/'result.json',{})
    (fabrication/'slicer.log').write_text('fixture slicer log\n')
    hashes={'mesh_sha256':qa['mesh_sha256'],'archive_sha256':sha(archive),
            'effective_settings_sha256':sha(fabrication/'effective_settings.json'),
            'result_sha256':sha(fabrication/'result.json'),'slicer_log_sha256':sha(fabrication/'slicer.log'),
            'gcode_sha256':hashlib.sha256(gcode).hexdigest()}
    write(source/'MANUFACTURING.json',{'passed':False,'mesh_sha256':qa['mesh_sha256'],
          'camera_digest':contract.digest,'target_hashes':list(contract.target_hashes),
          'geometry_digest':digest_json(geometry),'slicing':{'source_hashes':hashes}})
    review=tmp_path/'review.json'
    # Deliberately NOT a real human approval. Tests below patch the audit gates
    # only to observe call order, and patch publication to prevent any delivery.
    write(review,{'test_only':'gate order, not human approval','mesh_sha256':qa['mesh_sha256'],
                  'archive_sha256':hashes['archive_sha256'],'gcode_sha256':hashes['gcode_sha256']})
    output=tmp_path/'new';(output/'diagnostics').mkdir(parents=True)
    return SimpleNamespace(finalize_run=source,output=output,support_review=review),qa,hashes


def install_gates(monkeypatch,mod,qa,hashes,events,*,mesh_pass=True,slice_pass=True):
    def mesh_gate(mesh,contract,targets,stage,*,geometry_settings,serialized_source,physical_transform):
        events.append('mesh')
        assert mesh is None
        assert all(Path(p).is_relative_to(stage.parent) for p in serialized_source.values())
        for ext,path in serialized_source.items():assert sha(path)==qa['serialized'][ext]['sha256']
        assert physical_transform==qa['canonical_to_print_mm']
        result=copy.deepcopy(qa);result['passed']=mesh_pass
        return result,serialized_source
    def slice_gate(mesh,archive,effective,result,*,log_path,support_review):
        events.append('existing_slice')
        assert sha(mesh)==hashes['mesh_sha256'] and sha(archive)==hashes['archive_sha256']
        assert isinstance(support_review,dict)
        return {'passed':slice_pass,'status':'PASS' if slice_pass else 'REVIEW_REQUIRED',
                'support_removal_pass':slice_pass,'source_hashes':hashes,'checks':{}}
    def publish(output,qa,manufacturing,files):
        events.append('publish')
        assert manufacturing['source_slice']['archive_sha256']==hashes['archive_sha256']
        return {'test_only':'publication event; no canonical file created'}
    monkeypatch.setattr(mod,'audit_serialized_mesh',mesh_gate)
    monkeypatch.setattr(mod,'audit_sliced_artifacts',slice_gate)
    monkeypatch.setattr(mod,'publish_delivery',publish)
    monkeypatch.setattr(mod,'_save_final_handoff',lambda *a,**kw:events.append('handoff'),raising=False)


def test_finalizer_reaudits_exact_existing_bytes_before_publication_event(tmp_path,monkeypatch):
    args,qa,hashes=fixture(tmp_path);mod=module();events=[]
    original={str(p.relative_to(args.finalize_run)):sha(p) for p in args.finalize_run.rglob('*') if p.is_file()}
    install_gates(monkeypatch,mod,qa,hashes,events)
    assert mod.finalize_existing(args,prepared_output=args.output,source_hashes={'finalize.py':'a'*64})==0
    assert events==['mesh','existing_slice','handoff','publish']
    assert original=={str(p.relative_to(args.finalize_run)):sha(p) for p in args.finalize_run.rglob('*') if p.is_file()}


@pytest.mark.parametrize('failure',['missing_archive','stale_mesh','stale_archive','failed_qa','stale_source','stale_input','missing_review','stale_review'])
def test_unavailable_stale_or_failed_source_never_reaches_publish(tmp_path,monkeypatch,failure):
    args,qa,hashes=fixture(tmp_path);mod=module();events=[]
    install_gates(monkeypatch,mod,qa,hashes,events)
    if failure=='missing_archive':(args.finalize_run/'diagnostics/fabrication/slicer/shadowart_supported.gcode.3mf').unlink()
    elif failure=='stale_mesh':(args.finalize_run/'diagnostics/mesh/candidate_full.stl').write_bytes(b'changed')
    elif failure=='stale_archive':(args.finalize_run/'diagnostics/fabrication/slicer/shadowart_supported.gcode.3mf').write_bytes(b'changed')
    elif failure=='failed_qa':qa['passed']=False;write(args.finalize_run/'QA.json',qa)
    elif failure=='stale_source':(args.finalize_run/'source_snapshot/pipeline.py').write_text('# changed\n')
    elif failure=='stale_input':(tmp_path/'input.png').write_bytes(b'changed')
    elif failure=='missing_review':args.support_review=None
    elif failure=='stale_review':write(args.support_review,{'mesh_sha256':'0'*64,'archive_sha256':'0'*64,'gcode_sha256':'0'*64})
    assert mod.finalize_existing(args,prepared_output=args.output,source_hashes={})==2
    assert 'publish' not in events and not (args.output/'delivery').exists()


@pytest.mark.parametrize('mesh_pass,slice_pass',[(False,True),(True,False)])
def test_fresh_failed_audit_cannot_be_overridden_by_old_passing_qa(tmp_path,monkeypatch,mesh_pass,slice_pass):
    args,qa,hashes=fixture(tmp_path);mod=module();events=[]
    install_gates(monkeypatch,mod,qa,hashes,events,mesh_pass=mesh_pass,slice_pass=slice_pass)
    assert mod.finalize_existing(args,prepared_output=args.output,source_hashes={})==2
    assert 'publish' not in events


def test_review_change_after_capture_does_not_change_approved_bytes_record(tmp_path,monkeypatch):
    args,qa,hashes=fixture(tmp_path);mod=module();events=[]
    install_gates(monkeypatch,mod,qa,hashes,events)
    captured_hash=sha(args.support_review)
    mesh_gate=mod.audit_serialized_mesh
    def change_external_review(*a,**kw):
        result=mesh_gate(*a,**kw)
        write(args.support_review,{'test_only':'changed after capture, not an approval'})
        return result
    monkeypatch.setattr(mod,'audit_serialized_mesh',change_external_review)
    assert mod.finalize_existing(args,prepared_output=args.output,source_hashes={})==0
    report=json.loads((args.output/'MANUFACTURING.json').read_bytes())
    assert report['support_removal_review']['sha256']==captured_hash
    assert sha(report['support_removal_review']['staged_path'])==captured_hash
    assert report['support_removal_review']['path']==report['support_removal_review']['staged_path']
    assert report['support_removal_review']['original_path']==str(args.support_review.resolve())
    assert sha(args.support_review)!=captured_hash
