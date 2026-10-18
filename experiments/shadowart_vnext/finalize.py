"""Finalize existing immutable mesh/slice bytes; never train, reslice or print.

Old reports are prerequisite provenance, not substitutes for new audit results.
Source declarations/hashes bind captured bytes; they do not authenticate authors.
"""
import hashlib
import io
import json
from pathlib import Path
import traceback
from zipfile import ZipFile
import numpy as np

from .artifacts import write_json
from .contracts import CameraContract,TargetSet,digest_json,file_hash,mask_hash
from .delivery import (CONTINUOUS_MESH_PROTOCOL, STRICT_VOXEL_PROTOCOL,
                       acceptance_protocol, audit_serialized_mesh, converged_quality,
                       publish_delivery, validate_qa_acceptance_protocol)
from .slicer import audit_sliced_artifacts


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _matching(label,actual,expected):
    if actual!=expected:
        raise ValueError('FINALIZE_SOURCE_BINDING_MISMATCH: '+label)


def _save_final_handoff(output,qa,slicing,contract):
    """Only the freshly revoxelized serialized STL may supply this handoff."""
    from .handoff import save_handoff
    path=output/'diagnostics/mesh/stl_reloaded_voxels_xyz.npy'
    if not path.is_file():
        raise ValueError('FINALIZE_FRESH_STL_VOXELS_MISSING')
    volume=np.load(path,allow_pickle=False)
    if volume.dtype!=bool or volume.shape!=(contract.volume_resolution,)*3 or not volume.any():
        raise ValueError('FINALIZE_FRESH_STL_VOXELS_INVALID')
    return save_handoff(output/'diagnostics/structural_handoff',volume,qa['canonical_to_print_mm'],
                        extent=contract.extent,mesh_sha256=qa['mesh_sha256'],
                        camera_digest=contract.digest,geometry_passed=qa['passed'],slicing=slicing)


def finalize_existing(args, *, prepared_output, source_hashes):
    """Re-audit exact existing artifacts into NEW output and publish iff all pass.

    The caller must create only a fresh output outside the original run. This
    entry point retains the source files and support review as captured bytes,
    and passes the review dictionary parsed from those same bytes to slice QA.
    No review is synthesized and no historical failed QA is promoted to PASS.
    """
    output=Path(prepared_output).resolve();source=Path(args.finalize_run).resolve()
    if output.is_relative_to(source):
        raise ValueError('FINALIZE_OUTPUT_MUST_BE_OUTSIDE_ORIGINAL_RUN')
    run={'status':'RUNNING','mode':'finalize_existing_immutable_slice',
         'source_execution':'captured_source_snapshot','source_sha256':source_hashes,
         'finalize_run':str(source),'canonical_delivery_published':False,
         'physical_print_validated':False,'structural_status':'NOT_REQUESTED',
         'resliced':False,'retrained':False,'original_files_modified':False}
    write_json(output/'RUN.json',run)
    try:
        if not source.is_dir() or args.support_review is None:
            raise ValueError('FINALIZE_SOURCE_OR_SUPPORT_REVIEW_MISSING')
        captured={};identities={}
        def capture(relative):
            path=(source/relative).resolve()
            if not path.is_relative_to(source) or not path.is_file():
                raise ValueError('FINALIZE_SOURCE_FILE_MISSING_OR_EXTERNAL: '+relative)
            raw=path.read_bytes();captured[relative]=raw
            identities[relative]={'path':str(path),'sha256':_sha(raw)}
            return raw
        documents={name:json.loads(capture(name)) for name in
                   ('QA.json','CAMERA_CONTRACT.json','RUN.json','GEOMETRY_CONFIG.json','TARGETS.json','MANUFACTURING.json')}
        old_qa=documents['QA.json'];old_run=documents['RUN.json'];old_manufacturing=documents['MANUFACTURING.json']
        contract=CameraContract.from_dict(documents['CAMERA_CONTRACT.json'])
        geometry=documents['GEOMETRY_CONFIG.json']
        protocol=acceptance_protocol(geometry)
        if old_qa.get('acceptance_protocol',STRICT_VOXEL_PROTOCOL)!=protocol:
            raise ValueError('FINALIZE_ACCEPTANCE_PROTOCOL_MISMATCH')
        if (old_qa.get('passed') is not True or not old_qa.get('independent_serialized_mesh')
                or not converged_quality(old_qa.get('evaluations',[]))['passed']):
            raise ValueError('FINALIZE_ORIGINAL_QA_NOT_PASSED')
        validate_qa_acceptance_protocol(old_qa)
        canonical_resolution=(contract.volume_resolution in (128,256)
                              if protocol==CONTINUOUS_MESH_PROTOCOL
                              else contract.volume_resolution==256)
        if (not canonical_resolution or contract.image_size!=256
                or old_qa.get('native_resolution')!=contract.volume_resolution):
            raise ValueError('FINALIZE_ORIGINAL_NOT_CANONICAL')
        if protocol==CONTINUOUS_MESH_PROTOCOL and (
                old_manufacturing.get('acceptance_protocol')!=protocol
                or old_manufacturing.get('native_optimization_resolution')!=contract.volume_resolution):
            raise ValueError('FINALIZE_MANUFACTURING_PROTOCOL_MISMATCH')
        for label,record in (('QA',old_qa),('MANUFACTURING',old_manufacturing)):
            _matching(label+' camera',record.get('camera_digest'),contract.digest)
            _matching(label+' targets',record.get('target_hashes'),list(contract.target_hashes))
            _matching(label+' geometry',record.get('geometry_digest'),digest_json(geometry))
        _matching('RUN camera',old_run.get('camera_digest'),contract.digest)
        source_manifest=old_run.get('source_sha256')
        if old_run.get('source_execution')!='captured_source_snapshot' or not isinstance(source_manifest,dict) or not source_manifest:
            raise ValueError('FINALIZE_SOURCE_SNAPSHOT_PROVENANCE_MISSING')
        for name,expected in source_manifest.items():
            if Path(name).name!=name or not name.endswith('.py'):
                raise ValueError('FINALIZE_INVALID_SOURCE_SNAPSHOT_NAME')
            _matching('source '+name,_sha(capture('source_snapshot/'+name)),expected)
        original_files={}
        for ext in ('obj','stl'):
            relative='diagnostics/mesh/candidate_full.'+ext
            original_files[ext]=relative
            row=old_qa['serialized'][ext]
            if row.get('passed') is not True:
                raise ValueError('FINALIZE_ORIGINAL_SERIALIZATION_QA_NOT_PASSED')
            _matching(ext,_sha(capture(relative)),row['sha256'])
        mesh_sha=old_qa['serialized']['stl']['sha256']
        _matching('QA mesh',old_qa.get('mesh_sha256'),mesh_sha)
        _matching('MANUFACTURING mesh',old_manufacturing.get('mesh_sha256'),mesh_sha)
        slice_hashes=old_manufacturing.get('slicing',{}).get('source_hashes',{})
        _matching('sliced mesh',slice_hashes.get('mesh_sha256'),mesh_sha)
        slice_names={'archive_sha256':'shadowart_supported.gcode.3mf',
                     'effective_settings_sha256':'effective_settings.json',
                     'result_sha256':'result.json','slicer_log_sha256':'slicer.log'}
        slice_paths={key:'diagnostics/fabrication/slicer/'+name for key,name in slice_names.items()}
        for key,relative in slice_paths.items():
            _matching(key,_sha(capture(relative)),slice_hashes.get(key))
        with ZipFile(io.BytesIO(captured[slice_paths['archive_sha256']])) as archive:
            entries=[name for name in archive.namelist() if name.startswith('Metadata/plate_') and name.endswith('.gcode')]
            if len(entries)!=1:
                raise ValueError('FINALIZE_AMBIGUOUS_EXISTING_GCODE')
            _matching('gcode',_sha(archive.read(entries[0])),slice_hashes.get('gcode_sha256'))
        # Capture original source images from their recorded absolute paths;
        # hash them against the immutable camera/target binding before parsing.
        target_records=documents['TARGETS.json']
        if not isinstance(target_records,list) or len(target_records)!=contract.base_count:
            raise ValueError('FINALIZE_SOURCE_TARGET_RECORDS_INVALID')
        image_names=[]
        for index,row in enumerate(target_records):
            path=Path(row['input_path']).resolve();raw=path.read_bytes()
            expected=contract.input_hashes[index]
            _matching('target input record',row.get('input_sha256'),expected)
            _matching('target input bytes',_sha(raw),expected)
            relative=f'inputs/{index:03d}{path.suffix}'
            captured[relative]=raw;identities[relative]={'path':str(path),'sha256':_sha(raw)}
            image_names.append(relative)
        review_path=Path(args.support_review).resolve();review_bytes=review_path.read_bytes()
        review=json.loads(review_bytes)
        if not isinstance(review,dict):
            raise ValueError('FINALIZE_SUPPORT_REVIEW_MUST_BE_OBJECT')
        for key,expected in (('mesh_sha256',mesh_sha),('archive_sha256',slice_hashes['archive_sha256']),
                             ('gcode_sha256',slice_hashes['gcode_sha256'])):
            _matching('support review '+key,review.get(key),expected)
        captured['support_review.json']=review_bytes
        identities['support_review.json']={'path':str(review_path),'sha256':_sha(review_bytes)}
        stage=output/'diagnostics/finalize_inputs';stage.mkdir(parents=True,exist_ok=False)
        for relative,raw in captured.items():
            path=stage/relative;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:stream.write(raw)
        targets=TargetSet.from_paths([stage/name for name in image_names])
        _matching('staged source image hashes',targets.input_hashes,contract.input_hashes)
        masks=targets.render(contract.cameras,contract.image_size,contract.layout)
        _matching('source-derived target hashes',tuple(mask_hash(mask) for mask in masks),contract.target_hashes)
        write_json(output/'FINALIZE_SOURCE.json',{'source_run':str(source),'files':identities,
                   'identity_scope':'captured_bytes_and_declared_bindings_not_producer_authentication'})
        write_json(output/'CAMERA_CONTRACT.json',contract.to_dict())
        write_json(output/'GEOMETRY_CONFIG.json',geometry)
        write_json(output/'TARGETS.json',targets.diagnostics(contract.cameras,contract.image_size,contract.layout))
        run.update(camera_digest=contract.digest,source_bindings=identities,
                   acceptance_protocol=protocol,
                   arguments={'silhouettes':list(targets.paths),'finalize_run':str(source)})
        files={ext:stage/name for ext,name in original_files.items()}
        qa,files=audit_serialized_mesh(None,contract,targets,output/'diagnostics/mesh',
                    geometry_settings=geometry,serialized_source=files,
                    physical_transform=old_qa['canonical_to_print_mm'])
        write_json(output/'QA.json',qa)
        if not qa.get('passed'):
            run['status']='FINALIZE_FRESH_GEOMETRY_QA_FAILED';write_json(output/'RUN.json',run)
            return 2
        for ext,path in files.items():
            _matching('fresh '+ext+' bytes',file_hash(path),old_qa['serialized'][ext]['sha256'])
            _matching('fresh '+ext+' QA hash',qa['serialized'][ext]['sha256'],old_qa['serialized'][ext]['sha256'])
        _matching('fresh QA camera',qa.get('camera_digest'),contract.digest)
        _matching('fresh QA geometry',qa.get('geometry_digest'),digest_json(geometry))
        _matching('fresh QA mesh',qa.get('mesh_sha256'),mesh_sha)
        _matching('fresh QA protocol',qa.get('acceptance_protocol',STRICT_VOXEL_PROTOCOL),protocol)
        slicing=audit_sliced_artifacts(files['stl'],stage/slice_paths['archive_sha256'],
                    stage/slice_paths['effective_settings_sha256'],stage/slice_paths['result_sha256'],
                    log_path=stage/slice_paths['slicer_log_sha256'],support_review=review)
        for key in ('mesh_sha256','archive_sha256','effective_settings_sha256','result_sha256','slicer_log_sha256','gcode_sha256'):
            _matching('fresh slice '+key,slicing.get('source_hashes',{}).get(key),slice_hashes[key])
        topology=qa['serialized']['stl']['topology']
        checks={'one_body':qa['serialized']['stl'].get('component_count_6')==1,
                'watertight':topology.get('watertight') is True,
                'winding_consistent':topology.get('winding_consistent') is True,
                'self_intersections_clear':topology.get('self_intersection',{}).get('status')=='PASS',
                'thickness_pass':qa['serialized']['stl'].get('thickness',{}).get('passed') is True,
                'slicer_pass':slicing.get('passed') is True,'support_removal_pass':slicing.get('support_removal_pass') is True}
        manufacturing={'passed':all(checks.values()),'status':'PASS' if all(checks.values()) else 'REVIEW_REQUIRED',
            'mesh_sha256':mesh_sha,'camera_digest':contract.digest,'target_hashes':list(contract.target_hashes),
            'geometry_digest':digest_json(geometry),'checks':checks,'slicing':slicing,
            'acceptance_protocol':protocol,
            'native_optimization_resolution':contract.volume_resolution,
            'physical_print_validated':False,'structural_status':'NOT_REQUESTED',
            'support_removal_review':{'path':str(stage/'support_review.json'),
                                      'original_path':str(review_path),'staged_path':str(stage/'support_review.json'),
                                      'sha256':_sha(review_bytes),'reviewer':review.get('reviewer')},
            'source_slice':{'source_run':str(source),'source_archive':str(source/slice_paths['archive_sha256']),
                            'archive_path':str(stage/slice_paths['archive_sha256']),**slice_hashes},
            'source_bindings':identities,'resliced':False}
        write_json(output/'MANUFACTURING.json',manufacturing)
        if not manufacturing['passed']:
            run['status']='FINALIZE_MANUFACTURING_REVIEW_REQUIRED';write_json(output/'RUN.json',run)
            return 2
        # Check copied evidence again after external audit calls; final delivery
        # separately validates the final serialized bytes before atomic publish.
        for relative,row in identities.items():
            _matching('staged evidence '+relative,file_hash(stage/relative),row['sha256'])
        _save_final_handoff(output,qa,slicing,contract)
        run['structural_handoff']=str(output/'diagnostics/structural_handoff/STRUCTURAL_HANDOFF.json')
        run['delivery']=publish_delivery(output,qa,manufacturing,files)
        run['status']='DIGITAL_MANUFACTURING_PASS';run['canonical_delivery_published']=True
        write_json(output/'RUN.json',run)
        return 0
    except Exception as error:
        run.update(status='FINALIZE_FAILED',failure_reason=str(error),exception_type=type(error).__name__)
        write_json(output/'RUN.json',run)
        write_json(output/'diagnostics/finalize_exception.json',{'traceback':traceback.format_exc()})
        return 2
