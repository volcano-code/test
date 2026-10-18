"""Frozen native-128 two-island joint ray-demand probe. Diagnostic outputs only."""
import argparse
import importlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

FROZEN_CAMERA='01bf5e162d7e1868fa44bfe49ac70a0fbca5642f1011062933af92b829e9f2f8'
FROZEN_FIELDS={
    'guidance.npy':'c7b6be119e52cc5217c991d6b4c43e7e1653c09fa8e2d88c0dd8997bb27730fd',
    'density.npy':'803e523e9d8542f5cd03c83dd979182d792644ff8b74e6484f2f38bd9c4eb6e1',
    'occupancy.npy':'d3453299a5869d42b3fcf3bc0bceec712db88c7c7a26b505a0f75527161c448e'}
FROZEN_CLOSURE='29bf97877e2d80ef172bf82c46bd4732189a91c2ded5d87a5194b5e51bbe95f3'
FROZEN_ISLANDS=[(742,'b040460bfd4ab0a245d8c9d0f99a9b64dffb96a02e5993eacc4cf12c83a08297'),
                (663,'fe8539d0a43ebe911b76273e53350d5fff24e732362ffd2af436b1d8bceb1176')]


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--soft-artifact-dir',type=Path,required=True)
    p.add_argument('--silhouettes',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-primitives',type=int,default=64)
    p.add_argument('--time-limit',type=float,default=10.)
    p.add_argument('--single-only',action='store_true')
    a=p.parse_args(argv)
    if not 0<=a.max_primitives<=64: p.error('max-primitives must be in [0,64]')
    if not math.isfinite(a.time_limit) or not 0<a.time_limit<=10: p.error('time-limit must be in (0,10]')
    source,output=a.soft_artifact_dir.resolve(),a.output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        p.error('output must be separate from immutable source tree')
    return a


def main(argv=None):
    a=parse_args(argv)
    import numpy as np
    from experiments.shadowart_vnext.artifacts import create_run,write_json
    from experiments.shadowart_vnext.provenance import freeze_sources
    source=a.soft_artifact_dir.resolve(); output=create_run(a.output.resolve())
    package,hashes=freeze_sources(output/'source_snapshot',ROOT/'experiments/shadowart_vnext',__file__)
    def module(name): return importlib.import_module(package+'.'+name)
    contracts=module('contracts'); file_hash=contracts.file_hash
    report={'status':'RUNNING','canonical_delivery_published':False,'structural_status':'NOT_REQUESTED',
        'paper_method_equivalence':False,'physical_print_validated':False,'source_sha256':hashes,
        'source_execution':'captured_source_snapshot','python_executable':sys.executable,
        'python_version':platform.python_version(),'arguments':{'soft_artifact_dir':str(source),
        'silhouettes':[str(p.resolve()) for p in a.silhouettes],'max_primitives':a.max_primitives,
        'time_limit':a.time_limit,'single_only':a.single_only,'image_size':1024,'islands':2,
        'independent_transactions':True,'max_mesh_audits_per_island':4},'islands':{}}
    write_json(output/'REPORT.json',report)
    try:
        contract=contracts.CameraContract.from_dict(json.loads((source/'CAMERA_CONTRACT.json').read_text()))
        if contract.volume_resolution!=128: raise ValueError('native 128 required')
        if contract.digest!=FROZEN_CAMERA: raise ValueError('FAILED_FROZEN_CONTRACT')
        if any(not (source/name).is_file() or file_hash(source/name)!=digest
               for name,digest in FROZEN_FIELDS.items()):
            raise ValueError('FAILED_FROZEN_SOFT_FIELDS')
        targets=contracts.TargetSet.from_paths(a.silhouettes)
        if (targets.input_hashes!=contract.input_hashes or tuple(contracts.mask_hash(m) for m in
            targets.render(contract.cameras,contract.image_size,contract.layout))!=contract.target_hashes):
            raise ValueError('FAILED_TARGET_HASH')
        fields=module('artifacts').load_soft(source,contract)
        source_hashes={p.name:file_hash(p) for p in source.iterdir() if p.is_file()}
        closed,closure=module('ray_closure').shadow_closure(fields['occupancy'],contract.cameras,
            contract.extent,minimum_block_width=2)
        islands=module('regional_candidates').component_order(closed,2,order='largest')[:2]
        if len(islands)!=2: raise ValueError('probe requires exactly two non-main islands')
        if contracts.mask_hash(closed)!=FROZEN_CLOSURE: raise ValueError('FAILED_FROZEN_CLOSURE')
        selection_hashes=[]
        for points in islands:
            mask=np.zeros_like(closed); mask[tuple(points.T)]=True
            selection_hashes.append((int(len(points)),contracts.mask_hash(mask)))
        if selection_hashes!=FROZEN_ISLANDS: raise ValueError('FAILED_FROZEN_ISLANDS')
        np.save(output/'diagnostics/closure_xyz.npy',closed,allow_pickle=False)
        write_json(output/'CAMERA_CONTRACT.json',contract.to_dict())
        masks=targets.render(contract.cameras,1024,contract.layout)
        report.update(native_resolution=128,camera_digest=contract.digest,input_sha256=source_hashes,
            frozen_identity_verified=True,selected_islands=selection_hashes,
            closure=closure,initial_hard_quality=module('layout').iou_report(
                module('rendering').render_binary(closed,contract.cameras,1024,contract.extent),masks))
        write_json(output/'REPORT.json',report)
        for index,points in enumerate(islands,1):
            stage=output/f'diagnostics/island_{index}'; stage.mkdir()
            selected=np.zeros_like(closed); selected[tuple(points.T)]=True
            np.save(stage/'selected_xyz.npy',selected,allow_pickle=False)
            def progress(row):
                write_json(stage/'latest.json',row)
                brief={k:v for k,v in row.items() if k!='solver'}
                if 'solver' in row: brief['solver_status']=row['solver']['status']
                print(json.dumps({'island':index,**brief}),flush=True)
            result,trace,finalists=module('demand_replacement').replace_by_demand(closed,fields['guidance'],
                selected,masks,contract.cameras,extent=contract.extent,longest_mm=100.,
                max_primitives=a.max_primitives,single_only=a.single_only,time_limit=a.time_limit,callback=progress)
            np.save(stage/'result_xyz.npy',result,allow_pickle=False)
            write_json(stage/'TRACE.json',trace)
            row={k:v for k,v in trace.items() if k not in ('initial_manufacturing','families','solves','finalists')}
            row.update(selected_voxels=int(selected.sum()),selected_mask_sha256=contracts.mask_hash(selected),mesh_candidates=[])
            report['islands'][f'island_{index}']=row; write_json(output/'REPORT.json',report)
            for state in finalists:
                candidate_dir=stage/f'candidate_{state["id"]}'; candidate_dir.mkdir()
                candidate=state['volume']
                for name,array in [('candidate_xyz',candidate),('added_xyz',candidate & ~closed),
                                   ('removed_xyz',closed & ~candidate)]:
                    np.save(candidate_dir/(name+'.npy'),array,allow_pickle=False)
                mesh=module('geometry').occupancy_to_mesh(candidate,extent=contract.extent)
                qa,_=module('delivery').audit_serialized_mesh(mesh,contract,targets,candidate_dir/'serialized',
                    geometry_settings={'method':trace['algorithm'],'diagnostic_only':True,
                        'candidate_accepted_as_transaction':state['accepted'],'acceptance_protocol':'continuous_mesh_v1',
                        'longest_mm':100.,'min_wall_mm':.9,'min_neck_mm':1.35,
                        'candidate_sha256':file_hash(candidate_dir/'candidate_xyz.npy')})
                summary={'id':state['id'],'transaction_accepted':state['accepted'],'reasons':state['reasons'],
                    'hard_quality':state['quality'],'mesh_quality':qa['quality'],'mesh_qa_passed':qa['passed'],
                    'component_count_6':state['manufacturing']['component_count_6'],
                    'thin_voxels':state['manufacturing']['thin_voxels']}
                row['mesh_candidates'].append(summary); write_json(output/'REPORT.json',report)
                print(json.dumps({'island':index,**summary}),flush=True)
        if source_hashes!={p.name:file_hash(p) for p in source.iterdir() if p.is_file()}:
            raise RuntimeError('SOURCE_ARTIFACT_CHANGED')
        report.update(status='DIAGNOSTIC_COMPLETED_NOT_DELIVERED',source_hashes_unchanged=True)
    except Exception as error:
        report.update(status='ERROR',error=str(error)); write_json(output/'REPORT.json',report)
        raise
    write_json(output/'REPORT.json',report)
    return 2


if __name__=='__main__': raise SystemExit(main())
