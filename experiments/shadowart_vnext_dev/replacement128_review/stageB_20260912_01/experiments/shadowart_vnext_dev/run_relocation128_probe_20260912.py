"""Native-128 two-component relocation diagnostic; never publishes a delivery."""
import argparse
import importlib
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--soft-artifact-dir',type=Path,required=True)
    p.add_argument('--silhouettes',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-evaluations',type=int,default=128)
    p.add_argument('--no-joint-edits',action='store_true')
    a=p.parse_args(argv)
    if not 0<=a.max_evaluations<=1024: p.error('max-evaluations must be in [0,1024]')
    source,output=a.soft_artifact_dir.resolve(),a.output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        p.error('output must be separate from immutable source tree')
    return a


def require_two_islands(islands):
    if len(islands)!=2:
        raise ValueError('probe requires exactly two non-main islands')
    return islands


def main(argv=None):
    a=parse_args(argv)
    from experiments.shadowart_vnext.artifacts import create_run,write_json
    from experiments.shadowart_vnext.provenance import freeze_sources
    import numpy as np
    source=a.soft_artifact_dir.resolve(); output=create_run(a.output.resolve())
    package,hashes=freeze_sources(output/'source_snapshot',ROOT/'experiments/shadowart_vnext',__file__)
    def module(name): return importlib.import_module(package+'.'+name)
    contracts=module('contracts'); file_hash=contracts.file_hash
    report={'status':'RUNNING','canonical_delivery_published':False,'structural_status':'NOT_REQUESTED',
            'paper_method_equivalence':False,'physical_print_validated':False,'source_sha256':hashes,
            'source_execution':'captured_source_snapshot','python_executable':sys.executable,
            'python_version':platform.python_version(),'arguments':{'soft_artifact_dir':str(source),
            'max_evaluations':a.max_evaluations,'joint_edits':not a.no_joint_edits,'image_size':1024,
            'islands':2,'independent_transactions':True},'candidates':{}}
    write_json(output/'REPORT.json',report)
    try:
        contract=contracts.CameraContract.from_dict(json.loads((source/'CAMERA_CONTRACT.json').read_text()))
        targets=contracts.TargetSet.from_paths(a.silhouettes)
        if (targets.input_hashes!=contract.input_hashes or tuple(contracts.mask_hash(m) for m in
                targets.render(contract.cameras,contract.image_size,contract.layout))!=contract.target_hashes):
            raise ValueError('FAILED_TARGET_HASH')
        if contract.volume_resolution!=128: raise ValueError('native 128 required')
        fields=module('artifacts').load_soft(source,contract)
        source_hashes={f.name:file_hash(f) for f in source.iterdir() if f.is_file()}
        closed,closure=module('ray_closure').shadow_closure(fields['occupancy'],contract.cameras,
                                                          contract.extent,minimum_block_width=2)
        np.save(output/'diagnostics/closure_xyz.npy',closed,allow_pickle=False)
        write_json(output/'CAMERA_CONTRACT.json',contract.to_dict())
        masks=targets.render(contract.cameras,1024,contract.layout)
        report.update(native_resolution=128,camera_digest=contract.digest,input_sha256=source_hashes,
                      closure=closure,initial_hard_quality=module('layout').iou_report(
                          module('rendering').render_binary(closed,contract.cameras,1024,contract.extent),masks))
        write_json(output/'REPORT.json',report)
        islands=require_two_islands(module('regional_candidates').component_order(closed,2,order='largest')[:2])
        report['selected_islands']=[{'voxels':int(len(points)),
            'mask_sha256':contracts.mask_hash(np.isin(np.arange(closed.size).reshape(closed.shape),
                np.ravel_multi_index(points.T,closed.shape)))} for points in islands]
        write_json(output/'REPORT.json',report)
        for index,points in enumerate(islands,1):
            stage=output/f'diagnostics/island_{index}'; stage.mkdir()
            selected=np.zeros_like(closed); selected[tuple(points.T)]=True
            np.save(stage/'selected_xyz.npy',selected,allow_pickle=False)
            def progress(row):
                if row['id']%16==0:
                    write_json(stage/'latest.json',row)
                    print(json.dumps({'island':index,'candidate':row['id'],
                                      'quality':row['quality']}),flush=True)
            result,trace,candidate=module('material_relocation').relocate_component(closed,fields['guidance'],
                selected,masks,contract.cameras,extent=contract.extent,longest_mm=100.,
                max_evaluations=a.max_evaluations,joint_edits=not a.no_joint_edits,callback=progress)
            for name,array in [('result_xyz',result),('candidate_xyz',candidate),
                               ('added_xyz',candidate & ~closed),('removed_xyz',closed & ~candidate)]:
                np.save(stage/(name+'.npy'),array,allow_pickle=False)
            write_json(stage/'TRACE.json',trace)
            row={k:v for k,v in trace.items() if k not in ('history','audits')}
            report['candidates'][f'island_{index}']=row
            write_json(output/'REPORT.json',report)
            print(json.dumps({'island':index,'accepted':trace['accepted'],
                              'hard_quality':trace['candidate_quality'],'reasons':trace['reasons']}),flush=True)
            mesh=module('geometry').occupancy_to_mesh(candidate,extent=contract.extent)
            qa,_=module('delivery').audit_serialized_mesh(mesh,contract,targets,stage/'serialized',
                geometry_settings={'method':trace['algorithm'],'max_evaluations':a.max_evaluations,
                    'joint_edits':not a.no_joint_edits,'selected_component_sha256':file_hash(stage/'selected_xyz.npy'),
                    'candidate_accepted_as_transaction':trace['accepted'],'diagnostic_only':True,
                    'acceptance_protocol':'continuous_mesh_v1','longest_mm':100.,'min_wall_mm':.9,'min_neck_mm':1.35})
            row.update(mesh_quality=qa['quality'],mesh_qa_passed=qa['passed'])
            write_json(output/'REPORT.json',report)
            print(json.dumps({'island':index,'mesh_quality':qa['quality'],'mesh_qa_passed':qa['passed']}),flush=True)
        if source_hashes!={f.name:file_hash(f) for f in source.iterdir() if f.is_file()}:
            raise RuntimeError('SOURCE_ARTIFACT_CHANGED')
        report.update(status='DIAGNOSTIC_COMPLETED_NOT_DELIVERED',source_hashes_unchanged=True)
    except Exception as error:
        report.update(status='ERROR',error=str(error)); write_json(output/'REPORT.json',report)
        raise
    write_json(output/'REPORT.json',report)
    return 2


if __name__=='__main__': raise SystemExit(main())
