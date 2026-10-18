"""Equal-resolution point-ray stage comparison on immutable physical artifacts."""
import json
import platform
import sys
from pathlib import Path

import numpy as np
from .artifacts import load_soft,write_json
from .contracts import CameraContract,TargetSet,mask_hash,file_hash
from .geometry import occupancy_to_mesh,uniform_print_transform
from .rendering import render_binary,render_mesh
from .layout import iou_report
from .projection_ceiling import cardinal_ceiling
from .ray_closure import shadow_closure
from .cell_boundary import cell_boundary_mesh
from .delivery import audit_serialized_mesh
from .manufacturing import thickness_audit


def _serialized_reload_diagnostic(mesh, files, transform, contract, target_masks, stage):
    """Compare each exported physical mesh with its pre-export canonical mesh.

    This is deliberately supplemental to delivery QA: it always records all
    requested diagnostic resolutions, including 1024 when the lower pair is
    already stable.  The inverse is the exact uniform print transform used for
    export, so both rasterizations occupy the same canonical camera frame.
    """
    import trimesh

    inverse = np.linalg.inv(transform)
    rows = {}
    for extension in ('obj', 'stl'):
        loaded = trimesh.load(files[extension], force='mesh', process=True)
        canonical = loaded.copy()
        canonical.apply_transform(inverse)
        evaluations = []
        for size in (256, 512, 1024):
            pre_export = render_mesh(mesh, contract.cameras, size)
            reloaded = render_mesh(canonical, contract.cameras, size)
            pre_score = iou_report(pre_export, target_masks[size])
            reload_score = iou_report(reloaded, target_masks[size])
            pixels_disagree = np.count_nonzero(pre_export != reloaded, axis=(1, 2))
            np.save(stage / f'{extension}_preexport_canonical_{size}.npy', pre_export,
                    allow_pickle=False)
            np.save(stage / f'{extension}_reloaded_canonical_{size}.npy', reloaded,
                    allow_pickle=False)
            evaluations.append({
                'image_size': size,
                'preexport': pre_score,
                'reloaded': reload_score,
                'masks_identical': bool(not pixels_disagree.any()),
                'pixels_disagree': pixels_disagree.astype(int).tolist(),
                'per_view_iou_delta': (np.asarray(reload_score['per_view_iou']) -
                                       np.asarray(pre_score['per_view_iou'])).tolist(),
            })
        rows[extension] = {
            'sha256': file_hash(files[extension]),
            'inverse_canonical_transform': inverse.tolist(),
            'evaluations': evaluations,
        }
    return rows


def execute(args,output,source_hashes):
    source=Path(args.soft_artifact_dir).resolve()
    contract=CameraContract.from_dict(json.loads((source/'CAMERA_CONTRACT.json').read_text()))
    targets=TargetSet.from_paths(args.silhouettes)
    if targets.input_hashes!=contract.input_hashes or tuple(mask_hash(m) for m in targets.render(
            contract.cameras,contract.image_size,contract.layout))!=contract.target_hashes:
        raise ValueError('FAILED_TARGET_HASH')
    fields=load_soft(source,contract)
    effective_closure_width=min(args.closure_width,fields['occupancy'].shape[0])
    hashes={p.name:file_hash(p) for p in source.iterdir() if p.is_file()}
    report={'status':'RUNNING','scope':'shadowart_geometry_only',
            'structural_status':'NOT_REQUESTED','canonical_delivery_published':False,
            'method_equivalent_to_paper':False,'physical_print_validated':False,
            'native_resolution':contract.volume_resolution,'camera_digest':contract.digest,
            'soft_source':str(source),'input_files_sha256':hashes,
            'source_sha256':source_hashes,'source_execution':'captured_source_snapshot',
            'python_executable':sys.executable,'python_version':platform.python_version(),
            'arguments':{'restore_steps':args.restore_steps,
                         'working_image_size':args.working_image_size,
                         'closure_width':args.closure_width,
                         'effective_closure_width':effective_closure_width},
            'stages':{},'projection_ceilings':[],'hard_boundary_disagreement':[],
            'candidates':{},'note':'Diagnostic-only runner; no automatic publication or slicing.'}
    if args.probability_surface:
        report['arguments']['probability_surface']=True
        report['probability_surface_protocol']={'level':.5,'interpolation':'native_cell_center_marching_cubes',
            'padding_probability':0.,'upsampled':False,'source_rewritten':False,
            'manufacturing_source_field_is_not_mesh_revoxelization':True}
    write_json(output/'CAMERA_CONTRACT.json',contract.to_dict())
    write_json(output/'REPORT.json',report)
    v=fields['occupancy']; sizes=(256,512,1024)
    target_masks={s:targets.render(contract.cameras,s,contract.layout) for s in sizes}
    for s in sizes:
        report['projection_ceilings'].append(cardinal_ceiling(target_masks[s],contract.cameras,v.shape[0],contract.extent))
    boundary=cell_boundary_mesh(v,contract.extent)
    sdf=occupancy_to_mesh(v,contract.extent)
    closed,trace=shadow_closure(v,contract.cameras,contract.extent,
                                minimum_block_width=min(args.closure_width,v.shape[0]))
    write_json(output/'diagnostics/closure_trace.json',trace)
    np.save(output/'diagnostics/closure_occupancy.npy',closed,allow_pickle=False)
    np.save(output/'diagnostics/closure_added_indices.npy',np.flatnonzero(closed & ~v),allow_pickle=False)
    closure_mesh=occupancy_to_mesh(closed,contract.extent)
    restored=restored_mesh=None
    if args.restore_steps:
        from .ray_restore import restore
        restored,restore_trace=restore(closed,fields['guidance'],target_masks[args.working_image_size],contract.cameras,
                    extent=contract.extent,width=max(1,int(np.ceil(1.35/(100/v.shape[0])))),
                    steps=args.restore_steps,mesh_feedback=True,
                    callback=lambda row: print(json.dumps({'stage':'ray_restore',**row}),flush=True))
        restored_mesh=occupancy_to_mesh(restored,contract.extent)
        write_json(output/'diagnostics/restore_trace.json',restore_trace)
        np.save(output/'diagnostics/restored_occupancy.npy',restored,allow_pickle=False)
    stages={'source_hard':v,'source_cell_boundary':boundary,'source_sdf':sdf,
            'closure_hard':closed,'closure_sdf':closure_mesh}
    probability_mesh=None
    if args.probability_surface:
        from .continuous_surface import probability_surface
        probability_mesh=probability_surface(fields['guidance'],contract.extent)
        stages['source_probability']=probability_mesh
    if restored is not None:
        stages.update(restored_hard=restored,restored_sdf=restored_mesh)
    for stage,value in stages.items():
        rows=[]
        for size in sizes:
            pred=(render_binary(value,contract.cameras,size,contract.extent) if isinstance(value,np.ndarray)
                  else render_mesh(value,contract.cameras,size))
            rows.append(iou_report(pred,target_masks[size]))
            np.save(output/f'diagnostics/{stage}_{size}.npy',pred,allow_pickle=False)
            print(json.dumps({'stage':stage,**rows[-1]}),flush=True)
        report['stages'][stage]=rows
        write_json(output/'REPORT.json',report)
    for size in sizes:
        hard=np.load(output/f'diagnostics/source_hard_{size}.npy')
        surface=np.load(output/f'diagnostics/source_cell_boundary_{size}.npy')
        report['hard_boundary_disagreement'].append({'image_size':size,
                  'pixels_per_view':np.count_nonzero(hard!=surface,axis=(1,2)).tolist()})
    candidates=[('source',v,sdf),('closure',closed,closure_mesh)]
    if restored is not None: candidates.append(('restored',restored,restored_mesh))
    if probability_mesh is not None: candidates.append(('source_probability',v,probability_mesh))
    for name,volume,mesh in candidates:
        _,transform=uniform_print_transform(mesh,100.)
        physical=thickness_audit(volume,contract.extent/volume.shape[0]*np.linalg.norm(transform[:3,0]))
        if name=='source_probability':
            physical.update(occupancy_diagnostic_scope='source_threshold_field_not_mesh_voxelization',
                            mesh_manufacturing_certified=False)
        write_json(output/f'diagnostics/{name}_manufacturing.json',physical)
        serialized_stage = output / f'diagnostics/{name}_serialized'
        qa,files=audit_serialized_mesh(mesh,contract,targets,serialized_stage,
                                       geometry_settings={'method':name,'diagnostic_only':True})
        reload_rows = _serialized_reload_diagnostic(mesh, files, transform, contract,
                                                     target_masks, serialized_stage)
        report['candidates'][name]={'quality':qa['quality'],'mesh_qa_passed':qa['passed'],
                'occupancy_manufacturing_passed':physical['passed'],
                'components_6':physical['component_count_6'],'thin_voxels':physical['thin_voxels'],
                'serialized_reload':reload_rows}
        if name=='source_probability':
            report['candidates'][name]['occupancy_diagnostic_scope']='source_threshold_field_not_mesh_voxelization'
        quality=qa['quality']
        print(json.dumps({'stage':name+'_serialized','mean_iou':quality.get('mean_iou'),
                          'worst_iou':quality.get('worst_iou'),'converged':quality.get('converged'),
                          'mesh_qa_passed':qa['passed'],
                          'components_6':physical['component_count_6'],
                          'thin_voxels':physical['thin_voxels']}),flush=True)
        write_json(output/'REPORT.json',report)
    if hashes!={p.name:file_hash(p) for p in source.iterdir() if p.is_file()}:
        raise RuntimeError('SOURCE_ARTIFACT_CHANGED_DURING_DIAGNOSTIC')
    report.update(status='DIAGNOSTIC_COMPLETED_NOT_DELIVERED',source_hashes_unchanged=True)
    write_json(output/'REPORT.json',report)
    return 2
