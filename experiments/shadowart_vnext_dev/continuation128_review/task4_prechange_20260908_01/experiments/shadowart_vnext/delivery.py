"""Independent serialized geometry QA and fail-closed atomic delivery."""
from pathlib import Path
import os
import shutil
import tempfile

import numpy as np
from scipy import ndimage as ndi

from .artifacts import write_json
from .contracts import file_hash, digest_json
from .layout import iou_report, quality_pass


def converged_quality(evaluations):
    if len(evaluations)<2:
        return {'passed':False,'converged':False,'reason':'TWO_RESOLUTIONS_REQUIRED'}
    rows=sorted(evaluations,key=lambda v:v['image_size'])
    sizes=[r['image_size'] for r in rows]
    if len(set(sizes))!=len(sizes) or not {256,512}.issubset(sizes):
        return {'passed':False,'converged':False,'reason':'DISTINCT_256_512_REQUIRED'}
    values=[np.array(r['per_view_iou'],float) for r in rows]
    if any(v.ndim!=1 or not len(v) or not np.isfinite(v).all() or (v<0).any() or (v>1).any() for v in values):
        return {'passed':False,'converged':False,'reason':'INVALID_METRICS'}
    if any(v.shape!=values[0].shape for v in values):
        return {'passed':False,'converged':False,'reason':'VIEW_COUNT_MISMATCH'}
    delta=float(np.max(abs(values[-1]-values[-2])))
    conservative=np.minimum(values[-1],values[-2])
    report={'mean_iou':float(conservative.mean()),'worst_iou':float(conservative.min()),
            'per_view_iou':conservative.tolist(),'resolution_delta':delta,
            'converged':delta<=.001}
    enough=rows[0]['image_size']>=256 and rows[-1]['image_size']>=512
    every_gate=all(quality_pass({'mean_iou':float(v.mean()),'worst_iou':float(v.min())}) for v in values)
    needs_1024=np.max(abs(values[sizes.index(256)]-values[sizes.index(512)]))>.001
    allowed=set(sizes).issubset({256,512,1024}) and (not needs_1024 or 1024 in sizes)
    report['passed']=bool(enough and allowed and every_gate and report['converged'] and quality_pass(report))
    return report


def aggregate_serialized_quality(rows):
    sizes=sorted(set(r['image_size'] for row in rows.values() for r in row['evaluations']))
    missing={ext:[s for s in sizes if s not in {r['image_size'] for r in row['evaluations']}]
             for ext,row in rows.items()}
    missing={ext:sizes for ext,sizes in missing.items() if sizes}
    if missing:
        return {'passed':False,'requires_resolution_for':missing}
    conservative=[]
    for size in sizes:
        values=[r['per_view_iou'] for row in rows.values() for r in row['evaluations']
                if r['image_size']==size]
        conservative.append({'image_size':size,'per_view_iou':np.min(values,axis=0).tolist()})
    return {**converged_quality(conservative),'evaluations':conservative,'requires_resolution_for':{}}


def audit_serialized_mesh(mesh, contract, target_set, stage, *, geometry_settings,
                          serialized_source=None, physical_transform=None):
    import trimesh
    from .geometry import mesh_audit, uniform_print_transform
    from .manufacturing import thickness_audit
    from .rendering import render_mesh, render_binary, voxelize_mesh
    stage=Path(stage)
    stage.mkdir(parents=True,exist_ok=True)
    if serialized_source is None:
        print_mesh,transform=uniform_print_transform(mesh,100.)
    else:
        if set(serialized_source)!={'obj','stl'} or physical_transform is None:
            raise ValueError('complete OBJ/STL and original physical transform required')
        transform=np.asarray(physical_transform,float)
        if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3],[0,0,0,1]):
            raise ValueError('invalid original physical transform')
        norm=float(np.linalg.norm(transform[:3,0]))
        if norm<=0 or np.linalg.det(transform[:3,:3])<=0 or not np.allclose(
                transform[:3,:3].T@transform[:3,:3],np.eye(3)*norm**2,rtol=0,atol=1e-7):
            raise ValueError('physical transform must be uniformly scaled and right handed')
    inverse=np.linalg.inv(transform)
    scale=float(np.linalg.norm(transform[:3,0]))
    files={}
    rows={}
    all_pass=True
    for extension in ('obj','stl'):
        path=stage/('candidate_full.'+extension)
        if serialized_source is None:
            print_mesh.export(path)
        elif Path(serialized_source[extension]).resolve()!=path.resolve():
            raw=Path(serialized_source[extension]).read_bytes()
            with path.open('xb') as stream:
                stream.write(raw)
        files[extension]=str(path)
        loaded=trimesh.load(path,force='mesh',process=True)
        topology=mesh_audit(loaded)
        canonical=loaded.copy()
        canonical.apply_transform(inverse)
        evaluations=[]
        for size in (256,512):
            pred=render_mesh(canonical,contract.cameras,size)
            score=iou_report(pred,target_set.render(contract.cameras,size,contract.layout))
            evaluations.append(score)
            np.save(stage/f'{extension}_projection_{size}.npy',pred,allow_pickle=False)
        if np.max(abs(np.array(evaluations[0]['per_view_iou'])-evaluations[1]['per_view_iou']))>.001:
            pred=render_mesh(canonical,contract.cameras,1024)
            evaluations.append(iou_report(pred,target_set.render(contract.cameras,1024,contract.layout)))
            np.save(stage/f'{extension}_projection_1024.npy',pred,allow_pickle=False)
        quality=converged_quality(evaluations)
        if not topology['passed']:
            rows[extension]={'sha256':file_hash(path),'topology':topology,'evaluations':evaluations,
                             'quality':quality,'raster_evaluations':[],
                             'raster_quality':{'passed':False,'reason':'INVALID_MESH_NOT_VOXELIZED'},
                             'thickness':{'passed':False,'reasons':['INVALID_MESH_NOT_VOXELIZED'],
                                          'review_reasons':[]},
                             'component_count_6':None,'passed':False}
            all_pass=False
            write_json(stage/'QA.partial.json',{'serialized':rows,'passed':False})
            continue
        raster=voxelize_mesh(canonical,contract.volume_resolution,extent=contract.extent)
        np.save(stage/f'{extension}_reloaded_voxels_xyz.npy',raster,allow_pickle=False)
        raster_scores=[iou_report(render_binary(raster,contract.cameras,size,extent=contract.extent),
                                 target_set.render(contract.cameras,size,contract.layout)) for size in (256,512)]
        if np.max(abs(np.array(raster_scores[0]['per_view_iou'])-raster_scores[1]['per_view_iou']))>.001:
            raster_scores.append(iou_report(render_binary(raster,contract.cameras,1024,extent=contract.extent),
                                             target_set.render(contract.cameras,1024,contract.layout)))
        raster_quality=converged_quality(raster_scores)
        thickness=thickness_audit(raster,contract.extent/contract.volume_resolution*scale)
        # Higher-resolution thickness diagnostics are required when all earlier hard gates pass.
        # Center-sampled voxel certificates are explicitly engineering checks, not exact CAD bounds.
        if topology['passed'] and quality['passed'] and raster_quality['passed']:
            fine=voxelize_mesh(canonical,512,extent=contract.extent)
            thickness_fine=thickness_audit(fine,contract.extent/512*scale)
            thickness['mesh_revoxelized_512']=thickness_fine
            thickness['passed']=bool(thickness['passed'] and thickness_fine['passed'])
        body_count=int(ndi.label(raster,ndi.generate_binary_structure(3,1))[1])
        passed=bool(topology['passed'] and quality['passed'] and raster_quality['passed']
                    and thickness['passed'] and body_count==1 and raster.any())
        rows[extension]={'sha256':file_hash(path),'topology':topology,'evaluations':evaluations,
                         'quality':quality,'raster_evaluations':raster_scores,
                         'raster_quality':raster_quality,'thickness':thickness,
                         'component_count_6':body_count,'passed':passed}
        all_pass &= passed
    aggregate=aggregate_serialized_quality(rows)
    for ext,sizes in aggregate['requires_resolution_for'].items():
        canonical=trimesh.load(files[ext],force='mesh',process=True)
        canonical.apply_transform(inverse)
        for size in sizes:
            pred=render_mesh(canonical,contract.cameras,size)
            rows[ext]['evaluations'].append(iou_report(pred,target_set.render(contract.cameras,size,contract.layout)))
        rows[ext]['quality']=converged_quality(rows[ext]['evaluations'])
        rows[ext]['passed']=bool(rows[ext]['passed'] and rows[ext]['quality']['passed'])
    q=aggregate_serialized_quality(rows)
    conservative=q['evaluations']
    all_pass=all(row['passed'] for row in rows.values())
    report={'passed':bool(all_pass and q['passed']),'status':'PASS' if all_pass and q['passed'] else 'FAIL',
            'mesh_sha256':rows['stl']['sha256'],'camera_digest':contract.digest,
            'target_hashes':list(contract.target_hashes),'geometry_digest':digest_json(geometry_settings),
            'independent_serialized_mesh':True,'evaluations':conservative,'quality':q,
            'serialized':rows,'canonical_to_print_mm':transform.tolist(),'uniform_scale':scale,
            'native_resolution':contract.volume_resolution,'physical_print_validated':False,
            'structural_status':'NOT_REQUESTED'}
    write_json(stage/'QA.json',report)
    return report,files


def publish_delivery(output, qa, manufacturing, files):
    """All canonical files become visible in one directory rename, or none do."""
    if not qa.get('passed') or not manufacturing.get('passed'):
        raise ValueError('DELIVERY_GATE_FAILED')
    if not qa.get('independent_serialized_mesh') or not converged_quality(qa.get('evaluations',[]))['passed']:
        raise ValueError('INCOMPLETE_DELIVERY_QA')
    if qa.get('native_resolution')!=256:
        raise ValueError('ENGINEERING_RESOLUTION_NOT_CANONICAL')
    if manufacturing.get('mesh_sha256')!=qa.get('mesh_sha256'):
        raise ValueError('MANUFACTURING_MESH_MISMATCH')
    checks=manufacturing.get('checks',{})
    required=('one_body','watertight','winding_consistent','self_intersections_clear',
              'thickness_pass','slicer_pass','support_removal_pass')
    if not all(checks.get(k) is True for k in required):
        raise ValueError('INCOMPLETE_MANUFACTURING_GATES')
    for ext in ('obj','stl'):
        path=Path(files[ext])
        if not qa['serialized'][ext]['passed'] or file_hash(path)!=qa['serialized'][ext]['sha256']:
            raise ValueError('SERIALIZED_MESH_HASH_MISMATCH')
    if any(manufacturing.get(k)!=qa.get(k) for k in ('camera_digest','target_hashes','geometry_digest')):
        raise ValueError('MANUFACTURING_IDENTITY_MISMATCH')
    output=Path(output)
    destination=output/'delivery'
    if destination.exists():
        raise FileExistsError(destination)
    temporary=Path(tempfile.mkdtemp(prefix='.delivery-',dir=output/'diagnostics'))
    try:
        for ext in ('obj','stl'):
            shutil.copy2(files[ext],temporary/f'delivery_postprocessed.{ext}')
            if file_hash(temporary/f'delivery_postprocessed.{ext}')!=qa['serialized'][ext]['sha256']:
                raise ValueError('STAGED_SERIALIZED_MESH_HASH_MISMATCH')
        import trimesh
        loaded=trimesh.load(temporary/'delivery_postprocessed.stl',force='mesh',process=True)
        positive=[p for p in loaded.split(only_watertight=False) if p.volume>0]
        if len(positive)!=1:
            raise ValueError('EXTERNAL_SHELL_AMBIGUOUS')
        positive[0].export(temporary/'delivery_outer_preview.stl')
        positive[0].export(temporary/'delivery_outer_preview.obj')
        write_json(temporary/'QA.json',qa)
        write_json(temporary/'MANUFACTURING.json',manufacturing)
        write_json(temporary/'DELIVERY.json',{'status':'DIGITAL_MANUFACTURING_PASS',
                   'physical_print_validated':False,'structural_status':'NOT_REQUESTED',
                   'mesh_sha256':qa['mesh_sha256'],'camera_digest':qa['camera_digest']})
        os.replace(temporary,destination)
    except Exception:
        # Preserve staged diagnostic evidence; no broad deletion or overwrite.
        raise
    return {'directory':str(destination),'obj':str(destination/'delivery_postprocessed.obj'),
            'stl':str(destination/'delivery_postprocessed.stl')}
