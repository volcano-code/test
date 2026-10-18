"""Orchestrate frozen inputs, diagnostics and gated physical mesh delivery."""
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import platform
import traceback

import numpy as np
from PIL import Image

from .artifacts import create_run, write_json, save_soft, load_soft
from .contracts import build_contract, CameraContract, TargetSet, mask_hash, file_hash, digest_json
from .layout import search_layout, iou_report


def execute(args, *, prepared_output, source_hashes):
    output=Path(prepared_output)
    continuous=args.delivery_protocol=='continuous_mesh_v1'
    canonical_settings=(args.image_size==256 and
                        ((continuous and args.resolution in (128,256)) or
                         (not continuous and args.resolution==256)))
    run={'status':'RUNNING','scope':'shadowart_geometry_only','structural_status':'NOT_REQUESTED',
         'physical_print_validated':False,'paper_method_equivalence':False,
         'canonical_delivery_published':False,'engineering_only':not canonical_settings,
         'acceptance_protocol':args.delivery_protocol,
         'arguments':{k:str(v) if isinstance(v,Path) else [str(p) for p in v] if k=='silhouettes' else v
                      for k,v in vars(args).items()},
         'source_sha256':source_hashes,'source_execution':'captured_source_snapshot',
         'python_version':platform.python_version()}
    if continuous:
        run['experimental_vnext']=True
        run['protocol_scope']='continuous_serialized_mesh_authority_native_revoxel_diagnostic'
    write_json(output/'RUN.json',run)
    try:
        targets=TargetSet.from_paths(args.silhouettes)
        contract=build_contract(len(args.silhouettes),tsf=args.tsf,mirror=args.mirror,
                                camera_config=args.camera_config,image_size=args.image_size,
                                resolution=args.resolution)
        if args.soft_artifact_dir or args.layout_contract:
            source=Path(args.soft_artifact_dir)/'CAMERA_CONTRACT.json' if args.soft_artifact_dir else args.layout_contract
            if not source.is_file():
                raise ValueError('LEGACY_ARTIFACT: no physical vNext contract')
            saved=CameraContract.from_dict(json.loads(source.read_text()))
            contract=replace(contract,layout=saved.layout,input_hashes=targets.input_hashes,
                             target_hashes=saved.target_hashes)
            masks=targets.render(contract.cameras,contract.image_size,contract.layout)
            if tuple(mask_hash(m) for m in masks)!=contract.target_hashes:
                raise ValueError('FAILED_TARGET_HASH')
            if saved.digest!=contract.digest:
                raise ValueError('FAILED_CAMERA_CONTRACT')
            run['layout_contract_source']=str(source.resolve())
            if args.soft_artifact_dir:
                fields=load_soft(args.soft_artifact_dir,contract)
                run['soft_artifact_source']=str(args.soft_artifact_dir.resolve())
                run['soft_artifact_manifest_sha256']=fields['source_manifest_sha256']
        else:
            def progress(row):
                if row['index']%16==0:
                    print(json.dumps({'stage':'layout',**row}),flush=True)
            contract,masks,hull,preflight=search_layout(targets,contract,candidates=args.layout_candidates,
                                                       callback=progress)
            write_json(output/'PREFLIGHT.json',preflight)
            np.save(output/'diagnostics'/'visual_hull_xyz.npy',hull,allow_pickle=False)
        write_json(output/'CAMERA_CONTRACT.json',contract.to_dict())
        write_json(output/'TARGETS.json',targets.diagnostics(contract.cameras,contract.image_size,contract.layout))
        for i,mask in enumerate(masks):
            Image.fromarray(mask.astype(np.uint8)*255).save(output/f'target_view_{i}.png')
        run['camera_digest']=contract.digest
        if args.stage=='preflight':
            run['status']='PREFLIGHT_COMPLETED'
            write_json(output/'RUN.json',run)
            print(f'Preflight only: {output}',flush=True)
            return 0
        if not args.soft_artifact_dir:
            from .training import train_soft
            from .losses import LossConfig
            loss=LossConfig(**json.loads(args.loss_config.read_bytes())) if args.loss_config else LossConfig(
                core=args.loss_core,auxiliary=args.auxiliary,aux_weight=args.aux_weight)
            write_json(output/'LOSS_CONFIG.json',loss.to_dict())
            def training_progress(row):
                if row['step']==1 or row['step']%50==0 or row['step']==args.steps:
                    print(json.dumps({'stage':'soft',**row}),flush=True)
                    write_json(output/'diagnostics'/'training_latest.json',row)
            pixel_ss=max(2,int(np.floor(np.sqrt(2)*max(c.scale for c in contract.cameras)*
                         contract.volume_resolution/(contract.extent*contract.image_size)))+1)
            fields=train_soft(masks,contract.cameras,extent=contract.extent,resolution=contract.volume_resolution,
                              image_size=contract.image_size,steps=args.steps,lr=args.lr,seed=args.seed,
                              device=args.device,loss_config=loss,calibration_steps=args.calibration_steps,
                              projection_mode=args.projection_mode,
                              initialization_mode=args.initialization_mode,
                              pixel_supersampling=pixel_ss,
                              supersampled_targets=targets.render(contract.cameras,contract.image_size*pixel_ss,contract.layout),
                              callback=training_progress)
            save_soft(output/'soft_training',fields,contract)
        run['soft_metadata']=fields['metadata']
        from .metrics import soft_density_metrics,occupancy_metrics
        stage_metrics={'soft':soft_density_metrics(fields['density'],contract,targets,fields['metadata']),
                       'hard_occupancy':occupancy_metrics(fields['occupancy'],contract,targets,stage='hard_occupancy')}
        entity_mode = fields['metadata'].get('training_projection',{}).get('mode')
        if entity_mode in ('dense_entity_max', 'dense_entity_continuation'):
            from .entity_projection import entity_projection_metrics
            stage_metrics['entity_projection'] = entity_projection_metrics(fields,contract,targets)
            if entity_mode == 'dense_entity_continuation':
                stage_metrics['entity_projection'].update(
                    stage='entity_projection_actual_final_max',
                    actual_final_training_forward=True,
                    continuation_endpoint_phase='exact_max')
        write_json(output/'STAGE_METRICS.json',stage_metrics)
        if args.stage=='soft':
            run['status']='SOFT_DIAGNOSTIC_COMPLETED'
            write_json(output/'RUN.json',run)
            return 0
        from .geometry import repair_geometry, refine_surface_mesh, occupancy_to_mesh
        from .rendering import render_binary
        from .delivery import audit_serialized_mesh,publish_delivery
        geometry_settings={'method':'native_full_ray_guarded_repair','max_rounds':args.geometry_rounds,
                           'disabled_stages':list(args.disable_geometry),
                           'working_image_size':max(512,contract.image_size),
                           'longest_mm':100.,'min_wall_mm':.9,'min_neck_mm':1.35,
                           'mean_iou_strict':.98,'worst_iou_inclusive':.97,
                           'acceptance_protocol':args.delivery_protocol}
        if args.geometry_method=='joint':
            geometry_settings.update(method='joint_connected_block_replacement_v1',
                                     max_steps=args.joint_steps,block_width_rule='ceil(neck_mm/(100/resolution))')
            geometry_settings.pop('max_rounds')
        elif args.geometry_method=='regional':
            geometry_settings.update(method='regional_constrained_replacement_v1',
                max_regions=args.regional_regions,max_primitives=args.regional_primitives,
                region_order=args.regional_order,
                mean_search_budget=.001,view_search_budget=.002,solver_time_limit=5.,
                solver_backend='scipy.optimize.milp/HiGHS',budget_reference='immutable_initial_occupancy')
            geometry_settings.pop('max_rounds')
            if getattr(args,'regional_body_reinforcement',False):
                geometry_settings.update(method='regional_body_path_replacement_v1', body_reinforcement=True)
        write_json(output/'GEOMETRY_CONFIG.json',geometry_settings)
        print('Running native connected/thickness-aware geometry candidates',flush=True)
        geometry_targets=targets.render(contract.cameras,geometry_settings['working_image_size'],contract.layout)
        if args.geometry_method=='joint':
            from .joint_geometry import repair_joint
            def joint_progress(row):
                if row['step']==1 or row['step']%8==0:
                    print(json.dumps({'stage':'joint_geometry',**row}),flush=True)
                    write_json(output/'diagnostics'/'joint_latest.json',row)
            volume,trace,connected=repair_joint(fields['occupancy'],fields['guidance'],geometry_targets,
                contract.cameras,extent=contract.extent,longest_mm=100.,max_steps=args.joint_steps,
                callback=joint_progress)
            np.save(output/'diagnostics'/'joint_connected_candidate_xyz.npy',connected,allow_pickle=False)
            stage_metrics['joint_connected_candidate']=occupancy_metrics(connected,contract,targets,
                stage='uncommitted_connected_transaction_candidate')
        elif args.geometry_method=='regional':
            from .regional_geometry import repair_regional
            def regional_progress(row):
                brief={k:row[k] for k in ('region','accepted','primitive_count')}
                brief.update({k:row[k] for k in ('quality','physical_before','physical_after') if k in row})
                print(json.dumps({'stage':'regional_geometry',**brief}),flush=True)
                write_json(output/'diagnostics'/'regional_latest.json',row)
            volume,trace,search=repair_regional(fields['occupancy'],fields['guidance'],geometry_targets,
                contract.cameras,extent=contract.extent,longest_mm=100.,max_regions=args.regional_regions,
                max_primitives=args.regional_primitives,callback=regional_progress,region_order=args.regional_order,
                **({'body_reinforcement':True} if getattr(args,'regional_body_reinforcement',False) else {}))
            np.save(output/'diagnostics'/'regional_search_xyz.npy',search,allow_pickle=False)
            stage_metrics['regional_search']=occupancy_metrics(search,contract,targets,
                stage='unpublished_manufacturing_search')
        else:
            volume,trace=repair_geometry(fields['occupancy'],fields['guidance'],geometry_targets,contract.cameras,
                                        extent=contract.extent,longest_mm=100.,max_rounds=args.geometry_rounds,
                                        disabled_stages=tuple(s for s in args.disable_geometry if s!='surface'))
        write_json(output/'diagnostics'/'geometry_trace.json',trace)
        np.save(output/'diagnostics'/'postprocessed_voxels_xyz.npy',volume,allow_pickle=False)
        run['postprocessed_occupancy']=iou_report(render_binary(volume,contract.cameras,contract.image_size,
                                                               extent=contract.extent),masks)
        stage_metrics['postprocessed_occupancy']=occupancy_metrics(volume,contract,targets,stage='postprocessed_occupancy')
        write_json(output/'STAGE_METRICS.json',stage_metrics)
        if not volume.any():
            raise ValueError('EMPTY_GEOMETRY_CANDIDATE')
        if 'surface' in args.disable_geometry:
            mesh=occupancy_to_mesh(volume,extent=contract.extent)
            surface_trace={'status':'DISABLED_FOR_ABLATION','accepted':False}
        else:
            mesh,surface_trace=refine_surface_mesh(volume,geometry_targets,contract.cameras,extent=contract.extent)
        write_json(output/'diagnostics'/'surface_trace.json',surface_trace)
        print('Auditing serialized complete mesh with independent projection',flush=True)
        qa,files=audit_serialized_mesh(mesh,contract,targets,output/'diagnostics'/'mesh',
                                       geometry_settings=geometry_settings)
        write_json(output/'QA.json',qa)
        stage_metrics['final_serialized_mesh']={'scope':'independent_serialized_mesh_QA',
                                               'quality':qa['quality'],'passed':qa['passed']}
        write_json(output/'STAGE_METRICS.json',stage_metrics)
        run['final_quality']=qa['quality']
        from .handoff import save_handoff
        voxel_path=output/'diagnostics'/'mesh'/'stl_reloaded_voxels_xyz.npy'
        if voxel_path.exists():
            reloaded=np.load(voxel_path,allow_pickle=False)
            if reloaded.any():
                save_handoff(output/'diagnostics'/'structural_handoff',reloaded,qa['canonical_to_print_mm'],
                             extent=contract.extent,mesh_sha256=qa['mesh_sha256'],camera_digest=contract.digest,
                             geometry_passed=qa['passed'])
        if not qa['passed']:
            run['status']='FAILED_FINAL_GEOMETRY_OR_MANUFACTURING_QA'
            run['failure_reasons']={ext:{'topology':row['topology']['reasons'],
                'quality':row['quality'],'raster_quality':row['raster_quality'],
                'thickness':row['thickness']['reasons'],'thickness_review':row['thickness']['review_reasons']}
                for ext,row in qa['serialized'].items()}
            write_json(output/'RUN.json',run)
            print(json.dumps({'status':run['status'],'final_quality':qa['quality'],'output':str(output)}),flush=True)
            return 2
        if not canonical_settings:
            raise ValueError('ENGINEERING_SETTINGS_NOT_CANONICAL')
        from .slicer import run_slicer
        review=None
        review_record=None
        if args.support_review:
            review_bytes=args.support_review.read_bytes()
            review=json.loads(review_bytes)
            staged_review=output/'diagnostics'/'support_review.json'
            with staged_review.open('xb') as stream:
                stream.write(review_bytes)
            review_record={'path':str(staged_review),'original_path':str(args.support_review.resolve()),
                           'sha256':hashlib.sha256(review_bytes).hexdigest(),
                           'reviewer':review.get('reviewer')}
        slicing=run_slicer(files['stl'],output/'diagnostics'/'fabrication',
                           appimage=args.bambu_appimage,support_review=review)
        save_handoff(output/'diagnostics'/'structural_handoff',reloaded,qa['canonical_to_print_mm'],
                     extent=contract.extent,mesh_sha256=qa['mesh_sha256'],camera_digest=contract.digest,
                     geometry_passed=qa['passed'],slicing=slicing)
        topology=qa['serialized']['stl']['topology']
        checks={'one_body':True,'watertight':topology['watertight'],
                'winding_consistent':topology['winding_consistent'],
                'self_intersections_clear':topology['self_intersection']['status']=='PASS',
                'thickness_pass':True,'slicer_pass':bool(slicing.get('passed')),
                'support_removal_pass':bool(slicing.get('support_removal_pass'))}
        manufacturing={'passed':all(checks.values()),'status':'PASS' if all(checks.values()) else 'REVIEW_REQUIRED',
                       'mesh_sha256':qa['mesh_sha256'],'checks':checks,'slicing':slicing,
                       'camera_digest':contract.digest,'target_hashes':list(contract.target_hashes),
                       'geometry_digest':digest_json(geometry_settings),'physical_print_validated':False,
                       'structural_status':'NOT_REQUESTED',
                       'acceptance_protocol':args.delivery_protocol,
                       'native_optimization_resolution':contract.volume_resolution}
        if review_record:
            manufacturing['support_removal_review']=review_record
        write_json(output/'MANUFACTURING.json',manufacturing)
        if not manufacturing['passed']:
            run['status']='MANUFACTURING_REVIEW_REQUIRED'
            write_json(output/'RUN.json',run)
            return 2
        run['delivery']=publish_delivery(output,qa,manufacturing,files)
        run['canonical_delivery_published']=True
        run['status']='DIGITAL_MANUFACTURING_PASS'
        write_json(output/'RUN.json',run)
        return 0
    except Exception as error:
        run['status']='FAILED'
        run['failure_reason']=str(error)
        run['exception_type']=type(error).__name__
        write_json(output/'RUN.json',run)
        write_json(output/'diagnostics'/'exception.json',{'traceback':traceback.format_exc()})
        print(f'FAILED: {error}\nDiagnostics: {output}',flush=True)
        return 2
