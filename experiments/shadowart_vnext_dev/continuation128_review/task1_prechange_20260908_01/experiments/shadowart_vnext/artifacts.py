"""New immutable artifact handoff; legacy proposals are never relabelled."""
from pathlib import Path
import json
import hashlib
import io
import os
import tempfile
import math
import re

import numpy as np

from .contracts import CameraContract, VERSION, digest_json, file_hash


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    data = json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    fd,name = tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(data)
        os.replace(name,path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def create_run(path):
    path = Path(path).absolute()
    path.mkdir(parents=True,exist_ok=False)
    (path/'diagnostics').mkdir()
    return path


def _valid_digest(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _physical_provenance(metadata, history, contract):
    """Validate declared trainer semantics, not cryptographic producer authenticity.

    Historical source digests need not equal today's working tree. Trusted run
    source snapshots/external attestations belong to the pipeline; self-declared
    hashes here bind records and detect byte changes, not an honest producer.
    """
    from .losses import LossConfig
    if not isinstance(metadata,dict):
        raise ValueError('MISSING_PHYSICAL_PROVENANCE_METADATA')
    conventions={'version':'shadowart_vnext_soft_v1','optimizer':'Adam_default_betas_eps',
                 'view_update_mode':'joint_view_equal','checkpoint_selection':'final_step',
                 'density_units':'inverse_canonical_length',
                 'density_rule':'softplus(logits)/(extent/256)',
                 'occupancy_rule':'sigmoid(logits)>=0.5'}
    if any(metadata.get(k)!=v for k,v in conventions.items()):
        raise ValueError('INVALID_PHYSICAL_PROVENANCE_CONVENTION')
    expected={'resolution':contract.volume_resolution,'image_size':contract.image_size,
              'view_count':len(contract.cameras),'extent':contract.extent,
              'reference_length':contract.extent/256.,'occupancy_reference_opacity':.5}
    for key,value in expected.items():
        actual=metadata.get(key)
        if isinstance(actual,bool) or not isinstance(actual,(int,float)) or not math.isfinite(actual) or not math.isclose(actual,value,rel_tol=0,abs_tol=1e-12):
            raise ValueError('PHYSICAL_PROVENANCE_CONTRACT_MISMATCH: '+key)
    for key in ('steps','export_step','view_chunk_size','pixel_supersampling'):
        if type(metadata.get(key)) is not int or metadata[key]<=0:
            raise ValueError('INVALID_PHYSICAL_PROVENANCE_SETTING: '+key)
    if metadata['export_step']!=metadata['steps']:
        raise ValueError('INVALID_PHYSICAL_PROVENANCE_FINAL_STEP')
    if type(metadata.get('seed')) is not int or not 0<=metadata['seed']<2**63:
        raise ValueError('INVALID_PHYSICAL_PROVENANCE_SEED')
    lr=metadata.get('lr')
    if isinstance(lr,bool) or not isinstance(lr,(int,float)) or not math.isfinite(lr) or lr<=0:
        raise ValueError('INVALID_PHYSICAL_PROVENANCE_LEARNING_RATE')
    sources=metadata.get('source_hashes')
    if not isinstance(sources,dict) or not all(_valid_digest(sources.get(k)) for k in ('training.py','losses.py','rendering.py')):
        raise ValueError('MISSING_PHYSICAL_PROVENANCE_SOURCE_HASHES')
    if (len(contract.input_hashes)!=contract.base_count or not all(map(_valid_digest,contract.input_hashes))
            or len(contract.target_hashes)!=len(contract.cameras) or not all(map(_valid_digest,contract.target_hashes))
            or metadata.get('target_hashes')!=list(contract.target_hashes)):
        raise ValueError('PHYSICAL_PROVENANCE_TARGET_OR_INPUT_HASH_MISMATCH')
    environment=metadata.get('environment')
    if (not isinstance(environment,dict) or environment.get('device_type') not in ('cpu','cuda')
            or not isinstance(environment.get('torch_version'),str) or not environment['torch_version']
            or not isinstance(environment.get('device_name'),str) or not environment['device_name']
            or not isinstance(metadata.get('device'),str) or environment.get('device')!=metadata['device']
            or metadata['device'].split(':')[0]!=environment['device_type']
            or any(type(environment.get(k)) is not bool for k in ('deterministic_algorithms','cudnn_deterministic'))):
        raise ValueError('INVALID_PHYSICAL_PROVENANCE_ENVIRONMENT')
    loss=metadata.get('loss_config')
    try:
        if not isinstance(loss,dict) or set(loss)!=set(LossConfig().to_dict()):
            raise ValueError('incomplete loss configuration')
        LossConfig(**loss)
    except (TypeError,ValueError) as error:
        raise ValueError('INVALID_PHYSICAL_PROVENANCE_LOSS_CONFIG') from error
    initialization=metadata.get('initialization')
    init_required={'method':'canonical_visual_hull','inside_logit':2.,'outside_logit':-8.,
                   'noise_std':.01,'post_init_hull_clipping':False,
                   'hull_image_size':contract.image_size*metadata['pixel_supersampling']}
    if (not isinstance(initialization,dict) or any(initialization.get(k)!=v for k,v in init_required.items())
            or initialization.get('target_source') not in ('nearest_nominal_binary','caller_source_derived_binary')
            or not isinstance(initialization.get('target_hashes'),list)
            or len(initialization['target_hashes'])!=len(contract.cameras)
            or not all(map(_valid_digest,initialization['target_hashes']))):
        raise ValueError('INVALID_PHYSICAL_PROVENANCE_INITIALIZATION')
    if not isinstance(history,list) or len(history)!=metadata['steps']:
        raise ValueError('INVALID_PHYSICAL_PROVENANCE_HISTORY_LENGTH')
    for index,row in enumerate(history,1):
        norm=row.get('gradient_norm') if isinstance(row,dict) else None
        if (not isinstance(row,dict) or type(row.get('step')) is not int or row['step']!=index
                or isinstance(norm,bool) or not isinstance(norm,(int,float)) or not math.isfinite(norm) or norm<0):
            raise ValueError('INVALID_PHYSICAL_PROVENANCE_HISTORY_STEP')
    # Historical v1 records have neither field. New probe records must declare
    # their actual forward/gradient semantics together, not relabel a trajectory.
    if 'training_projection' in metadata or 'initial_logits_sha256' in metadata:
        projection = metadata.get('training_projection')
        if not isinstance(projection, dict) or not _valid_digest(metadata.get('initial_logits_sha256')):
            raise ValueError('INVALID_PROJECTION_PROVENANCE_HEADER')
        mode = projection.get('mode')
        if mode not in ('area_soft', 'dense_soft', 'dense_hard_st', 'dense_entity_max'):
            raise ValueError('INVALID_PROJECTION_PROVENANCE_MODE')
        if mode == 'dense_entity_max' and not _valid_digest(sources.get('entity_projection.py')):
            raise ValueError('INVALID_PROJECTION_PROVENANCE_ENTITY_SOURCE')
        dense = mode != 'area_soft'
        hard = mode == 'dense_hard_st'
        expected_projection = {
            'image_size': contract.image_size * metadata['pixel_supersampling'] if dense else contract.image_size,
            'target_rule': 'source_derived_binary' if dense else 'nominal_binary',
            'target_hashes': initialization['target_hashes'] if dense else metadata['target_hashes'],
            'hard_start_fraction': .25 if hard else None,
            'hard_full_fraction': .5 if hard else None,
            'backward_rule': ('ray_max_subgradient' if mode == 'dense_entity_max' else
                              'optical_straight_through_biased' if hard else 'optical_autograd'),
            'soft_projections_are': 'area_optical_diagnostic_not_training_forward',
        }
        if (set(projection) != set(expected_projection) | {'mode'}
                or type(projection.get('image_size')) is not int
                or any(projection.get(k) != v for k, v in expected_projection.items())
                or (dense and (initialization['target_source'] != 'caller_source_derived_binary'
                               or metadata.get('calibration_steps')))):
            raise ValueError('INVALID_PROJECTION_PROVENANCE_SETTINGS')
        for row in history:
            expected_weight = min(1., max(0., (row['step'] / metadata['steps'] - .25) / .25)) if hard else 0.
            weight = row.get('hard_forward_weight')
            if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                    or not math.isfinite(weight) or abs(weight - expected_weight) > 1e-12):
                raise ValueError('INVALID_PROJECTION_PROVENANCE_HISTORY')
    # Also reject nonfinite nested settings/calibration/history records.
    digest_json(metadata); digest_json(history)


def _physical_fields(fields, metadata, history, contract):
    _physical_provenance(metadata,history,contract)
    shape=(contract.volume_resolution,)*3
    arrays={key:np.asarray(fields[key]) for key in ('density','guidance','occupancy')}
    for key,value in arrays.items():
        if value.shape!=shape or value.dtype.kind not in 'bf' or not np.isfinite(value).all():
            raise ValueError('INVALID_PHYSICAL_FIELD_CONVENTION: '+key)
    density,guidance,occupancy=(arrays[k] for k in ('density','guidance','occupancy'))
    if density.dtype.kind!='f' or guidance.dtype.kind!='f' or occupancy.dtype!=bool:
        raise ValueError('INVALID_PHYSICAL_FIELD_DTYPE_CONVENTION')
    if np.any(density<0) or np.any(guidance<0) or np.any(guidance>1):
        raise ValueError('INVALID_PHYSICAL_FIELD_RANGE_CONVENTION')
    # Chunk float64 comparison to bound temporary memory for native256 arrays.
    flat_density=density.reshape(-1); flat_guidance=guidance.reshape(-1); flat_occupancy=occupancy.reshape(-1)
    for start in range(0,density.size,1_048_576):
        end=start+1_048_576
        expected=-np.expm1(-flat_density[start:end].astype(np.float64)*(contract.extent/256.))
        if not np.allclose(flat_guidance[start:end],expected,rtol=2e-6,atol=2e-7):
            raise ValueError('PHYSICAL_DENSITY_GUIDANCE_CONSISTENCY_FAILED')
        # Use stored float guidance for the exact threshold; density/opacity may
        # differ by float32 rounding at precisely0.5, which is covered above.
        if not np.array_equal(flat_occupancy[start:end],flat_guidance[start:end]>=.5):
            raise ValueError('PHYSICAL_OCCUPANCY_GUIDANCE_CONSISTENCY_FAILED')


def save_soft(directory, fields, contract):
    metadata=fields.get('metadata')
    _physical_fields(fields,metadata,fields.get('history'),contract)
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    shape = (contract.volume_resolution,)*3
    hashes = {}
    for key in ('occupancy','density','guidance'):
        a = np.asarray(fields[key])
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f'invalid soft {key}')
        if key == 'occupancy' and a.dtype != bool:
            raise ValueError('occupancy must already be binary; no artifact reinterpretation')
        if key == 'density' and (a<0).any():
            raise ValueError('negative physical density')
        if key == 'guidance' and ((a<0)|(a>1)).any():
            raise ValueError('guidance must be in[0,1]')
        name = key+'.npy'
        np.save(directory/name,a,allow_pickle=False)
        hashes[name] = file_hash(directory/name)
    for key in ('soft_projections','hard_projections'):
        if key in fields:
            a = np.asarray(fields[key])
            if not np.isfinite(a).all():
                raise ValueError('nonfinite saved projections')
            np.save(directory/(key+'.npy'),a,allow_pickle=False)
            hashes[key+'.npy'] = file_hash(directory/(key+'.npy'))
    write_json(directory/'CAMERA_CONTRACT.json',contract.to_dict())
    hashes['CAMERA_CONTRACT.json'] = file_hash(directory/'CAMERA_CONTRACT.json')
    manifest = {'version':VERSION,'camera_digest':contract.digest,'files':hashes,
                'metadata':metadata,'metadata_digest':digest_json(metadata),
                'history':fields.get('history',[]),'artifact_policy':'read_only_original_fields'}
    write_json(directory/'SOFT_MANIFEST.json',manifest)
    return manifest


def load_soft(directory, contract):
    directory = Path(directory).resolve()
    path = directory/'SOFT_MANIFEST.json'
    if not path.is_file():
        raise ValueError('LEGACY_ARTIFACT: diagnostic-only; cannot replay as physical vNext')
    manifest_bytes=path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (not isinstance(manifest,dict) or not {'metadata','metadata_digest','history','files'}.issubset(manifest)
            or not isinstance(manifest['files'],dict)
            or manifest.get('artifact_policy')!='read_only_original_fields'):
        raise ValueError('INVALID_PHYSICAL_PROVENANCE_ENVELOPE')
    if manifest.get('version') != VERSION:
        raise ValueError('LEGACY_ARTIFACT: unsupported version')
    if manifest.get('camera_digest') != contract.digest:
        raise ValueError('FAILED_CAMERA_CONTRACT')
    if digest_json(manifest['metadata']) != manifest['metadata_digest']:
        raise ValueError('FAILED_METADATA_HASH')
    required={'occupancy.npy','density.npy','guidance.npy','CAMERA_CONTRACT.json'}
    if not required.issubset(manifest['files']):
        raise ValueError('INCOMPLETE_SOFT_ARTIFACT')
    checked={}
    for name,expected in manifest['files'].items():
        p = (directory/name).resolve()
        if p.parent != directory or not p.is_file():
            raise ValueError('FAILED_ARTIFACT_HASH: '+name)
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError('FAILED_ARTIFACT_HASH: '+name)
        if name in required:
            checked[name]=raw
    saved = CameraContract.from_dict(json.loads(checked['CAMERA_CONTRACT.json']))
    if saved.digest != contract.digest:
        raise ValueError('FAILED_CAMERA_CONTRACT')
    fields = {key:np.load(io.BytesIO(checked[key+'.npy']),allow_pickle=False)
              for key in ('occupancy','density','guidance')}
    for key,value in fields.items():
        if value.shape != (contract.volume_resolution,)*3 or not np.isfinite(value).all():
            raise ValueError('INVALID_ARTIFACT_FIELD: '+key)
    if fields['occupancy'].dtype != bool or (fields['density']<0).any():
        raise ValueError('INVALID_ARTIFACT_FIELD_DTYPE_OR_RANGE')
    if ((fields['guidance']<0)|(fields['guidance']>1)).any():
        raise ValueError('INVALID_GUIDANCE_RANGE')
    _physical_fields(fields,manifest.get('metadata'),manifest.get('history'),contract)
    return {**fields,'metadata':manifest['metadata'],'history':manifest['history'],
            'source_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest()}
