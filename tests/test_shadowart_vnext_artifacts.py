import dataclasses
import copy
import json
import numpy as np
import pytest


@pytest.fixture(scope='module')
def trained_artifact():
    from experiments.shadowart_vnext.contracts import build_contract,mask_hash,file_hash
    from experiments.shadowart_vnext.training import train_soft
    targets=np.zeros((1,8,8),bool); targets[:,2:6,2:6]=True
    c=build_contract(1,image_size=8,resolution=8)
    c=dataclasses.replace(c,input_hashes=(file_hash(__file__),),target_hashes=(mask_hash(targets[0]),))
    fields=train_soft(targets,c.cameras,resolution=8,image_size=8,steps=1,device='cpu',
                      supersampled_targets=np.repeat(np.repeat(targets,2,1),2,2))
    return c,fields


def test_soft_artifact_binds_contract_and_all_original_field_hashes(tmp_path,trained_artifact):
    from experiments.shadowart_vnext.artifacts import save_soft, load_soft
    c,fields=trained_artifact
    save_soft(tmp_path/'soft', fields,c)
    got = load_soft(tmp_path/'soft',c)
    np.testing.assert_array_equal(got['occupancy'],fields['occupancy'])
    with pytest.raises(ValueError,match='CONTRACT'):
        load_soft(tmp_path/'soft', dataclasses.replace(c,extent=1.8))
    np.save(tmp_path/'soft'/'density.npy',np.zeros((8,8,8),np.float32))
    with pytest.raises(ValueError,match='HASH'):
        load_soft(tmp_path/'soft',c)


def test_old_artifact_is_not_relabelled_and_output_never_overwritten(tmp_path):
    from experiments.shadowart_vnext.artifacts import load_soft, create_run
    from experiments.shadowart_vnext.contracts import build_contract
    legacy=tmp_path/'legacy'
    legacy.mkdir()
    (legacy/'metrics.json').write_text('{}')
    with pytest.raises(ValueError,match='LEGACY'):
        load_soft(legacy,build_contract(1))
    create_run(tmp_path/'new')
    with pytest.raises(FileExistsError):
        create_run(tmp_path/'new')


@pytest.mark.parametrize('mutation',['missing','placeholder','reference_length','density_rule','optimizer',
                                    'target_hashes','source_hashes','loss_config','environment','final_step','history'])
def test_save_rejects_missing_or_inconsistent_physical_provenance(tmp_path,trained_artifact,mutation):
    from experiments.shadowart_vnext.artifacts import save_soft
    c,original=trained_artifact;fields=copy.deepcopy(original);metadata=fields['metadata']
    if mutation=='missing': fields.pop('metadata')
    elif mutation=='placeholder': fields['metadata']={'test':True}
    elif mutation=='reference_length': metadata['reference_length']=c.extent/c.volume_resolution
    elif mutation=='density_rule': metadata['density_rule']='normalized_per_image'
    elif mutation=='optimizer': metadata['optimizer']='unknown'
    elif mutation=='target_hashes': metadata['target_hashes']=['a'*64]
    elif mutation=='source_hashes': metadata['source_hashes']={}
    elif mutation=='loss_config': metadata['loss_config']={'anything':1}
    elif mutation=='environment': metadata.pop('environment')
    elif mutation=='final_step': metadata['export_step']=2
    elif mutation=='history': fields['history']=[]
    with pytest.raises(ValueError,match='PROVENANCE|CONVENTION|METADATA|HISTORY'):
        save_soft(tmp_path/'bad',fields,c)


@pytest.mark.parametrize('name',['density','guidance','occupancy'])
def test_save_rejects_physical_fields_that_disagree(tmp_path,trained_artifact,name):
    from experiments.shadowart_vnext.artifacts import save_soft
    c,original=trained_artifact;fields=copy.deepcopy(original)
    if name=='occupancy':fields[name]=~fields[name]
    else:fields[name][0,0,0]+=.2
    with pytest.raises(ValueError,match='CONSISTENCY|CONVENTION'):
        save_soft(tmp_path/'bad',fields,c)


@pytest.mark.parametrize('mutation',['metadata','missing_metadata','envelope_policy','density','guidance','occupancy','history'])
def test_load_validates_semantics_even_if_mutated_bytes_are_rehashed(tmp_path,trained_artifact,mutation):
    from experiments.shadowart_vnext.artifacts import save_soft,load_soft
    from experiments.shadowart_vnext.contracts import digest_json,file_hash
    c,fields=trained_artifact;directory=tmp_path/'soft';save_soft(directory,fields,c)
    path=directory/'SOFT_MANIFEST.json';manifest=json.loads(path.read_text())
    if mutation=='metadata':
        manifest['metadata']['view_update_mode']='sequential_last_view'
        manifest['metadata_digest']=digest_json(manifest['metadata'])
    elif mutation=='missing_metadata':manifest.pop('metadata')
    elif mutation=='envelope_policy':manifest['artifact_policy']='reinterpret_legacy_fields'
    elif mutation=='history': manifest['history']=[]
    else:
        array=fields[mutation].copy()
        if mutation=='occupancy':array=~array
        else:array[0,0,0]+=.2
        np.save(directory/(mutation+'.npy'),array,allow_pickle=False)
        manifest['files'][mutation+'.npy']=file_hash(directory/(mutation+'.npy'))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='PROVENANCE|CONVENTION|CONSISTENCY|HISTORY'):
        load_soft(directory,c)
