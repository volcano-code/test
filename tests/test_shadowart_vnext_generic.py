"""Small CPU integration diagnostics; no quality/manufacturing witness claim."""
import numpy as np
import pytest
import torch
import trimesh

from experiments.shadowart_vnext.contracts import build_contract
from experiments.shadowart_vnext.geometry import occupancy_to_mesh
from experiments.shadowart_vnext.rendering import render_binary,render_mesh,voxelize_mesh
from experiments.shadowart_vnext.training import train_soft


@pytest.mark.parametrize('n,tsf',[(1,0),(4,0),(4,1),(8,0),(12,0)])
def test_generic_camera_protocol_cpu_training_and_independent_mesh_roundtrip(tmp_path,n,tsf):
    contract=build_contract(n,tsf=tsf,resolution=16,image_size=32)
    # Symmetric analytic target source gives every view a nonempty diagnostic.
    source=trimesh.creation.icosphere(subdivisions=1,radius=.55)
    targets=render_mesh(source,contract.cameras,32)
    high_targets=render_mesh(source,contract.cameras,64)
    previous_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        fields=train_soft(targets,contract.cameras,resolution=16,image_size=32,
                          steps=1,device='cpu',supersampled_targets=high_targets)
    finally:
        torch.set_num_threads(previous_threads)
    assert fields['metadata']['device']=='cpu'
    assert fields['metadata']['view_count']==len(contract.cameras)
    assert fields['occupancy'].any() and fields['occupancy'].dtype==bool
    actual=render_binary(fields['occupancy'],contract.cameras,32)
    assert np.array_equal(actual,fields['hard_projections'])
    assert actual.shape==(len(contract.cameras),32,32)
    for i,camera in enumerate(contract.cameras):
        if camera.derived_from is not None:
            assert np.array_equal(actual[i],actual[camera.derived_from,:,::-1])
    mesh=occupancy_to_mesh(fields['occupancy'])
    before=render_mesh(mesh,contract.cameras,32)
    path=tmp_path/f'protocol_{n}_{tsf}.stl';mesh.export(path)
    loaded=trimesh.load(path,force='mesh')
    after=render_mesh(loaded,contract.cameras,32)
    assert np.array_equal(before,after)
    assert np.all(after.sum(axis=(1,2))>0)
    reconstructed=voxelize_mesh(loaded,16)
    assert np.array_equal(reconstructed,fields['occupancy'])
