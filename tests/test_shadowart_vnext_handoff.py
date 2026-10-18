import numpy as np


def test_handoff_records_physical_axes_mass_and_nonconservative_proxy(tmp_path):
    from experiments.shadowart_vnext.handoff import save_handoff
    v=np.zeros((16,16,16),bool)
    v[4:12,4:12,4:12]=True
    transform=np.array([[100,0,0,0],[0,0,-100,0],[0,100,0,50],[0,0,0,1]],float)
    r=save_handoff(tmp_path,v,transform,extent=1.7,mesh_sha256='a'*64,
                   camera_digest='b'*64,geometry_passed=False)
    assert r['structural_status']=='NOT_REQUESTED'
    assert r['structural_proxy']['resolution']==48
    assert r['structural_proxy']['conservative_strength_bound'] is False
    assert r['gravity_print_m_s2']==[0.,0.,-9.80665]
    assert r['cell_size_m']==1.7/16*100/1000
    assert r['contact']['type']=='candidate_contact_not_rigid_clamp'
    assert r['manufacturing_support_is_structural_support'] is False
    assert np.load(tmp_path/'structural_proxy_xyz_48.npy').shape==(48,48,48)


def test_handoff_never_uses_anisotropic_scale_as_physical_units(tmp_path):
    import pytest
    from experiments.shadowart_vnext.handoff import save_handoff
    with pytest.raises(ValueError):
        save_handoff(tmp_path,np.ones((8,8,8),bool),np.diag([100,50,100,1]),extent=1.7,
                     mesh_sha256='a'*64,camera_digest='b'*64,geometry_passed=True)
def test_degenerate_transform_rejected_without_division_warning(tmp_path):
    import numpy as np
    import pytest
    from experiments.shadowart_vnext.handoff import save_handoff
    matrix=np.eye(4);matrix[:3,:3]=0
    with pytest.raises(ValueError,match='uniformly scaled'):
        save_handoff(tmp_path,np.ones((4,)*3,bool),matrix,extent=1.7,
                     mesh_sha256='x',camera_digest='y',geometry_passed=False)
