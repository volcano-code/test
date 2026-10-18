"""SI/frame-aware future structural handoff; no structural predictions or claims."""
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

from .artifacts import write_json
from .contracts import file_hash


def save_handoff(directory,volume_xyz,canonical_to_print_mm,*,extent,mesh_sha256,
                 camera_digest,geometry_passed,slicing=None):
    import torch
    import torch.nn.functional as F
    v=np.asarray(volume_xyz)
    matrix=np.asarray(canonical_to_print_mm,float)
    if v.dtype!=bool or v.ndim!=3 or len(set(v.shape))!=1 or not v.any():
        raise ValueError('nonempty binary XYZ volume required')
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1]):
        raise ValueError('invalid physical transform')
    scale=float(np.linalg.norm(matrix[:3,0]))
    if scale<=0:
        raise ValueError('physical transform must be right-handed and uniformly scaled')
    rotation=matrix[:3,:3]/scale
    if scale<=0 or not np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-8) or np.linalg.det(rotation)<0:
        raise ValueError('physical transform must be right-handed and uniformly scaled')
    path=Path(directory)
    path.mkdir(parents=True,exist_ok=True)
    pitch=extent/v.shape[0]*scale
    sdf=(ndi.distance_transform_edt(v)-ndi.distance_transform_edt(~v))*pitch/1000.
    np.save(path/'occupancy_xyz.npy',v,allow_pickle=False)
    np.save(path/'sdf_xyz_m.npy',sdf.astype(np.float32),allow_pickle=False)
    tensor=torch.from_numpy(np.ascontiguousarray(v,dtype=np.float32))[None,None]
    proxy=F.adaptive_max_pool3d(tensor,(48,48,48))[0,0].numpy().astype(bool)
    reconstructed=F.interpolate(torch.tensor(proxy,dtype=torch.float32)[None,None],
                                size=v.shape,mode='nearest')[0,0].numpy().astype(bool)
    np.save(path/'structural_proxy_xyz_48.npy',proxy,allow_pickle=False)
    # Mark candidate surface-adjacent cells at the lowest physical height, not fixed FEM nodes.
    coords=np.argwhere(v)
    centers=(coords+.5)*(extent/v.shape[0])-extent/2
    heights=(centers@matrix[2,:3])+matrix[2,3]
    selected=coords[heights<=heights.min()+pitch*.51]
    contact=np.zeros_like(v)
    contact[tuple(selected.T)]=True
    np.save(path/'candidate_contact_xyz.npy',contact,allow_pickle=False)
    source_files={p.name:file_hash(p) for p in sorted(path.glob('*.npy'))}
    result={'version':'shadowart_structural_handoff_v1','structural_status':'NOT_REQUESTED',
            'pino_status':'NOT_REQUESTED','fem_status':'NOT_REQUESTED',
            'geometry_passed':bool(geometry_passed),'mesh_sha256':mesh_sha256,'camera_digest':camera_digest,
            'canonical_to_print_mm':matrix.tolist(),'axes':'canonical_XYZ_Y_up_to_print_XYZ_Z_up',
            'cell_size_m':pitch/1000.,'sdf_sign_convention':'positive_inside_negative_outside',
            'gravity_print_m_s2':[0.,0.,-9.80665],
            'density_kg_m3':1260.,'material_model':'PLA_nominal_density_only_ELASTIC_CONSTANTS_NOT_CALIBRATED',
            'solid_occupancy_mass_upper_model_kg':float(v.sum()*(pitch/1000.)**3*1260.),
            'printed_mass_and_anisotropy':'NOT_DERIVED_FROM_SOLID_OCCUPANCY',
            'contact':{'type':'candidate_contact_not_rigid_clamp','cells':int(contact.sum()),
                       'standing_stability_checked':False},
            'manufacturing_support_is_structural_support':False,
            'structural_proxy':{'resolution':48,'method':'adaptive_max_pool',
                  'conservative_strength_bound':False,'reconstruction_added_cells':int((reconstructed&~v).sum()),
                  'reconstruction_lost_cells':int((v&~reconstructed).sum()),
                  'component_count_6':int(ndi.label(proxy,ndi.generate_binary_structure(3,1))[1])},
            'slicing':slicing,'physical_print_validated':False,'files':source_files,
            'future_edit_policy':'rerun_all_camera_mesh_manufacturing_gates_after_any_structural_edit'}
    write_json(path/'STRUCTURAL_HANDOFF.json',result)
    return result
