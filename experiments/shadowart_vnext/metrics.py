"""Clearly separated stage diagnostics; no final geometry/manufacturing claims."""
import numpy as np

from .layout import iou_report

STAGE_SCOPE='stage_only_not_final_mesh_or_manufacturing_QA'


def soft_projection_metrics(projections_alpha, targets_bool):
    """Continuous Jaccard and thresholded projection IoU are different metrics."""
    alpha=np.asarray(projections_alpha,dtype=np.float64)
    target=np.asarray(targets_bool)
    if (alpha.ndim!=3 or alpha.shape!=target.shape or not all(alpha.shape)
            or alpha.shape[1]!=alpha.shape[2] or not np.isfinite(alpha).all()
            or np.any(alpha<0) or np.any(alpha>1) or not np.isin(target,[0,1]).all()):
        raise ValueError('finite physical alpha[0,1] and matching binary [V,H,H] targets required')
    target=target.astype(bool)
    intersection=np.sum(alpha*target,axis=(1,2))
    union=np.sum(alpha,axis=(1,2))+np.sum(target,axis=(1,2))-intersection
    jaccard=np.divide(intersection,union,out=np.ones_like(union),where=union>0)
    return {'stage':'soft_projection','scope':STAGE_SCOPE,
            'continuous_soft_jaccard':{'mean':float(jaccard.mean()),'worst':float(jaccard.min()),
                                      'per_view':jaccard.tolist()},
            'threshold_0_5_projection_iou':iou_report(alpha>=.5,target),
            'threshold_rule':'physical_projection_alpha>=0.5',
            'continuous_definition':'sum(alpha*target)/sum(alpha+target-alpha*target)',
            'empty_union_value':1.,'image_size':int(alpha.shape[-1])}


def soft_density_metrics(density_xyz, contract, target_set, metadata):
    """CPU replay of recorded full-ray supersampling from physical density.

    This retains the area-optical diagnostic for cross-run comparison, including
    dense/binary-forward experiments where it is NOT the training forward.
    Targets are rendered freshly from the original source at nominal resolution.
    Artifact convention/authenticity validation belongs to the artifact loader.
    """
    import torch
    from .training import render_training_projection
    density=np.asarray(density_xyz)
    supersampling=metadata.get('pixel_supersampling')
    if (type(supersampling) is not int or supersampling<=0 or density.dtype.kind!='f'
            or density.shape!=(contract.volume_resolution,)*3
            or not np.isfinite(density).all() or np.any(density<0)):
        raise ValueError('physical density and recorded positive pixel_supersampling required')
    with torch.no_grad():
        tensor=torch.from_numpy(np.array(density,copy=True,order='C'))
        # Bound intermediate image memory in arbitrary-N replay.
        projections=[render_training_projection(tensor,[camera],contract.image_size,
                        extent=contract.extent,pixel_supersampling=supersampling)[0].numpy()
                     for camera in contract.cameras]
    targets=target_set.render(contract.cameras,contract.image_size,contract.layout)
    report=soft_projection_metrics(np.stack(projections),targets)
    report.update(pixel_supersampling=supersampling,
                  projection_rule='area_average_of_complete_ray_Beer_Lambert_alpha',
                  target_rule='original_source_derived_at_nominal_resolution',replay_device='cpu')
    return report


def occupancy_metrics(volume, contract, target_set, *, stage='occupancy'):
    """Source-derived 256/512/conditional1024 binary occupancy diagnostics.

    Do not interpret these volume projections as independent serialized-mesh QA.
    Disagreement triggers extra measurement, never a feasibility/pass claim.
    """
    from .rendering import render_binary
    volume=np.asarray(volume)
    if volume.dtype!=bool or volume.shape!=(contract.volume_resolution,)*3:
        raise ValueError('boolean volume must match contract volume resolution')
    if not isinstance(stage,str) or not stage:
        raise ValueError('nonempty diagnostic stage label required')
    evaluations=[]
    def evaluate(size):
        prediction=render_binary(volume,contract.cameras,size,extent=contract.extent)
        target=target_set.render(contract.cameras,size,contract.layout)
        return iou_report(prediction,target)
    for size in (256,512):
        evaluations.append(evaluate(size))
    delta=float(np.max(np.abs(np.asarray(evaluations[0]['per_view_iou'])-evaluations[1]['per_view_iou'])))
    if delta>.001:
        evaluations.append(evaluate(1024))
    last_delta=float(np.max(np.abs(np.asarray(evaluations[-2]['per_view_iou'])-evaluations[-1]['per_view_iou'])))
    return {'stage':stage,'scope':STAGE_SCOPE,'representation':'XYZ_occupied_cells_full_ray',
            'target_rule':'original_source_derived_at_each_resolution','evaluations':evaluations,
            'resolution_delta_256_512':delta,'last_two_resolution_delta':last_delta,
            'escalation_rule':'any_view_256_512_absolute_IoU_disagreement>0.001',
            'occupied_voxels':int(volume.sum()),'volume_resolution':int(volume.shape[0])}
