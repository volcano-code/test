"""Stage-metric definitions and source-derived resolution escalation."""
import importlib
import numpy as np
import pytest


def metrics():
    spec=importlib.util.find_spec('experiments.shadowart_vnext.metrics')
    assert spec is not None, 'stage metrics have not been implemented'
    return importlib.import_module('experiments.shadowart_vnext.metrics')


def test_continuous_soft_jaccard_is_not_threshold_projection_iou():
    alpha=np.array([[[.8,.4],[.2,0.]]])
    target=np.array([[[1,0],[1,0]]],bool)
    report=metrics().soft_projection_metrics(alpha,target)
    assert report['continuous_soft_jaccard']['mean']==pytest.approx(1/2.4)
    assert report['threshold_0_5_projection_iou']['mean_iou']==pytest.approx(.5)
    assert report['threshold_0_5_projection_iou']['false_negative_pixels']==[1]
    assert report['scope']=='stage_only_not_final_mesh_or_manufacturing_QA'
    assert 'passed' not in report and 'feasible' not in report


@pytest.mark.parametrize('value',[-.1,1.1,np.nan])
def test_invalid_physical_alpha_is_not_normalized_into_metrics(value):
    alpha=np.full((1,2,2),value)
    with pytest.raises(ValueError):
        metrics().soft_projection_metrics(alpha,np.zeros((1,2,2),bool))


def test_soft_density_replay_uses_recorded_pixel_supersampling():
    from experiments.shadowart_vnext.contracts import build_contract,TargetSet
    contract=build_contract(1,image_size=2,resolution=4,extent=2.)
    target_set=TargetSet((np.ones((1,1),bool),),('a'*64,),('analytic',))
    density=np.zeros((4,)*3,np.float32);density[0,:, :]=1.
    replay=metrics().soft_density_metrics(density,contract,target_set,{'pixel_supersampling':2})
    # Half of left output pixel's subrays cross a length2 slab; the right half is empty.
    alpha=np.array([[[(1-np.exp(-2))/2,0],[(1-np.exp(-2))/2,0]]])
    expected=metrics().soft_projection_metrics(alpha,target_set.render(contract.cameras,2,contract.layout))
    assert replay['continuous_soft_jaccard']['mean']==pytest.approx(expected['continuous_soft_jaccard']['mean'],abs=1e-7)
    assert replay['threshold_0_5_projection_iou']['mean_iou']==expected['threshold_0_5_projection_iou']['mean_iou']
    assert replay['pixel_supersampling']==2


def test_occupancy_metrics_use_source_targets_and_escalate_disagreement():
    from experiments.shadowart_vnext.contracts import build_contract,TargetSet
    contract=build_contract(1,image_size=32,resolution=16)
    targets=TargetSet((np.ones((3,3),bool),),('b'*64,),('analytic',))
    volume=np.zeros((16,)*3,bool);volume[3:13,3:13,:]=1
    report=metrics().occupancy_metrics(volume,contract,targets,stage='soft_hard_occupancy')
    assert report['stage']=='soft_hard_occupancy'
    assert [row['image_size'] for row in report['evaluations']]==[256,512,1024]
    assert report['resolution_delta_256_512']>.001
    assert report['target_rule']=='original_source_derived_at_each_resolution'
    assert 'passed' not in report and 'status' not in report


def test_occupancy_metrics_do_not_escalate_when_resolution_metrics_agree():
    from experiments.shadowart_vnext.contracts import build_contract,TargetSet
    contract=build_contract(1,image_size=32,resolution=16)
    targets=TargetSet((np.ones((3,3),bool),),('b'*64,),('analytic',))
    report=metrics().occupancy_metrics(np.zeros((16,)*3,bool),contract,targets)
    assert [row['image_size'] for row in report['evaluations']]==[256,512]
    assert report['resolution_delta_256_512']==0
