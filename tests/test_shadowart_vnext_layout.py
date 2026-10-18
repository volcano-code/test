import numpy as np
import pytest


def test_layout_generator_is_bounded_reproducible_and_not_global_shrink():
    from experiments.shadowart_vnext.layout import candidate_layouts
    a = list(candidate_layouts(6, 128))
    b = list(candidate_layouts(6, 128))
    np.testing.assert_array_equal(a,b)
    assert len(a) == 128
    np.testing.assert_array_equal(a[0], np.tile([1.,0.,0.], (6,1)))
    for x in a:
        x = np.array(x)
        assert .9-1e-9 <= x[:,0].min() <= x[:,0].max() <= 1.1+1e-9
        assert abs(np.log(x[:,0]).mean()) < 1e-8
        assert abs(x[:,1:]).max() <= 10


def test_failed_search_is_unknown_not_certified_impossible():
    from experiments.shadowart_vnext.layout import feasibility_status, iou_report
    target = np.ones((2,8,8), bool)
    report = iou_report(np.zeros_like(target), target)
    assert report['mean_iou'] == 0
    assert feasibility_status(report) == 'UNKNOWN'
    assert feasibility_status({'mean_iou':.98,'worst_iou':.97}) == 'UNKNOWN'
    assert feasibility_status({'mean_iou':.981,'worst_iou':.97}) == 'SHADOW_FEASIBLE'


def test_projection_metrics_reject_nan_or_shape_mismatch():
    from experiments.shadowart_vnext.layout import iou_report
    with pytest.raises(ValueError):
        iou_report(np.ones((1,8,8))*np.nan, np.ones((1,8,8),bool))
    with pytest.raises(ValueError):
        iou_report(np.ones((2,8,8)),np.ones((1,8,8)))


def test_target_outside_projected_volume_is_not_silently_clipped():
    from experiments.shadowart_vnext.contracts import build_contract
    from experiments.shadowart_vnext.layout import inside_projected_volume
    cams = build_contract(1).cameras
    targets=np.ones((1,16,16),bool)
    assert not inside_projected_volume(targets,cams,extent=.5)
    targets[:]=False
    targets[:,7:9,7:9]=True
    assert inside_projected_volume(targets,cams,extent=.5)


def test_conditioning_hull_does_not_keep_material_between_nominal_pixel_rays():
    from experiments.shadowart_vnext.contracts import build_contract
    from experiments.shadowart_vnext.layout import conditioning_hull
    class EmptyTargets:
        def render(self,cameras,size,layout):
            return np.zeros((len(cameras),size,size),bool)
    c=build_contract(1,image_size=16,resolution=16)
    hull=conditioning_hull(EmptyTargets(),c,c.layout)
    assert not hull.any(), 'background beams must not leave invisible alternating material'
