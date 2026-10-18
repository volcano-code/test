"""Check a projection relaxation against literal cases and exhaustive activation."""
import importlib
import itertools
from pathlib import Path

import numpy as np
import pytest


def module():
    assert (Path(__file__).parents[1]/'experiments/shadowart_vnext/projection_ceiling.py').is_file(), 'projection ceiling module missing'
    return importlib.import_module('experiments.shadowart_vnext.projection_ceiling')


def test_best_group_mask_accounts_for_unreachable_foreground():
    selected, score = module().optimal_groups(np.array([4, 1, 0]), np.array([4, 4, 4]), 6)
    assert selected.tolist() == [True, False, False]
    assert score == pytest.approx(4/6)


def test_empty_foreground_selects_no_background():
    selected, score = module().optimal_groups(np.array([0, 0]), np.array([2, 3]), 0)
    assert not selected.any()
    assert score == 1.


def test_relaxation_matches_exhaustive_binary_choices():
    rng = np.random.default_rng(17)
    for _ in range(80):
        total = rng.integers(1, 9, 5)
        foreground = np.array([rng.integers(0, x+1) for x in total])
        positives = int(foreground.sum()) + int(rng.integers(0, 4))
        values=[]
        for flags in itertools.product([False, True], repeat=5):
            tp=int(foreground[list(flags)].sum()); fp=int((total-foreground)[list(flags)].sum())
            values.append(tp/(positives+fp) if positives+fp else 1.)
        _, score = module().optimal_groups(foreground, total, positives)
        assert score == pytest.approx(max(values))


def test_camera_grouping_is_an_upper_bound_not_a_feasible_shape_claim():
    from experiments.shadowart_vnext.contracts import angle_camera
    cam=angle_camera(0,0,0)
    target=np.zeros((1,8,8),bool); target[0,2:6,2:6]=True
    target[0,2,2]=False
    report=module().cardinal_ceiling(target,[cam],resolution=2,extent=2.)
    # Each 4x4 cell-column image has only 3 or 4 foreground pixels.
    # Best excludes the 3/16 group: TP=12, FP=36, P=15.
    assert report['mean_upper_bound'] == pytest.approx(12/51)
    assert report['feasible_object_certified'] is False
    assert report['cross_view_consistency_enforced'] is False


def test_oblique_camera_is_rejected_not_approximated_as_cardinal():
    from experiments.shadowart_vnext.contracts import angle_camera
    with pytest.raises(ValueError,match='cardinal'):
        module().cardinal_ceiling(np.zeros((1,8,8),bool),[angle_camera(0,30,0)],4,1.7)


@pytest.mark.parametrize('fg,total,p', [([2],[1],2),([-1],[2],1),([1],[2],0),([.5],[1],1)])
def test_invalid_group_counts_rejected(fg,total,p):
    with pytest.raises(ValueError):
        module().optimal_groups(np.array(fg),np.array(total),p)
