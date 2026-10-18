"""CPU optimizer and export regressions using the actual ray renderer."""
import importlib
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def api():
    return importlib.import_module("experiments.shadowart_vnext.training")


def cameras(n=1):
    return tuple(SimpleNamespace(name=f"z{i}", target_index=i, origin=(0.,0.,2.),
        right=(1.,0.,0.), up=(0.,1.,0.), direction=(0.,0.,-1.), scale=1.7,
        derived_from=None, reflect_x=False, azimuth=0., elevation=0.) for i in range(n))


def test_density_opacity_conversion_is_physical_and_resolution_independent():
    m = api()
    logits = torch.zeros(2,2,2, requires_grad=True)
    density, probability = m.logits_to_density(logits, 1.7)
    assert density[0,0,0].item() == pytest.approx(np.log(2) * 256 / 1.7)
    assert probability[0,0,0].item() == pytest.approx(.5)
    density.sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("kwargs", [{"steps":0}, {"resolution":0}, {"lr":float("nan")},
                                    {"extent":-1}, {"view_chunk_size":0}, {"seed":True}])
def test_invalid_parameters_fail_before_renderer_or_gpu_access(kwargs):
    with pytest.raises(ValueError):
        api().train_soft(np.zeros((1,4,4),dtype=bool), cameras(), image_size=4, **kwargs)


def test_nonbinary_or_mismatched_targets_rejected():
    for target in (np.full((1,4,4), .5), np.zeros((2,4,4), dtype=bool)):
        with pytest.raises(ValueError):
            api().train_soft(target, cameras(), image_size=4, resolution=4, steps=1)


def test_joint_chunking_matches_full_batch_and_exports_final_step_only():
    m = api()
    y = np.zeros((2,4,4), dtype=bool)
    y[:,1:3,1:3] = True
    events = []
    first = m.train_soft(y, cameras(2), resolution=4, image_size=4, steps=2, seed=17,
                         view_chunk_size=1, callback=events.append)
    second = m.train_soft(y, cameras(2), resolution=4, image_size=4, steps=2, seed=17,
                          view_chunk_size=2)
    np.testing.assert_allclose(first["density"], second["density"], rtol=2e-6, atol=2e-6)
    assert [row["step"] for row in events] == [1,2]
    assert first["metadata"]["export_step"] == 2
    assert first["metadata"]["checkpoint_selection"] == "final_step"
    assert first["occupancy"].dtype == np.bool_
    assert first["density"].dtype == np.float32
    np.testing.assert_array_equal(first["occupancy"], first["probabilities"] >= .5)
    assert np.isfinite(first["density"]).all()
    assert first["soft_projections"].shape == (2,4,4)
    assert first["hard_projections"].dtype == np.bool_
    assert first["hard_projections_512"].shape == (2,512,512)
    assert first["metadata"]["pixel_supersampling"] == 2
    assert first["metadata"]["hard_diagnostics_512"]["target_rule"] == "nearest_nominal_binary"


def test_training_does_not_change_callers_rng_or_reuse_input_buffers():
    m = api()
    torch.manual_seed(99)
    state = torch.get_rng_state().clone()
    y = np.zeros((1,4,4),dtype=bool)
    result = m.train_soft(y, cameras(), resolution=4, image_size=4, steps=1, seed=17)
    assert torch.equal(torch.get_rng_state(), state)
    assert not np.shares_memory(result["occupancy"], y)
    assert result["metadata"]["target_hashes"] == [hashlib.sha256(b"(4, 4)" + bytes(16)).hexdigest()]


def test_pixel_supersampling_observes_a_sheet_between_nominal_pixel_rays():
    m = api()
    density = torch.zeros(4,4,4)
    density[1,:,:] = 1.
    center = m.render_training_projection(density, cameras(), 1, extent=1.7, pixel_supersampling=1)
    area = m.render_training_projection(density, cameras(), 1, extent=1.7, pixel_supersampling=2)
    assert center.item() == 0.
    assert area.item() == pytest.approx(.5 * (1. - np.exp(-1.7)), abs=1e-6)


def test_hull_initialization_uses_supersampled_background_rays():
    result = api().train_soft(np.zeros((1,2,2), dtype=bool), cameras(),
                             resolution=4, image_size=2, steps=1, lr=1e-8)
    assert not result["occupancy"].any()
    assert result["metadata"]["initialization"]["hull_image_size"] == 4
    assert result["metadata"]["initialization"]["target_source"] == "nearest_nominal_binary"


def test_source_derived_hull_targets_are_validated_and_recorded():
    with pytest.raises(ValueError):
        api().train_soft(np.zeros((1,2,2), dtype=bool), cameras(), resolution=4,
                         image_size=2, steps=1, supersampled_targets=np.zeros((1,3,3)))


def test_opt_in_calibration_uses_real_joint_gradients_without_changing_the_update():
    m = api()
    y = np.zeros((2,2,2), dtype=bool)
    y[:,0,0] = True
    plain = m.train_soft(y, cameras(2), resolution=4, image_size=2, steps=1, seed=17)
    with_probe = m.train_soft(y, cameras(2), resolution=4, image_size=2, steps=1, seed=17,
                             calibration_steps=(1,))
    np.testing.assert_array_equal(plain["density"], with_probe["density"])
    probes = with_probe["metadata"]["calibration_records"]
    assert {row["auxiliary"] for row in probes} == {"lovasz", "ray_bce"}
    assert all(row["step"] == 1 and row["core_norm"] > 0 and row["aux_norm"] > 0 for row in probes)
    assert len({row["logits_sha256"] for row in probes}) == 1
    assert all(row["core_norm"] == pytest.approx(with_probe["history"][0]["gradient_norm"]) for row in probes)
    assert set(with_probe["metadata"]["source_hashes"]) == {"training.py", "losses.py", "rendering.py"}
    assert with_probe["metadata"]["environment"]["torch_version"] == torch.__version__
    assert with_probe["metadata"]["environment"]["device_type"] == "cpu"
    full_batch = m.train_soft(y, cameras(2), resolution=4, image_size=2, steps=1, seed=17,
                             calibration_steps=(1,), view_chunk_size=2)
    for chunked,batch in zip(probes,full_batch["metadata"]["calibration_records"]):
        assert chunked["aux_norm"] == pytest.approx(batch["aux_norm"],rel=2e-6)


def test_calibration_steps_are_explicit_unique_and_core_only():
    from experiments.shadowart_vnext.losses import LossConfig
    for extra in ({"calibration_steps":(0,)},{"calibration_steps":(1,1)},
                  {"calibration_steps":(2,)},
                  {"calibration_steps":(1,),"loss_config":LossConfig(auxiliary="lovasz",aux_weight=.1)}):
        with pytest.raises(ValueError):
            api().train_soft(np.zeros((1,2,2),dtype=bool), cameras(), image_size=2,
                             resolution=4, steps=1, **extra)


def test_logging_callback_cannot_mutate_saved_calibration_evidence():
    def mutate(event):
        event["calibration"][0]["core_norm"] = -99.
    y = np.zeros((1,2,2), dtype=bool)
    y[:,0,0] = True
    result = api().train_soft(y,cameras(),resolution=4,image_size=2,steps=1,
                             calibration_steps=(1,),callback=mutate)
    assert result["metadata"]["calibration_records"][0]["core_norm"] > 0
    assert result["history"][0]["calibration"][0]["core_norm"] > 0
