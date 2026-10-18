import dataclasses
import json
import numpy as np
import pytest
from PIL import Image


def api():
    from experiments.shadowart_vnext import contracts
    return contracts


@pytest.mark.parametrize('n', [1, 2, 3, 4, 6, 8, 12])
@pytest.mark.parametrize('tsf', [0, 1])
def test_every_positive_view_count_routes_to_explicit_fixed_cameras(n, tsf):
    c = api().build_contract(n, tsf=tsf)
    assert c.base_count == n
    assert len(c.cameras) == (n if n < 3 else 2*n)
    assert c.optimization == 'fixed_joint'
    assert [v.target_index for v in c.cameras[:n]] == list(range(n))


def test_two_view_tsf_is_same_physical_camera_alias():
    a, b = api().build_contract(2, tsf=0), api().build_contract(2, tsf=1)
    assert a.cameras == b.cameras
    assert b.protocol == 'two_wall_tsf_alias'


def test_signed_cameras_reverse_ray_and_image_horizontal_axis_exactly():
    c = api().build_contract(3, tsf=1)
    for a, b in zip(c.cameras[:3], c.cameras[3:]):
        np.testing.assert_allclose(a.direction, -np.array(b.direction), atol=1e-12)
        np.testing.assert_allclose(a.right, -np.array(b.right), atol=1e-12)
        np.testing.assert_allclose(a.up, b.up, atol=1e-12)
        assert b.derived_from is not None


def test_custom_cameras_reject_presets_and_bad_frames():
    C = api()
    spec = dataclasses.asdict(C.build_contract(1).cameras[0])
    with pytest.raises(ValueError):
        C.build_contract(1, tsf=0, camera_config={'cameras':[spec]})
    spec['up'] = spec['right']
    with pytest.raises(ValueError):
        C.build_contract(1, camera_config={'cameras':[spec]})


def test_contract_roundtrip_is_immutable_and_fingerprint_detects_geometry_change():
    C = api()
    c = C.build_contract(4, tsf=1)
    assert C.CameraContract.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.extent = 99
    assert dataclasses.replace(c, extent=1.8).digest != c.digest


def test_targets_preserve_aspect_derive_mirrors_and_rebuild_from_original(tmp_path):
    C = api()
    raw = np.zeros((30,50), np.uint8)
    raw[5:25,5:45] = 255
    raw[5:10,30:45] = 0
    path = tmp_path/'mask.png'
    Image.fromarray(raw).save(path)
    c = C.build_contract(2, mirror='1', image_size=100)
    batch = C.TargetSet.from_paths([path,path])
    layout = C.identity_layout(2)
    masks = batch.render(c.cameras, 100, layout)
    np.testing.assert_array_equal(masks[2], masks[0][:,::-1])
    ys,xs = np.where(masks[0])
    assert (xs.max()-xs.min()+1)/(ys.max()-ys.min()+1) == pytest.approx(2., abs=.06)
    large = batch.render(c.cameras, 200, layout)
    assert large.shape == (4,200,200)
    assert batch.input_hashes[0] == batch.input_hashes[1]


def test_layout_cannot_shrink_all_targets_or_clip():
    C = api()
    with pytest.raises(ValueError):
        C.validate_layout(((.9,0.,0.),(.9,0.,0.)))
    with pytest.raises(ValueError):
        C.validate_layout(((1.,11.,0.),))
    with pytest.raises(ValueError):
        C.build_contract(0)
    with pytest.raises(ValueError):
        C.build_contract(2, mirror='banana')


def test_opposite_camera_cannot_start_behind_object_and_face_away():
    C=api()
    c=C.build_contract(1,mirror='1')
    wrong=dataclasses.replace(c.cameras[1],origin=c.cameras[0].origin)
    with pytest.raises(ValueError,match='before'):
        dataclasses.replace(c,cameras=(c.cameras[0],wrong))


def test_camera_tolerance_matches_strict_renderer():
    C=api()
    camera=C.build_contract(1).cameras[0]
    with pytest.raises(ValueError,match='orthonormal'):
        dataclasses.replace(camera,right=(1.000001,0.,0.))
def test_source_image_hash_and_pixels_use_identical_captured_bytes(tmp_path,monkeypatch):
    import hashlib
    import io
    import numpy as np
    from PIL import Image
    from experiments.shadowart_vnext.contracts import TargetSet
    raw=np.zeros((16,16),np.uint8);raw[4:12,6:10]=255
    path=tmp_path/'target.png';Image.fromarray(raw).save(path)
    original=path.read_bytes()
    replacement=io.BytesIO();Image.fromarray(np.rot90(raw)).save(replacement,format='PNG')
    image_open=Image.open
    def mutate_after_decode(source,*args,**kwargs):
        result=image_open(source,*args,**kwargs);result.load()
        path.write_bytes(replacement.getvalue())
        return result
    monkeypatch.setattr(Image,'open',mutate_after_decode)
    target=TargetSet.from_paths([path])
    assert target.masks[0].shape==(8,4)
    assert target.input_hashes==(hashlib.sha256(original).hexdigest(),)
