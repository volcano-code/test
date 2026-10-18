"""Analytic regressions: missing complete rays/physical lengths must fail."""
from types import SimpleNamespace
import importlib
import numpy as np
import pytest
import torch
import trimesh


def renderer():
    spec = importlib.util.find_spec('experiments.shadowart_vnext.rendering')
    assert spec is not None, 'physical full-ray renderer has not been implemented'
    return importlib.import_module('experiments.shadowart_vnext.rendering')


def cam(direction=(0, 0, -1), origin=None, scale=2.):
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    right = np.cross(d, [0., 1., 0.])
    if np.linalg.norm(right) < .1:
        right = np.cross(d, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, d)
    return SimpleNamespace(origin=tuple(-d*3 if origin is None else origin),
                           direction=tuple(d), right=tuple(right), up=tuple(up), scale=scale)


@pytest.mark.parametrize('depth', [0, 3, 7])
def test_single_voxel_and_sheet_at_every_depth(depth):
    r = renderer(); volume = np.zeros((8, 8, 8), bool)
    volume[4, 4, depth] = True
    rendered = r.render_binary(volume, [cam(scale=2)], 8, extent=2)
    assert rendered.sum() == 1 and rendered[0, 3, 4]
    volume[:, :, depth] = True
    optical = r.render_optical(torch.tensor(volume, dtype=torch.float64), [cam()], 8, extent=2)
    np.testing.assert_allclose(optical.numpy(), 1-np.exp(-.25), atol=1e-12)


def test_empty_full_gradient_and_subdivision():
    r = renderer()
    density = torch.ones((8, 8, 8), dtype=torch.float64, requires_grad=True)
    out = r.render_optical(density, [cam()], 8, extent=2)
    np.testing.assert_allclose(out.detach(), 1-np.exp(-2), atol=1e-12)
    out.sum().backward()
    np.testing.assert_allclose(density.grad, .25*np.exp(-2), atol=1e-12)
    finer = torch.ones((16, 16, 16), dtype=torch.float64)
    torch.testing.assert_close(r.render_optical(finer, [cam()], 8, extent=2), out)
    zero = torch.zeros((8,8,8), requires_grad=True)
    empty = r.render_optical(zero, [cam()], 8, extent=2)
    assert not empty.any()
    empty.sum().backward(); assert torch.all(zero.grad > 0)


def test_oblique_analytic_lengths_and_far_camera():
    r = renderer(); d=(1, 1, 1)
    out = r.render_optical(torch.ones((8,)*3, dtype=torch.float64), [cam(d)], 1, extent=2)
    assert out.item() == pytest.approx(1-np.exp(-2*np.sqrt(3)), abs=1e-12)
    far = cam(d, origin=-np.ones(3)*100)
    assert r.render_optical(torch.ones((8,)*3,dtype=torch.float64), [far], 1, extent=2).item() == pytest.approx(out.item(), abs=1e-12)
    for xyz in [(0,0,0), (4,4,4), (7,7,7)]:
        v=np.zeros((8,)*3,bool); v[xyz]=1
        assert r.render_binary(v,[cam(d)],1,extent=2).item()
    v=np.zeros((8,)*3,bool); v[0,7,0]=1
    assert not r.render_binary(v,[cam(d)],1,extent=2).item()


def test_arbitrary_directions_match_independent_cell_aabb_reference():
    r=renderer(); n=5; size=7; extent=2.
    rng=np.random.default_rng(93); density=rng.random((n,)*3)
    for camera in [cam(),cam((0,0,1)),cam((1,.3,-.7)),cam((.1,-1,.4))]:
        expected=np.zeros((size,size))
        for y in range(size):
            for x in range(size):
                origin=np.array(camera.origin)+np.array(camera.right)*((x+.5)/size-.5)*camera.scale+np.array(camera.up)*(.5-(y+.5)/size)*camera.scale
                d=np.array(camera.direction); tau=0.
                for idx in np.ndindex(density.shape):
                    lo=-extent/2+np.array(idx)*extent/n; hi=lo+extent/n
                    near=0.; far=np.inf
                    for a in range(3):
                        if d[a] == 0:
                            if not lo[a] <= origin[a] < hi[a]: far=-1; break
                        else:
                            t=(np.array([lo[a],hi[a]])-origin[a])/d[a]
                            near=max(near,min(t)); far=min(far,max(t))
                    tau+=density[idx]*max(0.,far-near)
                expected[y,x]=-np.expm1(-tau)
        actual=r.render_optical(torch.tensor(density),[camera],size,extent)[0].numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_opposite_views_and_discrete_maximal_hull():
    r=renderer(); v=np.zeros((6,)*3,bool); v[1:3,2:5,1:5]=1
    cameras=[cam(),cam((0,0,1))]
    masks=r.render_binary(v,cameras,6,extent=2)
    assert np.array_equal(masks[0,:,::-1],masks[1])
    hull=r.visual_hull(masks,cameras,6,extent=2)
    assert np.array_equal(r.render_binary(hull,cameras,6,extent=2),masks)
    assert hull.sum()==36 and np.all(hull[v])
    for idx in np.argwhere(~hull):
        changed=hull.copy(); changed[tuple(idx)]=1
        assert np.any(r.render_binary(changed,cameras,6,extent=2)&~masks)


def test_independent_mesh_silhouette_cavity_and_triangle():
    r=renderer(); camera=cam()
    outer=trimesh.creation.box(extents=[1,1,1]); inner=trimesh.creation.box(extents=[.5,.5,.5]); inner.invert()
    shell=trimesh.util.concatenate([outer,inner])
    mask=r.render_mesh(shell,[camera],8)[0]
    expected=np.zeros((8,8),bool); expected[2:6,2:6]=1
    assert np.array_equal(mask,expected)
    triangle=trimesh.Trimesh(vertices=[[-1,-1,0],[1,-1,0],[-1,1,0]],faces=[[0,1,2]],process=False)
    tri=r.render_mesh(triangle,[camera],8)[0]
    assert tri[7,0] and not tri[0,7] and 28 <= tri.sum() <= 36
    assert not r.render_mesh(trimesh.Trimesh(vertices=[[0,0,0]],faces=[]),[camera],8).any()


def test_ray_map_lengths_and_inside_camera():
    r=renderer(); camera=cam(origin=(0,0,0))
    out=r.render_optical(torch.ones((4,)*3,dtype=torch.float64),[camera],1,extent=2)
    assert out.item()==pytest.approx(1-np.exp(-1))
    chunks=list(r.iter_ray_segments(cam((1,1,1)),1,8,extent=2,chunk_rays=1))
    assert sum(c.lengths.sum() for c in chunks)==pytest.approx(2*np.sqrt(3))
    assert sum(len(c.lengths) for c in chunks)==8


def test_oblique_gradient_subdivision_and_bounded_autograd_state():
    r=renderer(); camera=cam((1,.7,-.6))
    density=torch.full((16,)*3,.7,dtype=torch.float64,requires_grad=True)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(t.numel()) or t),lambda t:t):
        out=r.render_optical(density,[camera],16,extent=2)
    # Traversal belongs to a bounded cache/recomputation, not the whole graph.
    assert sum(saved)<16**3, 'backward retained the complete oblique intersection map'
    out.sum().backward()
    idx=(8,8,8); epsilon=1e-5
    plus=density.detach().clone(); minus=plus.clone()
    plus[idx]+=epsilon; minus[idx]-=epsilon
    finite=(r.render_optical(plus,[camera],16,2).sum()-r.render_optical(minus,[camera],16,2).sum())/(2*epsilon)
    assert density.grad[idx].item()==pytest.approx(finite.item(),rel=1e-7)
    fine=density.detach().repeat_interleave(2,0).repeat_interleave(2,1).repeat_interleave(2,2)
    torch.testing.assert_close(r.render_optical(fine,[camera],16,2),out)


def test_parallel_boundary_and_oblique_zero_contact_are_not_false_hits():
    r=renderer(); v=np.zeros((8,)*3,bool); v[4,4,:]=1
    assert r.render_binary(v,[cam()],1,2).item()
    v[:]=0; v[3,4,:]=1
    assert not r.render_binary(v,[cam()],1,2).item()
    v[:]=0; v[0,1,0]=1
    assert not r.render_binary(v,[cam((1,1,1))],1,2).item()


def test_mesh_oblique_matches_ray_triangle_and_roundtrip(tmp_path):
    r=renderer(); mesh=trimesh.creation.icosphere(subdivisions=1,radius=.73)
    path=tmp_path/'surface.stl'; mesh.export(path)
    loaded=trimesh.load(path,force='mesh'); camera=cam((1,.7,-.6)); size=19
    actual=r.render_mesh(loaded,[camera],size)[0]
    expected=np.zeros((size,size),bool)
    triangles=np.asarray(loaded.triangles); e1=triangles[:,1]-triangles[:,0]; e2=triangles[:,2]-triangles[:,0]
    d=np.asarray(camera.direction); h=np.cross(d,e2); determinant=np.einsum('ij,ij->i',e1,h)
    valid=np.abs(determinant)>1e-12; f=np.where(valid,1/np.where(valid,determinant,1),0)
    for y in range(size):
        for x in range(size):
            o=np.asarray(camera.origin)+np.asarray(camera.right)*((x+.5)/size-.5)*2+np.asarray(camera.up)*(.5-(y+.5)/size)*2
            s=o-triangles[:,0]; u=f*np.einsum('ij,ij->i',s,h); q=np.cross(s,e1)
            v=f*(q@d); t=f*np.einsum('ij,ij->i',e2,q)
            expected[y,x]=np.any(valid&(u>=0)&(v>=0)&(u+v<=1)&(t>=0))
    assert np.array_equal(actual,expected)


@pytest.mark.parametrize('shift',[(0,0,0),(.2,-.3,.1)])
def test_mesh_voxelization_canonical_cube_and_translation(shift):
    r=renderer(); assert hasattr(r,'voxelize_mesh'), 'mesh parity voxelizer missing'
    mesh=trimesh.creation.box(extents=[.8,1.2,.6]); mesh.apply_translation(shift)
    out=r.voxelize_mesh(mesh,20,extent=2)
    points=np.stack(np.meshgrid(*([np.linspace(-.95,.95,20)]*3),indexing='ij'),axis=-1)
    expected=np.all(np.abs(points-np.array(shift))<np.array([.4,.6,.3]),axis=-1)
    assert out.dtype==bool and np.array_equal(out,expected)


def test_mesh_voxelization_preserves_nested_cavities_and_separate_shells():
    r=renderer(); assert hasattr(r,'voxelize_mesh'), 'mesh parity voxelizer missing'
    boxes=[trimesh.creation.box(extents=[s]*3) for s in [1.6,1.2,.4]]
    boxes[1].invert()
    out=r.voxelize_mesh(trimesh.util.concatenate(boxes),20,2)
    expected=np.zeros((20,)*3,bool); expected[2:18,2:18,2:18]=1
    expected[4:16,4:16,4:16]=0; expected[8:12,8:12,8:12]=1
    assert np.array_equal(out,expected)
    parts=[]
    for x in [-.5,.5]:
        box=trimesh.creation.box(extents=[.4,.6,.8]); box.apply_translation([x,0,0]);parts.append(box)
    out=r.voxelize_mesh(trimesh.util.concatenate(parts),20,2)
    expected[:]=0;expected[3:7,7:13,6:14]=1;expected[13:17,7:13,6:14]=1
    assert np.array_equal(out,expected)


def test_mesh_voxelization_rejects_open_surface():
    r=renderer(); assert hasattr(r,'voxelize_mesh'), 'mesh parity voxelizer missing'
    triangle=trimesh.Trimesh(vertices=[[-1,-1,0],[1,-1,0],[-1,1,0]],faces=[[0,1,2]],process=False)
    with pytest.raises(ValueError,match='closed|watertight'):
        r.voxelize_mesh(triangle,8,2)


@pytest.mark.parametrize('azimuth,elevation',[(90.,0.),(0.,90.)])
def test_cardinal_grid_plane_ownership_is_identical_from_both_directions(azimuth,elevation):
    from dataclasses import replace
    from experiments.shadowart_vnext.contracts import angle_camera
    r=renderer(); base=angle_camera(0,azimuth,elevation)
    opposite=replace(base,name='opposite',origin=tuple(-np.array(base.origin)),
                     right=tuple(-np.array(base.right)),direction=tuple(-np.array(base.direction)))
    volume=np.random.default_rng(1).random((16,)*3)>.9
    projections=r.render_binary(volume,[base,opposite],32)
    assert np.array_equal(projections[0,:,::-1],projections[1])
    density=torch.tensor(volume,dtype=torch.float64)
    optical=r.render_optical(density,[base,opposite],32)
    torch.testing.assert_close(optical[0].flip(1),optical[1],rtol=0,atol=1e-12)
    # Independently reconstruct from complete segments, so the axis shortcut
    # cannot hide an inconsistent general path behind reflected cached outputs.
    full=[]
    for camera in (base,opposite):
        tau=np.zeros(32*32)
        for chunk in r.iter_ray_segments(camera,32,16):
            np.add.at(tau,chunk.start+chunk.ray_indices,
                      volume.reshape(-1)[chunk.voxel_indices]*chunk.lengths)
        full.append(-np.expm1(-tau.reshape(32,32)))
    np.testing.assert_allclose(full,optical.numpy(),atol=1e-12,rtol=0)
