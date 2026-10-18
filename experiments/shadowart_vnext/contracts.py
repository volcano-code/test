"""Immutable orthographic cameras and once-normalized silhouette geometry."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


VERSION = 'shadowart_physical_ortho_vnext_1'


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def mask_hash(mask) -> str:
    a = np.ascontiguousarray(mask, dtype=np.uint8)
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


@dataclass(frozen=True)
class CameraSpec:
    name: str
    target_index: int
    origin: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    direction: tuple[float, float, float]
    scale: float = 2.
    derived_from: int | None = None
    reflect_x: bool = False
    azimuth: float = 0.
    elevation: float = 0.

    def __post_init__(self):
        for key in ('origin', 'right', 'up', 'direction'):
            v = np.asarray(getattr(self, key), dtype=float)
            if v.shape != (3,) or not np.isfinite(v).all():
                raise ValueError(f'invalid camera {key}')
            object.__setattr__(self, key, tuple(float(x) for x in v))
        frame = np.array([self.right, self.up, self.direction])
        if not np.allclose(frame @ frame.T, np.eye(3), atol=1e-8, rtol=0):
            raise ValueError('camera frame must be orthonormal')
        if not np.allclose(np.cross(self.right, self.up), -np.array(self.direction), atol=1e-8, rtol=0):
            raise ValueError('camera frame handedness mismatch')
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('camera scale must be positive')
        if not isinstance(self.target_index, int) or self.target_index < 0:
            raise ValueError('invalid target index')
        if not all(np.isfinite([self.azimuth, self.elevation])):
            raise ValueError('invalid camera angles')


def angle_camera(index: int, azimuth: float, elevation: float, distance=1.7):
    a, e = np.deg2rad([azimuth, elevation])
    normal = np.array([np.sin(a)*np.cos(e), np.sin(e), np.cos(a)*np.cos(e)])
    right = np.array([np.cos(a), 0., -np.sin(a)])
    up = np.cross(normal, right)
    return CameraSpec(f'view_{index}', index, tuple(normal*distance), tuple(right),
                      tuple(up), tuple(-normal), azimuth=azimuth, elevation=elevation)


def identity_layout(n):
    return tuple((1., 0., 0.) for _ in range(n))


def validate_layout(layout):
    a = np.asarray(layout, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a) or not np.isfinite(a).all():
        raise ValueError('layout must be N x (scale, dx_256, dy_256)')
    if np.any(a[:,0] < .9-1e-10) or np.any(a[:,0] > 1.1+1e-10):
        raise ValueError('layout scales outside .9..1.1')
    if abs(float(np.log(a[:,0]).mean())) > 1e-8:
        raise ValueError('layout scale geometric mean must equal one')
    if np.any(abs(a[:,1:]) > 10.+1e-10):
        raise ValueError('layout translation outside +/-10px at256')
    return tuple(tuple(float(x) for x in row) for row in a)


@dataclass(frozen=True)
class CameraContract:
    cameras: tuple[CameraSpec, ...]
    base_count: int
    protocol: str
    extent: float = 1.7
    image_size: int = 256
    volume_resolution: int = 256
    input_hashes: tuple[str, ...] = ()
    target_hashes: tuple[str, ...] = ()
    layout: tuple[tuple[float, float, float], ...] = ()
    optimization: str = 'fixed_joint'
    version: str = VERSION
    coordinates: str = 'XYZ_right_handed_Y_up_pixel_centers_full_ray'

    def __post_init__(self):
        object.__setattr__(self, 'cameras', tuple(self.cameras))
        for name in ('input_hashes', 'target_hashes'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.version != VERSION or self.optimization != 'fixed_joint':
            raise ValueError('unsupported version/optimization')
        if self.base_count < 1 or not self.cameras:
            raise ValueError('positive number of cameras required')
        if self.image_size < 2 or self.volume_resolution < 2 or self.extent <= 0 or not np.isfinite(self.extent):
            raise ValueError('invalid resolution/extent')
        if len({c.name for c in self.cameras}) != len(self.cameras):
            raise ValueError('camera names must be unique')
        for i, c in enumerate(self.cameras):
            if c.target_index >= self.base_count:
                raise ValueError('target index out of range')
            if np.dot(c.origin,c.direction) > -.5*self.extent*np.abs(c.direction).sum()+1e-10:
                raise ValueError('camera must start before the entire cube along its ray direction')
            if c.derived_from is not None:
                if not 0 <= c.derived_from < i:
                    raise ValueError('derived camera must reference an earlier base')
                b = self.cameras[c.derived_from]
                if b.derived_from is not None or b.target_index != c.target_index or not c.reflect_x:
                    raise ValueError('invalid derived target mapping')
                if not (np.allclose(c.right, -np.array(b.right),atol=1e-8,rtol=0) and
                        np.allclose(c.up, b.up,atol=1e-8,rtol=0) and
                        np.allclose(c.direction, -np.array(b.direction),atol=1e-8,rtol=0) and
                        np.isclose(c.scale, b.scale,atol=1e-8,rtol=0)):
                    raise ValueError('opposite target/camera mismatch')
                delta = np.array(c.origin)-b.origin
                if abs(delta @ b.right) > 1e-8 or abs(delta @ b.up) > 1e-8:
                    raise ValueError('opposite image planes do not align')
            elif c.reflect_x:
                raise ValueError('base camera cannot request derived reflection')
        if set(c.target_index for c in self.cameras) != set(range(self.base_count)):
            raise ValueError('every input must have a camera')
        if self.input_hashes and len(self.input_hashes) != self.base_count:
            raise ValueError('input hashes/count mismatch')
        if self.target_hashes and len(self.target_hashes) != len(self.cameras):
            raise ValueError('target hashes/count mismatch')
        object.__setattr__(self, 'layout', validate_layout(self.layout or identity_layout(self.base_count)))
        if len(self.layout) != self.base_count:
            raise ValueError('layout count mismatch')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data['cameras'] = tuple(CameraSpec(**v) for v in data['cameras'])
        return cls(**data)

    @property
    def digest(self):
        return digest_json(self.to_dict())


def build_contract(n, *, tsf=None, mirror=None, camera_config=None,
                   image_size=256, resolution=256, extent=1.7):
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError('N must be a positive integer')
    if camera_config is not None:
        if tsf is not None or mirror is not None:
            raise ValueError('explicit cameras and tsf/mirror presets are mutually exclusive')
        data = json.loads(Path(camera_config).read_text()) if isinstance(camera_config, (str,Path)) else camera_config
        if data.get('projection', 'orthographic') != 'orthographic':
            raise ValueError('only orthographic projection is supported')
        cameras = tuple(CameraSpec(**c) for c in data['cameras'])
        protocol = 'explicit_orthographic'
    else:
        tsf = 0 if tsf is None else tsf
        mirror = 'auto' if mirror is None else str(mirror)
        if tsf not in (0,1) or mirror not in ('auto','0','1'):
            raise ValueError('tsf must be0/1 and mirror auto/0/1')
        signed = n >= 3 if mirror == 'auto' else mirror == '1'
        if n == 1:
            angles, protocol = [(0.,0.)], 'single_view_underconstrained'
        elif n == 2:
            angles = [(0.,0.),(90.,0.)]
            protocol = 'two_wall_tsf_alias' if tsf else 'two_wall'
            if signed:
                protocol += '_experimental_signed'
        elif tsf:
            angles = [(180.*i/(n-1),0.) for i in range(n-1)] + [(0.,90.)]
            protocol = f'fixed{n}_equatorial_polar'
        else:
            angles = [(180.*i/n,0.) for i in range(n)]
            protocol = f'fixed{n}_ring'
        cameras = tuple(angle_camera(i,a,e) for i,(a,e) in enumerate(angles))
        if signed:
            cameras += tuple(CameraSpec(f'opposite_{i}', c.target_index,
                                        tuple(-np.array(c.origin)), tuple(-np.array(c.right)),
                                        c.up, tuple(-np.array(c.direction)), c.scale, i, True,
                                        (c.azimuth+180.)%360., -c.elevation)
                             for i,c in enumerate(cameras))
    return CameraContract(cameras, n, protocol, extent=extent, image_size=image_size,
                          volume_resolution=resolution)


@dataclass(frozen=True)
class TargetSet:
    """Original binary pixel-cell geometry, cropped once, never repeatedly resized."""
    masks: tuple[np.ndarray, ...]
    input_hashes: tuple[str, ...]
    paths: tuple[str, ...]

    @classmethod
    def from_paths(cls, paths: Sequence):
        masks, hashes, names = [], [], []
        for path in paths:
            p = Path(path).resolve()
            raw=p.read_bytes()
            with Image.open(io.BytesIO(raw)) as im:
                rgba = np.asarray(im.convert('RGBA'))
                if np.any(rgba[:,:,3] < 255):
                    fg = rgba[:,:,3] >= 128
                else:
                    grey = np.asarray(im.convert('L'))
                    border = np.r_[grey[0],grey[-1],grey[:,0],grey[:,-1]]
                    fg = grey >= 128 if np.median(border) < 128 else grey < 128
            if not fg.any() or fg.all():
                raise ValueError(f'empty/constant silhouette: {p}')
            y,x = np.where(fg)
            crop = np.ascontiguousarray(fg[y.min():y.max()+1,x.min():x.max()+1])
            crop.setflags(write=False)
            masks.append(crop)
            hashes.append(hashlib.sha256(raw).hexdigest())
            names.append(str(p))
        if not masks:
            raise ValueError('no silhouettes')
        return cls(tuple(masks), tuple(hashes), tuple(names))

    def render(self, cameras, image_size, layout=None):
        layout = validate_layout(layout or identity_layout(len(self.masks)))
        if len(layout) != len(self.masks) or image_size < 2:
            raise ValueError('invalid target rendering dimensions')
        bases = []
        centers = np.arange(image_size)+.5
        for source,(scale,dx,dy) in zip(self.masks, layout):
            h,w = source.shape
            k = .70*image_size/max(h,w)*scale
            cx,cy = image_size/2+dx*image_size/256, image_size/2+dy*image_size/256
            if min(cx-k*w/2, cy-k*h/2) < 0 or max(cx+k*w/2,cy+k*h/2) > image_size:
                raise ValueError('layout clips source silhouette')
            x = np.floor((centers-cx)/k+w/2).astype(int)
            y = np.floor((centers-cy)/k+h/2).astype(int)
            valid = (y[:,None]>=0)&(y[:,None]<h)&(x[None,:]>=0)&(x[None,:]<w)
            out = source[np.clip(y,0,h-1)[:,None],np.clip(x,0,w-1)[None,:]] & valid
            if not out.any():
                raise ValueError('target vanishes at evaluation resolution')
            bases.append(out)
        views = []
        for c in cameras:
            target = bases[c.target_index]
            views.append(target[:,::-1] if c.reflect_x else target)
        return np.ascontiguousarray(views, dtype=bool)

    def diagnostics(self, cameras, image_size, layout):
        rendered = self.render(cameras, image_size, layout)
        result = []
        for i,source in enumerate(self.masks):
            idx = next(j for j,c in enumerate(cameras) if c.target_index == i and not c.reflect_x)
            y,x = np.where(rendered[idx])
            w,h = x.max()-x.min()+1, y.max()-y.min()+1
            result.append({'input_path':self.paths[i], 'input_sha256':self.input_hashes[i],
                           'source_bbox_wh':[source.shape[1],source.shape[0]],
                           'target_bbox_wh':[int(w),int(h)], 'foreground_pixels':int(rendered[idx].sum()),
                           'layout':list(layout[i]), 'source_aspect':source.shape[1]/source.shape[0],
                           'raster_aspect':float(w/h), 'transform':'uniform_scale_translation_only'})
        return result
