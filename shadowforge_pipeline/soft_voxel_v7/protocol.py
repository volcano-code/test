"""Corrected isolated six-view registered EAR renderer for soft-loss v7."""

from __future__ import annotations

import hashlib

import torch
from pytorch3d.renderer import (
    EmissionAbsorptionRaymarcher,
    FoVOrthographicCameras,
    NDCGridRaysampler,
    VolumeRenderer,
    look_at_view_transform,
)
from pytorch3d.structures import Volumes


CAMERA_AZIMUTH_DEG = (0.0, 90.0, 0.0, 180.0, 270.0, 0.0)
CAMERA_ELEVATION_DEG = (0.0, 0.0, 90.0, 0.0, 0.0, 270.0)
REGISTERED_VIEW_DESCRIPTORS = (
    ("z", "D4_3(Bz)", 0, 0.0, 0.0),
    ("x", "Bx", 1, 90.0, 0.0),
    ("y", "D4_7(By)", 2, 0.0, 90.0),
    ("mirror_z", "D4_7(Bz)", 3, 180.0, 0.0),
    ("mirror_x", "D4_4(Bx)", 4, 270.0, 0.0),
    ("mirror_y", "D4_3(By)", 5, 0.0, 270.0),
)
_DISTANCE = 1.7
_EXTENT = 1.7
_SAMPLES = 150
_SCALE = 20.0 / 17.0


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes(order="C")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_camera_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    elevation = torch.tensor(CAMERA_ELEVATION_DEG, dtype=torch.float32)
    azimuth = torch.tensor(CAMERA_AZIMUTH_DEG, dtype=torch.float32)
    rotation, translation = look_at_view_transform(
        dist=_DISTANCE,
        elev=elevation,
        azim=azimuth,
        device=torch.device("cpu"),
    )
    scale = torch.full((6, 3), _SCALE, dtype=torch.float32)
    cameras = FoVOrthographicCameras(
        device=torch.device("cpu"),
        R=rotation,
        T=translation,
        scale_xyz=scale,
    )
    projection = cameras.get_projection_transform().get_matrix().contiguous()
    return rotation.contiguous(), translation.contiguous(), cameras.scale_xyz.contiguous(), projection


def registered_camera_payload() -> dict[str, object]:
    """Return canonical metadata plus exact R/T/scale/projection bytes and hashes."""

    rotation, translation, scale, projection = _canonical_camera_tensors()
    rotation_bytes = _tensor_bytes(rotation)
    translation_bytes = _tensor_bytes(translation)
    scale_bytes = _tensor_bytes(scale)
    projection_bytes = _tensor_bytes(projection)
    metadata = {
        "schema": "pcraft_soft_voxel_v7_registered_ear_v1",
        "descriptors": [list(row) for row in REGISTERED_VIEW_DESCRIPTORS],
        "d4_stack": [3, 0, 7, 7, 4, 3],
        "xyz_to_dhw": "density[d,h,w]=Vxyz[N-1-w,N-1-h,N-1-d]",
        "mirror_policy": "registered-base-D4-only; no raw mirror inputs",
        "distance": _DISTANCE,
        "volume_extent_world": _EXTENT,
        "samples_per_ray": _SAMPLES,
        "ray_min_depth": 0.1,
        "ray_max_depth": 2.55,
        "readout": "red channel; P >= 0.8 * stopgrad(max(P)) per view",
        "rotation_dtype": "float32",
        "rotation_shape": [6, 3, 3],
        "translation_dtype": "float32",
        "translation_shape": [6, 3],
        "scale_dtype": "float32",
        "scale_shape": [6, 3],
        "projection_dtype": "float32",
        "projection_shape": [6, 4, 4],
    }
    return {
        "metadata": metadata,
        "rotation_bytes": rotation_bytes,
        "translation_bytes": translation_bytes,
        "scale_bytes": scale_bytes,
        "projection_bytes": projection_bytes,
        "rotation_sha256": _sha256(rotation_bytes),
        "translation_sha256": _sha256(translation_bytes),
        "scale_sha256": _sha256(scale_bytes),
        "projection_sha256": _sha256(projection_bytes),
    }


class RegisteredEARProtocol:
    """Frozen corrected cameras, explicit per-view scale, and red EAR readout."""

    def __init__(
        self,
        device: torch.device,
        *,
        _image_size: int = 256,
        _n_pts_per_ray: int = 150,
        _allow_test_size: bool = False,
    ) -> None:
        if type(_allow_test_size) is not bool:
            raise ValueError("_allow_test_size must be boolean")
        if type(_image_size) is not int or _image_size <= 0:
            raise ValueError("_image_size must be a positive integer")
        if type(_n_pts_per_ray) is not int or _n_pts_per_ray <= 0:
            raise ValueError("_n_pts_per_ray must be a positive integer")
        if not _allow_test_size and (_image_size != 256 or _n_pts_per_ray != 150):
            raise ValueError("production protocol is exactly 256 pixels and 150 samples/ray")
        try:
            resolved_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device must identify a torch device") from error
        self.device = resolved_device
        self.image_size = _image_size
        self.n_pts_per_ray = _n_pts_per_ray
        elevation = torch.tensor(CAMERA_ELEVATION_DEG, device=resolved_device, dtype=torch.float32)
        azimuth = torch.tensor(CAMERA_AZIMUTH_DEG, device=resolved_device, dtype=torch.float32)
        self.R, self.T = look_at_view_transform(
            dist=_DISTANCE,
            elev=elevation,
            azim=azimuth,
            device=resolved_device,
        )
        scale = torch.full((6, 3), _SCALE, device=resolved_device, dtype=torch.float32)
        self.cameras = FoVOrthographicCameras(
            device=resolved_device,
            R=self.R,
            T=self.T,
            scale_xyz=scale,
        )
        self.renderer = VolumeRenderer(
            raysampler=NDCGridRaysampler(
                image_width=_image_size,
                image_height=_image_size,
                n_pts_per_ray=_n_pts_per_ray,
                min_depth=0.1,
                max_depth=_EXTENT * 1.5,
            ),
            raymarcher=EmissionAbsorptionRaymarcher(),
        )

    def render(self, densities: torch.Tensor, view_index: int) -> torch.Tensor:
        """Render one camera-indexed red-channel projection."""

        if not isinstance(densities, torch.Tensor) or densities.dtype != torch.float32:
            raise ValueError("densities must be a float32 torch.Tensor")
        if densities.shape != (1, 1, self.image_size, self.image_size, self.image_size):
            raise ValueError("densities must have exact shape [1,1,N,N,N]")
        if densities.device != self.device:
            raise ValueError("densities must be on the protocol device")
        if not torch.isfinite(densities).all().item():
            raise ValueError("densities must be finite")
        if type(view_index) is not int or not 0 <= view_index < 6:
            raise ValueError("view_index must be an integer in [0,5]")
        colors = torch.zeros(
            (1, 3, self.image_size, self.image_size, self.image_size),
            device=self.device,
            dtype=torch.float32,
        )
        colors[:, 0] = 1.0
        volume = Volumes(
            densities=densities,
            features=torch.sigmoid(colors),
            voxel_size=_EXTENT / self.image_size,
        )
        camera = FoVOrthographicCameras(
            device=self.device,
            R=self.R[view_index : view_index + 1],
            T=self.T[view_index : view_index + 1],
            scale_xyz=((_SCALE, _SCALE, _SCALE),),
        )
        rendered = self.renderer(cameras=camera, volumes=volume)[0]
        projection = rendered[0, :, :, 0]
        if not torch.isfinite(projection).all().item():
            raise ValueError("rendered projection is non-finite")
        return projection

    def render_all(self, densities: torch.Tensor) -> torch.Tensor:
        """Render all six registered cameras in frozen delivery order."""

        return torch.stack([self.render(densities, index) for index in range(6)]).contiguous()

    @staticmethod
    def binary_mask(projection: torch.Tensor) -> torch.Tensor:
        """Apply the exact detached per-view 0.8-max delivery readout."""

        if not isinstance(projection, torch.Tensor) or not projection.is_floating_point():
            raise ValueError("projection must be a floating-point torch.Tensor")
        if projection.ndim not in (2, 3) or any(size == 0 for size in projection.shape):
            raise ValueError("projection must have shape [H,W] or [views,H,W]")
        if not torch.isfinite(projection).all().item():
            raise ValueError("projection must be finite")
        if projection.ndim == 2:
            maximum = projection.detach().amax().clamp_min(1e-8)
        else:
            maximum = projection.detach().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return projection >= projection.new_tensor(0.8) * maximum


__all__ = ["RegisteredEARProtocol", "registered_camera_payload"]
