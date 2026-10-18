"""Frozen Sadekar six-view volume-rendering protocol for soft-loss v3."""

from __future__ import annotations

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


class SadekarTFSProtocol:
    """Exact fixed cameras, red-channel rendering, and binary mask rule."""

    def __init__(
        self,
        device: torch.device,
        image_size: int = 256,
        zdist: float = 1.7,
        volume_extent_world: float = 1.7,
        n_pts_per_ray: int = 150,
    ) -> None:
        self.device = device
        self.image_size = int(image_size)
        self.volume_extent_world = float(volume_extent_world)
        elevation = torch.tensor(CAMERA_ELEVATION_DEG, device=device)
        azimuth = torch.tensor(CAMERA_AZIMUTH_DEG, device=device)
        rotation, translation = look_at_view_transform(
            dist=float(zdist), elev=elevation, azim=azimuth, device=device
        )
        self.cameras = FoVOrthographicCameras(
            device=device, R=rotation, T=translation
        )
        raysampler = NDCGridRaysampler(
            image_width=self.image_size,
            image_height=self.image_size,
            n_pts_per_ray=int(n_pts_per_ray),
            min_depth=0.1,
            max_depth=self.volume_extent_world * 1.5,
        )
        self.renderer = VolumeRenderer(
            raysampler=raysampler,
            raymarcher=EmissionAbsorptionRaymarcher(),
        )

    def render(self, densities: torch.Tensor, view_index: int) -> torch.Tensor:
        """Render one red-channel silhouette from ``[1,1,D,H,W]`` density."""

        if densities.ndim != 5 or densities.shape[:2] != (1, 1):
            raise ValueError("densities must have shape [1,1,D,H,W]")
        if not 0 <= int(view_index) < 6:
            raise IndexError("view_index must be in [0, 5]")
        resolution = int(densities.shape[-1])
        if densities.shape[-3:] != (resolution, resolution, resolution):
            raise ValueError("densities must be cubic")
        colors = torch.zeros(
            (1, 3, resolution, resolution, resolution),
            device=densities.device,
            dtype=densities.dtype,
        )
        colors[:, 0] = 1.0
        volumes = Volumes(
            densities=densities,
            features=torch.sigmoid(colors),
            voxel_size=self.volume_extent_world / resolution,
        )
        camera = FoVOrthographicCameras(
            device=self.device,
            R=self.cameras.R[view_index].unsqueeze(0),
            T=self.cameras.T[view_index].unsqueeze(0),
            znear=self.cameras.znear[view_index].unsqueeze(0),
            zfar=self.cameras.zfar[view_index].unsqueeze(0),
        )
        rendered = self.renderer(cameras=camera, volumes=volumes)[0]
        return rendered[0, :, :, 0]

    def render_all(self, densities: torch.Tensor) -> torch.Tensor:
        """Render all six registered views in target order."""

        return torch.stack([self.render(densities, index) for index in range(6)])

    @staticmethod
    def binary_mask(prediction: torch.Tensor) -> torch.Tensor:
        """Apply the frozen per-view ``0.8 * max`` silhouette rule."""

        if prediction.ndim == 2:
            maximum = prediction.detach().amax().clamp_min(1e-8)
        elif prediction.ndim == 3:
            maximum = prediction.detach().amax(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1e-8)
        else:
            raise ValueError("prediction must have shape [H,W] or [views,H,W]")
        return prediction >= maximum * 0.8
