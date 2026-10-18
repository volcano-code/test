"""
渲染和相机工具函数
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    look_at_view_transform,
    VolumeRenderer,
    NDCGridRaysampler,
)


class SilhouetteRaymarcher(nn.Module):
    """
    Soft-Max 投影 Raymarcher — 替代 EmissionAbsorptionRaymarcher

    核心区别：
      EAR:  opacity = 1 - prod(1 - d_i)  → 散布低密度体素累积成高不透明度 → 飞边
      本类: opacity ≈ max(d_i)           → 只有最大密度体素决定像素值 → 无累积飞边

    实现：用 softmax(d * tau) 加权求和近似 max，保持可微性。
    tau 越大越接近 hard max，tau=1 退化为平均。
    """
    def __init__(self, tau=10.0):
        super().__init__()
        self.tau = tau

    def forward(self, rays_densities, rays_features, **kwargs):
        d = rays_densities[..., 0]  # [B, H, W, N]

        weights = torch.softmax(d * self.tau, dim=-1)  # [B, H, W, N]

        opacity = (weights * d).sum(dim=-1, keepdim=True)  # [B, H, W, 1]

        features = (weights.unsqueeze(-1) * rays_features).sum(dim=-2)  # [B, H, W, C]

        return torch.cat([features, opacity], dim=-1)


def get_props(img, img_size=(256, 256)):
    """图像预处理"""
    rows, cols = img_size
    h0, w0 = (int(rows * 0.7), int(cols * 0.7))
    cx0, cy0 = (img_size[0] // 2, img_size[1] // 2)

    conts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conts = sorted(conts, key=lambda ctr: cv2.contourArea(ctr), reverse=True)

    if len(conts) == 0:
        return img, 0

    cnt = conts[0]
    x, y, w, h = cv2.boundingRect(cnt)
    cx1 = x + w / 2
    cy1 = y + h / 2

    sx = w0 * 1.0 / max(w, 1)
    sy = h0 * 1.0 / max(h, 1)
    tx = cx0 * (1 - sx) + (cx0 - cx1)
    ty = cy0 * (1 - sy) + (cy0 - cy1)

    M = np.float32([[sx, 0, tx], [0, sy, ty]])
    img = cv2.warpAffine(img, M, img_size)

    return img, len(conts) - 1


def get_voxel_renderer(device, cameras, image_size, volume_extent_world=1.7, volume_size=128):
    """创建体素渲染器"""
    n_pts_per_ray = max(150, int(150 * (volume_size / 128.0)))

    raysampler = NDCGridRaysampler(
        image_width=image_size,
        image_height=image_size,
        n_pts_per_ray=n_pts_per_ray,
        min_depth=0.1,
        max_depth=volume_extent_world * 1.5,
    )
    raymarcher = SilhouetteRaymarcher(tau=10.0)
    return VolumeRenderer(raysampler=raysampler, raymarcher=raymarcher)


def create_cameras(num_views, device, zdist, camera_mode, mirror_mode):
    """创建多视角相机"""
    if mirror_mode:
        num_total = num_views * 2
        elev = torch.linspace(0, 0, num_total)
        azim = torch.linspace(0, 360 - 360 / num_total, num_total)
    else:
        elev = torch.linspace(0, 0, num_views)
        azim = torch.linspace(0, 180 - 180 / num_views, num_views)

    R, T = look_at_view_transform(dist=zdist, elev=elev, azim=azim, device=device)
    R = R.to(device=device, dtype=torch.float32).contiguous()
    T = T.to(device=device, dtype=torch.float32).contiguous()

    if camera_mode == "perp":
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    else:
        cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    return cameras, R, T


def get_tfs_view_specs(num_input_views, mirror_mode):
    """Return dynamic TSF view labels and angles based on input-count and mirror-mode."""
    num_input_views = int(num_input_views)
    if num_input_views <= 3:
        primary_specs = [
            ("front", 0.0, 0.0),
            ("right", 0.0, 90.0),
            ("top", 90.0, 0.0),
        ]
        opposite_specs = [
            ("back", 0.0, 180.0),
            ("left", 0.0, 270.0),
            ("down", 270.0, 0.0),
        ]
    else:
        # Follow the existing 4VTFS camera family when users pass 4 input views
        # through the TSF entrypoint: three ring views plus a top view, then their
        # mirrored / opposite counterparts.
        primary_specs = [
            ("v0", 0.0, 0.0),
            ("v1", 0.0, 60.0),
            ("v2", 0.0, 120.0),
            ("top", 90.0, 0.0),
        ]
        opposite_specs = [
            ("v3", 0.0, 180.0),
            ("v4", 0.0, 240.0),
            ("v5", 0.0, 300.0),
            ("down", 270.0, 0.0),
        ]
    primary_count = max(0, min(num_input_views, len(primary_specs)))
    specs = list(primary_specs[:primary_count])
    if mirror_mode:
        specs.extend(opposite_specs[:primary_count])
    return specs


def create_cameras_TFS_mode(device, zdist, mirror_mode, camera_mode, num_input_views=3):
    """TSF模式相机，按输入视图数和镜像模式动态生成。"""
    specs = get_tfs_view_specs(num_input_views, mirror_mode)
    if not specs:
        raise ValueError("TSF mode requires at least one input view")
    elev = torch.tensor([spec[1] for spec in specs], dtype=torch.float32)
    azim = torch.tensor([spec[2] for spec in specs], dtype=torch.float32)

    R, T = look_at_view_transform(dist=zdist, elev=elev, azim=azim, device=device)
    R = R.to(device=device, dtype=torch.float32).contiguous()
    T = T.to(device=device, dtype=torch.float32).contiguous()

    if camera_mode == "perp":
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    else:
        cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    return cameras, R, T


def create_cameras_4VTFS_mode(device, zdist, mirror_mode, camera_mode):
    """4视图TSF模式"""
    if mirror_mode:
        elev = torch.tensor([0.0, 0.0, 0.0, 90.0, 0.0, 0.0, 0.0, 270.0])
        azim = torch.tensor([0.0, 60.0, 120.0, 0.0, 180.0, 240.0, 300.0, 0.0])
    else:
        elev = torch.tensor([0.0, 0.0, 0.0, 90.0])
        azim = torch.tensor([0.0, 60.0, 120.0, 0.0])

    R, T = look_at_view_transform(dist=zdist, elev=elev, azim=azim, device=device)
    R = R.to(device=device, dtype=torch.float32).contiguous()
    T = T.to(device=device, dtype=torch.float32).contiguous()

    if camera_mode == "perp":
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    else:
        cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    return cameras, R, T
