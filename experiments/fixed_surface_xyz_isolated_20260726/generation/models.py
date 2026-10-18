"""
VolumeModel — Marching Cubes 导出 + Taubin 平滑
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter
from pytorch3d.structures import Volumes

from voxel_ops import clean_binary_volume, save_voxel_mesh_as_obj

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class VolumeModel(nn.Module):
    def __init__(self, renderer, volume_size=128, voxel_size=0.1):
        super().__init__()
        self.log_densities = nn.Parameter(-8.0 * torch.ones(1, volume_size, volume_size, volume_size))

        colors_temp = torch.zeros(3, volume_size, volume_size, volume_size)
        colors_temp[:] = 10.0
        self.log_colors = nn.Parameter(colors_temp, requires_grad=False)

        self._voxel_size = voxel_size
        self._renderer = renderer
        self._volume_size = volume_size
        self.voxels = None
        self.colors = None
        self.temperature = 1.0

    def forward(self, cameras):
        batch_size = cameras.R.shape[0]

        densities = torch.sigmoid(self.temperature * self.log_densities).to(dtype=torch.float32).contiguous()
        colors = torch.sigmoid(self.log_colors).to(dtype=torch.float32).contiguous()

        volumes = Volumes(
            densities=densities[None].expand(batch_size, *self.log_densities.shape),
            features=colors[None].expand(batch_size, *self.log_colors.shape),
            voxel_size=self._voxel_size,
        )

        self.voxels = densities
        self.colors = colors

        return self._renderer(cameras=cameras, volumes=volumes)[0]

    def get_binary_voxels(self, threshold=0.5):
        """获取二值化体素（最大连通分量 + 温和形态学清理）"""
        with torch.no_grad():
            densities = torch.sigmoid(self.temperature * self.log_densities)
            binary_mask = (densities > threshold).float()

            np_mask = binary_mask[0].cpu().numpy()
            filtered_mask = clean_binary_volume(
                np_mask,
                fill_holes=True,
                closing_iters=1,
                opening_iters=0,
                keep_largest=True,
            ).astype(np.float32)
            binary_mask[0] = torch.from_numpy(filtered_mask).to(binary_mask.device)

            return binary_mask

    def _taubin_smooth(self, verts, faces, iterations=5, lamb=0.5, mu=-0.53):
        """向量化 Taubin 平滑（交替缩放/膨胀避免体积收缩）"""
        num_verts = verts.shape[0]
        if num_verts == 0 or faces.shape[0] == 0:
            return verts

        # 向量化构建边
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        src = torch.cat([i0, i1, i1, i2, i0, i2])
        dst = torch.cat([i1, i0, i2, i1, i2, i0])
        indices = torch.stack([src, dst], dim=0)  # [2, 6F]

        values = torch.ones(indices.shape[1], dtype=verts.dtype, device=verts.device)
        adj = torch.sparse_coo_tensor(indices, values, (num_verts, num_verts)).coalesce()

        degree = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1)  # [V]

        for it in range(iterations):
            factor = lamb if it % 2 == 0 else mu
            neighbor_sum = torch.sparse.mm(adj, verts)  # [V, 3]
            laplacian = neighbor_sum / degree.unsqueeze(1) - verts
            verts = verts + factor * laplacian

        return verts

    def to_mesh(self, threshold=0.5):
        """Marching Cubes + Taubin 平滑 → 平滑网格"""
        with torch.no_grad():
            binary = self.get_binary_voxels(threshold)
            vol = binary[0].cpu().numpy().astype(np.float64)

            # Gaussian 平滑二值场 → 生成光滑等值面
            vol_smooth = gaussian_filter(vol, sigma=1.0)

            if not HAS_SKIMAGE:
                print("WARNING: skimage not available, cannot export mesh")
                return torch.zeros(0, 3), torch.zeros(0, 3, dtype=torch.long)

            # level=0.5 是二值场平滑后的自然中点
            try:
                verts, faces, _, _ = marching_cubes(vol_smooth, level=0.5)
            except (ValueError, RuntimeError):
                print("WARNING: marching_cubes failed, no isosurface found")
                return torch.zeros(0, 3), torch.zeros(0, 3, dtype=torch.long)

            # 坐标映射到世界空间
            verts = verts / (self._volume_size - 1) * 1.7 - 0.85

            verts_t = torch.from_numpy(verts.copy()).float()
            faces_t = torch.from_numpy(faces.copy()).long()

            # Taubin 平滑
            if len(verts_t) > 0 and len(faces_t) > 0:
                verts_t = self._taubin_smooth(verts_t, faces_t, iterations=5)

            return verts_t, faces_t

    def save_obj(self, path, threshold=0.5, voxel_style=True):
        """保存 OBJ，默认导出真正的 voxel-style 面片网格。"""
        binary = self.get_binary_voxels(threshold)[0].cpu().numpy() > 0.5
        if voxel_style and save_voxel_mesh_as_obj(binary, path, extent_world=self._voxel_size * self._volume_size):
            print(f"Saved voxel mesh -> {path}")
            return

        verts, faces = self.to_mesh(threshold)
        if len(verts) == 0:
            print(f"WARNING: Empty mesh, skipping {path}")
            return

        with open(path, 'w') as f:
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        print(f"Saved marching-cubes mesh: {len(verts)} verts, {len(faces)} faces -> {path}")

    def save_voxels(self, path, threshold=0.5):
        with torch.no_grad():
            filtered = self.get_binary_voxels(threshold)
            np.save(path, filtered.cpu().numpy().astype(np.float32))
