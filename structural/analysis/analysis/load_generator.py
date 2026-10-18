"""
LoadGenerator — Stava 2012 风格的自动载荷生成

支持两种载荷工况:
1. Standing: 重力 + 底面支撑
2. Gripping: 对称夹持力 + 双面固定

输出: force_field [3, N, N, N], fixed_mask [1, N, N, N]
"""
import torch
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label


class LoadGenerator:
    """基于Stava 2012的自动载荷生成器"""

    def __init__(
        self,
        gravity: float = -9.81,
        material_density: float = 1.24e3,
        grip_force: float = 10.0,
        cell_size: float = 1.0,
    ):
        self.gravity = gravity
        self.material_density = material_density
        self.grip_force = grip_force
        self.cell_size = cell_size

    def standing_load(self, density: np.ndarray) -> dict:
        """重力载荷 + 底面固定

        Args:
            density: [N, N, N] 体素密度 (0~1)
        Returns:
            dict with 'force' [3, N, N, N], 'fixed_mask' [1, N, N, N]
        """
        N = density.shape[0]
        force = np.zeros((3, N, N, N), dtype=np.float32)
        fixed_mask = np.zeros((1, N, N, N), dtype=np.float32)

        # 重力: f_y = ρ * g * cell_volume (作用于每个实体体素)
        cell_vol = self.cell_size ** 3
        # Y轴（dim=1 in spatial）向下
        force[1] = density * self.material_density * self.gravity * cell_vol

        # 底面固定: 找到最低的实体层
        solid = density > 0.5
        if not solid.any():
            return {'force': force, 'fixed_mask': fixed_mask}

        # 沿Y轴(axis=1)找底面
        for y in range(N):
            if solid[:, y, :].any():
                # 底面往上扩展2层作为固定区域
                for dy in range(min(3, N - y)):
                    bottom_slice = solid[:, y + dy, :]
                    fixed_mask[0, :, y + dy, :] = bottom_slice.astype(np.float32)
                break

        return {'force': force, 'fixed_mask': fixed_mask}

    def gripping_load(self, density: np.ndarray, axis: int = 0) -> dict:
        """夹持载荷: 沿指定轴对称施加夹持力

        Args:
            density: [N, N, N]
            axis: 夹持轴 (0=x, 1=y, 2=z)
        Returns:
            dict with 'force' [3, N, N, N], 'fixed_mask' [1, N, N, N]
        """
        N = density.shape[0]
        force = np.zeros((3, N, N, N), dtype=np.float32)
        fixed_mask = np.zeros((1, N, N, N), dtype=np.float32)

        solid = density > 0.5
        if not solid.any():
            return {'force': force, 'fixed_mask': fixed_mask}

        # 重力仍然施加
        cell_vol = self.cell_size ** 3
        force[1] = density * self.material_density * self.gravity * cell_vol

        # 沿axis方向找两侧表面
        # 将夹持区域设置在质心附近的表面
        centroid = np.array([
            np.mean(np.where(solid)[i]) for i in range(3)
        ])

        # 两侧各找表面体素
        surface = solid ^ binary_erosion(solid)

        # 低侧夹持区
        grip_radius = max(2, N // 8)
        for side in [0, 1]:
            # side=0: 低端, side=1: 高端
            if side == 0:
                # 沿axis找最小非空层
                idx_range = range(N)
            else:
                idx_range = range(N - 1, -1, -1)

            for idx in idx_range:
                slicing = [slice(None)] * 3
                slicing[axis] = idx
                if solid[tuple(slicing)].any():
                    # 在这一层和附近2层标记固定
                    for d in range(min(3, N)):
                        actual_idx = idx + d if side == 0 else idx - d
                        if 0 <= actual_idx < N:
                            slicing2 = [slice(None)] * 3
                            slicing2[axis] = actual_idx
                            layer = solid[tuple(slicing2)]
                            fixed_mask[0][tuple(slicing2)] = layer.astype(np.float32)

                            # 在固定区域附近施加法向夹持力
                            sign = 1.0 if side == 0 else -1.0
                            n_surface = max(layer.sum(), 1)
                            force_per_voxel = sign * self.grip_force / n_surface
                            force[axis][tuple(slicing2)] += layer.astype(np.float32) * force_per_voxel
                    break

        return {'force': force, 'fixed_mask': fixed_mask}

    def generate_random_load(self, density: np.ndarray, rng: np.random.Generator | None = None) -> dict:
        """随机选择一种载荷工况

        50%概率 standing, 50%概率 gripping(随机轴)
        """
        if rng is None:
            rng = np.random.default_rng()

        if rng.random() < 0.5:
            return self.standing_load(density)
        else:
            axis = rng.integers(0, 3)
            return self.gripping_load(density, axis=int(axis))

    def to_torch(self, load_dict: dict, device: torch.device = None) -> dict:
        """将numpy载荷转为torch张量"""
        result = {}
        for k, v in load_dict.items():
            t = torch.from_numpy(v).float()
            if device is not None:
                t = t.to(device)
            result[k] = t
        return result
