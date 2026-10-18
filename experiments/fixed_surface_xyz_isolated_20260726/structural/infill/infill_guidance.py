"""
InfillGuidanceGenerator — 调用SGLDBench生成应力对齐填充先验

为RepairCNN的infill通道提供初始化引导:
1. PSL追踪 → 管状填充骨架
2. 应力梯度填充
3. 组合: 高应力→管状, 低应力→梯度
"""
import sys
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# 添加SGLDBench路径
SGLDBENCH_DIR = Path(__file__).parent.parent.parent / "SGLDBench" / "python" / "src"
if str(SGLDBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(SGLDBENCH_DIR))


class InfillGuidanceGenerator:
    """使用SGLDBench工具生成填充先验"""

    def __init__(
        self,
        target_volume_fraction: float = 0.3,
        tube_radius: float = 1.5,
        stress_threshold_ratio: float = 0.5,
    ):
        self.target_vf = target_volume_fraction
        self.tube_radius = tube_radius
        self.stress_threshold_ratio = stress_threshold_ratio

    def generate(
        self,
        density: np.ndarray,
        von_mises: np.ndarray,
        principal_directions: np.ndarray | None = None,
    ) -> np.ndarray:
        """生成填充引导场

        Args:
            density: [N, N, N] 体素密度
            von_mises: [N, N, N] von Mises应力
            principal_directions: [N_elements, 3] 主应力方向 (可选)
        Returns:
            [N, N, N] 填充引导场 (0~1)
        """
        try:
            from sgldbench.stress_aligned_infill import (
                generate_stress_aligned_tube_infill,
                generate_stress_graded_infill,
                combine_infill_methods,
                compute_boundary_shell,
            )
            return self._generate_with_sgldbench(
                density, von_mises, principal_directions
            )
        except ImportError:
            logger.warning("SGLDBench not available, falling back to simple infill generation")
            return self._generate_simple(density, von_mises, principal_directions)

    def _generate_with_sgldbench(
        self,
        density: np.ndarray,
        von_mises: np.ndarray,
        principal_directions: np.ndarray | None,
    ) -> np.ndarray:
        """使用SGLDBench完整管线"""
        from sgldbench.stress_aligned_infill import (
            generate_stress_aligned_tube_infill,
            generate_stress_graded_infill,
            combine_infill_methods,
            compute_boundary_shell,
            InfillConfig,
        )
        from scipy.ndimage import binary_erosion

        solid = density > 0.5
        N = density.shape[0]

        # 外壳 (参数名: num_layers)
        shell = compute_boundary_shell(solid, num_layers=2)

        # 内部区域
        interior = binary_erosion(solid, iterations=2)

        if not interior.any():
            return np.zeros_like(density)

        # 构建InfillConfig
        config = InfillConfig(
            target_volume_fraction=self.target_vf,
            tube_radius=self.tube_radius,
        )

        # 方案1: 应力对齐管状填充 (需要3个主应力方向场 [N,N,N,3])
        try:
            if principal_directions is not None and len(principal_directions) > 0:
                # 将 principal_directions [N_elements, 3] 映射到体素网格
                dir_field = np.zeros((N, N, N, 3), dtype=np.float32)
                solid_indices = np.argwhere(solid)
                n_dirs = min(len(principal_directions), len(solid_indices))
                for i in range(n_dirs):
                    x, y, z = solid_indices[i]
                    dir_field[x, y, z] = principal_directions[i]
                # 使用主方向作为major, 构造正交medium/minor
                major_dirs = dir_field
                # 简化: 用随机正交向量作为medium/minor
                medium_dirs = np.zeros_like(dir_field)
                minor_dirs = np.zeros_like(dir_field)
                for idx in zip(*np.where(solid)):
                    d = major_dirs[idx]
                    if np.linalg.norm(d) < 1e-8:
                        continue
                    d = d / np.linalg.norm(d)
                    # Gram-Schmidt: 找正交向量
                    t = np.array([1, 0, 0], dtype=np.float32)
                    if abs(np.dot(d, t)) > 0.9:
                        t = np.array([0, 1, 0], dtype=np.float32)
                    m = t - np.dot(t, d) * d
                    m = m / (np.linalg.norm(m) + 1e-8)
                    n = np.cross(d, m)
                    medium_dirs[idx] = m
                    minor_dirs[idx] = n

                tube_infill = generate_stress_aligned_tube_infill(
                    solid, major_dirs, medium_dirs, minor_dirs, config=config,
                )
            else:
                logger.info("No principal directions provided, skipping tube infill")
                tube_infill = np.zeros_like(density, dtype=bool)
        except Exception as e:
            logger.warning(f"Tube infill generation failed: {e}, using graded only")
            tube_infill = np.zeros_like(density, dtype=bool)

        # 方案2: 应力梯度填充
        graded_infill = generate_stress_graded_infill(
            solid, von_mises, config=config,
        )

        # 组合 (参数顺序: volume, tube_infill, graded_infill, von_mises)
        combined = combine_infill_methods(
            solid, tube_infill, graded_infill, von_mises,
            stress_threshold_fraction=self.stress_threshold_ratio,
        )

        # 只保留内部区域, 转为float
        combined = combined.astype(np.float32) * interior.astype(np.float32)

        return combined

    def _generate_simple(
        self,
        density: np.ndarray,
        von_mises: np.ndarray,
        principal_directions: np.ndarray | None = None,
    ) -> np.ndarray:
        """简化版填充: 应力梯度 + 主应力方向各向异性

        高应力区域 → 高密度填充
        沿主应力方向 → 拉伸填充（各向异性膨胀）
        """
        from scipy.ndimage import binary_erosion, gaussian_filter

        solid = density > 0.5
        interior = binary_erosion(solid, iterations=2)

        if not interior.any():
            return np.zeros_like(density)

        N = density.shape[0]

        # 归一化应力
        vm_interior = von_mises * interior
        vm_max = vm_interior.max()
        if vm_max < 1e-12:
            return np.zeros_like(density)

        vm_norm = vm_interior / vm_max

        # 基础填充: 应力驱动密度
        infill = vm_norm * interior.astype(np.float32)

        # 主应力方向增强: 沿主应力方向做高斯拉伸
        if principal_directions is not None and len(principal_directions) > 0:
            # principal_directions: [N_elements, 3] 主应力方向
            # 将方向场映射回体素网格
            dir_field = np.zeros((3, N, N, N), dtype=np.float32)
            solid_indices = np.argwhere(solid)

            n_dirs = min(len(principal_directions), len(solid_indices))
            for i in range(n_dirs):
                x, y, z = solid_indices[i]
                dir_field[:, x, y, z] = principal_directions[i]

            # 沿主应力方向做各向异性扩散:
            # 对高应力区域, 沿主方向做方向性高斯平滑增强连接
            high_stress = (vm_norm > 0.5) & interior
            if high_stress.any():
                # 用方向场加权的各向异性膨胀
                for axis in range(3):
                    dir_component = np.abs(dir_field[axis])
                    # 在该轴方向上做加权膨胀
                    sigma = [0.5, 0.5, 0.5]
                    sigma[axis] = 2.0  # 沿主应力方向更强的扩散
                    dir_weight = gaussian_filter(
                        (high_stress * dir_component).astype(np.float32),
                        sigma=sigma
                    )
                    infill = np.maximum(infill, dir_weight * interior.astype(np.float32))

        # 调整到目标体积分数
        target_vol = self.target_vf * interior.sum()
        current_vol = infill.sum()

        if current_vol > 0:
            scale = target_vol / current_vol
            infill = np.clip(infill * scale, 0, 1)

        return infill.astype(np.float32)
