"""
StructuralVolumeModel — 扩展ShadowArt的VolumeModel，集成结构分析

在原始VolumeModel基础上增加:
- 结构分析接口
- 修复网络集成
- 内外解耦支持
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 添加ShadowArt路径
SHADOWART_DIR = Path(__file__).parent.parent.parent / "mesh_combined2"
if str(SHADOWART_DIR) not in sys.path:
    sys.path.insert(0, str(SHADOWART_DIR))


class StructuralVolumeModel(nn.Module):
    """结构感知体素模型

    组合:
    - ShadowArt的log_densities (外形参数)
    - RepairCNN (结构修复)
    - PINO (快速应力分析)
    """

    def __init__(
        self,
        renderer,
        volume_size: int = 64,
        voxel_size: float = 0.1,
        repair_cnn=None,
        pino_model=None,
    ):
        super().__init__()

        # 体素密度参数 (ShadowArt核心)
        self.log_densities = nn.Parameter(
            -8.0 * torch.ones(1, volume_size, volume_size, volume_size)
        )
        colors_temp = torch.zeros(3, volume_size, volume_size, volume_size)
        colors_temp[:] = 10.0
        self.log_colors = nn.Parameter(colors_temp, requires_grad=False)

        self._voxel_size = voxel_size
        self._renderer = renderer
        self._volume_size = volume_size
        self.temperature = 1.0

        # 结构分析组件
        self.repair_cnn = repair_cnn
        self.pino = pino_model

    @property
    def density(self) -> torch.Tensor:
        """当前密度场 [1, N, N, N]"""
        return torch.sigmoid(self.temperature * self.log_densities)

    def get_shape_density(self) -> torch.Tensor:
        """获取外形密度 (不含修复)"""
        return self.density

    def get_repaired_density(
        self,
        force: torch.Tensor | None = None,
        fixed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """获取修复后密度

        Args:
            force: [1, 3, N, N, N] 力场
            fixed_mask: [1, 1, N, N, N] 固定掩码
        Returns:
            [1, 1, N, N, N] 修复后密度
        """
        rho = self.density.unsqueeze(0)  # [1, 1, N, N, N]

        if self.repair_cnn is None or self.pino is None:
            return rho

        if force is None:
            return rho

        # PINO应力分析
        from structural.models.load_model import forward_surrogate_physical
        with torch.no_grad():
            pino_fields = forward_surrogate_physical(self.pino, rho, force)

        # Von Mises (v3-aware)
        from structural.analysis.stress_eval import (
            get_yield_stress_for_model,
        )
        vm = pino_fields['von_mises']  # [1, 1, N, N, N]
        vm_norm = vm / (vm.max() + 1e-8)

        # 使用 VulnerabilityDetector 生成脆弱图
        from structural.analysis.vulnerability import VulnerabilityDetector
        vuln_detector = VulnerabilityDetector(
            yield_stress=get_yield_stress_for_model(self.pino)
        )
        vuln_result = vuln_detector.detect(
            rho[0, 0].detach().cpu().numpy(),
            vm[0, 0].detach().cpu().numpy(),
        )
        vuln = torch.from_numpy(
            vuln_result['vulnerability_map']
        ).float().unsqueeze(0).unsqueeze(0).to(rho.device)
        vuln = vuln / (vuln.max() + 1e-8)

        # 修复
        repair_input = torch.cat([rho, vm_norm, vuln], dim=1)  # [1, 3, N, N, N]
        repair_fields = self.repair_cnn(repair_input)

        # 动态内部掩码
        interior = self._compute_interior_mask(rho)

        return self.repair_cnn.apply_repair(rho, repair_fields, interior)

    def _compute_interior_mask(self, density: torch.Tensor) -> torch.Tensor:
        """计算内部区域掩码: 外壳2层以内为表面，其余为内部"""
        solid = (density > 0.5).float()
        eroded = -torch.nn.functional.max_pool3d(
            -solid, kernel_size=5, stride=1, padding=2
        )
        interior = eroded  # 深度腐蚀后剩余的为内部
        return interior

    def forward(self, cameras, use_repair: bool = False, force=None, fixed_mask=None):
        """渲染剪影

        Args:
            cameras: PyTorch3D相机
            use_repair: 是否使用修复后的密度
            force, fixed_mask: 载荷信息（repair时需要）
        """
        from pytorch3d.structures import Volumes

        batch_size = cameras.R.shape[0]

        if use_repair and self.repair_cnn is not None:
            densities = self.get_repaired_density(force, fixed_mask)[0]  # [1, N, N, N]
        else:
            densities = self.density

        colors = torch.sigmoid(self.log_colors)

        volumes = Volumes(
            densities=densities[None].expand(batch_size, *densities.shape),
            features=colors[None].expand(batch_size, *colors.shape),
            voxel_size=self._voxel_size,
        )

        return self._renderer(cameras=cameras, volumes=volumes)[0]

    def get_binary_voxels(self, threshold=0.5):
        """获取二值体素"""
        with torch.no_grad():
            densities = self.density
            return (densities > threshold).float()

    def structural_analysis(self, force: torch.Tensor) -> dict:
        """运行PINO结构分析

        Args:
            force: [1, 3, N, N, N]
        Returns:
            dict with displacement, stress, von_mises
        """
        if self.pino is None:
            raise RuntimeError("PINO model not attached")

        rho = self.density.unsqueeze(0)  # [1, 1, N, N, N]
        from structural.models.load_model import forward_surrogate_physical
        pino_fields = forward_surrogate_physical(self.pino, rho, force)
        out = pino_fields['output']
        displacement = pino_fields['displacement']
        stress = pino_fields['stress']
        vm = pino_fields['von_mises']

        return {
            'displacement': displacement,
            'stress': stress,
            'von_mises': vm,
            'pino_output': out,
        }
