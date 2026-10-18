"""
RepairCNN — 神经重参数化修复网络

输入: [ρ, σ_vm, vulnerability_map] → 3通道 64³
输出: [Δρ_thick, Δρ_strut, Δρ_hollow, Δρ_infill] → 4通道 64³

用CNN代替直接优化体素变量:
- 隐式正则化: CNN输出天然平滑
- 避免棋盘格伪影
- 参数空间更紧凑
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepairCNN(nn.Module):
    """轻量3D CNN，将结构信息映射为修复场

    各通道输出含义:
        0: Δρ_thick  ∈ [0, δ_max]   — 局部加厚
        1: Δρ_strut  ∈ [0, 1]       — 支撑柱密度
        2: Δρ_hollow ∈ [-1, 0]      — 挖空
        3: Δρ_infill ∈ [0, 1]       — 应力对齐填充
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: list = None,
        kernel_size: int = 3,
        delta_max: float = 0.3,
    ):
        super().__init__()
        if channels is None:
            channels = [16, 32, 32]

        self.delta_max = delta_max
        padding = kernel_size // 2

        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.extend([
                nn.Conv3d(prev_ch, ch, kernel_size, padding=padding),
                nn.InstanceNorm3d(ch),
                nn.GELU(),
            ])
            prev_ch = ch

        self.backbone = nn.Sequential(*layers)

        # 4个输出通道，各自独立的1x1卷积
        self.head_thick = nn.Conv3d(prev_ch, 1, 1)
        self.head_strut = nn.Conv3d(prev_ch, 1, 1)
        self.head_hollow = nn.Conv3d(prev_ch, 1, 1)
        self.head_infill = nn.Conv3d(prev_ch, 1, 1)

        # 初始化: 偏置设为负值，初始修复量接近0
        for head in [self.head_thick, self.head_strut, self.head_hollow, self.head_infill]:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -2.0)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B, 3, N, N, N] — [density, von_mises_normalized, vulnerability_map]
        Returns:
            dict with repair fields, each [B, 1, N, N, N]
        """
        feat = self.backbone(x)

        # 各通道不同激活
        thick = torch.sigmoid(self.head_thick(feat)) * self.delta_max     # [0, δ_max]
        strut = torch.sigmoid(self.head_strut(feat))                       # [0, 1]
        hollow = -torch.sigmoid(self.head_hollow(feat))                    # [-1, 0]
        infill = torch.sigmoid(self.head_infill(feat))                     # [0, 1]

        return {
            'thick': thick,
            'strut': strut,
            'hollow': hollow,
            'infill': infill,
            'combined': thick + strut + hollow + infill,
        }

    def apply_repair(
        self,
        density: torch.Tensor,
        repair_fields: dict,
        interior_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """将修复场应用到密度场

        Args:
            density: [B, 1, N, N, N] 原始密度
            repair_fields: forward()的输出
            interior_mask: [B, 1, N, N, N] 内部区域掩码（外表面不修改）
        Returns:
            [B, 1, N, N, N] 修复后密度
        """
        delta = repair_fields['combined']

        if interior_mask is not None:
            delta = delta * interior_mask

        repaired = torch.clamp(density + delta, 0, 1)
        return repaired
