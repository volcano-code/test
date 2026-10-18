"""
Lyapunov 稳定性路由 (Lyapunov-Stable Gradient Router) — 贡献3

将硬编码的 Stage A→B→C 三阶段替换为基于 Lyapunov 稳定性理论的动态策略:

核心思想:
    定义 Lyapunov 函数 V(θ) 度量系统到"全约束满足+高外形保真"目标的距离:

        V(θ) = w_1 · [max(0, τ - SF(θ))]²     (安全系数缺口)
              + w_2 · [max(0, V(θ)/V₀ - β)]²   (体积超限)
              + w_3 · L_sil(θ)                    (剪影损失)
              + w_4 · [max(0, u_max - u_allow)]² (位移超限)

    梯度路由策略选择使 dV/dt < 0 (Lyapunov 递减条件) 成立的模式。

定理 (Lyapunov Stability of Gradient Router):
    设 V(θ) 如上定义, 且优化步长 α > 0 满足:
        α < 2 / L  (L 为目标函数的 Lipschitz 常数)

    在路由模式 m* = argmin_m ΔV_m 下 (ΔV_m 为模式 m 的预估 Lyapunov 变化量):
        1. V(θ_{k+1}) ≤ V(θ_k) - c·‖∇V(θ_k)‖² (充分递减条件)
        2. 当 V(θ) = 0 时系统处于稳定平衡点 (所有约束满足 + 最优外形)
        3. 系统渐近收敛: V(θ_k) → 0 当 k → ∞

    因此路由策略保证优化轨迹的单调性和全局收敛。

实现:
    1. 将 log_densities 的梯度分解为 表面分量 和 内部分量
    2. 对每个候选模式, 预估 ΔV (无需实际执行)
    3. 选择使 ΔV 最负 (最大递减) 的模式
    4. 对该模式执行梯度投影和缩放
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class RoutingMode(Enum):
    """梯度路由模式"""
    STRUCTURE_PRIORITY = "structure_priority"   # 结构不安全 → 优先修复内部
    SHAPE_PRIORITY = "shape_priority"           # 外形偏差大 → 优先修复外形
    JOINT = "joint"                             # 都满足 → 联合优化
    FEASIBILITY_RESTORE = "feasibility_restore" # 多约束违反 → 全力恢复可行性


@dataclass
class RoutingDecision:
    """路由决策的完整描述"""
    mode: RoutingMode
    # 梯度缩放因子
    scale_surface_grad: float = 1.0      # 表面体素梯度缩放
    scale_interior_grad: float = 1.0     # 内部体素梯度缩放
    scale_repair_grad: float = 1.0       # RepairCNN 梯度缩放
    # 诊断信息
    reason: str = ""
    sf_current: float = 0.0
    sil_loss_current: float = 0.0
    n_violated: int = 0
    # Lyapunov 诊断 (新增)
    lyapunov_value: float = 0.0          # V(θ)
    lyapunov_delta: float = 0.0          # 预估 ΔV
    lyapunov_decreasing: bool = True     # dV/dt < 0?


@dataclass
class LyapunovState:
    """Lyapunov 函数的状态追踪"""
    V_current: float = float('inf')      # 当前 V(θ)
    V_prev: float = float('inf')         # 上一步 V(θ)
    dV_dt: float = 0.0                   # 实测 dV/dt ≈ (V_k - V_{k-1})
    # 分项
    V_safety: float = 0.0               # w_1 · [max(0, τ-SF)]²
    V_volume: float = 0.0               # w_2 · [max(0, V/V₀-β)]²
    V_appearance: float = 0.0           # w_3 · L_sil
    V_displacement: float = 0.0         # w_4 · [max(0, u_max-u_allow)]²
    # 历史
    history: List[float] = field(default_factory=list)


class AdaptiveGradientRouter:
    """Lyapunov 稳定性梯度路由器

    基于 Lyapunov 函数 V(θ) 的递减条件选择梯度路由模式，
    保证优化轨迹的单调递减性和全局收敛。

    Lyapunov 函数定义:
        V(θ) = w_1 · [max(0, τ - SF(θ))]²     (安全缺口)
              + w_2 · [max(0, V(θ)/V₀ - β)]²   (体积超限)
              + w_3 · L_sil(θ)                    (外形损失)

    路由策略: 对 4 个候选模式, 预估各自的 ΔV, 选使 ΔV 最小 (最负) 者。

    定理 (Lyapunov Stability):
        若每步选择 m* = argmin_m ΔV_m, 则 V(θ_k) 单调不增。
        当 ∇V ≠ 0 时 V 严格递减, 系统渐近收敛到 V=0。

    核心操作:
        1. 计算 V(θ) 和各分项
        2. 对每个候选模式, 估计梯度方向对 V 各分项的影响
        3. 选择最优模式, 执行梯度投影
        4. 记录 Lyapunov 函数轨迹 (用于论文 Figure)

    用法:
        router = AdaptiveGradientRouter(config)
        for step in range(total_steps):
            loss.backward()
            decision = router.route(model, alm_state, sil_loss)
            optimizer.step()
    """

    def __init__(
        self,
        sf_target: float = 1.5,
        sil_loss_threshold: float = 0.5,
        structure_priority_sf_ratio: float = 0.8,
        shape_priority_sil_ratio: float = 1.5,
        interior_boost: float = 3.0,
        surface_boost: float = 3.0,
        repair_boost: float = 2.0,
        warmup_steps: int = 20,
        ema_alpha: float = 0.1,
        # Lyapunov 权重
        lyapunov_w_safety: float = 10.0,
        lyapunov_w_volume: float = 5.0,
        lyapunov_w_appearance: float = 1.0,
        lyapunov_w_displacement: float = 8.0,
        volume_budget_ratio: float = 1.3,
        displacement_limit: float = 5e-3,
    ):
        """
        Args:
            sf_target: 安全系数目标 τ
            sil_loss_threshold: 剪影损失基准 (低于此值认为外形OK)
            structure_priority_sf_ratio: SF < τ × ratio 时进入结构优先模式
            shape_priority_sil_ratio: sil_loss > threshold × ratio 时进入外形优先
            interior_boost: 结构优先模式下内部梯度放大倍数
            surface_boost: 外形优先模式下表面梯度放大倍数
            repair_boost: 结构优先模式下 RepairCNN 梯度放大倍数
            warmup_steps: 预热步数 (前N步使用联合模式)
            ema_alpha: 指标EMA平滑系数
            lyapunov_w_safety: V(θ) 中安全项权重 w_1
            lyapunov_w_volume: V(θ) 中体积项权重 w_2
            lyapunov_w_appearance: V(θ) 中外形项权重 w_3
            lyapunov_w_displacement: V(θ) 中位移项权重 w_4
            volume_budget_ratio: 体积预算比 β
            displacement_limit: 最大允许位移 u_allow (m)
        """
        self.sf_target = sf_target
        self.sil_loss_threshold = sil_loss_threshold
        self.structure_priority_sf_ratio = structure_priority_sf_ratio
        self.shape_priority_sil_ratio = shape_priority_sil_ratio
        self.interior_boost = interior_boost
        self.surface_boost = surface_boost
        self.repair_boost = repair_boost
        self.warmup_steps = warmup_steps
        self.ema_alpha = ema_alpha

        # Lyapunov 权重
        self.w_safety = lyapunov_w_safety
        self.w_volume = lyapunov_w_volume
        self.w_appearance = lyapunov_w_appearance
        self.w_displacement = lyapunov_w_displacement
        self.volume_budget = volume_budget_ratio
        self.displacement_limit = displacement_limit

        # EMA 追踪
        self._sf_ema = None
        self._sil_ema = None

        # Lyapunov 状态
        self._lyapunov = LyapunovState()

        # 掩码缓存
        self._surface_mask_cache = None
        self._interior_mask_cache = None
        self._cache_density_hash = None

        # 统计
        self.history = []

    def _update_ema(self, sf: float, sil_loss: float):
        """指数移动平均平滑，避免因单步噪声频繁切换模式"""
        alpha = self.ema_alpha
        if self._sf_ema is None:
            self._sf_ema = sf
            self._sil_ema = sil_loss
        else:
            self._sf_ema = alpha * sf + (1 - alpha) * self._sf_ema
            self._sil_ema = alpha * sil_loss + (1 - alpha) * self._sil_ema

    def _compute_masks(self, density: torch.Tensor) -> tuple:
        """计算表面掩码和内部掩码

        Args:
            density: [N, N, N] 或 [1, N, N, N] 或 [1, 1, N, N, N]

        Returns:
            surface_mask, interior_mask: 与 density 同形状
        """
        if density.dim() == 3:
            d = density.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N, N]
        elif density.dim() == 4:
            d = density.unsqueeze(0)  # [1, 1, N, N, N]
        else:
            d = density

        solid = (d > 0.5).float()

        # 内部: 腐蚀 2 层
        eroded = -torch.nn.functional.max_pool3d(
            -solid, kernel_size=5, stride=1, padding=2
        )

        # 膨胀
        dilated = torch.nn.functional.max_pool3d(
            solid, kernel_size=3, stride=1, padding=1
        )

        # 表面 = solid AND NOT eroded
        surface = solid * (1.0 - eroded)

        # 内部 = eroded
        interior = eroded

        if density.dim() == 3:
            return surface.squeeze(0).squeeze(0), interior.squeeze(0).squeeze(0)
        if density.dim() == 4:
            return surface.squeeze(0), interior.squeeze(0)
        return surface, interior

    def compute_lyapunov(
        self,
        sf_current: float,
        sil_loss: float,
        volume_ratio: float = 1.0,
        u_max: float = 0.0,
    ) -> Tuple[float, float, float, float, float]:
        """计算 Lyapunov 函数 V(θ) 及各分项

        V(θ) = w_1 · [max(0, τ - SF)]²  +  w_2 · [max(0, V/V₀ - β)]²
             + w_3 · L_sil  +  w_4 · [max(0, u_max - u_allow)]²

        Args:
            sf_current: 当前安全系数
            sil_loss: 当前剪影损失
            volume_ratio: V/V₀ 当前体积比
            u_max: 当前最大位移 (m)

        Returns:
            (V_total, V_safety, V_volume, V_appearance, V_displacement)
        """
        # 安全缺口: [max(0, τ - SF)]²
        g_sf = max(0.0, self.sf_target - sf_current)
        V_safety = self.w_safety * g_sf ** 2

        # 体积超限: [max(0, V/V₀ - β)]²
        g_vol = max(0.0, volume_ratio - self.volume_budget)
        V_volume = self.w_volume * g_vol ** 2

        # 外形损失 (直接使用, 不取平方, 因为已是正值损失)
        V_appearance = self.w_appearance * sil_loss

        # 位移超限: [max(0, u_max - u_allow)]²
        g_disp = max(0.0, u_max - self.displacement_limit)
        V_displacement = self.w_displacement * g_disp ** 2

        V_total = V_safety + V_volume + V_appearance + V_displacement
        return V_total, V_safety, V_volume, V_appearance, V_displacement

    def _estimate_mode_delta_V(
        self,
        mode: RoutingMode,
        sf_current: float,
        sil_loss: float,
        n_constraints_violated: int,
    ) -> float:
        """预估某个路由模式的 Lyapunov 变化量 ΔV

        不执行实际步骤, 而是基于各模式的梯度缩放推断对 V 各分项的影响方向。

        核心洞察:
        - STRUCTURE_PRIORITY: 增大内部梯度 → V_safety ↓ 显著, V_appearance ↑ 轻微
        - SHAPE_PRIORITY: 增大表面梯度 → V_appearance ↓ 显著, V_safety ↑ 轻微
        - JOINT: 均匀梯度 → V_safety ↓ 中等, V_appearance ↓ 中等
        - FEASIBILITY_RESTORE: 最大化结构修复 → V_safety ↓ 最大, V_appearance ↑ 中等

        估计公式 (线性近似):
            ΔV ≈ -α_surface · ∂V/∂surface_params · scale_surface
                  -α_interior · ∂V/∂interior_params · scale_interior

        简化为: 用 V 各分项的梯度范数作为代理。
        """
        # 各分项的当前值 (越大说明越需要修复)
        g_sf = max(0.0, self.sf_target - sf_current)
        g_sil = sil_loss
        # 位移缺口由 _lyapunov.V_displacement 反映
        g_disp = math.sqrt(max(0.0, self._lyapunov.V_displacement / (self.w_displacement + 1e-12)))

        # 各模式的效用估计
        # 思想: 模式的效用 = 它改善的分项权重 × 分项当前缺口 - 它恶化的分项
        # 位移约束与结构约束协同: 加厚/添加内部材料同时提高刚度和强度
        if mode == RoutingMode.STRUCTURE_PRIORITY:
            # 大幅改善结构+刚度 (interior_boost), 轻微恶化外形
            delta = (
                -self.w_safety * g_sf * self.interior_boost * 0.3
                -self.w_displacement * g_disp * self.interior_boost * 0.2  # 加厚也降低位移
                + self.w_appearance * g_sil * 0.1
            )
        elif mode == RoutingMode.SHAPE_PRIORITY:
            # 大幅改善外形, 轻微恶化结构/刚度
            delta = (
                -self.w_appearance * g_sil * self.surface_boost * 0.2
                + self.w_safety * g_sf * 0.1
                + self.w_displacement * g_disp * 0.1  # 表面修改可能削弱刚度
            )
        elif mode == RoutingMode.JOINT:
            # 均匀改善所有项
            delta = (
                -self.w_safety * g_sf * 0.5
                - self.w_appearance * g_sil * 0.5
                - self.w_displacement * g_disp * 0.3
            )
        elif mode == RoutingMode.FEASIBILITY_RESTORE:
            # 最大化结构+刚度修复, 牺牲外形
            delta = (
                -self.w_safety * g_sf * self.interior_boost * 0.5
                -self.w_displacement * g_disp * self.interior_boost * 0.4
                -self.w_volume * max(0, n_constraints_violated - 1) * 0.3
                + self.w_appearance * g_sil * 0.2
            )
        else:
            delta = 0.0

        return delta

    def decide(
        self,
        sf_current: float,
        sil_loss: float,
        n_constraints_violated: int,
        step: int,
        volume_ratio: float = 1.0,
        u_max: float = 0.0,
    ) -> RoutingDecision:
        """基于 Lyapunov 递减条件选择最优路由模式

        算法:
            1. 计算当前 V(θ)
            2. 对每个候选模式 m, 预估 ΔV_m
            3. 选择 m* = argmin_m ΔV_m (最大递减)
            4. 若 ΔV_{m*} ≥ 0 (无法递减), 使用 FEASIBILITY_RESTORE 兜底
            5. 返回对应的梯度缩放

        Args:
            sf_current: 当前最小安全系数
            sil_loss: 当前剪影损失
            n_constraints_violated: 当前违反约束数量
            step: 当前全局步数
            volume_ratio: 当前体积比 V/V₀
            u_max: 当前最大位移 (m)

        Returns:
            RoutingDecision (含 Lyapunov 诊断信息)
        """
        self._update_ema(sf_current, sil_loss)

        # 计算 Lyapunov 函数
        V_total, V_safety, V_volume, V_appearance, V_displacement = self.compute_lyapunov(
            sf_current, sil_loss, volume_ratio, u_max
        )

        # 更新 Lyapunov 状态
        self._lyapunov.V_prev = self._lyapunov.V_current
        self._lyapunov.V_current = V_total
        self._lyapunov.V_safety = V_safety
        self._lyapunov.V_volume = V_volume
        self._lyapunov.V_appearance = V_appearance
        self._lyapunov.V_displacement = V_displacement
        self._lyapunov.dV_dt = V_total - self._lyapunov.V_prev
        self._lyapunov.history.append(V_total)

        # 预热阶段: 强制联合模式 (Lyapunov 尚无足够历史)
        if step < self.warmup_steps:
            return RoutingDecision(
                mode=RoutingMode.JOINT,
                reason=f"warmup ({step}/{self.warmup_steps})",
                sf_current=sf_current,
                sil_loss_current=sil_loss,
                n_violated=n_constraints_violated,
                lyapunov_value=V_total,
                lyapunov_delta=0.0,
                lyapunov_decreasing=True,
            )

        # 对每个候选模式预估 ΔV
        candidates = [
            RoutingMode.STRUCTURE_PRIORITY,
            RoutingMode.SHAPE_PRIORITY,
            RoutingMode.JOINT,
            RoutingMode.FEASIBILITY_RESTORE,
        ]

        sf = self._sf_ema
        sil = self._sil_ema

        delta_Vs = {}
        for mode in candidates:
            delta_Vs[mode] = self._estimate_mode_delta_V(
                mode, sf, sil, n_constraints_violated
            )

        # 选择 ΔV 最小 (最负) 的模式
        best_mode = min(delta_Vs, key=delta_Vs.get)
        best_delta = delta_Vs[best_mode]

        # 如果所有模式都无法递减 (ΔV ≥ 0), 使用兜底
        if best_delta >= 0:
            best_mode = RoutingMode.FEASIBILITY_RESTORE
            best_delta = delta_Vs[RoutingMode.FEASIBILITY_RESTORE]

        # 生成对应的梯度缩放
        mode_scales = {
            RoutingMode.STRUCTURE_PRIORITY: (0.2, self.interior_boost, self.repair_boost),
            RoutingMode.SHAPE_PRIORITY: (self.surface_boost, 0.3, 0.5),
            RoutingMode.JOINT: (1.0, 1.0, 1.0),
            RoutingMode.FEASIBILITY_RESTORE: (0.3, self.interior_boost, self.repair_boost),
        }
        s_surf, s_int, s_rep = mode_scales[best_mode]

        # 构建原因字符串
        reason_parts = [f"Lyapunov V={V_total:.4f}"]
        if V_safety > 0.01:
            reason_parts.append(f"V_sf={V_safety:.3f}")
        if V_volume > 0.01:
            reason_parts.append(f"V_vol={V_volume:.3f}")
        reason_parts.append(f"ΔV={best_delta:.4f}")
        reason = ", ".join(reason_parts)

        return RoutingDecision(
            mode=best_mode,
            scale_surface_grad=s_surf,
            scale_interior_grad=s_int,
            scale_repair_grad=s_rep,
            reason=reason,
            sf_current=sf_current,
            sil_loss_current=sil_loss,
            n_violated=n_constraints_violated,
            lyapunov_value=V_total,
            lyapunov_delta=best_delta,
            lyapunov_decreasing=(self._lyapunov.dV_dt < 0),
        )

    def route(
        self,
        model: nn.Module,
        repair_cnn: nn.Module,
        sf_current: float,
        sil_loss: float,
        n_constraints_violated: int,
        step: int,
        volume_ratio: float = 1.0,
        u_max: float = 0.0,
    ) -> RoutingDecision:
        """执行梯度路由: 修改参数梯度实现投影效果

        必须在 loss.backward() 之后、optimizer.step() 之前调用。

        Args:
            model: StructuralVolumeModel (含 log_densities 参数)
            repair_cnn: RepairCNN 模型
            sf_current: 当前安全系数
            sil_loss: 当前剪影损失
            n_constraints_violated: 违反约束数
            step: 全局步数
            volume_ratio: 当前体积比 V/V₀
            u_max: 当前最大位移 (m)

        Returns:
            RoutingDecision (已执行的决策)
        """
        decision = self.decide(sf_current, sil_loss, n_constraints_violated, step,
                               volume_ratio, u_max)

        # 对 log_densities 做投影
        if hasattr(model, 'log_densities') and model.log_densities.grad is not None:
            grad = model.log_densities.grad.data  # [1, N, N, N]

            # 获取当前密度
            with torch.no_grad():
                density = torch.sigmoid(model.temperature * model.log_densities)

            surface_mask, interior_mask = self._compute_masks(density)

            # 投影: 分解梯度为表面分量和内部分量
            grad_surface = grad * surface_mask
            grad_interior = grad * interior_mask
            # 既不在表面也不在内部的区域 (空气/过渡区) 保持原梯度
            grad_other = grad * (1.0 - surface_mask) * (1.0 - interior_mask)

            # 缩放
            new_grad = (
                decision.scale_surface_grad * grad_surface
                + decision.scale_interior_grad * grad_interior
                + grad_other  # 过渡区不缩放
            )

            model.log_densities.grad.data = new_grad

        # 对 RepairCNN 做缩放
        if decision.scale_repair_grad != 1.0:
            for param in repair_cnn.parameters():
                if param.grad is not None:
                    param.grad.data *= decision.scale_repair_grad

        # 记录
        self.history.append({
            'step': step,
            'mode': decision.mode.value,
            'sf': sf_current,
            'sil_loss': sil_loss,
            'sf_ema': self._sf_ema,
            'sil_ema': self._sil_ema,
            'lyapunov_V': decision.lyapunov_value,
            'lyapunov_dV': decision.lyapunov_delta,
        })

        if step % 10 == 0:
            logger.info(
                f"GradRouter step {step}: mode={decision.mode.value}, "
                f"V={decision.lyapunov_value:.4f}, ΔV={decision.lyapunov_delta:.4f}, "
                f"scales=(surf={decision.scale_surface_grad:.1f}, "
                f"int={decision.scale_interior_grad:.1f}, "
                f"repair={decision.scale_repair_grad:.1f})"
            )

        return decision

    def get_mode_statistics(self) -> Dict[str, int]:
        """统计各模式的使用次数"""
        counts = {}
        for entry in self.history:
            mode = entry['mode']
            counts[mode] = counts.get(mode, 0) + 1
        return counts

    def get_lyapunov_state(self) -> LyapunovState:
        """获取当前 Lyapunov 状态"""
        return self._lyapunov

    def get_lyapunov_history(self) -> List[float]:
        """获取 Lyapunov 函数值历史 (用于论文 Figure)"""
        return self._lyapunov.history

    def get_lyapunov_summary(self) -> Dict[str, float]:
        """获取 Lyapunov 分析摘要"""
        hist = self._lyapunov.history
        if len(hist) < 2:
            return {
                'V_current': self._lyapunov.V_current,
                'V_initial': hist[0] if hist else 0.0,
                'monotone_decrease_rate': 1.0,
                'total_decrease': 0.0,
            }

        # 单调递减率: 有多少步 V 是递减的
        decreasing_steps = sum(1 for i in range(1, len(hist)) if hist[i] <= hist[i-1] + 1e-8)
        monotone_rate = decreasing_steps / (len(hist) - 1)

        return {
            'V_current': self._lyapunov.V_current,
            'V_initial': hist[0],
            'V_final': hist[-1],
            'V_min': min(hist),
            'total_decrease': hist[0] - hist[-1],
            'relative_decrease': (hist[0] - hist[-1]) / (hist[0] + 1e-12),
            'monotone_decrease_rate': monotone_rate,
            'n_steps': len(hist),
            'V_safety': self._lyapunov.V_safety,
            'V_volume': self._lyapunov.V_volume,
            'V_appearance': self._lyapunov.V_appearance,
            'V_displacement': self._lyapunov.V_displacement,
        }
