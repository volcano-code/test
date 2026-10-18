"""
增广拉格朗日法 (Augmented Lagrangian Method) — 贡献1

将结构优化从 "加权求和多目标" 升级为 "带硬约束的约束优化":

    min   L_appearance(θ)                        ← 目标函数: 外形匹配
    s.t.  g1: SF_min(θ) ≥ τ                     ← 安全系数硬约束
          g2: V(θ) / V₀ ≤ β                     ← 体积预算约束
          g3: overhang_ratio(θ) ≤ ε_print        ← 可打印性约束
          g4: ‖u‖_∞ ≤ u_allow                   ← 位移/刚度约束

ALM 拉格朗日函数:

    L_ALM = L_app + Σᵢ [ μᵢ · cᵢ + (ρ/2) · cᵢ² ]

其中 cᵢ = max(0, gᵢ(θ)) 为约束违反量, μᵢ 为拉格朗日乘子, ρ 为罚因子。

ALM 相比加权求和的理论优势:
1. 不需要手动调节 λ 权重 — 乘子自动收敛到最优对偶值
2. 保证约束满足 — KKT 条件在收敛时自然成立
3. 避免 ill-conditioning — 罚因子 ρ 不需要趋于无穷
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ConstraintStatus(Enum):
    """约束状态"""
    SATISFIED = "satisfied"      # 约束已满足
    VIOLATED = "violated"        # 约束被违反
    ACTIVE = "active"            # 约束恰好在边界上 (|violation| < tol)


@dataclass
class ConstraintState:
    """单个约束的完整状态"""
    name: str
    violation: float = 0.0       # 约束违反量 max(0, g(θ))
    multiplier: float = 0.0      # 拉格朗日乘子 μ
    status: ConstraintStatus = ConstraintStatus.SATISFIED
    history: list = field(default_factory=list)  # 违反量历史

    def update_history(self, max_len: int = 200):
        self.history.append(self.violation)
        if len(self.history) > max_len:
            self.history = self.history[-max_len:]


@dataclass
class ALMConfig:
    """ALM 超参数配置"""
    # 约束阈值
    safety_factor_target: float = 1.5       # 安全系数目标 τ
    volume_budget_ratio: float = 1.3        # 体积预算 β
    overhang_limit: float = 0.15            # 悬垂比例上限
    displacement_limit: float = 5e-3        # 最大允许位移 u_allow (m), 默认 5mm

    # ALM 参数
    rho_init: float = 1.0                   # 初始罚因子
    rho_max: float = 1000.0                 # 罚因子上限
    rho_growth: float = 2.0                 # 罚因子增长倍率
    mu_max: float = 1000.0                  # 乘子上限

    # 收敛判据
    constraint_tol: float = 0.01            # 约束满足容差
    dual_update_interval: int = 5           # 对偶变量更新间隔 (每N步更新一次)
    improvement_threshold: float = 0.25     # 约束改善不足时增大 ρ 的阈值

    # 目标函数权重 (仅对 appearance loss)
    lambda_appearance: float = 10.0
    lambda_shape_preserve: float = 5.0
    lambda_smoothness: float = 1.0


class ALMOptimizer:
    """增广拉格朗日优化器

    将 {安全系数, 体积, 可打印性} 作为硬约束处理，
    用 ALM 迭代求解约束优化问题。

    核心公式:
        L_ALM(θ, μ, ρ) = L_appearance(θ)
                        + Σᵢ μᵢ · max(0, gᵢ(θ))
                        + (ρ/2) · Σᵢ max(0, gᵢ(θ))²

    对偶更新 (外层循环):
        μᵢ ← max(0, μᵢ + ρ · gᵢ(θ*))
        ρ  ← min(ρ_max, γ · ρ)  若约束改善不足

    用法:
        alm = ALMOptimizer(config)
        for step in range(total_steps):
            loss = alm.compute_alm_loss(L_app, constraints)
            loss.backward()
            optimizer.step()
            alm.step(step)
    """

    def __init__(self, config: ALMConfig | None = None):
        self.config = config or ALMConfig()

        # 罚因子 (所有约束共享)
        self.rho = self.config.rho_init

        # 约束状态
        self.constraints: Dict[str, ConstraintState] = {
            'safety': ConstraintState(name='safety'),
            'volume': ConstraintState(name='volume'),
            'overhang': ConstraintState(name='overhang'),
            'displacement': ConstraintState(name='displacement'),
        }

        # 上一轮约束违反总量 (用于判断是否需要增大 ρ)
        self._prev_total_violation = float('inf')
        self._step_count = 0

    def compute_constraint_violations(
        self,
        von_mises: torch.Tensor,
        density: torch.Tensor,
        density_original: torch.Tensor,
        yield_stress: float,
        displacement: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """计算四个约束的违反量（可微分）

        返回 gᵢ(θ)，其中 gᵢ > 0 表示违反, gᵢ ≤ 0 表示满足。

        Args:
            von_mises: [B, 1, N, N, N] — Von Mises 应力场
            density: [B, 1, N, N, N] — 当前密度场
            density_original: [B, 1, N, N, N] — 原始密度场
            yield_stress: 材料屈服强度 (Pa)
            displacement: [B, 3, N, N, N] — PINO 预测位移场 (可选)

        Returns:
            dict[str, Tensor]: 每个约束的违反量标量
        """
        violations = {}

        # --- 约束1: 安全系数 SF_min ≥ τ ---
        # 等价于: max(σ_vm) / σ_yield - 1/τ ≤ 0
        # 使用 log-sum-exp 近似 smooth maximum (数值稳定)
        solid_mask = (density > 0.1).float()
        vm_masked = von_mises * solid_mask
        # Normalize: 先除以 yield_stress 使数值合理
        vm_norm = vm_masked / (yield_stress + 1e-12)
        # log-sum-exp smooth max: max ≈ (1/β) * log(Σ exp(β * x))
        beta_sm = 20.0
        vm_flat = (vm_norm * solid_mask).reshape(-1)
        n_solid = solid_mask.sum().clamp(min=1.0)
        # 数值稳定: 减去最大值
        vm_max_val = vm_flat.max()
        lse = vm_max_val + torch.log(
            (torch.exp(beta_sm * (vm_flat - vm_max_val)) * solid_mask.reshape(-1)).sum()
            / n_solid + 1e-12
        ) / beta_sm
        # lse 现在是归一化后的 smooth max(σ_vm / σ_yield)
        # g1 = lse - 1/τ (若 lse > 1/τ 说明最大应力超标)
        violations['safety'] = lse - 1.0 / self.config.safety_factor_target

        # --- 约束2: 体积 V/V₀ ≤ β ---
        # g_volume = V_new / (β · V_original) - 1
        v_new = density.sum()
        v_orig = density_original.sum().clamp(min=1.0)
        v_budget = v_orig * self.config.volume_budget_ratio
        violations['volume'] = v_new / v_budget - 1.0

        # --- 约束3: 悬垂 overhang_ratio ≤ ε ---
        # 简化: 悬垂比例
        solid = (density > 0.5).float()
        n_solid = solid.sum().clamp(min=1.0)
        # 下方支撑 (Y轴=dim3)
        support_below = torch.zeros_like(solid)
        support_below[:, :, :, 1:, :] = solid[:, :, :, :-1, :]
        support_diag = torch.nn.functional.max_pool3d(
            support_below, kernel_size=(3, 1, 3), stride=1, padding=(1, 0, 1)
        )
        unsupported = solid * (1.0 - support_diag)
        unsupported[:, :, :, 0, :] = 0  # 底层不算
        overhang_ratio = unsupported.sum() / n_solid
        violations['overhang'] = overhang_ratio - self.config.overhang_limit

        # --- 约束4: 位移 ‖u‖_∞ ≤ u_allow ---
        # g4 = u_max / u_allow - 1  (若 > 0 表示位移超标)
        # 使用 log-sum-exp smooth max 保持可微分
        if displacement is not None:
            u_allow = self.config.displacement_limit
            # displacement: [B, 3, N, N, N] — 在固体区域取位移模
            u_magnitude = torch.norm(displacement, dim=1, keepdim=True)  # [B, 1, N, N, N]
            u_solid = u_magnitude * solid_mask
            u_flat = u_solid.reshape(-1)
            u_max_val = u_flat.max()
            # smooth max via log-sum-exp
            beta_u = 20.0
            u_lse = u_max_val + torch.log(
                (torch.exp(beta_u * (u_flat - u_max_val)) * solid_mask.reshape(-1)).sum()
                / n_solid + 1e-12
            ) / beta_u
            violations['displacement'] = u_lse / (u_allow + 1e-12) - 1.0
        else:
            # 无位移数据时，约束自动满足
            violations['displacement'] = torch.tensor(
                -1.0, device=von_mises.device, dtype=von_mises.dtype
            )

        return violations

    def compute_alm_loss(
        self,
        appearance_loss: torch.Tensor,
        violations: Dict[str, torch.Tensor],
        shape_preserve_loss: torch.Tensor | None = None,
        smoothness_loss_val: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """计算 ALM 总损失

        L_ALM = λ_app · L_app + λ_shape · L_shape + λ_smooth · L_smooth
              + Σᵢ [ μᵢ · max(0, gᵢ) + (ρ/2) · max(0, gᵢ)² ]

        Args:
            appearance_loss: 外形匹配损失 (剪影IOU+BCE+MS-SSIM)
            violations: compute_constraint_violations 的输出
            shape_preserve_loss: 外形保持正则项 (非约束，直接加入目标)
            smoothness_loss_val: 平滑正则项

        Returns:
            dict with 'total', 'appearance', 'penalty', per-constraint terms
        """
        cfg = self.config
        device = appearance_loss.device

        # 目标函数部分
        total = cfg.lambda_appearance * appearance_loss

        if shape_preserve_loss is not None:
            total = total + cfg.lambda_shape_preserve * shape_preserve_loss

        if smoothness_loss_val is not None:
            total = total + cfg.lambda_smoothness * smoothness_loss_val

        # ALM 惩罚项: Σ [ μ · c + (ρ/2) · c² ]
        penalty_total = torch.tensor(0.0, device=device)
        constraint_details = {}

        for name, violation in violations.items():
            state = self.constraints[name]
            mu = state.multiplier

            # c = max(0, g)  — 约束违反量的正部分
            c = torch.relu(violation)

            # ALM 项: μ·c + (ρ/2)·c²
            alm_term = mu * c + (self.rho / 2.0) * c ** 2

            penalty_total = penalty_total + alm_term

            # 记录
            constraint_details[f'c_{name}'] = c.detach()
            constraint_details[f'g_{name}'] = violation.detach()
            constraint_details[f'mu_{name}'] = mu

        total = total + penalty_total

        result = {
            'total': total,
            'appearance': appearance_loss.detach(),
            'penalty': penalty_total.detach(),
            'rho': self.rho,
        }
        result.update(constraint_details)

        return result

    def step(self, global_step: int):
        """ALM 外层更新: 更新乘子 μ 和罚因子 ρ

        在每个 dual_update_interval 步执行:
        1. μᵢ ← max(0, μᵢ + ρ · gᵢ)    — 乘子递增
        2. 如果约束改善不足: ρ ← γ · ρ   — 增大罚因子
        """
        self._step_count = global_step

        if global_step % self.config.dual_update_interval != 0:
            return

        if global_step == 0:
            return

        # 计算当前总违反量
        current_total_violation = sum(
            max(0.0, s.violation) for s in self.constraints.values()
        )

        # 判断是否需要增大罚因子
        if self._prev_total_violation > 0:
            improvement = 1.0 - current_total_violation / (self._prev_total_violation + 1e-12)
            if improvement < self.config.improvement_threshold:
                # 约束改善不足，增大 ρ
                self.rho = min(self.config.rho_max, self.rho * self.config.rho_growth)
                logger.info(f"ALM step {global_step}: ρ increased to {self.rho:.2f} "
                            f"(improvement={improvement:.3f})")

        self._prev_total_violation = current_total_violation

        # 更新每个约束的乘子
        for name, state in self.constraints.items():
            old_mu = state.multiplier
            # μ ← max(0, μ + ρ · g)
            state.multiplier = min(
                self.config.mu_max,
                max(0.0, old_mu + self.rho * state.violation)
            )

            # 更新状态
            if state.violation <= -self.config.constraint_tol:
                state.status = ConstraintStatus.SATISFIED
            elif abs(state.violation) <= self.config.constraint_tol:
                state.status = ConstraintStatus.ACTIVE
            else:
                state.status = ConstraintStatus.VIOLATED

            state.update_history()

            if global_step % 20 == 0:
                logger.info(
                    f"  [{name}] μ: {old_mu:.3f}→{state.multiplier:.3f}, "
                    f"violation: {state.violation:.4f}, status: {state.status.value}"
                )

    def update_violations(self, violations: Dict[str, torch.Tensor]):
        """将当前计算的约束违反量同步到 ConstraintState (用于 step() 中的对偶更新)"""
        for name, v in violations.items():
            if name in self.constraints:
                self.constraints[name].violation = float(v.detach().cpu())

    def is_feasible(self) -> bool:
        """检查所有约束是否均已满足"""
        return all(
            s.status != ConstraintStatus.VIOLATED
            for s in self.constraints.values()
        )

    def get_summary(self) -> Dict[str, float]:
        """获取当前 ALM 状态摘要"""
        summary = {'rho': self.rho, 'feasible': float(self.is_feasible())}
        for name, state in self.constraints.items():
            summary[f'mu_{name}'] = state.multiplier
            summary[f'violation_{name}'] = state.violation
        return summary

    def state_dict(self) -> dict:
        """序列化 ALM 状态以便断点续训"""
        return {
            'rho': self.rho,
            'prev_total_violation': self._prev_total_violation,
            'step_count': self._step_count,
            'constraints': {
                name: {
                    'multiplier': s.multiplier,
                    'violation': s.violation,
                    'history': s.history,
                }
                for name, s in self.constraints.items()
            },
        }

    def load_state_dict(self, state: dict):
        """恢复 ALM 状态"""
        self.rho = state['rho']
        self._prev_total_violation = state['prev_total_violation']
        self._step_count = state['step_count']
        for name, data in state['constraints'].items():
            if name in self.constraints:
                self.constraints[name].multiplier = data['multiplier']
                self.constraints[name].violation = data['violation']
                self.constraints[name].history = data.get('history', [])
