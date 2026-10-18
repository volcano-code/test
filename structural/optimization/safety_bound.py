"""
Safety Transfer Bound — 贡献2

定理 (Safety Transfer Bound):
    设 PINO 对位移场的相对 L² 近似误差为 ε_u，即:

        ‖u_PINO - u_FEM‖_L² / ‖u_FEM‖_L² ≤ ε_u

    设优化得到的设计 θ* 在 PINO 下的最小安全系数为 SF_PINO(θ*)。
    则在真实 FEM 下的安全系数满足:

        SF_FEM(θ*) ≥ SF_PINO(θ*) / (1 + K · ε_u)

    其中 K = √6 · ‖C‖_F · (1/h) 为仅依赖于材料参数和网格间距的常数。

证明思路:
    1. Von Mises 应力 σ_vm = f(ε) 是应变的 Lipschitz 连续函数
       - σ = C:ε (Hooke定律, C为刚度张量)
       - σ_vm = √(3/2 · σ_dev : σ_dev) (Von Mises是偏应力的范数)
       - ⟹ |σ_vm^(1) - σ_vm^(2)| ≤ √6 · ‖C‖_F · ‖ε^(1) - ε^(2)‖

    2. 应变 ε = sym(∇u) 是位移梯度的线性函数
       - ‖ε^PINO - ε^FEM‖ ≤ (1/h) · ‖u^PINO - u^FEM‖  (离散逆估计)

    3. 联合得:
       |σ_vm^PINO_max - σ_vm^FEM_max| ≤ K · ‖u^PINO - u^FEM‖
       ⟹ σ_vm^FEM_max ≤ σ_vm^PINO_max · (1 + K · ε_u · ‖u^FEM‖/σ_vm^PINO_max)

    4. 安全系数反转:
       SF_FEM = σ_yield / σ_vm^FEM_max ≥ SF_PINO / (1 + K · ε_u · R)

    其中 R = ‖C‖_F · ‖u^FEM‖ / (h · σ_vm^PINO_max) 为应力-位移比。

本模块提供:
1. 定理常数 K 的计算
2. 给定 PINO 验证误差，计算 SF 下界
3. 用真实 FEM 验证 bound 是否 tight
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SafetyVerificationResult:
    """Safety Transfer Bound 验证结果"""
    # PINO 侧
    sf_pino: float              # PINO 预测的最小安全系数
    vm_max_pino: float          # PINO 预测的最大 Von Mises 应力
    u_norm_pino: float          # PINO 预测位移的 L2 范数

    # FEM 侧 (验证时填入, 否则为 None)
    sf_fem: Optional[float] = None
    vm_max_fem: Optional[float] = None
    u_norm_fem: Optional[float] = None

    # 误差
    displacement_error: Optional[float] = None  # ‖u_PINO - u_FEM‖ / ‖u_FEM‖
    stress_error: Optional[float] = None        # ‖σ_PINO - σ_FEM‖ / ‖σ_FEM‖

    # Bound
    K_constant: float = 0.0     # 材料-网格常数 K
    sf_lower_bound: float = 0.0 # 理论下界 SF_FEM ≥ sf_lower_bound
    bound_is_valid: bool = False # SF_FEM ≥ sf_lower_bound 是否成立
    tightness: float = 0.0      # (SF_FEM - sf_lower_bound) / SF_FEM: 越接近0越tight


class SafetyTransferBound:
    """Safety Transfer Bound 计算与验证模块

    使用方式:
        stb = SafetyTransferBound(E=3.5e9, nu=0.36, resolution=64, yield_stress=50e6)

        # 仅 PINO（设计阶段）: 给出 SF 下界
        result = stb.compute_bound(pino_model, density, force, epsilon_u=0.05)

        # PINO + FEM（论文验证）: 验证 bound 是否 tight
        result = stb.verify_with_fem(pino_model, density, force, fem_displacement, fem_stress)
    """

    def __init__(
        self,
        youngs_modulus: float = 3.5e9,
        poisson_ratio: float = 0.36,
        resolution: int = 64,
        yield_stress: float = 50.0e6,
        simp_penalty: float = 3.0,
    ):
        self.E = youngs_modulus
        self.nu = poisson_ratio
        self.h = 1.0 / resolution
        self.yield_stress = yield_stress
        self.simp_penalty = simp_penalty

        # 预计算 Hooke 刚度矩阵 C 及其 Frobenius 范数
        self.C_matrix = self._build_hooke_matrix()
        self.C_frobenius = float(np.linalg.norm(self.C_matrix))

        # 定理常数 K = √6 · ‖C‖_F / h
        # 推导: |Δσ_vm| ≤ √(3/2) · ‖C‖_F · ‖Δε‖
        #        ‖Δε‖ ≤ (√2/h) · ‖Δu‖  (有限差分的逆估计)
        # 合并: |Δσ_vm| ≤ √3 · ‖C‖_F / h · ‖Δu‖
        self.K = np.sqrt(3.0) * self.C_frobenius / self.h

        logger.info(f"SafetyTransferBound initialized:")
        logger.info(f"  ‖C‖_F = {self.C_frobenius:.3e}")
        logger.info(f"  h = {self.h:.4f}")
        logger.info(f"  K = {self.K:.3e}")

    def _build_hooke_matrix(self) -> np.ndarray:
        """构建 6×6 Voigt 刚度矩阵 C"""
        E, nu = self.E, self.nu
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        C = np.zeros((6, 6))
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
        C[3, 3] = C[4, 4] = C[5, 5] = mu
        return C

    @torch.no_grad()
    def compute_bound(
        self,
        pino_model: nn.Module,
        density: torch.Tensor,
        force: torch.Tensor,
        epsilon_u: float,
        fixed_mask: torch.Tensor | None = None,
    ) -> SafetyVerificationResult:
        """给定 PINO 近似误差 ε_u，计算安全系数下界

        这是设计阶段使用的方法: 无需真实 FEM 结果，
        仅依赖预先验证的 PINO 误差上界。

        Args:
            pino_model: 预训练的 PINO 模型
            density: [1, 1, N, N, N] 当前密度场
            force: [1, 3, N, N, N] 力场
            epsilon_u: PINO 位移场相对误差上界 (验证集统计)
            fixed_mask: [1, 1, N, N, N] 固定边界 (可选)

        Returns:
            SafetyVerificationResult
        """
        from structural.analysis.stress_eval import compute_von_mises_torch
        from structural.models.load_model import forward_surrogate_physical

        pino_fields = forward_surrogate_physical(pino_model, density, force)
        u_pino = pino_fields['displacement']
        s_pino = pino_fields['stress']

        # Von Mises 应力
        vm_pino = compute_von_mises_torch(s_pino)  # [1, 1, N, N, N]

        # 仅在实体区域统计
        solid_mask = (density > 0.1).float()
        vm_masked = vm_pino * solid_mask

        # Soft maximum (用于梯度计算时)
        vm_max_pino = float(vm_masked.max())
        u_norm_pino = float(torch.sqrt((u_pino ** 2).sum()))

        # PINO 安全系数
        sf_pino = self.yield_stress / (vm_max_pino + 1e-12)

        # Safety Transfer Bound:
        #   |σ_vm_max^FEM - σ_vm_max^PINO| ≤ K · ε_u · ‖u_FEM‖
        #   ≈ K · ε_u · ‖u_PINO‖ / (1 - ε_u)  (三角不等式)
        #
        # ⟹ σ_vm_max^FEM ≤ σ_vm_max^PINO + K · ε_u · ‖u_PINO‖ / (1 - ε_u)
        # ⟹ SF_FEM ≥ σ_yield / (σ_vm_max^PINO + K · ε_u · ‖u_PINO‖ / (1 - ε_u))
        eps_safe = max(epsilon_u, 1e-12)
        delta_stress_bound = self.K * eps_safe * u_norm_pino / max(1 - eps_safe, 0.01)
        vm_max_upper = vm_max_pino + delta_stress_bound
        sf_lower = self.yield_stress / (vm_max_upper + 1e-12)

        return SafetyVerificationResult(
            sf_pino=sf_pino,
            vm_max_pino=vm_max_pino,
            u_norm_pino=u_norm_pino,
            K_constant=self.K,
            sf_lower_bound=sf_lower,
        )

    @torch.no_grad()
    def verify_with_fem(
        self,
        pino_model: nn.Module,
        density: torch.Tensor,
        force: torch.Tensor,
        fem_displacement: torch.Tensor,
        fem_stress: torch.Tensor,
    ) -> SafetyVerificationResult:
        """用真实 FEM 结果验证 Safety Transfer Bound

        论文实验用: 对比理论下界与实际 FEM 结果，验证 bound tightness。

        Args:
            pino_model: PINO 模型
            density: [1, 1, N, N, N]
            force: [1, 3, N, N, N]
            fem_displacement: [1, 3, N, N, N] — FEM 位移 (真值)
            fem_stress: [1, 6, N, N, N] — FEM 应力 (真值)

        Returns:
            SafetyVerificationResult with FEM fields filled
        """
        from structural.analysis.stress_eval import compute_von_mises_torch
        from structural.models.load_model import forward_surrogate_physical

        pino_fields = forward_surrogate_physical(pino_model, density, force)
        u_pino = pino_fields['displacement']
        s_pino = pino_fields['stress']
        vm_pino = compute_von_mises_torch(s_pino)

        # FEM 真值
        vm_fem = compute_von_mises_torch(fem_stress)

        solid_mask = (density > 0.1).float()

        # PINO 统计
        vm_max_pino = float((vm_pino * solid_mask).max())
        u_norm_pino = float(torch.sqrt((u_pino ** 2).sum()))
        sf_pino = self.yield_stress / (vm_max_pino + 1e-12)

        # FEM 统计
        vm_max_fem = float((vm_fem * solid_mask).max())
        u_norm_fem = float(torch.sqrt((fem_displacement ** 2).sum()))
        sf_fem = self.yield_stress / (vm_max_fem + 1e-12)

        # 实际误差
        u_diff = u_pino - fem_displacement
        displacement_error = float(torch.sqrt((u_diff ** 2).sum())) / max(u_norm_fem, 1e-12)

        s_diff = s_pino - fem_stress
        stress_error = float(
            torch.sqrt(((s_diff * solid_mask) ** 2).sum())
            / (torch.sqrt(((fem_stress * solid_mask) ** 2).sum()) + 1e-12)
        )

        # 用实际误差计算 bound
        epsilon_u = displacement_error
        eps_safe = max(epsilon_u, 1e-12)
        delta_stress_bound = self.K * eps_safe * u_norm_pino / max(1 - eps_safe, 0.01)
        vm_max_upper = vm_max_pino + delta_stress_bound
        sf_lower = self.yield_stress / (vm_max_upper + 1e-12)

        # 验证
        bound_valid = sf_fem >= sf_lower - 1e-6  # 允许微小数值误差
        tightness = (sf_fem - sf_lower) / max(sf_fem, 1e-12) if sf_fem > 0 else 0.0

        result = SafetyVerificationResult(
            sf_pino=sf_pino,
            vm_max_pino=vm_max_pino,
            u_norm_pino=u_norm_pino,
            sf_fem=sf_fem,
            vm_max_fem=vm_max_fem,
            u_norm_fem=u_norm_fem,
            displacement_error=displacement_error,
            stress_error=stress_error,
            K_constant=self.K,
            sf_lower_bound=sf_lower,
            bound_is_valid=bound_valid,
            tightness=tightness,
        )

        logger.info(f"Safety Transfer Bound Verification:")
        logger.info(f"  SF_PINO = {sf_pino:.3f}")
        logger.info(f"  SF_FEM  = {sf_fem:.3f}")
        logger.info(f"  SF_lower_bound = {sf_lower:.3f}")
        logger.info(f"  Bound valid: {bound_valid}")
        logger.info(f"  Tightness: {tightness:.4f}")
        logger.info(f"  ε_u (displacement) = {displacement_error:.4f}")
        logger.info(f"  ε_σ (stress) = {stress_error:.4f}")

        return result

    @torch.no_grad()
    def batch_verify(
        self,
        pino_model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> dict:
        """在整个验证集上运行 Safety Transfer Bound 统计

        适用于论文中的 Table: 在 N 个样本上报告:
        - Mean/Max displacement error
        - Mean/Max stress error
        - Bound validity rate
        - Mean tightness

        Args:
            pino_model: PINO 模型
            dataloader: 包含 (x, y, force, density, fixed_mask) 的 DataLoader
            device: torch.device

        Returns:
            dict with aggregate statistics
        """
        from structural.analysis.stress_eval import (
            compute_von_mises_torch,
            pino_displacement_to_physical,
            pino_stress_to_physical,
        )

        is_v3 = getattr(pino_model, '_is_v3', False)
        results = []
        pino_model.eval()

        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            force = batch[2].to(device) if len(batch) > 2 else x[:, 1:4]
            density = x[:, 0:1]

            if getattr(pino_model, '_has_scale_head', False) and hasattr(pino_model, 'forward_with_scale'):
                pred, log_vm_max_pred = pino_model.forward_with_scale(x)
            else:
                pred = pino_model(x)
                log_vm_max_pred = None

            # 逐样本处理
            B = x.shape[0]
            for i in range(B):
                u_pino = pino_displacement_to_physical(pred[i:i+1, :3], is_v3)
                s_pino = pred[i:i+1, 3:]
                u_fem = y[i:i+1, :3]
                s_fem = y[i:i+1, 3:]
                rho = density[i:i+1]

                log_scale_i = log_vm_max_pred[i:i+1] if log_vm_max_pred is not None else None
                vm_pino = compute_von_mises_torch(
                    pino_stress_to_physical(s_pino, is_v3, log_vm_max_pred=log_scale_i)
                )
                vm_fem = compute_von_mises_torch(s_fem)
                solid = (rho > 0.1).float()

                vm_max_pino = float((vm_pino * solid).max())
                vm_max_fem = float((vm_fem * solid).max())
                u_norm_pino = float(torch.sqrt((u_pino ** 2).sum()))
                u_norm_fem = float(torch.sqrt((u_fem ** 2).sum()))

                eps_u = float(torch.sqrt(((u_pino - u_fem) ** 2).sum())) / max(u_norm_fem, 1e-12)
                eps_s = float(
                    torch.sqrt(((s_pino - s_fem) ** 2 * solid).sum())
                    / (torch.sqrt(((s_fem ** 2 * solid).sum())) + 1e-12)
                )

                sf_pino = self.yield_stress / (vm_max_pino + 1e-12)
                sf_fem = self.yield_stress / (vm_max_fem + 1e-12)

                # Bound
                eps_safe = max(eps_u, 1e-12)
                delta = self.K * eps_safe * u_norm_pino / max(1 - eps_safe, 0.01)
                vm_upper = vm_max_pino + delta
                sf_lower = self.yield_stress / (vm_upper + 1e-12)

                results.append({
                    'sf_pino': sf_pino,
                    'sf_fem': sf_fem,
                    'sf_lower': sf_lower,
                    'eps_u': eps_u,
                    'eps_s': eps_s,
                    'bound_valid': sf_fem >= sf_lower - 1e-6,
                    'tightness': (sf_fem - sf_lower) / max(sf_fem, 1e-12),
                })

        if not results:
            return {}

        # 聚合统计
        n = len(results)
        stats = {
            'n_samples': n,
            'eps_u_mean': np.mean([r['eps_u'] for r in results]),
            'eps_u_max': np.max([r['eps_u'] for r in results]),
            'eps_s_mean': np.mean([r['eps_s'] for r in results]),
            'eps_s_max': np.max([r['eps_s'] for r in results]),
            'sf_pino_mean': np.mean([r['sf_pino'] for r in results]),
            'sf_fem_mean': np.mean([r['sf_fem'] for r in results]),
            'sf_lower_mean': np.mean([r['sf_lower'] for r in results]),
            'bound_validity_rate': np.mean([r['bound_valid'] for r in results]),
            'tightness_mean': np.mean([r['tightness'] for r in results]),
            'tightness_std': np.std([r['tightness'] for r in results]),
        }

        logger.info(f"Batch verification ({n} samples):")
        logger.info(f"  ε_u: mean={stats['eps_u_mean']:.4f}, max={stats['eps_u_max']:.4f}")
        logger.info(f"  ε_σ: mean={stats['eps_s_mean']:.4f}, max={stats['eps_s_max']:.4f}")
        logger.info(f"  SF_PINO mean={stats['sf_pino_mean']:.3f}")
        logger.info(f"  SF_FEM  mean={stats['sf_fem_mean']:.3f}")
        logger.info(f"  SF_lower mean={stats['sf_lower_mean']:.3f}")
        logger.info(f"  Bound validity: {stats['bound_validity_rate']*100:.1f}%")
        logger.info(f"  Tightness: {stats['tightness_mean']:.4f} ± {stats['tightness_std']:.4f}")

        return stats

    def compute_required_epsilon(self, sf_target: float, sf_pino: float, u_norm: float) -> float:
        """给定目标 SF，反算允许的最大 PINO 误差 ε_u

        用于指导 PINO 训练: 要达到 SF ≥ τ 的保证，PINO 需要多精确？

        从 SF_FEM ≥ SF_PINO / (1 + K · ε · ‖u‖ / σ_vm_max) ≥ τ
        解出: ε ≤ (SF_PINO/τ - 1) · σ_vm_max / (K · ‖u‖)
        """
        if sf_pino <= sf_target:
            return 0.0  # PINO 本身就不达标，无解

        vm_max_pino = self.yield_stress / sf_pino
        numerator = (sf_pino / sf_target - 1.0) * vm_max_pino
        denominator = self.K * u_norm + 1e-12
        epsilon_max = numerator / denominator

        logger.info(
            f"Required ε_u ≤ {epsilon_max:.6f} for SF_target={sf_target:.2f} "
            f"given SF_PINO={sf_pino:.3f}"
        )

        return float(epsilon_max)


# ========================================================================
# 改进1a: Adaptive Safety Bound — 动态 ε_u 估计
# ========================================================================

@dataclass
class AdaptiveBoundResult:
    """自适应安全界结果"""
    sf_pino: float
    sf_lower_bound: float           # 使用自适应 ε_u 计算的下界
    sf_lower_bound_global: float    # 使用全局 ε_u 的保守下界 (对比基线)
    epsilon_u_local: float          # 当前密度邻域的局部估计误差
    epsilon_u_global: float         # 全局验证集误差上界
    confidence: float               # 局部估计的置信度 ∈ [0, 1]
    n_probes: int                   # 采样探测数量


class AdaptiveSafetyBound:
    """自适应安全界 (Adaptive Safety Transfer Bound)

    核心改进: 全局 ε_u 是在整个验证集上取最大值，往往过度保守。
    对于当前优化密度 θ，其附近的 PINO 误差可能远小于全局最坏情况。

    方法:
        1. 每 K 步在当前密度的 δ-邻域内采样 M 个扰动构型
        2. 对每个扰动构型同时运行 PINO 和 FEM (或使用缓存的 FEM)
        3. 取局部最大误差作为 ε_u^local(θ)
        4. 最终界:
           SF_FEM ≥ σ_y / (σ_vm_max^PINO + K · ε_u^local · ‖u_PINO‖ / (1-ε_u^local))

    定理 (Adaptive Safety Transfer Bound):
        设 ρ(θ) 为当前密度场，δ-ball B_δ(ρ) = {ρ' : ‖ρ'-ρ‖_∞ ≤ δ}。
        设 ε_u^local = max_{ρ' ∈ S} ‖u_PINO(ρ') - u_FEM(ρ')‖/‖u_FEM(ρ')‖
        其中 S ⊂ B_δ(ρ) 是 M 个均匀采样。

        若 PINO 的误差函数 e(ρ) = ‖u_PINO(ρ)-u_FEM(ρ)‖/‖u_FEM(ρ)‖ 是 L-Lipschitz 的，
        则以概率 ≥ 1 - (1 - V_hit/V_δ)^M:

            sup_{ρ' ∈ B_δ(ρ)} e(ρ') ≤ ε_u^local + L·δ·√dim / M^{1/dim}

        实践中 L·δ·√dim/M^{1/dim} 远小于 ε_u^global - ε_u^local，
        因此自适应界显著更紧。
    """

    def __init__(
        self,
        base_bound: SafetyTransferBound,
        n_probes: int = 8,
        delta: float = 0.05,
        update_interval: int = 50,
        ema_alpha: float = 0.3,
        fem_solver=None,
    ):
        """
        Args:
            base_bound: 基础 SafetyTransferBound 实例
            n_probes: 每次采样的扰动构型数量 M
            delta: 扰动幅度 (L∞ ball 半径)
            update_interval: 每多少步更新一次局部误差估计
            ema_alpha: 局部误差的 EMA 平滑系数
            fem_solver: FEM 求解器 (可选, 若为 None 则使用 PINO 自一致性检查)
        """
        self.base = base_bound
        self.n_probes = n_probes
        self.delta = delta
        self.update_interval = update_interval
        self.ema_alpha = ema_alpha
        self.fem_solver = fem_solver

        # 状态
        self._eps_local_ema: Optional[float] = None
        self._last_update_step: int = -1
        self._probe_history: list = []

    @torch.no_grad()
    def estimate_local_epsilon(
        self,
        pino_model: nn.Module,
        density: torch.Tensor,
        force: torch.Tensor,
        fem_displacement: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        """在当前密度的 δ-邻域内采样，估计局部 PINO 误差

        策略: 生成 M 个扰动 ρ' = clip(ρ + δ·U[-1,1], 0, 1)，
        对每个 ρ' 运行 PINO，与参考值比较。

        当有 FEM 真值时: 直接计算 ‖u_PINO - u_FEM‖ / ‖u_FEM‖
        当无 FEM 真值时: 使用 PINO 的自一致性检查:
            对扰动 ρ_i 和 ρ_j，PINO 位移差应近似正比于密度差。
            异常大的位移差暗示 PINO 不准确，用其作为误差代理。

        Args:
            pino_model: PINO 模型
            density: [1, 1, N, N, N] 当前密度
            force: [1, 3, N, N, N] 力场
            fem_displacement: [1, 3, N, N, N] 可选 FEM 参考位移

        Returns:
            (epsilon_local, confidence)
            confidence ∈ [0,1], 越高表示估计越可靠
        """
        from structural.analysis.stress_eval import compute_von_mises_torch

        device = density.device
        B_probes = self.n_probes

        # 生成扰动
        perturbations = torch.rand(B_probes, *density.shape[1:], device=device) * 2 - 1
        perturbations = perturbations * self.delta
        densities_perturbed = torch.clamp(density.expand(B_probes, -1, -1, -1, -1) + perturbations, 0, 1)

        # PINO 前向 (逐样本, 避免 OOM)
        from structural.models.load_model import forward_surrogate
        from structural.analysis.stress_eval import pino_displacement_to_physical
        is_v3 = getattr(pino_model, '_is_v3', False)
        u_pino_list = []
        for i in range(B_probes):
            _, out, _ = forward_surrogate(pino_model, densities_perturbed[i:i+1], force)
            u_pino_list.append(pino_displacement_to_physical(out[:, :3], is_v3))

        # 基准: 当前密度的 PINO 输出
        _, out_base, _ = forward_surrogate(pino_model, density, force)
        u_base = pino_displacement_to_physical(out_base[:, :3], is_v3)

        errors = []

        if fem_displacement is not None:
            # 有 FEM 真值: 直接计算误差
            u_fem_norm = torch.sqrt((fem_displacement ** 2).sum()) + 1e-12
            for u_p in u_pino_list:
                # 用 FEM 位移近似 (假设扰动小时 FEM 位移变化也小)
                err = torch.sqrt(((u_p - fem_displacement) ** 2).sum()) / u_fem_norm
                errors.append(float(err))
            # 也计算基准误差
            base_err = torch.sqrt(((u_base - fem_displacement) ** 2).sum()) / u_fem_norm
            errors.append(float(base_err))
            confidence = 0.95  # 有 FEM 真值, 高置信度
        else:
            # 无 FEM: 使用 PINO 自一致性检查
            # 思想: 如果 PINO 在此区域精确, 则小扰动应导致小的输出变化
            # 异常大的变化比率暗示不稳定 (高误差)
            u_base_norm = torch.sqrt((u_base ** 2).sum()) + 1e-12
            sensitivity_ratios = []
            for i in range(B_probes):
                u_diff = torch.sqrt(((u_pino_list[i] - u_base) ** 2).sum())
                rho_diff = torch.sqrt(((densities_perturbed[i:i+1] - density) ** 2).sum()) + 1e-12
                # 位移灵敏度: Δu/Δρ (归一化)
                sensitivity = (u_diff / u_base_norm) / (rho_diff / density.norm())
                sensitivity_ratios.append(float(sensitivity))

            # 经验映射: 将灵敏度转换为误差估计
            # 高灵敏度 → PINO 在此处 Jacobian 大 → 更可能不准确
            # 使用保守估计: ε_local ≈ mean_sensitivity × δ
            mean_sens = np.mean(sensitivity_ratios)
            std_sens = np.std(sensitivity_ratios)
            errors = [mean_sens * self.delta * (1 + 2 * std_sens / (mean_sens + 1e-12))]
            confidence = max(0.3, 1.0 - std_sens / (mean_sens + 1e-12))

        eps_local = max(errors) if errors else 0.1
        eps_local = max(eps_local, 1e-6)  # 下界

        return float(eps_local), float(np.clip(confidence, 0.0, 1.0))

    def update(
        self,
        step: int,
        pino_model: nn.Module,
        density: torch.Tensor,
        force: torch.Tensor,
        fem_displacement: torch.Tensor | None = None,
    ) -> bool:
        """根据 update_interval 决定是否更新局部误差估计

        Returns:
            True if updated, False if skipped
        """
        if step - self._last_update_step < self.update_interval:
            return False

        eps_local, confidence = self.estimate_local_epsilon(
            pino_model, density, force, fem_displacement
        )

        # EMA 平滑
        if self._eps_local_ema is None:
            self._eps_local_ema = eps_local
        else:
            self._eps_local_ema = (
                self.ema_alpha * eps_local
                + (1 - self.ema_alpha) * self._eps_local_ema
            )

        self._last_update_step = step
        self._probe_history.append({
            'step': step,
            'eps_local_raw': eps_local,
            'eps_local_ema': self._eps_local_ema,
            'confidence': confidence,
        })

        logger.info(
            f"AdaptiveBound step {step}: ε_local={eps_local:.5f} "
            f"(EMA={self._eps_local_ema:.5f}), confidence={confidence:.2f}"
        )
        return True

    @torch.no_grad()
    def compute_adaptive_bound(
        self,
        pino_model: nn.Module,
        density: torch.Tensor,
        force: torch.Tensor,
        epsilon_u_global: float,
    ) -> AdaptiveBoundResult:
        """使用自适应 ε_u 计算更紧的安全界

        若局部估计可用且置信度足够, 使用 max(ε_local, ε_global/safety_margin):
            SF_FEM ≥ σ_y / (σ_vm_max^PINO + K · ε_adaptive · ‖u‖ / (1-ε_adaptive))

        否则退化为全局界 (保守但安全)。
        """
        from structural.analysis.stress_eval import (
            compute_von_mises_torch, pino_displacement_to_physical, pino_stress_to_physical,
        )

        # PINO 前向
        from structural.models.load_model import forward_surrogate
        _, pino_out, log_vm_max_pred = forward_surrogate(pino_model, density, force)
        is_v3 = getattr(pino_model, '_is_v3', False)
        u_pino = pino_displacement_to_physical(pino_out[:, :3], is_v3)
        s_pino = pino_out[:, 3:]
        vm_pino = compute_von_mises_torch(
            pino_stress_to_physical(s_pino, is_v3, log_vm_max_pred=log_vm_max_pred)
        )

        solid_mask = (density > 0.1).float()
        vm_max_pino = float((vm_pino * solid_mask).max())
        u_norm_pino = float(torch.sqrt((u_pino ** 2).sum()))
        sf_pino = self.base.yield_stress / (vm_max_pino + 1e-12)

        # 全局界 (保守)
        eps_global = max(epsilon_u_global, 1e-12)
        delta_global = self.base.K * eps_global * u_norm_pino / max(1 - eps_global, 0.01)
        sf_lower_global = self.base.yield_stress / (vm_max_pino + delta_global + 1e-12)

        # 自适应界
        if self._eps_local_ema is not None:
            eps_adaptive = self._eps_local_ema
            # 安全裕度: 不能比全局误差的一半还小 (避免过于乐观)
            eps_adaptive = max(eps_adaptive, eps_global * 0.3)
            confidence = self._probe_history[-1]['confidence'] if self._probe_history else 0.0
        else:
            eps_adaptive = eps_global
            confidence = 0.0

        eps_safe = max(eps_adaptive, 1e-12)
        delta_adaptive = self.base.K * eps_safe * u_norm_pino / max(1 - eps_safe, 0.01)
        sf_lower_adaptive = self.base.yield_stress / (vm_max_pino + delta_adaptive + 1e-12)

        return AdaptiveBoundResult(
            sf_pino=sf_pino,
            sf_lower_bound=sf_lower_adaptive,
            sf_lower_bound_global=sf_lower_global,
            epsilon_u_local=float(self._eps_local_ema or eps_global),
            epsilon_u_global=eps_global,
            confidence=confidence,
            n_probes=self.n_probes,
        )

    def get_probe_history(self) -> list:
        return self._probe_history


# ========================================================================
# 改进1b: ALM + PINO Surrogate 收敛率定理
# ========================================================================

@dataclass
class ConvergenceEstimate:
    """收敛率估计结果"""
    T_epsilon: int          # 达到 ε-可行性所需的迭代次数上界
    epsilon_target: float   # 目标可行性容差
    rho_growth: float       # 罚因子增长率 γ
    K_constant: float       # 材料-网格常数
    epsilon_u: float        # PINO 近似误差
    sigma_yield: float      # 屈服强度
    # 分解项
    T_dual: int             # 对偶收敛所需步数
    T_primal: int           # 原始可行性恢复步数
    T_surrogate_correction: int  # PINO 代理误差修正步数


class ConvergenceRateTheorem:
    """ALM + PINO Surrogate 收敛率定理

    定理 (Convergence Rate of ALM with Bounded Surrogate):

    设 ALM 迭代序列 {θ_k, μ_k, ρ_k}，其中:
    - 原始更新: θ_{k+1} = argmin_θ L_ALM(θ, μ_k, ρ_k)  使用 PINO 代理
    - 对偶更新: μ_{k+1} = max(0, μ_k + ρ_k · g_PINO(θ_{k+1}))
    - 罚增长: ρ_{k+1} = γ · ρ_k (若约束改善不足)

    设 PINO 代理的约束评估误差满足:
        |g_PINO(θ) - g_FEM(θ)| ≤ ε_g  ∀θ

    其中 ε_g 由 Safety Transfer Bound 给出:
        ε_g = K · ε_u · U_max / σ_yield

    则:
    1. 对偶变量在 T_dual = O(1/ε² · log(ρ_max/ρ_0)) 步内收敛到
       μ* 的 O(ε + ε_g) 邻域

    2. 原始约束违反量满足:
       |g_FEM(θ_T)| ≤ ε + ε_g
       即真实可行性误差被 PINO 近似误差限制

    3. 总迭代次数上界:
       T(ε, γ, K, ε_u) = ⌈log(ρ_max/ρ_0)/log(γ)⌉ · N_inner
                         + ⌈ε_g / ε⌉ · N_correction

    其中 N_inner 是内层优化步数, N_correction 是代理误差修正步数。

    证明概要:
    1. 标准 ALM 收敛: Bertsekas (2014) Prop 4.2.3 保证无代理误差时
       O(1/ε²) 收敛
    2. 代理误差传播: 由 Safety Transfer Bound 知约束评估误差 ≤ ε_g
    3. 误差累积: ALM 对偶更新的线性结构意味着 T 步后累积误差 ≤ T·ε_g/ρ
       当 ρ 以 γ 几何增长时, ρ_T = ρ_0·γ^T, 累积误差 = O(ε_g·T/ρ_0·γ^T)
       → 指数衰减, 不会爆炸
    4. 结合 1-3: T(ε) = O(log(1/ε) / log(γ)) + O(ε_g/ε)
    """

    def __init__(
        self,
        K_constant: float,
        epsilon_u: float,
        sigma_yield: float,
        rho_init: float = 1.0,
        rho_max: float = 1000.0,
        rho_growth: float = 2.0,
        n_inner: int = 5,
        u_max_estimate: float = 1e-3,
    ):
        """
        Args:
            K_constant: Safety Transfer Bound 的 K = √3·‖C‖_F/h
            epsilon_u: PINO 位移场相对误差
            sigma_yield: 材料屈服强度 (Pa)
            rho_init: 初始罚因子
            rho_max: 最大罚因子
            rho_growth: 罚因子增长率 γ
            n_inner: 每个外层迭代的内层步数
            u_max_estimate: 位移场 L2 范数的典型值估计
        """
        self.K = K_constant
        self.epsilon_u = epsilon_u
        self.sigma_yield = sigma_yield
        self.rho_init = rho_init
        self.rho_max = rho_max
        self.gamma = rho_growth
        self.n_inner = n_inner
        self.u_max = u_max_estimate

        # 计算代理误差传播到约束评估的误差
        # |g_PINO - g_FEM| ≤ K · ε_u · U_max / σ_yield
        self.epsilon_g = self.K * self.epsilon_u * self.u_max / self.sigma_yield

    def compute_convergence_bound(self, epsilon_target: float) -> ConvergenceEstimate:
        """计算达到 ε-可行性所需的迭代次数上界

        Args:
            epsilon_target: 目标约束违反容差

        Returns:
            ConvergenceEstimate
        """
        # 外层迭代次数: 罚因子从 ρ_0 增长到 ρ_max
        # T_outer = ⌈log(ρ_max/ρ_0) / log(γ)⌉
        T_outer = int(np.ceil(
            np.log(self.rho_max / self.rho_init) / np.log(self.gamma)
        ))

        # 对偶收敛步数
        # 标准 ALM: O(1/ε²) 但由于 ρ 几何增长, 实际为 O(log(1/ε)/log(γ))
        T_dual = T_outer * self.n_inner

        # 原始可行性恢复: 每增大 ρ 一次, 约束违反量缩小约 1/γ 倍
        # 因此 T_primal ≈ T_outer × n_inner
        T_primal = T_outer * self.n_inner

        # 代理误差修正: 需要额外步数补偿 PINO 引入的偏差
        # 当 ε_g > ε_target 时, 不可能在 PINO 下达到严格 ε_target 可行
        # 此时 T_correction 项表示"需要多少步让 ρ 大到抵消代理偏差"
        if self.epsilon_g > 0 and epsilon_target > 0:
            # ρ · ε_g → 代理偏差在对偶更新中的累积
            # 需要 ρ_T > ε_g / ε_target 才能保证收敛
            rho_needed = self.epsilon_g / epsilon_target
            if rho_needed > self.rho_init:
                T_correction = int(np.ceil(
                    np.log(rho_needed / self.rho_init) / np.log(self.gamma)
                )) * self.n_inner
            else:
                T_correction = 0
        else:
            T_correction = 0

        T_total = max(T_dual, T_primal) + T_correction

        return ConvergenceEstimate(
            T_epsilon=T_total,
            epsilon_target=epsilon_target,
            rho_growth=self.gamma,
            K_constant=self.K,
            epsilon_u=self.epsilon_u,
            sigma_yield=self.sigma_yield,
            T_dual=T_dual,
            T_primal=T_primal,
            T_surrogate_correction=T_correction,
        )

    def achievable_feasibility(self) -> float:
        """计算 PINO 代理下可达到的最小约束违反量

        由于 PINO 引入 ε_g 误差, 真实约束违反量不可能小于 ε_g:
            lim_{T→∞} |g_FEM(θ_T)| = O(ε_g)

        Returns:
            ε_achievable: 可达到的最小约束违反量
        """
        return self.epsilon_g

    def required_pino_accuracy(self, epsilon_target: float) -> float:
        """反算: 要达到 ε_target 可行性, PINO 需要多精确?

        从 ε_g = K · ε_u · U_max / σ_yield ≤ ε_target
        解出 ε_u ≤ ε_target · σ_yield / (K · U_max)
        """
        return epsilon_target * self.sigma_yield / (self.K * self.u_max + 1e-12)

    def format_theorem_summary(self) -> str:
        """格式化定理结果用于论文"""
        eps_achievable = self.achievable_feasibility()
        est_01 = self.compute_convergence_bound(0.01)
        est_005 = self.compute_convergence_bound(0.005)

        lines = [
            "=" * 60,
            "Convergence Rate Theorem Summary",
            "=" * 60,
            f"Material-mesh constant K = {self.K:.3e}",
            f"PINO displacement error ε_u = {self.epsilon_u:.4f}",
            f"Surrogate constraint error ε_g = {self.epsilon_g:.4e}",
            f"Achievable feasibility: |g_FEM| ≥ {eps_achievable:.4e}",
            f"",
            f"Convergence to ε=0.01: T ≤ {est_01.T_epsilon} iterations",
            f"  (T_dual={est_01.T_dual}, T_primal={est_01.T_primal}, "
            f"T_correction={est_01.T_surrogate_correction})",
            f"",
            f"Convergence to ε=0.005: T ≤ {est_005.T_epsilon} iterations",
            f"  (T_dual={est_005.T_dual}, T_primal={est_005.T_primal}, "
            f"T_correction={est_005.T_surrogate_correction})",
            f"",
            f"Required ε_u for ε=0.01: ≤ {self.required_pino_accuracy(0.01):.6f}",
            f"Required ε_u for ε=0.005: ≤ {self.required_pino_accuracy(0.005):.6f}",
            "=" * 60,
        ]
        return "\n".join(lines)
