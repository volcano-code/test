"""
消融实验 (Ablation Study)

5 个消融变体, 逐一移除方法论贡献以验证各组件效果:

Ablation 1: w/o ALM (移除增广拉格朗日法, 退化为加权求和)
Ablation 2: w/o Gradient Router (移除自适应梯度路由, 使用统一梯度)
Ablation 3: w/o Safety Bound (移除安全传递界, 不做安全保证验证)
Ablation 4: w/o RepairCNN (移除修复网络, 仅用 log_densities 优化)
Ablation 5: w/o Inner-Outer Decoupling (移除内外解耦, 不区分表面/内部)

每个消融变体修改 Stage C 的优化流程, 保持 Stage A/B 不变。
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Callable

import numpy as np
import torch
import torch.nn as nn

from structural.analysis.stress_eval import (
    get_yield_stress_for_model,
    structural_loss, volume_loss,
    shape_preservation_loss, smoothness_loss, overhang_penalty,
)
from structural.models.load_model import forward_surrogate_physical
from structural.optimization.alm_optimizer import ALMOptimizer, ALMConfig
from structural.optimization.gradient_router import AdaptiveGradientRouter

logger = logging.getLogger(__name__)


def _run_optimization_loop(
    density_original: np.ndarray,
    force: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float,
    device: torch.device,
    n_steps: int,
    lr: float,
    use_alm: bool = True,
    use_router: bool = True,
    use_repair: bool = True,
    use_decoupling: bool = True,
    use_conservative_calibration: bool = True,
    stress_inflation_factor: float = 1.15,
    alm_config: ALMConfig | None = None,
) -> np.ndarray:
    """通用优化循环, 通过标志控制各组件的开关

    Args:
        density_original: [N, N, N] 初始密度
        force: [3, N, N, N] 力场
        pino_model: 预训练 PINO
        yield_stress: 屈服强度
        device: GPU 设备
        n_steps: 优化步数
        lr: 学习率
        use_alm: 是否使用 ALM 硬约束 (False → 加权求和)
        use_router: 是否使用自适应梯度路由 (False → 统一梯度)
        use_repair: 是否使用 RepairCNN (False → 仅 log_densities)
        use_decoupling: 是否区分内外 (False → 不做内外分离)
        alm_config: ALM 配置

    Returns:
        [N, N, N] 优化后密度
    """
    if alm_config is None:
        alm_config = ALMConfig()

    # 初始化密度参数
    density_t = torch.from_numpy(density_original).float().to(device)
    density_t = torch.clamp(density_t, 1e-6, 1 - 1e-6)
    temperature = 15.0
    log_density = torch.nn.Parameter(
        torch.log(density_t / (1 - density_t)) / temperature
    )

    density_orig_t = torch.from_numpy(density_original).float().unsqueeze(0).unsqueeze(0).to(device)
    force_t = torch.from_numpy(force).float().unsqueeze(0).to(device)
    yield_eval = get_yield_stress_for_model(pino_model, yield_stress)

    # 可选: RepairCNN
    repair_cnn = None
    params = [{'params': [log_density], 'lr': lr}]
    if use_repair:
        from structural.models.repair_cnn import RepairCNN
        repair_cnn = RepairCNN(
            in_channels=3,
            channels=[16, 32, 32],
            delta_max=0.3,
        ).to(device)
        params.append({'params': repair_cnn.parameters(), 'lr': lr * 0.5})

    optimizer = torch.optim.Adam(params)

    # 可选: ALM
    alm = ALMOptimizer(alm_config) if use_alm else None

    # 可选: 梯度路由器
    grad_router = None
    if use_router:
        grad_router = AdaptiveGradientRouter(
            sf_target=alm_config.safety_factor_target,
            warmup_steps=20,
        )

    # 加权求和 fallback 权重 (当 use_alm=False 时使用)
    lambda_struct = 2.0
    lambda_vol = 1.0
    lambda_shape = 5.0

    for step in range(n_steps):
        density_current = torch.sigmoid(temperature * log_density).unsqueeze(0).unsqueeze(0)

        # 可选: RepairCNN 加固
        if use_repair and repair_cnn is not None:
            with torch.no_grad():
                pino_pre = forward_surrogate_physical(pino_model, density_current, force_t)
                vm_pre = pino_pre['von_mises']
                vm_norm = vm_pre / (vm_pre.max() + 1e-8)
                vuln_map = torch.zeros_like(vm_norm)  # 简化: 用零图代替漏洞检测

            repair_input = torch.cat([density_current, vm_norm, vuln_map], dim=1)
            repair_fields = repair_cnn(repair_input)

            if use_decoupling:
                solid = (density_current > 0.5).float()
                interior = -torch.nn.functional.max_pool3d(
                    -solid, kernel_size=5, stride=1, padding=2
                )
            else:
                interior = torch.ones_like(density_current)

            density_current = repair_cnn.apply_repair(
                density_current, repair_fields, interior
            )

        # PINO 分析
        pino_fields = forward_surrogate_physical(pino_model, density_current, force_t)
        vm = pino_fields['von_mises']
        if use_conservative_calibration:
            vm = vm * stress_inflation_factor

        # 正则项
        l_shape = shape_preservation_loss(density_current, density_orig_t)
        l_smooth = smoothness_loss(density_current)

        if use_alm and alm is not None:
            # ALM 模式: 硬约束
            violations = alm.compute_constraint_violations(
                von_mises=vm,
                density=density_current,
                density_original=density_orig_t,
                yield_stress=yield_eval,
            )

            # 用外形保持作为 appearance 代理
            alm_result = alm.compute_alm_loss(
                appearance_loss=l_shape,
                violations=violations,
                smoothness_loss_val=l_smooth,
            )
            total = alm_result['total']

            alm.update_violations(violations)
            alm.step(step)
        else:
            # 加权求和 fallback
            l_struct = structural_loss(vm, density_current, yield_eval)
            l_vol = volume_loss(density_current, density_orig_t, 1.3)
            total = (lambda_struct * l_struct + lambda_vol * l_vol
                     + lambda_shape * l_shape + l_smooth)
            violations = {}

        optimizer.zero_grad()
        total.backward()

        # 可选: 梯度路由
        if use_router and grad_router is not None:
            with torch.no_grad():
                solid_mask = (density_current > 0.1).float()
                vm_max = float((vm * solid_mask).max())
                sf_current = yield_eval / (vm_max + 1e-12)
                n_violated = sum(
                    1 for v in violations.values()
                    if float(v.detach()) > 0.01
                ) if violations else 0

            # 构造临时 model-like 对象用于路由
            class _DummyModel:
                pass
            dummy = _DummyModel()
            dummy.log_densities = log_density
            dummy.temperature = temperature

            if repair_cnn is not None:
                grad_router.route(
                    model=dummy,
                    repair_cnn=repair_cnn,
                    sf_current=sf_current,
                    sil_loss=float(l_shape.detach()),
                    n_constraints_violated=n_violated,
                    step=step,
                )

        torch.nn.utils.clip_grad_norm_([log_density], 1.0)
        if repair_cnn is not None:
            torch.nn.utils.clip_grad_norm_(repair_cnn.parameters(), 1.0)
        optimizer.step()

    with torch.no_grad():
        final = torch.sigmoid(temperature * log_density).cpu().numpy()

    return final


def run_ablation_no_alm(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    n_steps: int = 200,
    lr: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """Ablation 1: w/o ALM — 退化为加权求和

    移除增广拉格朗日法, 用固定 λ 权重的多目标加权求和代替。
    保留 Gradient Router 和 RepairCNN。
    预期: 约束满足率显著下降 (无硬约束保证)。
    """
    if device is None:
        device = next(pino_model.parameters()).device
    return _run_optimization_loop(
        density_original, force, pino_model, yield_stress, device,
        n_steps=n_steps, lr=lr,
        use_alm=False,
        use_router=True,
        use_repair=True,
        use_decoupling=True,
    )


def run_ablation_no_router(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    n_steps: int = 200,
    lr: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """Ablation 2: w/o Gradient Router — 使用统一梯度

    移除自适应梯度路由, 所有参数接收未修改的梯度。
    保留 ALM 和 RepairCNN。
    预期: IoU 和 SF 的帕累托前沿退化 (内外冲突)。
    """
    if device is None:
        device = next(pino_model.parameters()).device
    return _run_optimization_loop(
        density_original, force, pino_model, yield_stress, device,
        n_steps=n_steps, lr=lr,
        use_alm=True,
        use_router=False,
        use_repair=True,
        use_decoupling=True,
    )


def run_ablation_no_bound(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    n_steps: int = 200,
    lr: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """Ablation 3: w/o Safety Bound — 不做安全保证验证

    Safety Transfer Bound 不参与优化循环 (仅事后评估)。
    保留 ALM 和 Gradient Router。
    预期: 优化结果本身不变 (bound 是验证工具),
          但 FEM 验证时可能出现 SF_FEM < SF_PINO 的情况。

    注意: 这个消融主要通过实验结果表中的 bound_validity 列体现,
          优化过程与 full method 完全相同。
    """
    if device is None:
        device = next(pino_model.parameters()).device
    return _run_optimization_loop(
        density_original, force, pino_model, yield_stress, device,
        n_steps=n_steps, lr=lr,
        use_alm=True,
        use_router=True,
        use_repair=True,
        use_decoupling=True,
    )


def run_ablation_no_repair(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    n_steps: int = 200,
    lr: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """Ablation 4: w/o RepairCNN — 仅用 log_densities 优化

    移除 RepairCNN, 直接在体素密度场上做优化。
    保留 ALM 和 Gradient Router。
    预期: 结构加固能力下降 (缺少局部修复先验)。
    """
    if device is None:
        device = next(pino_model.parameters()).device
    return _run_optimization_loop(
        density_original, force, pino_model, yield_stress, device,
        n_steps=n_steps, lr=lr,
        use_alm=True,
        use_router=True,
        use_repair=False,
        use_decoupling=True,
    )


def run_ablation_no_decoupling(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    n_steps: int = 200,
    lr: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """Ablation 5: w/o Inner-Outer Decoupling — 不区分表面/内部

    RepairCNN 的修复应用到整个密度场, 而非仅内部。
    保留 ALM 和 Gradient Router。
    预期: 外形 IoU 下降 (修复操作扰动外表面)。
    """
    if device is None:
        device = next(pino_model.parameters()).device
    return _run_optimization_loop(
        density_original, force, pino_model, yield_stress, device,
        n_steps=n_steps, lr=lr,
        use_alm=True,
        use_router=True,
        use_repair=True,
        use_decoupling=False,
    )


def run_ablation_no_calibration(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    n_steps: int = 200,
    lr: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """移除保守 surrogate 校准."""
    if device is None:
        device = next(pino_model.parameters()).device
    return _run_optimization_loop(
        density_original, force, pino_model, yield_stress, device,
        n_steps=n_steps, lr=lr,
        use_alm=True,
        use_router=True,
        use_repair=True,
        use_decoupling=True,
        use_conservative_calibration=False,
    )


# 消融注册表
ABLATIONS = {
    'Ours w/o ALM': run_ablation_no_alm,
    'Ours w/o Router': run_ablation_no_router,
    'Ours w/o Bound': run_ablation_no_bound,
    'Ours w/o RepairCNN': run_ablation_no_repair,
    'Ours w/o Decoupling': run_ablation_no_decoupling,
    'Ours w/o Calibration': run_ablation_no_calibration,
}
