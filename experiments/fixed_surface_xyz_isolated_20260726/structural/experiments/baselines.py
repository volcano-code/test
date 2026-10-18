"""
7 种基线方法实现

Baseline 1: ShadowArt (无结构优化)
    仅做剪影重建, 不考虑结构安全

Baseline 2: ShadowArt + Stava 后处理
    先重建, 再用启发式规则加固 (Stava 2012 风格)

Baseline 3: ShadowArt + 加权求和 (旧 Stage C)
    用固定 λ 权重的多目标加权求和优化

Baseline 4: ShadowArt + 真实 FEM 优化 (计算上界)
    每步用真实 FEM 替代 PINO, 作为性能上界参考

Baseline 5: SIMP 拓扑优化
    从 ShadowArt 结果出发做标准 SIMP

Baseline 6: NTOPo (Neural Topology Optimization, Zehnder+ TOG 2021)
    用 MLP 参数化密度场, 替代显式体素优化
    对比意义: 神经参数化 vs 我们的 logit + RepairCNN 参数化

Baseline 7: TopoDiff 风格 (Mazé & Ahmed, AAAI 2023)
    扩散模型引导的拓扑优化 — 简化版: 先用去噪 score
    估计密度梯度方向, 再做投影步
    对比意义: 生成模型方法 vs 我们的约束优化方法
"""

from __future__ import annotations

import time
import logging

import numpy as np
import torch

from structural.analysis.stress_eval import get_yield_stress_for_model
from structural.models.load_model import forward_surrogate_physical

logger = logging.getLogger(__name__)


def run_baseline_shadowart_only(
    density_original: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Baseline 1: 纯 ShadowArt, 不做任何结构修改

    返回原始重建结果 (上界为 IoU, 下界为 SF)
    """
    return density_original.copy()


def run_baseline_stava_postprocess(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    thicken_iterations: int = 2,
    **kwargs,
) -> np.ndarray:
    """Baseline 2: ShadowArt + Stava 2012 风格后处理

    策略:
    1. 用 PINO 找到高应力区域
    2. 对高应力区域做形态学膨胀 (加厚)
    3. 添加底部支撑
    不修改外部轮廓的理念, 只在内部/薄弱处加固
    """
    from structural.analysis.stress_eval import compute_von_mises_torch
    from scipy.ndimage import binary_dilation, generate_binary_structure

    density = density_original.copy().astype(np.float32)

    if device is None:
        device = next(pino_model.parameters()).device
    yield_eval = get_yield_stress_for_model(pino_model, yield_stress)

    # PINO 应力分析
    with torch.no_grad():
        rho = torch.from_numpy(density).float().unsqueeze(0).unsqueeze(0).to(device)
        f = torch.from_numpy(force).float().unsqueeze(0).to(device)
        pino_fields = forward_surrogate_physical(pino_model, rho, f)
        vm = compute_von_mises_torch(pino_fields['stress'])
        vm_np = vm[0, 0].cpu().numpy()

    solid = density > 0.5

    # 找高应力区域 (超过屈服强度 80% 的区域)
    threshold = yield_eval * 0.8
    high_stress = (vm_np > threshold) & solid

    if high_stress.sum() > 0:
        # 策略1: 对高应力区域做膨胀 (加厚)
        struct = generate_binary_structure(3, 2)  # 18-connectivity
        thickened = high_stress.copy()
        for _ in range(thicken_iterations):
            thickened = binary_dilation(thickened, structure=struct)
        density = np.maximum(density, thickened.astype(np.float32))

    # 策略2: 底部加固 (增加3层底座)
    if solid.any():
        # 找到最低实体层
        for y in range(density.shape[1]):
            if solid[:, y, :].any():
                for dy in range(max(0, y - 2), y + 1):
                    # 将底部3层填充
                    base_layer = solid[:, y, :]
                    density[:, dy, :] = np.maximum(
                        density[:, dy, :],
                        binary_dilation(base_layer, iterations=1).astype(np.float32)
                    )
                break

    return np.clip(density, 0, 1)


def run_baseline_weighted_sum(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    n_steps: int = 200,
    lr: float = 1e-2,
    lambda_struct: float = 2.0,
    lambda_vol: float = 1.0,
    lambda_shape: float = 5.0,
    **kwargs,
) -> np.ndarray:
    """Baseline 3: 加权求和优化 (旧方法, 无ALM)

    直接用 λ 权重组合所有损失, 与 StructShadow ALM 方法对照。
    """
    from structural.analysis.stress_eval import (
        structural_loss, volume_loss,
        shape_preservation_loss, smoothness_loss,
    )

    if device is None:
        device = next(pino_model.parameters()).device
    yield_eval = get_yield_stress_for_model(pino_model, yield_stress)

    density_orig_t = torch.from_numpy(density_original).float().unsqueeze(0).unsqueeze(0).to(device)
    force_t = torch.from_numpy(force).float().unsqueeze(0).to(device)
    sweep = [
        {'lambda_struct': lambda_struct, 'lambda_vol': lambda_vol, 'lambda_shape': lambda_shape},
        {'lambda_struct': 4.0, 'lambda_vol': 2.0, 'lambda_shape': 4.0},
        {'lambda_struct': 6.0, 'lambda_vol': 2.5, 'lambda_shape': 3.0},
    ]

    best_score = float('-inf')
    best_final = density_original.copy()

    for cfg in sweep:
        density_t = torch.from_numpy(density_original).float().to(device)
        density_t = torch.clamp(density_t, 1e-6, 1 - 1e-6)
        log_density = torch.nn.Parameter(torch.log(density_t / (1 - density_t)))
        optimizer = torch.optim.Adam([log_density], lr=lr)
        temperature = 15.0

        for _ in range(n_steps):
            density_current = torch.sigmoid(temperature * log_density).unsqueeze(0).unsqueeze(0)
            pino_fields = forward_surrogate_physical(pino_model, density_current, force_t)
            vm = pino_fields['von_mises']

            l_struct = structural_loss(vm, density_current, yield_eval)
            l_vol = volume_loss(density_current, density_orig_t, 1.3)
            l_shape = shape_preservation_loss(density_current, density_orig_t)
            l_smooth = smoothness_loss(density_current)

            total = (
                cfg['lambda_struct'] * l_struct
                + cfg['lambda_vol'] * l_vol
                + cfg['lambda_shape'] * l_shape
                + l_smooth
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_([log_density], 1.0)
            optimizer.step()

        with torch.no_grad():
            final = torch.sigmoid(temperature * log_density).cpu().numpy()
            rho = torch.from_numpy(final).float().unsqueeze(0).unsqueeze(0).to(device)
            pino_fields = forward_surrogate_physical(pino_model, rho, force_t)
            vm = pino_fields['von_mises']
            vm_max = float((vm * (rho > 0.1).float()).max())
            sf = yield_eval / (vm_max + 1e-12)
            vol_ratio = float((final > 0.5).sum() / max((density_original > 0.5).sum(), 1))
            shape_iou = float((((final > 0.5) & (density_original > 0.5)).sum()) /
                              max((((final > 0.5) | (density_original > 0.5)).sum()), 1))
            score = 2.5 * min(sf, 1.5) + shape_iou - 0.2 * max(vol_ratio - 1.3, 0.0)
            if score > best_score:
                best_score = score
                best_final = final

    return best_final


def run_baseline_posthoc_repair(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    **kwargs,
) -> np.ndarray:
    """更强的后处理基线: 先启发式加固，再做轻量结构修复."""
    from structural.experiments.ablation import _run_optimization_loop

    if device is None:
        device = next(pino_model.parameters()).device

    stava_density = run_baseline_stava_postprocess(
        density_original=density_original,
        force=force,
        fixed_mask=fixed_mask,
        pino_model=pino_model,
        yield_stress=yield_stress,
        device=device,
    )
    return _run_optimization_loop(
        stava_density,
        force,
        pino_model,
        yield_stress,
        device,
        n_steps=80,
        lr=5e-3,
        use_alm=False,
        use_router=False,
        use_repair=True,
        use_decoupling=True,
    )


def run_baseline_fem_optimization(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    yield_stress: float = 50.0e6,
    n_steps: int = 50,
    lr: float = 0.05,
    **kwargs,
) -> np.ndarray:
    """Baseline 4: 真实 FEM 优化 (计算上界)

    每步用真实 FEM 计算应力, 再用有限差分估计梯度。
    非常慢 (50步可能需要几小时), 但作为性能参考上界。

    使用 SIMP 风格的灵敏度分析:
    - 目标: min compliance (最大刚度)
    - 约束: volume ≤ budget
    """
    from structural.data.generate_fem_dataset import density_to_fem_fields

    try:
        from sgldbench.config import MaterialProperties, SolverSettings
    except ImportError:
        logger.warning("SGLDBench not available, returning original density")
        return density_original.copy()

    material = MaterialProperties(E=3.5e9, nu=0.36, cell_size=1.0)
    solver = SolverSettings(tolerance=1e-5, max_iterations=500)

    density = density_original.copy().astype(np.float32)
    solid_orig = density_original > 0.5
    budget = solid_orig.sum() * 1.3

    for step in range(n_steps):
        # FEM 求解
        result = density_to_fem_fields(
            (density > 0.5).astype(bool),
            force, fixed_mask,
            material, solver,
        )

        if result is None:
            logger.warning(f"FEM failed at step {step}")
            break

        vm = result['von_mises']
        solid = density > 0.5

        # SIMP 灵敏度: 高应力区域增大密度
        vm_norm = vm / (vm.max() + 1e-12)
        sensitivity = vm_norm * solid

        # 梯度步: 加固高应力, 削减低应力
        threshold = yield_stress * 0.8
        add_mask = (vm > threshold) & solid
        remove_mask = (vm < threshold * 0.2) & solid

        density[add_mask] = np.minimum(density[add_mask] + lr, 1.0)
        density[remove_mask] = np.maximum(density[remove_mask] - lr * 0.5, 0.0)

        # 体积约束投影
        if (density > 0.5).sum() > budget:
            sorted_vals = np.sort(density[density > 0.01])
            cutoff_idx = len(sorted_vals) - int(budget)
            if cutoff_idx > 0 and cutoff_idx < len(sorted_vals):
                cutoff = sorted_vals[cutoff_idx]
                density[density < cutoff] = 0.0

        if step % 10 == 0:
            vm_max = vm[solid].max() if solid.any() else 0
            sf = yield_stress / (vm_max + 1e-12)
            logger.info(f"  FEM-opt step {step}: SF={sf:.2f}, vol_ratio={solid.sum() / solid_orig.sum():.2f}")

    return np.clip(density, 0, 1)


def run_baseline_simp(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    n_steps: int = 200,
    lr: float = 0.02,
    simp_penalty: float = 3.0,
    **kwargs,
) -> np.ndarray:
    """Baseline 5: SIMP 拓扑优化 (用 PINO 做前向)

    经典 SIMP: min compliance s.t. volume constraint
    用 PINO 替代 FEM 做前向分析, 但优化方式是标准 SIMP。
    不使用 RepairCNN, 不区分内外。
    """
    if device is None:
        device = next(pino_model.parameters()).device

    # 初始化: 从原始形状出发
    density_t = torch.from_numpy(density_original).float().to(device)
    density_t = torch.clamp(density_t, 0.01, 0.99)
    x = torch.nn.Parameter(density_t)

    force_t = torch.from_numpy(force).float().unsqueeze(0).to(device)
    orig_vol = (density_original > 0.5).sum()
    budget = orig_vol * 1.3

    optimizer = torch.optim.Adam([x], lr=lr)

    for step in range(n_steps):
        # SIMP 密度惩罚
        rho = torch.clamp(x, 0.0, 1.0).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N, N]
        rho_penalized = rho ** simp_penalty

        # PINO 分析
        pino_fields = forward_surrogate_physical(pino_model, rho, force_t)
        u = pino_fields['displacement']

        # 目标: 最小化柔顺度 (compliance = u^T f)
        compliance = (u * force_t).sum()

        # 体积约束惩罚
        vol = rho.sum()
        vol_penalty = torch.relu(vol / budget - 1.0) ** 2

        loss = compliance + 100.0 * vol_penalty

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x], 1.0)
        optimizer.step()

        # 投影到 [0, 1]
        with torch.no_grad():
            x.data.clamp_(0.0, 1.0)

    with torch.no_grad():
        final = torch.clamp(x, 0.0, 1.0).cpu().numpy()

    return final


def run_baseline_ntopo(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    n_steps: int = 200,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    n_hidden: int = 4,
    **kwargs,
) -> np.ndarray:
    """Baseline 6: NTOPo — 神经拓扑优化 (Zehnder+ TOG 2021 简化版)

    核心思想: 用 MLP φ(x,y,z; θ) → ρ 参数化密度场, 替代显式体素变量。
    MLP 的平滑归纳偏置自然产生连通结构, 但无法精确控制约束。

    优化目标与 WeightedSum 相同 (公平对比), 仅参数化方式不同。

    对比意义:
    - NTOPo 的隐式正则化 vs 我们的 RepairCNN 显式修复
    - NTOPo 无硬约束 vs 我们的 ALM 硬约束
    """
    from structural.analysis.stress_eval import (
        structural_loss, volume_loss,
        shape_preservation_loss, smoothness_loss,
    )

    if device is None:
        device = next(pino_model.parameters()).device

    N = density_original.shape[0]

    # ---- NTOPo MLP 定义 ----
    class DensityMLP(torch.nn.Module):
        """坐标 → 密度的 MLP (NTOPo 核心)"""
        def __init__(self, in_dim=3, out_dim=1, hidden=64, n_layers=4):
            super().__init__()
            layers = [torch.nn.Linear(in_dim, hidden), torch.nn.ReLU()]
            for _ in range(n_layers - 1):
                layers += [torch.nn.Linear(hidden, hidden), torch.nn.ReLU()]
            layers.append(torch.nn.Linear(hidden, out_dim))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, coords):
            return torch.sigmoid(self.net(coords))

    # 生成坐标网格
    coords_1d = torch.linspace(0, 1, N, device=device)
    gx, gy, gz = torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)  # [N³, 3]

    # 初始化 MLP (用原始密度做预训练热启动)
    mlp = DensityMLP(hidden=hidden_dim, n_layers=n_hidden).to(device)
    target_rho = torch.from_numpy(density_original).float().to(device).reshape(-1, 1)

    # 预训练: 让 MLP 拟合原始形状
    # 使用加权 BCE: 稀疏密度场 (~5% solid) 下普通 BCE 会收敛到全零
    solid_frac = float(target_rho.mean())
    pos_weight = torch.tensor([(1.0 - solid_frac) / max(solid_frac, 0.01)], device=device)
    pretrain_opt = torch.optim.Adam(mlp.parameters(), lr=5e-3)
    for _ in range(500):  # 更多步数确保收敛
        pred = mlp(coords)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.logit(pred.clamp(1e-6, 1-1e-6)), target_rho, pos_weight=pos_weight
        )
        pretrain_opt.zero_grad()
        loss.backward()
        pretrain_opt.step()

    # 主优化
    density_orig_t = torch.from_numpy(density_original).float().unsqueeze(0).unsqueeze(0).to(device)
    force_t = torch.from_numpy(force).float().unsqueeze(0).to(device)

    # 冻结 PINO 参数, 只优化 MLP
    for p in pino_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

    for step in range(n_steps):
        rho_flat = mlp(coords)  # [N³, 1]
        rho = rho_flat.reshape(1, 1, N, N, N)

        # PINO 分析
        pino_fields = forward_surrogate_physical(pino_model, rho, force_t)
        vm = pino_fields['von_mises']

        # 与 WeightedSum 相同的损失 (公平对比)
        l_struct = structural_loss(vm, rho, yield_stress)
        l_vol = volume_loss(rho, density_orig_t, 1.3)
        l_shape = shape_preservation_loss(rho, density_orig_t)
        l_smooth = smoothness_loss(rho)

        # 防止密度坍缩: 保持总实体体素量不低于原始的 70%
        v_orig = density_orig_t.sum()
        v_cur = rho.sum()
        l_preserve = torch.relu(0.7 - v_cur / (v_orig + 1e-8)) ** 2

        total = 2.0 * l_struct + 1.0 * l_vol + 5.0 * l_shape + l_smooth + 10.0 * l_preserve

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        optimizer.step()

    # 恢复 PINO 参数的 requires_grad 状态
    for p in pino_model.parameters():
        p.requires_grad_(True)

    with torch.no_grad():
        final = mlp(coords).reshape(N, N, N).cpu().numpy()

    return final


def run_baseline_topodiff(
    density_original: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    n_steps: int = 200,
    lr: float = 1e-2,
    noise_scale: float = 0.1,
    denoise_steps: int = 5,
    **kwargs,
) -> np.ndarray:
    """Baseline 7: TopoDiff 风格 — 扩散引导拓扑优化 (Mazé & Ahmed 2023 简化版)

    核心思想: 将拓扑优化视为条件去噪过程。
    每步: (1) 给当前密度加噪 → (2) 用 PINO 梯度引导去噪方向 → (3) 投影。

    简化版不训练扩散模型 (需要大量训练数据), 而是用:
    - 加噪: x_t = √(1-β)·x_{t-1} + √β·ε
    - 去噪方向: 结构梯度 (PINO) + 形状梯度 (IoU)
    - 类似 score-based guidance in optimization

    对比意义:
    - 生成模型的随机探索 vs 我们的确定性 ALM 约束优化
    - TopoDiff 天然多样性但约束满足率低 vs 我们的硬约束保证
    """
    from structural.analysis.stress_eval import (
        structural_loss,
        shape_preservation_loss, smoothness_loss,
    )

    if device is None:
        device = next(pino_model.parameters()).device

    N = density_original.shape[0]

    # 初始化
    density = torch.from_numpy(density_original).float().to(device)
    density = torch.clamp(density, 0.01, 0.99)
    density_orig_t = torch.from_numpy(density_original).float().unsqueeze(0).unsqueeze(0).to(device)
    force_t = torch.from_numpy(force).float().unsqueeze(0).to(device)

    budget = (density_original > 0.5).sum() * 1.3

    for step in range(n_steps):
        density.requires_grad_(True)

        rho = density.unsqueeze(0).unsqueeze(0)

        # PINO 分析
        pino_fields = forward_surrogate_physical(pino_model, rho, force_t)
        vm = pino_fields['von_mises']

        # 结构 + 形状损失 (作为 score function)
        l_struct = structural_loss(vm, rho, yield_stress)
        l_shape = shape_preservation_loss(rho, density_orig_t)
        l_smooth = smoothness_loss(rho)

        score_loss = 3.0 * l_struct + 5.0 * l_shape + l_smooth

        score_loss.backward()
        grad = density.grad.detach()
        density.requires_grad_(False)

        # 扩散引导步: 去噪方向 = -grad + noise
        with torch.no_grad():
            # 步长随 step 衰减 (模拟扩散 schedule)
            t_frac = 1.0 - step / n_steps  # 1→0
            beta = noise_scale * t_frac  # 噪声递减

            # Langevin 风格更新
            noise = torch.randn_like(density) * np.sqrt(beta) if beta > 1e-4 else 0
            density = density - lr * grad + noise

            # 投影到 [0, 1]
            density = torch.clamp(density, 0.0, 1.0)

            # 体积约束投影 (对超出 budget 的部分截断最小密度体素)
            n_solid = (density > 0.5).sum()
            if n_solid > budget:
                sorted_vals, _ = torch.sort(density.flatten())
                cutoff_idx = len(sorted_vals) - int(budget)
                if 0 < cutoff_idx < len(sorted_vals):
                    cutoff = sorted_vals[cutoff_idx]
                    density[density < cutoff] *= 0.9

    return density.cpu().numpy()


# 基线注册表
BASELINES = {
    'ShadowArt-only': run_baseline_shadowart_only,
    'WeightedSum-Tuned': run_baseline_weighted_sum,
    'PostHoc-Repair': run_baseline_posthoc_repair,
    'FEM-Guided': run_baseline_fem_optimization,
    'SIMP': run_baseline_simp,
}

APPENDIX_BASELINES = {
    'ShadowArt+Stava': run_baseline_stava_postprocess,
    'NTOPo': run_baseline_ntopo,
    'TopoDiff': run_baseline_topodiff,
}
