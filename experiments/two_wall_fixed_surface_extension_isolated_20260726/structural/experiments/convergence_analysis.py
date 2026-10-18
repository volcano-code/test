"""
收敛分析实验

追踪 ALM 优化过程中的关键指标随迭代步数的变化:

1. IoU 收敛曲线 (外形恢复速度)
2. SF 收敛曲线 (安全系数提升)
3. ALM 乘子 μ 演化轨迹
4. 罚因子 ρ 增长曲线
5. 约束违反量 g(θ) 衰减
6. 梯度路由模式分布随时间的变化

输出:
  - convergence_data.json: 逐步指标
  - routing_timeline.json: 路由模式时间线

对应论文:
  Figure 1: 收敛曲线 (IoU + SF vs step, 对比 ALM vs 加权求和)
  Figure 2: ALM 乘子 & 罚因子演化
  Figure 3: 路由模式分布饼图/时间线
"""

from __future__ import annotations

import logging
import json
from typing import Dict, List

import numpy as np
import torch

from structural.analysis.stress_eval import (
    structural_loss, volume_loss,
    shape_preservation_loss, smoothness_loss,
)
from structural.models.load_model import forward_surrogate_physical
from structural.optimization.alm_optimizer import ALMOptimizer, ALMConfig
from structural.optimization.gradient_router import AdaptiveGradientRouter
from structural.experiments.evaluate_metrics import compute_3d_iou

logger = logging.getLogger(__name__)


def run_convergence_analysis(
    density_original: np.ndarray,
    force: np.ndarray,
    pino_model: torch.nn.Module,
    yield_stress: float = 50.0e6,
    device: torch.device | None = None,
    n_steps: int = 300,
    lr: float = 1e-3,
    log_interval: int = 1,
) -> Dict[str, list]:
    """运行 ALM 优化并记录逐步收敛数据

    同时运行加权求和版本作为对照。

    Args:
        density_original: [N, N, N] 初始密度
        force: [3, N, N, N] 力场
        pino_model: 预训练 PINO
        yield_stress: 屈服强度
        device: GPU 设备
        n_steps: 总步数
        lr: 学习率
        log_interval: 记录间隔

    Returns:
        dict with 'alm' and 'weighted_sum' convergence data
    """
    if device is None:
        device = next(pino_model.parameters()).device

    density_orig_t = torch.from_numpy(density_original).float().unsqueeze(0).unsqueeze(0).to(device)
    force_t = torch.from_numpy(force).float().unsqueeze(0).to(device)
    temperature = 15.0

    results = {}

    # ============ Run 1: ALM 方法 (Ours) ============
    print("  Running convergence analysis: ALM...")
    alm_data = _run_single_convergence(
        density_original, force_t, density_orig_t, pino_model,
        yield_stress, device, temperature,
        n_steps, lr, log_interval,
        use_alm=True, label="ALM",
    )
    results['alm'] = alm_data

    # ============ Run 2: 加权求和 (Baseline 3 style) ============
    print("  Running convergence analysis: Weighted Sum...")
    ws_data = _run_single_convergence(
        density_original, force_t, density_orig_t, pino_model,
        yield_stress, device, temperature,
        n_steps, lr, log_interval,
        use_alm=False, label="WeightedSum",
    )
    results['weighted_sum'] = ws_data

    return results


def _run_single_convergence(
    density_original: np.ndarray,
    force_t: torch.Tensor,
    density_orig_t: torch.Tensor,
    pino_model: torch.nn.Module,
    yield_stress: float,
    device: torch.device,
    temperature: float,
    n_steps: int,
    lr: float,
    log_interval: int,
    use_alm: bool,
    label: str,
) -> Dict[str, list]:
    """单次优化运行, 记录逐步数据"""

    # 初始化
    density_t = torch.from_numpy(density_original).float().to(device)
    density_t = torch.clamp(density_t, 1e-6, 1 - 1e-6)
    log_density = torch.nn.Parameter(
        torch.log(density_t / (1 - density_t)) / temperature
    )

    optimizer = torch.optim.Adam([log_density], lr=lr)

    alm = None
    grad_router = None
    if use_alm:
        alm = ALMOptimizer(ALMConfig(
            safety_factor_target=1.5,
            volume_budget_ratio=1.3,
            dual_update_interval=5,
        ))
        grad_router = AdaptiveGradientRouter(
            sf_target=1.5,
            warmup_steps=20,
        )

    data = {
        'steps': [],
        'iou': [],
        'sf_min': [],
        'total_loss': [],
        'l_struct': [],
        'l_vol': [],
        'l_shape': [],
        # ALM specific
        'mu_safety': [],
        'mu_volume': [],
        'mu_overhang': [],
        'rho': [],
        'g_safety': [],
        'g_volume': [],
        'g_overhang': [],
        'penalty': [],
        # Router
        'routing_mode': [],
    }

    for step in range(n_steps):
        density_current = torch.sigmoid(temperature * log_density).unsqueeze(0).unsqueeze(0)

        # PINO
        pino_fields = forward_surrogate_physical(pino_model, density_current, force_t)
        vm = pino_fields['von_mises']

        l_shape = shape_preservation_loss(density_current, density_orig_t)
        l_smooth = smoothness_loss(density_current)

        if use_alm and alm is not None:
            violations = alm.compute_constraint_violations(
                vm, density_current, density_orig_t, yield_stress
            )
            alm_result = alm.compute_alm_loss(
                appearance_loss=l_shape,
                violations=violations,
                smoothness_loss_val=l_smooth,
            )
            total = alm_result['total']
            alm.update_violations(violations)
            alm.step(step)
        else:
            l_struct = structural_loss(vm, density_current, yield_stress)
            l_vol = volume_loss(density_current, density_orig_t, 1.3)
            total = 2.0 * l_struct + 1.0 * l_vol + 5.0 * l_shape + l_smooth
            violations = {}

        optimizer.zero_grad()
        total.backward()

        # 路由 (ALM only)
        routing_mode = "joint"
        if use_alm and grad_router is not None:
            with torch.no_grad():
                solid_mask = (density_current > 0.1).float()
                vm_max = float((vm * solid_mask).max())
                sf = yield_stress / (vm_max + 1e-12)
                n_violated = sum(1 for v in violations.values() if float(v.detach()) > 0.01)

            # Simplified routing (no repair_cnn here)
            decision = grad_router.decide(sf, float(l_shape.detach()), n_violated, step)
            routing_mode = decision.mode.value

        torch.nn.utils.clip_grad_norm_([log_density], 1.0)
        optimizer.step()

        # 记录
        if step % log_interval == 0:
            with torch.no_grad():
                d_np = torch.sigmoid(temperature * log_density).cpu().numpy()
                iou = compute_3d_iou(d_np, density_original)
                solid_mask = (density_current > 0.1).float()
                vm_max = float((vm * solid_mask).max())
                sf_min = yield_stress / (vm_max + 1e-12)

            data['steps'].append(step)
            data['iou'].append(iou)
            data['sf_min'].append(sf_min)
            data['total_loss'].append(float(total.detach()))

            if use_alm and alm is not None:
                summary = alm.get_summary()
                data['mu_safety'].append(summary['mu_safety'])
                data['mu_volume'].append(summary['mu_volume'])
                data['mu_overhang'].append(summary['mu_overhang'])
                data['rho'].append(summary['rho'])
                data['g_safety'].append(summary['violation_safety'])
                data['g_volume'].append(summary['violation_volume'])
                data['g_overhang'].append(summary['violation_overhang'])
                data['penalty'].append(float(alm_result.get('penalty', 0)))
            else:
                for key in ['mu_safety', 'mu_volume', 'mu_overhang', 'rho',
                            'g_safety', 'g_volume', 'g_overhang', 'penalty']:
                    data[key].append(0.0)

            data['routing_mode'].append(routing_mode)

    return data


def compute_routing_statistics(convergence_data: Dict) -> Dict:
    """从收敛数据中提取路由模式统计

    Returns:
        dict with mode counts and timeline segments
    """
    alm_data = convergence_data.get('alm', {})
    modes = alm_data.get('routing_mode', [])
    steps = alm_data.get('steps', [])

    if not modes:
        return {}

    # 模式计数
    counts = {}
    for m in modes:
        counts[m] = counts.get(m, 0) + 1
    total = sum(counts.values())
    fractions = {m: c / total for m, c in counts.items()}

    # 时间线分段: 连续相同模式的区间
    segments = []
    if modes:
        current_mode = modes[0]
        start_step = steps[0]
        for i in range(1, len(modes)):
            if modes[i] != current_mode:
                segments.append({
                    'mode': current_mode,
                    'start': start_step,
                    'end': steps[i - 1],
                    'duration': steps[i - 1] - start_step + 1,
                })
                current_mode = modes[i]
                start_step = steps[i]
        segments.append({
            'mode': current_mode,
            'start': start_step,
            'end': steps[-1],
            'duration': steps[-1] - start_step + 1,
        })

    return {
        'counts': counts,
        'fractions': fractions,
        'total_steps': total,
        'segments': segments,
    }


def format_convergence_table_latex(
    alm_data: Dict[str, list],
    ws_data: Dict[str, list],
    milestones: list | None = None,
) -> str:
    """格式化收敛对比表 (ALM vs WeightedSum 在关键步数处的指标)"""
    if milestones is None:
        milestones = [0, 50, 100, 150, 200]

    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Convergence comparison: ALM vs.~weighted-sum optimization.}\n"
        "\\label{tab:convergence}\n"
        "\\begin{tabular}{l|cc|cc}\n"
        "\\toprule\n"
        "& \\multicolumn{2}{c|}{ALM (Ours)} & \\multicolumn{2}{c}{Weighted Sum} \\\\\n"
        "Step & IoU$\\uparrow$ & SF$_{\\min}\\uparrow$ & IoU$\\uparrow$ & SF$_{\\min}\\uparrow$ \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for ms in milestones:
        # 找最近的记录步
        alm_idx = _find_nearest_idx(alm_data['steps'], ms)
        ws_idx = _find_nearest_idx(ws_data['steps'], ms)

        if alm_idx is not None and ws_idx is not None:
            rows.append(
                f"{ms} & "
                f"{alm_data['iou'][alm_idx]:.3f} & "
                f"{alm_data['sf_min'][alm_idx]:.2f} & "
                f"{ws_data['iou'][ws_idx]:.3f} & "
                f"{ws_data['sf_min'][ws_idx]:.2f} \\\\"
            )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

    return header + "\n".join(rows) + "\n" + footer


def _find_nearest_idx(steps: list, target: int) -> int | None:
    """在 steps 列表中找到最接近 target 的索引"""
    if not steps:
        return None
    diffs = [abs(s - target) for s in steps]
    return diffs.index(min(diffs))
