"""
Safety Transfer Bound 批量验证实验

验证定理 (Theorem 1) 在数据集上的有效性和紧致性:

1. 在 N 个测试样本上, 同时计算:
   - SF_PINO (PINO 预测安全系数)
   - SF_FEM  (FEM 真实安全系数)
   - SF_lower (Safety Transfer Bound 下界)

2. 报告:
   - Bound validity rate: SF_FEM ≥ SF_lower 的比例 (应为 ~100%)
   - Tightness: (SF_FEM - SF_lower) / SF_FEM 的均值 (越小越好)
   - ε_u 分布: PINO 位移误差统计

3. 生成 LaTeX 表格 (Table 2) 和散点图数据

4. 参数敏感性: 扫描不同 ε_u 对 bound 紧致性的影响
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from structural.models.load_model import forward_surrogate_physical
from structural.optimization.safety_bound import SafetyTransferBound, SafetyVerificationResult

logger = logging.getLogger(__name__)


def run_bound_verification_single(
    stb: SafetyTransferBound,
    pino_model: torch.nn.Module,
    density: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    device: torch.device,
    yield_stress: float = 50.0e6,
) -> Dict[str, float]:
    """对单个模型验证 Safety Transfer Bound

    需要 FEM 真值来比较。

    Args:
        stb: SafetyTransferBound 实例
        pino_model: 预训练 PINO
        density: [N, N, N] 密度场
        force: [3, N, N, N] 力场
        fixed_mask: [1, N, N, N] 固定边界
        device: GPU 设备
        yield_stress: 屈服强度

    Returns:
        dict with sf_pino, sf_fem, sf_lower, eps_u, tightness, bound_valid
    """
    # PINO 预测
    rho = torch.from_numpy(density).float().unsqueeze(0).unsqueeze(0).to(device)
    f = torch.from_numpy(force).float().unsqueeze(0).to(device)

    with torch.no_grad():
        pino_fields = forward_surrogate_physical(pino_model, rho, f)
        u_pino = pino_fields['displacement']
        s_pino = pino_fields['stress']
        vm_pino = pino_fields['von_mises']

    # FEM 真值
    try:
        from structural.data.generate_fem_dataset import density_to_fem_fields
        from sgldbench.config import MaterialProperties, SolverSettings

        material = MaterialProperties(E=stb.E, nu=stb.nu, cell_size=1.0)
        solver = SolverSettings(tolerance=1e-5, max_iterations=500)

        fem_result = density_to_fem_fields(
            density.astype(bool), force, fixed_mask,
            material, solver,
        )

        if fem_result is None:
            return {'fem_success': False}

        # 转为 tensor
        u_fem = torch.from_numpy(fem_result['displacement']).float().unsqueeze(0).to(device)
        s_fem = torch.from_numpy(fem_result['stress']).float().unsqueeze(0).to(device)

        # 使用 verify_with_fem
        result = stb.verify_with_fem(
            pino_model=pino_model,
            density=rho,
            force=f,
            fem_displacement=u_fem,
            fem_stress=s_fem,
        )

        return {
            'fem_success': True,
            'sf_pino': result.sf_pino,
            'sf_fem': result.sf_fem,
            'sf_lower': result.sf_lower_bound,
            'eps_u': result.displacement_error,
            'eps_s': result.stress_error,
            'tightness': result.tightness,
            'bound_valid': result.bound_is_valid,
            'vm_max_pino': result.vm_max_pino,
            'vm_max_fem': result.vm_max_fem,
            'K': result.K_constant,
        }

    except Exception as e:
        logger.warning(f"FEM verification failed: {e}")
        return {'fem_success': False}


def run_bound_verification_dataset(
    stb: SafetyTransferBound,
    pino_model: torch.nn.Module,
    data_dir: str,
    device: torch.device,
    yield_stress: float = 50.0e6,
    max_models: int | None = None,
) -> Dict:
    """在整个测试数据集上验证 Safety Transfer Bound

    读取 prepare_data.py 生成的数据集目录。

    Args:
        stb: SafetyTransferBound 实例
        pino_model: 预训练 PINO
        data_dir: prepare_experiment_dataset() 输出的目录
        device: GPU 设备
        yield_stress: 屈服强度
        max_models: 最多验证的模型数

    Returns:
        dict with aggregate statistics + per-model results
    """
    data_path = Path(data_dir)
    meta_file = data_path / "dataset_meta.json"

    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        model_ids = [m['model_id'] for m in meta['models']]
    else:
        model_dirs = sorted(data_path.glob("model_*"))
        model_ids = [d.name for d in model_dirs]

    if max_models is not None:
        model_ids = model_ids[:max_models]

    results = []
    print(f"Running bound verification on {len(model_ids)} models...")

    for i, model_id in enumerate(model_ids):
        model_dir = data_path / model_id

        voxel_file = model_dir / "voxels.npy"
        loads_file = model_dir / "loads.npz"

        if not voxel_file.exists() or not loads_file.exists():
            logger.warning(f"Skipping {model_id}: missing files")
            continue

        density = np.load(voxel_file)
        loads = np.load(loads_file)
        force = loads['force_standing']
        fixed_mask = loads['fixed_mask_standing']

        result = run_bound_verification_single(
            stb, pino_model, density, force, fixed_mask,
            device, yield_stress,
        )
        result['model_id'] = model_id
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Verified {i + 1}/{len(model_ids)} models")

    # 聚合
    valid_results = [r for r in results if r.get('fem_success', False)]

    if not valid_results:
        print("No valid FEM results!")
        return {'n_total': len(results), 'n_valid': 0}

    stats = {
        'n_total': len(results),
        'n_valid': len(valid_results),
        'bound_validity_rate': np.mean([r['bound_valid'] for r in valid_results]),
        'tightness_mean': np.mean([r['tightness'] for r in valid_results]),
        'tightness_std': np.std([r['tightness'] for r in valid_results]),
        'tightness_max': np.max([r['tightness'] for r in valid_results]),
        'eps_u_mean': np.mean([r['eps_u'] for r in valid_results]),
        'eps_u_std': np.std([r['eps_u'] for r in valid_results]),
        'eps_u_max': np.max([r['eps_u'] for r in valid_results]),
        'eps_s_mean': np.mean([r['eps_s'] for r in valid_results]),
        'eps_s_max': np.max([r['eps_s'] for r in valid_results]),
        'sf_pino_mean': np.mean([r['sf_pino'] for r in valid_results]),
        'sf_fem_mean': np.mean([r['sf_fem'] for r in valid_results]),
        'sf_lower_mean': np.mean([r['sf_lower'] for r in valid_results]),
        'per_model': valid_results,
    }

    print(f"\nBound Verification Summary ({stats['n_valid']} models):")
    print(f"  Validity rate: {stats['bound_validity_rate']*100:.1f}%")
    print(f"  Tightness: {stats['tightness_mean']:.4f} ± {stats['tightness_std']:.4f}")
    print(f"  ε_u: {stats['eps_u_mean']:.4f} ± {stats['eps_u_std']:.4f} (max={stats['eps_u_max']:.4f})")
    print(f"  SF_PINO: {stats['sf_pino_mean']:.3f}, SF_FEM: {stats['sf_fem_mean']:.3f}, "
          f"SF_lower: {stats['sf_lower_mean']:.3f}")

    return stats


def run_epsilon_sensitivity(
    stb: SafetyTransferBound,
    pino_model: torch.nn.Module,
    density: np.ndarray,
    force: np.ndarray,
    device: torch.device,
    epsilon_values: list | None = None,
) -> List[Dict]:
    """ε_u 参数敏感性分析

    扫描不同的 ε_u 值, 观察 bound 紧致性如何变化。
    用于论文 Figure: ε_u vs tightness 曲线。

    Args:
        stb: SafetyTransferBound 实例
        pino_model: PINO 模型
        density: [N, N, N] 一个测试密度场
        force: [3, N, N, N] 力场
        device: GPU 设备
        epsilon_values: 要扫描的 ε_u 列表

    Returns:
        list of dict with eps_u, sf_lower, ratio
    """
    if epsilon_values is None:
        epsilon_values = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

    rho = torch.from_numpy(density).float().unsqueeze(0).unsqueeze(0).to(device)
    f = torch.from_numpy(force).float().unsqueeze(0).to(device)

    results = []
    for eps in epsilon_values:
        with torch.no_grad():
            bound_result = stb.compute_bound(
                pino_model=pino_model,
                density=rho,
                force=f,
                epsilon_u=eps,
            )

        results.append({
            'eps_u': eps,
            'sf_pino': bound_result.sf_pino,
            'sf_lower': bound_result.sf_lower_bound,
            'ratio': bound_result.sf_lower_bound / (bound_result.sf_pino + 1e-12),
        })

    print("\nε_u Sensitivity Analysis:")
    print(f"  {'ε_u':>8s}  {'SF_PINO':>8s}  {'SF_lower':>8s}  {'ratio':>8s}")
    print(f"  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}")
    for r in results:
        print(f"  {r['eps_u']:8.3f}  {r['sf_pino']:8.3f}  {r['sf_lower']:8.3f}  {r['ratio']:8.4f}")

    return results


def format_bound_table_latex(stats: Dict) -> str:
    """格式化 Safety Transfer Bound 验证结果为 LaTeX 表格 (Table 2)"""
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Safety Transfer Bound verification on the test set.}\n"
        "\\label{tab:bound_verification}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Metric & Mean & Std & Max & Min \\\\\n"
        "\\midrule\n"
    )

    per_model = stats.get('per_model', [])
    if not per_model:
        return header + "\\bottomrule\n\\end{tabular}\n\\end{table}"

    eps_u = [r['eps_u'] for r in per_model]
    eps_s = [r['eps_s'] for r in per_model]
    tightness = [r['tightness'] for r in per_model]
    sf_pino = [r['sf_pino'] for r in per_model]
    sf_fem = [r['sf_fem'] for r in per_model]
    sf_lower = [r['sf_lower'] for r in per_model]

    def row(name, vals):
        return (f"{name} & {np.mean(vals):.4f} & {np.std(vals):.4f} "
                f"& {np.max(vals):.4f} & {np.min(vals):.4f} \\\\")

    rows = [
        row("$\\varepsilon_u$ (disp. error)", eps_u),
        row("$\\varepsilon_\\sigma$ (stress error)", eps_s),
        row("SF$_{\\text{PINO}}$", sf_pino),
        row("SF$_{\\text{FEM}}$", sf_fem),
        row("SF$_{\\text{lower}}$ (bound)", sf_lower),
        row("Tightness", tightness),
        "\\midrule",
        f"Validity rate & \\multicolumn{{4}}{{c}}{{{stats['bound_validity_rate']*100:.1f}\\%}} \\\\",
    ]

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

    return header + "\n".join(rows) + "\n" + footer


def format_epsilon_table_latex(results: List[Dict]) -> str:
    """格式化 ε_u 敏感性分析为 LaTeX 表格"""
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Sensitivity of Safety Transfer Bound tightness to PINO accuracy $\\varepsilon_u$.}\n"
        "\\label{tab:epsilon_sensitivity}\n"
        "\\begin{tabular}{cccc}\n"
        "\\toprule\n"
        "$\\varepsilon_u$ & SF$_{\\text{PINO}}$ & SF$_{\\text{lower}}$ & Ratio \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for r in results:
        rows.append(
            f"{r['eps_u']:.3f} & {r['sf_pino']:.3f} & {r['sf_lower']:.3f} & {r['ratio']:.4f} \\\\"
        )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

    return header + "\n".join(rows) + "\n" + footer
