"""
PINO 精度验证实验

验证 PINO 替代 FEM 的精度与保守性, 为保守 surrogate 筛查提供输入。

实验内容:
  Table 5: PINO vs FEM 精度对比
    - 位移场 L² 相对误差 ε_u
    - 应力场 L² 相对误差 ε_σ
    - Von Mises 最大应力误差
    - 安全系数误差 |SF_PINO - SF_FEM| / SF_FEM
    - 危险指标低估率 (VM underestimation / SF overestimation)

  Figure 4: 逐样本误差分布直方图
  Figure 5: 不同体积占比下的误差变化 (密度稀疏 vs 密实)
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def validate_pino_single(
    pino_model: torch.nn.Module,
    density: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    yield_stress: float,
    device: torch.device,
    youngs_modulus: float = 3.5e9,
    poisson_ratio: float = 0.36,
) -> Dict[str, float]:
    """对单个模型做 PINO vs FEM 精度验证

    Returns:
        dict with eps_u, eps_sigma, eps_vm_max, eps_sf, solid_ratio, ...
    """
    from structural.analysis.stress_eval import compute_von_mises_torch
    from structural.analysis.load_generator import LoadGenerator
    from structural.models.load_model import forward_surrogate_physical

    # PINO 预测
    density = np.asarray(density, dtype=np.float32)
    force = np.asarray(force, dtype=np.float32)
    fixed_mask = np.asarray(fixed_mask, dtype=np.float32)
    if fixed_mask.ndim == 3:
        fixed_mask = fixed_mask[None, ...]

    # Some prepared datasets contain standing support masks that do not overlap
    # with the occupied voxels. In that case, rebuild a valid standing-load
    # configuration so the FEM side remains meaningful.
    support_overlap = 0
    if fixed_mask.ndim == 4 and fixed_mask.shape[0] > 0:
        support_overlap = int(np.logical_and(density > 0.5, fixed_mask[0] > 0.5).sum())
    if support_overlap == 0:
        fallback_load = LoadGenerator().standing_load(density)
        force = fallback_load["force"].astype(np.float32)
        fixed_mask = fallback_load["fixed_mask"].astype(np.float32)

    rho = torch.from_numpy(density).float().unsqueeze(0).unsqueeze(0).to(device)
    f = torch.from_numpy(force).float().unsqueeze(0).to(device)

    pino_fields = forward_surrogate_physical(pino_model, rho, f)
    u_pino = pino_fields['displacement']
    s_pino = pino_fields['stress']
    vm_pino = compute_von_mises_torch(s_pino)

    # FEM 真值
    try:
        from structural.analysis.fem_solver import voxel_fem_solve

        fem_result = voxel_fem_solve(
            density=density,
            force_field=force,
            fixed_mask=fixed_mask,
            E=youngs_modulus,
            nu=poisson_ratio,
            density_threshold=0.3,
            verbose=False,
        )

        if fem_result is None:
            return {'fem_success': False}

        u_fem = torch.from_numpy(fem_result['displacement']).float().unsqueeze(0).to(device)
        s_fem = torch.from_numpy(fem_result['stress']).float().unsqueeze(0).to(device)
        vm_fem = compute_von_mises_torch(s_fem)

        solid_mask = (rho > 0.1).float()

        # 位移误差
        u_diff_norm = torch.sqrt(((u_pino - u_fem) ** 2).sum())
        u_fem_norm = torch.sqrt((u_fem ** 2).sum())
        eps_u = float(u_diff_norm / (u_fem_norm + 1e-12))

        # 应力误差
        s_diff = (s_pino - s_fem) * solid_mask
        s_fem_masked = s_fem * solid_mask
        eps_sigma = float(
            torch.sqrt((s_diff ** 2).sum()) /
            (torch.sqrt((s_fem_masked ** 2).sum()) + 1e-12)
        )

        # VM 最大值误差
        vm_max_pino = float((vm_pino * solid_mask).max())
        vm_max_fem = float((vm_fem * solid_mask).max())
        eps_vm_max = abs(vm_max_pino - vm_max_fem) / (vm_max_fem + 1e-12)

        # SF 误差
        sf_pino = yield_stress / (vm_max_pino + 1e-12)
        sf_fem = yield_stress / (vm_max_fem + 1e-12)
        eps_sf = abs(sf_pino - sf_fem) / (sf_fem + 1e-12)

        # 逐体素 VM 误差 (RMSE)
        vm_diff = (vm_pino - vm_fem) * solid_mask
        n_solid = solid_mask.sum().clamp(min=1)
        vm_rmse = float(torch.sqrt((vm_diff ** 2).sum() / n_solid))

        # 位移逐分量 MAE
        u_mae = float(((u_pino - u_fem).abs() * solid_mask).sum() / (n_solid * 3))

        solid_ratio = float((density > 0.5).mean())

        return {
            'fem_success': True,
            'eps_u': eps_u,
            'eps_sigma': eps_sigma,
            'eps_vm_max': eps_vm_max,
            'eps_sf': eps_sf,
            'vm_max_pino': vm_max_pino,
            'vm_max_fem': vm_max_fem,
            'sf_pino': sf_pino,
            'sf_fem': sf_fem,
            'vm_underestimate': float(vm_max_pino < vm_max_fem),
            'sf_overestimate': float(sf_pino > sf_fem),
            'vm_rmse': vm_rmse,
            'u_mae': u_mae,
            'solid_ratio': solid_ratio,
        }

    except Exception as e:
        logger.warning(f"FEM validation failed: {e}")
        return {'fem_success': False}


def validate_pino_dataset(
    pino_model: torch.nn.Module,
    data_dir: str,
    device: torch.device,
    yield_stress: float = 50.0e6,
    max_models: int | None = None,
) -> Dict:
    """在数据集上验证 PINO 精度

    Args:
        pino_model: 预训练 PINO
        data_dir: prepare_experiment_dataset() 输出目录
        device: GPU
        yield_stress: 屈服强度
        max_models: 最多验证的模型数

    Returns:
        dict with aggregate + per-model results
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
    print(f"Validating PINO on {len(model_ids)} models...")

    for i, model_id in enumerate(model_ids):
        model_dir = data_path / model_id

        voxel_file = model_dir / "voxels.npy"
        loads_file = model_dir / "loads.npz"

        if not voxel_file.exists() or not loads_file.exists():
            continue

        density = np.load(voxel_file).astype(np.float32)
        loads = np.load(loads_file)
        force = loads['force_standing']
        fixed_mask = loads['fixed_mask_standing']

        result = validate_pino_single(
            pino_model, density, force, fixed_mask,
            yield_stress, device,
        )
        result['model_id'] = model_id
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Validated {i + 1}/{len(model_ids)}")

    valid = [r for r in results if r.get('fem_success', False)]
    if not valid:
        print("No valid FEM results!")
        return {'n_total': len(results), 'n_valid': 0}

    # 聚合
    stats = {
        'n_total': len(results),
        'n_valid': len(valid),
        'eps_u_mean': float(np.mean([r['eps_u'] for r in valid])),
        'eps_u_std': float(np.std([r['eps_u'] for r in valid])),
        'eps_u_max': float(np.max([r['eps_u'] for r in valid])),
        'eps_u_median': float(np.median([r['eps_u'] for r in valid])),
        'eps_u_p95': float(np.percentile([r['eps_u'] for r in valid], 95)),
        'eps_sigma_mean': float(np.mean([r['eps_sigma'] for r in valid])),
        'eps_sigma_std': float(np.std([r['eps_sigma'] for r in valid])),
        'eps_sigma_max': float(np.max([r['eps_sigma'] for r in valid])),
        'eps_sigma_p95': float(np.percentile([r['eps_sigma'] for r in valid], 95)),
        'eps_vm_max_mean': float(np.mean([r['eps_vm_max'] for r in valid])),
        'eps_vm_max_std': float(np.std([r['eps_vm_max'] for r in valid])),
        'eps_vm_max_p95': float(np.percentile([r['eps_vm_max'] for r in valid], 95)),
        'eps_sf_mean': float(np.mean([r['eps_sf'] for r in valid])),
        'eps_sf_std': float(np.std([r['eps_sf'] for r in valid])),
        'eps_sf_p95': float(np.percentile([r['eps_sf'] for r in valid], 95)),
        'sf_pino_mean': float(np.mean([r['sf_pino'] for r in valid])),
        'sf_fem_mean': float(np.mean([r['sf_fem'] for r in valid])),
        'vm_underestimate_rate': float(np.mean([r['vm_underestimate'] for r in valid])),
        'sf_overestimate_rate': float(np.mean([r['sf_overestimate'] for r in valid])),
        'vm_rmse_mean': float(np.mean([r['vm_rmse'] for r in valid])),
        'u_mae_mean': float(np.mean([r['u_mae'] for r in valid])),
        'per_model': valid,
    }

    print(f"\nPINO Validation Summary ({stats['n_valid']} models):")
    print(f"  ε_u (displacement): {stats['eps_u_mean']:.4f} ± {stats['eps_u_std']:.4f} "
          f"(max={stats['eps_u_max']:.4f})")
    print(f"  ε_σ (stress):       {stats['eps_sigma_mean']:.4f} ± {stats['eps_sigma_std']:.4f}")
    print(f"  ε_vm_max:           {stats['eps_vm_max_mean']:.4f} ± {stats['eps_vm_max_std']:.4f}")
    print(f"  ε_SF:               {stats['eps_sf_mean']:.4f} ± {stats['eps_sf_std']:.4f}")
    print(f"  VM under-rate:      {stats['vm_underestimate_rate']:.2%}")
    print(f"  SF over-rate:       {stats['sf_overestimate_rate']:.2%}")
    print(f"  SF mean — PINO: {stats['sf_pino_mean']:.3f}, FEM: {stats['sf_fem_mean']:.3f}")

    return stats


def validate_pino_by_density_bin(
    pino_model: torch.nn.Module,
    data_dir: str,
    device: torch.device,
    yield_stress: float = 50.0e6,
    n_bins: int = 5,
    max_models: int | None = None,
) -> List[Dict]:
    """按体积占比分箱验证 PINO (Figure 5 数据)

    将模型按 solid_ratio 分成 n_bins 个区间,
    分别统计每个区间的 PINO 误差。

    用于验证 PINO 在不同密度水平下的泛化能力。
    """
    full_stats = validate_pino_dataset(
        pino_model, data_dir, device, yield_stress, max_models
    )

    valid = full_stats.get('per_model', [])
    if not valid:
        return []

    ratios = [r['solid_ratio'] for r in valid]
    bin_edges = np.linspace(min(ratios), max(ratios), n_bins + 1)

    binned_results = []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = [r for r in valid if lo <= r['solid_ratio'] < hi or
                  (b == n_bins - 1 and r['solid_ratio'] == hi)]

        if not in_bin:
            continue

        binned_results.append({
            'bin_lo': float(lo),
            'bin_hi': float(hi),
            'n_samples': len(in_bin),
            'solid_ratio_mean': float(np.mean([r['solid_ratio'] for r in in_bin])),
            'eps_u_mean': float(np.mean([r['eps_u'] for r in in_bin])),
            'eps_u_std': float(np.std([r['eps_u'] for r in in_bin])),
            'eps_sigma_mean': float(np.mean([r['eps_sigma'] for r in in_bin])),
            'eps_sf_mean': float(np.mean([r['eps_sf'] for r in in_bin])),
            'vm_underestimate_rate': float(np.mean([r['vm_underestimate'] for r in in_bin])),
            'sf_overestimate_rate': float(np.mean([r['sf_overestimate'] for r in in_bin])),
        })

    print(f"\nPINO accuracy by density bin:")
    print(f"  {'Solid Ratio':>12s}  {'N':>4s}  {'ε_u':>8s}  {'ε_σ':>8s}  {'ε_SF':>8s}  {'VM↓':>8s}  {'SF↑':>8s}")
    for b in binned_results:
        print(f"  [{b['bin_lo']:.2f}, {b['bin_hi']:.2f})  "
              f"{b['n_samples']:4d}  "
              f"{b['eps_u_mean']:.4f}  "
              f"{b['eps_sigma_mean']:.4f}  "
              f"{b['eps_sf_mean']:.4f}  "
              f"{b['vm_underestimate_rate']:.2%}  "
              f"{b['sf_overestimate_rate']:.2%}")

    return binned_results


def format_pino_table_latex(stats: Dict) -> str:
    """格式化 PINO 精度验证为 LaTeX 表格 (Table 5)"""
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{PINO approximation accuracy compared to FEM ground truth.}\n"
        "\\label{tab:pino_accuracy}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Error Metric & Mean & Std & P95 & Max & Conservative Risk \\\\\n"
        "\\midrule\n"
    )

    rows = [
        f"$\\varepsilon_u$ (displacement) & {stats['eps_u_mean']:.4f} "
        f"& {stats['eps_u_std']:.4f} & {stats['eps_u_p95']:.4f} & {stats['eps_u_max']:.4f} & --- \\\\",

        f"$\\varepsilon_\\sigma$ (stress) & {stats['eps_sigma_mean']:.4f} "
        f"& {stats['eps_sigma_std']:.4f} & {stats['eps_sigma_p95']:.4f} & {stats['eps_sigma_max']:.4f} & --- \\\\",

        f"$\\varepsilon_{{\\text{{VM\\_max}}}}$ & {stats['eps_vm_max_mean']:.4f} "
        f"& {stats['eps_vm_max_std']:.4f} & {stats['eps_vm_max_p95']:.4f} & --- & "
        f"{100.0 * stats['vm_underestimate_rate']:.1f}\\% \\\\",

        f"$\\varepsilon_{{\\text{{SF}}}}$ & {stats['eps_sf_mean']:.4f} "
        f"& {stats['eps_sf_std']:.4f} & {stats['eps_sf_p95']:.4f} & --- & "
        f"{100.0 * stats['sf_overestimate_rate']:.1f}\\% \\\\",

        "\\midrule",

        f"SF$_{{\\text{{PINO}}}}$ mean & \\multicolumn{{5}}{{c}}{{{stats['sf_pino_mean']:.3f}}} \\\\",
        f"SF$_{{\\text{{FEM}}}}$ mean & \\multicolumn{{5}}{{c}}{{{stats['sf_fem_mean']:.3f}}} \\\\",
    ]

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

    return header + "\n".join(rows) + "\n" + footer
