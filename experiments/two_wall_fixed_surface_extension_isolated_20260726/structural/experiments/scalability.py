"""
分辨率可扩展性实验 (Table 7, Figure 8)

在 32³ / 64³ / 128³ 三个分辨率下运行完整 StructShadow 管线,
对比:
  1. 外形保真度 (IoU) — 高分辨率是否带来更好的剪影匹配
  2. 结构安全 (SF_min) — 高分辨率 FEM 是否更精确
  3. 约束满足率 (CSR) — ALM 在不同规模下的收敛性
  4. PINO 精度 (ε_u) — 代理误差随分辨率的变化
  5. 计算时间 — 端到端用时 (训练 + 推理)
  6. GPU 显存峰值 — 实际硬件需求

对应论文:
  Table 7: Resolution scalability (32³ vs 64³ vs 128³)
  Figure 8: 各指标随分辨率的变化曲线

用法:
    python scalability.py \
        --pino_checkpoints pino_32.pt,pino_64.pt,pino_128.pt \
        --output_dir ./scalability_results
"""

from __future__ import annotations

import gc
import time
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from structural.models.load_model import forward_surrogate_physical

logger = logging.getLogger(__name__)

# 三档分辨率配置
RESOLUTION_CONFIGS = {
    32: {
        'pino_modes': 8,
        'pino_width': 24,
        'pino_padding': 4,
        'n_steps': 150,
        'batch_size': 16,
        'description': 'Low resolution (fast prototyping)',
    },
    64: {
        'pino_modes': 12,
        'pino_width': 32,
        'pino_padding': 8,
        'n_steps': 200,
        'batch_size': 8,
        'description': 'Default resolution',
    },
    128: {
        'pino_modes': 16,
        'pino_width': 32,
        'pino_padding': 12,
        'n_steps': 200,
        'batch_size': 2,
        'description': 'High resolution (publication quality)',
    },
}


def rescale_voxels(
    density: np.ndarray,
    target_resolution: int,
) -> np.ndarray:
    """将体素场重采样到目标分辨率

    使用最近邻插值 (保持二值性)。

    Args:
        density: [N, N, N] 原始密度场
        target_resolution: 目标分辨率 M

    Returns:
        [M, M, M] 重采样后的密度场
    """
    from scipy.ndimage import zoom

    N = density.shape[0]
    if N == target_resolution:
        return density.copy()

    scale = target_resolution / N
    # 用 order=0 (最近邻) 保持清晰边界
    resampled = zoom(density, scale, order=0)

    # 确保精确尺寸
    M = target_resolution
    result = np.zeros((M, M, M), dtype=density.dtype)
    sx = min(resampled.shape[0], M)
    sy = min(resampled.shape[1], M)
    sz = min(resampled.shape[2], M)
    result[:sx, :sy, :sz] = resampled[:sx, :sy, :sz]

    return result


def rescale_force(
    force: np.ndarray,
    target_resolution: int,
) -> np.ndarray:
    """将力场重采样到目标分辨率

    Args:
        force: [3, N, N, N]
        target_resolution: M

    Returns:
        [3, M, M, M]
    """
    from scipy.ndimage import zoom

    N = force.shape[1]
    if N == target_resolution:
        return force.copy()

    scale = target_resolution / N
    result = np.zeros((3, target_resolution, target_resolution, target_resolution),
                      dtype=force.dtype)
    for c in range(3):
        result[c] = zoom(force[c], scale, order=1)[:target_resolution,
                                                     :target_resolution,
                                                     :target_resolution]

    # 力密度需要按体积比缩放 (体素变小, 单体素力变小)
    volume_ratio = (1.0 / target_resolution) ** 3 / (1.0 / N) ** 3
    result *= volume_ratio

    return result


def rescale_fixed_mask(
    fixed_mask: np.ndarray,
    target_resolution: int,
) -> np.ndarray:
    """将固定约束掩码重采样到目标分辨率

    Args:
        fixed_mask: [1, N, N, N]
        target_resolution: M

    Returns:
        [1, M, M, M]
    """
    from scipy.ndimage import zoom

    N = fixed_mask.shape[1]
    if N == target_resolution:
        return fixed_mask.copy()

    scale = target_resolution / N
    result = np.zeros((1, target_resolution, target_resolution, target_resolution),
                      dtype=fixed_mask.dtype)
    resampled = zoom(fixed_mask[0], scale, order=0)
    M = target_resolution
    sx = min(resampled.shape[0], M)
    sy = min(resampled.shape[1], M)
    sz = min(resampled.shape[2], M)
    result[0, :sx, :sy, :sz] = resampled[:sx, :sy, :sz]

    return result


def measure_gpu_memory() -> float:
    """当前 GPU 显存使用量 (MB)"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def run_single_resolution(
    resolution: int,
    density_original_64: np.ndarray,
    force_64: np.ndarray,
    fixed_mask_64: np.ndarray,
    pino_checkpoint: str | None,
    device: torch.device,
    yield_stress: float = 50.0e6,
    n_steps: int | None = None,
) -> Dict:
    """在指定分辨率下运行完整管线 (Ours)

    Args:
        resolution: 目标分辨率 (32, 64, 128)
        density_original_64: [64, 64, 64] 基准密度 (始终从64³开始)
        force_64: [3, 64, 64, 64] 基准力场
        fixed_mask_64: [1, 64, 64, 64] 基准固定掩码
        pino_checkpoint: PINO 权重路径 (该分辨率的)
        device: GPU
        yield_stress: 屈服强度
        n_steps: 优化步数 (None 则用默认)

    Returns:
        dict with resolution, metrics, timing, memory
    """
    from structural.analysis.stress_eval import (
        structural_loss, volume_loss,
        shape_preservation_loss, smoothness_loss,
    )
    from structural.optimization.alm_optimizer import ALMOptimizer, ALMConfig
    from structural.experiments.evaluate_metrics import compute_3d_iou

    res_config = RESOLUTION_CONFIGS.get(resolution, RESOLUTION_CONFIGS[64])
    if n_steps is None:
        n_steps = res_config['n_steps']

    print(f"\n  Resolution {resolution}³ ({res_config['description']})")
    print(f"    Steps: {n_steps}, PINO modes: {res_config['pino_modes']}")

    # 重采样到目标分辨率
    density_orig = rescale_voxels(density_original_64, resolution)
    force = rescale_force(force_64, resolution)
    fixed_mask = rescale_fixed_mask(fixed_mask_64, resolution)

    # 加载或构建该分辨率的 PINO
    pino_model = _load_or_build_pino(
        checkpoint=pino_checkpoint,
        resolution=resolution,
        res_config=res_config,
        device=device,
    )

    # 重置显存峰值
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ========== 运行 StructShadow (ALM) ==========
    t_start = time.time()

    density_t = torch.from_numpy(density_orig).float().to(device)
    density_t = torch.clamp(density_t, 1e-6, 1 - 1e-6)
    temperature = 15.0
    log_density = torch.nn.Parameter(
        torch.log(density_t / (1 - density_t)) / temperature
    )

    density_orig_t = torch.from_numpy(density_orig).float().unsqueeze(0).unsqueeze(0).to(device)
    force_t = torch.from_numpy(force).float().unsqueeze(0).to(device)

    optimizer = torch.optim.Adam([log_density], lr=1e-3)
    alm = ALMOptimizer(ALMConfig(
        safety_factor_target=1.5,
        volume_budget_ratio=1.3,
        dual_update_interval=5,
    ))

    final_sf = 0.0
    final_iou = 0.0

    for step in range(n_steps):
        density_current = torch.sigmoid(temperature * log_density).unsqueeze(0).unsqueeze(0)

        pino_fields = forward_surrogate_physical(pino_model, density_current, force_t)
        vm = pino_fields['von_mises']

        l_shape = shape_preservation_loss(density_current, density_orig_t)
        l_smooth = smoothness_loss(density_current)

        violations = alm.compute_constraint_violations(
            vm, density_current, density_orig_t, yield_stress
        )
        alm_result = alm.compute_alm_loss(
            appearance_loss=l_shape,
            violations=violations,
            smoothness_loss_val=l_smooth,
        )

        optimizer.zero_grad()
        alm_result['total'].backward()
        torch.nn.utils.clip_grad_norm_([log_density], 1.0)
        optimizer.step()

        alm.update_violations(violations)
        alm.step(step)

    elapsed = time.time() - t_start
    peak_mem = measure_gpu_memory()

    # ========== 评估 ==========
    with torch.no_grad():
        final_density = torch.sigmoid(temperature * log_density).cpu().numpy()
        density_eval = torch.sigmoid(temperature * log_density).unsqueeze(0).unsqueeze(0)

        pino_fields = forward_surrogate_physical(pino_model, density_eval.to(device), force_t)
        vm = pino_fields['von_mises']

        solid_mask = (density_eval > 0.1).float().to(device)
        vm_max = float((vm * solid_mask).max())
        final_sf = yield_stress / (vm_max + 1e-12)
        final_iou = compute_3d_iou(final_density, density_orig)

    vol_orig = (density_orig > 0.5).sum()
    vol_final = (final_density > 0.5).sum()
    vol_ratio = float(vol_final) / max(float(vol_orig), 1.0)

    constraint_satisfied = (final_sf >= 1.5) and (vol_ratio <= 1.3)

    # 体素数 & FLOPS 估算
    n_voxels = resolution ** 3
    # F-FNO 的主要 FLOPS: 3 * modes * width² * N³
    modes = res_config['pino_modes']
    width = res_config['pino_width']
    pino_flops_approx = 3 * modes * width ** 2 * n_voxels * 4  # 4 layers

    result = {
        'resolution': resolution,
        'n_voxels': n_voxels,
        'iou': float(final_iou),
        'sf_min': float(final_sf),
        'volume_ratio': float(vol_ratio),
        'constraint_satisfied': constraint_satisfied,
        'elapsed_seconds': elapsed,
        'time_per_step': elapsed / n_steps,
        'peak_memory_mb': peak_mem,
        'n_steps': n_steps,
        'pino_modes': modes,
        'pino_width': width,
        'pino_flops_approx': pino_flops_approx,
    }

    print(f"    IoU={final_iou:.3f}, SF={final_sf:.2f}, Vol={vol_ratio:.2f}")
    print(f"    Time={elapsed:.1f}s ({elapsed/n_steps:.3f}s/step)")
    print(f"    Peak GPU memory: {peak_mem:.0f} MB")

    # 清理显存
    del pino_model, log_density, density_current, pino_fields, vm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def _load_or_build_pino(
    checkpoint: str | None,
    resolution: int,
    res_config: dict,
    device: torch.device,
) -> torch.nn.Module:
    """加载或构建指定分辨率的 PINO

    如果有 checkpoint 则加载; 否则用随机初始化的 PINO
    (随机 PINO 仅用于验证管线可扩展性, 不代表真实性能)
    """
    from structural.models.pino import PINO3d

    pino = PINO3d(
        in_channels=4,
        out_channels=9,
        modes=res_config['pino_modes'],
        width=res_config['pino_width'],
        n_layers=4,
        padding=res_config['pino_padding'],
    ).to(device)

    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device)
        if 'model_state_dict' in state:
            pino.load_state_dict(state['model_state_dict'])
        else:
            pino.load_state_dict(state)
        # Detect PINO v3 (log-scale stress output)
        if isinstance(state, dict):
            pino._is_v3 = (state.get('normalization_mode') == 'log_stress'
                           or state.get('config', {}).get('normalization_mode') == 'log_stress')
        else:
            pino._is_v3 = False
        print(f"    Loaded PINO from {checkpoint} (v3={getattr(pino, '_is_v3', False)})")
    else:
        pino._is_v3 = False
        print(f"    Using randomly initialized PINO (no checkpoint for {resolution}³)")

    pino.eval()
    return pino


def run_scalability_experiment(
    density_original_64: np.ndarray,
    force_64: np.ndarray,
    fixed_mask_64: np.ndarray,
    pino_checkpoints: Dict[int, str] | None = None,
    device: torch.device | None = None,
    yield_stress: float = 50.0e6,
    resolutions: list[int] | None = None,
    n_models: int = 5,
    models_data: list | None = None,
) -> List[Dict]:
    """运行多分辨率可扩展性实验

    Args:
        density_original_64: [64, 64, 64] 单个模型的基准密度
        force_64: [3, 64, 64, 64]
        fixed_mask_64: [1, 64, 64, 64]
        pino_checkpoints: {resolution: checkpoint_path}
        device: GPU
        yield_stress: 屈服强度
        resolutions: 要测试的分辨率列表
        n_models: 测试模型数
        models_data: 多模型数据列表 [(density, force, fixed_mask), ...]

    Returns:
        list of per-resolution result dicts
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if resolutions is None:
        resolutions = [32, 64, 128]

    if pino_checkpoints is None:
        pino_checkpoints = {}

    print(f"Scalability experiment: resolutions = {resolutions}")

    # 构建多模型数据
    if models_data is None:
        models_data = [(density_original_64, force_64, fixed_mask_64)]

    all_results = []

    for res in resolutions:
        res_results = []
        ckpt = pino_checkpoints.get(res)

        for i, (density, force, fmask) in enumerate(models_data[:n_models]):
            print(f"\n  Model {i+1}/{min(n_models, len(models_data))}, Resolution {res}³")
            result = run_single_resolution(
                resolution=res,
                density_original_64=density,
                force_64=force,
                fixed_mask_64=fmask,
                pino_checkpoint=ckpt,
                device=device,
                yield_stress=yield_stress,
            )
            result['model_idx'] = i
            res_results.append(result)

        # 聚合
        if res_results:
            aggregated = {
                'resolution': res,
                'n_voxels': res ** 3,
                'n_models': len(res_results),
                'iou_mean': float(np.mean([r['iou'] for r in res_results])),
                'iou_std': float(np.std([r['iou'] for r in res_results])),
                'sf_min_mean': float(np.mean([r['sf_min'] for r in res_results])),
                'sf_min_std': float(np.std([r['sf_min'] for r in res_results])),
                'vol_ratio_mean': float(np.mean([r['volume_ratio'] for r in res_results])),
                'csr': float(np.mean([r['constraint_satisfied'] for r in res_results])),
                'time_mean': float(np.mean([r['elapsed_seconds'] for r in res_results])),
                'time_std': float(np.std([r['elapsed_seconds'] for r in res_results])),
                'time_per_step': float(np.mean([r['time_per_step'] for r in res_results])),
                'peak_memory_mb': float(max(r['peak_memory_mb'] for r in res_results)),
                'pino_modes': res_results[0]['pino_modes'],
                'pino_width': res_results[0]['pino_width'],
                'per_model': res_results,
            }
            all_results.append(aggregated)

    return all_results


def format_scalability_table_latex(results: List[Dict]) -> str:
    """格式化可扩展性实验为 LaTeX 表格 (Table 7)"""
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Resolution scalability analysis. All methods use the same ALM pipeline\n"
        "with resolution-adapted PINO. Time is per-model end-to-end optimization.}\n"
        "\\label{tab:scalability}\n"
        "\\begin{tabular}{c|cccccc}\n"
        "\\toprule\n"
        "Resolution & \\#Voxels & IoU$\\uparrow$ & SF$_{\\min}\\uparrow$ "
        "& CSR(\\%)$\\uparrow$ & Time(s)$\\downarrow$ & Mem(MB) \\\\\n"
        "\\midrule\n"
    )

    rows = []
    for r in results:
        rows.append(
            f"${r['resolution']}^3$ & "
            f"{r['n_voxels']:,} & "
            f"{r['iou_mean']:.3f}$\\pm${r['iou_std']:.3f} & "
            f"{r['sf_min_mean']:.2f}$\\pm${r['sf_min_std']:.2f} & "
            f"{r['csr']*100:.1f} & "
            f"{r['time_mean']:.1f}$\\pm${r['time_std']:.1f} & "
            f"{r['peak_memory_mb']:.0f} \\\\"
        )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )

    return header + "\n".join(rows) + "\n" + footer


def format_scalability_markdown(results: List[Dict]) -> str:
    """格式化为 Markdown 表格"""
    lines = [
        "| Resolution | #Voxels | IoU↑ | SF_min↑ | CSR(%)↑ | Time(s)↓ | Mem(MB) |",
        "|------------|---------|------|---------|---------|----------|---------|",
    ]
    for r in results:
        lines.append(
            f"| {r['resolution']}³ | "
            f"{r['n_voxels']:,} | "
            f"{r['iou_mean']:.3f}±{r['iou_std']:.3f} | "
            f"{r['sf_min_mean']:.2f}±{r['sf_min_std']:.2f} | "
            f"{r['csr']*100:.1f} | "
            f"{r['time_mean']:.1f}±{r['time_std']:.1f} | "
            f"{r['peak_memory_mb']:.0f} |"
        )
    return "\n".join(lines)


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    parser = argparse.ArgumentParser(
        description='StructShadow resolution scalability experiment'
    )
    parser.add_argument('--pino_32', type=str, default=None,
                        help='PINO checkpoint for 32³')
    parser.add_argument('--pino_64', type=str, default=None,
                        help='PINO checkpoint for 64³')
    parser.add_argument('--pino_128', type=str, default=None,
                        help='PINO checkpoint for 128³')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Test dataset directory (64³ base)')
    parser.add_argument('--output_dir', type=str, default='./scalability_results')
    parser.add_argument('--n_models', type=int, default=5)
    parser.add_argument('--resolutions', type=str, default='32,64,128')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载测试数据
    data_path = Path(args.data_dir)
    model_dirs = sorted(data_path.glob("model_*"))[:args.n_models]

    models_data = []
    for md in model_dirs:
        density = np.load(md / "voxels.npy").astype(np.float32)
        loads = np.load(md / "loads.npz")
        force = loads['force_standing']
        fmask = loads['fixed_mask_standing']
        models_data.append((density, force, fmask))

    if not models_data:
        print("No models found!")
        exit(1)

    pino_ckpts = {}
    if args.pino_32:
        pino_ckpts[32] = args.pino_32
    if args.pino_64:
        pino_ckpts[64] = args.pino_64
    if args.pino_128:
        pino_ckpts[128] = args.pino_128

    resolutions = [int(r) for r in args.resolutions.split(',')]

    results = run_scalability_experiment(
        density_original_64=models_data[0][0],
        force_64=models_data[0][1],
        fixed_mask_64=models_data[0][2],
        pino_checkpoints=pino_ckpts,
        device=device,
        resolutions=resolutions,
        n_models=args.n_models,
        models_data=models_data,
    )

    # 保存结果
    with open(output_dir / "scalability_results.json", 'w') as f:
        serializable = []
        for r in results:
            s = {k: v for k, v in r.items() if k != 'per_model'}
            serializable.append(s)
        json.dump(serializable, f, indent=2)

    # LaTeX 表
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    latex = format_scalability_table_latex(results)
    with open(tables_dir / "table7_scalability.tex", 'w') as f:
        f.write(latex)

    md = format_scalability_markdown(results)
    with open(tables_dir / "table7_scalability.md", 'w') as f:
        f.write(md)

    print(f"\n{'='*60}")
    print("Scalability Results:")
    print(md)
    print(f"{'='*60}")
