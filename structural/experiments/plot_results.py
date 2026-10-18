"""
论文图表生成模块

从实验 JSON 数据生成 matplotlib 图表, 用于论文插图。

Figure 1: IoU vs SF 帕累托散点图 (各方法对比)
Figure 2: 收敛曲线 (ALM vs Weighted Sum)
Figure 3: ALM 乘子 & 罚因子演化
Figure 4: 路由模式时间线/饼图
Figure 5: SF_PINO vs SF_FEM 散点 + bound 区域
Figure 6: ε_u 敏感性曲线
Figure 7: PINO 误差分布直方图

用法:
    python plot_results.py --exp_dir ./paper_experiments
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict

import numpy as np

# matplotlib 延迟导入 (服务器可能无 display)
def _setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    return plt


# 方法颜色映射
METHOD_COLORS = {
    'ShadowArt': '#1f77b4',
    'ShadowArt+Stava': '#ff7f0e',
    'ShadowArt+WeightedSum': '#2ca02c',
    'ShadowArt+FEM-Opt': '#d62728',
    'SIMP': '#9467bd',
    'NTOPo': '#8c564b',
    'TopoDiff': '#17becf',
    'Ours': '#e41a1c',
    'Ours w/o ALM': '#377eb8',
    'Ours w/o Router': '#4daf4a',
    'Ours w/o Bound': '#984ea3',
    'Ours w/o RepairCNN': '#ff7f00',
    'Ours w/o Decoupling': '#a65628',
}

METHOD_MARKERS = {
    'ShadowArt': 'o',
    'ShadowArt+Stava': 's',
    'ShadowArt+WeightedSum': '^',
    'ShadowArt+FEM-Opt': 'D',
    'SIMP': 'v',
    'NTOPo': 'P',
    'TopoDiff': 'X',
    'Ours': '*',
}


def plot_pareto(pareto_file: str, output_file: str):
    """Figure 1: IoU vs SF 帕累托散点图"""
    plt = _setup_matplotlib()

    with open(pareto_file) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 6))

    for method, values in data.items():
        color = METHOD_COLORS.get(method, '#333333')
        marker = METHOD_MARKERS.get(method, 'o')
        size = 120 if method == 'Ours' else 60
        alpha = 1.0 if method == 'Ours' else 0.7

        ax.scatter(
            values['ious'], values['sfs'],
            c=color, marker=marker, s=size, alpha=alpha,
            label=method, edgecolors='black', linewidths=0.5,
        )

    # 安全线
    ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='SF target (1.5)')

    ax.set_xlabel('3D IoU ↑')
    ax.set_ylabel('Safety Factor (SF$_{min}$) ↑')
    ax.set_title('Shape Fidelity vs. Structural Safety')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_convergence(convergence_file: str, output_file: str):
    """Figure 2: ALM vs Weighted Sum 收敛曲线"""
    plt = _setup_matplotlib()

    with open(convergence_file) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: IoU 收敛
    ax = axes[0]
    for key, label, color in [('alm', 'ALM (Ours)', '#e41a1c'),
                                ('weighted_sum', 'Weighted Sum', '#377eb8')]:
        if key in data:
            ax.plot(data[key]['steps'], data[key]['iou'],
                    color=color, label=label, linewidth=2)
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('3D IoU ↑')
    ax.set_title('Shape Fidelity Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: SF 收敛
    ax = axes[1]
    for key, label, color in [('alm', 'ALM (Ours)', '#e41a1c'),
                                ('weighted_sum', 'Weighted Sum', '#377eb8')]:
        if key in data:
            ax.plot(data[key]['steps'], data[key]['sf_min'],
                    color=color, label=label, linewidth=2)
    ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='SF target')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Safety Factor (SF$_{min}$) ↑')
    ax.set_title('Structural Safety Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_alm_evolution(convergence_file: str, output_file: str):
    """Figure 3: ALM 乘子 & 罚因子演化"""
    plt = _setup_matplotlib()

    with open(convergence_file) as f:
        data = json.load(f)

    alm = data.get('alm', {})
    if not alm:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Left: 乘子 μ
    ax = axes[0]
    steps = alm['steps']
    for name, color in [('mu_safety', '#e41a1c'), ('mu_volume', '#377eb8'),
                         ('mu_overhang', '#4daf4a')]:
        if name in alm and alm[name]:
            ax.plot(steps, alm[name], color=color,
                    label=name.replace('mu_', 'μ_'), linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Multiplier μ')
    ax.set_title('Lagrangian Multipliers')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Center: 罚因子 ρ
    ax = axes[1]
    if 'rho' in alm and alm['rho']:
        ax.plot(steps, alm['rho'], color='#984ea3', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Penalty ρ')
    ax.set_title('Penalty Factor Growth')
    ax.grid(True, alpha=0.3)

    # Right: 约束违反量
    ax = axes[2]
    for name, color in [('g_safety', '#e41a1c'), ('g_volume', '#377eb8'),
                         ('g_overhang', '#4daf4a')]:
        if name in alm and alm[name]:
            ax.plot(steps, alm[name], color=color,
                    label=name.replace('g_', ''), linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Constraint Violation g(θ)')
    ax.set_title('Constraint Satisfaction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_routing_distribution(convergence_file: str, output_file: str):
    """Figure 4: 路由模式分布"""
    plt = _setup_matplotlib()

    with open(convergence_file) as f:
        data = json.load(f)

    alm = data.get('alm', {})
    modes = alm.get('routing_mode', [])
    if not modes:
        return

    # 计数
    counts = {}
    for m in modes:
        counts[m] = counts.get(m, 0) + 1

    mode_colors = {
        'joint': '#4daf4a',
        'structure_priority': '#e41a1c',
        'shape_priority': '#377eb8',
        'feasibility_restore': '#ff7f00',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: 饼图
    ax = axes[0]
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = [mode_colors.get(l, '#999999') for l in labels]
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Routing Mode Distribution')

    # Right: 时间线
    ax = axes[1]
    steps = alm.get('steps', list(range(len(modes))))
    for i, (step, mode) in enumerate(zip(steps, modes)):
        color = mode_colors.get(mode, '#999999')
        ax.barh(0, 1, left=step, color=color, height=0.5)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=mode_colors.get(m, '#999'), label=m)
        for m in set(modes)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax.set_xlabel('Optimization Step')
    ax.set_yticks([])
    ax.set_title('Routing Mode Timeline')

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_sf_scatter(scatter_file: str, output_file: str):
    """Figure 5: SF_PINO vs SF_FEM 散点 + bound"""
    plt = _setup_matplotlib()

    with open(scatter_file) as f:
        data = json.load(f)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    # 按方法分组
    methods = {}
    for point in data:
        m = point['method']
        if m not in methods:
            methods[m] = {'pino': [], 'fem': []}
        methods[m]['pino'].append(point['sf_pino'])
        methods[m]['fem'].append(point['sf_fem'])

    for method, vals in methods.items():
        color = METHOD_COLORS.get(method, '#333333')
        marker = METHOD_MARKERS.get(method, 'o')
        ax.scatter(vals['pino'], vals['fem'],
                   c=color, marker=marker, alpha=0.7, label=method, s=40)

    # 对角线 (SF_PINO = SF_FEM)
    lims = ax.get_xlim()
    ax.plot(lims, lims, 'k--', alpha=0.3, label='SF_PINO = SF_FEM')

    ax.set_xlabel('SF$_{PINO}$')
    ax.set_ylabel('SF$_{FEM}$')
    ax.set_title('PINO vs FEM Safety Factor')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_epsilon_sensitivity(epsilon_file: str, output_file: str):
    """Figure 6: ε_u 敏感性曲线"""
    plt = _setup_matplotlib()

    with open(epsilon_file) as f:
        data = json.load(f)

    if not data:
        return

    eps = [d['eps_u'] for d in data]
    ratios = [d['ratio'] for d in data]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(eps, ratios, 'o-', color='#e41a1c', linewidth=2, markersize=8)
    ax.set_xlabel('PINO Displacement Error ε$_u$')
    ax.set_ylabel('SF$_{lower}$ / SF$_{PINO}$ (Bound Ratio)')
    ax.set_title('Safety Transfer Bound Tightness vs. PINO Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_scalability(scalability_file: str, output_file: str):
    """Figure 8: 分辨率可扩展性 — 4 子图"""
    plt = _setup_matplotlib()

    with open(scalability_file) as f:
        data = json.load(f)

    if not data:
        return

    resolutions = [d['resolution'] for d in data]
    res_labels = [f"${r}^3$" for r in resolutions]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) IoU vs Resolution
    ax = axes[0, 0]
    ious = [d['iou_mean'] for d in data]
    iou_stds = [d.get('iou_std', 0) for d in data]
    ax.errorbar(resolutions, ious, yerr=iou_stds, fmt='o-',
                color='#1f77b4', linewidth=2, markersize=8, capsize=5)
    ax.set_xlabel('Resolution')
    ax.set_ylabel('3D IoU ↑')
    ax.set_title('(a) Shape Fidelity')
    ax.set_xticks(resolutions)
    ax.set_xticklabels(res_labels)
    ax.grid(True, alpha=0.3)

    # (b) SF vs Resolution
    ax = axes[0, 1]
    sfs = [d['sf_min_mean'] for d in data]
    sf_stds = [d.get('sf_min_std', 0) for d in data]
    ax.errorbar(resolutions, sfs, yerr=sf_stds, fmt='s-',
                color='#e41a1c', linewidth=2, markersize=8, capsize=5)
    ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='SF target')
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Safety Factor (SF$_{min}$) ↑')
    ax.set_title('(b) Structural Safety')
    ax.set_xticks(resolutions)
    ax.set_xticklabels(res_labels)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Time vs Resolution (log scale)
    ax = axes[1, 0]
    times = [d['time_mean'] for d in data]
    time_stds = [d.get('time_std', 0) for d in data]
    ax.errorbar(resolutions, times, yerr=time_stds, fmt='^-',
                color='#2ca02c', linewidth=2, markersize=8, capsize=5)
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Time (s) ↓')
    ax.set_title('(c) Computation Time')
    ax.set_xticks(resolutions)
    ax.set_xticklabels(res_labels)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # (d) Memory vs Resolution
    ax = axes[1, 1]
    mems = [d['peak_memory_mb'] for d in data]
    ax.bar(range(len(resolutions)), mems, color=['#4daf4a', '#377eb8', '#e41a1c'],
           tick_label=res_labels, alpha=0.8)
    ax.set_xlabel('Resolution')
    ax.set_ylabel('Peak GPU Memory (MB)')
    ax.set_title('(d) Memory Usage')
    ax.grid(True, alpha=0.3, axis='y')

    # RTX 5090 显存线
    ax.axhline(y=32768, color='red', linestyle='--', alpha=0.5,
               label='RTX 5090 D (32 GB)')
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_stress_heatmap(density: np.ndarray, stress: np.ndarray,
                        output_file: str, slice_axis: int = 2,
                        slice_idx: int | None = None):
    """应力热力图 — 在密度切片上叠加 von Mises 应力"""
    plt = _setup_matplotlib()
    from matplotlib.colors import Normalize

    if slice_idx is None:
        slice_idx = density.shape[slice_axis] // 2

    slices = [slice(None)] * 3
    slices[slice_axis] = slice_idx
    rho_slice = density[tuple(slices)]
    stress_slice = stress[tuple(slices)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Density
    ax = axes[0]
    im = ax.imshow(rho_slice.T, origin='lower', cmap='gray_r', vmin=0, vmax=1)
    ax.set_title('(a) Density ρ')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (b) von Mises stress
    ax = axes[1]
    masked_stress = np.where(rho_slice > 0.3, stress_slice, np.nan)
    im = ax.imshow(masked_stress.T, origin='lower', cmap='hot',
                   norm=Normalize(vmin=0, vmax=np.nanpercentile(stress_slice, 99)))
    ax.set_title('(b) von Mises Stress σ_vm')
    plt.colorbar(im, ax=ax, fraction=0.046, label='MPa')

    # (c) Overlay
    ax = axes[2]
    ax.imshow(rho_slice.T, origin='lower', cmap='gray_r', vmin=0, vmax=1)
    im = ax.imshow(masked_stress.T, origin='lower', cmap='hot', alpha=0.6,
                   norm=Normalize(vmin=0, vmax=np.nanpercentile(stress_slice, 99)))
    ax.set_title('(c) Overlay')
    plt.colorbar(im, ax=ax, fraction=0.046, label='MPa')

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.suptitle(f'Stress Heatmap (slice axis={slice_axis}, idx={slice_idx})', y=1.02)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_voxel_3d(density: np.ndarray, output_file: str,
                  threshold: float = 0.5, color_by: str | np.ndarray | None = None,
                  elev: float = 25, azim: float = -60):
    """体素 3D 渲染 — 使用 matplotlib voxels (适合论文级别小分辨率)"""
    plt = _setup_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    binary = density > threshold
    N = binary.shape[0]

    # 下采样到 ≤ 48³ 避免渲染过慢
    max_res = 48
    if N > max_res:
        step = max(1, N // max_res)
        binary = binary[::step, ::step, ::step]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(color_by, np.ndarray):
        # 用应力着色
        if color_by.shape != density.shape:
            color_by = None
        else:
            if N > max_res:
                color_by = color_by[::step, ::step, ::step]
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=0, vmax=np.percentile(color_by[binary], 95) if binary.any() else 1)
            cmap = plt.cm.hot
            facecolors = np.empty(binary.shape + (4,))
            for i in range(binary.shape[0]):
                for j in range(binary.shape[1]):
                    for k in range(binary.shape[2]):
                        if binary[i, j, k]:
                            facecolors[i, j, k] = cmap(norm(color_by[i, j, k]))

    if color_by is None or not isinstance(color_by, np.ndarray):
        ax.voxels(binary, facecolors='#4daf4a', edgecolor='#333333', linewidth=0.1, alpha=0.8)
    else:
        ax.voxels(binary, facecolors=facecolors, edgecolor='none', alpha=0.9)

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voxel Visualization')

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_silhouette_comparison(target_images: list[np.ndarray],
                               pred_images: list[np.ndarray],
                               view_names: list[str],
                               output_file: str,
                               method_names: list[str] | None = None):
    """多视角轮廓对比图 — 目标 vs 各方法预测"""
    plt = _setup_matplotlib()

    n_views = len(target_images)
    n_methods = len(pred_images) // n_views if method_names else 1
    if method_names is None:
        method_names = ['Predicted']
    n_rows = 1 + n_methods  # target + each method

    fig, axes = plt.subplots(n_rows, n_views, figsize=(3.5 * n_views, 3 * n_rows))
    if n_views == 1:
        axes = axes[:, np.newaxis]

    # Row 0: targets
    for j in range(n_views):
        ax = axes[0, j]
        ax.imshow(target_images[j], cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(f'Target ({view_names[j]})', fontsize=10)
        ax.axis('off')

    # Rows 1+: predictions
    for i, method in enumerate(method_names):
        for j in range(n_views):
            ax = axes[i + 1, j]
            idx = i * n_views + j
            pred = pred_images[idx] if idx < len(pred_images) else np.zeros_like(target_images[0])
            # overlay: green = TP, red = FN, blue = FP
            target_bin = target_images[j] > 0.5
            pred_bin = pred > 0.5
            overlay = np.zeros(target_bin.shape + (3,))
            overlay[target_bin & pred_bin] = [0.3, 0.8, 0.3]   # TP: green
            overlay[target_bin & ~pred_bin] = [0.9, 0.2, 0.2]  # FN: red
            overlay[~target_bin & pred_bin] = [0.2, 0.2, 0.9]  # FP: blue
            ax.imshow(overlay)
            ax.set_title(f'{method} ({view_names[j]})', fontsize=10)
            ax.axis('off')

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_training_loss(log_file: str, output_file: str):
    """训练损失曲线 — 从训练日志文件解析"""
    plt = _setup_matplotlib()

    # 解析日志: 支持 JSON lines 或 CSV 格式
    log_path = Path(log_file)
    epochs, train_losses, val_losses, lrs = [], [], [], []

    if log_path.suffix == '.json':
        with open(log_path) as f:
            data = json.load(f)
        epochs = data.get('epochs', [])
        train_losses = data.get('train_loss', [])
        val_losses = data.get('val_loss', [])
        lrs = data.get('lr', [])
    else:
        # 解析文本日志格式:
        # Train: X.XXXXe-XX | Val: X.XXXXe-XX (...) | LR: X.XXe-XX
        import re
        pattern = re.compile(
            r'Train:\s+([\d.e+-]+)\s*\|\s*Val:\s+([\d.e+-]+).*?LR:\s+([\d.e+-]+)')
        with open(log_path) as f:
            for i, line in enumerate(f):
                m = pattern.search(line)
                if m:
                    epochs.append(i + 1)
                    train_losses.append(float(m.group(1)))
                    val_losses.append(float(m.group(2)))
                    lrs.append(float(m.group(3)))

    if not epochs:
        print(f"  [WARN] No training data found in {log_file}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Loss curves
    ax = axes[0]
    ax.plot(epochs, train_losses, color='#1f77b4', label='Train Loss', linewidth=1.5)
    ax.plot(epochs, val_losses, color='#e41a1c', label='Val Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('PINO Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Right: Learning rate
    ax = axes[1]
    ax.plot(epochs, lrs, color='#2ca02c', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_method_comparison_bar(results_json: str, output_file: str):
    """方法对比柱状图 — 多指标分组柱状图"""
    plt = _setup_matplotlib()

    with open(results_json) as f:
        data = json.load(f)

    methods = [d['method'] for d in data]
    n_methods = len(methods)

    metrics = {
        'IoU ↑': ('iou_mean', False),
        'CD ↓': ('chamfer_mean', True),
        'SF_min ↑': ('sf_min_mean', False),
        'Sil. IoU ↑': ('silhouette_iou_mean', False),
    }
    # 只绘制 data 中实际存在的 metrics
    active_metrics = {k: v for k, v in metrics.items()
                      if v[0] in data[0]}

    if not active_metrics:
        print(f"  [WARN] No recognized metrics in {results_json}")
        return

    n_metrics = len(active_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_label, (key, lower_better)) in zip(axes, active_metrics.items()):
        values = [d.get(key, 0) for d in data]
        colors = []
        for i, d in enumerate(data):
            m = d['method']
            colors.append(METHOD_COLORS.get(m, '#999999'))

        bars = ax.bar(range(n_methods), values, color=colors, alpha=0.85,
                      edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.grid(True, alpha=0.3, axis='y')

        # 标注最优值
        if lower_better:
            best_idx = int(np.argmin(values))
        else:
            best_idx = int(np.argmax(values))
        bars[best_idx].set_edgecolor('#e41a1c')
        bars[best_idx].set_linewidth(2.5)

    fig.suptitle('Method Comparison', y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def plot_lyapunov_curve(convergence_file: str, output_file: str):
    """Lyapunov 函数 V(θ) 单调递减曲线"""
    plt = _setup_matplotlib()

    with open(convergence_file) as f:
        data = json.load(f)

    alm = data.get('alm', {})
    steps = alm.get('steps', [])
    lyapunov = alm.get('lyapunov_V', [])

    if not lyapunov:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, lyapunov, color='#e41a1c', linewidth=2, label='V(θ)')
    ax.fill_between(steps, lyapunov, alpha=0.15, color='#e41a1c')
    ax.set_xlabel('Optimization Step')
    ax.set_ylabel('Lyapunov Function V(θ)')
    ax.set_title('Lyapunov Stability: Monotonic Descent')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"  Saved: {output_file}")


def generate_all_figures(exp_dir: str):
    """从实验目录一键生成所有图表"""
    exp_path = Path(exp_dir)
    figures_dir = exp_path / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("Generating figures...")

    # Figure 1: Pareto
    pareto_file = figures_dir / "fig1_pareto.json"
    if pareto_file.exists():
        plot_pareto(str(pareto_file), str(figures_dir / "fig1_pareto.pdf"))

    # Figure 2-4: Convergence + ALM + Routing + Lyapunov
    conv_file = figures_dir / "convergence_data.json"
    if conv_file.exists():
        plot_convergence(str(conv_file), str(figures_dir / "fig2_convergence.pdf"))
        plot_alm_evolution(str(conv_file), str(figures_dir / "fig3_alm_evolution.pdf"))
        plot_routing_distribution(str(conv_file), str(figures_dir / "fig4_routing.pdf"))
        plot_lyapunov_curve(str(conv_file), str(figures_dir / "fig_lyapunov.pdf"))

    # Figure 5: SF scatter
    scatter_file = figures_dir / "fig2_sf_scatter.json"
    if scatter_file.exists():
        plot_sf_scatter(str(scatter_file), str(figures_dir / "fig5_sf_scatter.pdf"))

    # Figure 6: Epsilon sensitivity
    eps_file = figures_dir / "fig3_epsilon.json"
    if eps_file.exists():
        plot_epsilon_sensitivity(str(eps_file), str(figures_dir / "fig6_epsilon.pdf"))

    # Figure 7: PINO training loss
    train_log = exp_path / "pino_training.log"
    if not train_log.exists():
        train_log = exp_path / "pino_training.json"
    if train_log.exists():
        plot_training_loss(str(train_log), str(figures_dir / "fig7_training_loss.pdf"))

    # Figure 8: Scalability
    scale_file = figures_dir / "fig8_scalability.json"
    if not scale_file.exists():
        alt = exp_path / "scalability_results" / "scalability_results.json"
        if alt.exists():
            scale_file = alt
    if scale_file.exists():
        plot_scalability(str(scale_file), str(figures_dir / "fig8_scalability.pdf"))

    # Figure 9: Method comparison bar
    bar_file = figures_dir / "method_comparison.json"
    if bar_file.exists():
        plot_method_comparison_bar(str(bar_file), str(figures_dir / "fig9_comparison_bar.pdf"))

    print("All figures generated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Experiment output directory')
    args = parser.parse_args()
    generate_all_figures(args.exp_dir)
